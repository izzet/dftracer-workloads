"""Non-invasive DeepSpeed-Ulysses sequence parallelism for StorMer.

Switchable via --ulysses-sp N (or STORMER_ULYSSES_SP=N). Default N=1 -> no-op, behavior
identical to stock StorMer. When N>1 this monkeypatches, at runtime:
  * each Block.attn (MemEffAttention).forward  -> DeepSpeed DistributedAttention (two-phase
    all_to_all: scatter heads / gather sequence), and
  * Stormer.forward                            -> slice the token sequence across the sp_group
    after embedding, run the blocks on the local chunk (distributed attention gathers the full
    sequence internally), then a differentiable all_gather reassembles the full sequence before
    head + unpatchify.

models/stormer/stormer.py is NEVER edited. The attention math + reassembly ordering were
validated bit-exact vs a single-GPU reference in
liveflow_validation/stormer_ulysses_smoke.py (ROUNDTRIP=0.00, CORRECTNESS PASS).

Instrumentation: the forward all_to_all is wrapped in dft_ai.comm.all_to_all and the final
sequence gather in dft_ai.comm.all_gather, so DFAnalyzer sees the comm domain alongside the
POSIX read domain (the cross-layer signal the LiveFlow controller attributes).

Gradient note: DDP is kept over the FULL world. With the differentiable all_gather over the
sp_group, DDP's all_reduce averages the per-chunk gradients over all ranks, which equals the
correct minibatch gradient up to a constant 1/sp factor (absorbed by the learning rate) and
keeps every rank's parameters identical. This is intentional and correct for the timing study.
"""
import os
import types

import torch
import torch.nn.functional as F
import torch.distributed as dist

try:
    from utils.perf_tracer import dft_ai
except Exception:  # pragma: no cover - tracer optional
    class _NoOpRegion:
        def __getattr__(self, _):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    dft_ai = _NoOpRegion()


def sp_enabled():
    """SP degree from CLI/env; 1 (or unset) means disabled."""
    return int(os.environ.get("STORMER_ULYSSES_SP", "1"))


# --- optional low-overhead a2a timing (CUDA events; opt-in STORMER_SP_TIME_A2A=1) -----------
# Separates the Ulysses all_to_all from the rest of forward for the input<->comm MOVE study.
# Uses CUDA events (no per-block full sync); one sync per step happens in pop_a2a_ms().
_TIME_A2A = os.environ.get("STORMER_SP_TIME_A2A", "0") == "1"
_A2A_EVENTS = []


def pop_a2a_ms():
    """Sum the a2a CUDA-event intervals recorded this step and reset. Returns ms (0 if off)."""
    if not _A2A_EVENTS:
        return 0.0
    torch.cuda.synchronize()
    total = 0.0
    for s, e in _A2A_EVENTS:
        total += s.elapsed_time(e)
    _A2A_EVENTS.clear()
    return total


def init_sp_dp_groups(sp_degree):
    """Build sequence-parallel + data-parallel process groups.

    Rank layout: contiguous blocks of ``sp_degree`` ranks form one sp_group (same sample,
    different sequence chunks); ranks that share a position across blocks form a dp_group.
    All groups are created on all ranks (new_group is collective); we keep the ones we join.
    Returns (sp_group, dp_group, sp_size, sp_rank, dp_size, dp_rank).
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    assert world % sp_degree == 0, f"world {world} not divisible by sp_degree {sp_degree}"
    n_dp = world // sp_degree

    sp_group = None
    for g in range(n_dp):
        ranks = list(range(g * sp_degree, (g + 1) * sp_degree))
        grp = dist.new_group(ranks)
        if rank in ranks:
            sp_group = grp

    dp_group = None
    for j in range(sp_degree):
        ranks = list(range(j, world, sp_degree))
        grp = dist.new_group(ranks)
        if rank in ranks:
            dp_group = grp

    sp_size = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)
    dp_size = dist.get_world_size(dp_group)
    dp_rank = dist.get_rank(dp_group)
    return sp_group, dp_group, sp_size, sp_rank, dp_size, dp_rank


# --- pure-torch Ulysses all_to_all (no deepspeed dependency) ------------------------------
# Replicates DeepSpeed's single_all_to_all for the EVEN-division, batch_dim_idx=0 case, which
# is the layout validated bit-exact against DeepSpeed's DistributedAttention. Keeping this in
# pure torch means train.py's venv does not need deepspeed installed and the SP is portable.
def _timed_a2a_single(out, t, group):
    """dist.all_to_all_single, optionally bracketed by CUDA events (pure comm time)."""
    if _TIME_A2A:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        dist.all_to_all_single(out, t, group=group)
        e.record()
        _A2A_EVENTS.append((s, e))
    else:
        dist.all_to_all_single(out, t, group=group)


def _seq_all2all_raw(x, scatter_idx, gather_idx, group):
    """One all_to_all over `group`, batch-first [B, seq, heads, hd], even division.

    scatter_idx>=2 (scatter heads, gather sequence) = FORWARD:
        [B, local_seq, H, hd] -> [B, P*local_seq, H//P, hd]
    scatter_idx<2  (scatter sequence, gather heads) = REVERSE:
        [B, global_seq, H//P, hd] -> [B, global_seq//P, P*(H//P), hd]
    The CUDA-event timer (STORMER_SP_TIME_A2A=1) brackets ONLY the collective, so pop_a2a_ms()
    reports pure communication (not the SDPA attention compute).
    """
    P = dist.get_world_size(group)
    if scatter_idx >= 2:  # forward: scatter heads (dim 2), gather seq (dim 1)
        bs, local_seq, H, hd = x.shape
        t = x.reshape(bs, local_seq, P, H // P, hd).permute(2, 0, 1, 3, 4).contiguous()
        out = torch.empty_like(t)
        _timed_a2a_single(out, t, group)
        return out.permute(1, 0, 2, 3, 4).contiguous().reshape(bs, P * local_seq, H // P, hd)
    else:                 # reverse: scatter seq (dim 1), gather heads (dim 2)
        bs, gseq, Hl, hd = x.shape
        t = x.reshape(bs, P, gseq // P, Hl, hd).permute(1, 0, 2, 3, 4).contiguous()
        out = torch.empty_like(t)
        _timed_a2a_single(out, t, group)
        return out.permute(1, 2, 0, 3, 4).contiguous().reshape(bs, gseq // P, P * Hl, hd)


class _SeqAllToAll(torch.autograd.Function):
    """Autograd wrapper: backward is the inverse all_to_all (swap scatter/gather)."""

    @staticmethod
    def forward(ctx, group, x, scatter_idx, gather_idx):
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        return _seq_all2all_raw(x, scatter_idx, gather_idx, group)

    @staticmethod
    def backward(ctx, grad):
        return (None, _seq_all2all_raw(grad, ctx.gather_idx, ctx.scatter_idx, ctx.group), None, None)


def ulysses_attention(sp_group, q, k, v, dropout_p=0.0):
    """Two-phase Ulysses attention (batch-first [B, local_seq, heads, hd] in and out).

    Identical in effect to DeepSpeed DistributedAttention(scatter_idx=2, gather_idx=1).
    """
    q = _SeqAllToAll.apply(sp_group, q, 2, 1)   # [B, full_seq, heads/P, hd]
    k = _SeqAllToAll.apply(sp_group, k, 2, 1)
    v = _SeqAllToAll.apply(sp_group, v, 2, 1)
    qh, kh, vh = (t.permute(0, 2, 1, 3) for t in (q, k, v))   # -> [B, heads/P, full_seq, hd]
    ctx = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=dropout_p, is_causal=False)
    ctx = ctx.permute(0, 2, 1, 3).contiguous()               # -> [B, full_seq, heads/P, hd]
    return _SeqAllToAll.apply(sp_group, ctx, 1, 2)           # [B, local_seq, heads, hd]


def _make_attn_forward(attn, sp_group):
    """Return a bound MemEffAttention.forward that routes attention through Ulysses all_to_all."""
    heads, hd, dim, dp = attn.num_heads, attn.head_dim, attn.dim, attn.attn_drop.p

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, heads, hd)
        q, k, v = torch.unbind(qkv, dim=2)                       # [B, N, heads, hd]
        with dft_ai.comm.all_to_all:                             # a2a timing is inside the collective
            out = ulysses_attention(sp_group, q, k, v, dropout_p=dp)
        out = out.reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    return types.MethodType(forward, attn)


def apply_ulysses_sp(model, sp_group):
    """Monkeypatch attention (distributed) + model.forward (sequence slice + gather).

    stormer.py is untouched. Returns the (in-place patched) model.
    """
    sp_size = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)

    for blk in model.blocks:
        blk.attn.forward = _make_attn_forward(blk.attn, sp_group)

    def ulysses_forward(self, x, variables, time_interval):
        x = self.embedding(x, variables)                        # [B, L, D]
        x = self.embed_norm_layer(x)
        B, L, D = x.shape
        assert L % sp_size == 0, f"seq len {L} not divisible by sp {sp_size}"
        nl = L // sp_size
        x = x[:, sp_rank * nl:(sp_rank + 1) * nl, :].contiguous()   # local chunk [B, L/sp, D]

        t = self.t_embedder(time_interval)
        for blk in self.blocks:
            x = blk(x, t)                                       # distributed attention inside
        x = self.head(x, t)                                     # [B, L/sp, p*p*V] local

        # differentiable all_gather over sp_group -> full sequence, in rank order
        with dft_ai.comm.all_gather:
            parts = torch.distributed.nn.functional.all_gather(x.contiguous(), group=sp_group)
        x = torch.cat(list(parts), dim=1)                       # [B, L, p*p*V]
        x = self.unpatchify(x)
        return x

    model.forward = types.MethodType(ulysses_forward, model)
    return model


def maybe_init_deepspeed():
    """No-op retained for call-site compatibility.

    The SP all_to_all is now pure torch (no deepspeed), so nothing needs initializing beyond
    torch.distributed, which the caller sets up (train.py init_process_group; smoke likewise).
    Kept as a named hook so train.py/smoke call sites are unchanged.
    """
    assert dist.is_available() and dist.is_initialized(), (
        "torch.distributed must be initialized before applying Ulysses SP"
    )
