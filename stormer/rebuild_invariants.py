"""Byte-level invariants for a live DDP rebuild (bucket_cap_mb actuation).

A rebuild only re-wraps the SAME module, so parameters, buffers, and optimizer state must be
byte-identical across it, and it must happen at an optimizer-step boundary with accumulation off
(otherwise a partial accumulation window could be lost or double-applied).
"""

from __future__ import annotations

import hashlib

import torch


def _feed(h, tensor: torch.Tensor):
    t = tensor.detach().contiguous().cpu().reshape(-1)
    h.update(str(t.dtype).encode())
    h.update(str(tuple(tensor.shape)).encode())
    h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")


def state_digest(module: torch.nn.Module, optimizer=None) -> str:
    """SHA-256 over params, buffers (name order) and optimizer state (param-index order)."""
    h = hashlib.sha256()
    for name, p in module.named_parameters():
        h.update(name.encode()); _feed(h, p)
    for name, b in module.named_buffers():
        h.update(name.encode()); _feed(h, b)
    if optimizer is not None:
        sd = optimizer.state_dict()
        for pid in sorted(sd["state"]):
            for key in sorted(sd["state"][pid]):
                v = sd["state"][pid][key]
                h.update(("%s/%s" % (pid, key)).encode())
                if torch.is_tensor(v):
                    _feed(h, v)
                else:
                    h.update(repr(v).encode())
        h.update(repr(sd["param_groups"]).encode())
    return h.hexdigest()


def boundary_problem(accumulate_grad_batches: int):
    """None if a rebuild is safe, else the reason.

    The apply window runs after a completed training step, so with accumulation off the optimizer
    step has been applied and the next step zero_grads first: no gradient is lost or double-applied.
    With accumulation on, a window boundary may fall inside an accumulation window: refuse.
    """
    if int(accumulate_grad_batches) != 1:
        return ("accumulate_grad_batches=%d != 1: rebuild could split an accumulation window"
                % accumulate_grad_batches)
    return None


# ---------------------------------------------------------------------------------------------
# GPU fingerprint: a bit-exact, position-sensitive fingerprint computed where the tensors live.
# `state_digest` above copies every parameter and optimizer tensor to the host and SHA-256s it
# (seconds for the StorMer model, twice per rebuild).  The fingerprint below reinterprets each
# tensor's bytes as int64 and keeps two wrapping int64 reductions (plain sum and index-weighted
# sum), so it detects any bit change or permutation with overwhelming probability, in one small
# device-to-host copy.  It is an integrity check against accidental change, not a cryptographic one.
# ---------------------------------------------------------------------------------------------
_WEIGHT = 0x1E3779B97F4A7C15     # < 2**63, odd


def _fp_tensor(t: torch.Tensor) -> torch.Tensor:
    x = t.detach().reshape(-1)
    if x.numel() == 0:
        return torch.zeros(2, dtype=torch.int64, device=x.device)
    b = x.contiguous().view(torch.uint8)
    pad = (-b.numel()) % 8
    if pad:
        b = torch.cat([b, b.new_zeros(pad)])
    w = b.view(torch.int64)
    idx = torch.arange(1, w.numel() + 1, device=w.device, dtype=torch.int64)
    return torch.stack([w.sum(), (w * (idx * _WEIGHT)).sum()])


def _scalar_tensor(text: str) -> torch.Tensor:
    # process-independent (never Python's salted hash(): ranks must agree)
    return torch.tensor([int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)], dtype=torch.int64)


def _iter_state(module: torch.nn.Module, optimizer=None, params_only: bool = False):
    for name, p in module.named_parameters():
        yield name, p
    for name, b in module.named_buffers():
        yield "buf:" + name, b
    if optimizer is not None and not params_only:
        sd = optimizer.state_dict()
        for pid in sorted(sd["state"]):
            for key in sorted(sd["state"][pid]):
                v = sd["state"][pid][key]
                if torch.is_tensor(v):
                    yield "opt:%s/%s" % (pid, key), v
                else:
                    yield "opt:%s/%s" % (pid, key), _scalar_tensor(repr(v))
        yield "opt:param_groups", _scalar_tensor(repr(sd["param_groups"]))


def fingerprint(module: torch.nn.Module, optimizer=None, params_only: bool = False):
    """(hex digest, int64 numpy vector of per-tensor fingerprints).

    The vector is what ranks compare with an element-wise MIN/MAX allreduce; the digest folds the
    tensor names, dtypes and shapes in as well so a rename or reshape also changes it.
    """
    h = hashlib.sha256()
    parts = []
    for name, t in _iter_state(module, optimizer, params_only):
        h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        parts.append(_fp_tensor(t).to("cpu", non_blocking=False) if t.device.type == "cpu"
                     else _fp_tensor(t))
    if not parts:
        return h.hexdigest(), torch.zeros(0, dtype=torch.int64).numpy()
    dev = next((p.device for p in parts if p.device.type != "cpu"), parts[0].device)
    vec = torch.stack([p.to(dev) for p in parts]).reshape(-1).cpu().numpy()
    h.update(vec.tobytes())
    return h.hexdigest(), vec


def identical_across_ranks(vec, comm, ops=None) -> bool:
    """True iff every rank holds the same fingerprint vector (MPI MIN/MAX, no NCCL op).

    `ops` = (min_op, max_op) of the communicator; default mpi4py's MPI.MIN/MPI.MAX."""
    import numpy as np
    if ops is None:
        from mpi4py import MPI
        ops = (MPI.MIN, MPI.MAX)
    lo = np.empty_like(vec)
    hi = np.empty_like(vec)
    comm.Allreduce(vec, lo, op=ops[0])
    comm.Allreduce(vec, hi, op=ops[1])
    return bool((lo == hi).all())


def fast_rewrap(module, optimizer, ddp_cls, ddp_kwargs, comm, verify=True, sync=None, ops=None):
    """Re-wrap `module` in a new DDP without the host-side digest and without the redundant broadcast.

    Math-preserving by construction: the wrapper only changes how gradients are bucketed.
      * The module state (params, buffers) is fingerprinted on the device and compared across ranks
        (MPI MIN/MAX).  DDP keeps ranks identical, so normally every rank agrees and the new DDP is
        built with `init_sync=False` (no parameter/buffer broadcast from rank 0).  If any rank
        differs (it should not), every rank takes the broadcast path (`init_sync=True`), exactly the
        legacy behaviour, because the decision comes from the same allreduce on every rank.
      * The pre/post fingerprint of params, buffers and optimizer state is compared on each rank.
      * The default process group is reused (DDP's `process_group=None`); no group is created.
      * `broadcast_buffers` and every other DDP argument are untouched (the caller's kwargs).
    Returns (ddp, info) with info = {phases: {name: seconds}, init_sync, ok, before, after}.
    """
    import time
    ph = {}
    t = time.perf_counter()

    def lap(name):
        nonlocal t
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        ph[name] = now - t
        t = now

    before = after = None
    if verify:
        before, vec_all = fingerprint(module, optimizer)
        lap("fingerprint_before")
        _, vec_params = fingerprint(module, None, params_only=True)
        identical = identical_across_ranks(vec_params, comm, ops) if comm is not None else True
        lap("cross_rank_check")
    else:
        identical = False                         # unverified: keep the legacy broadcast
    init_sync = (not identical) if sync is None else bool(sync)
    ddp = ddp_cls(module, init_sync=init_sync, **ddp_kwargs)
    lap("ddp_init")
    ok = True
    if verify:
        after, _ = fingerprint(ddp.module, optimizer)
        lap("fingerprint_after")
        ok = (after == before)
    return ddp, {"phases": ph, "init_sync": init_sync, "ok": ok, "before": before, "after": after}
