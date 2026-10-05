"""DDP communication-probe hook shared by initial setup and every live DDP rebuild.

`STORMER_COMM_PROBE`:
  blocking (default, historical) - per-bucket hook that synchronizes, waits, and times the
                                   all-reduce inside the `dft_ai.comm.all_reduce` region.  This
                                   serializes each bucket (no comm/backward overlap).
  event                          - B1 (JUDGE-BATCH-4-2026-09-27.md E2): per-bucket hook records a
                                   pair of CUDA events on the comm stream around the async
                                   all-reduce and does NOT synchronize or wait inside the hook, so
                                   backward/comm overlap is preserved. Elapsed ms are read back
                                   once per step at the caller's existing sync point (train.py's
                                   `STORMER_COMM_TIMING`/`STORMER_COMM_OPTIMIZE` per-step
                                   `torch.cuda.synchronize()`, reused rather than duplicated) via
                                   `read_and_clear_bucket_ms`, which is safe to call only after
                                   that sync has run.  Event mode also supports EXPOSED comm
                                   (`compute_exposed_comm_ms`, called BEFORE the drain, from the
                                   raw event pairs): the tail of the last bucket's all-reduce that
                                   runs after backward finishes, i.e. the non-overlapped part.
  off                            - register NO hook: native, overlapped DDP.  No comm events.
Both the setup path and `_rebuild_ddp` must build the hook here so a rebuild cannot silently drop
the trace region or the hook.
"""

from __future__ import annotations

import contextlib
import os
import time

import torch
import torch.distributed as dist

PROBE_MODES = ("blocking", "event", "off", "ab")


def comm_probe_mode() -> str:
    mode = os.environ.get("STORMER_COMM_PROBE", "blocking") or "blocking"
    if mode not in PROBE_MODES:
        raise ValueError("STORMER_COMM_PROBE must be one of %s, got %r" % (PROBE_MODES, mode))
    return mode


def hook_wanted(comm_timing: bool, comm_optimize: bool, tracer_enabled: bool, world: int) -> bool:
    return bool((comm_timing or comm_optimize or tracer_enabled) and world > 1
                and comm_probe_mode() != "off")


def make_blocking_hook(region=None, sync_probe: bool = False, cuda_sync=None):
    """Return a DDP comm hook `(state_list, bucket) -> Future`; appends per-bucket ms to state."""
    if cuda_sync is None:
        cuda_sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    region = region if region is not None else contextlib.nullcontext()

    def _hook(state, bucket):
        with region:
            cuda_sync()
            if sync_probe:
                dist.barrier()
            t0 = time.time()
            _buf = bucket.buffer()
            _buf.div_(dist.get_world_size())   # default-hook semantics: average, not sum
            work = dist.all_reduce(_buf, async_op=True)
            work.wait()
            cuda_sync()
            state.append((time.time() - t0) * 1000)
        return work.get_future().then(lambda fut: fut.value()[0])

    return _hook


def make_event_hook(region=None):
    """B1 non-blocking hook: `(state_list, bucket) -> Future`; appends a `(ready, end)` CUDA
    `torch.cuda.Event` pair per bucket to `state`. `ready` is recorded at hook ENTRY, i.e. the
    instant this bucket's gradients became ready (before the async all-reduce is issued) - also
    used by `compute_exposed_comm_ms` as the reference point for the last bucket (see its
    docstring for why an external "backward end" event is NOT a valid reference). No
    `cuda.synchronize()`, no `work.wait()`: the all-reduce stays async and overlapped with
    backward. Events are read back once per step by `read_and_clear_bucket_ms`, after the caller
    has already synchronized elsewhere."""
    region = region if region is not None else contextlib.nullcontext()

    def _hook(state, bucket):
        with region:
            ready = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            ready.record()
            _buf = bucket.buffer()
            _buf.div_(dist.get_world_size())   # default-hook semantics: average, not sum (probe must not change gradients)
            work = dist.all_reduce(_buf, async_op=True)
            end.record()
            state.append((ready, end))
        return work.get_future().then(lambda fut: fut.value()[0])

    return _hook


def read_and_clear_bucket_ms(state) -> list[float]:
    """Convert `state` to a list of per-bucket comm ms and clear it in place.

    `state` holds plain floats under `blocking` mode (already elapsed ms) or `(start, end)`
    CUDA event pairs under `event` mode. In `event` mode the caller MUST have already run a
    `torch.cuda.synchronize()` (or otherwise guaranteed the events completed) since the last
    hook invocation, or `Event.elapsed_time` raises/returns garbage.
    """
    if state and isinstance(state[0], tuple):
        ms = [start.elapsed_time(end) for start, end in state]
    else:
        ms = list(state)
    state.clear()
    return ms


def record_backward_end():
    """A CUDA event on the current (compute) stream, recorded right after `loss.backward()` returns.

    Event/ab probe modes only; None otherwise (and when CUDA is absent). Math-neutral: it only
    timestamps the stream. Pass it to `compute_exposed_comm_ms`."""
    if comm_probe_mode() not in ("event", "ab") or not torch.cuda.is_available():
        return None
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


def exposed_vs_reference_ms(backward_ms, compute_only_ms) -> float:
    """Exposed comm = backward wall of a real step minus the comm-free backward reference (ms, >= 0).

    The reference is measured once at the start of the run (train.py `_calibrate_compute_only_backward`:
    first batch, DDP `no_sync()`, gradients dropped, RNG restored). 0.0 while no reference exists.
    Verified against the paired skip-all-reduce counterfactual at 16N (inventory/consolidate/comm-verify):
    the last-bucket tail below reads ~6 ms there because communication shows up as a DILATION of the
    backward compute, not as a wait after it."""
    if compute_only_ms is None or backward_ms is None:
        return 0.0
    return max(0.0, float(backward_ms) - float(compute_only_ms))


def compute_exposed_comm_ms(state, backward_end=None) -> float:
    """TAIL diagnostic (event/ab modes): the time the step WAITS for all-reduce completion beyond
    backward compute, i.e. from the instant the LAST bucket's gradients became ready to the
    instant `backward()` returned.

    `state[-1]` is the last bucket's `(ready, end)` pair. `ready` is recorded at hook ENTRY, when
    the bucket's gradients became ready; DDP reduces buckets in gradient-ready order, so the last
    one marks the end of ALL backward compute for the step and nothing is left to hide behind.
    `backward_end` is the event `record_backward_end()` recorded on the compute stream right
    after `backward()` returned. DDP's Reducer awaits every comm-hook future in
    `finalize_backward()` before `backward()` returns, and that wait is a STREAM wait (the
    compute stream blocks on the all-reduce's completion event), so an event recorded after
    `backward()` returns completes only after the last all-reduce finished. Therefore
    ready(last) -> backward_end is the completion-based exposed time: ~0 when communication was
    fully hidden, the unhidden tail (plus any all-reduce backlog behind it) otherwise.

    NOTE (2026-10-04, consolidation): this tail is NOT the exposed-comm value used by the loop (see
    `exposed_vs_reference_ms`); at 16N it reads ~6 ms against a 168-182 ms counterfactual. The earlier version measured
    `ready(last) -> end(last)`, where `end` is recorded right after the all-reduce is ISSUED
    (async), so it read the LAUNCH cost, 3-4 ms, while the paired counterfactual (all-reduce
    skipped) showed 136-212 ms exposed at 16N/32N (inventory/comm-exposure). The 2026-09-27
    note that "an event after backward() returns is always later than every bucket's end" was
    true but beside the point: it compared that event against the issue-time `end`, not against
    the last bucket's `ready`.

    `state` must still hold its raw `(ready, end)` CUDA event pairs for this step: call this
    BEFORE `read_and_clear_bucket_ms` drains it, after the per-step `torch.cuda.synchronize()`.
    Empty/non-event state, or no `backward_end`, returns 0.0 (never raises).
    """
    if backward_end is None or not state or not isinstance(state[0], tuple):
        return 0.0
    last_ready, _last_end = state[-1]
    return max(0.0, last_ready.elapsed_time(backward_end))


def compute_max_bucket_span_ms(state) -> float:
    """Diagnostic (event mode only): the single longest per-bucket [ready, end] span this step,
    over ALL buckets, not just the last. Unlike `compute_exposed_comm_ms` this is not necessarily
    "exposed" for any bucket but the last (earlier buckets can hide behind later backward work),
    but a large value here on a NON-last bucket is worth knowing about (e.g. a straggler transfer
    that happens to still fit inside the remaining backward window). 0.0 for empty/non-event state.
    """
    if not state or not isinstance(state[0], tuple):
        return 0.0
    return max(max(0.0, ready.elapsed_time(end)) for ready, end in state)


# ---------------------------------------------------------------------------------------------
# comm-exposure worker (2026-10-04): MEASUREMENT-ONLY "ab" probe.  Interleaves, in blocks of
# STORMER_COMM_AB_BLOCK steps (default 15), four hook behaviours so exposed communication can be
# measured as a paired step-time difference inside ONE job (same nodes, same Lustre state):
#   ev   - the campaign event hook (CUDA event pair + div_ + async all_reduce)  [production-like]
#   skip - same events, div_/all_reduce SKIPPED (counterfactual: zero communication; also a
#          control for what the event probe reads when nothing is communicated).  NOT math
#          preserving: ranks' gradients are not averaged in these blocks.  Timing study only.
#   nat  - no events, div_ + async all_reduce (closest to plain overlapped DDP)
#   blk  - barrier, then synchronous timed all_reduce per bucket (pure total all-reduce time,
#          serialized, straggler wait excluded and accumulated separately)
# The step counter is advanced by train.py once per training step on every rank, so the block
# schedule is identical across ranks (required: modes change the collective pattern).
_AB_MODES = tuple((os.environ.get("STORMER_COMM_AB_MODES", "ev,skip,nat,blk")).split(","))
_AB_STEP = 0
_AB_BARRIER_MS = [0.0]


def ab_block() -> int:
    return int(os.environ.get("STORMER_COMM_AB_BLOCK", "15"))


def ab_mode_for(step: int) -> str:
    return _AB_MODES[(step // ab_block()) % len(_AB_MODES)]


def ab_phase_for(step: int) -> int:
    return step // (ab_block() * len(_AB_MODES))


def ab_cap_for(step: int, default: float) -> float:
    """Bucket cap for this step's phase (one phase = one pass through all modes); STORMER_COMM_AB_CAPS is a
    comma list cycled by phase; empty = keep the starting cap."""
    caps = [c for c in os.environ.get("STORMER_COMM_AB_CAPS", "").split(",") if c]
    if not caps:
        return default
    return float(caps[ab_phase_for(step) % len(caps)])


def ab_cap_now(default: float) -> float:
    return ab_cap_for(max(_AB_STEP - 1, 0), default)


def ab_advance() -> str:
    """Called once at the start of each training step on every rank. Returns this step's mode."""
    global _AB_STEP
    mode = ab_mode_for(_AB_STEP)
    _AB_STEP += 1
    return mode


def ab_current_mode() -> str:
    return ab_mode_for(max(_AB_STEP - 1, 0))


def ab_pop_barrier_ms() -> float:
    v = _AB_BARRIER_MS[0]
    _AB_BARRIER_MS[0] = 0.0
    return v


def make_ab_hook(region=None):
    region = region if region is not None else contextlib.nullcontext()

    def _hook(state, bucket):
        mode = ab_current_mode()
        _buf = bucket.buffer()
        if mode == "nat":
            _buf.div_(dist.get_world_size())
            work = dist.all_reduce(_buf, async_op=True)
            return work.get_future().then(lambda fut: fut.value()[0])
        if mode == "sync":
            # synchronization only: a 1-element all-reduce per bucket keeps every rank coupled exactly as the
            # real hook does (skew is paid) but moves no gradient bytes. Gradients are NOT averaged: timing only.
            tiny = torch.zeros(1, device=_buf.device, dtype=_buf.dtype)
            work = dist.all_reduce(tiny, async_op=True)
            return work.get_future().then(lambda fut, _b=_buf: _b)
        if mode == "blk":
            with region:
                torch.cuda.synchronize()
                tb = time.time()
                dist.barrier()
                _AB_BARRIER_MS[0] += (time.time() - tb) * 1000
                t0 = time.time()
                _buf.div_(dist.get_world_size())
                work = dist.all_reduce(_buf, async_op=True)
                work.wait()
                torch.cuda.synchronize()
                state.append((time.time() - t0) * 1000)
            return work.get_future().then(lambda fut: fut.value()[0])
        # ev / skip: event pair around the (possibly skipped) all-reduce
        with region:
            ready = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            ready.record()
            if mode == "skip":
                fut = torch.futures.Future()
                fut.set_result(_buf)
                end.record()
                state.append((ready, end))
                return fut
            _buf.div_(dist.get_world_size())
            work = dist.all_reduce(_buf, async_op=True)
            end.record()
            state.append((ready, end))
        return work.get_future().then(lambda fut: fut.value()[0])

    return _hook


def make_hook_for_mode(region, sync_probe: bool):
    m = comm_probe_mode()
    if m == "event":
        return make_event_hook(region)
    if m == "ab":
        return make_ab_hook(region)
    return make_blocking_hook(region, sync_probe)
