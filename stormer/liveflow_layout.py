"""Runtime-changeable, math-neutral read-layout knob for the shared StorMer/SFNO HDF5 read path.

Context (inventory/layout-knob/HANDOFF.md).  GlanceD-IO's "dataset access" and "transfer size"
knobs rewrite the dataset (num_dset_per_record, HDF5 chunk_dims, Lustre stripe), which cannot
change at runtime.  This module is the runtime analogue on the READ side only: the file bytes are
untouched, the same bytes reach the model, and only HOW they are fetched changes:

  transfer_mb  bytes per read call.  0 (default) = today's behaviour: one h5py ``dset[:]`` per
               variable in name order (one 4.15 MB read per variable).  >0 = coalesced mode: the
               requested variables' contiguous extents are sorted by FILE OFFSET, extents whose gap
               is <= gap_kb are merged into one span (the gap bytes are read and discarded), and
               each span is fetched with preadv() calls of at most transfer_mb MB, aligned to
               absolute multiples of transfer_mb (so calls start on stripe boundaries).
  read_threads number of threads that issue those calls for ONE sample (intra-sample read
               concurrency; os.preadv releases the GIL).  1 = serial.
  gap_kb       maximum hole merged between two extents (read amplification bound).

Math neutrality: every variable is returned as a float32 ndarray whose bytes are exactly the bytes
of the HDF5 contiguous, unfiltered dataset (what ``dset[:]`` returns).  Anything that is not a
native-endian, contiguous, unfiltered, internally stored dataset falls back to the h5py path.

Runtime change: ``set_layout(...)`` writes three int64 slots in a RawArray created at import time
(i.e. in the trainer process, BEFORE DataLoader workers fork) so forked workers see the change.
Each ``read_variables`` call snapshots the knobs once at entry, so the safe point is a sample
boundary: a sample is read entirely under one setting, and because every setting yields identical
bytes a change is safe at any time on any rank independently.

Env defaults (read once at import): LIVEFLOW_LAYOUT_TRANSFER_MB=0, LIVEFLOW_LAYOUT_READ_THREADS=1,
LIVEFLOW_LAYOUT_GAP_KB=1024.  LIVEFLOW_LAYOUT_STATS_DIR=<dir> enables one JSON line per file read
(bytes, calls, seconds, mode) in <dir>/layout-<host>-<pid>.jsonl for mechanism evidence.
"""
from __future__ import annotations

import json
import multiprocessing as _mp
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import h5py
import numpy as np

_MB = 1 << 20
_KNOBS = _mp.RawArray("q", 4)  # [transfer_bytes, read_threads, gap_bytes, set_count]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


_KNOBS[0] = max(0, _env_int("LIVEFLOW_LAYOUT_TRANSFER_MB", 0)) * _MB
_KNOBS[1] = max(1, _env_int("LIVEFLOW_LAYOUT_READ_THREADS", 1))
_KNOBS[2] = max(0, _env_int("LIVEFLOW_LAYOUT_GAP_KB", 1024)) * 1024

# LIVEFLOW_READ_DONTNEED=1 (default off): after a sample file has been completely read into private memory, ask the kernel to
# drop its clean page-cache pages (posix_fadvise DONTNEED).  Math-neutral: it runs strictly AFTER the bytes were copied out of
# the file (h5py ``d[:]`` / preadv into private buffers; no mmap), never writes, and a failure is ignored.  Stall-diag 2026-10-05:
# tests whether page-cache pressure causes the arm-position effect.
_DONTNEED = _env_int("LIVEFLOW_READ_DONTNEED", 0) == 1 and hasattr(os, "posix_fadvise")


def _dontneed(paths) -> None:
    for p in set(paths):
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


def set_layout(transfer_mb: Optional[int] = None, read_threads: Optional[int] = None,
               gap_kb: Optional[int] = None) -> dict:
    """Runtime setter.  Takes effect for the next sample read by each worker (sample boundary)."""
    if transfer_mb is not None:
        if transfer_mb < 0:
            raise ValueError("transfer_mb must be >= 0 (0 = default per-variable h5py reads)")
        _KNOBS[0] = int(transfer_mb) * _MB
    if read_threads is not None:
        if read_threads < 1:
            raise ValueError("read_threads must be >= 1")
        _KNOBS[1] = int(read_threads)
    if gap_kb is not None:
        if gap_kb < 0:
            raise ValueError("gap_kb must be >= 0")
        _KNOBS[2] = int(gap_kb) * 1024
    _KNOBS[3] += 1
    return get_layout()


def get_layout() -> dict:
    return {"transfer_mb": _KNOBS[0] / _MB, "read_threads": int(_KNOBS[1]), "gap_kb": _KNOBS[2] / 1024,
            "set_count": int(_KNOBS[3])}


# --------------------------------------------------------------------------- per-process state
_POOL = {"pid": None, "n": 0, "ex": None}
_STATS = {"pid": None, "fh": None}


def _pool(n: int) -> ThreadPoolExecutor:
    if _POOL["pid"] != os.getpid() or _POOL["n"] != n:
        # a forked child must not reuse the parent's executor; a size change builds a new one
        _POOL.update(pid=os.getpid(), n=n, ex=ThreadPoolExecutor(max_workers=n, thread_name_prefix="lfl"))
    return _POOL["ex"]


def _emit(rec: dict) -> None:
    d = os.environ.get("LIVEFLOW_LAYOUT_STATS_DIR")
    if not d:
        return
    if _STATS["pid"] != os.getpid():
        os.makedirs(d, exist_ok=True)
        _STATS.update(pid=os.getpid(),
                      fh=open(os.path.join(d, f"layout-{socket.gethostname()}-{os.getpid()}.jsonl"), "a", buffering=1))
    _STATS["fh"].write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- default (today's) path
def _read_default(path: str, variables: Sequence[str], group: Optional[str]):
    f = h5py.File(path, "r")
    files = {path}
    try:
        g = f[group] if group else f
        out = []
        for v in variables:
            d = g[v]
            out.append(d[:])
            if _DONTNEED:
                files.add(d.file.filename)  # the file that actually holds the bytes (ExternalLink targets)
    finally:
        f.close()
    if _DONTNEED:
        _dontneed(files)
    return out


# --------------------------------------------------------------------------- coalesced path
def _extents(path: str, variables: Sequence[str], group: Optional[str]):
    """Return (target_path, [(var_index, offset, nbytes, dtype, shape)]) or None to fall back.

    ``path`` may be a flat view whose entries are ExternalLinks (all must resolve to ONE target
    file) or a plain file.  Only HDF5 metadata is read here (no raw data)."""
    f = h5py.File(path, "r")
    try:
        g = f[group] if group else f
        links = [g.get(v, getlink=True) for v in variables]
        if all(isinstance(l, h5py.HardLink) for l in links):
            target = path
            dpaths = [f"{group}/{v}" if group else v for v in variables]
        elif all(isinstance(l, h5py.ExternalLink) for l in links):
            names = {l.filename for l in links}
            if len(names) != 1:
                return None
            target = names.pop()
            if not os.path.isabs(target):
                target = os.path.join(os.path.dirname(path), target)
            dpaths = [l.path for l in links]
        else:
            return None
        tf = f if target == path else h5py.File(target, "r")
        try:
            ex = []
            for i, dp in enumerate(dpaths):
                d = tf[dp]
                pl = d.id.get_create_plist()
                if (pl.get_layout() != h5py.h5d.CONTIGUOUS or pl.get_nfilters() != 0
                        or pl.get_external_count() != 0 or not d.dtype.isnative or d.dtype.hasobject):
                    return None
                off = d.id.get_offset()
                nbytes = int(np.prod(d.shape)) * d.dtype.itemsize
                if off is None or off % d.dtype.itemsize != 0 or d.id.get_storage_size() != nbytes:
                    return None
                ex.append((i, int(off), nbytes, d.dtype, d.shape))
            return target, ex
        finally:
            if tf is not f:
                tf.close()
    finally:
        f.close()


def _plan(ex, transfer: int, gap: int):
    """Sort extents by offset, merge across holes <= gap, cut into calls <= transfer bytes aligned
    to absolute multiples of transfer.  Returns (spans, calls): spans = [(start, end, [extent...])],
    calls = [(span_idx, abs_offset, nbytes)]."""
    order = sorted(ex, key=lambda e: e[1])
    spans = []
    for e in order:
        if spans and e[1] - spans[-1][1] <= gap:
            spans[-1][1] = max(spans[-1][1], e[1] + e[2])
            spans[-1][2].append(e)
        else:
            spans.append([e[1], e[1] + e[2], [e]])
    calls = []
    for si, (s, t, _) in enumerate(spans):
        pos = s
        while pos < t:
            nxt = min(t, (pos // transfer + 1) * transfer)
            calls.append((si, pos, nxt - pos))
            pos = nxt
    return spans, calls


def _read_coalesced(path: str, variables: Sequence[str], group: Optional[str], transfer: int, nthr: int,
                    gap: int):
    t0 = time.perf_counter()
    got = _extents(path, variables, group)
    if got is None:
        return None
    target, ex = got
    spans, calls = _plan(ex, transfer, gap)
    bufs = [np.empty(t - s, dtype=np.uint8) for s, t, _ in spans]
    t1 = time.perf_counter()
    fd = os.open(target, os.O_RDONLY)
    try:
        def one(c):
            si, off, n = c
            base = off - spans[si][0]
            mv = memoryview(bufs[si])[base:base + n]
            done = 0
            while done < n:  # preadv may return short counts
                r = os.preadv(fd, [mv[done:]], off + done)
                if r <= 0:
                    raise OSError(f"short read at {off + done} in {target}")
                done += r
            return n
        if nthr <= 1 or len(calls) == 1:
            lat = []
            for c in calls:
                tc = time.perf_counter()
                one(c)
                lat.append(time.perf_counter() - tc)
        else:
            def timed(c):
                tc = time.perf_counter()
                one(c)
                return time.perf_counter() - tc
            lat = list(_pool(nthr).map(timed, calls))
    finally:
        os.close(fd)
    if _DONTNEED:
        _dontneed([target])
    t2 = time.perf_counter()
    out = [None] * len(variables)
    for si, (s, t, es) in enumerate(spans):
        for (i, off, nbytes, dt, shape) in es:
            out[i] = np.frombuffer(bufs[si], dtype=dt, count=nbytes // dt.itemsize, offset=off - s).reshape(shape)
    read_bytes = sum(c[2] for c in calls)
    _emit({"mode": "coalesced", "file": target, "meta_s": t1 - t0, "read_s": t2 - t1, "calls": len(calls),
           "spans": len(spans), "bytes": read_bytes, "useful_bytes": sum(e[2] for e in ex),
           "call_lat_s": lat, "transfer_mb": transfer / _MB, "threads": nthr, "t": time.time()})
    return out


def read_variables(path: str, variables: Sequence[str], group: Optional[str] = None):
    """Drop-in for ``[h5[group][v][:] for v in variables]``.  Returns a list of ndarrays."""
    transfer, nthr, gap = int(_KNOBS[0]), int(_KNOBS[1]), int(_KNOBS[2])  # one snapshot per sample
    if transfer > 0:
        try:
            out = _read_coalesced(path, variables, group, transfer, nthr, gap)
        except OSError:
            raise
        if out is not None:
            return out
    t0 = time.perf_counter()
    out = _read_default(path, variables, group)
    dt = time.perf_counter() - t0
    _emit({"mode": "default" if transfer == 0 else "fallback", "file": path, "read_s": dt, "calls": len(variables),
           "bytes": int(sum(a.nbytes for a in out)), "useful_bytes": int(sum(a.nbytes for a in out)),
           "transfer_mb": transfer / _MB, "threads": nthr, "t": time.time()})
    return out


def emit_sample(index: int, read_s: float, total_s: float) -> None:
    """Optional per-sample record from the dataset (both files read + preprocess), stats dir only."""
    _emit({"mode": "sample", "index": int(index), "read_s": read_s, "total_s": total_s, "t": time.time()})
