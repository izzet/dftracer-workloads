# Taken from https://github.com/tung-nd/stormer
# Modified by Troy Arcomano, Sam Foreman, Ray Andrew Sinurat
#
# MIT License
#
# Copyright (c) 2024 Tung Nguyen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Historical versionings:
# - Tung and Troy worked on Stormer
# - Troy and Sam worked on pure pytorch version + mpi4py
# - Ray modified the codebase for
#   - profiling I/O
#   - replicating the Stormer pipeline in DLIO Benchmark
#   - I/O Optimization
#
# This codebase is for single step pretraining only and no validation

from typing import Any, Optional
from functools import partial
import motivation_timers as _mot

import hashlib
import json
import logging
import faulthandler
import multiprocessing as _mp
import os
import signal
import socket
import sys
import threading
import time
import contextlib
import statistics
import argparse
import tempfile
from glob import glob
from contextlib import contextmanager
from collections.abc import Generator
import warnings
from pathlib import Path

# LC HACK: work around so that "import torch" will not change CPU affinity
# see https://rzlc.llnl.gov/jira/browse/ELCAP-386
os.environ.pop("OMP_PLACES", None)
os.environ.pop("OMP_PROC_BIND", None)

# libdftracer_preload is already mapped by the dynamic loader before this
# Python module starts.  Remove only the loader entries *before h5py/torch*
# import Python's platform path (which execs `uname -p` on this Cray Python).
# Ordinary DataLoader workers use fork and retain the mapping; a later exec
# child instead starts without an interposer and cannot claim a stray lane.
from preload_sanitization import remove_preload_from_exec_environment
_PRELOAD_EXEC_ENV_REMOVED = remove_preload_from_exec_environment()

# from mpi4py import MPI

import h5py
from liveflow_layout import read_variables as _layout_read_variables
import numpy as np
import apply_ack
import consensus as _consensus
import comm_hook
import rebuild_invariants

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.nn.parallel.distributed import _MixedPrecision
except ImportError:
    _MixedPrecision = None
from torch.utils.data import Dataset
import torch.distributed as dist

from models.stormer.stormer import Stormer
from models.scheduler import configure_lr_scheduler
from models.optimizer import configure_optimizers
from models.utils import replace_constant, pad
from models.metrics import lat_weighted_mse
from models.data_utils import WEIGHT_DICT
from latitude_metadata import load_latitude
from interval_schedule import INTERVAL_SELECTOR_VERSION, select_interval
from interval_index import verify_index
from live_data_loader import make_keyed_dataloader_generator

from utils.torch_utils import seed_everything
from utils.precision import setup_precision_context, get_precision_dtype
from utils.perf_tracer import PerfTracer, PERFTRACER_ENABLE, dft_ai
# BUGFIX 2026-09-27 (caught live: a bare-int log_event(int_args=...) crashed every rank at step 1
# with a TypeError, correctly aborted by the fail-fast fix rather than hanging). `int_args` values
# must be (tag_type, value) 2-tuples -- the same convention `liveflow_validation/halo3d_train.py`
# already established for this exact defect.
try:
    from dftracer.python.common import TagValue, TagDType, TagType
    def _TAG_INT(v): return TagValue(int(v), TagDType.INT, TagType.KEY).value()
except Exception:
    def _TAG_INT(v): return (0, int(v))
from utils.logging import configure_logging
from utils.mpi import MPIUtils, get_master_addr_and_port
from loop_exit_consensus import (LoopExitConsensus, UnequalLoaderExhaustion,
                                 retain_unequal_exhaustion, wrap_step_iterator)
from abort_on_loop_exception import abort_on_loop_exception as _abort_on_loop_exception


_WORKER_POSIX_ROSTER_SCHEMA = "stormer-worker-posix-roster.v1"


def _write_worker_posix_roster(
    worker_id: int,
    *,
    directory: str,
    run_prefix: str,
    rank: int,
    expected_workers: int,
) -> None:
    """Publish one worker-owned roster record without replacing a prior run's record."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None or worker_info.id != worker_id:
        raise RuntimeError("worker POSIX roster callback must run in its assigned DataLoader worker")
    directory_path = Path(directory)
    if not directory_path.is_absolute() or not directory_path.is_dir():
        raise RuntimeError("worker POSIX roster directory must be an existing absolute shared directory")
    if not run_prefix or rank < 0 or expected_workers < 1:
        raise RuntimeError("invalid worker POSIX roster identity")
    prefix_digest = hashlib.sha256(run_prefix.encode("utf-8")).hexdigest()[:16]
    destination = directory_path / (
        f"{_WORKER_POSIX_ROSTER_SCHEMA}.{prefix_digest}.rank-{rank}.worker-{worker_id}.json"
    )
    record = {
        "schema_version": _WORKER_POSIX_ROSTER_SCHEMA,
        "run_prefix": run_prefix,
        "host": socket.gethostname(),
        "rank": rank,
        "worker_id": worker_id,
        "pid": os.getpid(),
        "expected_workers": expected_workers,
    }
    descriptor, temporary = tempfile.mkstemp(prefix=".stormer-worker-roster-", dir=directory_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # link(2) is atomic and refuses an existing destination, unlike
            # replace(), so a stale/restarted worker cannot overwrite evidence.
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise RuntimeError(f"worker POSIX roster destination already exists: {destination}") from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _worker_posix_roster_init(args, loader_config: dict[str, Any]):
    """Return the opt-in ordinary-DataLoader worker callback, or ``None`` by default."""
    if os.environ.get("STORMER_WORKER_POSIX_GATE", "0") != "1":
        return None
    workers = loader_config.get("num_workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("STORMER_WORKER_POSIX_GATE requires one or more ordinary DataLoader workers")
    if int(getattr(args, "epochs", 0)) != 1:
        raise ValueError("STORMER_WORKER_POSIX_GATE requires exactly one epoch")
    if os.environ.get("DFOPTIMIZER_ENABLE", "0") != "0":
        raise ValueError("STORMER_WORKER_POSIX_GATE requires DFOPTIMIZER_ENABLE=0")
    if bool(loader_config.get("persistent_workers", False)):
        raise ValueError("STORMER_WORKER_POSIX_GATE forbids persistent/replaced workers")
    directory = os.environ.get("STORMER_WORKER_POSIX_ROSTER_DIR", "")
    trace_prefix = os.environ.get("DFTRACER_SHM_PREFIX", "")
    liveflow_prefix = os.environ.get("LIVEFLOW_SHM_PREFIX", "")
    if not directory or not trace_prefix or trace_prefix != liveflow_prefix:
        raise ValueError(
            "STORMER_WORKER_POSIX_GATE requires roster directory and matching DFTRACER/LIVEFLOW SHM prefixes"
        )
    directory_path = Path(directory)
    if not directory_path.is_absolute() or not directory_path.is_dir():
        raise ValueError("STORMER_WORKER_POSIX_ROSTER_DIR must name an existing absolute shared directory")
    return partial(
        _write_worker_posix_roster,
        directory=str(directory_path),
        run_prefix=trace_prefix,
        rank=MPIUtils.rank(),
        expected_workers=workers,
    )

_MPI_HELPER_ROSTER_SCHEMA = "stormer-mpi-helper-roster.v1"


def _read_proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _write_mpi_helper_roster() -> None:
    """Record this rank's unexec'd MPI-init fork children (Cray MPICH), independently of the trace stream.

    `from mpi4py import MPI` makes Cray MPICH fork a copy of the rank that inherits the DFTracer
    preload and claims its own trace lane.  It is not a DataLoader worker.  The rank enumerates its
    own live children right after MPI init (before any DataLoader exists) so the worker-POSIX gate
    can classify those lane owners exactly (ppid == this rank, same cmdline) instead of loosening
    the worker roster.  Opt-in with the worker-POSIX gate; a missing directory fails closed.
    """
    if os.environ.get("STORMER_WORKER_POSIX_GATE", "0") != "1":
        return
    directory = os.environ.get("STORMER_MPI_HELPER_ROSTER_DIR", "")
    run_prefix = os.environ.get("DFTRACER_SHM_PREFIX", "")
    directory_path = Path(directory) if directory else None
    if not run_prefix or directory_path is None or not directory_path.is_absolute() or not directory_path.is_dir():
        raise RuntimeError("STORMER_WORKER_POSIX_GATE requires STORMER_MPI_HELPER_ROSTER_DIR (existing absolute dir)")
    me = os.getpid()
    helpers = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        try:
            with open(f"/proc/{entry}/stat", "r", encoding="utf-8", errors="replace") as handle:
                ppid = int(handle.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if ppid == me:
            helpers.append({"pid": int(entry), "ppid": ppid, "cmdline": _read_proc_cmdline(int(entry))})
    rank = MPIUtils.rank()
    record = {
        "schema_version": _MPI_HELPER_ROSTER_SCHEMA,
        "run_prefix": run_prefix,
        "host": socket.gethostname(),
        "rank": rank,
        "rank_pid": me,
        "rank_cmdline": _read_proc_cmdline(me),
        "helpers": sorted(helpers, key=lambda h: h["pid"]),
    }
    digest = hashlib.sha256(run_prefix.encode("utf-8")).hexdigest()[:16]
    destination = directory_path / f"{_MPI_HELPER_ROSTER_SCHEMA}.{digest}.rank-{rank}.json"
    descriptor, temporary = tempfile.mkstemp(prefix=".stormer-mpi-helper-roster-", dir=directory_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)  # atomic; refuses to replace a prior record
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------------------------
# DECODE DOMAIN (LIVEFLOW-STATUS.md sec 36). Enabled by STORMER_TWOSTAGE=1 against a gzip+CHUNKED
# ERA5 copy (see liveflow_validation/compress_era5.py). Splits the sample read into
#   STAGE A storage: read_direct_chunk() -> raw compressed bytes   (h5py, sequential, not thread-safe)
#   STAGE B decode : zlib.decompress                                (pure CPU, thread-safe, PARALLEL)
# so `decode_threads` becomes a knob in its own domain, independent of `num_workers` (input).
# Measured on real ERA5: 69 vars = 287 MB/sample, ~1110 ms inflate at 1 thread -> ~139 ms at 8.
#
# The pool is PER PROCESS (one per DataLoader worker) and persistent: building a pool per sample
# would spawn threads 69x per step. It is resizable so the controller can retune it mid-run.
import atexit
import zlib
from concurrent.futures import ThreadPoolExecutor

_TWOSTAGE = os.environ.get("STORMER_TWOSTAGE", "0") == "1"

# dft_ai.data exposes a FIXED set of regions (init/item/preprocess/derive/iter); an undeclared
# name raises AttributeError. That is not loud -- _read_two_stage caught it and fell back to the
# inline read, so the decode domain was silently INACTIVE in every run until this was found.
# Declare the child region explicitly so `with dft_ai.data.decode:` is valid and emits
# cat="data", func_name="decode" -- exactly what the stormer_xlayer preset's reader_decode
# layer matches on.
if _TWOSTAGE:
    try:
        dft_ai.data.create_children({"decode": "decode"})
    except Exception:
        pass    # NoOp tracer (PERFTRACER disabled) already tolerates any attribute
_DECODE_POOL = None
_DECODE_POOL_N = 0
# Current decode concurrency LIMIT (the knob). The pool itself is fixed-width; this is what
# decode_threads actually moves, so a change never creates or destroys a thread.
#
# SHARED MEMORY, not a module int (LIVEFLOW-STATUS sec 82). The decode limit is READ inside
# DataLoader WORKER processes (_read_two_stage) and WRITTEN by the parent in _sync_apply_window.
# As a plain int those are different copies: the parent's assignment never reached a running
# worker, so the only way to deliver a new decode width was to RE-FORK the workers
# (`request_restart`). That re-fork is what makes every worker re-initialise its Mofka client, and
# an occasional child comes up with a connection whose pushes never return, starving a rank and
# killing the job (sec 81). A multiprocessing.Value lives in an mmap segment inherited across
# fork, so parent and workers read the SAME bytes and the knob applies with an integer store.
#
# lock=False: a single aligned int32. A torn read is not possible on x86-64/aarch64, and the value
# is a concurrency hint anyway -- the worst case is one batch decoding at the previous width.
# Created at import, i.e. before any worker is forked, which is what makes it inheritable.
_DECODE_LIMIT = _mp.Value('i', 1, lock=False)
_DECODE_POOL_PID = None

# ---- EXPOSURE: per-stage service time reported BACK from the DataLoader workers ----------------
#
# liveflow_shm/EXPOSURE-PORT-DESIGN.md argues that attribution must use EXPOSURE (how long the
# consumer stalled) rather than service time, because service time moves the wrong way when a
# stage is relieved. For halo3d that is exact: the runtime owns both queues, so at the instant the
# consumer blocks it can ask whether the upstream queue was empty (readers are the constraint,
# STORAGE) or full (decoders are, DECODE).
#
# THAT TEST DOES NOT EXIST HERE, and the design document was wrong to say StorMer had "the same
# in-process reader shape". Stage A and stage B run SERIALLY inside one DataLoader worker with no
# queue between them, and the parent blocks on a single result queue. So:
#
#   the TOTAL exposure at next(loader) is exact
#   the storage/decode SPLIT of it is estimated from the ratio of the two stage service times
#
# Only the RATIO is used, so a constant bias in either measurement cancels. Everything emitted
# from it is flagged as an estimate (StallAccount(exact=False)) so it can never be mistaken for
# the halo3d number.
#
# Shared memory rather than module floats, for the same reason _DECODE_LIMIT is: these are written
# in forked worker processes and read in the parent, and a plain float would be a per-process copy
# that never reaches the parent. Created at import, i.e. before any worker is forked.
# lock=True because several workers increment concurrently and a lost update would bias the ratio.
_STAGE_A_MS = _mp.Value('d', 0.0, lock=True)   # storage: read_direct_chunk
_STAGE_B_MS = _mp.Value('d', 0.0, lock=True)   # decode: zlib inflate


def _stage_times_drain():
    """Read and reset the worker-reported stage times. Parent side, once per consumer stall."""
    with _STAGE_A_MS.get_lock():
        a = _STAGE_A_MS.value
        _STAGE_A_MS.value = 0.0
    with _STAGE_B_MS.get_lock():
        b = _STAGE_B_MS.value
        _STAGE_B_MS.value = 0.0
    return a, b


def _ensure_decode_pool():
    """Fixed-width decode pool, created ONCE at startup -- not lazily on the first read.

    The previous version returned early for n<=1, so with DEC0=1 the pool was still built on the
    first read AFTER the knob rose, i.e. mid-run against a live HIP stream -- the exact thing the
    fixed-width design was supposed to eliminate. That made the D12 run a test of nothing.

    Re-created when the pid changes: pool threads do not survive fork, so each DataLoader worker
    needs its own. Same reinitialize-on-fork pattern DFTracer uses for its Mofka connection.
    """
    global _DECODE_POOL, _DECODE_POOL_N, _DECODE_POOL_PID
    _pid = os.getpid()
    if _DECODE_POOL is not None and _DECODE_POOL_PID == _pid:
        return _DECODE_POOL
    _wide = max(1, int(os.environ.get("STORMER_DECODE_THREADS_MAX", "16") or 16))
    _DECODE_POOL = ThreadPoolExecutor(max_workers=_wide, thread_name_prefix="decode")
    _DECODE_POOL_N = _wide
    _DECODE_POOL_PID = _pid
    # PRIME IT. ThreadPoolExecutor spawns worker threads LAZILY as tasks arrive, not at
    # construction -- so merely building it up front still leaves the threads to appear later,
    # mid-run, when the knob rises and more concurrent tasks are submitted. That is the very
    # event being eliminated. Force all _wide threads into existence now, while nothing is in
    # flight, by submitting one trivial task per worker and waiting for all of them.
    try:
        _barrier = threading.Barrier(_wide, timeout=30)
        _futs = [_DECODE_POOL.submit(_barrier.wait) for _ in range(_wide)]
        for _f in _futs:
            _f.result(timeout=30)
    except Exception:
        pass    # priming is best-effort; a failure here must never block startup
    return _DECODE_POOL
# Pools retired by a live decode_threads resize. They are shut down non-blockingly at the swap and
# JOINED at teardown (_shutdown_decode_pool), so the read path never waits on a thread join.
_RETIRED_POOLS = []
_TWOSTAGE_WARNED = False

# ---- PROCESS-WIDE HANG INSTRUMENT (LIVEFLOW-STATUS.md sec 43.7) --------------------------------
# The apply-window watchdog in _apply_knobs was scoped to the APPLY, and the evidence says the hang
# is NOT there: the sec 43.5/43.6 runs logged "APPLY_SYNC: worst-rank apply latency 1.4 ms" -- the
# apply finished and the barrier passed -- and then froze one step later, inside the training loop.
# That fits the decode pool's LAZY resize: _apply_knobs only sets STORMER_DECODE_THREADS, and
# _decode_pool() does the actual swap (including old.shutdown(wait=True)) on the NEXT read, inside
# _read_two_stage. Scoping an instrument to where I ASSUMED the bug was is how two runs produced no
# stack at all.
#
# Also note NO NCCL watchdog fired in either run even past its 600 s timeout, which means ALL ranks
# were stuck, not a subset -- a genuine deadlock, not a straggler. So the instrument must be
# process-wide and must not depend on any other rank making progress.
#
# SIGUSR1 costs nothing and allows an on-demand dump of every thread from outside.
# STORMER_STACK_EVERY_SEC>0 additionally arms a periodic dump for diagnostic runs.
# FATAL-SIGNAL TRACEBACK (LIVEFLOW-STATUS sec 80). `register(SIGUSR1)` and
# `dump_traceback_later` only cover on-demand and periodic dumps; NEITHER installs a handler for
# SIGSEGV. Two ranks died with "exited (Segmentation fault)" and produced no Python frame at all,
# which is why the crashes were uninformative. enable() installs handlers for SIGSEGV/SIGBUS/
# SIGFPE/SIGILL/SIGABRT and dumps every thread's Python stack to stderr before the process dies.
# Forked DataLoader workers inherit it, so a crash inside a worker is covered too.
faulthandler.enable(all_threads=True)
faulthandler.register(signal.SIGUSR1, all_threads=True)
_STACK_EVERY = float(os.environ.get("STORMER_STACK_EVERY_SEC", "0") or 0)
if _STACK_EVERY > 0:
    # repeat=True so successive dumps show whether the stack is stuck or merely slow.
    # NOTE: a blind periodic dump also fires during HEALTHY operation. A run with this at 120 s
    # froze at step 3 while two runs without it reached step 61, so 16 ranks periodically writing
    # every thread's stack into one aggregated stderr is suspected of perturbing the job itself.
    # Prefer the progress watchdog below, which only fires when progress actually stops.
    faulthandler.dump_traceback_later(_STACK_EVERY, repeat=True, exit=False)

# ---- PROGRESS WATCHDOG ------------------------------------------------------------------------
# Dumps stacks ONLY when the training loop stops advancing, so it costs nothing while healthy and
# cannot fire during warmup the way a periodic timer does. This is the instrument the sec 43.7
# analysis called for: process-wide (the hang is in the reader, not the apply) and independent of
# any other rank's progress (all ranks were stuck, so nothing external will notice).
_LAST_PROGRESS_T = time.time()


def _note_progress():
    global _LAST_PROGRESS_T
    _LAST_PROGRESS_T = time.time()


# ---- APPLICATION FLIGHT RECORDER --------------------------------------------------------------
# Why this exists (LIVEFLOW-STATUS.md sec 46): diagnosing a DISTRIBUTED stall needs a per-rank
# TIMELINE, and neither instrument we had provides one.
#   * NCCL's flight recorder logs collectives only -- it is silent when the stall is in application
#     code, and silent entirely when work.wait() already returned.
#   * A Python stack shows WHERE a rank is parked, not WHEN it diverged from the others. Four
#     debugging cycles went to the comm hook because that is where the stack pointed; it was the
#     symptom.
# A ring of (monotonic time, phase, step, thread) per rank, dumped on stall, reconstructs who was
# in which phase at which instant -- which is the only thing that distinguishes "this rank fell
# behind" from "this rank never started".
#
# Two deliberate properties:
#   NO LOCK. A lock here would add contention to the exact concurrency being investigated; a torn
#   entry costs one bad row and is worth far more than a perturbed subject.
#   PER-RANK FILE, not stderr. tqdm writes carriage returns to stderr and flux interleaves 16 ranks
#   into one stream, which shredded every dump taken so far.
_FLIGHT_SLOTS = int(os.environ.get("STORMER_FLIGHT_SLOTS", "8192") or 8192)
_FLIGHT_ON = os.environ.get("STORMER_FLIGHT", "0") == "1"
_FLIGHT = [None] * _FLIGHT_SLOTS
_FLIGHT_I = 0


def _fp(tag: str, step: int = -1):
    """Record one phase marker. Deliberately branch-light: this sits on the training critical path."""
    if not _FLIGHT_ON:
        return
    global _FLIGHT_I
    i = _FLIGHT_I
    _FLIGHT_I = i + 1
    _FLIGHT[i % _FLIGHT_SLOTS] = (time.monotonic(), tag, step, threading.get_ident())


def _flight_dump(reason: str):
    """Dump the ring + every thread's stack to a PER-RANK file."""
    try:
        d = os.environ.get("STORMER_FLIGHT_DIR",
                           f"/p/lustre5/{os.environ.get('USER','izzet')}/liveflow-validation/flight")
        os.makedirs(d, exist_ok=True)
        rank = os.environ.get("PMI_RANK") or os.environ.get("FLUX_TASK_RANK") or "x"
        path = os.path.join(d, f"{os.uname()[1]}_r{rank}_p{os.getpid()}.flight")
        with open(path, "a") as f:
            f.write(f"\n===== FLIGHT DUMP reason={reason} t={time.monotonic():.3f} "
                    f"decode_limit={_DECODE_LIMIT.value} =====\n")
            n = min(_FLIGHT_I, _FLIGHT_SLOTS)
            start = _FLIGHT_I - n
            for k in range(start, _FLIGHT_I):
                e = _FLIGHT[k % _FLIGHT_SLOTS]
                if e:
                    f.write(f"{e[0]:.6f} step={e[2]:<5} tid={e[3]} {e[1]}\n")
            f.write("--- stacks ---\n")
            f.flush()
            faulthandler.dump_traceback(file=f, all_threads=True)
            f.flush()
    except Exception:
        pass    # a recorder must never be able to break the run it is observing


_STALL_SEC = float(os.environ.get("STORMER_STALL_DUMP_SEC", "0") or 0)
_CDUMP_ARMED = False
# C-LEVEL BACKSTOP. _stall_watch below is a PYTHON thread and therefore needs the GIL to run at
# all. In the D20 capture only 4 of 16 ranks produced a dump: the ranks that matter -- the ones
# actually wedged -- could not run their watchdog, which means something in those processes holds
# the GIL continuously. (cuda.synchronize RELEASES the GIL, so they are not parked there.)
# faulthandler's timer is a C thread and dumps WITHOUT the GIL, so it sees exactly the ranks the
# Python watchdog cannot. Writing to the per-rank file, not stderr, so tqdm and 16-way interleaving
# cannot shred it -- and at a long interval, unlike the 120 s periodic dump that perturbed D10.
if _STALL_SEC > 0 and os.environ.get("STORMER_FLIGHT", "0") == "1":
    try:
        _d = os.environ.get("STORMER_FLIGHT_DIR",
                            f"/p/lustre5/{os.environ.get('USER','izzet')}/liveflow-validation/flight")
        os.makedirs(_d, exist_ok=True)
        _rk = os.environ.get("PMI_RANK") or os.environ.get("FLUX_TASK_RANK") or "x"
        _CFILE = open(os.path.join(_d, f"{os.uname()[1]}_r{_rk}_p{os.getpid()}.cdump"), "a")
        faulthandler.dump_traceback_later(_STALL_SEC * 2, repeat=True, exit=False, file=_CFILE)
        _CDUMP_ARMED = True
    except Exception:
        pass
if _STALL_SEC > 0:
    def _stall_watch():
        import sys as _sys
        dumped = 0
        while dumped < 3:          # bounded: 3 dumps is enough to tell stuck from slow
            time.sleep(15)
            idle = time.time() - _LAST_PROGRESS_T
            if idle > _STALL_SEC:
                _sys.stderr.write(f"STALL_DUMP: no step progress for {idle:.0f}s "
                                  f"(dump {dumped + 1}/3)\n")
                _sys.stderr.flush()
                # Ring + stacks to a PER-RANK FILE. The stderr copy above is only a breadcrumb;
                # stderr is unusable for the actual payload (tqdm + 16 interleaved ranks).
                _flight_dump(f"stall_{idle:.0f}s")
                # NATIVE stack, not just Python frames.
                #
                # WHY THIS EXISTS. Every instrument so far stopped at the Python frame: we learn
                # that a rank is inside `read_direct_chunk` and nothing about WHY. That call
                # descends HDF5 -> POSIX -> DFTracer's interception -> Mofka producer ->
                # Mercury/Argobots, and the fault has to be somewhere in there -- the stall is 8+
                # minutes on ONE read, and the arms that stall are exactly the ones running the
                # streaming pipeline against the node-local bedrock. Guessing at that boundary is
                # what produced (and then retracted) three wrong root causes.
                #
                # eu-stack is run from a CHILD process attaching to us: a thread cannot reliably
                # ptrace its own process, but a forked child can (ptrace_scope=0 here). h5py
                # releases the GIL during I/O, which is why this watchdog thread gets scheduled at
                # all while the main thread is blocked -- confirmed by diag1 producing 41 stall
                # breadcrumbs from a wedged rank.
                if os.environ.get("STORMER_NATIVE_STACK", "1") == "1":
                    try:
                        import subprocess as _sp
                        _tool = None
                        for _c in ("eu-stack", "pstack"):
                            if _sp.run(["which", _c], capture_output=True).returncode == 0:
                                _tool = _c
                                break
                        if _tool:
                            # PROFILE THE WORKERS TOO, not just this process.
                            #
                            # The stall we are chasing lives in a DataLoader WORKER: an unmatched
                            # [DFT_PUSH_ENTER] shows a forked child blocked inside a Mofka push
                            # while its siblings push in 0 ms. Dumping only os.getpid() shows the
                            # PARENT waiting for that worker's batch, which is the symptom, and
                            # tells us nothing about why the worker's producer queue is not
                            # draining. Enumerate our children and dump each of them as well.
                            #
                            # Observation only -- no behaviour change. The specific question this
                            # has to answer is whether ActiveProducerBatchQueue::loop (the batch
                            # sender) exists and is running in the stuck child, or whether it was
                            # never scheduled after fork.
                            _pids = [os.getpid()]
                            try:
                                _kids = os.path.join("/proc", str(os.getpid()),
                                                     "task", str(os.getpid()), "children")
                                with open(_kids) as _kf:
                                    _pids += [int(x) for x in _kf.read().split()]
                            except Exception:
                                pass
                            _nd = os.environ.get("STORMER_FLIGHT_DIR", "/tmp")
                            _nrk = (os.environ.get("PMI_RANK")
                                    or os.environ.get("FLUX_TASK_RANK") or "x")
                            _out = os.path.join(
                                _nd, f"{os.uname()[1]}_r{_nrk}_p{os.getpid()}.native")
                            with open(_out, "a") as _fh:
                                _fh.write(f"\n######## idle {idle:.0f}s dump {dumped + 1}: "
                                          f"{len(_pids)} process(es) ########\n")
                                for _tp in _pids:
                                    _kind = "SELF" if _tp == os.getpid() else "WORKER"
                                    try:
                                        _r = _sp.run(
                                            [_tool, "-p", str(_tp)] if _tool == "eu-stack"
                                            else [_tool, str(_tp)],
                                            capture_output=True, text=True, timeout=120)
                                        _fh.write(f"\n==== {_kind} pid={_tp} {_tool} ====\n"
                                                  f"{_r.stdout}\n{_r.stderr}\n")
                                    except Exception as _pe:
                                        _fh.write(f"\n==== {_kind} pid={_tp} FAILED: "
                                                  f"{type(_pe).__name__}: {_pe} ====\n")
                    except Exception as _e:
                        _sys.stderr.write(f"NATIVE_STACK failed: {type(_e).__name__}: {_e}\n")
                dumped += 1
                time.sleep(_STALL_SEC)   # space the dumps so they show movement, not one instant
    threading.Thread(target=_stall_watch, daemon=True, name="stallwatch").start()


def _decode_pool():
    """Persistent per-process decode pool, sized ONCE; decode_threads is a CONCURRENCY LIMIT.

    WHY NOT RESIZE THE POOL (LIVEFLOW-STATUS.md sec 43.8): creating/destroying pool threads while
    GPU work is in flight wedged the job every time. The stall watchdog caught all 16 ranks blocked
    at `torch.cuda.synchronize()` in the DDP comm hook (train.py `_comm_hook`), identical stacks at
    182 s and 377 s, with the decode threads themselves IDLE. `work.wait()` had already returned --
    which is why the NCCL watchdog never fired -- and the stream then never drained.

    Crucially the CAPABILITY is fine: decode_threads=8 set at STARTUP ran 120/120 steps and gave
    4.6x (sec 24). Only the mid-run TRANSITION breaks. So do not transition: allocate the widest
    pool once and let the knob move a semaphore instead. Changing decode_threads then never touches
    the thread set, which removes the trigger by construction.

    A fixed pool + semaphore delivers the same thing the 4.6x came from -- concurrent inflate --
    because the win is parallelism across the 69 per-variable chunks, not thread identity.
    """
    n = int(os.environ.get("STORMER_DECODE_THREADS", "1") or 1)
    _DECODE_LIMIT.value = max(1, n)
    # ALWAYS pooled, even at limit 1. Previously n<=1 ran inline in the calling thread and n>1 ran
    # on the pool, so crossing 1->2 mid-run switched CODE PATH, not just a parameter -- a second
    # discontinuity at exactly the transition that hangs. One path at every knob value removes it.
    return _ensure_decode_pool()


def _shutdown_decode_pool():
    """Drain the decode pool BEFORE the interpreter (and DFTracer's C extension) finalize.
    Registered with atexit so every DataLoader worker tears down cleanly."""
    global _DECODE_POOL, _DECODE_POOL_N
    # Join the pools retired by a live resize FIRST. Blocking is safe here (teardown, no training
    # loop to stall) and it preserves the invariant the inline wait=True was protecting: no pool
    # thread is still alive when the interpreter and DFTracer's C extension finalize.
    while _RETIRED_POOLS:
        try:
            _RETIRED_POOLS.pop().shutdown(wait=True)
        except Exception:
            pass
    if _DECODE_POOL is not None:
        try:
            _DECODE_POOL.shutdown(wait=True)
        except Exception:
            pass
        _DECODE_POOL, _DECODE_POOL_N = None, 0


atexit.register(_shutdown_decode_pool)


warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger(__name__)

# MPIUtils.rank() = int(os.environ["FLUX_TASK_RANK"])
# MPI_SIZE = int(os.environ["FLUX_JOB_SIZE"])
# MPI_LOCAL_RANK = int(os.environ["FLUX_TASK_LOCAL_ID"])
# MPI_NODES = int(os.environ["FLUX_JOB_NNODES"])
GPUS = os.environ.get("ROCR_VISIBLE_DEVICES", "")
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 # since we only see 1 gpu, so GPU that is observed by the process will be 0


def get_ddp_reduce_dtype(value: str | None):
    if value is None:
        return None
    value = value.strip().lower()
    if value in ("", "none", "off", "0"):
        return None
    if value in ("16", "fp16", "float16"):
        return torch.float16
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported STORMER_DDP_REDUCE_DTYPE={value}")

VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential_50",
    "geopotential_100",
    "geopotential_150",
    "geopotential_200",
    "geopotential_250",
    "geopotential_300",
    "geopotential_400",
    "geopotential_500",
    "geopotential_600",
    "geopotential_700",
    "geopotential_850",
    "geopotential_925",
    "geopotential_1000",
    "u_component_of_wind_50",
    "u_component_of_wind_100",
    "u_component_of_wind_150",
    "u_component_of_wind_200",
    "u_component_of_wind_250",
    "u_component_of_wind_300",
    "u_component_of_wind_400",
    "u_component_of_wind_500",
    "u_component_of_wind_600",
    "u_component_of_wind_700",
    "u_component_of_wind_850",
    "u_component_of_wind_925",
    "u_component_of_wind_1000",
    "v_component_of_wind_50",
    "v_component_of_wind_100",
    "v_component_of_wind_150",
    "v_component_of_wind_200",
    "v_component_of_wind_250",
    "v_component_of_wind_300",
    "v_component_of_wind_400",
    "v_component_of_wind_500",
    "v_component_of_wind_600",
    "v_component_of_wind_700",
    "v_component_of_wind_850",
    "v_component_of_wind_925",
    "v_component_of_wind_1000",
    "temperature_50",
    "temperature_100",
    "temperature_150",
    "temperature_200",
    "temperature_250",
    "temperature_300",
    "temperature_400",
    "temperature_500",
    "temperature_600",
    "temperature_700",
    "temperature_850",
    "temperature_925",
    "temperature_1000",
    "specific_humidity_50",
    "specific_humidity_100",
    "specific_humidity_150",
    "specific_humidity_200",
    "specific_humidity_250",
    "specific_humidity_300",
    "specific_humidity_400",
    "specific_humidity_500",
    "specific_humidity_600",
    "specific_humidity_700",
    "specific_humidity_850",
    "specific_humidity_925",
    "specific_humidity_1000",
]

def log0(*args, level: str = "info", **kwargs):
    if MPIUtils.rank() == 0:
        getattr(log, level)(*args, **kwargs)

class ERA5Dataset(Dataset):
    def __init__(
        self,
        args,
        root_dir: str,
        variables: list[str],
        inp_transform: torch.nn.Module,
        out_transform_dict: dict[int, torch.nn.Module],
        list_intervals: list[int] = [6, 12, 24],
        data_freq: int = 6,
        year_list: list[int] | None = None,
    ):
        self.perf_tracer: PerfTracer | None = None
        self.args = args
        self.root_dir = root_dir
        self.variables = variables
        self.inp_transform = inp_transform
        self.out_transform_dict = out_transform_dict
        self.steps = 1
        self.list_intervals = list_intervals
        self.data_freq = data_freq
        self.year_list = year_list
        self.interval_protocol = getattr(args, "interval_protocol", "legacy")
        if self.interval_protocol not in ("legacy", INTERVAL_SELECTOR_VERSION):
            raise ValueError(f"unsupported StorMer interval protocol: {self.interval_protocol}")
        self.interval_seed = int(getattr(args, "seed", 42))
        self.interval_split = "val" if os.path.basename(os.path.normpath(root_dir)) == "val" else "train"
        self.interval_epoch = 0
        self.year_idx_map: dict | None = None
        self.image_shape = self.args.in_img_size
        self.downsample = os.environ.get("STORMER_DOWNSAMPLE", "1") == "1"
        self.latitude_metadata_dir = getattr(self.args, "latitude_metadata_dir", None)
        if year_list is not None:
            self.year_idx_map = {year: i for i, year in enumerate(year_list)}

        file_paths = glob(os.path.join(root_dir, "*.h5"))
        file_paths = sorted(file_paths)
        # Opt-in (cached-regime study): keep only the first N sorted files of the TRAIN split (a
        # time-contiguous prefix; the last steps*max(interval)//freq of them stay target-only).
        _subset_n = int(os.environ.get("STORMER_TRAIN_SUBSET_FILES", "0") or 0)
        if _subset_n > 0 and os.path.basename(os.path.normpath(root_dir)) == "train":
            file_paths = file_paths[:_subset_n]

        final_file_paths: list[str] = []
        if self.year_list is not None:
            year_str = [str(year) for year in sorted(self.year_list)]
            for f in file_paths:
                for year in year_str:
                    if year in str(f):
                        final_file_paths.append(f)
        else:
            final_file_paths = file_paths

        self.inp_file_paths = final_file_paths[
            : -(self.steps * max(list_intervals) // data_freq)
        ]  # the last few points do not have ground-truth
        if self.interval_protocol == INTERVAL_SELECTOR_VERSION:
            index_manifest = getattr(args, "interval_index_manifest", None)
            if not index_manifest:
                raise ValueError("keyed StorMer intervals require --interval-index-manifest")
            verify_index(
                self.inp_file_paths, root_dir, index_manifest, self.interval_split,
                intervals=tuple(self.list_intervals), data_freq=self.data_freq,
                selector_version=self.interval_protocol,
            )

    def get_out_path(
        self,
        root_dir: str,
        year: int,
        inp_file_idx: int,
        steps: int,
        year_list: list[int] | None = None,
        year_idx_map: dict[int, int] | None = None,
    ):
        # year: current year
        # inp_file_idx: file index of the input in the current year
        # steps: number of steps forward
        out_file_idx = inp_file_idx + steps
        out_path = os.path.join(root_dir, f"{year}_{out_file_idx:04}.h5")
        if not os.path.exists(out_path):
            max_step_forward = 0
            for i in range(1, steps):
                out_file_idx = inp_file_idx + i
                out_path = os.path.join(root_dir, f"{year}_{out_file_idx:04}.h5")
                if os.path.exists(out_path):
                    max_step_forward = i
            remaining_steps = steps - max_step_forward
            if year_list is None or year_idx_map is None:
                next_year = year + 1
            else:
                next_year = year_list[year_idx_map[year] + 1]
            out_path = os.path.join(root_dir, f"{next_year}_{remaining_steps - 1:04}.h5")
            if not os.path.exists(out_path):
                # Both the same-year forward guesses and the next-year fallback are absent: this
                # only happens against a sparse/non-contiguous file set (e.g. a randomly-sampled
                # subset directory), where the dense-per-year assumption above does not hold. Fail
                # loudly and name the missing file, instead of handing h5py.File() a path that
                # does not exist and letting a bare FileNotFoundError surface deep in the read path.
                raise FileNotFoundError(
                    f"get_out_path: no forward-reference target exists for {year}_{inp_file_idx:04} "
                    f"(steps={steps}) under {root_dir}; last candidate tried: {out_path}. "
                    f"The file set under root_dir is not densely populated per year (a sampled "
                    f"subset directory, not the full corpus) -- use a time-contiguous subset."
                )
        return out_path

    def _read_two_stage(self, path: str, variables: list[str]):
        """STAGE A (storage) then STAGE B (decode). Returns None if the data is not
        chunked+compressed, so the caller falls back to the normal inline read."""
        try:
            with h5py.File(path, "r") as f:
                grp = f["input"]
                probe = grp[variables[0]]
                if probe.chunks is None or probe.compression is None:
                    return None                      # uncompressed data -> no decode domain
                # ---- STAGE A: raw compressed bytes. Pure I/O, no inflate. Sequential because
                # h5py is not thread-safe; it is also cheap (~0.7 ms/chunk vs ~16 ms to inflate).
                # (no app region here: DFTracer's POSIX interception already attributes these
                # reads to the reader_posix layer, which is the STORAGE signal the preset uses)
                _fp("readA.start")
                _t_a = time.time()
                raw = []
                for var in variables:
                    d = grp[var]
                    _mask, blob = d.id.read_direct_chunk((0,) * len(d.shape))
                    raw.append((blob, d.shape, d.dtype))
                # Reported to the parent through shared memory; only the ratio against stage B is
                # used, to split the consumer's exposure between storage and decode.
                with _STAGE_A_MS.get_lock():
                    _STAGE_A_MS.value += (time.time() - _t_a) * 1000.0
            # ---- STAGE B: inflate. Pure CPU, thread-safe, parallel across per-variable chunks.
            # This is the DECODE domain; its width is the `decode_threads` knob.
            def _inflate(item):
                blob, shape, dtype = item
                return np.frombuffer(zlib.decompress(blob), dtype=dtype).reshape(shape)

            _fp("readA.end/decode.start")
            _t_b = time.time()
            with dft_ai.data.decode:   # DECODE domain signal
                pool = _decode_pool()
                if pool is None:
                    arrs = [_inflate(i) for i in raw]
                else:
                    # The pool is sized WIDE and fixed (see _decode_pool); decode_threads is
                    # enforced here as a concurrency limit. Chunking the submissions to the current
                    # limit keeps at most _DECODE_LIMIT inflates running at once WITHOUT touching
                    # the thread set, so raising or lowering the knob mid-run costs an integer
                    # assignment instead of thread create/destroy against a live HIP stream.
                    _lim = max(1, _DECODE_LIMIT.value)
                    if _lim >= len(raw):
                        arrs = list(pool.map(_inflate, raw))
                    else:
                        arrs = []
                        for _i in range(0, len(raw), _lim):
                            arrs.extend(pool.map(_inflate, raw[_i:_i + _lim]))
            with _STAGE_B_MS.get_lock():
                _STAGE_B_MS.value += (time.time() - _t_b) * 1000.0
            _fp("decode.end")
            return [a.reshape(1, *a.shape[-2:]) for a in arrs]
        except Exception as e:
            # Do NOT call MPIUtils.rank() here: it asserts when MPI is not initialised, so the
            # handler would raise while handling and mask the real error (and break every
            # standalone/offline use of this reader).
            global _TWOSTAGE_WARNED
            if not _TWOSTAGE_WARNED:
                _TWOSTAGE_WARNED = True
                log.warning("two-stage read failed on %s (%s: %s); falling back to inline",
                            os.path.basename(path), type(e).__name__, e)
            return None

    @_mot.producer_timed("fused_hdf5_preprocess_unresolved")
    def get_data_given_path(self, path: str, variables: list[str]):
        # ------------------------------------------------------------------ TWO-STAGE (DECODE DOMAIN)
        # STORMER_TWOSTAGE=1 with a gzip+CHUNKED dataset splits the read into two separately
        # tunable stages, giving StorMer a third domain (LIVEFLOW-STATUS.md sec 36):
        #   STAGE A (storage): read_direct_chunk() -> RAW COMPRESSED bytes, no inflate.
        #   STAGE B (decode) : zlib.decompress, parallel across the per-variable chunks.
        # Parallelism lives in stage B because that is where the cost is (measured: ~0.7 ms to read
        # a chunk vs ~16 ms to inflate it) and because zlib-on-bytes is thread-safe while h5py is
        # NOT -- so stage A stays sequential on a single file handle.
        # Falls back to the inline path for uncompressed/unchunked data, so the default is unchanged.
        x = self._read_two_stage(path, variables) if _TWOSTAGE else None
        _fp("reader.return")   # our code done; what follows is collate/pin/loader handoff
        if x is None:                       # not enabled, or data not chunked/compressed
            # LiveFlow layout knob (liveflow_layout.py): defaults (transfer_mb=0) are the same
            # per-variable h5py reads as before; set_layout()/LIVEFLOW_LAYOUT_* change only HOW the
            # identical bytes are fetched (read granularity / intra-sample read threads).
            try:
                _arrs = _layout_read_variables(path, variables, "input")
            except OSError as exc:
                raise OSError(f"could not open StorMer sample {path}: {exc}") from exc
            x = [d.reshape(1, *d.shape[-2:]) for d in _arrs]
            del _arrs

        # STORMER_DROP_CACHE=1: evict this file from the OS page cache after reading so every
        # read stays COLD and the input pole is steady (does not self-relieve via cache warming).
        # This is what lets the optimizer's trend gate see a persistent bottleneck and ACTUATE
        # (add workers) — the actuating regime for the closed-loop demo. No effect when unset.
        if os.environ.get("STORMER_DROP_CACHE", "0") == "1":
            try:
                _fd = os.open(path, os.O_RDONLY)
                os.posix_fadvise(_fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(_fd)
            except OSError:
                pass

        out = torch.from_numpy(np.concatenate(x))  # (V, H_file, W_file)
        if self.latitude_metadata_dir and tuple(out.shape[-2:]) != (721, 1440):
            raise ValueError(
                "reviewed latitude metadata requires every input/target field to be native "
                f"721x1440 before resizing; got {tuple(out.shape[-2:])} in {path}"
            )
        if self.downsample and out.shape[-2:] != tuple(self.image_shape):
            out = torch.nn.functional.interpolate(
                out.unsqueeze(0),  # (1, V, H_file, W_file)
                size=self.image_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # (V, H_target, W_target)
        return out
    
    def __len__(self):
        return len(self.inp_file_paths)

    def set_epoch(self, epoch: int):
        if int(epoch) < 0:
            raise ValueError("StorMer interval epoch must be nonnegative")
        # Hold validation's target schedule fixed across evaluation epochs.
        self.interval_epoch = 0 if self.interval_split == "val" else int(epoch)

    @dft_ai.data.item
    def __getitem__(self, index: int):

        # 1. Get input data
        inp_path = self.inp_file_paths[index]
        inp_data = self.get_data_given_path(inp_path, self.variables)

        if self.interval_protocol == INTERVAL_SELECTOR_VERSION:
            chosen_interval = select_interval(
                tuple(self.list_intervals), experiment_seed=self.interval_seed,
                epoch=self.interval_epoch, dataset_index=index,
                split=self.interval_split,
            )
        else:
            chosen_interval = np.random.choice(self.list_intervals)
        year, inp_file_idx = os.path.basename(inp_path).split(".")[0].split("_")
        year, inp_file_idx = int(year), int(inp_file_idx)

        # 2. Get input data
        out_path = self.get_out_path(
            self.root_dir,
            year,
            inp_file_idx,
            steps=chosen_interval // self.data_freq,
            year_list=self.year_list,
            year_idx_map=self.year_idx_map,
        )
        out = self.get_data_given_path(out_path, self.variables)

        # 3. Preprocess
        with dft_ai.data.preprocess, _mot.producer_span("decode_preprocess"):
            diff = out - inp_data
            interval = torch.Tensor([chosen_interval]) / 10.0
            inp_transform = self.inp_transform(inp_data).unsqueeze(0)
            diff_transform = self.out_transform_dict[chosen_interval](diff).unsqueeze(0)

        sample = (inp_transform, diff_transform, interval)
        if os.environ.get("LIVEFLOW_SAMPLE_LOG", "0") == "1":
            return sample + (int(index),)
        return sample
    
collate_profiler = dft_ai.data.preprocess.derive(name="collate")

@collate_profiler
def collate_fn_train(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])  # B, V, H, W
    out = torch.stack([batch[i][1] for i in range(len(batch))])  # B, T, V, H, W
    sample_log = os.environ.get("LIVEFLOW_SAMPLE_LOG", "0") == "1"
    interval = torch.stack([batch[i][2 if sample_log else 4] for i in range(len(batch))])  # B, T
    if sample_log:
        return inp, out, interval, torch.tensor([item[3] for item in batch], dtype=torch.int64)
    return inp, out, interval  # type: ignore[return-value]

class DataModule:
    @dft_ai.data.init
    def __init__(
        self,
        args,
        root_dir: str,
        variables: list[str],
        intervals: list[int],
    ):
        self.args = args
        self.batch_size = self.args.batch_size
        self.data_freq = self.args.data_freq

        self.root_dir = root_dir
        self.variables = variables
        self.intervals = intervals

        # normalization for input
        normalize_mean = dict(np.load(os.path.join(self.root_dir, "normalize_mean.npz")))
        normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
        normalize_std = dict(np.load(os.path.join(self.root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)

        out_transforms = {}
        for intv in intervals:
            normalize_diff_std = dict(
                np.load(os.path.join(root_dir, f"normalize_diff_std_{intv}.npz"))
            )
            normalize_diff_std = np.concatenate(
                [normalize_diff_std[v] for v in variables], axis=0
            )
            out_transforms[intv] = transforms.Normalize(
                np.zeros_like(normalize_diff_std), normalize_diff_std
            )

        self.transforms = transforms.Normalize(normalize_mean, normalize_std)
        self.out_transforms = out_transforms

    def get_lat_lon(self):
        lat = load_latitude(
            self.root_dir,
            getattr(self.args, "latitude_metadata_dir", None),
            self.args.in_img_size,
            os.environ.get("STORMER_DOWNSAMPLE", "1") == "1",
        )
        lon = np.load(os.path.join(self.root_dir, "lon.npy"))
        return lat, lon

    def get_transforms(self):
        return self.transforms, self.out_transforms

    def val_dataloader(self, max_samples=50):
        """Create a small validation DataLoader from the val/ split."""
        val_dir = os.path.join(self.root_dir, "val")
        if not os.path.isdir(val_dir):
            return None
        dataset = ERA5Dataset(
            args=self.args,
            root_dir=val_dir,
            variables=self.variables,
            inp_transform=self.transforms,
            out_transform_dict=self.out_transforms,
            list_intervals=self.intervals,
            data_freq=self.data_freq,
        )
        # Use a small fixed subset for fast validation
        if len(dataset) > max_samples:
            indices = list(range(0, len(dataset), len(dataset) // max_samples))[:max_samples]
            dataset = torch.utils.data.Subset(dataset, indices)
        batch_size = self.batch_size
        collate_fn = collate_fn_train
        if self.args.disable_collation and batch_size == 1:
            batch_size = None
            collate_fn = None
        val_kwargs = {}
        if getattr(self.args, "interval_protocol", "legacy") == INTERVAL_SELECTOR_VERSION:
            val_kwargs["generator"] = make_keyed_dataloader_generator(
                INTERVAL_SELECTOR_VERSION, self.args.seed, 0)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0,  # keep it simple for val
            pin_memory=False, collate_fn=collate_fn, **val_kwargs,
        )

    def dataloader(self, loader_config: dict[str, Any] | None = None, epoch: int = 0):
        if loader_config is None:
            loader_config = {
                "num_workers": self.args.num_workers,
                "pin_memory": self.args.pin_memory,
                "persistent_workers": self.args.persistent_workers,
                "prefetch_factor": self.args.prefetch_factor if self.args.num_workers > 0 else None,
            }

        dataset = ERA5Dataset(
            args=self.args,
            root_dir=os.path.join(self.root_dir, "train"),
            variables=self.variables,
            inp_transform=self.transforms,
            out_transform_dict=self.out_transforms,
            list_intervals=self.intervals,
            data_freq=self.data_freq,
            year_list=None, # use all data
        )
        keyed_generator = None
        if getattr(self.args, "interval_protocol", "legacy") == INTERVAL_SELECTOR_VERSION:
            dataset.set_epoch(epoch)
            keyed_generator = make_keyed_dataloader_generator(
                INTERVAL_SELECTOR_VERSION, self.args.seed, epoch)

        batch_size = self.batch_size
        collate_fn = collate_fn_train

        if self.args.disable_collation:
            if batch_size > 1:
                log0(f"Cannot disable collation since batch_size is {batch_size}", mode="warning")

            if batch_size == 1:
                batch_size = None
                collate_fn = None
                log0("Disabling collation")

        # ONE loader class for every arm. The controller needs LiveDataLoader (it can resize in
        # place); Original / MaxResource / static arms run the SAME class with the controller off
        # (STORMER_LIVE_LOADER=1), so a loop-vs-static comparison never confounds the controller
        # with the loader implementation.
        use_live = (os.environ.get("DFOPTIMIZER_ENABLE", "0") == "1"
                    or os.environ.get("STORMER_LIVE_LOADER", "0") == "1")
        worker_posix_init = _worker_posix_roster_init(self.args, loader_config)

        # With Ulysses sequence parallelism, all sp ranks in a block consume the SAME sample
        # (sequence-split), so shard the sampler over data-parallel groups (world/sp) and give
        # every rank in a block the same shard index. sp=1 -> unchanged (rank/size).
        _sp = int(getattr(self.args, "ulysses_sp", 1) or 1)
        _smp_replicas = max(1, MPIUtils.size() // _sp)
        _smp_rank = MPIUtils.rank() // _sp

        if use_live:
            from live_sampler import LiveSampler
            from live_data_loader import LiveDataLoader

            sampler = LiveSampler(
                dataset,
                num_replicas=_smp_replicas,
                rank=_smp_rank,
                shuffle=True,
                seed=self.args.seed,
            )
            loader = LiveDataLoader(
                dataset,
                sampler,
                batch_size=batch_size,
                num_workers=loader_config["num_workers"],
                prefetch_factor=loader_config.get("prefetch_factor") or 2,
                pin_memory=loader_config["pin_memory"],
                persistent_workers=loader_config["persistent_workers"],
                collate_fn=collate_fn,
                generator_factory=(
                    lambda loader_epoch: make_keyed_dataloader_generator(
                        INTERVAL_SELECTOR_VERSION, self.args.seed, loader_epoch)
                ) if keyed_generator is not None else None,
            )
            return loader, sampler

        if MPIUtils.size() >= 1:
            sampler = DistributedSampler(
                dataset=dataset,
                rank=_smp_rank,
                num_replicas=_smp_replicas,
                seed=self.args.seed,
                shuffle=True,
                drop_last=False,
            )
        else:
            sampler = None

        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "drop_last": False,
            "sampler": sampler,
            "num_workers": loader_config["num_workers"],
            "pin_memory": loader_config["pin_memory"],
            "persistent_workers": loader_config["persistent_workers"],
            "prefetch_factor": loader_config["prefetch_factor"],
            "collate_fn": collate_fn,
        }
        if worker_posix_init is not None:
            # The gate is intentionally unavailable on LiveDataLoader, whose
            # worker replacement is part of its controller actuation contract.
            loader_kwargs["worker_init_fn"] = worker_posix_init
            # A preloaded rank's mapped interposer reaches ordinary workers by
            # fork inheritance.  Do not rely on a site-wide multiprocessing
            # default (or permit a caller's spawn/forkserver override) for the
            # architecture gate that validates that exact property.
            configured_start = _mp.get_start_method(allow_none=True)
            if configured_start not in (None, "fork"):
                raise RuntimeError(
                    "STORMER_WORKER_POSIX_GATE requires fork workers; "
                    f"global multiprocessing start method is {configured_start!r}"
                )
            loader_kwargs["multiprocessing_context"] = "fork"
        if keyed_generator is not None:
            loader_kwargs["generator"] = keyed_generator
        return (
            DataLoader(**loader_kwargs),
            sampler,
        )


class Trainer:
    def __init__(
        self,
        model: Stormer,
        datamodule: DataModule,
        args,
    ):
        self.args = args
        self.model = model
        model_id: str = self.model.__class__.__qualname__
        self.model_id = model_id.lower()

        self.datamodule = datamodule
        self.device = torch.device(f"{DEVICE_TYPE}:{DEVICE_ID}")

        self.optimizer = configure_optimizers(
            params=model.named_parameters(),
            lr=self.args.lr,
            beta_1=self.args.beta_1,
            beta_2=self.args.beta_2,
            weight_decay=self.args.weight_decay,
        )
        self.precision = self.args.precision
        self.scheduler = None

        self.epochs = self.args.epochs
        self.max_training_step = self.args.max_training_step
        self.weighted_loss = self.args.weighted_loss
        self.enable_progress_bar = self.args.enable_progress_bar
        self.variables = VARIABLES
        self.accumulate_grad_batches = self.args.accumulate_grad_batches
        # Origin for the gradient-accumulation phase. The sync decision below used raw
        # batch_idx, which is the PER-RANK loop counter: after a num_workers respawn a rank
        # can yield a different number of batches, batch_idx drifts, and then
        # `batch_idx % accum` differs across ranks -- some enter no_sync() while others run
        # the gradient all_reduce, and the job deadlocks (LIVEFLOW-STATUS sec 74/75: 3 ranks
        # past backward, 13 blocked inside it). Re-basing the phase on an origin that every
        # rank stamps at the MPI-synchronised apply window makes the phase agree by
        # construction. Harmless when accum == 1, which is why that case never wedged.
        self._accum_origin = 0

        self.interval_combinations = None
        self.set_lat_lon(*self.datamodule.get_lat_lon())
        self.set_transforms(*self.datamodule.get_transforms())
        self.list_train_intervals = self.datamodule.intervals

        self.sync_batchnorm = self.args.sync_batchnorm

        self.dtype = get_precision_dtype(precision=self.args.precision)

        self.scaler = None
        self._trajectory_retention = None
        self._last_observed_optimizer_update = False
        self._completed_updates = 0
        self.setup_model()

    @staticmethod
    def _trajectory_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _initialize_trajectory_retention(self) -> None:
        """Arm the reviewed, no-resume epoch-end retention protocol.

        This method intentionally runs only after scheduler construction.  The
        normal training path never calls it unless the explicit state directory
        argument is supplied.
        """
        state_dir_arg = self.args.trajectory_state_dir
        if state_dir_arg is None:
            return
        if any(value is None for value in (
            self.args.trajectory_update_points,
            self.args.trajectory_source_manifest,
        )):
            raise ValueError("trajectory retention requires state dir, update points, and source manifest")
        if MPIUtils.size() != 1 or MPIUtils.rank() != 0 or int(self.args.ulysses_sp) != 1:
            raise ValueError("trajectory state v1 requires world=1 and ulysses_sp=1")
        if str(self.args.precision) != "32" or int(self.accumulate_grad_batches) != 1:
            raise ValueError("trajectory state v1 requires precision=32 and accumulate_grad_batches=1")
        if (os.environ.get("STORMER_COMM_OPTIMIZE", "0") == "1"
                or os.environ.get("STORMER_TUNE_COMM", "0") == "1"):
            raise ValueError("trajectory state v1 forbids comm accumulation actuators")
        if os.environ.get("STORMER_PER_REP_REPORTING", "0") == "1":
            raise ValueError("trajectory state v1 forbids per-repetition timing reporting")
        if self.args.interval_protocol != INTERVAL_SELECTOR_VERSION:
            raise ValueError("trajectory state v1 requires the keyed interval protocol")

        points = tuple(self.args.trajectory_update_points)
        if not points or points[0] != 0 or any(
            isinstance(point, bool) or point < 0 for point in points
        ) or tuple(sorted(set(points))) != points:
            raise ValueError("trajectory update points must be strictly ordered nonnegative integers beginning with 0")

        source_manifest = Path(self.args.trajectory_source_manifest)
        config_path = Path(self.args.output_folder) / "config.json"
        latitude_manifest = Path(self.args.latitude_metadata_dir or "") / "manifest.json"
        index_manifest = Path(self.args.interval_index_manifest or "")
        required = (source_manifest, config_path, latitude_manifest, index_manifest)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("trajectory retention provenance missing: " + ", ".join(missing))

        state_dir = Path(state_dir_arg)
        validation_dir = state_dir.parent / "trajectory_validation_v1"
        if state_dir.exists() or validation_dir.exists():
            raise FileExistsError("trajectory state and sibling validation destinations must not already exist")
        state_dir.mkdir(parents=True)
        validation_dir.mkdir()

        from trajectory_state import SCHEMA_VERSION, capture_state, verify_retained_state, write_retained_state

        provenance = {
            "config_json": {"path": str(config_path), "sha256": self._trajectory_sha256(config_path)},
            "source_manifest": {"path": str(source_manifest), "sha256": self._trajectory_sha256(source_manifest)},
            "latitude_manifest": {"path": str(latitude_manifest), "sha256": self._trajectory_sha256(latitude_manifest)},
            "interval_index_manifest": {"path": str(index_manifest), "sha256": self._trajectory_sha256(index_manifest)},
        }
        index_payload = json.loads(index_manifest.read_text())
        validation_identity = {
            "interval_index_manifest": provenance["interval_index_manifest"],
            "held_out_split": index_payload.get("splits", {}).get("val"),
        }
        if validation_identity["held_out_split"] is None:
            raise ValueError("trajectory state v1 requires held-out val identity in the keyed index manifest")
        run_manifest = {
            "schema_version": SCHEMA_VERSION,
            "completed_update_points": list(points),
            "protocol": {
                "world_size": 1,
                "ulysses_sp": 1,
                "precision": "32",
                "accumulate_grad_batches": 1,
                "interval_protocol": INTERVAL_SELECTOR_VERSION,
                "resume": False,
            },
            "provenance": provenance,
            "validation_identity": validation_identity,
            "runtime": {"torch_version": torch.__version__},
        }
        (state_dir / "run-manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
        self._trajectory_retention = {
            "capture_state": capture_state,
            "write_retained_state": write_retained_state,
            "verify_retained_state": verify_retained_state,
            "state_dir": state_dir,
            "validation_dir": validation_dir,
            "points": points,
            "captured": set(),
            "validations": set(),
            "provenance": provenance,
            "validation_identity": validation_identity,
        }

    def _capture_trajectory_state(self, completed_update_id: int, *, epoch: int | None, kind: str) -> None:
        retention = self._trajectory_retention
        if retention is None or completed_update_id not in retention["points"]:
            return
        if int(self.accumulate_grad_batches) != 1:
            raise RuntimeError("trajectory state v1 observed a changed accumulation setting")
        if completed_update_id in retention["captured"]:
            raise RuntimeError(f"duplicate trajectory snapshot request for update {completed_update_id}")
        metadata = {
            "retention_kind": kind,
            "epoch": epoch,
            "observed_update": kind == "epoch_end",
            "loss_finite": True,
            "completed_update_id": completed_update_id,
            "provenance": retention["provenance"],
        }
        started = time.perf_counter_ns()
        log.info("TRAJECTORY_STATE marker=snapshot.begin update=%d kind=%s", completed_update_id, kind)
        snapshot = retention["capture_state"](
            self.net, self.optimizer, self.scheduler, self.scaler,
            completed_update_id=completed_update_id, metadata=metadata,
        )
        destination = retention["state_dir"] / f"update-{completed_update_id:012d}"
        retention["write_retained_state"](destination, snapshot)
        retention["captured"].add(completed_update_id)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        log.info("TRAJECTORY_STATE marker=snapshot.end update=%d kind=%s elapsed_ms=%.3f", completed_update_id, kind, elapsed_ms)

    def _record_trajectory_validation(self, completed_update_id: int, record: dict[str, Any]) -> None:
        retention = self._trajectory_retention
        if retention is None or completed_update_id not in retention["captured"]:
            return
        destination = retention["validation_dir"] / f"update-{completed_update_id:012d}.json"
        if destination.exists():
            raise FileExistsError(f"duplicate trajectory validation record: {destination}")
        snapshot_dir = retention["state_dir"] / f"update-{completed_update_id:012d}"
        snapshot_manifest = retention["verify_retained_state"](snapshot_dir)
        payload = {
            "schema_version": "stormer-trajectory-validation.v1",
            "completed_update_id": completed_update_id,
            "snapshot": {
                "path": str(snapshot_dir),
                "state_pt_sha256": snapshot_manifest["artifact"]["state_pt_sha256"],
                "manifest_sha256": self._trajectory_sha256(snapshot_dir / "manifest.json"),
                "tensor_component_sha256": snapshot_manifest["tensor_component_sha256"],
            },
            "validation": record,
            "validation_identity": retention["validation_identity"],
        }
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        retention["validations"].add(completed_update_id)
        if not record["complete"]:
            raise RuntimeError(f"trajectory validation incomplete at update {completed_update_id}: {record}")

    def _run_trajectory_validation(self, epoch: int, completed_update_id: int, *, initial: bool = False) -> dict[str, Any]:
        started = time.perf_counter_ns()
        log.info("TRAJECTORY_STATE marker=validation.begin update=%d initial=%d", completed_update_id, int(initial))
        try:
            record = self.validate_epoch(
                epoch,
                trajectory_update_id=completed_update_id,
                initial_trajectory_validation=initial,
            )
        except Exception:
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            log.info("TRAJECTORY_STATE marker=validation.end update=%d status=error elapsed_ms=%.3f", completed_update_id, elapsed_ms)
            raise
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        log.info("TRAJECTORY_STATE marker=validation.end update=%d status=%s elapsed_ms=%.3f", completed_update_id, "complete" if record.get("complete") else "incomplete", elapsed_ms)
        return record

    def _finish_trajectory_retention(self) -> None:
        retention = self._trajectory_retention
        if retention is None:
            return
        expected = set(retention["points"])
        if retention["captured"] != expected or retention["validations"] != expected:
            raise RuntimeError(
                "trajectory retention incomplete: "
                f"captured={sorted(retention['captured'])} validations={sorted(retention['validations'])} "
                f"expected={sorted(expected)}"
            )

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_transforms(self, inp_transform, diff_transform):
        self.inp_transform = inp_transform
        self.reverse_inp_transform = self.get_reverse_transform(inp_transform)

        self.diff_transform = diff_transform
        self.reverse_diff_transform = {
            k: self.get_reverse_transform(v) for k, v in diff_transform.items()
        }

    def get_reverse_transform(self, transform):
        mean, std = transform.mean, transform.std
        std_reverse = 1 / std
        mean_reverse = -mean * std_reverse
        return transforms.Normalize(mean_reverse, std_reverse)

    def setup_model(self):
        # config = Runtime.config()

        self.net = self.model
        if self.sync_batchnorm and MPIUtils.size() > 1:
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)

        if DEVICE_TYPE == "cuda":
            # torch.cuda.set_device(MPI_LOCAL_RANK)
            torch.cuda.set_device(DEVICE_ID)

        # log0(f"Moving model to device {self.device}")
        # self.net.to(MPI_LOCAL_RANK)
        self.net.to(device=self.device)

        # --- switchable DeepSpeed-Ulysses sequence parallelism (non-invasive) ---
        # Applied AFTER to(device) and BEFORE DDP wrap so DDP wraps the patched model and its
        # all_reduce averages the per-chunk gradients over the full world (correct up to a
        # constant 1/sp factor, params stay identical). stormer.py is never edited.
        self._sp_size = int(getattr(self.args, "ulysses_sp", 1) or 1)
        self._sp_group = None
        if self._sp_size > 1:
            if MPIUtils.size() % self._sp_size != 0:
                raise ValueError(
                    f"--ulysses-sp {self._sp_size} must divide world size {MPIUtils.size()}"
                )
            import ulysses_sp
            ulysses_sp.maybe_init_deepspeed()
            (
                self._sp_group, self._dp_group, sp_size, sp_rank, dp_size, dp_rank,
            ) = ulysses_sp.init_sp_dp_groups(self._sp_size)
            ulysses_sp.apply_ulysses_sp(self.net, self._sp_group)
            # global rank of sp-rank 0 in this contiguous sp block (broadcast source)
            self._sp_src = (MPIUtils.rank() // self._sp_size) * self._sp_size
            if MPIUtils.rank() == 0:
                log.info(
                    "ULYSSES_SP: enabled (sp_degree=%d -> %d dp groups); attention all_to_all "
                    "on forward critical path, stormer.py unedited",
                    self._sp_size, dp_size,
                )

        self._ddp_mixed_precision = None
        if MPIUtils.size() > 1:
            _bucket_cap = float(os.environ.get("STORMER_BUCKET_CAP_MB", "25"))
            reduce_dtype = get_ddp_reduce_dtype(os.environ.get("STORMER_DDP_REDUCE_DTYPE"))
            if reduce_dtype is not None:
                if _MixedPrecision is None:
                    raise RuntimeError("DDP mixed precision is unavailable in this torch build")
                self._ddp_mixed_precision = _MixedPrecision(reduce_dtype=reduce_dtype)
            if MPIUtils.rank() == 0:
                if self._ddp_mixed_precision is not None:
                    log.info(
                        "Setting up DDP (bucket_cap_mb=%.0f, reduce_dtype=%s)",
                        _bucket_cap,
                        str(reduce_dtype).replace("torch.", ""),
                    )
                else:
                    log.info("Setting up DDP (bucket_cap_mb=%.0f)", _bucket_cap)
            self.net = DDP(
                self.net,
                device_ids=[DEVICE_ID],
                # device_ids=[MPI_LOCAL_RANK],
                output_device=DEVICE_ID,
                bucket_cap_mb=_bucket_cap,
                mixed_precision=self._ddp_mixed_precision,
            )
        elif MPIUtils.rank() == 0:
            log.info("Running without DDP (single-process mode)")

        # Opt-in CPU-side agreement before each DDP training batch. Unequal
        # exhaustion raises on all ranks and takes the Abort(97) path; no
        # repeated sample may enter backward or alter the trajectory.
        self._loop_exit_consensus = None
        self._loop_exit_evidence_dir = ""
        if os.environ.get("STORMER_LOOP_EXIT_CONSENSUS", "0") == "1":
            self._loop_exit_consensus = LoopExitConsensus(MPIUtils)
            self._loop_exit_evidence_dir = os.environ.get("STORMER_LOOP_EXIT_EVIDENCE_DIR", "")
            if not self._loop_exit_evidence_dir:
                raise RuntimeError("loop-exit consensus requires STORMER_LOOP_EXIT_EVIDENCE_DIR")

        # --- Communication timing and optimization ---
        self._comm_timing = os.environ.get("STORMER_COMM_TIMING", "0") == "1"
        # Opt in. See the COMM_TIMING_RANK emitter in the training loop for why it exists.
        self._comm_timing_all_ranks = os.environ.get("STORMER_COMM_TIMING_ALL_RANKS", "0") == "1"
        self._comm_optimize = os.environ.get("STORMER_COMM_OPTIMIZE", "0") == "1"
        # Separate from COMM_TIMING and trajectory-state retention.  This opt-in
        # writes structured rank-local records rather than rank-0 log text.
        self._per_rep_reporting = os.environ.get("STORMER_PER_REP_REPORTING", "0") == "1"
        self._per_rep_step_writer = None
        self._per_rep_logical_batch_step_id = 0
        self._optimizer_update_count = 0
        if self._per_rep_reporting:
            from per_rep_reporting import StepSeriesWriter
            series_dir = os.environ.get("STORMER_PER_REP_STEP_SERIES_DIR", "")
            if not series_dir:
                raise RuntimeError("STORMER_PER_REP_REPORTING=1 requires STORMER_PER_REP_STEP_SERIES_DIR")
            self._per_rep_step_writer = StepSeriesWriter(series_dir, MPIUtils.rank())
        self._comm_bucket_ms = []  # filled by hook, cleared per step
        self._fwd_ms = 0.0
        self._bwd_ms = 0.0
        self._bwd_only_ms = 0.0
        self._c0_bwd_ms = None
        self._bucket_cap_mb = _bucket_cap if MPIUtils.size() > 1 else 25.0
        self._comm_window_steps = []  # per-step (step_ms, comm_ms, n_buckets) for current window
        self._comm_cooldown = 0  # windows to skip after a change
        if self._ddp_mixed_precision is not None and (self._comm_timing or self._comm_optimize):
            if MPIUtils.rank() == 0:
                log.warning(
                    "COMM_TIMING/COMM_OPTIMIZE disabled because DDP mixed_precision registers its own comm hook"
                )
            self._comm_timing = False
            self._comm_optimize = False
        # Hook is registered whenever any of COMM_TIMING / COMM_OPTIMIZE / DFTracer is
        # active. DFTracer activation emits per-bucket comm.all_reduce events so the
        # analyzer can compute comm_frac alongside fetch_frac / compute_frac.
        _comm_hook_enabled = (
            self._comm_timing or self._comm_optimize or PERFTRACER_ENABLE
        )
        # Clean comm probe: barrier BEFORE timing the all_reduce so the measured comm excludes
        # input-induced straggler wait (ranks synchronized first) -> straggler-free comm
        # attribution for the controller. Opt-in (STORMER_COMM_SYNC_PROBE=1); adds a barrier per
        # bucket, so only for the clean-attribution/CLBO mode. Default off = unchanged.
        _comm_sync_probe = os.environ.get("STORMER_COMM_SYNC_PROBE", "0") == "1"
        self._comm_sync_probe = _comm_sync_probe
        self._comm_hook_wanted = comm_hook.hook_wanted(
            self._comm_timing, self._comm_optimize, PERFTRACER_ENABLE, MPIUtils.size())
        if MPIUtils.rank() == 0 and _comm_hook_enabled and comm_hook.comm_probe_mode() == "off":
            log.info("COMM_HOOK: STORMER_COMM_PROBE=off -> native overlapped DDP, no comm events")
        if self._comm_hook_wanted:
            self.net.register_comm_hook(
                self._comm_bucket_ms,
                comm_hook.make_hook_for_mode(dft_ai.comm.all_reduce, _comm_sync_probe))
            if MPIUtils.rank() == 0:
                _reason_bits = []
                if self._comm_timing:
                    _reason_bits.append("timing")
                if self._comm_optimize:
                    _reason_bits.append("optimize")
                if PERFTRACER_ENABLE:
                    _reason_bits.append("dftracer")
                log.info(
                    "COMM_HOOK: registered (%s, all-reduce serialized)",
                    "+".join(_reason_bits) or "default",
                )

        if MPIUtils.rank() == 0:
            log.info("Setting up Scaler")

        self.scaler = torch.GradScaler(  # only for float16 stablility
            self.device.type,
            enabled=(self.dtype == torch.float16 and torch.cuda.is_available()),
        )

    def make_loader(self, epoch: int):
        # With LiveDataLoader, reuse the same loader across epochs
        # (set_epoch resets the sampler; the loader object persists for mid-epoch reconfigure)
        if not hasattr(self, "_live_loader"):
            self._live_loader, self._live_sampler = self.datamodule.dataloader(epoch=epoch)

        from live_data_loader import LiveDataLoader
        if isinstance(self._live_loader, LiveDataLoader):
            self._live_loader.set_epoch(epoch)
            log0(
                "make_loader: epoch=%s num_workers=%s prefetch_factor=%s (LiveDataLoader)",
                epoch, self._live_loader.num_workers, self._live_loader.prefetch_factor,
            )
            return self._live_loader, self._live_sampler

        # Standard DataLoader: recreate each epoch
        train_loader, train_sampler = self.datamodule.dataloader(epoch=epoch)
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        log0(
            "make_loader: epoch=%s num_workers=%s prefetch_factor=%s pin_memory=%s persistent_workers=%s",
            epoch,
            self.args.num_workers,
            self.args.prefetch_factor,
            self.args.pin_memory,
            self.args.persistent_workers,
        )
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        return train_loader, train_sampler

    def configure_scheduler(self, n_steps: int):
        return configure_lr_scheduler(
            self.optimizer,
            max_epochs=self.epochs,
            n_steps=n_steps,
            warmup_epochs=10,
            warmup_start_lr=1e-8,
            eta_min=1e-8,
            num_nodes=MPIUtils.num_nodes(),
            num_devices=self.args.ngpus_per_node,
        )

    def forward_train(self, x: torch.Tensor, variables, interval, *args, **kwargs):
        if "stormer" in self.model_id:
            padded_x, pad_size = pad(x=x, patch_size=self.model.patch_size)
            norm_pred_diff = self.net(padded_x, variables, interval)[
                :, :, pad_size:
            ]  # diff in the normalized space
        else:
            norm_pred_diff = self.net(x)

        return replace_constant(norm_pred_diff, variables)

    @dft_ai.device.transfer
    @_mot.timed("delivery.h2d")
    def send_to_device(self, batch):
        x, gt_diff, interval_tensors = batch[:3]
        x = x.to(device=self.device)  # , non_blocking=True)
        gt_diff = gt_diff.to(device=self.device)  # , non_blocking=True)
        interval_tensors = interval_tensors.to(device=self.device)  # , non_blocking=True)
        # Ulysses SP: all sp ranks in a block share one sample; the input read is deterministic
        # per index (identical across the block) but the interval is randomly chosen per
        # __getitem__, so sync it within the sp_group to keep the t-conditioning (and thus the
        # gathered attention sequence) coherent. Tiny [B] broadcast; does not perturb the read
        # or all_to_all timing. (Target gt_diff stays per-rank; it only affects the loss value,
        # not the systems measurement.)
        if getattr(self, "_sp_size", 1) > 1 and self._sp_group is not None:
            dist.broadcast(interval_tensors, src=self._sp_src, group=self._sp_group)
        return x, gt_diff, interval_tensors

    def autocast_context_manager(self):
        return setup_precision_context(
            dtype=self.dtype,
            device_type=DEVICE_TYPE,
            mixed=True,
            enabled=True,
            # enabled=Distributed.strategy() != DistributedStrategy.DEEPSPEED,
        )

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with self.autocast_context_manager():
            yield

    @dft_ai.compute.forward
    @_mot.timed("compute.forward_loss")
    def forward_step(self, batch: Any):
        _fp("fwd.start")
        return self._forward_loss(batch)

    def _forward_loss(self, batch: Any):
        x, gt_diff, interval = batch

        with self.forward_context():
            with _mot.span("compute.forward"):
                pred_diff = self.forward_train(x, self.variables, interval)

            with _mot.span("compute.loss"):
                loss_dict = lat_weighted_mse(
                    pred_diff,
                    gt_diff,
                    self.variables,
                    self.lat,
                    weighted=self.weighted_loss,
                    weight_dict=WEIGHT_DICT,
                )

        loss = loss_dict["w_mse_aggregate"]
        return loss, loss.detach()

    @dft_ai.compute.backward
    @_mot.timed("compute.backward_optimizer_inclusive")
    def backward_step(self, loss, batch_idx: int, n_accum_steps: int):
        # This is deliberately an observed-completion bit, rather than an
        # attempted-step counter.  Retained state v1 is FP32-only, but Trainer
        # normally owns a truthy *disabled* GradScaler, so count only after its
        # step path and the scheduler path have both run for an already-finite
        # loss. Snapshot capture checks retained parameters separately.
        if self._trajectory_retention is not None:
            self._last_observed_optimizer_update = False
        _fp("bwd.start")
        _time_bwd = (self._comm_timing or self._comm_optimize) and self._c0_active()
        if _time_bwd:
            torch.cuda.synchronize()
            _tb = time.time()
        if self.scaler:
            with _mot.span("compute.backward"):
                self.scaler.scale(loss).backward()
        else:
            with _mot.span("compute.backward"):
                loss.backward()
        if _time_bwd:
            torch.cuda.synchronize()
            self._bwd_only_ms = (time.time() - _tb) * 1000

        # Tail diagnostic only (comm_hook.compute_exposed_comm_ms): this event completes after the last
        # all-reduce (DDP's finalize_backward stream-waits on every comm future before backward() returns).
        # The production exposed-comm value is backward wall minus the comm-free reference, see
        # _calibrate_compute_only_backward.
        self._bwd_end_event = comm_hook.record_backward_end()

        # Only step optimizer at end of accumulation window
        if batch_idx % n_accum_steps == 0:
            _scale_before = self.scaler.get_scale() if self.scaler and self.scaler.is_enabled() else None
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            _gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            if os.environ.get("STORMER_LOG_GRAD_NORM", "0") == "1":
                self._last_grad_norm = repr(float(_gn))   # opt-in probe-equivalence check; float() syncs
            if self.scaler:
                with _mot.span("compute.optimizer"):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                with _mot.span("compute.optimizer"):
                    self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            # GradScaler can skip optimizer.step() on overflow. The FP32
            # campaign scaler is disabled; count only completed updates in
            # either mode, independently of optional trajectory retention.
            _did_update = _scale_before is None or self.scaler.get_scale() >= _scale_before
            if _did_update:
                self._optimizer_update_count += 1

            # `training_step` already excludes nonfinite losses.  In the
            # reviewed FP32 cell the disabled GradScaler takes the real
            # optimizer step above; snapshot capture performs the expensive
            # all-tensor finiteness check only at selected epoch-end points.
            if self._trajectory_retention is not None:
                self._last_observed_optimizer_update = True

    def _c0_active(self) -> bool:
        """The completion-based exposed-comm measure applies (event/ab probe, DDP over more than one rank)."""
        return comm_hook.comm_probe_mode() in ("event", "ab") and MPIUtils.size() > 1

    def _calibrate_compute_only_backward(self, batch) -> Optional[float]:
        """Median wall (ms) of a backward pass with NO gradient communication, measured once.

        Exposed comm = the backward wall of a real step minus this reference. The all-reduce does not show
        up as a tail after backward on this system (last-bucket ready -> backward() returned read 5.7 ms
        while the paired counterfactual gave 168-182 ms at 16N): it appears as DILATION of the backward
        compute itself, so only a comm-free reference separates it. Math-neutral: forward and backward on the
        first real batch inside DDP `no_sync()` (no collective for the gradients), gradients dropped, the
        torch/CUDA RNG states restored, parameter and optimizer state asserted unchanged by digest.
        Symmetric on every rank (same code path, SP collectives in forward run on all ranks)."""
        if not hasattr(self.net, "no_sync"):
            return None
        from rebuild_invariants import state_digest
        before = state_digest(self.net.module, self.optimizer)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.optimizer.zero_grad(set_to_none=True)
        walls = []
        try:
            with self.net.no_sync():
                for _ in range(5):
                    loss, _li = self._forward_loss(batch)
                    torch.cuda.synchronize()
                    t0 = time.time()
                    (self.scaler.scale(loss) if self.scaler else loss).backward()
                    torch.cuda.synchronize()
                    walls.append((time.time() - t0) * 1000)
                    self.optimizer.zero_grad(set_to_none=True)
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        after = state_digest(self.net.module, self.optimizer)
        if before != after:
            raise RuntimeError("comm calibration changed model/optimizer state")
        c0 = statistics.median(walls[1:])
        log0("COMM_CALIBRATION compute_only_backward_ms=%.3f samples=%s state_digest_equal=True",
             c0, ",".join("%.1f" % w for w in walls))
        return c0

    @dft_ai.compute
    @_mot.timed("compute.parent")
    def training_step(self, batch: Any, batch_idx: int):
        if self._trajectory_retention is not None:
            self._last_observed_optimizer_update = False
        n_accum_steps = self.accumulate_grad_batches
        # Phase relative to the last apply-window origin, NOT the raw per-rank batch_idx.
        # Every rank stamps the same origin at a synchronisation point, so this agrees job-wide.
        _phase = batch_idx - self._accum_origin

        # Only zero gradients at start of accumulation window
        if ((_phase - 1) % n_accum_steps) == 0:
            self.optimizer.zero_grad(set_to_none=True)

        # Apply same sync context to forward + backward
        if _phase % n_accum_steps != 0 and hasattr(self.net, "no_sync"):
            sync_context = self.net.no_sync()
        else:
            sync_context = contextlib.nullcontext()

        with sync_context:
            if self._comm_timing or self._comm_optimize:
                torch.cuda.synchronize()
                _t0 = time.time()

            loss, loss_item = self.forward_step(batch)

            if self._comm_timing or self._comm_optimize:
                torch.cuda.synchronize()
                self._fwd_ms = (time.time() - _t0) * 1000

            # Skip NaN losses to prevent gradient divergence at scale
            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad(set_to_none=True)
                return float("nan")

            # Normalize loss for gradient accumulation (keeps effective LR constant)
            if n_accum_steps > 1:
                loss = loss / n_accum_steps

            if self._comm_timing or self._comm_optimize:
                torch.cuda.synchronize()
                _t1 = time.time()

            self.backward_step(loss, _phase, n_accum_steps)

            if self._comm_timing or self._comm_optimize:
                torch.cuda.synchronize()
                self._bwd_ms = (time.time() - _t1) * 1000

        return loss_item.cpu().item()

    def _setup_optimizer_context(self):
        """Initialize dfoptimizer context for knob registration and plan reception.

        CONTEXT SCOPE (LIVEFLOW-STATUS.md sec 56). The Mofka client this creates is the confirmed
        cause of the boundary-2 wedge: G1 ran 120/120 with the observation plane at FULL rate
        (228 findings) and no in-rank context, while every arm carrying one died at step 48-49 in
        `hipDeviceSynchronize` with the GPU stream undrained. The in-rank consumer competes with
        NCCL's CPU-side proxy inside the same process, and no amount of external pinning reaches it.

        Fewer clients is therefore the lever. This is safe because plan VALUES are already
        distributed by the MPI `Allreduce(MAX)` in `_sync_apply_window`: a rank that drained nothing
        contributes the -1 sentinel and adopts the agreed value. `_optimizer_active` is gated on the
        rank-symmetric DFOPTIMIZER_ENABLE, not on holding a context, so every rank still joins that
        collective -- which is what keeps the ranks in lockstep and is why this cannot deadlock.

        per_node (default): one context per NODE. Plans are published to the NODE-LOCAL bedrock by
            that node's optimizer, so a per-node consumer is the smallest scope that still receives
            them; a single global consumer would never see the other nodes' plans.
        all: previous behaviour, one per rank -- kept so the wedge stays reproducible on demand.
        global: one for the whole job. Only correct if plans ever become globally published.
        """
        self._optimizer_ctx = None
        if os.environ.get("DFOPTIMIZER_ENABLE", "0") != "1":
            return
        _scope = os.environ.get("LIVEFLOW_CTX_SCOPE", "per_node").strip().lower()
        if _scope != "all":
            _local = os.environ.get("FLUX_TASK_LOCAL_ID")
            if _local is None:
                _ppn = max(1, int(os.environ.get("PPN", "4") or 4))
                _local = MPIUtils.rank() % _ppn
            _local = int(_local)
            _holds = (MPIUtils.rank() == 0) if _scope == "global" else (_local == 0)
            if not _holds:
                log.debug("optimizer context skipped on rank %d (scope=%s)",
                          MPIUtils.rank(), _scope)
                return
        try:
            from dfoptimizer.runtime import optimizer_context
            group_file = os.environ.get("DFTRACER_MOFKA_GROUP_FILE", "")
            self._optimizer_ctx = optimizer_context(
                namespace="stormer",
                group_file=group_file,
                topic_plans="optimizer_plans",
                topic_acks="optimizer_acks",
                topic_registry="optimizer_registry",
            )
            log0("Optimizer context started (group_file=%s)", group_file)
        except Exception as e:
            log0("WARNING: optimizer context setup failed: %s", e)

    def _refine_apply_cost(self):
        """One window after an apply: the stall its next step actually saw, worst rank, into the
        applied-vector snapshot. Symmetric on every rank (all ranks probed the same apply)."""
        lw = getattr(self, "_loop_window", None)
        if lw is None or getattr(lw, "_apply_probe", None) is None:
            return
        local = lw.apply_probe_result()
        if local is None:
            return      # the step after the apply has not completed yet; the probe stays armed
        cost = float(local)
        if MPIUtils.size() > 1 and MPIUtils.is_mpi():
            from mpi4py import MPI
            _c = np.array([cost], dtype=np.float64)
            _co = np.empty_like(_c)
            MPIUtils.comm_world().Allreduce(_c, _co, op=MPI.MAX)
            cost = float(_co[0])
        if getattr(self, "_phys_vec", None) is not None:
            lw.set_physical_config(self._phys_vec, self._phys_applied_step, apply_cost_s=cost)
        if MPIUtils.rank() == 0:
            log.info("APPLY_STALL: worst_rank_s=%.3f (first step after the apply less the median step before it)", cost)

    def _sync_apply_window(self, train_loader, step=None):
        """Apply optimizer plans SYMMETRICALLY across all ranks at a window boundary.

        The optimizer context is active on only a SUBSET of ranks and plans arrive at slightly
        different window boundaries. Two hard constraints shaped this design:
          1. Consensus must NOT use an NCCL collective. Under Ulysses SP the process groups are
             subgroups (sp_group all_to_all, dp_group all_reduce); injecting a WORLD NCCL collective
             here interleaves with those subgroup collectives and deadlocks (observed: ALLTOALL vs
             BROADCAST timeout). So consensus runs over MPI/mpi4py on the CPU instead - fully
             independent of the NCCL comm stream.
          2. Every applied knob must be collective-free. Input knobs (reconfigure) are per-rank.
             The comm knob is accumulate_grad_batches (sync cadence), applied by assignment - NO
             DDP rebuild (whose broadcast is what deadlocked with SP).
        Each rank drains its own actions (or none), then we MPI-allreduce(MAX) the desired values
        with a -1 sentinel for 'no plan seen': every rank converges on the same targets and applies
        together; the sentinel (not the stale current) lets a knob move DOWN too. Gated on the
        rank-symmetric DFOPTIMIZER_ENABLE so LiveIO/no-optimizer runs are byte-identical."""
        if not getattr(self, "_optimizer_active", False):
            return
        from live_data_loader import LiveDataLoader
        SENT = -1
        self._refine_apply_cost()
        want_w = want_pf = want_accum = want_bcap = want_dec = SENT
        want_tm = want_rt = SENT   # layout knobs: rank-local, math-neutral, applied at a sample boundary
        ctx = self._optimizer_ctx
        _pending_acks = []   # (action, knob, old_value): acked only AFTER confirmed all-rank apply
        # STORMER_CONSENSUS=unanimous (judge D1d): no MPI MAX across nodes.  Node-local knobs reconcile within the
        # node; bucket_cap_mb (world collective) needs cross-node unanimity or holds (consensus.py).
        _unan = os.environ.get("STORMER_CONSENSUS", "max") == "unanimous"
        _bcap_vote, _bcap_new = SENT, []
        if ctx is not None and not ctx._noop and isinstance(train_loader, LiveDataLoader):
            actions = ctx.drain_actions_for("stormer.make_loader", at="window_boundary") or []
            for action in actions:
                knob_name = action.knob_id
                for _pfx in ("stormer.", "layout.", "delivery."):
                    if knob_name.startswith(_pfx):
                        knob_name = knob_name[len(_pfx):]
                        break
                if knob_name in ("transfer_mb", "read_threads") and getattr(self, "_layout_live", False):
                    import liveflow_layout as _ll
                    if knob_name == "transfer_mb":
                        want_tm = int(action.new_value)
                    else:
                        want_rt = int(action.new_value)
                    _pending_acks.append((action, knob_name, int(_ll.get_layout()[knob_name])))
                elif knob_name == "num_workers":
                    want_w = int(action.new_value)
                    _pending_acks.append((action, knob_name, train_loader.num_workers))
                elif knob_name == "prefetch_factor":
                    want_pf = int(action.new_value)
                    _pending_acks.append((action, knob_name, train_loader.prefetch_factor))
                elif knob_name == "accumulate_grad_batches":
                    want_accum = int(action.new_value)
                    _pending_acks.append((action, knob_name, self.accumulate_grad_batches))
                elif knob_name == "bucket_cap_mb" and getattr(self, "_tune_bcap_live", False) and _unan:
                    _bcap_vote = int(action.new_value)
                    _bcap_new.append((action, knob_name, int(self._bucket_cap_mb)))
                elif knob_name == "bucket_cap_mb" and getattr(self, "_tune_bcap_live", False):
                    want_bcap = int(action.new_value)
                    _pending_acks.append((action, knob_name, int(self._bucket_cap_mb)))
                elif knob_name == "decode_threads" and getattr(self, "_tune_decode_live", False):
                    want_dec = int(action.new_value)
                    _pending_acks.append((action, knob_name,
                                          int(os.environ.get("STORMER_DECODE_THREADS", "1") or 1)))
                else:
                    ctx.ack_action(action, None, None, status="rejected")
        # TEST HOOK (STORMER_APPLY_SCHEDULE="<cum_step>:<workers>:<prefetch>,..."): apply a fixed
        # sequence of loader changes at the first window boundary at or after each cumulative
        # step, identically on every rank, to measure what an apply costs without the planner in
        # the way. Unset in every campaign arm.
        _sched = os.environ.get("STORMER_APPLY_SCHEDULE", "").strip()
        if _sched:
            _lw0 = getattr(self, "_loop_window", None)
            _cum0 = int(getattr(_lw0, "_completed_steps", 0) or 0)
            _done = getattr(self, "_sched_done", None)
            if _done is None:
                _done = self._sched_done = set()
            for _ent in _sched.split(","):
                _s_at, _s_w, _s_pf = (int(x) for x in _ent.split(":"))
                if _cum0 >= _s_at and _s_at not in _done:
                    _done.add(_s_at)
                    want_w, want_pf = _s_w, _s_pf
                    break
        # Consensus over MPI (CPU) - never NCCL, so it cannot interleave with SP subgroup
        # collectives. Buffer-based Allreduce on a numpy array does an ELEMENT-WISE max (the
        # pickle-based lowercase allreduce would compare the lists lexicographically instead).
        # UNCONDITIONAL PROBE (LIVEFLOW-STATUS sec 78). Everything below -- the consensus that
        # makes every rank adopt the same knob values, and the barriers around the collective
        # actuators -- is gated on this one condition. If it is False the gate fails SILENTLY:
        # each rank then applies only what IT drained, so with per_node scope exactly the four
        # context-holding ranks enter a collective DDP rebuild and the other twelve never do.
        # That is precisely the sec 68 stack (ranks 0/4/8/12 in _verify_param_shape_across_processes).
        # Log it once per window so the condition can never again be assumed.
        if not getattr(self, "_consensus_probe_logged", False):
            self._consensus_probe_logged = True
            log.info("APPLY_CONSENSUS_PROBE: rank=%s size=%s is_mpi=%s -> consensus_%s",
                     MPIUtils.rank(), MPIUtils.size(), MPIUtils.is_mpi(),
                     "ON" if (MPIUtils.size() > 1 and MPIUtils.is_mpi()) else "OFF")
        if _unan and MPIUtils.size() > 1 and MPIUtils.is_mpi():
            from mpi4py import MPI
            _world = MPIUtils.comm_world()
            if getattr(self, "_node_comm", None) is None:
                self._node_comm = _world.Split_type(MPI.COMM_TYPE_SHARED)
                self._bcap_sticky = _consensus.Sticky(int(os.environ.get("STORMER_CONSENSUS_STICKY_WINDOWS", "3")))
                self._consensus_agree = _consensus.Agreement()
            _nb = np.array([want_w, want_pf, want_accum, want_dec, want_tm, want_rt], dtype=np.int64)
            _nbo = np.empty_like(_nb)
            self._node_comm.Allreduce(_nb, _nbo, op=MPI.MAX)          # node-local knobs: node MAX only
            want_w, want_pf, want_accum, want_dec, want_tm, want_rt = (int(x) for x in _nbo)
            for _act, _kn, _old in self._bcap_sticky.offer(_bcap_vote, _bcap_new):
                if ctx is not None and not ctx._noop:
                    ctx.ack_action(_act, _old, None, status="failed", reason="consensus_superseded")
            _sv = np.array([self._bcap_sticky.value], dtype=np.int64)
            _svo = np.empty_like(_sv)
            self._node_comm.Allreduce(_sv, _svo, op=MPI.MAX)
            _lead = self._node_comm.Get_rank() == 0
            _c = _consensus.contribution(_lead, int(_svo[0]))
            _sums = np.array(_c[:2], dtype=np.int64); _sumo = np.empty_like(_sums)
            _world.Allreduce(_sums, _sumo, op=MPI.SUM)
            _mins = np.array(_c[2:], dtype=np.int64); _mino = np.empty_like(_mins)
            _world.Allreduce(_mins, _mino, op=MPI.MIN)
            _out, _val = _consensus.decide(int(_sumo[0]), int(_sumo[1]), int(_mino[0]), int(_mino[1]))
            self._consensus_agree.note(_out)
            if _out != "none" and MPIUtils.rank() == 0:
                log.info("CONSENSUS knob=bucket_cap_mb outcome=%s votes=%d nodes=%d vmin=%d vmax=%d "
                         "agreement_rate=%.3f (%d/%d voting windows)", _out, int(_sumo[1]), int(_sumo[0]),
                         int(_mino[0]), -int(_mino[1]), self._consensus_agree.agreement_rate,
                         self._consensus_agree.unanimous, self._consensus_agree.voting)
            if _out == "unanimous":
                want_bcap = _val
                _pending_acks.extend(self._bcap_sticky.clear())
            elif self._bcap_sticky.expired():
                for _act, _kn, _old in self._bcap_sticky.clear():
                    if ctx is not None and not ctx._noop:
                        ctx.ack_action(_act, _old, None, status="failed", reason="no_unanimity_within_sticky_window")
        elif MPIUtils.size() > 1 and MPIUtils.is_mpi():
            from mpi4py import MPI
            # VERDICT ARBITRATION (orchestrator decision 2026-10-04, ported from halo3d): raises
            # reconcile by MAX as before; a proposal BELOW the current value is a node planner's
            # REVERT vote, applied only on a strict majority of the node planners. One node cannot
            # undo a move most nodes confirmed (see dfoptimizer.runtime.reconcile).
            from dfoptimizer.runtime import reconcile as _reconcile
            _comm = MPIUtils.comm_world()
            _cur = {"num_workers": int(getattr(train_loader, "num_workers", 0) or 0),
                    "prefetch_factor": int(getattr(train_loader, "prefetch_factor", 0) or 0),
                    "bucket_cap_mb": int(self._bucket_cap_mb)}
            if not hasattr(self, "_revert_votes"):
                self._revert_votes = _reconcile.RevertVotes()

            def _red(op):
                def f(a):
                    o = np.empty_like(a)
                    _comm.Allreduce(a, o, op=op)
                    return o
                return f
            _agreed, _rev_events = _reconcile.reconcile(
                tuple(_cur), {"num_workers": want_w, "prefetch_factor": want_pf, "bucket_cap_mb": want_bcap},
                _cur, self._revert_votes, ctx is not None and not ctx._noop, _red(MPI.MAX), _red(MPI.SUM))
            if MPIUtils.rank() == 0:
                for _ev in _rev_events:
                    log.info("REVERT_VOTE %s", json.dumps(dict(step=step, **_ev), sort_keys=True))
            _buf = np.array([want_accum, want_dec, want_tm, want_rt], dtype=np.int64)
            _out = np.empty_like(_buf)
            _comm.Allreduce(_buf, _out, op=MPI.MAX)
            want_w, want_pf, want_bcap = (_agreed["num_workers"], _agreed["prefetch_factor"],
                                          _agreed["bucket_cap_mb"])
            want_accum, want_dec, want_tm, want_rt = (int(_out[0]), int(_out[1]), int(_out[2]), int(_out[3]))
            # STEP-SKEW PROBE (LIVEFLOW-STATUS sec 77). Both wedges we have stacks for are
            # consistent with ranks DISAGREEING ON THE STEP COUNT after a loader respawn:
            #   - with accum>1, a step-count difference flips `batch_idx % accum`, so some ranks
            #     enter no_sync() and others all_reduce (sec 74.2 split);
            #   - with bucket_cap, ranks land on window boundaries at different times, so only a
            #     SUBSET reaches the collective DDP rebuild (sec 68: 4 of 16).
            # One int of MPI per window settles whether that premise is true instead of inferred.
            if step is not None:
                _sk = np.array([step, -step], dtype=np.int64)
                _sko = np.empty_like(_sk)
                MPIUtils.comm_world().Allreduce(_sk, _sko, op=MPI.MAX)
                _smax, _smin = int(_sko[0]), -int(_sko[1])
                if _smax != _smin and MPIUtils.rank() == 0:
                    log.warning("STEP_SKEW: ranks disagree: min=%d max=%d spread=%d",
                                _smin, _smax, _smax - _smin)
        # Start the apply clock AFTER consensus: the Allreduce above is already synchronising, so
        # including it would charge every rank the slowest rank's ARRIVAL time and hide the thing
        # we actually want bounded -- how long the APPLY itself blocks this rank.
        _fp("apply.start")
        _t_apply0 = time.time()
        # STACK WATCHDOG. When the apply blocks we need to know WHERE, not merely that it did. The
        # sec 43.4 validation stalled with ZERO APPLY_SYNC lines and ZERO NCCL timeouts -- some rank
        # stuck inside the apply while the others waited in the barrier below -- and nothing in the
        # logs could name the blocking line. The time-symmetry barrier removes the NCCL watchdog
        # (ranks wait in MPI, not NCCL), so it also removes the only diagnostic we had; this
        # replaces it.
        #
        # dump_traceback_later fires on EVERY rank that overruns and prints EVERY thread's Python
        # stack, so a decode-pool thread blocked in shutdown(wait=True) shows up by name. repeat=True
        # so successive dumps reveal whether the stack is stuck or merely slow. exit=False so it
        # only observes. Default 90 s: the apply is normally sub-millisecond, so this cannot fire in
        # healthy operation. STORMER_APPLY_STACK_SEC=0 disables.
        # Disabled when the process-wide periodic dump is active: both use the SAME single
        # faulthandler timer, so arming here would reset it and the cancel below would DISARM the
        # global one for the rest of the run -- a local safety net silently blinding a global
        # instrument, which is the exact failure that made sec 43.5 undiagnosable.
        # THIRD occurrence of the shared-timer trap: faulthandler has ONE timer, so arming and
        # cancelling it here silently disarmed the C-level backstop at every window boundary --
        # which is why D21 produced 16 open .cdump files and zero dumps.
        _stack_sec = 0.0 if (_STACK_EVERY > 0 or _CDUMP_ARMED) else float(
            os.environ.get("STORMER_APPLY_STACK_SEC", "90") or 0)
        if _stack_sec > 0:
            faulthandler.dump_traceback_later(_stack_sec, repeat=True, exit=False)
        # INPUT knobs: apply the agreed values on every rank (symmetric -> no stragglers). Per-rank,
        # non-collective.
        # DECODE knob: the pool lives in each DataLoader WORKER process, which read
        # STORMER_DECODE_THREADS from the environment they were forked with. So set the env in the
        # parent and force a loader reconfigure -- the same actuator the input knobs already use --
        # and the restarted workers pick up the new width. Ordered BEFORE the input apply so a
        # simultaneous change costs ONE worker restart, not two.
        # ---- SAFETY GATE: RECOMMEND-ONLY KNOBS (LIVEFLOW-DECODE-RCA.md sec 15) ----------------
        # An action is admissible only if (a) its expected benefit clears the evidence bar AND
        # (b) its actuation path has a bounded, proven-safe apply. `decode_threads` passes (a) --
        # attribution identifies it correctly and it is worth 4.6x when set at configuration time
        # (D7, 120/120 steps) -- but fails (b): changing it ONLINE wedges the job, cause unknown
        # after fourteen eliminated hypotheses (sec 14). The controller therefore COMPUTES and
        # REPORTS the action and declines to apply it.
        #
        # This is a designed behaviour, not a workaround: a threshold-free controller needs a gate
        # that bounds not only whether an action is WORTH taking but whether taking it can HARM the
        # job. The recommendation is still evidence -- it shows attribution reaching the right
        # conclusion -- while the gate keeps the run alive.
        #
        # DFOPTIMIZER_RECOMMEND_ONLY is a comma-separated knob list; empty restores full actuation.
        _rec_only = {k.strip() for k in
                     os.environ.get("DFOPTIMIZER_RECOMMEND_ONLY", "").split(",") if k.strip()}
        _withheld = frozenset()
        _dec_changed = (getattr(self, "_tune_decode_live", False) and want_dec > SENT
                        and want_dec != int(os.environ.get("STORMER_DECODE_THREADS", "1") or 1))
        if _dec_changed and "decode_threads" in _rec_only:
            if MPIUtils.rank() == 0:
                log.info("SAFETY_GATE: decode_threads %s -> %d RECOMMENDED, not applied "
                         "(actuation path not proven safe; see LIVEFLOW-DECODE-RCA.md sec 15)",
                         os.environ.get("STORMER_DECODE_THREADS", "1"), want_dec)
            _dec_changed = False        # attribution recorded; actuation withheld
            _withheld = frozenset(["decode_threads"])
        if _dec_changed:
            _old_dec = os.environ.get("STORMER_DECODE_THREADS", "1")
            os.environ["STORMER_DECODE_THREADS"] = str(want_dec)
            # EAGER APPLY. Setting the env var alone defers the real change to whenever the next
            # read happens -- mid-step, GPU in flight, at a point no rank agreed on. Materialise it
            # HERE instead: every rank is already MPI-consensus on want_dec at this line, and the
            # time-symmetry barrier below holds them together until all have done it. The pool is
            # pre-built (see _ensure_decode_pool), so this is now an integer assignment, but doing
            # it here means the transition instant is one every rank shares.
            # Shared-memory store: every already-running worker sees this on its next read.
            _DECODE_LIMIT.value = max(1, int(want_dec))
            _ensure_decode_pool()
            if MPIUtils.rank() == 0:
                log.info("DECODE_OPT: decode_threads %s -> %d (applied at window boundary)",
                         _old_dec, want_dec)
        # LAYOUT knobs: one shared-memory store each, read by every worker at the start of its next
        # sample (a sample is read under a single setting). Identical bytes in every setting, so
        # neither correctness nor the sample order depends on all ranks agreeing; the node-local /
        # world MAX above only makes the applied vector coherent for the planner's credit rule.
        if getattr(self, "_layout_live", False) and (want_tm > SENT or want_rt > SENT):
            import liveflow_layout as _ll
            _ll.set_layout(transfer_mb=want_tm if want_tm > SENT else None,
                           read_threads=want_rt if want_rt > SENT else None)
        new_w = want_w if want_w > SENT else None
        new_pf = want_pf if want_pf > SENT else None
        _is_live = isinstance(train_loader, LiveDataLoader)
        _loader_changed = False
        if (new_w is not None or new_pf is not None) and _is_live:
            train_loader.reconfigure(num_workers=new_w, prefetch_factor=new_pf)
            _loader_changed = True
        # NO RESTART FOR DECODE (LIVEFLOW-STATUS sec 82). This used to call request_restart(),
        # because decode propagated through the worker ENVIRONMENT and only a re-fork could deliver
        # a new value to running workers. `_DECODE_LIMIT` is now a shared-memory Value, so the
        # store above is already visible in every worker on its next read -- the re-fork delivered
        # nothing that the shared store does not, and it cost a full Mofka re-initialisation in
        # every worker, which is where the sec 81 never-returning push comes from.
        #
        # Keeping this as an explicit no-op branch rather than deleting it: `_dec_changed` is still
        # the signal that decode moved, and a future actuator that genuinely needs a respawn should
        # be added here deliberately, not by resurrecting one.
        if _dec_changed and new_w is None and new_pf is None and _is_live:
            if MPIUtils.rank() == 0:
                log.info("DECODE_OPT: applied via shared memory, no worker restart")
        # ---- EAGER MATERIALISATION (LIVEFLOW-DECODE-RCA.md sec 16) ----------------------------
        # THE fix for the sec 16 abort. reconfigure()/request_restart() only REQUEST; the teardown
        # and the fork of `num_workers` worker processes used to happen lazily on the next fetch --
        # after the time-symmetry barrier below had already passed and certified "apply 0.0 ms".
        # Every rank then respawned at a moment no rank agreed on. The first to finish reached the
        # next Ulysses-SP dist.broadcast alone, sat in NCCL 600 s, and the watchdog killed the job.
        #
        # Doing it HERE puts the whole cost -- process creation AND first-batch warm-up -- inside
        # the window the barrier protects. Stragglers then wait for one another in MPI, which has
        # no watchdog and costs delay instead of death, and APPLY_SYNC finally reports the real
        # apply latency instead of the cost of a flag write.
        # GATE (2026-08-21). The sec 16 abort this eager path was written to fix was itself a
        # SYMPTOM of the control-plane bug fixed in LIVEFLOW-STATUS sec 61: sec 16.1 records that
        # "rank 2 is the rank that received the num_workers=5 / prefetch_factor=5 plans -- the ONLY
        # rank of 16 that did", i.e. each rank's consumer died after its first event, so ranks
        # applied different knobs and fell out of SP lockstep. With delivery repaired every rank now
        # gets every plan and the MPI consensus agrees the values, so the eager respawn may no
        # longer be needed -- and it does a worker fork plus a blocking first-batch fetch INSIDE the
        # control window, which is its own hazard. Default stays 1 (current behaviour); set
        # STORMER_APPLY_EAGER=0 to restore the lazy path that worked in sec 35 (2026-08-11).
        _eager = os.environ.get("STORMER_APPLY_EAGER", "1") == "1"
        _apply_error = None
        try:
            _eager_done = bool(_is_live and _eager and train_loader.apply_pending())
        except Exception as exc:     # reported through the all-rank flag, never acked as applied
            _eager_done, _apply_error = False, exc
            log.error("APPLY_EAGER failed on rank %s: %r", MPIUtils.rank(), exc)
        if _eager_done:
            if MPIUtils.rank() == 0:
                log.info("APPLY_EAGER: loader materialised in-window (workers=%s prefetch=%s)",
                         train_loader.num_workers, train_loader.prefetch_factor)
        # COMM knob: change the gradient-sync cadence by assignment. No collective. Clear any stale
        # partial-accumulation gradients so the next accumulation window starts clean (mirrors the
        # self-contained heuristic in _comm_optimize_window).
        if want_accum > SENT and want_accum != self.accumulate_grad_batches:
            old_accum = self.accumulate_grad_batches
            self.accumulate_grad_batches = want_accum
            self.optimizer.zero_grad(set_to_none=True)
            if MPIUtils.rank() == 0:
                log.info("COMM_OPT: accumulate_grad_batches %d -> %d (sync cadence)",
                         old_accum, want_accum)
        # COMM knob #2: bucket_cap_mb. This one IS collective (the DDP rebuild broadcasts module
        # state), which is exactly what deadlocked against the Ulysses SP all_to_all before. Make it
        # safe by construction rather than by timing luck:
        #   1. cuda.synchronize()  -> every rank drains its in-flight SP/DDP GPU collectives
        #   2. MPI barrier (CPU)   -> all ranks arrive at the SAME point; cannot interleave with NCCL
        #   3. rebuild             -> the broadcast is then the ONLY NCCL op anywhere in the job
        #   4. MPI barrier         -> nobody resumes training until every rank holds the new wrapper
        # want_bcap is already MPI-consensus so every rank rebuilds to the SAME value.
        if (getattr(self, "_tune_bcap_live", False) and want_bcap > SENT
                and want_bcap != int(self._bucket_cap_mb)):
            _ta = time.perf_counter()
            torch.cuda.synchronize()
            _tb = time.perf_counter()
            if MPIUtils.size() > 1 and MPIUtils.is_mpi():
                MPIUtils.comm_world().Barrier()
            _tc = time.perf_counter()
            self._rebuild_ddp(float(want_bcap))
            _td = time.perf_counter()
            if MPIUtils.size() > 1 and MPIUtils.is_mpi():
                MPIUtils.comm_world().Barrier()
            log.info("REBUILD_APPLY_PHASES rank=%d sync_s=%.6f barrier1_s=%.6f rebuild_s=%.6f barrier2_s=%.6f "
                     "since_apply_start_s=%.6f", MPIUtils.rank(), _tb - _ta, _tc - _tb, _td - _tc,
                     time.perf_counter() - _td, time.time() - _t_apply0)

        # ---- ACCUMULATION-PHASE RE-ALIGNMENT (LIVEFLOW-STATUS.md sec 75) ----------------------
        # Every rank reaches this point together (the Allreduce above blocks until all arrive), so
        # stamping the origin here makes `batch_idx - _accum_origin` read the SAME value on every
        # rank from now on, whatever per-rank drift a loader respawn introduced. Zero the gradients
        # so the next accumulation window starts clean rather than mixing pre- and post-realignment
        # partial sums. Both are no-ops in effect when accum == 1.
        if _loader_changed and step is not None and self._accum_origin != step:
            self._accum_origin = step
            self.optimizer.zero_grad(set_to_none=True)
            if MPIUtils.rank() == 0:
                log.info("ACCUM_REALIGN: phase origin -> step %d (loader restarted)", step)

        # ---- TIME-SYMMETRY BARRIER (LIVEFLOW-STATUS.md sec 42) --------------------------------
        # The lockstep above protects the one knob whose ACTUATOR IS A COLLECTIVE (bucket_cap ->
        # DDP rebuild broadcast). The input and decode knobs take the local path above and were
        # deliberately left unguarded because they are "symmetric -> no stragglers".
        #
        # That reasoning conflates two different things: applying the same VALUE on every rank does
        # not mean applying it takes the same TIME on every rank. A loader/decode-pool reconfigure
        # can block one rank far longer than another (pool teardown, worker respawn, filesystem).
        # The slow ranks then arrive late at the workload's OWN next SP all_to_all, the other ranks
        # sit in that collective, and NCCL's 600 s watchdog kills the job. Measured directly with
        # the flight recorder: 14 ranks at enqueued=3213, ranks 10 and 11 still at 3212 -- two ranks
        # merely LATE, no collective misuse anywhere.
        #
        # One synchronising collective here removes the whole failure mode: no rank can enter the
        # next NCCL op until every rank has finished applying, so lateness costs a delay instead of
        # a deadlock. Allreduce(MAX) rather than Barrier() because it synchronises AND reports the
        # worst-rank apply latency -- the quantity sec 42 says must stay well under the collective
        # timeout, now measured every window instead of assumed.
        #
        # NOTE this is unconditional: every rank reaches it on every window boundary regardless of
        # which knobs changed. A conditional sync would itself be an asymmetric collective, i.e.
        # the exact bug it is here to prevent.
        _fp("apply.end/barrier.enter")
        # ---- POST-APPLY CONFIRMATION (ack only after all ranks observe the consensus target) ----
        # Observed state, not requested state: the inner DataLoader actually built, the decode limit
        # read back from shared memory, the accumulation counter and DDP cap currently in force.
        _wants = {"num_workers": want_w, "prefetch_factor": want_pf,
                  "accumulate_grad_batches": want_accum, "bucket_cap_mb": want_bcap,
                  "decode_threads": want_dec, "transfer_mb": want_tm, "read_threads": want_rt}
        _obs = {"accumulate_grad_batches": self.accumulate_grad_batches,
                "bucket_cap_mb": (int(self._bucket_cap_mb) if getattr(self, "_rebuild_ok", True)
                                  else -2),
                "decode_threads": int(_DECODE_LIMIT.value)}
        if getattr(self, "_layout_live", False):
            import liveflow_layout as _ll
            _lay_now = _ll.get_layout()
            _obs["transfer_mb"] = _lay_now["transfer_mb"]
            _obs["read_threads"] = _lay_now["read_threads"]
        if _is_live:
            # What the loader is actually running (the live iterator's pool for an in-place
            # resize, the rebuilt DataLoader otherwise), never the requested value.
            _ow, _opf = train_loader.observed_state()
            _settled = (not getattr(train_loader, "_pending_reconfigure", False)
                        and _apply_error is None and _ow is not None)
            _obs["num_workers"] = _ow if _settled else None
            # Observed only: DataLoader reports prefetch_factor=None when num_workers==0, and that
            # None is passed through (never substituted by the requested value).
            _obs["prefetch_factor"] = _opf if _settled else None
        # MEASURED APPLY COST: wall seconds this rank was held by the apply (consensus excluded,
        # see _t_apply0), reduced to the worst rank. It is the stall training saw, it goes onto the
        # ACK, and the planner prices a later change of the same knob from it.
        _apply_ms = (time.time() - _t_apply0) * 1000.0
        if MPIUtils.size() > 1 and MPIUtils.is_mpi():
            from mpi4py import MPI
            _dt = np.array([_apply_ms], dtype=np.float64)
            _mx = np.empty_like(_dt)
            MPIUtils.comm_world().Allreduce(_dt, _mx, op=MPI.MAX)
            _apply_ms = float(_mx[0])
        if _loader_changed and getattr(self, "_loop_window", None) is not None:
            # Measure the stall the next step sees (refines the apply cost one window later).
            self._loop_window.begin_apply_probe(_apply_ms / 1000.0)
        _flags = np.array(apply_ack.local_flags(_wants, _obs, _withheld), dtype=np.int64)
        if MPIUtils.size() > 1 and MPIUtils.is_mpi():
            from mpi4py import MPI
            _fmin = np.empty_like(_flags)
            MPIUtils.comm_world().Allreduce(_flags, _fmin, op=MPI.MIN)
            _flags = _fmin
        _loader_apply = getattr(train_loader, "last_apply", None) if _is_live else None
        if _loader_changed and MPIUtils.rank() == 0:
            log.info("APPLY_COST: worst_rank_ms=%.1f rank0_loader=%s", _apply_ms, _loader_apply)
        # APPLIED VECTOR, snapshotted into the next window so every node planner credits and
        # proposes from the SAME configuration (not from its own ACKs). `applied_step` is the first
        # step AFTER the apply stall: the step that absorbed the stall belongs to neither arm.
        _lwp = getattr(self, "_loop_window", None)
        if _lwp is not None and hasattr(_lwp, "set_physical_config") and _obs.get("num_workers") is not None:
            _vec = {"num_workers": int(_obs["num_workers"]),
                    "prefetch_factor": int(_obs["prefetch_factor"] or 0),
                    "bucket_cap_mb": int(self._bucket_cap_mb)}
            if getattr(self, "_layout_live", False):
                _vec.update(transfer_mb=int(_obs["transfer_mb"]), read_threads=int(_obs["read_threads"]))
            _prev_vec = getattr(self, "_phys_vec", None)
            if _vec != _prev_vec:
                self._phys_vec = _vec
                # The run's initial vector took effect at step 0; a later one one step past this boundary.
                self._phys_applied_step = (0 if _prev_vec is None
                                           else int(getattr(_lwp, "_completed_steps", 0)) + 1)
                _lwp.set_physical_config(self._phys_vec, self._phys_applied_step,
                                         apply_cost_s=None if _prev_vec is None else _apply_ms / 1000.0)
        if ctx is not None and not ctx._noop and _pending_acks:
            # applied_step: CUMULATIVE completed steps at this window boundary (Window counts across epochs), the
            # same axis as the analyzer's completed_steps, so the planner credits only windows that start after it.
            _lw = getattr(self, "_loop_window", None)
            _applied_step = getattr(_lw, "_completed_steps", None) if _lw is not None else None
            for _k, _st in apply_ack.ack_pending(ctx, _pending_acks, _flags, _wants, _obs, _withheld,
                                                 applied_step=_applied_step,
                                                 apply_cost_s=_apply_ms / 1000.0):
                log.info("APPLY_ACK: knob=%s status=%s epoch_step=%s", _k, _st,
                         getattr(self, "_loop_step", None))
        if MPIUtils.rank() == 0 and _apply_ms >= 1.0:
            log.info("APPLY_SYNC: worst-rank apply latency %.1f ms", _apply_ms)
        # Cancel AFTER the barrier, deliberately: a rank that finished its own apply but is stuck
        # waiting here is also something we want to see, and its stack (blocked in Allreduce)
        # distinguishes "I am the straggler" from "I am waiting for the straggler".
        if _stack_sec > 0:
            faulthandler.cancel_dump_traceback_later()

    def _sync_input_exposure_window(self, step=None):
        """Cross-node sum-of-max input exposure (STORMER-LOOP-DESIGN sec 16, judge batch 5).

        `attach_input_step_max_sum_metrics` (dfanalyzer) is correct but PER NODE: each node_agent
        sees only its own local ranks, so its max-over-pid is a max over that node's ranks, not the
        job's true `max over ALL ranks per step`. Under a rotating straggler (sec 11.2) most step
        maxima land on OTHER nodes, so the per-node number understates the real critical-path cost
        (measured: 0.129 pooled vs ~0.372 all-16-rank offline, tree-v4.3 smoke).

        This closes that gap on the RANK side instead of the analyzer side: an `Iallreduce(MAX)` over
        every rank's own per-step fetch-wait vector for the window just closed gives every rank the
        job-wide per-step max; each rank then emits the job-level sum-of-max as ONE SHM event on its
        OWN lane, so every node's existing per-node analyzer sees an already-globally-correct number
        without any change to the per-node-agent architecture.

        Opt-in (`STORMER_INPUT_EXPOSURE_JOB_SYNC=1`, default off) and INDEPENDENT of
        `_optimizer_active`: unlike `_sync_apply_window` (which no-ops when the controller is off),
        this must still be able to run controller-off, for observability (this is exactly what the
        tree-v4.3 arm-1 diagnostic needed and did not have). Default off means every existing
        controller-off run keeps ZERO new MPI collectives, byte-for-byte, unless this is explicitly
        requested -- its own synchronization cost has not been measured (sec 16.5) and must not be
        silently imposed on a run that never asked for it.
        """
        if os.environ.get("STORMER_INPUT_EXPOSURE_JOB_SYNC", "0") != "1":
            self._job_exposure_fetch_buf = []
            return
        buf = getattr(self, "_job_exposure_fetch_buf", None) or []
        cadence = int(os.environ.get("STORMER_WINDOW_EVERY_N", "0") or 0) or len(buf) or 1
        if not (MPIUtils.size() > 1 and MPIUtils.is_mpi()):
            # Single-rank/non-MPI: nothing to reduce across, but still emit so the family exists
            # identically to the multi-rank case (a caller should not have to special-case rank
            # count to know whether this fact will be present).
            job_sum = float(sum(buf))
            n_valid = len(buf)
        else:
            from mpi4py import MPI
            _world = MPIUtils.comm_world()
            local = np.zeros(cadence, dtype=np.float64)
            n_valid_local = min(len(buf), cadence)
            if n_valid_local:
                local[:n_valid_local] = buf[:n_valid_local]
            out = np.empty_like(local)
            n_valid_arr = np.array([n_valid_local], dtype=np.int64)
            n_valid_out = np.empty_like(n_valid_arr)
            req_v = _world.Iallreduce(local, out, op=MPI.MAX)
            req_n = _world.Iallreduce(n_valid_arr, n_valid_out, op=MPI.MAX)
            timeout_s = float(os.environ.get("STORMER_INPUT_EXPOSURE_SYNC_TIMEOUT_S", "60") or 60)
            deadline = time.monotonic() + timeout_s
            while not (req_v.Test() and req_n.Test()):
                if time.monotonic() >= deadline:
                    # A rank cannot safely stop waiting while its peers are still inside the SAME
                    # collective (mpi4py_two_rank_fixture.py's own reasoning): a bounded, VISIBLE
                    # failure, never a silent skip or a partial-rank continuation.
                    log.error("INPUT_EXPOSURE_SYNC: Iallreduce timed out after %.1fs; aborting",
                              timeout_s)
                    _world.Abort(124)
                time.sleep(0.01)
            n_valid = int(n_valid_out[0])
            job_sum = float(out[:n_valid].sum()) if n_valid else 0.0
        self._job_exposure_fetch_buf = []
        inst = PerfTracer.get_instance()
        dur_us = int(job_sum * 1000)
        # BUGFIX (2026-09-27, caught by the fail-fast validation run): the bound `log_event` requires
        # `int_args: Dict[str, Tuple[int, int]]` -- a (tag_type, value) pair (`_TAG_INT`, module top)
        # -- not a bare int. A bare int crashed on the very first call, on every rank, with the
        # fail-fast fix correctly aborting instead of hanging (this call had never been exercised with
        # real int_args before that run; `_emit_stall_events`, the only prior sibling caller, always
        # passes `int_args={}`, which trivially avoided the type check).
        inst.log_event(
            name="input_step_max_sum_job", cat="input_exposure_job",
            start_time=inst.get_time() - dur_us, duration=dur_us,
            int_args={"step_count": _TAG_INT(n_valid)}, float_args={}, string_args={})

    @torch.no_grad()
    def validate_epoch(self, epoch, *, trajectory_update_id: int | None = None,
                       initial_trajectory_validation: bool = False,
                       preserve_trajectory_state: bool = False):
        """Validate normally, or under the v1 observational-state guard."""
        if trajectory_update_id is None and not preserve_trajectory_state:
            return self._validate_epoch(epoch)
        from trajectory_state import validation_preserving_state
        with validation_preserving_state(self.net, self.optimizer, self.scheduler, self.scaler):
            return self._validate_epoch(
                epoch,
                trajectory_update_id=trajectory_update_id,
                initial_trajectory_validation=initial_trajectory_validation,
            )

    def _reference_validation_receipt(self, epoch, record):
        """Share rank 0's full-precision validation result with every reference rank."""
        if os.environ.get("STORMER_REFERENCE_MODE") != "1":
            return
        payload = [record if MPIUtils.rank() == 0 else None]
        if MPIUtils.size() > 1:
            dist.broadcast_object_list(payload, src=0)
        result = payload[0]
        log.info(
            "VAL_RESULT rank=%d epoch=%d val_loss=%.17g (n=%d/%d complete=%d)",
            MPIUtils.rank(), epoch, result["val_loss"], result["observed_count"],
            result["expected_count"], int(result["complete"]),
        )

    def _validate_epoch(self, epoch, *, trajectory_update_id: int | None = None, initial_trajectory_validation: bool = False):
        """Run validation on held-out data (rank 0 only to avoid file contention)."""
        # Ulysses SP: validation runs the model forward on rank 0 only while other ranks wait at
        # a barrier, but the SP attention all_to_all is a collective over the whole sp_group, so
        # a rank-0-only forward DEADLOCKS. Skip validation entirely in SP mode (all ranks check
        # the same _sp_size and return together -> no collective mismatch). Not needed for the
        # systems/timing study.
        if getattr(self, "_sp_size", 1) > 1:
            if MPIUtils.rank() == 0:
                log.info("Skipping validation (Ulysses SP mode; rank-0-only val is incompatible with SP collectives)")
            return {"complete": False, "reason": "ulysses_sp"} if trajectory_update_id is not None else None
        if MPIUtils.rank() != 0:
            # Non-rank-0 processes wait at barrier
            if MPIUtils.size() > 1:
                dist.barrier()
            self._reference_validation_receipt(epoch, None)
            return {"complete": False, "reason": "nonzero_rank"} if trajectory_update_id is not None else None
        if not hasattr(self, '_val_loader'):
            try:
                self._val_loader = self.datamodule.val_dataloader(max_samples=30)
                log.info("Val loader created: %s (%d samples)",
                         "OK" if self._val_loader else "None",
                         len(self._val_loader.dataset) if self._val_loader else 0)
            except Exception as e:
                log.info("WARNING: val_dataloader failed: %s", e)
                self._val_loader = None
        if self._val_loader is None:
            if MPIUtils.size() > 1:
                dist.barrier()
            self._reference_validation_receipt(epoch, {
                "val_loss": float("nan"), "observed_count": 0,
                "expected_count": 0, "complete": False,
            })
            return {"complete": False, "reason": "val_loader_unavailable"} if trajectory_update_id is not None else None
        # Use the unwrapped model for single-rank eval
        model = self.net.module if isinstance(self.net, DDP) else self.net
        model.eval()
        val_losses = []
        validation_error = None
        try:
            for batch in self._val_loader:
                batch = self.send_to_device(batch)
                x, gt_diff, interval = batch
                with self.autocast_context_manager():
                    pred_diff = self.forward_train(x, self.variables, interval)
                    loss_dict = lat_weighted_mse(
                        pred_diff, gt_diff, self.variables, self.lat,
                        weighted=self.weighted_loss, weight_dict=WEIGHT_DICT,
                    )
                val_losses.append(loss_dict["w_mse_aggregate"].item())
        except Exception as e:
            log.info("WARNING: validation step failed: %s", e)
            validation_error = str(e)
        model.train()
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
        expected_val_samples = len(self._val_loader)
        val_complete = (
            len(val_losses) == expected_val_samples
            and bool(np.isfinite(avg_val_loss))
            and all(bool(np.isfinite(value)) for value in val_losses)
        )
        if initial_trajectory_validation:
            log.info(
                "TRAJECTORY_INITIAL_VAL update=%d val_loss=%.17g (n=%d/%d complete=%d)",
                trajectory_update_id, avg_val_loss, len(val_losses), expected_val_samples,
                int(val_complete),
            )
        else:
            log.info(
                "VAL_LOSS epoch=%d val_loss=%.17g (n=%d/%d complete=%d)",
                epoch, avg_val_loss, len(val_losses), expected_val_samples,
                int(val_complete),
            )
        if MPIUtils.size() > 1:
            dist.barrier()
        self._reference_validation_receipt(epoch, {
            "val_loss": avg_val_loss, "observed_count": len(val_losses),
            "expected_count": expected_val_samples, "complete": bool(val_complete),
        })
        if trajectory_update_id is not None:
            return {
                "epoch": epoch,
                "complete": bool(val_complete),
                "observed_count": len(val_losses),
                "expected_count": expected_val_samples,
                "val_loss": avg_val_loss,
                "error": validation_error,
            }
        return avg_val_loss

    def _rebuild_ddp(self, new_bucket_cap_mb):
        """Rebuild DDP with a new bucket_cap_mb. Must be called outside backward pass."""
        if not isinstance(self.net, DDP):
            return
        module = self.net.module
        old_cap = self._bucket_cap_mb
        self._rebuild_ok = True
        _entry_t0 = time.perf_counter()
        _verify = os.environ.get("STORMER_REBUILD_VERIFY", "1") == "1"
        _problem = rebuild_invariants.boundary_problem(self.accumulate_grad_batches)
        if _problem is not None:
            # Refuse (symmetrically: accum is rank-symmetric) instead of raising into a collective.
            self._rebuild_ok = False
            log.error("REBUILD_REFUSED: %s", _problem)
            return
        _impl = os.environ.get("STORMER_REBUILD_IMPL", "fast")
        _phases = {}
        _init_sync = True
        _rebuild_t0 = time.perf_counter()
        self._bucket_cap_mb = new_bucket_cap_mb
        if _impl == "fast":
            # Device-side fingerprint (not the host SHA-256 copy of the whole model and Adam state), and no
            # parameter broadcast when every rank already holds identical state; same process group,
            # broadcast_buffers and all other DDP arguments (see rebuild_invariants.fast_rewrap).
            self.net, _info = rebuild_invariants.fast_rewrap(
                module, self.optimizer, DDP,
                dict(device_ids=[DEVICE_ID], output_device=DEVICE_ID, bucket_cap_mb=new_bucket_cap_mb,
                     mixed_precision=self._ddp_mixed_precision),
                MPIUtils.comm_world() if (MPIUtils.size() > 1 and MPIUtils.is_mpi()) else None,
                verify=_verify)
            _phases, _init_sync = _info["phases"], _info["init_sync"]
            _before, _after, _digest_ok = _info["before"], _info["after"], _info["ok"]
        else:
            _t_ph = time.perf_counter()
            _before = rebuild_invariants.state_digest(module, self.optimizer) if _verify else None
            _phases["digest_before"] = time.perf_counter() - _t_ph
            _rebuild_t0 = time.perf_counter()
            _t_ph = time.perf_counter()
            self.net = DDP(
                module,
                device_ids=[DEVICE_ID],
                output_device=DEVICE_ID,
                bucket_cap_mb=new_bucket_cap_mb,
                mixed_precision=self._ddp_mixed_precision,
            )
            _phases["ddp_init"] = time.perf_counter() - _t_ph
        # Re-register comm hook on new DDP wrapper.
        # BUGFIX (LIVEFLOW-STATUS.md sec 23.1): this closure previously OMITTED the
        # _comm_sync_probe barrier that the setup_model hook has, so after any rebuild the
        # straggler decomposition (probe=1) silently reverted to measuring transfer+skew.
        # Same hook (and dft_ai.comm.all_reduce region) as setup: a rebuild must not change what
        # the analyzer sees.  Previously the rebuilt hook had no region and was skipped under
        # DFTracer-only, so the comm signal vanished exactly when the action landed.
        if self._comm_hook_wanted:
            self.net.register_comm_hook(
                self._comm_bucket_ms,
                comm_hook.make_hook_for_mode(dft_ai.comm.all_reduce, self._comm_sync_probe))
        if _verify:
            if _impl != "fast":
                _t_ph = time.perf_counter()
                _after = rebuild_invariants.state_digest(self.net.module, self.optimizer)
                _phases["digest_after"] = time.perf_counter() - _t_ph
                _digest_ok = (_after == _before)
            if not _digest_ok:
                self._rebuild_ok = False
                log.error("REBUILD_INVARIANT_FAILED: state digest changed across DDP rebuild "
                          "(%s -> %s)", _before[:12], _after[:12])
            else:
                log0("REBUILD_INVARIANT_OK: state digest %s", _after[:12])
        log0("COMM_OPT: rebuilt DDP bucket_cap_mb=%.0f→%.0f", old_cap, new_bucket_cap_mb)
        # Rank-local wall of the whole re-wrap (fingerprint/digest checks included: they are part of what a live
        # rebuild costs).  The measured input to a declared STORMER_BCAP_REBUILD_COST_S; a diagnostic line,
        # never consumed by the controller.  REBUILD_PHASES splits it (device-synchronised laps).
        _cost = time.perf_counter() - _rebuild_t0
        log.info("REBUILD_COST_S rank=%d from=%.0f to=%.0f seconds=%.6f",
                 MPIUtils.rank(), old_cap, new_bucket_cap_mb, _cost)
        log.info("REBUILD_PHASES rank=%d impl=%s init_sync=%d wall_s=%.6f cost_s=%.6f %s", MPIUtils.rank(), _impl,
                 int(_init_sync), time.perf_counter() - _entry_t0, _cost,
                 " ".join("%s=%.6f" % kv for kv in _phases.items()))
        self._first_step_after_rebuild = True

    def _comm_optimize_window(self):
        """Evaluate comm metrics for the completed window and tune knobs."""
        if not self._comm_optimize or not self._comm_window_steps:
            return
        if self._comm_cooldown > 0:
            self._comm_cooldown -= 1
            if MPIUtils.rank() == 0:
                log.info("COMM_OPT: cooldown=%d, skipping", self._comm_cooldown)
            return

        # Skip first window (MIOpen warmup distorts metrics)
        if not hasattr(self, '_comm_window_count'):
            self._comm_window_count = 0
        self._comm_window_count += 1
        if self._comm_window_count == 1:
            self._comm_window_steps.clear()
            if MPIUtils.rank() == 0:
                log.info("COMM_OPT: skipping first window (warmup)")
            return

        n = len(self._comm_window_steps)
        avg_step = sum(r[0] for r in self._comm_window_steps) / n
        avg_comm = sum(r[1] for r in self._comm_window_steps) / n
        avg_buckets = sum(r[2] for r in self._comm_window_steps) / n
        losses = [r[3] for r in self._comm_window_steps if not (r[3] != r[3])]  # filter NaN
        avg_loss = sum(losses) / len(losses) if losses else float("nan")
        comm_frac = avg_comm / avg_step if avg_step > 0 else 0
        self._comm_window_steps.clear()

        if MPIUtils.rank() == 0:
            log.info(
                "COMM_OPT: window avg_step=%.0fms avg_comm=%.0fms comm_frac=%.2f "
                "buckets=%.0f bucket_cap=%.0f accum=%d avg_loss=%.4f",
                avg_step, avg_comm, comm_frac, avg_buckets,
                self._bucket_cap_mb, self.accumulate_grad_batches, avg_loss,
            )

        if comm_frac < 0.70:
            if MPIUtils.rank() == 0:
                log.info("COMM_OPT: comm_frac=%.2f < 0.70, no action needed", comm_frac)
            return

        # Stage 1: Increase bucket_cap_mb (pure systems, no semantics impact)
        if avg_buckets > 15 and self._bucket_cap_mb < 300:
            new_cap = min(self._bucket_cap_mb * 4, 300)
            self._rebuild_ddp(new_cap)
            self._comm_cooldown = 1  # wait 1 window to observe effect
            return

        # Stage 2: Increase accumulation (higher leverage, changes sync cadence)
        if self.accumulate_grad_batches < 4:
            old_accum = self.accumulate_grad_batches
            self.accumulate_grad_batches = min(old_accum + 1, 4)
            # Clear stale gradients from previous accum window to prevent
            # the next no-zero-grad step from accumulating on old gradients
            self.optimizer.zero_grad(set_to_none=True)
            self._comm_cooldown = 1
            log0("COMM_OPT: accumulate_grad_batches=%d→%d", old_accum, self.accumulate_grad_batches)
            return

        if MPIUtils.rank() == 0:
            log.info("COMM_OPT: knobs exhausted (bucket_cap=%.0f, accum=%d)",
                     self._bucket_cap_mb, self.accumulate_grad_batches)

    @dft_ai.pipeline.train
    def train(self):
        _mot.identify(MPIUtils.rank(), MPIUtils.size())
        self._setup_optimizer_context()
        # Job-wide, rank-symmetric gate for the plan-apply collective. The optimizer context is
        # active on only a SUBSET of ranks, but DFOPTIMIZER_ENABLE is identical on every rank, so
        # gating the collective on it keeps all ranks in lockstep (and leaves LiveIO/no-optimizer
        # runs byte-identical: no extra collective when it is off).
        self._optimizer_active = os.environ.get("DFOPTIMIZER_ENABLE", "0") == "1"

        # RANK-SYMMETRIC ACTUATOR ENABLES (LIVEFLOW-STATUS sec 78). These were previously set ONLY
        # inside `if self._optimizer_ctx and not self._optimizer_ctx._noop:` below, i.e. only on
        # ranks that hold an optimizer context. Under CTX_SCOPE=per_node that is 4 ranks of 16, and
        # the actuator gates read them as `getattr(self, "_tune_bcap_live", False)` -- so on the
        # other twelve the attribute did not exist and silently defaulted to False.
        #
        # `bucket_cap_mb` actuates via `_rebuild_ddp`, whose `DDP.__init__` runs
        # `_verify_param_shape_across_processes`, a WORLD collective. Four ranks entering it while
        # twelve skip it is an unconditional deadlock -- exactly the stack in sec 68 and sec 78
        # (ranks 0/4/8/12 blocked, the rest absent). The MPI consensus was working the whole time;
        # what was asymmetric was permission to act on it.
        #
        # Every expression here depends only on env vars and MPIUtils.size(), both identical on
        # every rank, so computing them unconditionally is what they always should have been. The
        # assignments inside the context block below are now redundant but harmless (same values).
        _sym_multi = MPIUtils.size() > 1
        self._tune_comm_live = os.environ.get("STORMER_TUNE_COMM", "0") == "1" and _sym_multi
        self._tune_bcap_live = os.environ.get("STORMER_TUNE_BUCKET_CAP", "0") == "1" and _sym_multi
        self._tune_decode_live = (os.environ.get("STORMER_TUNE_DECODE", "0") == "1" and _TWOSTAGE)
        # The shared-memory decode limit (sec 82) is only shared if workers are FORKED. Under
        # 'spawn' each worker re-imports train.py and builds its OWN _DECODE_LIMIT, so the parent's
        # store would reach nobody and decode would silently stop applying -- a wrong-answer bug,
        # not a crash. Assert rather than assume.
        _start_method = _mp.get_start_method(allow_none=True) or "fork"
        if self._tune_decode_live and _start_method != "fork":
            raise RuntimeError(
                f"decode tuning needs the 'fork' start method for the shared _DECODE_LIMIT; "
                f"got '{_start_method}'. Pin multiprocessing_context='fork' on the DataLoader "
                f"or disable STORMER_TUNE_DECODE.")
        if MPIUtils.rank() == 0:
            log.info("ACTUATOR_ENABLES (rank-symmetric): comm=%s bcap=%s decode=%s start_method=%s",
                     self._tune_comm_live, self._tune_bcap_live, self._tune_decode_live,
                     _start_method)

        # Register knobs with optimizer (if context is active)
        if self._optimizer_ctx and not self._optimizer_ctx._noop:
            try:
                from dfoptimizer.runtime import knob
                from dfoptimizer.runtime.knob import knob_def_from_dict
                _knob_max_workers = int(os.environ.get("STORMER_KNOB_MAX_WORKERS", "8"))
                # INPUT knobs (num_workers / prefetch_factor) actuate via a mid-run DataLoader
                # worker respawn (_apply_reconfigure). Under Ulysses SP that respawn can hang a rank
                # and break the SP all_to_all lockstep, so it is gated (STORMER_TUNE_INPUT, default
                # on) to allow isolating the comm loop, which actuates collective-free.
                _tune_input = os.environ.get("STORMER_TUNE_INPUT", "1") == "1"
                knob_defs = {}
                _current = {}
                if _tune_input:
                    knob_defs["num_workers"] = knob(
                        default=self.args.num_workers,
                        range=(0, _knob_max_workers),
                        type=int,
                        responds_to={
                            "reader_parallelism": {
                                "direction": "increase",
                                "step_mode": "evidence",
                                "min_persistence": 2,
                                "cooldown_windows": 3,
                                "apply_when": "window_boundary",
                            },
                        },
                    )
                    knob_defs["prefetch_factor"] = knob(
                        default=self.args.prefetch_factor,
                        range=(1, 16),
                        type=int,
                        responds_to={
                            "dataloader_prefetch": {
                                "direction": "increase",
                                "step_mode": "evidence",
                                "min_persistence": 2,
                                "cooldown_windows": 3,
                                "apply_when": "window_boundary",
                            },
                        },
                    )
                    _current["num_workers"] = self.args.num_workers
                    _current["prefetch_factor"] = self.args.prefetch_factor
                # COMM knob: accumulate_grad_batches is the LIVE comm-domain knob. Raising it makes
                # DDP sync gradients less often (fewer all_reduces), so it tunes the comm domain
                # WITHOUT any collective to apply (unlike bucket_cap, whose mid-run DDP rebuild
                # broadcast interleaves with the Ulysses SP all_to_all and deadlocks). Opt-in
                # (STORMER_TUNE_COMM=1), multi-rank only. Domain=comm is inferred by CLBOPlanner.
                _tune_comm = (os.environ.get("LIVEFLOW_PRESET") != "liveflow-v1"
                              and os.environ.get("STORMER_TUNE_COMM", "0") == "1" and MPIUtils.size() > 1)
                self._tune_comm_live = _tune_comm
                if _tune_comm:
                    knob_defs["accumulate_grad_batches"] = knob(
                        default=int(self.accumulate_grad_batches),
                        range=(int(os.environ.get("STORMER_KNOB_ACCUM_MIN", "1")),
                               int(os.environ.get("STORMER_KNOB_ACCUM_MAX", "8"))),
                        type=int,
                        responds_to={
                            "comm_optimization": {
                                "direction": "increase",
                                "step_mode": "evidence",
                                "min_persistence": 1,
                                "cooldown_windows": 3,
                                "apply_when": "window_boundary",
                            },
                        },
                    )
                    _current["accumulate_grad_batches"] = int(self.accumulate_grad_batches)
                # COMM knob #2: bucket_cap_mb. MATH-PRESERVING (changes only WHEN gradients are
                # exchanged, never their values), unlike accum which changes the effective batch.
                # Measured authority is large: 25 -> 400 MB is -36% wall / -66% comm at 4N
                # (LIVEFLOW-STATUS.md sec 25.1). Applying it needs a DDP REBUILD, whose broadcast
                # previously deadlocked against the Ulysses SP all_to_all. The fix is lockstep, not
                # luck: cuda.synchronize() drains every rank's in-flight SP collective, then an MPI
                # (CPU) barrier aligns ranks, and only THEN does the NCCL broadcast run -- so no SP
                # collective can be in flight anywhere when the rebuild happens. See _apply_knobs.
                # Opt-in (STORMER_TUNE_BUCKET_CAP=1), multi-rank only. step_mode=multiply because the
                # useful range is geometric (25/50/100/200/400) and the Amdahl evidence formula
                # computes a PARALLELISM target, which is meaningless for a size in MB.
                _tune_bcap = (os.environ.get("STORMER_TUNE_BUCKET_CAP", "0") == "1"
                              and MPIUtils.size() > 1)
                self._tune_bcap_live = _tune_bcap
                if _tune_bcap:
                    # LEVELS ARE DECLARED, NOT DERIVED.  With only `range`, the planner's grid is six
                    # linear levels (25,100,175,250,325,400), each a collective DDP rebuild; the geometric
                    # ladder below is the one the step comment above documents.  A cost is declared only
                    # from a measured rebuild (REBUILD_COST_S in the log), never guessed: unset = 0.0, the
                    # historical registration.
                    _bcap_lo = int(os.environ.get("STORMER_KNOB_BCAP_MIN", "25"))
                    _bcap_hi = int(os.environ.get("STORMER_KNOB_BCAP_MAX", "400"))
                    _bcap_values = sorted({int(v) for v in os.environ.get(
                        "STORMER_KNOB_BCAP_VALUES", "").replace(",", " ").split()
                        if _bcap_lo <= int(v) <= _bcap_hi} | {int(self._bucket_cap_mb)})
                    knob_defs["bucket_cap_mb"] = knob(
                        default=int(self._bucket_cap_mb),
                        range=(_bcap_lo, _bcap_hi),
                        values=(_bcap_values if os.environ.get("STORMER_KNOB_BCAP_VALUES") else None),
                        cost_seconds=float(os.environ.get("STORMER_BCAP_REBUILD_COST_S", "0") or 0.0),
                        type=int,
                        responds_to={
                            "comm_optimization": {
                                "direction": "increase",
                                "step_mode": "multiply",
                                # REQUIRED for multiply mode: planner.py:417 does
                                # `old_value * response.step`, and KnobResponse.step defaults to
                                # None -> TypeError. This is the FIRST multiply-mode knob in the
                                # codebase, so that path had a latent crash nothing had hit.
                                # step=2 gives the intended geometric ramp 25->50->100->200->400.
                                "step": 2,
                                "min_persistence": 1,
                                "cooldown_windows": 3,
                                "apply_when": "window_boundary",
                            },
                        },
                    )
                    _current["bucket_cap_mb"] = int(self._bucket_cap_mb)
                # THIRD DOMAIN: decode. Only meaningful with STORMER_TWOSTAGE=1 against a
                # gzip+chunked dataset, where stage B (zlib inflate) is separately parallel from
                # stage A (raw chunk I/O). Measured effect on real ERA5: 463 -> 86 ms/sample from
                # 1 -> 16 threads. Math-safe: decoding is bit-identical regardless of width.
                # `decode_threads` is already mapped to domain "decode" in clbo_planner._DOMAIN_OF.
                _tune_dec = (os.environ.get("STORMER_TUNE_DECODE", "0") == "1" and _TWOSTAGE)
                self._tune_decode_live = _tune_dec
                if _tune_dec:
                    knob_defs["decode_threads"] = knob(
                        default=int(os.environ.get("STORMER_DECODE_THREADS", "1") or 1),
                        range=(int(os.environ.get("STORMER_KNOB_DEC_MIN", "1")),
                               int(os.environ.get("STORMER_KNOB_DEC_MAX", "16"))),
                        type=int,
                        responds_to={
                            # DEDICATED tag from stormer_xlayer.decode_pressure.v1. It must NOT be
                            # `reader_parallelism`: that is num_workers' tag, and sharing it makes
                            # decode pressure and input pressure indistinguishable to the planner,
                            # which then fires BOTH knobs on ONE signal. Requires
                            # DFANALYZER_PRESET=stormer_xlayer (only that preset defines the
                            # reader_decode layer the rule reads).
                            "decode_parallelism": {
                                "direction": "increase",
                                "step_mode": "multiply",
                                "step": 2,
                                "min_persistence": 1,
                                "cooldown_windows": 2,
                                "apply_when": "window_boundary",
                            },
                        },
                    )
                    _current["decode_threads"] = int(os.environ.get("STORMER_DECODE_THREADS", "1") or 1)
                built_defs = {
                    k: knob_def_from_dict(f"stormer.{k}", v, target_function="stormer.make_loader")
                    for k, v in knob_defs.items()
                }
                # LAYOUT knobs (liveflow-v1 preset): rank-local, math-neutral read-path knobs, defaults
                # = Original. Delivery mode is NOT registered for StorMer: its samples are MB-scale, so
                # the delivery family is declared and measured (stall.delivery) but has no actuator here
                # (the planner logs an exposure without a registered actuator as a finding, not a veto).
                self._layout_live = False
                if os.environ.get("LIVEFLOW_PRESET") == "liveflow-v1" and _tune_input:
                    from dfoptimizer.runtime.delivery_knobs import layout_delivery_knob_defs
                    import liveflow_layout as _ll
                    _lay = _ll.get_layout()
                    built_defs.update(layout_delivery_knob_defs(
                        "stormer", current={"transfer_mb": _lay["transfer_mb"], "read_threads": _lay["read_threads"]},
                        delivery=False))
                    _current["layout.transfer_mb"] = _lay["transfer_mb"]
                    _current["layout.read_threads"] = _lay["read_threads"]
                    self._layout_live = True
                self._optimizer_ctx.register_knobs(
                    "stormer.make_loader",
                    built_defs,
                    current_values=_current,
                )
                log0("Registered knobs: num_workers=%s, prefetch_factor=%s%s",
                     self.args.num_workers, self.args.prefetch_factor,
                     (", accumulate_grad_batches=%d" % int(self.accumulate_grad_batches)) if _tune_comm else "")
            except Exception as e:
                log0("WARNING: knob registration failed: %s", e)

        train_loader, train_sampler = self.make_loader(epoch=0)
        self.n_steps = len(train_sampler)
        self.scheduler = self.configure_scheduler(n_steps=self.n_steps)
        # This happens after optimizer + scheduler construction and before any
        # training window opens.  It is intentionally outside the per-batch
        # timing region below.
        self._initialize_trajectory_retention()
        if self._trajectory_retention is not None:
            self._capture_trajectory_state(0, epoch=None, kind="initial")
            initial_validation = self._run_trajectory_validation(-1, 0, initial=True)
            self._record_trajectory_validation(0, initial_validation)
        log0("Will run %s steps", self.n_steps)

        # Window manages analysis boundaries for the streaming pipeline.
        # Emits window.start/window.stop control events at the configured
        # cadence so the analyzer knows when to drain and analyze.
        _window = None
        _n_steps_eff = self.max_training_step if self.max_training_step > 0 else self.n_steps
        # Shared v1 sizing is owned by Window; explicit overrides serve legacy runs.
        _window_every_n = int(os.environ.get("STORMER_WINDOW_EVERY_N", "0") or 0) or 20
        try:
            from dfoptimizer.runtime import Window
            # HORIZON CONTRACT (liveflow_shm/HORIZON-CONTRACT.md). epochs_total and steps_per_epoch
            # are the two facts only the application knows; steps_per_epoch is _n_steps_eff (the
            # max_training_step cap, not the raw sampler length), matching the cadence derivation
            # above so windows_per_epoch (derived at emit time from the live cadence, never stored)
            # is arithmetically consistent with how many windows this run actually produces.
            _window = Window(
                PerfTracer.get_instance(),
                max_every_n=self.n_steps,
                epochs_total=self.epochs,
                steps_per_epoch=_n_steps_eff,
            )
            _window.update_cadence(_window_every_n)
            _window_every_n = _window.every_n
            self._loop_window = _window
            log0("Window initialized (every_n=%s)", _window.every_n)
        except Exception as e:
            log0("WARNING: Window init failed (no streaming analysis): %s", e)
        if self._trajectory_retention is not None and _window is None:
            raise RuntimeError("trajectory state v1 requires a Window so epoch-end snapshots follow Window.flush")

        # ---- EXPOSURE instrument (liveflow_shm/EXPOSURE-PORT-DESIGN.md) --------------------------
        # Without this, the agent decides about StorMer's input side from dur_sum, which is
        # per-operation service time summed across threads. It has been measured to move OPPOSITE
        # to wall when a stage is relieved, so it cannot identify a bottleneck.
        # Always-on, tiny (<= _window_every_n floats, cleared every window): the raw material for
        # _sync_input_exposure_window's cross-node reduction (sec 16). Independent of the debug-only
        # STORMER_STEP_FETCH buffer below (`_sf_fetch`), which stays gated as before.
        self._job_exposure_fetch_buf = []
        _stall_acct = None
        _emit_stall = None
        try:
            import sys as _sys
            for _p in (os.environ.get("LIVEFLOW_SHM_PYTHONPATH", ""),):
                if _p and _p not in _sys.path:
                    _sys.path.insert(0, _p)
            from exposure import StallAccount, emit_stall_events
            _stall_acct = StallAccount(exact=False, delivery=True)   # ratio split (storage/decode), see _STAGE_A_MS; delivery is measured

            def _emit_stall_events(n_steps):
                """One event per domain per window per rank, from the MAIN process.

                Accumulated, not per stall: per-stall emission would put tracer cost on the
                critical path in proportion to how badly the loader is starving, and would give
                dur_sum several events per window to sum, reintroducing the double count that
                makes every other time layer unusable.
                """
                inst = PerfTracer.get_instance()

                def _emit(name, cat, ms):
                    if ms <= 0:
                        return
                    dur_us = int(ms * 1000)
                    inst.log_event(name=name, cat=cat, start_time=inst.get_time() - dur_us,
                                   duration=dur_us, int_args={}, float_args={}, string_args={})

                emit_stall_events(_emit, _stall_acct.drain(), n_steps)

            _emit_stall = _emit_stall_events
            log0("EXPOSURE instrument armed (split is RATIO-ESTIMATED, not the halo3d queue test)")
        except Exception as e:
            log0("WARNING: exposure instrument unavailable: %s", e)

        def _emit_exposed_comm(step, ms, epoch):
            """One event per step, event family `comm.exposed`, step-tagged via int_args - the
            non-overlapped tail of the last bucket's all-reduce (comm_hook.compute_exposed_comm_ms).
            No-op below a sub-microsecond floor so a genuinely-zero (fully-hidden) step emits
            nothing rather than a degenerate zero-duration event."""
            if ms <= 0:
                return
            inst = PerfTracer.get_instance()
            dur_us = int(ms * 1000)
            # BUGFIX (2026-09-27, caught by the fail-fast validation run): the bound `log_event`
            # requires `int_args: Dict[str, Tuple[int, int]]` -- a (tag_type, value) pair (`_TAG_INT`,
            # module top) -- not a bare int. A bare int crashed on the very first call, on every rank,
            # with the fail-fast fix correctly aborting instead of hanging.
            # 2026-09-28 (judge batch 5 G4 root cause #2): this event MUST carry an explicit
            # epoch tag in the SAME convention as the dft_ai-annotated events it shares a window
            # with (`dft_ai.update(epoch=epoch, step=step)`, zero-indexed). Without one, the
            # analyzer-side window stamper fills the epoch from the horizon contract's one-based
            # control epoch (`_window.set_epoch(epoch + 1)`), so every window contained epoch 0
            # (fetch.iter/compute/data/device) AND epoch 1 (this event) among the authoritative
            # main-thread rows with a step -- `epoch_nunique == 2` in every window of every run
            # since this emitter was added, so the single-epoch fact-engine gate
            # (`window_control_vector_invalid`) suppressed BOTH rules in all 15/15 windows.
            inst.log_event(name="comm.exposed", cat="comm", start_time=inst.get_time() - dur_us,
                           duration=dur_us,
                           int_args={"epoch": _TAG_INT(epoch), "step": _TAG_INT(step)},
                           float_args={}, string_args={})

        def _emit_total_comm(step, ms, epoch):
            """One event per step, family `comm.total` (STORMER-LOOP-DESIGN sec 17 v3, orchestrator
            decision 2026-09-27): the SUM of every bucket's own [ready, end] span this step (not just
            the last bucket's non-overlapped tail, which is `comm.exposed`). Domain2's exposed-comm
            validation found exposed comm tiny (1.8ms cap25/W16 vs 7.6ms cap400/W16, HIGHER at cap400)
            while TOTAL comm was 173 vs 81ms -- Step 0c's cap400 gain looks like comm/compute
            INTERFERENCE (overlapped RCCL slows backward; fewer, larger buckets interfere less), not
            an exposed-tail effect. Total comm is an UPPER BOUND on what a bucket_cap change can
            possibly be worth (it can never save more than the whole comm cost, exposed or not), not
            a claim that all of it is recoverable -- see `DFOPTIMIZER_COMM_EXPOSURE_SOURCE=total` and
            JOINT_VERIFY's role as the arbiter (clbo_planner.py's docstring, cbandit_planner.py).
            No-op below a sub-microsecond floor, same convention as `_emit_exposed_comm`."""
            if ms <= 0:
                return
            inst = PerfTracer.get_instance()
            dur_us = int(ms * 1000)
            # 2026-09-28 (judge batch 5 G4 root cause #2): this event MUST carry an explicit
            # epoch tag in the SAME convention as the dft_ai-annotated events it shares a window
            # with (`dft_ai.update(epoch=epoch, step=step)`, zero-indexed). Without one, the
            # analyzer-side window stamper fills the epoch from the horizon contract's one-based
            # control epoch (`_window.set_epoch(epoch + 1)`), so every window contained epoch 0
            # (fetch.iter/compute/data/device) AND epoch 1 (this event) among the authoritative
            # main-thread rows with a step -- `epoch_nunique == 2` in every window of every run
            # since this emitter was added, so the single-epoch fact-engine gate
            # (`window_control_vector_invalid`) suppressed BOTH rules in all 15/15 windows.
            inst.log_event(name="comm.total", cat="comm", start_time=inst.get_time() - dur_us,
                           duration=dur_us,
                           int_args={"epoch": _TAG_INT(epoch), "step": _TAG_INT(step)},
                           float_args={}, string_args={})

        _force_rebuilds = {}
        for _tok in os.environ.get("STORMER_FORCE_REBUILD", "").split(","):
            if _tok.strip():
                _fe, _fs, _fc = _tok.strip().split(":")
                _force_rebuilds[(int(_fe), int(_fs))] = float(_fc)
        for epoch in dft_ai.pipeline.epoch.iter(
            range(self.epochs),
            include_iter=False,
        ):
            dft_ai.pipeline.epoch.start(metadata=True)
            pbar = None
            step = 0
            _epoch_update_start = self._optimizer_update_count
            _epoch_reported_steps = 0
            try:
                log0(f"Epoch: {epoch}")
                # Horizon contract: stamp epoch_index (one based) BEFORE the loader is built, so
                # the boundary that carries a staging decision already knows which epoch it is in.
                if _window is not None:
                    _window.set_epoch(epoch + 1)
                train_loader, train_sampler = self.make_loader(epoch=epoch)

                self.net.train()

                if self.enable_progress_bar:
                    import tqdm.auto as tqdm

                    pbar = tqdm.tqdm(
                        desc=f"epoch {epoch}",
                        total=len(train_loader),
                        position=MPIUtils.rank(),
                        leave=True,
                        disable=MPIUtils.rank() != 0,
                        bar_format="{l_bar}{bar}|{n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                    )

                step = 1
                dft_ai.update(epoch=epoch, step=step)
                collate_profiler.update(epoch=epoch, step=step)

                if self.enable_progress_bar:
                    from tqdm.contrib.logging import logging_redirect_tqdm
                    logging_ctx = logging_redirect_tqdm
                else:
                    from utils.context import noop_context
                    logging_ctx = noop_context

                # Comm timing accumulators for epoch summary
                _ct_steps = []
                # STORMER_STEP_WALL=1: rank-local host wall per step + per-step loss, independent of
                # the (blocking) comm probe, so probe-off arms have a step-time and loss series.
                # Steps include fetch (interval between consecutive step ends); step 1 of the epoch
                # starts at loop entry.  Diagnostic/Step-0 only; not the per-rep reporting seam.
                _sw_on = os.environ.get("STORMER_STEP_WALL", "0") == "1"
                _sw_wall, _sw_loss, _sw_prev = [], [], time.perf_counter()
                # Independent monotonic start for the loop-entry-to-N endpoint. Its
                # end is captured after the Nth optimizer update's charged control
                # work, before the quarantine fetch. The full-loop end is separate.
                _lc_elapsed_start = time.monotonic()
                _lc_wall_start = time.time()
                _lc_n_updates = int(os.environ.get("STORMER_LONG_CELL_TOTAL_UPDATES") or 0)
                _lc_n_end = None
                _lc_abort_seen = False
                # STORMER_STEP_FETCH=1 (needs STORMER_STEP_WALL=1): per-step consumer-boundary wait, measured from
                # the previous step's end (AFTER its loss sync) to batch arrival, so wall - fetch is the post-fetch
                # work (host copy + fwd/bwd/all-reduce/optimizer + waiting for slower ranks in the all-reduce).
                # STORMER_STEP_IO=1: cumulative /proc/<pid>/io rchar/read_bytes of the DataLoader worker
                # processes sampled once per step (pids discovered right after the first batch).
                _sf_on = _sw_on and os.environ.get("STORMER_STEP_FETCH", "0") == "1"
                _sio_on = _sw_on and os.environ.get("STORMER_STEP_IO", "0") == "1"
                # STORMER_STEP_EXPOSED_COMM=1 (needs STORMER_STEP_WALL=1 and STORMER_COMM_TIMING=1
                # or STORMER_COMM_OPTIMIZE=1, event probe): per-step exposed_comm_ms (the tail of
                # the last bucket's all-reduce that runs after backward finishes; see
                # comm_hook.compute_exposed_comm_ms). 0.0 for a step where it wasn't computed
                # (e.g. comm timing off that step), never absent, so the series stays step-aligned.
                _se_on = _sw_on and os.environ.get("STORMER_STEP_EXPOSED_COMM", "0") == "1"
                _sf_fetch = []
                _se_exposed = []
                _sio_r, _sio_b, _sio_pids, _sio_n = [], [], [], []

                with logging_ctx():
                    _t_fetch0 = time.time()
                    # This monotonic interval begins after the preceding batch
                    # step's control work (or at epoch start), includes consumer
                    # fetch, training, window close, in-loop progress handling,
                    # and synchronous apply work; it ends before validation.
                    # It deliberately excludes trajectory-state retention pauses;
                    # the ID is global across epochs and names a batch step, not
                    # necessarily an optimizer update under accumulation.
                    _per_rep_begin_ns = time.perf_counter_ns()
                    # WHERE THE WINDOW BEGINS. `for batch in ...` has already blocked on the fetch
                    # by the time the body runs, so a window opened inside the body excludes the
                    # fetch that opened it while the stall drained at its close includes it. On
                    # `stgi_capacity_gate1` that put 11.96 s of stall inside a 36.42 s wall which
                    # did not contain it and the domains summed to 131 percent of the window
                    # (STATUS-ATTRIBUTION-FIX-2026-09-08.md, defect C2). This is the instant the
                    # previous step ended, on the tracer's own clock.
                    _win_open_at = _window.now() if _window is not None else None
                    _fetch_iter = dft_ai.dataloader.fetch.iter(train_loader)
                    # Explicit GPU negative control only. Never used by a
                    # paper arm: truncate rank 0 after two batches so the
                    # next agreement must fail before another backward.
                    if os.environ.get("STORMER_LOOP_EXIT_NEGATIVE_CONTROL", "0") == "1":
                        if self._loop_exit_consensus is None:
                            raise RuntimeError("negative control requires loop-exit consensus")
                        if MPIUtils.rank() == 0:
                            import itertools
                            _fetch_iter = itertools.islice(_fetch_iter, 2)
                    _mot.start_gpu_capture()
                    _step_iter = _mot.observe(wrap_step_iterator(_fetch_iter, self._loop_exit_consensus), epoch)
                    for batch, _is_padding_step in _step_iter:
                        if _is_padding_step:
                            raise RuntimeError("PADDING_BATCH_FORBIDDEN: training would alter trajectory")
                        _fp("batch.received")   # loader returned an assembled batch
                        # UNCONDITIONAL now. This is the consumer-boundary exposure: the gap
                        # between the end of the previous step and the batch arriving. With a
                        # prefetching loader it is ~0 when the workers kept up, and it is the real
                        # stall when they did not. It used to be computed only under comm timing,
                        # so the quantity the attribution needs did not exist in a normal run.
                        _fetch_ms = (time.time() - _t_fetch0) * 1000
                        # This step's exposed wait per family, in seconds, for the window's per-step series
                        # (dfoptimizer step_series; the planner verifies a move on the targeted family's series).
                        _step_exp = {"storage": _fetch_ms / 1000.0}
                        self._job_exposure_fetch_buf.append(float(_fetch_ms))
                        if _sf_on:
                            _sf_fetch.append((time.perf_counter() - _sw_prev) * 1000.0)
                        if _sio_on:
                            import step_io
                            if not _sio_pids:
                                _sio_pids = step_io.child_pids()
                            _r, _b, _al = step_io.read_io(_sio_pids)
                            _sio_r.append(_r)
                            _sio_b.append(_b)
                            _sio_n.append(_al)
                        if _stall_acct is not None and _fetch_ms > 0:
                            _sa, _sb = _stage_times_drain()
                            _tot = _sa + _sb
                            # No stage evidence: charge storage rather than invent a split. Dropping
                            # it would understate the domain and read like a loader that kept up.
                            _frac = 1.0 if _tot <= 0 else _sa / _tot
                            # DELIVERY FAMILY: the part of the fetch wait blocked on the worker result
                            # queue is the upstream (storage/decode) wait; the REST is worker->trainer
                            # handoff mechanics on the critical path, measured as fetch - blocked wait.
                            _hs = (train_loader.take_handoff_s() if hasattr(train_loader, "take_handoff_s") else None)
                            if _hs is None:
                                _stall_acct.add_split(_fetch_ms, _frac)
                            else:
                                # Measured handoff only; every other part of the fetch interval (queue wait,
                                # peer skew at the loop-exit consensus) stays with the storage side exactly as
                                # before this family existed. Never a residual: see take_handoff_s.
                                _dms = min(_hs[1] * 1000.0, _fetch_ms)
                                _stall_acct.add_split(_fetch_ms - _dms, _frac)
                                _stall_acct.add_delivery(_dms)
                                _step_exp["storage"] = (_fetch_ms - _dms) / 1000.0
                                _step_exp["delivery"] = _dms / 1000.0

                        if self.max_training_step != -1 and step > self.max_training_step:
                            log0(f"Training step limit reached: {self.max_training_step}")
                            break

                        if _window is not None:
                            _window.start(at_time=_win_open_at)

                        if self._comm_timing or self._comm_optimize:
                            torch.cuda.synchronize()
                            _t_step0 = time.time()

                        sample_ids = None
                        if os.environ.get("LIVEFLOW_SAMPLE_LOG", "0") == "1":
                            sample_ids = batch[3]
                            batch = batch[:3]
                        batch = self.send_to_device(batch)
                        if (self._c0_bwd_ms is None and (self._comm_timing or self._comm_optimize)
                                and self._c0_active() and not getattr(self, "_c0_tried", False)):
                            self._c0_tried = True
                            try:
                                self._c0_bwd_ms = self._calibrate_compute_only_backward(batch)
                            except Exception as _exc:
                                log0("COMM_CALIBRATION failed (%r); exposed comm reads 0", _exc)
                        if comm_hook.comm_probe_mode() == "ab":
                            _ab_mode = comm_hook.ab_advance()
                            _ab_cap = comm_hook.ab_cap_now(self._bucket_cap_mb)
                            if _ab_cap != self._bucket_cap_mb:
                                # symmetric on every rank (same step counter); outside backward
                                self._rebuild_ddp(_ab_cap)
                        # Test/measurement hook (default off): STORMER_FORCE_REBUILD="epoch:step:cap,..." rebuilds
                        # DDP at that epoch-local step on every rank (same cap = a no-op rebuild), outside backward.
                        _force = _force_rebuilds.get((epoch, step))
                        if _force is not None:
                            torch.cuda.synchronize()
                            if MPIUtils.size() > 1 and MPIUtils.is_mpi():
                                MPIUtils.comm_world().Barrier()
                            _t_fr = time.perf_counter()
                            self._rebuild_ddp(float(_force))
                            if MPIUtils.size() > 1 and MPIUtils.is_mpi():
                                MPIUtils.comm_world().Barrier()
                            log.info("FORCED_REBUILD epoch=%d step=%d cap=%s wall_s=%.6f rank=%d", epoch, step,
                                     _force, time.perf_counter() - _t_fr, MPIUtils.rank())
                        _t_fs = time.perf_counter() if getattr(self, "_first_step_after_rebuild", False) else None
                        loss = self.training_step(batch, step)
                        if _t_fs is not None:
                            torch.cuda.synchronize()
                            self._first_step_after_rebuild = False
                            log.info("FIRST_STEP_AFTER_REBUILD epoch=%d step=%d seconds=%.6f rank=%d", epoch, step,
                                     time.perf_counter() - _t_fs, MPIUtils.rank())
                        if sample_ids is not None:
                            import hashlib, json
                            indices = torch.as_tensor(sample_ids).reshape(-1).tolist()
                            digest = hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()
                            print("SAMPLE_STEP " + json.dumps(dict(step=step, epoch=epoch, rank=dist.get_rank() if dist.is_initialized() else 0, sample_indices=indices, sample_id_sha256=digest, loss=float(loss), loss_repr=repr(float(loss)), grad_norm=getattr(self, '_last_grad_norm', None)), sort_keys=True), flush=True)

                        if self._comm_timing or self._comm_optimize:
                            torch.cuda.synchronize()
                            step_ms = (time.time() - _t_step0) * 1000
                            # Exposed comm (event probe only): must read the RAW event pairs
                            # BEFORE read_and_clear_bucket_ms drains them below. 0.0 in blocking/
                            # off mode. Completion-based: last bucket's `ready` event to the
                            # event recorded when backward() returned (see
                            # compute_exposed_comm_ms's docstring).
                            exposed_comm_ms = 0.0
                            if comm_hook.comm_probe_mode() in ("event", "ab"):
                                exposed_comm_ms = comm_hook.exposed_vs_reference_ms(
                                    self._bwd_only_ms, self._c0_bwd_ms)
                            if _se_on:
                                _se_exposed.append(exposed_comm_ms)
                            if comm_hook.comm_probe_mode() in ("event", "ab"):
                                _step_exp["comm"] = exposed_comm_ms / 1000.0
                            if PERFTRACER_ENABLE:
                                _emit_exposed_comm(step, exposed_comm_ms, epoch)
                            # read_and_clear_bucket_ms: floats already (blocking mode), or
                            # (start, end) CUDA event pairs (event/B1 mode) - the sync just above
                            # is what makes reading the events here safe. Same call either mode.
                            _bucket_ms = comm_hook.read_and_clear_bucket_ms(self._comm_bucket_ms)
                            comm_ms = sum(_bucket_ms)
                            n_buckets = len(_bucket_ms)
                            if comm_hook.comm_probe_mode() == "ab":
                                print("COMM_AB rank=%d e=%d s=%d mode=%s step_ms=%.3f fwd_ms=%.3f bwd_ms=%.3f comm_ms=%.3f nb=%d exposed_ms=%.3f barrier_ms=%.3f fetch_ms=%.3f cap=%.0f" % (
                                    MPIUtils.rank(), epoch, step, _ab_mode, step_ms, self._fwd_ms, self._bwd_ms, comm_ms, n_buckets,
                                    exposed_comm_ms, comm_hook.ab_pop_barrier_ms(), _fetch_ms, self._bucket_cap_mb), flush=True)
                            # sec 17 v3: TOTAL comm, event-probe mode only -- same exclusion the
                            # exposed-comm side already applies (blocking-probe numbers measure a
                            # different, serialized quantity and must not be mixed with these).
                            if PERFTRACER_ENABLE and comm_hook.comm_probe_mode() in ("event", "ab"):
                                _emit_total_comm(step, comm_ms, epoch)
                            fwd_ms = self._fwd_ms
                            bwd_ms = self._bwd_ms
                            # Ulysses SP all_to_all (inside forward) — the COMM pole. Opt-in
                            # CUDA-event timer (STORMER_SP_TIME_A2A=1); 0 when SP off / timer off.
                            a2a_ms = 0.0
                            if getattr(self, "_sp_size", 1) > 1:
                                import ulysses_sp
                                a2a_ms = ulysses_sp.pop_a2a_ms()
                            other_ms = step_ms - fwd_ms - bwd_ms
                            total_ms = _fetch_ms + step_ms
                            _loss_for_ct = loss if isinstance(loss, float) else float("nan")
                            _ct_steps.append((step_ms, fwd_ms, bwd_ms, comm_ms, n_buckets, other_ms, _loss_for_ct, _fetch_ms))
                            # Feed per-step data to comm optimizer (include loss for convergence tracking)
                            _loss_val = loss if isinstance(loss, float) else float("nan")
                            self._comm_window_steps.append((step_ms, comm_ms, n_buckets, _loss_val))
                            if self._comm_timing and MPIUtils.rank() != 0 and self._comm_timing_all_ranks:
                                # PER-RANK GROUND TRUTH, opt in (STORMER_COMM_TIMING_ALL_RANKS=1).
                                #
                                # The line below prints from rank 0 only, so it can validate a
                                # decomposition of the TYPICAL rank and nothing else. Our
                                # `domain_*_time_proc_crit` is the vector of the rank on the
                                # critical path through the data pipeline, and on this shape that
                                # is not rank 0: two of eight ranks stall 4.9 to 18.3 s on the
                                # loader while the other six stall 0.35 and wait the same time at
                                # the barrier instead (STATUS-ATTRIBUTION-FIX-2026-09-08.md
                                # section 2). Without this there is no instrument outside our
                                # pipeline that sees that rank at all. A DIFFERENT prefix, so every
                                # parser of the rank-0 line keeps reading exactly what it read.
                                log.info(
                                    "COMM_TIMING_RANK r=%d e=%d s=%d total=%.0f fetch=%.0f step=%.0f comm=%.0f",
                                    MPIUtils.rank(), epoch, step, total_ms, _fetch_ms, step_ms, comm_ms,
                                )
                            if MPIUtils.rank() == 0 and self._comm_timing:
                                # exposed comm = DDP all_reduce (bwd, usually hidden) + SP a2a (fwd, non-overlappable)
                                _exposed_comm = comm_ms + a2a_ms
                                log.info(
                                    "COMM_TIMING e=%d s=%d total=%.0f fetch=%.0f step=%.0f fwd=%.0f bwd=%.0f comm=%.0f(%d buckets) a2a=%.0f other=%.0f comm_pct=%.1f%% a2a_pct=%.1f%% fetch_pct=%.1f%%",
                                    epoch, step, total_ms, _fetch_ms, step_ms, fwd_ms, bwd_ms, comm_ms, n_buckets,
                                    a2a_ms, other_ms,
                                    100 * comm_ms / total_ms if total_ms > 0 else 0,
                                    100 * a2a_ms / total_ms if total_ms > 0 else 0,
                                    100 * _fetch_ms / total_ms if total_ms > 0 else 0,
                                )
                        # Reset unconditionally: the exposure measurement above needs it in
                        # every run, not only when comm timing is on.
                        _t_fetch0 = time.time()
                        # Same instant, on the tracer's clock, for the window that opens next.
                        _win_open_at = _window.now() if _window is not None else None

                        if _window is not None:
                            # Exposure into the trace BEFORE the window closes, so the window the
                            # analyzer is about to analyse carries the stall it must explain.
                            if _emit_stall is not None and _window.will_close:
                                _emit_stall(_window.pending_steps)
                            # Diagnostic only (APPLY_ACK log): the 0-based index of the step this
                            # window boundary closes, so an action's first affected step is after_step+1.
                            self._loop_step = step
                            _window.note_step_exposure(**_step_exp)
                            _window_closed = _window.stop()
                            _window_every_n = _window.every_n

                        if pbar:
                            pbar.set_postfix({"loss": loss})
                            pbar.update()

                        step += 1
                        _epoch_reported_steps += 1
                        if _sw_on:
                            _sw_now = time.perf_counter()
                            _sw_wall.append((_sw_now - _sw_prev) * 1000.0)
                            _sw_prev = _sw_now
                            _sw_loss.append(float(loss))
                        if self._trajectory_retention is not None:
                            if int(self.accumulate_grad_batches) != 1:
                                raise RuntimeError("trajectory state v1 observed a changed accumulation setting")
                            if self._last_observed_optimizer_update:
                                self._completed_updates += 1
                        _fp("step.end", step)
                        _note_progress()   # feeds the stall watchdog (module top)
                        dft_ai.update(epoch=epoch, step=step)
                        collate_profiler.update(epoch=epoch, step=step)

                        # Poll optimizer at window boundaries for mid-epoch reconfiguration
                        if _window is not None and _window_closed:
                            self._sync_apply_window(train_loader, step=step)
                            self._sync_input_exposure_window(step=step)
                            self._comm_optimize_window()
                        if self._per_rep_step_writer is not None:
                            torch.cuda.synchronize()
                            self._per_rep_logical_batch_step_id += 1
                            completion_status = (
                                "completed" if np.isfinite(float(loss)) else "nonfinite_loss_skipped"
                            )
                            self._per_rep_step_writer.write(
                                self._per_rep_logical_batch_step_id,
                                epoch,
                                step - 1,
                                (time.perf_counter_ns() - _per_rep_begin_ns) / 1_000_000.0,
                                completion_status,
                            )
                            # All control charged to this optimizer update has completed above.
                            # Record the Nth update before the quarantine fetch and update N+1.
                            if (_lc_n_updates and _lc_n_end is None
                                    and self._optimizer_update_count == _lc_n_updates):
                                _lc_n_end = time.monotonic()
                        _per_rep_begin_ns = time.perf_counter_ns()
            except Exception as _loop_exc:
                _lc_abort_seen = True
                if isinstance(_loop_exc, UnequalLoaderExhaustion):
                    try:
                        published = retain_unequal_exhaustion(
                            MPIUtils.comm_world(), rank=MPIUtils.rank(), epoch=epoch, step=step,
                            updates=self._optimizer_update_count - _epoch_update_start,
                            reported_steps=_epoch_reported_steps,
                            evidence_dir=self._loop_exit_evidence_dir, log=log)
                        if not published:
                            log.error("Unequal-exhaustion all-rank evidence could not be flushed")
                    except Exception:
                        log.exception("Unequal-exhaustion evidence Gather/publish failed")
                else:
                    log.info("STEP_SERIES_END rank=%d epoch=%d optimizer_updates=%d reported_steps=%d",
                             MPIUtils.rank(), epoch, self._optimizer_update_count - _epoch_update_start,
                             _epoch_reported_steps)
                # A rank-local exception anywhere in this per-step body (fetch,
                # forward/backward, window sync) means this rank has already
                # stopped joining whatever collective its peers are still
                # waiting inside. Letting it fall through to validate_epoch()'s
                # barrier below (which used to live in this same `finally`
                # block, reached unconditionally, exception or not) would let
                # this rank silently join a DIFFERENT collective while its
                # peers wait forever on this one -- see STORMER-STEP202-
                # FORENSICS.md. Abort the whole communicator instead.
                _abort_on_loop_exception(
                    _loop_exc, rank=MPIUtils.rank(), epoch=epoch, step=step,
                    comm=MPIUtils.comm_world(), log=log,
                )
                raise
            finally:
                log.info("STEP_SERIES_END rank=%d epoch=%d optimizer_updates=%d reported_steps=%d",
                         MPIUtils.rank(), epoch, self._optimizer_update_count - _epoch_update_start,
                         _epoch_reported_steps)
                if (_sw_on or self._per_rep_step_writer is not None) and not _lc_abort_seen:
                    # Normal exit path only (see _lc_abort_seen's own comment above): on the padfix
                    # Abort(97) path this whole block is skipped, so no elapsed record is written for
                    # an aborting rank -- that rep is already ineligible via the loop-exit marker.
                    _lc_full_loop_end = time.monotonic()
                    _lc_wall_end = time.time()
                    log.info("LONG_CELL_FULL_LOOP rank=%d epoch=%d elapsed_ms=%s",
                             MPIUtils.rank(), epoch,
                             repr(round((_lc_full_loop_end - _lc_elapsed_start) * 1000.0, 3)))
                    if self._per_rep_step_writer is not None and _lc_n_updates and _lc_n_end is not None:
                        log.info("LONG_CELL_TO_N rank=%d epoch=%d n=%d elapsed_ms=%s",
                                 MPIUtils.rank(), epoch, _lc_n_updates,
                                 repr(round((_lc_n_end - _lc_elapsed_start) * 1000.0, 3)))
                        from per_rep_reporting import write_elapsed_record
                        write_elapsed_record(self._per_rep_step_writer.directory, MPIUtils.rank(),
                                             _lc_elapsed_start, _lc_n_end, _lc_n_updates,
                                             _lc_full_loop_end, _lc_wall_start, _lc_wall_end)
                if _sw_on and _sw_wall:
                    log.info("STEP_SERIES rank=%d epoch=%d n=%d wall_ms=%s loss=%s",
                             MPIUtils.rank(), epoch, len(_sw_wall),
                             [repr(round(x, 3)) for x in _sw_wall], [repr(x) for x in _sw_loss])
                    if _sf_on and _sf_fetch:
                        log.info("STEP_SERIES_FETCH rank=%d epoch=%d n=%d fetch_ms=%s",
                                 MPIUtils.rank(), epoch, len(_sf_fetch), [repr(round(x, 3)) for x in _sf_fetch])
                    if _se_on and _se_exposed:
                        log.info("STEP_SERIES_EXPOSED_COMM rank=%d epoch=%d n=%d exposed_comm_ms=%s",
                                 MPIUtils.rank(), epoch, len(_se_exposed), [repr(round(x, 3)) for x in _se_exposed])
                    if _sio_on and _sio_r:
                        log.info("STEP_SERIES_IO rank=%d host=%s epoch=%d n=%d workers=%d rchar=%s read_bytes=%s alive=%s",
                                 MPIUtils.rank(), socket.gethostname(), epoch, len(_sio_r), len(_sio_pids),
                                 _sio_r, _sio_b, _sio_n)
                # Epoch-level comm timing summary
                if (self._comm_timing or self._comm_optimize) and _ct_steps and MPIUtils.rank() == 0:
                    n = len(_ct_steps)
                    avg_step = sum(r[0] for r in _ct_steps) / n
                    avg_fwd = sum(r[1] for r in _ct_steps) / n
                    avg_bwd = sum(r[2] for r in _ct_steps) / n
                    avg_comm = sum(r[3] for r in _ct_steps) / n
                    avg_other = sum(r[5] for r in _ct_steps) / n
                    avg_fetch = sum(r[7] for r in _ct_steps if len(r) > 7) / n
                    avg_total = avg_fetch + avg_step
                    epoch_losses = [r[6] for r in _ct_steps if len(r) > 6 and not (r[6] != r[6])]
                    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
                    log.info(
                        "COMM_TIMING EPOCH %d SUMMARY (%d steps): "
                        "avg_total=%.0fms avg_fetch=%.0fms avg_step=%.0fms avg_fwd=%.0fms avg_bwd=%.0fms "
                        "avg_comm=%.0fms avg_other=%.0fms comm_pct=%.1f%% fetch_pct=%.1f%% "
                        "avg_loss=%.4f",
                        epoch, n, avg_total, avg_fetch, avg_step, avg_fwd, avg_bwd, avg_comm, avg_other,
                        100 * avg_comm / avg_total if avg_total > 0 else 0,
                        100 * avg_fetch / avg_total if avg_total > 0 else 0,
                        avg_loss,
                    )

                # CLOSE THE TRAINING WINDOW BEFORE VALIDATION, AND DRAIN THE STALL INTO IT.
                #
                # Two defects lived here (STATUS-ATTRIBUTION-FIX-2026-09-08.md, defect C3). The
                # stall accumulator was drained only on `step % every_n == 0`, and a 200-step epoch
                # at cadence 24 never hits that on its last step, so the epoch's tail stall was
                # never emitted: the epoch's LAST window carried no `stall.storage` event at all,
                # which is ABSENCE rather than zero, so the planner silently took the
                # fetch-minus-decode fallback there, and the undrained tail then leaked into the
                # FIRST window of the next epoch. And `flush()` ran after `validate_epoch`, so that
                # last window's wall swallowed the whole validation pass, which emits none of the
                # four domains. Together they are why windows 9 and 18 of `stgi_capacity_gate1`
                # report domains summing to 28 and 14 percent of their wall while their neighbours
                # report 99.
                if _window is not None:
                    if _emit_stall is not None:
                        _emit_stall(_window.pending_steps)
                    _window.flush()
                if pbar:
                    pbar.close()
                dft_ai.pipeline.epoch.stop(metadata=True)

            # Validation at end of epoch -- deliberately OUTSIDE the try/except/
            # finally above, so it is reached only when the per-step loop
            # completed without raising. Before this fix, `validate_epoch()`
            # (and the trajectory-retention validation branch, which also
            # calls into rank-coordinated code) lived inside the `finally`
            # block and therefore ran unconditionally -- exception or not --
            # which is how a rank-local exception on one rank turned into
            # that rank silently reaching `validate_epoch()`'s `dist.barrier()`
            # while its peers hung forever in an earlier collective
            # (STORMER-STEP202-FORENSICS.md). An exception in the try block
            # now aborts the whole communicator via `_abort_on_loop_exception`
            # (see the `except` above) before control can ever reach here, so
            # this can no longer run on only some ranks.
            #
            # Epoch-end retained snapshots are deliberately after the window
            # flush (in `finally` above) and before validation. They are
            # never part of a per-repetition timing interval or a live
            # analysis window.
            if (self._trajectory_retention is not None
                    and self._completed_updates in self._trajectory_retention["points"]
                    and self._completed_updates not in self._trajectory_retention["captured"]):
                self._capture_trajectory_state(self._completed_updates, epoch=epoch, kind="epoch_end")
                trajectory_validation = self._run_trajectory_validation(epoch, self._completed_updates)
                self._record_trajectory_validation(self._completed_updates, trajectory_validation)
            else:
                self.validate_epoch(
                    epoch,
                    preserve_trajectory_state=self._trajectory_retention is not None,
                )

            # Opt-in (cached-regime study, STORMER_APPLY_AT_EPOCH_END=1): apply pending plans at the epoch
            # boundary. The in-loop apply point is a window boundary (_window.stop()); once the window is
            # sized (about 30 s) it exceeds a short epoch, and flush() closes the epoch window outside
            # the loop, so a plan produced from it was never drained or applied. Called by every rank at
            # the same program point after validation (the flush window's plan has arrived by then).
            if (os.environ.get("STORMER_APPLY_AT_EPOCH_END", "0") == "1" and _window is not None
                    and epoch + 1 < self.epochs):
                self._loop_step = step
                self._sync_apply_window(train_loader, step=step)

        if self._per_rep_step_writer is not None:
            self._per_rep_step_writer.close()
        self._finish_trajectory_retention()

    def finalize(self):
        _mot.flush()
        return None

def get_args():
    # Benchmark settings
    parser = argparse.ArgumentParser(
        description="Stormer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-folder", default="outputs", type=str)
    parser.add_argument("--log-folder", default=None, type=str)
    parser.add_argument("--ngpus-per-node", type=int, help="NGPUs per node (required)")
    # dataset
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument(
        "--latitude-metadata-dir", type=str, default=None,
        help="Separate reviewed latitude metadata artifact; required for the native 0.25-degree corpus",
    )
    parser.add_argument("--data-freq", type=int, default=6)
    parser.add_argument("--interval-protocol", choices=("legacy", INTERVAL_SELECTOR_VERSION),
                        default="legacy", help="Opt-in worker-independent target interval schedule")
    parser.add_argument("--interval-index-manifest", type=str, default=None,
                        help="Frozen train/val index mapping required by the keyed interval protocol")
    parser.add_argument("--trajectory-state-dir", type=str, default=None,
                        help="Opt-in new destination for state-v1 epoch-end trajectory artifacts")
    parser.add_argument("--trajectory-update-points", type=int, nargs="+", default=None,
                        help="Ordered completed-update IDs for update 0 plus epoch-end snapshots")
    parser.add_argument("--trajectory-source-manifest", type=str, default=None,
                        help="Immutable source-hash manifest bound into state-v1 provenance")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--intervals", type=int, nargs="+", required=True, help="List of intervals (required)")
    # dataloader
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--disable-collation", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    # optimizer
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta-1", type=float, default=0.9)
    parser.add_argument("--beta-2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    # scheduler
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Number of warmup epochs (default: 10)")
    parser.add_argument("--warmup-start-lr", type=float, default=1e-8, help="Starting learning rate for warmup (default: 1e-8)")
    parser.add_argument("--eta-min", type=float, default=1e-8, help="Minimum learning rate (default: 1e-8)")
    # model
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in-img-size", type=int, nargs='+', required=True, help="Input image size as a list or tuple of integers")
    # parser.add_argument("--variables", type=str, nargs='+', required=True, help="List of variable names as strings")
    parser.add_argument("--patch-size", type=int, default=2, help="Patch size (default: 2)")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Hidden size (default: 1024)")
    parser.add_argument("--depth", type=int, default=24, help="Depth (default: 24)")
    parser.add_argument("--num-heads", type=int, default=16, help="Number of heads (default: 16)")
    parser.add_argument("--mlp-ratio", type=float, default=4.0, help="MLP ratio (default: 4.0)")
    parser.add_argument(
        "--ulysses-sp",
        type=int,
        default=int(os.environ.get("STORMER_ULYSSES_SP", "1")),
        help="DeepSpeed-Ulysses sequence-parallel degree (1=off/stock; must divide world size). "
        "Non-invasive: monkeypatches attention + Stormer.forward at runtime, stormer.py unedited.",
    )
    # trainer
    parser.add_argument("--precision", type=str, default="16", help="Precision (str, int, or None). Default: 16")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1)")
    parser.add_argument("--max-training-step", type=int, default=-1, help="Maximum training steps (default: -1 for unlimited)")
    parser.add_argument("--weighted-loss", action="store_true", default=True, help="Use weighted loss (default: True)")
    parser.add_argument("--enable-progress-bar", action="store_true", default=True, help="Enable progress bar (default: True)")
    parser.add_argument("--sync-batchnorm", action="store_true", default=False, help="Synchronize batch normalization across ranks")
    parser.add_argument("--no-sync-batchnorm", dest="sync_batchnorm", action="store_false")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="Accumulate gradient batches (default: 1)")

    args = parser.parse_args()
    return args

def main() -> None:
    MPIUtils.initialize()
    _write_mpi_helper_roster()
    args = get_args()
    seed_everything(args.seed)
    os.makedirs(args.output_folder, exist_ok=True)
    log_folder = args.log_folder or args.output_folder
    os.makedirs(log_folder, exist_ok=True)
    configure_logging(output_dir=log_folder)

    if DEVICE_TYPE != "cuda":
        raise RuntimeError("Stormer training requires a CUDA/ROCm GPU environment.")

    torch.cuda.set_device(DEVICE_ID)

    hostname, port = get_master_addr_and_port(port=23456, set_env=True)

    if MPIUtils.rank() == 0:
        log.info("Master addresses = %s:%d", hostname, port)

    dist_initialized = False
    if MPIUtils.size() > 1:
        from datetime import timedelta
        _nccl_timeout = int(os.environ.get("STORMER_NCCL_TIMEOUT", "1800"))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=_nccl_timeout),
            world_size=MPIUtils.size(),
            rank=MPIUtils.rank(),
            device_id=DEVICE_ID,
        )
        dist.barrier()
        dist_initialized = True

    # ngpus_per_node = torch.cuda.device_count()
    # log.info("NGPUS %s", ngpus_per_node)
    # for i in range(torch.cuda.device_count()):
    #     print(f"Rank {MPIUtils.rank()} -> Device {i}: {torch.cuda.get_device_name(i)}")
    #     properties = torch.cuda.get_device_properties(i)
    #     print(f"  Total Memory: {properties.total_memory / (1024**3):.2f} GB")
    #     print(f"  Compute Capability: {properties.major}.{properties.minor}")
    # return

    if MPIUtils.rank() == 0:
        log.info("Arguments")
        for key, value in vars(args).items():
            log.info(f" {key}: {value}")

    if MPIUtils.rank() == 0:
        log.info("DFTracer init: DFTRACER_ENABLE=%s WRITER_TYPE=%s GROUP_FILE=%s data_dir=%s",
                 os.environ.get("DFTRACER_ENABLE"), os.environ.get("DFTRACER_WRITER_TYPE"),
                 os.environ.get("DFTRACER_MOFKA_GROUP_FILE"), args.data_folder)
    pfwlogger = PerfTracer.initialize_log(
        logfile=f"{log_folder}/trace-{MPIUtils.rank() + 1}-of-{MPIUtils.size()}.pfw",
        data_dir=args.data_folder,
        process_id=MPIUtils.rank(),
    )
    with dft_ai:
        model = Stormer(
            in_img_size=args.in_img_size,
            variables=VARIABLES,
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
        )

        datamodule = DataModule(
            args=args,
            root_dir=args.data_folder,
            variables=VARIABLES,
            intervals=args.intervals,
        )

        trainer = Trainer(
            args=args,
            model=model,
            datamodule=datamodule
        )

        trainer.train()
        trainer.finalize()

    pfwlogger.finalize()
    # Post-finalize teardown. Reached only after successful training, validation (trainer.train
    # runs it) and pfwlogger.finalize(), so any exception above keeps its nonzero exit.
    # (a) Hang diagnostic: the 4N Step-0 arms finished training and then sat ~23 min until the
    # allocation expired. Arm ONE non-fatal stack dump now (exit=False, no repeat; this runs after
    # training so the shared faulthandler timer holds no other client) so a hang here prints every
    # thread's stack. Cancelled just before the hard exit.
    _td_sec = float(os.environ.get("STORMER_TEARDOWN_STACK_SEC", "120") or 0)
    if _td_sec > 0:
        faulthandler.dump_traceback_later(_td_sec, repeat=False, exit=False)
    _hard_exit = os.environ.get("STORMER_HARD_EXIT_AFTER_FINALIZE", "0") == "1"
    if _hard_exit:
        # (b) fix A': DFTracer's POSIX close() wrapper is still bound at process exit and
        # segfaults when torchvision's HIP static destructor closes fds after the tracer is
        # finalized (native core, 2026-09-25). One explicit barrier so no rank leaves early, then
        # exit BEFORE dist.destroy_process_group()/MPIUtils.finalize(), which hung at 4N.
        if dist_initialized:
            dist.barrier()
        faulthandler.cancel_dump_traceback_later()
        sys.stdout.flush()
        sys.stderr.flush()
        _mot.flush(); os._exit(0)
    if dist_initialized:
        dist.barrier()
        dist.destroy_process_group()
    MPIUtils.finalize()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
