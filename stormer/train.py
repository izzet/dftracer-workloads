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

from typing import Any
import logging
import faulthandler
import multiprocessing as _mp
import os
import signal
import threading
import time
import contextlib
import argparse
from glob import glob
from contextlib import contextmanager
from collections.abc import Generator
import warnings

# LC HACK: work around so that "import torch" will not change CPU affinity
# see https://rzlc.llnl.gov/jira/browse/ELCAP-386
os.environ.pop("OMP_PLACES", None)
os.environ.pop("OMP_PROC_BIND", None)

# from mpi4py import MPI

import h5py
import numpy as np

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

from utils.torch_utils import seed_everything
from utils.precision import setup_precision_context, get_precision_dtype
from utils.perf_tracer import PerfTracer, PERFTRACER_ENABLE, dft_ai
from utils.logging import configure_logging
from utils.mpi import MPIUtils, get_master_addr_and_port

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
        self.year_idx_map: dict | None = None
        self.image_shape = self.args.in_img_size
        self.downsample = os.environ.get("STORMER_DOWNSAMPLE", "1") == "1"
        if year_list is not None:
            self.year_idx_map = {year: i for i, year in enumerate(year_list)}

        file_paths = glob(os.path.join(root_dir, "*.h5"))
        file_paths = sorted(file_paths)

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
            try:
                f = h5py.File(path, "r")
            except OSError as exc:
                raise OSError(f"could not open StorMer sample {path}: {exc}") from exc
            x = []
            for var in variables:
                d = f["input"][var][:]
                x.append(d.reshape(1, *d.shape[-2:]))
            f.close()
            del f

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

    @dft_ai.data.item
    def __getitem__(self, index: int):

        # 1. Get input data
        inp_path = self.inp_file_paths[index]
        inp_data = self.get_data_given_path(inp_path, self.variables)

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
        with dft_ai.data.preprocess:
            diff = out - inp_data
            interval = torch.Tensor([chosen_interval]) / 10.0
            inp_transform = self.inp_transform(inp_data).unsqueeze(0)
            diff_transform = self.out_transform_dict[chosen_interval](diff).unsqueeze(0)

        return (
            inp_transform,
            diff_transform,
            interval,
        )
    
collate_profiler = dft_ai.data.preprocess.derive(name="collate")

@collate_profiler
def collate_fn_train(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])  # B, V, H, W
    out = torch.stack([batch[i][1] for i in range(len(batch))])  # B, T, V, H, W
    interval = torch.stack([batch[i][4] for i in range(len(batch))])  # B, T
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
        lat = np.load(os.path.join(self.root_dir, "lat.npy"))
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
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # keep it simple for val
            pin_memory=False,
            collate_fn=collate_fn,
        )

    def dataloader(self, loader_config: dict[str, Any] | None = None):
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

        batch_size = self.batch_size
        collate_fn = collate_fn_train

        if self.args.disable_collation:
            if batch_size > 1:
                log0(f"Cannot disable collation since batch_size is {batch_size}", mode="warning")

            if batch_size == 1:
                batch_size = None
                collate_fn = None
                log0("Disabling collation")

        use_live = os.environ.get("DFOPTIMIZER_ENABLE", "0") == "1"

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

        return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=False,
                sampler=sampler,
                num_workers=loader_config["num_workers"],
                pin_memory=loader_config["pin_memory"],
                persistent_workers=loader_config["persistent_workers"],
                prefetch_factor=loader_config["prefetch_factor"],
                collate_fn=collate_fn,
            ),
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
        self.setup_model()

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

        # --- Communication timing and optimization ---
        self._comm_timing = os.environ.get("STORMER_COMM_TIMING", "0") == "1"
        self._comm_optimize = os.environ.get("STORMER_COMM_OPTIMIZE", "0") == "1"
        self._comm_bucket_ms = []  # filled by hook, cleared per step
        self._fwd_ms = 0.0
        self._bwd_ms = 0.0
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
        if _comm_hook_enabled and MPIUtils.size() > 1:
            def _comm_hook(state, bucket):
                with dft_ai.comm.all_reduce:
                    torch.cuda.synchronize()
                    if _comm_sync_probe:
                        dist.barrier()
                    t0 = time.time()
                    work = dist.all_reduce(bucket.buffer(), async_op=True)
                    work.wait()
                    torch.cuda.synchronize()
                    state.append((time.time() - t0) * 1000)
                # Return C++ Future via Work.get_future() (DDP requirement)
                return work.get_future().then(lambda fut: fut.value()[0])
            self.net.register_comm_hook(self._comm_bucket_ms, _comm_hook)
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
            self._live_loader, self._live_sampler = self.datamodule.dataloader()

        from live_data_loader import LiveDataLoader
        if isinstance(self._live_loader, LiveDataLoader):
            self._live_loader.set_epoch(epoch)
            log0(
                "make_loader: epoch=%s num_workers=%s prefetch_factor=%s (LiveDataLoader)",
                epoch, self._live_loader.num_workers, self._live_loader.prefetch_factor,
            )
            return self._live_loader, self._live_sampler

        # Standard DataLoader: recreate each epoch
        train_loader, train_sampler = self.datamodule.dataloader()
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
    def send_to_device(self, batch):
        x, gt_diff, interval_tensors = batch
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
    def forward_step(self, batch: Any):
        _fp("fwd.start")
        x, gt_diff, interval = batch

        with self.forward_context():
            pred_diff = self.forward_train(x, self.variables, interval)

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
    def backward_step(self, loss, batch_idx: int, n_accum_steps: int):
        _fp("bwd.start")
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Only step optimizer at end of accumulation window
        if batch_idx % n_accum_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

    @dft_ai.compute
    def training_step(self, batch: Any, batch_idx: int):
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
        want_w = want_pf = want_accum = want_bcap = want_dec = SENT
        ctx = self._optimizer_ctx
        if ctx is not None and not ctx._noop and isinstance(train_loader, LiveDataLoader):
            actions = ctx.drain_actions_for("stormer.make_loader", at="window_boundary") or []
            for action in actions:
                knob_name = action.knob_id
                if knob_name.startswith("stormer."):
                    knob_name = knob_name[len("stormer."):]
                if knob_name == "num_workers":
                    want_w = int(action.new_value)
                    ctx.ack_action(action, train_loader.num_workers, want_w)
                elif knob_name == "prefetch_factor":
                    want_pf = int(action.new_value)
                    ctx.ack_action(action, train_loader.prefetch_factor, want_pf)
                elif knob_name == "accumulate_grad_batches":
                    want_accum = int(action.new_value)
                    ctx.ack_action(action, self.accumulate_grad_batches, want_accum)
                elif knob_name == "bucket_cap_mb" and getattr(self, "_tune_bcap_live", False):
                    want_bcap = int(action.new_value)
                    ctx.ack_action(action, int(self._bucket_cap_mb), want_bcap)
                elif knob_name == "decode_threads" and getattr(self, "_tune_decode_live", False):
                    want_dec = int(action.new_value)
                    ctx.ack_action(action, int(os.environ.get("STORMER_DECODE_THREADS", "1") or 1),
                                   want_dec)
                else:
                    ctx.ack_action(action, None, None, status="rejected")
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
        if MPIUtils.size() > 1 and MPIUtils.is_mpi():
            from mpi4py import MPI
            _buf = np.array([want_w, want_pf, want_accum, want_bcap, want_dec], dtype=np.int64)
            _out = np.empty_like(_buf)
            MPIUtils.comm_world().Allreduce(_buf, _out, op=MPI.MAX)
            want_w, want_pf, want_accum, want_bcap, want_dec = (int(_out[0]), int(_out[1]),
                                                                int(_out[2]), int(_out[3]),
                                                                int(_out[4]))
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
        _dec_changed = (getattr(self, "_tune_decode_live", False) and want_dec > SENT
                        and want_dec != int(os.environ.get("STORMER_DECODE_THREADS", "1") or 1))
        if _dec_changed and "decode_threads" in _rec_only:
            if MPIUtils.rank() == 0:
                log.info("SAFETY_GATE: decode_threads %s -> %d RECOMMENDED, not applied "
                         "(actuation path not proven safe; see LIVEFLOW-DECODE-RCA.md sec 15)",
                         os.environ.get("STORMER_DECODE_THREADS", "1"), want_dec)
            _dec_changed = False        # attribution recorded; actuation withheld
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
        if _is_live and _eager and train_loader.apply_pending():
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
            torch.cuda.synchronize()
            if MPIUtils.size() > 1 and MPIUtils.is_mpi():
                MPIUtils.comm_world().Barrier()
            self._rebuild_ddp(float(want_bcap))
            if MPIUtils.size() > 1 and MPIUtils.is_mpi():
                MPIUtils.comm_world().Barrier()

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
        if MPIUtils.size() > 1 and MPIUtils.is_mpi():
            from mpi4py import MPI
            _dt = np.array([(time.time() - _t_apply0) * 1000.0], dtype=np.float64)
            _mx = np.empty_like(_dt)
            MPIUtils.comm_world().Allreduce(_dt, _mx, op=MPI.MAX)
            if MPIUtils.rank() == 0 and _mx[0] >= 1.0:
                log.info("APPLY_SYNC: worst-rank apply latency %.1f ms", float(_mx[0]))
        # Cancel AFTER the barrier, deliberately: a rank that finished its own apply but is stuck
        # waiting here is also something we want to see, and its stack (blocked in Allreduce)
        # distinguishes "I am the straggler" from "I am waiting for the straggler".
        if _stack_sec > 0:
            faulthandler.cancel_dump_traceback_later()

    @torch.no_grad()
    def validate_epoch(self, epoch):
        """Run validation on held-out data (rank 0 only to avoid file contention)."""
        # Ulysses SP: validation runs the model forward on rank 0 only while other ranks wait at
        # a barrier, but the SP attention all_to_all is a collective over the whole sp_group, so
        # a rank-0-only forward DEADLOCKS. Skip validation entirely in SP mode (all ranks check
        # the same _sp_size and return together -> no collective mismatch). Not needed for the
        # systems/timing study.
        if getattr(self, "_sp_size", 1) > 1:
            if MPIUtils.rank() == 0:
                log.info("Skipping validation (Ulysses SP mode; rank-0-only val is incompatible with SP collectives)")
            return
        if MPIUtils.rank() != 0:
            # Non-rank-0 processes wait at barrier
            if MPIUtils.size() > 1:
                dist.barrier()
            return
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
            return
        # Use the unwrapped model for single-rank eval
        model = self.net.module if isinstance(self.net, DDP) else self.net
        model.eval()
        val_losses = []
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
        model.train()
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
        log.info("VAL_LOSS epoch=%d val_loss=%.4f (n=%d samples)", epoch, avg_val_loss, len(val_losses))
        if MPIUtils.size() > 1:
            dist.barrier()
        return avg_val_loss

    def _rebuild_ddp(self, new_bucket_cap_mb):
        """Rebuild DDP with a new bucket_cap_mb. Must be called outside backward pass."""
        if not isinstance(self.net, DDP):
            return
        module = self.net.module
        old_cap = self._bucket_cap_mb
        self._bucket_cap_mb = new_bucket_cap_mb
        self.net = DDP(
            module,
            device_ids=[DEVICE_ID],
            output_device=DEVICE_ID,
            bucket_cap_mb=new_bucket_cap_mb,
            mixed_precision=self._ddp_mixed_precision,
        )
        # Re-register comm hook on new DDP wrapper.
        # BUGFIX (LIVEFLOW-STATUS.md sec 23.1): this closure previously OMITTED the
        # _comm_sync_probe barrier that the setup_model hook has, so after any rebuild the
        # straggler decomposition (probe=1) silently reverted to measuring transfer+skew.
        if (self._comm_timing or self._comm_optimize) and MPIUtils.size() > 1:
            _sync_probe = os.environ.get("STORMER_COMM_SYNC_PROBE", "0") == "1"

            def _comm_hook(state, bucket):
                torch.cuda.synchronize()
                if _sync_probe:
                    dist.barrier()
                t0 = time.time()
                work = dist.all_reduce(bucket.buffer(), async_op=True)
                work.wait()
                torch.cuda.synchronize()
                state.append((time.time() - t0) * 1000)
                return work.get_future().then(lambda fut: fut.value()[0])
            self.net.register_comm_hook(self._comm_bucket_ms, _comm_hook)
        log0("COMM_OPT: rebuilt DDP bucket_cap_mb=%.0f→%.0f", old_cap, new_bucket_cap_mb)

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
                _tune_comm = os.environ.get("STORMER_TUNE_COMM", "0") == "1" and MPIUtils.size() > 1
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
                    knob_defs["bucket_cap_mb"] = knob(
                        default=int(self._bucket_cap_mb),
                        range=(int(os.environ.get("STORMER_KNOB_BCAP_MIN", "25")),
                               int(os.environ.get("STORMER_KNOB_BCAP_MAX", "400"))),
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
        log0("Will run %s steps", self.n_steps)

        # Window manages analysis boundaries for the streaming pipeline.
        # Emits window.start/window.stop control events at the configured
        # cadence so the analyzer knows when to drain and analyze.
        _window = None
        _n_steps_eff = self.max_training_step if self.max_training_step > 0 else self.n_steps
        # cadence: STORMER_WINDOW_EVERY_N overrides (finer cadence -> more optimizer opportunities
        # to actuate); default = 5 windows per run.
        _window_every_n = int(os.environ.get("STORMER_WINDOW_EVERY_N", "0") or 0) or max(_n_steps_eff // 5, 1)
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
            log0("Window initialized (every_n=%s)", _window.every_n)
        except Exception as e:
            log0("WARNING: Window init failed (no streaming analysis): %s", e)

        # ---- EXPOSURE instrument (liveflow_shm/EXPOSURE-PORT-DESIGN.md) --------------------------
        # Without this, the agent decides about StorMer's input side from dur_sum, which is
        # per-operation service time summed across threads. It has been measured to move OPPOSITE
        # to wall when a stage is relieved, so it cannot identify a bottleneck.
        _stall_acct = None
        _emit_stall = None
        try:
            import sys as _sys
            for _p in (os.environ.get("LIVEFLOW_SHM_PYTHONPATH", ""),):
                if _p and _p not in _sys.path:
                    _sys.path.insert(0, _p)
            from exposure import StallAccount, emit_stall_events
            _stall_acct = StallAccount(exact=False)   # ratio split, see _STAGE_A_MS

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

        for epoch in dft_ai.pipeline.epoch.iter(
            range(self.epochs),
            include_iter=False,
        ):
            dft_ai.pipeline.epoch.start(metadata=True)
            pbar = None
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

                with logging_ctx():
                    _t_fetch0 = time.time()
                    for batch in dft_ai.dataloader.fetch.iter(train_loader):
                        _fp("batch.received")   # loader returned an assembled batch
                        # UNCONDITIONAL now. This is the consumer-boundary exposure: the gap
                        # between the end of the previous step and the batch arriving. With a
                        # prefetching loader it is ~0 when the workers kept up, and it is the real
                        # stall when they did not. It used to be computed only under comm timing,
                        # so the quantity the attribution needs did not exist in a normal run.
                        _fetch_ms = (time.time() - _t_fetch0) * 1000
                        if _stall_acct is not None and _fetch_ms > 0:
                            _sa, _sb = _stage_times_drain()
                            _tot = _sa + _sb
                            # No stage evidence: charge storage rather than invent a split. Dropping
                            # it would understate the domain and read like a loader that kept up.
                            _stall_acct.add_split(_fetch_ms, 1.0 if _tot <= 0 else _sa / _tot)

                        if self.max_training_step != -1 and step > self.max_training_step:
                            log0(f"Training step limit reached: {self.max_training_step}")
                            break

                        if _window is not None:
                            _window.start()

                        if self._comm_timing or self._comm_optimize:
                            torch.cuda.synchronize()
                            _t_step0 = time.time()

                        batch = self.send_to_device(batch)
                        loss = self.training_step(batch, step)

                        if self._comm_timing or self._comm_optimize:
                            torch.cuda.synchronize()
                            step_ms = (time.time() - _t_step0) * 1000
                            comm_ms = sum(self._comm_bucket_ms)
                            n_buckets = len(self._comm_bucket_ms)
                            self._comm_bucket_ms.clear()
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

                        if _window is not None:
                            # Exposure into the trace BEFORE the window closes, so the window the
                            # analyzer is about to analyse carries the stall it must explain.
                            if _emit_stall is not None and step % _window_every_n == 0:
                                _emit_stall(_window_every_n)
                            _window.stop()

                        if pbar:
                            pbar.set_postfix({"loss": loss})
                            pbar.update()

                        step += 1
                        _fp("step.end", step)
                        _note_progress()   # feeds the stall watchdog (module top)
                        dft_ai.update(epoch=epoch, step=step)
                        collate_profiler.update(epoch=epoch, step=step)

                        # Poll optimizer at window boundaries for mid-epoch reconfiguration
                        if step % _window_every_n == 0:
                            self._sync_apply_window(train_loader, step=step)
                            self._comm_optimize_window()
            finally:
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

                # Validation at end of epoch
                self.validate_epoch(epoch)

                if _window is not None:
                    _window.flush()
                if pbar:
                    pbar.close()
                dft_ai.pipeline.epoch.stop(metadata=True)

    def finalize(self):
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
    parser.add_argument("--data-freq", type=int, default=6)
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
    if dist_initialized:
        dist.barrier()
        dist.destroy_process_group()
    MPIUtils.finalize()


if __name__ == "__main__":
    main()
