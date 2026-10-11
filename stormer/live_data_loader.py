"""LiveDataLoader for StorMer — supports mid-epoch num_workers reconfiguration.

Wraps a standard PyTorch DataLoader + LiveSampler.  When reconfigure() is
called (e.g. by the optimizer), the current DataLoader is torn down and a new
one is created with updated num_workers / prefetch_factor, resuming iteration
from the exact sample position.

Usage in the training loop:

    loader = LiveDataLoader(dataset, sampler, ...)
    loader.set_epoch(epoch)

    for batch in loader:
        train_step(batch)
        if step % poll_interval == 0:
            plan = poll_plans()
            if plan:
                loader.reconfigure(num_workers=plan["num_workers"], ...)
"""

import motivation_timers as _mot
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import _utils as _tud_utils
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter

from live_sampler import LiveSampler

log = logging.getLogger(__name__)


def make_keyed_dataloader_generator(version_tag: str, experiment_seed: int, epoch: int) -> torch.Generator:
    """Return a fresh private CPU generator for one keyed-loader epoch.

    The generator supplies PyTorch's DataLoader worker base seed.  It is
    deliberately recreated from stable workload context on a live rebuild, so
    that rebuilding workers neither advances the parent Torch RNG nor changes
    the keyed protocol's worker seeds.
    """
    if not isinstance(version_tag, str) or not version_tag:
        raise ValueError("version_tag must be a nonempty string")
    for name, value in (("experiment_seed", experiment_seed), ("epoch", epoch)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    payload = json.dumps(
        [version_tag, experiment_seed, epoch, "dataloader-base-seed"],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")
    return torch.Generator(device="cpu").manual_seed(seed)


class _EpochKeyedDataset(Dataset):
    """Dataset whose items are addressed by (epoch, index), so persistent workers follow the epoch.

    The keyed interval protocol selects a sample's forecast target from (seed, epoch, dataset
    index, split). Workers forked once (persistent pool) cannot be told the new epoch by the
    parent, so every task carries it and the worker sets it on the wrapped dataset when it changes.
    The selection is the same function of the same epoch value as with per-epoch workers.
    """

    def __init__(self, dataset):
        self._ds = dataset
        self._cur = None

    def __len__(self):
        return len(self._ds)

    def _set(self, epoch):
        target = self._ds
        while hasattr(target, "dataset"):
            target = target.dataset
        if hasattr(target, "set_epoch"):
            target.set_epoch(epoch)
        self._cur = epoch

    def __getitem__(self, key):
        epoch, index = key
        if epoch != self._cur:
            self._set(epoch)
        return self._ds[index]


class _EpochKeySampler(Sampler):
    """Yields (epoch, index) for the loader's current epoch (read when the epoch's iteration starts)."""

    def __init__(self, sampler, owner):
        self._s = sampler
        self._owner = owner

    def __len__(self):
        return len(self._s)

    def __iter__(self):
        epoch = self._owner._epoch
        for index in self._s:
            yield (epoch, index)


class _ElasticIter(_MultiProcessingDataLoaderIter):
    """In-order multiprocessing iterator whose worker pool and prefetch depth change in place.

    WHY. A knob change used to tear down the whole DataLoader and fork the entire new pool
    (measured on StorMer 8N: 9 s at 16 workers, 70 s at 40, 207 s over six applies of a 1802-step
    run). The sample stream does not depend on which worker serves which batch (tasks are numbered
    and delivered in order, and the keyed interval protocol selects by dataset index), so neither
    knob needs a rebuild.

    WHAT A FORK COSTS, MEASURED (1 node, 4 ranks, tuolumne, stormer-eng small job): `fork()` itself
    is ~0.1 s per worker, but the step AFTER any fork event stalls training for 10-16 s whatever the
    number of workers forked (8, 16 or 32 forks: 10.2, 14.4, 9.7, 16.8 s; the epoch-start fork of
    the stock loader shows the same ~15 s first step). So the cost is per fork EVENT, not per
    worker, and growing by forking at an apply (even only the new workers) buys nothing.

      standby pool    the epoch's single fork event builds a pool of `STORMER_ELASTIC_POOL_MAX`
                      workers (the declared knob range); only the first `num_workers` receive
                      tasks, the rest idle in `index_queue.get`.
      grow            admit standby workers to the dispatch rotation (no process is created); only
                      a request beyond the pool forks, and then only the missing workers.
      shrink          stop dispatching to the newest active workers; they finish their outstanding
                      tasks and go idle. Nothing is terminated before the epoch ends.
      prefetch_factor outstanding tasks are bounded by prefetch_factor x active workers, a number
                      this class owns; changing it needs no process at all.

    Anything torch's iterator does is kept: the same queues, in-order reassembly, pin-memory
    thread, error propagation and shutdown. Only dispatch is replaced.
    """

    def __init__(self, loader, active=None):
        # A forked worker inherits copies of every live iterator. If its garbage collector ever
        # finalised one, torch's shutdown would set the SHARED done event and stop the parent's own
        # workers. Only the creating process may shut an iterator down.
        self._creator_pid = os.getpid()
        self._elastic_ctx = (loader.multiprocessing_context
                             if loader.multiprocessing_context is not None
                             else torch.multiprocessing)
        # Attributes the overridden dispatch needs while the base class primes the first tasks.
        n_pool = loader.num_workers
        n_active = n_pool if active is None else max(1, min(int(active), n_pool))
        self._active = list(range(n_active))
        self._standby = list(range(n_active, n_pool))
        self._rr = 0
        self._pf_target = int(loader.prefetch_factor)
        self._sampler_exhausted = False
        self._next_worker_id = loader.num_workers
        self.fork_seconds = []
        # Seconds the consumer spent blocked inside the result-queue get since the last
        # take_wait_s(): the upstream (storage/decode/queueing) part of the fetch wait. The rest of
        # a step's fetch wait is the delivery family (liveflow_delivery.py).
        self.wait_s = 0.0
        super().__init__(loader)

    def _get_data(self):
        t0 = time.perf_counter()
        try:
            return super()._get_data()
        finally:
            self.wait_s += time.perf_counter() - t0

    def start_epoch(self):
        """Begin the next epoch on the SAME workers (persistent pool, one fork event per run).

        Results still in flight from an epoch that ended early are discarded using torch's own
        persistent-worker protocol: every worker acknowledges a resume message that queues behind
        its stale tasks. Then dispatch state restarts from the loader's sampler.
        """
        live = [i for i, w in enumerate(self._workers) if self._workers_status[i]]
        for i in live:
            self._index_queues[i].put(_tud_utils.worker._ResumeIteration(self._shared_seed))
        waiting = len(live)
        while waiting > 0:
            idx, _data = self._get_data()
            if isinstance(idx, _tud_utils.worker._ResumeIteration):
                waiting -= 1
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._send_idx = 0
        self._rcvd_idx = 0
        self._task_info = {}
        self._tasks_outstanding = 0
        self._workers_num_tasks = [0 for _ in self._workers]
        self._sampler_exhausted = False
        self._rr = 0
        self._refill()

    def _shutdown_workers(self):
        if os.getpid() != self._creator_pid:
            return
        super()._shutdown_workers()

    # -- dispatch ---------------------------------------------------------------------------
    def _try_put_index(self):
        n_active = len(self._active)
        if n_active == 0 or self._tasks_outstanding >= self._pf_target * n_active:
            return
        try:
            index = self._next_index()
        except StopIteration:
            self._sampler_exhausted = True
            return
        wid = self._active[self._rr % n_active]
        self._rr += 1
        self._index_queues[wid].put((self._send_idx, index))
        self._task_info[self._send_idx] = (wid,)
        self._workers_num_tasks[wid] += 1
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _process_data(self, data, worker_idx):
        self._workers_num_tasks[worker_idx] -= 1
        self._try_put_index()
        if isinstance(data, _tud_utils.ExceptionWrapper):
            data.reraise()
        return data

    def _refill(self):
        limit = self._pf_target * len(self._active)
        while (not self._sampler_exhausted and self._tasks_outstanding < limit):
            before = self._send_idx
            self._try_put_index()
            if self._send_idx == before:
                break

    # -- resize -----------------------------------------------------------------------------
    def resize(self, num_workers: int, prefetch_factor: int) -> dict:
        """Move the pool to `num_workers` and the depth to `prefetch_factor`, in place."""
        num_workers, prefetch_factor = int(num_workers), int(prefetch_factor)
        if num_workers < 1 or prefetch_factor < 1:
            raise ValueError("elastic resize needs num_workers >= 1 and prefetch_factor >= 1")
        t0 = time.perf_counter()
        before = len(self._active)
        forked = admitted = parked = 0
        if num_workers > before:
            need = num_workers - before
            take = self._standby[:need]
            self._standby = self._standby[need:]
            self._active.extend(take)
            admitted = len(take)
            if need > admitted:
                forked = self._grow(need - admitted)
        elif num_workers < before:
            moved = self._active[num_workers:]
            self._active = self._active[:num_workers]
            self._standby = moved + self._standby
            parked = len(moved)
        self._pf_target = prefetch_factor
        self._refill()
        self._num_workers = len(self._active)
        return {"workers_before": before, "workers_after": len(self._active),
                "admitted_from_standby": admitted, "parked": parked, "forked": forked,
                "standby": len(self._standby), "prefetch": prefetch_factor,
                "seconds": time.perf_counter() - t0,
                "fork_seconds": list(self.fork_seconds[-forked:]) if forked else []}

    def _grow(self, n: int) -> int:
        ctx = self._elastic_ctx
        total = len(self._active) + n
        for _ in range(n):
            wid = self._next_worker_id
            self._next_worker_id += 1
            index_queue = ctx.Queue()
            index_queue.cancel_join_thread()
            w = ctx.Process(
                target=_tud_utils.worker._worker_loop,
                args=(self._dataset_kind, self._dataset, index_queue, self._worker_result_queue,
                      self._workers_done_event, self._auto_collation, self._collate_fn,
                      self._drop_last, self._base_seed, self._worker_init_fn, wid, total,
                      self._persistent_workers, self._shared_seed),
            )
            w.daemon = True
            t = time.perf_counter()
            w.start()
            self.fork_seconds.append(time.perf_counter() - t)
            # Worker ids are list positions in the base class's per-worker lists, never reused.
            if wid != len(self._workers):
                raise RuntimeError("elastic worker id out of step with the worker list")
            self._index_queues.append(index_queue)
            self._workers.append(w)
            self._workers_status.append(True)
            self._workers_num_tasks.append(0)
            self._active.append(wid)
        self._register_pids()
        return n

    def _register_pids(self, exclude=frozenset()):
        """Re-register the worker pids with torch's SIGCHLD watcher (set-once per iterator id)."""
        sig = _tud_utils.signal_handling
        if self._worker_pids_set:
            sig._remove_worker_pids(id(self))
        live = [w.pid for i, w in enumerate(self._workers)
                if i not in exclude and self._workers_status[i]]
        sig._set_worker_pids(id(self), tuple(live))
        self._worker_pids_set = True


class LiveDataLoader:

    def __init__(
        self,
        dataset: Dataset,
        sampler: LiveSampler,
        *,
        batch_size: int = 1,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        collate_fn=None,
        generator_factory: Callable[[int], torch.Generator] | None = None,
    ):
        self._dataset = dataset
        self._sampler = sampler
        self._batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers
        self._collate_fn = collate_fn
        self._generator_factory = generator_factory
        self._epoch = 0
        # In-place pool resize (see _ElasticIter). STORMER_ELASTIC_LOADER=0 restores the full
        # rebuild per knob change, which is the "before" arm of the apply-cost measurement.
        self._elastic = os.environ.get("STORMER_ELASTIC_LOADER", "1") == "1"
        self._elastic_it: _ElasticIter | None = None
        # PERSISTENT POOL: one fork event per run. Needs the keyed protocol (the epoch then travels
        # with each task; the legacy protocol seeds workers per epoch), the elastic iterator and
        # the loader's own persistence (not torch's persistent_workers flag).
        self._keep_pool = (self._elastic and generator_factory is not None and not persistent_workers
                           and os.environ.get("STORMER_LOADER_PERSISTENT", "1") == "1")
        if self._keep_pool:
            self._dataset = _EpochKeyedDataset(dataset)
        # What the most recent apply did and what it cost this rank (seconds, wall).
        self.last_apply: dict | None = None

        self._inner: DataLoader | None = None
        self._inner_iter = None
        self._samples_yielded: int = 0
        self._pending_reconfigure: bool = False
        # Batch produced by an EAGER apply (see apply_pending); handed out by the next __next__.
        self._primed = None
        self._primed_stop: bool = False
        self._next_s = 0.0   # seconds inside next(inner iterator) since take_handoff_s()

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        self._sampler.set_epoch(epoch)
        # The opt-in keyed interval protocol needs the same workload epoch in
        # every worker, including workers forked later by a live resize. Set it
        # on the dataset before constructing the new inner DataLoader.
        epoch_dataset = self._dataset
        while hasattr(epoch_dataset, "dataset"):
            epoch_dataset = epoch_dataset.dataset
        if hasattr(epoch_dataset, "set_epoch"):
            epoch_dataset.set_epoch(epoch)
        self._samples_yielded = 0
        self._primed = None
        self._primed_stop = False
        if self._keep_pool:
            # The inner DataLoader and its workers outlive the epoch; __iter__ restarts dispatch.
            if self._inner is None:
                self._inner = self._build_inner()
            return
        self._pending_reconfigure = False
        self._inner = self._build_inner()
        self._inner_iter = None
        self._elastic_it = None

    def _build_inner(self) -> DataLoader:
        self._sampler.advance_to(self._samples_yielded)
        kwargs = {}
        pool = self._pool_size()
        if self._generator_factory is not None:
            kwargs["generator"] = self._generator_factory(self._epoch)
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = max(self.prefetch_factor, 1)
            if self._persistent_workers:
                kwargs["persistent_workers"] = True
        dataset, collate = _mot.loader_inputs(self._dataset, self._collate_fn, self._batch_size is not None)
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            sampler=_EpochKeySampler(self._sampler, self) if self._keep_pool else self._sampler,
            num_workers=pool,
            pin_memory=self._pin_memory,
            collate_fn=collate,
            drop_last=False,
            **kwargs,
        )

    def _pool_size(self) -> int:
        """Workers forked at the epoch's fork event: the declared knob range, never fewer than asked.

        Only the first `num_workers` are given tasks (see _ElasticIter). Unset (controller-off arms)
        the pool is exactly `num_workers`, i.e. the stock loader's.
        """
        if not self._elastic or self.num_workers <= 0 or self._persistent_workers:
            return self.num_workers
        cap = int(os.environ.get("STORMER_ELASTIC_POOL_MAX", "0") or 0)
        return max(self.num_workers, cap)

    def reconfigure(
        self,
        num_workers: int | None = None,
        prefetch_factor: int | None = None,
    ):
        changed = False
        if num_workers is not None and num_workers != self.num_workers:
            self.num_workers = num_workers
            changed = True
        if prefetch_factor is not None and prefetch_factor != self.prefetch_factor:
            self.prefetch_factor = prefetch_factor
            changed = True
        if changed:
            self._pending_reconfigure = True
        return changed

    def request_restart(self) -> bool:
        """Request a worker respawn WITHOUT changing num_workers/prefetch_factor.

        The decode knob lives in the worker processes' environment, so propagating a new value
        needs the workers re-forked even though the loader's own parameters are unchanged.
        reconfigure() cannot express that: it compares against the current values and returns
        early, so passing the current values is a silent no-op. With num_workers == 0 there are no
        worker processes and nothing to restart, so this is correctly a no-op there.
        """
        if self.num_workers <= 0:
            return False
        self._pending_reconfigure = True
        return True

    def apply_pending(self, prime: bool = True) -> bool:
        """Materialise a pending reconfigure NOW, in the caller's synchronised window.

        WHY THIS EXISTS (LIVEFLOW-DECODE-RCA.md sec 16). reconfigure() is a REQUEST: it sets a
        flag. The expensive part -- tearing down the DataLoader and forking `num_workers` new
        worker processes -- used to happen lazily in __next__, i.e. AFTER the caller's
        time-symmetry barrier had already passed and reported "apply latency 0.0 ms". Each rank
        therefore respawned its workers at a moment no rank had agreed on, with unbounded and
        highly variable latency. The first rank to finish reached the next Ulysses-SP
        dist.broadcast while its peers were still forking, sat in NCCL for 600 s, and the watchdog
        aborted the job. A barrier can only certify work already done under it.

        Calling this between the consensus and the barrier moves the whole cost inside the
        protected window: stragglers then wait for each other in MPI (no watchdog, bounded by
        arrival) instead of in NCCL (watchdog, fatal).

        prime=True additionally pulls the first batch, so worker warm-up -- not just process
        creation -- is inside the window too. The batch is buffered and handed to the next
        __next__, so the sample stream is unchanged.

        Returns True if an apply was performed.
        """
        if not self._pending_reconfigure:
            return False
        if self._apply_elastic():
            return True
        self._apply_reconfigure()
        if prime:
            try:
                self._primed = next(self._inner_iter)
            except StopIteration:
                self._primed = None
                self._primed_stop = True
        return True

    def _apply_elastic(self) -> bool:
        """Resize the live iterator in place. False means the caller must rebuild instead.

        In place needs a running elastic iterator and workers on both sides of the change (0 <->
        N changes the iterator class). Between epochs there is no iterator yet: the next
        __iter__ simply starts at the new width, which already costs nothing extra.
        """
        it = self._elastic_it
        if it is None or self.num_workers < 1 or self._inner_iter is not it or it._shutdown:
            return False
        if it._sampler_exhausted and not self._keep_pool:
            # Every index of this epoch is already dispatched: nothing left to speed up, and the
            # next epoch's loader is built at the configured width.
            self._pending_reconfigure = False
            self.last_apply = {"mode": "deferred_to_next_epoch", "wall_s": 0.0}
            return True
        t0 = time.perf_counter()
        old = (len(it._active), it._pf_target)
        info = it.resize(self.num_workers, max(self.prefetch_factor, 1))
        self._pending_reconfigure = False
        self.last_apply = dict(info, mode="elastic", wall_s=time.perf_counter() - t0)
        log.info(
            "LiveDataLoader: elastic resize workers %s -> %s, prefetch %s -> %s, admitted=%d parked=%d "
            "forked=%d standby=%d (%.4f s; per-fork %s)",
            old[0], info["workers_after"], old[1], info["prefetch"], info["admitted_from_standby"],
            info["parked"], info["forked"], info["standby"], self.last_apply["wall_s"],
            ",".join("%.3f" % x for x in info["fork_seconds"]))
        return True

    def observed_state(self):
        """(workers, prefetch) the loader is actually running right now."""
        it = self._elastic_it
        if it is not None and self._inner_iter is it:
            if it._shutdown or (it._sampler_exhausted and not self._keep_pool):
                return self.num_workers, self.prefetch_factor
            return len(it._active), it._pf_target
        if self._inner is None:
            return None, None
        return self._inner.num_workers, self._inner.prefetch_factor

    def _apply_reconfigure(self):
        t_rebuild = time.perf_counter()
        old_w = self._inner.num_workers if self._inner else "?"
        # Tear down old DataLoader (kills worker processes)
        self._elastic_it = None
        self._inner_iter = None
        del self._inner
        self._inner = self._build_inner()
        self._inner_iter = self._new_iter()
        self._pending_reconfigure = False
        self.last_apply = {"mode": "rebuild", "wall_s": time.perf_counter() - t_rebuild}
        log.info(
            f"LiveDataLoader: reconfigured workers {old_w} -> {self.num_workers}, "
            f"prefetch={self.prefetch_factor}, "
            f"resuming from sample {self._samples_yielded}"
        )

    def take_handoff_s(self):
        """(blocked_wait_s, handoff_s) accumulated since the last call, or None when the current
        iterator is not instrumented (no workers / stock rebuild path): then the delivery split is
        unavailable and no stall.delivery is emitted (a declared family stays unmeasured).

        blocked_wait_s  time blocked on the worker result queue (upstream: storage, decode, queueing).
        handoff_s       time inside next(loader) NOT blocked: result unpickle/map and dispatch, i.e. the
                        worker -> trainer handoff mechanics. MEASURED, never a residual of the fetch wait:
                        on StorMer the fetch interval also contains the lockstep loop-exit consensus (peer
                        skew), which is neither storage nor delivery. Lower bound: the release of the
                        previous sample's mapping happens when the caller rebinds the batch and is not
                        included (sub-millisecond at StorMer's MB-scale samples; 250 ms at SFNO's 573 MB).
        """
        it = self._elastic_it
        if it is None or self._inner_iter is not it:
            return None
        w, it.wait_s = it.wait_s, 0.0
        n, self._next_s = self._next_s, 0.0
        return w, max(n - w, 0.0)

    def __len__(self) -> int:
        return len(self._sampler)

    def _new_iter(self):
        """Iterator over the current inner DataLoader; elastic when it has workers."""
        if self._elastic and self.num_workers > 0 and not self._persistent_workers:
            it = _ElasticIter(self._inner, active=self.num_workers)
            if self._keep_pool:
                it._persistent_workers = True      # StopIteration must not shut the pool down
            self._elastic_it = it
            return it
        self._elastic_it = None
        return iter(self._inner)

    def __iter__(self):
        it = self._elastic_it
        if self._keep_pool and it is not None and not it._shutdown and self.num_workers > 0:
            it.start_epoch()
            self._inner_iter = it
        else:
            self._inner_iter = self._new_iter()
        self._primed = None
        self._primed_stop = False
        return self

    def __next__(self):
        # A batch primed by an eager apply_pending() is handed out first, so the eager path costs
        # the stream nothing. Kept BEFORE the pending check: apply_pending already cleared the flag.
        if self._primed is not None:
            batch = _mot.unwrap(self._primed)
            self._primed = None
            self._samples_yielded += self._batch_size or 1
            return batch
        if self._primed_stop:
            self._primed_stop = False
            raise StopIteration
        # Fallback: a reconfigure that was requested but never materialised eagerly. This is the
        # old lazy path and it is UNSAFE under peer-group collectives (sec 16) -- kept only so a
        # caller that does not use apply_pending() still behaves correctly.
        if self._pending_reconfigure:
            if not self._apply_elastic():
                self._apply_reconfigure()

        t_next = time.perf_counter()
        try:
            _mot.wrap_get_data(self._inner_iter)
            batch = _mot.unwrap(next(self._inner_iter))
        except StopIteration:
            raise
        self._next_s += time.perf_counter() - t_next

        self._samples_yielded += self._batch_size or 1
        return batch
