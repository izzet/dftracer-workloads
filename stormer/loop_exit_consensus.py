"""Fail closed on unequal StorMer loader exhaustion before DDP backward.

CPU-side mpi4py agreement; opt-in with STORMER_LOOP_EXIT_CONSENSUS=1.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator, Optional


class UnequalLoaderExhaustion(RuntimeError):
    """At least one rank has no next batch while a peer does."""


class LoopExitConsensus:
    """One CPU MPI collective per batch, before any rank enters backward."""

    def __init__(self, mpi_utils):
        self._mpi = mpi_utils
        self._active = mpi_utils.size() > 1 and mpi_utils.is_mpi()

    def should_stop(self, local_stop: bool) -> bool:
        if not self._active:
            return local_stop
        from mpi4py import MPI
        stopped = self._mpi.comm_world().allreduce(int(local_stop), op=MPI.SUM)
        if stopped == 0:
            return False
        if stopped == self._mpi.size():
            return True
        raise UnequalLoaderExhaustion(
            f"UNEQUAL_LOADER_EXHAUSTION: {stopped}/{self._mpi.size()} ranks exhausted "
            "before the next training batch"
        )


class PaddedRankSyncIterator:
    """Legacy name retained for callers; this iterator never emits padding.

    Advance each source once, then agree before yielding. A mixed exhaustion
    answer raises on every rank. max_pad_steps remains accepted for old callers
    but has no effect.
    """

    def __init__(self, source: Iterator[Any], consensus: LoopExitConsensus, max_pad_steps: int = 1_000_000):
        self._source = source
        self._consensus = consensus

    def __iter__(self) -> "PaddedRankSyncIterator":
        return self

    def __next__(self):
        try:
            batch = next(self._source)
            exhausted = False
        except StopIteration:
            batch = None
            exhausted = True
        if self._consensus.should_stop(exhausted):
            raise StopIteration
        if exhausted:
            raise UnequalLoaderExhaustion("UNEQUAL_LOADER_EXHAUSTION: local rank exhausted")
        return batch, False


def wrap_step_iterator(base_iter: Iterator[Any], consensus: Optional[LoopExitConsensus]):
    """Return (batch, False) pairs; disabled mode preserves source order."""
    if consensus is None:
        return ((batch, False) for batch in base_iter)
    return PaddedRankSyncIterator(base_iter, consensus)


def _write_json_fsync(path: Path, value: dict) -> str:
    """Atomically publish a small record on the shared run filesystem."""
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return hashlib.sha256(payload).hexdigest()


def retain_unequal_exhaustion(comm, *, rank: int, epoch: int, step: int,
                              updates: int, reported_steps: int,
                              evidence_dir: str | Path, log: logging.Logger) -> bool:
    """Flush each rank, Gather, publish one rank-zero exclusion, then release peers.

    The final broadcast is essential: a nonroot rank must not call Abort(97)
    while rank zero is still writing the all-rank record and marker.
    Every rank calls this only after the same mixed-exhaustion SUM collective.
    """
    root = Path(evidence_dir)
    local = {"rank": rank, "epoch": epoch, "step": step,
             "optimizer_updates": updates, "reported_steps": reported_steps}
    log.info("STEP_SERIES_END rank=%d epoch=%d optimizer_updates=%d reported_steps=%d",
             rank, epoch, updates, reported_steps)
    for handler in list(getattr(log, "handlers", ())) + list(logging.getLogger().handlers):
        try:
            handler.flush()
        except Exception:
            pass
    local_error = ""
    try:
        root.mkdir(parents=True, exist_ok=True)
        _write_json_fsync(root / f"rank-{rank}-update-count.json", local)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
        log.exception("Could not retain local unequal-exhaustion update count")
    gathered = comm.gather({**local, "write_error": local_error}, root=0)
    published = False
    if rank == 0:
        try:
            if sorted(row["rank"] for row in gathered) != list(range(comm.Get_size())):
                raise ValueError("unequal-exhaustion Gather did not cover all ranks")
            record = {"schema_version": "stormer-unequal-exhaustion.v1",
                      "reason": "UNEQUAL_LOADER_EXHAUSTION", "ranks": gathered}
            record_sha = _write_json_fsync(root / "unequal-loader-all-ranks.json", record)
            marker = {"reason": "UNEQUAL_LOADER_EXHAUSTION",
                      "all_rank_record_sha256": record_sha,
                      "all_rank_update_records_flushed": all(not row["write_error"] for row in gathered)}
            _write_json_fsync(root / "unequal-loader-ineligible.json", marker)
            published = True
        except Exception:
            log.exception("Could not publish all-rank unequal-exhaustion evidence")
    # This acknowledges the rank-zero fsync to every peer before any Abort.
    published = comm.bcast(published, root=0)
    return bool(published)
