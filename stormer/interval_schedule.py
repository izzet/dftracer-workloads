"""Pure, versioned interval selection for opt-in StorMer loader wiring.

Callers must provide the epoch and split explicitly so a sample's interval choice is stable
across DataLoader workers, ranks, process restarts, and replay.  The selector
does not seed or read any Python, NumPy, or Torch random state.
"""

import hashlib
import json


INTERVAL_SELECTOR_VERSION = "stormer-interval-selector.v1"


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a nonnegative integer" % name)
    return value


def _validate_intervals(intervals):
    if not isinstance(intervals, tuple):
        raise TypeError("intervals must be an ordered tuple of positive integers")
    if not intervals:
        raise ValueError("intervals must not be empty")
    for interval in intervals:
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise ValueError("intervals must contain only positive integers")
    return intervals


def select_interval(
    intervals,
    *,
    experiment_seed,
    epoch,
    dataset_index,
    split,
    version_tag=INTERVAL_SELECTOR_VERSION,
):
    """Select one ordered interval from stable, explicitly supplied context.

    ``intervals`` is deliberately an ordered tuple: reordering it changes the
    mapped interval while preserving the digest-derived ordinal.  The digest
    payload includes no worker, rank, PID, process, or global RNG state.
    """
    intervals = _validate_intervals(intervals)
    experiment_seed = _nonnegative_int(experiment_seed, "experiment_seed")
    epoch = _nonnegative_int(epoch, "epoch")
    dataset_index = _nonnegative_int(dataset_index, "dataset_index")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a nonempty string")
    if not isinstance(version_tag, str) or not version_tag:
        raise ValueError("version_tag must be a nonempty string")

    payload = json.dumps(
        [version_tag, experiment_seed, epoch, dataset_index, split],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    ordinal = int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big") % len(intervals)
    return intervals[ordinal]
