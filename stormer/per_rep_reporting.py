"""Structured, launcher-owned StorMer per-repetition timing evidence.

This module is intentionally independent of analyzer, controller, and trajectory
state.  It writes rank-local logical-update records and reduces only this schema;
it never scrapes progress/log output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "stormer-per-rep-v1"
TIMING_RULE_ID = "stormer-batch-step-wall-v1-warmup20-measure100"
WARMUP_UPDATES = 20
MEASUREMENT_UPDATES = 100
REQUIRED_UPDATES = WARMUP_UPDATES + MEASUREMENT_UPDATES
TRAILING_BATCH_STEP = REQUIRED_UPDATES + 1
POOLABLE_ARMS = {"original", "original-notrace", "maxresource", "static-best", "glanced-ofat", "autotune", "loop"}
FIXED_CELL_FIELDS = (
    "num_nodes", "ppn", "total_ranks", "epochs", "max_training_step", "batch_size",
    "accumulate_grad_batches", "precision", "in_img_size", "patch_size", "depth", "num_heads",
    "mlp_ratio", "ulysses_sp", "seed", "intervals_hours", "data_freq_hours", "interval_protocol",
    "interval_index_manifest_sha256", "latitude_manifest_sha256", "data_dir",
    "dataset_content_identity_sha256", "downsample", "learning_rate", "beta_1", "beta_2",
    "weight_decay", "scheduler_warmup_epochs", "scheduler_warmup_start_lr", "scheduler_eta_min",
    "source_manifest_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_cell_key(config: dict[str, Any]) -> tuple[str, str]:
    """Digest all fixed factors; W/PF are intentionally absent treatment factors."""
    missing = [name for name in FIXED_CELL_FIELDS if config.get(name) in (None, "", [])]
    if missing:
        return "", "FIXED_CELL_IDENTITY_MISSING:" + ",".join(missing)
    payload = {name: config[name] for name in FIXED_CELL_FIELDS}
    return "stormer-fixed-v1-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest(), ""


class StepSeriesWriter:
    """Append-only, flushed rank-local logical-batch-step timing records."""

    def __init__(self, directory: str | Path, rank: int):
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"rank-{rank}.jsonl"
        self.rank = rank
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, logical_batch_step_id: int, epoch: int, step: int, total_ms: float,
              completion_status: str) -> None:
        if (not isinstance(logical_batch_step_id, int) or isinstance(logical_batch_step_id, bool)
                or logical_batch_step_id < 1):
            raise ValueError("logical_batch_step_id must be a positive integer")
        if not math.isfinite(total_ms) or total_ms <= 0:
            raise ValueError("total_ms must be finite and positive")
        if completion_status not in {"completed", "nonfinite_loss_skipped"}:
            raise ValueError("unknown completion_status")
        row = {
            "schema_version": SCHEMA_VERSION,
            "timing_semantics": "post_previous_control_to_gpu_synchronized_current_control_end",
            "logical_batch_step_id": logical_batch_step_id,
            "epoch": epoch,
            "step": step,
            "rank": self.rank,
            "total_ms": total_ms,
            "completion_status": completion_status,
        }
        self._handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

def _failure_reason(run_dir: Path, expected_ranks: int) -> str:
    evidence = run_dir / "loop-exit-evidence"
    marker = evidence / "unequal-loader-ineligible.json"
    if not marker.is_file():
        return ""
    record = evidence / "unequal-loader-all-ranks.json"
    try:
        flag = json.loads(marker.read_text(encoding="utf-8"))
        all_ranks = json.loads(record.read_text(encoding="utf-8"))
        if (flag.get("reason") != "UNEQUAL_LOADER_EXHAUSTION"
                or flag.get("all_rank_record_sha256") != sha256_file(record)
                or all_ranks.get("reason") != "UNEQUAL_LOADER_EXHAUSTION"
                or sorted(row["rank"] for row in all_ranks["ranks"]) != list(range(expected_ranks))):
            return "INVALID_LOOP_EXIT_MARKER"
    except (OSError, ValueError, KeyError, TypeError):
        return "INVALID_LOOP_EXIT_MARKER"
    return "UNEQUAL_LOADER_EXHAUSTION"


ELAPSED_SCHEMA_VERSION = "stormer-long-cell-elapsed-v2"


def write_elapsed_record(directory: str | Path, rank: int, start_s: float, end_s: float,
                         n_updates: int, full_loop_end_s: float,
                         wall_start_s: float, wall_end_s: float) -> Path:
    """Retain an independent monotonic loop-entry-to-N span and a separate full-loop span."""
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    if type(n_updates) is not int or n_updates < 1:
        raise ValueError("n_updates must be a positive integer")
    if not all(math.isfinite(value) for value in
               (start_s, end_s, full_loop_end_s, wall_start_s, wall_end_s)):
        raise ValueError("elapsed timestamps must be finite")
    if not start_s < end_s <= full_loop_end_s:
        raise ValueError("elapsed timestamps must be ordered")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank-{rank}.elapsed.json"
    path.write_text(json.dumps({
        "schema_version": ELAPSED_SCHEMA_VERSION, "rank": rank, "clock": "monotonic",
        "n_updates": n_updates, "start_s": start_s, "end_s": end_s,
        "full_loop_end_s": full_loop_end_s,
        "wall_start_s": wall_start_s, "wall_end_s": wall_end_s,
    }, sort_keys=True))
    return path


def _load_elapsed_records(series_dir: Path, expected_ranks: int, total_updates: int
                           ) -> tuple[dict[int, dict[str, float]], str]:
    result: dict[int, dict[str, float]] = {}
    for rank in range(expected_ranks):
        path = series_dir / f"rank-{rank}.elapsed.json"
        if not path.is_file():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("schema_version") != ELAPSED_SCHEMA_VERSION or row.get("rank") != rank
                or row.get("clock") != "monotonic" or row.get("n_updates") != total_updates):
            return {}, "LONG_CELL_ELAPSED_RECORD_INVALID"
        values = {name: row.get(name) for name in
                  ("start_s", "end_s", "full_loop_end_s", "wall_start_s", "wall_end_s")}
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or
               not math.isfinite(value) for value in values.values()):
            return {}, "LONG_CELL_ELAPSED_RECORD_INVALID"
        if not values["start_s"] < values["end_s"] <= values["full_loop_end_s"]:
            return {}, "LONG_CELL_ELAPSED_NONMONOTONIC"
        result[rank] = values
    if set(result) != set(range(expected_ranks)):
        return {}, "LONG_CELL_ELAPSED_COVERAGE"
    return result, ""


def _clock_resolution_epsilon_ms() -> float:
    """Coarser resolution of the two monotonic clocks used for rank-local intervals."""
    return max(time.get_clock_info("monotonic").resolution,
               time.get_clock_info("perf_counter").resolution) * 1000.0


def validate_long_cell_elapsed(series_dir: str | Path, expected_ranks: int, total_updates: int,
                               total_wall_ms: float
                                ) -> tuple[dict[str, float] | None, str]:
    """Validate each rank against its own interval; cross-rank gap is diagnostic only."""
    records, reason = _load_elapsed_records(Path(series_dir), expected_ranks, total_updates)
    if reason:
        return None, reason
    ranks = _load_rank_series(Path(series_dir), expected_ranks)
    if set(ranks) != set(range(expected_ranks)) or any(
            any(update not in ranks[rank] for update in range(1, total_updates + 1))
            for rank in ranks):
        return None, "LONG_CELL_ELAPSED_SERIES_INCOMPLETE"
    epsilon_ms = _clock_resolution_epsilon_ms()
    spans = {rank: (record["end_s"] - record["start_s"]) * 1000.0
             for rank, record in records.items()}
    for rank, span in spans.items():
        measured_ms = sum(ranks[rank][update][0] for update in range(1, total_updates + 1))
        if measured_ms > span + epsilon_ms:
            return None, "LONG_CELL_RANK_INTERVAL_EXCEEDS_ELAPSED"
    elapsed_ms = max(spans.values())
    return {"elapsed_ms": elapsed_ms, "gap_ms": elapsed_ms - total_wall_ms,
            "full_loop_ms": max((record["full_loop_end_s"] - record["start_s"]) * 1000.0
                                for record in records.values())}, ""


def _load_rank_series(series_dir: Path, expected_ranks: int) -> dict[int, dict[int, tuple[float, str, int, int]]]:
    result: dict[int, dict[int, tuple[float, str, int, int]]] = {}
    for rank in range(expected_ranks):
        path = series_dir / f"rank-{rank}.jsonl"
        if not path.is_file():
            continue
        updates: dict[int, tuple[float, str, int, int]] = {}
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            if row.get("schema_version") != SCHEMA_VERSION or row.get("rank") != rank:
                raise ValueError(f"{path}:{line_number}: schema or rank mismatch")
            update = row.get("logical_batch_step_id")
            value = row.get("total_ms")
            status = row.get("completion_status")
            epoch = row.get("epoch")
            step = row.get("step")
            if (status not in {"completed", "nonfinite_loss_skipped"}
                    or not isinstance(update, int) or isinstance(update, bool) or update < 1
                    or not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0
                    or not isinstance(step, int) or isinstance(step, bool) or step < 1
                    or not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0 or update in updates):
                raise ValueError(f"{path}:{line_number}: invalid or duplicate timing")
            updates[update] = (float(value), status, epoch, step)
        result[rank] = updates
    return result


def _load_and_validate_ranks(series_dir: str | Path, expected_ranks: int, trailing_batch_step: int
                              ) -> tuple[dict[int, dict[int, tuple[float, str, int, int]]] | None, str]:
    """Shared integrity checks (coverage, finiteness, epoch-tail quarantine, rank-step agreement)
    for updates 1..trailing_batch_step. Used by both the fixed 120-step rule and the long-cell rule
    (E3.4 rebind) so the same failure modes apply at any declared cell size."""
    if not isinstance(expected_ranks, int) or isinstance(expected_ranks, bool) or expected_ranks < 1:
        raise ValueError("expected_ranks must be a positive integer")
    ranks = _load_rank_series(Path(series_dir), expected_ranks)
    if set(ranks) != set(range(expected_ranks)):
        return None, "RAW_SERIES_RANK_COVERAGE"
    required = range(1, trailing_batch_step + 1)
    if any(any(update not in ranks[rank] for update in required) for rank in ranks):
        return None, "NO_STEADY_TIME"
    if any(ranks[rank][update][1] != "completed" for rank in ranks for update in required):
        return None, "NONFINITE_BATCH_STEP"
    # The writer records before epoch-tail Window.flush(). A same-epoch trailing
    # batch quarantines the measured region from that final drain/control work;
    # otherwise an epoch transition could hide it from the timing region.
    if (any(len({ranks[rank][update][2] for update in required}) != 1 for rank in ranks)
            or any(len({ranks[rank][update][2] for rank in ranks}) != 1 for update in required)):
        return None, "EPOCH_TAIL_NOT_QUARANTINED"
    if any(len({ranks[rank][update][3] for rank in ranks}) != 1 for update in required):
        return None, "RANK_STEP_MISMATCH"
    return ranks, ""


def reduce_critical_wall_ms(series_dir: str | Path, expected_ranks: int) -> tuple[float | None, str, int]:
    """Return fixed-rule mean critical-rank time (steps WARMUP_UPDATES+1..REQUIRED_UPDATES), or a
    declared exclusion. UNCHANGED behavior/schema: this is the original 120-step-cell rule
    (TIMING_RULE_ID), still used by every existing campaign; do not repoint it at a different cell."""
    ranks, reason = _load_and_validate_ranks(series_dir, expected_ranks, TRAILING_BATCH_STEP)
    if ranks is None:
        return None, reason, 0
    critical = [max(ranks[rank][update][0] for rank in ranks)
                for update in range(WARMUP_UPDATES + 1, REQUIRED_UPDATES + 1)]
    return sum(critical) / len(critical), "", len(critical)


def reduce_long_cell_wall_ms(series_dir: str | Path, expected_ranks: int, total_updates: int,
                              tail_window: int) -> tuple[dict[str, Any] | None, str]:
    """Long-cell (e.g. 1500-step) diagnostics: sum of stepwise rank maxima to
    `total_updates` and mean/median ms/step over the last `tail_window` steps (descriptive;
    reported unconditionally, it does not assert the window is 'steady'). Independent of, and does
    not replace, reduce_critical_wall_ms: a 1500-step cell's fixed_cell_key already differs from a
    120-step cell's (max_training_step is a FIXED_CELL_FIELD), so the two never pool together."""
    if not isinstance(total_updates, int) or isinstance(total_updates, bool) or total_updates < 1:
        raise ValueError("total_updates must be a positive integer")
    if not isinstance(tail_window, int) or isinstance(tail_window, bool) or not (0 < tail_window <= total_updates):
        raise ValueError("tail_window must be a positive integer <= total_updates")
    trailing_batch_step = total_updates + 1
    ranks, reason = _load_and_validate_ranks(series_dir, expected_ranks, trailing_batch_step)
    if ranks is None:
        return None, reason
    critical = [max(ranks[rank][update][0] for rank in ranks) for update in range(1, total_updates + 1)]
    tail = critical[-tail_window:]
    sorted_tail = sorted(tail)
    mid = len(sorted_tail) // 2
    median_tail = (sorted_tail[mid] if len(sorted_tail) % 2
                   else (sorted_tail[mid - 1] + sorted_tail[mid]) / 2.0)
    return {
        "total_wall_ms": sum(critical),
        "n_total": len(critical),
        "tail_window": tail_window,
        "tail_mean_ms_per_step": sum(tail) / len(tail),
        "tail_median_ms_per_step": median_tail,
        "per_window_ms_per_step": critical,
    }, ""


def build_report_row(args: argparse.Namespace) -> dict[str, str]:
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    source_path = run_dir / "submission-sources.sha256"
    verdict_path = run_dir / "shm-verdict.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    steady_ms, exclusion, n_steady = reduce_critical_wall_ms(args.series_dir, args.expected_ranks)
    exclusion = _failure_reason(run_dir, args.expected_ranks) or exclusion
    long_cell_total = getattr(args, "long_cell_total_updates", None)
    long_cell_tail = getattr(args, "long_cell_tail_window", None)
    long_cell = None
    elapsed = None
    if long_cell_total is None and long_cell_tail is not None:
        exclusion = exclusion or "LONG_CELL_TAIL_WITHOUT_N"
    if long_cell_total is not None:
        resolved_tail = long_cell_total if long_cell_tail is None else long_cell_tail
        if (type(config.get("max_training_step")) is not int
                or config["max_training_step"] != long_cell_total + 1
                or config.get("long_cell_total_updates") != long_cell_total
                or config.get("long_cell_tail_window") != resolved_tail
                or config.get("epochs") != 1
                or config.get("accumulate_grad_batches") != 1):
            exclusion = exclusion or "LONG_CELL_CONFIG_N_MISMATCH"
        long_cell, long_cell_exclusion = reduce_long_cell_wall_ms(
            args.series_dir, args.expected_ranks, long_cell_total, resolved_tail)
        if long_cell is None:
            exclusion = exclusion or long_cell_exclusion
        else:
            # The independently observed loop-entry-to-N interval is the primary endpoint.
            # The stepwise rank maxima are a diagnostic that may exceed this interval.
            elapsed, elapsed_reason = validate_long_cell_elapsed(
                args.series_dir, args.expected_ranks, long_cell_total, long_cell["total_wall_ms"])
            if elapsed is None:
                exclusion = exclusion or elapsed_reason
    scale, fixed_cell_error = fixed_cell_key(config)
    exclusion = exclusion or fixed_cell_error
    if args.arm not in POOLABLE_ARMS:
        exclusion = exclusion or "UNKNOWN_ARM"
    if not source_path.is_file():
        exclusion = exclusion or "SOURCE_MANIFEST_MISSING"
    elif config.get("source_manifest_sha256") != sha256_file(source_path):
        exclusion = exclusion or "SOURCE_MANIFEST_HASH_MISMATCH"
    verdict_ok = False
    verdict_mode = ""
    if verdict_path.is_file():
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        verdict_ok = verdict.get("ok") is True
        verdict_mode = verdict.get("mode", "shm")
    if args.training_rc != 0:
        exclusion = exclusion or f"RC_{args.training_rc}"
    reference = args.arm == "original-notrace"
    if not verdict_ok or verdict_mode != ("original_reference" if reference else "shm"):
        exclusion = exclusion or ("REFERENCE_VERDICT_MISSING_OR_RED" if reference
                                  else "SHM_VERDICT_MISSING_OR_RED")
    raw_dir = Path(args.series_dir)
    window_series_path = ""
    window_series_sha256 = ""
    if long_cell and not exclusion:
        series_file = raw_dir / "long-cell-window-series.json"
        series_file.write_text(json.dumps({
            "schema_version": "stormer-long-cell-window-series.v1",
            "total_updates": long_cell_total, "tail_window": long_cell["tail_window"],
            "per_window_ms_per_step": long_cell["per_window_ms_per_step"],
        }, sort_keys=True))
        window_series_path = str(series_file)
        window_series_sha256 = sha256_file(series_file)
    return {
        "record_version": SCHEMA_VERSION,
        "workload": "stormer",
        "cluster": "tuolumne",
        "arm": args.arm,
        "run_id": args.run_id,
        "rep": str(args.rep),
        "block_id": args.block_id,
        "scale": scale,
        "fixed_cell_fields": ",".join(FIXED_CELL_FIELDS),
        "nodes": str(config["num_nodes"]),
        "ppn": str(config["ppn"]),
        "ranks": str(config["total_ranks"]),
        "timing_rule": TIMING_RULE_ID,
        "warmup_steps": str(WARMUP_UPDATES),
        "steady_first_step": str(WARMUP_UPDATES + 1),
        "steady_last_step": str(REQUIRED_UPDATES),
        "required_trailing_step": str(TRAILING_BATCH_STEP),
        # pooled_table.py has no Tuolumne-side rc/verdict exclusion hook. Leave
        # the accepted timing field blank for every excluded row so it cannot
        # become a usable replicate while retaining the raw evidence below.
        "steady_ms_per_step": "" if steady_ms is None or exclusion else format(steady_ms, ".17g"),
        "n_steady": str(n_steady),
        "rc": str(args.training_rc),
        "flux_allocation_id": args.allocation_job_id,
        "flux_training_job_id": args.training_job_id,
        "shm_verdict_path": str(verdict_path),
        "shm_verdict_sha256": sha256_file(verdict_path) if verdict_path.is_file() else "",
        "shm_ok": str(verdict_ok and not reference).lower(),
        "reference_ok": str(verdict_ok and reference).lower(),
        "verdict_mode": verdict_mode,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        # Explicit copies of the config's dataset-content-identity binding, so the
        # reducer can rebind and re-verify (via verify_stormer_dataset_identity.py)
        # without silently trusting an unparsed config_path/config_sha256 pair.
        # The reviewed owner-freeze reference is captured by the launcher in
        # config.json, so the reducer can bind this row to the same run record.
        "identity_sha256": str(config.get("dataset_content_identity_sha256", "")),
        "identity_bundle": str(config.get("dataset_content_identity_bundle", "")),
        # v2.1 (identity + delta) binding: empty for a v1/base bundle row, unchanged from before.
        # When set, the reducer verifies via stormer_dataset_identity_delta.verify_identity_v2_1
        # instead of the v1 path (see pooled_table.py); identity_sha256 above is then the reviewed
        # digest of THIS envelope file, not of a bare identity.json.
        "identity_v2_1_envelope": str(config.get("dataset_content_identity_v2_1_envelope", "")),
        "identity_v2_2_envelope": str(config.get("dataset_content_identity_v2_2_envelope", "")),
        "identity_data_root": str(config.get("data_dir", "")),
        "identity_latitude_manifest": (
            str(Path(config["latitude_metadata_dir"]) / "manifest.json")
            if config.get("latitude_metadata_dir") else ""
        ),
        "identity_interval_manifest": str(config.get("interval_index_manifest", "")),
        "identity_write_freeze_reference": str(config.get("dataset_write_freeze_reference", "")),
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": sha256_file(source_path) if source_path.is_file() else "",
        "raw_step_series_path": str(raw_dir),
        "raw_step_series_sha256": sha256_file(raw_dir / "series-manifest.sha256") if (raw_dir / "series-manifest.sha256").is_file() else "",
        "loop_exit_marker_path": str(run_dir / "loop-exit-evidence/unequal-loader-ineligible.json") if (run_dir / "loop-exit-evidence/unequal-loader-ineligible.json").is_file() else "",
        "loop_exit_marker_sha256": sha256_file(run_dir / "loop-exit-evidence/unequal-loader-ineligible.json") if (run_dir / "loop-exit-evidence/unequal-loader-ineligible.json").is_file() else "",
        "loop_exit_all_rank_path": str(run_dir / "loop-exit-evidence/unequal-loader-all-ranks.json") if (run_dir / "loop-exit-evidence/unequal-loader-all-ranks.json").is_file() else "",
        "loop_exit_all_rank_sha256": sha256_file(run_dir / "loop-exit-evidence/unequal-loader-all-ranks.json") if (run_dir / "loop-exit-evidence/unequal-loader-all-ranks.json").is_file() else "",
        "raw_step_count": str(sum(len(v) for v in _load_rank_series(raw_dir, args.expected_ranks).values())),
        # The max-rank loop-entry-to-N span is the primary endpoint. Full-loop and
        # sum-of-stepwise-rank-maxima spans remain separate diagnostics; their gap
        # can have either sign because rank-local step boundaries need not align.
        # tail_* is the descriptive last-window figure (reported unconditionally, per E3.5.2, not
        # gated on "steady"); per-window series is written to its own file
        # (raw_dir/long-cell-window-series.json) rather than inlined, same pointer+hash pattern as
        # raw_step_series_path/sha256 above.
        "long_cell_total_updates": str(long_cell_total) if long_cell_total is not None else "",
        "long_cell_tail_window": str(long_cell["tail_window"]) if long_cell else "",
        "long_cell_elapsed_ms": format(elapsed["elapsed_ms"], ".17g") if elapsed and not exclusion else "",
        "total_wall_ms": format(long_cell["total_wall_ms"], ".17g") if long_cell and not exclusion else "",
        "long_cell_gap_ms": format(elapsed["gap_ms"], ".17g") if elapsed and not exclusion else "",
        "long_cell_full_loop_ms": format(elapsed["full_loop_ms"], ".17g") if elapsed and not exclusion else "",
        "tail_mean_ms_per_step": format(long_cell["tail_mean_ms_per_step"], ".17g") if long_cell and not exclusion else "",
        "tail_median_ms_per_step": format(long_cell["tail_median_ms_per_step"], ".17g") if long_cell and not exclusion else "",
        "long_cell_window_series_path": window_series_path,
        "long_cell_window_series_sha256": window_series_sha256,
        "eligible": str(not exclusion).lower(),
        "exclusion_reason": exclusion,
    }


def write_report(args: argparse.Namespace) -> Path:
    row = build_report_row(args)
    series_dir = Path(args.series_dir)
    series_dir.mkdir(parents=True, exist_ok=True)
    manifest = series_dir / "series-manifest.sha256"
    with manifest.open("w", encoding="utf-8") as handle:
        for path in sorted((*series_dir.glob("rank-*.jsonl"),
                            *series_dir.glob("rank-*.elapsed.json"))):
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    row["raw_step_series_sha256"] = sha256_file(manifest)
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--series-dir", required=True)
    parser.add_argument("--expected-ranks", required=True, type=int)
    parser.add_argument("--training-rc", required=True, type=int)
    parser.add_argument("--training-job-id", default="unknown")
    parser.add_argument("--allocation-job-id", default="unknown")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rep", required=True)
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--long-cell-total-updates", type=int, default=None,
                         help="Report loop-entry-to-N elapsed plus stepwise and tail diagnostics for a cell "
                              "longer than the fixed 120-step rule (e.g. 1500). Omit for existing "
                              "120-step campaigns; steady_ms_per_step/timing_rule are unaffected.")
    parser.add_argument("--long-cell-tail-window", type=int, default=None,
                         help="descriptive last-N-step window for --long-cell-total-updates "
                              "(default: the full long-cell range, i.e. no tail exclusion).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    write_report(parse_args())
