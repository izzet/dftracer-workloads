import argparse
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from per_rep_reporting import (MEASUREMENT_UPDATES, REQUIRED_UPDATES, TRAILING_BATCH_STEP,
                               StepSeriesWriter, WARMUP_UPDATES, fixed_cell_key,
                               reduce_critical_wall_ms, write_report)


POOL_SPEC = importlib.util.spec_from_file_location(
    "pooled_table", Path(__file__).parents[3] / "liveflow_shm/tests/pooled_table.py"
)
POOL = importlib.util.module_from_spec(POOL_SPEC)
assert POOL_SPEC.loader is not None
POOL_SPEC.loader.exec_module(POOL)


def write_series(root, ranks=2, updates=TRAILING_BATCH_STEP, epoch=0):
    for rank in range(ranks):
        writer = StepSeriesWriter(root, rank)
        for update in range(1, updates + 1):
            writer.write(update, epoch, update, 10.0 + rank, "completed")
        writer.close()


def base_config(**overrides):
    config = {
        "num_nodes": 1, "ppn": 1, "total_ranks": 1, "epochs": 1, "max_training_step": 121,
        "num_workers": 1, "prefetch_factor": 2, "batch_size": 1, "accumulate_grad_batches": 1,
        "precision": "32", "in_img_size": [128, 256], "patch_size": 16, "depth": 2,
        "num_heads": 16, "mlp_ratio": 4.0, "ulysses_sp": 1, "seed": 42,
        "intervals_hours": [6, 12, 24], "data_freq_hours": 6, "interval_protocol": "legacy",
        "interval_index_manifest_sha256": "none", "latitude_manifest_sha256": "a" * 64,
        "data_dir": "/native/era5", "dataset_content_identity_sha256": "b" * 64,
        "dataset_content_identity_bundle": "/identity/complete",
        "dataset_write_freeze_reference": "fixture-owner",
        "latitude_metadata_dir": "/latitude-v2",
        "interval_index_manifest": "/interval-v1/manifest.json",
        "downsample": "1", "learning_rate": 5e-4, "beta_1": 0.9, "beta_2": 0.95,
        "weight_decay": 1e-5, "scheduler_warmup_epochs": 10,
        "scheduler_warmup_start_lr": 1e-8, "scheduler_eta_min": 1e-8,
        "source_manifest_sha256": "",  # filled after the retained file is written
    }
    config.update(overrides)
    return config


def write_config_and_source(run, **overrides):
    source = run / "submission-sources.sha256"
    source.write_text("abc  source.py\n")
    if overrides.get("max_training_step", 0) > 121:
        overrides.setdefault("long_cell_total_updates", overrides["max_training_step"] - 1)
        overrides.setdefault("long_cell_tail_window", 500)
    config = base_config(**overrides)
    import hashlib
    config["source_manifest_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (run / "config.json").write_text(json.dumps(config))


def test_fixed_rule_uses_critical_rank_and_never_selects_by_observation(tmp_path):
    series = tmp_path / "series"
    write_series(series)
    value, reason, count = reduce_critical_wall_ms(series, 2)
    assert value == 11.0
    assert reason == ""
    assert count == MEASUREMENT_UPDATES


def test_short_architecture_rung_is_ineligible(tmp_path):
    series = tmp_path / "series"
    write_series(series, ranks=1, updates=12)
    value, reason, count = reduce_critical_wall_ms(series, 1)
    assert value is None
    assert reason == "NO_STEADY_TIME"
    assert count == 0


def test_exactly_120_steps_are_not_eligible_without_a_trailing_epoch_batch(tmp_path):
    series = tmp_path / "series"
    write_series(series, ranks=1, updates=REQUIRED_UPDATES)
    value, reason, count = reduce_critical_wall_ms(series, 1)
    assert value is None
    assert reason == "NO_STEADY_TIME"
    assert count == 0


def test_trailing_batch_must_stay_in_the_measurement_epoch(tmp_path):
    series = tmp_path / "series"
    write_series(series, ranks=1)
    path = series / "rank-0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["epoch"] = 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    value, reason, count = reduce_critical_wall_ms(series, 1)
    assert value is None
    assert reason == "EPOCH_TAIL_NOT_QUARANTINED"
    assert count == 0


def test_epoch_key_must_agree_across_ranks(tmp_path):
    series = tmp_path / "series"
    write_series(series, ranks=2)
    path = series / "rank-1.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["epoch"] = 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    value, reason, count = reduce_critical_wall_ms(series, 2)
    assert value is None
    assert reason == "EPOCH_TAIL_NOT_QUARANTINED"
    assert count == 0


def test_local_step_must_agree_across_ranks(tmp_path):
    series = tmp_path / "series"
    write_series(series, ranks=2)
    path = series / "rank-1.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[30]["step"] += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    value, reason, count = reduce_critical_wall_ms(series, 2)
    assert value is None
    assert reason == "RANK_STEP_MISMATCH"
    assert count == 0


def test_nonfinite_batch_step_is_retained_but_not_eligible(tmp_path):
    series = tmp_path / "series"
    write_series(series, ranks=1)
    writer = StepSeriesWriter(series, 0)
    # A duplicate is invalid by design, so construct a small independent series.
    writer.close()
    path = series / "rank-0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[30]["completion_status"] = "nonfinite_loss_skipped"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    value, reason, count = reduce_critical_wall_ms(series, 1)
    assert value is None
    assert reason == "NONFINITE_BATCH_STEP"
    assert count == 0


def test_report_binds_verdict_hashes_and_exact_cell(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_config_and_source(run)
    (run / "shm-verdict.json").write_text('{"ok": true}\n')
    series = run / "per-rep-step-series"
    write_series(series, ranks=1)
    output = run / "per-rep-report.csv"
    args = argparse.Namespace(run_dir=str(run), series_dir=str(series), expected_ranks=1,
                              training_rc=0, training_job_id="f123", allocation_job_id="f122",
                              arm="original", run_id="r1", rep="1", block_id="b1", scale="",
                              output=str(output))
    write_report(args)
    row = next(csv.DictReader(output.open()))
    assert row["eligible"] == "true"
    assert row["shm_ok"] == "true"
    assert row["flux_training_job_id"] == "f123"
    assert row["n_steady"] == str(MEASUREMENT_UPDATES)
    assert row["steady_first_step"] == str(WARMUP_UPDATES + 1)
    assert row["required_trailing_step"] == str(TRAILING_BATCH_STEP)
    assert row["scale"].startswith("stormer-fixed-v1-")
    assert row["identity_write_freeze_reference"] == "fixture-owner"
    assert row["identity_bundle"] == "/identity/complete"
    assert "num_workers" not in row["fixed_cell_fields"]
    assert "prefetch_factor" not in row["fixed_cell_fields"]


def test_original_notrace_uses_application_verdict_without_claiming_shm(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_config_and_source(run, liveio_mode="original_reference")
    (run / "shm-verdict.json").write_text(
        '{"ok": true, "mode": "original_reference", "reporting_checked": true}\n')
    series = run / "per-rep-step-series"
    write_series(series, ranks=1)
    output = run / "per-rep-report.csv"
    args = argparse.Namespace(run_dir=str(run), series_dir=str(series), expected_ranks=1,
                              training_rc=0, training_job_id="f123", allocation_job_id="f122",
                              arm="original-notrace", run_id="r1", rep="1", block_id="b1",
                              scale="", output=str(output))
    write_report(args)
    row = next(csv.DictReader(output.open()))
    assert row["eligible"] == "true"
    assert row["reference_ok"] == "true" and row["shm_ok"] == "false"
    assert row["verdict_mode"] == "original_reference"


def test_failed_run_retains_evidence_but_blanks_accepted_timing(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_config_and_source(run)
    (run / "shm-verdict.json").write_text('{"ok": true}\n')
    series = run / "per-rep-step-series"
    write_series(series, ranks=1)
    output = run / "per-rep-report.csv"
    args = argparse.Namespace(run_dir=str(run), series_dir=str(series), expected_ranks=1,
                              training_rc=9, training_job_id="f123", allocation_job_id="f122",
                              arm="original", run_id="r1", rep="1", block_id="b1", scale="",
                              output=str(output))
    write_report(args)
    row = next(csv.DictReader(output.open()))
    assert row["eligible"] == "false"
    assert row["exclusion_reason"] == "RC_9"
    assert row["steady_ms_per_step"] == ""
    runs = POOL.load_tuolumne_csv(output)
    assert len(runs) == 1
    assert runs[0].exclusion == "NO_STEADY_TIME"


def test_missing_source_manifest_and_unknown_arm_are_not_eligible(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(base_config(source_manifest_sha256="c" * 64)))
    (run / "shm-verdict.json").write_text('{"ok": true}\n')
    series = run / "per-rep-step-series"
    write_series(series, ranks=1)
    output = run / "per-rep-report.csv"
    args = argparse.Namespace(run_dir=str(run), series_dir=str(series), expected_ranks=1,
                              training_rc=0, training_job_id="f123", allocation_job_id="f122",
                              arm="shm", run_id="r1", rep="1", block_id="b1", scale="",
                              output=str(output))
    write_report(args)
    row = next(csv.DictReader(output.open()))
    assert row["eligible"] == "false"
    assert row["steady_ms_per_step"] == ""
    assert row["exclusion_reason"] == "UNKNOWN_ARM"


def test_missing_source_manifest_blocks_an_otherwise_known_arm(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(base_config(source_manifest_sha256="c" * 64)))
    (run / "shm-verdict.json").write_text('{"ok": true}\n')
    series = run / "per-rep-step-series"
    write_series(series, ranks=1)
    output = run / "per-rep-report.csv"
    args = argparse.Namespace(run_dir=str(run), series_dir=str(series), expected_ranks=1,
                              training_rc=0, training_job_id="f123", allocation_job_id="f122",
                              arm="original", run_id="r1", rep="1", block_id="b1", scale="",
                              output=str(output))
    write_report(args)
    row = next(csv.DictReader(output.open()))
    assert row["eligible"] == "false"
    assert row["exclusion_reason"] == "SOURCE_MANIFEST_MISSING"
    assert row["steady_ms_per_step"] == ""


def test_red_shm_verdict_is_not_poolable(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_config_and_source(run)
    (run / "shm-verdict.json").write_text('{"ok": false}\n')
    series = run / "per-rep-step-series"
    write_series(series, ranks=1)
    output = run / "per-rep-report.csv"
    args = argparse.Namespace(run_dir=str(run), series_dir=str(series), expected_ranks=1,
                              training_rc=0, training_job_id="f123", allocation_job_id="f122",
                              arm="original", run_id="r1", rep="1", block_id="b1", scale="",
                              output=str(output))
    write_report(args)
    row = next(csv.DictReader(output.open()))
    assert row["exclusion_reason"] == "SHM_VERDICT_MISSING_OR_RED"
    assert row["steady_ms_per_step"] == ""
    assert POOL.load_tuolumne_csv(output)[0].exclusion == "NO_STEADY_TIME"


def test_fixed_cell_key_excludes_only_worker_and_prefetch_treatments():
    base = base_config(source_manifest_sha256="c" * 64)
    key, error = fixed_cell_key(base)
    assert error == ""
    assert fixed_cell_key(base | {"num_workers": 0, "prefetch_factor": 8})[0] == key
    assert fixed_cell_key(base | {"num_heads": 8})[0] != key
    assert fixed_cell_key(base | {"in_img_size": [64, 128]})[0] != key
    assert fixed_cell_key(base | {"epochs": 2})[0] != key
    assert fixed_cell_key(base | {"scheduler_warmup_epochs": 5})[0] != key


# --- long-cell rebind (E3.4/E3.5): total wall + tail window, additive, v1 schema untouched above ---

from per_rep_reporting import reduce_long_cell_wall_ms  # noqa: E402


def test_long_cell_total_wall_and_tail_window(tmp_path):
    series = tmp_path / "series"
    write_series(series, ranks=2, updates=1501)  # 1500 + 1 trailing quarantine batch
    result, reason = reduce_long_cell_wall_ms(series, 2, total_updates=1500, tail_window=500)
    assert reason == ""
    # critical (max-rank) per step is 11.0 (rank1's 10.0+1) for every one of the 1500 steps
    assert result["total_wall_ms"] == 1500 * 11.0
    assert result["n_total"] == 1500
    assert result["tail_window"] == 500
    assert result["tail_mean_ms_per_step"] == 11.0
    assert result["tail_median_ms_per_step"] == 11.0
    assert len(result["per_window_ms_per_step"]) == 1500


def test_long_cell_requires_full_coverage_to_total_updates_plus_trailing(tmp_path):
    series = tmp_path / "series"
    write_series(series, ranks=1, updates=1500)  # missing the trailing quarantine batch
    result, reason = reduce_long_cell_wall_ms(series, 1, total_updates=1500, tail_window=500)
    assert result is None
    assert reason == "NO_STEADY_TIME"


def test_long_cell_rejects_bad_window_bounds(tmp_path):
    series = tmp_path / "series"
    write_series(series, ranks=1, updates=1501)
    import pytest
    with pytest.raises(ValueError):
        reduce_long_cell_wall_ms(series, 1, total_updates=1500, tail_window=0)
    with pytest.raises(ValueError):
        reduce_long_cell_wall_ms(series, 1, total_updates=1500, tail_window=1501)


def test_report_row_carries_long_cell_endpoints_when_requested(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_config_and_source(run, max_training_step=1501)
    (run / "shm-verdict.json").write_text('{"ok": true}\n')
    series = run / "per-rep-step-series"
    write_series(series, ranks=1, updates=1501)
    # total_wall_ms will be 1500*10.0=15000ms; elapsed >= that (includes actuation/tail), so 16000ms
    # (a 1000ms gap) is a valid independent observation.
    write_elapsed_record(series, 0, 1000.0, 1016.0, 1500, 1016.2, 2000.0, 2016.2)
    output = run / "per-rep-report.csv"
    args = argparse.Namespace(run_dir=str(run), series_dir=str(series), expected_ranks=1,
                              training_rc=0, training_job_id="f123", allocation_job_id="f122",
                              arm="loop", run_id="r1", rep="1", block_id="b1", scale="",
                              output=str(output), long_cell_total_updates=1500, long_cell_tail_window=500)
    write_report(args)
    row = next(csv.DictReader(output.open()))
    assert row["eligible"] == "true"
    assert row["long_cell_elapsed_ms"] == format(16000.0, ".17g")
    assert row["total_wall_ms"] == format(1500 * 10.0, ".17g")
    assert row["long_cell_gap_ms"] == format(1000.0, ".17g")
    assert float(row["long_cell_full_loop_ms"]) == pytest.approx(16200.0)
    assert "rank-0.elapsed.json" in (series / "series-manifest.sha256").read_text()
    assert row["long_cell_tail_window"] == "500"
    assert row["tail_mean_ms_per_step"] == format(10.0, ".17g")
    assert Path(row["long_cell_window_series_path"]).is_file()
    payload = json.loads(Path(row["long_cell_window_series_path"]).read_text())
    assert payload["schema_version"] == "stormer-long-cell-window-series.v1"
    assert len(payload["per_window_ms_per_step"]) == 1500


def test_report_row_without_long_cell_args_is_unaffected(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_config_and_source(run)
    (run / "shm-verdict.json").write_text('{"ok": true}\n')
    series = run / "per-rep-step-series"
    write_series(series, ranks=1)
    output = run / "per-rep-report.csv"
    args = argparse.Namespace(run_dir=str(run), series_dir=str(series), expected_ranks=1,
                              training_rc=0, training_job_id="f123", allocation_job_id="f122",
                              arm="original", run_id="r1", rep="1", block_id="b1", scale="",
                              output=str(output))
    write_report(args)
    row = next(csv.DictReader(output.open()))
    assert row["total_wall_ms"] == "" and row["long_cell_total_updates"] == ""
    assert row["long_cell_elapsed_ms"] == "" and row["long_cell_gap_ms"] == ""
    assert row["long_cell_window_series_path"] == ""


def test_pooler_rebinds_long_cell_endpoint_and_selector(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_config_and_source(run, max_training_step=1501, liveio_mode="shm")
    (run / "shm-verdict.json").write_text('{"ok": true, "mode": "shm"}\n')
    series = run / "per-rep-step-series"
    write_series(series, ranks=1, updates=1501)
    write_elapsed_record(series, 0, 0.0, 16.0, 1500, 16.1, 1000.0, 1016.1)
    output = run / "per-rep-report.csv"
    args = argparse.Namespace(run_dir=str(run), series_dir=str(series), expected_ranks=1,
                              training_rc=0, training_job_id="f123", allocation_job_id="f122",
                              arm="loop", run_id="r1", rep="1", block_id="b1", scale="",
                              output=str(output), long_cell_total_updates=1500, long_cell_tail_window=500)
    write_report(args)
    row = next(csv.DictReader(output.open()))
    config = json.loads((run / "config.json").read_text())
    assert POOL._stormer_report_artifact_exclusion(row, config) == ""
    bad = dict(row, long_cell_elapsed_ms="17000")
    assert POOL._stormer_report_artifact_exclusion(bad, config) == "STORMER_REPORT_LONG_CELL_ENDPOINT_MISMATCH"
    bad = dict(row, long_cell_total_updates="")
    assert POOL._stormer_report_artifact_exclusion(bad, config) == "STORMER_REPORT_LONG_CELL_SELECTOR_MISSING"
    bad = dict(row, long_cell_total_updates="1499")
    assert POOL._stormer_report_artifact_exclusion(bad, config) == "STORMER_REPORT_LONG_CELL_CONFIG_MISMATCH"
    bad = dict(row, long_cell_tail_window="499")
    assert POOL._stormer_report_artifact_exclusion(bad, config) == "STORMER_REPORT_LONG_CELL_CONFIG_MISMATCH"
    receipt = series / "rank-0.elapsed.json"
    payload = json.loads(receipt.read_text())
    payload["end_s"] = 15.5
    receipt.write_text(json.dumps(payload))
    assert POOL._stormer_report_artifact_exclusion(row, config) == "STORMER_REPORT_SERIES_MANIFEST_MISMATCH"


# --- G5 revision: elapsed is the PRIMARY endpoint, structural-only eligibility, no magic tolerance ---

from per_rep_reporting import (write_elapsed_record, validate_long_cell_elapsed,  # noqa: E402
                               _clock_resolution_epsilon_ms)


def test_report_row_ineligible_when_elapsed_record_missing(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_config_and_source(run, max_training_step=1501)
    (run / "shm-verdict.json").write_text('{"ok": true}\n')
    series = run / "per-rep-step-series"
    write_series(series, ranks=1, updates=1501)
    # No write_elapsed_record call: the primary endpoint has nothing to report.
    output = run / "per-rep-report.csv"
    args = argparse.Namespace(run_dir=str(run), series_dir=str(series), expected_ranks=1,
                              training_rc=0, training_job_id="f123", allocation_job_id="f122",
                              arm="loop", run_id="r1", rep="1", block_id="b1", scale="",
                              output=str(output), long_cell_total_updates=1500, long_cell_tail_window=500)
    write_report(args)
    row = next(csv.DictReader(output.open()))
    assert row["eligible"] == "false"
    assert row["exclusion_reason"] == "LONG_CELL_ELAPSED_COVERAGE"
    assert row["long_cell_elapsed_ms"] == "" and row["total_wall_ms"] == ""


def test_report_row_ineligible_when_rank_intervals_exceed_own_elapsed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_config_and_source(run, max_training_step=1501)
    (run / "shm-verdict.json").write_text('{"ok": true}\n')
    series = run / "per-rep-step-series"
    write_series(series, ranks=1, updates=1501)
    write_elapsed_record(series, 0, 0.0, 10.0, 1500, 10.2, 1000.0, 1010.2)
    output = run / "per-rep-report.csv"
    args = argparse.Namespace(run_dir=str(run), series_dir=str(series), expected_ranks=1,
                              training_rc=0, training_job_id="f123", allocation_job_id="f122",
                              arm="loop", run_id="r1", rep="1", block_id="b1", scale="",
                              output=str(output), long_cell_total_updates=1500, long_cell_tail_window=500)
    write_report(args)
    row = next(csv.DictReader(output.open()))
    assert row["eligible"] == "false"
    assert row["exclusion_reason"] == "LONG_CELL_RANK_INTERVAL_EXCEEDS_ELAPSED"
    assert row["long_cell_elapsed_ms"] == "" and row["total_wall_ms"] == ""


def test_validate_long_cell_elapsed_structural_checks_no_tolerance_parameter():
    import inspect
    assert "tolerance" not in inspect.signature(validate_long_cell_elapsed).parameters


def test_validate_long_cell_elapsed_checks_rank_local_intervals_and_allows_negative_gap(tmp_path):
    series = tmp_path / "series"
    for rank in (0, 1):
        writer = StepSeriesWriter(series, rank)
        for update in range(1, 5):
            writer.write(update, 0, update, 20.0 if (update + rank) % 2 else 5.0,
                         "completed")
        writer.close()
        write_elapsed_record(series, rank, 0.0, 0.06, 4, 0.07, 1000.0, 1000.07)
    result, reason = validate_long_cell_elapsed(series, 2, 4, total_wall_ms=80.0)
    assert reason == ""
    assert result["elapsed_ms"] == 60.0
    assert result["gap_ms"] == pytest.approx(-20.0)
    record = series / "rank-0.elapsed.json"
    payload = json.loads(record.read_text())
    payload["end_s"] = 0.04
    record.write_text(json.dumps(payload))
    result, reason = validate_long_cell_elapsed(series, 2, 4, total_wall_ms=80.0)
    assert result is None
    assert reason == "LONG_CELL_RANK_INTERVAL_EXCEEDS_ELAPSED"


def test_load_elapsed_records_rejects_nonmonotonic_start_end(tmp_path):
    series = tmp_path / "series"
    path = write_elapsed_record(series, 0, 5.0, 6.0, 1, 6.1, 100.0, 101.1)
    payload = json.loads(path.read_text())
    payload["end_s"] = 5.0
    path.write_text(json.dumps(payload))
    _, reason = validate_long_cell_elapsed(series, 1, 1, total_wall_ms=0.0)
    assert reason == "LONG_CELL_ELAPSED_NONMONOTONIC"


def test_write_elapsed_record_rejects_nonfinite_timestamps(tmp_path):
    with pytest.raises(ValueError):
        write_elapsed_record(tmp_path, 0, float("nan"), 1.0, 1, 2.0, 10.0, 11.0)
    with pytest.raises(ValueError):
        write_elapsed_record(tmp_path, 0, 0.0, float("inf"), 1, 2.0, 10.0, 11.0)


def test_unequal_loader_marker_explicitly_excludes_even_complete_series(tmp_path):
    import logging
    from loop_exit_consensus import retain_unequal_exhaustion

    class SingleRankComm:
        def __init__(self, evidence_dir):
            self.evidence_dir = evidence_dir
            self.calls = []

        def gather(self, value, root):
            assert root == 0
            assert (self.evidence_dir / "rank-0-update-count.json").is_file()
            self.calls.append("gather")
            return [value]

        def Get_size(self):
            return 1

        def bcast(self, value, root):
            assert root == 0
            assert (self.evidence_dir / "unequal-loader-all-ranks.json").is_file()
            assert (self.evidence_dir / "unequal-loader-ineligible.json").is_file()
            self.calls.append("bcast")
            return value

    run = tmp_path / "run"
    run.mkdir()
    write_config_and_source(run)
    (run / "shm-verdict.json").write_text('{"ok": true}\n')
    series = run / "per-rep-step-series"
    write_series(series, ranks=1)
    comm = SingleRankComm(run / "loop-exit-evidence")
    assert retain_unequal_exhaustion(
        comm, rank=0, epoch=0, step=3, updates=2, reported_steps=2,
        evidence_dir=comm.evidence_dir, log=logging.getLogger(__name__))
    assert comm.calls == ["gather", "bcast"]
    output = run / "per-rep-report.csv"
    args = argparse.Namespace(run_dir=str(run), series_dir=str(series), expected_ranks=1,
                              training_rc=0, training_job_id="f123", allocation_job_id="f122",
                              arm="original", run_id="r1", rep="1", block_id="b1", scale="",
                              output=str(output))
    write_report(args)
    row = next(csv.DictReader(output.open()))
    assert row["eligible"] == "false"
    assert row["exclusion_reason"] == "UNEQUAL_LOADER_EXHAUSTION"
    assert row["steady_ms_per_step"] == ""
    assert row["loop_exit_marker_sha256"]
    assert row["loop_exit_all_rank_sha256"]
    assert (run / "loop-exit-evidence/rank-0-update-count.json").is_file()
