"""Regression guard for the judge-batch-5-G4 root cause #2 (2026-09-28).

comm.exposed and comm.total (train.py `_emit_exposed_comm` / `_emit_total_comm`) are emitted via
direct `inst.log_event(...)`. Before this fix they carried only `step` in int_args, so the
analyzer-side window stamper filled their `epoch` from the horizon contract's ONE-BASED control
epoch (`_window.set_epoch(epoch + 1)`), while every other authoritative main-thread event
(`fetch.iter`, `compute.*`, `data.*`, `device.transfer` -- all via `dft_ai.update(epoch=epoch,
step=step)`) carries the ZERO-BASED epoch. Every window therefore contained BOTH epoch 0 and epoch
1 on the authoritative rows, so `attach_window_integrity_metrics` reported `epoch_nunique == 2` in
every window, and fact_engine.py's single-epoch gate (`window_control_vector_invalid`) suppressed
BOTH rules in 15/15 windows of the v4.5.9.1 1N run -- deterministically, since the emitters were
added. The gate itself is intentionally NOT loosened (metrics.py's own docstring: "The single-epoch
gate itself is not changed"): the fix is to make these two emitters agree with the rest of the
window on which epoch they belong to.

Static AST scan (same convention as test_log_event_int_args_shape.py): it must fail on the
unpatched file and pass on the fix.
"""
import ast
from pathlib import Path

TRAIN_PY = Path(__file__).resolve().parent / "train.py"


def _find_log_event_calls(tree, event_name):
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "log_event"):
            continue
        for kw in node.keywords:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant) and kw.value.value == event_name:
                hits.append(node)
    return hits


def _int_args_keys(call):
    for kw in call.keywords:
        if kw.arg == "int_args" and isinstance(kw.value, ast.Dict):
            return {k.value for k in kw.value.keys if isinstance(k, ast.Constant)}
    return set()


def test_comm_exposed_and_comm_total_log_events_carry_an_explicit_epoch():
    tree = ast.parse(TRAIN_PY.read_text())
    for event_name in ("comm.exposed", "comm.total"):
        calls = _find_log_event_calls(tree, event_name)
        assert len(calls) == 1, "expected exactly one log_event(name=%r) call, found %d" % (
            event_name, len(calls))
        keys = _int_args_keys(calls[0])
        assert "epoch" in keys, (
            "%s must tag `epoch` explicitly in int_args, in the same zero-based convention as the "
            "dft_ai-annotated events in its window; without it the window stamper's one-based "
            "control epoch makes epoch_nunique == 2 in every window and the fact engine's "
            "single-epoch gate suppresses every rule" % event_name
        )
        assert "step" in keys


def test_emit_helpers_take_epoch_and_call_sites_pass_it():
    text = TRAIN_PY.read_text()
    assert "def _emit_exposed_comm(step, ms, epoch):" in text
    assert "def _emit_total_comm(step, ms, epoch):" in text
    assert "_emit_exposed_comm(step, exposed_comm_ms, epoch)" in text
    assert "_emit_total_comm(step, comm_ms, epoch)" in text
