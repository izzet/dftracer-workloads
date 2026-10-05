"""Regression guard for the 2026-09-27 log_event(int_args=...) bare-int crash (STORMER-LOOP-DESIGN
sec 16/17): the bound C extension requires `int_args: Dict[str, Tuple[int, int]]`, a (tag_type,
value) pair. A bare int passed a strict-signature check but crashed every rank at step 1 with a
TypeError, correctly aborted by the fail-fast fix instead of hanging.

Two checks, since the real bound extension only exists on the pinned cluster shadow, not every dev
environment:
1. A STATIC scan of every `log_event(...)` call in `train.py` (grep the whole file, not just the two
   sites that broke): every `int_args=`/`float_args=`/`string_args=` dict literal's values must each
   be `{}` (empty, always safe) or a call to `_TAG_INT(...)` (or an inline tuple literal) -- never a
   bare name/int/float/str. This is the permanent, future-proof guard: it catches the NEXT emitter
   that makes the same mistake, not just the two already found.
2. A FAITHFUL STUB reproducing the real binding's runtime contract (`Dict[str, Tuple[int, int]]`,
   validated at call time): `_TAG_INT`'s own output is fed to it and must be accepted; a bare int is
   fed to it and must be REJECTED, proving the stub is strict enough to have caught this bug before
   it ever reached a live run.
"""
import ast
from pathlib import Path

TRAIN_PY = Path(__file__).resolve().parent / "train.py"


def _log_event_calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "log_event":
                yield node


def _dict_values_are_safe(dict_node):
    """A dict literal's values must each be `_TAG_INT(...)`, an inline 2-tuple, or the dict itself
    empty. Anything else (a bare Name/Constant) is the exact defect this test guards against."""
    if not dict_node.values:
        return True  # {} is always safe
    for value in dict_node.values:
        if isinstance(value, ast.Call):
            fname = value.func.id if isinstance(value.func, ast.Name) else None
            if fname == "_TAG_INT":
                continue
            return False
        if isinstance(value, ast.Tuple) and len(value.elts) == 2:
            continue
        return False
    return True


def test_every_log_event_arg_dict_in_train_py_uses_tag_int_or_is_empty():
    tree = ast.parse(TRAIN_PY.read_text())
    checked = 0
    for call in _log_event_calls(tree):
        for kw in call.keywords:
            if kw.arg in ("int_args", "float_args", "string_args") and isinstance(kw.value, ast.Dict):
                checked += 1
                assert _dict_values_are_safe(kw.value), (
                    f"{kw.arg} at train.py line {call.lineno} has a value that is neither "
                    f"_TAG_INT(...) nor an inline tuple nor an empty dict -- the exact bare-value "
                    f"defect that crashed every rank at step 1 on 2026-09-27"
                )
    # Sanity: this scan must actually find the calls it is meant to guard (a refactor that renames
    # the function or restructures the calls should not make this test silently pass on zero checks).
    assert checked >= 3, f"expected to check at least 3 log_event arg-dicts, found {checked}"


class _StrictBoundLoggerStub:
    """A faithful reproduction of the real binding's runtime contract for int_args: every value must
    be a 2-tuple of (int, int). This is what the pinned cluster shadow's C extension actually enforces
    (confirmed from the live TypeError message: "int_args: Dict[str, Tuple[int, int]]")."""

    def log_event(self, name, cat, start_time, duration, int_args=None, float_args=None,
                 string_args=None):
        for key, value in (int_args or {}).items():
            if not (isinstance(value, tuple) and len(value) == 2
                    and isinstance(value[0], int) and isinstance(value[1], int)):
                raise TypeError(
                    f"log_event(): int_args[{key!r}] must be a (int, int) tuple, got {value!r}")
        return None


def _tag_int(v):
    """The fallback branch of train.py's own `_TAG_INT` (no dftracer import needed for this test)."""
    return (0, int(v))


def test_tag_int_output_is_accepted_by_the_faithful_stub():
    stub = _StrictBoundLoggerStub()
    stub.log_event("x", "y", 0, 1, int_args={"step_count": _tag_int(9)})  # must not raise


def test_bare_int_is_rejected_by_the_faithful_stub():
    """Proves the stub is strict enough to have caught the actual 2026-09-27 defect."""
    stub = _StrictBoundLoggerStub()
    try:
        stub.log_event("x", "y", 0, 1, int_args={"step_count": 9})  # the exact original bug
    except TypeError:
        return
    raise AssertionError("faithful stub failed to reject a bare int -- it would not have caught "
                         "the 2026-09-27 defect")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
