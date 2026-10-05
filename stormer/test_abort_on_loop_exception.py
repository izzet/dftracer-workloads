#!/usr/bin/env python3
"""Unit tests for stormer/abort_on_loop_exception.py and the control-flow
shape it is called from in Trainer.train() -- no mpi4py, no GPU, no Trainer,
no torch. Matches the style of test_loop_exit_consensus.py: cover what can
be checked in-process; real cross-rank behavior needs a live allocation.

Two things are under test:
  1. abort_on_loop_exception() itself: logs with rank/epoch/step context,
     flushes, and calls comm.Abort(<distinct code>).
  2. The exact try/except/finally/then-validate shape now used in
     Trainer.train() (STORMER-STEP202-FORENSICS.md): reproduced here as a
     minimal standalone function so the CONTROL-FLOW PROPERTY is verified
     without needing the real 3000-line Trainer -- an exception during the
     "loop" must reach abort_on_loop_exception() and must NOT reach
     "validate", and the default success path must still call "validate"
     exactly as before.
"""

import logging
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..",
        "optimization", "dftracer_workloads", "stormer",
    ),
)

from abort_on_loop_exception import abort_on_loop_exception  # noqa: E402


class _FakeComm:
    def __init__(self):
        self.abort_calls = []

    def Abort(self, code):
        self.abort_calls.append(code)
        # Real MPI.Abort() does not return; the fixture leaves it returning
        # so the caller's own `raise` is what's under test, matching
        # mpi4py_two_rank_fixture.py's "Abort makes the bounded failure
        # visible" idiom without requiring a live communicator here.


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
        self.flushed = False

    def emit(self, record):
        self.records.append(record)

    def flush(self):
        self.flushed = True


class AbortOnLoopExceptionTest(unittest.TestCase):
    def test_logs_context_flushes_and_aborts_with_distinct_code(self):
        log = logging.getLogger("test_abort_on_loop_exception")
        log.setLevel(logging.ERROR)
        handler = _ListHandler()
        log.addHandler(handler)
        comm = _FakeComm()
        exc = OSError("could not open StorMer sample /fake/2017_0847.h5: boom")

        abort_on_loop_exception(exc, rank=14, epoch=0, step=203, comm=comm, log=log)

        self.assertEqual(comm.abort_calls, [97])
        self.assertTrue(handler.flushed)
        self.assertEqual(len(handler.records), 1)
        rec = handler.records[0]
        self.assertIn("STORMER_LOOP_EXCEPTION", rec.getMessage())
        self.assertIn("rank=14", rec.getMessage())
        self.assertIn("epoch=0", rec.getMessage())
        self.assertIn("step=203", rec.getMessage())
        # exc_info was attached, so the original exception (and its message,
        # which for the real OSError/FileNotFoundError classes already names
        # the offending sample path) is recoverable from the record.
        self.assertIsNotNone(rec.exc_info)
        self.assertIs(rec.exc_info[1], exc)

    def test_abort_code_distinct_from_existing_collective_timeout_code(self):
        # Trainer._sync_input_exposure_window already uses Abort(124) for a
        # different trigger (collective timeout); this fix's code must not
        # collide with it so the two are distinguishable in job logs.
        log = logging.getLogger("test_abort_on_loop_exception_2")
        comm = _FakeComm()
        abort_on_loop_exception(RuntimeError("x"), rank=0, epoch=0, step=1, comm=comm, log=log)
        self.assertNotEqual(comm.abort_calls[0], 124)


def _epoch_body_like_trainer(loop_fn, validate_fn, *, comm):
    """Reproduces the exact shape now used in Trainer.train() (see the
    140-finally-barrier-abort.diff hunk touching the per-epoch try/except/
    finally): a `finally` that only does non-collective cleanup, and
    `validate_fn` (standing in for validate_epoch()/the trajectory-retention
    branch) called AFTER the try/except/finally, so it is unreachable if
    `loop_fn` raised.
    """
    cleanup_calls = []
    try:
        loop_fn()
    except Exception as exc:
        abort_on_loop_exception(exc, rank=14, epoch=0, step=203, comm=comm)
        raise
    finally:
        cleanup_calls.append("cleanup")  # stands in for STEP_SERIES/window.flush
    validate_fn()
    return cleanup_calls


class EpochBodyShapeTest(unittest.TestCase):
    def test_exception_in_loop_aborts_and_never_reaches_validate(self):
        comm = _FakeComm()
        validate_calls = []

        def raising_loop():
            raise OSError("boom")

        with self.assertRaises(OSError):
            _epoch_body_like_trainer(raising_loop, lambda: validate_calls.append(1), comm=comm)

        self.assertEqual(comm.abort_calls, [97])
        self.assertEqual(validate_calls, [])  # the historical bug: this used to still run

    def test_default_success_path_still_calls_validate_and_no_abort(self):
        comm = _FakeComm()
        validate_calls = []

        def clean_loop():
            return None

        cleanup_calls = _epoch_body_like_trainer(
            clean_loop, lambda: validate_calls.append(1), comm=comm
        )

        self.assertEqual(comm.abort_calls, [])
        self.assertEqual(validate_calls, [1])
        self.assertEqual(cleanup_calls, ["cleanup"])  # finally still ran, exactly once


if __name__ == "__main__":
    unittest.main()
