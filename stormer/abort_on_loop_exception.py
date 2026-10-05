"""Abort helper for the STORMER-STEP202-FORENSICS.md finally-calls-
validate_epoch fix.

Kept in its own module -- no torch, mpi4py, or dft_ai import -- so its
control-flow contract (log with context, flush, Abort the given
communicator) is unit-testable without any of Trainer's heavy dependencies,
matching the pattern already used by loop_exit_consensus.py for the same
reason.
"""

import logging
import sys


def abort_on_loop_exception(exc, *, rank, epoch, step, comm, log=None, abort_code=97):
    """Log `exc` with rank/epoch/step context, flush, then Abort `comm`.

    Called from the per-epoch training loop's `except` clause. A rank-local
    exception anywhere in the per-step loop body means this rank has already
    stopped joining whatever collective its peers are still waiting inside
    (backward()/all-reduce, or a later validate_epoch() barrier). Letting it
    fall through to a barrier instead -- which is what happened before this
    fix, when validate_epoch() lived inside the loop's `finally` block and
    therefore ran unconditionally -- lets this rank silently join a
    DIFFERENT collective while its peers wait forever on the one it
    abandoned. Abort the whole communicator instead, so peers fail loudly
    rather than hang.

    Never returns in production (Abort terminates the job); callers must
    still `raise` afterward so that a test double which lets Abort() return
    normally does not fall through to code that assumes the loop succeeded.
    The exception's own message/traceback (via `exc_info=exc`) already names
    the offending file/sample where applicable, e.g. ERA5Dataset's
    OSError/FileNotFoundError -- no separate sample-path plumbing is needed
    here.
    """
    _log = log or logging.getLogger(__name__)
    _log.error(
        "STORMER_LOOP_EXCEPTION rank=%d epoch=%s step=%s: per-step loop "
        "raised -- aborting communicator instead of risking a silent "
        "collective mismatch (STORMER-STEP202-FORENSICS.md)",
        rank, epoch, step, exc_info=exc,
    )
    for handler in list(getattr(_log, "handlers", ())) + list(logging.getLogger().handlers):
        try:
            handler.flush()
        except Exception:
            pass
    sys.stdout.flush()
    sys.stderr.flush()
    # Distinct from the existing collective-timeout Abort(124) in
    # Trainer._sync_input_exposure_window: same failure CLASS (a rank cannot
    # safely continue while peers may still be inside a collective) but a
    # different trigger, so the two are distinguishable in job logs.
    comm.Abort(abort_code)
