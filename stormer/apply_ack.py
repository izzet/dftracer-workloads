"""Post-apply ACK bookkeeping for `Trainer._sync_apply_window`.

An optimizer plan is acknowledged as ``applied`` only after (a) the apply steps ran and (b) every
rank confirmed that its OBSERVED state equals the MPI-consensus target.  The confirmation is a
per-knob 0/1 flag vector reduced with MIN by the caller, so one rank that failed to apply turns the
holder's ACK into ``failed``.  Pure functions; no MPI, torch, or optimizer imports.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

SENT = -1
KNOBS = ("num_workers", "prefetch_factor", "accumulate_grad_batches", "bucket_cap_mb",
         "decode_threads", "transfer_mb", "read_threads")


def local_flags(wants: dict, observed: dict, withheld: frozenset = frozenset()) -> list:
    """1 = this rank confirms the knob (or it was not requested / was withheld), 0 = mismatch."""
    out = []
    for knob in KNOBS:
        want = wants.get(knob, SENT)
        if want <= SENT or knob in withheld:
            out.append(1)
        else:
            out.append(1 if observed.get(knob) == want else 0)
    return out


def failed_knobs(reduced_flags) -> frozenset:
    return frozenset(k for k, f in zip(KNOBS, reduced_flags) if int(f) == 0)


def ack_pending(ctx, pending, reduced_flags, wants, observed, withheld=frozenset(),
                effective_window_id=None, applied_step=None, apply_cost_s=None) -> list:
    """Publish one ACK per drained plan.  Returns [(knob, status)] for logging/tests.

    ``pending`` items are (action, knob, old_value).
    """
    bad = failed_knobs(reduced_flags)
    if pending and effective_window_id is None:
        # Known gap, kept visible: this is NOT the durable post-apply receipt contract
        # (STORMER-POSTAPPLY-RECEIPT-CONTRACT.md); no effective window id, no raw per-rank receipts.
        log.info("APPLY_ACK receipt-contract gap: effective_window_id=None, no per-rank raw receipts")
    sent = []
    for action, knob, old in pending:
        if knob in withheld:
            status, reason = "rejected", "recommend_only: action computed but withheld"
        elif knob in bad:
            status = "failed"
            reason = ("apply_not_confirmed_on_all_ranks: want=%s holder_observed=%s"
                      % (wants.get(knob), observed.get(knob)))
        else:
            status, reason = "applied", ""
        ctx.ack_action(action, old, observed.get(knob) if status != "rejected" else None,
                       status=status, reason=reason, effective_window_id=effective_window_id,
                       applied_step=applied_step, apply_cost_s=apply_cost_s)
        sent.append((knob, status))
    return sent
