"""Pure validation for StorMer's proposed post-apply action evidence.

This module deliberately has no MPI, loader, optimizer, filesystem, or
scheduler dependency.  A future transport layer must collect the raw headers
and receipts, then call these functions before it can regard an action as
applied.  It is intentionally fail-closed: an omitted proof field is not an
implicit success.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


class PostApplyValidationError(ValueError):
    """The collected plan or post-apply evidence does not prove an apply."""


@dataclass(frozen=True)
class PlanIdentity:
    """The four fields that must identify one reviewed action."""

    plan_id: str
    knob: str
    target: int
    boundary: str


_PLAN_FIELDS = ("plan_id", "knob", "target", "boundary")


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PostApplyValidationError("%s must be a mapping" % label)
    return value


def _identity(value: object, label: str) -> PlanIdentity:
    record = _require_mapping(value, label)
    missing = [field for field in _PLAN_FIELDS if field not in record]
    if missing:
        raise PostApplyValidationError("%s missing %s" % (label, ", ".join(missing)))
    plan_id, knob, target, boundary = (record[field] for field in _PLAN_FIELDS)
    if not isinstance(plan_id, str) or not plan_id:
        raise PostApplyValidationError("%s plan_id must be nonempty" % label)
    if not isinstance(knob, str) or not knob:
        raise PostApplyValidationError("%s knob must be nonempty" % label)
    if isinstance(target, bool) or not isinstance(target, int) or target < 0:
        raise PostApplyValidationError("%s target must be a nonnegative integer" % label)
    if not isinstance(boundary, str) or not boundary:
        raise PostApplyValidationError("%s boundary must be nonempty" % label)
    return PlanIdentity(plan_id, knob, target, boundary)


def validate_plan_headers(headers: Iterable[object]) -> PlanIdentity:
    """Return the one plan named by nonempty context-holder headers.

    Non-holders contribute ``None`` (or an empty mapping) and are deliberately
    ignored.  Every nonempty holder header must agree on all identity fields;
    equal targets do not make different plan IDs or boundaries acceptable.
    """
    agreed = None
    for index, header in enumerate(headers):
        if header is None or header == {}:
            continue
        candidate = _identity(header, "header[%d]" % index)
        if agreed is None:
            agreed = candidate
        elif candidate != agreed:
            raise PostApplyValidationError(
                "conflicting context-holder plan headers: %r != %r" % (candidate, agreed)
            )
    if agreed is None:
        raise PostApplyValidationError("no nonempty context-holder plan header")
    return agreed


def validate_post_apply_receipts(
    expected_ranks: Iterable[int], plan: PlanIdentity, receipts: Iterable[object]
) -> int:
    """Require a successful, unique, exact receipt from every expected rank.

    A receipt proves the materialised *inner* loader worker count and a
    successful eager prime.  ``apply_performed`` rejects a request/no-op;
    explicit ``exhausted`` and ``noop`` flags reject terminal or inert paths.
    The returned effective-window ID is the one the holder must put on its ACK.
    """
    if not isinstance(plan, PlanIdentity):
        raise PostApplyValidationError("plan must be a PlanIdentity")
    ranks = tuple(expected_ranks)
    if not ranks:
        raise PostApplyValidationError("expected_ranks must not be empty")
    if any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in ranks):
        raise PostApplyValidationError("expected_ranks must contain nonnegative integer ranks")
    if len(set(ranks)) != len(ranks):
        raise PostApplyValidationError("expected_ranks contains duplicates")
    expected = set(ranks)
    seen = set()
    effective_window_id = None

    for index, raw_receipt in enumerate(receipts):
        label = "receipt[%d]" % index
        receipt = _require_mapping(raw_receipt, label)
        if "rank" not in receipt:
            raise PostApplyValidationError("%s missing rank" % label)
        rank = receipt["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank not in expected:
            raise PostApplyValidationError("%s has unexpected rank %r" % (label, rank))
        if rank in seen:
            raise PostApplyValidationError("duplicate receipt for rank %d" % rank)
        seen.add(rank)
        if _identity(receipt, label) != plan:
            raise PostApplyValidationError("%s does not bind the selected plan/boundary" % label)
        if receipt.get("observed_inner_workers") != plan.target:
            raise PostApplyValidationError("%s inner worker count did not materialize target" % label)
        if receipt.get("apply_performed") is not True:
            raise PostApplyValidationError("%s did not prove a materialized apply" % label)
        if receipt.get("eager_prime_success") is not True:
            raise PostApplyValidationError("%s did not prove eager prime success" % label)
        if receipt.get("exhausted") is not False:
            raise PostApplyValidationError("%s is exhausted or lacks explicit exhaustion proof" % label)
        if receipt.get("noop") is not False:
            raise PostApplyValidationError("%s is noop or lacks explicit noop proof" % label)
        if receipt.get("errors") != []:
            raise PostApplyValidationError("%s reports errors or lacks an empty errors list" % label)
        apply_boundary = receipt.get("apply_boundary_window_id")
        if (isinstance(apply_boundary, bool) or not isinstance(apply_boundary, int)
                or apply_boundary < 0):
            raise PostApplyValidationError(
                "%s apply_boundary_window_id must be a nonnegative integer" % label)
        if receipt["boundary"] != "completed-window:%d" % apply_boundary:
            raise PostApplyValidationError("%s does not bind its completed apply boundary" % label)
        receipt_window = receipt.get("effective_window_id")
        if (isinstance(receipt_window, bool) or not isinstance(receipt_window, int)
                or receipt_window < 0):
            raise PostApplyValidationError(
                "%s effective_window_id must be a nonnegative integer" % label)
        if effective_window_id is None:
            effective_window_id = receipt_window
        elif receipt_window != effective_window_id:
            raise PostApplyValidationError("receipts disagree on effective_window_id")
        if receipt_window != apply_boundary + 1:
            raise PostApplyValidationError(
                "%s effective_window_id is not the next traced window" % label)

    missing = expected - seen
    if missing:
        raise PostApplyValidationError("missing receipts for ranks %s" % sorted(missing))
    return effective_window_id
