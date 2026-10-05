"""Scheduler-free unit tests for the StorMer post-apply evidence contract."""

import pytest

from post_apply_validation import (
    PostApplyValidationError,
    PlanIdentity,
    validate_plan_headers,
    validate_post_apply_receipts,
)


def _header(**overrides):
    value = {"plan_id": "plan-7", "knob": "stormer.num_workers", "target": 3,
             "boundary": "completed-window:11"}
    value.update(overrides)
    return value


def _receipt(rank, **overrides):
    value = {**_header(), "rank": rank, "observed_inner_workers": 3,
             "apply_performed": True, "eager_prime_success": True,
             "exhausted": False, "noop": False, "errors": [],
             "apply_boundary_window_id": 11, "effective_window_id": 12}
    value.update(overrides)
    return value


def _plan():
    return validate_plan_headers([None, _header(), {}, _header()])


def test_holder_headers_allow_nonholders_but_bind_all_identity_fields():
    assert _plan() == PlanIdentity("plan-7", "stormer.num_workers", 3, "completed-window:11")
    for changed in (
        {"plan_id": "other-plan"}, {"knob": "stormer.prefetch_factor"},
        {"target": 4}, {"boundary": "completed-window:12"},
    ):
        with pytest.raises(PostApplyValidationError, match="conflicting"):
            validate_plan_headers([_header(), _header(**changed)])


def test_headers_require_at_least_one_context_holder_plan():
    with pytest.raises(PostApplyValidationError, match="no nonempty"):
        validate_plan_headers([None, {}])


def test_complete_exact_success_receipts_are_accepted():
    assert validate_post_apply_receipts([0, 1], _plan(), [_receipt(0), _receipt(1)]) == 12


@pytest.mark.parametrize("field,value,match", [
    ("plan_id", "other-plan", "selected plan/boundary"),
    ("knob", "stormer.prefetch_factor", "selected plan/boundary"),
    ("target", 4, "selected plan/boundary"),
    ("boundary", "completed-window:12", "selected plan/boundary"),
    ("observed_inner_workers", 2, "inner worker count"),
    ("apply_performed", False, "materialized apply"),
    ("eager_prime_success", False, "eager prime"),
    ("exhausted", True, "exhausted"),
    ("noop", True, "noop"),
    ("errors", ["worker failed"], "errors"),
    ("effective_window_id", -1, "effective_window_id"),
    ("effective_window_id", True, "effective_window_id"),
])
def test_each_required_post_apply_proof_fails_closed(field, value, match):
    with pytest.raises(PostApplyValidationError, match=match):
        validate_post_apply_receipts([0, 1], _plan(), [_receipt(0, **{field: value}), _receipt(1)])


def test_receipts_require_exactly_one_from_every_expected_rank():
    with pytest.raises(PostApplyValidationError, match="missing receipts"):
        validate_post_apply_receipts([0, 1], _plan(), [_receipt(0)])
    with pytest.raises(PostApplyValidationError, match="duplicate"):
        validate_post_apply_receipts([0, 1], _plan(), [_receipt(0), _receipt(0)])
    with pytest.raises(PostApplyValidationError, match="unexpected rank"):
        validate_post_apply_receipts([0, 1], _plan(), [_receipt(0), _receipt(2)])


@pytest.mark.parametrize("field,value", [
    ("eager_prime_success", None), ("exhausted", None), ("noop", None), ("errors", None),
])
def test_omitted_or_implicit_receipt_proofs_are_not_success(field, value):
    receipt = _receipt(0)
    if value is None:
        receipt.pop(field)
    else:
        receipt[field] = value
    with pytest.raises(PostApplyValidationError):
        validate_post_apply_receipts([0], _plan(), [receipt])


def test_effective_window_id_must_be_present_and_agree_on_every_rank():
    with pytest.raises(PostApplyValidationError, match="effective_window_id"):
        validate_post_apply_receipts([0], _plan(), [_receipt(0, effective_window_id=None)])
    with pytest.raises(PostApplyValidationError, match="disagree"):
        validate_post_apply_receipts(
            [0, 1], _plan(), [_receipt(0, effective_window_id=12),
                              _receipt(1, effective_window_id=13)],
        )


def test_effective_window_is_next_after_the_completed_apply_boundary():
    with pytest.raises(PostApplyValidationError, match="apply_boundary_window_id"):
        validate_post_apply_receipts([0], _plan(), [_receipt(0, apply_boundary_window_id=True)])
    with pytest.raises(PostApplyValidationError, match="selected plan/boundary"):
        validate_post_apply_receipts([0], _plan(), [_receipt(0, boundary="completed-window:10")])
    with pytest.raises(PostApplyValidationError, match="next traced window"):
        validate_post_apply_receipts([0], _plan(), [_receipt(0, effective_window_id=11)])
