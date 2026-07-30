"""Real deterministic contracts for the frozen three-way Storey review."""

from __future__ import annotations

import copy

import pytest

from active_inference_power.storey_method_review import (
    CONSERVATIVE_SAME_VECTOR_STOREY,
    DEFAULT_STOREY_METHOD_REVIEW_PLAN,
    SAME_VECTOR_STOREY_PLUGIN,
    SPLIT_CONFIRMATION_STOREY,
    STOREY_METHOD_REVIEW_STATUS,
    StoreyMethodReviewPlan,
    evaluate_storey_method_review,
    validate_storey_method_review,
)


def _small_plan() -> StoreyMethodReviewPlan:
    """Keep real numerical review tests compact while retaining all methods."""
    return StoreyMethodReviewPlan(m=40, n_reps=12, seed=91)


def test_default_review_plan_is_exactly_the_frozen_requested_design() -> None:
    assert DEFAULT_STOREY_METHOD_REVIEW_PLAN.to_mapping() == {
        "pi0": 0.5,
        "m": 200,
        "effect": 3.0,
        "alpha": 0.05,
        "lambda": 0.5,
        "confidence": 0.99,
        "n_reps": 7_500,
        "seed": 0,
        "dependence": "independent",
        "sided": 1,
    }


def test_same_vector_storey_identifier_uses_neutral_current_terminology() -> None:
    """New review artifacts must not revive predecessor-facing terminology."""
    assert SAME_VECTOR_STOREY_PLUGIN == "same_vector_storey_plugin"


def test_three_way_review_preserves_stream_roles_metrics_and_hashes() -> None:
    plan = _small_plan()
    first = evaluate_storey_method_review(plan)
    second = evaluate_storey_method_review(plan)
    assert second == first
    assert validate_storey_method_review(first) == first
    assert first["status"] == STOREY_METHOD_REVIEW_STATUS
    assert first["plan_sha256"] == plan.configuration_hash
    assert first["schedule_sha256"] == first["stream_schedule"]["schedule_sha256"]
    assert len(first["result_sha256"]) == 64
    assert first["stream_schedule"]["same_vector"]["child_index_first"] == 0
    assert first["stream_schedule"]["same_vector"]["child_index_stride"] == 2
    assert first["stream_schedule"]["independent_confirmation"]["child_index_first"] == 1
    assert first["stream_schedule"]["selection_and_confirmation_streams_disjoint"] is True

    methods = first["methods"]
    assert set(methods) == {
        SAME_VECTOR_STOREY_PLUGIN,
        CONSERVATIVE_SAME_VECTOR_STOREY,
        SPLIT_CONFIRMATION_STOREY,
    }
    for method_id in (SAME_VECTOR_STOREY_PLUGIN, CONSERVATIVE_SAME_VECTOR_STOREY):
        method = methods[method_id]
        assert method["selection_stream"] == "same_vector"
        assert method["decision_stream"] == "same_vector"
        assert method["common_random_numbers_group"] == "same_vector_storey_comparison"
        for metric in ("fdr", "fwer", "power"):
            summary = method["metrics"][metric]
            assert summary["n"] == plan.n_reps
            assert summary["confidence"] == plan.confidence
            assert 0.0 <= summary["confidence_lower_band"] <= summary["estimate"]
            assert summary["estimate"] <= summary["confidence_upper_band"] <= 1.0
            assert summary["standard_error"] >= 0.0
    assert (
        methods[SAME_VECTOR_STOREY_PLUGIN]["method_label"]
        == "Same-vector Storey plug-in (diagnostic)"
    )
    split = methods[SPLIT_CONFIRMATION_STOREY]
    assert split["selection_stream"] == "same_vector"
    assert split["decision_stream"] == "independent_confirmation"
    assert first["stream_evidence"]["same_vector_methods"] == [
        SAME_VECTOR_STOREY_PLUGIN,
        CONSERVATIVE_SAME_VECTOR_STOREY,
    ]
    assert first["stream_evidence"]["same_vector_and_confirmation_draws_differ"] is True
    assert (
        first["stream_evidence"]["same_vector_draw_sha256"]
        != first["stream_evidence"]["confirmation_draw_sha256"]
    )
    assert "does not change certificate policy" in first["nonclaims"][2]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pi0": -0.01},
        {"pi0": "bad"},
        {"m": 0},
        {"m": True},
        {"m": "bad"},
        {"effect": float("nan")},
        {"alpha": 0.0},
        {"lambda_": 1.0},
        {"confidence": 1.0},
        {"n_reps": 0},
        {"seed": True},
        {"seed": -1},
        {"seed": "bad"},
    ],
)
def test_review_plan_rejects_invalid_frozen_contract_values(
    kwargs: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        StoreyMethodReviewPlan(**kwargs)  # type: ignore[arg-type]


def test_review_entry_point_and_serialized_witnesses_fail_closed() -> None:
    with pytest.raises(ValueError, match="plan"):
        evaluate_storey_method_review(object())  # type: ignore[arg-type]

    mutated = _small_plan()
    object.__setattr__(mutated, "n_reps", 0)
    with pytest.raises(ValueError, match="n_reps"):
        evaluate_storey_method_review(mutated)

    review = evaluate_storey_method_review(_small_plan())
    broken_metric = copy.deepcopy(review)
    broken_metric["methods"][SAME_VECTOR_STOREY_PLUGIN]["metrics"]["fdr"][
        "confidence_upper_band"
    ] = -0.01
    with pytest.raises(ValueError, match="upper"):
        validate_storey_method_review(broken_metric)

    broken_schedule = copy.deepcopy(review)
    broken_schedule["stream_schedule"]["same_vector"]["child_index_stride"] = 3
    with pytest.raises(ValueError, match="stream_schedule"):
        validate_storey_method_review(broken_schedule)

    broken_stream_hash = copy.deepcopy(review)
    broken_stream_hash["stream_evidence"]["stream_evidence_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="stream_evidence hash"):
        validate_storey_method_review(broken_stream_hash)

    broken_result_hash = copy.deepcopy(review)
    broken_result_hash["result_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="result_sha256"):
        validate_storey_method_review(broken_result_hash)


def test_review_validator_rejects_each_material_contract_boundary() -> None:
    """Tamper real output one field at a time to exercise fail-closed edges."""
    review = evaluate_storey_method_review(_small_plan())

    with pytest.raises(ValueError, match="mapping"):
        validate_storey_method_review([])  # type: ignore[arg-type]

    broken = copy.deepcopy(review)
    broken["extra"] = True
    with pytest.raises(ValueError, match="unexpected schema"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="schema_version"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["status"] = "passed"
    with pytest.raises(ValueError, match="review-only"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["review_plan"].pop("pi0")
    with pytest.raises(ValueError, match="review_plan has an unexpected schema"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["review_plan"]["dependence"] = "factor"
    with pytest.raises(ValueError, match="frozen independent"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["plan_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="review_plan hash"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["schedule_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="schedule_sha256"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["stream_evidence"]["same_vector_draw_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["stream_evidence"]["same_vector_draw_sha256"] = "short"
    with pytest.raises(ValueError, match="SHA-256"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["stream_evidence"].pop("split_decision_stream")
    with pytest.raises(ValueError, match="stream_evidence has an unexpected schema"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    digest = broken["stream_evidence"]["same_vector_draw_sha256"]
    broken["stream_evidence"]["confirmation_draw_sha256"] = digest
    with pytest.raises(ValueError, match="must differ"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["stream_evidence"]["same_vector_methods"] = []
    with pytest.raises(ValueError, match="same-vector comparison methods"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["stream_evidence"]["split_selection_stream"] = "confirmation"
    with pytest.raises(ValueError, match="split selection stream"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["stream_evidence"]["split_decision_stream"] = "same_vector"
    with pytest.raises(ValueError, match="split decision stream"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["stream_evidence"]["same_vector_and_confirmation_draws_differ"] = False
    with pytest.raises(ValueError, match="distinct confirmation"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["methods"].pop(SPLIT_CONFIRMATION_STOREY)
    with pytest.raises(ValueError, match="frozen three-way"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["assumptions"] = []
    with pytest.raises(ValueError, match="assumptions"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["nonclaims"] = []
    with pytest.raises(ValueError, match="nonclaims"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["claim_boundary"] = ""
    with pytest.raises(ValueError, match="claim_boundary"):
        validate_storey_method_review(broken)


def test_method_and_metric_validator_edges_are_bound_to_real_samples() -> None:
    """Cover method-level witnesses without substituting simulated data."""
    review = evaluate_storey_method_review(_small_plan())
    method_id = SAME_VECTOR_STOREY_PLUGIN

    broken = copy.deepcopy(review)
    broken["methods"][method_id]["extra"] = True
    with pytest.raises(ValueError, match="unexpected schema"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["methods"][method_id]["method_id"] = "wrong"
    with pytest.raises(ValueError, match="method_id"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["methods"][method_id]["method_label"] = "wrong"
    with pytest.raises(ValueError, match="method_label"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["methods"][method_id]["metrics"].pop("power")
    with pytest.raises(ValueError, match="FDR, FWER, and power"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["methods"][method_id]["metrics"]["fdr"]["extra"] = True
    with pytest.raises(ValueError, match="metric-summary schema"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["methods"][method_id]["metrics"]["fdr"]["n"] = 1
    with pytest.raises(ValueError, match="n does not match"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["methods"][method_id]["metrics"]["fdr"]["confidence_lower_band"] = 1.0
    with pytest.raises(ValueError, match="does not contain"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["methods"][method_id]["metrics"]["fdr"]["standard_error"] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["methods"][method_id]["metrics"]["fdr"]["confidence"] = 0.95
    with pytest.raises(ValueError, match="confidence does not match"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["methods"][method_id]["metrics"]["fdr"]["interval_method"] = "wrong"
    with pytest.raises(ValueError, match="interval_method"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["methods"][method_id]["mean_rejections"] = 999.0
    with pytest.raises(ValueError, match="rejection accounting"):
        validate_storey_method_review(broken)

    broken = copy.deepcopy(review)
    broken["methods"][method_id]["result_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="result hash"):
        validate_storey_method_review(broken)


def test_one_replication_review_has_zero_count_standard_error() -> None:
    """Exercise the finite-count uncertainty edge with a real one-rep draw."""
    review = evaluate_storey_method_review(StoreyMethodReviewPlan(m=12, n_reps=1, seed=5))
    for method in review["methods"].values():
        assert method["mean_rejections_standard_error"] == 0.0
    validate_storey_method_review(review)
