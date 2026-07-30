"""Real all-null split-stream sensitivity diagnostics."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from active_inference_power.scenario import (
    make_reference_action_scenario,
    make_static_null_binary_e_process_scenario,
)
from active_inference_power.selection import run_split_stream_confirmation
from active_inference_power.selection_sensitivity import (
    SplitStreamSensitivityPlan,
    _history_key,
    _pvalue_diagnostic,
    _seed_pairs,
    evaluate_split_stream_sensitivity,
)


def test_static_null_scenario_preserves_the_formal_channel_contract() -> None:
    scenario = make_static_null_binary_e_process_scenario(horizon=3)
    assert scenario.truth_consistent
    assert not scenario.process.target_truth.any()
    result = run_split_stream_confirmation(
        scenario, selection_seed=11, confirmation_seed=12
    )
    assert result.validity == "formal_conditional_split_confirmation"
    assert _history_key(result, 1.0).endswith("selected_evidence=at_or_below_alpha")


def test_sensitivity_plan_is_hashable_and_rejects_ambiguous_tuning() -> None:
    plan = SplitStreamSensitivityPlan(
        selection_steps=(1, 2), confirmation_steps=3, alpha=0.1
    )
    assert plan.configuration_hash == SplitStreamSensitivityPlan(
        selection_steps=(1, 2), confirmation_steps=3, alpha=0.1
    ).configuration_hash
    assert plan.policy_for(2).confirmation_steps == 3
    assert plan.to_mapping()["fixed_tuning"]
    with pytest.raises(ValueError, match="unique"):
        SplitStreamSensitivityPlan(selection_steps=(2, 2))
    with pytest.raises(ValueError, match="positive"):
        SplitStreamSensitivityPlan(selection_steps=(0,))
    with pytest.raises(ValueError, match="unsupported"):
        SplitStreamSensitivityPlan(strategy="future" )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="minimum_p"):
        SplitStreamSensitivityPlan(evidence_observable=False)
    with pytest.raises(ValueError, match="boolean"):
        SplitStreamSensitivityPlan(evidence_observable=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="confirmation_steps"):
        SplitStreamSensitivityPlan(confirmation_steps=0)
    with pytest.raises(ValueError, match="alpha"):
        SplitStreamSensitivityPlan(alpha=0.0)
    with pytest.raises(ValueError, match="confidence"):
        SplitStreamSensitivityPlan(confidence=1.0)


def test_stratified_pvalue_diagnostic_and_seed_schedule_are_explicit() -> None:
    empty = _pvalue_diagnostic([], 0.95, 0.1)
    assert empty["n"] == 0
    diagnostic = _pvalue_diagnostic([0.07, 0.2], 0.95, 0.1)
    assert diagnostic["n"] == 2
    assert diagnostic["rejection_rate"] == pytest.approx(0.5)
    assert _seed_pairs(7, 2) == _seed_pairs(7, 2)
    with pytest.raises(ValueError, match="seed"):
        _seed_pairs(True, 2)
    with pytest.raises(ValueError, match="seed"):
        _seed_pairs(-1, 2)


def test_split_stream_sensitivity_reports_target_and_history_strata_with_crn() -> None:
    scenario = make_static_null_binary_e_process_scenario(horizon=3)
    plan = SplitStreamSensitivityPlan(selection_steps=(1, 2), confirmation_steps=3)
    serial = evaluate_split_stream_sensitivity(scenario, plan, 3, seed=41, workers=1)
    parallel = evaluate_split_stream_sensitivity(scenario, plan, 3, seed=41, workers=2)
    assert serial["scenario_hash"] == scenario.configuration_hash
    common_random_numbers = cast(dict[str, bool], serial["common_random_numbers"])
    assert common_random_numbers["enabled"] is True
    assert serial["seed_schedule_hash"] == parallel["seed_schedule_hash"]
    assert serial["cells"] == parallel["cells"]
    cells = cast(list[dict[str, Any]], serial["cells"])
    assert len(cells) == 2
    for cell in cells:
        assert cell["n_reps"] == 3
        assert cell["independent_streams"] is True
        assert len(cell["stream_pairs"]) == 3
        assert len(cell["conditional_null_checks"]) == scenario.setting.family_size
        assert cell["validity"] == "formal_conditional_split_confirmation"
        assert cell["pooled"]["n"] == 3
        assert sum(cell["selected_target_counts"]) == 3
        for target in cast(list[dict[str, Any]], cell["conditional_null_checks"]):
            assert "by_visible_selection_history" in target


def test_split_stream_sensitivity_fails_closed_for_wrong_scope_and_inputs() -> None:
    plan = SplitStreamSensitivityPlan(selection_steps=(1,), confirmation_steps=2)
    with pytest.raises(ValueError, match="all-null"):
        evaluate_split_stream_sensitivity(
            make_reference_action_scenario(horizon=2), plan, 1
        )
    scenario = make_static_null_binary_e_process_scenario(horizon=2)
    with pytest.raises(ValueError, match="n_reps"):
        evaluate_split_stream_sensitivity(scenario, plan, 0)
    with pytest.raises(ValueError, match="workers"):
        evaluate_split_stream_sensitivity(scenario, plan, 1, workers=0)
    nonnull = replace(
        scenario,
        process=replace(scenario.process, target_truth=scenario.process.target_truth | True),
    )
    with pytest.raises(ValueError, match="all-null"):
        evaluate_split_stream_sensitivity(nonnull, plan, 1)
