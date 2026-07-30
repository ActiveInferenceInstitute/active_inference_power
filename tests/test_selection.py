"""Real split-stream selection and confirmation tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
from typing import Any

from active_inference_power import (
    SelectionPolicySpec,
    make_reference_action_scenario,
    make_static_binary_e_process_scenario,
    make_static_binary_split_confirmation_scenario,
    run_split_stream_confirmation,
    simulate_split_stream_confirmation,
)
from active_inference_power.schema import validate_suite_result


def test_split_stream_has_distinct_hashes_and_visible_filtration() -> None:
    scenario = make_static_binary_split_confirmation_scenario(horizon=4)
    result = run_split_stream_confirmation(
        scenario,
        SelectionPolicySpec(strategy="minimum_p", max_steps=4, confirmation_steps=4),
        selection_seed=12,
        confirmation_seed=13,
    )
    assert result.validity == "formal_conditional_split_confirmation"
    assert result.selection_stream_hash != result.confirmation_stream_hash
    assert result.selection.visible_only
    assert all(not event.hidden_state_exposed for event in result.selection.events)
    assert result.selection.selected_target in (0, 1)
    assert 0.0 <= result.confirmation_p_value <= 1.0
    assert result.validity_assumptions == scenario.setting.evidence_assumptions
    # Repeating the declared alpha is compatible but cannot create a distinct,
    # unrecorded decision threshold or confirmation-stream hash.
    declared_alpha = run_split_stream_confirmation(
        scenario,
        SelectionPolicySpec(strategy="minimum_p", max_steps=4, confirmation_steps=4),
        selection_seed=12,
        confirmation_seed=13,
        alpha=scenario.setting.alpha,
    )
    assert declared_alpha.confirmation_stream_hash == result.confirmation_stream_hash
    assert declared_alpha.rejected is (result.confirmation_p_value <= scenario.setting.alpha)
    with pytest.raises(ValueError, match="exactly match"):
        run_split_stream_confirmation(
            scenario,
            selection_seed=12,
            confirmation_seed=13,
            alpha=0.01,
        )


def test_split_stream_simulation_and_suite_schema() -> None:
    scenario = make_static_binary_split_confirmation_scenario(horizon=3)
    summary = simulate_split_stream_confirmation(scenario, 6, seed=22)
    assert summary["n_reps"] == 6
    assert summary["selection_streams"] == summary["confirmation_streams"] == 6
    assert summary["validity"] == "formal_conditional_split_confirmation"
    assert summary["power_estimand"] == (
        "conditional_confirmation_rejection_given_selected_target_true"
    )
    assert summary["fwer_estimand"] == "single_confirmed_target_false_rejection_rate"
    assert summary["assumptions"] == list(scenario.setting.evidence_assumptions)
    assert summary["rejection_count"] == (
        summary["true_target_joint_rejection_count"]
        + summary["false_rejection_count"]
    )
    assert summary["true_target_joint_rejection_count"] <= summary[
        "true_target_selection_count"
    ]
    assert summary["power_defined"] is (
        summary["true_target_selection_count"] > 0
    )
    if summary["power_defined"]:
        assert summary["power"] == pytest.approx(
            summary["true_target_joint_rejection_count"]
            / summary["true_target_selection_count"]
        )
    else:
        assert summary["power"] is None
    suite: dict[str, Any] = {
        "suite_name": "split",
        "scenarios": [scenario.scenario_id],
        "policies": ["adaptive_target"],
        "operating_characteristics": [],
        "online_fdr": {},
        "claim_boundary": "conditional",
        "split_stream_confirmation": summary,
    }
    with pytest.raises(ValueError):
        validate_suite_result(suite)
    suite["operating_characteristics"] = [{
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "policy": "adaptive_target",
        "n_reps": 1,
        "summary": {"power": 0.0, "fdr": 0.0, "fwer": 0.0},
        "validity": "diagnostic",
    }]
    validate_suite_result(suite)
    wrong_power = 0.0 if summary["power"] != 0.0 else 1.0
    with pytest.raises(ValueError, match="selected-true denominator"):
        validate_suite_result(
            {
                **suite,
                "split_stream_confirmation": {**summary, "power": wrong_power},
            }
        )


def test_split_stream_marks_conditional_power_not_estimable_without_true_selection() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=3)
    summary = simulate_split_stream_confirmation(
        scenario,
        3,
        seed=22,
        policy=SelectionPolicySpec(
            strategy="least_informed", max_steps=3, confirmation_steps=3
        ),
    )
    assert summary["true_target_selection_count"] == 0
    assert summary["true_target_joint_rejection_count"] == 0
    assert summary["power_defined"] is False
    assert summary["power"] is None
    assert summary["true_target_selection_rate"] == 0.0
    assert summary["true_target_joint_rejection_rate"] == 0.0


def test_richer_context_is_explicitly_diagnostic() -> None:
    result = run_split_stream_confirmation(
        make_reference_action_scenario(horizon=3), selection_seed=30, confirmation_seed=31
    )
    assert result.validity == "conditional_split_confirmation_diagnostic"
    with pytest.raises(ValueError):
        run_split_stream_confirmation(
            make_static_binary_e_process_scenario(horizon=2),
            selection_seed=4,
            confirmation_seed=4,
        )


def test_selection_contract_rejects_invalid_policy_and_missing_confirmation() -> None:
    with pytest.raises(ValueError):
        SelectionPolicySpec(strategy="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SelectionPolicySpec(max_steps=0)
    scenario = make_static_binary_e_process_scenario(horizon=2)
    altered = replace(scenario.process, missing_probability=0.999999)
    with pytest.raises(ValueError):
        run_split_stream_confirmation(replace(scenario, process=altered), selection_seed=5, confirmation_seed=6)
