"""Real paired dynamic-process stress controls."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from active_inference_power.dynamic_process_sensitivity import (
    DynamicProcessSensitivityPlan,
    dynamic_process_pairs,
    evaluate_dynamic_process_sensitivity,
)
from active_inference_power.scenario import make_reference_action_scenario


def test_dynamic_plan_validates_predeclared_controls() -> None:
    plan = DynamicProcessSensitivityPlan(horizon=3)
    assert plan.configuration_hash == DynamicProcessSensitivityPlan(horizon=3).configuration_hash
    assert len(cast(list[str], plan.to_mapping()["regimes"])) == 6
    for values, message in (
        ({"horizon": 0}, "horizon"),
        ({"context_persistence": 0.0}, "context_persistence"),
        ({"reliability_end_multiplier": 0.0}, "reliability_end_multiplier"),
        ({"mcar_missing_end": 1.0}, "mcar_missing_end"),
        ({"mar_missing_low": 0.4, "mar_missing_high": 0.2}, "mar missingness"),
        ({"mnar_missing_low": 0.4, "mnar_missing_high": 0.2}, "mnar missingness"),
        ({"policy": "unknown"}, "policies"),
    ):
        with pytest.raises(ValueError, match=message):
            DynamicProcessSensitivityPlan(**values)  # type: ignore[arg-type]


def test_dynamic_pairs_make_static_controls_and_all_required_process_regimes() -> None:
    base = make_reference_action_scenario(horizon=4)
    plan = DynamicProcessSensitivityPlan(horizon=3)
    pairs = dynamic_process_pairs(base, plan)
    assert set(pairs) == {
        "persistent_context", "reliability_drift", "action_transition",
        "mcar_missingness", "mar_action_missingness", "mnar_state_missingness",
    }
    for regime, (control, dynamic) in pairs.items():
        assert control.setting.max_steps == dynamic.setting.max_steps == 3
        assert control.metadata is not None and control.metadata["paired_role"] == "static_control"
        assert dynamic.metadata is not None and dynamic.metadata["paired_regime"] == regime
        assert dynamic.metadata["validity"] == "diagnostic_dynamic_process_stress"
    assert pairs["persistent_context"][1].process.transition_schedule is not None
    assert pairs["reliability_drift"][1].process.likelihood_schedule is not None
    assert pairs["mcar_missingness"][1].process.missing_probability_schedule is not None
    assert pairs["mar_action_missingness"][1].process.missingness_mode == "mar_action"
    assert pairs["mnar_state_missingness"][1].process.missingness_mode == "mnar_state"
    dynamic_base = replace(
        base,
        process=replace(base.process, transition_schedule=np.repeat(base.process.transition[None, ...], 1, axis=0)),
    )
    with pytest.raises(ValueError, match="static"):
        dynamic_process_pairs(dynamic_base, plan)


def test_dynamic_process_surface_is_paired_and_worker_invariant() -> None:
    base = make_reference_action_scenario(horizon=3)
    plan = DynamicProcessSensitivityPlan(horizon=3)
    serial = evaluate_dynamic_process_sensitivity(base, plan, 2, seed=53, workers=1)
    parallel = evaluate_dynamic_process_sensitivity(base, plan, 2, seed=53, workers=2)
    assert serial["seed_schedule"] == parallel["seed_schedule"]
    assert serial["regimes"] == parallel["regimes"]
    regimes = cast(list[dict[str, Any]], serial["regimes"])
    assert len(regimes) == 6
    for regime in regimes:
        dynamic = cast(dict[str, Any], regime["dynamic"])
        panel = cast(dict[str, Any], dynamic["process_panel"])
        summary = cast(dict[str, Any], dynamic["summary"])
        calibration = cast(dict[str, Any], dynamic["marginal_calibration"])
        paired = cast(dict[str, Any], regime["paired_static_to_dynamic"])
        assert regime["validity"] == "diagnostic_dynamic_process_stress"
        assert summary["n_reps"] == 2
        assert calibration["n_reps"] == 2
        assert len(panel["time"]) == 3
        assert len(panel["mean_observation_one_probability"]) == 3
        assert set(paired) == {"power", "fdr", "fwer", "cost", "stopping_time"}
        assert cast(dict[str, Any], paired["power"])["n"] == 2
    descriptions = {
        cast(dict[str, Any], cast(dict[str, Any], regime["dynamic"])["process_panel"])["missingness_description"]
        for regime in regimes
    }
    assert {"time-scheduled MCAR", "action-dependent MAR", "state-dependent MNAR"}.issubset(descriptions)


def test_dynamic_surface_rejects_invalid_execution_inputs() -> None:
    base = make_reference_action_scenario(horizon=2)
    plan = DynamicProcessSensitivityPlan(horizon=2)
    with pytest.raises(ValueError, match="n_reps"):
        evaluate_dynamic_process_sensitivity(base, plan, 0)
    with pytest.raises(ValueError, match="workers"):
        evaluate_dynamic_process_sensitivity(base, plan, 1, workers=0)
    with pytest.raises(ValueError, match="seed"):
        evaluate_dynamic_process_sensitivity(base, plan, 1, seed=True)
