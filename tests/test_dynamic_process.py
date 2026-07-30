"""Real time-varying and missingness process contract tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from active_inference_power import make_reference_action_scenario, run_action_loop


def test_time_varying_process_changes_cost_and_emission_paths() -> None:
    scenario = make_reference_action_scenario(horizon=4)
    transition_schedule = np.stack((scenario.process.transition, scenario.process.transition))
    likelihood_schedule = np.stack((scenario.process.likelihood, scenario.process.likelihood))
    cost_schedule = np.stack((scenario.process.action_costs, scenario.process.action_costs * 2.0))
    process = replace(
        scenario.process,
        transition_schedule=transition_schedule,
        likelihood_schedule=likelihood_schedule,
        action_cost_schedule=cost_schedule,
    )
    dynamic = replace(scenario, process=process)
    trace = run_action_loop(dynamic, "fixed_horizon", seed=61)
    assert trace.total_cost >= scenario.process.action_costs[0]
    assert dynamic.process.action_cost_at(1)[0] == pytest.approx(2.0 * scenario.process.action_costs[0])
    assert dynamic.process.likelihood_at(3).shape == scenario.process.likelihood.shape


def test_action_and_state_dependent_missingness_is_explicit() -> None:
    scenario = make_reference_action_scenario(horizon=3)
    action_missing = np.full(scenario.model.n_actions, 0.2)
    mar = replace(
        scenario.process,
        missingness_mode="mar_action",
        action_missing_probability=action_missing,
    )
    assert mar.missing_probability_at(0, 0, 0) == pytest.approx(0.2)
    state_missing = np.linspace(0.0, 0.8, scenario.model.n_states)
    mnar = replace(
        scenario.process,
        missingness_mode="mnar_state",
        state_missing_probability=state_missing,
    )
    assert mnar.missing_probability_at(0, 0, 3) == pytest.approx(state_missing[3])
    with pytest.raises(ValueError):
        replace(scenario.process, missingness_mode="mar_action")
    with pytest.raises(ValueError):
        replace(scenario.process, missingness_mode="mnar_state")


def test_schedule_extrapolation_is_declared_and_not_silent() -> None:
    scenario = make_reference_action_scenario(horizon=4)
    costs = np.stack((scenario.process.action_costs, scenario.process.action_costs * 2.0))
    held = replace(scenario.process, action_cost_schedule=costs, schedule_policy="hold_last")
    assert held.action_cost_at(5)[0] == pytest.approx(2.0 * scenario.process.action_costs[0])

    cycled = replace(scenario.process, action_cost_schedule=costs, schedule_policy="cycle")
    assert cycled.action_cost_at(2)[0] == pytest.approx(scenario.process.action_costs[0])

    strict = replace(scenario.process, action_cost_schedule=costs, schedule_policy="strict")
    with pytest.raises(IndexError, match="exceeds declared length"):
        strict.action_cost_at(2)
    with pytest.raises(ValueError, match="non-negative integer"):
        held.action_cost_at(-1)
    with pytest.raises(ValueError, match="schedule_policy"):
        replace(scenario.process, schedule_policy="implicit_wrap")
