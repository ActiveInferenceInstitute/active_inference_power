"""Real pymdp-backed agent policy and stopping tests."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference_power.active_inference import AITestConfig
from active_inference_power.agent_sampling import run_agent_policy, simulate_agent_policy


def test_agent_policies_return_explicit_validity_boundaries() -> None:
    config = AITestConfig(reliability=0.8, threshold=0.9, horizon=4)
    fixed = run_agent_policy([1, 1, 1, 1], config, "fixed_horizon")
    posterior = run_agent_policy([1, 1, 1, 1], config, "posterior_threshold")
    e_process = run_agent_policy([1, 1, 1, 1], config, "e_process")
    assert fixed.stopping_time == 4
    assert posterior.stopping_time <= 4
    assert not fixed.optional_stopping_valid
    assert not posterior.optional_stopping_valid
    assert e_process.optional_stopping_valid
    assert e_process.e_process_values.shape[0] == e_process.stopping_time + 1


def test_agent_policy_simulation_is_seeded_and_validates() -> None:
    config = AITestConfig(horizon=2)
    left = simulate_agent_policy(config, "e_process", 3, seed=4)
    right = simulate_agent_policy(config, "e_process", 3, seed=4)
    assert left["decision_rate"] == right["decision_rate"]
    with pytest.raises(ValueError):
        run_agent_policy(np.array([0, 1, 0]), config, "fixed_horizon")
    with pytest.raises(ValueError):
        run_agent_policy([0, 1], config, "unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        simulate_agent_policy(config, "fixed_horizon", 0)
