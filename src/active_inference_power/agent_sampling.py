"""Agent-controlled sampling and stopping traces.

This layer connects the pinned ``inferactively-pymdp`` binary agent to three explicitly
different policies: fixed horizon, posterior-threshold stopping, and an
e-process stopping rule.  Only the e-process policy receives an
optional-stopping validity claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .active_inference import AITestConfig, run_ai_test
from .sequential import EProcessResult, binary_e_process, fixed_horizon_binomial_pvalue

PolicyName = Literal["fixed_horizon", "posterior_threshold", "e_process"]


@dataclass(frozen=True)
class AgentSamplingTrace:
    """Per-observation trace for a policy-controlled binary agent."""

    policy: PolicyName
    actions: np.ndarray
    observations: np.ndarray
    posterior_signal: np.ndarray
    e_process_values: np.ndarray
    stopping_time: int
    decision: bool
    fixed_horizon_pvalue: float
    optional_stopping_valid: bool

def _validate_observations(observations: Sequence[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(observations)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isin(values, (0, 1))):
        raise ValueError("observations must be a non-empty binary vector")
    return values.astype(int, copy=False)


def run_agent_policy(
    observations: Sequence[int] | np.ndarray,
    config: AITestConfig,
    policy: PolicyName = "fixed_horizon",
    alpha: float = 0.05,
) -> AgentSamplingTrace:
    """Run a real agent and an explicit stopping policy on one stream."""
    values = _validate_observations(observations)
    if policy not in {"fixed_horizon", "posterior_threshold", "e_process"}:
        raise ValueError(f"unknown policy: {policy!r}")
    if values.size > config.horizon:
        raise ValueError("observation stream exceeds configured horizon")
    posterior = np.empty(values.size, dtype=float)
    actions = np.zeros(values.size, dtype=int)
    e_process_values = np.ones(values.size + 1, dtype=float)
    stop = values.size
    decision = False
    for index in range(1, values.size + 1):
        result = run_ai_test(values[:index], config)
        posterior[index - 1] = float(result.belief_trajectory[-1])
        if policy == "posterior_threshold" and posterior[index - 1] >= config.threshold:
            stop = index
            decision = True
            break
    e_result: EProcessResult | None = None
    if policy == "e_process":
        e_result = binary_e_process(values, 1.0 - config.reliability, config.reliability, alpha)
        e_process_values = e_result.e_process_values
        if e_result.crossing_time is not None:
            stop = e_result.crossing_time
            decision = True
    if policy == "fixed_horizon":
        decision = bool(posterior[-1] >= config.threshold)
    if policy == "posterior_threshold" and stop == values.size and not decision:
        decision = bool(posterior[-1] >= config.threshold)
    pvalue = fixed_horizon_binomial_pvalue(values[:stop], 1.0 - config.reliability, config.reliability)
    return AgentSamplingTrace(
        policy=policy,
        actions=actions[:stop].copy(),
        observations=values[:stop].copy(),
        posterior_signal=posterior[:stop].copy(),
        e_process_values=e_process_values[: stop + 1].copy(),
        stopping_time=stop,
        decision=decision,
        fixed_horizon_pvalue=pvalue,
        optional_stopping_valid=policy == "e_process",
    )


def simulate_agent_policy(
    config: AITestConfig,
    policy: PolicyName,
    n_reps: int,
    reliability: float | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Simulate binary observations and summarize policy decisions."""
    if isinstance(n_reps, bool) or int(n_reps) != n_reps or n_reps < 1:
        raise ValueError("n_reps must be a positive integer")
    rate = config.reliability if reliability is None else float(reliability)
    if not 0.0 < rate < 1.0:
        raise ValueError("reliability must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    traces = []
    for _ in range(int(n_reps)):
        observations = rng.binomial(1, rate, size=config.horizon)
        traces.append(run_agent_policy(observations, config, policy))
    decisions = np.asarray([trace.decision for trace in traces], dtype=float)
    stopping = np.asarray([trace.stopping_time for trace in traces], dtype=float)
    return {
        "policy": policy,
        "n_reps": int(n_reps),
        "seed": int(seed),
        "decision_rate": float(np.mean(decisions)),
        "mean_stopping_time": float(np.mean(stopping)),
        "traces": traces,
    }


__all__ = ["PolicyName", "AgentSamplingTrace", "run_agent_policy", "simulate_agent_policy"]
