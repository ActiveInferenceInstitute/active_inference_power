"""Composable generative-model, process, and testing-setting contracts.

Power is not an intrinsic property of an agent or a correction procedure.  It
is an operating characteristic conditional on the agent's model, the process
that generated observations, and the setting in which decisions are made.
This module makes those three objects explicit and hashable enough to serve as
the public scenario contract for the suite.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, cast

import numpy as np


def _labels(value: tuple[str, ...] | list[str], name: str) -> tuple[str, ...]:
    result = tuple(str(item) for item in value)
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique non-empty labels")
    return result


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-D array")
    return result.copy()


def _probability_rows(value: np.ndarray, name: str, axis: int) -> None:
    if np.any(value < 0.0) or not np.allclose(np.sum(value, axis=axis), 1.0, atol=1e-10):
        raise ValueError(f"{name} must be non-negative and sum to one along axis {axis}")


@dataclass(frozen=True)
class GenerativeModelSpec:
    """Agent-side generative model used for inference.

    ``likelihood[action, observation, state]`` and
    ``transition[action, next_state, state]`` are action-conditioned so the
    same contract supports active sensing and environment-level control.
    """

    state_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    action_names: tuple[str, ...]
    prior: np.ndarray
    likelihood: np.ndarray
    transition: np.ndarray
    state_truth: np.ndarray
    state_target: np.ndarray
    state_context: np.ndarray
    action_target: np.ndarray
    action_sensor: np.ndarray
    hypothesis_ids: tuple[str, ...]
    preferences: np.ndarray | None = None

    def __post_init__(self) -> None:
        states = _labels(self.state_names, "state_names")
        observations = _labels(self.observation_names, "observation_names")
        actions = _labels(self.action_names, "action_names")
        prior = _array(self.prior, "prior", 1)
        likelihood = _array(self.likelihood, "likelihood", 3)
        transition = _array(self.transition, "transition", 3)
        if prior.shape != (len(states),):
            raise ValueError("prior must align with state_names")
        if likelihood.shape != (len(actions), len(observations), len(states)):
            raise ValueError("likelihood shape must be (actions, observations, states)")
        if transition.shape != (len(actions), len(states), len(states)):
            raise ValueError("transition shape must be (actions, states, states)")
        _probability_rows(prior.reshape(1, -1), "prior", 1)
        _probability_rows(likelihood, "likelihood", 1)
        _probability_rows(transition, "transition", 1)
        truth = np.asarray(self.state_truth, dtype=int)
        target = np.asarray(self.state_target, dtype=int)
        context = np.asarray(self.state_context, dtype=int)
        action_target = np.asarray(self.action_target, dtype=int)
        action_sensor = np.asarray(self.action_sensor, dtype=int)
        if truth.shape != prior.shape or target.shape != prior.shape or context.shape != prior.shape:
            raise ValueError("state metadata must align with prior")
        if action_target.shape != (len(actions),) or action_sensor.shape != (len(actions),):
            raise ValueError("action metadata must align with action_names")
        hypotheses = _labels(self.hypothesis_ids, "hypothesis_ids")
        if np.any(target < 0) or np.any(target >= len(hypotheses)):
            raise ValueError("state_target contains an unknown hypothesis index")
        if np.any(action_target < -1) or np.any(action_target >= len(hypotheses)):
            raise ValueError("action_target contains an unknown hypothesis index")
        if np.any((truth < 0) | (truth > 1)) or np.any(context < 0):
            raise ValueError("state_truth must be binary and state_context non-negative")
        preference = np.zeros(len(observations), dtype=float) if self.preferences is None else _array(self.preferences, "preferences", 1)
        if preference.shape != (len(observations),):
            raise ValueError("preferences must align with observation_names")
        for name, value in (("state_truth", truth), ("state_target", target), ("state_context", context), ("action_target", action_target), ("action_sensor", action_sensor)):
            object.__setattr__(self, name, value.copy())
        object.__setattr__(self, "state_names", states)
        object.__setattr__(self, "observation_names", observations)
        object.__setattr__(self, "action_names", actions)
        object.__setattr__(self, "hypothesis_ids", hypotheses)
        object.__setattr__(self, "prior", prior / np.sum(prior))
        object.__setattr__(self, "likelihood", likelihood)
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "preferences", preference)

    @property
    def n_states(self) -> int:
        return len(self.state_names)

    @property
    def n_actions(self) -> int:
        return len(self.action_names)

    @property
    def n_observations(self) -> int:
        return len(self.observation_names)

    def to_mapping(self) -> dict[str, Any]:
        """Return the JSON-safe contract representation."""
        return {
            "state_names": list(self.state_names),
            "observation_names": list(self.observation_names),
            "action_names": list(self.action_names),
            "prior": self.prior.tolist(),
            "likelihood": self.likelihood.tolist(),
            "transition": self.transition.tolist(),
            "state_truth": self.state_truth.tolist(),
            "state_target": self.state_target.tolist(),
            "state_context": self.state_context.tolist(),
            "action_target": self.action_target.tolist(),
            "action_sensor": self.action_sensor.tolist(),
            "hypothesis_ids": list(self.hypothesis_ids),
            "preferences": np.asarray(self.preferences, dtype=float).tolist(),
        }

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GenerativeProcessSpec:
    """True data-generating process and action-dependent environment.

    ``action_costs`` express a simulated resource quantity, not an alpha
    accounting quantity or a likelihood-ratio evidence value.  ``cost_unit``
    and ``cost_scale_id`` make that interpretation explicit in the hashed
    scenario contract.  A local/default scale deliberately does not license
    comparisons with a differently declared scenario.
    """

    initial_state: np.ndarray
    transition: np.ndarray
    likelihood: np.ndarray
    target_truth: np.ndarray
    action_costs: np.ndarray
    missing_probability: float = 0.0
    name: str = "synthetic_action_environment"
    transition_schedule: np.ndarray | None = None
    likelihood_schedule: np.ndarray | None = None
    action_cost_schedule: np.ndarray | None = None
    missing_probability_schedule: np.ndarray | None = None
    schedule_policy: str = "hold_last"
    missingness_mode: str = "mcar"
    action_missing_probability: np.ndarray | None = None
    state_missing_probability: np.ndarray | None = None
    cost_unit: str = "simulation cost unit"
    cost_scale_id: str = "scenario-local"

    def __post_init__(self) -> None:
        initial = _array(self.initial_state, "initial_state", 1)
        transition = _array(self.transition, "process transition", 3)
        likelihood = _array(self.likelihood, "process likelihood", 3)
        truth = np.asarray(self.target_truth, dtype=bool)
        costs = _array(self.action_costs, "action_costs", 1)
        if transition.shape[1] != transition.shape[2] or transition.shape[1] != initial.size:
            raise ValueError("process transition must be (actions, states, states)")
        if likelihood.shape[0] != transition.shape[0] or likelihood.shape[2] != initial.size:
            raise ValueError("process likelihood must align with process actions and states")
        if costs.shape != (transition.shape[0],) or np.any(costs < 0.0):
            raise ValueError("action_costs must be non-negative and align with actions")
        _probability_rows(initial.reshape(1, -1), "initial_state", 1)
        _probability_rows(transition, "process transition", 1)
        _probability_rows(likelihood, "process likelihood", 1)
        if truth.ndim != 1 or truth.size < 1:
            raise ValueError("target_truth must be a non-empty vector")
        missing = float(self.missing_probability)
        if not math.isfinite(missing) or not 0.0 <= missing < 1.0:
            raise ValueError("missing_probability must lie in [0, 1)")
        if not self.name:
            raise ValueError("process name must be non-empty")
        if not isinstance(self.cost_unit, str) or not self.cost_unit.strip():
            raise ValueError("cost_unit must be a non-empty string")
        if not isinstance(self.cost_scale_id, str) or not self.cost_scale_id.strip():
            raise ValueError("cost_scale_id must be a non-empty string")
        if self.schedule_policy not in {"hold_last", "cycle", "strict"}:
            raise ValueError("schedule_policy must be hold_last, cycle, or strict")
        if self.missingness_mode not in {"mcar", "mar_action", "mnar_state"}:
            raise ValueError("missingness_mode must be mcar, mar_action, or mnar_state")
        transition_schedule = None if self.transition_schedule is None else _array(self.transition_schedule, "transition_schedule", 4)
        likelihood_schedule = None if self.likelihood_schedule is None else _array(self.likelihood_schedule, "likelihood_schedule", 4)
        action_cost_schedule = None if self.action_cost_schedule is None else _array(self.action_cost_schedule, "action_cost_schedule", 2)
        missing_schedule = None if self.missing_probability_schedule is None else _array(self.missing_probability_schedule, "missing_probability_schedule", 1)
        if transition_schedule is not None:
            if transition_schedule.shape[0] < 1:
                raise ValueError("transition_schedule must contain at least one time point")
            if transition_schedule.shape[1:] != transition.shape:
                raise ValueError("transition_schedule must have shape (time, actions, states, states)")
            for index, value in enumerate(transition_schedule):
                _probability_rows(value, f"transition_schedule[{index}]", 1)
        if likelihood_schedule is not None:
            if likelihood_schedule.shape[0] < 1:
                raise ValueError("likelihood_schedule must contain at least one time point")
            if likelihood_schedule.shape[1:] != likelihood.shape:
                raise ValueError("likelihood_schedule must have shape (time, actions, observations, states)")
            for index, value in enumerate(likelihood_schedule):
                _probability_rows(value, f"likelihood_schedule[{index}]", 1)
        if action_cost_schedule is not None and action_cost_schedule.shape[1:] != costs.shape:
            raise ValueError("action_cost_schedule must have shape (time, actions)")
        if action_cost_schedule is not None and action_cost_schedule.shape[0] < 1:
            raise ValueError("action_cost_schedule must contain at least one time point")
        if action_cost_schedule is not None and np.any(action_cost_schedule < 0.0):
            raise ValueError("action_cost_schedule must be non-negative")
        if missing_schedule is not None and (np.any(missing_schedule < 0.0) or np.any(missing_schedule >= 1.0)):
            raise ValueError("missing_probability_schedule must lie in [0, 1)")
        if missing_schedule is not None and missing_schedule.size < 1:
            raise ValueError("missing_probability_schedule must contain at least one time point")
        action_missing = None if self.action_missing_probability is None else _array(self.action_missing_probability, "action_missing_probability", 1)
        state_missing = None if self.state_missing_probability is None else _array(self.state_missing_probability, "state_missing_probability", 1)
        if action_missing is not None and (action_missing.shape != costs.shape or np.any((action_missing < 0.0) | (action_missing >= 1.0))):
            raise ValueError("action_missing_probability must align with actions and lie in [0, 1)")
        if state_missing is not None and (state_missing.shape != initial.shape or np.any((state_missing < 0.0) | (state_missing >= 1.0))):
            raise ValueError("state_missing_probability must align with states and lie in [0, 1)")
        if self.missingness_mode == "mar_action" and action_missing is None:
            raise ValueError("mar_action requires action_missing_probability")
        if self.missingness_mode == "mnar_state" and state_missing is None:
            raise ValueError("mnar_state requires state_missing_probability")
        object.__setattr__(self, "initial_state", initial / np.sum(initial))
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "likelihood", likelihood)
        object.__setattr__(self, "target_truth", truth)
        object.__setattr__(self, "action_costs", costs)
        object.__setattr__(self, "missing_probability", missing)
        object.__setattr__(self, "schedule_policy", str(self.schedule_policy))
        object.__setattr__(self, "transition_schedule", None if transition_schedule is None else transition_schedule.copy())
        object.__setattr__(self, "likelihood_schedule", None if likelihood_schedule is None else likelihood_schedule.copy())
        object.__setattr__(self, "action_cost_schedule", None if action_cost_schedule is None else action_cost_schedule.copy())
        object.__setattr__(self, "missing_probability_schedule", None if missing_schedule is None else missing_schedule.copy())
        object.__setattr__(self, "action_missing_probability", None if action_missing is None else action_missing.copy())
        object.__setattr__(self, "state_missing_probability", None if state_missing is None else state_missing.copy())
        object.__setattr__(self, "cost_unit", self.cost_unit.strip())
        object.__setattr__(self, "cost_scale_id", self.cost_scale_id.strip())

    def _schedule_index(self, time: int, length: int) -> int:
        """Resolve a schedule index under an explicit extrapolation policy."""
        if isinstance(time, bool) or int(time) != time or time < 0:
            raise ValueError("schedule time must be a non-negative integer")
        if self.schedule_policy == "cycle":
            return int(time) % length
        if int(time) < length:
            return int(time)
        if self.schedule_policy == "hold_last":
            return length - 1
        raise IndexError(
            f"schedule time {int(time)} exceeds declared length {length} under strict policy"
        )

    def transition_at(self, time: int) -> np.ndarray:
        """Return the process transition kernel for zero-based time."""
        return self.transition if self.transition_schedule is None else self.transition_schedule[self._schedule_index(time, self.transition_schedule.shape[0])]

    def likelihood_at(self, time: int) -> np.ndarray:
        """Return the process emission kernel for zero-based time."""
        return self.likelihood if self.likelihood_schedule is None else self.likelihood_schedule[self._schedule_index(time, self.likelihood_schedule.shape[0])]

    def action_cost_at(self, time: int) -> np.ndarray:
        """Return action costs for zero-based time."""
        return self.action_costs if self.action_cost_schedule is None else self.action_cost_schedule[self._schedule_index(time, self.action_cost_schedule.shape[0])]

    def missing_probability_at(self, time: int, action: int, state: int) -> float:
        """Return the evaluator-side missingness probability."""
        base = self.missing_probability if self.missing_probability_schedule is None else float(self.missing_probability_schedule[self._schedule_index(time, self.missing_probability_schedule.size)])
        if self.missingness_mode == "mar_action":
            action_missing = cast(np.ndarray, self.action_missing_probability)
            return float(action_missing[int(action)])
        if self.missingness_mode == "mnar_state":
            state_missing = cast(np.ndarray, self.state_missing_probability)
            return float(state_missing[int(state)])
        return base

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "initial_state": self.initial_state.tolist(),
            "transition": self.transition.tolist(),
            "likelihood": self.likelihood.tolist(),
            "target_truth": self.target_truth.astype(int).tolist(),
            "action_costs": self.action_costs.tolist(),
            "missing_probability": self.missing_probability,
            "schedule_policy": self.schedule_policy,
            "transition_schedule": None if self.transition_schedule is None else self.transition_schedule.tolist(),
            "likelihood_schedule": None if self.likelihood_schedule is None else self.likelihood_schedule.tolist(),
            "action_cost_schedule": None if self.action_cost_schedule is None else self.action_cost_schedule.tolist(),
            "missing_probability_schedule": None if self.missing_probability_schedule is None else self.missing_probability_schedule.tolist(),
            "missingness_mode": self.missingness_mode,
            "action_missing_probability": None if self.action_missing_probability is None else self.action_missing_probability.tolist(),
            "state_missing_probability": None if self.state_missing_probability is None else self.state_missing_probability.tolist(),
            "cost_unit": self.cost_unit,
            "cost_scale_id": self.cost_scale_id,
        }

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SettingSpec:
    """Decision setting and declared inferential boundary.

    Action-loop decisions are posterior- or e-value-based; they do not apply a
    family p-value correction to action-level p-values.  ``correction`` is
    therefore deliberately restricted to ``"none"``.  Fixed-family and online
    correction procedures have separate, executable contracts elsewhere in the
    package.
    """

    alpha: float = 0.05
    correction: str = "none"
    stopping_policy: str = "fixed_horizon"
    posterior_threshold: float = 0.9
    max_steps: int = 12
    cost_budget: float = 1.0e12
    confidence: float = 0.95
    formal_validity: str = "conditional_on_declared_process"
    family_size: int = 2
    evidence_assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (("alpha", self.alpha), ("confidence", self.confidence)):
            value = float(value)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
        threshold = float(self.posterior_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("posterior_threshold must lie in [0, 1]")
        if isinstance(self.max_steps, bool) or int(self.max_steps) != self.max_steps or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        budget = float(self.cost_budget)
        if not (math.isinf(budget) or math.isfinite(budget)) or budget <= 0.0:
            raise ValueError("cost_budget must be positive or infinity")
        if self.correction != "none":
            raise ValueError(
                "action-loop correction must be 'none'; its decisions do not apply p-value corrections"
            )
        if self.stopping_policy not in {
            "fixed_horizon",
            "posterior_threshold",
            "e_process",
            "information_gain",
            "cost_aware",
            "adaptive_target",
            "posterior_sampling",
        }:
            raise ValueError("unsupported stopping_policy")
        if isinstance(self.family_size, bool) or int(self.family_size) != self.family_size or self.family_size < 1:
            raise ValueError("family_size must be a positive integer")
        if not self.formal_validity:
            raise ValueError("formal_validity must be non-empty")
        assumptions = tuple(str(item) for item in self.evidence_assumptions)
        if any(not item for item in assumptions) or len(set(assumptions)) != len(assumptions):
            raise ValueError("evidence_assumptions must contain unique non-empty labels")
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "posterior_threshold", threshold)
        object.__setattr__(self, "max_steps", int(self.max_steps))
        object.__setattr__(self, "cost_budget", budget)
        object.__setattr__(self, "family_size", int(self.family_size))
        object.__setattr__(self, "evidence_assumptions", assumptions)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "correction": self.correction,
            "stopping_policy": self.stopping_policy,
            "posterior_threshold": self.posterior_threshold,
            "max_steps": self.max_steps,
            "cost_budget": self.cost_budget,
            "confidence": self.confidence,
            "formal_validity": self.formal_validity,
            "family_size": self.family_size,
            "evidence_assumptions": list(self.evidence_assumptions),
        }


@dataclass(frozen=True)
class PowerScenario:
    """Hashable model/process/setting contract evaluated with a separate policy.

    ``SettingSpec.stopping_policy`` is a hashed default used when no action
    policy is supplied. ``PowerScenario`` deliberately does not contain a
    separately supplied surface/action policy. Surface rows pair this stable
    contract with an explicitly recorded executed policy so an investigator can
    compare policies without implying that a surface-policy choice was hidden
    inside the scenario hash.
    """

    scenario_id: str
    model: GenerativeModelSpec
    process: GenerativeProcessSpec
    setting: SettingSpec
    description: str = ""
    claim_role: str = "synthetic_conditional_evidence"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip() or (
            not self.description and self.scenario_id != "reference_action_loop"
        ):
            raise ValueError("scenario_id and description must identify the scenario")
        if self.model.n_states != self.process.initial_state.size:
            raise ValueError("model and process state spaces must agree")
        if self.model.n_actions != self.process.transition.shape[0]:
            raise ValueError("model and process action spaces must agree")
        if self.model.n_observations != self.process.likelihood.shape[1]:
            raise ValueError("model and process observation spaces must agree")
        if self.setting.family_size != len(self.model.hypothesis_ids):
            raise ValueError("setting.family_size must match model hypotheses")
        if self.process.target_truth.size != self.setting.family_size:
            raise ValueError("process.target_truth must match setting.family_size")
        if not self.claim_role:
            raise ValueError("claim_role must be non-empty")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def truth_consistent(self) -> bool:
        """Return whether the process encodes fixed truth per target.

        ``GenerativeModelSpec`` contains hypothetical null and alternative
        states so that an agent can update beliefs.  The process, however,
        must start in states whose truth agrees with ``target_truth`` and keep
        reachable process mass inside that truth-consistent set.  This check is
        deliberately a predicate rather than a constructor failure: dynamic
        or deliberately misspecified scenarios remain useful diagnostic
        stress tests, but they cannot receive a formal fixed-truth label.
        """
        target_truth = np.asarray(self.process.target_truth, dtype=bool)
        truth_consistent = self.model.state_truth == target_truth[self.model.state_target]
        if np.any(self.process.initial_state[~truth_consistent] > 1e-12):
            return False
        # Normalize the static kernel and every scheduled kernel to
        # ``(time, action, next_state, state)`` before checking reachability.
        # Keeping the time axis explicit avoids accidentally iterating over
        # individual rows of the static kernel (which would silently weaken
        # this contract).
        kernels = [self.process.transition[None, ...]]
        if self.process.transition_schedule is not None:
            kernels.append(self.process.transition_schedule)
        for kernel_group in kernels:
            for time_kernel in kernel_group:
                for action_matrix in time_kernel:
                    # Rows are next states and columns are current states;
                    # only transitions from a truth-consistent current state
                    # into an inconsistent next state violate fixed truth.
                    if np.any(action_matrix[np.ix_(~truth_consistent, truth_consistent)] > 1e-12):
                        return False
        return True

    def _contract_mapping(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "claim_role": self.claim_role,
            "metadata": dict(self.metadata or {}),
            "model": self.model.to_mapping(),
            "process": self.process.to_mapping(),
            "setting": self.setting.to_mapping(),
        }

    def to_mapping(self) -> dict[str, Any]:
        """Return the contract plus its self-contained lineage digest."""
        return self._contract_mapping() | {"scenario_hash": self.configuration_hash}

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(self._contract_mapping(), sort_keys=True, separators=(",", ":"), allow_nan=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_reference_action_scenario(
    *,
    reliability: float = 0.8,
    horizon: int = 12,
    alpha: float = 0.05,
    scenario_id: str | None = None,
) -> PowerScenario:
    """Build the small, fully action-conditioned reference environment.

    There are two targets, two contexts, and binary truth.  Target/sensor
    actions set the active target and sensor quality; ``switch_context`` changes
    the latent context.  The hidden truth itself is fixed for each target.
    """
    r = float(reliability)
    if not 0.5 < r < 1.0:
        raise ValueError("reference reliability must lie in (0.5, 1)")
    resolved_scenario_id = scenario_id
    if resolved_scenario_id is None:
        if math.isclose(r, 0.8) and int(horizon) == 12 and math.isclose(float(alpha), 0.05):
            resolved_scenario_id = "reference_action_loop"
        else:
            def token(value: float) -> str:
                return format(value, ".8g").replace("-", "m").replace(".", "p")

            resolved_scenario_id = f"reference_action_loop__r{token(r)}__h{int(horizon)}__a{token(float(alpha))}"
    state_names = tuple(
        f"target{target}_context{context}_h{truth}"
        for target in range(2)
        for context in range(2)
        for truth in range(2)
    )
    state_target = np.repeat(np.arange(2), 4)
    state_context = np.tile(np.repeat(np.arange(2), 2), 2)
    state_truth = np.tile(np.arange(2), 4)
    target_truth = np.array([True, False], dtype=bool)
    action_names = ("target0_cheap", "target0_precise", "target1_cheap", "target1_precise", "switch_context", "stop")
    action_target = np.array([0, 0, 1, 1, -1, -1], dtype=int)
    action_sensor = np.array([0, 1, 0, 1, -1, -1], dtype=int)
    likelihood = np.empty((len(action_names), 2, len(state_names)), dtype=float)
    transition = np.zeros((len(action_names), len(state_names), len(state_names)), dtype=float)
    for action in range(len(action_names)):
        target = int(action_target[action])
        for state, (state_target_i, context, truth) in enumerate(zip(state_target, state_context, state_truth)):
            next_target = state_target_i if target < 0 else target
            next_context = 1 - context if action == 4 else context
            next_truth = int(target_truth[next_target]) if target >= 0 else int(truth)
            next_state = int(np.flatnonzero((state_target == next_target) & (state_context == next_context) & (state_truth == next_truth))[0])
            transition[action, next_state, state] = 1.0
            sensor = 1 if action_sensor[action] == 1 else 0
            quality = min(0.98, r + (0.08 if sensor else 0.0) + (0.04 if context else 0.0))
            p_high = quality if truth else 1.0 - quality
            likelihood[action, :, state] = (1.0 - p_high, p_high)
    prior = np.full(len(state_names), 1.0 / len(state_names), dtype=float)
    process_prior = np.zeros(len(state_names), dtype=float)
    for target, truth in enumerate((True, False)):
        index = int(np.flatnonzero((state_target == target) & (state_context == 0) & (state_truth == int(truth)))[0])
        process_prior[index] = 1.0
    process_prior /= np.sum(process_prior)
    model = GenerativeModelSpec(
        state_names, ("low", "high"), action_names, prior, likelihood, transition,
        state_truth, state_target, state_context, action_target, action_sensor,
        ("target_0", "target_1"), np.zeros(2),
    )
    process = GenerativeProcessSpec(
        process_prior, transition.copy(), likelihood.copy(), target_truth,
        np.array([1.0, 2.0, 1.0, 2.5, 0.5, 0.0]),
        name="binary_context_control",
        cost_unit="reference action-cost unit",
        cost_scale_id="reference_action_cost_v1",
    )
    setting = SettingSpec(
        alpha=alpha,
        correction="none",
        max_steps=horizon,
        cost_budget=1.0e12,
        family_size=2,
        stopping_policy="cost_aware",
        formal_validity="conditional_on_declared_superuniformity_assumptions",
        evidence_assumptions=(
            "fixed_truth",
            "conditional_superuniformity",
            "predictable_selection",
        ),
    )
    return PowerScenario(
        resolved_scenario_id, model, process, setting,
        "Generative model and process for two-target binary truth with action-conditioned sensing, context transitions, cost, and stopping.",
        metadata={"reliability": r, "horizon": int(horizon), "n_targets": 2},
    )


def make_static_binary_e_process_scenario(
    *, reliability: float = 0.8, horizon: int = 12, alpha: float = 0.05
) -> PowerScenario:
    """Build a reference action scenario satisfying the e-process channel contract.

    Unlike the richer context-control reference, each target/truth cell has a
    state-invariant Bernoulli emission rate. This makes the single-step
    likelihood ratio conditionally calibrated under predictable target
    selection; the context-changing environment remains a deliberate
    diagnostic contrast.
    """
    scenario = make_reference_action_scenario(
        reliability=reliability, horizon=horizon, alpha=alpha
    )
    likelihood = scenario.model.likelihood.copy()
    for action in range(scenario.model.n_actions):
        sensor = int(scenario.model.action_sensor[action])
        quality = min(0.98, float(reliability) + (0.08 if sensor == 1 else 0.0))
        for state, truth in enumerate(scenario.model.state_truth):
            p_high = quality if truth else 1.0 - quality
            likelihood[action, :, state] = (1.0 - p_high, p_high)
    model = replace(scenario.model, likelihood=likelihood)
    process = replace(
        scenario.process,
        likelihood=likelihood.copy(),
        name="static_binary_channel",
    )
    return replace(
        scenario,
        scenario_id="reference_static_e_process",
        model=model,
        process=process,
        setting=replace(
            scenario.setting,
            correction="none",
            stopping_policy="e_process",
            formal_validity=(
                "conditional_on_declared_e_factor_supermartingale_assumptions"
            ),
            evidence_assumptions=(
                "fixed_truth",
                "conditional_e_factor_mean_at_most_one",
                "nonnegative_e_process",
                "predictable_selection",
            ),
        ),
        description="Static binary action-conditioned channel with state-invariant truth rates for optional-stopping validation.",
        claim_role="synthetic_formal_conditional_evidence",
        metadata={**dict(scenario.metadata or {}), "state_invariant_emissions": True},
    )


def make_static_binary_split_confirmation_scenario(
    *, reliability: float = 0.8, horizon: int = 12, alpha: float = 0.05
) -> PowerScenario:
    """Build the separately declared static split-confirmation channel.

    The process is the same state-invariant binary channel used to make the
    likelihood-ratio e-process assumptions inspectable, but the inferential
    contract here is deliberately different: selection uses visible history,
    confirmation uses an independent child stream, and the fixed-horizon
    confirmation p-value is tested at the scenario-declared alpha.  Keeping
    this scenario distinct prevents a split-confirmation result from being
    misread as another action-policy operating cell or as an e-process claim.
    """
    scenario = make_static_binary_e_process_scenario(
        reliability=reliability, horizon=horizon, alpha=alpha
    )
    assumptions = (
        "fixed_truth",
        "independent_selection_and_confirmation_streams",
        "selection_uses_visible_filtration_only",
        "conditional_superuniform_confirmation_pvalue_under_null",
    )
    return replace(
        scenario,
        scenario_id="reference_static_split_confirmation",
        setting=replace(
            scenario.setting,
            stopping_policy="adaptive_target",
            formal_validity="conditional_on_declared_split_confirmation_assumptions",
            evidence_assumptions=assumptions,
        ),
        description=(
            "Static binary channel for independent split-stream target selection "
            "and fixed-horizon confirmation at the declared alpha."
        ),
        claim_role="synthetic_formal_conditional_evidence",
        metadata={
            **dict(scenario.metadata or {}),
            "split_confirmation_channel": True,
            "state_invariant_emissions": True,
        },
    )


def make_static_null_binary_e_process_scenario(
    *, reliability: float = 0.8, horizon: int = 12, alpha: float = 0.05
) -> PowerScenario:
    """Build an all-null static channel for split-confirmation diagnostics.

    This differs from :func:`make_reference_null_action_scenario`: both the
    agent-side transition model and the evaluator-side process are rebuilt to
    preserve the all-null fixed-truth contract.  That alignment matters when a
    split-confirmation calibration diagnostic is labelled formal conditional
    evidence.  It remains a finite diagnostic for this declared channel, not a
    general selective-inference result.
    """
    scenario = make_static_binary_e_process_scenario(
        reliability=reliability, horizon=horizon, alpha=alpha
    )
    state_target = np.asarray(scenario.model.state_target, dtype=int)
    state_context = np.asarray(scenario.model.state_context, dtype=int)
    state_truth = np.asarray(scenario.model.state_truth, dtype=int)
    action_target = np.asarray(scenario.model.action_target, dtype=int)
    switch_actions = {
        index
        for index, name in enumerate(scenario.model.action_names)
        if name == "switch_context"
    }
    transition = np.zeros_like(scenario.model.transition)
    for action, target in enumerate(action_target):
        for state, (current_target, context) in enumerate(
            zip(state_target, state_context)
        ):
            next_target = int(current_target) if int(target) < 0 else int(target)
            next_context = 1 - int(context) if action in switch_actions else int(context)
            next_state = np.flatnonzero(
                (state_target == next_target)
                & (state_context == next_context)
                & (state_truth == 0)
            )
            if next_state.size != 1:
                raise RuntimeError("static null scenario requires one null state per target/context")
            transition[action, int(next_state[0]), state] = 1.0
    initial = np.zeros_like(scenario.process.initial_state)
    null_context_states = (state_truth == 0) & (state_context == 0)
    initial[null_context_states] = 1.0 / float(np.count_nonzero(null_context_states))
    model = replace(scenario.model, transition=transition)
    process = replace(
        scenario.process,
        initial_state=initial,
        transition=transition.copy(),
        target_truth=np.zeros(scenario.setting.family_size, dtype=bool),
        name="static_all_null_binary_channel",
    )
    return replace(
        scenario,
        scenario_id="reference_static_all_null_e_process",
        model=model,
        process=process,
        description="All-null static binary channel for stratified split-stream calibration diagnostics.",
        claim_role="synthetic_calibration_diagnostic",
        metadata={
            **dict(scenario.metadata or {}),
            "all_null_process": True,
            "state_invariant_emissions": True,
            "split_confirmation_calibration": True,
        },
    )


def make_reference_null_action_scenario(
    *, reliability: float = 0.8, horizon: int = 12, alpha: float = 0.05
) -> PowerScenario:
    """Build the all-null process used by selected-target calibration.

    The agent-side model is unchanged and still contains both truth states;
    only the data-generating process is made all-null.  This prevents the
    policy from receiving truth while making the calibration estimand explicit.
    Results from this helper are diagnostics for the declared process, not
    post-selection validity proofs.
    """
    scenario = make_reference_action_scenario(
        reliability=reliability, horizon=horizon, alpha=alpha
    )
    initial = np.zeros_like(scenario.process.initial_state)
    null_context_states = (
        (scenario.model.state_truth == 0) & (scenario.model.state_context == 0)
    )
    initial[null_context_states] = 1.0 / float(np.count_nonzero(null_context_states))
    process = replace(
        scenario.process,
        initial_state=initial,
        target_truth=np.zeros(scenario.setting.family_size, dtype=bool),
        name="all_null_action_environment",
    )
    return replace(
        scenario,
        scenario_id="reference_all_null_action_loop",
        process=process,
        description="All-null action-conditioned process for selected-target p-value calibration diagnostics.",
        claim_role="synthetic_calibration_diagnostic",
        metadata={**dict(scenario.metadata or {}), "all_null_process": True},
    )


__all__ = [
    "GenerativeModelSpec", "GenerativeProcessSpec", "SettingSpec", "PowerScenario",
    "make_reference_action_scenario", "make_static_binary_e_process_scenario",
    "make_static_binary_split_confirmation_scenario",
    "make_static_null_binary_e_process_scenario",
    "make_reference_null_action_scenario",
]
