"""Selection/confirmation contracts for adaptive target testing.

The central validity boundary in an action-controlled experiment is the
filtration: a policy may use the selection stream, but it must not see the
independent confirmation stream.  This module keeps that boundary explicit and
serializable.  The split-stream result is a conditional operating
characteristic; it becomes a formal conditional evidence object only when the
declared static-channel and conditional-superuniformity assumptions hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from typing import Any, Literal

import numpy as np

from .action_loop import ActionLoopTrace, run_action_loop
from .scenario import PowerScenario
from .sequential import fixed_horizon_binomial_pvalue

SelectionStrategy = Literal["minimum_p", "least_informed", "round_robin"]


def _hash_mapping(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SelectionPolicySpec:
    """Pre-registered policy parameters for a selection stream."""

    name: str = "adaptive_target"
    strategy: SelectionStrategy = "minimum_p"
    max_steps: int = 12
    confirmation_steps: int = 12
    evidence_observable: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("selection policy name must be non-empty")
        if self.strategy not in {"minimum_p", "least_informed", "round_robin"}:
            raise ValueError("unsupported selection strategy")
        if not self.evidence_observable and self.strategy == "minimum_p":
            raise ValueError("minimum_p requires evidence_observable=True")
        for name, value in (("max_steps", self.max_steps), ("confirmation_steps", self.confirmation_steps)):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.evidence_observable, bool):
            raise ValueError("evidence_observable must be boolean")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "max_steps": int(self.max_steps),
            "confirmation_steps": int(self.confirmation_steps),
            "evidence_observable": self.evidence_observable,
        }

    @property
    def configuration_hash(self) -> str:
        return _hash_mapping(self.to_mapping())


@dataclass(frozen=True)
class FiltrationEvent:
    """One information state available before a target is selected."""

    time: int
    target_candidates: tuple[int, ...]
    visible_observations: tuple[int, ...]
    visible_actions: tuple[str, ...]
    selected_target: int | None
    hidden_state_exposed: bool = False
    visible_evidence: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.time, bool) or int(self.time) != self.time or self.time < 0:
            raise ValueError("filtration time must be a non-negative integer")
        candidates = tuple(int(value) for value in self.target_candidates)
        if len(set(candidates)) != len(candidates) or any(value < 0 for value in candidates):
            raise ValueError("target_candidates must contain unique non-negative indices")
        observations = tuple(int(value) for value in self.visible_observations)
        if any(value not in {0, 1} for value in observations):
            raise ValueError("visible_observations must be binary")
        actions = tuple(str(value) for value in self.visible_actions)
        if any(not value for value in actions):
            raise ValueError("visible_actions must be non-empty strings")
        if self.selected_target is not None and int(self.selected_target) not in candidates:
            raise ValueError("selected_target must be one of target_candidates")
        if not isinstance(self.hidden_state_exposed, bool):
            raise ValueError("hidden_state_exposed must be boolean")
        evidence = tuple(float(value) for value in self.visible_evidence)
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in evidence):
            raise ValueError("visible_evidence must contain finite values in [0, 1]")
        object.__setattr__(self, "target_candidates", candidates)
        object.__setattr__(self, "visible_observations", observations)
        object.__setattr__(self, "visible_actions", actions)
        object.__setattr__(self, "selected_target", None if self.selected_target is None else int(self.selected_target))
        object.__setattr__(self, "visible_evidence", evidence)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "time": int(self.time),
            "target_candidates": list(self.target_candidates),
            "visible_observations": list(self.visible_observations),
            "visible_actions": list(self.visible_actions),
            "selected_target": self.selected_target,
            "hidden_state_exposed": self.hidden_state_exposed,
            "visible_evidence": list(self.visible_evidence),
        }


@dataclass(frozen=True)
class SelectionTrace:
    """Selection-stream trace with a separately auditable visible filtration."""

    scenario_id: str
    scenario_hash: str
    policy: SelectionPolicySpec
    seed: int
    events: tuple[FiltrationEvent, ...]
    selected_target: int
    selected_target_evidence: float
    visible_only: bool
    stream_hash: str

    def __post_init__(self) -> None:
        if not self.scenario_id or len(self.scenario_hash) != 64:
            raise ValueError("selection trace requires a scenario id and SHA-256 hash")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise ValueError("selection seed must be a non-negative integer")
        if not self.events:
            raise ValueError("selection trace must contain at least one filtration event")
        if self.selected_target < 0:
            raise ValueError("selected_target must be non-negative")
        if not 0.0 <= self.selected_target_evidence <= 1.0:
            raise ValueError("selected_target_evidence must lie in [0, 1]")
        if not isinstance(self.visible_only, bool) or not self.visible_only:
            raise ValueError("selection traces must be marked visible_only")
        if len(self.stream_hash) != 64:
            raise ValueError("stream_hash must be a SHA-256 hash")
        if any(event.hidden_state_exposed for event in self.events):
            raise ValueError("selection trace cannot contain hidden-state exposure")
        times = [event.time for event in self.events]
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("selection filtration times must be strictly increasing")
        for event in self.events:
            if len(event.visible_evidence) > len(event.visible_actions):
                raise ValueError("visible evidence cannot contain information from the current action")
            if not self.policy.evidence_observable and event.visible_evidence:
                raise ValueError("evidence-blind selection cannot expose visible evidence")
        if self.selected_target not in self.events[-1].target_candidates:
            raise ValueError("selected_target must be in the final filtration candidate set")
        if self.stream_hash != self.expected_stream_hash:
            raise ValueError("selection stream hash does not match the visible payload")

    def stream_payload(self) -> dict[str, Any]:
        """Return the canonical visible payload covered by ``stream_hash``."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "policy": self.policy.to_mapping(),
            "seed": int(self.seed),
            "events": [event.to_mapping() for event in self.events],
            "selected_target": int(self.selected_target),
            "selected_target_evidence": float(self.selected_target_evidence),
            "visible_only": self.visible_only,
        }

    @property
    def expected_stream_hash(self) -> str:
        return _hash_mapping(self.stream_payload())

    def to_mapping(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "policy": self.policy.to_mapping(),
            "seed": int(self.seed),
            "events": [event.to_mapping() for event in self.events],
            "selected_target": int(self.selected_target),
            "selected_target_evidence": float(self.selected_target_evidence),
            "visible_only": self.visible_only,
            "stream_hash": self.stream_hash,
        }

    @property
    def configuration_hash(self) -> str:
        """Hash the serialized visible selection contract."""
        return _hash_mapping(self.to_mapping())


@dataclass(frozen=True)
class SplitConfirmationResult:
    """One split-stream selection/confirmation result."""

    scenario_id: str
    scenario_hash: str
    selection_seed: int
    confirmation_seed: int
    selection: SelectionTrace
    confirmation_observations: tuple[int, ...]
    selected_target_truth: bool
    confirmation_p_value: float
    rejected: bool
    validity: str
    validity_assumptions: tuple[str, ...]
    selection_stream_hash: str
    confirmation_stream_hash: str

    def __post_init__(self) -> None:
        if self.selection.scenario_id != self.scenario_id or self.selection.scenario_hash != self.scenario_hash:
            raise ValueError("selection trace does not match split result scenario")
        if self.selection.seed != self.selection_seed:
            raise ValueError("selection seed does not match selection trace")
        if self.selection.stream_hash != self.selection_stream_hash:
            raise ValueError("selection stream hash does not match selection trace")
        if self.selection_seed == self.confirmation_seed:
            raise ValueError("selection and confirmation streams require distinct seeds")
        observations = tuple(int(value) for value in self.confirmation_observations)
        if not observations or any(value not in {0, 1} for value in observations):
            raise ValueError("confirmation_observations must be non-empty and binary")
        if not 0.0 <= self.confirmation_p_value <= 1.0:
            raise ValueError("confirmation_p_value must lie in [0, 1]")
        if not self.validity or not self.validity_assumptions:
            raise ValueError("split confirmation requires a validity label and assumptions")
        if not isinstance(self.selected_target_truth, (bool, np.bool_)):
            raise ValueError("selected_target_truth must be boolean")
        for value in (self.selection_stream_hash, self.confirmation_stream_hash):
            if len(value) != 64:
                raise ValueError("stream hashes must be SHA-256 digests")
        object.__setattr__(self, "confirmation_observations", observations)
        object.__setattr__(self, "validity_assumptions", tuple(str(value) for value in self.validity_assumptions))
        if self.confirmation_stream_hash != self.expected_confirmation_hash:
            raise ValueError("confirmation stream hash does not match the confirmation payload")

    def confirmation_payload(self) -> dict[str, Any]:
        """Return the canonical confirmation payload covered by its hash."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "target": int(self.selection.selected_target),
            "seed": int(self.confirmation_seed),
            "observations": list(self.confirmation_observations),
            "selected_target_truth": bool(self.selected_target_truth),
            "confirmation_p_value": float(self.confirmation_p_value),
            "rejected": bool(self.rejected),
            "validity": self.validity,
            "validity_assumptions": list(self.validity_assumptions),
        }

    @property
    def expected_confirmation_hash(self) -> str:
        return _hash_mapping(self.confirmation_payload())

    def to_mapping(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "selection_seed": int(self.selection_seed),
            "confirmation_seed": int(self.confirmation_seed),
            "selection": self.selection.to_mapping(),
            "confirmation_observations": list(self.confirmation_observations),
            "selected_target_truth": bool(self.selected_target_truth),
            "confirmation_p_value": float(self.confirmation_p_value),
            "rejected": bool(self.rejected),
            "validity": self.validity,
            "validity_assumptions": list(self.validity_assumptions),
            "selection_stream_hash": self.selection_stream_hash,
            "confirmation_stream_hash": self.confirmation_stream_hash,
        }


def _selection_events(
    trace: ActionLoopTrace, family_size: int, evidence_observable: bool
) -> tuple[FiltrationEvent, ...]:
    events: list[FiltrationEvent] = []
    observations: list[int] = []
    actions: list[str] = []
    evidences: list[float] = []
    candidates = tuple(range(family_size))
    for step in trace.steps:
        event = FiltrationEvent(
            time=int(step.time - 1),
            target_candidates=candidates,
            visible_observations=tuple(observations),
            visible_actions=tuple(actions),
            selected_target=None,
            visible_evidence=tuple(evidences) if evidence_observable else (),
        )
        events.append(event)
        if step.observation is not None:
            observations.append(int(step.observation))
        actions.append(step.action)
        if evidence_observable:
            evidences.append(float(step.p_value))
    if not events:
        raise ValueError("selection policy produced no visible events")
    return tuple(events)


def _choose_target(
    trace: ActionLoopTrace,
    strategy: SelectionStrategy,
    family_size: int,
    evidence_observable: bool,
) -> tuple[int, float]:
    evidence = [
        (float(step.p_value), int(step.target))
        for step in trace.steps
        if 0 <= int(step.target) < family_size
    ]
    if not evidence:
        return 0, 1.0
    if not evidence_observable:
        if strategy == "round_robin":
            return int(evidence[-1][1]), 1.0
        counts = np.bincount(np.asarray([target for _, target in evidence]), minlength=family_size)
        return int(np.argmin(counts)), 1.0
    if strategy == "minimum_p":
        value, target = min(evidence, key=lambda item: (item[0], item[1]))
        return target, value
    if strategy == "round_robin":
        target = int(evidence[-1][1])
        return target, float(evidence[-1][0])
    counts = np.bincount(np.asarray([target for _, target in evidence]), minlength=family_size)
    target = int(np.argmin(counts))
    value = min((value for value, candidate in evidence if candidate == target), default=1.0)
    return target, float(value)


def _target_rates(scenario: PowerScenario, action: int, target: int) -> tuple[float, float]:
    truth = np.asarray(scenario.model.state_truth, dtype=bool)
    target_labels = np.asarray(scenario.model.state_target, dtype=int)
    rates = np.asarray(scenario.process.likelihood[action, 1], dtype=float)
    result: list[float] = []
    for truth_value in (False, True):
        mask = (target_labels == target) & (truth == truth_value)
        if not np.any(mask):
            raise ValueError("confirmation target lacks null or alternative states")
        result.append(float(np.mean(rates[mask])))
    q0, q1 = result
    if not (0.0 < q0 < 1.0 and 0.0 < q1 < 1.0 and q0 != q1):
        raise ValueError("confirmation channel must have non-degenerate distinct rates")
    return q0, q1


def _confirmation_stream(scenario: PowerScenario, target: int, seed: int, steps: int) -> tuple[int, ...]:
    actions = np.flatnonzero(np.asarray(scenario.model.action_target, dtype=int) == int(target))
    if actions.size == 0:
        raise ValueError("scenario has no action for selected target")
    action = int(actions[0])
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
    target_truth = bool(scenario.process.target_truth[int(target)])
    target_state_mask = (
        (scenario.model.state_target == int(target))
        & (scenario.model.state_truth == int(target_truth))
    )
    initial = np.asarray(scenario.process.initial_state, dtype=float) * target_state_mask
    if float(np.sum(initial)) <= 0.0:
        raise ValueError("confirmation stream has no initial mass for the selected target")
    initial /= np.sum(initial)
    state = int(rng.choice(scenario.model.n_states, p=initial))
    observations: list[int] = []
    for time in range(int(steps)):
        transition = scenario.process.transition_at(time)
        likelihood = scenario.process.likelihood_at(time)
        state = int(rng.choice(scenario.model.n_states, p=transition[action, :, state]))
        if rng.random() < scenario.process.missing_probability_at(time, action, state):
            continue
        observations.append(int(rng.choice(scenario.model.n_observations, p=likelihood[action, :, state])))
    if not observations:
        raise ValueError("confirmation stream produced no observed outcomes")
    return tuple(observations)


def run_split_stream_confirmation(
    scenario: PowerScenario,
    policy: SelectionPolicySpec | None = None,
    *,
    selection_seed: int = 0,
    confirmation_seed: int = 1,
    alpha: float | None = None,
) -> SplitConfirmationResult:
    """Select a target and confirm it on an independent child stream.

    The rejection threshold is the alpha recorded in ``scenario.setting`` and
    therefore covered by ``scenario_hash``.  The optional ``alpha`` keyword
    is accepted only when it is exactly that declared value; callers cannot
    use an unrecorded override to change the confirmation decision.
    """
    policy = policy or SelectionPolicySpec(max_steps=scenario.setting.max_steps, confirmation_steps=scenario.setting.max_steps)
    if isinstance(selection_seed, bool) or isinstance(confirmation_seed, bool) or selection_seed < 0 or confirmation_seed < 0:
        raise ValueError("stream seeds must be non-negative integers")
    if alpha is not None:
        if isinstance(alpha, bool):
            raise ValueError("alpha must lie in (0, 1)")
        requested_alpha = float(alpha)
        if not np.isfinite(requested_alpha) or not 0.0 < requested_alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if requested_alpha != scenario.setting.alpha:
            raise ValueError(
                "alpha override must exactly match the declared scenario alpha"
            )
    effective_alpha = scenario.setting.alpha
    trace = run_action_loop(
        scenario,
        "adaptive_target",
        seed=int(selection_seed),
        max_steps=int(policy.max_steps),
    )
    selected_target, selected_evidence = _choose_target(
        trace, policy.strategy, scenario.setting.family_size, policy.evidence_observable
    )
    events = _selection_events(trace, scenario.setting.family_size, policy.evidence_observable)
    selection_payload = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "policy": policy.to_mapping(),
        "seed": int(selection_seed),
        "events": [event.to_mapping() for event in events],
        "selected_target": selected_target,
        "selected_target_evidence": selected_evidence,
        "visible_only": True,
    }
    selection_trace = SelectionTrace(
        scenario.scenario_id,
        scenario.configuration_hash,
        policy,
        int(selection_seed),
        events,
        selected_target,
        selected_evidence,
        True,
        _hash_mapping(selection_payload),
    )
    observations = _confirmation_stream(scenario, selected_target, int(confirmation_seed), policy.confirmation_steps)
    actions = np.flatnonzero(np.asarray(scenario.model.action_target, dtype=int) == selected_target)
    q0, q1 = _target_rates(scenario, int(actions[0]), selected_target)
    p_value = fixed_horizon_binomial_pvalue(observations, q0, q1)
    truth = bool(np.asarray(scenario.process.target_truth, dtype=bool)[selected_target])
    contract = (
        np.allclose(scenario.model.likelihood, scenario.process.likelihood)
        and np.allclose(scenario.model.transition, scenario.process.transition)
        and scenario.process.missing_probability == 0.0
        and scenario.process.transition_schedule is None
        and scenario.process.likelihood_schedule is None
        and scenario.process.missing_probability_schedule is None
        and scenario.process.missingness_mode == "mcar"
        and scenario.process.action_missing_probability is None
        and scenario.process.state_missing_probability is None
        and scenario.truth_consistent
    )
    truth_labels = np.asarray(scenario.model.state_truth, dtype=int)
    if contract:
        for action_index in range(scenario.model.n_actions):
            rates = scenario.process.likelihood[action_index, 1]
            for truth_value in (0, 1):
                mask = (np.asarray(scenario.model.state_target) == selected_target) & (truth_labels == truth_value)
                if np.any(mask) and float(np.ptp(rates[mask])) > 1e-12:
                    contract = False
                    break
            if not contract:
                break
    validity = "formal_conditional_split_confirmation" if contract else "conditional_split_confirmation_diagnostic"
    assumptions = (
        "fixed_truth",
        "independent_selection_and_confirmation_streams",
        "selection_uses_visible_filtration_only",
        "conditional_superuniform_confirmation_pvalue_under_null",
    )
    confirmation_payload = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "target": selected_target,
        "seed": int(confirmation_seed),
        "observations": list(observations),
        "selected_target_truth": truth,
        "confirmation_p_value": float(p_value),
        "rejected": bool(p_value <= effective_alpha),
        "validity": validity,
        "validity_assumptions": list(assumptions),
    }
    return SplitConfirmationResult(
        scenario.scenario_id,
        scenario.configuration_hash,
        int(selection_seed),
        int(confirmation_seed),
        selection_trace,
        observations,
        truth,
        float(p_value),
        bool(p_value <= effective_alpha),
        validity,
        assumptions,
        selection_trace.stream_hash,
        _hash_mapping(confirmation_payload),
    )


def simulate_split_stream_confirmation(
    scenario: PowerScenario,
    n_reps: int,
    *,
    seed: int = 0,
    workers: int = 1,
    policy: SelectionPolicySpec | None = None,
) -> dict[str, Any]:
    r"""Estimate explicit split-stream rejection estimands across replications.

    ``power`` is the conventional conditional quantity
    :math:`P(R=1 \mid T=1)`, where ``T`` says that the selected target is
    truly non-null and ``R`` says that its independent confirmation stream is
    rejected.  The result records the selected-true rate and the unconditional
    joint rate :math:`P(R=1, T=1)` separately, so adaptive target prevalence
    cannot be mistaken for conditional confirmation performance.  If no true
    target is selected in a finite run, ``power`` is ``None`` and
    ``power_defined`` is false rather than silently reporting a zero rate.
    """
    if isinstance(n_reps, bool) or int(n_reps) != n_reps or n_reps < 1:
        raise ValueError("n_reps must be a positive integer")
    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    children = np.random.SeedSequence(int(seed)).spawn(int(n_reps) * 2)
    seeds = [
        (
            int(children[2 * index].generate_state(1, dtype=np.uint32)[0]),
            int(children[2 * index + 1].generate_state(1, dtype=np.uint32)[0]),
        )
        for index in range(int(n_reps))
    ]
    def run(pair: tuple[int, int]) -> SplitConfirmationResult:
        return run_split_stream_confirmation(
            scenario,
            policy,
            selection_seed=pair[0],
            confirmation_seed=pair[1],
        )
    if int(workers) == 1:
        results = [run(pair) for pair in seeds]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            results = list(executor.map(run, seeds))
    rejected = np.asarray([result.rejected for result in results], dtype=bool)
    selected_true = np.asarray(
        [result.selected_target_truth for result in results], dtype=bool
    )
    true_rejected = np.logical_and(rejected, selected_true)
    false_rejected = np.logical_and(rejected, np.logical_not(selected_true))
    true_target_selection_count = int(np.sum(selected_true))
    true_target_joint_rejection_count = int(np.sum(true_rejected))
    false_rejection_count = int(np.sum(false_rejected))
    rejection_count = int(np.sum(rejected))
    power_defined = true_target_selection_count > 0
    return {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "n_reps": int(n_reps),
        "seed": int(seed),
        "selection_streams": int(n_reps),
        "confirmation_streams": int(n_reps),
        "rejection_count": rejection_count,
        "rejection_rate": float(rejection_count / int(n_reps)),
        "true_target_selection_count": true_target_selection_count,
        "true_target_selection_rate": float(true_target_selection_count / int(n_reps)),
        "true_target_joint_rejection_count": true_target_joint_rejection_count,
        "true_target_joint_rejection_rate": float(
            true_target_joint_rejection_count / int(n_reps)
        ),
        "power": (
            float(true_target_joint_rejection_count / true_target_selection_count)
            if power_defined
            else None
        ),
        "power_defined": power_defined,
        "power_estimand": "conditional_confirmation_rejection_given_selected_target_true",
        "false_rejection_count": false_rejection_count,
        "fwer": float(false_rejection_count / int(n_reps)),
        "fwer_estimand": "single_confirmed_target_false_rejection_rate",
        "validity": results[0].validity,
        "assumptions": list(results[0].validity_assumptions),
        "representative": results[0].to_mapping(),
    }


__all__ = [
    "SelectionPolicySpec", "FiltrationEvent", "SelectionTrace", "SplitConfirmationResult",
    "run_split_stream_confirmation", "simulate_split_stream_confirmation",
]
