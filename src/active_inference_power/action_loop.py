"""Action-in-the-loop simulation for conditional active-inference power.

The environment is sampled from :class:`GenerativeProcessSpec`; the agent only
sees actions and observations.  Hidden states are retained in a separate
evaluation field so diagnostic operating characteristics cannot accidentally
be mistaken for information available to the policy.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, Sequence, cast

import numpy as np

from .action_statistics import summarize_action_traces
from .scenario import PowerScenario
from .pymdp_adapter import make_agent

ActionPolicy = Literal[
    "fixed_horizon",
    "posterior_threshold",
    "e_process",
    "learned",
    "information_gain",
    "cost_aware",
    "adaptive_target",
    "posterior_sampling",
]

_ACTION_POLICIES = frozenset({
    "fixed_horizon",
    "posterior_threshold",
    "e_process",
    "learned",
    "information_gain",
    "cost_aware",
    "adaptive_target",
    "posterior_sampling",
})
_TERMINAL_POSTERIOR_POLICIES = frozenset({
    "fixed_horizon",
    "information_gain",
    "cost_aware",
    "adaptive_target",
    "posterior_sampling",
    "learned",
})
_PER_TARGET_FAMILY_SCOPE = "per_target_no_family_error_claim"
_DIAGNOSTIC_FAMILY_SCOPE = "no_family_error_control_claim"
_PER_TARGET_OPTIONAL_STOPPING_BOUNDARY = "conditional_per_target_optional_stopping"

_PYMDP_AGENT_CACHE: dict[tuple[str, int], Any] = {}


@dataclass(frozen=True)
class ActionLoopStep:
    """One environment transition and agent update."""

    time: int
    action: str
    target: int
    sensor: int
    observation: int | None
    posterior_signal: float
    target_beliefs: tuple[float, ...]
    p_value: float
    e_value: float
    e_process_value: float
    target_e_process_values: tuple[float, ...]
    cost: float
    stopped: bool
    decision: bool
    evaluation_state_before: int
    evaluation_state_after: int

    def visible_mapping(self) -> dict[str, Any]:
        """Return only the action/observation trace visible to an agent."""
        return {
            "time": self.time,
            "action": self.action,
            "target": self.target,
            "sensor": self.sensor,
            "observation": self.observation,
            "posterior_signal": self.posterior_signal,
            "target_beliefs": list(self.target_beliefs),
            "p_value": self.p_value,
            "e_value": self.e_value,
            "e_process_value": self.e_process_value,
            "target_e_process_values": list(self.target_e_process_values),
            "evidence_kind": "likelihood_ratio_e_process",
            "cost": self.cost,
            "stopped": self.stopped,
            "decision": self.decision,
        }

    def to_mapping(self) -> dict[str, Any]:
        result = self.visible_mapping()
        result.update({
            "evaluation_state_before": self.evaluation_state_before,
            "evaluation_state_after": self.evaluation_state_after,
        })
        return result


@dataclass(frozen=True)
class ActionLoopTrace:
    """Complete policy-controlled trace for one replication."""

    scenario_id: str
    policy: str
    seed: int
    steps: tuple[ActionLoopStep, ...]
    stopping_time: int
    decision: bool
    total_cost: float
    terminal_e_process_value: float
    target_e_process_values: tuple[float, ...]
    rejected_targets: tuple[int, ...]
    optional_stopping_valid: bool
    validity_boundary: str
    validity_assumptions: tuple[str, ...]
    pymdp_oracle_max_abs_error: float
    declared_correction: str = ""
    executed_correction: str = ""
    declared_stopping_policy: str = ""
    executed_stopping_policy: str = ""
    policy_override: bool = False
    family_scope: str = ""

    @property
    def observations(self) -> np.ndarray:
        return np.asarray([step.observation for step in self.steps if step.observation is not None], dtype=int)

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(step.action for step in self.steps)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "policy": self.policy,
            "seed": self.seed,
            "steps": [step.to_mapping() for step in self.steps],
            "visible_steps": [step.visible_mapping() for step in self.steps],
            "stopping_time": self.stopping_time,
            "decision": self.decision,
            "total_cost": self.total_cost,
            "terminal_e_process_value": self.terminal_e_process_value,
            "target_e_process_values": list(self.target_e_process_values),
            "evidence_kind": "likelihood_ratio_e_process",
            "rejected_targets": list(self.rejected_targets),
            "optional_stopping_valid": self.optional_stopping_valid,
            "validity_boundary": self.validity_boundary,
            "validity_assumptions": list(self.validity_assumptions),
            "pymdp_oracle_max_abs_error": self.pymdp_oracle_max_abs_error,
            "declared_correction": self.declared_correction,
            "executed_correction": self.executed_correction,
            "declared_stopping_policy": self.declared_stopping_policy,
            "executed_stopping_policy": self.executed_stopping_policy,
            "policy_override": self.policy_override,
            "family_scope": self.family_scope,
        }


@dataclass(frozen=True)
class ActionLoopSummary:
    """Finite-simulation operating characteristics for one policy."""

    scenario_id: str
    policy: str
    n_reps: int
    decision_rate: float
    power: float
    fdr: float
    fwer: float
    mean_stopping_time: float
    mean_observations: float
    mean_cost: float
    decision_interval: tuple[float, float]
    power_interval: tuple[float, float]
    fdr_interval: tuple[float, float]
    fwer_interval: tuple[float, float]
    validity: str
    mean_rejections: float = 0.0
    mean_true_rejections: float = 0.0
    mean_false_rejections: float = 0.0
    any_rejection_rate: float = 0.0
    any_true_rejection_rate: float = 0.0
    power_per_cost: float = 0.0
    target_rejection_rates: tuple[float, ...] = ()
    target_true_rejection_rates: tuple[float | None, ...] = ()
    target_false_rejection_rates: tuple[float | None, ...] = ()
    cost_interval: tuple[float, float] = (0.0, 0.0)
    stopping_interval: tuple[float, float] = (0.0, 0.0)
    observation_interval: tuple[float, float] = (0.0, 0.0)
    power_per_cost_interval: tuple[float, float] = (0.0, 0.0)
    validity_assumptions: tuple[str, ...] = ()
    declared_correction: str = ""
    executed_correction: str = ""
    declared_stopping_policy: str = ""
    executed_stopping_policy: str = ""
    policy_override: bool = False
    family_scope: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "policy": self.policy,
            "n_reps": self.n_reps,
            "decision_rate": self.decision_rate,
            "power": self.power,
            "fdr": self.fdr,
            "fwer": self.fwer,
            "mean_stopping_time": self.mean_stopping_time,
            "mean_observations": self.mean_observations,
            "mean_cost": self.mean_cost,
            "decision_interval": list(self.decision_interval),
            "power_interval": list(self.power_interval),
            "fdr_interval": list(self.fdr_interval),
            "fwer_interval": list(self.fwer_interval),
            "validity": self.validity,
            "mean_rejections": self.mean_rejections,
            "mean_true_rejections": self.mean_true_rejections,
            "mean_false_rejections": self.mean_false_rejections,
            "any_rejection_rate": self.any_rejection_rate,
            "any_true_rejection_rate": self.any_true_rejection_rate,
            "power_per_cost": self.power_per_cost,
            "target_rejection_rates": list(self.target_rejection_rates),
            "target_true_rejection_rates": list(self.target_true_rejection_rates),
            "target_false_rejection_rates": list(self.target_false_rejection_rates),
            "cost_interval": list(self.cost_interval),
            "stopping_interval": list(self.stopping_interval),
            "observation_interval": list(self.observation_interval),
            "power_per_cost_interval": list(self.power_per_cost_interval),
            "validity_assumptions": list(self.validity_assumptions),
            "declared_correction": self.declared_correction,
            "executed_correction": self.executed_correction,
            "declared_stopping_policy": self.declared_stopping_policy,
            "executed_stopping_policy": self.executed_stopping_policy,
            "policy_override": self.policy_override,
            "family_scope": self.family_scope,
        }


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("posterior normalization failed")
    return values / total


def _resolve_execution_policy(
    scenario: PowerScenario, policy: ActionPolicy | None
) -> tuple[ActionPolicy, bool]:
    """Resolve the declared policy or label an explicit diagnostic override."""
    if policy is None:
        return cast(ActionPolicy, scenario.setting.stopping_policy), False
    if policy not in _ACTION_POLICIES:
        raise ValueError(f"unknown action policy: {policy!r}")
    return policy, policy != scenario.setting.stopping_policy


def _target_beliefs(posterior: np.ndarray, scenario: PowerScenario) -> tuple[float, ...]:
    beliefs: list[float] = []
    for target in range(len(scenario.model.hypothesis_ids)):
        mask = scenario.model.state_target == target
        denominator = float(np.sum(posterior[mask]))
        numerator = float(np.sum(posterior[mask & (scenario.model.state_truth == 1)]))
        beliefs.append(0.0 if denominator == 0.0 else numerator / denominator)
    return tuple(beliefs)


def _signal_belief(posterior: np.ndarray, scenario: PowerScenario) -> float:
    return float(np.sum(posterior[scenario.model.state_truth == 1]))


def _pymdp_update(prior: np.ndarray, observation: int, scenario: PowerScenario, action: int) -> np.ndarray:
    """Use the installed pymdp agent for one action-conditioned observation."""
    cache_key = (scenario.configuration_hash, int(action))
    agent = _PYMDP_AGENT_CACHE.get(cache_key)
    if agent is None:
        A = scenario.model.likelihood[action]
        B = np.eye(scenario.model.n_states)[:, :, None]
        C = scenario.model.preferences
        agent = make_agent(
            A=[A],
            B=[B],
            C=[C],
            D=[np.asarray(scenario.model.prior, dtype=float)],
            use_utility=False,
            use_states_info_gain=False,
        )
        _PYMDP_AGENT_CACHE[cache_key] = agent
    D = np.asarray(prior, dtype=float)
    qs = agent.infer_states(
        [np.asarray([int(observation)], dtype=int)],
        empirical_prior=[D[None, :]],
    )
    return _normalize(np.asarray(qs[0], dtype=float)[0, -1, :])


def _expected_information_gain(prior: np.ndarray, scenario: PowerScenario, action: int) -> float:
    current_entropy = -float(np.sum(prior * np.log(np.maximum(prior, 1e-15))))
    predicted = scenario.model.transition[action] @ prior
    expected_entropy = 0.0
    for observation in range(scenario.model.n_observations):
        mass = float(np.dot(scenario.model.likelihood[action, observation], predicted))
        if mass <= 0.0:
            continue
        posterior = _normalize(predicted * scenario.model.likelihood[action, observation])
        expected_entropy -= mass * float(np.sum(posterior * np.log(np.maximum(posterior, 1e-15))))
    return max(0.0, current_entropy - expected_entropy)


def _expected_posterior_entropy(prior: np.ndarray, scenario: PowerScenario, action: int) -> float:
    """Return the expected entropy reduction for one candidate action."""
    current_entropy = -float(np.sum(prior * np.log(np.maximum(prior, 1e-15))))
    return max(0.0, current_entropy - _expected_information_gain(prior, scenario, action))


def _choose_action(
    policy: str,
    posterior: np.ndarray,
    scenario: PowerScenario,
    time: int,
    *,
    total_cost: float = 0.0,
    rng: np.random.Generator | None = None,
    learned_parameters: np.ndarray | None = None,
) -> int:
    candidates = np.flatnonzero(scenario.model.action_target >= 0)
    if candidates.size == 0:
        return 0
    action_costs = scenario.process.action_cost_at(time)
    feasible = candidates[
        total_cost + action_costs[candidates]
        <= scenario.setting.cost_budget + 1e-12
    ]
    if feasible.size == 0:
        raise StopIteration("no target-selecting action fits the remaining cost budget")
    if policy == "fixed_horizon":
        return int(feasible[time % feasible.size])
    if policy in {"adaptive_target", "posterior_sampling"}:
        beliefs = np.asarray(_target_beliefs(posterior, scenario))
        if policy == "adaptive_target":
            target = int(np.argmin(np.abs(beliefs - 0.5)))
        else:
            if rng is None:
                raise ValueError("posterior_sampling requires a random generator")
            sampled_state = int(rng.choice(scenario.model.n_states, p=posterior))
            target = int(scenario.model.state_target[sampled_state])
        target_actions = feasible[scenario.model.action_target[feasible] == target]
        if target_actions.size:
            feasible = target_actions
        else:
            return int(feasible[time % feasible.size])
    if policy == "learned":
        if learned_parameters is None:
            raise ValueError("learned policy requires learned_parameters")
        parameters = np.asarray(learned_parameters, dtype=float)
        if parameters.shape != (3,) or not np.all(np.isfinite(parameters)):
            raise ValueError("learned_parameters must be a finite vector of length three")
        beliefs = np.asarray(_target_beliefs(posterior, scenario))

        def score(action: int) -> float:
            target = int(scenario.model.action_target[action])
            uncertainty = 1.0 if target < 0 else 1.0 - 2.0 * abs(float(beliefs[target]) - 0.5)
            information = _expected_information_gain(posterior, scenario, action)
            return float(
                parameters[0] * information
                + parameters[1] * uncertainty
                - parameters[2] * float(action_costs[action])
            )

        return int(feasible[int(np.argmax([score(int(action)) for action in feasible]))])
    scores = np.asarray([
        (
            _expected_information_gain(posterior, scenario, int(action))
            / max(float(action_costs[action]), 1e-12)
            if policy == "cost_aware"
            else _expected_information_gain(posterior, scenario, int(action))
        )
        for action in feasible
    ])
    return int(feasible[int(np.argmax(scores))])


def _single_step_pvalue(scenario: PowerScenario, action: int, observation: int | None) -> tuple[float, float]:
    if observation is None or scenario.model.action_target[action] < 0:
        return 1.0, 1.0
    target = int(scenario.model.action_target[action])
    null = scenario.model.state_target == target
    null &= scenario.model.state_truth == 0
    alternative = scenario.model.state_target == target
    alternative &= scenario.model.state_truth == 1
    null_prior = scenario.model.prior[null]
    alternative_prior = scenario.model.prior[alternative]
    if float(np.sum(null_prior)) <= 0.0 or float(np.sum(alternative_prior)) <= 0.0:
        raise ValueError("action evidence requires positive prior mass under null and alternative")
    q0 = float(np.dot(scenario.process.likelihood[action, 1, null], null_prior) / np.sum(null_prior))
    q1 = float(np.dot(scenario.process.likelihood[action, 1, alternative], alternative_prior) / np.sum(alternative_prior))
    if not (0.0 < q0 < 1.0 and 0.0 < q1 < 1.0):
        raise ValueError("action evidence requires non-degenerate Bernoulli null and alternative rates")
    if q1 > q0:
        return (q0 if observation == 1 else 1.0), (q1 / q0 if observation == 1 else (1.0 - q1) / (1.0 - q0))
    return (1.0 - q0 if observation == 0 else 1.0), ((1.0 - q1) / (1.0 - q0) if observation == 0 else q1 / q0)


def _e_process_contract_valid(scenario: PowerScenario) -> bool:
    """Check the stronger channel contract needed for action-loop e-values.

    A static truth label alone is insufficient: if the observation rate varies
    with an unobserved context, the fixed Bernoulli factor used by
    ``_single_step_pvalue`` is not conditionally calibrated.  The contract
    therefore requires truth-preserving transitions, matching model/process
    emissions, and state-invariant binary rates within each target/truth cell.
    """
    if scenario.process.transition_schedule is not None or scenario.process.likelihood_schedule is not None:
        return False
    if scenario.process.missingness_mode != "mcar" or scenario.process.missing_probability != 0.0:
        return False
    if not np.allclose(scenario.model.likelihood, scenario.process.likelihood):
        return False
    if not np.allclose(scenario.model.transition, scenario.process.transition):
        return False
    if not scenario.truth_consistent:
        return False
    truth = scenario.model.state_truth
    for action in range(scenario.model.n_actions):
        target = int(scenario.model.action_target[action])
        if target < 0:
            continue
        rates = scenario.process.likelihood[action, 1]
        for truth_value in (0, 1):
            mask = (scenario.model.state_target == target) & (truth == truth_value)
            if not np.any(mask) or float(np.ptp(rates[mask])) > 1e-12:
                return False
    return True


def run_action_loop(
    scenario: PowerScenario,
    policy: ActionPolicy | None = None,
    *,
    seed: int = 0,
    use_pymdp: bool = True,
    learned_parameters: np.ndarray | None = None,
    max_steps: int | None = None,
) -> ActionLoopTrace:
    """Run one policy-controlled environment trace with real pymdp updates.

    Omitting ``policy`` executes the stopping policy that is part of the
    scenario hash.  Passing a different policy is allowed for diagnostic
    policy-surface comparisons, but cannot receive the formal e-process label.
    """
    execution_policy, policy_override = _resolve_execution_policy(scenario, policy)
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    horizon = scenario.setting.max_steps if max_steps is None else int(max_steps)
    if isinstance(max_steps, bool) or (max_steps is not None and (int(max_steps) != max_steps or max_steps < 1)):
        raise ValueError("max_steps must be a positive integer when provided")
    environment_seed, policy_seed = np.random.SeedSequence(int(seed)).spawn(2)
    environment_rng = np.random.default_rng(environment_seed)
    policy_rng = np.random.default_rng(policy_seed)
    current_state = int(environment_rng.choice(scenario.model.n_states, p=scenario.process.initial_state))
    posterior = scenario.model.prior.copy()
    e_process_value = 1.0
    target_e_process_values = np.ones(scenario.setting.family_size, dtype=float)
    total_cost = 0.0
    steps: list[ActionLoopStep] = []
    max_error = 0.0
    decision = False
    for time in range(horizon):
        beliefs = _target_beliefs(posterior, scenario)
        if (
            execution_policy == "posterior_threshold"
            and max(beliefs) >= scenario.setting.posterior_threshold
        ):
            decision = True
            break
        try:
            action = _choose_action(
                execution_policy, posterior, scenario, time, total_cost=total_cost, rng=policy_rng,
                learned_parameters=learned_parameters,
            )
        except StopIteration:
            break
        state_before = current_state
        process_transition = scenario.process.transition_at(time)
        process_likelihood = scenario.process.likelihood_at(time)
        current_state = int(environment_rng.choice(scenario.model.n_states, p=process_transition[action, :, state_before]))
        missing = bool(environment_rng.random() < scenario.process.missing_probability_at(time, action, current_state))
        observation: int | None = None if missing else int(environment_rng.choice(scenario.model.n_observations, p=process_likelihood[action, :, current_state]))
        predicted = scenario.model.transition[action] @ posterior
        oracle = predicted if observation is None else _normalize(predicted * scenario.model.likelihood[action, observation])
        if observation is not None and use_pymdp:
            pymdp_posterior = _pymdp_update(predicted, observation, scenario, action)
            error = float(np.max(np.abs(pymdp_posterior - oracle)))
            max_error = max(max_error, error)
            if error > 1e-8:
                raise RuntimeError(f"pymdp/oracle posterior mismatch: {error}")
        posterior = oracle
        p_value, e_factor = _single_step_pvalue(scenario, action, observation)
        if observation is not None:
            e_process_value *= e_factor
            target = int(scenario.model.action_target[action])
            if target >= 0:
                target_e_process_values[target] *= e_factor
        total_cost += float(scenario.process.action_cost_at(time)[action])
        beliefs = _target_beliefs(posterior, scenario)
        step_decision = (
            execution_policy == "posterior_threshold"
            and max(beliefs) >= scenario.setting.posterior_threshold
        )
        if execution_policy == "e_process":
            step_decision = bool(
                np.max(target_e_process_values) >= 1.0 / scenario.setting.alpha
            )
        decision = decision or step_decision
        if time + 1 >= horizon and execution_policy in _TERMINAL_POSTERIOR_POLICIES:
            decision = max(beliefs) >= scenario.setting.posterior_threshold
        stop = bool(decision or time + 1 >= horizon or total_cost >= scenario.setting.cost_budget)
        steps.append(ActionLoopStep(
            time=time + 1,
            action=scenario.model.action_names[action],
            target=int(scenario.model.action_target[action]),
            sensor=int(scenario.model.action_sensor[action]),
            observation=observation,
            posterior_signal=_signal_belief(posterior, scenario),
            target_beliefs=beliefs,
            p_value=p_value,
            e_value=e_factor,
            e_process_value=e_process_value,
            target_e_process_values=tuple(target_e_process_values.tolist()),
            cost=float(scenario.process.action_cost_at(time)[action]),
            stopped=stop,
            decision=decision,
            evaluation_state_before=state_before,
            evaluation_state_after=current_state,
        ))
        if stop:
            break
    stopping_time = len(steps)
    rejected_targets: tuple[int, ...] = ()
    if steps and execution_policy == "e_process":
        boundary = 1.0 / scenario.setting.alpha
        rejected_targets = tuple(
            index
            for index, value in enumerate(target_e_process_values)
            if value >= boundary
        )
    elif steps and decision:
        rejected_targets = tuple(index for index, belief in enumerate(steps[-1].target_beliefs) if belief >= scenario.setting.posterior_threshold)
    assumptions = tuple(scenario.setting.evidence_assumptions)
    required = {
        "fixed_truth",
        "conditional_e_factor_mean_at_most_one",
        "nonnegative_e_process",
        "predictable_selection",
    }
    valid = (
        not policy_override
        and max_steps is None
        and execution_policy == "e_process"
        and execution_policy == scenario.setting.stopping_policy
        and scenario.setting.correction == "none"
        and required.issubset(assumptions)
        and _e_process_contract_valid(scenario)
    )
    # A maximum over target-specific e-processes is a multi-target decision
    # rule.  With correction="none", the stated e-factor contract supports
    # optional stopping for each target path, not a familywise FWER/FDR claim.
    validity = (
        _PER_TARGET_OPTIONAL_STOPPING_BOUNDARY
        if valid
        else "diagnostic_action_adaptive_evidence"
    )
    family_scope = (
        _PER_TARGET_FAMILY_SCOPE if valid else _DIAGNOSTIC_FAMILY_SCOPE
    )
    return ActionLoopTrace(
        scenario_id=scenario.scenario_id,
        policy=execution_policy,
        seed=int(seed),
        steps=tuple(steps),
        stopping_time=stopping_time,
        decision=decision,
        total_cost=total_cost,
        terminal_e_process_value=e_process_value,
        target_e_process_values=tuple(target_e_process_values.tolist()),
        rejected_targets=rejected_targets,
        optional_stopping_valid=valid,
        validity_boundary=validity,
        validity_assumptions=assumptions,
        pymdp_oracle_max_abs_error=max_error,
        declared_correction=scenario.setting.correction,
        executed_correction="none",
        declared_stopping_policy=scenario.setting.stopping_policy,
        executed_stopping_policy=execution_policy,
        policy_override=policy_override,
        family_scope=family_scope,
    )


def summarize_action_loop(
    scenario: PowerScenario,
    policy: ActionPolicy | None,
    traces: Sequence[ActionLoopTrace],
) -> ActionLoopSummary:
    """Summarize power, family errors, cost, and stopping burden."""
    execution_policy, _ = _resolve_execution_policy(scenario, policy)
    if not traces:
        raise ValueError("traces must be non-empty")
    family_scope = traces[0].family_scope
    if any(trace.family_scope != family_scope for trace in traces):
        raise ValueError("action-loop traces must share one family-scope contract")
    summary = summarize_action_traces(scenario, traces)
    return ActionLoopSummary(
        scenario_id=scenario.scenario_id,
        policy=execution_policy,
        n_reps=len(traces),
        decision_rate=float(summary["decision_rate"]),
        power=float(summary["power"]),
        fdr=float(summary["fdr"]),
        fwer=float(summary["fwer"]),
        mean_stopping_time=float(summary["mean_stopping_time"]),
        mean_observations=float(summary["mean_observations"]),
        mean_cost=float(summary["mean_cost"]),
        decision_interval=tuple(summary["decision_interval"]),
        power_interval=tuple(summary["power_interval"]),
        fdr_interval=tuple(summary["fdr_interval"]),
        fwer_interval=tuple(summary["fwer_interval"]),
        validity=(
            "conditional per-target optional-stopping claim"
            if traces[0].optional_stopping_valid
            else "diagnostic finite-simulation contrast"
        ),
        mean_rejections=float(summary["mean_rejections"]),
        mean_true_rejections=float(summary["mean_true_rejections"]),
        mean_false_rejections=float(summary["mean_false_rejections"]),
        any_rejection_rate=float(summary["any_rejection_rate"]),
        any_true_rejection_rate=float(summary["any_true_rejection_rate"]),
        power_per_cost=float(summary["power_per_cost"]),
        target_rejection_rates=tuple(summary["target_rejection_rates"]),
        target_true_rejection_rates=tuple(summary["target_true_rejection_rates"]),
        target_false_rejection_rates=tuple(summary["target_false_rejection_rates"]),
        cost_interval=tuple(summary["cost_interval"]),
        stopping_interval=tuple(summary["stopping_interval"]),
        observation_interval=tuple(summary["observation_interval"]),
        power_per_cost_interval=tuple(summary["power_per_cost_interval"]),
        validity_assumptions=tuple(summary["validity_assumptions"]),
        declared_correction=scenario.setting.correction,
        executed_correction="none",
        declared_stopping_policy=scenario.setting.stopping_policy,
        executed_stopping_policy=execution_policy,
        policy_override=any(trace.policy_override for trace in traces),
        family_scope=family_scope,
    )


def simulate_action_loop(
    scenario: PowerScenario,
    policy: ActionPolicy | None,
    n_reps: int,
    *,
    seed: int = 0,
    workers: int = 1,
    retain_traces: int = 1,
    learned_parameters: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run stable child-seeded replications; worker count cannot alter results."""
    execution_policy, _ = _resolve_execution_policy(scenario, policy)
    if isinstance(n_reps, bool) or int(n_reps) != n_reps or n_reps < 1:
        raise ValueError("n_reps must be a positive integer")
    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    if isinstance(retain_traces, bool) or int(retain_traces) != retain_traces or retain_traces < 0:
        raise ValueError("retain_traces must be a non-negative integer")
    children = np.random.SeedSequence(int(seed)).spawn(int(n_reps))
    seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    if workers == 1:
        traces = [run_action_loop(scenario, policy, seed=rep_seed, learned_parameters=learned_parameters) for rep_seed in seeds]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            traces = list(executor.map(lambda rep_seed: run_action_loop(scenario, policy, seed=rep_seed, learned_parameters=learned_parameters), seeds))
    summary = summarize_action_loop(scenario, execution_policy, traces)
    return {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "scenario_metadata": dict(scenario.metadata or {}),
        "policy": execution_policy,
        "n_reps": int(n_reps),
        "seed": int(seed),
        "seed_schedule": seeds,
        "summary": summary.to_mapping(),
        "traces": [trace.to_mapping() for trace in traces[: int(retain_traces)]],
        "validity": summary.validity,
        "learned_parameters": None if learned_parameters is None else np.asarray(learned_parameters, dtype=float).tolist(),
    }


def evaluate_power_surface(
    scenarios: Sequence[PowerScenario],
    policies: Sequence[ActionPolicy],
    n_reps: int,
    *,
    seed: int = 0,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Evaluate a deterministic scenario-by-policy power surface."""
    if not scenarios or not policies:
        raise ValueError("scenarios and policies must be non-empty")
    scenario_ids = tuple(str(scenario.scenario_id) for scenario in scenarios)
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenarios must have unique scenario identifiers")
    policy_names = tuple(str(policy) for policy in policies)
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("policies must have unique names")
    streams = np.random.SeedSequence(int(seed)).spawn(len(scenarios) * len(policies))
    results: list[dict[str, Any]] = []
    index = 0
    for scenario in scenarios:
        for policy in policies:
            child_seed = int(streams[index].generate_state(1, dtype=np.uint32)[0])
            results.append(simulate_action_loop(scenario, policy, n_reps, seed=child_seed, workers=workers))
            index += 1
    return results


def run_power_scenario(
    scenario: PowerScenario,
    policy: ActionPolicy | None = None,
    *,
    n_reps: int = 1000,
    seed: int = 0,
    workers: int = 1,
) -> dict[str, Any]:
    """Evaluate one investigator-declared scenario/policy design cell."""
    return simulate_action_loop(scenario, policy, n_reps, seed=seed, workers=workers)


__all__ = [
    "ActionPolicy", "ActionLoopStep", "ActionLoopTrace", "ActionLoopSummary",
    "run_action_loop", "summarize_action_loop", "simulate_action_loop",
    "evaluate_power_surface", "run_power_scenario",
]
