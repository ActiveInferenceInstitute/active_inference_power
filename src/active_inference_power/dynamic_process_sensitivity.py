"""Paired dynamic-process stress controls for action-loop evidence.

Each dynamic process is evaluated beside the same static scenario with stable
replication seeds.  The comparison is a diagnostic stress surface: a richer
environment does not create a stronger optional-stopping or family-level
validity claim.  Process schedules, missingness mechanisms, and pairwise
finite-simulation summaries are explicit so a reader can see exactly which
part of the model/process contract changed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from dataclasses import replace
from typing import Any

import numpy as np

from .action_loop import ActionPolicy, ActionLoopTrace, run_action_loop
from .action_statistics import metrics_for_trace, summarize_action_traces
from .scenario import GenerativeProcessSpec, PowerScenario
from .suite import evaluate_conditional_selection_calibration
from .uncertainty import paired_difference_summary, paired_numeric_difference_summary


def _hash(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class DynamicProcessSensitivityPlan:
    """Frozen controls for the paired dynamic-process stress surface."""

    horizon: int = 8
    context_persistence: float = 0.9
    reliability_end_multiplier: float = 0.65
    mcar_missing_end: float = 0.2
    mar_missing_low: float = 0.05
    mar_missing_high: float = 0.25
    mnar_missing_low: float = 0.05
    mnar_missing_high: float = 0.25
    policy: ActionPolicy = "cost_aware"
    calibration_policy: ActionPolicy = "adaptive_target"

    def __post_init__(self) -> None:
        _positive_integer(self.horizon, "horizon")
        for name, value in (
            ("context_persistence", self.context_persistence),
            ("mcar_missing_end", self.mcar_missing_end),
            ("mar_missing_low", self.mar_missing_low),
            ("mar_missing_high", self.mar_missing_high),
            ("mnar_missing_low", self.mnar_missing_low),
            ("mnar_missing_high", self.mnar_missing_high),
        ):
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 <= numeric < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        if not 0.0 < float(self.context_persistence) < 1.0:
            raise ValueError("context_persistence must lie in (0, 1)")
        if not 0.0 < float(self.reliability_end_multiplier) <= 1.0:
            raise ValueError("reliability_end_multiplier must lie in (0, 1]")
        if self.mar_missing_low > self.mar_missing_high:
            raise ValueError("mar missingness bounds must be ordered")
        if self.mnar_missing_low > self.mnar_missing_high:
            raise ValueError("mnar missingness bounds must be ordered")
        valid_policies = {
            "fixed_horizon", "posterior_threshold", "information_gain", "cost_aware",
            "adaptive_target", "posterior_sampling", "e_process",
        }
        if self.policy not in valid_policies or self.calibration_policy not in valid_policies:
            raise ValueError("dynamic sensitivity policies must be supported action policies")
        object.__setattr__(self, "horizon", int(self.horizon))
        for name in (
            "context_persistence", "reliability_end_multiplier", "mcar_missing_end",
            "mar_missing_low", "mar_missing_high", "mnar_missing_low", "mnar_missing_high",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))

    def to_mapping(self) -> dict[str, object]:
        return {
            "horizon": self.horizon,
            "context_persistence": self.context_persistence,
            "reliability_end_multiplier": self.reliability_end_multiplier,
            "mcar_missing_end": self.mcar_missing_end,
            "mar_missing_low": self.mar_missing_low,
            "mar_missing_high": self.mar_missing_high,
            "mnar_missing_low": self.mnar_missing_low,
            "mnar_missing_high": self.mnar_missing_high,
            "policy": self.policy,
            "calibration_policy": self.calibration_policy,
            "regimes": [
                "persistent_context", "reliability_drift", "action_transition",
                "mcar_missingness", "mar_action_missingness", "mnar_state_missingness",
            ],
        }

    @property
    def configuration_hash(self) -> str:
        return _hash(self.to_mapping())


def _context_transition(
    scenario: PowerScenario, *, persistence: float, action_dependent: bool
) -> np.ndarray:
    """Create a context-transition schedule while retaining target/truth roles."""
    state_target = np.asarray(scenario.model.state_target, dtype=int)
    state_context = np.asarray(scenario.model.state_context, dtype=int)
    state_truth = np.asarray(scenario.model.state_truth, dtype=int)
    base = np.asarray(scenario.process.transition, dtype=float)
    changed = np.zeros_like(base)
    for action in range(scenario.model.n_actions):
        for current in range(scenario.model.n_states):
            destination = int(np.argmax(base[action, :, current]))
            target = int(state_target[destination])
            truth = int(state_truth[destination])
            current_context = int(state_context[current])
            if action_dependent:
                sensor = int(scenario.model.action_sensor[action])
                keep = persistence if sensor == 1 else 1.0 - persistence
            else:
                keep = persistence
            matching = np.flatnonzero(
                (state_target == target) & (state_truth == truth)
            )
            same = matching[state_context[matching] == current_context]
            other = matching[state_context[matching] != current_context]
            if same.size != 1 or other.size != 1:
                raise ValueError("dynamic context control requires exactly two contexts per target/truth")
            changed[action, int(same[0]), current] = keep
            changed[action, int(other[0]), current] = 1.0 - keep
    return changed


def _reliability_schedule(scenario: PowerScenario, plan: DynamicProcessSensitivityPlan) -> np.ndarray:
    base = np.asarray(scenario.process.likelihood, dtype=float)
    multipliers = np.linspace(1.0, plan.reliability_end_multiplier, plan.horizon)
    schedule = np.empty((plan.horizon, *base.shape), dtype=float)
    for index, multiplier in enumerate(multipliers):
        high = 0.5 + (base[:, 1, :] - 0.5) * multiplier
        schedule[index, :, 1, :] = high
        schedule[index, :, 0, :] = 1.0 - high
    return schedule


def _clone_static_control(base: PowerScenario, regime: str) -> PowerScenario:
    return replace(
        base,
        scenario_id=f"{base.scenario_id}__{regime}__static_control",
        description=f"Static paired control for the {regime} dynamic-process stress regime.",
        metadata={
            **dict(base.metadata or {}),
            "paired_regime": regime,
            "paired_role": "static_control",
            "evidence_class": "conditional_operating_characteristic",
        },
    )


def _dynamic_scenario(base: PowerScenario, regime: str, process: GenerativeProcessSpec) -> PowerScenario:
    return replace(
        base,
        scenario_id=f"{base.scenario_id}__{regime}__dynamic",
        process=process,
        description=f"Dynamic {regime} stress scenario paired with a static control.",
        claim_role="synthetic_dynamic_process_stress",
        metadata={
            **dict(base.metadata or {}),
            "paired_regime": regime,
            "paired_role": "dynamic_stress",
            "evidence_class": "stress_test",
            "validity": "diagnostic_dynamic_process_stress",
        },
    )


def dynamic_process_pairs(
    base: PowerScenario, plan: DynamicProcessSensitivityPlan
) -> dict[str, tuple[PowerScenario, PowerScenario]]:
    """Return every static-control/dynamic-process pair in the frozen plan."""
    if base.process.transition_schedule is not None or base.process.likelihood_schedule is not None:
        raise ValueError("base scenario must be static before building dynamic controls")
    base = replace(base, setting=replace(base.setting, max_steps=plan.horizon))
    horizon = plan.horizon
    process = base.process
    persistent = replace(
        process,
        transition_schedule=np.repeat(
            _context_transition(base, persistence=plan.context_persistence, action_dependent=False)[None, ...],
            horizon,
            axis=0,
        ),
        schedule_policy="strict",
        name="persistent_context_stress",
    )
    reliability = replace(
        process,
        likelihood_schedule=_reliability_schedule(base, plan),
        schedule_policy="strict",
        name="reliability_drift_stress",
    )
    action_transition = replace(
        process,
        transition_schedule=np.repeat(
            _context_transition(base, persistence=plan.context_persistence, action_dependent=True)[None, ...],
            horizon,
            axis=0,
        ),
        schedule_policy="strict",
        name="action_transition_stress",
    )
    mcar = replace(
        process,
        missing_probability_schedule=np.linspace(0.0, plan.mcar_missing_end, horizon),
        schedule_policy="strict",
        missingness_mode="mcar",
        name="mcar_missingness_stress",
    )
    action_missing = np.linspace(
        plan.mar_missing_low, plan.mar_missing_high, base.model.n_actions
    )
    mar = replace(
        process,
        missingness_mode="mar_action",
        action_missing_probability=action_missing,
        name="mar_action_missingness_stress",
    )
    state_missing = np.linspace(
        plan.mnar_missing_low, plan.mnar_missing_high, base.model.n_states
    )
    mnar = replace(
        process,
        missingness_mode="mnar_state",
        state_missing_probability=state_missing,
        name="mnar_state_missingness_stress",
    )
    dynamic_processes = {
        "persistent_context": persistent,
        "reliability_drift": reliability,
        "action_transition": action_transition,
        "mcar_missingness": mcar,
        "mar_action_missingness": mar,
        "mnar_state_missingness": mnar,
    }
    return {
        regime: (_clone_static_control(base, regime), _dynamic_scenario(base, regime, value))
        for regime, value in dynamic_processes.items()
    }


def _process_panel(scenario: PowerScenario, horizon: int) -> dict[str, object]:
    process = scenario.process
    likelihood = np.stack([process.likelihood_at(time) for time in range(horizon)])
    transitions = np.stack([process.transition_at(time) for time in range(horizon)])
    costs = np.stack([process.action_cost_at(time) for time in range(horizon)])
    missing = [
        float(process.missing_probability_at(time, 0, 0))
        for time in range(horizon)
    ]
    return {
        "process_hash": process.configuration_hash,
        "transition_schedule_hash": _hash(transitions.tolist()),
        "likelihood_schedule_hash": _hash(likelihood.tolist()),
        "cost_schedule_hash": _hash(costs.tolist()),
        "time": list(range(horizon)),
        "mean_observation_one_probability": [
            float(np.mean(likelihood[time, :, 1, :])) for time in range(horizon)
        ],
        "mean_action_cost": [float(np.mean(costs[time])) for time in range(horizon)],
        "action0_missing_probability": missing,
        "missingness_mode": process.missingness_mode,
        "missingness_description": (
            "time-scheduled MCAR" if process.missing_probability_schedule is not None
            else "action-dependent MAR" if process.missingness_mode == "mar_action"
            else "state-dependent MNAR" if process.missingness_mode == "mnar_state"
            else "no scheduled missingness"
        ),
    }


def _paired_summary(
    control: PowerScenario,
    dynamic: PowerScenario,
    control_traces: list[ActionLoopTrace],
    dynamic_traces: list[ActionLoopTrace],
) -> dict[str, object]:
    left = [metrics_for_trace(control, trace) for trace in control_traces]
    right = [metrics_for_trace(dynamic, trace) for trace in dynamic_traces]
    confidence = control.setting.confidence
    bounded = {
        "power": ("true_positive_rate", paired_difference_summary),
        "fdr": ("false_discovery_proportion", paired_difference_summary),
        "fwer": ("family_wise_error", paired_difference_summary),
    }
    numeric = {
        "cost": "total_cost",
        "stopping_time": "step_count",
    }
    result: dict[str, object] = {}
    for name, (attribute, function) in bounded.items():
        result[name] = function(
            [getattr(item, attribute) for item in left],
            [getattr(item, attribute) for item in right],
            confidence,
        )
    for name, attribute in numeric.items():
        result[name] = paired_numeric_difference_summary(
            [getattr(item, attribute) for item in left],
            [getattr(item, attribute) for item in right],
            confidence,
        )
    return result


def _all_null_process(scenario: PowerScenario) -> PowerScenario:
    """Create a process-side all-null calibration counterpart.

    Truth is held exclusively in the evaluator-side process; the agent model is
    intentionally unchanged, preventing calibration diagnostics from leaking
    the all-null fact into policy decisions.
    """
    state_target = np.asarray(scenario.model.state_target, dtype=int)
    state_truth = np.asarray(scenario.model.state_truth, dtype=int)
    state_context = np.asarray(scenario.model.state_context, dtype=int)
    initial = np.zeros_like(scenario.process.initial_state)
    mask = (state_truth == 0) & (state_context == np.min(state_context))
    initial[mask] = 1.0 / float(np.count_nonzero(mask))

    def null_transition(matrix: np.ndarray) -> np.ndarray:
        redirected = np.zeros_like(matrix)
        for action in range(matrix.shape[0]):
            for destination in range(matrix.shape[1]):
                null_destination = np.flatnonzero(
                    (state_target == state_target[destination])
                    & (state_context == state_context[destination])
                    & (state_truth == 0)
                )
                if null_destination.size != 1:
                    raise ValueError("all-null calibration requires one null state per target/context")
                redirected[action, int(null_destination[0]), :] += matrix[action, destination, :]
        return redirected

    transition = null_transition(np.asarray(scenario.process.transition, dtype=float))
    transition_schedule = (
        None
        if scenario.process.transition_schedule is None
        else np.stack(
            [null_transition(matrix) for matrix in scenario.process.transition_schedule]
        )
    )
    process = replace(
        scenario.process,
        initial_state=initial,
        transition=transition,
        transition_schedule=transition_schedule,
        target_truth=np.zeros(scenario.setting.family_size, dtype=bool),
        name=f"{scenario.process.name}_all_null_calibration",
    )
    return replace(
        scenario,
        scenario_id=f"{scenario.scenario_id}__all_null_calibration",
        process=process,
        claim_role="synthetic_calibration_diagnostic",
        metadata={**dict(scenario.metadata or {}), "all_null_process": True},
    )


def evaluate_dynamic_process_sensitivity(
    base: PowerScenario,
    plan: DynamicProcessSensitivityPlan,
    n_reps: int,
    *,
    seed: int = 0,
    workers: int = 1,
) -> dict[str, object]:
    """Run paired static/dynamic controls plus marginal calibration panels."""
    n_reps = _positive_integer(n_reps, "n_reps")
    if isinstance(workers, bool) or int(workers) != workers or int(workers) < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    pairs = dynamic_process_pairs(base, plan)
    children = np.random.SeedSequence(int(seed)).spawn(n_reps)
    seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]

    def run_pair(
        control: PowerScenario, dynamic: PowerScenario, rep_seed: int
    ) -> tuple[ActionLoopTrace, ActionLoopTrace]:
        return (
            run_action_loop(control, plan.policy, seed=rep_seed),
            run_action_loop(dynamic, plan.policy, seed=rep_seed),
        )

    regimes: list[dict[str, object]] = []
    for index, (name, (control, dynamic)) in enumerate(pairs.items()):
        if workers == 1:
            traces = [run_pair(control, dynamic, rep_seed) for rep_seed in seeds]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                traces = list(executor.map(lambda rep_seed: run_pair(control, dynamic, rep_seed), seeds))
        control_traces = [pair[0] for pair in traces]
        dynamic_traces = [pair[1] for pair in traces]
        null_dynamic = _all_null_process(dynamic)
        calibration = evaluate_conditional_selection_calibration(
            null_dynamic,
            plan.calibration_policy,
            n_reps,
            seed=int(seed) + 10_000 + index,
            workers=workers,
        )
        regimes.append(
            {
                "regime": name,
                "static_control": {
                    "scenario_id": control.scenario_id,
                    "scenario_hash": control.configuration_hash,
                    "summary": summarize_action_traces(control, control_traces),
                    "process_panel": _process_panel(control, plan.horizon),
                },
                "dynamic": {
                    "scenario_id": dynamic.scenario_id,
                    "scenario_hash": dynamic.configuration_hash,
                    "summary": summarize_action_traces(dynamic, dynamic_traces),
                    "process_panel": _process_panel(dynamic, plan.horizon),
                    "marginal_calibration": calibration,
                },
                "paired_static_to_dynamic": _paired_summary(
                    control, dynamic, control_traces, dynamic_traces
                ),
                "validity": "diagnostic_dynamic_process_stress",
                "claim_boundary": (
                    "Paired finite-simulation stress comparison under the declared "
                    "process schedules; it does not establish a formal dynamic-process law."
                ),
            }
        )
    return {
        "base_scenario_id": base.scenario_id,
        "base_scenario_hash": base.configuration_hash,
        "plan": plan.to_mapping(),
        "plan_hash": plan.configuration_hash,
        "n_reps": n_reps,
        "seed": int(seed),
        "seed_schedule": seeds,
        "workers": workers,
        "common_random_numbers": True,
        "regimes": regimes,
        "evidence_class": "stress_test",
        "claim_boundary": (
            "Dynamic schedules, action-transition effects, and MCAR/MAR/MNAR "
            "missingness remain diagnostic/stress evidence unless a separate "
            "conditional law is verified."
        ),
    }


__all__ = [
    "DynamicProcessSensitivityPlan",
    "dynamic_process_pairs",
    "evaluate_dynamic_process_sensitivity",
]
