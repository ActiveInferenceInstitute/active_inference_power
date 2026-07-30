"""Additive multi-target arrival environment for diagnostic control studies.

This module intentionally has its own versioned contract instead of changing
the fixed-family :class:`~active_inference_power.scenario.PowerScenario`
schema.  Targets arrive over time, sensing actions change reliability and
cost, and the policy receives only a declared visible filtration.  Independent
confirmation draws are generated after selection.  The simulation is a
stress/operating-characteristic lane, not a selective-inference theorem.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Literal

import numpy as np
from scipy.stats import binom

from .uncertainty import paired_difference_summary, paired_numeric_difference_summary


SelectionPolicy = Literal["round_robin", "minimum_visible_p"]


def _hash(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class MultiTargetEnvironmentSpec:
    """Versioned, visible-filtration contract for a target-arrival environment."""

    target_ids: tuple[str, ...]
    arrival_times: tuple[int, ...]
    target_truth: tuple[bool, ...]
    reliability_schedule: tuple[float, ...]
    action_cost_schedule: tuple[float, ...]
    horizon: int
    confirmation_steps: int = 4
    alpha: float = 0.05
    policy: SelectionPolicy = "minimum_visible_p"
    precise_reliability_bonus: float = 0.1
    precise_cost_multiplier: float = 2.0
    contract_version: str = "2.0"

    def __post_init__(self) -> None:
        ids = tuple(str(value) for value in self.target_ids)
        if len(ids) < 3 or len(set(ids)) != len(ids) or any(not value for value in ids):
            raise ValueError("target_ids must contain at least three unique non-empty values")
        arrivals = tuple(int(value) for value in self.arrival_times)
        truth = tuple(bool(value) for value in self.target_truth)
        if len(arrivals) != len(ids) or len(truth) != len(ids):
            raise ValueError("arrival_times and target_truth must align with target_ids")
        if any(value < 0 for value in arrivals):
            raise ValueError("arrival_times must be non-negative")
        horizon = _positive_integer(self.horizon, "horizon")
        if any(value >= horizon for value in arrivals):
            raise ValueError("every target must arrive before horizon")
        reliability = tuple(float(value) for value in self.reliability_schedule)
        costs = tuple(float(value) for value in self.action_cost_schedule)
        if len(reliability) != horizon or len(costs) != horizon:
            raise ValueError("reliability and cost schedules must align with horizon")
        if any(not math.isfinite(value) or not 0.5 < value < 1.0 for value in reliability):
            raise ValueError("reliability_schedule must lie in (0.5, 1)")
        if any(not math.isfinite(value) or value <= 0.0 for value in costs):
            raise ValueError("action_cost_schedule must be finite and positive")
        _positive_integer(self.confirmation_steps, "confirmation_steps")
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.policy not in {"round_robin", "minimum_visible_p"}:
            raise ValueError("unsupported multi-target selection policy")
        for name, value in (
            ("precise_reliability_bonus", self.precise_reliability_bonus),
            ("precise_cost_multiplier", self.precise_cost_multiplier),
        ):
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.contract_version != "2.0":
            raise ValueError("multi-target contract_version must be '2.0'")
        object.__setattr__(self, "target_ids", ids)
        object.__setattr__(self, "arrival_times", arrivals)
        object.__setattr__(self, "target_truth", truth)
        object.__setattr__(self, "reliability_schedule", reliability)
        object.__setattr__(self, "action_cost_schedule", costs)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "confirmation_steps", int(self.confirmation_steps))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "precise_reliability_bonus", float(self.precise_reliability_bonus))
        object.__setattr__(self, "precise_cost_multiplier", float(self.precise_cost_multiplier))

    def visible_contract(self) -> dict[str, object]:
        """Return exactly the policy-visible portion of the environment."""
        return {
            "contract_version": self.contract_version,
            "target_ids": list(self.target_ids),
            "arrival_times": list(self.arrival_times),
            "reliability_schedule": list(self.reliability_schedule),
            "action_cost_schedule": list(self.action_cost_schedule),
            "horizon": self.horizon,
            "confirmation_steps": self.confirmation_steps,
            "alpha": self.alpha,
            "policy": self.policy,
            "precise_reliability_bonus": self.precise_reliability_bonus,
            "precise_cost_multiplier": self.precise_cost_multiplier,
            "hidden_from_policy": ["target_truth", "confirmation_observations"],
        }

    def to_mapping(self) -> dict[str, object]:
        return self.visible_contract() | {
            "target_truth": [int(value) for value in self.target_truth],
            "environment_hash": self.configuration_hash,
        }

    @property
    def configuration_hash(self) -> str:
        payload = self.visible_contract() | {
            "target_truth": [int(value) for value in self.target_truth]
        }
        return _hash(payload)


@dataclass(frozen=True)
class MultiTargetFiltrationEvent:
    """One policy-visible information state before a sensing action."""

    time: int
    available_targets: tuple[int, ...]
    selected_target: int
    action: Literal["cheap", "precise"]
    visible_observations: tuple[int, ...]
    visible_target_history: tuple[tuple[int, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        if isinstance(self.time, bool) or int(self.time) != self.time or self.time < 0:
            raise ValueError("filtration time must be a non-negative integer")
        available = tuple(int(value) for value in self.available_targets)
        if not available or len(set(available)) != len(available) or any(value < 0 for value in available):
            raise ValueError("available_targets must contain unique non-negative targets")
        if int(self.selected_target) not in available:
            raise ValueError("selected_target must be available")
        if self.action not in {"cheap", "precise"}:
            raise ValueError("filtration action must be cheap or precise")
        observations = tuple(int(value) for value in self.visible_observations)
        if any(value not in {0, 1} for value in observations):
            raise ValueError("visible observations must be binary")
        histories = tuple(
            (int(target), tuple(int(value) for value in values))
            for target, values in self.visible_target_history
        )
        if any(target < 0 or any(value not in {0, 1} for value in values) for target, values in histories):
            raise ValueError("visible target history must use non-negative targets and binary values")
        object.__setattr__(self, "available_targets", available)
        object.__setattr__(self, "selected_target", int(self.selected_target))
        object.__setattr__(self, "visible_observations", observations)
        object.__setattr__(self, "visible_target_history", histories)

    def to_mapping(self) -> dict[str, object]:
        return {
            "time": int(self.time),
            "available_targets": list(self.available_targets),
            "selected_target": self.selected_target,
            "action": self.action,
            "visible_observations": list(self.visible_observations),
            "visible_target_history": [
                {"target": target, "observations": list(values)}
                for target, values in self.visible_target_history
            ],
        }


@dataclass(frozen=True)
class MultiTargetTrace:
    """One arrival/control/confirmation replication with hash-bound streams."""

    environment_hash: str
    seed: int
    confirmation_seed: int
    events: tuple[MultiTargetFiltrationEvent, ...]
    selected_target: int
    confirmation_observations: tuple[int, ...]
    confirmation_p_value: float
    rejected: bool
    selected_target_truth: bool
    total_cost: float
    visible_trace_hash: str
    confirmation_stream_hash: str

    def __post_init__(self) -> None:
        if len(self.environment_hash) != 64 or len(self.visible_trace_hash) != 64 or len(self.confirmation_stream_hash) != 64:
            raise ValueError("multi-target traces require SHA-256 hashes")
        if isinstance(self.seed, bool) or isinstance(self.confirmation_seed, bool) or self.seed < 0 or self.confirmation_seed < 0:
            raise ValueError("stream seeds must be non-negative")
        if self.seed == self.confirmation_seed:
            raise ValueError("selection and confirmation seeds must be independent")
        if not self.events or self.selected_target != self.events[-1].selected_target:
            raise ValueError("selected target must match the final visible filtration event")
        observations = tuple(int(value) for value in self.confirmation_observations)
        if not observations or any(value not in {0, 1} for value in observations):
            raise ValueError("confirmation observations must be non-empty and binary")
        if not 0.0 <= float(self.confirmation_p_value) <= 1.0:
            raise ValueError("confirmation p-value must lie in [0, 1]")
        if not math.isfinite(float(self.total_cost)) or float(self.total_cost) <= 0.0:
            raise ValueError("total cost must be finite and positive")
        object.__setattr__(self, "confirmation_observations", observations)
        object.__setattr__(self, "total_cost", float(self.total_cost))

    def visible_payload(self) -> dict[str, object]:
        return {
            "environment_hash": self.environment_hash,
            "seed": int(self.seed),
            "events": [event.to_mapping() for event in self.events],
            "selected_target": self.selected_target,
            "total_cost": self.total_cost,
        }

    def confirmation_payload(self) -> dict[str, object]:
        return {
            "environment_hash": self.environment_hash,
            "selected_target": self.selected_target,
            "confirmation_seed": int(self.confirmation_seed),
            "confirmation_observations": list(self.confirmation_observations),
            "confirmation_p_value": float(self.confirmation_p_value),
            "rejected": bool(self.rejected),
            "selected_target_truth": bool(self.selected_target_truth),
        }

    def to_mapping(self) -> dict[str, object]:
        return self.visible_payload() | self.confirmation_payload() | {
            "visible_trace_hash": self.visible_trace_hash,
            "confirmation_stream_hash": self.confirmation_stream_hash,
        }


def _choose_target(
    spec: MultiTargetEnvironmentSpec, available: list[int], history: dict[int, list[int]], time: int
) -> int:
    if spec.policy == "round_robin":
        return available[time % len(available)]
    def score(target: int) -> tuple[float, int]:
        values = history[target]
        if not values:
            return (1.0, target)
        # A one-sided null reference is enough for an explicitly visible
        # selection score; it is not used as confirmation evidence.
        return (float(binom.sf(sum(values) - 1, len(values), 0.5)), target)
    return min(available, key=score)


def _action_for(history: dict[int, list[int]], target: int) -> Literal["cheap", "precise"]:
    """Use only visible history to choose the costly sensing action."""
    values = history[target]
    if not values or abs(float(np.mean(values)) - 0.5) < 0.2:
        return "precise"
    return "cheap"


def run_multi_target_environment(
    spec: MultiTargetEnvironmentSpec, *, seed: int = 0, confirmation_seed: int = 1
) -> MultiTargetTrace:
    """Run a visible-filtration target-arrival trace and independent confirmation."""
    if isinstance(seed, bool) or isinstance(confirmation_seed, bool) or seed < 0 or confirmation_seed < 0:
        raise ValueError("stream seeds must be non-negative integers")
    if seed == confirmation_seed:
        raise ValueError("selection and confirmation seeds must differ")
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
    history: dict[int, list[int]] = {index: [] for index in range(len(spec.target_ids))}
    events: list[MultiTargetFiltrationEvent] = []
    total_cost = 0.0
    for time in range(spec.horizon):
        available = [
            index for index, arrival in enumerate(spec.arrival_times) if arrival <= time
        ]
        target = _choose_target(spec, available, history, time)
        action = _action_for(history, target)
        reliability = min(
            0.999999,
            spec.reliability_schedule[time]
            + (spec.precise_reliability_bonus if action == "precise" else 0.0),
        )
        cost = spec.action_cost_schedule[time] * (
            spec.precise_cost_multiplier if action == "precise" else 1.0
        )
        visible_history = tuple(
            (index, tuple(history[index])) for index in available
        )
        events.append(
            MultiTargetFiltrationEvent(
                time, tuple(available), target, action,
                tuple(value for values in history.values() for value in values),
                visible_history,
            )
        )
        probability = reliability if spec.target_truth[target] else 1.0 - reliability
        history[target].append(int(rng.random() < probability))
        total_cost += cost
    selected = events[-1].selected_target
    confirmation_rng = np.random.default_rng(np.random.SeedSequence(int(confirmation_seed)))
    confirmation_reliability = min(0.999999, spec.reliability_schedule[-1])
    probability = confirmation_reliability if spec.target_truth[selected] else 1.0 - confirmation_reliability
    confirmation = tuple(
        int(confirmation_rng.random() < probability)
        for _ in range(spec.confirmation_steps)
    )
    p_value = float(
        binom.sf(sum(confirmation) - 1, len(confirmation), 1.0 - confirmation_reliability)
    )
    visible_payload = {
        "environment_hash": spec.configuration_hash,
        "seed": int(seed),
        "events": [event.to_mapping() for event in events],
        "selected_target": selected,
        "total_cost": total_cost,
    }
    confirmation_payload = {
        "environment_hash": spec.configuration_hash,
        "selected_target": selected,
        "confirmation_seed": int(confirmation_seed),
        "confirmation_observations": list(confirmation),
        "confirmation_p_value": p_value,
        "rejected": bool(p_value <= spec.alpha),
        "selected_target_truth": bool(spec.target_truth[selected]),
    }
    return MultiTargetTrace(
        spec.configuration_hash, int(seed), int(confirmation_seed), tuple(events), selected,
        confirmation, p_value, bool(p_value <= spec.alpha), bool(spec.target_truth[selected]),
        total_cost, _hash(visible_payload), _hash(confirmation_payload),
    )


def make_paired_static_control(spec: MultiTargetEnvironmentSpec) -> MultiTargetEnvironmentSpec:
    """Freeze arrivals/reliability/cost/action effects for a paired control."""
    return MultiTargetEnvironmentSpec(
        target_ids=spec.target_ids,
        arrival_times=(0,) * len(spec.target_ids),
        target_truth=spec.target_truth,
        reliability_schedule=(spec.reliability_schedule[0],) * spec.horizon,
        action_cost_schedule=(spec.action_cost_schedule[0],) * spec.horizon,
        horizon=spec.horizon,
        confirmation_steps=spec.confirmation_steps,
        alpha=spec.alpha,
        policy=spec.policy,
        precise_reliability_bonus=1.0e-12,
        precise_cost_multiplier=1.0,
    )


def _seed_pairs(seed: int, n_reps: int) -> list[tuple[int, int]]:
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    children = np.random.SeedSequence(int(seed)).spawn(2 * n_reps)
    return [
        (
            int(children[2 * index].generate_state(1, dtype=np.uint32)[0]),
            int(children[2 * index + 1].generate_state(1, dtype=np.uint32)[0]),
        )
        for index in range(n_reps)
    ]


def _summary(spec: MultiTargetEnvironmentSpec, traces: list[MultiTargetTrace]) -> dict[str, object]:
    rejected = np.asarray([trace.rejected for trace in traces], dtype=float)
    true = np.asarray([trace.rejected and trace.selected_target_truth for trace in traces], dtype=float)
    false = np.asarray([trace.rejected and not trace.selected_target_truth for trace in traces], dtype=float)
    cost = np.asarray([trace.total_cost for trace in traces], dtype=float)
    stopping = np.full(len(traces), spec.horizon, dtype=float)
    return {
        "n_reps": len(traces),
        "power": float(np.mean(true)),
        "fdr": float(np.mean(false / np.maximum(rejected, 1.0))),
        "fwer": float(np.mean(false)),
        "rejection_rate": float(np.mean(rejected)),
        "mean_cost": float(np.mean(cost)),
        "mean_stopping_time": float(np.mean(stopping)),
        "selection_target_counts": [
            int(sum(trace.selected_target == target for trace in traces))
            for target in range(len(spec.target_ids))
        ],
    }


def evaluate_multi_target_control_pair(
    dynamic: MultiTargetEnvironmentSpec,
    n_reps: int,
    *,
    seed: int = 0,
    workers: int = 1,
) -> dict[str, object]:
    """Compare the new dynamic environment against its paired static control."""
    n_reps = _positive_integer(n_reps, "n_reps")
    if isinstance(workers, bool) or int(workers) != workers or int(workers) < 1:
        raise ValueError("workers must be a positive integer")
    pairs = _seed_pairs(seed, n_reps)
    static = make_paired_static_control(dynamic)

    def run(pair: tuple[int, int]) -> tuple[MultiTargetTrace, MultiTargetTrace]:
        return (
            run_multi_target_environment(static, seed=pair[0], confirmation_seed=pair[1]),
            run_multi_target_environment(dynamic, seed=pair[0], confirmation_seed=pair[1]),
        )

    if workers == 1:
        values = [run(pair) for pair in pairs]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            values = list(executor.map(run, pairs))
    static_traces = [value[0] for value in values]
    dynamic_traces = [value[1] for value in values]
    left = _summary(static, static_traces)
    right = _summary(dynamic, dynamic_traces)
    left_power = [float(trace.rejected and trace.selected_target_truth) for trace in static_traces]
    right_power = [float(trace.rejected and trace.selected_target_truth) for trace in dynamic_traces]
    left_fdr = [float(trace.rejected and not trace.selected_target_truth) for trace in static_traces]
    right_fdr = [float(trace.rejected and not trace.selected_target_truth) for trace in dynamic_traces]
    left_cost = [trace.total_cost for trace in static_traces]
    right_cost = [trace.total_cost for trace in dynamic_traces]
    left_stop = [float(static.horizon)] * n_reps
    right_stop = [float(dynamic.horizon)] * n_reps
    return {
        "contract_version": "2.0",
        "dynamic_environment": dynamic.to_mapping(),
        "static_control": static.to_mapping(),
        "n_reps": n_reps,
        "seed": int(seed),
        "seed_pair_schedule_hash": _hash(pairs),
        "common_random_numbers": True,
        "formal_filtration_specification": {
            "visible_before_action": [
                "arrival_times", "available_targets", "past_actions", "past_observations",
                "reliability_schedule", "action_cost_schedule",
            ],
            "hidden_from_policy": ["target_truth", "confirmation_observations"],
            "confirmation_stream": "independent SeedSequence child after selection",
        },
        "static_summary": left,
        "dynamic_summary": right,
        "paired_dynamic_minus_static": {
            "power": paired_difference_summary(left_power, right_power),
            "fdr": paired_difference_summary(left_fdr, right_fdr),
            "fwer": paired_difference_summary(left_fdr, right_fdr),
            "cost": paired_numeric_difference_summary(left_cost, right_cost),
            "stopping_time": paired_numeric_difference_summary(left_stop, right_stop),
        },
        "representative": dynamic_traces[0].to_mapping(),
        "validity": "diagnostic_multi_target_environment_stress",
        "claim_boundary": (
            "Target arrival, action-dependent sensing, and independent confirmation "
            "are explicit simulation contracts, not a formal filtration proof or "
            "universal adaptive-confirmation guarantee."
        ),
    }


__all__ = [
    "MultiTargetEnvironmentSpec", "MultiTargetFiltrationEvent", "MultiTargetTrace",
    "run_multi_target_environment", "make_paired_static_control",
    "evaluate_multi_target_control_pair",
]
