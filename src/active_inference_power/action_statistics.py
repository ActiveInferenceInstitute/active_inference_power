"""Replication-level statistics for action-in-the-loop power experiments.

The action loop produces one trace per controlled environment replication.  This
module turns those traces into auditable estimands without mixing policy-visible
quantities with hidden evaluation state.  It intentionally keeps power as the
mean replication-level true-positive rate and FDR as the mean replication-level
false-discovery proportion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .scenario import PowerScenario
from .uncertainty import (
    mean_interval,
    paired_difference_summary,
    paired_numeric_difference_summary,
    summarize_bounded,
    wilson_interval,
)


@dataclass(frozen=True)
class ActionLoopReplicationMetrics:
    """Evaluation-only metrics for one action-loop replication."""

    decision: float
    true_positive_rate: float
    false_discovery_proportion: float
    family_wise_error: float
    rejection_count: float
    true_rejection_count: float
    false_rejection_count: float
    observed_count: float
    step_count: float
    total_cost: float
    power_per_cost: float
    target_rejections: tuple[float, ...]


def metrics_for_trace(
    scenario: PowerScenario, trace: Any
) -> ActionLoopReplicationMetrics:
    """Compute one trace's operating characteristics from hidden truth.

    ``trace.rejected_targets`` is the only decision object consulted.  Hidden
    states and target truth are used after the policy has stopped, exclusively
    for evaluation.
    """
    rejected = np.asarray(trace.rejected_targets, dtype=int)
    if rejected.ndim != 1 or np.any(rejected < 0) or np.any(rejected >= scenario.setting.family_size):
        raise ValueError("trace rejected_targets must contain valid hypothesis indices")
    if rejected.size and np.unique(rejected).size != rejected.size:
        raise ValueError("trace rejected_targets must be unique")
    truth = np.asarray(scenario.process.target_truth, dtype=bool)
    is_true = truth[rejected] if rejected.size else np.asarray([], dtype=bool)
    true_rejections = int(np.count_nonzero(is_true))
    false_rejections = int(rejected.size - true_rejections)
    true_count = max(1, int(np.count_nonzero(truth)))
    cost = float(trace.total_cost)
    target_rejections = np.zeros(scenario.setting.family_size, dtype=float)
    if rejected.size:
        target_rejections[rejected] = 1.0
    return ActionLoopReplicationMetrics(
        decision=float(bool(trace.decision)),
        true_positive_rate=float(true_rejections / true_count),
        false_discovery_proportion=float(false_rejections / max(1, rejected.size)),
        family_wise_error=float(false_rejections > 0),
        rejection_count=float(rejected.size),
        true_rejection_count=float(true_rejections),
        false_rejection_count=float(false_rejections),
        observed_count=float(len(trace.observations)),
        step_count=float(trace.stopping_time),
        total_cost=cost,
        power_per_cost=float((true_rejections / true_count) / cost) if cost > 0.0 else 0.0,
        target_rejections=tuple(target_rejections.tolist()),
    )


def summarize_action_traces(
    scenario: PowerScenario,
    traces: Sequence[Any],
) -> dict[str, Any]:
    """Return means, finite-simulation intervals, and target-level rates."""
    if not traces:
        raise ValueError("traces must be non-empty")
    metrics = [metrics_for_trace(scenario, trace) for trace in traces]
    confidence = scenario.setting.confidence

    def values(name: str) -> np.ndarray:
        return np.asarray([float(getattr(item, name)) for item in metrics], dtype=float)

    decisions = values("decision")
    power = values("true_positive_rate")
    fdr = values("false_discovery_proportion")
    fwer = values("family_wise_error")
    rejection_count = values("rejection_count")
    true_rejection_count = values("true_rejection_count")
    false_rejection_count = values("false_rejection_count")
    observations = values("observed_count")
    steps = values("step_count")
    costs = values("total_cost")
    power_per_cost = values("power_per_cost")
    target_rejections = np.asarray([item.target_rejections for item in metrics], dtype=float)
    target_truth = np.asarray(scenario.process.target_truth, dtype=bool)

    bounded = {
        "decision_interval": wilson_interval(int(np.sum(decisions)), len(metrics), confidence),
        "power_interval": (
            float(summarize_bounded(power, confidence)["normal_ci_lower"]),
            float(summarize_bounded(power, confidence)["normal_ci_upper"]),
        ),
        "fdr_interval": (
            float(summarize_bounded(fdr, confidence)["normal_ci_lower"]),
            float(summarize_bounded(fdr, confidence)["normal_ci_upper"]),
        ),
        "fwer_interval": wilson_interval(int(np.sum(fwer)), len(metrics), confidence),
    }
    summary: dict[str, Any] = {
        "n_reps": len(metrics),
        "decision_rate": float(np.mean(decisions)),
        "power": float(np.mean(power)),
        "fdr": float(np.mean(fdr)),
        "fwer": float(np.mean(fwer)),
        "mean_rejections": float(np.mean(rejection_count)),
        "mean_true_rejections": float(np.mean(true_rejection_count)),
        "mean_false_rejections": float(np.mean(false_rejection_count)),
        "any_rejection_rate": float(np.mean(rejection_count > 0.0)),
        "any_true_rejection_rate": float(np.mean(true_rejection_count > 0.0)),
        "mean_stopping_time": float(np.mean(steps)),
        "mean_steps": float(np.mean(steps)),
        "mean_observations": float(np.mean(observations)),
        "mean_cost": float(np.mean(costs)),
        "power_per_cost": float(np.mean(power_per_cost)),
        "target_truth": target_truth.tolist(),
        "target_rejection_rates": np.mean(target_rejections, axis=0).tolist(),
        "target_true_rejection_rates": [
            float(np.mean(target_rejections[:, index])) if truth else None
            for index, truth in enumerate(target_truth)
        ],
        "target_false_rejection_rates": [
            float(np.mean(target_rejections[:, index])) if not truth else None
            for index, truth in enumerate(target_truth)
        ],
        "cost_interval": list(mean_interval(costs, confidence)),
        "stopping_interval": list(mean_interval(steps, confidence)),
        "observation_interval": list(mean_interval(observations, confidence)),
        "power_per_cost_interval": list(mean_interval(power_per_cost, confidence)),
        "validity_assumptions": list(getattr(traces[0], "validity_assumptions", ())),
        "power_interval_method": "student_t_monte_carlo_mean_of_replication_tpr",
        "fdr_interval_method": "student_t_monte_carlo_mean_of_replication_fdp",
        "decision_interval_method": "wilson_binomial",
        "fwer_interval_method": "wilson_binomial",
    }
    summary.update({key: list(value) for key, value in bounded.items()})
    return summary


def paired_action_policy_comparisons(
    scenario: PowerScenario,
    policies: Sequence[str],
    n_reps: int,
    *,
    seed: int = 0,
    confidence: float | None = None,
    run_trace: Any,
) -> dict[str, Any]:
    """Compare policies on common replication seeds with simultaneous bands.

    A common seed aligns initial states and random-number streams by replication
    identifier.  Because actions change the process kernel, this is a paired
    design—not a claim that every policy observes identical trajectories.
    """
    if isinstance(n_reps, bool) or int(n_reps) != n_reps or n_reps < 1:
        raise ValueError("n_reps must be a positive integer")
    policy_names = tuple(str(policy) for policy in policies)
    if not policy_names or len(set(policy_names)) != len(policy_names):
        raise ValueError("policies must be a non-empty sequence of unique names")
    confidence = scenario.setting.confidence if confidence is None else float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    children = np.random.SeedSequence(int(seed)).spawn(int(n_reps))
    seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    traces_by_policy = {
        policy: [run_trace(scenario, policy, seed=rep_seed) for rep_seed in seeds]
        for policy in policy_names
    }
    summaries = {
        policy: summarize_action_traces(scenario, traces)
        for policy, traces in traces_by_policy.items()
    }
    reference = policy_names[0]
    metrics = (
        "decision",
        "true_positive_rate",
        "false_discovery_proportion",
        "family_wise_error",
        "rejection_count",
        "total_cost",
        "step_count",
        "power_per_cost",
    )
    bounded_metrics = {"decision", "true_positive_rate", "false_discovery_proportion", "family_wise_error", "power_per_cost"}
    comparisons = max(1, (len(policy_names) - 1) * len(metrics))
    adjusted_confidence = 1.0 - (1.0 - confidence) / comparisons
    contrasts: dict[str, Any] = {}
    for policy in policy_names[1:]:
        left_metrics = [metrics_for_trace(scenario, trace) for trace in traces_by_policy[reference]]
        right_metrics = [metrics_for_trace(scenario, trace) for trace in traces_by_policy[policy]]
        contrasts[policy] = {
            metric: (
                paired_difference_summary(
                    [getattr(item, metric) for item in left_metrics],
                    [getattr(item, metric) for item in right_metrics],
                    adjusted_confidence,
                )
                if metric in bounded_metrics
                else paired_numeric_difference_summary(
                    [getattr(item, metric) for item in left_metrics],
                    [getattr(item, metric) for item in right_metrics],
                    adjusted_confidence,
                )
            )
            for metric in metrics
        }
    return {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "reference_policy": reference,
        "policies": list(policy_names),
        "n_reps": int(n_reps),
        "seed": int(seed),
        "seed_schedule": seeds,
        "confidence": confidence,
        "adjusted_confidence": adjusted_confidence,
        "summaries": summaries,
        "contrasts": contrasts,
        "validity": "paired finite-simulation policy comparison; action-dependent trajectories are not identical",
    }


__all__ = [
    "ActionLoopReplicationMetrics",
    "metrics_for_trace",
    "summarize_action_traces",
    "paired_action_policy_comparisons",
]
