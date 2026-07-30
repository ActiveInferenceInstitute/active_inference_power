"""Stratified split-stream calibration diagnostics.

The adaptive selector may inspect its own visible selection stream but never
the confirmation stream.  This module evaluates a *predeclared* sensitivity
plan under an all-null scenario, varying only the selection-stream length while
holding policy tuning, alpha, and confirmation length fixed.  Reusing each
replication's two child streams across cells supplies common random numbers for
the resulting design comparison.  The output is a finite calibration
diagnostic, not a selective-inference theorem.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from .scenario import PowerScenario
from .selection import (
    SelectionPolicySpec,
    SelectionStrategy,
    SplitConfirmationResult,
    run_split_stream_confirmation,
)


def _hash_mapping(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class SplitStreamSensitivityPlan:
    """Frozen design for a stratified all-null split-confirmation audit.

    ``selection_steps`` is the sole sensitivity dimension.  The selection
    strategy, evidence visibility, confirmation length, alpha, confidence,
    and replication count are held fixed across every cell.
    """

    selection_steps: tuple[int, ...] = (2, 4, 8)
    confirmation_steps: int = 8
    strategy: SelectionStrategy = "minimum_p"
    evidence_observable: bool = True
    alpha: float = 0.05
    confidence: float = 0.95

    def __post_init__(self) -> None:
        steps = tuple(_positive_integer(value, "selection_steps") for value in self.selection_steps)
        if not steps or len(set(steps)) != len(steps):
            raise ValueError("selection_steps must contain unique positive integers")
        if self.strategy not in {"minimum_p", "least_informed", "round_robin"}:
            raise ValueError("unsupported selection strategy")
        if not isinstance(self.evidence_observable, bool):
            raise ValueError("evidence_observable must be boolean")
        if not self.evidence_observable and self.strategy == "minimum_p":
            raise ValueError("minimum_p requires evidence_observable=True")
        _positive_integer(self.confirmation_steps, "confirmation_steps")
        for name, value in (("alpha", self.alpha), ("confidence", self.confidence)):
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 < numeric < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
        object.__setattr__(self, "selection_steps", steps)
        object.__setattr__(self, "confirmation_steps", int(self.confirmation_steps))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "confidence", float(self.confidence))

    def to_mapping(self) -> dict[str, object]:
        return {
            "selection_steps": list(self.selection_steps),
            "confirmation_steps": self.confirmation_steps,
            "strategy": self.strategy,
            "evidence_observable": self.evidence_observable,
            "alpha": self.alpha,
            "confidence": self.confidence,
            "sensitivity_dimension": "selection_stream_steps",
            "fixed_tuning": [
                "strategy",
                "evidence_observable",
                "confirmation_steps",
                "alpha",
                "confidence",
            ],
        }

    @property
    def configuration_hash(self) -> str:
        return _hash_mapping(self.to_mapping())

    def policy_for(self, selection_steps: int) -> SelectionPolicySpec:
        return SelectionPolicySpec(
            strategy=self.strategy,
            max_steps=_positive_integer(selection_steps, "selection_steps"),
            confirmation_steps=self.confirmation_steps,
            evidence_observable=self.evidence_observable,
        )


def _pvalue_diagnostic(
    values: list[float], confidence: float, alpha: float
) -> dict[str, object]:
    """Return a small ECDF/DKW audit for one target/history stratum."""
    sample = np.sort(np.asarray(values, dtype=float))
    n = int(sample.size)
    if n == 0:
        return {
            "n": 0,
            "rejection_rate": None,
            "max_ecdf_excess": None,
            "dkw_radius": None,
            "passes_dkw_diagnostic": False,
        }
    excess = float(np.max(np.arange(1, n + 1, dtype=float) / n - sample))
    radius = float(math.sqrt(math.log(2.0 / (1.0 - confidence)) / (2.0 * n)))
    return {
        "n": n,
        "rejection_rate": float(np.mean(sample <= alpha)),
        "max_ecdf_excess": max(0.0, excess),
        "dkw_radius": radius,
        "passes_dkw_diagnostic": bool(excess <= radius),
    }


def _history_key(result: SplitConfirmationResult, alpha: float) -> str:
    """Bucket only visible pre-confirmation selection history.

    The final event is the visible history immediately before the last
    selection action.  Its count and number of successes are intentionally
    coarse strata: they are inspectable without making a one-observation cell
    masquerade as a meaningful calibration estimate.
    """
    event = result.selection.events[-1]
    observations = event.visible_observations
    evidence_band = "at_or_below_alpha" if result.selection.selected_target_evidence <= alpha else "above_alpha"
    return (
        f"visible_n={len(observations)}|visible_successes={sum(observations)}"
        f"|selected_evidence={evidence_band}"
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


def _run_cell(
    scenario: PowerScenario,
    policy: SelectionPolicySpec,
    seed_pairs: list[tuple[int, int]],
    alpha: float,
    workers: int,
) -> list[SplitConfirmationResult]:
    def run(pair: tuple[int, int]) -> SplitConfirmationResult:
        return run_split_stream_confirmation(
            scenario,
            policy,
            selection_seed=pair[0],
            confirmation_seed=pair[1],
            alpha=alpha,
        )

    if workers == 1:
        return [run(pair) for pair in seed_pairs]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run, seed_pairs))


def evaluate_split_stream_sensitivity(
    scenario: PowerScenario,
    plan: SplitStreamSensitivityPlan,
    n_reps: int,
    *,
    seed: int = 0,
    workers: int = 1,
) -> dict[str, object]:
    """Audit split-confirmation p-values by selected target and history.

    The supplied process must be all null.  Every cell reuses the same ordered
    selection/confirmation seed pairs, so the comparison across selection
    durations has common random numbers.  The results state their exact
    scenario and plan hashes and retain both independent stream hashes for
    every replication.
    """
    if np.any(np.asarray(scenario.process.target_truth, dtype=bool)):
        raise ValueError("split-stream sensitivity requires an all-null process")
    n_reps = _positive_integer(n_reps, "n_reps")
    if isinstance(workers, bool) or int(workers) != workers or int(workers) < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    pairs = _seed_pairs(seed, n_reps)
    cells: list[dict[str, object]] = []
    for steps in plan.selection_steps:
        policy = plan.policy_for(steps)
        results = _run_cell(scenario, policy, pairs, plan.alpha, workers)
        by_target_history: dict[int, dict[str, list[float]]] = {
            target: {} for target in range(scenario.setting.family_size)
        }
        for result in results:
            target = int(result.selection.selected_target)
            history = _history_key(result, plan.alpha)
            by_target_history[target].setdefault(history, []).append(
                float(result.confirmation_p_value)
            )
        target_checks: list[dict[str, object]] = []
        for target in range(scenario.setting.family_size):
            history_checks = {
                history: _pvalue_diagnostic(values, plan.confidence, plan.alpha)
                for history, values in sorted(by_target_history[target].items())
            }
            target_values = [
                value
                for values in by_target_history[target].values()
                for value in values
            ]
            target_checks.append(
                {
                    "target": target,
                    "pooled_over_visible_history": _pvalue_diagnostic(
                        target_values, plan.confidence, plan.alpha
                    ),
                    "by_visible_selection_history": history_checks,
                }
            )
        stream_hashes = [
            {
                "selection": result.selection_stream_hash,
                "confirmation": result.confirmation_stream_hash,
            }
            for result in results
        ]
        cells.append(
            {
                "selection_steps": steps,
                "selection_policy": policy.to_mapping(),
                "selection_policy_hash": policy.configuration_hash,
                "n_reps": n_reps,
                "stream_pairs": stream_hashes,
                "stream_pair_schedule_hash": _hash_mapping(stream_hashes),
                "independent_streams": all(
                    pair["selection"] != pair["confirmation"] for pair in stream_hashes
                ),
                "selected_target_counts": [
                    int(sum(result.selection.selected_target == target for result in results))
                    for target in range(scenario.setting.family_size)
                ],
                "conditional_null_checks": target_checks,
                "pooled": _pvalue_diagnostic(
                    [float(result.confirmation_p_value) for result in results],
                    plan.confidence,
                    plan.alpha,
                ),
                "validity": results[0].validity,
                "validity_assumptions": list(results[0].validity_assumptions),
                "representative": results[0].to_mapping(),
            }
        )
    return {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "all_null": True,
        "n_reps": n_reps,
        "seed": int(seed),
        "seed_schedule_hash": _hash_mapping(pairs),
        "workers": workers,
        "plan": plan.to_mapping(),
        "plan_hash": plan.configuration_hash,
        "common_random_numbers": {
            "enabled": True,
            "shared_selection_confirmation_seed_pairs_across_cells": True,
        },
        "cells": cells,
        "evidence_class": "calibration_diagnostic",
        "claim_boundary": (
            "Target- and visible-history-stratified finite calibration diagnostics "
            "under the declared all-null static channel. They do not prove "
            "selective-inference validity outside the independent split-stream contract."
        ),
    }


__all__ = ["SplitStreamSensitivityPlan", "evaluate_split_stream_sensitivity"]
