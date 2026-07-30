"""Factorial sensitivity surfaces for declared testing settings.

This module keeps two distinct decision interfaces explicitly separate:

* ``fixed_family_pvalues`` applies a declared fixed-family multiple-testing
  correction to one shared two-groups p-value vector per replication.
* ``action_loop_posterior`` and ``action_loop_e_process`` evaluate policies in
  a controlled action environment whose :class:`~.scenario.SettingSpec`
  correction remains ``"none"``.

The separation is deliberate.  A posterior or e-value action decision is not
made valid by attaching a family p-value correction to an action trace.  The
surface therefore reports separate operating characteristics and never
constructs a pooled utility score or a policy ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from typing import Any, Iterable, Literal, Sequence, cast

import numpy as np

from ._types import TwoGroupsConfig
from .action_loop import ActionPolicy, run_action_loop
from .action_statistics import metrics_for_trace
from .corrections import apply_correction
from .error_rates import classify
from .scenario import (
    PowerScenario,
    make_reference_action_scenario,
    make_static_binary_e_process_scenario,
)
from .simulation import generate_pvalues
from .uncertainty import mean_interval, required_reps_for_half_width, summarize_bounded


EvidenceInterface = Literal[
    "fixed_family_pvalues",
    "action_loop_posterior",
    "action_loop_e_process",
]

FIXED_FAMILY_PVALUES: EvidenceInterface = "fixed_family_pvalues"
ACTION_LOOP_POSTERIOR: EvidenceInterface = "action_loop_posterior"
ACTION_LOOP_E_PROCESS: EvidenceInterface = "action_loop_e_process"

_EVIDENCE_INTERFACES = {
    FIXED_FAMILY_PVALUES,
    ACTION_LOOP_POSTERIOR,
    ACTION_LOOP_E_PROCESS,
}
_FIXED_FAMILY_CORRECTIONS = {
    "bonferroni",
    "sidak",
    "holm",
    "hochberg",
    "benjamini_hochberg",
    "benjamini_yekutieli",
}
_POSTERIOR_ACTION_POLICIES = {
    "fixed_horizon",
    "posterior_threshold",
    "information_gain",
    "cost_aware",
    "adaptive_target",
    "posterior_sampling",
}


def _hash_mapping(value: dict[str, Any]) -> str:
    """Return a canonical SHA-256 identity for one JSON-safe contract."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _positive_integer(value: Any, name: str) -> int:
    """Return a positive integer or raise a source-located contract error."""
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    """Return a non-negative integer or raise a source-located contract error."""
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_probability(value: Any, name: str) -> float:
    """Normalize a strictly interior probability."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must lie in (0, 1)")
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return number


def _finite_positive(value: Any, name: str) -> float:
    """Normalize a finite, strictly positive quantity."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _unique(values: Iterable[object], name: str) -> tuple[object, ...]:
    """Return a non-empty sequence with duplicate values rejected."""
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _two_groups_mapping(config: TwoGroupsConfig) -> dict[str, Any]:
    """Serialize the complete fixed-family data-generating contract."""
    return {
        "m": int(config.m),
        "pi0": float(config.pi0),
        "effect": float(config.effect),
        "rho": float(config.rho),
        "sided": int(config.sided),
        "seed": int(config.seed),
        "dependence_kind": str(config.dependence_kind),
        "block_size": int(config.block_size),
    }


def _copy_two_groups_config(config: TwoGroupsConfig) -> TwoGroupsConfig:
    """Copy the mutable two-groups input before it enters a frozen plan."""
    values = _two_groups_mapping(config)
    return TwoGroupsConfig(**values)


@dataclass(frozen=True)
class TestingSettingSensitivityPlan:
    """Predeclared factorial design for testing-setting sensitivity.

    ``fixed_family_corrections`` is intentionally a factor only for the
    fixed-family p-value interface.  Action-loop rows record
    ``fixed_family_correction=None`` and keep ``SettingSpec.correction`` equal
    to ``"none"``.  This prevents a surface table from silently conflating
    family-level p-value procedures with posterior/e-value action decisions.
    """

    __test__ = False

    alpha_grid: tuple[float, ...] = (0.05,)
    horizon_grid: tuple[int, ...] = (12,)
    stopping_budget_grid: tuple[float, ...] = (12.0,)
    evidence_interfaces: tuple[EvidenceInterface, ...] = (
        FIXED_FAMILY_PVALUES,
        ACTION_LOOP_POSTERIOR,
        ACTION_LOOP_E_PROCESS,
    )
    fixed_family_corrections: tuple[str, ...] = (
        "bonferroni",
        "benjamini_hochberg",
    )
    action_policies: tuple[str, ...] = (
        "fixed_horizon",
        "posterior_threshold",
        "cost_aware",
    )
    fixed_family_config: TwoGroupsConfig = field(
        default_factory=lambda: TwoGroupsConfig(m=200, pi0=0.8, effect=3.0)
    )
    action_reliability: float = 0.8
    n_reps: int = 1_000
    seed: int = 0
    confidence: float = 0.95
    target_mc_half_width: float | None = None

    def __post_init__(self) -> None:
        """Validate and normalize all predeclared factorial dimensions."""
        alpha_values = tuple(
            _finite_probability(value, "alpha_grid entry")
            for value in _unique(self.alpha_grid, "alpha_grid")
        )
        horizon_values = tuple(
            _positive_integer(value, "horizon_grid entry")
            for value in _unique(self.horizon_grid, "horizon_grid")
        )
        budget_values = tuple(
            _finite_positive(value, "stopping_budget_grid entry")
            for value in _unique(self.stopping_budget_grid, "stopping_budget_grid")
        )
        interface_candidates = _unique(self.evidence_interfaces, "evidence_interfaces")
        if any(not isinstance(value, str) for value in interface_candidates):
            raise ValueError("evidence_interfaces entries must be canonical strings")
        interface_values = tuple(cast(str, value) for value in interface_candidates)
        invalid_interfaces = sorted(set(interface_values) - _EVIDENCE_INTERFACES)
        if invalid_interfaces:
            raise ValueError(f"unsupported evidence interfaces: {invalid_interfaces}")
        correction_candidates = _unique(
            self.fixed_family_corrections, "fixed_family_corrections"
        )
        if any(not isinstance(value, str) for value in correction_candidates):
            raise ValueError("fixed_family_corrections entries must be canonical strings")
        corrections = tuple(cast(str, value) for value in correction_candidates)
        invalid_corrections = sorted(set(corrections) - _FIXED_FAMILY_CORRECTIONS)
        if invalid_corrections:
            raise ValueError(
                "fixed_family_corrections must name declared non-adaptive "
                f"fixed-family procedures, got {invalid_corrections}"
            )
        policy_candidates = _unique(self.action_policies, "action_policies")
        if any(not isinstance(value, str) for value in policy_candidates):
            raise ValueError("action_policies entries must be canonical strings")
        policies = tuple(cast(str, value) for value in policy_candidates)
        invalid_policies = sorted(set(policies) - _POSTERIOR_ACTION_POLICIES)
        if invalid_policies:
            raise ValueError(
                "action_policies must contain posterior/action policies only; "
                f"e_process belongs to the action_loop_e_process interface: {invalid_policies}"
            )
        if not isinstance(self.fixed_family_config, TwoGroupsConfig):
            raise ValueError("fixed_family_config must be a TwoGroupsConfig")
        fixed_config = _copy_two_groups_config(self.fixed_family_config)
        if fixed_config.m < 1:
            raise ValueError("fixed_family_config.m must be positive for a sensitivity surface")
        reliability = _finite_probability(self.action_reliability, "action_reliability")
        if reliability <= 0.5:
            raise ValueError("action_reliability must lie in (0.5, 1)")
        n_reps = _positive_integer(self.n_reps, "n_reps")
        seed = _nonnegative_integer(self.seed, "seed")
        confidence = _finite_probability(self.confidence, "confidence")
        half_width = self.target_mc_half_width
        if half_width is not None:
            half_width = _finite_probability(half_width, "target_mc_half_width")

        object.__setattr__(self, "alpha_grid", alpha_values)
        object.__setattr__(self, "horizon_grid", horizon_values)
        object.__setattr__(self, "stopping_budget_grid", budget_values)
        object.__setattr__(self, "evidence_interfaces", interface_values)
        object.__setattr__(self, "fixed_family_corrections", corrections)
        object.__setattr__(self, "action_policies", policies)
        object.__setattr__(self, "fixed_family_config", fixed_config)
        object.__setattr__(self, "action_reliability", reliability)
        object.__setattr__(self, "n_reps", n_reps)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "target_mc_half_width", half_width)

    def to_mapping(self) -> dict[str, Any]:
        """Return the serializable preregistered factorial contract."""
        return {
            "alpha_grid": list(self.alpha_grid),
            "horizon_grid": list(self.horizon_grid),
            "stopping_budget_grid": list(self.stopping_budget_grid),
            "evidence_interfaces": list(self.evidence_interfaces),
            "fixed_family_corrections": list(self.fixed_family_corrections),
            "action_policies": list(self.action_policies),
            "fixed_family_config": _two_groups_mapping(self.fixed_family_config),
            "action_reliability": self.action_reliability,
            "n_reps": self.n_reps,
            "seed": self.seed,
            "confidence": self.confidence,
            "target_mc_half_width": self.target_mc_half_width,
        }

    @property
    def configuration_hash(self) -> str:
        """Return the immutable identity of the declared sensitivity plan."""
        return _hash_mapping(self.to_mapping())


def _stream_contract(
    *,
    cell_seed: int,
    n_reps: int,
    cell_settings: dict[str, Any],
) -> tuple[dict[str, Any], list[int]]:
    """Create one stable child-stream schedule shared by every cell policy."""
    children = np.random.SeedSequence(cell_seed).spawn(n_reps)
    replication_seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in children
    ]
    stream_input = {
        "cell_seed": int(cell_seed),
        "n_reps": int(n_reps),
        "cell_settings": cell_settings,
        "derivation": "SeedSequence(cell_seed).spawn(n_reps) in ascending replication order",
    }
    stream_hash = _hash_mapping(stream_input)
    return (
        {
            **stream_input,
            "stream_hash": stream_hash,
            "policy_pairing": "Every policy in this declared interface cell uses these same replication seeds.",
        },
        replication_seeds,
    )


def _bounded_metric(
    values: Sequence[float], plan: TestingSettingSensitivityPlan
) -> dict[str, Any]:
    """Return a separate bounded-operating-characteristic summary."""
    summary = summarize_bounded(
        values,
        confidence=plan.confidence,
        target_half_width=plan.target_mc_half_width,
    )
    return {
        "applicable": True,
        "estimate": float(summary["mean"]),
        "interval": [
            float(summary["normal_ci_lower"]),
            float(summary["normal_ci_upper"]),
        ],
        "standard_error": float(summary["standard_error"]),
        "interval_method": str(summary["interval_method"]),
        "n_reps": int(summary["n"]),
    }


def _numeric_metric(values: Sequence[float], plan: TestingSettingSensitivityPlan) -> dict[str, Any]:
    """Return a separate unbounded cost or stopping-time summary."""
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size < 1 or not np.all(np.isfinite(sample)):
        raise ValueError("numeric operating-characteristic samples must be finite")
    interval = mean_interval(sample, plan.confidence)
    standard_error = (
        0.0
        if sample.size == 1
        else float(np.std(sample, ddof=1) / math.sqrt(sample.size))
    )
    return {
        "applicable": True,
        "estimate": float(np.mean(sample)),
        "interval": [float(interval[0]), float(interval[1])],
        "standard_error": standard_error,
        "interval_method": "student_t_monte_carlo_mean",
        "n_reps": int(sample.size),
    }


def _not_applicable_metric(reason: str) -> dict[str, Any]:
    """Represent an inapplicable estimand without pretending it equals zero."""
    return {
        "applicable": False,
        "estimate": None,
        "interval": None,
        "standard_error": None,
        "interval_method": None,
        "n_reps": None,
        "reason": reason,
    }


def _fixed_family_record(
    *,
    plan: TestingSettingSensitivityPlan,
    alpha: float,
    horizon: int,
    stopping_budget: float,
    correction: str,
    stream: dict[str, Any],
    replication_seeds: Sequence[int],
) -> dict[str, Any]:
    """Score one fixed-family correction on the cell's shared p-value draws."""
    draws = [
        generate_pvalues(plan.fixed_family_config, np.random.default_rng(seed))
        for seed in replication_seeds
    ]
    summaries = [
        classify(apply_correction(correction, pvalues, alpha).rejected, is_alt)
        for pvalues, is_alt in draws
    ]
    power = [float(summary.power) for summary in summaries]
    fdr = [float(summary.fdp) for summary in summaries]
    fwer = [float(summary.fwer_event) for summary in summaries]
    setting = {
        "alpha": alpha,
        "horizon": horizon,
        "stopping_budget": stopping_budget,
        "evidence_interface": FIXED_FAMILY_PVALUES,
        "fixed_family_correction": correction,
        "action_policy": None,
    }
    scenario_contract = {
        "interface": FIXED_FAMILY_PVALUES,
        "two_groups_config": _two_groups_mapping(plan.fixed_family_config),
    }
    return {
        "settings": setting,
        "settings_hash": _hash_mapping(setting),
        "scenario_id": "two_groups_fixed_family",
        "scenario_hash": _hash_mapping(scenario_contract),
        "interface": {
            "name": FIXED_FAMILY_PVALUES,
            "kind": "fixed_family",
            "evidence": "one p-value vector per replication",
            "action_loop_correction": "not applicable",
        },
        "policy": correction,
        "fixed_family_correction": correction,
        "action_policy": None,
        "common_random_numbers": {
            "stream_hash": stream["stream_hash"],
            "paired_with": "other fixed-family corrections in this setting cell",
        },
        "metrics": {
            "power": _bounded_metric(power, plan),
            "fdr": _bounded_metric(fdr, plan),
            "fwer": _bounded_metric(fwer, plan),
            "cost": _not_applicable_metric(
                "fixed-family p-value procedures do not take environment actions"
            ),
            "stopping": _not_applicable_metric(
                "fixed-family p-value procedures have no sequential action stopping time"
            ),
        },
        "mean_rejections": float(np.mean([summary.num_rejected for summary in summaries])),
        "validity_boundary": (
            "Fixed-family operating characteristics remain conditional on the "
            "declared two-groups model, p-value validity, and correction assumptions."
        ),
    }


def _action_scenario(
    *,
    plan: TestingSettingSensitivityPlan,
    alpha: float,
    horizon: int,
    stopping_budget: float,
    interface: EvidenceInterface,
    policy: str,
) -> PowerScenario:
    """Build one action scenario while preserving ``correction='none'``."""
    if interface == ACTION_LOOP_POSTERIOR:
        base = make_reference_action_scenario(
            reliability=plan.action_reliability,
            horizon=horizon,
            alpha=alpha,
        )
    elif interface == ACTION_LOOP_E_PROCESS:
        base = make_static_binary_e_process_scenario(
            reliability=plan.action_reliability,
            horizon=horizon,
            alpha=alpha,
        )
    else:
        raise ValueError(f"{interface!r} is not an action-loop evidence interface")
    setting = replace(
        base.setting,
        alpha=alpha,
        max_steps=horizon,
        cost_budget=stopping_budget,
        stopping_policy=policy,
        correction="none",
    )
    return replace(base, setting=setting)


def _action_record(
    *,
    plan: TestingSettingSensitivityPlan,
    alpha: float,
    horizon: int,
    stopping_budget: float,
    interface: EvidenceInterface,
    policy: str,
    stream: dict[str, Any],
    replication_seeds: Sequence[int],
) -> dict[str, Any]:
    """Score one action policy on the cell's shared environment streams."""
    scenario = _action_scenario(
        plan=plan,
        alpha=alpha,
        horizon=horizon,
        stopping_budget=stopping_budget,
        interface=interface,
        policy=policy,
    )
    traces = [
        run_action_loop(scenario, cast(ActionPolicy, policy), seed=seed)
        for seed in replication_seeds
    ]
    metrics = [metrics_for_trace(scenario, trace) for trace in traces]
    setting = {
        "alpha": alpha,
        "horizon": horizon,
        "stopping_budget": stopping_budget,
        "evidence_interface": interface,
        "fixed_family_correction": None,
        "action_policy": policy,
    }
    return {
        "settings": setting,
        "settings_hash": _hash_mapping(setting),
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "interface": {
            "name": interface,
            "kind": "action_loop",
            "evidence": (
                "posterior/action decision trace"
                if interface == ACTION_LOOP_POSTERIOR
                else "state-invariant binary e-process trace"
            ),
            "action_loop_correction": "none",
        },
        "policy": policy,
        "fixed_family_correction": None,
        "action_policy": policy,
        "common_random_numbers": {
            "stream_hash": stream["stream_hash"],
            "paired_with": "other action policies in this setting cell",
        },
        "metrics": {
            "power": _bounded_metric(
                [metric.true_positive_rate for metric in metrics], plan
            ),
            "fdr": _bounded_metric(
                [metric.false_discovery_proportion for metric in metrics], plan
            ),
            "fwer": _bounded_metric(
                [metric.family_wise_error for metric in metrics], plan
            ),
            "cost": _numeric_metric([metric.total_cost for metric in metrics], plan),
            "stopping": _numeric_metric([metric.step_count for metric in metrics], plan),
        },
        "mean_rejections": float(np.mean([metric.rejection_count for metric in metrics])),
        "validity_boundary": (
            "Action-loop operating characteristics are conditional on the "
            "declared model, process, setting, policy, and finite simulation; "
            "they do not inherit fixed-family correction validity."
        ),
        "action_loop_validity": sorted({trace.validity_boundary for trace in traces}),
    }


def evaluate_testing_setting_sensitivity(
    plan: TestingSettingSensitivityPlan,
) -> dict[str, Any]:
    """Evaluate a deterministic, common-random-number factorial surface.

    The returned rows contain only separate operating characteristics.  In
    particular, no cross-metric utility, weighted score, or policy ranking is
    produced.  Policy contrasts are paired only within one declared interface
    cell because each such policy receives the identical stable child-stream
    schedule.
    """
    if not isinstance(plan, TestingSettingSensitivityPlan):
        raise ValueError("plan must be a TestingSettingSensitivityPlan")

    cells: list[dict[str, Any]] = [
        {
            "alpha": alpha,
            "horizon": horizon,
            "stopping_budget": stopping_budget,
            "evidence_interface": interface,
        }
        for alpha in plan.alpha_grid
        for horizon in plan.horizon_grid
        for stopping_budget in plan.stopping_budget_grid
        for interface in plan.evidence_interfaces
    ]
    master_children = np.random.SeedSequence(plan.seed).spawn(len(cells))
    rows: list[dict[str, Any]] = []
    streams: dict[str, dict[str, Any]] = {}

    for cell, child in zip(cells, master_children):
        interface = cast(EvidenceInterface, cell["evidence_interface"])
        cell_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        stream, replication_seeds = _stream_contract(
            cell_seed=cell_seed,
            n_reps=plan.n_reps,
            cell_settings=cell,
        )
        streams[str(stream["stream_hash"])] = stream
        if interface == FIXED_FAMILY_PVALUES:
            rows.extend(
                _fixed_family_record(
                    plan=plan,
                    alpha=float(cell["alpha"]),
                    horizon=int(cell["horizon"]),
                    stopping_budget=float(cell["stopping_budget"]),
                    correction=correction,
                    stream=stream,
                    replication_seeds=replication_seeds,
                )
                for correction in plan.fixed_family_corrections
            )
        elif interface == ACTION_LOOP_POSTERIOR:
            rows.extend(
                _action_record(
                    plan=plan,
                    alpha=float(cell["alpha"]),
                    horizon=int(cell["horizon"]),
                    stopping_budget=float(cell["stopping_budget"]),
                    interface=ACTION_LOOP_POSTERIOR,
                    policy=policy,
                    stream=stream,
                    replication_seeds=replication_seeds,
                )
                for policy in plan.action_policies
            )
        elif interface == ACTION_LOOP_E_PROCESS:
            rows.append(
                _action_record(
                    plan=plan,
                    alpha=float(cell["alpha"]),
                    horizon=int(cell["horizon"]),
                    stopping_budget=float(cell["stopping_budget"]),
                    interface=ACTION_LOOP_E_PROCESS,
                    policy="e_process",
                    stream=stream,
                    replication_seeds=replication_seeds,
                )
            )
        else:  # Defensive because the plan validates interface labels.
            raise RuntimeError(f"unrecognized evidence interface {interface!r}")

    required_for_target = (
        required_reps_for_half_width(plan.target_mc_half_width, plan.confidence)
        if plan.target_mc_half_width is not None
        else None
    )
    return {
        "contract_version": "1.0",
        "plan": plan.to_mapping(),
        "plan_hash": plan.configuration_hash,
        "sample_plan": {
            "n_reps_per_policy_cell": plan.n_reps,
            "confidence": plan.confidence,
            "target_mc_half_width": plan.target_mc_half_width,
            "required_reps_for_target_half_width": required_for_target,
            "factorial_interface_cells": len(cells),
            "surface_rows": len(rows),
            "common_random_numbers": (
                "Stable SeedSequence child streams are shared by policies within "
                "each declared interface setting cell."
            ),
            "metrics_reported_separately": ["power", "fdr", "fwer", "cost", "stopping"],
        },
        "streams": streams,
        "surface": rows,
        "claim_boundary": (
            "Every result is a finite-simulation operating characteristic of a "
            "declared interface and setting. Fixed-family corrections and "
            "action-loop decisions are reported separately; no pooled utility or "
            "policy ranking is computed."
        ),
        "pooled_utility_or_policy_ranking": "not_reported",
    }


__all__ = [
    "ACTION_LOOP_E_PROCESS",
    "ACTION_LOOP_POSTERIOR",
    "FIXED_FAMILY_PVALUES",
    "EvidenceInterface",
    "TestingSettingSensitivityPlan",
    "evaluate_testing_setting_sensitivity",
]
