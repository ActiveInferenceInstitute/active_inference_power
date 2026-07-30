#!/usr/bin/env python3
"""Build a bounded, non-promoted research atlas from the R1--R5 and G4 lanes.

This is deliberately a thin orchestration layer.  The statistical calculations
remain in the corresponding :mod:`active_inference_power` domain modules; this
script only freezes a compact plan, invokes those APIs, creates readable
figures from their returned mappings, and publishes one isolated artifact
bundle.  It is not a release generator, a manuscript hydrator, or a promotion
gate.

The output root is intentionally required and must be new.  The script stages
every artifact beside that root and atomically renames the complete directory
only after JSON, figures, registry, dashboard payload, and manifest are all
present and hash-bound.
"""

from __future__ import annotations

# Imports below intentionally follow the script-local path bootstrap so this
# orchestrator works both from the checkout and via an absolute script path.
# ruff: noqa: E402

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence, cast

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from active_inference_power._types import TwoGroupsConfig
from active_inference_power.dynamic_process_sensitivity import (
    DynamicProcessSensitivityPlan,
    evaluate_dynamic_process_sensitivity,
)
from active_inference_power.learned_policy import LearnedPolicyConfig
from active_inference_power.multi_target_environment import (
    MultiTargetEnvironmentSpec,
    evaluate_multi_target_control_pair,
)
from active_inference_power.repeated_policy_evaluation import (
    RepeatedLearnedPolicyPlan,
    evaluate_repeated_learned_policy,
)
from active_inference_power.scenario import (
    make_reference_action_scenario,
    make_static_null_binary_e_process_scenario,
)
from active_inference_power.selection_sensitivity import (
    SplitStreamSensitivityPlan,
    evaluate_split_stream_sensitivity,
)
from active_inference_power.sequential_atlas import (
    SequentialEvidenceAtlasConfig,
    evaluate_sequential_evidence_atlas,
)
from active_inference_power.testing_setting_sensitivity import (
    ACTION_LOOP_E_PROCESS,
    ACTION_LOOP_POSTERIOR,
    FIXED_FAMILY_PVALUES,
    TestingSettingSensitivityPlan,
    evaluate_testing_setting_sensitivity,
)
from scripts.artifacts import (
    artifact_record,
    atomic_write_json,
    require_safe_output_root,
    sha256_file,
)


ATLAS_CONTRACT_VERSION = "1.0"
ATLAS_STATUS = "research_only_not_for_promotion_or_release"
LANE_IDS = ("R1", "R2", "R3", "R4", "R5", "G4")
_SOURCE_FILES = {
    "orchestrator": Path(__file__),
    "R1": ROOT / "src/active_inference_power/testing_setting_sensitivity.py",
    "R2": ROOT / "src/active_inference_power/selection_sensitivity.py",
    "R3": ROOT / "src/active_inference_power/dynamic_process_sensitivity.py",
    "R4": ROOT / "src/active_inference_power/sequential_atlas.py",
    "R5": ROOT / "src/active_inference_power/repeated_policy_evaluation.py",
    "G4": ROOT / "src/active_inference_power/multi_target_environment.py",
}


def _positive_integer(value: object, name: str) -> int:
    """Normalize a strictly positive integer CLI or plan value."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        normalized = int(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if normalized < 1 or normalized != value:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _nonnegative_integer(value: object, name: str) -> int:
    """Normalize a non-negative integer CLI or plan value."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        normalized = int(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if normalized < 0 or normalized != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _canonical_hash(value: Mapping[str, Any]) -> str:
    """Return a canonical SHA-256 identity for one JSON-safe mapping."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResearchAtlasPlan:
    """Small but nontrivial, predeclared research-atlas execution contract.

    The plan is intentionally compact enough for a review lane.  It is not a
    precision campaign, theorem check, release bundle, or publication claim.
    A later frozen research configuration can serialize this exact mapping and
    increase replication budgets without changing the script's artifact shape.
    """

    seed: int = 20_260_725
    workers: int = 1
    r1_n_reps: int = 4
    r2_n_reps: int = 4
    r3_n_reps: int = 3
    r4_n_reps: int = 6
    r5_train_reps: int = 2
    r5_evaluation_reps: int = 2
    r5_partitions: int = 2
    g4_n_reps: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        object.__setattr__(self, "workers", _positive_integer(self.workers, "workers"))
        for name in (
            "r1_n_reps",
            "r2_n_reps",
            "r3_n_reps",
            "r4_n_reps",
            "r5_train_reps",
            "r5_evaluation_reps",
            "g4_n_reps",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        partitions = _positive_integer(self.r5_partitions, "r5_partitions")
        if partitions < 2:
            raise ValueError("r5_partitions must be at least 2 for repeated-policy bands")
        object.__setattr__(self, "r5_partitions", partitions)

    def to_mapping(self) -> dict[str, Any]:
        """Return the fully declared sample-size and execution contract."""
        return {
            "seed": self.seed,
            "workers": self.workers,
            "lanes": {
                "R1": {
                    "n_reps": self.r1_n_reps,
                    "grid": {
                        "alpha": [0.025, 0.05],
                        "horizon": [3],
                        "stopping_budget": [4.0],
                        "evidence_interfaces": [
                            FIXED_FAMILY_PVALUES,
                            ACTION_LOOP_POSTERIOR,
                            ACTION_LOOP_E_PROCESS,
                        ],
                        "fixed_family_corrections": ["bonferroni", "benjamini_hochberg"],
                        "action_policies": ["fixed_horizon", "cost_aware"],
                    },
                },
                "R2": {
                    "n_reps": self.r2_n_reps,
                    "selection_steps": [1, 3],
                    "confirmation_steps": 4,
                },
                "R3": {"n_reps": self.r3_n_reps, "horizon": 3},
                "R4": {"n_reps": self.r4_n_reps, "horizon": 5},
                "R5": {
                    "partitions": self.r5_partitions,
                    "train_reps": self.r5_train_reps,
                    "evaluation_reps": self.r5_evaluation_reps,
                    "candidate_count": 2,
                },
                "G4": {"n_reps": self.g4_n_reps, "horizon": 4, "targets": 3},
            },
            "status": ATLAS_STATUS,
        }


DEFAULT_RESEARCH_ATLAS_PLAN = ResearchAtlasPlan()


def _lane_seed_schedule(seed: int) -> dict[str, int]:
    """Derive stable independent seed roots for the six evidence lanes."""
    master = _nonnegative_integer(seed, "seed")
    children = np.random.SeedSequence(master).spawn(len(LANE_IDS))
    return {
        lane: int(child.generate_state(1, dtype=np.uint32)[0])
        for lane, child in zip(LANE_IDS, children)
    }


def _domain_contracts(plan: ResearchAtlasPlan, lane_seeds: Mapping[str, int]) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """Construct all declared source-domain inputs without mutating shared config."""
    r1 = TestingSettingSensitivityPlan(
        alpha_grid=(0.025, 0.05),
        horizon_grid=(3,),
        stopping_budget_grid=(4.0,),
        evidence_interfaces=(
            FIXED_FAMILY_PVALUES,
            ACTION_LOOP_POSTERIOR,
            ACTION_LOOP_E_PROCESS,
        ),
        fixed_family_corrections=("bonferroni", "benjamini_hochberg"),
        action_policies=("fixed_horizon", "cost_aware"),
        fixed_family_config=TwoGroupsConfig(
            m=24,
            pi0=0.75,
            effect=2.5,
            dependence_kind="independent",
        ),
        action_reliability=0.74,
        n_reps=plan.r1_n_reps,
        seed=int(lane_seeds["R1"]),
        confidence=0.9,
        target_mc_half_width=0.35,
    )
    r2_scenario = make_static_null_binary_e_process_scenario(
        reliability=0.74,
        horizon=3,
        alpha=0.05,
    )
    r2 = SplitStreamSensitivityPlan(
        selection_steps=(1, 3),
        confirmation_steps=4,
        strategy="minimum_p",
        evidence_observable=True,
        alpha=0.05,
        confidence=0.9,
    )
    shared_action_scenario = make_reference_action_scenario(
        reliability=0.74,
        horizon=3,
        alpha=0.05,
        scenario_id="bounded_research_atlas_reference",
    )
    r3 = DynamicProcessSensitivityPlan(
        horizon=3,
        context_persistence=0.8,
        reliability_end_multiplier=0.7,
        mcar_missing_end=0.15,
        mar_missing_low=0.05,
        mar_missing_high=0.2,
        mnar_missing_low=0.05,
        mnar_missing_high=0.2,
        policy="cost_aware",
        calibration_policy="adaptive_target",
    )
    r4 = SequentialEvidenceAtlasConfig(
        null_rates=(0.4,),
        alternative_rates=(0.7,),
        horizon=5,
        n_reps=plan.r4_n_reps,
        alpha=0.05,
        confidence=0.9,
        mixture_alternative_rates=(0.6, 0.8),
    )
    r5_plan = RepeatedLearnedPolicyPlan(
        partitions=plan.r5_partitions,
        evaluation_reps=plan.r5_evaluation_reps,
        reference_policy="cost_aware",
        confidence=0.9,
    )
    r5_search = LearnedPolicyConfig(
        train_reps=plan.r5_train_reps,
        cost_penalty=0.05,
        candidate_parameters=((1.0, 0.5, 0.0), (0.5, 1.0, 0.1)),
    )
    g4 = MultiTargetEnvironmentSpec(
        target_ids=("target_0", "target_1", "target_2"),
        arrival_times=(0, 1, 2),
        target_truth=(True, False, True),
        reliability_schedule=(0.72, 0.78, 0.74, 0.8),
        action_cost_schedule=(1.0, 1.2, 1.1, 1.3),
        horizon=4,
        confirmation_steps=4,
        alpha=0.05,
        policy="minimum_visible_p",
        precise_reliability_bonus=0.08,
        precise_cost_multiplier=1.8,
    )
    inputs = {
        "R1": {"plan": r1, "seed": int(lane_seeds["R1"])},
        "R2": {"scenario": r2_scenario, "plan": r2, "seed": int(lane_seeds["R2"])},
        "R3": {"scenario": shared_action_scenario, "plan": r3, "seed": int(lane_seeds["R3"])},
        "R4": {"config": r4, "seed": int(lane_seeds["R4"])},
        "R5": {
            "scenario": shared_action_scenario,
            "plan": r5_plan,
            "search": r5_search,
            "seed": int(lane_seeds["R5"]),
        },
        "G4": {"spec": g4, "seed": int(lane_seeds["G4"])},
    }
    contracts = {
        "R1": {"plan": r1.to_mapping(), "plan_hash": r1.configuration_hash},
        "R2": {
            "scenario": r2_scenario.to_mapping(),
            "scenario_hash": r2_scenario.configuration_hash,
            "plan": r2.to_mapping(),
            "plan_hash": r2.configuration_hash,
        },
        "R3": {
            "scenario": shared_action_scenario.to_mapping(),
            "scenario_hash": shared_action_scenario.configuration_hash,
            "plan": r3.to_mapping(),
            "plan_hash": r3.configuration_hash,
        },
        "R4": {"config": r4.to_mapping(), "config_hash": r4.configuration_hash},
        "R5": {
            "scenario": shared_action_scenario.to_mapping(),
            "scenario_hash": shared_action_scenario.configuration_hash,
            "plan": r5_plan.to_mapping(),
            "plan_hash": r5_plan.configuration_hash,
            "search_config": r5_search.to_mapping(),
            "search_config_hash": r5_search.configuration_hash,
        },
        "G4": {"environment": g4.to_mapping(), "environment_hash": g4.configuration_hash},
    }
    return inputs, contracts


def _source_hashes() -> dict[str, str]:
    """Hash every source module whose public API defines this atlas contract."""
    missing = [str(path) for path in _SOURCE_FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"research-atlas source file is missing: {missing}")
    return {name: sha256_file(path) for name, path in _SOURCE_FILES.items()}


def build_research_atlas(plan: ResearchAtlasPlan) -> dict[str, Any]:
    """Evaluate every declared lane and return one JSON-safe atlas mapping.

    This function deliberately performs no file I/O.  Its return value carries
    plan, source, scenario, and configuration identities so a renderer or a
    later frozen research configuration can reuse the same computation contract.
    """
    if not isinstance(plan, ResearchAtlasPlan):
        raise ValueError("plan must be a ResearchAtlasPlan")
    # ``to_mapping`` intentionally contains only a documented nested sample
    # plan; reconstructing field-by-field avoids accepting a low-level-mutated
    # frozen dataclass as an execution contract.
    normalized = ResearchAtlasPlan(
        seed=plan.seed,
        workers=plan.workers,
        r1_n_reps=plan.r1_n_reps,
        r2_n_reps=plan.r2_n_reps,
        r3_n_reps=plan.r3_n_reps,
        r4_n_reps=plan.r4_n_reps,
        r5_train_reps=plan.r5_train_reps,
        r5_evaluation_reps=plan.r5_evaluation_reps,
        r5_partitions=plan.r5_partitions,
        g4_n_reps=plan.g4_n_reps,
    )
    lane_seeds = _lane_seed_schedule(normalized.seed)
    inputs, contracts = _domain_contracts(normalized, lane_seeds)
    workers = normalized.workers
    results = {
        "R1": evaluate_testing_setting_sensitivity(cast(TestingSettingSensitivityPlan, inputs["R1"]["plan"])),
        "R2": evaluate_split_stream_sensitivity(
            cast(Any, inputs["R2"]["scenario"]),
            cast(SplitStreamSensitivityPlan, inputs["R2"]["plan"]),
            normalized.r2_n_reps,
            seed=int(inputs["R2"]["seed"]),
            workers=workers,
        ),
        "R3": evaluate_dynamic_process_sensitivity(
            cast(Any, inputs["R3"]["scenario"]),
            cast(DynamicProcessSensitivityPlan, inputs["R3"]["plan"]),
            normalized.r3_n_reps,
            seed=int(inputs["R3"]["seed"]),
            workers=workers,
        ),
        "R4": evaluate_sequential_evidence_atlas(
            cast(SequentialEvidenceAtlasConfig, inputs["R4"]["config"]),
            seed=int(inputs["R4"]["seed"]),
            workers=workers,
        ),
        "R5": evaluate_repeated_learned_policy(
            cast(Any, inputs["R5"]["scenario"]),
            cast(LearnedPolicyConfig, inputs["R5"]["search"]),
            plan=cast(RepeatedLearnedPolicyPlan, inputs["R5"]["plan"]),
            seed=int(inputs["R5"]["seed"]),
            workers=workers,
        ),
        "G4": evaluate_multi_target_control_pair(
            cast(MultiTargetEnvironmentSpec, inputs["G4"]["spec"]),
            normalized.g4_n_reps,
            seed=int(inputs["G4"]["seed"]),
            workers=workers,
        ),
    }
    canonical_plan = {
        "contract_version": ATLAS_CONTRACT_VERSION,
        "research_plan": normalized.to_mapping(),
        "lane_seed_schedule": lane_seeds,
        "domain_contracts": contracts,
    }
    plan_hash = _canonical_hash(canonical_plan)
    body = {
        "kind": "bounded_research_atlas",
        "contract_version": ATLAS_CONTRACT_VERSION,
        "status": ATLAS_STATUS,
        "canonical_plan": canonical_plan,
        "plan_hash": plan_hash,
        "source_hashes": _source_hashes(),
        "lanes": results,
        "claim_boundary": (
            "This compact atlas reports finite, declared, source-hash-bound research "
            "results. It does not promote any lane to a theorem, release certificate, "
            "publication decision, universal policy result, or family-level validity "
            "claim beyond each lane's own stated assumptions."
        ),
    }
    return body | {
        "content_hash": _canonical_hash(body),
        "content_hash_scope": "the atlas mapping before content_hash and content_hash_scope are added",
    }


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    """Return a result mapping or fail rather than drawing an invented panel."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _list(value: object, label: str) -> list[Any]:
    """Return a result sequence or fail rather than silently omitting a lane."""
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _float(value: object, label: str) -> float:
    """Normalize a finite figure value from a source-owned result mapping."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite numeric")
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite numeric") from exc
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite numeric")
    return number


def _metric_estimate(row: Mapping[str, Any], metric: str) -> float | None:
    """Extract an applicable metric estimate without converting N/A to zero."""
    metrics = _mapping(row.get("metrics"), "R1 row metrics")
    summary = _mapping(metrics.get(metric), f"R1 {metric} summary")
    if not bool(summary.get("applicable")):
        return None
    return _float(summary.get("estimate"), f"R1 {metric} estimate")


def _metric_interval(row: Mapping[str, Any], metric: str) -> tuple[float, float] | None:
    """Extract one applicable metric interval and validate its ordering."""
    metrics = _mapping(row.get("metrics"), "R1 row metrics")
    summary = _mapping(metrics.get(metric), f"R1 {metric} summary")
    if not bool(summary.get("applicable")):
        return None
    interval = _list(summary.get("interval"), f"R1 {metric} interval")
    if len(interval) != 2:
        raise ValueError(f"R1 {metric} interval must contain lower and upper bounds")
    lower = _float(interval[0], f"R1 {metric} lower interval")
    upper = _float(interval[1], f"R1 {metric} upper interval")
    estimate = _metric_estimate(row, metric)
    if estimate is None or lower > estimate or estimate > upper:
        raise ValueError(f"R1 {metric} interval must contain its estimate")
    return lower, upper


def _save_figure_atomic(
    figure: plt.Figure,
    path: Path,
    *,
    title: str,
    description: str,
) -> None:
    """Encode one accessibility-labelled PNG atomically inside the staged bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=180,
            bbox_inches="tight",
            metadata={"Title": title, "Description": description},
        )
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def _r1_row_label(row: Mapping[str, Any]) -> str:
    """Return a short direct label for one declared R1 interface/policy row."""
    interface = _mapping(row.get("interface"), "R1 interface")
    setting = _mapping(row.get("settings"), "R1 settings")
    alpha = _float(setting.get("alpha"), "R1 alpha")
    alpha_label = f"alpha={alpha:g}"
    name = str(interface.get("name"))
    policy = str(row.get("policy"))
    if name == FIXED_FAMILY_PVALUES:
        short_policy = {
            "bonferroni": "Bonferroni",
            "benjamini_hochberg": "BH",
        }.get(policy, policy.replace("_", " "))
        return f"{short_policy}\n{alpha_label}"
    if name == ACTION_LOOP_POSTERIOR:
        short_policy = {
            "fixed_horizon": "horizon",
            "cost_aware": "cost",
        }.get(policy, policy.replace("_", " "))
        return f"{short_policy}\n{alpha_label}"
    if name == ACTION_LOOP_E_PROCESS:
        return f"e-process\n{alpha_label}"
    raise ValueError(f"unrecognized R1 interface {name!r}")


def _write_overview_figure(atlas: Mapping[str, Any], path: Path) -> dict[str, str]:
    """Render metric-separated R1 and R3 panels from source-returned values."""
    lanes = _mapping(atlas.get("lanes"), "atlas lanes")
    r1 = _mapping(lanes.get("R1"), "R1 result")
    r1_rows = [_mapping(row, "R1 surface row") for row in _list(r1.get("surface"), "R1 surface")]
    if not r1_rows:
        raise ValueError("R1 surface must not be empty")
    labels = [_r1_row_label(row) for row in r1_rows]
    power = [_metric_estimate(row, "power") for row in r1_rows]
    fdr = [_metric_estimate(row, "fdr") for row in r1_rows]
    power_intervals = [_metric_interval(row, "power") for row in r1_rows]
    fdr_intervals = [_metric_interval(row, "fdr") for row in r1_rows]
    action_rows = [row for row in r1_rows if _metric_estimate(row, "cost") is not None]
    if not action_rows:
        raise ValueError("R1 must contain action-loop rows with cost estimates")

    r3 = _mapping(lanes.get("R3"), "R3 result")
    regimes = [_mapping(value, "R3 regime") for value in _list(r3.get("regimes"), "R3 regimes")]
    if not regimes:
        raise ValueError("R3 regimes must not be empty")
    regime_labels = [str(regime.get("regime")).replace("_", "\n") for regime in regimes]
    static_cost = [
        _float(
            _mapping(_mapping(regime.get("static_control"), "R3 static").get("summary"), "R3 static summary").get("mean_cost"),
            "R3 static mean_cost",
        )
        for regime in regimes
    ]
    dynamic_cost = [
        _float(
            _mapping(_mapping(regime.get("dynamic"), "R3 dynamic").get("summary"), "R3 dynamic summary").get("mean_cost"),
            "R3 dynamic mean_cost",
        )
        for regime in regimes
    ]

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    indices = np.arange(len(labels))
    power_values = [_float(value, "R1 power") for value in power]
    fdr_values = [_float(value, "R1 FDR") for value in fdr]
    power_errors = np.asarray(
        [
            [value - _float(interval[0], "R1 power lower"), _float(interval[1], "R1 power upper") - value]
            for value, interval in zip(power_values, power_intervals)
            if interval is not None
        ],
        dtype=float,
    ).T
    fdr_errors = np.asarray(
        [
            [value - _float(interval[0], "R1 FDR lower"), _float(interval[1], "R1 FDR upper") - value]
            for value, interval in zip(fdr_values, fdr_intervals)
            if interval is not None
        ],
        dtype=float,
    ).T
    if power_errors.shape != (2, len(power_values)) or fdr_errors.shape != (2, len(fdr_values)):
        raise ValueError("R1 figure requires one applicable interval for every plotted metric")
    axes[0, 0].bar(
        indices,
        power_values,
        yerr=power_errors,
        capsize=2.5,
        color="#2166ac",
        edgecolor="#17202a",
        hatch="//",
    )
    axes[0, 0].set_title("R1: power by declared interface and policy")
    axes[0, 0].set_ylabel("Finite-simulation power estimate")
    axes[0, 0].set_ylim(0.0, 1.05)
    fdr_bars = axes[0, 1].bar(
        indices,
        fdr_values,
        yerr=fdr_errors,
        capsize=2.5,
        color="#b2182b",
        edgecolor="#17202a",
        hatch="..",
    )
    axes[0, 1].axhline(0.05, color="#4a5568", linestyle=":", linewidth=1.4, label="nominal alpha")
    axes[0, 1].set_title("R1: FDR reported separately")
    axes[0, 1].set_ylabel("Finite-simulation FDR estimate")
    axes[0, 1].set_ylim(0.0, 1.05)
    # A valid compact run can have all-zero observed FDPs.  Zero-height bars
    # are otherwise visually indistinguishable from a missing series, so make
    # every reported FDR value explicit without altering the scale or
    # suggesting precision beyond the serialized intervals.
    for bar, value in zip(fdr_bars, fdr_values):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            max(0.018, value + 0.018),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#7f0000",
        )
    axes[0, 1].legend(fontsize=8, loc="upper right")
    for axis in axes[0, :]:
        axis.set_xticks(indices)
        axis.set_xticklabels(labels, fontsize=6.5)
        axis.grid(axis="y", color="#cbd5e1", alpha=0.65)

    action_cost = [_metric_estimate(row, "cost") for row in action_rows]
    action_power = [_metric_estimate(row, "power") for row in action_rows]
    action_cost_intervals = [_metric_interval(row, "cost") for row in action_rows]
    action_power_intervals = [_metric_interval(row, "power") for row in action_rows]
    action_labels = [_r1_row_label(row).replace("\n", " ") for row in action_rows]
    for index, (cost, observed_power, label) in enumerate(zip(action_cost, action_power, action_labels)):
        resolved_cost = _float(cost, "R1 action cost")
        resolved_power = _float(observed_power, "R1 action power")
        cost_interval = action_cost_intervals[index]
        power_interval = action_power_intervals[index]
        if cost_interval is None or power_interval is None:
            raise ValueError("R1 action-loop figure requires cost and power intervals")
        axes[1, 0].errorbar(
            resolved_cost,
            resolved_power,
            xerr=[[resolved_cost - cost_interval[0]], [cost_interval[1] - resolved_cost]],
            yerr=[[resolved_power - power_interval[0]], [power_interval[1] - resolved_power]],
            fmt=("o", "s", "^", "D")[index % 4],
            markersize=8,
            color=("#2166ac", "#1b7837", "#762a83", "#d6604d")[index % 4],
            ecolor=("#2166ac", "#1b7837", "#762a83", "#d6604d")[index % 4],
            capsize=3,
            label=label,
        )
    axes[1, 0].set_title("R1: action-loop power and cost (not a utility frontier)")
    axes[1, 0].set_xlabel("Mean cost per replication")
    axes[1, 0].set_ylabel("Power estimate")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(color="#cbd5e1", alpha=0.65)
    axes[1, 0].legend(fontsize=8, loc="best")

    regime_indices = np.arange(len(regime_labels))
    width = 0.38
    axes[1, 1].bar(
        regime_indices - width / 2,
        static_cost,
        width,
        color="#4a5568",
        edgecolor="#17202a",
        hatch="//",
        label="paired static control",
    )
    axes[1, 1].bar(
        regime_indices + width / 2,
        dynamic_cost,
        width,
        color="#dd8452",
        edgecolor="#17202a",
        hatch="..",
        label="dynamic stress process",
    )
    axes[1, 1].set_title("R3: paired static/dynamic process costs")
    axes[1, 1].set_ylabel("Mean cost per replication")
    axes[1, 1].set_xticks(regime_indices, regime_labels, fontsize=7)
    axes[1, 1].grid(axis="y", color="#cbd5e1", alpha=0.65)
    axes[1, 1].legend(fontsize=8, loc="best")
    figure.suptitle(
        "Bounded research atlas: metric-separated finite operating characteristics",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "All panels are conditional finite-simulation summaries under declared source contracts; no panel is a promotion or theorem claim.",
        ha="center",
        fontsize=8.5,
        color="#4a5568",
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.94))
    title = "Bounded research atlas operating characteristics"
    description = (
        "Four-panel chart. The top panels show R1 power and false discovery rate "
        "separately for fixed-family, posterior action-loop, and e-process interfaces; "
        "zero FDR values are printed directly so they are not mistaken for missing data. "
        "The lower panels show action-loop power versus cost without a utility ranking, "
        "and paired static versus dynamic R3 process costs across six stress regimes."
    )
    _save_figure_atomic(figure, path, title=title, description=description)
    return {
        "title": title,
        "alt": description,
        "caption": (
            "Metric-separated R1 and R3 finite-simulation summaries. Fixed-family "
            "corrections remain separate from action-loop decisions; paired dynamic-process "
            "bars are diagnostic stress evidence, not a dynamic-process theorem."
        ),
        "estimand": "Declared finite-simulation power, FDR, and mean cost summaries",
        "uncertainty": "Asymmetric finite-simulation intervals for each plotted R1 point; R3 panel retains source summaries in the atlas JSON",
        "evidence_class": "conditional_operating_characteristic",
        "validity_boundary": "No universal policy, optional-stopping, family-level, or release claim is implied.",
        "redundant_encoding": "Direct labels, hatches, marker shapes, color, and panel titles distinguish all series.",
    }


def _lane_validity(lane_id: str, result: Mapping[str, Any]) -> str:
    """Select the source-owned validity label for the evidence-boundary figure."""
    for key in ("validity", "evidence_class", "claim_boundary"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    if lane_id == "R1":
        return "interface-specific finite operating characteristics"
    raise ValueError(f"{lane_id} result lacks a validity or claim-boundary label")


def _write_boundary_figure(atlas: Mapping[str, Any], path: Path) -> dict[str, str]:
    """Render a visibly non-promotional roster of all six atlas lanes."""
    lanes = _mapping(atlas.get("lanes"), "atlas lanes")
    plan = _mapping(_mapping(atlas.get("canonical_plan"), "canonical plan").get("research_plan"), "research plan")
    plan_lanes = _mapping(plan.get("lanes"), "research plan lanes")
    labels = list(LANE_IDS)
    def sample_count(lane: str) -> int:
        lane_plan = _mapping(plan_lanes.get(lane), f"{lane} plan")
        raw = lane_plan.get("n_reps", lane_plan.get("partitions"))
        return _positive_integer(raw, f"{lane} sample count")

    replications = [sample_count(lane) for lane in labels]
    validity = [_lane_validity(lane, _mapping(lanes.get(lane), f"{lane} result")) for lane in labels]
    colors = ["#2166ac", "#4393c3", "#92c5de", "#f4a582", "#d6604d", "#b2182b"]
    hatches = ["//", "..", "xx", "\\\\", "++", "oo"]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.5),
        gridspec_kw={"width_ratios": [1.0, 1.65]},
    )
    positions = np.arange(len(labels))
    bars = axes[0].barh(positions, replications, color=colors, edgecolor="#17202a")
    for bar, hatch, count in zip(bars, hatches, replications):
        bar.set_hatch(hatch)
        axes[0].text(count + 0.06, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=9)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Declared replications / partitions")
    axes[0].set_title("Compact sample plan")
    axes[0].grid(axis="x", color="#cbd5e1", alpha=0.65)
    axes[0].set_xlim(0.0, max(replications) + 1.0)

    axes[1].axis("off")
    axes[1].set_title("Evidence boundaries retained by lane", loc="left", pad=12)
    for index, (lane, boundary, color, hatch) in enumerate(zip(labels, validity, colors, hatches)):
        y = 0.92 - index * 0.15
        axes[1].add_patch(
            plt.Rectangle((0.01, y - 0.045), 0.055, 0.08, facecolor=color, edgecolor="#17202a", hatch=hatch)
        )
        concise = boundary.replace("_", " ")
        if len(concise) > 92:
            concise = concise[:89].rstrip() + "..."
        axes[1].text(0.085, y, f"{lane}: {concise}", va="center", fontsize=9.4, wrap=True)
    axes[1].text(
        0.01,
        0.03,
        "Status: research-only. These rows preserve each source lane's own evidence class and do not create a release, promotion, theorem, or universal performance claim.",
        va="bottom",
        fontsize=9,
        color="#4a5568",
        wrap=True,
    )
    figure.suptitle("Bounded research atlas — evidence classes and explicit non-claims", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))
    title = "Bounded research atlas evidence boundaries"
    description = (
        "Two-panel research-atlas roster. A hatched horizontal bar chart gives the "
        "declared compact replication counts for R1 through R5 and G4. A text panel "
        "lists the retained validity or claim-boundary label for each lane and states "
        "that the bundle is research-only, not a release or promotion decision."
    )
    _save_figure_atomic(figure, path, title=title, description=description)
    return {
        "title": title,
        "alt": description,
        "caption": (
            "The compact atlas sample plan and its lane-level validity labels. The chart "
            "is a scope and provenance aid; it does not aggregate evidence classes or "
            "turn finite diagnostics into a publication decision."
        ),
        "estimand": "Declared lane sample plan and source-owned validity labels",
        "uncertainty": "Not applicable: this is a provenance and evidence-boundary figure",
        "evidence_class": "provenance_contract",
        "validity_boundary": "Every lane retains its own assumptions; no cross-lane promotion is computed.",
        "redundant_encoding": "Lane labels, direct text, hatches, color, and counts provide redundant identification.",
    }


def _write_figures(atlas: Mapping[str, Any], figures_dir: Path) -> dict[str, dict[str, str]]:
    """Render the complete small figure set from the supplied atlas only."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "operating_characteristics": _write_overview_figure(
            atlas, figures_dir / "research_atlas_operating_characteristics.png"
        ),
        "evidence_boundaries": _write_boundary_figure(
            atlas, figures_dir / "research_atlas_evidence_boundaries.png"
        ),
    }


def _figure_registry(
    atlas: Mapping[str, Any],
    atlas_file_hash: str,
    figures_dir: Path,
    metadata: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Return a portable, accessibility-complete figure registry payload."""
    records: list[dict[str, Any]] = []
    filenames = {
        "operating_characteristics": "research_atlas_operating_characteristics.png",
        "evidence_boundaries": "research_atlas_evidence_boundaries.png",
    }
    labels = {
        "operating_characteristics": "fig:research-atlas-operating-characteristics",
        "evidence_boundaries": "fig:research-atlas-evidence-boundaries",
    }
    sample_size = _sample_size_summary(atlas)
    scenario_hashes = _scenario_hash_summary(atlas)
    for name, filename in filenames.items():
        path = figures_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"research-atlas figure is missing: {path}")
        item = _mapping(metadata.get(name), f"figure metadata {name}")
        records.append(
            {
                "name": name,
                "path": f"figures/{filename}",
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "filename": filename,
                "label": labels[name],
                "generated_by": "scripts.run_research_atlas",
                "title": str(item["title"]),
                "alt": str(item["alt"]),
                "caption": str(item["caption"]),
                "metadata": {
                    key: str(item[key])
                    for key in (
                        "estimand",
                        "uncertainty",
                        "evidence_class",
                        "validity_boundary",
                        "redundant_encoding",
                    )
                }
                | {
                    "sample_size": sample_size,
                    "scenario_hashes": scenario_hashes,
                    "result_lineage_hash": str(atlas["content_hash"]),
                },
            }
        )
    return {
        "kind": "bounded_research_atlas_figure_registry",
        "schema_version": "2.0",
        "status": ATLAS_STATUS,
        "atlas_content_hash": str(atlas["content_hash"]),
        "atlas_file_sha256": atlas_file_hash,
        "figures": records,
    }


def _sample_size_summary(atlas: Mapping[str, Any]) -> str:
    """Summarize declared compact sample sizes for the reader-facing registry."""
    canonical = _mapping(atlas.get("canonical_plan"), "canonical plan")
    plan = _mapping(canonical.get("research_plan"), "research plan")
    lane_plans = _mapping(plan.get("lanes"), "research plan lanes")
    values: list[str] = []
    for lane in LANE_IDS:
        lane_plan = _mapping(lane_plans.get(lane), f"{lane} plan")
        count = lane_plan.get("n_reps", lane_plan.get("partitions"))
        values.append(f"{lane}={_positive_integer(count, f'{lane} sample count')}")
    return "declared compact plan: " + ", ".join(values)


def _scenario_hash_summary(atlas: Mapping[str, Any]) -> str:
    """Expose source scenario/configuration witnesses without inventing a common one."""
    lanes = _mapping(atlas.get("lanes"), "atlas lanes")
    values: list[str] = []
    for lane in LANE_IDS:
        result = _mapping(lanes.get(lane), f"{lane} result")
        for key in ("scenario_hash", "base_scenario_hash", "configuration_hash", "plan_hash"):
            value = result.get(key)
            if isinstance(value, str) and len(value) == 64:
                values.append(f"{lane}:{key}={value}")
    r1_rows = _list(_mapping(lanes.get("R1"), "R1 result").get("surface"), "R1 surface")
    for row in r1_rows:
        scenario_hash = _mapping(row, "R1 surface row").get("scenario_hash")
        if isinstance(scenario_hash, str) and len(scenario_hash) == 64:
            values.append(f"R1:scenario_hash={scenario_hash}")
    g4_environment = _mapping(_mapping(lanes.get("G4"), "G4 result").get("dynamic_environment"), "G4 environment")
    environment_hash = g4_environment.get("environment_hash")
    if isinstance(environment_hash, str) and len(environment_hash) == 64:
        values.append(f"G4:environment_hash={environment_hash}")
    if not values:
        raise ValueError("research atlas must expose at least one scenario or configuration hash")
    return "; ".join(sorted(set(values)))


def _dashboard_payload(
    atlas: Mapping[str, Any],
    atlas_file_hash: str,
    figure_registry_hash: str,
) -> dict[str, Any]:
    """Create a concise machine-readable dashboard without a release status."""
    lanes = _mapping(atlas.get("lanes"), "atlas lanes")
    canonical = _mapping(atlas.get("canonical_plan"), "canonical plan")
    research_plan = _mapping(canonical.get("research_plan"), "research plan")
    plan_lanes = _mapping(research_plan.get("lanes"), "research plan lanes")
    return {
        "kind": "bounded_research_atlas_dashboard_payload",
        "schema_version": "1.0",
        "status": ATLAS_STATUS,
        "atlas_content_hash": str(atlas["content_hash"]),
        "atlas_file_sha256": atlas_file_hash,
        "figure_registry_sha256": figure_registry_hash,
        "plan_hash": str(atlas["plan_hash"]),
        "lanes": [
            {
                "id": lane,
                "sample_plan": _mapping(plan_lanes.get(lane), f"{lane} plan"),
                "validity_or_evidence_class": _lane_validity(
                    lane, _mapping(lanes.get(lane), f"{lane} result")
                ),
            }
            for lane in LANE_IDS
        ],
        "claim_boundary": (
            "Dashboard values are source-linked finite research evidence. No pass/fail "
            "promotion, publication readiness, universal policy ranking, or theorem "
            "status is produced by this payload."
        ),
    }


def _validate_output_root(output_root: Path) -> Path:
    """Require one new, isolated destination and reject the canonical output tree."""
    root = Path(output_root).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    root = require_safe_output_root(root, project_root=ROOT)
    if root == ROOT.resolve() or root == ROOT.resolve() / "output":
        raise ValueError("research atlas requires a new isolated --output-root, not the project output root")
    if root == root.parent:
        raise ValueError("research atlas output root must be a concrete directory")
    if root.exists():
        raise FileExistsError(
            f"research atlas output root already exists: {root}; choose a new isolated directory"
        )
    return root


def _manifest(staging_root: Path, atlas: Mapping[str, Any]) -> dict[str, Any]:
    """Build artifact records after every staged file has been generated."""
    artifacts = {
        "atlas": artifact_record(staging_root / "research_atlas.json", staging_root, "research_atlas", root_label="output"),
        "figure_registry": artifact_record(
            staging_root / "figures" / "figure_registry.json",
            staging_root,
            "figure_registry",
            root_label="output",
        ),
        "dashboard_payload": artifact_record(
            staging_root / "dashboard.json", staging_root, "dashboard_payload", root_label="output"
        ),
    }
    for name in (
        "research_atlas_operating_characteristics.png",
        "research_atlas_evidence_boundaries.png",
    ):
        artifacts[f"figure:{name}"] = artifact_record(
            staging_root / "figures" / name,
            staging_root,
            "accessible_research_figure",
            root_label="output",
        )
    return {
        "kind": "bounded_research_atlas_manifest",
        "schema_version": "1.0",
        "status": ATLAS_STATUS,
        "plan_hash": str(atlas["plan_hash"]),
        "atlas_content_hash": str(atlas["content_hash"]),
        "source_hashes": _mapping(atlas.get("source_hashes"), "source hashes"),
        "artifacts": artifacts,
        "claim_boundary": (
            "This manifest binds a bounded research bundle only. It is intentionally "
            "barred from release finalization or publication promotion."
        ),
    }


def run_research_atlas(output_root: Path, plan: ResearchAtlasPlan | None = None) -> dict[str, Any]:
    """Generate and atomically publish a bounded research-atlas artifact bundle.

    ``output_root`` must be an explicit new directory.  No fallback to
    ``output/`` is available, and an existing root is never overwritten.
    """
    target = _validate_output_root(Path(output_root))
    active_plan = DEFAULT_RESEARCH_ATLAS_PLAN if plan is None else plan
    if not isinstance(active_plan, ResearchAtlasPlan):
        raise ValueError("plan must be a ResearchAtlasPlan")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{target.name}.research-atlas-", dir=target.parent))
    try:
        atlas = build_research_atlas(active_plan)
        atlas_path = staging_root / "research_atlas.json"
        atomic_write_json(atlas_path, atlas)
        atlas_file_hash = sha256_file(atlas_path)

        figures_dir = staging_root / "figures"
        figure_metadata = _write_figures(atlas, figures_dir)
        registry = _figure_registry(atlas, atlas_file_hash, figures_dir, figure_metadata)
        registry_path = figures_dir / "figure_registry.json"
        atomic_write_json(registry_path, registry)
        dashboard = _dashboard_payload(atlas, atlas_file_hash, sha256_file(registry_path))
        dashboard_path = staging_root / "dashboard.json"
        atomic_write_json(dashboard_path, dashboard)
        manifest = _manifest(staging_root, atlas)
        manifest_path = staging_root / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        os.replace(staging_root, target)
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    return {
        "output_root": str(target),
        "atlas_path": str(target / "research_atlas.json"),
        "manifest_path": str(target / "manifest.json"),
        "plan_hash": str(atlas["plan_hash"]),
        "atlas_content_hash": str(atlas["content_hash"]),
        "manifest_sha256": sha256_file(target / "manifest.json"),
        "status": ATLAS_STATUS,
    }


def _parser() -> argparse.ArgumentParser:
    """Create the intentionally small CLI boundary for the bounded atlas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="New isolated directory for this research-only atlas; never defaults to output/.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RESEARCH_ATLAS_PLAN.seed)
    parser.add_argument("--workers", type=int, default=DEFAULT_RESEARCH_ATLAS_PLAN.workers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit-root CLI and print a compact machine-readable receipt."""
    args = _parser().parse_args(argv)
    plan = ResearchAtlasPlan(seed=args.seed, workers=args.workers)
    receipt = run_research_atlas(args.output_root, plan)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main`` in tests.
    raise SystemExit(main())
