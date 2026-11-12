#!/usr/bin/env python3
"""Run the configured experiment and generate its evidence artifacts.

The script is intentionally an orchestrator: statistical logic lives in
``active_inference_power/`` while this file loads configuration, serializes results, and invokes
the figure/dashboard writers. Run from the project directory or by absolute
path; imports are made package-safe for both modes.
"""

from __future__ import annotations

# Imports below intentionally follow the script-local path bootstrap so the
# orchestrator also works when invoked by absolute path outside the checkout.
# ruff: noqa: E402

import argparse
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import sys
import time
from numbers import Real
from typing import Any, TypedDict, cast

import yaml
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from active_inference_power._types import TwoGroupsConfig  # noqa: E402
from active_inference_power.active_inference import AITestConfig, evaluate_ai_regime  # noqa: E402
from active_inference_power.agent_sampling import (
    AgentSamplingTrace,
    PolicyName,
    simulate_agent_policy,
)  # noqa: E402
from active_inference_power.action_loop import ActionPolicy, evaluate_power_surface, run_action_loop  # noqa: E402
from active_inference_power.action_statistics import paired_action_policy_comparisons  # noqa: E402
from active_inference_power.demo import build_demo_payload  # noqa: E402
from active_inference_power.benchmarks import BenchmarkSpec, load_agent_trace, load_normalized_golub_fixture  # noqa: E402
from active_inference_power.calibration import (  # noqa: E402
    calibration_replication_samples,
    null_pvalue_calibration,
    null_pvalue_calibration_from_samples,
)
from scripts.checkpointing import load_checkpoint, write_partial_checkpoint  # noqa: E402
from active_inference_power.certificates import run_all_certificates  # noqa: E402
from active_inference_power.certificate_gate import (  # noqa: E402
    certificate_gate_result,
    default_certificate_gate_policy,
    effective_certificate_gate_result,
    normalize_certificate_gate_policy,
)
from active_inference_power.computational_complexity import (  # noqa: E402
    ComputationalComplexityPlan,
    build_computational_complexity,
    complexity_plan_from_mapping,
)
from active_inference_power.figures import generate_figures, write_dashboard, write_figure_registry  # noqa: E402
from active_inference_power.formalism import formalism_registry  # noqa: E402
from active_inference_power.simulation import compare_procedures, power_curve, simulate_procedure  # noqa: E402
from active_inference_power.uncertainty import required_reps_for_half_width, summarize_bounded  # noqa: E402
from active_inference_power.schema import validate_checkpoint, validate_claim_ledger, validate_manifest, validate_results  # noqa: E402
from active_inference_power.scenario import (  # noqa: E402
    make_reference_action_scenario,
    make_reference_null_action_scenario,
    make_static_binary_split_confirmation_scenario,
)
from active_inference_power.suite import evaluate_adaptive_target_selection, evaluate_conditional_selection_calibration, evaluate_future_evidence_selection_failure, evaluate_split_stream_confirmation  # noqa: E402
from active_inference_power.learned_policy import LearnedPolicyConfig, evaluate_learned_policy, train_tabular_policy  # noqa: E402
from active_inference_power.online_fdr import (  # noqa: E402
    EValueLONDResult,
    OnlineFDRResult,
    default_gamma,
    e_lond,
    lord_plus_plus,
    saffron,
)
from active_inference_power.dependence import DependenceKind  # noqa: E402
from active_inference_power.sequential import bernoulli_confidence_sequence, binary_e_process, fixed_horizon_binomial_pvalue  # noqa: E402
from active_inference_power.uncertainty import paired_difference_summary, wilson_interval  # noqa: E402
from scripts.artifacts import (  # noqa: E402
    artifact_record,
    atomic_write_json,
    invalidate_renderer_artifact_records,
    require_safe_output_root,
)
from scripts.release_provenance import (  # noqa: E402
    campaign_lane_predecessor_artifacts,
    execution_source_sha256,
)
from scripts.run_ledger import RunLedger  # noqa: E402

LOGGER = logging.getLogger("active_inference_power.generate_results")

STAGE_NAMES = (
    "certificates",
    "method_comparison",
    "paired_comparisons",
    "power_curve",
    "dependence_sweep",
    "active_inference",
    "sequential",
    "online_fdr",
    "agent_sampling",
    "conditional_power",
    "benchmarks",
    "null_pvalue_calibration",
)

# The reference comparison is deliberately a fixed, ordered panel rather than
# an open-ended dispatcher surface.  These procedures share one two-groups
# p-value stream in ``compare_procedures``; the conservative and split-stream
# Storey variants belong to their separately specified review lanes instead.
# Keeping the order here makes the result table, paired contrasts, and sample
# accounting reproducible rather than configuration-order dependent.
REFERENCE_COMPARISON_METHODS: tuple[str, ...] = (
    "bonferroni",
    "sidak",
    "holm",
    "hochberg",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "storey_qvalue",
    "adaptive_bh",
    "weighted_bh",
)


def _reference_scenario_identifier(reliability: float, horizon: int, alpha: float) -> str:
    """Return the canonical lineage identifier for a reference surface cell.

    The identifier is deliberately shared by generation and resume validation.
    Keeping the parameter encoding in one place prevents a checkpoint from
    being rejected merely because the default reference cell and the swept
    surface use different explicit identifier paths.
    """
    def token(value: float) -> str:
        return format(value, ".8g").replace("-", "m").replace(".", "p")

    return f"reference_action_loop__r{token(float(reliability))}__h{int(horizon)}__a{token(float(alpha))}"


class _StageState(TypedDict):
    completed: int
    total: int
    elapsed_seconds: list[float]


def _stage_seed_schedule(master_seed: int) -> dict[str, int]:
    """Derive independent, stable child seeds for each release stage."""
    if isinstance(master_seed, bool) or int(master_seed) != master_seed or int(master_seed) < 0:
        raise ValueError("master seed must be a non-negative integer")
    children = np.random.SeedSequence(int(master_seed)).spawn(len(STAGE_NAMES))
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(STAGE_NAMES, children)
    }


def _load_config(path: Path) -> dict[str, Any]:
    """Load and validate the YAML experiment configuration at the CLI boundary."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise ValueError(f"{path} is not valid YAML") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a top-level mapping")
    experiment = data.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError(f"{path} must define a mapping at experiment:")
    required = (
        "alpha",
        "m",
        "pi0",
        "effect",
        "n_reps",
        "methods",
        "effect_grid",
        "dependence_rhos",
        "fast_replication_fraction",
        "certificate_m",
        "certificate_n_reps",
        "active_inference_reps",
        "active_inference",
        "target_mc_half_width",
        "mc_confidence",
        "calibration_n_reps",
        "calibration_seed",
    )
    missing = [key for key in required if key not in experiment]
    if missing:
        raise ValueError(f"{path} is missing experiment keys: {missing}")
    _validate_experiment(experiment, path)
    return data


def _require_real(experiment: dict[str, Any], key: str, path: Path) -> float:
    """Return a finite numeric experiment value or fail with a useful message."""
    value = experiment.get(key)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{path} experiment.{key} must be a finite number; got {value!r}")
    return float(value)


def _require_integer(
    experiment: dict[str, Any], key: str, path: Path, *, minimum: int
) -> int:
    """Return an integer-valued experiment parameter with a lower bound."""
    value = experiment.get(key)
    numeric_value = float(value) if isinstance(value, Real) and not isinstance(value, bool) else None
    if (
        isinstance(value, bool)
        or numeric_value is None
        or not math.isfinite(numeric_value)
        or int(numeric_value) != numeric_value
        or numeric_value < minimum
    ):
        raise ValueError(
            f"{path} experiment.{key} must be an integer >= {minimum}; got {value!r}"
        )
    return int(numeric_value)


def _validate_experiment(experiment: dict[str, Any], path: Path) -> None:
    """Validate every field consumed by the result generator and manuscript."""
    alpha = _require_real(experiment, "alpha", path)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"{path} experiment.alpha must lie in (0, 1); got {alpha!r}")
    for key in ("pi0", "rho"):
        value = _require_real(experiment, key, path) if key in experiment else 0.0
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{path} experiment.{key} must lie in [0, 1]; got {value!r}")
    _require_integer(experiment, "m", path, minimum=1)
    _require_integer(experiment, "n_reps", path, minimum=1)
    _require_integer(experiment, "certificate_m", path, minimum=1)
    _require_integer(experiment, "certificate_n_reps", path, minimum=1)
    _require_integer(experiment, "active_inference_reps", path, minimum=1)
    _require_integer(experiment, "calibration_n_reps", path, minimum=1)
    _require_integer(experiment, "calibration_seed", path, minimum=0)
    split_storey = experiment.get("storey_split_confirmation", {})
    if not isinstance(split_storey, dict):
        raise ValueError(f"{path} experiment.storey_split_confirmation must be a mapping")
    split_lambda = _require_real(split_storey, "lambda", path)
    split_confidence = _require_real(split_storey, "confidence", path)
    if not 0.0 <= split_lambda < 1.0:
        raise ValueError(f"{path} experiment.storey_split_confirmation.lambda must lie in [0, 1)")
    if not 0.0 < split_confidence < 1.0:
        raise ValueError(f"{path} experiment.storey_split_confirmation.confidence must lie in (0, 1)")
    if split_storey.get("streams") != "independent_selection_and_confirmation":
        raise ValueError(f"{path} experiment.storey_split_confirmation.streams must declare independent streams")
    if "seed" in experiment:
        _require_integer(experiment, "seed", path, minimum=0)
    if "certificate_seed" in experiment:
        _require_integer(experiment, "certificate_seed", path, minimum=0)
    _certificate_gate_policy(experiment, path)

    fast_fraction = _require_real(experiment, "fast_replication_fraction", path)
    if not 0.0 < fast_fraction <= 1.0:
        raise ValueError(
            f"{path} experiment.fast_replication_fraction must lie in (0, 1]; "
            f"got {fast_fraction!r}"
        )
    confidence = _require_real(experiment, "mc_confidence", path)
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"{path} experiment.mc_confidence must lie in (0, 1)")
    target_half_width = _require_real(experiment, "target_mc_half_width", path)
    if not 0.0 < target_half_width < 1.0:
        raise ValueError(f"{path} experiment.target_mc_half_width must lie in (0, 1)")
    if "figure_error_multiplier" in experiment:
        multiplier = _require_real(experiment, "figure_error_multiplier", path)
        if multiplier <= 0.0:
            raise ValueError(f"{path} experiment.figure_error_multiplier must be positive")
    if "figure_dpi" in experiment:
        _require_integer(experiment, "figure_dpi", path, minimum=72)
    if "pymdp_version" in experiment and (
        not isinstance(experiment["pymdp_version"], str)
        or not experiment["pymdp_version"].strip()
    ):
        raise ValueError(f"{path} experiment.pymdp_version must be a non-empty string")
    # Frozen predecessor configurations may omit source-work accounting. They
    # remain inspectable configuration witnesses, but a current generated
    # artifact must declare the plan at the production boundary below.
    if "computational_complexity" in experiment:
        try:
            complexity_plan_from_mapping(experiment["computational_complexity"])
        except ValueError as error:
            raise ValueError(
                f"{path} experiment.computational_complexity is invalid: {error}"
            ) from error
    methods = experiment["methods"]
    if not isinstance(methods, list) or tuple(methods) != REFERENCE_COMPARISON_METHODS:
        raise ValueError(
            f"{path} experiment.methods must exactly match the ordered canonical "
            f"reference comparison methods {list(REFERENCE_COMPARISON_METHODS)!r}; "
            f"got {methods!r}"
        )
    for key in ("effect_grid", "dependence_rhos"):
        values = experiment.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{path} experiment.{key} must be a non-empty list")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{path} experiment.{key} contains non-numeric value {value!r}")
            numeric_value = float(value)
            if key == "dependence_rhos":
                lower = -1.0 / (int(experiment["m"]) - 1) if int(experiment["m"]) > 1 else 0.0
                if not lower <= numeric_value <= 1.0:
                    raise ValueError(
                        f"{path} experiment.dependence_rhos values must lie in [{lower}, 1]"
                    )
    block = experiment.get("dependence_block")
    if block is not None:
        if not isinstance(block, dict):
            raise ValueError(f"{path} experiment.dependence_block must be a mapping")
        rho = block.get("rho")
        block_size = block.get("block_size")
        if isinstance(rho, bool) or not isinstance(rho, Real) or not math.isfinite(float(rho)):
            raise ValueError(f"{path} experiment.dependence_block.rho must be finite numeric")
        if isinstance(block_size, bool) or not isinstance(block_size, int) or not 1 <= block_size <= int(experiment["m"]):
            raise ValueError(f"{path} experiment.dependence_block.block_size must lie in [1, m]")
        lower = -1.0 / (int(block_size) - 1) if int(block_size) > 1 else 0.0
        if int(block_size) == 1 and float(rho) != 0.0:
            raise ValueError(f"{path} experiment.dependence_block.rho must be zero for block_size=1")
        if not lower <= float(rho) < 1.0:
            raise ValueError(f"{path} experiment.dependence_block.rho must lie in [{lower}, 1)")

    ai = experiment.get("active_inference")
    regimes = ai.get("regimes") if isinstance(ai, dict) else None
    if not isinstance(regimes, dict) or not {"strong_evidence", "weak_evidence"}.issubset(regimes):
        raise ValueError(
            f"{path} experiment.active_inference.regimes must define strong_evidence and weak_evidence"
        )
    for name in ("strong_evidence", "weak_evidence"):
        regime = regimes[name]
        if not isinstance(regime, dict):
            raise ValueError(f"{path} active-inference regime {name!r} must be a mapping")
        reliability = regime.get("reliability")
        threshold = regime.get("threshold")
        prior = regime.get("prior_signal", 0.5)
        reliability_value = (
            float(reliability)
            if isinstance(reliability, Real) and not isinstance(reliability, bool)
            else None
        )
        if (
            isinstance(reliability, bool)
            or reliability_value is None
            or not math.isfinite(reliability_value)
            or not 0.0 < reliability_value < 1.0
        ):
            raise ValueError(f"{path} active-inference {name}.reliability is invalid")
        for key, parameter_value in (("threshold", threshold), ("prior_signal", prior)):
            threshold_value = (
                float(parameter_value)
                if isinstance(parameter_value, Real) and not isinstance(parameter_value, bool)
                else None
            )
            if (
                isinstance(parameter_value, bool)
                or threshold_value is None
                or not math.isfinite(threshold_value)
                or not 0.0 <= threshold_value <= 1.0
            ):
                raise ValueError(f"{path} active-inference {name}.{key} is invalid")
        horizon = regime.get("horizon")
        horizon_value = (
            float(horizon) if isinstance(horizon, Real) and not isinstance(horizon, bool) else None
        )
        if (
            isinstance(horizon, bool)
            or horizon_value is None
            or not math.isfinite(horizon_value)
            or int(horizon_value) != horizon_value
            or horizon_value < 0
        ):
            raise ValueError(f"{path} active-inference {name}.horizon is invalid")

    action_loop = experiment.get("action_loop", {})
    if not isinstance(action_loop, dict):
        raise ValueError(f"{path} experiment.action_loop must be a mapping")
    action_horizon = action_loop.get("horizon", 12)
    action_horizon_value = (
        float(action_horizon)
        if isinstance(action_horizon, Real) and not isinstance(action_horizon, bool)
        else None
    )
    if isinstance(action_horizon, bool) or action_horizon_value is None or not math.isfinite(action_horizon_value) or int(action_horizon_value) != action_horizon_value or action_horizon_value < 1:
        raise ValueError(f"{path} experiment.action_loop.horizon is invalid")
    policies = action_loop.get("policies", ["fixed_horizon", "posterior_threshold", "cost_aware", "e_process"])
    allowed_policies = {"fixed_horizon", "posterior_threshold", "information_gain", "cost_aware", "adaptive_target", "posterior_sampling", "e_process"}
    if not isinstance(policies, list) or not policies or any(value not in allowed_policies for value in policies):
        raise ValueError(f"{path} experiment.action_loop.policies is invalid")
    reliability_grid = action_loop.get("reliability_grid", [0.8])
    if not isinstance(reliability_grid, list) or not reliability_grid:
        raise ValueError(f"{path} experiment.action_loop.reliability_grid must be non-empty")
    for value in reliability_grid:
        grid_value = float(value) if isinstance(value, Real) and not isinstance(value, bool) else None
        if isinstance(value, bool) or grid_value is None or not math.isfinite(grid_value) or not 0.5 < grid_value < 1.0:
            raise ValueError(f"{path} experiment.action_loop.reliability_grid contains invalid value {value!r}")
    campaign = experiment.get("campaign")
    if campaign is not None:
        if not isinstance(campaign, dict):
            raise ValueError(f"{path} experiment.campaign must be a mapping")
        if not isinstance(campaign.get("name"), str) or not campaign["name"].strip():
            raise ValueError(f"{path} experiment.campaign.name must be a non-empty string")
        required_campaign = (
            "n_reps", "certificate_n_reps", "active_inference_reps",
            "calibration_n_reps", "effect_grid", "dependence_rhos", "action_loop",
        )
        missing_campaign = [key for key in required_campaign if key not in campaign]
        if missing_campaign:
            raise ValueError(f"{path} experiment.campaign is missing keys: {missing_campaign}")
        for key in ("n_reps", "certificate_n_reps", "active_inference_reps", "calibration_n_reps"):
            _require_integer(campaign, key, path, minimum=1)
        if "target_mc_half_width" in campaign:
            target = _require_real(campaign, "target_mc_half_width", path)
            if not 0.0 < target < 1.0:
                raise ValueError(f"{path} experiment.campaign.target_mc_half_width must lie in (0, 1)")
        for key in ("effect_grid", "dependence_rhos"):
            values = campaign[key]
            if not isinstance(values, list) or not values:
                raise ValueError(f"{path} experiment.campaign.{key} must be a non-empty list")
            for value in values:
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                    raise ValueError(f"{path} experiment.campaign.{key} contains non-numeric value {value!r}")
                numeric_value = float(value)
                if key == "dependence_rhos":
                    lower = -1.0 / (int(experiment["m"]) - 1) if int(experiment["m"]) > 1 else 0.0
                    if not lower <= numeric_value <= 1.0:
                        raise ValueError(f"{path} experiment.campaign.dependence_rhos values must lie in [{lower}, 1]")
        campaign_action = campaign["action_loop"]
        if not isinstance(campaign_action, dict):
            raise ValueError(f"{path} experiment.campaign.action_loop must be a mapping")
        campaign_reliability = campaign_action.get("reliability_grid")
        if not isinstance(campaign_reliability, list) or not campaign_reliability:
            raise ValueError(f"{path} experiment.campaign.action_loop.reliability_grid must be a non-empty list")
        for value in campaign_reliability:
            campaign_grid_value = float(value) if isinstance(value, Real) and not isinstance(value, bool) else None
            if isinstance(value, bool) or campaign_grid_value is None or not math.isfinite(campaign_grid_value) or not 0.5 < campaign_grid_value < 1.0:
                raise ValueError(f"{path} experiment.campaign.action_loop.reliability_grid contains invalid value {value!r}")
        if "horizon" in campaign_action:
            _require_integer(campaign_action, "horizon", path, minimum=1)
        if "policies" in campaign_action:
            if not isinstance(campaign_action["policies"], list) or not campaign_action["policies"] or any(value not in allowed_policies for value in campaign_action["policies"]):
                raise ValueError(f"{path} experiment.campaign.action_loop.policies is invalid")


def _require_generated_complexity_plan(
    experiment: dict[str, Any], path: Path
) -> ComputationalComplexityPlan:
    """Return the declared plan required by every current generated artifact.

    ``_load_config`` intentionally accepts a frozen predecessor configuration
    that lacks this field, so provenance and campaign-boundary tooling can
    still inspect its original YAML. The generator must not turn that source
    into a seemingly current result bundle without the explicit accounting
    contract.
    """
    if "computational_complexity" not in experiment:
        raise ValueError(
            f"{path} current generated artifacts require "
            "experiment.computational_complexity; a missing plan is accepted "
            "only for predecessor-configuration validation"
        )
    try:
        return complexity_plan_from_mapping(experiment["computational_complexity"])
    except ValueError as error:
        raise ValueError(
            f"{path} experiment.computational_complexity is invalid: {error}"
        ) from error


def _certificate_gate_policy(experiment: dict[str, Any], path: Path) -> dict[str, list[str]]:
    """Return the explicit all-certificate partition or its conservative default."""
    raw_policy = experiment.get("certificate_gate_policy")
    if raw_policy is None:
        return default_certificate_gate_policy()
    try:
        return normalize_certificate_gate_policy(raw_policy)
    except ValueError as exc:
        raise ValueError(
            f"{path} experiment.certificate_gate_policy is invalid: {exc}"
        ) from exc


def _resolve_experiment_profile(experiment: dict[str, Any], profile: str) -> dict[str, Any]:
    """Resolve a named computational profile without mutating source config."""
    resolved = copy.deepcopy(experiment)
    if profile != "campaign":
        return resolved
    campaign = resolved.get("campaign")
    if not isinstance(campaign, dict):
        raise ValueError("campaign profile requires experiment.campaign")
    for key in (
        "n_reps", "certificate_n_reps", "active_inference_reps", "calibration_n_reps",
        "target_mc_half_width",
    ):
        if key in campaign:
            resolved[key] = campaign[key]
    for key in ("effect_grid", "dependence_rhos"):
        resolved[key] = list(campaign[key])
    resolved_action = dict(resolved.get("action_loop", {}))
    resolved_action.update(copy.deepcopy(campaign["action_loop"]))
    resolved["action_loop"] = resolved_action
    resolved["profile_design"] = {
        "name": str(campaign["name"]),
        "profile": "campaign",
        "expanded": True,
        "source": "experiment.campaign",
    }
    return resolved


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_input_hash(path: Path) -> str:
    """Return a ledger-safe input digest even when configuration loading fails."""
    try:
        return _config_hash(path)
    except OSError:
        # A failed or interrupted run still needs a valid, auditable ledger. The
        # fallback is deliberately not presented as a configuration digest for a
        # successful result; it identifies the unreadable input path only.
        return hashlib.sha256(
            f"unreadable-input:{Path(path).resolve()}".encode("utf-8")
        ).hexdigest()


def _artifact(path: Path, output: Path, role: str) -> dict[str, Any]:
    """Record a project source or output artifact in its safe namespace."""
    path = Path(path).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        return artifact_record(path, output, role, root_label="output")
    return artifact_record(path, ROOT, role, root_label="project")


def _benchmark_artifact_records(output: Path) -> dict[str, dict[str, Any]]:
    """Record every offline benchmark input required by terminal inventories.

    The generator, campaign finalizer, and release finalizer deliberately bind
    the same committed benchmark manifest and CSV fixtures.  Keep the
    producer-side names aligned with ``required_*_artifacts`` so a completed
    campaign never relies on manuscript hydration to supply scientific-input
    provenance.
    """
    benchmark_directory = ROOT / "data" / "benchmarks"
    records: dict[str, dict[str, Any]] = {}
    manifest = benchmark_directory / "manifest.yaml"
    if manifest.is_file():
        records["benchmark_manifest"] = _artifact(
            manifest, output, "benchmark_provenance"
        )
    fixture_names = {
        "golub_normalized_fixture.csv": "golub_normalized_offline_fixture",
    }
    for fixture in sorted(benchmark_directory.glob("*.csv")):
        records[f"benchmark:{fixture_names.get(fixture.name, fixture.stem)}"] = (
            _artifact(fixture, output, "benchmark_fixture")
        )
    return records


def _display_path(path: Path, output: Path) -> str:
    """Return a stable human-readable path for ledger events."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.relative_to(output))


def _load_claim_ledger() -> str:
    """Validate the claim ledger before a run and return its content hash."""
    path = ROOT / "data" / "claim_ledger.yaml"
    ledger = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    validate_claim_ledger(ledger, project_root=ROOT)
    return _config_hash(path)


def _checkpoint_dependency_hashes(stage: str, value: Any) -> dict[str, list[str]]:
    """Record composed scenario lineage for resumable suite stages."""
    empty: dict[str, list[str]] = {
        "scenario_hashes": [],
        "model_hashes": [],
        "process_hashes": [],
        "setting_hashes": [],
    }
    if stage != "conditional_power" or not isinstance(value, dict):
        return empty
    contracts = value.get("scenario_contracts", [])
    if not isinstance(contracts, list):
        return empty
    dependency_contracts = list(contracts)
    split_contract = value.get("split_confirmation_scenario")
    if isinstance(split_contract, dict):
        # The static split-confirmation channel is outside the operating grid,
        # but changing it must invalidate the conditional-power checkpoint.
        dependency_contracts.append(split_contract)

    def digest(payload: Any) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    scenario_hashes: list[str] = []
    model_hashes: list[str] = []
    process_hashes: list[str] = []
    setting_hashes: list[str] = []
    for contract in dependency_contracts:
        if not isinstance(contract, dict):
            continue
        declared_scenario_hash = contract.get("scenario_hash")
        scenario_hashes.append(
            str(declared_scenario_hash)
            if isinstance(declared_scenario_hash, str)
            else digest(contract)
        )
        model_hashes.append(digest(contract.get("model", {})))
        process_hashes.append(digest(contract.get("process", {})))
        setting_hashes.append(digest(contract.get("setting", {})))
    return {
        "scenario_hashes": scenario_hashes,
        "model_hashes": model_hashes,
        "process_hashes": process_hashes,
        "setting_hashes": setting_hashes,
    }


def _expected_checkpoint_dependency_hashes(
    stage: str, experiment: dict[str, Any]
) -> dict[str, list[str]]:
    """Build cheap current dependency hashes before accepting a resume.

    A stage-complete checkpoint is not contract-valid merely because its own
    value is internally self-consistent: it must also describe the contracts
    the current source/configuration would generate. This closes the subtle case
    where a scenario contract changes while the YAML configuration does not.
    """
    if stage != "conditional_power":
        return _checkpoint_dependency_hashes(stage, {})
    raw = experiment["active_inference"]["regimes"]["strong_evidence"]
    action_config = experiment.get("action_loop", {})
    reliabilities = [
        float(value)
        for value in action_config.get("reliability_grid", [raw["reliability"]])
    ]
    horizon = int(action_config.get("horizon", raw["horizon"]))
    scenarios = [
        make_reference_action_scenario(
            reliability=value,
            horizon=horizon,
            alpha=float(experiment["alpha"]),
            scenario_id=_reference_scenario_identifier(value, horizon, float(experiment["alpha"])),
        )
        for value in reliabilities
    ]
    split_confirmation_scenario = make_static_binary_split_confirmation_scenario(
        reliability=float(reliabilities[0]),
        horizon=int(action_config.get("split_confirmation_steps", horizon)),
        alpha=float(experiment["alpha"]),
    )
    return _checkpoint_dependency_hashes(
        stage,
        {
            "scenario_contracts": [scenario.to_mapping() for scenario in scenarios],
            "split_confirmation_scenario": split_confirmation_scenario.to_mapping(),
        },
    )


def _method_comparison(
    experiment: dict[str, Any], n_reps: int, confidence: float, workers: int = 1,
    seed: int | None = None,
) -> dict[str, dict[str, float]]:
    cfg = TwoGroupsConfig(
        m=int(experiment["m"]),
        pi0=float(experiment["pi0"]),
        effect=float(experiment["effect"]),
        rho=0.0,
        sided=int(experiment.get("sided", 1)),
    )
    results = compare_procedures(
        cfg,
        [str(method) for method in experiment["methods"]],
        float(experiment["alpha"]),
        n_reps,
        int(experiment.get("seed", 0) if seed is None else seed),
        workers,
    )
    return {
        method: {
            "fdr": r.fdr,
            "power": r.power,
            "fwer": r.fwer,
            "mean_rejections": r.mean_rejections,
            **{
                key: value
                for metric in ("fdr", "power", "fwer")
                for key, value in _metric_accounting(r, metric, confidence, float(experiment["target_mc_half_width"])).items()
            },
        }
        for method, r in results.items()
    }


def _metric_accounting(
    result: Any, metric: str, confidence: float, target_half_width: float | None = None
) -> dict[str, float]:
    if metric == "fdr":
        samples = getattr(result, "fdp_samples", None)
        if samples is None:
            samples = result.fdr_samples
    elif metric == "power":
        samples = result.power_samples
    elif metric == "fwer":
        samples = result.fwer_samples
    else:
        raise ValueError(f"unsupported metric {metric!r}")
    summary = summarize_bounded(samples, confidence, target_half_width)
    return {
        f"mc_stderr_{metric}": summary["standard_error"],
        f"mc_ci95_lower_{metric}": summary["normal_ci_lower"],
        f"mc_ci95_upper_{metric}": summary["normal_ci_upper"],
        f"mc_half_width_{metric}": summary["normal_half_width"],
        f"mc_hoeffding_radius_{metric}": summary["hoeffding_radius"],
        f"mc_required_n_for_target_{metric}": summary["required_n_for_target"],
    }


def _bh_power_curve(
    experiment: dict[str, Any], n_reps: int, confidence: float, workers: int = 1,
    seed: int | None = None,
) -> list[dict[str, float]]:
    base = TwoGroupsConfig(
        m=int(experiment["m"]),
        pi0=float(experiment["pi0"]),
        effect=0.0,
        rho=0.0,
        sided=int(experiment.get("sided", 1)),
    )
    points = power_curve(
        base,
        [float(effect) for effect in experiment["effect_grid"]],
        "benjamini_hochberg",
        float(experiment["alpha"]),
        n_reps,
        int(experiment.get("seed", 0) if seed is None else seed),
        workers,
    )
    return [
        {
            "effect": p.effect,
            "fdr": p.fdr,
            "power": p.power,
            **_metric_accounting(p, "fdr", confidence, float(experiment["target_mc_half_width"])),
            **_metric_accounting(p, "power", confidence, float(experiment["target_mc_half_width"])),
        }
        for p in points
    ]


def _dependence_sweep(
    experiment: dict[str, Any], n_reps: int, confidence: float, workers: int = 1,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    regimes: list[tuple[str, float, int | None]] = [
        ("rho", float(rho), None)
        for rho in experiment.get("dependence_rhos", [0.0, 0.25, 0.5, 0.75])
    ]
    block = experiment.get("dependence_block")
    if isinstance(block, dict):
        regimes.append(("block", float(block["rho"]), int(block["block_size"])))
    for regime_kind, rho_value, block_size in regimes:
        if regime_kind == "block":
            dependence_kind = "block"
            regime = f"block (size={block_size})"
        elif rho_value < 0.0:
            dependence_kind = "equicorrelated"
            regime = "negative equicorrelation"
        elif rho_value == 0.0:
            dependence_kind = "independent"
            regime = "independent"
        else:
            dependence_kind = "factor"
            regime = "positive factor"
        cfg = TwoGroupsConfig(
            m=int(experiment["m"]),
            pi0=float(experiment["pi0"]),
            effect=float(experiment["effect"]),
            rho=rho_value,
            sided=int(experiment.get("sided", 1)),
            dependence_kind=cast(DependenceKind, dependence_kind),
            block_size=1 if block_size is None else block_size,
        )
        bh = simulate_procedure(
            cfg,
            "benjamini_hochberg",
            float(experiment["alpha"]),
            n_reps,
            int(experiment.get("seed", 0) if seed is None else seed),
            workers,
        )
        by = simulate_procedure(
            cfg,
            "benjamini_yekutieli",
            float(experiment["alpha"]),
            n_reps,
            int(experiment.get("seed", 0) if seed is None else seed),
            workers,
        )
        row: dict[str, Any] = {
                "rho": rho_value,
                "regime": regime,
                "dependence_kind": dependence_kind,
                "bh_fdr": bh.fdr,
                "bh_power": bh.power,
                "by_fdr": by.fdr,
                "by_power": by.power,
            }
        for prefix, result in (("bh", bh), ("by", by)):
            for metric in ("fdr", "power"):
                accounting = _metric_accounting(result, metric, confidence, float(experiment["target_mc_half_width"]))
                row[f"{prefix}_{metric}_stderr"] = accounting[f"mc_stderr_{metric}"]
                row[f"{prefix}_{metric}_ci95_lower"] = accounting[
                    f"mc_ci95_lower_{metric}"
                ]
                row[f"{prefix}_{metric}_ci95_upper"] = accounting[
                    f"mc_ci95_upper_{metric}"
                ]
                row[f"{prefix}_{metric}_half_width"] = accounting[
                    f"mc_half_width_{metric}"
                ]
                row[f"{prefix}_{metric}_hoeffding_radius"] = accounting[
                    f"mc_hoeffding_radius_{metric}"
                ]
        output.append(row)
    return output


def _sample_size_accounting(
    experiment: dict[str, Any],
    n_reps: int,
    active_reps: int,
    certificate_reps: int,
    calibration: dict[str, Any],
    confidence: float,
) -> dict[str, Any]:
    """Record replication units, raw draws, and conservative precision floors."""
    m = int(experiment["m"])
    effects = len(experiment["effect_grid"])
    rhos = len(experiment["dependence_rhos"]) + int(isinstance(experiment.get("dependence_block"), dict))
    regimes = experiment["active_inference"]["regimes"]
    target = float(experiment["target_mc_half_width"])
    required = required_reps_for_half_width(target, confidence)
    action_config = experiment.get("action_loop", {})
    action_scenarios = len(action_config.get("reliability_grid", [1.0]))
    action_policies = len(
        tuple(value for value in action_config.get("policies", ("fixed_horizon", "posterior_threshold", "cost_aware", "e_process")) if value != "adaptive_target")
    )
    # All action-loop operating-characteristic tracks use the declared
    # replication budget.  Compact trace diagnostics are a separate
    # ``agent_sampling`` block below; they must not silently redefine the
    # sample size of the formal/conditional action-loop comparisons.
    action_reps = int(active_reps)
    active_observations = sum(
        int(raw["horizon"]) * m for raw in regimes.values()
    )
    blocks = {
        "method_comparison": {
            "replications_per_method": n_reps,
            "methods": len(experiment["methods"]),
            "datasets": n_reps * len(experiment["methods"]),
            "hypothesis_draws": n_reps * len(experiment["methods"]) * m,
        },
        "power_curve": {
            "replications_per_effect": n_reps,
            "effect_points": effects,
            "datasets": n_reps * effects,
            "hypothesis_draws": n_reps * effects * m,
        },
        "dependence_sweep": {
            "replications_per_rho_and_method": n_reps,
            "rho_points": rhos,
            "regime_points": rhos,
            "methods": 2,
            "datasets": n_reps * rhos * 2,
            "hypothesis_draws": n_reps * rhos * 2 * m,
        },
        "active_inference": {
            "replications_per_regime": active_reps,
            "regimes": len(regimes),
            "families": active_reps * len(regimes),
            "evidence_streams": active_reps * len(regimes) * m,
            "observation_draws": active_reps * active_observations,
        },
        "null_pvalue_calibration": {
            "replications": int(calibration["n_reps"]),
            "pvalues": int(calibration["n_pvalues"]),
            "hypothesis_draws": int(calibration["n_pvalues"]),
        },
        "certificates": {
            "replications_per_check": certificate_reps,
            "checks": 5,
            "simulation_runs": 5 * certificate_reps,
        },
        "paired_comparisons": {
            "replications": n_reps,
            "method_pairs": max(0, len(experiment["methods"]) - 1),
            "metric_bands": max(0, (len(experiment["methods"]) - 1) * 3),
            "paired_differences": n_reps * max(0, len(experiment["methods"]) - 1) * 3,
        },
        "sequential": {
            "replications": active_reps,
            "horizon": int(regimes["strong_evidence"]["horizon"]),
            "observation_draws": active_reps * int(regimes["strong_evidence"]["horizon"]),
        },
        "online_fdr": {
            "replications": active_reps,
            "horizon": min(256, max(32, int(active_reps // 4))),
            "pvalue_draws": active_reps * min(256, max(32, int(active_reps // 4))) * 2,
        },
        "agent_sampling": {
            "replications": min(64, active_reps),
            "policies": 3,
            "regimes": len(regimes),
            "observation_draws": min(64, active_reps) * 3 * len(regimes) * max(int(raw["horizon"]) for raw in regimes.values()),
        },
        "conditional_power": {
            "replications": active_reps,
            "scenarios": action_scenarios,
            "policies": action_policies,
            "action_steps": active_reps * action_scenarios * action_policies * int(action_config.get("horizon", regimes["strong_evidence"]["horizon"])),
            "paired_policy_replications": action_reps * action_scenarios,
            "chronological_selection_replications": action_reps,
            "future_selection_contrast_replications": action_reps,
            "split_confirmation_replications": action_reps,
            "split_confirmation_streams": 2 * action_reps,
            "learned_policy_train_replications": max(2, int(action_config.get("learned_policy_train_reps", max(2, int(active_reps // 16))))),
            "learned_policy_evaluation_replications": action_reps,
        },
    }
    replication_fields = (
        "replications_per_method", "replications_per_effect", "replications_per_rho_and_method",
        "replications_per_regime", "replications", "replications_per_check",
    )
    ratios = {
        name: float(next((block[field] for field in replication_fields if field in block), 0)) / required
        for name, block in blocks.items()
    }
    return {
        "confidence": confidence,
        "target_half_width": target,
        "worst_case_required_reps": required,
        "method": "Hoeffding bound for independent [0, 1] replication means",
        "blocks": blocks,
        "planned_replication_ratio_to_worst_case_floor": ratios,
    }


def _active_inference_policy(experiment: dict[str, Any], n_reps: int) -> dict[str, Any]:
    ai = experiment.get("active_inference", {})
    regimes: dict[str, Any] = {}
    for name, raw in ai.get("regimes", {}).items():
        config = AITestConfig(
            reliability=float(raw["reliability"]),
            threshold=float(raw["threshold"]),
            horizon=int(raw["horizon"]),
            prior_signal=float(raw.get("prior_signal", 0.5)),
        )
        regimes[str(name)] = evaluate_ai_regime(
            m=int(experiment["m"]),
            pi0=float(experiment["pi0"]),
            config=config,
            n_reps=max(1, int(n_reps * float(raw.get("replication_fraction", 1.0)))),
            alpha=float(experiment["alpha"]),
            seed=int(raw.get("seed", experiment.get("seed", 0))),
        )
    return {
        "alpha": float(experiment["alpha"]),
        "regimes": regimes,
        "note": (
            "A belief threshold is a per-test decision rule, not a multiple-testing "
            "guarantee. The calibrated Binomial evidence p-value can be passed to BH."
        ),
    }


def _conditional_power_suite(experiment: dict[str, Any], n_reps: int, seed: int, workers: int) -> dict[str, Any]:
    """Evaluate operating cells and a separately declared confirmation channel."""
    raw = experiment["active_inference"]["regimes"]["strong_evidence"]
    action_config = experiment.get("action_loop", {})
    reliabilities = [float(value) for value in action_config.get("reliability_grid", [raw["reliability"]])]
    horizon = int(action_config.get("horizon", raw["horizon"]))
    alpha = float(experiment["alpha"])

    scenarios = [
        make_reference_action_scenario(
            reliability=value,
            horizon=horizon,
            alpha=alpha,
            scenario_id=_reference_scenario_identifier(value, horizon, alpha),
        )
        for value in reliabilities
    ]
    scenario = scenarios[0]
    policies: tuple[ActionPolicy, ...] = tuple(
        cast(ActionPolicy, str(value))
        for value in action_config.get(
            "policies",
            ("fixed_horizon", "posterior_threshold", "cost_aware", "e_process"),
        )
    )
    policies = tuple(value for value in policies if value != "adaptive_target")
    operating = evaluate_power_surface(scenarios, policies, n_reps, seed=seed, workers=workers)
    action_reps = int(n_reps)
    paired = tuple(
        paired_action_policy_comparisons(
            scenario_item,
            policies,
            action_reps,
            seed=seed + index,
            run_trace=run_action_loop,
        )
        for index, scenario_item in enumerate(scenarios)
    )
    frontier = tuple(
        {
            "scenario_id": item["scenario_id"],
            "scenario_hash": item["scenario_hash"],
            "scenario_metadata": dict(item.get("scenario_metadata", {})),
            "policy": item["policy"],
            "power": item["summary"]["power"],
            "power_interval": item["summary"]["power_interval"],
            "fdr": item["summary"]["fdr"],
            "fdr_interval": item["summary"].get("fdr_interval", [item["summary"]["fdr"]] * 2),
            "n_reps": item["summary"]["n_reps"],
            "mean_cost": item["summary"]["mean_cost"],
            "cost_interval": item["summary"].get("cost_interval", [item["summary"]["mean_cost"]] * 2),
            "mean_stopping_time": item["summary"]["mean_stopping_time"],
            "power_per_cost": item["summary"].get("power_per_cost", 0.0),
            "validity": item["summary"].get("validity", item.get("validity", "diagnostic")),
        }
        for item in operating
    )
    online = evaluate_adaptive_target_selection(
        scenario, "adaptive_target", action_reps, seed=seed + 1,
    )
    null_calibration = evaluate_conditional_selection_calibration(
        make_reference_null_action_scenario(
            reliability=float(reliabilities[0]),
            horizon=horizon,
            alpha=float(experiment["alpha"]),
        ),
        "adaptive_target",
        action_reps,
        seed=seed + 2,
        workers=workers,
    )
    future_selection_failure = evaluate_future_evidence_selection_failure(
        make_reference_null_action_scenario(
            reliability=float(reliabilities[0]),
            horizon=horizon,
            alpha=float(experiment["alpha"]),
        ),
        action_reps,
        seed=seed + 3,
        workers=workers,
    )
    split_confirmation_scenario = make_static_binary_split_confirmation_scenario(
        reliability=float(reliabilities[0]),
        horizon=int(action_config.get("split_confirmation_steps", horizon)),
        alpha=float(experiment["alpha"]),
    )
    split_stream = evaluate_split_stream_confirmation(
        split_confirmation_scenario, action_reps, seed=seed + 4
    )
    learned_config = LearnedPolicyConfig(
        train_reps=max(2, int(action_config.get("learned_policy_train_reps", max(2, int(n_reps // 16))))),
        cost_penalty=float(action_config.get("learned_policy_cost_penalty", 0.02)),
    )
    learned_artifact = train_tabular_policy(
        scenario, learned_config, seed=seed + 5, workers=workers
    )
    learned_result = evaluate_learned_policy(
        scenario, learned_artifact, action_reps, seed=seed + 6, workers=workers
    )
    return {
        "contract_version": "1.0",
        "suite_name": "active_inference_power_suite",
        "scenarios": [item.scenario_id for item in scenarios],
        "scenario_contracts": [item.to_mapping() for item in scenarios],
        "scenario": scenario.to_mapping(),
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "policies": list(policies),
        "operating_characteristics": operating,
        "online_fdr": online,
        "conditional_selection_calibration": null_calibration,
        "future_evidence_selection_failure": future_selection_failure,
        # This channel is intentionally not one more operating/frontier cell:
        # it declares the static scenario used only for split confirmation.
        "split_confirmation_scenario": split_confirmation_scenario.to_mapping(),
        "split_stream_confirmation": split_stream,
        "learned_policy": {
            "artifact": learned_artifact.to_mapping(),
            "evaluation": learned_result,
            "claim_boundary": "Held-out learned-policy evaluation is conditional on the declared scenario and train/evaluation split; it is not a policy optimality theorem.",
        },
        "paired_policy_comparisons": list(paired),
        "power_cost_frontier": list(frontier),
        "claim_boundary": "Power is conditional on the declared generative model, generative process, testing setting, policy, and finite simulation. Environment-level control is diagnostic unless conditional evidence assumptions are satisfied.",
    }


def _paired_method_comparisons(
    experiment: dict[str, Any], n_reps: int, confidence: float, workers: int = 1,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compare methods on identical replication-level draws.

    Every reported interval is adjusted with a Bonferroni confidence level over
    the method-by-metric family, so the result is a simultaneous finite-sample
    Monte-Carlo band rather than a collection of unadjusted intervals.
    """
    cfg = TwoGroupsConfig(
        m=int(experiment["m"]), pi0=float(experiment["pi0"]),
        effect=float(experiment["effect"]), rho=0.0,
        sided=int(experiment.get("sided", 1)),
    )
    compared = compare_procedures(
        cfg, [str(method) for method in experiment["methods"]],
        float(experiment["alpha"]), n_reps,
        int(experiment.get("seed", 0) if seed is None else seed), workers,
    )
    reference = compared["benjamini_hochberg"]
    entries: dict[str, Any] = {}
    metrics = ("fdr", "power", "fwer")
    adjusted_confidence = 1.0 - (1.0 - confidence) / max(1, (len(compared) - 1) * len(metrics))
    for method, result in compared.items():
        if method == "benjamini_hochberg":
            continue
        entries[method] = {
            metric: paired_difference_summary(
                getattr(reference, f"{'fdp' if metric == 'fdr' else metric}_samples"),
                getattr(result, f"{'fdp' if metric == 'fdr' else metric}_samples"),
                adjusted_confidence,
            )
            for metric in metrics
        }
    return {
        "reference": "benjamini_hochberg",
        "difference_direction": "comparison minus reference",
        "confidence": confidence,
        "simultaneous_confidence": adjusted_confidence,
        "multiplicity_adjustment": "Bonferroni across method-by-metric bands",
        "n_reps": n_reps,
        "comparisons": entries,
    }


def _sequential_analysis(
    experiment: dict[str, Any], n_reps: int, seed: int | None = None
) -> dict[str, Any]:
    """Run valid e-process and deliberately invalid p-value stopping contrasts."""
    raw = experiment["active_inference"]["regimes"]["strong_evidence"]
    reliability = float(raw["reliability"])
    null_rate = 1.0 - reliability
    horizon = int(raw["horizon"])
    alpha = float(experiment["alpha"])
    stage_seed = int(experiment.get("seed", 0) if seed is None else seed)
    rng = np.random.default_rng(np.random.SeedSequence(stage_seed))
    e_crossed: list[float] = []
    p_stopped: list[float] = []
    p_terminal: list[float] = []
    representative: dict[str, Any] | None = None
    for index in range(int(n_reps)):
        observations = rng.binomial(1, null_rate, size=horizon)
        e_result = binary_e_process(observations, null_rate, reliability, alpha)
        confidence_sequence = bernoulli_confidence_sequence(observations, alpha=alpha)
        p_path = [
            fixed_horizon_binomial_pvalue(observations[:step], null_rate, reliability)
            for step in range(1, horizon + 1)
        ]
        p_stop = next((step for step, value in enumerate(p_path, start=1) if value <= alpha), None)
        e_crossed.append(float(e_result.crossed))
        p_stopped.append(float(p_stop is not None))
        p_terminal.append(float(p_path[-1] <= alpha))
        if representative is None:
            log_e_process_values = e_result.log_e_process_values
            assert log_e_process_values is not None
            representative = {
                "observations": observations.astype(int).tolist(),
                "e_process_values": e_result.e_process_values.tolist(),
                "log_e_process_values": log_e_process_values.tolist(),
                "evidence_kind": "likelihood_ratio_e_process",
                "e_boundary": e_result.boundary,
                "e_crossing_time": e_result.crossing_time,
                "fixed_horizon_pvalues": p_path,
                "confidence_sequence_lower": confidence_sequence.lower.tolist(),
                "confidence_sequence_upper": confidence_sequence.upper.tolist(),
                "confidence_sequence_empty": confidence_sequence.empty.tolist(),
            }
    assert representative is not None
    return {
        "null_rate": null_rate,
        "alternative_rate": reliability,
        "alpha": alpha,
        "horizon": horizon,
        "n_reps": int(n_reps),
        "e_process": {
            "crossing_rate": float(np.mean(e_crossed)),
            "crossing_interval_wilson": list(wilson_interval(int(np.sum(e_crossed)), len(e_crossed))),
            "validity": "Ville bound under the declared Bernoulli null and fixed alternative",
        },
        "fixed_horizon_pvalue_stopping": {
            "anytime_rejection_rate": float(np.mean(p_stopped)),
            "terminal_rejection_rate": float(np.mean(p_terminal)),
            "anytime_interval_wilson": list(wilson_interval(int(np.sum(p_stopped)), len(p_stopped))),
            "validity": "contrast only; fixed-horizon p-values are not time-uniform under optional stopping",
        },
        "confidence_sequence": {
            "validity": "time-uniform Bernoulli mixture-e-process inversion",
            "coverage_claim": "simultaneous-in-time coverage for each fixed Bernoulli null rate",
        },
        "representative": representative,
    }


def _online_fdr_analysis(
    experiment: dict[str, Any], n_reps: int, seed: int | None = None
) -> dict[str, Any]:
    """Calibrate LORD++ and SAFFRON on independent null p-value streams."""
    alpha = float(experiment["alpha"])
    horizon = min(256, max(32, int(n_reps // 4)))
    stage_seed = int(experiment.get("seed", 0) if seed is None else seed)
    rng = np.random.default_rng(np.random.SeedSequence(stage_seed))
    # ``default_gamma`` returns a NumPy array, while the public e-LOND
    # interface intentionally accepts an immutable scalar sequence.  Preserve
    # the exact schedule while making that boundary explicit for callers.
    e_lond_gamma: tuple[float, ...] = tuple(
        float(value) for value in default_gamma(horizon)
    )
    # Under U(0,1), c * 1{U <= 1/c} is a bounded mean-one e-value.  Choose c
    # to cross at least the largest initial e-LOND level, so a valid null
    # stream can exercise the rejection-reset recursion rather than being
    # deterministically all zero at the recorded threshold scale.
    e_lond_initial_levels = e_lond(
        np.zeros(horizon), alpha, gamma=e_lond_gamma
    ).levels
    e_lond_support = max(
        1, int(np.ceil(1.0 / float(np.max(e_lond_initial_levels))))
    )
    summaries: dict[str, Any] = {}
    for method in ("lord_plus_plus", "saffron", "e_lond"):
        any_rejections: list[float] = []
        counts: list[float] = []
        trace: Any = None
        for _ in range(int(n_reps)):
            pvalues = rng.random(horizon)
            if method == "lord_plus_plus":
                result: OnlineFDRResult | EValueLONDResult = lord_plus_plus(pvalues, alpha)
                result_rejections = result.num_rejections
            elif method == "saffron":
                result = saffron(pvalues, alpha)
                result_rejections = result.num_rejections
            else:
                # This bounded map has expectation one under U(0,1). Unlike
                # a {0, 2} map, its support reaches a recorded e-LOND level;
                # the nonintegrable 1/p transform remains excluded.
                evalues = np.where(
                    pvalues <= 1.0 / e_lond_support,
                    float(e_lond_support),
                    0.0,
                )
                result = e_lond(evalues, alpha, gamma=e_lond_gamma)
                result_rejections = result.num_rejections
            any_rejections.append(float(result_rejections > 0))
            counts.append(float(result_rejections))
            choose_trace = trace is None
            if method == "e_lond" and trace is not None and result.num_rejections:
                choose_trace = not bool(trace.get("num_rejections", 0))
            if choose_trace:
                if method == "e_lond":
                    e_result = cast(EValueLONDResult, result)
                    trace = {
                        "alpha": float(e_result.alpha),
                        "evalues": e_result.evalues.tolist(),
                        "levels": e_result.levels.tolist(),
                        "thresholds": e_result.thresholds.tolist(),
                        "rejections": e_result.rejections.astype(bool).tolist(),
                        "accounting_path": e_result.cumulative_allocations.tolist(),
                        "gamma": e_result.gamma.tolist(),
                        "accounting_kind": "cumulative_nominal_level_allocation",
                        "num_rejections": e_result.num_rejections,
                        "e_value_null_transform": "c * 1{U <= 1/c}; E_null[e] = 1",
                        "e_value_support": e_lond_support,
                        "trace_role": "first_seeded_rejection_bearing_valid_null_trace",
                    }
                else:
                    online_result = cast(OnlineFDRResult, result)
                    trace = {
                        "alpha": float(online_result.alpha),
                        # Retain the seeded null stream so strict validation
                        # can recompute the predictable recursion rather than
                        # merely trusting a serialized level/accounting path.
                        "pvalues": online_result.pvalues.tolist(),
                        "initial_wealth": float(online_result.initial_wealth),
                        "lambda": None if online_result.lambda_ is None else float(online_result.lambda_),
                        "levels": online_result.levels.tolist(),
                        "candidates": online_result.candidates.astype(bool).tolist(),
                        "rejections": online_result.rejections.astype(bool).tolist(),
                        "accounting_path": online_result.wealth.tolist(),
                        "gamma": online_result.gamma.tolist(),
                        "accounting_kind": "procedural_alpha_wealth",
                    }
        assert trace is not None
        if method == "e_lond":
            trace["trace_role"] = (
                "first_seeded_rejection_bearing_valid_null_trace"
                if bool(trace.get("num_rejections", 0))
                else "first_seeded_valid_null_trace_no_rejection"
            )
        summaries[method] = {
            "mean_rejections": float(np.mean(counts)),
            "any_rejection_rate": float(np.mean(any_rejections)),
            "any_rejection_interval_wilson": list(wilson_interval(int(np.sum(any_rejections)), len(any_rejections))),
            "trace": trace,
        }
        if method == "e_lond":
            summaries[method]["null_e_value_transform"] = (
                "c * 1{U <= 1/c}; E_null[e] = 1"
            )
            summaries[method]["null_e_value_support"] = e_lond_support
    return {
        "null": True,
        "alpha": alpha,
        "horizon": horizon,
        "n_reps": int(n_reps),
        "methods": summaries,
        "uncertainty": "Wilson intervals summarize the finite null-simulation event rate; they are not a proof of online FDR control.",
    }


def _agent_sampling_analysis(
    experiment: dict[str, Any], n_reps: int, seed: int | None = None
) -> dict[str, Any]:
    """Summarize fixed, posterior-stop, and e-process policies.

    The policy summaries use the real pymdp-backed implementation. A compact
    trace is retained for visual inspection; the larger fixed-horizon regime
    remains the primary operating-characteristic experiment.
    """
    policy_reps = min(64, int(n_reps))
    output: dict[str, Any] = {"n_reps": policy_reps, "regimes": {}}
    for name, raw in experiment["active_inference"]["regimes"].items():
        config = AITestConfig(
            reliability=float(raw["reliability"]), threshold=float(raw["threshold"]),
            horizon=int(raw["horizon"]), prior_signal=float(raw.get("prior_signal", 0.5)),
        )
        policies: dict[str, Any] = {}
        policies_to_run: tuple[PolicyName, ...] = (
            "fixed_horizon",
            "posterior_threshold",
            "e_process",
        )
        for offset, policy in enumerate(policies_to_run):
            base_seed = int(raw.get("seed", 0) if seed is None else seed)
            simulation = simulate_agent_policy(config, policy, policy_reps, seed=base_seed + offset)
            trace = cast(list[AgentSamplingTrace], simulation["traces"])[0]
            policies[policy] = {
                "decision_rate": simulation["decision_rate"],
                "mean_stopping_time": simulation["mean_stopping_time"],
                "optional_stopping_valid": policy == "e_process",
                "trace": {
                    "actions": trace.actions.tolist(),
                    "observations": trace.observations.tolist(),
                    "posterior_signal": trace.posterior_signal.tolist(),
                    "e_process_values": trace.e_process_values.tolist(),
                    "evidence_kind": "likelihood_ratio_e_process",
                    "stopping_time": trace.stopping_time,
                    "decision": trace.decision,
                    "fixed_horizon_pvalue": trace.fixed_horizon_pvalue,
                },
            }
        output["regimes"][name] = {
            "config": {"reliability": config.reliability, "horizon": config.horizon, "threshold": config.threshold},
            "policies": policies,
            "claim_boundary": "posterior and fixed-horizon decisions are not family-level evidence; only the e-process policy has an optional-stopping validity label.",
        }
    return output


def _benchmark_analysis(experiment: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    """Load only committed benchmark fixtures; never retrieve external data."""
    manifest_path = ROOT / "data" / "benchmarks" / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    records = {str(item["id"]): item for item in manifest["benchmarks"]}
    agent_record = records["public_agent_trace_adapter_stage_2"]
    agent_path = ROOT / "data" / "benchmarks" / str(agent_record["filename"])
    spec = BenchmarkSpec(
        benchmark_id=str(agent_record["id"]), source=str(agent_record["source"]),
        license=str(agent_record["license"]), sha256=str(agent_record["sha256"]),
        claim_role=str(agent_record["claim_role"]),
    )
    dataset = load_agent_trace(agent_path, spec)
    golub_record = records["golub_normalized_offline_fixture"]
    golub_spec = BenchmarkSpec(
        benchmark_id=str(golub_record["id"]), source=str(golub_record["source"]),
        license=str(golub_record["license"]), sha256=str(golub_record["sha256"]),
        claim_role=str(golub_record["claim_role"]), doi=str(golub_record.get("doi", "")),
        retrieval_date=str(golub_record.get("retrieval_date", "")),
        schema=str(golub_record.get("schema", "")),
        preprocessing=str(golub_record.get("preprocessing", "")),
        format=str(golub_record.get("format", "")),
    )
    golub_dataset = load_normalized_golub_fixture(
        ROOT / "data" / "benchmarks" / str(golub_record["filename"]), golub_spec
    )
    observations = dataset.values[:, 1].astype(int)
    benchmark_e = binary_e_process(observations, 0.5, 0.75, float(experiment["alpha"]))
    synthetic_seed = int(experiment.get("seed", 0) if seed is None else seed)
    synthetic_rng = np.random.default_rng(np.random.SeedSequence(synthetic_seed))
    synthetic_e = binary_e_process(
        synthetic_rng.binomial(1, 0.5, size=observations.size), 0.5, 0.75, float(experiment["alpha"])
    )
    return {
        "offline_only": True,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "golub": {
            "id": records["golub_leukemia_bioconductor_1_54_0"]["id"],
            "normalized_fixture_id": golub_spec.benchmark_id,
            "available": True,
            "fixture_rows": int(golub_dataset.values.shape[0]),
            "fixture_columns": list(golub_dataset.columns),
            "fixture_sha256": golub_spec.sha256,
            "full_archive_available": False,
            "reason": "committed normalized fixture is available offline; the checksum-pinned full archive is never downloaded during ordinary generation",
            "claim_role": records["golub_leukemia_bioconductor_1_54_0"]["claim_role"],
            "fixture_claim_role": golub_spec.claim_role,
        },
        "agent_trace": {
            "id": spec.benchmark_id,
            "rows": int(observations.size),
            "subjects": int(dataset.metadata.get("n_subjects", 0) if dataset.metadata else 0),
            "chronological": bool(dataset.metadata.get("chronological", False) if dataset.metadata else False),
            "missingness": str(dataset.metadata.get("missingness", "not recorded") if dataset.metadata else "not recorded"),
            "terminal_e": benchmark_e.terminal_e,
            "crossed": benchmark_e.crossed,
            "claim_role": spec.claim_role,
        },
        "synthetic_reference": {
            "n": int(observations.size),
            "terminal_e": synthetic_e.terminal_e,
            "claim_role": "synthetic calibration context only",
        },
        "provenance_boundary": "External benchmarks are robustness stress tests and cannot upgrade synthetic guarantees.",
    }


def _write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def _certificate_gate_exit_status(results: dict[str, Any]) -> int:
    """Return the CLI status for exact terminal certificate verdicts.

    Every certificate remains visible in the artifact.  Only names explicitly
    declared diagnostic are non-terminal; malformed or absent modern metadata
    falls back to the conservative all-terminal policy or fails closed.
    """
    certificates = results.get("certificates")
    if not isinstance(certificates, dict) or not certificates:
        return 1
    metadata = results.get("meta")
    serialized_gate = (
        metadata.get("certificate_gate") if isinstance(metadata, dict) else None
    )
    try:
        gate = effective_certificate_gate_result(certificates, serialized_gate)
    except ValueError:
        return 1
    return 0 if gate["terminal_certificates_pass"] else 1


def _run_logged_step(
    events: list[dict[str, Any]], name: str, operation: Any, ledger: RunLedger | None = None
) -> Any:
    """Run one analysis stage and append structured duration accounting."""
    started = time.perf_counter()
    LOGGER.info("step.start name=%s", name)
    if ledger is not None:
        ledger.event(name, "started", elapsed_seconds=0.0)
    try:
        value = operation()
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        events.append({"name": name, "status": "failed", "elapsed_seconds": elapsed})
        if ledger is not None:
            ledger.event(
                name,
                "failed",
                elapsed_seconds=elapsed,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        raise
    elapsed = time.perf_counter() - started
    events.append({"name": name, "status": "passed", "elapsed_seconds": elapsed})
    if ledger is not None:
        ledger.event(name, "passed", elapsed_seconds=elapsed)
    LOGGER.info("step.complete name=%s elapsed_seconds=%.3f", name, elapsed)
    return value


def _invalidate_existing_manifest(output: Path) -> list[str]:
    """Atomically revoke completion before a new run reuses an output root.

    A result run writes its ledger and stage artifacts incrementally.  Leaving
    a prior ``complete: true`` manifest in place during that interval could
    falsely advertise the old publication bundle while its recorded files are
    being replaced.  Preserve the physical artifacts for inspection, but
    remove their renderer records and persist an explicitly incomplete state
    before the new ledger is created.
    """
    path = output / "manifest.json"
    if not path.is_file():
        return []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"existing manifest is not valid JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"existing manifest must be a JSON object: {path}")
    validate_manifest(manifest)
    invalidate_renderer_artifact_records(manifest)
    pending = [
        "result_generation",
        "quality.json",
        "manuscript_hydration",
        "publication_render",
        "release_finalization",
    ]
    manifest["complete"] = False
    manifest["validation_pending"] = pending
    validate_manifest(manifest)
    atomic_write_json(path, manifest)
    return pending


def run(
    config_path: Path,
    *,
    profile: str | None = None,
    workers: int = 1,
    output_root: Path | None = None,
    resume: bool = False,
    checkpoint_every: int = 0,
    seed_override: int | None = None,
) -> dict[str, Any]:
    """Run the configured analysis with atomic artifacts and a live ledger."""
    started_clock = time.perf_counter()
    events: list[dict[str, Any]] = []
    selected_profile = profile or "release"
    if selected_profile not in {"fast", "release", "campaign", "publication", "benchmark"}:
        raise ValueError(f"unknown profile: {selected_profile!r}")
    if isinstance(workers, bool) or int(workers) != workers or int(workers) < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    if (
        isinstance(checkpoint_every, bool)
        or not isinstance(checkpoint_every, Real)
        or not math.isfinite(float(checkpoint_every))
        or int(checkpoint_every) != checkpoint_every
        or checkpoint_every < 0
    ):
        raise ValueError("checkpoint_every must be a non-negative integer")
    if seed_override is not None and (
        isinstance(seed_override, bool)
        or not isinstance(seed_override, Real)
        or not math.isfinite(float(seed_override))
        or int(seed_override) != seed_override
        or seed_override < 0
    ):
        raise ValueError("seed_override must be a non-negative integer when provided")
    output = Path(output_root) if output_root is not None else ROOT / "output"
    if not output.is_absolute():
        output = ROOT / output
    output = require_safe_output_root(output, project_root=ROOT)
    config_hash = _safe_input_hash(config_path)
    execution_digest = execution_source_sha256(ROOT)
    try:
        config = _load_config(config_path)
        # Resolve first because every output profile emits the current result
        # schema, including the campaign profile when a new campaign is run.
        experiment = _resolve_experiment_profile(
            dict(config["experiment"]), selected_profile
        )
        complexity_plan = _require_generated_complexity_plan(experiment, config_path)
    except BaseException as exc:
        # Do not mutate a retained output namespace when a predecessor source
        # is being inspected or an invalid current plan is rejected. A new root
        # still receives the normal failure ledger for operational diagnosis.
        retained_namespace = any(
            (output / name).exists()
            for name in ("manifest.json", "results.json", "run_log.json")
        )
        if not retained_namespace:
            output.mkdir(parents=True, exist_ok=True)
            preflight_ledger = RunLedger(
                output / "run_log.json",
                config_sha256=config_hash,
                profile=selected_profile,
                seed=0 if seed_override is None else int(seed_override),
                workers=workers,
                execution_source_sha256=execution_digest,
            )
            preflight_ledger.data.update(
                {"resume": bool(resume), "checkpoint_every": int(checkpoint_every)}
            )
            preflight_ledger.fail(exc)
        raise
    # Validate frozen campaign receipts before revoking an existing manifest.
    # A typo, missing predecessor, or mismatched lane root must leave a prior
    # campaign receipt untouched rather than replacing it with a failed run.
    campaign_predecessors = (
        campaign_lane_predecessor_artifacts(
            config_path,
            project_root=ROOT,
            output_root=output,
        )
        if selected_profile == "campaign"
        else {}
    )
    output.mkdir(parents=True, exist_ok=True)
    _invalidate_existing_manifest(output)
    ledger = RunLedger(
        output / "run_log.json",
        config_sha256=config_hash,
        profile=selected_profile,
        seed=0 if seed_override is None else int(seed_override),
        workers=workers,
        execution_source_sha256=execution_digest,
    )
    ledger.data.update({"resume": bool(resume), "checkpoint_every": int(checkpoint_every)})
    ledger.flush()
    try:
        claim_ledger_hash = _load_claim_ledger()
        certificate_gate_policy = _certificate_gate_policy(experiment, config_path)
        # Persist the normalized effective policy even for a config that
        # predates this explicit field.
        # that omitted the field. This makes every generated result explicit.
        experiment["certificate_gate_policy"] = certificate_gate_policy
        if seed_override is not None:
            experiment["seed"] = int(seed_override)
        fast_mode = selected_profile == "fast"
        multiplier = float(experiment.get("fast_replication_fraction", 0.3)) if fast_mode else 1.0
        n_reps = max(1, int(round(int(experiment["n_reps"]) * multiplier)))
        active_reps = max(
            4,
            int(
                round(
                    int(experiment.get("active_inference_reps", max(1, n_reps // 6)))
                    * multiplier
                )
            ),
        )
        confidence = float(experiment["mc_confidence"])
        calibration_reps = max(1, int(round(int(experiment["calibration_n_reps"]) * multiplier)))
        certificate_reps = max(1, int(round(int(experiment["certificate_n_reps"]) * multiplier)))
        seed = int(experiment.get("seed", 0))
        stage_seeds = _stage_seed_schedule(seed)
        # Certificate and calibration seeds are explicit preregistered inputs;
        # the remaining stages derive independent children from the master seed.
        stage_seeds["certificates"] = int(experiment.get("certificate_seed", stage_seeds["certificates"]))
        stage_seeds["null_pvalue_calibration"] = int(experiment["calibration_seed"])
        ledger.data.update({
            "seed": seed,
            "claim_ledger_sha256": claim_ledger_hash,
            "n_reps": n_reps,
            "active_inference_reps": active_reps,
            "calibration_n_reps": calibration_reps,
            "stage_seed_schedule": stage_seeds,
            "certificate_gate_policy": certificate_gate_policy,
        })
        ledger.flush()
        try:
            config_display_path = str(config_path.relative_to(ROOT))
        except ValueError:
            config_display_path = str(config_path)
        LOGGER.info("analysis.start profile=%s workers=%s m=%s n_reps=%s active_reps=%s", selected_profile, workers, experiment["m"], n_reps, active_reps)
        checkpoint_dir = output / "checkpoints"
        stage_state: _StageState = {
            "completed": 0,
            "total": len(STAGE_NAMES),
            "elapsed_seconds": [],
        }
        ledger.data["progress"] = {
            "completed_stages": 0,
            "total_stages": stage_state["total"],
            "completed_replications": 0,
            "estimated_remaining_seconds": None,
        }
        ledger.flush()

        def record_progress(stage_name: str, elapsed: float = 0.0) -> None:
            stage_state["completed"] += 1
            stage_state["elapsed_seconds"].append(float(elapsed))
            average = float(sum(stage_state["elapsed_seconds"]) / len(stage_state["elapsed_seconds"]))
            completed_replications = sum(
                stage_replications[name]
                for name in STAGE_NAMES[: stage_state["completed"]]
            )
            ledger.data["progress"] = {
                "completed_stages": stage_state["completed"],
                "total_stages": stage_state["total"],
                "completed_replications": completed_replications,
                "estimated_remaining_seconds": max(0.0, (stage_state["total"] - stage_state["completed"]) * average),
                "last_completed_stage": stage_name,
            }
            ledger.flush()

        stage_replications = {
            "certificates": certificate_reps,
            "method_comparison": n_reps,
            "paired_comparisons": n_reps,
            "power_curve": n_reps,
            "dependence_sweep": n_reps,
            "active_inference": active_reps,
            "sequential": active_reps,
            "online_fdr": active_reps,
            "agent_sampling": min(64, active_reps),
            "conditional_power": active_reps,
            "benchmarks": 1,
            "null_pvalue_calibration": calibration_reps,
        }

        def _run_chunked_null_calibration() -> dict[str, Any]:
            """Run null calibration with recoverable replication blocks.

            The canonical deterministic path remains the sequential
            ``null_pvalue_calibration`` call when checkpointing is disabled.
            When a checkpoint is requested, each replication receives a stable
            child stream and the partial accumulator stores the raw p-values
            needed to reproduce the exact ECDF/KS/DKW summary after restart.
            """
            name = "null_pvalue_calibration"
            checkpoint = checkpoint_dir / f"{name}.json"
            total = stage_replications[name]
            expected_dependencies = _expected_checkpoint_dependency_hashes(name, experiment)
            completed = 0
            samples = np.empty(0, dtype=float)
            if resume and checkpoint.is_file():
                saved = load_checkpoint(checkpoint)
                if (
                    saved.get("schema_version") != "2.0"
                    or saved.get("config_sha256") != config_hash
                    or saved.get("claim_ledger_sha256") != claim_ledger_hash
                    or saved.get("execution_source_sha256") != execution_digest
                    or saved.get("profile") != selected_profile
                    or saved.get("seed") != seed
                    or saved.get("stage_seed") != stage_seeds[name]
                    or saved.get("dependency_hashes") != expected_dependencies
                ):
                    raise ValueError(f"checkpoint {checkpoint} does not match this run contract")
                if saved.get("partial_metric_accumulators", {}).get("status") == "stage_complete":
                    if saved.get("completed_replications") != total:
                        raise ValueError(f"checkpoint {checkpoint} has mismatched replication accounting")
                    return saved["value"]
                saved_total = int(saved.get("total_replications", saved["value"].get("total_replications", 0)))
                if saved_total != total:
                    raise ValueError(f"checkpoint {checkpoint} has mismatched replication accounting")
                completed = int(saved["completed_replications"])
                raw_samples = saved["value"].get("samples")
                if not isinstance(raw_samples, list):
                    raise ValueError(f"checkpoint {checkpoint} is missing its partial sample accumulator")
                samples = np.asarray(raw_samples, dtype=float)
                expected_samples = completed * int(experiment["m"])
                if samples.ndim != 1 or samples.size != expected_samples:
                    raise ValueError(f"checkpoint {checkpoint} has an inconsistent partial sample accumulator")

            chunk_size = max(1, int(checkpoint_every) if checkpoint_every else total)
            while completed < total:
                next_completed = min(total, completed + chunk_size)
                block = calibration_replication_samples(
                    TwoGroupsConfig(
                        m=int(experiment["m"]),
                        pi0=1.0,
                        effect=0.0,
                        rho=0.0,
                        sided=int(experiment.get("sided", 1)),
                    ),
                    total,
                    seed=stage_seeds[name],
                    start=completed,
                    stop=next_completed,
                )
                samples = np.concatenate((samples, block))
                completed = next_completed
                write_partial_checkpoint(
                    checkpoint,
                    run_id=ledger.run_id,
                    config_sha256=config_hash,
                    claim_ledger_sha256=claim_ledger_hash,
                    execution_source_sha256=execution_digest,
                    profile=selected_profile,
                    seed=seed,
                    stage_seed=stage_seeds[name],
                    workers=workers,
                    stage=name,
                    total_replications=total,
                    completed_replications=completed,
                    chunk_size=chunk_size,
                    metrics={
                        "sample_count": int(samples.size),
                        "sample_sum": float(np.sum(samples)),
                        "sample_min": float(np.min(samples)),
                        "sample_max": float(np.max(samples)),
                    },
                    dependency_hashes=expected_dependencies,
                    value={
                        "total_replications": total,
                        "samples": samples.tolist(),
                    },
                )
                ledger.event(
                    name,
                    "checkpoint",
                    completed_replications=completed,
                    total_replications=total,
                    checkpoint=_display_path(checkpoint, output),
                )
                ledger.data["progress"].update({
                    "active_stage": name,
                    "completed_replications": sum(
                        stage_replications[item] for item in STAGE_NAMES[: STAGE_NAMES.index(name)]
                    ) + completed,
                    "stage_completed_replications": completed,
                    "stage_total_replications": total,
                })
                ledger.flush()
            return null_pvalue_calibration_from_samples(
                TwoGroupsConfig(
                    m=int(experiment["m"]),
                    pi0=1.0,
                    effect=0.0,
                    rho=0.0,
                    sided=int(experiment.get("sided", 1)),
                ),
                total,
                samples,
                seed=stage_seeds[name],
                confidence=confidence,
            )

        def stage(name: str, operation: Any) -> Any:
            checkpoint = checkpoint_dir / f"{name}.json"
            if name == "null_pvalue_calibration" and (checkpoint_every or resume):
                value = _run_logged_step(events, name, _run_chunked_null_calibration, ledger)
                record_progress(name, float(events[-1]["elapsed_seconds"]))
                atomic_write_json(
                    checkpoint,
                    {
                        "schema_version": "2.0",
                        "run_id": ledger.run_id,
                        "config_sha256": config_hash,
                        "claim_ledger_sha256": claim_ledger_hash,
                        "execution_source_sha256": execution_digest,
                        "profile": selected_profile,
                        "seed": seed,
                        "stage_seed": stage_seeds[name],
                        "seed_schedule": {
                            "kind": "numpy.SeedSequence",
                            "master_seed": seed,
                            "stage": name,
                            "stage_seed": stage_seeds[name],
                            "replication_order": "stable ascending child-stream order",
                        },
                        "workers": workers,
                        "stage": name,
                        "total_replications": stage_replications[name],
                        "chunk_size": max(1, int(checkpoint_every) if checkpoint_every else stage_replications[name]),
                        "replication_cursor": {
                            "completed": stage_replications[name],
                            "next": stage_replications[name],
                        },
                        "completed_replications": stage_replications[name],
                        "partial_metric_accumulators": {
                            "status": "stage_complete",
                            "completed_replications": stage_replications[name],
                            "metrics": {"stage_value_hash": _checkpoint_dependency_hashes(name, value)},
                        },
                        "dependency_hashes": _checkpoint_dependency_hashes(name, value),
                        "value": value,
                    },
                )
                return value
            if resume and checkpoint.is_file():
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                validate_checkpoint(saved)
                if (
                    saved.get("schema_version") != "2.0"
                    or saved.get("config_sha256") != config_hash
                    or saved.get("claim_ledger_sha256") != claim_ledger_hash
                    or saved.get("execution_source_sha256") != execution_digest
                    or saved.get("profile") != selected_profile
                    or saved.get("seed") != seed
                    or saved.get("stage_seed") != stage_seeds[name]
                ):
                    raise ValueError(f"checkpoint {checkpoint} does not match this run contract")
                if saved.get("completed_replications") != stage_replications[name]:
                    raise ValueError(f"checkpoint {checkpoint} has mismatched replication accounting")
                recomputed_dependencies = _checkpoint_dependency_hashes(name, saved["value"])
                if saved.get("dependency_hashes") != recomputed_dependencies:
                    raise ValueError(f"checkpoint {checkpoint} has inconsistent dependency hashes")
                expected_dependencies = _expected_checkpoint_dependency_hashes(name, experiment)
                if saved.get("dependency_hashes") != expected_dependencies:
                    raise ValueError(f"checkpoint {checkpoint} has mismatched current dependency hashes")
                ledger.event(
                    name,
                    "resumed",
                    checkpoint=_display_path(checkpoint, output),
                    elapsed_seconds=0.0,
                )
                record_progress(name)
                return saved["value"]
            value = _run_logged_step(events, name, operation, ledger)
            record_progress(name, float(events[-1]["elapsed_seconds"]))
            if checkpoint_every or resume:
                atomic_write_json(
                    checkpoint,
                    {
                        "schema_version": "2.0",
                        "run_id": ledger.run_id,
                        "config_sha256": config_hash,
                        "claim_ledger_sha256": claim_ledger_hash,
                        "execution_source_sha256": execution_digest,
                        "profile": selected_profile,
                        "seed": seed,
                        "stage_seed": stage_seeds[name],
                        "seed_schedule": {
                            "kind": "numpy.SeedSequence",
                            "master_seed": seed,
                            "stage": name,
                            "stage_seed": stage_seeds[name],
                            "replication_order": "stable ascending child-stream order",
                        },
                        "workers": workers,
                        "stage": name,
                        "total_replications": stage_replications[name],
                        "chunk_size": max(1, int(checkpoint_every) if checkpoint_every else stage_replications[name]),
                        "replication_cursor": {
                            "completed": stage_replications[name],
                            "next": stage_replications[name],
                        },
                        "completed_replications": stage_replications[name],
                        "partial_metric_accumulators": {
                            "status": "stage_complete",
                            "completed_replications": stage_replications[name],
                            "metrics": {"stage_value_hash": _checkpoint_dependency_hashes(name, value)},
                        },
                        "dependency_hashes": _checkpoint_dependency_hashes(name, value),
                        "value": value,
                    },
                )
            return value

        split_storey = experiment["storey_split_confirmation"]
        certificates = stage(
            "certificates",
            lambda: run_all_certificates(
                fast=fast_mode,
                m=int(experiment["certificate_m"]),
                n_reps=certificate_reps,
                seed=stage_seeds["certificates"],
                storey_lambda=float(split_storey["lambda"]),
                storey_confidence=float(split_storey["confidence"]),
            ),
        )
        certificate_gate = certificate_gate_result(
            certificates, certificate_gate_policy
        )
        # Record the complete pass/fail partition as soon as certificate work
        # completes, including diagnostic failures that do not block later
        # analysis stages.
        ledger.data["certificate_gate"] = certificate_gate
        ledger.flush()
        method_comparison = stage("method_comparison", lambda: _method_comparison(experiment, n_reps, confidence, workers, stage_seeds["method_comparison"]))
        paired_comparisons = stage("paired_comparisons", lambda: _paired_method_comparisons(experiment, n_reps, confidence, workers, stage_seeds["paired_comparisons"]))
        power_curve_result = stage("power_curve", lambda: _bh_power_curve(experiment, n_reps, confidence, workers, stage_seeds["power_curve"]))
        dependence_result = stage("dependence_sweep", lambda: _dependence_sweep(experiment, n_reps, confidence, workers, stage_seeds["dependence_sweep"]))
        active_result = stage("active_inference", lambda: _active_inference_policy(experiment, active_reps))
        sequential_result = stage("sequential", lambda: _sequential_analysis(experiment, active_reps, stage_seeds["sequential"]))
        online_fdr_result = stage("online_fdr", lambda: _online_fdr_analysis(experiment, active_reps, stage_seeds["online_fdr"]))
        agent_sampling_result = stage("agent_sampling", lambda: _agent_sampling_analysis(experiment, active_reps, stage_seeds["agent_sampling"]))
        conditional_power_result = stage("conditional_power", lambda: _conditional_power_suite(experiment, active_reps, stage_seeds["conditional_power"], workers))
        benchmark_result = stage("benchmarks", lambda: _benchmark_analysis(experiment, stage_seeds["benchmarks"]))
        calibration_result = stage("null_pvalue_calibration", lambda: null_pvalue_calibration(TwoGroupsConfig(m=int(experiment["m"]), pi0=1.0, effect=0.0, rho=0.0, sided=int(experiment.get("sided", 1))), n_reps=calibration_reps, seed=stage_seeds["null_pvalue_calibration"], confidence=confidence))

        results: dict[str, Any] = {
            "schema_version": "2.0",
            "run_id": ledger.run_id,
            "meta": {
                "alpha": float(experiment["alpha"]), "n_reps": n_reps, "fast": bool(fast_mode), "seed": seed,
                "config_path": config_display_path, "config_sha256": config_hash,
                "claim_ledger_sha256": claim_ledger_hash,
                "execution_source_sha256": execution_digest,
                "python": platform.python_version(), "platform": platform.platform(),
                "methods": [str(method) for method in experiment["methods"]],
                "mc_confidence": confidence, "target_mc_half_width": float(experiment["target_mc_half_width"]),
                "active_inference_reps": active_reps,
                "figure_error_multiplier": float(experiment.get("figure_error_multiplier", 3.0)),
                "figure_dpi": int(experiment.get("figure_dpi", 300)),
                "pymdp_version": str(experiment.get("pymdp_version", "1.0.5")),
                "profile": selected_profile,
                "stage_seed_schedule": stage_seeds,
                "certificate_gate": certificate_gate,
            },
            "experiment": experiment, "certificates": certificates,
            "method_comparison": method_comparison, "power_curve": power_curve_result,
            "dependence_sweep": dependence_result, "active_inference": active_result,
            "paired_comparisons": paired_comparisons,
            "sequential": sequential_result, "online_fdr": online_fdr_result,
            "agent_sampling": agent_sampling_result,
            "suite": conditional_power_result,
            "action_loop": {
                "contract_version": "1.0",
                "scenarios": conditional_power_result["scenario_contracts"],
                "policies": conditional_power_result["policies"],
                "operating_characteristics": conditional_power_result["operating_characteristics"],
                "online_fdr": conditional_power_result["online_fdr"],
                "paired_policy_comparisons": conditional_power_result["paired_policy_comparisons"],
                "power_cost_frontier": conditional_power_result["power_cost_frontier"],
                "split_stream_confirmation": conditional_power_result["split_stream_confirmation"],
                "learned_policy": conditional_power_result["learned_policy"],
                "claim_boundary": conditional_power_result["claim_boundary"],
            },
            "benchmarks": benchmark_result,
            "null_pvalue_calibration": calibration_result,
            "formalism": formalism_registry().to_mapping(),
        }
        results["sample_size_accounting"] = _sample_size_accounting(experiment, n_reps, active_reps, certificate_reps, results["null_pvalue_calibration"], confidence)
        action_config = experiment.get("action_loop", {})
        results["computational_complexity"] = build_computational_complexity(
            plan=complexity_plan,
            m=int(experiment["m"]),
            main_replications=n_reps,
            method_count=len(experiment["methods"]),
            effect_points=len(experiment["effect_grid"]),
            dependence_regime_count=len(dependence_result),
            sequential_horizon=int(sequential_result["horizon"]),
            online_horizon=int(online_fdr_result["horizon"]),
            action_horizon=int(action_config.get("horizon", 1)),
            split_confirmation_horizon=int(
                action_config.get("split_confirmation_steps", action_config.get("horizon", 1))
            ),
            suite=conditional_power_result,
        )
        results_path = output / "results.json"
        _write_json(results_path, results)
        _write_json(output / "certificates.json", results["certificates"])
        action_loop_path = output / "action_loop_suite.json"
        _write_json(action_loop_path, results["suite"])
        formalism_path = output / "formalism_registry.json"
        _write_json(formalism_path, results["formalism"])
        demo_path = output / "demo" / "payload.json"
        demo_raw = experiment["active_inference"]["regimes"]["strong_evidence"]
        _write_json(
            demo_path,
            build_demo_payload(
                reliability=float(experiment.get("action_loop", {}).get("reliability_grid", [demo_raw["reliability"]])[0]),
                horizon=int(experiment.get("action_loop", {}).get("horizon", demo_raw["horizon"])),
                policy="cost_aware",
                seed=int(stage_seeds["conditional_power"]),
            ),
        )
        figures = generate_figures(results, output / "figures")
        figure_registry = write_figure_registry(output / "figures", figures, results)
        dashboard_path = write_dashboard(results, output / "review" / "dashboard.html", figures)
        validate_results(results, require_suite=True)
        certificates_pass = bool(certificate_gate["terminal_certificates_pass"])
        artifacts: dict[str, dict[str, Any]] = {
            "config": _artifact(config_path, output, "configuration"),
            "claim_ledger": _artifact(ROOT / "data" / "claim_ledger.yaml", output, "claim_ledger"),
            "results": _artifact(results_path, output, "deterministic_results"),
            "certificates": _artifact(output / "certificates.json", output, "certificate_results"),
            "action_loop_suite": _artifact(action_loop_path, output, "action_loop_results"),
            "formalism_registry": _artifact(formalism_path, output, "formalism_registry"),
            "demo_payload": _artifact(demo_path, output, "demo_payload"),
            "figure_registry": _artifact(Path(figure_registry), output, "figure_registry"),
            "dashboard": _artifact(Path(dashboard_path), output, "review_dashboard"),
        }
        artifacts.update(_benchmark_artifact_records(output))
        for name, (predecessor_path, role) in campaign_predecessors.items():
            artifacts[name] = _artifact(predecessor_path, output, role)
        for name, path in figures.items():
            artifacts[f"figure:{name}"] = _artifact(Path(path), output, "publication_figure")
        if certificates_pass:
            ledger.finish(
                "passed",
                elapsed_seconds=time.perf_counter() - started_clock,
                certificates_pass=True,
                certificate_gate=certificate_gate,
                claim_boundary="Finite simulations and calibration diagnostics are evidence under declared assumptions, not universal proofs.",
            )
        else:
            # A completed computation is not a passed release when a
            # predeclared finite-simulation certificate fails. Keep all
            # generated evidence for diagnosis, but make the operational
            # ledger fail closed so downstream finalizers cannot mislabel it.
            ledger.fail(
                RuntimeError(
                    "one or more terminal finite-simulation certificates failed"
                ),
                certificates_pass=False,
                certificate_gate=certificate_gate,
            )
        artifacts["run_log"] = _artifact(output / "run_log.json", output, "operational_telemetry")
        for checkpoint in sorted(checkpoint_dir.glob("*.json")):
            validate_checkpoint(json.loads(checkpoint.read_text(encoding="utf-8")))
            artifacts[f"checkpoint:{checkpoint.stem}"] = _artifact(
                checkpoint, output, "stage_checkpoint"
            )
        manifest = {
            "schema_version": "2.0",
            "run_id": ledger.run_id,
            "complete": False,
            "validation_pending": ["quality.json", "manuscript_variables.json", "resolved_manuscript"],
            "config_sha256": config_hash,
            "execution_source_sha256": execution_digest,
            "certificates_pass": certificates_pass,
            "certificate_gate": certificate_gate,
            "artifacts": artifacts,
        }
        validate_manifest(manifest)
        _write_json(output / "manifest.json", manifest)
        LOGGER.info("analysis.complete elapsed_seconds=%.3f", time.perf_counter() - started_clock)
        return results
    except BaseException as exc:
        ledger.fail(exc)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "manuscript" / "config.yaml",
        help="Experiment YAML configuration.",
    )
    parser.add_argument(
        "--profile",
        choices=("fast", "release", "campaign", "publication", "benchmark"),
        default=None,
        help="Execution profile; campaign uses the expanded YAML-declared sweep.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Deterministic worker count for synthetic simulation scoring.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints matching the current contract.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help=(
            "Checkpoint the null-calibration stage after this many deterministic "
            "replications and record stage-complete snapshots for other stages; "
            "zero disables checkpoint files."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=None, help="Output directory; generated artifacts may live outside the project checkout.")
    parser.add_argument("--seed", type=int, default=None, help="Override the configured master seed.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console log level; structured run accounting is always written to output/run_log.json.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    results = run(
        args.config.resolve(),
        profile=args.profile,
        workers=args.workers,
        output_root=args.output_root,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        seed_override=args.seed,
    )
    output_path = (args.output_root or ROOT / "output") / "results.json"
    status = _certificate_gate_exit_status(results)
    if status:
        print(f"Wrote diagnostic artifacts to {output_path}; certificate gate failed.")
    else:
        print(f"Wrote {output_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
