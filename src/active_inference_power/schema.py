"""Lightweight validation for generated research artifacts.

The project deliberately keeps schemas executable and dependency-light.  These
checks catch stale or partial artifacts without coupling domain code to a JSON
schema package.
"""

from __future__ import annotations

from collections.abc import Mapping
import ast
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
from typing import AbstractSet, Any

import numpy as np
import yaml

from .formalism import EVIDENCE_CLASSES, validate_formalism_registry
from .certificate_gate import validate_certificate_gate_result
from .computational_complexity import (
    validate_computational_complexity,
    validate_computational_complexity_against_results,
)
from .learned_policy import LearnedPolicyArtifact
from .online_fdr import EValueLONDResult, OnlineFDRResult
from .scenario import (
    GenerativeModelSpec,
    GenerativeProcessSpec,
    PowerScenario,
    SettingSpec,
)
from .uncertainty import wilson_interval


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLAIM_KINDS = frozenset(
    {
        "citation",
        "computation_policy",
        "configuration",
        "empirical_comparison",
        "empirical_operating_characteristic",
        "empirical_stress_test",
        "process_contract",
        "provenance_contract",
        "rendering_policy",
        "schema_contract",
        "scope_definition",
        "validity_boundary",
        "verdict",
    }
)
_CLAIM_SOURCE_TIERS = frozenset({"project_source", "generated_metric", "bibliography", "cited_theory"})
_CLAIM_FRESHNESS = frozenset({"active", "generated"})
_CLAIM_BEARING_KINDS = frozenset(
    {
        "empirical_comparison",
        "empirical_operating_characteristic",
        "empirical_stress_test",
        "schema_contract",
        "scope_definition",
        "validity_boundary",
        "verdict",
    }
)
_CURRENT_SCENARIO_CONTRACT_KEYS = frozenset(
    {
        "scenario_id",
        "description",
        "claim_role",
        "metadata",
        "model",
        "process",
        "setting",
        "scenario_hash",
    }
)


def _require_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _scenario_contract_digest(value: Mapping[str, Any]) -> str:
    """Hash a serialized scenario contract without its redundant digest."""
    payload = dict(value)
    payload.pop("scenario_hash", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _payload_digest(value: Mapping[str, Any]) -> str:
    """Hash an already canonicalized payload without dropping any fields."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_scenario_contract(value: Any, name: str) -> Mapping[str, Any]:
    """Validate and recompute one model/process/setting lineage contract."""
    contract = _require_mapping(value, name)
    _require_keys(contract, {"scenario_id", "model", "process", "setting", "scenario_hash"}, name)
    if not isinstance(contract["scenario_id"], str) or not contract["scenario_id"]:
        raise ValueError(f"{name}.scenario_id must be non-empty")
    _require_sha256(contract["scenario_hash"], f"{name}.scenario_hash")
    for key in ("model", "process", "setting"):
        _require_mapping(contract[key], f"{name}.{key}")
    if contract["scenario_hash"] != _scenario_contract_digest(contract):
        raise ValueError(f"{name}.scenario_hash does not match the serialized contract")
    return contract


def _validate_current_scenario_contract(value: Any, name: str) -> Mapping[str, Any]:
    """Require a canonical, reconstructable scenario contract for release use.

    The baseline reader accepts compact prior-schema artifacts.  A current
    release instead needs the complete object contract to be both self-hashed
    and semantically valid under the domain constructors.  Reconstructing the
    objects prevents a self-consistent digest from laundering malformed model,
    process, or setting values into a release artifact.
    """
    contract = _validate_scenario_contract(value, name)
    keys = set(contract)
    if keys != _CURRENT_SCENARIO_CONTRACT_KEYS:
        missing = sorted(_CURRENT_SCENARIO_CONTRACT_KEYS - keys)
        unexpected = sorted(keys - _CURRENT_SCENARIO_CONTRACT_KEYS)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(f"{name} must contain exactly the current scenario contract keys ({'; '.join(details)})")
    try:
        json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON values") from error
    try:
        model = GenerativeModelSpec(**dict(_require_mapping(contract["model"], f"{name}.model")))
        process = GenerativeProcessSpec(
            **dict(_require_mapping(contract["process"], f"{name}.process"))
        )
        setting = SettingSpec(**dict(_require_mapping(contract["setting"], f"{name}.setting")))
        metadata = _require_mapping(contract["metadata"], f"{name}.metadata")
        scenario = PowerScenario(
            scenario_id=contract["scenario_id"],
            model=model,
            process=process,
            setting=setting,
            description=contract["description"],
            claim_role=contract["claim_role"],
            metadata=dict(metadata),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} is not a valid reconstructed scenario contract") from error
    if scenario.configuration_hash != contract["scenario_hash"]:
        raise ValueError(
            f"{name}.scenario_hash does not match its reconstructed scenario contract"
        )
    return contract


def _current_scenario_contract_index(
    value: Any, name: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Index non-empty, self-hashing scenario contracts for current releases.

    Prior-schema artifacts may use a compact list of scenario identifiers.  A
    release-facing validation request is intentionally stricter: it needs the
    complete model/process/setting declaration so that the row-level evidence
    can be tied back to a recomputable scenario lineage.
    """
    contracts = _require_list(value, name)
    if not contracts:
        raise ValueError(f"{name} must contain at least one full scenario contract")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    identifiers: set[str] = set()
    for index, contract_value in enumerate(contracts):
        contract = _validate_current_scenario_contract(
            contract_value, f"{name}[{index}]"
        )
        scenario_id = str(contract["scenario_id"])
        pair = (scenario_id, str(contract["scenario_hash"]))
        if scenario_id in identifiers or pair in indexed:
            raise ValueError(f"{name} contains duplicate scenario lineage")
        identifiers.add(scenario_id)
        indexed[pair] = contract
    return indexed


def _current_row_identity(value: Mapping[str, Any], name: str) -> tuple[str, str, str]:
    """Return the immutable scenario/policy key for a current release row."""
    scenario_id = value.get("scenario_id")
    scenario_hash = value.get("scenario_hash")
    policy = value.get("policy")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError(f"{name}.scenario_id must be non-empty")
    _require_sha256(scenario_hash, f"{name}.scenario_hash")
    if not isinstance(policy, str) or not policy:
        raise ValueError(f"{name}.policy must be non-empty")
    return (scenario_id, str(scenario_hash), policy)


def _validate_current_frontier_projection(suite: Mapping[str, Any]) -> None:
    """Require one exact frontier projection for every operating row.

    The current suite serializes the frontier as the reader-facing projection
    of each normal operating-characteristic row.  Checking the projection
    prevents a structurally valid but semantically mismatched table/figure
    input from being promoted through a release gate.
    """
    operations = _require_list(
        suite["operating_characteristics"], "suite.operating_characteristics"
    )
    if "power_cost_frontier" not in suite:
        raise ValueError("current suite contract requires power_cost_frontier")
    frontier_rows = _require_list(
        suite["power_cost_frontier"], "suite.power_cost_frontier"
    )
    if not frontier_rows:
        raise ValueError("current suite contract requires non-empty power_cost_frontier")

    operation_rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for index, raw_operation in enumerate(operations):
        operation = _require_mapping(
            raw_operation, f"suite.operating_characteristics[{index}]"
        )
        identity = _current_row_identity(
            operation, f"suite.operating_characteristics[{index}]"
        )
        if identity in operation_rows:
            raise ValueError("suite.operating_characteristics contains duplicate scenario/policy rows")
        operation_rows[identity] = operation

    frontier_by_identity: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for index, raw_frontier in enumerate(frontier_rows):
        frontier = _require_mapping(raw_frontier, f"suite.power_cost_frontier[{index}]")
        identity = _current_row_identity(
            frontier, f"suite.power_cost_frontier[{index}]"
        )
        if identity in frontier_by_identity:
            raise ValueError("suite.power_cost_frontier contains duplicate scenario/policy rows")
        frontier_by_identity[identity] = frontier

    if set(operation_rows) != set(frontier_by_identity):
        raise ValueError(
            "current suite operating characteristics and power-cost frontier rows differ"
        )

    fields = (
        "n_reps",
        "power",
        "power_interval",
        "fdr",
        "fdr_interval",
        "mean_cost",
        "cost_interval",
        "mean_stopping_time",
        "power_per_cost",
        "validity",
    )
    for identity, operation in operation_rows.items():
        summary = _require_mapping(
            operation["summary"],
            "suite operating-characteristic summary",
        )
        frontier = frontier_by_identity[identity]
        for field in fields:
            if field not in summary:
                raise ValueError(
                    "current suite operating-characteristic summary is missing "
                    f"{field!r}"
                )
            if frontier.get(field) != summary[field]:
                raise ValueError(
                    "current suite power-cost frontier does not match its "
                    f"operating-characteristic {field!r}"
                )
        if operation.get("n_reps") != summary["n_reps"]:
            raise ValueError(
                "current suite operating-characteristic n_reps does not match its summary"
            )
        if operation.get("validity") != summary["validity"]:
            raise ValueError(
                "current suite operating-characteristic validity does not match its summary"
            )


_CURRENT_ONLINE_FDR_METHODS = ("lord_plus_plus", "saffron", "e_lond")
_CURRENT_E_LOND_NULL_TRANSFORM = "c * 1{U <= 1/c}; E_null[e] = 1"
_CURRENT_SPLIT_FORMAL_VALIDITY = (
    "conditional_on_declared_split_confirmation_assumptions"
)
_CURRENT_SPLIT_ASSUMPTIONS = frozenset(
    {
        "fixed_truth",
        "independent_selection_and_confirmation_streams",
        "selection_uses_visible_filtration_only",
        "conditional_superuniform_confirmation_pvalue_under_null",
    }
)


def _validate_current_online_fdr_artifact(value: Any, name: str) -> None:
    """Recompute the current online-FDR representative traces.

    A current result stores the finite null-stream values needed to rebuild the
    published LORD++, SAFFRON, and e-LOND recursions.  This rejects a
    self-consistent-looking figure input that has been relabelled, has a stale
    accounting path, or treats e-LOND's cumulative allocation as alpha wealth.
    Summary rates stay finite-simulation diagnostics; trace reconstruction
    verifies the displayed path rather than promoting those rates to a proof.
    """
    artifact = _require_mapping(value, name)
    _require_keys(
        artifact,
        {"null", "alpha", "horizon", "n_reps", "methods", "uncertainty"},
        name,
    )
    if artifact["null"] is not True:
        raise ValueError(f"{name}.null must be true for the current null-stream diagnostic")
    alpha = _require_probability(artifact["alpha"], f"{name}.alpha")
    horizon = artifact["horizon"]
    n_reps = artifact["n_reps"]
    _require_positive_integer(horizon, f"{name}.horizon")
    _require_positive_integer(n_reps, f"{name}.n_reps")
    if not isinstance(artifact["uncertainty"], str) or not artifact["uncertainty"].strip():
        raise ValueError(f"{name}.uncertainty must be non-empty")
    methods = _require_mapping(artifact["methods"], f"{name}.methods")
    if set(methods) != set(_CURRENT_ONLINE_FDR_METHODS):
        raise ValueError(f"{name}.methods must contain exactly {_CURRENT_ONLINE_FDR_METHODS}")

    for method in _CURRENT_ONLINE_FDR_METHODS:
        summary = _require_mapping(methods[method], f"{name}.methods.{method}")
        _require_keys(
            summary,
            {
                "mean_rejections",
                "any_rejection_rate",
                "any_rejection_interval_wilson",
                "trace",
            },
            f"{name}.methods.{method}",
        )
        mean_rejections = summary["mean_rejections"]
        if (
            isinstance(mean_rejections, bool)
            or not isinstance(mean_rejections, (int, float))
            or not math.isfinite(float(mean_rejections))
            or float(mean_rejections) < 0.0
            or float(mean_rejections) > int(horizon)
        ):
            raise ValueError(
                f"{name}.methods.{method}.mean_rejections must lie in [0, horizon]"
            )
        rate = _require_probability(
            summary["any_rejection_rate"],
            f"{name}.methods.{method}.any_rejection_rate",
        )
        _validate_interval(
            summary["any_rejection_interval_wilson"],
            f"{name}.methods.{method}.any_rejection_interval_wilson",
            lower=0.0,
        )
        if any(float(item) > 1.0 for item in summary["any_rejection_interval_wilson"]):
            raise ValueError(
                f"{name}.methods.{method}.any_rejection_interval_wilson must lie in [0, 1]"
            )
        implied_count = int(round(rate * int(n_reps)))
        if not math.isclose(
            rate,
            implied_count / int(n_reps),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{name}.methods.{method}.any_rejection_rate must be a recorded replication proportion"
            )
        expected_interval = wilson_interval(implied_count, int(n_reps))
        observed_interval = summary["any_rejection_interval_wilson"]
        if not np.allclose(
            np.asarray(observed_interval, dtype=float),
            np.asarray(expected_interval, dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{name}.methods.{method}.any_rejection_interval_wilson does not match its recorded rate"
            )

        trace = _require_mapping(summary["trace"], f"{name}.methods.{method}.trace")
        if method in {"lord_plus_plus", "saffron"}:
            _require_keys(
                trace,
                {
                    "alpha",
                    "pvalues",
                    "initial_wealth",
                    "lambda",
                    "levels",
                    "candidates",
                    "rejections",
                    "accounting_path",
                    "gamma",
                    "accounting_kind",
                },
                f"{name}.methods.{method}.trace",
            )
            if trace["accounting_kind"] != "procedural_alpha_wealth":
                raise ValueError(
                    f"{name}.methods.{method}.trace.accounting_kind must be procedural_alpha_wealth"
                )
            if not math.isclose(float(trace["alpha"]), alpha, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{name}.methods.{method}.trace.alpha does not match {name}.alpha")
            reconstructed = OnlineFDRResult(
                method=method,
                pvalues=np.asarray(trace["pvalues"], dtype=float),
                levels=np.asarray(trace["levels"], dtype=float),
                candidates=np.asarray(trace["candidates"], dtype=bool),
                rejections=np.asarray(trace["rejections"], dtype=bool),
                wealth=np.asarray(trace["accounting_path"], dtype=float),
                alpha=float(trace["alpha"]),
                gamma=np.asarray(trace["gamma"], dtype=float),
                lambda_=trace["lambda"],
                initial_wealth=float(trace["initial_wealth"]),
            )
            if reconstructed.levels.size != int(horizon):
                raise ValueError(f"{name}.methods.{method}.trace length does not match horizon")
            continue

        _require_keys(
            trace,
            {
                "alpha",
                "evalues",
                "levels",
                "thresholds",
                "rejections",
                "accounting_path",
                "gamma",
                "accounting_kind",
                "num_rejections",
                "e_value_null_transform",
                "e_value_support",
                "trace_role",
            },
            f"{name}.methods.e_lond.trace",
        )
        _require_keys(
            summary,
            {"null_e_value_transform", "null_e_value_support"},
            f"{name}.methods.e_lond",
        )
        if trace["accounting_kind"] != "cumulative_nominal_level_allocation":
            raise ValueError(
                f"{name}.methods.e_lond.trace.accounting_kind must be cumulative_nominal_level_allocation"
            )
        if not math.isclose(float(trace["alpha"]), alpha, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{name}.methods.e_lond.trace.alpha does not match {name}.alpha")
        if (
            trace["e_value_null_transform"] != _CURRENT_E_LOND_NULL_TRANSFORM
            or summary["null_e_value_transform"] != _CURRENT_E_LOND_NULL_TRANSFORM
        ):
            raise ValueError(
                f"{name}.methods.e_lond must state the declared bounded null e-value transform"
            )
        support = trace["e_value_support"]
        _require_positive_integer(support, f"{name}.methods.e_lond.trace.e_value_support")
        if summary["null_e_value_support"] != support:
            raise ValueError(
                f"{name}.methods.e_lond null e-value support disagrees with its trace"
            )
        evalues = np.asarray(trace["evalues"], dtype=float)
        if not np.all(
            np.logical_or(
                np.isclose(evalues, 0.0, rtol=0.0, atol=1e-12),
                np.isclose(evalues, float(support), rtol=0.0, atol=1e-12),
            )
        ):
            raise ValueError(
                f"{name}.methods.e_lond.trace.evalues do not match the declared bounded transform support"
            )
        reconstructed_e_lond = EValueLONDResult(
            evalues=evalues,
            levels=np.asarray(trace["levels"], dtype=float),
            thresholds=np.asarray(trace["thresholds"], dtype=float),
            rejections=np.asarray(trace["rejections"], dtype=bool),
            cumulative_allocations=np.asarray(trace["accounting_path"], dtype=float),
            alpha=float(trace["alpha"]),
            gamma=np.asarray(trace["gamma"], dtype=float),
        )
        if reconstructed_e_lond.levels.size != int(horizon):
            raise ValueError(f"{name}.methods.e_lond.trace length does not match horizon")
        if trace["num_rejections"] != reconstructed_e_lond.num_rejections:
            raise ValueError(
                f"{name}.methods.e_lond.trace.num_rejections does not match rejections"
            )
        expected_role = (
            "first_seeded_rejection_bearing_valid_null_trace"
            if reconstructed_e_lond.num_rejections
            else "first_seeded_valid_null_trace_no_rejection"
        )
        if trace["trace_role"] != expected_role:
            raise ValueError(f"{name}.methods.e_lond.trace_role does not match its trace")


def _validate_current_suite_contract(suite: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Validate the complete current suite contract and return operating lineage.

    The normal scenario grid supplies the policy-by-scenario operating rows.
    Split confirmation instead uses one separately declared static channel, so
    its full contract is bound explicitly without pretending that it is an
    additional power/frontier cell.
    """
    if suite.get("contract_version") != "1.0":
        raise ValueError("current suite contract_version must be '1.0'")
    if "scenario_contracts" not in suite:
        raise ValueError("current suite contract requires scenario_contracts")
    contracts = _current_scenario_contract_index(
        suite["scenario_contracts"], "suite.scenario_contracts"
    )
    declared = suite.get("scenarios")
    if not isinstance(declared, list) or any(
        not isinstance(scenario_id, str) or not scenario_id for scenario_id in declared
    ):
        raise ValueError("current suite scenarios must be non-empty scenario identifiers")
    if {str(scenario_id) for scenario_id in declared} != {
        scenario_id for scenario_id, _ in contracts
    }:
        raise ValueError("current suite scenarios do not match scenario-contract lineage")
    _validate_current_frontier_projection(suite)
    split_scenario = _validate_current_scenario_contract(
        suite.get("split_confirmation_scenario"),
        "current suite split-confirmation scenario",
    )
    if str(split_scenario["scenario_id"]) in {
        scenario_id for scenario_id, _ in contracts
    }:
        raise ValueError(
            "current suite dedicated split-confirmation scenario must be outside "
            "the operating grid"
        )
    split_setting = _require_mapping(
        split_scenario.get("setting"),
        "current suite split-confirmation scenario.setting",
    )
    if split_setting.get("formal_validity") != _CURRENT_SPLIT_FORMAL_VALIDITY:
        raise ValueError(
            "current suite split-confirmation scenario must declare its conditional "
            "split-confirmation validity contract"
        )
    declared_split_assumptions = split_setting.get("evidence_assumptions")
    if (
        not isinstance(declared_split_assumptions, list)
        or len(declared_split_assumptions) != len(_CURRENT_SPLIT_ASSUMPTIONS)
        or frozenset(declared_split_assumptions) != _CURRENT_SPLIT_ASSUMPTIONS
    ):
        raise ValueError(
            "current suite split-confirmation scenario assumptions do not match "
            "the declared split-confirmation contract"
        )
    split_metadata = _require_mapping(
        split_scenario.get("metadata"),
        "current suite split-confirmation scenario.metadata",
    )
    if split_metadata.get("split_confirmation_channel") is not True:
        raise ValueError(
            "current suite split-confirmation scenario must identify its dedicated channel"
        )
    split_alpha = _require_probability(
        split_setting.get("alpha"),
        "current suite split-confirmation scenario.setting.alpha",
    )
    split_scenario_lineage = (
        str(split_scenario["scenario_id"]),
        str(split_scenario["scenario_hash"]),
    )
    split_confirmation = _require_mapping(
        suite.get("split_stream_confirmation"), "current suite split-stream confirmation"
    )
    _validate_current_split_confirmation_estimand(
        split_confirmation, "suite.split_stream_confirmation"
    )
    if split_confirmation.get("validity") != "formal_conditional_split_confirmation":
        raise ValueError(
            "current suite split-stream confirmation must carry its formal conditional validity label"
        )
    summary_assumptions = split_confirmation.get("assumptions")
    if (
        not isinstance(summary_assumptions, list)
        or len(summary_assumptions) != len(_CURRENT_SPLIT_ASSUMPTIONS)
        or frozenset(summary_assumptions) != _CURRENT_SPLIT_ASSUMPTIONS
    ):
        raise ValueError(
            "current suite split-stream confirmation assumptions do not match its scenario"
        )
    split_lineage = (
        str(split_confirmation.get("scenario_id", "")),
        str(split_confirmation.get("scenario_hash", "")),
    )
    if split_lineage != split_scenario_lineage:
        raise ValueError(
            "current suite split-stream confirmation does not match its declared "
            "split-confirmation scenario contract"
        )
    representative = _require_mapping(
        split_confirmation.get("representative"),
        "current suite split-stream confirmation representative",
    )
    representative_lineage = (
        str(representative.get("scenario_id", "")),
        str(representative.get("scenario_hash", "")),
    )
    if representative_lineage != split_lineage:
        raise ValueError(
            "current suite split-stream representative does not match its "
            "summary scenario lineage"
        )
    if representative.get("validity") != "formal_conditional_split_confirmation":
        raise ValueError(
            "current suite split-stream representative must carry its formal conditional validity label"
        )
    representative_assumptions = representative.get("validity_assumptions")
    if (
        not isinstance(representative_assumptions, list)
        or len(representative_assumptions) != len(_CURRENT_SPLIT_ASSUMPTIONS)
        or frozenset(representative_assumptions) != _CURRENT_SPLIT_ASSUMPTIONS
    ):
        raise ValueError(
            "current suite split-stream representative assumptions do not match its scenario"
        )
    confirmation_p_value = _require_probability(
        representative.get("confirmation_p_value"),
        "current suite split-stream representative.confirmation_p_value",
    )
    if not isinstance(representative.get("rejected"), bool):
        raise ValueError("current suite split-stream representative.rejected must be boolean")
    if representative["rejected"] is not (confirmation_p_value <= split_alpha):
        raise ValueError(
            "current suite split-stream representative rejection does not use the declared scenario alpha"
        )
    selection = _require_mapping(
        representative.get("selection"),
        "current suite split-stream representative selection",
    )
    selection_lineage = (
        str(selection.get("scenario_id", "")),
        str(selection.get("scenario_hash", "")),
    )
    if selection_lineage != split_lineage:
        raise ValueError(
            "current suite split-stream representative selection does not match "
            "its summary scenario lineage"
        )
    if selection.get("visible_only") is not True:
        raise ValueError(
            "current suite split-stream representative selection must use visible history only"
        )
    return contracts


def _validate_current_release_contract(results: Mapping[str, Any]) -> None:
    """Bind a release suite and action-loop projection to the same lineage."""
    suite = _require_mapping(results.get("suite"), "results.suite")
    action_loop = _require_mapping(results.get("action_loop"), "results.action_loop")
    suite_contracts = _validate_current_suite_contract(suite)
    _validate_current_online_fdr_artifact(
        results.get("online_fdr"), "results.online_fdr"
    )
    if action_loop.get("contract_version") != "1.0":
        raise ValueError("current action-loop contract_version must be '1.0'")
    action_contracts = _current_scenario_contract_index(
        action_loop.get("scenarios"), "results.action_loop.scenarios"
    )
    if suite_contracts != action_contracts:
        raise ValueError("current suite and action-loop scenario lineage differ")
    for field in ("policies", "operating_characteristics", "power_cost_frontier"):
        if action_loop.get(field) != suite.get(field):
            raise ValueError(f"current suite and action-loop {field} differ")
    validate_computational_complexity_against_results(
        _require_mapping(
            results.get("computational_complexity"),
            "results.computational_complexity",
        ),
        results,
    )


def _validate_interval(value: Any, name: str, *, lower: float | None = None) -> None:
    """Validate a finite two-sided interval used by operating characteristics."""
    interval = _require_list(value, name)
    if len(interval) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    numbers: list[float] = []
    for index, item in enumerate(interval):
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"{name}[{index}] must be finite numeric")
        numbers.append(float(item))
    if numbers[0] > numbers[1] or (lower is not None and numbers[0] < lower):
        raise ValueError(f"{name} must be ordered and respect its lower bound")


def validate_claim_ledger(
    value: Mapping[str, Any],
    *,
    project_root: Path | None = None,
    require_artifacts: bool = False,
    artifact_root: Path | None = None,
) -> None:
    """Validate the machine-readable claim ledger and its evidence boundary.

    The ledger is intentionally metadata rather than a second results store.
    This validator nevertheless fails closed on duplicate identifiers, unknown
    claim classes, unsafe paths, missing source files, and claim-bearing rows
    that omit an evidence class or explicit non-claim.
    """
    ledger = _require_mapping(value, "claim_ledger")
    claims = _require_list(ledger.get("claims"), "claim_ledger.claims")
    if not claims:
        raise ValueError("claim_ledger.claims must not be empty")
    identifiers: set[str] = set()
    for index, claim_value in enumerate(claims):
        claim = _require_mapping(claim_value, f"claim_ledger.claims[{index}]")
        _require_keys(
            claim,
            {"claim_id", "kind", "source", "source_tier", "freshness", "artifact_path"},
            f"claim_ledger.claims[{index}]",
        )
        claim_id = claim["claim_id"]
        if not isinstance(claim_id, str) or not claim_id or claim_id in identifiers:
            raise ValueError(f"claim_ledger claim_id is missing or duplicated: {claim_id!r}")
        identifiers.add(claim_id)
        kind = claim["kind"]
        if kind not in _CLAIM_KINDS:
            raise ValueError(f"claim_ledger claim {claim_id!r} has unknown kind")
        for key in ("source", "source_tier", "freshness", "artifact_path"):
            if not isinstance(claim[key], str) or not claim[key]:
                raise ValueError(f"claim_ledger claim {claim_id!r}.{key} must be non-empty")
        if claim["source_tier"] not in _CLAIM_SOURCE_TIERS:
            raise ValueError(f"claim_ledger claim {claim_id!r}.source_tier is invalid")
        if claim["freshness"] not in _CLAIM_FRESHNESS:
            raise ValueError(f"claim_ledger claim {claim_id!r}.freshness is invalid")
        artifact_path = claim["artifact_path"]
        if artifact_path.startswith("/") or ".." in artifact_path.split("/"):
            raise ValueError(f"claim_ledger claim {claim_id!r}.artifact_path is unsafe")
        if kind == "citation" and (not isinstance(claim.get("value"), str) or not claim["value"]):
            raise ValueError(f"citation claim {claim_id!r} must name a bibliography key")
        evidence_class = claim.get("evidence_class")
        if evidence_class is not None:
            if not isinstance(evidence_class, str) or not evidence_class:
                raise ValueError(
                    f"claim_ledger claim {claim_id!r}.evidence_class must be non-empty when supplied"
                )
            if evidence_class not in EVIDENCE_CLASSES:
                raise ValueError(
                    f"claim_ledger claim {claim_id!r} has unsupported evidence_class: "
                    f"{evidence_class!r}"
                )
        if kind in _CLAIM_BEARING_KINDS:
            non_claim = claim.get("non_claim")
            if not isinstance(evidence_class, str) or not evidence_class:
                raise ValueError(f"claim-bearing row {claim_id!r} needs evidence_class")
            if not isinstance(non_claim, str) or not non_claim:
                raise ValueError(f"claim-bearing row {claim_id!r} needs non_claim")
        if project_root is not None:
            source_path = Path(claim["source"].split(":", 1)[0])
            if source_path.is_absolute() or ".." in source_path.parts:
                raise ValueError(f"claim_ledger claim {claim_id!r}.source is unsafe")
            if not (project_root / source_path).is_file():
                raise ValueError(f"claim_ledger source file is missing for {claim_id!r}: {source_path}")
            _validate_source_locator(
                project_root / source_path,
                claim["source"].split(":", 1)[1] if ":" in claim["source"] else None,
                claim_id,
            )
            if require_artifacts:
                if artifact_root is None or not artifact_path.startswith("output/"):
                    candidate = project_root / artifact_path
                else:
                    candidate = Path(artifact_root) / artifact_path.removeprefix("output/")
                if not candidate.is_file():
                    raise ValueError(
                        f"claim_ledger artifact is missing for {claim_id!r}: {artifact_path}"
                    )


def _validate_source_locator(path: Path, locator: str | None, claim_id: str) -> None:
    """Fail closed when a claim's explicit source locator is stale.

    A ledger source without a locator identifies a whole document.  When a
    locator is present, its meaning is format-specific: Python symbols,
    dotted YAML keys, or bibliography identifiers.  This keeps provenance
    checks useful without requiring line numbers to remain stable after edits.
    """
    if locator is None:
        return
    locator = locator.strip()
    if not locator:
        raise ValueError(f"claim_ledger claim {claim_id!r}.source locator is empty")
    if path.suffix == ".py":
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            raise ValueError(f"claim_ledger Python source is not parseable for {claim_id!r}") from exc
        symbols: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                symbols.update(target.id for target in targets if isinstance(target, ast.Name))
        missing = [name for name in (part.strip() for part in locator.split(",")) if name not in symbols]
        if missing:
            raise ValueError(
                f"claim_ledger source locator is missing for {claim_id!r}: {missing}"
            )
        return
    if path.suffix in {".yaml", ".yml"}:
        try:
            payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"claim_ledger YAML source is not parseable for {claim_id!r}") from exc
        current = payload
        for key in locator.split("."):
            if not isinstance(current, Mapping) or key not in current:
                raise ValueError(
                    f"claim_ledger YAML locator is missing for {claim_id!r}: {locator}"
                )
            current = current[key]
        return
    if path.suffix == ".bib":
        bibliography = path.read_text(encoding="utf-8")
        if re.search(r"@(?:article|inproceedings|incollection|book|misc)\{" + re.escape(locator) + r"\s*,", bibliography) is None:
            raise ValueError(
                f"claim_ledger bibliography locator is missing for {claim_id!r}: {locator}"
            )


def _require_keys(value: Mapping[str, Any], keys: AbstractSet[str], name: str) -> None:
    missing = sorted(keys - set(value))
    if missing:
        raise ValueError(f"{name} is missing keys: {missing}")


def validate_results(
    value: Mapping[str, Any],
    *,
    version: str = "2.0",
    require_suite: bool = False,
    require_current_contract: bool = False,
) -> None:
    """Validate a result artifact, optionally as a current release contract.

    The default accepts schema-valid preserved artifacts.  Set
    ``require_current_contract`` only for release-facing consumers to require
    complete scenario lineage and matching suite/action-loop projections.
    """
    results = _require_mapping(value, "results")
    if results.get("schema_version") != version:
        raise ValueError(f"results schema must be {version!r}")
    _require_keys(
        results,
        {"schema_version", "run_id", "meta", "experiment", "certificates", "method_comparison", "paired_comparisons", "power_curve", "dependence_sweep", "active_inference", "sequential", "online_fdr", "agent_sampling", "benchmarks", "null_pvalue_calibration", "sample_size_accounting", "formalism"},
        "results",
    )
    meta = _require_mapping(results["meta"], "results.meta")
    _require_keys(meta, {"config_sha256", "seed", "n_reps"}, "results.meta")
    if not isinstance(results["run_id"], str) or not results["run_id"]:
        raise ValueError("results.run_id must be a non-empty string")
    _require_sha256(meta["config_sha256"], "results.meta.config_sha256")
    if "execution_source_sha256" in meta:
        _require_sha256(
            meta["execution_source_sha256"], "results.meta.execution_source_sha256"
        )
    _require_nonnegative_integer(meta["seed"], "results.meta.seed")
    _require_positive_integer(meta["n_reps"], "results.meta.n_reps")
    certificates = _require_mapping(results["certificates"], "results.certificates")
    for name, value in certificates.items():
        certificate = _require_mapping(value, f"results.certificates.{name}")
        _require_keys(
            certificate,
            {"name", "estimate", "bound", "stderr", "passes"},
            f"results.certificates.{name}",
        )
        if not isinstance(certificate["passes"], bool):
            raise ValueError(f"results.certificates.{name}.passes must be boolean")
        if name == "storey_power_gain":
            _require_keys(
                certificate,
                {
                    "bh_fdr_stderr",
                    "storey_fdr_stderr",
                    "bh_power_stderr",
                    "power_gain_stderr",
                    "bh_fdr_upper_band",
                    "storey_fdr_upper_band",
                    "storey_fdr_excess",
                    "storey_fdr_z",
                    "bh_fdr_passes",
                    "storey_fdr_passes",
                    "power_gain_passes",
                },
                "results.certificates.storey_power_gain",
            )
            for key in (
                "bh_fdr_passes",
                "storey_fdr_passes",
                "power_gain_passes",
            ):
                if not isinstance(certificate[key], bool):
                    raise ValueError(
                        f"results.certificates.storey_power_gain.{key} must be boolean"
                    )
    if "certificate_gate" in meta:
        validate_certificate_gate_result(
            _require_mapping(meta["certificate_gate"], "results.meta.certificate_gate"),
            certificates,
        )
    for name in (
        "experiment", "certificates", "method_comparison", "paired_comparisons",
        "active_inference", "sequential", "online_fdr", "agent_sampling",
        "benchmarks", "null_pvalue_calibration", "sample_size_accounting",
        "formalism",
    ):
        _require_mapping(results[name], f"results.{name}")
    _require_list(results["power_curve"], "results.power_curve")
    _require_list(results["dependence_sweep"], "results.dependence_sweep")
    validate_formalism_registry(_require_mapping(results["formalism"], "results.formalism"))
    if "computational_complexity" in results:
        validate_computational_complexity(
            _require_mapping(
                results["computational_complexity"],
                "results.computational_complexity",
            )
        )
    elif require_current_contract:
        raise ValueError(
            "results.computational_complexity is required for a current release contract"
        )
    if (require_suite or require_current_contract) and "suite" not in results:
        raise ValueError("results.suite is required for a release suite artifact")
    if (require_suite or require_current_contract) and "action_loop" not in results:
        raise ValueError("results.action_loop is required for a release suite artifact")
    if require_suite or require_current_contract:
        _require_sha256(meta.get("claim_ledger_sha256"), "results.meta.claim_ledger_sha256")
    if "suite" in results:
        validate_suite_result(
            results["suite"], require_current_contract=require_current_contract
        )
    if "action_loop" in results:
        _require_mapping(results["action_loop"], "results.action_loop")
        if "contract_version" in results["action_loop"] and results["action_loop"]["contract_version"] != "1.0":
            raise ValueError("results.action_loop.contract_version must be '1.0'")
        if "scenarios" not in results["action_loop"] or not isinstance(results["action_loop"]["scenarios"], list):
            raise ValueError("results.action_loop.scenarios must be a list")
        action_scenario_ids: set[str] = set()
        for index, scenario in enumerate(results["action_loop"]["scenarios"]):
            item = _require_mapping(scenario, f"results.action_loop.scenarios[{index}]")
            _require_keys(item, {"scenario_id", "model", "process", "setting"}, f"results.action_loop.scenarios[{index}]")
            if not isinstance(item["scenario_id"], str) or not item["scenario_id"]:
                raise ValueError(f"results.action_loop.scenarios[{index}].scenario_id must be non-empty")
            if item["scenario_id"] in action_scenario_ids:
                raise ValueError("results.action_loop.scenarios contains duplicate scenario identifiers")
            action_scenario_ids.add(str(item["scenario_id"]))
            if "scenario_hash" in item:
                _require_sha256(item["scenario_hash"], f"results.action_loop.scenarios[{index}].scenario_hash")
        if "operating_characteristics" in results["action_loop"]:
            if not isinstance(results["action_loop"]["operating_characteristics"], list):
                raise ValueError("results.action_loop.operating_characteristics must be a list")
            for item in results["action_loop"]["operating_characteristics"]:
                validate_action_loop_result(item)
        for key in ("paired_policy_comparisons", "power_cost_frontier"):
            if key in results["action_loop"] and not isinstance(results["action_loop"][key], list):
                raise ValueError(f"results.action_loop.{key} must be a list when present")
    if require_current_contract:
        _validate_current_release_contract(results)


def validate_manifest(value: Mapping[str, Any], *, version: str = "2.0") -> None:
    """Validate a complete artifact manifest."""
    manifest = _require_mapping(value, "manifest")
    if manifest.get("schema_version") != version:
        raise ValueError(f"manifest schema must be {version!r}")
    _require_keys(manifest, {"schema_version", "run_id", "complete", "config_sha256", "artifacts"}, "manifest")
    if not isinstance(manifest["complete"], bool):
        raise ValueError("manifest.complete must be boolean")
    _require_sha256(manifest["config_sha256"], "manifest.config_sha256")
    if "execution_source_sha256" in manifest:
        _require_sha256(
            manifest["execution_source_sha256"],
            "manifest.execution_source_sha256",
        )
    if not isinstance(manifest["run_id"], str) or not manifest["run_id"]:
        raise ValueError("manifest.run_id must be a non-empty string")
    artifacts = _require_mapping(manifest["artifacts"], "manifest.artifacts")
    if not artifacts:
        raise ValueError("manifest.artifacts must not be empty")
    for name, record in artifacts.items():
        item = _require_mapping(record, f"manifest.artifacts.{name}")
        _require_keys(item, {"path", "sha256", "bytes", "role"}, f"manifest.artifacts.{name}")
        if not isinstance(name, str) or not name:
            raise ValueError("manifest artifact names must be non-empty strings")
        if not isinstance(item["path"], str) or not item["path"] or item["path"].startswith("/") or ".." in item["path"].split("/"):
            raise ValueError(f"manifest artifact {name!r} has an unsafe relative path")
        if item.get("root", "project") not in {"project", "output"}:
            raise ValueError(f"manifest artifact {name!r}.root must be 'project' or 'output'")
        _require_sha256(item["sha256"], f"manifest artifact {name!r}.sha256")
        if isinstance(item["bytes"], bool) or not isinstance(item["bytes"], int) or item["bytes"] < 1:
            raise ValueError(f"manifest artifact {name!r}.bytes must be a positive integer")
        if not isinstance(item["role"], str) or not item["role"]:
            raise ValueError(f"manifest artifact {name!r}.role must be non-empty")


def validate_figure_registry(
    value: Mapping[str, Any], *, expected_filenames: set[str] | None = None
) -> None:
    """Validate the reader-facing figure registry and its lineage metadata.

    Reader-semantics fields were added in ``2.1``, the explicit result-bound
    visible comparison in ``2.2``, the role-aware evaluation perspective in
    ``2.3``, and explicit panel-validity records in ``2.4``. Preserved
    registries remain valid under their recorded schema, while a newly
    generated ``2.4`` record must carry every reader-first component and the
    panel-level contract (the whole-figure boundary remains
    ``metadata.validity_boundary``). This lets preserved receipts remain
    auditable without letting a newly generated release omit its reader-facing
    contract.
    """
    registry = _require_mapping(value, "figure_registry")
    schema_version = registry.get("schema_version")
    if schema_version not in {"2.0", "2.1", "2.2", "2.3", "2.4"}:
        raise ValueError(
            "figure_registry schema must be '2.0', '2.1', '2.2', '2.3', or '2.4'"
        )
    records = _require_list(registry.get("figures"), "figure_registry.figures")
    if not records:
        raise ValueError("figure_registry.figures must not be empty")
    filenames: set[str] = set()
    labels: set[str] = set()
    for index, value in enumerate(records):
        record = _require_mapping(value, f"figure_registry.figures[{index}]")
        _require_keys(
            record,
            {"filename", "label", "caption", "alt", "generated_by", "metadata"},
            f"figure_registry.figures[{index}]",
        )
        for key in ("filename", "label", "caption", "alt", "generated_by"):
            if not isinstance(record[key], str) or not record[key]:
                raise ValueError(f"figure_registry.figures[{index}].{key} must be non-empty")
        all_reader_semantics = (
            "reader_question",
            "takeaway",
            "visual_encoding",
            "visible_comparison",
            "evaluation_perspective",
        )
        required_reader_semantics: tuple[str, ...] = ()
        if schema_version in {"2.1", "2.2"}:
            required_reader_semantics = all_reader_semantics[:3]
        if schema_version == "2.2":
            required_reader_semantics = all_reader_semantics[:4]
        if schema_version in {"2.3", "2.4"}:
            required_reader_semantics = all_reader_semantics
        for key in all_reader_semantics:
            if key in required_reader_semantics and key not in record:
                raise ValueError(
                    f"figure_registry.figures[{index}].{key} is required by schema {schema_version}"
                )
            if key in record and (
                not isinstance(record[key], str) or not record[key].strip()
            ):
                raise ValueError(
                    f"figure_registry.figures[{index}].{key} must be a non-empty string"
                )
        if schema_version == "2.4":
            if "panel_validity" not in record:
                raise ValueError(
                    f"figure_registry.figures[{index}].panel_validity is required by schema 2.4"
                )
            panels = _require_list(
                record["panel_validity"],
                f"figure_registry.figures[{index}].panel_validity",
            )
            panel_names: set[str] = set()
            for panel_index, panel_value in enumerate(panels):
                panel = _require_mapping(
                    panel_value,
                    f"figure_registry.figures[{index}].panel_validity[{panel_index}]",
                )
                _require_keys(
                    panel,
                    {"panel", "validity"},
                    f"figure_registry.figures[{index}].panel_validity[{panel_index}]",
                )
                for key in ("panel", "validity"):
                    if not isinstance(panel[key], str) or not panel[key].strip():
                        raise ValueError(
                            f"figure_registry.figures[{index}].panel_validity[{panel_index}].{key} must be a non-empty string"
                        )
                if panel["panel"] in panel_names:
                    raise ValueError(
                        f"figure_registry.figures[{index}].panel_validity has duplicate panel labels"
                    )
                panel_names.add(panel["panel"])
        if record["filename"] in filenames or record["label"] in labels:
            raise ValueError("figure registry filenames and labels must be unique")
        filenames.add(record["filename"])
        labels.add(record["label"])
        metadata = _require_mapping(record["metadata"], f"figure_registry.figures[{index}].metadata")
        _require_keys(
            metadata,
            {
                "estimand",
                "uncertainty",
                "evidence_class",
                "validity_boundary",
                "redundant_encoding",
                "sample_size",
                "scenario_hashes",
                "result_lineage_hash",
            },
            f"figure_registry.figures[{index}].metadata",
        )
        for key in (
            "estimand",
            "uncertainty",
            "evidence_class",
            "validity_boundary",
            "redundant_encoding",
            "sample_size",
            "scenario_hashes",
        ):
            if not isinstance(metadata[key], str) or not metadata[key]:
                raise ValueError(f"figure_registry.figures[{index}].metadata.{key} must be non-empty")
        _require_sha256(
            metadata["result_lineage_hash"],
            f"figure_registry.figures[{index}].metadata.result_lineage_hash",
        )
    if expected_filenames is not None and filenames != set(expected_filenames):
        raise ValueError(
            f"figure registry filenames differ from expected set: {sorted(filenames ^ set(expected_filenames))}"
        )


def validate_quality_report(value: Mapping[str, Any], *, version: str = "2.0") -> None:
    """Validate the measured test/coverage release-quality evidence."""
    report = _require_mapping(value, "quality_report")
    if report.get("schema_version") != version:
        raise ValueError(f"quality_report schema must be {version!r}")
    _require_keys(
        report,
        {
            "schema_version",
            "test_count",
            "line_coverage_pct",
            "branch_coverage_pct",
            "coverage_gate_pct",
            "branch_coverage_gate_pct",
            "quality_pass",
            "quality_input_sha256",
            "pytest_exit_code",
            "command",
        },
        "quality_report",
    )
    _require_nonnegative_integer(report["test_count"], "quality_report.test_count")
    _require_sha256(report["quality_input_sha256"], "quality_report.quality_input_sha256")
    for key in (
        "line_coverage_pct",
        "branch_coverage_pct",
        "coverage_gate_pct",
        "branch_coverage_gate_pct",
    ):
        value_float = report[key]
        if (
            isinstance(value_float, bool)
            or not isinstance(value_float, (int, float))
            or not math.isfinite(float(value_float))
            or not 0.0 <= float(value_float) <= 100.0
        ):
            raise ValueError(f"quality_report.{key} must be a percentage in [0, 100]")
    if not isinstance(report["quality_pass"], bool):
        raise ValueError("quality_report.quality_pass must be boolean")
    if isinstance(report["pytest_exit_code"], bool) or not isinstance(
        report["pytest_exit_code"], int
    ):
        raise ValueError("quality_report.pytest_exit_code must be an integer")
    if float(report["coverage_gate_pct"]) != 95.0:
        raise ValueError("quality_report.coverage_gate_pct must be 95.0")
    if float(report["branch_coverage_gate_pct"]) != 95.0:
        raise ValueError("quality_report.branch_coverage_gate_pct must be 95.0")
    if report["quality_pass"] and report["test_count"] < 1:
        raise ValueError("passing quality_report requires test_count > 0")
    if report["quality_pass"] and report["pytest_exit_code"] != 0:
        raise ValueError("passing quality_report requires pytest_exit_code=0")
    if report["quality_pass"] and (
        float(report["line_coverage_pct"]) < float(report["coverage_gate_pct"])
        or float(report["branch_coverage_pct"])
        < float(report["branch_coverage_gate_pct"])
    ):
        raise ValueError("passing quality_report must satisfy both declared coverage gates")
    if not isinstance(report["command"], str) or not report["command"]:
        raise ValueError("quality_report.command must be non-empty")


def validate_run_log(value: Mapping[str, Any], *, version: str = "2.0") -> None:
    """Validate a structured run ledger snapshot."""
    log = _require_mapping(value, "run_log")
    if log.get("schema_version") != version:
        raise ValueError(f"run_log schema must be {version!r}")
    _require_keys(log, {"schema_version", "run_id", "status", "events", "config_sha256"}, "run_log")
    if log["status"] not in {"running", "passed", "failed", "interrupted"}:
        raise ValueError("run_log.status is invalid")
    if not isinstance(log["events"], list):
        raise ValueError("run_log.events must be a list")
    if not isinstance(log["run_id"], str) or not log["run_id"]:
        raise ValueError("run_log.run_id must be a non-empty string")
    _require_sha256(log["config_sha256"], "run_log.config_sha256")
    if "execution_source_sha256" in log:
        _require_sha256(
            log["execution_source_sha256"], "run_log.execution_source_sha256"
        )
    terminal = log["status"] in {"passed", "failed", "interrupted"}
    if terminal:
        finished_at = log.get("finished_at_utc")
        if not isinstance(finished_at, str) or not finished_at.endswith("+00:00"):
            raise ValueError("terminal run_log records require finished_at_utc in UTC ISO text")
    elif "finished_at_utc" in log:
        raise ValueError("running run_log records must not have finished_at_utc")
    if log["status"] in {"failed", "interrupted"}:
        error = _require_mapping(log.get("error"), "run_log.error")
        _require_keys(error, {"type", "message", "traceback"}, "run_log.error")
        for key in ("type", "message", "traceback"):
            if not isinstance(error[key], str) or not error[key]:
                raise ValueError(f"run_log.error.{key} must be non-empty")
    elif "error" in log:
        raise ValueError("passed/running run_log records must not contain an error")
    for index, event in enumerate(log["events"]):
        item = _require_mapping(event, f"run_log.events[{index}]")
        _require_keys(item, {"name", "status", "at_utc", "elapsed_seconds"}, f"run_log.events[{index}]")
        if not isinstance(item["name"], str) or not item["name"]:
            raise ValueError(f"run_log.events[{index}].name must be non-empty")
        if item["status"] not in {"started", "passed", "failed", "resumed", "checkpoint"}:
            raise ValueError(f"run_log.events[{index}].status is invalid")
        if not isinstance(item["at_utc"], str) or not item["at_utc"].endswith("+00:00"):
            raise ValueError(f"run_log.events[{index}].at_utc must be UTC ISO text")
        if isinstance(item["elapsed_seconds"], bool) or not isinstance(item["elapsed_seconds"], (int, float)) or item["elapsed_seconds"] < 0:
            raise ValueError(f"run_log.events[{index}].elapsed_seconds must be non-negative")
    progress = log.get("progress")
    if progress is not None:
        progress = _require_mapping(progress, "run_log.progress")
        _require_keys(progress, {"completed_stages", "total_stages", "completed_replications", "estimated_remaining_seconds"}, "run_log.progress")
        _require_nonnegative_integer(progress["completed_stages"], "run_log.progress.completed_stages")
        _require_positive_integer(progress["total_stages"], "run_log.progress.total_stages")
        _require_nonnegative_integer(progress["completed_replications"], "run_log.progress.completed_replications")
        if progress["completed_stages"] > progress["total_stages"]:
            raise ValueError("run_log.progress.completed_stages exceeds total_stages")
        remaining = progress["estimated_remaining_seconds"]
        if remaining is not None and (
            isinstance(remaining, bool)
            or not isinstance(remaining, (int, float))
            or remaining < 0
        ):
            raise ValueError("run_log.progress.estimated_remaining_seconds must be null or non-negative")
        if "last_completed_stage" in progress and (
            not isinstance(progress["last_completed_stage"], str)
            or not progress["last_completed_stage"]
        ):
            raise ValueError("run_log.progress.last_completed_stage must be non-empty when present")


def validate_benchmark_manifest(value: Mapping[str, Any]) -> None:
    """Validate offline benchmark provenance metadata."""
    manifest = _require_mapping(value, "benchmark_manifest")
    _require_keys(manifest, {"schema_version", "benchmarks"}, "benchmark_manifest")
    if manifest["schema_version"] != "2.0":
        raise ValueError("benchmark_manifest schema must be '2.0'")
    if not isinstance(manifest["benchmarks"], list):
        raise ValueError("benchmark_manifest.benchmarks must be a list")
    for index, item in enumerate(manifest["benchmarks"]):
        record = _require_mapping(item, f"benchmark[{index}]")
        _require_keys(record, {"id", "source", "filename", "license", "sha256", "retrieval_date", "schema", "preprocessing", "claim_role"}, f"benchmark[{index}]")
        _require_sha256(record["sha256"], f"benchmark[{index}].sha256")
        for key in ("id", "source", "filename", "license", "schema", "preprocessing", "claim_role"):
            if not isinstance(record[key], str) or not record[key]:
                raise ValueError(f"benchmark[{index}].{key} must be non-empty")
        if not isinstance(record["retrieval_date"], str):
            raise ValueError(f"benchmark[{index}].retrieval_date must be ISO date text")
        try:
            date.fromisoformat(record["retrieval_date"])
        except ValueError as exc:
            raise ValueError(f"benchmark[{index}].retrieval_date must be ISO date text") from exc
        if "source_revision" in record and (
            not isinstance(record["source_revision"], str) or not record["source_revision"]
        ):
            raise ValueError(f"benchmark[{index}].source_revision must be non-empty when present")


def validate_checkpoint(value: Mapping[str, Any], *, version: str = "2.0") -> None:
    """Validate a stage checkpoint before it can be used for resumption."""
    checkpoint = _require_mapping(value, "checkpoint")
    _require_keys(
        checkpoint,
        {
            "schema_version",
            "run_id",
            "config_sha256",
            "claim_ledger_sha256",
            "profile",
            "seed",
            "stage_seed",
            "seed_schedule",
            "workers",
            "stage",
            "completed_replications",
            "partial_metric_accumulators",
            "dependency_hashes",
            "value",
        },
        "checkpoint",
    )
    if checkpoint["schema_version"] != version:
        raise ValueError(f"checkpoint schema must be {version!r}")
    if not isinstance(checkpoint["stage"], str) or not checkpoint["stage"]:
        raise ValueError("checkpoint.stage must be non-empty")
    if not isinstance(checkpoint["run_id"], str) or not checkpoint["run_id"]:
        raise ValueError("checkpoint.run_id must be a non-empty string")
    if not isinstance(checkpoint["profile"], str) or not checkpoint["profile"]:
        raise ValueError("checkpoint.profile must be a non-empty string")
    _require_sha256(checkpoint["config_sha256"], "checkpoint.config_sha256")
    _require_sha256(checkpoint["claim_ledger_sha256"], "checkpoint.claim_ledger_sha256")
    if "execution_source_sha256" in checkpoint:
        _require_sha256(
            checkpoint["execution_source_sha256"],
            "checkpoint.execution_source_sha256",
        )
    _require_nonnegative_integer(checkpoint["seed"], "checkpoint.seed")
    _require_nonnegative_integer(checkpoint["stage_seed"], "checkpoint.stage_seed")
    _require_positive_integer(checkpoint["workers"], "checkpoint.workers")
    _require_positive_integer(checkpoint["completed_replications"], "checkpoint.completed_replications")
    if "total_replications" in checkpoint:
        _require_positive_integer(checkpoint["total_replications"], "checkpoint.total_replications")
        if checkpoint["completed_replications"] > checkpoint["total_replications"]:
            raise ValueError("checkpoint.completed_replications cannot exceed total_replications")
    if "chunk_size" in checkpoint:
        _require_positive_integer(checkpoint["chunk_size"], "checkpoint.chunk_size")
    if "replication_cursor" in checkpoint:
        cursor = _require_mapping(checkpoint["replication_cursor"], "checkpoint.replication_cursor")
        _require_keys(cursor, {"completed", "next"}, "checkpoint.replication_cursor")
        _require_nonnegative_integer(cursor["completed"], "checkpoint.replication_cursor.completed")
        _require_nonnegative_integer(cursor["next"], "checkpoint.replication_cursor.next")
        if cursor["completed"] != checkpoint["completed_replications"] or cursor["next"] < cursor["completed"]:
            raise ValueError("checkpoint.replication_cursor is inconsistent")
    schedule = _require_mapping(checkpoint["seed_schedule"], "checkpoint.seed_schedule")
    _require_keys(schedule, {"kind", "master_seed", "stage", "stage_seed", "replication_order"}, "checkpoint.seed_schedule")
    if schedule["kind"] != "numpy.SeedSequence" or schedule["stage"] != checkpoint["stage"] or schedule["stage_seed"] != checkpoint["stage_seed"]:
        raise ValueError("checkpoint.seed_schedule does not match checkpoint stage metadata")
    partial = _require_mapping(checkpoint["partial_metric_accumulators"], "checkpoint.partial_metric_accumulators")
    _require_keys(partial, {"status", "completed_replications"}, "checkpoint.partial_metric_accumulators")
    if partial["status"] not in {"partial", "stage_complete"} or partial["completed_replications"] != checkpoint["completed_replications"]:
        raise ValueError("checkpoint partial accumulator metadata is inconsistent")
    if partial["status"] == "partial":
        _require_mapping(partial.get("metrics", {}), "checkpoint.partial_metric_accumulators.metrics")
    dependencies = _require_mapping(checkpoint["dependency_hashes"], "checkpoint.dependency_hashes")
    _require_keys(
        dependencies,
        {"scenario_hashes", "model_hashes", "process_hashes", "setting_hashes"},
        "checkpoint.dependency_hashes",
    )
    for key in ("scenario_hashes", "model_hashes", "process_hashes", "setting_hashes"):
        values = _require_list(dependencies[key], f"checkpoint.dependency_hashes.{key}")
        for index, digest in enumerate(values):
            _require_sha256(digest, f"checkpoint.dependency_hashes.{key}[{index}]")


_ACTION_E_PROCESS_ASSUMPTIONS = frozenset(
    {
        "fixed_truth",
        "conditional_e_factor_mean_at_most_one",
        "nonnegative_e_process",
        "predictable_selection",
    }
)
_PER_TARGET_OPTIONAL_STOPPING_SCOPE = "per_target_no_family_error_claim"
_CONDITIONAL_TRACE_VALIDITY = "conditional_per_target_optional_stopping"
_CONDITIONAL_SUMMARY_VALIDITY = "conditional per-target optional-stopping claim"


def _validate_action_execution_contract(
    value: Mapping[str, Any],
    name: str,
    *,
    validity_key: str,
    conditional_validity: str,
    requires_optional_stopping_flag: bool,
    require_execution_contract: bool,
) -> None:
    """Validate modern declared-versus-executed action-loop provenance.

    Earlier artifacts predate the explicit executed fields and remain readable.
    Once an artifact claims the modern execution contract, however, its
    correction and policy provenance must be internally consistent.
    """
    has_executed_correction = "executed_correction" in value
    has_executed_policy = "executed_stopping_policy" in value
    current_fields = {
        "declared_correction",
        "executed_correction",
        "declared_stopping_policy",
        "executed_stopping_policy",
        "policy_override",
        "family_scope",
        "validity_assumptions",
    }
    if requires_optional_stopping_flag:
        current_fields.add("optional_stopping_valid")
    if require_execution_contract:
        missing = sorted(field for field in current_fields if field not in value)
        if missing:
            raise ValueError(
                f"{name} is missing current action-execution fields: {missing}"
            )
    if has_executed_correction:
        if value.get("declared_correction") != "none":
            raise ValueError(f"{name}.declared_correction must be 'none' when execution is recorded")
        if value["executed_correction"] != "none":
            raise ValueError(f"{name}.executed_correction must be 'none'")
    if has_executed_policy:
        declared = value.get("declared_stopping_policy")
        executed = value["executed_stopping_policy"]
        if not isinstance(declared, str) or not declared:
            raise ValueError(f"{name}.declared_stopping_policy must be non-empty")
        if not isinstance(executed, str) or not executed:
            raise ValueError(f"{name}.executed_stopping_policy must be non-empty")
        if value.get("policy") != executed:
            raise ValueError(f"{name}.policy must match executed_stopping_policy")
        override = value.get("policy_override")
        if not isinstance(override, bool):
            raise ValueError(f"{name}.policy_override must be boolean")
        if override != (executed != declared):
            raise ValueError(f"{name}.policy_override must match declared versus executed policy")
        validity = value.get(validity_key)
        if not isinstance(validity, str) or not validity:
            raise ValueError(f"{name}.{validity_key} must be non-empty")
        optional_stopping_flag = value.get("optional_stopping_valid")
        if requires_optional_stopping_flag and not isinstance(optional_stopping_flag, bool):
            raise ValueError(f"{name}.optional_stopping_valid must be boolean")
        if validity.startswith("conditional") and validity != conditional_validity:
            raise ValueError(
                f"{name} uses an unsupported unqualified conditional validity label"
            )
        if validity == conditional_validity:
            if executed != "e_process" or declared != "e_process":
                raise ValueError(
                    f"{name} conditional optional-stopping validity requires the declared and executed e_process policy"
                )
            if value.get("declared_correction") != "none" or value.get("executed_correction") != "none":
                raise ValueError(
                    f"{name} conditional optional-stopping validity requires correction='none'"
                )
            if value.get("family_scope") != _PER_TARGET_OPTIONAL_STOPPING_SCOPE:
                raise ValueError(
                    f"{name} conditional optional-stopping validity must state the per-target family scope"
                )
            assumptions = value.get("validity_assumptions")
            if not isinstance(assumptions, list) or not all(
                isinstance(item, str) for item in assumptions
            ):
                raise ValueError(f"{name}.validity_assumptions must be a list of strings")
            if not _ACTION_E_PROCESS_ASSUMPTIONS.issubset(assumptions):
                raise ValueError(
                    f"{name} conditional optional-stopping validity lacks required e-factor assumptions"
                )
            if requires_optional_stopping_flag and optional_stopping_flag is not True:
                raise ValueError(
                    f"{name} conditional optional-stopping validity requires optional_stopping_valid=true"
                )
        elif requires_optional_stopping_flag and optional_stopping_flag is True:
            raise ValueError(
                f"{name}.optional_stopping_valid=true requires {conditional_validity!r}"
            )
        if require_execution_contract:
            diagnostic_scope = "no_family_error_control_claim"
            diagnostic_validity = (
                "diagnostic_action_adaptive_evidence"
                if requires_optional_stopping_flag
                else "diagnostic finite-simulation contrast"
            )
            if validity not in {conditional_validity, diagnostic_validity}:
                raise ValueError(f"{name} has an unsupported current validity label")
            if validity == diagnostic_validity and value.get("family_scope") != diagnostic_scope:
                raise ValueError(
                    f"{name} diagnostic validity must state no family-error-control claim"
                )
            assumptions = value.get("validity_assumptions")
            if not isinstance(assumptions, list) or not all(
                isinstance(item, str) for item in assumptions
            ):
                raise ValueError(f"{name}.validity_assumptions must be a list of strings")


def validate_action_loop_result(
    value: Mapping[str, Any], *, require_current_contract: bool = False
) -> None:
    """Validate one JSON-shaped action-loop result without inspecting arrays."""
    result = _require_mapping(value, "action_loop_result")
    _require_keys(result, {"scenario_id", "scenario_hash", "policy", "n_reps", "summary", "validity"}, "action_loop_result")
    if not isinstance(result["scenario_id"], str) or not result["scenario_id"]:
        raise ValueError("action_loop_result.scenario_id must be non-empty")
    _require_sha256(result["scenario_hash"], "action_loop_result.scenario_hash")
    if not isinstance(result["policy"], str) or not result["policy"]:
        raise ValueError("action_loop_result.policy must be non-empty")
    _require_positive_integer(result["n_reps"], "action_loop_result.n_reps")
    _require_mapping(result["summary"], "action_loop_result.summary")
    summary = result["summary"]
    for key in ("power", "fdr", "fwer", "mean_cost", "mean_stopping_time", "mean_rejections", "power_per_cost"):
        if key in summary and (
            not isinstance(summary[key], (int, float))
            or isinstance(summary[key], bool)
            or not math.isfinite(float(summary[key]))
        ):
            raise ValueError(f"action_loop_result.summary.{key} must be numeric")
    for key in ("power", "fdr", "fwer"):
        if key in summary and not 0.0 <= float(summary[key]) <= 1.0:
            raise ValueError(f"action_loop_result.summary.{key} must lie in [0, 1]")
    for key in ("decision_interval", "power_interval", "fdr_interval", "fwer_interval"):
        if key in summary:
            _validate_interval(summary[key], f"action_loop_result.summary.{key}", lower=0.0)
            if any(float(item) > 1.0 for item in summary[key]):
                raise ValueError(f"action_loop_result.summary.{key} must lie in [0, 1]")
    if not isinstance(result["validity"], str) or not result["validity"]:
        raise ValueError("action_loop_result.validity must be non-empty")
    if "validity" in summary and result["validity"] != summary["validity"]:
        raise ValueError("action_loop_result.validity must match summary.validity")
    _validate_action_execution_contract(
        summary,
        "action_loop_result.summary",
        validity_key="validity",
        conditional_validity=_CONDITIONAL_SUMMARY_VALIDITY,
        requires_optional_stopping_flag=False,
        require_execution_contract=require_current_contract,
    )
    traces = result.get("traces", [])
    if not isinstance(traces, list):
        raise ValueError("action_loop_result.traces must be a list")
    for index, trace in enumerate(traces):
        item = _require_mapping(trace, f"action_loop_result.traces[{index}]")
        if "validity_assumptions" in item and not isinstance(item["validity_assumptions"], list):
            raise ValueError(f"action_loop_result.traces[{index}].validity_assumptions must be a list")
        _validate_action_execution_contract(
            item,
            f"action_loop_result.traces[{index}]",
            validity_key="validity_boundary",
            conditional_validity=_CONDITIONAL_TRACE_VALIDITY,
            requires_optional_stopping_flag=True,
            require_execution_contract=require_current_contract,
        )


def validate_suite_result(
    value: Mapping[str, Any], *, require_current_contract: bool = False
) -> None:
    """Validate a suite, optionally requiring the complete current contract."""
    suite = _require_mapping(value, "suite")
    _require_keys(suite, {"suite_name", "scenarios", "policies", "operating_characteristics", "online_fdr", "claim_boundary"}, "suite")
    if "contract_version" in suite and suite["contract_version"] != "1.0":
        raise ValueError("suite.contract_version must be '1.0'")
    if not isinstance(suite["suite_name"], str) or not suite["suite_name"]:
        raise ValueError("suite.suite_name must be non-empty")
    for name in ("scenarios", "policies", "operating_characteristics"):
        if not isinstance(suite[name], list) or not suite[name]:
            raise ValueError(f"suite.{name} must be a non-empty list")
    if any(not isinstance(policy, str) or not policy for policy in suite["policies"]):
        raise ValueError("suite.policies must contain non-empty strings")
    declared_scenario_ids: list[str] = []
    contract_scenario_ids: list[str] = []
    contract_pairs: set[tuple[str, str]] = set()
    split_confirmation_pair: tuple[str, str] | None = None
    if "scenario_contracts" in suite:
        contracts = _require_list(suite["scenario_contracts"], "suite.scenario_contracts")
        if not contracts:
            raise ValueError("suite.scenario_contracts must not be empty when present")
        for index, contract_value in enumerate(contracts):
            contract = _validate_scenario_contract(contract_value, f"suite.scenario_contracts[{index}]")
            scenario_id = str(contract["scenario_id"])
            if scenario_id in contract_scenario_ids:
                raise ValueError("suite.scenario_contracts contains duplicate lineage or scenario identifiers")
            contract_scenario_ids.append(scenario_id)
            pair = (str(contract["scenario_id"]), str(contract["scenario_hash"]))
            if pair in contract_pairs:
                raise ValueError("suite.scenario_contracts contains a duplicate lineage pair")
            contract_pairs.add(pair)
    for index, scenario in enumerate(suite["scenarios"]):
        if isinstance(scenario, Mapping):
            contract = _validate_scenario_contract(scenario, f"suite.scenarios[{index}]")
            scenario_id = str(contract["scenario_id"])
            if scenario_id in declared_scenario_ids:
                raise ValueError("suite.scenarios contains duplicate scenario identifiers")
            declared_scenario_ids.append(scenario_id)
            contract_pairs.add((str(contract["scenario_id"]), str(contract["scenario_hash"])))
        elif not isinstance(scenario, str) or not scenario:
            raise ValueError(f"suite.scenarios[{index}] must be a non-empty scenario id or contract")
        elif scenario in declared_scenario_ids:
            raise ValueError("suite.scenarios contains duplicate scenario identifiers")
        else:
            declared_scenario_ids.append(scenario)
    if "scenario_contracts" in suite:
        if set(declared_scenario_ids) != set(contract_scenario_ids):
            raise ValueError("suite.scenarios and suite.scenario_contracts must declare the same scenario identifiers")
    if "split_confirmation_scenario" in suite:
        split_contract = _validate_scenario_contract(
            suite["split_confirmation_scenario"], "suite.split_confirmation_scenario"
        )
        split_confirmation_pair = (
            str(split_contract["scenario_id"]),
            str(split_contract["scenario_hash"]),
        )
    for item in suite["operating_characteristics"]:
        validate_action_loop_result(
            item, require_current_contract=require_current_contract
        )
        if contract_pairs:
            pair = (str(item["scenario_id"]), str(item["scenario_hash"]))
            if pair not in contract_pairs:
                raise ValueError("suite operating characteristic has no matching scenario contract")
        if item["policy"] not in suite["policies"]:
            raise ValueError("suite operating characteristic uses an undeclared policy")
    for key in ("paired_policy_comparisons", "power_cost_frontier"):
        if key in suite and not isinstance(suite[key], list):
            raise ValueError(f"suite.{key} must be a list when present")
    for index, row in enumerate(suite.get("power_cost_frontier", [])):
        frontier = _require_mapping(row, f"suite.power_cost_frontier[{index}]")
        _require_keys(
            frontier,
            {
                "scenario_id", "scenario_hash", "policy", "n_reps", "power", "power_interval",
                "fdr", "fdr_interval", "mean_cost", "cost_interval", "mean_stopping_time",
                "power_per_cost", "validity",
            },
            f"suite.power_cost_frontier[{index}]",
        )
        _require_sha256(frontier["scenario_hash"], f"suite.power_cost_frontier[{index}].scenario_hash")
        _require_positive_integer(frontier["n_reps"], f"suite.power_cost_frontier[{index}].n_reps")
        if not isinstance(frontier["scenario_id"], str) or not frontier["scenario_id"]:
            raise ValueError(f"suite.power_cost_frontier[{index}].scenario_id must be non-empty")
        if not isinstance(frontier["policy"], str) or not frontier["policy"]:
            raise ValueError(f"suite.power_cost_frontier[{index}].policy must be non-empty")
        if frontier["policy"] not in suite["policies"]:
            raise ValueError(f"suite.power_cost_frontier[{index}] uses an undeclared policy")
        for key in ("power", "fdr"):
            value_float = frontier[key]
            if isinstance(value_float, bool) or not isinstance(value_float, (int, float)) or not 0.0 <= float(value_float) <= 1.0:
                raise ValueError(f"suite.power_cost_frontier[{index}].{key} must lie in [0, 1]")
        _validate_interval(frontier["power_interval"], f"suite.power_cost_frontier[{index}].power_interval", lower=0.0)
        _validate_interval(frontier["fdr_interval"], f"suite.power_cost_frontier[{index}].fdr_interval", lower=0.0)
        _validate_interval(frontier["cost_interval"], f"suite.power_cost_frontier[{index}].cost_interval", lower=0.0)
        for key in ("mean_cost", "mean_stopping_time", "power_per_cost"):
            metric = frontier[key]
            if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(float(metric)) or float(metric) < 0.0:
                raise ValueError(f"suite.power_cost_frontier[{index}].{key} must be finite and non-negative")
        if not isinstance(frontier["validity"], str) or not frontier["validity"]:
            raise ValueError(f"suite.power_cost_frontier[{index}].validity must be non-empty")
        if contract_pairs and (frontier["scenario_id"], frontier["scenario_hash"]) not in contract_pairs:
            raise ValueError("suite power-cost frontier row has no matching scenario contract")
    if not isinstance(suite["online_fdr"], Mapping):
        raise ValueError("suite.online_fdr must be a mapping")
    _validate_online_fdr_summary(suite["online_fdr"], "suite.online_fdr")
    if not isinstance(suite["claim_boundary"], str) or not suite["claim_boundary"]:
        raise ValueError("suite.claim_boundary must be non-empty")
    if "conditional_selection_calibration" in suite:
        _validate_selection_diagnostic(
            suite["conditional_selection_calibration"],
            "suite.conditional_selection_calibration",
            "calibration_diagnostic",
        )
    if "future_evidence_selection_failure" in suite:
        _validate_selection_diagnostic(
            suite["future_evidence_selection_failure"],
            "suite.future_evidence_selection_failure",
            "selection_failure_contrast",
        )
    if "split_stream_confirmation" in suite:
        _validate_split_confirmation(
            suite["split_stream_confirmation"],
            "suite.split_stream_confirmation",
        )
        if split_confirmation_pair is not None:
            split_summary = _require_mapping(
                suite["split_stream_confirmation"], "suite.split_stream_confirmation"
            )
            summary_pair = (
                str(split_summary.get("scenario_id", "")),
                str(split_summary.get("scenario_hash", "")),
            )
            if summary_pair != split_confirmation_pair:
                raise ValueError(
                    "suite.split_stream_confirmation does not match "
                    "suite.split_confirmation_scenario"
                )
    if "learned_policy" in suite:
        learned = _require_mapping(suite["learned_policy"], "suite.learned_policy")
        if not {"artifact", "evaluation"}.issubset(learned):
            raise ValueError("suite.learned_policy must include artifact and evaluation")
        _require_keys(
            learned,
            {"artifact", "evaluation"},
            "suite.learned_policy",
        )
        artifact = _require_mapping(learned["artifact"], "suite.learned_policy.artifact")
        evaluation = _require_mapping(learned["evaluation"], "suite.learned_policy.evaluation")
        _require_keys(
            artifact,
            {
                "scenario_id", "scenario_hash", "parameters", "train_seed",
                "train_reps", "objective", "visible_features",
                "search_config_hash", "policy_complexity", "policy_hash",
            },
            "suite.learned_policy.artifact",
        )
        _require_sha256(
            artifact["policy_hash"], "suite.learned_policy.artifact.policy_hash"
        )
        _require_sha256(
            artifact["search_config_hash"],
            "suite.learned_policy.artifact.search_config_hash",
        )
        complexity = _require_mapping(
            artifact["policy_complexity"],
            "suite.learned_policy.artifact.policy_complexity",
        )
        if complexity.get("search_type") != "finite_tabular_candidate_search":
            raise ValueError("suite.learned_policy.artifact.policy_complexity.search_type is invalid")
        _require_positive_integer(
            complexity.get("candidate_count"),
            "suite.learned_policy.artifact.policy_complexity.candidate_count",
        )
        _require_positive_integer(
            complexity.get("parameter_dimension"),
            "suite.learned_policy.artifact.policy_complexity.parameter_dimension",
        )
        parameters = _require_list(
            artifact["parameters"], "suite.learned_policy.artifact.parameters"
        )
        visible_features = _require_list(
            artifact["visible_features"],
            "suite.learned_policy.artifact.visible_features",
        )
        try:
            frozen_artifact = LearnedPolicyArtifact(
                scenario_id=artifact["scenario_id"],
                scenario_hash=artifact["scenario_hash"],
                parameters=tuple(parameters),
                train_seed=artifact["train_seed"],
                train_reps=artifact["train_reps"],
                objective=artifact["objective"],
                visible_features=tuple(visible_features),
                search_config_hash=artifact["search_config_hash"],
                candidate_count=complexity["candidate_count"],
                parameter_dimension=complexity["parameter_dimension"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "suite.learned_policy.artifact fails the frozen-policy contract"
            ) from exc
        if artifact["policy_hash"] != frozen_artifact.policy_hash:
            raise ValueError(
                "suite.learned_policy.artifact.policy_hash does not match the frozen artifact"
            )
        _require_keys(
            evaluation,
            {
                "policy", "policy_hash", "scenario_id", "scenario_hash",
                "train_seed", "evaluation_seed", "train_reps", "evaluation_reps",
                "search_config_hash", "policy_complexity", "summary", "seed_schedule",
                "validity", "claim_boundary",
            },
            "suite.learned_policy.evaluation",
        )
        if evaluation["policy"] != "learned":
            raise ValueError("suite.learned_policy.evaluation.policy must be 'learned'")
        _require_sha256(
            evaluation["policy_hash"], "suite.learned_policy.evaluation.policy_hash"
        )
        _require_sha256(
            evaluation["scenario_hash"], "suite.learned_policy.evaluation.scenario_hash"
        )
        _require_sha256(
            evaluation["search_config_hash"],
            "suite.learned_policy.evaluation.search_config_hash",
        )
        _require_nonnegative_integer(
            evaluation["train_seed"], "suite.learned_policy.evaluation.train_seed"
        )
        _require_nonnegative_integer(
            evaluation["evaluation_seed"],
            "suite.learned_policy.evaluation.evaluation_seed",
        )
        _require_positive_integer(
            evaluation["train_reps"], "suite.learned_policy.evaluation.train_reps"
        )
        evaluation_reps = evaluation["evaluation_reps"]
        _require_positive_integer(
            evaluation_reps, "suite.learned_policy.evaluation.evaluation_reps"
        )
        if not isinstance(evaluation["scenario_id"], str) or not evaluation["scenario_id"]:
            raise ValueError("suite.learned_policy.evaluation.scenario_id must be non-empty")
        if not isinstance(evaluation["summary"], Mapping):
            raise ValueError("suite.learned_policy.evaluation.summary must be a mapping")
        seed_schedule = _require_list(
            evaluation["seed_schedule"], "suite.learned_policy.evaluation.seed_schedule"
        )
        if len(seed_schedule) != evaluation_reps:
            raise ValueError(
                "suite.learned_policy.evaluation.seed_schedule must align with evaluation_reps"
            )
        for index, seed in enumerate(seed_schedule):
            _require_nonnegative_integer(
                seed, f"suite.learned_policy.evaluation.seed_schedule[{index}]"
            )
        if not isinstance(evaluation["validity"], str) or not evaluation["validity"].startswith("conditional_held_out"):
            raise ValueError(
                "suite.learned_policy.evaluation.validity must state the conditional held-out boundary"
            )
        if not isinstance(evaluation["claim_boundary"], str) or not evaluation["claim_boundary"]:
            raise ValueError("suite.learned_policy.evaluation.claim_boundary must be non-empty")
        if evaluation["scenario_id"] != frozen_artifact.scenario_id or evaluation["scenario_hash"] != frozen_artifact.scenario_hash:
            raise ValueError("suite.learned_policy evaluation does not match frozen scenario provenance")
        if evaluation["policy_hash"] != frozen_artifact.policy_hash:
            raise ValueError("suite.learned_policy evaluation does not match frozen policy hash")
        if evaluation["search_config_hash"] != frozen_artifact.search_config_hash:
            raise ValueError("suite.learned_policy evaluation does not match frozen search provenance")
        if evaluation["train_seed"] != frozen_artifact.train_seed or evaluation["train_reps"] != frozen_artifact.train_reps:
            raise ValueError("suite.learned_policy evaluation does not match frozen training provenance")
        if evaluation["evaluation_seed"] == frozen_artifact.train_seed:
            raise ValueError("suite.learned_policy evaluation must use a held-out seed")
        evaluation_complexity = _require_mapping(
            evaluation["policy_complexity"],
            "suite.learned_policy.evaluation.policy_complexity",
        )
        if dict(evaluation_complexity) != {
            "search_type": "finite_tabular_candidate_search",
            "candidate_count": frozen_artifact.candidate_count,
            "parameter_dimension": frozen_artifact.parameter_dimension,
        }:
            raise ValueError(
                "suite.learned_policy evaluation does not match frozen policy complexity"
            )
        if contract_pairs and (
            frozen_artifact.scenario_id, frozen_artifact.scenario_hash
        ) not in contract_pairs:
            raise ValueError(
                "suite.learned_policy artifact has no matching scenario contract"
            )
    if require_current_contract:
        _validate_current_suite_contract(suite)


_CURRENT_SPLIT_CONFIRMATION_ESTIMAND_FIELDS = frozenset(
    {
        "rejection_count",
        "true_target_selection_count",
        "true_target_selection_rate",
        "true_target_joint_rejection_count",
        "true_target_joint_rejection_rate",
        "power_defined",
        "power_estimand",
        "false_rejection_count",
        "fwer_estimand",
    }
)


def _require_probability(value: Any, name: str) -> float:
    """Validate one finite probability and return its float representation."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must lie in [0, 1]")
    return float(value)


def _validate_current_split_confirmation_estimand(
    result: Mapping[str, Any], name: str
) -> None:
    """Require the current, denominator-explicit split-confirmation metrics."""
    _require_keys(result, _CURRENT_SPLIT_CONFIRMATION_ESTIMAND_FIELDS, name)
    n_reps_value = result.get("n_reps")
    n_reps = _require_positive_integer(n_reps_value, f"{name}.n_reps")
    counts: dict[str, int] = {}
    for key in (
        "rejection_count",
        "true_target_selection_count",
        "true_target_joint_rejection_count",
        "false_rejection_count",
    ):
        value = result.get(key)
        counts[key] = _require_nonnegative_integer(value, f"{name}.{key}")
        if counts[key] > n_reps:
            raise ValueError(f"{name}.{key} cannot exceed n_reps")
    if counts["true_target_joint_rejection_count"] > counts["true_target_selection_count"]:
        raise ValueError(f"{name}.true_target_joint_rejection_count exceeds true selections")
    if counts["rejection_count"] != (
        counts["true_target_joint_rejection_count"] + counts["false_rejection_count"]
    ):
        raise ValueError(f"{name}.rejection_count does not partition by selected-target truth")

    rates = {
        "rejection_rate": ("rejection_count", "rejection rate"),
        "true_target_selection_rate": (
            "true_target_selection_count",
            "true-target selection rate",
        ),
        "true_target_joint_rejection_rate": (
            "true_target_joint_rejection_count",
            "joint true-target rejection rate",
        ),
        "fwer": ("false_rejection_count", "false-rejection rate"),
    }
    for key, (count_key, label) in rates.items():
        observed = _require_probability(result.get(key), f"{name}.{key}")
        expected = counts[count_key] / n_reps
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{name}.{label} does not match its recorded count")

    power_defined = result.get("power_defined")
    if not isinstance(power_defined, bool):
        raise ValueError(f"{name}.power_defined must be boolean")
    expected_power_defined = counts["true_target_selection_count"] > 0
    if power_defined is not expected_power_defined:
        raise ValueError(f"{name}.power_defined does not match true-target selection count")
    power = result.get("power")
    if power_defined:
        observed_power = _require_probability(power, f"{name}.power")
        expected_power = (
            counts["true_target_joint_rejection_count"]
            / counts["true_target_selection_count"]
        )
        if not math.isclose(observed_power, expected_power, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{name}.power does not match its selected-true denominator")
    elif power is not None:
        raise ValueError(f"{name}.power must be null when no true target was selected")
    if (
        result.get("power_estimand")
        != "conditional_confirmation_rejection_given_selected_target_true"
    ):
        raise ValueError(f"{name}.power_estimand is not the declared conditional estimand")
    if result.get("fwer_estimand") != "single_confirmed_target_false_rejection_rate":
        raise ValueError(f"{name}.fwer_estimand is not the declared one-target false-rejection rate")


def _validate_split_confirmation(value: Any, name: str) -> None:
    """Validate the split-stream selection/confirmation contract."""
    result = _require_mapping(value, name)
    _require_keys(
        result,
        {
            "scenario_id", "scenario_hash", "n_reps", "selection_streams",
            "confirmation_streams", "rejection_rate", "power", "fwer",
            "validity", "assumptions", "representative",
        },
        name,
    )
    if not isinstance(result["scenario_id"], str) or not result["scenario_id"]:
        raise ValueError(f"{name}.scenario_id must be non-empty")
    _require_sha256(result["scenario_hash"], f"{name}.scenario_hash")
    for key in ("n_reps", "selection_streams", "confirmation_streams"):
        _require_positive_integer(result[key], f"{name}.{key}")
    for key in ("rejection_rate", "fwer"):
        _require_probability(result[key], f"{name}.{key}")
    power = result["power"]
    if power is not None:
        _require_probability(power, f"{name}.power")
    present_estimand_fields = _CURRENT_SPLIT_CONFIRMATION_ESTIMAND_FIELDS.intersection(result)
    if present_estimand_fields and present_estimand_fields != _CURRENT_SPLIT_CONFIRMATION_ESTIMAND_FIELDS:
        missing = sorted(_CURRENT_SPLIT_CONFIRMATION_ESTIMAND_FIELDS.difference(result))
        raise ValueError(f"{name} has an incomplete split-confirmation estimand contract: {missing}")
    if present_estimand_fields:
        _validate_current_split_confirmation_estimand(result, name)
    if result["selection_streams"] != result["confirmation_streams"] or result["n_reps"] != result["selection_streams"]:
        raise ValueError(f"{name} stream counts must equal n_reps")
    if not isinstance(result["validity"], str) or not result["validity"]:
        raise ValueError(f"{name}.validity must be non-empty")
    if not isinstance(result["assumptions"], list) or not result["assumptions"]:
        raise ValueError(f"{name}.assumptions must be a non-empty list")
    representative = _require_mapping(result["representative"], f"{name}.representative")
    for key in ("selection_stream_hash", "confirmation_stream_hash"):
        _require_sha256(representative.get(key), f"{name}.representative.{key}")
    selection_value = representative.get("selection")
    # Compact summaries may retain only the two stream hashes.  Full release
    # representatives include the visible filtration and are checked below;
    # compact records remain structurally valid without pretending that a
    # payload absent from the artifact can be re-hashed.
    if selection_value is None:
        return
    selection = _require_mapping(selection_value, f"{name}.representative.selection")
    _require_keys(
        selection,
        {
            "scenario_id", "scenario_hash", "policy", "seed", "events",
            "selected_target", "selected_target_evidence", "visible_only", "stream_hash",
        },
        f"{name}.representative.selection",
    )
    _require_sha256(selection["stream_hash"], f"{name}.representative.selection.stream_hash")
    if selection["stream_hash"] != representative["selection_stream_hash"]:
        raise ValueError(f"{name}.representative selection hash is inconsistent")
    selection_payload = {
        key: selection[key]
        for key in (
            "scenario_id", "scenario_hash", "policy", "seed", "events",
            "selected_target", "selected_target_evidence", "visible_only",
        )
    }
    if _payload_digest(selection_payload) != selection["stream_hash"]:
        raise ValueError(f"{name}.representative selection hash does not match its payload")
    confirmation_payload = {
        "scenario_id": representative.get("scenario_id"),
        "scenario_hash": representative.get("scenario_hash"),
        "target": selection.get("selected_target"),
        "seed": representative.get("confirmation_seed"),
        "observations": representative.get("confirmation_observations"),
        "selected_target_truth": representative.get("selected_target_truth"),
        "confirmation_p_value": representative.get("confirmation_p_value"),
        "rejected": representative.get("rejected"),
        "validity": representative.get("validity"),
        "validity_assumptions": representative.get("validity_assumptions"),
    }
    if _payload_digest(confirmation_payload) != representative["confirmation_stream_hash"]:
        raise ValueError(f"{name}.representative confirmation hash does not match its payload")


def _validate_online_fdr_summary(value: Any, name: str) -> None:
    """Validate bounded adaptive-FDR summary metrics when they are present."""
    summary = _require_mapping(value, name)
    if not summary:
        return
    if "method" in summary and summary["method"] not in {"lord_plus_plus", "saffron", "e_lond"}:
        raise ValueError(f"{name}.method is invalid")
    if "n_reps" in summary:
        _require_positive_integer(summary["n_reps"], f"{name}.n_reps")
    for key in (
        "any_rejection_rate", "mean_fdr", "mean_power", "mean_test_power",
        "mean_target_fdr", "fwer",
    ):
        if key not in summary:
            continue
        metric = summary[key]
        if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(float(metric)) or not 0.0 <= float(metric) <= 1.0:
            raise ValueError(f"{name}.{key} must lie in [0, 1]")
    for key in (
        "any_rejection_interval", "mean_fdr_interval", "mean_power_interval",
        "mean_test_power_interval", "mean_target_fdr_interval", "fwer_interval",
    ):
        if key in summary:
            _validate_interval(summary[key], f"{name}.{key}", lower=0.0)
            if any(float(item) > 1.0 for item in summary[key]):
                raise ValueError(f"{name}.{key} must lie in [0, 1]")
    for key in ("power_estimand", "test_power_estimand", "fdr_estimand", "target_fdr_estimand", "fwer_estimand"):
        if key in summary and (not isinstance(summary[key], str) or not summary[key]):
            raise ValueError(f"{name}.{key} must be a non-empty string")


def _validate_selection_diagnostic(
    value: Any, name: str, expected_validity: str
) -> None:
    """Validate a selected-p-value diagnostic or explicit failure contrast."""
    calibration = _require_mapping(value, name)
    _require_keys(
        calibration,
        {"scenario_id", "scenario_hash", "policy", "n_reps", "pooled", "validity"},
        name,
    )
    for key in ("scenario_id", "policy"):
        if not isinstance(calibration[key], str) or not calibration[key]:
            raise ValueError(f"{name}.{key} must be non-empty")
    _require_sha256(calibration["scenario_hash"], f"{name}.scenario_hash")
    _require_positive_integer(calibration["n_reps"], f"{name}.n_reps")
    if calibration["validity"] != expected_validity:
        raise ValueError(f"{name}.validity must be {expected_validity}")
    pooled = _require_mapping(calibration["pooled"], f"{name}.pooled")
    _require_keys(
        pooled,
        {"n_selected", "max_ecdf_excess", "dkw_radius", "passes_dkw_diagnostic"},
        f"{name}.pooled",
    )
    _require_nonnegative_integer(pooled["n_selected"], f"{name}.pooled.n_selected")
    if not isinstance(pooled["passes_dkw_diagnostic"], bool):
        raise ValueError(f"{name}.pooled.passes_dkw_diagnostic must be boolean")
    for key in ("max_ecdf_excess", "dkw_radius"):
        number = pooled[key]
        if number is not None and (
            isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)) or number < 0
        ):
            raise ValueError(f"{name}.pooled.{key} must be null or non-negative finite")
    if "dkw_interpretation" in pooled and (
        not isinstance(pooled["dkw_interpretation"], str) or not pooled["dkw_interpretation"]
    ):
        raise ValueError(f"{name}.pooled.dkw_interpretation must be non-empty when present")


__all__ = [
    "validate_results", "validate_manifest", "validate_figure_registry", "validate_quality_report", "validate_run_log", "validate_benchmark_manifest", "validate_claim_ledger",
    "validate_checkpoint", "validate_action_loop_result", "validate_suite_result",
    "validate_computational_complexity",
]
