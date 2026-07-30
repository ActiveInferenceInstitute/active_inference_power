"""Fail-closed edge coverage for artifact, numerical, and policy boundaries."""

from __future__ import annotations

import hashlib
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from active_inference_power._types import CorrectionResult, ErrorSummary, SimulationResult, TwoGroupsConfig
from active_inference_power.active_inference import AITestConfig
from active_inference_power.agent_sampling import run_agent_policy, simulate_agent_policy
from active_inference_power.benchmarks import BenchmarkDataset, BenchmarkSpec, load_agent_trace, load_csv_benchmark
from active_inference_power.online_fdr import OnlineFDRResult, lord_plus_plus, saffron
from active_inference_power.dependence import DependenceConfig, correlation_matrix, sample_gaussian
from active_inference_power.schema import (
    _current_row_identity,
    _current_scenario_contract_index,
    _payload_digest,
    _scenario_contract_digest,
    _validate_current_frontier_projection,
    _validate_current_suite_contract,
    validate_benchmark_manifest,
    validate_checkpoint,
    validate_figure_registry,
    validate_manifest,
    validate_results,
    validate_run_log,
)
from active_inference_power.formalism import formalism_registry
from active_inference_power.uncertainty import (
    clopper_pearson_interval,
    empirical_bernstein_radius,
    hoeffding_radius,
    paired_difference_summary,
    required_reps_for_half_width,
    summarize_bounded,
    wilson_interval,
)


def test_schema_contracts_fail_closed_on_wrong_shapes() -> None:
    result = {
        "schema_version": "2.0",
        "run_id": "run",
        "meta": {"config_sha256": "a" * 64, "seed": 1, "n_reps": 2},
        **{key: {} for key in ("experiment", "certificates", "method_comparison", "paired_comparisons", "active_inference", "sequential", "online_fdr", "agent_sampling", "benchmarks", "null_pvalue_calibration", "sample_size_accounting")},
        "power_curve": [],
        "dependence_sweep": [],
        "formalism": formalism_registry().to_mapping(),
    }
    validate_results(result)
    with pytest.raises(ValueError):
        validate_results({"schema_version": "2.0", "run_id": "run", "meta": []})
    broken_meta = dict(result, meta={"seed": 1})
    with pytest.raises(ValueError):
        validate_results(broken_meta)
    with pytest.raises(ValueError):
        validate_results(dict(result, run_id=""))
    with pytest.raises(ValueError):
        validate_results(dict(result, power_curve={}))
    with pytest.raises(ValueError):
        validate_results({key: value for key, value in result.items() if key != "experiment"})

    artifact = {"path": "x", "sha256": "a" * 64, "bytes": 1, "role": "data"}
    validate_manifest({"schema_version": "2.0", "run_id": "run", "complete": True, "config_sha256": "a" * 64, "artifacts": {"x": artifact}})
    with pytest.raises(ValueError):
        validate_manifest({"schema_version": "2.0", "run_id": "run", "complete": "yes", "config_sha256": "a" * 64, "artifacts": {}})
    with pytest.raises(ValueError):
        validate_manifest({"schema_version": "2.0", "run_id": "run", "complete": True, "config_sha256": "a" * 64, "artifacts": {"x": {"sha256": "short"}}})
    with pytest.raises(ValueError):
        validate_manifest({"schema_version": "2.0", "run_id": "", "complete": True, "config_sha256": "a" * 64, "artifacts": {"x": artifact}})
    with pytest.raises(ValueError):
        validate_manifest({"schema_version": "2.0", "run_id": "run", "complete": True, "config_sha256": "a" * 64, "artifacts": {"x": {**artifact, "bytes": 0}}})
    with pytest.raises(ValueError):
        validate_manifest({"schema_version": "2.0", "run_id": "run", "complete": True, "config_sha256": "a" * 64, "artifacts": {"x": {**artifact, "role": ""}}})

    validate_run_log({"schema_version": "2.0", "run_id": "run", "status": "interrupted", "events": [], "config_sha256": "a" * 64, "finished_at_utc": "2026-07-19T00:00:00+00:00", "error": {"type": "KeyboardInterrupt", "message": "stop", "traceback": "KeyboardInterrupt"}})
    with pytest.raises(ValueError):
        validate_run_log({"schema_version": "2.0", "run_id": "run", "status": "unknown", "events": [], "config_sha256": "a" * 64})
    with pytest.raises(ValueError):
        validate_run_log({"schema_version": "2.0", "run_id": "run", "status": "running", "events": {}, "config_sha256": "a" * 64})
    with pytest.raises(ValueError):
        validate_run_log({"schema_version": "2.0", "run_id": "", "status": "running", "events": [], "config_sha256": "a" * 64})
    with pytest.raises(ValueError):
        validate_run_log({"schema_version": "2.0", "run_id": "run", "status": "running", "events": [{"name": "", "status": "started", "at_utc": "2026-07-19T00:00:00+00:00", "elapsed_seconds": 0.0}], "config_sha256": "a" * 64})
    with pytest.raises(ValueError, match="run_log.error"):
        validate_run_log({"schema_version": "2.0", "run_id": "run", "status": "failed", "events": [], "config_sha256": "a" * 64, "finished_at_utc": "2026-07-19T00:00:00+00:00"})

    benchmark = {"schema_version": "2.0", "benchmarks": [{"id": "b", "source": "local", "filename": "b.csv", "license": "MIT", "sha256": "b" * 64, "retrieval_date": "2026-07-19", "schema": "csv", "preprocessing": "none", "claim_role": "stress"}]}
    validate_benchmark_manifest(benchmark)
    with pytest.raises(ValueError):
        validate_benchmark_manifest({"schema_version": "2.0", "benchmarks": [{"id": "b"}]})
    checkpoint = {"schema_version": "2.0", "run_id": "run", "config_sha256": "a" * 64, "claim_ledger_sha256": "b" * 64, "profile": "small", "seed": 1, "stage_seed": 2, "seed_schedule": {"kind": "numpy.SeedSequence", "master_seed": 1, "stage": "power", "stage_seed": 2, "replication_order": "stable ascending child-stream order"}, "workers": 1, "stage": "power", "completed_replications": 1, "partial_metric_accumulators": {"status": "stage_complete", "completed_replications": 1}, "dependency_hashes": {"scenario_hashes": [], "model_hashes": [], "process_hashes": [], "setting_hashes": []}, "value": {}}
    validate_checkpoint(checkpoint)
    with pytest.raises(ValueError):
        validate_checkpoint(dict(checkpoint, stage=""))
    with pytest.raises(ValueError):
        validate_checkpoint(dict(checkpoint, completed_replications=0))
    with pytest.raises(ValueError):
        validate_checkpoint({key: value for key, value in checkpoint.items() if key != "stage_seed"})
    with pytest.raises(ValueError):
        validate_checkpoint(dict(checkpoint, run_id=""))


def test_benchmark_loaders_reject_missing_nonfinite_and_malformed_payloads(tmp_path: Path) -> None:
    csv_path = tmp_path / "values.csv"
    csv_path.write_text("x,y\n1,nan\n", encoding="utf-8")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    spec = BenchmarkSpec("values", "local", "MIT", digest)
    with pytest.raises(ValueError, match="finite"):
        load_csv_benchmark(csv_path, spec)
    with pytest.raises(ValueError, match="checksum"):
        load_csv_benchmark(csv_path, BenchmarkSpec("values", "local", "MIT", "0" * 64))
    with pytest.raises(FileNotFoundError):
        load_csv_benchmark(tmp_path / "missing.csv", spec)

    trace_path = tmp_path / "trace.csv"
    trace_path.write_text("subject,time,action\ns,1,0\n", encoding="utf-8")
    trace_spec = BenchmarkSpec("trace", "local", "MIT", hashlib.sha256(trace_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="columns"):
        load_agent_trace(trace_path, trace_spec)
    trace_path.write_text("subject,time,action,observation\ns,nan,0,2\n", encoding="utf-8")
    trace_spec = BenchmarkSpec("trace", "local", "MIT", hashlib.sha256(trace_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="missing|binary"):
        load_agent_trace(trace_path, trace_spec)
    trace_path.write_text("subject,time,action,observation\ns,1,observe,0\ns,1,observe,1\n", encoding="utf-8")
    trace_spec = BenchmarkSpec("trace", "local", "MIT", hashlib.sha256(trace_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="chronological"):
        load_agent_trace(trace_path, trace_spec)
    trace_path.write_text("subject,time,action,observation\ns,1,,0\n", encoding="utf-8")
    trace_spec = BenchmarkSpec("trace", "local", "MIT", hashlib.sha256(trace_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="missing"):
        load_agent_trace(trace_path, trace_spec)


def test_benchmark_metadata_and_payload_contracts_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        BenchmarkSpec("", "local", "MIT", "a" * 64)
    with pytest.raises(ValueError):
        BenchmarkSpec("x", "local", "MIT", "not-a-digest")
    spec = BenchmarkSpec("x", "local", "MIT", "a" * 64)
    with pytest.raises(ValueError):
        BenchmarkDataset("x", np.array([1.0]), ("x",), spec)
    with pytest.raises(ValueError):
        BenchmarkDataset("x", np.array([[np.nan]]), ("x",), spec)
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    empty_spec = BenchmarkSpec("empty", "local", "MIT", hashlib.sha256(empty.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="header"):
        load_csv_benchmark(empty, empty_spec)
    numeric = tmp_path / "numeric.csv"
    numeric.write_text("x\nnot-a-number\n", encoding="utf-8")
    numeric_spec = BenchmarkSpec("numeric", "local", "MIT", hashlib.sha256(numeric.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        load_csv_benchmark(numeric, numeric_spec)
    trace = tmp_path / "trace.csv"
    trace.write_text("subject,time,action,observation\ns,not-a-number,observe,0\n", encoding="utf-8")
    trace_spec = BenchmarkSpec("trace", "local", "MIT", hashlib.sha256(trace.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="numeric"):
        load_agent_trace(trace, trace_spec)
    empty_trace = tmp_path / "empty-trace.csv"
    empty_trace.write_text("subject,time,action,observation\n", encoding="utf-8")
    empty_trace_spec = BenchmarkSpec("empty-trace", "local", "MIT", hashlib.sha256(empty_trace.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="at least one row"):
        load_agent_trace(empty_trace, empty_trace_spec)


def test_online_fdr_result_and_schedule_contracts_fail_closed() -> None:
    arrays: dict[str, Any] = {
        "method": "test",
        "pvalues": np.array([0.1]),
        "levels": np.array([0.01]),
        "candidates": np.array([True]),
        "rejections": np.array([False]),
        "wealth": np.array([0.02, 0.01]),
        "alpha": 0.05,
        "gamma": np.array([1.0]),
        "initial_wealth": 0.02,
    }
    with pytest.raises(ValueError):
        OnlineFDRResult(**{**arrays, "pvalues": np.array([])})
    with pytest.raises(ValueError):
        OnlineFDRResult(**{**arrays, "levels": np.array([0.01, 0.01])})
    with pytest.raises(ValueError):
        OnlineFDRResult(**{**arrays, "wealth": np.array([0.02])})
    with pytest.raises(ValueError):
        OnlineFDRResult(**{**arrays, "levels": np.array([np.nan])})
    with pytest.raises(ValueError):
        OnlineFDRResult(**{**arrays, "levels": np.array([0.06])})
    with pytest.raises(ValueError):
        OnlineFDRResult(**{**arrays, "initial_wealth": 0.06})
    with pytest.raises(ValueError):
        OnlineFDRResult(**{**arrays, "wealth": np.array([0.01, 0.01])})
    with pytest.raises(ValueError):
        lord_plus_plus([0.1], w0=-0.1)
    with pytest.raises(ValueError):
        saffron([0.1], lambda_=0.8, w0=0.02)
    with pytest.raises(ValueError):
        lord_plus_plus([0.1, 0.2], gamma=[0.2, 0.9])


def test_strict_schema_types_reject_inconsistent_metadata() -> None:
    valid_result: dict[str, Any] = {
        "schema_version": "2.0",
        "run_id": "run",
        "meta": {"config_sha256": "a" * 64, "seed": 1, "n_reps": 2},
        **{key: {} for key in ("experiment", "certificates", "method_comparison", "paired_comparisons", "active_inference", "sequential", "online_fdr", "agent_sampling", "benchmarks", "null_pvalue_calibration", "sample_size_accounting")},
        "power_curve": [],
        "dependence_sweep": [],
        "formalism": formalism_registry().to_mapping(),
    }
    with pytest.raises(ValueError):
        validate_results(dict(valid_result, meta={**valid_result["meta"], "seed": True}))
    artifact = {"path": "data/x", "sha256": "a" * 64, "bytes": 1, "role": "data"}
    with pytest.raises(ValueError):
        validate_manifest({"schema_version": "2.0", "run_id": "r", "complete": True, "config_sha256": "a" * 64, "artifacts": {"x": {**artifact, "path": "../x"}}})
    event = {"name": "stage", "status": "started", "at_utc": "2026-07-19T00:00:00+00:00", "elapsed_seconds": 0.0}
    with pytest.raises(ValueError):
        validate_run_log({"schema_version": "2.0", "run_id": "r", "status": "running", "events": [{**event, "status": "bad"}], "config_sha256": "a" * 64})
    with pytest.raises(ValueError):
        validate_run_log({"schema_version": "2.0", "run_id": "r", "status": "running", "events": [{**event, "at_utc": "local"}], "config_sha256": "a" * 64})
    with pytest.raises(ValueError):
        validate_run_log({"schema_version": "2.0", "run_id": "r", "status": "running", "events": [{**event, "elapsed_seconds": -1.0}], "config_sha256": "a" * 64})
    with pytest.raises(ValueError):
        validate_benchmark_manifest({"schema_version": "2.0", "benchmarks": [{"id": "x", "source": "", "filename": "x", "license": "MIT", "sha256": "a" * 64, "retrieval_date": "2026-07-19", "schema": "csv", "preprocessing": "none", "claim_role": "stress"}]})
    with pytest.raises(ValueError):
        validate_benchmark_manifest({"schema_version": "2.0", "benchmarks": [{"id": "x", "source": "local", "filename": "x", "license": "MIT", "sha256": "a" * 64, "retrieval_date": "not-a-date", "schema": "csv", "preprocessing": "none", "claim_role": "stress"}]})


def test_nested_release_contracts_reject_stale_v2_blocks(
    generated_fast_bundle: Path,
) -> None:
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    for mutation, message in (
        (lambda value: value["action_loop"].update(contract_version="stale"), "contract_version"),
        (lambda value: value["action_loop"].update(scenarios={}), "scenarios"),
        (lambda value: value["action_loop"].update(operating_characteristics={}), "operating_characteristics"),
        (lambda value: value["action_loop"].update(paired_policy_comparisons={}), "paired_policy_comparisons"),
    ):
        broken = copy.deepcopy(results)
        mutation(broken)
        with pytest.raises(ValueError, match=message):
            validate_results(broken, require_suite=True)

    broken_hash = copy.deepcopy(results)
    broken_hash["action_loop"]["scenarios"][0]["scenario_hash"] = "short"
    with pytest.raises(ValueError, match="scenario_hash"):
        validate_results(broken_hash, require_suite=True)

    registry = json.loads(
        (generated_fast_bundle / "figures" / "figure_registry.json").read_text(
            encoding="utf-8"
        )
    )
    with pytest.raises(ValueError, match="schema"):
        validate_figure_registry({**registry, "schema_version": "1.0"})
    with pytest.raises(ValueError, match="must not be empty"):
        validate_figure_registry({"schema_version": "2.0", "figures": []})
    with pytest.raises(ValueError, match="differ from expected"):
        validate_figure_registry(registry, expected_filenames={"not-a-figure.png"})

    artifact = {"path": "data/x", "sha256": "a" * 64, "bytes": 1, "role": "data"}
    with pytest.raises(ValueError, match="root"):
        validate_manifest({"schema_version": "2.0", "run_id": "r", "complete": True, "config_sha256": "a" * 64, "artifacts": {"x": {**artifact, "root": "unsafe"}}})

    progress = {"completed_stages": 0, "total_stages": 1, "completed_replications": 0, "estimated_remaining_seconds": -1.0}
    with pytest.raises(ValueError, match="estimated_remaining"):
        validate_run_log({"schema_version": "2.0", "run_id": "r", "status": "running", "events": [], "config_sha256": "a" * 64, "progress": progress})

    with pytest.raises(ValueError, match="schema"):
        validate_benchmark_manifest({"schema_version": "1.0", "benchmarks": []})

    learned = copy.deepcopy(results)
    learned["suite"]["learned_policy"].pop("artifact")
    with pytest.raises(ValueError, match="artifact and evaluation"):
        validate_results(learned, require_suite=True)
    learned = copy.deepcopy(results)
    learned["suite"]["learned_policy"]["evaluation"]["validity"] = ""
    with pytest.raises(ValueError, match="evaluation.validity"):
        validate_results(learned, require_suite=True)

    malformed_certificate = copy.deepcopy(results)
    malformed_certificate["certificates"]["bh_fdr_control"]["passes"] = "true"
    with pytest.raises(ValueError, match="passes must be boolean"):
        validate_results(malformed_certificate, require_suite=True)


def test_current_release_contract_is_opt_in_and_binds_suite_action_lineage(
    generated_fast_bundle: Path,
) -> None:
    """Current release gates need complete, matching contracts; old readers do not."""
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )

    # The actual current producer emits the complete suite/action-loop projection.
    validate_results(results, require_suite=True, require_current_contract=True)
    split_contract = results["suite"]["split_confirmation_scenario"]
    split_summary = results["suite"]["split_stream_confirmation"]
    assert split_contract["scenario_id"] == split_summary["scenario_id"]
    assert split_contract["scenario_hash"] == split_summary["scenario_hash"]
    assert split_contract["scenario_id"] not in results["suite"]["scenarios"]

    # A compact suite remains readable under the default contract.
    compact = copy.deepcopy(results)
    compact["suite"].pop("scenario_contracts")
    validate_results(compact, require_suite=True)
    with pytest.raises(ValueError, match="requires scenario_contracts"):
        validate_results(
            compact, require_suite=True, require_current_contract=True
        )
    missing_split_contract = copy.deepcopy(results)
    missing_split_contract["suite"].pop("split_confirmation_scenario")
    with pytest.raises(ValueError, match="split-confirmation scenario"):
        validate_results(
            missing_split_contract, require_suite=True, require_current_contract=True
        )
    missing_execution_field = copy.deepcopy(results)
    missing_execution_field["suite"]["operating_characteristics"][0]["summary"].pop(
        "executed_stopping_policy"
    )
    with pytest.raises(ValueError, match="missing current action-execution fields"):
        validate_results(
            missing_execution_field, require_suite=True, require_current_contract=True
        )

    # Both blocks may validate independently, but strict mode rejects their
    # differing declared scenario lineages before a release can use them.
    mismatched_lineage = copy.deepcopy(results)
    mismatched_lineage["action_loop"]["scenarios"] = mismatched_lineage[
        "action_loop"
    ]["scenarios"][1:]
    with pytest.raises(ValueError, match="scenario lineage differ"):
        validate_results(
            mismatched_lineage, require_suite=True, require_current_contract=True
        )

    # A frontier is a projection of the normal operating rows, not a second,
    # independently selectable result table.
    mismatched_frontier = copy.deepcopy(results)
    mismatched_frontier["suite"]["power_cost_frontier"] = mismatched_frontier[
        "suite"
    ]["power_cost_frontier"][1:]
    with pytest.raises(ValueError, match="operating characteristics and power-cost frontier rows differ"):
        validate_results(
            mismatched_frontier, require_suite=True, require_current_contract=True
        )


def test_current_contract_helpers_reject_nonprojectable_real_suite_rows(
    generated_fast_bundle: Path,
) -> None:
    """Exercise strict-only release defenses with a real generated suite.

    The public validator first applies the preserved baseline contract,
    which deliberately rejects several malformed shapes before the stricter
    release projection runs.  These direct checks therefore retain coverage
    for the strict-only diagnostics using actual producer output rather than a
    hand-authored stand-in artifact.
    """
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    suite = results["suite"]
    contract = suite["scenario_contracts"][0]
    operation = suite["operating_characteristics"][0]
    frontier = suite["power_cost_frontier"][0]

    with pytest.raises(ValueError, match="at least one full scenario contract"):
        _current_scenario_contract_index([], "contracts")
    with pytest.raises(ValueError, match="duplicate scenario lineage"):
        _current_scenario_contract_index([contract, copy.deepcopy(contract)], "contracts")
    malformed_semantics = copy.deepcopy(contract)
    malformed_semantics["model"]["prior"][0] = -0.1
    malformed_semantics["model"]["prior"][1] += 0.1
    malformed_semantics["scenario_hash"] = _scenario_contract_digest(
        malformed_semantics
    )
    with pytest.raises(ValueError, match="valid reconstructed scenario contract"):
        _current_scenario_contract_index([malformed_semantics], "contracts")
    unexpected_contract_field = copy.deepcopy(contract)
    unexpected_contract_field["unscoped"] = True
    unexpected_contract_field["scenario_hash"] = _scenario_contract_digest(
        unexpected_contract_field
    )
    with pytest.raises(ValueError, match="exactly the current scenario contract keys"):
        _current_scenario_contract_index([unexpected_contract_field], "contracts")
    with pytest.raises(ValueError, match="scenario_id must be non-empty"):
        _current_row_identity(
            {
                "scenario_id": "",
                "scenario_hash": contract["scenario_hash"],
                "policy": operation["policy"],
            },
            "row",
        )
    with pytest.raises(ValueError, match="policy must be non-empty"):
        _current_row_identity(
            {
                "scenario_id": contract["scenario_id"],
                "scenario_hash": contract["scenario_hash"],
                "policy": "",
            },
            "row",
        )

    no_frontier = copy.deepcopy(suite)
    no_frontier.pop("power_cost_frontier")
    with pytest.raises(ValueError, match="requires power_cost_frontier"):
        _validate_current_frontier_projection(no_frontier)
    empty_frontier = copy.deepcopy(suite)
    empty_frontier["power_cost_frontier"] = []
    with pytest.raises(ValueError, match="requires non-empty power_cost_frontier"):
        _validate_current_frontier_projection(empty_frontier)

    duplicate_operations = copy.deepcopy(suite)
    duplicate_operations["operating_characteristics"] = [
        copy.deepcopy(operation),
        copy.deepcopy(operation),
    ]
    with pytest.raises(ValueError, match="duplicate scenario/policy rows"):
        _validate_current_frontier_projection(duplicate_operations)
    duplicate_frontier = copy.deepcopy(suite)
    duplicate_frontier["power_cost_frontier"] = [
        copy.deepcopy(frontier),
        copy.deepcopy(frontier),
    ]
    with pytest.raises(ValueError, match="duplicate scenario/policy rows"):
        _validate_current_frontier_projection(duplicate_frontier)

    missing_summary_field = copy.deepcopy(suite)
    missing_summary_field["operating_characteristics"][0]["summary"].pop("power")
    with pytest.raises(ValueError, match="summary is missing 'power'"):
        _validate_current_frontier_projection(missing_summary_field)
    mismatched_summary_projection = copy.deepcopy(suite)
    mismatched_summary_projection["power_cost_frontier"][0]["power"] += 1.0
    with pytest.raises(ValueError, match="does not match its operating-characteristic 'power'"):
        _validate_current_frontier_projection(mismatched_summary_projection)
    mismatched_operation_reps = copy.deepcopy(suite)
    mismatched_operation_reps["operating_characteristics"][0]["n_reps"] += 1
    with pytest.raises(ValueError, match="n_reps does not match its summary"):
        _validate_current_frontier_projection(mismatched_operation_reps)
    mismatched_operation_validity = copy.deepcopy(suite)
    mismatched_operation_validity["operating_characteristics"][0]["validity"] = "other"
    with pytest.raises(ValueError, match="validity does not match its summary"):
        _validate_current_frontier_projection(mismatched_operation_validity)

    wrong_version = copy.deepcopy(suite)
    wrong_version["contract_version"] = "2.0"
    with pytest.raises(ValueError, match="contract_version"):
        _validate_current_suite_contract(wrong_version)
    missing_contracts = copy.deepcopy(suite)
    missing_contracts.pop("scenario_contracts")
    with pytest.raises(ValueError, match="requires scenario_contracts"):
        _validate_current_suite_contract(missing_contracts)
    malformed_declaration = copy.deepcopy(suite)
    malformed_declaration["scenarios"] = [""]
    with pytest.raises(ValueError, match="non-empty scenario identifiers"):
        _validate_current_suite_contract(malformed_declaration)
    mismatched_declaration = copy.deepcopy(suite)
    mismatched_declaration["scenarios"] = mismatched_declaration["scenarios"][:-1]
    with pytest.raises(ValueError, match="do not match scenario-contract lineage"):
        _validate_current_suite_contract(mismatched_declaration)

    undeclared_split = copy.deepcopy(results)
    split_contract = copy.deepcopy(
        undeclared_split["suite"]["split_confirmation_scenario"]
    )
    # The static split channel may not be smuggled into the operating grid,
    # even if all hashes and summary lineage remain internally consistent.
    undeclared_split["suite"]["scenario_contracts"].append(split_contract)
    undeclared_split["suite"]["scenarios"].append(split_contract["scenario_id"])
    with pytest.raises(ValueError, match="must be outside the operating grid"):
        validate_results(
            undeclared_split,
            require_suite=True,
            require_current_contract=True,
        )

    # Current online-FDR figure inputs are reconstructable traces, not labels
    # attached to a hand-selected accounting path.
    missing_pvalue_trace = copy.deepcopy(results)
    missing_pvalue_trace["online_fdr"]["methods"]["lord_plus_plus"]["trace"].pop(
        "pvalues"
    )
    with pytest.raises(ValueError, match="pvalues"):
        validate_results(
            missing_pvalue_trace, require_suite=True, require_current_contract=True
        )
    relabelled_e_lond_accounting = copy.deepcopy(results)
    relabelled_e_lond_accounting["online_fdr"]["methods"]["e_lond"]["trace"][
        "accounting_kind"
    ] = "procedural_alpha_wealth"
    with pytest.raises(ValueError, match="cumulative_nominal_level_allocation"):
        validate_results(
            relabelled_e_lond_accounting,
            require_suite=True,
            require_current_contract=True,
        )

    # The dedicated split lane must retain the formal label, its declared
    # assumptions, and the alpha embedded in its separately hashed scenario.
    weakened_split_label = copy.deepcopy(results)
    weakened_split_label["suite"]["split_stream_confirmation"][
        "validity"
    ] = "diagnostic"
    with pytest.raises(ValueError, match="formal conditional validity label"):
        validate_results(
            weakened_split_label, require_suite=True, require_current_contract=True
        )
    weakened_split_assumptions = copy.deepcopy(results)
    weakened_split_assumptions["suite"]["split_stream_confirmation"][
        "assumptions"
    ] = ["fixed_truth"]
    with pytest.raises(ValueError, match="assumptions do not match its scenario"):
        validate_results(
            weakened_split_assumptions,
            require_suite=True,
            require_current_contract=True,
        )

    mismatched_representative = copy.deepcopy(results)
    representative = mismatched_representative["suite"]["split_stream_confirmation"][
        "representative"
    ]
    representative["scenario_hash"] = "b" * 64
    representative["confirmation_stream_hash"] = _payload_digest(
        {
            "scenario_id": representative["scenario_id"],
            "scenario_hash": representative["scenario_hash"],
            "target": representative["selection"]["selected_target"],
            "seed": representative["confirmation_seed"],
            "observations": representative["confirmation_observations"],
            "selected_target_truth": representative["selected_target_truth"],
            "confirmation_p_value": representative["confirmation_p_value"],
            "rejected": representative["rejected"],
            "validity": representative["validity"],
            "validity_assumptions": representative["validity_assumptions"],
        }
    )
    with pytest.raises(ValueError, match="representative does not match its summary scenario lineage"):
        validate_results(
            mismatched_representative,
            require_suite=True,
            require_current_contract=True,
        )

    mismatched_selection = copy.deepcopy(results)
    representative = mismatched_selection["suite"]["split_stream_confirmation"][
        "representative"
    ]
    selection = representative["selection"]
    selection["scenario_hash"] = "c" * 64
    selection["stream_hash"] = _payload_digest(
        {
            key: selection[key]
            for key in (
                "scenario_id",
                "scenario_hash",
                "policy",
                "seed",
                "events",
                "selected_target",
                "selected_target_evidence",
                "visible_only",
            )
        }
    )
    representative["selection_stream_hash"] = selection["stream_hash"]
    with pytest.raises(ValueError, match="representative selection does not match its summary scenario lineage"):
        validate_results(
            mismatched_selection,
            require_suite=True,
            require_current_contract=True,
        )


def test_current_contract_recomputes_online_and_split_semantics(
    generated_fast_bundle: Path,
) -> None:
    """Reject relabelled current traces even when their enclosing JSON is valid.

    The mutations start from a real seeded producer result.  They specifically
    cover figure-facing online accounting and the separate static
    split-confirmation lane, whose meanings cannot be established by a shape
    check alone.
    """
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )

    def require_rejection(value: dict[str, Any], pattern: str) -> None:
        with pytest.raises(ValueError, match=pattern):
            validate_results(value, require_suite=True, require_current_contract=True)

    def require_current_suite_rejection(value: dict[str, Any], pattern: str) -> None:
        """Reach a strict semantic branch after preserving baseline shape checks."""
        with pytest.raises(ValueError, match=pattern):
            _validate_current_suite_contract(value["suite"])

    bad_null = copy.deepcopy(results)
    bad_null["online_fdr"]["null"] = False
    require_rejection(bad_null, "null-stream diagnostic")

    missing_method = copy.deepcopy(results)
    missing_method["online_fdr"]["methods"].pop("e_lond")
    require_rejection(missing_method, "must contain exactly")

    bad_uncertainty = copy.deepcopy(results)
    bad_uncertainty["online_fdr"]["uncertainty"] = ""
    require_rejection(bad_uncertainty, "uncertainty must be non-empty")

    bad_mean = copy.deepcopy(results)
    bad_mean["online_fdr"]["methods"]["lord_plus_plus"]["mean_rejections"] = (
        bad_mean["online_fdr"]["horizon"] + 1
    )
    require_rejection(bad_mean, "mean_rejections")

    bad_rate = copy.deepcopy(results)
    bad_rate["online_fdr"]["methods"]["lord_plus_plus"]["any_rejection_rate"] = 0.123
    require_rejection(bad_rate, "recorded replication proportion")

    bad_interval = copy.deepcopy(results)
    bad_interval["online_fdr"]["methods"]["lord_plus_plus"][
        "any_rejection_interval_wilson"
    ] = [0.0, 1.01]
    require_rejection(bad_interval, r"must lie in \[0, 1\]")

    mismatched_interval = copy.deepcopy(results)
    mismatched_interval["online_fdr"]["methods"]["lord_plus_plus"][
        "any_rejection_interval_wilson"
    ] = [0.0, 0.0]
    require_rejection(mismatched_interval, "does not match its recorded rate")

    bad_p_accounting = copy.deepcopy(results)
    bad_p_accounting["online_fdr"]["methods"]["saffron"]["trace"][
        "accounting_kind"
    ] = "cumulative_nominal_level_allocation"
    require_rejection(bad_p_accounting, "procedural_alpha_wealth")

    bad_p_alpha = copy.deepcopy(results)
    bad_p_alpha["online_fdr"]["methods"]["saffron"]["trace"]["alpha"] = 0.11
    require_rejection(bad_p_alpha, "trace.alpha does not match")

    bad_transform = copy.deepcopy(results)
    bad_transform["online_fdr"]["methods"]["e_lond"]["trace"][
        "e_value_null_transform"
    ] = "unbounded"
    require_rejection(bad_transform, "bounded null e-value transform")

    bad_support = copy.deepcopy(results)
    bad_support["online_fdr"]["methods"]["e_lond"]["null_e_value_support"] += 1
    require_rejection(bad_support, "support disagrees")

    bad_e_alpha = copy.deepcopy(results)
    bad_e_alpha["online_fdr"]["methods"]["e_lond"]["trace"]["alpha"] = 0.11
    require_rejection(bad_e_alpha, "trace.alpha does not match")

    bad_evalues = copy.deepcopy(results)
    bad_evalues["online_fdr"]["methods"]["e_lond"]["trace"]["evalues"][0] = 1.0
    require_rejection(bad_evalues, "bounded transform support")

    bad_rejection_count = copy.deepcopy(results)
    bad_rejection_count["online_fdr"]["methods"]["e_lond"]["trace"][
        "num_rejections"
    ] += 1
    require_rejection(bad_rejection_count, "num_rejections")

    bad_trace_role = copy.deepcopy(results)
    bad_trace_role["online_fdr"]["methods"]["e_lond"]["trace"][
        "trace_role"
    ] = "unscoped"
    require_rejection(bad_trace_role, "trace_role")

    bad_split_contract = copy.deepcopy(results)
    split_contract = bad_split_contract["suite"]["split_confirmation_scenario"]
    split_contract["setting"]["formal_validity"] = "unscoped"
    split_contract["scenario_hash"] = _scenario_contract_digest(split_contract)
    require_current_suite_rejection(
        bad_split_contract, "conditional split-confirmation validity"
    )

    bad_split_assumptions = copy.deepcopy(results)
    split_contract = bad_split_assumptions["suite"]["split_confirmation_scenario"]
    split_contract["setting"]["evidence_assumptions"] = ["fixed_truth"]
    split_contract["scenario_hash"] = _scenario_contract_digest(split_contract)
    require_current_suite_rejection(
        bad_split_assumptions, "scenario assumptions do not match"
    )

    bad_split_metadata = copy.deepcopy(results)
    split_contract = bad_split_metadata["suite"]["split_confirmation_scenario"]
    split_contract["metadata"]["split_confirmation_channel"] = False
    split_contract["scenario_hash"] = _scenario_contract_digest(split_contract)
    require_current_suite_rejection(bad_split_metadata, "dedicated channel")

    def rehash_confirmation(representative: dict[str, Any]) -> None:
        representative["confirmation_stream_hash"] = _payload_digest(
            {
                "scenario_id": representative["scenario_id"],
                "scenario_hash": representative["scenario_hash"],
                "target": representative["selection"]["selected_target"],
                "seed": representative["confirmation_seed"],
                "observations": representative["confirmation_observations"],
                "selected_target_truth": representative["selected_target_truth"],
                "confirmation_p_value": representative["confirmation_p_value"],
                "rejected": representative["rejected"],
                "validity": representative["validity"],
                "validity_assumptions": representative["validity_assumptions"],
            }
        )

    bad_representative_label = copy.deepcopy(results)
    representative = bad_representative_label["suite"]["split_stream_confirmation"][
        "representative"
    ]
    representative["validity"] = "diagnostic"
    rehash_confirmation(representative)
    require_rejection(bad_representative_label, "representative must carry")

    bad_representative_assumptions = copy.deepcopy(results)
    representative = bad_representative_assumptions["suite"][
        "split_stream_confirmation"
    ]["representative"]
    representative["validity_assumptions"] = ["fixed_truth"]
    rehash_confirmation(representative)
    require_rejection(bad_representative_assumptions, "representative assumptions")

    bad_declared_alpha = copy.deepcopy(results)
    representative = bad_declared_alpha["suite"]["split_stream_confirmation"][
        "representative"
    ]
    representative["rejected"] = not representative["rejected"]
    rehash_confirmation(representative)
    require_rejection(bad_declared_alpha, "declared scenario alpha")

    bad_visible_filtration = copy.deepcopy(results)
    representative = bad_visible_filtration["suite"]["split_stream_confirmation"][
        "representative"
    ]
    selection = representative["selection"]
    selection["visible_only"] = False
    selection["stream_hash"] = _payload_digest(
        {
            key: selection[key]
            for key in (
                "scenario_id",
                "scenario_hash",
                "policy",
                "seed",
                "events",
                "selected_target",
                "selected_target_evidence",
                "visible_only",
            )
        }
    )
    representative["selection_stream_hash"] = selection["stream_hash"]
    require_rejection(bad_visible_filtration, "visible history only")


def test_dependence_modes_cover_mean_and_validation_contracts() -> None:
    rng = np.random.default_rng(12)
    assert np.allclose(sample_gaussian(rng, 3, DependenceConfig(kind="independent"), np.ones(3)).shape, (3,))
    assert sample_gaussian(rng, 3, DependenceConfig(kind="block", rho=0.2, block_size=2)).shape == (3,)
    assert sample_gaussian(rng, 3, DependenceConfig(kind="factor", loadings=np.array([0.1, -0.2, 0.3])), np.zeros(3)).shape == (3,)
    with pytest.raises(ValueError):
        DependenceConfig().validate(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DependenceConfig(rho=np.nan).validate(2)
    with pytest.raises(ValueError):
        DependenceConfig(kind="block", block_size=True).validate(2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DependenceConfig(kind="block", rho=0.1, block_size=1).validate(2)
    with pytest.raises(ValueError):
        DependenceConfig(kind="factor", loadings=np.array([0.1, 0.2])).validate(3)
    with pytest.raises(ValueError):
        DependenceConfig(kind="factor", loadings=np.array([np.inf, 0.2, 0.3])).validate(3)
    with pytest.raises(ValueError):
        DependenceConfig(kind="equicorrelated", rho=-1.1).validate(2)
    with pytest.raises(ValueError):
        sample_gaussian(np.random.default_rng(1), 3, DependenceConfig(), np.zeros(2))
    matrix = correlation_matrix(4, DependenceConfig(kind="block", rho=0.2, block_size=2))
    assert np.allclose(np.diag(matrix), 1.0)


def test_uncertainty_and_agent_policy_edges_are_explicit() -> None:
    with pytest.raises(ValueError):
        hoeffding_radius(1, confidence=0.0)
    with pytest.raises(ValueError):
        required_reps_for_half_width(0.1, confidence=0.0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
    with pytest.raises(ValueError):
        clopper_pearson_interval(-1, 10)
    with pytest.raises(ValueError):
        clopper_pearson_interval(0, 0)
    with pytest.raises(ValueError):
        empirical_bernstein_radius([0.1, 1.2])
    with pytest.raises(ValueError):
        paired_difference_summary([0.1, np.nan], [0.2, 0.3])
    with pytest.raises(ValueError):
        paired_difference_summary([0.1], [1.1])
    one = summarize_bounded([True])
    assert one["variance"] == 0.0

    config = AITestConfig(reliability=0.5, threshold=1.0, horizon=3)
    trace = run_agent_policy([0, 0, 0], config, "posterior_threshold")
    assert trace.stopping_time == 3
    with pytest.raises(ValueError):
        run_agent_policy([], config)
    with pytest.raises(ValueError):
        run_agent_policy([0, 1, 0, 1], config)
    with pytest.raises(ValueError):
        simulate_agent_policy(config, "fixed_horizon", 1, reliability=1.0)
    with pytest.raises(ValueError):
        simulate_agent_policy(config, "fixed_horizon", True)  # type: ignore[arg-type]
    summary = simulate_agent_policy(config, "posterior_threshold", 2, seed=9)
    assert summary["n_reps"] == 2


def test_dataclass_contracts_cover_fwer_fallback_and_config_modes() -> None:
    simulation = SimulationResult(
        "bonferroni",
        2,
        0.05,
        0.2,
        0.5,
        1.0,
        np.array([0.0, 0.1]),
        np.array([0.2, 0.2]),
    )
    assert simulation.mc_stderr("fwer") >= 0.0
    with pytest.raises(ValueError):
        SimulationResult(
            "bonferroni",
            2,
            0.1,
            0.2,
            0.3,
            1.0,
            np.zeros(2),
            np.zeros(2),
            np.array([0.0]),
        )
    with pytest.raises(ValueError):
        CorrectionResult("bonferroni", np.array([0.1]), np.array([True]), 0.05, np.nan)
    assert ErrorSummary(2, 1, 1, 1, 0, 0.0, 1.0, False).num_null == 1
    for config in (
        TwoGroupsConfig(2, 0.5, 1.0, dependence_kind="independent"),
        TwoGroupsConfig(2, 0.5, 1.0, dependence_kind="equicorrelated", rho=-0.5),
        TwoGroupsConfig(4, 0.5, 1.0, dependence_kind="block", rho=0.2, block_size=2),
    ):
        assert config.m0 + config.m1 == config.m
    with pytest.raises(ValueError):
        TwoGroupsConfig(2, 0.5, 1.0, dependence_kind="independent", rho=-0.1)
    with pytest.raises(ValueError):
        TwoGroupsConfig(2, 0.5, 1.0, dependence_kind="block", rho=0.1, block_size=1)
    with pytest.raises(ValueError):
        TwoGroupsConfig(2, 0.5, 1.0, sided=3)
