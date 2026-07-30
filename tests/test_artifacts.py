"""Artifact, schema, and offline benchmark contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml
import pytest

from active_inference_power.benchmarks import BenchmarkSpec, load_agent_trace, load_csv_benchmark, load_normalized_golub_fixture
from active_inference_power.schema import (
    validate_action_loop_result,
    validate_benchmark_manifest,
    validate_checkpoint,
    validate_claim_ledger,
    validate_figure_registry,
    validate_manifest,
    validate_results,
    validate_run_log,
    validate_suite_result,
)
from scripts.artifacts import artifact_record, atomic_write_json
from scripts.checkpointing import load_checkpoint, write_partial_checkpoint
from scripts.run_ledger import RunLedger


def test_atomic_json_and_manifest_records(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "value.json"
    atomic_write_json(path, {"value": 3})
    assert json.loads(path.read_text()) == {"value": 3}
    record = artifact_record(path, tmp_path, "test")
    assert record["bytes"] > 0
    assert record["root"] == "project"
    manifest = {"schema_version": "2.0", "run_id": "r", "complete": True, "config_sha256": "a" * 64, "artifacts": {"value": {**record, "sha256": "b" * 64}}}
    validate_manifest(manifest)


def test_atomic_json_rejects_nonfinite_values_without_replacing_a_receipt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    path.write_text('{"retained": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Out of range float values"):
        atomic_write_json(path, {"value": float("nan")})
    assert path.read_text(encoding="utf-8") == '{"retained": true}\n'


def test_generated_figure_registry_is_schema_valid(
    generated_fast_bundle: Path,
) -> None:
    registry = json.loads(
        (generated_fast_bundle / "figures" / "figure_registry.json").read_text(
            encoding="utf-8"
        )
    )
    validate_figure_registry(registry)


def test_partial_checkpoint_is_atomic_and_schema_validated(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints" / "stage.json"
    dependencies = {key: ["a" * 64] for key in ("scenario_hashes", "model_hashes", "process_hashes", "setting_hashes")}
    write_partial_checkpoint(
        path,
        run_id="run",
        config_sha256="b" * 64,
        claim_ledger_sha256="c" * 64,
        profile="fast",
        seed=4,
        stage_seed=5,
        workers=2,
        stage="conditional_power",
        total_replications=8,
        completed_replications=4,
        chunk_size=4,
        metrics={"rejections": 2, "cost_sum": 7.5},
        dependency_hashes=dependencies,
    )
    checkpoint = load_checkpoint(path)
    assert checkpoint["partial_metric_accumulators"]["status"] == "partial"
    assert checkpoint["replication_cursor"] == {"completed": 4, "next": 4}
    assert checkpoint["total_replications"] == 8
    with pytest.raises(ValueError):
        write_partial_checkpoint(
            tmp_path / "bad.json", run_id="run", config_sha256="b" * 64,
            claim_ledger_sha256="c" * 64, profile="fast", seed=4, stage_seed=5,
            workers=2, stage="conditional_power", total_replications=2,
            completed_replications=4, chunk_size=4, metrics={}, dependency_hashes=dependencies,
        )


def test_offline_benchmark_fixture_and_manifest() -> None:
    root = Path(__file__).resolve().parent.parent
    manifest = yaml.safe_load((root / "data" / "benchmarks" / "manifest.yaml").read_text())
    validate_benchmark_manifest(manifest)
    record = cast(dict[str, Any], manifest["benchmarks"][1])
    spec = BenchmarkSpec(record["id"], record["source"], record["license"], record["sha256"])
    dataset = load_agent_trace(root / "data" / "benchmarks" / record["filename"], spec)
    assert dataset.values.shape[1] == 2
    golub_record = cast(dict[str, Any], manifest["benchmarks"][2])
    golub_spec = BenchmarkSpec(
        golub_record["id"], golub_record["source"], golub_record["license"],
        golub_record["sha256"], golub_record["claim_role"],
        golub_record["doi"], golub_record["retrieval_date"], golub_record["schema"],
        golub_record["preprocessing"], golub_record["format"],
    )
    fixture = load_normalized_golub_fixture(root / "data" / "benchmarks" / golub_record["filename"], golub_spec)
    assert cast(dict[str, Any], fixture.metadata)["normalized"] is True
    assert fixture.values.shape == (16, 4)


def test_claim_ledger_validation_rejects_unknown_and_unscoped_rows() -> None:
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": []})
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [{"claim_id": "x", "kind": "unknown", "source": "x", "source_tier": "project_source", "freshness": "active", "artifact_path": "x"}]})
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [{"claim_id": "x", "kind": "validity_boundary", "source": "x", "source_tier": "project_source", "freshness": "active", "artifact_path": "x"}]})


def test_claim_ledger_validation_covers_path_and_metadata_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    artifact = tmp_path / "artifact.json"
    source.write_text("line = 1\n", encoding="utf-8")
    artifact.write_text("{}\n", encoding="utf-8")
    base = {
        "claim_id": "x",
        "kind": "configuration",
        "source": "source.py:line",
        "source_tier": "project_source",
        "freshness": "active",
        "artifact_path": "artifact.json",
    }
    validate_claim_ledger({"claims": [base]}, project_root=tmp_path, require_artifacts=True)
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [[]]})
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [{}]})
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [dict(base, claim_id=None)]})
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [dict(base, source_tier="unknown")]})
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [dict(base, source="")]})
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [dict(base, freshness="unknown")]})
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [dict(base, artifact_path="../artifact.json")]})
    citation = dict(base, kind="citation")
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [citation]})
    bearing = dict(base, kind="validity_boundary")
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [bearing]})
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [dict(bearing, evidence_class="diagnostic", non_claim="")]})
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [dict(base, source="../source.py:line")]}, project_root=tmp_path)
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [dict(base, source="missing.py:line")]}, project_root=tmp_path)
    with pytest.raises(ValueError, match="locator"):
        validate_claim_ledger({"claims": [dict(base, source="source.py:missing")]}, project_root=tmp_path)
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [dict(base, artifact_path="missing.json")]}, project_root=tmp_path, require_artifacts=True)


def test_claim_source_formats_and_locators_fail_closed(tmp_path: Path) -> None:
    """Python, YAML, and BibTeX provenance locators are checked as symbols."""
    (tmp_path / "source.py").write_text("value = 1\ndef method():\n    return value\n", encoding="utf-8")
    (tmp_path / "source.yaml").write_text("experiment:\n  alpha: 0.05\n", encoding="utf-8")
    (tmp_path / "source.bib").write_text("@article{valid-key,\n  title = {Example}\n}\n", encoding="utf-8")
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")

    def claim(source: str) -> dict[str, str]:
        return {
            "claim_id": source,
            "kind": "configuration",
            "source": source,
            "source_tier": "project_source",
            "freshness": "active",
            "artifact_path": "artifact.json",
        }

    validate_claim_ledger(
        {"claims": [claim("source.py:value,method"), claim("source.yaml:experiment.alpha"), claim("source.bib:valid-key")]},
        project_root=tmp_path,
        require_artifacts=True,
    )
    for source in ("source.py:", "source.py:missing", "source.yaml:experiment.missing", "source.bib:missing-key"):
        with pytest.raises(ValueError):
            validate_claim_ledger({"claims": [claim(source)]}, project_root=tmp_path)


def test_schema_release_lineage_and_terminal_contract_boundaries() -> None:
    """Exercise the strict release-only branches without relying on output files."""
    from active_inference_power.formalism import formalism_registry

    result: dict[str, Any] = {
        "schema_version": "2.0",
        "run_id": "run",
        "meta": {"config_sha256": "a" * 64, "seed": 1, "n_reps": 1},
        **{key: {} for key in ("experiment", "certificates", "method_comparison", "paired_comparisons", "active_inference", "sequential", "online_fdr", "agent_sampling", "benchmarks", "null_pvalue_calibration", "sample_size_accounting")},
        "power_curve": [],
        "dependence_sweep": [],
        "formalism": formalism_registry().to_mapping(),
    }
    operation = {"scenario_id": "s", "scenario_hash": "b" * 64, "policy": "fixed_horizon", "n_reps": 1, "summary": {}, "validity": "diagnostic"}
    result["suite"] = {"suite_name": "suite", "scenarios": ["s"], "policies": ["fixed_horizon"], "operating_characteristics": [operation], "online_fdr": {}, "claim_boundary": "conditional"}
    result["action_loop"] = {"scenarios": [{"scenario_id": "s", "model": {}, "process": {}, "setting": {}}]}
    with pytest.raises(ValueError, match="claim_ledger_sha256"):
        validate_results(result, require_suite=True)
    result["meta"] = {**result["meta"], "claim_ledger_sha256": "c" * 64}
    validate_results(result, require_suite=True)

    with pytest.raises(ValueError, match="finished_at_utc"):
        validate_run_log({"schema_version": "2.0", "run_id": "r", "status": "running", "events": [], "config_sha256": "a" * 64, "finished_at_utc": "2026-07-19T00:00:00+00:00"})
    with pytest.raises(ValueError, match="error.type"):
        validate_run_log({"schema_version": "2.0", "run_id": "r", "status": "failed", "events": [], "config_sha256": "a" * 64, "finished_at_utc": "2026-07-19T00:00:00+00:00", "error": {"type": "", "message": "m", "traceback": "t"}})
    with pytest.raises(ValueError, match="error"):
        validate_run_log({"schema_version": "2.0", "run_id": "r", "status": "passed", "events": [], "config_sha256": "a" * 64, "finished_at_utc": "2026-07-19T00:00:00+00:00", "error": {"type": "x", "message": "m", "traceback": "t"}})
    progress = {"completed_stages": 2, "total_stages": 1, "completed_replications": 0, "estimated_remaining_seconds": None}
    with pytest.raises(ValueError, match="exceeds"):
        validate_run_log({"schema_version": "2.0", "run_id": "r", "status": "running", "events": [], "config_sha256": "a" * 64, "progress": progress})
    with pytest.raises(ValueError, match="last_completed_stage"):
        validate_run_log({"schema_version": "2.0", "run_id": "r", "status": "running", "events": [], "config_sha256": "a" * 64, "progress": {"completed_stages": 0, "total_stages": 1, "completed_replications": 0, "estimated_remaining_seconds": None, "last_completed_stage": ""}})

    artifact = {"path": "x", "sha256": "a" * 64, "bytes": 1, "role": "data"}
    with pytest.raises(ValueError, match="must not be empty"):
        validate_manifest({"schema_version": "2.0", "run_id": "r", "complete": True, "config_sha256": "a" * 64, "artifacts": {}})
    with pytest.raises(ValueError, match="names"):
        validate_manifest({"schema_version": "2.0", "run_id": "r", "complete": True, "config_sha256": "a" * 64, "artifacts": {"": artifact}})

    with pytest.raises(ValueError, match="profile"):
        validate_checkpoint({"schema_version": "2.0", "run_id": "r", "config_sha256": "a" * 64, "claim_ledger_sha256": "b" * 64, "profile": "", "seed": 1, "stage_seed": 2, "seed_schedule": {"kind": "numpy.SeedSequence", "master_seed": 1, "stage": "x", "stage_seed": 2, "replication_order": "stable ascending child-stream order"}, "workers": 1, "stage": "x", "completed_replications": 1, "partial_metric_accumulators": {"status": "stage_complete", "completed_replications": 1}, "dependency_hashes": {"scenario_hashes": [], "model_hashes": [], "process_hashes": [], "setting_hashes": []}, "value": {}})


def test_schema_diagnostics_and_benchmark_date_boundaries() -> None:
    operation = {"scenario_id": "s", "scenario_hash": "b" * 64, "policy": "fixed_horizon", "n_reps": 1, "summary": {}, "validity": "diagnostic"}
    diagnostic: dict[str, Any] = {"scenario_id": "null", "scenario_hash": "c" * 64, "policy": "adaptive_target", "n_reps": 1, "pooled": {"n_selected": 0, "max_ecdf_excess": None, "dkw_radius": None, "passes_dkw_diagnostic": False}, "validity": "selection_failure_contrast"}
    validate_suite_result({"suite_name": "suite", "scenarios": ["s"], "policies": ["fixed_horizon"], "operating_characteristics": [operation], "online_fdr": {}, "claim_boundary": "conditional", "future_evidence_selection_failure": diagnostic})
    with pytest.raises(ValueError, match="passes_dkw"):
        validate_suite_result({"suite_name": "suite", "scenarios": ["s"], "policies": ["fixed_horizon"], "operating_characteristics": [operation], "online_fdr": {}, "claim_boundary": "conditional", "future_evidence_selection_failure": {**diagnostic, "pooled": {**diagnostic["pooled"], "passes_dkw_diagnostic": 0}}})
    with pytest.raises(ValueError, match="retrieval_date"):
        validate_benchmark_manifest({"schema_version": "2.0", "benchmarks": [{"id": "b", "source": "local", "filename": "b.csv", "license": "MIT", "sha256": "a" * 64, "retrieval_date": 1, "schema": "csv", "preprocessing": "none", "claim_role": "stress"}]})


def test_action_loop_result_field_contract_boundaries() -> None:
    valid = {"scenario_id": "s", "scenario_hash": "b" * 64, "policy": "fixed_horizon", "n_reps": 1, "summary": {}, "validity": "diagnostic"}
    validate_action_loop_result(valid)
    with pytest.raises(ValueError, match="scenario_id"):
        validate_action_loop_result({**valid, "scenario_id": ""})
    with pytest.raises(ValueError, match="policy"):
        validate_action_loop_result({**valid, "policy": ""})


def test_run_log_schema_contract(generated_fast_bundle: Path) -> None:
    log = json.loads(
        (generated_fast_bundle / "run_log.json").read_text(encoding="utf-8")
    )
    validate_run_log(log)


def test_schema_rejects_invalid_shapes_and_csv_contracts(tmp_path: Path) -> None:
    import hashlib

    from active_inference_power.schema import validate_results

    with pytest.raises(ValueError):
        validate_results({})
    with pytest.raises(ValueError):
        validate_results({"schema_version": "1.0"})
    path = tmp_path / "values.csv"
    path.write_text("x,y\n1,2\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = BenchmarkSpec("numeric", "local", "MIT", digest)
    assert load_csv_benchmark(path, spec).values.shape == (1, 2)
    with pytest.raises(ValueError):
        load_csv_benchmark(path, BenchmarkSpec("numeric", "local", "MIT", "0" * 64))


def test_schema_contracts_cover_failure_states() -> None:
    from active_inference_power.schema import validate_results

    with pytest.raises(ValueError):
        validate_results([])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_results({"schema_version": "2.0"})
    with pytest.raises(ValueError):
        validate_manifest([])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_manifest({"schema_version": "1.0"})
    with pytest.raises(ValueError):
        validate_manifest({"schema_version": "2.0", "run_id": "r", "complete": "yes", "config_sha256": "a" * 64, "artifacts": {}})
    with pytest.raises(ValueError):
        validate_manifest({"schema_version": "2.0", "run_id": "r", "complete": True, "config_sha256": "a" * 64, "artifacts": {"x": []}})  # type: ignore[list-item]
    with pytest.raises(ValueError):
        validate_manifest({"schema_version": "2.0", "run_id": "r", "complete": True, "config_sha256": "a" * 64, "artifacts": {"x": {"path": "x", "sha256": "0", "bytes": 1, "role": "x"}}})
    with pytest.raises(ValueError):
        validate_run_log([])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_run_log({"schema_version": "1.0"})
    with pytest.raises(ValueError):
        validate_run_log({"schema_version": "2.0", "run_id": "r", "status": "unknown", "events": [], "config_sha256": "a" * 64})
    with pytest.raises(ValueError):
        validate_run_log({"schema_version": "2.0", "run_id": "r", "status": "running", "events": {}, "config_sha256": "a" * 64})
    with pytest.raises(ValueError, match="finished_at_utc"):
        validate_run_log({"schema_version": "2.0", "run_id": "r", "status": "passed", "events": [], "config_sha256": "a" * 64})
    with pytest.raises(ValueError):
        validate_benchmark_manifest({"schema_version": "2.0", "benchmarks": {}})
    with pytest.raises(ValueError):
        validate_benchmark_manifest({"schema_version": "2.0", "benchmarks": [[]]})  # type: ignore[list-item]
    with pytest.raises(ValueError):
        validate_benchmark_manifest({"schema_version": "2.0", "benchmarks": [{"id": "x"}]})
    with pytest.raises(ValueError):
        validate_benchmark_manifest({"schema_version": "2.0", "benchmarks": [{"id": "x", "source": "x", "filename": "x", "license": "x", "sha256": "0", "retrieval_date": "x", "schema": "x", "preprocessing": "x", "claim_role": "x"}]})
    valid = {"schema_version": "2.0", "run_id": "r", "config_sha256": "a" * 64, "claim_ledger_sha256": "b" * 64, "profile": "fast", "seed": 0, "stage_seed": 2, "seed_schedule": {"kind": "numpy.SeedSequence", "master_seed": 0, "stage": "x", "stage_seed": 2, "replication_order": "stable ascending child-stream order"}, "workers": 1, "stage": "x", "completed_replications": 1, "partial_metric_accumulators": {"status": "stage_complete", "completed_replications": 1}, "dependency_hashes": {"scenario_hashes": [], "model_hashes": [], "process_hashes": [], "setting_hashes": []}, "value": {}}
    validate_checkpoint(valid)
    with pytest.raises(ValueError):
        validate_checkpoint({**valid, "schema_version": "1.0"})
    with pytest.raises(ValueError):
        validate_checkpoint({**valid, "stage": ""})
    with pytest.raises(ValueError):
        validate_checkpoint({**valid, "completed_replications": 0})


def test_offline_loaders_reject_missing_and_malformed_fixtures(tmp_path: Path) -> None:
    spec = BenchmarkSpec("x", "local", "MIT", "0" * 64)
    with pytest.raises(FileNotFoundError):
        load_csv_benchmark(tmp_path / "missing.csv", spec)
    with pytest.raises(FileNotFoundError):
        load_agent_trace(tmp_path / "missing.csv", spec)
    malformed = tmp_path / "malformed.csv"
    malformed.write_text("x\n1\n", encoding="utf-8")
    digest = __import__("hashlib").sha256(malformed.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_agent_trace(malformed, BenchmarkSpec("x", "local", "MIT", digest))


def test_run_ledger_is_deterministic_and_records_interruption(tmp_path: Path) -> None:
    first = RunLedger(tmp_path / "first.json", config_sha256="a" * 64, profile="fast", seed=3, workers=2)
    second = RunLedger(tmp_path / "second.json", config_sha256="a" * 64, profile="fast", seed=3, workers=4)
    assert first.run_id == second.run_id
    first.fail(KeyboardInterrupt())
    assert json.loads((tmp_path / "first.json").read_text())["status"] == "interrupted"
