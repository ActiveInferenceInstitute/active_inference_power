"""Real-file tests for the offline benchmark provenance preflight."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pytest
import yaml

from active_inference_power.benchmark_preflight import (
    _records_by_id,
    _required_text,
    _safe_file,
    _trajectory_reports,
    _validate_golub_lineage,
    preflight_offline_benchmark_fixtures,
)
from active_inference_power.benchmarks import BenchmarkDataset


_ROOT = Path(__file__).resolve().parent.parent
_BENCHMARK_DIRECTORY = _ROOT / "data" / "benchmarks"


def _manifest() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        yaml.safe_load((_BENCHMARK_DIRECTORY / "manifest.yaml").read_text(encoding="utf-8")),
    )


def _records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record["id"]): cast(dict[str, Any], record)
        for record in cast(list[dict[str, Any]], manifest["benchmarks"])
    }


def _stage_fixture_directory(tmp_path: Path, trace_text: str | None = None) -> tuple[Path, dict[str, Any]]:
    """Create actual local copies while leaving the source fixture untouched."""
    fixture_directory = tmp_path / "benchmarks"
    fixture_directory.mkdir(parents=True)
    fixture_path = fixture_directory / "golub_normalized_fixture.csv"
    fixture_path.write_bytes((_BENCHMARK_DIRECTORY / fixture_path.name).read_bytes())
    trace_path = fixture_directory / "trajectory.csv"
    trace_path.write_text(
        trace_text
        or "subject,time,action,observation\n"
        "alpha,0,observe,0\n"
        "alpha,1,observe,missing\n"
        "beta,0,observe,1\n"
        "beta,1,stop,0\n",
        encoding="utf-8",
    )
    manifest = copy.deepcopy(_manifest())
    records = _records(manifest)
    trajectory = records["public_agent_trace_adapter_stage_2"]
    trajectory["filename"] = trace_path.name
    trajectory["sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    return fixture_directory, manifest


def test_preflight_uses_committed_offline_fixtures_and_returns_dashboard_payload() -> None:
    preflight = preflight_offline_benchmark_fixtures(_BENCHMARK_DIRECTORY, _manifest())
    payload = preflight.dashboard_payload()

    assert payload == preflight.to_mapping()
    assert payload["kind"] == "offline_benchmark_provenance_dashboard"
    assert payload["offline_only"] is True
    assert payload["network_access"] == "not performed by this domain preflight"
    assert len(str(payload["manifest_contract_sha256"])) == 64
    golub = cast(dict[str, Any], payload["golub_conversion"])
    archive = cast(dict[str, Any], golub["source_archive"])
    fixture = cast(dict[str, Any], golub["normalized_fixture"])
    assert archive["checksum_status"] == "checksum_pinned_not_staged"
    assert archive["source"] == fixture["source"]
    assert archive["license"] == fixture["license"] == "LGPL"
    assert archive["doi"] == fixture["doi"] == "10.18129/B9.bioc.golubEsets"
    assert fixture["checksum_verified"] is True
    recipe = cast(dict[str, Any], fixture["recipe"])
    assert recipe["output_schema"] == ["sample", "feature", "class", "expression"]
    assert recipe["rectangular_sample_feature_grid"] is True
    assert recipe["n_samples"] == recipe["n_features"] == 4
    assert "does not reproduce" in recipe["claim_boundary"]

    trajectory = cast(dict[str, Any], payload["chronological_trajectory"])
    provenance = cast(dict[str, Any], trajectory["provenance"])
    assert provenance["source"] == "https://github.com/infer-actively/pymdp"
    assert provenance["license"] == "MIT"
    assert provenance["doi"] is None
    assert provenance["doi_status"] == "not_declared"
    schema = cast(dict[str, Any], trajectory["schema"])
    assert schema["chronological_within_subject"] is True
    assert schema["subject_disjoint_split"]["subject_disjoint"] is True
    missingness = cast(dict[str, Any], trajectory["missingness"])
    assert missingness["missing_observation_count"] == 0
    assert missingness["missing_observation_rate"] == 0.0
    assert "MCAR, MAR, or MNAR" in missingness["claim_boundary"]
    assert "do not establish" in str(payload["claim_boundary"])
    json.dumps(payload, allow_nan=False)


def test_preflight_reports_real_missingness_from_a_staged_chronological_trajectory(tmp_path: Path) -> None:
    fixture_directory, manifest = _stage_fixture_directory(tmp_path)
    payload = preflight_offline_benchmark_fixtures(fixture_directory, manifest).dashboard_payload()
    trajectory = cast(dict[str, Any], payload["chronological_trajectory"])
    missingness = cast(dict[str, Any], trajectory["missingness"])
    schema = cast(dict[str, Any], trajectory["schema"])

    assert missingness == {
        "rows": 4,
        "observed_observation_count": 3,
        "missing_observation_count": 1,
        "missing_observation_rate": 0.25,
        "representation": "missing observation tokens are parsed offline and encoded as -1",
        "claim_boundary": missingness["claim_boundary"],
    }
    assert schema["subjects"] == 2
    assert schema["subject_disjoint_split"]["train_rows"] == 2
    assert schema["subject_disjoint_split"]["evaluation_rows"] == 2


def test_preflight_fails_closed_for_recipe_lineage_path_and_chronology_violations(tmp_path: Path) -> None:
    fixture_directory, manifest = _stage_fixture_directory(tmp_path)
    records = _records(manifest)
    fixture = records["golub_normalized_offline_fixture"]
    fixture["preprocessing"] = "committed offline normalized fixture"
    with pytest.raises(ValueError, match="no biological claim"):
        preflight_offline_benchmark_fixtures(fixture_directory, manifest)

    manifest = _manifest()
    records = _records(manifest)
    records["golub_normalized_offline_fixture"]["doi"] = "10.0000/not-golub"
    with pytest.raises(ValueError, match="license and DOI"):
        preflight_offline_benchmark_fixtures(_BENCHMARK_DIRECTORY, manifest)

    fixture_directory, manifest = _stage_fixture_directory(
        tmp_path / "bad-chronology",
        "subject,time,action,observation\nalpha,0,observe,0\nalpha,0,observe,1\nbeta,0,observe,1\nbeta,1,stop,0\n",
    )
    with pytest.raises(ValueError, match="chronological"):
        preflight_offline_benchmark_fixtures(fixture_directory, manifest)

    manifest = _manifest()
    records = _records(manifest)
    records["public_agent_trace_adapter_stage_2"]["filename"] = "../outside.csv"
    with pytest.raises(ValueError, match="escapes"):
        preflight_offline_benchmark_fixtures(_BENCHMARK_DIRECTORY, manifest)

    manifest = _manifest()
    records = _records(manifest)
    records["golub_leukemia_bioconductor_1_54_0"]["filename"] = "golubEsets_1.54.0.tar.gz"
    staged_archive = fixture_directory / "golubEsets_1.54.0.tar.gz"
    staged_archive.write_bytes(b"not the checksum-pinned archive")
    with pytest.raises(ValueError, match="staged Golub archive"):
        preflight_offline_benchmark_fixtures(fixture_directory, manifest)


def test_preflight_defensive_contract_edges_remain_fail_closed(tmp_path: Path) -> None:
    """Exercise metadata and matrix failures using actual loaded fixture values."""
    with pytest.raises(ValueError, match="non-empty"):
        _required_text({"field": ""}, "field", "fixture")
    with pytest.raises(ValueError, match="benchmark list"):
        _records_by_id({"benchmarks": {}})
    with pytest.raises(ValueError, match="records"):
        _records_by_id({"benchmarks": [object()]})  # type: ignore[list-item]
    with pytest.raises(ValueError, match="duplicate"):
        _records_by_id({"benchmarks": [{"id": "x"}, {"id": "x"}]})
    with pytest.raises(ValueError, match="missing required"):
        _records_by_id({"benchmarks": []})
    with pytest.raises(FileNotFoundError):
        _safe_file(tmp_path / "missing", {"id": "x", "filename": "x.csv"})

    preflight = preflight_offline_benchmark_fixtures(_BENCHMARK_DIRECTORY, _manifest())
    archive_record = preflight.golub_archive_record
    fixture_record = preflight.golub_fixture_record
    archive_spec = preflight.golub_archive_spec
    fixture = preflight.golub_fixture
    fixture_spec = fixture.spec

    def validate(
        record: Mapping[str, Any] = fixture_record,
        spec: Any = fixture_spec,
        dataset: BenchmarkDataset = fixture,
        source_record: Mapping[str, Any] = archive_record,
        source_spec: Any = archive_spec,
    ) -> None:
        _validate_golub_lineage(source_record, record, source_spec, spec, dataset)

    changed_archive = dict(archive_record)
    changed_archive["source"] = "https://example.invalid/golub"
    with pytest.raises(ValueError, match="source"):
        validate(source_record=changed_archive, source_spec=replace(archive_spec, source=changed_archive["source"]))
    changed_fixture = dict(fixture_record)
    changed_fixture["source_revision"] = "unrelated-fixture"
    with pytest.raises(ValueError, match="source_revision"):
        validate(record=changed_fixture)
    with pytest.raises(ValueError, match="formats"):
        validate(spec=replace(fixture_spec, format="tsv"))
    claim_bearing_spec = replace(fixture_spec, claim_role="claim_bearing")
    with pytest.raises(ValueError, match="stress-test"):
        validate(spec=claim_bearing_spec, dataset=replace(fixture, spec=claim_bearing_spec))
    with pytest.raises(ValueError, match="columns"):
        validate(dataset=replace(fixture, columns=("sample", "feature", "class", "value")))

    non_integral = fixture.values.copy()
    non_integral[0, 0] = 1.5
    with pytest.raises(ValueError, match="integral"):
        validate(dataset=BenchmarkDataset(fixture.benchmark_id, non_integral, fixture.columns, fixture_spec))
    with pytest.raises(ValueError, match="at least two"):
        validate(dataset=BenchmarkDataset(fixture.benchmark_id, fixture.values[:1], fixture.columns, fixture_spec))
    duplicate = np.vstack((fixture.values, fixture.values[0]))
    with pytest.raises(ValueError, match="sample-feature pair"):
        validate(dataset=BenchmarkDataset(fixture.benchmark_id, duplicate, fixture.columns, fixture_spec))
    incomplete_grid = fixture.values.copy()
    incomplete_grid[0, 1] = 999
    with pytest.raises(ValueError, match="one class and every feature"):
        validate(dataset=BenchmarkDataset(fixture.benchmark_id, incomplete_grid, fixture.columns, fixture_spec))

    with pytest.raises(ValueError, match="columns"):
        _trajectory_reports(replace(preflight.trajectory, columns=("t", "y")))
    with pytest.raises(ValueError, match="metadata"):
        _trajectory_reports(replace(preflight.trajectory, metadata={"subjects": ["one"], "actions": ["observe"]}))
    nonchronological = preflight.trajectory.values.copy()
    nonchronological[1, 0] = nonchronological[0, 0]
    with pytest.raises(ValueError, match="chronological"):
        _trajectory_reports(
            BenchmarkDataset(
                preflight.trajectory.benchmark_id,
                nonchronological,
                preflight.trajectory.columns,
                preflight.trajectory.spec,
                preflight.trajectory.metadata,
            )
        )

    fixture_directory, manifest = _stage_fixture_directory(tmp_path / "archive-directory")
    archive_directory = fixture_directory / "golubEsets_1.54.0.tar.gz"
    archive_directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        preflight_offline_benchmark_fixtures(fixture_directory, manifest)
