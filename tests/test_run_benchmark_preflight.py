"""Focused integration tests for the explicit-output offline preflight command."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

import pytest
import yaml

from scripts.artifacts import sha256_file
from active_inference_power.benchmark_preflight import preflight_offline_benchmark_fixtures
from scripts.run_benchmark_preflight import (
    ROOT,
    _PREFLIGHT_SOURCE_PATHS,
    _fixture_hashes_with_optional_archive,
    _load_manifest,
    _preflight_source_witness,
    _resolve_output_root,
    _safe_fixture_file,
    _validate_observed_hashes,
    main,
    run_benchmark_preflight,
)


_BENCHMARK_DIRECTORY = ROOT / "data" / "benchmarks"
_MANIFEST_PATH = _BENCHMARK_DIRECTORY / "manifest.yaml"
_SCRIPT_PATH = ROOT / "scripts" / "run_benchmark_preflight.py"


def _manifest() -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(_MANIFEST_PATH.read_text(encoding="utf-8")))


def test_preflight_command_writes_hash_linked_offline_artifacts_to_explicit_root(tmp_path: Path) -> None:
    output_root = tmp_path / "isolated-bundle"
    result = run_benchmark_preflight(output_root)
    dashboard_path = Path(cast(str, result["dashboard"]))
    html_path = Path(cast(str, result["html_dashboard"]))
    manifest_path = Path(cast(str, result["manifest"]))

    assert dashboard_path.parent == html_path.parent == manifest_path.parent
    assert dashboard_path.parent.name == "benchmark_preflight"
    assert dashboard_path.is_file() and html_path.is_file() and manifest_path.is_file()
    assert dashboard_path.is_relative_to(output_root)
    assert html_path.is_relative_to(output_root)
    assert manifest_path.is_relative_to(output_root)
    assert result["dashboard_sha256"] == sha256_file(dashboard_path)
    assert result["manifest_sha256"] == sha256_file(manifest_path)

    payload = cast(dict[str, Any], json.loads(dashboard_path.read_text(encoding="utf-8")))
    artifact_manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    assert payload["kind"] == "external_benchmark_release_preflight"
    assert payload["offline_only"] is True
    assert payload["source_manifest"]["sha256"] == sha256_file(_MANIFEST_PATH)
    source_witness = cast(dict[str, Any], payload["preflight_source"])
    assert len(str(source_witness["sha256"])) == 64
    assert set(source_witness["files"]) == {
        relative.as_posix() for relative in _PREFLIGHT_SOURCE_PATHS
    }
    assert artifact_manifest["preflight_source_sha256"] == source_witness["sha256"]
    assert payload["external_stress_nonclaim"] == artifact_manifest["external_stress_nonclaim"]
    assert "do not establish" in payload["external_stress_nonclaim"]
    for record in payload["file_hashes"].values():
        if record["observed_sha256"] is not None:
            assert record["observed_sha256"] == record["declared_sha256"]
    assert artifact_manifest["dashboard_sha256"] == sha256_file(dashboard_path)
    for record in artifact_manifest["artifacts"].values():
        assert record["root"] == "output"
        assert ".." not in Path(record["path"]).parts
        artifact = dashboard_path.parent / record["path"]
        assert record["sha256"] == sha256_file(artifact)
        assert record["bytes"] == artifact.stat().st_size

    html = html_path.read_text(encoding="utf-8")
    assert "without network retrieval" in html
    assert "https://bioconductor.org/" in html
    assert "LGPL" in html and "10.18129/B9.bioc.golubEsets" in html
    assert payload["source_manifest"]["sha256"] in html
    assert source_witness["sha256"] in html
    assert result["dashboard_sha256"] in html
    assert payload["external_stress_nonclaim"] in html


def test_preflight_rejects_missing_fixture_traversal_and_implicit_project_root(tmp_path: Path) -> None:
    output_root = tmp_path / "should-remain-empty"
    missing_directory = tmp_path / "missing-benchmarks"
    missing_directory.mkdir()
    with pytest.raises(FileNotFoundError):
        run_benchmark_preflight(
            output_root,
            benchmark_directory=missing_directory,
            manifest_path=_MANIFEST_PATH,
        )
    assert not output_root.exists()

    manifest = copy.deepcopy(_manifest())
    records = cast(list[dict[str, Any]], manifest["benchmarks"])
    next(record for record in records if record["id"] == "golub_normalized_offline_fixture")["filename"] = "../outside.csv"
    malicious_manifest = tmp_path / "malicious.yaml"
    malicious_manifest.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        run_benchmark_preflight(
            output_root,
            benchmark_directory=_BENCHMARK_DIRECTORY,
            manifest_path=malicious_manifest,
        )
    assert not output_root.exists()

    with pytest.raises(ValueError, match="project root"):
        run_benchmark_preflight(ROOT)
    with pytest.raises(ValueError, match="output_root must"):
        run_benchmark_preflight(_BENCHMARK_DIRECTORY)


def test_preflight_source_witness_is_path_bound_and_tracks_runner_changes(tmp_path: Path) -> None:
    """The receipt's bounded source witness detects runner/domain drift."""
    for relative in _PREFLIGHT_SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative.as_posix()}\n", encoding="utf-8")
    first = _preflight_source_witness(tmp_path)
    assert set(cast(dict[str, str], first["files"])) == {
        relative.as_posix() for relative in _PREFLIGHT_SOURCE_PATHS
    }
    changed = tmp_path / "src" / "active_inference_power" / "schema.py"
    changed.write_text("# changed transitive schema validator\n", encoding="utf-8")
    second = _preflight_source_witness(tmp_path)
    assert first["sha256"] != second["sha256"]


def test_cli_requires_output_root_and_runs_only_when_one_is_explicit(tmp_path: Path) -> None:
    omitted = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert omitted.returncode == 2
    assert "--output-root" in omitted.stderr

    output_root = tmp_path / "cli-bundle"
    default_lane = ROOT / "output" / "benchmark_preflight"
    default_lane_was_present = default_lane.exists()
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--output-root", str(output_root)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = cast(dict[str, Any], json.loads(completed.stdout))
    assert Path(cast(str, result["dashboard"])).is_file()
    if not default_lane_was_present:
        assert not default_lane.exists()


def test_runner_helpers_fail_closed_on_invalid_inputs_and_changed_staged_hashes(tmp_path: Path) -> None:
    assert _resolve_output_root(Path("output/isolated-relative-output")) == (
        ROOT / "output" / "isolated-relative-output"
    ).resolve()
    with pytest.raises(FileNotFoundError, match="manifest"):
        _load_manifest(tmp_path / "missing.yaml")
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("[not: valid", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML"):
        _load_manifest(invalid_yaml)
    scalar_yaml = tmp_path / "scalar.yaml"
    scalar_yaml.write_text("only-a-scalar\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        _load_manifest(scalar_yaml)
    with pytest.raises(ValueError, match="id and filename"):
        _safe_fixture_file(tmp_path, {"id": "", "filename": "fixture.csv"})
    with pytest.raises(ValueError, match="escapes"):
        _safe_fixture_file(tmp_path, {"id": "fixture", "filename": "../fixture.csv"})
    with pytest.raises(FileNotFoundError, match="fixture is missing"):
        _safe_fixture_file(tmp_path, {"id": "fixture", "filename": "fixture.csv"})
    with pytest.raises(ValueError, match="declared benchmark checksum"):
        _validate_observed_hashes(
            {"fixture": {"declared_sha256": "a" * 64, "observed_sha256": "b" * 64}}
        )
    _validate_observed_hashes(
        {"not_staged": {"declared_sha256": "a" * 64, "observed_sha256": None}}
    )

    manifest = _manifest()
    preflight = preflight_offline_benchmark_fixtures(_BENCHMARK_DIRECTORY, manifest)
    escaped_archive = dict(preflight.golub_archive_record)
    escaped_archive["filename"] = "../archive.tar.gz"
    with pytest.raises(ValueError, match="archive filename escapes"):
        _fixture_hashes_with_optional_archive(
            replace(preflight, golub_archive_record=escaped_archive), _BENCHMARK_DIRECTORY
        )

    argv = sys.argv
    try:
        main_output = tmp_path / "main-bundle"
        sys.argv = [str(_SCRIPT_PATH), "--output-root", str(main_output)]
        assert main() == 0
    finally:
        sys.argv = argv
    assert (main_output / "benchmark_preflight" / "manifest.json").is_file()
