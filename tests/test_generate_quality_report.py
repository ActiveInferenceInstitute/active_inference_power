"""Provenance and lifecycle contracts for the measured quality producer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.artifacts import artifact_record, sha256_file
from scripts.generate_quality_report import (
    _invalidate_existing_manifest_before_quality,
    _passes_quality_gate,
    _record_quality_manifest,
    QUALITY_INPUT_DIRECTORIES,
    QUALITY_INPUT_ROOT_FILES,
    quality_input_paths,
    quality_input_sha256,
    validate_quality_input_provenance,
)


def _quality_report(digest: str) -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "test_count": 1,
        "line_coverage_pct": 100.0,
        "branch_coverage_pct": 100.0,
        "coverage_gate_pct": 95.0,
        "branch_coverage_gate_pct": 95.0,
        "quality_pass": True,
        "quality_input_sha256": digest,
        "pytest_exit_code": 0,
        "command": "python -m pytest tests/",
    }


def test_quality_pass_requires_real_collection_and_both_coverage_gates() -> None:
    assert _passes_quality_gate(
        test_count=1,
        pytest_exit_code=0,
        line_coverage_pct=95.0,
        branch_coverage_pct=95.0,
    )
    assert not _passes_quality_gate(
        test_count=0,
        pytest_exit_code=0,
        line_coverage_pct=100.0,
        branch_coverage_pct=100.0,
    )
    assert not _passes_quality_gate(
        test_count=1,
        pytest_exit_code=0,
        line_coverage_pct=94.9,
        branch_coverage_pct=100.0,
    )
    assert not _passes_quality_gate(
        test_count=1,
        pytest_exit_code=0,
        line_coverage_pct=100.0,
        branch_coverage_pct=94.9,
    )
    assert not _passes_quality_gate(
        test_count=1,
        pytest_exit_code=1,
        line_coverage_pct=100.0,
        branch_coverage_pct=100.0,
    )


def test_quality_input_witness_is_path_bound_and_tracks_test_harness_changes(
    tmp_path: Path,
) -> None:
    for directory in QUALITY_INPUT_DIRECTORIES:
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "active_inference_power").mkdir(exist_ok=True)
    (tmp_path / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    for relative in QUALITY_INPUT_ROOT_FILES:
        (tmp_path / relative).write_text(f"# {relative}\n", encoding="utf-8")
    (tmp_path / "scripts" / "generate_quality_report.py").write_text(
        "# runner\n", encoding="utf-8"
    )
    test_path = tmp_path / "tests" / "test_real.py"
    test_path.write_text("def test_real():\n    assert True\n", encoding="utf-8")
    fixture = tmp_path / "tests" / "fixtures" / "input.txt"
    fixture.write_text("first\n", encoding="utf-8")
    source_path = tmp_path / "src" / "active_inference_power" / "domain.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    script_path = tmp_path / "scripts" / "non_generator_helper.py"
    script_path.write_text("VALUE = 1\n", encoding="utf-8")
    example_path = tmp_path / "examples" / "example.py"
    example_path.write_text("VALUE = 1\n", encoding="utf-8")
    documentation_path = tmp_path / "docs" / "contract.md"
    documentation_path.write_text("first\n", encoding="utf-8")
    manuscript_path = tmp_path / "manuscript" / "section.md"
    manuscript_path.write_text("first\n", encoding="utf-8")
    data_path = tmp_path / "data" / "fixture.csv"
    data_path.write_text("x\n1\n", encoding="utf-8")

    included = {
        path.relative_to(tmp_path).as_posix() for path in quality_input_paths(tmp_path)
    }
    assert {
        "scripts/non_generator_helper.py",
        "examples/example.py",
        "docs/contract.md",
        "manuscript/section.md",
        "data/fixture.csv",
    } <= included

    first = quality_input_sha256(tmp_path)
    assert validate_quality_input_provenance(
        {"quality_input_sha256": first}, project_root=tmp_path
    ) == first

    fixture.write_text("changed\n", encoding="utf-8")
    assert quality_input_sha256(tmp_path) != first
    with pytest.raises(ValueError, match="quality-input provenance mismatch"):
        validate_quality_input_provenance(
            {"quality_input_sha256": first}, project_root=tmp_path
        )

    second = quality_input_sha256(tmp_path)
    source_path.write_text("VALUE = 2\n", encoding="utf-8")
    assert quality_input_sha256(tmp_path) != second
    with pytest.raises(ValueError, match="quality-input provenance mismatch"):
        validate_quality_input_provenance(
            {"quality_input_sha256": second}, project_root=tmp_path
        )

    third = quality_input_sha256(tmp_path)
    for path in (
        script_path,
        example_path,
        documentation_path,
        manuscript_path,
        data_path,
    ):
        path.write_text("changed\n", encoding="utf-8")
        assert quality_input_sha256(tmp_path) != third
        third = quality_input_sha256(tmp_path)


def test_quality_refresh_revokes_completion_before_replacing_quality_evidence(
    tmp_path: Path,
) -> None:
    quality_path = tmp_path / "quality.json"
    quality_path.write_text(json.dumps(_quality_report("a" * 64)), encoding="utf-8")
    html_path = tmp_path / "web" / "index.html"
    html_path.parent.mkdir()
    html_path.write_text("old render\n", encoding="utf-8")
    manifest = {
        "schema_version": "2.0",
        "run_id": "prior-run",
        "complete": True,
        "config_sha256": "b" * 64,
        "artifacts": {
            "quality": artifact_record(
                quality_path, tmp_path, "quality_report", root_label="output"
            ),
            "rendered_html": artifact_record(
                html_path, tmp_path, "rendered_html", root_label="output"
            ),
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    refreshed = _invalidate_existing_manifest_before_quality(tmp_path)
    assert refreshed is not None
    revoked = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert revoked["complete"] is False
    assert revoked["validation_pending"] == [
        "quality.json",
        "manuscript_hydration",
        "publication_render",
        "release_finalization",
    ]
    assert set(revoked["artifacts"]) == {"quality"}

    replacement = _quality_report("c" * 64)
    quality_path.write_text(json.dumps(replacement), encoding="utf-8")
    pending = _record_quality_manifest(tmp_path, refreshed, replacement)
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert pending == [
        "manuscript_hydration",
        "publication_render",
        "release_finalization",
    ]
    assert persisted["complete"] is False
    assert persisted["artifacts"]["quality"]["sha256"] == sha256_file(quality_path)
