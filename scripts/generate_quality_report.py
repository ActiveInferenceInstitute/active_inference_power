#!/usr/bin/env python3
"""Run the project quality gate and write its measured provenance snapshot."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from active_inference_power.schema import validate_manifest, validate_quality_report  # noqa: E402
from scripts.artifacts import (  # noqa: E402
    artifact_record,
    atomic_write_json,
    invalidate_renderer_artifact_records,
    require_safe_output_root,
    sha256_file,
)


_QUALITY_PENDING_BEFORE_RUN = (
    "quality.json",
    "manuscript_hydration",
    "publication_render",
    "release_finalization",
)
_QUALITY_PENDING_AFTER_PASS = (
    "manuscript_hydration",
    "publication_render",
    "release_finalization",
)
QUALITY_INPUT_DIRECTORIES = (
    "src",
    "scripts",
    "tests",
    "examples",
    "docs",
    "manuscript",
    "data",
    ".agents",
    ".github",
)
QUALITY_INPUT_ROOT_FILES = (
    ".gitignore",
    ".zenodo.json",
    "AGENTS.md",
    "CITATION.cff",
    "README.md",
    "STANDALONE.md",
    "TODO.md",
    "codemeta.json",
    "domain_profile.yaml",
    "experiment_plan.yaml",
    "export.toml",
    "pyproject.toml",
    "run.sh",
    "uv.lock",
)


def _resolve_output_root(output_root: Path | None) -> Path:
    output = ROOT / "output" if output_root is None else Path(output_root)
    if not output.is_absolute():
        output = ROOT / output
    return require_safe_output_root(output, project_root=ROOT)


def quality_input_paths(project_root: Path = ROOT) -> tuple[Path, ...]:
    """Return every authored input that can affect the full quality-gate run.

    This is an explicit project-surface allowlist, rather than a Git query, so
    it works in a dirty checkout while excluding generated output, virtual
    environments, build products, caches, and user-local tooling.  The full
    test run exercises source, scripts, examples, documentation, manuscript,
    and data fixtures; each belongs in the quality witness.
    """
    project = Path(project_root).resolve()
    root_files = tuple(project / relative for relative in QUALITY_INPUT_ROOT_FILES)
    for path in root_files:
        if not path.is_file():
            raise FileNotFoundError(f"quality-gate input is missing: {path}")
    tree_inputs: list[Path] = []
    for relative in QUALITY_INPUT_DIRECTORIES:
        directory = project / relative
        if not directory.is_dir():
            raise FileNotFoundError(
                f"quality-gate authored input directory is missing: {directory}"
            )
        tree_inputs.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
            and path.name != ".DS_Store"
        )
    if not tree_inputs:
        raise ValueError("quality-gate authored input surface has no files")
    paths = {path.resolve() for path in (*root_files, *tree_inputs)}
    return tuple(sorted(paths, key=lambda path: path.relative_to(project).as_posix()))


def quality_input_sha256(project_root: Path = ROOT) -> str:
    """Hash the exact test-suite and harness inputs in a path-bound order."""
    project = Path(project_root).resolve()
    digest = hashlib.sha256()
    for path in quality_input_paths(project):
        relative = path.relative_to(project).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def validate_quality_input_provenance(
    report: Mapping[str, Any], *, project_root: Path = ROOT
) -> str:
    """Require a quality report to attest to the current test-suite inputs."""
    recorded = report.get("quality_input_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("quality report has no valid quality-input witness")
    current = quality_input_sha256(project_root)
    if recorded != current:
        raise ValueError(
            "quality-input provenance mismatch between quality report and live test harness"
        )
    return recorded


def _artifact_for_output(path: Path, output: Path, role: str) -> dict[str, Any]:
    """Record an output file in the same namespace convention as hydration."""
    path = Path(path).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        return artifact_record(path, output, role, root_label="output")
    return artifact_record(path, ROOT, role, root_label="project")


def _invalidate_existing_manifest_before_quality(output: Path) -> dict[str, Any] | None:
    """Revoke completion before replacing quality evidence in an existing bundle."""
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"existing manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"existing manifest must be a JSON object: {manifest_path}")
    validate_manifest(manifest)
    invalidate_renderer_artifact_records(manifest)
    manifest["complete"] = False
    manifest["validation_pending"] = list(_QUALITY_PENDING_BEFORE_RUN)
    validate_manifest(manifest)
    atomic_write_json(manifest_path, manifest)
    return manifest


def _record_quality_manifest(
    output: Path, manifest: dict[str, Any], report: Mapping[str, Any]
) -> list[str]:
    """Bind a written quality report while keeping downstream release gates pending."""
    validate_quality_report(report)
    quality_path = output / "quality.json"
    artifacts = manifest.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("manifest artifacts must be a mapping")
    artifacts["quality"] = _artifact_for_output(quality_path, output, "quality_report")
    pending = (
        list(_QUALITY_PENDING_AFTER_PASS)
        if report["quality_pass"] is True
        else list(_QUALITY_PENDING_BEFORE_RUN)
    )
    manifest["complete"] = False
    manifest["validation_pending"] = pending
    validate_manifest(manifest)
    atomic_write_json(output / "manifest.json", manifest)
    return pending


def _collect_count(python: str) -> int:
    command = [python, "-m", "pytest", "tests/", "--collect-only", "-q"]
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=True
    )
    match = re.search(r"(\d+) tests? collected", completed.stdout + completed.stderr)
    if not match:
        raise RuntimeError("pytest collection output did not contain a test count")
    return int(match.group(1))


def _passes_quality_gate(
    *,
    test_count: int,
    pytest_exit_code: int,
    line_coverage_pct: float,
    branch_coverage_pct: float,
    coverage_gate_pct: float = 95.0,
    branch_coverage_gate_pct: float = 95.0,
) -> bool:
    """Return whether all measured release-quality requirements are met."""
    return bool(
        test_count > 0
        and pytest_exit_code == 0
        and line_coverage_pct >= coverage_gate_pct
        and branch_coverage_pct >= branch_coverage_gate_pct
    )


def main(output_root: Path | None = None, *, python: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python", default=sys.executable, help="Python interpreter for pytest."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for coverage.json and quality.json; relative paths use the project root.",
    )
    args = parser.parse_args() if output_root is None and python is None else None
    interpreter = python or (args.python if args is not None else sys.executable)
    selected_output = _resolve_output_root(
        output_root if output_root is not None else (args.output_root if args is not None else None)
    )
    selected_output.mkdir(parents=True, exist_ok=True)
    manifest = _invalidate_existing_manifest_before_quality(selected_output)
    coverage_path = selected_output / "coverage.json"

    quality_inputs_before = quality_input_sha256(ROOT)
    count = _collect_count(interpreter)
    command = [
        interpreter,
        "-m",
        "pytest",
        "tests/",
        "--cov=active_inference_power",
        "--cov-branch",
        f"--cov-report=json:{coverage_path}",
        "--cov-fail-under=95",
        "-q",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, check=False)
    if not coverage_path.is_file():
        raise RuntimeError(f"pytest did not write the coverage report: {coverage_path}")
    if quality_input_sha256(ROOT) != quality_inputs_before:
        raise RuntimeError("quality-gate inputs changed while pytest was running")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    total = coverage["totals"]
    line_coverage_pct = float(total["percent_covered"])
    branch_coverage_pct = float(
        100.0 * total["covered_branches"] / max(total["num_branches"], 1)
    )
    report = {
        "schema_version": "2.0",
        "test_count": count,
        "line_coverage_pct": line_coverage_pct,
        "branch_coverage_pct": branch_coverage_pct,
        "coverage_gate_pct": 95.0,
        "branch_coverage_gate_pct": 95.0,
        "quality_pass": _passes_quality_gate(
            test_count=count,
            pytest_exit_code=completed.returncode,
            line_coverage_pct=line_coverage_pct,
            branch_coverage_pct=branch_coverage_pct,
        ),
        "quality_input_sha256": quality_inputs_before,
        "pytest_exit_code": completed.returncode,
        "command": " ".join(command),
    }
    output = selected_output / "quality.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    if manifest is not None:
        _record_quality_manifest(selected_output, manifest, report)
    print(json.dumps(report, indent=2))
    return completed.returncode if report["quality_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
