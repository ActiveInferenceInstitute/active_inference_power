#!/usr/bin/env python3
"""Emit an offline, hash-linked preflight package for external benchmark fixtures.

The command never downloads a benchmark and never chooses an output directory
implicitly.  It validates the committed manifest and staged fixtures through
the deterministic domain preflight, then writes a JSON provenance dashboard,
a small human-readable HTML dashboard, and a manifest suitable for later
atlas/release linkage.
"""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(ROOT / "src"))

from active_inference_power.benchmark_preflight import (  # noqa: E402
    OfflineBenchmarkPreflight,
    preflight_offline_benchmark_fixtures,
)
from scripts.artifacts import (  # noqa: E402
    artifact_record,
    atomic_write_json,
    atomic_write_text,
    require_safe_output_root,
    sha256_file,
)


_MANIFEST_PATH = ROOT / "data" / "benchmarks" / "manifest.yaml"
_BENCHMARK_DIRECTORY = ROOT / "data" / "benchmarks"
_LANE_NAME = "benchmark_preflight"
_PREFLIGHT_SOURCE_PATHS = (
    Path("scripts/artifacts.py"),
    Path("scripts/run_benchmark_preflight.py"),
    Path("src/active_inference_power/benchmark_preflight.py"),
    Path("src/active_inference_power/benchmarks.py"),
    Path("src/active_inference_power/schema.py"),
)
_EXTERNAL_STRESS_NONCLAIM = (
    "These checksum-pinned external fixtures are stress evidence only. They do not "
    "establish biological, causal, sequential-validity, family-level, or synthetic "
    "guarantees, and they do not promote the release evidence regime."
)


def _resolve_output_root(output_root: Path) -> Path:
    output = Path(output_root)
    if not output.is_absolute():
        output = ROOT / output
    output = require_safe_output_root(output, project_root=ROOT)
    try:
        output.relative_to(_BENCHMARK_DIRECTORY.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output_root must not be inside the benchmark input directory")
    return output


def _load_manifest(manifest_path: Path) -> Mapping[str, Any]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"benchmark manifest is missing: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"benchmark manifest is not valid YAML: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("benchmark manifest must be a mapping")
    return value


def _safe_fixture_file(benchmark_directory: Path, record: Mapping[str, Any]) -> Path:
    benchmark_id = str(record.get("id", ""))
    filename = record.get("filename")
    if not benchmark_id or not isinstance(filename, str) or not filename:
        raise ValueError("benchmark record requires non-empty id and filename")
    root = Path(benchmark_directory).resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"benchmark {benchmark_id!r} filename escapes the benchmark directory") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"benchmark fixture is missing: {candidate}")
    return candidate


def _fixture_hashes_with_optional_archive(
    preflight: OfflineBenchmarkPreflight, benchmark_directory: Path
) -> dict[str, dict[str, object]]:
    """Keep an absent archive checksum-pinned without pretending it was read."""
    fixture_record = preflight.golub_fixture_record
    trajectory_record = preflight.trajectory_record
    fixture_path = _safe_fixture_file(benchmark_directory, fixture_record)
    trajectory_path = _safe_fixture_file(benchmark_directory, trajectory_record)
    archive_record = preflight.golub_archive_record
    archive_path = (Path(benchmark_directory).resolve() / str(archive_record["filename"])).resolve()
    try:
        archive_path.relative_to(Path(benchmark_directory).resolve())
    except ValueError as exc:
        raise ValueError("Golub archive filename escapes the benchmark directory") from exc
    archive_hash: dict[str, object] = {
        "filename": str(archive_record["filename"]),
        "declared_sha256": preflight.golub_archive_spec.sha256,
        "staged_locally": preflight.golub_archive_staged,
        "observed_sha256": (
            sha256_file(archive_path) if preflight.golub_archive_staged else None
        ),
    }
    return {
        "golub_source_archive": archive_hash,
        "golub_normalized_fixture": {
            "filename": fixture_path.name,
            "declared_sha256": preflight.golub_fixture.spec.sha256,
            "staged_locally": True,
            "observed_sha256": sha256_file(fixture_path),
        },
        "chronological_trajectory": {
            "filename": trajectory_path.name,
            "declared_sha256": preflight.trajectory.spec.sha256,
            "staged_locally": True,
            "observed_sha256": sha256_file(trajectory_path),
        },
    }


def _validate_observed_hashes(file_hashes: Mapping[str, Mapping[str, object]]) -> None:
    """Fail closed if a staged file changes after the domain preflight."""
    for record in file_hashes.values():
        if record["observed_sha256"] is not None and record["observed_sha256"] != record["declared_sha256"]:
            raise ValueError("observed fixture hash does not match the declared benchmark checksum")


def _preflight_source_witness(project_root: Path = ROOT) -> dict[str, object]:
    """Bind this isolated receipt to its runner and deterministic domain code."""
    project = Path(project_root).resolve()
    files: dict[str, str] = {}
    digest = hashlib.sha256()
    for relative in _PREFLIGHT_SOURCE_PATHS:
        path = project / relative
        if not path.is_file():
            raise FileNotFoundError(f"preflight source is missing: {path}")
        name = relative.as_posix()
        file_digest = sha256_file(path)
        files[name] = file_digest
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(file_digest))
    return {"sha256": digest.hexdigest(), "files": files}


def _render_dashboard(payload: Mapping[str, Any], *, dashboard_sha256: str) -> str:
    """Render a minimal, static dashboard with provenance instead of inference claims."""
    hashes = payload["file_hashes"]
    assert isinstance(hashes, Mapping)
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(name))}</td>"
        f"<td><code>{escape(str(record['declared_sha256']))}</code></td>"
        f"<td><code>{escape(str(record['observed_sha256']))}</code></td>"
        f"<td>{escape(str(record['staged_locally']))}</td>"
        "</tr>"
        for name, record in hashes.items()
        if isinstance(record, Mapping)
    )
    dashboard = payload["dashboard"]
    assert isinstance(dashboard, Mapping)
    source_witness = payload["preflight_source"]
    assert isinstance(source_witness, Mapping)
    golub = dashboard["golub_conversion"]
    trajectory = dashboard["chronological_trajectory"]
    assert isinstance(golub, Mapping) and isinstance(trajectory, Mapping)
    source = golub["source_archive"]
    fixture = golub["normalized_fixture"]
    trajectory_provenance = trajectory["provenance"]
    assert isinstance(source, Mapping) and isinstance(fixture, Mapping)
    assert isinstance(trajectory_provenance, Mapping)
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            "  <title>Offline benchmark preflight</title>",
            f'  <meta name="benchmark-manifest-sha256" content="{escape(str(payload["source_manifest"]["sha256"]))}">',
            f'  <meta name="preflight-payload-sha256" content="{escape(dashboard_sha256)}">',
            f'  <meta name="preflight-source-sha256" content="{escape(str(source_witness["sha256"]))}">',
            "</head>",
            "<body>",
            "  <h1>Offline benchmark preflight</h1>",
            "  <p>This dashboard was generated without network retrieval.</p>",
            f"  <p>Runner/domain source witness: <code>{escape(str(source_witness['sha256']))}</code></p>",
            "  <h2>Golub conversion provenance</h2>",
            f"  <p>Source: {escape(str(source['source']))}</p>",
            f"  <p>License: {escape(str(source['license']))}; DOI: {escape(str(source['doi']))}</p>",
            f"  <p>Normalized fixture: {escape(str(fixture['id']))}</p>",
            "  <h2>Chronological trajectory provenance</h2>",
            f"  <p>Source: {escape(str(trajectory_provenance['source']))}</p>",
            f"  <p>License: {escape(str(trajectory_provenance['license']))}; DOI: {escape(str(trajectory_provenance['doi']))}</p>",
            "  <h2>Exact fixture hashes</h2>",
            "  <table><thead><tr><th>fixture</th><th>declared SHA-256</th><th>observed SHA-256</th><th>staged locally</th></tr></thead>",
            f"  <tbody>{rows}</tbody></table>",
            "  <h2>External-stress nonclaim</h2>",
            f"  <p>{escape(str(payload['external_stress_nonclaim']))}</p>",
            f"  <p>{escape(str(dashboard['claim_boundary']))}</p>",
            "</body>",
            "</html>",
            "",
        )
    )


def run_benchmark_preflight(
    output_root: Path,
    *,
    benchmark_directory: Path = _BENCHMARK_DIRECTORY,
    manifest_path: Path = _MANIFEST_PATH,
) -> dict[str, object]:
    """Run the offline preflight and atomically write its isolated artifact lane."""
    output = _resolve_output_root(output_root)
    manifest_path = Path(manifest_path).resolve()
    benchmark_directory = Path(benchmark_directory).resolve()
    manifest = _load_manifest(manifest_path)
    preflight = preflight_offline_benchmark_fixtures(benchmark_directory, manifest)
    file_hashes = _fixture_hashes_with_optional_archive(preflight, benchmark_directory)
    _validate_observed_hashes(file_hashes)
    source_witness = _preflight_source_witness()

    lane = output / _LANE_NAME
    dashboard_path = lane / "provenance_dashboard.json"
    html_path = lane / "provenance_dashboard.html"
    manifest_output_path = lane / "manifest.json"
    dashboard = preflight.dashboard_payload()
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "kind": "external_benchmark_release_preflight",
        "offline_only": True,
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "preflight_contract_sha256": preflight.manifest_contract_sha256,
        },
        "preflight_source": source_witness,
        "benchmark_directory": str(benchmark_directory),
        "file_hashes": file_hashes,
        "external_stress_nonclaim": _EXTERNAL_STRESS_NONCLAIM,
        "dashboard": dashboard,
    }
    atomic_write_json(dashboard_path, payload)
    payload_hash = sha256_file(dashboard_path)
    atomic_write_text(html_path, _render_dashboard(payload, dashboard_sha256=payload_hash))
    artifact_manifest: dict[str, object] = {
        "schema_version": "1.0",
        "kind": "external_benchmark_release_preflight_manifest",
        "offline_only": True,
        "source_manifest_sha256": sha256_file(manifest_path),
        "preflight_contract_sha256": preflight.manifest_contract_sha256,
        "preflight_source_sha256": source_witness["sha256"],
        "dashboard_sha256": sha256_file(dashboard_path),
        "external_stress_nonclaim": _EXTERNAL_STRESS_NONCLAIM,
        "artifacts": {
            "provenance_dashboard_json": artifact_record(
                dashboard_path, lane, "benchmark_preflight_dashboard", root_label="output"
            ),
            "provenance_dashboard_html": artifact_record(
                html_path, lane, "benchmark_preflight_dashboard_html", root_label="output"
            ),
        },
    }
    atomic_write_json(manifest_output_path, artifact_manifest)
    return {
        "output_root": str(output),
        "dashboard": str(dashboard_path),
        "html_dashboard": str(html_path),
        "manifest": str(manifest_output_path),
        "manifest_sha256": sha256_file(manifest_output_path),
        "dashboard_sha256": sha256_file(dashboard_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="explicit directory that receives the isolated benchmark_preflight lane",
    )
    args = parser.parse_args()
    result = run_benchmark_preflight(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
