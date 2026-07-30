#!/usr/bin/env python3
"""Read-only integrity audit for a generated non-release fast bundle.

This command deliberately does not invoke the publication reviewer: a ``fast``
profile is useful for exercising current producers and their projections, but
it is not a release or publication candidate. It validates the generated
schemas, source and execution witnesses, result-derived artifacts, figure
registry and PNG accessibility, quality receipt, and every manifest-recorded
file without asserting rendered-publication readiness or changing the bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from active_inference_power.schema import (  # noqa: E402
    validate_figure_registry,
    validate_manifest,
    validate_quality_report,
    validate_results,
    validate_run_log,
)
from scripts.artifacts import require_safe_output_root, sha256_file  # noqa: E402
from scripts.audit_figure_accessibility import audit_figure_accessibility  # noqa: E402
from scripts.generate_quality_report import validate_quality_input_provenance  # noqa: E402
from scripts.release_artifact_contracts import validate_result_derivative_artifacts  # noqa: E402
from scripts.release_provenance import (  # noqa: E402
    manifest_path,
    require_file,
    validate_execution_provenance,
    validate_source_witnesses,
)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    """Load one required object-shaped receipt without accepting a stale stub."""
    source = require_file(path, description)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def audit_fast_bundle(
    output_root: Path | None = None, *, project_root: Path = ROOT
) -> dict[str, Any]:
    """Return a receipt after strictly validating one non-release fast bundle."""
    project = Path(project_root).resolve()
    requested = project / "output" if output_root is None else Path(output_root)
    if not requested.is_absolute():
        requested = project / requested
    output = require_safe_output_root(requested, project_root=project)

    results = _load_json(output / "results.json", "fast results")
    manifest = _load_json(output / "manifest.json", "fast manifest")
    run_log = _load_json(output / "run_log.json", "fast run log")
    quality = _load_json(output / "quality.json", "fast quality report")
    registry = _load_json(output / "figures" / "figure_registry.json", "fast figure registry")

    validate_results(results, require_suite=True, require_current_contract=True)
    validate_manifest(manifest)
    validate_run_log(run_log)
    validate_quality_report(quality)
    validate_figure_registry(registry)

    metadata = results.get("meta")
    if not isinstance(metadata, Mapping) or metadata.get("profile") != "fast":
        raise ValueError("fast-bundle audit requires results.meta.profile == 'fast'")
    if manifest.get("complete") is not False:
        raise ValueError("fast-bundle audit requires an incomplete non-release manifest")
    if quality.get("quality_pass") is not True:
        raise ValueError("fast-bundle audit requires a passing quality receipt")
    if not (
        manifest.get("run_id") == results.get("run_id") == run_log.get("run_id")
    ):
        raise ValueError("fast-bundle run identifiers disagree")

    source_witnesses = validate_source_witnesses(
        manifest,
        results,
        run_log,
        project_root=project,
        output_root=output,
        config_path=project / "manuscript" / "config.yaml",
        claim_ledger_path=project / "data" / "claim_ledger.yaml",
    )
    execution_witness = validate_execution_provenance(
        manifest, results, run_log, project_root=project
    )
    validate_quality_input_provenance(quality, project_root=project)
    derivatives = validate_result_derivative_artifacts(output, results)
    figure_audit = audit_figure_accessibility(output)
    if figure_audit.get("passed") is not True:
        raise ValueError("fast-bundle figure accessibility audit failed")

    artifact_count = 0
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("fast manifest artifacts must be a mapping")
    for name, record_value in artifacts.items():
        if not isinstance(record_value, Mapping):
            raise ValueError(f"fast manifest artifact {name!r} is not an object")
        path = require_file(
            manifest_path(output, project, record_value),
            f"fast manifest artifact {name!r}",
        )
        if sha256_file(path) != record_value.get("sha256"):
            raise ValueError(f"fast manifest artifact {name!r} digest mismatch")
        if path.stat().st_size != record_value.get("bytes"):
            raise ValueError(f"fast manifest artifact {name!r} size mismatch")
        artifact_count += 1

    return {
        "schema_version": "1.0",
        "passed": True,
        "profile": "fast",
        "output_root": str(output),
        "run_id": str(results["run_id"]),
        "artifact_count": artifact_count,
        "derived_artifact_count": derivatives["derived_artifact_count"],
        "figure_count": figure_audit["figure_count"],
        "config_sha256": source_witnesses["config_sha256"],
        "execution_source_sha256": execution_witness,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the non-mutating fast-bundle audit from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Generated fast-bundle root; defaults to project output/.",
    )
    args = parser.parse_args(argv)
    print(json.dumps(audit_fast_bundle(args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
