#!/usr/bin/env python3
"""Refresh figures, captions, dashboard, and their manifest records.

This command deliberately reuses the existing deterministic ``results.json``;
it never reruns simulation stages. All visual files are first rendered in a
same-filesystem staging directory, then atomically replaced, and the manifest
is rewritten last with fresh hashes. Use it after source-level figure or
caption changes when the scientific result artifact is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from active_inference_power.figures import (  # noqa: E402
    FIGURE_NAMES,
    generate_figures,
    write_dashboard,
    write_figure_registry,
)
from active_inference_power.schema import validate_manifest, validate_results  # noqa: E402
from scripts.artifacts import (  # noqa: E402
    artifact_record,
    atomic_write_json,
    invalidate_renderer_artifact_records,
    require_safe_output_root,
)
from scripts.release_provenance import validate_source_witnesses  # noqa: E402


def _resolve_output(root: Path | None) -> Path:
    output = Path(root) if root is not None else ROOT / "output"
    if not output.is_absolute():
        output = ROOT / output
    output = require_safe_output_root(output, project_root=ROOT)
    if not output.is_dir():
        raise ValueError(f"output_root does not exist: {output}")
    return output


def _artifact(path: Path, output: Path, role: str) -> dict[str, Any]:
    """Record a refreshed artifact in its project or output namespace."""
    try:
        path.resolve().relative_to(ROOT)
    except ValueError:
        return artifact_record(path, output, role, root_label="output")
    return artifact_record(path, ROOT, role, root_label="project")


def _replace_staged_files(staging: Path, output: Path) -> dict[str, str]:
    staged_figures = staging / "figures"
    final_figures = output / "figures"
    final_figures.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}
    for key, filename in zip(
        (name.removesuffix(".png") for name in FIGURE_NAMES), FIGURE_NAMES
    ):
        source = staged_figures / filename
        destination = final_figures / filename
        os.replace(source, destination)
        figures[key] = str(destination)
    os.replace(staged_figures / "figure_registry.json", final_figures / "figure_registry.json")
    os.replace(staging / "review" / "dashboard.html", output / "review" / "dashboard.html")
    return figures


def _invalidate_release_completion(manifest: dict[str, Any]) -> list[str]:
    """Require downstream hydration, rendering, and finalization after a refresh."""
    invalidate_renderer_artifact_records(manifest)
    pending = ["manuscript_hydration", "publication_render", "release_finalization"]
    manifest["complete"] = False
    manifest["validation_pending"] = pending
    return pending


def refresh(output_root: Path | None = None) -> dict[str, Any]:
    """Regenerate the visual artifact set from an existing result artifact."""
    output = _resolve_output(output_root)
    results_path = output / "results.json"
    manifest_path = output / "manifest.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_log = json.loads((output / "run_log.json").read_text(encoding="utf-8"))
    validate_results(results)
    validate_manifest(manifest)
    if manifest["run_id"] != results["run_id"]:
        raise ValueError("manifest and results run identifiers differ")
    if manifest["config_sha256"] != results["meta"]["config_sha256"]:
        raise ValueError("manifest and results configuration hashes differ")
    validate_source_witnesses(
        manifest,
        results,
        run_log,
        project_root=ROOT,
        output_root=output,
        config_path=ROOT / "manuscript" / "config.yaml",
        claim_ledger_path=ROOT / "data" / "claim_ledger.yaml",
    )
    # Persist an incomplete state before replacing any PNG. A process failure
    # cannot leave a complete manifest covering a partially refreshed figure
    # set or stale renderer output.
    pending = _invalidate_release_completion(manifest)
    validate_manifest(manifest)
    atomic_write_json(manifest_path, manifest)

    with tempfile.TemporaryDirectory(prefix=".figure-refresh-", dir=output) as name:
        staging = Path(name)
        staged_figures = generate_figures(results, staging / "figures")
        write_figure_registry(staging / "figures", staged_figures, results)
        write_dashboard(results, staging / "review" / "dashboard.html", staged_figures)
        figures = _replace_staged_files(staging, output)

    registry_path = output / "figures" / "figure_registry.json"
    dashboard_path = output / "review" / "dashboard.html"
    artifacts = manifest.setdefault("artifacts", {})
    artifacts["figure_registry"] = _artifact(registry_path, output, "figure_registry")
    artifacts["dashboard"] = _artifact(dashboard_path, output, "review_dashboard")
    for key, path in figures.items():
        artifacts[f"figure:{key}"] = _artifact(Path(path), output, "publication_figure")
    validate_manifest(manifest)
    atomic_write_json(manifest_path, manifest)
    return {
        "run_id": results["run_id"],
        "figure_count": len(figures),
        "manifest": str(manifest_path),
        "validation_pending": pending,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Existing output directory; it may be outside the project root (default: output).",
    )
    args = parser.parse_args()
    print(json.dumps(refresh(args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
