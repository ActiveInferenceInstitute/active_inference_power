#!/usr/bin/env python3
"""Fail-safe finalizer for a complete deterministic research release.

The simulation orchestrator writes a manifest before manuscript hydration can
finish.  This finalizer is the last gate: it validates every reader-facing
artifact, verifies every recorded digest and size, and only then atomically
marks the manifest complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from active_inference_power.figures import (
    FIGURE_NAMES,
    dashboard_result_lineage_hash,
    figure_metadata,
)
from active_inference_power.certificate_gate import effective_certificate_gate_result
from active_inference_power.schema import (
    validate_benchmark_manifest,
    validate_checkpoint,
    validate_figure_registry,
    validate_manifest,
    validate_quality_report,
    validate_results,
    validate_run_log,
)


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.artifacts import (  # noqa: E402
    artifact_record,
    atomic_write_json,
    refresh_renderer_artifact_records,
    require_safe_output_root,
    sha256_file,
)
from scripts.audit_figure_accessibility import audit_figure_accessibility  # noqa: E402
from scripts.generate_quality_report import validate_quality_input_provenance  # noqa: E402
from scripts.release_artifact_contracts import (  # noqa: E402
    validate_result_derivative_artifacts,
)
from scripts.review_release import (  # noqa: E402
    require_full_passing_review,
    review_release,
    validate_review_receipt,
    write_passing_review_receipt,
)
from scripts.renderer_provenance import (  # noqa: E402
    renderer_provenance_path,
    validate_renderer_provenance,
    write_renderer_provenance,
)
from scripts.release_provenance import (  # noqa: E402
    checkpoint_names_from_run_log,
    manifest_path as _manifest_path,
    require_file as _require_file,
    resolve_config_path,
    renderer_input_hash,
    required_release_artifacts,
    validate_artifact_inventory,
    validate_certificate_gate_witnesses,
    validate_checkpoint_provenance,
    validate_execution_provenance,
    validate_source_witnesses,
)


_TOKEN_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")


def _pdf_text_contains_witness(pdf_text: str, witness: str) -> bool:
    """Match an exact witness despite whitespace inserted by ``pdftotext``.

    Long hexadecimal values can be split at a PDF line boundary by text
    extraction.  Removing whitespace from the extracted text preserves every
    non-whitespace character, so this accepts only the exact witness rather
    than a reformatted or approximate value.
    """
    return bool(witness) and witness in re.sub(r"\s+", "", pdf_text)


def _validate_figure_registry(output: Path, results: dict[str, Any]) -> None:
    registry_path = _require_file(output / "figures" / "figure_registry.json", "figure registry")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_figure_registry(registry, expected_filenames=set(FIGURE_NAMES))
    expected = {Path(name).stem for name in FIGURE_NAMES}
    records = registry["figures"]
    if {Path(str(item.get("filename", ""))).stem for item in records} != expected:
        raise ValueError("figure registry does not enumerate exactly the configured figures")
    if len(records) != len(expected):
        raise ValueError("figure registry contains duplicate or missing figure records")
    expected_metadata = figure_metadata(results)
    for item in records:
        if not all(isinstance(item.get(key), str) and item[key] for key in ("filename", "label", "caption", "alt")):
            raise ValueError("every figure requires a filename, label, caption, and alt text")
        _require_file(output / "figures" / item["filename"], f"figure {item['filename']}")
        name = Path(str(item["filename"])).stem
        if item.get("metadata", {}).get("result_lineage_hash") != expected_metadata[name]["result_lineage_hash"]:
            raise ValueError(f"figure registry lineage does not match results for {item['filename']}")
    dashboard = _require_file(output / "review" / "dashboard.html", "review dashboard")
    dashboard_text = dashboard.read_text(encoding="utf-8")
    for filename in expected:
        if f"{filename}.png" not in dashboard_text:
            raise ValueError(f"dashboard does not link figure {filename}.png")
    if "validity boundary" not in dashboard_text.lower():
        raise ValueError("dashboard is missing validity-boundary metadata")
    expected_dashboard_lineage = dashboard_result_lineage_hash(results)
    if (
        f'<meta name="result-lineage-sha256" content="{expected_dashboard_lineage}">'
        not in dashboard_text
    ):
        raise ValueError("dashboard result-lineage digest does not match results")
    if not results.get("suite"):
        raise ValueError("results suite block is required for publication finalization")


def _validate_figure_accessibility(output: Path) -> None:
    """Require every reader-facing PNG to decode and meet its metadata contract."""
    report = audit_figure_accessibility(output)
    if not report["passed"]:
        raise ValueError(f"figure accessibility audit failed: {report['findings']}")


def _validate_manuscript(
    output: Path, *, expected_witnesses: dict[str, str]
) -> dict[str, str]:
    variables_path = _require_file(
        output / "data" / "manuscript_variables.json", "manuscript variables"
    )
    variables = json.loads(variables_path.read_text(encoding="utf-8"))
    if not isinstance(variables, dict):
        raise ValueError("manuscript variables must be a JSON object")
    witnesses = {
        "config_sha256": variables.get("RESULT_CONFIGURATION_HASH"),
        "claim_ledger_sha256": variables.get("RESULT_CLAIM_LEDGER_HASH"),
    }
    for name, value in witnesses.items():
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(
                f"manuscript variables omit a {name.replace('_', '-')} source witness"
            )
        if value != expected_witnesses.get(name):
            raise ValueError(
                f"manuscript variables {name.replace('_', '-')} witness does not match the release provenance"
            )
    render_input = variables.get("RESULT_RENDER_INPUT_HASH")
    if not isinstance(render_input, str) or len(render_input) != 64:
        raise ValueError("manuscript variables omit a render-input source witness")
    witnesses["render_input_sha256"] = render_input
    resolved = sorted((output / "manuscript").glob("*.md"))
    if not resolved:
        raise ValueError("resolved manuscript tree is empty")
    unresolved = [
        str(path)
        for path in resolved
        if _TOKEN_RE.search(path.read_text(encoding="utf-8"))
    ]
    if unresolved:
        raise ValueError(f"unresolved manuscript tokens remain: {unresolved}")
    return {name: str(value) for name, value in witnesses.items()}


def _rendered_record(path: Path, output: Path, role: str) -> dict[str, Any]:
    """Record a renderer output in its safe project/output namespace."""
    path = Path(path).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        return artifact_record(path, output, role, root_label="output")
    return artifact_record(path, ROOT, role, root_label="project")


def _validate_rendered_bundle(
    output: Path, witnesses: dict[str, str]
) -> dict[str, Path]:
    """Reject a PDF/HTML bundle predating or disagreeing with its inputs."""
    html = _require_file(output / "web" / "index.html", "HTML render")
    combined = _require_file(
        output / "web" / "_combined_manuscript.md", "combined rendered manuscript"
    )
    pdf = _require_file(
        output / "pdf" / "active_inference_power_combined.pdf", "PDF render"
    )
    rendered_text = {
        "HTML render": html.read_text(encoding="utf-8"),
        "combined rendered manuscript": combined.read_text(encoding="utf-8"),
    }
    for description, value in rendered_text.items():
        if _TOKEN_RE.search(value):
            raise ValueError(f"{description} contains unresolved manuscript tokens")
        for name, witness in witnesses.items():
            if witness not in value:
                raise ValueError(
                    f"{description} is stale relative to the hydrated "
                    f"{name.replace('_', '-')} witness"
                )
    text = subprocess.run(
        ["pdftotext", str(pdf), "-"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if text.returncode != 0:
        raise ValueError(f"pdftotext failed: {text.stderr.strip()}")
    for name, value in witnesses.items():
        if not _pdf_text_contains_witness(text.stdout, value):
            raise ValueError(
                f"rendered PDF is stale relative to the hydrated {name.replace('_', '-')} witness"
            )
    inputs = [*sorted((output / "manuscript").glob("*.md")), *sorted((output / "figures").glob("*.png"))]
    if not inputs:
        raise ValueError("render freshness cannot be established without manuscript and figure inputs")
    newest_input = max(path.stat().st_mtime_ns for path in inputs)
    stale = [
        str(path)
        for path in (html, combined, pdf)
        if path.stat().st_mtime_ns < newest_input
    ]
    if stale:
        raise ValueError(
            f"rendered artifacts are older than their manuscript or figure inputs: {stale}"
        )
    return {"pdf": pdf, "html": html, "combined": combined}


def _validate_release_state(
    results: dict[str, Any],
    manifest: dict[str, Any],
    run_log: dict[str, Any],
    quality: dict[str, Any],
) -> str:
    """Reject non-publication profiles and inconsistent terminal evidence."""
    metadata = results.get("meta")
    if not isinstance(metadata, dict):
        raise ValueError("release results metadata is missing")
    profile = metadata.get("profile")
    if profile not in {"release", "publication"}:
        raise ValueError("release finalization requires a release or publication profile")
    if run_log.get("profile") != profile:
        raise ValueError("release results and run-log profiles differ")
    certificates = results.get("certificates")
    if not isinstance(certificates, dict) or not certificates:
        raise ValueError("release results have no certificates")
    serialized_gate = metadata.get("certificate_gate")
    try:
        gate = effective_certificate_gate_result(certificates, serialized_gate)
    except ValueError as exc:
        raise ValueError(f"release certificate gate is invalid: {exc}") from exc
    if not gate["terminal_certificates_pass"]:
        raise ValueError(
            "one or more terminal release certificates failed: "
            f"{gate['terminal_failures']}"
        )
    validate_certificate_gate_witnesses(results, manifest, run_log)
    if manifest.get("certificates_pass") is not True:
        raise ValueError("manifest does not record passing terminal certificates")
    if run_log.get("certificates_pass") is not True:
        raise ValueError("run log does not record passing terminal certificates")
    if run_log.get("status") != "passed":
        raise ValueError(f"release run log is not passed: {run_log.get('status')!r}")
    if quality.get("quality_pass") is not True:
        raise ValueError("quality report does not pass the declared release gate")
    return str(profile)


def _rendered_artifacts_exist(output: Path) -> bool:
    """Detect whether a pre-final render exists before recording its receipt.

    A missing render is left for the read-only review to report alongside its
    other unmet gates.  When all required files exist, provenance capture is
    mandatory and any template/toolchain failure stops finalization.
    """
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in (
            output / "pdf" / "active_inference_power_combined.pdf",
            output / "web" / "index.html",
            output / "web" / "_combined_manuscript.md",
        )
    )


def finalize_release(
    output_root: Path | None = None,
    *,
    require_render: bool = True,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Validate and atomically complete one release manifest."""
    if not require_render:
        raise ValueError(
            "finalize_release only completes rendered publication bundles; use scripts/review_release.py for an unrendered review"
        )
    source_config = resolve_config_path(config_path, project_root=ROOT)
    output = (ROOT / "output") if output_root is None else Path(output_root)
    if not output.is_absolute():
        output = ROOT / output
    output = require_safe_output_root(output, project_root=ROOT)
    # Let a missing physical render be reported by the ordinary read-only
    # review.  Once a render exists, however, the renderer/template/toolchain
    # receipt is a mandatory post-render input to that review.
    if _rendered_artifacts_exist(output):
        write_renderer_provenance(project_root=ROOT, output_root=output)
    # Review before adding any renderer-owned records or changing completion.
    # An incomplete manifest is an expected pre-final state: review_release
    # still verifies the physical render, but only a completed manifest is
    # required to enumerate renderer-owned records and the prior receipt.
    review_report = review_release(
        output,
        require_render=True,
        check_clean_install=True,
        config_path=source_config,
    )
    require_full_passing_review(review_report)
    manifest_path = _require_file(output / "manifest.json", "release manifest")
    results_path = _require_file(output / "results.json", "results")
    run_log_path = _require_file(output / "run_log.json", "run log")
    quality_path = _require_file(output / "quality.json", "quality report")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    validate_results(results, require_suite=True, require_current_contract=True)
    validate_run_log(run_log)
    validate_manifest(manifest)
    validate_quality_report(quality)
    validate_quality_input_provenance(quality, project_root=ROOT)
    derived_artifacts = validate_result_derivative_artifacts(output, results)
    if manifest.get("run_id") != results.get("run_id") or run_log.get("run_id") != results.get("run_id"):
        raise ValueError("manifest, results, and run log identifiers differ")
    provenance = validate_source_witnesses(
        manifest,
        results,
        run_log,
        project_root=ROOT,
        output_root=output,
        config_path=source_config,
        claim_ledger_path=ROOT / "data" / "claim_ledger.yaml",
    )
    execution_digest = validate_execution_provenance(
        manifest, results, run_log, project_root=ROOT
    )
    profile = _validate_release_state(results, manifest, run_log, quality)
    benchmark_manifest_path = ROOT / "data" / "benchmarks" / "manifest.yaml"
    if benchmark_manifest_path.is_file():
        import yaml

        benchmark_payload = yaml.safe_load(benchmark_manifest_path.read_text(encoding="utf-8"))
        validate_benchmark_manifest(benchmark_payload)
        for record in benchmark_payload["benchmarks"]:
            fixture = ROOT / "data" / "benchmarks" / record["filename"]
            if fixture.is_file() and sha256_file(fixture) != record["sha256"]:
                raise ValueError(f"benchmark fixture checksum mismatch: {fixture}")
    checkpoint_names = checkpoint_names_from_run_log(run_log)
    checkpoints = sorted((output / "checkpoints").glob("*.json"))
    if {checkpoint.stem for checkpoint in checkpoints} != set(checkpoint_names):
        raise ValueError("checkpoint files do not match the terminal run-log stages")
    for checkpoint in checkpoints:
        checkpoint_value = json.loads(checkpoint.read_text(encoding="utf-8"))
        validate_checkpoint(checkpoint_value)
        validate_checkpoint_provenance(
            checkpoint_value,
            run_id=str(results["run_id"]),
            config_sha256=provenance["config_sha256"],
            claim_ledger_sha256=provenance["claim_ledger_sha256"],
            profile=profile,
            stage=checkpoint.stem,
            execution_source_sha256=execution_digest,
        )
    _validate_figure_registry(output, results)
    _validate_figure_accessibility(output)
    witnesses = _validate_manuscript(output, expected_witnesses=provenance)
    if (
        witnesses["render_input_sha256"]
        != renderer_input_hash(project_root=ROOT, output_root=output)
    ):
        raise ValueError("manuscript render-input witness does not match current renderer inputs")
    rendered = _validate_rendered_bundle(output, witnesses)
    receipt_result = write_passing_review_receipt(output, review_report)
    receipt_path = Path(receipt_result["path"])
    receipt = receipt_result["receipt"]
    if not isinstance(receipt, dict):
        raise ValueError("review receipt writer returned a malformed receipt")
    receipt_validation = validate_review_receipt(
        receipt,
        output=output,
        expected_run_id=str(results["run_id"]),
        expected_profile=profile,
        expected_witnesses={
            "config_sha256": provenance["config_sha256"],
            "claim_ledger_sha256": provenance["claim_ledger_sha256"],
            "execution_source_sha256": execution_digest,
            "render_input_sha256": witnesses["render_input_sha256"],
        },
    )
    # Re-capture validation after the full review and its own receipt write so
    # a concurrent sibling-template/toolchain/render change cannot be bound by
    # an earlier passing review alone.
    renderer_provenance_validation = validate_renderer_provenance(
        project_root=ROOT,
        output_root=output,
    )
    artifacts = manifest.setdefault("artifacts", {})
    artifacts["rendered_pdf"] = _rendered_record(
        rendered["pdf"], output, "rendered_pdf"
    )
    artifacts["rendered_html"] = _rendered_record(
        rendered["html"], output, "rendered_html"
    )
    artifacts["web:index.html"] = _rendered_record(
        rendered["html"], output, "rendered_html"
    )
    artifacts["web:_combined_manuscript.md"] = _rendered_record(
        rendered["combined"], output, "rendered_html"
    )
    artifacts["release_review"] = _rendered_record(
        receipt_path, output, "release_review_receipt"
    )
    artifacts["renderer_provenance"] = _rendered_record(
        renderer_provenance_path(output), output, "renderer_provenance_receipt"
    )

    refresh_renderer_artifact_records(manifest, output, ROOT)

    inventory = validate_artifact_inventory(
        manifest,
        required_release_artifacts(
            project_root=ROOT,
            output_root=output,
            figure_filenames=FIGURE_NAMES,
            checkpoint_names=checkpoint_names,
            require_render=True,
        ),
        project_root=ROOT,
        output_root=output,
    )

    # Verify every record in its declared project or output namespace before
    # publishing the new terminal state; no scientific file is rewritten here.
    for name, record in manifest.get("artifacts", {}).items():
        path = _manifest_path(output, ROOT, record)
        _require_file(path, f"manifest artifact {name}")
        if sha256_file(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"manifest digest or size mismatch for {name}")
    manifest["complete"] = True
    manifest.pop("validation_pending", None)
    validate_manifest(manifest)
    atomic_write_json(manifest_path, manifest)
    return {
        "complete": True,
        "run_id": manifest["run_id"],
        "artifact_count": len(manifest["artifacts"]),
        **inventory,
        "quality_pass": bool(quality["quality_pass"]),
        "render_checked": True,
        "review_receipt_sha256": receipt_validation["receipt_sha256"],
        "renderer_provenance_sha256": renderer_provenance_validation["receipt_sha256"],
        **derived_artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Exact preregistered configuration source; defaults to manuscript/config.yaml.",
    )
    parser.add_argument(
        "--allow-unrendered-review",
        action="store_true",
        help="Validate an incomplete review bundle without marking it publication-complete.",
    )
    args = parser.parse_args()
    if args.allow_unrendered_review:
        raise ValueError(
            "finalize_release only completes rendered publication bundles; use scripts/review_release.py for an unrendered review"
        )
    print(
        json.dumps(
            finalize_release(
                args.output_root, require_render=True, config_path=args.config
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
