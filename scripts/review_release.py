#!/usr/bin/env python3
"""Run a deterministic, read-only review of one release artifact bundle.

Unlike ``finalize_release.py``, this command never changes a manifest.  It
composes schema, source-witness, manifest, checkpoint, figure, dashboard,
resolved-manuscript, rendered-text, and clean-wheel-install checks into one
machine-readable report.  A report file is written only when ``--report`` is
explicitly requested.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from active_inference_power.figures import (  # noqa: E402
    FIGURE_NAMES,
    dashboard_result_lineage_hash,
    figure_metadata,
)
from active_inference_power.certificate_gate import effective_certificate_gate_result  # noqa: E402
from active_inference_power.schema import (  # noqa: E402
    validate_checkpoint,
    validate_figure_registry,
    validate_manifest,
    validate_quality_report,
    validate_results,
    validate_run_log,
)
from scripts.artifacts import (  # noqa: E402
    atomic_write_json,
    require_safe_output_root,
    sha256_file,
)
from scripts.audit_figure_accessibility import (  # noqa: E402
    audit_figure_accessibility,
    audit_rendered_figure_alts,
)
from scripts.audit_documentation_links import audit_documentation_links  # noqa: E402
from scripts.audit_manuscript_presentation import audit_manuscript_presentation  # noqa: E402
from scripts.generate_quality_report import validate_quality_input_provenance  # noqa: E402
from scripts.release_artifact_contracts import (  # noqa: E402
    validate_result_derivative_artifacts,
)
from scripts.release_provenance import (  # noqa: E402
    checkpoint_names_from_run_log,
    manifest_path,
    require_file,
    resolve_config_path,
    renderer_input_hash,
    required_release_artifacts,
    validate_artifact_inventory,
    validate_certificate_gate_witnesses,
    validate_checkpoint_provenance,
    validate_execution_provenance,
    validate_source_witnesses,
)
from scripts.renderer_provenance import (  # noqa: E402
    renderer_provenance_path,
    validate_renderer_provenance,
)


_TOKEN_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVIEW_RECEIPT_SCHEMA_VERSION = "1.1"
_FINAL_REVIEW_CHECKS = frozenset(
    {
        "schema_and_quality",
        "source_witnesses",
        "derived_artifacts",
        "manifest_hashes",
        "checkpoint_lineage",
        "figures_and_dashboard",
        "figure_accessibility",
        "documentation_links",
        "resolved_manuscript",
        "manuscript_presentation",
        "rendered_text",
        "rendered_figure_alt_text",
        "renderer_provenance",
        "clean_wheel_install",
    }
)


def _require_sha256(value: Any, name: str) -> str:
    """Return one strictly formatted digest used by a review receipt."""
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Hash receipt content without its redundant self-witness."""
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _indexed_review_checks(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Validate and index named review checks without trusting their order."""
    checks = report.get("checks")
    if not isinstance(checks, list):
        raise ValueError("release review report checks must be a list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in checks:
        if not isinstance(item, Mapping):
            raise ValueError("release review report contains a malformed check")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("release review report check name is missing")
        if name in indexed:
            raise ValueError(f"release review report duplicates check {name!r}")
        indexed[name] = item
    return indexed


def require_full_passing_review(report: Mapping[str, Any]) -> None:
    """Require all terminal read-only checks before publication state changes.

    A review run with ``--skip-render`` or ``--skip-clean-install`` remains
    useful during production, but it cannot authorize completion.  This
    explicit predicate is shared by the receipt writer and finalizer so a
    future optional check cannot accidentally downgrade the terminal path.
    """
    if report.get("passed") is not True:
        failures: list[str] = []
        for check in report.get("checks", []):
            if isinstance(check, Mapping) and check.get("passed") is not True:
                name = check.get("name", "unnamed")
                message = check.get("message")
                failures.append(f"{name}: {message}" if message else str(name))
        detail = "; ".join(failures) if failures else "report did not pass"
        raise ValueError(f"full read-only release review failed: {detail}")
    indexed = _indexed_review_checks(report)
    missing = sorted(_FINAL_REVIEW_CHECKS.difference(indexed))
    if missing:
        raise ValueError(f"full read-only release review is missing checks: {missing}")
    failed = sorted(
        name for name in _FINAL_REVIEW_CHECKS if indexed[name].get("passed") is not True
    )
    if failed:
        raise ValueError(f"full read-only release review has failed checks: {failed}")


def _review_receipt_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build a deterministic receipt from a passed full read-only review."""
    require_full_passing_review(report)
    indexed = _indexed_review_checks(report)

    def details(name: str) -> Mapping[str, Any]:
        value = indexed[name].get("details")
        if not isinstance(value, Mapping):
            raise ValueError(f"passed release review check {name!r} has no details")
        return value

    schema = details("schema_and_quality")
    run_id = schema.get("run_id")
    profile = schema.get("profile")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("passed release review does not identify its run")
    if profile not in {"release", "publication"}:
        raise ValueError("passed release review does not identify a publication profile")

    source = details("source_witnesses")
    resolved = details("resolved_manuscript")
    renderer = details("renderer_provenance")
    witnesses = {
        "config_sha256": _require_sha256(
            source.get("config_sha256"), "review configuration witness"
        ),
        "claim_ledger_sha256": _require_sha256(
            source.get("claim_ledger_sha256"), "review claim-ledger witness"
        ),
        "execution_source_sha256": _require_sha256(
            source.get("execution_source_sha256"), "review execution-source witness"
        ),
        "render_input_sha256": _require_sha256(
            resolved.get("render_input_sha256"), "review render-input witness"
        ),
        "renderer_provenance_sha256": _require_sha256(
            renderer.get("receipt_sha256"), "review renderer-provenance witness"
        ),
    }
    rendered = details("rendered_text").get("rendered_artifacts")
    if not isinstance(rendered, Mapping):
        raise ValueError("passed release review does not record rendered artifact digests")
    rendered_artifacts = {
        name: _require_sha256(rendered.get(name), f"review rendered {name} digest")
        for name in ("pdf", "html", "combined_manuscript")
    }
    receipt: dict[str, Any] = {
        "schema_version": _REVIEW_RECEIPT_SCHEMA_VERSION,
        "receipt_type": "release_review_receipt",
        "passed": True,
        "run_id": run_id,
        "profile": profile,
        "review_scope": {"require_render": True, "check_clean_install": True},
        "review_checks": sorted(_FINAL_REVIEW_CHECKS),
        "witnesses": witnesses,
        "rendered_artifacts": rendered_artifacts,
    }
    receipt["receipt_sha256"] = _canonical_receipt_sha256(receipt)
    return receipt


def validate_review_receipt(
    receipt: Mapping[str, Any],
    *,
    output: Path,
    expected_run_id: str,
    expected_profile: str,
    expected_witnesses: Mapping[str, str],
) -> dict[str, Any]:
    """Validate a persisted review receipt against the live release bundle."""
    if receipt.get("schema_version") != _REVIEW_RECEIPT_SCHEMA_VERSION:
        raise ValueError("review receipt schema version is invalid")
    if receipt.get("receipt_type") != "release_review_receipt":
        raise ValueError("review receipt type is invalid")
    if receipt.get("passed") is not True:
        raise ValueError("review receipt does not record a passing review")
    if receipt.get("run_id") != expected_run_id:
        raise ValueError("review receipt run identifier does not match the release")
    if receipt.get("profile") != expected_profile:
        raise ValueError("review receipt profile does not match the release")
    if receipt.get("review_scope") != {
        "require_render": True,
        "check_clean_install": True,
    }:
        raise ValueError("review receipt does not attest to rendered text and clean-wheel review")
    review_checks = receipt.get("review_checks")
    if (
        not isinstance(review_checks, list)
        or any(not isinstance(name, str) for name in review_checks)
        or set(review_checks) != _FINAL_REVIEW_CHECKS
    ):
        raise ValueError("review receipt does not list the full terminal review checks")
    if len(review_checks) != len(_FINAL_REVIEW_CHECKS):
        raise ValueError("review receipt duplicates terminal review checks")
    witnesses = receipt.get("witnesses")
    if not isinstance(witnesses, Mapping):
        raise ValueError("review receipt witnesses are missing")
    for name in (
        "config_sha256",
        "claim_ledger_sha256",
        "execution_source_sha256",
        "render_input_sha256",
    ):
        actual = _require_sha256(witnesses.get(name), f"review receipt {name}")
        expected = _require_sha256(expected_witnesses.get(name), f"expected {name}")
        if actual != expected:
            raise ValueError(f"review receipt {name} does not match the release witness")
    renderer_receipt = require_file(
        renderer_provenance_path(output), "renderer provenance receipt"
    )
    renderer_digest = _require_sha256(
        witnesses.get("renderer_provenance_sha256"),
        "review receipt renderer-provenance witness",
    )
    if renderer_digest != sha256_file(renderer_receipt):
        raise ValueError(
            "review receipt renderer-provenance digest does not match the bundle"
        )
    rendered = receipt.get("rendered_artifacts")
    if not isinstance(rendered, Mapping):
        raise ValueError("review receipt rendered artifact digests are missing")
    paths = {
        "pdf": output / "pdf" / "active_inference_power_combined.pdf",
        "html": output / "web" / "index.html",
        "combined_manuscript": output / "web" / "_combined_manuscript.md",
    }
    for name, path in paths.items():
        recorded = _require_sha256(rendered.get(name), f"review receipt rendered {name}")
        if recorded != sha256_file(require_file(path, f"reviewed rendered {name}")):
            raise ValueError(f"review receipt rendered {name} digest does not match the bundle")
    recorded_digest = _require_sha256(receipt.get("receipt_sha256"), "review receipt digest")
    if recorded_digest != _canonical_receipt_sha256(receipt):
        raise ValueError("review receipt digest does not match its content")
    return {
        "run_id": expected_run_id,
        "receipt_sha256": recorded_digest,
        "rendered_artifact_count": len(paths),
    }


def write_passing_review_receipt(output: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Persist only a passed full-review receipt; this never edits a manifest."""
    receipt = _review_receipt_from_report(report)
    safe_output = require_safe_output_root(output, project_root=ROOT)
    destination = safe_output / "review" / "release_review.json"
    atomic_write_json(destination, receipt)
    return {"path": destination, "receipt": receipt}


def _pdf_text_contains_witness(pdf_text: str, witness: str) -> bool:
    """Match an exact witness despite whitespace inserted by ``pdftotext``.

    Long hexadecimal values can be split at a PDF line boundary by text
    extraction.  Removing whitespace from the extracted text preserves every
    non-whitespace character, so this accepts only the exact witness rather
    than a reformatted or approximate value.
    """
    return bool(witness) and witness in re.sub(r"\s+", "", pdf_text)


def _resolve_output_root(output_root: Path | None) -> Path:
    output = ROOT / "output" if output_root is None else Path(output_root)
    if not output.is_absolute():
        output = ROOT / output
    return output.resolve()


def _check(
    checks: list[dict[str, Any]], name: str, action: Callable[[], dict[str, Any] | None]
) -> None:
    try:
        details = action() or {}
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        checks.append({"name": name, "passed": False, "message": str(exc)})
    else:
        checks.append({"name": name, "passed": True, "details": details})


def _load_bundle(output: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for name in ("manifest", "results", "run_log", "quality"):
        path = require_file(output / f"{name}.json", name.replace("_", " "))
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{name}.json must contain a JSON object")
        payloads[name] = value
    return payloads


def _verify_schema(bundle: dict[str, dict[str, Any]]) -> dict[str, Any]:
    validate_manifest(bundle["manifest"])
    validate_results(
        bundle["results"], require_suite=True, require_current_contract=True
    )
    validate_run_log(bundle["run_log"])
    quality = bundle["quality"]
    validate_quality_report(quality)
    quality_input_sha256 = validate_quality_input_provenance(quality, project_root=ROOT)
    if quality.get("quality_pass") is not True:
        raise ValueError("quality report does not pass the declared gate")
    if bundle["manifest"].get("run_id") != bundle["results"].get("run_id"):
        raise ValueError("manifest and results run identifiers differ")
    if bundle["run_log"].get("run_id") != bundle["results"].get("run_id"):
        raise ValueError("run log and results run identifiers differ")
    metadata = bundle["results"].get("meta")
    if not isinstance(metadata, dict):
        raise ValueError("release results metadata is missing")
    profile = metadata.get("profile")
    if profile not in {"release", "publication"}:
        raise ValueError("release review requires a release or publication profile")
    if bundle["run_log"].get("profile") != profile:
        raise ValueError("release results and run-log profiles differ")
    if bundle["run_log"].get("status") != "passed":
        raise ValueError(f"release run log is not passed: {bundle['run_log'].get('status')!r}")
    certificates = bundle["results"].get("certificates")
    if not isinstance(certificates, dict) or not certificates:
        raise ValueError("release results have no certificates")
    try:
        gate = effective_certificate_gate_result(
            certificates, metadata.get("certificate_gate")
        )
    except ValueError as exc:
        raise ValueError(f"release certificate gate is invalid: {exc}") from exc
    if not gate["terminal_certificates_pass"]:
        raise ValueError(
            "one or more terminal release certificates failed: "
            f"{gate['terminal_failures']}"
        )
    validate_certificate_gate_witnesses(
        bundle["results"], bundle["manifest"], bundle["run_log"]
    )
    if bundle["manifest"].get("certificates_pass") is not True:
        raise ValueError("manifest does not record passing terminal certificates")
    if bundle["run_log"].get("certificates_pass") is not True:
        raise ValueError("run log does not record passing terminal certificates")
    return {
        "run_id": bundle["results"]["run_id"],
        "profile": profile,
        "quality_input_sha256": quality_input_sha256,
    }


def _renderer_manifest_records_required(
    manifest: Mapping[str, Any], *, require_render: bool
) -> bool:
    """Require renderer records only after a manifest has completion authority."""
    return require_render and manifest.get("complete") is True


def _verify_manifest_records(
    output: Path, bundle: dict[str, dict[str, Any]], *, require_render: bool
) -> dict[str, Any]:
    manifest = bundle["manifest"]
    checkpoint_names = checkpoint_names_from_run_log(bundle["run_log"])
    # The sibling renderer owns the PDF/HTML records and the terminal receipt.
    # Before finalization those files may already exist, while their manifest
    # records intentionally do not.  A completed bundle, by contrast, must
    # carry the exact rendered and receipt inventory.
    renderer_records_required = _renderer_manifest_records_required(
        manifest, require_render=require_render
    )
    inventory = validate_artifact_inventory(
        manifest,
        required_release_artifacts(
            project_root=ROOT,
            output_root=output,
            figure_filenames=FIGURE_NAMES,
            checkpoint_names=checkpoint_names,
            require_render=renderer_records_required,
        ),
        project_root=ROOT,
        output_root=output,
    )
    count = 0
    for name, record_value in manifest.get("artifacts", {}).items():
        if not isinstance(record_value, dict):
            raise ValueError(f"manifest record is malformed: {name}")
        path = require_file(
            manifest_path(output, ROOT, record_value), f"manifest artifact {name}"
        )
        if sha256_file(path) != record_value.get("sha256") or path.stat().st_size != record_value.get("bytes"):
            raise ValueError(f"manifest digest or size mismatch for {name}")
        count += 1
    return {
        "artifact_count": count,
        "renderer_manifest_records_required": renderer_records_required,
        **inventory,
    }


def _verify_checkpoints(
    bundle: dict[str, dict[str, Any]], output: Path, *, config_path: Path
) -> dict[str, Any]:
    results = bundle["results"]
    provenance = validate_source_witnesses(
        bundle["manifest"],
        results,
        bundle["run_log"],
        project_root=ROOT,
        output_root=output,
        config_path=config_path,
        claim_ledger_path=ROOT / "data" / "claim_ledger.yaml",
    )
    execution_digest = validate_execution_provenance(
        bundle["manifest"], results, bundle["run_log"], project_root=ROOT
    )
    checkpoint_names = checkpoint_names_from_run_log(bundle["run_log"])
    checkpoints = sorted((output / "checkpoints").glob("*.json"))
    if {path.stem for path in checkpoints} != set(checkpoint_names):
        raise ValueError("checkpoint files do not match the terminal run-log stages")
    for path in checkpoints:
        value = json.loads(path.read_text(encoding="utf-8"))
        validate_checkpoint(value)
        validate_checkpoint_provenance(
            value,
            run_id=str(results["run_id"]),
            config_sha256=provenance["config_sha256"],
            claim_ledger_sha256=provenance["claim_ledger_sha256"],
            profile=str(results["meta"]["profile"]),
            stage=path.stem,
            execution_source_sha256=execution_digest,
        )
    return {"checkpoint_count": len(checkpoints), **provenance}


def _verify_figures_and_dashboard(output: Path, results: dict[str, Any]) -> dict[str, Any]:
    registry_path = require_file(output / "figures" / "figure_registry.json", "figure registry")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_figure_registry(registry, expected_filenames=set(FIGURE_NAMES))
    expected_metadata = figure_metadata(results)
    records = registry["figures"]
    for record in records:
        filename = str(record["filename"])
        name = Path(filename).stem
        require_file(output / "figures" / filename, f"figure {filename}")
        if record["metadata"]["result_lineage_hash"] != expected_metadata[name]["result_lineage_hash"]:
            raise ValueError(f"figure registry lineage is stale for {filename}")
    dashboard = require_file(output / "review" / "dashboard.html", "review dashboard")
    dashboard_text = dashboard.read_text(encoding="utf-8")
    for filename in FIGURE_NAMES:
        if filename not in dashboard_text:
            raise ValueError(f"dashboard does not link {filename}")
    if "validity boundary" not in dashboard_text.lower():
        raise ValueError("dashboard is missing validity-boundary metadata")
    expected_dashboard_lineage = dashboard_result_lineage_hash(results)
    if (
        f'<meta name="result-lineage-sha256" content="{expected_dashboard_lineage}">'
        not in dashboard_text
    ):
        raise ValueError("dashboard result-lineage digest does not match results")
    return {"figure_count": len(records), "dashboard": str(dashboard)}


def _verify_resolved_manuscript(
    output: Path, *, expected_witnesses: dict[str, str]
) -> dict[str, Any]:
    variables_path = require_file(
        output / "data" / "manuscript_variables.json", "manuscript variables"
    )
    variables = json.loads(variables_path.read_text(encoding="utf-8"))
    resolved = sorted((output / "manuscript").glob("*.md"))
    if not resolved:
        raise ValueError("resolved manuscript tree is empty")
    unresolved = [path.name for path in resolved if _TOKEN_RE.search(path.read_text(encoding="utf-8"))]
    if unresolved:
        raise ValueError(f"unresolved manuscript tokens remain: {unresolved}")
    witnesses = {
        "config_sha256": variables.get("RESULT_CONFIGURATION_HASH"),
        "claim_ledger_sha256": variables.get("RESULT_CLAIM_LEDGER_HASH"),
    }
    for name, value in witnesses.items():
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"manuscript variables omit a {name.replace('_', '-')} source witness")
        if value != expected_witnesses.get(name):
            raise ValueError(
                f"manuscript variables {name.replace('_', '-')} witness does not match source provenance"
            )
    render_input = variables.get("RESULT_RENDER_INPUT_HASH")
    if not isinstance(render_input, str) or len(render_input) != 64:
        raise ValueError("manuscript variables omit a render-input source witness")
    if render_input != renderer_input_hash(project_root=ROOT, output_root=output):
        raise ValueError("manuscript render-input witness does not match current renderer inputs")
    witnesses["render_input_sha256"] = render_input
    return {"resolved_file_count": len(resolved), **witnesses}


def _rendered_text(output: Path, *, expected_witnesses: dict[str, str]) -> dict[str, Any]:
    html = require_file(output / "web" / "index.html", "HTML render")
    combined = require_file(output / "web" / "_combined_manuscript.md", "combined rendered manuscript")
    pdf = require_file(output / "pdf" / "active_inference_power_combined.pdf", "PDF render")
    rendered_text = {
        "HTML render": html.read_text(encoding="utf-8"),
        "combined rendered manuscript": combined.read_text(encoding="utf-8"),
    }
    for description, text in rendered_text.items():
        if _TOKEN_RE.search(text):
            raise ValueError(f"{description} contains unresolved manuscript tokens")
        for name, value in expected_witnesses.items():
            if value not in text:
                raise ValueError(
                    f"{description} is stale relative to the hydrated "
                    f"{name.replace('_', '-')} witness"
                )
    process = subprocess.run(
        ["pdftotext", str(pdf), "-"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if process.returncode != 0:
        raise ValueError(f"pdftotext failed: {process.stderr.strip()}")
    for name, value in expected_witnesses.items():
        if not _pdf_text_contains_witness(process.stdout, value):
            raise ValueError(f"PDF render is stale relative to the hydrated {name.replace('_', '-')} witness")
    newest_input = max(
        [path.stat().st_mtime for path in (output / "manuscript").glob("*.md")]
        + [path.stat().st_mtime for path in (output / "figures").glob("*.png")]
    )
    stale = [str(path) for path in (html, combined, pdf) if path.stat().st_mtime < newest_input]
    if stale:
        raise ValueError(f"rendered artifacts are older than their manuscript or figure inputs: {stale}")
    return {
        "pdf": str(pdf),
        "html": str(html),
        "rendered_text_witnesses": expected_witnesses,
        "rendered_artifacts": {
            "pdf": sha256_file(pdf),
            "html": sha256_file(html),
            "combined_manuscript": sha256_file(combined),
        },
    }


def _verify_persisted_review_receipt(
    output: Path,
    *,
    bundle: Mapping[str, Mapping[str, Any]],
    source_witnesses: Mapping[str, str],
    resolved: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a complete bundle's receipt to its current review inputs."""
    path = require_file(output / "review" / "release_review.json", "release review receipt")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("release review receipt is not valid JSON") from exc
    if not isinstance(receipt, Mapping):
        raise ValueError("release review receipt must be a JSON object")
    metadata = bundle["results"].get("meta")
    profile = metadata.get("profile") if isinstance(metadata, Mapping) else None
    if not isinstance(profile, str):
        raise ValueError("release results metadata is missing a profile")
    return validate_review_receipt(
        receipt,
        output=output,
        expected_run_id=str(bundle["results"].get("run_id", "")),
        expected_profile=profile,
        expected_witnesses={
            "config_sha256": str(source_witnesses.get("config_sha256", "")),
            "claim_ledger_sha256": str(source_witnesses.get("claim_ledger_sha256", "")),
            "execution_source_sha256": str(
                source_witnesses.get("execution_source_sha256", "")
            ),
            "render_input_sha256": str(resolved.get("render_input_sha256", "")),
        },
    )


def _clean_wheel_install(project_root: Path) -> dict[str, Any]:
    """Build and import the wheel from a temporary, source-free location."""
    with tempfile.TemporaryDirectory(prefix="active-inference-power-review-") as name:
        temporary = Path(name)
        dist = temporary / "dist"
        environment = os.environ.copy()
        environment.pop("VIRTUAL_ENV", None)
        build = subprocess.run(
            ["uv", "build", "--out-dir", str(dist)],
            cwd=project_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if build.returncode != 0:
            raise ValueError(f"wheel build failed: {build.stderr.strip()[-1000:]}")
        wheels = sorted(dist.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("wheel build did not produce exactly one wheel")
        site = temporary / "site"
        install = subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(site),
                "--python",
                sys.executable,
                str(wheels[0]),
            ],
            cwd=temporary,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if install.returncode != 0:
            raise ValueError(f"wheel installation failed: {install.stderr.strip()[-1000:]}")
        check_source = (
            "import importlib, pathlib, sys; "
            "site = pathlib.Path(sys.argv[1]).resolve(); sys.path.insert(0, str(site)); "
            "module = importlib.import_module('active_inference_power'); "
            "location = pathlib.Path(module.__file__).resolve(); "
            "assert location.is_relative_to(site), location; print(location)"
        )
        imported = subprocess.run(
            [sys.executable, "-I", "-c", check_source, str(site)],
            cwd=temporary,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if imported.returncode != 0:
            raise ValueError(f"clean wheel import failed: {imported.stderr.strip()[-1000:]}")
        return {"wheel": wheels[0].name, "import_path": imported.stdout.strip()}


def review_release(
    output_root: Path | None = None,
    *,
    require_render: bool = True,
    check_clean_install: bool = True,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Return a complete read-only release-review report."""
    output = _resolve_output_root(output_root)
    checks: list[dict[str, Any]] = []
    source_witnesses: dict[str, str] = {}
    try:
        source_config = resolve_config_path(config_path, project_root=ROOT)
    except ValueError as exc:
        return {
            "schema_version": "1.0",
            "passed": False,
            "output_root": str(output),
            "checks": [
                {"name": "source_configuration", "passed": False, "message": str(exc)}
            ],
        }
    try:
        bundle = _load_bundle(output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "schema_version": "1.0",
            "passed": False,
            "output_root": str(output),
            "checks": [{"name": "load_bundle", "passed": False, "message": str(exc)}],
        }
    _check(checks, "schema_and_quality", lambda: _verify_schema(bundle))
    def check_source_witnesses() -> dict[str, str]:
        nonlocal source_witnesses
        source_witnesses = validate_source_witnesses(
            bundle["manifest"],
            bundle["results"],
            bundle["run_log"],
            project_root=ROOT,
            output_root=output,
            config_path=source_config,
            claim_ledger_path=ROOT / "data" / "claim_ledger.yaml",
        )
        source_witnesses["execution_source_sha256"] = validate_execution_provenance(
            bundle["manifest"], bundle["results"], bundle["run_log"], project_root=ROOT
        )
        return source_witnesses

    _check(checks, "source_witnesses", check_source_witnesses)
    _check(
        checks,
        "documentation_links",
        lambda: _require_passing_documentation_link_audit(ROOT),
    )
    _check(
        checks,
        "derived_artifacts",
        lambda: validate_result_derivative_artifacts(output, bundle["results"]),
    )
    _check(
        checks,
        "manifest_hashes",
        lambda: _verify_manifest_records(
            output, bundle, require_render=require_render
        ),
    )
    _check(
        checks,
        "checkpoint_lineage",
        lambda: _verify_checkpoints(bundle, output, config_path=source_config),
    )
    _check(checks, "figures_and_dashboard", lambda: _verify_figures_and_dashboard(output, bundle["results"]))
    _check(checks, "figure_accessibility", lambda: _require_passing_figure_audit(output))
    resolved: dict[str, Any] = {}

    def check_resolved() -> dict[str, Any]:
        nonlocal resolved
        resolved = _verify_resolved_manuscript(
            output, expected_witnesses=source_witnesses
        )
        return resolved

    _check(checks, "resolved_manuscript", check_resolved)
    _check(
        checks,
        "manuscript_presentation",
        lambda: _require_passing_manuscript_presentation_audit(ROOT / "manuscript", output),
    )
    if require_render:
        _check(
            checks,
            "rendered_text",
            lambda: _rendered_text(
                output,
                expected_witnesses={
                    "config_sha256": str(resolved.get("config_sha256", "")),
                    "claim_ledger_sha256": str(resolved.get("claim_ledger_sha256", "")),
                    "render_input_sha256": str(
                        resolved.get("render_input_sha256", "")
                    ),
                },
            ),
        )
        _check(
            checks,
            "rendered_figure_alt_text",
            lambda: _require_passing_rendered_figure_alt_audit(output),
        )
        _check(
            checks,
            "renderer_provenance",
            lambda: validate_renderer_provenance(
                project_root=ROOT,
                output_root=output,
            ),
        )
        if bundle["manifest"].get("complete") is True:
            _check(
                checks,
                "review_receipt",
                lambda: _verify_persisted_review_receipt(
                    output,
                    bundle=bundle,
                    source_witnesses=source_witnesses,
                    resolved=resolved,
                ),
            )
    if check_clean_install:
        _check(checks, "clean_wheel_install", lambda: _clean_wheel_install(ROOT))
    return {
        "schema_version": "1.0",
        "passed": all(bool(check["passed"]) for check in checks),
        "output_root": str(output),
        "checks": checks,
    }


def _require_passing_figure_audit(output: Path) -> dict[str, Any]:
    report = audit_figure_accessibility(output)
    if not report["passed"]:
        raise ValueError(f"figure accessibility audit failed: {report['findings']}")
    return {"figure_count": report["figure_count"]}


def _require_passing_documentation_link_audit(project_root: Path) -> dict[str, Any]:
    """Require every bounded local Markdown destination to resolve read-only."""
    report = audit_documentation_links(project_root)
    if report["passed"] is not True:
        raise ValueError(f"documentation-link audit failed: {report['findings']}")
    return {
        "document_count": report["document_count"],
        "local_link_count": report["local_link_count"],
    }


def _require_passing_rendered_figure_alt_audit(output: Path) -> dict[str, Any]:
    """Require combined and retained per-section HTML to retain authored alts."""
    report = audit_rendered_figure_alts(output, include_auxiliary_pages=True)
    if not report["passed"]:
        raise ValueError(f"rendered figure alt-text audit failed: {report['findings']}")
    return {
        "figure_count": report["figure_count"],
        "rendered_html": report["rendered_html"],
        "auxiliary_page_count": report["auxiliary_page_count"],
    }


def _require_passing_manuscript_presentation_audit(
    manuscript_dir: Path, output: Path
) -> dict[str, Any]:
    """Require source and hydrated reader-facing manuscript checks to be clean."""
    report = audit_manuscript_presentation(
        manuscript_dir,
        output / "data" / "manuscript_variables.json",
        resolved_manuscript_dir=output / "manuscript",
    )
    findings = report["findings"]
    if findings:
        raise ValueError(f"manuscript presentation audit failed: {findings}")
    return {
        "source_file_count": len(report["files_checked"]),
        "resolved_file_count": len(report["resolved_files_checked"]),
        "finding_count": int(report["summary"]["finding_count"]),
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
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-clean-install", action="store_true")
    parser.add_argument("--report", type=Path, default=None, help="Optional explicit JSON report destination.")
    args = parser.parse_args()
    report = review_release(
        args.output_root,
        require_render=not args.skip_render,
        check_clean_install=not args.skip_clean_install,
        config_path=args.config,
    )
    if args.report is not None:
        atomic_write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
