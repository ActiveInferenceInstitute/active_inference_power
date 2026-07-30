#!/usr/bin/env python3
"""Thin orchestrator: generate and inject manuscript variables.

Reads ``manuscript/config.yaml`` and the selected output root's
``results.json``, writes ``data/manuscript_variables.json`` there, and
substitutes ``{{TOKEN}}`` markers in ``manuscript/*.md`` into the selected
output root's ``manuscript/`` tree for PDF rendering.

All computation lives in ``active_inference_power/manuscript_variables``; all injection lives in
``infrastructure.rendering.manuscript_injection``.

Exit codes:
    0   variables written and injected successfully
    1   unexpected error (e.g. missing analysis outputs without --allow-draft)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT.parents[2]))  # repo root, for infrastructure.*
from scripts.artifacts import (  # noqa: E402
    artifact_record,
    atomic_write_json,
    atomic_write_text,
    invalidate_renderer_artifact_records,
    require_safe_output_root,
    refresh_renderer_artifact_records,
    sha256_file,
)
from scripts.generate_quality_report import validate_quality_input_provenance  # noqa: E402
from scripts.release_provenance import (  # noqa: E402
    manifest_path as _manifest_path,
    manuscript_source_paths,
    renderer_input_hash,
    validate_execution_provenance,
    validate_source_witnesses,
)
from active_inference_power.schema import (  # noqa: E402
    validate_benchmark_manifest,
    validate_checkpoint,
    validate_claim_ledger,
    validate_manifest,
    validate_quality_report,
    validate_results,
    validate_run_log,
)


def _resolve_output_root(output_root: Path | None) -> Path:
    output = _PROJECT_ROOT / "output" if output_root is None else Path(output_root)
    if not output.is_absolute():
        output = _PROJECT_ROOT / output
    return require_safe_output_root(output, project_root=_PROJECT_ROOT)


def _artifact_for_output(path: Path, output: Path, role: str) -> dict[str, Any]:
    """Record project sources and external output files in safe namespaces."""
    path = Path(path).resolve()
    try:
        path.relative_to(_PROJECT_ROOT)
    except ValueError:
        return artifact_record(path, output, role, root_label="output")
    return artifact_record(path, _PROJECT_ROOT, role, root_label="project")


def _hydration_pending(
    *, run_status: str, certificates_pass: bool, allow_draft: bool, unresolved: list[str]
) -> list[str]:
    """Return the gates that remain after writing renderer input.

    Hydration never completes a release: even a passing scientific bundle must
    be rendered and then checked by ``finalize_release``.
    """
    pending: list[str] = ["publication_render", "release_finalization"]
    if run_status != "passed":
        pending.insert(0, "run_log")
    if not certificates_pass:
        pending.insert(0, "certificates")
    if unresolved:
        pending.insert(0, "resolved_manuscript")
    if allow_draft:
        pending.insert(0, "draft_hydration")
    return pending


def _write_local_resolved_manuscript_tree(
    project_root: Path, output: Path, variables: dict[str, str]
) -> None:
    """Hydrate an isolated output root without mutating the template checkout."""
    project_root = Path(project_root).resolve()
    output = require_safe_output_root(output, project_root=project_root)
    token_re = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
    source = project_root / "manuscript"
    destination = output / "manuscript"
    destination.mkdir(parents=True, exist_ok=True)
    for stale in list(destination.glob("*.md")) + list(destination.glob("*.bib")):
        stale.unlink()
    for stale_name in ("config.yaml", "preamble.md"):
        stale = destination / stale_name
        if stale.exists():
            stale.unlink()
    for path in sorted(source.glob("*.md")):
        if path.name in {"AGENTS.md", "README.md", "SYNTAX.md"}:
            continue
        content = path.read_text(encoding="utf-8")
        resolved = token_re.sub(
            lambda match: variables.get(match.group(1), match.group(0)), content
        )
        atomic_write_text(destination / path.name, resolved)
    for auxiliary in ("config.yaml", "preamble.md"):
        path = source / auxiliary
        if path.is_file():
            shutil.copy2(path, destination / auxiliary)
    for path in sorted(source.glob("*.bib")):
        shutil.copy2(path, destination / path.name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate manuscript variables for active_inference_power"
    )
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Allow N/A/PENDING fallbacks when generated evidence is missing (draft mode).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory containing and receiving generated artifacts; relative paths use the project root.",
    )
    args = parser.parse_args()
    output = _resolve_output_root(args.output_root)

    if not args.allow_draft:
        quality_path = output / "quality.json"
        if not quality_path.exists():
            raise FileNotFoundError(
                f"{quality_path} not found; run scripts/generate_quality_report.py "
                "before strict manuscript hydration."
            )
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        validate_quality_report(quality)
        validate_quality_input_provenance(quality, project_root=_PROJECT_ROOT)
        if quality.get("quality_pass") is not True:
            raise RuntimeError("strict manuscript hydration requires a passing quality report")

    claim_ledger_path = _PROJECT_ROOT / "data" / "claim_ledger.yaml"
    results_path = output / "results.json"
    results: dict[str, Any] | None = None
    if results_path.is_file():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        validate_results(
            results,
            require_suite=True,
            require_current_contract=not args.allow_draft,
        )
    elif not args.allow_draft:
        raise FileNotFoundError(results_path)
    run_log_path = output / "run_log.json"
    run_status = "missing"
    run_log: dict[str, Any] | None = None
    if run_log_path.is_file():
        run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
        validate_run_log(run_log)
        run_status = str(run_log["status"])
    elif not args.allow_draft:
        raise FileNotFoundError(run_log_path)
    # Campaigns are complete computational receipts, never manuscript or
    # publication inputs.  Reject them before a permissive draft invocation
    # can revoke a finalized campaign manifest or write renderer artifacts.
    result_profile = (
        results.get("meta", {}).get("profile")
        if isinstance(results, dict) and isinstance(results.get("meta"), dict)
        else None
    )
    run_profile = run_log.get("profile") if isinstance(run_log, dict) else None
    if result_profile == "campaign" or run_profile == "campaign":
        raise ValueError(
            "campaign output roots cannot be hydrated; they are non-release computational evidence"
        )
    manifest_path = output / "manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        if not args.allow_draft:
            if results is None or run_log is None:
                raise ValueError("strict manuscript hydration requires results and a run log")
            validate_source_witnesses(
                manifest,
                results,
                run_log,
                project_root=_PROJECT_ROOT,
                output_root=output,
                config_path=_PROJECT_ROOT / "manuscript" / "config.yaml",
                claim_ledger_path=_PROJECT_ROOT / "data" / "claim_ledger.yaml",
            )
            validate_execution_provenance(
                manifest, results, run_log, project_root=_PROJECT_ROOT
            )
    elif not args.allow_draft:
        raise FileNotFoundError(manifest_path)
    if manifest is not None:
        # Persist the loss of completion authority before touching variables or
        # renderer input. A crash can then leave only a clearly incomplete
        # bundle, never a complete manifest over partially refreshed files.
        invalidate_renderer_artifact_records(manifest)
        manifest["complete"] = False
        manifest["validation_pending"] = [
            "manuscript_hydration",
            "publication_render",
            "release_finalization",
        ]
        validate_manifest(manifest)
        atomic_write_json(manifest_path, manifest)
    benchmark_manifest = _PROJECT_ROOT / "data" / "benchmarks" / "manifest.yaml"
    if benchmark_manifest.is_file():
        validate_benchmark_manifest(
            yaml.safe_load(benchmark_manifest.read_text(encoding="utf-8"))
        )
    if claim_ledger_path.is_file():
        validate_claim_ledger(
            yaml.safe_load(claim_ledger_path.read_text(encoding="utf-8")),
            project_root=_PROJECT_ROOT,
            # The default output tree can validate every active ledger target.
            # An isolated review root may intentionally contain only the
            # current run; its manifest is the authoritative artifact set.
            require_artifacts=not args.allow_draft and output == (_PROJECT_ROOT / "output").resolve(),
            artifact_root=output,
        )

    writer: Any = None
    if output == (_PROJECT_ROOT / "output").resolve():
        try:
            from infrastructure.rendering.manuscript_injection import (
                write_resolved_manuscript_tree as template_writer,
            )
            writer = template_writer
        except ImportError:
            writer = None

    from active_inference_power.manuscript_variables import generate_variables, save_variables

    variables = generate_variables(
        _PROJECT_ROOT,
        require_analysis_outputs=not args.allow_draft,
        output_root=output,
    )
    out_path = output / "data" / "manuscript_variables.json"

    def write_renderer_inputs() -> None:
        save_variables(variables, out_path)
        if writer is not None:
            writer(_PROJECT_ROOT, variables)
        else:
            _write_local_resolved_manuscript_tree(_PROJECT_ROOT, output, variables)

    # Render provenance is self-normalizing: write an initial placeholder,
    # hash all manuscript/variable/registry/figure inputs with that field
    # normalized to its source token, then write the stable digest and verify
    # that a second calculation is identical. The manifest was already made
    # incomplete above, so either write failure remains fail-closed.
    write_renderer_inputs()
    if not args.allow_draft:
        digest = renderer_input_hash(project_root=_PROJECT_ROOT, output_root=output)
        variables["RESULT_RENDER_INPUT_HASH"] = digest
        write_renderer_inputs()
        if renderer_input_hash(project_root=_PROJECT_ROOT, output_root=output) != digest:
            raise ValueError("renderer input digest is not self-consistent after hydration")
    unresolved = [
        str(path)
        for path in sorted((output / "manuscript").glob("*.md"))
        if re.search(r"\{\{[A-Z][A-Z0-9_]*\}\}", path.read_text(encoding="utf-8"))
    ]
    if unresolved and not args.allow_draft:
        raise ValueError(f"unresolved manuscript tokens remain: {unresolved}")
    if manifest is not None:
        refresh_renderer_artifact_records(manifest, output, _PROJECT_ROOT)
        artifacts = manifest.setdefault("artifacts", {})
        quality_path = output / "quality.json"
        if quality_path.is_file():
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            validate_quality_report(quality)
            validate_quality_input_provenance(quality, project_root=_PROJECT_ROOT)
            if not args.allow_draft and quality.get("quality_pass") is not True:
                raise ValueError("quality.json does not pass the declared release gate")
            artifacts["quality"] = _artifact_for_output(quality_path, output, "quality_report")
        artifacts["manuscript_variables"] = _artifact_for_output(
            out_path, output, "manuscript_variables"
        )
        formalism_path = output / "formalism_registry.json"
        if formalism_path.is_file():
            artifacts["formalism_registry"] = _artifact_for_output(
                formalism_path, output, "formalism_registry"
            )
        for resolved in sorted((output / "manuscript").iterdir()):
            if not resolved.is_file() or resolved.suffix not in {".md", ".bib", ".yaml"}:
                continue
            artifacts[f"resolved_manuscript:{resolved.name}"] = _artifact_for_output(
                resolved, output, "resolved_manuscript"
            )
        for source in manuscript_source_paths(_PROJECT_ROOT):
            relative = source.relative_to(_PROJECT_ROOT / "manuscript")
            artifacts[f"source_manuscript:{relative.as_posix()}"] = artifact_record(
                source, _PROJECT_ROOT, "manuscript_source"
            )
        if benchmark_manifest.is_file():
            artifacts["benchmark_manifest"] = artifact_record(
                benchmark_manifest, _PROJECT_ROOT, "benchmark_provenance"
            )
        benchmark_fixture = _PROJECT_ROOT / "data" / "benchmarks" / "public_agent_trace_adapter_stage_2.csv"
        if benchmark_fixture.is_file():
            artifacts["benchmark:public_agent_trace_adapter_stage_2"] = artifact_record(
                benchmark_fixture, _PROJECT_ROOT, "benchmark_fixture"
            )
        golub_fixture = _PROJECT_ROOT / "data" / "benchmarks" / "golub_normalized_fixture.csv"
        if golub_fixture.is_file():
            artifacts["benchmark:golub_normalized_offline_fixture"] = artifact_record(
                golub_fixture, _PROJECT_ROOT, "benchmark_fixture"
            )
        for checkpoint in sorted((output / "checkpoints").glob("*.json")):
            validate_checkpoint(json.loads(checkpoint.read_text(encoding="utf-8")))
            artifacts[f"checkpoint:{checkpoint.stem}"] = _artifact_for_output(
                checkpoint, output, "stage_checkpoint"
            )
        # Hydration creates renderer input, but it never has authority to
        # promote an evidence bundle.  A renderer may now consume stale input
        # and must be followed by the terminal finalizer, which rechecks every
        # source witness, rendered artifact, and digest.
        manifest["complete"] = False
        manifest["validation_pending"] = _hydration_pending(
            run_status=run_status,
            certificates_pass=bool(manifest.get("certificates_pass", False)),
            allow_draft=args.allow_draft,
            unresolved=unresolved,
        )
        validate_manifest(manifest)
        for record in manifest["artifacts"].values():
            artifact_path = _manifest_path(output, _PROJECT_ROOT, record)
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            if sha256_file(artifact_path) != record["sha256"] or artifact_path.stat().st_size != record["bytes"]:
                raise ValueError(f"manifest hash or size mismatch for {artifact_path}")
        atomic_write_json(manifest_path, manifest)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
