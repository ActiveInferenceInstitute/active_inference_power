#!/usr/bin/env python3
"""Validate and atomically finalize an expanded conditional-power campaign.

Campaigns intentionally share the v2 artifact contracts with publication
releases, but they do not require manuscript hydration or a rendered paper.
This gate verifies the completed scientific bundle, checkpoint lineage, figure
registry, dashboard links, and every recorded digest before setting
``manifest.complete``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    validate_results,
    validate_run_log,
)
from scripts.artifacts import (  # noqa: E402
    atomic_write_json,
    require_safe_output_root,
    sha256_file,
)
from scripts.audit_figure_accessibility import audit_figure_accessibility  # noqa: E402
from scripts.release_artifact_contracts import (  # noqa: E402
    validate_result_derivative_artifacts,
)
from scripts.release_provenance import (  # noqa: E402
    checkpoint_names_from_run_log,
    manifest_path as _manifest_path,
    require_file as _require_file,
    resolve_config_path,
    required_campaign_artifacts,
    validate_artifact_inventory,
    validate_certificate_gate_witnesses,
    validate_checkpoint_provenance,
    validate_execution_provenance,
    validate_source_witnesses,
)


def _validate_campaign_terminal_state(
    manifest: Mapping[str, Any],
    results: Mapping[str, Any],
    run_log: Mapping[str, Any],
) -> None:
    """Require one passed campaign profile and one passed certificate gate.

    A schema-valid campaign directory can legitimately preserve a failed run
    for diagnosis.  It must never become complete merely because its files
    still hash correctly, so all three terminal witnesses are checked before
    source provenance or artifact inventory can grant completion authority.
    """
    metadata = results.get("meta")
    if not isinstance(metadata, Mapping) or metadata.get("profile") != "campaign":
        raise ValueError("campaign results must have meta.profile='campaign'")
    if run_log.get("profile") != "campaign":
        raise ValueError("campaign run log must have profile='campaign'")
    if run_log.get("status") != "passed":
        raise ValueError(f"campaign run log is not passed: {run_log.get('status')!r}")

    certificates = results.get("certificates")
    if not isinstance(certificates, Mapping) or not certificates:
        raise ValueError("campaign results have no certificate records")
    metadata = results.get("meta")
    serialized_gate = (
        metadata.get("certificate_gate") if isinstance(metadata, Mapping) else None
    )
    try:
        gate = effective_certificate_gate_result(certificates, serialized_gate)
    except ValueError as exc:
        raise ValueError(f"campaign certificate gate is invalid: {exc}") from exc
    if not gate["terminal_certificates_pass"]:
        raise ValueError(
            "one or more terminal campaign certificates failed: "
            f"{gate['terminal_failures']}"
        )
    validate_certificate_gate_witnesses(results, manifest, run_log)
    if manifest.get("certificates_pass") is not True:
        raise ValueError(
            "campaign manifest certificates_pass must agree with terminal certificates"
        )
    if run_log.get("certificates_pass") is not True:
        raise ValueError(
            "campaign run log certificates_pass must agree with terminal certificates"
        )


def _validate_campaign_certificate_artifact(
    output: Path, results: Mapping[str, Any]
) -> None:
    """Bind the standalone certificate artifact to the result certificate map."""
    path = _require_file(output / "certificates.json", "campaign certificates")
    certificates = json.loads(path.read_text(encoding="utf-8"))
    if certificates != results.get("certificates"):
        raise ValueError(
            "campaign certificates artifact does not match results certificates"
        )


def _validate_campaign_figures(output: Path, results: dict[str, Any]) -> None:
    registry_path = _require_file(output / "figures" / "figure_registry.json", "campaign figure registry")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_figure_registry(registry, expected_filenames=set(FIGURE_NAMES))
    expected = {Path(name).stem for name in FIGURE_NAMES}
    records = registry["figures"]
    if {Path(str(item["filename"])).stem for item in records} != expected:
        raise ValueError("campaign figure registry does not enumerate the configured figures")
    if len(records) != len(expected):
        raise ValueError("campaign figure registry contains duplicate or missing figure records")
    expected_metadata = figure_metadata(results)
    for item in records:
        for key in ("caption", "alt"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"campaign figure {item.get('filename', '<unknown>')} lacks {key}")
        _require_file(output / "figures" / item["filename"], f"campaign figure {item['filename']}")
        name = Path(str(item["filename"])).stem
        if (
            item.get("metadata", {}).get("result_lineage_hash")
            != expected_metadata[name]["result_lineage_hash"]
        ):
            raise ValueError(
                f"campaign figure registry lineage does not match results for {item['filename']}"
            )
    dashboard = _require_file(output / "review" / "dashboard.html", "campaign dashboard")
    dashboard_text = dashboard.read_text(encoding="utf-8")
    for filename in expected:
        if f"{filename}.png" not in dashboard_text:
            raise ValueError(f"campaign dashboard does not link figure {filename}.png")
    if "validity boundary" not in dashboard_text.lower():
        raise ValueError("campaign dashboard is missing validity-boundary metadata")
    lineage = dashboard_result_lineage_hash(results)
    marker = f'<meta name="result-lineage-sha256" content="{lineage}">'
    if marker not in dashboard_text:
        raise ValueError("campaign dashboard lineage does not match results")


def _require_passing_figure_audit(output: Path) -> dict[str, Any]:
    """Run the read-only M5 PNG/reader-metadata audit as a campaign gate."""
    report = audit_figure_accessibility(output)
    if not report["passed"]:
        raise ValueError(f"campaign figure accessibility audit failed: {report['findings']}")
    return {"figure_count": int(report["figure_count"])}


def finalize_campaign(
    output_root: Path, *, config_path: Path | None = None
) -> dict[str, Any]:
    """Validate one campaign output namespace and mark its manifest complete."""
    source_config = resolve_config_path(config_path, project_root=ROOT)
    output = Path(output_root)
    if not output.is_absolute():
        output = ROOT / output
    output = require_safe_output_root(output, project_root=ROOT)
    manifest_path = _require_file(output / "manifest.json", "campaign manifest")
    results_path = _require_file(output / "results.json", "campaign results")
    run_log_path = _require_file(output / "run_log.json", "campaign run log")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    validate_run_log(run_log)
    validate_manifest(manifest)
    _validate_campaign_terminal_state(manifest, results, run_log)
    validate_results(results, require_suite=True, require_current_contract=True)
    _validate_campaign_certificate_artifact(output, results)
    derived_artifacts = validate_result_derivative_artifacts(output, results)
    design = results.get("experiment", {}).get("profile_design", {})
    if design.get("source") != "experiment.campaign" or not design.get("expanded"):
        raise ValueError("campaign results lack the resolved expanded profile design")
    if manifest.get("run_id") != results.get("run_id") or run_log.get("run_id") != results.get("run_id"):
        raise ValueError("campaign manifest, results, and run log run identifiers differ")
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
    checkpoint_names = checkpoint_names_from_run_log(run_log)
    checkpoints = sorted((output / "checkpoints").glob("*.json"))
    if not checkpoints:
        raise ValueError("campaign bundle has no checkpoint files")
    checkpoint_stems = tuple(path.stem for path in checkpoints)
    if checkpoint_stems != checkpoint_names:
        raise ValueError(
            "campaign checkpoint files do not match the terminal stages declared by the run log"
        )
    for checkpoint in checkpoints:
        checkpoint_value = json.loads(checkpoint.read_text(encoding="utf-8"))
        validate_checkpoint(checkpoint_value)
        validate_checkpoint_provenance(
            checkpoint_value,
            run_id=str(results["run_id"]),
            config_sha256=provenance["config_sha256"],
            claim_ledger_sha256=provenance["claim_ledger_sha256"],
            profile="campaign",
            stage=checkpoint.stem,
            execution_source_sha256=execution_digest,
        )
    expected_inventory = required_campaign_artifacts(
        project_root=ROOT,
        output_root=output,
        figure_filenames=FIGURE_NAMES,
        checkpoint_names=checkpoint_names,
        config_path=source_config,
    )
    inventory = validate_artifact_inventory(
        manifest,
        expected_inventory,
        project_root=ROOT,
        output_root=output,
    )
    _validate_campaign_figures(output, results)
    figure_audit = _require_passing_figure_audit(output)
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
        "run_id": str(manifest["run_id"]),
        "artifact_count": len(manifest["artifacts"]),
        "profile": "campaign",
        "checkpoint_count": len(checkpoints),
        **inventory,
        **figure_audit,
        **derived_artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Exact preregistered configuration source; defaults to manuscript/config.yaml.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            finalize_campaign(args.output_root, config_path=args.config), sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
