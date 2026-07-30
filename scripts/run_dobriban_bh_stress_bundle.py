#!/usr/bin/env python3
"""Write an isolated all-chunk receipt for the imported Dobriban/BH protocol.

The public command has one deliberately narrow input: a new ``--output-root``
outside this repository's canonical ``output/`` tree.  It executes all seven
checksum-pinned public-plan chunks with their declared model, seeds, strata,
batch target, and macro-replication counts.  It does not download anything,
run the external outward-rounded Arb certificate, alter a release/campaign/
manuscript artifact, or offer tuning flags.

The resulting local finite stress receipt is explicitly non-promotional.  Its
small figure and static dashboard make the seven finite estimates auditable;
they do not turn the receipt into a theorem, arbitrary-dependence result, or
Active Inference claim.
"""

from __future__ import annotations

# Imports follow the direct-script bootstrap so invocation is independent of
# the caller's current directory.  Ruff cannot infer that ordering.
# ruff: noqa: E402

import argparse
from collections.abc import Mapping, Sequence
import html
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from active_inference_power.protocols import (  # noqa: E402
    DOBRIBAN_BH_BUNDLE_FILENAME,
    DOBRIBAN_BH_FULL_EXECUTION_MODE,
    DobribanBHProtocol,
    build_dobriban_bh_stress_bundle,
    frozen_dobriban_bh_execution_profile,
    load_dobriban_bh_protocol,
    run_dobriban_bh_chunk,
    validate_dobriban_bh_execution_profile,
    validate_dobriban_bh_stress_bundle,
)
from active_inference_power.schema import validate_figure_registry  # noqa: E402
from scripts.artifacts import (  # noqa: E402
    artifact_record,
    atomic_write_json,
    atomic_write_text,
    require_safe_output_root,
    sha256_file,
)
from scripts.audit_figure_accessibility import (  # noqa: E402
    audit_figure_accessibility,
    audit_rendered_figure_alts,
)


BUNDLE_FIGURE_FILENAME = "dobriban_bh_fdr_stress.png"
FIGURE_REGISTRY_FILENAME = "figure_registry.json"
DASHBOARD_FILENAME = "dashboard.json"
INDEX_FILENAME = "index.html"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_KIND = "checksum_pinned_all_chunk_dobriban_bh_stress_manifest"

_SOURCE_PATHS = {
    "orchestrator": Path(__file__),
    "protocol_adapter": ROOT / "src" / "active_inference_power" / "protocols" / "dobriban_bh.py",
    "bundle_contract": ROOT / "src" / "active_inference_power" / "protocols" / "dobriban_bh_bundle.py",
    "protocol_plan": ROOT / "src" / "active_inference_power" / "protocols" / "dobriban_bh_plan.json",
    "artifact_io": ROOT / "scripts" / "artifacts.py",
    "figure_accessibility_audit": ROOT / "scripts" / "audit_figure_accessibility.py",
    "figure_registry_schema": ROOT / "src" / "active_inference_power" / "schema.py",
    "lockfile": ROOT / "uv.lock",
}


def _validate_output_root(output_root: Path) -> Path:
    """Require a new concrete directory outside every canonical output path."""
    target = Path(output_root).expanduser()
    if not target.is_absolute():
        target = ROOT / target
    target = require_safe_output_root(target, project_root=ROOT)
    canonical_output = (ROOT / "output").resolve()
    if target == ROOT.resolve():
        raise ValueError("Dobriban/BH stress bundle requires a new isolated --output-root, not the project root")
    try:
        target.relative_to(canonical_output)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Dobriban/BH stress bundle requires a new isolated --output-root outside canonical output/"
        )
    if target == target.parent:
        raise ValueError("Dobriban/BH stress bundle output root must be a concrete directory")
    if target.exists():
        raise FileExistsError(
            f"Dobriban/BH stress bundle output root already exists: {target}; choose a new isolated directory"
        )
    return target


def _source_hashes() -> dict[str, str]:
    """Hash every local source and frozen input used by the receipt."""
    missing = [name for name, path in _SOURCE_PATHS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Dobriban/BH bundle sources are missing: {missing}")
    return {name: sha256_file(path) for name, path in _SOURCE_PATHS.items()}


def _assert_stable_sources(
    initial_sources: Mapping[str, str],
    final_sources: Mapping[str, str],
    *,
    initial_protocol: DobribanBHProtocol,
    final_protocol: DobribanBHProtocol,
) -> None:
    """Fail closed instead of publishing a mixed-source finite receipt."""
    if dict(initial_sources) != dict(final_sources):
        raise RuntimeError("Dobriban/BH source witness changed during execution; no bundle was written")
    if initial_protocol.to_mapping() != final_protocol.to_mapping():
        raise RuntimeError("Dobriban/BH frozen protocol changed during execution; no bundle was written")


def _bundle_rows(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = bundle["chunks"]
    if not isinstance(rows, list):  # The source contract validates this before callers reach here.
        raise ValueError("Dobriban/BH bundle chunks must be a list")
    return [
        _require_mapping(row, "Dobriban/BH bundle row")
        for row in rows
    ]


def _figure_text(bundle: Mapping[str, Any]) -> dict[str, str]:
    """Keep visual question, caption, alternative text, and boundary distinct."""
    profile = bundle["execution_profile"]
    if not isinstance(profile, Mapping):
        raise ValueError("Dobriban/BH execution profile must be a mapping")
    mode = str(profile["mode"])
    profile_label = (
        "the exact frozen public-plan execution"
        if mode == DOBRIBAN_BH_FULL_EXECUTION_MODE
        else "a compact real smoke execution that retains every declared chunk"
    )
    boundary = str(bundle["claim_boundary"])
    return {
        "title": "Dobriban/BH protocol stress — finite FDR summaries by declared chunk",
        "reader_question": (
            "Across the seven declared chunks, what finite macro-level BH FDR estimates and "
            "uncertainty intervals are produced by the locally executed checksum-pinned protocol?"
        ),
        "visible_comparison": (
            "Seven labelled circular markers show one finite macro-level FDR estimate per declared chunk; "
            "vertical whiskers show its 95% Student-t interval and the horizontal dashed line marks alpha = 0.01."
        ),
        "takeaway": (
            "The panel preserves chunk-level, seed-specific finite stress summaries for audit rather than "
            "collapsing them into a theorem or a universal rate."
        ),
        "caption": (
            "Reader question: Across the seven declared chunks, what finite macro-level BH FDR summaries are "
            "observed? Visible comparison: circular markers show one estimate per labelled chunk, vertical "
            "whiskers show 95% Student-t intervals across macro replications, and the dashed line marks the "
            "nominal BH target alpha = 0.01. Takeaway: the panel retains auditable chunk-level finite stress "
            f"summaries under {profile_label}. Boundary: {boundary}"
        ),
        "alt": (
            "Single-panel categorical error-bar plot with seven declared Dobriban/BH chunks ordered from N=50 "
            "to N=200. Blue circular points show finite macro-level BH FDR estimates, vertical blue whiskers "
            "show 95 percent Student-t intervals, and a black dashed horizontal line marks alpha equals 0.01. "
            "Each x-axis label gives the chunk prefix, N, and declared seed; a legend explains the point, "
            "interval, and target line."
        ),
        "visual_encoding": (
            "Blue circular markers encode finite macro-level FDR estimates; vertical whiskers encode 95% "
            "Student-t intervals; chunk prefix/N/seed labels identify categories; a black dashed labelled line "
            "encodes the nominal alpha target."
        ),
        "boundary": boundary,
    }


def _write_figure(bundle: Mapping[str, Any], path: Path) -> dict[str, str]:
    """Render one high-resolution accessible stress figure inside staging only."""
    rows = _bundle_rows(bundle)
    if len(rows) != 7:
        raise ValueError("Dobriban/BH figure requires all seven declared chunks")
    text = _figure_text(bundle)
    estimates: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    labels: list[str] = []
    for row in rows:
        declared = row["declared"]
        result = row["result"]
        if not isinstance(declared, Mapping) or not isinstance(result, Mapping):
            raise ValueError("Dobriban/BH figure rows must contain declared/result mappings")
        estimate = float(result["fdr_hat"])
        lower = float(result["ci95_low"])
        upper = float(result["ci95_high"])
        if not lower <= estimate <= upper:
            raise ValueError("Dobriban/BH figure interval must contain its estimate")
        estimates.append(estimate)
        lowers.append(lower)
        uppers.append(upper)
        labels.append(
            f"{declared['prefix']}\nN={declared['N']}; seed={declared['master_seed']}"
        )
    alpha = float(
        _require_mapping(bundle["protocol"], "bundle protocol")["protocol"]["model"]["alpha"]
    )
    positions = list(range(len(rows)))
    lower_errors = [estimate - lower for estimate, lower in zip(estimates, lowers, strict=True)]
    upper_errors = [upper - estimate for estimate, upper in zip(estimates, uppers, strict=True)]
    figure, axes = plt.subplots(figsize=(14.4, 8.0))
    axes.errorbar(
        positions,
        estimates,
        yerr=[lower_errors, upper_errors],
        fmt="o",
        markersize=7.2,
        capsize=4.2,
        linewidth=1.8,
        color="#1769aa",
        ecolor="#1769aa",
        label="Finite macro-level FDR estimate ± 95% Student-t interval",
        zorder=3,
    )
    axes.axhline(
        alpha,
        color="#111827",
        linewidth=1.4,
        linestyle="--",
        label="Nominal BH target alpha = 0.01",
        zorder=2,
    )
    axes.set_xticks(positions, labels, fontsize=8.2)
    axes.set_xlabel(
        "Declared frozen-plan chunk (prefix; N count; independently declared seed)",
        fontsize=10.3,
    )
    axes.set_ylabel(
        "Finite macro-level BH false discovery rate estimate (dimensionless; 95% t interval)",
        fontsize=10.3,
    )
    axes.set_title(text["title"], fontsize=14, fontweight="bold", pad=14)
    lower_bound = min(0.0, alpha, *lowers)
    upper_bound = max(alpha, *uppers)
    padding = max(0.001, (upper_bound - lower_bound) * 0.12)
    axes.set_ylim(lower_bound - padding, upper_bound + padding)
    axes.grid(axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.85, zorder=0)
    axes.legend(loc="upper left", fontsize=9, title="Marks, interval, and threshold", title_fontsize=9)
    figure.text(
        0.5,
        0.025,
        "Finite offline stress evidence only: the external outward-rounded Arb certificate is not executed or reproduced.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#334155",
    )
    figure.tight_layout(rect=(0.0, 0.065, 1.0, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=180,
            bbox_inches="tight",
            metadata={"Title": text["title"], "Description": text["alt"]},
        )
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()
    return text


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _figure_registry(
    bundle: Mapping[str, Any],
    *,
    bundle_file_sha256: str,
    figures_dir: Path,
    text: Mapping[str, str],
) -> dict[str, Any]:
    """Create a standard, audit-ready figure registry for the one PNG."""
    figure_path = figures_dir / BUNDLE_FIGURE_FILENAME
    if not figure_path.is_file():
        raise FileNotFoundError(f"Dobriban/BH figure is missing: {figure_path}")
    profile = _require_mapping(bundle["execution_profile"], "bundle execution profile")
    macro = _require_mapping(profile["macro_replications"], "bundle macro replications")
    sample_size = (
        f"mode={profile['mode']}; seven declared chunks; strata={profile['strata']} per macro replication; "
        "macro replications="
        + ", ".join(f"{prefix}={macro[prefix]}" for prefix in profile["declared_chunk_prefixes"])
    )
    registry = {
        "kind": "dobriban_bh_stress_figure_registry",
        "schema_version": "2.3",
        "status": bundle["status"],
        "bundle_result_sha256": bundle["result_sha256"],
        "bundle_file_sha256": bundle_file_sha256,
        "protocol_hash": _require_mapping(bundle["protocol"], "bundle protocol")["protocol_hash"],
        "figures": [
            {
                "name": "dobriban_bh_fdr_stress",
                "path": f"figures/{BUNDLE_FIGURE_FILENAME}",
                "filename": BUNDLE_FIGURE_FILENAME,
                "sha256": sha256_file(figure_path),
                "bytes": figure_path.stat().st_size,
                "label": "fig:dobriban-bh-fdr-stress",
                "generated_by": "scripts.run_dobriban_bh_stress_bundle",
                "title": text["title"],
                "caption": text["caption"],
                "alt": text["alt"],
                "reader_question": text["reader_question"],
                "visible_comparison": text["visible_comparison"],
                "takeaway": text["takeaway"],
                "visual_encoding": text["visual_encoding"],
                "evaluation_perspective": (
                    "Investigator-facing offline stress protocol; it does not "
                    "estimate an agent's intrinsic power or a real-world environment distribution."
                ),
                "metadata": {
                    "estimand": (
                        "Per-declared-chunk finite macro-level estimate of the ordinary two-sided BH false "
                        "discovery rate (dimensionless)."
                    ),
                    "uncertainty": "Vertical whiskers are two-sided 95% Student-t intervals across macro replications.",
                    "sample_size": sample_size,
                    "evidence_class": "stress_test",
                    "validity_boundary": text["boundary"],
                    "redundant_encoding": (
                        "Blue circle markers and direct x-axis labels identify chunks; vertical whiskers show "
                        "intervals; a labelled dashed threshold line marks alpha."
                    ),
                    "scenario_hashes": (
                        "protocol_sha256="
                        + str(_require_mapping(bundle["protocol"], "bundle protocol")["protocol_hash"])
                    ),
                    "result_lineage_hash": bundle["result_sha256"],
                },
            }
        ],
    }
    validate_figure_registry(registry, expected_filenames={BUNDLE_FIGURE_FILENAME})
    return registry


def _dashboard(bundle: Mapping[str, Any], *, bundle_file_sha256: str, figure_registry_sha256: str) -> dict[str, Any]:
    """Expose the complete row set in a small static, non-promotional payload."""
    rows: list[dict[str, Any]] = []
    for row in _bundle_rows(bundle):
        declared = _require_mapping(row["declared"], "dashboard declared chunk")
        result = _require_mapping(row["result"], "dashboard result")
        rows.append(
            {
                "prefix": row["prefix"],
                "N": declared["N"],
                "block": declared["block"],
                "master_seed": declared["master_seed"],
                "macro_replications": result["macro_replications"],
                "datasets": result["datasets"],
                "fdr_hat": result["fdr_hat"],
                "ci95": [result["ci95_low"], result["ci95_high"]],
                "mcse": result["mcse"],
                "mean_rejections": result["mean_rejections"],
                "rejection_probability": result["rejection_probability"],
            }
        )
    return {
        "schema_version": "1.0",
        "kind": "dobriban_bh_stress_dashboard",
        "status": bundle["status"],
        "evidence_class": bundle["evidence_class"],
        "bundle_result_sha256": bundle["result_sha256"],
        "bundle_file_sha256": bundle_file_sha256,
        "figure_registry_sha256": figure_registry_sha256,
        "protocol_hash": _require_mapping(bundle["protocol"], "bundle protocol")["protocol_hash"],
        "claim_boundary": bundle["claim_boundary"],
        "chunks": rows,
    }


def _format_number(value: Any) -> str:
    return f"{float(value):.6g}"


def _index_html(bundle: Mapping[str, Any], text: Mapping[str, str]) -> str:
    """Render a no-script HTML index with an explicit non-claim and data table."""
    rows: list[str] = []
    for row in _bundle_rows(bundle):
        declared = _require_mapping(row["declared"], "index declared chunk")
        result = _require_mapping(row["result"], "index result")
        rows.append(
            "<tr>"
            f"<th scope=\"row\">{html.escape(str(row['prefix']))}</th>"
            f"<td>{html.escape(str(declared['N']))}</td>"
            f"<td>{html.escape(str(declared['master_seed']))}</td>"
            f"<td>{html.escape(str(result['macro_replications']))}</td>"
            f"<td>{_format_number(result['fdr_hat'])}</td>"
            f"<td>[{_format_number(result['ci95_low'])}, {_format_number(result['ci95_high'])}]</td>"
            f"<td>{_format_number(result['mcse'])}</td>"
            "</tr>"
        )
    boundary = html.escape(str(bundle["claim_boundary"]))
    result_hash = html.escape(str(bundle["result_sha256"]))
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="en">',
            "<head><meta charset=\"utf-8\"><title>Dobriban/BH all-chunk finite stress receipt</title></head>",
            "<body>",
            "<main>",
            "<h1>Dobriban/BH all-chunk finite stress receipt</h1>",
            f"<p><strong>Status:</strong> {html.escape(str(bundle['status']))}</p>",
            f"<p>{boundary}</p>",
            "<p>Machine-readable artifacts: ",
            f'<a href="{DOBRIBAN_BH_BUNDLE_FILENAME}">{DOBRIBAN_BH_BUNDLE_FILENAME}</a>; ',
            f'<a href="{DASHBOARD_FILENAME}">{DASHBOARD_FILENAME}</a>; ',
            f'<a href="figures/{FIGURE_REGISTRY_FILENAME}">figure registry</a>.</p>',
            "<figure>",
            f'<img src="figures/{BUNDLE_FIGURE_FILENAME}" alt="{html.escape(text["alt"], quote=True)}">',
            f"<figcaption>{html.escape(text['caption'])}</figcaption>",
            "</figure>",
            "<h2>Declared chunk summaries</h2>",
            "<table>",
            "<thead><tr><th>Chunk</th><th>N</th><th>Seed</th><th>Macro replications</th>"
            "<th>Finite FDR estimate</th><th>95% t interval</th><th>MCSE</th></tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table>",
            f"<p>Result SHA-256: <code>{result_hash}</code></p>",
            "</main>",
            "</body>",
            "</html>",
            "",
        )
    )


def _manifest(
    staging_root: Path,
    bundle: Mapping[str, Any],
    *,
    bundle_file_sha256: str,
    figure_registry_sha256: str,
) -> dict[str, Any]:
    """Bind every generated direct-root artifact without creating a release manifest."""
    artifact_specs = {
        "stress_bundle": (DOBRIBAN_BH_BUNDLE_FILENAME, "finite_dobriban_bh_stress_bundle"),
        "stress_figure": (f"figures/{BUNDLE_FIGURE_FILENAME}", "accessible_finite_stress_figure"),
        "figure_registry": (f"figures/{FIGURE_REGISTRY_FILENAME}", "finite_stress_figure_registry"),
        "dashboard": (DASHBOARD_FILENAME, "finite_stress_dashboard"),
        "index_html": (INDEX_FILENAME, "finite_stress_html_index"),
    }
    return {
        "schema_version": "1.0",
        "kind": MANIFEST_KIND,
        "status": bundle["status"],
        "evidence_class": bundle["evidence_class"],
        "bundle_result_sha256": bundle["result_sha256"],
        "bundle_file_sha256": bundle_file_sha256,
        "protocol_hash": _require_mapping(bundle["protocol"], "bundle protocol")["protocol_hash"],
        "source_hashes": bundle["source_hashes"],
        "claim_boundary": bundle["claim_boundary"],
        "figure_registry_sha256": figure_registry_sha256,
        "artifacts": {
            name: artifact_record(staging_root / relative, staging_root, role, root_label="output")
            for name, (relative, role) in artifact_specs.items()
        },
    }


def _validate_artifact_record(record: object, *, root: Path, name: str) -> None:
    item = _require_mapping(record, f"manifest artifact {name}")
    if set(item) != {"root", "path", "sha256", "bytes", "role"}:
        raise ValueError(f"manifest artifact {name} has an unexpected contract")
    if item["root"] != "output":
        raise ValueError(f"manifest artifact {name} must use the isolated output namespace")
    relative = item["path"]
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"manifest artifact {name} has an unsafe relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest artifact {name} escapes its isolated root") from exc
    if not path.is_file():
        raise FileNotFoundError(f"manifest artifact {name} is missing: {path}")
    if not isinstance(item["sha256"], str) or item["sha256"] != sha256_file(path):
        raise ValueError(f"manifest artifact {name} SHA-256 does not match")
    if isinstance(item["bytes"], bool) or item["bytes"] != path.stat().st_size:
        raise ValueError(f"manifest artifact {name} byte count does not match")
    if not isinstance(item["role"], str) or not item["role"]:
        raise ValueError(f"manifest artifact {name} role is invalid")


def _validate_dashboard(
    dashboard: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    bundle_file_sha256: str,
    figure_registry_sha256: str,
) -> None:
    expected_keys = {
        "schema_version",
        "kind",
        "status",
        "evidence_class",
        "bundle_result_sha256",
        "bundle_file_sha256",
        "figure_registry_sha256",
        "protocol_hash",
        "claim_boundary",
        "chunks",
    }
    if set(dashboard) != expected_keys:
        raise ValueError("Dobriban/BH dashboard differs from its frozen output contract")
    if dashboard["schema_version"] != "1.0" or dashboard["kind"] != "dobriban_bh_stress_dashboard":
        raise ValueError("Dobriban/BH dashboard schema or kind is invalid")
    expected_bundle_values = {
        "status": bundle["status"],
        "evidence_class": bundle["evidence_class"],
        "bundle_result_sha256": bundle["result_sha256"],
        "claim_boundary": bundle["claim_boundary"],
    }
    for key, expected in expected_bundle_values.items():
        if dashboard[key] != expected:
            raise ValueError(f"Dobriban/BH dashboard {key} does not match the stress bundle")
    if dashboard["bundle_file_sha256"] != bundle_file_sha256:
        raise ValueError("Dobriban/BH dashboard bundle file hash does not match")
    if dashboard["figure_registry_sha256"] != figure_registry_sha256:
        raise ValueError("Dobriban/BH dashboard figure registry hash does not match")
    if dashboard["protocol_hash"] != _require_mapping(bundle["protocol"], "bundle protocol")["protocol_hash"]:
        raise ValueError("Dobriban/BH dashboard protocol hash does not match")
    rows = dashboard["chunks"]
    if not isinstance(rows, list) or len(rows) != 7:
        raise ValueError("Dobriban/BH dashboard must retain seven chunk rows")
    source_rows = _bundle_rows(bundle)
    for source, observed in zip(source_rows, rows, strict=True):
        if not isinstance(observed, Mapping) or observed.get("prefix") != source["prefix"]:
            raise ValueError("Dobriban/BH dashboard chunk order does not match the bundle")


def _validate_staged_output(
    staging_root: Path,
    *,
    protocol: DobribanBHProtocol,
    source_hashes: Mapping[str, str],
) -> None:
    """Read every staged artifact back and reject incomplete or altered output."""
    bundle_path = staging_root / DOBRIBAN_BH_BUNDLE_FILENAME
    registry_path = staging_root / "figures" / FIGURE_REGISTRY_FILENAME
    dashboard_path = staging_root / DASHBOARD_FILENAME
    index_path = staging_root / INDEX_FILENAME
    manifest_path = staging_root / MANIFEST_FILENAME
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    validate_dobriban_bh_stress_bundle(bundle, protocol=protocol)
    if bundle["source_hashes"] != dict(source_hashes):
        raise ValueError("Dobriban/BH stress bundle source witness does not match the executed sources")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_figure_registry(registry, expected_filenames={BUNDLE_FIGURE_FILENAME})
    record = registry["figures"][0]
    if registry.get("status") != bundle["status"] or registry.get("bundle_result_sha256") != bundle["result_sha256"]:
        raise ValueError("Dobriban/BH figure registry does not match the stress bundle")
    if registry.get("bundle_file_sha256") != sha256_file(bundle_path):
        raise ValueError("Dobriban/BH figure registry bundle file hash does not match")
    figure_path = staging_root / "figures" / BUNDLE_FIGURE_FILENAME
    if record.get("sha256") != sha256_file(figure_path) or record.get("bytes") != figure_path.stat().st_size:
        raise ValueError("Dobriban/BH figure registry PNG record does not match")
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    _validate_dashboard(
        dashboard,
        bundle=bundle,
        bundle_file_sha256=sha256_file(bundle_path),
        figure_registry_sha256=sha256_file(registry_path),
    )
    index_html = index_path.read_text(encoding="utf-8")
    if str(record["alt"]) not in index_html or str(bundle["claim_boundary"]) not in index_html:
        raise ValueError("Dobriban/BH HTML index does not retain its authored alternative text and boundary")
    figure_audit = audit_figure_accessibility(staging_root)
    if not figure_audit["passed"]:
        raise ValueError(f"Dobriban/BH figure accessibility audit failed: {figure_audit['findings']}")
    alt_audit = audit_rendered_figure_alts(staging_root, rendered_html=index_path)
    if not alt_audit["passed"]:
        raise ValueError(f"Dobriban/BH HTML alternative-text audit failed: {alt_audit['findings']}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_keys = {
        "schema_version",
        "kind",
        "status",
        "evidence_class",
        "bundle_result_sha256",
        "bundle_file_sha256",
        "protocol_hash",
        "source_hashes",
        "claim_boundary",
        "figure_registry_sha256",
        "artifacts",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("Dobriban/BH manifest differs from its frozen output contract")
    if manifest["schema_version"] != "1.0" or manifest["kind"] != MANIFEST_KIND:
        raise ValueError("Dobriban/BH manifest schema or kind is invalid")
    expected_manifest_values = {
        "status": bundle["status"],
        "evidence_class": bundle["evidence_class"],
        "bundle_result_sha256": bundle["result_sha256"],
        "bundle_file_sha256": sha256_file(bundle_path),
        "protocol_hash": _require_mapping(bundle["protocol"], "bundle protocol")["protocol_hash"],
        "source_hashes": dict(source_hashes),
        "claim_boundary": bundle["claim_boundary"],
        "figure_registry_sha256": sha256_file(registry_path),
    }
    for key, expected in expected_manifest_values.items():
        if manifest[key] != expected:
            raise ValueError(f"Dobriban/BH manifest {key} does not match staged output")
    artifacts = _require_mapping(manifest["artifacts"], "Dobriban/BH manifest artifacts")
    expected_artifacts = {
        "stress_bundle",
        "stress_figure",
        "figure_registry",
        "dashboard",
        "index_html",
    }
    if set(artifacts) != expected_artifacts:
        raise ValueError("Dobriban/BH manifest artifact inventory is incomplete")
    for name, item in artifacts.items():
        _validate_artifact_record(item, root=staging_root, name=str(name))


def run_dobriban_bh_stress_bundle(
    output_root: Path,
    execution_profile: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Atomically run all declared chunks and write one non-promotional receipt.

    With no profile, the command uses exactly the frozen public plan.  Passing
    a compact profile is reserved for real test/smoke execution in Python; no
    corresponding CLI option exists and the resulting artifact is serialized
    with a distinct smoke-only status.
    """
    target = _validate_output_root(Path(output_root))
    protocol = load_dobriban_bh_protocol()
    if execution_profile is None:
        active_profile = frozen_dobriban_bh_execution_profile(protocol)
    elif isinstance(execution_profile, Mapping):
        active_profile = dict(execution_profile)
    else:
        raise ValueError("Dobriban/BH execution profile must be a mapping")
    validate_dobriban_bh_execution_profile(active_profile, protocol=protocol)
    initial_sources = _source_hashes()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.dobriban-bh-stress-", dir=target.parent)
    )
    try:
        macro = _require_mapping(active_profile.get("macro_replications"), "execution profile macro replications")
        results: list[dict[str, Any]] = []
        for declared in protocol.plan["chunks"]:
            prefix = str(declared["prefix"])
            result = run_dobriban_bh_chunk(
                int(declared["N"]),
                strata=int(active_profile["strata"]),
                macro_replications=int(macro[prefix]),
                master_seed=int(declared["master_seed"]),
                target_entries=int(active_profile["target_entries"]),
                protocol=protocol,
            )
            results.append(result.to_mapping())
        final_protocol = load_dobriban_bh_protocol()
        _assert_stable_sources(
            initial_sources,
            _source_hashes(),
            initial_protocol=protocol,
            final_protocol=final_protocol,
        )
        bundle = build_dobriban_bh_stress_bundle(
            protocol,
            execution_profile=active_profile,
            chunk_results=results,
            source_hashes=initial_sources,
        )
        bundle_path = staging_root / DOBRIBAN_BH_BUNDLE_FILENAME
        atomic_write_json(bundle_path, bundle)
        figures_dir = staging_root / "figures"
        text = _write_figure(bundle, figures_dir / BUNDLE_FIGURE_FILENAME)
        registry_path = figures_dir / FIGURE_REGISTRY_FILENAME
        registry = _figure_registry(
            bundle,
            bundle_file_sha256=sha256_file(bundle_path),
            figures_dir=figures_dir,
            text=text,
        )
        atomic_write_json(registry_path, registry)
        dashboard_path = staging_root / DASHBOARD_FILENAME
        atomic_write_json(
            dashboard_path,
            _dashboard(
                bundle,
                bundle_file_sha256=sha256_file(bundle_path),
                figure_registry_sha256=sha256_file(registry_path),
            ),
        )
        index_path = staging_root / INDEX_FILENAME
        atomic_write_text(index_path, _index_html(bundle, text))
        manifest_path = staging_root / MANIFEST_FILENAME
        atomic_write_json(
            manifest_path,
            _manifest(
                staging_root,
                bundle,
                bundle_file_sha256=sha256_file(bundle_path),
                figure_registry_sha256=sha256_file(registry_path),
            ),
        )
        _validate_staged_output(staging_root, protocol=protocol, source_hashes=initial_sources)
        _assert_stable_sources(
            initial_sources,
            _source_hashes(),
            initial_protocol=protocol,
            final_protocol=load_dobriban_bh_protocol(),
        )
        if target.exists():
            raise FileExistsError(
                f"Dobriban/BH stress bundle output root was created concurrently: {target}"
            )
        os.replace(staging_root, target)
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    return {
        "output_root": str(target),
        "bundle_path": str(target / DOBRIBAN_BH_BUNDLE_FILENAME),
        "figure_path": str(target / "figures" / BUNDLE_FIGURE_FILENAME),
        "figure_registry_path": str(target / "figures" / FIGURE_REGISTRY_FILENAME),
        "dashboard_path": str(target / DASHBOARD_FILENAME),
        "index_path": str(target / INDEX_FILENAME),
        "manifest_path": str(target / MANIFEST_FILENAME),
        "bundle_sha256": sha256_file(target / DOBRIBAN_BH_BUNDLE_FILENAME),
        "manifest_sha256": sha256_file(target / MANIFEST_FILENAME),
        "result_sha256": str(bundle["result_sha256"]),
        "status": str(bundle["status"]),
    }


def _parser() -> argparse.ArgumentParser:
    """Expose only the explicit isolated-root boundary to CLI callers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="New isolated root outside output/ for all seven frozen Dobriban/BH chunks.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the frozen all-chunk command and print a compact receipt."""
    args = _parser().parse_args(argv)
    receipt = run_dobriban_bh_stress_bundle(args.output_root)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main.
    raise SystemExit(main())
