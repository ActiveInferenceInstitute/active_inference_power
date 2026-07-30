"""Read-only release-review integration checks against the generated bundle."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
import shutil
from typing import cast

import pytest

from active_inference_power.figures import FIGURE_NAMES
from scripts.artifacts import atomic_write_json, sha256_file
from scripts.finalize_release import finalize_release
from scripts.finalize_release import _pdf_text_contains_witness as finalizer_pdf_witness
from scripts.generate_quality_report import quality_input_sha256
from scripts.release_provenance import required_release_artifacts
from scripts.review_release import (
    _clean_wheel_install,
    _pdf_text_contains_witness as review_pdf_witness,
    _require_passing_rendered_figure_alt_audit,
    _require_passing_documentation_link_audit,
    _require_passing_manuscript_presentation_audit,
    _renderer_manifest_records_required,
    require_full_passing_review,
    review_release,
    validate_review_receipt,
    write_passing_review_receipt,
)


def _fast_bundle_with_failed_quality(
    tmp_path: Path, generated_fast_bundle: Path
) -> Path:
    """Copy real fast artifacts and add an explicit non-terminal quality record.

    The release reviewer must inspect a complete JSON bundle before it can
    explain why that bundle is not promotable.  A fast run has no quality
    report by design, so this helper supplies a schema-valid *failed*
    diagnostic with the live quality-input witness.  It does not attest to a
    passing quality gate or use retained release artifacts.
    """
    output = tmp_path / "review-fast-output"
    shutil.copytree(generated_fast_bundle, output)
    atomic_write_json(
        output / "quality.json",
        {
            "schema_version": "2.0",
            "test_count": 0,
            "line_coverage_pct": 0.0,
            "branch_coverage_pct": 0.0,
            "coverage_gate_pct": 95.0,
            "branch_coverage_gate_pct": 95.0,
            "quality_pass": False,
            "quality_input_sha256": quality_input_sha256(),
            "pytest_exit_code": 1,
            "command": "non-terminal fresh-bundle review diagnostic",
        },
    )
    return output


def test_release_review_is_read_only_against_a_fresh_generated_bundle(
    tmp_path: Path,
    generated_fast_bundle: Path,
) -> None:
    output = _fast_bundle_with_failed_quality(tmp_path, generated_fast_bundle)
    manifest = output / "manifest.json"
    before = manifest.read_bytes()
    report = review_release(output, check_clean_install=False)
    # A fast evidence bundle deliberately lacks quality, hydration, rendering,
    # and finalization. This still exercises the reviewer's read-only contract
    # without asking clean CI to provide ignored publication artifacts.
    assert isinstance(report["passed"], bool)
    assert manifest.read_bytes() == before
    assert {check["name"] for check in report["checks"]} >= {
        "schema_and_quality",
        "source_witnesses",
        "manifest_hashes",
        "checkpoint_lineage",
        "figures_and_dashboard",
        "figure_accessibility",
        "documentation_links",
        "resolved_manuscript",
        "manuscript_presentation",
        "rendered_text",
        "rendered_figure_alt_text",
    }


def test_release_rendered_alt_gate_requires_auxiliary_page_fidelity(
    tmp_path: Path,
    generated_fast_bundle: Path,
) -> None:
    """The terminal review covers retained reader-facing section pages too."""
    output = tmp_path / "review-alt-output"
    shutil.copytree(generated_fast_bundle, output)
    records = json.loads(
        (output / "figures" / "figure_registry.json").read_text(encoding="utf-8")
    )["figures"]
    web = output / "web"
    web.mkdir()
    (web / "index.html").write_text(
        "".join(
            '<img src="../figures/{filename}" alt="{alt}">\n'.format(
                filename=record["filename"],
                alt=escape(str(record["alt"]), quote=True),
            )
            for record in records
        ),
        encoding="utf-8",
    )
    first = records[0]
    section = web / "manuscript__03_results.html"
    section.write_text(
        '<img src="../figures/{filename}" alt="Outdated section-page text.">\n'.format(
            filename=first["filename"]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rendered figure alt-text audit failed"):
        _require_passing_rendered_figure_alt_audit(output)

    section.write_text(
        '<img src="../figures/{filename}" alt="{alt}">\n'.format(
            filename=first["filename"],
            alt=escape(str(first["alt"]), quote=True),
        ),
        encoding="utf-8",
    )
    receipt = _require_passing_rendered_figure_alt_audit(output)

    assert receipt["auxiliary_page_count"] == 1


def test_release_review_rejects_missing_or_mismatched_explicit_config(
    tmp_path: Path,
    generated_fast_bundle: Path,
) -> None:
    root = Path(__file__).resolve().parent.parent
    output = _fast_bundle_with_failed_quality(tmp_path, generated_fast_bundle)
    missing = review_release(
        output,
        check_clean_install=False,
        config_path=root / "missing-preregistered-lane.yaml",
    )
    assert missing["passed"] is False
    assert missing["checks"] == [
        {
            "name": "source_configuration",
            "passed": False,
            "message": f"source configuration is missing or empty: {root / 'missing-preregistered-lane.yaml'}",
        }
    ]

    mismatched = review_release(
        output,
        check_clean_install=False,
        config_path=root / "data" / "claim_ledger.yaml",
    )
    source_check = next(
        check for check in mismatched["checks"] if check["name"] == "source_witnesses"
    )
    assert source_check["passed"] is False
    assert "expected source" in source_check["message"]


def test_release_finalizer_rejects_a_missing_explicit_lane_config(
    tmp_path: Path,
    generated_fast_bundle: Path,
) -> None:
    """The publication finalizer validates config selection before mutation."""
    manifest = generated_fast_bundle / "manifest.json"
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="source configuration is missing or empty"):
        finalize_release(
            generated_fast_bundle,
            config_path=tmp_path / "missing-preregistered-lane.yaml",
        )
    assert manifest.read_bytes() == before


def test_full_review_rejection_never_marks_a_manifest_complete(tmp_path: Path) -> None:
    """A rejected read-only review cannot publish even an on-disk manifest."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "run_id": "unreviewed-run",
                "complete": False,
                "config_sha256": "a" * 64,
                "artifacts": {
                    "placeholder": {
                        "root": "output",
                        "path": "placeholder.json",
                        "sha256": "b" * 64,
                        "bytes": 1,
                        "role": "placeholder",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    before = manifest.read_bytes()

    with pytest.raises(ValueError, match="full read-only release review failed: load_bundle"):
        finalize_release(tmp_path)

    assert manifest.read_bytes() == before
    assert json.loads(manifest.read_text(encoding="utf-8"))["complete"] is False


def test_pdf_witness_checks_accept_only_exact_hash_characters_across_line_wraps() -> None:
    witness = "0123456789abcdef" * 4
    extracted = f"Provenance witness: {witness[:43]}\n{witness[43:]}.\n"
    changed_witness = witness[:-1] + "0"
    changed = f"Provenance witness: {changed_witness[:43]}\n{changed_witness[43:]}.\n"

    for contains_witness in (review_pdf_witness, finalizer_pdf_witness):
        assert contains_witness(extracted, witness)
        assert not contains_witness(changed, witness)
        assert not contains_witness(extracted, "")


def test_clean_wheel_install_uses_the_pipless_environment_via_uv() -> None:
    root = Path(__file__).resolve().parent.parent

    result = _clean_wheel_install(root)

    imported = Path(result["import_path"])
    assert result["wheel"].endswith(".whl")
    assert imported.name == "__init__.py"
    assert imported.parent.name == "active_inference_power"


def test_pre_final_review_does_not_require_renderer_manifest_records(
    tmp_path: Path,
) -> None:
    """The final review can inspect a render before finalization records it."""
    root = Path(__file__).resolve().parent.parent
    output = tmp_path / "output"
    assert not _renderer_manifest_records_required({"complete": False}, require_render=True)
    assert _renderer_manifest_records_required({"complete": True}, require_render=True)
    assert not _renderer_manifest_records_required({"complete": True}, require_render=False)

    pre_final = required_release_artifacts(
        project_root=root,
        output_root=output,
        figure_filenames=(),
        checkpoint_names=(),
        require_render=False,
    )
    complete = required_release_artifacts(
        project_root=root,
        output_root=output,
        figure_filenames=(),
        checkpoint_names=(),
        require_render=True,
    )
    assert "release_review" not in pre_final
    assert complete["release_review"] == (
        output / "review" / "release_review.json",
        "release_review_receipt",
    )
    assert complete["renderer_provenance"] == (
        output / "review" / "renderer_provenance.json",
        "renderer_provenance_receipt",
    )


def _full_passing_review_report(output: Path) -> tuple[dict[str, object], dict[str, str]]:
    witnesses = {
        "config_sha256": "a" * 64,
        "claim_ledger_sha256": "b" * 64,
        "execution_source_sha256": "c" * 64,
        "render_input_sha256": "d" * 64,
    }
    rendered = {
        "pdf": output / "pdf" / "active_inference_power_combined.pdf",
        "html": output / "web" / "index.html",
        "combined_manuscript": output / "web" / "_combined_manuscript.md",
    }
    for name, path in rendered.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"real rendered {name}\n", encoding="utf-8")
    renderer_receipt = output / "review" / "renderer_provenance.json"
    renderer_receipt.parent.mkdir(parents=True, exist_ok=True)
    renderer_receipt.write_text('{"fixture": "renderer provenance"}\n', encoding="utf-8")
    check_names = (
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
    )
    details: dict[str, dict[str, object]] = {name: {} for name in check_names}
    details["schema_and_quality"] = {"run_id": "receipt-run", "profile": "release"}
    details["source_witnesses"] = dict(witnesses)
    details["resolved_manuscript"] = {"render_input_sha256": witnesses["render_input_sha256"]}
    details["manuscript_presentation"] = {
        "source_file_count": 2,
        "resolved_file_count": 2,
        "finding_count": 0,
    }
    details["documentation_links"] = {
        "document_count": 2,
        "local_link_count": 1,
    }
    details["rendered_text"] = {
        "rendered_artifacts": {name: sha256_file(path) for name, path in rendered.items()}
    }
    details["rendered_figure_alt_text"] = {
        "figure_count": len(FIGURE_NAMES),
        "rendered_html": str(rendered["html"]),
    }
    details["renderer_provenance"] = {
        "receipt_sha256": sha256_file(renderer_receipt),
        "template_git_head": "a" * 40,
        "template_dirty": False,
        "renderer_source_tree_sha256": "e" * 64,
    }
    return (
        {
            "schema_version": "1.0",
            "passed": True,
            "checks": [
                {"name": name, "passed": True, "details": details[name]}
                for name in check_names
            ],
        },
        witnesses,
    )


def test_manuscript_presentation_review_check_rejects_real_malformed_sources(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (output / "data").mkdir(parents=True)
    (output / "manuscript").mkdir(parents=True)
    (source / "00_abstract.md").write_text("FDR is 7%.\n", encoding="utf-8")
    (source / "01_body.md").write_text("Reader-facing body.\n", encoding="utf-8")
    (output / "data" / "manuscript_variables.json").write_text("{}\n", encoding="utf-8")
    (output / "manuscript" / "00_abstract.md").write_text(
        "FDR is 7%.\n", encoding="utf-8"
    )
    (output / "manuscript" / "01_body.md").write_text(
        "Reader-facing body.\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="manuscript presentation audit failed"):
        _require_passing_manuscript_presentation_audit(source, output)


def test_documentation_link_review_check_rejects_a_real_missing_destination(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    document = project / "README.md"
    document.write_text("[missing documentation](docs/missing.md)\n", encoding="utf-8")
    before = document.read_bytes()

    with pytest.raises(ValueError, match="documentation-link audit failed"):
        _require_passing_documentation_link_audit(project)

    assert document.read_bytes() == before


def test_full_review_rejects_a_failed_documentation_link_check(tmp_path: Path) -> None:
    report, _ = _full_passing_review_report(tmp_path)
    checks = cast(list[dict[str, object]], report["checks"])
    documentation = next(check for check in checks if check["name"] == "documentation_links")
    documentation["passed"] = False
    documentation["message"] = "real Markdown fixture has a missing local destination"
    report["passed"] = False

    with pytest.raises(ValueError, match="documentation_links"):
        require_full_passing_review(report)


def test_full_review_rejects_a_failed_manuscript_presentation_check(tmp_path: Path) -> None:
    report, _ = _full_passing_review_report(tmp_path)
    checks = cast(list[dict[str, object]], report["checks"])
    presentation = next(
        check for check in checks if check["name"] == "manuscript_presentation"
    )
    presentation["passed"] = False
    presentation["message"] = "real presentation fixture has findings"
    report["passed"] = False

    with pytest.raises(ValueError, match="manuscript_presentation"):
        require_full_passing_review(report)


def test_full_review_rejects_a_failed_rendered_figure_alt_text_check(
    tmp_path: Path,
) -> None:
    report, _ = _full_passing_review_report(tmp_path)
    checks = cast(list[dict[str, object]], report["checks"])
    rendered_alt = next(
        check for check in checks if check["name"] == "rendered_figure_alt_text"
    )
    rendered_alt["passed"] = False
    rendered_alt["message"] = "real rendered HTML substituted caption text"
    report["passed"] = False

    with pytest.raises(ValueError, match="rendered_figure_alt_text"):
        require_full_passing_review(report)


def test_full_review_rejects_a_failed_renderer_provenance_check(tmp_path: Path) -> None:
    report, _ = _full_passing_review_report(tmp_path)
    checks = cast(list[dict[str, object]], report["checks"])
    renderer = next(check for check in checks if check["name"] == "renderer_provenance")
    renderer["passed"] = False
    renderer["message"] = "sibling template checkout is unavailable"
    report["passed"] = False

    with pytest.raises(ValueError, match="renderer_provenance"):
        require_full_passing_review(report)


def test_passing_review_receipt_is_deterministic_and_hash_bound(tmp_path: Path) -> None:
    report, witnesses = _full_passing_review_report(tmp_path)

    first = write_passing_review_receipt(tmp_path, report)
    receipt_path = Path(first["path"])
    receipt = first["receipt"]
    assert isinstance(receipt, dict)
    before = receipt_path.read_bytes()
    validation = validate_review_receipt(
        receipt,
        output=tmp_path,
        expected_run_id="receipt-run",
        expected_profile="release",
        expected_witnesses=witnesses,
    )
    assert validation["run_id"] == "receipt-run"
    assert json.loads(before) == receipt

    second = write_passing_review_receipt(tmp_path, report)
    assert receipt_path.read_bytes() == before
    assert second["receipt"] == receipt

    (tmp_path / "web" / "index.html").write_text("changed render\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rendered html digest"):
        validate_review_receipt(
            receipt,
            output=tmp_path,
            expected_run_id="receipt-run",
            expected_profile="release",
            expected_witnesses=witnesses,
        )
