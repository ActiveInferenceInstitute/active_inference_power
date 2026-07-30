"""Real-PNG tests for the read-only figure accessibility audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from scripts.audit_figure_accessibility import (
    audit_figure_accessibility,
    audit_rendered_figure_alts,
)
from scripts.sync_rendered_figure_alts import sync_rendered_figure_alts


def _write_registry(
    tmp_path: Path,
    *,
    redundancy: str = "Color is paired with direct labels.",
    alt: str = "A gray rectangular fixture image.",
) -> None:
    figures = tmp_path / "figures"
    figures.mkdir()
    plt.imsave(figures / "fixture.png", np.full((12, 16, 3), 0.5))
    registry = {
        "schema_version": "2.3",
        "figures": [
            {
                "filename": "fixture.png",
                "label": "fig:fixture",
                "caption": "A real temporary PNG fixture.",
                "alt": alt,
                "generated_by": "test",
                "reader_question": "Can a real fixture carry reader semantics?",
                "visible_comparison": "The raster is a fixed gray rectangle with no series comparison.",
                "takeaway": "The fixture supplies a deterministic audit target.",
                "visual_encoding": "A grayscale raster with text metadata; no color comparison is required.",
                "evaluation_perspective": "Fixture-only reader context; no agent or environment is evaluated.",
                "metadata": {
                    "estimand": "Fixture pixels.",
                    "uncertainty": "Not applicable.",
                    "evidence_class": "provenance_contract",
                    "validity_boundary": "Temporary test fixture only.",
                    "redundant_encoding": redundancy,
                    "sample_size": "not applicable",
                    "scenario_hashes": "not_applicable_fixed_horizon",
                    "result_lineage_hash": "a" * 64,
                },
            }
        ],
    }
    (figures / "figure_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )


def _upgrade_to_schema_2_4(registry: dict[str, object]) -> None:
    registry["schema_version"] = "2.4"
    figures = registry["figures"]
    assert isinstance(figures, list)
    assert isinstance(figures[0], dict)
    figures[0]["panel_validity"] = []


def test_accessibility_audit_accepts_real_png_with_complete_reader_metadata(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    report = audit_figure_accessibility(tmp_path, min_width=1, min_height=1)
    assert report["passed"] is True
    assert report["figure_count"] == 1


def test_accessibility_audit_reports_dimension_and_encoding_failures(tmp_path: Path) -> None:
    _write_registry(tmp_path, redundancy="Color alone distinguishes the series.")
    report = audit_figure_accessibility(tmp_path, min_width=640, min_height=480)
    assert report["passed"] is False
    assert {finding["rule"] for finding in report["findings"]} == {
        "minimum_dimensions",
        "redundant_encoding",
    }


def test_accessibility_audit_requires_complete_reader_first_metadata(
    tmp_path: Path,
) -> None:
    _write_registry(tmp_path)
    registry_path = tmp_path / "figures" / "figure_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["figures"][0]["reader_question"] = ""
    registry["figures"][0].pop("visible_comparison")
    registry["figures"][0].pop("takeaway")
    registry["figures"][0].pop("visual_encoding")
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    report = audit_figure_accessibility(tmp_path, min_width=1, min_height=1)

    assert report["passed"] is False
    assert {finding["rule"] for finding in report["findings"]} == {"registry"}
    assert "reader_question" in report["findings"][0]["message"]


def test_accessibility_audit_requires_evaluation_perspective_for_schema_2_3(
    tmp_path: Path,
) -> None:
    _write_registry(tmp_path)
    registry_path = tmp_path / "figures" / "figure_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["figures"][0].pop("evaluation_perspective")
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    report = audit_figure_accessibility(tmp_path, min_width=1, min_height=1)

    assert report["passed"] is False
    assert report["findings"] == [
        {
            "filename": "figure_registry.json",
            "rule": "registry",
            "message": (
                "figure_registry.figures[0].evaluation_perspective is required by schema 2.3"
            ),
        }
    ]


def test_accessibility_audit_accepts_historical_schema_2_1_without_visible_comparison(
    tmp_path: Path,
) -> None:
    _write_registry(tmp_path)
    registry_path = tmp_path / "figures" / "figure_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["schema_version"] = "2.1"
    registry["figures"][0].pop("visible_comparison")
    registry["figures"][0].pop("evaluation_perspective")
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    report = audit_figure_accessibility(tmp_path, min_width=1, min_height=1)

    assert report["passed"] is True


def test_accessibility_audit_accepts_schema_2_2_without_role_field(
    tmp_path: Path,
) -> None:
    """The new role field must not invalidate preserved reader-first receipts."""
    _write_registry(tmp_path)
    registry_path = tmp_path / "figures" / "figure_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["schema_version"] = "2.2"
    registry["figures"][0].pop("evaluation_perspective")
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    report = audit_figure_accessibility(tmp_path, min_width=1, min_height=1)

    assert report["passed"] is True


def test_accessibility_audit_requires_current_alternative_text_to_be_visual(
    tmp_path: Path,
) -> None:
    _write_registry(tmp_path, alt="A line chart of fixture pixels.")
    registry_path = tmp_path / "figures" / "figure_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    _upgrade_to_schema_2_4(registry)
    registry["figures"][0]["caption"] = "A caption distinct from the visual alternative."
    registry["figures"][0]["metadata"]["evidence_class"] = "conditional_operating_characteristic"
    registry["figures"][0]["metadata"]["uncertainty"] = "Finite-simulation interval."
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    report = audit_figure_accessibility(tmp_path, min_width=1, min_height=1)

    assert report["passed"] is True


def test_accessibility_audit_rejects_caption_duplicate_in_current_registry(
    tmp_path: Path,
) -> None:
    repeated_text = "A line chart of fixture pixels."
    _write_registry(tmp_path, alt=repeated_text)
    registry_path = tmp_path / "figures" / "figure_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    _upgrade_to_schema_2_4(registry)
    registry["figures"][0]["caption"] = repeated_text
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    report = audit_figure_accessibility(tmp_path, min_width=1, min_height=1)

    assert report["passed"] is False
    assert report["findings"] == [
        {
            "filename": "fixture.png",
            "rule": "alt_equals_caption",
            "message": "authored alternative text must visually describe the figure rather than repeat the caption",
        }
    ]


def test_accessibility_audit_rejects_nonvisual_quantitative_alternative_text(
    tmp_path: Path,
) -> None:
    _write_registry(tmp_path, alt="A summary figure.")
    registry_path = tmp_path / "figures" / "figure_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    _upgrade_to_schema_2_4(registry)
    registry["figures"][0]["caption"] = "A caption distinct from the alternative text."
    registry["figures"][0]["metadata"]["evidence_class"] = "conditional_operating_characteristic"
    registry["figures"][0]["metadata"]["uncertainty"] = "Finite-simulation interval."
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    report = audit_figure_accessibility(tmp_path, min_width=1, min_height=1)

    assert report["passed"] is False
    assert report["findings"] == [
        {
            "filename": "fixture.png",
            "rule": "alt_visual_content",
            "message": "quantitative figure alternative text must name a visible axis, mark, panel, or other visual encoding",
        }
    ]


def test_rendered_html_alt_audit_binds_registry_authored_text(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    rendered = tmp_path / "web" / "index.html"
    rendered.parent.mkdir()
    rendered.write_text(
        '<figure id="fig:fixture"><img src="../figures/fixture.png" '
        'alt="A gray rectangular fixture image."></figure>\n',
        encoding="utf-8",
    )

    report = audit_rendered_figure_alts(tmp_path)

    assert report["passed"] is True
    assert report["figure_count"] == 1


def test_rendered_html_alt_audit_rejects_caption_substitution(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    rendered = tmp_path / "web" / "index.html"
    rendered.parent.mkdir()
    rendered.write_text(
        '<figure id="fig:fixture"><img src="../figures/fixture.png" '
        'alt="A real temporary PNG fixture."></figure>\n',
        encoding="utf-8",
    )

    report = audit_rendered_figure_alts(tmp_path)

    assert report["passed"] is False
    assert report["findings"] == [
        {
            "filename": "fixture.png",
            "rule": "rendered_alt_mismatch",
            "message": "rendered HTML alternative text differs from the authored figure-registry value",
        }
    ]


def test_auxiliary_section_pages_are_opt_in_and_can_be_synchronized(tmp_path: Path) -> None:
    authored_alt = "A gray rectangular fixture image."
    _write_registry(tmp_path, alt=authored_alt)
    web = tmp_path / "web"
    web.mkdir()
    index = web / "index.html"
    section = web / "manuscript__03_results.html"
    index.write_text(
        '<img src="../figures/fixture.png" alt="A gray rectangular fixture image.">\n',
        encoding="utf-8",
    )
    section.write_text(
        '<img src="../figures/fixture.png" '
        'alt="Can a real fixture carry reader semantics?">\n'
        '<img src="../assets/template-logo.svg" alt="Template logo remains untouched.">\n',
        encoding="utf-8",
    )

    default_receipt = sync_rendered_figure_alts(tmp_path)
    assert default_receipt["changed"] is False
    assert "Can a real fixture carry reader semantics?" in section.read_text(encoding="utf-8")
    assert audit_rendered_figure_alts(tmp_path)["passed"] is True

    before_sync = audit_rendered_figure_alts(tmp_path, include_auxiliary_pages=True)
    assert before_sync["passed"] is False
    assert before_sync["auxiliary_page_count"] == 1
    assert before_sync["auxiliary_pages"][0]["matched_image_count"] == 1
    assert before_sync["findings"] == [
        {
            "filename": "manuscript__03_results.html:fixture.png",
            "rule": "rendered_alt_mismatch",
            "message": "rendered HTML alternative text differs from the authored figure-registry value",
        }
    ]

    receipt = sync_rendered_figure_alts(tmp_path, include_auxiliary_pages=True)
    synchronized = section.read_text(encoding="utf-8")
    assert receipt["changed"] is True
    assert receipt["matched_image_count"] == 2
    assert receipt["updated_image_count"] == 1
    assert receipt["auxiliary_page_count"] == 1
    assert receipt["auxiliary_pages"][0]["matched_image_count"] == 1
    assert 'alt="A gray rectangular fixture image."' in synchronized
    assert 'alt="Template logo remains untouched."' in synchronized

    all_pages = audit_rendered_figure_alts(tmp_path, include_auxiliary_pages=True)
    assert all_pages["passed"] is True
    assert all_pages["auxiliary_pages"][0]["matched_image_count"] == 1


def test_post_render_synchronizer_updates_only_registered_image_alternative_text(
    tmp_path: Path,
) -> None:
    authored_alt = 'A gray "fixture" image with an ampersand & a direct label.'
    _write_registry(tmp_path, alt=authored_alt)
    rendered = tmp_path / "web" / "index.html"
    rendered.parent.mkdir()
    rendered.write_text(
        "<main>\n"
        '  <img class="publication-figure" src="../figures/fixture.png" '
        'alt="A real temporary PNG fixture.">\n'
        '  <img class="template-logo" src="../assets/template-logo.svg" '
        'alt="Template logo stays untouched.">\n'
        '  <!-- <img src="../figures/fixture.png" alt="Comment remains literal."> -->\n'
        '  <script>const example = \'<img src="../figures/fixture.png" '
        'alt="Script literal remains untouched.\'>;</script>\n'
        "</main>\n",
        encoding="utf-8",
    )

    receipt = sync_rendered_figure_alts(tmp_path)
    synchronized = rendered.read_text(encoding="utf-8")

    assert receipt["changed"] is True
    assert receipt["registered_figure_count"] == 1
    assert receipt["matched_image_count"] == 1
    assert receipt["updated_image_count"] == 1
    assert len(receipt["synchronizer_sha256"]) == 64
    assert (
        'alt="A gray &quot;fixture&quot; image with an ampersand &amp; a direct label."'
        in synchronized
    )
    assert 'alt="Template logo stays untouched."' in synchronized
    assert 'alt="Comment remains literal."' in synchronized
    assert "Script literal remains untouched." in synchronized
    assert audit_rendered_figure_alts(tmp_path)["passed"] is True


def test_post_render_synchronizer_adds_missing_registered_alt_and_is_idempotent(
    tmp_path: Path,
) -> None:
    _write_registry(tmp_path)
    rendered = tmp_path / "web" / "index.html"
    rendered.parent.mkdir()
    rendered.write_text(
        '<img src="../figures/fixture.png" class="publication-figure" />\n'
        '<img src="../figures/fixture.png" alt=caption class="second-copy">\n',
        encoding="utf-8",
    )

    first = sync_rendered_figure_alts(tmp_path)
    after_first = rendered.read_bytes()
    second = sync_rendered_figure_alts(tmp_path)

    assert first["changed"] is True
    assert first["matched_image_count"] == 2
    assert first["updated_image_count"] == 2
    assert b'alt="A gray rectangular fixture image."' in after_first
    assert second["changed"] is False
    assert second["updated_image_count"] == 0
    assert second["html_before_sha256"] == second["html_after_sha256"]
    assert rendered.read_bytes() == after_first


def test_current_generated_figure_bundle_meets_accessibility_contract(
    generated_fast_bundle: Path,
) -> None:
    report = audit_figure_accessibility(generated_fast_bundle)
    assert report["passed"] is True, report["findings"]
