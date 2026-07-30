"""Real-file coverage for the documentation navigation contract."""

from __future__ import annotations

from pathlib import Path

from scripts.audit_documentation_links import audit_documentation_links


ROOT = Path(__file__).resolve().parent.parent


def test_project_local_markdown_destinations_are_resolved() -> None:
    report = audit_documentation_links(ROOT)
    assert report["passed"] is True
    assert report["document_count"] >= 20
    assert report["local_link_count"] >= 30


def test_documentation_link_audit_reports_missing_local_destinations(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("[missing](docs/nope.md)\n", encoding="utf-8")
    report = audit_documentation_links(tmp_path)
    assert report["passed"] is False
    assert report["findings"] == [
        {
            "document": "README.md",
            "destination": "docs/nope.md",
            "message": "local Markdown destination does not exist",
        }
    ]


def test_documentation_link_audit_prunes_generated_output_before_traversal(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("[home](README.md)\n", encoding="utf-8")
    generated = tmp_path / "output" / "retained-release" / "manuscript"
    generated.mkdir(parents=True)
    (generated / "README.md").write_text("[stale](missing.md)\n", encoding="utf-8")

    report = audit_documentation_links(tmp_path)

    assert report["passed"] is True
    assert report["document_count"] == 1
    assert report["local_link_count"] == 1


def test_documentation_link_audit_does_not_treat_remote_uris_as_local_paths(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text(
        "[remote](https://example.org/guide) [anchor](#section)\n",
        encoding="utf-8",
    )

    report = audit_documentation_links(tmp_path)

    assert report["passed"] is True
    assert report["local_link_count"] == 0


def test_documentation_link_audit_defers_hydrated_url_tokens_to_the_rendered_check(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text(
        "[record]({{PUBLICATION_VERSION_RECORD}})\n", encoding="utf-8"
    )

    report = audit_documentation_links(tmp_path)

    assert report["passed"] is True
    assert report["local_link_count"] == 0
