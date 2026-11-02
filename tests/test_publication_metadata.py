"""Fail-closed v1 publication metadata contracts for the source bundle."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

try:  # Python 3.11+ ships `tomllib`; the dev extra declares `tomli` for the floor.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - executed only on Python 3.10
    import tomli as tomllib

import pytest
import yaml

from active_inference_power.manuscript_variables import (
    _publication_release_identity,
    _publication_repository_statement,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_ROOT = PROJECT_ROOT.parent / "template"
TARGET_REPOSITORY = "ActiveInferenceInstitute/active_inference_power"
TARGET_REPOSITORY_URL = f"https://github.com/{TARGET_REPOSITORY}"
RELEASE_TAG = "v1.0.0"
RELEASE_URL = f"{TARGET_REPOSITORY_URL}/releases/tag/{RELEASE_TAG}"
CONCEPT_DOI = "10.5281/zenodo.21695160"
VERSION_DOI = "10.5281/zenodo.21695161"
VERSION_RECORD = "https://zenodo.org/records/21695161"
DEPOSITION_ID = "21695161"
TITLE = (
    "Active Inference Power Suite: Conditional Statistical Power under "
    "Controlled Generative Settings"
)
PROJECT_DESCRIPTION = (
    "Conditional adaptive-study design comparisons, multiple testing, and "
    "reproducible FDR evidence"
)


def _cover_doi_line(publication: dict[str, str]) -> str:
    """Exercise the sibling renderer's real cover-DOI implementation."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys\n"
                "from infrastructure.rendering._pdf_title_page_publishing "
                "import _publication_doi_line\n"
                "print(_publication_doi_line(json.load(sys.stdin)))\n"
            ),
        ],
        cwd=TEMPLATE_ROOT,
        input=json.dumps({"publication": publication}),
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _published_metadata() -> dict[str, str]:
    """Return a syntactically valid, internally consistent publication pair."""
    return {
        "status": "published",
        "github_repository": TARGET_REPOSITORY,
        "repository_url": TARGET_REPOSITORY_URL,
        "github_release_tag": RELEASE_TAG,
        "github_release_url": RELEASE_URL,
        "doi": "10.5281/zenodo.1234567",
        "version_doi": "10.5281/zenodo.1234568",
        "version_record": "https://zenodo.org/records/1234568",
        "zenodo_deposition_id": "1234568",
    }


def _config() -> dict[str, object]:
    loaded = yaml.safe_load(
        (PROJECT_ROOT / "manuscript" / "config.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(loaded, dict)
    return loaded


def _draft_publication() -> dict[str, str]:
    return {
        "status": "draft",
        "doi_status": "to be assigned at publication",
        "doi": "",
        "version_doi": "",
        "version_record": "",
        "zenodo_deposition_id": "",
        "github_repository": "",
        "repository_url": "",
        "github_release_tag": "",
        "github_release_url": "",
        "planned_github_repository": TARGET_REPOSITORY,
        "planned_repository_url": TARGET_REPOSITORY_URL,
    }


def test_published_v1_metadata_and_sidecars_are_source_bound() -> None:
    config = _config()
    publication = config["publication"]
    assert isinstance(publication, dict)
    assert publication["status"] == "published"
    assert publication["doi_status"] == "to be assigned at publication"
    assert publication["doi"] == CONCEPT_DOI
    assert publication["version_doi"] == VERSION_DOI
    assert publication["version_record"] == VERSION_RECORD
    assert publication["zenodo_deposition_id"] == DEPOSITION_ID
    assert publication["github_repository"] == TARGET_REPOSITORY
    assert publication["repository_url"] == TARGET_REPOSITORY_URL
    assert publication["github_release_tag"] == RELEASE_TAG
    assert publication["github_release_url"] == RELEASE_URL
    assert publication["planned_github_repository"] == TARGET_REPOSITORY
    assert publication["planned_repository_url"] == TARGET_REPOSITORY_URL

    statement = _publication_repository_statement(config)
    assert TARGET_REPOSITORY in statement
    assert "verified GitHub repository" in statement
    assert "doi" not in statement.lower()
    assert "zenodo" not in statement.lower()
    assert _publication_release_identity(config) == {
        "PUBLICATION_CONCEPT_DOI": CONCEPT_DOI,
        "PUBLICATION_VERSION_DOI": VERSION_DOI,
        "PUBLICATION_VERSION_RECORD": VERSION_RECORD,
        "PUBLICATION_GITHUB_REPOSITORY": TARGET_REPOSITORY,
        "PUBLICATION_REPOSITORY_URL": TARGET_REPOSITORY_URL,
        "PUBLICATION_GITHUB_RELEASE_TAG": RELEASE_TAG,
        "PUBLICATION_GITHUB_RELEASE_URL": RELEASE_URL,
    }

    cff = yaml.safe_load((PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    zenodo = json.loads((PROJECT_ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    codemeta = json.loads((PROJECT_ROOT / "codemeta.json").read_text(encoding="utf-8"))
    assert cff["title"] == TITLE
    assert cff["version"] == "1.0.0"
    assert cff["doi"] == CONCEPT_DOI
    assert cff["repository-code"] == TARGET_REPOSITORY_URL
    assert cff["license"] == "MIT"
    assert cff["authors"][0]["orcid"] == "0000-0001-6232-9096"
    assert zenodo["title"] == TITLE
    assert zenodo["version"] == "1.0.0"
    assert zenodo["license"] == "MIT"
    assert zenodo["related_identifiers"] == [
        {"identifier": CONCEPT_DOI, "relation": "isVersionOf", "scheme": "doi"}
    ]
    assert codemeta["name"] == TITLE
    assert codemeta["identifier"] == CONCEPT_DOI
    assert codemeta["codeRepository"] == TARGET_REPOSITORY_URL
    assert codemeta["license"] == "https://spdx.org/licenses/MIT"

    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "1.0.0"' in pyproject
    assert f'description = "{PROJECT_DESCRIPTION}"' in pyproject
    assert f'Repository = "{TARGET_REPOSITORY_URL}"' in pyproject
    assert f'"Zenodo concept DOI" = "https://doi.org/{CONCEPT_DOI}"' in pyproject
    assert "Statistical power of active inference" not in pyproject


def test_public_license_and_fixture_notice_match_declared_metadata() -> None:
    """The public source tree cannot declare licenses without their notices."""
    license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
    notices = (PROJECT_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    benchmark_readme = (PROJECT_ROOT / "data" / "benchmarks" / "README.md").read_text(
        encoding="utf-8"
    )
    manifest = yaml.safe_load(
        (PROJECT_ROOT / "data" / "benchmarks" / "manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert license_text.startswith("MIT License\n\nCopyright (c) 2026 Daniel Ari Friedman")
    assert "Permission is hereby granted, free of charge" in license_text
    assert "THE SOFTWARE IS PROVIDED \"AS IS\"" in license_text
    assert "[MIT License](LICENSE)" in readme
    assert "THIRD_PARTY_NOTICES.md" in readme
    assert "THIRD_PARTY_NOTICES.md" in benchmark_readme
    assert "golub_normalized_fixture.csv" in notices
    assert "public_agent_trace_adapter_stage_2.csv" in notices
    assert "LGPL" in notices
    assert "MIT" in notices
    assert pyproject["project"]["license"] == {"file": "LICENSE"}

    assert isinstance(manifest, dict)
    records = manifest["benchmarks"]
    assert isinstance(records, list)
    for record in records:
        assert isinstance(record, dict)
        assert str(record["filename"]) in notices
        assert str(record["license"]) in notices


def test_public_guidance_has_no_local_path_or_private_renderer_dependency() -> None:
    """Keep the published source usable without a maintainer-only checkout."""
    root_guide = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    controlled = (PROJECT_ROOT / "docs" / "controlled_renderer.md").read_text(
        encoding="utf-8"
    )
    public_reader_files = (
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "STANDALONE.md",
        PROJECT_ROOT / "docs" / "quickstart.md",
        PROJECT_ROOT / "manuscript" / "06_reproducibility.md",
    )

    assert "/Users/" not in root_guide
    assert "private sidecar" not in root_guide.lower()
    assert "not a package dependency" in controlled
    assert "GitHub and Zenodo release assets" in controlled
    for path in public_reader_files:
        text = path.read_text(encoding="utf-8")
        assert "../template" not in text
        assert "controlled" in text.lower()


def test_abstract_stays_repository_only_while_release_identity_is_elsewhere() -> None:
    source = (PROJECT_ROOT / "manuscript" / "00_abstract.md").read_text(encoding="utf-8")
    assert "{{PUBLICATION_REPOSITORY_STATEMENT}}" in source
    assert "{{PUBLICATION_ARCHIVE_STATEMENT}}" not in source
    assert "doi" not in source.lower()
    assert "zenodo" not in source.lower()

    reproducibility = (PROJECT_ROOT / "manuscript" / "06_reproducibility.md").read_text(
        encoding="utf-8"
    )
    for token in (
        "PUBLICATION_CONCEPT_DOI",
        "PUBLICATION_VERSION_DOI",
        "PUBLICATION_VERSION_RECORD",
        "PUBLICATION_GITHUB_RELEASE_TAG",
        "PUBLICATION_GITHUB_RELEASE_URL",
    ):
        assert f"{{{{{token}}}}}" in reproducibility


def test_cover_keeps_a_visible_draft_field_and_prefers_the_configured_concept_doi() -> None:
    """The renderer owns presentation; source configuration owns identity."""
    if not TEMPLATE_ROOT.is_dir():
        pytest.skip("cover rendering is an optional sibling-template integration check")
    draft_line = _cover_doi_line({"doi_status": "to be assigned at publication"})
    published_line = _cover_doi_line(
        {"doi": CONCEPT_DOI, "doi_status": "to be assigned at publication"}
    )

    assert draft_line == "DOI: to be assigned at publication"
    assert "to be assigned at publication" not in published_line
    assert CONCEPT_DOI in published_line
    assert f"https://doi.org/{CONCEPT_DOI}" in published_line


def test_verified_publication_pair_keeps_the_abstract_repository_only() -> None:
    config = deepcopy(_config())
    publication = config["publication"]
    assert isinstance(publication, dict)
    publication.update(_published_metadata())

    statement = _publication_repository_statement(config)

    assert TARGET_REPOSITORY in statement
    assert "verified GitHub repository" in statement
    assert "doi" not in statement.lower()
    assert "zenodo" not in statement.lower()


def test_partial_or_wrong_target_doi_metadata_cannot_render_as_published() -> None:
    config = deepcopy(_config())
    publication = config["publication"]
    assert isinstance(publication, dict)
    publication["doi"] = "10.5281/zenodo.1234567"
    publication["version_doi"] = ""
    with pytest.raises(ValueError, match="published metadata requires"):
        _publication_repository_statement(config)


@pytest.mark.parametrize(
    "field, value",
    [
        ("doi_status", "TBD"),
        ("status", "draft"),
        ("github_repository", "ActiveInferenceInstitute/not_the_planned_repository"),
        ("repository_url", "https://github.com/ActiveInferenceInstitute/not_the_planned_repository"),
        ("github_release_tag", "v0.1.0"),
        ("github_release_url", "https://github.com/ActiveInferenceInstitute/active_inference_power/releases/tag/v0.1.0"),
        ("doi", "10.5281/zenodo.not-a-record"),
        ("version_record", "https://zenodo.org/records/9999999"),
        ("zenodo_deposition_id", "9999999"),
    ],
)
def test_verified_metadata_rejects_wrong_repository_release_or_archive_fields(
    field: str, value: str
) -> None:
    config = deepcopy(_config())
    publication = config["publication"]
    assert isinstance(publication, dict)
    publication.update(_published_metadata())
    publication[field] = value

    with pytest.raises(
        ValueError,
        match=(
            "doi_status must be exactly"
            if field == "doi_status"
            else "published metadata requires"
        ),
    ):
        _publication_repository_statement(config)


def test_published_metadata_rejects_an_identical_concept_and_version_doi() -> None:
    config = deepcopy(_config())
    publication = config["publication"]
    assert isinstance(publication, dict)
    publication.update(_published_metadata())
    publication["version_doi"] = publication["doi"]
    publication["version_record"] = "https://zenodo.org/records/1234567"
    publication["zenodo_deposition_id"] = "1234567"

    with pytest.raises(ValueError, match="must differ"):
        _publication_repository_statement(config)


def test_draft_metadata_rejects_live_repository_release_or_archive_fields() -> None:
    config = deepcopy(_config())
    config["publication"] = _draft_publication()
    statement = _publication_repository_statement(config)
    assert "intended" in statement
    assert TARGET_REPOSITORY in statement
    identity = _publication_release_identity(config)
    assert all(value == "N/A" for value in identity.values())

    publication = config["publication"]
    assert isinstance(publication, dict)
    publication["repository_url"] = TARGET_REPOSITORY_URL
    with pytest.raises(ValueError, match="draft publication metadata"):
        _publication_repository_statement(config)


def test_unplanned_identifier_metadata_is_rejected_before_any_rendering() -> None:
    """An identifier cannot bypass the planned-repository identity contract."""
    with pytest.raises(ValueError, match="without a planned repository"):
        _publication_repository_statement({"publication": {"doi": CONCEPT_DOI}})
