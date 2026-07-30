"""Real-file contracts for the deterministic manuscript presentation audit."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from active_inference_power.manuscript_variables import (
    _draft_variables,
    _load_config,
    generate_variables,
)
from scripts.audit_manuscript_presentation import audit_manuscript_presentation


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "audit_manuscript_presentation.py"


def _write_fixture(
    root: Path,
    *,
    abstract: str,
    body: str = "# Body\n\nFalse discovery rate (FDR) is defined here.\n",
    variables: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    manuscript = root / "manuscript"
    manuscript.mkdir()
    (manuscript / "00_abstract.md").write_text(abstract, encoding="utf-8")
    (manuscript / "01_body.md").write_text(body, encoding="utf-8")
    variable_file = root / "variables.json"
    variable_file.write_text(
        json.dumps(variables if variables is not None else {}, sort_keys=True),
        encoding="utf-8",
    )
    return manuscript, variable_file


def test_current_manuscript_source_presentation_audit_is_clean(
    tmp_path: Path,
    generated_fast_bundle: Path,
) -> None:
    """Lint source prose against variables from a fresh real evidence bundle."""
    variables = generate_variables(
        PROJECT_ROOT,
        require_analysis_outputs=True,
        output_root=generated_fast_bundle,
    )
    variables.update(_draft_variables(_load_config(PROJECT_ROOT)))
    variable_file = tmp_path / "current-source-variables.json"
    variable_file.write_text(json.dumps(variables, sort_keys=True), encoding="utf-8")
    report = audit_manuscript_presentation(
        PROJECT_ROOT / "manuscript",
        variable_file,
    )
    assert report["summary"] == {"finding_count": 0, "by_code": {}}
    assert report["findings"] == []


def test_fixture_reports_each_presentation_failure_class_deterministically(
    tmp_path: Path,
) -> None:
    manuscript, variable_file = _write_fixture(
        tmp_path,
        abstract=(
            "# Abstract\n\n"
            "FDR is 42% [@source2024], as Figure fig. demonstrates. "
            "{{UNKNOWN_TOKEN}}\n"
        ),
    )
    resolved = tmp_path / "resolved"
    resolved.mkdir()
    (resolved / "00_abstract.md").write_text(
        "# Abstract\n\n{{LEFTOVER}}\n", encoding="utf-8"
    )

    first = audit_manuscript_presentation(
        manuscript, variable_file, resolved_manuscript_dir=resolved
    )
    second = audit_manuscript_presentation(
        manuscript, variable_file, resolved_manuscript_dir=resolved
    )

    assert first == second
    assert first["summary"] == {
        "finding_count": 6,
        "by_code": {
            "abstract_citation": 1,
            "acronym_first_use": 1,
            "duplicate_cross_reference": 1,
            "numeric_literal": 1,
            "unresolved_token": 2,
        },
    }
    findings = first["findings"]
    assert {finding["code"] for finding in findings} == {
        "abstract_citation",
        "acronym_first_use",
        "duplicate_cross_reference",
        "numeric_literal",
        "unresolved_token",
    }
    assert any(
        finding["path"] == "resolved/00_abstract.md"
        and finding["value"] == "{{LEFTOVER}}"
        for finding in findings
    )


def test_numeric_exclusions_and_declared_source_token_are_not_findings(
    tmp_path: Path,
) -> None:
    manuscript, variable_file = _write_fixture(
        tmp_path,
        abstract=(
            "# Abstract\n\n"
            "False discovery rate (FDR) is stated qualitatively; see @sec:body. "
            "{{PUBLICATION_REPOSITORY_STATEMENT}}\n"
        ),
        body=(
            "# Body\n\n"
            "The equation is $x_0 = 1$. [@source2024]\n\n"
            "![caption](figure.png){#fig:example width=95% height=0.38\\textheight}\n\n"
            "1. A structural ordered-list marker.\n\n"
            "## S1. Supplement label\n\n"
            "The Secure Hash Algorithm digest (SHA-256) is structural syntax.\n"
        ),
        variables={"PUBLICATION_REPOSITORY_STATEMENT": "Prepared for release."},
    )

    report = audit_manuscript_presentation(manuscript, variable_file)

    assert report["summary"] == {"finding_count": 0, "by_code": {}}


def test_abstract_reader_focus_rejects_doi_quality_and_result_receipt_language(
    tmp_path: Path,
) -> None:
    manuscript, variable_file = _write_fixture(
        tmp_path,
        abstract=(
            "# Abstract\n\n"
            "False discovery rate (FDR) is qualitative. Digital object identifier "
            "(DOI) and Zenodo status appear here. Branch coverage is reported with "
            "{{QUALITY_BRANCH_COVERAGE}} and {{RESULT_BH_FDR}}.\n"
        ),
        variables={
            "QUALITY_BRANCH_COVERAGE": "95.00",
            "RESULT_BH_FDR": "0.05",
        },
    )

    report = audit_manuscript_presentation(manuscript, variable_file)

    assert report["summary"]["by_code"] == {
        "abstract_doi_status": 3,
        "abstract_internal_quality": 2,
        "abstract_result_token": 1,
    }


def test_command_writes_same_machine_readable_report_and_nonzero_for_findings(
    tmp_path: Path,
) -> None:
    manuscript, variable_file = _write_fixture(
        tmp_path,
        abstract="# Abstract\n\nFDR is 7%.\n",
    )
    report_path = tmp_path / "presentation-report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manuscript-dir",
            str(manuscript),
            "--variables-file",
            str(variable_file),
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout) == json.loads(
        report_path.read_text(encoding="utf-8")
    )
    assert json.loads(completed.stdout)["summary"]["by_code"] == {
        "acronym_first_use": 1,
        "numeric_literal": 1,
    }
