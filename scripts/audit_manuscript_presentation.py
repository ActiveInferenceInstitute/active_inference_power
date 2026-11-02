#!/usr/bin/env python3
"""Audit deterministic presentation contracts for the manuscript source.

The source manuscript intentionally contains ``{{TOKEN}}`` placeholders.  A
placeholder is therefore accepted only when it is declared in the generated
manuscript-variable map.  Passing ``--resolved-manuscript-dir`` additionally
checks that the hydrated tree contains no placeholder at all.

The command writes a stable JSON report to stdout and optionally to ``--report``.
It exits with status 1 when findings are present and status 2 for invalid input.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Final, TypedDict


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# ``re.NOFLAG`` is Python 3.11+; the project supports 3.10 (``requires-python``),
# where the same "no flags" value is ``re.RegexFlag(0)``.
_NO_FLAGS: Final[re.RegexFlag] = getattr(re, "NOFLAG", re.RegexFlag(0))
_SOURCE_EXCLUSIONS: Final[frozenset[str]] = frozenset(
    {"AGENTS.md", "README.md", "SYNTAX.md", "preamble.md"}
)
_INTERNAL_REFERENCE_PREFIXES: Final[tuple[str, ...]] = ("fig:", "sec:", "eq:", "tbl:")
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
_BRACED_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\{\{[^\n}]*\}\}")
_OPEN_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\{\{")
_CITATION_RE: Final[re.Pattern[str]] = re.compile(r"@([A-Za-z][A-Za-z0-9_:-]*)")
_NUMERIC_LITERAL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9_])(?:\d+(?:\.\d+)?|\.\d+)(?:%|[eE][+-]?\d+)?(?![A-Za-z0-9_])"
)
_ABSTRACT_DOI_STATUS_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:doi|zenodo|digital\s+object\s+identifier)\b", re.IGNORECASE
)
_ABSTRACT_INTERNAL_QUALITY_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:branch|line)\s+coverage\b|\btest\s+count\b|\{\{QUALITY_[A-Z0-9_]*\}\}",
    re.IGNORECASE,
)
_ABSTRACT_RESULT_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"\{\{RESULT_[A-Z0-9_]*\}\}"
)


@dataclass(frozen=True)
class AcronymDefinition:
    """One reader-facing abbreviation whose first use must be expanded."""

    acronym: str
    expansion: str
    expansion_pattern: str


# This is intentionally a small, declared registry rather than a heuristic for
# every all-caps word.  Add a term here when it becomes public manuscript
# vocabulary, along with the exact expansion that readers should encounter.
ACRONYM_REGISTRY: Final[tuple[AcronymDefinition, ...]] = (
    AcronymDefinition(
        "API",
        "Application programming interface",
        r"Application\s+programming\s+interface",
    ),
    AcronymDefinition("BH", "Benjamini-Hochberg", r"Benjamini(?:[-–—]|--)\s*Hochberg"),
    AcronymDefinition(
        "BY", "Benjamini-Yekutieli", r"Benjamini(?:[-–—]|--)\s*Yekutieli"
    ),
    AcronymDefinition(
        "CDF",
        "Cumulative distribution function",
        r"Cumulative\s+distribution\s+function",
    ),
    AcronymDefinition("CS", "Confidence sequence", r"Confidence\s+sequence"),
    AcronymDefinition(
        "DKW",
        "Dvoretzky-Kiefer-Wolfowitz",
        r"Dvoretzky(?:[-–—]|--)\s*Kiefer(?:[-–—]|--)\s*Wolfowitz",
    ),
    AcronymDefinition(
        "DOI", "Digital object identifier", r"Digital\s+object\s+identifier"
    ),
    AcronymDefinition("DPI", "Dots per inch", r"Dots\s+per\s+inch"),
    AcronymDefinition(
        "FAIR",
        "Findable, accessible, interoperable, and reusable",
        r"Findable,\s*accessible,\s*interoperable,\s*and\s+reusable",
    ),
    AcronymDefinition(
        "FDP", "False discovery proportion", r"False\s+discovery\s+proportion"
    ),
    AcronymDefinition("FDR", "False discovery rate", r"False\s+discovery\s+rate"),
    AcronymDefinition(
        "FWER", "Family-wise error rate", r"Family(?:[-–—]|--)?wise\s+error\s+rate"
    ),
    AcronymDefinition(
        "HTML", "HyperText Markup Language", r"HyperText\s+Markup\s+Language"
    ),
    AcronymDefinition(
        "JSON", "JavaScript Object Notation", r"JavaScript\s+Object\s+Notation"
    ),
    AcronymDefinition(
        "LOND",
        "Levels based On Number of Discoveries",
        r"Levels\s+based\s+On\s+Number\s+of\s+Discoveries",
    ),
    AcronymDefinition(
        "LORD",
        "Levels based On Recent Discovery",
        r"Levels\s+based\s+On\s+Recent\s+Discovery",
    ),
    AcronymDefinition(
        "MCSE", "Monte Carlo standard error", r"Monte\s+Carlo\s+standard\s+error"
    ),
    AcronymDefinition(
        "PDF", "Portable Document Format", r"Portable\s+Document\s+Format"
    ),
    AcronymDefinition(
        "PRDS",
        "Positive regression dependence on a subset",
        r"Positive\s+regression\s+dependence\s+on\s+a\s+subset",
    ),
    AcronymDefinition(
        "SAFFRON",
        "Serial estimate of the Alpha Fraction that is Futilely Rationed On true null hypotheses",
        r"Serial\s+estimate\s+of\s+the\s+Alpha\s+Fraction\s+that\s+is\s+Futilely\s+Rationed\s+On\s+true\s+null\s+hypotheses",
    ),
    AcronymDefinition("SE", "Standard error", r"Standard\s+error"),
    AcronymDefinition(
        "SHA-256",
        "Secure Hash Algorithm digest",
        r"Secure\s+Hash\s+Algorithm(?:\s+digest)?",
    ),
    AcronymDefinition("TPR", "True-positive rate", r"True(?:[-–—]|--)?positive\s+rate"),
    AcronymDefinition(
        "URL", "Uniform resource locator", r"Uniform\s+resource\s+locator"
    ),
)


@dataclass(frozen=True)
class SourceDocument:
    """A manuscript source file plus the path used in a stable report."""

    path: Path
    relative_path: str
    text: str


@dataclass(frozen=True)
class Finding:
    """One deterministic, source-located presentation violation."""

    code: str
    path: str
    line: int
    column: int
    message: str
    value: str


class FindingRecord(TypedDict):
    """JSON-serializable form of a presentation finding."""

    code: str
    path: str
    line: int
    column: int
    message: str
    value: str


class PresentationSummary(TypedDict):
    """Counted result of a presentation audit."""

    finding_count: int
    by_code: dict[str, int]


class PresentationReport(TypedDict):
    """Stable report schema emitted by the standalone command."""

    schema_version: int
    files_checked: list[str]
    resolved_files_checked: list[str]
    acronym_registry: list[str]
    summary: PresentationSummary
    findings: list[FindingRecord]


def _line_and_column(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    previous_newline = text.rfind("\n", 0, offset)
    return line, offset - previous_newline


def _finding(
    *, code: str, document: SourceDocument, offset: int, message: str, value: str
) -> Finding:
    line, column = _line_and_column(document.text, offset)
    return Finding(
        code=code,
        path=document.relative_path,
        line=line,
        column=column,
        message=message,
        value=value,
    )


def _documents(
    manuscript_dir: Path, *, exclude_source_support: bool
) -> list[SourceDocument]:
    if not manuscript_dir.is_dir():
        raise ValueError(f"manuscript directory does not exist: {manuscript_dir}")
    paths = [
        path
        for path in sorted(manuscript_dir.glob("*.md"))
        if not exclude_source_support or path.name not in _SOURCE_EXCLUSIONS
    ]
    if not paths:
        raise ValueError(f"no manuscript Markdown files found in: {manuscript_dir}")
    return [
        SourceDocument(
            path=path,
            relative_path=path.relative_to(manuscript_dir).as_posix(),
            text=path.read_text(encoding="utf-8"),
        )
        for path in paths
    ]


def _load_variable_names(variables_file: Path) -> frozenset[str]:
    if not variables_file.is_file():
        raise ValueError(
            f"manuscript-variable map does not exist: {variables_file}; run "
            "scripts/z_generate_manuscript_variables.py first"
        )
    try:
        payload = json.loads(variables_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid manuscript-variable JSON: {variables_file}") from exc
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise ValueError(
            f"manuscript-variable map must be a JSON object: {variables_file}"
        )
    return frozenset(payload)


def _acronym_token_pattern(acronym: str) -> re.Pattern[str]:
    # The registry contains uppercase abbreviations.  Matching case-insensitively
    # would mistake ordinary prose such as "by" for the BY procedure.
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(acronym)}(?![A-Za-z0-9_])")


def _is_expanded_first_use(
    text: str, occurrence: re.Match[str], definition: AcronymDefinition
) -> bool:
    pattern = re.compile(
        rf"{definition.expansion_pattern}\s*\(\s*(?P<acronym>{re.escape(definition.acronym)})\s*\)",
        flags=re.IGNORECASE,
    )
    for expanded in pattern.finditer(text):
        start, end = expanded.span("acronym")
        if start <= occurrence.start() < end:
            return True
    return False


def _audit_acronym_first_use(documents: list[SourceDocument]) -> list[Finding]:
    findings: list[Finding] = []
    for definition in ACRONYM_REGISTRY:
        occurrence_document: SourceDocument | None = None
        occurrence: re.Match[str] | None = None
        token_pattern = _acronym_token_pattern(definition.acronym)
        for document in documents:
            # Placeholder identifiers such as ``RESULT_BY_FDR`` are build
            # syntax, not a reader-facing first use of the BY abbreviation.
            presentation_text = _mask(
                _presentation_text(document.text), r"\{\{[^}\n]*\}\}"
            )
            found = token_pattern.search(presentation_text)
            if found is not None:
                occurrence_document = document
                occurrence = found
                break
        if occurrence_document is None or occurrence is None:
            continue
        presentation_text = _mask(
            _presentation_text(occurrence_document.text), r"\{\{[^}\n]*\}\}"
        )
        if not _is_expanded_first_use(presentation_text, occurrence, definition):
            findings.append(
                _finding(
                    code="acronym_first_use",
                    document=occurrence_document,
                    offset=occurrence.start(),
                    message=(
                        f"first use of {definition.acronym} must expand to "
                        f"'{definition.expansion} ({definition.acronym})'"
                    ),
                    value=definition.acronym,
                )
            )
    return findings


def _audit_abstract_citations(documents: list[SourceDocument]) -> list[Finding]:
    abstract = next(
        (document for document in documents if document.path.name == "00_abstract.md"),
        None,
    )
    if abstract is None:
        raise ValueError("source manuscript must contain 00_abstract.md")
    findings: list[Finding] = []
    for citation in _CITATION_RE.finditer(abstract.text):
        key = citation.group(1)
        if key.lower().startswith(_INTERNAL_REFERENCE_PREFIXES):
            continue
        findings.append(
            _finding(
                code="abstract_citation",
                document=abstract,
                offset=citation.start(),
                message="the abstract must be citation-free",
                value=f"@{key}",
            )
        )
    return findings


def _audit_abstract_reader_focus(documents: list[SourceDocument]) -> list[Finding]:
    """Keep the abstract conceptual rather than an internal release receipt."""
    abstract = next(
        (document for document in documents if document.path.name == "00_abstract.md"),
        None,
    )
    if abstract is None:
        raise ValueError("source manuscript must contain 00_abstract.md")
    findings: list[Finding] = []
    checks: tuple[tuple[str, re.Pattern[str], str], ...] = (
        (
            "abstract_doi_status",
            _ABSTRACT_DOI_STATUS_RE,
            "the abstract must not describe DOI or Zenodo publication status",
        ),
        (
            "abstract_internal_quality",
            _ABSTRACT_INTERNAL_QUALITY_RE,
            "the abstract must not report internal test-count or coverage evidence",
        ),
        (
            "abstract_result_token",
            _ABSTRACT_RESULT_TOKEN_RE,
            "the abstract must state qualitative findings rather than injected result values",
        ),
    )
    for code, pattern, message in checks:
        for match in pattern.finditer(abstract.text):
            findings.append(
                _finding(
                    code=code,
                    document=abstract,
                    offset=match.start(),
                    message=message,
                    value=match.group(0),
                )
            )
    return findings


_DUPLICATE_CROSS_REFERENCE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"\bFigure\s+(?P<token>@?fig(?:\.[A-Za-z0-9_-]*|:[A-Za-z0-9_-]+))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bTable\s+(?P<token>@?tbl(?:\.[A-Za-z0-9_-]*|:[A-Za-z0-9_-]+))", re.IGNORECASE
    ),
    re.compile(
        r"\bSection\s+(?P<token>@?sec(?:\.[A-Za-z0-9_-]*|:[A-Za-z0-9_-]+))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bEquation\s+(?P<token>@?eq(?:\.[A-Za-z0-9_-]*|:[A-Za-z0-9_-]+))",
        re.IGNORECASE,
    ),
)


def _blank_preserving_newlines(value: str) -> str:
    return re.sub(r"[^\n]", " ", value)


def _mask(text: str, pattern: str, *, flags: re.RegexFlag = _NO_FLAGS) -> str:
    return re.sub(
        pattern,
        lambda match: _blank_preserving_newlines(match.group(0)),
        text,
        flags=flags,
    )


def _presentation_text(text: str) -> str:
    """Mask code examples before detecting duplicate cross-reference prose."""
    masked = _mask(text, r"```.*?```", flags=re.DOTALL)
    return _mask(masked, r"`[^`\n]*`")


def _audit_duplicate_cross_references(documents: list[SourceDocument]) -> list[Finding]:
    findings: list[Finding] = []
    for document in documents:
        presentation_text = _presentation_text(document.text)
        for pattern in _DUPLICATE_CROSS_REFERENCE_PATTERNS:
            for reference in pattern.finditer(presentation_text):
                findings.append(
                    _finding(
                        code="duplicate_cross_reference",
                        document=document,
                        offset=reference.start(),
                        message=(
                            "do not combine a rendered reference label with a "
                            "figure/table/section/equation token"
                        ),
                        value=reference.group(0),
                    )
                )
    return findings


def _audit_source_tokens(
    documents: list[SourceDocument], variable_names: frozenset[str]
) -> list[Finding]:
    findings: list[Finding] = []
    for document in documents:
        braced_starts: set[int] = set()
        for braced in _BRACED_TOKEN_RE.finditer(document.text):
            braced_starts.add(braced.start())
            token = _TOKEN_RE.fullmatch(braced.group(0))
            if token is not None and token.group(1) in variable_names:
                continue
            value = braced.group(0)
            message = (
                "manuscript token is malformed or absent from the declared variable map"
            )
            findings.append(
                _finding(
                    code="unresolved_token",
                    document=document,
                    offset=braced.start(),
                    message=message,
                    value=value,
                )
            )
        for opening in _OPEN_TOKEN_RE.finditer(document.text):
            if opening.start() in braced_starts:
                continue
            findings.append(
                _finding(
                    code="unresolved_token",
                    document=document,
                    offset=opening.start(),
                    message="manuscript token has no closing braces",
                    value="{{",
                )
            )
    return findings


def _audit_resolved_tokens(
    resolved_manuscript_dir: Path,
) -> tuple[list[Finding], list[str]]:
    findings: list[Finding] = []
    documents = _documents(resolved_manuscript_dir, exclude_source_support=False)
    for document in documents:
        for braced in _BRACED_TOKEN_RE.finditer(document.text):
            findings.append(
                _finding(
                    code="unresolved_token",
                    document=SourceDocument(
                        path=document.path,
                        relative_path=f"resolved/{document.relative_path}",
                        text=document.text,
                    ),
                    offset=braced.start(),
                    message="resolved manuscript must not contain a template token",
                    value=braced.group(0),
                )
            )
        braced_starts = {
            match.start() for match in _BRACED_TOKEN_RE.finditer(document.text)
        }
        for opening in _OPEN_TOKEN_RE.finditer(document.text):
            if opening.start() in braced_starts:
                continue
            findings.append(
                _finding(
                    code="unresolved_token",
                    document=SourceDocument(
                        path=document.path,
                        relative_path=f"resolved/{document.relative_path}",
                        text=document.text,
                    ),
                    offset=opening.start(),
                    message="resolved manuscript token has no closing braces",
                    value="{{",
                )
            )
    return findings, [document.relative_path for document in documents]


def _numeric_mask(text: str) -> str:
    """Remove declared non-prose numeric contexts without changing offsets."""
    masked = _presentation_text(text)
    for pattern, flags in (
        (r"\\\[.*?\\\]", re.DOTALL),
        (r"\\begin\{[^}\n]+\}.*?\\end\{[^}\n]+\}", re.DOTALL),
        (r"\\\(.*?\\\)", re.DOTALL),
        (r"\$\$.*?\$\$", re.DOTALL),
        (r"(?<!\\)\$(?!\$)(?:\\.|[^$\n])*?(?<!\\)\$", _NO_FLAGS),
        (r"\[[^\]\n]*@[^\]\n]*\]", _NO_FLAGS),
        (r"@[A-Za-z][A-Za-z0-9_:-]*", _NO_FLAGS),
        (r"\]\([^\)\n]*\)", _NO_FLAGS),
        (r"\{(?:#|\.)[^}\n]*\}", _NO_FLAGS),
        (r"\{\{[^}\n]*\}\}", _NO_FLAGS),
        (r"\b(?:width|height)=[^\s}\]]+", re.IGNORECASE),
        (r"\bSHA-\d+\b", re.IGNORECASE),
        (r"(?m)^\s*\d+[.)]\s+", _NO_FLAGS),
        (r"(?m)^#{1,6}\s+S\d+(?=[.\s])", _NO_FLAGS),
    ):
        masked = _mask(masked, pattern, flags=flags)
    return masked


def _audit_numeric_literals(documents: list[SourceDocument]) -> list[Finding]:
    findings: list[Finding] = []
    for document in documents:
        masked = _numeric_mask(document.text)
        for literal in _NUMERIC_LITERAL_RE.finditer(masked):
            findings.append(
                _finding(
                    code="numeric_literal",
                    document=document,
                    offset=literal.start(),
                    message=(
                        "quantitative prose must use a generated manuscript token "
                        "or a documented non-evidence syntax exclusion"
                    ),
                    value=literal.group(0),
                )
            )
    return findings


def audit_manuscript_presentation(
    manuscript_dir: Path,
    variables_file: Path,
    *,
    resolved_manuscript_dir: Path | None = None,
) -> PresentationReport:
    """Return a stable, machine-readable presentation audit report."""
    documents = _documents(manuscript_dir, exclude_source_support=True)
    variable_names = _load_variable_names(variables_file)
    findings = [
        *_audit_acronym_first_use(documents),
        *_audit_abstract_citations(documents),
        *_audit_abstract_reader_focus(documents),
        *_audit_duplicate_cross_references(documents),
        *_audit_source_tokens(documents, variable_names),
        *_audit_numeric_literals(documents),
    ]
    resolved_files: list[str] = []
    if resolved_manuscript_dir is not None:
        resolved_findings, resolved_files = _audit_resolved_tokens(
            resolved_manuscript_dir
        )
        findings.extend(resolved_findings)
    findings.sort(
        key=lambda finding: (
            finding.path,
            finding.line,
            finding.column,
            finding.code,
            finding.value,
        )
    )
    by_code: dict[str, int] = {}
    for finding in findings:
        by_code[finding.code] = by_code.get(finding.code, 0) + 1
    report: PresentationReport = {
        "schema_version": 1,
        "files_checked": [document.relative_path for document in documents],
        "resolved_files_checked": resolved_files,
        "acronym_registry": [definition.acronym for definition in ACRONYM_REGISTRY],
        "summary": {"finding_count": len(findings), "by_code": by_code},
        "findings": [
            {
                "code": finding.code,
                "path": finding.path,
                "line": finding.line,
                "column": finding.column,
                "message": finding.message,
                "value": finding.value,
            }
            for finding in findings
        ],
    }
    return report


def _write_report(path: Path, report: PresentationReport) -> None:
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _input_error_report(message: str) -> dict[str, object]:
    return {"schema_version": 1, "status": "input_error", "error": message}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit source-manuscript presentation contracts and emit JSON."
    )
    parser.add_argument(
        "--manuscript-dir",
        type=Path,
        default=_PROJECT_ROOT / "manuscript",
        help="Source manuscript directory (default: project manuscript/).",
    )
    parser.add_argument(
        "--variables-file",
        type=Path,
        default=_PROJECT_ROOT / "output" / "data" / "manuscript_variables.json",
        help="JSON map of generated source tokens (default: output/data/manuscript_variables.json).",
    )
    parser.add_argument(
        "--resolved-manuscript-dir",
        type=Path,
        default=None,
        help="Optional hydrated manuscript directory; every remaining token is an error.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path; stdout always receives the same report.",
    )
    args = parser.parse_args()
    try:
        report = audit_manuscript_presentation(
            args.manuscript_dir,
            args.variables_file,
            resolved_manuscript_dir=args.resolved_manuscript_dir,
        )
    except (OSError, ValueError) as exc:
        error_report = _input_error_report(str(exc))
        print(json.dumps(error_report, indent=2, sort_keys=True))
        return 2
    if args.report is not None:
        _write_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if int(report["summary"]["finding_count"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
