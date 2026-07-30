#!/usr/bin/env python3
"""Read-only accessibility audit for the generated figure registry and PNGs.

The audit is deliberately small and deterministic.  It checks the actual PNG
dimensions and decodeability, then verifies the registry carries the reader
context that cannot reliably be inferred from pixels: estimand, uncertainty,
sample size, evidence class, validity boundary, redundant visual encoding, a
reader question, a visible comparison, a takeaway, and a visual-encoding
explanation, plus the evaluator/agent/investigator perspective that makes the
figure's role explicit. It never regenerates figures or rewrites a manifest.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import unquote, urlparse

from matplotlib.image import imread


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from active_inference_power.schema import validate_figure_registry  # noqa: E402


_EVIDENCE_CLASSES = {
    "formal_conditional",
    "analytic_reference",
    "conditional_operating_characteristic",
    "calibration_diagnostic",
    "trace_diagnostic",
    "stress_test",
    "provenance_contract",
}
_REDUNDANCY_TERMS = ("marker", "line", "label", "hatch", "shape", "panel", "annotation")
_REQUIRED_METADATA = (
    "estimand",
    "uncertainty",
    "sample_size",
    "evidence_class",
    "validity_boundary",
    "redundant_encoding",
)
_READER_SEMANTICS_2_1 = (
    "reader_question",
    "takeaway",
    "visual_encoding",
)
_READER_SEMANTICS_2_2 = (*_READER_SEMANTICS_2_1, "visible_comparison")
_READER_SEMANTICS_2_3 = (*_READER_SEMANTICS_2_2, "evaluation_perspective")
_PANEL_VALIDITY_REQUIREMENTS: dict[str, dict[str, tuple[str, ...]]] = {
    "sequential_evidence.png": {
        "Fixed-horizon p-value path": ("invalid if used for optional stopping",),
    },
    "online_fdr_trace.png": {
        "Nominal test level α_t": ("reciprocal", "not itself e-process"),
        "Procedure-specific accounting": (
            "not alpha-wealth",
            "not a remaining alpha",
        ),
    },
    "action_loop_trace.png": {
        "Target-specific e-process paths": ("per-target", "no fwer or fdr"),
        "All-action global-null likelihood-ratio product": (
            "diagnostic",
            "no family-error-control",
        ),
        "Fixed-horizon p-value contrast": ("invalid if used for optional stopping",),
    },
    "conditional_power_surface.png": {
        "Observed mean FDP": ("not an fdr-control claim",),
        "Fixed-horizon policy rows": ("invalid if used for optional stopping",),
    },
    "action_policy_frontier.png": {
        "Scenario-faceted power-cost frontier": ("cost scale", "finite-simulation"),
        "Fixed-horizon policy rows": ("invalid if used for optional stopping",),
    },
}
_QUANTITATIVE_EVIDENCE_CLASSES = {
    "formal_conditional",
    "analytic_reference",
    "conditional_operating_characteristic",
    "calibration_diagnostic",
    "trace_diagnostic",
    "stress_test",
}
_ALT_VISUAL_TERMS = (
    "axis",
    "axes",
    "band",
    "bar",
    "box",
    "chart",
    "curve",
    "diagram",
    "error bar",
    "grid",
    "legend",
    "line",
    "marker",
    "panel",
    "plot",
    "point",
    "raster",
    "ribbon",
    "row",
    "scatter",
    "table",
)


def _finding(filename: str, rule: str, message: str) -> dict[str, str]:
    return {"filename": filename, "rule": rule, "message": message}


def _normalize_reader_text(value: str) -> str:
    """Normalize prose only for duplicate-text comparisons."""
    return " ".join(value.casefold().split())


class _RenderedImageParser(HTMLParser):
    """Collect ``src`` and ``alt`` values from real rendered HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.images: list[tuple[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        values = {name.lower(): value for name, value in attrs}
        source = values.get("src")
        if isinstance(source, str) and source:
            self.images.append((source, values.get("alt")))


def _resolve_output_root(output_root: Path | None) -> Path:
    output = ROOT / "output" if output_root is None else Path(output_root)
    if not output.is_absolute():
        output = ROOT / output
    return output.resolve()


def _resolve_rendered_html_path(output: Path, rendered_html: Path | None) -> Path:
    html_path = output / "web" / "index.html" if rendered_html is None else Path(rendered_html)
    # Command-line paths are project-relative, matching every other project
    # input. This lets --output-root output --rendered-html output/web/... work
    # without accidentally prefixing output twice.
    if not html_path.is_absolute():
        html_path = ROOT / html_path
    return html_path.resolve()


def _auxiliary_rendered_html_paths(output: Path, primary_html: Path) -> tuple[Path, ...]:
    """Return safely resolved rendered per-section pages beside ``index.html``."""
    web_dir = (output / "web").resolve()
    if not web_dir.is_dir():
        return ()
    auxiliary_paths: list[Path] = []
    for candidate in sorted(web_dir.glob("manuscript__*.html")):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(web_dir)
        except ValueError as exc:
            raise ValueError(f"auxiliary rendered HTML escapes web directory: {candidate}") from exc
        if resolved != primary_html:
            auxiliary_paths.append(resolved)
    return tuple(auxiliary_paths)


def _rendered_alternatives(html_path: Path) -> dict[str, list[str | None]]:
    """Collect registered-image candidate alternatives from one rendered page."""
    parser = _RenderedImageParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    parser.close()
    rendered_by_filename: dict[str, list[str | None]] = {}
    for source, alt in parser.images:
        filename = Path(unquote(urlparse(source).path)).name
        if filename:
            rendered_by_filename.setdefault(filename, []).append(alt)
    return rendered_by_filename


def audit_figure_accessibility(
    output_root: Path | None = None,
    *,
    min_width: int = 640,
    min_height: int = 480,
) -> dict[str, Any]:
    """Inspect a figure bundle without changing it and return a JSON report."""
    if min_width < 1 or min_height < 1:
        raise ValueError("minimum figure dimensions must be positive")
    output = _resolve_output_root(output_root)
    registry_path = output / "figures" / "figure_registry.json"
    findings: list[dict[str, str]] = []
    if not registry_path.is_file():
        return {
            "schema_version": "1.0",
            "passed": False,
            "figure_count": 0,
            "minimum_dimensions": {"width": min_width, "height": min_height},
            "findings": [_finding("figure_registry.json", "registry", "figure registry is missing")],
        }
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        validate_figure_registry(registry)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "schema_version": "1.0",
            "passed": False,
            "figure_count": 0,
            "minimum_dimensions": {"width": min_width, "height": min_height},
            "findings": [_finding("figure_registry.json", "registry", str(exc))],
        }

    records = registry["figures"]
    for record in records:
        filename = str(record["filename"])
        metadata = record["metadata"]
        figure_path = output / "figures" / filename
        try:
            pixels = imread(figure_path)
        except (OSError, ValueError, SyntaxError) as exc:
            findings.append(_finding(filename, "decode", f"PNG cannot be opened: {exc}"))
            continue
        if pixels.ndim not in (2, 3):
            findings.append(_finding(filename, "decode", "PNG must decode to a grayscale or color image"))
            continue
        height, width = int(pixels.shape[0]), int(pixels.shape[1])
        if width < min_width or height < min_height:
            findings.append(
                _finding(
                    filename,
                    "minimum_dimensions",
                    f"PNG is {width}x{height}px; minimum is {min_width}x{min_height}px",
                )
            )
        for key in _REQUIRED_METADATA:
            value = metadata.get(key)
            if not isinstance(value, str) or not value.strip():
                findings.append(_finding(filename, key, f"missing reader-facing {key.replace('_', ' ')}"))
        reader_semantics = {
            "2.1": _READER_SEMANTICS_2_1,
            "2.2": _READER_SEMANTICS_2_2,
            "2.3": _READER_SEMANTICS_2_3,
            "2.4": _READER_SEMANTICS_2_3,
        }.get(str(registry.get("schema_version")), ())
        for key in reader_semantics:
            value = record.get(key)
            if not isinstance(value, str) or not value.strip():
                findings.append(
                    _finding(
                        filename,
                        key,
                        f"missing reader-facing {key.replace('_', ' ')}",
                    )
                )
        panel_requirements = _PANEL_VALIDITY_REQUIREMENTS.get(filename)
        if panel_requirements is not None:
            panel_validity = record.get("panel_validity")
            if not isinstance(panel_validity, list):
                findings.append(
                    _finding(
                        filename,
                        "panel_validity",
                        "mixed-semantics figure requires panel-level validity metadata",
                    )
                )
            else:
                by_panel = {
                    str(item.get("panel")): str(item.get("validity"))
                    for item in panel_validity
                    if isinstance(item, dict)
                }
                for panel, required_phrases in panel_requirements.items():
                    validity = by_panel.get(panel, "")
                    if not validity:
                        findings.append(
                            _finding(
                                filename,
                                "panel_validity",
                                f"required panel validity is missing for {panel!r}",
                            )
                        )
                        continue
                    normalized = validity.lower()
                    missing = [
                        phrase
                        for phrase in required_phrases
                        if phrase not in normalized
                    ]
                    if missing:
                        findings.append(
                            _finding(
                                filename,
                                "panel_validity",
                                f"{panel!r} is missing required validity wording: {', '.join(missing)}",
                            )
                        )
        evidence_class = metadata.get("evidence_class")
        if isinstance(evidence_class, str) and evidence_class not in _EVIDENCE_CLASSES:
            findings.append(
                _finding(filename, "evidence_class", f"unsupported evidence class: {evidence_class}")
            )
        redundancy = str(metadata.get("redundant_encoding", "")).lower()
        if not any(term in redundancy for term in _REDUNDANCY_TERMS):
            findings.append(
                _finding(
                    filename,
                    "redundant_encoding",
                    "state how color is paired with a marker, line, label, shape, panel, hatch, or annotation",
                )
            )
        if str(registry.get("schema_version")) == "2.4":
            caption = str(record["caption"])
            alternative = str(record["alt"])
            if _normalize_reader_text(caption) == _normalize_reader_text(alternative):
                findings.append(
                    _finding(
                        filename,
                        "alt_equals_caption",
                        "authored alternative text must visually describe the figure rather than repeat the caption",
                    )
                )
            if (
                evidence_class in _QUANTITATIVE_EVIDENCE_CLASSES
                and not any(term in _normalize_reader_text(alternative) for term in _ALT_VISUAL_TERMS)
            ):
                findings.append(
                    _finding(
                        filename,
                        "alt_visual_content",
                        "quantitative figure alternative text must name a visible axis, mark, panel, or other visual encoding",
                    )
                )
    return {
        "schema_version": "1.0",
        "passed": not findings,
        "figure_count": len(records),
        "minimum_dimensions": {"width": min_width, "height": min_height},
        "findings": findings,
    }


def audit_rendered_figure_alts(
    output_root: Path | None = None,
    *,
    rendered_html: Path | None = None,
    include_auxiliary_pages: bool = False,
) -> dict[str, Any]:
    """Verify reader-facing HTML uses each registry's authored alternative text.

    Captions describe the scientific interpretation, whereas alternative text
    describes the visible figure.  This check binds the source-owned registry
    to the renderer's actual ``<img alt>`` values without regenerating either
    artifact.
    """
    output = _resolve_output_root(output_root)
    registry_path = output / "figures" / "figure_registry.json"
    html_path = _resolve_rendered_html_path(output, rendered_html)
    findings: list[dict[str, str]] = []
    if not registry_path.is_file():
        return {
            "schema_version": "1.0",
            "passed": False,
            "figure_count": 0,
            "rendered_html": str(html_path),
            "findings": [_finding("figure_registry.json", "registry", "figure registry is missing")],
        }
    if not html_path.is_file():
        return {
            "schema_version": "1.0",
            "passed": False,
            "figure_count": 0,
            "rendered_html": str(html_path),
            "findings": [_finding("index.html", "rendered_html", "rendered HTML is missing")],
        }
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        validate_figure_registry(registry)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "schema_version": "1.0",
            "passed": False,
            "figure_count": 0,
            "rendered_html": str(html_path),
            "findings": [_finding("figure_registry.json", "registry", str(exc))],
        }
    try:
        rendered_by_filename = _rendered_alternatives(html_path)
    except OSError as exc:
        return {
            "schema_version": "1.0",
            "passed": False,
            "figure_count": 0,
            "rendered_html": str(html_path),
            "findings": [_finding("index.html", "rendered_html", str(exc))],
        }
    records = registry["figures"]
    for record in records:
        filename = str(record["filename"])
        expected_alt = str(record["alt"])
        observed = rendered_by_filename.get(filename, [])
        if not observed:
            findings.append(
                _finding(filename, "rendered_alt_missing", "rendered HTML does not include this registered figure")
            )
            continue
        if any(value != expected_alt for value in observed):
            findings.append(
                _finding(
                    filename,
                    "rendered_alt_mismatch",
                    "rendered HTML alternative text differs from the authored figure-registry value",
                )
            )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "passed": not findings,
        "figure_count": len(records),
        "rendered_html": str(html_path),
        "findings": findings,
    }
    if not include_auxiliary_pages:
        return report

    auxiliary_pages: list[dict[str, Any]] = []
    registered_filenames = {str(record["filename"]) for record in records}
    for auxiliary_html in _auxiliary_rendered_html_paths(output, html_path):
        page_findings: list[dict[str, str]] = []
        matched_image_count = 0
        try:
            auxiliary_by_filename = _rendered_alternatives(auxiliary_html)
        except OSError as exc:
            page_findings.append(
                _finding(
                    auxiliary_html.name,
                    "rendered_html",
                    f"cannot read auxiliary rendered HTML: {exc}",
                )
            )
        else:
            matched_image_count = sum(
                len(values)
                for filename, values in auxiliary_by_filename.items()
                if filename in registered_filenames
            )
            for record in records:
                filename = str(record["filename"])
                expected_alt = str(record["alt"])
                observed = auxiliary_by_filename.get(filename, [])
                if observed and any(value != expected_alt for value in observed):
                    page_findings.append(
                        _finding(
                            f"{auxiliary_html.name}:{filename}",
                            "rendered_alt_mismatch",
                            "rendered HTML alternative text differs from the authored "
                            "figure-registry value",
                        )
                    )
        findings.extend(page_findings)
        auxiliary_pages.append(
            {
                "rendered_html": str(auxiliary_html),
                "matched_image_count": matched_image_count,
                "findings": page_findings,
            }
        )
    report.update(
        {
            "include_auxiliary_pages": True,
            "auxiliary_page_count": len(auxiliary_pages),
            "auxiliary_pages": auxiliary_pages,
            "passed": not findings,
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--min-width", type=int, default=640)
    parser.add_argument("--min-height", type=int, default=480)
    parser.add_argument(
        "--rendered-html",
        type=Path,
        default=None,
        help="also verify authored registry alt text in a rendered HTML file",
    )
    parser.add_argument(
        "--include-auxiliary-pages",
        action="store_true",
        help="also verify rendered web/manuscript__*.html section pages",
    )
    args = parser.parse_args()
    report = audit_figure_accessibility(
        args.output_root, min_width=args.min_width, min_height=args.min_height
    )
    if args.rendered_html is None:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    rendered = audit_rendered_figure_alts(
        args.output_root,
        rendered_html=args.rendered_html,
        include_auxiliary_pages=args.include_auxiliary_pages,
    )
    combined = {
        "schema_version": "1.0",
        "passed": bool(report["passed"] and rendered["passed"]),
        "figure_bundle": report,
        "rendered_alt_text": rendered,
    }
    print(json.dumps(combined, indent=2, sort_keys=True))
    return 0 if combined["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
