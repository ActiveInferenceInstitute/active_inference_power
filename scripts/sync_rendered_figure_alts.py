#!/usr/bin/env python3
"""Synchronize rendered HTML figure alternative text from the source registry.

The sibling renderer owns HTML generation, but Pandoc can use an image's
Markdown description (the reader-facing caption) for ``img`` alternative text.
This post-render step restores the distinct, authored visual descriptions from
``figures/figure_registry.json``.  It is deliberately narrow: only ``img``
elements whose ``src`` basename is registered are changed; all other HTML and
all unregistered images are left byte-for-byte intact.

The command does not update a manifest.  Run it immediately after rendering
and before the read-only rendered-alt audit and release finalization, which
bind the resulting HTML to the publication bundle.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape, unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import stat
import sys
from typing import Any
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from active_inference_power.schema import validate_figure_registry  # noqa: E402
from scripts.artifacts import (  # noqa: E402
    atomic_write_text,
    require_safe_output_root,
    sha256_file,
)


@dataclass(frozen=True)
class _Attribute:
    """Offsets for one attribute inside an original ``img`` start tag."""

    name: str
    start: int
    end: int
    value_start: int | None
    value_end: int | None


class _RenderedImageTagParser(HTMLParser):
    """Locate real ``img`` start tags while retaining their original bytes."""

    def __init__(self, rendered_html: str) -> None:
        super().__init__(convert_charrefs=False)
        self._line_offsets = [0]
        self._line_offsets.extend(
            index + 1 for index, character in enumerate(rendered_html) if character == "\n"
        )
        self.spans: list[tuple[int, int]] = []

    def _record_image(self, tag: str) -> None:
        if tag.lower() != "img":
            return
        raw_tag = self.get_starttag_text()
        if raw_tag is None:
            return
        line, column = self.getpos()
        start = self._line_offsets[line - 1] + column
        self.spans.append((start, start + len(raw_tag)))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        self._record_image(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        self._record_image(tag)


def _resolve_output_root(output_root: Path | None) -> Path:
    output = ROOT / "output" if output_root is None else Path(output_root)
    if not output.is_absolute():
        output = ROOT / output
    return require_safe_output_root(output, project_root=ROOT)


def _resolve_html_path(output: Path, rendered_html: Path | None) -> Path:
    html_path = output / "web" / "index.html" if rendered_html is None else Path(rendered_html)
    if not html_path.is_absolute():
        html_path = ROOT / html_path
    return html_path.resolve()


def _auxiliary_rendered_html_paths(output: Path, primary_html: Path) -> tuple[Path, ...]:
    """Return safe, rendered per-section pages beside the primary HTML page.

    The template's combined ``index.html`` is the established default.  When a
    bundle also retains ``manuscript__*.html`` pages, callers may explicitly
    synchronize those auxiliary reader pages as well.  The helper keeps that
    expansion inside the rendered ``web`` directory and rejects a symlink that
    would escape it.
    """
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


def _parse_attributes(tag: str) -> tuple[_Attribute, ...]:
    """Parse enough of one image tag to locate ``src`` and ``alt`` safely.

    This is intentionally a position-preserving parser rather than an HTML
    serializer: serializing a rendered page would alter unrelated markup.
    """
    index = 1
    size = len(tag)
    while index < size and tag[index].isspace():
        index += 1
    while index < size and not tag[index].isspace() and tag[index] not in "/>":
        index += 1

    attributes: list[_Attribute] = []
    while index < size:
        while index < size and tag[index].isspace():
            index += 1
        if index >= size or tag[index] in "/>":
            break
        start = index
        while index < size and not tag[index].isspace() and tag[index] not in "=/>":
            index += 1
        if start == index:
            break
        name = tag[start:index]
        while index < size and tag[index].isspace():
            index += 1
        value_start: int | None = None
        value_end: int | None = None
        if index < size and tag[index] == "=":
            index += 1
            while index < size and tag[index].isspace():
                index += 1
            if index < size and tag[index] in "\"'":
                quote = tag[index]
                index += 1
                value_start = index
                while index < size and tag[index] != quote:
                    index += 1
                value_end = index
                if index < size:
                    index += 1
            else:
                value_start = index
                while index < size and not tag[index].isspace() and tag[index] != ">":
                    index += 1
                value_end = index
        attributes.append(
            _Attribute(
                name=name.lower(),
                start=start,
                end=index,
                value_start=value_start,
                value_end=value_end,
            )
        )
    return tuple(attributes)


def _image_tag_spans(rendered_html: str) -> tuple[tuple[int, int], ...]:
    parser = _RenderedImageTagParser(rendered_html)
    parser.feed(rendered_html)
    parser.close()
    return tuple(parser.spans)


def _attribute_value(tag: str, attribute: _Attribute) -> str | None:
    if attribute.value_start is None or attribute.value_end is None:
        return None
    return unescape(tag[attribute.value_start : attribute.value_end])


def _registered_filename(
    tag: str, attributes: tuple[_Attribute, ...], authored_alts: dict[str, str]
) -> str | None:
    sources = [attribute for attribute in attributes if attribute.name == "src"]
    if len(sources) != 1:
        return None
    source = _attribute_value(tag, sources[0])
    if not source:
        return None
    filename = Path(unquote(urlparse(source).path)).name
    return filename if filename in authored_alts else None


def _insert_alt_attribute(tag: str, escaped_alt: str) -> str:
    """Add one ``alt`` attribute without changing the surrounding tag layout."""
    end = len(tag) - 1
    cursor = end - 1
    while cursor >= 0 and tag[cursor].isspace():
        cursor -= 1
    insertion = cursor if cursor >= 0 and tag[cursor] == "/" else end
    while insertion > 0 and tag[insertion - 1].isspace():
        insertion -= 1
    return tag[:insertion] + f' alt="{escaped_alt}"' + tag[insertion:]


def _has_quoted_value(tag: str, attribute: _Attribute) -> bool:
    return (
        attribute.value_start is not None
        and attribute.value_start > 0
        and tag[attribute.value_start - 1] in "\"'"
    )


def _synchronized_tag(tag: str, authored_alts: dict[str, str]) -> tuple[str, bool]:
    attributes = _parse_attributes(tag)
    filename = _registered_filename(tag, attributes, authored_alts)
    if filename is None:
        return tag, False
    expected_alt = authored_alts[filename]
    escaped_alt = escape(expected_alt, quote=True)
    alt_attributes = [attribute for attribute in attributes if attribute.name == "alt"]
    if not alt_attributes:
        return _insert_alt_attribute(tag, escaped_alt), True
    if len(alt_attributes) != 1:
        raise ValueError(f"registered image has multiple alt attributes: {filename}")
    attribute = alt_attributes[0]
    observed_alt = _attribute_value(tag, attribute)
    if observed_alt == expected_alt:
        return tag, False
    if (
        attribute.value_start is None
        or attribute.value_end is None
        or not _has_quoted_value(tag, attribute)
    ):
        return tag[: attribute.start] + f'alt="{escaped_alt}"' + tag[attribute.end :], True
    return (
        tag[: attribute.value_start] + escaped_alt + tag[attribute.value_end :],
        True,
    )


def _authored_alternatives(registry_path: Path) -> dict[str, str]:
    if not registry_path.is_file():
        raise FileNotFoundError(f"figure registry is missing: {registry_path}")
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"figure registry is not valid JSON: {registry_path}") from exc
    if not isinstance(registry, dict):
        raise ValueError("figure registry must be a JSON object")
    validate_figure_registry(registry)
    return {
        str(record["filename"]): str(record["alt"])
        for record in registry["figures"]
    }


def _synchronize_rendered_html(
    html_path: Path, authored_alts: dict[str, str]
) -> dict[str, Any]:
    """Synchronize one HTML file while preserving every non-image byte."""
    if not html_path.is_file():
        raise FileNotFoundError(f"rendered HTML is missing: {html_path}")
    original = html_path.read_text(encoding="utf-8")
    html_before_sha256 = sha256_file(html_path)
    matched_image_count = 0
    updated_image_count = 0
    parts: list[str] = []
    cursor = 0
    for start, end in _image_tag_spans(original):
        tag = original[start:end]
        synchronized, changed = _synchronized_tag(tag, authored_alts)
        if _registered_filename(tag, _parse_attributes(tag), authored_alts) is not None:
            matched_image_count += 1
        if changed:
            updated_image_count += 1
        parts.extend((original[cursor:start], synchronized))
        cursor = end
    parts.append(original[cursor:])
    synchronized = "".join(parts)
    changed = synchronized != original
    if changed:
        original_mode = stat.S_IMODE(html_path.stat().st_mode)
        atomic_write_text(html_path, synchronized)
        html_path.chmod(original_mode)
    html_after_sha256 = sha256_file(html_path)
    return {
        "rendered_html": str(html_path),
        "matched_image_count": matched_image_count,
        "updated_image_count": updated_image_count,
        "changed": changed,
        "html_before_sha256": html_before_sha256,
        "html_after_sha256": html_after_sha256,
    }


def sync_rendered_figure_alts(
    output_root: Path | None = None,
    *,
    rendered_html: Path | None = None,
    include_auxiliary_pages: bool = False,
) -> dict[str, Any]:
    """Replace registered rendered-image alternative text with authored values.

    The result is a deterministic stdout-friendly receipt.  It contains this
    synchronizer's source digest but intentionally does not modify a manifest
    or the computational execution witness; the post-render review binds the
    resulting HTML separately.
    """
    output = _resolve_output_root(output_root)
    html_path = _resolve_html_path(output, rendered_html)
    authored_alts = _authored_alternatives(output / "figures" / "figure_registry.json")
    primary_receipt = _synchronize_rendered_html(html_path, authored_alts)
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "output_root": str(output),
        "synchronizer_sha256": sha256_file(Path(__file__).resolve()),
        "registered_figure_count": len(authored_alts),
        **primary_receipt,
    }
    if not include_auxiliary_pages:
        return receipt

    auxiliary_pages = [
        _synchronize_rendered_html(path, authored_alts)
        for path in _auxiliary_rendered_html_paths(output, html_path)
    ]
    receipt.update(
        {
            "include_auxiliary_pages": True,
            "auxiliary_page_count": len(auxiliary_pages),
            "auxiliary_pages": auxiliary_pages,
            "matched_image_count": int(receipt["matched_image_count"])
            + sum(int(page["matched_image_count"]) for page in auxiliary_pages),
            "updated_image_count": int(receipt["updated_image_count"])
            + sum(int(page["updated_image_count"]) for page in auxiliary_pages),
            "changed": bool(receipt["changed"])
            or any(bool(page["changed"]) for page in auxiliary_pages),
        }
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--rendered-html",
        type=Path,
        default=None,
        help="rendered HTML to synchronize (default: <output-root>/web/index.html)",
    )
    parser.add_argument(
        "--include-auxiliary-pages",
        action="store_true",
        help="also synchronize rendered web/manuscript__*.html section pages",
    )
    args = parser.parse_args()
    report = sync_rendered_figure_alts(
        args.output_root,
        rendered_html=args.rendered_html,
        include_auxiliary_pages=args.include_auxiliary_pages,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
