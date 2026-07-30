#!/usr/bin/env python3
"""Validate local Markdown links in the project documentation tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
_LINK = re.compile(r"(?<!!)\[[^\]]+\]\((?:<)?([^\s)>]+)(?:[^)]*)\)")
_IGNORED_PREFIXES = ("#", "//")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_GENERATED_DESTINATION = re.compile(r"^\{\{[A-Z][A-Z0-9_]*\}\}$")
_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "output",
    }
)


def _markdown_documents(root: Path) -> list[Path]:
    """Collect project Markdown while pruning generated and cache subtrees."""
    if not root.is_dir():
        raise ValueError(f"documentation root does not exist: {root}")
    documents: list[Path] = []
    # ``Path.rglob`` filters only after it descends.  Pruning ``output`` and
    # caches here keeps this source-only audit bounded even after many retained
    # release bundles have accumulated under the project root.
    for directory, child_directories, filenames in os.walk(root, topdown=True):
        child_directories[:] = sorted(
            name for name in child_directories if name not in _EXCLUDED_DIRECTORIES
        )
        base = Path(directory)
        documents.extend(
            base / filename
            for filename in sorted(filenames)
            if filename.endswith(".md") and (base / filename).is_file()
        )
    if not documents:
        raise ValueError(f"no Markdown documentation found under: {root}")
    return sorted(documents, key=lambda path: path.relative_to(root).as_posix())


def _is_nonlocal_destination(destination: str) -> bool:
    """Identify destinations deferred to a renderer or external resolver.

    Manuscript source may use a complete uppercase ``{{TOKEN}}`` as a link
    destination when strict hydration replaces it with a configuration-bound
    DOI, record, or release URL. The source-only navigation audit cannot
    resolve that future URL; the resolved-manuscript presentation audit is the
    fail-closed consumer that rejects an unexpanded token.
    """
    return (
        destination.startswith(_IGNORED_PREFIXES)
        or _URI_SCHEME.match(destination) is not None
        or _GENERATED_DESTINATION.fullmatch(destination) is not None
    )


def audit_documentation_links(root: Path = ROOT) -> dict[str, Any]:
    """Return a stable report of local Markdown destinations that do not exist."""
    root = Path(root).resolve()
    documents = _markdown_documents(root)
    findings: list[dict[str, str]] = []
    checked = 0
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for match in _LINK.finditer(text):
            destination = unquote(match.group(1))
            if _is_nonlocal_destination(destination):
                continue
            local_path = destination.split("#", 1)[0]
            if not local_path:
                continue
            checked += 1
            target = (document.parent / local_path).resolve()
            if not target.exists():
                findings.append(
                    {
                        "document": str(document.relative_to(root)),
                        "destination": destination,
                        "message": "local Markdown destination does not exist",
                    }
                )
    return {
        "schema_version": "1.0",
        "passed": not findings,
        "document_count": len(documents),
        "local_link_count": checked,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = audit_documentation_links(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
