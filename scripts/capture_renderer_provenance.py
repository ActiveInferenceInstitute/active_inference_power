#!/usr/bin/env python3
"""Record the sibling renderer and toolchain used for an existing PDF/HTML render."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.renderer_provenance import write_renderer_provenance  # noqa: E402


def _resolve_output_root(output_root: Path | None) -> Path:
    """Resolve the command destination; the writer enforces safe-root policy."""
    return ROOT / "output" if output_root is None else Path(output_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--template-root",
        type=Path,
        default=None,
        help="Optional sibling template checkout; defaults to ../template.",
    )
    args = parser.parse_args()
    result = write_renderer_provenance(
        project_root=ROOT,
        output_root=_resolve_output_root(args.output_root),
        template_root=args.template_root,
    )
    print(
        json.dumps(
            {
                "path": str(result["path"]),
                "template_git_head": result["receipt"]["template"]["git"]["head"],
                "template_dirty": result["receipt"]["template"]["git"]["dirty"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
