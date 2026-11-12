#!/usr/bin/env python3
"""Validate the shipped GNN specifications against their runtime models.

Reads ``models/*.gnn.md`` and prints a deterministic JSON report: every
declared matrix must match the runtime model built by the real contract
(``build_generative_model`` / ``make_reference_action_scenario``).

The pin, the ``.github/GNN.md`` pointer and the specifications' upstream syntax
are ``gnn.importers``' check, which the ``gnn`` arm runs as
``scripts/gnn_importer_check.py``. The upstream-parser digest comparison lives
in ``tests/test_gnn_specs.py`` because it needs the pinned checkout's
interpreter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from active_inference_power.gnn_specs import check_specs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="project root holding models/")
    args = parser.parse_args()
    report = check_specs(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
