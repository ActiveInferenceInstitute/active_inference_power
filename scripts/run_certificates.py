#!/usr/bin/env python3
"""Thin orchestrator: run the reproducible statistical certificates and emit JSON.

Business logic lives in ``active_inference_power/certificates.py``; this script only handles
argument parsing, invocation, and writing results to ``output/`` for the
manuscript's results section. Run with the project venv:

    ./.venv/bin/python scripts/run_certificates.py --profile fast
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from active_inference_power.certificates import run_all_certificates  # noqa: E402
from scripts.artifacts import atomic_write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("fast", "release"),
        default="release",
        help="Certificate replication profile (default: release).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "output" / "certificates.json",
        help="Where to write the certificate results JSON.",
    )
    parser.add_argument("--m", type=int, default=None, help="Hypotheses per certificate.")
    parser.add_argument(
        "--n-reps", type=int, default=None, help="Replications per certificate."
    )
    parser.add_argument("--seed", type=int, default=0, help="Shared certificate seed.")
    args = parser.parse_args()

    results = run_all_certificates(
        fast=args.profile == "fast", m=args.m, n_reps=args.n_reps, seed=args.seed
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, results)

    all_pass = all(r.get("passes", False) for r in results.values())
    for name, r in results.items():
        status = "PASS" if r.get("passes") else "FAIL"
        print(f"[{status}] {name}: {json.dumps(r, default=float)}")
    print(f"\nWrote {args.out}")
    print(f"ALL CERTIFICATES {'PASS' if all_pass else 'FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
