#!/usr/bin/env python3
"""Run one offline chunk of the imported Dobriban/BH stress protocol.

This command never downloads data. It runs the finite NumPy/SciPy protocol
adapter only; the external python-flint theorem certificate is not invoked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_inference_power.protocols import load_dobriban_bh_protocol, run_dobriban_bh_chunk  # noqa: E402
from scripts.artifacts import atomic_write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", default="fdr_N50", help="publicly documented chunk prefix")
    parser.add_argument("--N", type=int, default=None, help="override N for a controlled smoke run")
    parser.add_argument("--strata", type=int, default=None)
    parser.add_argument("--macro", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target-entries", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output path")
    args = parser.parse_args()
    protocol = load_dobriban_bh_protocol()
    chunks = {str(item["prefix"]): item for item in protocol.plan["chunks"]}
    if args.chunk not in chunks:
        raise SystemExit(f"unknown publicly documented chunk: {args.chunk}")
    chunk = chunks[args.chunk]
    result = run_dobriban_bh_chunk(
        int(chunk["N"] if args.N is None else args.N),
        strata=int(protocol.plan["strata"] if args.strata is None else args.strata),
        macro_replications=int(chunk["macro_replications"] if args.macro is None else args.macro),
        master_seed=int(chunk["master_seed"] if args.seed is None else args.seed),
        target_entries=args.target_entries,
        protocol=protocol,
    ).to_mapping()
    if args.output is not None:
        atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
