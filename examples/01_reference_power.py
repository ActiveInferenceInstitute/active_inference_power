#!/usr/bin/env python3
"""Run the small fixed-horizon/reference Active Inference power suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from active_inference_power.suite import reference_suite  # noqa: E402
from examples._common import atomic_json, checkpoint, output_path, resume_value  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("fast", "release"), default="fast")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--write-completed-checkpoint",
        action="store_true",
        help="Save a replayable snapshot only after the full example run completes.",
    )
    parser.add_argument(
        "--replay-completed-checkpoint",
        action="store_true",
        help="Write results from a previously completed example snapshot; this does not resume partial work.",
    )
    args = parser.parse_args()
    n_reps = 8 if args.profile == "fast" else 1000
    destination = output_path("01_reference_power", args.output_root)
    checkpoint_path = destination / "checkpoint.json"
    if args.replay_completed_checkpoint:
        payload = resume_value(checkpoint_path, n_reps)
        if payload is None:
            raise FileNotFoundError(
                "--replay-completed-checkpoint requested but no completed snapshot exists: "
                f"{checkpoint_path}"
            )
        atomic_json(destination / "results.json", payload)
        print(destination / "results.json")
        return 0
    result = reference_suite(n_reps=n_reps, seed=args.seed, workers=args.workers)
    if args.write_completed_checkpoint:
        checkpoint(checkpoint_path, result.to_mapping(), n_reps, n_reps)
    atomic_json(destination / "results.json", result.to_mapping())
    print(destination / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
