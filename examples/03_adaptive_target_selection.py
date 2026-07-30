#!/usr/bin/env python3
"""Run adaptive target-selection diagnostics and split-stream confirmation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from active_inference_power.scenario import make_reference_action_scenario  # noqa: E402
from active_inference_power.suite import evaluate_adaptive_target_selection, evaluate_split_stream_confirmation  # noqa: E402
from examples._common import atomic_json, checkpoint, output_path, resume_value  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("fast", "release"), default="fast")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=23)
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
    scenario = make_reference_action_scenario(horizon=12)
    destination = output_path("03_adaptive_target_selection", args.output_root)
    checkpoint_path = destination / "checkpoint.json"
    if args.replay_completed_checkpoint:
        resumed = resume_value(checkpoint_path, n_reps)
        if resumed is None:
            raise FileNotFoundError(
                "--replay-completed-checkpoint requested but no completed snapshot exists: "
                f"{checkpoint_path}"
            )
        atomic_json(destination / "results.json", resumed)
        print(destination / "results.json")
        return 0
    payload = {
        "suite": "adaptive_target_selection",
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "methods": {
            method: evaluate_adaptive_target_selection(
                scenario, "adaptive_target", n_reps, seed=args.seed + index, method=method,
                workers=args.workers,
            )
            for index, method in enumerate(("lord_plus_plus", "saffron"))
        },
        "split_stream_confirmation": evaluate_split_stream_confirmation(
            scenario, n_reps, seed=args.seed + 10, workers=args.workers
        ),
        "claim_boundary": "Online-FDR summaries are conditional diagnostics requiring predictable target selection and conditional super-uniform evidence; environment shifts are not universal guarantees.",
    }
    if args.write_completed_checkpoint:
        checkpoint(checkpoint_path, payload, n_reps, n_reps)
    atomic_json(destination / "results.json", payload)
    print(destination / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
