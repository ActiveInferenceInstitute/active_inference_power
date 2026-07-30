#!/usr/bin/env python3
"""Compare policies while actions change sensing, context, cost, and stopping."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from active_inference_power.action_loop import ActionPolicy, evaluate_power_surface  # noqa: E402
from active_inference_power.scenario import make_reference_action_scenario  # noqa: E402
from examples._common import atomic_json, checkpoint, output_path, resume_value  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("fast", "release"), default="fast")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
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
    scenarios = [
        make_reference_action_scenario(reliability=value, horizon=12)
        for value in (0.65, 0.8, 0.92)
    ]
    policies: tuple[ActionPolicy, ...] = (
        "fixed_horizon",
        "information_gain",
        "cost_aware",
        "adaptive_target",
        "posterior_sampling",
        "e_process",
    )
    results = evaluate_power_surface(scenarios, policies, n_reps, seed=args.seed, workers=args.workers)
    payload = {
        "suite": "environment_level_control",
        "conditional_dimensions": {"reliability": [0.65, 0.8, 0.92], "policies": list(policies)},
        "results": results,
        "claim_boundary": "Environment-level action loops are conditional operating characteristics; formal validity requires the declared action kernel and evidence assumptions.",
    }
    destination = output_path("02_environment_control", args.output_root)
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
    if args.write_completed_checkpoint:
        checkpoint(checkpoint_path, payload, n_reps, n_reps)
    atomic_json(destination / "results.json", payload)
    print(destination / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
