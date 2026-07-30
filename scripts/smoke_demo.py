#!/usr/bin/env python3
"""Headless integrity smoke test for an optional Streamlit trace bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from active_inference_power.demo import load_demo_payload  # noqa: E402


def smoke_demo_payload(path: Path) -> dict[str, Any]:
    """Load one committed/downloaded trace without importing a browser runtime."""
    payload = load_demo_payload(path)
    visible_trace = payload["visible_trace"]
    if not visible_trace:
        raise ValueError("demo payload must contain at least one visible trace step")
    trace = payload["trace"]
    return {
        "status": "passed",
        "payload": str(path),
        "scenario_hash": payload["configuration_hash"],
        "trace_sha256": payload["trace_sha256"],
        "visible_trace_sha256": payload["visible_trace_sha256"],
        "stopping_time": trace["stopping_time"],
        "first_visible_step": visible_trace[0],
        "browser_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a deterministic demo trace bundle.")
    parser.add_argument(
        "--payload",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "demo_payload.json",
        help="JSON trace payload to load and validate",
    )
    args = parser.parse_args()
    print(json.dumps(smoke_demo_payload(args.payload), sort_keys=True))


if __name__ == "__main__":
    main()
