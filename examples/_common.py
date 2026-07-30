"""Shared utilities for the offline Active Inference Power examples."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.artifacts import require_safe_output_root  # noqa: E402


def output_path(name: str, output_root: Path | None) -> Path:
    root = ROOT / "output" / "examples" if output_root is None else Path(output_root)
    if not root.is_absolute():
        root = ROOT / root
    root = require_safe_output_root(root, project_root=ROOT)
    # An explicit output root is a caller-owned artifact location.  It may be
    # outside the checkout (for example, a temporary or mounted release
    # volume); only the default remains project-local.
    destination = root / name
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def checkpoint(path: Path, payload: dict[str, Any], completed: int, total: int) -> None:
    atomic_json(path, {
        "schema_version": "2.0",
        "completed_replications": int(completed),
        "total_replications": int(total),
        "value": payload,
    })


def resume_value(path: Path, total: int) -> dict[str, Any] | None:
    """Return a completed example payload after a fail-safe checkpoint check."""
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "2.0":
        raise ValueError("checkpoint schema does not match this example")
    if value.get("total_replications") != int(total) or value.get("completed_replications") != int(total):
        raise ValueError("checkpoint replication accounting does not match this run")
    payload = value.get("value")
    if not isinstance(payload, dict):
        raise ValueError("completed checkpoint does not contain a mapping payload")
    return payload
