"""Chunk-level checkpoint contracts for deterministic release stages.

The release orchestrator currently checkpoints completed stages.  This module
also defines the stricter partial-chunk shape used by resumable extensions:
the cursor, seed schedule, dependency hashes, and serializable metric
accumulators travel together and are validated before a resume is accepted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from active_inference_power.schema import validate_checkpoint
from scripts.artifacts import atomic_write_json


def write_partial_checkpoint(
    path: Path,
    *,
    run_id: str,
    config_sha256: str,
    claim_ledger_sha256: str,
    execution_source_sha256: str | None = None,
    profile: str,
    seed: int,
    stage_seed: int,
    workers: int,
    stage: str,
    total_replications: int,
    completed_replications: int,
    chunk_size: int,
    metrics: Mapping[str, Any],
    dependency_hashes: Mapping[str, list[str]],
    value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write and validate a partial deterministic checkpoint."""
    if completed_replications < 1 or total_replications < completed_replications:
        raise ValueError("completed_replications must be positive and no greater than total_replications")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    checkpoint: dict[str, Any] = {
        "schema_version": "2.0",
        "run_id": str(run_id),
        "config_sha256": str(config_sha256),
        "claim_ledger_sha256": str(claim_ledger_sha256),
        "profile": str(profile),
        "seed": int(seed),
        "stage_seed": int(stage_seed),
        "seed_schedule": {
            "kind": "numpy.SeedSequence",
            "master_seed": int(seed),
            "stage": str(stage),
            "stage_seed": int(stage_seed),
            "replication_order": "stable ascending child-stream order",
        },
        "workers": int(workers),
        "stage": str(stage),
        "total_replications": int(total_replications),
        "completed_replications": int(completed_replications),
        "chunk_size": int(chunk_size),
        "replication_cursor": {"completed": int(completed_replications), "next": int(completed_replications)},
        "partial_metric_accumulators": {
            "status": "partial",
            "completed_replications": int(completed_replications),
            "metrics": dict(metrics),
        },
        "dependency_hashes": {key: list(value) for key, value in dependency_hashes.items()},
        "value": (
            dict(value)
            if value is not None
            else {"total_replications": int(total_replications), "metrics": dict(metrics)}
        ),
    }
    if execution_source_sha256 is not None:
        checkpoint["execution_source_sha256"] = str(execution_source_sha256)
    validate_checkpoint(checkpoint)
    atomic_write_json(path, checkpoint)
    return checkpoint


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint and fail closed on schema or cursor corruption."""
    import json

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_checkpoint(value)
    return value


__all__ = ["write_partial_checkpoint", "load_checkpoint"]
