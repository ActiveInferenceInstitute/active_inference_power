"""Semantic bindings between result JSON and its derived release artifacts.

Manifest hashes establish byte integrity.  These checks additionally establish
that separately serialized reader-facing artifacts are the exact projections
of the same result object, rather than merely well-formed files from another
run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from active_inference_power.demo import load_demo_payload
from active_inference_power.formalism import validate_formalism_registry

from scripts.release_provenance import require_file


def _load_json(path: Path, description: str) -> Any:
    source = require_file(path, description)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON") from exc


def _require_equal_projection(
    output: Path,
    filename: str,
    expected: Any,
    description: str,
) -> None:
    actual = _load_json(output / filename, description)
    if actual != expected:
        raise ValueError(f"{description} does not match its results projection")


def _validate_demo_lineage(payload: Mapping[str, Any], results: Mapping[str, Any]) -> None:
    """Bind a self-validating demo to the result suite's declared scenario."""
    suite = results.get("suite")
    metadata = results.get("meta")
    if not isinstance(suite, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("results lack suite or metadata needed for demo lineage")
    scenario_id = payload.get("scenario", {}).get("scenario_id") if isinstance(payload.get("scenario"), Mapping) else None
    scenario_hash = payload.get("scenario_hash")
    policy = payload.get("policy")
    contracts = suite.get("scenario_contracts")
    if not isinstance(contracts, list) or not any(
        isinstance(contract, Mapping)
        and contract.get("scenario_id") == scenario_id
        and contract.get("scenario_hash") == scenario_hash
        for contract in contracts
    ):
        raise ValueError("demo scenario is not declared by the result suite")
    operations = suite.get("operating_characteristics")
    if not isinstance(operations, list) or not any(
        isinstance(operation, Mapping)
        and operation.get("scenario_id") == scenario_id
        and operation.get("scenario_hash") == scenario_hash
        and operation.get("policy") == policy
        for operation in operations
    ):
        raise ValueError("demo policy/scenario is not declared by the result suite")
    schedule = metadata.get("stage_seed_schedule")
    if not isinstance(schedule, Mapping) or payload.get("seed") != schedule.get(
        "conditional_power"
    ):
        raise ValueError("demo seed does not match the conditional-power stage lineage")


def validate_result_derivative_artifacts(
    output_root: Path, results: Mapping[str, Any]
) -> dict[str, int]:
    """Validate every non-rendered release artifact derived from ``results``."""
    output = Path(output_root)
    for key in ("certificates", "suite", "formalism"):
        if key not in results:
            raise ValueError(f"results omit {key} required for derived artifact validation")
    _require_equal_projection(
        output,
        "certificates.json",
        results["certificates"],
        "certificate artifact",
    )
    _require_equal_projection(
        output,
        "action_loop_suite.json",
        results["suite"],
        "action-loop suite artifact",
    )
    _require_equal_projection(
        output,
        "formalism_registry.json",
        results["formalism"],
        "formalism registry artifact",
    )
    if not isinstance(results["formalism"], Mapping):
        raise ValueError("results formalism registry must be an object")
    validate_formalism_registry(results["formalism"])
    demo = load_demo_payload(output / "demo" / "payload.json")
    _validate_demo_lineage(demo, results)
    return {"derived_artifact_count": 4}


__all__ = ["validate_result_derivative_artifacts"]
