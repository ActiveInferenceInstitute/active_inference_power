"""Semantic result-projection contracts for release derivative artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from scripts.release_artifact_contracts import validate_result_derivative_artifacts

def _derivative_fixture(
    tmp_path: Path, generated_fast_bundle: Path
) -> tuple[Path, dict[str, object]]:
    """Copy fresh real derived files into a separate mutation namespace."""
    source = generated_fast_bundle
    output = tmp_path / "output"
    (output / "demo").mkdir(parents=True)
    for name in (
        "certificates.json",
        "action_loop_suite.json",
        "formalism_registry.json",
    ):
        shutil.copy2(source / name, output / name)
    shutil.copy2(source / "demo" / "payload.json", output / "demo" / "payload.json")
    results = json.loads((source / "results.json").read_text(encoding="utf-8"))
    return output, results


def test_derived_release_artifacts_are_exact_result_projections(
    tmp_path: Path, generated_fast_bundle: Path
) -> None:
    output, results = _derivative_fixture(tmp_path, generated_fast_bundle)
    assert validate_result_derivative_artifacts(output, results) == {
        "derived_artifact_count": 4
    }


def test_derived_release_artifacts_reject_a_stale_suite_projection(
    tmp_path: Path, generated_fast_bundle: Path
) -> None:
    output, results = _derivative_fixture(tmp_path, generated_fast_bundle)
    suite = json.loads((output / "action_loop_suite.json").read_text(encoding="utf-8"))
    suite["claim_boundary"] = "stale projection"
    (output / "action_loop_suite.json").write_text(
        json.dumps(suite), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="action-loop suite artifact"):
        validate_result_derivative_artifacts(output, results)
