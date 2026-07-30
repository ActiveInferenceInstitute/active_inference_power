"""Pytest configuration and real generated-artifact fixtures."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import pytest

# Force headless backend for matplotlib in tests
os.environ.setdefault("MPLBACKEND", "Agg")

# Add the source layout and project root so tests exercise the canonical package
# exactly as scripts do from a checkout.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for _path in (str(ROOT), str(SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


@pytest.fixture(scope="session")
def generated_fast_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate one real, isolated fast evidence bundle for artifact tests.

    The fixture deliberately invokes the same result orchestrator used by the
    package rather than consuming an ignored project-root ``output/`` tree.
    Checkpoints are enabled because provenance tests exercise their real
    dependency witnesses as well as the result, figure, and dashboard files.
    """
    from scripts.generate_results import run

    output_root = tmp_path_factory.mktemp("generated-fast-bundle") / "output"
    results = run(
        ROOT / "manuscript" / "config.yaml",
        profile="fast",
        workers=1,
        output_root=output_root,
        checkpoint_every=500,
    )
    assert results["meta"]["profile"] == "fast"
    assert (output_root / "results.json").is_file()
    assert (output_root / "run_log.json").is_file()
    assert (output_root / "manifest.json").is_file()
    return output_root


@pytest.fixture(scope="session")
def small_real_certificates() -> dict[str, dict[str, object]]:
    """Return a seeded real certificate map with the known Storey failure.

    The small deterministic run supplies a genuine failed certificate for
    campaign-status tests without treating a retained campaign receipt as a
    fixture.  Its numerical records are produced by the domain implementation,
    not hand-authored test doubles.
    """
    from active_inference_power.certificates import CERTIFICATE_NAMES, run_all_certificates

    certificates = run_all_certificates(fast=True, m=12, n_reps=3, seed=1)
    assert tuple(certificates) == CERTIFICATE_NAMES
    assert certificates["storey_power_gain"]["passes"] is False
    return certificates


@pytest.fixture
def temporary_status_bundle(
    tmp_path: Path,
    generated_fast_bundle: Path,
    small_real_certificates: dict[str, dict[str, object]],
) -> Callable[..., dict[str, Any]]:
    """Write minimal schema-valid release/campaign status witnesses.

    Scientific result content comes from the session's real fast run.  The
    terminal certificate map is independently recomputed at a small seeded
    scale, letting tests exercise failed, diagnostic, and passing status
    transitions without copying preserved output trees.
    """
    from active_inference_power.certificate_gate import certificate_gate_result
    from active_inference_power.certificates import CERTIFICATE_NAMES
    from active_inference_power.schema import (
        validate_manifest,
        validate_results,
        validate_run_log,
    )
    from scripts.artifacts import artifact_record, atomic_write_json
    from scripts.run_ledger import RunLedger

    source_results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    counter = 0

    def build(
        *,
        profile: str,
        diagnostic_storey_failure: bool = False,
        force_all_certificates_pass: bool = False,
        complete: bool = False,
    ) -> dict[str, Any]:
        nonlocal counter
        counter += 1
        output_root = tmp_path / f"{profile}-status-{counter}"
        output_root.mkdir()

        certificates = copy.deepcopy(small_real_certificates)
        if force_all_certificates_pass:
            for certificate in certificates.values():
                certificate["passes"] = True
        policy = {
            "terminal": [
                name for name in CERTIFICATE_NAMES
                if not (diagnostic_storey_failure and name == "storey_power_gain")
            ],
            "diagnostic": ["storey_power_gain"] if diagnostic_storey_failure else [],
        }
        gate = certificate_gate_result(certificates, policy)
        terminal_passed = bool(gate["terminal_certificates_pass"])

        results = copy.deepcopy(source_results)
        metadata = results["meta"]
        metadata["profile"] = profile
        metadata["certificate_gate"] = gate
        execution_digest = str(metadata["execution_source_sha256"])
        ledger = RunLedger(
            output_root / "run_log.json",
            config_sha256=str(metadata["config_sha256"]),
            profile=profile,
            seed=int(metadata["seed"]),
            workers=1,
            execution_source_sha256=execution_digest,
        )
        ledger.data["claim_ledger_sha256"] = str(metadata["claim_ledger_sha256"])
        if terminal_passed:
            ledger.finish(
                "passed",
                certificates_pass=True,
                certificate_gate=gate,
            )
        else:
            ledger.fail(
                RuntimeError("one or more terminal finite-simulation certificates failed"),
                certificates_pass=False,
                certificate_gate=gate,
            )
        run_log = json.loads(
            (output_root / "run_log.json").read_text(encoding="utf-8")
        )
        results["run_id"] = run_log["run_id"]
        results["certificates"] = certificates

        certificate_path = output_root / "certificates.json"
        atomic_write_json(certificate_path, certificates)
        manifest = {
            "schema_version": "2.0",
            "run_id": run_log["run_id"],
            "complete": complete,
            "config_sha256": metadata["config_sha256"],
            "execution_source_sha256": execution_digest,
            "certificates_pass": terminal_passed,
            "certificate_gate": gate,
            "artifacts": {
                "certificates": artifact_record(
                    certificate_path,
                    output_root,
                    "certificate_results",
                    root_label="output",
                )
            },
        }
        validate_results(results, require_suite=True)
        validate_run_log(run_log)
        validate_manifest(manifest)
        atomic_write_json(output_root / "results.json", results)
        atomic_write_json(output_root / "manifest.json", manifest)
        return {
            "output_root": output_root,
            "results": results,
            "run_log": run_log,
            "manifest": manifest,
            "certificates": certificates,
            "gate": gate,
        }

    return build
