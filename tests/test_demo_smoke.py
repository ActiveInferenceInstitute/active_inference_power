"""Headless contracts for the optional demo's downloadable trace bundle."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from active_inference_power import demo
from active_inference_power.demo import (
    build_demo_payload,
    canonical_demo_json,
    load_demo_payload,
    validate_demo_payload,
)
from scripts.smoke_demo import smoke_demo_payload


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "demo_payload.json"
SMOKE_SCRIPT = ROOT / "scripts" / "smoke_demo.py"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _rehash_trace(payload: dict[str, Any]) -> None:
    payload["trace_sha256"] = demo._sha256_payload(payload["trace"])


def _rehash_visible_trace(payload: dict[str, Any]) -> None:
    payload["visible_trace_sha256"] = demo._sha256_payload(
        {"visible_trace": payload["visible_trace"]}
    )


def _tamper_visible_projection(payload: dict[str, Any]) -> None:
    payload["visible_trace"] = []
    _rehash_visible_trace(payload)


def _tamper_trace_scenario(payload: dict[str, Any]) -> None:
    payload["trace"]["scenario_id"] = "wrong"
    _rehash_trace(payload)


def _tamper_trace_policy(payload: dict[str, Any]) -> None:
    payload["trace"]["policy"] = "wrong"
    _rehash_trace(payload)


def test_committed_demo_payload_and_new_builds_use_canonical_evidence_names() -> None:
    fixture = load_demo_payload(FIXTURE)
    built = build_demo_payload(
        reliability=0.72, horizon=3, policy="cost_aware", seed=12
    )
    assert json.loads(canonical_demo_json(fixture)) == fixture
    assert fixture["trace"]["visible_steps"] == fixture["visible_trace"]
    assert "terminal_e_process_value" in fixture["trace"]
    assert "terminal_e_process_value" in built["trace"]
    assert built["trace"]["visible_steps"] == built["visible_trace"]
    assert "e_process_value" in built["visible_trace"][0]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda value: value["trace"].pop("terminal_e_process_value"),
            "terminal_e_process_value",
        ),
        (
            lambda value: (
                value["visible_trace"][0].pop("e_process_value"),
                value["trace"]["visible_steps"][0].pop("e_process_value"),
            ),
            "e_process_value",
        ),
        (
            lambda value: value["trace"].update(evidence_kind="wrong"),
            "evidence_kind",
        ),
    ],
)
def test_demo_payload_requires_canonical_e_process_fields(
    mutate, message: str
) -> None:
    payload = _fixture()
    mutate(payload)
    _rehash_trace(payload)
    _rehash_visible_trace(payload)
    unsigned = dict(payload)
    unsigned.pop("payload_sha256")
    payload["payload_sha256"] = demo._sha256_payload(unsigned)
    with pytest.raises(ValueError, match=message):
        validate_demo_payload(payload)


def test_headless_smoke_loads_a_committed_trace_and_never_requires_a_browser() -> None:
    report = smoke_demo_payload(FIXTURE)
    assert report["status"] == "passed"
    assert report["browser_required"] is False
    assert report["stopping_time"] == 3
    assert report["first_visible_step"]["time"] == 1
    completed = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT), "--payload", str(FIXTURE)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == report


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda value: value.update(schema_version="other"), "schema_version"),
        (lambda value: value.update(scenario=[]), "scenario must be"),
        (lambda value: value.update(scenario_hash="not-a-digest"), "scenario_hash"),
        (lambda value: value.update(configuration_hash="b" * 64), "hashes disagree"),
        (
            lambda value: value["scenario"].update(scenario_hash="b" * 64),
            "hashes disagree",
        ),
        (
            lambda value: value["scenario"].update(description="tampered contract"),
            "does not match its configuration hash",
        ),
        (lambda value: value.update(trace=[]), "trace must be"),
        (lambda value: value.update(trace_sha256="b" * 64), "trace_sha256"),
        (lambda value: value.update(visible_trace={}), "visible_trace must"),
        (
            lambda value: value.update(visible_trace_sha256="b" * 64),
            "visible_trace_sha256",
        ),
        (_tamper_visible_projection, "exactly project"),
        (_tamper_trace_scenario, "scenario_id"),
        (_tamper_trace_policy, "policy or seed"),
        (lambda value: value.update(payload_sha256="b" * 64), "payload_sha256"),
    ],
)
def test_demo_validator_rejects_each_tamper_boundary(mutate, message: str) -> None:
    payload = deepcopy(_fixture())
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        validate_demo_payload(payload)


def test_demo_json_and_file_loader_reject_nonfinite_or_invalid_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite JSON"):
        canonical_demo_json({"value": float("nan")})
    with pytest.raises(ValueError, match="JSON object"):
        demo._require_mapping([], "payload")
    with pytest.raises(ValueError, match="SHA-256"):
        demo._require_digest(None, "digest")
    with pytest.raises(ValueError, match="could not load"):
        load_demo_payload(tmp_path / "missing.json")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="could not load"):
        load_demo_payload(malformed)


def test_demo_reader_labels_qualify_evidence_accounting_and_cost() -> None:
    """Keep the optional public UI from conflating E_t, alpha accounting, and C."""
    source = Path(demo.__file__).read_text(encoding="utf-8")
    for forbidden_term in (
        '"Terminal wealth"',
        '"e-wealth"',
        'ylabel="unit-scale evidence"',
    ):
        assert forbidden_term not in source
    for required in (
        "Terminal likelihood-ratio e-process E_T",
        "Likelihood-ratio e-process evidence E_t",
        "E-process value E_t\\n(dimensionless; linear scale)",
        "Simulated cost budget C (scenario-local units)",
        "Simulated cost C ({cost_unit}; {cost_scale_id})",
        "Cumulative cost C\\n({cost_unit}; {cost_scale_id})",
        "separate, noncommensurate scales",
        "per-target optional-stopping",
        "Posterior probability / p-value (unit scale)",
    ):
        assert required in source
    assert "terminal_e_process_value" in source
    assert "e_process_value" in source
