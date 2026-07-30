"""Real-file tests for the imported two-sided Gaussian BH protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from active_inference_power.protocols import (
    DobribanBHProtocol,
    load_dobriban_bh_protocol,
    run_dobriban_bh_chunk,
)


def test_protocol_loads_from_packaged_plan_and_runs_deterministically() -> None:
    protocol = load_dobriban_bh_protocol()
    assert protocol.plan["model"]["model_id"] == "bh-two-sided-gaussian-counterexample-v1"
    assert protocol.plan["model"]["group_multipliers"] == [96, 1, 3]
    assert protocol.to_mapping()["evidence_class"] == "stress_test"
    first = run_dobriban_bh_chunk(1, strata=8, macro_replications=4, master_seed=23, target_entries=32, protocol=protocol)
    second = run_dobriban_bh_chunk(1, strata=8, macro_replications=4, master_seed=23, target_entries=32, protocol=protocol)
    assert first.to_mapping() == second.to_mapping()
    assert first.m == 100
    assert first.seed_spawn_key == (0,)
    assert 0.0 <= first.fdr_hat <= 1.0
    assert first.mcse > 0.0


def test_protocol_rejects_tampering_and_invalid_run_arguments(tmp_path: Path) -> None:
    protocol = load_dobriban_bh_protocol()
    for mutation in (
        lambda value: value["model"].update(alpha=0.02),
        lambda value: value.update(schema_version=2),
        lambda value: value.update(rng_bit_generator="MT19937"),
        lambda value: value.update(seed_construction="default_rng"),
        lambda value: value.update(expected_seed_spawn_key=[1]),
        lambda value: value.update(strata=999),
        lambda value: value.update(expected_environment={}),
        lambda value: value.update(chunks=[]),
        lambda value: value.update(chunks=[{"prefix": "tampered", "N": 50, "block": 1, "master_seed": 20260711, "macro_replications": 100}]),
        lambda value: value.update(model="not-a-mapping"),
    ):
        payload = json.loads(json.dumps(protocol.plan))
        mutation(payload)
        with pytest.raises(ValueError):
            DobribanBHProtocol(payload)
    with pytest.raises(ValueError, match="HTTPS"):
        DobribanBHProtocol(protocol.plan, source_url="http://example.invalid")
    with pytest.raises(ValueError, match="SHA-256"):
        DobribanBHProtocol(protocol.plan, source_plan_sha256="0")
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_dobriban_bh_protocol(bad_path)
    for kwargs in (
        {"N": 0},
        {"N": 1, "strata": 0},
        {"N": 1, "macro_replications": 1},
        {"N": 1, "master_seed": -1},
        {"N": 1, "target_entries": 0},
    ):
        with pytest.raises(ValueError):
            run_dobriban_bh_chunk(protocol=protocol, **kwargs)
    with pytest.raises(ValueError, match="variance"):
        run_dobriban_bh_chunk(1, strata=1, macro_replications=2, master_seed=0, target_entries=32, protocol=protocol)


def test_protocol_source_metadata_is_hash_shaped() -> None:
    protocol = load_dobriban_bh_protocol()
    mapping = protocol.to_mapping()
    assert mapping["source_url"].startswith("https://")
    assert len(mapping["source_plan_sha256"]) == 64
    assert len(mapping["source_generator_sha256"]) == 64
