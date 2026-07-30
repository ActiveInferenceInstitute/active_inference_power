"""Real-stream tests for the sequential-evidence comparison atlas."""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pytest

from active_inference_power.sequential_atlas import (
    SequentialEvidenceAtlasConfig,
    _conditions,
    _fixed_horizon_peek_crossing_time,
    _mixture_rates,
    _simple_evalues,
    _stopping_summary,
    evaluate_sequential_evidence_atlas,
)


def _atlas_config() -> SequentialEvidenceAtlasConfig:
    return SequentialEvidenceAtlasConfig(
        null_rates=(0.4,),
        alternative_rates=(0.65, 0.8),
        horizon=6,
        n_reps=5,
        alpha=0.1,
        mixture_alternative_rates=(0.25, 0.65, 0.75),
    )


def test_atlas_compares_all_families_with_worker_invariant_child_streams() -> None:
    config = _atlas_config()
    serial = evaluate_sequential_evidence_atlas(config, seed=19, workers=1)
    parallel = evaluate_sequential_evidence_atlas(config, seed=19, workers=3)

    assert serial == parallel
    assert serial["configuration_hash"] == config.configuration_hash
    seed_schedule = cast(dict[str, Any], serial["seed_schedule"])
    assert seed_schedule["kind"] == "numpy.SeedSequence"
    assert seed_schedule["n_child_streams"] == 4 * config.n_reps
    assert serial["common_random_numbers"] == {
        "enabled": True,
        "scope": "each condition/replication stream is reused by fixed-simple, mixture, predictable, confidence-sequence, and contrast calculations",
    }
    assert "finite independent-Bernoulli" in str(serial["validity"])

    conditions = cast(list[dict[str, Any]], serial["conditions"])
    assert [condition["truth_regime"] for condition in conditions] == [
        "declared_null",
        "declared_alternative",
        "declared_null",
        "declared_alternative",
    ]
    assert {condition["true_rate"] for condition in conditions} == {0.4, 0.65, 0.8}
    for condition in conditions:
        assert condition["n_reps"] == config.n_reps
        families = cast(dict[str, dict[str, Any]], condition["families"])
        assert set(families) == {"fixed_simple", "mixture", "predictable"}
        specification = cast(dict[str, dict[str, Any]], condition["family_specifications"])
        assert len(specification["mixture"]["alternative_rates"]) >= 2
        assert len(specification["predictable"]["alternative_rates_by_time"]) == config.horizon
        assert all(
            rate != condition["null_rate"]
            for rate in specification["predictable"]["alternative_rates_by_time"]
        )
        for family in families.values():
            crossing = family["ville_boundary"]["crossing"]
            stopping = family["ville_boundary"]["stopping_time_distribution"]
            assert crossing["n_reps"] == config.n_reps
            assert 0.0 <= crossing["rate"] <= 1.0
            assert sum(item["count"] for item in stopping) == config.n_reps
            assert "declared nonnegative e-process null" in family["validity"]
        coverage = condition["confidence_sequence"]
        assert coverage["simultaneous_coverage"]["n_reps"] == config.n_reps
        assert coverage["coverage_claim"].startswith("simultaneous-in-time")
        assert "not a coverage proof" in coverage["validity"]
        e_lond = condition["e_lond"]
        trace = e_lond["representative_trace"]
        assert len(trace["evalues"]) == config.horizon
        assert len(trace["accounting_path"]) == config.horizon + 1
        assert trace["accounting_kind"] == "cumulative_nominal_level_allocation"
        assert "online-FDR validity" in e_lond["validity"]
        contrast = condition["fixed_horizon_optional_stopping_contrast"]
        assert contrast["validity"] == "invalid_optional_stopping_contrast"
        assert "not a Type-I-error-control claim" in contrast["claim_boundary"]
    json.dumps(serial, allow_nan=False)


def test_atlas_configuration_and_helper_boundaries_are_explicit() -> None:
    config = _atlas_config()
    assert config.configuration_hash == _atlas_config().configuration_hash
    assert len(_conditions(config)) == 4
    # Both global grid values collide with this cell's null/target, so the
    # deterministic companion makes the reported mixture genuinely nontrivial.
    assert len(_mixture_rates(0.25, 0.75, (0.25, 0.75))) == 2
    assert _fixed_horizon_peek_crossing_time(np.array([1]), 0.01, 0.99, 0.05) == 1
    assert _fixed_horizon_peek_crossing_time(np.array([0, 0]), 0.5, 0.9, 0.05) is None
    assert _simple_evalues(np.array([1, 0]), 0.4, 0.8) == pytest.approx([2.0, 1.0 / 3.0])
    stopping = _stopping_summary([1, 2, None], 3, 0.95)
    assert stopping["mean_crossing_time_given_crossed"] == pytest.approx(1.5)
    assert stopping["stopping_time_distribution"] == [
        {"time": 1, "count": 1},
        {"time": 2, "count": 1},
        {"time": None, "count": 1},
    ]

    with pytest.raises(ValueError, match="non-empty"):
        SequentialEvidenceAtlasConfig((), (0.7,), 3, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        SequentialEvidenceAtlasConfig((0.4, 0.4), (0.7,), 3, 1)
    with pytest.raises(ValueError, match="differ"):
        SequentialEvidenceAtlasConfig((0.4,), (0.4,), 3, 1)
    with pytest.raises(ValueError, match="horizon"):
        SequentialEvidenceAtlasConfig((0.4,), (0.7,), 0, 1)
    with pytest.raises(ValueError, match="n_reps"):
        SequentialEvidenceAtlasConfig((0.4,), (0.7,), 3, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha"):
        SequentialEvidenceAtlasConfig((0.4,), (0.7,), 3, 1, alpha=0.0)
    with pytest.raises(ValueError, match="confidence"):
        SequentialEvidenceAtlasConfig((0.4,), (0.7,), 3, 1, confidence=1.0)
    with pytest.raises(TypeError, match="config"):
        evaluate_sequential_evidence_atlas(cast(SequentialEvidenceAtlasConfig, object()))
    with pytest.raises(ValueError, match="seed"):
        evaluate_sequential_evidence_atlas(config, seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="workers"):
        evaluate_sequential_evidence_atlas(config, workers=0)
