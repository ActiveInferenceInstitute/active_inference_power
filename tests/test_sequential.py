"""Tests for fixed-horizon and time-uniform binary evidence."""

from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from active_inference_power.sequential import (
    bernoulli_confidence_sequence,
    binary_e_process,
    fixed_horizon_binomial_pvalue,
    mixture_e_process,
    predictable_e_process,
)


def test_e_process_has_path_and_optional_stopping_boundary() -> None:
    result = binary_e_process([1, 1, 1, 1], 0.4, 0.8, alpha=0.05)
    assert result.e_process_values.shape == (5,)
    assert result.e_process_values[0] == 1.0
    assert result.terminal_e == result.e_process_values[-1]
    assert result.crossing_time is None or 1 <= result.crossing_time <= 4


def test_e_process_log_path_agrees_with_the_uncapped_likelihood_ratio() -> None:
    result = binary_e_process([1, 0, 1], 0.4, 0.8)
    expected = np.array([1.0, 2.0, 2.0 / 3.0, 4.0 / 3.0])

    assert result.log_e_process_values is not None
    assert np.allclose(result.e_process_values, expected)
    assert np.allclose(np.exp(result.log_e_process_values), expected)
    assert result.terminal_log_e == pytest.approx(math.log(expected[-1]))
    assert not result.e_process_values_are_capped


def test_mixture_e_process_and_fixed_horizon_direction() -> None:
    result = mixture_e_process([0, 0, 1, 0], 0.6, [0.2, 0.4], [1, 2])
    assert np.all(np.isfinite(result.e_process_values))
    assert fixed_horizon_binomial_pvalue([0, 0, 0], 0.6, 0.2) < 1.0
    assert fixed_horizon_binomial_pvalue([1, 1, 1], 0.4, 0.8) < 1.0


def test_sequential_crossing_and_validation_boundaries() -> None:
    crossed = binary_e_process([1], 0.01, 0.99, alpha=0.05)
    assert crossed.crossed and crossed.crossing_time == 1
    with pytest.raises(ValueError):
        binary_e_process([], 0.0, 0.8)
    with pytest.raises(ValueError):
        binary_e_process([1], 0.4, 0.4)
    with pytest.raises(ValueError):
        binary_e_process([1], 0.4, 0.8, alpha=1.0)
    with pytest.raises(ValueError):
        mixture_e_process([1], 0.6, [])
    with pytest.raises(ValueError):
        mixture_e_process([1], 0.6, [0.2], weights=[0.0])
    with pytest.raises(ValueError):
        mixture_e_process([1], 0.6, [0.2], weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        fixed_horizon_binomial_pvalue([1], 0.0, 0.8)


def test_predictable_e_process_and_confidence_sequence_are_pathwise() -> None:
    result = predictable_e_process([1, 0, 1], 0.4, [0.8, 0.7, 0.75])
    assert result.e_process_values.shape == (4,)
    sequence = bernoulli_confidence_sequence([1, 0, 1, 1], alpha=0.1)
    assert sequence.lower.shape == sequence.upper.shape == (4,)
    assert np.all(sequence.lower <= sequence.upper)
    assert sequence.coverage_claim.startswith("simultaneous")
    with pytest.raises(ValueError):
        predictable_e_process([1, 0], 0.4, [0.8])
    with pytest.raises(ValueError):
        bernoulli_confidence_sequence([1], alternative_prior=(0.0, 1.0))


def test_e_processes_keep_long_stream_evidence_in_log_space() -> None:
    """Long favorable streams must not overflow the displayed e-process path."""
    observations = np.ones(240, dtype=int)
    results = (
        binary_e_process(observations, 0.001, 0.999),
        mixture_e_process(observations, 0.001, [0.9, 0.999], [1.0, 1.0]),
        predictable_e_process(observations, 0.001, np.full(observations.shape, 0.999)),
    )
    for result in results:
        assert np.all(np.isfinite(result.e_process_values))
        assert result.e_process_values[-1] == np.finfo(float).max
        assert result.terminal_e == result.e_process_values[-1]
        assert result.e_process_values_are_capped
        assert result.terminal_log_e > math.log(np.finfo(float).max)
        assert result.crossing_time == 1
        payload = result.to_mapping()
        assert payload["e_process_values_display_capped"] is True
        assert payload["terminal_log_e"] == pytest.approx(result.terminal_log_e)
        json.dumps(payload, allow_nan=False)


def test_e_process_uses_log_boundary_when_the_float_boundary_is_capped() -> None:
    """A capped float boundary must not cause an early Ville crossing."""
    observations = np.ones(160, dtype=int)
    alpha = np.nextafter(0.0, 1.0)
    result = binary_e_process(observations, 0.001, 0.999, alpha=float(alpha))
    first_capped = int(np.flatnonzero(result.e_process_values == np.finfo(float).max)[0])

    assert result.boundary_is_capped
    assert result.log_boundary > math.log(np.finfo(float).max)
    assert result.crossing_time is not None
    assert first_capped < result.crossing_time
    assert result.log_e_process_values is not None
    assert result.log_e_process_values[result.crossing_time] >= result.log_boundary
    assert result.log_e_process_values[result.crossing_time - 1] < result.log_boundary

    with pytest.raises(ValueError, match="finite display"):
        replace(result, log_e_process_values=result.log_e_process_values + 0.1)


def test_e_process_mapping_uses_only_unambiguous_evidence_names() -> None:
    result = binary_e_process([1, 0, 1], 0.4, 0.8)
    log_e_process_values = result.log_e_process_values
    assert log_e_process_values is not None

    payload = result.to_mapping()
    assert payload["e_process_values"] == result.e_process_values.tolist()
    assert payload["log_e_process_values"] == log_e_process_values.tolist()
    assert payload["evidence_kind"] == "likelihood_ratio_e_process"
    assert payload["terminal_e"] == result.terminal_e
    assert payload["terminal_log_e"] == result.terminal_log_e


@pytest.mark.parametrize("observations", [[2], [[0, 1]], [0, 3]])
def test_sequential_rejects_invalid_observations(observations: object) -> None:
    with pytest.raises(ValueError):
        binary_e_process(observations, 0.4, 0.8)  # type: ignore[arg-type]
