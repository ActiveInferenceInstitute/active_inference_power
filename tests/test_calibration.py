"""Real global-null p-value calibration diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference_power._types import TwoGroupsConfig
from active_inference_power.calibration import (
    calibration_replication_samples,
    null_pvalue_calibration,
    null_pvalue_calibration_from_samples,
)


def test_null_pvalue_calibration_uses_declared_independent_null() -> None:
    config = TwoGroupsConfig(m=20, pi0=0.8, effect=3.0, rho=0.5)
    result = null_pvalue_calibration(
        config, n_reps=200, seed=2026, grid=[0.0, 0.05, 0.5, 1.0]
    )
    assert result["regime"] == "global_null_independent"
    assert result["n_pvalues"] == 4000
    assert result["passes_dkw"] is True
    assert result["dkw_radius"] > 0.0
    assert len(result["grid"]) == len(result["empirical_cdf"]) == 4
    assert np.all(np.diff(result["empirical_cdf"]) >= 0.0)


def test_null_pvalue_calibration_is_reproducible() -> None:
    config = TwoGroupsConfig(m=8, pi0=1.0, effect=0.0)
    first = null_pvalue_calibration(config, 50, seed=7)
    second = null_pvalue_calibration(config, 50, seed=7)
    assert first == second


def test_chunked_calibration_blocks_are_reproducible_and_composable() -> None:
    config = TwoGroupsConfig(m=5, pi0=1.0, effect=0.0)
    first = calibration_replication_samples(config, 12, seed=19, start=0, stop=4)
    second = calibration_replication_samples(config, 12, seed=19, start=4, stop=12)
    assert calibration_replication_samples(config, 2, seed=19).size == 10
    combined = np.concatenate((first, second))
    repeated = np.concatenate(
        (
            calibration_replication_samples(config, 12, seed=19, start=0, stop=4),
            calibration_replication_samples(config, 12, seed=19, start=4, stop=12),
        )
    )
    assert np.array_equal(combined, repeated)
    result = null_pvalue_calibration_from_samples(config, 12, combined, seed=19)
    assert result["n_pvalues"] == 60
    assert result["seed"] == 19
    assert result["replication_seed_schedule"]["kind"] == "numpy.SeedSequence"


def test_chunked_calibration_rejects_inconsistent_ranges_and_accumulators() -> None:
    config = TwoGroupsConfig(m=3, pi0=1.0, effect=0.0)
    with pytest.raises(ValueError, match="stop"):
        calibration_replication_samples(config, 4, seed=1, start=3, stop=5)
    with pytest.raises(ValueError, match="length"):
        null_pvalue_calibration_from_samples(config, 4, [0.5] * 11)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        null_pvalue_calibration_from_samples(config, 4, [1.2] * 12)


@pytest.mark.parametrize("bad_n", [0, 1.5, True])
def test_chunked_calibration_rejects_invalid_replication_counts(bad_n: object) -> None:
    config = TwoGroupsConfig(m=2, pi0=1.0, effect=0.0)
    with pytest.raises(ValueError):
        calibration_replication_samples(config, bad_n)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        null_pvalue_calibration_from_samples(config, bad_n, [0.5] * 2)  # type: ignore[arg-type]


def test_chunked_calibration_rejects_invalid_seed_start_and_empty_model() -> None:
    config = TwoGroupsConfig(m=2, pi0=1.0, effect=0.0)
    with pytest.raises(ValueError):
        calibration_replication_samples(config, 2, seed=-1)
    with pytest.raises(ValueError):
        calibration_replication_samples(config, 2, start=-1)
    with pytest.raises(ValueError):
        calibration_replication_samples(TwoGroupsConfig(m=0, pi0=1.0, effect=0.0), 2)
    with pytest.raises(ValueError):
        null_pvalue_calibration_from_samples(
            TwoGroupsConfig(m=0, pi0=1.0, effect=0.0), 2, []
        )


def test_chunked_calibration_rejects_nonfinite_samples_and_grid_contracts() -> None:
    config = TwoGroupsConfig(m=2, pi0=1.0, effect=0.0)
    samples = [0.2, 0.8, 0.4, 0.6]
    with pytest.raises(ValueError, match="confidence"):
        null_pvalue_calibration_from_samples(config, 2, samples, confidence=1.0)
    with pytest.raises(ValueError, match="finite"):
        null_pvalue_calibration_from_samples(config, 2, [0.2, 0.8, np.nan, 0.6])
    with pytest.raises(ValueError, match="at least two"):
        null_pvalue_calibration_from_samples(config, 2, samples, grid=[0.5])
    with pytest.raises(ValueError, match="finite"):
        null_pvalue_calibration_from_samples(config, 2, samples, grid=[0.0, np.nan])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        null_pvalue_calibration_from_samples(config, 2, samples, grid=[-0.1, 1.0])


@pytest.mark.parametrize("bad_n", [0, 1.5, True])
def test_null_pvalue_calibration_rejects_invalid_replication_counts(bad_n: object) -> None:
    with pytest.raises(ValueError):
        null_pvalue_calibration(TwoGroupsConfig(m=2, pi0=1.0, effect=0.0), bad_n)  # type: ignore[arg-type]


def test_null_pvalue_calibration_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="m"):
        null_pvalue_calibration(TwoGroupsConfig(m=0, pi0=1.0, effect=0.0), 5)


def test_null_pvalue_calibration_rejects_invalid_confidence_and_grid() -> None:
    config = TwoGroupsConfig(m=2, pi0=1.0, effect=0.0)
    with pytest.raises(ValueError):
        null_pvalue_calibration(config, 2, confidence=1.0)
    with pytest.raises(ValueError):
        null_pvalue_calibration(config, 2, grid=[0.5])
    with pytest.raises(ValueError):
        null_pvalue_calibration(config, 2, grid=[0.0, np.nan])
