"""Real covariance and negative/block dependence contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference_power.dependence import DependenceConfig, correlation_matrix, sample_gaussian
from active_inference_power._types import TwoGroupsConfig
from active_inference_power.simulation import generate_pvalues


def test_negative_equicorrelation_is_psd_and_reproducible() -> None:
    config = DependenceConfig(kind="equicorrelated", rho=-0.2 / 4.0)
    matrix = correlation_matrix(5, config)
    assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-10
    left = sample_gaussian(np.random.default_rng(2), 5, config)
    right = sample_gaussian(np.random.default_rng(2), 5, config)
    assert np.array_equal(left, right)


def test_block_dependence_and_simulation_contract() -> None:
    config = TwoGroupsConfig(m=8, pi0=0.5, effect=1.0, rho=0.3, dependence_kind="block", block_size=4)
    pvalues, is_alt = generate_pvalues(config, np.random.default_rng(3))
    assert pvalues.shape == (8,)
    assert is_alt.shape == (8,)
    assert np.all((pvalues >= 0.0) & (pvalues <= 1.0))


def test_dependence_validation_and_all_matrix_kinds() -> None:
    assert correlation_matrix(3, DependenceConfig(kind="independent")).shape == (3, 3)
    factor = correlation_matrix(3, DependenceConfig(kind="factor", loadings=np.array([0.2, -0.1, 0.3])))
    assert np.allclose(np.diag(factor), 1.0)
    assert correlation_matrix(1, DependenceConfig(kind="equicorrelated", rho=0.0)).shape == (1, 1)
    with pytest.raises(ValueError):
        correlation_matrix(0, DependenceConfig())
    with pytest.raises(ValueError):
        correlation_matrix(3, DependenceConfig(kind="unknown"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        correlation_matrix(1, DependenceConfig(kind="equicorrelated", rho=0.1))
    with pytest.raises(ValueError):
        correlation_matrix(3, DependenceConfig(kind="factor"))
    with pytest.raises(ValueError):
        correlation_matrix(3, DependenceConfig(kind="factor", loadings=np.ones(3)))
    with pytest.raises(ValueError):
        correlation_matrix(3, DependenceConfig(kind="block", block_size=4))
    with pytest.raises(ValueError):
        sample_gaussian(np.random.default_rng(1), 3, DependenceConfig(kind="independent"), np.zeros(2))


def test_independent_rho_cannot_silently_claim_ignored_dependence() -> None:
    with pytest.raises(ValueError, match="must be zero for independent"):
        DependenceConfig(kind="independent", rho=0.25).validate(4)
    with pytest.raises(ValueError, match="must be zero for independent"):
        TwoGroupsConfig(
            m=4,
            pi0=0.5,
            effect=1.0,
            dependence_kind="independent",
            rho=0.25,
        )

    # The mutable public configuration is defensively rechecked at generation
    # time, so a post-construction mutation cannot mislabel an iid draw.
    mutated = TwoGroupsConfig(
        m=4,
        pi0=0.5,
        effect=1.0,
        dependence_kind="independent",
    )
    mutated.rho = 0.25
    with pytest.raises(ValueError, match="must be zero for independent"):
        generate_pvalues(mutated, np.random.default_rng(11))


@pytest.mark.parametrize(
    "config",
    [
        DependenceConfig(kind="equicorrelated", rho=-0.3),
        DependenceConfig(kind="block", rho=-0.6, block_size=3),
    ],
)
def test_dependence_rejects_invalid_psd_ranges(config: DependenceConfig) -> None:
    with pytest.raises(ValueError):
        correlation_matrix(5, config)


def test_factor_sampling_applies_valid_and_invalid_means() -> None:
    config = DependenceConfig(kind="factor", loadings=np.array([0.2, -0.1, 0.3]))
    draw = sample_gaussian(np.random.default_rng(4), 3, config, mean=np.zeros(3))
    assert draw.shape == (3,)
    with pytest.raises(ValueError):
        sample_gaussian(np.random.default_rng(4), 3, config, mean=np.zeros(2))
