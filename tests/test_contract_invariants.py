"""Boundary tests for the package's public numerical data contracts.

These tests deliberately construct invalid real values rather than mocking
callers. They keep the validation branches executable as the scientific API
evolves, which is especially important for generated publication artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import Any, cast

from active_inference_power._types import CorrectionResult, SimulationResult, TwoGroupsConfig
from active_inference_power.active_inference import (
    AITestConfig,
    evaluate_ai_regime,
    evidence_pvalue,
    run_ai_test,
)
from active_inference_power.simulation import generate_pvalues


def _valid_correction(**overrides: object) -> CorrectionResult:
    values: dict[str, Any] = {
        "method": "benjamini_hochberg",
        "adjusted": np.array([0.01, 0.2]),
        "rejected": np.array([True, False]),
        "alpha": 0.05,
        "threshold": 0.01,
    }
    values.update(overrides)
    return CorrectionResult(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "bh"},
        {"adjusted": np.zeros((1, 2))},
        {"rejected": np.zeros((1, 2), dtype=bool)},
        {"rejected": np.zeros(1, dtype=bool)},
        {"rejected": np.array([False, False])},
        {"adjusted": np.array([np.nan])},
        {"adjusted": np.array([-0.1])},
        {"alpha": np.nan},
        {"threshold": 1.1},
        {"alpha": -0.1},
        {"threshold": -0.1},
    ],
)
def test_correction_result_rejects_invalid_invariants(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _valid_correction(**overrides)


def _valid_simulation(**overrides: object) -> SimulationResult:
    values: dict[str, Any] = {
        "method": "benjamini_hochberg",
        "n_reps": 2,
        "fdr": 0.1,
        "power": 0.5,
        "fwer": 0.5,
        "mean_rejections": 1.0,
        "fdp_samples": np.array([0.0, 0.2]),
        "power_samples": np.array([0.4, 0.6]),
    }
    values.update(overrides)
    return SimulationResult(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "bh"},
        {"n_reps": 0},
        {"fdp_samples": np.array([0.0])},
        {"fdp_samples": np.array([np.nan, 0.2])},
        {"power_samples": np.array([0.4, np.inf])},
        {"fdp_samples": np.array([-0.1, 0.2])},
        {"power_samples": np.array([0.4, 1.1])},
        {"fdr": 1.1},
        {"power": -0.1},
        {"fwer": np.nan},
        {"fdr": 0.0},
        {"power": 0.4},
        {"fwer": 0.0},
        {"mean_rejections": -1.0},
    ],
)
def test_simulation_result_rejects_invalid_invariants(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _valid_simulation(**overrides)


def test_simulation_result_metric_contracts() -> None:
    single = _valid_simulation(
        n_reps=1,
        fdr=0.0,
        fwer=0.0,
        fdp_samples=np.array([0.0]),
        power_samples=np.array([0.5]),
    )
    assert single.mc_stderr("fdr") == 0.0
    assert single.mc_stderr("fwer") == 0.0
    with pytest.raises(ValueError, match="metric"):
        single.mc_stderr("unknown")
    single.fwer_samples = None
    assert single.mc_stderr("fwer") >= 0.0
    with pytest.raises(ValueError):
        _valid_simulation(fwer_samples=np.array([0.5, 0.0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"m": True, "pi0": 0.5, "effect": 1.0},
        {"m": 2.5, "pi0": 0.5, "effect": 1.0},
        {"m": -1, "pi0": 0.5, "effect": 1.0},
        {"m": 2, "pi0": np.nan, "effect": 1.0},
        {"m": 2, "pi0": -0.1, "effect": 1.0},
        {"m": 2, "pi0": 0.5, "effect": np.nan},
        {"m": 2, "pi0": 0.5, "effect": 1.0, "rho": np.nan},
    ],
)
def test_two_groups_config_rejects_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TwoGroupsConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"m": 3, "pi0": 0.5, "effect": 1.0, "dependence_kind": "unknown"},
        {"m": 1, "pi0": 0.5, "effect": 1.0, "dependence_kind": "equicorrelated", "rho": 0.1},
        {"m": 3, "pi0": 0.5, "effect": 1.0, "dependence_kind": "equicorrelated", "rho": -0.6},
        {"m": 3, "pi0": 0.5, "effect": 1.0, "dependence_kind": "block", "block_size": True},
        {"m": 3, "pi0": 0.5, "effect": 1.0, "dependence_kind": "block", "block_size": 4},
        {"m": 3, "pi0": 0.5, "effect": 1.0, "dependence_kind": "block", "block_size": 1, "rho": 0.1},
        {"m": 3, "pi0": 0.5, "effect": 1.0, "dependence_kind": "block", "block_size": 3, "rho": -0.6},
    ],
)
def test_two_groups_dependence_branches_reject_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TwoGroupsConfig(**kwargs)


def test_two_groups_valid_boundary_dependence_configs() -> None:
    assert TwoGroupsConfig(m=1, pi0=0.5, effect=1.0, dependence_kind="equicorrelated").m0 == 0
    assert TwoGroupsConfig(m=3, pi0=0.5, effect=1.0, dependence_kind="block", block_size=1).m1 == 1


def test_active_inference_rejects_non_numeric_and_non_integer_observations() -> None:
    config = AITestConfig()
    with pytest.raises(ValueError, match="numeric"):
        run_ai_test(np.asarray(["high"]), config)
    with pytest.raises(ValueError, match="integer"):
        run_ai_test(np.asarray([0.5]), config)
    with pytest.raises(ValueError, match="integer"):
        run_ai_test(np.asarray([np.nan]), config)


def test_active_inference_single_replication_precision_path() -> None:
    result = evaluate_ai_regime(2, 0.5, AITestConfig(horizon=1), 1, seed=3)
    precision = cast(dict[str, Any], result["mc_precision"])
    assert precision["posterior_threshold_fdr"]["standard_error"] == 0.0


@pytest.mark.parametrize(
    "args",
    [
        (1.5, 6, 0.8),
        (1, 2.5, 0.8),
        (1, -1, 0.8),
        (-1, 6, 0.8),
    ],
)
def test_evidence_pvalue_rejects_non_integer_or_invalid_counts(
    args: tuple[int | float, ...],
) -> None:
    with pytest.raises(ValueError):
        evidence_pvalue(cast(int, args[0]), cast(int, args[1]), cast(float, args[2]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"m": True, "pi0": 0.5, "n_reps": 1},
        {"m": 0, "pi0": 0.5, "n_reps": 1},
        {"m": 2, "pi0": np.nan, "n_reps": 1},
        {"m": 2, "pi0": 0.5, "n_reps": 0},
        {"m": 2, "pi0": 0.5, "n_reps": 1.5},
        {"m": 2, "pi0": 0.5, "n_reps": 1, "alpha": 0.0},
    ],
)
def test_ai_regime_rejects_invalid_experiment_contracts(
    kwargs: dict[str, object],
) -> None:
    base: dict[str, Any] = {
        "m": 2,
        "pi0": 0.5,
        "config": AITestConfig(horizon=1),
        "n_reps": 1,
        "alpha": 0.05,
        "seed": 0,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        evaluate_ai_regime(**base)


def test_generate_pvalues_defensive_validation_after_config_mutation() -> None:
    config = TwoGroupsConfig(m=4, pi0=0.5, effect=1.0)
    config.rho = -0.1
    with pytest.raises(ValueError, match="rho"):
        generate_pvalues(config, np.random.default_rng(0))
    config.rho = 0.0
    config.sided = 3
    with pytest.raises(ValueError, match="sided"):
        generate_pvalues(config, np.random.default_rng(0))
