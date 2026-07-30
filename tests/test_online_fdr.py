"""Known invariants for online FDR schedules."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference_power.online_fdr import _gamma_at, default_gamma, e_lond, lord_plus_plus, saffron


def test_gamma_is_positive_and_normalized() -> None:
    gamma = default_gamma(20)
    assert gamma.shape == (20,)
    assert np.all(gamma > 0)
    assert np.sum(gamma) == pytest.approx(1.0)


def test_lord_and_saffron_return_aligned_traces() -> None:
    pvalues = np.array([0.001, 0.8, 0.002, 0.9, 0.003])
    lord = lord_plus_plus(pvalues)
    saff = saffron(pvalues, lambda_=0.5)
    for result in (lord, saff):
        assert result.levels.shape == pvalues.shape
        assert result.rejections.shape == pvalues.shape
        assert np.all((result.levels >= 0.0) & (result.levels <= result.alpha))
        assert np.array_equal(result.rejections, result.pvalues <= result.levels)
        assert result.wealth.shape == (pvalues.size + 1,)
        assert result.wealth[0] == pytest.approx(result.initial_wealth)
        assert np.all(result.wealth >= 0.0)
        payload = result.to_mapping()
        assert payload["accounting_path"] == pytest.approx(result.wealth.tolist())
        assert payload["accounting_kind"] == "procedural_alpha_wealth"
        assert "wealth" not in payload


def test_lord_plus_plus_matches_known_recursion_vector() -> None:
    gamma = [0.5, 0.3, 0.2]
    result = lord_plus_plus([0.01, 0.9, 0.001], alpha=0.1, gamma=gamma, w0=0.05)
    # w0*gamma_t + (alpha-w0)*gamma_{t-tau_1}; tau_1 = 1.
    assert result.levels == pytest.approx([0.025, 0.04, 0.025])
    assert np.array_equal(result.rejections, [True, False, True])
    assert result.wealth == pytest.approx([0.05, 0.075, 0.035, 0.11])


def test_saffron_matches_candidate_indexed_recursion_vector() -> None:
    gamma = [0.5, 0.3, 0.2]
    result = saffron([0.01, 0.9, 0.001], alpha=0.1, gamma=gamma, lambda_=0.5, w0=0.05)
    # The candidate at t=1 advances the initial gamma index at t=2 and t=3;
    # the first-rejection reward is zero at the boundary w0=(1-lambda)*alpha.
    assert result.levels == pytest.approx([0.025, 0.025, 0.015])
    assert np.array_equal(result.candidates, [True, False, True])
    assert np.array_equal(result.rejections, [True, False, True])
    assert result.wealth == pytest.approx([0.05, 0.05, 0.025, 0.075])


def test_saffron_dense_rejections_are_capped_at_the_declared_family_level() -> None:
    """A valid dense path remains serializable under the shared level contract."""
    result = saffron(np.zeros(8), alpha=0.05, lambda_=0.5)

    assert np.array_equal(result.candidates, np.ones(8, dtype=bool))
    assert np.array_equal(result.rejections, np.ones(8, dtype=bool))
    assert np.all((result.levels >= 0.0) & (result.levels <= result.alpha))
    assert np.any(np.isclose(result.levels, result.alpha))


def test_online_fdr_rejects_bad_schedules() -> None:
    with pytest.raises(ValueError):
        lord_plus_plus([0.1, 0.2], gamma=[0.8])
    with pytest.raises(ValueError):
        saffron([0.1], lambda_=1.0)
    with pytest.raises(ValueError):
        default_gamma(2, exponent=1.0)
    with pytest.raises(ValueError):
        lord_plus_plus([0.1, np.nan])
    with pytest.raises(ValueError):
        saffron([0.1], gamma=[1.1], lambda_=0.5)
    with pytest.raises(ValueError):
        default_gamma(0)
    with pytest.raises(ValueError):
        default_gamma(2, exponent=float("nan"))
    with pytest.raises(ValueError):
        lord_plus_plus([0.1], alpha=1.0)
    with pytest.raises(ValueError):
        lord_plus_plus([0.1], gamma=[-0.1])
    with pytest.raises(ValueError):
        lord_plus_plus([0.1, 0.2], gamma=[0.2, 0.8])
    with pytest.raises(ValueError):
        saffron([0.1], gamma=[0.5, 0.2])
    with pytest.raises(ValueError):
        saffron([0.1], lambda_=0.8)
    with pytest.raises(ValueError):
        lord_plus_plus([1.1])
    with pytest.raises(ValueError):
        saffron([0.1], lambda_=0.5, w0=0.026)


def test_nonpositive_gamma_index_contributes_zero() -> None:
    assert _gamma_at(np.array([0.5]), 0) == 0.0


def test_e_lond_known_threshold_trace_and_validation() -> None:
    result = e_lond([100.0, 1.0, 100.0], alpha=0.1, gamma=[0.5, 0.3, 0.2])
    assert result.levels == pytest.approx([0.05, 0.05, 0.03])
    assert np.array_equal(result.rejections, [True, False, True])
    assert result.thresholds[0] == pytest.approx(20.0)
    assert result.cumulative_allocations.shape == (4,)
    assert result.cumulative_allocations == pytest.approx([0.0, 0.05, 0.10, 0.13])
    payload = result.to_mapping()
    assert payload["accounting_path"] == pytest.approx(
        result.cumulative_allocations.tolist()
    )
    assert payload["accounting_kind"] == "cumulative_nominal_level_allocation"
    assert "wealth" not in payload
    assert not hasattr(result, "wealth")


def test_e_lond_recycled_levels_are_not_a_remaining_budget() -> None:
    """Dense rejections can recycle levels after the initial alpha is spent."""
    result = e_lond([100.0, 100.0, 100.0], alpha=0.1, gamma=[0.5, 0.3, 0.2])

    assert result.levels == pytest.approx([0.05, 0.05, 0.05])
    assert np.array_equal(result.rejections, [True, True, True])
    assert result.cumulative_allocations[-1] == pytest.approx(0.15)
    assert result.cumulative_allocations[-1] > result.alpha
    with pytest.raises(ValueError):
        e_lond([1.0], gamma=[0.0])
