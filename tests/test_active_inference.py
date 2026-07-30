"""Real (no-mock) tests for the active-inference sequential test module.

Every stochastic test fixes a seed. The real ``pymdp.agent.Agent`` is exercised
directly (no mocks). Correctness is pinned against a closed-form sequential-Bayes
oracle, not merely against array shapes.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom

from active_inference_power.active_inference import (
    AITestConfig,
    AITestResult,
    _run_ai_batch,
    ai_test_decision,
    build_generative_model,
    evaluate_ai_regime,
    evidence_pvalue,
    generate_evidence_stream,
    run_ai_test,
)

pytestmark = pytest.mark.requires_pymdp


# --------------------------------------------------------------------------- #
# Closed-form oracle: exact sequential Bayesian update the agent must match.
# --------------------------------------------------------------------------- #
def analytic_signal(
    n_high: int, n_low: int, reliability: float, prior_signal: float
) -> float:
    """Posterior belief in H1 after ``n_high`` high and ``n_low`` low obs."""
    if prior_signal <= 0.0:
        return 0.0
    if prior_signal >= 1.0:
        return 1.0
    prior_odds = prior_signal / (1.0 - prior_signal)
    lr = reliability / (1.0 - reliability)
    odds = prior_odds * lr ** (n_high - n_low)
    return float(odds / (1.0 + odds))


# --------------------------------------------------------------------------- #
# AITestConfig validation
# --------------------------------------------------------------------------- #
def test_config_defaults():
    cfg = AITestConfig()
    assert cfg.reliability == 0.8
    assert cfg.threshold == 0.9
    assert cfg.horizon == 20
    assert cfg.prior_signal == 0.5


def test_config_accepts_boundary_prior_and_threshold():
    # threshold and prior may sit exactly on [0, 1] boundaries.
    cfg = AITestConfig(threshold=0.0, prior_signal=1.0, horizon=0)
    assert cfg.threshold == 0.0
    assert cfg.prior_signal == 1.0
    cfg2 = AITestConfig(threshold=1.0, prior_signal=0.0)
    assert cfg2.threshold == 1.0
    assert cfg2.prior_signal == 0.0


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_config_rejects_out_of_range_reliability(bad):
    with pytest.raises(ValueError, match="reliability"):
        AITestConfig(reliability=bad)


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_config_rejects_out_of_range_threshold(bad):
    with pytest.raises(ValueError, match="threshold"):
        AITestConfig(threshold=bad)


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_config_rejects_out_of_range_prior(bad):
    with pytest.raises(ValueError, match="prior_signal"):
        AITestConfig(prior_signal=bad)


def test_config_rejects_negative_horizon():
    with pytest.raises(ValueError, match="horizon"):
        AITestConfig(horizon=-1)


# --------------------------------------------------------------------------- #
# build_generative_model
# --------------------------------------------------------------------------- #
def test_build_generative_model_structure():
    cfg = AITestConfig(reliability=0.75, prior_signal=0.3)
    A, B, C, D = build_generative_model(cfg)

    # Likelihood: rows = obs {low, high}, cols = state {H0, H1}.
    expected_A = np.array([[0.75, 0.25], [0.25, 0.75]])
    np.testing.assert_allclose(A[0], expected_A)
    # Columns are valid conditional distributions.
    np.testing.assert_allclose(A[0].sum(axis=0), [1.0, 1.0])

    # Transition: identity, single action, shape (2, 2, 1).
    assert B[0].shape == (2, 2, 1)
    np.testing.assert_allclose(B[0][:, :, 0], np.eye(2))

    # Flat preferences.
    np.testing.assert_allclose(C[0], np.zeros(2))

    # Prior over {H0, H1}.
    np.testing.assert_allclose(D[0], [0.7, 0.3])


# --------------------------------------------------------------------------- #
# run_ai_test — correctness against the analytic oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "stream,reliability,prior",
    [
        ([1, 1, 1, 1, 1, 1], 0.8, 0.5),
        ([0, 0, 0, 0, 0, 0], 0.8, 0.5),
        ([1, 0, 1, 0, 1, 0], 0.8, 0.5),
        ([1, 1, 0], 0.7, 0.3),
        ([1, 1, 1], 0.3, 0.5),  # anti-correlated channel: high pushes toward H0
        ([1, 0, 1, 1, 0], 0.9, 0.6),
    ],
)
def test_run_ai_test_matches_analytic(stream, reliability, prior):
    cfg = AITestConfig(reliability=reliability, prior_signal=prior)
    result = run_ai_test(np.asarray(stream), cfg)

    n_high = int(np.sum(np.asarray(stream) == 1))
    n_low = len(stream) - n_high
    expected = analytic_signal(n_high, n_low, reliability, prior)

    assert result.final_belief_signal == pytest.approx(expected, abs=1e-9)


def test_run_ai_test_result_type_and_shapes():
    cfg = AITestConfig()
    result = run_ai_test(np.array([1, 0, 1]), cfg)
    assert isinstance(result, AITestResult)
    # trajectory has one entry per step plus the prior.
    assert result.belief_trajectory.shape == (4,)
    assert result.n_steps == 3
    assert result.belief_trajectory[0] == pytest.approx(0.5)  # prior
    assert result.final_belief_signal == pytest.approx(result.belief_trajectory[-1])


def test_belief_trajectory_bounded_in_unit_interval():
    cfg = AITestConfig(reliability=0.85, prior_signal=0.4)
    rng = np.random.default_rng(11)
    stream = rng.integers(0, 2, size=30)
    result = run_ai_test(stream, cfg)
    assert np.all(result.belief_trajectory >= 0.0)
    assert np.all(result.belief_trajectory <= 1.0)


def test_all_high_stream_drives_belief_to_one():
    cfg = AITestConfig(reliability=0.8, prior_signal=0.5, horizon=40)
    result = run_ai_test(np.ones(40, dtype=int), cfg)
    assert result.final_belief_signal > 0.999999


def test_all_low_stream_drives_belief_to_zero():
    cfg = AITestConfig(reliability=0.8, prior_signal=0.5, horizon=40)
    result = run_ai_test(np.zeros(40, dtype=int), cfg)
    assert result.final_belief_signal < 1e-6


def test_single_observation_exact_values():
    cfg = AITestConfig(reliability=0.8, prior_signal=0.5)
    assert run_ai_test(np.array([1]), cfg).final_belief_signal == pytest.approx(0.8)
    assert run_ai_test(np.array([0]), cfg).final_belief_signal == pytest.approx(0.2)


def test_balanced_stream_returns_to_prior():
    # Equal highs and lows with prior 0.5 => posterior odds unchanged => 0.5.
    cfg = AITestConfig(reliability=0.8, prior_signal=0.5)
    result = run_ai_test(np.array([1, 1, 1, 0, 0, 0]), cfg)
    assert result.final_belief_signal == pytest.approx(0.5, abs=1e-12)


def test_final_belief_monotone_nondecreasing_in_high_count():
    # Fixed horizon; increase the number of highs -> final belief must not drop.
    cfg = AITestConfig(reliability=0.8, prior_signal=0.5)
    horizon = 8
    finals = []
    for n_high in range(horizon + 1):
        stream = np.array([1] * n_high + [0] * (horizon - n_high))
        finals.append(run_ai_test(stream, cfg).final_belief_signal)
    diffs = np.diff(finals)
    assert np.all(diffs >= -1e-12)  # non-decreasing
    assert finals[-1] > finals[0]  # strictly informative overall


def test_higher_reliability_accumulates_faster():
    # Same evidence (k highs); a more reliable channel yields higher belief.
    stream = np.ones(5, dtype=int)
    weak = run_ai_test(stream, AITestConfig(reliability=0.6, prior_signal=0.5))
    strong = run_ai_test(stream, AITestConfig(reliability=0.9, prior_signal=0.5))
    assert strong.final_belief_signal > weak.final_belief_signal


def test_uninformative_channel_keeps_prior():
    # reliability == 0.5 => likelihood ratio 1 => belief never moves from prior.
    cfg = AITestConfig(reliability=0.5, prior_signal=0.37)
    result = run_ai_test(np.array([1, 0, 1, 1, 0, 1]), cfg)
    np.testing.assert_allclose(result.belief_trajectory, 0.37, atol=1e-12)


def test_prior_shifts_trajectory_start():
    cfg = AITestConfig(reliability=0.8, prior_signal=0.2)
    result = run_ai_test(np.array([1, 1]), cfg)
    assert result.belief_trajectory[0] == pytest.approx(0.2)
    # analytic: odds0 = 0.25, LR^2 = 16 -> odds = 4 -> belief = 0.8
    assert result.final_belief_signal == pytest.approx(0.8, abs=1e-9)


def test_empty_stream_stays_at_prior():
    cfg = AITestConfig(reliability=0.8, prior_signal=0.5, threshold=0.9)
    result = run_ai_test(np.array([], dtype=int), cfg)
    assert result.n_steps == 0
    assert result.belief_trajectory.shape == (1,)
    assert result.final_belief_signal == pytest.approx(0.5)
    assert result.decision is False  # 0.5 < 0.9


def test_batched_real_agent_rejects_malformed_stream_matrices():
    cfg = AITestConfig(reliability=0.8, horizon=2)
    with pytest.raises(ValueError, match="two-dimensional"):
        _run_ai_batch(np.array([1, 0]), cfg.prior_signal, cfg)
    with pytest.raises(ValueError, match="at least one"):
        _run_ai_batch(np.empty((0, 2), dtype=int), cfg.prior_signal, cfg)
    with pytest.raises(ValueError, match="binary"):
        _run_ai_batch(np.array([[0, 2]], dtype=int), cfg.prior_signal, cfg)
    prior_only = _run_ai_batch(np.empty((2, 0), dtype=int), cfg.prior_signal, cfg)
    assert prior_only.shape == (2, 2)
    assert np.allclose(prior_only, [[0.5, 0.5], [0.5, 0.5]])


def test_run_ai_test_accepts_python_list():
    cfg = AITestConfig()
    result = run_ai_test([1, 1, 1], cfg)
    assert result.n_steps == 3


def test_invalid_observation_index_raises():
    cfg = AITestConfig()
    with pytest.raises(ValueError, match="0 .low. or 1 .high."):
        run_ai_test(np.array([1, 2, 0]), cfg)


# --------------------------------------------------------------------------- #
# Decision rule: reject iff final belief >= threshold
# --------------------------------------------------------------------------- #
def test_decision_true_when_above_threshold():
    # Two highs at r=0.8, prior 0.5 => 0.9412.
    stream = np.array([1, 1])
    assert run_ai_test(stream, AITestConfig(reliability=0.8, threshold=0.94)).decision
    assert not run_ai_test(
        stream, AITestConfig(reliability=0.8, threshold=0.95)
    ).decision


def test_decision_boundary_is_inclusive():
    # Empty stream => final belief is exactly the prior (no agent float drift);
    # a threshold equal to that value must reject (>= is inclusive).
    cfg = AITestConfig(reliability=0.8, prior_signal=0.5, threshold=0.5)
    result = run_ai_test(np.array([], dtype=int), cfg)
    assert result.final_belief_signal == 0.5  # exact
    assert result.decision is True


# --------------------------------------------------------------------------- #
# generate_evidence_stream
# --------------------------------------------------------------------------- #
def test_generate_stream_shape_and_dtype():
    cfg = AITestConfig(horizon=15)
    rng = np.random.default_rng(0)
    stream = generate_evidence_stream(True, cfg, rng)
    assert stream.shape == (15,)
    assert set(np.unique(stream)).issubset({0, 1})


def test_generate_stream_horizon_zero_is_empty():
    cfg = AITestConfig(horizon=0)
    rng = np.random.default_rng(0)
    stream = generate_evidence_stream(False, cfg, rng)
    assert stream.shape == (0,)


def test_generate_stream_reproducible_with_seed():
    cfg = AITestConfig(horizon=50, reliability=0.8)
    s1 = generate_evidence_stream(True, cfg, np.random.default_rng(123))
    s2 = generate_evidence_stream(True, cfg, np.random.default_rng(123))
    np.testing.assert_array_equal(s1, s2)


def test_generate_stream_signal_high_rate():
    # is_signal True => P(high) = reliability.
    cfg = AITestConfig(horizon=20000, reliability=0.8)
    rng = np.random.default_rng(7)
    stream = generate_evidence_stream(True, cfg, rng)
    assert stream.mean() == pytest.approx(0.8, abs=0.02)


def test_generate_stream_null_high_rate():
    # is_signal False => P(high) = 1 - reliability.
    cfg = AITestConfig(horizon=20000, reliability=0.8)
    rng = np.random.default_rng(7)
    stream = generate_evidence_stream(False, cfg, rng)
    assert stream.mean() == pytest.approx(0.2, abs=0.02)


# --------------------------------------------------------------------------- #
# ai_test_decision — end to end
# --------------------------------------------------------------------------- #
def test_ai_test_decision_returns_bool():
    cfg = AITestConfig()
    out = ai_test_decision(True, cfg, np.random.default_rng(0))
    assert isinstance(out, bool)


def test_ai_test_decision_power_and_type_one_error():
    # Over many worlds: signal worlds are (almost always) rejected; null worlds
    # (almost never) rejected. A real end-to-end operating-characteristics check.
    cfg = AITestConfig(reliability=0.8, threshold=0.9, horizon=20, prior_signal=0.5)
    n = 200
    rng = np.random.default_rng(2024)
    signal_rejects = sum(ai_test_decision(True, cfg, rng) for _ in range(n))
    null_rejects = sum(ai_test_decision(False, cfg, rng) for _ in range(n))
    assert signal_rejects / n > 0.9  # high power
    assert null_rejects / n < 0.1  # low type-I error


def test_evidence_pvalue_is_calibrated_and_validates_counts():
    assert evidence_pvalue(6, 6, 0.8) < evidence_pvalue(3, 6, 0.8)
    assert 0.0 <= evidence_pvalue(0, 6, 0.8) <= 1.0
    with pytest.raises(ValueError):
        evidence_pvalue(7, 6, 0.8)
    with pytest.raises(ValueError):
        evidence_pvalue(1, 6, 1.0)


def test_evidence_pvalue_reverses_tail_for_anti_correlated_channel():
    reliability = 0.3
    # Under H0, high observations have rate 1 - reliability. For r < 0.5,
    # low observations support H1, so the calibrated evidence event is lower-tail.
    assert evidence_pvalue(0, 6, reliability) == pytest.approx(
        binom.cdf(0, 6, 0.7)
    )
    assert evidence_pvalue(0, 6, reliability) < evidence_pvalue(3, 6, reliability)


def test_evaluate_ai_regime_compares_real_agent_policies():
    result = evaluate_ai_regime(
        m=6,
        pi0=0.5,
        config=AITestConfig(reliability=0.8, threshold=0.9, horizon=3),
        n_reps=2,
        alpha=0.05,
        seed=4,
    )
    assert result["config"]["m"] == 6
    for key in (
        "posterior_threshold_fdr",
        "posterior_threshold_power",
        "bh_on_evidence_fdr",
        "bh_on_evidence_power",
    ):
        assert 0.0 <= result[key] <= 1.0
    for key in (
        "posterior_threshold_fdr_stderr",
        "posterior_threshold_power_stderr",
        "bh_on_evidence_fdr_stderr",
        "bh_on_evidence_power_stderr",
    ):
        assert result[key] >= 0.0
