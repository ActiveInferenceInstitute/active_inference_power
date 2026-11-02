"""Tests for ``theory.py`` — closed-form multiple-testing theory.

No mocks. All cross-checks use real numpy computations, an independent
scipy implementation of BH (``scipy.stats.false_discovery_control``), and
seeded Monte-Carlo experiments with ``np.random.default_rng``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import false_discovery_control

from active_inference_power.theory import (
    bh_fdr_bound,
    bh_fdr_bound_from_counts,
    bh_global_null_fdr,
    bh_threshold,
    bonferroni_fwer_bound,
    by_constant,
    expected_false_rejections_global_null,
    is_monotone_step_up,
    sidak_fwer,
    simes_global_pvalue,
)

# ---------------------------------------------------------------------------
# by_constant
# ---------------------------------------------------------------------------


class TestByConstant:
    def test_pinned_values(self) -> None:
        """Contract pins: c(1) == 1 and c(4) == 25/12."""
        assert by_constant(1) == 1.0
        assert by_constant(4) == pytest.approx(25.0 / 12.0, abs=1e-12)

    def test_m_two_exact(self) -> None:
        assert by_constant(2) == 1.5

    def test_m_zero_empty_sum(self) -> None:
        assert by_constant(0) == 0.0

    def test_strictly_increasing_in_m(self) -> None:
        values = [by_constant(m) for m in range(1, 20)]
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_large_m_matches_log_plus_gamma(self) -> None:
        """c(m) ~ ln(m) + Euler-Mascheroni for large m."""
        m = 100_000
        gamma = 0.5772156649015329
        assert by_constant(m) == pytest.approx(np.log(m) + gamma, abs=1e-4)

    def test_negative_m_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            by_constant(-1)


# ---------------------------------------------------------------------------
# bh_threshold
# ---------------------------------------------------------------------------


class TestBhThreshold:
    def test_hand_computed_six_pvalues(self) -> None:
        """Hand-worked BH on 6 p-values at alpha=0.05.

        Sorted p: .001 .008 .039 .041 .042 .060
        Criteria (k/6)*.05: .00833 .01667 .025 .0333 .04167 .05
        Passing ranks: k=1 (.001) and k=2 (.008); k=3..6 all fail.
        Largest passing k = 2 -> threshold = 0.008.
        """
        p = np.array([0.041, 0.008, 0.039, 0.001, 0.06, 0.042])  # unsorted
        assert bh_threshold(p, alpha=0.05) == pytest.approx(0.008)

    def test_no_rejections_returns_zero(self) -> None:
        p = np.array([0.5, 0.7, 0.9])
        assert bh_threshold(p, alpha=0.05) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert bh_threshold(np.array([]), alpha=0.05) == 0.0

    def test_m_one_reject(self) -> None:
        assert bh_threshold(np.array([0.01]), alpha=0.05) == pytest.approx(0.01)

    def test_m_one_no_reject(self) -> None:
        assert bh_threshold(np.array([0.2]), alpha=0.05) == 0.0

    def test_p_zero(self) -> None:
        """p == 0 always passes rank 1; a lone zero yields threshold 0.0."""
        p = np.array([0.0, 0.5])
        assert bh_threshold(p, alpha=0.05) == 0.0
        # ... and rejecting p <= threshold still rejects exactly the zero.
        assert np.array_equal(p <= bh_threshold(p, alpha=0.05), [True, False])

    def test_p_one_never_rejected_alone(self) -> None:
        assert bh_threshold(np.array([1.0]), alpha=0.05) == 0.0

    def test_all_ones_no_rejection(self) -> None:
        assert bh_threshold(np.ones(5), alpha=0.05) == 0.0

    def test_all_rejected_when_all_tiny(self) -> None:
        p = np.array([0.001, 0.002, 0.003, 0.004])
        # k=4: 0.004 <= (4/4)*0.05 -> threshold is the max p.
        assert bh_threshold(p, alpha=0.05) == pytest.approx(0.004)

    def test_step_up_revives_smaller_ranks(self) -> None:
        """Step-up: a passing large rank rejects earlier failing ranks.

        p = [.04, .045, .049], alpha=.05: k=1 fails (.04 > .0167),
        k=2 fails (.045 > .0333), k=3 passes (.049 <= .05) -> all rejected.
        """
        p = np.array([0.04, 0.045, 0.049])
        assert bh_threshold(p, alpha=0.05) == pytest.approx(0.049)

    def test_matches_scipy_bh_rejection_set(self) -> None:
        """p <= bh_threshold must equal scipy's BH adjusted <= alpha."""
        rng = np.random.default_rng(42)
        for _ in range(25):
            p = rng.uniform(size=40)
            p[:8] = rng.uniform(0, 0.01, size=8)  # plant signals
            for alpha in (0.01, 0.05, 0.2):
                ours = p <= bh_threshold(p, alpha=alpha)
                scipy_rej = false_discovery_control(p, method="bh") <= alpha
                assert np.array_equal(ours, scipy_rej)

    def test_bad_alpha_raises(self) -> None:
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="alpha"):
                bh_threshold(np.array([0.01]), alpha=bad)

    def test_bad_pvalues_raise(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            bh_threshold(np.array([0.1, 1.5]))
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            bh_threshold(np.array([-0.2, 0.1]))
        with pytest.raises(ValueError, match="NaN"):
            bh_threshold(np.array([0.1, np.nan]))
        with pytest.raises(ValueError, match="1-D"):
            bh_threshold(np.array([[0.1, 0.2]]))

    def test_eps_slack_clipped_not_rejected(self) -> None:
        """Values within PVAL_EPS of the boundary are clipped, not errors."""
        assert bh_threshold(np.array([-1e-16, 0.5]), alpha=0.05) == 0.0


# ---------------------------------------------------------------------------
# simes_global_pvalue
# ---------------------------------------------------------------------------


class TestSimesGlobalPvalue:
    def test_hand_computed(self) -> None:
        """min(3*.01/1, 3*.02/2, 3*.9/3) = min(.03, .03, .9) = .03."""
        p = np.array([0.9, 0.01, 0.02])  # unsorted
        assert simes_global_pvalue(p) == pytest.approx(0.03)

    def test_m_one_is_identity(self) -> None:
        assert simes_global_pvalue(np.array([0.37])) == pytest.approx(0.37)

    def test_empty_returns_one(self) -> None:
        assert simes_global_pvalue(np.array([])) == 1.0

    def test_p_zero_gives_zero(self) -> None:
        assert simes_global_pvalue(np.array([0.0, 0.8, 0.9])) == 0.0

    def test_all_ones_gives_one(self) -> None:
        assert simes_global_pvalue(np.ones(4)) == pytest.approx(1.0)

    def test_never_exceeds_one(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(50):
            p = rng.uniform(size=rng.integers(1, 30))
            assert 0.0 <= simes_global_pvalue(p) <= 1.0

    def test_simes_le_min_bh_adjusted(self) -> None:
        """Contract pin: Simes <= min(BH-adjusted p-value).

        Independent BH via scipy.stats.false_discovery_control. For a
        nonempty family they are in fact equal (min BH q = min_i (m/i)p_(i),
        which is <= 1 because the i=m term is p_(m) <= 1).
        """
        rng = np.random.default_rng(123)
        for _ in range(30):
            p = rng.uniform(size=rng.integers(1, 50))
            simes = simes_global_pvalue(p)
            min_bh_adj = float(np.min(false_discovery_control(p, method="bh")))
            assert simes <= min_bh_adj + 1e-12
            assert simes == pytest.approx(min_bh_adj, abs=1e-12)

    def test_global_null_validity_monte_carlo(self) -> None:
        """Under the global null, P(Simes <= alpha) == alpha (independence).

        Seeded MC: 4000 reps of m=20 uniform p-values; the exceedance
        proportion must sit within ~3 binomial stderr of alpha=0.05.
        """
        rng = np.random.default_rng(2026)
        alpha, n_reps, m = 0.05, 4000, 20
        hits = sum(
            simes_global_pvalue(rng.uniform(size=m)) <= alpha for _ in range(n_reps)
        )
        stderr = np.sqrt(alpha * (1 - alpha) / n_reps)
        assert abs(hits / n_reps - alpha) < 3.5 * stderr

    def test_bad_pvalues_raise(self) -> None:
        with pytest.raises(ValueError):
            simes_global_pvalue(np.array([0.1, 2.0]))


# ---------------------------------------------------------------------------
# bonferroni_fwer_bound / sidak_fwer
# ---------------------------------------------------------------------------


class TestFwerBounds:
    def test_bonferroni_pinned(self) -> None:
        assert bonferroni_fwer_bound(0.05, 10) == pytest.approx(0.5)
        assert bonferroni_fwer_bound(0.01, 3) == pytest.approx(0.03)

    def test_bonferroni_caps_at_one(self) -> None:
        assert bonferroni_fwer_bound(0.05, 30) == 1.0
        assert bonferroni_fwer_bound(0.5, 1000) == 1.0

    def test_bonferroni_m_zero(self) -> None:
        assert bonferroni_fwer_bound(0.05, 0) == 0.0

    def test_sidak_pinned(self) -> None:
        # 1 - 0.95**10, computed independently.
        assert sidak_fwer(0.05, 10) == pytest.approx(0.4012630607616213, abs=1e-15)

    def test_sidak_m_one_is_alpha(self) -> None:
        assert sidak_fwer(0.05, 1) == pytest.approx(0.05)

    def test_sidak_m_zero(self) -> None:
        assert sidak_fwer(0.05, 0) == 0.0

    def test_sidak_le_bonferroni_grid(self) -> None:
        """Contract pin: sidak_fwer <= bonferroni_fwer_bound everywhere."""
        for alpha in (0.001, 0.01, 0.05, 0.2, 0.9):
            for m in (0, 1, 2, 5, 10, 100, 1000):
                assert sidak_fwer(alpha, m) <= bonferroni_fwer_bound(alpha, m) + 1e-15

    def test_sidak_monotone_in_m(self) -> None:
        vals = [sidak_fwer(0.05, m) for m in range(0, 50)]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_sidak_approaches_one(self) -> None:
        assert sidak_fwer(0.05, 10_000) == pytest.approx(1.0, abs=1e-12)

    def test_negative_m_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            bonferroni_fwer_bound(0.05, -2)
        with pytest.raises(ValueError, match="non-negative"):
            sidak_fwer(0.05, -2)

    def test_bad_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            bonferroni_fwer_bound(0.0, 5)
        with pytest.raises(ValueError, match="alpha"):
            sidak_fwer(1.0, 5)

    def test_sidak_monte_carlo_global_null(self) -> None:
        """MC check: FWER of m independent alpha-level tests == sidak_fwer.

        m=5, alpha=0.1 -> analytic 1 - 0.9^5 = 0.40951. Seeded 5000-rep MC
        proportion of families with >=1 p <= alpha must agree within 3.5 SE.
        """
        rng = np.random.default_rng(11)
        m, alpha, n_reps = 5, 0.1, 5000
        analytic = sidak_fwer(alpha, m)
        assert analytic == pytest.approx(1.0 - 0.9**5, abs=1e-15)
        hits = int(np.sum(np.any(rng.uniform(size=(n_reps, m)) <= alpha, axis=1)))
        stderr = np.sqrt(analytic * (1 - analytic) / n_reps)
        assert abs(hits / n_reps - analytic) < 3.5 * stderr


# ---------------------------------------------------------------------------
# bh_fdr_bound
# ---------------------------------------------------------------------------


class TestBhFdrBound:
    def test_pinned(self) -> None:
        assert bh_fdr_bound(0.8, 0.05) == pytest.approx(0.04)
        assert bh_fdr_bound(1.0, 0.05) == pytest.approx(0.05)

    def test_pi0_zero(self) -> None:
        assert bh_fdr_bound(0.0, 0.05) == 0.0

    def test_monotone_in_both_arguments(self) -> None:
        assert bh_fdr_bound(0.5, 0.05) < bh_fdr_bound(0.9, 0.05)
        assert bh_fdr_bound(0.8, 0.01) < bh_fdr_bound(0.8, 0.1)

    def test_bad_pi0_raises(self) -> None:
        for bad in (-0.1, 1.1):
            with pytest.raises(ValueError, match="pi0"):
                bh_fdr_bound(bad, 0.05)

    def test_bad_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            bh_fdr_bound(0.8, 0.0)


# ---------------------------------------------------------------------------
# expected_false_rejections_global_null
# ---------------------------------------------------------------------------


class TestExpectedFalseRejections:
    def test_bonferroni_is_alpha(self) -> None:
        for m in (1, 10, 500):
            got = expected_false_rejections_global_null(0.05, m, "bonferroni")
            assert got == pytest.approx(0.05)

    def test_benjamini_hochberg_expected_count_matches_two_test_oracle(self) -> None:
        alpha = 0.05
        # For m=2, E[R] = P(R >= 1) + P(R >= 2) = alpha + alpha**2.
        assert expected_false_rejections_global_null(
            alpha, 2, "benjamini_hochberg"
        ) == pytest.approx(
            alpha + alpha**2
        )
        assert (
            expected_false_rejections_global_null(alpha, 100, "benjamini_hochberg")
            > alpha
        )

    def test_bh_global_null_fdr_is_distinct_from_expected_count(self) -> None:
        assert bh_global_null_fdr(0.05, 100) == pytest.approx(0.05)
        assert bh_global_null_fdr(0.05, 0) == 0.0

    def test_bh_global_null_fdr_rejects_negative_family_size(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            bh_global_null_fdr(0.05, -1)

    def test_sidak_pinned(self) -> None:
        # 10 * (1 - 0.95**0.1) = 0.05116196891823536 (independent evaluation).
        got = expected_false_rejections_global_null(0.05, 10, "sidak")
        assert got == pytest.approx(0.05116196891823536, abs=1e-14)

    def test_sidak_exceeds_alpha(self) -> None:
        """Bernoulli: (1-a)^(1/m) <= 1 - a/m, so the sidak count >= alpha."""
        for m in (2, 5, 50):
            assert expected_false_rejections_global_null(0.05, m, "sidak") > 0.05

    def test_sidak_m_one_is_alpha(self) -> None:
        got = expected_false_rejections_global_null(0.05, 1, "sidak")
        assert got == pytest.approx(0.05)

    def test_m_zero_all_methods(self) -> None:
        for method in ("bonferroni", "benjamini_hochberg", "sidak"):
            assert expected_false_rejections_global_null(0.05, 0, method) == 0.0

    def test_method_rejects_noncanonical_spelling(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            expected_false_rejections_global_null(0.05, 10, "Bonferroni")

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            expected_false_rejections_global_null(0.05, 10, "holm")

    def test_short_bh_alias_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            expected_false_rejections_global_null(0.05, 10, "bh")

    def test_negative_m_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            expected_false_rejections_global_null(0.05, -1)

    def test_bad_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            expected_false_rejections_global_null(-0.05, 10)

    def test_bonferroni_monte_carlo(self) -> None:
        """MC check: mean #{p <= alpha/m} over uniform nulls ~= alpha.

        m=50, alpha=0.05, 3000 seeded reps; the per-rep count is
        Binomial(m, alpha/m) with sd ~= sqrt(alpha) = 0.224.
        """
        rng = np.random.default_rng(99)
        m, alpha, n_reps = 50, 0.05, 3000
        counts = np.sum(rng.uniform(size=(n_reps, m)) <= alpha / m, axis=1)
        stderr = np.sqrt(alpha) / np.sqrt(n_reps)
        analytic = expected_false_rejections_global_null(alpha, m, "bonferroni")
        assert abs(float(np.mean(counts)) - analytic) < 3.5 * stderr


# ---------------------------------------------------------------------------
# is_monotone_step_up
# ---------------------------------------------------------------------------


class TestIsMonotoneStepUp:
    def test_monotone_mask_true(self) -> None:
        p = np.array([0.01, 0.2, 0.03, 0.8])
        rejected = np.array([True, False, True, False])
        assert is_monotone_step_up(p, rejected) is True

    def test_non_monotone_mask_detected(self) -> None:
        """Contract pin: skipping a smaller p while rejecting a larger one."""
        p = np.array([0.01, 0.02, 0.03])
        rejected = np.array([True, False, True])
        assert is_monotone_step_up(p, rejected) is False

    def test_tie_split_is_non_monotone(self) -> None:
        p = np.array([0.02, 0.02, 0.5])
        rejected = np.array([True, False, False])
        assert is_monotone_step_up(p, rejected) is False

    def test_tie_both_rejected_is_monotone(self) -> None:
        p = np.array([0.02, 0.02, 0.5])
        rejected = np.array([True, True, False])
        assert is_monotone_step_up(p, rejected) is True

    def test_all_false_vacuously_true(self) -> None:
        assert is_monotone_step_up(np.array([0.1, 0.9]), np.array([False, False]))

    def test_all_true_is_monotone(self) -> None:
        assert is_monotone_step_up(np.array([0.1, 0.9]), np.array([True, True]))

    def test_empty_is_true(self) -> None:
        assert is_monotone_step_up(np.array([]), np.array([], dtype=bool)) is True

    def test_m_one(self) -> None:
        assert is_monotone_step_up(np.array([0.5]), np.array([True])) is True
        assert is_monotone_step_up(np.array([0.5]), np.array([False])) is True

    def test_bh_masks_are_always_monotone(self) -> None:
        """Structural identity: every BH rejection set passes the check."""
        rng = np.random.default_rng(314)
        for _ in range(20):
            p = rng.uniform(size=30)
            p[:5] = rng.uniform(0, 0.005, size=5)
            rejected = p <= bh_threshold(p, alpha=0.1)
            assert is_monotone_step_up(p, rejected) is True

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="matching"):
            is_monotone_step_up(np.array([0.1, 0.2]), np.array([True]))

    def test_2d_mask_raises(self) -> None:
        with pytest.raises(ValueError, match="matching"):
            is_monotone_step_up(np.array([0.1, 0.2]), np.array([[True, False]]))

    def test_bad_pvalues_raise(self) -> None:
        with pytest.raises(ValueError):
            is_monotone_step_up(np.array([0.1, 1.7]), np.array([True, False]))


def test_bh_fdr_bound_from_counts_validates_realized_counts() -> None:
    """The finite-family helper fails closed on non-count arguments."""
    assert bh_fdr_bound_from_counts(2, 4, 0.05) == pytest.approx(0.025)
    assert bh_fdr_bound_from_counts(0, 4, 0.05) == 0.0
    with pytest.raises(ValueError, match="m must be a positive integer"):
        bh_fdr_bound_from_counts(0, 0, 0.05)
    with pytest.raises(ValueError, match=r"m0 must be an integer in \[0, m\]"):
        bh_fdr_bound_from_counts(5, 4, 0.05)
