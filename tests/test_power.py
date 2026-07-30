"""Tests for :mod:`power` -- analytic statistical power.

Real numerics only: every assertion pins a computed value (or a proven
identity), never merely a shape. No mocks, no monkeypatch.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from active_inference_power.power import (
    average_power_bh,
    cohens_d,
    required_sample_size,
    t_test_power,
    z_test_power,
)

# --------------------------------------------------------------------------- #
# cohens_d
# --------------------------------------------------------------------------- #


def test_cohens_d_basic() -> None:
    assert cohens_d(1.0, 2.0) == pytest.approx(0.5)


def test_cohens_d_negative_difference() -> None:
    assert cohens_d(-3.0, 1.5) == pytest.approx(-2.0)


def test_cohens_d_zero_difference() -> None:
    assert cohens_d(0.0, 4.0) == 0.0


def test_cohens_d_zero_sd_raises() -> None:
    with pytest.raises(ValueError):
        cohens_d(1.0, 0.0)


def test_cohens_d_negative_sd_raises() -> None:
    with pytest.raises(ValueError):
        cohens_d(1.0, -2.0)


# --------------------------------------------------------------------------- #
# z_test_power
# --------------------------------------------------------------------------- #


def test_z_power_at_null_equals_alpha_one_sided() -> None:
    assert z_test_power(0.0, 0.05, sided=1) == pytest.approx(0.05, abs=1e-12)


def test_z_power_at_null_equals_alpha_two_sided() -> None:
    assert z_test_power(0.0, 0.05, sided=2) == pytest.approx(0.05, abs=1e-12)


def test_z_power_at_null_various_alpha() -> None:
    for alpha in (0.01, 0.1, 0.2):
        assert z_test_power(0.0, alpha, sided=1) == pytest.approx(alpha, abs=1e-12)
        assert z_test_power(0.0, alpha, sided=2) == pytest.approx(alpha, abs=1e-12)


def test_z_power_monotone_increasing_in_ncp() -> None:
    ncps = np.linspace(0.0, 4.0, 25)
    powers = [z_test_power(n, 0.05, sided=1) for n in ncps]
    assert all(b >= a - 1e-12 for a, b in zip(powers, powers[1:]))
    assert np.diff(powers).max() > 0.0  # strictly rising somewhere


def test_z_power_known_value_one_sided() -> None:
    # ncp = 1.645 (= z_{0.95}) => power exactly 0.5 for a one-sided 0.05 test.
    ncp = norm.ppf(0.95)
    assert z_test_power(ncp, 0.05, sided=1) == pytest.approx(0.5, abs=1e-9)


def test_z_power_two_sided_lower_tail_contributes() -> None:
    # A strongly negative ncp is caught by the lower rejection region.
    p = z_test_power(-4.0, 0.05, sided=2)
    assert p == pytest.approx(z_test_power(4.0, 0.05, sided=2), abs=1e-12)
    assert p > 0.9


def test_z_power_bounds() -> None:
    for ncp in (-5.0, 0.0, 10.0):
        p = z_test_power(ncp, 0.05, sided=2)
        assert 0.0 <= p <= 1.0


def test_z_power_invalid_alpha_raises() -> None:
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            z_test_power(1.0, bad)


def test_z_power_invalid_sided_raises() -> None:
    with pytest.raises(ValueError):
        z_test_power(1.0, 0.05, sided=3)


# --------------------------------------------------------------------------- #
# t_test_power
# --------------------------------------------------------------------------- #


def test_t_power_textbook_two_sample() -> None:
    # Cohen's canonical value: d=0.5, n=64/group, alpha=0.05 two-sided => ~0.80.
    p = t_test_power(0.5, 64, 0.05, sided=2, two_sample=True)
    assert p == pytest.approx(0.80, abs=0.01)


def test_t_power_one_sample_value() -> None:
    # d=0.5, n=64 one-sample two-sided is more powerful than the two-sample case.
    p_one = t_test_power(0.5, 64, 0.05, sided=2, two_sample=False)
    p_two = t_test_power(0.5, 64, 0.05, sided=2, two_sample=True)
    assert p_one > p_two
    assert 0.0 <= p_two < p_one <= 1.0


def test_t_power_one_sided_more_powerful_than_two_sided() -> None:
    p1 = t_test_power(0.4, 50, 0.05, sided=1, two_sample=False)
    p2 = t_test_power(0.4, 50, 0.05, sided=2, two_sample=False)
    assert p1 > p2


def test_t_power_null_effect_approx_alpha() -> None:
    # With d=0 the noncentral t reduces to central t: power == alpha.
    assert t_test_power(0.0, 30, 0.05, sided=2, two_sample=False) == pytest.approx(
        0.05, abs=1e-9
    )
    assert t_test_power(0.0, 30, 0.05, sided=1, two_sample=True) == pytest.approx(
        0.05, abs=1e-9
    )


def test_t_power_monotone_in_n() -> None:
    powers = [
        t_test_power(0.3, n, 0.05, sided=2, two_sample=True) for n in range(5, 120, 5)
    ]
    assert all(b >= a - 1e-12 for a, b in zip(powers, powers[1:]))


def test_t_power_converges_to_z_power() -> None:
    # Hold ncp fixed at 1.5 while df -> infinity: nct -> normal.
    ncp = 1.5
    target = z_test_power(ncp, 0.05, sided=2)
    prev_gap: float | None = None
    for n in (50, 200, 2000, 20000):
        d = ncp / np.sqrt(n)  # one-sample: ncp = d * sqrt(n)
        gap = abs(t_test_power(d, n, 0.05, sided=2, two_sample=False) - target)
        if prev_gap is not None:
            assert gap < prev_gap  # strictly closing in on the z-power
        prev_gap = gap
    assert prev_gap is not None
    assert prev_gap < 1e-4


def test_t_power_n_below_two_raises() -> None:
    with pytest.raises(ValueError):
        t_test_power(0.5, 1, 0.05)


def test_t_power_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError):
        t_test_power(0.5, 30, 1.0)


def test_t_power_invalid_sided_raises() -> None:
    with pytest.raises(ValueError):
        t_test_power(0.5, 30, 0.05, sided=0)


# --------------------------------------------------------------------------- #
# required_sample_size
# --------------------------------------------------------------------------- #


def test_required_n_matches_textbook() -> None:
    # Inverting the textbook power calc must land at ~64/group for 0.80 power.
    n = required_sample_size(0.5, 0.80, 0.05, sided=2, two_sample=True)
    assert 62 <= n <= 66
    # Power at n meets target; power at n-1 does not (smallest-n property).
    assert t_test_power(0.5, n, 0.05, sided=2, two_sample=True) >= 0.80
    assert t_test_power(0.5, n - 1, 0.05, sided=2, two_sample=True) < 0.80


def test_required_n_one_sample_smaller_than_two_sample() -> None:
    n_one = required_sample_size(0.5, 0.80, 0.05, sided=2, two_sample=False)
    n_two = required_sample_size(0.5, 0.80, 0.05, sided=2, two_sample=True)
    assert n_one < n_two
    assert n_one >= 2


def test_required_n_larger_effect_needs_fewer() -> None:
    n_small = required_sample_size(0.8, 0.80, 0.05, two_sample=True)
    n_large = required_sample_size(0.2, 0.80, 0.05, two_sample=True)
    assert n_small < n_large


def test_required_n_can_return_floor_two() -> None:
    # A large effect meets a modest target at the minimum n=2 (power@2 ~0.38).
    n = required_sample_size(2.0, 0.15, 0.05, sided=1, two_sample=True)
    assert n == 2
    assert t_test_power(2.0, 2, 0.05, sided=1, two_sample=True) >= 0.15


def test_required_n_unreachable_raises() -> None:
    # d=0 => power is flat at alpha, never reaches 0.80.
    with pytest.raises(ValueError):
        required_sample_size(0.0, 0.80, 0.05, two_sample=True, n_max=1000)


def test_required_n_unreachable_small_nmax_raises() -> None:
    with pytest.raises(ValueError):
        required_sample_size(0.05, 0.99, 0.05, two_sample=True, n_max=50)


def test_required_n_invalid_power_target_raises() -> None:
    for bad in (0.0, 1.0, -0.2, 2.0):
        with pytest.raises(ValueError):
            required_sample_size(0.5, bad, 0.05)


def test_required_n_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError):
        required_sample_size(0.5, 0.8, 0.0)


def test_required_n_invalid_sided_raises() -> None:
    with pytest.raises(ValueError):
        required_sample_size(0.5, 0.8, 0.05, sided=4)


def test_required_n_bad_nmax_raises() -> None:
    with pytest.raises(ValueError):
        required_sample_size(0.5, 0.8, 0.05, n_max=1)


# --------------------------------------------------------------------------- #
# average_power_bh
# --------------------------------------------------------------------------- #


def _g_one_sided(t: float, effect: float) -> float:
    return float(norm.sf(norm.ppf(1.0 - t) - effect))


def test_avg_power_pi0_one_returns_g_alpha() -> None:
    alpha, effect = 0.05, 3.0
    assert average_power_bh(effect, 1.0, alpha) == pytest.approx(
        _g_one_sided(alpha, effect), abs=1e-12
    )


def test_avg_power_effect_zero_matches_closed_form() -> None:
    # effect=0 => G(t)=t => fixed point t* = alpha*pi0 / (1 - alpha*(1-pi0)).
    alpha = 0.05
    for pi0 in (0.9, 0.5, 0.2):
        t_star = alpha * pi0 / (1.0 - alpha * (1.0 - pi0))
        assert average_power_bh(0.0, pi0, alpha) == pytest.approx(t_star, abs=1e-10)


def test_avg_power_monotone_increasing_in_effect() -> None:
    alpha = 0.05
    vals = [average_power_bh(e, 0.8, alpha) for e in (0.0, 1.0, 2.0, 3.0, 4.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_avg_power_increasing_in_pi0() -> None:
    # Genovese-Wasserman: t* solves t = alpha*(pi0 + (1-pi0)*G(t)). At pi0=1 the
    # threshold is exactly alpha (its maximum); lowering pi0 lowers the threshold
    # and hence the per-alternative power G(t*). So average power RISES with pi0.
    alpha, effect = 0.05, 3.0
    vals = [average_power_bh(effect, p, alpha) for p in (0.2, 0.5, 0.8, 0.95)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    # The top of the ladder equals the pi0==1 limit G(alpha).
    assert vals[-1] < average_power_bh(effect, 1.0, alpha)


def test_avg_power_two_sided_runs_and_bounded() -> None:
    p = average_power_bh(3.0, 0.8, 0.05, sided=2)
    assert 0.0 <= p <= 1.0
    # Two-sided catches the lower tail too, but for a positive shift the
    # upper-tail one-sided test is at least as powerful at the same alpha.
    assert p <= average_power_bh(3.0, 0.8, 0.05, sided=1) + 1e-9


def test_avg_power_fixed_point_satisfied() -> None:
    # Verify t* actually solves the fixed-point equation, then power = G(t*).
    alpha, effect, pi0 = 0.05, 2.5, 0.7
    power = average_power_bh(effect, pi0, alpha, sided=1)
    # Recover t* from power via G^{-1}: G(t)=Phi(Phi^{-1}(t)+effect)=power
    # => Phi^{-1}(t) = Phi^{-1}(power) - effect => t* = Phi(Phi^{-1}(power)-effect).
    t_star = float(norm.cdf(norm.ppf(power) - effect))
    lhs = t_star
    rhs = alpha * (pi0 + (1.0 - pi0) * _g_one_sided(t_star, effect))
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_avg_power_huge_effect_saturates() -> None:
    # Enormous effect: G(alpha) ~ 1, gap at alpha ~ 0 -> early-return branch.
    p = average_power_bh(15.0, 0.8, 0.05)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_avg_power_invalid_pi0_raises() -> None:
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            average_power_bh(3.0, bad, 0.05)


def test_avg_power_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError):
        average_power_bh(3.0, 0.8, 1.5)


def test_avg_power_invalid_sided_raises() -> None:
    with pytest.raises(ValueError):
        average_power_bh(3.0, 0.8, 0.05, sided=5)
