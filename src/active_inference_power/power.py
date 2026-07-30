"""Analytic statistical power for the active-inference multiple-testing package.

Closed-form power calculations for the canonical z- and t-tests plus the
Genovese-Wasserman asymptotic average power of the Benjamini-Hochberg procedure.
Everything here is deterministic and built on :mod:`scipy.stats` survival
functions -- there is no Monte-Carlo estimation in this module.

Deterministic package domain code: imports ``numpy`` and ``scipy`` only, never
``infrastructure.*``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import nct, norm
from scipy.stats import t as t_dist


def _check_alpha(alpha: float) -> None:
    """Validate that a significance level lies strictly inside ``(0, 1)``.

    Args:
        alpha: Nominal significance level.

    Raises:
        ValueError: If ``alpha`` is not in the open interval ``(0, 1)``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")


def _check_sided(sided: int) -> None:
    """Validate a test sidedness argument.

    Args:
        sided: Number of tails; must be ``1`` or ``2``.

    Raises:
        ValueError: If ``sided`` is neither ``1`` nor ``2``.
    """
    if sided not in (1, 2):
        raise ValueError(f"sided must be 1 or 2, got {sided!r}")


def _nct_tail(x: float, df: int, ncp: float, upper: bool) -> float:
    """Noncentral-t tail probability with a stable large-``ncp`` fallback.

    ``scipy.stats.nct`` returns ``nan`` for large non-centrality parameters
    (the series it sums overflows). In that regime the noncentral t is
    accurately described by its normal approximation, so we fall back to the
    standard first-order approximation of the noncentral-t CDF, which keeps the
    power monotone in ``n`` rather than collapsing to ``nan``.

    Args:
        x: Evaluation point (a t critical value).
        df: Degrees of freedom (positive).
        ncp: Non-centrality parameter.
        upper: If ``True`` return ``P(T > x)``, else ``P(T <= x)``.

    Returns:
        The requested tail probability in ``[0, 1]``.
    """
    val = float(nct.sf(x, df, ncp)) if upper else float(nct.cdf(x, df, ncp))
    if not np.isnan(val):
        return val
    # First-order normal approximation to the noncentral-t CDF.
    z = (x * (1.0 - 1.0 / (4.0 * df)) - ncp) / np.sqrt(1.0 + x * x / (2.0 * df))
    return float(norm.sf(z)) if upper else float(norm.cdf(z))


def cohens_d(mean_diff: float, sd: float) -> float:
    """Standardized mean difference (Cohen's *d*).

    Args:
        mean_diff: Difference in means (raw units).
        sd: Pooled/population standard deviation. Must be strictly positive.

    Returns:
        The effect size ``mean_diff / sd``.

    Raises:
        ValueError: If ``sd <= 0``.
    """
    if sd <= 0:
        raise ValueError(f"sd must be strictly positive, got {sd!r}")
    return float(mean_diff) / float(sd)


def z_test_power(ncp: float, alpha: float, sided: int = 1) -> float:
    """Power of a z-test with a known non-centrality parameter.

    The non-centrality parameter for a one-sample z-test is ``ncp = sqrt(n) * d``.

    Args:
        ncp: Non-centrality parameter (mean of the test statistic under H1).
        alpha: Nominal significance level in ``(0, 1)``.
        sided: ``1`` for a one-sided (upper-tail) test, ``2`` for two-sided.

    Returns:
        The probability of rejecting H0, a value in ``[0, 1]``.

    Raises:
        ValueError: If ``alpha`` is outside ``(0, 1)`` or ``sided`` is invalid.
    """
    _check_alpha(alpha)
    _check_sided(sided)
    if sided == 1:
        z_crit = norm.ppf(1.0 - alpha)
        return float(norm.sf(z_crit - ncp))
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    lower = norm.cdf(-z_crit - ncp)
    upper = norm.sf(z_crit - ncp)
    return float(lower + upper)


def t_test_power(
    d: float,
    n: int,
    alpha: float,
    sided: int = 2,
    two_sample: bool = False,
) -> float:
    """Power of a Student t-test via the non-central t distribution.

    Args:
        d: Cohen's *d* effect size.
        n: Per-group sample size (``two_sample=True``) or total sample size
            (``two_sample=False``). Must be at least 2.
        alpha: Nominal significance level in ``(0, 1)``.
        sided: ``1`` for one-sided (upper-tail), ``2`` for two-sided.
        two_sample: If ``True``, equal-size two-sample test with
            ``ncp = d * sqrt(n / 2)`` and ``df = 2n - 2``. If ``False``,
            one-sample test with ``ncp = d * sqrt(n)`` and ``df = n - 1``.

    Returns:
        The probability of rejecting H0, a value in ``[0, 1]``.

    Raises:
        ValueError: If ``n < 2``, ``alpha`` is outside ``(0, 1)``, or ``sided``
            is invalid.
    """
    _check_alpha(alpha)
    _check_sided(sided)
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n!r}")
    if two_sample:
        ncp = d * np.sqrt(n / 2.0)
        df = 2 * n - 2
    else:
        ncp = d * np.sqrt(n)
        df = n - 1
    if sided == 1:
        t_crit = t_dist.ppf(1.0 - alpha, df)
        return _nct_tail(t_crit, df, ncp, upper=True)
    t_crit = t_dist.ppf(1.0 - alpha / 2.0, df)
    upper = _nct_tail(t_crit, df, ncp, upper=True)
    lower = _nct_tail(-t_crit, df, ncp, upper=False)
    return upper + lower


def required_sample_size(
    d: float,
    power_target: float,
    alpha: float,
    sided: int = 2,
    two_sample: bool = False,
    n_max: int = 100000,
) -> int:
    """Smallest sample size achieving a target power for a t-test.

    Exploits monotonicity of t-test power in ``n`` (for ``d > 0``) via a binary
    search over ``[2, n_max]``.

    Args:
        d: Cohen's *d* effect size.
        power_target: Desired power in ``(0, 1)``.
        alpha: Nominal significance level in ``(0, 1)``.
        sided: ``1`` for one-sided, ``2`` for two-sided.
        two_sample: See :func:`t_test_power`. ``n`` is per-group when ``True``.
        n_max: Largest sample size to consider.

    Returns:
        The smallest integer ``n >= 2`` whose power is at least ``power_target``.

    Raises:
        ValueError: If ``power_target`` is outside ``(0, 1)``, ``alpha`` is
            outside ``(0, 1)``, ``sided`` is invalid, ``n_max < 2``, or the
            target power is unreachable at ``n = n_max``.
    """
    _check_alpha(alpha)
    _check_sided(sided)
    if not (0.0 < power_target < 1.0):
        raise ValueError(f"power_target must be in (0, 1), got {power_target!r}")
    if n_max < 2:
        raise ValueError(f"n_max must be at least 2, got {n_max!r}")

    def power_at(n: int) -> float:
        return t_test_power(d, n, alpha, sided=sided, two_sample=two_sample)

    if power_at(n_max) < power_target:
        raise ValueError(
            f"power_target={power_target!r} unreachable by n_max={n_max!r} for d={d!r}"
        )
    lo, hi = 2, n_max
    while lo < hi:
        mid = (lo + hi) // 2
        if power_at(mid) >= power_target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _alt_pvalue_cdf(t: float, effect: float, sided: int) -> float:
    """Alternative-hypothesis CDF of a z-test p-value at threshold ``t``.

    ``G(t) = P(p <= t | H1)`` when the test statistic is shifted by ``effect``.

    Args:
        t: p-value threshold in ``(0, 1]``.
        effect: Mean shift of the z-statistic under the alternative.
        sided: ``1`` (upper-tail) or ``2`` (two-sided).

    Returns:
        The probability that a shifted-alternative p-value falls at or below
        ``t``, a value in ``[0, 1]``.
    """
    if sided == 1:
        # p = 1 - Phi(z); p <= t  <=>  z >= Phi^{-1}(1 - t).
        z_crit = norm.ppf(1.0 - t)
        return float(norm.sf(z_crit - effect))
    # Two-sided: p = 2(1 - Phi(|z|)); p <= t <=> |z| >= Phi^{-1}(1 - t/2).
    z_crit = norm.ppf(1.0 - t / 2.0)
    return float(norm.sf(z_crit - effect) + norm.cdf(-z_crit - effect))


def average_power_bh(
    effect: float,
    pi0: float,
    alpha: float,
    sided: int = 1,
) -> float:
    """Genovese-Wasserman asymptotic average power of Benjamini-Hochberg.

    Under independence, the BH rejection threshold converges to the largest
    fixed point ``t*`` in ``(0, alpha]`` of

        ``t = alpha * (pi0 + (1 - pi0) * G(t))``

    where ``G`` is the alternative p-value CDF (:func:`_alt_pvalue_cdf`). The
    asymptotic average power is ``G(t*)``.

    Args:
        effect: Mean shift of the alternative z-statistics (must be finite).
        pi0: Proportion of true nulls, in ``[0, 1]``.
        alpha: Nominal FDR level in ``(0, 1)``.
        sided: ``1`` (upper-tail) or ``2`` (two-sided) alternative CDF.

    Returns:
        The asymptotic average power ``G(t*)``, a value in ``[0, 1]``.

    Raises:
        ValueError: If ``pi0`` is outside ``[0, 1]``, ``alpha`` is outside
            ``(0, 1)``, or ``sided`` is invalid.
    """
    _check_alpha(alpha)
    _check_sided(sided)
    if not (0.0 <= pi0 <= 1.0):
        raise ValueError(f"pi0 must be in [0, 1], got {pi0!r}")

    def g(t: float) -> float:
        return _alt_pvalue_cdf(t, effect, sided)

    if pi0 >= 1.0:
        # Pure null: fixed point is t* = alpha exactly.
        return g(alpha)

    def fixed_point_gap(t: float) -> float:
        return alpha * (pi0 + (1.0 - pi0) * g(t)) - t

    # h(alpha) <= 0 and h(0+) = alpha * pi0 > 0, so a root exists in (0, alpha].
    # Scan downward from alpha to isolate the LARGEST sign change, then refine.
    gap_hi = fixed_point_gap(alpha)
    if abs(gap_hi) < 1e-14:
        return g(alpha)

    n_grid = 512
    grid = np.linspace(alpha, alpha / n_grid, n_grid)
    t_upper = alpha
    gap_upper = gap_hi
    for t_lower in grid[1:]:
        gap_lower = fixed_point_gap(t_lower)
        if gap_upper == 0.0:
            return g(t_upper)
        if gap_lower == 0.0:
            return g(t_lower)
        if gap_lower * gap_upper < 0.0:
            root = brentq(fixed_point_gap, t_lower, t_upper, xtol=1e-14)
            return g(root)
        t_upper, gap_upper = t_lower, gap_lower
    # Fallback: bracket the whole interval down to a tiny floor.
    t_floor = alpha / (n_grid * 1e3)
    root = brentq(fixed_point_gap, t_floor, alpha, xtol=1e-14)
    return g(root)
