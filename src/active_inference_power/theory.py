"""Closed-form multiple-testing theory and structural checks.

Analytic quantities that anchor the Monte-Carlo machinery elsewhere in the
package: the Benjamini–Yekutieli harmonic constant, the Benjamini–Hochberg
(BH) critical threshold, the Simes global-null p-value, single-step
family-wise error rate (FWER) bounds, expected false rejections under the
global null, and a structural monotonicity check for step-up rejection masks.

Everything here is deterministic and analytic: no simulation, no RNG.  The
global-null BH rejection-count expectation is evaluated by a finite dynamic
program over the ordered threshold intervals, rather than being confused with
the separate global-null FDR identity.
Deterministic package domain code — imports only ``numpy`` and package contract
modules, never ``infrastructure.*``.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from ._types import PVAL_EPS

__all__ = [
    "by_constant",
    "bh_threshold",
    "simes_global_pvalue",
    "bonferroni_fwer_bound",
    "sidak_fwer",
    "bh_fdr_bound",
    "bh_fdr_bound_from_counts",
    "bh_global_null_fdr",
    "expected_false_rejections_global_null",
    "is_monotone_step_up",
]


def _validate_pvalues(pvalues: np.ndarray) -> np.ndarray:
    """Coerce and validate a p-value vector.

    Args:
        pvalues: Array-like of p-values, expected shape ``(m,)`` with values
            in ``[0, 1]`` (up to ``PVAL_EPS`` floating-point slack).

    Returns:
        A 1-D float64 array with values clipped to ``[0, 1]``.

    Raises:
        ValueError: If the input is not 1-D, contains NaN, or contains values
            outside ``[-PVAL_EPS, 1 + PVAL_EPS]``.
    """
    arr = np.asarray(pvalues, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"pvalues must be a 1-D array, got ndim={arr.ndim}")
    if arr.size and (
        np.any(np.isnan(arr)) or np.any(arr < -PVAL_EPS) or np.any(arr > 1.0 + PVAL_EPS)
    ):
        raise ValueError("pvalues must lie in [0, 1] and contain no NaN")
    return np.clip(arr, 0.0, 1.0)


def _validate_alpha(alpha: float) -> float:
    """Validate a nominal significance level.

    Args:
        alpha: Nominal level; must satisfy ``0 < alpha < 1``.

    Returns:
        ``alpha`` as a float.

    Raises:
        ValueError: If ``alpha`` is not strictly inside ``(0, 1)``.
    """
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    return alpha


def by_constant(m: int) -> float:
    """Benjamini–Yekutieli harmonic constant ``c(m) = sum_{i=1}^m 1/i``.

    The BY procedure controls the false discovery rate (FDR) under arbitrary
    dependence by running BH at level ``alpha / c(m)``.

    Args:
        m: Number of hypotheses; must be ``>= 0``. ``m == 0`` gives an empty
            sum, i.e. ``0.0``.

    Returns:
        The m-th harmonic number as a float (``0.0`` for ``m == 0``).

    Raises:
        ValueError: If ``m`` is negative.
    """
    m = int(m)
    if m < 0:
        raise ValueError(f"m must be non-negative, got {m}")
    if m == 0:
        return 0.0
    return float(np.sum(1.0 / np.arange(1, m + 1)))


def bh_threshold(pvalues: np.ndarray, alpha: float = 0.05) -> float:
    """Benjamini–Hochberg critical raw p-value.

    The BH step-up procedure finds the largest rank ``k`` (1-indexed, over
    the ascending order statistics ``p_(1) <= ... <= p_(m)``) satisfying
    ``p_(k) <= (k/m) * alpha`` and rejects every hypothesis with a raw
    p-value ``<= p_(k)``.

    Args:
        pvalues: Raw p-values, shape ``(m,)``, values in ``[0, 1]``. Order
            does not matter. May be empty.
        alpha: Nominal FDR level in ``(0, 1)``.

    Returns:
        The critical raw p-value ``p_(k)``, or ``0.0`` when no rank
        satisfies the step-up condition (including the empty case).

    Raises:
        ValueError: On malformed ``pvalues`` or ``alpha``.
    """
    p = _validate_pvalues(pvalues)
    alpha = _validate_alpha(alpha)
    m = p.size
    if m == 0:
        return 0.0
    p_sorted = np.sort(p)
    ranks = np.arange(1, m + 1)
    passing = np.nonzero(p_sorted <= (ranks / m) * alpha)[0]
    if passing.size == 0:
        return 0.0
    return float(p_sorted[passing[-1]])


def simes_global_pvalue(pvalues: np.ndarray) -> float:
    """Simes (1986) combined p-value for the global null hypothesis.

    ``p_Simes = min_i m * p_(i) / i`` over the ascending order statistics.
    Under the global null with independent (or PRDS) p-values this is a
    valid p-value, and it equals the smallest BH-adjusted p-value.

    Args:
        pvalues: Raw p-values, shape ``(m,)``, values in ``[0, 1]``.

    Returns:
        The Simes global p-value in ``[0, 1]``. An empty family carries no
        evidence against the global null, so the empty case returns ``1.0``.

    Raises:
        ValueError: On malformed ``pvalues``.
    """
    p = _validate_pvalues(pvalues)
    m = p.size
    if m == 0:
        return 1.0
    p_sorted = np.sort(p)
    ranks = np.arange(1, m + 1)
    return float(np.min(m * p_sorted / ranks))


def bonferroni_fwer_bound(alpha: float, m: int) -> float:
    """Union (Boole) bound on the FWER of the uncorrected level-alpha family.

    Testing ``m`` hypotheses each at raw level ``alpha`` yields
    ``FWER <= min(m * alpha, 1)``. Equivalently, this is the level at which
    the Bonferroni correction ``alpha/m`` guarantees FWER ``<= alpha``.

    Args:
        alpha: Per-test level in ``(0, 1)``.
        m: Number of hypotheses; must be ``>= 0``.

    Returns:
        ``min(m * alpha, 1)`` (``0.0`` for ``m == 0``).

    Raises:
        ValueError: If ``alpha`` is outside ``(0, 1)`` or ``m`` is negative.
    """
    alpha = _validate_alpha(alpha)
    m = int(m)
    if m < 0:
        raise ValueError(f"m must be non-negative, got {m}")
    return float(min(m * alpha, 1.0))


def sidak_fwer(alpha: float, m: int) -> float:
    """Exact FWER of ``m`` independent level-alpha tests under the global null.

    ``FWER = 1 - (1 - alpha)^m``: the probability of at least one false
    rejection among ``m`` independent tests, each with false-positive
    probability ``alpha``. Always ``<= bonferroni_fwer_bound(alpha, m)``.

    Args:
        alpha: Per-test level in ``(0, 1)``.
        m: Number of hypotheses; must be ``>= 0``.

    Returns:
        ``1 - (1 - alpha)^m`` (``0.0`` for ``m == 0``).

    Raises:
        ValueError: If ``alpha`` is outside ``(0, 1)`` or ``m`` is negative.
    """
    alpha = _validate_alpha(alpha)
    m = int(m)
    if m < 0:
        raise ValueError(f"m must be non-negative, got {m}")
    return float(1.0 - (1.0 - alpha) ** m)


def bh_fdr_bound(pi0: float, alpha: float) -> float:
    """Exact BH FDR under independence: ``pi0 * alpha``.

    Benjamini–Hochberg at level ``alpha`` with a fraction ``pi0`` of true
    nulls attains FDR exactly ``pi0 * alpha`` for independent p-values
    (Benjamini & Yekutieli 2001, Theorem 1.2 gives ``(m0/m) * alpha``).

    Args:
        pi0: Fraction of true null hypotheses in ``[0, 1]``.
        alpha: Nominal FDR level in ``(0, 1)``.

    Returns:
        ``pi0 * alpha``.

    Raises:
        ValueError: If ``pi0`` is outside ``[0, 1]`` or ``alpha`` outside
            ``(0, 1)``.
    """
    alpha = _validate_alpha(alpha)
    pi0 = float(pi0)
    if not 0.0 <= pi0 <= 1.0:
        raise ValueError(f"pi0 must lie in [0, 1], got {pi0}")
    return pi0 * alpha


def bh_fdr_bound_from_counts(m0: int, m: int, alpha: float) -> float:
    """Return the exact finite-family BH bound ``(m0 / m) * alpha``.

    Simulation configurations often specify a nominal null fraction ``pi0``
    that is converted to an integer null count.  Certificates must compare
    against the realized finite-family fraction rather than silently using the
    unrounded request.  This helper makes that accounting explicit.
    """
    alpha = _validate_alpha(alpha)
    if isinstance(m, bool) or int(m) != m or int(m) < 1:
        raise ValueError("m must be a positive integer")
    if isinstance(m0, bool) or int(m0) != m0 or not 0 <= int(m0) <= int(m):
        raise ValueError("m0 must be an integer in [0, m]")
    return float(int(m0) / int(m) * alpha)


def bh_global_null_fdr(alpha: float, m: int) -> float:
    """Return the global-null BH FDR under independent continuous p-values.

    Under the global null, every rejection is false, so ``FDR`` is
    ``P(R > 0)`` rather than ``E[R]``.  The Simes identity therefore gives
    ``FDR = alpha`` for a non-empty family.  This intentionally has a separate
    name from :func:`expected_false_rejections_global_null`, which returns the
    expected *count* of rejections.
    """
    alpha = _validate_alpha(alpha)
    m = int(m)
    if m < 0:
        raise ValueError(f"m must be non-negative, got {m}")
    return alpha if m else 0.0


def _binomial_pmf(n: int, probability: float) -> np.ndarray:
    """Return a normalized binomial PMF without introducing a SciPy dependency."""
    if n == 0:
        return np.array([1.0], dtype=np.float64)
    k = np.arange(n + 1, dtype=np.float64)
    log_coefficients = np.array(
        [
            math.lgamma(n + 1.0)
            - math.lgamma(int(index) + 1.0)
            - math.lgamma(n - int(index) + 1.0)
            for index in k
        ],
        dtype=np.float64,
    )
    log_mass = (
        log_coefficients
        + k * math.log(probability)
        + (n - k) * math.log1p(-probability)
    )
    mass = np.exp(log_mass)
    return mass / float(mass.sum())


@lru_cache(maxsize=32)
def _expected_bh_rejections_global_null(alpha: float, m: int) -> float:
    """Compute ``E[R]`` for BH by propagating threshold-bin counts.

    For threshold rank ``j``, the interval
    ``((j - 1) * alpha / m, j * alpha / m]`` has mass ``alpha / m``.
    Conditional on the number of p-values already seen in earlier intervals,
    the number entering the next interval is binomial.  The state carries the
    probability-weighted current maximum BH rank, so the final sum is an exact
    finite calculation up to floating-point roundoff.
    """
    probabilities = np.zeros(m + 1, dtype=np.float64)
    weighted_ranks = np.zeros(m + 1, dtype=np.float64)
    probabilities[0] = 1.0
    interval_mass = alpha / m

    for rank in range(1, m + 1):
        remaining_mass = 1.0 - (rank - 1) * interval_mass
        next_probabilities = np.zeros_like(probabilities)
        next_weighted_ranks = np.zeros_like(weighted_ranks)
        conditional_probability = interval_mass / remaining_mass
        for seen in range(m + 1):
            state_probability = probabilities[seen]
            if state_probability == 0.0:
                continue
            remaining = m - seen
            pmf = _binomial_pmf(remaining, conditional_probability)
            for entered, transition in enumerate(pmf):
                destination = seen + entered
                mass = state_probability * float(transition)
                next_probabilities[destination] += mass
                if destination >= rank:
                    next_weighted_ranks[destination] += rank * mass
                else:
                    next_weighted_ranks[destination] += (
                        weighted_ranks[seen] * float(transition)
                    )
        probabilities = next_probabilities
        weighted_ranks = next_weighted_ranks

    return float(weighted_ranks.sum())


def expected_false_rejections_global_null(
    alpha: float, m: int, method: str = "bonferroni"
) -> float:
    """Expected number of false rejections under the global null.

    All ``m`` hypotheses are true nulls with independent uniform p-values.

    - ``"bonferroni"``: each test is run at ``alpha/m``, so the expectation
      is ``m * (alpha/m) = alpha``.
    - ``"benjamini_hochberg"``: the expected number of BH rejections is
      evaluated from the finite ordered-threshold distribution.  It is
      generally larger than ``alpha``; the separate global-null FDR identity is
      :func:`bh_global_null_fdr`.
    - ``"sidak"``: each test is run at ``1 - (1-alpha)^(1/m)``, so the
      expectation is ``m * (1 - (1-alpha)^(1/m))``.

    Args:
        alpha: Nominal level in ``(0, 1)``.
        m: Number of hypotheses; must be ``>= 0``. ``m == 0`` returns ``0.0``
            for every method (no tests, no rejections).
        method: One of ``"bonferroni"``, ``"benjamini_hochberg"``,
            ``"sidak"``.

    Returns:
        The expected count of false rejections.

    Raises:
        ValueError: On an unknown ``method``, invalid ``alpha``, or
            negative ``m``.
    """
    alpha = _validate_alpha(alpha)
    m = int(m)
    if m < 0:
        raise ValueError(f"m must be non-negative, got {m}")
    if method not in {"bonferroni", "benjamini_hochberg", "sidak"}:
        raise ValueError(
            "unknown method "
            f"{method!r}; expected 'bonferroni', 'benjamini_hochberg', or 'sidak'"
        )
    if m == 0:
        return 0.0
    if method == "bonferroni":
        return alpha
    if method == "benjamini_hochberg":
        return _expected_bh_rejections_global_null(alpha, m)
    # sidak
    return float(m * (1.0 - (1.0 - alpha) ** (1.0 / m)))


def is_monotone_step_up(pvalues: np.ndarray, rejected: np.ndarray) -> bool:
    """Check the step-up staircase property of a rejection mask.

    A rejection mask is *monotone step-up consistent* iff for every rejected
    p-value ``p_i``, every hypothesis with ``p_j <= p_i`` is also rejected.
    Equivalently: the rejected set is exactly ``{j : p_j <= t}`` for the
    threshold ``t = max`` rejected p-value. Every step-up / step-down /
    single-step procedure produces such a mask; a mask that skips a smaller
    p-value while rejecting a larger one violates it.

    Args:
        pvalues: Raw p-values, shape ``(m,)``, values in ``[0, 1]``.
        rejected: Boolean mask, shape ``(m,)``, aligned with ``pvalues``.

    Returns:
        True iff the staircase property holds. An empty family or an
        all-False mask is vacuously monotone.

    Raises:
        ValueError: If shapes differ or ``pvalues`` is malformed.
    """
    p = _validate_pvalues(pvalues)
    mask = np.asarray(rejected, dtype=bool)
    if mask.ndim != 1 or mask.shape != p.shape:
        raise ValueError(
            f"rejected must be a 1-D boolean mask matching pvalues; "
            f"got shapes {mask.shape} vs {p.shape}"
        )
    if not np.any(mask):
        return True
    threshold = float(np.max(p[mask]))
    return bool(np.all(mask[p <= threshold]))
