"""Multiple-testing correction procedures.

Every public procedure takes a raw p-value vector ``pvalues`` of shape ``(m,)``
with entries in ``[0, 1]`` and a nominal level ``alpha`` and returns a
:class:`~_types.CorrectionResult`. All procedures are implemented so that the
rejection mask is exactly ``adjusted <= alpha``; this keeps the adjusted
p-values (a.k.a. q-values) and the boolean decisions mutually consistent for
single-step, step-down, and step-up families alike.

Implemented procedures:

* :func:`bonferroni` -- single-step FWER procedure.
* :func:`sidak` -- single-step FWER procedure under independence.
* :func:`holm` -- step-down FWER procedure (uniformly dominates Bonferroni).
* :func:`hochberg` -- step-up FWER procedure (dominates Holm under independence
  / positive dependence).
* :func:`benjamini_hochberg` -- step-up FDR procedure (BH).
* :func:`benjamini_yekutieli` -- step-up FDR procedure designed for arbitrary
  dependence (BY).
* :func:`storey_qvalue` -- pi0-adaptive q-value procedure; its finite-sample
  guarantees depend on the estimator and assumptions.
* :func:`storey_qvalue_conservative` -- fixed-lambda adaptive q-values using
  an exact one-sided binomial upper confidence bound on the null count.
* :func:`storey_qvalue_split_confirmation` -- independent-stream Storey
  estimation followed by confirmation-only rejection decisions.
* :func:`adaptive_bh` -- Benjamini-Krieger-Yekutieli two-stage adaptive BH;
  guarantees depend on the declared assumptions.
* :func:`weighted_bh` -- fixed, externally specified weighted BH under the
  independence/PRDS regimes appropriate to weighted hypotheses.
* :func:`apply_correction` -- string dispatcher over all of the above.

Edge cases handled by every procedure: ``m == 0`` (empty input), ``m == 1``,
``p == 0``, ``p == 1``, and unsorted input (results are permutation-equivariant).
The ``adjusted`` array is always clipped to ``[0, 1]`` and ``threshold`` is the
largest *raw* p-value that was rejected (``0.0`` when nothing is rejected).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom

from ._types import CANONICAL_CORRECTION_METHODS, PVAL_EPS, CorrectionResult

__all__ = [
    "bonferroni",
    "sidak",
    "holm",
    "hochberg",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "storey_qvalue",
    "storey_qvalue_conservative",
    "storey_qvalue_split_confirmation",
    "adaptive_bh",
    "weighted_bh",
    "apply_correction",
]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _validate_alpha(alpha: float) -> float:
    """Validate and normalize a nominal level."""
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    return alpha


def _prepare(pvalues: np.ndarray) -> np.ndarray:
    """Validate and normalise a raw p-value vector.

    Args:
        pvalues: Array-like of raw p-values, shape ``(m,)``.

    Returns:
        A 1-D ``float`` array clipped to ``[0, 1]``.

    Raises:
        ValueError: If the input is not 1-dimensional, or if any value lies
            meaningfully outside ``[0, 1]`` (beyond floating-point tolerance).
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"pvalues must be 1-dimensional, got ndim={p.ndim}")
    if p.size and (
        np.any(~np.isfinite(p)) or np.any(p < -PVAL_EPS) or np.any(p > 1.0 + PVAL_EPS)
    ):
        raise ValueError("pvalues must be finite and lie in [0, 1]")
    return np.clip(p, 0.0, 1.0)


def _finalize(
    method: str, p: np.ndarray, adjusted: np.ndarray, alpha: float
) -> CorrectionResult:
    """Assemble a :class:`CorrectionResult` from adjusted p-values.

    The rejection mask is defined as ``adjusted <= alpha`` and the reported
    ``threshold`` is the largest raw p-value among the rejected hypotheses
    (``0.0`` when none are rejected).

    Args:
        method: Canonical procedure name.
        p: Raw p-values (already validated / clipped), shape ``(m,)``.
        adjusted: Adjusted p-values in input order, shape ``(m,)``.
        alpha: Nominal level.

    Returns:
        The populated :class:`CorrectionResult`.
    """
    adjusted = np.clip(np.asarray(adjusted, dtype=float), 0.0, 1.0)
    rejected = adjusted <= alpha
    if bool(np.any(rejected)):
        threshold = float(np.max(p[rejected]))
    else:
        threshold = 0.0
    return CorrectionResult(
        method=method,
        adjusted=adjusted,
        rejected=rejected,
        alpha=float(alpha),
        threshold=threshold,
    )


def _bh_sorted(ps: np.ndarray, m: int) -> np.ndarray:
    """Sorted BH q-values for an already-ascending p-value vector.

    Computes the reverse-cumulative minimum of ``(m / rank) * p_(rank)``.

    Args:
        ps: p-values sorted ascending, shape ``(m,)``.
        m: Number of hypotheses.

    Returns:
        BH q-values aligned with ``ps`` (still in ascending order).
    """
    ranks = np.arange(1, m + 1, dtype=float)
    raw = (m / ranks) * ps
    return np.minimum.accumulate(raw[::-1])[::-1]


def _by_constant(m: int) -> float:
    """Benjamini-Yekutieli harmonic constant ``c(m) = sum_{i=1..m} 1/i``.

    Args:
        m: Number of hypotheses (``>= 1``).

    Returns:
        The ``m``-th harmonic number as a float.
    """
    return float(np.sum(1.0 / np.arange(1, m + 1, dtype=float)))


def _invert(order: np.ndarray, sorted_vals: np.ndarray, m: int) -> np.ndarray:
    """Scatter sorted-order values back into original input order.

    Args:
        order: The ``argsort`` permutation that produced the sorted order.
        sorted_vals: Values aligned with the sorted order.
        m: Number of hypotheses.

    Returns:
        Values re-indexed into the original input order.
    """
    out = np.empty(m, dtype=float)
    out[order] = sorted_vals
    return out


# --------------------------------------------------------------------------- #
# Single-step procedures
# --------------------------------------------------------------------------- #
def bonferroni(pvalues: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Bonferroni single-step FWER correction.

    Rejects ``p <= alpha / m``; adjusted p-values are ``min(m * p, 1)``.

    Args:
        pvalues: Raw p-values, shape ``(m,)``.
        alpha: Nominal family-wise error level.

    Returns:
        A :class:`CorrectionResult`.
    """
    alpha = _validate_alpha(alpha)
    p = _prepare(pvalues)
    m = p.size
    adjusted = m * p
    return _finalize("bonferroni", p, adjusted, alpha)


def sidak(pvalues: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Šidák single-step FWER correction (valid under independence).

    Rejects ``p <= 1 - (1 - alpha) ** (1 / m)``; adjusted p-values are
    ``1 - (1 - p) ** m``.

    Args:
        pvalues: Raw p-values, shape ``(m,)``.
        alpha: Nominal family-wise error level.

    Returns:
        A :class:`CorrectionResult`.
    """
    alpha = _validate_alpha(alpha)
    p = _prepare(pvalues)
    m = p.size
    adjusted = 1.0 - np.power(1.0 - p, m)
    return _finalize("sidak", p, adjusted, alpha)


# --------------------------------------------------------------------------- #
# Step-down / step-up FWER procedures
# --------------------------------------------------------------------------- #
def holm(pvalues: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Holm step-down FWER correction.

    Adjusted p-values are the running maximum of ``(m - i + 1) * p_(i)``
    (1-indexed) over the ascending order; the rejection ``adjusted <= alpha``
    reproduces the classic "reject while ``p_(i) <= alpha / (m - i + 1)`` and
    stop at the first failure" rule.

    Args:
        pvalues: Raw p-values, shape ``(m,)``.
        alpha: Nominal family-wise error level.

    Returns:
        A :class:`CorrectionResult`.
    """
    alpha = _validate_alpha(alpha)
    p = _prepare(pvalues)
    m = p.size
    if m == 0:
        return _finalize("holm", p, p.copy(), alpha)
    order = np.argsort(p, kind="stable")
    ps = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    raw = (m - ranks + 1.0) * ps
    adj_sorted = np.maximum.accumulate(raw)
    adjusted = _invert(order, adj_sorted, m)
    return _finalize("holm", p, adjusted, alpha)


def hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Hochberg step-up FWER correction (the step-up dual of Holm).

    Adjusted p-values are the reverse-cumulative minimum of
    ``(m - i + 1) * p_(i)``; the rejection ``adjusted <= alpha`` reproduces
    "reject all hypotheses up to the largest ``i`` with
    ``p_(i) <= alpha / (m - i + 1)``".

    Args:
        pvalues: Raw p-values, shape ``(m,)``.
        alpha: Nominal family-wise error level.

    Returns:
        A :class:`CorrectionResult`.
    """
    alpha = _validate_alpha(alpha)
    p = _prepare(pvalues)
    m = p.size
    if m == 0:
        return _finalize("hochberg", p, p.copy(), alpha)
    order = np.argsort(p, kind="stable")
    ps = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    raw = (m - ranks + 1.0) * ps
    adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted = _invert(order, adj_sorted, m)
    return _finalize("hochberg", p, adjusted, alpha)


# --------------------------------------------------------------------------- #
# FDR procedures
# --------------------------------------------------------------------------- #
def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Benjamini-Hochberg step-up FDR correction.

    Adjusted p-values (BH q-values) are the reverse-cumulative minimum of
    ``(m / rank) * p_(rank)``; the rejection ``adjusted <= alpha`` reproduces
    "reject all ``p <= p_(k)`` for the largest ``k`` with
    ``p_(k) <= (k / m) * alpha``".

    Args:
        pvalues: Raw p-values, shape ``(m,)``.
        alpha: Nominal false-discovery-rate level.

    Returns:
        A :class:`CorrectionResult`.
    """
    alpha = _validate_alpha(alpha)
    p = _prepare(pvalues)
    m = p.size
    if m == 0:
        return _finalize("benjamini_hochberg", p, p.copy(), alpha)
    order = np.argsort(p, kind="stable")
    ps = p[order]
    adj_sorted = _bh_sorted(ps, m)
    adjusted = _invert(order, adj_sorted, m)
    return _finalize("benjamini_hochberg", p, adjusted, alpha)


def benjamini_yekutieli(pvalues: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Benjamini-Yekutieli step-up FDR correction (arbitrary dependence).

    Equivalent to BH at level ``alpha / c(m)`` with
    ``c(m) = sum_{i=1..m} 1/i``. Adjusted p-values are the BH q-values scaled
    by ``c(m)`` and clipped to ``[0, 1]``.

    Args:
        pvalues: Raw p-values, shape ``(m,)``.
        alpha: Nominal false-discovery-rate level.

    Returns:
        A :class:`CorrectionResult`.
    """
    alpha = _validate_alpha(alpha)
    p = _prepare(pvalues)
    m = p.size
    if m == 0:
        return _finalize("benjamini_yekutieli", p, p.copy(), alpha)
    order = np.argsort(p, kind="stable")
    ps = p[order]
    adj_sorted = _bh_sorted(ps, m) * _by_constant(m)
    adjusted = _invert(order, adj_sorted, m)
    return _finalize("benjamini_yekutieli", p, adjusted, alpha)


def storey_qvalue(
    pvalues: np.ndarray, alpha: float = 0.05, lambda_: float = 0.5
) -> CorrectionResult:
    """Storey q-value procedure with a plug-in ``pi0`` estimate.

    Estimates the null proportion
    ``pi0_hat = min(1, #{p > lambda} / (m * (1 - lambda)))`` and reports
    q-values ``pi0_hat * BH_q``; rejects ``q <= alpha``.

    Args:
        pvalues: Raw p-values, shape ``(m,)``.
        alpha: Nominal false-discovery-rate level.
        lambda_: Tuning parameter in ``[0, 1)`` for the ``pi0`` estimate.

    Returns:
        A :class:`CorrectionResult`.

    Raises:
        ValueError: If ``lambda_`` is not in ``[0, 1)``.
    """
    if not (0.0 <= lambda_ < 1.0):
        raise ValueError(f"lambda_ must be in [0, 1), got {lambda_}")
    alpha = _validate_alpha(alpha)
    p = _prepare(pvalues)
    m = p.size
    if m == 0:
        return _finalize("storey_qvalue", p, p.copy(), alpha)
    count_above = int(np.count_nonzero(p > lambda_))
    pi0_hat = min(1.0, count_above / (m * (1.0 - lambda_)))
    order = np.argsort(p, kind="stable")
    ps = p[order]
    adj_sorted = pi0_hat * _bh_sorted(ps, m)
    adjusted = _invert(order, adj_sorted, m)
    return _finalize("storey_qvalue", p, adjusted, alpha)


def storey_qvalue_conservative(
    pvalues: np.ndarray,
    alpha: float = 0.05,
    lambda_: float = 0.5,
    confidence: float = 0.99,
) -> CorrectionResult:
    """Apply a conservative fixed-lambda adaptive BH procedure.

    The unconstrained Storey plug-in can underestimate the finite-sample null
    count.
    Here ``K = #{p_i > lambda}`` is used with an exact one-sided binomial
    inversion over the unknown number of true nulls. The largest ``m0`` whose
    ``Binomial(m0, 1-lambda)`` lower-tail probability at ``K`` is at least
    ``1-confidence`` is retained, and ``m0 / m`` scales the BH adjusted values.
    Alternative hypotheses can only add observations to the upper tail, so the
    bound is conservative for independent, marginally uniform null p-values.

    This is an assumption-gated procedure: it is not a replacement for BH
    under arbitrary dependence, estimated weights, or invalid p-values.

    Args:
        pvalues: Raw p-values, shape ``(m,)``.
        alpha: Nominal false-discovery level.
        lambda_: Fixed tail threshold in ``[0, 1)``.
        confidence: One-sided confidence level for the null-count bound.

    Returns:
        A :class:`CorrectionResult` named ``storey_qvalue_conservative``.

    Raises:
        ValueError: If ``lambda_`` or ``confidence`` is outside its domain.
    """
    if not 0.0 <= lambda_ < 1.0:
        raise ValueError(f"lambda_ must be in [0, 1), got {lambda_}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    alpha = _validate_alpha(alpha)
    p = _prepare(pvalues)
    m = p.size
    if m == 0:
        return _finalize("storey_qvalue_conservative", p, p.copy(), alpha)
    upper_null_count = _storey_upper_null_count(p, lambda_, confidence)
    pi0_upper = min(1.0, upper_null_count / m)
    order = np.argsort(p, kind="stable")
    ps = p[order]
    adj_sorted = pi0_upper * _bh_sorted(ps, m)
    adjusted = _invert(order, adj_sorted, m)
    return _finalize("storey_qvalue_conservative", p, adjusted, alpha)


def _storey_upper_null_count(p: np.ndarray, lambda_: float, confidence: float) -> int:
    """Invert the fixed-lambda binomial lower tail for a null-count bound."""
    m = p.size
    count_above = int(np.count_nonzero(p > lambda_))
    candidate_null_counts = np.arange(m + 1, dtype=int)
    lower_tail = binom.cdf(count_above, candidate_null_counts, 1.0 - float(lambda_))
    admissible = candidate_null_counts[lower_tail >= (1.0 - float(confidence))]
    return int(admissible[-1]) if admissible.size else m


def storey_qvalue_split_confirmation(
    selection_pvalues: np.ndarray,
    confirmation_pvalues: np.ndarray,
    alpha: float = 0.05,
    lambda_: float = 0.5,
    confidence: float = 0.99,
) -> CorrectionResult:
    """Use an independent selection stream to scale confirmation q-values.

    The fixed-lambda null-count bound is estimated from ``selection_pvalues``;
    only ``confirmation_pvalues`` determine the returned rejection mask. The
    streams must contain the same indexed hypotheses and must be independent
    under the declared protocol. This separation removes the most direct
    same-vector adaptivity problem, but the function itself is still an
    assumption-gated procedure rather than a universal finite-sample theorem.
    """
    if not 0.0 <= lambda_ < 1.0:
        raise ValueError(f"lambda_ must be in [0, 1), got {lambda_}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    alpha = _validate_alpha(alpha)
    selection = _prepare(selection_pvalues)
    confirmation = _prepare(confirmation_pvalues)
    if selection.shape != confirmation.shape:
        raise ValueError("selection and confirmation p-values must have matching shapes")
    m = confirmation.size
    if m == 0:
        return _finalize("storey_qvalue_split_confirmation", confirmation, confirmation.copy(), alpha)
    upper_null_count = _storey_upper_null_count(selection, lambda_, confidence)
    pi0_upper = min(1.0, upper_null_count / m)
    order = np.argsort(confirmation, kind="stable")
    q_sorted = pi0_upper * _bh_sorted(confirmation[order], m)
    return _finalize(
        "storey_qvalue_split_confirmation",
        confirmation,
        _invert(order, q_sorted, m),
        alpha,
    )


def adaptive_bh(pvalues: np.ndarray, alpha: float = 0.05) -> CorrectionResult:
    """Benjamini-Krieger-Yekutieli (2006) two-stage adaptive BH.

    Stage 1 applies BH at ``alpha' = alpha / (1 + alpha)`` yielding ``r1``
    rejections. If ``r1 == 0`` nothing is rejected; if ``r1 == m`` everything is
    rejected. Otherwise the null count is estimated as ``m0_hat = m - r1`` and
    stage 2 applies BH at ``alpha * m / m0_hat`` (implemented as adjusted
    q-values ``BH_q * m0_hat / m``).

    Args:
        pvalues: Raw p-values, shape ``(m,)``.
        alpha: Nominal false-discovery-rate level.

    Returns:
        A :class:`CorrectionResult`.
    """
    alpha = _validate_alpha(alpha)
    p = _prepare(pvalues)
    m = p.size
    if m == 0:
        return _finalize("adaptive_bh", p, p.copy(), alpha)
    alpha1 = alpha / (1.0 + alpha)
    r1 = benjamini_hochberg(p, alpha1).num_rejected
    if r1 == 0:
        return _finalize("adaptive_bh", p, np.ones(m), alpha)
    if r1 == m:
        return _finalize("adaptive_bh", p, np.zeros(m), alpha)
    m0_hat = m - r1
    order = np.argsort(p, kind="stable")
    ps = p[order]
    adj_sorted = _bh_sorted(ps, m) * (m0_hat / m)
    adjusted = _invert(order, adj_sorted, m)
    return _finalize("adaptive_bh", p, adjusted, alpha)


def weighted_bh(
    pvalues: np.ndarray, weights: np.ndarray, alpha: float = 0.05
) -> CorrectionResult:
    """Fixed-weight Benjamini--Hochberg correction.

    ``weights`` must be finite and strictly positive with mean one.  The
    weights are fixed before seeing the p-values; they are not estimated from
    the same evidence.  Under the corresponding independence/PRDS conditions
    this is the standard weighted-BH rule.  The function deliberately records
    no arbitrary-dependence guarantee; callers should use BY when that regime
    is required.
    """
    alpha = _validate_alpha(alpha)
    p = _prepare(pvalues)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape != p.shape:
        raise ValueError(f"weights must have shape {p.shape}, got {w.shape}")
    if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("weights must be finite and strictly positive")
    if p.size == 0:
        return _finalize("weighted_bh", p, p.copy(), alpha)
    if not np.isclose(float(np.mean(w)), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("weights must have arithmetic mean one")
    order = np.argsort(p / w, kind="stable")
    weighted_p = p[order] / w[order]
    ranks = np.arange(1, p.size + 1, dtype=float)
    scaled = (p.size / ranks) * weighted_p
    # Weighted BH is ordinary BH on the normalized p-values p_i / w_i.
    # The adjusted values therefore remain on that normalized scale.  Do not
    # multiply the q-values by w again: doing so changes the rejection set and
    # is not the weighted step-up rule p_i <= w_i * alpha * k / m.
    q_weighted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(p.size, dtype=float)
    adjusted[order] = q_weighted_sorted
    return _finalize("weighted_bh", p, adjusted, alpha)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
_METHODS = {
    "bonferroni": bonferroni,
    "sidak": sidak,
    "holm": holm,
    "hochberg": hochberg,
    "benjamini_hochberg": benjamini_hochberg,
    "benjamini_yekutieli": benjamini_yekutieli,
    "storey_qvalue": storey_qvalue,
    "storey_qvalue_conservative": storey_qvalue_conservative,
    "storey_qvalue_split_confirmation": storey_qvalue_split_confirmation,
    "adaptive_bh": adaptive_bh,
    "weighted_bh": weighted_bh,
}

if frozenset(_METHODS) != CANONICAL_CORRECTION_METHODS:
    raise RuntimeError("correction dispatcher and result contract disagree")


def apply_correction(
    method: str, pvalues: np.ndarray, alpha: float = 0.05, **kwargs: object
) -> CorrectionResult:
    """Dispatch to a named multiple-testing correction.

    Recognised canonical names: ``bonferroni``, ``sidak``, ``holm``,
    ``hochberg``, ``benjamini_hochberg``, ``benjamini_yekutieli``,
    ``storey_qvalue``, ``storey_qvalue_conservative``,
    ``storey_qvalue_split_confirmation``, ``adaptive_bh``, and ``weighted_bh``.
    Extra keyword arguments are forwarded to the procedure (e.g. ``lambda_``
    for Storey, ``confidence`` for the conservative fixed-lambda procedure,
    and fixed ``weights`` for weighted BH).

    Args:
        method: Name of the correction procedure.
        pvalues: Raw p-values, shape ``(m,)``.
        alpha: Nominal level.
        **kwargs: Procedure-specific keyword arguments.

    Returns:
        A :class:`CorrectionResult`.

    Raises:
        ValueError: If ``method`` is not a recognised procedure name.
    """
    if not isinstance(method, str):
        raise ValueError(f"method must be a string, got {method!r}")
    if method not in _METHODS:
        raise ValueError(f"unknown correction method: {method!r}")
    if method == "weighted_bh":
        if "weights" not in kwargs:
            raise ValueError("weighted_bh requires fixed weights")
        weights = kwargs.pop("weights")
        return weighted_bh(pvalues, weights, alpha, **kwargs)  # type: ignore[arg-type]
    return _METHODS[method](pvalues, alpha, **kwargs)  # type: ignore[operator]
