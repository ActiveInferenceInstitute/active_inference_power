"""Realized error decomposition for multiple-testing rejection masks.

Given a boolean rejection mask and the ground-truth alternative mask, this
module computes the confusion-matrix quantities of the classical
multiple-testing table (Benjamini & Hochberg 1995):

===========  =============  ===================  =======
             Declared null  Declared significant  Total
===========  =============  ===================  =======
True null    U              V (false discovery)   m0
True alt     T              S (true discovery)    m1
Total        m - R          R                      m
===========  =============  ===================  =======

Derived per-dataset rates follow the standard conventions:

- False discovery proportion ``FDP = V / max(R, 1)`` (defined as 0 when
  nothing is rejected).
- True positive rate (per-dataset power) ``TPR = S / max(m1, 1)`` (defined as
  0 when there are no true alternatives).
- Family-wise error event: ``V >= 1``.

Aggregating these per-dataset quantities over Monte-Carlo replications yields
the estimated FDR (= E[FDP]), average power (= E[TPR]), and FWER
(= P(V >= 1)).

Deterministic package domain code: imports only ``numpy`` and package contracts.
"""

from __future__ import annotations

import numpy as np

from ._types import ErrorSummary

__all__ = [
    "confusion_counts",
    "false_discovery_proportion",
    "true_positive_rate",
    "family_wise_error",
    "classify",
    "aggregate",
]


def _as_bool_masks(
    rejected: np.ndarray, is_alt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce both masks to aligned 1-D boolean arrays.

    Args:
        rejected: Boolean mask of hypotheses declared significant.
        is_alt: Boolean mask of true alternative hypotheses.

    Returns:
        The two masks as 1-D ``bool`` arrays.

    Raises:
        ValueError: If the two masks have different lengths.
    """
    rej = np.asarray(rejected, dtype=bool).ravel()
    alt = np.asarray(is_alt, dtype=bool).ravel()
    if rej.shape != alt.shape:
        raise ValueError(
            f"rejected and is_alt must have the same length; "
            f"got {rej.shape[0]} and {alt.shape[0]}"
        )
    return rej, alt


def confusion_counts(
    rejected: np.ndarray, is_alt: np.ndarray
) -> tuple[int, int, int, int]:
    """Compute the confusion-matrix counts ``(V, S, R, m1)``.

    Args:
        rejected: Boolean mask of hypotheses declared significant, shape ``(m,)``.
        is_alt: Boolean mask of true alternatives (True == non-null), shape ``(m,)``.

    Returns:
        Tuple ``(V, S, R, m1)`` where ``V`` is the number of false discoveries
        (rejected true nulls), ``S`` the number of true discoveries (rejected
        true alternatives), ``R = V + S`` the total number of rejections, and
        ``m1`` the number of true alternative hypotheses.

    Raises:
        ValueError: If the masks have mismatched lengths.
    """
    rej, alt = _as_bool_masks(rejected, is_alt)
    v = int(np.count_nonzero(rej & ~alt))
    s = int(np.count_nonzero(rej & alt))
    m1 = int(np.count_nonzero(alt))
    return v, s, v + s, m1


def false_discovery_proportion(rejected: np.ndarray, is_alt: np.ndarray) -> float:
    """Realized false discovery proportion ``V / max(R, 1)``.

    Defined as ``0.0`` when nothing is rejected (``R == 0``), matching the
    Benjamini–Hochberg convention so that ``E[FDP] = FDR``.

    Args:
        rejected: Boolean rejection mask, shape ``(m,)``.
        is_alt: Boolean true-alternative mask, shape ``(m,)``.

    Returns:
        FDP in ``[0, 1]``.
    """
    v, _s, r, _m1 = confusion_counts(rejected, is_alt)
    return float(v) / float(max(r, 1))


def true_positive_rate(rejected: np.ndarray, is_alt: np.ndarray) -> float:
    """Realized per-dataset power ``S / max(m1, 1)``.

    Defined as ``0.0`` when there are no true alternatives (``m1 == 0``).

    Args:
        rejected: Boolean rejection mask, shape ``(m,)``.
        is_alt: Boolean true-alternative mask, shape ``(m,)``.

    Returns:
        TPR in ``[0, 1]``.
    """
    _v, s, _r, m1 = confusion_counts(rejected, is_alt)
    return float(s) / float(max(m1, 1))


def family_wise_error(rejected: np.ndarray, is_alt: np.ndarray) -> bool:
    """Whether a family-wise error occurred, i.e. ``V >= 1``.

    Args:
        rejected: Boolean rejection mask, shape ``(m,)``.
        is_alt: Boolean true-alternative mask, shape ``(m,)``.

    Returns:
        True iff at least one true null hypothesis was rejected.
    """
    v, _s, _r, _m1 = confusion_counts(rejected, is_alt)
    return v >= 1


def classify(rejected: np.ndarray, is_alt: np.ndarray) -> ErrorSummary:
    """Full realized error decomposition for one dataset.

    Args:
        rejected: Boolean rejection mask, shape ``(m,)``.
        is_alt: Boolean true-alternative mask, shape ``(m,)``.

    Returns:
        An :class:`~_types.ErrorSummary` with every field populated:
        ``num_hypotheses`` = m, ``num_alt`` = m1, ``num_rejected`` = R,
        ``true_discoveries`` = S, ``false_discoveries`` = V,
        ``fdp`` = V/max(R,1), ``power`` = S/max(m1,1), and
        ``fwer_event`` = (V >= 1).
    """
    rej, alt = _as_bool_masks(rejected, is_alt)
    v, s, r, m1 = confusion_counts(rej, alt)
    return ErrorSummary(
        num_hypotheses=int(rej.shape[0]),
        num_alt=m1,
        num_rejected=r,
        true_discoveries=s,
        false_discoveries=v,
        fdp=float(v) / float(max(r, 1)),
        power=float(s) / float(max(m1, 1)),
        fwer_event=v >= 1,
    )


def aggregate(
    summaries: list[ErrorSummary],
) -> tuple[float, float, float, float]:
    """Aggregate per-dataset summaries into Monte-Carlo operating characteristics.

    Args:
        summaries: Per-replication :class:`~_types.ErrorSummary` objects.

    Returns:
        Tuple ``(fdr, power, fwer, mean_rejections)`` where ``fdr`` is the mean
        FDP (estimates E[FDP] = FDR), ``power`` the mean TPR (average power),
        ``fwer`` the fraction of replications with at least one false rejection
        (estimates P(V >= 1)), and ``mean_rejections`` the mean rejection count.
        An empty list returns ``(0.0, 0.0, 0.0, 0.0)``.
    """
    if not summaries:
        return (0.0, 0.0, 0.0, 0.0)
    n = float(len(summaries))
    fdr = sum(s.fdp for s in summaries) / n
    power = sum(s.power for s in summaries) / n
    fwer = sum(1.0 for s in summaries if s.fwer_event) / n
    mean_rejections = sum(float(s.num_rejected) for s in summaries) / n
    return (float(fdr), float(power), float(fwer), float(mean_rejections))
