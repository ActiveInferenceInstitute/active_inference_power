"""Finite-simulation uncertainty and replication-budget accounting.

The quantities in this module describe Monte-Carlo error for independent,
bounded replication outcomes. They are not confidence intervals for a
scientific population parameter and do not turn a misspecified simulation into
evidence for a theorem.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.stats import beta, norm, t as t_dist

__all__ = [
    "hoeffding_radius",
    "required_reps_for_half_width",
    "summarize_bounded",
    "wilson_interval",
    "clopper_pearson_interval",
    "empirical_bernstein_radius",
    "paired_difference_summary",
]


def _validate_confidence(confidence: float) -> float:
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    return confidence


def hoeffding_radius(n: int, confidence: float = 0.95) -> float:
    """Return the distribution-free Hoeffding radius for a bounded mean.

    For independent observations in ``[0, 1]``, the sample mean lies within
    this radius of its expectation with probability at least ``confidence``.
    The radius is conservative and is used for replication-budget planning.
    """
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    confidence = _validate_confidence(confidence)
    return float(math.sqrt(math.log(2.0 / (1.0 - confidence)) / (2.0 * int(n))))


def required_reps_for_half_width(
    half_width: float, confidence: float = 0.95
) -> int:
    """Return a conservative replication count for a bounded mean target.

    This inverts the Hoeffding radius, so it is a worst-case budget for an
    ``[0, 1]`` estimand. It is deliberately more conservative than a pilot
    variance calculation and is therefore appropriate as a release-planning
    floor rather than as a promise about every future data-generating process.
    """
    confidence = _validate_confidence(confidence)
    half_width = float(half_width)
    if not math.isfinite(half_width) or not 0.0 < half_width < 1.0:
        raise ValueError(f"half_width must lie in (0, 1), got {half_width!r}")
    return int(
        math.ceil(math.log(2.0 / (1.0 - confidence)) / (2.0 * half_width**2))
    )


def summarize_bounded(
    values: np.ndarray | Sequence[float | bool],
    confidence: float = 0.95,
    target_half_width: float | None = None,
) -> dict[str, Any]:
    """Summarize an independent sample of finite values in ``[0, 1]``.

    The normal/t interval is a practical Monte-Carlo precision summary. The
    separate ``hoeffding_radius`` is the conservative distribution-free bound
    used for sample-size accounting. Both are clipped to the valid probability
    range in the returned JSON-shaped mapping.
    """
    confidence = _validate_confidence(confidence)
    if target_half_width is not None:
        target_half_width = float(target_half_width)
        if not math.isfinite(target_half_width) or not 0.0 < target_half_width < 1.0:
            raise ValueError("target_half_width must lie in (0, 1)")
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size < 1:
        raise ValueError("values must be a non-empty 1-D sample")
    if not np.all(np.isfinite(sample)) or np.any((sample < 0.0) | (sample > 1.0)):
        raise ValueError("values must be finite and lie in [0, 1]")

    n = int(sample.size)
    mean = float(np.mean(sample))
    if n == 1:
        variance = 0.0
        standard_error = 0.0
        t_multiplier = 0.0
    else:
        variance = float(np.var(sample, ddof=1))
        standard_error = float(math.sqrt(variance / n))
        t_multiplier = float(t_dist.ppf((1.0 + confidence) / 2.0, n - 1))
    normal_half_width = float(t_multiplier * standard_error)
    return {
        "n": n,
        "mean": mean,
        "variance": variance,
        "standard_error": standard_error,
        "t_multiplier": t_multiplier,
        "normal_ci_lower": max(0.0, mean - normal_half_width),
        "normal_ci_upper": min(1.0, mean + normal_half_width),
        "normal_half_width": normal_half_width,
        "hoeffding_radius": hoeffding_radius(n, confidence),
        "interval_method": "student_t_monte_carlo",
        "planning_method": "hoeffding_distribution_free",
        "confidence": confidence,
        "required_n_for_target": (
            required_reps_for_half_width(target_half_width, confidence)
            if target_half_width is not None
            else None
        ),
    }


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return the Wilson score interval for a Bernoulli proportion."""
    if isinstance(trials, bool) or int(trials) != trials or trials < 1:
        raise ValueError("trials must be a positive integer")
    if isinstance(successes, bool) or int(successes) != successes or not 0 <= successes <= trials:
        raise ValueError("successes must be an integer in [0, trials]")
    confidence = _validate_confidence(confidence)
    n = int(trials)
    z = float(norm.ppf((1.0 + confidence) / 2.0))
    phat = float(successes) / n
    denominator = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def clopper_pearson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return the exact two-sided Clopper--Pearson interval."""
    if isinstance(trials, bool) or int(trials) != trials or trials < 1:
        raise ValueError("trials must be a positive integer")
    if isinstance(successes, bool) or int(successes) != successes or not 0 <= successes <= trials:
        raise ValueError("successes must be an integer in [0, trials]")
    confidence = _validate_confidence(confidence)
    successes, trials = int(successes), int(trials)
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - tail, successes + 1, trials - successes))
    return lower, upper


def empirical_bernstein_radius(values: np.ndarray | list[float], confidence: float = 0.95) -> float:
    """Return an empirical-Bernstein radius for a bounded mean."""
    confidence = _validate_confidence(confidence)
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size < 2 or not np.all(np.isfinite(sample)) or np.any((sample < 0) | (sample > 1)):
        raise ValueError("values must be a finite 1-D sample of at least two values in [0, 1]")
    n = sample.size
    variance = float(np.var(sample, ddof=1))
    log_term = math.log(3.0 / (1.0 - confidence))
    return float(math.sqrt(2.0 * variance * log_term / n) + 3.0 * log_term / (n - 1))


def paired_difference_summary(
    first: np.ndarray | list[float],
    second: np.ndarray | list[float],
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Summarize paired bounded differences from common-random-number runs."""
    confidence = _validate_confidence(confidence)
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.ndim != 1 or right.shape != left.shape or left.size < 1:
        raise ValueError("paired samples must be non-empty 1-D arrays of equal shape")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("paired samples must be finite")
    if np.any((left < 0) | (left > 1) | (right < 0) | (right > 1)):
        raise ValueError("paired samples must lie in [0, 1]")
    differences = right - left
    n = int(differences.size)
    mean = float(np.mean(differences))
    if n == 1:
        standard_error = 0.0
        half_width = 0.0
    else:
        standard_error = float(np.std(differences, ddof=1) / math.sqrt(n))
        half_width = float(t_dist.ppf((1.0 + confidence) / 2.0, n - 1) * standard_error)
    return {
        "n": n,
        "mean_difference": mean,
        "standard_error": standard_error,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
        "half_width": half_width,
        "hoeffding_radius": float(math.sqrt(2.0 * math.log(2.0 / (1.0 - confidence)) / n)),
    }


def mean_interval(
    values: np.ndarray | list[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Return a finite-simulation Student-t interval for an unbounded mean.

    Unlike :func:`summarize_bounded`, this helper deliberately does not clip
    the endpoints.  It is used for action-loop cost, stopping-time, and
    observation-count accounting where the estimand is not a probability.
    """
    confidence = _validate_confidence(confidence)
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size < 1 or not np.all(np.isfinite(sample)):
        raise ValueError("values must be a non-empty finite 1-D sample")
    n = int(sample.size)
    mean = float(np.mean(sample))
    if n == 1:
        return mean, mean
    standard_error = float(np.std(sample, ddof=1) / math.sqrt(n))
    multiplier = float(t_dist.ppf((1.0 + confidence) / 2.0, n - 1))
    half_width = multiplier * standard_error
    return mean - half_width, mean + half_width


def paired_numeric_difference_summary(
    first: np.ndarray | list[float],
    second: np.ndarray | list[float],
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Summarize paired differences for finite, non-bounded metrics.

    This is the common-random-number analogue of
    :func:`paired_difference_summary` for cost, stopping time, and other
    quantities that do not lie in ``[0, 1]``.
    """
    confidence = _validate_confidence(confidence)
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.ndim != 1 or right.shape != left.shape or left.size < 1:
        raise ValueError("paired samples must be non-empty 1-D arrays of equal shape")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("paired samples must be finite")
    differences = right - left
    lower, upper = mean_interval(differences, confidence)
    n = int(differences.size)
    standard_error = 0.0 if n == 1 else float(np.std(differences, ddof=1) / math.sqrt(n))
    return {
        "n": n,
        "mean_difference": float(np.mean(differences)),
        "standard_error": standard_error,
        "ci_lower": lower,
        "ci_upper": upper,
        "half_width": (upper - lower) / 2.0,
    }
