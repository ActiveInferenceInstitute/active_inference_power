"""Null p-value calibration diagnostics for the declared independent regime."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Sequence

import numpy as np
from scipy.stats import kstest

from ._types import TwoGroupsConfig
from .simulation import generate_pvalues

__all__ = [
    "calibration_replication_samples",
    "null_pvalue_calibration",
    "null_pvalue_calibration_from_samples",
]


def calibration_replication_samples(
    config: TwoGroupsConfig,
    n_reps: int,
    *,
    seed: int = 0,
    start: int = 0,
    stop: int | None = None,
) -> np.ndarray:
    """Return one deterministic p-value block for a replication interval.

    This helper is the resumable counterpart to :func:`null_pvalue_calibration`.
    Each replication receives a child ``SeedSequence`` stream, so a block can
    be regenerated independently and concatenated without changing the values
    produced by another block.  It is intentionally separate from the
    validated reference calibration function, whose sequential stream remains
    the reproducibility anchor.
    """
    if isinstance(n_reps, bool) or int(n_reps) != n_reps or int(n_reps) < 1:
        raise ValueError(f"n_reps must be a positive integer, got {n_reps!r}")
    n_reps = int(n_reps)
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if isinstance(start, bool) or int(start) != start or int(start) < 0:
        raise ValueError(f"start must be a non-negative integer, got {start!r}")
    start = int(start)
    if stop is None:
        stop = n_reps
    if isinstance(stop, bool) or int(stop) != stop or int(stop) < start or int(stop) > n_reps:
        raise ValueError(f"stop must satisfy start <= stop <= n_reps, got {stop!r}")
    stop = int(stop)
    if config.m < 1:
        raise ValueError("calibration requires config.m >= 1")

    null_config = replace(config, pi0=1.0, effect=0.0, rho=0.0)
    children = np.random.SeedSequence(int(seed)).spawn(n_reps)
    samples = np.empty((stop - start) * config.m, dtype=float)
    for block_index, replication in enumerate(range(start, stop)):
        pvalues, _ = generate_pvalues(null_config, np.random.default_rng(children[replication]))
        offset = block_index * config.m
        samples[offset : offset + config.m] = pvalues
    return samples


def null_pvalue_calibration_from_samples(
    config: TwoGroupsConfig,
    n_reps: int,
    samples: Sequence[float] | np.ndarray,
    *,
    seed: int = 0,
    confidence: float = 0.95,
    grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Finalize a calibration diagnostic from a complete sample accumulator.

    The caller is responsible for assembling exactly ``n_reps * m`` samples
    from the deterministic blocks returned by
    :func:`calibration_replication_samples`.  No random state is consumed here;
    the seed is retained solely as provenance in the resulting artifact.
    """
    if isinstance(n_reps, bool) or int(n_reps) != n_reps or int(n_reps) < 1:
        raise ValueError(f"n_reps must be a positive integer, got {n_reps!r}")
    n_reps = int(n_reps)
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    if config.m < 1:
        raise ValueError("calibration requires config.m >= 1")
    values = np.asarray(samples, dtype=float)
    expected = n_reps * config.m
    if values.ndim != 1 or values.size != expected or not np.all(np.isfinite(values)):
        raise ValueError(f"samples must be a finite one-dimensional array of length {expected}")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("samples must lie in [0, 1]")

    sorted_samples = np.sort(values)
    n_pvalues = int(sorted_samples.size)
    ks = kstest(sorted_samples, "uniform")
    dkw_radius = float(
        math.sqrt(math.log(2.0 / (1.0 - confidence)) / (2.0 * n_pvalues))
    )
    if grid is None:
        grid_values = np.linspace(0.0, 1.0, 101)
    else:
        grid_values = np.asarray(grid, dtype=float)
        if grid_values.ndim != 1 or grid_values.size < 2:
            raise ValueError("grid must contain at least two points")
        if not np.all(np.isfinite(grid_values)) or np.any(
            (grid_values < 0.0) | (grid_values > 1.0)
        ):
            raise ValueError("grid values must be finite and lie in [0, 1]")
        grid_values = np.sort(np.unique(grid_values))
    empirical = np.searchsorted(sorted_samples, grid_values, side="right") / n_pvalues
    return {
        "regime": "global_null_independent",
        "m": int(config.m),
        "n_reps": n_reps,
        "n_pvalues": n_pvalues,
        "seed": int(seed),
        "replication_seed_schedule": {
            "kind": "numpy.SeedSequence",
            "assignment": "one child stream per replication",
            "order": "stable ascending replication order",
        },
        "confidence": confidence,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue_iid_reference": float(ks.pvalue),
        "dkw_radius": dkw_radius,
        "passes_dkw": bool(ks.statistic <= dkw_radius),
        "grid": [float(value) for value in grid_values],
        "empirical_cdf": [float(value) for value in empirical],
    }


def null_pvalue_calibration(
    config: TwoGroupsConfig,
    n_reps: int,
    seed: int = 0,
    confidence: float = 0.95,
    grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Audit marginal p-value calibration under an independent global null.

    The diagnostic replaces the configured mixture by ``pi0=1``, ``effect=0``
    and ``rho=0``. It pools ``m * n_reps`` raw p-values, compares their ECDF to
    the Uniform(0, 1) CDF, and reports a Kolmogorov--Smirnov distance together
    with a Dvoretzky--Kiefer--Wolfowitz radius. This is a finite calibration
    check for the declared independent null regime, not a proof under
    dependence or a guarantee for an arbitrary test statistic.
    """
    if isinstance(n_reps, bool) or int(n_reps) != n_reps or n_reps < 1:
        raise ValueError(f"n_reps must be a positive integer, got {n_reps!r}")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    if config.m < 1:
        raise ValueError("calibration requires config.m >= 1")

    null_config = replace(config, pi0=1.0, effect=0.0, rho=0.0)
    rng = np.random.default_rng(seed)
    samples = np.empty(int(n_reps) * config.m, dtype=float)
    for rep in range(int(n_reps)):
        pvalues, _ = generate_pvalues(null_config, rng)
        start = rep * config.m
        samples[start : start + config.m] = pvalues

    samples.sort()
    n_pvalues = int(samples.size)
    ks = kstest(samples, "uniform")
    dkw_radius = float(
        math.sqrt(math.log(2.0 / (1.0 - confidence)) / (2.0 * n_pvalues))
    )
    if grid is None:
        grid_values = np.linspace(0.0, 1.0, 101)
    else:
        grid_values = np.asarray(grid, dtype=float)
        if grid_values.ndim != 1 or grid_values.size < 2:
            raise ValueError("grid must contain at least two points")
        if not np.all(np.isfinite(grid_values)) or np.any(
            (grid_values < 0.0) | (grid_values > 1.0)
        ):
            raise ValueError("grid values must be finite and lie in [0, 1]")
        grid_values = np.sort(np.unique(grid_values))
    empirical = np.searchsorted(samples, grid_values, side="right") / n_pvalues
    return {
        "regime": "global_null_independent",
        "m": int(config.m),
        "n_reps": int(n_reps),
        "n_pvalues": n_pvalues,
        "seed": int(seed),
        "confidence": confidence,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue_iid_reference": float(ks.pvalue),
        "dkw_radius": dkw_radius,
        "passes_dkw": bool(ks.statistic <= dkw_radius),
        "grid": [float(value) for value in grid_values],
        "empirical_cdf": [float(value) for value in empirical],
    }
