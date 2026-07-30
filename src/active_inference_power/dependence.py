"""Validated Gaussian dependence models for multiple-testing simulations.

The default project model is a positive one-factor construction.  This module
adds explicit covariance-backed regimes so negative and block dependence are
represented only when their correlation matrix is positive semidefinite.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

DependenceKind = Literal["independent", "equicorrelated", "block", "factor"]


@dataclass(frozen=True)
class DependenceConfig:
    """Specification for a standardized Gaussian dependence regime."""

    kind: DependenceKind = "independent"
    rho: float = 0.0
    block_size: int = 1
    loadings: np.ndarray | None = None

    def validate(self, m: int) -> None:
        if isinstance(m, bool) or int(m) != m or int(m) < 1:
            raise ValueError(f"m must be a positive integer, got {m!r}")
        if self.kind not in {"independent", "equicorrelated", "block", "factor"}:
            raise ValueError(f"unknown dependence kind: {self.kind!r}")
        if not np.isfinite(self.rho):
            raise ValueError("rho must be finite")
        if self.kind == "independent" and self.rho != 0.0:
            raise ValueError(
                "rho must be zero for independent dependence; nonzero rho would be ignored"
            )
        if self.kind == "equicorrelated":
            if m == 1:
                if self.rho != 0.0:
                    raise ValueError("one-dimensional equicorrelation must be zero")
            elif not -1.0 / (m - 1) <= self.rho < 1.0:
                raise ValueError("equicorrelation must lie in [-1/(m-1), 1)")
        if self.kind == "block":
            if isinstance(self.block_size, bool) or int(self.block_size) != self.block_size:
                raise ValueError("block_size must be an integer")
            if not 1 <= int(self.block_size) <= m:
                raise ValueError("block_size must lie in [1, m]")
            if int(self.block_size) == 1:
                if self.rho != 0.0:
                    raise ValueError("block rho must be zero for block_size=1")
            elif not -1.0 / (int(self.block_size) - 1) <= self.rho < 1.0:
                raise ValueError("block rho is outside the positive-semidefinite range")
        if self.kind == "factor":
            if self.loadings is None:
                raise ValueError("factor dependence requires loadings")
            loadings = np.asarray(self.loadings, dtype=float)
            if loadings.shape != (m,) or not np.all(np.isfinite(loadings)):
                raise ValueError("factor loadings must be finite with shape (m,)")
            if np.any(np.abs(loadings) >= 1.0):
                raise ValueError("factor loadings must have absolute value < 1")


def correlation_matrix(m: int, config: DependenceConfig) -> np.ndarray:
    """Return the validated correlation matrix for ``config``."""
    config.validate(m)
    m = int(m)
    if config.kind == "independent":
        return np.eye(m)
    if config.kind == "factor":
        loading = np.asarray(config.loadings, dtype=float)
        matrix = np.outer(loading, loading)
        np.fill_diagonal(matrix, 1.0)
        return matrix
    if config.kind == "equicorrelated":
        return (1.0 - config.rho) * np.eye(m) + config.rho * np.ones((m, m))
    matrix = np.eye(m)
    size = int(config.block_size)
    for start in range(0, m, size):
        stop = min(start + size, m)
        block = (1.0 - config.rho) * np.eye(stop - start) + config.rho * np.ones(
            (stop - start, stop - start)
        )
        matrix[start:stop, start:stop] = block
    return matrix


def sample_gaussian(
    rng: np.random.Generator,
    m: int,
    config: DependenceConfig | None = None,
    mean: np.ndarray | None = None,
) -> np.ndarray:
    """Draw one standardized Gaussian vector under a declared regime."""
    config = config or DependenceConfig()
    matrix = correlation_matrix(m, config)
    if config.kind != "factor":
        factor = _cached_factor(int(m), config.kind, float(config.rho), int(config.block_size))
        draw = factor @ rng.standard_normal(int(m))
        if mean is not None:
            mean_array = np.asarray(mean, dtype=float)
            if mean_array.shape != (int(m),):
                raise ValueError("mean must have shape (m,)")
            draw = draw + mean_array
        return np.asarray(draw, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if float(np.min(eigenvalues)) < -1e-10:
        raise ValueError("dependence correlation matrix is not positive semidefinite")
    factor = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))
    draw = factor @ rng.standard_normal(int(m))
    if mean is not None:
        mean_array = np.asarray(mean, dtype=float)
        if mean_array.shape != (int(m),):
            raise ValueError("mean must have shape (m,)")
        draw = draw + mean_array
    return np.asarray(draw, dtype=float)


@lru_cache(maxsize=64)
def _cached_factor(m: int, kind: DependenceKind, rho: float, block_size: int) -> np.ndarray:
    """Cache spectral factors so large Monte-Carlo runs do not diagonalize per draw."""
    matrix = correlation_matrix(m, DependenceConfig(kind=kind, rho=rho, block_size=block_size))
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if float(np.min(eigenvalues)) < -1e-10:
        raise ValueError("dependence correlation matrix is not positive semidefinite")
    return eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))


__all__ = ["DependenceConfig", "DependenceKind", "correlation_matrix", "sample_gaussian"]
