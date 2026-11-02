"""Time-uniform evidence for binary observation streams.

The likelihood-ratio products in this module are nonnegative martingales under
the declared Bernoulli null.  Ville's inequality therefore supplies an
optional-stopping-safe boundary.  The fixed-horizon p-value is retained as a
contrast and is explicitly not time-uniform.

E-processes can grow beyond the largest IEEE-754 float on long, strongly
informative streams.  We therefore evaluate their products in log space.
``EProcessResult.log_e_process_values`` and ``terminal_log_e`` remain the
authoritative representation; ``e_process_values`` and ``terminal_e`` are
finite display values capped at ``np.finfo(float).max``. Boundary crossings are
always decided in log space, never from a capped display value. New public
names use ``e_process_values`` throughout, because online FDR uses *wealth*
for a different alpha-accounting quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import betaln, logsumexp
from scipy.stats import binom



_FLOAT_MAX = float(np.finfo(float).max)
_LOG_FLOAT_MAX = math.log(_FLOAT_MAX)


def _finite_e_process_values_from_log(log_e_process_values: np.ndarray) -> np.ndarray:
    """Return a finite display path for a log-domain e-process.

    The returned path preserves ordinary e-values exactly up to floating-point
    rounding.  Values whose true exponential is too large for a float are
    capped, while the caller retains ``log_e_process_values`` as the evaluable
    source of truth.
    """
    values = np.asarray(log_e_process_values, dtype=float)
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("log_e_process_values must not contain NaN or positive infinity")
    display = np.zeros(values.shape, dtype=float)
    finite = np.isfinite(values)
    display[finite] = np.exp(np.minimum(values[finite], _LOG_FLOAT_MAX))
    display[values >= _LOG_FLOAT_MAX] = _FLOAT_MAX
    return display


@dataclass(frozen=True)
class EProcessResult:
    """Pathwise e-process result with an authoritative log-domain path.

    ``e_process_values`` and ``terminal_e`` are finite display values. When
    :attr:`e_process_values_are_capped` is true, use
    :attr:`log_e_process_values`,
    :attr:`terminal_log_e`, and :attr:`log_boundary` for quantitative
    comparisons; the display values are intentionally capped rather than
    allowed to overflow. Public callers use :attr:`e_process_values` and
    :attr:`log_e_process_values` so this evidence cannot be confused with
    online alpha accounting.
    """

    e_process_values: np.ndarray
    alpha: float
    boundary: float
    crossing_time: int | None
    terminal_e: float
    log_e_process_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        e_process_values = np.asarray(self.e_process_values, dtype=float)
        if (
            e_process_values.ndim != 1
            or e_process_values.size < 1
            or not np.all(np.isfinite(e_process_values))
        ):
            raise ValueError("e_process_values must be a non-empty finite 1-D path")
        if np.any(e_process_values < 0.0):
            raise ValueError("e-process values must be non-negative")
        alpha = _validate_alpha(self.alpha)
        boundary = float(self.boundary)
        if not math.isclose(float(e_process_values[0]), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("e-process values must start at one")
        if self.log_e_process_values is None:
            log_e_process_values = np.full(e_process_values.shape, -math.inf, dtype=float)
            positive = e_process_values > 0.0
            log_e_process_values[positive] = np.log(e_process_values[positive])
        else:
            log_e_process_values = np.asarray(self.log_e_process_values, dtype=float)
            if (
                log_e_process_values.shape != e_process_values.shape
                or log_e_process_values.ndim != 1
            ):
                raise ValueError(
                    "log_e_process_values must be a one-dimensional path aligned with "
                    "e_process_values"
                )
            if np.any(np.isnan(log_e_process_values)) or np.any(
                np.isposinf(log_e_process_values)
            ):
                raise ValueError(
                    "log_e_process_values must not contain NaN or positive infinity"
                )
            expected_e_process_values = _finite_e_process_values_from_log(
                log_e_process_values
            )
            if not np.allclose(
                e_process_values, expected_e_process_values, rtol=1e-12, atol=0.0
            ):
                raise ValueError(
                    "e_process_values must be the finite display of "
                    "log_e_process_values"
                )
        if not math.isclose(
            float(log_e_process_values[0]), 0.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("log e-process values must start at zero")
        log_boundary = -math.log(alpha)
        expected_boundary = float(
            _finite_e_process_values_from_log(np.asarray([log_boundary]))[0]
        )
        if not math.isclose(boundary, expected_boundary, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("boundary must equal the finite display of 1 / alpha")
        if not math.isfinite(float(self.terminal_e)) or not math.isclose(
            float(self.terminal_e), float(e_process_values[-1]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("terminal_e must equal the terminal e-process value")
        hits = np.flatnonzero(log_e_process_values[1:] >= log_boundary)
        if self.crossing_time is not None:
            if isinstance(self.crossing_time, bool) or int(self.crossing_time) != self.crossing_time:
                raise ValueError("crossing_time must be an integer or None")
            crossing_time = int(self.crossing_time)
            if (
                not 1 <= crossing_time < e_process_values.size
                or not hits.size
                or crossing_time != int(hits[0] + 1)
            ):
                raise ValueError("crossing_time must be the first Ville-boundary crossing")
            object.__setattr__(self, "crossing_time", crossing_time)
        elif hits.size:
            raise ValueError("crossing_time is missing despite a Ville-boundary crossing")
        object.__setattr__(self, "e_process_values", e_process_values)
        object.__setattr__(self, "log_e_process_values", log_e_process_values)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "terminal_e", float(self.terminal_e))

    @property
    def crossed(self) -> bool:
        """Whether the process crossed its Ville boundary."""
        return self.crossing_time is not None

    @property
    def log_boundary(self) -> float:
        """Log of the Ville boundary, including when its display is capped."""
        return -math.log(self.alpha)

    @property
    def terminal_log_e(self) -> float:
        """Authoritative log e-value at the terminal observation."""
        assert self.log_e_process_values is not None
        return float(self.log_e_process_values[-1])

    @property
    def e_process_values_are_capped(self) -> bool:
        """Whether one or more finite display values were capped."""
        assert self.log_e_process_values is not None
        return bool(np.any(self.log_e_process_values >= _LOG_FLOAT_MAX))

    @property
    def boundary_is_capped(self) -> bool:
        """Whether the finite display boundary was capped."""
        return self.log_boundary >= _LOG_FLOAT_MAX

    def to_mapping(self) -> dict[str, object]:
        """Serialize the path and its explicit Ville-boundary status.

        Artifact mappings use the canonical ``e_process_values`` names and
        deliberately do not reuse ``wealth`` because online-FDR ``wealth``
        denotes alpha accounting rather than e-process evidence.
        """
        log_e_process_values = self.log_e_process_values
        assert log_e_process_values is not None
        return {
            "e_process_values": self.e_process_values.tolist(),
            "log_e_process_values": log_e_process_values.tolist(),
            "e_process_values_representation": "finite_float_display_with_authoritative_log_e_process_values",
            "e_process_values_display_capped": self.e_process_values_are_capped,
            "evidence_kind": "likelihood_ratio_e_process",
            "alpha": float(self.alpha),
            "boundary": float(self.boundary),
            "log_boundary": self.log_boundary,
            "boundary_display_capped": self.boundary_is_capped,
            "crossing_time": self.crossing_time,
            "terminal_e": float(self.terminal_e),
            "terminal_log_e": self.terminal_log_e,
            "crossed": self.crossed,
            "validity": "Ville-boundary diagnostic under the declared e-process null",
        }


@dataclass(frozen=True)
class ConfidenceSequenceResult:
    """A Bernoulli confidence sequence obtained by e-process inversion."""

    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    alternative_prior: tuple[float, float]
    observations: np.ndarray
    empty: np.ndarray
    validity: str = "time_uniform_e_process_inversion"

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        observations = _validate_observations(self.observations)
        empty = np.asarray(self.empty, dtype=bool)
        if lower.ndim != 1 or upper.shape != lower.shape or empty.shape != lower.shape:
            raise ValueError("confidence-sequence arrays must align")
        if lower.size != observations.size or lower.size < 1:
            raise ValueError("confidence-sequence arrays must align with observations")
        if not np.all(np.isfinite(lower[~empty])) or not np.all(np.isfinite(upper[~empty])):
            raise ValueError("non-empty confidence-sequence endpoints must be finite")
        if np.any(lower[~empty] < 0.0) or np.any(upper[~empty] > 1.0) or np.any(lower[~empty] > upper[~empty]):
            raise ValueError("confidence-sequence endpoints must lie in [0, 1]")
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        a, b = (float(self.alternative_prior[0]), float(self.alternative_prior[1]))
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0 or b <= 0.0:
            raise ValueError("alternative_prior parameters must be positive and finite")
        if not isinstance(self.validity, str) or not self.validity:
            raise ValueError("validity must be a non-empty string")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "empty", empty)
        object.__setattr__(self, "alternative_prior", (a, b))

    @property
    def coverage_claim(self) -> str:
        """Describe the exact scope of the inversion claim."""
        return "simultaneous-in-time Bernoulli coverage under each fixed null rate"

    def to_mapping(self) -> dict[str, object]:
        """Serialize the path without losing the explicit empty-set mask."""
        return {
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "alpha": float(self.alpha),
            "alternative_prior": list(self.alternative_prior),
            "observations": self.observations.tolist(),
            "empty": self.empty.tolist(),
            "validity": self.validity,
            "coverage_claim": self.coverage_claim,
        }


def _validate_observations(observations: Sequence[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(observations)
    if values.ndim != 1:
        raise ValueError("observations must be a one-dimensional sequence")
    if values.size and (not np.all(np.isin(values, (0, 1)))):
        raise ValueError("observations must contain only 0 and 1")
    return values.astype(int, copy=False)


def _validate_rate(rate: float, name: str) -> float:
    rate = float(rate)
    if not math.isfinite(rate) or not 0.0 < rate < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return rate


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return alpha


def _result(log_e_process_values: np.ndarray, alpha: float) -> EProcessResult:
    """Materialize a finite display result from its log-domain path."""
    log_path = np.asarray(log_e_process_values, dtype=float)
    e_process_values = _finite_e_process_values_from_log(log_path)
    log_boundary = -math.log(alpha)
    boundary = float(_finite_e_process_values_from_log(np.asarray([log_boundary]))[0])
    hits = np.flatnonzero(log_path[1:] >= log_boundary)
    crossing_time = int(hits[0] + 1) if hits.size else None
    return EProcessResult(
        e_process_values=e_process_values,
        alpha=alpha,
        boundary=boundary,
        crossing_time=crossing_time,
        terminal_e=float(e_process_values[-1]),
        log_e_process_values=log_path,
    )


def binary_e_process(
    observations: Sequence[int] | np.ndarray,
    null_rate: float,
    alternative_rate: float,
    alpha: float = 0.05,
) -> EProcessResult:
    """Evaluate a Bernoulli likelihood-ratio e-process.

    The alternative is simple and fixed before observing the stream.  Mixture
    alternatives are available through :func:`mixture_e_process`.
    """
    values = _validate_observations(observations)
    q0 = _validate_rate(null_rate, "null_rate")
    q1 = _validate_rate(alternative_rate, "alternative_rate")
    alpha = _validate_alpha(alpha)
    if q0 == q1:
        raise ValueError("alternative_rate must differ from null_rate")
    log_e_process_values = np.zeros(values.size + 1, dtype=float)
    log_success_factor = math.log(q1) - math.log(q0)
    log_failure_factor = math.log1p(-q1) - math.log1p(-q0)
    for index, observation in enumerate(values, start=1):
        log_e_process_values[index] = log_e_process_values[index - 1] + (
            log_success_factor if observation else log_failure_factor
        )
    return _result(log_e_process_values, alpha)


def mixture_e_process(
    observations: Sequence[int] | np.ndarray,
    null_rate: float,
    alternative_rates: Sequence[float],
    weights: Sequence[float] | None = None,
    alpha: float = 0.05,
) -> EProcessResult:
    """Evaluate a fixed mixture of Bernoulli likelihood-ratio e-processes."""
    values = _validate_observations(observations)
    q0 = _validate_rate(null_rate, "null_rate")
    rates = np.asarray(alternative_rates, dtype=float)
    if rates.ndim != 1 or rates.size == 0 or not np.all(np.isfinite(rates)):
        raise ValueError("alternative_rates must be a non-empty finite vector")
    if np.any((rates <= 0.0) | (rates >= 1.0)) or np.any(rates == q0):
        raise ValueError("alternative_rates must lie in (0, 1) and differ from null_rate")
    if weights is None:
        prior: np.ndarray = np.full(rates.size, 1.0 / rates.size)
    else:
        prior = np.asarray(weights, dtype=float)
        if prior.shape != rates.shape or not np.all(np.isfinite(prior)) or np.any(prior < 0.0):
            raise ValueError("weights must be finite, non-negative, and match rates")
        total = float(np.sum(prior))
        if total <= 0.0:
            raise ValueError("weights must have positive sum")
        prior = prior / total
    alpha = _validate_alpha(alpha)
    log_e_process_values = np.zeros(values.size + 1, dtype=float)
    log_prior = np.full(rates.shape, -math.inf, dtype=float)
    positive_prior = prior > 0.0
    log_prior[positive_prior] = np.log(prior[positive_prior])
    log_component = log_prior.copy()
    log_success_factor = np.log(rates) - math.log(q0)
    log_failure_factor = np.log1p(-rates) - math.log1p(-q0)
    for index, observation in enumerate(values, start=1):
        log_component += log_success_factor if observation else log_failure_factor
        log_e_process_values[index] = float(logsumexp(log_component))
    return _result(log_e_process_values, alpha)


def predictable_e_process(
    observations: Sequence[int] | np.ndarray,
    null_rate: float,
    alternative_rates: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> EProcessResult:
    """Evaluate a likelihood-ratio e-process with predictable alternatives.

    ``alternative_rates[t]`` must be fixed using information available before
    observation ``t``.  The function records the resulting path but cannot
    inspect or enforce a caller's filtration; the assumption is therefore
    carried as a validity boundary in generated evidence.
    """
    values = _validate_observations(observations)
    rates = np.asarray(alternative_rates, dtype=float)
    if rates.shape != values.shape or not np.all(np.isfinite(rates)):
        raise ValueError("alternative_rates must be finite and align with observations")
    q0 = _validate_rate(null_rate, "null_rate")
    if np.any((rates <= 0.0) | (rates >= 1.0)) or np.any(rates == q0):
        raise ValueError("alternative_rates must lie in (0, 1) and differ from null_rate")
    alpha = _validate_alpha(alpha)
    log_e_process_values = np.zeros(values.size + 1, dtype=float)
    for index, (observation, rate) in enumerate(zip(values, rates), start=1):
        log_factor = (
            math.log(float(rate)) - math.log(q0)
            if observation
            else math.log1p(-float(rate)) - math.log1p(-q0)
        )
        log_e_process_values[index] = log_e_process_values[index - 1] + log_factor
    return _result(log_e_process_values, alpha)


def _mixture_log_e(null_rate: float, successes: int, trials: int, a: float, b: float) -> float:
    """Log mixture likelihood ratio for a fixed Bernoulli null."""
    return float(
        betaln(successes + a, trials - successes + b)
        - betaln(a, b)
        - successes * math.log(null_rate)
        - (trials - successes) * math.log1p(-null_rate)
    )


def _confidence_endpoint(
    successes: int, trials: int, alpha: float, a: float, b: float, *, lower: bool
) -> float:
    boundary = -math.log(alpha)

    def log_e(rate: float) -> float:
        rate = min(max(float(rate), 1e-12), 1.0 - 1e-12)
        return _mixture_log_e(rate, successes, trials, a, b)

    mode = min(max(successes / trials, 1e-12), 1.0 - 1e-12)
    if log_e(mode) >= boundary:
        return float("nan")
    edge = 1e-12 if lower else 1.0 - 1e-12
    if log_e(edge) <= boundary:
        return 0.0 if lower else 1.0
    if lower:
        return float(brentq(lambda value: log_e(value) - boundary, 1e-12, mode))
    return float(brentq(lambda value: log_e(value) - boundary, mode, 1.0 - 1e-12))


def bernoulli_confidence_sequence(
    observations: Sequence[int] | np.ndarray,
    alpha: float = 0.05,
    alternative_prior: tuple[float, float] = (1.0, 1.0),
) -> ConfidenceSequenceResult:
    """Invert a Beta-mixture Bernoulli e-process at every time point."""
    values = _validate_observations(observations)
    alpha = _validate_alpha(alpha)
    a, b = (float(alternative_prior[0]), float(alternative_prior[1]))
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0 or b <= 0.0:
        raise ValueError("alternative_prior parameters must be positive and finite")
    lower = np.empty(values.size, dtype=float)
    upper = np.empty(values.size, dtype=float)
    empty = np.zeros(values.size, dtype=bool)
    for index in range(1, values.size + 1):
        successes = int(np.sum(values[:index]))
        low = _confidence_endpoint(successes, index, alpha, a, b, lower=True)
        high = _confidence_endpoint(successes, index, alpha, a, b, lower=False)
        if np.isnan(low) or np.isnan(high):
            empty[index - 1] = True
            lower[index - 1] = np.nan
            upper[index - 1] = np.nan
        else:
            lower[index - 1] = low
            upper[index - 1] = high
    return ConfidenceSequenceResult(lower, upper, alpha, (a, b), values, empty)


def fixed_horizon_binomial_pvalue(
    observations: Sequence[int] | np.ndarray,
    null_rate: float,
    alternative_rate: float,
) -> float:
    """Return a fixed-horizon directional Binomial p-value.

    This function is valid only after the horizon is fixed.  It is intentionally
    separate from e-process decisions so optional-stopping misuse is visible.
    """
    values = _validate_observations(observations)
    q0 = _validate_rate(null_rate, "null_rate")
    q1 = _validate_rate(alternative_rate, "alternative_rate")
    count = int(np.sum(values))
    if q1 > q0:
        return float(binom.sf(count - 1, values.size, q0))
    return float(binom.cdf(count, values.size, q0))


__all__ = [
    "EProcessResult", "ConfidenceSequenceResult",
    "binary_e_process",
    "mixture_e_process",
    "predictable_e_process",
    "bernoulli_confidence_sequence",
    "fixed_horizon_binomial_pvalue",
]
