"""Auditable finite-horizon LORD++ and conservatively capped SAFFRON traces.

The implementation retains the published history-dependent recursions while
applying the package's shared finite-level cap to SAFFRON.  It keeps the
candidate/rejection history and an explicit alpha-wealth accounting path so a
generated trace can be audited. Any validity statement remains conditional on
the published assumptions and the declared capped implementation; the
project's simulations are finite calibration diagnostics, not proofs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, cast

import numpy as np


@dataclass(frozen=True)
class OnlineFDRResult:
    """Trace emitted by an online FDR procedure.

    ``wealth`` is the procedure's non-negative alpha-budget accounting path:
    it starts at ``initial_wealth`` and records spending/rewards after each
    test.  It is diagnostic state, while ``levels`` are the actual predictable
    test thresholds.
    """

    method: str
    pvalues: np.ndarray
    levels: np.ndarray
    candidates: np.ndarray
    rejections: np.ndarray
    wealth: np.ndarray
    alpha: float
    gamma: np.ndarray
    lambda_: float | None = None
    initial_wealth: float = 0.0

    def __post_init__(self) -> None:
        pvalues = np.asarray(self.pvalues, dtype=float)
        levels = np.asarray(self.levels, dtype=float)
        candidates = np.asarray(self.candidates, dtype=bool)
        rejections = np.asarray(self.rejections, dtype=bool)
        wealth = np.asarray(self.wealth, dtype=float)
        gamma = np.asarray(self.gamma, dtype=float)
        if pvalues.ndim != 1 or pvalues.size < 1:
            raise ValueError("pvalues must be a non-empty 1-D vector")
        if self.method not in {"lord_plus_plus", "saffron"}:
            raise ValueError("method must be lord_plus_plus or saffron")
        if any(array.shape != pvalues.shape for array in (levels, candidates, rejections, gamma)):
            raise ValueError("online FDR trace arrays must align with pvalues")
        if wealth.shape != (pvalues.size + 1,):
            raise ValueError("wealth must have one initial value plus one value per test")
        if (
            not np.all(np.isfinite(pvalues))
            or np.any((pvalues < 0.0) | (pvalues > 1.0))
            or not np.all(np.isfinite(levels))
            or np.any(levels < 0.0)
            or not np.all(np.isfinite(gamma))
            or np.any(gamma < 0.0)
            or not np.all(np.isfinite(wealth))
            or np.any(wealth < -1e-12)
        ):
            raise ValueError("online FDR trace values are outside their valid ranges")
        alpha = _validate_alpha(self.alpha)
        schedule = _validate_gamma(gamma, pvalues.shape)
        if np.any(levels > alpha + 1e-12):
            raise ValueError("online FDR levels cannot exceed alpha")
        if not math.isfinite(self.initial_wealth) or not 0.0 <= self.initial_wealth <= alpha:
            raise ValueError("initial_wealth must lie in [0, alpha]")
        if not math.isclose(float(wealth[0]), float(self.initial_wealth), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("wealth[0] must equal initial_wealth")
        if self.method == "lord_plus_plus":
            if self.lambda_ is not None:
                raise ValueError("LORD++ does not accept lambda_")
            if not np.all(candidates):
                raise ValueError("LORD++ candidate indicators must all be true")
        else:
            if self.lambda_ is None:
                raise ValueError("SAFFRON requires lambda_")
            lambda_ = float(self.lambda_)
            if not math.isfinite(lambda_) or not 0.0 < lambda_ < 1.0:
                raise ValueError("lambda_ must lie in (0, 1)")
            if not np.array_equal(candidates, pvalues <= lambda_):
                raise ValueError("SAFFRON candidates must equal pvalues <= lambda_")
        expected_rejections = pvalues <= levels
        if not np.array_equal(rejections, expected_rejections):
            raise ValueError("rejections must equal pvalues <= levels")
        expected_levels = _expected_online_levels(
            self.method,
            pvalues,
            alpha,
            schedule,
            candidates,
            rejections,
            None if self.lambda_ is None else float(self.lambda_),
            float(self.initial_wealth),
        )
        if not np.allclose(levels, expected_levels, rtol=0.0, atol=1e-12):
            raise ValueError("levels do not match the declared online-FDR recursion")
        # The SAFFRON branch has already rejected a missing lambda above.  The
        # cast keeps that invariant visible to static type checkers without
        # weakening the runtime validation of the serialized result.
        upper = (
            alpha
            if self.method == "lord_plus_plus"
            else (1.0 - float(cast(float, self.lambda_))) * alpha
        )
        expected_wealth = _wealth_path(
            levels,
            rejections,
            float(self.initial_wealth),
            upper - float(self.initial_wealth),
            upper,
            costs=None if self.method == "lord_plus_plus" else np.where(candidates, 0.0, levels),
        )
        if not np.allclose(wealth, expected_wealth, rtol=0.0, atol=1e-12):
            raise ValueError("wealth does not match the declared online-FDR accounting")
        object.__setattr__(self, "pvalues", pvalues)
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "rejections", rejections)
        object.__setattr__(self, "wealth", wealth)
        object.__setattr__(self, "gamma", gamma)

    @property
    def num_rejections(self) -> int:
        """Number of rejected hypotheses in the observed finite stream."""
        return int(np.count_nonzero(self.rejections))

    def to_mapping(self) -> dict[str, object]:
        """Serialize the predictable online-FDR trace with explicit accounting semantics."""
        return {
            "method": self.method,
            "pvalues": self.pvalues.tolist(),
            "levels": self.levels.tolist(),
            "candidates": self.candidates.tolist(),
            "rejections": self.rejections.tolist(),
            "accounting_path": self.wealth.tolist(),
            "accounting_kind": "procedural_alpha_wealth",
            "alpha": float(self.alpha),
            "gamma": self.gamma.tolist(),
            "lambda": None if self.lambda_ is None else float(self.lambda_),
            "initial_wealth": float(self.initial_wealth),
            "num_rejections": self.num_rejections,
        }


@dataclass(frozen=True)
class EValueLONDResult:
    """Trace for the e-LOND thresholding rule.

    The e-value decision is ``e_t >= 1 / alpha_t``.  The schedule is
    predictable because it depends only on earlier rejection times.
    ``cumulative_allocations`` is the cumulative sum of nominal levels
    emitted by this recycled recursion. It is neither alpha-wealth nor a
    remaining alpha budget: levels can be recycled after a rejection and this
    trace can exceed ``alpha``.
    """

    evalues: np.ndarray
    levels: np.ndarray
    thresholds: np.ndarray
    rejections: np.ndarray
    cumulative_allocations: np.ndarray
    alpha: float
    gamma: np.ndarray

    def __post_init__(self) -> None:
        evalues = np.asarray(self.evalues, dtype=float)
        levels = np.asarray(self.levels, dtype=float)
        thresholds = np.asarray(self.thresholds, dtype=float)
        rejections = np.asarray(self.rejections, dtype=bool)
        cumulative_allocations = np.asarray(self.cumulative_allocations, dtype=float)
        gamma = np.asarray(self.gamma, dtype=float)
        if evalues.ndim != 1 or evalues.size < 1:
            raise ValueError("evalues must be a non-empty 1-D vector")
        if any(array.shape != evalues.shape for array in (levels, thresholds, rejections, gamma)):
            raise ValueError("e-LOND arrays must align with evalues")
        if cumulative_allocations.shape != (evalues.size + 1,):
            raise ValueError(
                "e-LOND cumulative allocations must have one initial value plus one value per e-value"
            )
        if not np.all(np.isfinite(evalues)) or np.any(evalues < 0.0):
            raise ValueError("evalues must be finite and non-negative")
        alpha = _validate_alpha(self.alpha)
        if not np.all(np.isfinite(gamma)) or np.any(gamma <= 0.0) or float(np.sum(gamma)) > 1.0 + 1e-12:
            raise ValueError("e-LOND gamma must be positive with sum at most one")
        if gamma.size > 1 and np.any(gamma[1:] > gamma[:-1] + 1e-12):
            raise ValueError("e-LOND gamma must be non-increasing")
        if not np.all(np.isfinite(levels)) or np.any(levels <= 0.0) or np.any(levels > alpha + 1e-12):
            raise ValueError("e-LOND levels must lie in (0, alpha]")
        if not np.all(np.isfinite(thresholds)) or np.any(thresholds <= 0.0):
            raise ValueError("e-LOND thresholds must be positive")
        if not np.all(np.isfinite(cumulative_allocations)) or np.any(
            cumulative_allocations < -1e-12
        ):
            raise ValueError("e-LOND cumulative allocations must be finite and non-negative")
        if not np.allclose(thresholds, 1.0 / levels, rtol=0.0, atol=1e-12):
            raise ValueError("e-LOND thresholds must equal 1 / levels")
        if not np.array_equal(rejections, evalues >= thresholds):
            raise ValueError("e-LOND rejections must equal evalues >= thresholds")
        expected_levels = _expected_e_lond_levels(alpha, gamma, rejections)
        if not np.allclose(levels, expected_levels, rtol=0.0, atol=1e-12):
            raise ValueError("e-LOND levels must match the declared e-LOND recursion")
        expected_allocations = np.concatenate(([0.0], np.cumsum(levels)))
        if not np.allclose(
            cumulative_allocations, expected_allocations, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "e-LOND cumulative allocations must equal the cumulative nominal levels"
            )
        object.__setattr__(self, "evalues", evalues)
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "rejections", rejections)
        object.__setattr__(self, "cumulative_allocations", cumulative_allocations)
        object.__setattr__(self, "gamma", gamma)

    @property
    def num_rejections(self) -> int:
        return int(np.count_nonzero(self.rejections))

    def to_mapping(self) -> dict[str, object]:
        """Serialize the e-value trace for result artifacts and figure inputs."""
        return {
            "evalues": self.evalues.tolist(),
            "levels": self.levels.tolist(),
            "thresholds": self.thresholds.tolist(),
            "rejections": self.rejections.tolist(),
            "accounting_path": self.cumulative_allocations.tolist(),
            "alpha": float(self.alpha),
            "gamma": self.gamma.tolist(),
            "num_rejections": self.num_rejections,
            "accounting_kind": "cumulative_nominal_level_allocation",
            "validity": "conditional e-value filtration accounting",
        }


def default_gamma(horizon: int, exponent: float = 1.6) -> np.ndarray:
    """Return a positive, normalized, non-increasing gamma sequence."""
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    exponent = float(exponent)
    if not math.isfinite(exponent) or exponent <= 1.0:
        raise ValueError("exponent must be finite and greater than one")
    values = 1.0 / np.power(np.arange(1, int(horizon) + 1, dtype=float), exponent)
    return values / np.sum(values)


def _validate_pvalues(pvalues: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("pvalues must be a non-empty finite vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("pvalues must lie in [0, 1]")
    return values


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return alpha


def _validate_schedule(
    pvalues: np.ndarray, alpha: float, gamma: Sequence[float] | None
) -> tuple[float, np.ndarray]:
    alpha = _validate_alpha(alpha)
    schedule = default_gamma(pvalues.size) if gamma is None else np.asarray(gamma, dtype=float)
    if schedule.shape != pvalues.shape or not np.all(np.isfinite(schedule)):
        raise ValueError("gamma must be finite and have one entry per p-value")
    if np.any(schedule < 0.0) or float(np.sum(schedule)) > 1.0 + 1e-12:
        raise ValueError("gamma must be non-negative with sum at most one")
    if schedule.size > 1 and np.any(schedule[1:] > schedule[:-1] + 1e-12):
        raise ValueError("gamma must be non-increasing")
    return alpha, schedule


def _validate_initial_wealth(value: float | None, default: float, upper: float, name: str) -> float:
    wealth = default if value is None else float(value)
    if not math.isfinite(wealth) or not 0.0 <= wealth <= upper:
        raise ValueError(f"{name} must lie in [0, {upper}]")
    return wealth


def _gamma_at(gamma: np.ndarray, index: int) -> float:
    """Return gamma at one-based index, with non-positive indices contributing zero."""
    if index <= 0:
        return 0.0
    return float(gamma[index - 1])


def _validate_gamma(gamma: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Validate a finite, non-increasing gamma schedule for a result trace."""
    if gamma.shape != shape or not np.all(np.isfinite(gamma)) or np.any(gamma < 0.0):
        raise ValueError("gamma must be finite, non-negative, and align with pvalues")
    if float(np.sum(gamma)) > 1.0 + 1e-12:
        raise ValueError("gamma must have sum at most one")
    if gamma.size > 1 and np.any(gamma[1:] > gamma[:-1] + 1e-12):
        raise ValueError("gamma must be non-increasing")
    return gamma


def _expected_online_levels(
    method: str,
    pvalues: np.ndarray,
    alpha: float,
    gamma: np.ndarray,
    candidates: np.ndarray,
    rejections: np.ndarray,
    lambda_: float | None,
    initial_wealth: float,
) -> np.ndarray:
    """Recompute the finite-horizon threshold recursion for validation.

    The package serializes every online-FDR level under the common finite
    threshold contract ``0 <= alpha_t <= alpha``.  SAFFRON therefore retains
    its candidate-screen cap while also applying the conservative declared
    family-level cap.  This is an implementation contract for the auditable
    finite trace, not a general complexity or validity statement about
    SAFFRON.
    """
    levels = np.empty(pvalues.size, dtype=float)
    rejection_times = np.flatnonzero(rejections) + 1
    for time in range(1, pvalues.size + 1):
        if method == "lord_plus_plus":
            level = initial_wealth * _gamma_at(gamma, time)
            if rejection_times.size:
                level += (alpha - initial_wealth) * _gamma_at(gamma, time - int(rejection_times[0]))
                level += sum(
                    alpha * _gamma_at(gamma, time - int(rejection))
                    for rejection in rejection_times[1:]
                )
            levels[time - 1] = min(alpha, level)
            continue
        if lambda_ is None:
            raise ValueError("SAFFRON level recursion requires lambda_")
        upper = (1.0 - lambda_) * alpha
        prior_candidates = np.flatnonzero(candidates[: time - 1]) + 1
        level = initial_wealth * _gamma_at(gamma, time - int(prior_candidates.size))
        if rejection_times.size:
            first = int(rejection_times[0])
            after_first = int(np.count_nonzero(candidates[first : time - 1]))
            level += (upper - initial_wealth) * _gamma_at(gamma, time - first - after_first)
            for rejection in rejection_times[1:]:
                rejection = int(rejection)
                after_rejection = int(np.count_nonzero(candidates[rejection : time - 1]))
                level += upper * _gamma_at(gamma, time - rejection - after_rejection)
        levels[time - 1] = min(alpha, lambda_, level)
    return levels


def _expected_e_lond_levels(
    alpha: float, gamma: np.ndarray, rejections: np.ndarray
) -> np.ndarray:
    """Recompute the predictable e-LOND levels from prior rejections only.

    With one-based indexing, e-LOND uses
    ``alpha_t = alpha * gamma[t - R_(t-1)]``.  The result validator calls this
    helper after independently checking the serialized rejection indicators,
    so an otherwise range-valid trace cannot substitute arbitrary levels.
    """
    levels = np.empty(rejections.size, dtype=float)
    rejection_count = 0
    for time in range(1, rejections.size + 1):
        levels[time - 1] = alpha * _gamma_at(gamma, time - rejection_count)
        if rejections[time - 1]:
            rejection_count += 1
    return levels


def _wealth_path(
    levels: np.ndarray,
    rejections: np.ndarray,
    initial_wealth: float,
    first_reward: float,
    later_reward: float,
    *,
    costs: np.ndarray | None = None,
) -> np.ndarray:
    """Build a non-negative accounting path for a rejection-reward schedule."""
    path = np.empty(levels.size + 1, dtype=float)
    path[0] = initial_wealth
    seen = 0
    spending = levels if costs is None else costs
    for index, (level_cost, rejected) in enumerate(zip(spending, rejections), start=1):
        reward = 0.0
        if rejected:
            reward = first_reward if seen == 0 else later_reward
            seen += 1
        path[index] = max(0.0, path[index - 1] - float(level_cost) + reward)
    return path


def lord_plus_plus(
    pvalues: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
    gamma: Sequence[float] | None = None,
    *,
    w0: float | None = None,
) -> OnlineFDRResult:
    """Run the finite-horizon LORD++ schedule.

    With rejection times ``tau_j`` and ``w0 <= alpha``, the level is
    ``w0*gamma_t + (alpha-w0)*gamma_{t-tau_1} +
    alpha*sum_{j>=2} gamma_{t-tau_j}``, using one-based gamma indices.
    """
    values = _validate_pvalues(pvalues)
    alpha, schedule = _validate_schedule(values, alpha, gamma)
    initial = _validate_initial_wealth(w0, alpha / 2.0, alpha, "w0")
    levels = np.empty(values.size, dtype=float)
    rejections = np.zeros(values.size, dtype=bool)
    rejection_times: list[int] = []
    for time in range(1, values.size + 1):
        level = initial * _gamma_at(schedule, time)
        if rejection_times:
            level += (alpha - initial) * _gamma_at(schedule, time - rejection_times[0])
            level += sum(
                alpha * _gamma_at(schedule, time - rejection)
                for rejection in rejection_times[1:]
            )
        levels[time - 1] = min(alpha, level)
        rejections[time - 1] = values[time - 1] <= levels[time - 1]
        if rejections[time - 1]:
            rejection_times.append(time)
    wealth = _wealth_path(
        levels,
        rejections,
        initial,
        alpha - initial,
        alpha,
    )
    return OnlineFDRResult(
        "lord_plus_plus", values, levels, np.ones(values.size, dtype=bool), rejections,
        wealth, alpha, schedule, None, initial,
    )


def saffron(
    pvalues: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
    gamma: Sequence[float] | None = None,
    lambda_: float = 0.5,
    *,
    w0: float | None = None,
) -> OnlineFDRResult:
    """Run the finite-horizon SAFFRON candidate-indexed schedule.

    Candidates are p-values at most ``lambda_``.  The default ``w0=alpha/2``
    is valid for the default ``lambda_=1/2``; callers may provide any
    ``0 <= w0 <= (1-lambda_) alpha`` accepted by the declared finite trace.
    The serialized package trace additionally caps each predictable threshold
    at ``min(alpha, lambda_)``.  That conservative cap keeps every generated
    level inside the shared finite online-FDR contract; it does not broaden
    the procedure's theorem-level assumptions.
    """
    values = _validate_pvalues(pvalues)
    alpha, schedule = _validate_schedule(values, alpha, gamma)
    lambda_ = float(lambda_)
    if not math.isfinite(lambda_) or not 0.0 < lambda_ < 1.0:
        raise ValueError("lambda_ must lie in (0, 1)")
    upper = (1.0 - lambda_) * alpha
    initial = _validate_initial_wealth(w0, alpha / 2.0, upper, "w0")
    candidates = values <= lambda_
    levels = np.empty(values.size, dtype=float)
    rejections = np.zeros(values.size, dtype=bool)
    rejection_times: list[int] = []

    for time in range(1, values.size + 1):
        prior_candidates = np.flatnonzero(candidates[: time - 1]) + 1
        base = initial * _gamma_at(schedule, time - int(prior_candidates.size))
        if rejection_times:
            first = rejection_times[0]
            after_first = int(np.count_nonzero(candidates[first : time - 1]))
            base += (upper - initial) * _gamma_at(schedule, time - first - after_first)
            for rejection in rejection_times[1:]:
                after_rejection = int(np.count_nonzero(candidates[rejection : time - 1]))
                base += upper * _gamma_at(schedule, time - rejection - after_rejection)
        levels[time - 1] = min(alpha, lambda_, base)
        rejections[time - 1] = values[time - 1] <= levels[time - 1]
        if rejections[time - 1]:
            rejection_times.append(time)

    # Candidate tests do not consume the SAFFRON null-budget; non-candidates
    # spend their allocated level.  The first rejection restores the gap to
    # the target candidate-budget and later rejections restore the full budget.
    costs = np.where(candidates, 0.0, levels)
    wealth = _wealth_path(levels, rejections, initial, upper - initial, upper, costs=costs)
    return OnlineFDRResult(
        "saffron", values, levels, candidates, rejections,
        wealth, alpha, schedule, lambda_, initial,
    )


def e_lond(
    evalues: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
    gamma: Sequence[float] | None = None,
) -> EValueLONDResult:
    """Apply the predictable e-LOND threshold schedule to e-values.

    At time ``t``, ``alpha_t = alpha * gamma[t - R_(t-1)]`` with the
    one-based convention used by the online-FDR literature.  Rejection occurs
    when ``e_t >= 1 / alpha_t``.  Validity requires the e-values to be
    conditionally calibrated with respect to the arrival filtration.
    """
    values = np.asarray(evalues, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("evalues must be a non-empty finite non-negative vector")
    alpha = _validate_alpha(alpha)
    schedule = default_gamma(values.size) if gamma is None else np.asarray(gamma, dtype=float)
    if schedule.shape != values.shape or not np.all(np.isfinite(schedule)) or np.any(schedule < 0.0):
        raise ValueError("gamma must be finite, non-negative, and align with evalues")
    if float(np.sum(schedule)) > 1.0 + 1e-12:
        raise ValueError("gamma must have sum at most one")
    if schedule.size > 1 and np.any(schedule[1:] > schedule[:-1] + 1e-12):
        raise ValueError("gamma must be non-increasing")
    levels = np.empty(values.size, dtype=float)
    thresholds = np.empty(values.size, dtype=float)
    rejections = np.zeros(values.size, dtype=bool)
    rejection_count = 0
    for time in range(1, values.size + 1):
        index = time - rejection_count
        level = alpha * _gamma_at(schedule, index)
        if level <= 0.0:
            raise ValueError("gamma must allocate positive mass over the observed horizon")
        levels[time - 1] = level
        thresholds[time - 1] = 1.0 / level
        rejections[time - 1] = values[time - 1] >= thresholds[time - 1]
        if rejections[time - 1]:
            rejection_count += 1
    # This is the cumulative emitted nominal level trace, including levels
    # recycled after rejections. It is not alpha-wealth or a remaining budget.
    cumulative_allocations = np.concatenate(([0.0], np.cumsum(levels)))
    return EValueLONDResult(
        values,
        levels,
        thresholds,
        rejections,
        cumulative_allocations,
        alpha,
        schedule,
    )


__all__ = ["OnlineFDRResult", "EValueLONDResult", "default_gamma", "lord_plus_plus", "saffron", "e_lond"]
