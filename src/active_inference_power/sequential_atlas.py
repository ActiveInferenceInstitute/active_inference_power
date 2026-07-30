"""Worker-invariant finite diagnostics for sequential Bernoulli evidence.

This module compares three predeclared likelihood-ratio e-process families on
the *same* independently generated Bernoulli streams: a fixed/simple
alternative, a fixed mixture, and a time-predeclared predictable alternative.
It also records confidence-sequence coverage, e-LOND's cumulative
nominal-level-allocation trace,
and the deliberately invalid practice of repeatedly peeking at a
fixed-horizon Binomial p-value.

The returned mapping is a finite Monte-Carlo diagnostic.  In particular, an
observed Ville-boundary crossing rate or confidence-sequence coverage rate is
not a proof of the corresponding theorem.  The theorem-level boundaries stay
visible in the result and require the declared Bernoulli null, nonnegative
e-process construction, and (for predictable/e-LOND procedures) the stated
filtration conditions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np

from .online_fdr import e_lond
from .sequential import (
    bernoulli_confidence_sequence,
    binary_e_process,
    fixed_horizon_binomial_pvalue,
    mixture_e_process,
    predictable_e_process,
)
from .uncertainty import wilson_interval


__all__ = [
    "SequentialEvidenceAtlasConfig",
    "evaluate_sequential_evidence_atlas",
]


_FAMILY_NAMES = ("fixed_simple", "mixture", "predictable")


def _hash_mapping(value: Any) -> str:
    """Return the canonical SHA-256 digest of a JSON-shaped mapping."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _rate(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 < numeric < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return numeric


def _rate_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    rates = tuple(_rate(value, name) for value in values)
    if not rates:
        raise ValueError(f"{name} must be non-empty")
    if len(set(rates)) != len(rates):
        raise ValueError(f"{name} must contain unique rates")
    return rates


@dataclass(frozen=True)
class SequentialEvidenceAtlasConfig:
    """Frozen design for a finite sequential-evidence comparison.

    Every declared alternative must differ from every declared null because
    each null/alternative pair is evaluated.  ``mixture_alternative_rates``
    is a global candidate grid: a rate equal to a cell's null is removed for
    that cell before the mixture is constructed.  A deterministic companion
    rate is added when needed, so the reported mixture always has at least two
    distinct components.
    """

    null_rates: tuple[float, ...]
    alternative_rates: tuple[float, ...]
    horizon: int
    n_reps: int
    alpha: float = 0.05
    confidence: float = 0.95
    mixture_alternative_rates: tuple[float, ...] = (0.25, 0.75)

    def __post_init__(self) -> None:
        null_rates = _rate_tuple(self.null_rates, "null_rates")
        alternative_rates = _rate_tuple(self.alternative_rates, "alternative_rates")
        mixture_rates = _rate_tuple(
            self.mixture_alternative_rates, "mixture_alternative_rates"
        )
        if any(alternative == null for null in null_rates for alternative in alternative_rates):
            raise ValueError("alternative_rates must differ from every declared null rate")
        horizon = _positive_integer(self.horizon, "horizon")
        n_reps = _positive_integer(self.n_reps, "n_reps")
        alpha = _rate(self.alpha, "alpha")
        confidence = _rate(self.confidence, "confidence")
        object.__setattr__(self, "null_rates", null_rates)
        object.__setattr__(self, "alternative_rates", alternative_rates)
        object.__setattr__(self, "mixture_alternative_rates", mixture_rates)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "n_reps", n_reps)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "confidence", confidence)

    def to_mapping(self) -> dict[str, object]:
        """Serialize the full, assumption-labeled comparison design."""
        return {
            "null_rates": list(self.null_rates),
            "alternative_rates": list(self.alternative_rates),
            "horizon": self.horizon,
            "n_reps": self.n_reps,
            "alpha": self.alpha,
            "confidence": self.confidence,
            "mixture_alternative_rates": list(self.mixture_alternative_rates),
            "data_generating_regime": "independent Bernoulli streams at declared fixed rates",
            "families": {
                "fixed_simple": "one alternative rate fixed before the stream",
                "mixture": "equal-weight fixed mixture over predeclared alternatives",
                "predictable": "time-predeclared alternating alternatives; predictability is a declared filtration assumption",
            },
            "fixed_horizon_contrast": "repeated prefix testing is retained only as an invalid optional-stopping contrast",
        }

    @property
    def configuration_hash(self) -> str:
        """Return a content hash that deliberately excludes worker count."""
        return _hash_mapping(self.to_mapping())


@dataclass(frozen=True)
class _Condition:
    null_rate: float
    true_rate: float
    truth_regime: str
    alternative_rate: float

    @property
    def identifier(self) -> str:
        return (
            f"null={self.null_rate:.12g}|truth={self.true_rate:.12g}"
            f"|regime={self.truth_regime}|alternative={self.alternative_rate:.12g}"
        )


@dataclass(frozen=True)
class _Replication:
    crossing_times: tuple[int | None, int | None, int | None]
    terminal_log_e: tuple[float, float, float]
    confidence_sequence_covered: bool
    confidence_sequence_empty: bool
    fixed_horizon_peek_crossing_time: int | None
    e_lond_num_rejections: int
    e_lond_trace: dict[str, object]


def _conditions(config: SequentialEvidenceAtlasConfig) -> list[_Condition]:
    conditions: list[_Condition] = []
    for null_rate in config.null_rates:
        for alternative_rate in config.alternative_rates:
            conditions.append(_Condition(null_rate, null_rate, "declared_null", alternative_rate))
            conditions.append(
                _Condition(null_rate, alternative_rate, "declared_alternative", alternative_rate)
            )
    return conditions


def _companion_rate(null_rate: float, alternative_rate: float) -> float:
    """Choose one deterministic, distinct rate for a two-component design."""
    candidates = (1.0 - alternative_rate, 0.2, 0.35, 0.65, 0.8)
    for candidate in candidates:
        if 0.0 < candidate < 1.0 and candidate != null_rate and candidate != alternative_rate:
            return float(candidate)
    raise RuntimeError("unable to construct a distinct predictable alternative")


def _mixture_rates(
    null_rate: float,
    alternative_rate: float,
    candidates: tuple[float, ...],
) -> tuple[float, ...]:
    """Build an equal-weight fixed mixture valid for one declared null."""
    rates: list[float] = [alternative_rate]
    for candidate in candidates:
        if candidate != null_rate and candidate not in rates:
            rates.append(candidate)
    if len(rates) == 1:
        rates.append(_companion_rate(null_rate, alternative_rate))
    return tuple(rates)


def _predictable_rates(horizon: int, null_rate: float, alternative_rate: float) -> np.ndarray:
    """Return a schedule fixed by time alone before any observations arrive."""
    companion = _companion_rate(null_rate, alternative_rate)
    return np.resize(np.asarray((alternative_rate, companion), dtype=float), horizon)


def _fixed_horizon_peek_crossing_time(
    observations: np.ndarray, null_rate: float, alternative_rate: float, alpha: float
) -> int | None:
    """Deliberately repeat fixed-horizon tests to expose the invalid contrast."""
    for time in range(1, observations.size + 1):
        if fixed_horizon_binomial_pvalue(observations[:time], null_rate, alternative_rate) <= alpha:
            return time
    return None


def _simple_evalues(
    observations: np.ndarray, null_rate: float, alternative_rate: float
) -> np.ndarray:
    """Return one-step likelihood-ratio e-values for e-LOND accounting."""
    return np.where(
        observations == 1,
        alternative_rate / null_rate,
        (1.0 - alternative_rate) / (1.0 - null_rate),
    ).astype(float, copy=False)


def _run_replication(
    condition: _Condition,
    config: SequentialEvidenceAtlasConfig,
    child: np.random.SeedSequence,
) -> _Replication:
    """Run one independent child stream shared by all compared families."""
    rng = np.random.default_rng(child)
    observations = rng.binomial(1, condition.true_rate, size=config.horizon).astype(int, copy=False)
    mixture_rates = _mixture_rates(
        condition.null_rate, condition.alternative_rate, config.mixture_alternative_rates
    )
    predictable_rates = _predictable_rates(
        config.horizon, condition.null_rate, condition.alternative_rate
    )
    simple = binary_e_process(
        observations, condition.null_rate, condition.alternative_rate, config.alpha
    )
    mixture = mixture_e_process(observations, condition.null_rate, mixture_rates, alpha=config.alpha)
    predictable = predictable_e_process(
        observations, condition.null_rate, predictable_rates, config.alpha
    )
    confidence_sequence = bernoulli_confidence_sequence(observations, config.alpha)
    covered = bool(
        np.all(
            ~confidence_sequence.empty
            & (confidence_sequence.lower <= condition.true_rate)
            & (condition.true_rate <= confidence_sequence.upper)
        )
    )
    trace = e_lond(
        _simple_evalues(observations, condition.null_rate, condition.alternative_rate),
        alpha=config.alpha,
    )
    return _Replication(
        crossing_times=(simple.crossing_time, mixture.crossing_time, predictable.crossing_time),
        terminal_log_e=(simple.terminal_log_e, mixture.terminal_log_e, predictable.terminal_log_e),
        confidence_sequence_covered=covered,
        confidence_sequence_empty=bool(np.any(confidence_sequence.empty)),
        fixed_horizon_peek_crossing_time=_fixed_horizon_peek_crossing_time(
            observations, condition.null_rate, condition.alternative_rate, config.alpha
        ),
        e_lond_num_rejections=trace.num_rejections,
        e_lond_trace=trace.to_mapping(),
    )


def _proportion_summary(
    values: Sequence[bool], confidence: float) -> dict[str, object]:
    """Summarize independent binary replications without overstating scope."""
    n = len(values)
    successes = int(sum(values))
    lower, upper = wilson_interval(successes, n, confidence)
    return {
        "n_reps": n,
        "events": successes,
        "rate": successes / n,
        "wilson_interval": [lower, upper],
        "confidence": confidence,
        "interpretation": "finite Monte-Carlo proportion diagnostic, not a theorem proof",
    }


def _stopping_summary(
    crossing_times: Sequence[int | None], horizon: int, confidence: float
) -> dict[str, object]:
    """Summarize a boundary-crossing stopping-time distribution."""
    crossed = [time for time in crossing_times if time is not None]
    counts: list[dict[str, int | None]] = [
        {"time": time, "count": int(sum(observed == time for observed in crossed))}
        for time in range(1, horizon + 1)
        if any(observed == time for observed in crossed)
    ]
    counts.append({"time": None, "count": len(crossing_times) - len(crossed)})
    return {
        "crossing": _proportion_summary([time is not None for time in crossing_times], confidence),
        "stopping_time_distribution": counts,
        "mean_crossing_time_given_crossed": (
            float(np.mean(crossed)) if crossed else None
        ),
    }


def _family_summary(
    family: str,
    replications: Sequence[_Replication],
    horizon: int,
    confidence: float,
) -> dict[str, object]:
    """Summarize one e-process family from the common-random-number runs."""
    index = _FAMILY_NAMES.index(family)
    terminal = np.asarray([result.terminal_log_e[index] for result in replications], dtype=float)
    return {
        "family": family,
        "ville_boundary": _stopping_summary(
            [result.crossing_times[index] for result in replications], horizon, confidence
        ),
        "terminal_log_e": {
            "mean": float(np.mean(terminal)),
            "median": float(np.median(terminal)),
            "minimum": float(np.min(terminal)),
            "maximum": float(np.max(terminal)),
        },
        "validity": "Ville-boundary finite diagnostic conditional on the declared nonnegative e-process null",
    }


def _condition_result(
    condition: _Condition,
    config: SequentialEvidenceAtlasConfig,
    children: Sequence[np.random.SeedSequence],
    workers: int,
) -> dict[str, object]:
    """Evaluate one declared truth condition with stable child-stream order."""
    if workers == 1:
        replications = [_run_replication(condition, config, child) for child in children]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            replications = list(
                executor.map(lambda child: _run_replication(condition, config, child), children)
            )
    mixture_rates = _mixture_rates(
        condition.null_rate, condition.alternative_rate, config.mixture_alternative_rates
    )
    predictable_rates = _predictable_rates(
        config.horizon, condition.null_rate, condition.alternative_rate
    )
    families = {
        family: _family_summary(family, replications, config.horizon, config.confidence)
        for family in _FAMILY_NAMES
    }
    fixed_crossings = [result.fixed_horizon_peek_crossing_time for result in replications]
    e_lond_rejections = [result.e_lond_num_rejections for result in replications]
    fixed_simple_rate = float(
        families["fixed_simple"]["ville_boundary"]["crossing"]["rate"]  # type: ignore[index]
    )
    return {
        "condition_id": condition.identifier,
        "null_rate": condition.null_rate,
        "true_rate": condition.true_rate,
        "truth_regime": condition.truth_regime,
        "under_declared_null": condition.true_rate == condition.null_rate,
        "n_reps": config.n_reps,
        "family_specifications": {
            "fixed_simple": {
                "alternative_rate": condition.alternative_rate,
                "fixed_before_stream": True,
            },
            "mixture": {
                "alternative_rates": list(mixture_rates),
                "weights": "equal",
                "fixed_before_stream": True,
            },
            "predictable": {
                "alternative_rates_by_time": predictable_rates.tolist(),
                "construction": "predeclared time-alternating schedule independent of observations",
                "filtration_assumption": "each rate is fixed before its corresponding observation",
            },
        },
        "families": families,
        "paired_crossing_rate_difference_vs_fixed_simple": {
            family: (
                float(families[family]["ville_boundary"]["crossing"]["rate"]) - fixed_simple_rate  # type: ignore[index]
            )
            for family in ("mixture", "predictable")
        },
        "confidence_sequence": {
            "simultaneous_coverage": _proportion_summary(
                [result.confidence_sequence_covered for result in replications], config.confidence
            ),
            "empty_path": _proportion_summary(
                [result.confidence_sequence_empty for result in replications], config.confidence
            ),
            "coverage_claim": "simultaneous-in-time Bernoulli coverage under each fixed null rate",
            "validity": "finite coverage simulation under the declared Bernoulli truth; not a coverage proof",
        },
        "e_lond": {
            "source_evalues": "one-step fixed-simple likelihood-ratio e-values",
            "mean_num_rejections": float(np.mean(e_lond_rejections)),
            "any_rejection": _proportion_summary(
                [count > 0 for count in e_lond_rejections], config.confidence
            ),
            "representative_trace": replications[0].e_lond_trace,
            "validity": "conditional e-value filtration accounting; finite simulations do not establish online-FDR validity",
        },
        "fixed_horizon_optional_stopping_contrast": {
            "repeated_prefix_rejection": _stopping_summary(
                fixed_crossings, config.horizon, config.confidence
            ),
            "validity": "invalid_optional_stopping_contrast",
            "claim_boundary": "Repeatedly applying a fixed-horizon Binomial p-value over prefixes is not a Type-I-error-control claim.",
        },
    }


def evaluate_sequential_evidence_atlas(
    config: SequentialEvidenceAtlasConfig,
    *,
    seed: int = 0,
    workers: int = 1,
) -> dict[str, object]:
    """Compare sequential-evidence procedures on stable independent streams.

    A condition-major ``SeedSequence`` schedule gives each replication one
    child stream that is shared by every evidence family.  Results therefore
    do not depend on the requested worker count.  The mapping intentionally
    distinguishes finite diagnostic estimates from the assumptions needed for
    Ville, confidence-sequence, predictable-process, and e-LOND theorems.
    """
    if not isinstance(config, SequentialEvidenceAtlasConfig):
        raise TypeError("config must be a SequentialEvidenceAtlasConfig")
    seed = _nonnegative_integer(seed, "seed")
    workers = _positive_integer(workers, "workers")
    conditions = _conditions(config)
    children = np.random.SeedSequence(seed).spawn(len(conditions) * config.n_reps)
    condition_results: list[dict[str, object]] = []
    for index, condition in enumerate(conditions):
        start = index * config.n_reps
        condition_results.append(
            _condition_result(condition, config, children[start : start + config.n_reps], workers)
        )
    seed_schedule = {
        "kind": "numpy.SeedSequence",
        "master_seed": seed,
        "assignment": "one child stream per condition and replication, shared across evidence families",
        "replication_order": "condition-major then ascending replication index",
        "n_child_streams": len(children),
    }
    return {
        "kind": "sequential_evidence_comparison_atlas",
        "configuration": config.to_mapping(),
        "configuration_hash": config.configuration_hash,
        "seed_schedule": seed_schedule,
        "seed_schedule_hash": _hash_mapping(seed_schedule),
        "common_random_numbers": {
            "enabled": True,
            "scope": "each condition/replication stream is reused by fixed-simple, mixture, predictable, confidence-sequence, and contrast calculations",
        },
        "conditions": condition_results,
        "validity": "finite independent-Bernoulli simulation diagnostic with theorem assumptions retained explicitly in each result",
    }
