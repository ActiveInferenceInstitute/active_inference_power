"""Monte-Carlo operating characteristics of the two-groups multiple-testing model.

This module wires the analytic pieces together into a simulation loop. It draws
p-values from the canonical two-groups model (a fraction ``pi0`` of true nulls,
the rest shifted alternatives, with optional one-factor equicorrelation), applies
a multiple-testing procedure from :mod:`corrections`, scores each replication with
:func:`error_rates.classify`, and aggregates the per-replication error summaries
into a :class:`~_types.SimulationResult`.

Data-generating model (per replication):

* ``m`` z-statistics ``z_i = sqrt(rho) * F + sqrt(1 - rho) * e_i + shift_i`` with
  a single common factor ``F ~ N(0, 1)`` and idiosyncratic ``e_i ~ N(0, 1)``.
  ``rho`` in ``[0, 1]`` is the one-factor equicorrelation (``0`` == independence).
* ``shift_i = effect`` for the ``m1`` true alternatives, ``0`` for true nulls.
* One-sided p-values ``p_i = 1 - Phi(z_i)``; two-sided ``p_i = 2 * (1 - Phi(|z_i|))``.
* Alternative positions are shuffled so they are not always the leading indices.

Deterministic package domain code: imports only ``numpy``/``scipy`` and the
package correction, error-rate, and contract modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import norm

from .dependence import DependenceConfig, sample_gaussian
from ._types import CANONICAL_CORRECTION_METHODS, SimulationResult, TwoGroupsConfig
from .corrections import apply_correction
from .error_rates import aggregate, classify

__all__ = [
    "PowerCurvePoint",
    "generate_pvalues",
    "replication_seed_schedule",
    "simulate_procedure",
    "compare_procedures",
    "power_curve",
]


def replication_seed_schedule(seed: int, n_reps: int) -> tuple[dict[str, int | tuple[int, ...]], ...]:
    """Return the deterministic child-stream schedule for a replication block.

    Each replication receives its own :class:`numpy.random.SeedSequence`
    child.  The returned spawn keys are serializable audit metadata; worker
    count and reduction order do not alter the numerical stream assignment.
    """
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(n_reps, bool) or int(n_reps) != n_reps or int(n_reps) < 1:
        raise ValueError("n_reps must be a positive integer")
    master = np.random.SeedSequence(int(seed))
    return tuple(
        {
            "replication": index,
            "spawn_key": tuple(int(part) for part in child.spawn_key),
        }
        for index, child in enumerate(master.spawn(int(n_reps)))
    )


@dataclass
class PowerCurvePoint:
    """One point on a power/FDR-versus-effect-size curve.

    Attributes:
        effect: Alternative mean shift (in null-standard-deviation units) used
            to generate the replications for this point.
        fdr: Monte-Carlo estimate of the false discovery rate at this effect.
        power: Monte-Carlo estimate of the average power at this effect.
        fdr_stderr: Monte-Carlo standard error of ``fdr``.
        power_stderr: Monte-Carlo standard error of ``power``.
    """

    effect: float
    fdr: float
    power: float
    fdr_stderr: float = 0.0
    power_stderr: float = 0.0
    fdr_samples: np.ndarray | None = field(default=None, repr=False)
    power_samples: np.ndarray | None = field(default=None, repr=False)


def generate_pvalues(
    config: TwoGroupsConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one replication of the two-groups model.

    Args:
        config: The two-groups model parameters. ``config.m1`` hypotheses are
            true alternatives (shift ``config.effect``); the remaining
            ``config.m0`` are true nulls (shift ``0``). ``config.rho`` sets the
            one-factor equicorrelation and ``config.sided`` selects one- or
            two-sided p-values.
        rng: A seeded ``numpy`` generator; all randomness is drawn from it so a
            fixed seed makes the replication reproducible.

    Returns:
        Tuple ``(pvalues, is_alt)`` of shape ``(m,)`` each. ``pvalues`` are raw
        p-values in ``[0, 1]`` and ``is_alt`` is the boolean ground-truth mask
        (``True`` == true alternative), aligned after a random shuffle so the
        alternatives are not always the leading indices.

    Raises:
        ValueError: If factor ``config.rho`` is outside ``[0, 1]``, independent
            ``config.rho`` is nonzero, or ``config.sided`` is not ``1`` or ``2``.
    """
    rho = float(config.rho)
    if config.dependence_kind == "independent" and rho != 0.0:
        raise ValueError(
            "rho must be zero for independent dependence; nonzero rho would be ignored"
        )
    if config.dependence_kind == "factor" and not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must lie in [0, 1] for factor dependence, got {rho}")
    if config.sided not in (1, 2):
        raise ValueError(f"sided must be 1 or 2, got {config.sided}")

    m = int(config.m)
    if m == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=bool)

    m1 = config.m1
    is_alt = np.zeros(m, dtype=bool)
    is_alt[:m1] = True
    shift = np.where(is_alt, float(config.effect), 0.0)

    if config.dependence_kind == "factor":
        common = rng.standard_normal()
        idiosyncratic = rng.standard_normal(m)
        z = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idiosyncratic + shift
    else:
        dependence = DependenceConfig(
            kind=config.dependence_kind,
            rho=rho,
            block_size=config.block_size,
        )
        z = sample_gaussian(rng, m, dependence, shift)

    if config.sided == 1:
        pvalues = norm.sf(z)
    else:  # sided == 2
        pvalues = 2.0 * norm.sf(np.abs(z))
    pvalues = np.clip(pvalues, 0.0, 1.0)

    perm = rng.permutation(m)
    return pvalues[perm], is_alt[perm]


def simulate_procedure(
    config: TwoGroupsConfig,
    method: str,
    alpha: float = 0.05,
    n_reps: int = 1000,
    seed: int = 0,
    workers: int = 1,
) -> SimulationResult:
    """Estimate a procedure's operating characteristics by Monte Carlo.

    Runs ``n_reps`` independent replications of the two-groups model from
    deterministic ``SeedSequence`` child streams, applies ``method`` via
    :func:`corrections.apply_correction`, scores each replication with
    :func:`error_rates.classify`, and aggregates in stable replication order.

    Args:
        config: Two-groups model parameters.
        method: Name of the multiple-testing procedure (any accepted by
            :func:`corrections.apply_correction`).
        alpha: Nominal level passed to the procedure.
        n_reps: Number of Monte-Carlo replications (``>= 1``).
        seed: Master seed for the replication child-stream schedule.

    Returns:
        A :class:`~_types.SimulationResult` whose ``fdr`` estimates ``E[FDP]``,
        ``power`` estimates the average power, ``fwer`` estimates ``P(V >= 1)``,
        and whose ``fdp_samples``/``power_samples`` hold the per-replication draws.

    Raises:
        ValueError: If ``n_reps < 1``, or if ``method``/``config`` are invalid
            (propagated from the callees).
    """
    if not isinstance(method, str) or method not in CANONICAL_CORRECTION_METHODS:
        raise ValueError(f"unknown canonical correction method: {method!r}")
    try:
        n_reps_int = int(n_reps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"n_reps must be an integer >= 1, got {n_reps!r}") from exc
    if isinstance(n_reps, bool) or n_reps_int != n_reps or n_reps_int < 1:
        raise ValueError(f"n_reps must be an integer >= 1, got {n_reps!r}")
    n_reps = n_reps_int
    if isinstance(workers, bool) or int(workers) != workers or int(workers) < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)

    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    child_streams = np.random.SeedSequence(int(seed)).spawn(n_reps)
    draws = [
        generate_pvalues(config, np.random.default_rng(child_stream))
        for child_stream in child_streams
    ]

    def score_draw(draw: tuple[np.ndarray, np.ndarray]):
        pvalues, is_alt = draw
        kwargs: dict[str, object] = {}
        if method == "weighted_bh":
            weights = np.linspace(0.5, 1.5, config.m, dtype=float)
            kwargs["weights"] = weights / np.mean(weights)
        result = apply_correction(method, pvalues, alpha, **kwargs)
        return classify(result.rejected, is_alt)

    if workers == 1:
        summaries = [score_draw(draw) for draw in draws]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            summaries = list(executor.map(score_draw, draws))
    fdp_samples = np.asarray([summary.fdp for summary in summaries], dtype=float)
    power_samples = np.asarray([summary.power for summary in summaries], dtype=float)
    fwer_samples = np.asarray([float(summary.fwer_event) for summary in summaries], dtype=float)

    fdr, power, fwer, mean_rejections = aggregate(summaries)
    return SimulationResult(
        method=method,
        n_reps=n_reps,
        fdr=fdr,
        power=power,
        fwer=fwer,
        mean_rejections=mean_rejections,
        fdp_samples=fdp_samples,
        power_samples=power_samples,
        fwer_samples=fwer_samples,
    )


def compare_procedures(
    config: TwoGroupsConfig,
    methods: Iterable[str],
    alpha: float = 0.05,
    n_reps: int = 1000,
    seed: int = 0,
    workers: int = 1,
) -> dict[str, SimulationResult]:
    """Compare several procedures on the *same* simulated data streams.

    Each procedure is evaluated with the identical ``seed`` so every method sees
    the same replications, making the comparison paired (and thus low-variance).

    Args:
        config: Two-groups model parameters.
        methods: Iterable of procedure names.
        alpha: Nominal level.
        n_reps: Number of Monte-Carlo replications per method.
        seed: Shared seed for every method's replication stream.

    Returns:
        Mapping from each method name to its :class:`~_types.SimulationResult`.
    """
    method_names = tuple(methods)
    if len(set(method_names)) != len(method_names):
        raise ValueError("methods must not contain duplicate procedure names")
    return {
        method: simulate_procedure(config, method, alpha, n_reps, seed, workers)
        for method in method_names
    }


def power_curve(
    base_config: TwoGroupsConfig,
    effects: Sequence[float],
    method: str,
    alpha: float = 0.05,
    n_reps: int = 1000,
    seed: int = 0,
    workers: int = 1,
) -> list[PowerCurvePoint]:
    """Trace FDR and power as the alternative effect size varies.

    For each value in ``effects`` a copy of ``base_config`` with that ``effect``
    is simulated (all other parameters held fixed). The shared ``seed`` keeps the
    common factor and idiosyncratic noise identical across effects, so the only
    thing that changes per replication is the alternative shift. Shared seeds
    therefore provide paired common random numbers that reduce comparison noise.
    For a one-sided positive-effect design, rejection paths are expected to be
    non-decreasing in effect; this is not a universal monotonicity guarantee for
    two-sided designs or arbitrary procedures.

    Args:
        base_config: Baseline two-groups parameters; its ``effect`` is overridden.
        effects: Effect sizes to sweep.
        method: Multiple-testing procedure name.
        alpha: Nominal level.
        n_reps: Replications per effect value.
        seed: Shared seed for every effect value's replication stream.

    Returns:
        A list of :class:`PowerCurvePoint`, one per entry in ``effects``.
    """
    points: list[PowerCurvePoint] = []
    for effect in effects:
        cfg = replace(base_config, effect=float(effect))
        result = simulate_procedure(cfg, method, alpha, n_reps, seed, workers)
        points.append(
            PowerCurvePoint(
                effect=float(effect),
                fdr=result.fdr,
                power=result.power,
                fdr_stderr=result.mc_stderr("fdr"),
                power_stderr=result.mc_stderr("power"),
                fdr_samples=result.fdp_samples,
                power_samples=result.power_samples,
            )
        )
    return points
