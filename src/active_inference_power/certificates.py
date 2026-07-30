"""Reproducible numeric certificates for multiple-testing procedures.

Each certificate runs a seeded Monte-Carlo experiment (via :mod:`simulation`),
compares an estimated operating characteristic against an explicit analytic
bound (from :mod:`theory`), and returns a plain ``dict`` recording the estimate,
the bound, the Monte-Carlo standard error, and a ``passes`` boolean. The bundle
echoes the spirit of the Dobriban / Benjamini-Hochberg reproducibility packages:
every claim is a re-runnable number with a pass/fail verdict at a fixed seed.

Certificates:

* :func:`certify_bh_fdr_control` -- empirically checks the BH FDR bound under
  the declared independence model.
* :func:`certify_bonferroni_fwer` -- empirically checks the Bonferroni FWER
  bound under the global-null simulation.
* :func:`certify_positive_dependence` -- checks the one-factor
  positive-dependence operating regime and compares BH with the more conservative BY.
* :func:`certify_storey_power_gain` -- checks a Storey-versus-BH power
  comparison together with finite-simulation FDR bands.
* :func:`certify_storey_split_confirmation` -- evaluates a preregistered
  independent selection/confirmation Storey remedy.
* :func:`run_all_certificates` -- run the whole suite (smaller when ``fast``).

Each ``passes`` is a *band* test: the estimate must not exceed its bound by more
than ``BAND_SIGMA`` Monte-Carlo standard errors, so a fixed-seed run gives a
deterministic verdict that is robust to sampling noise. A pass is evidence that
the selected finite simulation is consistent with its declared bound; it is not
a new proof of the underlying theorem.

Deterministic package domain code: imports ``numpy`` and package modules only,
never ``infrastructure.*``.
"""

from __future__ import annotations

import numpy as np

from ._types import TwoGroupsConfig
from .simulation import simulate_procedure
from .simulation import generate_pvalues
from .theory import bh_fdr_bound_from_counts
from .corrections import benjamini_hochberg, storey_qvalue_split_confirmation
from .error_rates import classify

__all__ = [
    "BAND_SIGMA",
    "CERTIFICATE_NAMES",
    "certify_bh_fdr_control",
    "certify_bonferroni_fwer",
    "certify_positive_dependence",
    "certify_storey_power_gain",
    "certify_storey_split_confirmation",
    "run_all_certificates",
]

# Number of Monte-Carlo standard errors of slack allowed above an analytic bound
# before a certificate is declared failed. Three sigma keeps false failures from
# sampling noise below ~0.1% while still catching a genuinely violated bound.
BAND_SIGMA: float = 3.0

# The configured release/campaign certificate surface.  Gate policies name
# this exact finite set so a result cannot silently omit or reclassify a
# certificate when the suite changes.
CERTIFICATE_NAMES: tuple[str, ...] = (
    "bh_fdr_control",
    "bonferroni_fwer",
    "positive_dependence",
    "storey_power_gain",
    "storey_split_confirmation",
)

Certificate = dict[str, object]


def _binomial_stderr(proportion: float, n: int) -> float:
    """Monte-Carlo standard error of a proportion (e.g. an FWER estimate).

    Args:
        proportion: Estimated event probability in ``[0, 1]``.
        n: Number of independent Monte-Carlo replications (``>= 1``).

    Returns:
        ``sqrt(p * (1 - p) / n)``; ``0.0`` when ``n <= 1``.
    """
    if n <= 1:
        return 0.0
    p = float(proportion)
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / n))


def certify_bh_fdr_control(
    pi0: float = 0.8,
    effect: float = 3.0,
    m: int = 200,
    alpha: float = 0.05,
    n_reps: int = 500,
    seed: int = 0,
) -> Certificate:
    """Check the BH FDR bound in the declared independent simulation.

    Simulates the two-groups model with independent statistics (``rho = 0``) and
    checks that the Monte-Carlo BH FDR does not exceed the analytic bound
    ``theory.bh_fdr_bound(pi0, alpha) = pi0 * alpha`` by more than
    ``BAND_SIGMA`` standard errors.

    Args:
        pi0: Fraction of true nulls in ``[0, 1]``.
        effect: Alternative mean shift in null-standard-deviation units.
        m: Number of hypotheses (``>= 1``).
        alpha: Nominal FDR level in ``(0, 1)``.
        n_reps: Monte-Carlo replications (``>= 1``).
        seed: RNG seed for reproducibility.

    Returns:
        A certificate ``dict`` with ``name``, ``estimate`` (BH FDR), ``bound``,
        ``stderr``, ``passes``, plus ``power``, ``mean_rejections`` and the
        configuration echoed back.
    """
    config = TwoGroupsConfig(m=m, pi0=pi0, effect=effect, rho=0.0, sided=1, seed=seed)
    sim = simulate_procedure(config, "benjamini_hochberg", alpha, n_reps, seed)
    realized_pi0 = config.m0 / config.m
    bound = bh_fdr_bound_from_counts(config.m0, config.m, alpha)
    estimate = sim.fdr
    stderr = sim.mc_stderr("fdr")
    passes = bool(estimate <= bound + BAND_SIGMA * stderr)
    return {
        "name": "bh_fdr_control",
        "bound_direction": "upper",
        "estimate": estimate,
        "bound": bound,
        "stderr": stderr,
        "passes": passes,
        "power": sim.power,
        "mean_rejections": sim.mean_rejections,
        "pi0": float(pi0),
        "realized_pi0": float(realized_pi0),
        "m0": int(config.m0),
        "effect": float(effect),
        "m": int(m),
        "alpha": float(alpha),
        "n_reps": int(n_reps),
        "seed": int(seed),
    }


def certify_bonferroni_fwer(
    m: int = 200,
    alpha: float = 0.05,
    n_reps: int = 500,
    seed: int = 0,
) -> Certificate:
    """Check the Bonferroni FWER bound in the global-null simulation.

    Simulates the global null (every hypothesis a true null: ``pi0 = 1``,
    ``effect = 0``) with independent statistics and checks that the Monte-Carlo
    family-wise error rate of the Bonferroni procedure does not exceed ``alpha``
    by more than ``BAND_SIGMA`` standard errors.

    Args:
        m: Number of hypotheses (``>= 1``).
        alpha: Nominal FWER level in ``(0, 1)``.
        n_reps: Monte-Carlo replications (``>= 1``).
        seed: RNG seed for reproducibility.

    Returns:
        A certificate ``dict`` with ``name``, ``estimate`` (FWER), ``bound``
        (``alpha``), ``stderr``, ``passes`` and the configuration echoed back.
    """
    config = TwoGroupsConfig(m=m, pi0=1.0, effect=0.0, rho=0.0, sided=1, seed=seed)
    sim = simulate_procedure(config, "bonferroni", alpha, n_reps, seed)
    estimate = sim.fwer
    bound = float(alpha)
    stderr = _binomial_stderr(estimate, n_reps)
    passes = bool(estimate <= bound + BAND_SIGMA * stderr)
    return {
        "name": "bonferroni_fwer",
        "bound_direction": "upper",
        "estimate": estimate,
        "bound": bound,
        "stderr": stderr,
        "passes": passes,
        "mean_rejections": sim.mean_rejections,
        "m": int(m),
        "alpha": float(alpha),
        "n_reps": int(n_reps),
        "seed": int(seed),
    }


def certify_positive_dependence(
    rho: float = 0.5,
    pi0: float = 0.8,
    effect: float = 3.0,
    m: int = 200,
    alpha: float = 0.05,
    n_reps: int = 500,
    seed: int = 0,
) -> Certificate:
    """Check BH/BY behavior in the one-factor positive-dependence regime.

    Under the one-factor equicorrelation model with ``rho > 0`` the p-values are
    positively regression dependent (PRDS), a regime in which BH has a
    theoretical FDR result at ``pi0 * alpha`` (Benjamini-Yekutieli 2001). This
    finite-simulation check reports both the BH and the (more conservative) BY
    FDR and checks that:
    both sit within band of ``theory.bh_fdr_bound(pi0, alpha)``, and BY does not
    exceed BH beyond sampling noise. It makes **no** claim about the non-PRDS
    two-sided boundary; see Dobriban for that counterexample regime.

    Args:
        rho: One-factor equicorrelation in ``[0, 1]``.
        pi0: Fraction of true nulls in ``[0, 1]``.
        effect: Alternative mean shift in null-standard-deviation units.
        m: Number of hypotheses (``>= 1``).
        alpha: Nominal FDR level in ``(0, 1)``.
        n_reps: Monte-Carlo replications (``>= 1``).
        seed: RNG seed for reproducibility.

    Returns:
        A certificate ``dict`` with ``name``, ``estimate`` (BH FDR), ``bound``,
        ``stderr``, ``passes``, plus ``bh_fdr``, ``by_fdr``, ``bh_power``,
        ``by_power`` and the configuration echoed back.
    """
    config = TwoGroupsConfig(m=m, pi0=pi0, effect=effect, rho=rho, sided=1, seed=seed)
    sim_bh = simulate_procedure(config, "benjamini_hochberg", alpha, n_reps, seed)
    sim_by = simulate_procedure(config, "benjamini_yekutieli", alpha, n_reps, seed)
    realized_pi0 = config.m0 / config.m
    bound = bh_fdr_bound_from_counts(config.m0, config.m, alpha)
    se_bh = sim_bh.mc_stderr("fdr")
    se_by = sim_by.mc_stderr("fdr")
    se_max = max(se_bh, se_by)
    passes = bool(
        all(
            [
                sim_bh.fdr <= bound + BAND_SIGMA * se_bh,
                sim_by.fdr <= bound + BAND_SIGMA * se_by,
                sim_by.fdr <= sim_bh.fdr + BAND_SIGMA * se_max,
            ]
        )
    )
    return {
        "name": "positive_dependence",
        "bound_direction": "upper",
        "estimate": sim_bh.fdr,
        "bound": bound,
        "stderr": se_bh,
        "passes": passes,
        "bh_fdr": sim_bh.fdr,
        "by_fdr": sim_by.fdr,
        "bh_power": sim_bh.power,
        "by_power": sim_by.power,
        "rho": float(rho),
        "pi0": float(pi0),
        "realized_pi0": float(realized_pi0),
        "m0": int(config.m0),
        "effect": float(effect),
        "m": int(m),
        "alpha": float(alpha),
        "n_reps": int(n_reps),
        "seed": int(seed),
    }


def certify_storey_power_gain(
    pi0: float = 0.5,
    effect: float = 3.0,
    m: int = 200,
    alpha: float = 0.05,
    n_reps: int = 500,
    seed: int = 0,
) -> Certificate:
    """Check a Storey-versus-BH power comparison with FDR bands.

    Storey's adaptive q-values plug in an estimate of ``pi0`` and can be less
    conservative than BH. This certificate checks that the Storey average power
    is at least the BH average power in this seeded design, while BH keeps FDR
    within band of ``pi0 * alpha`` and Storey keeps FDR within band of ``alpha``
    (Storey's target level). It does not establish dominance or universal FDR
    control outside the stated assumptions.

    Args:
        pi0: Fraction of true nulls in ``[0, 1]``.
        effect: Alternative mean shift in null-standard-deviation units.
        m: Number of hypotheses (``>= 1``).
        alpha: Nominal FDR level in ``(0, 1)``.
        n_reps: Monte-Carlo replications (``>= 1``).
        seed: RNG seed for reproducibility.

    Returns:
        A certificate ``dict`` with separate Monte-Carlo uncertainty for FDR,
        power, and the paired power gain, decomposed gate decisions, finite-band
        ceilings, ``power_gain``, ``bh_fdr``, ``storey_fdr``, and the
        configuration. The top-level ``stderr`` remains the Storey-power SE for
        the existing public field; the metric-specific fields are authoritative.
    """
    config = TwoGroupsConfig(m=m, pi0=pi0, effect=effect, rho=0.0, sided=1, seed=seed)
    sim_bh = simulate_procedure(config, "benjamini_hochberg", alpha, n_reps, seed)
    sim_st = simulate_procedure(config, "storey_qvalue", alpha, n_reps, seed)
    realized_pi0 = config.m0 / config.m
    bound_bh_fdr = bh_fdr_bound_from_counts(config.m0, config.m, alpha)
    bound_st_fdr = float(alpha)
    se_bh_fdr = sim_bh.mc_stderr("fdr")
    se_bh_power = sim_bh.mc_stderr("power")
    se_st_fdr = sim_st.mc_stderr("fdr")
    se_st_power = sim_st.mc_stderr("power")
    power_gain_samples = sim_st.power_samples - sim_bh.power_samples
    power_gain_stderr = (
        float(np.std(power_gain_samples, ddof=1) / np.sqrt(n_reps))
        if n_reps > 1
        else 0.0
    )
    bh_fdr_passes = bool(sim_bh.fdr <= bound_bh_fdr + BAND_SIGMA * se_bh_fdr)
    storey_fdr_passes = bool(sim_st.fdr <= bound_st_fdr + BAND_SIGMA * se_st_fdr)
    power_gain = float(sim_st.power - sim_bh.power)
    power_gain_lower_band = power_gain - BAND_SIGMA * power_gain_stderr
    power_gain_upper_band = power_gain + BAND_SIGMA * power_gain_stderr
    power_gain_passes = bool(power_gain_lower_band >= 0.0)
    passes = bool(
        all(
            [
                power_gain_passes,
                bh_fdr_passes,
                storey_fdr_passes,
            ]
        )
    )
    return {
        "name": "storey_power_gain",
        "bound_direction": "lower",
        "estimate": sim_st.power,
        "bound": sim_bh.power,
        "stderr": se_st_power,
        "passes": passes,
        "power_gain_passes": power_gain_passes,
        "bh_fdr_passes": bh_fdr_passes,
        "storey_fdr_passes": storey_fdr_passes,
        "power_gain": power_gain,
        "power_gain_stderr": power_gain_stderr,
        "power_gain_lower_band": power_gain_lower_band,
        "power_gain_upper_band": power_gain_upper_band,
        "bh_power": sim_bh.power,
        "bh_power_stderr": se_bh_power,
        "storey_power": sim_st.power,
        "bh_fdr_stderr": se_bh_fdr,
        "storey_fdr_stderr": se_st_fdr,
        "bh_fdr_upper_band": bound_bh_fdr + BAND_SIGMA * se_bh_fdr,
        "storey_fdr_upper_band": bound_st_fdr + BAND_SIGMA * se_st_fdr,
        "storey_fdr_excess": sim_st.fdr - bound_st_fdr,
        "storey_fdr_z": (sim_st.fdr - bound_st_fdr) / se_st_fdr if se_st_fdr > 0.0 else 0.0,
        "bh_fdr": sim_bh.fdr,
        "storey_fdr": sim_st.fdr,
        "pi0": float(pi0),
        "realized_pi0": float(realized_pi0),
        "m0": int(config.m0),
        "effect": float(effect),
        "m": int(m),
        "alpha": float(alpha),
        "n_reps": int(n_reps),
        "seed": int(seed),
        "uncertainty_policy": "marginal_3_mc_standard_errors; paired_power_gain",
        "gate_scope": "three_marginal_finite_simulation_gates; not a simultaneous theorem",
    }


def certify_storey_split_confirmation(
    pi0: float = 0.8,
    effect: float = 3.0,
    m: int = 200,
    alpha: float = 0.05,
    n_reps: int = 500,
    seed: int = 0,
    lambda_: float = 0.5,
    confidence: float = 0.99,
) -> Certificate:
    """Check a predeclared independent-stream Storey remedy.

    Every replication draws a selection p-value vector and an independent
    confirmation vector from separate ``SeedSequence`` children. The null-count
    upper bound uses only the selection vector; confirmation rejections use
    only the confirmation vector. The result is a finite calibration check
    under the declared split-stream assumptions, not a claim that the same
    procedure is valid after arbitrary adaptive selection.
    """
    if not 0.0 <= lambda_ < 1.0:
        raise ValueError("lambda_ must be in [0, 1)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    config = TwoGroupsConfig(m=m, pi0=pi0, effect=effect, rho=0.0, sided=1, seed=seed)
    children = np.random.SeedSequence(int(seed)).spawn(int(n_reps) * 2)
    fdp_split: list[float] = []
    power_split: list[float] = []
    fwer_split: list[float] = []
    fdp_bh: list[float] = []
    power_bh: list[float] = []
    for index in range(int(n_reps)):
        selection_p, _ = generate_pvalues(config, np.random.default_rng(children[2 * index]))
        confirmation_p, is_alt = generate_pvalues(config, np.random.default_rng(children[2 * index + 1]))
        split = storey_qvalue_split_confirmation(
            selection_p, confirmation_p, alpha, lambda_=lambda_, confidence=confidence
        )
        bh = benjamini_hochberg(confirmation_p, alpha)
        split_summary = classify(split.rejected, is_alt)
        bh_summary = classify(bh.rejected, is_alt)
        fdp_split.append(split_summary.fdp)
        power_split.append(split_summary.power)
        fwer_split.append(float(split_summary.fwer_event))
        fdp_bh.append(bh_summary.fdp)
        power_bh.append(bh_summary.power)
    split_fdr = float(np.mean(fdp_split))
    split_power = float(np.mean(power_split))
    split_fwer = float(np.mean(fwer_split))
    split_stderr = _binomial_stderr(split_fdr, int(n_reps))
    bh_fdr = float(np.mean(fdp_bh))
    bh_power = float(np.mean(power_bh))
    passes = bool(split_fdr <= float(alpha) + BAND_SIGMA * split_stderr)
    return {
        "name": "storey_split_confirmation",
        "bound_direction": "upper",
        "estimate": split_fdr,
        "bound": float(alpha),
        "stderr": split_stderr,
        "passes": passes,
        "validity": "conditional_split_stream_calibration_diagnostic",
        "assumptions": [
            "independent_selection_and_confirmation_streams",
            "fixed_lambda_and_confidence_declared_before_simulation",
            "marginally_superuniform_null_confirmation_pvalues",
        ],
        "selection_confidence": float(confidence),
        "lambda": float(lambda_),
        "split_fdr": split_fdr,
        "split_power": split_power,
        "split_fwer": split_fwer,
        "bh_fdr_confirmation": bh_fdr,
        "bh_power_confirmation": bh_power,
        "power_difference_vs_bh": split_power - bh_power,
        "pi0": float(pi0),
        "effect": float(effect),
        "m": int(m),
        "alpha": float(alpha),
        "n_reps": int(n_reps),
        "seed": int(seed),
    }


def run_all_certificates(
    fast: bool = True,
    m: int | None = None,
    n_reps: int | None = None,
    seed: int = 0,
    storey_lambda: float = 0.5,
    storey_confidence: float = 0.99,
) -> dict[str, Certificate]:
    """Run the whole certificate suite and collect the results.

    Args:
        fast: When True use smaller defaults for a quick, still-valid pass;
            when False use the larger documented defaults.
        m: Optional explicit hypothesis count overriding the fast/full default.
        n_reps: Optional explicit replication count overriding the fast/full
            default.
        seed: Shared seed used by every certificate.
        storey_lambda: Fixed lambda in the preregistered split-stream remedy.
        storey_confidence: One-sided null-count confidence in that remedy.

    Returns:
        Mapping from certificate name to its result ``dict``.
    """
    resolved_n_reps = n_reps if n_reps is not None else (150 if fast else 500)
    resolved_m = m if m is not None else (120 if fast else 200)
    return {
        "bh_fdr_control": certify_bh_fdr_control(
            m=resolved_m, n_reps=resolved_n_reps, seed=seed
        ),
        "bonferroni_fwer": certify_bonferroni_fwer(
            m=resolved_m, n_reps=resolved_n_reps, seed=seed
        ),
        "positive_dependence": certify_positive_dependence(
            m=resolved_m, n_reps=resolved_n_reps, seed=seed
        ),
        "storey_power_gain": certify_storey_power_gain(
            m=resolved_m, n_reps=resolved_n_reps, seed=seed
        ),
        "storey_split_confirmation": certify_storey_split_confirmation(
            m=resolved_m,
            n_reps=resolved_n_reps,
            seed=seed,
            lambda_=storey_lambda,
            confidence=storey_confidence,
        ),
    }
