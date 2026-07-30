"""Tests for :mod:`simulation` -- Monte-Carlo operating characteristics.

Real numerics only. No mocks / monkeypatch. Every stochastic test fixes a seed.
The data-generating process is pinned by *independently reconstructing* the exact
draws from a same-seeded ``numpy`` generator and recomputing the p-values from the
documented formula -- this binds the math (not just shapes) without hard-coding
platform-fragile float literals. The headline FDR-control identities are pinned
against the analytic Benjamini-Hochberg bound within Monte-Carlo error.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from active_inference_power._types import SimulationResult, TwoGroupsConfig
from active_inference_power.simulation import (
    PowerCurvePoint,
    compare_procedures,
    generate_pvalues,
    power_curve,
    replication_seed_schedule,
    simulate_procedure,
)

# --------------------------------------------------------------------------- #
# Helpers: independent reconstruction of the two-groups DGP
# --------------------------------------------------------------------------- #


def _reconstruct(config: TwoGroupsConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Recompute ``generate_pvalues`` output from the documented formula.

    Draws are taken from a fresh generator in the exact order the implementation
    consumes them, so this is a genuine independent oracle for the DGP math.
    """
    rng = np.random.default_rng(seed)
    m = config.m
    m1 = config.m1
    is_alt = np.zeros(m, dtype=bool)
    is_alt[:m1] = True
    shift = np.where(is_alt, float(config.effect), 0.0)
    common = rng.standard_normal()
    idiosyncratic = rng.standard_normal(m)
    z = np.sqrt(config.rho) * common + np.sqrt(1.0 - config.rho) * idiosyncratic + shift
    if config.sided == 1:
        p = norm.sf(z)
    else:
        p = 2.0 * norm.sf(np.abs(z))
    p = np.clip(p, 0.0, 1.0)
    perm = rng.permutation(m)
    return p[perm], is_alt[perm]


# --------------------------------------------------------------------------- #
# PowerCurvePoint
# --------------------------------------------------------------------------- #


def test_power_curve_point_fields() -> None:
    pt = PowerCurvePoint(effect=2.0, fdr=0.03, power=0.7)
    assert pt.effect == 2.0
    assert pt.fdr == 0.03
    assert pt.power == 0.7
    assert pt.fdr_stderr == 0.0
    assert pt.power_stderr == 0.0


# --------------------------------------------------------------------------- #
# generate_pvalues -- DGP math
# --------------------------------------------------------------------------- #


def test_generate_matches_independent_reconstruction_one_sided() -> None:
    cfg = TwoGroupsConfig(m=25, pi0=0.6, effect=2.5, rho=0.0, sided=1)
    p, alt = generate_pvalues(cfg, np.random.default_rng(2024))
    p_exp, alt_exp = _reconstruct(cfg, 2024)
    np.testing.assert_allclose(p, p_exp, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(alt, alt_exp)


def test_generate_matches_independent_reconstruction_two_sided() -> None:
    cfg = TwoGroupsConfig(m=30, pi0=0.5, effect=1.5, rho=0.3, sided=2)
    p, alt = generate_pvalues(cfg, np.random.default_rng(77))
    p_exp, alt_exp = _reconstruct(cfg, 77)
    np.testing.assert_allclose(p, p_exp, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(alt, alt_exp)


def test_generate_reproducible_same_seed() -> None:
    cfg = TwoGroupsConfig(m=40, pi0=0.7, effect=2.0, rho=0.2)
    p1, a1 = generate_pvalues(cfg, np.random.default_rng(9))
    p2, a2 = generate_pvalues(cfg, np.random.default_rng(9))
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(a1, a2)


def test_generate_different_seeds_differ() -> None:
    cfg = TwoGroupsConfig(m=40, pi0=0.7, effect=2.0)
    p1, _ = generate_pvalues(cfg, np.random.default_rng(1))
    p2, _ = generate_pvalues(cfg, np.random.default_rng(2))
    assert not np.array_equal(p1, p2)


def test_generate_alt_count_equals_m1_and_shuffled() -> None:
    cfg = TwoGroupsConfig(m=20, pi0=0.6, effect=3.0)  # m1 = 8
    p, alt = generate_pvalues(cfg, np.random.default_rng(5))
    assert int(alt.sum()) == cfg.m1 == 8
    # Alternatives are shuffled: they are NOT the leading m1 indices.
    leading = np.zeros(cfg.m, dtype=bool)
    leading[: cfg.m1] = True
    assert not np.array_equal(alt, leading)


def test_generate_pvalues_in_unit_interval_both_sided() -> None:
    for sided in (1, 2):
        cfg = TwoGroupsConfig(m=200, pi0=0.5, effect=3.0, rho=0.4, sided=sided)
        p, _ = generate_pvalues(cfg, np.random.default_rng(sided))
        assert p.shape == (200,)
        assert float(p.min()) >= 0.0
        assert float(p.max()) <= 1.0


def test_generate_alternatives_have_smaller_pvalues_one_sided() -> None:
    # Strong positive shift => alternative p-values are stochastically smaller.
    cfg = TwoGroupsConfig(m=400, pi0=0.5, effect=3.0, rho=0.0, sided=1)
    p, alt = generate_pvalues(cfg, np.random.default_rng(0))
    assert float(np.median(p[alt])) < float(np.median(p[~alt]))


# --------------------------------------------------------------------------- #
# generate_pvalues -- edge cases
# --------------------------------------------------------------------------- #


def test_generate_empty_model() -> None:
    cfg = TwoGroupsConfig(m=0, pi0=0.5, effect=2.0)
    p, alt = generate_pvalues(cfg, np.random.default_rng(0))
    assert p.shape == (0,)
    assert alt.shape == (0,)
    assert p.dtype == float
    assert alt.dtype == bool


def test_generate_single_hypothesis_alt() -> None:
    cfg = TwoGroupsConfig(m=1, pi0=0.0, effect=5.0)  # m1 = 1
    p, alt = generate_pvalues(cfg, np.random.default_rng(3))
    assert p.shape == (1,)
    assert bool(alt[0]) is True


def test_generate_all_null_pi0_one() -> None:
    cfg = TwoGroupsConfig(m=50, pi0=1.0, effect=3.0)  # m1 = 0
    p, alt = generate_pvalues(cfg, np.random.default_rng(0))
    assert int(alt.sum()) == 0
    # Under the global null one-sided p-values are ~Uniform(0,1).
    assert 0.3 < float(p.mean()) < 0.7


def test_generate_all_alt_pi0_zero() -> None:
    cfg = TwoGroupsConfig(m=30, pi0=0.0, effect=3.0)  # m1 = 30
    _, alt = generate_pvalues(cfg, np.random.default_rng(0))
    assert int(alt.sum()) == 30


def test_generate_rho_zero_and_one_are_valid() -> None:
    for rho in (0.0, 1.0):
        cfg = TwoGroupsConfig(m=10, pi0=0.5, effect=2.0, rho=rho)
        p, _ = generate_pvalues(cfg, np.random.default_rng(0))
        assert np.all(np.isfinite(p))


def test_generate_invalid_rho_raises() -> None:
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            generate_pvalues(
                TwoGroupsConfig(m=5, pi0=0.5, effect=1.0, rho=bad),
                np.random.default_rng(0),
            )


def test_generate_invalid_sided_raises() -> None:
    with pytest.raises(ValueError):
        generate_pvalues(
            TwoGroupsConfig(m=5, pi0=0.5, effect=1.0, sided=3),
            np.random.default_rng(0),
        )


# --------------------------------------------------------------------------- #
# simulate_procedure
# --------------------------------------------------------------------------- #


def test_simulate_bh_controls_fdr_within_mc_band() -> None:
    # Certification run: BH FDR <= pi0*alpha within 3 Monte-Carlo standard errors.
    cfg = TwoGroupsConfig(m=200, pi0=0.8, effect=3.0, rho=0.0)
    res = simulate_procedure(cfg, "benjamini_hochberg", alpha=0.05, n_reps=500, seed=0)
    bound = 0.8 * 0.05
    assert res.fdr <= bound + 3.0 * res.mc_stderr("fdr")
    assert res.power > 0.5  # strong effect => substantial power


def test_simulate_result_fields_consistent() -> None:
    cfg = TwoGroupsConfig(m=100, pi0=0.7, effect=2.5)
    res = simulate_procedure(
        cfg, "benjamini_hochberg", alpha=0.05, n_reps=200, seed=1
    )
    assert isinstance(res, SimulationResult)
    assert res.n_reps == 200
    assert res.method == "benjamini_hochberg"
    assert res.fdp_samples.shape == (200,)
    assert res.power_samples.shape == (200,)
    assert 0.0 <= res.fwer <= 1.0
    assert 0.0 <= res.fdr <= 1.0
    assert 0.0 <= res.power <= 1.0
    # Aggregates equal the means of the per-replication samples.
    assert res.fdr == pytest.approx(float(res.fdp_samples.mean()))
    assert res.power == pytest.approx(float(res.power_samples.mean()))


def test_simulate_reproducible_rho_zero() -> None:
    cfg = TwoGroupsConfig(m=150, pi0=0.6, effect=2.0, rho=0.0)
    a = simulate_procedure(
        cfg, "benjamini_hochberg", alpha=0.05, n_reps=100, seed=42
    )
    b = simulate_procedure(
        cfg, "benjamini_hochberg", alpha=0.05, n_reps=100, seed=42
    )
    np.testing.assert_array_equal(a.fdp_samples, b.fdp_samples)
    np.testing.assert_array_equal(a.power_samples, b.power_samples)
    assert a.fdr == b.fdr and a.power == b.power and a.fwer == b.fwer


def test_worker_count_does_not_change_seeded_samples() -> None:
    cfg = TwoGroupsConfig(m=40, pi0=0.6, effect=1.5, rho=0.2)
    serial = simulate_procedure(
        cfg, "benjamini_hochberg", n_reps=24, seed=9, workers=1
    )
    parallel = simulate_procedure(
        cfg, "benjamini_hochberg", n_reps=24, seed=9, workers=3
    )
    np.testing.assert_array_equal(serial.fdp_samples, parallel.fdp_samples)
    np.testing.assert_array_equal(serial.power_samples, parallel.power_samples)


def test_replication_seed_schedule_is_child_stream_based_and_stable() -> None:
    schedule = replication_seed_schedule(17, 3)
    assert [item["replication"] for item in schedule] == [0, 1, 2]
    assert [item["spawn_key"] for item in schedule] == [(0,), (1,), (2,)]
    assert schedule == replication_seed_schedule(17, 3)
    with pytest.raises(ValueError):
        replication_seed_schedule(True, 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        replication_seed_schedule(17, 0)


def test_simulation_validates_worker_count_and_exercises_weighted_bh() -> None:
    cfg = TwoGroupsConfig(m=12, pi0=0.5, effect=1.5, rho=0.0)
    with pytest.raises(ValueError):
        simulate_procedure(cfg, "benjamini_hochberg", n_reps=2, workers=0)
    result = simulate_procedure(cfg, "weighted_bh", n_reps=2, seed=11)
    assert result.n_reps == 2


def test_simulate_bonferroni_fdr_le_bh_fdr() -> None:
    # On matched data streams Bonferroni rejects a subset of BH => lower FDR.
    cfg = TwoGroupsConfig(m=200, pi0=0.8, effect=3.0, rho=0.0)
    bh = simulate_procedure(
        cfg, "benjamini_hochberg", alpha=0.05, n_reps=300, seed=0
    )
    bonf = simulate_procedure(cfg, "bonferroni", alpha=0.05, n_reps=300, seed=0)
    assert bonf.fdr <= bh.fdr + 1e-12
    assert bonf.power < bh.power  # and strictly less powerful


def test_simulate_power_increases_with_effect() -> None:
    weak = TwoGroupsConfig(m=200, pi0=0.7, effect=1.0, rho=0.0)
    strong = TwoGroupsConfig(m=200, pi0=0.7, effect=3.5, rho=0.0)
    p_weak = simulate_procedure(
        weak, "benjamini_hochberg", 0.05, 200, seed=3
    ).power
    p_strong = simulate_procedure(
        strong, "benjamini_hochberg", 0.05, 200, seed=3
    ).power
    assert p_strong > p_weak


def test_simulate_no_alternatives_zero_power() -> None:
    cfg = TwoGroupsConfig(m=80, pi0=1.0, effect=3.0)  # m1 == 0
    res = simulate_procedure(cfg, "benjamini_hochberg", 0.05, 100, seed=0)
    assert res.power == 0.0  # TPR is 0 by convention when m1 == 0


def test_simulate_global_null_fwer_controlled_bonferroni() -> None:
    # Under the complete null Bonferroni FWER <= alpha (with MC slack).
    cfg = TwoGroupsConfig(m=200, pi0=1.0, effect=0.0, rho=0.0)
    res = simulate_procedure(cfg, "bonferroni", 0.05, 400, seed=0)
    fwer_stderr = float(np.std(res.fdp_samples > 0.0, ddof=1) / np.sqrt(res.n_reps))
    assert res.fwer <= 0.05 + 3.0 * fwer_stderr
    # With m1 == 0 every rejection is false (V == R), so fdp is exactly the
    # FWER indicator per replication => FDR equals FWER, and power is 0.
    assert res.fdr == pytest.approx(res.fwer)
    assert res.power == 0.0


def test_simulate_empty_model_all_zero() -> None:
    cfg = TwoGroupsConfig(m=0, pi0=0.5, effect=2.0)
    res = simulate_procedure(cfg, "benjamini_hochberg", 0.05, 5, seed=0)
    assert res.fdr == 0.0
    assert res.power == 0.0
    assert res.fwer == 0.0
    assert res.mean_rejections == 0.0


def test_simulate_invalid_n_reps_raises() -> None:
    cfg = TwoGroupsConfig(m=10, pi0=0.5, effect=2.0)
    for bad in (0, -1):
        with pytest.raises(ValueError):
            simulate_procedure(cfg, "benjamini_hochberg", 0.05, bad, seed=0)
    with pytest.raises(ValueError):
        simulate_procedure(cfg, "benjamini_hochberg", 0.05, 2, seed=-1)


def test_simulate_unknown_method_raises() -> None:
    cfg = TwoGroupsConfig(m=10, pi0=0.5, effect=2.0)
    with pytest.raises(ValueError):
        simulate_procedure(cfg, "not_a_method", 0.05, 3, seed=0)


def test_simulate_positive_dependence_bh_and_by_control() -> None:
    # One-factor PRDS dependence: BH still controls FDR; BY is more conservative.
    cfg = TwoGroupsConfig(m=200, pi0=0.8, effect=3.0, rho=0.5)
    bh = simulate_procedure(cfg, "benjamini_hochberg", 0.05, 300, seed=0)
    by = simulate_procedure(cfg, "benjamini_yekutieli", 0.05, 300, seed=0)
    bound = 0.8 * 0.05
    assert by.fdr <= bh.fdr + 1e-12
    assert bh.fdr <= bound + 3.0 * bh.mc_stderr("fdr")


# --------------------------------------------------------------------------- #
# compare_procedures
# --------------------------------------------------------------------------- #


def test_compare_procedures_keys_and_pairing() -> None:
    cfg = TwoGroupsConfig(m=100, pi0=0.7, effect=2.5, rho=0.0)
    out = compare_procedures(
        cfg, ["benjamini_hochberg", "bonferroni", "holm"], 0.05, 100, seed=11
    )
    assert set(out) == {"benjamini_hochberg", "bonferroni", "holm"}
    assert all(isinstance(v, SimulationResult) for v in out.values())
    # Paired seed => Bonferroni (FWER) never more powerful than BH (FDR).
    assert out["bonferroni"].power <= out["benjamini_hochberg"].power
    # Holm dominates Bonferroni in power.
    assert out["holm"].power >= out["bonferroni"].power


def test_compare_procedures_empty_methods() -> None:
    cfg = TwoGroupsConfig(m=20, pi0=0.5, effect=2.0)
    assert compare_procedures(cfg, [], 0.05, 10, seed=0) == {}


def test_compare_procedures_rejects_duplicate_method_names() -> None:
    cfg = TwoGroupsConfig(m=20, pi0=0.5, effect=2.0)
    with pytest.raises(ValueError, match="duplicate"):
        compare_procedures(
            cfg,
            ["benjamini_hochberg", "benjamini_hochberg"],
            0.05,
            10,
            seed=0,
        )


# --------------------------------------------------------------------------- #
# power_curve
# --------------------------------------------------------------------------- #


def test_power_curve_monotone_and_controlled() -> None:
    base = TwoGroupsConfig(m=200, pi0=0.8, effect=0.0, rho=0.0)
    effects = [0.0, 1.0, 2.0, 3.0, 4.0]
    curve = power_curve(base, effects, "benjamini_hochberg", 0.05, 200, seed=1)
    assert len(curve) == len(effects)
    assert [pt.effect for pt in curve] == effects
    powers = [pt.power for pt in curve]
    assert all(pt.fdr_stderr >= 0.0 and pt.power_stderr >= 0.0 for pt in curve)
    # Shared seed => power is monotone non-decreasing in effect (matched noise).
    assert all(b >= a - 1e-12 for a, b in zip(powers, powers[1:]))
    assert powers[-1] > powers[0]
    # FDR stays controlled across the whole curve.
    for pt in curve:
        assert pt.fdr <= 0.8 * 0.05 + 0.03


def test_power_curve_empty_effects() -> None:
    base = TwoGroupsConfig(m=20, pi0=0.5, effect=2.0)
    assert power_curve(base, [], "benjamini_hochberg", 0.05, 10, seed=0) == []


def test_power_curve_returns_power_curve_points() -> None:
    base = TwoGroupsConfig(m=50, pi0=0.6, effect=0.0)
    curve = power_curve(
        base, [1.0, 2.0], "benjamini_hochberg", 0.05, 50, seed=0
    )
    assert all(isinstance(pt, PowerCurvePoint) for pt in curve)
    assert curve[0].effect == 1.0 and curve[1].effect == 2.0
