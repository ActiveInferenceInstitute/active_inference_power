"""Real-number tests for :mod:`corrections`.

No mocks: every assertion pins hand-computed numeric values or exact
cross-procedure identities on real p-value vectors. The central worked example
is the 6-p-value vector ``P6`` whose BH / Holm / Hochberg / BY rejection sets and
adjusted p-values are derived by hand in the block comments below.
"""

from __future__ import annotations

import numpy as np
import pytest

from active_inference_power._types import CorrectionResult
from active_inference_power.corrections import (
    adaptive_bh,
    apply_correction,
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    hochberg,
    holm,
    sidak,
    storey_qvalue,
    storey_qvalue_conservative,
    storey_qvalue_split_confirmation,
    weighted_bh,
)

# --------------------------------------------------------------------------- #
# Canonical hand-computed example.
#
# P6 = [0.005, 0.011, 0.02, 0.04, 0.13, 0.5], m = 6, alpha = 0.05 (already sorted)
#
#   BH:  largest k with p_(k) <= (k/6)*0.05 is k=3 -> reject {0.005,0.011,0.02}
#        q = revcummin[(6/rank)*p] = [0.03, 0.033, 0.04, 0.06, 0.156, 0.5]
#   Holm: cummax[(6-i+1)*p_(i)] = [0.03, 0.055, 0.08, 0.12, 0.26, 0.5]
#         reject {0.005} only
#   Hochberg: revcummin of the same seq (monotone) -> identical -> reject {0.005}
#   Bonferroni: min(6p,1) = [0.03,0.066,0.12,0.24,0.78,1.0] -> reject {0.005}
#   BY: BH_q * c(6), c(6)=2.45 -> [0.0735,0.08085,0.098,0.147,0.3822,1.0]
#       -> reject none
# --------------------------------------------------------------------------- #
P6 = np.array([0.005, 0.011, 0.02, 0.04, 0.13, 0.5])
ALPHA = 0.05
TOL = 1e-9


# --------------------------------------------------------------------------- #
# Bonferroni
# --------------------------------------------------------------------------- #
def test_bonferroni_pins_adjusted_and_rejection() -> None:
    res = bonferroni(P6, ALPHA)
    assert isinstance(res, CorrectionResult)
    assert res.method == "bonferroni"
    assert res.adjusted == pytest.approx([0.03, 0.066, 0.12, 0.24, 0.78, 1.0], abs=TOL)
    assert res.rejected.tolist() == [True, False, False, False, False, False]
    assert res.num_rejected == 1
    assert res.threshold == pytest.approx(0.005, abs=TOL)
    assert res.alpha == ALPHA


def test_bonferroni_clips_large_product_to_one() -> None:
    # 6 * 0.5 = 3.0 must clip to 1.0, never exceed the [0,1] contract.
    res = bonferroni(P6, ALPHA)
    assert res.adjusted.max() <= 1.0
    assert res.adjusted[-1] == pytest.approx(1.0, abs=TOL)


# --------------------------------------------------------------------------- #
# Sidak
# --------------------------------------------------------------------------- #
def test_sidak_pins_adjusted_first_value() -> None:
    res = sidak(P6, ALPHA)
    assert res.method == "sidak"
    # 1 - 0.995**6 = 0.0296275...
    assert res.adjusted[0] == pytest.approx(1.0 - 0.995**6, abs=TOL)
    assert res.adjusted[0] == pytest.approx(0.02962753, abs=1e-7)
    assert res.rejected.tolist() == [True, False, False, False, False, False]
    assert res.threshold == pytest.approx(0.005, abs=TOL)


def test_sidak_never_more_conservative_than_bonferroni() -> None:
    # Sidak adjusted <= Bonferroni adjusted elementwise (before clipping ties).
    s = sidak(P6, ALPHA).adjusted
    b = bonferroni(P6, ALPHA).adjusted
    assert np.all(s <= b + TOL)


# --------------------------------------------------------------------------- #
# Holm
# --------------------------------------------------------------------------- #
def test_holm_pins_adjusted_and_single_rejection() -> None:
    res = holm(P6, ALPHA)
    assert res.method == "holm"
    assert res.adjusted == pytest.approx([0.03, 0.055, 0.08, 0.12, 0.26, 0.5], abs=TOL)
    assert res.rejected.tolist() == [True, False, False, False, False, False]
    assert res.threshold == pytest.approx(0.005, abs=TOL)


def test_holm_running_max_is_monotone_nondecreasing() -> None:
    # Adjusted values of the *sorted* p-values must be non-decreasing.
    res = holm(P6, ALPHA)
    order = np.argsort(P6)
    adj_sorted = res.adjusted[order]
    assert np.all(np.diff(adj_sorted) >= -TOL)


# --------------------------------------------------------------------------- #
# Hochberg (and the strict Holm subset relation)
# --------------------------------------------------------------------------- #
def test_hochberg_matches_holm_on_monotone_example() -> None:
    res = hochberg(P6, ALPHA)
    assert res.method == "hochberg"
    assert res.adjusted == pytest.approx([0.03, 0.055, 0.08, 0.12, 0.26, 0.5], abs=TOL)
    assert res.rejected.tolist() == [True, False, False, False, False, False]


def test_hochberg_strictly_dominates_holm_with_gap() -> None:
    # p = [0.01, 0.04, 0.045], alpha = 0.05.
    #   Holm cummax[3*.01, 2*.04, 1*.045] = [0.03, 0.08, 0.08] -> reject {.01}
    #   Hochberg revcummin -> [0.03, 0.045, 0.045] -> reject all three.
    p = np.array([0.01, 0.04, 0.045])
    h_holm = holm(p, 0.05)
    h_hoch = hochberg(p, 0.05)
    assert h_holm.num_rejected == 1
    assert h_hoch.num_rejected == 3
    assert h_hoch.adjusted == pytest.approx([0.03, 0.045, 0.045], abs=TOL)
    # Holm subset of Hochberg elementwise.
    assert np.all(h_hoch.rejected | ~h_holm.rejected)


# --------------------------------------------------------------------------- #
# Benjamini-Hochberg
# --------------------------------------------------------------------------- #
def test_bh_pins_qvalues_and_rejection_set() -> None:
    res = benjamini_hochberg(P6, ALPHA)
    assert res.method == "benjamini_hochberg"
    assert res.adjusted == pytest.approx([0.03, 0.033, 0.04, 0.06, 0.156, 0.5], abs=TOL)
    assert res.rejected.tolist() == [True, True, True, False, False, False]
    assert res.num_rejected == 3
    assert res.threshold == pytest.approx(0.02, abs=TOL)


def test_bh_qvalues_are_monotone_in_sorted_order() -> None:
    res = benjamini_hochberg(P6, ALPHA)
    order = np.argsort(P6)
    q_sorted = res.adjusted[order]
    assert np.all(np.diff(q_sorted) >= -TOL)


# --------------------------------------------------------------------------- #
# Benjamini-Yekutieli
# --------------------------------------------------------------------------- #
def test_by_pins_qvalues_and_rejects_none() -> None:
    res = benjamini_yekutieli(P6, ALPHA)
    assert res.method == "benjamini_yekutieli"
    c6 = 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 5 + 1 / 6  # = 2.45
    assert c6 == pytest.approx(2.45, abs=TOL)
    expected = np.clip(np.array([0.03, 0.033, 0.04, 0.06, 0.156, 0.5]) * c6, 0.0, 1.0)
    assert res.adjusted == pytest.approx(expected, abs=TOL)
    assert res.num_rejected == 0
    assert res.threshold == 0.0


def test_bh_rejects_superset_of_by() -> None:
    bh = benjamini_hochberg(P6, ALPHA)
    by = benjamini_yekutieli(P6, ALPHA)
    assert np.all(bh.rejected | ~by.rejected)
    assert bh.num_rejected >= by.num_rejected


# --------------------------------------------------------------------------- #
# Storey q-values
# --------------------------------------------------------------------------- #
def test_storey_pi0_zero_rejects_everything() -> None:
    # No p exceeds lambda=0.5 (0.5 is not > 0.5) -> pi0_hat = 0 -> all q = 0.
    res = storey_qvalue(P6, ALPHA, lambda_=0.5)
    assert res.method == "storey_qvalue"
    assert np.all(res.adjusted == 0.0)
    assert res.num_rejected == 6
    assert res.threshold == pytest.approx(0.5, abs=TOL)


def test_storey_positive_pi0_pins_values() -> None:
    # p = [0.005,0.011,0.02,0.04,0.6,0.8]: 2 exceed 0.5 -> pi0_hat = 2/3.
    p = np.array([0.005, 0.011, 0.02, 0.04, 0.6, 0.8])
    res = storey_qvalue(p, ALPHA, lambda_=0.5)
    # BH_q here = [0.03,0.033,0.04,0.06,0.72,0.8]; * 2/3 -> first is 0.02.
    assert res.adjusted[0] == pytest.approx(0.02, abs=TOL)
    assert res.adjusted == pytest.approx(
        (2 / 3) * np.array([0.03, 0.033, 0.04, 0.06, 0.72, 0.8]), abs=TOL
    )
    assert res.rejected.tolist() == [True, True, True, True, False, False]
    assert res.threshold == pytest.approx(0.04, abs=TOL)


def test_storey_never_less_powerful_than_bh() -> None:
    # pi0_hat <= 1 so Storey q <= BH q -> Storey rejects a superset of BH.
    p = np.array([0.005, 0.011, 0.02, 0.04, 0.6, 0.8])
    st = storey_qvalue(p, ALPHA, lambda_=0.5)
    bh = benjamini_hochberg(p, ALPHA)
    assert np.all(st.adjusted <= bh.adjusted + TOL)
    assert np.all(st.rejected | ~bh.rejected)


def test_storey_rejects_invalid_lambda() -> None:
    with pytest.raises(ValueError):
        storey_qvalue(P6, ALPHA, lambda_=1.0)
    with pytest.raises(ValueError):
        storey_qvalue(P6, ALPHA, lambda_=-0.1)


def test_conservative_storey_uses_an_upper_null_count_bound() -> None:
    unconstrained = storey_qvalue(P6, ALPHA, lambda_=0.5)
    conservative = storey_qvalue_conservative(P6, ALPHA, lambda_=0.5)
    assert conservative.method == "storey_qvalue_conservative"
    assert np.all((0.0 <= conservative.adjusted) & (conservative.adjusted <= 1.0))
    assert np.all(conservative.adjusted >= unconstrained.adjusted - TOL)
    assert conservative.num_rejected <= unconstrained.num_rejected
    # For this small vector the exact 99% upper bound is the full null count,
    # so the conservative rule reduces to ordinary BH rather than the
    # anti-conservative zero-pi0 Storey plug-in.
    assert conservative.adjusted == pytest.approx(
        benjamini_hochberg(P6, ALPHA).adjusted, abs=TOL
    )


def test_conservative_storey_validates_confidence_and_lambda() -> None:
    with pytest.raises(ValueError, match="confidence"):
        storey_qvalue_conservative(P6, ALPHA, confidence=0.0)
    with pytest.raises(ValueError, match="confidence"):
        storey_qvalue_conservative(P6, ALPHA, confidence=1.0)
    with pytest.raises(ValueError, match="lambda"):
        storey_qvalue_conservative(P6, ALPHA, lambda_=1.0)
    with pytest.raises(ValueError, match="lambda"):
        storey_qvalue_conservative(P6, ALPHA, lambda_=-0.1)


def test_conservative_storey_is_permutation_equivariant() -> None:
    rng = np.random.default_rng(20260722)
    permutation = rng.permutation(P6.size)
    result = storey_qvalue_conservative(P6, ALPHA, confidence=0.95)
    permuted = storey_qvalue_conservative(P6[permutation], ALPHA, confidence=0.95)
    assert result.adjusted[permutation] == pytest.approx(permuted.adjusted, abs=TOL)
    assert np.array_equal(result.rejected[permutation], permuted.rejected)


def test_split_storey_uses_selection_only_for_the_null_count() -> None:
    selection = np.array([0.01, 0.02, 0.03, 0.04, 0.9, 0.95])
    confirmation = np.array([0.005, 0.011, 0.02, 0.04, 0.6, 0.8])
    result = storey_qvalue_split_confirmation(selection, confirmation, ALPHA)
    assert result.method == "storey_qvalue_split_confirmation"
    assert result.adjusted == pytest.approx(benjamini_hochberg(confirmation, ALPHA).adjusted, abs=TOL)
    altered_confirmation = storey_qvalue_split_confirmation(selection, confirmation[::-1], ALPHA)
    assert altered_confirmation.adjusted[::-1] == pytest.approx(result.adjusted, abs=TOL)
    assert np.array_equal(altered_confirmation.rejected[::-1], result.rejected)


def test_split_storey_validates_stream_contract() -> None:
    with pytest.raises(ValueError, match="matching"):
        storey_qvalue_split_confirmation(P6, P6[:-1], ALPHA)
    with pytest.raises(ValueError, match="lambda"):
        storey_qvalue_split_confirmation(P6, P6, ALPHA, lambda_=1.0)
    with pytest.raises(ValueError, match="confidence"):
        storey_qvalue_split_confirmation(P6, P6, ALPHA, confidence=1.0)
    empty = storey_qvalue_split_confirmation(np.array([]), np.array([]), ALPHA)
    assert empty.num_rejected == 0


def test_empty_and_dispatcher_error_contracts() -> None:
    empty = storey_qvalue_conservative(np.array([]), ALPHA)
    assert empty.num_rejected == 0
    with pytest.raises(ValueError, match="strictly positive"):
        weighted_bh(P6, np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0]), ALPHA)
    with pytest.raises(ValueError, match="method must be a string"):
        apply_correction(1, P6, ALPHA)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Adaptive BH (Benjamini-Krieger-Yekutieli)
# --------------------------------------------------------------------------- #
def test_adaptive_bh_middle_case_gains_power_over_bh() -> None:
    # Stage 1 BH at alpha/(1+alpha) rejects r1=3 -> m0_hat=3 ->
    # stage 2 BH at alpha*6/3=0.1 rejects the first 4.
    res = adaptive_bh(P6, ALPHA)
    assert res.method == "adaptive_bh"
    assert res.rejected.tolist() == [True, True, True, True, False, False]
    assert res.num_rejected == 4
    # adjusted = BH_q * (m0_hat/m) = [0.03,0.033,0.04,0.06,0.156,0.5]*0.5
    assert res.adjusted == pytest.approx(
        np.array([0.03, 0.033, 0.04, 0.06, 0.156, 0.5]) * 0.5, abs=TOL
    )
    assert res.threshold == pytest.approx(0.04, abs=TOL)
    # Never less powerful than plain BH here.
    assert res.num_rejected >= benjamini_hochberg(P6, ALPHA).num_rejected


def test_adaptive_bh_stage1_rejects_nothing() -> None:
    # All p large -> stage-1 BH rejects nothing -> BKY stops, rejects nothing.
    p = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.9])
    res = adaptive_bh(p, ALPHA)
    assert res.num_rejected == 0
    assert np.all(res.adjusted == 1.0)
    assert res.threshold == 0.0


def test_adaptive_bh_stage1_rejects_everything() -> None:
    # All p tiny -> stage-1 BH rejects all -> reject all, q = 0.
    p = np.array([0.001, 0.001, 0.001])
    res = adaptive_bh(p, ALPHA)
    assert res.num_rejected == 3
    assert np.all(res.adjusted == 0.0)
    assert res.threshold == pytest.approx(0.001, abs=TOL)


# --------------------------------------------------------------------------- #
# Cross-procedure power ordering (bonferroni <= holm <= hochberg; BH superset)
# --------------------------------------------------------------------------- #
def test_power_ordering_on_random_vectors() -> None:
    rng = np.random.default_rng(20260715)
    for _ in range(200):
        m = int(rng.integers(1, 25))
        # Mixture of nulls (U[0,1]) and strong signals near 0.
        p = rng.uniform(size=m)
        p[: m // 3] *= 0.02
        bonf = bonferroni(p, ALPHA)
        hlm = holm(p, ALPHA)
        hoch = hochberg(p, ALPHA)
        bh = benjamini_hochberg(p, ALPHA)
        by = benjamini_yekutieli(p, ALPHA)
        # Rejection-set containment.
        assert np.all(hlm.rejected | ~bonf.rejected)  # Bonferroni ⊆ Holm
        assert np.all(hoch.rejected | ~hlm.rejected)  # Holm ⊆ Hochberg
        assert np.all(bh.rejected | ~by.rejected)  # BY ⊆ BH
        assert np.all(bh.rejected | ~hlm.rejected)  # Holm ⊆ BH (FDR ⊇ FWER)
        # Adjusted bounds.
        for res in (bonf, hlm, hoch, bh, by):
            assert np.all(res.adjusted >= 0.0)
            assert np.all(res.adjusted <= 1.0)


# --------------------------------------------------------------------------- #
# Permutation equivariance (unsorted input)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "func",
    [bonferroni, sidak, holm, hochberg, benjamini_hochberg, benjamini_yekutieli],
)
def test_unsorted_input_is_permutation_equivariant(func) -> None:
    rng = np.random.default_rng(7)
    p = np.array([0.5, 0.005, 0.13, 0.011, 0.04, 0.02])  # a shuffle of P6
    perm = rng.permutation(6)
    res = func(p, ALPHA)
    res_perm = func(p[perm], ALPHA)
    assert res.adjusted[perm] == pytest.approx(res_perm.adjusted, abs=TOL)
    assert np.array_equal(res.rejected[perm], res_perm.rejected)
    assert res.num_rejected == res_perm.num_rejected


def test_shuffled_p6_recovers_sorted_rejection_set() -> None:
    shuffled = np.array([0.5, 0.005, 0.13, 0.011, 0.04, 0.02])
    res = benjamini_hochberg(shuffled, ALPHA)
    # The three smallest (0.005, 0.011, 0.02) at positions 1, 3, 5 are rejected.
    assert res.rejected.tolist() == [False, True, False, True, False, True]
    assert res.threshold == pytest.approx(0.02, abs=TOL)


# --------------------------------------------------------------------------- #
# Edge cases: empty, m==1, p==0, p==1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "func",
    [
        bonferroni,
        sidak,
        holm,
        hochberg,
        benjamini_hochberg,
        benjamini_yekutieli,
        storey_qvalue,
        adaptive_bh,
    ],
)
def test_empty_input(func) -> None:
    res = func(np.array([]), ALPHA)
    assert res.adjusted.shape == (0,)
    assert res.rejected.shape == (0,)
    assert res.num_rejected == 0
    assert res.threshold == 0.0


@pytest.mark.parametrize(
    "func",
    [
        bonferroni,
        sidak,
        holm,
        hochberg,
        benjamini_hochberg,
        benjamini_yekutieli,
        adaptive_bh,
    ],
)
def test_single_hypothesis_m1(func) -> None:
    # With m == 1 these procedures reduce to the raw test: reject iff p <= alpha.
    # (Storey is excluded: with no p > lambda its pi0_hat=0 collapses q to 0 and
    # rejects even a large p -- a documented small-sample estimator degeneracy.)
    rej = func(np.array([0.01]), ALPHA)
    assert rej.num_rejected == 1
    assert rej.threshold == pytest.approx(0.01, abs=TOL)
    non = func(np.array([0.2]), ALPHA)
    assert non.num_rejected == 0
    assert non.threshold == 0.0


def test_storey_m1_pi0_degeneracy() -> None:
    # m==1 with p <= lambda -> count_above=0 -> pi0_hat=0 -> q=0 -> rejects.
    rej = storey_qvalue(np.array([0.01]), ALPHA, lambda_=0.5)
    assert rej.num_rejected == 1
    assert rej.adjusted[0] == 0.0
    assert rej.threshold == pytest.approx(0.01, abs=TOL)
    # A single p > lambda -> count_above=1 -> pi0_hat=min(1, 1/0.5)=1 -> q=p.
    non = storey_qvalue(np.array([0.8]), ALPHA, lambda_=0.5)
    assert non.num_rejected == 0
    assert non.adjusted[0] == pytest.approx(0.8, abs=TOL)


@pytest.mark.parametrize(
    "func",
    [bonferroni, sidak, holm, hochberg, benjamini_hochberg, benjamini_yekutieli],
)
def test_single_hypothesis_m1_adjusted_equals_raw(func) -> None:
    # For FWER + BH/BY, the m==1 adjusted p-value equals the raw p-value.
    # (Storey collapses to q=0 via pi0_hat=0; adaptive_bh to the r1==m branch.)
    rej = func(np.array([0.01]), ALPHA)
    assert rej.adjusted[0] == pytest.approx(0.01, abs=TOL)


@pytest.mark.parametrize(
    "func",
    [
        bonferroni,
        sidak,
        holm,
        hochberg,
        benjamini_hochberg,
        benjamini_yekutieli,
        adaptive_bh,
    ],
)
def test_p_equals_zero_and_one(func) -> None:
    # p == 0 always rejected (adjusted 0); p == 1 never rejected.
    p = np.array([0.0, 1.0])
    res = func(p, ALPHA)
    assert res.rejected[0]  # p == 0 rejected
    assert not res.rejected[1]  # p == 1 not rejected
    assert res.adjusted[0] == pytest.approx(0.0, abs=TOL)
    assert res.threshold == pytest.approx(0.0, abs=TOL)


@pytest.mark.parametrize(
    "func",
    [bonferroni, sidak, holm, hochberg, benjamini_hochberg, benjamini_yekutieli],
)
def test_p_equals_one_adjusted_is_one(func) -> None:
    # For FWER + BH/BY the adjusted value of a raw p==1 stays exactly 1.0.
    # (adaptive_bh scales BH q by m0_hat/m so its p==1 adjusted can be < 1.)
    res = func(np.array([0.0, 1.0]), ALPHA)
    assert res.adjusted[1] == pytest.approx(1.0, abs=TOL)


def test_all_p_zero_bonferroni() -> None:
    res = bonferroni(np.zeros(4), ALPHA)
    assert res.num_rejected == 4
    assert np.all(res.adjusted == 0.0)
    assert res.threshold == 0.0


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def test_rejects_out_of_range_pvalues() -> None:
    with pytest.raises(ValueError):
        bonferroni(np.array([0.1, 1.5]), ALPHA)
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([-0.2, 0.3]), ALPHA)


def test_rejects_nan_pvalues_and_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="finite"):
        bonferroni(np.array([np.nan, 0.2]), ALPHA)
    with pytest.raises(ValueError, match="alpha"):
        benjamini_hochberg(P6, 0.0)
    with pytest.raises(ValueError, match="alpha"):
        benjamini_hochberg(P6, 1.0)
    with pytest.raises(ValueError, match="unknown correction method"):
        apply_correction("unknown", P6, ALPHA)


def test_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError):
        holm(np.array([[0.01, 0.02], [0.03, 0.04]]), ALPHA)


def test_accepts_tiny_floating_point_overshoot() -> None:
    # Survival functions can emit 1 + 1e-16; must clip, not raise.
    res = bonferroni(np.array([1.0 + 5e-16, 0.0]), ALPHA)
    assert res.adjusted[0] == pytest.approx(1.0, abs=TOL)
    assert not res.rejected[0]


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,func",
    [
        ("bonferroni", bonferroni),
        ("sidak", sidak),
        ("holm", holm),
        ("hochberg", hochberg),
        ("benjamini_hochberg", benjamini_hochberg),
        ("benjamini_yekutieli", benjamini_yekutieli),
        ("storey_qvalue", storey_qvalue),
        ("storey_qvalue_conservative", storey_qvalue_conservative),
        ("adaptive_bh", adaptive_bh),
    ],
)
def test_apply_correction_dispatch_matches_direct_call(name, func) -> None:
    via = apply_correction(name, P6, ALPHA)
    direct = func(P6, ALPHA)
    assert via.method == direct.method
    assert via.adjusted == pytest.approx(direct.adjusted, abs=TOL)
    assert np.array_equal(via.rejected, direct.rejected)


@pytest.mark.parametrize("noncanonical", ["Benjamini_Hochberg", " benjamini_hochberg "])
def test_apply_correction_rejects_noncanonical_spelling(noncanonical: str) -> None:
    with pytest.raises(ValueError, match="unknown correction method"):
        apply_correction(noncanonical, P6, ALPHA)


def test_apply_correction_forwards_kwargs_to_storey() -> None:
    res = apply_correction("storey_qvalue", P6, ALPHA, lambda_=0.5)
    assert res.method == "storey_qvalue"
    assert res.num_rejected == 6


def test_apply_correction_forwards_kwargs_to_conservative_storey() -> None:
    via = apply_correction(
        "storey_qvalue_conservative", P6, ALPHA, lambda_=0.5, confidence=0.95
    )
    direct = storey_qvalue_conservative(P6, ALPHA, lambda_=0.5, confidence=0.95)
    assert via.method == direct.method
    assert via.adjusted == pytest.approx(direct.adjusted, abs=TOL)


@pytest.mark.parametrize("alias", ["bh", "by", "storey", "storey_conservative", "bky"])
def test_apply_correction_rejects_noncanonical_aliases(alias: str) -> None:
    with pytest.raises(ValueError, match="unknown correction method"):
        apply_correction(alias, P6, ALPHA)


def test_apply_correction_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        apply_correction("bogus_method", P6, ALPHA)


# --------------------------------------------------------------------------- #
# CorrectionResult contract sanity
# --------------------------------------------------------------------------- #
def test_result_fields_and_property() -> None:
    res = benjamini_hochberg(P6, ALPHA)
    assert res.rejected.dtype == bool
    assert res.adjusted.dtype == float
    assert res.num_rejected == int(np.count_nonzero(res.rejected))
    assert 0.0 <= res.threshold <= 1.0


def test_weighted_bh_is_permutation_equivariant_and_validates_fixed_weights() -> None:
    pvalues = np.array([0.001, 0.02, 0.4, 0.8])
    weights = np.array([2.0, 1.0, 0.5, 0.5])
    result = weighted_bh(pvalues, weights)
    permutation = np.array([2, 0, 3, 1])
    permuted = weighted_bh(pvalues[permutation], weights[permutation])
    assert np.array_equal(result.rejected[permutation], permuted.rejected)
    with pytest.raises(ValueError):
        weighted_bh(pvalues, np.ones(3))
    with pytest.raises(ValueError):
        weighted_bh(pvalues, np.array([1.0, 1.0, 1.0, 2.0]))


def test_weighted_bh_empty_and_dispatcher_contracts() -> None:
    assert weighted_bh(np.array([]), np.array([])).adjusted.size == 0
    with pytest.raises(ValueError):
        apply_correction("weighted_bh", np.array([0.1]))


def test_weighted_bh_matches_normalized_step_up_rejection_rule() -> None:
    """Pin the weighted rule p_i / w_i <= alpha*k/m on a nontrivial vector."""
    pvalues = np.array([0.04, 0.015, 0.50])
    weights = np.array([2.0, 0.5, 0.5])
    result = weighted_bh(pvalues, weights, alpha=0.05)

    # Normalized p-values are [.02, .03, 1.0].  The largest admissible rank is
    # k=2 because .03 <= .05*2/3, so both first hypotheses are rejected.
    assert result.rejected.tolist() == [True, True, False]
    assert result.adjusted == pytest.approx([0.045, 0.045, 1.0], abs=TOL)
    with pytest.raises(ValueError):
        apply_correction("unknown", np.array([0.1]))
