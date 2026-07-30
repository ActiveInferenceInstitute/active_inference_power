"""Tests for ``error_rates`` — realized error decomposition.

No mocks: every test uses real numpy arrays with hand-computed expected values
or seeded RNG draws cross-checked against an independent brute-force loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from active_inference_power._types import ErrorSummary
from active_inference_power.error_rates import (
    aggregate,
    classify,
    confusion_counts,
    false_discovery_proportion,
    family_wise_error,
    true_positive_rate,
)

# ---------------------------------------------------------------------------
# confusion_counts
# ---------------------------------------------------------------------------


def test_confusion_counts_hand_computed():
    # m = 6: alt at positions 0,1,2 (m1=3). Rejections at 0,1,3.
    rejected = np.array([True, True, False, True, False, False])
    is_alt = np.array([True, True, True, False, False, False])
    v, s, r, m1 = confusion_counts(rejected, is_alt)
    assert (v, s, r, m1) == (1, 2, 3, 3)


def test_confusion_counts_r_equals_v_plus_s():
    rng = np.random.default_rng(42)
    for _ in range(20):
        m = int(rng.integers(1, 50))
        rejected = rng.random(m) < 0.4
        is_alt = rng.random(m) < 0.3
        v, s, r, m1 = confusion_counts(rejected, is_alt)
        # brute-force independent recount
        v_bf = sum(1 for i in range(m) if rejected[i] and not is_alt[i])
        s_bf = sum(1 for i in range(m) if rejected[i] and is_alt[i])
        m1_bf = sum(1 for i in range(m) if is_alt[i])
        assert (v, s, m1) == (v_bf, s_bf, m1_bf)
        assert r == v + s
        assert isinstance(v, int) and isinstance(s, int)
        assert isinstance(r, int) and isinstance(m1, int)


def test_confusion_counts_empty_input():
    empty = np.array([], dtype=bool)
    assert confusion_counts(empty, empty) == (0, 0, 0, 0)


def test_confusion_counts_single_hypothesis():
    # m == 1, four possible (rejected, is_alt) cells
    t, f = np.array([True]), np.array([False])
    assert confusion_counts(t, t) == (0, 1, 1, 1)  # true discovery
    assert confusion_counts(t, f) == (1, 0, 1, 0)  # false discovery
    assert confusion_counts(f, t) == (0, 0, 0, 1)  # miss
    assert confusion_counts(f, f) == (0, 0, 0, 0)  # correct non-rejection


def test_confusion_counts_accepts_numeric_masks():
    # 0/1 int and float arrays must coerce to booleans
    rejected = np.array([1, 0, 1, 0])
    is_alt = np.array([1.0, 1.0, 0.0, 0.0])
    assert confusion_counts(rejected, is_alt) == (1, 1, 2, 2)


def test_confusion_counts_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        confusion_counts(np.array([True, False]), np.array([True]))


# ---------------------------------------------------------------------------
# false_discovery_proportion
# ---------------------------------------------------------------------------


def test_fdp_hand_computed():
    rejected = np.array([True, True, False, True, False, False])
    is_alt = np.array([True, True, True, False, False, False])
    # V=1, R=3
    assert false_discovery_proportion(rejected, is_alt) == pytest.approx(1.0 / 3.0)


def test_fdp_zero_when_nothing_rejected():
    rejected = np.zeros(5, dtype=bool)
    is_alt = np.array([True, False, True, False, True])
    assert false_discovery_proportion(rejected, is_alt) == 0.0


def test_fdp_one_when_all_rejections_false():
    # everything null, everything rejected → FDP = 1
    rejected = np.ones(4, dtype=bool)
    is_alt = np.zeros(4, dtype=bool)
    assert false_discovery_proportion(rejected, is_alt) == 1.0


def test_fdp_zero_when_all_rejections_true():
    # everything alt, everything rejected → FDP = 0
    rejected = np.ones(4, dtype=bool)
    is_alt = np.ones(4, dtype=bool)
    assert false_discovery_proportion(rejected, is_alt) == 0.0


def test_fdp_empty_input():
    empty = np.array([], dtype=bool)
    assert false_discovery_proportion(empty, empty) == 0.0


# ---------------------------------------------------------------------------
# true_positive_rate
# ---------------------------------------------------------------------------


def test_tpr_hand_computed():
    rejected = np.array([True, True, False, True, False, False])
    is_alt = np.array([True, True, True, False, False, False])
    # S=2, m1=3
    assert true_positive_rate(rejected, is_alt) == pytest.approx(2.0 / 3.0)


def test_tpr_zero_when_m1_zero():
    # everything null: even with rejections, power is 0 (no alt to discover)
    rejected = np.array([True, True, False])
    is_alt = np.zeros(3, dtype=bool)
    assert true_positive_rate(rejected, is_alt) == 0.0


def test_tpr_one_when_all_alt_rejected():
    rejected = np.ones(5, dtype=bool)
    is_alt = np.ones(5, dtype=bool)
    assert true_positive_rate(rejected, is_alt) == 1.0


def test_tpr_zero_when_nothing_rejected():
    rejected = np.zeros(5, dtype=bool)
    is_alt = np.ones(5, dtype=bool)
    assert true_positive_rate(rejected, is_alt) == 0.0


def test_tpr_empty_input():
    empty = np.array([], dtype=bool)
    assert true_positive_rate(empty, empty) == 0.0


# ---------------------------------------------------------------------------
# family_wise_error
# ---------------------------------------------------------------------------


def test_fwer_event_true_iff_any_false_rejection():
    is_alt = np.array([True, False, False, True])
    assert family_wise_error(np.array([True, True, False, False]), is_alt) is True
    assert family_wise_error(np.array([True, False, False, True]), is_alt) is False


def test_fwer_no_event_when_nothing_rejected():
    rejected = np.zeros(4, dtype=bool)
    is_alt = np.array([True, False, True, False])
    assert family_wise_error(rejected, is_alt) is False


def test_fwer_empty_input():
    empty = np.array([], dtype=bool)
    assert family_wise_error(empty, empty) is False


def test_fwer_returns_python_bool():
    is_alt = np.array([False])
    assert isinstance(family_wise_error(np.array([True]), is_alt), bool)
    assert isinstance(family_wise_error(np.array([False]), is_alt), bool)


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


def test_classify_fills_every_field_hand_computed():
    rejected = np.array([True, True, False, True, False, False])
    is_alt = np.array([True, True, True, False, False, False])
    summary = classify(rejected, is_alt)
    assert isinstance(summary, ErrorSummary)
    assert summary.num_hypotheses == 6
    assert summary.num_alt == 3
    assert summary.num_rejected == 3
    assert summary.true_discoveries == 2
    assert summary.false_discoveries == 1
    assert summary.fdp == pytest.approx(1.0 / 3.0)
    assert summary.power == pytest.approx(2.0 / 3.0)
    assert summary.fwer_event is True
    assert summary.num_null == 3  # dataclass property: m - m1


def test_classify_nothing_rejected():
    rejected = np.zeros(5, dtype=bool)
    is_alt = np.array([True, True, False, False, False])
    summary = classify(rejected, is_alt)
    assert summary.num_rejected == 0
    assert summary.true_discoveries == 0
    assert summary.false_discoveries == 0
    assert summary.fdp == 0.0
    assert summary.power == 0.0
    assert summary.fwer_event is False


def test_classify_everything_null():
    rejected = np.array([True, False, True])
    is_alt = np.zeros(3, dtype=bool)
    summary = classify(rejected, is_alt)
    assert summary.num_alt == 0
    assert summary.num_null == 3
    assert summary.false_discoveries == 2
    assert summary.true_discoveries == 0
    assert summary.fdp == 1.0
    assert summary.power == 0.0  # m1 == 0 convention
    assert summary.fwer_event is True


def test_classify_everything_alt():
    rejected = np.array([True, True, False])
    is_alt = np.ones(3, dtype=bool)
    summary = classify(rejected, is_alt)
    assert summary.num_alt == 3
    assert summary.num_null == 0
    assert summary.false_discoveries == 0
    assert summary.fdp == 0.0
    assert summary.power == pytest.approx(2.0 / 3.0)
    assert summary.fwer_event is False


def test_classify_empty_input():
    empty = np.array([], dtype=bool)
    summary = classify(empty, empty)
    assert summary.num_hypotheses == 0
    assert summary.num_alt == 0
    assert summary.num_rejected == 0
    assert summary.fdp == 0.0
    assert summary.power == 0.0
    assert summary.fwer_event is False


def test_classify_single_hypothesis_false_discovery():
    summary = classify(np.array([True]), np.array([False]))
    assert summary.num_hypotheses == 1
    assert summary.fdp == 1.0
    assert summary.power == 0.0
    assert summary.fwer_event is True


def test_classify_matches_component_functions_randomized():
    """Cross-check identity: classify agrees with each standalone function."""
    rng = np.random.default_rng(7)
    for _ in range(25):
        m = int(rng.integers(1, 40))
        rejected = rng.random(m) < rng.uniform(0.0, 1.0)
        is_alt = rng.random(m) < rng.uniform(0.0, 1.0)
        summary = classify(rejected, is_alt)
        v, s, r, m1 = confusion_counts(rejected, is_alt)
        assert summary.false_discoveries == v
        assert summary.true_discoveries == s
        assert summary.num_rejected == r
        assert summary.num_alt == m1
        assert summary.fdp == pytest.approx(
            false_discovery_proportion(rejected, is_alt)
        )
        assert summary.power == pytest.approx(true_positive_rate(rejected, is_alt))
        assert summary.fwer_event == family_wise_error(rejected, is_alt)
        # structural identities
        assert 0.0 <= summary.fdp <= 1.0
        assert 0.0 <= summary.power <= 1.0
        assert (
            summary.num_rejected == summary.true_discoveries + summary.false_discoveries
        )
        assert summary.num_hypotheses == summary.num_alt + summary.num_null


def test_classify_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        classify(np.array([True]), np.array([True, False]))


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def _summary(rejected: list[bool], is_alt: list[bool]) -> ErrorSummary:
    return classify(np.array(rejected), np.array(is_alt))


def test_aggregate_empty_list_all_zero():
    assert aggregate([]) == (0.0, 0.0, 0.0, 0.0)


def test_aggregate_hand_computed_means():
    # Rep 1: V=1,S=2,R=3,m1=3 → fdp=1/3, power=2/3, fwer=True
    s1 = _summary(
        [True, True, False, True, False, False],
        [True, True, True, False, False, False],
    )
    # Rep 2: nothing rejected → fdp=0, power=0, fwer=False, R=0
    s2 = _summary([False] * 6, [True, True, True, False, False, False])
    # Rep 3: V=0,S=3,R=3 → fdp=0, power=1, fwer=False
    s3 = _summary(
        [True, True, True, False, False, False],
        [True, True, True, False, False, False],
    )
    fdr, power, fwer, mean_rej = aggregate([s1, s2, s3])
    assert fdr == pytest.approx((1.0 / 3.0 + 0.0 + 0.0) / 3.0)
    assert power == pytest.approx((2.0 / 3.0 + 0.0 + 1.0) / 3.0)
    assert fwer == pytest.approx(1.0 / 3.0)
    assert mean_rej == pytest.approx((3 + 0 + 3) / 3.0)


def test_aggregate_single_summary_is_identity():
    s = _summary([True, False, True], [False, False, True])
    fdr, power, fwer, mean_rej = aggregate([s])
    assert fdr == pytest.approx(s.fdp)
    assert power == pytest.approx(s.power)
    assert fwer == pytest.approx(1.0)  # s has a false discovery
    assert mean_rej == pytest.approx(float(s.num_rejected))


def test_aggregate_all_clean_reps_zero_fwer():
    # Every rep rejects only true alternatives → fdr 0, fwer 0
    reps = [
        _summary([True, True, False, False], [True, True, False, False])
        for _ in range(4)
    ]
    fdr, power, fwer, mean_rej = aggregate(reps)
    assert fdr == 0.0
    assert power == 1.0
    assert fwer == 0.0
    assert mean_rej == 2.0


def test_aggregate_monte_carlo_fwer_matches_bernoulli_mean():
    """Seeded MC cross-check: fwer equals fraction of reps with V >= 1."""
    rng = np.random.default_rng(123)
    m = 20
    summaries = []
    expected_fwer_events = 0
    total_rejections = 0
    n_reps = 50
    for _ in range(n_reps):
        rejected = rng.random(m) < 0.15
        is_alt = rng.random(m) < 0.3
        summaries.append(classify(rejected, is_alt))
        if np.any(rejected & ~is_alt):
            expected_fwer_events += 1
        total_rejections += int(np.count_nonzero(rejected))
    fdr, power, fwer, mean_rej = aggregate(summaries)
    assert fwer == pytest.approx(expected_fwer_events / n_reps)
    assert mean_rej == pytest.approx(total_rejections / n_reps)
    assert fdr == pytest.approx(float(np.mean([s.fdp for s in summaries])))
    assert power == pytest.approx(float(np.mean([s.power for s in summaries])))
    assert 0.0 <= fdr <= 1.0 and 0.0 <= power <= 1.0 and 0.0 <= fwer <= 1.0


def test_aggregate_returns_python_floats():
    s = _summary([True], [True])
    out = aggregate([s])
    assert all(isinstance(x, float) for x in out)
