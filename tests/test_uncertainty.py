"""Real finite-simulation precision and sample-size accounting tests."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference_power.uncertainty import (
    clopper_pearson_interval,
    empirical_bernstein_radius,
    hoeffding_radius,
    mean_interval,
    paired_difference_summary,
    paired_numeric_difference_summary,
    required_reps_for_half_width,
    summarize_bounded,
    wilson_interval,
)


def test_hoeffding_budget_is_conservative_and_monotone() -> None:
    small = required_reps_for_half_width(0.05)
    tight = required_reps_for_half_width(0.02)
    assert tight > small
    assert hoeffding_radius(tight) <= 0.02


def test_summarize_bounded_records_precision_and_bounds() -> None:
    summary = summarize_bounded(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert summary["n"] == 5
    assert summary["mean"] == pytest.approx(0.5)
    assert 0.0 <= summary["normal_ci_lower"] <= summary["normal_ci_upper"] <= 1.0
    assert summary["normal_half_width"] >= 0.0
    assert summary["hoeffding_radius"] > summary["normal_half_width"]


@pytest.mark.parametrize(
    "values",
    [[], np.array([[0.1, 0.2]]), np.array([np.nan]), np.array([-0.1, 0.2])],
)
def test_summarize_bounded_rejects_invalid_samples(values: object) -> None:
    with pytest.raises(ValueError):
        summarize_bounded(values)  # type: ignore[arg-type]


def test_uncertainty_validates_inputs() -> None:
    with pytest.raises(ValueError):
        hoeffding_radius(0)
    with pytest.raises(ValueError):
        required_reps_for_half_width(0.0)


def test_exact_intervals_and_paired_accounting() -> None:
    wilson = wilson_interval(5, 10)
    exact = clopper_pearson_interval(5, 10)
    assert 0.0 <= wilson[0] <= 0.5 <= wilson[1] <= 1.0
    assert 0.0 <= exact[0] <= 0.5 <= exact[1] <= 1.0
    summary = paired_difference_summary([0.1, 0.2, 0.3], [0.2, 0.25, 0.35])
    assert summary["mean_difference"] == pytest.approx(1.0 / 15.0)
    assert summary["hoeffding_radius"] > 0.0
    assert empirical_bernstein_radius([0.1, 0.2, 0.3]) > 0.0
    assert summarize_bounded([0.1, 0.2], target_half_width=0.1)["required_n_for_target"] > 0
    assert paired_difference_summary([0.2], [0.3])["half_width"] == 0.0
    assert clopper_pearson_interval(0, 1)[0] == 0.0
    assert clopper_pearson_interval(1, 1)[1] == 1.0
    with pytest.raises(ValueError):
        summarize_bounded([0.1], confidence=1.0)
    with pytest.raises(ValueError):
        summarize_bounded([0.1], target_half_width=0.0)
    with pytest.raises(ValueError):
        empirical_bernstein_radius([0.1])
    with pytest.raises(ValueError):
        paired_difference_summary([0.1], [0.2, 0.3])


def test_unbounded_mean_and_paired_intervals_preserve_scale() -> None:
    assert mean_interval([2.0], 0.95) == (2.0, 2.0)
    summary = paired_numeric_difference_summary([10.0, 12.0], [13.0, 16.0])
    assert summary["mean_difference"] == pytest.approx(3.5)
    assert summary["ci_lower"] < summary["mean_difference"] < summary["ci_upper"]


def test_unbounded_paired_intervals_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        paired_numeric_difference_summary([np.nan], [1.0])
