"""Real-value tests for the testing-setting factorial sensitivity surface."""

from __future__ import annotations

import json

import numpy as np
import pytest

from active_inference_power import (
    ACTION_LOOP_E_PROCESS,
    ACTION_LOOP_POSTERIOR,
    FIXED_FAMILY_PVALUES,
    SettingSpec,
    TestingSettingSensitivityPlan,
    TwoGroupsConfig,
    evaluate_testing_setting_sensitivity,
)
from active_inference_power.corrections import apply_correction
from active_inference_power.error_rates import classify
from active_inference_power.simulation import generate_pvalues
from active_inference_power.testing_setting_sensitivity import (
    _action_scenario,
    _numeric_metric,
)


def _small_plan(**overrides: object) -> TestingSettingSensitivityPlan:
    values: dict[str, object] = {
        "alpha_grid": (0.05,),
        "horizon_grid": (2,),
        "stopping_budget_grid": (2.5,),
        "evidence_interfaces": (
            FIXED_FAMILY_PVALUES,
            ACTION_LOOP_POSTERIOR,
            ACTION_LOOP_E_PROCESS,
        ),
        "fixed_family_corrections": ("bonferroni", "benjamini_hochberg"),
        "action_policies": ("fixed_horizon", "cost_aware"),
        "fixed_family_config": TwoGroupsConfig(
            m=16,
            pi0=0.5,
            effect=2.5,
            dependence_kind="independent",
        ),
        "action_reliability": 0.72,
        "n_reps": 3,
        "seed": 41,
        "confidence": 0.9,
        "target_mc_half_width": 0.2,
    }
    values.update(overrides)
    return TestingSettingSensitivityPlan(**values)  # type: ignore[arg-type]


def test_factorial_surface_is_deterministic_and_separates_interfaces() -> None:
    plan = _small_plan()
    left = evaluate_testing_setting_sensitivity(plan)
    right = evaluate_testing_setting_sensitivity(plan)

    assert left == right
    json.dumps(left, sort_keys=True)
    assert left["plan_hash"] == plan.configuration_hash
    assert left["sample_plan"]["factorial_interface_cells"] == 3
    assert left["sample_plan"]["surface_rows"] == 5
    assert left["sample_plan"]["metrics_reported_separately"] == [
        "power",
        "fdr",
        "fwer",
        "cost",
        "stopping",
    ]
    assert left["pooled_utility_or_policy_ranking"] == "not_reported"

    rows = left["surface"]
    assert len({row["settings_hash"] for row in rows}) == len(rows)
    assert all(len(row["scenario_hash"]) == 64 for row in rows)
    assert all(len(row["settings_hash"]) == 64 for row in rows)

    fixed = [row for row in rows if row["interface"]["name"] == FIXED_FAMILY_PVALUES]
    posterior = [
        row for row in rows if row["interface"]["name"] == ACTION_LOOP_POSTERIOR
    ]
    e_process = [
        row for row in rows if row["interface"]["name"] == ACTION_LOOP_E_PROCESS
    ]
    assert {row["fixed_family_correction"] for row in fixed} == {
        "bonferroni",
        "benjamini_hochberg",
    }
    assert {row["action_policy"] for row in posterior} == {
        "fixed_horizon",
        "cost_aware",
    }
    assert e_process[0]["action_policy"] == "e_process"
    assert all(row["fixed_family_correction"] is None for row in posterior + e_process)

    for row in fixed:
        assert row["interface"]["kind"] == "fixed_family"
        assert not row["metrics"]["cost"]["applicable"]
        assert not row["metrics"]["stopping"]["applicable"]
        assert row["metrics"]["cost"]["estimate"] is None
    for row in posterior + e_process:
        assert row["interface"]["kind"] == "action_loop"
        assert row["interface"]["action_loop_correction"] == "none"
        assert all(row["metrics"][name]["applicable"] for name in row["metrics"])
        assert row["metrics"]["cost"]["estimate"] >= 0.0
        assert row["metrics"]["stopping"]["estimate"] >= 0.0


def test_fixed_family_corrections_use_the_same_real_replication_draws() -> None:
    plan = _small_plan(
        evidence_interfaces=(FIXED_FAMILY_PVALUES,),
        fixed_family_corrections=("bonferroni", "benjamini_hochberg"),
        n_reps=5,
    )
    result = evaluate_testing_setting_sensitivity(plan)
    rows = result["surface"]
    assert len(rows) == 2
    stream_hashes = {row["common_random_numbers"]["stream_hash"] for row in rows}
    assert len(stream_hashes) == 1
    stream = result["streams"][stream_hashes.pop()]
    children = np.random.SeedSequence(stream["cell_seed"]).spawn(plan.n_reps)
    replication_seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in children
    ]
    draws = [
        generate_pvalues(plan.fixed_family_config, np.random.default_rng(seed))
        for seed in replication_seeds
    ]

    for row in rows:
        correction = row["fixed_family_correction"]
        summaries = [
            classify(apply_correction(correction, pvalues, 0.05).rejected, is_alt)
            for pvalues, is_alt in draws
        ]
        assert row["metrics"]["power"]["estimate"] == pytest.approx(
            float(np.mean([summary.power for summary in summaries]))
        )
        assert row["metrics"]["fdr"]["estimate"] == pytest.approx(
            float(np.mean([summary.fdp for summary in summaries]))
        )
        assert row["metrics"]["fwer"]["estimate"] == pytest.approx(
            float(np.mean([summary.fwer_event for summary in summaries]))
        )


def test_full_factorial_dimensions_are_hashed_without_policy_ranking() -> None:
    plan = _small_plan(
        alpha_grid=(0.025, 0.05),
        horizon_grid=(2, 3),
        stopping_budget_grid=(1.0, 3.0),
        evidence_interfaces=(FIXED_FAMILY_PVALUES,),
        fixed_family_corrections=("benjamini_hochberg",),
        n_reps=2,
    )
    result = evaluate_testing_setting_sensitivity(plan)

    assert result["sample_plan"]["factorial_interface_cells"] == 8
    assert len(result["surface"]) == 8
    assert len(result["streams"]) == 8
    assert len({row["settings_hash"] for row in result["surface"]}) == 8
    assert "ranking" not in result
    assert "utility" not in result
    assert all(
        row["settings"]["fixed_family_correction"] == "benjamini_hochberg"
        for row in result["surface"]
    )
    assert _small_plan().configuration_hash != plan.configuration_hash


@pytest.mark.parametrize(
    "overrides",
    [
        {"alpha_grid": (0.05, 0.05)},
        {"alpha_grid": ()},
        {"alpha_grid": (True,)},
        {"horizon_grid": (0,)},
        {"stopping_budget_grid": (True,)},
        {"stopping_budget_grid": (float("inf"),)},
        {"evidence_interfaces": ("unknown",)},
        {"evidence_interfaces": (" fixed_family_pvalues ",)},
        {"fixed_family_corrections": ("storey_qvalue",)},
        {"fixed_family_corrections": ("Benjamini_Hochberg",)},
        {"action_policies": ("e_process",)},
        {"action_policies": (" cost_aware ",)},
        {"fixed_family_config": object()},
        {
            "fixed_family_config": TwoGroupsConfig(
                m=0,
                pi0=0.5,
                effect=2.5,
                dependence_kind="independent",
            )
        },
        {"action_reliability": 0.5},
        {"n_reps": 0},
        {"seed": -1},
        {"target_mc_half_width": 1.0},
    ],
)
def test_plan_rejects_invalid_or_cross_interface_contracts(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _small_plan(**overrides)


def test_action_loop_correction_boundary_remains_in_force() -> None:
    with pytest.raises(ValueError, match="action-loop correction"):
        SettingSpec(correction="benjamini_hochberg")

    result = evaluate_testing_setting_sensitivity(
        _small_plan(
            evidence_interfaces=(ACTION_LOOP_POSTERIOR, ACTION_LOOP_E_PROCESS),
            action_policies=("fixed_horizon",),
            n_reps=2,
        )
    )
    for row in result["surface"]:
        assert row["interface"]["action_loop_correction"] == "none"
        assert row["fixed_family_correction"] is None
        assert row["settings"]["action_policy"] == row["policy"]


def test_optional_precision_plan_and_numeric_metrics_have_real_edge_contracts() -> None:
    plan = _small_plan(
        evidence_interfaces=(FIXED_FAMILY_PVALUES,),
        target_mc_half_width=None,
    )
    result = evaluate_testing_setting_sensitivity(plan)

    assert result["sample_plan"]["target_mc_half_width"] is None
    assert result["sample_plan"]["required_reps_for_target_half_width"] is None
    singleton = _numeric_metric((1.25,), plan)
    assert singleton["estimate"] == pytest.approx(1.25)
    assert singleton["standard_error"] == 0.0
    with pytest.raises(ValueError, match="must be finite"):
        _numeric_metric((float("nan"),), plan)


def test_internal_action_and_evaluation_boundaries_fail_closed() -> None:
    plan = _small_plan(evidence_interfaces=(ACTION_LOOP_POSTERIOR,))
    with pytest.raises(ValueError, match="not an action-loop"):
        _action_scenario(
            plan=plan,
            alpha=0.05,
            horizon=2,
            stopping_budget=2.5,
            interface=FIXED_FAMILY_PVALUES,
            policy="fixed_horizon",
        )
    with pytest.raises(ValueError, match="plan must"):
        evaluate_testing_setting_sensitivity(object())  # type: ignore[arg-type]

    # Frozen construction rejects unknown interfaces. This only models a corrupt
    # in-memory object and verifies that the evaluator still fails closed.
    object.__setattr__(plan, "evidence_interfaces", ("unknown",))
    with pytest.raises(RuntimeError, match="unrecognized evidence interface"):
        evaluate_testing_setting_sensitivity(plan)


def test_plan_rejects_non_canonical_string_entries() -> None:
    """Frozen construction fails closed on non-string grid entries."""
    for field in ("evidence_interfaces", "fixed_family_corrections", "action_policies"):
        with pytest.raises(ValueError, match="canonical strings"):
            _small_plan(**{field: (1,)})
