"""Targeted branch coverage for the composable suite contracts.

These are deliberately small real numerical cases: they exercise failure-safe
boundaries and degenerate-but-valid paths that ordinary release simulations do
not encounter often enough to cover.
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace
from typing import Any, cast

from active_inference_power.action_loop import (
    _choose_action,
    _expected_posterior_entropy,
    _normalize,
    _single_step_pvalue,
    _target_beliefs,
    run_action_loop,
)
from active_inference_power.formalism import formalism_registry
from active_inference_power.action_statistics import metrics_for_trace, paired_action_policy_comparisons
from active_inference_power.scenario import (
    GenerativeModelSpec,
    GenerativeProcessSpec,
    PowerScenario,
    SettingSpec,
    make_reference_action_scenario,
)
from active_inference_power.schema import (
    validate_results,
    validate_run_log,
    validate_suite_result,
)
from active_inference_power.benchmarks import BenchmarkSpec, load_agent_trace
from active_inference_power._types import CorrectionResult


def _one_state_scenario(*, target_action: int = 0, observations: int = 1) -> PowerScenario:
    """Build a finite one-target scenario with an explicit action contract."""
    likelihood = np.zeros((1, observations, 1), dtype=float)
    likelihood[0, -1, 0] = 1.0
    model = GenerativeModelSpec(
        ("s",), tuple(f"o{i}" for i in range(observations)), ("a",),
        np.array([1.0]), likelihood, np.array([[[1.0]]]),
        np.array([0]), np.array([0]), np.array([0]), np.array([target_action]),
        np.array([0]), ("h",),
    )
    process = GenerativeProcessSpec(
        np.array([1.0]), np.array([[[1.0]]]), likelihood.copy(),
        np.array([True]), np.array([1.0]),
    )
    return PowerScenario(
        "branch_scenario", model, process, SettingSpec(family_size=1, max_steps=1),
        "A branch-coverage scenario.",
    )


def _base_results() -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "run_id": "run",
        "meta": {"config_sha256": "a" * 64, "seed": 1, "n_reps": 2},
        **{
            key: {}
            for key in (
                "experiment", "certificates", "method_comparison", "paired_comparisons",
                "active_inference", "sequential", "online_fdr", "agent_sampling",
                "benchmarks", "null_pvalue_calibration", "sample_size_accounting",
            )
        },
        "power_curve": [],
        "dependence_sweep": [],
        "formalism": formalism_registry().to_mapping(),
    }


def test_action_loop_degenerate_visible_and_invalid_paths() -> None:
    scenario = make_reference_action_scenario(horizon=2)
    posterior = np.zeros(scenario.model.n_states)
    posterior[0] = 1.0
    beliefs = _target_beliefs(posterior, scenario)
    assert beliefs[1] == 0.0
    assert _choose_action("fixed_horizon", posterior, scenario, 0) >= 0
    assert _choose_action("adaptive_target", posterior, scenario, 0) >= 0
    from dataclasses import replace

    immediate = replace(
        scenario,
        setting=replace(scenario.setting, posterior_threshold=0.0),
    )
    assert run_action_loop(immediate, "posterior_threshold", seed=3, use_pymdp=False).stopping_time == 0
    with pytest.raises(ValueError):
        _normalize(np.array([0.0, np.nan]))
    assert _single_step_pvalue(scenario, 4, 0)[0] == 1.0


def test_action_loop_empty_action_and_zero_mass_entropy_paths() -> None:
    no_target = _one_state_scenario(target_action=-1)
    posterior = np.array([1.0])
    assert _choose_action("cost_aware", posterior, no_target, 0) == 0
    assert _expected_posterior_entropy(posterior, no_target, 0) == 0.0
    zero_mass = _one_state_scenario(observations=2)
    assert _expected_posterior_entropy(np.array([1.0]), zero_mass, 0) == 0.0


def test_action_loop_policy_and_evidence_failure_branches() -> None:
    scenario = make_reference_action_scenario(horizon=2)
    posterior = scenario.model.prior.copy()
    with pytest.raises(ValueError):
        _choose_action("posterior_sampling", posterior, scenario, 0)
    # No action for target 1: adaptive selection falls back to a feasible action.
    action_target = np.zeros_like(scenario.model.action_target)
    adaptive_model = replace(scenario.model, action_target=action_target)
    adaptive = replace(scenario, model=adaptive_model)
    assert _choose_action("adaptive_target", adaptive.model.prior, adaptive, 0) >= 0
    with pytest.raises(ValueError):
        _single_step_pvalue(replace(scenario, model=replace(scenario.model, prior=np.array([0.5] + [0.0] * 7))), 0, 1)
    deterministic = scenario.process.likelihood.copy()
    deterministic[:, 1, :] = 1.0
    deterministic[:, 0, :] = 0.0
    with pytest.raises(ValueError):
        _single_step_pvalue(replace(scenario, process=replace(scenario.process, likelihood=deterministic)), 0, 1)
    no_target = _one_state_scenario(target_action=-1)
    trace = run_action_loop(no_target, "fixed_horizon", seed=2, use_pymdp=False)
    assert trace.actions == ("a",)
    assert trace.target_e_process_values == (1.0,)


def test_action_statistics_fail_closed_and_single_policy_pair() -> None:
    scenario = make_reference_action_scenario(horizon=2)
    trace = run_action_loop(scenario, "fixed_horizon", seed=5, use_pymdp=False)
    with pytest.raises(ValueError):
        metrics_for_trace(scenario, replace(trace, rejected_targets=(99,)))
    with pytest.raises(ValueError):
        metrics_for_trace(scenario, replace(trace, rejected_targets=(0, 0)))
    with pytest.raises(ValueError):
        paired_action_policy_comparisons(scenario, (), 1, run_trace=run_action_loop)
    with pytest.raises(ValueError):
        paired_action_policy_comparisons(scenario, ("fixed_horizon", "fixed_horizon"), 1, run_trace=run_action_loop)
    with pytest.raises(ValueError):
        paired_action_policy_comparisons(scenario, ("fixed_horizon",), 1, confidence=1.0, run_trace=run_action_loop)


def test_action_loop_reverse_evidence_branch_is_explicit() -> None:
    scenario = make_reference_action_scenario(horizon=2)
    reversed_likelihood = scenario.process.likelihood.copy()
    # For target 0, make high observations more likely under the null than the
    # alternative; this exercises the reverse likelihood-ratio branch.
    target0 = scenario.model.state_target == 0
    reversed_likelihood[0, 1, target0 & (scenario.model.state_truth == 0)] = 0.8
    reversed_likelihood[0, 0, target0 & (scenario.model.state_truth == 0)] = 0.2
    reversed_likelihood[0, 1, target0 & (scenario.model.state_truth == 1)] = 0.2
    reversed_likelihood[0, 0, target0 & (scenario.model.state_truth == 1)] = 0.8
    from dataclasses import replace

    reverse = replace(scenario.process, likelihood=reversed_likelihood)
    altered = replace(scenario, process=reverse)
    assert _single_step_pvalue(altered, 0, 0)[0] == pytest.approx(0.2)
    assert _single_step_pvalue(altered, 0, 1)[0] == pytest.approx(1.0)


def test_results_suite_and_action_loop_schema_branches() -> None:
    result = _base_results()
    operation = {
        "scenario_id": "s", "scenario_hash": "b" * 64, "policy": "fixed_horizon",
        "n_reps": 1, "summary": {}, "validity": "diagnostic",
    }
    result["suite"] = {
        "suite_name": "suite", "scenarios": ["s"], "policies": ["fixed_horizon"],
        "operating_characteristics": [operation], "online_fdr": {}, "claim_boundary": "conditional",
    }
    result["action_loop"] = {
        "scenarios": [{"scenario_id": "s", "model": {}, "process": {}, "setting": {}}],
    }
    validate_results(result)
    with pytest.raises(ValueError, match="results.suite"):
        validate_results({key: value for key, value in result.items() if key != "suite"}, require_suite=True)
    with pytest.raises(ValueError, match="results.action_loop"):
        validate_results({key: value for key, value in result.items() if key != "action_loop"}, require_suite=True)
    validate_suite_result(result["suite"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_results({**_base_results(), "suite": {"suite_name": "x"}})
    with pytest.raises(ValueError):
        validate_results({**_base_results(), "action_loop": {"scenarios": {}}})
    with pytest.raises(ValueError):
        validate_results({**_base_results(), "action_loop": {"scenarios": [{}]}})
    with pytest.raises(ValueError):
        validate_results({**_base_results(), "action_loop": {"scenarios": [{"scenario_id": ""}]}})
    with pytest.raises(ValueError):
        validate_suite_result({"suite_name": "x", "scenarios": [], "policies": ["x"], "operating_characteristics": [operation], "online_fdr": {}, "claim_boundary": "x"})


def test_schema_calibration_and_checkpoint_lineage_boundaries() -> None:
    operation = {
        "scenario_id": "s", "scenario_hash": "b" * 64, "policy": "fixed_horizon",
        "n_reps": 1, "summary": {}, "validity": "diagnostic",
    }
    suite: dict[str, Any] = {
        "suite_name": "suite", "scenarios": ["s"], "policies": ["fixed_horizon"],
        "operating_characteristics": [operation], "online_fdr": {}, "claim_boundary": "conditional",
        "conditional_selection_calibration": {
            "scenario_id": "null", "scenario_hash": "c" * 64, "policy": "adaptive_target",
            "n_reps": 1, "pooled": {"n_selected": 1, "max_ecdf_excess": 0.1, "dkw_radius": 0.2, "passes_dkw_diagnostic": True},
            "validity": "calibration_diagnostic",
        },
    }
    validate_suite_result(suite)
    calibration = cast(dict[str, Any], suite["conditional_selection_calibration"])
    bad_values: tuple[dict[str, Any], ...] = (
        {key: value for key, value in calibration.items() if key != "scenario_hash"},
        {**calibration, "scenario_hash": "short"},
        {**calibration, "n_reps": 0},
        {**calibration, "validity": "formal"},
        {**calibration, "pooled": {}},
    )
    for bad in bad_values:
        with pytest.raises(ValueError):
            validate_suite_result({**suite, "conditional_selection_calibration": bad})
    with pytest.raises(ValueError):
        validate_suite_result({**suite, "conditional_selection_calibration": {**calibration, "n_reps": 0}})
    checkpoint: dict[str, Any] = {"schema_version": "2.0", "run_id": "run", "config_sha256": "a" * 64, "claim_ledger_sha256": "b" * 64, "profile": "fast", "seed": 1, "stage_seed": 2, "seed_schedule": {"kind": "numpy.SeedSequence", "master_seed": 1, "stage": "power", "stage_seed": 2, "replication_order": "stable ascending child-stream order"}, "workers": 1, "stage": "power", "completed_replications": 1, "partial_metric_accumulators": {"status": "stage_complete", "completed_replications": 1}, "dependency_hashes": {"scenario_hashes": [], "model_hashes": [], "process_hashes": [], "setting_hashes": []}, "value": {}}
    from active_inference_power.schema import validate_checkpoint
    with pytest.raises(ValueError):
        validate_checkpoint({**checkpoint, "seed_schedule": {**checkpoint["seed_schedule"], "stage": "other"}})


def test_scenario_probability_and_setting_contract_branches() -> None:
    scenario = make_reference_action_scenario(horizon=2)
    with pytest.raises(ValueError):
        GenerativeModelSpec(
            scenario.model.state_names, scenario.model.observation_names, scenario.model.action_names,
            scenario.model.prior, scenario.model.likelihood, np.zeros_like(scenario.model.transition),
            scenario.model.state_truth, scenario.model.state_target, scenario.model.state_context,
            scenario.model.action_target, scenario.model.action_sensor, scenario.model.hypothesis_ids,
        )
    with pytest.raises(ValueError):
        GenerativeProcessSpec(
            scenario.process.initial_state, scenario.process.transition, scenario.process.likelihood,
            scenario.process.target_truth, np.full_like(scenario.process.action_costs, -1.0),
        )
    with pytest.raises(ValueError):
        GenerativeProcessSpec(
            scenario.process.initial_state, scenario.process.transition[:, :, :1], scenario.process.likelihood,
            scenario.process.target_truth, scenario.process.action_costs,
        )
    for kwargs in (
        {"correction": "invalid"}, {"stopping_policy": "invalid"},
        {"cost_budget": float("nan")}, {"family_size": 0}, {"formal_validity": ""},
        {"evidence_assumptions": ("x", "x")}, {"evidence_assumptions": ("",)},
    ):
        with pytest.raises(ValueError):
            SettingSpec(**kwargs)
    with pytest.raises(ValueError):
        PowerScenario("", scenario.model, scenario.process, scenario.setting, "description")
    with pytest.raises(ValueError):
        PowerScenario("x", scenario.model, scenario.process, replace(scenario.setting, family_size=1), "description")
    with pytest.raises(ValueError):
        PowerScenario("x", scenario.model, replace(scenario.process, initial_state=np.ones(9) / 9, transition=np.repeat(np.eye(9)[None, :, :], 6, axis=0), likelihood=np.ones((6, 2, 9)) / 2), scenario.setting, "description")
    with pytest.raises(ValueError):
        PowerScenario("x", scenario.model, replace(scenario.process, transition=scenario.process.transition[:5], likelihood=scenario.process.likelihood[:5], action_costs=scenario.process.action_costs[:5]), scenario.setting, "description")
    with pytest.raises(ValueError):
        PowerScenario("x", scenario.model, replace(scenario.process, likelihood=np.ones((6, 3, 8)) / 3), scenario.setting, "description")
    with pytest.raises(ValueError):
        PowerScenario("x", scenario.model, replace(scenario.process, target_truth=np.array([True])), scenario.setting, "description")
    with pytest.raises(ValueError):
        PowerScenario("x", scenario.model, scenario.process, scenario.setting, "description", claim_role="")


def test_suite_empty_trace_and_benchmark_checksum_boundaries(tmp_path) -> None:
    scenario = make_reference_action_scenario(horizon=1)
    immediate = replace(
        scenario,
        setting=replace(scenario.setting, posterior_threshold=0.0),
    )
    with pytest.raises(ValueError, match="target evidence"):
        from active_inference_power.suite import evaluate_adaptive_target_selection

        evaluate_adaptive_target_selection(immediate, "posterior_threshold", 1)
    from active_inference_power.suite import run_power_suite

    with pytest.raises(ValueError):
        run_power_suite([], ("fixed_horizon",), n_reps=1)
    with pytest.raises(ValueError):
        run_power_suite([scenario], (), n_reps=1)

    path = tmp_path / "trace.csv"
    path.write_text("subject,time,action,observation\ns,1,observe,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_agent_trace(path, BenchmarkSpec("trace", "local", "MIT", "c" * 64))


def test_run_log_progress_optional_fields_are_validated() -> None:
    validate_run_log(
        {
            "schema_version": "2.0", "run_id": "run", "status": "running",
            "events": [], "config_sha256": "a" * 64,
            "progress": {
                "completed_stages": 0, "total_stages": 1,
                "completed_replications": 0, "estimated_remaining_seconds": None,
                "last_completed_stage": "initialized",
            },
        }
    )


def test_correction_result_rejects_nonfinite_and_out_of_range_adjusted_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        CorrectionResult(
            "bonferroni", np.array([np.nan]), np.array([False]), 0.05, 0.0
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CorrectionResult(
            "bonferroni", np.array([1.1]), np.array([False]), 0.05, 0.0
        )
