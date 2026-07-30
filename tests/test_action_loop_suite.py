"""Real-value tests for the conditional Active Inference power suite."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from active_inference_power import (
    GenerativeModelSpec,
    GenerativeProcessSpec,
    PowerScenario,
    SettingSpec,
    evaluate_adaptive_target_selection,
    evaluate_conditional_selection_calibration,
    evaluate_future_evidence_selection_failure,
    evaluate_power_surface,
    make_reference_action_scenario,
    make_reference_null_action_scenario,
    make_static_binary_e_process_scenario,
    run_action_loop,
    run_power_scenario,
    simulate_action_loop,
)
from active_inference_power.action_statistics import (
    paired_action_policy_comparisons,
    summarize_action_traces,
)
from active_inference_power.action_loop import _single_step_pvalue, summarize_action_loop
from active_inference_power.scenario import (
    make_static_binary_split_confirmation_scenario,
)
from active_inference_power.suite import PowerSuiteResult, reference_suite, run_power_suite
from active_inference_power.schema import validate_action_loop_result, validate_results, validate_suite_result
from active_inference_power.demo import build_demo_payload


def test_reference_scenario_separates_model_process_setting_and_hashes() -> None:
    scenario = make_reference_action_scenario(reliability=0.75, horizon=5)
    assert scenario.model.n_states == scenario.process.initial_state.size == 8
    assert scenario.model.n_actions == scenario.process.transition.shape[0]
    assert scenario.configuration_hash == scenario.configuration_hash
    assert scenario.metadata is not None
    assert scenario.metadata["reliability"] == 0.75
    assert "generative model" in scenario.description.lower()


def test_scenario_hash_and_power_summary_keep_policy_and_estimands_separate() -> None:
    """A scenario contract is reusable across policies; mean TPR is explicit."""
    scenario = make_reference_action_scenario(horizon=3)
    result = run_power_scenario(scenario, "fixed_horizon", n_reps=12, seed=19)

    assert "policy" not in PowerScenario.__dataclass_fields__
    assert result["scenario_hash"] == scenario.configuration_hash
    assert result["policy"] == "fixed_horizon"
    summary = result["summary"]
    assert "any_true_rejection_rate" in summary
    assert "target_true_rejection_rates" in summary
    traces = [run_action_loop(scenario, "fixed_horizon", seed=seed) for seed in range(4)]
    metric_summary = summarize_action_traces(scenario, traces)
    assert metric_summary["power_interval_method"] == "student_t_monte_carlo_mean_of_replication_tpr"
    assert "any_true_rejection_rate" in metric_summary


def test_action_trace_has_visible_and_hidden_evaluation_views() -> None:
    scenario = make_reference_action_scenario(horizon=4)
    trace = run_action_loop(scenario, "fixed_horizon", seed=7)
    assert trace.stopping_time == 4
    assert len(trace.steps) == len(trace.to_mapping()["visible_steps"])
    assert "evaluation_state_after" not in trace.steps[0].visible_mapping()
    assert trace.pymdp_oracle_max_abs_error < 1e-8
    assert np.all(np.isfinite([step.posterior_signal for step in trace.steps]))


def test_policies_have_distinct_stopping_and_validity_contracts() -> None:
    scenario = make_reference_action_scenario(horizon=8)
    fixed = run_action_loop(scenario, "fixed_horizon", seed=3)
    posterior = run_action_loop(scenario, "posterior_threshold", seed=3)
    e_process = run_action_loop(scenario, "e_process", seed=3)
    assert fixed.stopping_time == 8
    assert posterior.stopping_time <= 8
    assert not fixed.optional_stopping_valid
    assert not posterior.optional_stopping_valid
    assert not e_process.optional_stopping_valid


def test_static_binary_action_channel_receives_optional_stopping_label() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=8)
    e_process = run_action_loop(scenario, "e_process", seed=3)
    assert e_process.optional_stopping_valid
    assert e_process.validity_boundary == "conditional_per_target_optional_stopping"
    assert e_process.family_scope == "per_target_no_family_error_claim"
    assert set(scenario.setting.evidence_assumptions) == {
        "fixed_truth",
        "conditional_e_factor_mean_at_most_one",
        "nonnegative_e_process",
        "predictable_selection",
    }


def test_action_loop_schema_binds_optional_stopping_to_per_target_e_process_scope() -> None:
    """A label cannot upgrade a fixed-horizon or unscoped family decision."""
    scenario = make_static_binary_e_process_scenario(horizon=3)
    valid = simulate_action_loop(scenario, "e_process", 2, seed=24, retain_traces=1)
    validate_action_loop_result(valid)

    unqualified = json.loads(json.dumps(valid))
    unqualified["validity"] = "conditional optional-stopping claim"
    unqualified["summary"]["validity"] = "conditional optional-stopping claim"
    with pytest.raises(ValueError, match="unsupported unqualified"):
        validate_action_loop_result(unqualified)

    fixed = simulate_action_loop(scenario, "fixed_horizon", 2, seed=25, retain_traces=1)
    fixed["validity"] = "conditional per-target optional-stopping claim"
    fixed["summary"]["validity"] = "conditional per-target optional-stopping claim"
    fixed["summary"]["family_scope"] = "per_target_no_family_error_claim"
    fixed["summary"]["validity_assumptions"] = list(
        scenario.setting.evidence_assumptions
    )
    fixed["traces"][0]["optional_stopping_valid"] = True
    fixed["traces"][0]["validity_boundary"] = "conditional_per_target_optional_stopping"
    fixed["traces"][0]["family_scope"] = "per_target_no_family_error_claim"
    with pytest.raises(ValueError, match="declared and executed e_process"):
        validate_action_loop_result(fixed)


def test_action_e_process_requires_e_factor_not_pvalue_assumptions() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=4)
    pvalue_assumptions = replace(
        scenario.setting,
        evidence_assumptions=(
            "fixed_truth",
            "conditional_superuniformity",
            "predictable_selection",
        ),
    )
    trace = run_action_loop(
        replace(scenario, setting=pvalue_assumptions), "e_process", seed=4
    )
    assert not trace.optional_stopping_valid
    assert trace.validity_boundary == "diagnostic_action_adaptive_evidence"


def test_action_loop_serializes_e_process_evidence_with_canonical_names() -> None:
    trace = run_action_loop(make_reference_action_scenario(horizon=3), "e_process", seed=3)
    serialized = trace.to_mapping()
    visible_step = serialized["visible_steps"][0]

    assert serialized["evidence_kind"] == "likelihood_ratio_e_process"
    assert serialized["terminal_e_process_value"] == trace.terminal_e_process_value
    assert visible_step["evidence_kind"] == "likelihood_ratio_e_process"
    assert visible_step["e_process_value"] == trace.steps[0].e_process_value
    assert visible_step["target_e_process_values"] == list(
        trace.steps[0].target_e_process_values
    )


def test_action_setting_executes_or_records_fail_closed_overrides() -> None:
    reference = make_reference_action_scenario(horizon=3)
    declared = replace(
        reference,
        setting=replace(
            reference.setting,
            stopping_policy="posterior_threshold",
            posterior_threshold=0.0,
        ),
    )
    assert declared.configuration_hash != reference.configuration_hash

    trace = run_action_loop(declared, seed=32, use_pymdp=False)
    assert trace.policy == trace.executed_stopping_policy == "posterior_threshold"
    assert trace.stopping_time == 0
    assert not trace.policy_override
    assert trace.declared_correction == trace.executed_correction == "none"
    serialized = trace.to_mapping()
    assert serialized["declared_stopping_policy"] == serialized["executed_stopping_policy"]
    assert serialized["declared_correction"] == serialized["executed_correction"] == "none"

    override = run_action_loop(declared, "fixed_horizon", seed=32, use_pymdp=False)
    assert override.stopping_time == 3
    assert override.policy_override
    assert override.executed_stopping_policy == "fixed_horizon"
    assert not override.optional_stopping_valid
    assert override.validity_boundary == "diagnostic_action_adaptive_evidence"

    simulation = simulate_action_loop(declared, None, 2, seed=33, retain_traces=1)
    assert simulation["policy"] == "posterior_threshold"
    assert simulation["summary"]["executed_correction"] == "none"
    assert simulation["summary"]["executed_stopping_policy"] == "posterior_threshold"
    validate_action_loop_result(simulation)
    with pytest.raises(ValueError, match="executed_correction"):
        validate_action_loop_result(
            {
                **simulation,
                "summary": {
                    **simulation["summary"],
                    "executed_correction": "benjamini_hochberg",
                },
            }
        )
    with pytest.raises(ValueError, match="action-loop correction"):
        SettingSpec(correction="benjamini_hochberg")


def test_action_loop_is_seeded_and_worker_invariant() -> None:
    scenario = make_reference_action_scenario(horizon=4)
    left = simulate_action_loop(scenario, "cost_aware", 5, seed=21, workers=1, retain_traces=5)
    right = simulate_action_loop(scenario, "cost_aware", 5, seed=21, workers=4, retain_traces=5)
    assert left == right
    assert run_power_scenario(scenario, "fixed_horizon", n_reps=2, seed=1)["n_reps"] == 2


def test_surface_and_online_target_selection_are_composable() -> None:
    scenarios = [make_reference_action_scenario(reliability=value, horizon=3) for value in (0.65, 0.85)]
    assert len({scenario.scenario_id for scenario in scenarios}) == 2
    surface = evaluate_power_surface(scenarios, ("fixed_horizon", "cost_aware"), 3, seed=5)
    assert len(surface) == 4
    assert {item["scenario_metadata"]["reliability"] for item in surface} == {0.65, 0.85}
    online = evaluate_adaptive_target_selection(scenarios[0], "adaptive_target", 4, seed=6, method="saffron")
    assert online["method"] == "saffron"
    assert 0.0 <= online["any_rejection_rate"] <= 1.0


def test_all_null_selected_target_calibration_is_explicitly_diagnostic() -> None:
    scenario = make_reference_null_action_scenario(horizon=3)
    result = evaluate_conditional_selection_calibration(
        scenario, "adaptive_target", 6, seed=17, workers=2
    )
    assert result["validity"] == "calibration_diagnostic"
    assert result["pooled"]["n_selected"] > 0
    assert result["pooled"]["dkw_radius"] > 0.0
    assert all(not value for value in scenario.process.target_truth)
    with pytest.raises(ValueError):
        evaluate_conditional_selection_calibration(
            make_reference_action_scenario(horizon=3), "adaptive_target", 2
        )
    with pytest.raises(ValueError):
        evaluate_conditional_selection_calibration(scenario, "adaptive_target", 0)
    empty = replace(scenario, setting=replace(scenario.setting, posterior_threshold=0.0))
    empty_result = evaluate_conditional_selection_calibration(empty, "posterior_threshold", 2)
    assert empty_result["pooled"]["n_selected"] == 0
    assert empty_result["pooled"]["dkw_radius"] is None


def test_future_evidence_selection_failure_is_explicitly_invalid() -> None:
    result = evaluate_future_evidence_selection_failure(
        make_reference_null_action_scenario(horizon=4), 6, seed=18, workers=2
    )
    assert result["validity"] == "selection_failure_contrast"
    assert result["selection_definition"].startswith("post-hoc")
    assert result["pooled"]["n_selected"] == 6
    with pytest.raises(ValueError):
        evaluate_future_evidence_selection_failure(
            make_reference_action_scenario(horizon=4), 2
        )


def test_schema_validates_suite_and_rejects_missing_hash() -> None:
    scenario = make_reference_action_scenario(horizon=2)
    result = simulate_action_loop(scenario, "fixed_horizon", 2, seed=4)
    validate_action_loop_result(result)
    suite = {
        "suite_name": "test",
        "scenarios": [scenario.scenario_id],
        "policies": ["fixed_horizon"],
        "operating_characteristics": [result],
        "online_fdr": {"method": "lord_plus_plus"},
        "claim_boundary": "conditional",
    }
    validate_suite_result(suite)
    with pytest.raises(ValueError):
        validate_action_loop_result({key: value for key, value in result.items() if key != "scenario_hash"})


def test_suite_schema_rejects_duplicate_scenario_identifiers() -> None:
    scenario = make_reference_action_scenario(horizon=2)
    result = simulate_action_loop(scenario, "fixed_horizon", 1, seed=4)
    suite = {
        "suite_name": "test",
        "scenarios": ["reference_a", "reference_a"],
        "scenario_contracts": [scenario.to_mapping()],
        "policies": ["fixed_horizon"],
        "operating_characteristics": [result],
        "online_fdr": {},
        "claim_boundary": "conditional",
    }
    with pytest.raises(ValueError, match="duplicate scenario identifiers"):
        validate_suite_result(suite)


def test_contract_validation_rejects_invalid_model_process_and_setting() -> None:
    with pytest.raises(ValueError):
        GenerativeModelSpec(
            ("s",), ("o",), ("a",), np.array([1.0]), np.array([[[0.5]]]),
            np.array([[[1.0]]]), np.array([0]), np.array([0]), np.array([0]),
            np.array([0]), np.array([0]), ("h",),
        )
    with pytest.raises(ValueError):
        GenerativeProcessSpec(np.array([1.0]), np.array([[[1.0]]]), np.array([[[1.0]]]), np.array([]), np.array([1.0]))
    with pytest.raises(ValueError):
        SettingSpec(alpha=1.0)
    with pytest.raises(ValueError):
        run_action_loop(make_reference_action_scenario(horizon=2), "unknown")  # type: ignore[arg-type]


def test_trace_json_is_serializable() -> None:
    scenario = make_reference_action_scenario(horizon=2)
    result = simulate_action_loop(scenario, "fixed_horizon", 1, seed=9)
    encoded = json.dumps(result, sort_keys=True)
    assert "scenario_hash" in encoded


def test_demo_payload_is_deterministic_and_downloadable() -> None:
    left = build_demo_payload(reliability=0.72, horizon=3, policy="cost_aware", seed=12)
    right = build_demo_payload(reliability=0.72, horizon=3, policy="cost_aware", seed=12)
    assert left == right
    assert left["trace"]["visible_steps"]
    assert "claim_boundary" in left
    configured = build_demo_payload(
        reliability=0.72,
        horizon=3,
        policy="cost_aware",
        seed=12,
        cost_budget=4.0,
        posterior_threshold=0.9,
    )
    assert configured["configuration_hash"] != left["configuration_hash"]
    assert configured["diagnostic_evaluation"]["hidden_states"]


def test_action_loop_missing_cost_budget_and_argument_boundaries() -> None:
    scenario = make_reference_action_scenario(horizon=3)
    missing = replace(scenario.process, missing_probability=0.999999)
    missing_scenario = replace(scenario, process=missing)
    trace = run_action_loop(missing_scenario, "fixed_horizon", seed=1)
    assert all(step.observation is None for step in trace.steps)
    budget_scenario = replace(scenario, setting=replace(scenario.setting, cost_budget=0.5))
    assert run_action_loop(budget_scenario, "cost_aware", seed=1).stopping_time <= 1
    with pytest.raises(ValueError):
        run_action_loop(scenario, "fixed_horizon", seed=-1)
    with pytest.raises(ValueError):
        simulate_action_loop(scenario, "fixed_horizon", 0)
    with pytest.raises(ValueError):
        simulate_action_loop(scenario, "fixed_horizon", 1, workers=0)
    with pytest.raises(ValueError):
        simulate_action_loop(scenario, "fixed_horizon", 1, retain_traces=-1)
    with pytest.raises(ValueError):
        summarize_action_loop(scenario, "fixed_horizon", [])
    assert _single_step_pvalue(scenario, 5, None) == (1.0, 1.0)


def test_action_statistics_report_target_rates_and_cost_efficiency() -> None:
    scenario = make_reference_action_scenario(horizon=5)
    result = simulate_action_loop(scenario, "information_gain", 8, seed=13, retain_traces=8)
    summary = result["summary"]
    assert summary["mean_rejections"] >= summary["mean_true_rejections"]
    assert len(summary["target_rejection_rates"]) == scenario.setting.family_size
    assert len(summary["target_true_rejection_rates"]) == scenario.setting.family_size
    assert summary["power_per_cost"] >= 0.0
    assert len(summary["cost_interval"]) == 2


def test_posterior_sampling_and_paired_policy_comparison_are_seeded() -> None:
    scenario = make_reference_action_scenario(horizon=4)
    left = simulate_action_loop(scenario, "posterior_sampling", 4, seed=14, retain_traces=4)
    right = simulate_action_loop(scenario, "posterior_sampling", 4, seed=14, retain_traces=4)
    assert left == right
    comparison = paired_action_policy_comparisons(
        scenario,
        ("fixed_horizon", "information_gain", "posterior_sampling"),
        5,
        seed=15,
        run_trace=run_action_loop,
    )
    assert comparison["reference_policy"] == "fixed_horizon"
    assert set(comparison["contrasts"]) == {"information_gain", "posterior_sampling"}
    assert comparison["adjusted_confidence"] > comparison["confidence"]


def test_action_e_process_requires_declared_evidence_assumptions() -> None:
    scenario = make_reference_action_scenario(horizon=3)
    setting = replace(scenario.setting, evidence_assumptions=())
    trace = run_action_loop(replace(scenario, setting=setting), "e_process", seed=16)
    assert not trace.optional_stopping_valid
    assert trace.validity_boundary == "diagnostic_action_adaptive_evidence"


def test_action_e_process_does_not_claim_validity_under_context_dependent_rates() -> None:
    scenario = make_reference_action_scenario(horizon=3)
    trace = run_action_loop(scenario, "e_process", seed=16)
    assert not trace.optional_stopping_valid
    assert trace.validity_boundary == "diagnostic_action_adaptive_evidence"


def test_action_loop_worker_invariance_holds_for_policy_randomness() -> None:
    scenario = make_reference_action_scenario(horizon=4)
    serial = simulate_action_loop(scenario, "posterior_sampling", 6, seed=17, workers=1, retain_traces=6)
    parallel = simulate_action_loop(scenario, "posterior_sampling", 6, seed=17, workers=4, retain_traces=6)
    assert serial == parallel


def test_suite_wrappers_and_invalid_online_method() -> None:
    scenario = make_reference_action_scenario(horizon=2)
    suite = run_power_suite([scenario], ("fixed_horizon",), n_reps=2, seed=2)
    assert isinstance(suite, PowerSuiteResult)
    suite_mapping = suite.to_mapping()
    assert suite_mapping["suite_name"] == "active_inference_power_suite"
    assert suite_mapping["contract_version"] == "1.0"
    assert suite_mapping["scenario_contracts"] == [scenario.to_mapping()]
    split_contract = cast(dict[str, Any], suite_mapping["split_confirmation_scenario"])
    split_summary = cast(dict[str, Any], suite_mapping["split_stream_confirmation"])
    assert split_contract["scenario_id"] not in suite_mapping["scenarios"]
    assert split_summary["scenario_id"] == split_contract["scenario_id"]
    assert split_summary["scenario_hash"] == split_contract["scenario_hash"]
    assert split_contract["setting"]["formal_validity"] == (
        "conditional_on_declared_split_confirmation_assumptions"
    )
    assert split_summary["validity"] == "formal_conditional_split_confirmation"
    validate_suite_result(suite_mapping, require_current_contract=True)

    # Public suite construction is current-contract only: an incomplete
    # positional payload cannot be serialized without scenario lineage.
    positional_constructor = cast(Any, PowerSuiteResult)
    with pytest.raises(TypeError):
        positional_constructor(
            "incomplete",
            (scenario.scenario_id,),
            ("fixed_horizon",),
            (),
            {},
            "unbound suite",
        )
    with pytest.raises(ValueError, match="requires operating scenario contracts"):
        PowerSuiteResult(
            "unbound",
            (scenario.scenario_id,),
            ("fixed_horizon",),
            (),
            {},
            "unbound suite",
            {},
            (),
            {},
        ).to_mapping()
    assert scenario.metadata is not None
    frontier = cast(list[dict[str, Any]], suite_mapping["power_cost_frontier"])
    assert frontier[0]["scenario_metadata"]["reliability"] == scenario.metadata["reliability"]
    assert reference_suite(n_reps=2, seed=2).scenarios == ("reference_action_loop",)

    # If callers put the canonical split channel in the operating grid, the
    # public suite still gives its confirmation lane distinct lineage rather
    # than silently reusing that operating cell.
    split_operating = make_static_binary_split_confirmation_scenario(horizon=2)
    collision_suite = run_power_suite(
        [split_operating], ("fixed_horizon",), n_reps=1, seed=3
    ).to_mapping()
    collision_contract = cast(
        dict[str, Any], collision_suite["split_confirmation_scenario"]
    )
    assert collision_contract["scenario_id"] != split_operating.scenario_id
    assert collision_contract["metadata"]["separate_from_operating_grid"] is True
    with pytest.raises(ValueError):
        evaluate_adaptive_target_selection(scenario, "adaptive_target", 1, method="invalid")
    with pytest.raises(ValueError):
        evaluate_power_surface([], ("fixed_horizon",), 1)
    with pytest.raises(ValueError):
        evaluate_power_surface([scenario], (), 1)
    with pytest.raises(ValueError, match="unique scenario"):
        evaluate_power_surface([scenario, scenario], ("fixed_horizon",), 1)
    with pytest.raises(ValueError, match="unique names"):
        evaluate_power_surface([scenario], ("fixed_horizon", "fixed_horizon"), 1)


def test_suite_serialization_rejects_each_broken_lineage_component() -> None:
    """A real suite must fail closed when callers corrupt any lineage channel."""
    scenario = make_reference_action_scenario(horizon=2)
    suite = run_power_suite([scenario], ("fixed_horizon",), n_reps=1, seed=29)
    split_contract = dict(suite.split_confirmation_scenario)
    split_summary = dict(suite.split_stream_confirmation)

    invalid_contracts = cast(
        PowerSuiteResult, replace(suite, scenario_contracts=cast(Any, ("not a mapping",)))
    )
    with pytest.raises(ValueError, match="scenario_contracts must contain mappings"):
        invalid_contracts.to_mapping()

    mismatched_order = replace(suite, scenarios=("wrong_operating_scenario",))
    with pytest.raises(ValueError, match="operating scenario order"):
        mismatched_order.to_mapping()

    invalid_split_contract = cast(
        PowerSuiteResult,
        replace(suite, split_confirmation_scenario=cast(Any, ["not a mapping"])),
    )
    with pytest.raises(ValueError, match="split_confirmation_scenario must be a mapping"):
        invalid_split_contract.to_mapping()

    colliding_split_contract = replace(
        suite, split_confirmation_scenario=split_contract | {"scenario_id": scenario.scenario_id}
    )
    with pytest.raises(ValueError, match="outside the operating grid"):
        colliding_split_contract.to_mapping()

    invalid_split_summary = cast(
        PowerSuiteResult,
        replace(suite, split_stream_confirmation=cast(Any, ["not a mapping"])),
    )
    with pytest.raises(ValueError, match="split_stream_confirmation must be a mapping"):
        invalid_split_summary.to_mapping()

    mismatched_split_summary = replace(
        suite, split_stream_confirmation=split_summary | {"scenario_hash": "wrong-hash"}
    )
    with pytest.raises(ValueError, match="match its declared scenario contract"):
        mismatched_split_summary.to_mapping()


def test_public_suite_honors_replication_budget_above_former_cap() -> None:
    scenario = make_reference_action_scenario(horizon=3)
    suite = run_power_suite(
        [scenario], ("fixed_horizon",), n_reps=257, seed=13, workers=2
    )
    assert suite.operating_characteristics[0]["n_reps"] == 257
    assert suite.paired_policy_comparisons[0]["n_reps"] == 257
    assert suite.online_fdr["n_reps"] == 257
    assert suite.split_stream_confirmation is not None
    assert suite.split_stream_confirmation["n_reps"] == 257


def test_scenario_mapping_carries_recomputable_lineage_hash() -> None:
    scenario = make_reference_action_scenario(horizon=2)
    mapping = scenario.to_mapping()
    assert mapping["scenario_hash"] == scenario.configuration_hash
    changed = replace(scenario, setting=replace(scenario.setting, alpha=0.01)).to_mapping()
    assert changed["scenario_hash"] != mapping["scenario_hash"]


def _minimal_contract() -> tuple[GenerativeModelSpec, GenerativeProcessSpec, SettingSpec]:
    model = GenerativeModelSpec(
        ("s",), ("o",), ("a",), np.array([1.0]), np.array([[[1.0]]]),
        np.array([[[1.0]]]), np.array([0]), np.array([0]), np.array([0]),
        np.array([0]), np.array([0]), ("h",),
    )
    process = GenerativeProcessSpec(
        np.array([1.0]), np.array([[[1.0]]]), np.array([[[1.0]]]),
        np.array([True]), np.array([1.0]), name="minimal",
    )
    return model, process, SettingSpec(family_size=1)


def test_model_validation_covers_shape_metadata_and_label_boundaries() -> None:
    model, _, _ = _minimal_contract()
    common: dict[str, Any] = dict(
        prior=np.array([1.0]), likelihood=np.array([[[1.0]]]), transition=np.array([[[1.0]]]),
        state_truth=np.array([0]), state_target=np.array([0]), state_context=np.array([0]),
        action_target=np.array([0]), action_sensor=np.array([0]), hypothesis_ids=("h",),
    )
    for field, invalid_name in (("state_names", ()), ("observation_names", ()), ("action_names", ()), ("hypothesis_ids", ("h", "h"))):
        values_by_name: dict[str, Any] = {"state_names": ("s",), "observation_names": ("o",), "action_names": ("a",), **common}
        values_by_name[field] = invalid_name
        with pytest.raises(ValueError):
            GenerativeModelSpec(**values_by_name)
    for field, invalid_shape in (("prior", np.array([1.0, 0.0])), ("likelihood", np.ones((1, 1))), ("transition", np.ones((1, 1))), ("state_truth", np.array([0, 0])), ("action_target", np.array([0, 0])), ("preferences", np.array([0.0, 0.0]))):
        values_by_shape: dict[str, Any] = {"state_names": ("s",), "observation_names": ("o",), "action_names": ("a",), **common}
        values_by_shape[field] = invalid_shape
        with pytest.raises(ValueError):
            GenerativeModelSpec(**values_by_shape)
    for field, invalid_index in (("state_target", np.array([1])), ("action_target", np.array([-2])), ("state_truth", np.array([2])), ("state_context", np.array([-1]))):
        values_by_index: dict[str, Any] = {"state_names": ("s",), "observation_names": ("o",), "action_names": ("a",), **common}
        values_by_index[field] = invalid_index
        with pytest.raises(ValueError):
            GenerativeModelSpec(**values_by_index)
    assert model.preferences is not None
    assert model.preferences.shape == (1,)


def test_process_setting_and_power_scenario_validation_boundaries() -> None:
    model, process, setting = _minimal_contract()
    process_common: dict[str, Any] = dict(
        initial_state=np.array([1.0]), transition=np.array([[[1.0]]]), likelihood=np.array([[[1.0]]]),
        target_truth=np.array([True]), action_costs=np.array([1.0]),
    )
    for field, invalid_process_shape in (("initial_state", np.array([0.5])), ("transition", np.array([[[0.5]]])), ("likelihood", np.array([[[0.5]]])), ("action_costs", np.array([1.0, 2.0]))):
        process_values: dict[str, Any] = dict(process_common)
        process_values[field] = invalid_process_shape
        with pytest.raises(ValueError):
            GenerativeProcessSpec(**process_values)
    for field, invalid_process_contract in (("target_truth", np.array([])), ("missing_probability", 1.0), ("name", ""), ("action_costs", np.array([-1.0]))):
        process_boundary_values: dict[str, Any] = dict(process_common)
        process_boundary_values[field] = invalid_process_contract
        with pytest.raises(ValueError):
            GenerativeProcessSpec(**process_boundary_values)
    for kwargs in (
        {"confidence": 1.0}, {"posterior_threshold": 2.0}, {"max_steps": 0},
        {"cost_budget": -1.0}, {"correction": "unknown"}, {"stopping_policy": "unknown"},
        {"family_size": 0}, {"formal_validity": ""},
    ):
        with pytest.raises(ValueError):
            SettingSpec(**kwargs)
    with pytest.raises(ValueError):
        PowerScenario("", model, process, setting)
    with pytest.raises(ValueError):
        PowerScenario("bad", model, process, replace(setting, family_size=2), "bad")
    assert PowerScenario("minimal", model, process, setting, "minimal").configuration_hash


def test_schema_action_loop_and_suite_failure_boundaries() -> None:
    scenario = make_reference_action_scenario(horizon=2)
    result = simulate_action_loop(scenario, "fixed_horizon", 1, seed=1)
    suite = {
        "suite_name": "suite", "scenarios": [scenario.scenario_id], "policies": ["fixed_horizon"],
        "operating_characteristics": [result], "online_fdr": {}, "claim_boundary": "conditional",
    }
    with pytest.raises(ValueError):
        validate_action_loop_result(dict(result, scenario_id=""))
    with pytest.raises(ValueError):
        validate_action_loop_result(dict(result, policy=""))
    with pytest.raises(ValueError):
        validate_action_loop_result(dict(result, n_reps=0))
    with pytest.raises(ValueError):
        validate_action_loop_result(dict(result, validity=""))
    with pytest.raises(ValueError):
        validate_action_loop_result(dict(result, traces="bad"))
    for key in ("power", "fdr", "fwer", "mean_cost", "mean_stopping_time", "mean_rejections", "power_per_cost"):
        with pytest.raises(ValueError):
            validate_action_loop_result(dict(result, summary={**result["summary"], **{key: "bad"}}))
    with pytest.raises(ValueError):
        validate_action_loop_result(dict(result, summary={**result["summary"], "power": 1.1}))
    with pytest.raises(ValueError):
        validate_action_loop_result(dict(result, traces=[{"validity_assumptions": "bad"}]))
    with pytest.raises(ValueError):
        validate_suite_result(dict(suite, suite_name=""))
    with pytest.raises(ValueError):
        validate_suite_result(dict(suite, scenarios=[]))
    with pytest.raises(ValueError):
        validate_suite_result(dict(suite, online_fdr=[]))
    with pytest.raises(ValueError):
        validate_suite_result(dict(suite, claim_boundary=""))
    with pytest.raises(ValueError):
        validate_suite_result(dict(suite, paired_policy_comparisons="bad"))
    with pytest.raises(ValueError):
        validate_suite_result(dict(suite, power_cost_frontier="bad"))
    with pytest.raises(ValueError):
        validate_results({"schema_version": "2.0"})
