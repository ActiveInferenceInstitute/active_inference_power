"""Real-value edge coverage for the gated research contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from active_inference_power.learned_policy import (
    LearnedPolicyArtifact,
    LearnedPolicyConfig,
    evaluate_learned_policy,
    train_tabular_policy,
)
from active_inference_power.online_fdr import EValueLONDResult, OnlineFDRResult, e_lond
from active_inference_power.scenario import make_reference_action_scenario, make_static_binary_e_process_scenario
from active_inference_power.scenario import GenerativeModelSpec
from active_inference_power.action_loop import _choose_action, _single_step_pvalue, run_action_loop, simulate_action_loop
from active_inference_power.benchmarks import (
    BenchmarkSpec,
    load_agent_trace,
    load_csv_benchmark,
    load_normalized_golub_fixture,
    sha256_file,
)
from active_inference_power.dependence import DependenceConfig, sample_gaussian
from active_inference_power.formalism import formalism_registry
from active_inference_power.schema import (
    validate_checkpoint,
    validate_manifest,
    validate_results,
    validate_run_log,
    validate_suite_result,
)
from active_inference_power.selection import (
    FiltrationEvent,
    SelectionPolicySpec,
    SelectionTrace,
    _target_rates,
    run_split_stream_confirmation,
)
from active_inference_power.sequential import ConfidenceSequenceResult, bernoulli_confidence_sequence, mixture_e_process, predictable_e_process


def _event(*, hidden: bool = False) -> FiltrationEvent:
    return FiltrationEvent(0, (0, 1), (), ("observe",), None, hidden)


def test_selection_contract_validation_boundaries() -> None:
    invalid_policy_values: tuple[dict[str, Any], ...] = (
        {"name": ""},
        {"strategy": "unknown"},
        {"max_steps": 0},
        {"confirmation_steps": False},
        {"evidence_observable": 1},
    )
    for policy_values in invalid_policy_values:
        with pytest.raises(ValueError):
            SelectionPolicySpec(**policy_values)

    invalid_events: tuple[dict[str, Any], ...] = (
        {"time": -1},
        {"time": 0.5},
        {"target_candidates": (0, 0)},
        {"target_candidates": (-1, 0)},
        {"visible_observations": (2,)},
        {"visible_actions": ("",)},
        {"selected_target": 2},
        {"hidden_state_exposed": 1},
    )
    for event_values in invalid_events:
        event_arguments: dict[str, Any] = {
            "time": 0,
            "target_candidates": (0, 1),
            "visible_observations": (),
            "visible_actions": ("observe",),
            "selected_target": None,
            "hidden_state_exposed": False,
        }
        event_arguments.update(event_values)
        with pytest.raises(ValueError):
            FiltrationEvent(**event_arguments)

    policy = SelectionPolicySpec()
    event = _event()
    invalid_traces: tuple[dict[str, Any], ...] = (
        {"scenario_id": "", "scenario_hash": "a" * 64},
        {"scenario_id": "s", "scenario_hash": "short"},
        {"scenario_id": "s", "scenario_hash": "a" * 64, "seed": -1},
        {"scenario_id": "s", "scenario_hash": "a" * 64, "events": ()},
        {"scenario_id": "s", "scenario_hash": "a" * 64, "selected_target_evidence": 2.0},
        {"scenario_id": "s", "scenario_hash": "a" * 64, "visible_only": False},
        {"scenario_id": "s", "scenario_hash": "a" * 64, "stream_hash": "short"},
        {"scenario_id": "s", "scenario_hash": "a" * 64, "events": (_event(hidden=True),)},
    )
    for trace_values in invalid_traces:
        trace_arguments: dict[str, Any] = {
            "scenario_id": "s", "scenario_hash": "a" * 64, "policy": policy,
            "seed": 1, "events": (event,), "selected_target": 0,
            "selected_target_evidence": 0.5, "visible_only": True,
            "stream_hash": "b" * 64,
        }
        trace_arguments.update(trace_values)
        with pytest.raises(ValueError):
            SelectionTrace(**trace_arguments)


def test_split_result_and_strategy_edges() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=3)
    policy = SelectionPolicySpec(strategy="round_robin", max_steps=2, confirmation_steps=2)
    result = run_split_stream_confirmation(scenario, policy, selection_seed=3, confirmation_seed=4)
    assert result.to_mapping()["selection"]["visible_only"] is True
    assert result.selection.configuration_hash != result.selection.stream_hash

    with pytest.raises(ValueError):
        run_split_stream_confirmation(scenario, policy, selection_seed=True, confirmation_seed=4)
    with pytest.raises(ValueError):
        run_split_stream_confirmation(scenario, policy, selection_seed=3, confirmation_seed=3)
    with pytest.raises(ValueError):
        replace(result, scenario_id="other")
    with pytest.raises(ValueError):
        replace(result, selection_seed=4)
    with pytest.raises(ValueError):
        replace(result, selection_stream_hash="c" * 64)
    with pytest.raises(ValueError):
        replace(result, confirmation_observations=(2,))
    with pytest.raises(ValueError):
        replace(result, confirmation_p_value=2.0)
    with pytest.raises(ValueError):
        replace(result, validity="")
    with pytest.raises(ValueError):
        replace(result, confirmation_stream_hash="short")


def test_learned_policy_parallel_stability_and_validation() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=2)
    config = LearnedPolicyConfig(train_reps=1, candidate_parameters=((1.0, 0.5, 0.0),))
    artifact = train_tabular_policy(scenario, config, seed=11, workers=2)
    serial = evaluate_learned_policy(scenario, artifact, 1, seed=12, workers=1)
    parallel = evaluate_learned_policy(scenario, artifact, 1, seed=12, workers=2)
    assert serial["seed_schedule"] == parallel["seed_schedule"]
    for value in (
        {"train_reps": 0}, {"train_reps": True}, {"cost_penalty": -1.0},
        {"cost_penalty": np.inf}, {"candidate_parameters": ()},
        {"candidate_parameters": ((1.0, 2.0),)},
    ):
        with pytest.raises(ValueError):
            LearnedPolicyConfig(**value)
    with pytest.raises(ValueError):
        train_tabular_policy(scenario, config, seed=True)
    with pytest.raises(ValueError):
        train_tabular_policy(scenario, config, workers=0)
    with pytest.raises(ValueError):
        evaluate_learned_policy(scenario, artifact, 1, seed=11)
    with pytest.raises(ValueError):
        evaluate_learned_policy(scenario, artifact, 1, seed=12, workers=0)


def test_sequential_and_online_serialization_edges() -> None:
    assert "coverage_claim" in bernoulli_confidence_sequence((0, 1, 0)).to_mapping()
    with pytest.raises(ValueError):
        ConfidenceSequenceResult(np.array([0.1]), np.array([0.2, 0.3]), 0.05, (1, 1), np.array([0]), np.array([False]))
    with pytest.raises(ValueError):
        ConfidenceSequenceResult(np.array([0.1]), np.array([0.2]), 1.0, (1, 1), np.array([0]), np.array([False]))
    with pytest.raises(ValueError):
        ConfidenceSequenceResult(np.array([0.1]), np.array([0.2]), 0.05, (0, 1), np.array([0]), np.array([False]))
    with pytest.raises(ValueError):
        mixture_e_process((0, 1), 0.5, (0.5,))
    with pytest.raises(ValueError):
        mixture_e_process((0, 1), 0.5, (0.2, 0.8), weights=(0.0, 0.0))
    with pytest.raises(ValueError):
        predictable_e_process((0, 1), 0.5, (0.2,))
    with pytest.raises(ValueError):
        predictable_e_process((0, 1), 0.5, (0.5, 0.2))
    assert "num_rejections" in e_lond((0.0, 2.0), gamma=(0.6, 0.4)).to_mapping()
    with pytest.raises(ValueError):
        e_lond((-1.0,), gamma=(1.0,))
    with pytest.raises(ValueError):
        e_lond((1.0, 1.0), gamma=(0.5, 0.6))
    with pytest.raises(ValueError):
        e_lond((1.0, 1.0), gamma=(0.4, 0.5))
    with pytest.raises(ValueError):
        e_lond((1.0, 1.0), gamma=(1.0,))
    valid = e_lond((0.0, 2.0), gamma=(0.6, 0.4))
    with pytest.raises(ValueError):
        EValueLONDResult(np.array([]), np.array([]), np.array([]), np.array([]), np.array([0.0]), 0.05, np.array([]))
    with pytest.raises(ValueError):
        OnlineFDRResult("x", np.array([0.1]), np.array([0.1]), np.array([True]), np.array([False]), np.array([0.05]), 0.05, np.array([1.0]), initial_wealth=0.05)
    assert valid.num_rejections >= 0


def test_process_model_and_action_filtration_edges() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=3)
    with pytest.raises(ValueError):
        GenerativeModelSpec(
            scenario.model.state_names, scenario.model.observation_names,
            scenario.model.action_names, scenario.model.prior,
            scenario.model.likelihood[:, :, :-1], scenario.model.transition,
            scenario.model.state_truth, scenario.model.state_target,
            scenario.model.state_context, scenario.model.action_target,
            scenario.model.action_sensor, scenario.model.hypothesis_ids,
        )
    base_process = scenario.process
    invalid_processes = (
        {"missing_probability": 1.0},
        {"name": ""},
        {"missingness_mode": "unknown"},
        {"transition_schedule": np.zeros((1, 1, 2, 2))},
        {"likelihood_schedule": np.zeros((1, 1, 2, 2))},
        {"action_cost_schedule": np.zeros((1, 2))},
        {"action_cost_schedule": np.full((1, 1), -1.0)},
        {"missing_probability_schedule": np.array([1.0])},
        {"action_missing_probability": np.array([0.1, 0.2])},
        {"state_missing_probability": np.array([0.1])},
        {"missingness_mode": "mar_action"},
        {"missingness_mode": "mnar_state"},
    )
    for changes in invalid_processes:
        with pytest.raises(ValueError):
            replace(base_process, **changes)
    altered = replace(
        base_process, missingness_mode="mar_action",
        action_missing_probability=np.full_like(base_process.action_costs, 0.2),
    )
    assert altered.missing_probability_at(0, 0, 0) == 0.2
    state_probabilities = np.linspace(0.2, 0.3, base_process.initial_state.size)
    altered_state = replace(base_process, missingness_mode="mnar_state", state_missing_probability=state_probabilities)
    assert altered_state.missing_probability_at(0, 0, 1) == state_probabilities[1]

    with pytest.raises(ValueError):
        run_action_loop(scenario, "unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        run_action_loop(scenario, "fixed_horizon", max_steps=True)
    with pytest.raises(ValueError):
        run_action_loop(scenario, "fixed_horizon", max_steps=0)
    posterior = scenario.model.prior.copy()
    with pytest.raises(ValueError):
        _choose_action("posterior_sampling", posterior, scenario, 0)
    with pytest.raises(ValueError):
        _choose_action("learned", posterior, scenario, 0)
    with pytest.raises(ValueError):
        _choose_action("learned", posterior, scenario, 0, learned_parameters=np.array([1.0]))
    assert _single_step_pvalue(scenario, 0, None) == (1.0, 1.0)
    assert run_action_loop(scenario, "fixed_horizon", seed=2, use_pymdp=False, max_steps=1).stopping_time == 1
    budget_scenario = replace(scenario, setting=replace(scenario.setting, cost_budget=1e-12))
    assert run_action_loop(budget_scenario, "fixed_horizon", seed=2, use_pymdp=False).stopping_time == 0


def test_selection_strategies_and_target_contracts_are_explicit() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=3)
    minimum = run_split_stream_confirmation(
        scenario, SelectionPolicySpec(strategy="minimum_p", max_steps=2, confirmation_steps=2),
        selection_seed=21, confirmation_seed=22,
    )
    round_robin = run_split_stream_confirmation(
        scenario, SelectionPolicySpec(strategy="round_robin", max_steps=2, confirmation_steps=2),
        selection_seed=23, confirmation_seed=24,
    )
    least = run_split_stream_confirmation(
        scenario, SelectionPolicySpec(strategy="least_informed", max_steps=2, confirmation_steps=2),
        selection_seed=25, confirmation_seed=26,
    )
    assert minimum.selection.selected_target >= 0
    assert round_robin.selection.selected_target >= 0
    assert least.selection.selected_target >= 0
    assert _target_rates(scenario, 0, 0)[0] != _target_rates(scenario, 0, 0)[1]
    with pytest.raises(ValueError):
        _target_rates(scenario, 0, 99)
    with pytest.raises(ValueError):
        SelectionTrace(
            "s", "a" * 64, SelectionPolicySpec(), True, (_event(),), 0, 0.5, True, "b" * 64
        )

    # A target can be visible in the posterior while no action measures it;
    # the selector must fall back to a feasible target-selecting action.
    one_target_model = replace(scenario.model, action_target=np.zeros(scenario.model.n_actions, dtype=int))
    one_target = replace(scenario, model=one_target_model)
    assert _choose_action("adaptive_target", one_target.model.prior, one_target, 0) in range(one_target.model.n_actions)
    assert _choose_action(
        "posterior_sampling", one_target.model.prior, one_target, 0,
        rng=np.random.default_rng(27),
    ) in range(one_target.model.n_actions)


def test_action_evidence_contract_labels_dynamic_and_mismatched_processes() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=2)
    mismatched_likelihood = replace(scenario.process, likelihood=scenario.process.likelihood * 0.0 + scenario.model.likelihood)
    # The model/process differ in the reference scenario; a matching process is
    # the formal branch, while schedules and mismatches remain diagnostic.
    formal = replace(scenario, process=mismatched_likelihood)
    formal_trace = run_action_loop(formal, "e_process", seed=28, use_pymdp=False)
    assert formal_trace.validity_boundary == "conditional_per_target_optional_stopping"
    assert formal_trace.family_scope == "per_target_no_family_error_claim"
    dynamic = replace(
        formal.process,
        likelihood_schedule=np.stack([formal.process.likelihood, formal.process.likelihood]),
    )
    assert run_action_loop(replace(formal, process=dynamic), "e_process", seed=29, use_pymdp=False).validity_boundary == "diagnostic_action_adaptive_evidence"
    altered = replace(formal.process, transition=np.roll(formal.process.transition, 1, axis=1))
    assert run_action_loop(replace(formal, process=altered), "e_process", seed=30, use_pymdp=False).validity_boundary == "diagnostic_action_adaptive_evidence"
    early = replace(formal, setting=replace(formal.setting, posterior_threshold=0.0))
    assert run_action_loop(early, "posterior_threshold", seed=31, use_pymdp=False).stopping_time == 0


def test_benchmark_loaders_fail_closed_on_real_temp_files(tmp_path: Path) -> None:
    def spec(path: Path, benchmark_id: str = "fixture") -> BenchmarkSpec:
        return BenchmarkSpec(benchmark_id, "local", "CC-BY", sha256_file(path), claim_role="stress")

    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_csv_benchmark(empty, spec(empty))
    header_only = tmp_path / "header.csv"
    header_only.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_csv_benchmark(header_only, spec(header_only))
    nonfinite = tmp_path / "nonfinite.csv"
    nonfinite.write_text("a,b\n1,nan\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_csv_benchmark(nonfinite, spec(nonfinite))
    with pytest.raises(FileNotFoundError):
        load_csv_benchmark(tmp_path / "missing.csv", spec(nonfinite))
    with pytest.raises(ValueError):
        load_csv_benchmark(nonfinite, replace(spec(nonfinite), sha256="0" * 64))

    trace = tmp_path / "trace.csv"
    trace.write_text("subject,time,action,observation\ns1,0,a,0\ns1,1,b,1\n", encoding="utf-8")
    trace_spec = spec(trace, "trace")
    dataset = load_agent_trace(trace, trace_spec)
    assert cast(dict[str, Any], dataset.metadata)["chronological"] is True
    for payload in (
        "subject,time,action\ns1,0,a\n",
        "subject,time,action,observation\n",
        "subject,time,action,observation\ns1,missing,a,0\n",
        "subject,time,action,observation\ns1,x,a,0\n",
        "subject,time,action,observation\ns1,0,a,2\n",
        "subject,time,action,observation\ns1,1,a,0\ns1,1,b,1\n",
    ):
        bad = tmp_path / "bad_trace.csv"
        bad.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError):
            load_agent_trace(bad, spec(bad, "bad-trace"))

    golub = tmp_path / "golub.csv"
    golub.write_text("sample,feature,class,expression\n1,1,0,0.1\n2,1,1,0.2\n", encoding="utf-8")
    assert load_normalized_golub_fixture(golub, spec(golub, "golub")).values.shape == (2, 4)
    golub.write_text("sample,feature,class,expression\n1,1,2,0.1\n2,1,1,0.2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_normalized_golub_fixture(golub, spec(golub, "golub"))


def test_schema_nested_release_contracts_reject_stale_structures() -> None:
    operation = {
        "scenario_id": "s", "scenario_hash": "b" * 64, "policy": "fixed_horizon",
        "n_reps": 1, "summary": {}, "validity": "diagnostic",
    }
    base_suite: dict[str, Any] = {
        "suite_name": "suite", "scenarios": ["s"], "policies": ["fixed_horizon"],
        "operating_characteristics": [operation], "online_fdr": {}, "claim_boundary": "conditional",
    }
    with pytest.raises(ValueError):
        validate_suite_result({**base_suite, "contract_version": "2.0"})
    with pytest.raises(ValueError):
        validate_suite_result({**base_suite, "paired_policy_comparisons": {}})
    with pytest.raises(ValueError):
        validate_suite_result({**base_suite, "online_fdr": []})
    with pytest.raises(ValueError):
        validate_suite_result({**base_suite, "claim_boundary": ""})
    with pytest.raises(ValueError):
        validate_suite_result({**base_suite, "learned_policy": {"artifact": {}, "evaluation": {}}})
    with pytest.raises(ValueError):
        validate_suite_result({**base_suite, "learned_policy": {"artifact": {"policy_hash": "short"}, "evaluation": {"validity": "x"}}})
    frozen_artifact = LearnedPolicyArtifact(
        scenario_id="s",
        scenario_hash="a" * 64,
        parameters=(1.0, 0.5, 0.0),
        train_seed=1,
        train_reps=1,
        objective=0.25,
        search_config_hash="b" * 64,
        candidate_count=1,
        parameter_dimension=3,
    )
    valid_learned: dict[str, Any] = {
        "artifact": frozen_artifact.to_mapping(),
        "evaluation": {
            "policy": "learned",
            "policy_hash": frozen_artifact.policy_hash,
            "scenario_id": "s",
            "scenario_hash": "a" * 64,
            "train_seed": 1,
            "evaluation_seed": 2,
            "train_reps": 1,
            "evaluation_reps": 1,
            "search_config_hash": "b" * 64,
            "policy_complexity": {
                "search_type": "finite_tabular_candidate_search",
                "candidate_count": 1,
                "parameter_dimension": 3,
            },
            "summary": {},
            "seed_schedule": [3],
            "validity": "conditional_held_out_policy_operating_characteristic",
            "claim_boundary": "Frozen policy evaluation is conditional on the declared split.",
        },
    }
    validate_suite_result({**base_suite, "learned_policy": valid_learned})

    tampered_artifact = {
        **valid_learned["artifact"],
        "parameters": [0.0, 0.5, 0.0],
    }
    with pytest.raises(ValueError, match="policy_hash"):
        validate_suite_result(
            {**base_suite, "learned_policy": {**valid_learned, "artifact": tampered_artifact}}
        )

    confirmation: dict[str, Any] = {
        "scenario_id": "s", "scenario_hash": "c" * 64, "n_reps": 1,
        "selection_streams": 1, "confirmation_streams": 1,
        "rejection_rate": 0.0, "power": 0.0, "fwer": 0.0,
        "validity": "formal", "assumptions": ["independent"],
        "representative": {"selection_stream_hash": "d" * 64, "confirmation_stream_hash": "e" * 64},
    }
    bad_changes: tuple[dict[str, Any], ...] = (
        {"scenario_id": ""}, {"rejection_rate": True}, {"power": 2.0},
        {"selection_streams": 2}, {"validity": ""}, {"assumptions": []},
        {"representative": {"selection_stream_hash": "short", "confirmation_stream_hash": "e" * 64}},
    )
    for change in bad_changes:
        with pytest.raises(ValueError):
            validate_suite_result({**base_suite, "split_stream_confirmation": {**confirmation, **change}})
    validate_suite_result({**base_suite, "split_stream_confirmation": confirmation})

    result = {
        "schema_version": "2.0", "run_id": "run",
        "meta": {"config_sha256": "a" * 64, "claim_ledger_sha256": "b" * 64, "seed": 1, "n_reps": 1},
        **{key: {} for key in ("experiment", "certificates", "method_comparison", "paired_comparisons", "active_inference", "sequential", "online_fdr", "agent_sampling", "benchmarks", "null_pvalue_calibration", "sample_size_accounting")},
        "power_curve": [], "dependence_sweep": [], "formalism": formalism_registry().to_mapping(),
        "suite": {**base_suite, "contract_version": "1.0"},
        "action_loop": {"scenarios": [{"scenario_id": "s", "model": {}, "process": {}, "setting": {}}]},
    }
    validate_results(result, require_suite=True)
    with pytest.raises(ValueError, match="duplicate scenario identifiers"):
        validate_results(
            {
                **result,
                "action_loop": {
                    "scenarios": [
                        {"scenario_id": "s", "model": {}, "process": {}, "setting": {}},
                        {"scenario_id": "s", "model": {}, "process": {}, "setting": {}},
                    ],
                },
            },
            require_suite=True,
        )
    with pytest.raises(ValueError):
        validate_results({**result, "action_loop": {"contract_version": "2.0", "scenarios": []}}, require_suite=True)
    with pytest.raises(ValueError):
        validate_results({**result, "action_loop": {"scenarios": "bad"}}, require_suite=True)
    with pytest.raises(ValueError):
        validate_results({**result, "action_loop": {"scenarios": [{"scenario_id": "", "model": {}, "process": {}, "setting": {}}]}}, require_suite=True)


def test_schema_validates_scenario_and_frontier_lineage() -> None:
    scenario = make_reference_action_scenario(horizon=2)
    operation = simulate_action_loop(scenario, "fixed_horizon", 2, seed=8)
    contract = scenario.to_mapping()
    frontier = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "scenario_metadata": dict(scenario.metadata or {}),
        "policy": "fixed_horizon",
        "n_reps": 2,
        "power": operation["summary"]["power"],
        "power_interval": operation["summary"]["power_interval"],
        "fdr": operation["summary"]["fdr"],
        "fdr_interval": operation["summary"]["fdr_interval"],
        "mean_cost": operation["summary"]["mean_cost"],
        "cost_interval": operation["summary"]["cost_interval"],
        "mean_stopping_time": operation["summary"]["mean_stopping_time"],
        "power_per_cost": operation["summary"]["power_per_cost"],
        "validity": operation["summary"]["validity"],
    }
    suite = {
        "suite_name": "suite",
        "scenarios": [scenario.scenario_id],
        "scenario_contracts": [contract],
        "policies": ["fixed_horizon"],
        "operating_characteristics": [operation],
        "power_cost_frontier": [frontier],
        "online_fdr": {},
        "claim_boundary": "conditional",
    }
    validate_suite_result(suite)
    with pytest.raises(ValueError, match="does not match"):
        validate_suite_result({**suite, "scenario_contracts": [{**contract, "scenario_hash": "a" * 64}]})
    with pytest.raises(ValueError, match="no matching scenario contract"):
        validate_suite_result({**suite, "operating_characteristics": [{**operation, "scenario_hash": "b" * 64}]})

    with pytest.raises(ValueError, match="policies must contain"):
        validate_suite_result({**suite, "policies": [""]})
    with pytest.raises(ValueError, match="scenario_contracts must not"):
        validate_suite_result({**suite, "scenario_contracts": []})
    with pytest.raises(ValueError, match="duplicate lineage"):
        validate_suite_result({**suite, "scenario_contracts": [contract, contract]})
    validate_suite_result({**suite, "scenarios": [contract]})
    with pytest.raises(ValueError, match="scenario id or contract"):
        validate_suite_result({**suite, "scenarios": [1]})
    with pytest.raises(ValueError, match="scenario_id must be non-empty"):
        validate_suite_result({**suite, "scenario_contracts": [{**contract, "scenario_id": ""}]})
    with pytest.raises(ValueError, match="undeclared policy"):
        validate_suite_result({**suite, "operating_characteristics": [{**operation, "policy": "unknown"}]})

    def broken_frontier(**changes: object) -> dict[str, object]:
        return {**frontier, **changes}

    for changes, message in (
        ({"scenario_id": ""}, "scenario_id"),
        ({"policy": ""}, "policy"),
        ({"policy": "unknown"}, "undeclared policy"),
        ({"power": True}, "power must lie"),
        ({"power_interval": [0.0]}, "exactly two"),
        ({"power_interval": ["bad", 0.0]}, "finite numeric"),
        ({"cost_interval": [2.0, 1.0]}, "ordered"),
        ({"mean_cost": True}, "finite and non-negative"),
        ({"validity": ""}, "validity"),
        ({"scenario_hash": "c" * 64}, "no matching scenario"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_suite_result({**suite, "power_cost_frontier": [broken_frontier(**changes)]})

    lineage_artifact = LearnedPolicyArtifact(
        scenario_id=scenario.scenario_id,
        scenario_hash=scenario.configuration_hash,
        parameters=(1.0, 0.5, 0.0),
        train_seed=1,
        train_reps=1,
        objective=0.25,
        search_config_hash="b" * 64,
        candidate_count=1,
        parameter_dimension=3,
    )
    invalid_complexity = {
        "artifact": {
            **lineage_artifact.to_mapping(),
            "policy_complexity": {
                "search_type": "unknown",
                "candidate_count": 1,
                "parameter_dimension": 3,
            },
        },
        "evaluation": {},
    }
    with pytest.raises(ValueError, match="search_type"):
        validate_suite_result({**suite, "learned_policy": invalid_complexity})

def test_artifact_ledgers_and_checkpoint_cursors_reject_inconsistent_values() -> None:
    artifact = {"path": "x.json", "sha256": "a" * 64, "bytes": 1, "role": "results"}
    manifest = {"schema_version": "2.0", "run_id": "run", "complete": True, "config_sha256": "a" * 64, "artifacts": {"x": artifact}}
    validate_manifest(manifest)
    for change in (
        {"x": {**artifact, "path": "/absolute"}},
        {"x": {**artifact, "path": "../x"}},
        {"x": {**artifact, "bytes": True}},
        {"x": {**artifact, "sha256": "short"}},
    ):
        with pytest.raises(ValueError):
            validate_manifest({**manifest, "artifacts": change})
    failed = {"schema_version": "2.0", "run_id": "run", "status": "failed", "events": [], "config_sha256": "a" * 64, "finished_at_utc": "2026-07-22T00:00:00+00:00", "error": {"type": "RuntimeError", "message": "boom", "traceback": "trace"}}
    validate_run_log(failed)
    event = {"name": "stage", "status": "checkpoint", "at_utc": "2026-07-22T00:00:00+00:00", "elapsed_seconds": 0.0}
    validate_run_log({"schema_version": "2.0", "run_id": "run", "status": "running", "events": [event], "config_sha256": "a" * 64, "progress": {"completed_stages": 0, "total_stages": 1, "completed_replications": 0, "estimated_remaining_seconds": None, "last_completed_stage": "stage"}})
    for bad_event in (
        {**event, "status": "bad"},
        {**event, "at_utc": "local"},
        {**event, "elapsed_seconds": -1},
    ):
        with pytest.raises(ValueError):
            validate_run_log({"schema_version": "2.0", "run_id": "run", "status": "running", "events": [bad_event], "config_sha256": "a" * 64})

    dependencies = {key: ["c" * 64] for key in ("scenario_hashes", "model_hashes", "process_hashes", "setting_hashes")}
    checkpoint: dict[str, Any] = {
        "schema_version": "2.0", "run_id": "run", "config_sha256": "a" * 64, "claim_ledger_sha256": "b" * 64,
        "profile": "fast", "seed": 1, "stage_seed": 2,
        "seed_schedule": {"kind": "numpy.SeedSequence", "master_seed": 1, "stage": "stage", "stage_seed": 2, "replication_order": "stable"},
        "workers": 1, "stage": "stage", "completed_replications": 2, "chunk_size": 2,
        "replication_cursor": {"completed": 2, "next": 2},
        "partial_metric_accumulators": {"status": "partial", "completed_replications": 2, "metrics": {}},
        "dependency_hashes": dependencies, "value": {},
    }
    validate_checkpoint(checkpoint)
    with pytest.raises(ValueError):
        validate_checkpoint({**checkpoint, "replication_cursor": {"completed": 1, "next": 2}})
    with pytest.raises(ValueError):
        validate_checkpoint({**checkpoint, "seed_schedule": {**checkpoint["seed_schedule"], "stage": "other"}})
    with pytest.raises(ValueError):
        validate_checkpoint({**checkpoint, "partial_metric_accumulators": {"status": "bad", "completed_replications": 2}})
    with pytest.raises(ValueError):
        validate_checkpoint({**checkpoint, "partial_metric_accumulators": {"status": "partial", "completed_replications": 2, "metrics": []}})


def test_real_rng_and_uncertainty_single_replication_boundaries() -> None:
    from active_inference_power.uncertainty import mean_interval, paired_numeric_difference_summary

    assert mean_interval([3.0]) == (3.0, 3.0)
    assert paired_numeric_difference_summary([1.0], [2.0])["standard_error"] == 0.0
    with pytest.raises(ValueError):
        mean_interval([1.0, np.nan])
    with pytest.raises(ValueError):
        paired_numeric_difference_summary([1.0], [2.0, 3.0])
    draw = sample_gaussian(np.random.default_rng(32), 3, DependenceConfig(kind="independent"), mean=np.zeros(3))
    assert draw.shape == (3,)
