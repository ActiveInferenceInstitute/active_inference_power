"""Adversarial checks for the newly explicit statistical and provenance contracts."""

from __future__ import annotations

from dataclasses import replace
import copy
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from active_inference_power.benchmarks import (
    BenchmarkDataset,
    BenchmarkSpec,
    chronological_subject_split,
    load_agent_trace,
    sha256_file,
)
from active_inference_power.online_fdr import (
    _expected_online_levels,
    _validate_gamma,
    e_lond,
    lord_plus_plus,
    saffron,
)
from active_inference_power.scenario import make_static_binary_e_process_scenario
from active_inference_power.schema import (
    validate_action_loop_result,
    validate_benchmark_manifest,
    validate_checkpoint,
    validate_figure_registry,
    validate_suite_result,
)
from active_inference_power.selection import (
    SelectionPolicySpec,
    _choose_target,
    _confirmation_stream,
    _selection_events,
    run_split_stream_confirmation,
    simulate_split_stream_confirmation,
)
from active_inference_power.sequential import (
    bernoulli_confidence_sequence,
    binary_e_process,
    mixture_e_process,
)
from active_inference_power.action_loop import (
    _e_process_contract_valid,
    _single_step_pvalue,
    run_action_loop,
    simulate_action_loop,
)


def _local_spec(path: Path, benchmark_id: str = "trace") -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id,
        "local fixture",
        "CC-BY",
        sha256_file(path),
        claim_role="external_stress_test_only",
        retrieval_date="2026-07-25",
    )


def test_chronological_subject_split_is_subject_disjoint_and_missingness_explicit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.csv"
    path.write_text(
        "subject,time,action,observation\n"
        "early,0,inspect,0\n"
        "early,1,inspect,missing\n"
        "middle,2,measure,1\n"
        "middle,3,measure,0\n"
        "late,4,inspect,1\n",
        encoding="utf-8",
    )
    dataset = load_agent_trace(path, _local_spec(path), allow_missing_observations=True)
    assert dataset.metadata is not None
    assert dataset.metadata["missing_observation_count"] == 1
    assert dataset.values[1, 1] == -1.0
    split = chronological_subject_split(dataset, train_fraction=0.5)
    assert set(split.train_subjects).isdisjoint(split.evaluation_subjects)
    assert split.to_mapping()["chronological"] is True
    assert split.train.values.shape[0] + split.evaluation.values.shape[0] == dataset.values.shape[0]
    assert split.split_hash == chronological_subject_split(dataset, train_fraction=0.5).split_hash
    with pytest.raises(ValueError, match="train_fraction"):
        chronological_subject_split(dataset, train_fraction=1.0)
    with pytest.raises(ValueError, match="split_hash"):
        replace(split, split_hash="short")
    with pytest.raises(ValueError, match="metadata"):
        chronological_subject_split(replace(dataset, metadata={"subjects": []}), train_fraction=0.5)


def test_benchmark_metadata_and_dataset_contracts_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        BenchmarkSpec("x", "local", "MIT", "short")
    with pytest.raises(ValueError, match="ISO"):
        BenchmarkSpec("x", "local", "MIT", "a" * 64, retrieval_date="yesterday")
    spec = _local_spec(path)
    with pytest.raises(ValueError, match="non-empty"):
        BenchmarkDataset("x", np.empty((0, 2)), ("a", "b"), spec)
    with pytest.raises(ValueError, match="finite"):
        BenchmarkDataset("x", np.array([[np.nan, 1.0]]), ("a", "b"), spec)


def test_selection_filtration_visibility_and_alpha_are_enforced() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=3)
    with pytest.raises(ValueError, match="minimum_p"):
        SelectionPolicySpec(strategy="minimum_p", evidence_observable=False)
    policy = SelectionPolicySpec(strategy="round_robin", evidence_observable=False, max_steps=2, confirmation_steps=2)
    result = run_split_stream_confirmation(scenario, policy, selection_seed=41, confirmation_seed=42)
    assert all(not event.visible_evidence for event in result.selection.events)
    with pytest.raises(ValueError, match="alpha"):
        run_split_stream_confirmation(scenario, policy, selection_seed=41, confirmation_seed=42, alpha=1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(
            result.selection,
            events=(result.selection.events[0], result.selection.events[0]),
        )


def test_online_fdr_serialized_traces_reject_tampering() -> None:
    lord = lord_plus_plus([0.01, 0.9, 0.001], alpha=0.1, gamma=[0.5, 0.3, 0.2], w0=0.05)
    with pytest.raises(ValueError, match="align"):
        replace(lord, levels=lord.levels[:-1])
    with pytest.raises(ValueError, match="one initial"):
        replace(lord, wealth=lord.wealth[:-1])
    with pytest.raises(ValueError, match="outside"):
        replace(lord, pvalues=np.array([np.nan, 0.9, 0.001]))
    with pytest.raises(ValueError, match="exceed alpha"):
        replace(lord, levels=np.full(3, 0.2))
    with pytest.raises(ValueError, match="initial_wealth"):
        replace(lord, initial_wealth=0.2)
    with pytest.raises(ValueError, match="lambda"):
        replace(lord, lambda_=0.2)
    with pytest.raises(ValueError, match="method"):
        replace(lord, method="unknown")
    with pytest.raises(ValueError, match="candidate"):
        replace(lord, candidates=np.array([True, False, True]))
    with pytest.raises(ValueError, match="levels"):
        replace(lord, levels=lord.levels + np.array([0.001, 0.0, 0.0]))
    with pytest.raises(ValueError, match="wealth"):
        replace(lord, wealth=lord.wealth + np.array([0.0, 0.01, 0.0, 0.0]))

    saff = saffron([0.01, 0.9, 0.001], alpha=0.1, gamma=[0.5, 0.3, 0.2], lambda_=0.5, w0=0.05)
    with pytest.raises(ValueError, match="lambda"):
        replace(saff, lambda_=None)
    with pytest.raises(ValueError, match="candidates"):
        replace(saff, candidates=np.logical_not(saff.candidates))
    with pytest.raises(ValueError, match="rejections"):
        replace(saff, rejections=np.logical_not(saff.rejections))

    lond = e_lond([100.0, 1.0, 100.0], alpha=0.1, gamma=[0.5, 0.3, 0.2])
    with pytest.raises(ValueError, match="thresholds"):
        replace(lond, thresholds=lond.thresholds + 1.0)
    with pytest.raises(ValueError, match="cumulative allocations"):
        replace(
            lond,
            cumulative_allocations=lond.cumulative_allocations
            + np.array([0.0, 0.01, 0.0, 0.0]),
        )
    with pytest.raises(ValueError, match="rejections"):
        replace(lond, rejections=np.logical_not(lond.rejections))
    with pytest.raises(ValueError, match="declared e-LOND recursion"):
        replace(
            lond,
            levels=np.full(3, 0.01),
            thresholds=np.full(3, 100.0),
            cumulative_allocations=np.array([0.0, 0.01, 0.02, 0.03]),
        )
    with pytest.raises(ValueError, match="non-increasing"):
        replace(lond, gamma=np.array([0.3, 0.5, 0.2]))
    with pytest.raises(ValueError, match="evalues"):
        replace(lond, evalues=np.array([np.nan, 1.0, 1.0]))
    with pytest.raises(ValueError, match="positive with sum"):
        replace(lond, gamma=np.array([0.5, 0.5, 0.5]))
    with pytest.raises(ValueError, match="levels"):
        replace(lond, levels=np.array([0.2, 0.05, 0.03]))
    with pytest.raises(ValueError, match="thresholds"):
        replace(lond, thresholds=np.zeros(3))
    with pytest.raises(ValueError, match="non-negative"):
        replace(lond, cumulative_allocations=np.array([0.0, -1.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="align"):
        _validate_gamma(np.array([0.5]), (3,))
    with pytest.raises(ValueError, match="sum"):
        _validate_gamma(np.array([0.5, 0.5, 0.5]), (3,))
    with pytest.raises(ValueError, match="non-increasing"):
        _validate_gamma(np.array([0.2, 0.3, 0.1]), (3,))
    with pytest.raises(ValueError, match="SAFFRON level"):
        _expected_online_levels(
            "saffron",
            np.array([0.1]),
            0.1,
            np.array([1.0]),
            np.array([True]),
            np.array([False]),
            None,
            0.01,
        )


def test_e_process_and_confidence_sequence_paths_reject_inconsistent_metadata() -> None:
    e_result = binary_e_process([1, 1, 1], 0.01, 0.99, alpha=0.05)
    with pytest.raises(ValueError, match="finite"):
        replace(
            e_result,
            e_process_values=np.array(
                [1.0, np.nan, e_result.e_process_values[-1], e_result.e_process_values[-1]]
            ),
        )
    with pytest.raises(ValueError, match="non-negative"):
        replace(
            e_result,
            e_process_values=np.array(
                [1.0, -1.0, e_result.e_process_values[2], e_result.e_process_values[3]]
            ),
        )
    with pytest.raises(ValueError, match="start"):
        replace(e_result, e_process_values=np.array([0.9, *e_result.e_process_values[1:]]))
    with pytest.raises(ValueError, match="integer"):
        replace(e_result, crossing_time=True)
    with pytest.raises(ValueError, match="first Ville"):
        replace(e_result, crossing_time=2)
    with pytest.raises(ValueError, match="missing despite"):
        replace(e_result, crossing_time=None)
    with pytest.raises(ValueError, match="terminal"):
        replace(e_result, terminal_e=1.0)
    with pytest.raises(ValueError, match="boundary"):
        replace(e_result, boundary=2.0)
    sequence = bernoulli_confidence_sequence([1, 0, 1, 1], alpha=0.1)
    with pytest.raises(ValueError, match="validity"):
        replace(sequence, validity="")
    with pytest.raises(ValueError, match="arrays"):
        replace(sequence, empty=np.array([False]))
    with pytest.raises(ValueError, match="align with observations"):
        replace(sequence, observations=np.array([1, 0, 1]))
    with pytest.raises(ValueError, match="finite"):
        replace(sequence, lower=np.array([np.nan, *sequence.lower[1:]]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        replace(
            sequence,
            upper=np.array([2.0, *sequence.upper[1:]]),
            empty=np.zeros(sequence.empty.shape, dtype=bool),
        )
    mixture = mixture_e_process([1, 0], 0.4, [0.7, 0.8])
    assert mixture.e_process_values.size == 3
    extreme = bernoulli_confidence_sequence([1] * 32, alpha=0.5)
    assert not np.any(extreme.empty)


def test_selection_and_e_process_dataclass_edge_values() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=2)
    result = run_split_stream_confirmation(scenario, selection_seed=61, confirmation_seed=62)
    with pytest.raises(ValueError, match="non-negative"):
        replace(result.selection, selected_target=-1)
    with pytest.raises(ValueError, match="current action"):
        replace(
            result.selection,
            events=(replace(result.selection.events[0], visible_evidence=(0.2,)),),
        )
    blind = SelectionPolicySpec(strategy="round_robin", evidence_observable=False, max_steps=2, confirmation_steps=2)
    blind_result = run_split_stream_confirmation(scenario, blind, selection_seed=63, confirmation_seed=64)
    with pytest.raises(ValueError, match="evidence-blind"):
        replace(
            blind_result.selection,
            events=(replace(blind_result.selection.events[0], visible_actions=("observe",), visible_evidence=(0.2,)),),
        )
    with pytest.raises(ValueError, match="final filtration"):
        replace(result.selection, selected_target=99)
    with pytest.raises(ValueError, match="stream hash"):
        replace(result.selection, stream_hash="f" * 64)


def test_selection_empty_and_parallel_paths_are_explicit() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=2)
    empty_trace = run_action_loop(
        replace(scenario, setting=replace(scenario.setting, posterior_threshold=0.0)),
        "posterior_threshold",
        seed=71,
    )
    assert empty_trace.steps == ()
    with pytest.raises(ValueError, match="no visible events"):
        _selection_events(empty_trace, 2, True)
    assert _choose_target(empty_trace, "minimum_p", 2, True) == (0, 1.0)
    with pytest.raises(ValueError, match="no action"):
        _confirmation_stream(scenario, target=99, seed=74, steps=2)
    initial = np.zeros_like(scenario.process.initial_state)
    incompatible_target0 = (scenario.model.state_target == 0) & (scenario.model.state_truth == 0)
    initial[incompatible_target0] = 1.0
    initial /= np.sum(initial)
    altered = replace(scenario, process=replace(scenario.process, initial_state=initial))
    with pytest.raises(ValueError, match="initial mass"):
        _confirmation_stream(altered, target=0, seed=75, steps=2)
    action_trace = run_action_loop(scenario, "fixed_horizon", seed=76)
    assert _choose_target(action_trace, "minimum_p", 2, True)[0] >= 0
    serial = simulate_split_stream_confirmation(scenario, 3, seed=78, workers=1)
    parallel = simulate_split_stream_confirmation(scenario, 3, seed=78, workers=2)
    assert serial == parallel
    with pytest.raises(ValueError, match="n_reps"):
        simulate_split_stream_confirmation(scenario, 0)
    with pytest.raises(ValueError, match="workers"):
        simulate_split_stream_confirmation(scenario, 1, workers=0)


def test_action_evidence_contract_edges_are_fail_closed() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=2)
    assert _single_step_pvalue(scenario, 4, None) == (1.0, 1.0)
    no_null_prior = np.zeros_like(scenario.model.prior)
    no_null_prior[scenario.model.state_target == 0] = scenario.model.state_truth[scenario.model.state_target == 0]
    no_null_prior /= np.sum(no_null_prior)
    with pytest.raises(ValueError, match="positive prior"):
        _single_step_pvalue(replace(scenario, model=replace(scenario.model, prior=no_null_prior)), 0, 1)
    bad_likelihood = scenario.process.likelihood.copy()
    bad_likelihood[0, 1, :] = 0.0
    bad_likelihood[0, 0, :] = 1.0
    with pytest.raises(ValueError, match="non-degenerate"):
        _single_step_pvalue(replace(scenario, process=replace(scenario.process, likelihood=bad_likelihood)), 0, 1)
    assert _e_process_contract_valid(replace(scenario, process=replace(scenario.process, missing_probability=0.1))) is False
    # The process constructor still enforces normalized likelihood rows; use a
    # normalized but semantically different channel for the contract.
    assert _e_process_contract_valid(
        replace(scenario, process=replace(scenario.process, likelihood=np.roll(scenario.process.likelihood, 1, axis=1)))
    ) is False
    initial = np.zeros_like(scenario.process.initial_state)
    initial[(scenario.model.state_target == 0) & (scenario.model.state_truth == 0)] = 1.0
    initial /= np.sum(initial)
    assert _e_process_contract_valid(replace(scenario, process=replace(scenario.process, initial_state=initial))) is False


def test_result_schema_rejects_nonfinite_and_unbounded_summary_values() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=2)
    result = simulate_action_loop(scenario, "fixed_horizon", 2, seed=51, retain_traces=1)
    for value in (float("nan"), float("inf"), -float("inf")):
        broken = {**result, "summary": {**result["summary"], "power": value}}
        with pytest.raises(ValueError, match="numeric"):
            validate_action_loop_result(broken)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_action_loop_result({**result, "summary": {**result["summary"], "power_interval": [0.0, 2.0]}})

    operation = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "policy": "fixed_horizon",
        "n_reps": 1,
        "summary": {},
        "validity": "diagnostic",
    }
    base: dict[str, Any] = {
        "suite_name": "suite",
        "scenarios": [scenario.scenario_id],
        "policies": ["fixed_horizon"],
        "operating_characteristics": [operation],
        "online_fdr": {"method": "lord_plus_plus", "mean_fdr": 0.0},
        "claim_boundary": "conditional",
    }
    with pytest.raises(ValueError, match="method"):
        validate_suite_result({**base, "online_fdr": {"method": "invalid"}})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_suite_result({**base, "online_fdr": {"mean_fdr": 2.0}})
    with pytest.raises(ValueError, match="estimand"):
        validate_suite_result({**base, "online_fdr": {"fdr_estimand": ""}})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_suite_result({**base, "online_fdr": {"mean_fdr_interval": [0.0, 2.0]}})


def test_nested_hash_and_registry_schema_fail_closed(
    tmp_path: Path,
    generated_fast_bundle: Path,
) -> None:
    scenario = make_static_binary_e_process_scenario(horizon=2)
    summary = simulate_split_stream_confirmation(scenario, 1, seed=91)
    operation = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "policy": "fixed_horizon",
        "n_reps": 1,
        "summary": {},
        "validity": "diagnostic",
    }
    suite: dict[str, Any] = {
        "suite_name": "suite",
        "scenarios": [scenario.scenario_id],
        "policies": ["fixed_horizon"],
        "operating_characteristics": [operation],
        "online_fdr": {},
        "claim_boundary": "conditional",
        "split_stream_confirmation": summary,
    }
    broken = copy.deepcopy(suite)
    representative = cast(
        dict[str, Any], broken["split_stream_confirmation"]["representative"]
    )
    representative["selection_stream_hash"] = "f" * 64
    with pytest.raises(ValueError, match="selection hash is inconsistent"):
        validate_suite_result(broken)
    broken = copy.deepcopy(suite)
    representative = cast(
        dict[str, Any], broken["split_stream_confirmation"]["representative"]
    )
    cast(dict[str, Any], representative["selection"])["selected_target_evidence"] = 0.123
    with pytest.raises(ValueError, match="selection hash does not"):
        validate_suite_result(broken)
    broken = copy.deepcopy(suite)
    representative = cast(
        dict[str, Any], broken["split_stream_confirmation"]["representative"]
    )
    representative["confirmation_stream_hash"] = "f" * 64
    with pytest.raises(ValueError, match="confirmation hash"):
        validate_suite_result(broken)

    diagnostic: dict[str, Any] = {
        "scenario_id": "s",
        "scenario_hash": "a" * 64,
        "policy": "adaptive_target",
        "n_reps": 1,
        "pooled": {
            "n_selected": 1,
            "max_ecdf_excess": 0.0,
            "dkw_radius": 0.5,
            "passes_dkw_diagnostic": True,
            "dkw_interpretation": "nominal iid",
        },
        "validity": "calibration_diagnostic",
    }
    for mutation, message in (
        ({"policy": ""}, "policy"),
        ({"pooled": {**diagnostic["pooled"], "max_ecdf_excess": -1.0}}, "non-negative"),
        ({"pooled": {**diagnostic["pooled"], "dkw_interpretation": ""}}, "interpretation"),
    ):
        broken_diag = {**diagnostic, **mutation}
        with pytest.raises(ValueError, match=message):
            validate_suite_result({**suite, "conditional_selection_calibration": broken_diag})

    registry_path = generated_fast_bundle / "figures" / "figure_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    duplicate = copy.deepcopy(registry)
    duplicate["figures"][0]["filename"] = duplicate["figures"][1]["filename"]
    with pytest.raises(ValueError, match="unique"):
        validate_figure_registry(duplicate)
    empty_caption = copy.deepcopy(registry)
    empty_caption["figures"][0]["caption"] = ""
    with pytest.raises(ValueError, match="non-empty"):
        validate_figure_registry(empty_caption)
    empty_estimand = copy.deepcopy(registry)
    empty_estimand["figures"][0]["metadata"]["estimand"] = ""
    with pytest.raises(ValueError, match="non-empty"):
        validate_figure_registry(empty_estimand)
    malformed_reader_question = copy.deepcopy(registry)
    malformed_reader_question["figures"][0]["reader_question"] = ""
    with pytest.raises(ValueError, match="reader_question"):
        validate_figure_registry(malformed_reader_question)
    missing_reader_semantics = copy.deepcopy(registry)
    missing_reader_semantics["schema_version"] = "2.1"
    missing_reader_semantics["figures"][0].pop("reader_question")
    with pytest.raises(ValueError, match="required by schema 2.1"):
        validate_figure_registry(missing_reader_semantics)
    missing_visible_comparison = copy.deepcopy(registry)
    missing_visible_comparison["schema_version"] = "2.2"
    missing_visible_comparison["figures"][0].pop("visible_comparison")
    with pytest.raises(ValueError, match="visible_comparison is required by schema 2.2"):
        validate_figure_registry(missing_visible_comparison)
    missing_evaluation_perspective = copy.deepcopy(registry)
    missing_evaluation_perspective["schema_version"] = "2.4"
    missing_evaluation_perspective["figures"][0].pop("evaluation_perspective")
    with pytest.raises(ValueError, match="evaluation_perspective is required by schema 2.4"):
        validate_figure_registry(missing_evaluation_perspective)
    missing_panel_validity = copy.deepcopy(registry)
    missing_panel_validity["figures"][0].pop("panel_validity")
    with pytest.raises(ValueError, match="panel_validity is required by schema 2.4"):
        validate_figure_registry(missing_panel_validity)
    historical_reader_semantics = copy.deepcopy(registry)
    historical_reader_semantics["schema_version"] = "2.1"
    historical_reader_semantics["figures"][0].pop("visible_comparison")
    validate_figure_registry(historical_reader_semantics)

    benchmark_manifest = {
        "schema_version": "2.0",
        "benchmarks": [{
            "id": "x", "source": "local", "filename": "x", "license": "MIT",
            "sha256": "a" * 64, "retrieval_date": "2026-07-25", "schema": "csv",
            "preprocessing": "none", "claim_role": "stress", "source_revision": "",
        }],
    }
    with pytest.raises(ValueError, match="source_revision"):
        validate_benchmark_manifest(benchmark_manifest)

    checkpoint_path = generated_fast_bundle / "checkpoints" / "power_curve.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("completed_replications", 0) > 1:
        checkpoint["total_replications"] = 1
        with pytest.raises(ValueError, match="exceed"):
            validate_checkpoint(checkpoint)


def test_model_process_and_truth_contracts_reject_shape_and_schedule_errors() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=2)
    with pytest.raises(ValueError, match="transition shape"):
        replace(scenario.model, transition=np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="process likelihood"):
        replace(scenario.process, likelihood=scenario.process.likelihood[:, :, :-1])
    with pytest.raises(ValueError, match="transition_schedule must contain"):
        replace(scenario.process, transition_schedule=np.empty((0,) + scenario.process.transition.shape))
    with pytest.raises(ValueError, match="likelihood_schedule must contain"):
        replace(scenario.process, likelihood_schedule=np.empty((0,) + scenario.process.likelihood.shape))
    with pytest.raises(ValueError, match="action_cost_schedule must contain"):
        replace(scenario.process, action_cost_schedule=np.empty((0, scenario.process.action_costs.size)))
    with pytest.raises(ValueError, match="action_cost_schedule must be non-negative"):
        replace(
            scenario.process,
            action_cost_schedule=np.full((1, scenario.process.action_costs.size), -1.0),
        )
    with pytest.raises(ValueError, match="missing_probability_schedule must contain"):
        replace(scenario.process, missing_probability_schedule=np.array([], dtype=float))
    with pytest.raises(ValueError, match="action_missing_probability"):
        replace(scenario.process, action_missing_probability=np.zeros(1))
    with pytest.raises(ValueError, match="state_missing_probability"):
        replace(scenario.process, state_missing_probability=np.zeros(1))
    with pytest.raises(ValueError, match="reliability"):
        # The builder is the real reference constructor, so this also checks
        # the public boundary rather than a hand-built invalid scenario.
        from active_inference_power.scenario import make_reference_action_scenario

        make_reference_action_scenario(reliability=1.0)

    initial = np.zeros_like(scenario.process.initial_state)
    inconsistent = (scenario.model.state_target == 0) & (scenario.model.state_truth == 0)
    initial[inconsistent] = 1.0
    initial /= np.sum(initial)
    altered = replace(scenario, process=replace(scenario.process, initial_state=initial))
    assert altered.truth_consistent is False
    schedule = np.stack((scenario.process.transition,))
    current = int(np.flatnonzero((scenario.model.state_target == 0) & (scenario.model.state_truth == 1))[0])
    next_state = int(np.flatnonzero(scenario.model.state_truth != scenario.process.target_truth[scenario.model.state_target])[0])
    schedule[0, 0, :, current] = 0.0
    schedule[0, 0, next_state, current] = 1.0
    scheduled = replace(scenario, process=replace(scenario.process, transition_schedule=schedule))
    assert scheduled.truth_consistent is False
