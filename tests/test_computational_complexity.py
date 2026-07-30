"""Real-artifact tests for source-level computational-work accounting."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from active_inference_power.computational_complexity import (
    ComputationalComplexityPlan,
    action_workload_ledger,
    build_computational_complexity,
    complexity_plan_from_mapping,
    linear_work,
    lord_history_work_upper,
    rank_work,
    saffron_dense_history_work_upper,
    saffron_sparse_history_work_upper,
    validate_computational_complexity,
    validate_computational_complexity_against_results,
)
from active_inference_power.schema import validate_results


def _plan() -> ComputationalComplexityPlan:
    """Return a small, fully declared real plan for deterministic work curves."""
    return ComputationalComplexityPlan(
        hypothesis_grid=(2, 4, 8),
        stream_horizon_grid=(2, 4, 8),
        action_horizon_grid=(2, 4, 8),
        timing_warmups=1,
        timing_repetitions=7,
        correction_methods=("bonferroni", "benjamini_hochberg"),
        online_methods=("lord_plus_plus", "saffron", "e_lond"),
        action_policies=(
            "fixed_horizon",
            "information_gain",
            "cost_aware",
            "e_process",
        ),
    )


def _small_declared_action_suite() -> dict[str, Any]:
    """Return a compact, internally reconciled execution-ledger contract."""
    return {
        "operating_characteristics": [{"n_reps": 2}],
        "paired_policy_comparisons": [
            {"n_reps": 2, "policies": ["fixed_horizon", "e_process"]}
        ],
        "online_fdr": {"n_reps": 2},
        "conditional_selection_calibration": {"n_reps": 2},
        "future_evidence_selection_failure": {"n_reps": 2},
        "split_stream_confirmation": {
            "n_reps": 3,
            "selection_streams": 3,
            "confirmation_streams": 3,
        },
        "learned_policy": {
            "artifact": {
                "policy_complexity": {"candidate_count": 2},
                "train_reps": 2,
            },
            "evaluation": {"evaluation_reps": 2},
        },
    }


def _build_from_real_results(results: dict[str, Any]) -> dict[str, Any]:
    """Derive a release accounting block from the real fast suite artifact."""
    experiment = results["experiment"]
    action_loop = experiment["action_loop"]
    action_horizon = int(action_loop["horizon"])
    return build_computational_complexity(
        plan=complexity_plan_from_mapping(experiment["computational_complexity"]),
        m=int(experiment["m"]),
        main_replications=int(results["meta"]["n_reps"]),
        method_count=len(experiment["methods"]),
        effect_points=len(experiment["effect_grid"]),
        dependence_regime_count=len(results["dependence_sweep"]),
        sequential_horizon=int(results["sequential"]["horizon"]),
        online_horizon=int(results["online_fdr"]["horizon"]),
        action_horizon=action_horizon,
        split_confirmation_horizon=int(
            action_loop.get("split_confirmation_steps", action_horizon)
        ),
        suite=results["suite"],
    )


def test_source_work_formulae_have_hand_checked_values_and_ordering() -> None:
    """Exercise formulae without turning source work into elapsed-time claims."""
    assert linear_work(1) == 1
    assert linear_work(8) == 8
    assert rank_work(1) == 1
    assert rank_work(8) == 24
    assert rank_work(9) == 36
    assert lord_history_work_upper(4) == 40
    assert saffron_sparse_history_work_upper(4) == 16
    assert saffron_dense_history_work_upper(4) == 44

    horizons = (2, 4, 8, 16)
    lord = [lord_history_work_upper(horizon) for horizon in horizons]
    sparse = [saffron_sparse_history_work_upper(horizon) for horizon in horizons]
    dense = [saffron_dense_history_work_upper(horizon) for horizon in horizons]
    assert all(later > earlier for earlier, later in zip(dense, dense[1:]))
    assert all(
        dense_value > sparse_value for dense_value, sparse_value in zip(dense, sparse)
    )
    assert all(
        lord_value > sparse_value for lord_value, sparse_value in zip(lord, sparse)
    )

    with pytest.raises(ValueError, match="positive integer"):
        rank_work(0)
    with pytest.raises(ValueError, match="positive integer"):
        lord_history_work_upper(True)


def test_stream_plot_combines_only_the_coincident_linear_proxies() -> None:
    """Avoid occluding distinct labels when two displayed proxy curves coincide."""
    accounting = build_computational_complexity(
        plan=_plan(),
        m=8,
        main_replications=3,
        method_count=2,
        effect_points=3,
        dependence_regime_count=2,
        sequential_horizon=8,
        online_horizon=8,
        action_horizon=4,
        split_confirmation_horizon=2,
        suite=_small_declared_action_suite(),
    )
    stream_curves = accounting["plot_data"]["stream_work_curves"]
    assert [curve["name"] for curve in stream_curves] == [
        "Binary e-process and e-LOND (coincident linear proxy)",
        "LORD++ recurrence-plus-validator upper proxy",
        "SAFFRON sparse-history upper proxy",
        "SAFFRON dense-history upper proxy",
    ]
    assert stream_curves[0]["work_proxy_values"] == [2, 4, 8]
    assert {row["name"] for row in accounting["analytical_rows"]} >= {
        "binary_e_process",
        "e_lond",
    }
    validate_computational_complexity(accounting)


def test_complexity_plan_round_trip_and_rejects_unstable_timing_protocol() -> None:
    """The timing protocol is configured strictly even though release skips timing."""
    plan = _plan()
    assert complexity_plan_from_mapping(plan.to_mapping()) == plan

    too_few_repetitions = plan.to_mapping()
    too_few_repetitions["timing_repetitions"] = 6
    with pytest.raises(ValueError, match="at least seven"):
        complexity_plan_from_mapping(too_few_repetitions)

    repeated_grid = plan.to_mapping()
    repeated_grid["stream_horizon_grid"] = [2, 2, 8]
    with pytest.raises(ValueError, match="strictly increasing"):
        complexity_plan_from_mapping(repeated_grid)

    unknown_method = plan.to_mapping()
    unknown_method["online_methods"] = ["e_lond", "unbounded"]
    with pytest.raises(ValueError, match="ordered subset"):
        complexity_plan_from_mapping(unknown_method)

    unreviewed_field = plan.to_mapping()
    unreviewed_field["runtime_multiplier"] = 1
    with pytest.raises(ValueError, match="unexpected keys"):
        complexity_plan_from_mapping(unreviewed_field)

    incomplete_plan = plan.to_mapping()
    incomplete_plan.pop("hypothesis_grid")
    with pytest.raises(ValueError, match="missing keys"):
        complexity_plan_from_mapping(incomplete_plan)


def test_complexity_plan_rejects_noncanonical_configuration_shapes() -> None:
    """Configuration validation rejects values that could blur a declared plan."""
    plan = _plan()

    nonmapping_plan: Any = []
    with pytest.raises(ValueError, match="must be a mapping"):
        complexity_plan_from_mapping(nonmapping_plan)

    invalid_warmup_type = plan.to_mapping()
    invalid_warmup_type["timing_warmups"] = False
    with pytest.raises(ValueError, match="non-negative integer"):
        complexity_plan_from_mapping(invalid_warmup_type)

    invalid_warmup_value = plan.to_mapping()
    invalid_warmup_value["timing_warmups"] = -1
    with pytest.raises(ValueError, match="non-negative integer"):
        complexity_plan_from_mapping(invalid_warmup_value)

    non_sequence_grid = plan.to_mapping()
    non_sequence_grid["hypothesis_grid"] = "2,4,8"
    with pytest.raises(ValueError, match="non-empty sequence"):
        complexity_plan_from_mapping(non_sequence_grid)

    non_sequence_methods = plan.to_mapping()
    non_sequence_methods["correction_methods"] = "bonferroni"
    with pytest.raises(ValueError, match="non-empty sequence"):
        complexity_plan_from_mapping(non_sequence_methods)


def test_action_ledger_reconciles_each_split_stream_without_generator() -> None:
    """Selection and confirmation streams are distinct but must reconcile exactly."""
    suite = _small_declared_action_suite()
    ledger = action_workload_ledger(
        suite,
        action_horizon=4,
        split_confirmation_horizon=2,
    )
    categories = {record["key"]: record for record in ledger["trace_categories"]}
    assert categories["split_stream_selection"] == {
        "key": "split_stream_selection",
        "label": "Split-stream visible selection",
        "trace_calls": 3,
        "max_steps_per_trace": 2,
    }
    assert categories["split_stream_confirmation"] == {
        "key": "split_stream_confirmation",
        "label": "Split-stream independent confirmation",
        "trace_calls": 3,
        "max_steps_per_trace": 2,
    }
    assert ledger["action_loop_trace_calls"] == 24
    assert ledger["action_loop_step_upper_bound"] == 84

    mismatched_selection = copy.deepcopy(suite)
    mismatched_selection["split_stream_confirmation"]["selection_streams"] = 2
    with pytest.raises(ValueError, match="must equal"):
        action_workload_ledger(
            mismatched_selection,
            action_horizon=4,
            split_confirmation_horizon=2,
        )

    mismatched_confirmation = copy.deepcopy(suite)
    mismatched_confirmation["split_stream_confirmation"]["confirmation_streams"] = 2
    with pytest.raises(ValueError, match="must equal"):
        action_workload_ledger(
            mismatched_confirmation,
            action_horizon=4,
            split_confirmation_horizon=2,
        )


def test_action_ledger_uses_real_held_out_evaluation_replications(
    generated_fast_bundle: Path,
) -> None:
    """The real suite distinguishes held-out, selection, and confirmation paths."""
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    action_loop = results["experiment"]["action_loop"]
    ledger = action_workload_ledger(
        results["suite"],
        action_horizon=int(action_loop["horizon"]),
        split_confirmation_horizon=int(action_loop["split_confirmation_steps"]),
    )
    evaluation = results["suite"]["learned_policy"]["evaluation"]
    learned_evaluation = next(
        row
        for row in ledger["trace_categories"]
        if row["key"] == "learned_policy_evaluation"
    )
    assert learned_evaluation["trace_calls"] == evaluation["evaluation_reps"]
    split = results["suite"]["split_stream_confirmation"]
    split_selection = next(
        row
        for row in ledger["trace_categories"]
        if row["key"] == "split_stream_selection"
    )
    split_confirmation = next(
        row
        for row in ledger["trace_categories"]
        if row["key"] == "split_stream_confirmation"
    )
    assert split_selection["trace_calls"] == split["selection_streams"]
    assert (
        split_selection["max_steps_per_trace"]
        == action_loop["split_confirmation_steps"]
    )
    assert split_confirmation["trace_calls"] == split["confirmation_streams"]
    assert (
        split_confirmation["max_steps_per_trace"]
        == action_loop["split_confirmation_steps"]
    )
    assert ledger["static_confirmation_streams"] == split["confirmation_streams"]
    assert ledger["confirmation_observation_slots_upper_bound"] == (
        split["confirmation_streams"] * action_loop["split_confirmation_steps"]
    )
    assert ledger["action_loop_trace_calls"] == sum(
        row["trace_calls"] for row in ledger["trace_categories"]
    )
    assert ledger["action_loop_step_upper_bound"] == sum(
        row["trace_calls"] * row["max_steps_per_trace"]
        for row in ledger["trace_categories"]
    )


def test_action_ledger_rejects_incomplete_real_suite_records(
    generated_fast_bundle: Path,
) -> None:
    """Each declared execution family must retain its real replication contract."""
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    action_loop = results["experiment"]["action_loop"]
    action_horizon = int(action_loop["horizon"])
    split_confirmation_horizon = int(action_loop["split_confirmation_steps"])

    fallback_evaluation_key = copy.deepcopy(results["suite"])
    evaluation = fallback_evaluation_key["learned_policy"]["evaluation"]
    evaluation["n_reps"] = evaluation.pop("evaluation_reps")
    fallback_ledger = action_workload_ledger(
        fallback_evaluation_key,
        action_horizon=action_horizon,
        split_confirmation_horizon=split_confirmation_horizon,
    )
    assert (
        fallback_ledger["trace_categories"][-1]["trace_calls"] == evaluation["n_reps"]
    )

    missing_operating = copy.deepcopy(results["suite"])
    missing_operating["operating_characteristics"] = []
    with pytest.raises(ValueError, match="operating_characteristics"):
        action_workload_ledger(
            missing_operating,
            action_horizon=action_horizon,
            split_confirmation_horizon=split_confirmation_horizon,
        )

    missing_paired = copy.deepcopy(results["suite"])
    missing_paired["paired_policy_comparisons"] = []
    with pytest.raises(ValueError, match="paired_policy_comparisons"):
        action_workload_ledger(
            missing_paired,
            action_horizon=action_horizon,
            split_confirmation_horizon=split_confirmation_horizon,
        )

    missing_policies = copy.deepcopy(results["suite"])
    missing_policies["paired_policy_comparisons"][0]["policies"] = []
    with pytest.raises(ValueError, match="policies"):
        action_workload_ledger(
            missing_policies,
            action_horizon=action_horizon,
            split_confirmation_horizon=split_confirmation_horizon,
        )

    nonmapping_online = copy.deepcopy(results["suite"])
    nonmapping_online["online_fdr"] = []
    with pytest.raises(ValueError, match="suite.online_fdr must be a mapping"):
        action_workload_ledger(
            nonmapping_online,
            action_horizon=action_horizon,
            split_confirmation_horizon=split_confirmation_horizon,
        )

    missing_online_repetitions = copy.deepcopy(results["suite"])
    missing_online_repetitions["online_fdr"] = {}
    with pytest.raises(ValueError, match="must declare"):
        action_workload_ledger(
            missing_online_repetitions,
            action_horizon=action_horizon,
            split_confirmation_horizon=split_confirmation_horizon,
        )


def test_complexity_schema_is_strict_when_present_but_optional_for_nonrelease_readers(
    generated_fast_bundle: Path,
) -> None:
    """Strict release readers require accounting; nonrelease readers may omit it."""
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    assert "computational_complexity" in results
    accounting = results["computational_complexity"]
    validate_computational_complexity(accounting)

    project_root = Path(__file__).resolve().parent.parent
    configured = yaml.safe_load(
        (project_root / "manuscript" / "config.yaml").read_text(encoding="utf-8")
    )["experiment"]["computational_complexity"]
    plan = complexity_plan_from_mapping(configured)
    emitted_plan = complexity_plan_from_mapping(
        results["experiment"]["computational_complexity"]
    )
    assert emitted_plan == plan
    assert accounting["plot_data"]["hypothesis_work_curves"][0]["input_values"] == list(
        plan.hypothesis_grid
    )
    assert accounting["plot_data"]["stream_work_curves"][0]["input_values"] == list(
        plan.stream_horizon_grid
    )
    dependence_row = next(
        row
        for row in accounting["analytical_rows"]
        if row["name"] == "dependence_paths"
    )
    assert (
        dependence_row["work_proxy_expression"]
        == "2 D B m^2 dense-draw upper proxy after cache construction"
    )
    assert "generate_pvalues factor fast path" in dependence_row["implementation_scope"]
    assert "dependence.sample_gaussian" not in dependence_row["implementation_scope"]
    assert "min(non-factor regimes, 64)" in dependence_row["retained_space_order"]
    assert dependence_row["work_proxy_at_declared_input"] == (
        2
        * accounting["dimensions"]["D_dependence_regimes"]
        * accounting["dimensions"]["B_main_replications"]
        * accounting["dimensions"]["m_hypotheses"] ** 2
    )
    action_row = next(
        row
        for row in accounting["analytical_rows"]
        if row["name"] == "action_loop_execution_ledger"
    )
    assert action_row["input_symbol"] == "N_j,H_j"
    assert "sum_j N_j H_j" in action_row["work_proxy_expression"]
    assert all(
        "T_method" not in str(row["time_order"])
        for row in accounting["analytical_rows"]
    )

    rebuilt = _build_from_real_results(results)
    validate_computational_complexity(rebuilt)
    assert rebuilt == accounting

    current = copy.deepcopy(results)
    validate_results(current, require_suite=True, require_current_contract=True)

    compact = copy.deepcopy(current)
    compact.pop("computational_complexity")
    validate_results(compact, require_suite=True)
    with pytest.raises(ValueError, match="computational_complexity is required"):
        validate_results(
            compact,
            require_suite=True,
            require_current_contract=True,
        )


def test_release_accounting_rejects_timing_fields_and_preserves_term_boundaries(
    generated_fast_bundle: Path,
) -> None:
    """A deterministic release block cannot silently become a host benchmark."""
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    accounting = _build_from_real_results(results)
    boundary = accounting["claim_boundary"]
    assert "simulated resource cost C" in boundary
    assert "e-process evidence E_t" in boundary
    assert "online alpha-wealth W^alpha" in boundary
    assert "elapsed time" in accounting["units"]["work_proxy"]

    malformed = copy.deepcopy(accounting)
    malformed["release_workload"]["wall_time_ns"] = 17
    with pytest.raises(ValueError, match="elapsed-time field"):
        validate_computational_complexity(malformed)

    results_with_malformed_accounting = copy.deepcopy(results)
    results_with_malformed_accounting["computational_complexity"] = malformed
    with pytest.raises(ValueError, match="elapsed-time field"):
        validate_results(results_with_malformed_accounting, require_suite=True)


def test_release_accounting_validator_rejects_inconsistent_contract_parts(
    generated_fast_bundle: Path,
) -> None:
    """The release receipt is closed against inconsistent analytical summaries."""
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    accounting = _build_from_real_results(results)
    invalid_records: list[tuple[dict[str, Any], str]] = []

    blank_boundary = copy.deepcopy(accounting)
    blank_boundary["claim_boundary"] = " "
    invalid_records.append((blank_boundary, "non-empty text"))

    unknown_version = copy.deepcopy(accounting)
    unknown_version["contract_version"] = "unrecognized"
    invalid_records.append((unknown_version, "contract_version"))

    wrong_evidence_class = copy.deepcopy(accounting)
    wrong_evidence_class["evidence_class"] = "performance_claim"
    invalid_records.append((wrong_evidence_class, "evidence_class"))

    no_analytical_rows = copy.deepcopy(accounting)
    no_analytical_rows["analytical_rows"] = []
    invalid_records.append((no_analytical_rows, "analytical_rows"))

    duplicate_analytical_name = copy.deepcopy(accounting)
    duplicate_analytical_name["analytical_rows"].append(
        copy.deepcopy(duplicate_analytical_name["analytical_rows"][0])
    )
    invalid_records.append((duplicate_analytical_name, "names must be unique"))

    reordered_analytical_rows = copy.deepcopy(accounting)
    rows = reordered_analytical_rows["analytical_rows"]
    rows[0], rows[1] = rows[1], rows[0]
    invalid_records.append((reordered_analytical_rows, "ordered source-work rows"))

    no_hypothesis_curves = copy.deepcopy(accounting)
    no_hypothesis_curves["plot_data"]["hypothesis_work_curves"] = []
    invalid_records.append((no_hypothesis_curves, "hypothesis_work_curves"))

    incomplete_stream_curve_family = copy.deepcopy(accounting)
    incomplete_stream_curve_family["plot_data"]["stream_work_curves"].pop()
    invalid_records.append((incomplete_stream_curve_family, "declared curve family"))

    altered_curve_metadata = copy.deepcopy(accounting)
    altered_curve_metadata["plot_data"]["stream_work_curves"][0]["name"] = (
        "Undeclared linear proxy"
    )
    invalid_records.append((altered_curve_metadata, "curve metadata"))

    altered_curve_unit = copy.deepcopy(accounting)
    altered_curve_unit["plot_data"]["stream_work_curves"][0]["unit"] = "operations"
    invalid_records.append((altered_curve_unit, "curve unit"))

    misaligned_curve = copy.deepcopy(accounting)
    misaligned_curve["plot_data"]["stream_work_curves"][0]["work_proxy_values"].pop()
    invalid_records.append((misaligned_curve, "input/proxy arrays"))

    unordered_curve = copy.deepcopy(accounting)
    curve_inputs = unordered_curve["plot_data"]["stream_work_curves"][0]["input_values"]
    curve_inputs[1] = curve_inputs[0]
    invalid_records.append((unordered_curve, "strictly increasing"))

    no_action_categories = copy.deepcopy(accounting)
    no_action_categories["plot_data"]["action_trace_categories"] = []
    invalid_records.append((no_action_categories, "action_trace_categories"))

    duplicate_action_category = copy.deepcopy(accounting)
    duplicate_action_category["plot_data"]["action_trace_categories"][1] = (
        copy.deepcopy(
            duplicate_action_category["plot_data"]["action_trace_categories"][0]
        )
    )
    invalid_records.append((duplicate_action_category, "keys must be unique"))

    altered_action_label = copy.deepcopy(accounting)
    altered_action_label["plot_data"]["action_trace_categories"][0]["label"] = (
        "Unreviewed execution family"
    )
    invalid_records.append((altered_action_label, "declared labels and order"))

    unmatched_categories = copy.deepcopy(accounting)
    unmatched_categories["release_workload"]["action_loop"]["trace_categories"] = []
    invalid_records.append((unmatched_categories, "categories must equal"))

    wrong_trace_total = copy.deepcopy(accounting)
    wrong_trace_total["release_workload"]["action_loop"]["action_loop_trace_calls"] += 1
    invalid_records.append((wrong_trace_total, "trace total disagrees"))

    wrong_step_total = copy.deepcopy(accounting)
    wrong_step_total["release_workload"]["action_loop"][
        "action_loop_step_upper_bound"
    ] += 1
    invalid_records.append((wrong_step_total, "step bound disagrees"))

    wrong_static_confirmation = copy.deepcopy(accounting)
    wrong_static_confirmation["release_workload"]["action_loop"][
        "static_confirmation_streams"
    ] += 1
    invalid_records.append((wrong_static_confirmation, "static confirmation streams"))

    wrong_confirmation_slots = copy.deepcopy(accounting)
    wrong_confirmation_slots["release_workload"]["action_loop"][
        "confirmation_observation_slots_upper_bound"
    ] += 1
    invalid_records.append((wrong_confirmation_slots, "confirmation slots"))

    wrong_training_ledger = copy.deepcopy(accounting)
    wrong_training_ledger["release_workload"]["action_loop"][
        "learned_policy_candidate_count"
    ] += 1
    invalid_records.append((wrong_training_ledger, "learned-policy training calls"))

    unreviewed_root_field = copy.deepcopy(accounting)
    unreviewed_root_field["runtime_ms"] = 1
    invalid_records.append((unreviewed_root_field, "unexpected keys"))

    unreviewed_nested_field = copy.deepcopy(accounting)
    unreviewed_nested_field["plot_data"]["stream_work_curves"][0][
        "host_elapsed_observation"
    ] = 1
    invalid_records.append((unreviewed_nested_field, "unexpected keys"))

    altered_curve_proxy = copy.deepcopy(accounting)
    altered_curve_proxy["plot_data"]["stream_work_curves"][0]["work_proxy_values"][
        0
    ] += 1
    invalid_records.append(
        (altered_curve_proxy, "do not match their declared source-work formula")
    )

    altered_analytical_proxy = copy.deepcopy(accounting)
    altered_analytical_proxy["analytical_rows"][0]["work_proxy_at_declared_input"] += 1
    invalid_records.append((altered_analytical_proxy, "analytical source-work values"))

    altered_dimension = copy.deepcopy(accounting)
    altered_dimension["dimensions"]["m_hypotheses"] += 1
    invalid_records.append((altered_dimension, "fixed workload does not match"))

    for malformed, expected_message in invalid_records:
        with pytest.raises(ValueError, match=expected_message):
            validate_computational_complexity(malformed)


def test_result_bound_complexity_validation_rejects_missing_result_inputs(
    generated_fast_bundle: Path,
) -> None:
    """The strict ledger rebuild requires every result-owned source-work input."""
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    accounting = _build_from_real_results(results)

    missing_methods = copy.deepcopy(results)
    missing_methods["experiment"]["methods"] = []
    with pytest.raises(ValueError, match="methods must be a non-empty list"):
        validate_computational_complexity_against_results(accounting, missing_methods)

    missing_effect_grid = copy.deepcopy(results)
    missing_effect_grid["experiment"]["effect_grid"] = []
    with pytest.raises(ValueError, match="effect_grid must be a non-empty list"):
        validate_computational_complexity_against_results(
            accounting, missing_effect_grid
        )

    missing_dependence_sweep = copy.deepcopy(results)
    missing_dependence_sweep["dependence_sweep"] = []
    with pytest.raises(ValueError, match="dependence_sweep must be a non-empty list"):
        validate_computational_complexity_against_results(
            accounting, missing_dependence_sweep
        )


def test_current_results_rebuild_complexity_ledger_before_accepting_provenance(
    generated_fast_bundle: Path,
) -> None:
    """A plausible-but-substituted accounting block cannot inherit strict release status."""
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    substituted = copy.deepcopy(results)
    experiment = substituted["experiment"]
    action_loop = experiment["action_loop"]
    substituted["computational_complexity"] = build_computational_complexity(
        plan=complexity_plan_from_mapping(experiment["computational_complexity"]),
        m=int(experiment["m"]) + 1,
        main_replications=int(substituted["meta"]["n_reps"]),
        method_count=len(experiment["methods"]),
        effect_points=len(experiment["effect_grid"]),
        dependence_regime_count=len(substituted["dependence_sweep"]),
        sequential_horizon=int(substituted["sequential"]["horizon"]),
        online_horizon=int(substituted["online_fdr"]["horizon"]),
        action_horizon=int(action_loop["horizon"]),
        split_confirmation_horizon=int(action_loop["split_confirmation_steps"]),
        suite=substituted["suite"],
    )
    validate_computational_complexity(substituted["computational_complexity"])
    with pytest.raises(
        ValueError, match="does not exactly match the current results contract"
    ):
        validate_results(substituted, require_suite=True, require_current_contract=True)
