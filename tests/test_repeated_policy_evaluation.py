"""Real deterministic tests for repeated learned-policy partitions."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference_power.learned_policy import LearnedPolicyConfig
from active_inference_power.repeated_policy_evaluation import (
    RepeatedLearnedPolicyPlan,
    _normalized_plan,
    _normalized_search_config,
    _paired_replication_summary,
    _partition_bands,
    _require_unique_trace_seeds,
    _stream_record,
    _validate_partition_streams,
    evaluate_repeated_learned_policy,
)
from active_inference_power.scenario import make_static_binary_e_process_scenario


def _search_config() -> LearnedPolicyConfig:
    """Keep the real pymdp train/evaluate tests compact and deterministic."""
    return LearnedPolicyConfig(
        train_reps=1,
        cost_penalty=0.02,
        candidate_parameters=((1.0, 0.5, 0.0),),
    )


def test_repeated_policy_partitions_are_frozen_disjoint_and_paired() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=2)
    plan = RepeatedLearnedPolicyPlan(
        partitions=2,
        evaluation_reps=2,
        reference_policy="cost_aware",
    )
    result = evaluate_repeated_learned_policy(
        scenario, _search_config(), plan=plan, seed=71
    )

    assert (
        result["validity"]
        == "conditional_repeated_held_out_policy_operating_characteristic"
    )
    assert "does not establish policy optimality" in result["claim_boundary"]
    assert result["plan_hash"] == plan.configuration_hash
    assert result["search_config_hash"] == _search_config().configuration_hash
    assert result["candidate_space"] == {
        "search_type": "finite_tabular_candidate_search",
        "candidate_count": 1,
        "parameter_dimension": 3,
        "train_reps_per_partition": 1,
    }

    stream_proof = result["stream_proof"]
    assert stream_proof["all_materialized_trace_seeds_unique"] is True
    assert len(stream_proof["stream_proof_hash"]) == 64
    assert len(result["partitions"]) == plan.partitions
    assert len(result["frozen_policy_hashes"]) == plan.partitions

    materialized_seeds: list[int] = []
    for partition, proof in zip(result["partitions"], stream_proof["partitions"]):
        training = proof["training"]
        evaluation = proof["evaluation"]
        materialized_seeds.extend(training["trace_seed_schedule"])
        materialized_seeds.extend(evaluation["trace_seed_schedule"])
        assert proof["streams_disjoint"] is True
        assert proof["trace_seed_overlap"] == []
        assert set(training["trace_seed_schedule"]).isdisjoint(
            evaluation["trace_seed_schedule"]
        )
        assert partition["artifact"]["train_seed"] == training["root_seed"]
        assert partition["evaluation"]["evaluation_seed"] == evaluation["root_seed"]
        assert (
            partition["evaluation"]["seed_schedule"]
            == evaluation["trace_seed_schedule"]
        )
        assert (
            partition["reference_evaluation"]["seed_schedule"]
            == evaluation["trace_seed_schedule"]
        )
        assert (
            partition["evaluation"]["policy_hash"]
            == partition["artifact"]["policy_hash"]
        )
        assert (
            partition["evaluation"]["search_config_hash"]
            == result["search_config_hash"]
        )
        assert (
            partition["paired_held_out_differences"]["direction"]
            == "learned_minus_reference"
        )
        for metric in ("power", "cost", "stopping"):
            assert partition["paired_held_out_differences"][metric]["n"] == 2

    assert len(materialized_seeds) == len(set(materialized_seeds))
    bands = result["paired_held_out_bands"]
    assert bands["direction"] == "learned_minus_reference"
    assert bands["confidence"] == plan.confidence
    assert "do not pool" in bands["aggregation"]
    for metric in ("power", "cost", "stopping"):
        assert bands[metric]["n"] == plan.partitions
        assert (
            len(bands["partition_mean_values"]["reference"][metric]) == plan.partitions
        )
        assert len(bands["partition_mean_values"]["learned"][metric]) == plan.partitions


def test_repeated_policy_result_is_worker_invariant() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=2)
    plan = RepeatedLearnedPolicyPlan(partitions=2, evaluation_reps=1)
    serial = evaluate_repeated_learned_policy(
        scenario, _search_config(), plan=plan, seed=81, workers=1
    )
    parallel = evaluate_repeated_learned_policy(
        scenario, _search_config(), plan=plan, seed=81, workers=2
    )
    assert parallel == serial


def test_repeated_policy_plan_and_entry_point_fail_closed() -> None:
    for kwargs in (
        {"partitions": 1},
        {"partitions": True},
        {"evaluation_reps": 0},
        {"evaluation_reps": True},
        {"reference_policy": "learned"},
        {"reference_policy": "not_a_policy"},
        {"confidence": 1.0},
        {"confidence": float("nan")},
    ):
        with pytest.raises(ValueError):
            RepeatedLearnedPolicyPlan(**kwargs)

    scenario = make_static_binary_e_process_scenario(horizon=2)
    valid_plan = RepeatedLearnedPolicyPlan(partitions=2, evaluation_reps=1)
    with pytest.raises(ValueError, match="plan"):
        evaluate_repeated_learned_policy(
            scenario,
            _search_config(),
            plan=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="config"):
        evaluate_repeated_learned_policy(
            scenario,
            object(),  # type: ignore[arg-type]
            plan=valid_plan,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="seed"):
        evaluate_repeated_learned_policy(
            scenario, _search_config(), plan=valid_plan, seed=True
        )
    with pytest.raises(ValueError, match="workers"):
        evaluate_repeated_learned_policy(
            scenario, _search_config(), plan=valid_plan, workers=0
        )

    tampered_plan = RepeatedLearnedPolicyPlan(partitions=2, evaluation_reps=1)
    object.__setattr__(tampered_plan, "reference_policy", "learned")
    with pytest.raises(ValueError, match="reference_policy"):
        evaluate_repeated_learned_policy(scenario, _search_config(), plan=tampered_plan)

    tampered_config = _search_config()
    object.__setattr__(tampered_config, "candidate_parameters", ())
    with pytest.raises(ValueError, match="candidate_parameters"):
        evaluate_repeated_learned_policy(scenario, tampered_config, plan=valid_plan)


def test_repeated_plan_hash_binds_the_declared_comparator_and_precision() -> None:
    first = RepeatedLearnedPolicyPlan(partitions=2, evaluation_reps=1)
    second = RepeatedLearnedPolicyPlan(
        partitions=2, evaluation_reps=1, reference_policy="information_gain"
    )
    third = RepeatedLearnedPolicyPlan(partitions=2, evaluation_reps=1, confidence=0.9)
    assert first.configuration_hash != second.configuration_hash
    assert first.configuration_hash != third.configuration_hash


def test_repeated_policy_scalar_contracts_reject_noncoercible_and_fractional_values() -> (
    None
):
    """Exercise public scalar validation without beginning a policy search."""
    for kwargs in (
        {"partitions": "two"},
        {"evaluation_reps": float("inf")},
        {"confidence": "0.95"},
    ):
        with pytest.raises(ValueError):
            RepeatedLearnedPolicyPlan(**kwargs)

    scenario = make_static_binary_e_process_scenario(horizon=2)
    plan = RepeatedLearnedPolicyPlan(partitions=2, evaluation_reps=1)
    for seed in ("not-an-integer", 1.5):
        with pytest.raises(ValueError, match="seed"):
            evaluate_repeated_learned_policy(
                scenario,
                _search_config(),
                plan=plan,
                seed=seed,  # type: ignore[arg-type]
            )


def test_repeated_policy_normalizers_supply_fresh_default_contracts() -> None:
    """Default normalization is explicit and produces independently owned contracts."""
    normalized_plan = _normalized_plan(None)
    normalized_config = _normalized_search_config(None)
    second_plan = _normalized_plan(None)
    second_config = _normalized_search_config(None)

    assert normalized_plan == RepeatedLearnedPolicyPlan()
    assert normalized_plan == second_plan
    assert normalized_plan is not second_plan
    assert normalized_config == LearnedPolicyConfig()
    assert normalized_config == second_config
    assert normalized_config is not second_config


def test_repeated_policy_stream_proof_rejects_all_collision_modes() -> None:
    """Exercise the three independent stream-collision defenses with real seeds."""
    scenario = make_static_binary_e_process_scenario(horizon=2)
    training_branch, evaluation_branch = np.random.SeedSequence(313).spawn(2)
    training = _stream_record(
        scenario=scenario,
        master_seed=313,
        partition_index=0,
        role="training",
        branch=training_branch,
        n_reps=2,
    )
    evaluation = _stream_record(
        scenario=scenario,
        master_seed=313,
        partition_index=0,
        role="evaluation",
        branch=evaluation_branch,
        n_reps=2,
    )
    colliding_evaluation = _stream_record(
        scenario=scenario,
        master_seed=313,
        partition_index=0,
        role="evaluation",
        branch=training_branch,
        n_reps=2,
    )

    assert _validate_partition_streams(training, evaluation, set(), set()) == (
        training["trace_seed_schedule"],
        evaluation["trace_seed_schedule"],
    )
    with pytest.raises(RuntimeError, match="disjoint"):
        _validate_partition_streams(training, colliding_evaluation, set(), set())
    with pytest.raises(RuntimeError, match="root seeds"):
        _validate_partition_streams(
            training,
            evaluation,
            {int(training["root_seed"])},
            set(),
        )
    with pytest.raises(RuntimeError, match="SeedSequence branches"):
        _validate_partition_streams(
            training,
            evaluation,
            set(),
            {tuple(training["branch_spawn_key"])},
        )
    with pytest.raises(RuntimeError, match="collide"):
        _require_unique_trace_seeds([101, 202, 101])


def test_repeated_policy_pair_and_partition_bands_require_real_observations() -> None:
    """Empty pair inputs cannot be mistaken for a held-out policy comparison."""
    scenario = make_static_binary_e_process_scenario(horizon=2)
    with pytest.raises(ValueError, match="non-empty"):
        _paired_replication_summary(scenario, [], [], confidence=0.95)
    with pytest.raises(ValueError, match="at least two"):
        _partition_bands([], confidence=0.95)
