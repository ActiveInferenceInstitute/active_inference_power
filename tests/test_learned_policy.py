"""Real deterministic train/evaluation tests for learned policies."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from active_inference_power import (
    LearnedPolicyArtifact,
    LearnedPolicyConfig,
    evaluate_learned_policy,
    make_static_binary_e_process_scenario,
    train_tabular_policy,
)
from active_inference_power.scenario import PowerScenario


@pytest.fixture
def learned_artifact() -> tuple[PowerScenario, LearnedPolicyArtifact]:
    """Train one real, small policy artifact for contract-edge tests."""
    scenario = make_static_binary_e_process_scenario(horizon=2)
    artifact = train_tabular_policy(
        scenario,
        LearnedPolicyConfig(
            train_reps=1,
            candidate_parameters=((1.0, 0.5, 0.0),),
        ),
        seed=31,
    )
    return scenario, artifact


def test_learned_policy_is_frozen_hashed_and_held_out() -> None:
    scenario = make_static_binary_e_process_scenario(horizon=3)
    config = LearnedPolicyConfig(train_reps=3)
    artifact = train_tabular_policy(
        scenario, config, seed=41
    )
    evaluated = evaluate_learned_policy(scenario, artifact, 3, seed=42, workers=2)
    assert len(artifact.policy_hash) == 64
    assert artifact.search_config_hash == config.configuration_hash
    assert artifact.candidate_count == len(config.candidate_parameters)
    assert artifact.parameter_dimension == 3
    assert evaluated["search_config_hash"] == config.configuration_hash
    assert evaluated["policy_complexity"]["candidate_count"] == len(config.candidate_parameters)
    assert evaluated["policy_hash"] == artifact.policy_hash
    assert evaluated["train_seed"] != evaluated["evaluation_seed"]
    assert evaluated["validity"].startswith("conditional_held_out")
    with pytest.raises(ValueError):
        evaluate_learned_policy(scenario, artifact, 1, seed=41)


def test_learned_search_configuration_changes_provenance_hash() -> None:
    first = LearnedPolicyConfig(train_reps=2, cost_penalty=0.02)
    second = LearnedPolicyConfig(train_reps=2, cost_penalty=0.03)
    assert first.configuration_hash != second.configuration_hash


def test_learned_policy_rejects_wrong_scenario() -> None:
    first = make_static_binary_e_process_scenario(horizon=2)
    second = make_static_binary_e_process_scenario(horizon=3)
    artifact = train_tabular_policy(first, LearnedPolicyConfig(train_reps=1), seed=51)
    with pytest.raises(ValueError):
        evaluate_learned_policy(second, artifact, 1, seed=52)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"scenario_id": " "}, "scenario_id"),
        ({"scenario_hash": "A" * 64}, "scenario_hash"),
        ({"parameters": (1.0, 0.5)}, "parameters"),
        ({"parameters": (1.0, np.nan, 0.0)}, "parameters"),
        ({"train_seed": -1}, "train_seed"),
        ({"train_reps": 0}, "train_reps"),
        ({"objective": np.inf}, "objective"),
        ({"visible_features": ("action_cost",)}, "visible_features"),
        ({"search_config_hash": "short"}, "search_config_hash"),
        ({"candidate_count": 0}, "candidate_count"),
        ({"parameter_dimension": 2}, "parameter_dimension"),
    ),
)
def test_learned_policy_artifact_rejects_malformed_provenance_and_parameters(
    learned_artifact: tuple[PowerScenario, LearnedPolicyArtifact],
    changes: dict[str, object],
    message: str,
) -> None:
    _, artifact = learned_artifact
    with pytest.raises(ValueError, match=message):
        replace(artifact, **cast(Any, changes))


def test_learned_policy_revalidates_low_level_tampering_before_evaluation(
    learned_artifact: tuple[PowerScenario, LearnedPolicyArtifact],
) -> None:
    scenario, artifact = learned_artifact
    object.__setattr__(artifact, "candidate_count", 0)
    with pytest.raises(ValueError, match="candidate_count"):
        _ = artifact.policy_hash
    with pytest.raises(ValueError, match="candidate_count"):
        evaluate_learned_policy(scenario, artifact, 1, seed=32)


def test_learned_policy_evaluation_binds_identity_and_held_out_inputs(
    learned_artifact: tuple[PowerScenario, LearnedPolicyArtifact],
) -> None:
    scenario, artifact = learned_artifact
    wrong_identity = replace(artifact, scenario_id="different_scenario_id")
    with pytest.raises(ValueError, match="scenario id"):
        evaluate_learned_policy(scenario, wrong_identity, 1, seed=32)
    with pytest.raises(ValueError, match="n_reps"):
        evaluate_learned_policy(scenario, artifact, 0, seed=32)
    with pytest.raises(ValueError, match="evaluation seed"):
        evaluate_learned_policy(scenario, artifact, 1, seed=True)
