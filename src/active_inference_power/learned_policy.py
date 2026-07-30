"""Deterministic learned-policy evaluation with train/evaluation separation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, cast

import numpy as np

from .action_loop import run_action_loop
from .action_statistics import summarize_action_traces
from .scenario import PowerScenario


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VISIBLE_POLICY_FEATURES = (
    "expected_information_gain",
    "target_uncertainty",
    "action_cost",
)


def _positive_integer(value: object, name: str) -> int:
    """Return an integral, strictly positive public contract value."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        normalized = int(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if normalized < 1 or normalized != value:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _nonnegative_integer(value: object, name: str) -> int:
    """Return an integral, non-negative public contract value."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        normalized = int(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if normalized < 0 or normalized != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _finite_float(value: object, name: str) -> float:
    """Normalize one finite scalar while rejecting booleans and strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be finite numeric")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite numeric")
    return normalized


def _sha256(value: object, name: str) -> str:
    """Validate a canonical SHA-256 witness before using it as provenance."""
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_workers(workers: int) -> int:
    if isinstance(workers, bool) or int(workers) != workers or int(workers) < 1:
        raise ValueError("workers must be a positive integer")
    return int(workers)


def _learned_trace(
    scenario: PowerScenario, seed: int, parameters: tuple[float, float, float]
):
    return run_action_loop(
        scenario, "learned", seed=int(seed), learned_parameters=np.asarray(parameters)
    )


def _learned_traces(
    scenario: PowerScenario,
    seeds: list[int],
    parameters: tuple[float, float, float],
    workers: int,
):
    """Evaluate child streams in stable seed order, optionally concurrently."""
    if workers == 1:
        return [_learned_trace(scenario, seed, parameters) for seed in seeds]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda seed: _learned_trace(scenario, seed, parameters), seeds))


@dataclass(frozen=True)
class LearnedPolicyConfig:
    """Finite candidate policy search configuration.

    The candidate policy uses only posterior uncertainty, expected information
    gain, and visible action cost.  The evaluator may use hidden truth to score
    training replications, but never passes that truth to the policy.
    """

    train_reps: int = 64
    cost_penalty: float = 0.02
    candidate_parameters: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.5, 0.0),
        (1.0, 1.0, 0.05),
        (0.5, 1.0, 0.1),
        (1.5, 0.25, 0.2),
    )

    def __post_init__(self) -> None:
        if isinstance(self.train_reps, bool) or int(self.train_reps) != self.train_reps or self.train_reps < 1:
            raise ValueError("train_reps must be a positive integer")
        if not np.isfinite(float(self.cost_penalty)) or float(self.cost_penalty) < 0.0:
            raise ValueError("cost_penalty must be finite and non-negative")
        if not self.candidate_parameters:
            raise ValueError("candidate_parameters must not be empty")
        normalized = tuple(tuple(float(value) for value in candidate) for candidate in self.candidate_parameters)
        if any(len(candidate) != 3 or not np.all(np.isfinite(candidate)) for candidate in normalized):
            raise ValueError("each candidate policy must contain three finite parameters")
        object.__setattr__(self, "train_reps", int(self.train_reps))
        object.__setattr__(self, "candidate_parameters", normalized)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "train_reps": self.train_reps,
            "cost_penalty": self.cost_penalty,
            "candidate_parameters": [list(candidate) for candidate in self.candidate_parameters],
        }

    @property
    def configuration_hash(self) -> str:
        """Hash the complete finite search configuration.

        The selected parameter vector is not sufficient provenance: two
        different candidate spaces or training penalties can select the same
        vector.  The search hash therefore travels with the frozen artifact
        and contributes to its policy hash.
        """
        payload = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LearnedPolicyArtifact:
    """Frozen policy parameters and their independently auditable provenance.

    This is a fail-closed boundary rather than an unvalidated metadata
    container.  Evaluation accepts only artifacts that identify their exact
    scenario and finite candidate search, expose precisely the features the
    policy implementation uses, and contain the three finite parameters the
    learned action policy can execute.
    """

    scenario_id: str
    scenario_hash: str
    parameters: tuple[float, float, float]
    train_seed: int
    train_reps: int
    objective: float
    visible_features: tuple[str, ...] = (
        "expected_information_gain",
        "target_uncertainty",
        "action_cost",
    )
    search_config_hash: str = ""
    candidate_count: int = 0
    parameter_dimension: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id or self.scenario_id != self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty, trimmed string")
        try:
            raw_parameters = tuple(self.parameters)
        except TypeError as exc:
            raise ValueError("parameters must contain exactly three finite values") from exc
        parameters = tuple(
            _finite_float(value, f"parameters[{index}]")
            for index, value in enumerate(raw_parameters)
        )
        if len(parameters) != 3:
            raise ValueError("parameters must contain exactly three finite values")
        try:
            features = tuple(self.visible_features)
        except TypeError as exc:
            raise ValueError("visible_features must contain strings") from exc
        if any(not isinstance(feature, str) for feature in features):
            raise ValueError("visible_features must contain strings")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "train_seed", _nonnegative_integer(self.train_seed, "train_seed"))
        object.__setattr__(self, "train_reps", _positive_integer(self.train_reps, "train_reps"))
        object.__setattr__(self, "objective", _finite_float(self.objective, "objective"))
        object.__setattr__(self, "visible_features", features)
        object.__setattr__(self, "candidate_count", _positive_integer(self.candidate_count, "candidate_count"))
        object.__setattr__(self, "parameter_dimension", _positive_integer(self.parameter_dimension, "parameter_dimension"))
        self.validate()

    def validate(self) -> None:
        """Re-check this artifact before hashing or evaluating it.

        ``frozen=True`` prevents ordinary mutation, but callers can still
        deserialize or deliberately alter objects through low-level APIs.
        Revalidating here makes those paths fail before policy execution or
        before a new digest is emitted for malformed provenance.
        """
        if not isinstance(self.scenario_id, str) or not self.scenario_id or self.scenario_id != self.scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty, trimmed string")
        _sha256(self.scenario_hash, "scenario_hash")
        _sha256(self.search_config_hash, "search_config_hash")
        if not isinstance(self.parameters, tuple) or len(self.parameters) != 3:
            raise ValueError("parameters must contain exactly three finite values")
        for index, value in enumerate(self.parameters):
            _finite_float(value, f"parameters[{index}]")
        _nonnegative_integer(self.train_seed, "train_seed")
        _positive_integer(self.train_reps, "train_reps")
        _finite_float(self.objective, "objective")
        if self.visible_features != _VISIBLE_POLICY_FEATURES:
            raise ValueError(
                "visible_features must exactly describe the learned policy's visible inputs"
            )
        _positive_integer(self.candidate_count, "candidate_count")
        parameter_dimension = _positive_integer(
            self.parameter_dimension, "parameter_dimension"
        )
        if parameter_dimension != len(self.parameters):
            raise ValueError("parameter_dimension must match the frozen parameter vector")

    @property
    def policy_hash(self) -> str:
        self.validate()
        payload = json.dumps(self.to_mapping(include_hash=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_mapping(self, *, include_hash: bool = True) -> dict[str, Any]:
        self.validate()
        result = {
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "parameters": list(self.parameters),
            "train_seed": self.train_seed,
            "train_reps": self.train_reps,
            "objective": self.objective,
            "visible_features": list(self.visible_features),
            "search_config_hash": self.search_config_hash,
            "policy_complexity": {
                "search_type": "finite_tabular_candidate_search",
                "candidate_count": self.candidate_count,
                "parameter_dimension": self.parameter_dimension,
            },
        }
        if include_hash:
            result["policy_hash"] = self.policy_hash
        return result


def train_tabular_policy(
    scenario: PowerScenario,
    config: LearnedPolicyConfig | None = None,
    *,
    seed: int = 0,
    workers: int = 1,
) -> LearnedPolicyArtifact:
    """Select frozen parameters using independent training replications."""
    config = config or LearnedPolicyConfig()
    workers = _validate_workers(workers)
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    children = np.random.SeedSequence(int(seed)).spawn(config.train_reps)
    seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    best: tuple[float, tuple[float, float, float]] | None = None
    for candidate in config.candidate_parameters:
        traces = _learned_traces(scenario, seeds, candidate, workers)
        summary = summarize_action_traces(scenario, traces)
        objective = float(summary["power"] - config.cost_penalty * summary["mean_cost"])
        if best is None or objective > best[0]:
            best = (objective, candidate)
    if best is None:
        raise RuntimeError("learned policy search produced no candidate")
    parameters: tuple[float, float, float] = (
        float(best[1][0]), float(best[1][1]), float(best[1][2])
    )
    return LearnedPolicyArtifact(
        scenario.scenario_id,
        scenario.configuration_hash,
        parameters,
        int(seed),
        config.train_reps,
        float(best[0]),
        search_config_hash=config.configuration_hash,
        candidate_count=len(config.candidate_parameters),
        parameter_dimension=3,
    )


def evaluate_learned_policy(
    scenario: PowerScenario,
    artifact: LearnedPolicyArtifact,
    n_reps: int,
    *,
    seed: int = 1,
    workers: int = 1,
) -> dict[str, Any]:
    """Evaluate frozen policy parameters on fresh child streams."""
    artifact.validate()
    if artifact.scenario_id != scenario.scenario_id:
        raise ValueError("learned policy scenario id does not match evaluation scenario")
    if artifact.scenario_hash != scenario.configuration_hash:
        raise ValueError("learned policy scenario hash does not match evaluation scenario")
    workers = _validate_workers(workers)
    evaluation_seed = _nonnegative_integer(seed, "evaluation seed")
    evaluation_reps = _positive_integer(n_reps, "n_reps")
    if evaluation_seed == artifact.train_seed:
        raise ValueError("evaluation seed must differ from training seed")
    children = np.random.SeedSequence(evaluation_seed).spawn(evaluation_reps)
    seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    traces = _learned_traces(scenario, seeds, artifact.parameters, workers)
    summary = summarize_action_traces(scenario, traces)
    return {
        "policy": "learned",
        "policy_hash": artifact.policy_hash,
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "train_seed": artifact.train_seed,
        "evaluation_seed": evaluation_seed,
        "train_reps": artifact.train_reps,
        "evaluation_reps": evaluation_reps,
        "search_config_hash": artifact.search_config_hash,
        "policy_complexity": {
            "search_type": "finite_tabular_candidate_search",
            "candidate_count": artifact.candidate_count,
            "parameter_dimension": artifact.parameter_dimension,
        },
        "summary": summary,
        "seed_schedule": seeds,
        "validity": "conditional_held_out_policy_operating_characteristic",
        "claim_boundary": "Frozen policy evaluation is conditional on the declared scenario and train/evaluation split; it is not a policy optimality theorem.",
    }


__all__ = ["LearnedPolicyConfig", "LearnedPolicyArtifact", "train_tabular_policy", "evaluate_learned_policy"]
