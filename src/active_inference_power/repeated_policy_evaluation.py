"""Repeated, held-out evaluation for finite learned-policy searches.

One train/evaluation split is useful for inspecting a frozen policy, but it can
be unusually favourable or unfavourable to that one training sample.  This
module repeats the entire train/freeze/evaluate procedure on independent
``SeedSequence`` branches.  Its uncertainty bands use one paired held-out mean
per independently trained partition, rather than pooling replications across
different learned policies as if their fitted parameters were fixed.

The resulting quantities remain conditional simulation operating
characteristics.  In particular, a positive learned-minus-reference contrast
does not establish policy optimality, a universal utility ordering, or
validity beyond the declared scenario and finite candidate search.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Sequence, cast

import numpy as np

from .action_loop import ActionLoopTrace, ActionPolicy, run_action_loop
from .action_statistics import metrics_for_trace, summarize_action_traces
from .learned_policy import LearnedPolicyArtifact, LearnedPolicyConfig, train_tabular_policy
from .scenario import PowerScenario
from .uncertainty import paired_difference_summary, paired_numeric_difference_summary


_REFERENCE_POLICIES = frozenset(
    {
        "fixed_horizon",
        "posterior_threshold",
        "e_process",
        "information_gain",
        "cost_aware",
        "adaptive_target",
        "posterior_sampling",
    }
)


def _hash_mapping(value: Any) -> str:
    """Return a canonical SHA-256 witness for JSON-safe provenance data."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _positive_integer(value: object, name: str) -> int:
    """Normalize one strictly positive public contract value."""
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
    """Normalize one non-negative public contract value."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        normalized = int(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if normalized < 0 or normalized != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _confidence(value: object) -> float:
    """Validate a finite, open-unit-interval confidence level."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("confidence must lie in (0, 1)")
    confidence = float(value)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    return confidence


@dataclass(frozen=True)
class RepeatedLearnedPolicyPlan:
    """Predeclared repeated train/freeze/evaluate design.

    ``partitions`` is deliberately at least two: the final bands summarize
    independently trained policy partitions, so a single split is not a
    repeated-policy estimate.  The reference policy is fixed before execution
    and receives the same held-out action-loop seeds as the learned policy in
    each partition.
    """

    partitions: int = 4
    evaluation_reps: int = 32
    reference_policy: ActionPolicy = "cost_aware"
    confidence: float = 0.95

    def __post_init__(self) -> None:
        partitions = _positive_integer(self.partitions, "partitions")
        if partitions < 2:
            raise ValueError("partitions must be at least 2 for repeated-policy bands")
        evaluation_reps = _positive_integer(self.evaluation_reps, "evaluation_reps")
        reference_policy = str(self.reference_policy)
        if reference_policy not in _REFERENCE_POLICIES:
            raise ValueError("reference_policy must be a supported non-learned action policy")
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "evaluation_reps", evaluation_reps)
        object.__setattr__(self, "reference_policy", cast(ActionPolicy, reference_policy))
        object.__setattr__(self, "confidence", _confidence(self.confidence))

    def to_mapping(self) -> dict[str, Any]:
        """Return the serializable repeated-evaluation contract."""
        return {
            "partitions": self.partitions,
            "evaluation_reps": self.evaluation_reps,
            "reference_policy": self.reference_policy,
            "confidence": self.confidence,
        }

    @property
    def configuration_hash(self) -> str:
        """Return the immutable identity of the repeated-evaluation design."""
        return _hash_mapping(self.to_mapping())


def _normalized_plan(plan: RepeatedLearnedPolicyPlan | None) -> RepeatedLearnedPolicyPlan:
    """Revalidate a possibly low-level-mutated frozen plan before execution."""
    if plan is None:
        return RepeatedLearnedPolicyPlan()
    if not isinstance(plan, RepeatedLearnedPolicyPlan):
        raise ValueError("plan must be a RepeatedLearnedPolicyPlan")
    return RepeatedLearnedPolicyPlan(
        partitions=plan.partitions,
        evaluation_reps=plan.evaluation_reps,
        reference_policy=plan.reference_policy,
        confidence=plan.confidence,
    )


def _normalized_search_config(config: LearnedPolicyConfig | None) -> LearnedPolicyConfig:
    """Reconstruct the finite search configuration before it enters a run."""
    if config is None:
        return LearnedPolicyConfig()
    if not isinstance(config, LearnedPolicyConfig):
        raise ValueError("config must be a LearnedPolicyConfig")
    return LearnedPolicyConfig(
        train_reps=config.train_reps,
        cost_penalty=config.cost_penalty,
        candidate_parameters=config.candidate_parameters,
    )


def _seed_from_branch(branch: np.random.SeedSequence) -> int:
    """Derive a wide integer root seed without platform-dependent byte order."""
    words = branch.generate_state(4, dtype=np.uint32)
    return sum(int(word) << (32 * index) for index, word in enumerate(words))


def _trace_seed_schedule(root_seed: int, n_reps: int) -> list[int]:
    """Materialize the exact action-loop seeds used by one root seed."""
    children = np.random.SeedSequence(root_seed).spawn(n_reps)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def _stream_record(
    *,
    scenario: PowerScenario,
    master_seed: int,
    partition_index: int,
    role: str,
    branch: np.random.SeedSequence,
    n_reps: int,
) -> dict[str, Any]:
    """Create an auditable root branch and its materialized trace schedule."""
    root_seed = _seed_from_branch(branch)
    trace_seed_schedule = _trace_seed_schedule(root_seed, n_reps)
    payload = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "master_seed": master_seed,
        "partition_index": partition_index,
        "role": role,
        "branch_spawn_key": [int(value) for value in branch.spawn_key],
        "root_seed": root_seed,
        "n_reps": n_reps,
        "trace_seed_schedule": trace_seed_schedule,
        "derivation": (
            "SeedSequence(master_seed).spawn(2 * partitions) sibling branch, "
            "then SeedSequence(root_seed).spawn(n_reps) in ascending replication order"
        ),
    }
    return {**payload, "stream_hash": _hash_mapping(payload)}


def _validate_partition_streams(
    training: dict[str, Any],
    evaluation: dict[str, Any],
    prior_root_seeds: set[int],
    prior_spawn_keys: set[tuple[int, ...]],
) -> tuple[list[int], list[int]]:
    """Fail closed if either branch overlaps another materialized partition."""
    training_schedule = [int(value) for value in training["trace_seed_schedule"]]
    evaluation_schedule = [int(value) for value in evaluation["trace_seed_schedule"]]
    overlap = sorted(set(training_schedule).intersection(evaluation_schedule))
    training_key = tuple(int(value) for value in training["branch_spawn_key"])
    evaluation_key = tuple(int(value) for value in evaluation["branch_spawn_key"])
    training_root = int(training["root_seed"])
    evaluation_root = int(evaluation["root_seed"])
    if training_root == evaluation_root or training_key == evaluation_key or overlap:
        raise RuntimeError("training and evaluation streams must be disjoint")
    if training_root in prior_root_seeds or evaluation_root in prior_root_seeds:
        raise RuntimeError("partition root seeds must be unique")
    if training_key in prior_spawn_keys or evaluation_key in prior_spawn_keys:
        raise RuntimeError("partition SeedSequence branches must be unique")
    return training_schedule, evaluation_schedule


def _require_unique_trace_seeds(trace_seeds: Sequence[int]) -> None:
    """Reject a materialized schedule that could reuse an action-loop stream."""
    if len(set(trace_seeds)) != len(trace_seeds):
        raise RuntimeError(
            "materialized action-loop seeds collide across repeated partitions; choose another seed"
        )


def _build_stream_proof(
    scenario: PowerScenario,
    search_config: LearnedPolicyConfig,
    plan: RepeatedLearnedPolicyPlan,
    master_seed: int,
) -> dict[str, Any]:
    """Materialize and fail-close-check all independent partition streams."""
    branches = np.random.SeedSequence(master_seed).spawn(2 * plan.partitions)
    partitions: list[dict[str, Any]] = []
    all_trace_seeds: list[int] = []
    root_seeds: set[int] = set()
    spawn_keys: set[tuple[int, ...]] = set()

    for partition_index in range(plan.partitions):
        training = _stream_record(
            scenario=scenario,
            master_seed=master_seed,
            partition_index=partition_index,
            role="training",
            branch=branches[2 * partition_index],
            n_reps=search_config.train_reps,
        )
        evaluation = _stream_record(
            scenario=scenario,
            master_seed=master_seed,
            partition_index=partition_index,
            role="evaluation",
            branch=branches[2 * partition_index + 1],
            n_reps=plan.evaluation_reps,
        )
        training_schedule, evaluation_schedule = _validate_partition_streams(
            training, evaluation, root_seeds, spawn_keys
        )
        overlap = sorted(set(training_schedule).intersection(evaluation_schedule))
        training_key = tuple(int(value) for value in training["branch_spawn_key"])
        evaluation_key = tuple(int(value) for value in evaluation["branch_spawn_key"])
        training_root = int(training["root_seed"])
        evaluation_root = int(evaluation["root_seed"])
        root_seeds.update((training_root, evaluation_root))
        spawn_keys.update((training_key, evaluation_key))
        all_trace_seeds.extend(training_schedule)
        all_trace_seeds.extend(evaluation_schedule)
        partitions.append(
            {
                "partition_index": partition_index,
                "training": training,
                "evaluation": evaluation,
                "trace_seed_overlap": overlap,
                "streams_disjoint": True,
            }
        )

    _require_unique_trace_seeds(all_trace_seeds)
    payload = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "master_seed": master_seed,
        "search_config_hash": search_config.configuration_hash,
        "plan_hash": plan.configuration_hash,
        "derivation": (
            "Each partition receives separate sibling SeedSequence branches for training "
            "and held-out evaluation; all materialized action-loop seeds are checked "
            "for cross-partition uniqueness before any policy is trained."
        ),
        "partitions": partitions,
        "all_materialized_trace_seed_schedule_hash": _hash_mapping(all_trace_seeds),
    }
    return {
        **payload,
        "all_materialized_trace_seeds_unique": True,
        "stream_proof_hash": _hash_mapping(payload),
    }


def _run_traces(
    seeds: Sequence[int],
    workers: int,
    run_one: Callable[[int], ActionLoopTrace],
) -> list[ActionLoopTrace]:
    """Execute a fixed seed order with an optional, order-preserving worker pool."""
    if workers == 1:
        return [run_one(int(seed)) for seed in seeds]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run_one, (int(seed) for seed in seeds)))


def _learned_traces(
    scenario: PowerScenario,
    artifact: LearnedPolicyArtifact,
    seeds: Sequence[int],
    workers: int,
) -> list[ActionLoopTrace]:
    """Run one frozen learned policy on its held-out partition schedule."""
    artifact.validate()

    def run_one(seed: int) -> ActionLoopTrace:
        return run_action_loop(
            scenario,
            "learned",
            seed=seed,
            learned_parameters=np.asarray(artifact.parameters, dtype=float),
        )

    return _run_traces(seeds, workers, run_one)


def _reference_traces(
    scenario: PowerScenario,
    reference_policy: ActionPolicy,
    seeds: Sequence[int],
    workers: int,
) -> list[ActionLoopTrace]:
    """Run the fixed reference policy on the learned policy's held-out seeds."""

    def run_one(seed: int) -> ActionLoopTrace:
        return run_action_loop(scenario, reference_policy, seed=seed)

    return _run_traces(seeds, workers, run_one)


def _paired_replication_summary(
    scenario: PowerScenario,
    reference_traces: Sequence[ActionLoopTrace],
    learned_traces: Sequence[ActionLoopTrace],
    confidence: float,
) -> dict[str, Any]:
    """Summarize within-partition held-out learned-minus-reference contrasts."""
    if len(reference_traces) != len(learned_traces) or not reference_traces:
        raise ValueError("paired held-out traces must be non-empty and have equal length")
    reference_metrics = [metrics_for_trace(scenario, trace) for trace in reference_traces]
    learned_metrics = [metrics_for_trace(scenario, trace) for trace in learned_traces]
    return {
        "direction": "learned_minus_reference",
        "n_reps": len(learned_metrics),
        "power": paired_difference_summary(
            [metric.true_positive_rate for metric in reference_metrics],
            [metric.true_positive_rate for metric in learned_metrics],
            confidence,
        ),
        "cost": paired_numeric_difference_summary(
            [metric.total_cost for metric in reference_metrics],
            [metric.total_cost for metric in learned_metrics],
            confidence,
        ),
        "stopping": paired_numeric_difference_summary(
            [metric.step_count for metric in reference_metrics],
            [metric.step_count for metric in learned_metrics],
            confidence,
        ),
        "claim_boundary": (
            "Common held-out seeds pair environment random-number roots by replication, "
            "but action-dependent policies need not observe identical trajectories."
        ),
    }


def _learned_evaluation_mapping(
    scenario: PowerScenario,
    artifact: LearnedPolicyArtifact,
    traces: Sequence[ActionLoopTrace],
    stream: dict[str, Any],
) -> dict[str, Any]:
    """Serialize a held-out frozen-policy result in the existing evaluation shape."""
    return {
        "policy": "learned",
        "policy_hash": artifact.policy_hash,
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "train_seed": artifact.train_seed,
        "evaluation_seed": int(stream["root_seed"]),
        "train_reps": artifact.train_reps,
        "evaluation_reps": len(traces),
        "search_config_hash": artifact.search_config_hash,
        "policy_complexity": {
            "search_type": "finite_tabular_candidate_search",
            "candidate_count": artifact.candidate_count,
            "parameter_dimension": artifact.parameter_dimension,
        },
        "summary": summarize_action_traces(scenario, traces),
        "seed_schedule": [int(value) for value in stream["trace_seed_schedule"]],
        "evaluation_stream_hash": str(stream["stream_hash"]),
        "validity": "conditional_held_out_policy_operating_characteristic",
        "claim_boundary": (
            "Frozen policy evaluation is conditional on the declared scenario and "
            "independent train/evaluation partition; it is not a policy optimality theorem."
        ),
    }


def _reference_evaluation_mapping(
    scenario: PowerScenario,
    reference_policy: ActionPolicy,
    traces: Sequence[ActionLoopTrace],
    stream: dict[str, Any],
) -> dict[str, Any]:
    """Serialize the fixed comparator without elevating it to an optimal baseline."""
    return {
        "policy": reference_policy,
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "evaluation_seed": int(stream["root_seed"]),
        "evaluation_reps": len(traces),
        "summary": summarize_action_traces(scenario, traces),
        "seed_schedule": [int(value) for value in stream["trace_seed_schedule"]],
        "evaluation_stream_hash": str(stream["stream_hash"]),
        "validity_boundaries": sorted({trace.validity_boundary for trace in traces}),
        "claim_boundary": (
            "The reference is a fixed diagnostic comparator on the same held-out "
            "streams, not an optimal-policy benchmark."
        ),
    }


def _partition_record(
    scenario: PowerScenario,
    search_config: LearnedPolicyConfig,
    plan: RepeatedLearnedPolicyPlan,
    stream_partition: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    """Train once, freeze once, and evaluate both policies on one held-out split."""
    training_stream = cast(dict[str, Any], stream_partition["training"])
    evaluation_stream = cast(dict[str, Any], stream_partition["evaluation"])
    artifact = train_tabular_policy(
        scenario,
        search_config,
        seed=int(training_stream["root_seed"]),
        workers=workers,
    )
    held_out_seeds = [int(value) for value in evaluation_stream["trace_seed_schedule"]]
    learned_traces = _learned_traces(scenario, artifact, held_out_seeds, workers)
    reference_traces = _reference_traces(
        scenario, plan.reference_policy, held_out_seeds, workers
    )
    paired = _paired_replication_summary(
        scenario, reference_traces, learned_traces, plan.confidence
    )
    learned_evaluation = _learned_evaluation_mapping(
        scenario, artifact, learned_traces, evaluation_stream
    )
    reference_evaluation = _reference_evaluation_mapping(
        scenario, plan.reference_policy, reference_traces, evaluation_stream
    )
    return {
        "partition_index": int(stream_partition["partition_index"]),
        "stream_proof": stream_partition,
        "artifact": artifact.to_mapping(),
        "evaluation": learned_evaluation,
        "reference_evaluation": reference_evaluation,
        "paired_held_out_differences": paired,
        "partition_mean_differences": {
            metric: float(cast(dict[str, Any], paired[metric])["mean_difference"])
            for metric in ("power", "cost", "stopping")
        },
    }


def _partition_bands(
    records: Sequence[dict[str, Any]], confidence: float
) -> dict[str, Any]:
    """Band paired means over independently trained policy partitions."""
    if len(records) < 2:
        raise ValueError("at least two partition records are required for repeated-policy bands")
    metric_fields = {
        "power": "power",
        "cost": "mean_cost",
        "stopping": "mean_stopping_time",
    }
    reference_values: dict[str, list[float]] = {}
    learned_values: dict[str, list[float]] = {}
    for output_name, summary_field in metric_fields.items():
        reference_values[output_name] = [
            float(record["reference_evaluation"]["summary"][summary_field])
            for record in records
        ]
        learned_values[output_name] = [
            float(record["evaluation"]["summary"][summary_field]) for record in records
        ]
    return {
        "direction": "learned_minus_reference",
        "confidence": confidence,
        "unit": "one paired held-out mean from one independently trained partition",
        "aggregation": (
            "Bands use partition-level paired means and do not pool held-out "
            "replications across differently trained frozen policies."
        ),
        "power": paired_difference_summary(
            reference_values["power"], learned_values["power"], confidence
        ),
        "cost": paired_numeric_difference_summary(
            reference_values["cost"], learned_values["cost"], confidence
        ),
        "stopping": paired_numeric_difference_summary(
            reference_values["stopping"], learned_values["stopping"], confidence
        ),
        "partition_mean_values": {
            "reference": reference_values,
            "learned": learned_values,
        },
    }


def evaluate_repeated_learned_policy(
    scenario: PowerScenario,
    config: LearnedPolicyConfig | None = None,
    *,
    plan: RepeatedLearnedPolicyPlan | None = None,
    seed: int = 0,
    workers: int = 1,
) -> dict[str, Any]:
    """Repeat deterministic train/freeze/evaluate policy partitions.

    Each partition receives distinct training and held-out branches.  The
    learned policy and the predeclared fixed reference policy share a held-out
    replication schedule within that partition.  Parallel workers only change
    execution placement: child seed order, frozen artifacts, and the returned
    mapping remain identical.
    """
    repeated_plan = _normalized_plan(plan)
    search_config = _normalized_search_config(config)
    master_seed = _nonnegative_integer(seed, "seed")
    worker_count = _positive_integer(workers, "workers")
    stream_proof = _build_stream_proof(
        scenario, search_config, repeated_plan, master_seed
    )
    records = [
        _partition_record(
            scenario,
            search_config,
            repeated_plan,
            cast(dict[str, Any], stream_partition),
            worker_count,
        )
        for stream_partition in cast(list[dict[str, Any]], stream_proof["partitions"])
    ]
    return {
        "contract_version": "1.0",
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "seed": master_seed,
        "plan": repeated_plan.to_mapping(),
        "plan_hash": repeated_plan.configuration_hash,
        "search_config": search_config.to_mapping(),
        "search_config_hash": search_config.configuration_hash,
        "candidate_space": {
            "search_type": "finite_tabular_candidate_search",
            "candidate_count": len(search_config.candidate_parameters),
            "parameter_dimension": 3,
            "train_reps_per_partition": search_config.train_reps,
        },
        "stream_proof": stream_proof,
        "partitions": records,
        "frozen_policy_hashes": [
            str(record["artifact"]["policy_hash"]) for record in records
        ],
        "paired_held_out_bands": _partition_bands(records, repeated_plan.confidence),
        "worker_invariance": (
            "Stable child-stream schedules and ordered reductions make this mapping "
            "invariant to the requested worker count."
        ),
        "validity": "conditional_repeated_held_out_policy_operating_characteristic",
        "claim_boundary": (
            "Repeated frozen-policy evaluation is conditional on the declared scenario, "
            "finite candidate search, independent train/evaluation partitions, and "
            "finite simulation. It does not establish policy optimality, universal "
            "performance, causal benefit, or family-level validity."
        ),
    }


__all__ = ["RepeatedLearnedPolicyPlan", "evaluate_repeated_learned_policy"]
