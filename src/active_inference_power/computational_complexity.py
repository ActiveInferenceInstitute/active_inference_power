"""Source-level computational-work accounting and timing-profile contracts.

The release artifact reports only deterministic, version-specific workload
proxies.  Those proxies make the declared input shape and dominant loops
auditable, but they are not CPU-instruction counts, elapsed-time measurements,
or claims about dependency internals.  Repeated wall-clock measurements belong
to the separate, explicitly non-promotional timing experiment lane.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


COMPUTATIONAL_COMPLEXITY_CONTRACT_VERSION = "1.0"
TIMING_PROFILE_CONTRACT_VERSION = "1.0"
TIMING_PROFILE_STATUS = (
    "research_only_local_runtime_diagnostic_not_for_promotion_or_release"
)
TIMING_PROFILE_FROZEN_PLAN_SOURCE = (
    "manuscript/config.yaml#experiment.computational_complexity"
)
TIMING_PROFILE_PROGRAMMATIC_PLAN_SOURCE = "explicit_programmatic_real_diagnostic_plan"
TIMING_PROFILE_SOURCE_NAMES = (
    "orchestrator",
    "computational_complexity",
    "corrections",
    "online_fdr",
    "sequential",
    "action_loop",
    "scenario",
    "pymdp_adapter",
)
TIMING_PROFILE_MEASUREMENT_PROTOCOL = {
    "input_construction": "deterministic inputs are constructed outside the timed interval",
    "warmups": "declared warmups run before and are excluded from raw samples",
    "repetitions": "every raw sample calls the real source-owned kernel once",
    "clocks": ["time.perf_counter_ns", "time.process_time_ns"],
    "uncertainty": "quartiles describe observed local-run variation; they are not Monte-Carlo uncertainty or an MCSE",
}
_TIMING_ROW_KEYS = {
    "kernel",
    "input_size",
    "condition",
    "warmups",
    "wall_time_ns_samples",
    "process_time_ns_samples",
    "wall_time_ns_summary",
    "process_time_ns_summary",
}
_TIMING_CORRECTION_CONDITION = "deterministic_pvalue_vector"
_TIMING_ONLINE_CONDITIONS = (
    "sparse_or_no_rejection_history",
    "dense_rejection_history",
)
_TIMING_SEQUENTIAL_CASES = (
    ("sequential:binary_e_process", "likelihood_ratio_evidence_path"),
    (
        "sequential:bernoulli_confidence_sequence",
        "time_uniform_confidence_sequence_inversion",
    ),
    (
        "sequential:fixed_horizon_binomial_pvalue",
        "fixed_horizon_only_invalid_for_optional_stopping",
    ),
)
_TIMING_ACTION_CONDITION = "real_pymdp_action_observation_belief_evidence_trace"
_TIMING_ENVIRONMENT_KEYS = {
    "python",
    "runtime",
    "libraries",
    "thread_environment",
    "redaction",
}
_TIMING_RUNTIME_KEYS = {"system", "release", "machine"}
_TIMING_PYTHON_KEYS = {"implementation", "version"}
_TIMING_LIBRARY_KEYS = {
    "active_inference_power",
    "numpy",
    "scipy",
    "matplotlib",
    "inferactively_pymdp",
}
_TIMING_THREAD_ENVIRONMENT_KEYS = {
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
}
_TIMING_REDACTION_TERMS = (
    "hostname",
    "username",
    "home path",
    "current working directory",
    "process identifier",
    "full environment",
)
_FORBIDDEN_TIMING_IDENTITY_KEYS = frozenset(
    {"hostname", "host", "username", "user", "home", "cwd", "pid"}
)

_CORRECTION_METHODS = frozenset({"bonferroni", "benjamini_hochberg"})
_ONLINE_METHODS = frozenset({"lord_plus_plus", "saffron", "e_lond"})
_ACTION_POLICIES = frozenset(
    {
        "fixed_horizon",
        "information_gain",
        "cost_aware",
        "e_process",
    }
)
_COMPUTATIONAL_COMPLEXITY_PLAN_KEYS = frozenset(
    {
        "hypothesis_grid",
        "stream_horizon_grid",
        "action_horizon_grid",
        "timing_warmups",
        "timing_repetitions",
        "correction_methods",
        "online_methods",
        "action_policies",
    }
)
_FORBIDDEN_RELEASE_TIMING_TERMS = frozenset(
    {
        "elapsed",
        "elapsed_ns",
        "elapsed_seconds",
        "wall_time",
        "wall_time_ns",
        "cpu_time",
        "cpu_time_ns",
        "perf_counter",
        "duration",
        "duration_ns",
        "timing_samples",
    }
)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _strictly_increasing_positive_sequence(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence of positive integers")
    values = tuple(
        _positive_integer(item, f"{name}[{index}]") for index, item in enumerate(value)
    )
    if not values or any(
        later <= earlier for earlier, later in zip(values, values[1:])
    ):
        raise ValueError(f"{name} must be strictly increasing and non-empty")
    return values


def _ordered_choice_sequence(
    value: Any, name: str, allowed: frozenset[str]
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence")
    values = tuple(str(item) for item in value)
    if (
        not values
        or len(set(values)) != len(values)
        or any(item not in allowed for item in values)
    ):
        raise ValueError(
            f"{name} must be a non-empty ordered subset of {sorted(allowed)!r}"
        )
    return values


@dataclass(frozen=True)
class ComputationalComplexityPlan:
    """Frozen input grids and repetitions for workload and timing inspection.

    The plan is declared in the experiment configuration.  The release uses
    the grids for deterministic work curves; the isolated timing runner uses
    the same plan for its local, repeated diagnostic measurements.
    """

    hypothesis_grid: tuple[int, ...]
    stream_horizon_grid: tuple[int, ...]
    action_horizon_grid: tuple[int, ...]
    timing_warmups: int
    timing_repetitions: int
    correction_methods: tuple[str, ...]
    online_methods: tuple[str, ...]
    action_policies: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hypothesis_grid",
            _strictly_increasing_positive_sequence(
                self.hypothesis_grid, "hypothesis_grid"
            ),
        )
        object.__setattr__(
            self,
            "stream_horizon_grid",
            _strictly_increasing_positive_sequence(
                self.stream_horizon_grid, "stream_horizon_grid"
            ),
        )
        object.__setattr__(
            self,
            "action_horizon_grid",
            _strictly_increasing_positive_sequence(
                self.action_horizon_grid, "action_horizon_grid"
            ),
        )
        object.__setattr__(
            self,
            "timing_warmups",
            _nonnegative_integer(self.timing_warmups, "timing_warmups"),
        )
        repetitions = _positive_integer(self.timing_repetitions, "timing_repetitions")
        if repetitions < 7:
            raise ValueError("timing_repetitions must be at least seven")
        object.__setattr__(self, "timing_repetitions", repetitions)
        object.__setattr__(
            self,
            "correction_methods",
            _ordered_choice_sequence(
                self.correction_methods, "correction_methods", _CORRECTION_METHODS
            ),
        )
        object.__setattr__(
            self,
            "online_methods",
            _ordered_choice_sequence(
                self.online_methods, "online_methods", _ONLINE_METHODS
            ),
        )
        object.__setattr__(
            self,
            "action_policies",
            _ordered_choice_sequence(
                self.action_policies, "action_policies", _ACTION_POLICIES
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "hypothesis_grid": list(self.hypothesis_grid),
            "stream_horizon_grid": list(self.stream_horizon_grid),
            "action_horizon_grid": list(self.action_horizon_grid),
            "timing_warmups": self.timing_warmups,
            "timing_repetitions": self.timing_repetitions,
            "correction_methods": list(self.correction_methods),
            "online_methods": list(self.online_methods),
            "action_policies": list(self.action_policies),
        }


def complexity_plan_from_mapping(
    value: Mapping[str, Any],
) -> ComputationalComplexityPlan:
    """Parse the configuration-owned complexity/timing plan fail-closed."""
    if not isinstance(value, Mapping):
        raise ValueError("computational_complexity must be a mapping")
    missing = sorted(_COMPUTATIONAL_COMPLEXITY_PLAN_KEYS - set(value))
    if missing:
        raise ValueError(f"computational_complexity is missing keys: {missing}")
    unexpected = sorted(set(value) - _COMPUTATIONAL_COMPLEXITY_PLAN_KEYS)
    if unexpected:
        raise ValueError(f"computational_complexity has unexpected keys: {unexpected}")
    return ComputationalComplexityPlan(
        hypothesis_grid=_strictly_increasing_positive_sequence(
            value["hypothesis_grid"], "computational_complexity.hypothesis_grid"
        ),
        stream_horizon_grid=_strictly_increasing_positive_sequence(
            value["stream_horizon_grid"], "computational_complexity.stream_horizon_grid"
        ),
        action_horizon_grid=_strictly_increasing_positive_sequence(
            value["action_horizon_grid"], "computational_complexity.action_horizon_grid"
        ),
        timing_warmups=_nonnegative_integer(
            value["timing_warmups"], "computational_complexity.timing_warmups"
        ),
        timing_repetitions=_positive_integer(
            value["timing_repetitions"], "computational_complexity.timing_repetitions"
        ),
        correction_methods=_ordered_choice_sequence(
            value["correction_methods"],
            "computational_complexity.correction_methods",
            _CORRECTION_METHODS,
        ),
        online_methods=_ordered_choice_sequence(
            value["online_methods"],
            "computational_complexity.online_methods",
            _ONLINE_METHODS,
        ),
        action_policies=_ordered_choice_sequence(
            value["action_policies"],
            "computational_complexity.action_policies",
            _ACTION_POLICIES,
        ),
    )


def linear_work(size: int) -> int:
    """Return the source-level linear-work proxy for a one-vector pass."""
    return _positive_integer(size, "size")


def rank_work(size: int) -> int:
    """Return a sort-dominant source-work proxy for rank-based corrections."""
    normalized = _positive_integer(size, "size")
    return normalized * max(1, math.ceil(math.log2(normalized)))


def lord_history_work_upper(horizon: int) -> int:
    """Return a conservative combined proxy for the LORD++ source path.

    The recurrence scans prior rejection history while emitting a trace, then
    ``OnlineFDRResult`` replays the serialized trace to validate the declared
    levels. The validator currently loops over the final rejection list at
    each time point (zero-index terms simply contribute no gamma mass), so a
    single ``T^2`` allowance would understate the combined source path. This
    deliberately conservative ``2 T (T + 1)`` proxy reserves one full
    horizon-by-history pass plus linear updates for each phase. It is not a
    count of CPU instructions or a complexity statement about LORD++ itself.
    """
    normalized = _positive_integer(horizon, "horizon")
    return 2 * normalized * (normalized + 1)


def saffron_sparse_history_work_upper(horizon: int) -> int:
    """Return the no-rejection upper proxy for SAFFRON candidate scans."""
    normalized = _positive_integer(horizon, "horizon")
    return normalized + normalized * (normalized - 1)


def saffron_dense_history_work_upper(horizon: int) -> int:
    """Return a dense-history upper proxy for the current SAFFRON code path.

    In addition to a candidate-prefix scan on every step, each prior rejection
    triggers a candidate-slice count.  With dense history, this creates a
    cubic upper proxy in the current implementation and its validation path.
    """
    normalized = _positive_integer(horizon, "horizon")
    triangular = normalized * (normalized - 1) // 2
    squares = normalized * (normalized - 1) * (2 * normalized - 1) // 6
    return normalized + 2 * (triangular + squares)


def _curve(
    name: str, values: Sequence[int], function, *, time_order: str
) -> dict[str, Any]:
    scale = [int(value) for value in values]
    return {
        "name": name,
        "input_values": scale,
        "work_proxy_values": [int(function(value)) for value in scale],
        "time_order": time_order,
        "unit": "dimensionless source-work proxy",
    }


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _suite_replications(
    value: Any,
    name: str,
    *,
    keys: tuple[str, ...] = ("n_reps",),
) -> int:
    """Read one declared replication count without inventing a common key.

    Most suite summaries label independent replications ``n_reps``.  The
    held-out learned-policy evaluator deliberately uses ``evaluation_reps``
    to keep training and evaluation partitions distinct.  The execution
    ledger therefore names the accepted producer field explicitly rather than
    coercing its schema into the more common label.
    """
    block = _require_mapping(value, name)
    for key in keys:
        if key in block:
            return _positive_integer(block[key], f"{name}.{key}")
    expected = " or ".join(repr(key) for key in keys)
    raise ValueError(f"{name} must declare {expected}")


def action_workload_ledger(
    suite: Mapping[str, Any],
    *,
    action_horizon: int,
    split_confirmation_horizon: int,
) -> dict[str, Any]:
    """Count declared action-loop and confirmation execution paths exactly.

    These counts are kept separate from scientific sample-size accounting:
    they enumerate code paths and maximum allowed action steps, while the
    latter names independent replication units for statistical precision.
    """
    action_horizon = _positive_integer(action_horizon, "action_horizon")
    split_confirmation_horizon = _positive_integer(
        split_confirmation_horizon, "split_confirmation_horizon"
    )
    operating = suite.get("operating_characteristics")
    if not isinstance(operating, list) or not operating:
        raise ValueError("suite.operating_characteristics must be a non-empty list")
    surface = sum(
        _suite_replications(item, f"suite.operating_characteristics[{index}]")
        for index, item in enumerate(operating)
    )

    paired = suite.get("paired_policy_comparisons")
    if not isinstance(paired, list) or not paired:
        raise ValueError("suite.paired_policy_comparisons must be a non-empty list")
    paired_traces = 0
    for index, item in enumerate(paired):
        record = _require_mapping(item, f"suite.paired_policy_comparisons[{index}]")
        policies = record.get("policies")
        if not isinstance(policies, list) or not policies:
            raise ValueError("paired policy comparison policies must be non-empty")
        paired_traces += _positive_integer(
            record.get("n_reps"), "paired policy comparison n_reps"
        ) * len(policies)

    online = _suite_replications(suite.get("online_fdr"), "suite.online_fdr")
    calibration = _suite_replications(
        suite.get("conditional_selection_calibration"),
        "suite.conditional_selection_calibration",
    )
    future = _suite_replications(
        suite.get("future_evidence_selection_failure"),
        "suite.future_evidence_selection_failure",
    )
    split_confirmation = _require_mapping(
        suite.get("split_stream_confirmation"), "suite.split_stream_confirmation"
    )
    split = _suite_replications(split_confirmation, "suite.split_stream_confirmation")
    selection_streams = _positive_integer(
        split_confirmation.get("selection_streams"),
        "suite.split_stream_confirmation.selection_streams",
    )
    confirmation_streams = _positive_integer(
        split_confirmation.get("confirmation_streams"),
        "suite.split_stream_confirmation.confirmation_streams",
    )
    if selection_streams != split or confirmation_streams != split:
        raise ValueError(
            "split-stream selection and confirmation counts must equal the declared replication count"
        )
    learned = _require_mapping(suite.get("learned_policy"), "suite.learned_policy")
    learned_artifact = _require_mapping(
        learned.get("artifact"), "suite.learned_policy.artifact"
    )
    search = _require_mapping(
        learned_artifact.get("policy_complexity"),
        "suite.learned_policy.artifact.policy_complexity",
    )
    candidate_count = _positive_integer(
        search.get("candidate_count"), "candidate_count"
    )
    train_reps = _positive_integer(learned_artifact.get("train_reps"), "train_reps")
    learned_training = candidate_count * train_reps
    learned_evaluation = _suite_replications(
        learned.get("evaluation"),
        "suite.learned_policy.evaluation",
        keys=("evaluation_reps", "n_reps"),
    )

    categories = [
        {
            "key": "operating_surface",
            "label": "Operating-characteristic surface",
            "trace_calls": surface,
            "max_steps_per_trace": action_horizon,
        },
        {
            "key": "paired_policy_comparisons",
            "label": "Paired policy comparisons",
            "trace_calls": paired_traces,
            "max_steps_per_trace": action_horizon,
        },
        {
            "key": "chronological_online_selection",
            "label": "Chronological online selection",
            "trace_calls": online,
            "max_steps_per_trace": action_horizon,
        },
        {
            "key": "all_null_selection_calibration",
            "label": "All-null selection calibration",
            "trace_calls": calibration,
            "max_steps_per_trace": action_horizon,
        },
        {
            "key": "future_evidence_contrast",
            "label": "Future-evidence contrast",
            "trace_calls": future,
            "max_steps_per_trace": action_horizon,
        },
        {
            "key": "split_stream_selection",
            "label": "Split-stream visible selection",
            "trace_calls": selection_streams,
            # The selection trace is run on the suite's separately declared
            # ``split_confirmation_scenario``, whose horizon differs from the
            # main action-loop operating grid when configured to do so.
            "max_steps_per_trace": split_confirmation_horizon,
        },
        {
            "key": "split_stream_confirmation",
            "label": "Split-stream independent confirmation",
            "trace_calls": confirmation_streams,
            "max_steps_per_trace": split_confirmation_horizon,
        },
        {
            "key": "learned_policy_training",
            "label": "Learned-policy candidate training",
            "trace_calls": learned_training,
            "max_steps_per_trace": action_horizon,
        },
        {
            "key": "learned_policy_evaluation",
            "label": "Held-out learned-policy evaluation",
            "trace_calls": learned_evaluation,
            "max_steps_per_trace": action_horizon,
        },
    ]
    trace_calls = sum(
        _positive_integer(item["trace_calls"], "action category trace_calls")
        for item in categories
    )
    upper_steps = sum(
        _positive_integer(item["trace_calls"], "action category trace_calls")
        * _positive_integer(
            item["max_steps_per_trace"], "action category max_steps_per_trace"
        )
        for item in categories
    )
    return {
        "trace_categories": categories,
        "action_loop_trace_calls": trace_calls,
        "action_loop_step_upper_bound": upper_steps,
        "static_confirmation_streams": confirmation_streams,
        "confirmation_observation_slots_upper_bound": confirmation_streams
        * split_confirmation_horizon,
        "learned_policy_candidate_count": candidate_count,
        "learned_policy_training_replications_per_candidate": train_reps,
        "boundary": (
            "Trace calls and maximum steps are source-level execution counts. "
            "They are not elapsed time, simulated resource cost C, e-process evidence E_t, "
            "or online alpha-wealth W^alpha."
        ),
    }


def build_computational_complexity(
    *,
    plan: ComputationalComplexityPlan,
    m: int,
    main_replications: int,
    method_count: int,
    effect_points: int,
    dependence_regime_count: int,
    sequential_horizon: int,
    online_horizon: int,
    action_horizon: int,
    split_confirmation_horizon: int,
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic release accounting from declared inputs/results."""
    m = _positive_integer(m, "m")
    main_replications = _positive_integer(main_replications, "main_replications")
    method_count = _positive_integer(method_count, "method_count")
    effect_points = _positive_integer(effect_points, "effect_points")
    dependence_regime_count = _positive_integer(
        dependence_regime_count, "dependence_regime_count"
    )
    sequential_horizon = _positive_integer(sequential_horizon, "sequential_horizon")
    online_horizon = _positive_integer(online_horizon, "online_horizon")
    action_horizon = _positive_integer(action_horizon, "action_horizon")
    split_confirmation_horizon = _positive_integer(
        split_confirmation_horizon, "split_confirmation_horizon"
    )
    action_ledger = action_workload_ledger(
        suite,
        action_horizon=action_horizon,
        split_confirmation_horizon=split_confirmation_horizon,
    )
    fixed_curves = [
        _curve(
            "Bonferroni / Sidak",
            plan.hypothesis_grid,
            linear_work,
            time_order="Theta(m)",
        ),
        _curve(
            "BH-family rank correction",
            plan.hypothesis_grid,
            rank_work,
            time_order="Theta(m log2 m)",
        ),
    ]
    stream_curves = [
        _curve(
            "Binary e-process and e-LOND (coincident linear proxy)",
            plan.stream_horizon_grid,
            linear_work,
            time_order="Theta(T)",
        ),
        _curve(
            "LORD++ recurrence-plus-validator upper proxy",
            plan.stream_horizon_grid,
            lord_history_work_upper,
            time_order="O(T^2) implementation upper proxy",
        ),
        _curve(
            "SAFFRON sparse-history upper proxy",
            plan.stream_horizon_grid,
            saffron_sparse_history_work_upper,
            time_order="O(T^2) implementation upper proxy",
        ),
        _curve(
            "SAFFRON dense-history upper proxy",
            plan.stream_horizon_grid,
            saffron_dense_history_work_upper,
            time_order="O(T^3) implementation upper proxy",
        ),
    ]
    analytical_rows = [
        {
            "name": "single_step_corrections",
            "input_symbol": "m",
            "time_order": "Theta(m)",
            "retained_space_order": "Theta(m)",
            "work_proxy_expression": "m",
            "work_proxy_at_declared_input": linear_work(m),
            "implementation_scope": "bonferroni and sidak vector paths in corrections.py",
            "boundary": "Source-level proxy; not a CPU-instruction or cross-hardware claim.",
        },
        {
            "name": "rank_based_corrections",
            "input_symbol": "m",
            "time_order": "Theta(m log2 m)",
            "retained_space_order": "Theta(m)",
            "work_proxy_expression": "m max(1, ceil(log2 m))",
            "work_proxy_at_declared_input": rank_work(m),
            "implementation_scope": "stable-sort paths for Holm, Hochberg, BH, BY, Storey, adaptive BH, and weighted BH",
            "boundary": "The sort-dominant proxy does not expand NumPy or SciPy internals.",
        },
        {
            "name": "fixed_family_simulation",
            "input_symbol": "B,m",
            "time_order": "Theta(B[m + selected-correction work at m]) per procedure",
            "retained_space_order": "Theta(Bm) generated draws before scoring",
            "work_proxy_expression": "B m per procedure draw set",
            "work_proxy_at_declared_input": main_replications * m,
            "implementation_scope": "simulate_procedure materializes p-value and truth draws before worker scoring",
            "boundary": "Worker count does not remove the pre-scoring retained-draw allocation.",
        },
        {
            "name": "comparison_surface",
            "input_symbol": "K,B,m",
            "time_order": "Theta(K B [m + selected-correction work at m])",
            "retained_space_order": "Theta(Bm) during each sequential procedure call",
            "work_proxy_expression": "K B m generated hypothesis positions",
            "work_proxy_at_declared_input": method_count * main_replications * m,
            "implementation_scope": "compare_procedures repeats generation per method while reusing the same seed schedule",
            "boundary": "The proxy describes this implementation, not the mathematical correction procedures in general.",
        },
        {
            "name": "dependence_paths",
            "input_symbol": "D,B,m",
            "time_order": "Two procedure passes: release factor fast path Theta(Bm); cached non-factor paths O(m^3) setup plus two dense O(Bm^2) draws",
            "retained_space_order": "O(min(non-factor regimes, 64) m^2) cached covariance factors",
            "work_proxy_expression": "2 D B m^2 dense-draw upper proxy after cache construction",
            "work_proxy_at_declared_input": 2
            * dependence_regime_count
            * main_replications
            * m
            * m,
            "implementation_scope": "release dependence_sweep -> simulate_procedure -> generate_pvalues factor fast path and cached non-factor Gaussian branches",
            "boundary": "The proxy includes separate BH and BY draws after cache construction. It excludes standalone dependence.sample_gaussian factorization, which is not a release generator path; branch-specific paths preclude one universal m-scaling.",
        },
        {
            "name": "binary_e_process",
            "input_symbol": "T",
            "time_order": "Theta(T)",
            "retained_space_order": "Theta(T)",
            "work_proxy_expression": "T",
            "work_proxy_at_declared_input": linear_work(sequential_horizon),
            "implementation_scope": "binary_e_process likelihood-ratio updates and stored finite path",
            "boundary": "This is a source-level loop count, separate from the e-process evidence value E_t.",
        },
        {
            "name": "confidence_sequence_inversion",
            "input_symbol": "T",
            "time_order": "at least Omega(T^2) prefix scans plus numerical root solves",
            "retained_space_order": "Theta(T)",
            "work_proxy_expression": "T(T+1)/2 prefix observations plus 2T endpoints",
            "work_proxy_at_declared_input": sequential_horizon
            * (sequential_horizon + 1)
            // 2
            + 2 * sequential_horizon,
            "implementation_scope": "bernoulli_confidence_sequence with repeated prefix sums and bracketing roots",
            "boundary": "The numerical root solver is profiled empirically rather than reduced to a universal closed-form runtime.",
        },
        {
            "name": "lord_plus_plus_history_scan",
            "input_symbol": "T,R",
            "time_order": "O(TR), worst O(T^2)",
            "retained_space_order": "Theta(T)",
            "work_proxy_expression": "2 T(T+1) conservative recurrence-plus-validator upper proxy",
            "work_proxy_at_declared_input": lord_history_work_upper(online_horizon),
            "implementation_scope": "current LORD++ recurrence plus serialized-trace validator replay",
            "boundary": "An implementation property; not a complexity theorem for LORD++.",
        },
        {
            "name": "saffron_history_scan",
            "input_symbol": "T,R",
            "time_order": "O(T^2) sparse history; worst O(T^3) dense history",
            "retained_space_order": "Theta(T)",
            "work_proxy_expression": "candidate-prefix and per-rejection candidate-slice upper proxy",
            "work_proxy_at_declared_input": saffron_dense_history_work_upper(
                online_horizon
            ),
            "implementation_scope": "current SAFFRON recurrence and serialized-trace validator",
            "boundary": "An implementation property; not a complexity theorem for SAFFRON.",
        },
        {
            "name": "e_lond",
            "input_symbol": "T",
            "time_order": "Theta(T)",
            "retained_space_order": "Theta(T)",
            "work_proxy_expression": "T",
            "work_proxy_at_declared_input": linear_work(online_horizon),
            "implementation_scope": "e_lond predictable schedule, trace validation, and stored accounting path",
            "boundary": "The proxy is separate from nominal alpha accounting and its validity assumptions.",
        },
        {
            "name": "action_loop_execution_ledger",
            "input_symbol": "N_j,H_j",
            "time_order": "Declared trace and step-count accounting; pymdp internals remain unexpanded",
            "retained_space_order": "Trace retention depends on the named diagnostic path",
            "work_proxy_expression": "sum_j N_j H_j permitted action or confirmation slots",
            "work_proxy_at_declared_input": int(
                action_ledger["action_loop_step_upper_bound"]
            ),
            "implementation_scope": "action-loop surface, paired comparisons, selection diagnostics, and learned-policy train/freeze/evaluate paths",
            "boundary": "Trace counts are not a physical runtime model, simulated resource cost C, or policy-quality ranking.",
        },
    ]
    return {
        "contract_version": COMPUTATIONAL_COMPLEXITY_CONTRACT_VERSION,
        "evidence_class": "provenance_contract",
        "claim_boundary": (
            "Source-level dominant-work accounting for this version of the package. "
            "It is not a throughput benchmark, lower bound, dependency-internal runtime claim, "
            "simulated resource cost C, e-process evidence E_t, or online alpha-wealth W^alpha."
        ),
        "units": {
            "work_proxy": "dimensionless source-work proxy; not CPU instructions or elapsed time",
            "trace_call": "one call to the named action-loop or confirmation execution path",
            "step_upper_bound": "maximum permitted action or confirmation observation slot count",
        },
        "dimensions": {
            "m_hypotheses": m,
            "B_main_replications": main_replications,
            "K_methods": method_count,
            "G_effect_points": effect_points,
            "D_dependence_regimes": dependence_regime_count,
            "T_sequential_horizon": sequential_horizon,
            "T_online_horizon": online_horizon,
            "H_action_horizon": action_horizon,
            "H_split_confirmation": split_confirmation_horizon,
        },
        "plot_data": {
            "hypothesis_work_curves": fixed_curves,
            "stream_work_curves": stream_curves,
            "action_trace_categories": action_ledger["trace_categories"],
        },
        "analytical_rows": analytical_rows,
        "release_workload": {
            "fixed_family": {
                "method_comparison_procedure_calls": method_count * main_replications,
                "method_comparison_generated_hypothesis_positions": method_count
                * main_replications
                * m,
                "power_curve_procedure_calls": effect_points * main_replications,
                "power_curve_generated_hypothesis_positions": effect_points
                * main_replications
                * m,
                "dependence_procedure_calls": dependence_regime_count
                * main_replications
                * 2,
                "per_procedure_retained_pvalue_elements": main_replications * m,
                "per_procedure_retained_truth_elements": main_replications * m,
                "per_procedure_retained_summary_values": 3 * main_replications,
                "boundary": "Retained elements describe current Python-level draw materialization, not allocated bytes or peak process memory.",
            },
            "action_loop": action_ledger,
        },
    }


def _require_nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _require_mapping_keys(value: Mapping[str, Any], keys: set[str], name: str) -> None:
    missing = sorted(keys - set(value))
    if missing:
        raise ValueError(f"{name} is missing keys: {missing}")


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], name: str) -> None:
    """Require a closed receipt shape where omission could hide provenance."""
    _require_mapping_keys(value, keys, name)
    unexpected = sorted(set(value) - keys)
    if unexpected:
        raise ValueError(f"{name} has unexpected keys: {unexpected}")


def _reject_release_timing_fields(
    value: Any, name: str = "computational_complexity"
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _FORBIDDEN_RELEASE_TIMING_TERMS:
                raise ValueError(f"{name} must not contain elapsed-time field {key!r}")
            _reject_release_timing_fields(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_release_timing_fields(item, f"{name}[{index}]")


def validate_computational_complexity(value: Mapping[str, Any]) -> None:
    """Validate a deterministic release workload-accounting block."""
    record = _require_mapping(value, "computational_complexity")
    _reject_release_timing_fields(record)
    _require_exact_keys(
        record,
        {
            "contract_version",
            "evidence_class",
            "claim_boundary",
            "units",
            "dimensions",
            "plot_data",
            "analytical_rows",
            "release_workload",
        },
        "computational_complexity",
    )
    if record["contract_version"] != COMPUTATIONAL_COMPLEXITY_CONTRACT_VERSION:
        raise ValueError("computational_complexity.contract_version is unsupported")
    if record["evidence_class"] != "provenance_contract":
        raise ValueError(
            "computational_complexity.evidence_class must be provenance_contract"
        )
    _require_nonempty_text(
        record["claim_boundary"], "computational_complexity.claim_boundary"
    )
    units = _require_mapping(record["units"], "computational_complexity.units")
    _require_exact_keys(
        units,
        {"work_proxy", "trace_call", "step_upper_bound"},
        "computational_complexity.units",
    )
    for key, item in units.items():
        _require_nonempty_text(item, f"computational_complexity.units.{key}")
    dimensions = _require_mapping(
        record["dimensions"], "computational_complexity.dimensions"
    )
    expected_dimensions = {
        "m_hypotheses",
        "B_main_replications",
        "K_methods",
        "G_effect_points",
        "D_dependence_regimes",
        "T_sequential_horizon",
        "T_online_horizon",
        "H_action_horizon",
        "H_split_confirmation",
    }
    _require_exact_keys(
        dimensions, expected_dimensions, "computational_complexity.dimensions"
    )
    normalized_dimensions = {
        key: _positive_integer(
            dimensions[key], f"computational_complexity.dimensions.{key}"
        )
        for key in expected_dimensions
    }
    analytical = record["analytical_rows"]
    if not isinstance(analytical, list) or not analytical:
        raise ValueError("computational_complexity.analytical_rows must be non-empty")
    expected_analytical_names = (
        "single_step_corrections",
        "rank_based_corrections",
        "fixed_family_simulation",
        "comparison_surface",
        "dependence_paths",
        "binary_e_process",
        "confidence_sequence_inversion",
        "lord_plus_plus_history_scan",
        "saffron_history_scan",
        "e_lond",
        "action_loop_execution_ledger",
    )
    names: set[str] = set()
    analytical_by_name: dict[str, Mapping[str, Any]] = {}
    required_row = {
        "name",
        "input_symbol",
        "time_order",
        "retained_space_order",
        "work_proxy_expression",
        "work_proxy_at_declared_input",
        "implementation_scope",
        "boundary",
    }
    for index, item in enumerate(analytical):
        row = _require_mapping(
            item, f"computational_complexity.analytical_rows[{index}]"
        )
        _require_exact_keys(
            row, required_row, f"computational_complexity.analytical_rows[{index}]"
        )
        name = _require_nonempty_text(row["name"], "computational complexity row name")
        if name in names:
            raise ValueError(
                "computational_complexity.analytical_rows names must be unique"
            )
        names.add(name)
        analytical_by_name[name] = row
        for key in required_row - {"name", "work_proxy_at_declared_input"}:
            _require_nonempty_text(
                row[key], f"computational complexity row {name}.{key}"
            )
        _nonnegative_integer(
            row["work_proxy_at_declared_input"],
            f"computational complexity row {name}.work_proxy_at_declared_input",
        )
    if tuple(item["name"] for item in analytical) != expected_analytical_names:
        raise ValueError(
            "computational_complexity.analytical_rows must use the declared ordered source-work rows"
        )
    plot_data = _require_mapping(
        record["plot_data"], "computational_complexity.plot_data"
    )
    _require_exact_keys(
        plot_data,
        {"hypothesis_work_curves", "stream_work_curves", "action_trace_categories"},
        "computational_complexity.plot_data",
    )
    curve_specs = {
        "hypothesis_work_curves": (
            ("Bonferroni / Sidak", "Theta(m)", linear_work),
            ("BH-family rank correction", "Theta(m log2 m)", rank_work),
        ),
        "stream_work_curves": (
            (
                "Binary e-process and e-LOND (coincident linear proxy)",
                "Theta(T)",
                linear_work,
            ),
            (
                "LORD++ recurrence-plus-validator upper proxy",
                "O(T^2) implementation upper proxy",
                lord_history_work_upper,
            ),
            (
                "SAFFRON sparse-history upper proxy",
                "O(T^2) implementation upper proxy",
                saffron_sparse_history_work_upper,
            ),
            (
                "SAFFRON dense-history upper proxy",
                "O(T^3) implementation upper proxy",
                saffron_dense_history_work_upper,
            ),
        ),
    }
    for key, expected_curves in curve_specs.items():
        curves = plot_data[key]
        if not isinstance(curves, list) or not curves:
            raise ValueError(
                f"computational_complexity.plot_data.{key} must be non-empty"
            )
        if len(curves) != len(expected_curves):
            raise ValueError(
                f"computational_complexity.plot_data.{key} must contain the declared curve family"
            )
        for index, (item, expected_curve) in enumerate(zip(curves, expected_curves)):
            curve = _require_mapping(
                item, f"computational_complexity.plot_data.{key}[{index}]"
            )
            _require_exact_keys(
                curve,
                {"name", "input_values", "work_proxy_values", "time_order", "unit"},
                "complexity curve",
            )
            expected_name, expected_order, work_function = expected_curve
            if curve["name"] != expected_name or curve["time_order"] != expected_order:
                raise ValueError(
                    "complexity curve metadata does not match the declared source-work proxy"
                )
            if curve["unit"] != "dimensionless source-work proxy":
                raise ValueError(
                    "complexity curve unit must be the declared source-work proxy unit"
                )
            inputs = curve["input_values"]
            proxies = curve["work_proxy_values"]
            if (
                not isinstance(inputs, list)
                or not isinstance(proxies, list)
                or len(inputs) != len(proxies)
                or not inputs
            ):
                raise ValueError(
                    "complexity curve input/proxy arrays must be non-empty and aligned"
                )
            normalized_inputs = [
                _positive_integer(item, "complexity curve input") for item in inputs
            ]
            if any(
                later <= earlier
                for earlier, later in zip(normalized_inputs, normalized_inputs[1:])
            ):
                raise ValueError("complexity curve inputs must be strictly increasing")
            for proxy in proxies:
                _nonnegative_integer(proxy, "complexity curve work proxy")
            expected_proxies = [work_function(item) for item in normalized_inputs]
            if proxies != expected_proxies:
                raise ValueError(
                    "complexity curve work proxies do not match their declared source-work formula"
                )
    categories = plot_data["action_trace_categories"]
    if not isinstance(categories, list) or not categories:
        raise ValueError(
            "computational_complexity.plot_data.action_trace_categories must be non-empty"
        )
    expected_category_specs = (
        ("operating_surface", "Operating-characteristic surface"),
        ("paired_policy_comparisons", "Paired policy comparisons"),
        ("chronological_online_selection", "Chronological online selection"),
        ("all_null_selection_calibration", "All-null selection calibration"),
        ("future_evidence_contrast", "Future-evidence contrast"),
        ("split_stream_selection", "Split-stream visible selection"),
        ("split_stream_confirmation", "Split-stream independent confirmation"),
        ("learned_policy_training", "Learned-policy candidate training"),
        ("learned_policy_evaluation", "Held-out learned-policy evaluation"),
    )
    if len(categories) != len(expected_category_specs):
        raise ValueError(
            "computational_complexity action categories must contain the declared execution families"
        )
    category_keys: set[str] = set()
    category_by_key: dict[str, Mapping[str, Any]] = {}
    for item, expected_category in zip(categories, expected_category_specs):
        category = _require_mapping(item, "computational complexity action category")
        _require_exact_keys(
            category,
            {"key", "label", "trace_calls", "max_steps_per_trace"},
            "computational complexity action category",
        )
        key = _require_nonempty_text(
            category["key"], "computational complexity action category key"
        )
        if key in category_keys:
            raise ValueError(
                "computational complexity action category keys must be unique"
            )
        category_keys.add(key)
        if (key, category["label"]) != expected_category:
            raise ValueError(
                "computational complexity action categories must use the declared labels and order"
            )
        category_by_key[key] = category
        _positive_integer(
            category["trace_calls"], "computational complexity action trace calls"
        )
        _positive_integer(
            category["max_steps_per_trace"], "computational complexity action max steps"
        )
    workload = _require_mapping(
        record["release_workload"], "computational_complexity.release_workload"
    )
    _require_exact_keys(
        workload,
        {"fixed_family", "action_loop"},
        "computational_complexity.release_workload",
    )
    fixed = _require_mapping(
        workload["fixed_family"],
        "computational_complexity.release_workload.fixed_family",
    )
    fixed_keys = {
        "method_comparison_procedure_calls",
        "method_comparison_generated_hypothesis_positions",
        "power_curve_procedure_calls",
        "power_curve_generated_hypothesis_positions",
        "dependence_procedure_calls",
        "per_procedure_retained_pvalue_elements",
        "per_procedure_retained_truth_elements",
        "per_procedure_retained_summary_values",
        "boundary",
    }
    _require_exact_keys(
        fixed, fixed_keys, "computational_complexity.release_workload.fixed_family"
    )
    for key in fixed_keys - {"boundary"}:
        _positive_integer(fixed[key], f"computational fixed workload {key}")
    _require_nonempty_text(fixed["boundary"], "computational fixed workload boundary")
    m = normalized_dimensions["m_hypotheses"]
    main_replications = normalized_dimensions["B_main_replications"]
    method_count = normalized_dimensions["K_methods"]
    effect_points = normalized_dimensions["G_effect_points"]
    dependence_regime_count = normalized_dimensions["D_dependence_regimes"]
    expected_fixed_workload = {
        "method_comparison_procedure_calls": method_count * main_replications,
        "method_comparison_generated_hypothesis_positions": method_count
        * main_replications
        * m,
        "power_curve_procedure_calls": effect_points * main_replications,
        "power_curve_generated_hypothesis_positions": effect_points
        * main_replications
        * m,
        "dependence_procedure_calls": dependence_regime_count * main_replications * 2,
        "per_procedure_retained_pvalue_elements": main_replications * m,
        "per_procedure_retained_truth_elements": main_replications * m,
        "per_procedure_retained_summary_values": 3 * main_replications,
    }
    if any(fixed[key] != expected for key, expected in expected_fixed_workload.items()):
        raise ValueError(
            "computational fixed workload does not match the declared dimensions"
        )
    action = _require_mapping(
        workload["action_loop"], "computational_complexity.release_workload.action_loop"
    )
    action_keys = {
        "trace_categories",
        "action_loop_trace_calls",
        "action_loop_step_upper_bound",
        "static_confirmation_streams",
        "confirmation_observation_slots_upper_bound",
        "learned_policy_candidate_count",
        "learned_policy_training_replications_per_candidate",
        "boundary",
    }
    _require_exact_keys(
        action, action_keys, "computational_complexity.release_workload.action_loop"
    )
    for key in action_keys - {"trace_categories", "boundary"}:
        _positive_integer(action[key], f"computational action workload {key}")
    if action["trace_categories"] != categories:
        raise ValueError(
            "computational action workload categories must equal plot data categories"
        )
    if int(action["action_loop_trace_calls"]) != sum(
        int(item["trace_calls"]) for item in categories
    ):
        raise ValueError(
            "computational action-loop trace total disagrees with categories"
        )
    expected_steps = sum(
        int(item["trace_calls"]) * int(item["max_steps_per_trace"])
        for item in categories
    )
    if int(action["action_loop_step_upper_bound"]) != expected_steps:
        raise ValueError(
            "computational action-loop step bound disagrees with categories"
        )
    confirmation_category = category_by_key["split_stream_confirmation"]
    training_category = category_by_key["learned_policy_training"]
    if action["static_confirmation_streams"] != confirmation_category["trace_calls"]:
        raise ValueError(
            "computational static confirmation streams must match the confirmation category"
        )
    if action["confirmation_observation_slots_upper_bound"] != (
        confirmation_category["trace_calls"]
        * normalized_dimensions["H_split_confirmation"]
    ):
        raise ValueError(
            "computational confirmation slots must match the split-confirmation horizon"
        )
    if training_category["trace_calls"] != (
        action["learned_policy_candidate_count"]
        * action["learned_policy_training_replications_per_candidate"]
    ):
        raise ValueError(
            "computational learned-policy training calls must match the declared search ledger"
        )
    expected_analytical_work = {
        "single_step_corrections": linear_work(m),
        "rank_based_corrections": rank_work(m),
        "fixed_family_simulation": main_replications * m,
        "comparison_surface": method_count * main_replications * m,
        "dependence_paths": 2 * dependence_regime_count * main_replications * m * m,
        "binary_e_process": linear_work(normalized_dimensions["T_sequential_horizon"]),
        "confidence_sequence_inversion": normalized_dimensions["T_sequential_horizon"]
        * (normalized_dimensions["T_sequential_horizon"] + 1)
        // 2
        + 2 * normalized_dimensions["T_sequential_horizon"],
        "lord_plus_plus_history_scan": lord_history_work_upper(
            normalized_dimensions["T_online_horizon"]
        ),
        "saffron_history_scan": saffron_dense_history_work_upper(
            normalized_dimensions["T_online_horizon"]
        ),
        "e_lond": linear_work(normalized_dimensions["T_online_horizon"]),
        "action_loop_execution_ledger": expected_steps,
    }
    if any(
        analytical_by_name[name]["work_proxy_at_declared_input"] != expected
        for name, expected in expected_analytical_work.items()
    ):
        raise ValueError(
            "computational analytical source-work values do not match the declared dimensions"
        )
    _require_nonempty_text(action["boundary"], "computational action workload boundary")


def validate_computational_complexity_against_results(
    value: Mapping[str, Any], results: Mapping[str, Any]
) -> None:
    """Rebuild a current release ledger from its declared result contracts.

    Shape and formula checks above make the ledger self-consistent.  A current
    release also has to bind that ledger to the actual experiment, suite, and
    result dimensions so a plausible but substituted accounting block cannot
    inherit the release's provenance witnesses.
    """
    record = _require_mapping(value, "computational_complexity")
    validate_computational_complexity(record)
    result_record = _require_mapping(results, "results")
    experiment = _require_mapping(result_record.get("experiment"), "results.experiment")
    meta = _require_mapping(result_record.get("meta"), "results.meta")
    action_loop = _require_mapping(
        experiment.get("action_loop"), "results.experiment.action_loop"
    )
    plan = complexity_plan_from_mapping(
        _require_mapping(
            experiment.get("computational_complexity"),
            "results.experiment.computational_complexity",
        )
    )
    methods = experiment.get("methods")
    effect_grid = experiment.get("effect_grid")
    dependence_sweep = result_record.get("dependence_sweep")
    if not isinstance(methods, list) or not methods:
        raise ValueError("results.experiment.methods must be a non-empty list")
    if not isinstance(effect_grid, list) or not effect_grid:
        raise ValueError("results.experiment.effect_grid must be a non-empty list")
    if not isinstance(dependence_sweep, list) or not dependence_sweep:
        raise ValueError("results.dependence_sweep must be a non-empty list")
    sequential = _require_mapping(result_record.get("sequential"), "results.sequential")
    online_fdr = _require_mapping(result_record.get("online_fdr"), "results.online_fdr")
    suite = _require_mapping(result_record.get("suite"), "results.suite")
    action_horizon = _positive_integer(
        action_loop.get("horizon"), "results.experiment.action_loop.horizon"
    )
    split_confirmation_horizon = _positive_integer(
        action_loop.get("split_confirmation_steps", action_horizon),
        "results.experiment.action_loop.split_confirmation_steps",
    )
    expected = build_computational_complexity(
        plan=plan,
        m=_positive_integer(experiment.get("m"), "results.experiment.m"),
        main_replications=_positive_integer(meta.get("n_reps"), "results.meta.n_reps"),
        method_count=len(methods),
        effect_points=len(effect_grid),
        dependence_regime_count=len(dependence_sweep),
        sequential_horizon=_positive_integer(
            sequential.get("horizon"), "results.sequential.horizon"
        ),
        online_horizon=_positive_integer(
            online_fdr.get("horizon"), "results.online_fdr.horizon"
        ),
        action_horizon=action_horizon,
        split_confirmation_horizon=split_confirmation_horizon,
        suite=suite,
    )
    if dict(record) != expected:
        raise ValueError(
            "computational_complexity does not exactly match the current results contract"
        )


def summarize_timing_samples(
    samples_ns: Sequence[int | float],
) -> dict[str, float | int]:
    """Summarize real repeated timings without assigning scientific uncertainty."""
    values = np.asarray(samples_ns, dtype=float)
    if (
        values.ndim != 1
        or values.size < 1
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError(
            "timing samples must be a non-empty finite non-negative vector"
        )
    return {
        "n": int(values.size),
        "min_ns": float(np.min(values)),
        "q1_ns": float(np.quantile(values, 0.25)),
        "median_ns": float(np.median(values)),
        "q3_ns": float(np.quantile(values, 0.75)),
        "max_ns": float(np.max(values)),
    }


def timing_profile_digest(value: Mapping[str, Any]) -> str:
    """Hash an isolated timing profile while normalizing its self-digest."""
    payload = json.loads(json.dumps(value, sort_keys=True, default=float))
    payload.pop("result_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def timing_plan_digest(value: Mapping[str, Any] | ComputationalComplexityPlan) -> str:
    """Hash the canonical timing-plan snapshot independently of a profile.

    A timing profile retains the configuration-file digest as provenance, but
    this canonical digest binds the executed matrix itself.  The two witnesses
    answer different questions and must not be substituted for one another.
    """
    plan = (
        value
        if isinstance(value, ComputationalComplexityPlan)
        else complexity_plan_from_mapping(_require_mapping(value, "timing plan"))
    )
    encoded = json.dumps(
        plan.to_mapping(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    """Require one canonical hexadecimal SHA-256 witness."""
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest") from exc
    return value


def _reject_identity_receipt_fields(value: Any, name: str) -> None:
    """Reject host-identity fields even when a caller supplies valid timing rows."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_TIMING_IDENTITY_KEYS:
                raise ValueError(f"{name} must not contain host-identity field {key!r}")
            _reject_identity_receipt_fields(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_identity_receipt_fields(item, f"{name}[{index}]")


def _validate_timing_plan_source(
    profile: Mapping[str, Any],
    plan: ComputationalComplexityPlan,
    *,
    expected_frozen_plan: ComputationalComplexityPlan | None,
    expected_plan_source_sha256: str | None,
) -> None:
    """Bind a timing plan either to the frozen config or explicit test-only plan."""
    source = _require_nonempty_text(
        profile["plan_source"], "timing profile plan_source"
    )
    digest = profile["plan_source_sha256"]
    if source == TIMING_PROFILE_FROZEN_PLAN_SOURCE:
        _require_sha256(digest, "timing profile plan_source_sha256")
        if expected_frozen_plan is None or expected_plan_source_sha256 is None:
            raise ValueError(
                "a frozen timing profile requires the expected frozen plan and configuration digest"
            )
        expected_digest = _require_sha256(
            expected_plan_source_sha256,
            "expected frozen timing configuration digest",
        )
        if digest != expected_digest:
            raise ValueError(
                "timing profile plan_source_sha256 does not match the expected frozen configuration"
            )
        if plan != expected_frozen_plan:
            raise ValueError(
                "timing profile plan does not match the expected frozen configuration plan"
            )
        return
    if source == TIMING_PROFILE_PROGRAMMATIC_PLAN_SOURCE:
        if digest is not None:
            raise ValueError(
                "programmatic timing profile plan_source_sha256 must be null"
            )
        return
    raise ValueError(
        "timing profile plan_source is not an allowed isolated-lane source"
    )


def _validate_timing_environment(value: Any) -> None:
    """Validate the deliberately redacted environment receipt's closed structure."""
    environment = _require_mapping(value, "timing profile environment")
    _require_exact_keys(
        environment, _TIMING_ENVIRONMENT_KEYS, "timing profile environment"
    )
    _reject_identity_receipt_fields(environment, "timing profile environment")
    python = _require_mapping(
        environment["python"], "timing profile environment.python"
    )
    _require_exact_keys(
        python, _TIMING_PYTHON_KEYS, "timing profile environment.python"
    )
    runtime = _require_mapping(
        environment["runtime"], "timing profile environment.runtime"
    )
    _require_exact_keys(
        runtime, _TIMING_RUNTIME_KEYS, "timing profile environment.runtime"
    )
    libraries = _require_mapping(
        environment["libraries"], "timing profile environment.libraries"
    )
    _require_exact_keys(
        libraries, _TIMING_LIBRARY_KEYS, "timing profile environment.libraries"
    )
    threads = _require_mapping(
        environment["thread_environment"],
        "timing profile environment.thread_environment",
    )
    _require_exact_keys(
        threads,
        _TIMING_THREAD_ENVIRONMENT_KEYS,
        "timing profile environment.thread_environment",
    )
    for section_name, section in (
        ("python", python),
        ("runtime", runtime),
        ("libraries", libraries),
        ("thread_environment", threads),
    ):
        for key, item in section.items():
            _require_nonempty_text(
                item, f"timing profile environment.{section_name}.{key}"
            )
    redaction = _require_nonempty_text(
        environment["redaction"], "timing profile environment.redaction"
    ).lower()
    if any(term not in redaction for term in _TIMING_REDACTION_TERMS):
        raise ValueError(
            "timing profile environment.redaction must name every redacted identity field"
        )


def _validate_timing_source_hashes(value: Any) -> dict[str, str]:
    """Validate direct roots plus safely named transitive local dependencies."""
    source_hashes = _require_mapping(value, "timing profile source_hashes")
    missing = sorted(set(TIMING_PROFILE_SOURCE_NAMES) - set(source_hashes))
    if missing:
        raise ValueError(f"timing profile source_hashes is missing keys: {missing}")
    unexpected = sorted(
        str(key)
        for key in source_hashes
        if key not in TIMING_PROFILE_SOURCE_NAMES
        and (
            not isinstance(key, str)
            or not key.startswith("dependency:")
            or not key.removeprefix("dependency:").endswith(".py")
            or key.removeprefix("dependency:").startswith("/")
            or ".." in key.removeprefix("dependency:").split("/")
        )
    )
    if unexpected:
        raise ValueError(
            f"timing profile source_hashes has unexpected keys: {unexpected}"
        )
    normalized: dict[str, str] = {}
    for name, digest in source_hashes.items():
        normalized[str(name)] = _require_sha256(
            digest, f"timing profile source_hashes.{name}"
        )
    return normalized


def _validate_measurement_protocol(value: Any) -> None:
    """Keep timing methodology visible rather than accepting an opaque row set."""
    protocol = _require_mapping(value, "timing profile measurement_protocol")
    _require_exact_keys(
        protocol,
        set(TIMING_PROFILE_MEASUREMENT_PROTOCOL),
        "timing profile measurement_protocol",
    )
    if dict(protocol) != TIMING_PROFILE_MEASUREMENT_PROTOCOL:
        raise ValueError(
            "timing profile measurement_protocol must equal the declared isolated-lane protocol"
        )


def _expected_timing_row_identifiers(
    plan: ComputationalComplexityPlan,
) -> set[tuple[str, int, str]]:
    """Return the complete declared timing matrix for one isolated plan.

    A timing receipt is useful only if it covers the plan that it claims to
    exercise. The identifiers deliberately name source kernels and input grids,
    not performance thresholds; this remains a research-only local diagnostic.
    """
    identifiers: set[tuple[str, int, str]] = set()
    for method in plan.correction_methods:
        identifiers.update(
            (f"correction:{method}", int(size), _TIMING_CORRECTION_CONDITION)
            for size in plan.hypothesis_grid
        )
    for method in plan.online_methods:
        for condition in _TIMING_ONLINE_CONDITIONS:
            identifiers.update(
                (f"online:{method}", int(size), condition)
                for size in plan.stream_horizon_grid
            )
    for kernel, condition in _TIMING_SEQUENTIAL_CASES:
        identifiers.update(
            (kernel, int(size), condition) for size in plan.stream_horizon_grid
        )
    for policy in plan.action_policies:
        identifiers.update(
            (f"action_loop:{policy}", int(size), _TIMING_ACTION_CONDITION)
            for size in plan.action_horizon_grid
        )
    return identifiers


def validate_timing_profile(
    value: Mapping[str, Any],
    *,
    expected_frozen_plan: ComputationalComplexityPlan | None = None,
    expected_plan_source_sha256: str | None = None,
    expected_source_hashes: Mapping[str, str] | None = None,
) -> None:
    """Validate the isolated, explicitly non-promotional timing profile."""
    profile = _require_mapping(value, "timing_profile")
    _reject_identity_receipt_fields(profile, "timing_profile")
    _require_exact_keys(
        profile,
        {
            "contract_version",
            "kind",
            "status",
            "non_claim",
            "plan",
            "plan_sha256",
            "plan_source",
            "plan_source_sha256",
            "environment",
            "source_hashes",
            "measurement_protocol",
            "rows",
            "result_sha256",
        },
        "timing_profile",
    )
    if profile["contract_version"] != TIMING_PROFILE_CONTRACT_VERSION:
        raise ValueError("timing profile contract version is unsupported")
    if profile["kind"] != "research_only_local_runtime_diagnostic":
        raise ValueError(
            "timing profile must remain a research-only local runtime diagnostic"
        )
    if profile["status"] != TIMING_PROFILE_STATUS:
        raise ValueError("timing profile status must remain non-promotional")
    _require_nonempty_text(profile["non_claim"], "timing profile non_claim")
    plan = complexity_plan_from_mapping(
        _require_mapping(profile["plan"], "timing profile plan")
    )
    if _require_sha256(
        profile["plan_sha256"], "timing profile plan_sha256"
    ) != timing_plan_digest(plan):
        raise ValueError("timing profile plan_sha256 does not match the canonical plan")
    _validate_timing_plan_source(
        profile,
        plan,
        expected_frozen_plan=expected_frozen_plan,
        expected_plan_source_sha256=expected_plan_source_sha256,
    )
    _validate_timing_environment(profile["environment"])
    observed_source_hashes = _validate_timing_source_hashes(profile["source_hashes"])
    if expected_source_hashes is not None:
        expected = _validate_timing_source_hashes(expected_source_hashes)
        if observed_source_hashes != expected:
            raise ValueError(
                "timing profile source_hashes do not exactly match the expected source closure"
            )
    _validate_measurement_protocol(profile["measurement_protocol"])
    rows = profile["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("timing profile rows must be non-empty")
    identifiers: set[tuple[str, int, str]] = set()
    for index, item in enumerate(rows):
        row = _require_mapping(item, f"timing profile rows[{index}]")
        _require_exact_keys(
            row,
            _TIMING_ROW_KEYS,
            f"timing profile rows[{index}]",
        )
        kernel = _require_nonempty_text(row["kernel"], "timing profile kernel")
        input_size = _positive_integer(row["input_size"], "timing profile input_size")
        condition = _require_nonempty_text(row["condition"], "timing profile condition")
        identifier = (kernel, input_size, condition)
        if identifier in identifiers:
            raise ValueError(
                "timing profile rows must have unique kernel/input/condition identifiers"
            )
        identifiers.add(identifier)
        if (
            _nonnegative_integer(row["warmups"], "timing profile warmups")
            != plan.timing_warmups
        ):
            raise ValueError("timing profile warmups must match the declared plan")
        for samples_key, summary_key in (
            ("wall_time_ns_samples", "wall_time_ns_summary"),
            ("process_time_ns_samples", "process_time_ns_summary"),
        ):
            samples = row[samples_key]
            if not isinstance(samples, list) or len(samples) != plan.timing_repetitions:
                raise ValueError(
                    "timing profile sample count must match the declared plan"
                )
            for sample_index, sample in enumerate(samples):
                _nonnegative_integer(
                    sample,
                    f"timing profile {samples_key}[{sample_index}]",
                )
            summary = summarize_timing_samples(samples)
            if row[summary_key] != summary:
                raise ValueError(
                    f"timing profile {summary_key} does not match raw samples"
                )
    if identifiers != _expected_timing_row_identifiers(plan):
        raise ValueError(
            "timing profile rows must exactly cover the declared plan kernel/grid cases"
        )
    digest = profile["result_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != timing_profile_digest(profile)
    ):
        raise ValueError("timing profile result_sha256 does not match its content")


__all__ = [
    "COMPUTATIONAL_COMPLEXITY_CONTRACT_VERSION",
    "TIMING_PROFILE_CONTRACT_VERSION",
    "TIMING_PROFILE_STATUS",
    "TIMING_PROFILE_FROZEN_PLAN_SOURCE",
    "TIMING_PROFILE_PROGRAMMATIC_PLAN_SOURCE",
    "TIMING_PROFILE_SOURCE_NAMES",
    "TIMING_PROFILE_MEASUREMENT_PROTOCOL",
    "ComputationalComplexityPlan",
    "complexity_plan_from_mapping",
    "linear_work",
    "rank_work",
    "lord_history_work_upper",
    "saffron_sparse_history_work_upper",
    "saffron_dense_history_work_upper",
    "action_workload_ledger",
    "build_computational_complexity",
    "validate_computational_complexity",
    "validate_computational_complexity_against_results",
    "summarize_timing_samples",
    "timing_profile_digest",
    "timing_plan_digest",
    "validate_timing_profile",
]
