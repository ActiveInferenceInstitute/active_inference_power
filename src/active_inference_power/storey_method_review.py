"""Frozen three-way Storey method-review simulation.

This domain module compares three explicitly different uses of Storey-style
adaptivity in the canonical independent two-groups setting:

* a same-vector Storey plug-in q-value on one p-value vector;
* the existing fixed-lambda conservative Storey correction on that same vector;
* fixed-lambda Storey estimation on the first vector with rejection decisions
  made only from a distinct confirmation vector.

The first two procedures receive identical p-values and truth labels in every
replication, preserving common random numbers for their direct comparison. The
split-confirmation procedure reuses that first vector only for its selection
estimate and receives an independent deterministic confirmation stream for its
decisions and operating-characteristic scoring.

All reported bands are finite Monte-Carlo precision summaries. This review does
not create a universal FDR theorem, a promotion decision, or a replacement for
the evidence regime declared elsewhere in the package.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, cast

import numpy as np

from ._types import TwoGroupsConfig
from .corrections import (
    storey_qvalue,
    storey_qvalue_conservative,
    storey_qvalue_split_confirmation,
)
from .error_rates import classify
from .simulation import generate_pvalues
from .uncertainty import summarize_bounded


STOREY_METHOD_REVIEW_STATUS = "review_only_not_for_promotion_or_release"
# The active review lane uses neutral terminology. Preserved predecessor
# receipts retain their predecessor identifiers as hash-bound evidence, but new
# review artifacts must not reintroduce that terminology.
SAME_VECTOR_STOREY_PLUGIN = "same_vector_storey_plugin"
CONSERVATIVE_SAME_VECTOR_STOREY = "frozen_fixed_lambda_conservative_same_vector_storey"
SPLIT_CONFIRMATION_STOREY = "independent_split_confirmation_storey"

_METHOD_CONTRACTS: dict[str, dict[str, str]] = {
    SAME_VECTOR_STOREY_PLUGIN: {
        "method_label": "Same-vector Storey plug-in (diagnostic)",
        "correction": "storey_qvalue",
        "selection_stream": "same_vector",
        "decision_stream": "same_vector",
        "common_random_numbers_group": "same_vector_storey_comparison",
    },
    CONSERVATIVE_SAME_VECTOR_STOREY: {
        "method_label": "Frozen fixed-lambda conservative same-vector Storey",
        "correction": "storey_qvalue_conservative",
        "selection_stream": "same_vector",
        "decision_stream": "same_vector",
        "common_random_numbers_group": "same_vector_storey_comparison",
    },
    SPLIT_CONFIRMATION_STOREY: {
        "method_label": "Independent split-confirmation Storey",
        "correction": "storey_qvalue_split_confirmation",
        "selection_stream": "same_vector",
        "decision_stream": "independent_confirmation",
        "common_random_numbers_group": "same_vector_storey_comparison",
    },
}

_METRICS = ("fdr", "fwer", "power")


def _canonical_hash(value: Any) -> str:
    """Return the stable SHA-256 digest of JSON-safe contract data."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _positive_integer(value: object, name: str) -> int:
    """Normalize a positive integral public plan value."""
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
    """Normalize a non-negative integral public plan value."""
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
    """Normalize a finite real-valued public plan value."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _unit_interval(value: object, name: str) -> float:
    """Validate a finite probability-like emitted value."""
    normalized = _finite_float(value, name)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return normalized


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    """Return one JSON-style mapping or fail closed."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _require_digest(value: object, name: str) -> str:
    """Validate a lowercase SHA-256 witness."""
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


@dataclass(frozen=True)
class StoreyMethodReviewPlan:
    """Frozen design for the non-promotional three-way Storey review."""

    pi0: float = 0.5
    m: int = 200
    effect: float = 3.0
    alpha: float = 0.05
    lambda_: float = 0.5
    confidence: float = 0.99
    n_reps: int = 7_500
    seed: int = 0

    def __post_init__(self) -> None:
        pi0 = _unit_interval(self.pi0, "pi0")
        m = _positive_integer(self.m, "m")
        effect = _finite_float(self.effect, "effect")
        alpha = _finite_float(self.alpha, "alpha")
        lambda_ = _finite_float(self.lambda_, "lambda_")
        confidence = _finite_float(self.confidence, "confidence")
        n_reps = _positive_integer(self.n_reps, "n_reps")
        seed = _nonnegative_integer(self.seed, "seed")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 <= lambda_ < 1.0:
            raise ValueError("lambda_ must lie in [0, 1)")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")
        object.__setattr__(self, "pi0", pi0)
        object.__setattr__(self, "m", m)
        object.__setattr__(self, "effect", effect)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "lambda_", lambda_)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "n_reps", n_reps)
        object.__setattr__(self, "seed", seed)

    def to_mapping(self) -> dict[str, float | int | str]:
        """Return the fully resolved simulation contract."""
        return {
            "pi0": self.pi0,
            "m": self.m,
            "effect": self.effect,
            "alpha": self.alpha,
            "lambda": self.lambda_,
            "confidence": self.confidence,
            "n_reps": self.n_reps,
            "seed": self.seed,
            "dependence": "independent",
            "sided": 1,
        }

    @property
    def configuration_hash(self) -> str:
        """Return the stable identity of the frozen review plan."""
        return _canonical_hash(self.to_mapping())


DEFAULT_STOREY_METHOD_REVIEW_PLAN = StoreyMethodReviewPlan()


def _normalized_plan(plan: StoreyMethodReviewPlan | None) -> StoreyMethodReviewPlan:
    """Rebuild a plan before running in case low-level mutation occurred."""
    if plan is None:
        return DEFAULT_STOREY_METHOD_REVIEW_PLAN
    if not isinstance(plan, StoreyMethodReviewPlan):
        raise ValueError("plan must be a StoreyMethodReviewPlan")
    return StoreyMethodReviewPlan(
        pi0=plan.pi0,
        m=plan.m,
        effect=plan.effect,
        alpha=plan.alpha,
        lambda_=plan.lambda_,
        confidence=plan.confidence,
        n_reps=plan.n_reps,
        seed=plan.seed,
    )


def _stream_schedule(
    plan: StoreyMethodReviewPlan,
) -> tuple[list[np.random.SeedSequence], list[np.random.SeedSequence], dict[str, Any]]:
    """Materialize disjoint same-vector and confirmation branch schedules."""
    children = np.random.SeedSequence(plan.seed).spawn(2 * plan.n_reps)
    same_vector = children[::2]
    confirmation = children[1::2]
    same_keys = [[int(part) for part in child.spawn_key] for child in same_vector]
    confirmation_keys = [
        [int(part) for part in child.spawn_key] for child in confirmation
    ]
    if len(same_vector) != plan.n_reps or len(confirmation) != plan.n_reps:
        raise RuntimeError("Storey review stream schedule has an invalid branch count")
    if set(map(tuple, same_keys)).intersection(map(tuple, confirmation_keys)):
        raise RuntimeError("Storey review streams must be disjoint")
    body: dict[str, Any] = {
        "algorithm": "numpy.SeedSequence.spawn",
        "master_seed": plan.seed,
        "n_reps": plan.n_reps,
        "total_child_streams": 2 * plan.n_reps,
        "same_vector": {
            "child_index_first": 0,
            "child_index_stride": 2,
            "branch_spawn_keys_sha256": _canonical_hash(same_keys),
            "used_by": [
                SAME_VECTOR_STOREY_PLUGIN,
                CONSERVATIVE_SAME_VECTOR_STOREY,
                "split_confirmation_selection",
            ],
        },
        "independent_confirmation": {
            "child_index_first": 1,
            "child_index_stride": 2,
            "branch_spawn_keys_sha256": _canonical_hash(confirmation_keys),
            "used_by": [SPLIT_CONFIRMATION_STOREY],
        },
        "same_vector_methods_share_draws": True,
        "selection_and_confirmation_streams_disjoint": True,
    }
    return same_vector, confirmation, {
        **body,
        "schedule_sha256": _canonical_hash(body),
    }


def _update_draw_digest(
    digest: Any, pvalues: np.ndarray, is_alt: np.ndarray
) -> None:
    """Append one canonical simulated vector and truth mask to a digest."""
    digest.update(np.asarray(pvalues, dtype="<f8").tobytes())
    digest.update(np.asarray(is_alt, dtype=np.uint8).tobytes())


def _metric_summary(values: list[float], confidence: float) -> dict[str, Any]:
    """Return bounded finite-simulation estimates and visible upper bands."""
    summary = summarize_bounded(np.asarray(values, dtype=float), confidence=confidence)
    return {
        "n": int(summary["n"]),
        "estimate": float(summary["mean"]),
        "standard_error": float(summary["standard_error"]),
        "confidence_lower_band": float(summary["normal_ci_lower"]),
        "confidence_upper_band": float(summary["normal_ci_upper"]),
        "confidence": float(summary["confidence"]),
        "interval_method": "student_t_monte_carlo",
        "hoeffding_radius": float(summary["hoeffding_radius"]),
    }


def _mean_stderr(values: list[float]) -> float:
    """Compute the ordinary Monte-Carlo standard error for finite counts."""
    sample = np.asarray(values, dtype=float)
    if sample.shape[0] == 1:
        return 0.0
    return float(np.std(sample, ddof=1) / math.sqrt(sample.shape[0]))


def _method_result(
    method_id: str,
    samples: Mapping[str, list[float]],
    plan: StoreyMethodReviewPlan,
) -> dict[str, Any]:
    """Finalize one method's immutable, hash-bound summary."""
    contract = _METHOD_CONTRACTS[method_id]
    rejections = samples["rejections"]
    body: dict[str, Any] = {
        "method_id": method_id,
        **contract,
        "metrics": {
            metric: _metric_summary(samples[metric], plan.confidence)
            for metric in _METRICS
        },
        "mean_rejections": float(np.mean(np.asarray(rejections, dtype=float))),
        "mean_rejections_standard_error": _mean_stderr(rejections),
    }
    return {**body, "result_sha256": _canonical_hash(body)}


def evaluate_storey_method_review(
    plan: StoreyMethodReviewPlan | None = None,
) -> dict[str, Any]:
    """Run the frozen three-way Storey simulation and return an audit payload.

    The default plan is the requested ``pi0=0.5``, ``m=200``, ``effect=3``,
    ``alpha=0.05``, ``lambda=0.5``, ``confidence=0.99``, ``n_reps=7500``, and
    ``seed=0`` review. Callers may pass a smaller real plan for test execution,
    but the command-line artifact writer exposes no tuning arguments.
    """
    active_plan = _normalized_plan(plan)
    config = TwoGroupsConfig(
        m=active_plan.m,
        pi0=active_plan.pi0,
        effect=active_plan.effect,
        rho=0.0,
        sided=1,
        seed=active_plan.seed,
        dependence_kind="independent",
    )
    same_branches, confirmation_branches, schedule = _stream_schedule(active_plan)
    samples: dict[str, dict[str, list[float]]] = {
        method_id: {metric: [] for metric in (*_METRICS, "rejections")}
        for method_id in _METHOD_CONTRACTS
    }
    same_vector_digest = hashlib.sha256()
    confirmation_digest = hashlib.sha256()

    for same_branch, confirmation_branch in zip(same_branches, confirmation_branches):
        same_pvalues, same_is_alt = generate_pvalues(
            config, np.random.default_rng(same_branch)
        )
        confirmation_pvalues, confirmation_is_alt = generate_pvalues(
            config, np.random.default_rng(confirmation_branch)
        )
        _update_draw_digest(same_vector_digest, same_pvalues, same_is_alt)
        _update_draw_digest(
            confirmation_digest, confirmation_pvalues, confirmation_is_alt
        )
        decisions = {
            SAME_VECTOR_STOREY_PLUGIN: (
                storey_qvalue(
                    same_pvalues, active_plan.alpha, lambda_=active_plan.lambda_
                ),
                same_is_alt,
            ),
            CONSERVATIVE_SAME_VECTOR_STOREY: (
                storey_qvalue_conservative(
                    same_pvalues,
                    active_plan.alpha,
                    lambda_=active_plan.lambda_,
                    confidence=active_plan.confidence,
                ),
                same_is_alt,
            ),
            SPLIT_CONFIRMATION_STOREY: (
                storey_qvalue_split_confirmation(
                    same_pvalues,
                    confirmation_pvalues,
                    active_plan.alpha,
                    lambda_=active_plan.lambda_,
                    confidence=active_plan.confidence,
                ),
                confirmation_is_alt,
            ),
        }
        for method_id, (decision, truth) in decisions.items():
            summary = classify(decision.rejected, truth)
            samples[method_id]["fdr"].append(float(summary.fdp))
            samples[method_id]["fwer"].append(float(summary.fwer_event))
            samples[method_id]["power"].append(float(summary.power))
            samples[method_id]["rejections"].append(float(summary.num_rejected))

    methods = {
        method_id: _method_result(method_id, method_samples, active_plan)
        for method_id, method_samples in samples.items()
    }
    stream_evidence_body: dict[str, Any] = {
        "same_vector_draw_sha256": same_vector_digest.hexdigest(),
        "confirmation_draw_sha256": confirmation_digest.hexdigest(),
        "same_vector_methods": [
            SAME_VECTOR_STOREY_PLUGIN,
            CONSERVATIVE_SAME_VECTOR_STOREY,
        ],
        "split_selection_stream": "same_vector",
        "split_decision_stream": "independent_confirmation",
        "same_vector_and_confirmation_draws_differ": (
            same_vector_digest.hexdigest() != confirmation_digest.hexdigest()
        ),
    }
    stream_evidence = {
        **stream_evidence_body,
        "stream_evidence_sha256": _canonical_hash(stream_evidence_body),
    }
    body: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "three_way_storey_method_review",
        "status": STOREY_METHOD_REVIEW_STATUS,
        "review_plan": active_plan.to_mapping(),
        "plan_sha256": active_plan.configuration_hash,
        "stream_schedule": schedule,
        "schedule_sha256": schedule["schedule_sha256"],
        "stream_evidence": stream_evidence,
        "methods": methods,
        "assumptions": [
            "independent_two_groups_pvalue_draws",
            "fixed_lambda_and_confidence_declared_before_simulation",
            "same_vector_methods_share_identical_draws",
            "split_confirmation_decisions_use_a_distinct_seedsequence_stream",
            "marginally_superuniform_null_pvalues_under_the_declared_model",
        ],
        "nonclaims": [
            "Same-vector Storey plug-in is displayed for diagnostic method comparison only.",
            "Finite Monte-Carlo bands do not establish universal FDR or FWER control.",
            "The review does not change certificate policy, release status, or publication readiness.",
            "Independent split confirmation is evaluated only under the declared simulation assumptions.",
        ],
        "claim_boundary": (
            "Three finite-simulation operating-characteristic summaries under one "
            "frozen independent two-groups design; no promotion, theorem, or "
            "arbitrary-dependence claim is produced."
        ),
    }
    return {**body, "result_sha256": _canonical_hash(body)}


def _validate_metric_summary(
    value: object, plan: StoreyMethodReviewPlan, name: str
) -> None:
    """Verify one emitted bounded metric summary including its precision band."""
    metric = _require_mapping(value, name)
    expected_keys = {
        "n",
        "estimate",
        "standard_error",
        "confidence_lower_band",
        "confidence_upper_band",
        "confidence",
        "interval_method",
        "hoeffding_radius",
    }
    if set(metric) != expected_keys:
        raise ValueError(f"{name} has an unexpected metric-summary schema")
    if _positive_integer(metric["n"], f"{name}.n") != plan.n_reps:
        raise ValueError(f"{name}.n does not match review_plan.n_reps")
    estimate = _unit_interval(metric["estimate"], f"{name}.estimate")
    lower = _unit_interval(metric["confidence_lower_band"], f"{name}.lower")
    upper = _unit_interval(metric["confidence_upper_band"], f"{name}.upper")
    if not lower <= estimate <= upper:
        raise ValueError(f"{name} confidence band does not contain its estimate")
    standard_error = _finite_float(metric["standard_error"], f"{name}.standard_error")
    radius = _finite_float(metric["hoeffding_radius"], f"{name}.hoeffding_radius")
    if standard_error < 0.0 or radius < 0.0:
        raise ValueError(f"{name} uncertainty values must be non-negative")
    confidence = _finite_float(metric["confidence"], f"{name}.confidence")
    if confidence != plan.confidence:
        raise ValueError(f"{name}.confidence does not match review_plan.confidence")
    if metric["interval_method"] != "student_t_monte_carlo":
        raise ValueError(f"{name}.interval_method is invalid")


def _validate_method_result(
    method_id: str, value: object, plan: StoreyMethodReviewPlan
) -> None:
    """Validate one complete, hash-bound method payload."""
    method = _require_mapping(value, f"methods.{method_id}")
    expected_keys = {
        "method_id",
        "method_label",
        "correction",
        "selection_stream",
        "decision_stream",
        "common_random_numbers_group",
        "metrics",
        "mean_rejections",
        "mean_rejections_standard_error",
        "result_sha256",
    }
    if set(method) != expected_keys:
        raise ValueError(f"methods.{method_id} has an unexpected schema")
    expected_contract = _METHOD_CONTRACTS[method_id]
    if method["method_id"] != method_id:
        raise ValueError(f"methods.{method_id}.method_id is invalid")
    for field, expected in expected_contract.items():
        if method[field] != expected:
            raise ValueError(f"methods.{method_id}.{field} is invalid")
    metrics = _require_mapping(method["metrics"], f"methods.{method_id}.metrics")
    if set(metrics) != set(_METRICS):
        raise ValueError(f"methods.{method_id}.metrics must contain FDR, FWER, and power")
    for metric_name in _METRICS:
        _validate_metric_summary(
            metrics[metric_name], plan, f"methods.{method_id}.metrics.{metric_name}"
        )
    mean_rejections = _finite_float(
        method["mean_rejections"], f"methods.{method_id}.mean_rejections"
    )
    mean_rejections_stderr = _finite_float(
        method["mean_rejections_standard_error"],
        f"methods.{method_id}.mean_rejections_standard_error",
    )
    if not 0.0 <= mean_rejections <= plan.m or mean_rejections_stderr < 0.0:
        raise ValueError(f"methods.{method_id} has invalid rejection accounting")
    body = dict(method)
    recorded_hash = _require_digest(body.pop("result_sha256"), f"methods.{method_id}.result_sha256")
    if recorded_hash != _canonical_hash(body):
        raise ValueError(f"methods.{method_id} result hash does not match its payload")


def _plan_from_mapping(value: object) -> StoreyMethodReviewPlan:
    """Reconstruct one exact serialized review plan."""
    mapping = _require_mapping(value, "review_plan")
    expected_keys = {
        "pi0",
        "m",
        "effect",
        "alpha",
        "lambda",
        "confidence",
        "n_reps",
        "seed",
        "dependence",
        "sided",
    }
    if set(mapping) != expected_keys:
        raise ValueError("review_plan has an unexpected schema")
    if mapping["dependence"] != "independent" or mapping["sided"] != 1:
        raise ValueError("review_plan must retain the frozen independent one-sided setting")
    return StoreyMethodReviewPlan(
        pi0=cast(float, mapping["pi0"]),
        m=cast(int, mapping["m"]),
        effect=cast(float, mapping["effect"]),
        alpha=cast(float, mapping["alpha"]),
        lambda_=cast(float, mapping["lambda"]),
        confidence=cast(float, mapping["confidence"]),
        n_reps=cast(int, mapping["n_reps"]),
        seed=cast(int, mapping["seed"]),
    )


def validate_storey_method_review(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless a serialized review retains all hashes and boundaries."""
    review = _require_mapping(value, "storey_method_review")
    expected_keys = {
        "schema_version",
        "kind",
        "status",
        "review_plan",
        "plan_sha256",
        "stream_schedule",
        "schedule_sha256",
        "stream_evidence",
        "methods",
        "assumptions",
        "nonclaims",
        "claim_boundary",
        "result_sha256",
    }
    if set(review) != expected_keys:
        raise ValueError("storey_method_review has an unexpected schema")
    if review["schema_version"] != "1.0" or review["kind"] != "three_way_storey_method_review":
        raise ValueError("storey_method_review kind or schema_version is invalid")
    if review["status"] != STOREY_METHOD_REVIEW_STATUS:
        raise ValueError("storey_method_review status must remain review-only")
    plan = _plan_from_mapping(review["review_plan"])
    if review["plan_sha256"] != plan.configuration_hash:
        raise ValueError("review_plan hash does not match its payload")
    _same, _confirmation, expected_schedule = _stream_schedule(plan)
    if review["stream_schedule"] != expected_schedule:
        raise ValueError("stream_schedule does not match the frozen review plan")
    if review["schedule_sha256"] != expected_schedule["schedule_sha256"]:
        raise ValueError("schedule_sha256 does not match stream_schedule")
    stream_evidence = _require_mapping(review["stream_evidence"], "stream_evidence")
    expected_stream_evidence_keys = {
        "same_vector_draw_sha256",
        "confirmation_draw_sha256",
        "same_vector_methods",
        "split_selection_stream",
        "split_decision_stream",
        "same_vector_and_confirmation_draws_differ",
        "stream_evidence_sha256",
    }
    if set(stream_evidence) != expected_stream_evidence_keys:
        raise ValueError("stream_evidence has an unexpected schema")
    for key in ("same_vector_draw_sha256", "confirmation_draw_sha256"):
        _require_digest(stream_evidence[key], f"stream_evidence.{key}")
    if stream_evidence["same_vector_draw_sha256"] == stream_evidence["confirmation_draw_sha256"]:
        raise ValueError("same-vector and confirmation draw witnesses must differ")
    if stream_evidence["same_vector_methods"] != [
        SAME_VECTOR_STOREY_PLUGIN,
        CONSERVATIVE_SAME_VECTOR_STOREY,
    ]:
        raise ValueError("stream_evidence must name the same-vector comparison methods")
    if stream_evidence["split_selection_stream"] != "same_vector":
        raise ValueError("split selection stream is invalid")
    if stream_evidence["split_decision_stream"] != "independent_confirmation":
        raise ValueError("split decision stream is invalid")
    if stream_evidence["same_vector_and_confirmation_draws_differ"] is not True:
        raise ValueError("stream_evidence must record distinct confirmation draws")
    stream_evidence_body = dict(stream_evidence)
    stream_evidence_hash = _require_digest(
        stream_evidence_body.pop("stream_evidence_sha256"),
        "stream_evidence.stream_evidence_sha256",
    )
    if stream_evidence_hash != _canonical_hash(stream_evidence_body):
        raise ValueError("stream_evidence hash does not match its payload")
    methods = _require_mapping(review["methods"], "methods")
    if set(methods) != set(_METHOD_CONTRACTS):
        raise ValueError("methods must contain the frozen three-way comparison")
    for method_id in _METHOD_CONTRACTS:
        _validate_method_result(method_id, methods[method_id], plan)
    if not isinstance(review["assumptions"], list) or not review["assumptions"]:
        raise ValueError("assumptions must be a non-empty list")
    if not isinstance(review["nonclaims"], list) or not review["nonclaims"]:
        raise ValueError("nonclaims must be a non-empty list")
    if not isinstance(review["claim_boundary"], str) or not review["claim_boundary"]:
        raise ValueError("claim_boundary must be a non-empty string")
    body = dict(review)
    result_hash = _require_digest(body.pop("result_sha256"), "result_sha256")
    if result_hash != _canonical_hash(body):
        raise ValueError("result_sha256 does not match the review payload")
    return dict(review)


__all__ = [
    "STOREY_METHOD_REVIEW_STATUS",
    "SAME_VECTOR_STOREY_PLUGIN",
    "CONSERVATIVE_SAME_VECTOR_STOREY",
    "SPLIT_CONFIRMATION_STOREY",
    "StoreyMethodReviewPlan",
    "DEFAULT_STOREY_METHOD_REVIEW_PLAN",
    "evaluate_storey_method_review",
    "validate_storey_method_review",
]
