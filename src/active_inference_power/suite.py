"""High-level suite orchestration for conditional Active Inference power."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .action_loop import (
    ActionPolicy,
    evaluate_power_surface,
    run_action_loop,
    simulate_action_loop,
)
from .action_statistics import paired_action_policy_comparisons
from .online_fdr import lord_plus_plus, saffron
from .scenario import (
    PowerScenario,
    make_reference_action_scenario,
    make_static_binary_split_confirmation_scenario,
)
from .selection import simulate_split_stream_confirmation
from .uncertainty import summarize_bounded, wilson_interval


@dataclass(frozen=True)
class PowerSuiteResult:
    """Machine-readable result for a scenario/policy power suite run.

    ``scenario_contracts`` covers only the operating scenario grid.  A
    split-stream confirmation can use a separately declared static channel;
    its full scenario contract is carried in ``split_confirmation_scenario``
    rather than being presented as another operating or frontier row.  Both
    fields are required so every public serialization carries the complete
    ``1.0`` lineage contract.
    """

    suite_name: str
    scenarios: tuple[str, ...]
    policies: tuple[str, ...]
    operating_characteristics: tuple[dict[str, Any], ...]
    online_fdr: dict[str, Any]
    claim_boundary: str
    split_stream_confirmation: dict[str, Any]
    scenario_contracts: tuple[dict[str, Any], ...]
    split_confirmation_scenario: dict[str, Any]
    paired_policy_comparisons: tuple[dict[str, Any], ...] = ()
    power_cost_frontier: tuple[dict[str, Any], ...] = ()
    future_evidence_selection_failure: dict[str, Any] | None = None
    learned_policy: dict[str, Any] | None = None

    def to_mapping(self) -> dict[str, Any]:
        """Serialize a suite with its complete current lineage contract."""
        if not self.scenario_contracts:
            raise ValueError("a suite contract requires operating scenario contracts")

        mapping: dict[str, Any] = {
            "suite_name": self.suite_name,
            "scenarios": list(self.scenarios),
            "policies": list(self.policies),
            "operating_characteristics": list(self.operating_characteristics),
            "online_fdr": self.online_fdr,
            "claim_boundary": self.claim_boundary,
            "split_stream_confirmation": self.split_stream_confirmation,
            "paired_policy_comparisons": list(self.paired_policy_comparisons),
            "power_cost_frontier": list(self.power_cost_frontier),
            **(
                {"future_evidence_selection_failure": self.future_evidence_selection_failure}
                if self.future_evidence_selection_failure is not None
                else {}
            ),
            **(
                {"learned_policy": self.learned_policy}
                if self.learned_policy is not None
                else {}
            ),
        }
        operating_contracts = list(self.scenario_contracts)
        if any(not isinstance(contract, Mapping) for contract in operating_contracts):
            raise ValueError("scenario_contracts must contain mappings")
        contract_ids = [str(contract.get("scenario_id", "")) for contract in operating_contracts]
        if tuple(contract_ids) != self.scenarios:
            raise ValueError(
                "scenario_contracts must match the declared operating scenario order"
            )
        split_contract = self.split_confirmation_scenario
        if not isinstance(split_contract, Mapping):
            raise ValueError("split_confirmation_scenario must be a mapping")
        split_scenario_id = str(split_contract.get("scenario_id", ""))
        split_scenario_hash = str(split_contract.get("scenario_hash", ""))
        if not split_scenario_id or split_scenario_id in self.scenarios:
            raise ValueError(
                "split_confirmation_scenario must be declared outside the operating grid"
            )
        split_summary = self.split_stream_confirmation
        if not isinstance(split_summary, Mapping):
            raise ValueError("split_stream_confirmation must be a mapping")
        if (
            str(split_summary.get("scenario_id", "")) != split_scenario_id
            or str(split_summary.get("scenario_hash", "")) != split_scenario_hash
        ):
            raise ValueError(
                "split_stream_confirmation must match its declared scenario contract"
            )
        return {
            "contract_version": "1.0",
            "scenario_contracts": operating_contracts,
            "split_confirmation_scenario": dict(split_contract),
            **mapping,
        }


def _static_split_confirmation_scenario(
    operating_scenario: PowerScenario,
    operating_scenario_ids: set[str],
) -> PowerScenario:
    """Build the dedicated static channel used only for split confirmation.

    A public suite may receive arbitrary operating scenarios.  The default
    split lane therefore uses the dedicated canonical split-confirmation
    channel, anchors its horizon and alpha to the first declared operating
    setting, and uses the first scenario's declared reliability when that
    metadata is available. The factory supplies the dedicated formal
    split-confirmation assumptions. Its canonical identifier is retained unless
    an operating caller has already used it, in which case a deterministic
    suffix keeps the split lane outside the grid.
    """
    metadata = dict(operating_scenario.metadata or {})
    raw_reliability = metadata.get("reliability", 0.8)
    try:
        reliability = float(raw_reliability)
    except (TypeError, ValueError):
        reliability = 0.8
    if (
        isinstance(raw_reliability, bool)
        or not math.isfinite(reliability)
        or not 0.5 < reliability < 1.0
    ):
        reliability = 0.8

    static_scenario = make_static_binary_split_confirmation_scenario(
        reliability=reliability,
        horizon=operating_scenario.setting.max_steps,
        alpha=operating_scenario.setting.alpha,
    )
    scenario_id = static_scenario.scenario_id
    if scenario_id not in operating_scenario_ids:
        return static_scenario
    scenario_id = f"{scenario_id}__suite__{operating_scenario.configuration_hash[:12]}"
    while scenario_id in operating_scenario_ids:
        scenario_id = f"{scenario_id}_static"
    return replace(
        static_scenario,
        scenario_id=scenario_id,
        metadata={
            **dict(static_scenario.metadata or {}),
            "suite_role": "split_confirmation_only",
            "separate_from_operating_grid": True,
        },
    )


def _selection_calibration_diagnostic(
    values: Sequence[float], confidence: float
) -> dict[str, Any]:
    """Summarize one selected-p-value sample against its DKW radius."""
    sample = np.sort(np.asarray(values, dtype=float))
    n_selected = int(sample.size)
    if not n_selected:
        return {
            "n_selected": 0,
            "max_ecdf_excess": None,
            "dkw_radius": None,
            "passes_dkw_diagnostic": False,
            "dkw_interpretation": "nominal_iid_reference_only",
        }
    excess = float(np.max(np.arange(1, n_selected + 1) / n_selected - sample))
    radius = float(np.sqrt(np.log(2.0 / (1.0 - confidence)) / (2.0 * n_selected)))
    return {
        "n_selected": n_selected,
        "max_ecdf_excess": max(0.0, excess),
        "dkw_radius": radius,
        "passes_dkw_diagnostic": bool(excess <= radius),
        "dkw_interpretation": "nominal_iid_reference_only; serial dependence is not controlled",
    }


def evaluate_conditional_selection_calibration(
    scenario: PowerScenario,
    policy: ActionPolicy,
    n_reps: int,
    *,
    seed: int = 0,
    workers: int = 1,
) -> dict[str, Any]:
    """Diagnose selected-target p-value calibration in an all-null process.

    For each target, the empirical CDF excess ``max(F_n(p)-p)`` is compared
    with a DKW radius.  This is a finite-simulation diagnostic for the exact
    model/process/policy composition; it is not a selective-inference proof,
    and a pass cannot justify arbitrary adaptive selection.
    """
    if np.any(np.asarray(scenario.process.target_truth, dtype=bool)):
        raise ValueError("conditional selection calibration requires an all-null process")
    if isinstance(n_reps, bool) or int(n_reps) != n_reps or n_reps < 1:
        raise ValueError("n_reps must be a positive integer")
    simulation = simulate_action_loop(
        scenario,
        policy,
        int(n_reps),
        seed=seed,
        workers=workers,
        retain_traces=int(n_reps),
    )
    selected_by_target: list[list[float]] = [
        [] for _ in range(scenario.setting.family_size)
    ]
    for trace in simulation["traces"]:
        for step in trace["steps"]:
            target = int(step["target"])
            if 0 <= target < scenario.setting.family_size:
                selected_by_target[target].append(float(step["p_value"]))

    def diagnostic(values: list[float]) -> dict[str, Any]:
        return _selection_calibration_diagnostic(values, scenario.setting.confidence) | {
            "validity": "calibration_diagnostic"
        }

    per_target = [diagnostic(values) for values in selected_by_target]
    pooled = diagnostic([value for values in selected_by_target for value in values])
    return {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "policy": policy,
        "n_reps": int(n_reps),
        "seed": int(seed),
        "selected_pvalues_by_target": per_target,
        "pooled": pooled,
        "confidence": scenario.setting.confidence,
        "selection_definition": "one p-value per target-selecting action step, in chronological order",
        "claim_boundary": "All-null DKW calibration diagnostic for the declared process and policy; not a selective-inference proof or arbitrary adaptive-selection guarantee.",
        "validity": "calibration_diagnostic",
    }


def evaluate_future_evidence_selection_failure(
    scenario: PowerScenario,
    n_reps: int,
    *,
    seed: int = 0,
    workers: int = 1,
) -> dict[str, Any]:
    """Construct a deliberately invalid post-hoc selection contrast.

    The selector sees the complete fixed-horizon trace and reports the smallest
    target-step p-value.  This is intentionally not a policy-valid analysis:
    the selected target is chosen using future evidence.  Its purpose is to
    make the Stage 1 failure mode measurable and visible beside the
    chronological calibration diagnostic.
    """
    if np.any(np.asarray(scenario.process.target_truth, dtype=bool)):
        raise ValueError("future-evidence selection contrast requires an all-null process")
    if isinstance(n_reps, bool) or int(n_reps) != n_reps or n_reps < 1:
        raise ValueError("n_reps must be a positive integer")
    simulation = simulate_action_loop(
        scenario,
        "fixed_horizon",
        int(n_reps),
        seed=seed,
        workers=workers,
        retain_traces=int(n_reps),
    )
    selected_pvalues: list[float] = []
    selected_targets: list[int] = []
    for trace in simulation["traces"]:
        candidates = [
            (float(step["p_value"]), int(step["target"]))
            for step in trace["steps"]
            if int(step["target"]) >= 0
        ]
        if candidates:
            selected, target = min(candidates, key=lambda item: (item[0], item[1]))
            selected_pvalues.append(selected)
            selected_targets.append(target)
    pooled = _selection_calibration_diagnostic(
        selected_pvalues, scenario.setting.confidence
    )
    pooled["validity"] = "selection_failure_contrast"
    return {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.configuration_hash,
        "policy": "future_evidence_min_p_selector",
        "n_reps": int(n_reps),
        "selected_pvalues": selected_pvalues,
        "selected_targets": selected_targets,
        "pooled": pooled,
        "confidence": scenario.setting.confidence,
        "selection_definition": "post-hoc target selected by the minimum p-value over the complete fixed-horizon trace",
        "claim_boundary": "Deliberately invalid future-evidence selection contrast; selection uses information unavailable at the testing time and cannot support online-FDR or selective-inference validity.",
        "validity": "selection_failure_contrast",
    }


def evaluate_adaptive_target_selection(
    scenario: PowerScenario,
    policy: ActionPolicy,
    n_reps: int,
    *,
    seed: int = 0,
    workers: int = 1,
    method: str = "lord_plus_plus",
) -> dict[str, Any]:
    """Apply chronological online FDR accounting to selected-target evidence."""
    if method not in {"lord_plus_plus", "saffron"}:
        raise ValueError("method must be lord_plus_plus or saffron")
    if isinstance(n_reps, bool) or int(n_reps) != n_reps or n_reps < 1:
        raise ValueError("n_reps must be a positive integer")
    simulation = simulate_action_loop(
        scenario, policy, int(n_reps), seed=seed, workers=workers, retain_traces=int(n_reps)
    )
    rejection_counts: list[int] = []
    any_rejections: list[float] = []
    false_discovery_proportions: list[float] = []
    powers: list[float] = []
    test_powers: list[float] = []
    target_fdrs: list[float] = []
    family_errors: list[float] = []
    target_ids: list[list[int]] = []
    valid_selection_targets: list[list[int]] = []
    representative: dict[str, Any] | None = None
    for trace in simulation["traces"]:
        steps = trace["steps"]
        pvalues = np.asarray([step["p_value"] for step in steps], dtype=float)
        trace_targets = [int(step["target"]) for step in steps]
        target_ids.append(trace_targets)
        valid_selection_targets.append([target for target in trace_targets if target >= 0])
        if pvalues.size == 0:
            # A policy may stop before collecting evidence.  It remains a
            # valid zero-rejection replication rather than disappearing from
            # the denominator of the operating characteristic.
            rejection_counts.append(0)
            any_rejections.append(0.0)
            false_discovery_proportions.append(0.0)
            powers.append(0.0)
            test_powers.append(0.0)
            target_fdrs.append(0.0)
            family_errors.append(0.0)
            continue
        result = lord_plus_plus(pvalues, scenario.setting.alpha) if method == "lord_plus_plus" else saffron(pvalues, scenario.setting.alpha)
        rejection_counts.append(result.num_rejections)
        any_rejections.append(float(result.num_rejections > 0))
        truth = np.asarray(scenario.process.target_truth, dtype=bool)
        submitted_target_ids = np.asarray(target_ids[-1], dtype=int)
        valid_submissions = (submitted_target_ids >= 0) & (submitted_target_ids < truth.size)
        submitted_truth = truth[submitted_target_ids[valid_submissions]]
        rejected_target_ids = submitted_target_ids[result.rejections & valid_submissions]
        test_true_rejections = int(np.count_nonzero(truth[rejected_target_ids])) if rejected_target_ids.size else 0
        test_false_rejections = int(rejected_target_ids.size - test_true_rejections)
        unique_rejected_target_ids = np.unique(rejected_target_ids)
        unique_truth = truth[unique_rejected_target_ids] if unique_rejected_target_ids.size else np.asarray([], dtype=bool)
        unique_true_rejections = int(np.count_nonzero(unique_truth))
        unique_false_rejections = int(unique_rejected_target_ids.size - unique_true_rejections)
        false_discovery_proportions.append(float(test_false_rejections / max(1, rejected_target_ids.size)))
        target_fdrs.append(float(unique_false_rejections / max(1, unique_rejected_target_ids.size)))
        powers.append(float(unique_true_rejections / max(1, int(np.count_nonzero(truth)))))
        test_powers.append(float(test_true_rejections / max(1, int(np.count_nonzero(submitted_truth)))))
        family_errors.append(float(unique_false_rejections > 0))
        if representative is None:
            representative = {
                "pvalues": pvalues.tolist(),
                "levels": result.levels.tolist(),
                "candidates": result.candidates.tolist(),
                "rejections": result.rejections.tolist(),
                "accounting_path": result.wealth.tolist(),
                "accounting_kind": "procedural_alpha_wealth",
                "target_ids": list(target_ids[-1]),
                "stopping_time": int(trace["stopping_time"]),
            }
    if not rejection_counts or representative is None:
        raise ValueError("action loop did not produce any target evidence")
    def mean_interval(values: list[float]) -> list[float]:
        interval = summarize_bounded(values, scenario.setting.confidence)
        return [float(interval["normal_ci_lower"]), float(interval["normal_ci_upper"])]

    return {
        "method": method,
        "scenario_id": scenario.scenario_id,
        "n_reps": len(rejection_counts),
        "mean_rejections": float(np.mean(rejection_counts)),
        "any_rejection_rate": float(np.mean(any_rejections)),
        "any_rejection_interval": list(wilson_interval(int(np.sum(any_rejections)), len(any_rejections), scenario.setting.confidence)),
        "mean_fdr": float(np.mean(false_discovery_proportions)),
        "mean_fdr_interval": mean_interval(false_discovery_proportions),
        "mean_power": float(np.mean(powers)),
        "mean_power_interval": mean_interval(powers),
        "mean_test_power": float(np.mean(test_powers)),
        "mean_test_power_interval": mean_interval(test_powers),
        "mean_target_fdr": float(np.mean(target_fdrs)),
        "mean_target_fdr_interval": mean_interval(target_fdrs),
        "fwer": float(np.mean(family_errors)),
        "fwer_interval": list(wilson_interval(int(np.sum(family_errors)), len(family_errors), scenario.setting.confidence)),
        "power_estimand": "unique_target_family_true_positive_rate",
        "test_power_estimand": "per_submitted_target_true_positive_rate",
        "fdr_estimand": "per_online_test_false_discovery_proportion",
        "target_fdr_estimand": "unique_rejected_target_false_discovery_proportion",
        "fwer_estimand": "any_false_unique_target_rejection",
        "selection_count_mean": float(np.mean([len(ids) for ids in valid_selection_targets])),
        "selection_counts_by_target": np.bincount(
            np.asarray([target for ids in valid_selection_targets for target in ids], dtype=int),
            minlength=scenario.setting.family_size,
        ).astype(int).tolist(),
        "representative": representative,
        "validity": "conditional online-FDR diagnostic; requires predictable target selection and conditional super-uniform evidence",
    }


def evaluate_split_stream_confirmation(
    scenario: PowerScenario,
    n_reps: int,
    *,
    seed: int = 0,
    workers: int = 1,
) -> dict[str, Any]:
    """Evaluate independent selection/confirmation streams.

    This is the primary Stage 1 pathway.  Its formal label is conditional on
    the result's explicit static-channel assumptions; richer environment
    configurations remain diagnostic operating characteristics.
    """
    return simulate_split_stream_confirmation(scenario, n_reps, seed=seed, workers=workers)


def run_power_suite(
    scenarios: Sequence[PowerScenario],
    policies: Sequence[ActionPolicy],
    *,
    n_reps: int = 1000,
    seed: int = 0,
    workers: int = 1,
    online_fdr_policy: ActionPolicy = "adaptive_target",
) -> PowerSuiteResult:
    """Run the composable scenario/policy suite with stable child streams.

    Operating rows are evaluated only for ``scenarios``.  Split confirmation
    receives a separately declared static channel, so no result can imply that
    confirmation was evaluated on an arbitrary operating scenario or that it
    belongs on the power/cost frontier.
    """
    operating_scenarios = tuple(scenarios)
    if not operating_scenarios or not policies:
        raise ValueError("scenarios and policies must be non-empty")
    if isinstance(n_reps, bool) or int(n_reps) != n_reps or n_reps < 1:
        raise ValueError("n_reps must be a positive integer")
    declared_reps = int(n_reps)
    operating = evaluate_power_surface(
        operating_scenarios, policies, declared_reps, seed=seed, workers=workers
    )
    paired = tuple(
        paired_action_policy_comparisons(
            scenario,
            policies,
            declared_reps,
            seed=int(seed) + index,
            run_trace=run_action_loop,
        )
        for index, scenario in enumerate(operating_scenarios)
    )
    frontier = tuple(
        {
            "scenario_id": item["scenario_id"],
            "scenario_hash": item["scenario_hash"],
            "scenario_metadata": dict(item.get("scenario_metadata", {})),
            "policy": item["policy"],
            "power": item["summary"]["power"],
            "power_interval": item["summary"]["power_interval"],
            "fdr": item["summary"]["fdr"],
            "fdr_interval": item["summary"].get("fdr_interval", [item["summary"]["fdr"]] * 2),
            "n_reps": item["summary"]["n_reps"],
            "mean_cost": item["summary"]["mean_cost"],
            "cost_interval": item["summary"].get("cost_interval", [item["summary"]["mean_cost"]] * 2),
            "mean_stopping_time": item["summary"]["mean_stopping_time"],
            "power_per_cost": item["summary"].get("power_per_cost", 0.0),
            "validity": item["summary"].get("validity", item.get("validity", "diagnostic")),
        }
        for item in operating
    )
    online = evaluate_adaptive_target_selection(
        operating_scenarios[0], online_fdr_policy, declared_reps, seed=seed + 1, workers=workers,
    )
    split_confirmation_scenario = _static_split_confirmation_scenario(
        operating_scenarios[0],
        {scenario.scenario_id for scenario in operating_scenarios},
    )
    split_stream = evaluate_split_stream_confirmation(
        split_confirmation_scenario,
        declared_reps,
        seed=seed + 2,
        workers=workers,
    )
    return PowerSuiteResult(
        suite_name="active_inference_power_suite",
        scenarios=tuple(scenario.scenario_id for scenario in operating_scenarios),
        policies=tuple(policies),
        operating_characteristics=tuple(operating),
        online_fdr=online,
        claim_boundary=(
            "Power and error rates are conditional on each declared model, "
            "process, setting, policy, and finite simulation design; benchmark "
            "and environment-shift results are stress evidence unless separately "
            "certified."
        ),
        paired_policy_comparisons=paired,
        power_cost_frontier=frontier,
        split_stream_confirmation=split_stream,
        scenario_contracts=tuple(
            scenario.to_mapping() for scenario in operating_scenarios
        ),
        split_confirmation_scenario=split_confirmation_scenario.to_mapping(),
    )


def reference_suite(
    *,
    reliability: float = 0.8,
    horizon: int = 12,
    alpha: float = 0.05,
    n_reps: int = 100,
    seed: int = 0,
    workers: int = 1,
) -> PowerSuiteResult:
    """Small offline suite used by examples and smoke validation."""
    scenario = make_reference_action_scenario(reliability=reliability, horizon=horizon, alpha=alpha)
    return run_power_suite(
        [scenario],
        ("fixed_horizon", "posterior_threshold", "information_gain", "cost_aware", "adaptive_target", "posterior_sampling", "e_process"),
        n_reps=n_reps,
        seed=seed,
        workers=workers,
    )


__all__ = [
    "PowerSuiteResult", "evaluate_adaptive_target_selection",
    "evaluate_conditional_selection_calibration",
    "evaluate_future_evidence_selection_failure", "evaluate_split_stream_confirmation",
    "run_power_suite", "reference_suite",
]
