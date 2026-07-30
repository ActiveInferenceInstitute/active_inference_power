"""Optional interactive demo helpers for the action-in-the-loop suite."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, cast

from .action_loop import ActionPolicy, run_action_loop
from .scenario import make_reference_action_scenario


_DEMO_SCHEMA_VERSION = "1.0"
_SHA256_HEX_LENGTH = 64


def canonical_demo_json(payload: Mapping[str, Any]) -> str:
    """Return the unique JSON representation used by demo download bundles.

    The optional browser UI must not be the sole producer or validator of a
    trace.  Keeping this serializer in the deterministic domain package lets
    the headless smoke lane validate precisely the bytes offered for download.
    """
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("demo payload must be finite JSON-serializable data") from error


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_demo_json(payload).encode("utf-8")).hexdigest()


def _require_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_finite_number(value: Any, name: str) -> float:
    """Return one finite JSON number, excluding Boolean values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite numeric")
    return number


def _require_e_process_values(value: Any, name: str) -> list[float]:
    """Validate one explicit, non-empty e-process evidence path."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty JSON array")
    values = [_require_finite_number(item, f"{name}[{index}]") for index, item in enumerate(value)]
    if any(item < 0.0 for item in values):
        raise ValueError(f"{name} must be non-negative")
    return values


def _validate_e_process_trace_fields(
    trace: Mapping[str, Any], visible_trace: list[Any]
) -> None:
    """Require the canonical e-process fields used by every demo reader.

    The UI consumes only canonical evidence names.  Validating them before a
    payload is accepted makes an outdated or incomplete trace fail closed
    instead of leaving a delayed KeyError in an interactive chart.
    """
    if trace.get("evidence_kind") != "likelihood_ratio_e_process":
        raise ValueError("trace.evidence_kind must identify likelihood-ratio e-process evidence")
    _require_finite_number(trace.get("terminal_e_process_value"), "trace.terminal_e_process_value")
    _require_e_process_values(
        trace.get("target_e_process_values"), "trace.target_e_process_values"
    )
    steps = trace.get("steps")
    if not isinstance(steps, list) or len(steps) != len(visible_trace):
        raise ValueError("trace.steps must align with visible_trace")
    for index, (raw_step, raw_visible_step) in enumerate(zip(steps, visible_trace, strict=True)):
        step = _require_mapping(raw_step, f"trace.steps[{index}]")
        visible_step = _require_mapping(raw_visible_step, f"visible_trace[{index}]")
        for name, mapping in (("trace.steps", step), ("visible_trace", visible_step)):
            if mapping.get("evidence_kind") != "likelihood_ratio_e_process":
                raise ValueError(f"{name}[{index}].evidence_kind must identify likelihood-ratio e-process evidence")
            _require_finite_number(
                mapping.get("e_process_value"), f"{name}[{index}].e_process_value"
            )
            _require_e_process_values(
                mapping.get("target_e_process_values"),
                f"{name}[{index}].target_e_process_values",
            )


def validate_demo_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless a downloaded demo trace has intact lineage.

    The configuration digest is recomputed from the self-contained scenario
    contract, while the trace and visible projection are each independently
    bound.  This makes copied or edited JSON visibly untrustworthy rather than
    silently rendering it as a provenance-bearing trace.
    """
    bundle = dict(_require_mapping(payload, "demo payload"))
    if bundle.get("schema_version") != _DEMO_SCHEMA_VERSION:
        raise ValueError("unsupported demo payload schema_version")
    scenario = dict(_require_mapping(bundle.get("scenario"), "scenario"))
    scenario_hash = _require_digest(bundle.get("scenario_hash"), "scenario_hash")
    configuration_hash = _require_digest(
        bundle.get("configuration_hash"), "configuration_hash"
    )
    embedded_hash = _require_digest(
        scenario.pop("scenario_hash", None), "scenario.scenario_hash"
    )
    if scenario_hash != configuration_hash or scenario_hash != embedded_hash:
        raise ValueError("demo configuration hashes disagree")
    if _sha256_payload(scenario) != scenario_hash:
        raise ValueError("demo scenario does not match its configuration hash")

    trace = dict(_require_mapping(bundle.get("trace"), "trace"))
    trace_hash = _require_digest(bundle.get("trace_sha256"), "trace_sha256")
    if _sha256_payload(trace) != trace_hash:
        raise ValueError("demo trace does not match trace_sha256")
    visible_trace = bundle.get("visible_trace")
    if not isinstance(visible_trace, list):
        raise ValueError("visible_trace must be a JSON array")
    visible_hash = _require_digest(
        bundle.get("visible_trace_sha256"), "visible_trace_sha256"
    )
    if _sha256_payload({"visible_trace": visible_trace}) != visible_hash:
        raise ValueError("visible trace does not match visible_trace_sha256")
    if trace.get("visible_steps") != visible_trace:
        raise ValueError("visible_trace must exactly project trace.visible_steps")
    _validate_e_process_trace_fields(trace, visible_trace)
    if trace.get("scenario_id") != scenario.get("scenario_id"):
        raise ValueError("trace scenario_id does not match scenario")
    if (
        trace.get("policy") != bundle.get("policy")
        or trace.get("seed") != bundle.get("seed")
    ):
        raise ValueError("trace policy or seed does not match bundle metadata")

    payload_hash = _require_digest(bundle.get("payload_sha256"), "payload_sha256")
    unsigned_bundle = dict(bundle)
    unsigned_bundle.pop("payload_sha256")
    if _sha256_payload(unsigned_bundle) != payload_hash:
        raise ValueError("demo payload does not match payload_sha256")
    return bundle


def load_demo_payload(path: Path) -> dict[str, Any]:
    """Load and validate a downloaded/demo-fixture JSON payload."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load demo payload: {path}") from error
    return validate_demo_payload(_require_mapping(raw, "demo payload"))


def build_demo_payload(
    *,
    reliability: float = 0.8,
    horizon: int = 12,
    policy: ActionPolicy = "cost_aware",
    seed: int = 0,
    cost_budget: float | None = None,
    posterior_threshold: float | None = None,
) -> dict[str, Any]:
    """Build a deterministic JSON-shaped payload for the interactive UI."""
    scenario = make_reference_action_scenario(reliability=reliability, horizon=horizon)
    setting = scenario.setting
    if cost_budget is not None:
        setting = replace(setting, cost_budget=float(cost_budget))
    if posterior_threshold is not None:
        setting = replace(setting, posterior_threshold=float(posterior_threshold))
    if setting is not scenario.setting:
        scenario = replace(scenario, setting=setting)
    trace = run_action_loop(scenario, policy, seed=seed)
    scenario_mapping = scenario.to_mapping()
    trace_mapping = trace.to_mapping()
    visible_trace = [step.visible_mapping() for step in trace.steps]
    payload: dict[str, Any] = {
        "schema_version": _DEMO_SCHEMA_VERSION,
        "scenario": scenario_mapping,
        "scenario_hash": scenario.configuration_hash,
        "configuration_hash": scenario.configuration_hash,
        "policy": policy,
        "seed": int(seed),
        "trace": trace_mapping,
        "visible_trace": visible_trace,
        "diagnostic_evaluation": {
            "hidden_states": [
                {"time": step.time, "before": step.evaluation_state_before, "after": step.evaluation_state_after}
                for step in trace.steps
            ],
            "claim_boundary": "Hidden states are evaluation-only and are never policy inputs.",
        },
        "evidence_status": trace.validity_boundary,
        "evidence_class": "conditional_operating_characteristic" if trace.validity_boundary.startswith("conditional") else "diagnostic",
        "claim_boundary": "An investigator estimates power conditionally on the declared model, process, setting, policy, and finite simulation design; it is not an intrinsic agent score.",
    }
    # Retain hashes over the exact JSON objects that a headless consumer sees.
    # The scenario mapping carries its own configuration hash, so remove that
    # self-reference before reproducing PowerScenario.configuration_hash.
    scenario_contract = dict(scenario_mapping)
    scenario_contract.pop("scenario_hash", None)
    payload["trace_sha256"] = _sha256_payload(trace_mapping)
    payload["visible_trace_sha256"] = _sha256_payload({"visible_trace": visible_trace})
    # This is intentionally checked before the payload-wide digest: it binds
    # the UI's scenario display to the same source-derived contract.
    if _sha256_payload(scenario_contract) != scenario.configuration_hash:
        raise RuntimeError("demo scenario configuration hash is inconsistent")
    payload["payload_sha256"] = _sha256_payload(payload)
    return validate_demo_payload(payload)


def run_streamlit_app() -> None:  # pragma: no cover - exercised by the headless app smoke lane
    """Render the optional Streamlit application."""
    import streamlit as st  # type: ignore[import-not-found]  # optional demo dependency
    import matplotlib.pyplot as plt
    import numpy as np

    st.set_page_config(page_title="Active Inference Power Suite", layout="wide")
    st.title("Active Inference Power Suite")
    st.caption("Conditional power is a functional of the generative model, generative process, testing setting, and agent policy.")
    with st.sidebar:
        st.header("Scenario controls")
        reliability = st.slider("Observation reliability", 0.55, 0.95, 0.8, 0.01)
        horizon = st.slider("Maximum steps", 2, 30, 12)
        policy_options: tuple[ActionPolicy, ...] = (
            "fixed_horizon",
            "posterior_threshold",
            "cost_aware",
            "adaptive_target",
            "e_process",
        )
        policy = cast(ActionPolicy, st.selectbox("Action policy", policy_options))
        posterior_threshold = st.slider("Posterior decision threshold", 0.5, 0.999, 0.95, 0.005)
        cost_budget = st.slider(
            "Simulated cost budget C (scenario-local units)",
            0.25,
            30.0,
            12.0,
            0.25,
        )
        seed = st.number_input("Seed", min_value=0, value=0, step=1)
    payload = build_demo_payload(
        reliability=reliability, horizon=horizon, policy=policy, seed=int(seed),
        cost_budget=cost_budget, posterior_threshold=posterior_threshold,
    )
    trace = payload["trace"]
    process = cast(Mapping[str, Any], payload["scenario"]["process"])
    cost_unit = str(process.get("cost_unit", "scenario-defined simulation-cost unit"))
    cost_scale_id = str(process.get("cost_scale_id", "unrecorded cost scale"))
    badge = "FORMAL / CONDITIONAL" if payload["evidence_class"] == "conditional_operating_characteristic" else "DIAGNOSTIC"
    st.info(f"Evidence badge: {badge} · {payload['evidence_status']} · scenario hash `{payload['scenario_hash']}`")
    outcome_cols = st.columns(2)
    outcome_cols[0].metric("Decision", "reject" if trace["decision"] else "continue/accept")
    outcome_cols[1].metric("Stopping time (steps)", trace["stopping_time"])
    accounting_cols = st.columns(2)
    accounting_cols[0].metric(
        f"Simulated cost C ({cost_unit}; {cost_scale_id})",
        f"{trace['total_cost']:.2f}",
    )
    accounting_cols[1].metric(
        "Terminal likelihood-ratio e-process E_T",
        f"{float(trace['terminal_e_process_value']):.3g}",
    )
    if trace.get("family_scope") == "per_target_no_family_error_claim":
        st.caption(
            "Target-specific e-process decisions are per-target optional-stopping "
            "objects only; this trace makes no FWER/FDR claim, and its all-action "
            "likelihood-ratio product is diagnostic."
        )
    steps = trace["steps"]
    if steps:
        times = np.asarray([step["time"] for step in steps], dtype=float)
        frame = {
            "posterior signal": np.asarray([step["posterior_signal"] for step in steps], dtype=float),
            "p-value": np.asarray([step["p_value"] for step in steps], dtype=float),
            "e-process evidence E_t": np.asarray(
                [
                    step["e_process_value"]
                    for step in steps
                ],
                dtype=float,
            ),
            "cumulative cost": np.cumsum(np.asarray([step["cost"] for step in steps], dtype=float)),
        }
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(times, frame["posterior signal"], marker="o", label="posterior signal")
            ax.plot(times, frame["p-value"], marker="o", label="single-step p-value")
            ax.set(
                xlabel="Action-loop time t (steps)",
                ylabel="Posterior probability / p-value (unit scale)",
                ylim=(0, 1.05),
            )
            ax.legend(loc="best", fontsize=8)
            ax.grid(alpha=0.2)
            st.pyplot(fig, clear_figure=True)
        with right:
            fig, (evidence_ax, cost_ax) = plt.subplots(
                2, 1, figsize=(7, 4.6), sharex=True, height_ratios=(1.2, 1.0)
            )
            evidence_ax.plot(
                times,
                frame["e-process evidence E_t"],
                marker="o",
                label="Likelihood-ratio e-process evidence E_t",
            )
            evidence_ax.set(
                ylabel="E-process value E_t\n(dimensionless; linear scale)",
                title="Evidence and cost use separate, noncommensurate scales",
            )
            evidence_ax.legend(fontsize=8, loc="best")
            evidence_ax.grid(alpha=0.2)
            cost_ax.plot(
                times,
                frame["cumulative cost"],
                color="#c2410c",
                marker="s",
                label="Cumulative simulated cost C",
            )
            cost_ax.set(
                xlabel="Action-loop time t (steps)",
                ylabel=f"Cumulative cost C\n({cost_unit}; {cost_scale_id})",
            )
            cost_ax.legend(fontsize=8, loc="best")
            cost_ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        st.subheader("Visible action/observation history")
        st.dataframe(payload["visible_trace"], width="stretch", hide_index=True)
    st.caption(payload["claim_boundary"])
    st.download_button(
        "Download JSON trace bundle",
        canonical_demo_json(payload),
        file_name="action_loop_trace.json",
        mime="application/json",
    )


__all__ = [
    "build_demo_payload",
    "canonical_demo_json",
    "load_demo_payload",
    "run_streamlit_app",
    "validate_demo_payload",
]
