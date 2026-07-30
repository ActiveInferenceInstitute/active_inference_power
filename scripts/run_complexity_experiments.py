#!/usr/bin/env python3
"""Write a local, non-promotional runtime diagnostic for declared kernels.

The publication bundle reports deterministic source-work accounting only.  This
separate command measures selected real code paths repeatedly on the invoking
machine so a developer can inspect practical scaling while preserving the
distinction between source work, simulated resource cost ``C``, e-process
evidence ``E_t``, and online alpha-accounting ``W^alpha``.  It never writes
under ``output/``, modifies a release artifact, hydrates a manuscript, or
creates a promotion decision.

The command has one public parameter: a previously absent ``--output-root``.
The timing plan is frozen in ``manuscript/config.yaml``.  Programmatic callers
may pass a compact :class:`ComputationalComplexityPlan` for a real, bounded
test; that escape hatch is intentionally unavailable from the CLI.
"""

from __future__ import annotations

# Imports below intentionally follow the script-local path bootstrap so direct
# invocation works from an arbitrary current working directory.
# ruff: noqa: E402

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "manuscript" / "config.yaml"
TIMING_PROFILE_FILENAME = "complexity_experiments.json"
TIMING_FIGURE_FILENAME = "complexity_experiments.png"
TIMING_INDEX_FILENAME = "index.html"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from active_inference_power.action_loop import ActionPolicy, run_action_loop
from active_inference_power.computational_complexity import (
    ComputationalComplexityPlan,
    TIMING_PROFILE_FROZEN_PLAN_SOURCE,
    TIMING_PROFILE_MEASUREMENT_PROTOCOL,
    TIMING_PROFILE_PROGRAMMATIC_PLAN_SOURCE,
    TIMING_PROFILE_STATUS,
    complexity_plan_from_mapping,
    summarize_timing_samples,
    timing_plan_digest,
    timing_profile_digest,
    validate_timing_profile,
)
from active_inference_power.corrections import benjamini_hochberg, bonferroni
from active_inference_power.online_fdr import e_lond, lord_plus_plus, saffron
from active_inference_power.scenario import (
    PowerScenario,
    make_reference_action_scenario,
)
from active_inference_power.sequential import (
    bernoulli_confidence_sequence,
    binary_e_process,
    fixed_horizon_binomial_pvalue,
)
from scripts.artifacts import (
    artifact_record,
    atomic_write_json,
    atomic_write_text,
    require_safe_output_root,
    sha256_file,
)


_SOURCE_PATHS = {
    "orchestrator": Path(__file__),
    "computational_complexity": ROOT
    / "src"
    / "active_inference_power"
    / "computational_complexity.py",
    "corrections": ROOT / "src" / "active_inference_power" / "corrections.py",
    "online_fdr": ROOT / "src" / "active_inference_power" / "online_fdr.py",
    "sequential": ROOT / "src" / "active_inference_power" / "sequential.py",
    "action_loop": ROOT / "src" / "active_inference_power" / "action_loop.py",
    "scenario": ROOT / "src" / "active_inference_power" / "scenario.py",
    "pymdp_adapter": ROOT / "src" / "active_inference_power" / "pymdp_adapter.py",
}
_THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_CORRECTIONS: dict[str, Callable[[np.ndarray], object]] = {
    "bonferroni": bonferroni,
    "benjamini_hochberg": benjamini_hochberg,
}
TIMING_STATUS = TIMING_PROFILE_STATUS


def _validate_output_root(output_root: Path) -> Path:
    """Require a new, concrete root outside the canonical release tree."""
    target = Path(output_root).expanduser()
    if not target.is_absolute():
        target = ROOT / target
    target = require_safe_output_root(target, project_root=ROOT)
    project = ROOT.resolve()
    canonical_output = (ROOT / "output").resolve()
    try:
        target.relative_to(canonical_output)
        is_within_canonical_output = True
    except ValueError:
        is_within_canonical_output = False
    if target == project or is_within_canonical_output:
        raise ValueError(
            "complexity experiments require a new isolated --output-root, not the project output root or a descendant"
        )
    if target == target.parent:
        raise ValueError(
            "complexity experiment output root must be a concrete directory"
        )
    if target.exists():
        raise FileExistsError(
            f"complexity experiment output root already exists: {target}; choose a new isolated directory"
        )
    return target


def _frozen_plan() -> tuple[ComputationalComplexityPlan, str]:
    """Read the one canonical timing plan without accepting CLI overrides."""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"frozen complexity plan is missing: {CONFIG_PATH}")
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("manuscript/config.yaml must contain a mapping")
    experiment = payload.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("manuscript/config.yaml must define experiment as a mapping")
    raw_plan = experiment.get("computational_complexity")
    if not isinstance(raw_plan, Mapping):
        raise ValueError(
            "manuscript/config.yaml must define experiment.computational_complexity as a mapping"
        )
    return complexity_plan_from_mapping(raw_plan), sha256_file(CONFIG_PATH)


def _local_module_path(candidate: Path) -> Path | None:
    """Resolve one local module stem without following external packages."""
    module_file = candidate.with_suffix(".py")
    if module_file.is_file():
        return module_file.resolve()
    package_file = candidate / "__init__.py"
    if package_file.is_file():
        return package_file.resolve()
    return None


def _local_imports(path: Path) -> set[Path]:
    """Resolve direct imports that live inside this package's source tree."""
    source_root = ROOT / "src"
    package_root = source_root / "active_inference_power"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[Path] = set()
    for node in ast.walk(tree):
        module_names: list[str] = []
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                if node.module:
                    candidate = _local_module_path(base / node.module.replace(".", "/"))
                    if candidate is not None:
                        imports.add(candidate)
                else:
                    for alias in node.names:
                        candidate = _local_module_path(
                            base / alias.name.replace(".", "/")
                        )
                        if candidate is not None:
                            imports.add(candidate)
            elif node.module and (
                node.module == "active_inference_power"
                or node.module.startswith("active_inference_power.")
            ):
                module_names.append(node.module)
        elif isinstance(node, ast.Import):
            module_names.extend(
                alias.name
                for alias in node.names
                if alias.name == "active_inference_power"
                or alias.name.startswith("active_inference_power.")
            )
        for module_name in module_names:
            candidate = _local_module_path(source_root / module_name.replace(".", "/"))
            if candidate is not None:
                imports.add(candidate)
    return {
        item for item in imports if item.is_file() and item.is_relative_to(package_root)
    }


def _timing_source_closure() -> tuple[Path, ...]:
    """Return the transitive local source closure for every timed kernel root."""
    pending = [
        path.resolve() for name, path in _SOURCE_PATHS.items() if name != "orchestrator"
    ]
    closure: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in closure:
            continue
        closure.add(path)
        pending.extend(item for item in _local_imports(path) if item not in closure)
    return tuple(sorted(closure))


def _source_hashes() -> dict[str, str]:
    """Return direct roots plus their transitive local source-closure witness."""
    missing = [name for name, path in _SOURCE_PATHS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"complexity timing sources are missing: {missing}")
    direct = {name: sha256_file(path) for name, path in _SOURCE_PATHS.items()}
    direct_paths = {path.resolve() for path in _SOURCE_PATHS.values()}
    dependencies = {
        f"dependency:{path.relative_to(ROOT).as_posix()}": sha256_file(path)
        for path in _timing_source_closure()
        if path not in direct_paths
    }
    return {**direct, **dependencies}


def _assert_stable_timing_witnesses(
    initial_source_hashes: Mapping[str, str],
    final_source_hashes: Mapping[str, str],
    *,
    initial_frozen: tuple[ComputationalComplexityPlan, str] | None,
    final_frozen: tuple[ComputationalComplexityPlan, str] | None,
) -> None:
    """Fail closed if a measured source or frozen plan changes mid-run."""
    if dict(initial_source_hashes) != dict(final_source_hashes):
        raise RuntimeError(
            "timing source witness changed during measurement; no diagnostic receipt was written"
        )
    if (initial_frozen is None) != (final_frozen is None):
        raise RuntimeError(
            "timing frozen-plan provenance changed during measurement; no diagnostic receipt was written"
        )
    if initial_frozen is not None and final_frozen is not None:
        if initial_frozen != final_frozen:
            raise RuntimeError(
                "timing frozen-plan provenance changed during measurement; no diagnostic receipt was written"
            )


def _validate_frozen_timing_profile(profile: Mapping[str, Any]) -> None:
    """Bind a frozen profile to the live plan, config, and exact source closure."""
    plan, config_digest = _frozen_plan()
    validate_timing_profile(
        profile,
        expected_frozen_plan=plan,
        expected_plan_source_sha256=config_digest,
        expected_source_hashes=_source_hashes(),
    )


def _package_version(distribution: str) -> str:
    """Return the installed version without recording a host-specific path."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _environment_receipt() -> dict[str, Any]:
    """Capture execution-relevant runtime facts while deliberately redacting identity."""
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "runtime": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "libraries": {
            "active_inference_power": _package_version("active_inference_power"),
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "matplotlib": _package_version("matplotlib"),
            "inferactively_pymdp": _package_version("inferactively-pymdp"),
        },
        "thread_environment": {
            key: os.environ.get(key, "unset") for key in _THREAD_ENVIRONMENT_KEYS
        },
        "redaction": (
            "This receipt intentionally omits hostname, username, home path, current working "
            "directory, process identifier, and the full environment."
        ),
    }


def _pvalue_vector(size: int, *, dense: bool = False) -> np.ndarray:
    """Construct deterministic inputs before timing a correction or online kernel."""
    if dense:
        return np.zeros(size, dtype=float)
    return np.linspace(0.001, 0.999, size, dtype=float)


def _online_pvalue_vector(size: int, *, dense: bool) -> np.ndarray:
    """Construct a genuine sparse/no-rejection or dense-rejection arrival history."""
    return np.zeros(size, dtype=float) if dense else np.full(size, 0.99, dtype=float)


def _observation_vector(size: int) -> np.ndarray:
    """Construct a deterministic binary stream before sequential timing begins."""
    return (np.arange(size, dtype=int) % 3 != 0).astype(int)


def _measure(
    callback: Callable[[], object], *, warmups: int, repetitions: int
) -> tuple[list[int], list[int]]:
    """Measure one real callable after excluded warmups using two monotonic clocks."""
    for _ in range(warmups):
        callback()
    wall_samples: list[int] = []
    process_samples: list[int] = []
    for _ in range(repetitions):
        process_started = time.process_time_ns()
        wall_started = time.perf_counter_ns()
        callback()
        wall_samples.append(time.perf_counter_ns() - wall_started)
        process_samples.append(time.process_time_ns() - process_started)
    return wall_samples, process_samples


def _timing_row(
    *,
    kernel: str,
    input_size: int,
    condition: str,
    callback: Callable[[], object],
    plan: ComputationalComplexityPlan,
) -> dict[str, Any]:
    """Return one raw-and-summary timing record with no benchmark threshold."""
    wall_samples, process_samples = _measure(
        callback,
        warmups=plan.timing_warmups,
        repetitions=plan.timing_repetitions,
    )
    return {
        "kernel": kernel,
        "input_size": input_size,
        "condition": condition,
        "warmups": plan.timing_warmups,
        "wall_time_ns_samples": wall_samples,
        "process_time_ns_samples": process_samples,
        "wall_time_ns_summary": summarize_timing_samples(wall_samples),
        "process_time_ns_summary": summarize_timing_samples(process_samples),
    }


def _correction_rows(plan: ComputationalComplexityPlan) -> list[dict[str, Any]]:
    """Measure real fixed-family correction paths over the declared m grid."""
    rows: list[dict[str, Any]] = []
    for method in plan.correction_methods:
        correction = _CORRECTIONS[method]
        for size in plan.hypothesis_grid:
            pvalues = _pvalue_vector(size)

            def correction_callback(
                correction: Callable[[np.ndarray], object] = correction,
                pvalues: np.ndarray = pvalues,
            ) -> object:
                return correction(pvalues)

            rows.append(
                _timing_row(
                    kernel=f"correction:{method}",
                    input_size=size,
                    condition="deterministic_pvalue_vector",
                    callback=correction_callback,
                    plan=plan,
                )
            )
    return rows


def _online_callback(method: str, size: int, dense: bool) -> Callable[[], object]:
    """Build one source-owned online-FDR call without timing input creation."""
    if method == "e_lond":
        evalues = np.full(size, 1.0e12 if dense else 0.0, dtype=float)

        def e_lond_callback() -> object:
            return e_lond(evalues)

        return e_lond_callback
    pvalues = _online_pvalue_vector(size, dense=dense)
    if method == "lord_plus_plus":

        def lord_callback() -> object:
            return lord_plus_plus(pvalues)

        return lord_callback
    if method == "saffron":

        def saffron_callback() -> object:
            return saffron(pvalues)

        return saffron_callback
    raise ValueError(f"unsupported declared online timing method: {method!r}")


def _online_rows(plan: ComputationalComplexityPlan) -> list[dict[str, Any]]:
    """Measure sparse and dense arrival histories for each declared online method."""
    rows: list[dict[str, Any]] = []
    for method in plan.online_methods:
        for dense, condition in (
            (False, "sparse_or_no_rejection_history"),
            (True, "dense_rejection_history"),
        ):
            for size in plan.stream_horizon_grid:
                rows.append(
                    _timing_row(
                        kernel=f"online:{method}",
                        input_size=size,
                        condition=condition,
                        callback=_online_callback(method, size, dense),
                        plan=plan,
                    )
                )
    return rows


def _sequential_rows(plan: ComputationalComplexityPlan) -> list[dict[str, Any]]:
    """Measure e-process, confidence-sequence, and explicit fixed-horizon paths."""
    rows: list[dict[str, Any]] = []
    for size in plan.stream_horizon_grid:
        observations = _observation_vector(size)

        def binary_e_callback(observations: np.ndarray = observations) -> object:
            return binary_e_process(observations, null_rate=0.5, alternative_rate=0.7)

        def confidence_sequence_callback(
            observations: np.ndarray = observations,
        ) -> object:
            return bernoulli_confidence_sequence(observations)

        def fixed_horizon_callback(observations: np.ndarray = observations) -> object:
            return fixed_horizon_binomial_pvalue(
                observations, null_rate=0.5, alternative_rate=0.7
            )

        entries: tuple[tuple[str, str, Callable[[], object]], ...] = (
            (
                "sequential:binary_e_process",
                "likelihood_ratio_evidence_path",
                binary_e_callback,
            ),
            (
                "sequential:bernoulli_confidence_sequence",
                "time_uniform_confidence_sequence_inversion",
                confidence_sequence_callback,
            ),
            (
                "sequential:fixed_horizon_binomial_pvalue",
                "fixed_horizon_only_invalid_for_optional_stopping",
                fixed_horizon_callback,
            ),
        )
        for kernel, condition, callback in entries:
            rows.append(
                _timing_row(
                    kernel=kernel,
                    input_size=size,
                    condition=condition,
                    callback=callback,
                    plan=plan,
                )
            )
    return rows


def _action_rows(plan: ComputationalComplexityPlan) -> list[dict[str, Any]]:
    """Measure real pymdp-backed action loops with construction outside each timer."""
    rows: list[dict[str, Any]] = []
    for horizon in plan.action_horizon_grid:
        scenario = make_reference_action_scenario(
            reliability=0.8,
            horizon=horizon,
            alpha=0.05,
            scenario_id=f"complexity_timing_reference_h{horizon}",
        )
        for policy in plan.action_policies:

            def action_callback(
                scenario: PowerScenario = scenario,
                policy: ActionPolicy = cast(ActionPolicy, policy),
            ) -> object:
                return run_action_loop(scenario, policy=policy, seed=29, use_pymdp=True)

            rows.append(
                _timing_row(
                    kernel=f"action_loop:{policy}",
                    input_size=horizon,
                    condition="real_pymdp_action_observation_belief_evidence_trace",
                    callback=action_callback,
                    plan=plan,
                )
            )
    return rows


def build_timing_profile(
    plan: ComputationalComplexityPlan,
    *,
    plan_source: str,
    plan_source_sha256: str | None,
    source_hashes: Mapping[str, str],
    expected_frozen_plan: ComputationalComplexityPlan | None = None,
    expected_source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute every declared real kernel and return its self-hashed profile."""
    if not isinstance(plan, ComputationalComplexityPlan):
        raise ValueError("plan must be a ComputationalComplexityPlan")
    rows = [
        *_correction_rows(plan),
        *_online_rows(plan),
        *_sequential_rows(plan),
        *_action_rows(plan),
    ]
    profile: dict[str, Any] = {
        "contract_version": "1.0",
        "kind": "research_only_local_runtime_diagnostic",
        "status": TIMING_STATUS,
        "non_claim": (
            "Repeated local timing is a machine-scoped engineering diagnostic, not a release "
            "result, performance threshold, cross-machine comparison, simulated resource cost C, "
            "e-process evidence E_t, online alpha-wealth W^alpha, or statistical-validity claim."
        ),
        "plan": plan.to_mapping(),
        "plan_sha256": timing_plan_digest(plan),
        "plan_source": plan_source,
        "plan_source_sha256": plan_source_sha256,
        "environment": _environment_receipt(),
        "source_hashes": dict(source_hashes),
        "measurement_protocol": dict(TIMING_PROFILE_MEASUREMENT_PROTOCOL),
        "rows": rows,
    }
    profile["result_sha256"] = timing_profile_digest(profile)
    validate_timing_profile(
        profile,
        expected_frozen_plan=expected_frozen_plan,
        expected_plan_source_sha256=plan_source_sha256
        if plan_source == TIMING_PROFILE_FROZEN_PLAN_SOURCE
        else None,
        expected_source_hashes=expected_source_hashes,
    )
    return profile


def _row_summary(row: Mapping[str, Any]) -> tuple[float, float, float]:
    """Return observed median and quartile endpoints in milliseconds."""
    summary = row.get("wall_time_ns_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("timing row is missing wall_time_ns_summary")
    try:
        q1 = float(summary["q1_ns"]) / 1_000_000.0
        median = float(summary["median_ns"]) / 1_000_000.0
        q3 = float(summary["q3_ns"]) / 1_000_000.0
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("timing row has an invalid wall-time summary") from exc
    if q1 < 0.0 or median < q1 or q3 < median:
        raise ValueError("timing row has unordered wall-time quartiles")
    return q1, median, q3


def _rows_for_prefix(
    profile: Mapping[str, Any], prefix: str
) -> list[Mapping[str, Any]]:
    """Select timing rows by one explicit source-kernel prefix."""
    rows = profile.get("rows")
    if not isinstance(rows, list):
        raise ValueError("timing profile rows must be a list")
    selected = [
        row
        for row in rows
        if isinstance(row, Mapping) and str(row.get("kernel", "")).startswith(prefix)
    ]
    if not selected:
        raise ValueError(f"timing profile has no rows for {prefix!r}")
    return selected


def _errorbar_series(
    axes: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    color: str,
    marker: str,
    linestyle: str,
) -> None:
    """Plot direct, marker-and-line encoded local quartile timing observations."""
    ordered = sorted(rows, key=lambda row: int(row["input_size"]))
    inputs = [int(row["input_size"]) for row in ordered]
    summaries = [_row_summary(row) for row in ordered]
    medians = [item[1] for item in summaries]
    lower = [item[1] - item[0] for item in summaries]
    upper = [item[2] - item[1] for item in summaries]
    axes.errorbar(
        inputs,
        medians,
        yerr=[lower, upper],
        marker=marker,
        linestyle=linestyle,
        linewidth=1.6,
        markersize=5.8,
        capsize=3,
        color=color,
        ecolor=color,
        label=label,
    )


def _configure_timing_axes(axes: plt.Axes, *, xlabel: str, title: str) -> None:
    """State local estimand, aggregation, scale, and uncertainty on every panel."""
    axes.set_title(title, fontsize=10.5, fontweight="bold")
    axes.set_xlabel(xlabel, fontsize=9)
    axes.set_ylabel(
        "Median local wall time $\\tau_{wall}$ (ms/call; IQR bars)", fontsize=8.6
    )
    axes.set_xscale("log", base=2)
    axes.grid(color="#cbd5e1", alpha=0.7, linewidth=0.7)
    axes.legend(
        fontsize=7.1, loc="best", title="Kernel / condition", title_fontsize=7.2
    )


def _friendly_kernel(kernel: str) -> str:
    """Make one source-kernel key readable without changing its provenance key."""
    labels = {
        "correction:bonferroni": "Bonferroni",
        "correction:benjamini_hochberg": "BH rank correction",
        "online:lord_plus_plus": "LORD++",
        "online:saffron": "SAFFRON",
        "online:e_lond": "e-LOND",
        "sequential:binary_e_process": "Likelihood-ratio e-process",
        "sequential:bernoulli_confidence_sequence": "Confidence-sequence inversion",
        "sequential:fixed_horizon_binomial_pvalue": "Fixed-horizon p-value (invalid under optional stopping)",
        "action_loop:fixed_horizon": "Fixed horizon",
        "action_loop:information_gain": "Information gain",
        "action_loop:cost_aware": "Cost aware",
        "action_loop:e_process": "e-process action policy",
    }
    return labels.get(kernel, kernel.replace("_", " "))


def _condition_label(condition: str) -> str:
    """Shrink serialized conditions only for figure legend readability."""
    labels = {
        "deterministic_pvalue_vector": "deterministic vector",
        "sparse_or_no_rejection_history": "sparse/no-rejection history",
        "dense_rejection_history": "dense-rejection history",
        "likelihood_ratio_evidence_path": "likelihood-ratio path",
        "time_uniform_confidence_sequence_inversion": "time-uniform inversion",
        "fixed_horizon_only_invalid_for_optional_stopping": "fixed horizon only",
        "real_pymdp_action_observation_belief_evidence_trace": "real pymdp trace",
    }
    return labels.get(condition, condition.replace("_", " "))


def _write_timing_figure(profile: Mapping[str, Any], path: Path) -> str:
    """Render a reader-facing local-time figure, explicitly outside release evidence."""
    figure, axes_grid = plt.subplots(2, 2, figsize=(16.4, 10.8))
    axes = axes_grid.ravel()
    styles = (
        ("#2166ac", "o", "-"),
        ("#b2182b", "s", "--"),
        ("#1b7837", "^", "-"),
        ("#762a83", "D", ":"),
        ("#d6604d", "P", "-."),
        ("#4a5568", "X", "-"),
        ("#8c510a", "v", "--"),
        ("#5ab4ac", "h", ":"),
    )

    correction_rows = _rows_for_prefix(profile, "correction:")
    for index, kernel in enumerate(
        sorted({str(row["kernel"]) for row in correction_rows})
    ):
        series = [row for row in correction_rows if row["kernel"] == kernel]
        color, marker, line = styles[index]
        _errorbar_series(
            axes[0],
            series,
            label=_friendly_kernel(kernel),
            color=color,
            marker=marker,
            linestyle=line,
        )
    _configure_timing_axes(
        axes[0],
        xlabel="Hypotheses $m$ (count; log2)",
        title="Fixed-family correction kernels",
    )

    online_rows = _rows_for_prefix(profile, "online:")
    online_groups = sorted(
        {(str(row["kernel"]), str(row["condition"])) for row in online_rows}
    )
    for index, (kernel, condition) in enumerate(online_groups):
        series = [
            row
            for row in online_rows
            if row["kernel"] == kernel and row["condition"] == condition
        ]
        color, marker, line = styles[index % len(styles)]
        _errorbar_series(
            axes[1],
            series,
            label=f"{_friendly_kernel(kernel)} — {_condition_label(condition)}",
            color=color,
            marker=marker,
            linestyle=line,
        )
    _configure_timing_axes(
        axes[1],
        xlabel="Arrival horizon $T$ (count; log2)",
        title="Online FDR/e-value accounting kernels",
    )

    sequential_rows = _rows_for_prefix(profile, "sequential:")
    for index, kernel in enumerate(
        sorted({str(row["kernel"]) for row in sequential_rows})
    ):
        series = [row for row in sequential_rows if row["kernel"] == kernel]
        color, marker, line = styles[index]
        _errorbar_series(
            axes[2],
            series,
            label=_friendly_kernel(kernel),
            color=color,
            marker=marker,
            linestyle=line,
        )
    _configure_timing_axes(
        axes[2],
        xlabel="Observation horizon $T$ (count; log2)",
        title="Sequential evidence and fixed-horizon contrast",
    )

    action_rows = _rows_for_prefix(profile, "action_loop:")
    for index, kernel in enumerate(sorted({str(row["kernel"]) for row in action_rows})):
        series = [row for row in action_rows if row["kernel"] == kernel]
        color, marker, line = styles[index]
        _errorbar_series(
            axes[3],
            series,
            label=_friendly_kernel(kernel),
            color=color,
            marker=marker,
            linestyle=line,
        )
    _configure_timing_axes(
        axes[3],
        xlabel="Action horizon $H$ (steps; log2)",
        title="Real pymdp action/observation/belief/evidence traces",
    )

    figure.suptitle(
        "Machine-scoped runtime diagnostic for declared computational kernels",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.078,
        "Points are medians of repeated local calls; bars span the local interquartile range, "
        "not Monte Carlo standard errors (MCSEs).",
        ha="center",
        va="bottom",
        fontsize=8.25,
        color="#334155",
    )
    figure.text(
        0.5,
        0.057,
        "This local-machine diagnostic is neither a cross-machine benchmark nor release evidence.",
        ha="center",
        va="bottom",
        fontsize=8.25,
        color="#334155",
    )
    figure.text(
        0.5,
        0.036,
        r"Separate quantities: simulated resource cost $C$; likelihood-ratio e-process evidence $E_t$; "
        r"online alpha-wealth accounting $W_t^{\alpha}$.",
        ha="center",
        va="bottom",
        fontsize=8.25,
        color="#334155",
    )
    figure.text(
        0.5,
        0.015,
        "The fixed-horizon p-value path is displayed only as an invalid optional-stopping contrast.",
        ha="center",
        va="bottom",
        fontsize=8.25,
        color="#334155",
    )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.94), h_pad=2.0, w_pad=2.0)
    description = (
        "Four-panel local runtime diagnostic. Each panel plots median wall-clock milliseconds "
        "per call against a logarithmic input count, with interquartile-range error bars. The "
        "panels show two fixed-family corrections, online FDR and e-value accounting under sparse "
        "and dense histories, sequential e-process and confidence-sequence paths plus a fixed-horizon "
        "p-value explicitly marked invalid under optional stopping, and four real pymdp action-loop "
        "policies. Colors, markers, line styles, titles, and legends redundantly identify series. "
        "The footer identifies these as local timing summaries rather than MCSEs, cross-machine "
        "benchmarks, or release evidence, and distinguishes them from simulated resource cost, "
        "likelihood-ratio e-process evidence, and online alpha-wealth accounting."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=180,
            bbox_inches="tight",
            metadata={
                "Title": "Machine-scoped runtime diagnostic for declared computational kernels",
                "Description": description,
            },
        )
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()
    return description


def _index_html(profile: Mapping[str, Any], figure_alt: str) -> str:
    """Create a minimal, visibly non-promotional landing page for the isolated bundle."""
    non_claim = html.escape(str(profile["non_claim"]))
    digest = html.escape(str(profile["result_sha256"]))
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="en">',
            '<head><meta charset="utf-8"><title>Local complexity timing diagnostic</title></head>',
            "<body>",
            "<main>",
            "<h1>Local computational-complexity timing diagnostic</h1>",
            f"<p><strong>Status:</strong> {html.escape(TIMING_STATUS)}</p>",
            f"<p>{non_claim}</p>",
            "<p>The raw repeated samples, summaries, environment receipt, source witness, and self-hash are in ",
            f'<a href="{TIMING_PROFILE_FILENAME}">{TIMING_PROFILE_FILENAME}</a>.</p>',
            "<figure>",
            f'<img src="{TIMING_FIGURE_FILENAME}" alt="{html.escape(figure_alt, quote=True)}">',
            "<figcaption>Local timing medians and interquartile ranges for declared real code paths; not a release result.</figcaption>",
            "</figure>",
            f"<p>Profile SHA-256: <code>{digest}</code></p>",
            "</main>",
            "</body>",
            "</html>",
            "",
        )
    )


def _manifest(
    staging_root: Path,
    profile: Mapping[str, Any],
    *,
    figure_alt: str,
) -> dict[str, Any]:
    """Bind every direct-root artifact without creating a release manifest."""
    profile_path = staging_root / TIMING_PROFILE_FILENAME
    figure_path = staging_root / TIMING_FIGURE_FILENAME
    index_path = staging_root / TIMING_INDEX_FILENAME
    return {
        "schema_version": "1.0",
        "kind": "research_only_local_runtime_diagnostic_manifest",
        "status": TIMING_STATUS,
        "profile_result_sha256": profile["result_sha256"],
        "profile_file_sha256": sha256_file(profile_path),
        "plan": profile["plan"],
        "plan_sha256": profile["plan_sha256"],
        "plan_source": profile["plan_source"],
        "plan_source_sha256": profile["plan_source_sha256"],
        "source_hashes": profile["source_hashes"],
        "environment_redaction": profile["environment"]["redaction"],
        "claim_boundary": profile["non_claim"],
        "figure_alt": figure_alt,
        "artifacts": {
            "timing_profile": artifact_record(
                profile_path,
                staging_root,
                "local_runtime_timing_profile",
                root_label="output",
            ),
            "timing_figure": artifact_record(
                figure_path,
                staging_root,
                "local_runtime_timing_figure",
                root_label="output",
            ),
            "index_html": artifact_record(
                index_path,
                staging_root,
                "local_runtime_diagnostic_index",
                root_label="output",
            ),
        },
    }


def run_complexity_experiments(
    output_root: Path,
    plan: ComputationalComplexityPlan | None = None,
) -> dict[str, Any]:
    """Atomically write one isolated timing bundle at a previously absent root.

    With no explicit ``plan``, this reads exactly the canonical plan from the
    manuscript configuration.  Passing a plan is reserved for real compact
    tests and library-level diagnostic experiments; the public CLI exposes no
    corresponding tuning parameters.
    """
    target = _validate_output_root(Path(output_root))
    if plan is None:
        active_plan, plan_digest = _frozen_plan()
        plan_source = TIMING_PROFILE_FROZEN_PLAN_SOURCE
        expected_frozen_plan: ComputationalComplexityPlan | None = active_plan
        initial_frozen: tuple[ComputationalComplexityPlan, str] | None = (
            active_plan,
            plan_digest,
        )
    else:
        if not isinstance(plan, ComputationalComplexityPlan):
            raise ValueError("plan must be a ComputationalComplexityPlan")
        active_plan = plan
        plan_digest = None
        plan_source = TIMING_PROFILE_PROGRAMMATIC_PLAN_SOURCE
        expected_frozen_plan = None
        initial_frozen = None
    initial_source_hashes = _source_hashes()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.complexity-experiments-", dir=target.parent
        )
    )
    try:
        profile = build_timing_profile(
            active_plan,
            plan_source=plan_source,
            plan_source_sha256=plan_digest,
            source_hashes=initial_source_hashes,
            expected_frozen_plan=expected_frozen_plan,
            expected_source_hashes=initial_source_hashes,
        )
        final_frozen = _frozen_plan() if initial_frozen is not None else None
        _assert_stable_timing_witnesses(
            initial_source_hashes,
            _source_hashes(),
            initial_frozen=initial_frozen,
            final_frozen=final_frozen,
        )
        if initial_frozen is not None:
            _validate_frozen_timing_profile(profile)
        profile_path = staging_root / TIMING_PROFILE_FILENAME
        atomic_write_json(profile_path, profile)
        figure_path = staging_root / TIMING_FIGURE_FILENAME
        figure_alt = _write_timing_figure(profile, figure_path)
        index_path = staging_root / TIMING_INDEX_FILENAME
        atomic_write_text(index_path, _index_html(profile, figure_alt))
        manifest_path = staging_root / "manifest.json"
        atomic_write_json(
            manifest_path,
            _manifest(staging_root, profile, figure_alt=figure_alt),
        )
        if target.exists():
            raise FileExistsError(
                f"complexity experiment output root was created concurrently: {target}"
            )
        os.replace(staging_root, target)
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    return {
        "output_root": str(target),
        "profile_path": str(target / TIMING_PROFILE_FILENAME),
        "figure_path": str(target / TIMING_FIGURE_FILENAME),
        "index_path": str(target / TIMING_INDEX_FILENAME),
        "manifest_path": str(target / "manifest.json"),
        "profile_sha256": sha256_file(target / TIMING_PROFILE_FILENAME),
        "figure_sha256": sha256_file(target / TIMING_FIGURE_FILENAME),
        "manifest_sha256": sha256_file(target / "manifest.json"),
        "result_sha256": profile["result_sha256"],
        "status": TIMING_STATUS,
    }


def _parser() -> argparse.ArgumentParser:
    """Create the deliberately non-tunable command-line boundary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="New isolated directory for the frozen local timing diagnostic; never defaults to output/.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write a compact, machine-readable receipt for the isolated bundle."""
    args = _parser().parse_args(argv)
    receipt = run_complexity_experiments(args.output_root)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main.
    raise SystemExit(main())
