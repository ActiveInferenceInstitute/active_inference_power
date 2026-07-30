"""Deterministic publication figures for the active-inference study.

The functions in this module consume the JSON-shaped result dictionary emitted
by ``scripts/generate_results.py``. They do not rerun experiments or embed
scientific constants beyond plotting defaults; every plotted value therefore
has a traceable source in the result artifact.
"""

from __future__ import annotations

import html
import hashlib
import json
import os
import re
import tempfile
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyBboxPatch, Patch

__all__ = [
    "FIGURE_NAMES",
    "FIGURE_METADATA",
    "FIGURE_READER_CONTEXT",
    "figure_metadata",
    "result_lineage_hash",
    "dashboard_result_lineage_hash",
    "FIGURE_SPECS",
    "FigureSpec",
    "figure_alts",
    "figure_captions",
    "figure_visible_comparisons",
    "generate_figures",
    "write_figure_registry",
    "write_dashboard",
]

FIGURE_NAMES = (
    "method_comparison.png",
    "power_curve.png",
    "dependence_sweep.png",
    "active_inference_policy.png",
    "certificate_summary.png",
    "pvalue_calibration.png",
    "sample_size_precision.png",
    "paired_method_contrasts.png",
    "sequential_evidence.png",
    "online_fdr_trace.png",
    "conditional_selection_calibration.png",
    "benchmark_stress.png",
    "scholarship_boundary.png",
    "action_loop_trace.png",
    "conditional_power_surface.png",
    "action_policy_frontier.png",
    "scenario_decomposition.png",
    "testing_setting_contract.png",
    "formalism_code_map.png",
    "formalism_code_map_continued.png",
    "formalism_code_map_final.png",
    "adaptive_fdr_audit.png",
    "computational_complexity.png",
)

# Reader-facing dashboard headings are source-owned display labels, not a
# mechanical title-casing of artifact names. Statistical terms such as p-value
# and FDR therefore retain their conventional typography.
FIGURE_DASHBOARD_TITLES = {
    "method_comparison": "Method comparison",
    "power_curve": "Power curve",
    "dependence_sweep": "Dependence sweep",
    "active_inference_policy": "Active-inference policy comparison",
    "certificate_summary": "Certificate summary",
    "pvalue_calibration": "P-value calibration",
    "sample_size_precision": "Sample-size precision",
    "paired_method_contrasts": "Paired method contrasts",
    "sequential_evidence": "Sequential evidence",
    "online_fdr_trace": "Online FDR trace",
    "conditional_selection_calibration": "Conditional-selection calibration",
    "benchmark_stress": "Benchmark stress",
    "scholarship_boundary": "Scholarship boundary",
    "action_loop_trace": "Action-loop trace",
    "conditional_power_surface": "Conditional-power surface",
    "action_policy_frontier": "Action-policy frontier",
    "scenario_decomposition": "Scenario decomposition",
    "testing_setting_contract": "Testing-setting contract",
    "formalism_code_map": "Formalism-to-code map",
    "formalism_code_map_continued": "Formalism-to-code map (continued)",
    "formalism_code_map_final": "Formalism-to-code map (final panel)",
    "adaptive_fdr_audit": "Adaptive FDR audit",
    "computational_complexity": "Computational-work accounting",
}

# The formalism registry is deliberately rendered as multiple full-page panels
# rather than compressed into a visually dense table.  Nine rows preserve
# readable source/test and boundary text at the manuscript's print size.  If
# the registry grows beyond this fixed publication surface, the generator
# fails closed instead of silently clipping a new entry.
_FORMALISM_ROWS_PER_PANEL = 9
_FORMALISM_PANEL_COUNT = 3

# A restrained, color-vision-friendly palette is shared across every figure.
# Shape, line style, and text labels carry the semantic distinction as well, so
# the figures remain interpretable in grayscale and on low-quality printouts.
COLORS = {
    "blue": "#2f6f9f",
    "teal": "#2f855a",
    "orange": "#c05621",
    "purple": "#805ad5",
    "magenta": "#b83280",
    "slate": "#4a5568",
    "grid": "#cbd5e1",
    "pass": "#2f855a",
    "fail": "#c05621",
}

_POLICY_COLORS = (
    COLORS["blue"], COLORS["teal"], COLORS["orange"],
    COLORS["purple"], COLORS["magenta"], COLORS["slate"],
)
_POLICY_MARKERS = ("o", "s", "^", "D", "P", "X", "v")

_POLICY_LABELS = {
    "fixed_horizon": "Fixed horizon",
    "e_process": "E-process",
    "posterior_threshold": "Posterior threshold",
    "posterior_sampling": "Posterior sampling",
    "information_gain": "Information gain",
    "cost_aware": "Cost aware",
}

# Source-owned mark contracts make grayscale/low-vision distinctions auditable
# in tests, rather than relying on a visual review to notice color-only drift.
_DEPENDENCE_SERIES_STYLES = {
    "fdr": (
        ("bh_fdr", "BH FDR", COLORS["blue"], "o-", -0.075),
        ("by_fdr", "BY FDR", COLORS["purple"], "s--", 0.075),
    ),
    "power": (
        ("bh_power", "BH power", COLORS["teal"], "o-", -0.075),
        ("by_power", "BY power", COLORS["orange"], "s--", 0.075),
    ),
}
_ONLINE_FDR_EVENT_ROWS = (
    # Keep the order readers encounter in the legends top-to-bottom in the
    # event raster.  Tick labels below are derived from this same contract, so
    # a future row reordering cannot detach a procedure's marks from its name.
    ("lord_plus_plus", "LORD++", COLORS["blue"], 2),
    ("saffron", "SAFFRON", COLORS["purple"], 1),
    ("e_lond", "e-LOND", COLORS["orange"], 0),
)
_ONLINE_FDR_TRACE_STYLES = {
    "lord_plus_plus": ("o", "-"),
    "saffron": ("s", "--"),
    "e_lond": ("D", ":"),
}
_TARGET_TRACE_STYLES = (
    ("o", "-"),
    ("s", "--"),
    ("D", ":"),
    ("^", "-."),
)


def _policy_label(value: object) -> str:
    """Return a publication-readable policy label without changing result keys."""
    text = str(value)
    return _POLICY_LABELS.get(text, text.replace("_", " ").capitalize())


def _scenario_cell_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    """Return the immutable result identity used to keep figure cells distinct."""
    scenario_id = str(record.get("scenario_id", "")).strip()
    scenario_hash = str(record.get("scenario_hash", "")).strip()
    if not scenario_id or not scenario_hash:
        raise ValueError("action-loop figure rows require scenario_id and scenario_hash")
    return scenario_id, scenario_hash


def _scenario_cell_label(
    scenario_id: str, scenario_hash: str, metadata: Mapping[str, Any]
) -> str:
    """Return a compact, collision-resistant label for one declared scenario cell."""
    compact_id = scenario_id
    if len(compact_id) > 24:
        compact_id = f"{compact_id[:12]}…{compact_id[-10:]}"
    raw_reliability = metadata.get("reliability")
    if raw_reliability is None:
        reliability = "r=not recorded"
    else:
        try:
            reliability = f"r={float(raw_reliability):g}"
        except (TypeError, ValueError):
            reliability = "r=not recorded"
    if scenario_id.startswith("reference_action_loop__"):
        return f"{reliability}\n{scenario_hash[:8]}"
    return f"{compact_id}\n{scenario_hash[:8]} · {reliability}"


def _declared_cost_contracts(
    suite: Mapping[str, Any],
) -> dict[tuple[str, str], tuple[str, str]]:
    """Return each scenario's hashed simulated-cost unit and scale identifier.

    Cost is deliberately scenario-owned.  A shared label makes a cross-cell
    visual comparison interpretable; different labels instead trigger a
    reader-facing warning rather than silently treating arbitrary simulation
    costs as a common currency.
    """
    contracts: dict[tuple[str, str], tuple[str, str]] = {}
    raw_contracts = suite.get("scenario_contracts", [])
    if not isinstance(raw_contracts, list):
        return contracts
    for raw_contract in raw_contracts:
        if not isinstance(raw_contract, Mapping):
            continue
        scenario_id = raw_contract.get("scenario_id")
        scenario_hash = raw_contract.get("scenario_hash")
        process = raw_contract.get("process")
        if (
            not isinstance(scenario_id, str)
            or not scenario_id
            or not isinstance(scenario_hash, str)
            or not scenario_hash
            or not isinstance(process, Mapping)
        ):
            continue
        unit = process.get("cost_unit")
        scale = process.get("cost_scale_id")
        resolved_unit = (
            unit.strip()
            if isinstance(unit, str) and unit.strip()
            else "scenario-defined simulation-cost unit"
        )
        resolved_scale = (
            scale.strip()
            if isinstance(scale, str) and scale.strip()
            else "unrecorded cost scale"
        )
        contracts[(scenario_id, scenario_hash)] = (resolved_unit, resolved_scale)
    return contracts


def _cost_display_contract(
    suite: Mapping[str, Any], cells: Sequence[tuple[str, str]]
) -> tuple[str, str, bool]:
    """Summarize cost semantics for the exact scenario cells in one graphic."""
    contracts = _declared_cost_contracts(suite)
    declared = [contracts[cell] for cell in cells if cell in contracts]
    if declared and len(declared) == len(cells) and len(set(declared)) == 1:
        unit, scale = declared[0]
        return unit, scale, True
    return "scenario-specific simulation-cost units", "multiple or unrecorded scales", False


@dataclass(frozen=True)
class FigureSpec:
    """Source-owned metadata required to register a publication figure."""

    name: str
    label: str
    caption: str
    alt: str
    generated_by: str = "active_inference_power.figures.generate_figures"
    # These additive fields preserve the public constructor shape for existing
    # callers while giving the registry and dashboard a compact reader-first
    # account of what a figure is for.  The release registry rejects empty
    # values for its source-owned specifications.
    reader_question: str = ""
    takeaway: str = ""
    visual_encoding: str = ""
    visible_comparison: str = ""
    evaluation_perspective: str = ""


FIGURE_SPECS: tuple[FigureSpec, ...] = (
    FigureSpec(
        "method_comparison",
        "fig:method-comparison",
        "Replication means of FDP (FDR), true-positive rate (average power), and any-false-rejection events (FWER) for the configured correction procedures under the seeded two-groups model; error bars show the declared Monte-Carlo precision band.",
        "Three-panel bar chart comparing empirical FDR, average power, and FWER across the configured correction procedures, with uncertainty bars.",
    ),
    FigureSpec(
        "power_curve",
        "fig:power-curve",
        "Scenario-indexed BH average power and empirical FDR across the configured effect-size grid; three vertically aligned small multiples share the effect-size axis, while translucent ribbons, on-top capped error marks, and a lower log-scale panel show Monte-Carlo precision for replication means.",
        "Three-row small-multiple chart with one shared standardized-effect-size axis: BH average power and empirical FDR appear on separate panels, with translucent precision ribbons, capped error marks, a nominal alpha reference, and a lower log-scale precision panel.",
    ),
    FigureSpec(
        "dependence_sweep",
        "fig:dependence",
        "Empirical BH and BY FDR and average power across PSD-validated negative, independent, block, and factor-dependence regimes; BH uses circles and solid lines, BY uses squares and dashed lines, small horizontal offsets prevent overlapping marks, and error bars show Monte-Carlo precision.",
        "Two-panel uncertainty-aware dependence chart. BH circles with solid lines and BY squares with dashed lines are slightly horizontally offset within each declared regime, with error bars and a nominal-alpha FDR reference.",
    ),
    FigureSpec(
        "active_inference_policy",
        "fig:active-inference",
        "Replication means of FDR and average power for posterior-threshold decisions versus BH on reference-horizon calibrated evidence p-values; bars show the declared Monte-Carlo precision band and keep belief decisions separate from family-level evidence.",
        "Grouped uncertainty-aware bars comparing belief thresholding with BH on calibrated evidence p-values in strong- and weak-evidence regimes; point labels, Monte-Carlo caps, and a dotted nominal-alpha reference make the quantities legible without color alone.",
    ),
    FigureSpec(
        "certificate_summary",
        "fig:certificates",
        "Seeded certificate estimates with configured Monte-Carlo precision intervals; diamonds mark comparison bounds and labels distinguish control checks from adaptive calibration diagnostics.",
        "Horizontal forest plot of the configured seeded certificate estimates with uncertainty intervals, comparison bounds, and pass labels.",
    ),
    FigureSpec(
        "pvalue_calibration",
        "fig:pvalue-calibration",
        "Global-null p-value calibration in the independent regime; the empirical CDF is shown with the uniform reference and the configured DKW band.",
        "Empirical cumulative distribution of independent global-null p-values against the uniform reference with a DKW calibration band.",
    ),
    FigureSpec(
        "sample_size_precision",
        "fig:sample-size-precision",
        "Monte-Carlo precision and replication-budget accounting; observed t-based half-widths are compared with the configured target and the conservative Hoeffding planning floor.",
        "Two-panel chart showing Monte-Carlo FDR half-widths and planned replication counts against the conservative precision floor.",
    ),
    FigureSpec(
        "paired_method_contrasts",
        "fig:paired-contrasts",
        "Paired method-minus-BH differences from common random numbers; point intervals show Bonferroni-simultaneous finite-simulation confidence bands.",
        "Two-panel paired contrast chart showing method-minus-BH FDR and power differences with simultaneous confidence intervals.",
    ),
    FigureSpec(
        "sequential_evidence",
        "fig:sequential-evidence",
        "Representative null likelihood-ratio e-process evidence E_t, fixed-horizon p-value path, and inverted Bernoulli confidence sequence; the E_t panel is logarithmic.",
        "Three-panel sequential evidence plot showing logarithmic likelihood-ratio e-process evidence E_t with its Ville boundary, fixed-horizon p-values with the nominal threshold, and a Bernoulli confidence sequence with the null-rate reference.",
    ),
    FigureSpec(
        "online_fdr_trace",
        "fig:online-fdr",
        "Representative LORD++ and SAFFRON nominal levels, candidate/rejection events, and procedural alpha-wealth, alongside e-LOND nominal levels, reciprocal e-value decision thresholds, rejections, and cumulative nominal-level allocation under a null stream; the event rows and tick labels use one shared procedure map.",
        "Three-panel online-FDR diagnostic: LORD++ circles and solid lines occupy the top event row, SAFFRON squares and dashed lines the middle row, and e-LOND diamonds and dotted lines the bottom row. The top panel plots nominal levels alpha_t; e-LOND rejects an e-value only when it reaches the reciprocal threshold 1/alpha_t. The lower panel distinguishes procedural alpha-wealth from e-LOND cumulative nominal-level allocation.",
    ),
    FigureSpec(
        "conditional_selection_calibration",
        "fig:conditional-selection-calibration",
        "All-null selected-target calibration beside a deliberately invalid future-evidence minimum-p contrast; bars show maximum empirical CDF excess against a nominal iid DKW radius, while any n=0 target is a hatched no-selection cell rather than a zero estimate.",
        "Bar chart comparing chronological selected-target calibration with a future-evidence minimum-p failure contrast against nominal iid DKW radii. A gray hatched n=0 cell explicitly means no selected p-values and no calibration estimate, not zero excess.",
    ),
    FigureSpec(
        "benchmark_stress",
        "fig:benchmark-stress",
        "Offline benchmark stress metadata and the staged agent trajectory e-process; benchmark observations remain external stress tests and do not upgrade synthetic validity claims.",
        "Benchmark stress-test chart comparing synthetic evidence context with the offline staged agent trajectory.",
    ),
    FigureSpec(
        "scholarship_boundary",
        "fig:scholarship-boundary",
        "Scholarly boundary map linking primary literature families to the objects implemented here, the evidence actually generated, and the non-claims that remain; this is a provenance aid rather than a quantitative result.",
        "Six-row scholarship map connecting statistical, active-inference, sequential, selective-inference, and time-uniform literature to implemented objects, finite evidence, and explicit validity boundaries.",
    ),
    FigureSpec(
        "action_loop_trace",
        "fig:action-loop-trace",
        "One retained trace aligns actions, observations, posterior belief, calibrated p-values, target-specific likelihood-ratio e-process paths, the all-action global-null likelihood-ratio product shown only as a diagnostic, declared simulation cost C, and stopping. Target position and sensor marker/color identify actions; target paths use distinct marks and stop events use an X plus outline.",
        "Multi-panel action-loop trace showing target choice by vertical position and sensor choice by marker and color. Target posterior and target-specific e-process paths use distinct marker/line styles; the all-action global-null likelihood-ratio product is separately labeled as diagnostic. An X plus outline marks stopping alongside observation and cumulative declared simulation cost C.",
    ),
    FigureSpec(
        "conditional_power_surface",
        "fig:conditional-power-surface",
        "Four panels compare scenario-indexed replication-mean power, observed mean FDP, simulated cost, and stopping across declared action-loop cells and policies; capped intervals and the nominal-alpha reference are labeled.",
        "Four-panel conditional action-loop comparison of replication-mean power, observed mean FDP, total simulated cost C, and stopping across declared scenario cells and policies, with capped finite-simulation intervals, a directly labeled nominal-alpha reference in the observed-FDP panel, a shared policy legend, and a fixed-horizon optional-stopping boundary.",
    ),
    FigureSpec(
        "action_policy_frontier",
        "fig:action-policy-frontier",
        "Faceted policy power-cost comparison for declared action-loop scenarios; each panel fixes one scenario cell and shows finite-simulation intervals.",
        "Multi-panel power-cost frontier with one declared scenario cell per panel; colors and markers identify policies, horizontal and vertical intervals show finite-simulation cost and power uncertainty, and a footer names the fixed-horizon boundary.",
    ),
    FigureSpec(
        "scenario_decomposition",
        "fig:scenario-decomposition",
        "Investigator-facing decomposition of a declared design: the investigator specifies a model, testing setting, and simulated niche/process in a hashed scenario, pairs it with a separately evaluated agent policy, then repeatedly simulates and scores the design.",
        "Flow diagram showing an investigator specifying a model, a specified simulated niche/process, and a testing setting in a hashed PowerScenario. A separate agent policy joins repeated visible-history traces; evaluator-only hidden truth scores conditional true-positive rate, error rates, cost, stopping, and Monte-Carlo precision.",
    ),
    FigureSpec(
        "testing_setting_contract",
        "fig:testing-setting-contract",
        "Testing-setting contract for the release: the hypothesis family, null law and evidence interface, decision or accounting rule, default stopping rule and cost contract, estimands, and validity boundary are declared before simulation.",
        "Testing-setting contract diagram showing hypotheses and truth, the null law and evidence interface, multiple-testing or sequential accounting, a default stopping rule and cost contract, reported estimands, and the declared validity boundary.",
    ),
    FigureSpec(
        "formalism_code_map",
        "fig:formalism-code-map",
        "First print-sized formalism-to-code overview panel; each row shows a formal object, one representative code topic, one representative test topic, an evidence class, and an explicit non-claim boundary. Two following panels continue the registry; its hashed artifact retains the complete record.",
        "First print-sized traceability overview panel with formal-object row headings, evidence classes, representative code and test topics, and non-claim boundaries. Two following panels continue the registry, whose hashed artifact retains the complete symbols and paths.",
    ),
    FigureSpec(
        "formalism_code_map_continued",
        "fig:formalism-code-map-continued",
        "Second print-sized formalism-to-code overview panel; each row shows representative code and test topics with its evidence class and non-claim boundary, while the hashed registry retains complete symbols and paths.",
        "Second print-sized traceability overview with formal-object row headings, evidence classes, representative code and test topics, and non-claim boundaries.",
    ),
    FigureSpec(
        "formalism_code_map_final",
        "fig:formalism-code-map-final",
        "Final print-sized formalism-to-code overview panel; it continues the representative-topic mapping from formal objects to bounded evidence while the hashed registry retains complete symbols and paths.",
        "Final print-sized traceability overview with formal-object row headings, evidence classes, representative code and test topics, and non-claim boundaries.",
    ),
    FigureSpec(
        "adaptive_fdr_audit",
        "fig:adaptive-fdr-audit",
        "Metric-separated audit of the adaptive Storey comparison: the left panel shows BH and Storey empirical FDR with metric-specific Monte-Carlo precision, declared bounds, and finite gate ceilings; the right panel shows power, the paired Storey-minus-BH gain, and its separate lower-bound comparison.",
        "Two-panel adaptive-FDR audit. The left panel compares BH and Storey FDR estimates with metric-specific uncertainty, declared bounds, finite gate ceilings, and pass or fail labels. The right panel compares BH and Storey power and annotates the paired power gain separately.",
    ),
    FigureSpec(
        "computational_complexity",
        "fig:computational-complexity",
        "Log-scaled source-work curves compare vector, rank, and history-scan paths; bars give the release action-loop ledger and one combined line marks coincident binary-e-process/e-LOND proxies.",
        "Four-panel computational-work figure: log-scaled source-work curves for hypothesis-family and stream horizons, including one clearly labeled combined line for coincident binary-e-process and e-LOND linear proxies; horizontal bars for declared action-loop trace calls; and a boxed release ledger that distinguishes source-work from simulated cost C, likelihood-ratio e-process evidence, online alpha-wealth accounting, and local elapsed time.",
    ),
)


# Reader context is intentionally source-owned rather than inferred from a
# caption.  It makes every figure answer a concrete question, exposes the
# visual comparison that supports the answer, and preserves a second channel
# beyond color for readers of the dashboard, manuscript, and registry.
FIGURE_READER_CONTEXT: dict[str, dict[str, str]] = {
    "method_comparison": {
        "reader_question": "Within the declared two-groups scenario, how do correction procedures trade false discoveries, true positives, and any false rejection?",
        "takeaway": "The panels make the configured error-control trade-offs inspectable within one finite simulation, not universally ranked.",
        "visual_encoding": "Three labeled metric panels share procedure order; numeric bar labels and ±MC-SE caps accompany direct FWER-oriented (pale orange, left of the dashed divider) and FDR-oriented (pale blue, right) family labels, while dotted lines mark nominal alpha.",
    },
    "power_curve": {
        "reader_question": "For the declared fixed-horizon design, how do BH power and false-discovery summaries change across the effect-size grid?",
        "takeaway": "The panels support a design-specific comparison, not a population curve or universal monotonicity.",
        "visual_encoding": "Three vertically aligned small multiples share the effect-size axis: teal circles and a solid line encode power, magenta squares and a dashed line encode empirical FDR with a dotted nominal-alpha reference, and a lower log-scale panel directly plots both displayed MC-SE half-widths. Matching translucent ribbons and on-top caps show precision without relying on color alone.",
    },
    "dependence_sweep": {
        "reader_question": "How do BH and BY behave across the named covariance stress regimes?",
        "takeaway": "The comparison makes the declared dependence sensitivity visible but does not generalize to arbitrary dependence.",
        "visual_encoding": "Separate FDR and power panels use direct BH/BY legend labels, BH circles with solid lines, BY squares with dashed lines, small within-regime horizontal offsets that prevent mark overlap, compact regime-plus-rho labels, and MC-SE error bars.",
    },
    "active_inference_policy": {
        "reader_question": "What changes when posterior-threshold decisions are compared with BH applied to calibrated evidence p-values?",
        "takeaway": "Belief-based decisions and family-level evidence remain distinct objects even when shown side by side.",
        "visual_encoding": "Paired magenta and blue bars are labeled by decision layer, carry numeric values and MC-SE caps, and use a dotted alpha reference in the FDR panel.",
    },
    "certificate_summary": {
        "reader_question": "Which seeded certificates meet their declared replication-level error-rate or power comparison bounds?",
        "takeaway": "The forest plot separates finite-simulation agreement with a bound from theorem-level validity.",
        "visual_encoding": "Points with horizontal MC-SE intervals, diamond comparison bounds, explicit less-than-or-equal or greater-than-or-equal labels, and visible PASS/FAIL text encode direction and status without color alone.",
    },
    "pvalue_calibration": {
        "reader_question": "Are global-null p-values visually consistent with the uniform reference in the declared independent calibration regime?",
        "takeaway": "The ECDF is a regime-specific calibration diagnostic, not a general guarantee of p-value validity.",
        "visual_encoding": "A stepped empirical CDF is compared directly with a dashed Uniform(0, 1) line and a labeled shaded DKW reference band; the annotation names both the pooled p-value count and the independent replication count.",
    },
    "sample_size_precision": {
        "reader_question": "How precise are the simulated FDR summaries, and how is the replication budget allocated?",
        "takeaway": "Observed precision and conservative planning are shown as separate accounting quantities rather than interchangeable guarantees.",
        "visual_encoding": "Log-scaled, numerically labeled half-width bars are separated from horizontal replication bars; target and Hoeffding-floor references use labeled dashed lines.",
    },
    "paired_method_contrasts": {
        "reader_question": "Relative to BH on shared random streams, which configured procedures differ in FDR or power?",
        "takeaway": "Paired intervals reveal design-specific contrasts without establishing a global method ranking.",
        "visual_encoding": "Two metric panels use a direct method-minus-BH y-axis, point intervals, labeled method order, and a zero-difference reference line.",
    },
    "sequential_evidence": {
        "reader_question": "Which sequential summaries remain interpretable under optional stopping, and which do not?",
        "takeaway": "The e-process route supports sequential evidence interpretation; the fixed-horizon p-value is invalid if used for optional stopping.",
        "visual_encoding": "Teal circle markers show likelihood-ratio e-process evidence E_t and its Ville boundary on a log scale; magenta square markers show the fixed-horizon p-value path as invalid for optional stopping; blue bands show confidence-sequence endpoints.",
    },
    "online_fdr_trace": {
        "reader_question": "How do online procedures allocate accounting on a null stream?",
        "takeaway": "LORD++/SAFFRON procedural alpha-wealth and e-LOND cumulative allocation are accounting, not likelihood-ratio e-process evidence; e-LOND recycles levels.",
        "visual_encoding": "Three full-width stacked panels use LORD++ circles with solid lines, SAFFRON squares with dashed lines, and e-LOND diamonds with dotted lines. The top panel labels nominal level alpha_t and the e-LOND reciprocal decision rule E_t >= 1/alpha_t; the event raster's top-to-bottom LORD++/SAFFRON/e-LOND procedure labels derive from the same map as the marks. The lower panel labels procedural alpha-wealth separately from cumulative nominal-level allocation, without color alone.",
    },
    "conditional_selection_calibration": {
        "reader_question": "How does chronological target selection compare with a deliberately invalid future-evidence selector under the all-null diagnostic?",
        "takeaway": "The contrast exposes the filtration boundary; split-stream confirmation is reported separately.",
        "visual_encoding": "Category labels name the chronological and future-evidence selectors; paired bars show maximum ECDF excess and the nominal iid DKW radius, while a gray hatched n=0 cell explicitly means no selected p-values and no calibration estimate rather than a bar at zero. The footer states the diagnostic boundary.",
    },
    "benchmark_stress": {
        "reader_question": "What does the staged external trajectory contribute beyond the synthetic reference context?",
        "takeaway": "The figure supplies an external stress comparison without promoting the fixture to a validity claim.",
        "visual_encoding": "Directly labeled bars, provenance annotations, and an explicit stress-test note distinguish the synthetic reference from the staged trajectory.",
    },
    "scholarship_boundary": {
        "reader_question": "Which literature families motivate each implemented object, and what evidence or non-claim remains?",
        "takeaway": "The map keeps scholarly context, implementation, generated evidence, and unmade claims visibly separate.",
        "visual_encoding": "A four-column labeled matrix uses row headings, boxed cells, and evidence-class text rather than color alone to trace each literature layer.",
    },
    "action_loop_trace": {
        "reader_question": "What can an agent see on a simulated trace, and what is scored later?",
        "takeaway": "Posterior belief, p-values, target e-processes, the diagnostic all-action product, and simulated cost C are distinct; only target paths carry optional-stopping decisions.",
        "visual_encoding": "Aligned panels name posterior probabilities, p-value evidence, target-specific e-process evidence E_t, a separately styled all-action global-null likelihood-ratio product diagnostic, e-factors, scenario-defined simulation-cost units C, actions, observations, and stop marks. In the action panel, vertical position selects the target and marker/color selects the sensor; target traces pair distinct markers with solid, dashed, or dotted lines, and an X plus outline marks stopping. A footer names the fixed-horizon and no-family-error-control boundaries.",
    },
    "conditional_power_surface": {
        "reader_question": "How do policy-specific power, mean FDP, cost, and stopping compare?",
        "takeaway": "Declared agent-process designs—not an agent's intrinsic power or a distribution over environments—are compared; mean FDP is a finite-simulation estimand.",
        "visual_encoding": "Four labeled panels use shared policy colors plus markers and one figure-level legend; unique scenario-cell labels on the x-axis prevent equal-reliability cells from being conflated, capped bars show finite-simulation intervals, titles carry each estimand and unit, and the observed-mean-FDP panel directly labels its dotted nominal-alpha reference with an explicit no-control-claim boundary.",
    },
    "action_policy_frontier": {
        "reader_question": "Within each declared scenario, what power-cost trade-offs are visible among policies?",
        "takeaway": "The frontier supports conditional design choice, not an agent-optimality, utility, or deployment claim.",
        "visual_encoding": "Facets fix immutable scenario cells, whose titles use a compact scenario ID when needed plus a hash prefix and reliability; markers and colors identify policies, horizontal intervals show mean simulated cost C on the declared cost scale, vertical intervals show mean power uncertainty, and a footer states the fixed-horizon and finite-simulation boundaries.",
    },
    "scenario_decomposition": {
        "reader_question": "Who specifies the design and simulated niche, what can the agent observe, and what does the investigator estimate?",
        "takeaway": "Scenario plus policy defines the estimand; neither the agent nor the simulated niche has intrinsic power.",
        "visual_encoding": "A top investigator band declares the design, specified simulated niche, and replication plan; role headings, labeled boxes, and directional arrows show three inputs flowing into a hashed PowerScenario, a separate policy joining repeated simulated agent-niche traces, and evaluator-only scoring/output for conditional replication-mean TPR and MCSE without relying on color.",
    },
    "testing_setting_contract": {
        "reader_question": "What must be declared before simulated power or error summaries can be compared?",
        "takeaway": "The contract turns otherwise implicit hypothesis, evidence, accounting, default stopping, cost, and validity choices into auditable inputs.",
        "visual_encoding": "A six-box labeled flow uses headings, ordered arrows, and explicit validity text to distinguish testing-setting interfaces, accounting, and reported estimands; the separately evaluated agent policy is named in the heading text.",
    },
    "formalism_code_map": {
        "reader_question": "How does the first registry panel connect formal objects to executable symbols, tests, and limits of interpretation?",
        "takeaway": "Traceability supports auditability but does not convert implemented code into a proof.",
        "visual_encoding": "A print-sized three-column overview uses row headings, evidence-class text, one representative code topic, one representative test topic, and an explicit non-claim column; the registry hash and continuation-panel labels direct readers to the complete record.",
    },
    "formalism_code_map_continued": {
        "reader_question": "How do the next formalism entries connect to executable symbols, tests, and claim boundaries?",
        "takeaway": "The continuation remains a print-sized overview; the complete symbols, tests, assumptions, and non-claims remain in the registry artifact.",
        "visual_encoding": "The continuation uses the same three-column overview, registry range, hash, row labels, evidence-class text, one representative code topic, one representative test topic, and non-claim column as the first panel.",
    },
    "formalism_code_map_final": {
        "reader_question": "How do the final formalism entries complete the trace from formal object to bounded evidence?",
        "takeaway": "The final panel closes the registry without implying that traceability alone proves the listed claims.",
        "visual_encoding": "The final panel keeps the same labeled print-sized overview, registry range, hash, row headings, evidence-class text, one representative code topic, one representative test topic, and non-claim column for direct comparison.",
    },
    "adaptive_fdr_audit": {
        "reader_question": "How do the adaptive FDR and paired power components compare with their separate declared gate criteria?",
        "takeaway": "Separating FDR control checks from power gain prevents an apparent power improvement from being mistaken for a validity certificate.",
        "visual_encoding": "The FDR forest panel uses points, MC-SE intervals, diamonds, gate-ceiling bars, and PASS/FAIL text; separate power and paired-gain panels use their own labeled scales.",
    },
    "computational_complexity": {
        "reader_question": "What computational work must be declared before adaptive-study designs are compared?",
        "takeaway": "The curves and ledger keep workload separate from likelihood-ratio e-process evidence, online alpha-wealth, simulated resource cost C, and host-specific runtime.",
        "visual_encoding": "Log-scaled lines use legend labels, distinct markers, and line styles; one combined label makes the coincident binary-e-process and e-LOND linear proxies explicit. Horizontal bars label action-loop trace counts and maximum steps, while boxed text names the separate accounting quantities and the no timing boundary.",
    },
}


# This field names the outer evaluation role that a static plot cannot convey
# from marks alone.  It distinguishes an investigator's repeated simulation
# study from the embedded agent's within-replication beliefs and actions.
_FIXED_DESIGN_PERSPECTIVE = (
    "Investigator-facing simulation of a declared test over repeated generated "
    "datasets; no agent or task environment is assigned an intrinsic power."
)
_ACTION_DESIGN_PERSPECTIVE = (
    "Investigator-facing policy evaluation: an embedded agent acts only on "
    "visible history in the declared synthetic task environment (the specified "
    "simulated niche), while hidden truth is retained only for post-trace "
    "scoring across replications."
)
_AUDIT_PERSPECTIVE = (
    "Investigator-facing interpretation or audit aid; it does not estimate an "
    "agent's ability or a distribution of real-world environments."
)
_FIGURE_EVALUATION_PERSPECTIVES: dict[str, str] = {
    "method_comparison": _FIXED_DESIGN_PERSPECTIVE,
    "power_curve": _FIXED_DESIGN_PERSPECTIVE,
    "dependence_sweep": _FIXED_DESIGN_PERSPECTIVE,
    "active_inference_policy": _FIXED_DESIGN_PERSPECTIVE,
    "certificate_summary": _FIXED_DESIGN_PERSPECTIVE,
    "pvalue_calibration": _FIXED_DESIGN_PERSPECTIVE,
    "sample_size_precision": _FIXED_DESIGN_PERSPECTIVE,
    "paired_method_contrasts": _FIXED_DESIGN_PERSPECTIVE,
    "sequential_evidence": _FIXED_DESIGN_PERSPECTIVE,
    "online_fdr_trace": _FIXED_DESIGN_PERSPECTIVE,
    "conditional_selection_calibration": _ACTION_DESIGN_PERSPECTIVE,
    "benchmark_stress": _FIXED_DESIGN_PERSPECTIVE,
    "scholarship_boundary": _AUDIT_PERSPECTIVE,
    "action_loop_trace": _ACTION_DESIGN_PERSPECTIVE,
    "conditional_power_surface": _ACTION_DESIGN_PERSPECTIVE,
    "action_policy_frontier": _ACTION_DESIGN_PERSPECTIVE,
    "scenario_decomposition": _ACTION_DESIGN_PERSPECTIVE,
    "testing_setting_contract": _ACTION_DESIGN_PERSPECTIVE,
    "formalism_code_map": _AUDIT_PERSPECTIVE,
    "formalism_code_map_continued": _AUDIT_PERSPECTIVE,
    "formalism_code_map_final": _AUDIT_PERSPECTIVE,
    "adaptive_fdr_audit": _FIXED_DESIGN_PERSPECTIVE,
    "computational_complexity": _AUDIT_PERSPECTIVE,
}
if set(_FIGURE_EVALUATION_PERSPECTIVES) != set(FIGURE_READER_CONTEXT):
    raise RuntimeError("figure evaluation-perspective keys must match reader context")
for _figure_name, _perspective in _FIGURE_EVALUATION_PERSPECTIVES.items():
    FIGURE_READER_CONTEXT[_figure_name]["evaluation_perspective"] = _perspective


# Machine-readable scholarly metadata travels with every figure. Captions are
# still written as fluent prose, while these fields make the estimand,
# uncertainty convention, and interpretation boundary independently auditable.
FIGURE_METADATA: dict[str, dict[str, str]] = {
    "method_comparison": {
        "estimand": "Replication mean of empirical FDP, true-positive rate, and any-false-rejection indicator by procedure.",
        "uncertainty": "Bars show the configured Monte-Carlo standard-error multiplier.",
        "validity_boundary": "One seeded two-groups design; guarantees remain assumption-specific.",
    },
    "power_curve": {
        "estimand": "Replication mean of BH true-positive rate and FDP across the configured effect grid.",
        "uncertainty": "Ribbons and on-top caps show configured Monte-Carlo bands; a lower log-scale panel plots their displayed half-widths.",
        "validity_boundary": "Finite-simulation operating characteristic; no universal monotonicity or guarantee.",
    },
    "dependence_sweep": {
        "estimand": "Replication mean of BH/BY FDP and power across declared covariance regimes.",
        "uncertainty": "Point error bars show the configured Monte-Carlo precision band.",
        "validity_boundary": "Declared covariance stress does not establish arbitrary-dependence behavior.",
    },
    "active_inference_policy": {
        "estimand": "Replication mean of FDR and power for belief thresholding versus BH on calibrated evidence p-values.",
        "uncertainty": "Grouped bars show the configured Monte-Carlo precision band.",
        "validity_boundary": "Fixed binary channel and horizon; posterior probability is not an error-rate quantity.",
    },
    "certificate_summary": {
        "estimand": "Certificate estimate relative to its predeclared comparison bound.",
        "uncertainty": "Intervals show the configured Monte-Carlo precision band.",
        "validity_boundary": "Pass means finite-simulation consistency with a declared bound, not a theorem.",
    },
    "pvalue_calibration": {
        "estimand": "Empirical CDF of global-null p-values in the declared independent calibration regime.",
        "uncertainty": "The shaded region is the configured-confidence DKW band over the stated pooled p-value sample.",
        "validity_boundary": "Marginal calibration diagnostic for this null regime only.",
    },
    "sample_size_precision": {
        "estimand": "Observed Monte-Carlo half-widths and per-stage replication allocation.",
        "uncertainty": "t-based precision and conservative Hoeffding planning quantities are shown separately.",
        "validity_boundary": "Planning and accounting aid; estimands need not share a variance.",
    },
    "paired_method_contrasts": {
        "estimand": "Paired method-minus-BH differences in FDR and power under common random numbers.",
        "uncertainty": "Intervals are Bonferroni-simultaneous across the configured method-by-metric grid.",
        "validity_boundary": "Finite-simulation contrast, not a universal method ranking.",
    },
    "sequential_evidence": {
        "estimand": "Representative Bernoulli likelihood-ratio e-process evidence E_t, fixed-horizon p-value path, and inverted confidence-sequence endpoints.",
        "uncertainty": "The path is a representative diagnostic; the confidence sequence is time-uniform only under its declared Bernoulli mixture construction.",
        "validity_boundary": "Time-uniform scope is limited to the declared Bernoulli e-process null and confidence-sequence inversion; the p-value path is fixed-horizon.",
    },
    "online_fdr_trace": {
        "estimand": "Online nominal levels alpha_t, candidate events, rejections, and procedure-specific accounting paths under a null stream.",
        "uncertainty": "Finite-stream summaries are calibration diagnostics with separate Wilson accounting.",
        "validity_boundary": "For e-LOND, E_t >= 1/alpha_t is reciprocal; cumulative allocation is not alpha-wealth or likelihood-ratio e-process evidence. Scope requires declared arrival and evidence assumptions.",
    },
    "conditional_selection_calibration": {
        "estimand": "Maximum empirical CDF excess max(F_n(p)-p) among p-values selected by each target and pooled.",
        "uncertainty": "DKW radii are nominal iid reference radii at the declared confidence level; chronological action-step dependence is not removed.",
        "validity_boundary": "Finite all-null diagnostic for the declared model, process, setting, and policy; not selective-inference validity.",
    },
    "benchmark_stress": {
        "estimand": "Terminal e-process value E_T context for the synthetic reference and staged agent trajectory.",
        "uncertainty": "Provenance counts are shown; no population interval is claimed for the fixture.",
        "validity_boundary": "Offline benchmark stress cannot upgrade synthetic validity claims.",
    },
    "scholarship_boundary": {
        "estimand": "Literature role, implemented object, evidence class, and explicit non-claim boundary.",
        "uncertainty": "Not applicable: this is a provenance map, not a quantitative estimate.",
        "validity_boundary": "Scholarship and implementation context must not be read as measured evidence.",
    },
    "action_loop_trace": {
        "estimand": "One representative visible trajectory from an investigator-declared scenario/policy design; hidden truth is retained only for post-trace scoring.",
        "uncertainty": "Pathwise diagnostic with target-specific e-process paths and an all-action global-null likelihood-ratio product display; population summaries are reported separately with Wilson intervals.",
        "validity_boundary": "Hidden state is evaluation-only. Decisions are per target, not an FWER or FDR-control claim; fixed-horizon contrasts are invalid if used for optional stopping.",
    },
    "conditional_power_surface": {
        "estimand": "Replication means of true-positive rate (reported power), observed FDP, simulated cost, and stopping time indexed by the declared scenario/policy design.",
        "uncertainty": "Replication-mean power and observed-FDP summaries use Student-t Monte-Carlo intervals; binary decision/FWER rates use Wilson intervals; cost and stopping use Student-t mean intervals.",
        "validity_boundary": "Observed mean FDP is a finite-simulation estimand, not an FDR-control claim for correction-none or per-target e-process rows; fixed-horizon comparisons are invalid if used for optional stopping.",
    },
    "action_policy_frontier": {
        "estimand": "Investigator-estimated scenario/policy replication means of true-positive rate and total simulated cost.",
        "uncertainty": "Vertical and horizontal intervals use finite-simulation policy summaries; labels show replication means.",
        "validity_boundary": "Each facet fixes a declared scenario and cost scale; finite simulations do not establish policy optimality, and fixed-horizon comparisons are invalid if used for optional stopping.",
    },
    "scenario_decomposition": {
        "estimand": "Conceptual decomposition of the investigator-declared scenario/policy design and its replication-mean true-positive-rate estimand.",
        "uncertainty": "Not applicable: this is an interface diagram, not a quantitative estimate.",
        "validity_boundary": "The diagram makes conditioning explicit; it does not establish validity for any unimplemented scenario.",
    },
    "testing_setting_contract": {
        "estimand": "Declared hypothesis family, evidence interface, accounting rule, default stopping rule, cost contract, and operating characteristics for the testing setting.",
        "uncertainty": "Not applicable: this is a preregistration and interface diagram, not a quantitative estimate.",
        "validity_boundary": "A declared setting makes assumptions auditable; it does not establish validity outside its null, filtration, dependence, and stopping contracts.",
    },
    "formalism_code_map": {
        "estimand": "Registry rows linking formal objects to executable symbols, tests, artifacts, and claim boundaries.",
        "uncertainty": "Not applicable: this is a traceability matrix, not a quantitative estimate.",
        "validity_boundary": "Traceability makes assumptions auditable; it does not prove theorems or broaden statistical validity.",
    },
    "formalism_code_map_continued": {
        "estimand": "Continuation registry rows linking formal objects to executable symbols, tests, artifacts, and claim boundaries.",
        "uncertainty": "Not applicable: this is a traceability matrix, not a quantitative estimate.",
        "validity_boundary": "Traceability makes assumptions auditable; it does not prove theorems or broaden statistical validity.",
    },
    "formalism_code_map_final": {
        "estimand": "Final registry rows linking formal objects to executable symbols, tests, artifacts, and claim boundaries.",
        "uncertainty": "Not applicable: this is a traceability matrix, not a quantitative estimate.",
        "validity_boundary": "Traceability makes assumptions auditable; it does not prove theorems or broaden statistical validity.",
    },
    "adaptive_fdr_audit": {
        "estimand": "Separate BH/Storey empirical FDR estimates and Storey-minus-BH paired power gain under the declared two-groups certificate design.",
        "uncertainty": "Metric-specific point intervals show the configured Monte-Carlo standard-error multiplier; FDR gate ceilings are plotted separately from the estimate intervals.",
        "validity_boundary": "Finite-simulation audit of the declared design; a passing or failing gate is not a theorem, and same-stream adaptive calibration remains assumption-dependent.",
    },
    "computational_complexity": {
        "estimand": "Declared dimensionless source-work proxies and execution-path counts for the current package version and release input shape.",
        "uncertainty": "Not applicable: deterministic accounting has no Monte-Carlo, confidence, or elapsed-time error bars.",
        "validity_boundary": "Source-work accounting excludes dependency internals and supports no cross-hardware runtime or statistical-validity claim.",
    },
}

# Each registry row names the evidence class separately from its prose caption
# and validity boundary.  The visual-encoding contract ensures a reader is not
# asked to recover a category from color alone; figures may use different
# drawing primitives, but every one supplies a textual or shape-based second
# channel.
FIGURE_EVIDENCE_CLASSES: dict[str, str] = {
    "method_comparison": "conditional_operating_characteristic",
    "power_curve": "conditional_operating_characteristic",
    "dependence_sweep": "stress_test",
    "active_inference_policy": "conditional_operating_characteristic",
    "certificate_summary": "calibration_diagnostic",
    "pvalue_calibration": "calibration_diagnostic",
    "sample_size_precision": "provenance_contract",
    "paired_method_contrasts": "conditional_operating_characteristic",
    "sequential_evidence": "formal_conditional",
    "online_fdr_trace": "calibration_diagnostic",
    "conditional_selection_calibration": "calibration_diagnostic",
    "benchmark_stress": "stress_test",
    "scholarship_boundary": "provenance_contract",
    "action_loop_trace": "trace_diagnostic",
    "conditional_power_surface": "conditional_operating_characteristic",
    "action_policy_frontier": "conditional_operating_characteristic",
    "scenario_decomposition": "provenance_contract",
    "testing_setting_contract": "provenance_contract",
    "formalism_code_map": "provenance_contract",
    "formalism_code_map_continued": "provenance_contract",
    "formalism_code_map_final": "provenance_contract",
    "adaptive_fdr_audit": "calibration_diagnostic",
    "computational_complexity": "provenance_contract",
}

# A whole-figure validity class can be insufficient when a deliberately
# invalid comparison appears beside a formally conditional object.  The
# registry carries panel-level labels separately so the dashboard and the
# accessibility audit do not flatten the fixed-horizon p-value contrast into
# the e-process panel's narrower contract.
FIGURE_PANEL_VALIDITY: dict[str, tuple[dict[str, str], ...]] = {
    name: () for name in FIGURE_EVIDENCE_CLASSES
}
FIGURE_PANEL_VALIDITY["sequential_evidence"] = (
    {
        "panel": "Likelihood-ratio e-process evidence E_t",
        "validity": "Time-uniform only under the declared Bernoulli null, predictable update rule, and filtration.",
    },
    {
        "panel": "Fixed-horizon p-value path",
        "validity": "Invalid if used for optional stopping; shown only as a fixed-horizon reference contrast.",
    },
    {
        "panel": "Confidence-sequence inversion",
        "validity": "Time-uniform only under the declared Bernoulli mixture construction and filtration.",
    },
)
FIGURE_PANEL_VALIDITY["online_fdr_trace"] = (
    {
        "panel": "Nominal test level α_t",
        "validity": "For e-LOND, the plotted alpha_t is a nominal level with a reciprocal e-value decision rule E_t >= 1/alpha_t; it is not itself e-process evidence.",
    },
    {
        "panel": "Candidate and rejection events",
        "validity": "Finite null-stream diagnostic for the declared predictable recursions; it is not a proof of online error control.",
    },
    {
        "panel": "Procedure-specific accounting",
        "validity": "LORD++/SAFFRON show procedural alpha-wealth; e-LOND shows cumulative nominal-level allocation, which can recycle after rejection and is not alpha-wealth, not a remaining alpha budget, and not likelihood-ratio evidence.",
    },
)
FIGURE_PANEL_VALIDITY["action_loop_trace"] = (
    {
        "panel": "Target-specific e-process paths",
        "validity": "Conditional per-target optional-stopping interpretation requires the declared e-factor assumptions and makes no FWER or FDR claim.",
    },
    {
        "panel": "All-action global-null likelihood-ratio product",
        "validity": "Diagnostic display only; it is not the per-target decision object and supports no family-error-control claim.",
    },
    {
        "panel": "Fixed-horizon p-value contrast",
        "validity": "Invalid if used for optional stopping; shown only as a fixed-horizon reference contrast.",
    },
)
FIGURE_PANEL_VALIDITY["conditional_power_surface"] = (
    {
        "panel": "Observed mean FDP",
        "validity": "Finite-simulation estimand, not an FDR-control claim for correction-none or per-target e-process policies.",
    },
    {
        "panel": "Power, cost, and stopping summaries",
        "validity": "Conditional finite-simulation operating characteristics for the declared scenario, policy, and cost scale; not a universal design ranking.",
    },
    {
        "panel": "Fixed-horizon policy rows",
        "validity": "Invalid if used for optional stopping; retained only as a fixed-horizon comparison.",
    },
)
FIGURE_PANEL_VALIDITY["action_policy_frontier"] = (
    {
        "panel": "Scenario-faceted power-cost frontier",
        "validity": "Each facet conditions on its declared scenario and cost scale; intervals are finite-simulation uncertainty, not a policy-optimality result.",
    },
    {
        "panel": "Fixed-horizon policy rows",
        "validity": "Invalid if used for optional stopping; retained only as a fixed-horizon comparison.",
    },
)

def _reader_caption(
    reader_question: str,
    visible_comparison: str,
    takeaway: str,
    boundary: str,
) -> str:
    """Compose a fluent caption with the shared reader-first semantic order.

    The registry retains the four components as structured fields. The
    reader-facing caption uses them as connected prose rather than repeating
    editorial labels beneath every plot.
    """
    question = reader_question.strip()
    if question and question[-1] not in ".?!":
        question = f"{question}?"
    parts = (question, visible_comparison.strip(), takeaway.strip(), boundary.strip())
    return " ".join(part for part in parts if part)


_FIGURE_SPEC_NAMES = {spec.name for spec in FIGURE_SPECS}
if set(FIGURE_DASHBOARD_TITLES) != _FIGURE_SPEC_NAMES:
    missing_titles = sorted(_FIGURE_SPEC_NAMES - set(FIGURE_DASHBOARD_TITLES))
    extra_titles = sorted(set(FIGURE_DASHBOARD_TITLES) - _FIGURE_SPEC_NAMES)
    raise RuntimeError(
        "figure dashboard-title mismatch; "
        f"missing={missing_titles}, extra={extra_titles}"
    )
if set(FIGURE_READER_CONTEXT) != _FIGURE_SPEC_NAMES:
    missing_context = sorted(_FIGURE_SPEC_NAMES - set(FIGURE_READER_CONTEXT))
    extra_context = sorted(set(FIGURE_READER_CONTEXT) - _FIGURE_SPEC_NAMES)
    raise RuntimeError(
        "figure reader-context mismatch; "
        f"missing={missing_context}, extra={extra_context}"
    )
if any(
    not value.strip()
    for context in FIGURE_READER_CONTEXT.values()
    for value in context.values()
):
    raise RuntimeError("figure reader-context values must be non-empty")

# Preserve the original detailed descriptions as the visible-comparison part
# of the no-result API caption.  Result-bound captions below replace those
# summaries with the actual replication and scenario values from the artifact.
_SOURCE_VISIBLE_COMPARISONS = {spec.name: spec.caption for spec in FIGURE_SPECS}
FIGURE_SPECS = tuple(
    replace(
        spec,
        caption=_reader_caption(
            FIGURE_READER_CONTEXT[spec.name]["reader_question"],
            _SOURCE_VISIBLE_COMPARISONS[spec.name],
            FIGURE_READER_CONTEXT[spec.name]["takeaway"],
            FIGURE_METADATA[spec.name]["validity_boundary"],
        ),
        reader_question=FIGURE_READER_CONTEXT[spec.name]["reader_question"],
        visible_comparison=_SOURCE_VISIBLE_COMPARISONS[spec.name],
        takeaway=FIGURE_READER_CONTEXT[spec.name]["takeaway"],
        visual_encoding=FIGURE_READER_CONTEXT[spec.name]["visual_encoding"],
        evaluation_perspective=FIGURE_READER_CONTEXT[spec.name]["evaluation_perspective"],
    )
    for spec in FIGURE_SPECS
)
FIGURE_SPEC_BY_NAME = {spec.name: spec for spec in FIGURE_SPECS}
FIGURE_REDUNDANT_ENCODINGS: dict[str, str] = {
    spec.name: spec.visual_encoding for spec in FIGURE_SPECS
}


def result_lineage_hash(results: dict[str, Any]) -> str:
    """Hash scientific result content while excluding machine telemetry."""
    payload = json.loads(json.dumps(results, sort_keys=True, default=float))
    payload.pop("run_id", None)
    meta = payload.get("meta")
    if isinstance(meta, dict):
        meta.pop("python", None)
        meta.pop("platform", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def dashboard_result_lineage_hash(results: dict[str, Any]) -> str:
    """Return the full result-lineage witness shared by dashboard and figures.

    Release validators can use this helper instead of reimplementing the
    canonical result serialization.  It also verifies that every generated
    figure metadata row is bound to the same complete result witness.
    """
    expected = result_lineage_hash(results)
    figure_lineages = {
        item["result_lineage_hash"] for item in figure_metadata(results).values()
    }
    if figure_lineages != {expected}:
        raise ValueError("figure metadata does not share one result-lineage hash")
    return expected


def _scenario_hashes(results: dict[str, Any]) -> str:
    """Return the declared scenario lineage as a stable registry string."""
    hashes: set[str] = set()
    for block_name in ("suite", "action_loop"):
        block = results.get(block_name, {})
        if not isinstance(block, dict):
            continue
        candidates = block.get("scenario_contracts", block.get("scenarios", []))
        if not isinstance(candidates, list):
            continue
        for item in candidates:
            if not isinstance(item, dict):
                continue
            declared = item.get("scenario_hash")
            if isinstance(declared, str):
                hashes.add(declared)
                continue
            # ``PowerScenario.configuration_hash`` is the SHA-256 of this
            # exact JSON contract. Some nested release blocks omit
            # a redundant field, so derive the same digest here rather than
            # silently downgrading action figures to fixed-horizon lineage.
            encoded = json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=True)
            hashes.add(hashlib.sha256(encoded.encode("utf-8")).hexdigest())
        # The split-confirmation static channel is deliberately outside the
        # operating grid, but figures that use its diagnostic still need the
        # complete scenario lineage in their registry metadata.
        split_contract = block.get("split_confirmation_scenario")
        if isinstance(split_contract, dict):
            declared = split_contract.get("scenario_hash")
            if isinstance(declared, str):
                hashes.add(declared)
    return ";".join(sorted(hashes)) or "not_applicable_fixed_horizon"


def _figure_sample_sizes(results: dict[str, Any]) -> dict[str, str]:
    """Describe the replication unit behind each figure without hand literals."""
    meta = results.get("meta", {})
    main = _format_count(meta.get("n_reps", "configured"))
    active = _format_count(meta.get("active_inference_reps", "configured"))
    certificates = results.get("certificates", {})
    certificate_reps = max(
        (int(item.get("n_reps", 0)) for item in certificates.values() if isinstance(item, dict)),
        default=0,
    )
    paired = results.get("paired_comparisons", {})
    paired_reps = _format_count(paired.get("n_reps", main) if isinstance(paired, dict) else main)
    suite = results.get("suite", {})
    action_items = suite.get("operating_characteristics", []) if isinstance(suite, dict) else []
    action_reps = _format_count(action_items[0].get("n_reps", "configured") if action_items else "configured")
    calibration = results.get("null_pvalue_calibration", {})
    selection = suite.get("conditional_selection_calibration", {}) if isinstance(suite, dict) else {}
    return {
        "method_comparison": f"n={main} two-groups replications per procedure",
        "power_curve": f"n={main} replications per effect point",
        "dependence_sweep": f"n={main} replications per regime and procedure",
        "active_inference_policy": f"n={active} replications per active-inference regime",
        "certificate_summary": f"n={certificate_reps:,} replications per certificate check",
        "pvalue_calibration": (
            f"n={int(calibration.get('n_pvalues', 0)):,} pooled p-values from "
            f"{int(calibration.get('n_reps', 0)):,} independent calibration replications"
        ),
        "sample_size_precision": f"planned n={main} main replications; target-driven floor",
        "paired_method_contrasts": f"n={paired_reps} common-random-number replications",
        "sequential_evidence": f"n={active} sequential replications; one representative path",
        "online_fdr_trace": f"n={_format_count(results.get('online_fdr', {}).get('n_reps', active))} null-stream replications; one representative path",
        "conditional_selection_calibration": f"n={_format_count(selection.get('n_reps', 'configured'))} all-null selection replications",
        "benchmark_stress": "fixture-level counts; no population sample size claimed",
        "scholarship_boundary": f"n={main} release replications referenced; provenance map",
        "action_loop_trace": f"n={action_reps} action-loop replications; one representative path",
        "conditional_power_surface": f"n={action_reps} action-loop replications per scenario and policy",
        "action_policy_frontier": f"n={action_reps} action-loop replications per scenario and policy",
        "scenario_decomposition": f"n={main} main and n={action_reps} action-loop replications referenced",
        "testing_setting_contract": "not applicable; preregistered testing-setting contract",
        "formalism_code_map": "not applicable; registry traceability map",
        "formalism_code_map_continued": "not applicable; continuation registry traceability map",
        "formalism_code_map_final": "not applicable; final registry traceability map",
        "adaptive_fdr_audit": f"n={certificate_reps:,} certificate replications; BH and Storey FDR/power metrics separated",
        "computational_complexity": "not a sample-size result; declared source-work curves and release execution-path counts",
    }


def figure_metadata(results: dict[str, Any] | None = None) -> dict[str, dict[str, str]]:
    """Return complete figure metadata, including result and scenario lineage."""
    metadata = {name: dict(value) for name, value in FIGURE_METADATA.items()}
    if results is None:
        lineage = "0" * 64
        scenario_hashes = "not_available"
        sample_sizes = {name: "not available" for name in metadata}
    else:
        lineage = result_lineage_hash(results)
        scenario_hashes = _scenario_hashes(results)
        sample_sizes = _figure_sample_sizes(results)
    for name, values in metadata.items():
        values.update(
            {
                "evidence_class": FIGURE_EVIDENCE_CLASSES[name],
                "redundant_encoding": FIGURE_REDUNDANT_ENCODINGS[name],
                "sample_size": sample_sizes[name],
                "scenario_hashes": scenario_hashes,
                "result_lineage_hash": lineage,
            }
        )
    return metadata


_METHOD_LABELS = {
    "bonferroni": "Bonferroni",
    "sidak": "Šidák",
    "holm": "Holm",
    "hochberg": "Hochberg",
    "benjamini_hochberg": "BH",
    "benjamini_yekutieli": "BY",
    "storey_qvalue": "Storey q-value",
    "adaptive_bh": "Adaptive BH",
    "weighted_bh": "Weighted BH",
}
_METHOD_ORDER = {
    "bonferroni": 0,
    "sidak": 1,
    "holm": 2,
    "hochberg": 3,
    "benjamini_hochberg": 4,
    "benjamini_yekutieli": 5,
    "storey_qvalue": 6,
    "adaptive_bh": 7,
    "weighted_bh": 8,
}


def _method_label(name: str) -> str:
    """Use canonical method names in plots rather than implementation keys."""
    return _METHOD_LABELS.get(name, name.replace("_", " ").title())


def _ordered_methods(names: Any) -> list[str]:
    """Order methods by their declared error-control family and canonical name."""
    return sorted((str(name) for name in names), key=lambda name: _METHOD_ORDER.get(name, len(_METHOD_ORDER)))


def _format_count(value: Any) -> str:
    """Format a replication or method count for human-facing metadata."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "configured"


def _format_multiplier(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return "configured"


def _dependence_label(row: dict[str, Any], *, newline: bool = False) -> str:
    """Give every dependence point a unique regime-plus-correlation label."""
    regime = str(row.get("regime", row.get("dependence_kind", "declared regime")))
    rho = f"{float(row['rho']):g}"
    if newline:
        regime = regime.replace("equicorrelation", "equicorr.")
        regime = regime.replace(" ", "\n", 1)
        return f"{regime}\n(ρ={rho})"
    return f"{regime} (ρ={rho})"


def _formalism_panel_bounds(entry_count: int, panel_index: int) -> tuple[int, int]:
    """Return one inclusive, one-based formalism-registry panel range.

    The publication surface has three full-page panels.  Refuse an unsupported
    registry size rather than producing an apparently complete figure with
    missing or clipped rows.
    """
    if panel_index < 0 or panel_index >= _FORMALISM_PANEL_COUNT:
        raise ValueError(f"formalism panel index is out of range: {panel_index}")
    if entry_count < 1:
        raise ValueError("formalism registry must contain at least one entry")
    capacity = _FORMALISM_ROWS_PER_PANEL * _FORMALISM_PANEL_COUNT
    if entry_count > capacity:
        raise ValueError(
            "formalism registry exceeds the three-panel publication layout; "
            "add a panel before rendering"
        )
    start = panel_index * _FORMALISM_ROWS_PER_PANEL + 1
    if start > entry_count:
        raise ValueError(f"formalism panel {panel_index + 1} has no registry entries")
    return start, min(start + _FORMALISM_ROWS_PER_PANEL - 1, entry_count)


def _formalism_code_map_layout(row_count: int) -> dict[str, float]:
    """Return geometry for a print-sized, non-overlapping registry overview."""
    if row_count < 1 or row_count > _FORMALISM_ROWS_PER_PANEL:
        raise ValueError(
            f"formalism panel must contain 1–{_FORMALISM_ROWS_PER_PANEL} rows"
        )
    header_y = 0.81
    footer_top = 0.10
    row_height = (header_y - footer_top) / row_count
    row_gap = min(0.008, row_height * 0.14)
    return {
        "header_y": header_y,
        "footer_top": footer_top,
        "row_height": row_height,
        "row_gap": row_gap,
        "last_row_bottom": footer_top,
    }


def _formalism_display_reference(reference: object, *, test_path: bool = False) -> str:
    """Return a word-spaced publication label for a registry reference.

    The formalism registry keeps the exact executable symbols and test paths.
    Its print-sized matrix instead uses this display-only label, so an
    underscore-delimited or camel-cased identifier cannot be broken mid-token
    across a narrow row.  That keeps the visual map readable without weakening
    the registry's source-level traceability.
    """
    display = str(reference)
    if test_path:
        display = Path(display).stem
        if display.startswith("test_"):
            display = f"{display.removeprefix('test_')} test"
    display = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", display)
    display = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", display)
    display = re.sub(r"[._/\\-]+", " ", display)
    return re.sub(r"\s+", " ", display).strip()


def figure_visible_comparisons(results: dict[str, Any] | None = None) -> dict[str, str]:
    """Return the result-bound comparison each reader-first caption reports.

    This is deliberately distinct from both the visual-encoding description
    and the fluent caption.  The registry and dashboard expose it as the
    second reader-first component, while :func:`figure_captions` joins it with
    the question, takeaway, and validity boundary without editorial labels.
    """
    defaults = {spec.name: spec.visible_comparison for spec in FIGURE_SPECS}
    if results is None:
        return defaults
    meta = results.get("meta", {})
    n_reps = _format_count(meta.get("n_reps"))
    method_count = _format_count(len(meta.get("methods", ())))
    multiplier = _format_multiplier(meta.get("figure_error_multiplier", 3.0))
    confidence = f"{100.0 * float(meta.get('mc_confidence', 0.95)):g}%"
    band = f"±{multiplier} MC SE"
    dependence = results.get("dependence_sweep", [])
    calibration = results.get("null_pvalue_calibration", {})
    calibration_reps = _format_count(calibration.get("n_reps"))
    calibration_pvalues = _format_count(calibration.get("n_pvalues"))
    active_reps = _format_count(meta.get("active_inference_reps", "configured"))
    certificate_reps = max(
        (int(item.get("n_reps", 0)) for item in results.get("certificates", {}).values() if isinstance(item, dict)),
        default=0,
    )
    paired_reps = _format_count(
        results.get("paired_comparisons", {}).get("n_reps", n_reps)
        if isinstance(results.get("paired_comparisons", {}), dict)
        else n_reps
    )
    sequential_reps = _format_count(results.get("sequential", {}).get("n_reps", active_reps))
    online_reps = _format_count(results.get("online_fdr", {}).get("n_reps", active_reps))
    action_loop = results.get("action_loop", {})
    action_scenarios = len(action_loop.get("scenarios", []))
    action_operations = action_loop.get("operating_characteristics", [])
    suite = results.get("suite", {})
    selection_calibration = (
        suite.get("conditional_selection_calibration", {})
        if isinstance(suite, dict)
        else {}
    )
    selected_target_rows = selection_calibration.get("selected_pvalues_by_target", [])
    no_selection_targets = (
        sum(
            int(row.get("n_selected", 0)) == 0
            for row in selected_target_rows
            if isinstance(row, dict)
        )
        if isinstance(selected_target_rows, list)
        else 0
    )
    no_selection_note = (
        f" {no_selection_targets} target row(s) are gray-hatched n=0 cells: no selected p-values and no calibration estimate, not zero excess."
        if no_selection_targets
        else ""
    )
    action_reps = _format_count(
        action_operations[0].get("summary", {}).get("n_reps", "configured")
        if action_operations else "configured"
    )
    formal_entry_count = int(results.get("formalism", {}).get("entry_count", 0))
    formal_first = _formalism_panel_bounds(formal_entry_count, 0)
    formal_second = _formalism_panel_bounds(formal_entry_count, 1)
    formal_third = _formalism_panel_bounds(formal_entry_count, 2)
    complexity = results.get("computational_complexity", {})
    complexity_action = (
        complexity.get("release_workload", {}).get("action_loop", {})
        if isinstance(complexity, dict)
        else {}
    )
    complexity_trace_calls = _format_count(complexity_action.get("action_loop_trace_calls"))
    complexity_step_bound = _format_count(complexity_action.get("action_loop_step_upper_bound"))
    visible_comparisons = {
        "method_comparison": (
            f"Empirical FDR, average power, and FWER for {method_count} configured "
            f"correction procedures under the seeded two-groups model; pale orange shading left of the dashed divider identifies FWER-oriented methods and pale blue shading right identifies FDR-oriented methods. Bars show "
            f"means with {band} across {n_reps} replications. The dotted line marks "
            f"the nominal alpha level ({float(meta.get('alpha', 0.05)):g})."
        ),
        "power_curve": (
            f"Three aligned BH panels share the configured effect-size axis: power, empirical FDR, and precision half-widths. "
            f"Shaded ribbons, on-top capped error marks, and the lower log-scale half-width panel show {band} across {n_reps} replications."
        ),
        "dependence_sweep": (
            f"BH circles/solid lines and BY squares/dashed lines show empirical FDR and average power across the {len(dependence)} "
            f"declared covariance regimes (including ρ=-0.004 and ρ=0.75); error bars show {band}. This is a finite-simulation stress comparison, not arbitrary-dependence evidence."
        ),
        "active_inference_policy": (
            f"Empirical FDR and average power for posterior-threshold decisions versus "
            f"BH on calibrated evidence p-values; bars show {band} across the "
            f"configured active-inference replications ({active_reps} per regime). "
            "Belief thresholds and family-level evidence are intentionally shown "
            "as separate decision layers; posterior probability is not treated as "
            "an error-rate quantity."
        ),
        "certificate_summary": (
            f"Seeded certificate estimates from {certificate_reps:,} replications per check with {band} intervals; diamonds mark "
            "comparison bounds. Control checks, the same-vector Storey power comparison, "
            "and the independent split-stream Storey calibration are separated by panel "
            "and label. Intervals quantify finite-simulation precision and are not proof "
            "certificates by themselves."
        ),
        "pvalue_calibration": (
            f"Global-null p-value calibration in the independent regime; the "
            f"empirical CDF is shown with the uniform reference and a {confidence} DKW band "
            f"over {calibration_pvalues} pooled p-values from {calibration_reps} "
            "independent calibration replications."
        ),
        "sample_size_precision": (
            f"Monte-Carlo precision and replication-budget accounting; observed "
            f"half-widths from the {n_reps} main replications are compared with the "
            f"configured target and conservative planning floor for the {confidence} summaries."
        ),
        "paired_method_contrasts": (
            f"Paired method-minus-BH differences from common random numbers; point "
            f"intervals show Bonferroni-simultaneous finite-simulation confidence bands over "
            f"the {method_count} configured procedures and {paired_reps} paired replications."
        ),
        "sequential_evidence": (
            f"Over {int(results.get('sequential', {}).get('horizon', 0))} observations from {sequential_reps} replications, panels show representative null "
            "likelihood-ratio e-process evidence E_t, fixed-horizon p-values, and an inverted "
            "Bernoulli confidence sequence; the E_t axis is logarithmic."
        ),
        "online_fdr_trace": (
            f"Across {int(results.get('online_fdr', {}).get('horizon', 0))} null tests from {online_reps} replications, panels show LORD++/SAFFRON "
            "nominal levels, events, and procedural alpha-wealth; e-LOND nominal levels, decisions, and cumulative allocation. "
            "Rows/ticks run top-to-bottom LORD++/SAFFRON/e-LOND; shapes and lines distinguish procedures without color."
        ),
        "conditional_selection_calibration": (
            "Bars compare maximum empirical CDF excess for "
            f"{_format_count(selection_calibration.get('pooled', {}).get('n_selected', 0))} chronological action-step p-values and "
            f"{_format_count((suite.get('future_evidence_selection_failure') or {}).get('pooled', {}).get('n_selected', 0))} post-hoc selected p-values against a nominal iid DKW radius."
            f"{no_selection_note}"
        ),
        "benchmark_stress": (
            f"Offline benchmark stress comparison of the synthetic-null and staged-agent terminal e-process values E_T "
            f"({int(results.get('benchmarks', {}).get('agent_trace', {}).get('rows', 0))} staged rows)."
        ),
        "scholarship_boundary": (
            "Scholarly boundary map linking FDR and dependence theory, active-"
            "inference generative modeling, sequential evidence, and online FDR "
            "to implemented objects, evidence classes, and explicit non-claims. "
            "This is a provenance aid, not a quantitative result or a simulation-derived estimate."
        ),
        "action_loop_trace": (
            f"One trace from {action_reps} declared replications aligns actions, observations, posterior belief, p-value evidence, "
            "target e-process paths, an all-action global-null product, declared simulation cost C, and stopping. "
            "Target paths—not the product—carry decisions; X outline marks stopping."
        ),
        "conditional_power_surface": (
            f"Across {action_scenarios} investigator-declared scenario/policy cells, panels show replication-mean power, observed mean FDP, "
            f"simulated cost C, and stopping from {action_reps} replications. Student-t/Wilson intervals are shown; "
            "the observed-mean-FDP panel directly labels its dotted nominal-alpha reference."
        ),
        "action_policy_frontier": (
            f"Across {action_scenarios} investigator-declared scenario cells, facets compare replication-mean power with simulated cost C "
            f"from {action_reps} action-loop replications; horizontal and vertical intervals show finite-simulation uncertainty."
        ),
        "scenario_decomposition": (
            "Investigator-facing design decomposition: a model, specified simulated niche/process, and testing setting form a hashable scenario; a separate agent policy completes it. "
            "Evaluator-only truth scores conditional replication-mean true-positive rate, error, cost, and stopping."
        ),
        "testing_setting_contract": (
            "Testing-setting contract for the declared release: hypothesis family, null law and "
            "evidence interface, accounting procedure, default stopping rule and cost contract, estimands, and "
            "validity boundary are fixed before simulation. This visualization records the "
            "interface contract and does not add an operating-characteristic estimate."
        ),
        "formalism_code_map": (
            f"First formalism-to-code traceability matrix panel for registry entries {formal_first[0]}–{formal_first[1]}; "
            "each print-sized row shows a formal object, evidence class, representative code and test topics, and an explicit non-claim boundary. "
            "The hashed registry retains complete symbols, test paths, assumptions, and non-claims; two separately rendered panels continue the overview."
        ),
        "formalism_code_map_continued": (
            f"Second formalism-to-code traceability matrix panel for registry entries "
            f"{formal_second[0]}–{formal_second[1]}; "
            "each print-sized row shows evidence class, representative code/test topics, and an explicit non-claim boundary; the hashed registry retains the complete record."
        ),
        "formalism_code_map_final": (
            f"Final formalism-to-code traceability matrix panel for registry entries "
            f"{formal_third[0]}–{formal_third[1]}; "
            "it completes the print-sized overview of evidence class, representative code/test topics, and explicit non-claim boundaries; the hashed registry retains the complete record."
        ),
        "adaptive_fdr_audit": (
            f"Metric-separated adaptive-FDR audit from {certificate_reps:,} certificate replications. "
            "The FDR panel shows BH and Storey estimates with metric-specific Monte-Carlo intervals, "
            "declared bounds, finite gate ceilings, and gate status; the power panel shows BH and Storey "
            "power and the paired Storey-minus-BH gain."
        ),
        "computational_complexity": (
            "Log-scaled source-work accounting curves compare vector, rank, and history-scan work; one combined line marks the coincident "
            "binary-e-process/e-LOND proxies. The action-loop ledger reports "
            f"{complexity_trace_calls} trace calls and at most {complexity_step_bound} permitted slots. Marks are deterministic, "
            "not timing measurements or statistical uncertainty."
        ),
    }
    return visible_comparisons


def figure_captions(results: dict[str, Any] | None = None) -> dict[str, str]:
    """Return fluent captions in question, comparison, takeaway, boundary order."""
    return {
        name: _reader_caption(
            FIGURE_SPEC_BY_NAME[name].reader_question,
            visible_comparison,
            FIGURE_SPEC_BY_NAME[name].takeaway,
            FIGURE_METADATA[name]["validity_boundary"],
        )
        for name, visible_comparison in figure_visible_comparisons(results).items()
    }


def figure_alts(results: dict[str, Any] | None = None) -> dict[str, str]:
    """Return accessible alternative text, optionally expanded from results."""
    defaults = {spec.name: spec.alt for spec in FIGURE_SPECS}
    if results is None:
        return defaults
    meta = results.get("meta", {})
    method_count = _format_count(len(meta.get("methods", ())))
    n_reps = _format_count(meta.get("n_reps"))
    calibration = results.get("null_pvalue_calibration", {})
    calibration_pvalues = _format_count(calibration.get("n_pvalues"))
    calibration_reps = _format_count(calibration.get("n_reps"))
    dependence = results.get("dependence_sweep", [])
    regimes = ", ".join(_dependence_label(row) for row in dependence)
    horizon = int(results.get("sequential", {}).get("horizon", 0))
    online_horizon = int(results.get("online_fdr", {}).get("horizon", 0))
    agent_rows = int(results.get("benchmarks", {}).get("agent_trace", {}).get("rows", 0))
    suite = results.get("suite", {})
    selection_calibration = (
        suite.get("conditional_selection_calibration", {})
        if isinstance(suite, dict)
        else {}
    )
    selected_target_rows = selection_calibration.get("selected_pvalues_by_target", [])
    no_selection_targets = (
        sum(
            int(row.get("n_selected", 0)) == 0
            for row in selected_target_rows
            if isinstance(row, dict)
        )
        if isinstance(selected_target_rows, list)
        else 0
    )
    no_selection_alt = (
        f" The {no_selection_targets} gray hatched n=0 cell(s) mean no selected p-values and no calibration estimate, not zero excess."
        if no_selection_targets
        else ""
    )
    formal_entry_count = int(results.get("formalism", {}).get("entry_count", 0))
    formal_first = _formalism_panel_bounds(formal_entry_count, 0)
    formal_second = _formalism_panel_bounds(formal_entry_count, 1)
    formal_third = _formalism_panel_bounds(formal_entry_count, 2)
    complexity = results.get("computational_complexity", {})
    complexity_action = (
        complexity.get("release_workload", {}).get("action_loop", {})
        if isinstance(complexity, dict)
        else {}
    )
    complexity_trace_calls = _format_count(complexity_action.get("action_loop_trace_calls"))
    complexity_step_bound = _format_count(complexity_action.get("action_loop_step_upper_bound"))
    return {
        **defaults,
        "method_comparison": (
            f"Multi-panel bar chart for {method_count} correction procedures, "
            f"showing empirical FDR, average power, and FWER over {n_reps} seeded "
            "replications with precision bars and a nominal-alpha reference. Pale orange shading left of the dashed divider labels FWER-oriented procedures; pale blue shading right labels FDR-oriented procedures."
        ),
        "power_curve": (
            "Three vertically aligned small multiples share the configured effect-size axis: teal circles and a solid line show BH average power, magenta squares and a dashed line show empirical FDR with a dotted nominal-alpha reference, and a lower log-scale panel plots both displayed Monte-Carlo half-widths. Translucent ribbons and capped error marks show precision."
        ),
        "dependence_sweep": (
            f"Two-panel dependence stress plot across {regimes}; BH circles with solid lines and BY squares with dashed lines are slightly horizontally offset within each regime, so FDR or power marks remain distinct with uncertainty bars and a nominal-alpha reference."
        ),
        "active_inference_policy": (
            "Grouped bars compare belief-threshold decisions with BH on evidence "
            "p-values in strong- and weak-evidence regimes, with point labels and "
            "Monte-Carlo caps; a dotted nominal-alpha reference appears in the FDR panel."
        ),
        "certificate_summary": (
            "Two-panel horizontal forest plot showing certificate point estimates, "
            "Monte-Carlo precision intervals, diamond comparison bounds, and explicit PASS or FAIL text."
        ),
        "pvalue_calibration": (
            "Global-null p-value ECDF with a uniform reference, a DKW band, and a "
            f"calibration statistic annotation naming {calibration_pvalues} pooled p-values "
            f"from {calibration_reps} calibration replications."
        ),
        "sample_size_precision": (
            "Two-panel accounting plot with logarithmic FDR precision half-widths "
            "and replication budgets compared with the conservative planning floor."
        ),
        "paired_method_contrasts": (
            "Two-panel paired method-minus-BH contrast plot with simultaneous "
            "uncertainty intervals and a zero-difference reference."
        ),
        "sequential_evidence": (
            f"Sequential plot over {horizon} observations comparing logarithmic "
            "likelihood-ratio e-process evidence E_t with its Ville boundary, fixed-horizon p-values with the "
            "nominal threshold, and a Bernoulli confidence sequence with the null-rate "
            "reference."
        ),
        "online_fdr_trace": (
            f"Online-FDR diagnostic with three stacked panels over {online_horizon} tests: logarithmic "
            "nominal levels alpha_t with LORD++ circles/solid lines, SAFFRON squares/dashed lines, and e-LOND diamonds/dotted lines; e-LOND decisions use E_t >= 1/alpha_t; a top-to-bottom LORD++/SAFFRON/e-LOND procedure-row candidate/rejection raster titled | = candidate and * = rejection; and separately labeled procedural alpha-wealth versus cumulative nominal-level allocation, not likelihood-ratio e-process evidence."
        ),
        "conditional_selection_calibration": (
            "All-null calibration bars compare chronological target-selected p-values with "
            "a future-evidence minimum-p failure contrast, each against a nominal iid DKW "
            "reference radius; any target with no selected p-values is shown as a gray hatched n=0 no-selection cell rather than a zero estimate."
            f"{no_selection_alt}"
        ),
        "benchmark_stress": (
            f"Benchmark stress bar chart comparing synthetic and staged agent "
            f"terminal e-process values E_T; the external trajectory contains {agent_rows} rows "
            "and is labeled as stress-test evidence only."
        ),
        "scholarship_boundary": (
            "Six-row map with columns for literature layer, implementation, "
            "evidence class, and explicit boundary: statistical procedures, "
            "binary active inference, sequential e-processes, online FDR, "
            "selective inference, and time-uniform concentration."
        ),
        "action_loop_trace": (
            "Eight aligned panels show action target by vertical position and sensor by marker/color, binary observation with an X plus outline for stopping, global "
            "and target posterior beliefs with distinct marker/line styles, p-value evidence, target-specific e-process evidence E_t with distinct marker/line styles, and an all-action global-null likelihood-ratio product marked diagnostic. Declared simulation cost C and stopping are shown; hidden evaluation state is intentionally absent, and a footer states the fixed-horizon optional-stopping boundary and per-target no-family-error-control boundary."
        ),
        "conditional_power_surface": (
            "Multi-panel conditional power surface across uniquely labeled declared scenario cells and action "
            "policies, showing power, observed mean FDP, mean cost, and mean stopping time with capped intervals, a directly labeled nominal-alpha reference, a shared policy legend, and a fixed-horizon optional-stopping boundary plus a no-FDR-control boundary for correction-none or per-target e-process rows."
        ),
        "action_policy_frontier": (
            "Multi-panel power-cost scatter plot for action-loop policies, with one declared "
            "scenario ID/reliability cell per panel, shared power scale, horizontal cost intervals, vertical power "
            "intervals, an explicit policy legend, and a footer naming the scenario-specific cost-scale, finite-simulation, and fixed-horizon optional-stopping boundary."
        ),
        "scenario_decomposition": (
            "Flow diagram showing an investigator specifying a hashed PowerScenario from a generative model, a specified simulated niche/process, and a testing setting; a separate agent policy joins repeated visible-history traces, while evaluator-only hidden truth is used to score conditional replication-mean true-positive rate, error rates, cost, stopping, and Monte-Carlo precision."
        ),
        "testing_setting_contract": (
            "Contract diagram showing the testing-setting inputs and outputs: hypotheses and truth, "
            "null law and evidence interface, fixed-family or sequential accounting, stopping and "
            "cost contract, reported estimands, and the validity boundary declared for interpretation."
        ),
        "formalism_code_map": (
            f"First traceability matrix panel connecting registry entries {formal_first[0]} through {formal_first[1]} to public package "
            "representative code and test topics, evidence classes, and non-claim boundaries; two following print-sized overview panels continue the registry, whose hash directs readers to complete symbols and paths."
        ),
        "formalism_code_map_continued": (
            f"Second traceability matrix panel connecting registry entries {formal_second[0]} through {formal_second[1]} "
            "in a print-sized overview to representative code and test topics, evidence classes, and non-claim boundaries; the hashed registry retains complete symbols and paths."
        ),
        "formalism_code_map_final": (
            f"Final traceability matrix panel connecting registry entries {formal_third[0]} through {formal_third[1]} "
            "in a print-sized overview to representative code and test topics, evidence classes, and non-claim boundaries; the hashed registry retains complete symbols and paths."
        ),
        "adaptive_fdr_audit": (
            "Two-panel certificate audit: the left FDR panel separates BH and Storey point estimates, "
            "metric-specific precision intervals, declared bounds, finite gate ceilings, and pass or fail "
            "labels; the right power panel shows BH and Storey power and the paired power gain."
        ),
        "computational_complexity": (
            "Four-panel deterministic computational-work figure. The top row uses log-scaled lines with "
            "legends, distinct markers, and line styles for hypothesis-family and stream source-work proxies; one combined line "
            "explicitly represents the coincident binary-e-process and e-LOND linear proxies. The lower left "
            f"uses labeled horizontal bars for action-loop trace categories totaling {complexity_trace_calls} calls and {complexity_step_bound} maximum permitted steps. "
            "The lower right boxed ledger separates source work from simulated cost C, likelihood-ratio e-process evidence, online alpha-wealth accounting, and local wall time."
        ),
    }


def _figure_error_multiplier(results: dict[str, Any]) -> float:
    return float(results.get("meta", {}).get("figure_error_multiplier", 3.0))


def _figure_dpi(results: dict[str, Any]) -> int:
    return int(results.get("meta", {}).get("figure_dpi", 300))


def _save(fig: plt.Figure, path: Path, dpi: int = 300) -> str:
    """Save a figure using fixed publication settings and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    try:
        fig.savefig(temporary_name, dpi=dpi, bbox_inches="tight", facecolor="white")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    finally:
        plt.close(fig)
    return str(path)


def _three_se(values: list[float], multiplier: float = 3.0) -> np.ndarray:
    """Convert standard errors to the configured plotted precision band."""
    return float(multiplier) * np.asarray(values, dtype=float)


def _power_curve_errors(
    data: list[dict[str, Any]], multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Read current power-curve MCSE fields with an earlier-schema fallback."""
    power_stderr = [
        float(row.get("mc_stderr_power", row.get("power_stderr", 0.0)))
        for row in data
    ]
    fdr_stderr = [
        float(row.get("mc_stderr_fdr", row.get("fdr_stderr", 0.0)))
        for row in data
    ]
    return (
        _three_se(power_stderr, multiplier),
        _three_se(fdr_stderr, multiplier),
    )


def _annotate_bars(
    ax: plt.Axes,
    bars: Any,
    values: list[float],
    errors: np.ndarray,
    *,
    places: int = 3,
    upper_limit: float | None = None,
) -> None:
    """Put compact point estimates above bars without hiding error caps."""
    for bar, value, error in zip(bars, values, errors):
        y = float(value) + float(error) + 0.012
        if upper_limit is not None:
            y = min(y, upper_limit)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            f"{float(value):.{places}f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=COLORS["slate"],
        )


def _method_comparison(results: dict[str, Any], path: Path) -> str:
    data = results["method_comparison"]
    labels = _ordered_methods(data)
    display = [_method_label(label) for label in labels]
    x = np.arange(len(labels))
    alpha = float(results["meta"]["alpha"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharex=True)
    for ax, key, title, color in zip(
        axes,
        ("fdr", "power", "fwer"),
        ("False discovery rate", "Average power", "Family-wise error rate"),
        (COLORS["blue"], COLORS["teal"], COLORS["orange"]),
    ):
        values = np.asarray([float(data[label][key]) for label in labels])
        stderr_key = f"mc_stderr_{key}"
        errors = _three_se(
            [float(data[label].get(stderr_key, 0.0)) for label in labels],
            _figure_error_multiplier(results),
        )
        bars = ax.bar(
            x,
            values,
            yerr=errors,
            capsize=4,
            color=color,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.3f}" for value in values],
            padding=2,
            fontsize=6,
        )
        ax.set_title(title)
        if key == "fdr":
            ax.set_ylim(0, max(0.1, max(values + errors) * 1.15))
        else:
            ax.set_ylim(0, 1.05)
        if key in {"fdr", "fwer"}:
            ax.axhline(
                alpha,
                color=COLORS["slate"],
                linestyle=":",
                linewidth=1,
                label=f"nominal α={alpha:g}",
            )
        ax.grid(axis="y", color=COLORS["grid"], alpha=0.55, linewidth=0.8)
        ax.set_xticks(x, display, rotation=55, ha="right", fontsize=8)
    fwer_count = sum(
        name in {"bonferroni", "sidak", "holm", "hochberg"} for name in labels
    )
    split = float(fwer_count) - 0.5
    for ax in axes:
        ax.axvspan(-0.5, split, color=COLORS["orange"], alpha=0.035, zorder=0)
        ax.axvspan(split, len(labels) - 0.5, color=COLORS["blue"], alpha=0.035, zorder=0)
        ax.axvline(split, color=COLORS["grid"], linewidth=1.2, linestyle="--")
    axes[0].set_ylabel(
        "Replicate mean FDP\n(empirical FDR; proportion; linear)",
        fontsize=8,
        labelpad=4,
    )
    axes[1].set_ylabel(
        "Replicate mean power\n(true-positive rate; proportion; linear)",
        fontsize=8,
        labelpad=4,
    )
    axes[2].set_ylabel(
        "Replicate mean FWER\n(any false rejection; proportion; linear)",
        fontsize=8,
        labelpad=4,
    )
    axes[0].legend(fontsize=8, loc="upper left", frameon=True)
    fig.suptitle(
        f"Empirical operating characteristics (means ±{_format_multiplier(_figure_error_multiplier(results))} MC SE)",
        fontsize=12,
        y=0.985,
    )
    fig.text(
        0.5,
        0.905,
        "Pale orange, left of dashed divider: FWER-oriented procedures   •   Pale blue, right: FDR-oriented procedures",
        ha="center",
        va="center",
        fontsize=8.2,
        color=COLORS["slate"],
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.85), w_pad=1.4)
    return _save(fig, path, _figure_dpi(results))


def _power_curve(results: dict[str, Any], path: Path) -> str:
    """Plot power and FDR as aligned panels rather than dual y axes."""
    data = results["power_curve"]
    effect = [float(row["effect"]) for row in data]
    power = [float(row["power"]) for row in data]
    fdr = [float(row["fdr"]) for row in data]
    alpha = float(results["meta"]["alpha"])
    fig, (power_ax, fdr_ax, precision_ax) = plt.subplots(
        3,
        1,
        figsize=(7.4, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": (3.0, 3.0, 1.25)},
    )
    power_error, fdr_error = _power_curve_errors(
        data,
        _figure_error_multiplier(results),
    )
    power_lower = np.clip(np.asarray(power) - power_error, 0.0, 1.0)
    power_upper = np.clip(np.asarray(power) + power_error, 0.0, 1.0)
    power_ax.fill_between(effect, power_lower, power_upper, color=COLORS["teal"], alpha=0.20)
    power_ax.plot(
        effect,
        power,
        "o-",
        color=COLORS["teal"],
        markersize=5.5,
        markeredgecolor="white",
        markeredgewidth=0.55,
        label="BH average power",
    )
    power_ax.errorbar(
        effect,
        power,
        yerr=power_error,
        fmt="none",
        ecolor=COLORS["teal"],
        elinewidth=1.15,
        capsize=3.5,
        alpha=0.95,
        zorder=4,
    )
    power_ax.set_ylabel(
        "Replicate mean power\n(true-positive rate; proportion; linear)",
        color=COLORS["teal"],
        fontsize=8,
        labelpad=5,
    )
    power_ax.set_ylim(0, 1.05)
    power_ax.set_title("BH average power (replication mean true-positive rate)", loc="left", fontsize=10)
    power_ax.grid(color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    power_ax.legend(loc="upper left", fontsize=8)

    fdr_lower = np.clip(np.asarray(fdr) - fdr_error, 0.0, 1.0)
    fdr_upper = np.clip(np.asarray(fdr) + fdr_error, 0.0, 1.0)
    fdr_ax.fill_between(effect, fdr_lower, fdr_upper, color=COLORS["magenta"], alpha=0.20)
    fdr_ax.plot(
        effect,
        fdr,
        "s--",
        color=COLORS["magenta"],
        markersize=5.5,
        markeredgecolor="white",
        markeredgewidth=0.55,
        label="BH empirical FDR",
    )
    fdr_ax.errorbar(
        effect,
        fdr,
        yerr=fdr_error,
        fmt="none",
        ecolor=COLORS["magenta"],
        elinewidth=1.15,
        capsize=3.5,
        alpha=0.95,
        zorder=4,
    )
    fdr_ax.axhline(
        alpha,
        color=COLORS["magenta"],
        alpha=0.5,
        linestyle=":",
        label=f"nominal α={alpha:g}",
    )
    fdr_ax.set_ylabel(
        "Replicate mean FDP\n(empirical FDR; proportion; linear)",
        color=COLORS["magenta"],
        fontsize=8,
        labelpad=5,
    )
    fdr_ax.set_ylim(0, max(0.1, max(fdr) + max(fdr_error), alpha * 1.5))
    fdr_ax.set_title("BH empirical FDR (replication mean FDP)", loc="left", fontsize=10)
    fdr_ax.grid(color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    fdr_ax.legend(loc="upper left", fontsize=8)

    precision_ax.plot(
        effect,
        np.maximum(power_error, np.finfo(float).tiny),
        "o-",
        color=COLORS["teal"],
        markersize=3.5,
        label="Power ±3 MC SE",
    )
    precision_ax.plot(
        effect,
        np.maximum(fdr_error, np.finfo(float).tiny),
        "s--",
        color=COLORS["magenta"],
        markersize=3.5,
        label="FDR ±3 MC SE",
    )
    precision_ax.set_yscale("log")
    precision_ax.set_ylabel("Displayed ±3 MC SE\n(proportion; log)", fontsize=7)
    precision_ax.set_xlabel("Alternative effect size (null-standard-deviation units)")
    precision_ax.set_title("Finite-simulation precision half-widths", fontsize=8, loc="left", pad=2)
    precision_ax.grid(color=COLORS["grid"], alpha=0.55, linewidth=0.8, which="both")
    precision_ax.legend(fontsize=6.5, ncol=2, loc="upper center")
    precision_ax.tick_params(axis="both", labelsize=7)
    fig.suptitle(
        f"BH operating characteristics under the declared effect-size grid (±{_format_multiplier(_figure_error_multiplier(results))} MC SE)",
        fontsize=12,
        y=0.995,
    )
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.10, top=0.94, hspace=0.28)
    return _save(fig, path, _figure_dpi(results))


def _dependence_sweep(results: dict[str, Any], path: Path) -> str:
    data = results["dependence_sweep"]
    positions = np.arange(len(data))
    regime_labels = [
        _dependence_label(row, newline=True) for row in data
    ]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for key, label, color, style, offset in _DEPENDENCE_SERIES_STYLES["fdr"]:
        left.errorbar(
            positions + offset,
            [float(row[key]) for row in data],
            yerr=_three_se(
                [float(row.get(f"{key}_stderr", 0.0)) for row in data],
                _figure_error_multiplier(results),
            ),
            fmt=style,
            capsize=3,
            label=label,
            color=color,
            linewidth=1.65,
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
    for key, label, color, style, offset in _DEPENDENCE_SERIES_STYLES["power"]:
        right.errorbar(
            positions + offset,
            [float(row[key]) for row in data],
            yerr=_three_se(
                [float(row.get(f"{key}_stderr", 0.0)) for row in data],
                _figure_error_multiplier(results),
            ),
            fmt=style,
            capsize=3,
            label=label,
            color=color,
            linewidth=1.65,
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )
    alpha = float(results["meta"]["alpha"])
    left.axhline(
        alpha,
        color=COLORS["slate"],
        linestyle=":",
        linewidth=1,
        label=f"nominal α={alpha:g}",
    )
    left.set_title("Replication mean FDP (empirical FDR)")
    left.set_ylabel(
        "Replicate mean FDP\n(empirical FDR; proportion; linear)",
        fontsize=8,
        labelpad=4,
    )
    right.set_title("Replication mean true-positive rate (power)")
    right.set_ylabel(
        "Replicate mean power\n(true-positive rate; proportion; linear)",
        fontsize=8,
        labelpad=4,
    )
    left.set_xlabel("Declared dependence regime")
    right.set_xlabel("Declared dependence regime")
    left.set_xticks(positions, regime_labels, rotation=0, ha="center", fontsize=8)
    right.set_xticks(positions, regime_labels, rotation=0, ha="center", fontsize=8)
    fdr_upper = max(
        float(row[key])
        + float(_figure_error_multiplier(results)) * float(
            row.get(f"{key}_stderr", 0.0)
        )
        for key, _label, _color, _style, _offset in _DEPENDENCE_SERIES_STYLES[
            "fdr"
        ]
        for row in data
    )
    left.set_ylim(0, min(1.05, max(0.1, alpha * 1.2, fdr_upper * 1.15)))
    right.set_ylim(0, 1.05)
    for ax in (left, right):
        ax.grid(color=COLORS["grid"], alpha=0.55, linewidth=0.8)
        ax.legend(
            title="Procedure (marker + line)",
            fontsize=8,
            title_fontsize=7.5,
        )
    fig.suptitle(
        f"Dependence-regime comparison (mean {_format_multiplier(_figure_error_multiplier(results))} MC SE band)",
        fontsize=12,
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), w_pad=1.6)
    return _save(fig, path, _figure_dpi(results))


def _active_inference_policy(results: dict[str, Any], path: Path) -> str:
    regimes = results["active_inference"].get("regimes", results["active_inference"])
    names = ["strong_evidence", "weak_evidence"]
    labels = ["Strong evidence", "Weak evidence"]
    x = np.arange(len(names))
    width = 0.18
    alpha = float(results["meta"]["alpha"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric, title in zip(axes, ("fdr", "power"), ("FDR", "Average power")):
        posterior = [float(regimes[name][f"posterior_threshold_{metric}"]) for name in names]
        posterior_error = _three_se(
            [float(regimes[name].get(f"posterior_threshold_{metric}_stderr", 0.0)) for name in names],
            _figure_error_multiplier(results),
        )
        calibrated = [
            float(regimes[name][f"bh_on_evidence_{metric}"]) for name in names
        ]
        calibrated_error = _three_se(
            [
                float(regimes[name].get(f"bh_on_evidence_{metric}_stderr", 0.0))
                for name in names
            ],
            _figure_error_multiplier(results),
        )
        posterior_bars = ax.bar(
            x - width / 2,
            posterior,
            width,
            yerr=posterior_error,
            capsize=3,
            label="Posterior threshold",
            color=COLORS["magenta"],
            edgecolor="white",
            linewidth=0.6,
        )
        calibrated_bars = ax.bar(
            x + width / 2,
            calibrated,
            width,
            yerr=calibrated_error,
            capsize=3,
            label="BH on evidence p-values",
            color=COLORS["blue"],
            edgecolor="white",
            linewidth=0.6,
        )
        _annotate_bars(
            ax,
            posterior_bars,
            posterior,
            posterior_error,
            upper_limit=1.035 if metric == "power" else None,
        )
        _annotate_bars(
            ax,
            calibrated_bars,
            calibrated,
            calibrated_error,
            upper_limit=1.035 if metric == "power" else None,
        )
        ax.set_title(title)
        ax.set_ylabel(
            "Replicate mean FDP\n(empirical FDR; proportion; linear)"
            if metric == "fdr"
            else "Replicate mean power\n(true-positive rate; proportion; linear)",
            fontsize=8,
            labelpad=4,
        )
        ax.set_xticks(x, labels)
        ax.grid(axis="y", color=COLORS["grid"], alpha=0.55, linewidth=0.8)
        if metric == "fdr":
            observed = posterior + calibrated + list(posterior_error) + list(calibrated_error)
            ax.set_ylim(0, max(0.1, max(observed) * 1.15))
        else:
            ax.set_ylim(0, 1.05)
    axes[0].axhline(
        alpha,
        color=COLORS["slate"],
        linestyle=":",
        linewidth=1,
        label=f"nominal α={alpha:g}",
    )
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Active-inference decisions: calibration gap", fontsize=12, y=0.985)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89), w_pad=1.5)
    return _save(fig, path, _figure_dpi(results))


def _certificate_summary(results: dict[str, Any], path: Path) -> str:
    certs = results["certificates"]
    labels = list(certs)
    estimates = [float(certs[name]["estimate"]) for name in labels]
    bounds = [float(certs[name]["bound"]) for name in labels]
    errors = _three_se(
        [float(certs[name]["stderr"]) for name in labels],
        _figure_error_multiplier(results),
    )
    passed = [bool(certs[name]["passes"]) for name in labels]
    directions = [str(certs[name].get("bound_direction", "upper")) for name in labels]
    rate_indices = [index for index, name in enumerate(labels) if name != "storey_power_gain"]
    power_indices = [index for index, name in enumerate(labels) if name == "storey_power_gain"]
    groups = tuple(
        group
        for group in (("Control and calibration", rate_indices), ("Adaptive power", power_indices))
        if group[1]
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    for ax, (title, indices) in zip(axes, groups):
        positions = np.arange(len(indices))
        ax.errorbar(
            [estimates[i] for i in indices],
            positions,
            xerr=[errors[i] for i in indices],
            fmt="o",
            capsize=4,
            color=COLORS["blue"],
            label=f"Estimate ± {_format_multiplier(_figure_error_multiplier(results))} MC SE",
        )
        for position, i in enumerate(indices):
            direction = directions[i]
            ax.scatter(
                bounds[i],
                position,
                marker="D",
                s=55,
                color=COLORS["magenta"] if direction == "upper" else COLORS["teal"],
                label="Comparison bound" if position == 0 else None,
            )
            ax.text(
                1.01,
                position,
                "PASS" if passed[i] else "FAIL",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=COLORS["pass"] if passed[i] else COLORS["fail"],
                clip_on=False,
            )
        name_labels = {
            "bh_fdr_control": "BH FDR",
            "bonferroni_fwer": "Bonferroni FWER",
            "positive_dependence": "BH PRDS",
            "storey_power_gain": "Storey power",
            "storey_split_confirmation": "Split Storey",
        }
        display = [
            f"{name_labels.get(labels[i], labels[i].replace('_', ' ').title())} ({'≤' if directions[i] == 'upper' else '≥'})"
            for i in indices
        ]
        ax.set_yticks(positions, display)
        min_value = min(
            min(estimates[i] - errors[i] for i in indices),
            min(bounds[i] for i in indices),
        )
        max_value = max(
            max(estimates[i] + errors[i] for i in indices),
            max(bounds[i] for i in indices),
        )
        margin = max(0.005, (max_value - min_value) * 0.18)
        ax.set_xlim(max(0.0, min_value - margin), min(1.0, max_value + margin))
        ax.set_xlabel(
            "Replication-level error-rate estimate\n(proportion; linear scale)"
            if title == "Control and calibration"
            else "Replication-level mean power estimate\n(proportion; linear scale)"
        )
        ax.set_title(title)
        ax.grid(axis="x", color=COLORS["grid"], alpha=0.55, linewidth=0.8)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Seeded certificates: estimates versus bounds", fontsize=12, y=0.985)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.91), w_pad=1.2)
    return _save(fig, path, _figure_dpi(results))


def _adaptive_fdr_audit(results: dict[str, Any], path: Path) -> str:
    """Plot the adaptive certificate as separate FDR and power estimands.

    The broad certificate forest plot is intentionally retained for release
    overview. This companion makes the gate decomposition explicit: FDR
    estimates are compared with their declared bounds and finite ceilings,
    while power and the paired gain are displayed on their own scales. The
    distinction prevents a visually attractive power improvement from being
    mistaken for evidence of FDR control.
    """
    certificate = results["certificates"]["storey_power_gain"]
    multiplier = _figure_error_multiplier(results)
    alpha = float(certificate["alpha"])
    pi0 = float(certificate["pi0"])
    fdr_labels = ("BH", "Storey")
    fdr_estimates = np.asarray(
        [float(certificate["bh_fdr"]), float(certificate["storey_fdr"])],
        dtype=float,
    )
    fdr_errors = _three_se(
        [
            float(certificate["bh_fdr_stderr"]),
            float(certificate["storey_fdr_stderr"]),
        ],
        multiplier,
    )
    fdr_bounds = np.asarray([pi0 * alpha, alpha], dtype=float)
    fdr_ceilings = np.asarray(
        [
            float(certificate["bh_fdr_upper_band"]),
            float(certificate["storey_fdr_upper_band"]),
        ],
        dtype=float,
    )
    fdr_passed = [
        bool(certificate["bh_fdr_passes"]),
        bool(certificate["storey_fdr_passes"]),
    ]

    figure = plt.figure(figsize=(10.5, 5.3))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.35, 1.0), height_ratios=(2.0, 1.0))
    fdr_axis = figure.add_subplot(grid[:, 0])
    power_axis = figure.add_subplot(grid[0, 1])
    gain_axis = figure.add_subplot(grid[1, 1])

    positions = np.arange(len(fdr_labels))
    fdr_axis.errorbar(
        fdr_estimates,
        positions,
        xerr=fdr_errors,
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["blue"],
        capsize=4,
        linewidth=1.4,
        label=f"Estimate ± {_format_multiplier(multiplier)} MC SE",
        zorder=3,
    )
    for position, estimate, bound, ceiling, passed in zip(
        positions, fdr_estimates, fdr_bounds, fdr_ceilings, fdr_passed
    ):
        fdr_axis.scatter(
            bound,
            float(position),
            marker="D",
            s=58,
            color=COLORS["magenta"],
            label="Declared bound" if position == 0 else None,
            zorder=4,
        )
        fdr_axis.scatter(
            ceiling,
            float(position),
            marker="|",
            s=180,
            linewidths=2.2,
            color=COLORS["orange"],
            label="Finite gate ceiling" if position == 0 else None,
            zorder=4,
        )
        fdr_axis.text(
            1.01,
            float(position),
            "PASS" if passed else "FAIL",
            transform=fdr_axis.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=COLORS["pass"] if passed else COLORS["fail"],
            clip_on=False,
        )
        fdr_axis.text(
            0.02,
            float(position),
            f"estimate={float(estimate):.4f}",
            transform=fdr_axis.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8,
            color=COLORS["slate"],
            zorder=5,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
        )
    fdr_axis.set_yticks(positions, fdr_labels)
    fdr_axis.set_xlabel("Empirical FDR (replication mean FDP; proportion; linear scale)")
    fdr_axis.set_title("FDR calibration component")
    fdr_axis.set_xlim(0.0, max(float(np.max(fdr_estimates + fdr_errors)), float(np.max(fdr_ceilings))) * 1.22)
    fdr_axis.grid(axis="x", color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    fdr_axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.02, 0.97))
    fdr_axis.text(
        0.02,
        0.55,
        f"Declared limits: BH π₀α={pi0 * alpha:.4f}; Storey α={alpha:g}",
        transform=fdr_axis.transAxes,
        fontsize=8,
        color=COLORS["slate"],
    )

    power_estimates = np.asarray(
        [float(certificate["bh_power"]), float(certificate["storey_power"])],
        dtype=float,
    )
    power_errors = _three_se(
        [
            float(certificate["bh_power_stderr"]),
            float(certificate["stderr"]),
        ],
        multiplier,
    )
    power_axis.errorbar(
        power_estimates,
        positions,
        xerr=power_errors,
        fmt="o",
        color=COLORS["teal"],
        ecolor=COLORS["teal"],
        capsize=4,
        linewidth=1.4,
    )
    for position, estimate in zip(positions, power_estimates):
        power_axis.text(
            float(estimate) + float(power_errors[position]) + 0.004,
            float(position),
            f"{float(estimate):.3f}",
            va="center",
            fontsize=8,
            color=COLORS["slate"],
        )
    power_axis.set_yticks(positions, ("BH power", "Storey power"))
    power_axis.set_xlabel("Average power (replication mean TPR; proportion; linear scale)")
    power_axis.set_title("Power comparison")
    power_axis.set_xlim(0.0, min(1.0, float(np.max(power_estimates + power_errors)) * 1.12))
    power_axis.grid(axis="x", color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    power_axis.text(
        0.02,
        0.52,
        "Power is an operating-characteristic comparison, not an FDR certificate.",
        transform=power_axis.transAxes,
        fontsize=7.5,
        color=COLORS["slate"],
    )

    gain = float(certificate["power_gain"])
    gain_error = float(multiplier) * float(certificate["power_gain_stderr"])
    gain_axis.errorbar(
        [gain],
        [0],
        xerr=[gain_error],
        fmt="o",
        color=COLORS["purple"],
        ecolor=COLORS["purple"],
        capsize=4,
        linewidth=1.4,
    )
    gain_axis.axvline(0.0, color=COLORS["slate"], linestyle=":", linewidth=1)
    gain_axis.text(
        gain + gain_error + 0.004,
        0,
        f"{gain:+.3f}",
        va="center",
        fontsize=8,
        color=COLORS["slate"],
    )
    gain_axis.set_yticks([0], ["Storey − BH"])
    gain_axis.set_xlabel("Paired replication-mean power difference (proportion; linear scale)")
    gain_axis.set_title(
        f"Gain ≥ 0: {'PASS' if bool(certificate['power_gain_passes']) else 'FAIL'}",
        color=COLORS["pass"] if bool(certificate["power_gain_passes"]) else COLORS["fail"],
    )
    gain_axis.set_xlim(min(0.0, gain - gain_error) - 0.01, max(0.1, gain + gain_error) * 1.18)
    gain_axis.grid(axis="x", color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    figure.suptitle(
        f"Adaptive Storey audit ({int(certificate['n_reps']):,} replications)",
        fontsize=12,
    )
    figure.tight_layout()
    return _save(figure, path, _figure_dpi(results))


def _pvalue_calibration(results: dict[str, Any], path: Path) -> str:
    """Plot the global-null ECDF with its finite-sample DKW band."""
    data = results["null_pvalue_calibration"]
    grid = np.asarray(data["grid"], dtype=float)
    empirical = np.asarray(data["empirical_cdf"], dtype=float)
    radius = float(data["dkw_radius"])
    pooled_pvalues = _format_count(data.get("n_pvalues"))
    calibration_reps = _format_count(data.get("n_reps"))
    uniform = grid
    lower = np.clip(uniform - radius, 0.0, 1.0)
    upper = np.clip(uniform + radius, 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    confidence = 100.0 * float(results["meta"].get("mc_confidence", 0.95))
    ax.fill_between(
        grid,
        lower,
        upper,
        color=COLORS["blue"],
        alpha=0.45,
        label=f"{confidence:g}% DKW band",
    )
    ax.plot(grid, uniform, color=COLORS["slate"], linestyle="--", label="Uniform(0, 1)")
    ax.step(
        grid,
        empirical,
        where="post",
        color=COLORS["blue"],
        linewidth=2,
        label="Empirical CDF",
    )
    ax.set_xlabel("p-value threshold t (dimensionless)")
    ax.set_ylabel("Empirical CDF Fₙ(t) (pooled-p-value fraction; dimensionless)")
    ax.set_title(
        f"Global-null p-value calibration (D={float(data['ks_statistic']):.4f})"
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    ax.text(
        0.98,
        0.04,
        f"n={pooled_pvalues} pooled p-values\n"
        f"from {calibration_reps} replications\n"
        f"DKW radius={radius:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=COLORS["slate"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": COLORS["grid"]},
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return _save(fig, path, _figure_dpi(results))


def _sample_size_precision(results: dict[str, Any], path: Path) -> str:
    """Plot observed precision and configured replication accounting."""
    method_data = results["method_comparison"]
    labels = _ordered_methods(method_data)
    half_widths = [float(method_data[name]["mc_half_width_fdr"]) for name in labels]
    target = float(results["sample_size_accounting"]["target_half_width"])
    accounting = results["sample_size_accounting"]
    blocks = accounting["blocks"]
    rep_keys = (
        ("method", "method_comparison", "replications_per_method"),
        ("power", "power_curve", "replications_per_effect"),
        ("dependence", "dependence_sweep", "replications_per_rho_and_method"),
        ("active inference", "active_inference", "replications_per_regime"),
        ("null calibration", "null_pvalue_calibration", "replications"),
        ("certificates", "certificates", "replications_per_check"),
    )
    block_labels = [label for label, _name, _key in rep_keys]
    rep_values = [float(blocks[name][key]) for _label, name, key in rep_keys]
    required = float(accounting["worst_case_required_reps"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    display = [_method_label(name) for name in labels]
    precision_bars = axes[0].bar(
        np.arange(len(labels)),
        half_widths,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.6,
    )
    axes[0].bar_label(
        precision_bars,
        labels=[f"{value:.4f}" for value in half_widths],
        padding=3,
        fontsize=7,
    )
    axes[0].axhline(target, color=COLORS["orange"], linestyle="--", label=f"target={target:g}")
    axes[0].set_xticks(np.arange(len(labels)), display, rotation=55, ha="right", fontsize=8)
    confidence = 100.0 * float(results["meta"].get("mc_confidence", 0.95))
    axes[0].set_ylabel(
        f"{confidence:g}% t-based FDR half-width (proportion; log scale)"
    )
    axes[0].set_yscale("log")
    axes[0].set_ylim(max(min(half_widths) * 0.5, 1e-5), max(max(half_widths), target) * 2.0)
    axes[0].set_title("Observed Monte-Carlo precision")
    axes[0].grid(axis="y", color=COLORS["grid"], alpha=0.55, linewidth=0.8, which="both")
    axes[0].legend(fontsize=8)

    positions = np.arange(len(block_labels))
    budget_bars = axes[1].barh(
        positions,
        rep_values,
        color=COLORS["teal"],
        edgecolor="white",
        linewidth=0.6,
    )
    axes[1].bar_label(
        budget_bars,
        labels=[f"{int(value):,}" for value in rep_values],
        padding=3,
        fontsize=7,
    )
    axes[1].axvline(required, color=COLORS["orange"], linestyle="--", label=f"Hoeffding floor={required:,.0f}")
    axes[1].set_yticks(positions, block_labels)
    axes[1].set_xlabel("Independent replications per unit (count)")
    axes[1].set_xlim(0.0, max(rep_values) * 1.14)
    axes[1].set_title("Configured replication budget")
    axes[1].grid(axis="x", color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Monte-Carlo precision and sample-size accounting", fontsize=12)
    fig.tight_layout()
    return _save(fig, path, _figure_dpi(results))


def _complexity_order_display(value: object) -> str:
    """Typeset declared complexity orders without changing the raw contract.

    The results artifact deliberately stores portable ASCII order strings so
    source-work validation can compare exact values.  The publication figure
    is a separate reader-facing surface: displaying those strings verbatim
    makes mathematical orders look like identifiers (for example,
    ``Theta(m log2 m)``).  Keep this mapping explicit and fail closed if the
    declared plotting contract adds an order that has no visual equivalent.
    """
    displays = {
        "Theta(m)": r"$\Theta(m)$",
        "Theta(m log2 m)": r"$\Theta(m\log_2 m)$",
        "Theta(T)": r"$\Theta(T)$",
        "O(T^2) implementation upper proxy": r"$O(T^2)$ implementation upper proxy",
        "O(T^3) implementation upper proxy": r"$O(T^3)$ implementation upper proxy",
    }
    raw = str(value)
    if raw not in displays:
        raise ValueError(
            "computational-complexity figure has no typeset display form for "
            f"declared order {raw!r}"
        )
    return displays[raw]


def _computational_complexity(results: dict[str, Any], path: Path) -> str:
    """Plot deterministic, source-level work accounting without timing claims."""
    complexity = results["computational_complexity"]
    plot_data = complexity["plot_data"]
    action = complexity["release_workload"]["action_loop"]
    fixed = complexity["release_workload"]["fixed_family"]

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.7))
    hypothesis_axis, stream_axis = axes[0]
    action_axis, ledger_axis = axes[1]

    fixed_styles = (
        (COLORS["blue"], "o", "-"),
        (COLORS["orange"], "s", "--"),
    )
    for curve, (color, marker, style) in zip(
        plot_data["hypothesis_work_curves"], fixed_styles, strict=True
    ):
        hypothesis_axis.plot(
            curve["input_values"],
            curve["work_proxy_values"],
            color=color,
            marker=marker,
            linestyle=style,
            linewidth=1.9,
            markersize=5,
            label=f"{curve['name']} ({_complexity_order_display(curve['time_order'])})",
        )
    hypothesis_axis.set_xscale("log", base=2)
    hypothesis_axis.set_yscale("log", base=2)
    hypothesis_axis.set_xlabel("Declared family size m (hypotheses; log₂ scale)")
    hypothesis_axis.set_ylabel(
        "Source-work proxy $Q_{\\mathrm{comp}}$\n(dimensionless count; log₂ scale)",
        fontsize=9.3,
        labelpad=5,
    )
    hypothesis_axis.set_title("Fixed-family correction paths")
    hypothesis_axis.grid(color=COLORS["grid"], alpha=0.55, linewidth=0.8, which="both")
    hypothesis_axis.legend(
        fontsize=7.4,
        title="line + marker = code path; no timing or MC bars",
        title_fontsize=7.0,
        loc="upper left",
    )

    stream_styles = (
        (COLORS["teal"], "o", "-"),
        (COLORS["purple"], "s", "--"),
        (COLORS["orange"], "^", ":"),
        (COLORS["slate"], "D", "-."),
    )
    for curve, (color, marker, style) in zip(
        plot_data["stream_work_curves"], stream_styles, strict=True
    ):
        stream_axis.plot(
            curve["input_values"],
            curve["work_proxy_values"],
            color=color,
            marker=marker,
            linestyle=style,
            linewidth=1.75,
            markersize=4.8,
            label=curve["name"],
        )
    stream_axis.set_xscale("log", base=2)
    stream_axis.set_yscale("log", base=2)
    stream_axis.set_xlabel("Declared stream horizon T (observations/tests; log₂ scale)")
    stream_axis.set_ylabel(
        "History/path work proxy $Q_{\\mathrm{comp}}$\n(dimensionless count; log₂ scale)",
        fontsize=9.3,
        labelpad=5,
    )
    stream_axis.set_title("Sequential and online-accounting paths")
    stream_axis.grid(color=COLORS["grid"], alpha=0.55, linewidth=0.8, which="both")
    stream_axis.legend(
        fontsize=6.9,
        title="legend identifies path; combined line = coincident pair",
        title_fontsize=7.0,
        loc="upper left",
    )

    categories = plot_data["action_trace_categories"]
    positions = np.arange(len(categories))
    trace_calls = [int(item["trace_calls"]) for item in categories]
    labels = [
        f"{item['label']}\n(max H={int(item['max_steps_per_trace'])})"
        for item in categories
    ]
    bars = action_axis.barh(
        positions,
        trace_calls,
        color=[_POLICY_COLORS[index % len(_POLICY_COLORS)] for index in positions],
        edgecolor="white",
        linewidth=0.7,
    )
    action_axis.bar_label(
        bars,
        labels=[f"{value:,}" for value in trace_calls],
        padding=3,
        fontsize=7.5,
    )
    action_axis.set_yticks(positions, labels, fontsize=7.5)
    action_axis.invert_yaxis()
    action_axis.set_xlabel("Declared action-loop trace calls (count; linear scale)")
    action_axis.set_title("Release action-loop execution ledger")
    action_axis.grid(axis="x", color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    action_axis.set_xlim(0, max(trace_calls) * 1.22)
    action_axis.text(
        0.99,
        0.02,
        f"Total: {int(action['action_loop_trace_calls']):,} traces\n"
        f"Upper bound: {int(action['action_loop_step_upper_bound']):,} slots\n"
        "bars = calls; parenthetical H = permitted steps",
        transform=action_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.45,
        color=COLORS["slate"],
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.92,
            "edgecolor": COLORS["grid"],
        },
    )

    ledger_axis.axis("off")
    ledger_axis.set_title("What the release accounting does—and does not—measure")
    ledger_lines = (
        (
            "Fixed-family generation",
            f"K×B×m: {int(fixed['method_comparison_generated_hypothesis_positions']):,} "
            "generated hypothesis positions",
        ),
        (
            "Per-procedure retention",
            f"p-values: {int(fixed['per_procedure_retained_pvalue_elements']):,}; "
            f"truth labels: {int(fixed['per_procedure_retained_truth_elements']):,} elements",
        ),
        (
            "Action-loop workload",
            f"{int(action['action_loop_trace_calls']):,} trace calls; "
            f"{int(action['action_loop_step_upper_bound']):,} maximum permitted slots",
        ),
        (
            "Boundary: source-work proxy",
            "$Q_{\\mathrm{comp}}$ counts declared source work, not:\n"
            "local wall time $\\tau_{\\mathrm{wall}}$, simulated cost $C$,\n"
            "e-process evidence $E_t$, or online alpha-wealth $W_t^\\alpha$.",
        ),
    )
    for index, (heading, text) in enumerate(ledger_lines):
        y = 0.89 - index * 0.225
        ledger_axis.text(
            0.03,
            y,
            heading,
            transform=ledger_axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            fontweight="bold",
            color=COLORS["slate"],
        )
        ledger_axis.text(
            0.03,
            y - 0.07,
            text if "\n" in text else textwrap.fill(text, width=47),
            transform=ledger_axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.25,
            color=COLORS["slate"],
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "#f8fafc",
                "edgecolor": COLORS["grid"],
                "linewidth": 0.7,
            },
        )

    figure.suptitle(
        "Computational-work accounting for the declared implementation (no runtime benchmark)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.035, 0.0, 1.0, 0.94), h_pad=2.3, w_pad=2.2)
    return _save(figure, path, _figure_dpi(results))


def _paired_method_contrasts(results: dict[str, Any], path: Path) -> str:
    data = results["paired_comparisons"]["comparisons"]
    labels = _ordered_methods(data)
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, metric, title, color in (
        (axes[0], "fdr", "FDR difference", COLORS["magenta"]),
        (axes[1], "power", "Power difference", COLORS["teal"]),
    ):
        means = np.asarray([float(data[name][metric]["mean_difference"]) for name in labels])
        errors = np.asarray([float(data[name][metric]["half_width"]) for name in labels])
        ax.axhline(0.0, color=COLORS["slate"], linewidth=1)
        ax.errorbar(positions, means, yerr=errors, fmt="o", capsize=4, color=color)
        ax.set_title(title)
        metric_label = "FDR" if metric == "fdr" else "power"
        ax.set_ylabel(
            f"Paired replication mean\n{metric_label} − BH (proportion)"
        )
        ax.grid(axis="y", color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    display = [_method_label(name) for name in labels]
    axes[0].set_xticks(positions, display, rotation=55, ha="right", fontsize=8)
    axes[1].set_xticks(positions, display, rotation=55, ha="right", fontsize=8)
    fig.suptitle("Paired common-random-number contrasts (simultaneous bands)", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return _save(fig, path, _figure_dpi(results))


def _sequential_evidence(results: dict[str, Any], path: Path) -> str:
    data = results["sequential"]["representative"]
    pvalues = np.asarray(data["fixed_horizon_pvalues"], dtype=float)
    e_process_values = np.asarray(data["e_process_values"], dtype=float)
    lower = np.asarray(data["confidence_sequence_lower"], dtype=float)
    upper = np.asarray(data["confidence_sequence_upper"], dtype=float)
    empty = np.asarray(data.get("confidence_sequence_empty", np.zeros(lower.size, dtype=bool)), dtype=bool)
    steps = np.arange(1, pvalues.size + 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharex=False)
    e_process_ax, pvalue_ax, sequence_ax = axes
    e_process_ax.plot(
        np.arange(e_process_values.size),
        e_process_values,
        color=COLORS["teal"],
        marker="o",
        markersize=4,
        label="likelihood-ratio e-process evidence Eₜ",
    )
    e_process_ax.axhline(
        float(data["e_boundary"]),
        color=COLORS["teal"],
        linestyle="--",
        label=f"Ville boundary 1/α={float(data['e_boundary']):g}",
    )
    e_process_ax.set_xlabel("Sequential observation t (count)")
    e_process_ax.set_ylabel(
        "Likelihood-ratio e-process evidence Eₜ\n(dimensionless; log scale)",
        color=COLORS["teal"],
    )
    e_process_ax.set_yscale("log")
    e_process_ax.set_ylim(
        max(float(np.min(e_process_values)) * 0.5, 1e-8),
        max(float(data["e_boundary"]), float(np.max(e_process_values))) * 1.25,
    )
    e_process_ax.set_title("Time-uniform e-process evidence")
    e_process_ax.grid(
        color=COLORS["grid"], alpha=0.55, linewidth=0.8, which="both"
    )
    pvalue_ax.plot(
        steps,
        pvalues,
        color=COLORS["magenta"],
        marker="s",
        markersize=4,
        label="fixed-horizon p-value (peeking contrast)",
    )
    pvalue_ax.axhline(
        float(results["sequential"]["alpha"]),
        color=COLORS["magenta"],
        linestyle=":",
        label=f"nominal α={float(results['sequential']['alpha']):g}",
    )
    pvalue_ax.set_xlabel("Sequential observation t (count)")
    pvalue_ax.set_ylabel("Fixed-horizon p-value (dimensionless)", color=COLORS["magenta"])
    pvalue_ax.set_ylim(0.0, 1.0)
    pvalue_ax.set_title("INVALID for optional stopping")
    pvalue_ax.text(
        0.5,
        0.93,
        "Reference path only; do not stop when it crosses α.",
        transform=pvalue_ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color=COLORS["fail"],
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": COLORS["fail"], "alpha": 0.9},
    )
    pvalue_ax.grid(color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    visible = ~empty
    if np.any(visible):
        sequence_ax.fill_between(
            steps[visible],
            lower[visible],
            upper[visible],
            color=COLORS["blue"],
            alpha=0.28,
            label="confidence sequence",
        )
        sequence_ax.plot(steps[visible], lower[visible], color=COLORS["blue"], linewidth=1.2)
        sequence_ax.plot(steps[visible], upper[visible], color=COLORS["blue"], linewidth=1.2)
    sequence_ax.axhline(
        float(results["sequential"]["null_rate"]),
        color=COLORS["slate"],
        linestyle="--",
        label=f"null rate={float(results['sequential']['null_rate']):g}",
    )
    sequence_ax.set_xlabel("Sequential observation t (count)")
    sequence_ax.set_ylabel("Bernoulli success-rate parameter (probability)")
    sequence_ax.set_ylim(0.0, 1.0)
    sequence_ax.set_title("Confidence sequence")
    sequence_ax.grid(color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    lines, labels = e_process_ax.get_legend_handles_labels()
    lines2, labels2 = pvalue_ax.get_legend_handles_labels()
    lines3, labels3 = sequence_ax.get_legend_handles_labels()
    fig.legend(lines + lines2 + lines3, labels + labels2 + labels3, fontsize=8, loc="lower center", ncol=3)
    fig.suptitle("Sequential evidence: time-uniform paths and a fixed-horizon contrast", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.94))
    return _save(fig, path, _figure_dpi(results))


def _conditional_selection_calibration(results: dict[str, Any], path: Path) -> str:
    """Plot selected-target calibration excess against DKW radii."""
    calibration = results.get("suite", {}).get("conditional_selection_calibration", {})
    target_rows = calibration.get("selected_pvalues_by_target", [])
    labels = [f"target {index}" for index in range(len(target_rows))]
    rows = list(target_rows)
    pooled = calibration.get("pooled")
    if isinstance(pooled, dict):
        labels.append("chronological\npooled")
        rows.append(pooled)
    failure = results.get("suite", {}).get("future_evidence_selection_failure", {})
    failure_pooled = failure.get("pooled")
    if isinstance(failure_pooled, dict):
        labels.append("future-evidence\nminimum p")
        rows.append(failure_pooled)
    # The multi-line category labels carry the sample size and diagnostic
    # status.  Reserve a dedicated footer band so those labels cannot collide
    # with the validity note when the release sample grows.
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    if not rows:
        ax.text(
            0.5,
            0.5,
            "No selected-target calibration diagnostic in this result artifact",
            ha="center",
            va="center",
            color=COLORS["slate"],
        )
        ax.axis("off")
        return _save(fig, path, _figure_dpi(results))
    positions = np.arange(len(rows))
    selected_counts = np.asarray([int(row.get("n_selected", 0)) for row in rows])
    has_selected_values = selected_counts > 0
    excess = np.asarray([float(row.get("max_ecdf_excess") or 0.0) for row in rows])
    radii = np.asarray([float(row.get("dkw_radius") or 0.0) for row in rows])
    width = 0.36
    colors = [COLORS["orange"]] * (len(rows) - (1 if isinstance(failure_pooled, dict) else 0))
    if isinstance(failure_pooled, dict):
        colors.append(COLORS["fail"])
    ax.bar(
        positions - width / 2,
        np.where(has_selected_values, excess, np.nan),
        width,
        color=colors,
        label="max ECDF excess",
    )
    ax.bar(
        positions + width / 2,
        np.where(has_selected_values, radii, np.nan),
        width,
        color=COLORS["blue"],
        label="nominal iid DKW radius",
    )
    for no_selection_index in np.flatnonzero(~has_selected_values):
        position = float(no_selection_index)
        ax.axvspan(
            position - 0.46,
            position + 0.46,
            facecolor="#e2e8f0",
            edgecolor=COLORS["slate"],
            hatch="///",
            alpha=0.72,
            label="_nolegend_",
            zorder=0,
        )
        ax.text(
            position,
            0.52,
            "n=0\nno selected p-values\nno calibration estimate",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=7.2,
            color=COLORS["slate"],
            fontweight="bold",
        )
    labels = [
        f"{label}\n"
        f"n={int(row.get('n_selected', 0)):,}\n"
        f"{'within nominal iid DKW radius' if row.get('passes_dkw_diagnostic') else 'exceeds nominal iid DKW radius' if int(row.get('n_selected', 0)) else 'no calibration estimate'}"
        for label, row in zip(labels, rows)
    ]
    ax.set_xticks(positions, labels, fontsize=7.2)
    ax.set_ylabel("Maximum empirical CDF excess max(Fₙ(p) − p) (proportion)")
    ax.set_title(
        "Action-step p-value calibration diagnostic and invalid future-evidence contrast"
    )
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if np.any(~has_selected_values):
        legend_handles.append(
            Patch(
                facecolor="#e2e8f0",
                edgecolor=COLORS["slate"],
                hatch="///",
                label="n=0: no selected p-values (no estimate)",
            )
        )
        legend_labels.append("n=0: no selected p-values (no estimate)")
    ax.legend(legend_handles, legend_labels, frameon=False, ncol=1, fontsize=7.5)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    footer = (
        "Chronological action-step values = process/policy diagnostic; "
        "future-evidence minimum-p = invalid contrast. "
        "A gray hatched n=0 cell means no selected p-values, not zero ECDF excess. "
        "Dedicated split-stream confirmation uses a separately declared static channel and is reported outside this diagnostic."
    )
    fig.text(
        0.5,
        0.018,
        textwrap.fill(footer, width=130),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=COLORS["slate"],
        style="italic",
    )
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 0.97))
    return _save(fig, path, _figure_dpi(results))


def _online_fdr_trace(results: dict[str, Any], path: Path) -> str:
    """Show online levels, events, and accounting without conflating them."""
    methods = results["online_fdr"]["methods"]
    method_style = _ONLINE_FDR_EVENT_ROWS
    event_positions = tuple(row[3] for row in method_style)
    event_labels = tuple(row[1] for row in method_style)
    if len(set(event_positions)) != len(event_positions) or set(event_positions) != set(
        range(len(event_positions))
    ):
        raise ValueError("online-FDR event rows must be a unique zero-based sequence")
    alpha = float(results["online_fdr"]["alpha"])
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 8.8), squeeze=False)
    levels_ax, events_ax, accounting_ax = axes[:, 0]
    for method, display, color, y_position in method_style:
        trace = methods[method]["trace"]
        marker, linestyle = _ONLINE_FDR_TRACE_STYLES[method]
        expected_accounting_kind = (
            "cumulative_nominal_level_allocation"
            if method == "e_lond"
            else "procedural_alpha_wealth"
        )
        accounting_kind = str(
            trace.get("accounting_kind", expected_accounting_kind)
        )
        if accounting_kind != expected_accounting_kind:
            raise ValueError(
                f"{method} trace has accounting_kind={accounting_kind!r}; "
                f"expected {expected_accounting_kind!r}"
            )
        levels = np.asarray(trace["levels"], dtype=float)
        accounting_path = np.asarray(trace["accounting_path"], dtype=float)
        candidates = np.asarray(trace.get("candidates", []), dtype=bool)
        rejections = np.asarray(trace["rejections"], dtype=bool)
        steps = np.arange(1, levels.size + 1)
        markevery = max(1, levels.size // 8)
        levels_ax.plot(
            steps,
            levels,
            color=color,
            linewidth=1.6,
            marker=marker,
            linestyle=linestyle,
            markevery=markevery,
            label=(
                "e-LOND nominal level αₜ; reject Eₜ ≥ 1/αₜ"
                if method == "e_lond"
                else f"{display} p-value level αₜ"
            ),
        )
        if method != "e_lond":
            events_ax.scatter(
                steps[candidates],
                np.full(int(np.count_nonzero(candidates)), y_position),
                marker="|",
                s=110,
                linewidths=1.5,
                color=color,
            )
        events_ax.scatter(
            steps[rejections],
            np.full(int(np.count_nonzero(rejections)), y_position),
            marker="*",
            s=55,
            color=color,
        )
        accounting_label = (
            "e-LOND cumulative nominal-level allocation"
            if method == "e_lond"
            else f"{display} procedural α-wealth"
        )
        accounting_ax.plot(
            np.arange(accounting_path.size),
            accounting_path,
            color=color,
            linewidth=1.6,
            marker=marker,
            linestyle=linestyle,
            markevery=max(1, accounting_path.size // 8),
            label=accounting_label,
        )
    levels_ax.set_title("Nominal test level αₜ (dimensionless; log scale)", pad=8, fontsize=10.5)
    events_ax.set_title(
        "Candidate / rejection events  (| = p-value candidate; * = rejection)",
        pad=8,
        fontsize=10.5,
    )
    accounting_ax.set_title(
        "Procedural α-wealth / cumulative nominal-level allocation", pad=8, fontsize=10.5
    )
    levels_ax.set_yscale("log")
    levels_ax.set_ylabel("Nominal level αₜ\n(dimensionless; log scale)")
    levels_ax.set_xlabel("Online test index t (count)")
    levels_ax.axhline(
        alpha,
        color=COLORS["slate"],
        linestyle=":",
        linewidth=1,
        label=f"nominal α={alpha:g}",
    )
    events_ax.set_yticks(event_positions, event_labels)
    events_ax.set_ylim(min(event_positions) - 0.5, max(event_positions) + 0.5)
    events_ax.set_xlabel("Online test index t (count)")
    events_ax.set_ylabel("Procedure row\n(top to bottom)")
    accounting_ax.set_yscale("symlog", linthresh=1e-4)
    accounting_ax.set_xlabel("Post-test accounting index t (count)")
    accounting_ax.set_ylabel(
        "Procedure-specific accounting\n(dimensionless; symlog)"
    )
    for ax in (levels_ax, events_ax, accounting_ax):
        ax.grid(color=COLORS["grid"], alpha=0.55, linewidth=0.8, which="both")
    levels_ax.legend(fontsize=8, frameon=True, loc="upper right", ncol=2)
    accounting_ax.legend(fontsize=8, frameon=True, loc="upper right", ncol=2)
    fig.suptitle(
        "Online FDR accounting diagnostics under a seeded null stream",
        fontsize=13,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Null-stream accounting diagnostic: LORD++/SAFFRON procedural α-wealth; e-LOND cumulative nominal-level allocation can recycle after rejection; neither is likelihood-ratio e-process evidence.",
        ha="center",
        va="bottom",
        fontsize=8.35,
        color=COLORS["slate"],
    )
    fig.tight_layout(rect=(0.035, 0.055, 0.99, 0.94), h_pad=1.45)
    return _save(fig, path, _figure_dpi(results))


def _action_loop_trace(results: dict[str, Any], path: Path) -> str:
    """Render the visible action-loop trajectory without exposing hidden states."""
    suite = results.get("suite", {})
    operations = suite.get("operating_characteristics", [])
    trace = operations[0].get("traces", [{}])[0] if operations else {}
    steps = trace.get("visible_steps", [])
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
    if not steps:
        for ax in axes.flat:
            ax.axis("off")
        axes[1, 0].text(0.5, 0.5, "No action-loop trace in this artifact", ha="center", va="center")
        fig.suptitle("Action-in-the-loop trace unavailable", fontsize=13)
        return _save(fig, path, _figure_dpi(results))
    trace_cell = _scenario_cell_identity(operations[0]) if operations else ("", "")
    cost_unit, cost_scale_id, _ = _cost_display_contract(suite, [trace_cell])
    times = np.asarray([step["time"] for step in steps], dtype=float)
    actions = [str(step["action"]) for step in steps]

    def action_components(action: str) -> tuple[int, str]:
        """Decode the compact visible target/sensor action naming convention."""
        target_token, separator, sensor = action.partition("_")
        target_suffix = target_token.removeprefix("target")
        if target_token.startswith("target") and target_suffix.isdigit():
            return int(target_suffix), sensor if separator and sensor else "unspecified"
        return 0, action.replace("_", " ") or "unspecified"

    action_targets, action_sensors = zip(
        *(action_components(action) for action in actions), strict=True
    )
    observations = ["missing" if step["observation"] is None else str(step["observation"]) for step in steps]
    axes[0, 0].plot(times, [step["posterior_signal"] for step in steps], "o-", color=COLORS["blue"], label="P(any signal | visible history)")
    axes[0, 0].set_title("Posterior signal belief")
    axes[0, 0].set_ylabel("Posterior probability\n[0, 1]", fontsize=8, labelpad=3)
    axes[0, 0].set_ylim(0, 1.05)
    target_beliefs = np.asarray([step["target_beliefs"] for step in steps], dtype=float)
    for target in range(target_beliefs.shape[1]):
        marker, linestyle = _TARGET_TRACE_STYLES[target % len(_TARGET_TRACE_STYLES)]
        axes[0, 1].plot(
            times,
            target_beliefs[:, target],
            marker=marker,
            linestyle=linestyle,
            color=_POLICY_COLORS[target % len(_POLICY_COLORS)],
            markevery=max(1, times.size // 8),
            label=f"P(target {target} signal | history)",
        )
    axes[0, 1].set_title("Target-specific posterior beliefs")
    axes[0, 1].set_ylabel("Posterior probability\n[0, 1]", fontsize=8, labelpad=3)
    axes[0, 1].set_ylim(0, 1.05)
    pvalues = np.maximum(np.asarray([step["p_value"] for step in steps], dtype=float), 1e-12)
    axes[1, 0].plot(times, -np.log10(pvalues), "o-", color=COLORS["magenta"], label="−log₁₀ p")
    alpha = float(results.get("suite", {}).get("scenario", {}).get("setting", {}).get("alpha", 0.05))
    axes[1, 0].axhline(-np.log10(alpha), color=COLORS["slate"], linestyle=":", linewidth=1, label="alpha")
    axes[1, 0].set_title("Evidence p-value (separate from posterior belief)")
    axes[1, 0].set_ylabel("−log₁₀(p)\n(dimensionless)", fontsize=8, labelpad=3)
    target_e_process_values = np.asarray(
        [
            step["target_e_process_values"]
            for step in steps
        ],
        dtype=float,
    )
    if target_e_process_values.ndim == 2 and target_e_process_values.shape[1]:
        for target in range(target_e_process_values.shape[1]):
            marker, linestyle = _TARGET_TRACE_STYLES[target % len(_TARGET_TRACE_STYLES)]
            axes[1, 1].plot(
                times,
                target_e_process_values[:, target],
                marker=marker,
                linestyle=linestyle,
                color=_POLICY_COLORS[target % len(_POLICY_COLORS)],
                linewidth=1.0,
                markevery=max(1, times.size // 8),
                label=f"target {target} e-process (per-target decision object)",
            )
    axes[1, 1].step(
        times,
        [step["e_process_value"] for step in steps],
        where="mid",
        color=COLORS["slate"],
        linestyle="-.",
        linewidth=1.25,
        alpha=0.85,
        label="all-action global-null LR product (diagnostic)",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].axhline(1.0 / alpha, color=COLORS["slate"], linestyle=":", linewidth=1, label="1/α: per-target decision rule")
    axes[1, 1].set_title("Target-specific e-process paths (per-target rule; log scale)")
    axes[1, 1].set_ylabel(
        "Target e-process / LR product\n(dimensionless; log scale)",
        fontsize=8,
        labelpad=3,
    )
    axes[2, 0].plot(times, [step["e_value"] for step in steps], "D-", color=COLORS["teal"], label="e-value factor eₜ")
    axes[2, 0].axhline(1.0, color=COLORS["slate"], linestyle=":", linewidth=1)
    axes[2, 0].set_yscale("log")
    axes[2, 0].set_title("Per-step e-value factor eₜ (log scale)")
    axes[2, 0].set_ylabel(
        "e-value factor eₜ\n(dimensionless; log scale)",
        fontsize=8,
        labelpad=3,
    )
    axes[2, 1].plot(times, [step["cost"] for step in steps], "s-", color=COLORS["orange"], label="step cost Cₜ")
    axes[2, 1].plot(times, np.cumsum([step["cost"] for step in steps]), "s--", color=COLORS["purple"], label="cumulative cost ΣCₜ")
    axes[2, 1].set_title("Declared simulation-cost accounting")
    axes[2, 1].set_ylabel(
        f"Simulated cost C\n({cost_unit}; {cost_scale_id})",
        fontsize=7.3,
        labelpad=3,
    )
    axes[2, 1].legend(fontsize=8, loc="best")
    action_targets_array = np.asarray(action_targets, dtype=float)
    sensor_styles = {
        "cheap": (COLORS["orange"], "o", "cheap sensor"),
        "precise": (COLORS["purple"], "D", "precise sensor"),
    }
    for sensor in dict.fromkeys(action_sensors):
        color, marker, display = sensor_styles.get(
            sensor,
            (COLORS["slate"], "s", sensor.replace("_", " ")),
        )
        mask = np.asarray([item == sensor for item in action_sensors], dtype=bool)
        axes[3, 0].scatter(
            times[mask],
            action_targets_array[mask],
            color=color,
            marker=marker,
            s=48,
            label=display,
        )
    target_levels = sorted(set(action_targets))
    axes[3, 0].set_title("Actions: target position; sensor marker/color")
    axes[3, 0].set_ylabel("Selected target", fontsize=8, labelpad=3)
    axes[3, 0].set_yticks(target_levels, [f"Target {target}" for target in target_levels])
    axes[3, 0].set_ylim(min(target_levels) - 0.5, max(target_levels) + 0.5)
    axes[3, 0].legend(fontsize=7, loc="best", title="Sensor choice")
    stop_mask = np.asarray([bool(step["stopped"]) for step in steps], dtype=bool)
    stop_colors = [COLORS["fail"] if stopped else COLORS["blue"] for stopped in stop_mask]
    axes[3, 1].scatter(
        times,
        np.zeros_like(times),
        color=[COLORS["magenta"] if value == "1" else COLORS["blue"] for value in observations],
        s=45,
        edgecolors=stop_colors,
        linewidths=1.5,
    )
    axes[3, 1].scatter(
        times[stop_mask],
        np.zeros(int(np.count_nonzero(stop_mask))),
        marker="x",
        color=COLORS["slate"],
        s=70,
        linewidths=1.8,
    )
    for x, label, stopped in zip(times, observations, stop_mask, strict=True):
        visible_label = f"{label}\nSTOP" if stopped else label
        axes[3, 1].text(x, 0.04, visible_label, ha="center", fontsize=8)
    axes[3, 1].set_title("Observations; X/red outline = stop")
    axes[3, 1].set_yticks([])
    for ax in axes.flat:
        ax.grid(color=COLORS["grid"], alpha=0.55, linewidth=0.8)
        ax.set_xlabel("Action-loop step t (count)")
        if ax not in (axes[3, 0], axes[3, 1]):
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        f"Action-in-the-loop trace: {_policy_label(trace.get('policy', 'policy'))} | visible history only | "
        f"stopped at t={trace.get('stopping_time', len(steps))}", fontsize=13,
    )
    footer = (
        "Target e-process paths—not the all-action product—carry a per-target optional-stopping role; no FWER/FDR claim. "
        "Fixed-horizon p-value contrast is invalid under optional stopping. "
        f"Cost C uses declared {cost_unit}; scale {cost_scale_id}."
    )
    fig.text(
        0.5,
        0.012,
        textwrap.fill(footer, width=155),
        ha="center",
        va="bottom",
        fontsize=7.45,
        color=COLORS["fail"],
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.93), w_pad=2.0, h_pad=1.1)
    return _save(fig, path, _figure_dpi(results))


def _conditional_power_surface(results: dict[str, Any], path: Path) -> str:
    """Render scenario-indexed power, error, cost, and stopping characteristics."""
    suite = results.get("suite", {})
    operations = suite.get("operating_characteristics", [])
    contracts = {
        (str(contract.get("scenario_id")), str(contract.get("scenario_hash", ""))): contract.get("metadata", {})
        for contract in suite.get("scenario_contracts", [])
        if isinstance(contract, dict)
    }
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    scenario_metadata: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in operations:
        scenario_id, scenario_hash = _scenario_cell_identity(item)
        metadata = item.get("scenario_metadata", {}) or contracts.get((scenario_id, scenario_hash), {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        summary = item.get("summary", {})
        policy = str(item.get("policy", "")).strip()
        if not policy:
            raise ValueError("action-loop figure rows require a policy")
        key = (scenario_id, scenario_hash, policy)
        if key in grouped:
            raise ValueError("duplicate action-loop scenario/policy row would overwrite a figure cell")
        grouped[key] = summary
        scenario_metadata[(scenario_id, scenario_hash)] = metadata
    def reliability_sort_key(cell: tuple[str, str]) -> tuple[float, str, str]:
        try:
            reliability = float(scenario_metadata[cell].get("reliability", 0.0))
        except (TypeError, ValueError):
            reliability = 0.0
        return reliability, cell[0], cell[1]

    scenario_cells = sorted(scenario_metadata, key=reliability_sort_key)
    policies = sorted({key[2] for key in grouped})
    cost_unit, cost_scale_id, shared_cost_scale = _cost_display_contract(
        suite, scenario_cells
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), sharex=True)
    if not grouped:
        for ax in axes.flat:
            ax.axis("off")
        axes[0, 0].text(0.5, 0.5, "No action-loop operating characteristics in this artifact", ha="center", va="center")
        return _save(fig, path, _figure_dpi(results))
    colors = _POLICY_COLORS
    markers = _POLICY_MARKERS
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    def interval_errors(
        means: list[float], intervals: list[Any]
    ) -> np.ndarray:
        """Return asymmetric display errors from result-bound intervals."""
        return np.asarray(
            [
                [
                    max(0.0, mean - float(interval[0])),
                    max(0.0, float(interval[1]) - mean),
                ]
                for mean, interval in zip(means, intervals)
            ],
            dtype=float,
        ).T

    x_positions = np.arange(len(scenario_cells), dtype=float)
    scenario_labels = [
        _scenario_cell_label(scenario_id, scenario_hash, scenario_metadata[(scenario_id, scenario_hash)])
        for scenario_id, scenario_hash in scenario_cells
    ]
    for policy_index, policy in enumerate(policies):
        color = colors[policy_index % len(colors)]
        marker = markers[policy_index % len(markers)]
        available = [
            (position, grouped[(scenario_id, scenario_hash, policy)])
            for position, (scenario_id, scenario_hash) in zip(x_positions, scenario_cells, strict=True)
            if (scenario_id, scenario_hash, policy) in grouped
        ]
        positions = [position for position, _summary in available]
        summaries = [summary for _position, summary in available]
        values = [summary["power"] for summary in summaries]
        power_intervals = [summary.get("power_interval", [value, value]) for summary, value in zip(summaries, values)]
        power_errors = interval_errors(values, power_intervals)
        fdr = [summary["fdr"] for summary in summaries]
        fdr_intervals = [summary.get("fdr_interval", [value, value]) for summary, value in zip(summaries, fdr)]
        fdr_errors = interval_errors(fdr, fdr_intervals)
        cost = [float(summary["mean_cost"]) for summary in summaries]
        cost_intervals = [
            summary.get("cost_interval", [value, value])
            for summary, value in zip(summaries, cost)
        ]
        cost_errors = interval_errors(cost, cost_intervals)
        stopping = [float(summary["mean_stopping_time"]) for summary in summaries]
        stopping_intervals = [
            summary.get("stopping_interval", [value, value])
            for summary, value in zip(summaries, stopping)
        ]
        stopping_errors = interval_errors(stopping, stopping_intervals)
        label = _policy_label(policy)
        power_handle = axes[0, 0].errorbar(
            positions,
            values,
            yerr=power_errors,
            fmt=marker,
            color=color,
            capsize=3,
            label=label,
        )
        axes[0, 1].errorbar(positions, fdr, yerr=fdr_errors, fmt=marker, color=color, capsize=3, label=label)
        axes[1, 0].errorbar(positions, cost, yerr=cost_errors, fmt=marker, color=color, capsize=3, label=label)
        axes[1, 1].errorbar(positions, stopping, yerr=stopping_errors, fmt=marker, color=color, capsize=3, label=label)
        legend_handles.append(power_handle[0])
        legend_labels.append(label)
    titles = (
        (
            "Power (replication mean true-positive rate; proportion)",
            "Observed mean FDP (finite-simulation estimand; proportion)",
        ),
        (
            (
                f"Total simulated cost C (replication mean; {cost_unit}; shared {cost_scale_id})"
                if shared_cost_scale
                else "Total simulated cost C (replication mean; scenario-specific units; no cross-cell comparison)"
            ),
            "Stopping time (replication mean; action steps)",
        ),
    )
    ylabels = (
        (
            "Power\n(replication mean TPR; proportion)",
            "Observed mean FDP\n(finite simulation; proportion)",
        ),
        (
            f"Total cost C\n(replication mean; {cost_unit})"
            if shared_cost_scale
            else "Total cost C\n(replication mean; within scenario)",
            "Stopping time\n(replication mean; action steps)",
        ),
    )
    for row in range(2):
        for column in range(2):
            axes[row, column].set_title(titles[row][column], fontsize=9.5)
            if ylabels[row][column]:
                axes[row, column].set_ylabel(ylabels[row][column], fontsize=8, labelpad=3)
            axes[row, column].grid(color=COLORS["grid"], alpha=0.55, linewidth=0.8)
            axes[row, column].set_xticks(x_positions, scenario_labels, fontsize=6.8)
            if row == 1:
                axes[row, column].set_xlabel("Declared scenario cell (ID / hash prefix / base reliability)", fontsize=7.2)
    axes[0, 0].set_ylim(0, 1.05)
    alpha = float(results.get("meta", {}).get("alpha", 0.05))
    axes[0, 1].axhline(
        alpha,
        color=COLORS["slate"],
        linestyle=":",
        linewidth=1,
        label=f"nominal α={alpha:g}",
    )
    axes[0, 1].text(
        0.98,
        0.06,
        f"dotted line: nominal α={alpha:g}",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=COLORS["slate"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    axes[0, 1].set_ylim(0, 1.05)
    fig.legend(
        legend_handles,
        legend_labels,
        fontsize=7,
        loc="lower center",
        ncol=3,
        title="Policy (color + marker)",
        frameon=True,
        bbox_to_anchor=(0.5, 0.065),
    )
    fig.suptitle(
        "Conditional action-loop operating characteristics (scenario-cell marks ± finite-simulation intervals)",
        fontsize=12,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Observed mean FDP is a finite-simulation estimand, not an FDR-control claim for correction-none or per-target e-process rows. "
            "Fixed-horizon policy: comparison only; invalid if used for optional stopping. "
            + (
                f"All displayed cells share declared cost scale {cost_scale_id}."
                if shared_cost_scale
                else "Cost comparison is within each declared scenario only; scales differ or are unrecorded."
            )
        ),
        ha="center",
        va="bottom",
        fontsize=7.1,
        color=COLORS["fail"],
    )
    fig.tight_layout(rect=(0.04, 0.17, 0.98, 0.90), w_pad=1.0, h_pad=2.4)
    return _save(fig, path, _figure_dpi(results))


def _action_policy_frontier(results: dict[str, Any], path: Path) -> str:
    """Render a faceted action policy power-cost frontier."""
    suite = results.get("suite", {})
    rows = suite.get("power_cost_frontier", [])
    if not rows:
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No action policy frontier in this artifact", ha="center", va="center")
        return _save(fig, path, _figure_dpi(results))
    colors = _POLICY_COLORS
    markers = _POLICY_MARKERS
    policies = list(dict.fromkeys(str(row.get("policy", "policy")) for row in rows))
    scenario_metadata: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        scenario_id, scenario_hash = _scenario_cell_identity(row)
        metadata = row.get("scenario_metadata", {})
        scenario_metadata[(scenario_id, scenario_hash)] = (
            metadata if isinstance(metadata, Mapping) else {}
        )

    def reliability_sort_key(cell: tuple[str, str]) -> tuple[float, str, str]:
        try:
            reliability = float(scenario_metadata[cell].get("reliability", 0.0))
        except (TypeError, ValueError):
            reliability = 0.0
        return reliability, cell[0], cell[1]

    scenario_cells = sorted(scenario_metadata, key=reliability_sort_key)
    cost_unit, cost_scale_id, shared_cost_scale = _cost_display_contract(
        suite, scenario_cells
    )
    n_columns = min(3, len(scenario_cells))
    n_rows = int(np.ceil(len(scenario_cells) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.7 * n_columns, 4.8 * n_rows),
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    all_costs = [float(row.get("mean_cost", 0.0)) for row in rows]
    cost_padding = max(0.5, 0.05 * (max(all_costs) - min(all_costs)))
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    seen_policies: set[str] = set()
    for panel_index, (scenario_id, scenario_hash) in enumerate(scenario_cells):
        ax = axes_flat[panel_index]
        panel_rows: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if _scenario_cell_identity(row) != (scenario_id, scenario_hash):
                continue
            policy = str(row.get("policy", "")).strip()
            if policy in panel_rows:
                raise ValueError("duplicate action-policy frontier row would overwrite a scenario cell")
            panel_rows[policy] = row
        for policy_index, policy in enumerate(policies):
            row = panel_rows.get(policy)
            if row is None:
                continue
            color = colors[policy_index % len(colors)]
            marker = markers[policy_index % len(markers)]
            power = float(row.get("power", 0.0))
            cost = float(row.get("mean_cost", 0.0))
            power_interval = row.get("power_interval", [power, power])
            cost_interval = row.get("cost_interval", [cost, cost])
            handle = ax.errorbar(
                cost,
                power,
                xerr=[[max(0.0, cost - float(cost_interval[0]))], [max(0.0, float(cost_interval[1]) - cost)]],
                yerr=[[max(0.0, power - float(power_interval[0]))], [max(0.0, float(power_interval[1]) - power)]],
                fmt=marker,
                color=color,
                ecolor=color,
                capsize=3,
                markersize=7,
                label=None,
            )
            if policy not in seen_policies:
                legend_handles.append(handle[0])
                legend_labels.append(_policy_label(policy))
                seen_policies.add(policy)
        ax.set_title(
            "Scenario cell\n"
            + _scenario_cell_label(
                scenario_id,
                scenario_hash,
                scenario_metadata[(scenario_id, scenario_hash)],
            ),
            fontsize=9,
        )
        panel_costs = [float(row.get("mean_cost", 0.0)) for row in panel_rows.values()]
        panel_padding = max(
            0.5,
            0.05 * (max(panel_costs) - min(panel_costs)),
        )
        if shared_cost_scale:
            ax.set_xlabel(
                f"Mean total simulated cost C\n({cost_unit}; shared {cost_scale_id})",
                fontsize=7.4,
            )
            ax.set_xlim(min(all_costs) - cost_padding, max(all_costs) + cost_padding)
        else:
            ax.set_xlabel(
                "Mean total simulated cost C\n(scenario-specific units; no cross-panel comparison)",
                fontsize=7.0,
            )
            ax.set_xlim(min(panel_costs) - panel_padding, max(panel_costs) + panel_padding)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    for ax in axes_flat[len(scenario_cells):]:
        ax.axis("off")
    axes_flat[0].set_ylabel("Mean true-positive rate (power; proportion; linear scale)")
    fig.legend(
        legend_handles,
        legend_labels,
        fontsize=8,
        loc="upper center",
        ncol=min(3, len(legend_labels)),
        title="Policy (color + marker)",
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.text(
        0.5,
        0.018,
        (
            "Horizontal intervals: finite-simulation mean simulated cost C; vertical intervals: finite-simulation mean power. "
            "Each point is conditional on its declared scenario and policy; fixed-horizon policy: comparison only; invalid if used for optional stopping. "
            + (
                f"All facets share declared cost scale {cost_scale_id}."
                if shared_cost_scale
                else "Cost is only comparable within a facet because scale labels differ or are unrecorded."
            )
        ),
        ha="center",
        va="bottom",
        fontsize=7.45,
        color=COLORS["slate"],
    )
    fig.suptitle("Action-policy power-cost frontier under declared scenario cells", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 0.84))
    return _save(fig, path, _figure_dpi(results))


def _scenario_decomposition(results: dict[str, Any], path: Path) -> str:
    """Render the investigator, agent, and evaluator roles without conflation."""
    suite = results.get("suite", {})
    scenario_count = len(suite.get("scenario_contracts", []))
    has_split_confirmation_contract = isinstance(
        suite.get("split_confirmation_scenario"), dict
    )
    action_count = len(results.get("action_loop", {}).get("operating_characteristics", []))
    fig, ax = plt.subplots(figsize=(13.6, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.04,
        0.975,
        "Conditional power is investigator-estimated and conditional on a design—not an agent or simulated-niche constant",
        fontsize=14.4,
        fontweight="bold",
        color=COLORS["slate"],
        va="top",
    )
    ax.text(
        0.04,
        0.935,
        "The investigator specifies the simulated niche/process P_N; the embedded agent sees partial history and the evaluator scores hidden truth after each replicate.",
        fontsize=9.5,
        color=COLORS["slate"],
        va="top",
    )

    investigator_x, investigator_y, investigator_w, investigator_h = 0.04, 0.82, 0.92, 0.07
    ax.add_patch(
        FancyBboxPatch(
            (investigator_x, investigator_y),
            investigator_w,
            investigator_h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#eef6ff",
            edgecolor=COLORS["blue"],
            linewidth=1.8,
        )
    )
    ax.text(
        0.50,
        0.855,
        "Investigator specifies design d = (s, π), the simulated niche/process P_N, and replication plan B\nthen compares declared cells—not a distribution of real-world niches",
        ha="center",
        va="center",
        fontsize=8.7,
        color=COLORS["slate"],
        fontweight="bold",
    )

    input_boxes = (
        (
            0.04,
            0.58,
            0.26,
            0.15,
            "Agent model M_A",
            "states · observations · priors\naction kernels · preferences",
            COLORS["purple"],
        ),
        (
            0.04,
            0.37,
            0.26,
            0.15,
            "Specified simulated niche P_N",
            "synthetic task environment\nnot a real-world niche distribution\nhidden truth · context · emissions\nmissingness · costs",
            COLORS["teal"],
        ),
        (
            0.04,
            0.16,
            0.26,
            0.15,
            "Testing setting T",
            "hypotheses · evidence · error rule\nstopping/cost contract · assumptions",
            COLORS["orange"],
        ),
    )
    for x, y, width, height, title, body, color in input_boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor="white",
                edgecolor=color,
                linewidth=2.0,
            )
        )
        ax.text(x + 0.016, y + height - 0.035, title, fontsize=10.3, fontweight="bold", color=color, va="top")
        ax.text(x + 0.016, y + height - 0.078, body, fontsize=8.0, color=COLORS["slate"], va="top", linespacing=1.22)

    scenario_x, scenario_y, scenario_w, scenario_h = 0.37, 0.49, 0.26, 0.19
    ax.add_patch(
        FancyBboxPatch(
            (scenario_x, scenario_y),
            scenario_w,
            scenario_h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor="#f7fafc",
            edgecolor=COLORS["slate"],
            linewidth=2.3,
        )
    )
    ax.text(0.50, 0.627, "PowerScenario s = (M_A, P_N, T)", ha="center", va="center", fontsize=11.1, fontweight="bold", color=COLORS["slate"])
    confirmation_label = (
        " + one split-confirmation-only static contract"
        if has_split_confirmation_contract
        else ""
    )
    ax.text(
        0.50,
        0.585,
        f"stable scenario hash · {scenario_count} operating contracts{confirmation_label}",
        ha="center",
        va="center",
        fontsize=8.1,
        color=COLORS["slate"],
    )
    ax.text(0.50, 0.545, "policy π is intentionally not inside this hash", ha="center", va="center", fontsize=8.3, color=COLORS["slate"])

    policy_x, policy_y, policy_w, policy_h = 0.37, 0.19, 0.26, 0.16
    ax.add_patch(
        FancyBboxPatch(
            (policy_x, policy_y),
            policy_w,
            policy_h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="white",
            edgecolor=COLORS["blue"],
            linewidth=2.0,
        )
    )
    ax.text(0.50, 0.307, "Agent policy π (separate row key)", ha="center", va="center", fontsize=10.0, fontweight="bold", color=COLORS["blue"])
    ax.text(0.50, 0.255, "visible filtration F_t → target · sensor\ncontinue or stop", ha="center", va="center", fontsize=8.5, color=COLORS["slate"], linespacing=1.28)

    trace_x, trace_y, trace_w, trace_h = 0.70, 0.49, 0.26, 0.19
    ax.add_patch(
        FancyBboxPatch(
            (trace_x, trace_y),
            trace_w,
            trace_h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor="#f7fafc",
            edgecolor=COLORS["purple"],
            linewidth=2.0,
        )
    )
    ax.text(0.83, 0.627, "Repeated simulated agent–niche trace b", ha="center", va="center", fontsize=10.0, fontweight="bold", color=COLORS["purple"])
    ax.text(0.83, 0.58, "agent sees visible history only", ha="center", va="center", fontsize=8.5, color=COLORS["slate"])
    ax.text(0.83, 0.54, "evaluator retains hidden truth only for scoring", ha="center", va="center", fontsize=8.0, color=COLORS["slate"])

    output_x, output_y, output_w, output_h = 0.70, 0.15, 0.26, 0.22
    ax.add_patch(
        FancyBboxPatch(
            (output_x, output_y),
            output_w,
            output_h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor="#f0fdf4",
            edgecolor=COLORS["teal"],
            linewidth=2.1,
        )
    )
    ax.text(0.83, 0.325, "Investigator-estimated conditional operating characteristics", ha="center", va="center", fontsize=9.3, fontweight="bold", color=COLORS["teal"])
    ax.text(0.83, 0.277, "mean TPR (reported power | d) ± MCSE\nFDR · FWER · cost · stopping", ha="center", va="center", fontsize=8.8, color=COLORS["slate"], linespacing=1.3)
    ax.text(0.83, 0.192, f"{action_count} action-loop records · validity labels", ha="center", va="center", fontsize=7.8, color=COLORS["slate"])

    for x, y, width, height, *_ in input_boxes:
        ax.annotate(
            "",
            xy=(scenario_x, scenario_y + scenario_h / 2),
            xytext=(x + width, y + height / 2),
            arrowprops={"arrowstyle": "-|>", "color": COLORS["grid"], "lw": 1.45, "connectionstyle": "arc3,rad=0.07"},
        )
    ax.annotate(
        "",
        xy=(trace_x, trace_y + trace_h / 2),
        xytext=(scenario_x + scenario_w, scenario_y + scenario_h / 2),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["grid"], "lw": 1.75},
    )
    ax.annotate(
        "",
        xy=(trace_x + 0.02, trace_y + 0.03),
        xytext=(policy_x + policy_w, policy_y + policy_h / 2),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["grid"], "lw": 1.55, "connectionstyle": "arc3,rad=-0.16"},
    )
    ax.annotate(
        "",
        xy=(output_x + output_w / 2, output_y + output_h),
        xytext=(trace_x + trace_w / 2, trace_y),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["grid"], "lw": 1.7},
    )
    ax.text(
        0.04,
        0.055,
        "A fixed grid of specified simulated niches is a sensitivity comparison, not an unknown or empirical distribution of real-world niches; the same role contract drives configuration, traces, captions, and claim boundaries.",
        fontsize=8.6,
        color=COLORS["slate"],
        style="italic",
    )
    fig.tight_layout()
    return _save(fig, path, _figure_dpi(results))


def _assert_text_fits_contract_boxes(
    fig: plt.Figure,
    box_texts: Sequence[tuple[str, Any, FancyBboxPatch]],
) -> None:
    """Fail closed if a contract-box label would render outside its patch."""
    fig.canvas.draw()
    # This module explicitly selects the Agg backend above; the base canvas
    # annotation does not expose Agg's renderer accessor to static checkers.
    renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()
    tolerance = 1.5
    for name, text, patch in box_texts:
        text_bounds = text.get_window_extent(renderer)
        patch_bounds = patch.get_window_extent(renderer)
        smallest_margin = min(
            text_bounds.x0 - patch_bounds.x0,
            patch_bounds.x1 - text_bounds.x1,
            text_bounds.y0 - patch_bounds.y0,
            patch_bounds.y1 - text_bounds.y1,
        )
        if smallest_margin < -tolerance:
            raise ValueError(f"testing-setting contract text exceeds its box: {name}")


def _testing_setting_contract(results: dict[str, Any], path: Path) -> str:
    """Render the declared testing setting as a readable preregistration map."""
    experiment = results.get("experiment", {})
    action_setting = experiment.get("action_loop", {})
    meta = results.get("meta", {})
    alpha = float(experiment.get("alpha", meta.get("alpha", 0.05)))
    hypothesis_count = int(experiment.get("m", 0))
    horizon = int(action_setting.get("horizon", 0))
    evidence_roles = experiment.get("suite", {}).get("evidence_roles", {})
    conditional_role = str(
        evidence_roles.get("synthetic", "conditional evidence under declared assumptions")
    ).replace("_", " ")
    blocks = (
        (
            "Hypotheses and truth",
            f"{hypothesis_count} hypotheses; one-sided p-values; null and alternative states; effect grid and null fraction fixed.",
            COLORS["purple"],
        ),
        (
            "Null law and evidence",
            f"Binomial upper-tail p-values; Bernoulli e-processes; confidence sequences; horizon={horizon}.",
            COLORS["teal"],
        ),
        (
            "Decision and accounting",
            f"alpha={alpha:g}; fixed-family rules, likelihood-ratio e-process evidence, and online alpha-accounting are tracked separately.",
            COLORS["orange"],
        ),
        (
            "Stopping and cost contract",
            f"Default stopping, budget, and trace accounting; {len(action_setting.get('policies', ())) } separately evaluated policies are recorded separately.",
            COLORS["blue"],
        ),
        (
            "Reported estimands",
            "Power, false discovery rate, family-wise error rate, rejections, stopping time, observations, cost, and target selection.",
            COLORS["magenta"],
        ),
        (
            "Validity boundary",
            f"Role: {conditional_role}; interpretation remains conditional on the declared model, process, setting, policy, dependence, and stopping assumptions.",
            COLORS["slate"],
        ),
    )
    # Render at approximately manuscript width rather than relying on a very
    # wide raster that the PDF renderer must downscale.  This keeps the box
    # text at a useful printed point size while retaining a high-resolution
    # source image.
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.04,
        0.972,
        "Testing-setting contract: declare before simulation",
        fontsize=16,
        fontweight="bold",
        color=COLORS["slate"],
        va="top",
    )
    ax.text(
        0.04,
        0.925,
        textwrap.fill(
            "The setting defines the evidence object, decision/accounting rule, stopping/cost contract, estimands, and boundary. A policy is evaluated separately against that setting.",
            width=92,
        ),
        fontsize=10,
        color=COLORS["slate"],
        va="top",
    )
    positions = (
        (0.04, 0.555),
        (0.355, 0.555),
        (0.67, 0.555),
        (0.04, 0.16),
        (0.355, 0.16),
        (0.67, 0.16),
    )
    width = 0.285
    height = 0.275
    box_texts: list[tuple[str, Any, FancyBboxPatch]] = []
    for (title, body, color), (x, y) in zip(blocks, positions):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="white",
            edgecolor=color,
            linewidth=2.0,
        )
        ax.add_patch(patch)
        title_text = ax.text(
            x + 0.018,
            y + height - 0.035,
            textwrap.fill(title, width=22),
            fontsize=10,
            fontweight="bold",
            color=color,
            va="top",
        )
        body_text = ax.text(
            x + 0.018,
            y + height - 0.105,
            textwrap.fill(body, width=35 if title == "Validity boundary" else 31),
            fontsize=8.25,
            color=COLORS["slate"],
            va="top",
            linespacing=1.18,
        )
        box_texts.extend(
            (
                (f"{title} title", title_text, patch),
                (f"{title} body", body_text, patch),
            )
        )
    for left, right in ((positions[0], positions[1]), (positions[1], positions[2]), (positions[3], positions[4]), (positions[4], positions[5])):
        ax.annotate(
            "",
            xy=(right[0] - 0.008, right[1] + height / 2),
            xytext=(left[0] + width + 0.008, left[1] + height / 2),
            arrowprops={"arrowstyle": "-|>", "color": COLORS["grid"], "lw": 1.6},
        )
    ax.annotate(
        "",
        xy=(positions[3][0] + width / 2, positions[3][1] + height + 0.012),
        xytext=(positions[0][0] + width / 2, positions[0][1] - 0.012),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["grid"], "lw": 1.5},
    )
    ax.annotate(
        "",
        xy=(positions[4][0] + width / 2, positions[4][1] + height + 0.012),
        xytext=(positions[1][0] + width / 2, positions[1][1] - 0.012),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["grid"], "lw": 1.5},
    )
    ax.annotate(
        "",
        xy=(positions[5][0] + width / 2, positions[5][1] + height + 0.012),
        xytext=(positions[2][0] + width / 2, positions[2][1] - 0.012),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["grid"], "lw": 1.5},
    )
    ax.text(
        0.04,
        0.07,
        textwrap.fill(
            "Contract status: parameters, seeds, evidence class, uncertainty convention, and non-claim are serialized with the result artifact.",
            width=116,
        ),
        fontsize=8.7,
        color=COLORS["slate"],
        style="italic",
        va="center",
    )
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.035, top=0.98)
    _assert_text_fits_contract_boxes(fig, box_texts)
    return _save(fig, path, _figure_dpi(results))


def _benchmark_stress(results: dict[str, Any], path: Path) -> str:
    benchmark = results["benchmarks"]
    trace = benchmark["agent_trace"]
    synthetic = benchmark["synthetic_reference"]
    labels = ["Synthetic null", "Staged agent trace"]
    values = [float(synthetic["terminal_e"]), float(trace["terminal_e"])]
    colors = [COLORS["blue"], COLORS["magenta"]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
    ax.set_ylabel(
        "Terminal e-process value E_T\n(dimensionless; linear scale)",
        fontsize=9,
        labelpad=7,
    )
    ax.set_title("Offline benchmark stress track")
    ax.set_ylim(0.0, max(values) * 1.18)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    ax.text(
        0.5,
        -0.18,
        f"agent rows={int(trace['rows']):,} | subjects={int(trace['subjects']):,}\n"
        f"Golub fixture available={bool(benchmark['golub']['available'])}\n"
        "claim role: stress test only",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=COLORS["slate"],
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.88, "edgecolor": COLORS["grid"]},
    )
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.28, top=0.88)
    return _save(fig, path, _figure_dpi(results))


def _scholarship_boundary(results: dict[str, Any], path: Path) -> str:
    """Render a source-to-claim map without presenting it as numeric evidence."""
    meta = results.get("meta", {})
    n_reps = _format_count(meta.get("n_reps"))
    rows = (
        (
            "FDR and dependence",
            "BH · BY · weighted BH",
            "Finite fixed-horizon simulations\nand certificates",
            "Declared p-value and dependence\nassumptions; no universal ranking",
            COLORS["blue"],
        ),
        (
            "Active-inference\ngenerative models",
            "Binary pymdp agent +\nclosed-form oracle",
            "Posterior/evidence calibration\nunder one binary channel",
            "Narrow two-state instantiation;\nnot active inference in general",
            COLORS["purple"],
        ),
        (
            "Sequential evidence",
            "Bernoulli likelihood-ratio\ne-process",
            "Likelihood-ratio e-process E_t\nand Ville-bound crossing diagnostic",
            "Optional-stopping validity only\nunder the declared null",
            COLORS["teal"],
        ),
        (
            "Online multiple testing",
            "LORD++ · SAFFRON\nfinite traces",
            "Levels, candidates, rejections,\nprocedural α-accounting,\nWilson uncertainty",
            "Calibration evidence, not a new\nuniversal online-FDR theorem",
            COLORS["orange"],
        ),
        (
            "Selective inference",
            "Predictable selection +\nheld-out controls",
            "Filtration contract and\nroadmap entry criteria",
            "Chronology alone does not\nvalidate post-selection p-values",
            COLORS["magenta"],
        ),
        (
            "Time-uniform concentration",
            "Confidence sequences +\nVille-bound vocabulary",
            "Bernoulli e-process path\nand bounded uncertainty tools",
            "Only the declared Bernoulli\ne-process is implemented",
            COLORS["slate"],
        ),
    )
    headers = ("Literature layer", "Implemented object", "Evidence shown", "Boundary")
    x_positions = (0.02, 0.245, 0.49, 0.745)
    widths = (0.205, 0.225, 0.235, 0.235)
    header_y = 0.80
    row_height = 0.112
    fig, ax = plt.subplots(figsize=(13, 8.0))
    ax.axis("off")
    ax.text(
        0.02,
        0.965,
        "Scholarship, implementation, and validity boundaries",
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        color=COLORS["slate"],
        va="top",
    )
    ax.text(
        0.02,
        0.915,
        "Read each row from established theory → coded object → observed evidence → explicit non-claim.",
        transform=ax.transAxes,
        fontsize=10,
        color=COLORS["slate"],
        va="top",
    )
    for x, width, header in zip(x_positions, widths, headers):
        ax.add_patch(
            FancyBboxPatch(
                (x, header_y),
                width,
                0.065,
                boxstyle="round,pad=0.008,rounding_size=0.008",
                transform=ax.transAxes,
                facecolor=COLORS["slate"],
                edgecolor="white",
                linewidth=1.0,
            )
        )
        ax.text(
            x + width / 2.0,
            header_y + 0.032,
            header,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
    for row_index, (layer, implementation, evidence, boundary, color) in enumerate(rows):
        y = header_y - (row_index + 1) * row_height
        values = (layer, implementation, evidence, boundary)
        for column_index, (x, width, value) in enumerate(zip(x_positions, widths, values)):
            facecolor = f"{color}18" if column_index == 0 else "#f8fafc"
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    width,
                    row_height - 0.012,
                    boxstyle="round,pad=0.008,rounding_size=0.008",
                    transform=ax.transAxes,
                    facecolor=facecolor,
                    edgecolor=COLORS["grid"],
                    linewidth=0.8,
                )
            )
            ax.text(
                x + width / 2.0,
                y + (row_height - 0.012) / 2.0,
                value,
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=COLORS["slate"],
                fontsize=9,
                fontweight="bold" if column_index == 0 else "normal",
                linespacing=1.25,
            )
    ax.text(
        0.02,
        0.095,
        "Primary source families: Benjamini–Hochberg / Benjamini–Yekutieli; "
        "Friston / Da Costa; Shafer / safe testing; Ramdas / SAFFRON.",
        transform=ax.transAxes,
        fontsize=9,
        color=COLORS["slate"],
        va="center",
    )
    ax.text(
        0.02,
        0.045,
        f"Release evidence context: {n_reps} seeded synthetic replications. This map is a provenance aid, not a measured outcome.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["slate"],
        va="center",
        style="italic",
    )
    return _save(fig, path, _figure_dpi(results))


def _formalism_code_map(
    results: dict[str, Any], path: Path, *, panel_index: int = 0
) -> str:
    """Render one readable full-page panel of the formalism registry."""
    registry = results.get("formalism", {})
    entries = list(registry.get("entries", []))
    start, end = _formalism_panel_bounds(len(entries), panel_index)
    panel_entries = entries[start - 1 : end]
    layout = _formalism_code_map_layout(len(panel_entries))
    # The matrix occupies its own manuscript page.  It is a deliberately
    # print-sized overview: every row exposes one representative code/test
    # topic while the hashed registry remains the complete executable record.
    fig, ax = plt.subplots(figsize=(8.0, 10.6))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.985)
    ax.axis("off")
    ax.text(
        0.02,
        0.985,
        "Formal objects → executable code → bounded evidence",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=COLORS["slate"],
        va="top",
    )
    ax.text(
        0.02,
        0.935,
        textwrap.fill(
            f"Print-sized overview for registry entries {start}–{end} of {len(entries)} | hash {registry.get('registry_hash', 'unavailable')} | complete symbols, paths, assumptions, and non-claims remain in the registry artifact",
            width=92,
        ),
        transform=ax.transAxes,
        fontsize=9,
        color=COLORS["slate"],
        va="top",
    )
    columns = (
        (0.02, 0.29, "Registry concept / evidence"),
        (0.33, 0.34, "Representative code + check"),
        (0.69, 0.29, "Short claim boundary"),
    )
    header_y = layout["header_y"]
    row_height = layout["row_height"]
    row_gap = layout["row_gap"]
    for x, width, label in columns:
        ax.add_patch(
            FancyBboxPatch(
                (x, header_y),
                width,
                0.05,
                transform=ax.transAxes,
                boxstyle="round,pad=0.005,rounding_size=0.005",
                facecolor=COLORS["slate"],
                edgecolor="white",
                linewidth=0.8,
            )
        )
        ax.text(
            x + width / 2,
            header_y + 0.025,
            textwrap.fill(label, width=20),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.7,
            color="white",
            fontweight="bold",
        )
    palette = [COLORS["blue"], COLORS["purple"], COLORS["teal"], COLORS["orange"], COLORS["magenta"]]
    for index, entry in enumerate(panel_entries):
        y = header_y - (index + 1) * row_height
        color = palette[(panel_index * _FORMALISM_ROWS_PER_PANEL + index) % len(palette)]
        code_topics = list(entry.get("code_symbols", []) or [])
        test_topics = list(entry.get("test_paths", []) or [])
        representative_code = _formalism_display_reference(
            code_topics[0] if code_topics else "no public symbol recorded"
        )
        representative_test = _formalism_display_reference(
            test_topics[0] if test_topics else "no test path recorded",
            test_path=True,
        )
        compact_title = textwrap.shorten(
            str(entry.get("title", entry.get("entry_id", "formal object"))),
            width=58,
            placeholder="…",
        )
        compact_boundary = textwrap.shorten(
            str(entry.get("non_claim", "")), width=58, placeholder="…"
        )
        evidence_label = str(entry.get("evidence_class", "")).replace("_", " ")
        values = (
            textwrap.fill(
                compact_title,
                width=28,
                break_long_words=False,
                break_on_hyphens=False,
            )
            + "\n["
            + textwrap.fill(evidence_label, width=28, break_long_words=False)
            + "]",
            textwrap.fill(
                f"Code: {representative_code}",
                width=36,
                break_long_words=False,
                break_on_hyphens=False,
            )
            + "\n"
            + textwrap.fill(
                f"Check: {representative_test}",
                width=36,
                break_long_words=False,
                break_on_hyphens=False,
            ),
            textwrap.fill(
                f"Not claimed: {compact_boundary}",
                width=30,
                break_long_words=False,
                break_on_hyphens=False,
            ),
        )
        for column_index, (x, width, value) in enumerate(
            (columns[column_index][0], columns[column_index][1], values[column_index])
            for column_index in range(len(columns))
        ):
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    width,
                    row_height - row_gap,
                    transform=ax.transAxes,
                    boxstyle="round,pad=0.005,rounding_size=0.005",
                    facecolor=f"{color}18" if column_index == 0 else "#f8fafc",
                    edgecolor=COLORS["grid"],
                    linewidth=0.6,
                )
            )
            ax.text(
                x + 0.008,
                y + (row_height - row_gap) / 2,
                value,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=8.65 if column_index != 0 else 8.8,
                color=COLORS["slate"],
                fontweight="bold" if column_index == 0 else "normal",
                linespacing=1.08,
                clip_on=True,
            )
    ax.text(
        0.02,
        0.06,
        textwrap.fill(
            "Read across each row: formal object and evidence class, then one representative code topic, one representative test topic, and a short non-claim. Exact symbols and test paths remain in the registry artifact; it also retains assumptions and complete non-claims.",
            width=116,
        ),
        transform=ax.transAxes,
        fontsize=8.7,
        color=COLORS["slate"],
        style="italic",
        va="center",
    )
    return _save(fig, path, _figure_dpi(results))


def _formalism_code_map_continued(results: dict[str, Any], path: Path) -> str:
    """Render the second full-page formalism-registry panel."""
    return _formalism_code_map(results, path, panel_index=1)


def _formalism_code_map_final(results: dict[str, Any], path: Path) -> str:
    """Render the final full-page formalism-registry panel."""
    return _formalism_code_map(results, path, panel_index=2)


def generate_figures(results: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Generate all publication figures from a result dictionary.

    Returns a mapping from stable artifact names to absolute paths. The input
    schema is intentionally the same JSON schema written by the result script,
    so a figure cannot silently drift to a second data source.
    """
    output_dir = Path(output_dir)
    return {
        "method_comparison": _method_comparison(results, output_dir / FIGURE_NAMES[0]),
        "power_curve": _power_curve(results, output_dir / FIGURE_NAMES[1]),
        "dependence_sweep": _dependence_sweep(results, output_dir / FIGURE_NAMES[2]),
        "active_inference_policy": _active_inference_policy(
            results, output_dir / FIGURE_NAMES[3]
        ),
        "certificate_summary": _certificate_summary(
            results, output_dir / FIGURE_NAMES[4]
        ),
        "pvalue_calibration": _pvalue_calibration(
            results, output_dir / FIGURE_NAMES[5]
        ),
        "sample_size_precision": _sample_size_precision(
            results, output_dir / FIGURE_NAMES[6]
        ),
        "paired_method_contrasts": _paired_method_contrasts(
            results, output_dir / FIGURE_NAMES[7]
        ),
        "sequential_evidence": _sequential_evidence(
            results, output_dir / FIGURE_NAMES[8]
        ),
        "online_fdr_trace": _online_fdr_trace(
            results, output_dir / FIGURE_NAMES[9]
        ),
        "conditional_selection_calibration": _conditional_selection_calibration(
            results, output_dir / FIGURE_NAMES[10]
        ),
        "benchmark_stress": _benchmark_stress(
            results, output_dir / FIGURE_NAMES[11]
        ),
        "scholarship_boundary": _scholarship_boundary(
            results, output_dir / FIGURE_NAMES[12]
        ),
        "action_loop_trace": _action_loop_trace(
            results, output_dir / FIGURE_NAMES[13]
        ),
        "conditional_power_surface": _conditional_power_surface(
            results, output_dir / FIGURE_NAMES[14]
        ),
        "action_policy_frontier": _action_policy_frontier(
            results, output_dir / FIGURE_NAMES[15]
        ),
        "scenario_decomposition": _scenario_decomposition(
            results, output_dir / FIGURE_NAMES[16]
        ),
        "testing_setting_contract": _testing_setting_contract(
            results, output_dir / FIGURE_NAMES[17]
        ),
        "formalism_code_map": _formalism_code_map(
            results, output_dir / FIGURE_NAMES[18]
        ),
        "formalism_code_map_continued": _formalism_code_map_continued(
            results, output_dir / FIGURE_NAMES[19]
        ),
        "formalism_code_map_final": _formalism_code_map_final(
            results, output_dir / FIGURE_NAMES[20]
        ),
        "adaptive_fdr_audit": _adaptive_fdr_audit(
            results, output_dir / FIGURE_NAMES[21]
        ),
        "computational_complexity": _computational_complexity(
            results, output_dir / FIGURE_NAMES[22]
        ),
    }


def write_figure_registry(
    output_dir: Path,
    figures: dict[str, str],
    results: dict[str, Any] | None = None,
) -> str:
    """Write a deterministic registry matching all manuscript figure labels."""
    output_dir = Path(output_dir)
    records = []
    captions = figure_captions(results)
    visible_comparisons = figure_visible_comparisons(results)
    alts = figure_alts(results)
    metadata = figure_metadata(results)
    expected = {spec.name for spec in FIGURE_SPECS}
    if set(figures) != expected:
        missing = sorted(expected - set(figures))
        extra = sorted(set(figures) - expected)
        raise ValueError(f"figure set mismatch; missing={missing}, extra={extra}")
    if set(FIGURE_METADATA) != expected:
        missing = sorted(expected - set(FIGURE_METADATA))
        extra = sorted(set(FIGURE_METADATA) - expected)
        raise ValueError(f"figure metadata mismatch; missing={missing}, extra={extra}")
    if set(FIGURE_PANEL_VALIDITY) != expected:
        missing = sorted(expected - set(FIGURE_PANEL_VALIDITY))
        extra = sorted(set(FIGURE_PANEL_VALIDITY) - expected)
        raise ValueError(
            f"figure panel-validity metadata mismatch; missing={missing}, extra={extra}"
        )
    for spec in sorted(FIGURE_SPECS, key=lambda item: item.label):
        path = Path(figures[spec.name])
        if not path.is_file():
            raise FileNotFoundError(f"generated figure is missing: {path}")
        records.append(
            {
                "caption": captions[spec.name],
                "filename": path.name,
                "generated_by": spec.generated_by,
                "label": spec.label,
                "alt": alts[spec.name],
                "reader_question": spec.reader_question,
                "visible_comparison": visible_comparisons[spec.name],
                "takeaway": spec.takeaway,
                "visual_encoding": spec.visual_encoding,
                "evaluation_perspective": spec.evaluation_perspective,
                "panel_validity": [dict(item) for item in FIGURE_PANEL_VALIDITY[spec.name]],
                "metadata": metadata[spec.name],
            }
        )
    registry = {"schema_version": "2.4", "figures": records}
    registry_path = output_dir / "figure_registry.json"
    _atomic_text(registry_path, json.dumps(registry, indent=2, sort_keys=True) + "\n")
    return str(registry_path)


def write_dashboard(
    results: dict[str, Any], output_path: Path, figures: dict[str, str]
) -> str:
    """Write a self-contained, dependency-light HTML results dashboard."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    specs = {spec.name: spec for spec in FIGURE_SPECS}
    captions = figure_captions(results)
    visible_comparisons = figure_visible_comparisons(results)
    alts = figure_alts(results)
    metadata_by_name = figure_metadata(results)
    result_lineage = dashboard_result_lineage_hash(results)
    for name, path in figures.items():
        relative = Path(path).relative_to(output_path.parent.parent)
        spec = specs[name]
        metadata = metadata_by_name[name]
        scenario_hashes = str(metadata["scenario_hashes"])
        scenario_prefixes = (
            "; ".join(
                f"{digest[:12]} (prefix)"
                for digest in scenario_hashes.split(";")
                if digest
            )
            if scenario_hashes != "not_applicable_fixed_horizon"
            else scenario_hashes
        )
        panel_validity = FIGURE_PANEL_VALIDITY[name]
        panel_validity_html = ""
        if panel_validity:
            panel_validity_html = (
                "<dt>Panel validity</dt><dd>"
                + html.escape(
                    "; ".join(
                        f"{item['panel']}: {item['validity']}"
                        for item in panel_validity
                    )
                )
                + "</dd>"
            )
        cards.append(
            f'<figure><img src="../figures/{html.escape(relative.name)}" alt="{html.escape(alts[name])}">'
            f"<figcaption><strong>{html.escape(FIGURE_DASHBOARD_TITLES[name])}</strong> — "
            f"{html.escape(captions[name])}</figcaption>"
            f'<dl class="figure-metadata">'
            f"<dt>Reader question</dt><dd>{html.escape(spec.reader_question)}</dd>"
            f"<dt>Visible comparison</dt><dd>{html.escape(visible_comparisons[name])}</dd>"
            f"<dt>Takeaway</dt><dd>{html.escape(spec.takeaway)}</dd>"
            f"<dt>Evaluation role</dt><dd>{html.escape(spec.evaluation_perspective)}</dd>"
            f"<dt>Estimand</dt><dd>{html.escape(metadata['estimand'])}</dd>"
            f"<dt>Uncertainty</dt><dd>{html.escape(metadata['uncertainty'])}</dd>"
            f"<dt>Evidence class</dt><dd>{html.escape(metadata['evidence_class'].replace('_', ' '))}</dd>"
            f"<dt>Sample size</dt><dd>{html.escape(metadata['sample_size'])}</dd>"
            f"<dt>Validity boundary</dt><dd>{html.escape(metadata['validity_boundary'])}</dd>"
            f"{panel_validity_html}"
            f"<dt>Visual encoding</dt><dd>{html.escape(spec.visual_encoding)}</dd>"
            f"<dt>Lineage</dt><dd>result <code>{html.escape(metadata['result_lineage_hash'][:12])}</code>; declared scenario hash prefixes {html.escape(scenario_prefixes)}</dd>"
            f"</dl></figure>"
        )
    meta = results.get("meta", {})
    context = (
        f"Seeded run: m={html.escape(str(results.get('experiment', {}).get('m', 'n/a')))}, "
        f"alpha={html.escape(str(meta.get('alpha', 'n/a')))}, "
        f"replications={html.escape(str(meta.get('n_reps', 'n/a')))}, "
        f"seed={html.escape(str(meta.get('seed', 'n/a')))}."
    )
    certificates = results.get("certificates", {})
    passed = sum(bool(record.get("passes")) for record in certificates.values())
    certificate_total = len(certificates)
    status = "PASS" if certificate_total and passed == certificate_total else "REVIEW"
    status_class = "pass" if status == "PASS" else "review"
    suite = results.get("suite", {})
    operating_contract_count = len(suite.get("scenario_contracts", []))
    split_contract_note = (
        "; plus one separately declared split-confirmation static scenario"
        if isinstance(suite.get("split_confirmation_scenario"), dict)
        else ""
    )
    scope_note = (
        "An investigator fixes the declared generative model, generative process, testing setting, "
        "policy, and replication plan; the embedded agent acts only on visible history and hidden truth is scored after each trace. "
        "Synthetic fixed-horizon simulations and "
        "certificates carry the formal evidence role; environment-level action control "
        "and benchmarks are conditional diagnostics or stress tests unless their stated "
        "validity assumptions are met. The artifact contains "
        f"{operating_contract_count} operating scenario contracts{split_contract_note}."
    )
    payload = html.escape(json.dumps(results, indent=2, sort_keys=True))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="result-lineage-sha256" content="{html.escape(result_lineage, quote=True)}"><title>Active Inference Power Suite results</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17202a;background:#fbfcfe}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}}figure{{margin:0;padding:1rem;border:1px solid #d7dde5;border-radius:8px;background:#fff;box-shadow:0 1px 3px #0001}}img{{max-width:100%;height:auto;display:block}}figcaption{{font-weight:600;margin-top:.75rem;line-height:1.4}}.figure-metadata{{display:grid;grid-template-columns:7.5rem minmax(0,1fr);gap:.25rem .6rem;margin:.8rem 0 0;color:#475569;font-size:.9rem;line-height:1.3}}.figure-metadata dt{{font-weight:800;color:#334155}}.figure-metadata dd{{margin:0}}.summary{{padding:1rem 1.2rem;margin:1.2rem 0;border:1px solid #d7dde5;border-radius:8px;background:#fff}}.badge{{display:inline-block;padding:.25rem .6rem;border-radius:999px;font-weight:800;letter-spacing:.04em;font-size:.8rem}}.pass{{color:#14532d;background:#dcfce7}}.review{{color:#7c2d12;background:#ffedd5}}.scope{{color:#475569;margin-bottom:0}}pre{{overflow:auto;background:#f6f8fa;padding:1rem;border-radius:8px}}</style></head>
<body><h1>Active Inference Power Suite</h1><p>Generated from the deterministic result artifact. Formal objects, executable code, figures, captions, and claims share one source.</p><section class="summary" aria-labelledby="evidence-status"><h2 id="evidence-status">Finite-simulation certificate gate <span class="badge {status_class}">{status}</span></h2><p>{context} {html.escape(str(passed))} of {html.escape(str(certificate_total))} certificates satisfy the declared local finite-simulation gate.</p><p class="scope"><strong>Scope:</strong> {html.escape(scope_note)}</p></section><main>{"".join(cards)}</main><h2>Machine-readable results</h2><pre>{payload}</pre></body></html>"""
    _atomic_text(output_path, document)
    return str(output_path)


def _atomic_text(path: Path, content: str) -> None:
    """Replace a text artifact only after a complete same-directory write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
