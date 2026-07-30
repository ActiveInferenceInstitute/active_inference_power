"""Real-output tests for the deterministic figure and dashboard layer."""

from __future__ import annotations

import json
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import FancyBboxPatch
import pytest
import yaml

from active_inference_power.figures import (
    FIGURE_METADATA,
    FIGURE_NAMES,
    FIGURE_READER_CONTEXT,
    FIGURE_SPECS,
    _DEPENDENCE_SERIES_STYLES,
    _ONLINE_FDR_EVENT_ROWS,
    _ONLINE_FDR_TRACE_STYLES,
    _TARGET_TRACE_STYLES,
    _assert_text_fits_contract_boxes,
    _complexity_order_display,
    _formalism_code_map_layout,
    _formalism_display_reference,
    _formalism_panel_bounds,
    _power_curve_errors,
    _scenario_cell_label,
    _testing_setting_contract,
    dashboard_result_lineage_hash,
    figure_alts,
    figure_captions,
    figure_metadata,
    figure_visible_comparisons,
    generate_figures,
    result_lineage_hash,
    write_dashboard,
    write_figure_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_testing_setting_contract_layout_guard_writes_a_complete_figure(
    tmp_path: Path,
    generated_fast_bundle: Path,
) -> None:
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    path = tmp_path / "testing_setting_contract.png"

    _testing_setting_contract(results, path)

    assert path.is_file()
    assert path.stat().st_size > 1000


def test_testing_setting_contract_layout_guard_rejects_text_outside_a_box() -> None:
    fig, ax = plt.subplots()
    patch = FancyBboxPatch((0.1, 0.1), 0.1, 0.1)
    ax.add_patch(patch)
    text = ax.text(0.95, 0.95, "outside")
    try:
        with pytest.raises(ValueError, match="text exceeds its box: fixture"):
            _assert_text_fits_contract_boxes(fig, [("fixture", text, patch)])
    finally:
        plt.close(fig)


def test_generate_figures_writes_all_publication_pngs(
    tmp_path: Path,
    generated_fast_bundle: Path,
) -> None:
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    n_reps_text = f"{int(results['meta']['n_reps']):,} replications"
    paths = generate_figures(results, tmp_path / "figures")
    assert len(paths) == len(FIGURE_NAMES) == 23
    for path in paths.values():
        artifact = Path(path)
        assert artifact.exists() and artifact.stat().st_size > 1000
        pixels = imread(artifact)
        assert pixels.ndim in (2, 3)
        assert pixels.shape[0] > 10 and pixels.shape[1] > 10
    registry_path = Path(write_figure_registry(tmp_path / "figures", paths, results))
    registry = json.loads(registry_path.read_text())
    assert registry["schema_version"] == "2.4"
    assert {item["label"] for item in registry["figures"]} == {
        "fig:method-comparison",
        "fig:power-curve",
        "fig:dependence",
        "fig:active-inference",
        "fig:certificates",
        "fig:pvalue-calibration",
        "fig:sample-size-precision",
        "fig:paired-contrasts",
        "fig:sequential-evidence",
        "fig:online-fdr",
        "fig:conditional-selection-calibration",
        "fig:benchmark-stress",
        "fig:scholarship-boundary",
        "fig:action-loop-trace",
        "fig:conditional-power-surface",
        "fig:action-policy-frontier",
        "fig:scenario-decomposition",
        "fig:testing-setting-contract",
        "fig:formalism-code-map",
        "fig:formalism-code-map-continued",
        "fig:formalism-code-map-final",
        "fig:adaptive-fdr-audit",
        "fig:computational-complexity",
    }
    assert all("9 configured correction procedures" in item["caption"] or item["label"] != "fig:method-comparison" for item in registry["figures"])
    assert all(n_reps_text in item["caption"] or item["label"] not in {"fig:method-comparison", "fig:power-curve"} for item in registry["figures"])
    assert all(item["caption"] for item in registry["figures"])
    assert all(item["alt"] for item in registry["figures"])
    assert all(
        item["reader_question"]
        and item["visible_comparison"]
        and item["takeaway"]
        and item["visual_encoding"]
        and item["evaluation_perspective"]
        for item in registry["figures"]
    )
    records_by_label = {item["label"]: item for item in registry["figures"]}
    assert {
        (spec.label, field): records_by_label[spec.label][field]
        for spec in FIGURE_SPECS
        for field in ("reader_question", "takeaway", "visual_encoding", "evaluation_perspective")
    } == {
        (spec.label, field): getattr(spec, field)
        for spec in FIGURE_SPECS
        for field in ("reader_question", "takeaway", "visual_encoding", "evaluation_perspective")
    }
    visible_comparisons = figure_visible_comparisons(results)
    assert {
        spec.label: records_by_label[spec.label]["visible_comparison"]
        for spec in FIGURE_SPECS
    } == {
        spec.label: visible_comparisons[spec.name]
        for spec in FIGURE_SPECS
    }
    assert all(
        item["caption"].startswith(f"{item['reader_question']} ")
        and item["visible_comparison"] in item["caption"]
        and item["takeaway"] in item["caption"]
        and item["metadata"]["validity_boundary"] in item["caption"]
        and "Reader question:" not in item["caption"]
        and item["caption"] != item["alt"]
        for item in registry["figures"]
    )
    assert all(
        set(item["metadata"]) == {
            "estimand", "uncertainty", "evidence_class", "validity_boundary",
            "redundant_encoding", "sample_size", "scenario_hashes", "result_lineage_hash",
        }
        and all(item["metadata"].values())
        for item in registry["figures"]
    )
    assert set(FIGURE_METADATA) == {Path(name).stem for name in FIGURE_NAMES}
    dependence_record = next(
        item for item in registry["figures"] if item["label"] == "fig:dependence"
    )
    assert "ρ=-0.004" in dependence_record["caption"]
    assert "ρ=0.75" in dependence_record["alt"]
    assert "BH circles/solid lines" in dependence_record["caption"]
    assert "BY squares with dashed lines" in dependence_record["alt"]
    assert "horizontally offset" in dependence_record["alt"]
    online_record = next(
        item for item in registry["figures"] if item["label"] == "fig:online-fdr"
    )
    assert "top-to-bottom LORD++/SAFFRON/e-LOND" in online_record["caption"]
    assert "top-to-bottom LORD++/SAFFRON/e-LOND" in online_record["alt"]
    sequential_record = next(
        item for item in registry["figures"] if item["label"] == "fig:sequential-evidence"
    )
    fixed_horizon_panel = next(
        item
        for item in sequential_record["panel_validity"]
        if item["panel"] == "Fixed-horizon p-value path"
    )
    assert "invalid if used for optional stopping" in fixed_horizon_panel["validity"].lower()
    selection_record = next(
        item
        for item in registry["figures"]
        if item["label"] == "fig:conditional-selection-calibration"
    )
    assert "gray-hatched n=0 cells" in selection_record["caption"]
    assert "n=0 cell" in selection_record["alt"]
    assert "no calibration estimate" in selection_record["alt"]
    assert "9 correction procedures" in figure_alts(results)["method_comparison"]
    scholarship_record = next(
        item
        for item in registry["figures"]
        if item["label"] == "fig:scholarship-boundary"
    )
    assert "provenance aid" in scholarship_record["caption"]
    assert "literature layer" in scholarship_record["alt"]
    decomposition_record = next(
        item
        for item in registry["figures"]
        if item["label"] == "fig:scenario-decomposition"
    )
    assert "Investigator-facing design decomposition" in decomposition_record["caption"]
    assert "investigator specifying" in decomposition_record["alt"]
    assert "separate agent policy" in decomposition_record["alt"]
    assert "specified simulated niche/process" in decomposition_record["alt"]
    assert "conditional replication-mean true-positive rate" in decomposition_record["alt"]
    assert "evaluation_perspective" in decomposition_record
    assert "embedded agent acts only on visible history" in decomposition_record["evaluation_perspective"]
    setting_record = next(
        item
        for item in registry["figures"]
        if item["label"] == "fig:testing-setting-contract"
    )
    assert "testing-setting contract" in setting_record["caption"].lower()
    assert "hypotheses and truth" in setting_record["alt"]
    formalism_record = next(
        item for item in registry["figures"] if item["label"] == "fig:formalism-code-map"
    )
    assert "entries 1" in formalism_record["caption"].lower()
    assert "first traceability" in formalism_record["alt"].lower()
    formal_entry_count = int(results["formalism"]["entry_count"])
    formal_ranges = [_formalism_panel_bounds(formal_entry_count, index) for index in range(3)]
    assert formal_ranges[0][0] == 1
    assert formal_ranges[-1][1] == formal_entry_count
    assert sum(end - start + 1 for start, end in formal_ranges) == formal_entry_count
    for start, end in formal_ranges:
        layout = _formalism_code_map_layout(end - start + 1)
        assert layout["last_row_bottom"] >= layout["footer_top"]
        assert layout["row_height"] > layout["row_gap"]
    continued_record = next(
        item
        for item in registry["figures"]
        if item["label"] == "fig:formalism-code-map-continued"
    )
    assert "second" in continued_record["caption"].lower()
    assert (tmp_path / "figures" / "formalism_code_map_continued.png").is_file()
    final_record = next(
        item
        for item in registry["figures"]
        if item["label"] == "fig:formalism-code-map-final"
    )
    assert "final" in final_record["caption"].lower()
    assert (tmp_path / "figures" / "formalism_code_map_final.png").is_file()
    adaptive_record = next(
        item for item in registry["figures"] if item["label"] == "fig:adaptive-fdr-audit"
    )
    assert "metric-specific" in adaptive_record["caption"]
    assert "pass or fail" in adaptive_record["alt"]
    assert adaptive_record["metadata"]["sample_size"].startswith("n=")
    assert "adaptive FDR" in adaptive_record["reader_question"]
    complexity_record = next(
        item
        for item in registry["figures"]
        if item["label"] == "fig:computational-complexity"
    )
    assert "source-work accounting" in complexity_record["caption"]
    assert "local wall time" in complexity_record["alt"]
    assert complexity_record["metadata"]["evidence_class"] == "provenance_contract"
    assert "no timing" in complexity_record["visual_encoding"]
    trace_record = next(
        item for item in registry["figures"] if item["label"] == "fig:action-loop-trace"
    )
    assert trace_record["metadata"]["evidence_class"] == "trace_diagnostic"
    assert "representative visible trajectory" in trace_record["metadata"]["estimand"]
    action_record = next(
        item for item in registry["figures"] if item["label"] == "fig:conditional-power-surface"
    )
    assert action_record["metadata"]["scenario_hashes"] != "not_applicable_fixed_horizon"
    assert all(len(value) == 64 for value in action_record["metadata"]["scenario_hashes"].split(";"))


def test_all_figure_specs_define_reader_context_and_caption_structure(
    generated_fast_bundle: Path,
) -> None:
    assert set(FIGURE_READER_CONTEXT) == {spec.name for spec in FIGURE_SPECS}
    assert len(FIGURE_SPECS) == len(FIGURE_NAMES) == 23
    for spec in FIGURE_SPECS:
        assert spec.reader_question
        assert spec.visible_comparison
        assert spec.takeaway
        assert spec.visual_encoding
        assert spec.evaluation_perspective
        assert spec.caption.startswith(f"{spec.reader_question} ")
        assert spec.visible_comparison in spec.caption
        assert spec.takeaway in spec.caption
        assert FIGURE_METADATA[spec.name]["validity_boundary"] in spec.caption
        assert "Reader question:" not in spec.caption

    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    visible_comparisons = figure_visible_comparisons(results)
    assert set(visible_comparisons) == {spec.name for spec in FIGURE_SPECS}
    assert all(value.strip() for value in visible_comparisons.values())
    captions = figure_captions(results)
    assert set(captions) == {spec.name for spec in FIGURE_SPECS}
    for spec in FIGURE_SPECS:
        assert captions[spec.name].startswith(f"{spec.reader_question} ")
        assert visible_comparisons[spec.name] in captions[spec.name]
        assert spec.takeaway in captions[spec.name]
        assert FIGURE_METADATA[spec.name]["validity_boundary"] in captions[spec.name]
    assert "likelihood-ratio e-process evidence E_t" in captions["sequential_evidence"]
    assert "procedural alpha-wealth" in captions["online_fdr_trace"]
    assert "declared simulation cost C" in captions["action_loop_trace"]
    assert "invalid if used for optional stopping" in captions["sequential_evidence"]
    for name in ("action_loop_trace", "conditional_power_surface", "action_policy_frontier"):
        assert "invalid if used for optional stopping" in captions[name]
    alts = figure_alts(results)
    assert "fixed-horizon optional-stopping boundary" in alts["action_loop_trace"]
    assert "fixed-horizon optional-stopping boundary" in alts["conditional_power_surface"]
    assert "fixed-horizon optional-stopping boundary" in alts["action_policy_frontier"]
    assert "marker" in FIGURE_READER_CONTEXT["sequential_evidence"]["visual_encoding"].lower()
    assert "capped error marks" in alts["power_curve"]
    assert "lower log-scale panel" in alts["power_curve"]
    assert "Dual-axis" not in alts["power_curve"]
    assert "Three vertically aligned small multiples" in alts["power_curve"]
    assert "share the configured effect-size axis" in alts["power_curve"]
    assert "BH circles/solid lines" in captions["dependence_sweep"]
    assert "BY squares with dashed lines" in alts["dependence_sweep"]
    assert "gray hatched n=0" in alts["conditional_selection_calibration"]
    assert "vertical position" in FIGURE_READER_CONTEXT["action_loop_trace"]["visual_encoding"]
    assert "sensor" in FIGURE_READER_CONTEXT["action_loop_trace"]["visual_encoding"]
    assert "dotted nominal-alpha" in FIGURE_READER_CONTEXT["conditional_power_surface"]["visual_encoding"]
    assert "directly labels its dotted nominal-alpha reference" in captions["conditional_power_surface"]
    assert "not timing measurements" in captions["computational_complexity"]
    assert "simulated resource cost C" in captions["computational_complexity"]
    assert "local wall time" in alts["computational_complexity"]
    for name in ("action_loop_trace", "conditional_power_surface", "action_policy_frontier"):
        assert captions[name].count("invalid if used for optional stopping") == 1
    assert captions["benchmark_stress"].count("upgrade synthetic validity claims") == 1
    for name in (
        "power_curve",
        "sequential_evidence",
        "online_fdr_trace",
        "conditional_selection_calibration",
        "action_loop_trace",
        "conditional_power_surface",
        "action_policy_frontier",
        "scenario_decomposition",
        "computational_complexity",
    ):
        assert len(captions[name]) <= 700


def test_figure_mark_and_role_contracts_prevent_semantic_regressions() -> None:
    """Keep tested redundant marks and investigator-facing role cues in sync."""
    assert _DEPENDENCE_SERIES_STYLES["fdr"] == (
        ("bh_fdr", "BH FDR", "#2f6f9f", "o-", -0.075),
        ("by_fdr", "BY FDR", "#805ad5", "s--", 0.075),
    )
    assert _DEPENDENCE_SERIES_STYLES["power"] == (
        ("bh_power", "BH power", "#2f855a", "o-", -0.075),
        ("by_power", "BY power", "#c05621", "s--", 0.075),
    )
    assert tuple((name, row) for name, _label, _color, row in _ONLINE_FDR_EVENT_ROWS) == (
        ("lord_plus_plus", 2),
        ("saffron", 1),
        ("e_lond", 0),
    )
    assert tuple(label for _name, label, _color, _row in _ONLINE_FDR_EVENT_ROWS) == (
        "LORD++",
        "SAFFRON",
        "e-LOND",
    )
    assert _ONLINE_FDR_TRACE_STYLES == {
        "lord_plus_plus": ("o", "-"),
        "saffron": ("s", "--"),
        "e_lond": ("D", ":"),
    }
    source = (
        PROJECT_ROOT / "src" / "active_inference_power" / "figures" / "__init__.py"
    ).read_text(encoding="utf-8")
    online_start = source.index("def _online_fdr_trace(")
    online_end = source.index("def _action_loop_trace(", online_start)
    online_trace = source[online_start:online_end]
    assert "events_ax.set_yticks(event_positions, event_labels)" in online_trace
    assert "top to bottom" in online_trace
    power_start = source.index("def _power_curve(")
    power_end = source.index("def _dependence_sweep(", power_start)
    power_curve = source[power_start:power_end]
    assert "plt.subplots(\n        3," in power_curve
    assert "sharex=True" in power_curve
    assert "twinx" not in power_curve
    assert _TARGET_TRACE_STYLES[:2] == (("o", "-"), ("s", "--"))
    assert "X plus outline marks stopping" in FIGURE_READER_CONTEXT["action_loop_trace"]["visual_encoding"]
    scenario_encoding = FIGURE_READER_CONTEXT["scenario_decomposition"]["visual_encoding"]
    assert "labeled boxes" in scenario_encoding
    assert "directional arrows" in scenario_encoding
    assert "specified simulated niche" in scenario_encoding
    assert "conditional replication-mean TPR" in scenario_encoding
    assert "scenario-cell labels" in FIGURE_READER_CONTEXT["conditional_power_surface"]["visual_encoding"]
    assert "immutable scenario cells" in FIGURE_READER_CONTEXT["action_policy_frontier"]["visual_encoding"]
    for name in ("action_loop_trace", "conditional_power_surface", "action_policy_frontier"):
        assert "Investigator-facing" in FIGURE_READER_CONTEXT[name]["evaluation_perspective"]
        assert "visible history" in FIGURE_READER_CONTEXT[name]["evaluation_perspective"]
    assert "simulated niche/process" in figure_alts()["scenario_decomposition"]
    assert "conditional true-positive rate" in figure_alts()["scenario_decomposition"]


def test_reader_paths_explain_evaluator_roles_and_dense_figure_encodings(
    generated_fast_bundle: Path,
) -> None:
    """Keep the investigator estimand and non-color figure semantics auditable.

    These assertions intentionally use the source-owned reader fields and the
    generated, result-bound caption/alternative-text fields together.  That
    prevents a future visual polish from restoring the misleading shorthand
    that power belongs intrinsically to an agent or its task environment, or
    from leaving dense marks intelligible only by color or a plot-adjacent note.
    """
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    captions = figure_captions(results)
    alts = figure_alts(results)
    registry = json.loads(
        (generated_fast_bundle / "figures" / "figure_registry.json").read_text(
            encoding="utf-8"
        )
    )
    records_by_label = {record["label"]: record for record in registry["figures"]}
    specs_by_name = {spec.name: spec for spec in FIGURE_SPECS}
    assert {
        spec.name: records_by_label[spec.label]["alt"] for spec in FIGURE_SPECS
    } == alts

    scenario_context = FIGURE_READER_CONTEXT["scenario_decomposition"]
    assert scenario_context["reader_question"].startswith("Who specifies the design")
    assert (
        "neither the agent nor the simulated niche has intrinsic power"
        in scenario_context["takeaway"]
    )
    assert "evaluator-only scoring" in scenario_context["visual_encoding"]
    assert "specified simulated niche/process" in alts["scenario_decomposition"]
    assert "evaluator-only hidden truth" in alts["scenario_decomposition"]

    for name in (
        "scenario_decomposition",
        "action_loop_trace",
        "conditional_power_surface",
        "action_policy_frontier",
    ):
        context = FIGURE_READER_CONTEXT[name]
        record = records_by_label[specs_by_name[name].label]
        assert "Investigator-facing" in context["evaluation_perspective"]
        assert "visible history" in context["evaluation_perspective"]
        assert record["evaluation_perspective"] == context["evaluation_perspective"]
        assert captions[name].startswith(f"{context['reader_question']} ")
        assert captions[name] != alts[name]

    surface_context = FIGURE_READER_CONTEXT["conditional_power_surface"]
    assert "not an agent's intrinsic power or a distribution over environments" in (
        surface_context["takeaway"]
    )
    assert "investigator-declared scenario/policy cells" in captions[
        "conditional_power_surface"
    ]
    assert "hidden evaluation state is intentionally absent" in alts["action_loop_trace"]

    method_encoding = FIGURE_READER_CONTEXT["method_comparison"]["visual_encoding"]
    assert "FWER-oriented" in method_encoding
    assert "FDR-oriented" in method_encoding
    assert "pale orange" in method_encoding and "pale blue" in method_encoding
    assert "FWER-oriented" in alts["method_comparison"]
    assert "FDR-oriented" in alts["method_comparison"]

    online_encoding = FIGURE_READER_CONTEXT["online_fdr_trace"]["visual_encoding"]
    assert "Three full-width stacked panels" in online_encoding
    assert "top-to-bottom LORD++/SAFFRON/e-LOND" in online_encoding
    for phrase in (
        "three stacked",
        "| = candidate",
        "* = rejection",
        "not likelihood-ratio e-process evidence",
    ):
        assert phrase in alts["online_fdr_trace"]
    assert "procedural alpha-wealth" in captions["online_fdr_trace"]
    assert "not likelihood-ratio e-process evidence" in captions["online_fdr_trace"]

    dependence_encoding = FIGURE_READER_CONTEXT["dependence_sweep"]["visual_encoding"]
    assert "BH circles with solid lines" in dependence_encoding
    assert "BY squares with dashed lines" in dependence_encoding
    assert "horizontal offsets" in dependence_encoding
    assert "horizontally offset" in alts["dependence_sweep"]

    selection_encoding = FIGURE_READER_CONTEXT["conditional_selection_calibration"]["visual_encoding"]
    assert "gray hatched n=0 cell" in selection_encoding
    assert "no calibration estimate" in alts["conditional_selection_calibration"]

    for name in (
        "formalism_code_map",
        "formalism_code_map_continued",
        "formalism_code_map_final",
    ):
        context = FIGURE_READER_CONTEXT[name]
        assert "print-sized" in context["takeaway"] or "print-sized" in context[
            "visual_encoding"
        ]
        assert "representative code" in context["visual_encoding"]
        assert "representative test" in context["visual_encoding"]
        assert "overview" in alts[name]
        assert "representative" in alts[name]


def test_figure_documentation_inventories_track_every_generated_figure() -> None:
    """Keep the public inventories synchronized with the source-owned registry."""
    stems = tuple(Path(name).stem for name in FIGURE_NAMES)
    labels = tuple(spec.label for spec in FIGURE_SPECS)
    profile = yaml.safe_load((PROJECT_ROOT / "domain_profile.yaml").read_text(encoding="utf-8"))
    plan = yaml.safe_load((PROJECT_ROOT / "experiment_plan.yaml").read_text(encoding="utf-8"))
    assert set(profile["figure_types"]) == set(stems)
    assert len(profile["figure_types"]) == len(stems)
    expected_artifacts = set(profile["artifact_expectations"])
    assert {f"output/figures/{name}" for name in FIGURE_NAMES}.issubset(expected_artifacts)
    assert tuple(plan["expected_figures"]) == labels

    syntax = (PROJECT_ROOT / "manuscript" / "SYNTAX.md").read_text(encoding="utf-8")
    inventory = (PROJECT_ROOT / "docs" / "output_inventory.md").read_text(encoding="utf-8")
    for spec, filename in zip(FIGURE_SPECS, FIGURE_NAMES, strict=True):
        assert spec.label in syntax
        assert f"output/figures/{filename}" in syntax
        assert f"<bundle-root>/figures/{filename}" in inventory
    for document in ("docs/output_inventory.md", "docs/output_conventions.md"):
        text = (PROJECT_ROOT / document).read_text(encoding="utf-8")
        assert "evaluation-perspective" in text


def test_scenario_cell_labels_resist_equal_reliability_collisions() -> None:
    """A figure must preserve scenario identity even when a metadata value agrees."""
    first = _scenario_cell_label("same_reliability_a", "a" * 64, {"reliability": 0.8})
    second = _scenario_cell_label("same_reliability_b", "b" * 64, {"reliability": 0.8})
    assert first != second
    assert "r=0.8" in first
    assert "r=not recorded" in _scenario_cell_label("unrecorded", "c" * 64, {})
    assert "…" in _scenario_cell_label("a_deliberately_long_nonreference_scenario_identifier", "d" * 64, {"reliability": 0.8})
    reference = _scenario_cell_label("reference_action_loop__r0p8__h12__a0p05", "e" * 64, {"reliability": 0.8})
    assert reference == "r=0.8\neeeeeeee"
    source = (PROJECT_ROOT / "src" / "active_inference_power" / "figures" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "duplicate action-loop scenario/policy row would overwrite a figure cell" in source
    assert "duplicate action-policy frontier row would overwrite a scenario cell" in source


def test_complexity_orders_are_typeset_for_reader_facing_legends() -> None:
    """Keep raw accounting strings out of the publication legend surface."""
    assert _complexity_order_display("Theta(m)") == r"$\Theta(m)$"
    assert _complexity_order_display("Theta(m log2 m)") == r"$\Theta(m\log_2 m)$"
    assert _complexity_order_display("O(T^2) implementation upper proxy") == (
        r"$O(T^2)$ implementation upper proxy"
    )
    with pytest.raises(ValueError, match="no typeset display form"):
        _complexity_order_display("Theta(m^4)")


def test_adaptive_fdr_annotations_reserve_clear_space_from_gate_marks() -> None:
    """Keep estimate text away from data-space bounds and gate markers."""
    source = (PROJECT_ROOT / "src" / "active_inference_power" / "figures" / "__init__.py").read_text(
        encoding="utf-8"
    )
    start = source.index("def _adaptive_fdr_audit(")
    end = source.index("def _computational_complexity(", start)
    audit = source[start:end]
    estimate_text = audit.index('f"estimate={float(estimate):.4f}"')
    estimate = audit[audit.rfind("fdr_axis.text(", 0, estimate_text):]
    assert "0.02" in estimate.split("fdr_axis.set_yticks", 1)[0]
    assert 'ha="left"' in estimate.split("fdr_axis.set_yticks", 1)[0]
    assert '"facecolor": "white"' in estimate.split("fdr_axis.set_yticks", 1)[0]


def test_power_curve_reads_current_serialized_mcse_fields(
    generated_fast_bundle: Path,
) -> None:
    """Keep the plot tied to the result schema's current MCSE field names."""
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    multiplier = float(results["meta"]["figure_error_multiplier"])
    power_errors, fdr_errors = _power_curve_errors(results["power_curve"], multiplier)
    assert all(error > 0.0 for error in power_errors)
    assert all(error > 0.0 for error in fdr_errors)
    assert power_errors[0] == multiplier * float(
        results["power_curve"][0]["mc_stderr_power"]
    )
    assert fdr_errors[0] == multiplier * float(
        results["power_curve"][0]["mc_stderr_fdr"]
    )


def test_pvalue_calibration_semantics_name_pooled_count_and_replications(
    tmp_path: Path,
    generated_fast_bundle: Path,
) -> None:
    """Keep DKW context tied to pooled p-values, not only run count."""
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    calibration = results["null_pvalue_calibration"]
    pooled = f"{int(calibration['n_pvalues']):,} pooled p-values"
    replications = f"{int(calibration['n_reps']):,} calibration replications"
    independent_replications = (
        f"{int(calibration['n_reps']):,} independent calibration replications"
    )
    captions = figure_captions(results)
    alts = figure_alts(results)
    metadata = figure_metadata(results)
    for text in (
        captions["pvalue_calibration"],
        alts["pvalue_calibration"],
        metadata["pvalue_calibration"]["sample_size"],
    ):
        assert pooled in text
        assert replications in text or independent_replications in text
    assert "DKW" in captions["pvalue_calibration"]
    assert "DKW" in alts["pvalue_calibration"]

    figures = generate_figures(results, tmp_path / "figures")
    dashboard = Path(write_dashboard(results, tmp_path / "web" / "dashboard.html", figures))
    html = dashboard.read_text(encoding="utf-8")
    assert "<strong>P-value calibration</strong>" in html
    assert "<strong>Adaptive FDR audit</strong>" in html


def test_quantitative_axes_name_the_estimand_aggregation_and_scale() -> None:
    """Keep dense panels interpretable without relying on the caption alone."""
    source = (PROJECT_ROOT / "src" / "active_inference_power" / "figures" / "__init__.py").read_text(
        encoding="utf-8"
    )
    for forbidden_term in (
        'ax.set_xlabel("Rate or power")',
        'ax.set_ylabel("method − BH")',
        'ax.set_ylabel("terminal e-value")',
    ):
        assert forbidden_term not in source
    for required in (
        "Replicate mean FDP\\n(empirical FDR; proportion; linear)",
        "Replicate mean power\\n(true-positive rate; proportion; linear)",
        "Replicate mean FWER\\n(any false rejection; proportion; linear)",
        "Replication-level error-rate estimate\\n(proportion; linear scale)",
        "Replication-level mean power estimate\\n(proportion; linear scale)",
        "Empirical CDF Fₙ(t) (pooled-p-value fraction; dimensionless)",
        "Paired replication mean\\n{metric_label} − BH (proportion)",
        "Terminal e-process value E_T\\n(dimensionless; linear scale)",
        "Likelihood-ratio e-process evidence Eₜ\\n(dimensionless; log scale)",
        "Procedure-specific accounting\\n(dimensionless; symlog)",
        "Simulated cost C\\n({cost_unit}; {cost_scale_id})",
        "Mean total simulated cost C\\n({cost_unit}; shared {cost_scale_id})",
        "Observed mean FDP\\n(finite simulation; proportion)",
        "Stopping time\\n(replication mean; action steps)",
        "Fixed-horizon policy: comparison only; invalid if used for optional stopping.",
        r"Source-work proxy $Q_{\\mathrm{comp}}$\n(dimensionless count; log₂ scale)",
        r"History/path work proxy $Q_{\\mathrm{comp}}$\n(dimensionless count; log₂ scale)",
        "Declared action-loop trace calls (count; linear scale)",
    ):
        assert required in source


def test_computational_complexity_boundary_typesets_distinct_quantities() -> None:
    """Keep the dense source-work boundary readable rather than identifier-like."""
    source = (PROJECT_ROOT / "src" / "active_inference_power" / "figures" / "__init__.py").read_text(
        encoding="utf-8"
    )
    boundary_start = source.index('"Boundary: source-work proxy"')
    boundary_end = source.index("figure.suptitle", boundary_start)
    boundary = source[boundary_start:boundary_end]
    for typeset_quantity in (
        r"$Q_{\\mathrm{comp}}$",
        r"$\\tau_{\\mathrm{wall}}$",
        r"$E_t$",
        r"$W_t^\\alpha$",
    ):
        assert typeset_quantity in boundary
    assert "Q_comp is source-level work" not in boundary
    assert 'text if "\\n" in text else textwrap.fill(text, width=47)' in boundary


def test_formalism_display_references_are_word_spaced_not_raw_identifiers() -> None:
    """Keep narrow traceability cells readable while preserving exact registry values."""
    implementation = _formalism_display_reference(
        "action_loop._e_process_contract_valid"
    )
    verification = _formalism_display_reference(
        "tests/test_dynamic_process_sensitivity.py", test_path=True
    )
    assert implementation == "action loop e process contract valid"
    assert verification == "dynamic process sensitivity test"
    assert all(token not in implementation + verification for token in ("_", ".", "/"))

    source = (PROJECT_ROOT / "src" / "active_inference_power" / "figures" / "__init__.py").read_text(
        encoding="utf-8"
    )
    formalism_start = source.index("def _formalism_code_map(")
    formalism_end = source.index("def _formalism_code_map_continued", formalism_start)
    formalism = source[formalism_start:formalism_end]
    assert "Exact symbols and test paths remain in the registry artifact" in formalism
    assert "break_long_words=False" in formalism


def test_write_dashboard_contains_result_payload_and_relative_images(
    tmp_path: Path,
    generated_fast_bundle: Path,
) -> None:
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    n_reps_text = f"{int(results['meta']['n_reps']):,} replications"
    figure_dir = tmp_path / "figures"
    paths = generate_figures(results, figure_dir)
    dashboard = Path(
        write_dashboard(results, tmp_path / "web" / "dashboard.html", paths)
    )
    html = dashboard.read_text()
    assert "Active Inference Power" in html
    assert "method_comparison.png" in html
    assert "schema_version" in html
    assert "9 configured correction procedures" in html
    assert n_reps_text in html
    assert "replications=" in html
    assert "scholarship boundary" in html.lower()
    assert "Reader question" in html
    assert "Visible comparison" in html
    assert "Takeaway" in html
    assert "Evaluation role" in html
    assert "embedded agent acts only on visible history" in html
    assert "Visual encoding" in html
    assert "Estimand" in html
    assert "validity boundary" in html.lower()
    lineage = dashboard_result_lineage_hash(results)
    assert (
        f'<meta name="result-lineage-sha256" content="{lineage}">' in html
    )
    assert len(lineage) == 64
    assert set(item["result_lineage_hash"] for item in figure_metadata(results).values()) == {
        lineage
    }


def test_dashboard_lineage_hash_tracks_scientific_result_content(
    generated_fast_bundle: Path,
) -> None:
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    lineage = dashboard_result_lineage_hash(results)
    assert lineage == result_lineage_hash(results)

    changed = json.loads(json.dumps(results))
    changed["meta"]["n_reps"] = int(changed["meta"]["n_reps"]) + 1
    assert dashboard_result_lineage_hash(changed) != lineage


def test_figure_metadata_ignores_non_list_scenario_blocks() -> None:
    metadata = figure_metadata({"suite": {"scenario_contracts": None}, "action_loop": {"scenarios": None}})
    assert all(item["scenario_hashes"] == "not_applicable_fixed_horizon" for item in metadata.values())


def test_cover_asset_is_configured_and_valid() -> None:
    publication_config = yaml.safe_load(
        (PROJECT_ROOT / "manuscript" / "config.yaml").read_text(encoding="utf-8")
    )
    cover = publication_config["paper"]["cover"]
    cover_path = PROJECT_ROOT / "manuscript" / cover["image"]
    assert cover["title"]
    assert cover["description"]
    assert cover_path.is_file()
    pixels = imread(cover_path)
    assert pixels.ndim == 3
    assert pixels.shape[:2] == (1536, 1024)


def test_generated_dashboard_is_preserved_for_publication_render(
    generated_fast_bundle: Path,
) -> None:
    manifest = json.loads(
        (generated_fast_bundle / "manifest.json").read_text(encoding="utf-8")
    )
    artifacts = manifest["artifacts"]
    dashboard = generated_fast_bundle / artifacts["dashboard"]["path"]
    assert artifacts["dashboard"]["root"] == "output"
    assert artifacts["dashboard"]["path"] == "review/dashboard.html"
    assert artifacts["run_log"]["root"] == "output"
    assert artifacts["run_log"]["path"] == "run_log.json"
    assert dashboard.is_file() and dashboard.stat().st_size > 1000
