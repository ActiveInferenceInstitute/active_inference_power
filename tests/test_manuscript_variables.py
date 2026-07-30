"""Tests for manuscript variable generation and injection integrity.

Real files only (the actual project manuscript + results.json), no mocks.
The token-coverage test binds the manuscript to the variable generator: every
``{{TOKEN}}`` that appears in a rendered manuscript section must resolve.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml

from active_inference_power.manuscript_variables import (
    _MISSING,
    _action_loop_rows,
    _cert_rows,
    _comparison_rows,
    _complexity_rows,
    _evidence_display_label,
    _interval_text,
    _learned_policy_rows,
    _load_config,
    _load_quality,
    _load_results,
    _paired_action_rows,
    _probability_or_not_estimable,
    _publication_repository_statement,
    _reader_evidence_status,
    _split_confirmation_rows,
    generate_variables,
    save_variables,
)
from active_inference_power.certificates import BAND_SIGMA
from active_inference_power.figures import FIGURE_SPECS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Same rule as infrastructure.rendering.manuscript_injection.
_TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
_EXCLUDED = {"AGENTS.md", "README.md", "SYNTAX.md"}


def _manuscript_tokens() -> set[str]:
    tokens: set[str] = set()
    for md in (PROJECT_ROOT / "manuscript").glob("*.md"):
        if md.name in _EXCLUDED:
            continue
        tokens |= set(_TOKEN_RE.findall(md.read_text(encoding="utf-8")))
    return tokens


def test_results_json_exists(generated_fast_bundle: Path) -> None:
    """The generator's single source of truth must be present."""
    assert (generated_fast_bundle / "results.json").is_file()


def test_every_manuscript_token_resolves(generated_fast_bundle: Path) -> None:
    """Every {{TOKEN}} used in the manuscript is produced by the generator."""
    variables = generate_variables(
        PROJECT_ROOT, require_analysis_outputs=True, output_root=generated_fast_bundle
    )
    used = _manuscript_tokens()
    missing = used - set(variables)
    assert not missing, f"manuscript uses tokens with no variable: {sorted(missing)}"


def test_all_figure_captions_and_alternative_text_are_injected_from_results(
    generated_fast_bundle: Path,
) -> None:
    variables = generate_variables(
        PROJECT_ROOT, require_analysis_outputs=True, output_root=generated_fast_bundle
    )
    used = _manuscript_tokens()
    caption_tokens = {token for token in used if token.startswith("FIGURE_") and token.endswith("_CAPTION")}
    alt_tokens = {token for token in used if token.startswith("FIGURE_") and token.endswith("_ALT")}
    assert len(caption_tokens) == len(FIGURE_SPECS)
    assert len(alt_tokens) == len(FIGURE_SPECS)
    assert "FIGURE_ERROR_BAND" in used
    assert all(variables[token] != _MISSING for token in caption_tokens)
    assert all(variables[token] != _MISSING for token in alt_tokens)


def test_online_fdr_method_variables_use_reader_facing_labels(
    generated_fast_bundle: Path,
) -> None:
    """Rendered prose must not expose serialized method identifiers."""
    variables = generate_variables(
        PROJECT_ROOT, require_analysis_outputs=True, output_root=generated_fast_bundle
    )
    assert variables["RESULT_ONLINE_TARGET_METHOD"] == "LORD++"


def test_action_evidence_display_keeps_per_target_scope_visible() -> None:
    """Publication tables must not compress a scoped e-process claim to generic validity."""
    assert _evidence_display_label("conditional_per_target_optional_stopping") == (
        "cond. per-target"
    )
    assert _evidence_display_label("conditional per-target optional-stopping claim") == (
        "cond. per-target"
    )


def test_every_manuscript_figure_pairs_a_caption_with_authored_alternative_text() -> None:
    """Source figures must keep reader captions distinct from rendered alt text."""
    image_pattern = re.compile(
        r"!\[\{\{(?P<caption>FIGURE_[A-Z0-9_]+_CAPTION)\}\}\]"
        r"\([^)]*\)\{#[^\s]+\s+alt=\"\{\{(?P<alt>FIGURE_[A-Z0-9_]+_ALT)\}\}\""
        r"[^}]*\}"
    )
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "manuscript").glob("*.md")
        if path.name not in _EXCLUDED
    )
    images = list(image_pattern.finditer(source))
    assert len(images) == len(FIGURE_SPECS)
    for image in images:
        caption = image.group("caption")
        expected_alt = caption.removesuffix("_CAPTION") + "_ALT"
        assert image.group("alt") == expected_alt


def test_manuscript_has_no_known_stale_quantitative_literals():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "manuscript").glob("*.md")
        if path.name not in _EXCLUDED
    )
    stale = (
        "±3 MC SE",
        "90\\% gate",
        "95\\% DKW",
        "pymdp==1.0.3",
        "fixed 300-DPI",
        "threshold of 0.9",
    )
    assert not [literal for literal in stale if literal in source]


def test_manuscript_numeric_literals_are_only_declared_allowlist_contexts():
    """Reject quantitative prose digits that bypass generated variables."""
    source_parts = []
    for path in (PROJECT_ROOT / "manuscript").glob("*.md"):
        if path.name in _EXCLUDED or path.name == "preamble.md":
            continue
        source_parts.append(path.read_text(encoding="utf-8"))
    source = "\n".join(source_parts)
    # Equations and citation/reference syntax are mathematical or scholarly
    # notation, not prose estimands. Figure widths are rendering metadata.
    source = re.sub(r"\\\[.*?\\\]", "", source, flags=re.DOTALL)
    source = re.sub(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", "", source, flags=re.DOTALL)
    source = re.sub(r"\\\(.*?\\\)", "", source, flags=re.DOTALL)
    source = re.sub(r"\$(?:\\.|[^$])*\$", "", source, flags=re.DOTALL)
    source = re.sub(r"\{[^}]*\}", "", source)
    source = re.sub(r"width=\d+(?:\.\d+)?%", "", source)
    source = re.sub(r"`[^`]*`", "", source)
    source = re.sub(r"@[A-Za-z][A-Za-z0-9_:-]*", "", source)
    source = re.sub(r"SHA-\d+", "", source)
    source = re.sub(r"(?m)^\s*\d+[.)]\s+", "", source)
    literals = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?%?", source)
    assert not literals, (
        "scientific numeric literals bypass generated variables; see "
        "docs/manuscript_numeric_audit.md: " + ", ".join(literals)
    )


def test_no_token_resolves_to_missing(generated_fast_bundle: Path) -> None:
    """With real results present, no used token falls back to the N/A sentinel."""
    variables = generate_variables(
        PROJECT_ROOT, require_analysis_outputs=True, output_root=generated_fast_bundle
    )
    used = _manuscript_tokens()
    na = [t for t in used if variables.get(t) == _MISSING]
    assert not na, f"tokens unexpectedly N/A: {na}"


def test_table_row_blocks_have_expected_shape(generated_fast_bundle: Path) -> None:
    variables = generate_variables(
        PROJECT_ROOT, require_analysis_outputs=True, output_root=generated_fast_bundle
    )
    config = yaml.safe_load((PROJECT_ROOT / "manuscript" / "config.yaml").read_text())
    # Configured procedures, effect grid, dependence grid, four certs, four AI rows.
    assert len(variables["RESULT_COMPARISON_ROWS"].splitlines()) == len(
        config["experiment"]["methods"]
    )
    assert len(variables["RESULT_POWERCURVE_ROWS"].splitlines()) == 9
    expected_dependence_rows = len(config["experiment"]["dependence_rhos"])
    if config["experiment"].get("dependence_block") is not None:
        expected_dependence_rows += 1
    assert len(variables["RESULT_DEPENDENCE_ROWS"].splitlines()) == expected_dependence_rows
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    assert len(variables["RESULT_CERT_ROWS"].splitlines()) == len(results["certificates"])
    assert len(variables["RESULT_AI_ROWS"].splitlines()) == 4
    assert len(variables["RESULT_SEQUENTIAL_ROWS"].splitlines()) == 2
    assert len(variables["RESULT_ONLINE_FDR_ROWS"].splitlines()) == len(
        results["online_fdr"]["methods"]
    )
    complexity_rows = variables["RESULT_COMPLEXITY_ROWS"].splitlines()
    assert len(complexity_rows) == len(results["computational_complexity"]["analytical_rows"])
    complexity_text = "\n".join(complexity_rows)
    assert "$\\Theta(m\\log_2 m)$" in complexity_text
    assert "$N_j, H_j$" in complexity_text
    assert "$E_t$" in complexity_text
    assert "$C$" in complexity_text
    complexity_prose = re.sub(r"\$(?:\\.|[^$])*\$", "", complexity_text)
    assert "Theta(" not in complexity_prose
    assert "Omega(" not in complexity_prose
    assert "log2" not in complexity_prose
    assert "| N_j,H_j |" not in complexity_prose
    assert len(variables["RESULT_AGENT_SAMPLING_ROWS"].splitlines()) == 6
    assert len(variables["RESULT_BENCHMARK_ROWS"].splitlines()) == 3
    sample_size_labels = {
        line.split("|")[1].strip()
        for line in variables["RESULT_SAMPLE_SIZE_ROWS"].splitlines()
    }
    assert sample_size_labels == {
        "Method comparison",
        "Power curve",
        "Dependence sweep",
        "Paired method comparisons",
        "Active inference",
        "Sequential evidence",
        "Online FDR",
        "Agent sampling traces",
        "Conditional action loop",
        "Null calibration",
        "Certificates",
    }
    assert len(variables["RESULT_FORMALISM_ROWS"].splitlines()) == int(
        json.loads(
            (generated_fast_bundle / "formalism_registry.json").read_text(
                encoding="utf-8"
            )
        )["entry_count"]
    )
    # Every row is a well-formed markdown table row.
    for key in ("RESULT_COMPARISON_ROWS", "RESULT_AI_ROWS"):
        for line in variables[key].splitlines():
            assert line.startswith("|") and line.rstrip().endswith("|")


def test_complexity_formatter_rejects_unmapped_source_order(
    generated_fast_bundle: Path,
) -> None:
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    complexity = deepcopy(results["computational_complexity"])
    complexity["analytical_rows"][0]["time_order"] = "unreviewed source-work expression"

    with pytest.raises(ValueError, match="unexpected source-work order"):
        _complexity_rows(complexity)


def test_values_match_results_json(generated_fast_bundle: Path) -> None:
    """Injected inline stats equal the computed results (no transcription drift)."""
    variables = generate_variables(
        PROJECT_ROOT, require_analysis_outputs=True, output_root=generated_fast_bundle
    )
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    bh = results["method_comparison"]["benjamini_hochberg"]
    assert variables["RESULT_BH_FDR"] == f"{bh['fdr']:.4f}"
    assert variables["RESULT_BH_POWER"] == f"{bh['power']:.3f}"
    gain = results["certificates"]["storey_power_gain"]["power_gain"]
    assert variables["RESULT_STOREY_GAIN"] == f"{gain:.3f}"
    storey_cert = results["certificates"]["storey_power_gain"]
    assert variables["RESULT_STOREY_FDR_ESTIMATE"] == f"{storey_cert['storey_fdr']:.5f}"
    assert variables["RESULT_STOREY_FDR_STDERR"] == f"{storey_cert['storey_fdr_stderr']:.6f}"
    assert variables["RESULT_STOREY_POWER_GAIN_STDERR"] == f"{storey_cert['power_gain_stderr']:.6f}"
    assert variables["RESULT_STOREY_FDR_GATE"] == (
        "passed" if storey_cert["storey_fdr_passes"] else "failed"
    )
    assert variables["CONFIG_CERTIFICATE_MCSE_MULTIPLIER"] == f"{BAND_SIGMA:g}"
    ai_results = results["active_inference"]
    ai_regimes = ai_results.get("regimes", ai_results)
    weak = ai_regimes["weak_evidence"]["posterior_threshold_fdr"]
    assert variables["RESULT_AI_WEAK_POSTERIOR_FDR"] == f"{weak:.3f}"


def test_release_status_tokens_follow_run_ledger_and_manifest(
    temporary_status_bundle,
) -> None:
    bundle = temporary_status_bundle(
        profile="release", force_all_certificates_pass=True
    )
    variables = generate_variables(
        PROJECT_ROOT,
        require_analysis_outputs=True,
        output_root=bundle["output_root"],
    )
    run_log = bundle["run_log"]
    manifest = bundle["manifest"]
    assert variables["RESULT_RUN_STATUS"] == run_log["status"]
    assert variables["RESULT_CERTIFICATE_GATE"] == (
        "passed" if manifest["certificates_pass"] else "failed"
    )
    assert run_log["profile"] == "release"
    assert manifest["certificates_pass"] is True
    assert variables["RESULT_RELEASE_STATUS"] == "release candidate bound to the current source"


def test_campaign_completion_never_emits_a_publication_status(
    temporary_status_bundle,
) -> None:
    campaign = temporary_status_bundle(
        profile="campaign", diagnostic_storey_failure=True, complete=True
    )["output_root"]
    variables = generate_variables(
        PROJECT_ROOT,
        require_analysis_outputs=True,
        output_root=campaign,
    )
    assert variables["RESULT_RELEASE_STATUS"] == "campaign-complete non-release"


def test_hydration_refuses_a_campaign_receipt_without_mutating_it(
    temporary_status_bundle,
) -> None:
    """Even --allow-draft cannot turn a campaign receipt into render input."""
    campaign = temporary_status_bundle(
        profile="campaign", diagnostic_storey_failure=True, complete=True
    )["output_root"]
    manifest = campaign / "manifest.json"
    before = manifest.read_bytes()
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "z_generate_manuscript_variables.py"),
            "--allow-draft",
            "--output-root",
            str(campaign),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "campaign output roots cannot be hydrated" in completed.stderr
    assert manifest.read_bytes() == before


def test_certificate_rows_encode_bound_direction(generated_fast_bundle: Path) -> None:
    rows = generate_variables(
        PROJECT_ROOT, require_analysis_outputs=True, output_root=generated_fast_bundle
    )[
        "RESULT_CERT_ROWS"
    ]
    assert "≤" in rows
    assert "≥" in rows


def test_split_confirmation_table_uses_compact_evidence_label(
    generated_fast_bundle: Path,
) -> None:
    rows = generate_variables(
        PROJECT_ROOT, require_analysis_outputs=True, output_root=generated_fast_bundle
    )[
        "RESULT_SPLIT_CONFIRMATION_ROWS"
    ]
    assert "formal conditional" in rows
    assert "formal_conditional_split_confirmation" not in rows


def test_split_confirmation_variables_bind_the_declared_static_scenario(
    generated_fast_bundle: Path,
) -> None:
    """The manuscript exposes the dedicated confirmation-channel lineage."""
    variables = generate_variables(
        PROJECT_ROOT, require_analysis_outputs=True, output_root=generated_fast_bundle
    )
    results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    contract = results["suite"]["split_confirmation_scenario"]
    assert variables["RESULT_SPLIT_CONFIRMATION_SCENARIO_ID"] == contract["scenario_id"]
    assert variables["RESULT_SPLIT_CONFIRMATION_SCENARIO_HASH"] == contract["scenario_hash"]


def test_action_loop_table_uses_compact_display_labels(
    generated_fast_bundle: Path,
) -> None:
    rows = generate_variables(
        PROJECT_ROOT, require_analysis_outputs=True, output_root=generated_fast_bundle
    )[
        "RESULT_ACTION_LOOP_ROWS"
    ]
    assert "diag." in rows
    assert "| diagnostic |" not in rows
    assert "optional-stopping claim" not in rows
    assert "| Posterior cutoff |" in rows
    assert "| Posterior draw |" in rows
    assert "| $E_t$ rule |" in rows
    assert "Posterior threshold" not in rows
    assert "Posterior sampling" not in rows


def test_action_loop_table_reader_key_preserves_policy_semantics() -> None:
    """Compact cells must not make the declared action rules ambiguous."""
    source = (PROJECT_ROOT / "manuscript" / "03_results.md").read_text(
        encoding="utf-8"
    )
    start = source.index("## Conditional action-loop operating characteristics")
    end = source.index("Adaptive target selection uses", start)
    section = source[start:end]
    for meaning in (
        "Fixed-$T$ (fixed horizon)",
        "Posterior cutoff\n(posterior-threshold stopping)",
        "Info gain (expected-information-gain policy)",
        "Cost rule (cost-aware policy)",
        "Posterior draw (posterior-sampling policy)",
        "$E_t$ rule (likelihood-ratio e-process stopping)",
    ):
        assert meaning in section
    assert "Which declared policy trades off power, error, cost, and stopping" in section
    assert "not a universal policy ranking or utility claim" in section


def test_config_tokens_from_config_yaml(generated_fast_bundle: Path) -> None:
    variables = generate_variables(
        PROJECT_ROOT, require_analysis_outputs=True, output_root=generated_fast_bundle
    )
    config = yaml.safe_load((PROJECT_ROOT / "manuscript" / "config.yaml").read_text())
    assert variables["CONFIG_M"] == str(config["experiment"]["m"])
    assert variables["QUALITY_TEST_COUNT"] == "PENDING"
    assert variables["QUALITY_BRANCH_COVERAGE"] == "PENDING"


def test_missing_results_raises(tmp_path):
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript" / "config.yaml").write_text(
        "experiment: {m: 200, pi0: 0.8}\n"
    )
    with pytest.raises(FileNotFoundError):
        generate_variables(tmp_path, require_analysis_outputs=True)


def test_draft_mode_falls_back_to_na(tmp_path):
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript" / "config.yaml").write_text(
        "experiment: {m: 200, pi0: 0.8, effect: 3.0}\n"
    )
    variables = generate_variables(tmp_path, require_analysis_outputs=False)
    assert variables["RESULT_COMPARISON_ROWS"] == _MISSING
    assert variables["CONFIG_M"] == "200"
    assert variables["CONFIG_PI0"] == "0.8"
    assert variables["CONFIG_EFFECT"] == "3.0"


def test_draft_mode_empty_config(tmp_path):
    variables = generate_variables(tmp_path, require_analysis_outputs=False)
    assert variables["CONFIG_M"] == _MISSING
    assert variables["CONFIG_PI0"] == _MISSING


def test_variable_helpers_fail_closed_and_preserve_explicit_empty_boundaries(
    tmp_path: Path,
) -> None:
    """Helper fallbacks are real-file based and never invent publication evidence."""
    assert _load_config(tmp_path) == {}
    assert _load_results(tmp_path) == {}
    assert _load_quality(tmp_path) == {}

    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    (manuscript / "config.yaml").write_text("experiment: {m: 7}\n", encoding="utf-8")
    payload_root = tmp_path / "isolated"
    payload_root.mkdir()
    (payload_root / "results.json").write_text('{"observed": true}', encoding="utf-8")
    (payload_root / "quality.json").write_text('{"test_count": 7}', encoding="utf-8")
    assert _load_config(tmp_path) == {"experiment": {"m": 7}}
    assert _load_results(tmp_path, Path("isolated")) == {"observed": True}
    assert _load_quality(tmp_path, Path("isolated")) == {"test_count": 7}

    assert _probability_or_not_estimable(None, 3) == "not estimable"
    assert _probability_or_not_estimable(0.125, 3) == "0.125"
    for invalid in (True, "0.125", float("inf")):
        with pytest.raises(ValueError, match="finite number"):
            _probability_or_not_estimable(invalid, 3)

    assert "intended public source repository" in _publication_repository_statement({})
    with pytest.raises(ValueError, match="owner/repository"):
        _publication_repository_statement(
            {"publication": {"planned_github_repository": "not-a-repository"}}
        )
    with pytest.raises(ValueError, match="canonical GitHub URL"):
        _publication_repository_statement(
            {
                "publication": {
                    "planned_github_repository": "ActiveInferenceInstitute/active_inference_power",
                    "planned_repository_url": "https://example.invalid/not-the-planned-repository",
                    "doi_status": "to be assigned at publication",
                }
            }
        )
    with pytest.raises(ValueError, match="doi_status"):
        _publication_repository_statement(
            {
                "publication": {
                    "planned_github_repository": "ActiveInferenceInstitute/active_inference_power",
                    "planned_repository_url": "https://github.com/ActiveInferenceInstitute/active_inference_power",
                    "doi_status": "draft",
                }
            }
        )
    draft_publication = {
        "publication": {
            "status": "draft",
            "planned_github_repository": "ActiveInferenceInstitute/active_inference_power",
            "planned_repository_url": "https://github.com/ActiveInferenceInstitute/active_inference_power",
            "doi_status": "to be assigned at publication",
        }
    }
    assert "ActiveInferenceInstitute/active_inference_power" in (
        _publication_repository_statement(draft_publication)
    )

    assert _comparison_rows({}) == ""
    assert _cert_rows({}) == ""
    assert _interval_text(None) == "N/A"
    assert _action_loop_rows({}) == ""
    assert _split_confirmation_rows({}) == ""
    assert _learned_policy_rows({}) == ""
    assert _paired_action_rows({}) == ""
    assert _reader_evidence_status({"profile": "campaign"}, {"complete": False}, {}) == (
        "campaign review-only non-release"
    )
    assert _reader_evidence_status(
        {"profile": "campaign"},
        {"complete": True, "certificates_pass": True},
        {"profile": "campaign", "status": "passed"},
    ) == "campaign-complete non-release"
    assert _reader_evidence_status(
        {"profile": "release"},
        {"certificates_pass": True},
        {"profile": "release", "status": "passed"},
    ) == "release candidate bound to the current source"


def test_save_variables_roundtrip(tmp_path: Path, generated_fast_bundle: Path) -> None:
    variables = generate_variables(
        PROJECT_ROOT, require_analysis_outputs=True, output_root=generated_fast_bundle
    )
    out = tmp_path / "sub" / "manuscript_variables.json"
    save_variables(variables, out)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded == variables


def test_variables_can_hydrate_an_isolated_output_root(
    generated_fast_bundle: Path,
) -> None:
    """A review output root must not silently fall back to the default tree."""
    variables = generate_variables(
        PROJECT_ROOT,
        require_analysis_outputs=True,
        output_root=generated_fast_bundle,
    )
    assert variables["RESULT_BH_FDR"] != _MISSING
    assert variables["QUALITY_TEST_COUNT"] == "PENDING"
