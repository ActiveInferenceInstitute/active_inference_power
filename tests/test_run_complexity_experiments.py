"""Real-file tests for the isolated local computational-timing diagnostic."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from matplotlib import pyplot as plt
import pytest

from active_inference_power.computational_complexity import (
    ComputationalComplexityPlan,
    TIMING_PROFILE_FROZEN_PLAN_SOURCE,
    summarize_timing_samples,
    timing_plan_digest,
    timing_profile_digest,
    validate_timing_profile,
)
from scripts.artifacts import sha256_file
from scripts.run_complexity_experiments import (
    CONFIG_PATH,
    ROOT,
    TIMING_FIGURE_FILENAME,
    TIMING_INDEX_FILENAME,
    TIMING_PROFILE_FILENAME,
    TIMING_STATUS,
    _assert_stable_timing_witnesses,
    _frozen_plan,
    _parser,
    _source_hashes,
    _validate_frozen_timing_profile,
    _validate_output_root,
    main,
    run_complexity_experiments,
)


def _small_real_plan() -> ComputationalComplexityPlan:
    """Keep the test compact while exercising every actual timing kernel family."""
    return ComputationalComplexityPlan(
        hypothesis_grid=(8,),
        stream_horizon_grid=(4,),
        action_horizon_grid=(2,),
        timing_warmups=0,
        timing_repetitions=7,
        correction_methods=("bonferroni",),
        online_methods=("e_lond",),
        action_policies=("fixed_horizon",),
    )


def test_cli_plan_is_loaded_only_from_the_frozen_configuration() -> None:
    plan, config_hash = _frozen_plan()

    assert CONFIG_PATH.is_file()
    assert config_hash == sha256_file(CONFIG_PATH)
    assert plan.hypothesis_grid == (25, 50, 100, 200, 400, 800)
    assert plan.stream_horizon_grid == (16, 32, 64, 128, 256, 512)
    assert plan.action_horizon_grid == (2, 4, 8, 12, 16, 32)
    assert plan.timing_warmups == 2
    assert plan.timing_repetitions == 7


def test_local_timing_bundle_is_atomic_hash_bound_and_non_promotional(
    tmp_path: Path,
) -> None:
    target = tmp_path / "complexity-local-diagnostic"
    receipt = run_complexity_experiments(target, _small_real_plan())

    profile_path = target / TIMING_PROFILE_FILENAME
    figure_path = target / TIMING_FIGURE_FILENAME
    index_path = target / TIMING_INDEX_FILENAME
    manifest_path = target / "manifest.json"
    assert target.is_dir()
    assert {path.name for path in target.iterdir()} == {
        TIMING_PROFILE_FILENAME,
        TIMING_FIGURE_FILENAME,
        TIMING_INDEX_FILENAME,
        "manifest.json",
    }
    assert receipt["status"] == TIMING_STATUS
    assert Path(receipt["profile_path"]) == profile_path
    assert Path(receipt["figure_path"]) == figure_path
    assert Path(receipt["index_path"]) == index_path
    assert Path(receipt["manifest_path"]) == manifest_path
    assert receipt["profile_sha256"] == sha256_file(profile_path)
    assert receipt["figure_sha256"] == sha256_file(figure_path)
    assert receipt["manifest_sha256"] == sha256_file(manifest_path)
    assert not list(
        tmp_path.glob(".complexity-local-diagnostic.complexity-experiments-*")
    )

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    validate_timing_profile(profile)
    assert profile["status"] == TIMING_STATUS
    assert profile["plan"] == _small_real_plan().to_mapping()
    assert profile["plan_sha256"] == timing_plan_digest(_small_real_plan())
    assert profile["plan_source"] == "explicit_programmatic_real_diagnostic_plan"
    assert profile["plan_source_sha256"] is None
    assert "not a release" in profile["non_claim"]
    assert "hostname" in profile["environment"]["redaction"]
    assert "hostname" not in profile["environment"]
    assert set(profile["source_hashes"]) >= {
        "orchestrator",
        "computational_complexity",
        "corrections",
        "online_fdr",
        "sequential",
        "action_loop",
        "scenario",
        "pymdp_adapter",
    }
    assert "dependency:src/active_inference_power/_types.py" in profile["source_hashes"]
    assert (
        "dependency:src/active_inference_power/action_statistics.py"
        in profile["source_hashes"]
    )
    assert len(profile["rows"]) == 7
    assert {(row["kernel"], row["condition"]) for row in profile["rows"]} >= {
        ("correction:bonferroni", "deterministic_pvalue_vector"),
        ("online:e_lond", "sparse_or_no_rejection_history"),
        ("online:e_lond", "dense_rejection_history"),
        ("sequential:binary_e_process", "likelihood_ratio_evidence_path"),
        (
            "sequential:bernoulli_confidence_sequence",
            "time_uniform_confidence_sequence_inversion",
        ),
        (
            "sequential:fixed_horizon_binomial_pvalue",
            "fixed_horizon_only_invalid_for_optional_stopping",
        ),
        (
            "action_loop:fixed_horizon",
            "real_pymdp_action_observation_belief_evidence_trace",
        ),
    }
    for row in profile["rows"]:
        assert len(row["wall_time_ns_samples"]) == 7
        assert len(row["process_time_ns_samples"]) == 7
        assert row["wall_time_ns_summary"]["n"] == 7
        assert row["process_time_ns_summary"]["n"] == 7

    image = plt.imread(figure_path)
    assert image.shape[0] >= 1_000
    assert image.shape[1] >= 1_500
    index_html = index_path.read_text(encoding="utf-8")
    assert profile["non_claim"] in index_html
    assert "invalid under optional stopping" in index_html

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == TIMING_STATUS
    assert manifest["profile_result_sha256"] == profile["result_sha256"]
    assert manifest["profile_file_sha256"] == sha256_file(profile_path)
    assert manifest["plan_sha256"] == profile["plan_sha256"]
    assert manifest["claim_boundary"] == profile["non_claim"]
    assert f'alt="{manifest["figure_alt"]}"' in index_html
    for phrase in (
        "local timing summaries rather than MCSEs",
        "cross-machine benchmarks, or release evidence",
        "simulated resource cost",
        "likelihood-ratio e-process evidence",
        "online alpha-wealth accounting",
    ):
        assert phrase in manifest["figure_alt"]
    assert set(manifest["artifacts"]) == {
        "timing_profile",
        "timing_figure",
        "index_html",
    }
    for record in manifest["artifacts"].values():
        assert record["root"] == "output"
        artifact_path = target / record["path"]
        assert artifact_path.is_file()
        assert record["sha256"] == sha256_file(artifact_path)
        assert record["bytes"] == artifact_path.stat().st_size


def test_timing_figure_footer_typesets_runtime_and_evidence_quantities() -> None:
    """Keep local timing annotations reader-facing and scientifically separate."""
    source = (ROOT / "scripts" / "run_complexity_experiments.py").read_text(
        encoding="utf-8"
    )
    footer_start = source.index('"Points are medians of repeated local calls')
    footer_end = source.index("figure.tight_layout", footer_start)
    footer = source[footer_start:footer_end]

    for typeset_quantity in (
        r"$C$",
        r"$E_t$",
        r"$W_t^{\alpha}$",
    ):
        assert typeset_quantity in footer
    for boundary in (
        "not Monte Carlo standard errors (MCSEs)",
        "neither a cross-machine benchmark nor release evidence",
        "Separate quantities: simulated resource cost",
    ):
        assert boundary in footer


def test_timing_profile_rejects_missing_or_non_promotional_provenance_fields(
    tmp_path: Path,
) -> None:
    """The isolated-lane validator cannot silently accept a weaker receipt."""
    target = tmp_path / "complexity-provenance-contract"
    run_complexity_experiments(target, _small_real_plan())
    profile = json.loads((target / TIMING_PROFILE_FILENAME).read_text(encoding="utf-8"))

    missing_source_witness = copy.deepcopy(profile)
    missing_source_witness.pop("source_hashes")
    missing_source_witness["result_sha256"] = timing_profile_digest(
        missing_source_witness
    )
    with pytest.raises(ValueError, match="source_hashes"):
        validate_timing_profile(missing_source_witness)

    promotable = copy.deepcopy(profile)
    promotable["status"] = "release_complete"
    promotable["result_sha256"] = timing_profile_digest(promotable)
    with pytest.raises(ValueError, match="non-promotional"):
        validate_timing_profile(promotable)

    insufficient_redaction = copy.deepcopy(profile)
    insufficient_redaction["environment"]["redaction"] = "identity omitted"
    insufficient_redaction["result_sha256"] = timing_profile_digest(
        insufficient_redaction
    )
    with pytest.raises(ValueError, match="redaction"):
        validate_timing_profile(insufficient_redaction)

    leaked_identity = copy.deepcopy(profile)
    leaked_identity["hostname"] = "not-allowed"
    leaked_identity["result_sha256"] = timing_profile_digest(leaked_identity)
    with pytest.raises(ValueError, match="host-identity"):
        validate_timing_profile(leaked_identity)

    unreviewed_extension = copy.deepcopy(profile)
    unreviewed_extension["unreviewed_runtime_metric"] = "not-allowed"
    unreviewed_extension["result_sha256"] = timing_profile_digest(unreviewed_extension)
    with pytest.raises(ValueError, match="unexpected keys"):
        validate_timing_profile(unreviewed_extension)

    unreviewed_row_extension = copy.deepcopy(profile)
    unreviewed_row_extension["rows"][0]["machine_serial"] = "not-allowed"
    unreviewed_row_extension["result_sha256"] = timing_profile_digest(
        unreviewed_row_extension
    )
    with pytest.raises(ValueError, match="unexpected keys"):
        validate_timing_profile(unreviewed_row_extension)

    incomplete_plan_coverage = copy.deepcopy(profile)
    incomplete_plan_coverage["rows"].pop()
    incomplete_plan_coverage["result_sha256"] = timing_profile_digest(
        incomplete_plan_coverage
    )
    with pytest.raises(ValueError, match="exactly cover"):
        validate_timing_profile(incomplete_plan_coverage)

    unplanned_kernel_case = copy.deepcopy(profile)
    unplanned_kernel_case["rows"][0]["condition"] = "unplanned_case"
    unplanned_kernel_case["result_sha256"] = timing_profile_digest(
        unplanned_kernel_case
    )
    with pytest.raises(ValueError, match="exactly cover"):
        validate_timing_profile(unplanned_kernel_case)

    unsupported_contract = copy.deepcopy(profile)
    unsupported_contract["contract_version"] = "not-supported"
    unsupported_contract["result_sha256"] = timing_profile_digest(unsupported_contract)
    with pytest.raises(ValueError, match="contract version"):
        validate_timing_profile(unsupported_contract)

    promotable_kind = copy.deepcopy(profile)
    promotable_kind["kind"] = "benchmark"
    promotable_kind["result_sha256"] = timing_profile_digest(promotable_kind)
    with pytest.raises(ValueError, match="research-only"):
        validate_timing_profile(promotable_kind)

    frozen_source = copy.deepcopy(profile)
    frozen_source["plan_source"] = TIMING_PROFILE_FROZEN_PLAN_SOURCE
    frozen_source["plan_source_sha256"] = sha256_file(CONFIG_PATH)
    frozen_source["result_sha256"] = timing_profile_digest(frozen_source)
    with pytest.raises(ValueError, match="requires the expected frozen plan"):
        validate_timing_profile(frozen_source)
    configured_plan, configured_digest = _frozen_plan()
    with pytest.raises(
        ValueError,
        match="plan_source_sha256 does not match the expected frozen configuration",
    ):
        validate_timing_profile(
            frozen_source,
            expected_frozen_plan=configured_plan,
            expected_plan_source_sha256="0" * 64,
        )
    with pytest.raises(
        ValueError,
        match="timing profile plan does not match the expected frozen configuration plan",
    ):
        validate_timing_profile(
            frozen_source,
            expected_frozen_plan=configured_plan,
            expected_plan_source_sha256=configured_digest,
        )
    with pytest.raises(
        ValueError,
        match="timing profile plan does not match the expected frozen configuration plan",
    ):
        _validate_frozen_timing_profile(frozen_source)

    programmatic_source_with_digest = copy.deepcopy(profile)
    programmatic_source_with_digest["plan_source_sha256"] = sha256_file(CONFIG_PATH)
    programmatic_source_with_digest["result_sha256"] = timing_profile_digest(
        programmatic_source_with_digest
    )
    with pytest.raises(ValueError, match="must be null"):
        validate_timing_profile(programmatic_source_with_digest)

    unknown_plan_source = copy.deepcopy(profile)
    unknown_plan_source["plan_source"] = "unreviewed_plan_source"
    unknown_plan_source["result_sha256"] = timing_profile_digest(unknown_plan_source)
    with pytest.raises(ValueError, match="not an allowed"):
        validate_timing_profile(unknown_plan_source)

    malformed_source_digest = copy.deepcopy(profile)
    malformed_source_digest["source_hashes"]["online_fdr"] = "not-a-sha256"
    malformed_source_digest["result_sha256"] = timing_profile_digest(
        malformed_source_digest
    )
    with pytest.raises(ValueError, match="64-character SHA-256"):
        validate_timing_profile(malformed_source_digest)

    missing_direct_source = copy.deepcopy(profile)
    missing_direct_source["source_hashes"].pop("orchestrator")
    missing_direct_source["result_sha256"] = timing_profile_digest(
        missing_direct_source
    )
    with pytest.raises(
        ValueError, match="source_hashes is missing keys: .*orchestrator"
    ):
        validate_timing_profile(missing_direct_source)

    unsafe_transitive_source = copy.deepcopy(profile)
    unsafe_transitive_source["source_hashes"]["dependency:../unreviewed.py"] = "0" * 64
    unsafe_transitive_source["result_sha256"] = timing_profile_digest(
        unsafe_transitive_source
    )
    with pytest.raises(ValueError, match="source_hashes has unexpected keys"):
        validate_timing_profile(unsafe_transitive_source)

    missing_closure_dependency = copy.deepcopy(profile)
    missing_closure_dependency["source_hashes"].pop(
        "dependency:src/active_inference_power/_types.py"
    )
    missing_closure_dependency["result_sha256"] = timing_profile_digest(
        missing_closure_dependency
    )
    with pytest.raises(
        ValueError, match="do not exactly match the expected source closure"
    ):
        validate_timing_profile(
            missing_closure_dependency,
            expected_source_hashes=_source_hashes(),
        )

    substituted_closure_digest = copy.deepcopy(profile)
    substituted_closure_digest["source_hashes"]["online_fdr"] = "0" * 64
    substituted_closure_digest["result_sha256"] = timing_profile_digest(
        substituted_closure_digest
    )
    with pytest.raises(
        ValueError, match="do not exactly match the expected source closure"
    ):
        validate_timing_profile(
            substituted_closure_digest,
            expected_source_hashes=_source_hashes(),
        )

    altered_plan_digest = copy.deepcopy(profile)
    altered_plan_digest["plan_sha256"] = "0" * 64
    altered_plan_digest["result_sha256"] = timing_profile_digest(altered_plan_digest)
    with pytest.raises(ValueError, match="plan_sha256"):
        validate_timing_profile(altered_plan_digest)

    changed_protocol = copy.deepcopy(profile)
    changed_protocol["measurement_protocol"]["warmups"] = "unreviewed alteration"
    changed_protocol["result_sha256"] = timing_profile_digest(changed_protocol)
    with pytest.raises(ValueError, match="measurement_protocol"):
        validate_timing_profile(changed_protocol)

    no_rows = copy.deepcopy(profile)
    no_rows["rows"] = []
    no_rows["result_sha256"] = timing_profile_digest(no_rows)
    with pytest.raises(ValueError, match="rows must be non-empty"):
        validate_timing_profile(no_rows)

    duplicate_row = copy.deepcopy(profile)
    duplicate_row["rows"].append(copy.deepcopy(duplicate_row["rows"][0]))
    duplicate_row["result_sha256"] = timing_profile_digest(duplicate_row)
    with pytest.raises(ValueError, match="unique kernel"):
        validate_timing_profile(duplicate_row)

    unexpected_warmups = copy.deepcopy(profile)
    unexpected_warmups["rows"][0]["warmups"] = 1
    unexpected_warmups["result_sha256"] = timing_profile_digest(unexpected_warmups)
    with pytest.raises(ValueError, match="warmups must match"):
        validate_timing_profile(unexpected_warmups)

    truncated_samples = copy.deepcopy(profile)
    truncated_samples["rows"][0]["wall_time_ns_samples"].pop()
    truncated_samples["result_sha256"] = timing_profile_digest(truncated_samples)
    with pytest.raises(ValueError, match="sample count"):
        validate_timing_profile(truncated_samples)

    altered_summary = copy.deepcopy(profile)
    altered_summary["rows"][0]["process_time_ns_summary"]["median_ns"] += 1.0
    altered_summary["result_sha256"] = timing_profile_digest(altered_summary)
    with pytest.raises(ValueError, match="does not match raw samples"):
        validate_timing_profile(altered_summary)

    boolean_sample = copy.deepcopy(profile)
    boolean_sample["rows"][0]["wall_time_ns_samples"][0] = True
    boolean_sample["result_sha256"] = timing_profile_digest(boolean_sample)
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_timing_profile(boolean_sample)

    fractional_sample = copy.deepcopy(profile)
    fractional_sample["rows"][0]["process_time_ns_samples"][0] = 1.5
    fractional_sample["result_sha256"] = timing_profile_digest(fractional_sample)
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_timing_profile(fractional_sample)

    invalid_digest = copy.deepcopy(profile)
    invalid_digest["result_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="result_sha256"):
        validate_timing_profile(invalid_digest)

    for invalid_samples in ([], [-1], [1.0, float("nan")]):
        with pytest.raises(ValueError, match="finite non-negative"):
            summarize_timing_samples(invalid_samples)


def test_timing_profile_matrix_requires_exact_row_keys_and_plan_coverage(
    tmp_path: Path,
) -> None:
    """A real small receipt must not hide a row field or an undeclared grid case."""
    target = tmp_path / "complexity-matrix-contract"
    run_complexity_experiments(target, _small_real_plan())
    profile = json.loads((target / TIMING_PROFILE_FILENAME).read_text(encoding="utf-8"))
    validate_timing_profile(profile)

    missing_row_key = copy.deepcopy(profile)
    missing_row_key["rows"][0].pop("kernel")
    missing_row_key["result_sha256"] = timing_profile_digest(missing_row_key)
    with pytest.raises(ValueError, match="missing keys"):
        validate_timing_profile(missing_row_key)

    expanded_plan = copy.deepcopy(profile)
    expanded_plan["plan"]["correction_methods"].append("benjamini_hochberg")
    expanded_plan["plan_sha256"] = timing_plan_digest(expanded_plan["plan"])
    expanded_plan["result_sha256"] = timing_profile_digest(expanded_plan)
    with pytest.raises(ValueError, match="exactly cover"):
        validate_timing_profile(expanded_plan)


def test_timing_witnesses_reject_mid_run_source_or_frozen_plan_drift() -> None:
    """The receipt can only describe inputs that remained stable while measured."""
    source_hashes = _source_hashes()
    frozen_plan, frozen_digest = _frozen_plan()
    frozen = (frozen_plan, frozen_digest)
    _assert_stable_timing_witnesses(
        source_hashes,
        source_hashes,
        initial_frozen=frozen,
        final_frozen=frozen,
    )

    changed_source = dict(source_hashes)
    changed_source["online_fdr"] = "0" * 64
    with pytest.raises(RuntimeError, match="source witness changed"):
        _assert_stable_timing_witnesses(
            source_hashes,
            changed_source,
            initial_frozen=frozen,
            final_frozen=frozen,
        )

    with pytest.raises(RuntimeError, match="frozen-plan provenance changed"):
        _assert_stable_timing_witnesses(
            source_hashes,
            source_hashes,
            initial_frozen=frozen,
            final_frozen=(frozen_plan, "0" * 64),
        )


def test_timing_runner_preserves_existing_or_nonisolated_roots(tmp_path: Path) -> None:
    plan = _small_real_plan()
    with pytest.raises(ValueError, match="not the project output root"):
        run_complexity_experiments(ROOT / "output", plan)
    with pytest.raises(ValueError, match="not the project output root"):
        _validate_output_root(ROOT / "output" / "new-local-complexity-diagnostic")
    target = tmp_path / "preserve-existing"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        run_complexity_experiments(target, plan)
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    with pytest.raises(ValueError, match="concrete directory"):
        _validate_output_root(Path("/"))


def test_cli_requires_explicit_root_and_exposes_no_timing_tuning_flags() -> None:
    with pytest.raises(SystemExit):
        main([])
    help_text = _parser().format_help()
    assert "--output-root" in help_text
    assert "--warmups" not in help_text
    assert "--repetitions" not in help_text
    assert "--seed" not in help_text
    assert "--config" not in help_text
