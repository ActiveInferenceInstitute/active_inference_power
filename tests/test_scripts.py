"""Real configuration-boundary tests for the result-generation orchestrator."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.generate_results import (
    REFERENCE_COMPARISON_METHODS,
    _artifact,
    _benchmark_artifact_records,
    _certificate_gate_exit_status,
    _checkpoint_dependency_hashes,
    _expected_checkpoint_dependency_hashes,
    _invalidate_existing_manifest,
    _load_config,
    _require_generated_complexity_plan,
    _sample_size_accounting,
    _resolve_experiment_profile,
    run,
)
from scripts.artifacts import (
    artifact_record,
    atomic_write_json,
    refresh_renderer_artifact_records,
    sha256_file,
)
from scripts.finalize_release import _validate_manuscript, _validate_release_state, finalize_release
from scripts.release_provenance import (
    campaign_lane_predecessor_artifacts,
    required_campaign_artifacts,
    required_release_artifacts,
)
from scripts.regenerate_figures import _invalidate_release_completion
from scripts.run_ledger import RunLedger
from scripts.z_generate_manuscript_variables import (
    _hydration_pending,
    _requires_ledger_artifacts,
)
from active_inference_power.schema import validate_run_log
from active_inference_power.certificates import CERTIFICATE_NAMES


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_new_generation_atomically_revokes_prior_completion(tmp_path: Path) -> None:
    """A new run must not leave an older complete manifest authoritative."""
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": "2.0",
        "run_id": "prior-run",
        "complete": True,
        "config_sha256": "a" * 64,
        "artifacts": {
            "results": {
                "root": "output",
                "path": "results.json",
                "sha256": "b" * 64,
                "bytes": 1,
                "role": "deterministic_results",
            },
            "rendered_html": {
                "root": "output",
                "path": "web/index.html",
                "sha256": "c" * 64,
                "bytes": 1,
                "role": "rendered_html",
            },
            "web:index.html": {
                "root": "output",
                "path": "web/index.html",
                "sha256": "c" * 64,
                "bytes": 1,
                "role": "rendered_html",
            },
            "release_review": {
                "root": "output",
                "path": "review/release_review.json",
                "sha256": "d" * 64,
                "bytes": 1,
                "role": "release_review_receipt",
            },
            "renderer_provenance": {
                "root": "output",
                "path": "review/renderer_provenance.json",
                "sha256": "e" * 64,
                "bytes": 1,
                "role": "renderer_provenance_receipt",
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    pending = _invalidate_existing_manifest(tmp_path)

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert pending == [
        "result_generation",
        "quality.json",
        "manuscript_hydration",
        "publication_render",
        "release_finalization",
    ]
    assert persisted["complete"] is False
    assert persisted["validation_pending"] == pending
    assert set(persisted["artifacts"]) == {"results"}


def test_visual_refresh_invalidates_completion_and_stale_renderer_records() -> None:
    manifest: dict[str, Any] = {
        "complete": True,
        "artifacts": {
            "rendered_pdf": {"role": "rendered_pdf"},
            "web:index.html": {"role": "rendered_html"},
            "results": {"role": "deterministic_results"},
        },
    }
    pending = _invalidate_release_completion(manifest)
    assert manifest["complete"] is False
    assert pending == ["manuscript_hydration", "publication_render", "release_finalization"]
    assert set(manifest["artifacts"]) == {"results"}


def test_release_finalizer_refreshes_only_renderer_owned_hashes(tmp_path: Path) -> None:
    web = tmp_path / "web" / "index.html"
    web.parent.mkdir()
    web.write_text("old", encoding="utf-8")
    record = artifact_record(web, tmp_path, "rendered_html", root_label="output")
    web.write_text("new renderer output", encoding="utf-8")
    manifest = {
        "artifacts": {
            "web:index.html": record,
            "results": {**record, "path": "results.json", "role": "deterministic_results"},
        }
    }
    refresh_renderer_artifact_records(manifest, tmp_path, PROJECT_ROOT)
    refreshed = manifest["artifacts"]["web:index.html"]
    assert refreshed["sha256"] == sha256_file(web)
    assert refreshed["bytes"] == web.stat().st_size
    assert manifest["artifacts"]["results"] == {**record, "path": "results.json", "role": "deterministic_results"}


def test_manuscript_hydration_never_has_completion_authority() -> None:
    assert _hydration_pending(
        run_status="passed", certificates_pass=True, allow_draft=False, unresolved=[]
    ) == ["publication_render", "release_finalization"]
    assert _hydration_pending(
        run_status="failed", certificates_pass=False, allow_draft=True, unresolved=["x.md"]
    ) == [
        "draft_hydration",
        "resolved_manuscript",
        "certificates",
        "run_log",
        "publication_render",
        "release_finalization",
    ]


def test_release_finalizer_has_no_unrendered_completion_path() -> None:
    with pytest.raises(ValueError, match="only completes rendered publication bundles"):
        finalize_release(require_render=False)


def test_manuscript_witnesses_must_bind_the_exact_release_provenance(
    tmp_path: Path,
) -> None:
    variables = tmp_path / "data" / "manuscript_variables.json"
    resolved = tmp_path / "manuscript" / "00_abstract.md"
    variables.parent.mkdir()
    resolved.parent.mkdir()
    variables.write_text(
        json.dumps(
            {
                "RESULT_CONFIGURATION_HASH": "a" * 64,
                "RESULT_CLAIM_LEDGER_HASH": "b" * 64,
                "RESULT_RENDER_INPUT_HASH": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    resolved.write_text("Resolved source.\n", encoding="utf-8")
    assert _validate_manuscript(
        tmp_path,
        expected_witnesses={"config_sha256": "a" * 64, "claim_ledger_sha256": "b" * 64},
    ) == {
        "config_sha256": "a" * 64,
        "claim_ledger_sha256": "b" * 64,
        "render_input_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="does not match the release provenance"):
        _validate_manuscript(
            tmp_path,
            expected_witnesses={"config_sha256": "c" * 64, "claim_ledger_sha256": "b" * 64},
        )


def _write_config(tmp_path: Path, mutate) -> Path:
    source = yaml.safe_load(
        (PROJECT_ROOT / "manuscript" / "config.yaml").read_text(encoding="utf-8")
    )
    mutate(source["experiment"])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return path


def test_load_config_accepts_canonical_experiment() -> None:
    config = _load_config(PROJECT_ROOT / "manuscript" / "config.yaml")
    assert config["experiment"]["m"] == 200
    assert tuple(config["experiment"]["methods"]) == REFERENCE_COMPARISON_METHODS
    assert config["experiment"]["n_reps"] >= 5000
    assert config["experiment"]["target_mc_half_width"] == 0.02
    assert config["experiment"]["campaign"]["name"] == "expanded_conditional_power_campaign_v1"
    assert len(config["experiment"]["campaign"]["effect_grid"]) == 13
    assert len(config["experiment"]["campaign"]["dependence_rhos"]) == 7


def test_isolated_certificate_policy_config_is_explicit_and_preserved(
    tmp_path: Path,
) -> None:
    """A preregistered lane can classify one named certificate as diagnostic."""
    def mutate(experiment: dict[str, Any]) -> None:
        experiment["certificate_gate_policy"] = {
            "terminal": [
                name for name in CERTIFICATE_NAMES if name != "storey_power_gain"
            ],
            "diagnostic": ["storey_power_gain"],
        }

    path = _write_config(tmp_path, mutate)
    config = _load_config(path)
    policy = config["experiment"]["certificate_gate_policy"]
    assert policy["diagnostic"] == ["storey_power_gain"]
    assert set(policy["terminal"]).union(policy["diagnostic"]) == set(
        CERTIFICATE_NAMES
    )


def test_campaign_profile_resolves_expanded_design_without_mutating_source() -> None:
    experiment = _load_config(PROJECT_ROOT / "manuscript" / "config.yaml")["experiment"]
    resolved = _resolve_experiment_profile(experiment, "campaign")
    assert resolved["n_reps"] == 15000
    assert resolved["certificate_n_reps"] == 7500
    assert resolved["active_inference_reps"] == 1500
    assert resolved["calibration_n_reps"] == 7500
    assert resolved["target_mc_half_width"] == 0.015
    assert len(resolved["effect_grid"]) == 13
    assert len(resolved["dependence_rhos"]) == 7
    assert resolved["action_loop"]["reliability_grid"] == [0.55, 0.65, 0.8, 0.92, 0.98]
    assert resolved["profile_design"]["source"] == "experiment.campaign"
    assert "profile_design" not in experiment
    assert _resolve_experiment_profile(experiment, "release")["n_reps"] == 10000


def test_action_loop_sample_size_accounting_matches_declared_budget() -> None:
    experiment = _load_config(PROJECT_ROOT / "manuscript" / "config.yaml")["experiment"]
    resolved = _resolve_experiment_profile(experiment, "release")
    accounting = _sample_size_accounting(
        resolved,
        n_reps=10000,
        active_reps=1000,
        certificate_reps=5000,
        calibration={"n_reps": 5000, "n_pvalues": 5000},
        confidence=0.95,
    )["blocks"]["conditional_power"]
    assert accounting["paired_policy_replications"] == 1000 * len(
        resolved["action_loop"]["reliability_grid"]
    )
    assert accounting["chronological_selection_replications"] == 1000
    assert accounting["future_selection_contrast_replications"] == 1000
    assert accounting["split_confirmation_replications"] == 1000
    assert accounting["split_confirmation_streams"] == 2000
    assert accounting["learned_policy_evaluation_replications"] == 1000


def test_campaign_profile_rejects_missing_campaign() -> None:
    experiment = _load_config(PROJECT_ROOT / "manuscript" / "config.yaml")["experiment"]
    experiment.pop("campaign")
    with pytest.raises(ValueError, match="requires experiment.campaign"):
        _resolve_experiment_profile(experiment, "campaign")


@pytest.mark.parametrize(
    "controls, message",
    [
        ({"checkpoint_every": True}, "checkpoint_every"),
        ({"checkpoint_every": -1}, "checkpoint_every"),
        ({"checkpoint_every": "one"}, "checkpoint_every"),
        ({"seed_override": True}, "seed_override"),
        ({"seed_override": -1}, "seed_override"),
        ({"seed_override": 1.5}, "seed_override"),
    ],
)
def test_run_rejects_invalid_control_values_before_mutating_output(
    tmp_path: Path, controls: dict[str, Any], message: str
) -> None:
    sentinel = tmp_path / "manifest.json"
    sentinel.write_text('{"retained": "receipt"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        run(
            PROJECT_ROOT / "manuscript" / "config.yaml",
            profile="fast",
            workers=1,
            output_root=tmp_path,
            **controls,
        )
    assert sentinel.read_text(encoding="utf-8") == '{"retained": "receipt"}\n'


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda experiment: experiment.update(n_reps=1.5), "n_reps"),
        (lambda experiment: experiment.update(alpha=1.0), "alpha"),
        (lambda experiment: experiment.update(methods=["bh"]), "methods"),
        (
            lambda experiment: experiment["active_inference"]["regimes"][
                "weak_evidence"
            ].update(reliability=1.0),
            "reliability",
        ),
        (lambda experiment: experiment.update(target_mc_half_width=1.0), "target_mc_half_width"),
    ],
)
def test_load_config_rejects_invalid_experiment_boundary(
    tmp_path: Path, mutate, message: str
) -> None:
    path = _write_config(tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        _load_config(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda experiment: experiment.update(
            methods=list(reversed(experiment["methods"]))
        ),
        lambda experiment: experiment.update(
            methods=[*experiment["methods"], "bonferroni"]
        ),
        lambda experiment: experiment.update(
            methods=[
                "bh" if method == "benjamini_hochberg" else method
                for method in experiment["methods"]
            ]
        ),
        lambda experiment: experiment.update(
            methods=[
                "storey_qvalue_split_confirmation"
                if method == "storey_qvalue"
                else method
                for method in experiment["methods"]
            ]
        ),
    ],
)
def test_load_config_rejects_nonreference_comparison_method_contract(
    tmp_path: Path, mutate
) -> None:
    path = _write_config(tmp_path, mutate)
    with pytest.raises(ValueError, match="ordered canonical reference comparison"):
        _load_config(path)


def test_load_config_rejects_missing_required_key(tmp_path: Path) -> None:
    path = _write_config(tmp_path, lambda experiment: experiment.pop("effect_grid"))
    with pytest.raises(ValueError, match="effect_grid"):
        _load_config(path)


@pytest.mark.parametrize(
    "content, message",
    [
        ("- not\n- a mapping\n", "top-level mapping"),
        ("experiment: [\n", "not valid YAML"),
    ],
)
def test_load_config_rejects_nonmapping_or_malformed_yaml(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "invalid-config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _load_config(path)


def test_campaign_v2_config_declares_current_complexity_plan() -> None:
    """The refreshable v2 source carries the current deterministic work plan."""
    config_path = (
        PROJECT_ROOT
        / "data"
        / "research_lanes"
        / "conditional_power_campaign_v2.yaml"
    )
    config = _load_config(config_path)
    plan = _require_generated_complexity_plan(
        _resolve_experiment_profile(config["experiment"], "campaign"), config_path
    )

    assert plan.to_mapping() == {
        "hypothesis_grid": [25, 50, 100, 200, 400, 800],
        "stream_horizon_grid": [16, 32, 64, 128, 256, 512],
        "action_horizon_grid": [2, 4, 8, 12, 16, 32],
        "timing_warmups": 2,
        "timing_repetitions": 7,
        "correction_methods": ["bonferroni", "benjamini_hochberg"],
        "online_methods": ["lord_plus_plus", "saffron", "e_lond"],
        "action_policies": [
            "fixed_horizon",
            "information_gain",
            "cost_aware",
            "e_process",
        ],
    }


@pytest.mark.parametrize("profile", ("fast", "release", "campaign"))
def test_predecessor_campaign_config_without_complexity_plan_loads_but_cannot_emit_current_artifacts(
    tmp_path: Path, profile: str
) -> None:
    """A frozen predecessor source stays inspectable but cannot create current artifacts."""
    current_config_path = (
        PROJECT_ROOT
        / "data"
        / "research_lanes"
        / "conditional_power_campaign_v2.yaml"
    )
    payload = yaml.safe_load(current_config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    experiment = payload["experiment"]
    assert isinstance(experiment, dict)
    experiment.pop("computational_complexity")
    config_path = tmp_path / "archived_campaign_without_complexity_plan.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = _load_config(config_path)
    assert "computational_complexity" not in config["experiment"]

    output = tmp_path / profile
    output.mkdir()
    sentinel = output / "manifest.json"
    sentinel.write_text('{"retained": "campaign receipt"}\n', encoding="utf-8")

    with pytest.raises(
        ValueError, match="current generated artifacts require experiment.computational_complexity"
    ):
        run(config_path, profile=profile, workers=1, output_root=output)

    assert sentinel.read_text(encoding="utf-8") == '{"retained": "campaign receipt"}\n'
    assert not (output / "run_log.json").exists()


def test_generated_run_log_records_stage_accounting(
    generated_fast_bundle: Path,
) -> None:
    run_log = yaml.safe_load(
        (generated_fast_bundle / "run_log.json").read_text(encoding="utf-8")
    )
    assert run_log["status"] == "passed"
    assert run_log["certificates_pass"] is True
    assert {event["name"] for event in run_log["events"]} == {
        "certificates",
        "method_comparison",
        "paired_comparisons",
        "power_curve",
        "dependence_sweep",
        "active_inference",
        "sequential",
        "online_fdr",
        "agent_sampling",
        "conditional_power",
        "benchmarks",
        "null_pvalue_calibration",
    }
    assert all(event["elapsed_seconds"] >= 0.0 for event in run_log["events"])


def test_failed_certificate_is_nonzero_for_cli_and_preserves_failure_ledger(
    tmp_path: Path,
    generated_fast_bundle: Path,
    temporary_status_bundle,
) -> None:
    """A real failed certificate must not look successful to automation."""
    release_results = json.loads(
        (generated_fast_bundle / "results.json").read_text(encoding="utf-8")
    )
    campaign_results = temporary_status_bundle(profile="campaign")["results"]
    assert _certificate_gate_exit_status(release_results) == 0
    assert _certificate_gate_exit_status(campaign_results) == 1

    ledger = RunLedger(
        tmp_path / "run_log.json",
        config_sha256="a" * 64,
        profile="campaign",
        seed=0,
        workers=1,
    )
    ledger.fail(
        RuntimeError("one or more predeclared finite-simulation certificates failed"),
        certificates_pass=False,
    )
    recorded = json.loads((tmp_path / "run_log.json").read_text(encoding="utf-8"))
    validate_run_log(recorded)
    assert recorded["status"] == "failed"
    assert recorded["certificates_pass"] is False
    assert recorded["error"]["type"] == "RuntimeError"


def test_release_terminal_gate_allows_an_explicit_visible_diagnostic_failure(
    temporary_status_bundle,
) -> None:
    """Release finalization follows terminal names, not an implicit all-pass rule."""
    bundle = temporary_status_bundle(
        profile="release", diagnostic_storey_failure=True
    )
    assert (
        _validate_release_state(
            bundle["results"],
            bundle["manifest"],
            bundle["run_log"],
            {"quality_pass": True},
        )
        == "release"
    )


def test_checkpoint_dependency_hashes_change_when_scenario_contract_changes() -> None:
    value = {
        "scenario_contracts": [
            {
                "scenario_id": "reference",
                "model": {"prior": [0.5, 0.5]},
                "process": {"truth": [True, False]},
                "setting": {"alpha": 0.05},
            }
        ]
    }
    original = _checkpoint_dependency_hashes("conditional_power", value)
    changed = _checkpoint_dependency_hashes(
        "conditional_power",
        {**value, "scenario_contracts": [{**value["scenario_contracts"][0], "setting": {"alpha": 0.01}}]},
    )
    assert original != changed
    assert original["scenario_hashes"]
    changed_split_contract = _checkpoint_dependency_hashes(
        "conditional_power",
        {
            **value,
            "split_confirmation_scenario": {
                "scenario_id": "static-split",
                "model": {"prior": [0.5, 0.5]},
                "process": {"truth": [True, False]},
                "setting": {"alpha": 0.05},
            },
        },
    )
    changed_split_setting = _checkpoint_dependency_hashes(
        "conditional_power",
        {
            **value,
            "split_confirmation_scenario": {
                "scenario_id": "static-split",
                "model": {"prior": [0.5, 0.5]},
                "process": {"truth": [True, False]},
                "setting": {"alpha": 0.01},
            },
        },
    )
    assert changed_split_contract != changed_split_setting
    assert _checkpoint_dependency_hashes("power_curve", value) == {
        "scenario_hashes": [],
        "model_hashes": [],
        "process_hashes": [],
        "setting_hashes": [],
    }


def test_current_conditional_dependencies_are_derived_before_resume() -> None:
    config = _load_config(PROJECT_ROOT / "manuscript" / "config.yaml")["experiment"]
    dependencies = _expected_checkpoint_dependency_hashes("conditional_power", config)
    assert len(dependencies["scenario_hashes"]) == (
        len(config["action_loop"]["reliability_grid"]) + 1
    )
    assert all(len(value) == 64 for value in dependencies["scenario_hashes"])
    assert _expected_checkpoint_dependency_hashes("power_curve", config) == {
        "scenario_hashes": [],
        "model_hashes": [],
        "process_hashes": [],
        "setting_hashes": [],
    }


def test_generated_conditional_checkpoint_matches_current_lineage(
    generated_fast_bundle: Path,
) -> None:
    """A fresh fast checkpoint must be resumable under the current contract."""
    config = _resolve_experiment_profile(
        _load_config(PROJECT_ROOT / "manuscript" / "config.yaml")["experiment"],
        "fast",
    )
    checkpoint = json.loads(
        (generated_fast_bundle / "checkpoints" / "conditional_power.json").read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint["dependency_hashes"] == _expected_checkpoint_dependency_hashes(
        "conditional_power", config
    )


def test_external_output_artifacts_use_output_namespace(tmp_path: Path) -> None:
    artifact = tmp_path / "results.json"
    artifact.write_text("{}\n", encoding="utf-8")
    record = _artifact(artifact, tmp_path, "deterministic_results")
    assert record["root"] == "output"
    assert record["path"] == "results.json"


def test_generator_records_all_finalizer_required_benchmark_inputs(tmp_path: Path) -> None:
    """The producer records the real offline inputs required by both finalizers."""
    records = _benchmark_artifact_records(tmp_path)
    campaign_expected = required_campaign_artifacts(
        project_root=PROJECT_ROOT,
        output_root=tmp_path,
        figure_filenames=(),
        checkpoint_names=(),
    )
    release_expected = required_release_artifacts(
        project_root=PROJECT_ROOT,
        output_root=tmp_path,
        figure_filenames=(),
        checkpoint_names=(),
        require_render=False,
    )
    campaign_benchmarks = {
        name: value
        for name, value in campaign_expected.items()
        if name == "benchmark_manifest" or name.startswith("benchmark:")
    }
    release_benchmarks = {
        name: value
        for name, value in release_expected.items()
        if name == "benchmark_manifest" or name.startswith("benchmark:")
    }
    assert campaign_benchmarks == release_benchmarks
    assert set(records) == set(campaign_benchmarks)
    for name, (path, role) in campaign_benchmarks.items():
        record = records[name]
        assert record == {
            "root": "project",
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": role,
        }


def test_campaign_generator_binds_the_frozen_predecessor_receipts(
    tmp_path: Path,
    small_real_certificates: dict[str, dict[str, object]],
) -> None:
    """Producer and campaign inventory agree on the v2 policy inputs."""
    project_root = tmp_path / "isolated-project"
    config = (
        project_root
        / "data"
        / "research_lanes"
        / "conditional_power_campaign_v2.yaml"
    )
    config.parent.mkdir(parents=True)
    config.write_text(
        (
            PROJECT_ROOT
            / "data"
            / "research_lanes"
            / "conditional_power_campaign_v2.yaml"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    output = project_root / "output" / "campaign-v2"
    for predecessor in ("campaign", "precision"):
        atomic_write_json(
            project_root / "output" / predecessor / "certificates.json",
            copy.deepcopy(small_real_certificates),
        )
    expected = required_campaign_artifacts(
        project_root=project_root,
        output_root=output,
        figure_filenames=(),
        checkpoint_names=(),
        config_path=config,
    )
    predecessors = campaign_lane_predecessor_artifacts(
        config, project_root=project_root, output_root=output
    )
    assert {
        name: value for name, value in expected.items() if name.startswith("predecessor:")
    } == predecessors
    for name, (path, role) in predecessors.items():
        assert artifact_record(path, project_root, role, root_label="project") == {
            "root": "project",
            "path": path.relative_to(project_root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": "predecessor_evidence",
        }


def test_invalid_run_writes_failure_ledger_before_reraising(tmp_path: Path) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("experiment: [not-a-mapping]\n", encoding="utf-8")
    output = tmp_path / "release-output"
    with pytest.raises(ValueError, match="experiment"):
        run(config, profile="fast", workers=1, output_root=output, seed_override=17)
    ledger = yaml.safe_load((output / "run_log.json").read_text(encoding="utf-8"))
    validate_run_log(ledger)
    assert ledger["status"] == "failed"
    assert ledger["error"]["type"] == "ValueError"


def test_ledger_artifact_requirement_is_scoped_to_complete_default_trees() -> None:
    """A clean-checkout fast bundle proves its own artifacts, not other roots'."""
    assert (
        _requires_ledger_artifacts(
            allow_draft=False, output_is_default=True, complete_before_hydration=True
        )
        is True
    )
    assert (
        _requires_ledger_artifacts(
            allow_draft=False, output_is_default=True, complete_before_hydration=False
        )
        is False
    )
    assert (
        _requires_ledger_artifacts(
            allow_draft=False, output_is_default=False, complete_before_hydration=True
        )
        is False
    )
    assert (
        _requires_ledger_artifacts(
            allow_draft=True, output_is_default=True, complete_before_hydration=True
        )
        is False
    )
