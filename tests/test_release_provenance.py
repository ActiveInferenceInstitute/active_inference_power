"""Real-file source-witness checks for release provenance boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.artifacts import artifact_record
from scripts.release_provenance import (
    campaign_lane_predecessor_artifacts,
    execution_source_sha256,
    renderer_input_hash,
    resolve_config_path,
    validate_execution_provenance,
    validate_checkpoint_provenance,
    validate_source_witnesses,
)


def _bundle(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = tmp_path / "manuscript" / "config.yaml"
    ledger = tmp_path / "data" / "claim_ledger.yaml"
    config.parent.mkdir()
    ledger.parent.mkdir()
    config.write_text("experiment: {alpha: 0.05}\n", encoding="utf-8")
    ledger.write_text("claims: []\n", encoding="utf-8")
    config_record = artifact_record(config, tmp_path, "configuration")
    ledger_record = artifact_record(ledger, tmp_path, "claim_ledger")
    config_digest = str(config_record["sha256"])
    ledger_digest = str(ledger_record["sha256"])
    manifest: dict[str, Any] = {
        "config_sha256": config_digest,
        "artifacts": {"config": config_record, "claim_ledger": ledger_record},
    }
    results: dict[str, Any] = {
        "meta": {
            "config_sha256": config_digest,
            "claim_ledger_sha256": ledger_digest,
        }
    }
    run_log: dict[str, Any] = {
        "config_sha256": config_digest,
        "claim_ledger_sha256": ledger_digest,
    }
    return manifest, results, run_log


def test_source_witnesses_bind_manifest_results_run_log_and_live_sources(tmp_path: Path) -> None:
    manifest, results, run_log = _bundle(tmp_path)
    proven = validate_source_witnesses(
        manifest,
        results,
        run_log,
        project_root=tmp_path,
        output_root=tmp_path / "output",
        config_path=tmp_path / "manuscript" / "config.yaml",
        claim_ledger_path=tmp_path / "data" / "claim_ledger.yaml",
    )
    assert proven["config_sha256"] == results["meta"]["config_sha256"]
    assert proven["claim_ledger_sha256"] == results["meta"]["claim_ledger_sha256"]


def test_isolated_lane_config_requires_its_exact_preregistered_path(
    tmp_path: Path,
) -> None:
    manifest, results, run_log = _bundle(tmp_path)
    lane_config = tmp_path / "research_lanes" / "campaign_preregistered.yaml"
    lane_config.parent.mkdir()
    lane_config.write_text("experiment: {alpha: 0.025}\n", encoding="utf-8")
    lane_record = artifact_record(lane_config, tmp_path, "configuration")
    lane_digest = str(lane_record["sha256"])
    manifest["artifacts"]["config"] = lane_record
    manifest["config_sha256"] = lane_digest
    results["meta"]["config_sha256"] = lane_digest
    run_log["config_sha256"] = lane_digest

    resolved = resolve_config_path(
        Path("research_lanes/campaign_preregistered.yaml"), project_root=tmp_path
    )
    assert resolved == lane_config.resolve()
    provenance = validate_source_witnesses(
        manifest,
        results,
        run_log,
        project_root=tmp_path,
        output_root=tmp_path / "output",
        config_path=resolved,
        claim_ledger_path=tmp_path / "data" / "claim_ledger.yaml",
    )
    assert provenance["config_sha256"] == lane_digest

    with pytest.raises(ValueError, match="expected source"):
        validate_source_witnesses(
            manifest,
            results,
            run_log,
            project_root=tmp_path,
            output_root=tmp_path / "output",
            config_path=tmp_path / "manuscript" / "config.yaml",
            claim_ledger_path=tmp_path / "data" / "claim_ledger.yaml",
        )
    with pytest.raises(ValueError, match="source configuration is missing or empty"):
        resolve_config_path(tmp_path / "missing.yaml", project_root=tmp_path)


def test_campaign_lane_predecessor_receipts_are_safe_and_bound(tmp_path: Path) -> None:
    """Frozen campaign policy inputs must be real, external receipts."""
    predecessor_root = tmp_path / "output" / "campaign"
    predecessor_root.mkdir(parents=True)
    first = predecessor_root / "certificates.json"
    first.write_text('{"status": "failed"}\n', encoding="utf-8")
    second_root = tmp_path / "output" / "precision"
    second_root.mkdir(parents=True)
    second = second_root / "certificates.json"
    second.write_text('{"status": "failed"}\n', encoding="utf-8")
    config = tmp_path / "data" / "research_lanes" / "campaign.yaml"
    config.parent.mkdir(parents=True)
    lane: dict[str, Any] = {
        "research_lane": {
            "lane_id": "campaign-v2",
            "output_namespace": "output/campaign-v2",
            "predecessor_evidence": [
                "output/campaign/certificates.json",
                "output/precision/certificates.json",
            ],
            "decision": "retain prior failures as explicit policy inputs",
            "non_claim": "does not promote predecessor evidence",
        },
        "experiment": {"alpha": 0.05},
    }
    config.write_text(yaml.safe_dump(lane), encoding="utf-8")
    output = tmp_path / "output" / "campaign-v2"

    records = campaign_lane_predecessor_artifacts(
        config, project_root=tmp_path, output_root=output
    )
    assert records == {
        "predecessor:output/campaign/certificates.json": (
            first.resolve(),
            "predecessor_evidence",
        ),
        "predecessor:output/precision/certificates.json": (
            second.resolve(),
            "predecessor_evidence",
        ),
    }

    with pytest.raises(ValueError, match="output root"):
        campaign_lane_predecessor_artifacts(
            config, project_root=tmp_path, output_root=tmp_path / "output" / "wrong"
        )

    lane["research_lane"]["predecessor_evidence"] = ["../outside.json"]
    config.write_text(yaml.safe_dump(lane), encoding="utf-8")
    with pytest.raises(ValueError, match="safe output-relative"):
        campaign_lane_predecessor_artifacts(
            config, project_root=tmp_path, output_root=output
        )

    lane["research_lane"]["predecessor_evidence"] = [
        "output/campaign/certificates.json",
        "output/campaign/certificates.json",
    ]
    config.write_text(yaml.safe_dump(lane), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated"):
        campaign_lane_predecessor_artifacts(
            config, project_root=tmp_path, output_root=output
        )

    lane["research_lane"]["predecessor_evidence"] = [
        "output/campaign-v2/self.json"
    ]
    (output).mkdir(parents=True)
    (output / "self.json").write_text("self\n", encoding="utf-8")
    config.write_text(yaml.safe_dump(lane), encoding="utf-8")
    with pytest.raises(ValueError, match="own output namespace"):
        campaign_lane_predecessor_artifacts(
            config, project_root=tmp_path, output_root=output
        )


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        ("manuscript/config.yaml", "experiment: {alpha: 0.01}\n", "configuration"),
        ("data/claim_ledger.yaml", "claims: [changed]\n", "claim-ledger"),
    ],
)
def test_source_witnesses_reject_live_input_drift(
    tmp_path: Path, path: str, replacement: str, message: str
) -> None:
    manifest, results, run_log = _bundle(tmp_path)
    (tmp_path / path).write_text(replacement, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_source_witnesses(
            manifest,
            results,
            run_log,
            project_root=tmp_path,
            output_root=tmp_path / "output",
            config_path=tmp_path / "manuscript" / "config.yaml",
            claim_ledger_path=tmp_path / "data" / "claim_ledger.yaml",
        )


def test_source_witnesses_reject_manifest_result_disagreement(tmp_path: Path) -> None:
    manifest, results, run_log = _bundle(tmp_path)
    results["meta"] = {
        **results["meta"],
        "config_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="configuration provenance mismatch"):
        validate_source_witnesses(
            manifest,
            results,
            run_log,
            project_root=tmp_path,
            output_root=tmp_path / "output",
            config_path=tmp_path / "manuscript" / "config.yaml",
            claim_ledger_path=tmp_path / "data" / "claim_ledger.yaml",
        )


def test_checkpoint_provenance_requires_exact_bundle_lineage() -> None:
    checkpoint = {
        "run_id": "run",
        "config_sha256": "a" * 64,
        "claim_ledger_sha256": "b" * 64,
        "execution_source_sha256": "d" * 64,
        "profile": "release",
    }
    validate_checkpoint_provenance(
        checkpoint,
        run_id="run",
        config_sha256="a" * 64,
        claim_ledger_sha256="b" * 64,
        profile="release",
        execution_source_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="claim-ledger"):
        validate_checkpoint_provenance(
            checkpoint,
            run_id="run",
            config_sha256="a" * 64,
            claim_ledger_sha256="c" * 64,
            profile="release",
        )
    with pytest.raises(ValueError, match="stage"):
        validate_checkpoint_provenance(
            checkpoint,
            run_id="run",
            config_sha256="a" * 64,
            claim_ledger_sha256="b" * 64,
            profile="release",
            stage="different_stage",
            execution_source_sha256="d" * 64,
        )
    with pytest.raises(ValueError, match="execution-source"):
        validate_checkpoint_provenance(
            checkpoint,
            run_id="run",
            config_sha256="a" * 64,
            claim_ledger_sha256="b" * 64,
            profile="release",
            execution_source_sha256="e" * 64,
        )


def test_renderer_input_hash_normalizes_its_embedded_witness_and_tracks_inputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "manuscript"
    resolved = tmp_path / "output" / "manuscript"
    figures = tmp_path / "output" / "figures"
    data = tmp_path / "output" / "data"
    source.mkdir(parents=True)
    resolved.mkdir(parents=True)
    figures.mkdir(parents=True)
    data.mkdir(parents=True)
    source_text = "Render digest: {{RESULT_RENDER_INPUT_HASH}}\n"
    (source / "00_section.md").write_text(source_text, encoding="utf-8")
    first = "a" * 64
    second = "b" * 64
    (resolved / "00_section.md").write_text(
        source_text.replace("{{RESULT_RENDER_INPUT_HASH}}", first), encoding="utf-8"
    )
    (data / "manuscript_variables.json").write_text(
        '{"RESULT_RENDER_INPUT_HASH": "' + first + '"}', encoding="utf-8"
    )
    (figures / "figure_registry.json").write_text("{}\n", encoding="utf-8")
    (figures / "figure.png").write_bytes(b"first figure bytes")
    first_digest = renderer_input_hash(project_root=tmp_path, output_root=tmp_path / "output")

    (resolved / "00_section.md").write_text(
        source_text.replace("{{RESULT_RENDER_INPUT_HASH}}", second), encoding="utf-8"
    )
    (data / "manuscript_variables.json").write_text(
        '{"RESULT_RENDER_INPUT_HASH": "' + second + '"}', encoding="utf-8"
    )
    assert renderer_input_hash(project_root=tmp_path, output_root=tmp_path / "output") == first_digest

    # The sibling renderer discovers generated top-level manuscript sections
    # (for example transmission bookends) in addition to source-paired files.
    # Their bytes must therefore participate in the self-normalizing witness.
    (resolved / "00_00_transmission_begin.md").write_text(
        "Generated transmission bookend.\n", encoding="utf-8"
    )
    assert renderer_input_hash(project_root=tmp_path, output_root=tmp_path / "output") != first_digest

    (figures / "figure.png").write_bytes(b"changed figure bytes")
    assert renderer_input_hash(project_root=tmp_path, output_root=tmp_path / "output") != first_digest


def test_execution_source_digest_binds_manifest_results_run_log_and_live_tree(
    tmp_path: Path,
) -> None:
    package = tmp_path / "src" / "active_inference_power"
    scripts = tmp_path / "scripts"
    benchmarks = tmp_path / "data" / "benchmarks"
    package.mkdir(parents=True)
    scripts.mkdir()
    benchmarks.mkdir(parents=True)
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    for name in (
        "generate_results.py",
        "checkpointing.py",
        "run_ledger.py",
        "artifacts.py",
        "release_provenance.py",
        "review_release.py",
        "audit_manuscript_presentation.py",
        "audit_documentation_links.py",
        "sync_rendered_figure_alts.py",
    ):
        (scripts / name).write_text(f"# {name}\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "domain_profile.yaml").write_text("profile: test\n", encoding="utf-8")
    (tmp_path / "experiment_plan.yaml").write_text("plan: test\n", encoding="utf-8")
    (manuscript / "config.yaml").write_text("experiment: {}\n", encoding="utf-8")
    (benchmarks / "fixture.csv").write_text("x\n1\n", encoding="utf-8")
    digest = execution_source_sha256(tmp_path)
    manifest: dict[str, Any] = {"execution_source_sha256": digest}
    results: dict[str, Any] = {"meta": {"execution_source_sha256": digest}}
    run_log: dict[str, Any] = {"execution_source_sha256": digest}
    assert (
        validate_execution_provenance(
            manifest, results, run_log, project_root=tmp_path
        )
        == digest
    )
    (scripts / "non_generator_helper.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="execution-source provenance mismatch"):
        validate_execution_provenance(
            manifest, results, run_log, project_root=tmp_path
        )
    (scripts / "non_generator_helper.py").unlink()
    (package / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="execution-source provenance mismatch"):
        validate_execution_provenance(
            manifest, results, run_log, project_root=tmp_path
        )

    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (scripts / "sync_rendered_figure_alts.py").write_text(
        "# changed rendered-alt synchronization helper\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="execution-source provenance mismatch"):
        validate_execution_provenance(
            manifest, results, run_log, project_root=tmp_path
        )

    (scripts / "sync_rendered_figure_alts.py").write_text(
        "# sync_rendered_figure_alts.py\n", encoding="utf-8"
    )
    (scripts / "audit_manuscript_presentation.py").write_text(
        "# changed manuscript presentation helper\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="execution-source provenance mismatch"):
        validate_execution_provenance(
            manifest, results, run_log, project_root=tmp_path
        )

    (scripts / "audit_manuscript_presentation.py").write_text(
        "# audit_manuscript_presentation.py\n", encoding="utf-8"
    )
    (scripts / "audit_documentation_links.py").write_text(
        "# changed documentation-link audit helper\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="execution-source provenance mismatch"):
        validate_execution_provenance(
            manifest, results, run_log, project_root=tmp_path
        )

    (scripts / "audit_documentation_links.py").write_text(
        "# audit_documentation_links.py\n", encoding="utf-8"
    )
    (scripts / "release_provenance.py").write_text(
        "# changed release provenance helper\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="execution-source provenance mismatch"):
        validate_execution_provenance(
            manifest, results, run_log, project_root=tmp_path
        )

    (scripts / "release_provenance.py").write_text(
        "# release_provenance.py\n", encoding="utf-8"
    )
    (scripts / "review_release.py").write_text(
        "# changed release review helper\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="execution-source provenance mismatch"):
        validate_execution_provenance(
            manifest, results, run_log, project_root=tmp_path
        )
