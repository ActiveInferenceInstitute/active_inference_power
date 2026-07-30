"""Fail-closed terminal-state tests for the expanded campaign finalizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.finalize_campaign import (
    _validate_campaign_certificate_artifact,
    _validate_campaign_terminal_state,
    finalize_campaign,
)
from scripts.artifacts import artifact_record
from scripts.release_provenance import (
    checkpoint_names_from_run_log,
    required_campaign_artifacts,
    validate_artifact_inventory,
)

def test_existing_failed_campaign_cannot_mutate_its_manifest(
    temporary_status_bundle,
) -> None:
    bundle = temporary_status_bundle(profile="campaign")
    manifest_path = bundle["output_root"] / "manifest.json"
    before = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="campaign run log is not passed"):
        finalize_campaign(bundle["output_root"])
    assert manifest_path.read_bytes() == before


def test_campaign_finalizer_rejects_a_missing_explicit_lane_config(
    tmp_path: Path,
    temporary_status_bundle,
) -> None:
    """Configuration validation occurs before any terminal manifest mutation."""
    bundle = temporary_status_bundle(profile="campaign")
    manifest_path = bundle["output_root"] / "manifest.json"
    before = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="source configuration is missing or empty"):
        finalize_campaign(bundle["output_root"], config_path=tmp_path / "missing.yaml")
    assert manifest_path.read_bytes() == before


def test_campaign_inventory_binds_the_explicit_preregistered_config(
    tmp_path: Path,
) -> None:
    """Inventory validation follows the exact campaign config, not a default."""
    project_root = tmp_path / "project"
    config = project_root / "data" / "custom_campaign.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("experiment: {}\n", encoding="utf-8")
    output_root = project_root / "output" / "campaign"
    inventory = required_campaign_artifacts(
        project_root=project_root,
        output_root=output_root,
        figure_filenames=(),
        checkpoint_names=(),
        config_path=config,
    )
    assert inventory["config"] == (config.resolve(), "configuration")
    record = artifact_record(config, project_root, "configuration", root_label="project")
    assert validate_artifact_inventory(
        {"artifacts": {"config": record}},
        {"config": inventory["config"]},
        project_root=project_root,
        output_root=tmp_path,
    ) == {"required_artifact_count": 1}


def test_campaign_terminal_state_requires_profile_and_certificate_agreement(
    temporary_status_bundle,
) -> None:
    bundle = temporary_status_bundle(
        profile="campaign", force_all_certificates_pass=True
    )
    manifest = bundle["manifest"]
    results = bundle["results"]
    run_log = bundle["run_log"]
    _validate_campaign_terminal_state(manifest, results, run_log)

    results["meta"]["profile"] = "release"
    with pytest.raises(ValueError, match="results must have meta.profile"):
        _validate_campaign_terminal_state(manifest, results, run_log)
    results["meta"]["profile"] = "campaign"

    run_log["profile"] = "release"
    with pytest.raises(ValueError, match="run log must have profile"):
        _validate_campaign_terminal_state(manifest, results, run_log)
    run_log["profile"] = "campaign"

    manifest["certificates_pass"] = False
    with pytest.raises(ValueError, match="manifest certificates_pass"):
        _validate_campaign_terminal_state(manifest, results, run_log)
    manifest["certificates_pass"] = True

    run_log["certificates_pass"] = False
    with pytest.raises(ValueError, match="run log certificates_pass"):
        _validate_campaign_terminal_state(manifest, results, run_log)
    run_log["certificates_pass"] = True

    results["certificates"]["storey_power_gain"]["passes"] = False
    results["meta"].pop("certificate_gate")
    manifest.pop("certificate_gate")
    run_log.pop("certificate_gate")
    with pytest.raises(ValueError, match="campaign certificates failed"):
        _validate_campaign_terminal_state(manifest, results, run_log)


def test_campaign_terminal_state_allows_a_visible_preregistered_diagnostic_failure(
    temporary_status_bundle,
) -> None:
    """A real failed Storey record is retained but need not block this lane."""
    bundle = temporary_status_bundle(
        profile="campaign", diagnostic_storey_failure=True
    )
    manifest = bundle["manifest"]
    results = bundle["results"]
    run_log = bundle["run_log"]
    gate = bundle["gate"]
    assert gate["terminal_certificates_pass"] is True
    assert gate["diagnostic_failures"] == ["storey_power_gain"]
    _validate_campaign_terminal_state(manifest, results, run_log)

    # An explicit diagnostic designation is evidence, not an optional
    # annotation: every terminal witness must retain the exact partition.
    manifest.pop("certificate_gate")
    with pytest.raises(ValueError, match="does not record the explicit certificate gate"):
        _validate_campaign_terminal_state(manifest, results, run_log)


def test_campaign_certificate_artifact_must_match_results(
    temporary_status_bundle,
) -> None:
    bundle = temporary_status_bundle(profile="campaign")
    output_root = bundle["output_root"]
    results = bundle["results"]
    artifact = output_root / "certificates.json"
    _validate_campaign_certificate_artifact(output_root, results)

    standalone = json.loads(artifact.read_text(encoding="utf-8"))
    standalone["storey_power_gain"]["passes"] = not standalone["storey_power_gain"][
        "passes"
    ]
    artifact.write_text(json.dumps(standalone), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match results certificates"):
        _validate_campaign_certificate_artifact(output_root, results)


def test_generated_checkpoint_inventory_uses_terminal_run_log_stages(
    generated_fast_bundle: Path,
) -> None:
    run_log = json.loads(
        (generated_fast_bundle / "run_log.json").read_text(encoding="utf-8")
    )
    declared = checkpoint_names_from_run_log(run_log)
    actual = tuple(
        sorted(
            path.stem
            for path in (generated_fast_bundle / "checkpoints").glob("*.json")
        )
    )
    assert declared == actual
