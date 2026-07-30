"""Real-file tests for the bounded R1--R5/G4 research-atlas orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

from matplotlib import pyplot as plt
import pytest

from scripts.audit_figure_accessibility import audit_figure_accessibility
from scripts.artifacts import sha256_file
from scripts.run_research_atlas import (
    ATLAS_STATUS,
    LANE_IDS,
    ROOT,
    ResearchAtlasPlan,
    _canonical_hash,
    main,
    run_research_atlas,
)


def _small_real_plan() -> ResearchAtlasPlan:
    """Return a compact plan that still invokes every real domain lane."""
    return ResearchAtlasPlan(
        seed=90210,
        workers=1,
        r1_n_reps=3,
        r2_n_reps=3,
        r3_n_reps=2,
        r4_n_reps=4,
        r5_train_reps=1,
        r5_evaluation_reps=1,
        r5_partitions=2,
        g4_n_reps=3,
    )


def test_real_atlas_bundle_is_atomic_hash_bound_and_accessible(tmp_path: Path) -> None:
    target = tmp_path / "bounded-research-atlas"
    receipt = run_research_atlas(target, _small_real_plan())

    assert target.is_dir()
    assert receipt["status"] == ATLAS_STATUS
    assert Path(receipt["atlas_path"]) == target / "research_atlas.json"
    assert Path(receipt["manifest_path"]) == target / "manifest.json"
    assert sha256_file(target / "manifest.json") == receipt["manifest_sha256"]
    assert not list(tmp_path.glob(".bounded-research-atlas.research-atlas-*"))

    atlas = json.loads((target / "research_atlas.json").read_text(encoding="utf-8"))
    atlas_body = dict(atlas)
    content_hash = atlas_body.pop("content_hash")
    atlas_body.pop("content_hash_scope")
    assert _canonical_hash(atlas_body) == content_hash
    assert atlas["status"] == ATLAS_STATUS
    assert set(atlas["canonical_plan"]["lane_seed_schedule"]) == set(LANE_IDS)
    assert set(atlas["lanes"]) == set(LANE_IDS)
    assert set(atlas["canonical_plan"]["domain_contracts"]) == set(LANE_IDS)
    assert set(atlas["source_hashes"]) == {"orchestrator", *LANE_IDS}
    assert all(len(value) == 64 for value in atlas["source_hashes"].values())
    assert "does not promote" in atlas["claim_boundary"]
    assert atlas["lanes"]["R1"]["pooled_utility_or_policy_ranking"] == "not_reported"
    assert atlas["lanes"]["R2"]["all_null"] is True
    assert atlas["lanes"]["R3"]["evidence_class"] == "stress_test"
    assert atlas["lanes"]["R4"]["kind"] == "sequential_evidence_comparison_atlas"
    assert atlas["lanes"]["R5"]["stream_proof"]["all_materialized_trace_seeds_unique"]
    assert atlas["lanes"]["G4"]["common_random_numbers"] is True

    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == ATLAS_STATUS
    assert manifest["plan_hash"] == atlas["plan_hash"]
    assert manifest["atlas_content_hash"] == atlas["content_hash"]
    assert set(manifest["artifacts"]) == {
        "atlas",
        "figure_registry",
        "dashboard_payload",
        "figure:research_atlas_operating_characteristics.png",
        "figure:research_atlas_evidence_boundaries.png",
    }
    for record in manifest["artifacts"].values():
        assert record["root"] == "output"
        artifact_path = target / record["path"]
        assert artifact_path.is_file()
        assert sha256_file(artifact_path) == record["sha256"]
        assert artifact_path.stat().st_size == record["bytes"]

    registry_path = target / "figures" / "figure_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["schema_version"] == "2.0"
    assert registry["atlas_content_hash"] == atlas["content_hash"]
    assert registry["atlas_file_sha256"] == sha256_file(target / "research_atlas.json")
    assert len(registry["figures"]) == 2
    for figure in registry["figures"]:
        figure_path = target / figure["path"]
        image = plt.imread(figure_path)
        assert image.shape[0] >= 900
        assert image.shape[1] >= 1_500
        assert figure["sha256"] == sha256_file(figure_path)
        assert figure["alt"]
        assert figure["caption"]
        assert figure["metadata"]["estimand"]
        assert figure["metadata"]["sample_size"]
        assert figure["metadata"]["scenario_hashes"]
        assert figure["metadata"]["result_lineage_hash"] == atlas["content_hash"]
        assert figure["metadata"]["validity_boundary"]
        assert figure["metadata"]["redundant_encoding"]

    accessibility = audit_figure_accessibility(target)
    assert accessibility["passed"], accessibility["findings"]

    dashboard = json.loads((target / "dashboard.json").read_text(encoding="utf-8"))
    assert dashboard["status"] == ATLAS_STATUS
    assert dashboard["atlas_content_hash"] == atlas["content_hash"]
    assert dashboard["figure_registry_sha256"] == sha256_file(registry_path)
    assert [lane["id"] for lane in dashboard["lanes"]] == list(LANE_IDS)
    assert "No pass/fail promotion" in dashboard["claim_boundary"]


def test_atlas_refuses_default_or_existing_output_roots_without_overwrite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="new isolated"):
        run_research_atlas(ROOT / "output", _small_real_plan())

    target = tmp_path / "existing-atlas"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        run_research_atlas(target, _small_real_plan())
    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_plan_and_cli_boundaries_are_explicit() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ResearchAtlasPlan(workers=0)
    with pytest.raises(ValueError, match="at least 2"):
        ResearchAtlasPlan(r5_partitions=1)
    with pytest.raises(SystemExit):
        main([])
