"""Real-file tests for the isolated frozen Storey method-review command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from active_inference_power.storey_method_review import (
    DEFAULT_STOREY_METHOD_REVIEW_PLAN,
    STOREY_METHOD_REVIEW_STATUS,
    validate_storey_method_review,
)
from scripts.artifacts import sha256_file
from scripts.run_storey_method_review import (
    ROOT,
    _parser,
    _validate_output_root,
    main,
    run_storey_method_review,
)


def test_frozen_runner_writes_direct_root_hash_bound_review_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "frozen-storey-review"
    assert main(["--output-root", str(target)]) == 0
    receipt = json.loads(capsys.readouterr().out)

    review_path = target / "storey_method_review.json"
    manifest_path = target / "manifest.json"
    assert target.is_dir()
    assert set(path.name for path in target.iterdir()) == {
        "storey_method_review.json",
        "manifest.json",
    }
    assert Path(receipt["output_root"]) == target
    assert Path(receipt["review_path"]) == review_path
    assert Path(receipt["manifest_path"]) == manifest_path
    assert receipt["status"] == STOREY_METHOD_REVIEW_STATUS
    assert receipt["review_sha256"] == sha256_file(review_path)
    assert receipt["manifest_sha256"] == sha256_file(manifest_path)
    assert not list(tmp_path.glob(".frozen-storey-review.storey-method-review-*"))

    review = json.loads(review_path.read_text(encoding="utf-8"))
    validate_storey_method_review(review)
    assert review["review_plan"] == DEFAULT_STOREY_METHOD_REVIEW_PLAN.to_mapping()
    assert review["result_sha256"] == receipt["result_sha256"]
    assert review["status"] == STOREY_METHOD_REVIEW_STATUS
    assert "does not change certificate policy" in review["nonclaims"][2]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == STOREY_METHOD_REVIEW_STATUS
    assert manifest["review_result_sha256"] == review["result_sha256"]
    assert manifest["plan_sha256"] == review["plan_sha256"]
    assert manifest["schedule_sha256"] == review["schedule_sha256"]
    assert set(manifest["source_hashes"]) == {
        "orchestrator",
        "storey_method_review",
        "corrections",
        "simulation",
        "error_rates",
        "uncertainty",
    }
    artifact = manifest["artifacts"]["storey_method_review"]
    assert artifact["root"] == "output"
    assert artifact["path"] == "storey_method_review.json"
    assert artifact["sha256"] == sha256_file(review_path)
    assert artifact["bytes"] == review_path.stat().st_size


def test_runner_rejects_existing_or_nonisolated_roots_without_overwrite(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="not the project output root"):
        run_storey_method_review(ROOT / "output")
    target = tmp_path / "preserve-existing"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        run_storey_method_review(target)
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    with pytest.raises(ValueError, match="concrete directory"):
        _validate_output_root(Path("/"))


def test_cli_boundary_requires_explicit_output_root_without_tuning_flags() -> None:
    with pytest.raises(SystemExit):
        main([])
    help_text = _parser().format_help()
    assert "--output-root" in help_text
    assert "--n-reps" not in help_text
    assert "--seed" not in help_text
