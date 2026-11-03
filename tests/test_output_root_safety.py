"""Fail-closed boundaries for every script that can write an artifact bundle."""

from __future__ import annotations

from pathlib import Path

import pytest

from examples._common import output_path
from scripts.artifacts import require_safe_output_root
from scripts.finalize_campaign import finalize_campaign
from scripts.finalize_release import finalize_release
from scripts.generate_quality_report import _resolve_output_root as resolve_quality_output
from scripts.generate_results import run
from scripts.regenerate_figures import _resolve_output as resolve_figure_output
from scripts.renderer_provenance import write_renderer_provenance
from scripts.run_benchmark_preflight import _resolve_output_root as resolve_benchmark_output
from scripts.run_complexity_experiments import _validate_output_root as validate_complexity_output
from scripts.run_dobriban_bh_stress_bundle import _validate_output_root as validate_dobriban_output
from scripts.run_research_atlas import _validate_output_root as validate_atlas_output
from scripts.run_storey_method_review import _validate_output_root as validate_storey_output
from scripts.sync_rendered_figure_alts import _resolve_output_root as resolve_sync_output
from scripts.z_generate_manuscript_variables import (
    _resolve_output_root as resolve_hydration_output,
    _write_local_resolved_manuscript_tree,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AUTHORED_FILES = (
    PROJECT_ROOT / "manuscript" / "00_abstract.md",
    PROJECT_ROOT / "manuscript" / "config.yaml",
    PROJECT_ROOT / "manuscript" / "references.bib",
)


@pytest.mark.parametrize(
    "candidate",
    (
        PROJECT_ROOT,
        PROJECT_ROOT / "manuscript",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "output" / ".." / "manuscript",
        PROJECT_ROOT / "output" / "..",
    ),
)
def test_shared_output_root_guard_rejects_source_paths_and_resolved_aliases(
    candidate: Path,
) -> None:
    before = {path: path.read_bytes() for path in _AUTHORED_FILES}
    with pytest.raises(ValueError, match="output_root"):
        require_safe_output_root(candidate, project_root=PROJECT_ROOT)
    assert {path: path.read_bytes() for path in _AUTHORED_FILES} == before


@pytest.mark.parametrize(
    "resolver",
    (
        resolve_quality_output,
        resolve_hydration_output,
        resolve_figure_output,
        resolve_sync_output,
        resolve_benchmark_output,
        validate_complexity_output,
        validate_dobriban_output,
        validate_atlas_output,
        validate_storey_output,
    ),
)
def test_artifact_writer_resolvers_reject_the_project_root_without_writing(
    resolver,
) -> None:
    before = {path: path.read_bytes() for path in _AUTHORED_FILES}
    with pytest.raises(ValueError, match="output_root"):
        resolver(PROJECT_ROOT)
    assert {path: path.read_bytes() for path in _AUTHORED_FILES} == before


def test_primary_producers_and_finalizers_reject_project_root_before_mutation() -> None:
    before = {path: path.read_bytes() for path in _AUTHORED_FILES}
    with pytest.raises(ValueError, match="output_root"):
        run(
            PROJECT_ROOT / "manuscript" / "config.yaml",
            profile="fast",
            workers=1,
            output_root=PROJECT_ROOT,
        )
    with pytest.raises(ValueError, match="output_root"):
        finalize_release(PROJECT_ROOT)
    with pytest.raises(ValueError, match="output_root"):
        finalize_campaign(PROJECT_ROOT)
    with pytest.raises(ValueError, match="output_root"):
        output_path("safe-example", PROJECT_ROOT)
    with pytest.raises(ValueError, match="output_root"):
        _write_local_resolved_manuscript_tree(PROJECT_ROOT, PROJECT_ROOT, {})
    with pytest.raises(ValueError, match="output_root"):
        write_renderer_provenance(project_root=PROJECT_ROOT, output_root=PROJECT_ROOT)
    assert {path: path.read_bytes() for path in _AUTHORED_FILES} == before


def test_safe_output_roots_remain_available_inside_output_and_outside_project(
    tmp_path: Path,
) -> None:
    assert require_safe_output_root(
        PROJECT_ROOT / "output" / "isolated-candidate", project_root=PROJECT_ROOT
    ) == (PROJECT_ROOT / "output" / "isolated-candidate").resolve()
    external = tmp_path / "isolated-candidate"
    assert require_safe_output_root(external, project_root=PROJECT_ROOT) == external.resolve()


@pytest.mark.parametrize("authored_directory", ("manuscript", "src"))
@pytest.mark.parametrize("descendant", (False, True))
def test_shared_output_root_guard_rejects_canonical_output_symlink_into_authored_tree(
    tmp_path: Path,
    authored_directory: str,
    descendant: bool,
) -> None:
    project = tmp_path / "project"
    authored_root = project / authored_directory
    authored_root.mkdir(parents=True)
    sentinel = authored_root / "authored-sentinel.txt"
    sentinel.write_text("must not be touched\n", encoding="utf-8")
    output_link = project / "output"
    output_link.symlink_to(authored_root, target_is_directory=True)
    candidate = output_link / "candidate" if descendant else output_link

    with pytest.raises(ValueError, match="canonical project output"):
        require_safe_output_root(candidate, project_root=project)

    assert output_link.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "must not be touched\n"
    assert not (authored_root / "candidate").exists()


def test_shared_output_root_guard_allows_canonical_output_symlink_to_external_root(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external_output = tmp_path / "external-output"
    external_output.mkdir()
    output_link = project / "output"
    output_link.symlink_to(external_output, target_is_directory=True)

    assert require_safe_output_root(
        output_link / "candidate", project_root=project
    ) == (external_output / "candidate").resolve()


def test_build_tree_is_ignored_and_uv_configuration_makes_no_workspace_claim() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "build/" in gitignore
    assert "Inherits workspace settings" not in pyproject
    assert "[tool.uv]" not in pyproject


def test_ci_pins_the_lock_compatible_uv_release_and_covers_declared_python_floor() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert 'python-version: ["3.11", "3.12", "3.13", "3.14"]' in workflow
    # The test matrix, fast-artifacts, and the dedicated GNN conformance job.
    assert workflow.count('version: "0.13.6"') == 3
    assert "version: latest" not in workflow
    assert "uv lock --check" in workflow
    assert "compileall -q src scripts tests examples" in workflow
    assert "ruff check src scripts tests examples" in workflow
    assert "scripts/audit_fast_bundle.py --output-root output" in workflow
    assert "review_release.py --output-root output" not in workflow
