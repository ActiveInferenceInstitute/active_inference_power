"""Real-Git checks for the post-render sibling-template provenance receipt."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.renderer_provenance import (
    capture_renderer_provenance,
    renderer_provenance_path,
    renderer_source_paths,
    validate_renderer_provenance,
    write_renderer_provenance,
)


def _run_git(root: Path, *arguments: str) -> None:
    """Run an actual local Git setup command for a temporary template fixture."""
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _template_checkout(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create the minimum real sibling template closure and commit it."""
    project = tmp_path / "project"
    template = tmp_path / "template"
    output = tmp_path / "candidate"
    project.mkdir()
    for relative, content in {
        "pyproject.toml": "[project]\nname = 'template-fixture'\n",
        "uv.lock": "version = 1\n",
        "README.md": "template fixture\n",
        "scripts/__init__.py": "def ensure_repo_root_on_path():\n    return None\n",
        "scripts/pipeline/stage_03_render.py": "from infrastructure.rendering.pipeline import execute_render_pipeline\n",
        "infrastructure/rendering/pipeline.py": "def execute_render_pipeline():\n    return 0\n",
        "infrastructure/rendering/ide_style.css": "body { color: black; }\n",
        "infrastructure/orchestration/__init__.py": "\n",
        "infrastructure/orchestration/cli.py": "def main():\n    return 0\n",
        "infrastructure/orchestration/link_sync.py": "def maybe_sync_all_links(_root):\n    return []\n",
        "infrastructure/project/discovery.py": "def resolve_project_root(_root, _name):\n    return _root\n",
        "infrastructure/project/linking.py": "SKIP_ENV_VAR = 'TEMPLATE_SKIP_LINK_SYNC'\n",
        "infrastructure/core/sidecar_linking.py": "class SidecarLinkConfig:\n    pass\n",
        "infrastructure/publishing/transmission_bookends.py": "def write_transmission_bookends(*_args, **_kwargs):\n    return None\n",
    }.items():
        path = template / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _run_git(template, "init")
    _run_git(template, "add", ".")
    _run_git(
        template,
        "-c",
        "user.name=Renderer provenance test",
        "-c",
        "user.email=renderer-provenance@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    rendered_bytes: dict[str, bytes] = {
        "pdf/active_inference_power_combined.pdf": b"real rendered PDF bytes\n",
        "web/index.html": b"<html>real rendered HTML</html>\n",
        "web/_combined_manuscript.md": b"real combined manuscript\n",
    }
    for relative, rendered_content in rendered_bytes.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rendered_content)
    return project, template, output


def test_renderer_receipt_binds_entrypoints_renderer_tree_toolchain_and_rendered_bytes(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("Git is required for renderer provenance")
    project, template, output = _template_checkout(tmp_path)

    receipt = capture_renderer_provenance(
        project_root=project,
        output_root=output,
    )

    assert receipt["template"]["checkout"] == "sibling_template"
    assert receipt["template"]["git"]["dirty"] is False
    assert len(str(receipt["template"]["git"]["head"])) == 40
    source_paths = {
        item["path"] for item in receipt["template"]["renderer_source_files"]
    }
    assert "scripts/pipeline/stage_03_render.py" in source_paths
    assert "infrastructure/rendering/pipeline.py" in source_paths
    assert "infrastructure/orchestration/link_sync.py" in source_paths
    assert "infrastructure/project/linking.py" in source_paths
    assert "infrastructure/publishing/transmission_bookends.py" in source_paths
    assert set(receipt["toolchain"]) == {"pandoc", "xelatex", "lualatex", "pdflatex"}
    assert all(
        isinstance(record["available"], bool)
        for record in receipt["toolchain"].values()
    )
    assert set(receipt["rendered_artifacts"]) == {
        "pdf",
        "html",
        "combined_manuscript",
    }

    written = write_renderer_provenance(project_root=project, output_root=output)
    assert Path(written["path"]) == renderer_provenance_path(output)
    validated = validate_renderer_provenance(project_root=project, output_root=output)
    assert validated["template_dirty"] is False
    assert validated["renderer_source_tree_sha256"] == receipt["template"][
        "renderer_source_tree_sha256"
    ]


def test_renderer_receipt_rejects_live_stage_entrypoint_or_render_drift(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("Git is required for renderer provenance")
    project, template, output = _template_checkout(tmp_path)
    write_renderer_provenance(project_root=project, output_root=output)

    stage = template / "scripts" / "pipeline" / "stage_03_render.py"
    stage.write_text("# changed real stage entry point\n", encoding="utf-8")
    assert "scripts/pipeline/stage_03_render.py" in {
        path.relative_to(template).as_posix() for path in renderer_source_paths(template)
    }
    with pytest.raises(ValueError, match="does not match the live sibling template"):
        validate_renderer_provenance(project_root=project, output_root=output)

    stage.write_text(
        "from infrastructure.rendering.pipeline import execute_render_pipeline\n",
        encoding="utf-8",
    )
    (output / "web" / "index.html").write_text("changed rendered HTML\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the live sibling template"):
        validate_renderer_provenance(project_root=project, output_root=output)


def test_renderer_dirty_state_digest_binds_content_not_only_modified_paths(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("Git is required for renderer provenance")
    project, template, output = _template_checkout(tmp_path)
    unrelated = template / "README.md"
    unrelated.write_text("first dirty content\n", encoding="utf-8")
    first = capture_renderer_provenance(project_root=project, output_root=output)
    unrelated.write_text("second dirty content\n", encoding="utf-8")
    second = capture_renderer_provenance(project_root=project, output_root=output)

    assert first["template"]["git"]["dirty"] is True
    assert second["template"]["git"]["dirty"] is True
    assert first["template"]["git"]["dirty_entry_count"] == 1
    assert second["template"]["git"]["dirty_entry_count"] == 1
    assert (
        first["template"]["git"]["dirty_state_sha256"]
        != second["template"]["git"]["dirty_state_sha256"]
    )


def test_renderer_receipt_rejects_malformed_or_unavailable_template_boundary(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("Git is required for renderer provenance")
    project, _template, output = _template_checkout(tmp_path)
    write_renderer_provenance(project_root=project, output_root=output)
    path = renderer_provenance_path(output)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["template"]["git"]["head"] = "not-a-git-object"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="Git identity"):
        validate_renderer_provenance(project_root=project, output_root=output)

    with pytest.raises(FileNotFoundError, match="sibling renderer checkout is unavailable"):
        capture_renderer_provenance(
            project_root=project,
            output_root=output,
            template_root=tmp_path / "missing-template",
        )
