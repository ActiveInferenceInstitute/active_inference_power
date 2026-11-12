"""Real-file checks for the reversible controlled-renderer output binding."""
from pathlib import Path

import pytest

from scripts.render_manuscript import _bind_output


@pytest.mark.parametrize("renderer_failure", (False, True))
def test_render_binding_restores_original_tree_and_retains_rendered_candidate(
    tmp_path: Path, renderer_failure: bool,
) -> None:
    output = tmp_path / "output"
    candidate = output / "release-candidate"
    candidate.mkdir(parents=True)
    (output / "original.txt").write_text("preserve me", encoding="utf-8")
    try:
        with _bind_output(output, candidate):
            assert output.is_symlink()
            (output / "rendered.html").write_text("real candidate output", encoding="utf-8")
            if renderer_failure:
                raise ValueError("renderer failed")
    except ValueError:
        if not renderer_failure:
            raise
    assert output.is_dir() and not output.is_symlink()
    assert (output / "original.txt").read_text(encoding="utf-8") == "preserve me"
    assert (candidate / "rendered.html").read_text(encoding="utf-8") == "real candidate output"
    assert not list(tmp_path.glob(".output-render-bind-*"))


def test_render_binding_preserves_both_trees_when_output_is_replaced(tmp_path: Path) -> None:
    output = tmp_path / "output"
    candidate = output / "release-candidate"
    candidate.mkdir(parents=True)
    (output / "original.txt").write_text("original", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unexpected output path"):
        with _bind_output(output, candidate):
            output.unlink()
            output.mkdir()
            (output / "other-writer.txt").write_text("do not replace", encoding="utf-8")
    assert (output / "other-writer.txt").read_text(encoding="utf-8") == "do not replace"
    saved = next(tmp_path.glob(".output-render-bind-*/output"))
    assert (saved / "original.txt").read_text(encoding="utf-8") == "original"
