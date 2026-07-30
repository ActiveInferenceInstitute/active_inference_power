"""Fail-closed provenance for a rendered publication bundle.

The manuscript project deliberately delegates PDF and HTML production to the
sibling ``template`` checkout.  Rendered bytes alone therefore do not identify
the renderer revision, local renderer edits, or the executable rendering
toolchain.  This module records those inputs after a render and requires an
exact live match during terminal review.  It intentionally does not make a
portability claim when the sibling checkout is unavailable: capture and
validation both fail in that case.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping

from scripts.artifacts import atomic_write_json, require_safe_output_root, sha256_file
from scripts.release_provenance import require_file


RENDERER_PROVENANCE_SCHEMA_VERSION = "1.0"
RENDERER_PROVENANCE_RELATIVE_PATH = Path("review") / "renderer_provenance.json"
_GIT_HEAD_RE = re.compile(r"^[0-9a-f]{40,64}$")
_RENDERER_ROOT_INPUTS = (
    "pyproject.toml",
    "uv.lock",
    # The stage entry point imports this bootstrap before it reaches the
    # renderer package, so it is an actual part of the invoked render path.
    "scripts/__init__.py",
    "scripts/pipeline/stage_03_render.py",
    # Project discovery and link synchronization can select or replace the
    # working sidecar exposed to stage 03; bind their source explicitly.
    "infrastructure/project/discovery.py",
    "infrastructure/project/linking.py",
    "infrastructure/core/sidecar_linking.py",
    # The render pipeline imports this hook dynamically before producing the
    # combined output, so it is part of the concrete render path as well.
    "infrastructure/publishing/transmission_bookends.py",
)
_RENDERER_ORCHESTRATION_DIRECTORY = Path("infrastructure") / "orchestration"
_TOOL_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pandoc", ("pandoc", "--version")),
    ("xelatex", ("xelatex", "--version")),
    ("lualatex", ("lualatex", "--version")),
    ("pdflatex", ("pdflatex", "--version")),
)
_RENDERED_ARTIFACTS = {
    "pdf": Path("pdf") / "active_inference_power_combined.pdf",
    "html": Path("web") / "index.html",
    "combined_manuscript": Path("web") / "_combined_manuscript.md",
}


def default_template_root(project_root: Path) -> Path:
    """Return the conventional sibling checkout without creating it."""
    return Path(project_root).resolve().parent / "template"


def _require_template_root(template_root: Path) -> Path:
    root = Path(template_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            "sibling renderer checkout is unavailable; cannot establish renderer provenance: "
            f"{root}"
        )
    return root


def _is_generated_renderer_path(path: Path) -> bool:
    """Exclude interpreter/cache files that are not renderer source inputs."""
    return "__pycache__" in path.parts or path.suffix == ".pyc" or path.name == ".DS_Store"


def renderer_source_paths(template_root: Path) -> tuple[Path, ...]:
    """Return the complete path-bound renderer implementation closure.

    The closure includes every tracked-style source file under
    ``infrastructure/rendering`` (Python, Lua, CSS, templates, and local
    references), the stage-03 render entry point, the linked-project
    orchestration component, and the template environment's lock and package
    metadata. Generated interpreter cache files and symlink escapes are
    rejected rather than silently changing the identity of the renderer being
    witnessed.
    """
    root = _require_template_root(template_root)
    renderer = root / "infrastructure" / "rendering"
    if not renderer.is_dir():
        raise FileNotFoundError(f"sibling renderer source directory is missing: {renderer}")
    paths: list[Path] = []
    for candidate in renderer.rglob("*"):
        if _is_generated_renderer_path(candidate) or not candidate.is_file():
            continue
        if candidate.is_symlink():
            raise ValueError(f"renderer source must not be a symlink: {candidate}")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"renderer source escapes the template checkout: {candidate}") from exc
        paths.append(resolved)
    orchestration = root / _RENDERER_ORCHESTRATION_DIRECTORY
    if not orchestration.is_dir():
        raise FileNotFoundError(
            "sibling renderer orchestration directory is missing: "
            f"{orchestration}"
        )
    for candidate in orchestration.rglob("*"):
        if _is_generated_renderer_path(candidate) or not candidate.is_file():
            continue
        if candidate.is_symlink():
            raise ValueError(f"renderer orchestration source must not be a symlink: {candidate}")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"renderer orchestration source escapes the template checkout: {candidate}"
            ) from exc
        paths.append(resolved)
    for relative in _RENDERER_ROOT_INPUTS:
        path = require_file(root / relative, f"sibling renderer input {relative}")
        if path.is_symlink():
            raise ValueError(f"renderer root input must not be a symlink: {path}")
        paths.append(path.resolve())
    unique = {path.resolve() for path in paths}
    if not unique:
        raise ValueError("sibling renderer source closure is empty")
    return tuple(sorted(unique, key=lambda path: path.relative_to(root).as_posix()))


def _path_bound_digest(root: Path, paths: tuple[Path, ...]) -> str:
    """Hash source bytes together with their checkout-relative paths."""
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _git_bytes(template_root: Path, *arguments: str) -> bytes:
    """Run one bounded Git query and turn failures into release-gate errors."""
    try:
        result = subprocess.run(
            ["git", "-C", str(template_root), *arguments],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except OSError as exc:
        raise ValueError("Git is unavailable; cannot establish renderer provenance") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            "sibling renderer checkout is not a readable Git worktree: "
            f"{detail or template_root}"
        )
    return result.stdout


def _git_provenance(template_root: Path) -> dict[str, Any]:
    """Capture the revision and a path-private exact dirty-worktree witness."""
    head = _git_bytes(template_root, "rev-parse", "--verify", "HEAD").decode(
        "ascii", errors="strict"
    ).strip()
    if _GIT_HEAD_RE.fullmatch(head) is None:
        raise ValueError("sibling renderer Git HEAD is not an object identifier")
    status = _git_bytes(
        template_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    # A porcelain status alone distinguishes paths and index states but not
    # two different edits to the same modified file. Bind the binary diff for
    # tracked changes and path-bound bytes for untracked files, too. This keeps
    # the receipt path-private while making its dirty-state digest an exact
    # content witness rather than merely a Boolean/status snapshot.
    tracked_diff = _git_bytes(template_root, "diff", "--binary", "HEAD")
    untracked_names = _git_bytes(
        template_root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    untracked_digest = hashlib.sha256()
    root = Path(template_root).resolve()
    for encoded_relative in (item for item in untracked_names.split(b"\0") if item):
        relative = Path(encoded_relative.decode("utf-8", errors="surrogateescape"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("sibling renderer has an unsafe untracked path")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("sibling renderer untracked path escapes its checkout") from exc
        untracked_digest.update(len(encoded_relative).to_bytes(4, "big"))
        untracked_digest.update(encoded_relative)
        untracked_digest.update(
            bytes.fromhex(sha256_file(require_file(path, "untracked renderer input")))
        )
    dirty_material = (
        status
        + b"\0tracked-diff\0"
        + tracked_diff
        + b"\0untracked-content\0"
        + untracked_digest.digest()
    )
    return {
        "head": head,
        "dirty": bool(status),
        "dirty_entry_count": status.count(b"\0"),
        "dirty_state_sha256": hashlib.sha256(dirty_material).hexdigest(),
    }


def _tool_provenance(name: str, command: tuple[str, ...]) -> dict[str, Any]:
    """Record version identity for one optional rendering executable."""
    if shutil.which(command[0]) is None:
        return {"available": False}
    try:
        result = subprocess.run(
            list(command), check=False, capture_output=True, timeout=30
        )
    except OSError as exc:
        raise ValueError(f"renderer tool {name!r} is present but cannot be queried") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            f"renderer tool {name!r} failed its version query: {detail or result.returncode}"
        )
    version_bytes = result.stdout + result.stderr
    lines = version_bytes.decode("utf-8", errors="replace").splitlines()
    first_line = next((line.strip() for line in lines if line.strip()), "")
    if not first_line:
        raise ValueError(f"renderer tool {name!r} returned no version identity")
    return {
        "available": True,
        "version": first_line,
        "version_sha256": hashlib.sha256(version_bytes).hexdigest(),
    }


def _rendered_artifact_hashes(output_root: Path) -> dict[str, str]:
    """Require the physical PDF and HTML render before writing a receipt."""
    output = Path(output_root).resolve()
    return {
        name: sha256_file(require_file(output / relative, f"rendered {name}"))
        for name, relative in _RENDERED_ARTIFACTS.items()
    }


def capture_renderer_provenance(
    *,
    project_root: Path,
    output_root: Path,
    template_root: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic receipt for one already-rendered publication bundle.

    ``template_root`` is injectable only for isolated verification.  Production
    callers use the conventional sibling checkout.  No absolute local path is
    serialized, so the receipt describes inputs rather than claiming another
    checkout can reproduce a render without its own verified sibling renderer.
    """
    project = Path(project_root).resolve()
    output = require_safe_output_root(output_root, project_root=project)
    template = _require_template_root(
        default_template_root(project) if template_root is None else template_root
    )
    source_paths = renderer_source_paths(template)
    source_files = [
        {
            "path": path.relative_to(template).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in source_paths
    ]
    return {
        "schema_version": RENDERER_PROVENANCE_SCHEMA_VERSION,
        "receipt_type": "renderer_provenance",
        "template": {
            "checkout": "sibling_template",
            "git": _git_provenance(template),
            "renderer_source_tree_sha256": _path_bound_digest(template, source_paths),
            "renderer_source_files": source_files,
        },
        "toolchain": {
            name: _tool_provenance(name, command)
            for name, command in _TOOL_COMMANDS
        },
        "rendered_artifacts": _rendered_artifact_hashes(output),
    }


def renderer_provenance_path(output_root: Path) -> Path:
    """Return the fixed receipt location inside one artifact root."""
    return Path(output_root).resolve() / RENDERER_PROVENANCE_RELATIVE_PATH


def write_renderer_provenance(
    *,
    project_root: Path,
    output_root: Path,
    template_root: Path | None = None,
) -> dict[str, Any]:
    """Atomically persist a post-render renderer receipt in a safe output root."""
    project = Path(project_root).resolve()
    output = require_safe_output_root(output_root, project_root=project)
    receipt = capture_renderer_provenance(
        project_root=project, output_root=output, template_root=template_root
    )
    destination = renderer_provenance_path(output)
    atomic_write_json(destination, receipt)
    return {"path": destination, "receipt": receipt}


def _require_receipt_shape(receipt: Mapping[str, Any]) -> None:
    """Reject malformed receipts before exact live comparison."""
    if receipt.get("schema_version") != RENDERER_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("renderer provenance receipt schema version is invalid")
    if receipt.get("receipt_type") != "renderer_provenance":
        raise ValueError("renderer provenance receipt type is invalid")
    template = receipt.get("template")
    if not isinstance(template, Mapping) or template.get("checkout") != "sibling_template":
        raise ValueError("renderer provenance receipt template identity is invalid")
    git = template.get("git")
    if not isinstance(git, Mapping) or _GIT_HEAD_RE.fullmatch(str(git.get("head", ""))) is None:
        raise ValueError("renderer provenance receipt Git identity is invalid")
    if not isinstance(git.get("dirty"), bool):
        raise ValueError("renderer provenance receipt dirty-state flag is invalid")
    for name in ("dirty_entry_count",):
        if (
            not isinstance(git.get(name), int)
            or isinstance(git.get(name), bool)
            or int(git[name]) < 0
        ):
            raise ValueError("renderer provenance receipt dirty-state count is invalid")
    for name in ("dirty_state_sha256", "renderer_source_tree_sha256"):
        value = git.get(name) if name in git else template.get(name)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"renderer provenance receipt {name} is invalid")
    files = template.get("renderer_source_files")
    if not isinstance(files, list) or not files:
        raise ValueError("renderer provenance receipt source-file closure is invalid")
    seen_paths: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("renderer provenance receipt source-file entry is invalid")
        path = item.get("path")
        digest = item.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or path in seen_paths
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("renderer provenance receipt source-file entry is invalid")
        seen_paths.add(path)
    if not isinstance(receipt.get("toolchain"), Mapping):
        raise ValueError("renderer provenance receipt toolchain is invalid")
    rendered = receipt.get("rendered_artifacts")
    if not isinstance(rendered, Mapping):
        raise ValueError("renderer provenance receipt rendered-artifact hashes are invalid")
    for name in _RENDERED_ARTIFACTS:
        value = rendered.get(name)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"renderer provenance receipt rendered {name} digest is invalid")


def validate_renderer_provenance(
    *,
    project_root: Path,
    output_root: Path,
    template_root: Path | None = None,
) -> dict[str, Any]:
    """Fail unless a persisted receipt exactly matches live render inputs.

    A missing sibling template is an explicit validation failure, not a claim
    that the render is portable or renderer-independent.
    """
    project = Path(project_root).resolve()
    output = Path(output_root).resolve()
    path = require_file(renderer_provenance_path(output), "renderer provenance receipt")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("renderer provenance receipt is not valid JSON") from exc
    if not isinstance(receipt, Mapping):
        raise ValueError("renderer provenance receipt must be a JSON object")
    _require_receipt_shape(receipt)
    live = capture_renderer_provenance(
        project_root=project,
        output_root=output,
        template_root=template_root,
    )
    if dict(receipt) != live:
        raise ValueError(
            "renderer provenance receipt does not match the live sibling template, "
            "toolchain, or rendered artifacts"
        )
    template = receipt["template"]
    git = template["git"]
    return {
        "receipt_path": str(path),
        "receipt_sha256": sha256_file(path),
        "template_git_head": str(git["head"]),
        "template_dirty": bool(git["dirty"]),
        "renderer_source_tree_sha256": str(template["renderer_source_tree_sha256"]),
        "rendered_artifact_count": len(_RENDERED_ARTIFACTS),
    }


__all__ = [
    "RENDERER_PROVENANCE_RELATIVE_PATH",
    "RENDERER_PROVENANCE_SCHEMA_VERSION",
    "capture_renderer_provenance",
    "default_template_root",
    "renderer_provenance_path",
    "renderer_source_paths",
    "validate_renderer_provenance",
    "write_renderer_provenance",
]
