"""Atomic, hashable artifact utilities for research orchestration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def require_safe_output_root(output_root: Path, *, project_root: Path) -> Path:
    """Resolve an output root without allowing it to overlap authored sources.

    Project-local generated artifacts may live in ``output/`` (including an
    isolated candidate below it), while external absolute or project-relative
    destinations remain supported.  The canonical ``output/`` entry may resolve
    outside the checkout, but it may not be a symlink into a different
    project-local tree.  Every other project-local path is refused:
    a resolved alias such as ``output/../manuscript`` must never let a producer
    overwrite or clean authored files.  Ancestors of the checkout and the
    filesystem root are likewise not meaningful artifact destinations.
    """
    project = Path(project_root).expanduser().resolve()
    output = Path(output_root).expanduser()
    if not output.is_absolute():
        output = project / output
    output = output.resolve()
    declared_output = project / "output"
    canonical_output = declared_output.resolve()
    try:
        canonical_output.relative_to(project)
    except ValueError:
        # A checkout-local ``output/`` link may deliberately point at an
        # external artifact volume.  Its resolved descendants remain outside
        # the checkout and are covered by the normal namespace checks below.
        pass
    else:
        if canonical_output != declared_output:
            raise ValueError(
                "output_root canonical project output/ must not resolve into "
                f"a different project tree: {canonical_output}"
            )
    if output == output.parent:
        raise ValueError("output_root must be a concrete directory, not the filesystem root")
    if output == project:
        raise ValueError("output_root must not be the project root")
    try:
        output.relative_to(canonical_output)
    except ValueError:
        pass
    else:
        return output
    try:
        relative = output.relative_to(project)
    except ValueError:
        relative = None
    if relative is not None:
        raise ValueError(
            "output_root must be within project output/ or outside the project; "
            f"refusing project-local path: {output}"
        )
    try:
        project.relative_to(output)
    except ValueError:
        return output
    raise ValueError("output_root must not contain the project checkout")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file in bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Replace ``path`` atomically after fully writing and flushing content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    """Serialize finite standard JSON deterministically and replace the target atomically."""
    atomic_write_text(
        Path(path),
        json.dumps(value, indent=2, sort_keys=True, default=float, allow_nan=False) + "\n",
    )


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Replace a binary artifact atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def artifact_record(
    path: Path,
    root: Path,
    role: str,
    *,
    root_label: str = "project",
) -> dict[str, Any]:
    """Return a manifest record with an explicit path namespace.

    ``project`` keeps the established project-relative paths.  ``output`` is
    used for an output directory supplied outside the checkout, so a manifest
    remains portable without unsafe ``..`` components or absolute paths.
    """
    path = Path(path)
    root = Path(root).resolve()
    if root_label not in {"project", "output"}:
        raise ValueError("root_label must be 'project' or 'output'")
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        relative = path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact path is outside its declared root: {path}") from exc
    return {
        "root": root_label,
        "path": str(relative),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "role": role,
    }


def refresh_renderer_artifact_records(
    manifest: dict[str, Any], output_root: Path, project_root: Path
) -> None:
    """Refresh only hashes for artifacts rewritten by the publication renderer.

    Scientific records remain immutable at this boundary. Renderer-owned
    records may legitimately change when the sibling template regenerates
    HTML, combined markdown, or PDF files. Quality evidence is deliberately
    excluded: it must be validated before hydration and remain byte-stable
    through terminal finalization.
    """
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("manifest artifacts must be a mapping")
    output_root = Path(output_root).resolve()
    project_root = Path(project_root).resolve()
    for name, record in list(artifacts.items()):
        if not isinstance(record, dict):
            if name.startswith("web:") or name.startswith("rendered_"):
                raise ValueError(f"renderer artifact record is malformed: {name}")
            continue
        if not (name.startswith("web:") or name.startswith("rendered_")):
            continue
        root_label = str(record.get("root", "project"))
        if root_label not in {"project", "output"}:
            raise ValueError(f"renderer artifact root is invalid: {name}")
        root = project_root if root_label == "project" else output_root
        path = (root / str(record["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"renderer artifact escapes its namespace: {name}") from exc
        if not path.is_file():
            raise ValueError(f"renderer artifact is missing: {path}")
        artifacts[name] = artifact_record(
            path,
            root,
            str(record.get("role", "rendered_artifact")),
            root_label=root_label,
        )


def invalidate_renderer_artifact_records(manifest: dict[str, Any]) -> list[str]:
    """Remove renderer-owned records after a scientific/render input change.

    A byte-valid old PDF or HTML is not evidence for newly hydrated manuscript
    input or regenerated figures.  Callers retain the physical files for
    inspection, but must render again before those files can be recorded in a
    completed release manifest.
    """
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("manifest artifacts must be a mapping")
    removed: list[str] = []
    for name, record in list(artifacts.items()):
        renderer_owned = (
            name.startswith("web:")
            or name.startswith("rendered_")
            or name in {"release_review", "renderer_provenance"}
        )
        if isinstance(record, dict):
            renderer_owned = renderer_owned or record.get("role") in {
                "rendered_pdf",
                "rendered_html",
                "release_review_receipt",
                "renderer_provenance_receipt",
            }
        if renderer_owned:
            artifacts.pop(name)
            removed.append(name)
    return sorted(removed)


__all__ = [
    "require_safe_output_root",
    "sha256_file",
    "atomic_write_text",
    "atomic_write_json",
    "atomic_write_bytes",
    "artifact_record",
    "invalidate_renderer_artifact_records",
    "refresh_renderer_artifact_records",
]
