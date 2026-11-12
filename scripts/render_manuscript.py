#!/usr/bin/env python3
"""Render one audited release candidate with the provisioned controlled renderer.

This command delegates typesetting to Stage 03; it does not install a renderer,
change project links, hydrate inputs, or finalize a release.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Iterator


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.renderer_provenance import (  # noqa: E402
    default_template_root,
    renderer_source_paths,
)


@contextmanager
def _defer_interrupts() -> Iterator[None]:
    """Keep a signal from interrupting a filesystem binding transition."""
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _interrupt(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    """Stop the render process group before restoring its output binding."""
    child = subprocess.Popen(
        command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        status = child.wait()
    except BaseException:
        with _defer_interrupts():
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            # Parent exit alone does not prove its descendants stopped.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        raise
    if status:
        raise subprocess.CalledProcessError(status, command)


@contextmanager
def _bind_output(output: Path, candidate: Path) -> Iterator[None]:
    """Temporarily expose the candidate; preserve backups on unexpected changes."""
    relative = candidate.relative_to(output)
    holder = Path(tempfile.mkdtemp(prefix=".output-render-bind-", dir=output.parent))
    saved = holder / "output"
    bound = saved / relative
    original_identity = (output.stat().st_dev, output.stat().st_ino)
    link_identity: tuple[int, int] | None = None
    try:
        with _defer_interrupts():
            output.rename(saved)
            output.symlink_to(bound, target_is_directory=True)
            info = output.lstat()
            link_identity = (info.st_dev, info.st_ino)
        yield
    finally:
        with _defer_interrupts():
            if saved.exists():
                info = saved.stat()
                if (info.st_dev, info.st_ino) != original_identity:
                    raise RuntimeError(f"Original output identity changed; preserve and inspect {holder}")
                if output.is_symlink():
                    info = output.lstat()
                    if (
                        link_identity != (info.st_dev, info.st_ino)
                        or os.readlink(output) != str(bound)
                    ):
                        raise RuntimeError(f"Unexpected output symlink; original preserved at {saved}")
                    output.unlink()
                elif output.exists():
                    raise RuntimeError(f"Unexpected output path; original preserved at {saved}")
                saved.rename(output)
            else:
                info = output.stat()
                if output.is_symlink() or (info.st_dev, info.st_ino) != original_identity:
                    raise RuntimeError(f"Original output missing; inspect {holder}")
            holder.rmdir()


def _candidate_root(value: Path) -> Path:
    output = ROOT / "output"
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"Canonical output must be an existing real directory: {output}")
    candidate = value.expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve(strict=True)
    try:
        relative = candidate.relative_to(output)
    except ValueError as exc:
        raise ValueError(f"Candidate must be a descendant of {output}: {candidate}") from exc
    if not relative.parts or relative.parts[0] in {"campaign", "precision", "campaign-v2"}:
        raise ValueError("Select an isolated release candidate, not canonical or historical output")
    if not candidate.is_dir():
        raise ValueError(f"Candidate directory is missing: {candidate}")
    return candidate


def _renderer_root(value: Path | None) -> tuple[Path, Path]:
    conventional = default_template_root(ROOT)
    configured = value
    if configured is None and os.environ.get("AIP_RELEASE_RENDERER_ROOT"):
        configured = Path(os.environ["AIP_RELEASE_RENDERER_ROOT"])
    renderer = conventional if configured is None else configured.expanduser()
    if not renderer.is_absolute():
        renderer = ROOT / renderer
    renderer = renderer.resolve()
    if not renderer.is_dir():
        raise ValueError(
            f"Controlled renderer checkout is missing: {renderer}. Provision the compatible "
            f"renderer at {conventional}; this command never installs it or creates links."
        )
    if renderer != conventional.resolve():
        raise ValueError(
            f"Selected renderer {renderer} is not the provenance checkout {conventional}. "
            "Existing release review/finalization require that conventional checkout; "
            "an arbitrary reference template is not a release renderer."
        )
    renderer_source_paths(renderer)
    interpreter = renderer / ".venv" / "bin" / "python"
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError(f"Provisioned renderer interpreter is missing or not executable: {interpreter}")
    slot = renderer / "projects" / "working" / ROOT.name
    if not slot.is_symlink() or slot.resolve() != ROOT:
        raise ValueError(
            f"Provisioned project link must resolve to {ROOT}: {slot}. "
            "The wrapper will not create, replace, or synchronize renderer links."
        )
    for executable in ("pandoc", "xelatex"):
        if shutil.which(executable) is None:
            raise ValueError(f"Required controlled-renderer tool is missing from PATH: {executable}")
    return renderer, interpreter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Existing strictly hydrated release candidate below project output/ (required).",
    )
    parser.add_argument(
        "--renderer-root", type=Path,
        help="Provisioned checkout; overrides AIP_RELEASE_RENDERER_ROOT, then defaults to ../template.",
    )
    args = parser.parse_args()
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        renderer, interpreter = _renderer_root(args.renderer_root)
        candidate = _candidate_root(args.output_root)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TEMPLATE_SKIP_LINK_SYNC"] = "1"
        # This advisory directory lock serializes this wrapper only. All other
        # output readers/writers must be stopped by the operator before entry.
        output_descriptor = os.open(ROOT / "output", os.O_RDONLY)
        try:
            fcntl.flock(output_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Recheck after locking: another wrapper may have bound output
            # between initial admission and opening the directory descriptor.
            candidate = _candidate_root(args.output_root)
            locked = os.fstat(output_descriptor)
            current = (ROOT / "output").stat()
            if (locked.st_dev, locked.st_ino) != (current.st_dev, current.st_ino):
                raise ValueError("Canonical output changed while acquiring its render lock")
            _run(
                [
                    sys.executable, "-B", str(ROOT / "scripts" / "review_release.py"),
                    "--output-root", str(candidate), "--skip-render", "--skip-clean-install",
                ],
                cwd=ROOT, env=env,
            )
            with _bind_output(ROOT / "output", candidate):
                _run(
                    [
                        str(interpreter), "-B", "scripts/pipeline/stage_03_render.py",
                        "--project", f"working/{ROOT.name}", "--skip-manuscript-hydration",
                    ],
                    cwd=renderer, env=env,
                )
        finally:
            os.close(output_descriptor)
        print(f"Rendered candidate: {candidate}; run post-render audits, review, and finalization separately.")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Manuscript render refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
