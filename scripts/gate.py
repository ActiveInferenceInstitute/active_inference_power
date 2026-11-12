#!/usr/bin/env python3
"""Run the same gate arms CI invokes through ./run.sh; never synchronize implicitly."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv/bin/python")
PATHS = ("src", "scripts", "tests", "examples")
COMMANDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "lint": (
        (PYTHON, "-m", "compileall", "-q", *PATHS),
        (PYTHON, "-m", "ruff", "check", "--no-respect-gitignore", *PATHS),
    ),
    "types": ((PYTHON, "-m", "mypy", *PATHS, "--ignore-missing-imports"),),
    "test-fast": ((PYTHON, "-m", "pytest", "tests/", "-m", "not slow", "-q"),),
    "test-full": ((PYTHON, "scripts/generate_quality_report.py"),),
    "docs": (
        (PYTHON, "scripts/audit_documentation_links.py"),
        (PYTHON, "scripts/measure_gates.py", "--check"),
    ),
    "gnn": (
        (PYTHON, "scripts/validate_gnn_specs.py"),
        (PYTHON, "-m", "pytest", "tests/test_gnn_specs.py", "-q"),
    ),
}
GROUPS = {
    "fast": ("lint", "types", "test-fast", "docs"),
    "all": ("lint", "types", "test-full", "docs", "gnn", "package"),
}


def call(argv: tuple[str, ...], *, env: dict[str, str] | None = None) -> None:
    print(f"$ {shlex.join(argv)}", flush=True)
    subprocess.run(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, check=True)


def package() -> None:
    """Build once, install the actual wheel, and import outside the source tree."""
    with tempfile.TemporaryDirectory(prefix="aip-wheel-") as directory:
        root = Path(directory)
        call(("uv", "build", "--python", PYTHON, "--out-dir", str(root / "dist")))
        wheels = list((root / "dist").glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"expected one wheel, found {len(wheels)}")
        interpreter = str(root / "venv/bin/python")
        call(("uv", "venv", "--python", PYTHON, str(root / "venv")))
        constraints = str(root / "locked-runtime.txt")
        call(("uv", "export", "--locked", "--no-dev", "--no-hashes", "--no-emit-project",
              "--output-file", constraints))
        call(("uv", "pip", "install", "--python", interpreter, "--constraint", constraints, str(wheels[0])))
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        subprocess.run(
            (interpreter, "-I", "-c", "import active_inference_power as a; print(a.PowerScenario.__name__)"),
            cwd=root, env=env, stdin=subprocess.DEVNULL, check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="*", metavar="ARM")
    parser.add_argument("--list", action="store_true", help="print the exact plan without running")
    args = parser.parse_args()
    selected: list[str] = []
    for arm in args.arms or ["fast"]:
        if arm not in COMMANDS and arm not in GROUPS and arm != "package":
            parser.error(f"unknown arm {arm!r}; choose {', '.join((*COMMANDS, 'package', *GROUPS))}")
        for member in GROUPS.get(arm, (arm,)):
            if member not in selected:
                selected.append(member)
    if args.list:
        for arm in selected:
            print(f"{arm}:")
            for command in COMMANDS.get(arm, ()):
                print(f"  {shlex.join(command)}")
            if arm == "package":
                print("  uv build; uv venv; uv pip install wheel; isolated installed import")
        return 0
    for arm in selected:
        started = time.perf_counter()
        if arm == "package":
            package()
        else:
            env = dict(os.environ)
            if arm == "gnn":
                # Always explicit: unavailable upstream is a failure, never a passing skip.
                key = "ACTIVE_INFERENCE_POWER_GNN_ROOT"
                root = Path(env.get(key, str(ROOT.parent / "Generalized_Notation_Notation")))
                if not str(env.get(key, "default")).strip():
                    raise ValueError(f"{key} must not be empty")
                if not (root / ".venv/bin/python").is_file():
                    raise ValueError(f"provisioned pinned GNN checkout unavailable at {root}; set {key}")
                env[key] = str(root.resolve())
            for command in COMMANDS[arm]:
                call(command, env=env)
        print(f"gate: {arm} passed in {time.perf_counter() - started:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"gate: {exc}", file=sys.stderr)
        raise SystemExit(1)
