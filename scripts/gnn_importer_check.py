#!/usr/bin/env python3
"""Run the shared GNN importer reader against the pinned upstream revision.

``python -m gnn.importers check --repo .`` is the estate's single reader for an
importer's pin, its ``.github/GNN.md`` pointer and its ``*.gnn.md``
specifications. This repository calls it instead of carrying a reader of its
own: the command runs under the interpreter the ``gnn`` arm already provisions
(the pinned checkout's ``.venv``), with the pinned commit's ``src/gnn``
extracted from Git onto ``PYTHONPATH``, so both the reader and the parser it
validates with are the pinned release's bytes rather than the checkout's live
working tree.

The arm fails closed: an unavailable checkout or an absent pinned commit is an
error, never a silent skip.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from active_inference_power.gnn_specs import (  # noqa: E402
    ENV_UPSTREAM_ROOT,
    checkout_available,
    extract_pinned_src,
    gnn_checkout_python,
    pinned_revision,
    resolve_gnn_root,
)


def main() -> int:
    root = resolve_gnn_root(ROOT)
    if not checkout_available(root):
        print(
            f"gate: provisioned pinned GNN checkout unavailable at {root}; "
            f"set {ENV_UPSTREAM_ROOT}",
            file=sys.stderr,
        )
        return 1
    revision = pinned_revision(ROOT)
    with tempfile.TemporaryDirectory(prefix="aip-gnn-pin-") as directory:
        src = extract_pinned_src(root, revision.commit, Path(directory))
        command = (
            str(gnn_checkout_python(root)),
            "-m",
            "gnn.importers",
            "check",
            "--repo",
            str(ROOT),
        )
        print(f"$ PYTHONPATH={src} {' '.join(command)}", flush=True)
        return subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(src)},
            stdin=subprocess.DEVNULL,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
