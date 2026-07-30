#!/usr/bin/env python3
"""Write one isolated, non-promotional three-way Storey method-review bundle.

The command intentionally has no tuning arguments.  It writes the frozen
default review only when an explicit, new output root is supplied.  The target
contains exactly the hash-bound ``storey_method_review.json`` and
``manifest.json`` artifacts at its direct root; it is not a release, quality,
manuscript, or certificate-promotion producer.
"""

from __future__ import annotations

# Imports below follow the script-local path bootstrap so direct invocation is
# equivalent to invoking this script from the project checkout.
# ruff: noqa: E402

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from active_inference_power.storey_method_review import (  # noqa: E402
    STOREY_METHOD_REVIEW_STATUS,
    evaluate_storey_method_review,
    validate_storey_method_review,
)
from scripts.artifacts import (  # noqa: E402
    artifact_record,
    atomic_write_json,
    require_safe_output_root,
    sha256_file,
)


_SOURCE_PATHS = {
    "orchestrator": Path(__file__),
    "storey_method_review": ROOT / "src" / "active_inference_power" / "storey_method_review.py",
    "corrections": ROOT / "src" / "active_inference_power" / "corrections.py",
    "simulation": ROOT / "src" / "active_inference_power" / "simulation.py",
    "error_rates": ROOT / "src" / "active_inference_power" / "error_rates.py",
    "uncertainty": ROOT / "src" / "active_inference_power" / "uncertainty.py",
}


def _validate_output_root(output_root: Path) -> Path:
    """Require a concrete, new destination outside the canonical output root."""
    target = Path(output_root).expanduser()
    if not target.is_absolute():
        target = ROOT / target
    target = require_safe_output_root(target, project_root=ROOT)
    project = ROOT.resolve()
    canonical_output = (ROOT / "output").resolve()
    if target == project or target == canonical_output:
        raise ValueError(
            "Storey method review requires a new isolated --output-root, not the project output root"
        )
    if target == target.parent:
        raise ValueError("Storey method review output root must be a concrete directory")
    if target.exists():
        raise FileExistsError(
            f"Storey method review output root already exists: {target}; choose a new isolated directory"
        )
    return target


def _source_hashes() -> dict[str, str]:
    """Hash every source file required to reproduce the review bundle."""
    missing = [name for name, path in _SOURCE_PATHS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Storey method review source files are missing: {missing}")
    return {name: sha256_file(path) for name, path in _SOURCE_PATHS.items()}


def _manifest(
    staging_root: Path,
    review: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Return the final direct-root manifest after review JSON is staged."""
    review_path = staging_root / "storey_method_review.json"
    return {
        "schema_version": "1.0",
        "kind": "three_way_storey_method_review_manifest",
        "status": STOREY_METHOD_REVIEW_STATUS,
        "review_result_sha256": review["result_sha256"],
        "plan_sha256": review["plan_sha256"],
        "schedule_sha256": review["schedule_sha256"],
        "source_hashes": dict(source_hashes),
        "claim_boundary": review["claim_boundary"],
        "nonclaims": review["nonclaims"],
        "artifacts": {
            "storey_method_review": artifact_record(
                review_path,
                staging_root,
                "storey_method_review",
                root_label="output",
            )
        },
    }


def run_storey_method_review(output_root: Path) -> dict[str, str]:
    """Atomically generate the review at one explicit, previously absent root.

    This writer always invokes the frozen default plan and offers no
    value-level overrides.
    """
    target = _validate_output_root(Path(output_root))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.storey-method-review-", dir=target.parent
        )
    )
    try:
        review = evaluate_storey_method_review()
        validate_storey_method_review(review)
        source_hashes = _source_hashes()
        review_path = staging_root / "storey_method_review.json"
        atomic_write_json(review_path, review)
        manifest_path = staging_root / "manifest.json"
        atomic_write_json(manifest_path, _manifest(staging_root, review, source_hashes))
        if target.exists():
            raise FileExistsError(
                f"Storey method review output root was created concurrently: {target}"
            )
        os.replace(staging_root, target)
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    final_review_path = target / "storey_method_review.json"
    final_manifest_path = target / "manifest.json"
    return {
        "output_root": str(target),
        "review_path": str(final_review_path),
        "manifest_path": str(final_manifest_path),
        "review_sha256": sha256_file(final_review_path),
        "manifest_sha256": sha256_file(final_manifest_path),
        "result_sha256": str(review["result_sha256"]),
        "status": STOREY_METHOD_REVIEW_STATUS,
    }


def _parser() -> argparse.ArgumentParser:
    """Create the intentionally non-tunable command-line boundary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="New isolated directory for the frozen Storey method review; never defaults to output/.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write a compact receipt for the frozen review bundle."""
    args = _parser().parse_args(argv)
    receipt = run_storey_method_review(args.output_root)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main.
    raise SystemExit(main())
