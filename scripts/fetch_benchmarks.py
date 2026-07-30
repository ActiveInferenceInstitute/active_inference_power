#!/usr/bin/env python3
"""Explicitly fetch checksum-pinned benchmark archives.

This command is never called by ordinary tests or result generation.  It is an
operator action because external data are evidence provenance, not dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from urllib.request import urlopen

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.artifacts import atomic_write_bytes  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_id")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "benchmarks" / "manifest.yaml")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "benchmarks" / "raw")
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    records = {str(item["id"]): item for item in manifest["benchmarks"]}
    if args.benchmark_id not in records:
        raise SystemExit(f"unknown benchmark id: {args.benchmark_id}")
    record = records[args.benchmark_id]
    if not str(record.get("source", "")).startswith("http"):
        raise SystemExit("benchmark source is not a downloadable URL")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / str(record.get("filename", f"{args.benchmark_id}.download"))
    with urlopen(str(record["source"]), timeout=60) as response:
        atomic_write_bytes(target, response.read())
    observed = _sha256(target)
    if observed != record["sha256"]:
        target.unlink(missing_ok=True)
        raise SystemExit(f"checksum mismatch: expected {record['sha256']}, got {observed}")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
