"""Offline benchmark loading and provenance checks.

Network retrieval belongs in ``scripts/fetch_benchmarks.py``.  Scientific code
accepts only already-downloaded files and an explicit checksum-bearing spec.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import numpy as np


@dataclass(frozen=True)
class BenchmarkSpec:
    """Source and interpretation metadata for one benchmark fixture."""

    benchmark_id: str
    source: str
    license: str
    sha256: str
    claim_role: str = "external_stress_test_only"
    doi: str = ""
    retrieval_date: str = ""
    schema: str = ""
    preprocessing: str = ""
    format: str = ""

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.source or not self.license or not self.claim_role:
            raise ValueError("benchmark metadata fields must be non-empty")
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("benchmark sha256 must be a lowercase SHA-256 digest")
        if self.retrieval_date:
            try:
                from datetime import date

                date.fromisoformat(self.retrieval_date)
            except ValueError as exc:
                raise ValueError("retrieval_date must be ISO date text") from exc

    def to_mapping(self) -> dict[str, str]:
        """Return provenance as a stable, hashable contract."""
        return {
            "benchmark_id": self.benchmark_id,
            "source": self.source,
            "license": self.license,
            "sha256": self.sha256,
            "claim_role": self.claim_role,
            "doi": self.doi,
            "retrieval_date": self.retrieval_date,
            "schema": self.schema,
            "preprocessing": self.preprocessing,
            "format": self.format,
        }


@dataclass(frozen=True)
class BenchmarkDataset:
    """Validated numerical benchmark payload."""

    benchmark_id: str
    values: np.ndarray
    columns: tuple[str, ...]
    spec: BenchmarkSpec
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.columns) or values.shape[0] < 1:
            raise ValueError("benchmark values must be a non-empty 2-D matrix aligned to columns")
        if not np.all(np.isfinite(values)):
            raise ValueError("benchmark values must be finite")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class ChronologicalBenchmarkSplit:
    """Subject-disjoint chronological train/evaluation partition."""

    train: BenchmarkDataset
    evaluation: BenchmarkDataset
    train_subjects: tuple[str, ...]
    evaluation_subjects: tuple[str, ...]
    split_hash: str

    def __post_init__(self) -> None:
        if not self.train_subjects or not self.evaluation_subjects:
            raise ValueError("chronological split requires non-empty train and evaluation subjects")
        if set(self.train_subjects) & set(self.evaluation_subjects):
            raise ValueError("chronological split subjects must be disjoint")
        if re.fullmatch(r"[0-9a-f]{64}", self.split_hash) is None:
            raise ValueError("split_hash must be a lowercase SHA-256 digest")

    def to_mapping(self) -> dict[str, object]:
        return {
            "train_subjects": list(self.train_subjects),
            "evaluation_subjects": list(self.evaluation_subjects),
            "split_hash": self.split_hash,
            "train_rows": int(self.train.values.shape[0]),
            "evaluation_rows": int(self.evaluation.values.shape[0]),
            "subject_disjoint": True,
            "chronological": True,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv_benchmark(path: Path, spec: BenchmarkSpec, delimiter: str = ",") -> BenchmarkDataset:
    """Load a numeric CSV fixture after verifying its declared checksum."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != spec.sha256:
        raise ValueError(f"checksum mismatch for benchmark {spec.benchmark_id!r}")
    if path.stat().st_size == 0:
        raise ValueError("benchmark CSV must contain a header and at least one row")
    raw = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=float)
    if raw.dtype.names is None or raw.size == 0:
        raise ValueError("benchmark CSV must contain a header and at least one row")
    columns = tuple(str(name) for name in raw.dtype.names)
    values = np.column_stack([np.asarray(raw[name], dtype=float) for name in columns])
    if not np.all(np.isfinite(values)):
        raise ValueError("benchmark values must be finite")
    return BenchmarkDataset(spec.benchmark_id, values, columns, spec)


def load_agent_trace(
    path: Path,
    spec: BenchmarkSpec,
    delimiter: str = ",",
    *,
    allow_missing_observations: bool = False,
) -> BenchmarkDataset:
    """Load a chronological action/observation trace with categorical IDs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != spec.sha256:
        raise ValueError(f"checksum mismatch for benchmark {spec.benchmark_id!r}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        names = tuple(str(name) for name in (reader.fieldnames or ()))
        rows = list(reader)
    required = {"subject", "time", "action", "observation"}
    if not required.issubset(names):
        raise ValueError(f"agent trace must contain columns {sorted(required)}")
    if not rows:
        raise ValueError("agent trace must contain at least one row")
    subjects_raw: list[str] = []
    actions_raw: list[str] = []
    times_raw: list[float] = []
    observations_raw: list[float] = []
    missing_tokens = {"", "nan", "none", "null", "missing"}
    for row in rows:
        fields = {name: str(row.get(name, "")).strip() for name in required}
        if any(value.lower() in missing_tokens for name, value in fields.items() if name != "observation"):
            raise ValueError("agent trace required fields must not be missing")
        observation_missing = fields["observation"].lower() in missing_tokens
        if observation_missing and not allow_missing_observations:
            raise ValueError("agent trace required fields must not be missing")
        try:
            time_value = float(fields["time"])
            observation_value = np.nan if observation_missing else float(fields["observation"])
        except ValueError as exc:
            raise ValueError("agent trace time and observation fields must be numeric") from exc
        subjects_raw.append(fields["subject"])
        actions_raw.append(fields["action"])
        times_raw.append(time_value)
        observations_raw.append(observation_value)
    subjects = np.asarray(subjects_raw, dtype=str)
    actions = np.asarray(actions_raw, dtype=str)
    times = np.asarray(times_raw, dtype=float)
    observations = np.asarray(observations_raw, dtype=float)
    if times.size == 0 or not np.all(np.isfinite(times)) or not np.all(
        np.isin(observations, (-1.0, 0.0, 1.0)) | np.isnan(observations)
    ):
        raise ValueError("agent trace times must be finite and observations must be binary or missing")
    if np.any(np.isnan(observations)):
        if not allow_missing_observations:
            raise ValueError("agent trace observations must not be missing")
        observations = np.where(np.isnan(observations), -1.0, observations)
    for subject in dict.fromkeys(subjects.tolist()):
        subject_times = times[subjects == subject]
        if subject_times.size > 1 and not np.all(np.diff(subject_times) > 0.0):
            raise ValueError("agent trace must be strictly chronological within each subject")
    metadata = {
        "subjects": subjects.tolist(),
        "actions": actions.tolist(),
        "n_subjects": int(np.unique(subjects).size),
        "missingness": "missing observations encoded as -1" if allow_missing_observations else "required fields validated and missing rows rejected",
        "missing_observation_count": int(np.count_nonzero(observations < 0.0)),
        "chronological": True,
    }
    return BenchmarkDataset(
        spec.benchmark_id,
        np.column_stack((times, observations)),
        ("time", "observation"),
        spec,
        metadata,
    )


def chronological_subject_split(
    dataset: BenchmarkDataset,
    train_fraction: float = 0.5,
) -> ChronologicalBenchmarkSplit:
    """Split a loaded agent trace by subject in first-observation order.

    Subjects never cross the train/evaluation boundary.  The ordering is
    deterministic by each subject's first time and then subject identifier;
    this is a reproducible chronological stress-test partition, not a claim of
    exchangeability or causal generalization.
    """
    if not 0.0 < float(train_fraction) < 1.0 or not np.isfinite(float(train_fraction)):
        raise ValueError("train_fraction must lie in (0, 1)")
    metadata = dict(dataset.metadata or {})
    subjects = np.asarray(metadata.get("subjects", ()), dtype=str)
    actions = np.asarray(metadata.get("actions", ()), dtype=str)
    if subjects.shape != (dataset.values.shape[0],) or actions.shape != subjects.shape:
        raise ValueError("agent trace metadata must retain subject and action rows for splitting")
    unique_subjects = list(dict.fromkeys(subjects.tolist()))
    if len(unique_subjects) < 2:
        raise ValueError("chronological subject split requires at least two subjects")
    first_times = {
        subject: float(np.min(dataset.values[subjects == subject, 0]))
        for subject in unique_subjects
    }
    ordered = tuple(sorted(unique_subjects, key=lambda subject: (first_times[subject], subject)))
    n_train = min(len(ordered) - 1, max(1, int(np.floor(len(ordered) * float(train_fraction)))))
    train_subjects = ordered[:n_train]
    evaluation_subjects = ordered[n_train:]
    train_mask = np.isin(subjects, train_subjects)
    evaluation_mask = np.isin(subjects, evaluation_subjects)
    def subset(mask: np.ndarray, role: str) -> BenchmarkDataset:
        subset_metadata = dict(metadata)
        subset_metadata.update({
            "subjects": subjects[mask].tolist(),
            "actions": actions[mask].tolist(),
            "split_role": role,
            "subject_disjoint": True,
            "chronological": True,
        })
        return BenchmarkDataset(
            dataset.benchmark_id,
            dataset.values[mask],
            dataset.columns,
            dataset.spec,
            subset_metadata,
        )
    payload = {
        "benchmark_id": dataset.benchmark_id,
        "source_sha256": dataset.spec.sha256,
        "train_subjects": list(train_subjects),
        "evaluation_subjects": list(evaluation_subjects),
        "train_rows": np.flatnonzero(train_mask).tolist(),
        "evaluation_rows": np.flatnonzero(evaluation_mask).tolist(),
    }
    split_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ChronologicalBenchmarkSplit(
        subset(train_mask, "train"),
        subset(evaluation_mask, "evaluation"),
        train_subjects,
        evaluation_subjects,
        split_hash,
    )


def load_normalized_golub_fixture(path: Path, spec: BenchmarkSpec) -> BenchmarkDataset:
    """Load the committed numeric Golub normalization fixture offline.

    The fixture is intentionally a compact, normalized projection of the
    source ExpressionSet.  It is suitable for adapter and preprocessing tests
    and stress comparisons; it does not substitute for the full Bioconductor
    archive or establish biological ground truth.
    """
    dataset = load_csv_benchmark(path, spec)
    required = {"sample", "feature", "class", "expression"}
    if not required.issubset(dataset.columns):
        raise ValueError(f"normalized Golub fixture must contain columns {sorted(required)}")
    if not np.all(np.isin(dataset.values[:, dataset.columns.index("class")], (0.0, 1.0))):
        raise ValueError("normalized Golub fixture class must be binary")
    metadata = dict(dataset.metadata or {})
    metadata.update({
        "normalized": True,
        "preprocessing": spec.preprocessing or "offline normalized fixture",
        "claim_role": spec.claim_role,
        "full_source_required_for_biological_claims": True,
    })
    return BenchmarkDataset(dataset.benchmark_id, dataset.values, dataset.columns, spec, metadata)


__all__ = [
    "BenchmarkSpec", "BenchmarkDataset", "ChronologicalBenchmarkSplit",
    "sha256_file", "load_csv_benchmark", "load_normalized_golub_fixture",
    "load_agent_trace", "chronological_subject_split",
]
