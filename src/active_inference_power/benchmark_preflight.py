"""Offline audit of checksum-pinned external benchmark fixtures.

The preflight deliberately has no retrieval code: it reads only files already
present in a supplied benchmark directory.  It connects the committed Golub
normalized projection to its declared archive metadata, validates a staged
chronological trajectory, and returns a JSON-safe provenance dashboard payload
whose claim boundary remains explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .benchmarks import (
    BenchmarkDataset,
    BenchmarkSpec,
    ChronologicalBenchmarkSplit,
    chronological_subject_split,
    load_agent_trace,
    load_normalized_golub_fixture,
    sha256_file,
)
from .schema import validate_benchmark_manifest


__all__ = [
    "OfflineBenchmarkPreflight",
    "preflight_offline_benchmark_fixtures",
]


_GOLUB_ARCHIVE_ID = "golub_leukemia_bioconductor_1_54_0"
_GOLUB_FIXTURE_ID = "golub_normalized_offline_fixture"
_TRAJECTORY_ID = "public_agent_trace_adapter_stage_2"
_GOLUB_COLUMNS = ("sample", "feature", "class", "expression")
_GOLUB_CLAIM_BOUNDARY = (
    "The checksum-pinned compact normalized Golub fixture is an offline "
    "preprocessing and adapter stress test; it does not reproduce the full "
    "ExpressionSet, establish biological ground truth, or upgrade synthetic guarantees."
)
_TRAJECTORY_CLAIM_BOUNDARY = (
    "The staged chronological trajectory is descriptive external stress evidence; "
    "its missingness report does not identify an MCAR, MAR, or MNAR mechanism and "
    "does not establish sequential or family-level validity."
)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _required_text(record: Mapping[str, Any], field: str, record_id: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"benchmark {record_id!r} requires non-empty {field}")
    return value.strip()


def _records_by_id(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = manifest.get("benchmarks")
    if not isinstance(records, list):
        raise ValueError("benchmark manifest must contain a benchmark list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("benchmark manifest records must be mappings")
        benchmark_id = _required_text(record, "id", "unknown")
        if benchmark_id in indexed:
            raise ValueError(f"benchmark manifest has duplicate id {benchmark_id!r}")
        indexed[benchmark_id] = record
    missing = {_GOLUB_ARCHIVE_ID, _GOLUB_FIXTURE_ID, _TRAJECTORY_ID} - set(indexed)
    if missing:
        raise ValueError(f"benchmark manifest is missing required records: {sorted(missing)}")
    return indexed


def _spec(record: Mapping[str, Any]) -> BenchmarkSpec:
    benchmark_id = _required_text(record, "id", "unknown")
    return BenchmarkSpec(
        benchmark_id=benchmark_id,
        source=_required_text(record, "source", benchmark_id),
        license=_required_text(record, "license", benchmark_id),
        sha256=_required_text(record, "sha256", benchmark_id),
        claim_role=_required_text(record, "claim_role", benchmark_id),
        doi=str(record.get("doi", "")).strip(),
        retrieval_date=str(record.get("retrieval_date", "")).strip(),
        schema=str(record.get("schema", "")).strip(),
        preprocessing=str(record.get("preprocessing", "")).strip(),
        format=str(record.get("format", "")).strip(),
    )


def _safe_file(benchmark_directory: Path, record: Mapping[str, Any]) -> Path:
    record_id = _required_text(record, "id", "unknown")
    filename = _required_text(record, "filename", record_id)
    root = Path(benchmark_directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"benchmark {record_id!r} filename escapes the benchmark directory") from exc
    return candidate


def _provenance(record: Mapping[str, Any], spec: BenchmarkSpec, *, require_doi: bool) -> dict[str, object]:
    benchmark_id = spec.benchmark_id
    doi = _required_text(record, "doi", benchmark_id) if require_doi else spec.doi
    return {
        "id": benchmark_id,
        "source": spec.source,
        "license": spec.license,
        "doi": doi if doi else None,
        "doi_status": "declared" if doi else "not_declared",
        "source_revision": _required_text(record, "source_revision", benchmark_id),
        "retrieval_date": spec.retrieval_date,
        "schema": spec.schema,
        "preprocessing": spec.preprocessing,
        "format": spec.format,
        "sha256": spec.sha256,
        "claim_role": spec.claim_role,
    }


def _validate_golub_lineage(
    archive_record: Mapping[str, Any],
    fixture_record: Mapping[str, Any],
    archive_spec: BenchmarkSpec,
    fixture_spec: BenchmarkSpec,
    fixture: BenchmarkDataset,
) -> dict[str, object]:
    """Validate metadata and matrix structure for the committed conversion recipe."""
    archive_provenance = _provenance(archive_record, archive_spec, require_doi=True)
    fixture_provenance = _provenance(fixture_record, fixture_spec, require_doi=True)
    if archive_spec.source != fixture_spec.source:
        raise ValueError("Golub fixture source must match the checksum-pinned archive source")
    if archive_spec.license != fixture_spec.license or archive_spec.doi != fixture_spec.doi:
        raise ValueError("Golub fixture license and DOI must match the archive provenance")
    archive_revision = str(archive_provenance["source_revision"])
    fixture_revision = str(fixture_provenance["source_revision"])
    if not fixture_revision.startswith(f"{archive_revision}-"):
        raise ValueError("Golub fixture source_revision must derive from the archive revision")
    if archive_spec.format != "bioconductor_rda_archive" or fixture_spec.format != "csv":
        raise ValueError("Golub archive and normalized fixture formats are not the declared conversion pair")
    recipe_text = f"{fixture_spec.schema} {fixture_spec.preprocessing}".lower()
    for required_term in ("sample", "feature", "class", "expression", "normalized", "offline", "no biological claim"):
        if required_term not in recipe_text:
            raise ValueError(f"Golub normalized fixture recipe metadata is missing {required_term!r}")
    if fixture.spec.claim_role != "external_stress_test_only" or archive_spec.claim_role != "external_stress_test_only":
        raise ValueError("Golub archive and fixture must remain external stress-test evidence")
    if fixture.columns != _GOLUB_COLUMNS:
        raise ValueError(f"Golub normalized fixture columns must equal {list(_GOLUB_COLUMNS)}")

    values = fixture.values
    sample, feature, labels, expression = (values[:, index] for index in range(4))
    if not np.allclose(sample, np.round(sample)) or not np.allclose(feature, np.round(feature)):
        raise ValueError("Golub normalized fixture sample and feature identifiers must be integral")
    sample_ids = np.unique(sample)
    feature_ids = np.unique(feature)
    if sample_ids.size < 2 or feature_ids.size < 2:
        raise ValueError("Golub normalized fixture must contain at least two samples and two features")
    if np.unique(np.column_stack((sample, feature)), axis=0).shape[0] != values.shape[0]:
        raise ValueError("Golub normalized fixture must have one row per sample-feature pair")
    for sample_id in sample_ids:
        mask = sample == sample_id
        if np.unique(labels[mask]).size != 1 or not np.array_equal(np.sort(feature[mask]), feature_ids):
            raise ValueError("Golub normalized fixture must have one class and every feature per sample")
    return {
        "declared_recipe": fixture_spec.preprocessing,
        "output_schema": list(fixture.columns),
        "n_rows": int(values.shape[0]),
        "n_samples": int(sample_ids.size),
        "n_features": int(feature_ids.size),
        "rectangular_sample_feature_grid": True,
        "class_counts": {
            str(label): int(np.count_nonzero(labels == label)) for label in np.unique(labels)
        },
        "expression_summary": {
            "minimum": float(np.min(expression)),
            "maximum": float(np.max(expression)),
            "mean": float(np.mean(expression)),
            "standard_deviation": float(np.std(expression)),
        },
        "claim_boundary": _GOLUB_CLAIM_BOUNDARY,
    }


def _trajectory_reports(
    dataset: BenchmarkDataset,
) -> tuple[ChronologicalBenchmarkSplit, dict[str, object], dict[str, object]]:
    """Return chronology and missingness summaries from a validated trace dataset."""
    if dataset.columns != ("time", "observation"):
        raise ValueError("trajectory loader must retain time and observation columns")
    metadata = dict(dataset.metadata or {})
    subjects = np.asarray(metadata.get("subjects", ()), dtype=str)
    actions = np.asarray(metadata.get("actions", ()), dtype=str)
    times = dataset.values[:, 0]
    observations = dataset.values[:, 1]
    if subjects.shape != times.shape or actions.shape != times.shape or np.any(actions == ""):
        raise ValueError("trajectory metadata must retain one non-empty subject and action per row")
    for subject in dict.fromkeys(subjects.tolist()):
        if not np.all(np.diff(times[subjects == subject]) > 0.0):
            raise ValueError("trajectory must be strictly chronological within each subject")
    split = chronological_subject_split(dataset)
    missing = observations < 0.0
    missing_count = int(np.count_nonzero(missing))
    rows = int(observations.size)
    schema_report = {
        "required_input_columns": ["subject", "time", "action", "observation"],
        "loaded_columns": list(dataset.columns),
        "rows": rows,
        "subjects": int(np.unique(subjects).size),
        "chronological_within_subject": True,
        "subject_disjoint_split": split.to_mapping(),
    }
    missingness_report = {
        "rows": rows,
        "observed_observation_count": rows - missing_count,
        "missing_observation_count": missing_count,
        "missing_observation_rate": missing_count / rows,
        "representation": "missing observation tokens are parsed offline and encoded as -1",
        "claim_boundary": _TRAJECTORY_CLAIM_BOUNDARY,
    }
    return split, schema_report, missingness_report


@dataclass(frozen=True)
class OfflineBenchmarkPreflight:
    """Validated offline fixtures and a dashboard-ready provenance summary."""

    manifest_contract_sha256: str
    golub_archive_record: Mapping[str, Any]
    golub_archive_spec: BenchmarkSpec
    golub_archive_staged: bool
    golub_fixture_record: Mapping[str, Any]
    golub_fixture: BenchmarkDataset
    golub_recipe: Mapping[str, object]
    trajectory_record: Mapping[str, Any]
    trajectory: BenchmarkDataset
    trajectory_split: ChronologicalBenchmarkSplit
    trajectory_schema: Mapping[str, object]
    missingness_report: Mapping[str, object]

    def dashboard_payload(self) -> dict[str, object]:
        """Return the bounded, JSON-safe offline provenance payload."""
        archive_provenance = _provenance(
            self.golub_archive_record, self.golub_archive_spec, require_doi=True
        )
        fixture_provenance = _provenance(
            self.golub_fixture_record, self.golub_fixture.spec, require_doi=True
        )
        trajectory_provenance = _provenance(
            self.trajectory_record, self.trajectory.spec, require_doi=False
        )
        return {
            "kind": "offline_benchmark_provenance_dashboard",
            "offline_only": True,
            "network_access": "not performed by this domain preflight",
            "manifest_contract_sha256": self.manifest_contract_sha256,
            "golub_conversion": {
                "source_archive": {
                    **archive_provenance,
                    "staged_locally": self.golub_archive_staged,
                    "checksum_status": "verified" if self.golub_archive_staged else "checksum_pinned_not_staged",
                },
                "normalized_fixture": {
                    **fixture_provenance,
                    "checksum_verified": True,
                    "recipe": dict(self.golub_recipe),
                },
            },
            "chronological_trajectory": {
                "provenance": trajectory_provenance,
                "checksum_verified": True,
                "schema": dict(self.trajectory_schema),
                "missingness": dict(self.missingness_report),
            },
            "claim_boundary": (
                "External fixtures are offline robustness stress evidence only. They do not change "
                "the synthetic/formal evidence regime. They do not establish a biological, causal, "
                "sequential-validity, or family-level claim."
            ),
        }

    def to_mapping(self) -> dict[str, object]:
        """Return the dashboard payload using the package's standard serializer name."""
        return self.dashboard_payload()


def preflight_offline_benchmark_fixtures(
    benchmark_directory: Path,
    manifest: Mapping[str, Any],
) -> OfflineBenchmarkPreflight:
    """Audit committed benchmark fixtures without retrieval or network access.

    The checksum-pinned full Golub archive may be absent: ordinary offline
    generation uses only the committed normalized fixture.  If that archive is
    staged locally, its digest is checked before the preflight can succeed.
    """
    validate_benchmark_manifest(manifest)
    records = _records_by_id(manifest)
    archive_record = records[_GOLUB_ARCHIVE_ID]
    fixture_record = records[_GOLUB_FIXTURE_ID]
    trajectory_record = records[_TRAJECTORY_ID]
    archive_spec = _spec(archive_record)
    fixture_spec = _spec(fixture_record)
    trajectory_spec = _spec(trajectory_record)

    archive_path = _safe_file(benchmark_directory, archive_record)
    archive_staged = archive_path.exists()
    if archive_staged and not archive_path.is_file():
        raise ValueError("staged Golub archive path must be a regular file")
    if archive_staged and sha256_file(archive_path) != archive_spec.sha256:
        raise ValueError("checksum mismatch for staged Golub archive")
    golub_fixture = load_normalized_golub_fixture(
        _safe_file(benchmark_directory, fixture_record), fixture_spec
    )
    recipe = _validate_golub_lineage(
        archive_record, fixture_record, archive_spec, fixture_spec, golub_fixture
    )
    trajectory = load_agent_trace(
        _safe_file(benchmark_directory, trajectory_record),
        trajectory_spec,
        allow_missing_observations=True,
    )
    split, schema_report, missingness_report = _trajectory_reports(trajectory)
    return OfflineBenchmarkPreflight(
        manifest_contract_sha256=_canonical_hash(manifest),
        golub_archive_record=archive_record,
        golub_archive_spec=archive_spec,
        golub_archive_staged=archive_staged,
        golub_fixture_record=fixture_record,
        golub_fixture=golub_fixture,
        golub_recipe=recipe,
        trajectory_record=trajectory_record,
        trajectory=trajectory,
        trajectory_split=split,
        trajectory_schema=schema_report,
        missingness_report=missingness_report,
    )
