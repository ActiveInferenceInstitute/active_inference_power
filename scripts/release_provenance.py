"""Shared source-witness checks for release and campaign artifact bundles.

The manifest is not only an inventory of files.  Its configuration and claim
ledger records are the scientific-input witnesses for the results and run log.
These helpers keep those witnesses bound to the exact run that produced a
bundle, rather than merely checking that every individual file still hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from active_inference_power.certificate_gate import effective_certificate_gate_result
from scripts.artifacts import sha256_file


_RENDER_INPUT_TOKEN = "{{RESULT_RENDER_INPUT_HASH}}"
_RENDER_TEXT_SUFFIXES = frozenset({".bib", ".json", ".md", ".tex", ".yaml", ".yml"})
_EXECUTION_ROOT_FILES = (
    "pyproject.toml",
    "uv.lock",
    "domain_profile.yaml",
    "experiment_plan.yaml",
    "manuscript/config.yaml",
)


def require_file(path: Path, description: str) -> Path:
    """Return a non-empty regular file or raise a stable validation error."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 1:
        raise ValueError(f"{description} is missing or empty: {path}")
    return path


def resolve_config_path(config_path: Path | None, *, project_root: Path) -> Path:
    """Resolve one explicit, non-empty scientific configuration source.

    A publication bundle defaults to ``manuscript/config.yaml``.  An isolated
    research lane may name a different preregistered file, but callers must
    supply that path explicitly so provenance validation can bind it to the
    manifest witness instead of silently accepting whichever config happens to
    be present in the canonical location.
    """
    project = Path(project_root).resolve()
    candidate = project / "manuscript" / "config.yaml" if config_path is None else Path(config_path)
    if not candidate.is_absolute():
        candidate = project / candidate
    return require_file(candidate.resolve(), "source configuration")


def manifest_path(
    output_root: Path, project_root: Path, record: Mapping[str, Any]
) -> Path:
    """Resolve one manifest record without allowing namespace escape."""
    root_label = record.get("root", "project")
    if root_label not in {"project", "output"}:
        raise ValueError(f"manifest artifact root is invalid: {root_label!r}")
    root = Path(project_root) if root_label == "project" else Path(output_root)
    root = root.resolve()
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("manifest artifact path is missing")
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"manifest artifact escapes its namespace: {raw_path}") from exc
    return path


def _require_digest(value: Any, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{description} must be a SHA-256 digest")
    return value


def _source_record(
    manifest: Mapping[str, Any],
    name: str,
    role: str,
    *,
    project_root: Path,
    output_root: Path,
    expected_path: Path | None,
) -> tuple[Mapping[str, Any], str]:
    display_name = "configuration" if name == "config" else name.replace("_", "-")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("manifest artifacts are missing")
    record = artifacts.get(name)
    if not isinstance(record, Mapping):
        raise ValueError(f"manifest is missing required {name} source witness")
    if record.get("role") != role:
        raise ValueError(f"manifest {name} source witness has role {record.get('role')!r}")
    path = require_file(
        manifest_path(output_root, project_root, record),
        f"manifest {name} source witness",
    )
    if expected_path is not None and path != Path(expected_path).resolve():
        raise ValueError(
            f"manifest {name} source witness does not identify the expected source: {path}"
        )
    recorded = _require_digest(record.get("sha256"), f"manifest {name} source witness")
    actual = sha256_file(path)
    if actual != recorded:
        raise ValueError(
            f"manifest {display_name} source witness digest does not match its file"
        )
    return record, actual


def validate_source_witnesses(
    manifest: Mapping[str, Any],
    results: Mapping[str, Any],
    run_log: Mapping[str, Any],
    *,
    project_root: Path,
    output_root: Path,
    config_path: Path | None = None,
    claim_ledger_path: Path | None = None,
) -> dict[str, str]:
    """Bind source records, result metadata, and run-log provenance.

    ``config_path`` and ``claim_ledger_path`` are optional so a caller can
    validate a portable bundle whose source witnesses live outside the current
    checkout.  Release and hydration callers pass their canonical paths, which
    additionally detects live-source drift before a bundle can be promoted.
    """
    metadata = results.get("meta")
    if not isinstance(metadata, Mapping):
        raise ValueError("results metadata is missing")

    config_record, config_digest = _source_record(
        manifest,
        "config",
        "configuration",
        project_root=project_root,
        output_root=output_root,
        expected_path=config_path,
    )
    declared_config = _require_digest(
        manifest.get("config_sha256"), "manifest configuration"
    )
    result_config = _require_digest(
        metadata.get("config_sha256"), "results configuration"
    )
    log_config = _require_digest(
        run_log.get("config_sha256"), "run-log configuration"
    )
    if len({declared_config, result_config, log_config, config_digest}) != 1:
        raise ValueError(
            "configuration provenance mismatch among manifest, results, run log, and source witness"
        )
    if config_record.get("sha256") != declared_config:
        raise ValueError("manifest configuration source witness is not bound to manifest.config_sha256")

    ledger_record, ledger_digest = _source_record(
        manifest,
        "claim_ledger",
        "claim_ledger",
        project_root=project_root,
        output_root=output_root,
        expected_path=claim_ledger_path,
    )
    result_ledger = _require_digest(
        metadata.get("claim_ledger_sha256"), "results claim ledger"
    )
    log_ledger = _require_digest(
        run_log.get("claim_ledger_sha256"), "run-log claim ledger"
    )
    if len({result_ledger, log_ledger, ledger_digest}) != 1:
        raise ValueError(
            "claim-ledger provenance mismatch among results, run log, and source witness"
        )
    if ledger_record.get("sha256") != result_ledger:
        raise ValueError("manifest claim-ledger source witness is not bound to results metadata")
    return {"config_sha256": declared_config, "claim_ledger_sha256": result_ledger}


def validate_certificate_gate_witnesses(
    results: Mapping[str, Any],
    manifest: Mapping[str, Any],
    run_log: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the terminal certificate verdict to all bundle witnesses.

    Bundles that lack a serialized policy are interpreted
    conservatively as all-terminal.  Once a result serializes an explicit
    policy, both terminal witnesses must serialize the identical effective
    result, including any failed diagnostic names.
    """
    metadata = results.get("meta")
    certificates = results.get("certificates")
    if not isinstance(metadata, Mapping):
        raise ValueError("results metadata is missing")
    if not isinstance(certificates, Mapping) or not certificates:
        raise ValueError("results have no certificate records")
    serialized = metadata.get("certificate_gate")
    if serialized is not None and not isinstance(serialized, Mapping):
        raise ValueError("results certificate gate is malformed")
    gate = effective_certificate_gate_result(certificates, serialized)
    explicit_policy = serialized is not None
    for name, witness in (("manifest", manifest), ("run log", run_log)):
        recorded = witness.get("certificate_gate")
        if explicit_policy and not isinstance(recorded, Mapping):
            raise ValueError(f"{name} does not record the explicit certificate gate")
        if recorded is not None:
            if not isinstance(recorded, Mapping) or dict(recorded) != gate:
                raise ValueError(f"{name} certificate gate does not match results")
        expected_terminal = bool(gate["terminal_certificates_pass"])
        if witness.get("certificates_pass") is not expected_terminal:
            raise ValueError(
                f"{name} certificates_pass does not match terminal certificate results"
            )
    return gate


def execution_source_paths(project_root: Path) -> tuple[Path, ...]:
    """Return the complete executable, configuration, and data input surface.

    All script modules are bound because the orchestrator, quality gate,
    hydration, checks, and release finalizers call one another in a normal
    release workflow.  The result also depends on the declared configuration
    and data inputs.  Keeping this an explicit filesystem allowlist avoids
    Git-state dependence while excluding generated output and caches.
    """
    project = Path(project_root).resolve()
    paths: list[Path] = []
    package = project / "src" / "active_inference_power"
    scripts = project / "scripts"
    data = project / "data"
    for directory, description in (
        (package, "source package"),
        (scripts, "scripts directory"),
        (data, "data input directory"),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"execution {description} is missing: {directory}")
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
            and path.name != ".DS_Store"
        )
    for name in _EXECUTION_ROOT_FILES:
        path = project / name
        if not path.is_file():
            raise FileNotFoundError(f"execution input is missing: {path}")
        paths.append(path)
    unique = {path.resolve() for path in paths}
    if not unique:
        raise ValueError("no execution-source files were found")
    return tuple(sorted(unique, key=lambda path: path.relative_to(project).as_posix()))


def execution_source_sha256(project_root: Path) -> str:
    """Hash exact executable/result-input files in a path-bound order."""
    project = Path(project_root).resolve()
    digest = hashlib.sha256()
    for path in execution_source_paths(project):
        relative = path.relative_to(project).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def validate_execution_provenance(
    manifest: Mapping[str, Any],
    results: Mapping[str, Any],
    run_log: Mapping[str, Any],
    *,
    project_root: Path,
) -> str:
    """Bind a result bundle to the exact current executable source tree."""
    metadata = results.get("meta")
    if not isinstance(metadata, Mapping):
        raise ValueError("results metadata is missing")
    declared = _require_digest(
        manifest.get("execution_source_sha256"), "manifest execution source"
    )
    result_digest = _require_digest(
        metadata.get("execution_source_sha256"), "results execution source"
    )
    log_digest = _require_digest(
        run_log.get("execution_source_sha256"), "run-log execution source"
    )
    live_digest = execution_source_sha256(project_root)
    if len({declared, result_digest, log_digest, live_digest}) != 1:
        raise ValueError(
            "execution-source provenance mismatch among manifest, results, run log, and live source tree"
        )
    return declared


def validate_checkpoint_provenance(
    checkpoint: Mapping[str, Any],
    *,
    run_id: str,
    config_sha256: str,
    claim_ledger_sha256: str,
    profile: str | None = None,
    stage: str | None = None,
    execution_source_sha256: str | None = None,
) -> None:
    """Require a checkpoint to belong to the exact release/campaign lineage."""
    if checkpoint.get("run_id") != run_id:
        raise ValueError("checkpoint run identifier does not match the artifact bundle")
    if checkpoint.get("config_sha256") != config_sha256:
        raise ValueError("checkpoint configuration digest does not match the artifact bundle")
    if checkpoint.get("claim_ledger_sha256") != claim_ledger_sha256:
        raise ValueError("checkpoint claim-ledger digest does not match the artifact bundle")
    if profile is not None and checkpoint.get("profile") != profile:
        raise ValueError("checkpoint profile does not match the artifact bundle")
    if stage is not None and checkpoint.get("stage") != stage:
        raise ValueError("checkpoint stage does not match its artifact name")
    if (
        execution_source_sha256 is not None
        and checkpoint.get("execution_source_sha256") != execution_source_sha256
    ):
        raise ValueError("checkpoint execution-source digest does not match the artifact bundle")


def checkpoint_names_from_run_log(run_log: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact terminal-stage checkpoint set required by a run log.

    The ledger records periodic partial-checkpoint events for long stages, but
    the generator writes one final checkpoint for every passed (or resumed)
    stage.  Treat those terminal events as the canonical set so repeated
    partial updates neither weaken nor make the inventory ambiguous.
    """
    events = run_log.get("events")
    if not isinstance(events, list):
        raise ValueError("run-log events are missing")
    names: set[str] = set()
    for event in events:
        if not isinstance(event, Mapping) or event.get("status") not in {
            "passed",
            "resumed",
        }:
            continue
        name = event.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("terminal run-log event has no stage name")
        names.add(name)
    if not names:
        raise ValueError("terminal artifact bundle has no declared stages")
    return tuple(sorted(names))


def manuscript_source_paths(project_root: Path) -> tuple[Path, ...]:
    """Return every source file that contributes to a resolved manuscript tree."""
    manuscript = Path(project_root) / "manuscript"
    excluded = {"AGENTS.md", "README.md", "SYNTAX.md", "preamble.md"}
    paths = [
        path
        for path in sorted(manuscript.glob("*.md"))
        if path.name not in excluded
    ]
    for name in ("config.yaml", "preamble.md"):
        path = manuscript / name
        if path.is_file():
            paths.append(path)
    paths.extend(sorted(manuscript.glob("*.bib")))
    assets = manuscript / "assets"
    if assets.is_dir():
        paths.extend(sorted(path for path in assets.rglob("*") if path.is_file()))
    return tuple(paths)


def renderer_input_hash(*, project_root: Path, output_root: Path) -> str:
    """Return a self-normalizing digest of every publication-render input.

    The digest covers manuscript sources, hydrated source files, variables,
    figure registry, and PNGs.  ``RESULT_RENDER_INPUT_HASH`` is normalized to
    its token in both the variable JSON and hydrated text, so it can be
    embedded in the rendered manuscript without creating a self-referential
    hash.  A stale render therefore cannot be accepted merely by forging file
    modification times or retaining config/ledger witnesses.
    """
    project = Path(project_root).resolve()
    output = Path(output_root).resolve()
    variables_path = require_file(
        output / "data" / "manuscript_variables.json", "manuscript variables"
    )
    try:
        variables = json.loads(variables_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("manuscript variables are not valid JSON") from exc
    if not isinstance(variables, dict):
        raise ValueError("manuscript variables must be a JSON object")
    current_witness = variables.get("RESULT_RENDER_INPUT_HASH")
    if not isinstance(current_witness, str) or not current_witness:
        raise ValueError("manuscript variables omit RESULT_RENDER_INPUT_HASH")

    components: dict[str, bytes] = {}

    def add(name: str, path: Path, *, normalize_witness: bool = False) -> None:
        if name in components:
            raise ValueError(f"duplicate renderer-input component: {name}")
        content = require_file(path, f"renderer input {name}").read_bytes()
        if normalize_witness:
            content = content.replace(
                current_witness.encode("utf-8"), _RENDER_INPUT_TOKEN.encode("utf-8")
            )
        components[name] = content

    resolved_source_paths: set[str] = set()
    for source in manuscript_source_paths(project):
        relative = source.relative_to(project / "manuscript").as_posix()
        add(f"source:{relative}", source)
        if relative.startswith("assets/"):
            continue
        resolved_source_paths.add(relative)
        add(
            f"resolved:{relative}",
            output / "manuscript" / relative,
            normalize_witness=True,
        )

    # The sibling renderer discovers every injected top-level Markdown/LaTeX
    # file, and may add generated transmission bookends.  Bind any such
    # extra resolved file as well: otherwise it could enter a render without
    # changing this witness merely because it has no source-manuscript peer.
    resolved_manuscript = output / "manuscript"
    for path in sorted(
        (candidate for candidate in resolved_manuscript.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(resolved_manuscript).as_posix(),
    ):
        relative = path.relative_to(resolved_manuscript).as_posix()
        if relative in resolved_source_paths:
            continue
        add(
            f"resolved_extra:{relative}",
            path,
            normalize_witness=path.suffix.lower() in _RENDER_TEXT_SUFFIXES,
        )

    normalized_variables = dict(variables)
    normalized_variables["RESULT_RENDER_INPUT_HASH"] = _RENDER_INPUT_TOKEN
    components["variables"] = json.dumps(
        normalized_variables,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    figures = output / "figures"
    add("figure_registry", figures / "figure_registry.json")
    figure_paths = sorted(figures.glob("*.png"))
    if not figure_paths:
        raise ValueError("renderer input has no publication figures")
    for figure in figure_paths:
        add(f"figure:{figure.name}", figure)

    digest = hashlib.sha256()
    for name in sorted(components):
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(hashlib.sha256(components[name]).digest())
    return digest.hexdigest()


def _expected_record(
    expected: dict[str, tuple[Path, str]], name: str, path: Path, role: str
) -> None:
    if name in expected:
        raise ValueError(f"duplicate required manifest artifact name: {name}")
    expected[name] = (Path(path).resolve(), role)


def campaign_lane_predecessor_artifacts(
    config_path: Path | None,
    *,
    project_root: Path,
    output_root: Path,
) -> dict[str, tuple[Path, str]]:
    """Return the hash-bound predecessor receipts declared by a campaign lane.

    A frozen research lane may name prior failed or diagnostic evidence that
    motivates its terminal policy.  Those receipts are inputs to the campaign
    decision, not prose-only references: bind their exact bytes before a
    campaign can be generated or finalized.  Configurations without a
    ``research_lane`` block retain the ordinary campaign contract.
    """
    project = Path(project_root).resolve()
    output = Path(output_root).resolve()
    source_config = resolve_config_path(config_path, project_root=project)
    try:
        payload = yaml.safe_load(source_config.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"campaign configuration YAML is invalid: {source_config}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("campaign configuration must contain a mapping")
    lane_value = payload.get("research_lane")
    if lane_value is None:
        return {}
    if not isinstance(lane_value, Mapping):
        raise ValueError("campaign research_lane must be a mapping")
    for key in ("lane_id", "output_namespace", "decision", "non_claim"):
        value = lane_value.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"campaign research_lane.{key} must be non-empty")
    namespace = Path(str(lane_value["output_namespace"]))
    if (
        namespace.is_absolute()
        or not namespace.parts
        or namespace.parts[0] != "output"
        or ".." in namespace.parts
    ):
        raise ValueError("campaign research_lane.output_namespace must be a safe output-relative path")
    expected_output = (project / namespace).resolve()
    if output != expected_output:
        raise ValueError(
            "campaign output root does not match research_lane.output_namespace: "
            f"{output} != {expected_output}"
        )
    predecessor_value = lane_value.get("predecessor_evidence")
    if not isinstance(predecessor_value, list) or not predecessor_value:
        raise ValueError("campaign research_lane.predecessor_evidence must be a non-empty list")
    output_namespace = (project / "output").resolve()
    records: dict[str, tuple[Path, str]] = {}
    seen: set[str] = set()
    for value in predecessor_value:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("campaign predecessor evidence paths must be non-empty strings")
        relative = Path(value)
        if (
            relative.is_absolute()
            or not relative.parts
            or relative.parts[0] != "output"
            or ".." in relative.parts
        ):
            raise ValueError("campaign predecessor evidence must be a safe output-relative path")
        key = relative.as_posix()
        if key in seen:
            raise ValueError(f"campaign predecessor evidence is duplicated: {key}")
        seen.add(key)
        path = (project / relative).resolve()
        try:
            path.relative_to(output_namespace)
        except ValueError as exc:
            raise ValueError("campaign predecessor evidence escapes the output namespace") from exc
        try:
            path.relative_to(output)
        except ValueError:
            pass
        else:
            raise ValueError("campaign predecessor evidence cannot come from its own output namespace")
        require_file(path, f"campaign predecessor evidence {key}")
        records[f"predecessor:{key}"] = (path, "predecessor_evidence")
    return records


def required_release_artifacts(
    *,
    project_root: Path,
    output_root: Path,
    figure_filenames: Iterable[str],
    checkpoint_names: Iterable[str],
    require_render: bool,
) -> dict[str, tuple[Path, str]]:
    """Define the complete source-bound inventory of a release bundle."""
    project = Path(project_root).resolve()
    output = Path(output_root).resolve()
    expected: dict[str, tuple[Path, str]] = {}
    for name, relative, role in (
        ("config", project / "manuscript" / "config.yaml", "configuration"),
        ("claim_ledger", project / "data" / "claim_ledger.yaml", "claim_ledger"),
        ("results", output / "results.json", "deterministic_results"),
        ("certificates", output / "certificates.json", "certificate_results"),
        ("action_loop_suite", output / "action_loop_suite.json", "action_loop_results"),
        ("formalism_registry", output / "formalism_registry.json", "formalism_registry"),
        ("demo_payload", output / "demo" / "payload.json", "demo_payload"),
        ("figure_registry", output / "figures" / "figure_registry.json", "figure_registry"),
        ("dashboard", output / "review" / "dashboard.html", "review_dashboard"),
        ("run_log", output / "run_log.json", "operational_telemetry"),
        ("quality", output / "quality.json", "quality_report"),
        (
            "manuscript_variables",
            output / "data" / "manuscript_variables.json",
            "manuscript_variables",
        ),
    ):
        _expected_record(expected, name, relative, role)
    for filename in figure_filenames:
        stem = Path(filename).stem
        _expected_record(
            expected,
            f"figure:{stem}",
            output / "figures" / filename,
            "publication_figure",
        )
    for checkpoint in checkpoint_names:
        _expected_record(
            expected,
            f"checkpoint:{checkpoint}",
            output / "checkpoints" / f"{checkpoint}.json",
            "stage_checkpoint",
        )
    for source in manuscript_source_paths(project):
        relative = source.relative_to(project / "manuscript")
        _expected_record(
            expected,
            f"source_manuscript:{relative.as_posix()}",
            source,
            "manuscript_source",
        )
    resolved = output / "manuscript"
    for source in manuscript_source_paths(project):
        relative = source.relative_to(project / "manuscript")
        if relative.parts and relative.parts[0] == "assets":
            continue
        _expected_record(
            expected,
            f"resolved_manuscript:{relative.as_posix()}",
            resolved / relative,
            "resolved_manuscript",
        )
    benchmark_manifest = project / "data" / "benchmarks" / "manifest.yaml"
    if benchmark_manifest.is_file():
        _expected_record(
            expected,
            "benchmark_manifest",
            benchmark_manifest,
            "benchmark_provenance",
        )
    benchmark_names = {
        "golub_normalized_fixture.csv": "golub_normalized_offline_fixture",
    }
    for fixture in sorted((project / "data" / "benchmarks").glob("*.csv")):
        _expected_record(
            expected,
            f"benchmark:{benchmark_names.get(fixture.name, fixture.stem)}",
            fixture,
            "benchmark_fixture",
        )
    if require_render:
        for name, path, role in (
            ("rendered_pdf", output / "pdf" / "active_inference_power_combined.pdf", "rendered_pdf"),
            ("rendered_html", output / "web" / "index.html", "rendered_html"),
            ("web:index.html", output / "web" / "index.html", "rendered_html"),
            (
                "web:_combined_manuscript.md",
                output / "web" / "_combined_manuscript.md",
                "rendered_html",
            ),
            (
                "release_review",
                output / "review" / "release_review.json",
                "release_review_receipt",
            ),
            (
                "renderer_provenance",
                output / "review" / "renderer_provenance.json",
                "renderer_provenance_receipt",
            ),
        ):
            _expected_record(expected, name, path, role)
    return expected


def required_campaign_artifacts(
    *,
    project_root: Path,
    output_root: Path,
    figure_filenames: Iterable[str],
    checkpoint_names: Iterable[str],
    config_path: Path | None = None,
) -> dict[str, tuple[Path, str]]:
    """Define the non-publication inventory required for a campaign bundle.

    Campaigns are complete computational evidence bundles, but they are not
    manuscript releases.  In particular, a campaign must not be forced to
    carry a quality report, hydrated manuscript, or renderer output merely to
    establish that its results, figures, checkpoints, and scientific inputs
    form one coherent lineage.  ``config_path`` binds the inventory to the
    exact preregistered lane configuration selected by the finalizer.
    """
    project = Path(project_root).resolve()
    output = Path(output_root).resolve()
    source_config = resolve_config_path(config_path, project_root=project)
    expected: dict[str, tuple[Path, str]] = {}
    for name, path, role in (
        ("config", source_config, "configuration"),
        ("claim_ledger", project / "data" / "claim_ledger.yaml", "claim_ledger"),
        ("results", output / "results.json", "deterministic_results"),
        ("certificates", output / "certificates.json", "certificate_results"),
        ("action_loop_suite", output / "action_loop_suite.json", "action_loop_results"),
        ("formalism_registry", output / "formalism_registry.json", "formalism_registry"),
        ("demo_payload", output / "demo" / "payload.json", "demo_payload"),
        ("figure_registry", output / "figures" / "figure_registry.json", "figure_registry"),
        ("dashboard", output / "review" / "dashboard.html", "review_dashboard"),
        ("run_log", output / "run_log.json", "operational_telemetry"),
    ):
        _expected_record(expected, name, path, role)
    for filename in figure_filenames:
        stem = Path(filename).stem
        _expected_record(
            expected,
            f"figure:{stem}",
            output / "figures" / filename,
            "publication_figure",
        )
    for checkpoint in checkpoint_names:
        _expected_record(
            expected,
            f"checkpoint:{checkpoint}",
            output / "checkpoints" / f"{checkpoint}.json",
            "stage_checkpoint",
        )
    benchmark_manifest = project / "data" / "benchmarks" / "manifest.yaml"
    if benchmark_manifest.is_file():
        _expected_record(
            expected,
            "benchmark_manifest",
            benchmark_manifest,
            "benchmark_provenance",
        )
    benchmark_names = {
        "golub_normalized_fixture.csv": "golub_normalized_offline_fixture",
    }
    for fixture in sorted((project / "data" / "benchmarks").glob("*.csv")):
        _expected_record(
            expected,
            f"benchmark:{benchmark_names.get(fixture.name, fixture.stem)}",
            fixture,
            "benchmark_fixture",
        )
    for name, (path, role) in campaign_lane_predecessor_artifacts(
        source_config,
        project_root=project,
        output_root=output,
    ).items():
        _expected_record(expected, name, path, role)
    return expected


def validate_artifact_inventory(
    manifest: Mapping[str, Any],
    expected: Mapping[str, tuple[Path, str]],
    *,
    project_root: Path,
    output_root: Path,
) -> dict[str, int]:
    """Require exact-name/path/role records for all release-critical inputs."""
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("manifest artifacts are missing")
    missing = sorted(set(expected) - set(artifacts))
    if missing:
        raise ValueError(f"manifest is missing required artifact records: {missing}")
    for name, (expected_path, expected_role) in expected.items():
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"manifest artifact record is malformed: {name}")
        if record.get("role") != expected_role:
            raise ValueError(
                f"manifest artifact {name} must have role {expected_role!r}"
            )
        actual_path = manifest_path(output_root, project_root, record)
        if actual_path != expected_path:
            raise ValueError(
                f"manifest artifact {name} does not identify its required path"
            )
        require_file(actual_path, f"manifest artifact {name}")
    return {"required_artifact_count": len(expected)}


__all__ = [
    "manifest_path",
    "manuscript_source_paths",
    "renderer_input_hash",
    "require_file",
    "resolve_config_path",
    "validate_certificate_gate_witnesses",
    "checkpoint_names_from_run_log",
    "campaign_lane_predecessor_artifacts",
    "execution_source_paths",
    "execution_source_sha256",
    "required_campaign_artifacts",
    "required_release_artifacts",
    "validate_artifact_inventory",
    "validate_checkpoint_provenance",
    "validate_execution_provenance",
    "validate_source_witnesses",
]
