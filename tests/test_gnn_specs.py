"""Real-file tests for the GNN specifications and their runtime binding.

The shipped ``models/*.gnn.md`` files are checked three ways: the project-owned
reader parses them, every declared matrix binds to the real runtime model
(``build_generative_model`` / ``make_reference_action_scenario``), and — when
the pinned upstream checkout is available — the pinned Generalized Notation
Notation parser reproduces the project digest exactly. The upstream parser runs
under the checkout's own interpreter with the pinned commit's ``src/gnn``
extracted from Git onto ``PYTHONPATH``, so the bytes under test are the pinned
release's bytes, not the live working tree's. The checkout is treated as
read-only (``git archive`` / ``git cat-file`` only).

The pin is read by ``gnn.importers``, which the ``gnn`` gate arm runs as
``python -m gnn.importers check --repo .``; nothing here re-implements that
reader. These tests assert the pin's declared identity and drive the member
evidence — runtime binding plus pinned-parser digest equality.

When the checkout or its ``.venv`` is unavailable the upstream tests skip with
an explicit reason; setting ``ACTIVE_INFERENCE_POWER_GNN_ROOT`` selects a
checkout explicitly (and then an unavailable checkout is a failure, not a skip).
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from active_inference_power.gnn_specs import (
    DIGEST_DRIVER_SOURCE,
    ENV_UPSTREAM_ROOT,
    PIN_RELATIVE_PATH,
    SPEC_DIRNAME,
    GnnSpecError,
    PinnedRevision,
    UpstreamInputError,
    binding_issues,
    check_specs,
    checkout_available,
    default_gnn_root,
    digest_divergences,
    extract_pinned_src,
    gnn_checkout_python,
    normalize_upstream_digest,
    parse_spec,
    pinned_revision,
    resolve_gnn_root,
    runtime_matrices,
    spec_digest,
    spec_paths,
    variable_size,
)

ROOT = Path(__file__).resolve().parent.parent
PIN_PATH = ROOT / PIN_RELATIVE_PATH
SHIPPED_SPECS = sorted((ROOT / SPEC_DIRNAME).glob("*.gnn.md"))

#: The canonical workstream pin. Bumping it is a deliberate act: this test pins
#: the pin, so a silent re-pin cannot pass unnoticed.
CANONICAL_TAG = "v5.1.0"
CANONICAL_COMMIT = "75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf"
CANONICAL_VERSION = "5.1.0"
CANONICAL_RECORDED = "2026-11-12"

#: Normative required sections of the GNN v1 compliance contract
#: (``gnn.schemas.section_contract.REQUIRED_SECTIONS``); the pinned upstream
#: checker enforces them, so the shipped specs must carry all of them.
REQUIRED_SPEC_SECTIONS = (
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "ModelAnnotation",
    "StateSpaceBlock",
    "Connections",
    "InitialParameterization",
    "Time",
    "Footer",
)


@dataclass(frozen=True)
class UpstreamRun:
    """One execution of the pinned upstream parser over the shipped specs."""

    revision: PinnedRevision
    payload: dict[str, Any]
    mutant: Path
    extract_dir: Path


@pytest.fixture(scope="module")
def upstream_run(tmp_path_factory: pytest.TempPathFactory) -> UpstreamRun:
    """Execute the pinned upstream parser on every shipped spec + a divergence mutant."""
    revision = pinned_revision(ROOT)
    root = resolve_gnn_root(ROOT)
    explicit = ENV_UPSTREAM_ROOT in os.environ
    if not checkout_available(root):
        message = (
            f"upstream GNN checkout unavailable at {root}; set {ENV_UPSTREAM_ROOT} to select one"
        )
        if explicit:
            pytest.fail(message)
        pytest.skip(message)

    extract_dir = tmp_path_factory.mktemp("gnn_upstream_pin")
    src = extract_pinned_src(root, revision.commit, extract_dir)

    # Divergence mutant: the project reader keeps the last
    # ``InitialParameterization`` block for a variable and the pinned upstream
    # parser keeps the first. Used as the non-vacuity control.
    mutant = extract_dir / "duplicate_parameter.gnn.md"
    mutant.write_text(
        (ROOT / SPEC_DIRNAME / "sequential_evidence_agent.gnn.md")
        .read_text(encoding="utf-8")
        .replace("C={(0.0, 0.0)}", "C={(0.0, 0.0)}\n\nC={(0.5, 0.5)}"),
        encoding="utf-8",
    )

    spec_files = [str(path.resolve()) for path in SHIPPED_SPECS] + [str(mutant)]
    env = {**os.environ, "PYTHONPATH": str(src)}
    parser_module, parser_class = revision.parser_module, revision.parser_class
    command = [
        str(gnn_checkout_python(root)),
        "-c",
        DIGEST_DRIVER_SOURCE,
        "--module",
        parser_module,
        "--class-name",
        parser_class,
    ]
    for spec in spec_files:
        command += ["--spec", spec]
    proc = subprocess.run(command, capture_output=True, text=True, env=env, cwd=extract_dir)
    assert proc.returncode == 0, f"upstream digest driver failed:\n{proc.stderr}"

    return UpstreamRun(
        revision=revision,
        payload=json.loads(proc.stdout),
        mutant=mutant,
        extract_dir=extract_dir,
    )


def test_pin_file_names_the_canonical_release() -> None:
    """The shipped pin is the canonical workstream pin (v5.1.0)."""
    data = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    assert data["tag"] == CANONICAL_TAG
    assert data["commit"] == CANONICAL_COMMIT
    assert data["version"] == CANONICAL_VERSION
    assert data["date_recorded"] == CANONICAL_RECORDED
    revision = pinned_revision(ROOT)
    assert revision.commit == CANONICAL_COMMIT
    assert (revision.parser_module, revision.parser_class) == (
        "gnn.parsers.markdown_parser",
        "MarkdownGNNParser",
    )


def test_pin_file_declares_the_importer_pin_schema() -> None:
    """The pin carries the canonical importer-pin schema and its record date."""
    data = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    assert data["schema"] == "gnn-importer-pin/v1"
    assert data["recorded"] == CANONICAL_RECORDED


def test_pinned_revision_refuses_a_pin_without_the_needed_fields(tmp_path: Path) -> None:
    """Selection refuses an absent or partial pin instead of guessing a revision."""
    with pytest.raises(UpstreamInputError, match="cannot read the pinned revision"):
        pinned_revision(tmp_path)
    models = tmp_path / SPEC_DIRNAME
    models.mkdir()
    data = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    partial = {key: value for key, value in data.items() if key != "parser_class"}
    (models / "gnn-pin.json").write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(UpstreamInputError, match="must declare"):
        pinned_revision(tmp_path)
    (models / "gnn-pin.json").write_text('["not", "an", "object"]', encoding="utf-8")
    with pytest.raises(UpstreamInputError, match="must be a JSON object"):
        pinned_revision(tmp_path)


def test_checkout_resolution_and_availability(tmp_path: Path) -> None:
    """An explicit checkout selection is final; a default checkout is the sibling."""
    chosen = tmp_path / "gnn"
    chosen.mkdir()
    assert resolve_gnn_root(ROOT, {ENV_UPSTREAM_ROOT: str(chosen)}) == chosen.resolve()
    assert resolve_gnn_root(ROOT, {}) == default_gnn_root(ROOT)
    with pytest.raises(UpstreamInputError, match="must not be empty"):
        resolve_gnn_root(ROOT, {ENV_UPSTREAM_ROOT: "  "})
    assert checkout_available(chosen) is False
    assert gnn_checkout_python(chosen) == chosen / ".venv" / "bin" / "python"


def test_shipped_specs_parse_and_bind_to_the_runtime_models() -> None:
    """Every shipped specification parses and matches its runtime model exactly."""
    assert [path.name for path in SHIPPED_SPECS] == [
        "reference_action_loop.gnn.md",
        "sequential_evidence_agent.gnn.md",
    ]
    report = check_specs(ROOT)
    assert report["passed"] is True
    assert "pin" not in report  # the pin is read once, by gnn.importers
    rows = {str(row["runtime_model"]): row for row in cast(list[dict[str, Any]], report["specs"])}
    assert set(rows) == {"reference_action_loop", "sequential_evidence_agent"}
    assert all(row["passed"] is True and row["binding_issues"] == [] for row in rows.values())
    assert rows["sequential_evidence_agent"]["ontology_terms"] == 8


@pytest.mark.parametrize("spec", SHIPPED_SPECS, ids=lambda path: path.name)
def test_shipped_specs_carry_every_required_section(spec: Path) -> None:
    """A spec missing a normative section fails the pinned compliance gate."""
    headings = {
        line[3:].strip()
        for line in spec.read_text(encoding="utf-8").splitlines()
        if line.startswith("## ")
    }
    assert set(REQUIRED_SPEC_SECTIONS) - headings == set()


@pytest.mark.parametrize("spec", SHIPPED_SPECS, ids=lambda path: path.name)
def test_every_connection_endpoint_is_a_declared_variable(spec: Path) -> None:
    """An edge names declared variables; its ``:annotation`` is only a label."""
    parsed = parse_spec(spec.read_text(encoding="utf-8"), source=str(spec))
    endpoints = {edge.source for edge in parsed.connections} | {
        edge.target for edge in parsed.connections
    }
    assert sorted(endpoints - set(parsed.variables)) == []


def test_sequential_agent_spec_declares_the_runtime_matrices() -> None:
    """The sequential spec's declared arrays reproduce the real generative model."""
    spec = parse_spec((ROOT / SPEC_DIRNAME / "sequential_evidence_agent.gnn.md").read_text(encoding="utf-8"))
    runtime = runtime_matrices("sequential_evidence_agent")
    assert tuple(runtime) == ("A", "B", "C", "D")
    assert runtime["A"].shape == (2, 2) and runtime["B"].shape == (2, 2, 1)
    assert runtime["C"].shape == (2,) and runtime["D"].shape == (2,)
    assert np.array_equal(runtime["B"][:, :, 0], np.eye(2))
    assert spec.variables["A"].value is not None
    assert np.allclose(np.asarray(spec.variables["A"].value).reshape(2, 2), runtime["A"], atol=1e-12)
    assert spec.variables["B"].value is None  # identity transition declared by shape
    assert binding_issues(spec, runtime) == []
    assert spec.ontology["s"] == "HiddenState"
    assert spec.model_parameters["posterior_threshold"] == "0.9"
    first_edge = spec.connections[0]
    assert (first_edge.source, first_edge.target, first_edge.label) == (
        "D",
        "s",
        "prior_initialization",
    )


def test_reference_action_loop_spec_declares_the_runtime_shapes() -> None:
    """The action-loop spec's declared dims reproduce the real scenario contract."""
    spec = parse_spec((ROOT / SPEC_DIRNAME / "reference_action_loop.gnn.md").read_text(encoding="utf-8"))
    runtime = runtime_matrices("reference_action_loop")
    assert runtime["A"].shape == (6, 2, 8)
    assert runtime["B"].shape == (6, 8, 8)
    assert runtime["C"].shape == (2,) and runtime["D"].shape == (8,)
    assert np.allclose(runtime["D"], np.full(8, 0.125))
    assert binding_issues(spec, runtime) == []
    assert spec.variables["s"].dims == (8, 1)


def test_runtime_matrices_rejects_unknown_specification() -> None:
    """An unregistered specification name is an explicit error."""
    with pytest.raises(GnnSpecError, match="no runtime model is registered"):
        runtime_matrices("not_a_shipped_spec")


def test_binding_issues_flag_every_disagreement() -> None:
    """Missing, mis-shaped, mis-sized, and mis-valued declarations all fail closed."""
    from dataclasses import replace as dataclass_replace

    text = (ROOT / SPEC_DIRNAME / "sequential_evidence_agent.gnn.md").read_text(encoding="utf-8")
    spec = parse_spec(text)
    runtime = runtime_matrices("sequential_evidence_agent")
    assert binding_issues(spec, {"A": runtime["A"], "Z": np.zeros(3)}) == [
        "Z: runtime matrix with shape (3,) is not declared"
    ]
    assert binding_issues(spec, {"A": np.zeros((2, 3))}) == ["A: declared dims (2, 2) != runtime shape (2, 3)"]
    assert binding_issues(spec, {"C": np.zeros(4)}) == ["C: declared dims (2,) != runtime shape (4,)"]
    assert binding_issues(spec, {"A": np.zeros((2, 2))}) == ["A: declared values differ from the runtime model"]
    # A declared-value size that disagrees with the runtime matrix is caught even
    # when the declaration and runtime dims agree (defense in depth).
    undersized = dataclass_replace(
        spec,
        variables={**spec.variables, "A": dataclass_replace(spec.variables["A"], value=np.zeros(3))},
    )
    assert binding_issues(undersized, {"A": runtime["A"]}) == ["A: declared 3 values != runtime matrix size 4"]
    # Values whose size contradicts their own declared dims fail at parse time.
    with pytest.raises(GnnSpecError, match="size mismatch"):
        parse_spec(text.replace("  (0.8, 0.2),\n  (0.2, 0.8)\n}", "  (0.8, 0.2)\n}"))
    assert variable_size((2, 2, 1)) == 4


def test_check_specs_fails_closed_on_a_drifted_copy(tmp_path: Path) -> None:
    """A spec whose declared dims drift from the runtime model fails the report."""
    spec_dir = tmp_path / SPEC_DIRNAME
    spec_dir.mkdir()
    drifted = (ROOT / SPEC_DIRNAME / "sequential_evidence_agent.gnn.md").read_text(encoding="utf-8")
    (spec_dir / "sequential_evidence_agent.gnn.md").write_text(
        drifted.replace("B[2,2,1,type=float]", "B[2,2,2,type=float]"), encoding="utf-8"
    )
    report = check_specs(tmp_path)
    assert report["passed"] is False
    specs = cast(list[dict[str, Any]], report["specs"])
    assert specs[0]["binding_issues"] == ["B: declared dims (2, 2, 2) != runtime shape (2, 2, 1)"]


def test_spec_paths_is_sorted_and_complete() -> None:
    """The shipped specification inventory is deterministic."""
    paths = spec_paths(ROOT)
    assert paths == sorted(paths)
    assert [path.name for path in paths] == [path.name for path in SHIPPED_SPECS]


@pytest.mark.parametrize(
    "text, message",
    [
        ("## GNNSection\n\nS\n", "missing required section"),
        ("no section headers at all\n", "missing required section"),
        (
            "## GNNSection\n\nS\n\n## GNNVersionAndFlags\n\nGNN v1.1\n\n## ModelName\n\nM\n\n"
            "## StateSpaceBlock\n\nA[2,2,type=float]\n\n## Connections\n\nbroken edge line\n",
            "malformed connection edge",
        ),
        (
            "## GNNSection\n\nS\n\n## GNNVersionAndFlags\n\nGNN v1.1\n\n## ModelName\n\nM\n\n"
            "## StateSpaceBlock\n\nnot a declaration\n\n## Connections\n\nA>A\n",
            "malformed StateSpaceBlock",
        ),
        (
            "## GNNSection\n\nS\n\n## GNNVersionAndFlags\n\nGNN v1.1\n\n## ModelName\n\nM\n\n"
            "## StateSpaceBlock\n\nA[2,type=float,default=uniform]\n\n## Connections\n\nA>A\n",
            None,
        ),
        (
            "## GNNSection\n\nS\n\n## GNNVersionAndFlags\n\nGNN v1.1\n\n## ModelName\n\nM\n\n"
            "## StateSpaceBlock\n\nA[2,,type=float]\n\n## Connections\n\n# comment\nA>A\n\nA-A\n\n"
            "## ModelParameters\n\nnum_states: 2\n# comment\nnot-a-pair\n",
            None,
        ),
        (
            "## GNNSection\n\nS\n\n## GNNVersionAndFlags\n\nGNN v1.1\n\n## ModelName\n\nM\n\n"
            "## StateSpaceBlock\n\nA[2,n,type=float]\n\n## Connections\n\nA>A\n",
            "unsupported dimension token",
        ),
        (
            "## GNNSection\n\nS\n\n## GNNVersionAndFlags\n\nGNN v1.1\n\n## ModelName\n\nM\n\n"
            "## StateSpaceBlock\n\nA[2]\n\n## Connections\n\nA>A\n",
            "missing required 'type=' declaration",
        ),
        (
            "## GNNSection\n\nS\n\n## GNNVersionAndFlags\n\nGNN v1.1\n\n## ModelName\n\nM\n\n"
            "## StateSpaceBlock\n\nA[type=float]\n\n## Connections\n\nA>A\n",
            "no integer dimensions declared",
        ),
        (
            "## GNNSection\n\nS\n\n## GNNVersionAndFlags\n\nGNN v1.1\n\n## ModelName\n\nM\n\n"
            "## StateSpaceBlock\n\nA[2,2,type=float]\n\n## Connections\n\nA>A\n\n"
            "## InitialParameterization\n\nA={(1.0, 2.0)\n",
            "unbalanced braces",
        ),
        (
            "## GNNSection\n\nS\n\n## GNNVersionAndFlags\n\nGNN v1.1\n\n## ModelName\n\nM\n\n"
            "## StateSpaceBlock\n\nA[2,2,type=float]\n\n## Connections\n\nA>A\n\n"
            "## InitialParameterization\n\nA={oops}\n",
            "cannot read InitialParameterization",
        ),
        (
            "## GNNSection\n\nS\n\n## GNNVersionAndFlags\n\nGNN v1.1\n\n## ModelName\n\nM\n\n"
            "## StateSpaceBlock\n\nA[2,2,type=float]\n\n## Connections\n\nA>A\n\n"
            "## InitialParameterization\n\nA={(1.0, 2.0)}\n",
            "size mismatch",
        ),
        (
            "## GNNSection\n\nS\n\n## GNNVersionAndFlags\n\nGNN v1.1\n\n## ModelName\n\nM\n\n"
            "## StateSpaceBlock\n\nA[2,2,type=float]\n\n## Connections\n\nA>A\n\n"
            "## InitialParameterization\n\nZ={(1.0, 2.0)}\n",
            "undeclared variables",
        ),
    ],
)
def test_parse_spec_boundaries(text: str, message: str | None) -> None:
    """Malformed specifications fail loudly; the documented default hint is accepted."""
    if message is None:
        spec = parse_spec(text)
        assert spec.variables["A"].dims == (2,)
        assert spec.variables["A"].dtype == "float"
        return
    with pytest.raises(GnnSpecError, match=message):
        parse_spec(text)


def test_spec_digest_matches_upstream_normalization() -> None:
    """Digest fields normalize versions, dtypes, times, and comma lists like upstream."""
    spec = parse_spec(
        """## GNNSection

S

## GNNVersionAndFlags

GNN v1.1

## ModelName

M

## StateSpaceBlock

A[2,2,type=float]

## Connections

A>A:label
A-A

## InitialParameterization

A={(1.0, 0.0), (0.0, 1.0)}

## Time

Dynamic

## ActInfOntologyAnnotation

A=LikelihoodMatrix

## ModelParameters

num_states: 2
labels: a,b
"""
    )
    digest = spec_digest(spec)
    assert digest["version"] == "1.1"
    assert digest["time"] == "Dynamic"
    assert digest["variables"] == {"A": {"dims": [2, 2], "dtype": "float"}}
    assert digest["connections"] == [["A", "A", "directed", "label"], ["A", "A", "undirected", ""]]
    assert digest["ontology"] == {"A": "LikelihoodMatrix"}
    assert digest["parameters"] == {"A": [[1.0, 0.0], [0.0, 1.0]]}
    assert digest_divergences(digest, digest) == []


def test_digest_normalization_and_divergence_controls() -> None:
    """Enum names, sequences, and every divergence family are handled explicitly."""
    payload = {
        "model_name": " M ",
        "version": "GNN v1.1",
        "time": "",
        "variables": [{"name": "A", "dims": [2, 1], "dtype": "DataType.INTEGER"}],
        "connections": [["A", "B", "ConnectionType.DIRECTED", None]],
        "ontology": [{"variable": "A", "term": "LikelihoodMatrix"}],
        "model_parameters": {"num_states": 2, "labels": ["a", "b"]},
        "parameters": {"A": [[0.5], [0.5]]},
    }
    digest = normalize_upstream_digest(payload)
    assert digest["model_name"] == "M"
    assert digest["version"] == "1.1"
    assert digest["time"] == "Static"
    assert digest["variables"] == {"A": {"dims": [2, 1], "dtype": "int"}}
    assert digest["connections"] == [["A", "B", "directed", ""]]
    assert digest["ontology"] == {"A": "LikelihoodMatrix"}
    assert digest_divergences(digest, {**digest, "connections": [["A", "B", "DIRECTED", ""]]}) == []
    assert digest_divergences(digest, {**digest, "parameters": {"A": [[0.5], [0.25]]}})
    assert digest_divergences(digest, {**digest, "model_parameters": {"labels": "a"}})
    assert digest_divergences(digest, {**digest, "model_parameters": {"num_states": 2, "extra": 1}})
    assert digest_divergences(digest, {**digest, "ontology": {}})
    assert digest_divergences(digest, {**digest, "variables": {}})
    assert digest_divergences(digest, {**digest, "time": "Dynamic"})
    assert any(
        "declared upstream only" in issue
        for issue in digest_divergences({**digest, "variables": {}}, digest)
    )
    assert any(
        "variables['A']" in issue
        for issue in digest_divergences(
            digest, {**digest, "variables": {"A": {"dims": [2, 2], "dtype": "int"}}}
        )
    )
    assert any(
        "connections differ" in issue
        for issue in digest_divergences(
            digest, {**digest, "connections": [["A", "C", "DIRECTED", ""]]}
        )
    )
    assert any(
        "present upstream only" in issue
        for issue in digest_divergences(
            digest, {**digest, "parameters": {"A": [[0.5], [0.5]], "extra": [1.0]}}
        )
    )
    assert any(
        "present in the project parser only" in issue
        for issue in digest_divergences(
            {**digest, "parameters": {"A": [[0.5], [0.5]], "extra": [1.0]}}, digest
        )
    )
    assert any(
        "parameters['A']" in issue
        for issue in digest_divergences(digest, {**digest, "parameters": {"A": [[0.5]]}})
    )
    assert normalize_upstream_digest({**payload, "time": "static"})["time"] == "Static"
    assert normalize_upstream_digest({**payload, "time": "unmarked"})["time"] == "Static"
    with pytest.raises(UpstreamInputError, match="must be a mapping"):
        digest_divergences({"variables": []}, {})
    with pytest.raises(UpstreamInputError, match="missing fields"):
        normalize_upstream_digest({"model_name": "M"})
    for field, value in (
        ("variables", {}),
        ("connections", {}),
        ("ontology", {}),
        ("parameters", []),
        ("model_parameters", []),
    ):
        broken = {key: payload[key] for key in payload}
        broken[field] = value
        with pytest.raises(UpstreamInputError, match="must be"):
            normalize_upstream_digest(broken)
    with pytest.raises(UpstreamInputError, match="variable entries must be mappings"):
        normalize_upstream_digest({**payload, "variables": [["A"]]})


def test_upstream_parser_ran_from_the_pinned_commit(upstream_run: UpstreamRun) -> None:
    """The executed parser module is the extracted pinned release, not the live tree."""
    parser_info = upstream_run.payload["parser"]
    assert parser_info["module"] == upstream_run.revision.parser_module
    assert parser_info["class"] == upstream_run.revision.parser_class
    assert Path(parser_info["file"]).resolve().is_relative_to(upstream_run.extract_dir.resolve())
    assert parser_info["gnn_version"] == CANONICAL_VERSION


def test_upstream_parser_accepts_every_shipped_spec(upstream_run: UpstreamRun) -> None:
    """The pinned upstream parser parses every shipped spec without errors."""
    for spec in SHIPPED_SPECS:
        digest = upstream_run.payload["specs"][str(spec.resolve())]
        assert digest["diagnostics"]["success"] is True, f"{spec.name}: {digest['diagnostics']}"
        assert digest["diagnostics"]["errors"] == [], f"{spec.name}: {digest['diagnostics']['errors']}"


@pytest.mark.parametrize("spec", SHIPPED_SPECS, ids=lambda path: path.name)
def test_project_digest_matches_the_pinned_upstream_parser(upstream_run: UpstreamRun, spec: Path) -> None:
    """The project digest equals the pinned upstream parser's digest for every spec."""
    project = spec_digest(parse_spec(spec.read_text(encoding="utf-8"), source=str(spec)))
    upstream = normalize_upstream_digest(upstream_run.payload["specs"][str(spec.resolve())])
    assert digest_divergences(project, upstream) == []


def test_conformance_check_discriminates_on_a_duplicated_value_block(
    upstream_run: UpstreamRun,
) -> None:
    """Non-vacuity control: the two readers resolve a duplicate block differently.

    The project reader keeps the last ``InitialParameterization`` block for a
    variable and the pinned upstream parser keeps the first, so the same bytes
    declare different values. The comparison must report that divergence rather
    than agree vacuously.
    """
    project = spec_digest(
        parse_spec(upstream_run.mutant.read_text(encoding="utf-8"), source=str(upstream_run.mutant))
    )
    upstream = normalize_upstream_digest(upstream_run.payload["specs"][str(upstream_run.mutant.resolve())])
    project_parameters = cast(dict[str, Any], project["parameters"])
    upstream_parameters = cast(dict[str, Any], upstream["parameters"])
    assert project_parameters["C"] == [[0.5, 0.5]]
    assert upstream_parameters["C"] == [[0.0, 0.0]]
    assert any(
        issue.startswith("parameters['C']") for issue in digest_divergences(project, upstream)
    )
