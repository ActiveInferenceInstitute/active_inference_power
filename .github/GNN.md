# GNN conformance

[`gnn conformance (pinned)`](workflows/ci.yml) is the display name of job key
`gnn-conformance`. Its source of truth is
[`models/gnn-pin.json`](../models/gnn-pin.json), schema
`gnn-importer-pin/v1`, pinning
`ActiveInferenceInstitute/Generalized_Notation_Notation` at tag `v5.1.0`,
version `5.1.0`, commit `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`,
with parser entry point `gnn.parsers.markdown_parser:MarkdownGNNParser`.

The job clones that repository at the pinned tag into
`../Generalized_Notation_Notation`, asserts the checkout's `HEAD` equals the
pin commit, installs it into its own virtual environment, and exports
`ACTIVE_INFERENCE_POWER_GNN_ROOT` before running the public `./run.sh gnn`
arm.

## Adopted: one reader, run from the pinned revision

The arm's first step is `python -m gnn.importers check --repo .` — the estate's
single reader for an importer's pin, its `.github/GNN.md` pointer and its
`*.gnn.md` specifications. It discovers every specification in the checkout
(build and cache directories excluded) and validates each with the parser
installed in the pinned revision.

[`scripts/gnn_importer_check.py`](../scripts/gnn_importer_check.py) runs that
command under the checkout's interpreter with the pinned commit's `src/gnn`
extracted from Git onto `PYTHONPATH`, so the reader and the parser it validates
with are the pinned release's bytes rather than the checkout's live working
tree. The checkout stays read-only: the script reads it with
`git cat-file` / `git archive` only.

The cutover is recorded here because it changed what this job proves:

- **Old pin:** `v5.0.3` at `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`. That
  revision predates `src/gnn/importers.py`, so this repository carried an
  interim local pin reader (`active_inference_power.gnn_specs.load_upstream_pin`)
  behind the same arm and could not call `python -m gnn.importers`.
- **New pin:** `v5.1.0` at `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`. It ships
  the reader, so the interim reader and its pin-shape tests were deleted in the
  same change: reading a pin, validating it with its pointer, and validating
  specification syntax each have one home again.

After the reader, the arm keeps the two member checks it already owned:
[`scripts/validate_gnn_specs.py`](../scripts/validate_gnn_specs.py) binds every
declared variable and matrix to the runtime models built by
`build_generative_model` and `make_reference_action_scenario`, and
[`tests/test_gnn_specs.py`](../tests/test_gnn_specs.py) requires the pinned
upstream parser to reproduce the project digest for each specification. Those
two read the two shipped specifications,
[`models/reference_action_loop.gnn.md`](../models/reference_action_loop.gnn.md)
and
[`models/sequential_evidence_agent.gnn.md`](../models/sequential_evidence_agent.gnn.md);
the shared reader, not the member checks, is what scans the rest of the
repository for `*.gnn.md`. The digest comparison also carries a non-vacuity
control: a specification whose duplicated `InitialParameterization` block the
two readers resolve differently must be reported as a divergence.

For an offline local run, provision the upstream checkout and its parser
environment in advance. Select it with `ACTIVE_INFERENCE_POWER_GNN_ROOT`;
when unset, the gate resolves the sibling `../Generalized_Notation_Notation`
checkout and passes the root explicitly to its conformance subprocesses.
Follow the [run guide](../docs/RUN_GUIDE.md). The gate does not clone, fetch,
install into, or modify that reference checkout, and a missing checkout or an
absent pinned commit is not a successful conformance result. The workflow
provisions its own CI checkout separately.

CI uses `astral-sh/setup-uv` at
`68559e99cc807c1eb7e9238fd43a15bba460eec2`, selecting uv `0.13.6`;
it does not use an organization composite setup action.
