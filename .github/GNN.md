# GNN conformance

[`gnn conformance (pinned)`](workflows/ci.yml) is the display name of job key
`gnn-conformance`. Its source of truth is
[`models/gnn-pin.json`](../models/gnn-pin.json), schema
`gnn-importer-pin/v1`, pinning
`ActiveInferenceInstitute/Generalized_Notation_Notation` at tag `v5.0.3`,
version `5.0.3`, commit `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`,
with parser entry point `gnn.parsers.markdown_parser:MarkdownGNNParser`.

The job clones that repository at the pinned tag into
`../Generalized_Notation_Notation`, asserts the checkout's `HEAD` equals the
pin commit, installs it into its own virtual environment, and exports
`ACTIVE_INFERENCE_POWER_GNN_ROOT` before running the public `./run.sh gnn`
arm. That arm composes specification/runtime validation with
[`tests/test_gnn_specs.py`](../tests/test_gnn_specs.py). These read the two
shipped specifications,
[`models/reference_action_loop.gnn.md`](../models/reference_action_loop.gnn.md)
and
[`models/sequential_evidence_agent.gnn.md`](../models/sequential_evidence_agent.gnn.md),
require every declared variable and
matrix to match the runtime models built by `build_generative_model` and
`make_reference_action_scenario`, and require the pinned upstream parser to
reproduce the project digest for each specification. It does not scan the rest
of the repository for `*.gnn.md`.

For an offline local run, provision the upstream checkout and its parser
environment in advance. Select it with `ACTIVE_INFERENCE_POWER_GNN_ROOT`;
when unset, the gate resolves the sibling `../Generalized_Notation_Notation`
checkout and passes the root explicitly to its conformance subprocesses.
Follow the [run guide](../docs/RUN_GUIDE.md). The gate does not clone, fetch,
install into, or modify that reference checkout, and a missing checkout is
not a successful conformance result. The workflow provisions its own CI
checkout separately.

CI uses `astral-sh/setup-uv` at
`68559e99cc807c1eb7e9238fd43a15bba460eec2`, selecting uv `0.13.6`;
it does not use an organization composite setup action.
