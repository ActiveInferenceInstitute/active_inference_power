# GNN conformance

`gnn conformance (pinned)` is defined in `.github/workflows/ci.yml` (job key
`gnn-conformance`) and reads `models/gnn-pin.json`, which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.0.3`
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`) with the parser entry point
`gnn.parsers.markdown_parser:MarkdownGNNParser`.

The job clones that repository at the pinned tag into
`../Generalized_Notation_Notation`, asserts the checkout's `HEAD` equals the
pin commit, installs it into its own virtual environment, and exports
`ACTIVE_INFERENCE_POWER_GNN_ROOT` before running
`uv run --extra dev python scripts/validate_gnn_specs.py` and
`uv run --extra dev pytest tests/test_gnn_specs.py -q`. Those two read the two
shipped specifications, `models/reference_action_loop.gnn.md` and
`models/sequential_evidence_agent.gnn.md`, require every declared variable and
matrix to match the runtime models built by `build_generative_model` and
`make_reference_action_scenario`, and require the pinned upstream parser to
reproduce the project digest for each specification. It does not scan the rest
of the repository for `*.gnn.md`.
