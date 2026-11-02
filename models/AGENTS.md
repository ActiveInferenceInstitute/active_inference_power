# Models agent guide

`models/` holds the shipped GNN specifications and the upstream custody pin.

- Keep one specification per declared generative model; the file stem is the
  runtime-model key used by `active_inference_power.gnn_specs.runtime_matrices`.
- Declare structure and, where a value is stable and clean, the exact
  `InitialParameterization`; never copy generated arrays or result numbers out
  of `output/`. Runtime values remain owned by their producers
  (`active_inference.build_generative_model`,
  `scenario.make_reference_action_scenario`).
- Use the canonical `## ActInfOntologyAnnotation` heading (the spaced variant is
  accepted only as a historical alias) and prefer terms from the upstream GNN
  vocabulary.
- After editing a specification, run
  `./.venv/bin/python scripts/validate_gnn_specs.py` and
  `./.venv/bin/python -m pytest tests/test_gnn_specs.py`; the second check also
  exercises the pinned upstream parser when the sibling checkout is present.
- A pin bump is a deliberate act: update `gnn-pin.json` with the new exact
  commit, the upstream `version` from that revision's `pyproject.toml`, and the
  `date_recorded`, keep `tag` when the release is tagged, then re-run both
  checks.
