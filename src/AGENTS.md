# Source guide

`src/active_inference_power/` is the deterministic domain layer. Keep
statistical calculations, scenario contracts, and artifact-schema validation
here; scripts own command-line parsing, paths, subprocesses, and rendering.

- Do not import `infrastructure.*`, mutate `output/`, or depend on the sibling
  template from package modules.
- Treat public dataclasses and `to_mapping()` output as versioned contracts.
  New release-facing fields must be validated at the boundary and covered by a
  real producer/consumer test.
- Preserve the distinction between likelihood-ratio e-process evidence,
  LORD++/SAFFRON procedural alpha-wealth, e-LOND's cumulative nominal-level
  allocation trace (which can exceed $\alpha$ after recycled levels), simulated
  cost, and source-work accounting. e-LOND is neither alpha-wealth nor a
  remaining alpha budget.
- An action-loop target-specific e-process can carry only the declared
  per-target optional-stopping role. It does not establish FWER/FDR control;
  the all-action product is diagnostic and finite observed error summaries are
  estimands rather than control claims.
- Treat `GenerativeProcessSpec.cost_unit` and `cost_scale_id` as part of the
  hashed scenario contract. Compare cost or power-per-cost across scenarios
  only when both identifiers agree.
- Prefer seeded NumPy/SciPy/pymdp computations; reject malformed or
  non-finite inputs rather than coercing them silently.

Run the focused tests for a changed module, then the project test, type, and
lint gates from the repository root.
