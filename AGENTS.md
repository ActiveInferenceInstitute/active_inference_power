# Agent guide — `active_inference_power`

This repository contains the `active_inference_power` package. A separately
provisioned controlled renderer owns publication typesetting; do not copy that
engine into this package or present it as a public dependency.

## Hard project contracts

- `src/` contains deterministic domain logic and may import NumPy, SciPy,
  PyYAML, pymdp, matplotlib for figures, and sibling project modules; it must
  not import `infrastructure.*`.
- `scripts/` orchestrates configuration, subprocesses, file I/O, and rendering
  calls. Scientific calculations belong in `src/`.
- Tests use real values, real RNGs, real files, and the real pymdp dependency;
  mocks are not permitted. The release coverage gate is 95% branch coverage
  on `active_inference_power`.
- Results, figures, quality metrics, and resolved manuscript files are
  generated artifacts. Never hand-edit `output/`.
- Every changing manuscript number comes from
  `src/active_inference_power/manuscript_variables.py`;
  measured test and coverage values come from the selected bundle root's
  `quality.json` (default: `output/quality.json`).
- Claims must stay inside the declared evidence regime. A PRDS simulation is
  not arbitrary-dependence evidence; a Monte-Carlo band is not a proof.
- A current figure registry is an interpretation contract as well as a file
  inventory: retain its reader question, comparison, takeaway, visual
  encoding, and validity metadata. When panels have different inferential
  status, `panel_validity` must identify the exceptional panel rather than let
  a whole-figure label conceal it.

## Required reading before edits

Read `docs/agent_instructions.md`, `docs/architecture.md`, and
`docs/testing_philosophy.md` before source/test changes. Read
`docs/rendering_pipeline.md` and `docs/syntax_guide.md` before manuscript or
output changes. Read the corresponding directory guide under `src/`, `tests/`,
`scripts/`, `manuscript/`, or `data/` when present.

For documentation or contributor-guidance changes, read `docs/AGENTS.md` and
the guide in the affected directory first. Keep generated-artifact paths,
commands, and release states source-bound: do not copy a transient terminal
value into prose or turn a local render into a publication claim. Run the
Markdown-link audit after changing navigation or local links.

## Local guide hierarchy

Read the closest applicable guide in addition to this root contract:

| Path | Focus |
|---|---|
| `docs/AGENTS.md` | Reader-facing claims, commands, links, and rendered-publication boundaries |
| `data/AGENTS.md` | Claim ledger, frozen lane, and benchmark source contracts |
| `data/benchmarks/AGENTS.md` | Checksum-pinned external fixtures and authorized acquisition boundary |
| `scripts/AGENTS.md` | Thin orchestration and generated-artifact provenance |
| `src/AGENTS.md` | Deterministic domain logic, canonical serialization, and validation boundaries |
| `tests/AGENTS.md` | Real-fixture testing, deterministic seeds, and fail-closed coverage |
| `examples/AGENTS.md` | Offline public-API demonstrations and non-promotional outputs |
| `models/AGENTS.md` | GNN specifications for the declared generative models and the upstream custody pin |
| `manuscript/AGENTS.md` | Source templates, variables, figures, and render order |
| `manuscript/assets/AGENTS.md` | Authored source art and cover-asset provenance |
| `.agents/AGENTS.md` | Project-local skill scaffolding |

## Standard source loop

```bash
./.venv/bin/python -m compileall -q src scripts tests examples
./.venv/bin/ruff check src scripts tests examples
./.venv/bin/python -m pytest tests/ --cov=active_inference_power --cov-branch --cov-fail-under=95
env -u VIRTUAL_ENV uv run mypy src scripts tests examples --ignore-missing-imports
./.venv/bin/python scripts/validate_gnn_specs.py
```

For a source-bound candidate, use the complete recipe in
`docs/agent_instructions.md` or `docs/rendering_pipeline.md`: result generation
→ quality report → strict hydration → audits → render → unskipped review
→ finalizer. The quality suite creates its own real, session-scoped temporary
bundles, so candidate draft hydration is neither needed nor render-eligible.
Do not substitute an abbreviated result/quality/hydration sequence for that
ordering.

For publication work, use the controlled-renderer procedure in
`docs/rendering_pipeline.md`, inspect the PDF/HTML, and verify that every
figure and table is present. Use one new candidate root for the entire release
chain; never use `output/campaign/`, `output/precision/`, or
`output/campaign-v2/` as a release candidate. Keep deferred work in `TODO.md`
with a precise scope.

The CI fast-artifact lane is a narrower, read-only integrity check: after its
fast producer, quality receipt, hydration, and source audits, it runs
`scripts/audit_fast_bundle.py`. That audit deliberately does not render,
review, finalize, or publish a bundle, so it cannot replace the isolated
candidate chain above.
