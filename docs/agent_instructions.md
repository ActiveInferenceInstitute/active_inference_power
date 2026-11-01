# Agent instructions

Read this file before modifying the project.

1. Preserve the `src/` domain / `scripts/` orchestration boundary.
2. Keep tests real and deterministic; do not use mocks or synthetic test doubles.
3. Validate configuration and evidence at boundaries; fail closed when required
   result artifacts are missing.
4. Keep changing manuscript numbers in generated variables, not prose.
5. Regenerate result, quality, figure, dashboard, and resolved-manuscript
   artifacts after method or configuration changes.
6. State the scope of every scientific claim: finite simulation, dependence
   regime, calibration assumption, or formal result.
7. Use the separately provisioned controlled renderer for release PDF/HTML
   rendering; never copy or present that infrastructure as a public dependency
   of this project.

Required source/test gates:

```bash
./.venv/bin/python -m compileall -q src scripts tests examples
./.venv/bin/ruff check src scripts tests examples
./.venv/bin/python -m pytest tests/ --cov=active_inference_power --cov-branch --cov-fail-under=95
env -u VIRTUAL_ENV uv run mypy src scripts tests examples --ignore-missing-imports
./.venv/bin/python scripts/validate_gnn_specs.py
```

For a release candidate, pass the same new `--output-root` to result generation,
manuscript hydration, the manuscript and figure audits, both release reviews,
and release finalization. Keep the root stable so hashes, resolved manuscript
files, rendered files, and the final manifest refer to one evidence bundle.
Never point this chain at `output/`, `output/campaign/`, `output/precision/`,
or `output/campaign-v2/`: those are a default or retained historical
namespaces, not a fresh release candidate.

The quality producer can write `quality.json` to that candidate root, but it
always runs the repository's real-file tests from the project root. Keep the
source tree stable while it runs: generated-artifact tests create real,
session-scoped temporary bundles and status records rather than consuming an
ignored canonical `output/` tree. Retained campaign and precision receipts are
provenance inputs to their own campaign finalizer, not current-source test
fixtures. Consequently, the candidate does not need a draft hydration before
quality; strict hydration begins only after the passing quality receipt. See
[rendering_pipeline.md](rendering_pipeline.md) for the exact boundary.

```bash
release_root="$PWD/output/release-candidate-$(date +%Y%m%d%H%M%S)"
test ! -e "$release_root"
./.venv/bin/python scripts/generate_results.py \
  --profile release --checkpoint-every 500 --output-root "$release_root"
# The quality suite creates its own real temporary artifact bundles.
./.venv/bin/python scripts/generate_quality_report.py --output-root "$release_root"
# Strict hydration binds this candidate's resolved manuscript to quality.json.
./.venv/bin/python scripts/z_generate_manuscript_variables.py --output-root "$release_root"
./.venv/bin/python scripts/audit_manuscript_presentation.py \
  --variables-file "$release_root/data/manuscript_variables.json" \
  --resolved-manuscript-dir "$release_root/manuscript"
./.venv/bin/python scripts/audit_figure_accessibility.py --output-root "$release_root"
./.venv/bin/python scripts/review_release.py --output-root "$release_root" \
  --skip-render --skip-clean-install
```

Render that exact root through the reversible binding procedure in
[`rendering_pipeline.md`](rendering_pipeline.md). After the controlled renderer
returns, synchronize the rendered figure alternatives, audit the rendered HTML,
capture renderer provenance, then run the unskipped review and finalizer in
this order:

```bash
./.venv/bin/python scripts/sync_rendered_figure_alts.py \
  --output-root "$release_root" --rendered-html "$release_root/web/index.html" \
  --include-auxiliary-pages
./.venv/bin/python scripts/audit_figure_accessibility.py \
  --output-root "$release_root" --rendered-html "$release_root/web/index.html" \
  --include-auxiliary-pages
./.venv/bin/python scripts/capture_renderer_provenance.py \
  --output-root "$release_root"
./.venv/bin/python scripts/review_release.py --output-root "$release_root" \
  --report "$release_root/review/release_review_final.json"
./.venv/bin/python scripts/finalize_release.py --output-root "$release_root"
```

`finalize_release.py` only completes a rendered publication bundle; hydration
alone never completes a release. A campaign finalizer has a separate contract:
it validates campaign artifacts directly and does not require a quality
snapshot, manuscript hydration, a template render, or `review_release.py`.
