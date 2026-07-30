# Figure accessibility audit

Every current generated figure registry entry (schema 2.4) records its
estimand, uncertainty encoding, sample size, evidence class, validity boundary,
and redundant visual encoding statement. It also records a reader question, a
result-bound visible comparison, a takeaway, an evaluation perspective, and a
concise explanation of how the marks, axes, and legend encode the comparison.
Together with `metadata.validity_boundary`, those are the five reader-first
components. The evaluation perspective states who declares the design, what
the agent can observe, and who scores a simulated trace. Schema 2.0 through
2.3 do not satisfy the current reader-facing contract; new bundles must carry
the fuller schema-2.4 fields, including `panel_validity` (an empty list when
one integrated contract covers the whole visual). The evidence class makes
clear whether a figure is a formal conditional result, analytic reference,
operating characteristic, calibration diagnostic, representative-trace
diagnostic, stress test, or provenance contract.

Captions and alternative text serve different readers. A caption states the
scientific interpretation and validity boundary; the registry's authored
alternative text describes what is visibly plotted. Hydration emits both as
separate tokens, and the rendered-HTML check below rejects a renderer that
substitutes caption text for the authored description.

For current schema-2.4 registries, the source audit also rejects an authored
alternative text that normalizes to the caption and requires quantitative
alternatives to name a visible axis, mark, panel, or comparable encoding. This
keeps a faithful renderer from preserving a source-level accessibility defect.

Run the read-only audit after result generation or figure refresh. For a
candidate, use that same fresh bundle root rather than the default `output/`
path:

```bash
./.venv/bin/python scripts/audit_figure_accessibility.py \
  --output-root "<candidate-root>"
```

The audit decodes every registered PNG, enforces a minimum 640×480 pixel
dimension, checks registry schema/lineage and reader-semantics fields, and
verifies that the reader is not asked to rely on color alone. It emits stable
JSON to standard output, returns zero only when every check passes, and never
rewrites figures or a manifest.

This is a release-quality check, not a substitute for reviewing the rendered
PDF/HTML. Inspect the actual pages after the controlled-renderer render and keep
scientific claim boundaries visible in captions and surrounding prose.
The current renderer preserves HTML structure and authored alternative text,
but does not create a tagged PDF/UA document; PDF structural checks and visual
inspection therefore do not certify screen-reader semantics for the PDF.

After rendering, synchronize the reader-facing HTML image attributes from the
registry, then run the read-only audit:

```bash
./.venv/bin/python scripts/sync_rendered_figure_alts.py \
  --output-root "<candidate-root>" --rendered-html "<candidate-root>/web/index.html" \
  --include-auxiliary-pages
./.venv/bin/python scripts/audit_figure_accessibility.py \
  --output-root "<candidate-root>" --rendered-html "<candidate-root>/web/index.html" \
  --include-auxiliary-pages
```

`<candidate-root>` is the exact fresh root already hydrated and rendered; do
not replace it with a retained campaign root.

The synchronizer changes only the `alt` attribute of registered `img`
elements. It leaves template icons and all other HTML untouched, prints its
own source digest and before/after HTML digests, and does not edit a manifest.
For the combined one-page artifact alone, omit `--include-auxiliary-pages` and
only `web/index.html` is processed. The opt-in also binds every retained
`web/manuscript__*.html` section page without requiring each page to contain
the whole figure suite. The audit remains read-only. `review_release.py`
requires this combined and auxiliary rendered-alternative-text check whenever
PDF/HTML render checks are enabled.

See [visualization notes](visualization_notes.md),
[release review](release_review.md), and the
[generated artifact inventory](output_inventory.md).
