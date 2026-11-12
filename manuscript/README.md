# Manuscript directory

The Markdown files are source templates. Numbers that vary with the experiment
use substitution tokens produced by `src/active_inference_power/manuscript_variables.py`; the renderer
consumes the resolved copies under the selected bundle root's `manuscript/`
directory (default: `output/manuscript/`).

## Sections

- `00_abstract.md` through `08_supplemental_certificate_audit.md` contain the
  narrative and the generated adaptive-FDR supplement.
- `99_references.md` is the bibliography boundary.
- `config.yaml` is the experiment and publication policy.
- `assets/active_inference_power_cover.png` is the title-page artwork; it is
  source art, not a quantitative figure or scientific evidence artifact.
- `SYNTAX.md` records labels, figures, and token rules.
- `preamble.md` contains renderer-specific LaTeX packages.

## Reader-facing metadata

The abstract explains the investigator-facing conditional-power question in
qualitative terms. It may name the intended repository, but it must not contain
result values, quality receipts, DOI-status language, or archive-status claims.
`config.yaml` supplies the cover's neutral `DOI: to be assigned at publication`
line until real, verified publication metadata is available; never add a
placeholder identifier.

Figures are declared in `SYNTAX.md` and generated from the result artifact.
Their captions and alternative text come from the figure registry through
separate tokens, so a source edit must preserve the matching caption/alt pair
and registered figure label. A mixed-status multi-panel visual also needs its
registered `panel_validity` metadata: it records which panel has the different
contract, such as the fixed-horizon p-value path that is invalid if used for
optional stopping in the sequential figure.

Action-loop prose must identify target e-processes as per-target
optional-stopping evidence only, with no FWER/FDR claim; an all-action product
is a diagnostic. Treat simulated cost as scenario-local and compare cost or
power-per-cost across scenarios only when their declared `cost_unit` and
`cost_scale_id` agree.

## Source-to-render route

Run the analysis and quality scripts before hydration. Strict hydration refuses
to proceed when result or quality artifacts are missing.

After hydration, run the source and resolved-tree checks in
[the presentation-linter guide](../docs/manuscript_presentation_linter.md).
Set `isolated_root` to the prepared candidate. The authoritative
repository-owned rendering command is:

```bash
./.venv/bin/python scripts/render_manuscript.py --output-root "$isolated_root"
```

It delegates to the provisioned controlled renderer's Stage 03, checks the
candidate's existing strict variables and figure registry through the
read-only pre-render release review, and temporarily binds that candidate
without installing an engine or changing renderer links. Follow
[controlled-renderer provisioning](../docs/controlled_renderer.md) and
[the complete rendering pipeline](../docs/rendering_pipeline.md) for the
required candidate preparation, post-render audits, review, and finalization.
Missing renderer provisioning fails closed; neither a reference template
checkout nor host Pandoc/LaTeX alone supplies a compatible release renderer.

Do not edit a resolved `output/manuscript/` file to repair a token or value.
Changing numbers remain generated in the selected candidate's
`data/manuscript_variables.json`; captions, alternative text, and validity
metadata remain generated in `figures/figure_registry.json`. Do not duplicate
their numeric contents in this guide or scientific source prose. A prose,
figure, configuration, or bibliography edit changes render inputs, so hydrate
and render a fresh candidate before treating a PDF or HTML review as current.
For an isolated candidate, the presentation audit must receive that candidate's
explicit variables file and resolved manuscript directory; do not let its
default `output/` paths select a different bundle.
