# Manuscript agent guide

The Markdown files here are source templates. The resolved manuscript under
the selected bundle root's `manuscript/` directory (default:
`output/manuscript/`), all quantitative tables, figure PNGs, registry metadata,
and quality receipts are generated artifacts: never repair them by hand.

## Reader and claim contract

- Lead with the investigator's declared design. The embedded agent faces
  within-trace uncertainty; reported power is a conditional operating
  characteristic estimated across replications, not an agent, niche, or
  deployment score.
- Keep posterior belief, calibrated p/e-value evidence, LORD++/SAFFRON
  alpha-wealth $W_t^\alpha$, e-LOND's cumulative nominal-level allocation
  trace $\sum_{i\le t}\alpha_i$, simulated cost $C$, source work
  $Q_{\mathrm{comp}}$, and local elapsed time $\tau_{\mathrm{wall}}$ distinct.
  e-LOND's recycled trace can exceed $\alpha$; it is not a remaining budget or
  alpha-wealth. A common trace does not make their claims or units
  interchangeable.
- Describe action-loop target e-processes as per-target optional-stopping
  evidence only. They do not supply FWER/FDR control; an all-action product is
  diagnostic and finite observed error summaries are estimands, not guarantees.
- Describe $C$ as scenario-local simulated resource cost. Cross-scenario cost
  or power-per-cost comparisons require matching hash-bound `cost_unit` and
  `cost_scale_id` values.
- State the estimand, replication unit, uncertainty convention, dependence,
  filtration, stopping, and validity boundary before interpreting a result.
  Monte-Carlo evidence, PRDS-style stress results, and calibration diagnostics
  are not proofs or arbitrary-dependence guarantees.
- Keep the abstract qualitative and repository-only. It must not contain result
  values, test or coverage metrics, Zenodo wording, or DOI-status discussion.
  The cover's neutral DOI-status line is controlled by `config.yaml` and is
  replaced only by verified publication metadata.

## Syntax, figures, and scholarship

- Use `../output/figures/<name>.png` and matching generated
  `{{FIGURE_*_CAPTION}}` and `{{FIGURE_*_ALT}}` tokens. Captions explain the
  reader question, visible comparison, takeaway, and boundary; alternative
  text describes the visual encoding rather than duplicating the caption.
  When a multi-panel figure includes a different validity status, preserve its
  registry `panel_validity` entry and name the affected panel in the caption;
  a whole-figure label cannot conceal an optional-stopping-ineligible
  fixed-horizon p-value contrast.
- Use Pandoc citations (for example, `[@benjamini1995controlling]`) and cross-references (`[@fig:label]`,
  `[@tbl:label]`, `[@sec:label]`, `[@eq:label]`). Do not hardcode display
  numbers or use raw `\ref{}`/`\eqref{}` in prose. `SYNTAX.md` records the
  registered figures and token rules.
- Add a primary citation only for the role it actually supports. Literature
  provides theory or context; it does not upgrade a finite synthetic result to
  a theorem.

## Required route after a source change

When changing methods or prose that affects a rendered claim, update the
corresponding methodology/setup/claim documentation and real tests, then
regenerate results, quality, variables, figures, and the resolved manuscript
before rendering through the controlled release environment. Follow
`../docs/rendering_pipeline.md`; do not finalize or publish a bundle unless the
separate release authority has been granted.
