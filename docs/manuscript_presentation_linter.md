# Manuscript presentation linter

`scripts/audit_manuscript_presentation.py` is a deterministic, read-only source
audit for reader-facing manuscript presentation. It complements strict
hydration: source `{{TOKEN}}` markers are permitted only when they occur in
`output/data/manuscript_variables.json`, while the optional resolved-tree check
requires every placeholder to have been replaced.

Run the complete presentation check after manuscript-variable generation. For
an isolated candidate, pass its explicit variables file rather than allowing
the default `output/` location to select another bundle:

```bash
./.venv/bin/python scripts/audit_manuscript_presentation.py \
  --variables-file "<candidate-root>/data/manuscript_variables.json" \
  --resolved-manuscript-dir "<candidate-root>/manuscript" \
  --report /tmp/active-inference-power-presentation-audit.json
```

`<candidate-root>` must be the same fresh root used for results, quality, and
strict hydration; it is not `output/` or a retained campaign namespace.

The command emits the same stable JSON object to standard output and to
`--report`, exits `0` for no findings, `1` for presentation findings, and `2`
for invalid inputs. The report contains a schema version, source and resolved
files checked, the declared acronym registry, a per-code count, and sorted
source locations. It does not write to `output/` unless the caller explicitly
selects an output path for the report.

## Failure classes

| Code | Rule | Repair |
|---|---|---|
| `acronym_first_use` | A term in the declared registry must first appear as its complete expansion followed by the abbreviation in parentheses. | Expand the first appearance, or add a genuinely new public abbreviation to `ACRONYM_REGISTRY` with its approved expansion. |
| `abstract_citation` | `manuscript/00_abstract.md` must not contain a Pandoc citation key. Internal `@fig:`, `@sec:`, `@eq:`, and `@tbl:` references are not citations. | Move the citation and its supporting discussion to a later section. |
| `abstract_doi_status` | The abstract must not discuss a DOI, Zenodo, or identifier-reservation status. | Keep prospective repository context concise; render a real DOI only through verified publication metadata and the cover field. |
| `abstract_internal_quality` | The abstract must not report test-count or coverage evidence. | Move release-quality evidence to reproducibility and release documentation. |
| `abstract_result_token` | The abstract must not inject a scenario-specific numerical result. | State the qualitative finding and direct readers to the generated results section. |
| `duplicate_cross_reference` | Do not prefix a renderer-owned cross-reference token with the same label, such as `Figure fig.` or `Figure @fig:overview`. | Use the cross-reference token by itself, or write a non-duplicative sentence. |
| `unresolved_token` | A source placeholder must have a declared generated value; a resolved manuscript may contain no placeholder, including an unterminated one. | Add the token to the variable generator and regenerate, or correct/remove the misspelled marker. |
| `numeric_literal` | Quantitative prose must use a generated `{{TOKEN}}`, rather than a hand-entered result, threshold, count, or percentage. | Route changing scientific values through `src/active_inference_power/manuscript_variables.py` and regenerate the bundle. |

The acronym registry is deliberately explicit rather than an all-caps heuristic.
It covers the manuscript's public statistical, artifact, and format vocabulary:
API, BH, BY, CDF, CS, DKW, DOI, DPI, FAIR, FDP, FDR, FWER, HTML, JSON, LOND,
LORD, MCSE, PDF, PRDS, SAFFRON, SE, SHA-256, TPR, and URL.

## Numeric exclusions

The numeric rule preserves line positions while excluding non-evidence syntax:

- inline/display mathematics and LaTeX environments;
- code spans and fenced commands;
- Pandoc citations and cross-reference/attribute syntax;
- Markdown link destinations;
- figure layout attributes such as `width=95%` and `height=0.38\textheight`;
- ordered-list markers, supplement labels such as `S1`, and the structural
  `SHA-256` spelling; and
- source support files (`AGENTS.md`, `README.md`, `SYNTAX.md`, and
  `preamble.md`), which are not rendered manuscript prose.

These exclusions are narrowly structural. They do not allow prose sample
sizes, effect sizes, error rates, coverage values, thresholds, replication
counts, or other scientific numbers to bypass manuscript variables.
