# Manuscript numeric-literal audit

Quantitative manuscript prose is hydrated from `output/data/manuscript_variables.json`.
The regression test in `tests/test_manuscript_variables.py` scans the source
sections for digit-bearing literals after removing the explicitly non-evidence
contexts below:

- inline and display mathematics, where numeric constants are part of the
  declared equations;
- Pandoc citation and cross-reference syntax;
- figure layout attributes such as `width=95%`, which control rendering rather
  than an estimand;
- ordered-list markers and provenance spellings such as `SHA-256`, which are
  structural/documentation syntax rather than measured values;
- `manuscript/preamble.md`, whose numeric values are LaTeX page/layout settings.

Any remaining digit-bearing prose literal fails the test. This intentionally
does not allow hand-written sample sizes, uncertainty values, thresholds,
reliabilities, figure counts, coverage values, or benchmark counts. Add a new
allowlist category only when it is non-scientific rendering syntax and document
the reason here; changing scientific values must instead add a generated token.
