# Syntax guide

Use Pandoc citations (`[@key]`), cross-reference labels (`{#fig:name}`), and
relative figure paths under `../output/figures/`. Never hardcode figure/table
numbers or use raw LaTeX `\ref{}` in Markdown source.

The live substitution contract is the return value of
`active_inference_power.manuscript_variables.generate_variables`. The test suite scans manuscript
sections for uppercase token syntax and rejects missing or `N/A` values in strict
mode. `manuscript/SYNTAX.md` lists the current figure labels.
