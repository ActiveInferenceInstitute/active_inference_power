---
name: active-inference-power
description: Project-local workflow for statistical methods, evidence calibration, generated figures, and publication artifacts.
version: 0.2.0
author: docxology
license: MIT
tags: [statistics, multiple-testing, active-inference, reproducibility]
---

# active-inference-power

Use this skill when changing the package, experiment, figures, or manuscript.

## Quick reference

```bash
./run.sh fast
./run.sh all
```

See [the gate run guide](../../../docs/RUN_GUIDE.md) for individual arms;
CI calls the same dispatcher. The fast selection is not release-quality evidence.

For an isolated release candidate, follow the complete documented sequence in
[`docs/agent_instructions.md`](../../../docs/agent_instructions.md): result
generation, quality report, strict hydration, audits, sibling render,
unskipped review, and finalization all use the same new output root. The
quality suite creates its own real session-scoped temporary bundles, so draft
hydration of the candidate is not needed. The short source checks above are
not a substitute for that evidence chain.

For CI-style producer coverage, a fast-profile bundle with a passing quality
receipt and strict hydration can be checked read-only with
`scripts/audit_fast_bundle.py`. It verifies source witnesses and generated
derivatives while deliberately leaving the manifest incomplete; it neither
renders nor replaces the isolated candidate chain.

## Contracts

- No mocks; use real distributions, RNGs, files, and the pinned pymdp agent.
- Keep calculations in `src/` and orchestration in `scripts/`.
- Regenerate outputs; never hand-edit them.
- Keep PRDS, Monte-Carlo, calibration, and active-inference claims bounded.

## Pitfalls

- Do not combine a historical campaign receipt with a source-current release
  candidate, or rerun a producer against a retained campaign path.
- Do not report investigator-estimated conditional power as an intrinsic agent
  or task-environment property, and do not conflate e-process evidence with
  online alpha accounting or simulated cost.
- Do not let a whole-figure validity label conceal an exceptional panel: keep
  the registry's reader semantics and `panel_validity` metadata current,
  especially for fixed-horizon p-value contrasts under optional stopping.
- Do not render or finalize before results, quality, strict hydration, and the
  reader-facing audits agree on one new output root.

## Cross-references

- [`AGENTS.md`](../../../AGENTS.md)
- [`README.md`](../../../README.md)
- [`docs/PACKAGE_CONTRACT.md`](../../../docs/PACKAGE_CONTRACT.md)
