# Docs agent guide

Documentation is a reader-facing contract for this project, not a copy of the
controlled renderer's internal documentation. Before editing a technical page, read the
nearest directory guide and the producer or consumer it describes. Keep
commands project-local, use paths and flags that the current CLI accepts, and
update the relevant contract when a schema, artifact, or release boundary
changes.

## Evidence and publication boundaries

- Distinguish a deterministic source contract, a finite simulation result, a
  rendered local candidate, and an external publication. A green local check
  never by itself establishes a DOI, repository release, theorem, or deployed
  result.
- Explain the investigator, embedded agent, and synthetic task-environment
  roles before using the word “power”; do not imply that statistical power is a
  property of an agent or niche alone.
- Keep posterior beliefs, e-process evidence, online alpha accounting,
  simulated cost, source work, and host timing separate in prose just as they
  are separate in result fields and figures.
- Name LORD++/SAFFRON's procedural alpha-wealth $W_t^\alpha$ separately from
  e-LOND's cumulative nominal-level allocation trace
  $\sum_{i\le t}\alpha_i$. e-LOND emits nominal test levels $\alpha_t$ and
  uses the e-value rule $E_t\ge 1/\alpha_t$; after rejection recycling its
  cumulative trace can exceed $\alpha$. It is neither a remaining alpha budget
  nor alpha-wealth, and neither accounting quantity is e-process evidence or
  simulated resource cost.
- For a mixed-status multi-panel figure, document the registry's
  `panel_validity` entry and the affected panel's boundary. A figure-level
  validity label must not conceal, for example, a fixed-horizon p-value path
  that is invalid if used for optional stopping.
- Never hand-edit `output/` or quote changing values from a rendered artifact.
  Link to the producer and state the applicable evidence boundary instead.

The read-only `scripts/audit_fast_bundle.py` is a non-release fast-profile
integrity audit. Describe it only after its producer, quality receipt, and
strict hydration; it does not render, finalize, or establish publication
readiness.

## Documentation checks

Run this after changing navigation or a local Markdown destination:

```bash
./.venv/bin/python scripts/audit_documentation_links.py
```

Use `git diff --check` for Markdown whitespace and run the focused real-file
tests when a documentation change is covered by a contract test. For
reader-facing figures or publication instructions, also follow the rendered
HTML/PDF review order in [rendering_pipeline.md](rendering_pipeline.md).
