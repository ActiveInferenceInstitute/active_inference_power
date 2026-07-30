# .agents — AGENTS.md

Per-folder technical reference for the project-local agent scaffolding.

| Subdirectory | Purpose |
| --- | --- |
| [`skills/`](skills/AGENTS.md) | Project-local skills. Each ships `SKILL.md` with YAML frontmatter. |

## When to update

- A new skill specific to this project lands → add a folder under
  `skills/<name>/` with `SKILL.md`, `AGENTS.md`, `README.md`.
- An existing project workflow changes → update its skill, folder guide, and
  cross-references together; a skill must not silently point at a sibling
  project or an obsolete command sequence.
