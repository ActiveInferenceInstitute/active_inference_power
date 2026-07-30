# Scripts agent guide

Read `../docs/architecture.md` and `CONVENTIONS.md` before changing a script.
Preserve thin orchestration: configuration, subprocesses, file I/O, and
rendering calls belong here; statistical calculations belong in
`src/active_inference_power/`.

When a script adds, removes, or changes an artifact, update the output
inventory and the relevant schema/provenance contract in the same change. Use
an explicit fresh root for an isolated diagnostic or candidate; do not repair a
failed receipt by editing it or reuse campaign roots as release inputs.

Run the standalone quality gate and at least one real fast generation pass
after behavioral changes. For command, navigation, or artifact-description
changes, also run `../.venv/bin/python audit_documentation_links.py` from this
directory (or the equivalent project-root command) and `git diff --check`.

After the fast producer has a passing quality receipt and strict hydration,
`audit_fast_bundle.py` can validate its source witnesses, derivatives, figures,
and manifest bytes without mutation. It is intentionally a non-release audit:
do not substitute it for rendering, unskipped release review, or finalization.
