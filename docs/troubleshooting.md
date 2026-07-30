# Troubleshooting

## `import active_inference_power` fails with a sibling-module error

Use the project interpreter and run from the project root. The package uses
relative imports and should pass:

```bash
./.venv/bin/python -c "import active_inference_power; print(active_inference_power.__all__[:3])"
```

If this fails, the checkout is stale or the dependency environment is incomplete.
Recreate the locked development environment before changing source or bypassing
the test command:

```bash
env -u VIRTUAL_ENV uv sync --locked --extra dev
```

Then repeat the import with `./.venv/bin/python`. Add `--extra demo` only when
the optional Streamlit application is needed.

## Manuscript hydration says results or quality are missing

Do not repair a partially generated candidate in place. Start one new,
previously absent candidate root and leave the default development tree and
retained campaign receipts untouched. Run, in order:

```bash
project_root="$(git rev-parse --show-toplevel)"
candidate_root="$project_root/output/hydration-recovery-$(date +%Y%m%d%H%M%S)"
test ! -e "$candidate_root"
test ! -L "$candidate_root"
./.venv/bin/python scripts/generate_results.py \
  --profile release --checkpoint-every 500 --output-root "$candidate_root"
# The quality suite makes its own real temporary generated bundles.
./.venv/bin/python scripts/generate_quality_report.py --output-root "$candidate_root"
# Strict hydration follows the passing quality receipt.
./.venv/bin/python scripts/z_generate_manuscript_variables.py --output-root "$candidate_root"
```

Keep `candidate_root` in the same shell, or set it again to the exact candidate
path before every later audit, render, review, or finalization command. Do not
use `output/campaign/`, `output/precision/`, or `output/campaign-v2/` as this
root. The direct `output/` root is a development namespace, not a recovery
candidate.

The quality producer writes its report at the selected root, while its test run
remains project-source anchored. Generated-artifact tests build real temporary
bundles; they do not depend on an ignored canonical `output/` fixture. If a
campaign finalizer reports that predecessor evidence escapes its output
namespace, inspect the resolved locations of `output/campaign/` and
`output/precision/`: the v2 campaign's declared predecessor receipts must
remain in the resolved canonical `output` namespace. Preserve the declared
predecessor receipts and repair the campaign provenance rather than redirecting
a release candidate, copying a predecessor file, or weakening the check.

Draft mode is available for editorial work, but it may emit `N/A` and must not
be used for a publication render.

## A literal token appears in rendered output

For the recovery candidate above, inspect its variables and scan its resolved
tree:

```bash
rg -n '\{\{' "$candidate_root/manuscript" || true
```

The live token test and strict generator should fail before rendering if a token
is missing.

Run the full reader-facing source check after hydration:

```bash
./.venv/bin/python scripts/audit_manuscript_presentation.py \
  --variables-file "$candidate_root/data/manuscript_variables.json" \
  --resolved-manuscript-dir "$candidate_root/manuscript"
```

## Figures are missing

For a candidate, restart the chain at its own `results.json` rather than
repairing a missing plot or registry by hand. The figure generator consumes
that exact artifact, so a missing plot usually means the analysis did not
finish.

If PNGs exist but the accessibility audit fails, regenerate through the result
producer rather than editing a registry or image by hand. For the recovery
candidate above, run:

```bash
./.venv/bin/python scripts/audit_figure_accessibility.py --output-root "$candidate_root"
```

An abbreviated invocation against the canonical `output/` root is
development-only and cannot validate a release candidate.

## Certificate failure

Read the estimate, bound, standard error, seed, and configuration in the
selected bundle's `certificates.json` (for the recovery flow,
`"$candidate_root/certificates.json"`). A failure is evidence that the selected
finite run is not consistent with its declared band; do not change the threshold
or delete the certificate without changing the method and documenting why.

## PDF rendering fails

The exact renderer is available only in the controlled release environment.
First validate the standalone chain. Release maintainers then use the
configured renderer environment described in
[controlled_renderer.md](controlled_renderer.md). Mermaid/Chrome and LaTeX
dependencies are renderer-environment concerns, not reasons to weaken project
evidence; public users can inspect the released PDF and HTML without that
environment.

On a user-owned TeX Live installation, a pre-flight check can pass while
XeLaTeX still needs packages in the user tree. Initialize that tree and install
the reported packages without changing the system TeX installation:

```bash
tlmgr init-usertree
tlmgr --usermode install ifoddpage relsize siunitx
```

Confirm discovery with `kpsewhich ifoddpage.sty`, then rerun the controlled
release renderer. Keep project-specific font paths out of `manuscript/preamble.md`:
the renderer supplies a portable XeLaTeX baseline.

## The release review or finalizer rejects the bundle

Do not bypass the failing witness. A changed source configuration, claim
ledger, figure refresh, or hydration step invalidates an earlier rendered
bundle. Regenerate serially, render again, then run the read-only review and
finalizer in that order. [Release review](release_review.md) lists the exact
checks and distinguishes ordinary review from the command that can mark a
rendered bundle complete.

## A downloaded demo payload is rejected

The optional demo downloads canonical JSON with configuration, trace,
visible-trace, and payload SHA-256 witnesses. Use the browser-free smoke lane
to diagnose the payload:

```bash
./.venv/bin/python scripts/smoke_demo.py --payload path/to/action_loop_trace.json
```

An edited or partial JSON file should fail validation instead of being treated
as a provenance-bearing trace.
