# Rendering pipeline

The project-local candidate chain is described in full below. The abbreviated
default-root development sequence can be useful while developing, but it is
not a release-candidate recipe:

```bash
./.venv/bin/python scripts/generate_results.py
./.venv/bin/python scripts/generate_quality_report.py
./.venv/bin/python scripts/z_generate_manuscript_variables.py
```

For a fresh candidate, use result generation → quality → strict hydration.
The quality suite creates real, session-scoped temporary generated bundles of
its own, so it does not need draft hydration of the candidate before running.
Use the complete order in [Rendering a new candidate
bundle](#rendering-a-new-candidate-bundle).

For a release candidate, pass the same `--output-root <DIR>` to result
generation, hydration, the reader-facing audits, both release reviews, and
release finalization. `generate_quality_report.py` can also write its report to
that root. The controlled release renderer currently does not: its Stage 03
interface resolves the project's canonical `output/` path. It is not
distributed as a public dependency; see
[controlled_renderer.md](controlled_renderer.md) for the public
reproducibility boundary.

## Quality-gate fixture boundary

`--output-root` controls where a candidate's generated artifacts and its
`quality.json` are written; it does **not** relocate the repository test
harness. `generate_quality_report.py` always runs the real-file `tests/` suite
from the project root. Its generated-artifact tests create real,
session-scoped temporary bundles and schema-valid temporary status records;
they do not consume an ignored canonical `$project_root/output/` tree or a
retained campaign receipt as a fixture. A passing report binds the selected
candidate's quality artifact to the current authored quality surface through
`quality_input_sha256`: source and orchestration code, tests and fixtures,
examples, documentation, manuscript and data inputs, project-agent/CI guidance,
and declared root metadata. That quality witness excludes generated output,
build products, virtual environments, and caches. It is broader than
`execution_source_sha256`, which instead binds the executable result/review
closure (the package, every script, data inputs, and declared execution
metadata) to a run.

The frozen v2 campaign's declared predecessor receipts — currently
`output/campaign/certificates.json` and `output/precision/certificates.json`
— still must resolve *inside* `$project_root/output/` when the campaign
finalizer validates that lane. That campaign-specific provenance rule does not
make those receipts release-quality fixtures. Preserve and archive historical
receipts before a campaign refresh; do not copy them into a release candidate
merely to make a test pass.

The final rendering engine is available only in the separately provisioned
controlled release environment. This page is a maintainer runbook: public
users can reproduce results, figures, and hydrated manuscript inputs here, and
inspect the shipped PDF/HTML assets, but do not need an inaccessible renderer
checkout. The controlled renderer must consume resolved files, not the
token-bearing templates.

Because the real-file quality suite builds its own generated test bundles,
candidate strict hydration starts only after a passing `quality.json` exists.
Draft hydration remains available for editorial work when evidence is absent,
but it is not part of a source-bound candidate chain and is never eligible for
render review or finalization.

## Rendering a new candidate bundle

Stage 03 has no `--output-root` option. To render a new candidate without
overwriting the existing canonical output, first hydrate and audit that bundle
with its explicit root. Confirm the quality-gate fixture boundary above before
running the quality producer. Then, only while
no other process is reading or writing this project's `output/`, temporarily
bind the renderer's canonical path to the candidate bundle. This is a
reversible filesystem binding, not a copy of the template engine.

```bash
project_root="$(git rev-parse --show-toplevel)"
cd "$project_root"
bundle_name=release-candidate-$(date +%Y%m%d%H%M%S)
isolated_root="$project_root/output/$bundle_name"
test ! -e "$isolated_root"
test ! -L "$isolated_root"

./.venv/bin/python scripts/generate_results.py \
  --profile release --checkpoint-every 500 --output-root "$isolated_root"
# This writes the candidate report; pytest creates its own real temporary
# artifact fixtures from the current source tree.
./.venv/bin/python scripts/generate_quality_report.py --output-root "$isolated_root"
# Strict hydration binds the resolved manuscript to the candidate's quality
# receipt and is the first candidate hydration in this chain.
./.venv/bin/python scripts/z_generate_manuscript_variables.py --output-root "$isolated_root"
./.venv/bin/python scripts/audit_manuscript_presentation.py \
  --variables-file "$isolated_root/data/manuscript_variables.json" \
  --resolved-manuscript-dir "$isolated_root/manuscript"
./.venv/bin/python scripts/audit_figure_accessibility.py --output-root "$isolated_root"
./.venv/bin/python scripts/review_release.py --output-root "$isolated_root" \
  --skip-render --skip-clean-install

```

In the controlled maintainer environment, set
`AIP_RELEASE_RENDERER_ROOT` to the configured renderer checkout. First check
the link state without changing it, then render without automatic manuscript
hydration. Use the scoped function below rather than a manual
move/link/restore sequence: it installs an `EXIT`/`INT`/`TERM` trap before it
moves `output/`, restores only the exact binding it created, and refuses to
overwrite an unexpected path.

```bash
render_candidate_through_template() (
  set -eu
  : "${AIP_RELEASE_RENDERER_ROOT:?set to the controlled renderer checkout}"
  output_path="$project_root/output"
  candidate_root="$isolated_root"
  case "$candidate_root" in
    "$output_path"/*) ;;
    *)
      printf '%s\n' 'candidate must be a descendant of the project output root' >&2
      exit 2
      ;;
  esac

  relative_root=${candidate_root#"$output_path"/}
  backup_root="$project_root/.output-render-bind-$(date +%Y%m%d%H%M%S)"
  bound_root="$backup_root/$relative_root"

  path_present() {
    test -e "$1" || test -L "$1"
  }

  restore_output() {
    # `status` is a read-only shell parameter in zsh; use a portable local
    # name so the EXIT trap can restore the original output tree under either
    # zsh or bash.
    exit_status=$?
    trap - EXIT INT TERM

    if [ -L "$output_path" ]; then
      if [ "$(readlink "$output_path")" != "$bound_root" ]; then
        printf '%s\n' 'refusing to replace an unexpected output symlink' >&2
        exit 1
      fi
      unlink "$output_path"
    elif path_present "$output_path" && path_present "$backup_root"; then
      printf '%s\n' 'refusing to overwrite an unexpected output path during restoration' >&2
      exit 1
    fi

    if path_present "$backup_root"; then
      if path_present "$output_path"; then
        printf '%s\n' 'refusing to restore while output is unexpectedly occupied' >&2
        exit 1
      fi
      mv "$backup_root" "$output_path"
    elif ! path_present "$output_path"; then
      printf '%s\n' 'output restoration failed: neither original tree nor backup remains' >&2
      exit 1
    fi

    exit "$exit_status"
  }

  trap restore_output EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  test -d "$candidate_root"
  test -d "$output_path"
  test ! -L "$output_path"
  test ! -e "$backup_root"
  test ! -L "$backup_root"
  mv "$output_path" "$backup_root"
  test -d "$bound_root"
  ln -s "$bound_root" "$output_path"
  test -L "$output_path"
  test "$(readlink "$output_path")" = "$bound_root"

  (
    cd "$AIP_RELEASE_RENDERER_ROOT"
    uv run python -m infrastructure.orchestration link-projects --dry-run
    env TEMPLATE_SKIP_LINK_SYNC=1 uv run python scripts/pipeline/stage_03_render.py \
      --project working/active_inference_power \
      --skip-manuscript-hydration
  )
)

render_candidate_through_template
```

The function restores the original `output/` tree before it returns whether the
renderer succeeds, fails, or receives `INT` or `TERM`. The explicit skip is
essential: otherwise Stage 03 would hydrate the canonical path rather than the
already validated candidate bundle. After it returns, the rendered files remain
under `$isolated_root/web/` and `$isolated_root/pdf/`.

Then run the post-render synchronization, audits, review, and finalizer against
that same candidate:

```bash

./.venv/bin/python scripts/sync_rendered_figure_alts.py \
  --output-root "$isolated_root" --rendered-html "$isolated_root/web/index.html" \
  --include-auxiliary-pages
./.venv/bin/python scripts/audit_figure_accessibility.py \
  --output-root "$isolated_root" --rendered-html "$isolated_root/web/index.html" \
  --include-auxiliary-pages
./.venv/bin/python scripts/capture_renderer_provenance.py \
  --output-root "$isolated_root"
./.venv/bin/python scripts/review_release.py --output-root "$isolated_root" \
  --report "$isolated_root/review/release_review_final.json"
./.venv/bin/python scripts/finalize_release.py --output-root "$isolated_root"
```

Pandoc may emit a figure caption as an HTML image's alternative text, so run
the project-owned synchronizer before the rendered-alt audit,
renderer-provenance capture, review, and finalizer. It changes only registered
figure `img` attributes, records its own source and before/after HTML digests
on stdout, and does not edit the manifest. The auxiliary option covers retained
per-section HTML while leaving the combined one-page interface available for
focused diagnostics. `capture_renderer_provenance.py` writes the deterministic
`review/renderer_provenance.json` receipt only after PDF, HTML, and combined
Markdown exist. It binds their final bytes to the controlled renderer Git HEAD and
dirty-state digest, the complete `infrastructure/rendering` source tree, the
actual Stage 03 entrypoint and project-link/orchestration sources, and available
Pandoc/LaTeX version identities. It records a local renderer dependency; it is
not a claim that a bundle is portable when the controlled renderer is
unavailable. The
unskipped review validates this receipt fail-closed, and the finalizer captures
it again when a physical render exists. The final read-only review must pass
before the finalizer may mark the release complete. Do not use this binding for
`output/campaign/`, `output/precision/`, or `output/campaign-v2/`; campaign
receipts are never release-render inputs.

Before calling a publication-ready result, verify:

- `scripts/audit_manuscript_presentation.py` passes with
  `$isolated_root/data/manuscript_variables.json` and
  `$isolated_root/manuscript`;
- `scripts/audit_figure_accessibility.py --output-root "$isolated_root"` passes;
- `scripts/review_release.py --output-root "$isolated_root" --skip-render --skip-clean-install`
  passes before rendering, and the unskipped review passes afterward;
- `$isolated_root/review/renderer_provenance.json` is captured after the final
  HTML-alt synchronization and validates the controlled renderer revision,
  dirty-state, renderer/entrypoint/link source closure, available toolchain,
  and final render bytes;
- `$isolated_root/manifest.json` binds the current configuration, claim ledger, and
  execution-source fingerprint to results, run log, and checkpoints;
- the release profile, passed run log, every declared terminal certificate,
  and `quality.json` (including its current test-harness witness) all satisfy
  their strict terminal gates; diagnostic
  certificate records, if any, remain serialized and visible;
- `$isolated_root/review/dashboard.html` survives the controlled renderer's
  cleanup of `$isolated_root/web/`; and
- the rendered PDF/HTML contains the figures, tables, citations, limits, and
  source-witness hashes.

Hydration writes `RESULT_RENDER_INPUT_HASH` after it assembles the manuscript
inputs. The self-normalizing digest covers manuscript source and resolved
files, variables, the figure registry, and all publication PNGs. It must be
present in the resolved manuscript and in each rendered form (HTML, combined
Markdown, and PDF). If any of those inputs changes, hydrate and render again;
timestamps alone are not treated as sufficient freshness evidence.

After rendering, `scripts/review_release.py --output-root "$isolated_root"`
checks rendered text, the renderer provenance receipt, and a clean wheel
import. Then `scripts/finalize_release.py` refreshes hashes only for
renderer-owned `web:*` and `rendered_*` records and records the two terminal
receipts; it may mark the bundle complete only after they validate. Scientific
results, certificates, figures, configuration, claim ledger, checkpoints, and
resolved manuscript sources remain fail-closed: an unexpected change in any of
those artifacts is still a finalization error.

The campaign output root is not part of this rendering chain.
`finalize_campaign.py` validates its campaign-specific provenance, sidecars,
figures, dashboard, checkpoints, and terminal certificate witnesses directly;
it does not require a quality snapshot, hydrated manuscript, controlled render,
or `review_release.py`. A campaign may be retained as a schema-valid failed
sensitivity record, but a failed campaign certificate cannot be converted into
release or publication evidence by rendering it.

If the controlled renderer needs Chrome for Mermaid or LaTeX packages, fix
that maintainer environment; do not replace figures with unvalidated images.

See [release review](release_review.md) for the exact review/finalization
division and [output conventions](output_conventions.md) for the manifest
namespace contract.
