# Release review and finalization

The project distinguishes a read-only evidence review from the state-changing
publication finalizer. Both bind a result bundle to the current configuration,
claim ledger, and execution tree; neither turns a failed campaign into a
release.

## Read-only review

After generating results, quality metrics, and resolved manuscript files, bind
one shell variable to that exact candidate root and use it throughout the
review. In particular, the manuscript-presentation audit must receive that
candidate's variables file rather than the default `output/data` path:

```bash
candidate_root="<candidate-root>"
test -d "$candidate_root"
./.venv/bin/python scripts/audit_manuscript_presentation.py \
  --variables-file "$candidate_root/data/manuscript_variables.json" \
  --resolved-manuscript-dir "$candidate_root/manuscript"
./.venv/bin/python scripts/audit_figure_accessibility.py --output-root "$candidate_root"
./.venv/bin/python scripts/review_release.py --output-root "$candidate_root" \
  --skip-render --skip-clean-install
```

`review_release.py` never changes the manifest. It checks artifact schemas,
source-witness hashes, checkpoint provenance, figure/dashboard lineage,
resolved manuscript tokens, the manuscript-presentation audit, bounded local
Markdown destinations across the project documentation tree, and the figure
accessibility audit. The documentation check reads only local files; it neither
fetches remote URLs nor edits the candidate bundle. Once the
controlled renderer has created the PDF and HTML, omit `--skip-render`; the
default review also compares each rendered HTML image `alt` value with the
registry-authored alternative text and performs a clean wheel build/install
import outside the source tree. A pre-final bundle (`manifest.complete: false`)
may be reviewed after the renderer has created PDF/HTML files but before its
renderer-owned manifest records exist. A completed bundle must carry and
validate those records instead.

After the controlled render, synchronize registered alternative text, audit the
final HTML, capture renderer provenance, then run the unskipped review against
the same root:

```bash
./.venv/bin/python scripts/sync_rendered_figure_alts.py \
  --output-root "$candidate_root" --rendered-html "$candidate_root/web/index.html" \
  --include-auxiliary-pages
./.venv/bin/python scripts/audit_figure_accessibility.py \
  --output-root "$candidate_root" --rendered-html "$candidate_root/web/index.html" \
  --include-auxiliary-pages
./.venv/bin/python scripts/capture_renderer_provenance.py \
  --output-root "$candidate_root"
./.venv/bin/python scripts/review_release.py --output-root "$candidate_root" \
  --report "$candidate_root/review/release_review_final.json"
```

The synchronizer corrects Pandoc caption substitution only for registered
figure images in the combined HTML and any retained per-section pages. The
renderer receipt binds the final PDF/HTML/combined-Markdown bytes to the controlled
renderer Git HEAD and exact dirty-state digest, its renderer tree, the actual
Stage 03 and project-link/orchestration entry sources, and available
Pandoc/LaTeX versions. The reviewer remains read-only and verifies both
artifacts independently. A missing controlled renderer is a validation failure,
not evidence that the render is portable or renderer-independent.

Use `--report <PATH>` only when a persistent JSON report is useful. The report
destination is explicit so ordinary review never writes into a result bundle.

## Release identities and terminal gates

The reviewer and finalizer require the same independent identities. A release
bundle must satisfy all of the following:

- `results.meta.profile` is `release` or `publication`, agrees with
  `run_log.json`, and the run log has terminal status `passed`;
- every predeclared certificate record has the exact JSON Boolean
  `passes: true`, while both the manifest and run log record
  `certificates_pass: true`;
- `quality.json` validates as a structured quality snapshot and records the
  exact Boolean `quality_pass: true`; a nonempty report or a truthy text value
  is not a substitute for that gate. Its `quality_input_sha256` must also
  match the live, explicit authored quality surface: source and orchestration
  code, tests/fixtures, examples, documentation, manuscript and data inputs,
  project-agent/CI guidance, and declared root metadata. It excludes generated
  output, build products, virtual environments, and caches. The test suite is
  project-source anchored but creates its generated artifact fixtures in
  isolated temporary roots; it does not treat an ignored canonical `output/`
  tree or retained campaign receipt as current-source test input;
- configuration and claim-ledger SHA-256 witnesses agree among the live source
  files, manifest, results metadata, and run log;
- `execution_source_sha256` agrees among the manifest, results metadata, run
  log, and current checkout. This path-bound fingerprint covers every regular,
  non-cache file in the executable package, `scripts/`, and `data/`, together
  with the declared execution metadata (`pyproject.toml`, `uv.lock`, domain and
  experiment profiles, and `manuscript/config.yaml`). It deliberately differs
  from the broader quality witness, which also includes the authored review
  surface; and
- the checkpoint set is exactly the terminal `passed` or `resumed` stage set
  declared by the run log. Each checkpoint must agree with its filename stage,
  run identifier, profile, configuration and claim-ledger witnesses, and the
  execution-source fingerprint.

The execution witness deliberately detects code, lockfile, or benchmark-input
changes after a run. Regenerate the result bundle rather than trying to reuse a
checkpoint from a different execution tree.

## Semantic sidecar bindings

Digest checks establish byte identity, while the release contract also verifies
that the generated sidecars mean the same thing as the result document:

- `certificates.json` must exactly equal `results.certificates`;
- `action_loop_suite.json` must exactly equal `results.suite`;
- `formalism_registry.json` must exactly equal `results.formalism` and still
  resolve its declared code/formalism contract; and
- `demo/payload.json` must self-validate and bind its scenario, policy, and
  conditional-power seed to the declared suite.

Figure records are separately compared with result-derived lineage metadata,
and the review dashboard carries a full-result lineage digest. The reader
accessibility and manuscript-presentation audits remain required release checks
rather than cosmetic post-processing steps.

## Renderer-input witness

Manuscript hydration writes `RESULT_RENDER_INPUT_HASH`. It is a
self-normalizing SHA-256 witness over manuscript sources and their hydrated
counterparts, manuscript variables, the figure registry, and every publication
PNG. Its own value is normalized while hashing, so the witness can appear in
the rendered text without creating a circular digest.

The resolved manuscript, HTML, combined rendered Markdown, and PDF must all
carry the current configuration, claim-ledger, and renderer-input witnesses.
Changing a manuscript source, variable, registry, or figure therefore requires
hydration followed by a new render. The finalizer refreshes records only for
renderer-owned files after these checks; it does not refresh scientific or
source-bound records to make a changed bundle appear complete.

## Controlled-renderer provenance witness

After final HTML-alt synchronization, run:

```bash
./.venv/bin/python scripts/capture_renderer_provenance.py \
  --output-root "$candidate_root"
```

This generated `review/renderer_provenance.json` receipt is deterministic for
unchanged inputs. It records the controlled renderer's Git HEAD and
exact dirty-state digest, a path-bound closure of its renderer, Stage 03,
project-link/orchestration sources, and package/lock metadata, plus available
Pandoc and LaTeX version identities and the hashes of the final PDF, HTML, and
combined Markdown. The unskipped review validates it against the live controlled
environment and final rendered bytes; the finalizer records it in the completed
manifest. A missing controlled renderer, source change, toolchain change, or
render-byte change fails closed. The receipt makes the local rendering
dependency inspectable; it does not state that an unaccompanied bundle is
portable across machines or renderer checkouts.

## Publication finalization

Render the hydrated `active_inference_power` candidate through the controlled
release environment, synchronize the rendered alternatives, audit them, and capture
renderer provenance before the unskipped review. The finalizer may revalidate
the current receipt, but it does not replace that pre-review sequence. Then
run:

```bash
./.venv/bin/python scripts/finalize_release.py --output-root "$candidate_root"
```

The finalizer now requires the rendered PDF and HTML. It is the sole command
that may mark the selected candidate's `manifest.json` complete, and it refuses stale source
witnesses, checkpoints, figures, resolved manuscript files, or renderer-owned
records. Before it can set `complete: true`, it runs the unskipped read-only
 review, including manuscript presentation, rendered-text, rendered-alt-text,
 and clean-wheel-install checks, then writes
`<candidate-root>/review/release_review.json`. This deterministic receipt records a
passing verdict, run identifier, configuration, claim-ledger,
execution-source, and render-input witnesses, plus reviewed PDF/HTML/combined
Markdown hashes and the renderer-provenance receipt hash; the final manifest
hashes both `release_review` and `renderer_provenance`. Refreshing figures or
hydrating variables invalidates completion and requires a new render and
finalization.

## Campaign boundary

`output/campaign/` is a separate computational-sensitivity namespace, not an
alternate path to publication. `finalize_campaign.py` does not require a
quality report, hydrated manuscript, or rendered paper, but it still requires a
`campaign` profile, passed run log, an exact serialized certificate-policy
partition, passing declared terminal certificate results and matching terminal
flags, current source and execution witnesses, exact checkpoints, semantic
sidecars, figures, dashboard lineage, and accessibility audit. Diagnostic
certificate verdicts remain serialized and hash-bound even when they are not
terminal for the isolated campaign.

A campaign whose *terminal* certificate fails remains incomplete even when its
schemas and recorded hashes validate. A declared diagnostic failure may remain
visible in an otherwise complete isolated campaign, but it is not a pass,
release input, or publication claim. Preserve failed results as sensitivity or
method-review records; do not copy them into the release namespace, relabel
them as publication evidence, or bypass the campaign finalizer. The frozen v2
policy is documented in [the Storey decision record](storey_certificate_decision.md).

See the [rendering pipeline](rendering_pipeline.md),
[output inventory](output_inventory.md), and
[manuscript presentation linter](manuscript_presentation_linter.md) for the
producer-specific contracts.
