# Output conventions

Release-derived files under `output/` are generated from `manuscript/config.yaml`,
`src/`, and the scripts that orchestrate them, and should be regenerated rather
than hand-edited. This does not authorize removal of the explicitly preserved
campaign, precision, campaign-v2, or review receipts described below: those
paths are hash-bound historical evidence with their own retention boundaries.
The `output/` paths in this default-layout table are descriptive, not a
candidate command target: an isolated release uses one fresh explicit
`--output-root` in place of that prefix throughout its chain.

| Path | Producer | Role |
|---|---|---|
| `output/results.json` | `scripts/generate_results.py` | Full machine-readable experiment, including configuration, claim-ledger, execution-source witnesses in `meta`, and a deterministic source-work accounting block that excludes elapsed-time fields |
| `output/action_loop_suite.json` | `scripts/generate_results.py` | Scenario-indexed action-loop evidence, validity boundaries, and all-null selected-target calibration/DKW accounting |
| `output/formalism_registry.json` | `scripts/generate_results.py` | Executable formalism-to-code traceability, assumptions, evidence classes, and non-claims |
| `output/demo/payload.json` | `scripts/generate_results.py` | Deterministic optional-demo bundle with configuration, trace, visible-trace, and payload SHA-256 witnesses |
| `output/certificates.json` | `scripts/generate_results.py` / `run_certificates.py` | Exact projection of `results.certificates`, with seeded Boolean verdicts |
| `output/quality.json` | `scripts/generate_quality_report.py` | Structured test/coverage/exit-status snapshot with a path-bound test-harness witness; release requires `quality_pass: true` |
| `output/figures/*.png` | `src/active_inference_power/figures/` | Publication plots |
| `output/figures/figure_registry.json` | `src/active_inference_power/figures/` | Figure labels, fluent captions, authored alternative text, explicit reader question/comparison/takeaway/evaluation-perspective/boundary metadata, estimands, sample sizes, evidence classes, redundant encodings, scenario/result lineage, uncertainty conventions, and generated-file contract |
| `output/review/dashboard.html` | `src/active_inference_power/figures/` | Lightweight HTML dashboard; preserved across controlled-renderer rendering |
| `output/manifest.json` | `scripts/generate_results.py` | Artifact inventory plus configuration, claim-ledger, and execution-source linkage; generated output files may use an explicit `root: output` namespace when the output root is external |
| `output/run_log.json` | `scripts/generate_results.py` | Profile, terminal stage status, seeds, execution-source witness, and operational accounting |
| `output/checkpoints/*.json` | `scripts/generate_results.py` | Terminal stage state bound to its stage, profile, run identifier, source witnesses, and execution-source witness |
| `output/data/manuscript_variables.json` | `scripts/z_generate_manuscript_variables.py` | Resolved tokens, including the self-normalizing `RESULT_RENDER_INPUT_HASH` witness |
| `output/manuscript/*.md` | `scripts/z_generate_manuscript_variables.py` | Resolved renderer input carrying source and renderer-input witnesses |
| `output/web/index.html`, `output/web/manuscript__*.html` | controlled release renderer; `scripts/sync_rendered_figure_alts.py --include-auxiliary-pages` updates registered HTML figure alt text | Reader-facing combined and retained per-section HTML bound to hydrated witnesses; synchronize registered figure alternatives before rendered-alt audit, review, and finalization |
| `output/web/_combined_manuscript.md`, `output/pdf/active_inference_power_combined.pdf` | controlled release renderer | Reader-facing render artifacts, each bound to the hydrated witnesses |
| `output/review/renderer_provenance.json` | `scripts/capture_renderer_provenance.py` or the rendered finalizer path | Deterministic post-render receipt for controlled-renderer Git/dirty state, renderer/entrypoint/link source closure, available Pandoc/LaTeX identities, and final render bytes; required for a complete release, not a portability assertion |
| `output/review/release_review.json` | `scripts/finalize_release.py` after `scripts/review_release.py` passes | Deterministic terminal review receipt binding run/source/render witnesses, manuscript-presentation and rendered-alt-text checks, and PDF/HTML/combined-manuscript hashes |
| `manuscript/assets/active_inference_power_cover.png` | publication source asset | PDF title-page cover art; hashed in `output/manifest.json` |
| `output/campaign/`, `output/precision/` | prior campaign/audit executions | Preserved failed evidence bundles; neither is a reusable release root |
| `output/campaign-v2/` | campaign generator and finalizer with `conditional_power_campaign_v2.yaml` | Isolated certificate-policy campaign; results retain every certificate verdict and policy membership |
| `output/storey-method-review/storey_method_review.json`, `output/storey-method-review/manifest.json` | `scripts/run_storey_method_review.py` | Isolated frozen three-way Storey review, with direct-root result and manifest hashes; never a release or campaign finalizer input |
| `<new-root>/research_atlas.json`, `<new-root>/figures/`, `<new-root>/dashboard.json`, `<new-root>/manifest.json` | `scripts/run_research_atlas.py` | Bounded, non-promoted research-atlas bundle; `<new-root>` must not exist before the command runs |
| `<new-root>/complexity_experiments.json`, `<new-root>/complexity_experiments.png`, `<new-root>/index.html`, `<new-root>/manifest.json` | `scripts/run_complexity_experiments.py` | Isolated, host-specific repeated-timing diagnostic with raw samples, plan, exact transitive source-closure witness, and explicit non-claim; never a release or campaign input |
| `<new-root>/dobriban_bh_stress_bundle.json`, `<new-root>/figures/`, `<new-root>/dashboard.json`, `<new-root>/index.html`, `<new-root>/manifest.json` | `scripts/run_dobriban_bh_stress_bundle.py` | Isolated all-seven-chunk checksum-pinned finite Dobriban/BH stress receipt with source/plan witnesses, accessible figure registry, HTML alternative-text audit, and artifact hashes; never an Arb certificate, release, campaign, manuscript, certificate, or theorem input |
| `<explicit-root>/benchmark_preflight/provenance_dashboard.{json,html}`, `manifest.json` | `scripts/run_benchmark_preflight.py` | Offline fixture-provenance and checksum receipt; external-stress evidence only |

For an isolated release bundle, pass the same `--output-root` to result
generation, manuscript hydration, and release finalization. The quality
producer can write its report there too, so the report binds the candidate
bundle; however, it runs the repository's complete real-file test suite from
the project root and generates its artifact fixtures in session-scoped
temporary roots. Replace `output/` in the release rows of this table with the
candidate root for generated release artifacts; project-source paths in the
manifest remain in the project namespace. Campaign finalization is separate:
`finalize_campaign.py` consumes the generated campaign artifacts and does not
require a quality snapshot, hydrated manuscript, controlled render, or release
review.

Do not use the direct default root or a retained campaign root as the fresh
candidate: `output/`, `output/campaign/`, `output/precision/`, and
`output/campaign-v2/` are not interchangeable release namespaces.

## Quality-gate namespace

The full quality suite is anchored to the project source tree, but its
generated-artifact tests build real session-scoped temporary bundles. It does
not require an ignored canonical `$PROJECT_ROOT/output/` tree to exist or use
retained campaign receipts as fixture data. The frozen v2 predecessor paths,
currently `output/campaign/certificates.json` and
`output/precision/certificates.json`, remain subject to their campaign
finalizer's namespace and hash rules; that campaign-specific provenance
constraint does not turn them into release-quality fixtures. External or
nested candidate roots remain supported through manifest `root: output`
records.

## Certificate-policy lineage

`certificate_gate_policy` is a source-declared, exact partition of the
currently produced certificate names into `terminal` and `diagnostic`. The
generated result stores the policy, every individual Boolean verdict, terminal
results, diagnostic results, and the two failure lists. A diagnostic failure
therefore stays visible and hash-bound; it cannot disappear from a completed
campaign by omission.

The canonical release configuration assigns all certificates to `terminal`.
The preserved `campaign/` and `precision/` bundles keep their original
all-terminal interpretation and remain fail-closed. The separately configured
v2 campaign places only the same-vector `storey_power_gain` diagnostic while
requiring the independent `storey_split_confirmation` alongside the other
terminal checks. This is a finite, frozen campaign policy—not a proof of
same-vector Storey control or an authorization to promote campaign evidence.
See [the Storey decision record](storey_certificate_decision.md).

## Isolated research lanes

The research atlas is intentionally not a release-output variant. Its runner
requires a new, concrete output directory and rejects both the project root
and canonical `output/`; it atomically writes a self-contained diagnostic
bundle. The bundle carries a plan hash, source hashes, figure registry, and
explicit non-claim, but never a release-complete or publication-ready state.

The benchmark preflight runner accepts an explicit output root and writes a
`benchmark_preflight/` subdirectory there. It performs no network retrieval;
it verifies the committed manifest and locally staged fixture hashes, records
an unstaged source archive as unstaged, and writes a provenance dashboard. Its
manifest records dashboard paths relative to that lane and binds the runner and
deterministic benchmark-domain source files separately from fixture hashes. Its
receipt is preparation for independently reviewed acquisition, not a data
release or a biological, causal, family-level, or synthetic-validity claim.

The Storey method-review runner likewise requires a new output root and writes
only its frozen `storey_method_review.json` and manifest there. It compares a
same-vector Storey plug-in rule, the frozen fixed-lambda conservative same-vector rule,
and independent split confirmation on declared streams. The review is
hash-bound diagnostic evidence for the G1 policy decision, not a certificate
substitution, release, theorem, or tuning interface.

The complexity-experiment runner also requires a new output root and writes its
timing profile, local figure, HTML summary, and manifest directly there. It
records raw repeated wall and process elapsed-time samples after its declared
warmups, together with a bounded runner/domain source witness. The release
artifact's $Q_{\mathrm{comp}}$ block remains a deterministic source-work proxy;
the isolated $\tau_{\mathrm{wall}}$ profile is host-specific and cannot be
copied into a release, certificate, campaign, or statistical claim.

## Release lineage

The release chain uses five distinct provenance layers:

- configuration and claim-ledger witnesses bind the declared scientific
  inputs;
- `execution_source_sha256` binds every regular non-cache file in the
  executable package, `scripts/`, and `data/`, plus declared execution metadata
  (package/lock, domain/experiment profiles, and manuscript configuration), to
  the result, run log, manifest, and checkpoints; and
- `quality_input_sha256` binds `quality.json` to the broader authored quality
  surface: source/orchestration, tests and fixtures, examples, documentation,
  manuscript/data inputs, project-agent/CI guidance, and declared root
  metadata. It excludes generated output, build products, environments, and
  caches; and
- `RESULT_RENDER_INPUT_HASH` binds manuscript source/hydrated files,
  variables, registry, and PNG inputs to the final HTML and PDF.

For an actual PDF/HTML render, `renderer_provenance.json` is an additional,
post-render local-renderer witness: it binds the rendered bytes to the verified
controlled renderer and available toolchain. It is required for completed release
inventory, but does not make a bundle portable if the controlled renderer is not
available for validation.

The source, execution, and quality witnesses are validated against the live checkout.
`RESULT_RENDER_INPUT_HASH` is self-normalizing so its own printed value does
not change the manuscript-render digest. The separate renderer receipt binds
the final rendered bytes and is validated against the live controlled renderer;
it is not included in those rendered-byte hashes. Re-running the quality
producer first revokes any prior completion, then leaves hydration, rendering,
and finalization pending. Result-derived
sidecars are not merely listed in the manifest:
`certificates.json`, `action_loop_suite.json`, and `formalism_registry.json`
must exactly match their respective result blocks, while the demo payload must
validate its scenario, policy, and seed lineage. See [release review](release_review.md)
for the terminal profile, certificate, quality, and checkpoint gates.

Never hand-edit generated files. Do not delete, overwrite, or reuse preserved
failed campaign roots (`output/campaign/` and `output/precision/`) to make a
new run look complete; select a fresh isolated root instead. Delete only a
project-local regenerated output root when its scope is known, then rerun the
generation sequence. Use the read-only
[`scripts/review_release.py`](../scripts/review_release.py) before a rendered
finalization, and [figure accessibility](figure_accessibility.md) to audit the
registry/PNG reader-facing contract.
