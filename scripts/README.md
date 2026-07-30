# Scripts

Scripts are thin command-line orchestrators:

- `generate_results.py` loads YAML, calls `active_inference_power`, and writes results/figures;
  `--profile fast|release|campaign|benchmark|publication`, `--workers`, `--resume`,
  `--checkpoint-every`, `--output-root`, and `--seed` control reproducible
  execution without changing the scientific layer. `--output-root` may point
  outside the checkout; generated files are recorded in the manifest's output
  namespace. Most checkpoints are stage-complete resume boundaries. The
  independent null-calibration stage is incrementally resumable: each chunk
  records a deterministic child-stream cursor and a serializable raw-p-value
  accumulator. Other stages remain stage-complete until an associative,
  scientifically equivalent reducer is implemented;
- `campaign` resolves the expanded YAML-declared effect/dependence/reliability
  design and larger replication budgets into an isolated output namespace;
- `finalize_campaign.py` validates a completed campaign bundle and marks its
  manifest complete without conflating it with the publication release;
- `finalize_release.py` is the rendered-publication transition only: it binds
  current configuration/ledger witnesses, checkpoints, resolved manuscript,
  figures, PDF, HTML, and the full reader-facing audit suite before it can mark
  a release manifest complete;
- `review_release.py` composes the same source-bound checks into a read-only
  JSON review, including manuscript presentation, bounded local Markdown-link,
  figure-accessibility, and post-render controlled-renderer provenance audits,
  with optional rendered-text and
  clean-wheel-install checks;
- `capture_renderer_provenance.py` runs after sibling rendering and HTML-alt
  synchronization. It writes a deterministic receipt for the final PDF/HTML/
  combined Markdown bytes, controlled-renderer Git/dirty state, renderer/Stage-03/
  link-orchestration source closure, and available Pandoc/LaTeX identities;
  the receipt is required for a completed release but is not a portability
  claim when the controlled renderer is unavailable;
- `regenerate_figures.py` refreshes all visual files and result-derived captions
  from an existing selected-bundle `results.json` (default:
  `output/results.json`) without rerunning simulations. It
  stages the complete figure set, atomically replaces the visual artifacts,
  and refreshes their manifest hashes last while invalidating any prior release
  completion until hydration, rendering, and finalization are repeated;
- `audit_figure_accessibility.py` decodes the generated PNGs and validates the
  reader-facing registry metadata, including panel-specific validity for a
  mixed-status visual, without changing the bundle;
- `sync_rendered_figure_alts.py` runs after the sibling HTML render and updates
  only registered figure-image `alt` attributes from the authored registry;
  its stdout receipt records synchronizer and before/after HTML digests while
  leaving the manifest untouched for the subsequent review/finalizer boundary;
  pass `--include-auxiliary-pages` to bind retained per-section HTML pages too;
- `audit_manuscript_presentation.py` checks source/resolved manuscript
  presentation contracts and emits a deterministic JSON report;
- `audit_documentation_links.py` verifies local Markdown destinations;
- `smoke_demo.py` loads a committed or downloaded demo JSON bundle and checks
  its configuration and trace hashes without importing Streamlit or a browser;
- `generate_quality_report.py` runs the real test/coverage gate and accepts
  `--output-root` so `coverage.json` and `quality.json` stay beside the result
  bundle. It records a path-bound test-harness witness and first revokes any
  existing completion before replacing quality evidence;
- `audit_fast_bundle.py` is the read-only non-release companion for a
  fast-profile bundle after its passing quality receipt and strict hydration.
  It validates current-source and execution witnesses, schemas, result-derived
  artifacts, figure accessibility, and every manifest-recorded file while
  requiring the manifest to remain incomplete; it never renders, reviews, or
  finalizes a publication bundle;
- `run_certificates.py` emits the certificate-only JSON artifact and accepts
  explicit `--m`, `--n-reps`, and `--seed` accounting overrides;
- `run_research_atlas.py` requires a new explicit `--output-root` and writes a
  bounded R1--R5/G4 research-atlas receipt with figures, a registry, dashboard,
  manifest, and explicit non-claims. It is an isolated diagnostic, never a
  release or manuscript producer;
- `run_storey_method_review.py` requires a new explicit `--output-root` and
  writes the frozen three-way Storey method-review result and manifest at that
  direct root. It has no tuning flags and is a review record, not a campaign or
  release transition;
- `run_benchmark_preflight.py` requires an explicit root, performs no download,
  and writes an offline `benchmark_preflight/` provenance lane after verifying
  the committed manifest and locally staged fixture hashes. It is external
  stress provenance only;
- `fetch_benchmarks.py` is the separate operator-facing network acquisition
  command. Ordinary tests, result generation, review, and preflight never call
  it. It checks the selected manifest record's checksum before retaining an
  archive;
- `run_complexity_experiments.py` requires a new explicit `--output-root` and
  atomically writes a host-specific repeated-timing profile, local figure, HTML
  summary, source witness, and manifest there. It is an isolated diagnostic:
  its elapsed-time samples never enter `output/results.json`, certificates,
  campaign finalization, or release finalization;
- `run_dobriban_bh_stress_bundle.py` requires a new explicit `--output-root`
  outside canonical `output/` and runs all seven checksum-pinned Dobriban/BH
  chunks with no tuning flags. It atomically writes a self-hashed finite stress
  bundle, audited figure/registry, dashboard, HTML index, and manifest. This
  offline receipt never executes the external Arb certificate and is not a
  release, campaign, manuscript, certificate, or theorem input;
- `run_dobriban_bh_protocol.py` is a controlled single-chunk adapter/smoke
  interface. Its optional overrides are for a bounded local diagnostic; it
  does not replace the all-chunk stress-bundle receipt or external certificate;
- `z_generate_manuscript_variables.py` performs strict manuscript hydration;
  pass the same `--output-root` used by the result and quality stages when
  working in an isolated bundle.

Keep statistical logic in `src/active_inference_power/` so it can be imported
and tested independently. Support modules such as `artifacts.py`,
`checkpointing.py`, `release_provenance.py`, and `run_ledger.py` are imported
by the public producers; do not invoke them as alternate release paths.

See [release review](../docs/release_review.md),
[figure accessibility](../docs/figure_accessibility.md), and the
[documentation hub](../docs/README.md) for the public command contracts.
