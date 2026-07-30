# Generated artifact inventory

Project-source paths are relative to this project. For current generated
release or candidate artifacts, `<bundle-root>` means one concrete, previously
absent project-relative or absolute output root selected before the first
producer (for example, `output/release-candidate-<timestamp>`). Pass that same
root to every producer, audit, renderer binding, review, and finalizer. Literal `output/campaign/`,
`output/precision/`, and `output/campaign-v2/` rows below are retained
historical namespaces, not aliases for `<bundle-root>`.

| Artifact | Generator | Consumer |
|---|---|---|
| `<bundle-root>/results.json` | `scripts/generate_results.py` | manuscript, figures, dashboard; carries source and execution-tree witnesses plus deterministic computational-work accounting |
| `<bundle-root>/certificates.json` | `scripts/generate_results.py` | release gate and manuscript; must exactly match `results.certificates` |
| `<bundle-root>/action_loop_suite.json` | `scripts/generate_results.py` | conditional action-loop evidence, demo consumers, all-null selected-target calibration/DKW accounting; must exactly match `results.suite` |
| `<bundle-root>/formalism_registry.json` | `scripts/generate_results.py` | executable formalism-to-code traceability and claim audit; must exactly match `results.formalism` |
| `data/claim_ledger.yaml` | source contract validated by result and hydration scripts | machine-checked claim/evidence/non-claim inventory; hashed in the manifest |
| `<bundle-root>/demo/payload.json` | `scripts/generate_results.py` | deterministic Streamlit demo payload with configuration/trace/visible-trace/payload SHA-256 witnesses |
| `<bundle-root>/quality.json` | `scripts/generate_quality_report.py` | reproducibility manuscript and strict `quality_pass` release gate, bound to the current test harness by `quality_input_sha256` |
| `<bundle-root>/manifest.json` | `scripts/generate_results.py` | artifact audit plus source/execution witnesses |
| `manuscript/assets/active_inference_power_cover.png` | curated publication source asset | rendered PDF title page |
| `<bundle-root>/figures/method_comparison.png` | `active_inference_power.figures` | results manuscript |
| `<bundle-root>/figures/power_curve.png` | `active_inference_power.figures` | results manuscript |
| `<bundle-root>/figures/dependence_sweep.png` | `active_inference_power.figures` | results manuscript |
| `<bundle-root>/figures/active_inference_policy.png` | `active_inference_power.figures` | results manuscript |
| `<bundle-root>/figures/certificate_summary.png` | `active_inference_power.figures` | results manuscript |
| `<bundle-root>/figures/adaptive_fdr_audit.png` | `active_inference_power.figures` | metric-separated adaptive-FDR audit |
| `<bundle-root>/figures/pvalue_calibration.png` | `active_inference_power.figures` | null calibration manuscript |
| `<bundle-root>/figures/sample_size_precision.png` | `active_inference_power.figures` | replication accounting manuscript |
| `<bundle-root>/figures/computational_complexity.png` | `active_inference_power.figures` | source-work scaling curves and declared action-trace accounting; not a timing benchmark |
| `<bundle-root>/figures/paired_method_contrasts.png` | `active_inference_power.figures` | paired common-random-number contrasts |
| `<bundle-root>/figures/sequential_evidence.png` | `active_inference_power.figures` | e-process and optional-stopping contrast |
| `<bundle-root>/figures/online_fdr_trace.png` | `active_inference_power.figures` | LORD++ / SAFFRON alpha-wealth and e-LOND cumulative-allocation trace |
| `<bundle-root>/figures/conditional_selection_calibration.png` | `active_inference_power.figures` | all-null selected-target calibration diagnostic |
| `<bundle-root>/figures/benchmark_stress.png` | `active_inference_power.figures` | offline benchmark stress track |
| `<bundle-root>/figures/scholarship_boundary.png` | `active_inference_power.figures` | literature-to-implementation evidence and validity-boundary map |
| `<bundle-root>/figures/action_loop_trace.png` | `active_inference_power.figures` | visible action/observation/belief/target-e-process/cost/stopping trace |
| `<bundle-root>/figures/conditional_power_surface.png` | `active_inference_power.figures` | conditional scenario/policy operating-characteristic surface |
| `<bundle-root>/figures/action_policy_frontier.png` | `active_inference_power.figures` | conditional action-policy power-cost frontier |
| `<bundle-root>/figures/scenario_decomposition.png` | `active_inference_power.figures` | investigator/agent/evaluator role and conditional-design decomposition |
| `<bundle-root>/figures/testing_setting_contract.png` | `active_inference_power.figures` | declared testing-setting hypotheses, evidence, accounting, default stopping rule, cost contract, estimands, and validity boundary |
| `<bundle-root>/figures/formalism_code_map.png` | `active_inference_power.figures` | formalism-to-code evidence and claim-boundary map |
| `<bundle-root>/figures/formalism_code_map_continued.png` | `active_inference_power.figures` | continuation of the formalism-to-code evidence and claim-boundary map |
| `<bundle-root>/figures/formalism_code_map_final.png` | `active_inference_power.figures` | final continuation of the formalism-to-code evidence and claim-boundary map |
| `<bundle-root>/figures/figure_registry.json` | `active_inference_power.figures.write_figure_registry` | template figure-reference validator plus fluent captions, authored alternative text, explicit reader question/comparison/takeaway/evaluation-perspective/boundary fields, estimand, uncertainty, evidence class, redundant encoding, and result-lineage audit metadata |
| `<bundle-root>/review/dashboard.html` | `active_inference_power.figures` | interactive review; preserved across controlled-renderer rendering |
| `<bundle-root>/run_log.json` | `scripts/generate_results.py` | profile, terminal stage, execution witness, and sample-size accounting |
| `<bundle-root>/checkpoints/*.json` | `scripts/generate_results.py` | schema-validated terminal stages with child-seed, profile/stage, source, and execution witnesses |
| `<bundle-root>/data/manuscript_variables.json` | `scripts/z_generate_manuscript_variables.py` | resolved tokens and self-normalizing renderer-input witness |
| `<bundle-root>/manuscript/*.md` | `scripts/z_generate_manuscript_variables.py` | controlled-renderer inputs bound to source and renderer-input witnesses |
| `<bundle-root>/web/index.html`, `<bundle-root>/web/manuscript__*.html` | controlled release renderer through the documented reversible canonical-path binding; then `scripts/sync_rendered_figure_alts.py --output-root <bundle-root> --rendered-html <bundle-root>/web/index.html --include-auxiliary-pages` | reader-facing combined and retained per-section HTML that must contain hydrated witnesses; figure alternative text is synchronized from the authored registry before rendered-alt review |
| `<bundle-root>/web/_combined_manuscript.md`, `<bundle-root>/pdf/active_inference_power_combined.pdf` | controlled release renderer | reader-facing artifacts that must contain the hydrated witnesses |
| `<bundle-root>/review/renderer_provenance.json` | `scripts/capture_renderer_provenance.py --output-root <bundle-root>` after rendered-alt synchronization **and rendered-alt audit**, before unskipped review | post-render receipt binding final PDF/HTML/combined-Markdown hashes to controlled-renderer revision/dirty state, renderer/Stage-03/link-orchestration source closure, and available Pandoc/LaTeX identities; required in a complete release manifest, not a portability claim |
| `<bundle-root>/review/release_review.json` | `scripts/finalize_release.py` after a full `scripts/review_release.py` pass | terminal review receipt with run/source/render witnesses, passing manuscript-presentation and rendered-alt-text checks, and reviewed-render hashes; required in a complete release manifest |
| `output/campaign/`, `output/precision/` | preserved prior campaign/audit runs | fail-closed method-review evidence; do not overwrite, reclassify, or use as release evidence |
| `output/campaign-v2/` | guarded source-contract refresh only: after the [archive/preflight sequence](storey_certificate_decision.md#frozen-v2-lane), `scripts/generate_results.py --profile campaign --config data/research_lanes/conditional_power_campaign_v2.yaml --output-root output/campaign-v2` | isolated expanded campaign with a serialized exact certificate policy and two hash-bound predecessor receipts; the frozen config requires this exact root, while same-vector Storey remains diagnostic and the independent split-stream check remains terminal |
| `<fresh-storey-review-root>/storey_method_review.json`, `<fresh-storey-review-root>/manifest.json` | `scripts/run_storey_method_review.py --output-root <fresh-storey-review-root>` | frozen three-way G1 comparison of same-vector Storey plug-in, conservative same-vector, and independent split-confirmation Storey procedures; hash-bound review only, never a campaign or release completion signal |
| `<fresh-research-atlas-root>/research_atlas.json` | `scripts/run_research_atlas.py --output-root <fresh-research-atlas-root>` | bounded R1–R5/G4 research-atlas plan, finite summaries, source hashes, and explicit non-claims |
| `<fresh-research-atlas-root>/figures/*.png`, `<fresh-research-atlas-root>/figures/figure_registry.json`, `<fresh-research-atlas-root>/dashboard.json`, `<fresh-research-atlas-root>/manifest.json` | `scripts/run_research_atlas.py --output-root <fresh-research-atlas-root>` | isolated, accessible research-atlas figures and hash-linked diagnostic receipt; not a manuscript or release artifact |
| `<fresh-complexity-root>/complexity_experiments.json`, `<fresh-complexity-root>/complexity_experiments.png`, `<fresh-complexity-root>/index.html`, `<fresh-complexity-root>/manifest.json` | `scripts/run_complexity_experiments.py --output-root <fresh-complexity-root>` | isolated local timing profile with raw repeated wall/process samples, plan, exact transitive source-closure witness, and non-claim; not a manuscript, campaign, or release artifact |
| `<fresh-benchmark-preflight-root>/benchmark_preflight/provenance_dashboard.json`, `<fresh-benchmark-preflight-root>/benchmark_preflight/provenance_dashboard.html`, `<fresh-benchmark-preflight-root>/benchmark_preflight/manifest.json` | `scripts/run_benchmark_preflight.py --output-root <fresh-benchmark-preflight-root>` | offline-only benchmark provenance, staged-fixture checksums, and external-stress non-claim; no download or external-data release |

The final PDF/HTML is produced in the controlled release environment and is not
treated as standalone source evidence until it has been rendered and inspected.
Each `<bundle-root>` and `<fresh-...-root>` denotes one previously absent,
concrete destination; it is never an alias for `output/` or a retained receipt.
The frozen v2 lane is the deliberate exception because its declared namespace
is exactly `output/campaign-v2/` and must use its guarded archive/preflight
sequence.
For an isolated release bundle, pass `--output-root` consistently to result
generation, quality, manuscript hydration, both reader-facing audits, both
reviews, and release finalization. The quality producer may write its report to
that root, but its full test suite remains anchored at the project source tree
and uses real session-scoped temporary artifacts rather than the canonical
`output/` tree as a fixture. Generated candidate paths remain relative to their selected output root in the manifest,
while project source paths retain the project namespace. The preserved campaign
and precision predecessor receipts retain their separate namespace and hash
requirements for campaign finalization. This release-only chain does not apply
to campaigns: `finalize_campaign.py` validates campaign artifacts directly and
does not require quality, hydration, rendering, or release review.

Never reuse a literal campaign root as `<bundle-root>`. In particular,
`output/campaign/`, `output/precision/`, and `output/campaign-v2/` may not be
passed to release result generation, hydration, rendering, review, or release
finalization; their retention rules are not a shortcut to a current candidate.

Campaign namespaces are intentionally separate from the release anchor. Their
results, figures, captions, and dashboards may be refreshed only after the
current result schema validates; an incomplete campaign manifest is a retained
failure record, not a publication artifact. The declared
`certificate_gate_policy` is serialized with every certificate verdict and
must exactly partition the current certificate names. A finalizer requires a
passed campaign run and every *terminal* certificate to pass. Diagnostic
failures stay present in the bundle and cannot be silently omitted. The
canonical release and the preserved prior campaign/precision runs retain an
all-terminal policy; only the frozen v2 campaign declares
`storey_power_gain` diagnostic and `storey_split_confirmation` terminal. See
[the Storey decision record](storey_certificate_decision.md). No campaign
policy transfers evidence to the release namespace.

`run_research_atlas.py` rejects the project root, canonical `output/`, and any
pre-existing output directory, so its command must name a new isolated root.
The atlas manifest is explicitly barred from release finalization and
publication promotion. `run_benchmark_preflight.py` also requires an explicit
root, but writes only its `benchmark_preflight/` lane below that root. It
never downloads a source archive; an absent archive stays recorded as absent
rather than being represented as staged evidence. Its manifest paths are
relative to the lane itself and it separately records a bounded runner/domain
source witness, so fixture hashes are not presented as code provenance.

For a release bundle, first run
`scripts/audit_manuscript_presentation.py` with
`--variables-file <bundle-root>/data/manuscript_variables.json` and
`--resolved-manuscript-dir <bundle-root>/manuscript`, then the pre-render
`scripts/audit_figure_accessibility.py`, then optionally the read-only
`scripts/review_release.py --skip-render`. The unskipped reader-facing order is
strict: (1) render the already audited, fresh `<bundle-root>` through the
controlled renderer's reversible canonical-path binding; (2) run
`scripts/sync_rendered_figure_alts.py` with that root and its rendered HTML;
(3) run `scripts/audit_figure_accessibility.py` with the rendered-HTML option;
(4) capture `scripts/capture_renderer_provenance.py`; (5) run unskipped
`scripts/review_release.py`; and only then (6) run
`scripts/finalize_release.py`. The unskipped review requires the completed
reader-facing audits and renderer receipt. See [figure accessibility](figure_accessibility.md) and
[release review](release_review.md). `finalize_campaign.py` runs the figure
accessibility audit as part of its own campaign contract and does not consume a
release-review or renderer artifact.
