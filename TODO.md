# Active Inference Power — forward workboard

This workboard distinguishes completed release hardening from future research
work. Completed items remain listed with their source-bound evidence so later
extensions can retain the same gates. There are no open, locally executable
workboard tickets; one minor metadata decision remains open below. The R and G
sections preserve completed evidence lanes and claim decisions. Any new
extension must first be entered as a bounded
task with its estimand, evidence class, and acceptance gate. The retained
evidence boundaries remain in
[`docs/research_roadmap.md`](docs/research_roadmap.md),
[`docs/claim_audit.md`](docs/claim_audit.md), the formalism registry, and the
hash-linked, source-bound release bundle recorded below.

## Current release boundary

- The fixed-horizon binary experiment remains the validated reference anchor.
- `output/` is an active development and producer namespace whose manifest can
  be incomplete while results, quality, hydration, or render producers run. It
  is not a test fixture and not the selector for a reader-facing release
  candidate: the real-file test suite creates its generated bundles in
  session-scoped temporary roots. A local
  release-complete statement must instead name one immutable candidate root and
  be supported by that root's `manifest.complete: true` and final review.
  Source or configuration changes require a new finalized candidate before
  another release-complete statement is made.
- Action-loop, dynamic-process, online-testing, learned-policy, and benchmark
  results remain conditional, diagnostic, or stress evidence according to their
  stored validity labels.
- The expanded campaign and precision audit remain fail-closed because the
  Storey finite-simulation certificate exceeded its nominal target. No seed,
  tolerance, caption, or claim should be changed to obtain a pass.
- `output/campaign/` and `output/precision/` are preserved fail-closed evidence
  bundles. Never reuse either path as a new campaign output root.
- `output/campaign-v2/` retains the finalized v2 campaign receipt for its
  frozen certificate-policy witness. Its manifest binds the source
  configuration, execution witness, artifact inventory, checkpoints, and
  predecessor evidence. It is historical campaign evidence, not a
  source-current release candidate; archive it before any source-contract
  refresh. Any prior complete v2 receipt is byte-preserved in its own
  timestamped archival sibling; its exact directory and digest are artifact
  metadata, not a workboard constant.
  Campaign finalization does not consume a quality snapshot, hydrated
  manuscript, PDF/HTML render, or release-review record. Completion is limited
  to that campaign contract; it is not a promoted result or release evidence.
- External data retrieval remains explicit and offline-safe; ordinary tests and
  release generation must not fetch benchmark data. The completed benchmark
  preflight receipt is not an acquisition, license adjudication, or external
  benchmark release.

## Open minor — version metadata drift

| ID | Open scope | Evidence and boundary |
|---|---|---|
| M6 | The frozen manuscript metadata still declares `pymdp_version: "1.0.3"` in `manuscript/config.yaml` and `manuscript/config.yaml.example`, while the package pin in `pyproject.toml` is `inferactively-pymdp==1.0.5`, advanced from `1.0.4` in #23 without touching the manuscript. The 1.0.3 value is the v1.0.0 release's recorded dependency version. Because both manuscript files declare the key explicitly, the `1.0.5` default at [`scripts/generate_results.py:2026`](scripts/generate_results.py#L2026) does not apply to this configuration, so a freshly generated `meta.pymdp_version` — and with it the `CONFIG_PYMDP_VERSION` manuscript variable — would state 1.0.3 while 1.0.5 is installed. The manuscript files are frozen release evidence and are not edited by this pass. Open decision for the release owner (Daniel Ari Friedman): at the next source-contract refresh, either re-record both manuscript files to the current pin, or keep 1.0.3 and state in the manuscript scope that the value is the v1.0.0 release's dependency record rather than the installed pin. Nothing outside this repository blocks it; it is deferred only because it must be settled before the release chain runs again, not after. | [`manuscript/config.yaml:127`](manuscript/config.yaml#L127) and [`manuscript/config.yaml.example:64`](manuscript/config.yaml.example#L64) (frozen), [`scripts/generate_results.py:2026`](scripts/generate_results.py#L2026) (the `pymdp_version` fallback default, now `1.0.5`), [`src/active_inference_power/manuscript_variables.py:857`](src/active_inference_power/manuscript_variables.py#L857) (the `CONFIG_PYMDP_VERSION` binding), and [`pyproject.toml`](pyproject.toml) (pin `inferactively-pymdp==1.0.5`). [`README.md:29`](README.md#L29) and [`domain_profile.yaml`](domain_profile.yaml) already state the 1.0.5 pin; only the frozen manuscript metadata remains at 1.0.3. |

## Completed minor — maintenance and reader-facing quality

These completed changes do not alter a scientific estimand or promote an
evidence class. Their focused tests and release checks remain required for all
future work.

| ID | Completed scope | Retained evidence |
|---|---|---|
| M1 | Complete — manuscript presentation linter for acronym expansion, citation-free abstract, duplicate cross-reference wording, unresolved tokens, and numeric-literal allowlists. | `scripts/audit_manuscript_presentation.py`, `tests/test_manuscript_presentation_linter.py`, and `docs/manuscript_presentation_linter.md`. |
| M2 | Complete — read-only release review composing schema, manifest, registry, dashboard, render, figure-opening, and clean-install checks. | `scripts/review_release.py`, `tests/test_release_review.py`, and `docs/release_review.md`. |
| M3 | Complete — project-root release-path cross-links for quickstart, FAQ, examples, offline benchmarks, and the failure-safe campaign workflow. | `scripts/audit_documentation_links.py`, `tests/test_documentation_links.py`, and the documentation hub. |
| M4 | Complete — browser-free demo smoke lane validating a committed payload, visible trace, configuration hash, and trace hashes. | `scripts/smoke_demo.py`, `tests/test_demo_smoke.py`, and `tests/fixtures/demo_payload.json`. |
| M5 | Complete — figure accessibility audit for dimensions, redundant encoding, estimand, uncertainty, sample size, evidence class, and validity boundary. | `scripts/audit_figure_accessibility.py`, `tests/test_figure_accessibility.py`, and `docs/figure_accessibility.md`. |
| M7 | Complete — adopted the estate's single GNN importer reader on custody pin `v5.1.0` and deleted the interim local pin reader (`active_inference_power.gnn_specs.load_upstream_pin`) plus its pin-shape tests, so reading a pin, validating it against its `.github/GNN.md` pointer, and validating specification syntax each have one home again. | `models/gnn-pin.json` (schema `gnn-importer-pin/v1`, tag `v5.1.0`, commit `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`), `.github/GNN.md`, `docs/RUN_GUIDE.md`, `scripts/gnn_importer_check.py`, `scripts/validate_gnn_specs.py`, and `tests/test_gnn_specs.py`; merged as #22 (`97f2388`). Custody plumbing only: it promotes no claim and adds no estimand. The `gnn` arm now runs `python -m gnn.importers check --repo .` against the pinned revision, and the two member checks (runtime binding and pinned-parser digest equality) remain. |

## Completed medium — bounded research extensions

These implementation/evidence lanes have real test coverage, formalism and
claim-ledger records, and hash-linked isolated receipts. Their finite summaries
remain non-promoted: none supplies a manuscript result, release certificate,
universal ranking, or theorem.

| ID | Completed scope | Retained evidence and boundary |
|---|---|---|
| R1 | Complete — testing-setting sensitivity surface over alpha, horizon, stopping budget, evidence interface, and correction, with common random numbers and separate power, FDR, FWER, cost, and stopping summaries. | `testing_setting_sensitivity.py`, `tests/test_testing_setting_sensitivity.py`, and the R1 lane in `output/research-atlas/research_atlas.json`. Finite setting-specific diagnostics only; no pooled utility or universal policy ranking. |
| R2 | Complete — stratified split-stream all-null calibration by target and visible selection history, with fixed tuning, independent child streams, and conditional diagnostics. | `selection_sensitivity.py`, `tests/test_selection_sensitivity.py`, and the R2 atlas lane. This is assumption-gated finite evidence under the declared independent streams, not a selective-inference theorem. |
| R3 | Complete — paired static and dynamic-process controls spanning persistent context, reliability drift, action transitions, and missingness schedules. | `dynamic_process_sensitivity.py`, `tests/test_dynamic_process_sensitivity.py`, and the R3 atlas lane. Dynamic summaries remain diagnostic/stress evidence unless a stronger conditional law is separately verified. |
| R4 | Complete — worker-invariant sequential comparison atlas for e-process families, confidence sequences, e-LOND, stopping distributions, and an explicit invalid-peeking contrast. | `sequential_atlas.py`, `tests/test_sequential_atlas.py`, and the R4 atlas lane. Finite Ville/coverage summaries remain separate from their theorem assumptions. |
| R5 | Complete — repeated train/freeze/evaluate policy partitions with frozen-policy hashes, candidate-space accounting, leakage checks, and paired held-out power-cost bands. | `repeated_policy_evaluation.py`, `tests/test_repeated_policy_evaluation.py`, and the R5 atlas lane. It reports conditional finite performance for the declared search space, never optimality or universal validity. |
| R6 | Complete — offline benchmark preflight, checksum-pinned fixture/provenance validation, and a JSON/HTML dashboard without downloading data. | `benchmark_preflight.py`, `scripts/run_benchmark_preflight.py`, their real-file tests, and `output/benchmark_preflight/`. This is an external-stress receipt; independent source/license/DOI/acquisition review remains required before any benchmark release. |

## Major — gated campaigns and claim decisions

These tasks consume substantial compute or could alter the publication claim
surface. They require a written preregistration/configuration change before
execution and a fail-closed output namespace.

| ID | Status and scope | Evidence, remaining gate, and claim boundary |
|---|---|---|
| G1 | Complete decision/evidence lane — frozen three-way review compares the same-vector Storey plug-in, fixed-lambda conservative same-vector Storey, and independent split confirmation; v2 predeclares the same-vector record as diagnostic and the split stream as terminal. | `docs/storey_certificate_decision.md`, `storey_method_review.py`, `scripts/run_storey_method_review.py`, their tests, and `output/storey-method-review/` have status `review_only_not_for_promotion_or_release`. The prior failed campaign/precision receipts remain fail-closed; no threshold, seed, or claim was retuned. |
| G2 | Complete campaign evidence lane — the isolated v2 campaign was finalized under its frozen statistical/policy configuration, deterministic source-work accounting contract, checkpoints, hashes, and terminal/diagnostic certificate partition. | The retained `output/campaign-v2/manifest.json` binds its configuration, execution witness, predecessor receipts, inventory, figures, and stage checkpoints. Its terminal certificate records passed under the declared partition; the same-vector `storey_power_gain` failure remains serialized as a visible diagnostic rather than being erased or relabeled as passing. The source-work contract is provenance accounting, not a timing, cost, or statistical result. This historical campaign completion requires no quality snapshot, hydration, PDF/HTML render, or `review_release.py`; it does not promote a release claim. A source-contract refresh must archive this receipt and generate a new one before any current-state statement. |
| G3 | Complete implementation/evidence lane — an isolated research atlas composes R1–R5 and G4 into finite model/process/setting/policy summaries, accessible figures, a figure registry, dashboard payload, and manifest. | `scripts/run_research_atlas.py`, `tests/test_research_atlas_script.py`, and `output/research-atlas/` have status `research_only_not_for_promotion_or_release`. It is deliberately not a manuscript-variable, publication, or release artifact. |
| G4 | Complete implementation/evidence lane — additive multi-target arrival environment with action-dependent transitions, time-varying reliability/cost, independent confirmation streams, visible-filtration specification, hidden-state non-leakage tests, and paired static control. | `multi_target_environment.py`, `tests/test_multi_target_environment.py`, and the G4 atlas lane. The contract is an inspectable simulation interface, not a formal filtration proof or a broader adaptive-confirmation claim. |
| G5 | Complete preflight evidence lane — offline source-manifest, fixture checksum, preprocessing/provenance, and external-stress dashboard receipt are generated without retrieval. | `scripts/run_benchmark_preflight.py`, `tests/test_run_benchmark_preflight.py`, and `output/benchmark_preflight/`. Independent source, license, DOI, acquisition, and any biological/benchmark review are still required; no external benchmark release is authorized. |
| G6 | Complete — base local reproducibility/release recipe generated, quality-gated, hydrated, rendered, visually inspected, finalized, and independently reviewed. | Historical bundle metrics remain in their own manifests and review receipts; they are not a substitute for a source-current candidate after later changes. The local release certificate does not authorize external benchmark acquisition, broader theoretical claims, an external publication action, or a placeholder DOI. |

## Ordering

1. Retain M1–M5 and M7 as required gates so future changes fail early at the
   presentation, custody-pin, and artifact boundaries.
2. Retain the completed G1 review and R1–R6/G3/G4/G5 receipts as isolated,
   non-promoted evidence; rerun them only through their declared generators
   when a source or configuration change requires it.
3. Retain the completed G2 receipt at `output/campaign-v2/` alongside any
   byte-preserved prior v2 receipt in its timestamped archival sibling. Its
   historical campaign finalization is distinct from release work: a
   quality snapshot may be kept as reproducibility evidence, but manuscript
   hydration, rendering, and `review_release.py` are release-only steps. The
   diagnostic same-vector Storey verdict remains visible and non-promoted.
4. Do not turn the G5 preflight into data acquisition or an external release
   without independent provenance, license, DOI, and source review.
5. For future source or configuration changes, repeat the release chain from a
   new clean output root: regenerate results, figures, variables, manuscript,
   PDF, HTML, and manifest serially, then inspect and finalize that one
   source-bound bundle.

## Permanent claim boundaries

- Conditional power remains a functional of the declared generative model,
  generative process, testing setting, and policy.
- Monte Carlo intervals describe simulation precision; they do not repair a
  misspecified process, filtration, dependence structure, or testing rule.
- Chronological selection, posterior decisions, learned policies, dynamic
  environments, and external benchmarks do not acquire family-level validity
  from visualization or replication count alone.
- The package does not claim arbitrary-dependence control, universal
  active-inference power, policy optimality, causal effects, or benchmark
  ground truth without a separate contract and evidence gate.
- Regenerable release-derived artifacts remain outputs: source, configuration,
  results, provenance, and validation gates are authoritative. Explicitly
  preserved campaign and review receipts retain their own evidence boundaries.
