# Visualization notes

The publication figures are part of the evidence chain, not decorative
summaries. They are generated from `output/results.json`; registry, dashboard,
and manuscript captions share that result-derived metadata and name the
estimand, uncertainty convention, and boundary of the interpretation. The
separate local timing diagnostic has its own profile and manifest, and is never
substituted for a publication figure or release result.

The schema-2.4 figure registry stores a nonempty `reader_question`,
result-bound `visible_comparison`, `takeaway`, and `evaluation_perspective` for
every figure; its structured `metadata.validity_boundary` is the fifth
reader-first component. It also stores `visual_encoding`, `estimand`,
`uncertainty`, `sample_size`, `scenario_hashes`, and a result-lineage digest.
The dashboard renders the same fields. `evaluation_perspective` makes visible
who fixes a design, who acts from visible information, and who scores a trace;
it prevents an action-loop plot from being read as an intrinsic agent score.
This separates machine-readable claim metadata from fluent manuscript captions
and lets reviewers audit the interpretation contract without reverse-
engineering plotted pixels or guessing which scenario produced a visual. Its
`panel_validity` list keeps a whole-figure label from concealing an intentionally
invalid panel when a multi-panel visual mixes evidence objects.

## Reader-first caption and terminology contract

Every caption follows the same semantic order: a reader question, the visible
comparison, a takeaway, and an explicit boundary. The registry retains these
as `reader_question`, `visible_comparison`, `takeaway`, and
`metadata.validity_boundary`; the role-aware registry additionally retains
`evaluation_perspective`, while the caption presents the scientific argument as
fluent prose. Alternative text instead describes panels, marks, labels, and annotations
visible in the raster. It must never be substituted for the caption in
rendered HTML.

Six quantities that can otherwise be conflated are deliberately named and
plotted separately:

| Symbol / term | Meaning in this project | Must not be read as |
|---|---|---|
| Likelihood-ratio e-process evidence `E_t` | A dimensionless, multiplicative evidence path evaluated under its declared null and filtration; its Ville boundary supports the stated time-uniform interpretation only when that contract holds | A posterior probability, online alpha accounting, utility, monetary value, or resource cost |
| Procedural alpha-wealth $W_t^\alpha$ | The LORD++/SAFFRON bookkeeping balance used to allocate or replenish online p-value test levels | Likelihood-ratio e-process evidence, e-LOND cumulative allocation, or a currency/resource account |
| e-LOND cumulative nominal-level allocation $\sum_{i\le t}\alpha_i$ | The cumulative nominal e-value test levels emitted by e-LOND; it pairs with the decision rule $E_t\ge 1/\alpha_t$ and can exceed $\alpha$ after recycled levels | Alpha-wealth, a remaining alpha budget, likelihood-ratio evidence, or resource cost |
| Simulated cost `C` | The configured action cost accumulated in declared simulation units for a scenario and policy | Alpha-wealth, e-value evidence, a real-world price, or a utility function |
| Source-work proxy $Q_{\mathrm{comp}}$ | A deterministic, dimensionless proxy for named source-level array passes, scans, and declared trace calls at a stated input shape; retained draws are reported separately as retention accounting | Simulated cost, elapsed time, CPU instructions, peak memory, evidence, or online alpha accounting |
| Local elapsed time $\tau_{\mathrm{wall}}$ | Repeated wall-clock observations from one declared host/runtime in the isolated timing diagnostic | A release metric, cross-hardware benchmark, statistical uncertainty interval, or evidence of validity |

The sequential plot labels the fixed-horizon p-value path as **INVALID for
optional stopping**. The online-FDR plot labels LORD++/SAFFRON alpha-wealth and
e-LOND cumulative nominal-level allocation as different accounting paths, not
`E_t`. The action-loop trace labels posterior beliefs, p-value evidence, target
e-processes, the all-action global-null likelihood-ratio-product diagnostic,
and `C` on separate axes.

Error bars are also literal: `± k MC SE` is the configured display band, not a
population confidence interval; Wilson intervals are used for the indicated
binary rates; Student-t mean intervals are used for the indicated continuous
or fractional summaries. In the power-cost frontier, horizontal bars are
finite-simulation intervals for mean `C` and vertical bars are intervals for
mean power.

| Figure | Estimand | Uncertainty encoding | Interpretation boundary |
|---|---|---|---|
| `method_comparison.png` | Mean empirical FDR, power, and FWER by procedure | Canonical method labels; direct pale-orange FWER-oriented and pale-blue FDR-oriented family labels separated by a dashed divider; numeric-labeled bars and the configured Monte-Carlo SE band | One seeded two-groups design; nominal lines are targets, not observed guarantees |
| `power_curve.png` | Mean BH power and empirical FDR over the effect grid | Aligned power and FDR small multiples with shaded configured MC-SE precision ribbons, capped marks, and a separate lower log-scale precision panel; common random numbers across effects | Finite-simulation operating characteristic; monotonicity is design-specific |
| `dependence_sweep.png` | Mean BH/BY FDR and power over declared negative-equicorrelation, independent, positive-factor, and block-correlated regimes | Compact regime-plus-ρ labels, BH circles/solid lines, BY squares/dashed lines, and line error bars with the configured MC-SE band | Does not test arbitrary dependence or two-sided non-PRDS behavior; block rows are finite Gaussian stress settings |
| `paired_method_contrasts.png` | Paired method-minus-BH FDR and power differences | Bonferroni-simultaneous intervals over method/metric bands | Common-random-number finite-simulation comparison, not a universal ranking |
| `sequential_evidence.png` | Likelihood-ratio e-process evidence `E_t`, an invalid fixed-horizon p-value stopping contrast, and a Bernoulli confidence sequence | Three aligned panels separate log-scaled `E_t`, the explicitly invalid optional-stopping p-value path, and e-process-inverted rate endpoints; direct Ville and null-rate references | The e-process and confidence sequence are time-uniform only under their declared Bernoulli assumptions; the p-value panel is a fixed-horizon contrast |
| `online_fdr_trace.png` | LORD++/SAFFRON p-value test levels, candidates, rejections, and procedural alpha-wealth; e-LOND nominal e-value test levels $\alpha_t$, reciprocal e-value condition $E_t\ge1/\alpha_t$, rejections, and cumulative nominal-level allocation | Three full-width stacked panels: a log-scaled nominal-level panel with procedure-specific line/marker encodings and a direct e-LOND decision-rule label; a procedure-row candidate/rejection raster whose title defines bar/star marks; and a directly labeled accounting panel that distinguishes wealth from cumulative allocation | Accounting is not likelihood-ratio e-process evidence; e-LOND cumulative allocation can exceed $\alpha$ after recycling; finite null calibration does not establish online FDR control |
| `conditional_selection_calibration.png` | Maximum empirical-CDF excess for chronological selected-target p-values and a deliberately invalid future-evidence contrast | Direct DKW comparison labels; a gray hatched no-selection cell when a target has no selected p-values rather than a zero bar | Selected-target calibration is conditional on the declared chronological null; post-hoc future-evidence selection is intentionally diagnostic |
| `benchmark_stress.png` | Offline staged-agent benchmark versus synthetic e-value context | Numeric-labeled, provenance-annotated bars with external availability status | Stress test only; no external benchmark upgrades synthetic claims |
| `active_inference_policy.png` | Mean FDR and power for posterior-threshold decisions versus BH on calibrated evidence | Grouped bars with the configured MC-SE band by evidence regime | Reference horizon and fixed observation channel; no optional-stopping or sequential-FDR claim |
| `certificate_summary.png` | Certificate estimate relative to its declared comparison bound | Horizontal estimate intervals, bound diamonds, explicit ≤/≥ direction, and separate control/adaptive-calibration panels | Pass means finite-simulation consistency with the declared bound, not proof; the split-stream Storey lane remains assumption-gated |
| `pvalue_calibration.png` | ECDF of independent global-null p-values | Uniform reference and configured-confidence DKW band; annotation names the pooled p-value count and independent calibration-replication count | Marginal calibration diagnostic for the declared independent null only |
| `sample_size_precision.png` | MC precision and replication budget | Log-scaled numeric half-widths, target line, labeled replication bars, and conservative Hoeffding floor | Planning/accounting aid, not a guarantee that every estimand has equal variance |
| `computational_complexity.png` | Version-specific source-work curves over declared hypothesis/stream grids and action-loop trace-call accounting | Log-scaled legend-labeled work-proxy curves; one combined line explicitly names the coincident binary-e-process and e-LOND linear proxies; a separate labelled action-path ledger; line style, marker, and text distinguish single-pass, rank-dominant, and history-scan paths | Provenance/feasibility contract only; no elapsed-time, processor, memory, statistical-validity, or cross-hardware claim |
| `scholarship_boundary.png` | Literature role, implemented object, evidence class, and validity boundary | Six-row, four-column source-to-claim map covering statistical, active-inference, sequential, online, selective-inference, and time-uniform layers | Provenance and claim-discipline aid; it is not a quantitative result |
| `action_loop_trace.png` | Visible action/observation/posterior/target-e-process evidence `E_t`/simulated-cost `C`/stopping trace | Aligned panels name posterior probability, p-value evidence, target-specific e-processes (the per-target decision objects), the all-action global-null likelihood-ratio product marked as diagnostic-only, e-value factors, and `C` with the scenario's declared cost unit and scale identifier; target traces use distinct marks/line styles and an X plus outline marks stopping; hidden-state exclusion note | Pathwise diagnostic from an investigator-declared scenario/policy design; hidden states are evaluation-only, the all-action product is not the per-target decision object, no e-process is alpha-wealth, and cost is not cross-scenario comparable without matching declared scale identifiers |
| `conditional_power_surface.png` | Scenario/policy replication means of true-positive rate (reported power), observed mean FDP (the finite-simulation FDR estimand), total simulated cost `C`, and stopping | Policy colors plus markers, direct legends, finite-simulation intervals on all four estimands, and unique ID/hash-prefix/reliability labels for declared scenario cells | Operating characteristics are conditional on the investigator-declared model/process/setting/policy design, not an agent or environment intrinsic; observed mean FDP is not an FDR-control guarantee; cost comparisons retain the declared scenario cost units |
| `action_policy_frontier.png` | Policy power-cost operating points with total simulated cost `C` in hash-bound scenario cost units and scales | Horizontal finite-simulation intervals for mean `C`, vertical intervals for mean power, policy color-plus-marker legend, faceted immutable scenario cells identified by a compact ID when needed plus hash prefix/reliability, direct unit/scale labels, and an explicit conditional-scope note | Conditional decision-design comparison within one declared cost scale; not a universal policy ranking, cross-scale cost comparison, or utility theorem |
| `scenario_decomposition.png` | Investigator-declared scenario/policy design and replication-mean TPR estimand | Top role band, three-input scenario hash, separate policy node, visible-history trace, evaluator-only scoring node, and MCSE output | Interface and claim-boundary diagram; not a quantitative result or a distribution over real-world niches |
| `testing_setting_contract.png` | Declared hypothesis family, evidence interface, accounting rule, default stopping rule/cost contract, estimands, and validity boundary | Six-box contract map with dynamic alpha, horizon, methods, and policy labels from the result configuration | Preregistration and interface visual; the boxes describe assumptions and outputs rather than measured operating characteristics |
| `formalism_code_map.png` | First declared registry panel linking formal objects to bounded executable evidence | Print-sized three-column overview with explicit registry hash, evidence class, one representative code topic, one representative test topic, and a short non-claim; the hashed registry retains complete symbols, paths, assumptions, and non-claims | Provenance and formal-to-code audit aid; it is not a quantitative result or proof |
| `formalism_code_map_continued.png` | Second declared registry panel and its executable traceability overview | Continuation overview with the same registry hash, representative-topic fields, and reserved footer band; complete paths remain in the hashed registry | Completes part of the provenance map; it is not a quantitative result or proof |
| `formalism_code_map_final.png` | Final declared registry panel and its executable traceability overview | Continuation overview with the same registry hash, representative-topic fields, and reserved footer band; complete paths remain in the hashed registry | Completes the provenance map; it is not a quantitative result or proof |
| `adaptive_fdr_audit.png` | Separate BH/Storey FDR estimates and Storey-minus-BH paired power gain | FDR estimate intervals, declared bounds, finite gate ceilings, explicit pass/fail labels, and a separate power-gain panel | Finite-simulation audit of the declared design; a gate outcome is not a theorem and does not broaden the adaptive validity claim |

## Design rules

1. Prefer an explicit estimand over an unlabeled score. In action-loop results,
   “power” means the declared scenario/policy-indexed expected true-positive
   rate, estimated by a replication average; the separately reported
   `any_true_rejection_rate` is the distinct probability of at least one true
   rejection. “FDR” means expected FDP, likewise estimated by a replication
   average rather than a ratio of aggregate counts.
2. Show uncertainty whenever replication-level outcomes are available. The
   configured MC-SE multiplier is a readability and release-policy choice, not
   a confidence interval or a multiplicity correction.
3. Put theoretical targets and empirical estimates in different visual forms:
   horizontal reference lines identify nominal levels, while markers and bars
   identify realized simulation quantities.
4. Encode inequality direction. Upper-bound checks and lower-bound comparisons
   cannot share an unlabeled “bound” marker without inviting a reversal.
5. Keep scope in the caption. A visually smooth curve cannot establish
   arbitrary-dependence, optional-stopping, or universal active-inference
   claims.
6. Make visual semantics redundant. Use color, marker shape, line style, direct
   numeric labels, and text annotations together so key comparisons survive
   grayscale printing and accessibility tooling.
7. Treat figure refreshes as release artifacts. After changing plotting code,
   run `./.venv/bin/python scripts/regenerate_figures.py`, then rerun manuscript
   hydration and render validation; do not hand-edit files under `output/`.
8. Use a dedicated scholarly boundary visual when a project combines theory,
   implementation, and finite evidence. Keep source names, evidence classes,
   and non-claims in separate columns so a literature citation cannot be read
   as a result estimate.
9. Keep action-loop figures honest about observability: visible history is
   plotted as evidence, while hidden process state is never shown as an input
   to the policy.
10. When a formalism is central to the contribution, show its executable
    realization and non-claim boundary directly; a formula without a source
    symbol and test path is not a complete implementation claim.
11. Treat a scenario configuration hash as its content address and its required
    unique scenario ID as a human-readable label. Captions and registries must
    retain both, and figures must use scenario metadata from the
    operating-characteristic row rather than an ID-only lookup.
12. Use distinct policy colors and marker shapes in action-loop surfaces. A
    repeated hue must never be the only way to distinguish policies, especially
    where power, FDR, cost, and stopping are compared in small multiples.
13. Give contract diagrams the same provenance treatment as quantitative plots:
    derive labels from the configuration, register the evidence class, and state
    that a setting visualization records a design rather than estimating an
    operating characteristic.
14. Keep captions and alternative text distinct. The caption explains the
    estimand and boundary; alternative text describes visible marks, panels,
    annotations, and comparisons. Verify the renderer preserves the authored
    alternative text rather than silently duplicating the caption.
15. Name terms of art at the axis or legend where a reader needs the
    distinction. In particular, label likelihood-ratio e-process evidence
    `E_t`, LORD++/SAFFRON procedural alpha-wealth, e-LOND cumulative
    nominal-level allocation, and simulated cost `C` separately. State the
    e-LOND rule $E_t\ge1/\alpha_t$ where its test level appears; do not call its
    cumulative trace a remaining budget. When computational feasibility is shown,
    label source-work $Q_{\mathrm{comp}}$ and host-specific elapsed time
    $\tau_{\mathrm{wall}}$ separately as well; none is a synonym for another.
16. State interval orientation and units in the figure itself. A horizontal
    cost interval and a vertical power interval answer different questions,
    and a simulated action cost must identify its declared model units rather
    than imply a real-world price or utility.
17. Never plot a host-specific timing value as if it were a release operating
    characteristic. The local timing diagnostic must name its host scope,
    repetitions, summary statistic, and raw-sample availability; the release
    figure must instead label its deterministic source-work proxy and input
    shape.
