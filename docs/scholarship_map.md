# Scholarship map

This project uses scholarship as an interface contract. A reference may define
an estimand, justify an algorithmic boundary, motivate a generative-model
interpretation, or establish a reproducibility practice; it is not evidence
that the seeded experiment has evaluated every setting discussed in that
reference. The map below makes those roles inspectable.

| Literature layer | Primary source | What it establishes | What is implemented | Evidence artifact | Boundary / non-claim |
|---|---|---|---|---|---|
| Design-based power | Neyman and Pearson (1933) | Power is an operating characteristic relative to a declared testing design and sampling distribution | `PowerScenario` hashes the agent model, evaluator-side process, and testing setting; an investigator evaluates a separate policy against that contract | `scenario`, `action_loop`, conditional power surface | No invariant power for an agent or field; changing the policy changes the estimand |
| FWER procedures | Holm (1979); Hochberg (1988) | Strong family-wise error criteria and stepwise procedures | Holm, Hochberg, and Bonferroni dispatch paths | `output/results.json`, certificates | Finite operating characteristics for the declared two-groups design; no new FWER theorem |
| FDR control | Benjamini and Hochberg (1995) | FDR target and linear step-up procedure | BH correction and empirical FDP summaries | `output/results.json`, `method_comparison.png` | Independence/declared dependence assumptions remain part of the claim |
| Global-null BH reference | Benjamini and Hochberg (1995) | Under the independent continuous global null, FDR is the event probability `P(R > 0)`; the expected rejection count is a separate distributional quantity | `bh_global_null_fdr` plus the finite ordered-threshold calculation in `expected_false_rejections_global_null` | `output/formalism_registry.json`, `output/results.json` | The count/FDR distinction does not establish arbitrary-dependence validity or a universal finite-sample ranking |
| Weighted FDR | Benjamini and Hochberg (1997) | Fixed positive weights can encode pre-specified hypothesis priorities | Fixed-weight weighted BH with weight validation and permutation tests | `src/active_inference_power/corrections.py`, paired contrasts | Weights are fixed before seeing p-values; no data-adaptive weighting claim |
| Dependence | Benjamini and Yekutieli (2001) | Arbitrary-dependence correction boundary and PRDS context | BY plus negative, independent, block, and factor stress regimes | `dependence_sweep`, `dependence_sweep.png` | Stress regimes do not prove arbitrary-dependence behavior |
| Adaptive FDR | Storey (2002); Benjamini--Krieger--Yekutieli (2006) | Storey/q-value and adaptive linear step-up motivations | Same-vector Storey plug-in diagnostic, conservative fixed-lambda null-count-bound diagnostic, adaptive-BH comparison, and preregistered independent split-stream confirmation | `method_comparison`, certificate outputs, `corrections.storey_qvalue_conservative`, `corrections.storey_qvalue_split_confirmation` | The larger campaign exposed a small same-vector Storey finite-simulation FDR overrun; the split-stream remedy is a conditional calibration diagnostic, not a universal finite-sample theorem |
| Proof-oriented FDR conditions | Blanchard and Roquain (2008) | Self-consistency and dependency-control conditions connect algorithms to assumptions | The release registry records correction rules, dependence regimes, and non-claims; no new proof is asserted | `formalism_registry`, certificate audit | Traceability makes proof obligations visible but does not establish them |
| Operating characteristics | Genovese and Wasserman (2002) | Fixed-point/asymptotic power reference for BH-type procedures | Analytic reference alongside finite simulations | methodology equations, power curve | Asymptotic reference is not a finite-sample guarantee |
| Active-inference process theory | Friston et al. (2016, 2017); Parr and Friston (2019) | Generative-model, learning, policy, and epistemic/pragmatic vocabulary | Narrow binary hidden-state agent and policy trace | `active_inference`, `agent_sampling`, policy figure | No claim to reproduce the full active-inference framework |
| Policy-conditional design and role separation | Neyman and Pearson (1933); Morris, White, and Crowther (2019) | Operating characteristics and their simulation estimators are defined relative to a declared design, data-generating mechanism, and replication plan | Fixed-horizon lanes compare declared streams, evidence, and corrections; action-loop rows add an embedded policy plus evaluator-only scoring, with scenario-hash plus policy identifying each evaluation row | `scenario`, `action_loop`, conditional power surface | Power is not an invariant property of an agent or task environment; neither lane averages a real-world niche distribution or supplies an agent-capability benchmark |
| Sequential experimental design | Chernoff (1959); Lindley (1956) | Later sampling actions can depend on observed information, and information can guide experiment choice | Information-gain and cost-aware policies with explicit stopping traces | `action_loop`, `action_policy_frontier.png`, action-loop examples | Finite policy comparisons do not establish asymptotic optimality, a universal value of information, or utility maximization |
| Adaptive sensing and control | Friston et al. (2016, 2017); Parr and Friston (2019) | Generative-model and policy vocabulary for action-conditioned inference | Action-conditioned kernels and visible/evaluation trace separation in a controlled synthetic process | `action_loop`, environment-control example | Environment shifts are diagnostic unless conditional evidence assumptions hold |
| Policy evaluation under adaptive data collection | Chernoff (1959); Lindley (1956); Morris, White, and Crowther (2019) | Sequential design and simulation reporting motivate declaring replication-level estimands and their uncertainty | Expected-information-gain, cost-normalized, posterior-sampling policies plus paired policy contrasts | `action_statistics`, power-cost frontier | A power-cost frontier is conditional decision-design evidence, not an optimality or utility theorem |
| Discrete active inference | Da Costa et al. (2020) | Mathematical context for discrete state-space inference | Real `pymdp` binary channel plus closed-form oracle | oracle-equivalence tests, `active_inference_policy.png` | One static factor, identity transition, and fixed observation channel only |
| Software realization | Heins et al. (2022) | Open implementation context for discrete active inference | Real `pymdp` integration without mocks | package tests and release log | Package integration is not a scientific validity guarantee |
| Test martingales | Shafer et al. (2011) | Sequential evidence and nonnegative martingale perspective | Bernoulli likelihood-ratio e-process interface | `sequential`, `sequential_evidence.png` | Optional-stopping validity is limited to the declared null and filtration |
| Safe testing | Grünwald et al. (2024) | Optional-continuation/e-value validity boundary | Used as adjacent theory and explicit contrast | scope section, claim ledger | Safe-testing theory is not silently substituted for the implemented e-process |
| Online FDR foundations | Javanmard and Montanari (2018) | Online FDR control requires an ordered filtration and explicit dependence conditions | LORD-family lineage is recorded alongside the implemented recursion | `online_fdr`, formalism registry | A procedural implementation does not validate an unexamined arrival process |
| Online FDR | Ramdas et al. (2017); Ramdas et al. (2018) | History-dependent LORD++ and candidate-indexed SAFFRON recursions | Auditable finite-horizon levels, candidates, rejections, procedural alpha-wealth $W_t^\alpha$, and e-LOND cumulative nominal-level allocation; the package applies its declared conservative shared level cap to SAFFRON traces | `online_fdr`, `online_fdr_trace.png` | Finite null calibration is not a new online-FDR proof, and the capped trace does not broaden the published procedure assumptions |
| Sequential error-budget procedures | Foster and Stine (2008); G'Sell et al. (2016) | Alpha-investing and sequential selection make the evolving error budget and selection history explicit | Procedural alpha-accounting and selection traces are serialized alongside online procedures | `online_fdr`, `conditional_selection_calibration` | The implementation does not claim validity for arbitrary dependence, adaptive allocation, or every sequential rule |
| Adaptive allocation and bandit testing | Yang et al. (2017) | Online testing can be coupled to allocation across arms or targets | Controlled target-selection actions with chronological evidence accounting | `action_loop`, `split_stream_confirmation`, adaptive-target example | Target-selection traces remain conditional or diagnostic until the conditional-null contract is verified |
| Adaptive target selection | Ramdas et al. (2017, 2018); Fithian et al. (2014) | Chronological evidence accounting after predictable target selection | Adaptive-target example with LORD++/SAFFRON diagnostics | `evaluate_adaptive_target_selection`, action-loop surface | Conditional super-uniformity and predictable selection remain required; no post-selection theorem is claimed |
| Time-uniform concentration | Howard et al. (2020, 2021) | Time-uniform nonnegative-supermartingale and confidence-sequence vocabulary for optional continuation | Bernoulli e-process, Ville-bound diagnostic, and bounded uncertainty helpers | `sequential`, `uncertainty`, `sequential_evidence.png` | The suite implements only the declared Bernoulli e-process and finite-simulation diagnostics |
| Split-stream confirmation | Fithian et al. (2014); Ramdas et al. (2017) | Independent confirmation separates selection information from confirmatory evidence | `SelectionPolicySpec`, visible filtration events, independent stream hashes, and target-level confirmation | `output/action_loop_suite.json`, conditional-selection figure | Formal label is conditional on the declared confirmation null law; same-stream selection remains diagnostic |
| Confidence sequences | Howard et al. (2020, 2021) | Time-uniform confidence via nonnegative supermartingales | Beta-mixture Bernoulli e-process inversion and confidence-sequence trace | `sequential`, `sequential_evidence.png` | The implementation is limited to the declared Bernoulli family |
| Online e-values | Ramdas et al.; Xu and Ramdas | Predictable e-value test levels and procedure-specific alpha accounting | e-LOND nominal level $\alpha_t$, reciprocal e-value rule $E_t\ge1/\alpha_t$, rejection, and cumulative nominal-level allocation trace | `online_fdr`, `online_fdr_trace.png` | Recycled levels can make the cumulative trace exceed $\alpha$; it is neither wealth nor a remaining budget, and conditional calibration and filtration assumptions remain required |
| Held-out policy evaluation | Project-specific held-out protocol; Morris, White, and Crowther (2019) for declared simulation reporting | Independent train/evaluation streams separate policy development from operating-characteristic estimation; the finite search space is part of the estimand provenance | Frozen deterministic candidate policy with parameter hash, search-configuration hash, candidate count, parameter dimension, and held-out evaluation | `output/action_loop_suite.json`, policy frontier | Held-out improvement is not optimality, utility, causal, or universal evidence; search-space accounting does not establish generalization |
| Research objects and provenance | Wilkinson et al. (2016) | Findable, accessible, interoperable, reusable artifact principles | Hash-linked configuration, results, figures, captions, and manuscript variables | `output/manifest.json`, registry, dashboard | Provenance strengthens auditability; it does not strengthen statistical assumptions |
| Monte-Carlo precision and simulation design | Project-configured Hoeffding/Wilson/DKW/paired-CRN reporting conventions; Morris, White, and Crowther (2019) for simulation-study reporting | Replication budgets and uncertainty are part of the estimand report, not decoration | Release anchor plus the expanded campaign profile and sample-size accounting | `output/results.json`, `output/campaign/results.json` | More replications narrow simulation uncertainty; the campaign certificate failure remains a method-review result, not a universal claim |
| Simulation-study reporting | Morris, White, and Crowther (2019) | Estimands, data-generating mechanisms, method settings, replication, and uncertainty should be declared before interpretation | Configuration, seed schedule, replication accounting, interval conventions, and generated captions | `output/results.json`, `output/figures/figure_registry.json`, manuscript variables | Reporting completeness improves auditability and does not substitute for a theorem or external validation |
| External Dobriban/BH stress protocol | Dobriban (2026 preprint) | An externally authored finite stress construction and outward-rounded certificate target | Offline, checksum-pinned protocol adapter and retained stress receipt | `dobriban_bh_stress_bundle`, protocol manifest | Preprint status is retained; local simulation neither reproduces the external certificate nor upgrades it to a theorem |
| Local runtime measurement | Georges, Buytaert, and Eeckhout (2007) | Repeated performance observations need a declared measurement protocol and are sensitive to execution context | Isolated `run_complexity_experiments.py` timing profile with configured warmups, raw wall/process samples, environment context, and an exact transitive source-closure witness | `<new-root>/complexity_experiments.json`, `manifest.json` | A host-specific diagnostic is not a release result, CPU/memory model, cross-hardware ranking, or statistical-validity claim |
| Reproducible computational research | Sandve et al. (2013) | Executable inputs, provenance, and reproducible environments support computational reuse | Atomic artifacts, hashes, clean-install checks, deterministic worker invariance, and render validation | `output/manifest.json`, run ledger, wheel smoke test | Provenance does not repair a misspecified model, process, filtration, or testing setting |
| Formal methods-to-code traceability | Project traceability protocol; Sandve et al. (2013); Wilkinson et al. (2016) for research-object practice | A formal statement is connected to symbols, tests, artifacts, evidence class, and a non-claim | `active_inference_power.formalism`, `output/formalism_registry.json`, `fig:formalism-code-map` | Traceability makes the boundary inspectable; it does not replace a proof or validate an untested process |

## How to read the map

The first three columns identify the literature layer, its primary source, and
the specific role that source plays. The fourth column names the concrete
object implemented in the canonical package. The fifth column identifies the
evidence artifact that makes that implementation inspectable. The final column
is deliberately restrictive: it records what a reader must not infer from a
figure, certificate, benchmark, or successful render.

Formal validity claims, where made, rest on the cited theory together with
satisfied declared contracts. Synthetic regimes instantiate and check those
contracts while supplying finite operating-characteristic evidence; they do
not create a theorem on their own. The Golub fixture and staged agent
trajectory are offline benchmark stress tracks and remain stress tests. A
Monte Carlo interval quantifies simulation precision; it is not a proof. A
posterior belief is a model-relative state; it is not automatically a p-value,
e-value, FDR, or FWER quantity. An e-process can support optional stopping
only under the null, filtration, and update rule that define it.

## Citation integrity policy

Every Pandoc citation key used by the manuscript must occur in
`manuscript/references.bib`. Every `kind: citation` entry in
`data/claim_ledger.yaml` must resolve to that same bibliography, and every
ledger source path must exist. New methods, regimes, benchmarks, and validity
boundaries require both a source-backed role and a claim-ledger entry. The
real-file tests in `tests/test_scholarship.py` enforce these contracts without
network access.

Literature sources support definitions, theoretical positioning, and explicit
non-claims; they do not substitute for the generated experiment, certificates,
rendering checks, or benchmark provenance.

The formalism-to-code connection is maintained as a machine-readable registry
rather than a prose-only analogy. See [`docs/formalism_to_code.md`](formalism_to_code.md)
and [`docs/claim_audit.md`](claim_audit.md) for the mapping protocol and object-level
claim review.

## Research sequencing

The scholarship is intentionally ordered by validity risk. First, the suite
must establish a filtration and conditional-null contract for adaptive target
selection. Only then should it add broader online e-value procedures or learned
policies. The full entry criteria, outputs, and non-claims are maintained in
[`docs/research_roadmap.md`](research_roadmap.md).
