# Claim audit and formal boundary review

This audit is the release discipline for the Active Inference Power Suite. It
asks four questions of every statement: what is the formal object, what code
realizes it, what evidence was generated, and what stronger statement remains
out of scope?

## Audit result

The current suite supports six kinds of reader-facing statement:

| Evidence class | What the reader may conclude | What the reader may not conclude |
|---|---|---|
| Formal conditional | The implementation follows the declared contract under its assumptions and passes its oracle or invariant checks | Validity outside the declared null, filtration, process, or dependence assumptions |
| Analytic reference | The stated formula provides a model-based reference calculation | Finite-sample agreement in every design |
| Conditional operating characteristic | An investigator's finite-simulation estimate for the declared model, process, setting, policy, and replication plan | An intrinsic power of an agent, method, task environment, or active-inference system |
| Calibration diagnostic | The observed finite-run behavior is consistent with the declared calibration target within its uncertainty accounting | A proof of a new theorem or a guarantee under an untested regime |
| Trace diagnostic | One retained path makes a declared action, observation, evidence, cost, and stopping contract inspectable | A replication-mean operating characteristic, population estimate, or validity guarantee |
| Stress test | The implementation was exercised under a deliberately broader or external condition | Synthetic formal validity, universal robustness, or benchmark ground truth |
| Provenance contract | The artifact lineage, hashes, and source/test links are inspectable | Statistical assumptions become stronger because provenance is better |

## Presentation integrity

The abstract is citation-free so it can stand alone as a self-contained
summary. Acronyms are expanded at first use, and the introduction's notation
table is the authoritative vocabulary for symbols, error criteria, sequential
procedures, data formats, and software interfaces. Figure captions and
alternative text are generated from the validated result/configuration pair;
contract diagrams state when they describe a design rather than estimate a
quantity. These checks improve reader orientation and accessibility while
preserving the distinction between scholarship, executable contracts, and
measured evidence.

## Object-level review

The machine-readable registry in `output/formalism_registry.json` is the
authoritative row-level audit. The following review summarizes its interpretation
for readers.

1. **Design-based conditional power.** The operating-characteristic tradition
   defines power relative to a declared test and sampling design
   [@neyman1933efficient]. Here `PowerScenario` hashes the generative model,
   generative process, and testing setting; a separately recorded policy makes
   the evaluated design. Action-loop power is the replication-mean TPR, not the
   different any-true-rejection event. An investigator estimates that design's
   operating characteristic; it is not a universal property of an agent or
   synthetic task environment.
2. **Model/process separation.** Agent-side beliefs and evaluator-side truth
   are represented by separate objects and traces. The separation is a code
   contract and a test target, not a claim that every environment has a valid
   inferential likelihood.
3. **Dynamic process schedules.** `GenerativeProcessSpec` requires every
   time-varying schedule to declare its out-of-support behavior (`hold_last`,
   `cycle`, or `strict`). This makes simulation semantics reproducible; it does
   not promote nonstationary or action-dependent processes to formal validity.
4. **Fixed-horizon corrections.** BH, BY, weighted BH, and related procedures
   are implemented with input and weight invariants. Their interpretation still
   depends on valid p-values and the named dependence/weight assumptions.
5. **Global-null BH quantities.** Under the independent continuous global null,
   `bh_global_null_fdr` represents the event probability `P(R > 0)`, while
   `expected_false_rejections_global_null` computes the separate expected count
   `E[R]` from the finite ordered-threshold distribution. The identity does not
   extend the package's claims to arbitrary dependence.
6. **Sequential evidence.** The Bernoulli likelihood-ratio e-process has a
   Ville-bound interpretation under its declared null and filtration. A
   fixed-horizon p-value repeatedly inspected until rejection is retained only
   as an explicitly invalid contrast.
7. **Online FDR.** LORD++ and SAFFRON recursions are implemented and checked
   against known vectors. Null simulations quantify finite behavior; they do not
   create a new online-FDR proof, especially after unverified adaptive target
   selection.
8. **Action policies.** Information gain, cost-aware sensing, posterior
   sampling, target selection, and stopping are evaluated through paired
   replication-level metrics. A power-cost frontier is a conditional design
   comparison, not an optimal policy theorem.
9. **Uncertainty.** Intervals describe Monte-Carlo precision or paired
   differences under the declared replication schedule. They do not correct a
   misspecified scenario or convert a stress result into a guarantee.
10. **Benchmarks.** Offline fixtures test data loading and robustness workflows.
   They are external stress tests, not a source of synthetic validity claims.
11. **Two-sided non-PRDS protocol.** The imported Dobriban/BH plan is an
    offline, checksum-pinned finite stress protocol. Its model and seed
    construction are executable here, but the external outward-rounded Arb
    certificate remains a separate theorem artifact; local Monte Carlo does
    not reproduce or strengthen it.
12. **Split-stream Storey review.** The selection-only null-count estimate and
    confirmation-only rejection rule make the data split auditable. The
    generated certificate is conditional calibration evidence under independent
    streams and fixed tuning, not a universal FDR proof.
13. **Provenance.** Registry, results, figures, captions, variables, run logs,
    and manifests are hash-linked. This makes scientific differences explainable
   without implying that reproducibility metadata proves the statistical claim.
    Surface cells additionally use required unique scenario identifiers;
    these are indexing aids, while the complete scenario hash remains the
    identity used for artifact matching.

The learned-policy provenance is deliberately two-level: the frozen parameter
hash identifies the deployed rule, while a separate search-configuration hash
identifies the candidate space, training replication budget, and cost penalty
that produced it. A matching parameter vector under a different search space
is therefore a distinct research object.

## Review procedure for future methods

Before a new method is added to the publication surface, the maintainer must
complete this sequence:

1. state the estimand and its conditioning variables;
2. write the formal object and assumptions in the registry;
3. point to the public symbol and source module;
4. add a real-value test, ideally an analytic or known-vector oracle;
5. predeclare replication, interval, and paired-comparison accounting;
6. generate a result artifact and a caption that names the evidence class;
7. add a claim-ledger row and a roadmap entry criterion;
8. write the non-claim before reviewing the measured result.

The audit is intentionally conservative. A result can be scientifically useful
without being a formal guarantee; the evidence class tells the reader which
interpretation is justified.

## Machine-checked claim ledger

`data/claim_ledger.yaml` is validated by `validate_claim_ledger()` during result
generation and strict manuscript hydration. Every claim-bearing row must name
an evidence class and an explicit non-claim; every source file must exist; and
strict hydration requires every declared artifact to exist. The release result
records the ledger SHA-256 digest and the manifest hashes the ledger itself.
This makes a stale or partially reviewed claim inventory a release failure
rather than a documentation discrepancy.

The action-loop e-process has an additional executable boundary. Its label is
formal only when truth-preserving transitions, matching model/process
emissions, predictable selection, and state-invariant Bernoulli rates within
each target/truth cell are all satisfied. The richer context-dependent
reference environment is intentionally diagnostic even though it has static
truth and chronological actions.

## Expanded campaign gate

The first expanded campaign used the declared 15,000 main, 1,500 action-loop,
7,500 certificate, and 7,500 calibration replication budgets. All simulation
stages completed, but the terminal certificate gate failed for the
Storey-versus-BH finite-simulation comparison: the estimated Storey FDR was
slightly above the nominal target after the larger sample narrowed the Monte
Carlo band. The failure is retained in `output/campaign/results.json` and the
run ledger; `output/campaign/manifest.json` is intentionally incomplete. This
must be treated as a method/tolerance review item, not silently repaired by
changing the caption, seed, or pass threshold after seeing the result.

The review adds `storey_qvalue_conservative` as a separately named diagnostic
and `storey_qvalue_split_confirmation` as the preregistered independent-stream
remedy. The latter estimates its fixed-lambda scale from a selection vector and
uses an independent confirmation vector for decisions. Its release certificate
is still a finite calibration diagnostic: the confidence accounting and stream
independence are explicit, but this package does not claim a new theorem.
It inverts a fixed-lambda binomial tail to obtain an upper confidence bound on
the null count, so it is more conservative than the unconstrained plug-in on the same
vector and removes the small-family zero-`pi0` degeneracy. The construction is
useful for sensitivity analysis, but the bound and the rejection decisions use
the same p-values. Therefore the package does not promote it to a finite-sample
FDR certificate without an independent selection/confirmation stream or a
separate theorem establishing the required adaptive validity conditions.
