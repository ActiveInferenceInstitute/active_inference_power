# Active Inference Power research roadmap

This roadmap defines the order in which the suite may acquire broader claims
and records explicit non-claims for every incomplete stage.
It is a research control document, not a promise that every proposed method is
already implemented. Each stage has an entry criterion, a deterministic output,
and a claim boundary.

## North-star estimand

For a hashable scenario $s=(M_A,P_N,T)$ and separately evaluated policy
$\pi$, let $d=(s,\pi)$. The action-loop power surface reports

\[
\theta_{\mathrm{TPR}}(d)=
\mathbb E_{P_N,U_\pi}\!\left[
\frac{|\mathcal R_d\cap\mathcal H_1(P_N)|}
{\max\{1,|\mathcal H_1(P_N)|\}}
\right].
\]

The separately reported `any_true_rejection_rate` is the different event
\(\Pr(|\mathcal R_d\cap\mathcal H_1|>0)\). The investigator uses a declared
replication plan to estimate these fixed quantities and their Monte-Carlo
precision; the plan does not define the estimand. They are not intrinsic
properties of an agent or a distribution over real-world niches.

Here \(P_N\) denotes the declared process randomness and \(U_\pi\) denotes
policy and trace-level randomization used by \(\pi\). Both are conditioned on
the fixed scenario-and-policy contract when the investigator estimates the
surface.

The evaluated design combines these non-interchangeable layers:

1. **Generative model** — the agent's states, observations, priors,
   action-conditioned likelihoods, transitions, and preferences.
2. **Generative process** — the hidden truth, environment transitions,
   emissions, missingness, and action costs that generate data.
3. **Testing setting** — the hypothesis family, error criterion, alpha,
   stopping rule, budget, and validity assumptions.
4. **Agent policy** — a separately recorded rule that maps visible history to
   target, sensor, context, or stopping actions.

Changing any layer changes the estimand. Results are therefore never reported
as an unconditional property of an agent.

## Evidence classes

| Class | Permitted use | Required evidence | Prohibited interpretation |
|---|---|---|---|
| Formal conditional | A theorem or exact validity result under declared assumptions | Assumption contract, oracle tests, and proof-linked implementation | Universal validity outside the contract |
| Conditional operating characteristic | Power, FDR, FWER, cost, or stopping under one scenario/policy | Seed schedule, finite-simulation interval, scenario hash, and sensitivity grid | An invariant agent-level power number |
| Calibration diagnostic | Check marginal calibration or finite-stream behavior | Null simulation, Wilson/DKW/MC uncertainty, and explicit calibration target | A proof of a new theorem |
| Trace diagnostic | Make one retained path's visible information and decision contract inspectable | Explicit pathwise labels plus a separate replication-level summary | A population estimate, operating-characteristic mean, or optional-stopping guarantee |
| Stress/exploratory | Probe model/process misspecification or external data | Provenance, leakage controls, and a no-claim label | Promotion of benchmark or stress behavior into synthetic guarantees |

## Staged program

### Stage 0 — canonical surface and terminology (complete)

The public namespace is `active_inference_power`. Domain logic is separated
from orchestration, the installed `pymdp.agent.Agent` is exercised directly,
and the posterior-threshold decision is named as a decision layer rather than
an error-control method. The reference-horizon baseline remains available as a
calibration anchor; it is not treated as the suite's scientific endpoint.

Acceptance: clean wheel import, real-agent/oracle agreement, schema validation,
one canonical public namespace, and reproducible release artifacts.

Stage 0 also includes the formalism-to-code registry. Its role is to keep the
conditional-power functional, model/process/setting contracts, Bayesian oracle,
corrections, e-processes, online-FDR recursions, action filtration, uncertainty
accounting, and artifact lineage connected to real source symbols and tests.
The registry is a traceability contract; it does not convert a finite simulation
into a theorem.

### Stage 1 — predictable adaptive target selection (formal-conditional contract implemented; finite diagnostic assumption-gated)

Pre-register the target-selection policy, define the filtration visible to the
policy, and prove or verify conditional super-uniformity for each selected
test. The primary promotion path is independent split-stream confirmation:
selection uses only its visible filtration and confirmation uses a fresh child
stream. Chronological online-FDR accounting remains a diagnostic comparison.

Deliverables: `SelectionPolicySpec`, `FiltrationEvent`, `SelectionTrace`,
`SplitConfirmationResult`, independent stream hashes, conditional-null
calibration accounting, paired power/cost surface, and an explicit failure case
showing what breaks when selection uses future evidence.

The current release adds the executable protocol and measurement infrastructure
for this lane:
expected-information-gain, cost-normalized, and posterior-sampling policies;
target-specific rejection rates; cost/stopping intervals; and simultaneous
common-random-number policy contrasts. These additions improve design
accounting but do not by themselves satisfy the Stage 1 validity certificate.

Boundary: adaptive target selection is not selective inference merely because
it is chronological. Selection-adjusted inference and online-FDR validity are
distinct contracts.

The release now also enforces the adjacent evidence contract mechanically:
claim-bearing ledger rows require an evidence class and non-claim, and action
e-process labels are withheld when latent context makes the Bernoulli rates
state-dependent. It computes the null conditional-selection calibration
diagnostic using an explicit all-null process, target-level and pooled empirical
CDF excess, and DKW radii, alongside a deliberately invalid full-trace
minimum-p contrast. The split-stream path receives a formal-conditional label
only when the static confirmation channel and conditional-null contract pass.
Its finite simulations remain conditional calibration evidence and uncertainty
accounting, not a proof for an unexamined process or arbitrary adaptive
selection.

### Stage 2 — time-uniform evidence and online e-values (implemented; assumption-gated)

The suite now exposes predictable e-factors, Beta-mixture Bernoulli confidence
sequences, and e-LOND nominal test levels with a cumulative nominal-level
allocation trace. Compare optional stopping, reference-horizon calibration,
LORD++/SAFFRON, and e-value methods using common random numbers.

Deliverables: known-vector tests, Ville-bound simulation, optional-stopping
stress grid, confidence-sequence coverage checks, online alpha-accounting checks,
and simultaneous uncertainty bands.

Boundary: a valid e-process for one stream does not automatically give online
FDR control for adaptively selected targets. e-LOND is reported only with its
conditional e-value filtration assumptions. Its accounting is the cumulative
emitted nominal levels, not alpha-wealth or a remaining budget; recycling can
make the trace exceed $\alpha$.

### Stage 3 — process realism and policy sensitivity (implemented; diagnostic lane)

Add serializable nonstationary emissions, persistent hidden context,
action-dependent transition kernels, MCAR/MAR/MNAR missingness, and reliability
drift. Keep
truth generation and agent belief states separate. Every extension requires a
paired static-process control and a sensitivity surface over the new process
parameters.

Deliverables: process-spec versioning, marginal-calibration panels, drift
diagnostics, cost-adjusted power frontiers, and a predeclared policy comparison.

Process schedules now have an explicit extrapolation contract: `hold_last`,
`cycle`, or `strict`. This is a reproducibility safeguard for dynamic regimes,
not a claim that any schedule is realistic.

Boundary: a richer environment is a better stress test, not evidence of a
general active-inference guarantee. Dynamic schedules and MAR/MNAR regimes are
never promoted to formal optional-stopping claims by simulation alone.

### Stage 4 — policy learning with evaluation separation (implemented; conditional lane)

The first learned-policy lane is a deterministic finite candidate search over
visible information gain, target uncertainty, and action cost. It trains only
on independent simulation streams, evaluates on fresh child streams, and
records frozen parameters and a policy hash. The artifact also hashes the
complete finite search configuration and records candidate-space size and
parameter dimension, so a repeated selected vector cannot hide a changed
search procedure.

Deliverables: train/evaluate seed partition, policy artifact hash, search
configuration hash, leakage test, held-out operating characteristics, and
explicit policy-complexity accounting.

Boundary: improved reward, expected free energy, or posterior accuracy is not
statistical power and does not establish family-level error control. The
released policy search is finite and held out; it is not an optimality claim.

### Stage 5 — external benchmark stress (implemented; offline fixture lane)

Stage the checksum-pinned Golub fixture through a reviewed reader and fixed
feature-level recipe. Add public agent trajectories only with chronological
splits, explicit missingness handling, and versioned provenance. Keep external
results separate from synthetic certificates.

Deliverables: offline loader, checksum test, preprocessing manifest, benchmark
dashboard, and a claim-ledger entry that states the benchmark's non-claim. The
offline preflight runner additionally validates the committed manifest and
locally staged fixture hashes, then emits a JSON/HTML provenance dashboard and
manifest without downloading a source archive. An unstaged archive remains
explicitly unstaged in that receipt; the preflight is not an acquisition,
license adjudication, biological reanalysis, or external-data release.

Boundary: external data can demonstrate robustness or failure modes; it cannot
upgrade synthetic assumptions into universal guarantees. The committed Golub
projection is an adapter/stress fixture, not a biological reanalysis.

### Cross-stage bounded research atlas (implemented runner; non-promoted)

`scripts/run_research_atlas.py` composes the R1–R5 and G4 extensions into one
predeclared, bounded diagnostic plan: testing-setting sensitivity, stratified
split-stream calibration, paired dynamic-process controls, sequential-evidence
comparisons, repeated held-out policy partitions, and multi-target arrival
filtration. It requires a new isolated `--output-root` and writes an atlas,
two accessible figures, a figure registry, dashboard payload, and a
hash-linked manifest. It rejects the canonical release root and any existing
destination.

The atlas records finite operating characteristics, scenario/configuration
hashes, evidence classes, and non-claims. It deliberately emits no terminal
certificate, manuscript result, publication-ready signal, universal policy
ranking, or theorem status. Its role is to make the scope of these implemented
research lanes inspectable together, not to merge their assumptions or expand
their claims.

### Stage 6 — publication and release

Regenerate results, uncertainty summaries, figures, captions, dashboard,
manuscript variables, resolved manuscript, PDF, and HTML from one validated
configuration. Require complete hashes, clean installation, branch coverage,
rendering integrity, and zero unresolved numeric or token audits.

### Stage 7 — expanded conditional-power campaign (v2 finalized; non-promoted)

The original `campaign` profile was the reproducible path for increasing
resolution and Monte-Carlo precision without overwriting the release anchor.
Its historical grid, replication plan, target band, and observed summaries are
receipt fields, not maintained roadmap constants; inspect the retained campaign
configuration, `results.json`, `certificates.json`, and `run_log.json` when an
exact historical value matters.

Entry criterion: the source configuration validates, campaign profile
resolution is covered by real-file tests, and the isolated output namespace is
empty or contains only contract-matching checkpoints. The first campaign
execution completed every simulation stage and all result/figure/schema/hash
checks except the terminal certificate gate. Its retained Storey-versus-BH
receipt did not satisfy the predeclared finite-simulation band, so its manifest
remains incomplete and its run log records a failed terminal state. The exact
estimate, target, interval, and replication count belong to that retained
receipt rather than this narrative. This is a research finding, not a reason to
relabel the campaign as passed: a future rerun would require review of the
Storey procedure/tolerance contract and a preregistered remedy, or retention of
the method as diagnostic. It is not an open execution instruction: the
separately frozen v2 lane below has completed its own artifact contract without
rewriting this failed evidence.

A subsequent precision audit used a larger independent campaign output root and
was also retained fail-closed. Its larger replication budget reduced
Monte-Carlo uncertainty but did not remove the Storey certificate shortfall.
This is the expected behavior of the gate: increasing sample size can reveal a
small finite-sample discrepancy more clearly. The precision artifact is useful
for method review and rendering checks, but it is not substituted for the
passed release anchor and does not change the synthetic claim boundary.

The frozen v2 lane is
`data/research_lanes/conditional_power_campaign_v2.yaml`, with the exact
canonical `output/campaign-v2/` root. It preserves the enlarged simulation grid and
fixed Storey tuning while serializing an exact certificate-policy partition:
BH FDR, Bonferroni FWER, positive-dependence, and independent split-stream
Storey confirmation are terminal; the same-vector Storey power-gain record is
diagnostic. The diagnostic record is still generated, hashed, and displayed
with its verdict; it is not relabeled as passing. The canonical release and
the preserved campaign/precision bundles retain all-terminal policies; the
preserved campaign and precision receipts remain fail-closed.

The v2 campaign was source-contract refreshed and finalized under that frozen
policy plus deterministic source-work provenance. Its retained canonical receipt
at `output/campaign-v2/` binds a complete manifest, configuration and execution
witnesses, figures, stage checkpoints, and two hash-bound predecessor
certificate receipts. The source-work plan is accounting rather than a timing,
cost, or statistical claim. A prior complete v2 receipt, when retained, is
byte-preserved in a timestamped archival sibling; its exact directory and
digest belong to the archived manifest rather than this roadmap. All four
terminal certificate records passed. The same-vector
`storey_power_gain` record remains the sole diagnostic failure, not a passing
or promoted result. This is completion of the v2 campaign artifact contract;
it does not certify the canonical release or authorize publication. Treat the
receipt as historical evidence after any source change: archive and regenerate
the frozen lane before describing a campaign artifact as source-current.

The fixed-lambda conservative null-count upper-bound procedure remains a
separately named same-vector diagnostic when estimation and rejection share
the p-value vector. A formal adaptive lane must use the independent
split-stream contract or supply a verified finite-sample proof. Changing a
threshold, seed, or policy classification after observing the result is out of
scope. See [the Storey certificate decision](storey_certificate_decision.md)
for the frozen policy and execution sequence.

### Stage 8 — imported two-sided non-PRDS protocol boundary

The project includes an offline adapter for the publicly documented Dobriban/BH finite
protocol accompanying the publicly posted preprint. The committed plan fixes the two-sided Gaussian factor model
(`pi0=0.96`, correlations `0.30`, `-0.30`, and `-0.72`, effects `2.4` and
`4.4`), 1,000 common-factor strata, PCG64, and the seven publicly documented child
seeds. `run_dobriban_bh_chunk()` executes that finite stratified Monte Carlo
for a declared chunk and emits macro-level t intervals, but its evidence class
is `stress_test`. The external repository's outward-rounded Arb certificate is
not reimplemented here; no local Monte Carlo number is presented as that
theorem. See the primary paper and bundle [@dobriban2026bh].

The adapter is deliberately separate from the release's positive-factor
one-sided certificates. It cannot be used to claim arbitrary-dependence BH
control, universal failure rates, or a stronger Active Inference result.

### Stage 9 — split-stream Storey review

The Storey remedy is preregistered in `manuscript/config.yaml`: fixed
`lambda=0.5`, 99% one-sided null-count confidence, and independent selection
and confirmation streams. `storey_qvalue_split_confirmation()` estimates the
null-count scale only from the selection stream and applies it only to the
confirmation vector. Its release certificate is a finite calibration
diagnostic under those contracts; it is not a universal finite-sample FDR
theorem. The v2 campaign retains this independent check as terminal but does
not thereby turn the same-vector record into a theorem. The original
same-vector comparison and its failed expanded-campaign gate remain visible
and are not retroactively relabeled.

The campaign remains conditional operating-characteristic evidence; larger
samples reduce simulation uncertainty but do not establish arbitrary
dependence validity, policy optimality, or universal active-inference power.

## Change-control rules

- New methods require a primary citation, an implementation contract, a test
  vector or oracle, an uncertainty plan, and a claim-ledger entry.
- New environments require a process specification, calibration check,
  scenario hash, and a stress/conditional/formal label.
- New policy actions require a visible-history test and a cost/stopping account.
- New figures require a registry record containing estimand, uncertainty,
  alt text, caption, and validity boundary.
- Generated numbers never enter manuscript prose by hand.
- A failed or incomplete stage remains visible in the ledger; it is not renamed
  as a success by changing the caption.

## Current status

Stages 0–5 have executable local lanes with the evidence labels stated above.
The bounded research-atlas runner makes the R1–R5/G4 extension contracts
inspectable in a separate, non-promoted output bundle; it does not replace a
release run or create a broader claim. The frozen v2 campaign has been
generated and finalized under its own exact configuration witness. Its
campaign-specific terminal records passed, while the same-vector Storey record
remains serialized as a diagnostic failure. That limited completion neither
changes the canonical release's all-terminal policy nor creates a release or
publication claim.
Stage 1 split confirmation is the primary formal-conditional
adaptive-selection contract only under its independent-stream,
static-confirmation-channel, and conditional-null assumptions; its finite
simulation summaries remain conditional diagnostics rather than a new theorem.
Stage 6 has historical local release-validation receipts, including a then-current
finalizer, branch gate, clean wheel import, offline examples, and
controlled-renderer PDF/HTML rendering. They are not source-current after later
source, configuration, and template changes. Before Stage 6 can be described
as current again, one new candidate must complete result generation, quality,
strict hydration, reader-facing audits, controlled-renderer rendering,
renderer-provenance capture, and unskipped review; finalization remains a
separate authorized action. No retained local render creates a publication
claim. Any next research work is a new claim expansion within the declared
gates—not a relabeling of diagnostics as universal validity—and must be added
with the change-control requirements above before execution. This ordering
keeps the suite composable and prevents action-in-the-loop demonstrations from
outrunning their statistical assumptions.
