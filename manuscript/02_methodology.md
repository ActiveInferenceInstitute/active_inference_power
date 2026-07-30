# Methodology {#sec:methodology}

The package separates deterministic domain functions from orchestration. The
public API is specified in the [package contract](../docs/PACKAGE_CONTRACT.md);
the experiment script only loads configuration, calls those functions, and
writes artifacts. Every stochastic path accepts or records a seed.

## Estimands, error criteria, and evidence layers {#sec:error-criteria}

The suite connects formal statements to executable objects through a versioned
traceability registry. Each entry identifies a formal object, public symbol,
source module, real-value test, assumptions, evidence class, artifact, and
non-claim. The registry is validated during result generation and carried into
the result artifact with hash `{{RESULT_FORMALISM_HASH}}`.

The table uses word-spaced display labels so long identifiers remain readable
in narrow publication columns; the exact importable symbols, source paths, and
tests remain available in the hashed registry artifact.

| Layer | Formal object | Public anchor / evidence class |
|---|---|---|---|
{{RESULT_FORMALISM_ROWS}}

: Formalism-to-code traceability rows generated from the validated registry;
the table is a lineage contract, not an additional numerical result. {#tbl:formalism-code}

\clearpage

![{{FIGURE_FORMALISM_CODE_MAP_CAPTION}}](../output/figures/formalism_code_map.png){#fig:formalism-code-map alt="{{FIGURE_FORMALISM_CODE_MAP_ALT}}" width=100% height={{FIGURE_FORMALISM_MAP_HEIGHT}}}

\clearpage

![{{FIGURE_FORMALISM_CODE_MAP_CONTINUED_CAPTION}}](../output/figures/formalism_code_map_continued.png){#fig:formalism-code-map-continued alt="{{FIGURE_FORMALISM_CODE_MAP_CONTINUED_ALT}}" width=100% height={{FIGURE_FORMALISM_MAP_HEIGHT}}}

\clearpage

\vspace*{ {{FIGURE_FORMALISM_FINAL_TOP_CLEARANCE}} }

![{{FIGURE_FORMALISM_CODE_MAP_FINAL_CAPTION}}](../output/figures/formalism_code_map_final.png){#fig:formalism-code-map-final alt="{{FIGURE_FORMALISM_CODE_MAP_FINAL_ALT}}" width=100% height={{FIGURE_FORMALISM_MAP_HEIGHT}}}

\clearpage

The testing setting is a separate executable object. It fixes the hypothesis
family, null law, evidence interface, accounting rule, default stopping rule,
cost contract, reported estimands, and validity boundary before replications
begin.
The testing-setting contract in @fig:testing-setting-contract makes that
preregistration surface inspectable; its boxes describe inputs and contracts
rather than measured outcomes.

![{{FIGURE_TESTING_SETTING_CONTRACT_CAPTION}}](../output/figures/testing_setting_contract.png){#fig:testing-setting-contract alt="{{FIGURE_TESTING_SETTING_CONTRACT_ALT}}" width=100%}

### Three contracts, one policy, and two roles

The implementation keeps three probability objects distinct. The agent model
defines the likelihood and transition quantities used for belief updating. The
process defines the evaluator-side distribution from which truth and
observations are drawn, including action-dependent changes. The testing
setting defines the null law, evidence statistic, rejection rule, and error
criterion. The policy is a measurable rule that maps visible history to
actions, target choices, and stopping. A policy can be aligned with the
agent model yet be evaluated under a different generative process; that
mismatch is part of the declared evaluated design, not an implementation
detail.

This separation also fixes the role of simulation. The investigator declares a
scenario, policy, and replication plan; the embedded agent sees only visible
history, while the evaluator retains hidden truth only to score a completed
trace. A random seed controls a reproducible approximation to an operating
characteristic; it is not an additional scientific condition. Seed schedules,
worker counts, and checkpoint cursors are therefore recorded as computational
provenance, while the model, process, setting, and policy define the scientific
estimand.

## The estimand: power as a conditional suite functional {#sec:conditional-power}

The suite makes the conditioning set explicit. A `GenerativeModelSpec`
describes the agent's state, observation, prior, and action-conditioned model;
a `GenerativeProcessSpec` describes the evaluator-side hidden-state transitions
and action-conditioned observations; and a `SettingSpec` declares the
hypothesis family, alpha, correction, default stopping rule, cost budget, and
validity boundary. These three objects are the scientific components
$s=(M_A,P_N,T)$ of the hashable `PowerScenario`; the serialized hash also
binds scenario identity and claim metadata. An action policy $\pi$ is recorded
per result row. When no policy is supplied, execution uses the hashed default
stopping rule in $T$; action-surface comparisons explicitly record their
executed policy. The investigator therefore evaluates $d=(s,\pi)$ rather than
hiding a separately supplied surface policy inside the scenario hash.

For the action-loop surface, the reported power estimand is the scenario-
and-policy-indexed expected replication-level true-positive rate

\begin{equation}
\theta_{\mathrm{TPR}}(d)=
\mathbb E_{P_N,U_\pi}\!\left[
\frac{|\mathcal R_d\cap\mathcal H_1(P_N)|}
{\max\{1,|\mathcal H_1(P_N)|\}}
\right], \qquad d=(s,\pi).
\label{eq:conditional-power}
\end{equation}

Here $P_N$ supplies the declared process randomness, while $U_\pi$ denotes any
policy and trace-level randomization used by $\pi$. The expectation is over
independent replications conditional on the fixed model, setting, and policy
contract. Finite independent replications produce its sample-mean estimator,
with Monte-Carlo standard error (MCSE) describing simulation precision. The
distinct event “at least one true target is rejected” is recorded separately as
`any_true_rejection_rate`; it is not silently substituted for mean TPR.
Changing any conditioning object changes the estimand. The suite therefore
reports scenario-indexed power, observed mean FDP (the finite-simulation FDR
estimate), FWER, stopping time, observation count, target-selection rate, and
cost rather than treating power as an invariant property of an agent or task
environment.

For $m$ hypotheses, let $\mathcal H_0$ and $\mathcal H_1$ denote the true null
and alternative sets. If $\mathcal R$ is the rejection set, then
$R=|\mathcal R|$, $V=|\mathcal R\cap\mathcal H_0|$, and
$S=|\mathcal R\cap\mathcal H_1|$. The true-positive rate (TPR) and other
realized quantities are

\begin{equation}
\operatorname{FDP}=\frac{V}{\max(R,1)}, \qquad
\operatorname{TPR}=\frac{S}{\max(|\mathcal H_1|,1)}, \qquad
\operatorname{FWER\ event}=\mathbf 1\{V\ge 1\}.
\label{eq:error-rates}
\end{equation}

Their Monte-Carlo averages estimate FDR, conditional mean-TPR power, and FWER
respectively.
The denominator conventions make empty-rejection and global-null cases
explicit rather than silently dropping them. They are replication-level
estimands: a ratio of expected counts is not substituted for the expected FDP,
and a mean stopping time is not substituted for a validity guarantee.

## Fixed-horizon correction formalisms {#sec:corrections}

All procedures accept a one-dimensional p-value vector, validate its range, and
return adjusted p-values in input order. The rejection mask is always defined
by adjusted p-value $\le\alpha$.

- **Bonferroni** uses $p_i\le\alpha/m$ and controls FWER under valid marginal
  p-values without a dependence assumption.
- **Šidák** uses $p_i\le1-(1-\alpha)^{1/m}$ under independence.
- **Holm** is the step-down FWER procedure with critical values
  $\alpha/(m-i+1)$.
- **Hochberg** is the step-up FWER procedure under its stated dependence
  conditions [@hochberg1988sharper].
- **BH** rejects through the largest $k$ satisfying
  $p_{(k)}\le(k/m)\alpha$ and reports reverse-cumulative-minimum q-values
  [@benjamini1995controlling].
  Under the independent continuous global null, its FDR is the event
  probability $P(R>0)=\alpha$; the expected false-rejection count $E[R]$ is a
  separate quantity computed from the finite ordered-threshold distribution.
- **Benjamini--Yekutieli (BY)** scales BH by $c(m)=\sum_{i=1}^m1/i$ for an arbitrary-dependence FDR
  guarantee under the procedure's validity assumptions
  [@benjamini2001control].
- **Storey q-values** plug in
  $\hat\pi_0=\min\{1,\#\{p_i>\lambda\}/[m(1-\lambda)]\}$
  [@storey2002direct]. The finite-sample estimate is reported, not treated as
  an exact guarantee.
- **Adaptive BH** implements the two-stage Benjamini--Krieger--Yekutieli
  construction [@benjamini2006adaptive].
- **Weighted BH** applies fixed, positive, normalized hypothesis weights, as in
  the weighted multiple-testing extension of Benjamini and Hochberg
  [@benjamini1997weighted]. Its
  interpretation is conditional on the declared fixed weights and the usual
  independence or PRDS assumptions; weights are not estimated from the same
  p-values in this release track. A weight reallocates the testing budget; it
  does not create information or make a misspecified p-value valid.

The implementation also exposes a dispatcher and a structural staircase check.
Cross-checks include the BH q-values against
`scipy.stats.false_discovery_control`, permutation equivariance, and the
expected ordering of conservative procedures.

### Adaptive-FDR audit: separate calibration from power

The adaptive comparison is intentionally decomposed into distinct estimands.
The Storey power comparison uses common random-number replication streams and
reports the paired difference in average TPR with its own Monte-Carlo standard
error. The BH and Storey FDR checks instead use the replication-level FDP
samples, with separate standard errors and separate finite-band ceilings. The
certificate passes only when the power comparison and both FDR checks satisfy
their declared inequalities. Thus a power gain cannot compensate for an FDR
shortfall.

This distinction is important because the same-vector Storey plug-in estimates
the null proportion from the same p-value vector used for rejection. That
construction is retained as a diagnostic comparison, not
treated as a universal finite-sample guarantee. The conservative fixed-lambda
and independent split-confirmation implementations provide stricter alternatives
whose assumptions are recorded separately. The audit is designed to expose a
small but reproducible calibration discrepancy when increased replication
precision resolves it; it does not convert a finite simulation into a proof.

The finite band is a Monte-Carlo consistency rule around the declared target,
not an inferential interval for an unknown universal FDR. A point estimate may
lie above the nominal target while remaining inside a wider low-replication
band; the publication artifact therefore retains both the point estimate and
the band ceiling. The full metric decomposition is generated in the
supplementary certificate table.

## Analytic operating-characteristic references {#sec:power}

For a one-sample z-test with standardized effect $d$ and noncentrality
$\delta=\sqrt n\,d$,

\begin{equation}
\operatorname{Power}_{1}=1-\Phi(z_{1-\alpha}-\delta),
\label{eq:z-power}
\end{equation}

and the two-sided calculation adds the lower tail. The t-test uses the
noncentral t distribution with $\delta=d\sqrt n$ and one fewer degree of
freedom than $n$ for a one-sample test, or $\delta=d\sqrt{n/2}$ and degrees of
freedom two times $n$ less two for equal-sized groups.
`required_sample_size` uses a monotone integer search. A stable normal-tail
fallback is used only when SciPy returns a non-finite noncentral-t tail at very
large noncentrality.

For independent two-groups p-values, the asymptotic BH threshold $t^*$ is the
largest solution of

\begin{equation}
t^*=\alpha\left[\pi_0+(1-\pi_0)G(t^*)\right],
\qquad G(t)=\Phi(z_t+\text{effect}),\quad \Phi(z_t)=t,
\label{eq:bh-fixed-point}
\end{equation}

and asymptotic average power is $G(t^*)$ [@genovese2002operating]. This is a
model-based asymptotic comparison, not a replacement for the finite-sample
simulation.

## Executable generative-process regimes {#sec:simulation}

Each replication draws

\begin{equation}
z_i=\sqrt\rho F+\sqrt{1-\rho}\,e_i+\mu_i,
\qquad F,e_i\sim\mathcal N(0,1),
\label{eq:factor-model}
\end{equation}

where $\mu_i$ is zero for true nulls and the configured effect for alternatives.
One- or two-sided Gaussian p-values are then corrected and scored against the
known alternative mask. The covariance generator validates positive
semidefiniteness and audits marginal calibration separately for independent,
negative-equicorrelation, block, and factor regimes. The positive-factor
construction is the declared positive-dependence setting for the one-sided
certificate; it is not evidence for arbitrary dependence or for two-sided BH
control. Negative and block regimes are stress comparisons, not theorem
extensions. The Dobriban/BH result is cited for that boundary rather than
re-claimed here [@dobriban2026bh]. The publicly documented finite protocol is
imported by an offline adapter in `active_inference_power.protocols`; its local Monte
Carlo retains the stated model and stratification but remains a stress track.
The external outward-rounded Arb certificate is a separate theorem artifact
and is not reproduced by this package [@dobriban2026bh].

## Belief updates, calibrated evidence, and the p-value boundary {#sec:ai}

The hidden state has levels $H_0$ and $H_1$, observations are `low` and `high`,
and $r=\Pr(\text{high}\mid H_1)=\Pr(\text{low}\mid H_0)$ is the channel
reliability. The real `pymdp.agent.Agent` uses an identity transition, so the
posterior at one step becomes the prior at the next. With $n_h$ high and
$n_l$ low observations, posterior odds are

\begin{equation}
\frac{P(H_1\mid y)}{P(H_0\mid y)}
=\frac{P(H_1)}{P(H_0)}
\left(\frac r{1-r}\right)^{n_h-n_l}.
\label{eq:posterior-odds}
\end{equation}

The belief threshold is intentionally separated from multiplicity control.
Under $H_0$, $n_h\sim\operatorname{Binomial}(T,1-r)$. For $r>0.5$,
`evidence_pvalue` computes the upper-tail Binomial p-value; for the supported
anti-correlated channel $r<0.5$, it uses the corresponding lower tail. The experiment
compares the posterior-threshold decision with BH applied to those calibrated p-values
on identical evidence streams. This is a fixed-horizon calibration step, not a
test-martingale, e-process, or optional-stopping construction
[@shafer2011test_martingales; @grunwald2024safe]. The active-inference
literature motivates the generative-model interpretation, but the statistical
validity claim comes only from the stated Binomial null and the downstream
multiple-testing procedure [@friston2017active_process; @dacosta2020discrete].

## Action-in-the-loop filtration and policy evaluation {#sec:action-loop-method}

The reference action environment has two targets, binary truth, latent
contexts, action-conditioned sensing quality, a context-changing action,
unequal action costs, and a stop action. Policies include fixed horizon,
posterior threshold, expected information gain, cost-normalized information
gain, posterior sampling, adaptive target selection, and e-process stopping.
The real `pymdp` posterior is checked against a closed-form Bayesian update at
every observed step.

The policy receives only its visible history. Hidden process states are retained
in a separate evaluation trace for scoring and never enter action selection,
target choice, or stopping. Target-selection p-values are fed chronologically
to LORD++ or SAFFRON only as conditional diagnostics unless the declared
predictable-selection and conditional-super-uniformity assumptions hold. This
separation prevents a posterior decision, an environment shift, or an
evaluator-only truth label from being mislabeled as family-level evidence. Each
replication retains true and false rejection counts, target-specific rejection
rates, cost and stopping burdens, power per unit cost, and policy-paired
contrasts. Power remains the mean replication-level true-positive rate; FDR
remains the mean replication-level FDP.

The interface contract is summarized in @fig:scenario-decomposition. Reading
from left to right, the model, process, and setting define the hashed scenario;
the separately recorded policy completes the evaluated design, and evaluator-
only truth scores replicated traces. This decomposition is the organizing
object for the roadmap, not an assertion that every downstream class is
formally valid.

### Split-stream confirmation and sequential evidence {#sec:split-confirmation-method}

The formal adaptive-selection protocol is deliberately narrow. The selection
policy is a measurable function of a visible action/observation filtration and
does not receive hidden states or confirmation observations. It selects one
target from the family. A fresh, independent confirmation stream then evaluates
only that target with the implemented fixed-horizon Binomial p-value. Under a
conditional-null law that is super-uniform given the selection history, this
separates discovery information from confirmation information. An e-value
confirmation variant would require its own implemented conditionally valid
e-factor and filtration contract; it is not supplied by this protocol. Split
streams alone do not rescue a misspecified likelihood, truth-changing process,
or invalid stopping rule; they make the required conditional law inspectable.
They also do not make chronological same-stream selection or post-hoc
minimum-p selection valid.

The static confirmation channel is declared separately from the static
e-process optional-stopping contrast. The former tests one selected target on
fresh confirmation data; the latter concerns time-uniform evidence along a
single declared trace. Their shared use of a static process does not make their
filtrations, estimands, or validity labels interchangeable.

The sequential module exposes binary likelihood-ratio e-processes, predictable
alternative schedules, Beta-mixture confidence-sequence inversion, and Levels
based On Number of Discoveries (LOND) e-value accounting.
Each path is serialized with either likelihood-ratio e-process evidence $E_t$,
LORD++/SAFFRON alpha-wealth accounting $W_t^\alpha$, or e-LOND's cumulative
nominal-level allocation $\sum_{i\le t}\alpha_i$, together with the applicable
test levels, e-value thresholds, crossing or stopping time, and filtration
assumptions. For e-LOND, the nominal level is $\alpha_t$, the corresponding
e-value condition is $E_t\ge 1/\alpha_t$, and recycled levels can make the
cumulative allocation exceed $\alpha$; it is neither alpha-wealth nor a
remaining budget. LORD++ and SAFFRON are online p-value procedures whose
levels depend on prior history [@javanmard2018online;
@ramdas2017online_memory; @ramdas2018saffron]; e-LOND uses predictable e-value
accounting [@xu2024online_evalues]. They are reported as finite traces with
uncertainty accounting. Their theorem-level guarantees remain conditional on
the published independence, conditional p-value super-uniformity, or
conditional e-factor mean-at-most-one assumptions and on the arrival filtration
actually used by the policy.

Dynamic process schedules are explicit contracts rather than implicit array
indexing: each schedule declares `hold_last`, `cycle`, or `strict` behavior
outside its stated time support. This removes a source of silent simulation
variation while leaving nonstationary and action-dependent regimes in the
diagnostic or stress-test evidence classes unless their inferential assumptions
are separately established.

### Held-out policy learning {#sec:learned-policy-method}

The learned-policy implementation is a finite tabular search over visible
expected information gain, target uncertainty, and action cost. Candidate
parameters are selected using child streams reserved for training, then frozen
and hashed before evaluation on disjoint child streams. The artifact also hashes
the search configuration and records the candidate count and parameter
dimension, so the finite search burden is part of the provenance rather than an
unstated degree of freedom. This makes leakage auditable and keeps the estimand
explicit. The search is intentionally not
described as optimization of a universal utility function: a held-out policy
comparison is conditional design evidence for the declared scenario.

![{{FIGURE_SCENARIO_DECOMPOSITION_CAPTION}}](../output/figures/scenario_decomposition.png){#fig:scenario-decomposition alt="{{FIGURE_SCENARIO_DECOMPOSITION_ALT}}" width=100%}

## Replication accounting and uncertainty {#sec:mc-accounting}

For every bounded replication-level estimand, the result artifact retains the
replication count, standard error, a Student-$t$ precision interval, and a
distribution-free Hoeffding radius. The implementation uses Wilson or exact
binomial intervals for indicated event rates, paired intervals from
within-replication differences, a configured Hoeffding planning floor, and a
configured Dvoretzky–Kiefer–Wolfowitz (DKW) band for its independent
global-null cumulative distribution function (CDF) audit. These are declared finite-simulation reporting
conventions [@morris2019simulation], not claims that those displays widen a
theorem's assumptions, repair a misspecified process, or create
arbitrary-dependence evidence.

## Source-level workload accounting and isolated timing diagnostics {#sec:complexity}

The suite reports computational feasibility on a separate evidence layer. For
a declared input shape, $Q_{\mathrm{comp}}$ is a deterministic source-work
proxy: it counts dominant passes, history scans, and declared action-loop calls
in the current code path. The same ledger separately records retained draw
elements as implementation-specific retention accounting. For example, the
single-pass correction proxy is $q_{\mathrm{linear}}(m)=m$, while the
sort-dominant proxy is
$q_{\mathrm{rank}}(m)=m\max\{1,\lceil\log_2(m)\rceil\}$. For online traces,
the proxy covers both the recurrence and the package's serialized-trace
validation replay; it does not pretend that a single procedure update is the
whole implementation path. The binary e-process and e-LOND have coincident
displayed linear proxies, so the figure uses one combined curve label while
the analytical rows retain their different evidence and accounting roles. For
named action-loop categories $j$, the ledger
records the permitted step bound
$Q_{\mathrm{action}}=\sum_j N_j H_j$, where $N_j$ is the declared trace-call
count and $H_j$ is that category's maximum permitted horizon. These expressions
are accounting choices tied to this source version; they are not lower bounds,
processor-instruction counts, or complexity statements about a statistical method in
the abstract.

The compact ledger notation is local to this accounting: $m$ is family size,
$B$ is the main independent-replication count, $K$ is the number of compared
procedures, $D$ is the number of declared dependence regimes, $T$ is a
sequential or online horizon, and $R$ is the number of prior rejections in an
online history. Let $w_{\mathrm{corr}}(m)$ denote the source-work cost of
the particular selected correction implementation, rather than an undefined
universal operation. The action row uses the already defined category-specific
$N_j$ and $H_j$, not a second set of opaque symbols.

This layer does not reuse any statistical quantity. Simulated resource cost
$C$ remains the scenario-defined burden accumulated by actions; $E_t$ remains
likelihood-ratio e-process evidence; $W_t^\alpha$ remains LORD++/SAFFRON
procedural alpha-wealth; and e-LOND's $\sum_{i\le t}\alpha_i$ remains a
cumulative nominal-level allocation. $Q_{\mathrm{comp}}$ is dimensionless and
does not measure any of them. The release artifact contains the deterministic
ledger and explicitly
excludes elapsed-time fields, so replication accounting continues to describe
statistical precision rather than computational speed.

Host-specific elapsed time is inspected only in a separate local diagnostic.
It records configured warmups, repeated raw wall and process time samples, and
the runtime environment while keeping input construction outside the timed
region. The use of declared warmups and repeated samples follows benchmark
reporting discipline narrowly [@georges2007rigorous]. The resulting
$\tau_{\mathrm{wall}}$ is an observation of one host and runtime configuration,
not a cross-machine benchmark, a release criterion, or evidence about power,
calibration, or validity.

## Finite-simulation certificates and claim boundaries {#sec:certificates}

Each certificate records an estimate, the analytic or procedural bound, a
Monte-Carlo standard error, the seed, configuration, and a Boolean pass flag.
The predeclared pass rule allows {{CONFIG_CERTIFICATE_MCSE_MULTIPLIER}} Monte-Carlo
standard errors for sampling noise. A passing certificate therefore means
“consistent with the stated bound under this finite simulation,” not “proved by
simulation.” The release finalizer is fail-closed:
if a predeclared certificate fails, the run and manuscript may still be
hydrated for review, but the manifest cannot be marked complete. This makes a
negative result a preserved method-review object rather than an operational
failure that disappears before publication.

## Reporting hierarchy and interpretation rules {#sec:reporting-hierarchy}

The analysis uses a four-level reporting hierarchy. A *formal* result is one
whose stated theorem assumptions are part of the declared scenario-and-policy
(design) contract and whose
implementation has a direct semantic test. A *conditional* result is an
operating characteristic for a fully specified model, process, setting, and
policy, but not a theorem over a broader class. A *diagnostic* result probes
calibration, filtration behavior, or a deliberate contrast within a declared
simulation. A *stress* result changes dependence, missingness, dynamics, or
external data conditions to locate where the conditional interpretation may
fail. These labels are evidence metadata, not prose decoration: they are
serialized with result blocks, figure registries, captions, and claim-ledger
entries.

The hierarchy prevents three common category errors. A high posterior is not a
family-level rejection; a low Monte-Carlo error is not a validity theorem; and
an action policy that performs well in one process is not an optimal or
transportable policy. The same discipline applies to the manuscript: every
reported number is attached to an estimand, replication unit, uncertainty rule,
scenario hash, and claim boundary. The findable, accessible, interoperable,
and reusable (FAIR) research-object framework motivates the artifact lineage,
while the statistical contract determines what the
lineage can and cannot justify [@wilkinson2016fair].

## Why the action loop changes the estimand

In a fixed design, the process distribution can be written before the data are
collected. In an action loop, the policy is part of the data-generating map:
the next action changes the observation kernel, may change the latent context,
and can determine whether another observation exists. Consequently, two
policies evaluated under the same initial state do not generally generate
paired observations. Common random numbers pair the exogenous replication
stream for a contrast, but they do not make the resulting trajectories
identical or erase the policy-induced process difference. The reported paired
quantity is therefore a policy contrast under a shared seed schedule, not a
causal effect without additional design assumptions.

The suite reports this distinction directly. Power is the mean of the declared
replication-level true-positive estimand; FDR is the mean FDP; stopping burden
and cost are means over completed traces; and power per cost is the descriptive
mean of replication-level $\mathrm{TPR}/C$ scores (recorded as zero for a
zero-cost trace), not a ratio of scenario summaries. No ratio of pooled counts
is substituted for mean FDP, and no cost-normalized score is presented as a
utility function. This separation keeps inferential validity, resource
efficiency, and agent belief quality as related but non-interchangeable
outcomes.
