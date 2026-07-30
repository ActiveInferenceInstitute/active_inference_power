# Introduction {#sec:introduction}

## A concrete adaptive-design problem

Consider the controlled action loop used in this suite. A policy faces two
candidate targets with binary truth and a latent context. On each round, it
chooses a target and a sensing quality, receives an observation, and either pays
to continue or stops. Buying a more reliable observation after a promising
signal can change the chance of identifying a true alternative, the data that
will be available later, and the resources that the run consumes. Repeatedly
looking at a fixed-horizon statistic can also change the inferential question.
The study therefore needs more than a headline power value: it needs a
predeclared account of the process, decision rule, and stopping rule
[@chernoff1959sequential; @lindley1956information].

There are two roles in that study. An embedded agent uses only its visible
history to form beliefs and choose an action. An investigator specifies the
synthetic task environment, testing contract, policy, and replication plan,
then uses evaluator-only hidden truth to score completed traces. Thus the
agent faces uncertainty *within* a specified process, whereas power is
estimated *across* replications of an investigator-declared design
[@neyman1933efficient; @morris2019simulation]. In this manuscript, “simulated
niche” is only informal shorthand for that evaluator-side process; it is not a
claim about an empirical ecological, social, or deployment population.

The suite consequently has two related study lanes. The fixed-horizon reference
is an investigator simulation of declared observation streams, calibrated
evidence, and correction procedures; it does not by itself instantiate an
action-controlled agent--environment loop. The action-loop lane adds an
embedded agent whose policy chooses actions under uncertainty within the fixed
synthetic process. In both lanes, reported power is an operating characteristic
of the investigator's declared design, not a general measure of agent
competence, fitness, or task-environment quality.

Active inference supplies a language for the data-collection part of that
problem. An agent represents hidden states, updates model-relative beliefs from
observations, and selects actions or policies using stated preferences and
epistemic value [@friston2016active_learning; @friston2017active_process;
@parr2019generalised]. In a discrete state-space model, those ingredients can
be written as explicit likelihoods, transitions, observations, and policy
evaluations [@dacosta2020discrete; @heins2022pymdp]. Multiple testing supplies
the complementary decision contract: a family procedure controls a named error
criterion only for evidence objects and dependence conditions in its stated
scope [@benjamini1995controlling; @benjamini2001control].

The practical question is consequently not whether active inference has one
universal power. It is **how an investigator-estimated operating characteristic
changes when a generative model, generative process, testing setting, or policy
changes**. A
posterior belief can help a policy choose an observation, but it is not thereby
a calibrated p-value or e-value. A likelihood-ratio e-process can support a
stopping-time statement only under its declared null and filtration conditions.
An online false discovery rate (FDR) procedure uses a different accounting
contract again. This manuscript makes those distinctions visible before it
compares procedures or policies.

## Reader's guide to the conditional-power question

Readers can use the manuscript in three passes. First, identify the scenario:
what process creates observations, what policy can see, and what it costs to
collect another observation. Next, identify the evidence interface and error
criterion: a posterior, p-value, e-process, and online alpha-accounting balance
are related computational objects but do not license the same decision claim.
Finally, read the uncertainty and validity label with every result. The label
states whether a row is a formal conditional statement, a scenario-specific
operating characteristic, a calibration diagnostic, a stress test, or a
deliberately invalid contrast. The public application programming interface
(API) names what downstream code can call; JavaScript Object Notation (JSON)
names the result, trace, manifest, and dashboard payloads through which the
same declared objects can be inspected.

## Study aims and evidence boundary

The reader guide yields four linked, executable questions. The fixed-horizon
reference asks whether a posterior-threshold decision and a family correction
applied to calibrated evidence behave differently on the same observation
streams. The action-loop studies ask how sensing quality, target choice,
environment-changing actions, stopping, and resource cost alter conditional
power and error summaries. The adaptive-selection studies ask whether the
information used to select a target is separated from its confirmation evidence.
The sequential studies ask which conclusion remains available when the evidence
object and filtration support continued sampling.

Each answer has a bounded evidence class. The package supplies a tested
comparison surface for established correction procedures, not a new FDR
procedure or a general active-inference model. Its practical contribution is an
inspectable, reproducible path [@sandve2013ten_rules; @wilkinson2016fair] from
scenario contract to rendered artifact; provenance cannot make an uncalibrated
object valid. Independent-confirmation, e-process, confidence-sequence, and
online-FDR tracks retain separate assumptions, and common-random-number
simulations estimate declared-scenario operating characteristics and paired
contrasts rather than proofs [@morris2019simulation].

### Notation and terminology: a reader guide

#### Design and information contract

| Reader term | Meaning in this study | Why it matters |
|---|---|---|
| **Scenario** $s$ | The scientific model/process/setting components of one `PowerScenario`: a fixed agent-side generative model, evaluator-side generative process, and testing setting. | Its serialized hash also binds scenario identity and claim metadata; a separately supplied surface/action policy is not hidden in that hash. |
| **Evaluated design** $d=(s,\pi)$ | One scenario paired with an executed agent policy. The replication plan $B$ controls the estimator's precision, not the definition of $d$. | This is the investigator-facing object whose operating characteristics are estimated and compared. |
| **Generative model** | The agent-side representation of hidden states, observations, priors, likelihoods, transitions, preferences, and available actions. | It determines how the agent forms beliefs and evaluates actions. |
| **Generative process** $P_N$ | The evaluator-side mechanism that generates hidden truth, context, action-dependent observations, missingness, reliability, and costs. “Simulated niche” is informal shorthand only. | It determines the data and truth against which an operating characteristic is scored; it is not a sampled population of real-world environments. |
| **Testing setting** | The hypothesis family, null law, evidence statistic, correction, nominal level, stopping rule, cost budget (a cap in the declared $C$ units), and stated validity assumptions. | It fixes what a rejection and an error criterion mean without turning a cost ceiling into alpha accounting. |
| **Policy** $\pi$ and visible filtration $\mathcal F_t$ | A policy maps only the history visible before time $t$ to an action, target choice, or stopping decision. Hidden evaluator truth and future confirmation observations are excluded. | This information boundary determines whether adaptive selection or stopping has the claimed interpretation. |
| **Resource cost** $C$ | Cumulative simulated sampling cost in the scenario's hash-bound `cost_unit` and `cost_scale_id`. The current scenarios do not identify those units with money, time, energy, or physical resource use; cross-scenario cost comparisons require matching identifiers. | It reports what a trace consumed; it is neither a probability, evidence score, nor an accounting balance. |
| **Source-work proxy** $Q_{\mathrm{comp}}$ | A deterministic, version-specific proxy assembled from named array passes, history scans, and declared execution paths; the ledger reports retained-draw elements separately as retention accounting. | It makes the code path and input shape auditable; it is not simulated cost $C$, elapsed time, a memory measurement, statistical evidence, or alpha accounting. |
| **Local elapsed time** $\tau_{\mathrm{wall}}$ | Repeated wall-clock samples from one named local runtime configuration, retained only in an isolated diagnostic. | It can describe that host's observed timing protocol but cannot supply a cross-hardware ranking, a release gate, or a statistical guarantee. |
| **Conditional power / mean true-positive rate (TPR)** $\theta_{\mathrm{TPR}}(d)$ | The expected replication-level true-positive rate under the fixed scenario and policy. Independent replications estimate it with MCSE; the different “any true target rejected” event is reported separately. | It makes the conditioning explicit instead of treating power as a property of an algorithm, agent, or task environment alone. |

This separation is substantive. The model can be misspecified relative to the
process. A policy can alter the process by choosing a sensing channel or a
target. A testing setting can require evidence that a posterior does not
provide. The suite records the scenario components and their hashes so a
comparison remains attached to the design that produced it.

#### Belief, calibrated evidence, and accounting are not interchangeable

For a visible history $\mathcal F_t$, a posterior such as
$q_t=\Pr(H_1\mid\mathcal F_t)$ answers a model-relative question: given the
prior, likelihood, and observations, how much probability does the agent assign
to a hidden state? It may be useful for action selection, but it has no
family-level error interpretation by itself. The fixed-horizon reference turns
the same observation stream into a p-value only by specifying a null
distribution, then passes that calibrated evidence to a family procedure.

| Reader term | Meaning in this study | Not the same thing as |
|---|---|---|
| **Posterior belief** $q_t$ | Model-relative probability assigned to a hidden state after the visible observations. | A calibrated p-value, e-value, FDR guarantee, or resource budget. |
| **P-value** $p_i$ | A statistic whose null-distribution behavior is declared for hypothesis $i$; the fixed-horizon reference uses it as the input to a family correction. | A posterior probability or a license to keep peeking after the stated horizon. |
| **Named online procedures** | Levels based On Recent Discovery (LORD) is an online p-value family; the implemented LORD++ rule is one member. Serial estimate of the Alpha Fraction that is Futilely Rationed On true null hypotheses (SAFFRON) uses candidate information under its stated assumptions. Levels based On Number of Discoveries (LOND) underlies the distinct predictable e-value procedure e-LOND described below. | A substitute for an e-process or a proof about an undeclared arrival process. |
| **Likelihood-ratio e-process** $E_t$ | A nonnegative sequential evidence process that is a supermartingale under its declared null (a martingale for a correctly specified likelihood ratio); each predictable multiplicative e-factor has conditional mean at most one. Its stopping-time interpretation is conditional on the predictable update rule and model/process likelihood agreement [@shafer2011test_martingales; @grunwald2024safe; @howard2020time_uniform]. | Online alpha accounting, a posterior, or a simulated resource cost. |
| **Online alpha-wealth** $W_t^\alpha$ (LORD++/SAFFRON) | A bookkeeping balance that determines which future p-value testing levels an online procedure may spend or replenish under its stated rule. Its update depends on the declared history and assumptions [@foster2008alpha_investing; @ramdas2017online_memory; @ramdas2018saffron]. | The e-process value $E_t$, an effect size, expected utility, or a sampling budget. |
| **e-LOND nominal levels and cumulative allocation** | e-LOND emits predictable nominal test levels $\alpha_t$ for e-values and rejects when $E_t\ge 1/\alpha_t$. Its accounting trace is the cumulative emitted level $\sum_{i\le t}\alpha_i$; recycled levels after rejections can make that trace exceed the nominal $\alpha$ [@xu2024online_evalues]. | Alpha-wealth, a remaining alpha budget, the e-process value $E_t$, or a resource cost. |
| **Resource cost** $C$ | The cost accumulated by chosen actions in the declared generative process. | Either form of statistical evidence or alpha accounting. |

To avoid ambiguity, this guide qualifies the accounting quantities. A plotted
e-process path is called the e-process value $E_t$; LORD++/SAFFRON panels call
their balance alpha-wealth $W_t^\alpha$; and e-LOND panels call their path the
cumulative nominal-level allocation $\sum_{i\le t}\alpha_i$. Captions also
state the e-LOND decision rule $E_t\ge 1/\alpha_t$ where it matters. Neither quantity is the resource cost $C$. This naming convention prevents a
numerical display label from suggesting a statistical guarantee that belongs to
another object.

Computational feasibility has two further, deliberately separate labels.
The release ledger uses $Q_{\mathrm{comp}}$ for a dimensionless source-work
proxy derived from the current implementation and the declared input shape.
An optional local diagnostic uses $\tau_{\mathrm{wall}}$ for repeated elapsed
time on one host. Neither is simulated resource cost $C$, e-process evidence
$E_t$, LORD++/SAFFRON alpha-wealth $W_t^\alpha$, or e-LOND cumulative
nominal-level allocation; none changes a power, FDR, FWER, or validity label.
This distinction lets a reader ask whether a design is computationally feasible
without mistaking a machine observation for statistical evidence.

#### Error rates, uncertainty, and evidence labels

| Reader term | Working definition and boundary |
|---|---|
| **True nulls, alternatives, and rejections** | $\mathcal H_0$ and $\mathcal H_1$ are the declared true-null and true-alternative sets; $\mathcal R$ is the rejection set, with total rejections $R$, false rejections $V$, and true rejections $S$. |
| **False discovery proportion (FDP)** | The realized ratio $V/\max(R,1)$ for one replication. It is a path-level outcome, not a guarantee. |
| **False discovery rate (FDR)** | The expectation of FDP under the declared model, process, evidence, and correction. Independent replications estimate it with MCSE. |
| **Family-wise error rate (FWER)** | The probability of at least one false rejection under the declared setting. It is a different target from FDR and typically has a different power trade-off. |
| **Calibration** | Agreement of an evidence object's behavior with its declared null-law requirement. A model-relative posterior, even if assessed as calibrated under its own generative model, is not automatically a calibrated p-value or e-value. |
| **Positive regression dependence on a subset (PRDS)** | A dependence condition relevant to selected FDR guarantees. The suite's positive-factor simulations are stress evidence under a declared regime; they do not establish arbitrary-dependence behavior [@benjamini2001control]. |
| **Monte Carlo standard error (MCSE)** | The simulation standard error of a replication-based estimate. It describes finite-simulation precision, not a theorem about a broader population of processes [@morris2019simulation]. |
| **Split-stream confirmation** | Selection uses one visible data stream and the selected target is tested on a fresh independent confirmation stream. Its conditional-null contract must still be stated; a split alone cannot repair a misspecified process or stopping rule [@fithian2014selective]. |
| **Confidence sequence** | A sequence of intervals with a stated time-uniform interpretation under its declared process and filtration assumptions [@howard2021confidence_sequences]. |
| **Validity label** | A compact evidence class carried with results and figures: *formal conditional* when a stated theorem contract and its implementation checks apply, *conditional* for a fully specified operating characteristic, *trace diagnostic* for one retained path, *diagnostic* for a calibration or contrast check, *stress* for a broader perturbation, and *invalid contrast* for a deliberately unsuitable decision path. |
