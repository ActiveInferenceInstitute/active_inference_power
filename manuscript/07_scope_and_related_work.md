# Scope, Scholarship, and Formal-to-Code Positioning {#sec:scope}

This project sits at an interface between statistical design and
active-inference implementation. The two literatures retain distinct objects
and responsibilities. Statistical theory supplies estimands, null laws, error
criteria, and validity conditions. Active-inference theory supplies a language
for generative models, belief updating, action, and policy selection. The suite
connects them through an executable boundary: beliefs remain model-relative
states, evidence is calibrated under a named testing setting, and family-level
decisions are made by a declared procedure.

The same division assigns two roles. The investigator declares and repeatedly
simulates a testing design; the embedded agent performs within-trace inference
and action from visible history. Evaluator-only hidden truth scores outcomes
after the trace. This is a design-based operating-characteristic study in the
sense of the testing and simulation-reporting literatures
[@neyman1933efficient; @morris2019simulation], not evidence that a synthetic
task environment represents an empirical niche or that an agent possesses an
intrinsic power value.

The fixed-horizon comparison lanes are investigator simulations of declared
streams, evidence interfaces, and correction procedures; they do not by
themselves instantiate an action-controlled agent--environment loop. The
action-loop lanes add an embedded policy acting under uncertainty within a
specified synthetic process. In both cases, the reported power belongs to the
investigator's declared design and does not benchmark an agent's general
competence, fitness, or ecological adaptation.

## Design-based power and statistical error criteria

The classical testing tradition makes power meaningful relative to a specified
null, alternative, test, and sampling distribution [@neyman1933efficient]. In
this suite, that design dependence is extended to action: the policy can alter
which target is observed, which channel is used, how the process evolves, and
when sampling stops. The resulting quantity is therefore a conditional
operating characteristic indexed by the complete design tuple.

Sequential experimental design gives this extension an important historical
precedent. Chernoff's work formalizes information acquisition when later
sampling decisions depend on earlier observations [@chernoff1959sequential],
while Lindley provides an information-based account of experiment choice
[@lindley1956information]. The action-loop policies here use finite,
scenario-specific information and cost summaries inspired by that design
tradition. They provide operating-characteristic comparisons for declared
policies; the suite does not claim asymptotic optimality or a universal value of
information.

Bonferroni, Holm, and Hochberg provide FWER-oriented procedures
[@holm1979simple; @hochberg1988sharper]. Benjamini and Hochberg introduced the
FDR criterion [@benjamini1995controlling], while Benjamini and Yekutieli
established a dependence-sensitive boundary and the PRDS context
[@benjamini2001control]. Storey and Benjamini--Krieger--Yekutieli provide
adaptive alternatives [@storey2002direct; @benjamini2006adaptive]. Genovese
and Wasserman connect the BH threshold to operating characteristics and
asymptotic power [@genovese2002operating].

Blanchard and Roquain provide a proof-oriented decomposition of FDR control
into self-consistency and dependency-control conditions
[@blanchard2008simple]. This is the natural scholarly template for extending
the suite beyond finite calibration diagnostics: an implementation should name
the rejection rule, the filtration, and the dependence contract before a
simulation result is promoted to a formal claim.

The weighted extension is a separate design choice rather than a cosmetic
rescaling: fixed positive weights encode priorities that must be specified
before inspecting the p-values [@benjamini1997weighted]. The implementation
validates positivity and normalization, tests permutation behavior under the
declared weighted rule, and labels the independence/PRDS boundary in its
registry. No data-adaptive weight selection is folded into the release
comparison.

The project implements these methods as a tested comparison surface; it does
not introduce a new correction. The fixed-point calculation is an asymptotic
reference, while two-groups simulations expose finite behavior for explicit
factor, block, and negative-correlation designs. A procedure's guarantee is
conditional on its assumptions and target error criterion, not on the visual
appearance of a power curve.

## Sequential selection, online testing, and adaptive sensing

Online testing changes the information structure: hypotheses arrive in an
ordered stream and the testing level may depend on past decisions. Javanmard
and Montanari established online FDR control for generalized alpha-investing
rules under explicit dependence conditions [@javanmard2018online]. Ramdas et
al. developed LORD++-style online control with decaying memory
[@ramdas2017online_memory], and their SAFFRON work introduced candidate-indexed
adaptation [@ramdas2018saffron]. Xu and Ramdas develop an e-value formulation
that makes predictable e-value evidence and a separate cumulative nominal-level
allocation path central
[@xu2024online_evalues].

These methods are natural comparators for an agent that chooses what to sample
next, but chronological order alone is not enough. The policy must use a
declared filtration: a null p-value needs conditional super-uniformity, while
an e-factor needs conditional mean at most one. The suite therefore
implements finite LORD++, SAFFRON, and e-LOND traces while labeling their
simulation summaries as calibration diagnostics unless the assumptions are
verified for the scenario.

The broader sequential-testing literature also includes alpha-investing and
selection procedures that make the error budget evolve with prior discoveries
[@foster2008alpha_investing; @gsell2016sequential]. Bandit-testing work connects
online false discovery control to adaptive allocation across arms
[@yang2017bandit_testing]. These references clarify why target selection,
arrival order, and procedure-specific alpha accounting belong to the testing setting.
LORD++ and SAFFRON use alpha-wealth; e-LOND instead records the cumulative
nominal levels it has emitted. Its e-value decision rule is
$E_t\ge 1/\alpha_t$, and recycled levels can make the cumulative trace exceed
$\alpha$, so it is neither alpha-wealth nor a remaining budget. The suite
implements a controlled binary target environment and records those contracts;
its finite policy traces do not establish validity for arbitrary adaptive
allocation or an unexamined arrival process.

Adaptive target selection is also adjacent to selective inference, where the
selection event is part of the inferential contract [@fithian2014selective]. A
target chosen because it looked promising is not automatically a target with a
super-uniform confirmatory p-value. The suite separates three cases: independent
split-stream confirmation, chronological same-stream selection as a diagnostic,
and post-hoc future-evidence minimum-p selection as an invalid contrast. This
is a distinction between information design and statistical validity, not a
terminological preference.

## Active inference as a generative-model formalism

Active inference is a family of generative-model formulations in which
perception, learning, and action are expressed through variational inference
and policy evaluation. The process-theory account derives belief-propagation
dynamics from a Markov decision-process generative model
[@friston2017active_process]. The learning account emphasizes the interaction
of epistemic information-seeking and pragmatic preference-driven behavior
[@friston2016active_learning]. Generalised free energy extends the formal
vocabulary for inference over future states and policies
[@parr2019generalised]. Da Costa et al. provide a mathematical synthesis for
discrete state-space models, which is the closest conceptual neighbor of the
binary `pymdp` construction used here [@dacosta2020discrete]. Smith, Friston,
and Whyte provide a step-by-step POMDP-oriented tutorial for this modelling
context [@smith2022active]; it is cited for exposition, not as evidence that
the binary environment here represents the broader active-inference field.

The present package instantiates only a small, auditable slice of that space:
the reference agent has a binary hidden-state factor, a binary observation
modality, an identity transition, configured preferences, and a fixed-horizon
calibration path. The action-loop extension adds environment-level control,
but it remains a controlled synthetic process rather than a claim to represent
all active-inference architectures or applications. The real `pymdp` agent and
closed-form oracle establish semantic agreement for the declared likelihood
and transition objects; they do not establish that every active-inference
implementation has the same statistical behavior. The software context is
documented by the `pymdp` library paper [@heins2022pymdp].

## From belief states to executable evidence interfaces

A posterior belief answers a model-relative question: how much probability the
agent assigns to a hidden state under its prior, likelihood, and observations.
A p-value answers a different question: how surprising an evidence statistic
would be under a null distribution. An e-value is different again: it is a
nonnegative evidence object whose expectation is controlled under the null.
The bridge in this project is therefore not an identification of these
quantities. It is the explicit map

\begin{equation}
  \text{observation stream}
  \longrightarrow n_{\mathrm{high}}
  \longrightarrow
  \Pr_{H_0}\{\operatorname{Binomial}(T,1-r)\ge n_{\mathrm{high}}\}
  \longrightarrow \text{BH or another family procedure}.
\label{eq:evidence-interface}
\end{equation}

The null Binomial model is exact for the configured channel and fixed horizon.
It becomes a valid testing interface only because the null law is stated and
the evidence statistic is calibrated under that law. This separation makes the
formalism-to-code connection inspectable without treating posterior probability
as FDR, FWER, p-value, or e-value control.

The same boundary applies to power. In the suite, power is conditional on the
agent's generative model, the underlying generative process, the testing
setting, and the action policy. Action-dependent sensing, target selection,
environment transitions, costs, and stopping therefore produce scenario-indexed
operating characteristics. They do not establish a universal power value for
Active Inference as a field.

## Sequential validity and time-uniform evidence

Test martingales connect nonnegative evidence processes to Bayes factors and
p-values, including under stopping rules [@shafer2011test_martingales]. Safe
testing develops e-values whose type-I guarantees are designed for optional
continuation [@grunwald2024safe]. Time-uniform concentration and confidence
sequences provide a complementary way to state uncertainty over a continuing
filtration [@howard2020time_uniform; @howard2021confidence_sequences].

The package implements a binary likelihood-ratio e-process, its Ville-bound
crossing diagnostic, and a Bernoulli confidence-sequence inversion. The
reference evidence p-value remains a fixed-horizon Binomial upper tail, and
posterior or fixed-p-value stopping are deliberately invalid contrasts. The
e-process guarantee is consequently limited to its declared Bernoulli null,
predictable update rule, and filtration; it is not a general active-inference
or sequential multiple-testing theorem.

## Reproducible research objects and formal-to-code traceability

The suite treats a method claim as a chain: **formal object → public symbol →
test → artifact → evidence class → non-claim.**

This design follows the research-object emphasis on findability,
accessibility, interoperability, and reuse [@wilkinson2016fair], but adds a
statistical discipline: provenance cannot make an uncalibrated evidence object
valid. The machine-readable registry, claim ledger, scenario hashes, figure
metadata, generated captions, manuscript variables, and release finalizer make
that boundary inspectable.

The simulation workflow also follows reporting principles for simulation
studies: identify the estimand, data-generating mechanism, method parameters,
replication unit, uncertainty calculation, and interpretation before reading
the result [@morris2019simulation]. Reproducible-computation guidance motivates
the separation of deterministic scientific artifacts from operational telemetry
and the preservation of executable provenance [@sandve2013ten_rules]. These
practices improve inspection and reuse; they do not add statistical information
to a generated sample.

| Layer | Established reference frame | Implemented contribution here | Explicit boundary |
|---|---|---|---|
| Design and power | Design-based operating characteristics | Scenario-indexed power and cost/stopping estimands | No invariant agent-level power |
| Family error control | FWER/FDR procedures and dependence conditions | Tested correction dispatcher, invariants, simulations, and certificates | No new multiple-testing procedure |
| Generative inference | Active-inference process and discrete-state formulations | Real `pymdp` binary agent with closed-form oracle | No general active-inference benchmark |
| Evidence calibration | Null-distribution p-values; martingale/e-value alternatives | Fixed-horizon Binomial p-value, Bernoulli e-process, and confidence sequence | Each object retains its own assumptions |
| Adaptive selection | Selective inference and online-FDR filtration | Split confirmation plus chronological and future-selection contrasts | No post-selection theorem without conditional calibration |
| Policy evaluation | Held-out simulation and adaptive design | Frozen learned policy, paired common-random-number contrasts, cost frontier | No optimality, causal, or utility theorem |
| Provenance | FAIR research objects | Hash-linked results, figures, claims, manuscript, and render | Provenance does not strengthen validity |

The boundary map in @fig:scholarship-boundary makes these distinctions visible.
Each row connects a literature family to the concrete object implemented in the
package, the artifact that can be inspected, and the inference that remains out
of scope. Its purpose is scholarly provenance and claim discipline, not the
display of an additional effect estimate.

![{{FIGURE_SCHOLARSHIP_BOUNDARY_CAPTION}}](../output/figures/scholarship_boundary.png){#fig:scholarship-boundary alt="{{FIGURE_SCHOLARSHIP_BOUNDARY_ALT}}" width=100%}
