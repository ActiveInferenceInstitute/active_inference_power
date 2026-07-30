# FAQ

## Why is the belief threshold not enough?

It sets a model-relative decision rule for one agent under one evidence model;
it is not, by itself, a calibrated p-value or error-rate guarantee. FDR and
FWER are family-level quantities; repeating the decision across hypotheses
changes the error problem. `evidence_pvalue` makes the bridge explicit and BH
supplies the family-level rule.

## What does power mean in this suite?

Power is an investigator-estimated operating characteristic of a declared
design. The hashable `PowerScenario` contains the generative model, generative
process, and testing setting; a separately recorded policy completes the
evaluated design. For action-loop results, reported power is the replication-
mean true-positive rate, while the different “at least one true target” event
is reported separately as `any_true_rejection_rate`. Changing any of those
layers changes the operating characteristic, so the suite reports scenario
surfaces and policy keys rather than one universal agent power number.

## Is this an agent's power in an uncertain niche?

No. The closer description is an investigator simulating an agent under a
declared synthetic task environment, then estimating a design-level operating
characteristic across replications. In an action-loop replicate, the embedded
agent is uncertain about latent state and context and acts only on visible
history. That within-trace uncertainty is not the object called statistical
power. The fixed-horizon method and calibration lanes do not, by themselves,
instantiate that action-controlled agent--environment loop at all.

“Simulated niche” is only informal shorthand for the evaluator-side generative
process; it is not an ecological claim, fitness measure, agent-capability
score, or real-world deployment population. See [evaluation
perspective](evaluation_perspective.md) for the exact roles and estimand.

## Does the scenario grid average over real-world environmental uncertainty?

No. It is a declared sensitivity grid: each cell is a separate synthetic
process/policy comparison. A niche-averaged or deployment-population estimand
would need a justified distribution of environments, weights, and external
validation. The current suite intentionally does not claim one.

## What does the replication count $B$ control?

For a fixed declared design, $B$ controls the Monte-Carlo precision of the
investigator's estimator, such as the replication-mean TPR. It does not change
what the estimand means, grant the embedded agent additional information, or
turn the scenario grid into a population distribution over environments. The
reported MCSE and interval describe finite-simulation precision for that
declared design; they are not uncertainty intervals for an undeclared real
deployment population.

## What do posterior belief, e-process evidence, online alpha accounting, cost, source work, and local time mean?

They are separate quantities with different units and inferential roles. A
posterior belief is the agent's model-relative probability for a hidden state
given its visible history. A likelihood-ratio e-process value $E_t$ is a
nonnegative sequential evidence process whose stopping-time interpretation
depends on its stated null, likelihood, and filtration assumptions. Online
alpha-wealth $W_t^\alpha$ is a LORD++/SAFFRON bookkeeping balance that
determines which future p-value testing levels those procedures may spend or
replenish. e-LOND instead emits nominal levels $\alpha_t$ for e-values, uses
the rejection condition $E_t\ge 1/\alpha_t$, and records the cumulative
nominal-level allocation $\sum_{i\le t}\alpha_i$. Recycled levels can make that
trace exceed $\alpha$, so it is not alpha-wealth or a remaining budget.
Cumulative cost $C$ is the scenario-defined resource consumed by selected
actions.

The deterministic source-work proxy $Q_{\mathrm{comp}}$ instead counts named
current-code passes, scans, and execution paths at a declared input shape. The
ledger reports retained draws separately as implementation-specific retention
accounting. Local elapsed time $\tau_{\mathrm{wall}}$ is a repeated wall-clock
observation recorded only in the isolated timing diagnostic. $Q_{\mathrm{comp}}$
is not elapsed time or a memory measurement, and $\tau_{\mathrm{wall}}$ is not a
cross-hardware benchmark or a release metric.

An e-process value and online accounting are not interchangeable: the former
is sequential evidence, while LORD++/SAFFRON alpha-wealth and e-LOND cumulative
nominal-level allocation are different testing-procedure traces. None is a
posterior, p-value, effect size, expected utility, or resource budget. Use the
surrounding procedure and caption to determine whether a trace displays
e-process value $E_t$, LORD++/SAFFRON alpha-wealth $W_t^\alpha$, or e-LOND's
cumulative nominal levels; do not infer one from another.

Source work and local elapsed time are likewise not interchangeable with the
statistical quantities. $Q_{\mathrm{comp}}$ is neither simulated cost $C$ nor
e-process evidence $E_t$, and $\tau_{\mathrm{wall}}$ cannot certify power,
calibration, or validity. The source-work ledger is release-bound and
deterministic; local timing is host-bound and remains a separate, explicitly
non-promotional diagnostic.

## Do actions receive hidden-state truth?

No. The visible trace contains only actions, observations, beliefs, evidence,
cost, and stopping. Hidden states are retained in a separate evaluation trace
for computing power and error rates. This prevents simulation bookkeeping from
becoming an information leak into the policy.

## Are the certificates proofs?

No. They are seeded Monte-Carlo checks against stated bounds with an explicit
standard-error band. Formal claims come from the cited theory, not from a JSON
Boolean.

## Why can the Storey certificate fail?

The certificate deliberately checks calibration and power separately. The
Storey power gain is a paired operating-characteristic comparison, whereas
Storey FDR is the mean replication-level false-discovery proportion. A power
gain cannot compensate for an FDR excess. The release gate also uses the
FDR-specific Monte-Carlo standard error; the power standard error is not a
valid substitute.

In the declared reference design, the same-vector Storey plug-in q-value estimates the null
fraction from the same p-value vector used for rejection. That plug-in can make
the rejection threshold permissive enough to produce a small finite-simulation
FDR excess. A smaller run may leave that excess inside its wider predeclared
band, while a larger precision run can resolve it as a gate failure. This is
evidence about the declared procedure and data-generating process, not a claim
that replication changes the mathematical theorem.

The response is to preserve the failed bundle, report the separate estimands,
and compare conservative fixed-lambda or independent split-stream confirmation
under their own assumptions. Raising a threshold after seeing the result,
removing the method, or relabeling the diagnostic as formal evidence would
break the preregistered accounting contract.

## Is this online or sequential FDR?

The package now includes finite-horizon LORD++ and SAFFRON traces, a Bernoulli
e-process, and explicit agent-controlled stopping policies. Their validity
labels remain assumption-specific: the e-process uses Ville's bound under its
declared Bernoulli null, while the online-FDR result is a finite calibration
diagnostic under independent/conditionally super-uniform p-values. The
fixed-horizon Binomial p-value remains a deliberately invalid contrast for
optional stopping.

## Why do e-process results include a displayed path and a log path?

E-processes can grow beyond the largest finite floating-point number on a long
stream. Reader-facing text calls the displayed path the e-process value $E_t$
to distinguish it from online accounting. The displayed terminal value can
therefore be capped when necessary, while its log-domain companion retains the
authoritative pathwise representation, including Ville-boundary crossings. This
is numeric protection, not an extra validity claim.

## Why is BY present if the simulation is positive dependence?

To show the power price of a more conservative arbitrary-dependence procedure.
The positive-factor experiment does not establish arbitrary-dependence behavior.

## Why does `import active_inference_power` matter?

The public API should work outside pytest. The package uses one canonical
namespace and relative internal imports, so the same API works from the project
root and from an isolated wheel install.

## What is generated?

Everything under `output/`: results, certificates, quality metrics, figures,
dashboard, manifest, and resolved manuscript. Re-run the scripts after changing
configuration or source; do not edit those files manually.

For a source-bound check that does not alter the bundle, use
[`scripts/review_release.py`](../scripts/review_release.py). It is distinct
from the rendered-bundle finalizer described in
[release review](release_review.md).
