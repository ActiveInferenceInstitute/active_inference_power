# Action-in-the-loop suite

The reference environment has two targets, binary truth, latent context, two
sensing qualities, action-dependent transitions, unequal costs, and explicit
stopping. It is a controlled synthetic task environment for an investigator's
conditional-power experiment, not a claim that every active-inference agent
has this structure or that the process represents an empirical niche. The
public composition is:

```text
GenerativeModelSpec + GenerativeProcessSpec + SettingSpec
                       = hashable PowerScenario s
separate policy pi + s = evaluated design d
                       -> repeated ActionLoopTrace values
                       -> evaluator-only scoring -> operating-characteristic summary
```

This document describes the action-loop lane only. The fixed-horizon method and
calibration lanes separately simulate declared observation streams, evidence,
and correction procedures; they do not, by themselves, instantiate an
action-controlled agent--task-environment loop. In both cases, power is an
investigator-estimated operating characteristic rather than an agent attribute.

The investigator fixes `s`, `pi`, and the replication plan before simulation.
Within a trace, the agent-side model and the true process are represented
separately. A model-relative posterior under one specified model/process pair
does not, by itself, establish calibrated evidence after an environment shift.
`PowerScenario` hashes model, process,
and setting, including its default stopping rule; each operating row separately
records the executed policy. Frozen learned policies additionally have their
own parameter hash, but an ordinary policy label is not silently presented as a
scenario hash.

Each trace contains a visible view and an evaluation view. The visible view
contains actions, observations, posterior beliefs, p-values or e-process values,
LORD++/SAFFRON alpha-wealth or an e-LOND cumulative nominal-level allocation
trace where an online procedure is present, resource cost, and stopping. The
evaluation view additionally records hidden states and is used
only for power, FDR, and FWER scoring.

For action-loop rows, “power” is the replication mean of the true-positive
rate. The separately emitted `any_true_rejection_rate` answers the different
question of whether at least one true target was rejected. Neither is an
intrinsic property of the agent or synthetic task environment. The scenario
grid is a sensitivity comparison, not a probability distribution over niches.

These labels are deliberately non-interchangeable. A posterior is a
model-relative belief used by the policy. An e-process value $E_t$ is a
sequential evidence object with a stated null and filtration contract. Online
alpha-wealth $W_t^\alpha$ is the balance that an online testing rule uses to
allocate or replenish future LORD++/SAFFRON testing levels. e-LOND instead
emits nominal levels $\alpha_t$ and records their cumulative allocation
$\sum_{i\le t}\alpha_i$; its e-value decision rule is $E_t\ge 1/\alpha_t$.
Recycled levels can make that trace exceed $\alpha$, so it is neither a
remaining budget nor alpha-wealth. Resource cost $C$ is the scenario-local
simulation-cost amount consumed by chosen actions. Its
hash-bound `cost_unit` and `cost_scale_id` identify the scale; costs or
power-per-cost values are comparable across scenarios only when both match.
Serialized trace
values are always read with their procedure label: they distinguish an
e-process path from online alpha accounting, and neither denotes resource cost.

Policies include fixed horizon, posterior threshold, expected-information-gain,
cost-aware information gain per unit cost, posterior sampling, adaptive target
selection, and e-process stopping. The e-process label is
optional-stopping-valid only when fixed truth, predictable selection, conditional
e-factor mean at most one, nonnegative e-process construction, model/process
likelihood agreement, and state-invariant Bernoulli rates within each
target/truth cell are explicitly declared and checked. Conditional
super-uniformity remains the p-value requirement, not the e-process one. The
context-dependent reference environment therefore remains
diagnostic. The static e-process optional-stopping contrast and the separate
static split-confirmation channel have different evidence contracts; neither
borrows the other's formal label. Posterior and fixed-horizon p-value decisions
remain diagnostic under adaptive stopping.

`SettingSpec.stopping_policy` is the executed policy when a caller does not
provide an explicit policy. A supplied different policy is recorded as a
diagnostic `policy_override`, with both declared and executed policies in the
serialized trace; it cannot receive the formal e-process label. Action-loop
decisions do not apply a p-value family correction to action-level evidence, so
the only valid action-loop setting is `correction: none`. Fixed-family and
online-FDR corrections are separate executable analyses, not action-loop
metadata.

Action-loop summaries report replication-level true and false rejection counts,
target-specific rejection rates, finite-simulation intervals for power, FDR,
cost, and stopping, the mean of replication-level power-per-unit-cost
($\mathrm{TPR}/C$) scores with zero assigned to a zero-cost trace, and
common-random-number policy contrasts. This descriptive mean is not a ratio of
scenario-level summaries. These estimands make the power-cost trade-off
auditable rather than compressing a policy into one decision rate.

The release also runs an explicit all-null selected-target calibration
diagnostic. It records the empirical CDF excess of target-selected p-values and
compares it with a nominal iid DKW radius at the declared confidence level,
both per target and pooled. The radius is not a serial-dependence correction;
it is a transparent reference calculation for the declared finite simulation.
This is the entry measurement for the Stage 1 roadmap: a diagnostic pass is
evidence about the declared process and policy composition, not a
selective-inference proof and not permission to generalize online-FDR validity
to other action policies or environment dynamics.

The release pairs that chronological diagnostic with an explicit failure case.
The future-evidence contrast selects the minimum p-value after observing the
complete fixed-horizon trace. Its selected-p-value excess is reported as a
stress-test contrast, not as a method: because selection uses information from
the future relative to the testing time, it violates the visible-history
filtration and cannot support online-FDR or selective-inference validity.

The formal promotion rule is intentionally asymmetric: a passing simulation
can support a conditional operating-characteristic statement, but only a
verified filtration and conditional-null contract can support a formal validity
label. This prevents increasingly realistic environments from being mistaken
for increasingly strong theorems.

The examples under `examples/` and the optional Streamlit app use the same
public API. All simulations are offline and deterministic under their recorded
seed schedules.
