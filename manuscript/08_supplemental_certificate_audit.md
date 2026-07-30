# Supplement: Adaptive-FDR certificate audit {#sec:supplemental-certificate-audit}

This supplement gives the complete accounting behind the adaptive-FDR result
in @sec:results-adaptive-audit. It is included to keep the main narrative
readable while preserving the distinction between a power comparison, a
calibration check, and a release gate.

## S1. Audit contract

The audit fixes the declared two-groups model, one-sided p-values, hypothesis
family, nominal alpha, Storey tuning rule, and replication seed schedule. Each
replication applies BH and the same-vector Storey plug-in q-value to the same generated
p-value vector. This common-random-number construction is appropriate for a
paired power contrast, but it does not make Storey FDR valid. FDR is computed
as the mean replication-level FDP, not as a ratio of pooled false and total
discoveries.

The reported certificate has three logically separate conditions: Storey power
must meet the BH reference, BH FDR must remain inside its declared bound, and
Storey FDR must remain inside the nominal target's finite Monte-Carlo band.
Only their conjunction produces a pass. The metric-specific uncertainty is
serialized because the power SE cannot be used as an FDR SE.

## S2. Generated audit table

| Metric | Estimate | MC SE | Finite-band ceiling / reference | Gate |
|---|---:|---:|---:|:---:|
{{RESULT_STOREY_AUDIT_ROWS}}

: Complete adaptive-FDR audit generated from the certificate artifact. The
values are conditional on the current model, process, setting, correction,
seed schedule, and replication profile. {#tbl:supplemental-adaptive-fdr}

The Storey FDR estimate is {{RESULT_STOREY_FDR_ESTIMATE}} and its finite-band
ceiling is {{RESULT_STOREY_FDR_UPPER_BAND}}. The excess over the nominal target
is {{RESULT_STOREY_FDR_EXCESS}}, corresponding to
{{RESULT_STOREY_FDR_Z}} Monte-Carlo SE. The bundle's Storey FDR gate is
{{RESULT_STOREY_FDR_GATE}}, while its power-gain gate is
{{RESULT_STOREY_POWER_GATE}}. These labels describe the declared finite check;
they do not establish or refute a universal theorem.

## S3. Interpretation and diagnostic ablations

The same-vector Storey plug-in estimates the null fraction from the same p-values used
to form the rejection set. A lower estimated null fraction makes the adjusted
q-values more permissive. The current audit therefore motivates, but does not
by itself prove, a same-vector plug-in calibration mechanism as the source of
the observed overrun. The next diagnostic should record the distribution of
the estimated null fraction, its underestimation frequency, and its
replication-level association with FDP.

The method comparison retains three separate lanes. First, the same-vector
Storey plug-in is a transparent sensitivity diagnostic. Second, a conservative fixed-lambda
null-count upper bound is a more cautious estimator-based comparator. Third,
independent split-stream confirmation is the primary conditional adaptive
protocol: selection information estimates the tuning quantity, while fresh
confirmation evidence determines rejection. The latter still requires its
conditional null law and independence contract.

## S4. Reporting rule

The publication surface reports both passed and failed finite checks. A failed
certificate leaves the underlying results, figures, and manuscript variables
available for review but prevents the bundle from being marked complete. This
is a scientific result: higher replication can narrow uncertainty and reveal a
shortfall that a smaller run could not resolve. It is not a reason to increase
the threshold, remove the method, or relabel a diagnostic as formal evidence.

## S5. Action-loop estimands and uncertainty accounting

The same accounting rule applies to the action-in-the-loop results. Let
$Z_{b,k}$ denote the replication-level outcome for policy $k$ in seed block
$b$, where the outcome may be a true-positive rate, FDP, stopping time, cost,
or an indicator of any rejection. The reported operating characteristic is

\begin{equation}
\widehat{\theta}_k=\frac{1}{B}\sum_{b=1}^{B}Z_{b,k},
\qquad
\widehat{\operatorname{SE}}(\widehat{\theta}_k)
=\sqrt{\frac{1}{B(B-1)}\sum_{b=1}^{B}(Z_{b,k}-\widehat{\theta}_k)^2}.
\label{eq:supplemental-mc-estimator}
\end{equation}

For a paired policy contrast, the unit is the seed block rather than an
individual observation. The contrast is computed first,
$D_b=Z_{b,k}-Z_{b,\mathrm{ref}}$, and its uncertainty is estimated from the
empirical variance of $D_b$. This preserves the common-random-number design
without pretending that action-dependent trajectories share the same observed
history. Simultaneous bands apply the declared finite-simulation adjustment
across the prespecified policy-by-metric family. They quantify precision for
the generated scenario; they do not provide a confidence set over policies,
processes, or unobserved environments.

For bounded event rates, Wilson or exact binomial intervals are retained when
the estimand is an event probability. For continuous bounded summaries, the
finite band uses the configured Monte-Carlo standard-error multiplier, while a
Hoeffding radius supplies a distribution-free planning reference. The DKW
calibration radius applies to the pooled independent null p-values only. These
choices keep interval meaning tied to the sampling unit and avoid using a
single generic error bar for every object. The interval conventions are
therefore part of the result schema and figure metadata, not a rendering
choice.

## S6. Filtration, evidence, and action chronology

The selection trace contains only actions and observations available at the
time a target is selected. Hidden truth, confirmation observations, and future
outcomes are evaluation fields and are not admissible policy inputs. This is
the executable version of the filtration condition: a selection rule is
predictable only with respect to the history it is allowed to see. Independent
split confirmation makes the next evidence stream fresh, but it does not
repair a misspecified null or an action-dependent emission law that violates
conditional super-uniformity.

The three adaptive lanes consequently answer different questions. Split
confirmation asks whether a selected target can be tested on fresh evidence
under its conditional null. Chronological same-stream selection asks how the
current policy behaves when discovery and confirmation share a stream. The
future-evidence minimum-p lane demonstrates the failure mode created by using
information that arrives after the selection decision. The latter is included
as a negative control for the filtration audit, not as a candidate method.

This interpretation aligns the implementation with selective-inference and
online-testing scholarship: selection is part of the inferential object, and
online validity depends on the history-conditioned null law rather than on
chronological ordering alone [@fithian2014selective; @javanmard2018online].

## S7. What a larger sample can and cannot resolve

Increasing the number of independent replications narrows Monte-Carlo
uncertainty and can separate a small finite-simulation excess from sampling
noise. It does not change the generative model, process, testing setting,
policy, or theorem assumptions. In particular, a larger run can turn a
previously inconclusive finite gate into a failure without making the
underlying procedure less or more valid in a mathematical sense. The precision
campaign is therefore retained as an auditable review bundle, while the
release manifest remains fail-closed whenever a predeclared certificate does
not pass.

This rule is central to the suite's scholarship. The simulation is an empirical
study of a specified operating characteristic, not a substitute for a proof.
The proof-relevant conditions remain the null distribution, dependence or
conditional-superuniformity contract, filtration, and stopping rule. The
artifact chain supplies reproducibility and inspection; it cannot promote
diagnostic evidence to a formal theorem.

## S8. Reader checklist for adaptive claims

Before interpreting an adaptive result, a reader should identify: (i) the
target population and true-null definition; (ii) the visible filtration at
selection and stopping; (iii) whether confirmation evidence is independent;
(iv) the p-value or e-value null law; (v) the family-level accounting rule;
(vi) the independent replication unit and interval convention; and (vii) the
scenario hash and separately recorded policy that bind the prose to the
generated artifact (a frozen learned policy may additionally carry its own
parameter hash).
If any item is absent, the result should be read as exploratory or diagnostic
until the missing contract is supplied. This checklist is deliberately more
restrictive than a plot-level comparison because adaptive sensing can change
both what is observed and the distribution against which it must be judged.
