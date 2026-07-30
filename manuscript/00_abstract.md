# Abstract {#sec:abstract}

Statistical power is an investigator-facing operating characteristic of an
adaptive-study design. Before simulation, the investigator fixes an
agent-side model, evaluator-side process, testing setting, policy, and
replication plan. Each embedded agent acts only on visible history; simulated
hidden truth is retained for scoring. `active_inference_power` makes
that conditional estimand inspectable. The suite combines fixed-horizon
procedures, analytic references, dependence/calibration experiments, and a
discrete-state active-inference agent checked against a binary oracle. It
extends this to action loops with sensing reliability, latent context, target
choice, cost, and stopping.

The study distinguishes model-relative posterior belief from calibrated p-value
and likelihood-ratio e-process evidence. It compares Benjamini–Hochberg (BH)
false discovery rate (FDR) procedures with family-wise error rate
(FWER) alternatives, and separates either evidence object from online FDR
procedure-specific accounting. Results are scenario-indexed finite-simulation
estimates with Monte Carlo standard error (MCSE) and declared error,
dependence, filtration, and optional-stopping boundaries; they do not assign a
universal power value to an agent, task environment, or active inference.
Instead, they support auditable comparisons among explicitly declared
adaptive-study designs. Contracts, seed schedules, certificates, figures,
claim ledger, and rendered manuscript form a linked evidence chain, allowing
readers to trace each claim to its design and artifact.
{{PUBLICATION_REPOSITORY_STATEMENT}}
