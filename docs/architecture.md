# Architecture

## First-principles contract

The project exists to produce inspectable statistical decisions and, where a
named contract applies, calibrated statistical evidence.
The irreducible requirements are mathematical correctness, reproducible random
streams, explicit assumptions, and a traceable artifact chain. File layout and
renderer choice are implementation choices; they must not change those facts.
The diagram calls the selected generated-artifact directory
`<bundle-root>`: it is `output/` only for ordinary local work, while a release
candidate uses one new explicit root throughout its evidence chain.

```text
manuscript/config.yaml
        |
scripts/generate_results.py --output-root <bundle-root> ----> src/
        |                                                    |
        +--> <bundle-root>/results.json <--------------------+
        +--> <bundle-root>/figures/  (src/active_inference_power/figures)
        +--> <bundle-root>/review/dashboard.html
        |
scripts/generate_quality_report.py --> <bundle-root>/quality.json
        |
scripts/z_generate_manuscript_variables.py --> <bundle-root>/manuscript/
        |
controlled release renderer --> PDF / HTML
```

Four bounded lanes deliberately sit outside that release chain:

```text
scripts/run_research_atlas.py --output-root <new isolated directory>
        |
        +--> research_atlas.json + figures + dashboard.json + manifest.json
        |       (finite research diagnostics; no release-complete status)

scripts/run_benchmark_preflight.py --output-root <explicit directory>
        |
        +--> benchmark_preflight/provenance_dashboard.{json,html} + manifest.json
                (offline fixture provenance; external-stress evidence only)

scripts/run_storey_method_review.py --output-root <new isolated directory>
        |
        +--> storey_method_review.json + manifest.json
                (frozen three-way method review; no promotion or release state)

scripts/run_complexity_experiments.py --output-root <new isolated directory>
        |
        +--> complexity_experiments.json + complexity_experiments.png + index.html + manifest.json
                (host-specific timing diagnostic; no promotion or release state)
```

None of these lanes changes a release certificate, supplies a manuscript result, or
promotes an external fixture into a synthetic validity claim.

Conditional power is evaluated as an investigator-declared design:

```text
investigator fixes model M_A + process P_N + setting T
                                  |
                    hashable PowerScenario s = (M_A, P_N, T)
                                  + separate policy pi
                                  |
       repeated action / observation / belief / evidence traces (agent sees F_t only)
                                  |
          evaluator-only truth scoring -> mean TPR (power), FDR, FWER, cost,
                                      stopping, and MCSE/validity boundary
```

The process is a declared synthetic task environment. It may be called a
simulated niche informally, but the architecture does not model or average over
a real-world niche distribution. See [evaluation perspective](evaluation_perspective.md).

Formal-to-code traceability is a parallel release edge:

```text
formalism registry --> public symbols --> real tests --> results/artifacts
         |                  |                |                  |
         +------------ claim class, assumptions, and non-claim --------+
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `src/active_inference_power/_types.py` | Validated result/config data contracts |
| `src/active_inference_power/corrections.py` | Multiple-testing procedures and dispatcher |
| `src/active_inference_power/error_rates.py` | Realized error decomposition |
| `src/active_inference_power/power.py` | Analytic z/t and BH asymptotic power |
| `src/active_inference_power/theory.py` | Closed-form bounds and structural checks |
| `src/active_inference_power/simulation.py` | Two-groups data-generating process and Monte Carlo |
| `src/active_inference_power/calibration.py` | Independent global-null p-value calibration audit |
| `src/active_inference_power/uncertainty.py` | Bounded Monte-Carlo intervals and replication budgets |
| `src/active_inference_power/active_inference.py` | Real pymdp agent and evidence calibration |
| `src/active_inference_power/pymdp_adapter.py` | Pinned pymdp dependency boundary and numerical configuration |
| `src/active_inference_power/certificates.py` | Seeded bound checks |
| `src/active_inference_power/certificate_gate.py` | Exact terminal/diagnostic certificate-policy partition and fail-closed reduction |
| `src/active_inference_power/dependence.py` | PSD-validated dependence regimes |
| `src/active_inference_power/sequential.py` | Fixed-horizon and time-uniform evidence |
| `src/active_inference_power/online_fdr.py` | LORD++, SAFFRON, and e-LOND schedules with auditable accounting |
| `src/active_inference_power/computational_complexity.py` | Version-specific deterministic source-work proxies, action-trace accounting, and isolated local-timing profile validation |
| `src/active_inference_power/agent_sampling.py` | Policy-controlled stopping traces |
| `src/active_inference_power/scenario.py` | Generative model/process/setting contracts and scenario hashes |
| `src/active_inference_power/action_loop.py` | Environment-level action-in-the-loop traces and operating characteristics |
| `src/active_inference_power/suite.py` | Scenario/policy surfaces and adaptive target-selection summaries |
| `src/active_inference_power/selection.py` | Visible-filtration selection and independent confirmation contracts |
| `src/active_inference_power/testing_setting_sensitivity.py` | Factorial testing-setting surfaces with fixed-family, posterior-action, and e-process interfaces kept separate |
| `src/active_inference_power/selection_sensitivity.py` | Stratified split-stream all-null calibration diagnostics with fixed tuning and common random numbers |
| `src/active_inference_power/dynamic_process_sensitivity.py` | Paired static-versus-dynamic process stress controls and calibration summaries |
| `src/active_inference_power/sequential_atlas.py` | Worker-invariant finite comparison of e-process families, confidence sequences, e-LOND, and an invalid peeking contrast |
| `src/active_inference_power/repeated_policy_evaluation.py` | Repeated held-out train/freeze/evaluate policy partitions and partition-level accounting |
| `src/active_inference_power/multi_target_environment.py` | Multi-target arrival environment, visible filtration, and future-information isolation contract |
| `src/active_inference_power/learned_policy.py` | Deterministic train/freeze/evaluate policy artifacts |
| `src/active_inference_power/demo.py` | Optional Streamlit payload and interactive review application |
| `src/active_inference_power/benchmarks.py` | Offline benchmark validation and normalized Golub fixture loading |
| `src/active_inference_power/benchmark_preflight.py` | Deterministic, offline validation of benchmark manifests, staged fixtures, and provenance fields |
| `src/active_inference_power/protocols/` | Offline, checksum-pinned external statistical protocols; finite stress evidence only |
| `src/active_inference_power/schema.py` | Generated artifact contracts |
| `src/active_inference_power/figures/` | Data-driven publication plots and dashboard |
| `src/active_inference_power/formalism.py` | Versioned formalism-to-code registry and manuscript traceability rows |
| `src/active_inference_power/manuscript_variables.py` | Strict result/quality-to-token mapping |
| `scripts/` | Configuration, subprocess, file orchestration, isolated research-atlas, Storey-review, and local-timing generation, and offline benchmark-preflight receipt generation |

`import active_inference_power` must work from the project root and from a
clean wheel install. Domain modules use package-relative imports; rendering
and file orchestration remain in `scripts/` and the controlled release
environment.

## Evidence boundary

The model, process, and setting are not interchangeable. A calibrated agent
under one model/process pair does not establish power under another process,
and a posterior threshold does not establish family-level error control. The
suite therefore reports scenario hashes and validity labels alongside every
action-loop result. Environment-level action traces are formal only when the
declared conditional evidence assumptions hold; otherwise they are diagnostic
stress evidence.

Computational accounting is deliberately outside the statistical evidence
chain. The release block names $Q_{\mathrm{comp}}$, a dimensionless source-work
proxy for named loops and declared input shapes. Scenario-defined simulated cost
$C$, likelihood-ratio e-process evidence $E_t$, and online alpha-wealth
$W_t^\alpha$ retain distinct meanings; e-LOND cumulative nominal-level
allocation is a further distinct quantity, not alpha-wealth or a remaining
budget. The isolated runner may record local $\tau_{\mathrm{wall}}$, but
host-specific elapsed time neither changes a release result nor supplies a
statistical or cross-machine performance claim.
