# Active Inference Power Suite

`active_inference_power` is a reproducible, investigator-facing simulation
suite for statistical power in adaptive studies that embed an active-inference
agent. Before simulation, the investigator declares the agent-side generative
model, evaluator-side generative process, testing setting, policy, and
replication plan. The agent then acts only on its visible history; the evaluator
retains simulated hidden truth solely to score traces after the fact. Power is
therefore a conditional operating characteristic of the declared design—not a
universal property of an agent or of its task environment.

The fixed-horizon method and calibration lanes compare declared observation
streams, evidence objects, and correction procedures; they do not by themselves
instantiate an action-controlled agent--task-environment loop. The action-loop
lane adds an embedded agent that acts under uncertainty within a fixed synthetic
process, while the investigator estimates operating characteristics across
replications. In neither lane is power a capability score for the agent.

The action-loop process is a synthetic task environment. “Simulated niche” is
only informal shorthand for that declared process; it does not denote an
empirical ecological, social, or deployment population, and the scenario grid
is not a distribution over such niches. See the
[evaluation perspective](docs/evaluation_perspective.md) for the role boundary
and the exact action-loop estimand.

The suite includes a tested family of correction procedures, replication-level
FDP, true-positive-rate, and any-false-rejection-event scoring aggregated as
FDR, power, and FWER, analytic z/t power, the Genovese--Wasserman BH fixed
point, seeded dependence-aware simulations, real `inferactively-pymdp==1.0.3`
inference, and action-in-the-loop environments with adaptive sensing, target
selection, costs, latent context transitions, and stopping.

The project’s central rule is simple: a posterior belief threshold is not a
family-level error guarantee. Under the declared null, the observed stream is
mapped to a calibrated Binomial p-value before BH is applied in the fixed-horizon reference;
the action-loop and sequential tracks use separately declared evidence
contracts and validity labels.

In the action loop, a target-specific e-process crossing can support only the
declared **per-target** optional-stopping interpretation. It is not an FWER or
FDR guarantee for the target family; the all-action likelihood-ratio product is
a diagnostic display, and finite observed FDR/FWER values are estimands rather
than control claims.

## Quick start

Prepare the locked development environment once. The `dev` extra supplies the
test, coverage, lint, and type-check tools used below; the optional `demo`
extra is needed only for the Streamlit application.

```bash
env -u VIRTUAL_ENV uv sync --locked --extra dev
```

Start with a fresh, disposable fast-profile bundle rather than overwriting a
retained result or campaign receipt:

```bash
quickstart_parent="$(mktemp -d "${TMPDIR:-/tmp}/active_inference_power_fast.XXXXXX")"
quickstart_root="$quickstart_parent/results"
./.venv/bin/python scripts/generate_results.py --profile fast --workers 1 \
  --checkpoint-every 500 --output-root "$quickstart_root"
./.venv/bin/python scripts/audit_figure_accessibility.py --output-root "$quickstart_root"
./.venv/bin/python scripts/smoke_demo.py --payload "$quickstart_root/demo/payload.json"
```

This smoke bundle exercises the real project contracts but is not a release or
publication candidate. For a full isolated candidate and the PDF/HTML review
sequence, use the [quickstart](docs/quickstart.md) and
[rendering pipeline](docs/rendering_pipeline.md).

For the stronger non-release fast-bundle check used by continuous integration,
keep the same fresh root and add a passing quality receipt and strict
hydration before running the read-only integrity audit:

```bash
./.venv/bin/python scripts/generate_quality_report.py \
  --output-root "$quickstart_root"
./.venv/bin/python scripts/z_generate_manuscript_variables.py \
  --output-root "$quickstart_root"
./.venv/bin/python scripts/audit_manuscript_presentation.py \
  --variables-file "$quickstart_root/data/manuscript_variables.json" \
  --resolved-manuscript-dir "$quickstart_root/manuscript"
./.venv/bin/python scripts/audit_fast_bundle.py --output-root "$quickstart_root"
```

This check verifies the fast profile, current-source witnesses, result-derived
files, figure registry, quality receipt, and manifest bytes. It neither renders
nor marks a bundle complete, so it remains distinct from a release candidate.

Validate source changes with:

```bash
./.venv/bin/python -m compileall -q src scripts tests examples
./.venv/bin/ruff check src scripts tests examples
./.venv/bin/python -m pytest tests/ --cov=active_inference_power --cov-branch --cov-fail-under=95
env -u VIRTUAL_ENV uv run mypy src scripts tests examples --ignore-missing-imports
```

At the selected output root, analysis writes `results.json`, `certificates.json`,
the configured publication PNGs plus `figures/figure_registry.json`,
`review/dashboard.html`, and `manifest.json`.
The registry carries the caption/alternative-text pair plus reader question,
visible comparison, takeaway, visual encoding, and validity metadata; a
mixed-status multi-panel figure also records panel-specific validity so an
exception cannot be hidden by a whole-figure label.
It also writes `formalism_registry.json`, which connects formal objects
to public symbols, tests, assumptions, evidence classes, and non-claims.
It also writes `run_log.json` with stage durations, seeds, and
replication accounting. The result schema includes a global-null p-value
calibration audit and conservative Monte-Carlo precision budgets.
Release-derived files under `output/` are generated and should be regenerated
rather than hand-edited. Do not discard the named preserved campaign,
precision, and campaign-v2 receipts: they are hash-bound historical evidence
with their own retention rules, not disposable scratch output.

The manuscript title page uses the source artwork at
`manuscript/assets/active_inference_power_cover.png`. Its provenance and role
are documented beside the asset; it is publication design, not scientific
evidence.

## Navigation

Start with the [quickstart](docs/quickstart.md), then use the
[documentation hub](docs/README.md) for the package contract, architecture,
the investigator/agent/task-environment distinction, action-loop
model/process/setting contract, research roadmap, and failure
boundaries. The [release review guide](docs/release_review.md) distinguishes
read-only verification from rendered-bundle finalization; the
[FAQ](docs/faq.md) explains the scientific scope and retained campaign failure.

## Key package surfaces

This is an orientation map rather than an exhaustive module inventory. The
public exports are defined in
[`src/active_inference_power/__init__.py`](src/active_inference_power/__init__.py);
the formalism-to-code map and
[package contract](docs/PACKAGE_CONTRACT.md) provide the complete audited
symbol, test, and boundary cross-reference.

- `src/active_inference_power/corrections.py`: Bonferroni, Šidák, Holm, Hochberg, BH, BY, Storey,
  a conservative fixed-lambda null-count-bound diagnostic, adaptive BH, and
  fixed-weight weighted BH.
- `src/active_inference_power/error_rates.py`: confusion counts, FDP, TPR, FWER, and aggregation.
- `src/active_inference_power/power.py`: analytic z/t power, sample-size inversion, and BH asymptotics.
- `src/active_inference_power/simulation.py`: two-groups Gaussian model and paired Monte Carlo.
- `src/active_inference_power/active_inference.py`: real pinned `pymdp` sequential testing plus
  calibrated evidence p-values.
- `src/active_inference_power/certificates.py`: seeded numeric certificates with explicit bounds.
- `src/active_inference_power/dependence.py`: independent, negative, block, and factor regimes.
- `src/active_inference_power/sequential.py`: fixed-horizon p-values and time-uniform e-processes.
- `src/active_inference_power/online_fdr.py`: LORD++, SAFFRON, and e-LOND online schedules.
- `src/active_inference_power/computational_complexity.py`: deterministic
  source-work proxies, declared action-trace accounting, and validation for an
  isolated host-specific timing profile.
- `src/active_inference_power/selection.py`: visible-filtration selection traces and independent split-stream confirmation.
- `src/active_inference_power/learned_policy.py`: deterministic train/evaluate policy artifacts with frozen hashes.
- `src/active_inference_power/agent_sampling.py`: real `pymdp` policy/stopping traces.
- `src/active_inference_power/scenario.py`: composable generative-model,
  generative-process, and testing-setting contracts with stable hashes; policies
  are evaluated separately against those contracts.
- `src/active_inference_power/action_loop.py`: action/observation/belief/
  evidence/cost/stopping traces and conditional power summaries.
- `src/active_inference_power/suite.py`: scenario surfaces and adaptive
  target-selection accounting.
- `src/active_inference_power/action_statistics.py`: replication-level
  true/false rejection accounting, mean-TPR power and separately named
  any-true-detection rates, cost/stopping
  intervals, power-per-cost summaries, and paired policy contrasts.
- `examples/`: deterministic offline reference, environment-control, and
  adaptive-target examples.
- `src/active_inference_power/demo.py`: optional Streamlit review application.
- `src/active_inference_power/benchmarks.py`: checksum-verified offline benchmark adapters.
- `src/active_inference_power/figures/`: deterministic publication figures,
  scenario-aware captions, and a lightweight dashboard.
- `src/active_inference_power/formalism.py`: executable formalism-to-code
  traceability registry and generated manuscript rows.

The public API is re-exported from `src/active_inference_power/__init__.py`, so `import active_inference_power` works
without test-specific path manipulation. See
[`docs/PACKAGE_CONTRACT.md`](docs/PACKAGE_CONTRACT.md) for invariants.

## Rendering

The publication PDF/HTML renderer is a separately provisioned controlled
release environment, not a public package dependency. This repository
reproduces results, figures, and the hydrated manuscript; the immutable
release PDF and accessible HTML are distributed as release assets. See
[the controlled-renderer boundary](docs/controlled_renderer.md) and the
maintainer [rendering pipeline](docs/rendering_pipeline.md).

## Publication and citation

Version 1.0.0 is released at
[`ActiveInferenceInstitute/active_inference_power`](https://github.com/ActiveInferenceInstitute/active_inference_power)
and its exact GitHub release is
[`v1.0.0`](https://github.com/ActiveInferenceInstitute/active_inference_power/releases/tag/v1.0.0).
Use the concept DOI [10.5281/zenodo.21695160](https://doi.org/10.5281/zenodo.21695160)
for a citation that follows the release family, and the immutable version DOI
[10.5281/zenodo.21695161](https://doi.org/10.5281/zenodo.21695161) for this
specific v1 archive. The matching Zenodo record is
[21695161](https://zenodo.org/records/21695161). The source-bound release
chain and its deliberately bounded scientific claims are documented in
[`docs/publication_metadata.md`](docs/publication_metadata.md).

## License and third-party material

Project source is licensed under the [MIT License](LICENSE). The compact
offline benchmark fixtures retain their own source provenance and license
contexts; they are not relicensed by the project MIT grant. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the checksum-pinned
[`data/benchmarks/manifest.yaml`](data/benchmarks/manifest.yaml) before
redistributing or extending those fixtures.

## Scope

The Monte-Carlo certificates are evidence under stated assumptions, not formal
proofs. Negative and block dependence are stress regimes; BY is the
arbitrary-dependence comparator, not a simulation proof. Fixed-horizon
posterior thresholds remain distinct from e-process optional-stopping claims.
The fixed-horizon binary model remains the numerical reference baseline. The
action-loop suite is deliberately conditional: environment-level control,
adaptive target selection, and process shifts are stress/operating-characteristic
tracks unless the declared conditional evidence assumptions are satisfied. The
power-cost frontier is a conditional decision-design comparison, not a policy
optimality or utility theorem. The manuscript’s
[`docs/scholarship_map.md`](docs/scholarship_map.md) records how active-inference
process theory, discrete-state inference, multiple-testing theory, and the
sequential martingale/e-value literature map onto implemented methods and
explicit non-claims.
The release's computational-work ledger is a version-specific source-work
accounting aid, not a processor, memory, or timing benchmark. It keeps the
dimensionless $Q_{\mathrm{comp}}$ proxy separate from simulated resource cost
$C$, e-process evidence $E_t$, and online alpha accounting: LORD++ and
SAFFRON use procedural alpha-wealth $W_t^\alpha$, whereas e-LOND serializes
the cumulative nominal test levels $\sum_{i\le t}\alpha_i$. The e-LOND rule
uses $E_t\ge 1/\alpha_t$ at each step; recycled levels can make its cumulative
trace exceed $\alpha$, so it is neither alpha-wealth nor a remaining budget.
Simulated cost $C$ is a scenario-local resource quantity: compare cost or
power-per-cost across scenario cells only when their hash-bound `cost_unit` and
`cost_scale_id` agree. An optional $\tau_{\mathrm{wall}}$ diagnostic is tied to
one host and runtime and does not affect certificates, release status, or
statistical interpretation.
The formalism-to-code method is documented in
[`docs/formalism_to_code.md`](docs/formalism_to_code.md), and the release claim
review is recorded in [`docs/claim_audit.md`](docs/claim_audit.md).

The staged research agenda is documented in
[`docs/research_roadmap.md`](docs/research_roadmap.md). Stage 1—predictable
adaptive target selection has an independent split-stream confirmation path.
The formal conditional split-confirmation result uses a separately declared,
hash-bound static channel outside the operating grid; an action-loop example's
split result remains a context-specific conditional diagnostic. Chronological
selection remains diagnostic; richer process dynamics, learned policies, and
external benchmarks remain separately labeled conditional or stress tracks.

## Expanded campaign lanes

The original canonical campaign profile produced the preserved
`output/campaign/` receipt for a broader effect/dependence/reliability surface.
That all-terminal campaign completed its simulations but failed the
same-vector Storey FDR certificate, so `manifest.complete` remains `false`.
It is retained as a method-sensitivity finding; never reuse that output root
or promote it into the publication manuscript.

The separately frozen v2 configuration at
[`data/research_lanes/conditional_power_campaign_v2.yaml`](data/research_lanes/conditional_power_campaign_v2.yaml)
has a retained canonical receipt at `output/campaign-v2/`. Its campaign-specific
policy keeps BH FDR, Bonferroni FWER, positive dependence, and independent
split-stream Storey confirmation terminal, while retaining the same-vector
Storey power-gain record as a visible diagnostic. The v2 receipt is complete
for that frozen campaign contract only: it neither changes the canonical
release's all-terminal policy nor promotes a release or publication claim. See
[`docs/storey_certificate_decision.md`](docs/storey_certificate_decision.md)
for the exact decision and receipt boundary. It is historical, hash-bound
campaign evidence rather than a source-current release candidate: a
source-contract refresh of this exact frozen lane first archives its prior
canonical receipt, and a genuinely new campaign requires a new reviewed
configuration and a previously absent output namespace.

The package also exposes `storey_qvalue_conservative`. It uses a fixed lambda
and an exact one-sided binomial upper confidence bound for the null count,
preventing the zero-null plug-in degeneracy seen in tiny families. Because the
bound is estimated from the same p-values used for rejection, this is an
assumption-gated diagnostic. An independent split avoids reusing the selection
stream but supplies only the declared conditional calibration evidence, not a
general finite-sample FDR theorem.
