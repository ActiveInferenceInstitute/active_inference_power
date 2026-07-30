# Reproducibility {#sec:reproducibility}

Reproducibility is treated here as a scientific control: a reader must be able
to identify the scenario, rerun the declared computation, distinguish
deterministic evidence from operational telemetry, and see when a release gate
has failed. The suite therefore binds code, seeds, results, figures, captions,
manuscript variables, claims, and rendered outputs without treating hashes as
statistical evidence.

## Release identity

The cover cites the persistent concept DOI
[`{{PUBLICATION_CONCEPT_DOI}}`](https://doi.org/{{PUBLICATION_CONCEPT_DOI}}).
The immutable v1 archive is
[`{{PUBLICATION_VERSION_DOI}}`]({{PUBLICATION_VERSION_RECORD}}), and the exact
source release is
[`{{PUBLICATION_GITHUB_RELEASE_TAG}}`]({{PUBLICATION_GITHUB_RELEASE_URL}}) in
[`{{PUBLICATION_GITHUB_REPOSITORY}}`]({{PUBLICATION_REPOSITORY_URL}}). These
links identify the released implementation and archived version; they do not
broaden the finite, scenario-indexed statistical claims made below.

## Verification chain

1. The canonical package imports as `active_inference_power` without test-only
   path mutation, including from an isolated wheel installation.
2. Corrections validate p-values and alpha, and BH q-values are cross-checked
   against SciPy on real vectors.
3. The active-inference posterior is checked against its closed-form likelihood
   ratio and its evidence p-value is derived from the null Binomial law.
4. Seeded simulations produce comparison, paired, power, dependence, policy,
   sequential, online-FDR, benchmark, action-loop, and calibration blocks in
   the selected bundle's `results.json`.
5. `scripts/generate_quality_report.py` measures the test count, line coverage,
   and {{QUALITY_BRANCH_GATE}}\% branch gate; it writes quality evidence to the
   selected bundle.
6. `scripts/z_generate_manuscript_variables.py` fails closed when results or
   quality evidence is missing, then writes the resolved manuscript tree.
7. The renderer consumes the resolved tree; the terminal release finalizer then
   confirms the hashed result, figure, dashboard, benchmark, checkpoint,
   quality, claim, manuscript, Portable Document Format (PDF), and HyperText
   Markup Language (HTML) artifacts.
8. The scholarship map and citation-integrity tests keep literature roles,
   implementation status, evidence artifacts, and validity boundaries linked;
   this follows the research-object emphasis on findable, accessible,
   interoperable, and reusable outputs [@wilkinson2016fair].
9. The result artifact carries configuration hash
   `{{RESULT_CONFIGURATION_HASH}}` and claim-ledger hash
   `{{RESULT_CLAIM_LEDGER_HASH}}`; strict hydration rejects source drift before
   it writes renderer input, and the manifest binds both witnesses to the run.
   The complete manuscript, variable, registry, and figure input tree is bound
   by render-input hash `{{RESULT_RENDER_INPUT_HASH}}`, which terminal review
   requires to appear in each reader-facing rendered format.
10. The conditional-power suite hashes each generative model, generative
    process, and testing-setting scenario; every action-loop operating row
    carries that scenario hash together with its separately recorded policy.
    Action-loop hidden states are retained only in the evaluation trace, while
    the demo consumes a separate deterministic payload.
11. The release result carries a deterministic computational-complexity ledger
    with its declared input plan and source-work boundary. The optional local
    timing runner writes a separate hash-bound diagnostic bundle whose frozen
    profile is checked against the exact current transitive source closure;
    its elapsed time samples are excluded from release results, certificates,
    and publication status.

The selected generated bundle is `{{RESULT_RELEASE_STATUS}}`: the run ledger
reports `{{RESULT_RUN_STATUS}}` and the certificate gate reports
`{{RESULT_CERTIFICATE_GATE}}`. If a gate fails, the structured run log records
the exception and the manifest remains incomplete. This is deliberate: a
review artifact can be useful for diagnosing a method, but it is not silently
reclassified as a passed release.
The rendered status is an evidence label fixed during hydration. Only the
post-render manifest together with its current passing review receipt can
declare terminal publication completion, so a reader-facing token never races
the fail-closed finalization step.

The measured quality snapshot records {{QUALITY_TEST_COUNT}} real no-mock tests,
{{QUALITY_BRANCH_COVERAGE}}\% branch coverage, and
{{QUALITY_LINE_COVERAGE}}\% line coverage. The line gate is
{{QUALITY_LINE_GATE}}\% and the branch gate is {{QUALITY_BRANCH_GATE}}\%; counts
and percentages are generated, not asserted in prose by hand.

The generated result schema also records a null p-value calibration audit,
Student-$t$ Monte-Carlo precision summaries, conservative Hoeffding planning
radii, and the selected bundle's `run_log.json` with stage durations and seeds.
These records make sample-size and execution accounting inspectable without
treating runtime logging as scientific evidence. Scientific artifacts exclude
timestamps and worker timing from their content hashes wherever possible, so
operational variation cannot create unexplained numerical differences.

The computational ledger makes a different kind of execution evidence
reproducible. The configuration declares its input grids, named correction and
online procedures, action policies, warmup policy, and repetition policy;
$Q_{\mathrm{comp}}$ is then regenerated deterministically from that declared
input shape and closed source-work contract. It records correction-vector and
stream-horizon proxies, named action-loop trace counts, and separate
retained-draw accounting after the scientific result blocks are constructed.
It adds neither a simulation stage nor a statistical estimand.

By contrast, the isolated timing profile builds inputs before timing, excludes
configured warmups, and retains repeated local $\tau_{\mathrm{wall}}$ and
process-time samples, a canonical plan digest alongside the frozen
configuration digest, runtime context, a transitive local source-closure
witness, and a self-digest. It rejects a source or frozen-plan change observed
across the measurement interval, and frozen-profile review requires the exact
current closure rather than a self-consistent subset. That profile can be
audited or repeated on the same host, but it is not merged into a selected
bundle's `results.json`, a manifest completion decision, or a statistical claim.
Simulated resource cost $C$, e-process evidence $E_t$, and online alpha-wealth
$W_t^\alpha$ retain their separate meanings; e-LOND's cumulative nominal-level
allocation trace $\sum_{i\le t}\alpha_i$ is a fourth, separate accounting
quantity, not alpha-wealth or a remaining budget.

## Reproduction commands

The following is a **development-only** smoke sequence. It writes a new,
disposable bundle rather than mutating the canonical `output/` tree, and it is
not a source-bound release-candidate recipe:

```bash
repro_parent="$(mktemp -d "${TMPDIR:-/tmp}/active_inference_power_repro.XXXXXX")"
repro_root="$repro_parent/results"
uv run pytest tests/ --cov=active_inference_power --cov-branch --cov-fail-under={{QUALITY_BRANCH_GATE}}
uv run python scripts/generate_results.py --profile fast --output-root "$repro_root"
uv run python scripts/generate_quality_report.py --output-root "$repro_root"
uv run python scripts/z_generate_manuscript_variables.py --output-root "$repro_root"
```

Any abbreviated producer command that omits `--output-root` mutates the
canonical `output/` development tree. Such commands are development-only and
cannot create, repair, or identify a release candidate.

For a source-bound candidate, use the complete explicit-root sequence in
`docs/rendering_pipeline.md`: results, quality, strict hydration, audits,
rendering, rendered-alt synchronization, renderer provenance, unskipped review,
and finalization all bind one candidate root. Do not substitute this
development-only root, `output/`, `output/campaign/`, `output/precision/`, or
`output/campaign-v2/` for a fresh candidate root. For a larger development
campaign, use an explicit profile, seed, worker count, checkpoint interval, and
output root. Resume accepts a checkpoint only when the scenario, configuration,
schema, and seed-schedule hashes match; this prevents a partial run from being
mistaken for a different experiment.

The publication PDF/HTML is produced in a separately provisioned controlled
release environment, rather than by a renderer bundled with this package. The
public source reproduces results, figures, and the hydrated manuscript; the
immutable PDF and accessible HTML are shipped with the GitHub and Zenodo
release. The renderer-provenance receipt binds the final bytes to that
controlled environment. Inspect both the Portable Document Format (PDF) and
HyperText Markup Language (HTML) outputs: figures, tables, captions, alt text,
dashboard links, and manuscript variables are part of the release, not
post-processing decoration.

## Honest limitations

The following map separates what a released artifact checks from what a reader
may infer from it.

| Component | Release evidence | Not established |
|---|---|---|
| Certificate gate | Configured finite Monte-Carlo band checks. | A formal proof. |
| Adaptive FDR | Estimate and uncertainty under the declared model; a larger run can expose finite-sample behavior different from BH. | Universal adaptive-FDR control. |
| Dependence and theorem protocol | Bounded PRDS-style and stress simulations; an offline checksum-pinned adapter permits controlled reruns. The external outward-rounded Arb certificate remains the theorem artifact. | Arbitrary-dependence control or a replacement for the external theorem. |
| Active-inference model | Binary observation channel, static truth, and fixed-horizon action-loop contract. | A general agent evaluation. |
| Sequential and online path | E-process interpretation under the declared Bernoulli null, predictable filtration, and update rule; finite online-FDR diagnostics. | Validity of posterior or fixed-horizon p-value stopping, or universal sequential-FDR control. |
| Computational workload | Deterministic, source-version-specific $Q_{\mathrm{comp}}$ accounting for named code paths and declared inputs; separately hash-bound local timing diagnostics. | Processor-instruction counts, peak memory, cross-hardware performance, a release gate, or statistical evidence. |
| Benchmark fixture | Offline transport and stress workflow checks. | Ground truth for synthetic claims or implicit external retrieval. |

### Claim-admission checklist

A stronger label requires a declared estimand, primary scholarship, code
contract, real-value test, uncertainty calculation, inspectable artifact, and
explicit non-claim. A favorable finite result can motivate the next scenario or
theorem check; it cannot silently promote itself to one.
