# Package contract — `active_inference_power`

`src/active_inference_power/__init__.py` is the public API. `import
active_inference_power` must work from the project root and from a clean wheel
install; every exported callable must be typed and documented.

## Global invariants

- p-values are one-dimensional finite arrays in `[0, 1]`;
- correction adjusted p-values are aligned to input order and decisions are
  exactly `adjusted <= alpha`;
- FDP is `V / max(R, 1)`, TPR is `S / max(m1, 1)`, and FWER is `V >= 1`;
- stochastic functions use an explicit NumPy generator or recorded seed;
- worker count cannot change seeded synthetic simulation results;
- result samples and aggregate values are finite and in their valid ranges;
- the pinned `pymdp` agent is the real dependency, not a test double;
- `src/` contains no template `infrastructure.*` imports.
- conditional power results identify their generative model, generative
  process, testing setting, policy, and scenario hash; `PowerScenario` itself
  hashes model/process/setting, while the policy is a separate result-row key;
- hidden evaluation states are never passed to action-selection policies;
- the pinned agent dependency is constructed only through the package adapter;
- action-loop evidence and adaptive target-selection results carry an explicit
  validity boundary and cannot silently replace fixed-family BH results.
- raw evidence fields are semantically disjoint: likelihood-ratio sequential
  evidence serializes as `e_process_value`, `e_process_values`, and
  `log_e_process_values` with `evidence_kind: likelihood_ratio_e_process`;
  online-testing bookkeeping serializes as `accounting_path` with either
  `procedural_alpha_wealth` or `cumulative_nominal_level_allocation`. The
  latter is e-LOND's cumulative emitted nominal levels and can exceed $\alpha$
  after recycled levels; it is neither a remaining budget nor wealth. New
  artifact mappings never label e-process evidence as `wealth`.
- action-loop summaries retain replication-level true/false rejections,
  target-specific rates, cost/stopping intervals, and paired policy contrasts;
  reported action-loop power is mean replication-level TPR, while any-true
  detection is separately named; power-per-cost is descriptive within a declared scenario and is not a utility
  guarantee.
- every declared formal object has an executable entry in
  `formalism.formalism_registry()` naming its public symbols, source and test
  paths, assumptions, evidence class, artifact, and non-claim.
- split-stream selection records visible filtration events, independent stream
  seeds, and selection/confirmation hashes. Its declared
  `split_confirmation_scenario` is a self-hashed static channel outside the
  operating/frontier grid. It is separate from the static e-process
  optional-stopping channel: the split-confirmation contract, not the e-process
  trace, carries the formal conditional selection label.
- learned policy artifacts record train/evaluation seed partitions, frozen
  parameters, visible features, policy hash, search-configuration hash,
  candidate-space size/dimension, and a non-optimality boundary.
- time-varying process schedules carry an explicit `schedule_policy`: `hold_last`
  (the final declared value persists), `cycle` (periodic repetition), or
  `strict` (out-of-range access fails). A schedule never silently wraps.
- the release computational-complexity block contains only deterministic,
  version-specific source-work and trace-accounting fields; elapsed-time fields
  are rejected so host telemetry cannot become release evidence. Its keys,
  proxy formulae, dimensions, and action ledger are closed and recomputed from
  the current result contract before a strict release reader accepts it.

## Scientific boundaries

BH control is asserted analytically under the assumptions of its theory and
empirically checked in the declared independent/positive-factor regimes. BY is
the arbitrary-dependence procedure in the comparison, but the project does not
claim to prove that property by simulation. Storey/adaptive results are
finite-sample operating characteristics, not universal dominance claims.

## Artifact contract

`scripts/generate_results.py` emits schema version `2.0` with `meta`,
`experiment`, `certificates`, `method_comparison`, `power_curve`,
`dependence_sweep`, `active_inference`, `paired_comparisons`, `sequential`,
`online_fdr`, `agent_sampling`, `benchmarks`, `null_pvalue_calibration`, and
`sample_size_accounting`, and `computational_complexity` blocks. Release runs
additionally include `suite` and `action_loop` blocks with scenario hashes,
action-loop summaries, and online target-selection diagnostics, paired policy
contrasts, and a power-cost frontier, plus the validated `formalism` registry
block. The computational block is a source-version-specific feasibility ledger,
not a timing benchmark: it separates $Q_{\mathrm{comp}}$ from simulated cost
$C$, e-process evidence $E_t$, LORD++/SAFFRON alpha-wealth $W_t^\alpha$, and
e-LOND cumulative nominal-level allocation. The figure generator consumes that
same dictionary. `scripts/generate_quality_report.py`
emits the measured quality snapshot, while `output/run_log.json` records live
operational stage accounting and failure state. `output/manifest.json` hashes
the complete artifact set, including `output/formalism_registry.json`.
The suite and action-loop blocks carry additive contract version `1.0`; the
split-confirmation, e-value, confidence-sequence, learned-policy, and
benchmark payloads remain separately labeled so a richer process cannot be
mistaken for the formal reference lane.
Strict manuscript hydration requires both result and quality artifacts.
The claim ledger is validated before generation and its SHA-256 digest is
carried in results, checkpoints, and the manifest. Stage-complete checkpoints
also retain the SeedSequence schedule and scenario/model/process/setting
dependency hashes; the partial checkpoint contract additionally records a
total replication count, chunk cursor, and serializable metric accumulators.
The null-calibration stage uses this contract to resume deterministic
child-stream blocks without recomputing completed replications. Other stages
remain stage-complete boundaries because their current aggregate values do not
yet expose an associative scientific reducer. Resume rejects a changed
configuration, claim ledger, dependency hash, or replication design. An output root may be outside the checkout;
manifests namespace generated files under `root: output` while source
provenance remains under `root: project`.

`scripts/run_complexity_experiments.py` is deliberately not a release stage.
It requires a new explicit root and emits an isolated local timing profile with
raw repeated $\tau_{\mathrm{wall}}$ and process-time samples, its plan, and a
source witness. The profile carries a canonical plan digest separately from
the frozen configuration digest, and snapshots the transitive local source
closure before measurement, then rejects a source or config change observed
after measurement. Revalidating a frozen profile also requires an exact match
to the current source closure, so a self-hashed receipt cannot omit or
substitute a local dependency. That profile is host-specific diagnostic evidence only: it
cannot be a certificate input, a release-finalization input, or a comparison of
hardware-independent performance.

## Fast smoke (not a release)

```bash
./.venv/bin/python -c "import active_inference_power"
uv build --out-dir /tmp/active_inference_power_build
./.venv/bin/python -m pytest tests/ --cov=active_inference_power --cov-branch --cov-fail-under=95
env -u VIRTUAL_ENV uv run mypy src scripts tests examples --ignore-missing-imports
smoke_parent="$(mktemp -d "${TMPDIR:-/tmp}/active_inference_power_fast.XXXXXX")"
smoke_root="$smoke_parent/results"
./.venv/bin/python scripts/generate_results.py --profile fast --output-root "$smoke_root"
```

The fast profile is an isolated execution smoke only. It does not carry the
release/publication profile, quality/hydration/render evidence, or an authority
to run `finalize_release.py`.

## Release candidate recipe

For a source-bound release candidate, use one new output root throughout. A
release finalizer accepts only the `release` or `publication` profile and only
after the renderer and the unskipped final review have succeeded. The quality
report may be written to that root, but its full test run remains anchored at
the repository source tree: generated-artifact tests create real
session-scoped temporary bundles instead of reading the canonical `output/`
tree. An external candidate root remains a supported artifact namespace; the
campaign and precision predecessor receipts retain their separate
campaign-finalizer provenance rules. Do not use `output/`, `output/campaign/`,
`output/precision/`, or `output/campaign-v2/` as this candidate root: each is a
default or retained namespace rather than a fresh release bundle.

```bash
release_root="$PWD/output/release-candidate-$(date +%Y%m%d%H%M%S)"
test ! -e "$release_root"
./.venv/bin/python scripts/generate_results.py \
  --profile release --checkpoint-every 500 --output-root "$release_root"
# Quality tests create their own real session-scoped generated bundles.
./.venv/bin/python scripts/generate_quality_report.py --output-root "$release_root"
# Strict hydration starts only after the passing quality receipt exists.
./.venv/bin/python scripts/z_generate_manuscript_variables.py --output-root "$release_root"
./.venv/bin/python scripts/audit_manuscript_presentation.py \
  --variables-file "$release_root/data/manuscript_variables.json" \
  --resolved-manuscript-dir "$release_root/manuscript"
./.venv/bin/python scripts/audit_figure_accessibility.py --output-root "$release_root"
./.venv/bin/python scripts/review_release.py --output-root "$release_root" \
  --skip-render --skip-clean-install
```

Render that same root through the reversible binding procedure in
[`rendering_pipeline.md`](rendering_pipeline.md). After the sibling render,
sync registered HTML alternatives, audit rendered alternatives, and capture
renderer provenance before the unskipped review and finalizer:

```bash
./.venv/bin/python scripts/sync_rendered_figure_alts.py \
  --output-root "$release_root" --rendered-html "$release_root/web/index.html" \
  --include-auxiliary-pages
./.venv/bin/python scripts/audit_figure_accessibility.py \
  --output-root "$release_root" --rendered-html "$release_root/web/index.html" \
  --include-auxiliary-pages
./.venv/bin/python scripts/capture_renderer_provenance.py \
  --output-root "$release_root"
./.venv/bin/python scripts/review_release.py --output-root "$release_root" \
  --report "$release_root/review/release_review_final.json"
./.venv/bin/python scripts/finalize_release.py --output-root "$release_root"
```
