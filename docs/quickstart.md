# Quickstart

From this project directory, prepare the locked development environment once.
The `dev` extra supplies the test and analysis tools; add `--extra demo` only
when you plan to launch the optional Streamlit application.

```bash
env -u VIRTUAL_ENV uv sync --locked --extra dev
```

## Safe execution smoke

Use a fresh temporary root for a fast execution smoke. This avoids replacing a
retained release or campaign receipt and makes the non-promotional scope
visible in the command itself:

```bash
quickstart_parent="$(mktemp -d "${TMPDIR:-/tmp}/active_inference_power_fast.XXXXXX")"
quickstart_root="$quickstart_parent/results"
./.venv/bin/python scripts/generate_results.py --profile fast --workers 1 \
  --checkpoint-every 500 --output-root "$quickstart_root"
./.venv/bin/python scripts/audit_figure_accessibility.py --output-root "$quickstart_root"
./.venv/bin/python scripts/smoke_demo.py --payload "$quickstart_root/demo/payload.json"
```

The resulting root is a real generated bundle but not a release candidate. It
does not carry release-profile evidence, quality/hydration/render receipts, or
authority to run `finalize_release.py`.

## Isolated release candidate

For a release-reviewable bundle, use one new candidate root throughout the
configured full analysis and serial evidence chain:

```bash
release_root="$PWD/output/release-candidate-$(date +%Y%m%d%H%M%S)"
test ! -e "$release_root"
./.venv/bin/python -m compileall -q src scripts tests examples
./.venv/bin/ruff check src scripts tests examples
./.venv/bin/python -m pytest tests/ --cov=active_inference_power --cov-branch --cov-fail-under=95
env -u VIRTUAL_ENV uv run mypy src scripts tests examples --ignore-missing-imports
./.venv/bin/python scripts/generate_results.py --profile release --workers 4 \
  --checkpoint-every 500 --output-root "$release_root"
# The quality suite creates its own real session-scoped artifact bundles.
./.venv/bin/python scripts/generate_quality_report.py --output-root "$release_root"
# Strict hydration binds the resolved manuscript to the passing quality receipt.
./.venv/bin/python scripts/z_generate_manuscript_variables.py --output-root "$release_root"
./.venv/bin/python scripts/audit_manuscript_presentation.py \
  --variables-file "$release_root/data/manuscript_variables.json" \
  --resolved-manuscript-dir "$release_root/manuscript"
./.venv/bin/python scripts/audit_figure_accessibility.py --output-root "$release_root"
./.venv/bin/python scripts/review_release.py --output-root "$release_root" \
  --skip-render --skip-clean-install
```

Null calibration is
checkpointed in deterministic 500-replication blocks; the remaining stages
write safe stage-complete resume boundaries. In the candidate root, publication
PNGs are under `figures/`, the dashboard is `review/dashboard.html`, and the
machine-readable evidence is `results.json` and `certificates.json`.

For the frozen expanded sensitivity campaign, the preregistered configuration
requires the exact `output/campaign-v2` root. `output/campaign/` and
`output/precision/` are preserved fail-closed predecessor evidence and must
never be passed to result generation or finalization. Do not rerun this
historical lane as a routine quickstart, and do not use `--resume` to reopen a
completed or retained campaign receipt. A deliberate, authorized
source-contract refresh must use the [guarded archive/preflight
sequence](storey_certificate_decision.md#frozen-v2-lane): it confirms the
literal frozen namespace and declared predecessor receipt locations before
archiving the retained receipt, then regenerates and finalizes only the
configured canonical root. Do not select an arbitrary replacement root or
reuse a retained campaign directory.

An optional quality snapshot belongs **after** campaign result generation and
**before** `finalize_campaign.py` only when an additional reproducibility
record is wanted. It is not a campaign-finalizer input and must never run in
place of the archive/preflight checks, before campaign results exist, or as a
release-quality substitution.

The campaign expands the effect, dependence, and action-reliability grids and
uses the larger replication budgets and exact certificate policy declared by
the v2 configuration witness. Its explicitly declared computational-complexity
plan is deterministic source-work accounting shared with the release input,
not a timing benchmark or additional statistical result. It is a conditional
operating-characteristic campaign; it does not replace the validated release
anchor or broaden formal validity claims. The finalizer is fail-closed for
every declared terminal certificate. A same-vector Storey diagnostic failure
remains serialized with its verdict and policy membership, but it does not by
itself block this isolated v2 split-stream campaign; it never becomes release
evidence or a general FDR guarantee. `finalize_campaign.py` validates the campaign artifacts
directly; it does not require the optional quality snapshot, manuscript
hydration, PDF/HTML rendering, or `review_release.py`. See [the Storey decision
record](storey_certificate_decision.md).
Use `--resume` only for a known interrupted regeneration after the documented
archive/preflight sequence has made the frozen canonical root available and its
checkpoints match the declared contract. It is not a way to reuse a completed
v2 receipt or a preserved failed campaign or precision bundle.

`--output-root` accepts a project-relative or absolute directory for isolated
release-review artifacts. Pass the same root to result generation, manuscript
hydration, and release finalization; `generate_quality_report.py` can also
write its report there. Its full pytest run is still anchored at this checkout,
but generated-artifact tests construct real session-scoped temporary bundles
rather than reading an ignored canonical `output/` fixture. The v2 campaign
predecessor receipts under `output/campaign/` and `output/precision/` retain
their own namespace and hash rules for campaign finalization. Checkpoints
include the configuration and claim-ledger witnesses, execution-source
fingerprint, and seed schedule, and `--resume` rejects artifacts with a
mismatched contract. That release chain is distinct from the
campaign-finalization contract above.

The candidate root must be new. Do not substitute the direct `output/` path or
the retained `output/campaign/`, `output/precision/`, or
`output/campaign-v2/` namespaces for `release_root`.

For a fast-profile bundle, use the read-only
`audit_fast_bundle.py` after its passing quality receipt and strict manuscript
hydration; that audit deliberately keeps the manifest incomplete. For a
release or publication candidate, `review_release.py --skip-render
--skip-clean-install` is the read-only pre-render review, and the unskipped
review is part of the release gate. After the controlled publication renderer
has produced both the PDF and HTML
outputs for this exact root, synchronize the registered alternatives, audit the
rendered HTML, capture provenance, then run unskipped review and finalization:

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

It requires rendered artifacts and is the only command
that can mark a bundle complete. Follow the reversible renderer binding,
rendered-alt synchronization, unskipped review, and PDF/HTML inspection steps
in [rendering_pipeline.md](rendering_pipeline.md); do not render or finalize a
campaign root. Keep `release_root` in the same shell, or set it again to the
same immutable candidate path before the finalizer command.

Exact PDF/HTML typesetting is performed only in the controlled release
environment, not required for public source-level reproduction. Inspect the
immutable PDF and accessible HTML attached to the GitHub and Zenodo releases;
see [controlled_renderer.md](controlled_renderer.md) for the boundary and
[rendering_pipeline.md](rendering_pipeline.md) for the maintainer runbook.

## Optional local complexity timing diagnostic

The release reports deterministic source-work accounting, not wall-clock
performance. To inspect the frozen timing plan on one machine, write a new
isolated directory **outside** `output/`:

```bash
timing_root="../active_inference_power_diagnostics/complexity-timing-$(date -u +%Y%m%dT%H%M%SZ)"
./.venv/bin/python scripts/run_complexity_experiments.py \
  --output-root "$timing_root"
```

The command writes raw repeated wall/process-time samples, a local figure, an
HTML summary, and a hash-bound manifest. It is deliberately not a release,
campaign, manuscript, or statistical-evidence input: its timing values apply
only to the recorded host/runtime and must not be read as a cross-machine
benchmark, simulated resource cost $C$, e-process evidence $E_t$, or online
accounting. In particular, it is neither LORD++/SAFFRON alpha-wealth
$W_t^\alpha$ nor e-LOND cumulative nominal-level allocation.

## Action-in-the-loop suite

An investigator evaluates a design made of three hashable scenario layers—the
agent's generative model, the underlying synthetic generative process, and the
testing setting—plus a separately recorded action policy and replication plan.
The reference scenario lets the policy select targets and sensors, change
latent context, incur cost, and stop sampling. The agent sees only its visible
trace; evaluator-only truth is used after the trace to estimate mean-TPR power
and other operating characteristics. Keep the example outputs in a new
temporary root so they cannot overwrite retained evidence:

```bash
examples_parent="$(mktemp -d "${TMPDIR:-/tmp}/active_inference_power_examples.XXXXXX")"
examples_root="$examples_parent/results"
./.venv/bin/python examples/01_reference_power.py --profile fast --output-root "$examples_root"
./.venv/bin/python examples/02_environment_control.py --profile fast --output-root "$examples_root"
./.venv/bin/python examples/03_adaptive_target_selection.py --profile fast --output-root "$examples_root"
```

Install the optional demo dependency and launch the local review application:

```bash
env -u VIRTUAL_ENV uv sync --locked --extra dev --extra demo
./.venv/bin/streamlit run scripts/active_inference_power_demo.py
```

The visible trace contains only actions and observations available to the
agent. Hidden states are retained separately for operating-characteristic
evaluation and are never policy inputs. The download is canonical JSON with
configuration, full-trace, visible-trace, and payload hashes. Validate the
committed headless fixture without Streamlit or a browser:

```bash
./.venv/bin/python scripts/smoke_demo.py
```

See [release review](release_review.md),
[figure accessibility](figure_accessibility.md), and
[troubleshooting](troubleshooting.md) for the corresponding diagnostics.
