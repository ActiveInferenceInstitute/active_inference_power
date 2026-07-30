# Dobriban/BH all-chunk finite stress receipt

`scripts/run_dobriban_bh_stress_bundle.py` is an isolated offline runner for
the checksum-pinned, publicly documented Dobriban/BH two-sided Gaussian
protocol. It executes all seven declared chunks, with their declared seeds,
strata, batching target, and macro-replication counts. It never downloads
inputs and does not execute the external outward-rounded Arb certificate.

Run it only into a new directory outside this repository's canonical
`output/` tree:

```bash
project_root="$(git rev-parse --show-toplevel)"
cd "$project_root"
bundle_parent="$(mktemp -d "${TMPDIR:-/tmp}/active_inference_power_dobriban.XXXXXX")"
bundle_root="$bundle_parent/bundle"
test ! -e "$bundle_root"
test ! -L "$bundle_root"
./.venv/bin/python scripts/run_dobriban_bh_stress_bundle.py --output-root "$bundle_root"
```

The `mktemp` parent keeps the bundle outside canonical `output/` and makes its
new-root contract explicit. To retain a completed receipt elsewhere, choose a
different caller-owned parent outside `output/`, keep `bundle_root` previously
absent, and pass that exact path to the runner.

The command intentionally has no chunk, seed, replication, or batching
overrides. The full public plan can be computationally substantial; compact
real executions exist only as a programmatic test/smoke contract and are
labelled `offline_protocol_compact_smoke_only_not_for_promotion_or_theorem`.

The completed root contains a self-hashed `dobriban_bh_stress_bundle.json`,
an accessible PNG and figure registry, `dashboard.json`, a static `index.html`,
and a manifest with SHA-256 records for every generated artifact. The runner
checks source and plan witnesses before and after execution, validates every
declared chunk/result pairing, audits PNG accessibility and HTML alternative
text, and publishes atomically only after those checks pass.

These artifacts are finite `stress_test` evidence only. They are not release,
campaign, manuscript, certificate, theorem, arbitrary-dependence, universal
failure-rate, or Active Inference evidence.
