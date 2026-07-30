# External benchmark track

The benchmark manifest records sources, licenses, and checksums. Retrieval is
explicit through `scripts/fetch_benchmarks.py`; tests and ordinary result runs
never download external data.

Treat retrieval as an operator action, not as package setup. Fetch only the
named manifest record into a controlled directory, retain its observed
checksum, and run the offline preflight before drawing any conclusion from a
staged fixture. The preflight itself does not download data and produces a
provenance receipt only. Use a fresh temporary root so this diagnostic cannot
replace a retained result or release artifact. From the project root:

```bash
preflight_parent="$(mktemp -d "${TMPDIR:-/tmp}/active_inference_power_benchmark_preflight.XXXXXX")"
preflight_root="$preflight_parent/results"
./.venv/bin/python scripts/run_benchmark_preflight.py --output-root "$preflight_root"
```

Its `benchmark_preflight/` lane verifies the committed manifest and locally
staged fixture hashes; it does not turn an external fixture into a release,
biological, causal, or synthetic-validity claim.

Downloaded archives are not automatically a reviewed project input. Keep them
out of commits unless their source, license, checksum, preprocessing, and
claim role have been reviewed through the manifest and claim-ledger workflow.

The manifest retains the checksum-pinned Bioconductor archive provenance and a
small committed normalized Golub projection for offline adapter tests. The
projection is useful for feature-level stress comparisons but cannot support a
biological claim about the full ExpressionSet. The staged agent-trace adapter
is intentionally marked as a sequential stress test and cannot contribute to
synthetic validity claims.

The project MIT license does not replace the declared provenance or license of
these fixtures. Read the repository-level
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) together with the
manifest before redistributing a fixture or substituting another source.

See the [research roadmap](../../docs/research_roadmap.md) for the gate before
external-data expansion and the [claim audit](../../docs/claim_audit.md) for
the evidence-boundary convention. See
[output conventions](../../docs/output_conventions.md) for the preflight
artifact layout.
