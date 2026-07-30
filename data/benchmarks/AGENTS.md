# Benchmark-data agent guide

This directory holds checksum-pinned external-data metadata and compact
offline fixtures. It is not an automatic download cache. Tests, ordinary
result generation, and release review must stay offline; use
`scripts/fetch_benchmarks.py` only as an explicitly authorized acquisition
step for a named manifest record.

Before changing a fixture or manifest entry, record and verify its source,
license, revision, schema, preprocessing, checksum, and claim role. Run the
offline benchmark preflight afterward. A staged archive or fixture is external
stress provenance only unless a separately reviewed study contract says more;
never use it to expand synthetic statistical-validity, biological, causal, or
publication claims.
