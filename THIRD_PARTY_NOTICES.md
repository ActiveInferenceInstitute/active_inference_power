# Third-party notices

The project source is made available under the repository's [MIT License](LICENSE).
That grant does not change the provenance, rights, or license context of a
third-party source or fixture. This notice records the small offline fixtures
included for deterministic stress tests; it is not a statement that the
project owns, relicenses, or extends the terms of an upstream data source.

`data/benchmarks/manifest.yaml` is the authoritative, checksum-pinned record
for each fixture's source URL, source revision, retrieval date, preprocessing,
declared license, checksum, and claim boundary. Verify it before redistribution
or replacement. The fixtures are external stress inputs only, not biological
ground truth or evidence of general statistical validity.

## Bioconductor golubEsets material

The compact `data/benchmarks/golub_normalized_fixture.csv` is a normalized
numeric projection recorded as deriving from the Bioconductor
[`golubEsets`](https://bioconductor.org/packages/release/data/experiment/manuals/golubEsets/man/golubEsets.pdf)
source archive `golubEsets_1.54.0.tar.gz`, version `golubEsets_1.54.0`. The
manifest declares `LGPL`,
the source DOI `10.18129/B9.bioc.golubEsets`, its preprocessing boundary, and
the fixture SHA-256
`4e0cbd265402016cb9e9aa75f865da1f3e6aebc5f068b7c2bd813f21250ab789`.
Consult the original package, its license, and its data-provenance information
before using or redistributing this projection. This repository does not
include the full source archive and makes no biological claim from the compact
fixture.

## pymdp-associated adapter fixture

`data/benchmarks/public_agent_trace_adapter_stage_2.csv` is a project-local,
chronological demonstration trace used only as a sequential stress fixture.
Its manifest record associates the adapter context with
[`pymdp`](https://github.com/infer-actively/pymdp) release
`inferactively-pymdp==1.0.3`, whose declared license there is `MIT`, and pins
the CSV SHA-256
`11de9b1dbbdaabf66e5e6ab579287db633deaaeb8f74aa5ee9d5799439b18f8c`.
The CSV is not a claim about an upstream empirical dataset, a test of the
upstream package, or a replacement for its own license and attribution.
