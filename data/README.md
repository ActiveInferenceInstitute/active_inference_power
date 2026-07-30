# Data and claim contracts

This directory holds authored inputs to the evidence chain, not generated
results. Its files serve different roles:

- `claim_ledger.yaml` is a provenance index for manuscript-level claims. It is
  not a second numeric source of truth. Generated values must come from the
  relevant result, certificate, or quality artifact and point back to the
  source function/configuration; literature-grounded claims use bibliography
  keys instead.
- `research_lanes/` contains frozen, separately scoped execution contracts.
  Changing a lane is a new source input and requires a new validated run; it
  does not retroactively change a retained campaign receipt.
- `benchmarks/` contains checksum-pinned manifest metadata and small offline
  fixtures. See its [README](benchmarks/README.md) for the external-evidence
  boundary.

The ledger records what supports a claim and what the claim does *not* show.
It must preserve distinctions between theory, finite simulation, calibration
diagnostics, stress tests, and provenance checks. See the
[claim audit](../docs/claim_audit.md) and
[output conventions](../docs/output_conventions.md) for the corresponding
artifact contracts.
