# Controlled release renderer

This repository is self-contained for installing the package, running the
deterministic simulations and tests, generating figures, and hydrating the
manuscript. The exact publication PDF/HTML renderer is a separately
provisioned, controlled release environment; it is not a package dependency or
a public checkout requirement.

## What public users can reproduce

Use the commands in [quickstart.md](quickstart.md) to create a fresh isolated
bundle, run the quality gate, generate result-bound figures, and hydrate the
manuscript. Those steps reproduce the scientific inputs and the traceability
artifacts from this repository. The released PDF and accessible HTML are
provided as immutable GitHub and Zenodo release assets for direct inspection.

## What the release-maintainer environment adds

Before a release is finalized, maintainers run the hydrated manuscript through
the controlled renderer, synchronize the authored HTML alternative text, and
record `renderer_provenance.json`. That receipt binds the rendered PDF/HTML
bytes to the renderer source closure, toolchain identities, and exact
maintainer environment state used for that release. It is a provenance record,
not a claim that an unaccompanied local checkout can reproduce the identical
typesetting.

This separation is intentional. It keeps the public research package usable
without an inaccessible rendering dependency, while making the publication
surface and its local-renderer boundary explicit. For release maintainers, the
complete candidate sequence remains in
[rendering_pipeline.md](rendering_pipeline.md); it is a controlled-release
runbook rather than a public installation guide.
