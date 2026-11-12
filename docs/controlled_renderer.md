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

## Authoritative render command and provisioning

From the repository root, after result generation, a passing quality report,
strict hydration, and the pre-render audits, set `isolated_root` to that prepared
candidate and run:

```bash
./.venv/bin/python scripts/render_manuscript.py --output-root "$isolated_root"
```

This is a thin Stage 03 wrapper, not another renderer. `--output-root` is
required and must select an existing isolated candidate below the real
`output/` directory, never a retained campaign root. The wrapper reuses the
read-only pre-render release review before binding the candidate. It neither
hydrates inputs nor substitutes a direct Pandoc pipeline. The existing
post-render synchronization, provenance capture, unskipped review, and
finalization remain separate, required steps.

The controlled environment must already provide:

- A compatible checkout at `<project-parent>/template`, which is the exact
  discovery convention in `scripts/renderer_provenance.py`. An explicit
  `--renderer-root` or `AIP_RELEASE_RENDERER_ROOT` selects a root but cannot
  bypass that provenance convention: the selected root must resolve to the
  same checkout. The environment variable being unset is not a missing-tool
  diagnosis.
- The source closure required by `renderer_source_paths`, including Stage 03,
  rendering/orchestration sources, lock/package metadata, and
  `infrastructure/publishing/transmission_bookends.py`. A reference revision
  that instead imports `infrastructure/transmission/transmission_bookends.py`
  does not satisfy this contract merely because it contains a render engine.
  Do not modify that reference, copy its engine here, or fabricate a source
  shim to satisfy provenance.
- That checkout's provisioned, executable `.venv/bin/python` and its installed
  renderer dependencies. The wrapper invokes this interpreter directly with
  bytecode writes disabled, rather than allowing `uv run` to synchronize or
  modify the controlled environment.
- Its already configured
  `projects/working/active_inference_power` symlink resolving to this exact
  repository. The wrapper checks it without creating, replacing, or
  synchronizing links, and passes `TEMPLATE_SKIP_LINK_SYNC=1` to Stage 03.
- Pandoc and XeLaTeX on `PATH`, with the LaTeX packages required by the existing
  renderer. Executable presence is only preflight evidence; Stage 03 must
  still succeed.

If any prerequisite is missing or incompatible, the command exits nonzero
before binding `output/` and names the missing prerequisite. Provisioning is a
release-environment responsibility, not permission to alter a read-only
reference checkout. No output bundle, resolved manuscript, or quality receipt
is fabricated to make rendering appear successful.

Stop other output readers and writers before running the wrapper. Restoration
refuses unexpected paths and preserves the original tree for recovery; see the
[binding and recovery procedure](rendering_pipeline.md#rendering-a-new-candidate-bundle)
before handling a signal, renderer failure, or interrupted host.

This separation is intentional. It keeps the public research package usable
without an inaccessible rendering dependency, while making the publication
surface and its local-renderer boundary explicit. For release maintainers, the
complete candidate sequence remains in
[rendering_pipeline.md](rendering_pipeline.md); it is a controlled-release
runbook rather than a public installation guide.
