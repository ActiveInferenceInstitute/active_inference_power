# Standalone project guide

This directory can be run independently of the template renderer. Prepare the
locked development environment, run the quality gate, generate results/figures,
and hydrate the manuscript as documented in `README.md` and
`docs/quickstart.md`:

```bash
env -u VIRTUAL_ENV uv sync --locked --extra dev
```

The rendered publication workflow is intentionally external. This repository
reproduces the package, results, figures, and hydrated manuscript without a
renderer checkout; immutable PDF and accessible HTML are release assets. Exact
typesetting is performed only in the separately provisioned controlled
release-rendering environment, whose receipt is recorded with the release. See
[the controlled-renderer boundary](docs/controlled_renderer.md).
