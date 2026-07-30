# Forking guide

1. Copy the project into a new independent working directory.
2. Update package metadata, `manuscript/config.yaml`, citations, and claim
   ledger before changing result prose.
3. Keep the `src/` / `scripts/` boundary and replace tests with real domain
   cases.
4. Run the quality gate and regenerate every output artifact.
5. Treat exact publication rendering as a separately provisioned maintainer
   concern; a fork must not claim access to this project's controlled renderer
   or copy its unpublished infrastructure.

Do not retain claims, figures, or stale metadata from this project after changing
the data-generating process.
