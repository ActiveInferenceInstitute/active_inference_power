# Testing philosophy

The suite uses real NumPy arrays, SciPy distributions, seeded RNGs, actual
files, and the pinned `pymdp` agent. Mocks are prohibited because they
would test interaction shape rather than mathematical output.

The gate is:

```bash
./run.sh test-full
./run.sh types
```

Use `./run.sh test-fast` for the existing `not slow` selection without a
release-coverage claim. See the [gate run guide](RUN_GUIDE.md).

Tests consuming the session-scoped `generated_fast_bundle` fixture (including
indirect fixture dependencies) are marked `slow` during collection. They stay
in the unfiltered full suite; default feedback never generates that bundle.

Coverage must be accompanied by a non-zero collection count and a successful
quality snapshot. Add property tests for invariants (permutation equivariance,
monotonicity, bounded probabilities), hand-computed cases, invalid-input
branches, and at least one independent oracle where available.

When a test touches generated output, use a real temporary directory and the
actual generator. Do not weaken a test because a certificate is noisy; record
the Monte-Carlo standard-error band in the certificate instead.
