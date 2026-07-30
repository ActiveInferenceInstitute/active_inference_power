# Testing philosophy

The suite uses real NumPy arrays, SciPy distributions, seeded RNGs, actual
files, and the pinned `pymdp` agent. Mocks are prohibited because they
would test interaction shape rather than mathematical output.

The gate is:

```bash
./.venv/bin/python -m pytest tests/ --cov=active_inference_power --cov-branch --cov-fail-under=95
env -u VIRTUAL_ENV uv run mypy src scripts tests examples --ignore-missing-imports
```

Coverage must be accompanied by a non-zero collection count and a successful
quality snapshot. Add property tests for invariants (permutation equivariance,
monotonicity, bounded probabilities), hand-computed cases, invalid-input
branches, and at least one independent oracle where available.

When a test touches generated output, use a real temporary directory and the
actual generator. Do not weaken a test because a certificate is noisy; record
the Monte-Carlo standard-error band in the certificate instead.
