# Test guide

Tests exercise real numerical code, real seeded RNG streams, real files, and
the installed `pymdp` dependency. Mocks and synthetic interaction doubles are
not permitted.

- Build generated-artifact tests in `tmp_path` through their actual producer;
  never use `output/` as a writable fixture.
- Cover valid invariants and fail-closed boundaries together: malformed input,
  non-finite values, mismatched hashes, missing canonical fields, and stale
  provenance must be rejected explicitly.
- Keep seeds and expected values deterministic. For parallel paths, assert
  worker-count invariance using the same child-stream schedule.
- When a public serialization changes, test its exact current keys and the
  corresponding consumer or validator; do not retain alias fallbacks.
- For figure-registry changes, test reader semantics and any required
  panel-specific validity metadata, then separately test that rendered HTML
  retains authored alternative text rather than substituting a caption.

Run the focused file first, then:

```bash
./.venv/bin/python -m pytest tests/ --cov=active_inference_power --cov-branch --cov-fail-under=95
```
