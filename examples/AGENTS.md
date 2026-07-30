# Examples agent guide

Examples are deterministic, offline demonstrations of the public API. Keep
them separate from the release producer: they may write caller-owned example
artifacts, but they must not alter release, campaign, manuscript, or
publication state.

Each example must state the investigator/design interpretation in ordinary
language, keep evaluator-only truth out of the policy-visible trace, and use
the real pinned agent rather than a mock. Preserve explicit `--output-root`,
seed, profile, worker, and checkpoint behavior so a reader can rerun a compact
example without inferring a universal agent or task-environment claim.
If an example exposes online testing, name LORD++/SAFFRON procedural
alpha-wealth separately from e-LOND's cumulative nominal-level allocation
trace. The latter can exceed $\alpha$ after recycled levels and is neither a
remaining budget nor alpha-wealth; keep both accounting quantities separate
from e-process evidence and simulated cost.
For action-loop examples, identify target-specific e-process crossings as
per-target optional-stopping decisions only: they make no family-error-control
claim, and any all-action product is diagnostic. If an example emits split
confirmation, state whether it is a context-specific diagnostic or the
separately declared static, hash-bound formal channel.

After an example behavior or output-contract change, run that example with the
fast profile and its focused real-file tests. Update
[README.md](README.md) when its command, artifact layout, or scope changes.
