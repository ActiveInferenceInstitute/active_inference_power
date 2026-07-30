# Action-in-the-loop examples

These examples are deterministic, offline, and use the real pinned `pymdp`
agent through the public `active_inference_power` API. They demonstrate that
an investigator evaluates a design with a hashable generative model, generative
process, and testing setting plus a separately recorded policy; power is not an
intrinsic property of the embedded agent.

The default destination, `output/examples/`, is developer scratch space. Use a
fresh explicit root when retaining an example run for comparison or review; an
example output is never a release, campaign, manuscript, or publication input.

```bash
example_parent="$(mktemp -d "${TMPDIR:-/tmp}/active_inference_power_examples.XXXXXX")"
example_root="$example_parent/results"
./.venv/bin/python examples/01_reference_power.py --profile fast --output-root "$example_root"
./.venv/bin/python examples/02_environment_control.py --profile fast --output-root "$example_root"
./.venv/bin/python examples/03_adaptive_target_selection.py --profile fast --output-root "$example_root"
```

Use `--profile release --workers 4` for larger local runs. Each example is one
deterministic completed-run calculation, not an incrementally resumable
producer. Add `--write-completed-checkpoint` only to save a finished snapshot
that can later be replayed with `--replay-completed-checkpoint`; neither option
recovers partial work. Outputs are written below `output/examples/` (or
`--output-root`) and are disposable real result artifacts, not hash-bound
release or campaign certificates.

The traces include both an evaluation view with hidden state labels and a
visible view containing only the actions and observations available to the
agent. The adaptive example also emits the independent split-stream
confirmation result for its context-control scenario; that example output is a
conditional diagnostic, not the suite's separately declared static,
hash-bound formal split-confirmation channel. Target-specific e-process
crossings in the action loop have only the declared per-target
optional-stopping role, make no FWER/FDR claim, and are distinct from the
diagnostic all-action likelihood-ratio product. Online-FDR and e-process
summaries remain conditional on their declared assumptions: LORD++/SAFFRON's procedural alpha-wealth and
e-LOND's cumulative nominal-level allocation trace are accounting quantities,
not e-process evidence or simulated cost. e-LOND emits nominal levels
$\alpha_t$ with the rule $E_t\ge 1/\alpha_t$; recycled levels can make the
cumulative trace exceed $\alpha$, so it is neither alpha-wealth nor a
remaining budget. A posterior decision alone is not family-level evidence.
Cost comparisons are scenario-local: compare $C$ or power-per-cost only when
the declared `cost_unit` and `cost_scale_id` match.

For the full producer/review chain, see the project
[quickstart](../docs/quickstart.md) and the
[action-in-the-loop contract](../docs/action_in_the_loop.md).
