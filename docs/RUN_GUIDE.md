# Gate run guide

The repository-root [`run.sh`](../run.sh) delegates to
[`scripts/gate.py`](../scripts/gate.py). That dispatcher is the single source
of gate commands and composition for local work and
[CI](../.github/workflows/ci.yml); this guide documents the public arms,
not a second list of backend tool invocations.

## One locked environment

From the repository root, prepare the development environment once:

```bash
env -u VIRTUAL_ENV uv sync --locked --extra dev
```

The `dev` extra supplies the test, coverage, lint, and type-check tools.
Reuse that project environment across arms; the gate runner does not need a
new dependency synchronization for every check. The optional demo application
and controlled publication renderer are not prerequisites for the default
source checks. GNN conformance has a separately provisioned, read-only upstream
parser checkout, described below.

## Public arms

Run commands from the repository root. Select one leaf for focused work, the
default fast aggregate for ordinary source feedback, or the all aggregate for
the complete source-gate composition.

| Public command | Contract |
|---|---|
| `./run.sh` or `./run.sh fast` | Default aggregate: lint, types, test-fast, docs, in that order |
| `./run.sh lint` | Source compilation and Ruff checks |
| `./run.sh types` | Project type checking |
| `./run.sh test-fast` | Existing `not slow` selector; the full-lane marker also covers consumers of the real generated-bundle fixture; no release coverage claim |
| `./run.sh test-full` | Existing quality-report producer: full pytest with configured 95% line/branch coverage gates, writing coverage and quality evidence under `output/` |
| `./run.sh docs` | Local Markdown-link audit and checked generated gate documentation; checks do not rewrite evidence |
| `./run.sh gnn` | The shared `gnn.importers` reader over the pin, this repository's GNN pointer and every specification, then specification/runtime binding and GNN specification tests against the pinned upstream parser; missing upstream prerequisites fail rather than silently skip conformance |
| `./run.sh package` | Package build and clean wheel-install/import checks |
| `./run.sh all` | Aggregate: lint, types, test-full, docs, gnn, package, in that order |

`test-full` is the full-test arm; `all` is the full source-gate aggregate.
The fast path deliberately omits slow tests, full coverage, upstream GNN
conformance, and packaging. Passing it does not establish a release coverage
receipt. Passing `all` also does not render, finalize, or publish a scientific
bundle: follow the separate [rendering pipeline](rendering_pipeline.md) and
[release review](release_review.md) for that evidence chain. The
[testing philosophy](testing_philosophy.md) defines the real-fixture and
coverage policy.

Collection marks tests whose fixture closure includes `generated_fast_bundle`
as `slow`. This avoids generating the scientific bundle for default feedback,
without removing those tests or changing the unfiltered full quality gate.

The full-test arm delegates to
[`scripts/generate_quality_report.py`](../scripts/generate_quality_report.py)
and writes `output/quality.json` with its coverage evidence. Unlike the docs
check, it is not a read-only audit. The quality receipt alone does not render,
finalize, or publish a bundle.

## Offline pinned GNN conformance

The authoritative pin is [`models/gnn-pin.json`](../models/gnn-pin.json), using
schema `gnn-importer-pin/v1`. It identifies
`ActiveInferenceInstitute/Generalized_Notation_Notation`, tag `v5.1.0`,
version `5.1.0`, and exact commit
`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`. The parser entry point is
`gnn.parsers.markdown_parser:MarkdownGNNParser`; the pin also retains its
recording dates. Do not replace this identity with the current upstream branch.

The pin, this repository's `.github/GNN.md` pointer, and the specifications'
syntax are read by `gnn.importers` — the estate's one reader — which the arm
runs as `python -m gnn.importers check --repo .` against the pinned revision.
This repository carries no pin reader of its own: until `v5.1.0` shipped that
module the pin was `v5.0.3` at `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, which
predates it, and an interim local reader stood in. The local reader was deleted
in the same change that advanced the pin.

Provision the pinned upstream Git checkout and its parser environment before
an offline run. Set `ACTIVE_INFERENCE_POWER_GNN_ROOT` to that checkout's path
to select it explicitly; when unset, the runner uses the sibling
`../Generalized_Notation_Notation` checkout. The GNN gate passes the resolved
root explicitly to its conformance subprocesses so the upstream comparison
cannot be silently skipped. A missing checkout or parser prerequisite is a
failure, not proof of conformance.

The gate does not clone, fetch, install into, or modify the reference checkout.
Do not repair a missing prerequisite by mutating a shared reference repository;
provision a separate checkout/environment outside the gate. CI provisions its
own checkout before calling the same public arm.

The [model inventory](../models/README.md) describes the shipped models;
[`.github/GNN.md`](../.github/GNN.md) is the focused workflow pointer. The exact
CI job key is `gnn-conformance` and its display name is
`gnn conformance (pinned)`.

## CI mapping

The [workflow source](../.github/workflows/ci.yml) remains authoritative for
job dependencies, timeouts, and setup steps.

| CI responsibility | Shared gate mapping |
|---|---|
| Static source and package checks, once in the shared Python 3.14 matrix environment | `./run.sh lint types docs package` |
| Python test matrix, retaining Python 3.11, 3.12, 3.13, and 3.14 | test-full in each matrix cell |
| `gnn-conformance` / `gnn conformance (pinned)` | Provision pinned upstream checkout, then gnn |
| Fast scientific artifacts within the Python 3.14 test lane | Generate results before test-full; reuse its quality receipt for hydration, audits, and examples afterward |

There is no separate fast-artifacts job or duplicate full-test invocation for
that artifact sequence. The Python 3.14 lane retains one output root across
its producer, quality report, hydration, and integrity checks. This remains a
non-release scientific-artifact lane, not the fast source aggregate or a
rendered publication candidate.

The static checks share that same Python 3.14 setup and locked environment;
there is no extra static-job installation. Other matrix runtimes and the
pinned upstream parser keep their separately required environments.

The workflow uses `astral-sh/setup-uv` pinned at
`68559e99cc807c1eb7e9238fd43a15bba460eec2` with uv `0.13.6`.
It does **not** use an organization composite setup action. Project dependency
preparation uses the locked development environment; upstream parser setup is
a distinct CI prerequisite, not an extra synchronization hidden inside every
leaf arm.

## Measured evidence and freshness

Read the generated [gate runtime evidence](_generated/gate-runtimes.md) for
before/after per-arm observations and their provenance. Its companion JSON
is machine-readable evidence, not another hand-maintained table.
[`scripts/measure_gates.py`](../scripts/measure_gates.py) owns generation and
checking: `--phase before|after` selects the measurement phase, `--write`
updates generated documentation from recorded evidence, and `--check`
checks freshness. The docs arm uses check mode.

These bounded observations are host- and environment-specific diagnostics,
not repeated benchmark estimates, scientific results, coverage receipts, or
speed guarantees. Keep all measured durations and comparisons in the generated
page; never hand-copy them into this guide, the hub, or the manuscript. A
configured coverage threshold is a contract, not a measured coverage claim.

For the full document inventory, return to the
[flat index](documentation-index.md) or [documentation hub](README.md).
