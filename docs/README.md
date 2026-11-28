# Documentation hub

**Quick navigation:** [flat documentation index](documentation-index.md) |
[gate run guide](RUN_GUIDE.md) |
[pinned GNN conformance](../.github/GNN.md) |
[generated gate runtime evidence](_generated/gate-runtimes.md).

## Choose a reading path

- **First local run:** prepare the locked environment and make a disposable
  smoke bundle in [quickstart.md](guides/quickstart.md).
- **Source checks:** use the shared local/CI interface in the
  [gate run guide](RUN_GUIDE.md); its fast path is not a release coverage claim.
- **Methods reader:** start with the
  [evaluation perspective](research/evaluation_perspective.md), then the
  [action-in-the-loop contract](architecture/action_in_the_loop.md),
  [formalism-to-code map](architecture/formalism_to_code.md), and
  [visualization notes](publication/visualization_notes.md).
- **Contributor:** read the operating
  [agent instructions](guides/agent_instructions.md),
  [architecture](architecture/architecture.md), and
  [package contract](publication/PACKAGE_CONTRACT.md) before changing code or contracts;
  use the [script inventory](../scripts/README.md) to distinguish the
  non-release fast-bundle audit from the release chain.
- **Release reviewer:** follow the serial
  [rendering pipeline](publication/rendering_pipeline.md) and distinguish its read-only
  [release review](publication/release_review.md) from the state-changing finalizer.
- **External or campaign evidence:** read the
  [research roadmap](research/research_roadmap.md),
  [Storey decision record](research/storey_certificate_decision.md), and benchmark
  documentation before treating a diagnostic lane as evidence.

| Document | Purpose |
|---|---|
| [documentation-index.md](documentation-index.md) | Flat inventory of long-lived documentation, manuscript sources, and directory guides |
| [RUN_GUIDE.md](RUN_GUIDE.md) | Composable local and CI gates, locked environment, and evidence boundaries |
| [_generated/gate-runtimes.md](_generated/gate-runtimes.md) | Generated, host-specific before/after gate measurements |
| [../.github/GNN.md](../.github/GNN.md) | Pinned GNN workflow job and explicit offline checkout contract |
| [agent_instructions.md](guides/agent_instructions.md) | Project operating contract |
| [PACKAGE_CONTRACT.md](publication/PACKAGE_CONTRACT.md) | API and mathematical invariants |
| [architecture.md](architecture/architecture.md) | Source → artifact dependency graph |
| [evaluation_perspective.md](research/evaluation_perspective.md) | Investigator, simulated-agent, and task-environment role boundary |
| [testing_philosophy.md](validation/testing_philosophy.md) | No-mock and coverage policy |
| [action_in_the_loop.md](architecture/action_in_the_loop.md) | Model/process/setting and policy execution contracts |
| [../models/README.md](../models/README.md) | GNN specifications for the declared generative models and the upstream custody pin |
| [formalism_to_code.md](architecture/formalism_to_code.md) | Executable formalism and source-symbol traceability |
| [claim_audit.md](validation/claim_audit.md) | Claim/evidence/non-claim ledger review |
| [scholarship_map.md](research/scholarship_map.md) | Literature roles, implementation status, and evidence boundaries |
| [research_roadmap.md](research/research_roadmap.md) | Gated research extensions and decision points |
| [storey_certificate_decision.md](research/storey_certificate_decision.md) | Frozen v2 campaign policy and retained Storey-failure boundary |
| [dobriban_bh_stress_bundle.md](research/dobriban_bh_stress_bundle.md) | Isolated all-chunk offline Dobriban/BH finite stress receipt |
| [visualization_notes.md](publication/visualization_notes.md) | Figure estimands, uncertainty encoding, and scope of interpretation |
| [figure_accessibility.md](publication/figure_accessibility.md) | Read-only generated-figure accessibility audit |
| [scripts/README.md](../scripts/README.md) | CLI roles, including the non-release fast-bundle integrity audit |
| [controlled_renderer.md](publication/controlled_renderer.md) | Public reproducibility boundary and controlled publication renderer |
| [rendering_pipeline.md](publication/rendering_pipeline.md) | Maintainer candidate generation and controlled-renderer runbook |
| [release_review.md](publication/release_review.md) | Read-only review and rendered-bundle finalization |
| [publication_metadata.md](publication/publication_metadata.md) | Prospective GitHub/Zenodo handoff and DOI identity policy |
| [output_inventory.md](reference/output_inventory.md) | Generated artifact contract |
| [output_conventions.md](reference/output_conventions.md) | Output namespaces and provenance conventions |
| [manuscript_numeric_audit.md](validation/manuscript_numeric_audit.md) | Numeric-token audit contract |
| [manuscript_presentation_linter.md](validation/manuscript_presentation_linter.md) | Reader-facing manuscript presentation audit |
| [syntax_guide.md](reference/syntax_guide.md) | Manuscript syntax and token policy |
| [style_guide.md](reference/style_guide.md) | Project documentation and manuscript style |
| [forking_guide.md](guides/forking_guide.md) | Safe derived-project workflow |
| [quickstart.md](guides/quickstart.md) | First successful run |
| [troubleshooting.md](guides/troubleshooting.md) | Failure diagnosis |
| [faq.md](guides/faq.md) | Scientific and operational questions |
| [AGENTS.md](AGENTS.md) | Documentation conventions for maintainers and agents |

Use `./run.sh docs` after changing navigation or local Markdown destinations.
It checks documentation links and generated gate documentation; changing
measurements belong in the generated evidence, never copied into this hub.
