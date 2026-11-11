# Documentation hub

## Choose a reading path

- **First local run:** prepare the locked environment and make a disposable
  smoke bundle in [quickstart.md](quickstart.md).
- **Methods reader:** start with the
  [evaluation perspective](evaluation_perspective.md), then the
  [action-in-the-loop contract](action_in_the_loop.md),
  [formalism-to-code map](formalism_to_code.md), and
  [visualization notes](visualization_notes.md).
- **Contributor:** read the operating
  [agent instructions](agent_instructions.md),
  [architecture](architecture.md), and
  [package contract](PACKAGE_CONTRACT.md) before changing code or contracts;
  use the [script inventory](../scripts/README.md) to distinguish the
  non-release fast-bundle audit from the release chain.
- **Release reviewer:** follow the serial
  [rendering pipeline](rendering_pipeline.md) and distinguish its read-only
  [release review](release_review.md) from the state-changing finalizer.
- **External or campaign evidence:** read the
  [research roadmap](research_roadmap.md),
  [Storey decision record](storey_certificate_decision.md), and benchmark
  documentation before treating a diagnostic lane as evidence.

| Document | Purpose |
|---|---|
| [agent_instructions.md](agent_instructions.md) | Project operating contract |
| [PACKAGE_CONTRACT.md](PACKAGE_CONTRACT.md) | API and mathematical invariants |
| [architecture.md](architecture.md) | Source → artifact dependency graph |
| [evaluation_perspective.md](evaluation_perspective.md) | Investigator, simulated-agent, and task-environment role boundary |
| [testing_philosophy.md](testing_philosophy.md) | No-mock and coverage policy |
| [action_in_the_loop.md](action_in_the_loop.md) | Model/process/setting and policy execution contracts |
| [../models/README.md](../models/README.md) | GNN specifications for the declared generative models and the upstream custody pin |
| [formalism_to_code.md](formalism_to_code.md) | Executable formalism and source-symbol traceability |
| [claim_audit.md](claim_audit.md) | Claim/evidence/non-claim ledger review |
| [scholarship_map.md](scholarship_map.md) | Literature roles, implementation status, and evidence boundaries |
| [research_roadmap.md](research_roadmap.md) | Gated research extensions and decision points |
| [storey_certificate_decision.md](storey_certificate_decision.md) | Frozen v2 campaign policy and retained Storey-failure boundary |
| [dobriban_bh_stress_bundle.md](dobriban_bh_stress_bundle.md) | Isolated all-chunk offline Dobriban/BH finite stress receipt |
| [visualization_notes.md](visualization_notes.md) | Figure estimands, uncertainty encoding, and scope of interpretation |
| [figure_accessibility.md](figure_accessibility.md) | Read-only generated-figure accessibility audit |
| [scripts/README.md](../scripts/README.md) | CLI roles, including the non-release fast-bundle integrity audit |
| [controlled_renderer.md](controlled_renderer.md) | Public reproducibility boundary and controlled publication renderer |
| [rendering_pipeline.md](rendering_pipeline.md) | Maintainer candidate generation and controlled-renderer runbook |
| [release_review.md](release_review.md) | Read-only review and rendered-bundle finalization |
| [publication_metadata.md](publication_metadata.md) | Prospective GitHub/Zenodo handoff and DOI identity policy |
| [output_inventory.md](output_inventory.md) | Generated artifact contract |
| [output_conventions.md](output_conventions.md) | Output namespaces and provenance conventions |
| [manuscript_numeric_audit.md](manuscript_numeric_audit.md) | Numeric-token audit contract |
| [manuscript_presentation_linter.md](manuscript_presentation_linter.md) | Reader-facing manuscript presentation audit |
| [syntax_guide.md](syntax_guide.md) | Manuscript syntax and token policy |
| [style_guide.md](style_guide.md) | Project documentation and manuscript style |
| [forking_guide.md](forking_guide.md) | Safe derived-project workflow |
| [quickstart.md](quickstart.md) | First successful run |
| [troubleshooting.md](troubleshooting.md) | Failure diagnosis |
| [faq.md](faq.md) | Scientific and operational questions |
| [AGENTS.md](AGENTS.md) | Documentation conventions for maintainers and agents |

Run `./.venv/bin/python scripts/audit_documentation_links.py` after changing
navigation or local Markdown destinations.
