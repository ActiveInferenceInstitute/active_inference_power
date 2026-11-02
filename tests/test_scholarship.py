"""Executable scholarship and citation-integrity contracts.

These checks operate on the real manuscript, bibliography, claim ledger, and
generated figure registry. They intentionally distinguish Pandoc citations
from internal cross-references such as ``@fig:power-curve``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import pytest
import yaml

from active_inference_power.schema import validate_claim_ledger
from active_inference_power.formalism import formalism_registry
from active_inference_power.manuscript_variables import _publication_repository_statement


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AT_KEY = re.compile(r"@([A-Za-z][A-Za-z0-9_:-]*)")
_BIB_KEY = re.compile(r"@(?:article|inproceedings|incollection|book|misc)\{([^,\s]+)")
_INTERNAL_PREFIXES = ("fig:", "sec:", "eq:", "tbl:")


def _bibliography_keys() -> set[str]:
    source = (PROJECT_ROOT / "manuscript" / "references.bib").read_text(
        encoding="utf-8"
    )
    return set(_BIB_KEY.findall(source))


def _manuscript_citation_keys() -> set[str]:
    keys: set[str] = set()
    for manuscript in (PROJECT_ROOT / "manuscript").glob("*.md"):
        keys.update(
            key
            for key in _AT_KEY.findall(manuscript.read_text(encoding="utf-8"))
            if not key.startswith(_INTERNAL_PREFIXES)
        )
    return keys


def test_every_manuscript_citation_key_exists_in_bibliography() -> None:
    missing = sorted(_manuscript_citation_keys() - _bibliography_keys())
    assert not missing, f"manuscript citation keys missing from references.bib: {missing}"


def test_formalism_registry_citation_keys_exist_in_bibliography() -> None:
    """Keep executable formalism references connected to real bibliography rows."""
    citations = {
        citation
        for entry in formalism_registry().entries
        for citation in entry.citations
    }
    assert citations
    missing = sorted(citations - _bibliography_keys())
    assert not missing, f"formalism citation keys missing from references.bib: {missing}"


def test_scholarship_map_is_an_evidence_matrix() -> None:
    source = (PROJECT_ROOT / "docs" / "scholarship_map.md").read_text(
        encoding="utf-8"
    )
    for heading in (
        "Literature layer",
        "What it establishes",
        "What is implemented",
        "Evidence artifact",
        "Boundary / non-claim",
        "Citation integrity policy",
    ):
        assert heading in source
    for required in (
        "Benjamini and Hochberg",
        "Benjamini and Hochberg (1997)",
        "Shafer et al.",
        "Wilkinson et al.",
        "LORD++",
        "benchmark stress",
        "Research sequencing",
        "Adaptive target selection",
        "Time-uniform concentration",
        "Dobriban (2026 preprint)",
        "formalism-to-code",
        "claim review",
    ):
        assert required in source


def test_dobriban_preprint_and_hochberg_reference_metadata_are_explicit() -> None:
    """Keep the two-source boundary and stable bibliographic identifiers auditable."""
    bibliography = (PROJECT_ROOT / "manuscript" / "references.bib").read_text(
        encoding="utf-8"
    )
    dobriban = re.search(
        r"^@misc\{dobriban2026bh,\n(?P<entry>.*?)(?=^@|\Z)",
        bibliography,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert dobriban is not None
    for required in (
        "The Benjamini--Hochberg Procedure Can Fail to Control the FDR for Correlated Two-Sided Gaussian Tests",
        "eprint={2607.12208}",
        "archivePrefix={arXiv}",
        "Preprint submitted to arXiv on 13 Jul 2026.",
        "10.48550/arXiv.2607.12208",
    ):
        assert required in dobriban.group("entry")

    hochberg = re.search(
        r"^@article\{hochberg1988sharper,\n(?P<entry>.*?)(?=^@|\Z)",
        bibliography,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert hochberg is not None
    assert "doi={10.1093/biomet/75.4.800}" in hochberg.group("entry")

    protocol = (
        PROJECT_ROOT / "src" / "active_inference_power" / "protocols" / "dobriban_bh.py"
    ).read_text(encoding="utf-8")
    roadmap = (PROJECT_ROOT / "docs" / "research_roadmap.md").read_text(
        encoding="utf-8"
    )
    scholarship_map = (PROJECT_ROOT / "docs" / "scholarship_map.md").read_text(
        encoding="utf-8"
    )
    references_policy = (PROJECT_ROOT / "manuscript" / "99_references.md").read_text(
        encoding="utf-8"
    )
    assert "publicly documented" in protocol
    assert "published protocol" not in protocol
    assert "publicly posted preprint" in roadmap
    assert "The first three columns identify the literature layer" in scholarship_map
    assert "DOI links are retained whenever a stable DOI exists" in references_policy.replace(
        "\n", " "
    )


def test_foundational_and_selective_reference_dois_are_explicit() -> None:
    """Retain stable identifiers without relabeling the selective-inference preprint."""
    bibliography = (PROJECT_ROOT / "manuscript" / "references.bib").read_text(
        encoding="utf-8"
    )
    expected = {
        "holm1979simple": "10.2307/4615733",
        "genovese2002operating": "10.1111/1467-9868.00347",
        "fithian2014selective": "10.48550/arXiv.1410.2597",
    }
    for key, doi in expected.items():
        entry = re.search(
            rf"^@(?:article|misc)\{{{key},\n(?P<entry>.*?)(?=^@|\Z)",
            bibliography,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert entry is not None
        assert f"doi={{{doi}}}" in entry.group("entry")

    fithian = re.search(
        r"^@misc\{fithian2014selective,\n(?P<entry>.*?)(?=^@|\Z)",
        bibliography,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert fithian is not None
    for required in (
        "eprint={1410.2597}",
        "archivePrefix={arXiv}",
        "doi={10.48550/arXiv.1410.2597}",
        "url={https://arxiv.org/abs/1410.2597}",
    ):
        assert required in fithian.group("entry")


def test_public_surface_uses_canonical_terminology() -> None:
    roots = (
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "manuscript",
        PROJECT_ROOT / "examples",
    )
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in roots
        for path in ((root,) if root.is_file() else root.rglob("*.md"))
    )
    assert "active_inference_power" in text
    assert "naive belief" not in text.lower()
    assert "naive" not in text.lower()
    assert "naïve" not in text.lower()
    introduction = (PROJECT_ROOT / "manuscript" / "01_introduction.md").read_text(
        encoding="utf-8"
    )
    assert "trace field is named `wealth`" not in introduction
    assert "trace field named `wealth`" not in introduction
    experimental_setup = (PROJECT_ROOT / "manuscript" / "05_experimental_setup.md").read_text(
        encoding="utf-8"
    )
    assert "{{CONFIG_SUITE_NAME}}` suite" not in experimental_setup
    assert not (PROJECT_ROOT / "manuscript" / "layer_contract.yaml").exists()


def test_abstract_and_introduction_define_acronyms_and_notation() -> None:
    abstract = (PROJECT_ROOT / "manuscript" / "00_abstract.md").read_text(
        encoding="utf-8"
    )
    introduction = (PROJECT_ROOT / "manuscript" / "01_introduction.md").read_text(
        encoding="utf-8"
    )
    assert "[@" not in abstract
    abstract_words = re.findall(r"\b[\w’'-]+\b", abstract)
    assert 180 <= len(abstract_words) <= 210
    assert "zenodo" not in abstract.lower()
    assert "doi" not in abstract.lower()
    assert "{{RESULT_" not in abstract
    assert "{{QUALITY_" not in abstract
    assert "{{PUBLICATION_REPOSITORY_STATEMENT}}" in abstract
    config = yaml.safe_load(
        (PROJECT_ROOT / "manuscript" / "config.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(config, dict)
    rendered_abstract = abstract.replace(
        "{{PUBLICATION_REPOSITORY_STATEMENT}}",
        _publication_repository_statement(config),
    )
    rendered_abstract_words = re.findall(r"\b[\w’'-]+\b", rendered_abstract)
    assert 180 <= len(rendered_abstract_words) <= 210
    for phrase in (
        "Benjamini–Hochberg (BH)",
        "false discovery rate (FDR)",
        "family-wise error rate\n(FWER)",
        "Monte Carlo standard error (MCSE)",
    ):
        assert phrase in abstract
    for phrase in (
        "## A concrete adaptive-design problem",
        "## Study aims and evidence boundary",
        "### Notation and terminology: a reader guide",
        "**Conditional power / mean true-positive rate (TPR)** $\\theta_{\\mathrm{TPR}}(d)$",
        "visible filtration $\\mathcal F_t$",
        "**Likelihood-ratio e-process** $E_t$",
        "**Online alpha-wealth** $W_t^\\alpha$",
        "**Resource cost** $C$",
        "False discovery proportion (FDP)",
        "Positive regression dependence on a subset (PRDS)",
        "Levels based On Recent Discovery (LORD) is an online p-value family",
        "(SAFFRON) uses candidate information",
        "application programming interface\n(API)",
        "JavaScript Object Notation (JSON)",
    ):
        assert phrase in introduction
    assert introduction.index("## Study aims and evidence boundary") < introduction.index(
        "### Notation and terminology: a reader guide"
    )
    assert "Neither quantity is the resource cost $C$." in introduction
    assert "investigator" in abstract.lower()
    assert "embedded agent acts only on visible history" in abstract
    assert "simulated\nniche" in introduction


def test_reader_facing_power_distinguishes_investigator_agent_and_task_environment() -> None:
    """Keep the central estimand aligned with the executable scenario/policy split."""
    perspective = (PROJECT_ROOT / "docs" / "evaluation_perspective.md").read_text(
        encoding="utf-8"
    )
    methodology = (PROJECT_ROOT / "manuscript" / "02_methodology.md").read_text(
        encoding="utf-8"
    )
    action_docs = (PROJECT_ROOT / "docs" / "action_in_the_loop.md").read_text(
        encoding="utf-8"
    )
    scenario_source = (
        PROJECT_ROOT / "src" / "active_inference_power" / "scenario.py"
    ).read_text(encoding="utf-8")
    formalism = formalism_registry()
    power_entry = next(
        entry for entry in formalism.entries if entry.entry_id == "conditional-power-functional"
    )
    normalized_perspective = re.sub(r"\s+", " ", perspective).lower()
    for phrase in (
        "investigator-facing operating characteristic",
        "simulated niche",
        "not a new api object",
        "not a probability distribution over niches",
        "any_true_rejection_rate",
    ):
        assert phrase in normalized_perspective
    assert "s=(M_A,P_N,T)" in methodology
    assert "d=(s,\\pi)" in methodology
    assert "replication-level true-positive rate" in methodology
    assert "PowerScenario` hashes model, process,\nand setting" in action_docs
    assert "SettingSpec.stopping_policy`` is a hashed default" in scenario_source
    assert "does not contain a\n    separately supplied surface/action policy" in scenario_source
    assert "P(|R ∩ H1| > 0)" in power_entry.formal_object
    assert "scenario model, process, setting, and policy are declared" not in power_entry.assumptions


def test_study_lanes_keep_agent_uncertainty_separate_from_the_estimand() -> None:
    """Prevent an action-loop mechanism from being misread as an agent score."""
    perspective = (PROJECT_ROOT / "docs" / "evaluation_perspective.md").read_text(
        encoding="utf-8"
    )
    action_docs = (PROJECT_ROOT / "docs" / "action_in_the_loop.md").read_text(
        encoding="utf-8"
    )
    faq = (PROJECT_ROOT / "docs" / "faq.md").read_text(encoding="utf-8")
    introduction = (PROJECT_ROOT / "manuscript" / "01_introduction.md").read_text(
        encoding="utf-8"
    )
    scope = (PROJECT_ROOT / "manuscript" / "07_scope_and_related_work.md").read_text(
        encoding="utf-8"
    )

    assert "## Two study lanes, one evaluative perspective" in perspective
    assert "Fixed-horizon method and calibration comparison" in perspective
    assert "Action-loop design comparison" in perspective
    assert "Neither lane assigns statistical power to the agent itself." in perspective
    assert "This document describes the action-loop lane only." in action_docs
    assert "The closer description is an investigator simulating an agent" in faq
    assert "The suite consequently has two related study lanes." in introduction
    assert "The fixed-horizon comparison lanes are investigator simulations" in scope


def test_reader_terms_keep_evidence_randomness_cost_and_provenance_distinct() -> None:
    """Protect the technical distinctions that make the reader guide auditable."""
    introduction = (PROJECT_ROOT / "manuscript" / "01_introduction.md").read_text(
        encoding="utf-8"
    )
    methodology = (PROJECT_ROOT / "manuscript" / "02_methodology.md").read_text(
        encoding="utf-8"
    )
    roadmap = (PROJECT_ROOT / "docs" / "research_roadmap.md").read_text(
        encoding="utf-8"
    )
    action_docs = (PROJECT_ROOT / "docs" / "action_in_the_loop.md").read_text(
        encoding="utf-8"
    )
    scholarship_map = (PROJECT_ROOT / "docs" / "scholarship_map.md").read_text(
        encoding="utf-8"
    )
    formalism_docs = (PROJECT_ROOT / "docs" / "formalism_to_code.md").read_text(
        encoding="utf-8"
    )
    visualization_docs = (PROJECT_ROOT / "docs" / "visualization_notes.md").read_text(
        encoding="utf-8"
    )
    faq = (PROJECT_ROOT / "docs" / "faq.md").read_text(encoding="utf-8")
    ledger = yaml.safe_load(
        (PROJECT_ROOT / "data" / "claim_ledger.yaml").read_text(encoding="utf-8")
    )
    normalized_intro = re.sub(r"\s+", " ", introduction)
    normalized_methodology = re.sub(r"\s+", " ", methodology)
    normalized_action_docs = re.sub(r"\s+", " ", action_docs)
    normalized_roadmap = re.sub(r"\s+", " ", roadmap)
    normalized_scholarship_map = re.sub(r"\s+", " ", scholarship_map)
    normalized_formalism_docs = re.sub(r"\s+", " ", formalism_docs)
    normalized_visualization_docs = re.sub(r"\s+", " ", visualization_docs)
    assert "supermartingale under its declared null" in normalized_intro
    assert "multiplicative e-factor has conditional mean at most one" in normalized_intro
    assert "P_N$ supplies the declared process randomness" in normalized_methodology
    assert "U_\\pi$ denotes any policy and trace-level randomization" in normalized_methodology
    assert "mean of replication-level $\\mathrm{TPR}/C$ scores" in normalized_methodology
    assert "P_N\\) denotes the declared process randomness" in normalized_roadmap
    assert "U_\\pi\\) denotes policy and trace-level randomization" in normalized_roadmap
    assert "mean of replication-level power-per-unit-cost" in normalized_action_docs
    assert "not a ratio of scenario-level summaries" in normalized_action_docs
    assert "Formal validity claims, where made, rest on the cited theory" in normalized_scholarship_map
    assert "they do not create a theorem on their own" in normalized_scholarship_map
    assert "unique scenario identifier" in normalized_formalism_docs
    assert "user-supplied" in normalized_formalism_docs
    assert "scenario configuration hash as its content address" in normalized_visualization_docs
    assert "not, by itself, a calibrated p-value" in faq
    lineage = next(
        claim
        for claim in ledger["claims"]
        if claim["claim_id"] == "scenario-surface-lineage-identifiers"
    )
    assert lineage["value"] == (
        "required_unique_scenario_identifiers_and_configuration_hashes_"
        "match_serialized_contracts"
    )

    entries = {entry.entry_id: set(entry.citations) for entry in formalism_registry().entries}
    assert {"neyman1933efficient", "morris2019simulation"} <= entries[
        "conditional-power-functional"
    ]
    assert {"javanmard2018online", "xu2024online_evalues"} <= entries[
        "online-fdr-recursions"
    ]
    assert "xu2024online_evalues" in entries["sequential-evidence-comparison-atlas"]


def test_conclusion_gives_a_design_decision_rule_not_a_universal_power_claim() -> None:
    conclusion = (PROJECT_ROOT / "manuscript" / "04_conclusion.md").read_text(
        encoding="utf-8"
    )
    for phrase in (
        "Declare the new scenario",
        "visible filtration",
        "finite simulation",
        "results do not show",
        "conditional evidence chain",
    ):
        assert phrase in conclusion
    assert "universal power" in conclusion


def test_dense_adaptive_audit_starts_on_a_fresh_pdf_page_for_caption_legibility() -> None:
    results = (PROJECT_ROOT / "manuscript" / "03_results.md").read_text(
        encoding="utf-8"
    )
    assert (
        "\\clearpage\n\n"
        "![{{FIGURE_ADAPTIVE_FDR_AUDIT_CAPTION}}]"
    ) in results


def test_final_formalism_panel_has_top_clearance_on_its_dedicated_page() -> None:
    """Prevent the final print matrix title from entering the PDF crop edge."""
    methodology = (PROJECT_ROOT / "manuscript" / "02_methodology.md").read_text(
        encoding="utf-8"
    )
    assert (
        "\\clearpage\n\n"
        "\\vspace*{ {{FIGURE_FORMALISM_FINAL_TOP_CLEARANCE}} }\n\n"
        "![{{FIGURE_FORMALISM_CODE_MAP_FINAL_CAPTION}}]"
        "(../output/figures/formalism_code_map_final.png)"
        '{#fig:formalism-code-map-final alt="{{FIGURE_FORMALISM_CODE_MAP_FINAL_ALT}}" '
        "width=100% height={{FIGURE_FORMALISM_MAP_HEIGHT}}}"
        "\n\n\\clearpage"
    ) in methodology


def test_benchmark_stress_float_is_flushed_before_the_concluding_sections() -> None:
    """Keep the Results figure from splitting a Conclusion sentence in PDF."""
    results = (PROJECT_ROOT / "manuscript" / "03_results.md").read_text(
        encoding="utf-8"
    )
    assert (
        "![{{FIGURE_BENCHMARK_STRESS_CAPTION}}]"
        "(../output/figures/benchmark_stress.png)"
        "{#fig:benchmark-stress alt=\"{{FIGURE_BENCHMARK_STRESS_ALT}}\" width=90%}"
        "\n\n\\clearpage\n\n"
        "## Evidence-class synthesis"
    ) in results


def test_conclusion_states_the_deliverable_directly() -> None:
    conclusion = (PROJECT_ROOT / "manuscript" / "04_conclusion.md").read_text(
        encoding="utf-8"
    )
    assert "The central result is a change in the question" not in conclusion
    assert "This suite estimates statistical power" in conclusion
    assert "conditional evidence chain" in conclusion


def test_public_surface_has_no_stale_namespace_terms() -> None:
    """Keep the canonical package namespace clean across public surfaces."""
    roots = (
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "manuscript",
        PROJECT_ROOT / "examples",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "data",
    )
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for root in roots
        for path in ((root,) if root.is_file() else root.rglob("*"))
        if path.is_file() and path.suffix in {".md", ".py", ".yaml", ".yml", ".toml"}
    ).lower()
    forbidden = (
        "leg" + "acy",
        "depre" + "cated",
        "flat-" + "import",
        "import" + " src",
        "from" + " src",
        "src" + ".active",
        "compati" + "b",
    )
    for token in forbidden:
        assert token not in text, f"stale public terminology remains: {token}"
    flat_modules = (
        "__init__.py",
        "active_inference.py",
        "certificates.py",
        "corrections.py",
        "error_rates.py",
        "manuscript_variables.py",
        "power.py",
        "simulation.py",
        "theory.py",
    )
    assert not any((PROJECT_ROOT / "src" / name).is_file() for name in flat_modules)


def test_research_roadmap_has_stage_controls() -> None:
    roadmap = (PROJECT_ROOT / "docs" / "research_roadmap.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "North-star estimand",
        "Evidence classes",
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "Stage 4",
        "Stage 5",
        "Change-control rules",
        "non-claims",
    ):
        assert required in roadmap


def test_citation_claims_and_scholarship_sources_are_real_files() -> None:
    ledger = yaml.safe_load(
        (PROJECT_ROOT / "data" / "claim_ledger.yaml").read_text(encoding="utf-8")
    )
    claims = ledger["claims"]
    bibliography = _bibliography_keys()
    citation_claims = [claim for claim in claims if claim.get("kind") == "citation"]
    assert citation_claims
    for claim in citation_claims:
        assert claim["value"] in bibliography
    for claim in claims:
        source_path = str(claim["source"]).split(":", 1)[0]
        assert (PROJECT_ROOT / source_path).exists(), claim["claim_id"]
    ids = {claim["claim_id"] for claim in claims}
    assert "scholarship-boundary-map" in ids
    assert "weighted-bh-literature-grounding" in ids
    assert "fair-artifact-provenance" in ids
    assert "formalism-to-code-registry" in ids
    assert "formalism-registry-validation" in ids
    assert "claim-audit-protocol" in ids
    assert "learned-policy-complexity-accounting" in ids


def test_claim_ledger_is_schema_checked_and_claim_bearing_rows_have_boundaries() -> None:
    ledger = yaml.safe_load(
        (PROJECT_ROOT / "data" / "claim_ledger.yaml").read_text(encoding="utf-8")
    )
    # Schema-only against the shipped ledger: whether the declared artifacts exist
    # is a candidate-bundle property, and the artifact requirement itself is
    # exercised with a producer-built file in tests/test_artifacts.py. Requiring a
    # generated output/ bundle here made the unit suite fail on any clean checkout.
    validate_claim_ledger(ledger, project_root=PROJECT_ROOT)
    claim_rows = {
        claim["claim_id"]: claim
        for claim in ledger["claims"]
        if claim["kind"] in {"validity_boundary", "empirical_stress_test", "empirical_operating_characteristic", "scope_definition", "verdict", "schema_contract"}
    }
    assert claim_rows
    assert all(row.get("evidence_class") and row.get("non_claim") for row in claim_rows.values())
    with pytest.raises(ValueError):
        validate_claim_ledger({"claims": [dict(ledger["claims"][0], claim_id="duplicate"), dict(ledger["claims"][0], claim_id="duplicate")]})

    unsupported = copy.deepcopy(ledger)
    claim_bearing = next(
        claim
        for claim in unsupported["claims"]
        if claim["kind"] in {
            "validity_boundary",
            "empirical_stress_test",
            "empirical_operating_characteristic",
            "scope_definition",
            "verdict",
            "schema_contract",
        }
    )
    claim_bearing["evidence_class"] = "unbounded_guarantee"
    with pytest.raises(ValueError, match="unsupported evidence_class"):
        validate_claim_ledger(unsupported, project_root=PROJECT_ROOT)

    evidence_bearing_policy = copy.deepcopy(ledger)
    policy_row = next(
        claim
        for claim in evidence_bearing_policy["claims"]
        if claim["kind"] == "computation_policy" and claim.get("evidence_class")
    )
    policy_row["evidence_class"] = "unbounded_guarantee"
    with pytest.raises(ValueError, match="unsupported evidence_class"):
        validate_claim_ledger(evidence_bearing_policy, project_root=PROJECT_ROOT)


def test_storey_review_ledger_uses_reader_facing_same_vector_language() -> None:
    ledger = yaml.safe_load(
        (PROJECT_ROOT / "data" / "claim_ledger.yaml").read_text(encoding="utf-8")
    )
    review = next(
        claim
        for claim in ledger["claims"]
        if claim["claim_id"] == "frozen-three-way-storey-method-review"
    )
    value = str(review["value"])
    assert "same_vector_storey_plugin" in value
    assert "naive_same_vector" not in value


def test_formalism_and_claim_audit_docs_are_linked() -> None:
    formalism = (PROJECT_ROOT / "docs" / "formalism_to_code.md").read_text(encoding="utf-8")
    audit = (PROJECT_ROOT / "docs" / "claim_audit.md").read_text(encoding="utf-8")
    for text in (formalism, audit):
        assert "formal object" in text
        assert "non-claim" in text
        assert "evidence class" in text
    assert "formalism_registry.json" in formalism
    assert "output/formalism_registry.json" in audit


def test_scope_chain_uses_reader_facing_arrows() -> None:
    """Keep the formal-to-code chain legible in the rendered reader guide."""
    scope = (PROJECT_ROOT / "manuscript" / "07_scope_and_related_work.md").read_text(
        encoding="utf-8"
    )
    assert "formal object → public symbol →" in scope
    assert "formal object->" not in scope


def test_generated_registry_contains_the_scholarship_boundary(
    generated_fast_bundle: Path,
) -> None:
    registry = json.loads(
        (generated_fast_bundle / "figures" / "figure_registry.json").read_text(
            encoding="utf-8"
        )
    )
    labels = {item["label"] for item in registry["figures"]}
    assert "fig:scholarship-boundary" in labels
    record = next(
        item for item in registry["figures"] if item["label"] == "fig:scholarship-boundary"
    )
    assert record["filename"] == "scholarship_boundary.png"
    assert record["caption"] and record["alt"]
