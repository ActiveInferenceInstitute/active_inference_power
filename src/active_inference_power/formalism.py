"""Executable traceability between statistical formalisms and package code.

The registry is deliberately small and explicit.  It is not a second source of
scientific results: it records which formal object a public symbol realizes,
which assumptions make that realization meaningful, which tests exercise it,
and what the current evidence must not be interpreted to establish.

Keeping this map in the domain package makes the connection inspectable by
users, release tooling, and manuscript hydration without importing the
publication renderer into scientific code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


EVIDENCE_CLASSES = frozenset(
    {
        "formal_conditional",
        "analytic_reference",
        "conditional_operating_characteristic",
        "calibration_diagnostic",
        "stress_test",
        "provenance_contract",
    }
)
LAYERS = frozenset(
    {
        "estimand",
        "generative_model",
        "generative_process",
        "testing_setting",
        "inference",
        "evidence",
        "decision",
        "uncertainty",
        "artifact",
    }
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _labels(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or any(not value for value in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique non-empty strings")
    return result


def resolve_code_symbol(reference: str) -> Any:
    """Resolve a registry anchor against the installed canonical package.

    Formalism entries use dotted references such as ``scenario.PowerScenario``
    or ``protocols.DobribanBHResult``.  The longest importable prefix is
    treated as a package module and the remaining components are traversed as
    attributes.  This makes the registry an executable crosswalk rather than
    a collection of unchecked strings while keeping the public API free of
    renderer or filesystem dependencies.
    """
    parts = tuple(reference.split(".")) if isinstance(reference, str) else ()
    if len(parts) < 2 or any(not part or not part.isidentifier() for part in parts):
        raise ValueError(f"invalid code symbol reference: {reference!r}")
    for module_length in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:module_length])
        try:
            value: Any = importlib.import_module(f"active_inference_power.{module_name}")
        except ModuleNotFoundError:
            continue
        for attribute in parts[module_length:]:
            try:
                value = getattr(value, attribute)
            except AttributeError as error:
                raise ValueError(
                    f"code symbol {reference!r} resolves module {module_name!r} "
                    f"but is missing attribute {attribute!r}"
                ) from error
        return value
    raise ValueError(f"code symbol module cannot be imported: {reference!r}")


@dataclass(frozen=True)
class FormalismEntry:
    """One formal object and its executable, evidentiary, and scholarly links."""

    entry_id: str
    layer: str
    title: str
    formal_object: str
    code_symbols: tuple[str, ...]
    source_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    assumptions: tuple[str, ...]
    evidence_class: str
    claim: str
    non_claim: str
    artifact_paths: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.entry_id or not self.title or not self.formal_object:
            raise ValueError("formalism entries require an id, title, and formal object")
        if self.layer not in LAYERS:
            raise ValueError(f"unsupported formalism layer: {self.layer}")
        if self.evidence_class not in EVIDENCE_CLASSES:
            raise ValueError(f"unsupported evidence class: {self.evidence_class}")
        if not self.claim or not self.non_claim:
            raise ValueError("formalism entries require a claim and non-claim")
        for name, values in (
            ("code_symbols", self.code_symbols),
            ("source_paths", self.source_paths),
            ("test_paths", self.test_paths),
            ("assumptions", self.assumptions),
        ):
            _labels(values, name)
        for symbol in self.code_symbols:
            if "." not in symbol:
                raise ValueError("code_symbols must use dotted package-relative references")
        for name, values in (("artifact_paths", self.artifact_paths), ("citations", self.citations)):
            if any(not str(value) for value in values):
                raise ValueError(f"{name} cannot contain empty strings")
        if any(not path.startswith(("src/", "scripts/")) for path in self.source_paths):
            raise ValueError("source_paths must be project-relative src or scripts paths")
        if any(not path.startswith("tests/") for path in self.test_paths):
            raise ValueError("test_paths must be project-relative test paths")

    def to_mapping(self) -> dict[str, Any]:
        """Return the deterministic JSON representation used in artifacts."""
        return {
            "entry_id": self.entry_id,
            "layer": self.layer,
            "title": self.title,
            "formal_object": self.formal_object,
            "code_symbols": list(self.code_symbols),
            "source_paths": list(self.source_paths),
            "test_paths": list(self.test_paths),
            "assumptions": list(self.assumptions),
            "evidence_class": self.evidence_class,
            "claim": self.claim,
            "non_claim": self.non_claim,
            "artifact_paths": list(self.artifact_paths),
            "citations": list(self.citations),
        }

    @property
    def entry_hash(self) -> str:
        payload = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FormalismRegistry:
    """Versioned collection of formalism-to-code mappings."""

    entries: tuple[FormalismEntry, ...]
    schema_version: str = "2.0"

    def __post_init__(self) -> None:
        if self.schema_version != "2.0":
            raise ValueError("formalism registry schema must be 2.0")
        ids = [entry.entry_id for entry in self.entries]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("formalism registry entry ids must be unique and non-empty")

    @property
    def registry_hash(self) -> str:
        payload = json.dumps(
            [entry.to_mapping() for entry in self.entries],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_hash": self.registry_hash,
            "entry_count": len(self.entries),
            "entries": [entry.to_mapping() | {"entry_hash": entry.entry_hash} for entry in self.entries],
        }


def _entry(
    entry_id: str,
    layer: str,
    title: str,
    formal_object: str,
    code_symbols: tuple[str, ...],
    source_paths: tuple[str, ...],
    test_paths: tuple[str, ...],
    assumptions: tuple[str, ...],
    evidence_class: str,
    claim: str,
    non_claim: str,
    *,
    artifact_paths: tuple[str, ...] = (),
    citations: tuple[str, ...] = (),
) -> FormalismEntry:
    return FormalismEntry(
        entry_id,
        layer,
        title,
        formal_object,
        code_symbols,
        source_paths,
        test_paths,
        assumptions,
        evidence_class,
        claim,
        non_claim,
        artifact_paths,
        citations,
    )


def formalism_registry() -> FormalismRegistry:
    """Return the release traceability registry in stable scholarly order."""
    return FormalismRegistry(
        (
            _entry(
                "conditional-power-functional",
                "estimand",
                "Conditional power functional",
                "θ_TPR(s, π) = E_process,π[|R ∩ H1| / max(1, |H1|)]; P(|R ∩ H1| > 0) is reported separately",
                ("scenario.PowerScenario", "action_loop.run_power_scenario", "suite.run_power_suite"),
                ("src/active_inference_power/scenario.py", "src/active_inference_power/action_loop.py", "src/active_inference_power/suite.py"),
                ("tests/test_action_loop_suite.py", "tests/test_scholarship.py"),
                ("hashed scenario contains model, process, and setting; policy is a separate row key", "truth is known to the evaluator but hidden from the policy", "replication-level TPR and any-true detection are separately defined"),
                "conditional_operating_characteristic",
                "Replication-mean TPR is reported as an investigator-estimated scenario/policy operating characteristic.",
                "No unconditional or agent-invariant power claim, niche average, or any-true-event substitution is made.",
                artifact_paths=("output/results.json", "output/action_loop_suite.json"),
                citations=(
                    "neyman1933efficient",
                    "morris2019simulation",
                    "genovese2002operating",
                    "friston2017active_process",
                ),
            ),
            _entry(
                "model-process-setting-contract",
                "generative_model",
                "Generative model, process, and testing-setting composition",
                "s = (M_A, P_N, T); d = (s, π), with model M_A distinct from process P_N and policy π",
                ("scenario.GenerativeModelSpec", "scenario.GenerativeProcessSpec", "scenario.SettingSpec", "scenario.PowerScenario"),
                ("src/active_inference_power/scenario.py",),
                ("tests/test_action_loop_suite.py", "tests/test_branch_contracts.py"),
                ("action-conditioned kernels have aligned state, observation, and action spaces", "probability rows are normalized", "the setting declares the family and validity boundary"),
                "formal_conditional",
                "The public scenario contract hashes model/process/setting, while each operating row records the separately evaluated policy.",
                "The contract does not prove that an arbitrary user-supplied process is statistically valid.",
                artifact_paths=("output/results.json", "output/action_loop_suite.json"),
                citations=("friston2017active_process", "dacosta2020discrete"),
            ),
            _entry(
                "bayesian-oracle-equivalence",
                "inference",
                "Bayesian belief update and executable oracle",
                "posterior odds = prior odds × product of action-conditioned likelihood ratios",
                ("active_inference.run_ai_test", "active_inference.evidence_pvalue", "action_loop.run_action_loop", "pymdp_adapter.make_agent"),
                ("src/active_inference_power/active_inference.py", "src/active_inference_power/action_loop.py", "src/active_inference_power/pymdp_adapter.py"),
                ("tests/test_active_inference.py", "tests/test_agent_sampling.py", "tests/test_action_loop_suite.py"),
                ("the declared likelihood is the agent's likelihood", "the binary oracle and real pymdp agent use the same state semantics", "hidden process states are excluded from policy inputs"),
                "formal_conditional",
                "The real pymdp agent is checked against a closed-form binary update on reference configurations.",
                "Oracle agreement is not evidence that the binary channel represents active inference in general.",
                artifact_paths=("output/results.json", "output/figures/active_inference_policy.png"),
                citations=("dacosta2020discrete", "heins2022pymdp"),
            ),
            _entry(
                "fixed-horizon-multiple-testing",
                "testing_setting",
                "Fixed-horizon multiple-testing procedures",
                "R = {i: adjusted p_i ≤ α}; FDP = V/max(R,1), FDR = E[FDP]",
                ("corrections.benjamini_hochberg", "corrections.benjamini_yekutieli", "corrections.weighted_bh", "error_rates.false_discovery_proportion"),
                ("src/active_inference_power/corrections.py", "src/active_inference_power/error_rates.py"),
                ("tests/test_corrections.py", "tests/test_certificates.py", "tests/test_contract_invariants.py"),
                ("p-values are in [0,1]", "weighted BH weights are fixed, positive, and normalized", "BH interpretation follows the declared independence or PRDS boundary"),
                "formal_conditional",
                "The package exposes fixed-horizon corrections with validated input and rejection invariants.",
                "Finite simulations do not establish a new theorem or universal method ranking.",
                artifact_paths=("output/results.json", "output/certificates.json"),
                citations=("benjamini1995controlling", "benjamini1997weighted", "benjamini2001control"),
            ),
            _entry(
                "global-null-bh-count-fdr-distinction",
                "testing_setting",
                "Global-null BH rejection count and FDR distinction",
                "E[R] = sum_{r>=1} P(R>=r), while FDR = P(R>0) = alpha under the independent continuous global null",
                ("theory.expected_false_rejections_global_null", "theory.bh_global_null_fdr", "theory.simes_global_pvalue"),
                ("src/active_inference_power/theory.py",),
                ("tests/test_theory.py",),
                (
                    "all p-values are independent continuous Uniform(0,1) under the global null",
                    "the BH thresholds are the declared finite family thresholds",
                    "the expected rejection count is computed by the ordered-threshold dynamic program",
                ),
                "analytic_reference",
                "The implementation keeps the exact global-null FDR identity separate from the expected number of false rejections.",
                "This identity and finite expectation calculation do not establish arbitrary-dependence validity or a universal finite-sample method ranking.",
                artifact_paths=("output/formalism_registry.json", "output/results.json"),
                citations=("benjamini1995controlling",),
            ),
            _entry(
                "conservative-adaptive-fdr",
                "testing_setting",
                "Conservative fixed-lambda adaptive FDR",
                "K = #{i: p_i > λ}; U_m0 = max{u: P(Binomial(u, 1−λ) ≤ K) ≥ 1−δ}; q_i = (U_m0/m) q_i^BH",
                ("corrections.storey_qvalue_conservative", "corrections.apply_correction"),
                ("src/active_inference_power/corrections.py",),
                ("tests/test_corrections.py", "tests/test_formalism.py"),
                (
                    "null p-values are independent and marginally uniform",
                    "lambda and confidence are fixed before p-values are inspected",
                    "alternative tail counts are nonnegative",
                ),
                "calibration_diagnostic",
                "The package exposes an exact fixed-lambda null-count upper-bound construction under its independent-null contract.",
                "Applying that bound to the same p-values is an assumption-gated adaptive diagnostic, not a finite-sample FDR theorem under arbitrary dependence, data-adaptive lambda, or misspecified p-values.",
                artifact_paths=("output/results.json", "output/certificates.json"),
                citations=("storey2002direct",),
            ),
            _entry(
                "split-stream-storey-confirmation",
                "testing_setting",
                "Independent split-stream Storey confirmation",
                "U = U(p_sel); R = {i: q_i(p_conf, U) ≤ α}, with p_sel ⟂ p_conf",
                ("corrections.storey_qvalue_split_confirmation", "certificates.certify_storey_split_confirmation"),
                ("src/active_inference_power/corrections.py", "src/active_inference_power/certificates.py"),
                ("tests/test_corrections.py", "tests/test_certificates.py"),
                ("selection and confirmation streams are independent", "lambda and confidence are preregistered", "confirmation null p-values are marginally super-uniform"),
                "calibration_diagnostic",
                "The release separately audits a selection-only null-count estimate and confirmation-only decisions.",
                "The finite certificate is not a new universal Storey FDR theorem and does not validate same-stream or arbitrary adaptive selection.",
                artifact_paths=("output/certificates.json", "output/results.json"),
                citations=("storey2002direct", "fithian2014selective"),
            ),
            _entry(
                "dependence-generators",
                "generative_process",
                "Dependence-aware Gaussian process generators",
                "Z = L F with Σ = LLᵀ and declared negative, block, or factor structure",
                ("dependence.correlation_matrix", "dependence.sample_gaussian", "simulation.generate_pvalues"),
                ("src/active_inference_power/dependence.py", "src/active_inference_power/simulation.py"),
                ("tests/test_dependence.py", "tests/test_calibration.py"),
                ("the covariance matrix is positive semidefinite", "marginal null calibration is checked separately", "the dependence regime is named in the result"),
                "calibration_diagnostic",
                "The release compares declared dependence regimes and checks their covariance construction.",
                "Negative, block, and factor regimes do not constitute arbitrary-dependence evidence.",
                artifact_paths=("output/results.json", "output/figures/dependence_sweep.png"),
                citations=("benjamini2001control", "dobriban2026bh"),
            ),
            _entry(
                "imported-two-sided-non-prds-protocol",
                "generative_process",
                "Imported two-sided non-PRDS stress protocol",
                "X_g = μ_g + ρ_g Z + sqrt(1−ρ_g²)ε_g; p_g = 2Φ(−|X_g|)",
                ("protocols.load_dobriban_bh_protocol", "protocols.run_dobriban_bh_chunk", "protocols.DobribanBHResult"),
                ("src/active_inference_power/protocols/dobriban_bh.py",),
                ("tests/test_dobriban_protocol.py",),
                ("the imported plan is unchanged and checksum-pinned", "PCG64 and child-stream construction match the publicly documented protocol", "finite Monte Carlo remains separate from the external Arb certificate"),
                "stress_test",
                "The package can rerun controlled finite chunks of the publicly documented two-sided non-PRDS stress protocol offline.",
                "The adapter does not reproduce the external outward-rounded theorem or establish arbitrary-dependence behavior.",
                artifact_paths=("output/results.json", "output/formalism_registry.json"),
                citations=("dobriban2026bh",),
            ),
            _entry(
                "time-uniform-evidence",
                "evidence",
                "Likelihood-ratio e-process and Ville boundary",
                "E_t = ∏_{j≤t} f_1(X_j)/f_0(X_j), with P_0(sup_t E_t ≥ 1/α) ≤ α",
                ("sequential.binary_e_process", "sequential.mixture_e_process", "sequential.EProcessResult"),
                ("src/active_inference_power/sequential.py",),
                ("tests/test_sequential.py", "tests/test_branch_contracts.py"),
                ("null likelihood is correctly specified", "updates are adapted to the visible filtration", "the e-process remains nonnegative"),
                "formal_conditional",
                "The declared Bernoulli likelihood-ratio process has an optional-stopping interpretation under its null contract.",
                "The result does not validate optional stopping for fixed-horizon p-values or arbitrary action-dependent processes.",
                artifact_paths=("output/results.json", "output/figures/sequential_evidence.png"),
                citations=("shafer2011test_martingales", "grunwald2024safe", "howard2020time_uniform"),
            ),
            _entry(
                "action-e-process-channel-contract",
                "evidence",
                "Action-loop e-process channel contract",
                "E_t = ∏_{j≤t} f_1(X_j | H_{j-1}) / f_0(X_j | H_{j-1}), with fixed Bernoulli rates within each target/truth cell",
                ("action_loop._e_process_contract_valid", "action_loop.run_action_loop"),
                ("src/active_inference_power/action_loop.py",),
                ("tests/test_action_loop_suite.py", "tests/test_branch_contracts.py"),
                ("truth transitions are absent", "selection is predictable", "model and process emissions agree", "null and alternative rates are state-invariant within each target"),
                "formal_conditional",
                "The action-loop e-process receives an optional-stopping label only under the stronger channel contract.",
                "Static truth and chronological selection alone do not validate a fixed Bernoulli factor under context-dependent emissions.",
                artifact_paths=("output/action_loop_suite.json", "output/figures/action_loop_trace.png"),
                citations=("shafer2011test_martingales", "grunwald2024safe"),
            ),
            _entry(
                "online-fdr-recursions",
                "decision",
                "LORD++, SAFFRON, and e-LOND online evidence accounting",
                "α_t and online accounting paths are predictable recursions over candidate and rejection history; LORD++/SAFFRON use procedural alpha-wealth W_t^α, while e-LOND reports cumulative nominal-level allocation rather than a remaining budget",
                ("online_fdr.default_gamma", "online_fdr.lord_plus_plus", "online_fdr.saffron", "online_fdr.e_lond"),
                ("src/active_inference_power/online_fdr.py",),
                ("tests/test_online_fdr.py", "tests/test_action_loop_suite.py"),
                ("gamma is positive and normalized", "test order is chronological", "candidate and rejection events are recorded separately", "e-values are conditionally calibrated in the arrival filtration"),
                "formal_conditional",
                "Known-vector recursions and finite null diagnostics expose levels, candidates, rejections, and a procedure-specific accounting path; e-LOND's path is cumulative emitted nominal level, not wealth or a remaining budget.",
                "The package does not claim a new online-FDR theorem or validity after unverified adaptive selection.",
                artifact_paths=("output/results.json", "output/figures/online_fdr_trace.png"),
                citations=(
                    "javanmard2018online",
                    "ramdas2017online_memory",
                    "ramdas2018saffron",
                    "xu2024online_evalues",
                ),
            ),
            _entry(
                "action-filtration-and-policy",
                "decision",
                "Action-in-the-loop filtration and policy trace",
                "A_t = π(H_t), where H_t contains only observations and prior visible actions",
                ("action_loop.run_action_loop", "action_loop.ActionLoopStep", "action_loop.ActionLoopTrace"),
                ("src/active_inference_power/action_loop.py", "src/active_inference_power/action_statistics.py"),
                ("tests/test_action_loop_suite.py", "tests/test_branch_contracts.py"),
                ("the policy cannot see hidden state truth", "cost and stopping are scored at replication level", "hidden evaluation state is a separate diagnostic channel"),
                "conditional_operating_characteristic",
                "Action policies are evaluated by paired power, error, cost, stopping, and target-selection estimands.",
                "A policy frontier is not an optimality theorem, utility theorem, or family-level validity result.",
                artifact_paths=("output/action_loop_suite.json", "output/figures/action_loop_trace.png", "output/figures/action_policy_frontier.png"),
                citations=("friston2016active_learning", "parr2019generalised", "fithian2014selective"),
            ),
            _entry(
                "split-stream-confirmation",
                "decision",
                "Independent split-stream selection and confirmation",
                "S belongs to F_sel and P_sel is independent of P_conf given S; p_conf is conditionally super-uniform under H0",
                ("selection.SelectionPolicySpec", "selection.FiltrationEvent", "selection.run_split_stream_confirmation", "selection.SplitConfirmationResult"),
                ("src/active_inference_power/selection.py", "src/active_inference_power/suite.py"),
                ("tests/test_selection.py", "tests/test_action_loop_suite.py"),
                ("selection uses visible history only", "confirmation seed is independent of selection seed", "the confirmation null law is correctly specified"),
                "formal_conditional",
                "Split-stream confirmation makes the adaptive-selection filtration and independent confirmation data explicit.",
                "The contract does not prove validity for a misspecified process, same-stream selection, or arbitrary selective procedure.",
                artifact_paths=("output/action_loop_suite.json", "output/figures/conditional_selection_calibration.png"),
                citations=("fithian2014selective", "ramdas2017online_memory"),
            ),
            _entry(
                "confidence-sequence-inversion",
                "evidence",
                "Bernoulli mixture confidence sequence",
                "C_t = {q : E_t(q) < 1/α} for a Beta-mixture likelihood-ratio e-process",
                ("sequential.bernoulli_confidence_sequence", "sequential.ConfidenceSequenceResult"),
                ("src/active_inference_power/sequential.py",),
                ("tests/test_sequential.py",),
                ("the Bernoulli null is fixed for each candidate q", "the mixture e-process is nonnegative", "the interval is interpreted simultaneously over time"),
                "formal_conditional",
                "The package exposes a time-uniform Bernoulli confidence sequence by e-process inversion.",
                "The sequence is not a general confidence sequence for arbitrary emissions or hidden-state processes.",
                artifact_paths=("output/results.json", "output/figures/sequential_evidence.png"),
                citations=("howard2020time_uniform", "howard2021confidence_sequences"),
            ),
            _entry(
                "held-out-learned-policy",
                "decision",
                "Frozen learned-policy evaluation",
                "π̂ = Train(D_train); evaluate π̂ on D_eval independent of D_train",
                ("learned_policy.LearnedPolicyArtifact", "learned_policy.train_tabular_policy", "learned_policy.evaluate_learned_policy"),
                ("src/active_inference_power/learned_policy.py",),
                ("tests/test_learned_policy.py",),
                ("training and evaluation child streams are disjoint", "policy parameters are frozen and hashed", "the policy sees visible features only", "the search configuration is hashed and candidate-space complexity is recorded"),
                "conditional_operating_characteristic",
                "Held-out learned-policy power and cost are reported for the declared scenario.",
                "Held-out improvement is not a policy optimality theorem or universal active-inference guarantee.",
                artifact_paths=("output/action_loop_suite.json", "output/figures/action_policy_frontier.png"),
                citations=("friston2016active_learning",),
            ),
            _entry(
                "testing-setting-sensitivity-surface",
                "testing_setting",
                "Testing-setting sensitivity surface",
                "g(α, T, B, I, c, π) = (power, FDR, FWER, cost, stopping) under one declared interface cell",
                (
                    "testing_setting_sensitivity.TestingSettingSensitivityPlan",
                    "testing_setting_sensitivity.evaluate_testing_setting_sensitivity",
                ),
                ("src/active_inference_power/testing_setting_sensitivity.py",),
                ("tests/test_testing_setting_sensitivity.py",),
                (
                    "the factorial grid is fixed before execution",
                    "common child streams are shared only within comparable interface cells",
                    "fixed-family corrections and action-loop decisions remain separate estimands",
                ),
                "conditional_operating_characteristic",
                "The package can report separate finite-simulation operating characteristics across declared testing settings.",
                "The surface does not produce a pooled utility score, a universal policy ranking, or a theorem outside the declared model and testing assumptions.",
                artifact_paths=("output/research-atlas/research_atlas.json",),
                citations=("benjamini1995controlling", "morris2019simulation"),
            ),
            _entry(
                "stratified-split-stream-calibration",
                "decision",
                "Stratified independent split-stream calibration audit",
                "P_conf | (target, visible selection history) is audited under an all-null static confirmation channel",
                (
                    "scenario.make_static_null_binary_e_process_scenario",
                    "selection_sensitivity.SplitStreamSensitivityPlan",
                    "selection_sensitivity.evaluate_split_stream_sensitivity",
                ),
                (
                    "src/active_inference_power/scenario.py",
                    "src/active_inference_power/selection_sensitivity.py",
                ),
                ("tests/test_selection_sensitivity.py", "tests/test_selection.py"),
                (
                    "the process is all-null and the static confirmation channel is correctly specified",
                    "selection sees only the declared visible filtration",
                    "selection and confirmation child streams are independent",
                    "selection-step sensitivity holds all other tuning fixed",
                ),
                "calibration_diagnostic",
                "Target- and visible-history-stratified finite calibration diagnostics are recorded for a predeclared split-stream design.",
                "A DKW or empirical-CDF pass is not a selective-inference theorem, nor does it validate same-stream selection or a misspecified process.",
                artifact_paths=("output/research-atlas/research_atlas.json",),
                citations=("fithian2014selective", "ramdas2017online_memory"),
            ),
            _entry(
                "paired-dynamic-process-sensitivity",
                "generative_process",
                "Paired dynamic-process stress controls",
                "Δ_g = n⁻¹∑ᵣ[g_r(dynamic) − g_r(static control)] over declared schedules and missingness regimes",
                (
                    "dynamic_process_sensitivity.DynamicProcessSensitivityPlan",
                    "dynamic_process_sensitivity.dynamic_process_pairs",
                    "dynamic_process_sensitivity.evaluate_dynamic_process_sensitivity",
                ),
                ("src/active_inference_power/dynamic_process_sensitivity.py",),
                ("tests/test_dynamic_process_sensitivity.py", "tests/test_dynamic_process.py"),
                (
                    "each dynamic cell has a paired static control",
                    "schedule extrapolation and missingness mechanisms are explicit",
                    "paired traces share declared common random numbers",
                    "marginal calibration remains a finite diagnostic",
                ),
                "stress_test",
                "The package compares persistent context, drift, action transitions, and missingness regimes against paired static controls.",
                "Dynamic-process stress results do not establish a formal dynamic-process law, optional-stopping guarantee, or universal robustness claim.",
                artifact_paths=("output/research-atlas/research_atlas.json",),
                citations=("morris2019simulation",),
            ),
            _entry(
                "sequential-evidence-comparison-atlas",
                "evidence",
                "Sequential e-process comparison atlas",
                "Compare simple, mixture, and predictable e-processes on shared Bernoulli streams with confidence-sequence and e-LOND diagnostics",
                (
                    "sequential_atlas.SequentialEvidenceAtlasConfig",
                    "sequential_atlas.evaluate_sequential_evidence_atlas",
                ),
                ("src/active_inference_power/sequential_atlas.py",),
                ("tests/test_sequential_atlas.py", "tests/test_sequential.py"),
                (
                    "Bernoulli rates and evidence families are predeclared",
                    "each evidence family receives the same condition-specific child stream",
                    "predictability and e-LOND filtration conditions remain explicit",
                    "repeated fixed-horizon p-value peeking is retained only as an invalid contrast",
                ),
                "calibration_diagnostic",
                "Finite crossing, coverage, e-LOND-accounting, and stopping summaries are available for declared Bernoulli conditions.",
                "Observed rates do not prove Ville, confidence-sequence, predictable-evidence, or online-FDR theorems outside their assumptions.",
                artifact_paths=("output/research-atlas/research_atlas.json",),
                citations=(
                    "shafer2011test_martingales",
                    "howard2021confidence_sequences",
                    "ramdas2017online_memory",
                    "xu2024online_evalues",
                ),
            ),
            _entry(
                "repeated-held-out-policy-partitions",
                "decision",
                "Repeated train/freeze/evaluate policy partitions",
                "π̂_j = Train(D_train,j); Δ_j = g(π̂_j, D_eval,j) − g(π_ref, D_eval,j) for disjoint streams j",
                (
                    "repeated_policy_evaluation.RepeatedLearnedPolicyPlan",
                    "repeated_policy_evaluation.evaluate_repeated_learned_policy",
                ),
                ("src/active_inference_power/repeated_policy_evaluation.py",),
                ("tests/test_repeated_policy_evaluation.py", "tests/test_learned_policy.py"),
                (
                    "training and evaluation branches are disjoint and materialized",
                    "the finite candidate search and every frozen policy hash are recorded",
                    "reference and learned policies share held-out seeds within each partition",
                    "partition-level bands do not pool changing fitted policies as one fixed policy",
                ),
                "conditional_operating_characteristic",
                "Repeated independent train/evaluate partitions report held-out learned-minus-reference power, cost, and stopping bands.",
                "These conditional comparisons do not establish optimality, universal benefit, causal effects, or family-level validity.",
                artifact_paths=("output/research-atlas/research_atlas.json",),
                citations=("friston2016active_learning", "morris2019simulation"),
            ),
            _entry(
                "multi-target-arrival-filtration",
                "generative_process",
                "Multi-target arrival and independent-confirmation environment",
                "H_t contains arrived targets, visible actions and observations, reliability and cost schedules; truth and confirmation data remain hidden",
                (
                    "multi_target_environment.MultiTargetEnvironmentSpec",
                    "multi_target_environment.run_multi_target_environment",
                    "multi_target_environment.evaluate_multi_target_control_pair",
                ),
                ("src/active_inference_power/multi_target_environment.py",),
                ("tests/test_multi_target_environment.py",),
                (
                    "target truth and confirmation observations are excluded from the policy-visible history",
                    "arrival, reliability, cost, and sensing-action contracts are versioned",
                    "confirmation streams are independent child streams",
                    "dynamic environment results are paired with a static control",
                ),
                "stress_test",
                "The package exposes an additive, versioned multi-target environment with visible-filtration and paired-control diagnostics.",
                "The environment contract is not a formal filtration proof, a universal adaptive-confirmation guarantee, or an expanded active-inference claim.",
                artifact_paths=("output/research-atlas/research_atlas.json",),
                citations=("fithian2014selective", "friston2017active_process"),
            ),
            _entry(
                "certificate-gate-policy-partition",
                "artifact",
                "Exact terminal and diagnostic certificate partition",
                "C = T ⊔ D; terminal_pass = ∧_{c∈T} pass(c); all_pass = ∧_{c∈C} pass(c)",
                (
                    "certificate_gate.normalize_certificate_gate_policy",
                    "certificate_gate.certificate_gate_result",
                    "certificate_gate.validate_certificate_gate_result",
                ),
                ("src/active_inference_power/certificate_gate.py",),
                ("tests/test_certificate_gate.py", "tests/test_campaign_finalizer.py"),
                (
                    "the policy exactly partitions the currently emitted certificate names",
                    "all individual verdicts and both partitions are serialized in every explicit witness",
                    "the configuration source is hash-bound to the run before finalization",
                ),
                "provenance_contract",
                "The package records an exact, fail-closed terminal gate while retaining every diagnostic certificate verdict.",
                "A diagnostic classification is not a validity theorem, an erased failure, or a release authorization outside the declared lane.",
                artifact_paths=("output/campaign-v2/results.json", "output/campaign-v2/manifest.json"),
            ),
            _entry(
                "frozen-three-way-storey-method-review",
                "evidence",
                "Frozen three-way Storey method review",
                "g_j = (FDR_j, FWER_j, power_j) for same-vector plug-in, conservative same-vector, and independent confirmation rules on declared streams",
                (
                    "storey_method_review.StoreyMethodReviewPlan",
                    "storey_method_review.evaluate_storey_method_review",
                    "storey_method_review.validate_storey_method_review",
                ),
                (
                    "src/active_inference_power/storey_method_review.py",
                    "scripts/run_storey_method_review.py",
                ),
                (
                    "tests/test_storey_method_review.py",
                    "tests/test_run_storey_method_review.py",
                ),
                (
                    "the independent two-groups design, lambda, confidence, seed, and replication count are frozen before execution",
                    "the same-vector methods share identical p-value and truth draws",
                    "split-confirmation decisions use a distinct deterministic confirmation stream",
                    "finite Monte-Carlo bands describe only the declared simulation regime",
                ),
                "calibration_diagnostic",
                "The package can produce a hash-bound, three-way finite operating-characteristic review with explicit common-random-number and independent-stream witnesses.",
                "The review does not prove arbitrary-dependence FDR control, tune a threshold after inspection, alter certificate policy, or promote a release.",
                artifact_paths=("output/storey-method-review/storey_method_review.json",),
            ),
            _entry(
                "offline-benchmark-preflight",
                "artifact",
                "Offline checksum and provenance benchmark preflight",
                "manifest record → checksum-verified fixture → preprocessing lineage → chronological split and missingness report",
                (
                    "benchmark_preflight.OfflineBenchmarkPreflight",
                    "benchmark_preflight.preflight_offline_benchmark_fixtures",
                ),
                ("src/active_inference_power/benchmark_preflight.py",),
                ("tests/test_benchmark_preflight.py", "tests/test_artifacts.py"),
                (
                    "all fixture reads are offline and checksum verified",
                    "the Golub archive-to-fixture recipe is declared",
                    "trajectory subjects are chronologically split without overlap",
                    "external outputs retain a stress-only claim boundary",
                ),
                "provenance_contract",
                "The package can produce an offline provenance dashboard for the committed external stress fixtures.",
                "Metadata, checksums, and chronological leakage checks do not establish biological ground truth, external validity, or a synthetic theorem.",
                artifact_paths=("output/benchmark_preflight/provenance_dashboard.json",),
                citations=("wilkinson2016fair",),
            ),
            _entry(
                "monte-carlo-and-paired-uncertainty",
                "uncertainty",
                "Replication-level uncertainty and common-random-number contrasts",
                "estimate = n⁻¹∑ᵣ g_r; paired contrast = n⁻¹∑ᵣ(gᵣ,A − gᵣ,B)",
                ("uncertainty.wilson_interval", "uncertainty.clopper_pearson_interval", "uncertainty.summarize_bounded", "uncertainty.paired_numeric_difference_summary"),
                ("src/active_inference_power/uncertainty.py", "src/active_inference_power/action_statistics.py"),
                ("tests/test_uncertainty.py", "tests/test_action_loop_suite.py", "tests/test_calibration.py"),
                ("replications are independent child streams", "interval conventions are declared", "simultaneous bands account for the configured comparison grid"),
                "calibration_diagnostic",
                "Reported operating characteristics include finite-simulation intervals and paired policy contrasts.",
                "Monte-Carlo precision intervals are not population guarantees and cannot repair a misspecified estimand.",
                artifact_paths=("output/results.json", "output/figures/paired_method_contrasts.png", "output/figures/action_policy_frontier.png"),
                citations=("howard2021confidence_sequences",),
            ),
            _entry(
                "artifact-and-claim-lineage",
                "artifact",
                "Schema, hash, and claim-ledger lineage",
                "formal object → code symbol → test → artifact → claim boundary",
                ("schema.validate_results", "schema.validate_manifest", "schema.validate_run_log", "schema.validate_checkpoint"),
                ("src/active_inference_power/schema.py", "scripts/artifacts.py", "scripts/run_ledger.py"),
                ("tests/test_artifacts.py", "tests/test_scholarship.py", "tests/test_scripts.py"),
                ("artifacts are written atomically", "hashes are computed after generation", "the manifest is written last"),
                "provenance_contract",
                "The release makes formal-to-code-to-artifact links machine-readable and auditable.",
                "Provenance and reproducibility metadata do not strengthen statistical assumptions or convert diagnostics into theorems.",
                artifact_paths=("output/formalism_registry.json", "output/manifest.json", "data/claim_ledger.yaml"),
                citations=("wilkinson2016fair",),
            ),
        )
    )


def validate_formalism_registry(
    value: Mapping[str, Any], *, project_root: Path | None = None
) -> None:
    """Fail closed on malformed, stale, or tampered registry artifacts.

    ``project_root`` exists for isolated release-bundle validation.  The
    executable registry names source and test files as part of its traceability
    contract, so those declarations must resolve against the checkout that is
    producing or reviewing the artifact.
    """
    if not isinstance(value, Mapping):
        raise ValueError("formalism registry must be a mapping")
    if value.get("schema_version") != "2.0":
        raise ValueError("formalism registry schema must be 2.0")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("formalism registry entries must be a non-empty list")
    expected = formalism_registry()
    expected_mapping = expected.to_mapping()
    if value.get("registry_hash") != expected_mapping["registry_hash"]:
        raise ValueError("formalism registry hash does not match the executable registry")
    if value.get("entry_count") != len(entries) or entries != expected_mapping["entries"]:
        raise ValueError("formalism registry entries do not match the executable registry")
    root = _PROJECT_ROOT if project_root is None else Path(project_root)
    if not root.is_dir():
        raise ValueError(f"formalism project root does not exist: {root}")
    for entry in expected.entries:
        for path in entry.source_paths:
            if not (root / path).is_file():
                raise ValueError(f"formalism source path does not exist: {path}")
        for path in entry.test_paths:
            if not (root / path).is_file():
                raise ValueError(f"formalism test path does not exist: {path}")
        for symbol in entry.code_symbols:
            resolve_code_symbol(symbol)


def _reader_identifier(identifier: str) -> str:
    """Convert a registry identifier into a compact manuscript display label."""
    acronym = {"ai": "AI", "bh": "BH", "fdr": "FDR", "fwer": "FWER"}
    return " ".join(
        acronym.get(token, token)
        for token in identifier.split("_")
        if token
    )


def formalism_rows(value: Mapping[str, Any]) -> str:
    """Render a compact manuscript table from a validated registry artifact."""
    validate_formalism_registry(value)
    rows = []
    for entry in value["entries"]:
        module, symbol = entry["code_symbols"][0].split(".", 1)
        # The registry retains exact public symbols. The manuscript table is a
        # reader-facing lineage aid, so split snake-case identifiers into words
        # to create stable line-breaking opportunities in PDF and HTML output.
        display_symbol = _reader_identifier(symbol)
        anchor = f"{_reader_identifier(module)} / {display_symbol}"
        if len(entry["code_symbols"]) > 1:
            anchor += f" (+{len(entry['code_symbols']) - 1} symbols)"
        evidence = _reader_identifier(entry["evidence_class"])
        rows.append(
            f"| {_reader_identifier(entry['layer'])} | {entry['title']} | {anchor}; {evidence} |"
        )
    return "\n".join(rows)


__all__ = [
    "EVIDENCE_CLASSES",
    "LAYERS",
    "FormalismEntry",
    "FormalismRegistry",
    "formalism_registry",
    "resolve_code_symbol",
    "validate_formalism_registry",
    "formalism_rows",
]
