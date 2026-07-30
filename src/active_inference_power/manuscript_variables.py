"""Manuscript variable generation for active_inference_power.

Single source of truth for every number that appears in the manuscript. Reads
``manuscript/config.yaml`` (experiment parameters) and ``output/results.json``
(computed results from :mod:`simulation`, :mod:`certificates`, and the
active-inference experiment) and returns a flat ``dict[str, str]`` of
``UPPERCASE_KEY -> value`` for ``{{TOKEN}}`` substitution via
:func:`infrastructure.rendering.manuscript_injection.write_resolved_manuscript_tree`.

Deterministic package domain code: no ``infrastructure.*`` import. Called by the thin
orchestrator ``scripts/z_generate_manuscript_variables.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

import yaml

from .certificates import BAND_SIGMA
from .figures import FIGURE_SPECS, figure_alts, figure_captions
from .formalism import formalism_rows

_MISSING = "N/A"
_GITHUB_REPOSITORY_RE = re.compile(r"^[^/\s]+/[^/\s]+$")
_ZENODO_DOI_RE = re.compile(r"^10\.5281/zenodo\.[1-9][0-9]*$")

# Display names for the correction procedures, in report order.
_PROC_LABELS = {
    "bonferroni": "Bonferroni",
    "sidak": "Šidák",
    "holm": "Holm",
    "hochberg": "Hochberg",
    "benjamini_hochberg": "Benjamini--Hochberg",
    "benjamini_yekutieli": "Benjamini--Yekutieli",
    "storey_qvalue": "Storey q-value",
    "adaptive_bh": "Adaptive BH (BKY)",
    "weighted_bh": "Weighted BH",
}

_ACTION_POLICY_LABELS = {
    "fixed_horizon": "Fixed horizon",
    "posterior_threshold": "Posterior threshold",
    "information_gain": "Information gain",
    "cost_aware": "Cost aware",
    "posterior_sampling": "Posterior sampling",
    "e_process": "E-process",
}

# The action-loop operating-characteristics table has ten columns.  These
# labels preserve the decision-rule meaning while avoiding automatic
# hyphenation of long words inside its deliberately narrow policy column.  The
# surrounding caption maps each compact label back to the declared policy; the
# full identifiers remain in the result artifact and traces.
_ACTION_POLICY_TABLE_LABELS = {
    "fixed_horizon": "Fixed-$T$",
    "posterior_threshold": "Posterior cutoff",
    "information_gain": "Info gain",
    "cost_aware": "Cost rule",
    "posterior_sampling": "Posterior draw",
    "e_process": "$E_t$ rule",
}


def _load_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "manuscript" / "config.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_output_root(project_root: Path, output_root: Path | None = None) -> Path:
    """Resolve the output namespace used by one hydration run."""
    output = project_root / "output" if output_root is None else Path(output_root)
    if not output.is_absolute():
        output = project_root / output
    return output.resolve()


def _load_results(project_root: Path, output_root: Path | None = None) -> dict[str, Any]:
    path = _resolve_output_root(project_root, output_root) / "results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_quality(project_root: Path, output_root: Path | None = None) -> dict[str, Any]:
    path = _resolve_output_root(project_root, output_root) / "quality.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _num(x: float, places: int) -> str:
    """Format a float with fixed decimal places (no thousands separators)."""
    return f"{float(x):.{places}f}"


def _probability_or_not_estimable(value: object, places: int) -> str:
    """Render a finite conditional probability or its explicit finite-run boundary."""
    if value is None:
        return "not estimable"
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError("conditional probability must be a finite number or None")
    return _num(value, places)


def _publication_repository_statement(config: dict[str, Any]) -> str:
    """Return a repository-only abstract sentence after fail-closed validation.

    The abstract needs a useful source-materials statement without exposing
    deposit workflow or DOI status.  Canonical publication metadata is still
    validated here because the same variable generator produces both draft and
    DOI-bearing manuscripts.  A configured DOI therefore cannot conceal a
    wrong repository, a malformed Zenodo identifier, or an incomplete archive
    record merely because the abstract intentionally does not display it.
    """
    raw_publication = config.get("publication", {})
    publication = raw_publication if isinstance(raw_publication, dict) else {}
    planned = str(publication.get("planned_github_repository", "")).strip()
    planned_url = str(publication.get("planned_repository_url", "")).strip()
    repository = str(publication.get("github_repository", "")).strip()
    repository_url = str(publication.get("repository_url", "")).strip()
    concept_doi = str(publication.get("doi", "")).strip()
    version_doi = str(publication.get("version_doi", "")).strip()
    version_record = str(publication.get("version_record", "")).strip()
    deposition_id = str(publication.get("zenodo_deposition_id", "")).strip()
    doi_status = str(publication.get("doi_status", "")).strip()
    publication_status = str(publication.get("status", "")).strip()
    release_tag = str(publication.get("github_release_tag", "")).strip()
    release_url = str(publication.get("github_release_url", "")).strip()

    if not planned:
        if any((repository, repository_url, concept_doi, version_doi, version_record)):
            raise ValueError(
                "publication metadata cannot declare identifiers without a planned repository"
            )
        return (
            "The intended public source repository will be designated before "
            "publication."
        )
    if not _GITHUB_REPOSITORY_RE.fullmatch(planned):
        raise ValueError(
            "publication.planned_github_repository must use owner/repository form"
        )
    expected_repository_url = f"https://github.com/{planned}"
    if planned_url != expected_repository_url:
        raise ValueError(
            "publication.planned_repository_url must equal the canonical GitHub URL"
        )
    if doi_status != "to be assigned at publication":
        raise ValueError(
            "publication.doi_status must be exactly 'to be assigned at publication'"
        )
    if not concept_doi:
        if publication_status != "draft":
            raise ValueError("draft publication metadata must have status 'draft'")
        if any((repository, repository_url, version_doi, version_record, deposition_id, release_tag, release_url)):
            raise ValueError(
                "draft publication metadata must not declare a live repository, release, archive record, or DOI"
            )
        return (
            "At publication, source materials are intended for the GitHub repository:  \n"
            f"`{planned}`."
        )
    if (
        repository != planned
        or repository_url != expected_repository_url
        or not _ZENODO_DOI_RE.fullmatch(concept_doi)
        or not _ZENODO_DOI_RE.fullmatch(version_doi)
        or deposition_id != version_doi.rsplit(".", 1)[-1]
        or version_record
        != f"https://zenodo.org/records/{version_doi.rsplit('.', 1)[-1]}"
    ):
        raise ValueError(
            "published metadata requires the planned repository, canonical GitHub "
            "URL, and a complete distinct Zenodo DOI pair"
        )
    if concept_doi == version_doi:
        raise ValueError("Zenodo concept and version DOI values must differ")
    paper = config.get("paper", {})
    paper_version = str(paper.get("version", "")).strip() if isinstance(paper, dict) else ""
    expected_release_tag = f"v{paper_version}" if paper_version else ""
    expected_release_url = f"{expected_repository_url}/releases/tag/{expected_release_tag}"
    if (
        publication_status != "published"
        or not paper_version
        or release_tag != expected_release_tag
        or release_url != expected_release_url
    ):
        raise ValueError(
            "published metadata requires status 'published' and the canonical v<paper.version> GitHub release URL"
        )
    return (
        "Source and release materials are available at the verified GitHub "
        f"repository `{repository}`."
    )


def _publication_release_identity(config: dict[str, Any]) -> dict[str, str]:
    """Return validated publication identifiers for non-abstract manuscript text.

    The cover presents the concept DOI, while this generated identity block
    exposes the immutable archive record and source tag. This keeps the three
    identifiers synchronized with the fail-closed publication configuration.
    """
    _publication_repository_statement(config)
    raw_publication = config.get("publication", {})
    publication = raw_publication if isinstance(raw_publication, dict) else {}
    return {
        "PUBLICATION_CONCEPT_DOI": str(publication.get("doi", "")).strip() or _MISSING,
        "PUBLICATION_VERSION_DOI": str(publication.get("version_doi", "")).strip() or _MISSING,
        "PUBLICATION_VERSION_RECORD": str(publication.get("version_record", "")).strip() or _MISSING,
        "PUBLICATION_GITHUB_REPOSITORY": str(publication.get("github_repository", "")).strip() or _MISSING,
        "PUBLICATION_REPOSITORY_URL": str(publication.get("repository_url", "")).strip() or _MISSING,
        "PUBLICATION_GITHUB_RELEASE_TAG": str(publication.get("github_release_tag", "")).strip() or _MISSING,
        "PUBLICATION_GITHUB_RELEASE_URL": str(publication.get("github_release_url", "")).strip() or _MISSING,
    }


def _comparison_rows(mc: dict[str, Any]) -> str:
    rows = []
    for key, label in _PROC_LABELS.items():
        v = mc.get(key)
        if v is None:
            continue
        rows.append(
            f"| {label} | {_num(v['fdr'], 4)} | {_num(v['power'], 3)} | "
            f"{_num(v['fwer'], 3)} | {_num(v['mean_rejections'], 1)} |"
        )
    return "\n".join(rows)


def _powercurve_rows(pc: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"| {_num(p['effect'], 1)} | {_num(p['power'], 3)} | {_num(p['fdr'], 3)} |"
        for p in pc
    )


def _dependence_rows(ds: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"| {d.get('regime', d.get('dependence_kind', 'declared'))} | {_num(d['rho'], 2)} | {_num(d['bh_fdr'], 4)} | {_num(d['bh_power'], 3)} | "
        f"{_num(d['by_fdr'], 4)} | {_num(d['by_power'], 3)} |"
        for d in ds
    )


def _cert_rows(certs: dict[str, Any]) -> str:
    order = [
        ("bh_fdr_control", "BH FDR control", "$\\pi_0\\alpha$"),
        ("bonferroni_fwer", "Bonferroni FWER", "$\\alpha$"),
        ("positive_dependence", "BH under PRDS", ""),
        ("storey_power_gain", "Storey power gain", "BH power"),
        ("storey_split_confirmation", "Split-stream Storey calibration", "$\\alpha$"),
    ]
    rows = []
    for key, label, bound_name in order:
        c = certs.get(key)
        if c is None:
            continue
        est = c.get("estimate", c.get("storey_power"))
        mark = "✓" if c.get("passes") else "✗"
        direction = "≤" if c.get("bound_direction", "upper") == "upper" else "≥"
        bound_txt = f"{direction} {_num(c['bound'], 4)}"
        if bound_name:
            bound_txt += f" ({bound_name})"
        rows.append(f"| {label} | {_num(est, 4)} | {bound_txt} | {mark} |")
    return "\n".join(rows)


def _storey_audit_rows(cert: dict[str, Any]) -> str:
    """Render the adaptive-FDR certificate with metric-specific uncertainty.

    The top-level certificate ``stderr`` historically described Storey power.
    This table deliberately exposes separate FDR and power uncertainty so a
    reader cannot mistake a precise power estimate for an FDR calibration band.
    """
    rows = [
        (
            "BH FDR",
            cert["bh_fdr"],
            cert["bh_fdr_stderr"],
            cert["bh_fdr_upper_band"],
            "passed" if cert["bh_fdr_passes"] else "failed",
        ),
        (
            "Storey FDR",
            cert["storey_fdr"],
            cert["storey_fdr_stderr"],
            cert["storey_fdr_upper_band"],
            "passed" if cert["storey_fdr_passes"] else "failed",
        ),
        (
            "BH power",
            cert["bh_power"],
            cert["bh_power_stderr"],
            "reference",
            "reference",
        ),
        (
            "Storey power",
            cert["storey_power"],
            cert["stderr"],
            cert["bh_power"],
            "passed" if cert["power_gain_passes"] else "failed",
        ),
        (
            "Paired power gain",
            cert["power_gain"],
            cert["power_gain_stderr"],
            "0.0000",
            "passed" if cert["power_gain_passes"] else "failed",
        ),
    ]
    rendered: list[str] = []
    for metric, estimate, stderr, ceiling, status in rows:
        estimate_text = estimate if isinstance(estimate, str) else _num(estimate, 4)
        stderr_text = stderr if isinstance(stderr, str) else _num(stderr, 6)
        ceiling_text = ceiling if isinstance(ceiling, str) else _num(ceiling, 4)
        rendered.append(
            f"| {metric} | {estimate_text} | {stderr_text} | {ceiling_text} | {status} |"
        )
    return "\n".join(rendered)


def _ai_rows(ai: dict[str, Any]) -> str:
    regimes = ai.get("regimes", ai)
    s, w = regimes["strong_evidence"], regimes["weak_evidence"]
    sc, wc = s["config"], w["config"]
    return "\n".join(
        [
            f"| Strong ($r{{=}}{_num(sc['reliability'], 2)}$, $T{{=}}{int(sc['horizon'])}$) "
            f"| Posterior threshold $\\ge {_num(sc['threshold'], 1)}$ "
            f"| {_num(s['posterior_threshold_fdr'], 3)} | {_num(s['posterior_threshold_power'], 3)} | {_num(s['posterior_threshold_fwer'], 3)} |",
            f"| Strong | BH on evidence p-values "
            f"| {_num(s['bh_on_evidence_fdr'], 3)} | {_num(s['bh_on_evidence_power'], 3)} | {_num(s['bh_on_evidence_fwer'], 3)} |",
            f"| Weak ($r{{=}}{_num(wc['reliability'], 2)}$, $T{{=}}{int(wc['horizon'])}$) "
            f"| Posterior threshold $\\ge {_num(wc['threshold'], 1)}$ "
            f"| **{_num(w['posterior_threshold_fdr'], 3)}** | {_num(w['posterior_threshold_power'], 3)} | {_num(w['posterior_threshold_fwer'], 3)} |",
            f"| Weak | BH on evidence p-values "
            f"| {_num(w['bh_on_evidence_fdr'], 3)} | {_num(w['bh_on_evidence_power'], 3)} | {_num(w['bh_on_evidence_fwer'], 3)} |",
        ]
    )


def _sample_size_rows(accounting: dict[str, Any]) -> str:
    labels = {
        "method_comparison": ("Method comparison", "datasets", "hypothesis_draws"),
        "power_curve": ("Power curve", "datasets", "hypothesis_draws"),
        "dependence_sweep": ("Dependence sweep", "datasets", "hypothesis_draws"),
        "paired_comparisons": ("Paired method comparisons", "replications", "paired_differences"),
        "active_inference": ("Active inference", "families", "observation_draws"),
        "sequential": ("Sequential evidence", "replications", "observation_draws"),
        "online_fdr": ("Online FDR", "replications", "pvalue_draws"),
        "agent_sampling": ("Agent sampling traces", "replications", "observation_draws"),
        "conditional_power": ("Conditional action loop", "replications", "action_steps"),
        "null_pvalue_calibration": ("Null calibration", "replications", "pvalues"),
        "certificates": ("Certificates", "checks", "simulation_runs"),
    }
    rows = []
    for key, (label, rep_field, draw_field) in labels.items():
        block = accounting["blocks"][key]
        rows.append(
            f"| {label} | {int(block[rep_field]):,} | {int(block[draw_field]):,} |"
        )
    return "\n".join(rows)


def _complexity_rows(complexity: dict[str, Any]) -> str:
    """Render compact, result-bound source-work rows for the manuscript.

    The release ledger retains detailed implementation scopes and boundaries in
    ``results.json``. This table uses the same rows but keeps the print surface
    scannable: it names the declared dimension, asymptotic source-work label,
    and actual configured proxy without recasting the proxy as elapsed time.
    """
    labels = {
        "single_step_corrections": "Single-step corrections",
        "rank_based_corrections": "Rank-based corrections",
        "fixed_family_simulation": "Fixed-family simulation",
        "comparison_surface": "Procedure comparison surface",
        "dependence_paths": "Dependence paths",
        "binary_e_process": "Binary e-process path",
        "confidence_sequence_inversion": "Confidence-sequence inversion",
        "lord_plus_plus_history_scan": "LORD++ history scan",
        "saffron_history_scan": "SAFFRON history scan",
        "e_lond": "e-LOND path",
        "action_loop_execution_ledger": "Action-loop execution ledger",
    }
    scales = {
        "m": "$m$",
        "B,m": "$B, m$",
        "K,B,m": "$K, B, m$",
        "D,B,m": "$D, B, m$",
        "T": "$T$",
        "T,R": "$T, R$",
        "N_j,H_j": "$N_j, H_j$",
    }
    expected_inputs = {
        "single_step_corrections": "m",
        "rank_based_corrections": "m",
        "fixed_family_simulation": "B,m",
        "comparison_surface": "K,B,m",
        "dependence_paths": "D,B,m",
        "binary_e_process": "T",
        "confidence_sequence_inversion": "T",
        "lord_plus_plus_history_scan": "T,R",
        "saffron_history_scan": "T,R",
        "e_lond": "T",
        "action_loop_execution_ledger": "N_j,H_j",
    }
    expected_orders = {
        "single_step_corrections": "Theta(m)",
        "rank_based_corrections": "Theta(m log2 m)",
        "fixed_family_simulation": "Theta(B[m + selected-correction work at m]) per procedure",
        "comparison_surface": "Theta(K B [m + selected-correction work at m])",
        "dependence_paths": (
            "Two procedure passes: release factor fast path Theta(Bm); cached non-factor paths "
            "O(m^3) setup plus two dense O(Bm^2) draws"
        ),
        "binary_e_process": "Theta(T)",
        "confidence_sequence_inversion": "at least Omega(T^2) prefix scans plus numerical root solves",
        "lord_plus_plus_history_scan": "O(TR), worst O(T^2)",
        "saffron_history_scan": "O(T^2) sparse history; worst O(T^3) dense history",
        "e_lond": "Theta(T)",
        "action_loop_execution_ledger": (
            "Declared trace and step-count accounting; pymdp internals remain unexpanded"
        ),
    }
    accounting = {
        "single_step_corrections": "$\\Theta(m)$",
        "rank_based_corrections": "$\\Theta(m\\log_2 m)$",
        "fixed_family_simulation": (
            "$\\Theta\\!\\bigl(B[m + w_{\\mathrm{corr}}(m)]\\bigr)$ "
            "per procedure"
        ),
        "comparison_surface": (
            "$\\Theta\\!\\bigl(KB[m + w_{\\mathrm{corr}}(m)]\\bigr)$"
        ),
        "dependence_paths": (
            "Two procedure passes: release factor fast path $\\Theta(Bm)$; "
            "cached non-factor paths $O(m^3)$ setup plus two dense $O(Bm^2)$ draws"
        ),
        "binary_e_process": "$\\Theta(T)$",
        "confidence_sequence_inversion": (
            "at least $\\Omega(T^2)$ prefix scans plus numerical root solves"
        ),
        "lord_plus_plus_history_scan": "$O(TR)$; worst case $O(T^2)$",
        "saffron_history_scan": (
            "$O(T^2)$ for sparse history; worst case $O(T^3)$ for dense history"
        ),
        "e_lond": "$\\Theta(T)$",
        "action_loop_execution_ledger": (
            "Declared trace-and-step accounting; package internals remain unexpanded"
        ),
    }
    boundaries = {
        "single_step_corrections": (
            "Not a processor-instruction or cross-hardware claim."
        ),
        "rank_based_corrections": (
            "Does not expand NumPy or SciPy internals."
        ),
        "fixed_family_simulation": (
            "Worker count does not remove retained-draw allocation."
        ),
        "comparison_surface": (
            "Implementation-specific; not a general correction theorem."
        ),
        "dependence_paths": (
            "Includes BH/BY draws after cache; excludes a standalone helper path."
        ),
        "binary_e_process": (
            "Loop count, not e-process evidence $E_t$."
        ),
        "confidence_sequence_inversion": (
            "Root-solver cost remains empirical."
        ),
        "lord_plus_plus_history_scan": (
            "Implementation property, not a LORD++ theorem."
        ),
        "saffron_history_scan": (
            "Implementation property, not a SAFFRON theorem."
        ),
        "e_lond": (
            "Separate from alpha accounting and validity assumptions."
        ),
        "action_loop_execution_ledger": (
            "Not physical runtime, simulated cost $C$, or policy quality."
        ),
    }
    rows: list[str] = []
    for row in complexity["analytical_rows"]:
        name = str(row["name"])
        label = labels.get(name, name.replace("_", " ").capitalize())
        try:
            expected_input = expected_inputs[name]
            expected_order = expected_orders[name]
            if str(row["input_symbol"]) != expected_input:
                raise ValueError(
                    f"unexpected input symbol for {name}: {row['input_symbol']!r}"
                )
            if str(row["time_order"]) != expected_order:
                raise ValueError(
                    f"unexpected source-work order for {name}: {row['time_order']!r}"
                )
            scale = scales[expected_input]
            order = accounting[name]
            boundary = boundaries[name]
        except KeyError as exc:
            raise ValueError(
                "computational-complexity manuscript formatter lacks a reader-facing "
                f"representation for {exc.args[0]!r}"
            ) from exc
        proxy = f"{order}; declared proxy {int(row['work_proxy_at_declared_input']):,}"
        rows.append(f"| {label} | {scale} | {proxy} | {boundary} |")
    return "\n".join(rows)


def _interval_text(interval: Any, places: int = 3) -> str:
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return "N/A"
    return f"[{_num(interval[0], places)}, {_num(interval[1], places)}]"


def _sequential_rows(sequential: dict[str, Any]) -> str:
    e_process = sequential["e_process"]
    pvalue = sequential["fixed_horizon_pvalue_stopping"]
    return "\n".join(
        (
            f"| E-process (Ville-valid) | {_num(e_process['crossing_rate'], 3)} | "
            f"{_interval_text(e_process['crossing_interval_wilson'])} | valid under declared Bernoulli null |",
            f"| Fixed-horizon p-value stopping (contrast) | {_num(pvalue['anytime_rejection_rate'], 3)} | "
            f"{_interval_text(pvalue['anytime_interval_wilson'])} | invalid under optional stopping |",
        )
    )


def _online_fdr_method_label(method: Any) -> str:
    """Return the reader-facing name for a serialized online-FDR method."""
    labels = {"lord_plus_plus": "LORD++", "saffron": "SAFFRON", "e_lond": "e-LOND"}
    value = str(method)
    return labels.get(value, value)


def _online_fdr_rows(online: dict[str, Any]) -> str:
    return "\n".join(
        f"| {_online_fdr_method_label(name)} | {_num(value['mean_rejections'], 3)} | "
        f"{_num(value['any_rejection_rate'], 3)} | {_interval_text(value['any_rejection_interval_wilson'])} |"
        for name, value in online["methods"].items()
    )


def _agent_sampling_rows(agent: dict[str, Any]) -> str:
    regime_labels = {"strong_evidence": "Strong", "weak_evidence": "Weak"}
    policy_labels = {
        "fixed_horizon": "Fixed horizon",
        "posterior_threshold": "Posterior threshold",
        "e_process": "E-process",
    }
    rows = []
    for regime, payload in agent["regimes"].items():
        config = payload["config"]
        for policy, value in payload["policies"].items():
            validity = "Yes" if value["optional_stopping_valid"] else "No"
            rows.append(
                f"| {regime_labels.get(regime, regime)} ($r={_num(config['reliability'], 2)}$, $T={int(config['horizon'])}$) | "
                f"{policy_labels.get(policy, policy)} | {_num(value['decision_rate'], 3)} | "
                f"{_num(value['mean_stopping_time'], 2)} | {validity} |"
            )
    return "\n".join(rows)


def _benchmark_rows(benchmarks: dict[str, Any]) -> str:
    agent = benchmarks["agent_trace"]
    synthetic = benchmarks["synthetic_reference"]
    golub = benchmarks["golub"]
    def role_text(value: Any) -> str:
        return str(value).replace("_", " ")

    return "\n".join(
        (
            f"| Synthetic reference | {int(synthetic['n']):,} | {_num(synthetic['terminal_e'], 3)} | {role_text(synthetic['claim_role'])} |",
            f"| Staged agent trajectory | {int(agent['rows']):,} ({int(agent['subjects'])} subjects) | {_num(agent['terminal_e'], 3)} | {role_text(agent['claim_role'])} |",
            f"| Golub microarray fixture | {'available' if golub['available'] else 'not staged'} ({int(golub.get('fixture_rows', 0)):,} rows) | not estimated | {role_text(golub.get('fixture_claim_role', golub['claim_role']))} |",
        )
    )


def _action_loop_rows(suite: dict[str, Any]) -> str:
    """Render compact, readable scenario/policy operating characteristics.

    Raw scenario identifiers and long validity labels are useful in JSON but
    make a publication table unreadable.  The table therefore uses generated
    reliability labels and a short evidence-class label while retaining the
    full identifiers and boundary text in the surrounding prose and artifacts.
    """
    rows = []
    for item in suite.get("operating_characteristics", []):
        summary = item.get("summary", {})
        metadata = item.get("scenario_metadata", {}) or {}
        reliability = metadata.get("reliability")
        if reliability is not None:
            scenario_label = f"r={float(reliability):.2f}"
        else:
            scenario_label = str(item.get("scenario_id", "N/A"))[:16]
        policy = str(item.get("policy", "N/A"))
        policy_label = _ACTION_POLICY_TABLE_LABELS.get(
            policy, _ACTION_POLICY_LABELS.get(policy, policy.replace("_", " "))
        )
        validity = str(summary.get("validity", "N/A"))
        evidence_class = _evidence_display_label(validity)
        rows.append(
            f"| {scenario_label} | {policy_label} | "
            f"{_num(summary.get('power', 0), 3)} | {_num(summary.get('fdr', 0), 3)} | "
            f"{_num(summary.get('fwer', 0), 3)} | {_num(summary.get('mean_rejections', 0), 2)} | "
            f"{_num(summary.get('power_per_cost', 0), 3)} | {_num(summary.get('mean_cost', 0), 2)} | "
            f"{_num(summary.get('mean_stopping_time', 0), 2)} | "
            f"{evidence_class} |"
        )
    return "\n".join(rows)


def _evidence_display_label(value: object) -> str:
    """Keep machine-readable evidence labels compact in publication tables."""
    text = str(value)
    labels = {
        "formal_conditional_split_confirmation": "formal conditional",
        "conditional_split_confirmation": "conditional",
        "conditional_optional_stopping_claim": "cond.",
        "conditional optional-stopping claim": "cond.",
        "conditional_per_target_optional_stopping": "cond. per-target",
        "conditional per-target optional-stopping claim": "cond. per-target",
        "conditional_online_fdr_diagnostic": "diag.",
    }
    return labels.get(text, "diag." if text.startswith("diagnostic") else text.replace("_", " "))


def _split_confirmation_rows(suite: dict[str, Any]) -> str:
    result = suite.get("split_stream_confirmation") or {}
    if not result:
        return ""
    return (
        f"| Independent split-stream confirmation | {int(result.get('n_reps', 0)):,} | "
        f"{_num(result.get('rejection_rate', 0), 3)} | "
        f"{_num(result.get('true_target_selection_rate', 0), 3)} | "
        f"{_num(result.get('true_target_joint_rejection_rate', 0), 3)} | "
        f"{_probability_or_not_estimable(result.get('power'), 3)} | "
        f"{_num(result.get('fwer', 0), 3)} | "
        f"{_evidence_display_label(result.get('validity', 'N/A'))} |"
    )


def _learned_policy_rows(suite: dict[str, Any]) -> str:
    learned = suite.get("learned_policy") or {}
    evaluation = learned.get("evaluation", {})
    summary = evaluation.get("summary", {})
    artifact = learned.get("artifact", {})
    if not evaluation:
        return ""
    return (
        f"| Held-out learned policy | {int(evaluation.get('evaluation_reps', 0)):,} | "
        f"{_num(summary.get('power', 0), 3)} | {_num(summary.get('mean_cost', 0), 2)} | "
        f"{_num(summary.get('mean_stopping_time', 0), 2)} | {str(artifact.get('policy_hash', 'N/A'))[:12]} |"
    )


def _paired_action_rows(suite: dict[str, Any]) -> str:
    """Render common-random-number policy contrasts for the first scenario."""
    comparisons = suite.get("paired_policy_comparisons", [])
    if not comparisons:
        return ""
    comparison = comparisons[0]
    labels = {
        "true_positive_rate": "Power",
        "false_discovery_proportion": "FDR",
        "family_wise_error": "FWER",
        "total_cost": "Mean cost",
        "step_count": "Stopping time",
        "power_per_cost": "Power / cost",
    }
    rows = []
    for policy, metrics in comparison.get("contrasts", {}).items():
        for metric in ("true_positive_rate", "false_discovery_proportion", "family_wise_error", "total_cost", "step_count", "power_per_cost"):
            value = metrics.get(metric, {})
            rows.append(
                f"| {_ACTION_POLICY_LABELS.get(policy, policy.replace('_', ' '))} vs "
                f"{_ACTION_POLICY_LABELS.get(comparison.get('reference_policy', 'reference'), str(comparison.get('reference_policy', 'reference')).replace('_', ' '))} | {labels[metric]} | "
                f"{_num(value.get('mean_difference', 0), 3)} | "
                f"{_interval_text([value.get('ci_lower', 0), value.get('ci_upper', 0)], 3)} |"
            )
    return "\n".join(rows)


def _reader_evidence_status(
    meta: dict[str, Any], manifest: dict[str, Any], run_log: dict[str, Any]
) -> str:
    """Describe evidence readiness without depending on render-finalization order.

    Hydration deliberately revokes ``manifest.complete`` before it writes
    renderer inputs, so reader-facing prose must not use that mutable terminal
    flag.  The final manifest and its bound review receipt remain the sole
    authority for publication completion.  Campaigns are never publication
    evidence, even when their isolated computational receipt is complete.
    """
    profile = str(meta.get("profile", ""))
    run_profile = str(run_log.get("profile", ""))
    run_passed = run_log.get("status") == "passed"
    terminal_certificates = bool(manifest.get("certificates_pass", False))
    if profile == "campaign" or run_profile == "campaign":
        if bool(manifest.get("complete", False)) and run_passed and terminal_certificates:
            return "campaign-complete non-release"
        return "campaign review-only non-release"
    if (
        profile in {"release", "publication"}
        and run_profile in {"release", "publication"}
        and run_passed
        and terminal_certificates
    ):
        return "release candidate bound to the current source"
    return "review-only"


def generate_variables(
    project_root: Path,
    require_analysis_outputs: bool = True,
    *,
    output_root: Path | None = None,
) -> dict[str, str]:
    """Build the ``{{TOKEN}} -> value`` map for manuscript injection.

    Args:
        project_root: Project directory containing ``manuscript/`` and the source
            configuration.
        require_analysis_outputs: If True, raise when ``output/results.json`` is
            absent; if False, fall back to ``N/A`` placeholders (draft mode).
            Quality measurements are optional at this pure-function layer and
            resolve to ``PENDING`` until the quality orchestrator has written
            ``quality.json`` in the selected output root. The publication
            orchestrator enforces that file before strict hydration.
        output_root: Optional project-relative or absolute directory containing
            ``results.json`` and ``quality.json``. Defaults to ``project_root /
            output``.

    Returns:
        Flat mapping of uppercase token names to their string replacements.

    Raises:
        FileNotFoundError: results are required but ``results.json`` is missing
            from the selected output root.
    """
    project_root = Path(project_root)
    resolved_output_root = _resolve_output_root(project_root, output_root)
    config = _load_config(project_root)
    results = _load_results(project_root, resolved_output_root)
    quality = _load_quality(project_root, resolved_output_root)

    if not results:
        if require_analysis_outputs:
            raise FileNotFoundError(
                f"{resolved_output_root / 'results.json'} not found; run "
                "scripts/generate_results.py first (or pass --allow-draft)."
            )
        return _draft_variables(config)
    exp = config.get("experiment", {})
    mc = results["method_comparison"]
    certs = results["certificates"]
    ai = results["active_inference"]
    bh = mc["benjamini_hochberg"]
    storey_cert = certs["storey_power_gain"]
    storey_split_cert = certs.get("storey_split_confirmation", {})
    ai_regimes = ai.get("regimes", ai)
    calibration = results["null_pvalue_calibration"]
    accounting = results["sample_size_accounting"]
    complexity = results["computational_complexity"]
    suite = results.get("suite", {})
    suite_operations = suite.get("operating_characteristics", [])
    first_action = suite_operations[0] if suite_operations else {}
    first_action_summary = first_action.get("summary", {})
    online_target = suite.get("online_fdr", {})
    selection_calibration = suite.get("conditional_selection_calibration", {})
    selection_pooled = selection_calibration.get("pooled", {})
    future_selection = suite.get("future_evidence_selection_failure", {})
    future_pooled = future_selection.get("pooled", {})
    split_confirmation_scenario = suite.get("split_confirmation_scenario", {})
    split_confirmation = suite.get("split_stream_confirmation", {})
    learned_policy = suite.get("learned_policy", {})
    learned_artifact = learned_policy.get("artifact", {})
    learned_evaluation = learned_policy.get("evaluation", {})
    learned_summary = learned_evaluation.get("summary", {})
    formalism = results["formalism"]
    ordered_methods = list(mc)
    fwer_methods = [
        name for name in ordered_methods if name in {"bonferroni", "holm", "hochberg"}
    ]
    fdr_methods = [
        name
        for name in ordered_methods
        if name
        in {"benjamini_hochberg", "benjamini_yekutieli", "storey_qvalue", "adaptive_bh", "weighted_bh"}
    ]
    quality_test_count = quality.get("test_count", "PENDING")
    quality_branch_coverage = quality.get("branch_coverage_pct", "PENDING")
    quality_line_coverage = quality.get("line_coverage_pct", "PENDING")
    quality_line_gate = quality.get("coverage_gate_pct", "PENDING")
    quality_branch_gate = quality.get("branch_coverage_gate_pct", "PENDING")
    publication_repository_statement = _publication_repository_statement(config)
    run_log_path = resolved_output_root / "run_log.json"
    run_log: dict[str, Any] = {}
    if run_log_path.is_file():
        loaded_run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
        if isinstance(loaded_run_log, dict):
            run_log = loaded_run_log
    run_status = str(run_log.get("status", "missing"))
    manifest_path = resolved_output_root / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded_manifest, dict):
            manifest = loaded_manifest
    meta = results["meta"]
    certificate_gate = "passed" if bool(manifest.get("certificates_pass", False)) else "failed"
    release_status = _reader_evidence_status(meta, manifest, run_log)
    captions = figure_captions(results)
    alts = figure_alts(results)
    strong_config = ai_regimes["strong_evidence"]["config"]
    weak_config = ai_regimes["weak_evidence"]["config"]

    variables: dict[str, str] = {
        # configuration
        "CONFIG_M": str(exp.get("m", _MISSING)),
        "CONFIG_PI0": _num(exp.get("pi0", 0), 1) if "pi0" in exp else _MISSING,
        "CONFIG_EFFECT": _num(exp.get("effect", 0), 1) if "effect" in exp else _MISSING,
        "CONFIG_ALPHA": _num(meta["alpha"], 2),
        "CONFIG_NREPS": str(meta["n_reps"]),
        "CONFIG_SEED": str(meta["seed"]),
        "CONFIG_WORKERS": "stable SeedSequence child-stream reduction",
        "CONFIG_PROFILE": str(meta.get("profile", _MISSING)),
        "RESULT_RUN_STATUS": run_status,
        "RESULT_CERTIFICATE_GATE": certificate_gate,
        "RESULT_RELEASE_STATUS": release_status,
        "RESULT_CONFIGURATION_HASH": str(meta.get("config_sha256", _MISSING)),
        "RESULT_CLAIM_LEDGER_HASH": str(meta.get("claim_ledger_sha256", _MISSING)),
        # The hydration orchestrator replaces this placeholder with the
        # self-normalizing digest of all publication renderer inputs.
        "RESULT_RENDER_INPUT_HASH": "PENDING_RENDER_INPUT_HASH",
        "CONFIG_PYTHON": str(meta.get("python", _MISSING)),
        "CONFIG_PYMDP_VERSION": str(meta.get("pymdp_version", exp.get("pymdp_version", _MISSING))),
        "CONFIG_RHO_MAX": _num(max(d["rho"] for d in results["dependence_sweep"]), 2),
        "CONFIG_METHOD_COUNT": str(len(ordered_methods)),
        "CONFIG_AI_REPS": str(ai_regimes["strong_evidence"]["n_reps"]),
        "CONFIG_AI_PRIOR_SIGNAL": _num(exp.get("active_inference", {}).get("regimes", {}).get("strong_evidence", {}).get("prior_signal", 0.5), 2),
        "CONFIG_AI_STRONG_RELIABILITY": _num(strong_config["reliability"], 2),
        "CONFIG_AI_STRONG_THRESHOLD": _num(strong_config["threshold"], 2),
        "CONFIG_AI_STRONG_HORIZON": str(strong_config["horizon"]),
        "CONFIG_AI_WEAK_RELIABILITY": _num(weak_config["reliability"], 2),
        "CONFIG_AI_WEAK_THRESHOLD": _num(weak_config["threshold"], 2),
        "CONFIG_AI_WEAK_HORIZON": str(weak_config["horizon"]),
        "CONFIG_MC_CONFIDENCE": _num(meta["mc_confidence"], 2),
        "CONFIG_MC_CONFIDENCE_PERCENT": _num(100.0 * meta["mc_confidence"], 0),
        "CONFIG_TARGET_MC_HALF_WIDTH": _num(accounting["target_half_width"], 3),
        "CONFIG_CERTIFICATE_MCSE_MULTIPLIER": f"{BAND_SIGMA:g}",
        "CONFIG_STOREY_SPLIT_LAMBDA": _num(exp.get("storey_split_confirmation", {}).get("lambda", 0.5), 2),
        "CONFIG_STOREY_SPLIT_CONFIDENCE_PERCENT": _num(100.0 * exp.get("storey_split_confirmation", {}).get("confidence", 0.99), 0),
        "CONFIG_FIGURE_ERROR_MULTIPLIER": _num(meta.get("figure_error_multiplier", 3.0), 1),
        "CONFIG_FIGURE_DPI": str(meta.get("figure_dpi", 300)),
        "CONFIG_FIGURE_COUNT": str(len(FIGURE_SPECS)),
        "CONFIG_SUITE_NAME": str(exp.get("suite", {}).get("name", "active_inference_power_suite")),
        "CONFIG_SUITE_REFERENCE": str(exp.get("suite", {}).get("reference_scenario", "reference_action_loop")),
        "CONFIG_ACTION_HORIZON": str(exp.get("action_loop", {}).get("horizon", _MISSING)),
        "CONFIG_ACTION_POLICY_COUNT": str(len(exp.get("action_loop", {}).get("policies", []))),
        "CONFIG_ACTION_RELIABILITY_GRID": ", ".join(_num(value, 2) for value in exp.get("action_loop", {}).get("reliability_grid", [])),
        "FIGURE_ERROR_BAND": f"±{float(meta.get('figure_error_multiplier', 3.0)):g} MC SE",
        "FIGURE_FORMALISM_MAP_HEIGHT": "0.90\\textheight",
        "FIGURE_FORMALISM_FINAL_TOP_CLEARANCE": "0.75in",
        "FIGURE_ONLINE_FDR_HEIGHT": "0.62\\textheight",
        # table row blocks (single source of truth)
        "RESULT_COMPARISON_ROWS": _comparison_rows(mc),
        "RESULT_POWERCURVE_ROWS": _powercurve_rows(results["power_curve"]),
        "RESULT_DEPENDENCE_ROWS": _dependence_rows(results["dependence_sweep"]),
        "RESULT_CERT_ROWS": _cert_rows(certs),
        "RESULT_STOREY_AUDIT_ROWS": _storey_audit_rows(storey_cert),
        "RESULT_AI_ROWS": _ai_rows(ai),
        "RESULT_SAMPLE_SIZE_ROWS": _sample_size_rows(accounting),
        "RESULT_COMPLEXITY_ROWS": _complexity_rows(complexity),
        "RESULT_SEQUENTIAL_ROWS": _sequential_rows(results["sequential"]),
        "RESULT_ONLINE_FDR_ROWS": _online_fdr_rows(results["online_fdr"]),
        "RESULT_AGENT_SAMPLING_ROWS": _agent_sampling_rows(results["agent_sampling"]),
        "RESULT_BENCHMARK_ROWS": _benchmark_rows(results["benchmarks"]),
        "RESULT_ACTION_LOOP_ROWS": _action_loop_rows(suite),
        "RESULT_ACTION_PAIRED_ROWS": _paired_action_rows(suite),
        "RESULT_SPLIT_CONFIRMATION_ROWS": _split_confirmation_rows(suite),
        "RESULT_LEARNED_POLICY_ROWS": _learned_policy_rows(suite),
        "RESULT_FORMALISM_ROWS": formalism_rows(formalism),
        "RESULT_FORMALISM_COUNT": str(formalism["entry_count"]),
        "RESULT_FORMALISM_HASH": str(formalism["registry_hash"]),
        "RESULT_ACTION_SCENARIO_ID": str(suite.get("scenario_id", _MISSING)),
        "RESULT_ACTION_SCENARIO_HASH": str(suite.get("scenario_hash", _MISSING)),
        "RESULT_ACTION_REPS": str(first_action.get("n_reps", _MISSING)),
        "RESULT_ACTION_POWER": _num(first_action_summary.get("power", 0), 3) if first_action_summary else _MISSING,
        "RESULT_ACTION_FDR": _num(first_action_summary.get("fdr", 0), 3) if first_action_summary else _MISSING,
        "RESULT_ACTION_COST": _num(first_action_summary.get("mean_cost", 0), 2) if first_action_summary else _MISSING,
        "RESULT_ACTION_STOPPING": _num(first_action_summary.get("mean_stopping_time", 0), 2) if first_action_summary else _MISSING,
        "RESULT_ACTION_VALIDITY": str(first_action_summary.get("validity", _MISSING)),
        "RESULT_ACTION_PAIRED_REPS": str(
            suite.get("paired_policy_comparisons", [{}])[0].get("n_reps", _MISSING)
            if suite.get("paired_policy_comparisons") else _MISSING
        ),
        "RESULT_ONLINE_TARGET_METHOD": _online_fdr_method_label(
            online_target.get("method", _MISSING)
        ),
        "RESULT_ONLINE_TARGET_REJECTION_RATE": _num(online_target.get("any_rejection_rate", 0), 3) if online_target else _MISSING,
        "RESULT_ONLINE_TARGET_FDR": _num(online_target.get("mean_fdr", 0), 3) if online_target else _MISSING,
        "RESULT_ONLINE_TARGET_POWER": _num(online_target.get("mean_power", 0), 3) if online_target else _MISSING,
        "RESULT_ONLINE_TARGET_FWER": _num(online_target.get("fwer", 0), 3) if online_target else _MISSING,
        "RESULT_ONLINE_TARGET_SELECTIONS": _num(online_target.get("selection_count_mean", 0), 2) if online_target else _MISSING,
        "RESULT_SELECTION_CALIBRATION_N": str(selection_pooled.get("n_selected", _MISSING)),
        "RESULT_SELECTION_CALIBRATION_EXCESS": _num(selection_pooled.get("max_ecdf_excess", 0), 3) if selection_pooled else _MISSING,
        "RESULT_SELECTION_CALIBRATION_DKW": _num(selection_pooled.get("dkw_radius", 0), 3) if selection_pooled else _MISSING,
        "RESULT_SELECTION_CALIBRATION_STATUS": "passed" if selection_pooled.get("passes_dkw_diagnostic") else "flagged" if selection_pooled else _MISSING,
        "RESULT_FUTURE_SELECTION_N": str(future_pooled.get("n_selected", _MISSING)),
        "RESULT_FUTURE_SELECTION_EXCESS": _num(future_pooled.get("max_ecdf_excess", 0), 3) if future_pooled else _MISSING,
        "RESULT_FUTURE_SELECTION_DKW": _num(future_pooled.get("dkw_radius", 0), 3) if future_pooled else _MISSING,
        "RESULT_FUTURE_SELECTION_STATUS": "flagged" if future_pooled and not future_pooled.get("passes_dkw_diagnostic") else "diagnostic_pass" if future_pooled else _MISSING,
        "RESULT_SPLIT_CONFIRMATION_REPS": f"{int(split_confirmation.get('n_reps', 0)):,}" if split_confirmation else _MISSING,
        "RESULT_SPLIT_CONFIRMATION_TRUE_TARGET_SELECTIONS": str(
            int(split_confirmation.get("true_target_selection_count", 0))
        ) if split_confirmation else _MISSING,
        "RESULT_SPLIT_CONFIRMATION_TRUE_TARGET_SELECTION_RATE": _num(
            split_confirmation.get("true_target_selection_rate", 0), 3
        ) if split_confirmation else _MISSING,
        "RESULT_SPLIT_CONFIRMATION_JOINT_TRUE_TARGET_REJECTION_RATE": _num(
            split_confirmation.get("true_target_joint_rejection_rate", 0), 3
        ) if split_confirmation else _MISSING,
        "RESULT_SPLIT_CONFIRMATION_CONDITIONAL_TRUE_TARGET_REJECTION_RATE": _probability_or_not_estimable(
            split_confirmation.get("power"), 3
        ) if split_confirmation else _MISSING,
        "RESULT_SPLIT_CONFIRMATION_FWER": _num(split_confirmation.get("fwer", 0), 3) if split_confirmation else _MISSING,
        "RESULT_SPLIT_CONFIRMATION_VALIDITY": str(split_confirmation.get("validity", _MISSING)),
        "RESULT_SPLIT_CONFIRMATION_SCENARIO_ID": str(
            split_confirmation_scenario.get("scenario_id", _MISSING)
        ),
        "RESULT_SPLIT_CONFIRMATION_SCENARIO_HASH": str(
            split_confirmation_scenario.get("scenario_hash", _MISSING)
        ),
        "RESULT_SPLIT_SELECTION_HASH": str(split_confirmation.get("representative", {}).get("selection_stream_hash", _MISSING)),
        "RESULT_SPLIT_CONFIRMATION_HASH": str(split_confirmation.get("representative", {}).get("confirmation_stream_hash", _MISSING)),
        "RESULT_LEARNED_POLICY_REPS": f"{int(learned_evaluation.get('evaluation_reps', 0)):,}" if learned_evaluation else _MISSING,
        "RESULT_LEARNED_POLICY_POWER": _num(learned_summary.get("power", 0), 3) if learned_summary else _MISSING,
        "RESULT_LEARNED_POLICY_COST": _num(learned_summary.get("mean_cost", 0), 2) if learned_summary else _MISSING,
        "RESULT_LEARNED_POLICY_HASH": str(learned_artifact.get("policy_hash", _MISSING)),
        "RESULT_LEARNED_POLICY_SEARCH_HASH": str(learned_artifact.get("search_config_hash", _MISSING)),
        "RESULT_LEARNED_POLICY_CANDIDATES": str(
            learned_artifact.get("policy_complexity", {}).get("candidate_count", _MISSING)
        ),
        "RESULT_LEARNED_POLICY_DIMENSION": str(
            learned_artifact.get("policy_complexity", {}).get("parameter_dimension", _MISSING)
        ),
        "RESULT_COMPLEXITY_ACTION_TRACE_CALLS": f"{int(complexity['release_workload']['action_loop']['action_loop_trace_calls']):,}",
        "RESULT_COMPLEXITY_ACTION_STEP_BOUND": f"{int(complexity['release_workload']['action_loop']['action_loop_step_upper_bound']):,}",
        "RESULT_COMPLEXITY_FIXED_RETAINED_ELEMENTS": f"{int(complexity['release_workload']['fixed_family']['per_procedure_retained_pvalue_elements']):,}",
        "RESULT_SEQUENTIAL_REPS": f"{int(results['sequential']['n_reps']):,}",
        "RESULT_SEQUENTIAL_HORIZON": str(results["sequential"]["horizon"]),
        "RESULT_ONLINE_FDR_REPS": f"{int(results['online_fdr']['n_reps']):,}",
        "RESULT_ONLINE_FDR_HORIZON": str(results["online_fdr"]["horizon"]),
        "RESULT_AGENT_POLICY_REPS": f"{int(results['agent_sampling']['n_reps']):,}",
        "RESULT_BENCHMARK_ROWS_COUNT": f"{int(results['benchmarks']['agent_trace']['rows']):,}",
        "RESULT_BENCHMARK_SUBJECTS": str(results["benchmarks"]["agent_trace"]["subjects"]),
        "RESULT_CALIBRATION_CONFIDENCE_PERCENT": _num(100.0 * calibration["confidence"], 0),
        # key inline statistics
        "RESULT_BH_FDR": _num(bh["fdr"], 4),
        "RESULT_BH_POWER": _num(bh["power"], 3),
        "RESULT_STOREY_GAIN": _num(storey_cert["power_gain"], 3),
        "RESULT_STOREY_POWER": _num(storey_cert["storey_power"], 3),
        "RESULT_STOREY_AUDIT_REPS": f"{int(storey_cert['n_reps']):,}",
        "RESULT_STOREY_FDR_ESTIMATE": _num(storey_cert["storey_fdr"], 5),
        "RESULT_STOREY_FDR_STDERR": _num(storey_cert["storey_fdr_stderr"], 6),
        "RESULT_STOREY_FDR_UPPER_BAND": _num(storey_cert["storey_fdr_upper_band"], 5),
        "RESULT_STOREY_FDR_EXCESS": _num(storey_cert["storey_fdr_excess"], 5),
        "RESULT_STOREY_FDR_Z": _num(storey_cert["storey_fdr_z"], 2),
        "RESULT_STOREY_FDR_GATE": "passed" if storey_cert["storey_fdr_passes"] else "failed",
        "RESULT_STOREY_BH_FDR_ESTIMATE": _num(storey_cert["bh_fdr"], 5),
        "RESULT_STOREY_BH_FDR_STDERR": _num(storey_cert["bh_fdr_stderr"], 6),
        "RESULT_STOREY_BH_FDR_UPPER_BAND": _num(storey_cert["bh_fdr_upper_band"], 5),
        "RESULT_STOREY_POWER_ESTIMATE": _num(storey_cert["storey_power"], 5),
        "RESULT_STOREY_POWER_STDERR": _num(storey_cert["stderr"], 6),
        "RESULT_STOREY_POWER_GAIN": _num(storey_cert["power_gain"], 5),
        "RESULT_STOREY_POWER_GAIN_STDERR": _num(storey_cert["power_gain_stderr"], 6),
        "RESULT_STOREY_POWER_GATE": "passed" if storey_cert["power_gain_passes"] else "failed",
        "RESULT_AI_WEAK_POSTERIOR_FDR": _num(ai_regimes["weak_evidence"]["posterior_threshold_fdr"], 3),
        "RESULT_AI_WEAK_POSTERIOR_FWER": _num(ai_regimes["weak_evidence"]["posterior_threshold_fwer"], 3),
        "RESULT_AI_STRONG_POSTERIOR_FDR": _num(
            ai_regimes["strong_evidence"]["posterior_threshold_fdr"], 3
        ),
        "RESULT_AI_STRONG_POSTERIOR_POWER": _num(
            ai_regimes["strong_evidence"]["posterior_threshold_power"], 3
        ),
        "RESULT_DEP_FDR_MIN": _num(results["dependence_sweep"][-1]["bh_fdr"], 3),
        "RESULT_BH_POWER_MAX": _num(max(p["power"] for p in results["power_curve"]), 3),
        "RESULT_FDR_POWER_MIN": _num(min(mc[name]["power"] for name in fdr_methods), 3),
        "RESULT_FDR_POWER_MAX": _num(max(mc[name]["power"] for name in fdr_methods), 3),
        "RESULT_FWER_POWER_MIN": _num(
            min(mc[name]["power"] for name in fwer_methods), 3
        ),
        "RESULT_FWER_POWER_MAX": _num(
            max(mc[name]["power"] for name in fwer_methods), 3
        ),
        "RESULT_FWER_REJECTIONS_MIN": _num(
            min(mc[name]["mean_rejections"] for name in fwer_methods), 1
        ),
        "RESULT_BY_FDR": _num(mc["benjamini_yekutieli"]["fdr"], 3),
        "RESULT_BY_POWER": _num(mc["benjamini_yekutieli"]["power"], 3),
        "RESULT_STOREY_FDR": _num(mc["storey_qvalue"]["fdr"], 3),
        "RESULT_STOREY_SPLIT_FDR": _num(storey_split_cert.get("estimate", 0), 3) if storey_split_cert else _MISSING,
        "RESULT_AI_STRONG_POSTERIOR_FWER": _num(
            ai_regimes["strong_evidence"]["posterior_threshold_fwer"], 3
        ),
        "RESULT_AI_WEAK_BH_FDR": _num(
            ai_regimes["weak_evidence"]["bh_on_evidence_fdr"], 3
        ),
        "RESULT_AI_WEAK_BH_POWER": _num(
            ai_regimes["weak_evidence"]["bh_on_evidence_power"], 3
        ),
        "RESULT_AI_WEAK_BH_FWER": _num(
            ai_regimes["weak_evidence"]["bh_on_evidence_fwer"], 3
        ),
        "RESULT_CALIBRATION_KS": _num(calibration["ks_statistic"], 4),
        "RESULT_CALIBRATION_DKW": _num(calibration["dkw_radius"], 4),
        "RESULT_CALIBRATION_PVALUES": f"{int(calibration['n_pvalues']):,}",
        "RESULT_CALIBRATION_REPS": f"{int(calibration['n_reps']):,}",
        "RESULT_REQUIRED_REPS": f"{int(accounting['worst_case_required_reps']):,}",
        "RESULT_BH_MC_HALF_WIDTH": _num(bh["mc_half_width_fdr"], 4),
        "RESULT_BH_HOEFFDING_RADIUS": _num(bh["mc_hoeffding_radius_fdr"], 4),
        "QUALITY_TEST_COUNT": str(quality_test_count),
        "QUALITY_BRANCH_COVERAGE": _num(quality_branch_coverage, 2)
        if quality_branch_coverage != "PENDING"
        else "PENDING",
        "QUALITY_LINE_COVERAGE": _num(quality_line_coverage, 2)
        if quality_line_coverage != "PENDING"
        else "PENDING",
        "QUALITY_LINE_GATE": _num(quality_line_gate, 0)
        if quality_line_gate != "PENDING"
        else "PENDING",
        "QUALITY_BRANCH_GATE": _num(quality_branch_gate, 0)
        if quality_branch_gate != "PENDING"
        else "PENDING",
        "PUBLICATION_REPOSITORY_STATEMENT": publication_repository_statement,
        **_publication_release_identity(config),
    }
    if set(captions) != set(alts):
        raise ValueError("figure caption and alternative-text keys differ")
    for name, caption in captions.items():
        variables[f"FIGURE_{name.upper()}_CAPTION"] = caption
        variables[f"FIGURE_{name.upper()}_ALT"] = alts[name]
    return variables


def _draft_variables(config: dict[str, Any]) -> dict[str, str]:
    """Placeholder variable map used when analysis outputs are absent."""
    exp = config.get("experiment", {})
    keys = [
        "RESULT_COMPARISON_ROWS",
        "RESULT_POWERCURVE_ROWS",
        "RESULT_DEPENDENCE_ROWS",
        "RESULT_CERT_ROWS",
        "RESULT_STOREY_AUDIT_ROWS",
        "RESULT_AI_ROWS",
        "RESULT_SEQUENTIAL_ROWS",
        "RESULT_ONLINE_FDR_ROWS",
        "RESULT_AGENT_SAMPLING_ROWS",
        "RESULT_BENCHMARK_ROWS",
        "RESULT_ACTION_LOOP_ROWS",
        "RESULT_ACTION_PAIRED_ROWS",
        "RESULT_SPLIT_CONFIRMATION_ROWS",
        "RESULT_LEARNED_POLICY_ROWS",
        "RESULT_FORMALISM_ROWS",
        "RESULT_FORMALISM_COUNT",
        "RESULT_FORMALISM_HASH",
        "RESULT_CLAIM_LEDGER_HASH",
        "RESULT_RENDER_INPUT_HASH",
        "RESULT_RUN_STATUS",
        "RESULT_CERTIFICATE_GATE",
        "RESULT_RELEASE_STATUS",
        "RESULT_ACTION_SCENARIO_ID",
        "RESULT_ACTION_SCENARIO_HASH",
        "RESULT_ACTION_REPS",
        "RESULT_ACTION_POWER",
        "RESULT_ACTION_FDR",
        "RESULT_ACTION_COST",
        "RESULT_ACTION_STOPPING",
        "RESULT_ACTION_VALIDITY",
        "RESULT_ACTION_PAIRED_REPS",
        "RESULT_ONLINE_TARGET_METHOD",
        "RESULT_ONLINE_TARGET_REJECTION_RATE",
        "RESULT_ONLINE_TARGET_FDR",
        "RESULT_ONLINE_TARGET_POWER",
        "RESULT_ONLINE_TARGET_FWER",
        "RESULT_ONLINE_TARGET_SELECTIONS",
        "RESULT_SPLIT_CONFIRMATION_REPS",
        "RESULT_SPLIT_CONFIRMATION_TRUE_TARGET_SELECTIONS",
        "RESULT_SPLIT_CONFIRMATION_TRUE_TARGET_SELECTION_RATE",
        "RESULT_SPLIT_CONFIRMATION_JOINT_TRUE_TARGET_REJECTION_RATE",
        "RESULT_SPLIT_CONFIRMATION_CONDITIONAL_TRUE_TARGET_REJECTION_RATE",
        "RESULT_SPLIT_CONFIRMATION_FWER",
        "RESULT_SPLIT_CONFIRMATION_VALIDITY",
        "RESULT_SPLIT_CONFIRMATION_SCENARIO_ID",
        "RESULT_SPLIT_CONFIRMATION_SCENARIO_HASH",
        "RESULT_SPLIT_SELECTION_HASH",
        "RESULT_SPLIT_CONFIRMATION_HASH",
        "RESULT_LEARNED_POLICY_REPS",
        "RESULT_LEARNED_POLICY_POWER",
        "RESULT_LEARNED_POLICY_COST",
        "RESULT_LEARNED_POLICY_HASH",
        "RESULT_LEARNED_POLICY_SEARCH_HASH",
        "RESULT_LEARNED_POLICY_CANDIDATES",
        "RESULT_LEARNED_POLICY_DIMENSION",
        "RESULT_SELECTION_CALIBRATION_N",
        "RESULT_SELECTION_CALIBRATION_EXCESS",
        "RESULT_SELECTION_CALIBRATION_DKW",
        "RESULT_SELECTION_CALIBRATION_STATUS",
        "RESULT_SEQUENTIAL_REPS",
        "RESULT_SEQUENTIAL_HORIZON",
        "RESULT_ONLINE_FDR_REPS",
        "RESULT_ONLINE_FDR_HORIZON",
        "RESULT_AGENT_POLICY_REPS",
        "RESULT_BENCHMARK_ROWS_COUNT",
        "RESULT_BENCHMARK_SUBJECTS",
        "RESULT_BH_FDR",
        "RESULT_BH_POWER",
        "RESULT_STOREY_GAIN",
        "RESULT_STOREY_POWER",
        "RESULT_STOREY_AUDIT_REPS",
        "RESULT_STOREY_FDR_ESTIMATE",
        "RESULT_STOREY_FDR_STDERR",
        "RESULT_STOREY_FDR_UPPER_BAND",
        "RESULT_STOREY_FDR_EXCESS",
        "RESULT_STOREY_FDR_Z",
        "RESULT_STOREY_FDR_GATE",
        "RESULT_STOREY_BH_FDR_ESTIMATE",
        "RESULT_STOREY_BH_FDR_STDERR",
        "RESULT_STOREY_BH_FDR_UPPER_BAND",
        "RESULT_STOREY_POWER_ESTIMATE",
        "RESULT_STOREY_POWER_STDERR",
        "RESULT_STOREY_POWER_GAIN",
        "RESULT_STOREY_POWER_GAIN_STDERR",
        "RESULT_STOREY_POWER_GATE",
        "RESULT_STOREY_SPLIT_FDR",
        "RESULT_AI_WEAK_POSTERIOR_FDR",
        "RESULT_AI_WEAK_POSTERIOR_FWER",
        "RESULT_AI_STRONG_POSTERIOR_FDR",
        "RESULT_AI_STRONG_POSTERIOR_POWER",
        "RESULT_DEP_FDR_MIN",
        "CONFIG_ALPHA",
        "CONFIG_NREPS",
        "CONFIG_SEED",
        "CONFIG_WORKERS",
        "CONFIG_PROFILE",
        "CONFIG_PYTHON",
        "CONFIG_PYMDP_VERSION",
        "CONFIG_RHO_MAX",
        "CONFIG_METHOD_COUNT",
        "CONFIG_AI_REPS",
        "CONFIG_AI_PRIOR_SIGNAL",
        "CONFIG_AI_STRONG_RELIABILITY",
        "CONFIG_AI_STRONG_THRESHOLD",
        "CONFIG_AI_STRONG_HORIZON",
        "CONFIG_AI_WEAK_RELIABILITY",
        "CONFIG_AI_WEAK_THRESHOLD",
        "CONFIG_AI_WEAK_HORIZON",
        "RESULT_BH_POWER_MAX",
        "RESULT_FDR_POWER_MIN",
        "RESULT_FDR_POWER_MAX",
        "RESULT_FWER_POWER_MIN",
        "RESULT_FWER_POWER_MAX",
        "RESULT_FWER_REJECTIONS_MIN",
        "RESULT_BY_FDR",
        "RESULT_BY_POWER",
        "RESULT_STOREY_FDR",
        "RESULT_STOREY_SPLIT_FDR",
        "CONFIG_STOREY_SPLIT_LAMBDA",
        "CONFIG_STOREY_SPLIT_CONFIDENCE_PERCENT",
        "RESULT_AI_STRONG_POSTERIOR_FWER",
        "RESULT_AI_WEAK_BH_FDR",
        "RESULT_AI_WEAK_BH_POWER",
        "RESULT_AI_WEAK_BH_FWER",
        "RESULT_SAMPLE_SIZE_ROWS",
        "RESULT_COMPLEXITY_ROWS",
        "RESULT_COMPLEXITY_ACTION_TRACE_CALLS",
        "RESULT_COMPLEXITY_ACTION_STEP_BOUND",
        "RESULT_COMPLEXITY_FIXED_RETAINED_ELEMENTS",
        "CONFIG_MC_CONFIDENCE",
        "CONFIG_MC_CONFIDENCE_PERCENT",
        "CONFIG_TARGET_MC_HALF_WIDTH",
        "CONFIG_CERTIFICATE_MCSE_MULTIPLIER",
        "CONFIG_FIGURE_ERROR_MULTIPLIER",
        "CONFIG_FIGURE_DPI",
        "CONFIG_FIGURE_COUNT",
        "CONFIG_SUITE_NAME",
        "CONFIG_SUITE_REFERENCE",
        "CONFIG_ACTION_HORIZON",
        "CONFIG_ACTION_POLICY_COUNT",
        "CONFIG_ACTION_RELIABILITY_GRID",
        "FIGURE_ERROR_BAND",
        "FIGURE_FORMALISM_MAP_HEIGHT",
        "FIGURE_FORMALISM_FINAL_TOP_CLEARANCE",
        "FIGURE_ONLINE_FDR_HEIGHT",
        "RESULT_CALIBRATION_KS",
        "RESULT_CALIBRATION_DKW",
        "RESULT_CALIBRATION_PVALUES",
        "RESULT_CALIBRATION_REPS",
        "RESULT_REQUIRED_REPS",
        "RESULT_BH_MC_HALF_WIDTH",
        "RESULT_BH_HOEFFDING_RADIUS",
        "RESULT_CALIBRATION_CONFIDENCE_PERCENT",
        "QUALITY_TEST_COUNT",
        "QUALITY_BRANCH_COVERAGE",
        "QUALITY_LINE_COVERAGE",
        "QUALITY_LINE_GATE",
        "QUALITY_BRANCH_GATE",
    ]
    variables = {k: _MISSING for k in keys}
    variables["CONFIG_M"] = str(exp.get("m", _MISSING))
    variables["CONFIG_PI0"] = _num(exp.get("pi0", 0), 1) if "pi0" in exp else _MISSING
    variables["CONFIG_EFFECT"] = (
        _num(exp.get("effect", 0), 1) if "effect" in exp else _MISSING
    )
    variables["CONFIG_CERTIFICATE_MCSE_MULTIPLIER"] = f"{BAND_SIGMA:g}"
    variables["PUBLICATION_REPOSITORY_STATEMENT"] = _publication_repository_statement(
        config
    )
    variables.update(_publication_release_identity(config))
    variables["FIGURE_FORMALISM_MAP_HEIGHT"] = "0.90\\textheight"
    variables["FIGURE_FORMALISM_FINAL_TOP_CLEARANCE"] = "0.75in"
    variables["FIGURE_ONLINE_FDR_HEIGHT"] = "0.62\\textheight"
    for spec in FIGURE_SPECS:
        variables[f"FIGURE_{spec.name.upper()}_CAPTION"] = _MISSING
        variables[f"FIGURE_{spec.name.upper()}_ALT"] = _MISSING
    return variables


def save_variables(variables: dict[str, str], out_path: Path) -> None:
    """Write the variable map to JSON for provenance."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.", dir=out_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(variables, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, out_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
