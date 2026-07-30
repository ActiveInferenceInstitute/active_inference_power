"""Contracts for an isolated all-chunk Dobriban/BH stress receipt.

The bundled output is deliberately narrower than the accompanying external
work.  It records finite local executions of the checksum-pinned plan and
never represents those simulations as the external outward-rounded Arb
certificate, arbitrary-dependence evidence, or an Active Inference result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .dobriban_bh import DobribanBHProtocol


DOBRIBAN_BH_BUNDLE_SCHEMA_VERSION = "1.0"
DOBRIBAN_BH_BUNDLE_KIND = "checksum_pinned_all_chunk_dobriban_bh_stress_bundle"
DOBRIBAN_BH_BUNDLE_STATUS = "offline_protocol_stress_only_not_for_promotion_or_theorem"
DOBRIBAN_BH_SMOKE_STATUS = "offline_protocol_compact_smoke_only_not_for_promotion_or_theorem"
DOBRIBAN_BH_FULL_EXECUTION_MODE = "frozen_public_plan"
DOBRIBAN_BH_SMOKE_EXECUTION_MODE = "compact_real_smoke"
DOBRIBAN_BH_BUNDLE_FILENAME = "dobriban_bh_stress_bundle.json"

_CLAIM_BOUNDARY = (
    "This offline finite Monte Carlo bundle executes the checksum-pinned seven-chunk "
    "two-sided Gaussian BH protocol. It neither executes nor reproduces the external "
    "outward-rounded Arb certificate and cannot establish arbitrary-dependence behavior, "
    "a universal failure rate, or an Active Inference claim."
)
_NONCLAIMS = (
    "No external theorem or Arb certificate is executed, reproduced, or certified here.",
    "Finite stress estimates do not establish arbitrary-dependence BH behavior or a universal failure rate.",
    "This isolated receipt is not a release, campaign, manuscript, certificate, or promotion input.",
)
_RESULT_KEYS = {
    "schema_version",
    "model_id",
    "protocol_hash",
    "N",
    "m",
    "strata",
    "macro_replications",
    "datasets",
    "master_seed",
    "seed_spawn_key",
    "target_entries",
    "fdr_hat",
    "mcse",
    "ci95_low",
    "ci95_high",
    "one_sided_99_low",
    "t_stat_vs_alpha",
    "one_sided_p_value",
    "mean_rejections",
    "rejection_probability",
    "evidence_class",
}


def _canonical_digest(value: Mapping[str, Any]) -> str:
    """Hash JSON-serializable mappings with a stable representation."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def dobriban_bh_bundle_digest(value: Mapping[str, Any]) -> str:
    """Return the self-normalizing digest for one all-chunk bundle mapping."""
    body = dict(value)
    body.pop("result_sha256", None)
    return _canonical_digest(body)


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], name: str) -> None:
    observed = set(value)
    if observed != keys:
        raise ValueError(
            f"{name} keys differ from the frozen contract: "
            f"missing={sorted(keys - observed)}, unexpected={sorted(observed - keys)}"
        )


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_positive_integer(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _require_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    return numeric


def _validate_source_hashes(value: object) -> None:
    sources = _require_mapping(value, "bundle.source_hashes")
    if not sources:
        raise ValueError("bundle.source_hashes must be non-empty")
    for name, digest in sources.items():
        if not isinstance(name, str) or not name:
            raise ValueError("bundle.source_hashes keys must be non-empty strings")
        _require_sha256(digest, f"bundle.source_hashes.{name}")


def _declared_chunks(protocol: DobribanBHProtocol) -> tuple[Mapping[str, Any], ...]:
    chunks = protocol.plan["chunks"]
    if not isinstance(chunks, list):  # Defensive: protocol validates this on construction.
        raise ValueError("protocol chunks must be a list")
    return tuple(_require_mapping(chunk, "protocol chunk") for chunk in chunks)


def frozen_dobriban_bh_execution_profile(
    protocol: DobribanBHProtocol,
) -> dict[str, Any]:
    """Return the exact public-plan execution parameters for the CLI runner."""
    return {
        "mode": DOBRIBAN_BH_FULL_EXECUTION_MODE,
        "strata": int(protocol.plan["strata"]),
        "target_entries": int(protocol.plan["target_entries"]),
        "macro_replications": {
            str(chunk["prefix"]): int(chunk["macro_replications"])
            for chunk in _declared_chunks(protocol)
        },
        "declared_chunk_prefixes": [
            str(chunk["prefix"]) for chunk in _declared_chunks(protocol)
        ],
    }


def compact_dobriban_bh_execution_profile(
    protocol: DobribanBHProtocol,
    *,
    strata: int,
    macro_replications: int,
    target_entries: int,
) -> dict[str, Any]:
    """Return a bounded real-execution profile for tests and local smoke checks.

    The function intentionally remains programmatic: the command-line runner
    has no tuning flags and always selects :func:`frozen_dobriban_bh_execution_profile`.
    Every declared chunk, its ``N``, block label, and seed remain unchanged;
    only the finite replication budget is reduced and serialized as smoke-only.
    """
    profile = {
        "mode": DOBRIBAN_BH_SMOKE_EXECUTION_MODE,
        "strata": strata,
        "target_entries": target_entries,
        "macro_replications": {
            str(chunk["prefix"]): macro_replications
            for chunk in _declared_chunks(protocol)
        },
        "declared_chunk_prefixes": [
            str(chunk["prefix"]) for chunk in _declared_chunks(protocol)
        ],
    }
    _validate_execution_profile(profile, protocol)
    return profile


def _validate_execution_profile(
    value: object, protocol: DobribanBHProtocol
) -> Mapping[str, Any]:
    profile = _require_mapping(value, "bundle.execution_profile")
    _require_exact_keys(
        profile,
        {
            "mode",
            "strata",
            "target_entries",
            "macro_replications",
            "declared_chunk_prefixes",
        },
        "bundle.execution_profile",
    )
    mode = profile["mode"]
    if mode not in {
        DOBRIBAN_BH_FULL_EXECUTION_MODE,
        DOBRIBAN_BH_SMOKE_EXECUTION_MODE,
    }:
        raise ValueError("bundle.execution_profile.mode is not supported")
    strata = _require_positive_integer(profile["strata"], "bundle.execution_profile.strata")
    target_entries = _require_positive_integer(
        profile["target_entries"], "bundle.execution_profile.target_entries"
    )
    expected_chunks = _declared_chunks(protocol)
    prefixes = [str(chunk["prefix"]) for chunk in expected_chunks]
    if profile["declared_chunk_prefixes"] != prefixes:
        raise ValueError("bundle.execution_profile must retain the ordered seven declared chunks")
    macro = _require_mapping(
        profile["macro_replications"], "bundle.execution_profile.macro_replications"
    )
    if set(macro) != set(prefixes):
        raise ValueError("bundle.execution_profile macro replications must exactly cover declared chunks")
    expected_macro = {
        str(chunk["prefix"]): int(chunk["macro_replications"])
        for chunk in expected_chunks
    }
    for prefix in prefixes:
        _require_positive_integer(
            macro[prefix], f"bundle.execution_profile.macro_replications.{prefix}", minimum=2
        )
    if mode == DOBRIBAN_BH_FULL_EXECUTION_MODE:
        if strata != int(protocol.plan["strata"]):
            raise ValueError("frozen public-plan profile must retain protocol strata")
        if target_entries != int(protocol.plan["target_entries"]):
            raise ValueError("frozen public-plan profile must retain protocol target_entries")
        if {name: int(value) for name, value in macro.items()} != expected_macro:
            raise ValueError("frozen public-plan profile must retain every declared macro replication count")
    else:
        if strata >= int(protocol.plan["strata"]):
            raise ValueError("compact smoke profile must use fewer strata than the public plan")
        if target_entries > int(protocol.plan["target_entries"]):
            raise ValueError("compact smoke profile cannot exceed public-plan target_entries")
        if any(int(macro[name]) > expected_macro[name] for name in prefixes):
            raise ValueError("compact smoke profile cannot exceed declared macro replication counts")
    return profile


def validate_dobriban_bh_execution_profile(
    value: Mapping[str, Any], *, protocol: DobribanBHProtocol
) -> None:
    """Validate a full or bounded smoke execution profile before computation."""
    _validate_execution_profile(value, protocol)


def _validate_result(
    value: object,
    *,
    declared: Mapping[str, Any],
    profile: Mapping[str, Any],
    protocol: DobribanBHProtocol,
) -> None:
    result = _require_mapping(value, f"bundle chunk {declared['prefix']} result")
    _require_exact_keys(result, _RESULT_KEYS, f"bundle chunk {declared['prefix']} result")
    if result["schema_version"] != "2.0":
        raise ValueError("Dobriban/BH chunk result schema must be '2.0'")
    if result["model_id"] != protocol.plan["model"]["model_id"]:
        raise ValueError("Dobriban/BH chunk result model_id does not match the frozen protocol")
    if result["protocol_hash"] != protocol.protocol_hash:
        raise ValueError("Dobriban/BH chunk result protocol hash does not match the frozen protocol")
    N = _require_positive_integer(result["N"], "chunk result.N")
    if N != int(declared["N"]):
        raise ValueError("Dobriban/BH chunk result N does not match its declared chunk")
    if _require_positive_integer(result["m"], "chunk result.m") != 100 * N:
        raise ValueError("Dobriban/BH chunk result m must equal 100*N")
    strata = _require_positive_integer(result["strata"], "chunk result.strata")
    if strata != int(profile["strata"]):
        raise ValueError("Dobriban/BH chunk result strata does not match the execution profile")
    macro = _require_positive_integer(
        result["macro_replications"], "chunk result.macro_replications", minimum=2
    )
    profile_macro = _require_mapping(profile["macro_replications"], "execution profile macro replications")
    if macro != int(profile_macro[str(declared["prefix"])]):
        raise ValueError("Dobriban/BH chunk macro replications do not match the execution profile")
    if _require_positive_integer(result["datasets"], "chunk result.datasets") != strata * macro:
        raise ValueError("Dobriban/BH chunk datasets must equal strata times macro replications")
    if _require_positive_integer(result["master_seed"], "chunk result.master_seed", minimum=0) != int(
        declared["master_seed"]
    ):
        raise ValueError("Dobriban/BH chunk master seed does not match the declared chunk")
    if result["seed_spawn_key"] != [0]:
        raise ValueError("Dobriban/BH chunk seed spawn key must be the documented child stream [0]")
    if _require_positive_integer(result["target_entries"], "chunk result.target_entries") != int(
        profile["target_entries"]
    ):
        raise ValueError("Dobriban/BH chunk target_entries do not match the execution profile")
    fdr_hat = _require_finite_number(result["fdr_hat"], "chunk result.fdr_hat")
    if not 0.0 <= fdr_hat <= 1.0:
        raise ValueError("Dobriban/BH chunk fdr_hat must be in [0, 1]")
    mcse = _require_finite_number(result["mcse"], "chunk result.mcse")
    if mcse <= 0.0:
        raise ValueError("Dobriban/BH chunk mcse must be positive")
    ci_low = _require_finite_number(result["ci95_low"], "chunk result.ci95_low")
    ci_high = _require_finite_number(result["ci95_high"], "chunk result.ci95_high")
    if not ci_low <= fdr_hat <= ci_high:
        raise ValueError("Dobriban/BH chunk 95% interval must contain fdr_hat")
    _require_finite_number(result["one_sided_99_low"], "chunk result.one_sided_99_low")
    _require_finite_number(result["t_stat_vs_alpha"], "chunk result.t_stat_vs_alpha")
    p_value = _require_finite_number(result["one_sided_p_value"], "chunk result.one_sided_p_value")
    if not 0.0 <= p_value <= 1.0:
        raise ValueError("Dobriban/BH chunk one-sided p-value must be in [0, 1]")
    if _require_finite_number(result["mean_rejections"], "chunk result.mean_rejections") < 0.0:
        raise ValueError("Dobriban/BH chunk mean rejections must be non-negative")
    rejection_probability = _require_finite_number(
        result["rejection_probability"], "chunk result.rejection_probability"
    )
    if not 0.0 <= rejection_probability <= 1.0:
        raise ValueError("Dobriban/BH chunk rejection probability must be in [0, 1]")
    if result["evidence_class"] != "stress_test":
        raise ValueError("Dobriban/BH chunk evidence class must be stress_test")


def build_dobriban_bh_stress_bundle(
    protocol: DobribanBHProtocol,
    *,
    execution_profile: Mapping[str, Any],
    chunk_results: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build and validate one exact all-chunk receipt before it is written.

    ``chunk_results`` must be in the public plan's declared order.  The
    orchestration layer supplies real results from :func:`run_dobriban_bh_chunk`;
    this domain helper only assembles and validates their finite evidence.
    """
    profile = _validate_execution_profile(execution_profile, protocol)
    expected_chunks = _declared_chunks(protocol)
    if len(chunk_results) != len(expected_chunks):
        raise ValueError("Dobriban/BH bundle must contain every declared chunk exactly once")
    _validate_source_hashes(source_hashes)
    rows: list[dict[str, Any]] = []
    for declared, result in zip(expected_chunks, chunk_results, strict=True):
        _validate_result(result, declared=declared, profile=profile, protocol=protocol)
        rows.append(
            {
                "prefix": str(declared["prefix"]),
                "declared": dict(declared),
                "result": dict(result),
            }
        )
    status = (
        DOBRIBAN_BH_BUNDLE_STATUS
        if profile["mode"] == DOBRIBAN_BH_FULL_EXECUTION_MODE
        else DOBRIBAN_BH_SMOKE_STATUS
    )
    bundle: dict[str, Any] = {
        "schema_version": DOBRIBAN_BH_BUNDLE_SCHEMA_VERSION,
        "kind": DOBRIBAN_BH_BUNDLE_KIND,
        "status": status,
        "evidence_class": "stress_test",
        "protocol": protocol.to_mapping(),
        "execution_profile": dict(profile),
        "chunks": rows,
        "source_hashes": dict(source_hashes),
        "claim_boundary": _CLAIM_BOUNDARY,
        "nonclaims": list(_NONCLAIMS),
    }
    bundle["result_sha256"] = dobriban_bh_bundle_digest(bundle)
    validate_dobriban_bh_stress_bundle(bundle, protocol=protocol)
    return bundle


def validate_dobriban_bh_stress_bundle(
    value: Mapping[str, Any], *, protocol: DobribanBHProtocol | None = None
) -> None:
    """Fail closed on a malformed, incomplete, promotable, or altered receipt."""
    bundle = _require_mapping(value, "Dobriban/BH stress bundle")
    _require_exact_keys(
        bundle,
        {
            "schema_version",
            "kind",
            "status",
            "evidence_class",
            "protocol",
            "execution_profile",
            "chunks",
            "source_hashes",
            "claim_boundary",
            "nonclaims",
            "result_sha256",
        },
        "Dobriban/BH stress bundle",
    )
    if bundle["schema_version"] != DOBRIBAN_BH_BUNDLE_SCHEMA_VERSION:
        raise ValueError("Dobriban/BH stress bundle schema version is unsupported")
    if bundle["kind"] != DOBRIBAN_BH_BUNDLE_KIND:
        raise ValueError("Dobriban/BH stress bundle kind is unsupported")
    if bundle["evidence_class"] != "stress_test":
        raise ValueError("Dobriban/BH stress bundle evidence class must be stress_test")
    active_protocol = DobribanBHProtocol(dict(_require_mapping(bundle["protocol"], "bundle.protocol")["protocol"])) if protocol is None else protocol
    if bundle["protocol"] != active_protocol.to_mapping():
        raise ValueError("Dobriban/BH stress bundle protocol mapping does not match the frozen protocol")
    profile = _validate_execution_profile(bundle["execution_profile"], active_protocol)
    expected_status = (
        DOBRIBAN_BH_BUNDLE_STATUS
        if profile["mode"] == DOBRIBAN_BH_FULL_EXECUTION_MODE
        else DOBRIBAN_BH_SMOKE_STATUS
    )
    if bundle["status"] != expected_status:
        raise ValueError("Dobriban/BH stress bundle status does not match its non-promotional execution mode")
    if bundle["claim_boundary"] != _CLAIM_BOUNDARY or bundle["nonclaims"] != list(_NONCLAIMS):
        raise ValueError("Dobriban/BH stress bundle claim boundary differs from the frozen non-claim contract")
    _validate_source_hashes(bundle["source_hashes"])
    rows = bundle["chunks"]
    if not isinstance(rows, list):
        raise ValueError("Dobriban/BH stress bundle chunks must be a list")
    expected_chunks = _declared_chunks(active_protocol)
    if len(rows) != len(expected_chunks):
        raise ValueError("Dobriban/BH stress bundle must retain all seven declared chunks")
    for declared, row in zip(expected_chunks, rows, strict=True):
        record = _require_mapping(row, f"Dobriban/BH bundle row {declared['prefix']}")
        _require_exact_keys(record, {"prefix", "declared", "result"}, "Dobriban/BH bundle row")
        if record["prefix"] != declared["prefix"] or record["declared"] != dict(declared):
            raise ValueError("Dobriban/BH bundle row differs from the frozen declared chunk")
        _validate_result(record["result"], declared=declared, profile=profile, protocol=active_protocol)
    if _require_sha256(bundle["result_sha256"], "bundle.result_sha256") != dobriban_bh_bundle_digest(bundle):
        raise ValueError("Dobriban/BH stress bundle result_sha256 does not match its contents")


__all__ = [
    "DOBRIBAN_BH_BUNDLE_FILENAME",
    "DOBRIBAN_BH_BUNDLE_KIND",
    "DOBRIBAN_BH_BUNDLE_SCHEMA_VERSION",
    "DOBRIBAN_BH_BUNDLE_STATUS",
    "DOBRIBAN_BH_FULL_EXECUTION_MODE",
    "DOBRIBAN_BH_SMOKE_EXECUTION_MODE",
    "DOBRIBAN_BH_SMOKE_STATUS",
    "build_dobriban_bh_stress_bundle",
    "compact_dobriban_bh_execution_profile",
    "dobriban_bh_bundle_digest",
    "frozen_dobriban_bh_execution_profile",
    "validate_dobriban_bh_execution_profile",
    "validate_dobriban_bh_stress_bundle",
]
