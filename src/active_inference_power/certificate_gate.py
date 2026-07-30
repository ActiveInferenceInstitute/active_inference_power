"""Fail-closed policy reduction for finite-simulation certificate suites.

Every certificate record remains in the result artifact.  A declared policy
only distinguishes which predeclared names block a terminal release from which
remain visible diagnostic evidence.  The policy must be an exact partition of
the actual certificate names, preventing a failed check from being omitted or
quietly reclassified after execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .certificates import CERTIFICATE_NAMES


def _certificate_names(values: Sequence[object]) -> tuple[str, ...]:
    """Validate a non-empty, unique set of certificate identifiers."""
    names = tuple(values)
    if (
        not names
        or any(not isinstance(name, str) or not name or name != name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("certificate names must be unique non-empty strings")
    return tuple(str(name) for name in names)


def default_certificate_gate_policy() -> dict[str, list[str]]:
    """Return the conservative default: every configured certificate is terminal."""
    return {"terminal": list(CERTIFICATE_NAMES), "diagnostic": []}


def normalize_certificate_gate_policy(
    value: Mapping[str, Any],
    *,
    certificate_names: Sequence[object] = CERTIFICATE_NAMES,
) -> dict[str, list[str]]:
    """Validate one exact terminal/diagnostic partition.

    A diagnostic classification affects only the terminal gate.  The
    certificate itself, its verdict, and its membership in the policy remain
    serialized, so diagnostic failures cannot disappear from the evidence
    bundle.
    """
    expected = _certificate_names(certificate_names)
    if not isinstance(value, Mapping) or set(value) != {"terminal", "diagnostic"}:
        raise ValueError(
            "certificate_gate_policy must contain exactly terminal and diagnostic lists"
        )
    terminal = value["terminal"]
    diagnostic = value["diagnostic"]
    if not isinstance(terminal, list) or not isinstance(diagnostic, list):
        raise ValueError("certificate_gate_policy terminal and diagnostic must be lists")
    terminal_names = _certificate_names(terminal)
    diagnostic_names = tuple(diagnostic)
    if any(
        not isinstance(name, str) or not name or name != name.strip()
        for name in diagnostic_names
    ) or len(set(diagnostic_names)) != len(diagnostic_names):
        raise ValueError("certificate_gate_policy diagnostic names must be unique non-empty strings")
    if set(terminal_names).intersection(diagnostic_names):
        raise ValueError("certificate_gate_policy terminal and diagnostic names must be disjoint")
    declared = set(terminal_names).union(diagnostic_names)
    expected_set = set(expected)
    if declared != expected_set:
        missing = sorted(expected_set - declared)
        unknown = sorted(declared - expected_set)
        raise ValueError(
            "certificate_gate_policy must exactly partition certificate names; "
            f"missing={missing}, unknown={unknown}"
        )
    return {
        "terminal": list(terminal_names),
        "diagnostic": [str(name) for name in diagnostic_names],
    }


def certificate_gate_result(
    certificates: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate a declared policy against every actual certificate verdict."""
    if not isinstance(certificates, Mapping):
        raise ValueError("certificates must be a mapping")
    names = _certificate_names(list(certificates))
    normalized_policy = normalize_certificate_gate_policy(
        policy, certificate_names=names
    )
    verdicts: dict[str, bool] = {}
    for name in names:
        certificate = certificates[name]
        if not isinstance(certificate, Mapping) or not isinstance(
            certificate.get("passes"), bool
        ):
            raise ValueError(f"certificate {name!r} must contain a boolean passes verdict")
        verdicts[name] = bool(certificate["passes"])
    terminal_results = {
        name: verdicts[name] for name in normalized_policy["terminal"]
    }
    diagnostic_results = {
        name: verdicts[name] for name in normalized_policy["diagnostic"]
    }
    terminal_failures = [
        name for name, passed in terminal_results.items() if not passed
    ]
    diagnostic_failures = [
        name for name, passed in diagnostic_results.items() if not passed
    ]
    return {
        "policy": normalized_policy,
        "effective_terminal_results": terminal_results,
        "diagnostic_results": diagnostic_results,
        "terminal_failures": terminal_failures,
        "diagnostic_failures": diagnostic_failures,
        "terminal_certificates_pass": not terminal_failures,
        "diagnostic_certificates_pass": not diagnostic_failures,
        "all_certificates_pass": not terminal_failures and not diagnostic_failures,
    }


def validate_certificate_gate_result(
    value: Mapping[str, Any], certificates: Mapping[str, Any]
) -> dict[str, Any]:
    """Require serialized gate metadata to equal its recomputed verdicts."""
    if not isinstance(value, Mapping):
        raise ValueError("certificate gate result must be a mapping")
    policy = value.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("certificate gate result must contain a policy mapping")
    expected = certificate_gate_result(certificates, policy)
    if dict(value) != expected:
        raise ValueError("certificate gate result does not match certificate verdicts")
    return expected


def effective_certificate_gate_result(
    certificates: Mapping[str, Any], value: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return a checked serialized result or the conservative unannotated default.

    Prior bundles may not serialize a gate policy.  Treating those records as
    all-terminal is conservative and preserves their original semantics;
    removing a newer policy therefore cannot turn a failed diagnostic into a
    passing release.
    """
    if value is None:
        return certificate_gate_result(certificates, default_certificate_gate_policy())
    return validate_certificate_gate_result(value, certificates)


__all__ = [
    "default_certificate_gate_policy",
    "normalize_certificate_gate_policy",
    "certificate_gate_result",
    "validate_certificate_gate_result",
    "effective_certificate_gate_result",
]
