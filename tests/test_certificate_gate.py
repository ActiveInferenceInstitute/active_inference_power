"""Real certificate-gate partition and provenance tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from active_inference_power.certificate_gate import (
    certificate_gate_result,
    default_certificate_gate_policy,
    effective_certificate_gate_result,
    normalize_certificate_gate_policy,
    validate_certificate_gate_result,
)
from active_inference_power.certificates import CERTIFICATE_NAMES, run_all_certificates
from scripts.generate_results import _certificate_gate_exit_status, _load_config


ROOT = Path(__file__).resolve().parent.parent


def test_default_gate_policy_is_the_exact_actual_certificate_partition() -> None:
    certificates = run_all_certificates(fast=True, m=12, n_reps=3, seed=17)
    assert tuple(certificates) == CERTIFICATE_NAMES
    policy = default_certificate_gate_policy()
    assert policy == {"terminal": list(CERTIFICATE_NAMES), "diagnostic": []}
    gate = certificate_gate_result(certificates, policy)
    assert gate["policy"] == policy
    assert gate["effective_terminal_results"] == {
        name: certificates[name]["passes"] for name in CERTIFICATE_NAMES
    }
    assert gate["diagnostic_results"] == {}
    assert gate["terminal_certificates_pass"] == all(
        certificates[name]["passes"] for name in CERTIFICATE_NAMES
    )
    assert validate_certificate_gate_result(gate, certificates) == gate


def test_isolated_diagnostic_policy_keeps_a_real_failed_certificate_visible(
    small_real_certificates: dict[str, dict[str, object]],
) -> None:
    certificates = small_real_certificates
    assert certificates["storey_power_gain"]["passes"] is False
    policy = {
        "terminal": [name for name in CERTIFICATE_NAMES if name != "storey_power_gain"],
        "diagnostic": ["storey_power_gain"],
    }
    gate = certificate_gate_result(certificates, policy)
    assert gate["terminal_certificates_pass"] is True
    assert gate["diagnostic_certificates_pass"] is False
    assert gate["diagnostic_failures"] == ["storey_power_gain"]
    assert gate["diagnostic_results"] == {"storey_power_gain": False}
    result = {"meta": {"certificate_gate": gate}, "certificates": certificates}
    assert _certificate_gate_exit_status(result) == 0

    tampered = {**gate, "diagnostic_failures": []}
    with pytest.raises(ValueError, match="does not match"):
        validate_certificate_gate_result(tampered, certificates)
    assert (
        _certificate_gate_exit_status(
            {"meta": {"certificate_gate": tampered}, "certificates": certificates}
        )
        == 1
    )


@pytest.mark.parametrize(
    "policy",
    [
        {"terminal": list(CERTIFICATE_NAMES), "diagnostic": ["storey_power_gain"]},
        {"terminal": list(CERTIFICATE_NAMES[:-1]), "diagnostic": []},
        {"terminal": ["unknown"], "diagnostic": list(CERTIFICATE_NAMES[1:])},
        {"terminal": [], "diagnostic": list(CERTIFICATE_NAMES)},
        {"terminal": list(CERTIFICATE_NAMES), "diagnostic": [], "extra": []},
    ],
)
def test_gate_policy_rejects_non_partitioned_or_unknown_names(
    policy: dict[str, list[str]],
) -> None:
    with pytest.raises(ValueError):
        normalize_certificate_gate_policy(policy)


def test_gate_policy_and_certificate_payload_type_boundaries_fail_closed() -> None:
    """Exercise real malformed payload boundaries without changing the evidence set."""
    certificates = run_all_certificates(fast=True, m=12, n_reps=3, seed=29)
    policy = default_certificate_gate_policy()

    with pytest.raises(ValueError, match="must be lists"):
        normalize_certificate_gate_policy(
            {"terminal": tuple(CERTIFICATE_NAMES), "diagnostic": []}
        )
    with pytest.raises(ValueError, match="diagnostic names"):
        normalize_certificate_gate_policy(
            {"terminal": list(CERTIFICATE_NAMES[:-1]), "diagnostic": [" "]}
        )
    with pytest.raises(ValueError, match="certificates must be a mapping"):
        certificate_gate_result([], policy)  # type: ignore[arg-type]

    malformed_verdict = dict(certificates)
    malformed_verdict[CERTIFICATE_NAMES[0]] = {"passes": "yes"}
    with pytest.raises(ValueError, match="boolean passes"):
        certificate_gate_result(malformed_verdict, policy)


def test_serialized_gate_result_requires_a_mapping_and_policy_mapping() -> None:
    certificates = run_all_certificates(fast=True, m=12, n_reps=3, seed=31)
    with pytest.raises(ValueError, match="result must be a mapping"):
        validate_certificate_gate_result([], certificates)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="contain a policy mapping"):
        validate_certificate_gate_result({}, certificates)


def test_canonical_config_is_explicitly_all_terminal(
    small_real_certificates: dict[str, dict[str, object]],
) -> None:
    experiment = _load_config(ROOT / "manuscript" / "config.yaml")["experiment"]
    assert experiment["certificate_gate_policy"] == default_certificate_gate_policy()
    # Missing metadata remains conservative, never diagnostic.
    unannotated = effective_certificate_gate_result(small_real_certificates)
    assert unannotated["terminal_certificates_pass"] is False
    assert unannotated["terminal_failures"] == ["storey_power_gain"]


def test_frozen_v2_campaign_config_keeps_same_vector_storey_visible_but_diagnostic() -> (
    None
):
    config_path = (
        ROOT / "data" / "research_lanes" / "conditional_power_campaign_v2.yaml"
    )
    config = _load_config(config_path)
    assert config["research_lane"]["output_namespace"] == "output/campaign-v2"
    assert config["experiment"]["certificate_gate_policy"] == {
        "terminal": [
            "bh_fdr_control",
            "bonferroni_fwer",
            "positive_dependence",
            "storey_split_confirmation",
        ],
        "diagnostic": ["storey_power_gain"],
    }
