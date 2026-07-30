"""Measured-quality schema boundaries used by hydration and finalization."""

from __future__ import annotations

import copy

import pytest

from active_inference_power.schema import validate_quality_report


def _quality() -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "test_count": 7,
        "line_coverage_pct": 96.0,
        "branch_coverage_pct": 95.0,
        "coverage_gate_pct": 95.0,
        "branch_coverage_gate_pct": 95.0,
        "quality_pass": True,
        "quality_input_sha256": "a" * 64,
        "pytest_exit_code": 0,
        "command": "python -m pytest tests",
    }


def test_quality_report_requires_measured_boolean_terminal_evidence() -> None:
    report = _quality()
    validate_quality_report(report)

    malformed = copy.deepcopy(report)
    malformed["quality_pass"] = "true"
    with pytest.raises(ValueError, match="quality_pass must be boolean"):
        validate_quality_report(malformed)

    missing_input_witness = copy.deepcopy(report)
    missing_input_witness.pop("quality_input_sha256")
    with pytest.raises(ValueError, match="quality_input_sha256"):
        validate_quality_report(missing_input_witness)

    failed_exit = copy.deepcopy(report)
    failed_exit["pytest_exit_code"] = 1
    with pytest.raises(ValueError, match="pytest_exit_code=0"):
        validate_quality_report(failed_exit)

    no_collected_tests = copy.deepcopy(report)
    no_collected_tests["test_count"] = 0
    with pytest.raises(ValueError, match="test_count > 0"):
        validate_quality_report(no_collected_tests)

    invalid_percent = copy.deepcopy(report)
    invalid_percent["branch_coverage_pct"] = 101.0
    with pytest.raises(ValueError, match="branch_coverage_pct"):
        validate_quality_report(invalid_percent)

    weakened_line_gate = copy.deepcopy(report)
    weakened_line_gate["coverage_gate_pct"] = 90.0
    with pytest.raises(ValueError, match="coverage_gate_pct must be 95.0"):
        validate_quality_report(weakened_line_gate)

    weakened_branch_gate = copy.deepcopy(report)
    weakened_branch_gate["branch_coverage_gate_pct"] = 90.0
    with pytest.raises(ValueError, match="branch_coverage_gate_pct must be 95.0"):
        validate_quality_report(weakened_branch_gate)

    low_line_coverage = copy.deepcopy(report)
    low_line_coverage["line_coverage_pct"] = 94.9
    with pytest.raises(ValueError, match="must satisfy both declared coverage gates"):
        validate_quality_report(low_line_coverage)

    low_branch_coverage = copy.deepcopy(report)
    low_branch_coverage["branch_coverage_pct"] = 94.9
    with pytest.raises(ValueError, match="must satisfy both declared coverage gates"):
        validate_quality_report(low_branch_coverage)

    unsuccessful = copy.deepcopy(report)
    unsuccessful["quality_pass"] = False
    validate_quality_report(unsuccessful)
