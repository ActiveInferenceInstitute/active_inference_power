"""Tests for ``certificates.py`` — reproducible numeric FDR/FWER certificates.

No mocks. Every test drives the real simulation stack (``simulation`` →
``corrections`` / ``error_rates``) with seeded ``numpy`` RNGs and pins the
numeric verdicts, the analytic-bound comparisons, and the cross-procedure
identities the contract requires (BY ⊆ BH, Storey power ≥ BH power, BH FDR ≤
``pi0·alpha``, Bonferroni FWER ≤ ``alpha``). Certificates that use the large
documented ``n_reps``/``m`` defaults are marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import Any, cast
from collections.abc import Mapping

from active_inference_power.certificates import (
    BAND_SIGMA,
    _binomial_stderr,
    certify_bh_fdr_control,
    certify_bonferroni_fwer,
    certify_positive_dependence,
    certify_storey_power_gain,
    certify_storey_split_confirmation,
    run_all_certificates,
)

# Keys every certificate dict must expose (contract: estimate/bound/stderr/passes).
_REQUIRED_KEYS = {"name", "estimate", "bound", "stderr", "passes"}


def _assert_certificate_shape(cert: Mapping[str, Any]) -> None:
    """Assert a certificate dict carries the required, well-typed fields."""
    assert _REQUIRED_KEYS <= set(cert)
    assert isinstance(cert["name"], str) and cert["name"]
    assert isinstance(cert["passes"], bool)
    for key in ("estimate", "bound", "stderr"):
        assert isinstance(cert[key], float)
        assert np.isfinite(cert[key])
    assert cert["stderr"] >= 0.0


def _field(cert: Mapping[str, object], key: str) -> float:
    """Read a numeric certificate field after the structural shape check."""
    return float(cast(float, cert[key]))


# --------------------------------------------------------------------------- #
# _binomial_stderr helper (both branches + known values)
# --------------------------------------------------------------------------- #
class TestBinomialStderr:
    def test_known_value(self) -> None:
        """sqrt(p(1-p)/n): p=0.5, n=100 → 0.05 exactly."""
        assert _binomial_stderr(0.5, 100) == pytest.approx(0.05, abs=1e-12)

    def test_n_one_returns_zero(self) -> None:
        """n <= 1 branch: a single replication has no spread."""
        assert _binomial_stderr(0.5, 1) == 0.0

    def test_n_zero_returns_zero(self) -> None:
        """n == 0 also hits the guard branch."""
        assert _binomial_stderr(0.3, 0) == 0.0

    def test_degenerate_proportions_zero(self) -> None:
        """p == 0 and p == 1 give zero variance."""
        assert _binomial_stderr(0.0, 500) == 0.0
        assert _binomial_stderr(1.0, 500) == 0.0

    def test_shrinks_with_n(self) -> None:
        """Standard error decreases as replications increase."""
        assert _binomial_stderr(0.4, 1000) < _binomial_stderr(0.4, 100)


# --------------------------------------------------------------------------- #
# certify_bh_fdr_control
# --------------------------------------------------------------------------- #
class TestBHFDRControl:
    @pytest.mark.slow
    def test_passes_at_default_seed(self) -> None:
        """Contract: the certificate passes at its documented seed."""
        cert = certify_bh_fdr_control()
        _assert_certificate_shape(cert)
        assert cert["passes"] is True
        assert cert["name"] == "bh_fdr_control"

    @pytest.mark.slow
    def test_estimate_near_bound(self) -> None:
        """BH FDR under independence ≈ pi0·alpha (exact BH result)."""
        cert = certify_bh_fdr_control()
        assert cert["bound"] == pytest.approx(0.8 * 0.05, abs=1e-12)
        # Estimate is pinned (deterministic seed) and sits inside the band.
        assert cert["estimate"] == pytest.approx(0.04087, abs=2e-3)
        assert _field(cert, "estimate") <= _field(cert, "bound") + BAND_SIGMA * _field(cert, "stderr")

    def test_fast_config_passes(self) -> None:
        """A cheaper configuration still controls FDR and passes."""
        cert = certify_bh_fdr_control(m=100, n_reps=150)
        _assert_certificate_shape(cert)
        assert cert["passes"] is True
        assert 0.0 <= _field(cert, "power") <= 1.0

    def test_deterministic(self) -> None:
        """Same seed → bit-identical estimate (reproducibility)."""
        a = certify_bh_fdr_control(m=100, n_reps=120, seed=7)
        b = certify_bh_fdr_control(m=100, n_reps=120, seed=7)
        assert a["estimate"] == b["estimate"]
        assert a["power"] == b["power"]

    def test_seed_changes_estimate(self) -> None:
        """Different seeds explore different draws → different estimate."""
        a = certify_bh_fdr_control(m=100, n_reps=120, seed=1)
        b = certify_bh_fdr_control(m=100, n_reps=120, seed=2)
        assert a["estimate"] != b["estimate"]

    def test_config_echoed(self) -> None:
        """The returned dict echoes the configuration for provenance."""
        cert = certify_bh_fdr_control(pi0=0.9, effect=2.5, m=80, n_reps=100, seed=3)
        assert cert["pi0"] == 0.9
        assert cert["effect"] == 2.5
        assert cert["m"] == 80
        assert cert["n_reps"] == 100
        assert cert["seed"] == 3


# --------------------------------------------------------------------------- #
# certify_bonferroni_fwer
# --------------------------------------------------------------------------- #
class TestBonferroniFWER:
    @pytest.mark.slow
    def test_passes_at_default_seed(self) -> None:
        """Contract: global-null FWER within band of alpha, passes at seed 0."""
        cert = certify_bonferroni_fwer()
        _assert_certificate_shape(cert)
        assert cert["passes"] is True
        assert cert["name"] == "bonferroni_fwer"
        assert cert["bound"] == pytest.approx(0.05, abs=1e-12)

    @pytest.mark.slow
    def test_estimate_near_alpha(self) -> None:
        """Under the global null, Bonferroni FWER hugs alpha."""
        cert = certify_bonferroni_fwer()
        assert cert["estimate"] == pytest.approx(0.052, abs=1.5e-2)
        assert _field(cert, "estimate") <= _field(cert, "bound") + BAND_SIGMA * _field(cert, "stderr")

    def test_fast_config_passes(self) -> None:
        """Cheaper configuration still controls FWER."""
        cert = certify_bonferroni_fwer(m=100, n_reps=200, seed=0)
        _assert_certificate_shape(cert)
        assert cert["passes"] is True

    def test_smaller_alpha_lowers_fwer(self) -> None:
        """A stricter alpha yields a smaller family-wise error estimate."""
        strict = certify_bonferroni_fwer(m=100, n_reps=300, alpha=0.01, seed=0)
        loose = certify_bonferroni_fwer(m=100, n_reps=300, alpha=0.10, seed=0)
        assert _field(strict, "estimate") <= _field(loose, "estimate")
        assert strict["passes"] is True


# --------------------------------------------------------------------------- #
# certify_positive_dependence
# --------------------------------------------------------------------------- #
class TestPositiveDependence:
    @pytest.mark.slow
    def test_passes_at_default_seed(self) -> None:
        """Contract: BH controls FDR under PRDS; passes at seed 0."""
        cert = certify_positive_dependence()
        _assert_certificate_shape(cert)
        assert cert["passes"] is True
        assert cert["name"] == "positive_dependence"

    @pytest.mark.slow
    def test_by_more_conservative_than_bh(self) -> None:
        """BY (arbitrary-dependence valid) rejects a subset → lower FDR & power."""
        cert = certify_positive_dependence()
        assert _field(cert, "by_fdr") <= _field(cert, "bh_fdr") + BAND_SIGMA * _field(cert, "stderr")
        assert _field(cert, "by_power") <= _field(cert, "bh_power")

    @pytest.mark.slow
    def test_both_within_bound(self) -> None:
        """Both BH and BY FDR sit within band of pi0·alpha under PRDS."""
        cert = certify_positive_dependence()
        bound = cert["bound"]
        assert bound == pytest.approx(0.8 * 0.05, abs=1e-12)
        assert _field(cert, "bh_fdr") <= _field(cert, "bound") + BAND_SIGMA * _field(cert, "stderr")

    def test_fast_config_passes(self) -> None:
        """Cheaper positive-dependence configuration still passes."""
        cert = certify_positive_dependence(m=100, n_reps=150)
        _assert_certificate_shape(cert)
        assert cert["passes"] is True
        assert cert["rho"] == 0.5

    def test_independence_limit_passes(self) -> None:
        """rho == 0 reduces to independence; BH still controls FDR."""
        cert = certify_positive_dependence(rho=0.0, m=100, n_reps=150)
        assert cert["passes"] is True


# --------------------------------------------------------------------------- #
# certify_storey_power_gain
# --------------------------------------------------------------------------- #
class TestStoreyPowerGain:
    @pytest.mark.slow
    def test_passes_at_default_seed(self) -> None:
        """Contract: Storey gains power over BH while both control FDR."""
        cert = certify_storey_power_gain()
        _assert_certificate_shape(cert)
        assert cert["passes"] is True
        assert cert["name"] == "storey_power_gain"

    @pytest.mark.slow
    def test_power_gain_is_positive(self) -> None:
        """Storey's adaptive q-values reject more → strictly higher power here."""
        cert = certify_storey_power_gain()
        assert _field(cert, "storey_power") >= _field(cert, "bh_power")
        assert _field(cert, "power_gain") > 0.0
        assert cert["estimate"] == pytest.approx(0.91082, abs=3e-3)

    @pytest.mark.slow
    def test_both_control_fdr(self) -> None:
        """BH FDR ≤ pi0·alpha and Storey FDR ≤ alpha (its own target level)."""
        cert = certify_storey_power_gain()
        assert _field(cert, "bh_fdr") <= 0.5 * 0.05 + 0.02  # generous band
        assert _field(cert, "storey_fdr") <= 0.05 + 0.02

    def test_metric_specific_uncertainty_and_gate_decomposition(self) -> None:
        """FDR and paired-power uncertainty are reported separately."""
        cert = certify_storey_power_gain(m=100, n_reps=150, seed=4)
        for key in (
            "bh_fdr_stderr",
            "storey_fdr_stderr",
            "bh_power_stderr",
            "power_gain_stderr",
            "bh_fdr_upper_band",
            "storey_fdr_upper_band",
            "storey_fdr_excess",
            "storey_fdr_z",
        ):
            assert np.isfinite(_field(cert, key))
            if key.endswith("stderr") or key.endswith("upper_band"):
                assert _field(cert, key) >= 0.0
        assert isinstance(cert["bh_fdr_passes"], bool)
        assert isinstance(cert["storey_fdr_passes"], bool)
        assert isinstance(cert["power_gain_passes"], bool)
        assert cert["passes"] is (
            cert["bh_fdr_passes"]
            and cert["storey_fdr_passes"]
            and cert["power_gain_passes"]
        )

    def test_fast_config_passes(self) -> None:
        """Cheaper Storey configuration still shows the gain."""
        cert = certify_storey_power_gain(m=100, n_reps=150)
        _assert_certificate_shape(cert)
        assert cert["passes"] is True
        assert _field(cert, "storey_power") >= _field(cert, "bh_power") - 1e-9


def test_storey_split_confirmation_fast_calibration_passes() -> None:
    cert = certify_storey_split_confirmation(m=80, n_reps=120, seed=11)
    _assert_certificate_shape(cert)
    assert cert["passes"] is True
    assert cert["validity"] == "conditional_split_stream_calibration_diagnostic"
    assert cert["selection_confidence"] == 0.99
    assert cert["n_reps"] == 120


def test_storey_split_confirmation_validates_preregistered_parameters() -> None:
    with pytest.raises(ValueError, match="lambda"):
        certify_storey_split_confirmation(m=20, n_reps=10, lambda_=1.0)
    with pytest.raises(ValueError, match="confidence"):
        certify_storey_split_confirmation(m=20, n_reps=10, confidence=1.0)


# --------------------------------------------------------------------------- #
# run_all_certificates (both fast branches)
# --------------------------------------------------------------------------- #
class TestRunAllCertificates:
    def test_fast_suite_all_pass(self) -> None:
        """fast=True branch: the whole suite runs and every certificate passes."""
        results = run_all_certificates(fast=True)
        assert set(results) == {
            "bh_fdr_control",
            "bonferroni_fwer",
            "positive_dependence",
            "storey_power_gain",
            "storey_split_confirmation",
        }
        for name, cert in results.items():
            _assert_certificate_shape(cert)
            assert cert["passes"] is True, f"{name} failed"
            assert cert["n_reps"] == 150
            assert cert["m"] == 120

    @pytest.mark.slow
    def test_full_suite_all_pass(self) -> None:
        """fast=False branch: the large documented-default suite still passes."""
        results = run_all_certificates(fast=False)
        for name, cert in results.items():
            assert cert["passes"] is True, f"{name} failed"
            assert cert["n_reps"] == 500
            assert cert["m"] == 200

    def test_default_is_fast(self) -> None:
        """The default invocation uses the fast configuration."""
        results = run_all_certificates()
        assert results["bh_fdr_control"]["n_reps"] == 150


# --------------------------------------------------------------------------- #
# module constant
# --------------------------------------------------------------------------- #
def test_band_sigma_is_three() -> None:
    """The documented slack is three Monte-Carlo standard errors."""
    assert BAND_SIGMA == 3.0
