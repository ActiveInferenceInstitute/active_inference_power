"""Shared data contracts for the active-inference multiple-testing package.

Every scientific module in ``src/`` imports its result containers from here so
that procedures (``corrections``), evaluators (``error_rates``), simulators
(``simulation``), the active-inference decision rule (``active_inference``),
and the reproducible ``certificates`` all speak one vocabulary.

All arrays are ``numpy.ndarray`` of dtype float/bool. Nothing in this module
imports ``infrastructure.*`` — it is pure Layer-2 domain code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# A p-value never legitimately exceeds 1 nor drops below 0; clamp tolerance for
# floating-point noise arriving from analytic survival functions.
PVAL_EPS: float = 1e-15

# Canonical serialized method identifiers shared by correction results and
# simulation summaries. Short aliases are rejected at every public boundary so
# a result record cannot silently reintroduce a retired procedure spelling.
CANONICAL_CORRECTION_METHODS = frozenset(
    {
        "bonferroni",
        "sidak",
        "holm",
        "hochberg",
        "benjamini_hochberg",
        "benjamini_yekutieli",
        "storey_qvalue",
        "storey_qvalue_conservative",
        "storey_qvalue_split_confirmation",
        "adaptive_bh",
        "weighted_bh",
    }
)


@dataclass
class CorrectionResult:
    """Outcome of applying one multiple-testing correction to a p-value vector.

    Attributes:
        method: Canonical name of the procedure (e.g. ``"benjamini_hochberg"``).
        adjusted: Adjusted p-values (a.k.a. q-values), aligned to input order.
            Comparing ``adjusted <= alpha`` reproduces ``rejected`` for step-up
            and single-step procedures.
        rejected: Boolean mask of hypotheses declared significant.
        alpha: Nominal level the rejection mask was computed at.
        threshold: Largest raw p-value that was rejected (0.0 if none rejected).
    """

    method: str
    adjusted: np.ndarray
    rejected: np.ndarray
    alpha: float
    threshold: float

    def __post_init__(self) -> None:
        """Validate the result invariant at the public API boundary."""
        if (
            not isinstance(self.method, str)
            or self.method not in CANONICAL_CORRECTION_METHODS
        ):
            raise ValueError(
                "method must be one of the canonical correction identifiers, "
                f"got {self.method!r}"
            )
        adjusted = np.asarray(self.adjusted, dtype=float)
        rejected = np.asarray(self.rejected, dtype=bool)
        if adjusted.ndim != 1 or rejected.ndim != 1:
            raise ValueError(
                "adjusted and rejected must be 1-D arrays; got "
                f"{adjusted.ndim} and {rejected.ndim} dimensions"
            )
        if adjusted.shape != rejected.shape:
            raise ValueError(
                "adjusted and rejected must have matching shapes; got "
                f"{adjusted.shape} and {rejected.shape}"
            )
        if not np.all(np.isfinite(adjusted)):
            raise ValueError("adjusted p-values must be finite")
        if not np.all((0.0 <= adjusted) & (adjusted <= 1.0)):
            raise ValueError("adjusted p-values must lie in [0, 1]")
        if not np.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got {self.alpha!r}")
        if not np.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must lie in [0, 1], got {self.threshold!r}")
        if not np.array_equal(rejected, adjusted <= float(self.alpha)):
            raise ValueError("rejected must equal adjusted <= alpha")
        self.adjusted = adjusted
        self.rejected = rejected

    @property
    def num_rejected(self) -> int:
        """Number of hypotheses declared significant."""
        return int(np.count_nonzero(self.rejected))


@dataclass
class ErrorSummary:
    """Realized error decomposition for a single simulated/observed dataset.

    Given a rejection mask and the ground-truth alternative mask, this records
    the confusion-matrix quantities and the derived per-dataset rates. The false
    discovery proportion (``fdp``) is defined as 0 when nothing is rejected,
    matching the Benjamini-Hochberg convention.
    """

    num_hypotheses: int
    num_alt: int
    num_rejected: int
    true_discoveries: int
    false_discoveries: int
    fdp: float
    power: float
    fwer_event: bool

    @property
    def num_null(self) -> int:
        """Number of true null hypotheses."""
        return self.num_hypotheses - self.num_alt


@dataclass
class SimulationResult:
    """Monte-Carlo aggregate of a procedure's operating characteristics.

    Aggregated over ``n_reps`` independent replications of a data-generating
    model. ``fdr`` estimates E[FDP], ``power`` estimates E[TPR] (average power),
    and ``fwer`` estimates P(at least one false rejection).
    """

    method: str
    n_reps: int
    fdr: float
    power: float
    fwer: float
    mean_rejections: float
    fdp_samples: np.ndarray = field(repr=False)
    power_samples: np.ndarray = field(repr=False)
    fwer_samples: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate aggregate/sample alignment and probability ranges."""
        if (
            not isinstance(self.method, str)
            or self.method not in CANONICAL_CORRECTION_METHODS
        ):
            raise ValueError(
                "method must be one of the canonical correction identifiers, "
                f"got {self.method!r}"
            )
        if self.n_reps < 1:
            raise ValueError(f"n_reps must be >= 1, got {self.n_reps!r}")
        fdp = np.asarray(self.fdp_samples, dtype=float)
        power = np.asarray(self.power_samples, dtype=float)
        fwer = (
            np.asarray(self.fwer_samples, dtype=float)
            if self.fwer_samples is not None
            else (fdp > 0.0).astype(float)
        )
        if fdp.shape != (self.n_reps,) or power.shape != (self.n_reps,):
            raise ValueError(
                "sample arrays must each have shape (n_reps,), got "
                f"{fdp.shape} and {power.shape} for n_reps={self.n_reps}"
            )
        if fwer.shape != (self.n_reps,):
            raise ValueError(
                "fwer_samples must have shape (n_reps,), got "
                f"{fwer.shape} for n_reps={self.n_reps}"
            )
        if (
            not np.all(np.isfinite(fdp))
            or not np.all(np.isfinite(power))
            or not np.all(np.isfinite(fwer))
        ):
            raise ValueError("simulation samples must be finite")
        if not np.all((0.0 <= fdp) & (fdp <= 1.0)):
            raise ValueError("fdp_samples must lie in [0, 1]")
        if not np.all((0.0 <= power) & (power <= 1.0)):
            raise ValueError("power_samples must lie in [0, 1]")
        if not np.all(np.isin(fwer, (0.0, 1.0))):
            raise ValueError("fwer_samples must contain only 0/1 events")
        for name in ("fdr", "power", "fwer"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
        expected_aggregates = {
            "fdr": float(np.mean(fdp)),
            "power": float(np.mean(power)),
            "fwer": float(np.mean(fwer)),
        }
        for name, expected in expected_aggregates.items():
            reported = float(getattr(self, name))
            if not np.isclose(reported, expected, rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"{name} must equal the mean of its per-replication samples"
                )
        if not np.isfinite(self.mean_rejections) or self.mean_rejections < 0.0:
            raise ValueError(
                f"mean_rejections must be finite and non-negative, got {self.mean_rejections!r}"
            )
        self.fdp_samples = fdp
        self.power_samples = power
        self.fwer_samples = fwer

    def mc_stderr(self, metric: str = "fdr") -> float:
        """Monte-Carlo standard error of ``fdr`` or ``power`` across reps."""
        fwer_samples = self.fwer_samples
        if fwer_samples is None:
            fwer_samples = (self.fdp_samples > 0.0).astype(float)
        samples_by_metric: dict[str, np.ndarray] = {
            "fdr": self.fdp_samples,
            "power": self.power_samples,
            "fwer": fwer_samples,
        }
        if metric not in samples_by_metric:
            raise ValueError(
                f"metric must be 'fdr', 'power', or 'fwer', got {metric!r}"
            )
        samples = samples_by_metric[metric]
        n = len(samples)
        if n <= 1:
            return 0.0
        return float(np.std(samples, ddof=1) / np.sqrt(n))


@dataclass
class TwoGroupsConfig:
    """Parameters of the canonical two-groups multiple-testing model.

    ``m`` hypotheses, a fraction ``pi0`` of which are true nulls. Non-null
    ("alternative") test statistics have mean shift ``effect`` (in units of the
    test statistic's null standard deviation). ``rho`` and ``dependence_kind``
    select the Gaussian dependence regime. ``sided`` is 1 or 2.
    """

    m: int
    pi0: float
    effect: float
    rho: float = 0.0
    sided: int = 1
    seed: int = 0
    dependence_kind: Literal["independent", "equicorrelated", "block", "factor"] = "factor"
    block_size: int = 1

    def __post_init__(self) -> None:
        """Validate the two-groups data-generating process."""
        if isinstance(self.m, bool) or int(self.m) != self.m or self.m < 0:
            raise ValueError(f"m must be a non-negative integer, got {self.m!r}")
        if not np.isfinite(self.pi0) or not 0.0 <= self.pi0 <= 1.0:
            raise ValueError(f"pi0 must lie in [0, 1], got {self.pi0!r}")
        if not np.isfinite(self.effect):
            raise ValueError(f"effect must be finite, got {self.effect!r}")
        if self.dependence_kind not in {"factor", "independent", "equicorrelated", "block"}:
            raise ValueError(f"unknown dependence_kind: {self.dependence_kind!r}")
        if not np.isfinite(self.rho):
            raise ValueError(f"rho must be finite, got {self.rho!r}")
        if self.dependence_kind == "independent" and self.rho != 0.0:
            raise ValueError(
                "rho must be zero for independent dependence; nonzero rho would be ignored"
            )
        if self.dependence_kind == "factor" and not 0.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must lie in [0, 1] for factor dependence, got {self.rho!r}")
        if self.dependence_kind == "equicorrelated":
            if self.m <= 1:
                if self.rho != 0.0:
                    raise ValueError("one-dimensional equicorrelation must be zero")
            elif not -1.0 / (self.m - 1) <= self.rho < 1.0:
                raise ValueError("equicorrelated rho is outside the PSD range")
        if self.dependence_kind == "block":
            if isinstance(self.block_size, bool) or int(self.block_size) != self.block_size:
                raise ValueError("block_size must be an integer")
            if not 1 <= int(self.block_size) <= max(self.m, 1):
                raise ValueError("block_size must lie in [1, m]")
            if int(self.block_size) == 1:
                if self.rho != 0.0:
                    raise ValueError("block rho must be zero when block_size=1")
            elif not -1.0 / (int(self.block_size) - 1) <= self.rho < 1.0:
                raise ValueError("block rho is outside the PSD range")
        if self.sided not in (1, 2):
            raise ValueError(f"sided must be 1 or 2, got {self.sided!r}")

    @property
    def m0(self) -> int:
        """Number of true null hypotheses (nearest integer, documented rule)."""
        return int(round(self.m * self.pi0))

    @property
    def m1(self) -> int:
        """Number of true alternative hypotheses."""
        return self.m - self.m0
