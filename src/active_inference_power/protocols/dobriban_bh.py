"""Offline adapter for the publicly documented two-sided Gaussian BH stress protocol.

This module imports the *protocol*, not the external theorem.  It implements
the publicly documented finite stratified Monte-Carlo design with the same model,
two-sided p-values, ordinary BH rule, PCG64 generator, and child-stream seed
construction.  The outward-rounded Arb certificate in the source bundle is a
separate evidence layer and is never represented as a local Python result.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import t as student_t

_PLAN_PATH = Path(__file__).with_name("dobriban_bh_plan.json")
_SOURCE_URL = "https://github.com/dobriban/BH"
_SOURCE_PLAN_SHA256 = "d9805d2478f02ab3ed2b88a7933fdbe37d0129f3996d6a15449c4efb7d1baccf"
_SOURCE_GENERATOR_SHA256 = "8a6a5ad656a66675eb86ed0fb8f2ba8e38751acbd89a1ef00795f8a817a38861"


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DobribanBHProtocol:
    """Validated parameters of the imported publicly documented protocol."""

    plan: dict[str, Any]
    source_url: str = _SOURCE_URL
    source_plan_sha256: str = _SOURCE_PLAN_SHA256
    source_generator_sha256: str = _SOURCE_GENERATOR_SHA256

    def __post_init__(self) -> None:
        if not self.source_url.startswith("https://"):
            raise ValueError("protocol source_url must be HTTPS")
        for name in ("source_plan_sha256", "source_generator_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        model = self.plan.get("model")
        if not isinstance(model, dict):
            raise ValueError("protocol plan must contain a model mapping")
        expected = {
            "model_id": "bh-two-sided-gaussian-counterexample-v1",
            "alpha": 0.01,
            "pi0": 0.96,
            "rho0": 0.3,
            "rho1": -0.3,
            "rho2": -0.72,
            "mu1": 2.4,
            "mu2": 4.4,
            "group_multipliers": [96, 1, 3],
        }
        if model != expected:
            raise ValueError("protocol model differs from the imported publicly documented plan")
        if self.plan.get("schema_version") != 1:
            raise ValueError("protocol schema_version must be 1")
        if self.plan.get("rng_bit_generator") != "PCG64":
            raise ValueError("protocol RNG must be PCG64")
        if self.plan.get("seed_construction") != "SeedSequence(master_seed).spawn(1)[0]":
            raise ValueError("protocol seed construction differs from the publicly documented plan")
        if self.plan.get("expected_seed_spawn_key") != [0]:
            raise ValueError("protocol child spawn key differs from the publicly documented plan")
        if self.plan.get("strata") != 1000 or self.plan.get("target_entries") != 1500000:
            raise ValueError("protocol batching parameters differ from the publicly documented plan")
        if self.plan.get("expected_environment") != {
            "python": "3.12.3",
            "numpy": "1.26.4",
            "scipy": "1.14.1",
            "python_flint": "0.8.0",
        }:
            raise ValueError("protocol environment lock differs from the publicly documented plan")
        if int(self.plan.get("strata", 0)) < 1 or int(self.plan.get("target_entries", 0)) < 1:
            raise ValueError("protocol strata and target_entries must be positive")
        chunks = self.plan.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("protocol must declare non-empty chunks")
        required = {"prefix", "N", "block", "master_seed", "macro_replications"}
        if any(not isinstance(chunk, dict) or not required <= set(chunk) for chunk in chunks):
            raise ValueError("protocol chunks are incomplete")
        expected_chunks = [
            {"prefix": "fdr_N50", "N": 50, "block": 1, "master_seed": 20260711, "macro_replications": 100},
            {"prefix": "fdr_N100_a", "N": 100, "block": 1, "master_seed": 20260712, "macro_replications": 50},
            {"prefix": "fdr_N100_b", "N": 100, "block": 2, "master_seed": 20260713, "macro_replications": 50},
            {"prefix": "fdr_N200_a", "N": 200, "block": 1, "master_seed": 20260720, "macro_replications": 25},
            {"prefix": "fdr_N200_b", "N": 200, "block": 2, "master_seed": 20260721, "macro_replications": 25},
            {"prefix": "fdr_N200_c", "N": 200, "block": 3, "master_seed": 20260722, "macro_replications": 25},
            {"prefix": "fdr_N200_d", "N": 200, "block": 4, "master_seed": 20260723, "macro_replications": 25},
        ]
        if chunks != expected_chunks:
            raise ValueError("protocol chunks differ from the imported publicly documented plan")

    @property
    def protocol_hash(self) -> str:
        return _digest(self.plan)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "2.0",
            "protocol": self.plan,
            "protocol_hash": self.protocol_hash,
            "source_url": self.source_url,
            "source_plan_sha256": self.source_plan_sha256,
            "source_generator_sha256": self.source_generator_sha256,
            "evidence_class": "stress_test",
            "theorem_execution": "external_arb_certificate_not_executed_by_this_adapter",
        }


@dataclass(frozen=True)
class DobribanBHResult:
    """Finite stratified-Monte-Carlo summary for one protocol chunk."""

    model_id: str
    protocol_hash: str
    N: int
    m: int
    strata: int
    macro_replications: int
    datasets: int
    master_seed: int
    seed_spawn_key: tuple[int, ...]
    target_entries: int
    fdr_hat: float
    mcse: float
    ci95_low: float
    ci95_high: float
    one_sided_99_low: float
    t_stat_vs_alpha: float
    one_sided_p_value: float
    mean_rejections: float
    rejection_probability: float
    evidence_class: str = "stress_test"

    def to_mapping(self) -> dict[str, Any]:
        value = {
            "schema_version": "2.0",
            "model_id": self.model_id,
            "protocol_hash": self.protocol_hash,
            "N": self.N,
            "m": self.m,
            "strata": self.strata,
            "macro_replications": self.macro_replications,
            "datasets": self.datasets,
            "master_seed": self.master_seed,
            "seed_spawn_key": list(self.seed_spawn_key),
            "target_entries": self.target_entries,
            "fdr_hat": self.fdr_hat,
            "mcse": self.mcse,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "one_sided_99_low": self.one_sided_99_low,
            "t_stat_vs_alpha": self.t_stat_vs_alpha,
            "one_sided_p_value": self.one_sided_p_value,
            "mean_rejections": self.mean_rejections,
            "rejection_probability": self.rejection_probability,
            "evidence_class": self.evidence_class,
        }
        return value


def load_dobriban_bh_protocol(path: Path | None = None) -> DobribanBHProtocol:
    """Load and validate the committed offline copy of the publicly documented plan."""
    source = _PLAN_PATH if path is None else Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("protocol plan must be a JSON object")
    return DobribanBHProtocol(payload)


def _validate_run_args(N: int, strata: int, macro_replications: int, seed: int, target_entries: int) -> None:
    for name, value in (("N", N), ("strata", strata), ("macro_replications", macro_replications), ("seed", seed), ("target_entries", target_entries)):
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{name} must be an integer")
        if int(value) < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be positive" if name != "seed" else "seed must be non-negative")
    if macro_replications < 2:
        raise ValueError("macro_replications must be at least 2 for a Monte-Carlo interval")


def _bh_fdp_given_z(protocol: DobribanBHProtocol, N: int, z: np.ndarray, rng: np.random.Generator, target_entries: int) -> tuple[np.ndarray, np.ndarray]:
    model = protocol.plan["model"]
    n0, n1, n2 = (int(multiplier) * N for multiplier in model["group_multipliers"])
    m = n0 + n1 + n2
    batch = max(1, int(target_entries) // m)
    fdp = np.empty(z.size, dtype=float)
    rejections = np.empty(z.size, dtype=np.int32)
    critical = float(model["alpha"]) * np.arange(1, m + 1, dtype=float) / m
    scales = tuple(float(1.0 - float(model[key]) ** 2) ** 0.5 for key in ("rho0", "rho1", "rho2"))
    mus = (0.0, float(model["mu1"]), float(model["mu2"]))
    rhos = tuple(float(model[key]) for key in ("rho0", "rho1", "rho2"))
    for start in range(0, z.size, batch):
        stop = min(start + batch, z.size)
        zz = z[start:stop]
        p0 = 2.0 * ndtr(-np.abs(rhos[0] * zz[:, None] + scales[0] * rng.standard_normal((zz.size, n0))))
        p_all = np.empty((zz.size, m), dtype=float)
        p_all[:, :n0] = p0
        offset = n0
        for rho, scale, mu, count in zip(rhos[1:], scales[1:], mus[1:], (n1, n2), strict=True):
            x = mu + rho * zz[:, None] + scale * rng.standard_normal((zz.size, count))
            p_all[:, offset : offset + count] = 2.0 * ndtr(-np.abs(x))
            offset += count
        p_all.sort(axis=1)
        feasible = p_all <= critical
        has_rejection = feasible.any(axis=1)
        distance_from_right = np.argmax(feasible[:, ::-1], axis=1)
        R = np.where(has_rejection, m - distance_from_right, 0).astype(np.int32)
        threshold = float(model["alpha"]) * R / m
        V = (p0 <= threshold[:, None]).sum(axis=1)
        fdp[start:stop] = np.divide(V, R, out=np.zeros(zz.size), where=R > 0)
        rejections[start:stop] = R
    return fdp, rejections


def run_dobriban_bh_chunk(
    N: int,
    *,
    strata: int = 1000,
    macro_replications: int = 100,
    master_seed: int = 20260711,
    target_entries: int | None = None,
    protocol: DobribanBHProtocol | None = None,
) -> DobribanBHResult:
    """Run one exact finite chunk of the imported protocol.

    The default parameters match the publicly documented ``fdr_N50`` chunk. Smaller
    values are useful for offline tests and smoke validation; they remain
    finite stress evidence, not the external Arb theorem.
    """
    protocol = load_dobriban_bh_protocol() if protocol is None else protocol
    target = int(protocol.plan["target_entries"] if target_entries is None else target_entries)
    _validate_run_args(N, strata, macro_replications, master_seed, target)
    child = np.random.SeedSequence(int(master_seed)).spawn(1)[0]
    rng = np.random.Generator(np.random.PCG64(child))
    macro_fdr = np.empty(int(macro_replications), dtype=float)
    macro_rejections = np.empty(int(macro_replications), dtype=float)
    macro_any = np.empty(int(macro_replications), dtype=float)
    for index in range(int(macro_replications)):
        u = (np.arange(int(strata), dtype=float) + rng.random(int(strata))) / int(strata)
        fdp, rejections = _bh_fdp_given_z(protocol, int(N), ndtri(u), rng, target)
        macro_fdr[index] = float(np.mean(fdp))
        macro_rejections[index] = float(np.mean(rejections))
        macro_any[index] = float(np.mean(rejections > 0))
    estimate = float(np.mean(macro_fdr))
    mcse = float(np.std(macro_fdr, ddof=1) / np.sqrt(int(macro_replications)))
    if not np.isfinite(mcse) or mcse <= 0.0:
        raise ValueError("macro-level Monte-Carlo variance must be positive")
    df = int(macro_replications) - 1
    q975 = float(student_t.ppf(0.975, df))
    q99 = float(student_t.ppf(0.99, df))
    alpha = float(protocol.plan["model"]["alpha"])
    return DobribanBHResult(
        model_id=str(protocol.plan["model"]["model_id"]),
        protocol_hash=protocol.protocol_hash,
        N=int(N), m=100 * int(N), strata=int(strata),
        macro_replications=int(macro_replications), datasets=int(strata) * int(macro_replications),
        master_seed=int(master_seed), seed_spawn_key=tuple(int(part) for part in child.spawn_key),
        target_entries=target,
        fdr_hat=estimate, mcse=mcse, ci95_low=estimate - q975 * mcse,
        ci95_high=estimate + q975 * mcse, one_sided_99_low=estimate - q99 * mcse,
        t_stat_vs_alpha=(estimate - alpha) / mcse,
        one_sided_p_value=float(student_t.sf((estimate - alpha) / mcse, df)),
        mean_rejections=float(np.mean(macro_rejections)),
        rejection_probability=float(np.mean(macro_any)),
    )


__all__ = ["DobribanBHProtocol", "DobribanBHResult", "load_dobriban_bh_protocol", "run_dobriban_bh_chunk"]
