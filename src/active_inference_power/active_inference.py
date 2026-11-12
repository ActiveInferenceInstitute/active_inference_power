"""Active inference as a sequential evidence-calibration experiment.

The pinned ``inferactively-pymdp`` agent is used here to generate a
*sequential belief update* for a fixed-horizon calibration experiment. There is
a single hidden-state factor "truth" taking two
values, ``H0`` (null) and ``H1`` (signal). At every step the agent receives one
noisy binary observation — ``low`` or ``high`` — whose reliability is governed
by :class:`AITestConfig`. Because the transition matrix ``B`` is the identity
under a single (dummy) action, the posterior belief at step ``t`` becomes the
prior at step ``t + 1``: the agent performs *exact sequential Bayesian evidence
accumulation*, realized as active inference.

The agent "rejects H0" iff its final posterior belief in ``H1`` reaches the
configured ``threshold``.

The implementation uses the installed ``pymdp.agent.Agent`` interface from
``inferactively-pymdp==1.0.5``. The live pin is ``inferactively-pymdp==1.0.5``
(``pyproject.toml`` and ``uv.lock``); the frozen v1.0.0 manuscript metadata
(``manuscript/config.yaml``) still records 1.0.3 as dated release evidence. The
project's public API is independent of that dependency namespace.

Closed form (used as the correctness oracle in the tests). With prior odds
``o0 = prior_signal / (1 - prior_signal)`` and per-observation likelihood ratio
``LR = reliability / (1 - reliability)``, after observing ``n_high`` "high" and
``n_low`` "low" observations the posterior odds on ``H1`` are

    o = o0 * LR ** n_high * LR ** (-n_low)

and the posterior signal belief is ``o / (1 + o)``. The real agent reproduces
this to floating-point precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.stats import binom

from .corrections import benjamini_hochberg
from .error_rates import classify
from .pymdp_adapter import make_agent
from .uncertainty import summarize_bounded

__all__ = [
    "AITestConfig",
    "AITestResult",
    "build_generative_model",
    "run_ai_test",
    "generate_evidence_stream",
    "ai_test_decision",
    "evidence_pvalue",
    "evaluate_ai_regime",
]

# Observation indices in the generative model.
OBS_LOW: int = 0
OBS_HIGH: int = 1


def _sample_stderr(values: Sequence[float | bool]) -> float:
    """Return the Monte-Carlo standard error for a finite sample."""
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1) / np.sqrt(len(values)))


@dataclass
class AITestConfig:
    """Configuration of the active-inference sequential test.

    Attributes:
        reliability: ``P(high | H1) = P(low | H0)`` — the probability that a
            single observation points at the true state. Must lie strictly in
            ``(0, 1)``; ``0.5`` is uninformative, values ``> 0.5`` are
            informative. Values ``< 0.5`` model an anti-correlated channel.
        threshold: Posterior signal belief at or above which ``H0`` is rejected.
            Must lie in ``[0, 1]``.
        horizon: Number of observations drawn per evidence stream. Must be a
            non-negative integer.
        prior_signal: Prior probability of ``H1`` (signal present). Must lie in
            ``[0, 1]``.
    """

    reliability: float = 0.8
    threshold: float = 0.9
    horizon: int = 20
    prior_signal: float = 0.5

    def __post_init__(self) -> None:
        """Validate the configuration ranges.

        Raises:
            ValueError: If ``reliability`` is not strictly in ``(0, 1)``, if
                ``threshold`` or ``prior_signal`` is outside ``[0, 1]``, or if
                ``horizon`` is negative.
        """
        if not np.isfinite(self.reliability) or not (0.0 < self.reliability < 1.0):
            raise ValueError(
                f"reliability must be strictly in (0, 1); got {self.reliability!r}"
            )
        if not np.isfinite(self.threshold) or not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1]; got {self.threshold!r}")
        if not np.isfinite(self.prior_signal) or not (0.0 <= self.prior_signal <= 1.0):
            raise ValueError(
                f"prior_signal must be in [0, 1]; got {self.prior_signal!r}"
            )
        if (
            isinstance(self.horizon, bool)
            or int(self.horizon) != self.horizon
            or self.horizon < 0
        ):
            raise ValueError(f"horizon must be non-negative; got {self.horizon!r}")


@dataclass
class AITestResult:
    """Result of one active-inference sequential test.

    Attributes:
        belief_trajectory: Posterior belief in ``H1`` (signal) at each step,
            shape ``(n_steps + 1,)``. Element ``0`` is the prior; element ``t``
            is the posterior after the ``t``-th observation. Every value lies in
            ``[0, 1]``.
        final_belief_signal: The last entry of ``belief_trajectory`` — the
            posterior belief in ``H1`` after all observations.
        decision: ``True`` iff ``final_belief_signal >= config.threshold``
            (i.e. "reject H0").
        n_steps: Number of observations consumed (``len(obs_indices)``).
    """

    belief_trajectory: np.ndarray
    final_belief_signal: float
    decision: bool
    n_steps: int


def build_generative_model(
    config: AITestConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the ``(A, B, C, D)`` generative model for the test.

    The single observation modality has two levels ``{low, high}`` and the
    single hidden-state factor has two levels ``{H0, H1}``.

    Args:
        config: Test configuration supplying ``reliability`` and
            ``prior_signal``.

    Returns:
        A tuple ``(A, B, C, D)`` of ``pymdp`` object arrays:

        - ``A[0]`` — ``2x2`` likelihood, rows are observations ``{low, high}``,
          columns are states ``{H0, H1}``.
        - ``B[0]`` — ``2x2x1`` identity transition (truth is static; one action).
        - ``C[0]`` — flat preferences (this is inference, not control).
        - ``D[0]`` — prior ``[1 - prior_signal, prior_signal]`` over ``{H0, H1}``.
    """
    r = config.reliability
    prior = config.prior_signal

    A = np.empty(1, dtype=object)
    # rows = obs {low, high}, cols = state {H0, H1}
    #   P(low  | H0) = r        P(low  | H1) = 1 - r
    #   P(high | H0) = 1 - r    P(high | H1) = r
    A[0] = np.array([[r, 1.0 - r], [1.0 - r, r]], dtype=float)

    B = np.empty(1, dtype=object)
    B[0] = np.eye(2)[:, :, None]  # identity, single action: truth is static

    C = np.empty(1, dtype=object)
    C[0] = np.zeros(2)  # flat preferences (inference, not control)

    D = np.empty(1, dtype=object)
    D[0] = np.array([1.0 - prior, prior], dtype=float)  # prior over {H0, H1}

    return A, B, C, D


@lru_cache(maxsize=16)
def _cached_agent(reliability: float):
    """Reuse the real agent for repeated streams in one release run.

    Agent construction triggers JAX/Equinox setup, so constructing it inside
    every Monte-Carlo replication makes the scientific run needlessly
    dominated by dependency initialization.  The empirical prior is supplied
    to ``infer_states`` for each stream; caching therefore changes no
    inferential input while preserving the real dependency path.
    """
    model = AITestConfig(reliability=float(reliability), prior_signal=0.5)
    A, B, C, D = build_generative_model(model)
    return make_agent(
        A=[A[0]],
        B=[B[0]],
        C=[C[0]],
        D=[D[0]],
        use_utility=False,
        use_states_info_gain=False,
    )


@lru_cache(maxsize=32)
def _cached_batch_agent(reliability: float, batch_size: int):
    """Construct one real agent for a batch of independent evidence streams."""
    model = AITestConfig(reliability=float(reliability), prior_signal=0.5)
    A, B, C, D = build_generative_model(model)
    return make_agent(
        A=[A[0]],
        B=[B[0]],
        C=[C[0]],
        D=[D[0]],
        use_utility=False,
        use_states_info_gain=False,
        batch_size=int(batch_size),
    )


def _run_ai_batch(
    observations: np.ndarray,
    prior_signal: float,
    config: AITestConfig,
) -> np.ndarray:
    """Return final posteriors for independent streams in one real-agent batch."""
    streams = np.asarray(observations, dtype=int)
    if streams.ndim != 2:
        raise ValueError("observations must be a two-dimensional stream matrix")
    batch_size, horizon = streams.shape
    if batch_size < 1:
        raise ValueError("observations must contain at least one stream")
    if np.any((streams < OBS_LOW) | (streams > OBS_HIGH)):
        raise ValueError("observations must contain only binary indices")
    prior = np.tile(
        np.asarray([1.0 - float(prior_signal), float(prior_signal)], dtype=float),
        (batch_size, 1),
    )
    if horizon == 0:
        return prior
    agent = _cached_batch_agent(float(config.reliability), int(batch_size))
    for time in range(horizon):
        qs = agent.infer_states(
            [streams[:, time]],
            empirical_prior=[prior],
        )
        prior = np.asarray(qs[0], dtype=float)[:, -1, :]
    return prior


def run_ai_test(obs_indices: np.ndarray, config: AITestConfig) -> AITestResult:
    """Run the sequential test on a fixed observation stream.

    Instantiates the pinned ``pymdp.agent.Agent``, feeds it each binary
    observation in turn, and records the posterior belief in ``H1`` after every
    step. Because ``B`` is identity with a single action, each posterior rolls
    forward to become the next step's prior — exact sequential Bayesian updating.

    Args:
        obs_indices: Sequence of observation indices, each ``0`` (low) or ``1``
            (high). May be empty (no evidence: the belief stays at the prior).
        config: Test configuration.

    Returns:
        An :class:`AITestResult` with the full belief trajectory, final signal
        belief, decision, and step count.

    Raises:
        ValueError: If any observation index is not ``0`` or ``1``.
    """
    agent = _cached_agent(float(config.reliability))

    obs = np.asarray(obs_indices).ravel()
    prior = np.asarray([1.0 - config.prior_signal, config.prior_signal], dtype=float)[None, :]
    trajectory: list[float] = [float(prior[0, OBS_HIGH])]  # prior belief in H1

    for raw in obs:
        try:
            raw_float = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"observation indices must be numeric 0 or 1; got {raw!r}"
            ) from exc
        if not np.isfinite(raw_float) or raw_float != int(raw_float):
            raise ValueError(f"observation indices must be integer 0 or 1; got {raw!r}")
        obs_index = int(raw_float)
        if obs_index not in (OBS_LOW, OBS_HIGH):
            raise ValueError(
                f"observation indices must be 0 (low) or 1 (high); got {obs_index!r}"
            )
        qs = agent.infer_states(
            [np.asarray([obs_index], dtype=int)],
            empirical_prior=[prior],
        )
        posterior = np.asarray(qs[0], dtype=float)[0, -1, :]
        prior = posterior[None, :]
        trajectory.append(float(posterior[OBS_HIGH]))  # belief in H1

    belief_trajectory = np.asarray(trajectory, dtype=float)
    final_belief_signal = float(belief_trajectory[-1])
    decision = bool(final_belief_signal >= config.threshold)

    return AITestResult(
        belief_trajectory=belief_trajectory,
        final_belief_signal=final_belief_signal,
        decision=decision,
        n_steps=int(obs.size),
    )


def generate_evidence_stream(
    is_signal: bool,
    config: AITestConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a ``horizon``-length binary observation stream from the true world.

    Args:
        is_signal: Ground truth. If ``True`` the world is in ``H1`` and emits
            ``high`` with probability ``reliability``; if ``False`` the world is
            in ``H0`` and emits ``high`` with probability ``1 - reliability``.
        config: Test configuration supplying ``reliability`` and ``horizon``.
        rng: Seeded NumPy generator for reproducibility.

    Returns:
        Integer array of shape ``(horizon,)`` with values in ``{0, 1}``
        (``0`` == low, ``1`` == high).
    """
    if is_signal:
        p_high = config.reliability  # P(high | H1)
    else:
        p_high = 1.0 - config.reliability  # P(high | H0)

    draws = rng.random(config.horizon) < p_high
    return draws.astype(int)


def ai_test_decision(
    is_signal: bool,
    config: AITestConfig,
    rng: np.random.Generator,
) -> bool:
    """Run one end-to-end test: draw a stream from the world and decide.

    Args:
        is_signal: Ground truth used to generate the observation stream.
        config: Test configuration.
        rng: Seeded NumPy generator.

    Returns:
        ``True`` iff the agent rejects ``H0`` (final signal belief reaches the
        threshold) on a stream drawn from the true world.
    """
    stream = generate_evidence_stream(is_signal, config, rng)
    result = run_ai_test(stream, config)
    return result.decision


def evidence_pvalue(n_high: int, horizon: int, reliability: float) -> float:
    """Return the calibrated one-sided null p-value for an evidence stream.

    Under ``H0`` the number of ``high`` observations is
    ``Binomial(horizon, 1 - reliability)``. For an informative channel with
    reliability above ``0.5``, the posterior signal evidence is an upper-tail
    event and the survival function is used. For an anti-correlated channel
    below ``0.5``, signal evidence is a lower-tail event and the CDF is used.
    This keeps the documented anti-correlated configuration calibrated while
    preserving the explicit boundary that a posterior belief is not itself a
    multiplicity-adjusted p-value.

    Args:
        n_high: Observed number of high observations.
        horizon: Total number of observations in the stream.
        reliability: Channel reliability in ``(0, 1)``.

    Returns:
        The direction-calibrated Binomial p-value in ``[0, 1]``.

    Raises:
        ValueError: If counts or reliability are outside their valid ranges.
    """
    if isinstance(n_high, bool) or int(n_high) != n_high:
        raise ValueError(f"n_high must be an integer, got {n_high!r}")
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 0:
        raise ValueError(f"horizon must be a non-negative integer, got {horizon!r}")
    if not 0 <= n_high <= horizon:
        raise ValueError(
            f"n_high must lie in [0, horizon], got {n_high!r} for horizon={horizon!r}"
        )
    if not np.isfinite(reliability) or not 0.0 < reliability < 1.0:
        raise ValueError(f"reliability must lie in (0, 1), got {reliability!r}")
    null_high_rate = 1.0 - reliability
    if reliability < 0.5:
        return float(binom.cdf(n_high, horizon, null_high_rate))
    return float(binom.sf(n_high - 1, horizon, null_high_rate))


def evaluate_ai_regime(
    m: int,
    pi0: float,
    config: AITestConfig,
    n_reps: int,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, object]:
    """Compare posterior-threshold decisions with BH-calibrated evidence decisions.

    This is the reusable domain implementation of the experiment used by the
    manuscript. Both policies see identical generated evidence streams within
    each replication; only the final decision rule differs.
    """
    if isinstance(m, bool) or int(m) != m or m < 1:
        raise ValueError(f"m must be a positive integer, got {m!r}")
    if not np.isfinite(pi0) or not 0.0 <= pi0 <= 1.0:
        raise ValueError(f"pi0 must lie in [0, 1], got {pi0!r}")
    try:
        n_reps_int = int(n_reps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"n_reps must be an integer >= 1, got {n_reps!r}") from exc
    if isinstance(n_reps, bool) or n_reps_int != n_reps or n_reps_int < 1:
        raise ValueError(f"n_reps must be an integer >= 1, got {n_reps!r}")
    n_reps = n_reps_int
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")

    rng = np.random.default_rng(seed)
    m1 = m - int(round(m * pi0))
    posterior_fdp: list[float] = []
    posterior_power: list[float] = []
    posterior_fwer: list[bool] = []
    bh_fdp: list[float] = []
    bh_power: list[float] = []
    bh_fwer: list[bool] = []

    for _ in range(n_reps):
        is_alt = np.zeros(m, dtype=bool)
        is_alt[rng.choice(m, size=m1, replace=False)] = True
        streams = np.empty((m, config.horizon), dtype=int)
        for i, is_signal in enumerate(is_alt):
            streams[i] = generate_evidence_stream(bool(is_signal), config, rng)
        final_posteriors = _run_ai_batch(streams, config.prior_signal, config)
        posterior_threshold = final_posteriors[:, OBS_HIGH] >= config.threshold
        pvalues = np.empty(m, dtype=float)
        for i in range(m):
            pvalues[i] = evidence_pvalue(
                int(streams[i].sum()), config.horizon, config.reliability
            )
        corrected = benjamini_hochberg(pvalues, alpha).rejected
        posterior_summary = classify(posterior_threshold, is_alt)
        bh_summary = classify(corrected, is_alt)
        posterior_fdp.append(posterior_summary.fdp)
        posterior_power.append(posterior_summary.power)
        posterior_fwer.append(posterior_summary.fwer_event)
        bh_fdp.append(bh_summary.fdp)
        bh_power.append(bh_summary.power)
        bh_fwer.append(bh_summary.fwer_event)

    return {
        "config": {
            "m": int(m),
            "pi0": float(pi0),
            "reliability": float(config.reliability),
            "threshold": float(config.threshold),
            "horizon": int(config.horizon),
            "prior_signal": float(config.prior_signal),
        },
        "posterior_threshold_fdr": float(np.mean(posterior_fdp)),
        "posterior_threshold_power": float(np.mean(posterior_power)),
        "posterior_threshold_fwer": float(np.mean(posterior_fwer)),
        "posterior_threshold_fdr_stderr": _sample_stderr(posterior_fdp),
        "posterior_threshold_power_stderr": _sample_stderr(posterior_power),
        "posterior_threshold_fwer_stderr": _sample_stderr(posterior_fwer),
        "bh_on_evidence_fdr": float(np.mean(bh_fdp)),
        "bh_on_evidence_power": float(np.mean(bh_power)),
        "bh_on_evidence_fwer": float(np.mean(bh_fwer)),
        "bh_on_evidence_fdr_stderr": _sample_stderr(bh_fdp),
        "bh_on_evidence_power_stderr": _sample_stderr(bh_power),
        "bh_on_evidence_fwer_stderr": _sample_stderr(bh_fwer),
        "mc_precision": {
            "posterior_threshold_fdr": summarize_bounded(posterior_fdp),
            "posterior_threshold_power": summarize_bounded(posterior_power),
            "posterior_threshold_fwer": summarize_bounded(posterior_fwer),
            "bh_on_evidence_fdr": summarize_bounded(bh_fdp),
            "bh_on_evidence_power": summarize_bounded(bh_power),
            "bh_on_evidence_fwer": summarize_bounded(bh_fwer),
        },
        "n_reps": int(n_reps),
        "seed": int(seed),
    }
