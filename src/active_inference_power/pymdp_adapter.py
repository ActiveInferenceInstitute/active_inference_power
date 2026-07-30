"""Small dependency boundary for the pinned ``inferactively-pymdp`` agent.

The suite exposes its own scenario and trace contracts; this module is the only
place where dependency-specific construction and numerical configuration are
centralized. The installed pymdp release emits a harmless Equinox warning while
constructing static policy metadata, so the boundary suppresses that known
dependency diagnostic without hiding errors from inference itself.
"""

from __future__ import annotations

import warnings
from typing import Any

import jax
from pymdp.agent import Agent

jax.config.update("jax_enable_x64", True)


def make_agent(**kwargs: Any) -> Agent:
    """Construct the pinned real agent with deterministic 64-bit arithmetic."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A JAX array is being set as static!",
            category=UserWarning,
        )
        return Agent(**kwargs)


__all__ = ["make_agent"]
