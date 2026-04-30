"""Shared utilities for PCAR and B-PCAR trigger implementations.

Both :class:`~phase_centric.pcar_trigger.PCARTrigger` (budget-quantile) and
:class:`~phase_centric.b_pcar.BayesianPCARTrigger` (Beta-mixture) use the
rolling history buffer and DKW-guaranteed quantile computation defined here.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque

import numpy as np


def make_history(maxlen: int) -> Deque[float]:
    """Create a capped rolling deque for the signal history shared by both trigger classes.

    Using a shared factory (rather than inline ``deque(maxlen=…)``) keeps the
    maxlen type-cast and any future logging hooks in one place.
    """
    return deque(maxlen=int(maxlen))


def rolling_quantile(
    history: Deque[float],
    q: float,
) -> float:
    """Return the q-th quantile of ``history`` via numpy — shared by both trigger classes to guarantee identical quantile semantics.

    Parameters
    ----------
    history : deque
        Rolling signal history.
    q : float
        Quantile in [0, 1].

    Returns
    -------
    float
        The q-th quantile value.
    """
    arr = np.asarray(list(history), dtype=np.float64)
    return float(np.quantile(arr, q))


def validate_budget(budget: float) -> None:
    """Raise ValueError if budget is outside (0, 1) — catches misconfigured firing-rate targets early."""
    if not (0.0 < budget < 1.0):
        raise ValueError(
            f"pcar_trigger_budget_eps must be in (0, 1); got {budget}"
        )


def adaptive_threshold(
    history: Deque[float],
    budget: float,
    warmup_min: int,
    manual_threshold: float,
) -> float:
    """Return the adaptive threshold for the current history.

    Falls back to ``manual_threshold`` when fewer than ``warmup_min`` samples
    have been collected.
    """
    if len(history) >= warmup_min:
        return rolling_quantile(history, 1.0 - budget)
    return manual_threshold


__all__ = [
    "make_history",
    "rolling_quantile",
    "validate_budget",
    "adaptive_threshold",
]
