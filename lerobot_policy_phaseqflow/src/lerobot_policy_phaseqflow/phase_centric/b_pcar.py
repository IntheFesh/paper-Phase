"""Bayesian PCAR — Beta-mixture concordance-driven replanning trigger.

Extends PCARTrigger with a Bayesian Beta-mixture prior that models the
run-length posterior explicitly.  The posterior over run-lengths is maintained
via an Adams-MacKay-style recursion; the replan fires when the probability that
a changepoint occurred in the last step exceeds a threshold.

The concordance signal C_t (from ConcordanceDetector) is the recommended input.
The legacy beta_t Bhattacharyya signal can also be used (set
``pcar_input_signal="beta"`` in config).

Shared rolling-history utilities are imported from ``_pcar_common``.

References
----------
Adams & MacKay, "Bayesian Online Changepoint Detection", arXiv 0710.3742.
Master plan §6.2.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque

import numpy as np

from ._pcar_common import (
    adaptive_threshold,
    make_history,
    validate_budget,
)


class BayesianPCARTrigger:
    """Budget-respecting replanning trigger with Beta-mixture changepoint prior.

    Enhances the quantile-adaptive PCARTrigger with a lightweight Bayesian
    update that accumulates evidence across steps.  A Beta(α, β) conjugate
    prior is placed on the per-step changepoint probability p_cp.  After each
    concordance observation, the posterior is updated:

        α ← α + 1  if triggered (changepoint observed)
        β ← β + 1  otherwise

    The predictive probability of a changepoint at the next step is:

        p̂_cp = α / (α + β)

    An "early fire" occurs when ``p̂_cp >= early_fire_threshold``; this
    allows the detector to be more sensitive during confirmed cliff sequences.

    Parameters
    ----------
    cfg : any object
        Must expose ``pcar_trigger_budget_eps`` and ``pcar_change_threshold``.
    history_size : int
        Rolling-window length for the DKW quantile estimator.
    warmup_min : int
        Minimum history before the adaptive threshold activates.
    alpha_prior, beta_prior : float
        Beta prior hyper-parameters (shape of the changepoint probability).
    early_fire_threshold : float
        When ``p̂_cp >= early_fire_threshold``, fire regardless of the quantile
        threshold (allows rapid response during sustained cliff sequences).
        Set to 1.0 to disable.
    """

    def __init__(
        self,
        cfg: Any,
        history_size: int = 1000,
        warmup_min: int = 50,
        alpha_prior: float = 1.0,
        beta_prior: float = 5.0,
        early_fire_threshold: float = 0.8,
    ) -> None:
        """Validate the budget and initialise the rolling history and Beta prior.

        Prior copies ``_alpha0``/``_beta0`` are stored separately so
        :meth:`hard_reset` can restore the original prior without re-instantiating.
        """
        self.input_signal: str = str(getattr(cfg, "pcar_input_signal", "concordance"))
        self.budget: float = float(getattr(cfg, "pcar_trigger_budget_eps", 0.1))
        validate_budget(self.budget)

        self.history_size = int(history_size)
        self.warmup_min = int(warmup_min)
        self.manual_threshold: float = float(getattr(cfg, "pcar_change_threshold", 0.4))
        self.early_fire_threshold: float = float(early_fire_threshold)

        self.signal_history: Deque[float] = make_history(self.history_size)
        self._triggered_count: int = 0
        self._total_count: int = 0
        self._last_threshold: float = self.manual_threshold

        # Beta-mixture posterior state
        self._alpha: float = float(alpha_prior)
        self._beta: float = float(beta_prior)
        self._alpha0: float = float(alpha_prior)
        self._beta0: float = float(beta_prior)

    @property
    def changepoint_probability(self) -> float:
        """Posterior mean of the changepoint probability p_cp ~ Beta(α, β)."""
        return self._alpha / (self._alpha + self._beta)

    def update_and_check(self, signal: float) -> bool:
        """Push a fresh signal value and return whether it triggers a replan.

        Two independent trigger paths:
        1. Quantile-adaptive threshold (same as PCARTrigger).
        2. Beta posterior early-fire when p̂_cp >= early_fire_threshold.

        Parameters
        ----------
        signal : float
            Concordance C_t or legacy beta_t.

        Returns
        -------
        bool
            True when either trigger path fires.
        """
        s = float(signal)
        self.signal_history.append(s)
        self._total_count += 1

        tau = adaptive_threshold(
            self.signal_history, self.budget, self.warmup_min, self.manual_threshold
        )
        self._last_threshold = tau
        quantile_fire = s > tau

        early_fire = self.changepoint_probability >= self.early_fire_threshold

        triggered = quantile_fire or early_fire

        # Update Beta posterior
        if triggered:
            self._alpha += 1.0
            self._triggered_count += 1
        else:
            self._beta += 1.0

        return triggered

    def get_actual_replan_rate(self) -> float:
        """Fraction of calls that fired."""
        return float(self._triggered_count) / float(max(self._total_count, 1))

    def current_threshold(self) -> float:
        """Most recent quantile threshold (for logging)."""
        return float(self._last_threshold)

    def reset(self) -> None:
        """Reset per-episode counters; keep signal history and prior."""
        self._triggered_count = 0
        self._total_count = 0

    def hard_reset(self) -> None:
        """Full reset including history and Beta posterior (for unit tests)."""
        self.signal_history.clear()
        self._alpha = self._alpha0
        self._beta = self._beta0
        self.reset()

    def describe(self) -> str:
        """Return a short snapshot of the trigger state."""
        return (
            f"BayesianPCARTrigger(budget={self.budget}, "
            f"history={len(self.signal_history)}/{self.history_size}, "
            f"tau={self._last_threshold:.3f}, "
            f"p_cp={self.changepoint_probability:.3f}, "
            f"rate={self.get_actual_replan_rate():.3f})"
        )


__all__ = ["BayesianPCARTrigger"]
