"""PCAR: Phase-Change-Aware Replanning (Innovation 1).

Core idea
---------
Round 4's posterior emits ``beta_t``, the Hellinger boundary signal. During
online rollout, every time ``beta_t`` crosses a threshold ``tau_cp`` we
terminate the remainder of the current chunk and re-predict immediately.
That way the action head gets to see the observation conditioned on "we
just entered a new phase", instead of dragging the previous phase's chunk
into the new phase (which produces mis-aligned actions).

Choosing ``tau_cp`` is the tricky part:

- Static: fix ``pcar_change_threshold`` (say 0.4). Problem: ``beta``'s
  distribution depends on task difficulty; a static threshold is either too
  strict (misses boundaries) or too loose (interrupts too often).
- Adaptive (what this module does): set ``tau_cp`` from a replan budget
  ``epsilon`` (the expected firing rate) using a rolling-window quantile so
  :math:`\\Pr[\\beta_t > \\tau_{cp}] \\approx \\epsilon`. For ``epsilon=0.1``,
  ``tau_cp`` is the 90th percentile of the ``beta`` history.

Theory note (full derivation in the Round 7 summary)
-----------------------------------------------------
Let ``p_*`` be the true boundary density, ``epsilon`` the replan budget, and
``1 - delta`` the boundary recall. Then:

  E[mis-aligned actions per step] <= epsilon + delta

The right-hand side goes to 0 when ``epsilon`` and ``delta`` go to 0; the
budget-adaptive threshold in this module directly controls ``epsilon``.

API layout
----------
- :class:`PCARTrigger` is the main class: ``update_and_check(beta) -> bool``,
  ``get_actual_replan_rate()``, ``reset()``.
- :class:`DualFlowHead` (optional): runs two sibling heads - "current phase"
  (pre) and "next phase" (post) - so that when a replan fires at inference
  we can splice a cross-phase chunk.
- :class:`BayesianChangepointDetector` is a lightweight stub reserved for a
  future Adams-MacKay path; the budget-quantile implementation is enough
  for now.
- :class:`PCARReplanTrigger` is an alias of :class:`PCARTrigger` kept for
  legacy import paths (``phase_centric.pcar_trigger.PCARReplanTrigger``).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
import torch.nn as nn


class PCARTrigger:
    """Budget-respecting adaptive replan trigger based on ``beta_t`` history.

    Parameters
    ----------
    cfg : ``PhaseQFlowConfig``
        Reads ``pcar_trigger_budget_eps`` (expected firing rate
        ``epsilon`` in (0, 1)) and ``pcar_change_threshold`` (manual
        ``tau`` used before the history fills up).
    history_size : int, optional
        Rolling-window length; default 1000 covers ~5 episodes of
        ``beta`` observations.
    warmup_min : int, optional
        Below this many observed steps the trigger falls back to the
        manual threshold; default 50.
    """

    def __init__(
        self,
        cfg: Any,
        history_size: int = 1000,
        warmup_min: int = 50,
    ) -> None:
        """Validate the budget and initialise the rolling history."""
        self.cfg = cfg
        self.budget: float = float(getattr(cfg, "pcar_trigger_budget_eps", 0.1))
        if not (0.0 < self.budget < 1.0):
            raise ValueError(
                f"pcar_trigger_budget_eps must be in (0,1); got {self.budget}"
            )
        self.history_size: int = int(history_size)
        self.warmup_min: int = int(warmup_min)
        self.manual_threshold: float = float(
            getattr(cfg, "pcar_change_threshold", 0.4)
        )
        self.beta_history: deque[float] = deque(maxlen=self.history_size)
        self._triggered_count: int = 0
        self._total_count: int = 0
        self._last_threshold: float = self.manual_threshold

    def update_and_check(self, beta: float) -> bool:
        """Push a fresh ``beta_t`` and return whether it triggers a replan.

        Adaptive rule: once the rolling window holds ``>= warmup_min``
        samples, the threshold is the ``1 - budget`` quantile of the
        history. Below that size the trigger uses ``pcar_change_threshold``
        as a manual warm-start.
        """

        b = float(beta)
        self.beta_history.append(b)
        self._total_count += 1

        if len(self.beta_history) >= self.warmup_min:
            tau = float(
                np.quantile(np.asarray(self.beta_history, dtype=np.float64), 1.0 - self.budget)
            )
        else:
            tau = self.manual_threshold
        self._last_threshold = tau

        triggered = b > tau
        if triggered:
            self._triggered_count += 1
        return triggered

    def get_actual_replan_rate(self) -> float:
        """Fraction of calls that fired; an empty history returns 0."""

        return float(self._triggered_count) / float(max(self._total_count, 1))

    def current_threshold(self) -> float:
        """Most recent ``tau_cp`` used by ``update_and_check`` (for logging)."""

        return float(self._last_threshold)

    def reset(self) -> None:
        """Clear per-episode trigger counters but keep the ``beta`` history.

        The rolling quantile is a global property of ``beta``'s distribution;
        wiping it every episode would re-enter warmup and bias early steps
        toward the manual threshold.
        """

        self._triggered_count = 0
        self._total_count = 0

    def hard_reset(self) -> None:
        """Full reset including the ``beta`` history (mostly for unit tests)."""

        self.beta_history.clear()
        self.reset()

    def describe(self) -> str:
        """Return a short snapshot of the trigger state."""
        return (
            f"PCARTrigger(budget={self.budget}, "
            f"history={len(self.beta_history)}/{self.history_size}, "
            f"tau={self._last_threshold:.3f}, "
            f"rate={self.get_actual_replan_rate():.3f})"
        )


PCARReplanTrigger = PCARTrigger


class BayesianChangepointDetector:
    """Lightweight stub for Adams-MacKay online changepoint detection.

    The production path uses :class:`PCARTrigger`'s budget-quantile strategy,
    which makes no parametric assumption about the ``beta_t`` distribution.
    This stub is here so a future switch to a Bayesian run-length posterior
    can drop in without touching the wiring.

    At the moment the only hook is :func:`theory_utils.run_length_posterior`,
    which is still a placeholder.
    """

    def __init__(self, hazard: float = 0.01) -> None:
        """Validate ``hazard`` and initialise an empty run-length vector."""
        if not (0.0 < hazard < 1.0):
            raise ValueError(f"hazard must be in (0,1); got {hazard}")
        self.hazard = float(hazard)
        self._run_length_log_probs = np.zeros(1)

    def update(self, observation: float) -> float:
        """Naive CUSUM-like update: return raw ``|observation|`` as score."""

        return float(abs(observation))


class DualFlowHead(nn.Module):
    """Sibling pre/post flow heads that cover cross-phase chunks.

    Structure
    ---------
    - ``pre_head``: handles the current phase's full chunk (length
      ``action_chunk_size``).
    - ``post_head``: handles the leading slice of the upcoming phase, length
      ``Ta * pcar_post_head_ratio`` (default Ta/2).
    - Both heads instantiate the same class with independent parameters, and
      share the exact forward signature.

    Training
    --------
    ``pre`` is trained against the original chunk as its target. ``post`` is
    trained against ``actions[:, :post_len]`` as a pseudo-target (in synthetic
    data there is no real next-phase ground truth; a future dataloader will
    align ``chunk_{t+1}`` head onto this position). The post head contributes
    an independent MSE component ``pcar_post_loss``, weighted by
    ``pcar_post_loss_weight`` (defaulting to match the 0.3 of
    ``correction_loss_weight``).

    Inference
    ---------
    - Returns ``pre_action_pred`` of shape ``(B, Ta, Da)`` by default.
    - When ``next_phase_embed`` is supplied, also computes
      ``post_action_pred`` of shape ``(B, post_len, Da)``. When the PCAR
      trigger fires inside ``select_action``, the next chunk is
      ``pre[:, :offset] + post``.
    """

    def __init__(self, cfg: Any, base_flow_head_cls: Type[nn.Module]) -> None:
        """Instantiate two sibling heads with a shared class but separate params."""
        super().__init__()
        self.cfg = cfg
        self.Ta = int(cfg.action_chunk_size)
        ratio = float(getattr(cfg, "pcar_post_head_ratio", 0.5))
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"pcar_post_head_ratio must be in (0, 1]; got {ratio}")
        self.post_len = max(1, int(round(self.Ta * ratio)))
        self.pre_head = base_flow_head_cls(cfg)
        self.post_head = base_flow_head_cls(cfg)

    def forward(
        self,
        fused_obs: torch.Tensor,
        phase_embed: torch.Tensor,
        skill_latent: torch.Tensor,
        next_phase_embed: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Run the pre head; optionally also run the post head.

        ``kwargs`` is forwarded to both sub-heads; this keeps API parity with
        ShortcutFlowActionHead (``actions_gt=`` / ``training=``).
        """

        pre_out = self.pre_head(
            fused_obs=fused_obs,
            phase_embed=phase_embed,
            skill_latent=skill_latent,
            **kwargs,
        )
        result: Dict[str, torch.Tensor] = {
            **pre_out,
            "pre_action_pred": pre_out["action_pred"],
        }

        if next_phase_embed is not None:
            post_out = self.post_head(
                fused_obs=fused_obs,
                phase_embed=next_phase_embed,
                skill_latent=skill_latent,
                **kwargs,
            )
            post_action = post_out["action_pred"]
            if post_action.ndim == 3:
                post_action = post_action[:, : self.post_len].contiguous()
            result["post_action_pred"] = post_action
            if "fm_loss" in post_out:
                result["post_fm_loss"] = post_out["fm_loss"]
            if "sc_loss" in post_out:
                result["post_sc_loss"] = post_out["sc_loss"]
        return result

    def reset(self, batch_size: int = 1) -> None:
        """Forward the rollout-state reset to sub-heads that support it."""

        for h in (self.pre_head, self.post_head):
            if hasattr(h, "reset_switching"):
                h.reset_switching(batch_size=batch_size)


__all__ = [
    "PCARTrigger",
    "PCARReplanTrigger",
    "BayesianChangepointDetector",
    "DualFlowHead",
]
