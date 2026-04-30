"""Phase posterior smoothing and boundary signal ``beta_t`` (Innovation 4).

Motivation
----------
Round 3 made ``HierarchicalPlanner.phase_logits`` identifiable across seeds.
Round 5 PACE-A sample reweighting, Round 6 PACE-B MoE smooth switching, and
Round 7 PCAR triggering all need two derived signals:

1. Smoothed posterior ``p_hat_t = alpha * softmax(logits_t) + (1-alpha) * p_hat_{t-1}``
2. Boundary posterior ``beta_t = P(z_t != z_{t-1} | o_{1:t})``

The naive choice ``(argmax(p_hat_t) != argmax(p_hat_{t-1})).float()`` has two
fatal issues: (a) not differentiable, which blocks end-to-end gradients, and
(b) high variance, since argmax flips are only loosely correlated with real
phase transitions. This round defines ``beta_t`` via Bhattacharyya distance:

.. math::

    \\beta_t = 1 - BC(\\hat{p}_t, \\hat{p}_{t-1}) = 1 - \\sum_k \\sqrt{\\hat{p}_t[k] \\cdot \\hat{p}_{t-1}[k]}

Properties (proved in round-4-summary.md):

- ``BC(p, q) = sum sqrt(p_k q_k) in [0, 1]``: by Cauchy-Schwarz,
  ``sum sqrt(p_k q_k) <= sqrt(sum p_k * sum q_k) = 1``; the bound is tight
  when the two distributions coincide.
- ``beta_t = 1 - BC in [0, 1]``, and ``beta_t = 0`` iff
  ``p_hat_t == p_hat_{t-1}``.
- Bhattacharyya distance ``d_B = -log BC`` relates to total variation via
  ``d_TV <= sqrt(1 - BC^2)``; so ``beta_t`` is a smooth, monotone
  replacement for TV.
- Differentiable: ``sqrt`` is differentiable under ``clamp_min(eps)``, so
  this slot into end-to-end loss functions cleanly.

Activation
----------
Only wired in when ``PhaseQFlowConfig.use_phase_boundary_posterior = True``.
Otherwise ``PhaseQFlowPolicy.__init__`` sets ``self.phase_posterior`` to
None, and forward skips this branch; behaviour stays identical to Round 1-3.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_phaseqflow import PhaseQFlowConfig


def _infer_planner_k(cfg: PhaseQFlowConfig) -> int:
    """Infer the planner's ``K`` (matches ``ChunkInfoNCEHead``'s rule).

    ``preds["phase_logits"]`` comes out of HierarchicalPlanner with its last
    dim set by the discrete encoder: ``K = prod(fsq_levels)`` under FSQ,
    ``K = num_skills`` under Gumbel. ``cfg.num_phases`` is a semantic field
    (the 4 human-task phases) and does *not* match the planner logits dim,
    so it must not be used here.
    """

    if bool(getattr(cfg, "use_fsq", False)):
        levels = list(getattr(cfg, "fsq_levels", []))
        if not levels:
            raise ValueError("use_fsq=True requires non-empty fsq_levels.")
        k = 1
        for lv in levels:
            k *= int(lv)
        return k
    return int(cfg.num_skills)


class PhasePosteriorEstimator(nn.Module):
    """Produces the smoothed posterior ``p_hat_t`` and boundary signal ``beta_t``.

    Training mode
    -------------
    Call :meth:`forward_sequence` with ``(B, T, K)``; it returns the full
    trajectory's ``p_hat`` and ``beta``. Useful for teacher-forcing style
    training or offline rollout analysis.

    Inference / rollout mode
    ------------------------
    Call :meth:`step` with ``(B, K)``; internal ``_running_p`` tracks the
    rolling state. Call :meth:`reset` at the start of every episode.

    Parameters
    ----------
    cfg : PhaseQFlowConfig
        Fields used:
        - ``phase_posterior_smooth_alpha`` (alpha, default 0.3)
        - ``use_fsq`` / ``fsq_levels`` / ``num_skills`` for inferring K
    """

    def __init__(self, cfg: PhaseQFlowConfig) -> None:
        """Validate alpha, derive K, and initialise the running posterior."""
        super().__init__()
        self.cfg = cfg
        self.alpha = float(cfg.phase_posterior_smooth_alpha)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(
                f"phase_posterior_smooth_alpha must be in (0, 1]; got {self.alpha}"
            )
        self.K = _infer_planner_k(cfg)
        self.register_buffer(
            "_running_p",
            torch.full((1, self.K), 1.0 / self.K),
            persistent=False,
        )

    @staticmethod
    def _bhattacharyya_beta(
        p_cur: torch.Tensor, p_prev: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Return ``beta = 1 - sum_k sqrt(p_hat_t[k] * p_hat_{t-1}[k])`` in [0, 1]."""

        bc = (p_cur.clamp_min(eps) * p_prev.clamp_min(eps)).sqrt().sum(dim=-1)
        return (1.0 - bc).clamp(0.0, 1.0)

    def forward_sequence(
        self, phase_logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Process an entire time series at once (training / offline).

        Parameters
        ----------
        phase_logits : torch.Tensor
            ``(B, T, K)``.

        Returns
        -------
        dict
            ``p_hat``: ``(B, T, K)`` smoothed posterior;
            ``p_hat[:, 0] = softmax(logits[:, 0])``.
            ``beta``: ``(B, T)`` with ``beta[:, 0] = 0`` (no predecessor).
        """

        if phase_logits.ndim != 3 or phase_logits.shape[-1] != self.K:
            raise ValueError(
                f"forward_sequence expects (B, T, {self.K}); got {tuple(phase_logits.shape)}"
            )

        probs = F.softmax(phase_logits, dim=-1)
        B, T, K = probs.shape
        p_list = [probs[:, 0]]
        for t in range(1, T):
            p_list.append(self.alpha * probs[:, t] + (1.0 - self.alpha) * p_list[-1])
        p_hat = torch.stack(p_list, dim=1)

        beta = torch.zeros(B, T, device=phase_logits.device, dtype=p_hat.dtype)
        if T >= 2:
            beta_body = self._bhattacharyya_beta(p_hat[:, 1:], p_hat[:, :-1])
            beta[:, 1:] = beta_body

        return {"p_hat": p_hat, "beta": beta}

    def reset(self, batch_size: int = 1, device: torch.device | None = None) -> None:
        """Reset the running state to a uniform distribution at episode start.

        ``batch_size`` can exceed 1 for parallel rollouts across envs.
        ``device=None`` keeps the buffer on whichever device it already lives.
        """

        if device is None:
            device = self._running_p.device
        self._running_p = torch.full(
            (batch_size, self.K), 1.0 / self.K,
            device=device, dtype=self._running_p.dtype,
        )

    def step(self, phase_logits_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single-step inference: update the running state and return ``p_hat``, ``beta``.

        Parameters
        ----------
        phase_logits_t : torch.Tensor
            ``(B, K)``.

        Returns
        -------
        dict
            ``p_hat``: ``(B, K)``; ``beta``: ``(B,)``.

        Notes
        -----
        - When ``B`` does not match the running state's batch dim (common
          when switching between single-env and multi-env rollouts),
          :meth:`reset` is called automatically with the new ``B``.
        - The running state is written back via ``.detach()``; cross-step
          gradients do not flow, which avoids unbounded TBPTT.
        """

        if phase_logits_t.ndim != 2 or phase_logits_t.shape[-1] != self.K:
            raise ValueError(
                f"step expects (B, {self.K}); got {tuple(phase_logits_t.shape)}"
            )

        B = phase_logits_t.shape[0]
        if (
            self._running_p.shape[0] != B
            or self._running_p.device != phase_logits_t.device
        ):
            self.reset(batch_size=B, device=phase_logits_t.device)

        probs = F.softmax(phase_logits_t, dim=-1)
        prev_p = self._running_p
        new_p = self.alpha * probs + (1.0 - self.alpha) * prev_p
        beta = self._bhattacharyya_beta(new_p, prev_p)

        self._running_p = new_p.detach()
        return {"p_hat": new_p, "beta": beta}


def boundary_prob_from_logits(
    phase_logits: torch.Tensor, alpha: float = 0.3, eps: float = 1e-8
) -> Dict[str, torch.Tensor]:
    """Functional interface: stateless ``p_hat`` and ``beta`` from a logits segment.

    For use by ``scripts/verify_phase_posterior.py`` and unit tests that do
    not want to hold a Module. Semantics match
    :meth:`PhasePosteriorEstimator.forward_sequence` but there is no check
    on ``K`` (any last-dim is accepted) and no ``nn.Module`` is instantiated.

    Parameters
    ----------
    phase_logits : torch.Tensor
        ``(B, T, K)`` or ``(T, K)``; the latter is treated as ``B = 1``.
    alpha : float
        EMA smoothing coefficient.
    eps : float
        Numerical floor to avoid ``sqrt(0)`` NaN gradients.
    """

    if phase_logits.ndim == 2:
        phase_logits = phase_logits.unsqueeze(0)
        squeeze_back = True
    elif phase_logits.ndim == 3:
        squeeze_back = False
    else:
        raise ValueError(
            f"boundary_prob_from_logits expects (B,T,K) or (T,K); got {tuple(phase_logits.shape)}"
        )

    probs = F.softmax(phase_logits, dim=-1)
    B, T, K = probs.shape
    p_list = [probs[:, 0]]
    for t in range(1, T):
        p_list.append(alpha * probs[:, t] + (1.0 - alpha) * p_list[-1])
    p_hat = torch.stack(p_list, dim=1)

    beta = torch.zeros(B, T, device=phase_logits.device, dtype=p_hat.dtype)
    if T >= 2:
        bc = (
            p_hat[:, 1:].clamp_min(eps) * p_hat[:, :-1].clamp_min(eps)
        ).sqrt().sum(dim=-1)
        beta[:, 1:] = (1.0 - bc).clamp(0.0, 1.0)

    if squeeze_back:
        return {"p_hat": p_hat.squeeze(0), "beta": beta.squeeze(0)}
    return {"p_hat": p_hat, "beta": beta}
