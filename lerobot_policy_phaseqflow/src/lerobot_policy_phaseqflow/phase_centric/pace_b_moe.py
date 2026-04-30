"""PACE-B: phase-gated mixture-of-experts with smooth switching.

Core idea
---------
Round 5's PACE-A uses ``beta_t`` to *reweight* the FM loss, injecting the
prior that "boundary steps matter more" into the gradient. PACE-B swaps
mechanisms: replace the flow action head with a small expert per phase so
each phase lives in its own function space, then drive expert switching with
a ``beta_t``-aware time-smoothed gate ``g_t`` that controls how quickly the
transition happens:

.. math::

   g_t & = \\alpha_t \\cdot \\hat p_t + (1 - \\alpha_t) \\cdot g_{t-1} \\\\
   \\alpha_t & = \\sigma(\\kappa \\cdot \\beta_t - \\mu)

where :math:`\\hat p_t` is the ``phase_p_hat`` posterior from Round 4,
:math:`\\beta_t` is the Hellinger boundary signal, and
:math:`\\kappa,\\mu` are set through ``moe_switch_kappa`` / ``moe_switch_mu``.

Key properties
--------------
- At inference, ``g_{t-1}`` is cached in ``_running_gate``; while we are
  inside a phase ``beta`` is near 0, so ``alpha`` goes to 0, and ``g`` stays
  close to the previous step. That prevents the gate from jittering within
  a phase.
- At a boundary, ``beta`` spikes, ``alpha`` goes to 1, and ``g`` snaps to
  :math:`\\hat p_t`, switching experts quickly.
- In training (``self.training=True``) there is no temporal ordering among
  batch elements, so ``g_t`` degenerates to :math:`\\hat p_t`.
- If :math:`\\hat p_t` or :math:`\\beta_t` are missing (``use_phase_boundary_posterior
  =False``), PhaseMoE falls back to uniform gating (each expert weighted
  1/K), equivalent to a plain MoE ensemble.

Math intuition
--------------
``alpha_t = sigmoid(kappa * beta - mu)`` gives a soft switch:
- ``beta = 0`` => ``alpha = sigmoid(-mu)``. At the default
  ``moe_switch_mu=2.0``, ``alpha ~ 0.12`` (stay put).
- ``beta ~ 0.3`` (boundary peak under EMA alpha=0.3) =>
  ``alpha = sigmoid(5*0.3 - 2) = sigmoid(-0.5) ~ 0.38``.
- ``beta ~ 0.6`` => ``alpha = sigmoid(1.0) ~ 0.73`` (meaningful switch).
- ``beta -> 1`` => ``alpha -> sigmoid(3) ~ 0.95``.

The shape of this curve treats ``beta >= 0.4`` as a credible boundary and
``beta <= 0.2`` as noise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _infer_planner_k(cfg: Any) -> int:
    """Mirror of ``phase_posterior._infer_planner_k`` for the PACE-B head.

    ``p_hat`` arriving at PACE-B during inference is the planner's softmaxed
    code distribution: last dim equals ``prod(fsq_levels)`` when ``use_fsq``
    is on, else ``num_skills``. PhaseMoE expects ``moe_num_experts``
    (typically equal to the semantic ``num_phases``); a learned projection
    bridges the two when they disagree.
    """

    if bool(getattr(cfg, "use_fsq", False)):
        levels = list(getattr(cfg, "fsq_levels", []))
        if not levels:
            raise ValueError("use_fsq=True requires non-empty fsq_levels.")
        k = 1
        for lv in levels:
            k *= int(lv)
        return k
    return int(getattr(cfg, "num_skills"))


def smooth_phase_gate(
    beta: torch.Tensor,
    kappa: float,
    mu: float,
) -> torch.Tensor:
    """Return ``alpha_t = sigmoid(kappa * beta_t - mu)``.

    Parameters
    ----------
    beta : torch.Tensor
        Boundary probability, any shape.
    kappa : float
        ``moe_switch_kappa``; higher values make the switch steeper.
    mu : float
        ``moe_switch_mu``; when ``beta < mu/kappa``, ``alpha < 0.5`` (stay
        with the current expert).

    Returns
    -------
    torch.Tensor
        ``alpha_t`` in ``(0, 1)`` with the same shape as ``beta``.
    """

    return torch.sigmoid(float(kappa) * beta - float(mu))


class PhaseMoE(nn.Module):
    """Soft MoE velocity predictor conditioned on the phase posterior.

    Each expert is a small MLP: Linear -> SiLU -> Linear -> SiLU -> Linear.
    Parameter count scales roughly as ``moe_expert_hidden_dim ** 2``; at the
    default ``hidden=128`` one expert is ~33K params, so K=4 is ~130K total,
    meeting the "each expert <= 100K" acceptance target.

    Forward inputs
    --------------
    cond : ``(B, cond_dim)``
        Flow head conditioning vector (already projected by the outer
        conditioner).
    u : ``(B, latent_dim)``
        Current flow latent.
    tau : ``(B, 1)``
        Flow time step in [0, 1].
    p_hat : ``(B, K)`` or None
        Round 4 posterior; ``None`` falls back to a uniform gate.
    beta : ``(B,)`` or None
        Boundary signal; ``None`` makes ``alpha`` identically 0 so the gate
        stays pinned to the previous step.

    Output
    ------
    velocity : ``(B, latent_dim)``, computed as the weighted sum
    ``sum_k g[k] * E_k(input)``.
    """

    def __init__(self, cfg: Any) -> None:
        """Instantiate ``K`` expert MLPs and the running-gate buffer."""
        super().__init__()
        self.cfg = cfg
        self.K: int = int(cfg.moe_num_experts)
        self.latent_dim: int = int(cfg.latent_dim)
        self.cond_dim: int = int(cfg.fusion_hidden_dim)
        self.expert_hidden: int = int(cfg.moe_expert_hidden_dim)

        expert_in = self.latent_dim + self.cond_dim + 1
        self.experts = nn.ModuleList([
            self._build_expert(expert_in, self.expert_hidden, self.latent_dim)
            for _ in range(self.K)
        ])

        self.register_buffer("_running_gate", torch.empty(0), persistent=False)

    @staticmethod
    def _build_expert(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
        """Build one expert MLP with two SiLU nonlinearities.

        Intentionally light so each expert stays under 100K params at the
        default ``moe_expert_hidden_dim=128``.
        """

        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def expert_param_count(self, index: int = 0) -> int:
        """Return the parameter count of expert ``index`` (for validation)."""

        return sum(p.numel() for p in self.experts[index].parameters())

    def reset_switching(self, batch_size: int = 1) -> None:
        """Clear ``g_{t-1}``; the next step will initialise from ``p_hat``."""

        self._running_gate = torch.empty(0, device=self._running_gate.device)

    def compute_gate(
        self,
        p_hat: torch.Tensor,
        beta: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        """Compute the smooth-switching gate ``g_t``.

        Training path (``training=True``):
          - ``g_t = p_hat`` (there is no temporal ordering in a batch).
          - ``_running_gate`` is left untouched.

        Inference path (``training=False``):
          - ``alpha_t = sigmoid(kappa * beta - mu)`` broadcast to ``(B, 1)``.
          - ``g_t = alpha_t * p_hat + (1 - alpha_t) * g_{t-1}``.
          - Cold starts (first call, batch-size or device change) initialise
            ``g_t`` from ``p_hat``.
          - When ``moe_top_k > 0``, top-k sparsify and renormalise.
          - The detached result is written back to ``_running_gate``.
        """

        B, K = p_hat.shape
        if K != self.K:
            raise ValueError(f"p_hat last dim {K} != K={self.K}")

        if training:
            return p_hat

        alpha = smooth_phase_gate(
            beta, kappa=float(self.cfg.moe_switch_kappa),
            mu=float(self.cfg.moe_switch_mu),
        ).unsqueeze(-1)

        running = self._running_gate
        cold_start = (
            running.numel() == 0
            or running.shape != p_hat.shape
            or running.device != p_hat.device
        )
        if cold_start:
            g_t = p_hat
        else:
            g_t = alpha * p_hat + (1.0 - alpha) * running

        top_k = int(getattr(self.cfg, "moe_top_k", 0))
        if top_k > 0 and top_k < K:
            topk_vals, topk_idx = g_t.topk(top_k, dim=-1)
            mask = torch.zeros_like(g_t).scatter_(1, topk_idx, 1.0)
            g_t = g_t * mask
            g_t = g_t / g_t.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        self._running_gate = g_t.detach().clone()
        return g_t

    def forward(
        self,
        cond: torch.Tensor,
        u: torch.Tensor,
        tau: torch.Tensor,
        p_hat: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Route the flow latent through the MoE and return the velocity."""
        B = cond.shape[0]
        if p_hat is None:
            p_hat = cond.new_full((B, self.K), 1.0 / self.K)
        if beta is None:
            beta = cond.new_zeros(B)

        gate = self.compute_gate(p_hat=p_hat, beta=beta, training=self.training)

        expert_in = torch.cat([u, cond, tau], dim=-1)
        v_experts = torch.stack([e(expert_in) for e in self.experts], dim=1)
        v = (gate.unsqueeze(-1) * v_experts).sum(dim=1)
        return v


class FlowActionHeadPACE(nn.Module):
    """PACE-B flow action head: Euler integration with a PhaseMoE velocity.

    Mirrors the ``FlowActionHeadEuler`` signature, taking
    ``(fused_obs, phase_embed, skill_latent)`` and producing
    ``action_pred``, but the velocity comes from PhaseMoE so each flow step
    can pick a phase-aware network.

    Output shapes
    -------------
    - When ``action_chunk_size > 1``: ``(B, Ta, action_dim)``, decoded in one
      shot through a linear layer. Keeping the dim layout identical to the
      shortcut branch means downstream smoothness / imitation components in
      compute_loss do not have to change.
    - Otherwise: ``(B, action_dim)`` (legacy Euler compatibility).
    """

    def __init__(self, cfg: Any) -> None:
        """Build the conditioner, MoE, optional ``p_hat`` bridge, and decoder."""
        super().__init__()
        self.cfg = cfg
        self.Ta = int(getattr(cfg, "action_chunk_size", 1))
        self.Da = int(cfg.action_dim)
        self.latent_dim = int(cfg.latent_dim)

        cond_in = int(cfg.fusion_hidden_dim) + int(cfg.skill_embedding_dim) + int(cfg.continuous_skill_dim)
        self.conditioner = nn.Linear(cond_in, int(cfg.fusion_hidden_dim))

        self.moe = PhaseMoE(cfg)

        k_planner = _infer_planner_k(cfg)
        k_moe = int(cfg.moe_num_experts)
        if k_planner != k_moe:
            self.p_hat_proj: Optional[nn.Linear] = nn.Linear(k_planner, k_moe)
        else:
            self.p_hat_proj = None

        out_dim = self.Ta * self.Da if self.Ta > 1 else self.Da
        self.action_decoder = nn.Linear(self.latent_dim, out_dim)

    def forward(
        self,
        fused_obs: torch.Tensor,
        phase_embed: torch.Tensor,
        skill_latent: torch.Tensor,
        p_hat: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        actions_gt: Optional[torch.Tensor] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run Euler integration with MoE velocity; ``actions_gt`` / ``training`` are accepted only for API parity.

        PACE-B does not follow the shortcut self-consistency path; compute_loss
        uses ``action_pred`` directly as the MSE target.
        """

        _ = actions_gt, training
        B = fused_obs.shape[0]
        device = fused_obs.device

        cond = self.conditioner(torch.cat([fused_obs, phase_embed, skill_latent], dim=-1))
        u = torch.randn(B, self.latent_dim, device=device)

        if p_hat is not None and self.p_hat_proj is not None:
            p_hat = F.softmax(self.p_hat_proj(p_hat), dim=-1)

        flow_steps = max(int(getattr(self.cfg, "flow_steps", 4)), 1)
        dt = 1.0 / float(flow_steps)
        for i in range(flow_steps):
            tau = u.new_full((B, 1), i * dt)
            v = self.moe(cond=cond, u=u, tau=tau, p_hat=p_hat, beta=beta)
            u = u + dt * v

        decoded = self.action_decoder(u)
        if self.Ta > 1:
            action_pred = decoded.view(B, self.Ta, self.Da)
        else:
            action_pred = decoded

        return {
            "action_pred": action_pred,
            "latent_action_pred": u,
        }

    def reset_switching(self, batch_size: int = 1) -> None:
        """Pass-through to ``PhaseMoE.reset_switching`` for rollout episodes."""

        self.moe.reset_switching(batch_size=batch_size)

    def current_gate(self) -> Optional[torch.Tensor]:
        """Return the cached ``g_{t-1}`` (``None`` when empty)."""

        g = self.moe._running_gate
        return None if g.numel() == 0 else g
