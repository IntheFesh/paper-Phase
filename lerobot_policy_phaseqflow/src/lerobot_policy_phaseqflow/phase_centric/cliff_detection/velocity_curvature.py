"""I^(3) Predictability-Cliff estimator — velocity-field conditioning difference.

I_hat_3(t) = -||v_theta(x_tau, tau, c_t) - v_theta(x_tau, tau, c_{t-1})||_2^2

Evaluates the flow velocity at a fixed anchor (x_tau, tau) under two
consecutive conditioning vectors c_t and c_{t-1}.  Compatible only with
ShortcutFlowActionHead (which exposes .velocity() and .compute_cond()).
FlowActionHeadPACE uses PhaseMoE and does not expose these methods.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

_ANCHOR_TAU: float = 0.5   # midpoint of the [0,1] flow time interval
_ANCHOR_D: float = 1.0     # shortcut step-size anchor


class VelocityCurvatureEstimator(nn.Module):
    """I^(3) cliff estimator: velocity-field conditioning difference.

    Maintains a one-step cache of the previous conditioning vector c_{t-1}.
    On the first call after :meth:`reset`, returns ``i_hat_3 = None``
    (no predecessor exists yet).

    Parameters
    ----------
    anchor_tau : float
        Flow time anchor tau in (0, 1). Default 0.5 (midpoint).
    anchor_x_std : float
        Standard deviation for the fixed anchor noise x_tau drawn at init.
        Default 1.0.
    """

    def __init__(
        self,
        anchor_tau: float = _ANCHOR_TAU,
        anchor_x_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.anchor_tau = float(anchor_tau)
        self.anchor_x_std = float(anchor_x_std)

        self._prev_cond: Optional[torch.Tensor] = None
        self._anchor_x: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Clear cached state; call at episode start."""
        self._prev_cond = None
        self._anchor_x = None

    @torch.no_grad()
    def update(
        self,
        flow_head: nn.Module,
        fused_obs: torch.Tensor,
        phase_embed: torch.Tensor,
        skill_latent: torch.Tensor,
    ) -> Dict[str, object]:
        """Compute I^(3) for the current step.

        Parameters
        ----------
        flow_head : nn.Module
            ``ShortcutFlowActionHead`` exposing ``compute_cond()`` and
            ``velocity()``.
        fused_obs, phase_embed, skill_latent : Tensor
            Shape ``(B, *)``.

        Returns
        -------
        dict
            ``i_hat_3``     : ``(B,)`` Tensor or ``None`` at step 0.
            ``cond_diff_sq``: ``(B,)`` Tensor (positive; equals ``-i_hat_3``)
                              or ``None`` at step 0.
        """
        cond_t = flow_head.compute_cond(fused_obs, phase_embed, skill_latent)  # (B, H)
        B = cond_t.shape[0]
        device = cond_t.device
        dtype = cond_t.dtype

        if self._anchor_x is None or self._anchor_x.shape[0] != B:
            Ta = getattr(flow_head, "Ta", None)
            Da = getattr(flow_head, "Da", None)
            if Ta is None or Da is None:
                raise AttributeError(
                    "flow_head must expose .Ta and .Da; "
                    "is this a ShortcutFlowActionHead?"
                )
            self._anchor_x = self.anchor_x_std * torch.randn(
                B, Ta, Da, device=device, dtype=dtype
            )

        if self._prev_cond is None:
            self._prev_cond = cond_t.detach()
            return {"i_hat_3": None, "cond_diff_sq": None}

        v_t = flow_head.velocity(
            self._anchor_x, self.anchor_tau, cond_t, d=_ANCHOR_D
        )  # (B, Ta, Da)
        v_prev = flow_head.velocity(
            self._anchor_x, self.anchor_tau, self._prev_cond, d=_ANCHOR_D
        )  # (B, Ta, Da)

        diff_sq = ((v_t - v_prev) ** 2).mean(dim=(1, 2))  # (B,)
        self._prev_cond = cond_t.detach()
        return {"i_hat_3": -diff_sq, "cond_diff_sq": diff_sq}


__all__ = ["VelocityCurvatureEstimator"]
