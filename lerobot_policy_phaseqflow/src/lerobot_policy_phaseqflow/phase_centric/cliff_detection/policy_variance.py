"""I^(2) Predictability-Cliff estimator — action-ensemble variance.

I_hat_2(t) = -sigma_t^2 = -(1/N) * sum_i ||a_t^(i) - a_bar_t||^2

Calls the flow head N times with fixed observation conditioning; each forward
pass samples independent Gaussian noise, yielding N distinct action predictions
from which the empirical variance is computed.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class PolicyVarianceEstimator(nn.Module):
    """I^(2) cliff estimator via policy action-sample variance.

    Parameters
    ----------
    n_samples : int
        Number of independent rollouts used to estimate sigma_t^2. Default 8.
    """

    def __init__(self, n_samples: int = 8) -> None:
        super().__init__()
        if n_samples < 2:
            raise ValueError(f"n_samples must be >= 2; got {n_samples}")
        self.n_samples = n_samples

    @torch.no_grad()
    def estimate(
        self,
        flow_head: nn.Module,
        fused_obs: torch.Tensor,
        phase_embed: torch.Tensor,
        skill_latent: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute I^(2) via N independent action samples.

        Each call to ``flow_head.forward(... training=False)`` samples fresh
        Gaussian noise internally, so N calls yield N independent predictions.

        Parameters
        ----------
        flow_head : nn.Module
            A ``ShortcutFlowActionHead`` instance; must support
            ``forward(fused_obs, phase_embed, skill_latent, training=False)``.
        fused_obs, phase_embed, skill_latent : Tensor
            Shape ``(B, *)``.

        Returns
        -------
        dict
            ``i_hat_2``  : ``(B,)`` cliff signal in (-inf, 0]; higher = more predictable.
            ``sigma_sq`` : ``(B,)`` mean squared deviation (positive).
        """
        samples = []
        for _ in range(self.n_samples):
            out = flow_head(fused_obs, phase_embed, skill_latent, training=False)
            samples.append(out["action_pred"])  # (B, Ta, Da)

        stacked = torch.stack(samples, dim=1)  # (B, N, Ta, Da)
        a_bar = stacked.mean(dim=1, keepdim=True)  # (B, 1, Ta, Da)
        sq_dev = ((stacked - a_bar) ** 2).mean(dim=(1, 2, 3))  # (B,)
        return {"i_hat_2": -sq_dev, "sigma_sq": sq_dev}


__all__ = ["PolicyVarianceEstimator"]
