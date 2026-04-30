"""I^(1) Predictability-Cliff estimator — Bhattacharyya boundary signal wrapper.

Thin delegator over PhasePosteriorEstimator in the cliff_detection public
interface.  I_hat_1(t) = -beta_t.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch
import torch.nn as nn

from lerobot_policy_phaseqflow.phase_centric.cliff_estimators import compute_I_hat_1
from lerobot_policy_phaseqflow.phase_centric.phase_posterior import (
    PhasePosteriorEstimator,
)

if TYPE_CHECKING:
    from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig


class PosteriorBhattacharyyaEstimator(nn.Module):
    """I^(1) cliff estimator wrapping PhasePosteriorEstimator.

    Parameters
    ----------
    cfg : PhaseQFlowConfig
        Forwarded unchanged to PhasePosteriorEstimator.
    """

    def __init__(self, cfg: "PhaseQFlowConfig") -> None:
        super().__init__()
        self._posterior = PhasePosteriorEstimator(cfg)

    @property
    def K(self) -> int:
        """Codebook / skill dimension K."""
        return self._posterior.K

    def reset(self, batch_size: int = 1, device: torch.device | None = None) -> None:
        """Reset running posterior state; call at episode start."""
        self._posterior.reset(batch_size=batch_size, device=device)

    def step(self, phase_logits_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single-step update and cliff-signal computation.

        Parameters
        ----------
        phase_logits_t : Tensor
            ``(B, K)`` raw phase logits.

        Returns
        -------
        dict
            ``i_hat_1`` : ``(B,)`` cliff signal in [-1, 0]; higher = more predictable.
            ``beta``    : ``(B,)`` raw Bhattacharyya distance in [0, 1].
            ``p_hat``   : ``(B, K)`` smoothed phase posterior.
        """
        out = self._posterior.step(phase_logits_t)
        i_hat_1 = compute_I_hat_1(out["beta"])
        return {"i_hat_1": i_hat_1, "beta": out["beta"], "p_hat": out["p_hat"]}


__all__ = ["PosteriorBhattacharyyaEstimator"]
