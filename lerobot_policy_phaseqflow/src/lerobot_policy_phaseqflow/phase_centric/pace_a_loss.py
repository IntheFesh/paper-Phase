"""DEPRECATED: import from boundary_aware_flow instead.

This file is kept as a backward-compatibility shim.  All symbols were moved to
``phase_centric/boundary_aware_flow.py`` in Phase C.  The entropy-regulariser
helpers (``pace_a_entropy_reg``, ``pace_a_reweight``) are preserved here for
ablation scripts that still reference them directly.
"""

# ruff: noqa: F401, F403
from .boundary_aware_flow import (  # noqa: F401
    _align_beta,
    compute_boundary_aware_flow_loss,
    boundary_aware_reweight,
)

# ---------------------------------------------------------------------------
# Legacy names kept for ablation scripts / old checkpoints
# ---------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from typing import Dict


def compute_pace_a_flow_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    beta_t: torch.Tensor,
    lambda_weight: float,
    entropy_weight: float,
    ablation_mode: str = "full",
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Legacy PACE-A weighted FM + Bernoulli entropy (backward-compatible shim).

    The boundary_aware_flow main path drops the entropy term; this shim
    preserves the original behaviour for ablation scripts and existing tests.
    """
    if ablation_mode not in {"full", "no_weight", "no_entropy"}:
        raise ValueError(f"unknown ablation_mode: {ablation_mode!r}")

    B, Ta, _Da = v_pred.shape
    beta_aligned = _align_beta(beta_t, B, Ta).clamp(0.0, 1.0)
    per_step_mse = (v_pred - v_target).pow(2).mean(dim=-1)

    if ablation_mode == "no_weight":
        weights = torch.ones_like(per_step_mse)
    else:
        weights = 1.0 + float(lambda_weight) * beta_aligned

    weighted_mse = weights * per_step_mse
    fm_loss = weighted_mse.mean()

    if ablation_mode == "no_entropy":
        entropy_reg = torch.zeros((), device=v_pred.device, dtype=v_pred.dtype)
    else:
        b = beta_aligned.clamp(eps, 1.0 - eps)
        bernoulli_H = -(b * b.log() + (1.0 - b) * (1.0 - b).log())
        H_mean = bernoulli_H.mean()
        entropy_reg = -float(entropy_weight) * H_mean

    total = fm_loss + entropy_reg

    with torch.no_grad():
        mean_beta = beta_aligned.mean().detach()
        max_beta = beta_aligned.max().detach()

    return {
        "fm_loss": fm_loss,
        "entropy_reg": entropy_reg,
        "total": total,
        "mean_beta": mean_beta,
        "max_beta": max_beta,
        "weighted_mse_per_step": weighted_mse.detach(),
    }


def pace_a_reweight(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    beta_t: torch.Tensor,
    lambda_weight: float,
    ablation_mode: str = "full",
) -> torch.Tensor:
    """Legacy helper shim."""
    return boundary_aware_reweight(v_pred, v_target, beta_t, lambda_weight)


def pace_a_entropy_reg(
    beta_t: torch.Tensor, entropy_weight: float, eps: float = 1e-6,
) -> torch.Tensor:
    """Legacy Bernoulli entropy regulariser ``-eta * H(beta)`` (backward-compat shim).

    Note: the Phase C main-path (boundary_aware_flow) dropped this term.
    This function is preserved for ablation scripts and tests.
    """
    b = beta_t.clamp(eps, 1.0 - eps)
    H = -(b * b.log() + (1.0 - b) * (1.0 - b).log()).mean()
    return -float(entropy_weight) * H
