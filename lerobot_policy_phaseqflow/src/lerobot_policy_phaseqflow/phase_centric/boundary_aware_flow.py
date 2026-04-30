"""Boundary-Aware Flow Loss (Phase C rename + simplification of PACE-A).

Renamed from ``pace_a_loss.py``; the original shim at that path re-exports
everything from here for backward compatibility.

Changes from PACE-A:
- Entropy regulariser ``-η H(β)`` removed (master plan §3.3 decision).
- Weight uses ``beta_t^micro`` (micro-level Bhattacharyya signal) when
  available; falls back to the macro ``beta_t`` in flat mode.
- New config switch ``use_boundary_reweight: bool = True``; when False the
  weighting is disabled (pure FM, useful for ablation).
- Functional signature simplified: no ``entropy_weight`` / ``ablation_mode``.

Formula
-------
L_policy = L_FM + λ_SC * L_SC

where L_FM uses per-step weights:
    w(β_t) = 1 + λ_β * β_t   if use_boundary_reweight else 1.0

(``boundary_reweight_lambda`` λ_β defaults to 0.5.)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _align_beta(beta_t: torch.Tensor, B: int, Ta: int) -> torch.Tensor:
    """Broadcast ``beta_t`` to shape ``(B, Ta)``."""
    if beta_t.ndim == 1:
        if beta_t.shape[0] != B:
            raise ValueError(f"beta_t (B,) has B={beta_t.shape[0]} != {B}")
        return beta_t.unsqueeze(-1).expand(B, Ta)
    if beta_t.ndim == 2:
        if beta_t.shape == (B, Ta):
            return beta_t
        if beta_t.shape == (B, 1):
            return beta_t.expand(B, Ta)
        raise ValueError(
            f"beta_t (B, ?) must be (B, 1) or (B, Ta); got {tuple(beta_t.shape)}"
        )
    if beta_t.ndim == 3:
        if beta_t.shape == (B, Ta, 1):
            return beta_t.squeeze(-1)
        raise ValueError(f"beta_t 3D must be (B, Ta, 1); got {tuple(beta_t.shape)}")
    raise ValueError(f"beta_t ndim must be 1/2/3; got {beta_t.ndim}")


def compute_boundary_aware_flow_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    beta_t: torch.Tensor,
    lambda_weight: float = 0.5,
    use_boundary_reweight: bool = True,
) -> Dict[str, torch.Tensor]:
    """Boundary-aware weighted FM loss (no entropy regulariser).

    Parameters
    ----------
    v_pred, v_target : ``(B, Ta, Da)``
        Predicted and target velocity fields from Shortcut FM.
    beta_t : ``(B,)`` | ``(B, Ta)`` | ``(B, Ta, 1)``
        Boundary signal (Bhattacharyya β_t). Prefer passing the micro-level
        signal ``beta_micro`` when available.  Pass already ``.detach()``'d if
        you want to stop gradients to the planner.
    lambda_weight : float
        λ_β for ``w(β) = 1 + λ_β * β``. Default 0.5.
    use_boundary_reweight : bool
        When False, weights are pinned to 1.0 (vanilla FM; ablation mode).

    Returns
    -------
    dict
        ``fm_loss`` : scalar weighted MSE.
        ``mean_beta`` : detached scalar for logging.
        ``max_beta`` : detached scalar for logging.
        ``weighted_mse_per_step`` : ``(B, Ta)`` for diagnostic scripts.
    """
    if v_pred.ndim != 3 or v_target.ndim != 3:
        raise ValueError(
            f"v_pred / v_target must be (B, Ta, Da); got "
            f"{tuple(v_pred.shape)}, {tuple(v_target.shape)}"
        )
    if v_pred.shape != v_target.shape:
        raise ValueError(
            f"v_pred {tuple(v_pred.shape)} != v_target {tuple(v_target.shape)}"
        )

    B, Ta, _Da = v_pred.shape
    beta_aligned = _align_beta(beta_t, B, Ta).clamp(0.0, 1.0)

    per_step_mse = (v_pred - v_target).pow(2).mean(dim=-1)  # (B, Ta)

    if use_boundary_reweight:
        weights = 1.0 + float(lambda_weight) * beta_aligned
    else:
        weights = torch.ones_like(per_step_mse)

    weighted_mse = weights * per_step_mse
    fm_loss = weighted_mse.mean()

    with torch.no_grad():
        mean_beta = beta_aligned.mean().detach()
        max_beta = beta_aligned.max().detach()

    return {
        "fm_loss": fm_loss,
        "mean_beta": mean_beta,
        "max_beta": max_beta,
        "weighted_mse_per_step": weighted_mse.detach(),
    }


def boundary_aware_reweight(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    beta_t: torch.Tensor,
    lambda_weight: float = 0.5,
) -> torch.Tensor:
    """Functional helper: boundary-aware weighted MSE scalar."""
    out = compute_boundary_aware_flow_loss(
        v_pred=v_pred,
        v_target=v_target,
        beta_t=beta_t,
        lambda_weight=lambda_weight,
        use_boundary_reweight=True,
    )
    return out["fm_loss"]


__all__ = [
    "_align_beta",
    "compute_boundary_aware_flow_loss",
    "boundary_aware_reweight",
]
