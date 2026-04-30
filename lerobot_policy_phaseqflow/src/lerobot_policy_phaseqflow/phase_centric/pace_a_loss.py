"""PACE-A: phase-aware loss reweighting plus entropy regularisation.

Core idea
---------
Shortcut Flow Matching weights its MSE loss uniformly across timesteps, but
the action predictions most likely to be wrong sit right at phase
transitions (the H1 diagnostic shows boundary loss is 10-50% higher than
interior loss). PACE-A treats the boundary signal ``beta_t`` from Round 4 as
a soft weight that explicitly amplifies gradients on boundary samples:

.. math::

   \\mathcal{L}_\\text{FM-PACE}(\\theta)
     = \\mathbb{E}_t\\!\\bigl[(1 + \\lambda\\, \\beta_t)\\, \\| v_\\theta(x_\\tau, \\tau, c_t) - v^*_t \\|^2\\bigr]
       - \\eta\\, H(\\beta)

where:

- :math:`\\lambda =` ``pace_a_lambda``: the boundary-step multiplier.
  ``lambda=0`` collapses back to vanilla FM.
- :math:`\\eta =` ``pace_a_entropy_weight``: weight on the Bernoulli entropy
  regulariser (negative sign means we maximise entropy).
- :math:`H(\\beta) = -\\mathbb{E}_t [\\beta_t \\log \\beta_t + (1-\\beta_t) \\log (1-\\beta_t)]`
  is the Bernoulli entropy of ``beta_t``, preventing all ``beta`` from
  collapsing to 0 (global = vanilla FM) or 1 (global = overfit boundary).

Theoretical motivation
----------------------
Under the Bhattacharyya formulation from Round 4, ``beta_t`` is in [0, 1],
differentiable, and bounded. Multiplying the per-timestep FM loss by
``1 + lambda * beta_t`` amounts to a weighted-Lipschitz constraint on the
flow velocity (Appendix B shows this beta-weighted MSE is a tight
variational lower bound on the weighted Lipschitz constant). The entropy
term blocks the degenerate solutions beta identically 0 or identically 1,
so PACE-A always contributes a non-trivial gradient during training.

Design decisions
----------------
1. ``pace_a_detach_beta=True`` by default: ``beta_t`` gradients do not flow
   back through the planner, so joint optimisation cannot exploit the trivial
   "beta identically 0 drives fm_loss down" minimum. Flip it for ablation.
2. ``pace_a_ablation_mode``:
   - ``"full"``: weighting plus entropy (default).
   - ``"no_weight"``: entropy only; loss degenerates to vanilla FM.
   - ``"no_entropy"``: weighting only, no entropy regulariser.
3. ``beta`` shape compatibility: accepts ``(B,)``, ``(B, Ta)``, or
   ``(B, Ta, 1)``; broadcast internally to ``(B, Ta)`` to line up with the
   per-step MSE tensor.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _align_beta(beta_t: torch.Tensor, B: int, Ta: int) -> torch.Tensor:
    """Broadcast ``beta_t`` to shape ``(B, Ta)``.

    Accepts three input shapes:
    - ``(B,)``: one ``beta`` per sample, tiled along Ta.
    - ``(B, Ta)``: returned as-is.
    - ``(B, Ta, 1)``: last dim squeezed.
    """

    if beta_t.ndim == 1:
        if beta_t.shape[0] != B:
            raise ValueError(f"beta_t (B,) has B={beta_t.shape[0]} != {B}")
        return beta_t.unsqueeze(-1).expand(B, Ta)
    if beta_t.ndim == 2:
        if beta_t.shape == (B, Ta):
            return beta_t
        if beta_t.shape == (B, 1):
            return beta_t.expand(B, Ta)
        raise ValueError(f"beta_t (B, ?) must be (B, 1) or (B, Ta); got {tuple(beta_t.shape)}")
    if beta_t.ndim == 3:
        if beta_t.shape == (B, Ta, 1):
            return beta_t.squeeze(-1)
        raise ValueError(f"beta_t 3D must be (B, Ta, 1); got {tuple(beta_t.shape)}")
    raise ValueError(f"beta_t ndim must be 1/2/3; got {beta_t.ndim}")


def compute_pace_a_flow_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    beta_t: torch.Tensor,
    lambda_weight: float,
    entropy_weight: float,
    ablation_mode: str = "full",
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """PACE-A weighted FM loss plus Bernoulli entropy regulariser.

    Parameters
    ----------
    v_pred, v_target : ``(B, Ta, Da)``
        Predicted and target velocity fields from Shortcut FM.
    beta_t : ``(B,)`` | ``(B, Ta)`` | ``(B, Ta, 1)``
        Boundary signal produced in Round 4. Pass it already ``.detach()``'d
        if you want to stop gradients to the planner; otherwise it flows
        back (decision controlled upstream by ``pace_a_detach_beta``).
    lambda_weight : float
        ``lambda``; ignored when ``ablation_mode="no_weight"``.
    entropy_weight : float
        ``eta``; ignored when ``ablation_mode="no_entropy"``.
    ablation_mode : {"full", "no_weight", "no_entropy"}
        Ablation switch. ``"no_weight"`` pins per-step weights to 1.0 (vanilla
        FM); ``"no_entropy"`` zeros the entropy term.
    eps : float
        Lower clamp for the entropy term, avoiding ``log 0``.

    Returns
    -------
    dict
        - ``fm_loss``: weighted MSE scalar.
        - ``entropy_reg``: ``-eta * H(beta)`` scalar; 0 under ``"no_entropy"``.
        - ``total``: ``fm_loss + entropy_reg``; compute_loss drops this in
          place of the original ``fm_loss``.
        - ``mean_beta``: detached scalar tensor for logging.
        - ``max_beta``: detached scalar tensor.
        - ``weighted_mse_per_step``: ``(B, Ta)`` slice-friendly tensor for
          boundary-vs-interior sanity scripts.
    """

    if ablation_mode not in {"full", "no_weight", "no_entropy"}:
        raise ValueError(f"unknown ablation_mode: {ablation_mode!r}")

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
    beta_aligned = _align_beta(beta_t, B, Ta)
    beta_aligned = beta_aligned.clamp(0.0, 1.0)

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
    """Functional helper: weighted MSE only, no entropy term.

    Equivalent to the ``fm_loss`` return value of
    :func:`compute_pace_a_flow_loss` (same implementation under the hood);
    exposed so Round 6 MoE code can assemble its own composite loss.
    """

    out = compute_pace_a_flow_loss(
        v_pred=v_pred,
        v_target=v_target,
        beta_t=beta_t,
        lambda_weight=lambda_weight,
        entropy_weight=0.0,
        ablation_mode="no_entropy" if ablation_mode == "no_weight" else ablation_mode,
    )
    if ablation_mode == "no_weight":
        out = compute_pace_a_flow_loss(
            v_pred=v_pred, v_target=v_target, beta_t=beta_t,
            lambda_weight=0.0, entropy_weight=0.0,
            ablation_mode="no_weight",
        )
    return out["fm_loss"]


def pace_a_entropy_reg(
    beta_t: torch.Tensor, entropy_weight: float, eps: float = 1e-6,
) -> torch.Tensor:
    """Functional helper: Bernoulli entropy regulariser ``-eta * H(beta)``."""

    b = beta_t.clamp(eps, 1.0 - eps)
    H = -(b * b.log() + (1.0 - b) * (1.0 - b).log()).mean()
    return -float(entropy_weight) * H
