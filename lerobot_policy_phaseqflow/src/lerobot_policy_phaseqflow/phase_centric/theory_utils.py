"""Shared math helpers for the Phase-Centric subpackage.

This module holds the small mathematical primitives that multiple phase-centric
innovations need: run-length posteriors, Shannon entropy, Jensen-Shannon
divergence, empirical CDFs. Keeping them here avoids re-implementing the same
formulas in ``identifiability.py`` and ``pcar_trigger.py``.

Current state
-------------
- ``PredictiveInfoEstimator`` — implemented (Phase B).
- Other helpers remain as lazy-fill placeholders.

Expected API sketch
-------------------
- ``run_length_posterior(prior, likelihood) -> torch.Tensor``
- ``shannon_entropy(probs, dim=-1) -> torch.Tensor``
- ``empirical_cdf(samples, grid) -> np.ndarray``
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PredictiveInfoEstimator — Phase B implementation
# ---------------------------------------------------------------------------

class PredictiveInfoEstimator(nn.Module):
    """Upper-bound mutual-information estimator I(X; C) via InfoNCE.

    Used as the oracle for calibrating the chunk-level InfoNCE loss
    temperature τ and weight λ (see scripts/calibration/
    train_predictive_info_oracle.py).

    The estimator uses a bilinear critic:
        f(x, c) = x^T W c
    and computes the per-step InfoNCE lower bound to I(X; C).

    Parameters
    ----------
    x_dim : int
        Dimension of the future context / action embedding X.
    c_dim : int
        Dimension of the conditioning vector C.
    hidden_dim : int
        Width of the bilinear projection. Default 128.
    """

    def __init__(self, x_dim: int, c_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.x_proj = nn.Linear(x_dim, hidden_dim, bias=False)
        self.c_proj = nn.Linear(c_dim, hidden_dim, bias=False)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute the InfoNCE MI lower bound for a batch of (x, c) pairs.

        Parameters
        ----------
        x : Tensor
            ``(B, x_dim)`` future context embeddings.
        c : Tensor
            ``(B, c_dim)`` conditioning vectors (paired with the same index).

        Returns
        -------
        dict
            ``mi_lower_bound``: scalar — InfoNCE lower bound on I(X; C).
            ``logits``        : ``(B, B)`` critic score matrix (for diagnostics).
        """
        xp = self.x_proj(x)  # (B, H)
        cp = self.c_proj(c)  # (B, H)
        logits = xp @ cp.T   # (B, B)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = F.cross_entropy(logits, labels)
        mi_lower_bound = torch.log(torch.tensor(float(logits.shape[0]))) - loss
        return {"mi_lower_bound": mi_lower_bound, "logits": logits}

    @torch.no_grad()
    def estimate_per_timestep(
        self,
        x_seq: torch.Tensor,
        c_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate MI lower bound at each time step independently.

        Parameters
        ----------
        x_seq : Tensor
            ``(T, B, x_dim)`` sequence of future context embeddings.
        c_seq : Tensor
            ``(T, B, c_dim)`` sequence of conditioning vectors.

        Returns
        -------
        Tensor
            ``(T,)`` per-timestep InfoNCE lower bound estimates.
        """
        T = x_seq.shape[0]
        mi_vals = []
        for t in range(T):
            out = self.forward(x_seq[t], c_seq[t])
            mi_vals.append(out["mi_lower_bound"])
        return torch.stack(mi_vals)


# ---------------------------------------------------------------------------
# Placeholder helpers (lazy fill-in when callers need them)
# ---------------------------------------------------------------------------

def run_length_posterior(*args: Any, **kwargs: Any) -> Any:
    """Placeholder; will be filled when first needed by PCAR."""

    raise NotImplementedError("run_length_posterior: lazy fill-in.")


def shannon_entropy(*args: Any, **kwargs: Any) -> Any:
    """Placeholder; will be filled when first needed by PACE-A entropy regulariser."""

    raise NotImplementedError("shannon_entropy: lazy fill-in.")


def empirical_cdf(*args: Any, **kwargs: Any) -> Any:
    """Placeholder reserved for future statistical diagnostics."""

    raise NotImplementedError("empirical_cdf: lazy fill-in.")
