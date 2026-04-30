"""Phase identifiability via chunk-level InfoNCE (Innovation 3).

Core idea
---------
HierarchicalPlanner produces discrete phase codes through Gumbel-Softmax or
FSQ, but nothing in the existing training objective (flow / imitation / phase
CE) explicitly asks that ``z_t`` actually correspond to the task's latent
phase. The symptoms are codebook collapse and phase codes that do not line up
across seeds, which makes downstream phase-based innovations (PACE-A/B, PCAR)
unreliable because their "phase signal" cannot be trusted.

For every training sample ``(o_t, a_{t:t+H}, z_t)`` this head constructs an
InfoNCE contrastive loss:

    L_InfoNCE = - E log [ exp(<f(o,a), g(z)> / tau)
                          / sum_{z' in negatives} exp(<f(o,a), g(z')> / tau) ]

where ``f(.)`` is a context encoder over ``(obs, action chunk)`` and ``g(.)``
is a learnable embedding per phase id (``nn.Embedding(K, D)``). The positive
pair is the same row ``(i, i)``; negatives are other rows in the batch that
sit in a *different* phase (same-phase off-diagonal rows are masked out so
they stop contributing spurious signal).

Identifiability rationale
-------------------------
- Khemakhem et al., NeurIPS 2020 (iVAE): given an auxiliary variable ``u``,
  the latent ``z`` is identifiable up to permutation and per-axis rescaling.
- Hyvarinen et al., 2024 review on nonlinear ICA with auxiliary variables.
- InfoNCE(u, z) is a variational lower bound on the mutual information
  ``I(u; z)`` (van den Oord, 2018, CPC); maximising it pushes ``I(u; z)`` up,
  so ``z`` retains information about ``u``. Here ``u = (o_t, a_{t:t+H})`` and
  ``z`` is the phase latent.

Implementation notes
--------------------
- ``fused_obs`` comes straight from ``PhaseQFlowPolicy.forward``'s
  ``preds["encoded_obs"]`` (shape ``(B, fusion_hidden_dim)``): the context
  vector after timestep embedding is folded in, which is the densest
  representation available.
- ``action_chunk`` prefers ground-truth ``actions`` of shape ``(B, Ta, Da)``.
  When the batch only carries a single-step ``(B, Da)``, compute_loss pads
  it to ``(B, 1, Da)``; this is a weaker but still usable substitute, and
  later rounds plug in real chunks once the dataloader delivers them.
- Negative sampling is batch-internal. No DDP all-gather is required: we use
  only the local-rank batch for simplicity; if the effective batch is too
  small (< 4), the step silently skips.

Activation
----------
This module is wired into ``compute_loss`` only when
``PhaseQFlowConfig.use_chunk_infonce = True``. Otherwise ``Policy.__init__``
sets ``chunk_infonce_head`` to None and the path is skipped.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_phaseqflow import PhaseQFlowConfig


class ChunkInfoNCEHead(nn.Module):
    """Chunk-level InfoNCE head for phase identifiability.

    Parameters
    ----------
    cfg : PhaseQFlowConfig
        Fields used:
        - ``num_phases`` (K)
        - ``fusion_hidden_dim`` (D)
        - ``action_chunk_size`` (Ta)
        - ``action_dim`` (Da)
        - ``chunk_infonce_temperature`` (tau)

    Forward inputs
    --------------
    fused_obs : (B, D)
        Context vector.
    action_chunk : (B, Ta, Da)
        Action chunk.
    phase_logits : (B, K)
        Phase logits from HierarchicalPlanner.

    Returns
    -------
    loss : scalar tensor
    diag : dict
        - ``info_nce_acc`` : top-1 accuracy of recovering the same-row phase
          embedding.
        - ``phase_entropy`` : entropy of ``softmax(phase_logits)``; higher
          means a more active codebook.
        - ``num_valid_rows`` : number of rows that actually participated in
          the InfoNCE computation (degenerate batches are skipped).
    """

    def __init__(self, cfg: PhaseQFlowConfig) -> None:
        """Build the context encoder and per-phase embedding tables."""
        super().__init__()
        self.cfg = cfg
        self.hierarchical: bool = (
            bool(getattr(cfg, "use_fsq", False))
            and str(getattr(cfg, "phase_mode", "flat")) == "hierarchical"
        )

        if bool(getattr(cfg, "use_fsq", False)):
            if self.hierarchical:
                levels_macro: List[int] = list(getattr(cfg, "fsq_levels_macro", [5, 4]))
                levels_micro: List[int] = list(getattr(cfg, "fsq_levels_micro", [6, 5]))
                self.K = int(math.prod(levels_macro))
                self.K_micro: int = int(math.prod(levels_micro))
            else:
                levels = list(getattr(cfg, "fsq_levels", []))
                if not levels:
                    raise ValueError("use_fsq=True requires non-empty fsq_levels.")
                self.K = 1
                for lv in levels:
                    self.K *= int(lv)
                self.K_micro = 0
        else:
            self.K = int(cfg.num_skills)
            self.K_micro = 0

        self.D = int(cfg.fusion_hidden_dim)
        self.Ta = int(cfg.action_chunk_size)
        self.Da = int(cfg.action_dim)
        self.tau = float(cfg.chunk_infonce_temperature)

        chunk_flat_dim = self.Ta * self.Da

        self.context_encoder = nn.Sequential(
            nn.Linear(self.D + chunk_flat_dim, self.D),
            nn.SiLU(),
            nn.Linear(self.D, self.D),
        )
        self.phase_embed = nn.Embedding(self.K, self.D)
        if self.hierarchical and self.K_micro > 0:
            self.phase_embed_micro = nn.Embedding(self.K_micro, self.D)
        else:
            self.phase_embed_micro = None  # type: ignore[assignment]

    def _encode_chunk(
        self,
        fused_obs: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Encode (fused_obs, action_chunk) into a unit-norm context vector for the InfoNCE critic.

        Pads or trims ``action_chunk`` to exactly ``(B, Ta, Da)`` so the encoder
        weight shape is fixed regardless of the dataloader's chunk length.
        """
        B = fused_obs.shape[0]
        device = fused_obs.device
        Ta_in = action_chunk.shape[1]
        if Ta_in >= self.Ta:
            chunk = action_chunk[:, : self.Ta, :]
        else:
            pad = torch.zeros(
                B, self.Ta - Ta_in, action_chunk.shape[2],
                device=device, dtype=action_chunk.dtype,
            )
            chunk = torch.cat([action_chunk, pad], dim=1)
        if chunk.shape[2] != self.Da:
            if chunk.shape[2] > self.Da:
                chunk = chunk[..., : self.Da]
            else:
                pad_d = torch.zeros(
                    B, self.Ta, self.Da - chunk.shape[2],
                    device=device, dtype=chunk.dtype,
                )
                chunk = torch.cat([chunk, pad_d], dim=-1)
        chunk_flat = chunk.reshape(B, self.Ta * self.Da)
        ctx = self.context_encoder(torch.cat([fused_obs, chunk_flat], dim=-1))
        return F.normalize(ctx, dim=-1)

    def _single_level_infonce(
        self,
        ctx: torch.Tensor,
        logits: torch.Tensor,
        embed_table: nn.Embedding,
        K: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Core InfoNCE loss for one codebook level (macro or micro).

        Same-phase off-diagonal pairs are masked out of the denominator so they
        cannot act as spurious negatives — without this mask, two rows that share
        a phase id would push each other's embeddings apart, collapsing the code.
        """
        B = ctx.shape[0]
        device = ctx.device
        zero = torch.zeros((), device=device, dtype=ctx.dtype)

        if B < 2 or K < 2 or logits.shape[1] != K:
            return zero, {"info_nce_acc": 0.0, "phase_entropy": 0.0, "num_valid_rows": 0}

        with torch.no_grad():
            phase_probs = F.softmax(logits, dim=-1)
        phase_id = logits.argmax(dim=-1)
        z_embeds = F.normalize(embed_table(phase_id), dim=-1)

        sim = ctx @ z_embeds.T / self.tau
        same_phase = phase_id.unsqueeze(0) == phase_id.unsqueeze(1)
        eye = torch.eye(B, device=device, dtype=torch.bool)
        neg_plus_self_mask = (~same_phase) | eye
        has_any_neg = (~same_phase & ~eye).any(dim=1)

        sim_masked = sim.masked_fill(~neg_plus_self_mask, float("-inf"))
        log_prob = sim.diagonal() - torch.logsumexp(sim_masked, dim=1)

        n_valid = int(has_any_neg.sum().item())
        if n_valid == 0:
            loss = zero
        else:
            loss = -(log_prob * has_any_neg.float()).sum() / float(n_valid)

        with torch.no_grad():
            sim_all = ctx @ F.normalize(embed_table.weight, dim=-1).T
            acc = (sim_all.argmax(dim=-1) == phase_id).float().mean()
            entropy = -(phase_probs * phase_probs.clamp_min(1e-8).log()).sum(-1).mean()

        return loss, {
            "info_nce_acc": float(acc.item()),
            "phase_entropy": float(entropy.item()),
            "num_valid_rows": n_valid,
        }

    def forward(
        self,
        fused_obs: torch.Tensor,
        action_chunk: torch.Tensor,
        phase_logits: torch.Tensor,
        logits_micro: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the InfoNCE loss and a diagnostics dict.

        In hierarchical mode (when ``logits_micro`` is not None):
            L_NCE_total = L_NCE_macro + 0.5 * L_NCE_micro

        In flat mode (``logits_micro is None``):
            L_NCE_total = L_NCE_macro  (single-level, backward compatible)
        """
        if fused_obs.ndim != 2 or fused_obs.shape[1] != self.D:
            raise ValueError(
                f"fused_obs must be (B, {self.D}); got {tuple(fused_obs.shape)}"
            )
        if action_chunk.ndim != 3:
            raise ValueError(
                f"action_chunk must be (B, Ta, Da); got {tuple(action_chunk.shape)}"
            )
        if phase_logits.ndim != 2 or phase_logits.shape[1] != self.K:
            raise ValueError(
                f"phase_logits must be (B, {self.K}); got {tuple(phase_logits.shape)}"
            )

        B = fused_obs.shape[0]
        device = fused_obs.device
        zero = torch.zeros((), device=device, dtype=fused_obs.dtype)

        if B < 2 or self.K < 2:
            return zero, {"info_nce_acc": 0.0, "phase_entropy": 0.0, "num_valid_rows": 0}

        ctx = self._encode_chunk(fused_obs, action_chunk)

        # Macro (primary) level
        loss_macro, diag_macro = self._single_level_infonce(
            ctx, phase_logits, self.phase_embed, self.K
        )

        # Micro level (hierarchical mode only)
        if (
            self.hierarchical
            and logits_micro is not None
            and self.phase_embed_micro is not None
            and logits_micro.shape[1] == self.K_micro
        ):
            loss_micro, diag_micro = self._single_level_infonce(
                ctx, logits_micro, self.phase_embed_micro, self.K_micro
            )
            total_loss = loss_macro + 0.5 * loss_micro
            diag: Dict[str, float] = {
                "info_nce_acc": diag_macro["info_nce_acc"],
                "info_nce_acc_micro": diag_micro["info_nce_acc"],
                "phase_entropy": diag_macro["phase_entropy"],
                "phase_entropy_micro": diag_micro["phase_entropy"],
                "num_valid_rows": diag_macro["num_valid_rows"],
            }
        else:
            total_loss = loss_macro
            diag = diag_macro

        return total_loss, diag


def chunk_infonce_loss(
    head: ChunkInfoNCEHead,
    fused_obs: torch.Tensor,
    action_chunk: torch.Tensor,
    phase_logits: torch.Tensor,
    logits_micro: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Functional wrapper around ``ChunkInfoNCEHead.forward``.

    Handy for external callers (e.g. ``scripts/verify_identifiability.py`` and
    tests) that do not want to manage the module's eval/train state.
    In hierarchical mode pass ``logits_micro`` to enable macro+micro InfoNCE.
    """
    return head(
        fused_obs=fused_obs,
        action_chunk=action_chunk,
        phase_logits=phase_logits,
        logits_micro=logits_micro,
    )


def sample_positive_negative_pairs(*args: Any, **kwargs: Any) -> Any:
    """Reserved hook for an offline positive/negative sampler.

    Unused in the batch-internal formulation: pairs are implicit in the
    similarity matrix plus the same_phase mask. Kept for future variants that
    need an explicit sampler (e.g. a contrastive replay buffer).
    """

    raise NotImplementedError(
        "sample_positive_negative_pairs is unused in the batch-internal "
        "InfoNCE path; reserved for future offline/negative-buffer variants."
    )
