"""Bidirectional Decoding test-time sampler (Liu et al., arXiv 2408.17355).

At every rollout step the flow action head samples ``N`` candidate chunks.
BIDSampler scores each candidate by

    score_i = w_b * backward_coherence_i + w_f * forward_contrast_i

where
  * ``backward_coherence`` rewards agreement with the *previous* selected
    chunk (so action streams stay temporally consistent);
  * ``forward_contrast`` rewards being distinct from a *weak policy*
    (here approximated by an EMA of the candidate means); this prevents the
    system from collapsing to the weak-policy average.

The sampler is stateful across an episode and must be :meth:`reset` between
episodes - done automatically by :meth:`PhaseQFlowPolicy.reset`.
"""

from __future__ import annotations

from typing import Optional

import torch


class BIDSampler:
    """Stateful BID scorer / selector for test-time action chunks."""

    def __init__(self, config) -> None:
        """Store the config and initialise the per-episode state to empty."""
        self.config = config
        self._prev_chunk: Optional[torch.Tensor] = None
        self._weak_mean: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Clear per-episode state; call at the start of every episode."""
        self._prev_chunk = None
        self._weak_mean = None

    def score(self, chunks: torch.Tensor) -> torch.Tensor:
        """Score each of ``N`` candidate chunks. Higher is better.

        ``chunks``: ``(N, Ta, Da)``
        returns : ``(N,)``
        """
        if chunks.ndim != 3:
            raise ValueError(f"BIDSampler.score expects (N, Ta, Da); got {tuple(chunks.shape)}")

        if self._prev_chunk is not None:
            prev = self._prev_chunk.to(chunks.device).to(chunks.dtype)
            overlap = min(chunks.shape[1], prev.shape[1])
            bw = -((chunks[:, :overlap] - prev[:overlap].unsqueeze(0)) ** 2).mean(dim=(1, 2))
        else:
            bw = torch.zeros(chunks.shape[0], device=chunks.device, dtype=chunks.dtype)

        if self._weak_mean is not None:
            weak = self._weak_mean.to(chunks.device).to(chunks.dtype)
            overlap = min(chunks.shape[1], weak.shape[0])
            fw = ((chunks[:, :overlap] - weak[:overlap].unsqueeze(0)) ** 2).mean(dim=(1, 2))
        else:
            fw = torch.zeros_like(bw)

        w_b = float(self.config.bid_backward_weight)
        w_f = float(self.config.bid_forward_weight)
        return w_b * bw + w_f * fw

    def select(
        self,
        chunks: torch.Tensor,
        aux_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the best chunk out of ``N`` candidates and update state.

        Parameters
        ----------
        chunks : torch.Tensor
            Candidate chunks ``(N, Ta, Da)``.
        aux_scores : Optional[torch.Tensor]
            Additional per-candidate scores (e.g. IQL advantage) added into
            the selection score. Shape ``(N,)``.
        """
        scores = self.score(chunks)
        if aux_scores is not None:
            scores = scores + aux_scores.to(scores.device).to(scores.dtype)
        best_idx = int(scores.argmax().item())
        best = chunks[best_idx].detach()

        batch_mean = chunks.mean(dim=0).detach()
        decay = float(self.config.bid_weak_policy_ema_decay)
        if self._weak_mean is None:
            self._weak_mean = batch_mean.clone()
        else:
            weak = self._weak_mean.to(batch_mean.device).to(batch_mean.dtype)
            self._weak_mean = decay * weak + (1.0 - decay) * batch_mean

        self._prev_chunk = best.clone()
        return best

    @property
    def prev_chunk(self) -> Optional[torch.Tensor]:
        """Most recently selected chunk, or ``None`` before the first call."""
        return self._prev_chunk

    @property
    def weak_mean(self) -> Optional[torch.Tensor]:
        """Running EMA of candidate means, or ``None`` before the first call."""
        return self._weak_mean
