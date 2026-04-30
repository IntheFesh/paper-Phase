"""Concordance detector C_t — rank-based fusion of the three cliff estimators.

C_t = (1/3) * [rank_W(I^(1)) + rank_W(I^(2)) + rank_W(I^(3))]

rank_W is the cliff-oriented percentile rank within a rolling window of size W:
  rank_W(v) = 1 - fraction_of_window_values_strictly_below(v)

so a LOW estimator value (cliff) maps to a HIGH rank, and C_t is HIGH when all
three estimators agree a cliff is imminent.  Triggers when C_t >= threshold.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ConcordanceDetector(nn.Module):
    """Rank-based fusion concordance detector for the three cliff estimators.

    Parameters
    ----------
    window_size : int
        Rolling history length W for percentile ranking. Default 50.
    threshold : float
        C_t value in (0, 1] above which a cliff trigger fires. Default 0.8.
    warmup_steps : int, optional
        Number of steps before any trigger fires (history warm-up).
        Defaults to ``window_size``.
    """

    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 0.8,
        warmup_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2; got {window_size}")
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1]; got {threshold}")
        self.window_size = window_size
        self.threshold = threshold
        self.warmup_steps: int = warmup_steps if warmup_steps is not None else window_size

        self._history: Deque[Tuple[float, float, float]] = deque(maxlen=window_size)
        self._step_count: int = 0

    def reset(self) -> None:
        """Clear rolling history; call at episode start."""
        self._history.clear()
        self._step_count = 0

    @staticmethod
    def _cliff_rank(value: float, window: List[float]) -> float:
        """Cliff-oriented percentile rank using mid-rank convention for ties.

        rank = 1 - (count_strictly_below + 0.5 * count_equal) / len(window).
        Low estimator value (cliff) → high rank.  Empty window → 0.5.
        Ties map to 0.5 (neutral), preventing spurious cliffs on stable signals.
        """
        if not window:
            return 0.5
        n = len(window)
        below = sum(1 for v in window if v < value)
        equal = sum(1 for v in window if v == value)
        return 1.0 - (below + 0.5 * equal) / n

    def step(
        self,
        i_hat_1: Optional[float],
        i_hat_2: Optional[float],
        i_hat_3: Optional[float],
    ) -> Dict[str, object]:
        """Update the detector with scalar cliff-estimator values.

        Parameters
        ----------
        i_hat_1, i_hat_2, i_hat_3 : float or None
            Scalar cliff estimator values.  Pass ``None`` for any estimator
            that is unavailable (e.g. I^(3) at step 0); that estimator's rank
            contribution is replaced by 0.5 (neutral).

        Returns
        -------
        dict
            ``triggered``  : bool — True when C_t >= threshold after warmup.
            ``concordance``: float — C_t in [0, 1].
            ``ranks``      : tuple of three floats — individual cliff ranks.
        """
        w1: List[float] = [h[0] for h in self._history]
        w2: List[float] = [h[1] for h in self._history]
        w3: List[float] = [h[2] for h in self._history]

        r1 = self._cliff_rank(i_hat_1, w1) if i_hat_1 is not None else 0.5
        r2 = self._cliff_rank(i_hat_2, w2) if i_hat_2 is not None else 0.5
        r3 = self._cliff_rank(i_hat_3, w3) if i_hat_3 is not None else 0.5

        hist_1 = i_hat_1 if i_hat_1 is not None else 0.0
        hist_2 = i_hat_2 if i_hat_2 is not None else 0.0
        hist_3 = i_hat_3 if i_hat_3 is not None else 0.0
        self._history.append((hist_1, hist_2, hist_3))
        self._step_count += 1

        concordance = (r1 + r2 + r3) / 3.0
        triggered = (self._step_count > self.warmup_steps) and (
            concordance >= self.threshold
        )

        return {
            "triggered": triggered,
            "concordance": concordance,
            "ranks": (r1, r2, r3),
        }


__all__ = ["ConcordanceDetector"]
