"""ACT-style temporal ensembling (Zhao et al., arXiv 2304.13705).

Maintains a rolling buffer of recent action chunks predicted at different
execution steps. When asked for the action at the current step, it averages
(with exponentially decaying weights) all predictions that overlap the current
global step - predictions made further in the past get smaller weight
``w_i = exp(-m * i)`` where ``i`` is the chunk age in steps.

Design notes:
  - Pure numpy. The ensembler runs in the *rollout loop*, not in
    ``compute_loss``; it does not need autograd.
  - Each pushed chunk is stored together with the global step at which it
    started so that, for a given query step, we can pick out the correct
    intra-chunk offset ``q - chunk_start`` into each chunk.
  - If no chunks cover the queried step (e.g. right after ``reset``), we
    return zeros of shape ``(action_dim,)`` - the caller is expected to push
    a chunk first via ``update_and_get``.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


class ACTTemporalEnsembler:
    """Rolling-window temporal ensembler with exponential age decay.

    Parameters
    ----------
    chunk_size : int
        Length ``Ta`` of each predicted action chunk.
    decay_m : float
        Exponential decay coefficient ``m`` in ``w_i = exp(-m * i)``. Larger
        ``m`` means older chunks contribute less.
    buffer_size : int
        Maximum number of chunks to remember. Older chunks are evicted FIFO.
    action_dim : int
        Dimensionality ``Da`` of a single action.
    """

    def __init__(self, chunk_size: int, decay_m: float, buffer_size: int, action_dim: int) -> None:
        """Cache shapes and create an empty FIFO buffer."""
        self.chunk_size = int(chunk_size)
        self.decay_m = float(decay_m)
        self.buffer_size = int(buffer_size)
        self.action_dim = int(action_dim)
        self._buffer: Deque[Tuple[int, np.ndarray]] = deque(maxlen=self.buffer_size)
        self._global_step: int = 0

    def reset(self) -> None:
        """Clear all cached chunks and reset the global step counter."""
        self._buffer.clear()
        self._global_step = 0

    def _action_at(self, step: int) -> np.ndarray:
        """Weighted ensemble of all buffered chunks covering ``step``.

        Weights follow ``w_i = exp(-m * age)`` where ``age`` is the age (in
        steps) of the chunk that produced the prediction.
        """
        acc = np.zeros(self.action_dim, dtype=np.float32)
        total_w = 0.0
        for start_step, chunk in self._buffer:
            offset = step - start_step
            if 0 <= offset < self.chunk_size:
                age = float(offset)
                w = float(np.exp(-self.decay_m * age))
                acc += w * chunk[offset].astype(np.float32)
                total_w += w
        if total_w <= 0.0:
            return acc
        return acc / total_w

    def update_and_get(self, new_chunk: np.ndarray) -> np.ndarray:
        """Push ``new_chunk`` starting at the current global step, return action.

        Returns the ensembled action for the current step and advances the
        internal step counter by one so the next call corresponds to the
        next timestep.
        """
        new_chunk = np.asarray(new_chunk, dtype=np.float32)
        if new_chunk.ndim != 2 or new_chunk.shape != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"new_chunk must be shape ({self.chunk_size}, {self.action_dim}); "
                f"got {tuple(new_chunk.shape)}"
            )
        self._buffer.append((self._global_step, new_chunk))
        action = self._action_at(self._global_step)
        self._global_step += 1
        return action

    def peek(self, step: Optional[int] = None) -> np.ndarray:
        """Return the ensembled action at ``step`` (default: current step) without advancing."""
        if step is None:
            step = self._global_step
        return self._action_at(int(step))

    @property
    def current_step(self) -> int:
        """Current global step counter value."""
        return self._global_step
