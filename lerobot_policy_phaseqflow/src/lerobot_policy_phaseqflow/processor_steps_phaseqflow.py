"""Online helpers for skill-phase/value-guided inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np


def compute_skill_id_from_logits(skill_logits: np.ndarray) -> int:
    """Return discrete skill token id from 1D logits/probabilities."""
    logits = np.asarray(skill_logits, dtype=float)
    if logits.ndim != 1:
        raise ValueError("skill_logits must be a 1D array")
    return int(np.argmax(logits))


def compute_value_weight(q_value: float, beta: float = 2.0, min_weight: float = 1e-3) -> float:
    """Convert critic value into a positive exponential sample weight."""
    weight = float(np.exp(beta * float(q_value)))
    return float(max(weight, min_weight))


@dataclass
class OnlineSkillState:
    """Bounded online state with incremental normalization statistics."""

    num_skills: int = 16
    beta: float = 2.0
    action_buffer_maxlen: int = 128
    _action_buffer: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=128))
    _weight_buffer: Deque[float] = field(default_factory=lambda: deque(maxlen=128))
    _weight_sum: float = 0.0

    def __post_init__(self) -> None:
        """Reset bounded buffers after dataclass initialization."""
        self._action_buffer = deque(maxlen=self.action_buffer_maxlen)
        self._weight_buffer = deque(maxlen=self.action_buffer_maxlen)
        self._weight_sum = 0.0

    def reset(self) -> None:
        """Clear all online statistics and buffered actions."""
        self._action_buffer.clear()
        self._weight_buffer.clear()
        self._weight_sum = 0.0

    def _append_weight(self, weight: float) -> None:
        """Push one weight into the running buffer and maintain its sum."""
        if len(self._weight_buffer) == self._weight_buffer.maxlen:
            oldest = self._weight_buffer[0]
            self._weight_sum -= oldest
        self._weight_buffer.append(weight)
        self._weight_sum += weight

    def step(self, action_t: np.ndarray, skill_logits_t: np.ndarray, q_value_t: float) -> tuple[int, float]:
        """Process one timestep and return `(skill_id, normalized_weight)`."""
        self._action_buffer.append(np.asarray(action_t, dtype=float))

        skill_id = compute_skill_id_from_logits(skill_logits_t)
        raw_weight = compute_value_weight(q_value_t, beta=self.beta)
        self._append_weight(raw_weight)

        avg_weight = self._weight_sum / max(len(self._weight_buffer), 1)
        normalized_weight = raw_weight / max(avg_weight, 1e-6)
        return skill_id, float(normalized_weight)
