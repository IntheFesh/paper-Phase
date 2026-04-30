"""Abstract base class for cross-policy universality adapters.

Each concrete adapter wraps one VLA policy (OpenVLA, π0, BC-ACT, Diffusion
Policy) behind a uniform interface so that ``scripts/phenomenon/universality.py``
can evaluate the Predictability Cliff phenomenon without policy-specific logic
leaking into the experiment script.

Interface contract
------------------
- :meth:`load` must be called once before :meth:`rollout`.
- :meth:`is_available` returns False when required libraries / checkpoints are
  missing.  The universality script logs any unavailable adapter to
  MIGRATION_NOTES and skips it.
- :meth:`rollout` returns a :class:`RolloutResult` dict.
- :meth:`cliff_steps_from_actions` provides a universal action-change proxy
  that subclasses may override with policy-specific estimators.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class RolloutResult:
    """Data returned by a single episode rollout.

    Attributes
    ----------
    trajectory_len : int
        Total number of timesteps executed.
    success : bool
        Whether the episode ended in task success.
    failure_step : Optional[int]
        Step index of first failure event (gripper drop, timeout, etc.).
        ``None`` for successful episodes.
    cliff_steps : List[int]
        Timesteps where a cliff was detected (by the adapter's cliff proxy).
    action_seq : List[Any]
        Raw per-step actions; used by universal cliff proxy if cliff_steps empty.
    extra : Dict[str, Any]
        Adapter-specific diagnostics (phase logits, concordance values, etc.).
    """

    trajectory_len: int
    success: bool
    failure_step: Optional[int]
    cliff_steps: List[int] = field(default_factory=list)
    action_seq: List[Any] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def failure_distance(self) -> Optional[int]:
        """Distance (steps) from last cliff before failure to the failure step.

        Returns ``None`` when the episode succeeded or no cliff preceded failure.
        """
        if self.success or self.failure_step is None:
            return None
        preceding = [s for s in self.cliff_steps if s < self.failure_step]
        if not preceding:
            return None
        return self.failure_step - preceding[-1]


class PolicyAdapter(ABC):
    """Uniform interface for a VLA policy in the universality experiment.

    Parameters
    ----------
    name : str
        Human-readable policy identifier (e.g. ``"openvla"``).
    checkpoint_path : str, optional
        Path or HuggingFace model-id for the policy checkpoint.
    """

    def __init__(self, name: str, checkpoint_path: Optional[str] = None) -> None:
        self.name = name
        self.checkpoint_path = checkpoint_path
        self._loaded = False

    # ------------------------------------------------------------------
    # Subclass must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """Return True iff required libraries and checkpoint are accessible."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory (called once before :meth:`rollout`)."""

    @abstractmethod
    def rollout(self, env: Any, n_steps: int, seed: int = 0) -> RolloutResult:
        """Execute one episode and return a :class:`RolloutResult`."""

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def cliff_steps_from_actions(
        self,
        actions: Sequence[Any],
        *,
        percentile: float = 90.0,
    ) -> List[int]:
        """Universal action-change proxy for cliff detection.

        Detects timesteps where ``||a_t - a_{t-1}||_2`` exceeds the
        ``percentile``-th percentile of the full-episode distribution.
        This is a fallback proxy; adapters with richer signals should
        override :meth:`rollout` to fill ``cliff_steps`` directly.

        Parameters
        ----------
        actions : sequence of array-like, length T
        percentile : float
            Percentile threshold for spike detection.
        """
        try:
            import numpy as np
        except ImportError:
            return []

        actions_np = [np.asarray(a).ravel() for a in actions]
        if len(actions_np) < 2:
            return []
        diffs = [
            float(np.linalg.norm(actions_np[t] - actions_np[t - 1]))
            for t in range(1, len(actions_np))
        ]
        threshold = float(np.percentile(diffs, percentile))
        return [t + 1 for t, d in enumerate(diffs) if d >= threshold]
