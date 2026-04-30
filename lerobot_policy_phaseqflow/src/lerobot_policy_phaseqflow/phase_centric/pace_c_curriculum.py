# ============================================================
# DEPRECATED for PACE v2 main path.
# Phase-density curriculum is moved to ablation only. PACE v2
# main path does not bias sampling toward phase boundaries during
# training — boundary signal enters via boundary-aware flow loss
# (Phase C) and concordance replanning (Phase B), not data sampling.
# Curriculum sampling also conflicts with the new DKW-guaranteed
# PCAR budget quantile, which assumes an unbiased β distribution.
# To re-enable: set cfg.use_pace_c = True (currently False).
# ============================================================
"""PACE-C: phase-density curriculum learning.

Motivation
----------
Multi-phase tasks in LIBERO-Long have wildly different boundary counts across
episodes (pick-then-place has 2 boundaries; stack-3 has 4+). Training on high
boundary-count samples from step 0 asks the model to solve imitation and
phase transitions simultaneously, slowing convergence. PACE-C treats
``boundary_count`` as a difficulty score and ramps it up over three stages:

====== ================================================== ===========================
stage  range                                               max ``boundary_count``
====== ================================================== ===========================
1      global_step < ``curriculum_stage_steps[0]``         ``curriculum_max_boundaries_stage1``
2      global_step < ``curriculum_stage_steps[1]``         ``curriculum_max_boundaries_stage2``
3      global_step < ``curriculum_stage_steps[2]``         inf (unrestricted)
3+     >= ``curriculum_stage_steps[2]``                    inf (same as stage 3)
====== ================================================== ===========================

Default ``curriculum_stage_steps=(1000, 3000, 10000)`` means:
- 0-1K steps: single-phase chunks only (boundary_count <= 1).
- 1K-3K steps: up to 3 boundaries allowed.
- 3K+ steps: everything.

Boundary-count estimate
-----------------------
Round 1's ``synthetic_demos.make_synthetic_demos`` uses ``actions[:, -1]`` as
a gripper proxy that flips at every phase change. So for one trajectory:

    boundary_count = #{t : |gripper[t] - gripper[t-1]| > 0}

:func:`compute_episode_boundaries` implements this and accepts either
``np.ndarray`` or ``torch.Tensor`` input, in shape ``(T, D)`` or
``(B, T, D)``.

API layout
----------
- :class:`PhaseDensityCurriculum`: stateful scheduler with ``step()``,
  ``current_stage()``, ``current_max_boundaries()``,
  ``should_include_episode()``.
- :func:`filter_chunks_by_boundary_count`: stateless filter returning indices
  that satisfy the current cap.
- :func:`compute_episode_boundaries`: boundary count from the gripper proxy.
- :func:`build_curriculum_filter`: walks a dataset and precomputes
  boundary_count per episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

try:
    import torch
    _TORCH_OK = True
except Exception: # pragma: no cover - torch must be present
    _TORCH_OK = False


def _to_numpy_actions(actions: Any) -> np.ndarray:
    """Coerce an action container to ``np.ndarray`` with at least 2 dims."""

    if _TORCH_OK and isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()
    arr = np.asarray(actions)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def compute_episode_boundaries(
    actions: Any,
    threshold: float = 0.5,
    gripper_dim: int = -1,
) -> int:
    """Count boundary transitions in a single episode.

    Parameters
    ----------
    actions : ``np.ndarray`` / ``torch.Tensor``
        shape ``(T, Da)``. Last dim treated as gripper proxy (consistent
        with ``make_synthetic_demos``).
    threshold : float
        Binarisation threshold on the gripper channel.
    gripper_dim : int
        Index of the gripper channel within the last dim.

    Returns
    -------
    int
        Number of ``|binarised[t] - binarised[t-1]| > 0`` transitions.
    """

    arr = _to_numpy_actions(actions)
    if arr.ndim == 3:
        return int(
            sum(
                compute_episode_boundaries(arr[b], threshold=threshold, gripper_dim=gripper_dim)
                for b in range(arr.shape[0])
            )
        )
    gripper = (arr[:, gripper_dim] > threshold).astype(np.int8)
    if gripper.shape[0] < 2:
        return 0
    return int(np.abs(np.diff(gripper)).sum())


@dataclass
class PhaseDensityCurriculum:
    """Three-stage phase-density curriculum scheduler.

    Usage::

        curriculum = PhaseDensityCurriculum(cfg)
        for step in range(total_steps):
            curriculum.step()
            max_b = curriculum.current_max_boundaries()
            valid_idx = [i for i, n in enumerate(boundary_counts)
                         if curriculum.should_include_episode(n)]
            batch = sample_from_indices(dataset, valid_idx, batch_size)

    """

    cfg: Any
    current_step: int = 0

    def __post_init__(self) -> None:
        """Cache stage-boundary tuples resolved from the config."""
        stage_steps = tuple(int(x) for x in self.cfg.curriculum_stage_steps)
        if len(stage_steps) != 3:
            raise ValueError(
                f"curriculum_stage_steps must have length 3; got {stage_steps!r}"
            )
        self._stage_steps: Sequence[int] = stage_steps
        self._max_bounds: Sequence[float] = (
            float(self.cfg.curriculum_max_boundaries_stage1),
            float(self.cfg.curriculum_max_boundaries_stage2),
            float("inf"),
        )

    def reset(self) -> None:
        """Rewind to step 0 (mostly for unit tests)."""

        self.current_step = 0

    def step(self, n: int = 1) -> None:
        """Advance the internal step counter by ``n``."""

        self.current_step += int(n)

    def current_stage(self) -> int:
        """Return 0 / 1 / 2 matching the boundary-count cap list.

        ``curriculum_stage_steps`` are absolute, monotonically increasing
        step boundaries. E.g. ``(1000, 3000, 10000)`` means:
        - step < 1000 -> stage 0
        - step < 3000 -> stage 1
        - step >= 3000 -> stage 2 (stays at stage 2 beyond stage_steps[2]).
        """

        for i, boundary in enumerate(self._stage_steps):
            if self.current_step < int(boundary):
                return i
        return len(self._stage_steps) - 1

    def current_max_boundaries(self) -> float:
        """Return the boundary-count cap for the current step."""

        return float(self._max_bounds[self.current_stage()])

    def should_include_episode(self, episode_num_boundaries: int) -> bool:
        """True iff ``episode_num_boundaries`` <= current cap."""

        return int(episode_num_boundaries) <= self.current_max_boundaries()

    def state_dict(self) -> dict:
        """Minimal serialisable state for checkpointing."""

        return {"current_step": int(self.current_step)}

    def load_state_dict(self, state: dict) -> None:
        """Restore the step counter from a saved state dict."""
        self.current_step = int(state.get("current_step", 0))

    def describe(self) -> str:
        """Return a short human-readable snapshot of the scheduler."""
        return (
            f"PhaseDensityCurriculum(step={self.current_step}, "
            f"stage={self.current_stage()}, "
            f"max_boundaries={self.current_max_boundaries()})"
        )


def build_curriculum_filter(
    cfg: Any,
    dataset: Iterable[Any],
    key: str = "action",
) -> List[int]:
    """Pre-compute boundary_count for each episode in ``dataset``.

    Expects ``dataset[i]`` to be dict-like with ``key`` pointing at ``(T, Da)``
    actions, or an object with attribute ``actions`` / ``key``.
    """

    _ = cfg
    counts: List[int] = []
    for ep in dataset:
        actions = None
        if isinstance(ep, dict):
            actions = ep.get(key)
        if actions is None:
            actions = getattr(ep, key, None)
        if actions is None:
            actions = getattr(ep, "actions", None)
        if actions is None:
            raise KeyError(
                f"episode missing {key!r} (and .actions attribute); cannot "
                f"compute boundary count"
            )
        counts.append(compute_episode_boundaries(actions))
    return counts


def filter_chunks_by_boundary_count(
    boundary_counts: Sequence[int],
    max_boundaries: float,
) -> List[int]:
    """Return indices whose boundary_count <= ``max_boundaries``.

    ``max_boundaries = float('inf')`` means "no filter"; the return order
    matches the input order so samplers can map back directly.
    """

    if max_boundaries == float("inf"):
        return list(range(len(boundary_counts)))
    cap = int(max_boundaries)
    return [i for i, n in enumerate(boundary_counts) if int(n) <= cap]
