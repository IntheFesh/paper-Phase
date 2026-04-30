"""Unit tests for the PACE-C phase-density curriculum and boundary counting."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_SRC = _REPO_ROOT / "lerobot_policy_phaseqflow" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from lerobot_policy_phaseqflow.phase_centric.pace_c_curriculum import ( # noqa: E402
    PhaseDensityCurriculum,
    build_curriculum_filter,
    compute_episode_boundaries,
    filter_chunks_by_boundary_count,
)


@dataclass
class _Cfg:
    """Minimal config stand-in carrying the curriculum knobs."""

    curriculum_stage_steps: tuple = (1000, 3000, 10000)
    curriculum_max_boundaries_stage1: int = 1
    curriculum_max_boundaries_stage2: int = 3


def _trajectory_with_boundaries(num_boundaries: int, T: int = 40, Da: int = 4) -> np.ndarray:
    """Build a (T, Da) array whose last dim toggles exactly ``num_boundaries`` times.

    Boundaries are placed at evenly spaced interior positions; the gripper
    channel is initialised to 0 and flipped at each boundary.
    """

    gripper = np.zeros(T, dtype=np.float32)
    state = 0.0
    if num_boundaries > 0:
        spacing = T // (num_boundaries + 1)
        flip_points = [min(T - 1, max(1, spacing * (i + 1))) for i in range(num_boundaries)]
        for flip in flip_points:
            state = 1.0 - state
            gripper[flip:] = state
    arr = np.random.randn(T, Da).astype(np.float32)
    arr[:, -1] = gripper
    return arr


def test_compute_boundaries_np() -> None:
    """Counting on a NumPy trajectory returns the expected number of flips."""
    arr = _trajectory_with_boundaries(num_boundaries=2)
    n = compute_episode_boundaries(arr)
    assert n == 2


def test_compute_boundaries_torch() -> None:
    """A ``torch.Tensor`` input yields the same count as its NumPy equivalent."""
    arr = _trajectory_with_boundaries(num_boundaries=3)
    tensor = torch.from_numpy(arr)
    n_np = compute_episode_boundaries(arr)
    n_torch = compute_episode_boundaries(tensor)
    assert n_np == n_torch == 3


def test_compute_boundaries_batch() -> None:
    """A batched ``(B, T, Da)`` input aggregates boundary counts across the batch."""
    a = _trajectory_with_boundaries(num_boundaries=1)
    b = _trajectory_with_boundaries(num_boundaries=4)
    batch = np.stack([a, b], axis=0)
    n = compute_episode_boundaries(batch)
    assert n == 5


def test_curriculum_stage_progression() -> None:
    """``current_stage`` advances as the accumulated step count crosses each threshold."""
    curr = PhaseDensityCurriculum(cfg=_Cfg())
    assert curr.current_stage() == 0
    assert curr.current_max_boundaries() == 1.0
    curr.step(999)
    assert curr.current_stage() == 0
    curr.step(1)
    assert curr.current_stage() == 1
    assert curr.current_max_boundaries() == 3.0
    curr.step(2000)
    assert curr.current_stage() == 2
    assert curr.current_max_boundaries() == float("inf")


def test_curriculum_stage_post_end() -> None:
    """Stepping far past the final stage keeps the curriculum at stage 2."""
    curr = PhaseDensityCurriculum(cfg=_Cfg())
    curr.step(100_000)
    assert curr.current_stage() == 2
    assert curr.current_max_boundaries() == float("inf")


def test_should_include_episode_filters() -> None:
    """``should_include_episode`` honours the per-stage boundary cap."""
    curr = PhaseDensityCurriculum(cfg=_Cfg())
    assert curr.should_include_episode(0)
    assert curr.should_include_episode(1)
    assert not curr.should_include_episode(2)
    curr.step(1000)
    assert curr.should_include_episode(3)
    assert not curr.should_include_episode(4)


def test_filter_chunks_by_boundary_count() -> None:
    """``filter_chunks_by_boundary_count`` returns indices whose count is under the cap."""
    counts = [0, 1, 2, 3, 4, 0]
    assert filter_chunks_by_boundary_count(counts, max_boundaries=1) == [0, 1, 5]
    assert filter_chunks_by_boundary_count(counts, max_boundaries=3) == [0, 1, 2, 3, 5]


def test_filter_inf_cap_returns_all() -> None:
    """A cap of ``inf`` accepts every index."""
    counts = [0, 5, 10]
    assert filter_chunks_by_boundary_count(counts, max_boundaries=float("inf")) == [0, 1, 2]


def test_state_dict_roundtrip() -> None:
    """``state_dict``/``load_state_dict`` preserve the current step."""
    curr = PhaseDensityCurriculum(cfg=_Cfg())
    curr.step(777)
    saved = curr.state_dict()
    curr2 = PhaseDensityCurriculum(cfg=_Cfg())
    curr2.load_state_dict(saved)
    assert curr2.current_step == 777
    assert curr2.current_stage() == 0


def test_build_curriculum_filter_dicts() -> None:
    """``build_curriculum_filter`` precomputes per-episode boundary counts for a dict dataset."""
    a = _trajectory_with_boundaries(num_boundaries=1)
    b = _trajectory_with_boundaries(num_boundaries=3)
    ds = [{"action": a}, {"action": b}]
    counts = build_curriculum_filter(cfg=_Cfg(), dataset=ds, key="action")
    assert counts == [1, 3]


def test_invalid_stage_steps_length() -> None:
    """A ``curriculum_stage_steps`` tuple without three entries raises ``ValueError``."""
    @dataclass
    class _Bad:
        curriculum_stage_steps: tuple = (100, 200)
        curriculum_max_boundaries_stage1: int = 1
        curriculum_max_boundaries_stage2: int = 3

    with pytest.raises(ValueError):
        PhaseDensityCurriculum(cfg=_Bad())
