"""SyntheticLongHorizonEnv.

Fallback for H2 when the LIBERO environment is not available on the
current machine (almost always true in Round 1, which only aims to get
the pipeline running in a synthetic environment). Defined here is a 2D
navigation task.

Task description
----------------
The agent is a point mass on a 2D plane and must visit three waypoints
in order: ``W0 = (-1, 0), W1 = (+1, 0), W2 = (0, +1)`` (defaults).

State space (state)
    ``(x, y, current_wp_idx / num_wp)`` (3-D, normalised to roughly ``[-1, 1]``).

Action space (action)
    ``Da`` dims (default 7, matching LIBERO). Only the first two dims
    ``(dx, dy)`` are used by the env; the **last dim
    ``gripper = current_wp_idx % 2``** serves as the ground-truth phase
    signal — the gripper flips whenever a waypoint switches (phase
    boundary), which matches the ``gripper`` proxy assumption exactly.
    The remaining intermediate dims are ignored.

Success condition
    All three waypoints visited in order (each within
    ``waypoint_radius``) and total steps <= ``max_steps``.

Return contract (aligned with the classic Gym API)
    ``reset() -> obs_dict``;
    ``step(action: np.ndarray) -> (obs_dict, reward, done, info)``.

The ``obs_dict`` structure matches the fields that
``PhaseQFlowPolicy.select_action`` expects:

    {
        "images": torch.Tensor (V=1, C=3, H=64, W=64),
        "states": torch.Tensor (state_dim,),
        "language": torch.Tensor (L,),
        "history": torch.Tensor (history_dim,),
        "masks": torch.Tensor (1,),
    }

Images are currently all zeros — the synthetic env has no vision, but a
placeholder tensor is required so the tokenizer still forwards cleanly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch


class SyntheticLongHorizonEnv:
    """3-waypoint 2D navigation env with explicit phase-boundary ground truth.

    Parameters
    ----------
    action_dim : action dimension (must match the policy's ``action_dim``; default 7).
    state_dim : state dimension (fed to the ``vision_tokenizer`` states input; default 8).
    history_dim : history-state dimension (default 8).
    waypoint_radius : success-check radius at each waypoint.
    max_steps : per-episode step cap.
    step_size : per-step ``(dx, dy)`` multiplier so that one synthetic episode spans dozens of steps.
    success_noise : observation noise std, simulating real-env stochasticity.
    seed : random seed used per episode.
    """

    def __init__(
        self,
        action_dim: int = 7,
        state_dim: int = 8,
        history_dim: int = 8,
        waypoint_radius: float = 0.2,
        max_steps: int = 200,
        step_size: float = 0.05,
        success_noise: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.history_dim = int(history_dim)
        self.waypoint_radius = float(waypoint_radius)
        self.max_steps = int(max_steps)
        self.step_size = float(step_size)
        self.success_noise = float(success_noise)
        self._rng = np.random.default_rng(seed)

        self.waypoints = np.array(
            [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        )
        self._pos = np.zeros(2, dtype=np.float32)
        self._wp_idx = 0
        self._step = 0

    def reset(self, start_pos: np.ndarray | None = None) -> Dict[str, Any]:
        """Start a new episode and return the initial obs."""
        if start_pos is None:
            self._pos = self._rng.uniform(-0.1, 0.1, size=2).astype(np.float32)
        else:
            self._pos = np.asarray(start_pos, dtype=np.float32)
        self._wp_idx = 0
        self._step = 0
        return self._make_obs()

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Advance one step. ``action`` may be a numpy array or torch tensor."""
        a = action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
        a = a.flatten().astype(np.float32)
        if a.size < 2:
            raise ValueError("SyntheticLongHorizonEnv requires at least a 2-D action")
        dx, dy = float(a[0]), float(a[1])
        dx = float(np.clip(dx, -1.0, 1.0))
        dy = float(np.clip(dy, -1.0, 1.0))
        self._pos = self._pos + self.step_size * np.array([dx, dy], dtype=np.float32)
        self._pos = self._pos + self._rng.normal(0.0, self.success_noise, size=2).astype(np.float32)
        self._step += 1

        reward = 0.0
        info: Dict[str, Any] = {"success": False, "wp_idx": int(self._wp_idx)}
        if self._wp_idx < len(self.waypoints):
            target = self.waypoints[self._wp_idx]
            if float(np.linalg.norm(self._pos - target)) < self.waypoint_radius:
                self._wp_idx += 1
                reward = 0.5
                info["wp_boundary"] = True
        done = (self._wp_idx >= len(self.waypoints)) or (self._step >= self.max_steps)
        if self._wp_idx >= len(self.waypoints):
            info["success"] = True
            reward = 1.0
        return self._make_obs(), reward, done, info

    def _make_obs(self) -> Dict[str, Any]:
        """Build an obs dict compatible with the PhaseQFlowPolicy input contract."""
        progress = float(self._wp_idx) / max(1.0, len(self.waypoints))
        raw_state = np.concatenate(
            [self._pos, np.array([progress], dtype=np.float32)], axis=0
        )
        if raw_state.size < self.state_dim:
            pad = np.zeros(self.state_dim - raw_state.size, dtype=np.float32)
            state = np.concatenate([raw_state, pad], axis=0)
        else:
            state = raw_state[: self.state_dim]
        history = state.copy()
        if history.size < self.history_dim:
            history = np.concatenate(
                [history, np.zeros(self.history_dim - history.size, dtype=np.float32)],
                axis=0,
            )
        else:
            history = history[: self.history_dim]
        return {
            "images": torch.zeros(1, 3, 64, 64, dtype=torch.float32),
            "states": torch.from_numpy(state),
            "language": torch.zeros(16, dtype=torch.float32),
            "history": torch.from_numpy(history),
            "masks": torch.ones(1, dtype=torch.float32),
        }

    @property
    def num_waypoints(self) -> int:
        """Number of waypoints in this env."""
        return int(len(self.waypoints))


def make_fallback_env(
    action_dim: int = 7,
    state_dim: int = 8,
    history_dim: int = 8,
    seed: int = 0,
) -> SyntheticLongHorizonEnv:
    """Convenience factory for :class:`SyntheticLongHorizonEnv`."""
    return SyntheticLongHorizonEnv(
        action_dim=action_dim,
        state_dim=state_dim,
        history_dim=history_dim,
        seed=seed,
    )
