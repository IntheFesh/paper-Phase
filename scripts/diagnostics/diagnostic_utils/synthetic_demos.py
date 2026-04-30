"""Synthetic demo generator.

Fallback used when the real HuggingFace dataset (``HuggingFaceVLA/libero``
and friends) is unavailable. It produces a batch of trajectories with
manually inserted phase boundaries for H1 pipeline smoke tests.

**Note**: the H1 ratio measured on synthetic data has no scientific
value; it only confirms that the script runs. Real experiments must use
a real dataset and a trained checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class DemoSample:
    """Lightweight container for a single demo.

    Attributes
    ----------
    actions : ``(T, action_dim)`` numpy array. The last dim is the gripper proxy.
    states : ``(T, state_dim)`` numpy array.
    images : ``(T, V=1, C=3, H, W)`` numpy array (float32, already in [0, 1]).
    history : ``(T, history_dim)`` numpy array.
    instruction : task instruction string (fixed for synthetic demos).
    """

    actions: np.ndarray
    states: np.ndarray
    images: np.ndarray
    history: np.ndarray
    instruction: str

    def __len__(self) -> int:
        return int(self.actions.shape[0])


def make_synthetic_demos(
    num_demos: int,
    action_dim: int = 7,
    state_dim: int = 8,
    history_dim: int = 8,
    episode_len_range: tuple = (80, 140),
    num_phases_range: tuple = (3, 5),
    image_hw: int = 64,
    seed: int = 42,
) -> List[DemoSample]:
    """Generate ``num_demos`` synthetic demos with explicit phase boundaries.

    Each demo is split into ``num_phases`` equal-length time windows.
    Within a window:

    - the first two action dims form a noisy step towards the current waypoint;
    - the last dim (gripper proxy) = ``phase_idx % 2``;
      so the gripper flips at every phase switch and ``boundary_mask``
      lines up with those timesteps.

    Parameters
    ----------
    num_demos : number of demos to generate.
    action_dim, state_dim, history_dim : must match the policy config.
    episode_len_range : demo length sampled uniformly from ``[lo, hi]``.
    num_phases_range : phase count sampled uniformly from ``[lo, hi]``.
    image_hw : image side length.
    seed : random seed.

    Returns
    -------
    List[DemoSample].
    """
    rng = np.random.default_rng(seed)
    demos: List[DemoSample] = []
    for i in range(int(num_demos)):
        T = int(rng.integers(episode_len_range[0], episode_len_range[1] + 1))
        n_phases = int(rng.integers(num_phases_range[0], num_phases_range[1] + 1))
        phase_len = max(1, T // n_phases)
        phase_ids = np.zeros(T, dtype=np.int64)
        for k in range(n_phases):
            phase_ids[k * phase_len : (k + 1) * phase_len] = k
        phase_ids[phase_len * n_phases :] = n_phases - 1

        actions = np.zeros((T, action_dim), dtype=np.float32)
        states = np.zeros((T, state_dim), dtype=np.float32)
        pos = np.zeros(2, dtype=np.float32)
        for t in range(T):
            pid = int(phase_ids[t])
            angle = 2.0 * np.pi * pid / n_phases
            target = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            direction = target - pos
            norm = float(np.linalg.norm(direction)) + 1e-6
            direction = direction / norm
            step = 0.05 * direction + rng.normal(0, 0.01, size=2).astype(np.float32)
            actions[t, 0:2] = step
            if action_dim > 3:
                actions[t, 2 : action_dim - 1] = rng.normal(0, 0.02, size=action_dim - 3).astype(np.float32)
            actions[t, -1] = float(pid % 2)
            pos = pos + step
            states[t, 0:2] = pos
            states[t, 2] = float(pid + 1) / max(1, n_phases)

        history = states.copy()
        images = np.zeros((T, 1, 3, image_hw, image_hw), dtype=np.float32)
        demos.append(
            DemoSample(
                actions=actions,
                states=states,
                images=images,
                history=history,
                instruction=f"synthetic navigation demo #{i} with {n_phases} phases",
            )
        )
    return demos


def try_load_real_demos(
    dataset_name: str,
    num_demos: int,
    action_dim: int,
    state_dim: int,
    history_dim: int,
) -> Optional[List[DemoSample]]:
    """Attempt to load real demos from HuggingFace / LeRobot; return None on failure.

    The loader is deliberately minimal: it relies on
    ``datasets.load_dataset(..., split="train")`` and groups timesteps
    by ``episode_index``. On any failure (network, auth, missing
    fields) it returns None so the caller can fall back to
    :func:`make_synthetic_demos`.

    Field mapping (probed in priority order):
        - action: ``action`` | ``actions``
        - state: ``observation.state`` | ``state`` | ``states``
        - image: ``observation.images.image`` | ``observation.image`` | ``image``
        - episode_id: ``episode_index`` | ``episode_id``
    """
    try:
        from datasets import load_dataset # type: ignore
    except Exception:
        return None
    try:
        ds = load_dataset(dataset_name, split="train")
    except Exception:
        return None

    action_key = next((k for k in ("action", "actions") if k in ds.column_names), None)
    state_key = next(
        (k for k in ("observation.state", "state", "states") if k in ds.column_names),
        None,
    )
    episode_key = next(
        (k for k in ("episode_index", "episode_id") if k in ds.column_names),
        None,
    )
    if action_key is None or state_key is None or episode_key is None:
        return None

    episodes: Dict[int, Dict[str, List[Any]]] = {}
    for idx in range(len(ds)):
        row = ds[idx]
        ep = int(row[episode_key])
        bucket = episodes.setdefault(ep, {"action": [], "state": [], "image": []})
        bucket["action"].append(np.asarray(row[action_key], dtype=np.float32))
        bucket["state"].append(np.asarray(row[state_key], dtype=np.float32))
        img = None
        for ikey in ("observation.images.image", "observation.image", "image"):
            if ikey in row:
                img = row[ikey]
                break
        bucket["image"].append(img)
        if len(episodes) >= int(num_demos) and ep not in episodes:
            break
        if len(episodes) > int(num_demos):
            break

    out: List[DemoSample] = []
    for ep_id, bucket in list(episodes.items())[: int(num_demos)]:
        actions = np.stack(bucket["action"], axis=0).astype(np.float32)
        states = np.stack(bucket["state"], axis=0).astype(np.float32)
        T = actions.shape[0]
        if actions.shape[-1] != action_dim:
            Da = action_dim
            new_actions = np.zeros((T, Da), dtype=np.float32)
            ncopy = min(actions.shape[-1], Da)
            new_actions[:, : ncopy - 1] = actions[:, : ncopy - 1]
            new_actions[:, -1] = actions[:, -1]
            actions = new_actions
        if states.shape[-1] != state_dim:
            new_states = np.zeros((T, state_dim), dtype=np.float32)
            ncopy = min(states.shape[-1], state_dim)
            new_states[:, :ncopy] = states[:, :ncopy]
            states = new_states
        history = states.copy()
        if history.shape[-1] != history_dim:
            new_hist = np.zeros((T, history_dim), dtype=np.float32)
            ncopy = min(history.shape[-1], history_dim)
            new_hist[:, :ncopy] = history[:, :ncopy]
            history = new_hist

        imgs = []
        for raw in bucket["image"]:
            if raw is None:
                imgs.append(np.zeros((1, 3, 64, 64), dtype=np.float32))
                continue
            arr = np.asarray(raw)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = arr.transpose(2, 0, 1)
            elif arr.ndim == 3 and arr.shape[0] == 3:
                pass
            else:
                arr = np.zeros((3, 64, 64), dtype=np.float32)
            if arr.max() > 1.5:
                arr = arr.astype(np.float32) / 255.0
            imgs.append(arr[None, ...].astype(np.float32))
        images = np.stack(imgs, axis=0)

        out.append(
            DemoSample(
                actions=actions,
                states=states,
                images=images,
                history=history,
                instruction=f"hf_dataset episode {ep_id}",
            )
        )
    return out if len(out) > 0 else None
