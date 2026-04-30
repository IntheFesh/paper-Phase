"""Episode rollout + misalignment computation (H2 prior measurement).

H2 hinges on recording, during a rollout, the timesteps at which each
replan happens (``replan_times``) and the phase id at each timestep
(``phase_ids``), then computing::

    boundary_ts = [t for t in range(1, T) if phase_ids[t] != phase_ids[t-1]]
    distances = [min(|r - b| for b in boundary_ts) for r in replan_times]
    misalignment(episode) = mean(distances)

Across ``num_episodes`` trajectories we then compute
``pearsonr(misalignments, success)``. The expected result is r <= -0.5,
which says that episodes whose replans cluster near phase boundaries are
more likely to succeed.

This module does **not** use the BID / ACT state machine embedded in
``policy.select_action``; instead it drives chunk prediction and
step-by-step execution itself, strictly matching the spec pseudocode.
That way the semantics of ``replan_times`` are unambiguous (= every
call to ``predict_chunk``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch


@torch.no_grad()
def predict_chunk(policy, obs: Dict[str, Any]) -> torch.Tensor:
    """Single chunk prediction; returns a ``(Ta, Da)`` torch tensor.

    This bypasses the BID / ACT / cache-replay logic inside
    ``select_action`` and calls ``policy.forward`` once to fetch
    ``action_pred``.
        - Shortcut head -> ``(1, Ta, Da)``, squeezed to ``(Ta, Da)``.
        - Euler head -> ``(1, Da)``, unsqueezed to ``(1, Da)``.
    """
    policy.eval()
    device = next(policy.parameters()).device
    batched = policy._batchify_obs(obs)
    allowed = ("images", "states", "language", "history", "masks", "language_ids", "language_mask")
    kwargs = {k: batched[k] for k in allowed if k in batched}
    out = policy.forward(**kwargs)
    chunk = out["action_pred"]
    if chunk.ndim == 2:
        chunk = chunk.unsqueeze(1)
    return chunk.squeeze(0).detach().to(device)


def rollout_episode(
    env, # noqa: ANN001
    policy,
    max_steps: int = 200,
    gripper_dim: int = -1,
    gripper_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run a single episode and record replan + phase info.

    Parameters
    ----------
    env : a gym-like environment implementing ``reset()`` and ``step(action)``.
    policy : a PhaseQFlowPolicy instance (checkpoint already loaded).
    max_steps : maximum number of steps.
    gripper_dim : gripper dimension index in the action vector (default -1).
    gripper_threshold : binarisation threshold for the gripper.

    Returns
    -------
    dict with keys:
        - ``replan_times`` : List[int], timestep of each predict_chunk call.
        - ``phase_ids`` : List[int], binarised gripper state (0/1) per timestep.
        - ``success`` : bool.
        - ``misalignment`` : float.
        - ``total_steps`` : int.
    """
    policy.reset()
    obs = env.reset()
    buffer: List[np.ndarray] = []
    replan_times: List[int] = []
    phase_ids: List[int] = []
    success = False

    for t in range(int(max_steps)):
        if len(buffer) == 0:
            chunk = predict_chunk(policy, obs)
            buffer = [chunk[i].cpu().numpy() for i in range(chunk.shape[0])]
            replan_times.append(int(t))

        action = buffer.pop(0)
        phase_bin = int(float(action[gripper_dim]) > float(gripper_threshold))
        phase_ids.append(phase_bin)

        obs, _reward, done, info = env.step(action)
        if done:
            success = bool(info.get("success", False))
            break

    misalignment = _compute_misalignment(replan_times, phase_ids)

    return {
        "replan_times": replan_times,
        "phase_ids": phase_ids,
        "success": success,
        "misalignment": float(misalignment),
        "total_steps": int(len(phase_ids)),
    }


def _compute_misalignment(replan_times: List[int], phase_ids: List[int]) -> float:
    """Average distance from each replan to the nearest phase boundary timestep.

    Boundary defined as ``phase_ids[t] != phase_ids[t-1]``. If the whole
    episode has no boundary, return 0.0 (that episode is essentially
    single-phase; the misalignment degenerates and is unusable for the
    H2 correlation).
    """
    if len(phase_ids) < 2:
        return 0.0
    boundary_ts = [
        t for t in range(1, len(phase_ids))
        if phase_ids[t] != phase_ids[t - 1]
    ]
    if not boundary_ts or not replan_times:
        return 0.0
    dists = [min(abs(r - b) for b in boundary_ts) for r in replan_times]
    return float(np.mean(dists))


def correlate_misalignment_and_success(
    results: List[Dict[str, Any]],
) -> Tuple[float, float, float, float]:
    """Aggregate ``pearsonr(misalignments, success_flag)`` over ``results``.

    Returns
    -------
    (pearson_r, p_value, mean_mis_success, mean_mis_failure)
    """
    from scipy import stats

    mis = np.array([r["misalignment"] for r in results], dtype=np.float64)
    suc = np.array([int(r["success"]) for r in results], dtype=np.float64)
    if mis.size < 2 or (mis.std() < 1e-9) or (suc.std() < 1e-9):
        return float("nan"), float("nan"), float(np.mean(mis[suc == 1]) if (suc == 1).any() else 0.0), float(np.mean(mis[suc == 0]) if (suc == 0).any() else 0.0)
    r, p = stats.pearsonr(mis, suc)
    return (
        float(r),
        float(p),
        float(np.mean(mis[suc == 1])) if (suc == 1).any() else float("nan"),
        float(np.mean(mis[suc == 0])) if (suc == 0).any() else float("nan"),
    )
