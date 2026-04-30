"""Phase-boundary proxy signals.

Because LIBERO ships no official sub-phase labels, Round 1 adopts three
unsupervised proxies to mark "phase-boundary timesteps":

1. ``gripper`` — gripper open/close state jumps. We binarise the last
   dimension (gripper) of LIBERO's 7-D action; a boundary occurs where
   adjacent timesteps flip binary state. Widely used in the robot
   learning community; almost every LIBERO phase coincides with a
   gripper switch.
2. ``velocity_change`` — large change in end-effector velocity direction
   (cosine similarity < 0.3). Useful for sub-tasks where the gripper
   barely moves.
3. ``planner_output`` — calls ``policy.hierarchical_planner.forward``
   for a discrete ``phase_id``; a boundary is a change of phase_id
   between adjacent timesteps. Falls back to ``gripper`` if the policy
   does not expose this interface.

Each proxy returns a 1-D bool numpy array ``boundary_mask[t]`` where
``boundary_mask[t] = True`` means ``t`` is a boundary timestep or lies
within the +/-2 neighbourhood of one (after
``scipy.ndimage.binary_dilation(iterations=2)``-style dilation).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _dilate(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Pure-NumPy binary dilation (avoids a hard dependency on scipy).

    Parameters
    ----------
    mask : 1-D bool array.
    iterations : dilation radius. Each True position is expanded to a
        ``+/-iterations`` neighbourhood.

    Returns
    -------
    1-D bool array the same length as ``mask``.
    """
    if iterations <= 0 or mask.size == 0:
        return mask.astype(bool)
    out = mask.astype(bool).copy()
    n = out.size
    for _ in range(iterations):
        shifted_left = np.concatenate([[False], out[:-1]])
        shifted_right = np.concatenate([out[1:], [False]])
        out = out | shifted_left | shifted_right
    return out


def gripper_boundary_mask(
    actions: np.ndarray,
    gripper_dim: int = -1,
    threshold: float = 0.5,
    dilate: int = 2,
) -> np.ndarray:
    """Boundary mask from gripper-action jumps (the default proxy).

    Parameters
    ----------
    actions : ``(T, Da)`` or ``(T,)`` action array.
    gripper_dim : index of the gripper dimension (default -1, last dim).
    threshold : binarisation threshold (> threshold means close).
    dilate : dilation radius (the +/-dilate neighbourhood also counts).

    Returns
    -------
    ``(T,)`` bool numpy array.
    """
    arr = np.asarray(actions)
    if arr.ndim == 1:
        gripper = arr.astype(np.float32)
    else:
        gripper = arr[:, gripper_dim].astype(np.float32)
    gripper_bin = (gripper > threshold).astype(np.int32)
    if gripper_bin.size == 0:
        return np.zeros(0, dtype=bool)
    delta = np.abs(np.diff(gripper_bin, prepend=gripper_bin[0]))
    boundary = (delta > 0)
    return _dilate(boundary, iterations=int(dilate))


def velocity_change_boundary_mask(
    ee_positions: np.ndarray,
    cosine_threshold: float = 0.3,
    dilate: int = 2,
) -> np.ndarray:
    """Boundary mask from large swings in end-effector velocity direction.

    Steps:
        1. velocity ``v[t] = pos[t+1] - pos[t]``, shape ``(T-1, D)``.
        2. cosine similarity ``c[t]`` between adjacent velocities;
           ``c < cosine_threshold`` marks a boundary.
        3. dilate by ``dilate``.

    Parameters
    ----------
    ee_positions : ``(T, D)`` end-effector trajectory (usually D=3 XYZ).
    cosine_threshold : similarity below which is a direction jump.
    dilate : dilation radius.

    Returns
    -------
    ``(T,)`` bool numpy array.
    """
    arr = np.asarray(ee_positions, dtype=np.float32)
    T = arr.shape[0]
    if T < 3:
        return np.zeros(T, dtype=bool)
    v = np.diff(arr, axis=0)
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
    v_hat = v / n
    cos = np.einsum("td,td->t", v_hat[:-1], v_hat[1:])
    boundary_inner = cos < cosine_threshold
    boundary = np.concatenate([[False], boundary_inner, [False]])
    return _dilate(boundary, iterations=int(dilate))


def planner_output_boundary_mask(
    policy, # noqa: ANN001
    fused_obs_seq, # noqa: ANN001
    dilate: int = 2,
    fallback_actions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boundary mask from the HierarchicalPlanner's phase_id output.

    Calls ``policy.hierarchical_planner(fused_obs=...)["phase_id"]`` and
    checks whether the discrete phase_id changes between adjacent
    timesteps. Falls back to the gripper proxy (which needs
    ``fallback_actions``) when the policy does not expose the interface
    or the call fails.

    Parameters
    ----------
    policy : PhaseQFlowPolicy instance.
    fused_obs_seq : ``(T, fusion_hidden_dim)`` torch.Tensor of already-fused observation features.
    dilate : dilation radius.
    fallback_actions : ``(T, Da)`` actions needed by the gripper proxy fallback.

    Returns
    -------
    ``(T,)`` bool numpy array.
    """
    try:
        import torch
    except Exception: # noqa: BLE001
        if fallback_actions is None:
            raise RuntimeError("planner_output proxy unavailable and no fallback_actions provided")
        return gripper_boundary_mask(fallback_actions, dilate=dilate)

    try:
        planner = getattr(policy, "hierarchical_planner", None)
        if planner is None:
            raise AttributeError("policy has no hierarchical_planner")
        with torch.no_grad():
            plan = planner(fused_obs=fused_obs_seq, phase_labels=None)
        phase_ids = plan.get("phase_id")
        if phase_ids is None:
            raise KeyError("planner output missing phase_id")
        phase_np = phase_ids.detach().cpu().numpy().astype(np.int64)
    except Exception as exc: # noqa: BLE001
        if fallback_actions is None:
            raise RuntimeError(f"planner_output proxy failed ({exc!r}) and no fallback_actions provided")
        return gripper_boundary_mask(fallback_actions, dilate=dilate)

    if phase_np.size == 0:
        return np.zeros(0, dtype=bool)
    delta = np.abs(np.diff(phase_np, prepend=phase_np[0]))
    boundary = (delta > 0)
    return _dilate(boundary, iterations=int(dilate))


def compute_boundary_mask(
    proxy: str,
    actions: Optional[np.ndarray] = None,
    ee_positions: Optional[np.ndarray] = None,
    policy=None, # noqa: ANN001
    fused_obs_seq=None, # noqa: ANN001
    dilate: int = 2,
) -> np.ndarray:
    """Unified dispatcher for the three proxies.

    Parameters
    ----------
    proxy : ``gripper`` / ``velocity_change`` / ``planner_output``.
    actions : default fallback data; required by ``gripper``.
    ee_positions : required by ``velocity_change``.
    policy, fused_obs_seq : required by ``planner_output``.
    dilate : dilation radius.

    Returns
    -------
    ``(T,)`` bool numpy array.
    """
    p = proxy.lower()
    if p == "gripper":
        if actions is None:
            raise ValueError("gripper proxy requires actions")
        return gripper_boundary_mask(actions, dilate=dilate)
    if p == "velocity_change":
        if ee_positions is None:
            raise ValueError("velocity_change proxy requires ee_positions")
        return velocity_change_boundary_mask(ee_positions, dilate=dilate)
    if p == "planner_output":
        return planner_output_boundary_mask(
            policy=policy,
            fused_obs_seq=fused_obs_seq,
            dilate=dilate,
            fallback_actions=actions,
        )
    raise ValueError(f"unknown phase_proxy={proxy!r}; expected gripper|velocity_change|planner_output")
