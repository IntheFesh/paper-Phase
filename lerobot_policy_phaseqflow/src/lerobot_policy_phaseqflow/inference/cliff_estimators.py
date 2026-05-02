"""
三个 Predictability Cliff 估计器。

I^(1) - Phase Posterior Bhattacharyya（已在 HierarchicalPlanner 内部计算为 beta_t）
         这里提供独立接口便于外部访问；policy 内部已有，无需重复计算。

I^(2) - Policy Sample Variance
         对 N 个 chunk 候选计算第一步动作的方差；
         方差大 ⟺ 动作分布宽 ⟺ 当前时刻接近 cliff。

I^(3) - Velocity Field Curvature
         flow matching velocity field 对条件 c_t 的有限差分灵敏度；
         灵敏度大 ⟺ 动作 ODE 发生急剧调整 ⟺ 当前时刻接近 cliff。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# I^(2)：Policy Sample Variance
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_policy_variance(
    policy,
    obs_dict: dict,
    n_samples: int = 8,
    tau_sample: float = 0.5,
) -> float:
    """计算 N 个 chunk 候选中第一步动作的平均方差（I^(2) 估计器）。

    Parameters
    ----------
    policy      : PhaseQFlowPolicy（已 .eval()）
    obs_dict    : 与 select_action 相同格式的观测字典
    n_samples   : 采样 chunk 数量，默认 8（计算成本低）
    tau_sample  : flow integration 起点 τ，默认 0.5（中间截面方差更稳定）

    Returns
    -------
    sigma2 : float  第一步动作方差的迹（trace of covariance）；越大越像 cliff
    """
    actions = []
    for _ in range(n_samples):
        try:
            act = policy.select_action(obs_dict)        # shape: (Da,) or (Ta, Da)
            act_tensor = act if isinstance(act, torch.Tensor) else torch.tensor(act)
            if act_tensor.dim() > 1:
                act_tensor = act_tensor[0]               # 取第一步
            actions.append(act_tensor.float().cpu())
        except Exception:
            continue

    if len(actions) < 2:
        return 0.0

    stacked = torch.stack(actions, dim=0)               # (N, Da)
    # 方差的迹 = Σ_d Var[a_d]
    sigma2 = stacked.var(dim=0).sum().item()
    return sigma2


# ---------------------------------------------------------------------------
# I^(3)：Velocity Field Curvature（有限差分）
# ---------------------------------------------------------------------------

def compute_velocity_curvature(
    policy,
    obs_dict: dict,
    prev_condition: Optional[torch.Tensor],
    tau: float = 0.5,
) -> "tuple[float, torch.Tensor]":
    """计算 velocity field 对条件 c_t 的有限差分灵敏度（I^(3) 估计器）。

    需要 policy 支持 `get_condition(obs_dict)` 接口（见 §7-A），
    且 policy.flow_action_head 支持 `velocity(x, tau, cond)` 方法。

    Parameters
    ----------
    policy          : PhaseQFlowPolicy
    obs_dict        : 当前时刻观测字典
    prev_condition  : 上一时刻的条件向量 c_{t-1}（首步传 None）
    tau             : flow integration 时刻，默认 0.5

    Returns
    -------
    curvature : float           ||v(c_t) - v(c_{t-1})||₂；越大越像 cliff
    condition : torch.Tensor    当前 c_t，供下一步传入 prev_condition
    """
    try:
        c_t = policy.get_condition(obs_dict)            # (1, D_c)
    except AttributeError:
        return 0.0, None

    if prev_condition is None:
        return 0.0, c_t.detach()

    try:
        device = c_t.device
        Ta = int(policy.config.action_chunk_size)
        Da = int(policy.config.action_dim)
        x_noise = torch.randn(1, Ta, Da, device=device)

        with torch.no_grad():
            v_t   = policy.flow_action_head.velocity(x_noise, tau, c_t)
            v_tm1 = policy.flow_action_head.velocity(x_noise, tau, prev_condition.to(device))

        curvature = (v_t - v_tm1).norm(dim=-1).mean().item()
    except Exception:
        curvature = 0.0

    return curvature, c_t.detach()
