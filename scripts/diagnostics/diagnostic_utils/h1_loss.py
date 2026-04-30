"""Per-timestep flow-matching loss extractor (H1 prior measurement).

Goal: for a trained ``PhaseQFlowPolicy``, compute the flow-matching loss
``L_FM[t]`` at every demo timestep ``t``, then split by a phase-boundary mask:

    boundary_loss = mean(L_FM[t] for t in boundary)
    interior_loss = mean(L_FM[t] for t in interior)
    ratio = boundary_loss / interior_loss

Requirements (aligned with the upstream spec):
    1. Do **not** modify ``modeling_phaseqflow.py`` / ``configuration_phaseqflow.py``.
    2. Do **not** go through data augmentation — build the batch directly
       from raw demo fields.
    3. Use option 2C batched-over-t inference: pack all timesteps of a
       demo into a ``batch_size=T`` tensor and forward in one shot.
    4. For the Shortcut flow action head, the FM loss is a function of
       randomly sampled ``(t, d, noise)``; by default ``num_samples=3``
       independent resamples are averaged to reduce variance.
    5. For the Euler flow action head, use MSE(action_pred, actions_gt)
       as a proxy FM loss.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .synthetic_demos import DemoSample


def _build_demo_batch(
    demo: DemoSample,
    chunk_len: int,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Pack a single demo into a ``batch_size=T'`` tensor dictionary.

    Here ``T' = T - chunk_len + 1`` (option 1A: real-chunk supervision,
    trailing tail discarded). Each virtual sample ``i`` corresponds to
    demo timestep ``i`` and is supervised by the action chunk
    ``actions[i : i + chunk_len]``.

    Returns
    -------
    ``None`` when ``T < chunk_len``.
    """
    T = len(demo)
    if T < chunk_len:
        return None
    T_eff = T - chunk_len + 1

    images = torch.from_numpy(demo.images[:T_eff]).to(device=device, dtype=torch.float32)
    states = torch.from_numpy(demo.states[:T_eff]).to(device=device, dtype=torch.float32)
    history = torch.from_numpy(demo.history[:T_eff]).to(device=device, dtype=torch.float32)

    stacks = np.stack(
        [demo.actions[i : i + chunk_len] for i in range(T_eff)], axis=0
    )
    actions = torch.from_numpy(stacks).to(device=device, dtype=torch.float32)

    language = torch.zeros(T_eff, 16, device=device, dtype=torch.float32)
    language_ids = torch.zeros(T_eff, 16, device=device, dtype=torch.long)
    language_mask = torch.zeros(T_eff, 16, device=device, dtype=torch.long)
    masks = torch.ones(T_eff, 1, device=device, dtype=torch.float32)

    return {
        "images": images,
        "states": states,
        "language": language,
        "language_ids": language_ids,
        "language_mask": language_mask,
        "history": history,
        "masks": masks,
        "actions": actions,
    }


@torch.no_grad()
def _encode_obs(policy, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Run tokenizer + context_backbone + planner and return fused_obs, phase_embed, skill_latent."""
    tok = policy.vision_tokenizer(
        images=batch["images"],
        states=batch["states"],
        language=batch["language"],
        history=batch["history"],
        masks=batch["masks"],
        language_ids=batch["language_ids"],
        language_mask=batch["language_mask"],
    )
    context = policy.context_backbone(tok["context_tokens"])
    fused_obs = context.mean(dim=1) + tok["fused"]
    timestep = torch.zeros(fused_obs.size(0), dtype=torch.long, device=fused_obs.device)
    timestep = torch.clamp(timestep.long(), 0, policy.config.max_timestep - 1)
    fused_obs = fused_obs + policy.timestep_embedding(timestep)
    plan = policy.hierarchical_planner(fused_obs=fused_obs, phase_labels=None)
    return {
        "fused_obs": fused_obs,
        "phase_embed": plan["phase_embed"],
        "skill_latent": plan["skill_latent"],
    }


def _per_sample_shortcut_fm_loss(
    policy,
    encoded: Dict[str, torch.Tensor],
    actions_gt: torch.Tensor,
    num_samples: int = 3,
) -> torch.Tensor:
    """Reproduce the ``ShortcutFlowActionHead`` training-path FM loss with reduction='none'.

    Mathematically equivalent to::

        x_t = (1 - t) * noise + t * actions_gt
        v_target = actions_gt - noise
        v_pred = head._velocity(x_t, t, d, cond)
        fm_loss[i] = mean_{Ta,Da} (v_pred - v_target)^2

    ``num_samples`` independent ``(t, d, noise)`` draws are averaged.

    Parameters
    ----------
    policy : PhaseQFlowPolicy (flow_type='shortcut').
    encoded : output of ``_encode_obs``.
    actions_gt : ``(B, Ta, Da)`` tensor.
    num_samples : number of independent resamples.

    Returns
    -------
    ``(B,)`` float tensor of per-sample FM loss.
    """
    head = policy.flow_action_head
    fused_obs = encoded["fused_obs"]
    phase_embed = encoded["phase_embed"]
    skill_latent = encoded["skill_latent"]
    cond = head.conditioner(
        torch.cat([fused_obs, phase_embed, skill_latent], dim=-1)
    )
    B = cond.shape[0]
    device = cond.device
    k_max = int(head.config.shortcut_d_log2_bins)

    acc = torch.zeros(B, device=device)
    for _ in range(max(1, int(num_samples))):
        t = torch.rand(B, 1, device=device)
        k = torch.randint(0, k_max + 1, (B, 1), device=device).float()
        d = torch.pow(2.0, -k)
        noise = torch.randn_like(actions_gt)
        t3 = t.unsqueeze(-1)
        x_t = (1.0 - t3) * noise + t3 * actions_gt
        v_target = actions_gt - noise
        v_pred = head._velocity(x_t, t, d, cond)
        per_sample = ((v_pred - v_target) ** 2).reshape(B, -1).mean(dim=-1)
        acc = acc + per_sample
    return acc / max(1, int(num_samples))


def _per_sample_euler_proxy_loss(
    policy,
    encoded: Dict[str, torch.Tensor],
    actions_gt: torch.Tensor,
) -> torch.Tensor:
    """Proxy FM loss for the Euler action head (no explicit FM path).

    Uses ``MSE(action_pred, actions_gt[:, 0])``, equivalent to a
    single-step regression loss.
    """
    head = policy.flow_action_head
    out = head(
        fused_obs=encoded["fused_obs"],
        phase_embed=encoded["phase_embed"],
        skill_latent=encoded["skill_latent"],
    )
    pred = out["action_pred"]
    gt = actions_gt[:, 0, :] if actions_gt.ndim == 3 else actions_gt
    return ((pred - gt) ** 2).reshape(pred.shape[0], -1).mean(dim=-1)


@torch.no_grad()
def per_timestep_flow_loss(
    policy,
    demo: DemoSample,
    chunk_len: int,
    num_samples: int = 3,
    device: Optional[torch.device] = None,
) -> Optional[np.ndarray]:
    """Compute per-timestep flow-matching loss for one demo.

    Returns
    -------
    ``None`` when the demo is too short; otherwise a ``(T_eff,)`` numpy
    array where ``T_eff = T - chunk_len + 1``.
    """
    policy.eval()
    device = device or next(policy.parameters()).device
    batch = _build_demo_batch(demo, chunk_len=chunk_len, device=device)
    if batch is None:
        return None
    encoded = _encode_obs(policy, batch)

    flow_type = str(getattr(policy.config, "flow_type", "shortcut")).lower()
    if flow_type == "shortcut":
        per_sample = _per_sample_shortcut_fm_loss(
            policy, encoded, batch["actions"], num_samples=num_samples
        )
    else:
        per_sample = _per_sample_euler_proxy_loss(policy, encoded, batch["actions"])
    return per_sample.detach().cpu().float().numpy()
