#!/usr/bin/env python

"""Smoke test for Round 2 hierarchical planner FSQ + InfoNCE changes.

Validates three things without requiring internet for vision backbones:
  1. FSQ-enabled planner: phase_id range, unique code diversity before/after
     20 training steps (no codebook collapse).
  2. compute_loss produces a finite scalar with all six sub-loss components
     printed (imitation / flow / smoothness / verifier / phase / infonce).
  3. ``use_fsq=False`` fallback path with the original Gumbel-Softmax still
     runs end-to-end for A/B comparison.

Uses ``use_dual_backbone_vision=False`` to stay offline.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import torch

from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig
from lerobot_policy_phaseqflow.modeling_phaseqflow import (
    FSQSkillEncoder,
    HierarchicalPlanner,
    PhaseQFlowPolicy,
    infonce_phase_loss,
)


def _make_config(*, use_fsq: bool) -> PhaseQFlowConfig:
    """Light offline config for smoke tests (legacy vision tokenizer)."""
    return PhaseQFlowConfig(
        fusion_hidden_dim=64,
        vision_token_dim=64,
        state_token_dim=64,
        language_token_dim=64,
        history_token_dim=64,
        cross_attn_heads=4,
        num_skills=16,
        skill_embedding_dim=32,
        continuous_skill_dim=32,
        latent_dim=16,
        action_dim=7,
        dit_hidden_dim=64,
        dit_num_layers=2,
        dit_num_heads=4,
        critic_hidden_dim=64,
        flow_steps=2,
        verifier_hidden_dim=32,
        max_timestep=128,
        use_dual_backbone_vision=False,
        use_fsq=use_fsq,
        fsq_levels=[8, 6, 5],
        fsq_dim=3,
        infonce_temperature=0.1,
        infonce_loss_weight=0.1,
        use_infonce_phase_aux=True,
        infonce_chunk_len=4,
    )


def _dummy_batch(cfg: PhaseQFlowConfig, batch_size: int) -> dict:
    """Build a dummy batch for compute_loss; shapes match Legacy tokenizer."""
    return {
        "obs": {
            "images": torch.randn(batch_size, 3, 64, 64),
            "states": torch.randn(batch_size, cfg.state_dim),
            "language": torch.randn(batch_size, 16),
            "history": torch.randn(batch_size, cfg.history_dim),
        },
        "action": torch.randn(batch_size, cfg.action_dim),
        "timestep": torch.zeros(batch_size, dtype=torch.long),
    }


def smoke_planner_fsq_diversity() -> None:
    """Check FSQ phase_id range, diversity, and anti-collapse after 20 steps."""
    print("\n[1] FSQ diversity & anti-collapse check")
    torch.manual_seed(0)
    cfg = _make_config(use_fsq=True)
    planner = HierarchicalPlanner(cfg).train()
    codebook_size = int(np.prod(cfg.fsq_levels))

    fused_obs = torch.randn(32, cfg.fusion_hidden_dim)
    out = planner(fused_obs)
    pid = out["phase_id"]
    assert pid.dtype == torch.long, f"phase_id dtype {pid.dtype}"
    assert pid.min().item() >= 0, "phase_id underflow"
    assert pid.max().item() < codebook_size, f"phase_id overflow >= {codebook_size}"
    unique_before = torch.unique(pid).numel()
    print(f" codebook_size = {codebook_size}")
    print(f" initial unique codes in B=32 batch: {unique_before}")
    assert unique_before >= 15, (
        f"expected >= 15 unique codes at init, got {unique_before} "
        "(possible collapse or too-small batch)"
    )

    opt = torch.optim.Adam(planner.parameters(), lr=1e-3)
    target_embed = torch.randn(32, cfg.skill_embedding_dim)
    for _ in range(20):
        opt.zero_grad()
        fused_obs = torch.randn(32, cfg.fusion_hidden_dim)
        out = planner(fused_obs)
        loss = torch.nn.functional.mse_loss(out["phase_embed"], target_embed)
        loss.backward()
        opt.step()
    with torch.no_grad():
        fused_obs = torch.randn(32, cfg.fusion_hidden_dim)
        pid_after = planner(fused_obs)["phase_id"]
    unique_after = torch.unique(pid_after).numel()
    print(f" unique codes after 20 steps: {unique_after}")
    assert unique_after >= 15, (
        f"codebook seems to collapse: unique after={unique_after}"
    )


def smoke_compute_loss_components(*, use_fsq: bool) -> None:
    """Run ``compute_loss`` once and print all six sub-loss components."""
    label = "FSQ" if use_fsq else "Gumbel (fallback)"
    print(f"\n[2] compute_loss component trace ({label})")
    torch.manual_seed(1)
    cfg = _make_config(use_fsq=use_fsq)
    policy = PhaseQFlowPolicy(cfg).train()
    batch = _dummy_batch(cfg, batch_size=16)
    loss = policy.compute_loss(batch)
    assert torch.isfinite(loss), f"total loss is not finite: {loss}"
    print(f" total loss = {loss.item():.6f}")
    comps = policy._last_loss_components
    for name in ("imitation", "flow", "smoothness", "verifier", "phase", "infonce"):
        val = float(comps[name].item())
        finite = np.isfinite(val)
        print(f" {name:>10s} = {val:.6f} (finite={finite})")
        assert finite, f"component {name} not finite: {val}"


def smoke_infonce_shortcircuit() -> None:
    """InfoNCE with T=1 should return a zero scalar instead of erroring."""
    print("\n[3] InfoNCE short-circuit when T < 2")
    z = torch.randn(4, 1, 8)
    loss = infonce_phase_loss(z, temperature=0.1)
    assert loss.numel() == 1 and loss.item() == 0.0, loss
    print(" infonce(T=1) == 0.0 ✓")


def smoke_infonce_nonzero() -> None:
    """InfoNCE with T>=2 should produce a positive finite scalar."""
    print("\n[3b] InfoNCE with T=4 produces finite positive loss")
    torch.manual_seed(2)
    z = torch.randn(8, 4, 16, requires_grad=True)
    loss = infonce_phase_loss(z, temperature=0.1)
    assert torch.isfinite(loss) and loss.item() > 0, loss
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    print(f" infonce(B=8,T=4) = {loss.item():.6f} ✓ (grad OK)")


def main() -> int:
    """Run all Round 2 planner/FSQ smoke checks and return a process exit code."""
    try:
        smoke_planner_fsq_diversity()
        smoke_compute_loss_components(use_fsq=True)
        smoke_compute_loss_components(use_fsq=False)
        smoke_infonce_shortcircuit()
        smoke_infonce_nonzero()
    except Exception: # noqa: BLE001
        print("\n[FAIL] Smoke test raised an exception:\n")
        traceback.print_exc()
        return 1
    print("\n[PASS] All Round 2 smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
