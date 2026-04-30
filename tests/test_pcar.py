"""Unit tests for the PCAR phase-change trigger and the dual flow head."""

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

from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig # noqa: E402
from lerobot_policy_phaseqflow.modeling_phaseqflow import ( # noqa: E402
    PhaseQFlowPolicy,
    ShortcutFlowActionHead,
)
from lerobot_policy_phaseqflow.phase_centric.pcar_trigger import ( # noqa: E402
    DualFlowHead,
    PCARTrigger,
)


@dataclass
class _TriggerCfg:
    """Minimal config stand-in for the PCAR trigger."""

    pcar_trigger_budget_eps: float = 0.1
    pcar_change_threshold: float = 0.4


def _small_cfg(**overrides) -> PhaseQFlowConfig:
    """Return a compact ``PhaseQFlowConfig`` for CPU-only tests."""
    cfg = PhaseQFlowConfig(
        use_dual_backbone_vision=False,
        use_fsq=False,
        use_bid_sampling=False,
        use_temporal_ensembling=False,
        use_correction_head=False,
        use_ema=False,
        use_bf16=False,
        use_gradient_checkpointing=False,
        use_paged_adamw_8bit=False,
        action_dim=7,
        state_dim=8,
        history_dim=8,
        fusion_hidden_dim=64,
        vision_token_dim=64,
        state_token_dim=64,
        language_token_dim=64,
        history_token_dim=64,
        cross_attn_heads=4,
        num_skills=4,
        skill_embedding_dim=16,
        continuous_skill_dim=16,
        latent_dim=16,
        dit_hidden_dim=64,
        dit_num_layers=2,
        dit_num_heads=4,
        critic_hidden_dim=64,
        flow_steps=2,
        verifier_hidden_dim=32,
        max_timestep=64,
        action_chunk_size=8,
        action_execute_size=4,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _dummy_batch(cfg: PhaseQFlowConfig, B: int = 4) -> dict:
    """Build a random batch with a full ``(B, Ta, Da)`` action tensor."""
    Ta = int(cfg.action_chunk_size)
    return {
        "obs": {
            "images": torch.randn(B, 3, 64, 64),
            "states": torch.randn(B, cfg.state_dim),
            "language": torch.randn(B, 16),
            "history": torch.randn(B, cfg.history_dim),
        },
        "action": torch.randn(B, Ta, cfg.action_dim),
        "timestep": torch.zeros(B, dtype=torch.long),
    }


def test_trigger_warmup_uses_manual_threshold() -> None:
    """Before warmup completes the trigger falls back on ``pcar_change_threshold``."""
    cfg = _TriggerCfg(pcar_trigger_budget_eps=0.1, pcar_change_threshold=0.5)
    trig = PCARTrigger(cfg, history_size=1000, warmup_min=50)
    for _ in range(20):
        assert trig.update_and_check(0.1) is False
    assert trig.update_and_check(0.9) is True


def test_trigger_adaptive_matches_budget() -> None:
    """Long-run replan rate tracks the configured budget within 0.05."""
    cfg = _TriggerCfg(pcar_trigger_budget_eps=0.2, pcar_change_threshold=0.4)
    trig = PCARTrigger(cfg, history_size=500, warmup_min=50)
    rng = np.random.default_rng(0)
    betas = rng.uniform(0.0, 1.0, size=2000)
    for b in betas:
        trig.update_and_check(float(b))
    rate = trig.get_actual_replan_rate()
    assert abs(rate - 0.2) < 0.05, rate


def test_trigger_reset_keeps_history() -> None:
    """``reset`` zeroes the replan counters while keeping the beta history intact."""
    cfg = _TriggerCfg(pcar_trigger_budget_eps=0.1, pcar_change_threshold=0.3)
    trig = PCARTrigger(cfg)
    for _ in range(100):
        trig.update_and_check(0.2)
    hist_before = len(trig.beta_history)
    trig.reset()
    assert len(trig.beta_history) == hist_before
    assert trig._triggered_count == 0
    assert trig._total_count == 0


def test_trigger_hard_reset_clears_history() -> None:
    """``hard_reset`` drops both the history and the counters."""
    cfg = _TriggerCfg(pcar_trigger_budget_eps=0.1, pcar_change_threshold=0.3)
    trig = PCARTrigger(cfg)
    for _ in range(30):
        trig.update_and_check(0.2)
    trig.hard_reset()
    assert len(trig.beta_history) == 0
    assert trig._total_count == 0


def test_trigger_budget_out_of_range_raises() -> None:
    """Budgets outside the open interval (0, 1) raise ``ValueError``."""
    with pytest.raises(ValueError):
        PCARTrigger(_TriggerCfg(pcar_trigger_budget_eps=0.0))
    with pytest.raises(ValueError):
        PCARTrigger(_TriggerCfg(pcar_trigger_budget_eps=1.0))


def test_dual_flow_head_shapes() -> None:
    """``DualFlowHead`` returns pre and post predictions of the expected shape."""
    cfg = _small_cfg(pcar_post_head_ratio=0.5)
    head = DualFlowHead(cfg, ShortcutFlowActionHead).eval()
    B = 2
    Ta = int(cfg.action_chunk_size)
    Da = int(cfg.action_dim)
    fused = torch.randn(B, cfg.fusion_hidden_dim)
    phase_embed = torch.randn(B, cfg.skill_embedding_dim)
    next_phase_embed = torch.randn(B, cfg.skill_embedding_dim)
    skill_latent = torch.randn(B, cfg.continuous_skill_dim)
    out = head(
        fused_obs=fused, phase_embed=phase_embed, skill_latent=skill_latent,
        next_phase_embed=next_phase_embed, actions_gt=None, training=False,
    )
    assert out["pre_action_pred"].shape == (B, Ta, Da)
    assert out["post_action_pred"].shape == (B, Ta // 2, Da)
    assert torch.allclose(out["action_pred"], out["pre_action_pred"])


def test_dual_flow_head_no_next_phase_skips_post() -> None:
    """Without a ``next_phase_embed`` the post head is skipped."""
    cfg = _small_cfg()
    head = DualFlowHead(cfg, ShortcutFlowActionHead).eval()
    B = 2
    fused = torch.randn(B, cfg.fusion_hidden_dim)
    phase_embed = torch.randn(B, cfg.skill_embedding_dim)
    skill_latent = torch.randn(B, cfg.continuous_skill_dim)
    out = head(
        fused_obs=fused, phase_embed=phase_embed, skill_latent=skill_latent,
        next_phase_embed=None, actions_gt=None, training=False,
    )
    assert "post_action_pred" not in out


def test_dual_flow_head_shares_nothing() -> None:
    """The pre and post heads must not share parameters."""
    cfg = _small_cfg()
    head = DualFlowHead(cfg, ShortcutFlowActionHead)
    pre_ids = {id(p) for p in head.pre_head.parameters()}
    post_ids = {id(p) for p in head.post_head.parameters()}
    assert pre_ids.isdisjoint(post_ids), "pre_head and post_head must not share params"


def test_policy_integration_pcar_off() -> None:
    """With PCAR disabled the trigger stays ``None`` and the dual-head flag is off."""
    cfg = _small_cfg(use_pcar=False)
    policy = PhaseQFlowPolicy(cfg)
    assert policy._pcar_trigger is None
    assert policy._use_pcar_dual_head is False


def test_policy_integration_pcar_on_dual() -> None:
    """PCAR with the dual head runs ``compute_loss`` and produces a non-zero post component."""
    cfg = _small_cfg(
        use_pcar=True,
        pcar_dual_head=True,
        use_phase_boundary_posterior=True,
    )
    policy = PhaseQFlowPolicy(cfg).train()
    batch = _dummy_batch(cfg, B=3)
    out = policy.compute_loss(batch, return_dict=True)
    loss = out["loss"]
    assert torch.isfinite(loss)
    pcar_component = out["components"]["pcar_post"]
    assert pcar_component.item() > 0.0
    loss.backward()
    post_grads = [p.grad for p in policy.flow_action_head.post_head.parameters() if p.grad is not None]
    assert len(post_grads) > 0


def test_policy_select_action_triggers_replan() -> None:
    """Repeated calls to ``select_action`` populate the PCAR trigger and its replan rate."""
    cfg = _small_cfg(
        use_pcar=True,
        pcar_dual_head=False,
        use_phase_boundary_posterior=True,
        action_chunk_size=4,
        action_execute_size=4,
    )
    policy = PhaseQFlowPolicy(cfg).eval()
    policy._pcar_trigger = None
    obs = {
        "images": torch.randn(3, 64, 64),
        "states": torch.randn(cfg.state_dim),
        "language": torch.randn(16),
        "history": torch.randn(cfg.history_dim),
    }
    for _ in range(60):
        policy.select_action(obs)
    assert policy._pcar_trigger is not None
    rate = policy._pcar_trigger.get_actual_replan_rate()
    assert 0.0 <= rate <= 1.0


def test_dual_head_post_loss_decreases() -> None:
    """Training for 30 steps shrinks the post loss by at least 20% on a fixed batch."""
    torch.manual_seed(0)
    cfg = _small_cfg(
        use_pcar=True,
        pcar_dual_head=True,
        use_phase_boundary_posterior=True,
        pcar_post_loss_weight=1.0,
    )
    policy = PhaseQFlowPolicy(cfg).train()
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)

    batch = _dummy_batch(cfg, B=4)
    initial = None
    for step in range(30):
        opt.zero_grad(set_to_none=True)
        out = policy.compute_loss(batch, return_dict=True)
        out["loss"].backward()
        opt.step()
        if step == 0:
            initial = float(out["components"]["pcar_post"].item())
        last = float(out["components"]["pcar_post"].item())

    assert initial is not None and initial > 0
    assert last < initial * 0.8, (
        f"pcar_post should drop >=20% over 30 steps; initial={initial:.4f}, last={last:.4f}"
    )
