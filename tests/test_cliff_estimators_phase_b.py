"""Unit tests for the Phase B cliff estimator classes.

Covers:
  §2.1  PosteriorBhattacharyyaEstimator
  §2.2  PolicyVarianceEstimator
  §2.3  VelocityCurvatureEstimator + ShortcutFlowActionHead.velocity / compute_cond
  §2.5  PredictiveInfoEstimator
  §2.6  PCARTrigger.update_and_check renamed signal param + pcar_input_signal config
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_SRC = _REPO_ROOT / "lerobot_policy_phaseqflow" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig
from lerobot_policy_phaseqflow.modeling_phaseqflow import ShortcutFlowActionHead
from lerobot_policy_phaseqflow.phase_centric.cliff_detection.concordance import (
    ConcordanceDetector,
)
from lerobot_policy_phaseqflow.phase_centric.cliff_detection.policy_variance import (
    PolicyVarianceEstimator,
)
from lerobot_policy_phaseqflow.phase_centric.cliff_detection.posterior_bhattacharyya import (
    PosteriorBhattacharyyaEstimator,
)
from lerobot_policy_phaseqflow.phase_centric.cliff_detection.velocity_curvature import (
    VelocityCurvatureEstimator,
)
from lerobot_policy_phaseqflow.phase_centric.pcar_trigger import PCARTrigger
from lerobot_policy_phaseqflow.phase_centric.theory_utils import PredictiveInfoEstimator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _small_cfg(**overrides) -> PhaseQFlowConfig:
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
        action_dim=4,
        state_dim=8,
        history_dim=8,
        fusion_hidden_dim=32,
        vision_token_dim=32,
        state_token_dim=32,
        language_token_dim=32,
        history_token_dim=32,
        cross_attn_heads=4,
        num_skills=4,
        skill_embedding_dim=8,
        continuous_skill_dim=8,
        latent_dim=8,
        dit_hidden_dim=32,
        dit_num_layers=1,
        dit_num_heads=4,
        critic_hidden_dim=32,
        flow_steps=2,
        verifier_hidden_dim=16,
        max_timestep=64,
        action_chunk_size=4,
        action_execute_size=2,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_flow_head(cfg: PhaseQFlowConfig) -> ShortcutFlowActionHead:
    return ShortcutFlowActionHead(cfg).eval()


@dataclass
class _TriggerCfg:
    pcar_trigger_budget_eps: float = 0.1
    pcar_change_threshold: float = 0.4
    pcar_input_signal: str = "concordance"


# ---------------------------------------------------------------------------
# §2.1  PosteriorBhattacharyyaEstimator
# ---------------------------------------------------------------------------

def test_posterior_bhat_step_returns_keys() -> None:
    """step() must return i_hat_1, beta, and p_hat keys."""
    cfg = _small_cfg()
    est = PosteriorBhattacharyyaEstimator(cfg)
    logits = torch.randn(2, est.K)
    out = est.step(logits)
    assert "i_hat_1" in out and "beta" in out and "p_hat" in out


def test_posterior_bhat_i_hat_1_equals_neg_beta() -> None:
    """i_hat_1 == -beta exactly (delegating to compute_I_hat_1)."""
    cfg = _small_cfg()
    est = PosteriorBhattacharyyaEstimator(cfg)
    logits = torch.randn(3, est.K)
    out = est.step(logits)
    assert torch.allclose(out["i_hat_1"], -out["beta"])


def test_posterior_bhat_reset_clears_state() -> None:
    """reset() resets the running posterior so beta drops on the next step."""
    cfg = _small_cfg(phase_posterior_smooth_alpha=0.99)
    est = PosteriorBhattacharyyaEstimator(cfg)
    # Drive the EMA to a strong one-hot
    one_hot = torch.zeros(1, est.K)
    one_hot[0, 0] = 10.0
    for _ in range(30):
        est.step(one_hot)
    # Force a hard switch
    switch = torch.zeros(1, est.K)
    switch[0, -1] = 10.0
    out_before = est.step(switch)
    est.reset()
    # After reset, beta should be near 0 again (uniform prior, no history)
    out_after = est.step(one_hot)
    assert out_before["beta"].item() > out_after["beta"].item()


# ---------------------------------------------------------------------------
# §2.2  PolicyVarianceEstimator
# ---------------------------------------------------------------------------

def test_policy_variance_returns_keys() -> None:
    """estimate() must return i_hat_2 and sigma_sq."""
    cfg = _small_cfg()
    head = _make_flow_head(cfg)
    est = PolicyVarianceEstimator(n_samples=4)
    B = 2
    fused = torch.randn(B, cfg.fusion_hidden_dim)
    pe = torch.randn(B, cfg.skill_embedding_dim)
    sl = torch.randn(B, cfg.continuous_skill_dim)
    out = est.estimate(head, fused, pe, sl)
    assert "i_hat_2" in out and "sigma_sq" in out


def test_policy_variance_nonpositive_i_hat_2() -> None:
    """i_hat_2 = -sigma_sq is always <= 0."""
    cfg = _small_cfg()
    head = _make_flow_head(cfg)
    est = PolicyVarianceEstimator(n_samples=6)
    B = 3
    out = est.estimate(
        head,
        torch.randn(B, cfg.fusion_hidden_dim),
        torch.randn(B, cfg.skill_embedding_dim),
        torch.randn(B, cfg.continuous_skill_dim),
    )
    assert torch.all(out["i_hat_2"] <= 0.0 + 1e-9)
    assert torch.all(out["sigma_sq"] >= 0.0 - 1e-9)


def test_policy_variance_n_samples_lt2_raises() -> None:
    """n_samples < 2 must raise ValueError."""
    with pytest.raises(ValueError):
        PolicyVarianceEstimator(n_samples=1)


# ---------------------------------------------------------------------------
# §2.3  ShortcutFlowActionHead.velocity / compute_cond
# ---------------------------------------------------------------------------

def test_flow_head_velocity_shape() -> None:
    """velocity() returns (B, Ta, Da)."""
    cfg = _small_cfg()
    head = _make_flow_head(cfg)
    B = 2
    x = torch.randn(B, cfg.action_chunk_size, cfg.action_dim)
    cond = torch.randn(B, cfg.dit_hidden_dim)
    v = head.velocity(x, 0.5, cond)
    assert v.shape == (B, cfg.action_chunk_size, cfg.action_dim)


def test_flow_head_compute_cond_shape() -> None:
    """compute_cond() returns (B, H) with the same H as the conditioner output."""
    cfg = _small_cfg()
    head = _make_flow_head(cfg)
    B = 3
    fused = torch.randn(B, cfg.fusion_hidden_dim)
    pe = torch.randn(B, cfg.skill_embedding_dim)
    sl = torch.randn(B, cfg.continuous_skill_dim)
    c = head.compute_cond(fused, pe, sl)
    assert c.ndim == 2
    assert c.shape[0] == B


def test_flow_head_velocity_consistent_with_private() -> None:
    """Public velocity() gives the same result as calling _velocity directly."""
    cfg = _small_cfg()
    head = _make_flow_head(cfg)
    B = 2
    x = torch.randn(B, cfg.action_chunk_size, cfg.action_dim)
    cond = torch.randn(B, cfg.dit_hidden_dim)
    tau = 0.3
    d_val = 1.0
    t = torch.full((B, 1), tau)
    d_t = torch.full((B, 1), d_val)
    v_pub = head.velocity(x, tau, cond, d=d_val)
    v_priv = head._velocity(x, t, d_t, cond)
    assert torch.allclose(v_pub, v_priv, atol=1e-6)


# ---------------------------------------------------------------------------
# §2.3  VelocityCurvatureEstimator
# ---------------------------------------------------------------------------

def test_velocity_curvature_first_step_is_none() -> None:
    """First call after reset must return i_hat_3=None (no predecessor)."""
    cfg = _small_cfg()
    head = _make_flow_head(cfg)
    est = VelocityCurvatureEstimator()
    B = 2
    out = est.update(
        head,
        torch.randn(B, cfg.fusion_hidden_dim),
        torch.randn(B, cfg.skill_embedding_dim),
        torch.randn(B, cfg.continuous_skill_dim),
    )
    assert out["i_hat_3"] is None
    assert out["cond_diff_sq"] is None


def test_velocity_curvature_second_step_has_value() -> None:
    """Second call returns a finite tensor of shape (B,)."""
    cfg = _small_cfg()
    head = _make_flow_head(cfg)
    est = VelocityCurvatureEstimator()
    B = 2
    args = (
        torch.randn(B, cfg.fusion_hidden_dim),
        torch.randn(B, cfg.skill_embedding_dim),
        torch.randn(B, cfg.continuous_skill_dim),
    )
    est.update(head, *args)
    out = est.update(head, *args)
    assert out["i_hat_3"] is not None
    assert out["i_hat_3"].shape == (B,)
    assert torch.isfinite(out["i_hat_3"]).all()


def test_velocity_curvature_nonpositive() -> None:
    """i_hat_3 = -||v_t - v_{t-1}||^2 is always <= 0."""
    cfg = _small_cfg()
    head = _make_flow_head(cfg)
    est = VelocityCurvatureEstimator()
    B = 2
    for _ in range(4):
        out = est.update(
            head,
            torch.randn(B, cfg.fusion_hidden_dim),
            torch.randn(B, cfg.skill_embedding_dim),
            torch.randn(B, cfg.continuous_skill_dim),
        )
    assert out["i_hat_3"] is not None
    assert torch.all(out["i_hat_3"] <= 0.0 + 1e-9)


def test_velocity_curvature_reset() -> None:
    """After reset() the first call again returns None."""
    cfg = _small_cfg()
    head = _make_flow_head(cfg)
    est = VelocityCurvatureEstimator()
    B = 2
    args = (
        torch.randn(B, cfg.fusion_hidden_dim),
        torch.randn(B, cfg.skill_embedding_dim),
        torch.randn(B, cfg.continuous_skill_dim),
    )
    est.update(head, *args)
    est.update(head, *args)
    est.reset()
    out = est.update(head, *args)
    assert out["i_hat_3"] is None


# ---------------------------------------------------------------------------
# §2.5  PredictiveInfoEstimator
# ---------------------------------------------------------------------------

def test_predictive_info_estimator_mi_lb_scalar() -> None:
    """forward() returns a scalar mi_lower_bound."""
    est = PredictiveInfoEstimator(x_dim=16, c_dim=16, hidden_dim=32)
    x = torch.randn(8, 16)
    c = torch.randn(8, 16)
    out = est.forward(x, c)
    assert "mi_lower_bound" in out
    assert out["mi_lower_bound"].ndim == 0
    assert torch.isfinite(out["mi_lower_bound"])


def test_predictive_info_per_timestep_shape() -> None:
    """estimate_per_timestep returns shape (T,)."""
    est = PredictiveInfoEstimator(x_dim=8, c_dim=8, hidden_dim=16)
    T, B = 5, 4
    x_seq = torch.randn(T, B, 8)
    c_seq = torch.randn(T, B, 8)
    mi_seq = est.estimate_per_timestep(x_seq, c_seq)
    assert mi_seq.shape == (T,)
    assert torch.isfinite(mi_seq).all()


# ---------------------------------------------------------------------------
# §2.6  PCAR pcar_input_signal config + update_and_check renamed param
# ---------------------------------------------------------------------------

def test_pcar_input_signal_default_is_concordance() -> None:
    """PhaseQFlowConfig default pcar_input_signal is 'concordance'."""
    cfg = _small_cfg()
    assert cfg.pcar_input_signal == "concordance"


def test_pcar_trigger_reads_input_signal() -> None:
    """PCARTrigger stores the pcar_input_signal from config."""
    tcfg = _TriggerCfg(pcar_input_signal="concordance")
    trig = PCARTrigger(tcfg)
    assert trig.input_signal == "concordance"


def test_pcar_update_and_check_signal_positional() -> None:
    """update_and_check works with positional signal argument (backward compat)."""
    trig = PCARTrigger(_TriggerCfg())
    for _ in range(20):
        trig.update_and_check(0.1)
    assert trig.update_and_check(0.95) is True


def test_pcar_update_and_check_signal_kwarg() -> None:
    """update_and_check works when calling with signal= keyword."""
    trig = PCARTrigger(_TriggerCfg())
    for _ in range(20):
        trig.update_and_check(signal=0.1)
    result = trig.update_and_check(signal=0.99)
    assert isinstance(result, bool)
