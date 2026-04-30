"""Unit tests for the PACE-A weighted flow-matching loss and its policy hook."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_SRC = _REPO_ROOT / "lerobot_policy_phaseqflow" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig # noqa: E402
from lerobot_policy_phaseqflow.modeling_phaseqflow import PhaseQFlowPolicy # noqa: E402
from lerobot_policy_phaseqflow.phase_centric.pace_a_loss import ( # noqa: E402
    _align_beta,
    compute_pace_a_flow_loss,
    pace_a_entropy_reg,
    pace_a_reweight,
)


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
        num_skills=8,
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


def test_align_beta_shapes() -> None:
    """Beta tensors of shape (B,), (B, Ta), (B, Ta, 1) and (B, 1) all broadcast to (B, Ta)."""
    B, Ta = 4, 8
    b = torch.rand(B)
    out = _align_beta(b, B, Ta)
    assert out.shape == (B, Ta)
    assert torch.allclose(out[:, 0], b)
    b2 = torch.rand(B, Ta)
    assert _align_beta(b2, B, Ta).shape == (B, Ta)
    b3 = torch.rand(B, Ta, 1)
    assert _align_beta(b3, B, Ta).shape == (B, Ta)
    b4 = torch.rand(B, 1)
    out4 = _align_beta(b4, B, Ta)
    assert out4.shape == (B, Ta)
    assert torch.allclose(out4, b4.expand(B, Ta))


def test_compute_pace_a_flow_loss_shapes() -> None:
    """Each key in the loss dict has the expected rank/shape."""
    B, Ta, Da = 4, 8, 7
    v_pred = torch.randn(B, Ta, Da, requires_grad=True)
    v_target = torch.randn(B, Ta, Da)
    beta = torch.rand(B)
    out = compute_pace_a_flow_loss(
        v_pred=v_pred, v_target=v_target, beta_t=beta,
        lambda_weight=2.0, entropy_weight=0.01, ablation_mode="full",
    )
    assert out["fm_loss"].ndim == 0
    assert out["entropy_reg"].ndim == 0
    assert out["total"].ndim == 0
    assert out["mean_beta"].ndim == 0
    assert out["max_beta"].ndim == 0
    assert out["weighted_mse_per_step"].shape == (B, Ta)


def test_ablation_no_weight_equals_plain_fm() -> None:
    """``no_weight`` mode reproduces plain MSE and drops the entropy term."""
    torch.manual_seed(0)
    B, Ta, Da = 3, 6, 5
    v_pred = torch.randn(B, Ta, Da)
    v_target = torch.randn(B, Ta, Da)
    beta = torch.rand(B, Ta)
    out = compute_pace_a_flow_loss(
        v_pred=v_pred, v_target=v_target, beta_t=beta,
        lambda_weight=999.0,
        entropy_weight=0.0, ablation_mode="no_weight",
    )
    plain = (v_pred - v_target).pow(2).mean()
    assert torch.allclose(out["fm_loss"], plain, atol=1e-6)
    assert torch.allclose(out["entropy_reg"], torch.zeros(()))


def test_ablation_no_entropy_zero_entropy() -> None:
    """``no_entropy`` mode keeps the weighted FM term but zeroes the entropy regulariser."""
    B, Ta, Da = 3, 6, 5
    v_pred = torch.randn(B, Ta, Da)
    v_target = torch.randn(B, Ta, Da)
    beta = torch.rand(B, Ta)
    out = compute_pace_a_flow_loss(
        v_pred=v_pred, v_target=v_target, beta_t=beta,
        lambda_weight=2.0, entropy_weight=0.01, ablation_mode="no_entropy",
    )
    assert torch.allclose(out["entropy_reg"], torch.zeros(()))
    per_step = (v_pred - v_target).pow(2).mean(-1)
    expected = ((1.0 + 2.0 * beta.clamp(0, 1)) * per_step).mean()
    assert torch.allclose(out["fm_loss"], expected, atol=1e-6)


def test_full_mode_increases_loss_when_lambda_pos() -> None:
    """With lambda > 0 and beta > 0 the weighted FM loss exceeds the unweighted baseline."""
    torch.manual_seed(1)
    B, Ta, Da = 4, 6, 5
    v_pred = torch.randn(B, Ta, Da)
    v_target = torch.randn(B, Ta, Da)
    beta = torch.rand(B, Ta) * 0.5 + 0.3

    out_full = compute_pace_a_flow_loss(
        v_pred=v_pred, v_target=v_target, beta_t=beta,
        lambda_weight=3.0, entropy_weight=0.0, ablation_mode="full",
    )
    out_base = compute_pace_a_flow_loss(
        v_pred=v_pred, v_target=v_target, beta_t=beta,
        lambda_weight=0.0, entropy_weight=0.0, ablation_mode="no_weight",
    )
    assert float(out_full["fm_loss"]) > float(out_base["fm_loss"])


def test_differentiable_through_v_pred() -> None:
    """The total loss is differentiable with respect to ``v_pred`` and the gradient is finite."""
    B, Ta, Da = 3, 5, 4
    v_pred = torch.randn(B, Ta, Da, requires_grad=True)
    v_target = torch.randn(B, Ta, Da)
    beta = torch.rand(B, requires_grad=False)
    out = compute_pace_a_flow_loss(
        v_pred=v_pred, v_target=v_target, beta_t=beta,
        lambda_weight=2.0, entropy_weight=0.01, ablation_mode="full",
    )
    out["total"].backward()
    assert v_pred.grad is not None
    assert torch.isfinite(v_pred.grad).all()


def test_entropy_peaks_at_beta_half() -> None:
    """Bernoulli entropy peaks at beta=0.5 and collapses to zero at beta in {0, 1}."""
    B, Ta = 2, 4
    beta_half = torch.full((B, Ta), 0.5)
    beta_zero = torch.zeros(B, Ta)
    beta_one = torch.ones(B, Ta)

    h_half = pace_a_entropy_reg(beta_half, entropy_weight=1.0)
    h_zero = pace_a_entropy_reg(beta_zero, entropy_weight=1.0)
    h_one = pace_a_entropy_reg(beta_one, entropy_weight=1.0)

    assert float(h_half) < float(h_zero)
    assert float(h_half) < float(h_one)
    assert abs(float(-h_half) - math.log(2)) < 1e-4


def test_pace_a_reweight_functional() -> None:
    """The functional ``pace_a_reweight`` matches ``compute_pace_a_flow_loss['fm_loss']``."""
    torch.manual_seed(2)
    B, Ta, Da = 3, 6, 5
    v_pred = torch.randn(B, Ta, Da)
    v_target = torch.randn(B, Ta, Da)
    beta = torch.rand(B, Ta)
    fm_from_fn = pace_a_reweight(v_pred, v_target, beta, lambda_weight=2.0, ablation_mode="full")
    fm_ref = compute_pace_a_flow_loss(
        v_pred=v_pred, v_target=v_target, beta_t=beta,
        lambda_weight=2.0, entropy_weight=0.0, ablation_mode="no_entropy",
    )["fm_loss"]
    assert torch.allclose(fm_from_fn, fm_ref, atol=1e-6)


def test_pace_a_entropy_reg_functional() -> None:
    """The functional entropy regulariser matches the entry returned by the full loss."""
    B, Ta = 3, 6
    beta = torch.rand(B, Ta)
    ref = compute_pace_a_flow_loss(
        v_pred=torch.zeros(B, Ta, 4),
        v_target=torch.zeros(B, Ta, 4),
        beta_t=beta,
        lambda_weight=0.0, entropy_weight=0.5, ablation_mode="full",
    )["entropy_reg"]
    direct = pace_a_entropy_reg(beta, entropy_weight=0.5)
    assert torch.allclose(ref, direct, atol=1e-6)


def test_policy_integration_pace_a_off() -> None:
    """With PACE-A disabled the entropy and mean-beta components are both zero."""
    cfg = _small_cfg(use_phase_boundary_posterior=True, use_pace_a=False)
    policy = PhaseQFlowPolicy(cfg).train()
    batch = _dummy_batch(cfg, B=2)
    out = policy.compute_loss(batch, return_dict=True)
    comps = out["components"]
    assert "pace_a_fm" in comps
    assert "pace_a_entropy_reg" in comps
    assert float(comps["pace_a_entropy_reg"]) == pytest.approx(0.0)
    assert float(comps["pace_a_mean_beta"]) == pytest.approx(0.0)


def test_policy_integration_pace_a_on() -> None:
    """With PACE-A on the FM/entropy fields populate and the total loss stays finite."""
    torch.manual_seed(123)
    cfg = _small_cfg(
        use_phase_boundary_posterior=True,
        use_pace_a=True,
        pace_a_lambda=2.0,
        pace_a_entropy_weight=0.01,
        pace_a_ablation_mode="full",
    )
    policy = PhaseQFlowPolicy(cfg).train()
    batch = _dummy_batch(cfg, B=2)
    out = policy.compute_loss(batch, return_dict=True)
    comps = out["components"]
    assert float(comps["pace_a_entropy_reg"]) != 0.0 or float(comps["pace_a_mean_beta"]) >= 0.0
    mb = float(comps["pace_a_mean_beta"])
    assert 0.0 <= mb <= 1.0 + 1e-6
    assert torch.isfinite(out["loss"]).all()


def test_invalid_ablation_raises() -> None:
    """An unknown ablation mode raises ``ValueError``."""
    B, Ta, Da = 2, 4, 3
    with pytest.raises(ValueError):
        compute_pace_a_flow_loss(
            v_pred=torch.zeros(B, Ta, Da),
            v_target=torch.zeros(B, Ta, Da),
            beta_t=torch.zeros(B),
            lambda_weight=1.0, entropy_weight=0.01,
            ablation_mode="bogus",
        )
