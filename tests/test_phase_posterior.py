"""Unit tests for ``PhasePosteriorEstimator`` and its policy integration."""

from __future__ import annotations

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
from lerobot_policy_phaseqflow.phase_centric.phase_posterior import ( # noqa: E402
    PhasePosteriorEstimator, boundary_prob_from_logits,
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
    """Build a random batch with the expected observation/action keys."""
    return {
        "obs": {
            "images": torch.randn(B, 3, 64, 64),
            "states": torch.randn(B, cfg.state_dim),
            "language": torch.randn(B, 16),
            "history": torch.randn(B, cfg.history_dim),
        },
        "action": torch.randn(B, cfg.action_dim),
        "timestep": torch.zeros(B, dtype=torch.long),
    }


def test_forward_sequence_shapes() -> None:
    """``forward_sequence`` returns ``p_hat`` of shape (B, T, K) and ``beta`` of shape (B, T)."""
    cfg = _small_cfg()
    est = PhasePosteriorEstimator(cfg)
    B, T, K = 3, 12, cfg.num_skills
    logits = torch.randn(B, T, K)
    out = est.forward_sequence(logits)
    assert out["p_hat"].shape == (B, T, K)
    assert out["beta"].shape == (B, T)
    assert torch.allclose(out["p_hat"].sum(-1), torch.ones(B, T), atol=1e-5)


def test_beta_bounds() -> None:
    """Beta values stay within [0, 1] and the first step is zero by definition."""
    cfg = _small_cfg()
    est = PhasePosteriorEstimator(cfg)
    logits = torch.randn(2, 20, cfg.num_skills) * 5.0
    beta = est.forward_sequence(logits)["beta"]
    assert torch.all(beta >= 0.0)
    assert torch.all(beta <= 1.0 + 1e-6)
    assert torch.allclose(beta[:, 0], torch.zeros(2))


def test_step_updates_running_state() -> None:
    """``step`` reproduces ``forward_sequence`` once the first prior is aligned to match."""
    cfg = _small_cfg()
    est = PhasePosteriorEstimator(cfg)
    torch.manual_seed(0)
    B, T, K = 2, 5, cfg.num_skills
    logits = torch.randn(B, T, K)
    seq_out = est.forward_sequence(logits)

    est.reset(batch_size=B)
    est._running_p = logits[:, 0].softmax(-1).detach().clone()
    for t in range(1, T):
        step_out = est.step(logits[:, t])
        assert torch.allclose(step_out["p_hat"], seq_out["p_hat"][:, t], atol=1e-5)
        assert torch.allclose(step_out["beta"], seq_out["beta"][:, t], atol=1e-5)


def test_step_batch_size_auto_reset() -> None:
    """Switching batch size across calls triggers an automatic reset without raising."""
    cfg = _small_cfg()
    est = PhasePosteriorEstimator(cfg)
    est.reset(batch_size=2)
    out = est.step(torch.randn(5, cfg.num_skills))
    assert out["p_hat"].shape == (5, cfg.num_skills)
    assert out["beta"].shape == (5,)


def test_reset_makes_beta_nonzero_on_next_step() -> None:
    """After a reset, a sharp logit causes a meaningful beta (> 0.01)."""
    cfg = _small_cfg()
    est = PhasePosteriorEstimator(cfg)
    est.reset(batch_size=1)
    strong_logits = torch.full((1, cfg.num_skills), -5.0)
    strong_logits[0, 3] = 5.0
    out = est.step(strong_logits)
    assert float(out["beta"]) > 0.01


def test_differentiable() -> None:
    """``forward_sequence`` is differentiable with respect to the input logits."""
    cfg = _small_cfg()
    est = PhasePosteriorEstimator(cfg)
    logits = torch.randn(2, 8, cfg.num_skills, requires_grad=True)
    out = est.forward_sequence(logits)
    out["beta"].sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_boundary_prob_from_logits_functional_parity() -> None:
    """The stateless ``boundary_prob_from_logits`` matches ``forward_sequence`` output."""
    cfg = _small_cfg()
    est = PhasePosteriorEstimator(cfg)
    torch.manual_seed(1)
    logits = torch.randn(3, 10, cfg.num_skills)
    seq = est.forward_sequence(logits)
    fn = boundary_prob_from_logits(logits, alpha=est.alpha)
    assert torch.allclose(seq["p_hat"], fn["p_hat"], atol=1e-6)
    assert torch.allclose(seq["beta"], fn["beta"], atol=1e-6)


def test_policy_integration_off() -> None:
    """With the posterior switch off, the policy has no estimator and no ``phase_beta`` key."""
    cfg = _small_cfg(use_phase_boundary_posterior=False)
    policy = PhaseQFlowPolicy(cfg).eval()
    assert policy.phase_posterior is None
    batch = _dummy_batch(cfg, B=2)
    out = policy.predict_action(batch)
    assert "phase_beta" not in out


def test_policy_integration_on() -> None:
    """With the switch on, ``predict_action`` exposes ``phase_p_hat`` and ``phase_beta``."""
    cfg = _small_cfg(use_phase_boundary_posterior=True)
    policy = PhaseQFlowPolicy(cfg).eval()
    assert policy.phase_posterior is not None
    batch = _dummy_batch(cfg, B=2)
    out = policy.predict_action(batch)
    assert "phase_p_hat" in out and "phase_beta" in out
    assert out["phase_p_hat"].shape == (2, cfg.num_skills)
    assert out["phase_beta"].shape == (2,)
    assert torch.isfinite(out["phase_beta"]).all()
    assert torch.all(out["phase_beta"] >= 0.0)
    assert torch.all(out["phase_beta"] <= 1.0 + 1e-6)


def test_policy_reset_resets_posterior() -> None:
    """``policy.reset`` snaps the posterior's running state back to the uniform prior."""
    cfg = _small_cfg(use_phase_boundary_posterior=True)
    policy = PhaseQFlowPolicy(cfg).eval()
    _ = policy.predict_action(_dummy_batch(cfg, B=1))
    before = policy.phase_posterior._running_p.clone()
    policy.phase_posterior._running_p = torch.zeros_like(before)
    policy.phase_posterior._running_p[:, 0] = 1.0
    policy.reset()
    expected = torch.full_like(policy.phase_posterior._running_p, 1.0 / cfg.num_skills)
    assert torch.allclose(policy.phase_posterior._running_p, expected)


def test_save_load_roundtrip_with_use_phase_boundary_posterior(tmp_path) -> None:
    """The posterior config fields survive a save/load round-trip."""
    cfg = PhaseQFlowConfig(
        use_phase_boundary_posterior=True, phase_posterior_smooth_alpha=0.55,
    )
    cfg.save_pretrained(str(tmp_path))
    cfg2 = PhaseQFlowConfig.from_pretrained(str(tmp_path))
    assert cfg2.use_phase_boundary_posterior is True
    assert cfg2.phase_posterior_smooth_alpha == pytest.approx(0.55)


def test_invalid_alpha_raises() -> None:
    """An alpha outside (0, 1] raises ``ValueError``."""
    cfg = _small_cfg(phase_posterior_smooth_alpha=0.0)
    with pytest.raises(ValueError):
        PhasePosteriorEstimator(cfg)
    cfg2 = _small_cfg(phase_posterior_smooth_alpha=1.5)
    with pytest.raises(ValueError):
        PhasePosteriorEstimator(cfg2)
