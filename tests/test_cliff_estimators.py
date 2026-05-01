"""Unit tests for the Predictability-Cliff estimator interfaces."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_SRC = _REPO_ROOT / "lerobot_policy_phaseqflow" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig
from lerobot_policy_phaseqflow.modeling_phaseqflow import PhaseQFlowPolicy
from lerobot_policy_phaseqflow.phase_centric.cliff_estimators import (
    compute_I_hat_1,
    compute_I_hat_2,
    compute_I_hat_3,
    compute_concordance_C,
)


# ---------------------------------------------------------------------------
# Helpers
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


def _dummy_batch(cfg: PhaseQFlowConfig, B: int = 2) -> dict:
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


# ---------------------------------------------------------------------------
# I_hat_1 unit tests
# ---------------------------------------------------------------------------

def test_I_hat_1_is_negation_of_beta() -> None:
    """I_hat_1 equals -phase_beta exactly."""
    beta = torch.tensor([0.0, 0.3, 0.7, 1.0])
    i1 = compute_I_hat_1(beta)
    assert torch.allclose(i1, -beta)


def test_I_hat_1_range() -> None:
    """I_hat_1 values lie in [-1, 0] since beta is in [0, 1]."""
    beta = torch.rand(100)
    i1 = compute_I_hat_1(beta)
    assert torch.all(i1 >= -1.0 - 1e-6)
    assert torch.all(i1 <= 0.0 + 1e-6)


def test_I_hat_1_stable_state() -> None:
    """A stable state (beta=0) gives I_hat_1=0 (maximum predictability)."""
    beta_stable = torch.zeros(4)
    assert torch.all(compute_I_hat_1(beta_stable) == 0.0)


def test_I_hat_1_cliff_state() -> None:
    """A full phase boundary (beta=1) gives I_hat_1=-1 (minimum predictability)."""
    beta_cliff = torch.ones(4)
    assert torch.allclose(compute_I_hat_1(beta_cliff), torch.full((4,), -1.0))


def test_I_hat_1_preserves_shape() -> None:
    """compute_I_hat_1 returns the same shape as the input."""
    for shape in [(3,), (2, 5), (1, 4, 7)]:
        beta = torch.rand(*shape)
        assert compute_I_hat_1(beta).shape == beta.shape


def test_I_hat_1_differentiable() -> None:
    """I_hat_1 is differentiable with respect to beta."""
    beta = torch.rand(8, requires_grad=True)
    compute_I_hat_1(beta).sum().backward()
    assert beta.grad is not None
    assert torch.isfinite(beta.grad).all()


# ---------------------------------------------------------------------------
# I_hat_2 unit tests (action variance)
# ---------------------------------------------------------------------------

def test_I_hat_2_shape_and_sign() -> None:
    """compute_I_hat_2 returns shape (B,) with non-positive values."""
    samples = torch.randn(8, 4, 16, 7)  # N=8, B=4, Ta=16, Da=7
    out = compute_I_hat_2(samples)
    assert out.shape == (4,)
    assert torch.all(out <= 0.0 + 1e-6)


def test_I_hat_2_zero_variance() -> None:
    """Identical action samples produce I_hat_2 == 0."""
    a = torch.randn(1, 3, 8, 5)
    samples = a.expand(6, -1, -1, -1).contiguous()  # N=6 identical copies
    out = compute_I_hat_2(samples)
    assert torch.allclose(out, torch.zeros(3), atol=1e-6)


def test_I_hat_2_high_variance_more_negative() -> None:
    """Higher action sample spread produces a more negative I_hat_2."""
    low = torch.randn(8, 2, 4, 4) * 0.1
    high = torch.randn(8, 2, 4, 4) * 1.0
    assert compute_I_hat_2(low).mean() > compute_I_hat_2(high).mean()


def test_I_hat_2_rejects_low_N() -> None:
    """compute_I_hat_2 requires N≥2 samples; raises ValueError otherwise."""
    with pytest.raises(ValueError):
        compute_I_hat_2(torch.randn(1, 2, 4, 4))


# ---------------------------------------------------------------------------
# I_hat_3 unit tests (velocity curvature)
# ---------------------------------------------------------------------------

def test_I_hat_3_shape_and_sign() -> None:
    """compute_I_hat_3 returns shape (B,) with non-positive values."""
    v_t = torch.randn(3, 16, 7)
    v_prev = torch.randn(3, 16, 7)
    out = compute_I_hat_3(v_t, v_prev)
    assert out.shape == (3,)
    assert torch.all(out <= 0.0 + 1e-6)


def test_I_hat_3_zero_when_identical() -> None:
    """Identical velocities produce I_hat_3 == 0."""
    v = torch.randn(2, 8, 4)
    out = compute_I_hat_3(v, v.clone())
    assert torch.allclose(out, torch.zeros(2), atol=1e-6)


def test_I_hat_3_rejects_shape_mismatch() -> None:
    """compute_I_hat_3 raises ValueError when shapes don't match."""
    with pytest.raises(ValueError):
        compute_I_hat_3(torch.randn(2, 4, 4), torch.randn(3, 4, 4))


# ---------------------------------------------------------------------------
# Concordance C_t unit tests
# ---------------------------------------------------------------------------

def test_concordance_C_in_unit_interval() -> None:
    """compute_concordance_C returns values in [0, 1]."""
    i_hats = [torch.randn(4) for _ in range(3)]
    out = compute_concordance_C(i_hats, window_size=10)
    assert out.shape == (4,)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_concordance_C_stateful_window() -> None:
    """compute_concordance_C with persistent _state honors the rolling window."""
    state: dict = {}
    for _ in range(20):
        i_hats = [torch.randn(2) for _ in range(3)]
        out = compute_concordance_C(i_hats, window_size=5, _state=state)
        assert out.shape == (2,)
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_concordance_C_rejects_empty() -> None:
    """compute_concordance_C raises ValueError on empty input."""
    with pytest.raises(ValueError):
        compute_concordance_C([], window_size=10)


def test_concordance_C_rejects_shape_mismatch() -> None:
    """compute_concordance_C raises ValueError when batch sizes differ."""
    with pytest.raises(ValueError):
        compute_concordance_C(
            [torch.randn(4), torch.randn(5)], window_size=10
        )


# ---------------------------------------------------------------------------
# Policy integration: I_hat_1 in predict_action output
# ---------------------------------------------------------------------------

def test_I_hat_1_absent_when_posterior_off() -> None:
    """I_hat_1 must not appear in the output when use_phase_boundary_posterior=False."""
    cfg = _small_cfg(use_phase_boundary_posterior=False)
    policy = PhaseQFlowPolicy(cfg).eval()
    out = policy.predict_action(_dummy_batch(cfg))
    assert "I_hat_1" not in out


def test_I_hat_1_present_when_posterior_on() -> None:
    """I_hat_1 appears in predict_action output when use_phase_boundary_posterior=True."""
    cfg = _small_cfg(use_phase_boundary_posterior=True)
    policy = PhaseQFlowPolicy(cfg).eval()
    out = policy.predict_action(_dummy_batch(cfg, B=2))
    assert "I_hat_1" in out


def test_I_hat_1_equals_neg_phase_beta_in_policy() -> None:
    """I_hat_1 == -phase_beta in the policy output (exact equality)."""
    cfg = _small_cfg(use_phase_boundary_posterior=True)
    policy = PhaseQFlowPolicy(cfg).eval()
    out = policy.predict_action(_dummy_batch(cfg, B=3))
    assert torch.allclose(out["I_hat_1"], -out["phase_beta"])


def test_I_hat_1_shape_and_range_in_policy() -> None:
    """I_hat_1 has shape (B,) and values in [-1, 0] from the policy."""
    B = 4
    cfg = _small_cfg(use_phase_boundary_posterior=True)
    policy = PhaseQFlowPolicy(cfg).eval()
    out = policy.predict_action(_dummy_batch(cfg, B=B))
    assert out["I_hat_1"].shape == (B,)
    assert torch.all(out["I_hat_1"] >= -1.0 - 1e-6)
    assert torch.all(out["I_hat_1"] <= 0.0 + 1e-6)


def test_phase_beta_still_present_when_posterior_on() -> None:
    """Legacy phase_beta key is still present (backward-compat with existing tests)."""
    cfg = _small_cfg(use_phase_boundary_posterior=True)
    policy = PhaseQFlowPolicy(cfg).eval()
    out = policy.predict_action(_dummy_batch(cfg, B=2))
    assert "phase_beta" in out
    assert "phase_p_hat" in out
