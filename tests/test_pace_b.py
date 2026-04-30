"""Unit tests for the PACE-B MoE smooth-switching gate and its policy integration."""

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
from lerobot_policy_phaseqflow.phase_centric.pace_b_moe import ( # noqa: E402
    FlowActionHeadPACE,
    PhaseMoE,
    smooth_phase_gate,
)


def _small_cfg(**overrides) -> PhaseQFlowConfig:
    """Return a compact ``PhaseQFlowConfig`` with ``num_skills`` matching ``moe_num_experts``."""
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
        moe_num_experts=4,
        moe_expert_hidden_dim=64,
        moe_switch_kappa=5.0,
        moe_switch_mu=2.0,
        moe_top_k=0,
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


def test_smooth_phase_gate_scalar() -> None:
    """``smooth_phase_gate`` returns the expected sigmoid values at beta in {0, 0.4, 1}."""
    beta = torch.tensor([0.0, 0.4, 1.0])
    alpha = smooth_phase_gate(beta, kappa=5.0, mu=2.0)
    assert torch.allclose(alpha[0], torch.tensor(0.1192), atol=1e-3)
    assert torch.allclose(alpha[1], torch.tensor(0.5000), atol=1e-3)
    assert torch.allclose(alpha[2], torch.tensor(0.9526), atol=1e-3)


def test_smooth_phase_gate_monotonic() -> None:
    """``smooth_phase_gate`` is monotonically increasing in beta."""
    beta = torch.linspace(0.0, 1.0, 20)
    alpha = smooth_phase_gate(beta, kappa=5.0, mu=2.0)
    diffs = alpha[1:] - alpha[:-1]
    assert (diffs > 0).all()


def test_phase_moe_forward_shape() -> None:
    """``PhaseMoE`` forward returns ``(B, latent_dim)``."""
    cfg = _small_cfg()
    K = int(cfg.moe_num_experts)
    B = 4
    moe = PhaseMoE(cfg).eval()
    cond = torch.randn(B, cfg.fusion_hidden_dim)
    u = torch.randn(B, cfg.latent_dim)
    tau = torch.zeros(B, 1)
    p_hat = torch.softmax(torch.randn(B, K), dim=-1)
    beta = torch.rand(B)
    v = moe(cond=cond, u=u, tau=tau, p_hat=p_hat, beta=beta)
    assert v.shape == (B, cfg.latent_dim)


def test_phase_moe_training_gate_equals_phat() -> None:
    """During training the gate passes ``p_hat`` through unchanged."""
    cfg = _small_cfg()
    K = int(cfg.moe_num_experts)
    B = 3
    moe = PhaseMoE(cfg).train()
    p_hat = torch.softmax(torch.randn(B, K), dim=-1)
    beta = torch.rand(B)
    gate = moe.compute_gate(p_hat=p_hat, beta=beta, training=True)
    assert torch.allclose(gate, p_hat)


def test_phase_moe_cold_start_uses_phat() -> None:
    """At inference the first call cold-starts from ``p_hat`` and populates the running gate."""
    cfg = _small_cfg()
    K = int(cfg.moe_num_experts)
    B = 2
    moe = PhaseMoE(cfg).eval()
    p_hat = torch.softmax(torch.randn(B, K), dim=-1)
    beta = torch.rand(B)
    gate = moe.compute_gate(p_hat=p_hat, beta=beta, training=False)
    assert torch.allclose(gate, p_hat)
    assert moe._running_gate.shape == (B, K)


def test_phase_moe_reset_switching_clears_state() -> None:
    """``reset_switching`` clears the running gate and the next call cold-starts again."""
    cfg = _small_cfg()
    K = int(cfg.moe_num_experts)
    B = 2
    moe = PhaseMoE(cfg).eval()
    p_hat = torch.softmax(torch.randn(B, K), dim=-1)
    beta = torch.rand(B)
    moe.compute_gate(p_hat=p_hat, beta=beta, training=False)
    assert moe._running_gate.numel() > 0
    moe.reset_switching()
    assert moe._running_gate.numel() == 0
    new_p = torch.softmax(torch.randn(B, K), dim=-1)
    gate = moe.compute_gate(p_hat=new_p, beta=beta, training=False)
    assert torch.allclose(gate, new_p)


def test_phase_moe_topk_sparsification() -> None:
    """With ``top_k=1`` each row has exactly one non-zero entry and re-normalises to sum one."""
    cfg = _small_cfg(moe_top_k=1, moe_num_experts=4)
    B = 3
    moe = PhaseMoE(cfg).eval()
    p_hat = torch.softmax(torch.randn(B, 4), dim=-1)
    beta = torch.ones(B)
    gate = moe.compute_gate(p_hat=p_hat, beta=beta, training=False)
    nonzero = (gate > 0).sum(dim=-1)
    assert torch.all(nonzero == 1), nonzero
    assert torch.allclose(gate.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_expert_param_count_under_100k() -> None:
    """A single expert stays under 100K parameters with the default hidden size."""
    cfg = _small_cfg(moe_expert_hidden_dim=128)
    moe = PhaseMoE(cfg)
    n = moe.expert_param_count(0)
    assert n <= 100_000, f"expert param count {n} exceeds 100K budget"


def test_gate_l2_high_at_boundary() -> None:
    """At high beta the gate tracks ``p_hat`` aggressively, giving L2 > 0.3 across a switch."""
    cfg = _small_cfg()
    K = int(cfg.moe_num_experts)
    B = 1
    moe = PhaseMoE(cfg).eval()
    p_old = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    moe.compute_gate(p_hat=p_old, beta=torch.zeros(B), training=False)
    p_new = torch.tensor([[0.01, 0.01, 0.97, 0.01]])
    beta_hi = torch.ones(B)
    g_next = moe.compute_gate(p_hat=p_new, beta=beta_hi, training=False)
    running_prev = p_old.clone()
    l2 = (g_next - running_prev).pow(2).sum(dim=-1).sqrt().item()
    assert l2 > 0.3, f"expected L2 > 0.3 at boundary, got {l2}"


def test_gate_l2_low_inside_phase() -> None:
    """At low beta the gate barely moves, giving L2 < 0.05 inside a phase."""
    cfg = _small_cfg()
    K = int(cfg.moe_num_experts)
    B = 1
    moe = PhaseMoE(cfg).eval()
    p_old = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    moe.compute_gate(p_hat=p_old, beta=torch.zeros(B), training=False)
    p_new = torch.tensor([[0.90, 0.05, 0.03, 0.02]])
    beta_lo = torch.zeros(B)
    running_prev = moe._running_gate.clone()
    g_next = moe.compute_gate(p_hat=p_new, beta=beta_lo, training=False)
    l2 = (g_next - running_prev).pow(2).sum(dim=-1).sqrt().item()
    assert l2 < 0.05, f"expected L2 < 0.05 inside phase, got {l2}"


def test_flow_head_pace_output_shape() -> None:
    """``FlowActionHeadPACE`` produces action and latent predictions of the expected shape."""
    cfg = _small_cfg()
    B = 3
    head = FlowActionHeadPACE(cfg).eval()
    fused = torch.randn(B, cfg.fusion_hidden_dim)
    phase_emb = torch.randn(B, cfg.skill_embedding_dim)
    skill_latent = torch.randn(B, cfg.continuous_skill_dim)
    p_hat = torch.softmax(torch.randn(B, int(cfg.moe_num_experts)), dim=-1)
    beta = torch.rand(B)
    out = head(
        fused_obs=fused, phase_embed=phase_emb, skill_latent=skill_latent,
        p_hat=p_hat, beta=beta,
    )
    assert out["action_pred"].shape == (B, int(cfg.action_chunk_size), int(cfg.action_dim))
    assert out["latent_action_pred"].shape == (B, int(cfg.latent_dim))


def test_policy_integration_pace_b_off() -> None:
    """With ``use_pace_b=False`` the policy does not build the PACE-B head."""
    cfg = _small_cfg(use_pace_b=False)
    policy = PhaseQFlowPolicy(cfg)
    assert policy.pace_b_flow_head is None


def test_policy_integration_pace_b_on() -> None:
    """With ``use_pace_b=True`` the policy runs ``compute_loss`` and backprop end-to-end."""
    cfg = _small_cfg(
        use_pace_b=True,
        use_phase_boundary_posterior=True,
    )
    policy = PhaseQFlowPolicy(cfg).train()
    assert policy.pace_b_flow_head is not None

    batch = _dummy_batch(cfg, B=3)
    out = policy.compute_loss(batch, return_dict=True)
    loss = out["loss"]
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in policy.pace_b_flow_head.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_flow_head_pace_projects_planner_k_to_moe_k() -> None:
    """A planner K distinct from ``moe_num_experts`` triggers a learned projection layer."""
    cfg = _small_cfg(
        use_pace_b=True,
        use_phase_boundary_posterior=True,
        num_skills=240,
        moe_num_experts=4,
    )
    head = FlowActionHeadPACE(cfg).eval()
    assert head.p_hat_proj is not None
    assert head.p_hat_proj.in_features == 240
    assert head.p_hat_proj.out_features == 4

    B = 2
    fused = torch.randn(B, cfg.fusion_hidden_dim)
    phase_emb = torch.randn(B, cfg.skill_embedding_dim)
    skill_latent = torch.randn(B, cfg.continuous_skill_dim)
    p_hat_planner = torch.softmax(torch.randn(B, 240), dim=-1)
    beta = torch.rand(B)
    out = head(
        fused_obs=fused, phase_embed=phase_emb, skill_latent=skill_latent,
        p_hat=p_hat_planner, beta=beta,
    )
    assert out["action_pred"].shape == (B, int(cfg.action_chunk_size), int(cfg.action_dim))


def test_flow_head_pace_identity_when_dims_match() -> None:
    """When the planner K already matches ``moe_num_experts`` no projection layer is built."""
    cfg = _small_cfg(num_skills=4, moe_num_experts=4)
    head = FlowActionHeadPACE(cfg)
    assert head.p_hat_proj is None


def test_select_action_pace_b_with_fsq_default() -> None:
    """``select_action`` runs end-to-end when FSQ and MoE have different K sizes."""
    cfg = _small_cfg(
        use_pace_b=True,
        use_phase_boundary_posterior=True,
        use_fsq=True,
        fsq_levels=[3, 4, 5],
        fsq_dim=3,
        moe_num_experts=4,
    )
    policy = PhaseQFlowPolicy(cfg).eval()
    assert policy.pace_b_flow_head is not None
    assert policy.pace_b_flow_head.p_hat_proj is not None
    assert policy.pace_b_flow_head.p_hat_proj.in_features == 3 * 4 * 5

    obs = {
        "images": torch.zeros(3, 64, 64),
        "states": torch.zeros(cfg.state_dim),
        "language": torch.zeros(16),
        "history": torch.zeros(cfg.history_dim),
    }
    with torch.no_grad():
        action = policy.select_action(obs)
    assert torch.isfinite(action).all()
