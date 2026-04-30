"""Unit tests for ``ChunkInfoNCEHead`` and its policy integration."""

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
from lerobot_policy_phaseqflow.phase_centric.identifiability import ( # noqa: E402
    ChunkInfoNCEHead, chunk_infonce_loss,
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


def test_head_forward_shapes() -> None:
    """Head forward returns a scalar loss and a diagnostics dict with expected keys."""
    cfg = _small_cfg()
    head = ChunkInfoNCEHead(cfg)
    B, D, Ta, Da, K = 6, cfg.fusion_hidden_dim, cfg.action_chunk_size, cfg.action_dim, cfg.num_skills
    fused = torch.randn(B, D)
    logits = torch.full((B, K), -5.0)
    for i in range(B):
        logits[i, i % K] = 10.0
    loss, diag = head(
        fused_obs=fused,
        action_chunk=torch.randn(B, Ta, Da),
        phase_logits=logits,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    for key in ("info_nce_acc", "phase_entropy", "num_valid_rows"):
        assert key in diag
    assert diag["num_valid_rows"] > 0


def test_head_degenerate_batch() -> None:
    """B=1 has no contrast partners, so loss is zero and no rows are valid."""
    cfg = _small_cfg()
    head = ChunkInfoNCEHead(cfg)
    loss, diag = head(
        fused_obs=torch.randn(1, cfg.fusion_hidden_dim),
        action_chunk=torch.randn(1, cfg.action_chunk_size, cfg.action_dim),
        phase_logits=torch.randn(1, cfg.num_skills),
    )
    assert float(loss) == 0.0
    assert diag["num_valid_rows"] == 0


def test_head_action_dim_mismatch_is_padded() -> None:
    """Chunk tensors shorter than the configured shape are padded rather than rejected."""
    cfg = _small_cfg(action_chunk_size=8, action_dim=7)
    head = ChunkInfoNCEHead(cfg)
    B = 4
    loss, _ = head(
        fused_obs=torch.randn(B, cfg.fusion_hidden_dim),
        action_chunk=torch.randn(B, 3, 5),
        phase_logits=torch.randn(B, cfg.num_skills),
    )
    assert torch.isfinite(loss)


def test_head_k_inference() -> None:
    """K is taken from ``num_skills`` under Gumbel and from ``prod(fsq_levels)`` under FSQ."""
    cfg_gumbel = _small_cfg(use_fsq=False, num_skills=13)
    head_g = ChunkInfoNCEHead(cfg_gumbel)
    assert head_g.K == 13

    cfg_fsq = _small_cfg(use_fsq=True)
    head_f = ChunkInfoNCEHead(cfg_fsq)
    assert head_f.K == 8 * 6 * 5


def test_functional_wrapper_parity() -> None:
    """The functional ``chunk_infonce_loss`` matches the head's own output."""
    cfg = _small_cfg()
    head = ChunkInfoNCEHead(cfg).eval()
    torch.manual_seed(0)
    B, D, Ta, Da, K = 4, cfg.fusion_hidden_dim, cfg.action_chunk_size, cfg.action_dim, cfg.num_skills
    fused = torch.randn(B, D)
    chunk = torch.randn(B, Ta, Da)
    logits = torch.randn(B, K)
    l1, d1 = head(fused_obs=fused, action_chunk=chunk, phase_logits=logits)
    l2, d2 = chunk_infonce_loss(head, fused, chunk, logits)
    assert torch.allclose(l1, l2)
    assert d1 == d2


def test_policy_integration_off() -> None:
    """With ``use_chunk_infonce=False`` the policy never builds the head and the component is zero."""
    cfg = _small_cfg(use_chunk_infonce=False)
    policy = PhaseQFlowPolicy(cfg)
    assert policy.chunk_infonce_head is None
    out = policy.compute_loss(_dummy_batch(cfg, B=4), return_dict=True)
    comps = out["components"]
    assert "chunk_infonce" in comps
    assert float(comps["chunk_infonce"]) == 0.0


def test_policy_integration_on() -> None:
    """With the switch on, the policy attaches the head and emits a finite component."""
    cfg = _small_cfg(use_chunk_infonce=True, chunk_infonce_weight=0.5)
    policy = PhaseQFlowPolicy(cfg)
    assert policy.chunk_infonce_head is not None
    out = policy.compute_loss(_dummy_batch(cfg, B=4), return_dict=True)
    comps = out["components"]
    v = float(comps["chunk_infonce"])
    assert v == v
    assert v < float("inf")


def test_policy_integration_backward() -> None:
    """Backward pass reaches the head parameters and a single optimizer step moves them."""
    cfg = _small_cfg(use_chunk_infonce=True, chunk_infonce_weight=1.0)
    policy = PhaseQFlowPolicy(cfg).train()
    _ = policy.compute_loss(_dummy_batch(cfg, B=4))
    head = policy.chunk_infonce_head
    assert head is not None
    before = {
        name: p.detach().clone() for name, p in head.named_parameters()
    }
    out = policy.compute_loss(_dummy_batch(cfg, B=8), return_dict=True)
    loss = out["loss"]
    loss.backward()
    got_grad = False
    for name, p in head.named_parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            got_grad = True
            break
    assert got_grad, "chunk_infonce_head parameters received no gradient"

    opt = torch.optim.SGD(head.parameters(), lr=1e-1)
    opt.step()
    changed = any(
        not torch.allclose(before[name], p.detach()) for name, p in head.named_parameters()
    )
    assert changed


def test_save_load_roundtrip_with_use_chunk_infonce(tmp_path) -> None:
    """The chunk-InfoNCE fields survive a save/load round-trip."""
    cfg = PhaseQFlowConfig(use_chunk_infonce=True, chunk_infonce_weight=0.7)
    cfg.save_pretrained(str(tmp_path))
    cfg2 = PhaseQFlowConfig.from_pretrained(str(tmp_path))
    assert cfg2.use_chunk_infonce is True
    assert cfg2.chunk_infonce_weight == pytest.approx(0.7)
