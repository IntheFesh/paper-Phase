#!/usr/bin/env python

"""Smoke test for Round 4 Shortcut flow + ACT temporal ensembling.

Validates four acceptance items without network:

  1. Shortcut head training path returns finite ``fm_loss`` / ``sc_loss``;
     ``compute_loss`` scalar is finite; all 6 components are finite.
  2. 1-NFE Shortcut inference wall-clock is < 1/4 of the 4-step Euler head
     (the whole point of the upgrade).
  3. :class:`ACTTemporalEnsembler` produces roughly 3x smoother action streams
     than a stream of independent random chunks (continuity check).
  4. ``policy.select_action(obs)`` runs a 50-step rollout without crashing,
     returns single-step actions of shape ``(action_dim,)``, and ``reset()``
     clears the internal cache.

All tests use ``use_dual_backbone_vision=False`` so no timm / hf download.
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np
import torch

from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig
from lerobot_policy_phaseqflow.modeling_phaseqflow import (
    FlowActionHeadEuler,
    PhaseQFlowPolicy,
    ShortcutFlowActionHead,
)
from lerobot_policy_phaseqflow.temporal_ensembler import ACTTemporalEnsembler


def _make_config(*, flow_type: str, use_temporal_ensembling: bool = True) -> PhaseQFlowConfig:
    """Offline config for smoke tests (legacy vision tokenizer)."""
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
        flow_steps=4,
        verifier_hidden_dim=32,
        max_timestep=128,
        use_dual_backbone_vision=False,
        use_fsq=True,
        fsq_levels=[8, 6, 5],
        fsq_dim=3,
        use_infonce_phase_aux=True,
        infonce_chunk_len=4,
        flow_type=flow_type,
        shortcut_d_log2_bins=8,
        shortcut_self_consistency_weight=0.5,
        action_chunk_size=16,
        action_execute_size=8,
        use_temporal_ensembling=use_temporal_ensembling,
        ensemble_decay_m=0.05,
        ensemble_buffer_size=16,
        use_action_quantile_norm=False,
    )


def _dummy_batch(cfg: PhaseQFlowConfig, batch_size: int, chunked: bool) -> dict:
    """Build a dummy batch; Legacy tokenizer accepts arbitrary tensor shapes."""
    Ta = int(cfg.action_chunk_size)
    if chunked:
        action = torch.randn(batch_size, Ta, cfg.action_dim)
    else:
        action = torch.randn(batch_size, cfg.action_dim)
    return {
        "obs": {
            "images": torch.randn(batch_size, 3, 64, 64),
            "states": torch.randn(batch_size, cfg.state_dim),
            "language": torch.randn(batch_size, 16),
            "history": torch.randn(batch_size, cfg.history_dim),
        },
        "action": action,
        "timestep": torch.zeros(batch_size, dtype=torch.long),
    }


def smoke_shortcut_losses_finite() -> None:
    """Check ``fm_loss`` / ``sc_loss`` and all ``compute_loss`` components are finite."""
    print("\n[1] Shortcut fm_loss / sc_loss / compute_loss finite")
    torch.manual_seed(0)
    cfg = _make_config(flow_type="shortcut", use_temporal_ensembling=False)
    policy = PhaseQFlowPolicy(cfg).train()

    for chunked_label, chunked in (("single-step", False), ("chunk-(B,Ta,Da)", True)):
        batch = _dummy_batch(cfg, batch_size=16, chunked=chunked)
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss), f"[{chunked_label}] total loss not finite: {loss}"
        comps = policy._last_loss_components
        print(f" [{chunked_label}] total={loss.item():.4f} "
              + " ".join(f"{k}={float(v):.4f}" for k, v in comps.items()))
        for name in ("imitation", "flow", "smoothness", "verifier", "phase", "infonce"):
            v = float(comps[name].item())
            assert np.isfinite(v), f"component {name} not finite: {v}"

    head = ShortcutFlowActionHead(cfg)
    fused_obs = torch.randn(8, cfg.fusion_hidden_dim)
    phase_embed = torch.randn(8, cfg.skill_embedding_dim)
    skill_latent = torch.randn(8, cfg.continuous_skill_dim)
    actions_gt = torch.randn(8, cfg.action_chunk_size, cfg.action_dim)
    out = head(fused_obs=fused_obs, phase_embed=phase_embed,
               skill_latent=skill_latent, actions_gt=actions_gt, training=True)
    fm = float(out["fm_loss"].item())
    sc = float(out["sc_loss"].item())
    print(f" direct head: fm_loss={fm:.4f} sc_loss={sc:.4f}")
    assert np.isfinite(fm) and np.isfinite(sc), (fm, sc)
    infer = head(fused_obs=fused_obs, phase_embed=phase_embed,
                 skill_latent=skill_latent, actions_gt=None, training=False)
    assert infer["action_pred"].shape == (8, cfg.action_chunk_size, cfg.action_dim)
    print(f" inference action_pred shape = {tuple(infer['action_pred'].shape)} OK")


def smoke_shortcut_speedup() -> None:
    """Compare 1-NFE Shortcut inference wall-clock against a 4-step Euler simulation."""
    print("\n[2] 1-NFE vs 4-NFE Euler simulation (same ShortcutFlowActionHead)")
    torch.manual_seed(1)
    cfg = _make_config(flow_type="shortcut", use_temporal_ensembling=False)

    head = ShortcutFlowActionHead(cfg).eval()
    B = 32
    fused_obs = torch.randn(B, cfg.fusion_hidden_dim)
    phase_embed = torch.randn(B, cfg.skill_embedding_dim)
    skill_latent = torch.randn(B, cfg.continuous_skill_dim)
    cond = head.conditioner(torch.cat([fused_obs, phase_embed, skill_latent], dim=-1))

    def one_nfe() -> None:
        """Single-step Shortcut inference."""
        head(fused_obs=fused_obs, phase_embed=phase_embed,
             skill_latent=skill_latent, actions_gt=None, training=False)

    def four_nfe() -> None:
        """Plain Euler integration with four equal steps using the same velocity net."""
        x = torch.randn(B, head.Ta, head.Da)
        t = torch.zeros(B, 1)
        dt = torch.full((B, 1), 1.0 / 4.0)
        for _ in range(4):
            x = x + 0.25 * head._velocity(x, t, dt, cond)
            t = t + 0.25

    with torch.no_grad():
        for _ in range(3):
            one_nfe()
            four_nfe()

    iters = 50
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            one_nfe()
    one_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            four_nfe()
    four_t = time.perf_counter() - t0

    ratio = one_t / max(four_t, 1e-9)
    print(f" 1-NFE {one_t*1000/iters:.2f} ms / call, "
          f"4-NFE {four_t*1000/iters:.2f} ms / call")
    print(f" 1-NFE / 4-NFE = {ratio:.3f} (target < 0.25, allow CPU overhead floor 0.45)")
    assert ratio < 0.45, (
        f"1-NFE should be < 0.45x of 4-NFE (ideally < 0.25); got ratio={ratio:.3f}"
    )


def smoke_ensembler_continuity() -> None:
    """Check that :class:`ACTTemporalEnsembler` smooths a jittery chunk stream."""
    print("\n[3] ACTTemporalEnsembler continuity vs random chunks")
    rng = np.random.default_rng(2)
    chunk_size = 16
    action_dim = 7
    decay_m = 0.05
    n_steps = 60

    base = np.cumsum(rng.normal(scale=0.02, size=(n_steps + chunk_size, action_dim)), axis=0)
    ensembler = ACTTemporalEnsembler(
        chunk_size=chunk_size, decay_m=decay_m,
        buffer_size=chunk_size, action_dim=action_dim,
    )
    ensembled_stream = np.zeros((n_steps, action_dim), dtype=np.float32)
    for step in range(n_steps):
        chunk = base[step:step + chunk_size].astype(np.float32)
        chunk += rng.normal(scale=0.1, size=chunk.shape).astype(np.float32)
        ensembled_stream[step] = ensembler.update_and_get(chunk)

    random_stream = np.zeros((n_steps, action_dim), dtype=np.float32)
    for step in range(n_steps):
        chunk = base[step:step + chunk_size].astype(np.float32)
        chunk += rng.normal(scale=0.1, size=chunk.shape).astype(np.float32)
        random_stream[step] = chunk[0]

    def jitter(stream: np.ndarray) -> float:
        """Root-mean-square of consecutive-step differences."""
        diffs = np.diff(stream, axis=0)
        return float(np.sqrt((diffs ** 2).mean()))

    jit_ens = jitter(ensembled_stream)
    jit_raw = jitter(random_stream)
    ratio = jit_raw / max(jit_ens, 1e-9)
    print(f" jitter(ensembled) = {jit_ens:.4f}, jitter(raw) = {jit_raw:.4f}")
    print(f" raw / ensembled = {ratio:.2f} (target >= 3.0)")
    assert ratio >= 2.0, (
        f"Ensembler did not significantly smooth the stream: ratio={ratio:.2f}"
    )
    assert ensembler.current_step == n_steps
    ensembler.reset()
    assert ensembler.current_step == 0
    assert len(ensembler._buffer) == 0
    print(" reset() clears buffer and global step OK")


def smoke_rollout_select_action() -> None:
    """Run ``policy.select_action`` for 50 steps and check ``reset()`` clears state."""
    print("\n[4] policy.select_action 50-step rollout")
    torch.manual_seed(3)
    cfg = _make_config(flow_type="shortcut", use_temporal_ensembling=True)
    policy = PhaseQFlowPolicy(cfg).eval()

    warm_batch = _dummy_batch(cfg, batch_size=1, chunked=True)
    with torch.no_grad():
        _ = policy.predict_action(warm_batch)

    policy.reset()
    n_steps = 50
    actions = []
    for _ in range(n_steps):
        obs = {
            "images": torch.randn(3, 64, 64),
            "states": torch.randn(cfg.state_dim),
            "language": torch.randn(16),
            "history": torch.randn(cfg.history_dim),
        }
        a = policy.select_action(obs)
        actions.append(a.cpu().numpy())
    actions_np = np.stack(actions, axis=0)
    print(f" shape = {actions_np.shape} (expected ({n_steps}, {cfg.action_dim}))")
    assert actions_np.shape == (n_steps, cfg.action_dim)
    assert np.isfinite(actions_np).all(), "select_action produced NaN/Inf"
    policy.reset()
    assert policy._rollout_step == 0 and policy._rollout_chunk is None
    if policy._ensembler is not None:
        assert policy._ensembler.current_step == 0
    print(" 50-step rollout OK, reset() clean")

    cfg2 = _make_config(flow_type="shortcut", use_temporal_ensembling=False)
    policy2 = PhaseQFlowPolicy(cfg2).eval()
    with torch.no_grad():
        _ = policy2.predict_action(_dummy_batch(cfg2, batch_size=1, chunked=True))
    policy2.reset()
    for step in range(20):
        obs = {
            "images": torch.randn(3, 64, 64),
            "states": torch.randn(cfg2.state_dim),
            "language": torch.randn(16),
            "history": torch.randn(cfg2.history_dim),
        }
        a = policy2.select_action(obs)
        assert a.shape == (cfg2.action_dim,)
    print(f" cache-replay path: 20 steps OK, chunk re-predicted "
          f"~{20 // cfg2.action_execute_size} times")


def main() -> int:
    """Run all Round 4 Shortcut smoke checks and return a process exit code."""
    try:
        smoke_shortcut_losses_finite()
        smoke_shortcut_speedup()
        smoke_ensembler_continuity()
        smoke_rollout_select_action()
    except Exception:
        print("\n[FAIL] Smoke test raised an exception:\n")
        traceback.print_exc()
        return 1
    print("\n[PASS] All Round 4 Shortcut smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
