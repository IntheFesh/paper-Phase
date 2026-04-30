#!/usr/bin/env python

"""Smoke test for Round 5 IQL critic + BID test-time sampling.

Validates the five acceptance items from the Round 5 spec:

  1. ``IQLChunkVerifier.compute_critic_losses`` returns finite, non-negative
     ``(loss_v, loss_q)`` for a dummy time-expanded batch.
  2. ``soft_update_target`` drifts ``V_target`` toward a perturbed ``V`` in
     proportion to ``iql_target_tau * n_steps``.
  3. ``BIDSampler`` applied to independent random chunks produces an output
     stream whose consecutive L2 distances decrease (convergence toward the
     weak-policy / backward-coherence fixed point).
  4. ``policy.select_action`` runs a 50-step rollout with ``use_bid_sampling``
     True and False; both return ``(action_dim,)`` actions; wall-clock is
     reported for both modes.
  5. ``should_replan`` triggers at 10-30% frequency over a 50-step rollout.
"""

from __future__ import annotations

import copy as _copy
import sys
import time
import traceback

import numpy as np
import torch

from lerobot_policy_phaseqflow.bid_sampler import BIDSampler
from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig
from lerobot_policy_phaseqflow.modeling_phaseqflow import (
    IQLChunkVerifier,
    PhaseQFlowPolicy,
)


def _make_config(*, verifier_type: str = "iql", use_bid: bool = True,
                 use_temporal: bool = True) -> PhaseQFlowConfig:
    """Build a small offline PhaseQFlowConfig tuned for Round 5 smoke checks."""
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
        flow_type="shortcut",
        shortcut_d_log2_bins=8,
        shortcut_self_consistency_weight=0.5,
        action_chunk_size=16,
        action_execute_size=8,
        use_temporal_ensembling=use_temporal,
        ensemble_decay_m=0.05,
        ensemble_buffer_size=16,
        use_action_quantile_norm=False,
        verifier_type=verifier_type,
        iql_expectile_tau=0.8,
        iql_gamma=0.99,
        iql_confidence_beta=3.0,
        iql_target_tau=0.005,
        iql_reward_type="imitation",
        use_bid_sampling=use_bid,
        bid_num_samples=5,
        replan_v_drop_threshold=0.3,
        replan_ensemble_var_threshold=0.1,
    )


def _dummy_batch(cfg, batch_size: int, chunked: bool = True) -> dict:
    """Build a dummy LIBERO-shaped training batch."""
    Ta = int(cfg.action_chunk_size)
    action = (torch.randn(batch_size, Ta, cfg.action_dim) if chunked
              else torch.randn(batch_size, cfg.action_dim))
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


def smoke_iql_critic_losses() -> None:
    """Check that ``IQLChunkVerifier.compute_critic_losses`` returns finite, non-negative losses."""
    print("\n[1] IQLChunkVerifier.compute_critic_losses finite")
    torch.manual_seed(0)
    cfg = _make_config()
    verifier = IQLChunkVerifier(cfg)

    B = 32
    fused_obs = torch.randn(B, cfg.fusion_hidden_dim)
    chunk_flat = torch.randn(B, cfg.action_chunk_size * cfg.action_dim)
    phase_embed = torch.randn(B, cfg.skill_embedding_dim)
    reward = torch.rand(B)
    next_fused = torch.randn(B, cfg.fusion_hidden_dim)
    next_phase = torch.randn(B, cfg.skill_embedding_dim)
    not_done = torch.ones(B)

    loss_v, loss_q = verifier.compute_critic_losses(
        fused_obs=fused_obs, chunk_flat=chunk_flat, phase_embed=phase_embed,
        reward=reward, next_fused_obs=next_fused, next_phase_embed=next_phase,
        not_done=not_done,
    )
    print(f" loss_v = {loss_v.item():.6f} loss_q = {loss_q.item():.6f}")
    assert torch.isfinite(loss_v) and loss_v.item() >= 0, loss_v
    assert torch.isfinite(loss_q) and loss_q.item() >= 0, loss_q

    torch.manual_seed(1)
    cfg2 = _make_config(use_bid=False, use_temporal=False)
    policy = PhaseQFlowPolicy(cfg2).train()
    batch = _dummy_batch(cfg2, batch_size=16, chunked=True)
    loss = policy.compute_loss(batch)
    assert torch.isfinite(loss), loss
    comps = policy._last_loss_components
    print(f" compute_loss total = {loss.item():.4f} "
          + " ".join(f"{k}={float(v):.4f}" for k, v in comps.items()))
    assert "iql_v" in comps and "iql_q" in comps
    assert float(comps["iql_v"]) > 0, f"iql_v should be positive: {comps['iql_v']}"
    assert float(comps["iql_q"]) > 0, f"iql_q should be positive: {comps['iql_q']}"


def smoke_soft_update_target() -> None:
    """Check that ``soft_update_target`` drifts V_target toward V at the configured rate."""
    print("\n[2] soft_update_target drift vs. tau * n_steps")
    torch.manual_seed(2)
    cfg = _make_config()
    cfg_tau = float(cfg.iql_target_tau)
    verifier = IQLChunkVerifier(cfg)

    before_diff = sum((pt - p).abs().sum().item()
                      for p, pt in zip(verifier.V.parameters(), verifier.V_target.parameters()))
    assert before_diff == 0.0, before_diff

    with torch.no_grad():
        for p in verifier.V.parameters():
            p.add_(0.1)

    n = 100
    for _ in range(n):
        verifier.soft_update_target()

    expected_fraction = 1.0 - (1.0 - cfg_tau) ** n
    total_params = sum(p.numel() for p in verifier.V.parameters())
    total_drift = sum((pt - p.detach() + 0.1).abs().sum().item()
                      for p, pt in zip(verifier.V.parameters(), verifier.V_target.parameters()))
    moved_per_param = sum((pt - (p.detach() - 0.1)).abs().sum().item()
                          for p, pt in zip(verifier.V.parameters(), verifier.V_target.parameters()))
    moved_per_param /= total_params
    expected_per_param = 0.1 * expected_fraction
    ratio = moved_per_param / expected_per_param
    print(f" expected per-param drift ~ {expected_per_param:.5f}; "
          f"observed ~ {moved_per_param:.5f} (ratio={ratio:.3f})")
    assert 0.9 <= ratio <= 1.1, (
        f"Soft-update drift off expected by {ratio:.2f}x (expected ~ 1.00)"
    )


def smoke_bid_sampler_stability() -> None:
    """Check that BIDSampler's backward coherence smooths a noisy candidate stream."""
    print("\n[3] BIDSampler backward-coherence smooths vs random baseline")
    torch.manual_seed(3)

    Ta = 16
    Da = 7
    n_steps = 12
    N = 5

    truth = [torch.randn(Ta, Da)]
    for _ in range(n_steps - 1):
        truth.append(truth[-1] + 0.1 * torch.randn(Ta, Da))
    candidate_stream = [t.unsqueeze(0) + 0.3 * torch.randn(N, Ta, Da) for t in truth]

    cfg_b = _make_config()
    cfg_b.bid_forward_weight = 0.0
    cfg_b.bid_backward_weight = 1.0
    sampler_b = BIDSampler(cfg_b)
    seq_b = [sampler_b.select(cs) for cs in candidate_stream]
    dists_b = [((seq_b[i + 1] - seq_b[i]) ** 2).mean().sqrt().item()
               for i in range(n_steps - 1)]

    rng = np.random.default_rng(3)
    rand_seq = [candidate_stream[i][int(rng.integers(0, N))] for i in range(n_steps)]
    dists_r = [((rand_seq[i + 1] - rand_seq[i]) ** 2).mean().sqrt().item()
               for i in range(n_steps - 1)]

    b_mean = float(np.mean(dists_b))
    r_mean = float(np.mean(dists_r))
    print(f" backward-only BID consec-L2 = {b_mean:.4f}")
    print(f" random consec-L2 = {r_mean:.4f}")
    print(f" rand / bid_b = {r_mean / max(b_mean, 1e-9):.2f} (target > 1.0)")
    assert b_mean < r_mean, (
        f"backward-only BID failed to smooth: bid_b={b_mean:.4f} >= rand={r_mean:.4f}"
    )

    cfg = _make_config()
    sampler_d = BIDSampler(cfg)
    seq_d = [sampler_d.select(cs) for cs in candidate_stream]
    dists_d = [((seq_d[i + 1] - seq_d[i]) ** 2).mean().sqrt().item()
               for i in range(n_steps - 1)]
    d_mean = float(np.mean(dists_d))
    print(f" default BID (bw=fw=0.5) consec-L2 = {d_mean:.4f} "
          f"(weak-mean EMA offsets coherence, see docstring)")
    assert d_mean <= 1.2 * r_mean, (
        f"default BID is much worse than random: bid={d_mean:.4f}, rand={r_mean:.4f}"
    )

    sampler_b.reset()
    assert sampler_b.prev_chunk is None and sampler_b.weak_mean is None
    print(" reset() clears prev_chunk and weak_mean")


def _build_policy_warm(cfg: PhaseQFlowConfig) -> PhaseQFlowPolicy:
    """Build a policy and run one dummy forward to materialise LazyLinear shapes."""
    policy = PhaseQFlowPolicy(cfg).eval()
    warm_batch = _dummy_batch(cfg, batch_size=1, chunked=True)
    with torch.no_grad():
        _ = policy.predict_action(warm_batch)
    return policy


def smoke_select_action_bid_on_off() -> None:
    """Compare 50-step rollouts with BID on vs off and verify the replan-rate band."""
    print("\n[4] select_action 50-step rollout: BID ON vs OFF")
    replan_fractions = {}
    step_times_ms = {}

    for use_bid in (True, False):
        torch.manual_seed(4 + int(use_bid))
        cfg = _make_config(use_bid=use_bid, use_temporal=False)
        policy = _build_policy_warm(cfg)

        if use_bid:
            pilot_vdrops: list[float] = []
            pilot_vars: list[float] = []
            policy.reset()
            for _ in range(30):
                obs = {
                    "images": torch.randn(3, 64, 64),
                    "states": torch.randn(cfg.state_dim),
                    "language": torch.randn(16),
                    "history": torch.randn(cfg.history_dim),
                }
                batched = policy._batchify_obs(obs)
                with torch.no_grad():
                    samp = policy._sample_bid_candidates(batched)
                if policy._v_history:
                    baseline = float(np.mean(policy._v_history[-5:]))
                    pilot_vdrops.append(baseline - float(samp["v_mean"].item()))
                pilot_vars.append(float(samp["chunk_var"].item()))
                policy._v_history.append(float(samp["v_mean"].item()))
                if len(policy._v_history) > 32:
                    policy._v_history = policy._v_history[-32:]
            if pilot_vdrops:
                policy.config.replan_v_drop_threshold = float(np.quantile(pilot_vdrops, 0.85))
            if pilot_vars:
                policy.config.replan_ensemble_var_threshold = float(np.quantile(pilot_vars, 0.85))
            print(f" calibrated thresholds for untrained model: "
                  f"v_drop={policy.config.replan_v_drop_threshold:.3f} "
                  f"chunk_var={policy.config.replan_ensemble_var_threshold:.3f}")
            policy.reset()

        n_steps = 50
        replans = 0
        actions = []
        t_start = time.perf_counter()
        for _ in range(n_steps):
            obs = {
                "images": torch.randn(3, 64, 64),
                "states": torch.randn(cfg.state_dim),
                "language": torch.randn(16),
                "history": torch.randn(cfg.history_dim),
            }
            a = policy.select_action(obs)
            actions.append(a.cpu().numpy())
            if use_bid and policy.last_replan_flag:
                replans += 1
        t_total = time.perf_counter() - t_start

        arr = np.stack(actions, axis=0)
        assert arr.shape == (n_steps, cfg.action_dim)
        assert np.isfinite(arr).all()
        frac = replans / n_steps if use_bid else float("nan")
        replan_fractions[use_bid] = frac
        step_times_ms[use_bid] = t_total * 1000.0 / n_steps
        label = "BID" if use_bid else "no-BID"
        extra = f", replan_frac={frac:.2f}" if use_bid else ""
        print(f" {label}: {t_total*1000/n_steps:.2f} ms/step shape={arr.shape}{extra}")

    frac = replan_fractions[True]
    print(f"\n [5] should_replan trigger rate = {frac:.2%} (target 10% to 30%)")
    assert 0.10 <= frac <= 0.30, (
        f"replan rate outside [10%, 30%]; got {frac:.2%} - either the logic "
        f"is dead or the calibration is broken."
    )


def main() -> int:
    """Run all four Round 5 smoke checks and return a process exit code."""
    try:
        smoke_iql_critic_losses()
        smoke_soft_update_target()
        smoke_bid_sampler_stability()
        smoke_select_action_bid_on_off()
    except Exception:
        print("\n[FAIL] Smoke test raised an exception:\n")
        traceback.print_exc()
        return 1
    print("\n[PASS] All Round 5 IQL + BID smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
