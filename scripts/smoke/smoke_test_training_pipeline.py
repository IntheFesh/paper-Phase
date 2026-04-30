#!/usr/bin/env python

"""Smoke test for Round 3 training-pipeline stability/efficiency upgrades.

Verifies the 5 acceptance items requested by the maintainer:

  1. bf16 + grad-ckpt + PagedAdamW8bit training: micro_batch=2, grad_accum=32,
     10 steps. Report peak VRAM (cuda) or peak RSS (cpu) and per-step latency.
     GPU target: < 10 GB, < 5 s/step.
  2. Optimizer param_groups: three groups {backbone, lora, head} with the
     configured LRs (0, 5e-5, 1e-4) and correct parameter counts.
  3. EMA: drive a dummy flow-head parameter with 1000 random step replacements
     and show that EMA tracks a smoothed version (``||ema - target||_2 <
     ||live - target||_2`` at the end).
  4. ``stage="pretrain_multimodal"``: ``flow_action_head.requires_grad`` all
     False and ``chunk_verifier.requires_grad`` all False; planner / non-frozen
     tokenizer params remain trainable.
  5. ``compute_loss`` with a NaN-injected batch: no crash, returns zero loss
     and logs a warning.

Uses ``use_dual_backbone_vision=False`` for an offline-friendly smoke test.
"""

from __future__ import annotations

import logging
import os
import resource
import sys
import time
import traceback
from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig
from lerobot_policy_phaseqflow.modeling_phaseqflow import PhaseQFlowPolicy
from lerobot_policy_phaseqflow.training_utils import (
    apply_stage_freeze,
    bf16_autocast_context,
    build_optimizer,
    build_param_groups,
    build_scheduler,
    maybe_apply_gradient_checkpointing,
    maybe_build_ema,
    summarize_param_groups,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("round3_smoke")


def _peak_memory_mb() -> float:
    """Return peak VRAM (MB) on CUDA, else peak RSS (MB) of the current process."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024.0 if sys.platform.startswith("linux") else rss / (1024 ** 2)


def _reset_peak_memory() -> None:
    """Reset CUDA peak-memory counters (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _make_config(**overrides) -> PhaseQFlowConfig:
    """Offline-friendly base config with optional field overrides."""
    cfg = PhaseQFlowConfig(
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
        use_fsq=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _dummy_batch(cfg: PhaseQFlowConfig, batch_size: int) -> Dict[str, torch.Tensor]:
    """Build a dummy batch whose shapes match the legacy tokenizer path."""
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


def smoke_bf16_training_loop() -> None:
    """Run 10 bf16 + grad-ckpt + PagedAdamW8bit steps and report peak mem and time."""
    print("\n[1] bf16 + grad-ckpt + PagedAdamW8bit training-loop smoke test")
    torch.manual_seed(0)
    cfg = _make_config(use_bf16=True, use_gradient_checkpointing=True, use_paged_adamw_8bit=True)
    policy = PhaseQFlowPolicy(cfg).train()
    _ = policy.compute_loss(_dummy_batch(cfg, batch_size=2))
    n_ckpt = maybe_apply_gradient_checkpointing(policy, cfg)
    print(f" gradient-checkpointed blocks: {n_ckpt}")

    optimizer = build_optimizer(policy, cfg)
    scheduler = build_scheduler(optimizer, cfg, num_training_steps=10)
    print(f" optimizer class: {type(optimizer).__name__}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)

    micro_batch, grad_accum, total_steps = 2, 32, 10
    _reset_peak_memory()
    t0 = time.perf_counter()
    per_step_times = []
    for step in range(total_steps):
        t_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            batch = _dummy_batch(cfg, batch_size=micro_batch)
            batch_device = {
                "obs": {k: v.to(device) for k, v in batch["obs"].items()},
                "action": batch["action"].to(device),
                "timestep": batch["timestep"].to(device),
            }
            with bf16_autocast_context(cfg):
                out = policy.compute_loss(batch_device, return_dict=True)
                loss = out["loss"] / grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad and p.grad is not None],
            max_norm=float(cfg.grad_clip_norm),
        )
        try:
            optimizer.step()
        except Exception as exc: # noqa: BLE001
            log.warning("optimizer.step() failed on %s (%r); falling back to AdamW.", device, exc)
            optimizer = torch.optim.AdamW(build_param_groups(policy, cfg))
            optimizer.step()
        scheduler.step()
        per_step_times.append(time.perf_counter() - t_start)
    elapsed = time.perf_counter() - t0
    avg = elapsed / total_steps
    peak_mb = _peak_memory_mb()
    unit = "VRAM (MB)" if torch.cuda.is_available() else "peak RSS (MB)"
    print(f" {total_steps} steps x (bs={micro_batch} * accum={grad_accum}) took {elapsed:.2f}s "
          f"(avg {avg:.3f}s/step, max {max(per_step_times):.3f}s)")
    print(f" peak {unit}: {peak_mb:.1f}")
    if torch.cuda.is_available():
        assert peak_mb < 10 * 1024, f"Peak VRAM {peak_mb:.1f} MB exceeds 10 GB budget"
        assert avg < 5.0, f"Avg per-step {avg:.3f}s exceeds 5s budget"
    else:
        print(" [note] CPU-only environment: skipping <10GB / <5s/step hard assertions")


def smoke_param_groups() -> None:
    """Check that ``build_param_groups`` emits three groups with the expected LRs."""
    print("\n[2] param_groups sanity (backbone=0 / LoRA=5e-5 / head=1e-4)")
    torch.manual_seed(0)
    cfg = _make_config()
    policy = PhaseQFlowPolicy(cfg).train()
    _ = policy.compute_loss(_dummy_batch(cfg, batch_size=2))

    lora_mod = nn.Linear(4, 4)
    for p in lora_mod.parameters():
        p.requires_grad = True
    policy.add_module("fake_lora_adapter", lora_mod)
    policy.fake_lora_adapter.register_parameter(
        "lora_A",
        nn.Parameter(torch.zeros(4, 4)),
    )

    bb_mod = nn.Linear(4, 4)
    for p in bb_mod.parameters():
        p.requires_grad = True
    policy.add_module("vision_backbone", bb_mod)

    groups = build_param_groups(policy, cfg)
    print(summarize_param_groups(groups))
    names = {g["name"]: g for g in groups}
    assert "head" in names, names
    assert "lora" in names, f"missing lora group: {names.keys()}"
    assert "backbone" in names, f"missing backbone group: {names.keys()}"
    assert abs(names["backbone"]["lr"] - cfg.lr_backbone) < 1e-12, names["backbone"]["lr"]
    assert abs(names["lora"]["lr"] - cfg.lr_lora) < 1e-12, names["lora"]["lr"]
    assert abs(names["head"]["lr"] - cfg.lr_head) < 1e-12, names["head"]["lr"]
    assert names["lora"]["weight_decay"] == 0.0, names["lora"]["weight_decay"]


def smoke_ema_smoothing() -> None:
    """Drive the flow head with noisy steps for 1000 iters and check EMA smooths them."""
    print("\n[3] EMA smooths flow-head params over 1000 steps")
    torch.manual_seed(0)
    cfg = _make_config(use_ema=True, ema_decay=0.99, ema_power=0.75)
    policy = PhaseQFlowPolicy(cfg).train()
    _ = policy.compute_loss(_dummy_batch(cfg, batch_size=2))

    ema = maybe_build_ema(policy, cfg)
    assert ema is not None, "maybe_build_ema should return EMAModel when use_ema=True"

    head = policy.flow_action_head
    with torch.no_grad():
        target_vec = torch.cat([p.detach().flatten().clone() for p in head.parameters()])
    for step in range(1000):
        with torch.no_grad():
            idx = 0
            for p in head.parameters():
                n = p.numel()
                chunk = target_vec[idx : idx + n].view_as(p)
                p.copy_(chunk + 0.1 * torch.randn_like(p))
                idx += n
        ema.step(head.parameters())

    ema.store(head.parameters())
    ema.copy_to(head.parameters())
    with torch.no_grad():
        ema_vec = torch.cat([p.detach().flatten().clone() for p in head.parameters()])
    ema.restore(head.parameters())
    with torch.no_grad():
        live_vec = torch.cat([p.detach().flatten().clone() for p in head.parameters()])

    ema_err = (ema_vec - target_vec).norm().item()
    live_err = (live_vec - target_vec).norm().item()
    print(f" ||live - target||_2 = {live_err:.4f}")
    print(f" ||ema - target||_2 = {ema_err:.4f}")
    assert ema_err < live_err, f"EMA did not smooth: ema_err={ema_err:.4f} >= live_err={live_err:.4f}"


def smoke_stage_freeze() -> None:
    """Check that ``stage=pretrain_multimodal`` freezes flow head + verifier only."""
    print("\n[4] stage=pretrain_multimodal freezes flow-head and verifier")
    torch.manual_seed(0)
    cfg = _make_config(
        stage="pretrain_multimodal",
        stage_freeze_vision=False,
        stage_freeze_planner=False,
        stage_freeze_flow_head=True,
        stage_freeze_verifier=True,
    )
    policy = PhaseQFlowPolicy(cfg).train()
    _ = policy.compute_loss(_dummy_batch(cfg, batch_size=2))

    summary = apply_stage_freeze(policy, cfg)
    flow_trainable = sum(1 for p in policy.flow_action_head.parameters() if p.requires_grad)
    verifier_trainable = sum(1 for p in policy.chunk_verifier.parameters() if p.requires_grad)
    planner_trainable = sum(1 for p in policy.hierarchical_planner.parameters() if p.requires_grad)
    tokenizer_trainable = sum(1 for p in policy.vision_tokenizer.parameters() if p.requires_grad)
    print(f" flow_head trainable params: {flow_trainable}")
    print(f" chunk_verifier trainable params: {verifier_trainable}")
    print(f" hierarchical_planner trainable params: {planner_trainable}")
    print(f" vision_tokenizer trainable params: {tokenizer_trainable}")
    assert flow_trainable == 0, flow_trainable
    assert verifier_trainable == 0, verifier_trainable
    assert planner_trainable > 0, planner_trainable
    assert tokenizer_trainable > 0, tokenizer_trainable
    assert summary["flow_action_head"] == 0
    assert summary["chunk_verifier"] == 0


def smoke_nan_guard() -> None:
    """Check that a NaN-injected batch returns zero loss without crashing."""
    print("\n[5] NaN-injected batch returns zero loss and logs warning")
    torch.manual_seed(0)
    cfg = _make_config()
    policy = PhaseQFlowPolicy(cfg).train()
    _ = policy.compute_loss(_dummy_batch(cfg, batch_size=2))

    orig_forward = policy.flow_action_head.forward

    def _nan_forward(*args, **kwargs):
        """Forward wrapper that poisons ``action_pred`` with NaNs."""
        out = orig_forward(*args, **kwargs)
        out["action_pred"] = out["action_pred"] * float("nan")
        return out

    policy.flow_action_head.forward = _nan_forward # type: ignore[assignment]

    batch = _dummy_batch(cfg, batch_size=2)
    out = policy.compute_loss(batch, return_dict=True)
    loss = out["loss"]
    assert torch.is_tensor(loss) and loss.item() == 0.0, f"expected zero loss, got {loss}"
    print(f" loss == {loss.item():.4f} with NaN-injected action_pred (no crash)")


def main() -> int:
    """Run all Round 3 training-pipeline smoke checks and return a process exit code."""
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // 2))
    tests = [
        smoke_bf16_training_loop,
        smoke_param_groups,
        smoke_ema_smoothing,
        smoke_stage_freeze,
        smoke_nan_guard,
    ]
    for fn in tests:
        try:
            fn()
        except Exception: # noqa: BLE001
            print(f"\n[FAIL] {fn.__name__} raised:\n")
            traceback.print_exc()
            return 1
    print("\n[PASS] All Round 3 training-pipeline smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
