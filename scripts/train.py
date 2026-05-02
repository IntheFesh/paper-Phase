#!/usr/bin/env python
"""PACE v2 stage-based training entry point.

Supports the 3-stage curriculum defined in configs/train/:
  01_pretrain_multimodal.yaml  — tokenizer + cross-attention pre-training
  02_train_phase_and_flow.yaml — hierarchical FSQ + boundary-aware flow
  03_finetune_replan.yaml      — calibration-only (PCAR / B-PCAR / Concordance)

Usage
-----
::

    # Stage 1 (3 smoke steps)
    python scripts/train.py --stage configs/train/01_pretrain_multimodal.yaml \\
        --epochs 1 --smoke_mode

    # Stage 2 (real training)
    python scripts/train.py --stage configs/train/02_train_phase_and_flow.yaml \\
        --epochs 50 --device cuda

    # Stage 3 (calibration only, no gradient updates to main net)
    python scripts/train.py --stage configs/train/03_finetune_replan.yaml \\
        --epochs 1 --smoke_mode

Note
----
This script wraps ``scripts/training/train_dummy_batch.py`` for smoke runs and
is designed to remain forward-compatible with the ``lerobot-train`` CLI once
a real dataloader is wired in.

The ``calibration_only: true`` flag in stage 3 skips gradient updates and
delegates to the calibration scripts in ``scripts/calibration/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_SRC = _REPO_ROOT / "lerobot_policy_phaseqflow" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("train")


def _load_stage_cfg(stage_path: str) -> Dict[str, Any]:
    """Load a stage YAML file and return the config dict."""
    p = Path(stage_path)
    if not p.exists():
        # Allow just the stem name, e.g. "01_pretrain_multimodal"
        candidates = list((_REPO_ROOT / "configs" / "train").glob(f"{p.stem}*.yaml"))
        if candidates:
            p = candidates[0]
        else:
            raise FileNotFoundError(f"Stage config not found: {stage_path}")
    if not _HAS_YAML:
        log.warning("PyYAML not installed; stage config parsed as plain text")
        return {"_path": str(p), "_raw": p.read_text()}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _is_calibration_only(stage_cfg: Dict[str, Any]) -> bool:
    """Return True when the stage config sets calibration_only: true."""
    phaseqflow_sec = stage_cfg.get("phaseqflow", {})
    return bool(phaseqflow_sec.get("calibration_only", False))


def _run_smoke_stage(stage_cfg: Dict[str, Any], device: str, steps: int = 3) -> None:
    """Run a 3-step smoke check for one stage using train_dummy_batch."""
    from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig
    from lerobot_policy_phaseqflow.modeling_phaseqflow import PhaseQFlowPolicy
    import torch

    phaseqflow_overrides: Dict[str, Any] = stage_cfg.get("phaseqflow", {})
    stage_name = stage_cfg.get("stage", "unknown")
    log.info("smoke stage=%s device=%s steps=%d", stage_name, device, steps)

    cfg = PhaseQFlowConfig(
        use_dual_backbone_vision=False,
        use_fsq=True,
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
    # Apply stage-specific overrides (skip unknown config fields silently)
    from dataclasses import fields as dc_fields
    known = {f.name for f in dc_fields(cfg)}
    for k, v in phaseqflow_overrides.items():
        if k in known:
            setattr(cfg, k, v)

    calibration_only = bool(phaseqflow_overrides.get("calibration_only", False))

    if calibration_only:
        log.info("[smoke] calibration_only=True — skipping gradient steps")
        log.info("[smoke] stage=%s DONE (calibration-only smoke)", stage_name)
        return

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    policy = PhaseQFlowPolicy(cfg).to(dev).train()
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    B = 2
    Ta = int(cfg.action_chunk_size)
    batch = {
        "obs": {
            "images": torch.randn(B, 3, 64, 64, device=dev),
            "states": torch.randn(B, cfg.state_dim, device=dev),
            "language": torch.randn(B, 16, device=dev),
            "history": torch.randn(B, cfg.history_dim, device=dev),
        },
        "action": torch.randn(B, Ta, cfg.action_dim, device=dev),
        "timestep": torch.zeros(B, dtype=torch.long, device=dev),
    }
    t0 = time.time()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = policy.compute_loss(batch)
        loss.backward()
        opt.step()
        log.info(" step %d/%d loss=%.4f", step, steps, float(loss.item()))
    elapsed = time.time() - t0
    log.info("[smoke] stage=%s DONE (%.2fs, %d steps)", stage_name, elapsed, steps)


_STAGE_TO_MODE = {
    "pretrain_multimodal": "off",
    "train_phase_and_flow": "a",
    "finetune_replan": "pcar",
}


def _delegate_to_trainer(
    stage_cfg: Dict[str, Any],
    *,
    data_root: str,
    max_steps: int,
    device: str,
    seed: int,
    output_dir: str,
    micro_batch: int,
    resume_from_checkpoint: Optional[str],
    extra_args: list,
) -> int:
    """Forward to scripts/training/train_dummy_batch.py with stage-derived flags."""
    import subprocess

    stage_name = stage_cfg.get("stage", "")
    mode = _STAGE_TO_MODE.get(stage_name, "off")

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "training" / "train_dummy_batch.py"),
        "--phase-centric-mode", mode,
        "--steps", str(max_steps),
        "--device", device,
        "--seed", str(seed),
        "--output_dir", output_dir,
        "--micro-batch", str(micro_batch),
        "--data_root", data_root,
        "--enable_diagnostics",
        "--enable_checkpointing",
    ]
    if resume_from_checkpoint:
        cmd += ["--resume_from_checkpoint", resume_from_checkpoint]
    cmd += list(extra_args)

    log.info("delegating to: %s", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="PACE v2 stage-based trainer")
    parser.add_argument("--stage", required=True, help="Path to stage YAML config")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (smoke only)")
    parser.add_argument("--device", default="cpu", help="Training device")
    parser.add_argument("--smoke_mode", action="store_true",
                        help="Run 3-step smoke check and exit")
    parser.add_argument("--steps", type=int, default=3,
                        help="Steps per epoch (smoke mode only)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="LeRobotDataset path or HuggingFace repo-id")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Total training steps (full mode)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where artifacts (checkpoints, eval, diagnostics) land")
    parser.add_argument("--micro_batch", type=int, default=32)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args, extra = parser.parse_known_args()

    stage_cfg = _load_stage_cfg(args.stage)
    stage_name = stage_cfg.get("stage", Path(args.stage).stem)
    log.info("Loaded stage config: %s (stage=%s)", args.stage, stage_name)

    if args.smoke_mode:
        _run_smoke_stage(stage_cfg, device=args.device, steps=args.steps)
        print(f" [OK] stage={stage_name}")
        return

    if _is_calibration_only(stage_cfg):
        log.info("calibration_only=True — running calibration scripts")
        calibrate_list = stage_cfg.get("calibrate", [])
        for module in calibrate_list:
            log.info("  calibrating: %s (placeholder — Phase C fill-in)", module)
        log.info("Calibration stage complete.")
        return

    if args.data_root is None or args.max_steps is None or args.output_dir is None:
        log.error(
            "Full training requires --data_root, --max_steps, and --output_dir. "
            "Use --smoke_mode for a quick CPU sanity check."
        )
        sys.exit(2)

    rc = _delegate_to_trainer(
        stage_cfg,
        data_root=args.data_root,
        max_steps=int(args.max_steps),
        device=args.device,
        seed=int(args.seed),
        output_dir=args.output_dir,
        micro_batch=int(args.micro_batch),
        resume_from_checkpoint=args.resume_from_checkpoint,
        extra_args=extra,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
