#!/usr/bin/env python

"""Phase-Centric VLA diagnostic entry point.

Measures two core hypotheses:
    H1 Phase-Boundary Loss Gap
        On a trained PhaseQFlow++ policy, the flow-matching loss near
        phase boundaries is significantly higher than in phase interiors.
        We expect ratio >= 2.0 and p < 0.01.

    H2 Misalignment x Failure Correlation
        During rollouts, failed episodes have higher chunk-phase
        misalignment than successful ones. We expect Pearson r <= -0.5
        and p < 0.01.

Design principles:
    1. Do **not** modify ``modeling_phaseqflow.py`` / ``configuration_phaseqflow.py``.
    2. All helper logic lives under ``scripts/diagnostics/diagnostic_utils/``,
       toggled or falling back by config. The script runs in a CPU-only
       environment on synthetic data plus a synthetic environment.
    3. Real experiments require a trained checkpoint, a usable HF
       dataset, and a LIBERO env. When the latter two are missing the
       script automatically falls back to SYNTHETIC mode and flags that
       prominently in the report.

Usage:
    # Real ckpt + real dataset + LIBERO (needs GPU and libero)
    python scripts/diagnostics/diagnostic_phase_centric.py \\
        --policy_path outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model \\
        --dataset_name HuggingFaceVLA/libero \\
        --num_episodes 50 --num_demos 200

    # CPU smoke: allow synthetic demos (pipeline validation)
    python scripts/diagnostics/diagnostic_phase_centric.py \\
        --policy_path outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model \\
        --device cpu --num_demos 5 --num_episodes 4 --allow_synthetic_demos
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from diagnostic_utils.h1_loss import per_timestep_flow_loss # noqa: E402
from diagnostic_utils.h2_rollout import ( # noqa: E402
    correlate_misalignment_and_success,
    rollout_episode,
)
from diagnostic_utils.phase_proxies import compute_boundary_mask # noqa: E402
from diagnostic_utils.report import ( # noqa: E402
    build_report_payload,
    save_h1_figure,
    save_h2_figure,
    verdict_h1,
    verdict_h2,
    write_json_report,
    write_markdown_report,
)
from diagnostic_utils.synthetic_demos import ( # noqa: E402
    DemoSample,
    make_synthetic_demos,
    try_load_real_demos,
)
from diagnostic_utils.synthetic_env import make_fallback_env # noqa: E402

log = logging.getLogger("round1_diagnostic")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _load_policy(policy_path: str, device: str) -> Any:
    """Load a PhaseQFlow policy from a checkpoint directory.

    Handles two failure modes gracefully:
      * Missing directory: raises ``FileNotFoundError`` and lets the
        caller decide on a fallback.
      * ``config.json`` exists but heavy dependencies (timm /
        transformers / vector_quantize_pytorch) are missing: the caller
        should turn the corresponding fields off before ``save_pretrained``
        to a new directory, or use an in-memory config in
        ``scripts/smoke/smoke_test_diagnostic.py``.
    """
    from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig
    from lerobot_policy_phaseqflow.modeling_phaseqflow import PhaseQFlowPolicy

    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"policy_path does not exist: {policy_path}")

    config = PhaseQFlowConfig.from_pretrained(str(path))
    policy = PhaseQFlowPolicy(config=config)
    for name in ("pytorch_model.bin", "model.safetensors"):
        weights_path = path / name
        if weights_path.exists():
            try:
                if weights_path.suffix == ".safetensors":
                    from safetensors.torch import load_file # type: ignore
                    state = load_file(str(weights_path))
                else:
                    state = torch.load(str(weights_path), map_location="cpu")
                policy.load_state_dict(state, strict=False)
                log.info("loaded policy weights from %s", weights_path)
            except Exception as exc: # noqa: BLE001
                log.warning("failed to load weights from %s: %r; continuing with random init",
                            weights_path, exc)
            break
    else:
        log.warning("no weights file found under %s; using random init (H1 will NOT be meaningful)",
                    path)
    policy = policy.to(device).eval()
    return policy


def run_h1(
    policy: Any,
    demos: List[DemoSample],
    phase_proxy: str,
    chunk_len: int,
    num_samples: int,
    dilate: int,
    device: torch.device,
) -> Dict[str, Any]:
    """H1 entry point: compute per-timestep FM loss for each demo and aggregate by boundary mask."""
    from scipy import stats

    boundary_losses: List[float] = []
    interior_losses: List[float] = []

    t_start = time.perf_counter()
    for i, demo in enumerate(demos):
        losses = per_timestep_flow_loss(
            policy=policy,
            demo=demo,
            chunk_len=chunk_len,
            num_samples=num_samples,
            device=device,
        )
        if losses is None:
            log.info("demo #%d skipped (T=%d < chunk_len=%d)", i, len(demo), chunk_len)
            continue
        T_eff = losses.shape[0]
        full_mask = compute_boundary_mask(
            proxy=phase_proxy, actions=demo.actions,
            ee_positions=demo.states[:, :2] if demo.states.shape[-1] >= 2 else None,
            policy=policy, fused_obs_seq=None, dilate=dilate,
        )
        mask = full_mask[:T_eff].astype(bool)
        boundary_losses.extend(losses[mask].tolist())
        interior_losses.extend(losses[~mask].tolist())
        log.info(
            "demo #%d: T_eff=%d boundary=%d interior=%d",
            i, T_eff, int(mask.sum()), int((~mask).sum()),
        )
    elapsed = time.perf_counter() - t_start
    log.info("H1: processed %d demos in %.2fs", len(demos), elapsed)

    bl = np.asarray(boundary_losses, dtype=np.float64)
    il = np.asarray(interior_losses, dtype=np.float64)

    if bl.size < 2 or il.size < 2:
        ratio = float("nan")
        t_stat = float("nan")
        p_value = float("nan")
    else:
        mean_b, mean_i = float(bl.mean()), float(il.mean())
        ratio = mean_b / max(mean_i, 1e-12)
        t_stat, p_value = stats.ttest_ind(bl, il, equal_var=False)
        t_stat, p_value = float(t_stat), float(p_value)

    return {
        "phase_proxy": phase_proxy,
        "chunk_len": int(chunk_len),
        "num_samples_per_t": int(num_samples),
        "n_boundary": int(bl.size),
        "n_interior": int(il.size),
        "boundary_loss_mean": float(bl.mean()) if bl.size else float("nan"),
        "boundary_loss_std": float(bl.std()) if bl.size else float("nan"),
        "interior_loss_mean": float(il.mean()) if il.size else float("nan"),
        "interior_loss_std": float(il.std()) if il.size else float("nan"),
        "ratio": float(ratio),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "verdict": verdict_h1(float(ratio) if np.isfinite(ratio) else 0.0,
                              float(p_value) if np.isfinite(p_value) else 1.0),
        "boundary_losses": boundary_losses,
        "interior_losses": interior_losses,
    }


def run_h2(
    policy: Any,
    env: Any,
    num_episodes: int,
    max_steps: int,
) -> Dict[str, Any]:
    """H2 entry point: run ``num_episodes`` rollouts and aggregate misalignment x success."""
    episodes: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    for ep in range(int(num_episodes)):
        result = rollout_episode(
            env=env, policy=policy, max_steps=max_steps,
        )
        episodes.append(result)
        log.info(
            "episode %d/%d: steps=%d success=%s mis=%.3f replans=%d",
            ep + 1, num_episodes, result["total_steps"], result["success"],
            result["misalignment"], len(result["replan_times"]),
        )
    elapsed = time.perf_counter() - t_start
    log.info("H2: %d episodes in %.2fs", num_episodes, elapsed)

    r, p, mean_suc, mean_fail = correlate_misalignment_and_success(episodes)
    mis = np.array([e["misalignment"] for e in episodes], dtype=np.float64)
    suc = np.array([int(e["success"]) for e in episodes], dtype=np.float64)
    return {
        "num_episodes": int(num_episodes),
        "max_steps": int(max_steps),
        "success_rate": float(suc.mean()) if suc.size else 0.0,
        "mean_misalignment_success": float(mean_suc),
        "mean_misalignment_failure": float(mean_fail),
        "pearson_r": float(r),
        "p_value": float(p),
        "verdict": verdict_h2(
            float(r) if np.isfinite(r) else 0.0,
            float(p) if np.isfinite(p) else 1.0,
        ),
        "misalignments": mis.tolist(),
        "successes": suc.tolist(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Orchestrate the Round 1 diagnostic run end-to-end."""
    parser = argparse.ArgumentParser(description="Phase-Centric VLA Round 1 diagnostic")
    parser.add_argument("--policy_path", required=True,
                        help="Path to a trained PhaseQFlow checkpoint directory (config.json + weights).")
    parser.add_argument("--dataset_name", default="HuggingFaceVLA/libero",
                        help="HuggingFace dataset name for H1 demos (default: HuggingFaceVLA/libero).")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of rollout episodes for H2 (default: 50).")
    parser.add_argument("--num_demos", type=int, default=200,
                        help="Number of demos for H1 (default: 200).")
    parser.add_argument("--output_dir", default="artifacts/diagnostic",
                        help="Directory for report.json / report.md / figures.")
    parser.add_argument("--phase_proxy", choices=("gripper", "velocity_change", "planner_output"),
                        default="gripper",
                        help="Phase-boundary proxy (default: gripper).")
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda).")
    parser.add_argument("--num_samples_h1", type=int, default=3,
                        help="Shortcut FM noise-resamplings per timestep (default: 3).")
    parser.add_argument("--max_steps_h2", type=int, default=200,
                        help="Max timesteps per rollout episode (default: 200).")
    parser.add_argument("--boundary_dilate", type=int, default=2,
                        help="Dilation radius for boundary mask (default: 2).")
    parser.add_argument("--allow_synthetic_demos", action="store_true",
                        help="If the HF dataset cannot be loaded, fall back to synthetic demos.")
    parser.add_argument("--force_synthetic_env", action="store_true",
                        help="Skip LIBERO probing and always use SyntheticLongHorizonEnv.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log.info("device=%s", device)

    policy = _load_policy(args.policy_path, str(device))
    cfg = policy.config
    action_dim = int(cfg.action_dim)
    state_dim = int(cfg.state_dim)
    history_dim = int(cfg.history_dim)
    chunk_len = int(getattr(cfg, "action_chunk_size", 16))

    synthetic_demo = False
    demos = try_load_real_demos(
        dataset_name=args.dataset_name,
        num_demos=int(args.num_demos),
        action_dim=action_dim,
        state_dim=state_dim,
        history_dim=history_dim,
    )
    if demos is None:
        if not args.allow_synthetic_demos:
            log.error(
                "failed to load real dataset %s and --allow_synthetic_demos not set; "
                "aborting. Use --allow_synthetic_demos to smoke-test the pipeline.",
                args.dataset_name,
            )
            return 2
        log.warning("real dataset unavailable; falling back to synthetic demos")
        synthetic_demo = True
        demos = make_synthetic_demos(
            num_demos=int(args.num_demos),
            action_dim=action_dim, state_dim=state_dim, history_dim=history_dim,
        )
    log.info("H1 source: %s, num_demos=%d", "SYNTHETIC" if synthetic_demo else "REAL", len(demos))

    env = None
    synthetic_env = False
    if not args.force_synthetic_env:
        try:
            from libero.libero import benchmark # type: ignore # noqa: F401
            raise NotImplementedError(
                "LIBERO factory not wired up in this template; override by editing this block "
                "or pass --force_synthetic_env to skip probing."
            )
        except Exception as exc: # noqa: BLE001
            log.warning("LIBERO env unavailable (%r); falling back to SyntheticLongHorizonEnv", exc)
            synthetic_env = True
    else:
        synthetic_env = True
    if synthetic_env:
        env = make_fallback_env(
            action_dim=action_dim, state_dim=state_dim, history_dim=history_dim,
            seed=int(args.seed),
        )

    h1 = run_h1(
        policy=policy, demos=demos, phase_proxy=args.phase_proxy,
        chunk_len=chunk_len, num_samples=int(args.num_samples_h1),
        dilate=int(args.boundary_dilate), device=device,
    )

    h2 = run_h2(
        policy=policy, env=env, num_episodes=int(args.num_episodes),
        max_steps=int(args.max_steps_h2),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_h1_figure(
        boundary_losses=np.asarray(h1["boundary_losses"], dtype=np.float64),
        interior_losses=np.asarray(h1["interior_losses"], dtype=np.float64),
        out_path=out_dir / "fig_h1.png",
    )
    save_h2_figure(
        misalignments=np.asarray(h2["misalignments"], dtype=np.float64),
        successes=np.asarray(h2["successes"], dtype=np.float64),
        out_path=out_dir / "fig_h2.png",
        pearson_r=float(h2.get("pearson_r") or 0.0),
    )

    meta = {
        "policy_path": str(args.policy_path),
        "dataset_name": args.dataset_name,
        "phase_proxy": args.phase_proxy,
        "num_demos": len(demos),
        "num_episodes": int(args.num_episodes),
        "device": str(device),
        "flow_type": str(getattr(cfg, "flow_type", "")),
        "action_dim": action_dim,
        "chunk_len": chunk_len,
        "synthetic_demos": bool(synthetic_demo),
        "synthetic_env": bool(synthetic_env),
        "seed": int(args.seed),
    }
    h1_json = {k: v for k, v in h1.items() if k not in ("boundary_losses", "interior_losses")}
    h2_json = {k: v for k, v in h2.items()}
    payload = build_report_payload(h1=h1_json, h2=h2_json, meta=meta)
    write_json_report(payload=payload, out_path=out_dir / "report.json")
    write_markdown_report(
        payload=payload, out_path=out_dir / "report.md",
        synthetic_demo=synthetic_demo, synthetic_env=synthetic_env,
    )
    log.info("wrote report to %s", out_dir.resolve())
    print("\n=== SUMMARY ===")
    print(f"H1 ratio = {h1['ratio']:.4f} (n_b={h1['n_boundary']}, n_i={h1['n_interior']}, p={h1['p_value']:.3g}) "
          f"verdict={h1['verdict']}")
    print(f"H2 pearsonr = {h2['pearson_r']:.4f} (p={h2['p_value']:.3g}, success_rate={h2['success_rate']:.2f}) "
          f"verdict={h2['verdict']}")
    if synthetic_demo or synthetic_env:
        print("\nNOTE: synthetic data path; results are for pipeline validation only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
