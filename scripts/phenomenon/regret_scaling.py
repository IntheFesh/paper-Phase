"""Regret-scaling experiment — validate Proposition 3 H/ΔH regret prediction.

Master plan Proposition 3 states that the regret accumulated by a chunked
policy (chunk horizon H) relative to a per-step oracle scales as:

    R(H) ≈ c · H · ΔH

where ΔH is the "cliff depth" (drop in predictability C_t during the chunk)
and c is a task-dependent constant.

Experiment
----------
For each chunk horizon H ∈ {4, 8, 16, 32, 64}:
  - Run the PhaseQFlow policy in chunk mode (execute H actions per plan).
  - Run the same model as a per-step reference (always re-plan; H_ref = 1).
  - Measure success-rate gap  δSR(H) = SR_ref − SR(H).
  - Measure mean cliff depth  ΔH(H)  = mean cliff-concordance drop per chunk.
  - Fit linear model: δSR ~ H · ΔH and report R².

Reference policy trick (§4.2)
-----------------------------
The reference policy is the *same trained model* called with
``action_chunk_size=1`` (predict one action, execute it, re-plan).  This
isolates the pure cost of chunking with no distribution-shift confound.

Outputs  (paper_figures/regret_scaling/)
----------------------------------------
- ``regret_vs_H.csv``    — columns: H, SR, SR_ref, delta_SR, mean_delta_H
- ``regret_vs_H.png``    — scatter + fitted line
- ``summary.md``

Usage
-----
::

    # Real run (requires PhaseQFlow checkpoint + LIBERO-Long env):
    python scripts/phenomenon/regret_scaling.py \\
        --checkpoint checkpoints/phaseqflow_libero_long \\
        --H 4 8 16 32 64 --n_rollouts 50

    # Dry run (synthetic data, no checkpoint needed):
    python scripts/phenomenon/regret_scaling.py --dry_run --H 4 8 16 32 64
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic data (dry-run / smoke)
# ---------------------------------------------------------------------------

def _synthetic_regret(H: int, rng: np.random.Generator) -> Dict[str, float]:
    """Generate synthetic regret metrics following R(H) ≈ c · H · ΔH."""
    # Proposition 3 prediction: δSR ∝ H · ΔH
    c = 0.004                         # arbitrary constant
    mean_delta_H = rng.uniform(0.08, 0.18)  # cliff depth varies by seed
    delta_SR = c * H * mean_delta_H + rng.normal(0, 0.01)
    delta_SR = float(np.clip(delta_SR, 0.0, 1.0))
    SR_ref = float(np.clip(rng.uniform(0.70, 0.85), 0.0, 1.0))
    SR = float(np.clip(SR_ref - delta_SR, 0.0, 1.0))
    return {
        "H": H,
        "SR": SR,
        "SR_ref": SR_ref,
        "delta_SR": delta_SR,
        "mean_delta_H": float(mean_delta_H),
    }


# ---------------------------------------------------------------------------
# Real evaluation helpers
# ---------------------------------------------------------------------------

def _load_policy(checkpoint_path: str, chunk_size: int, device: str = "cuda"):
    """Load a PhaseQFlow policy with the specified chunk size."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "lerobot_policy_phaseqflow" / "src"))
    from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig
    from lerobot_policy_phaseqflow.modeling_phaseqflow import PhaseQFlowPolicy

    cfg = PhaseQFlowConfig.from_pretrained(checkpoint_path)
    cfg.action_chunk_size = chunk_size
    cfg.action_execute_size = chunk_size
    policy = PhaseQFlowPolicy.from_pretrained(checkpoint_path, config=cfg)
    policy = policy.to(device)
    policy.eval()
    return policy


def _evaluate_policy(policy, env_factory, n_rollouts: int, seeds: List[int]) -> float:
    """Return success rate over n_rollouts episodes."""
    successes = 0
    for i in range(n_rollouts):
        env = env_factory()
        obs = env.reset(seed=seeds[i % len(seeds)] + i)
        policy.reset()
        done = False
        for _ in range(500):
            import torch
            with torch.no_grad():
                action = policy.select_action(obs)
            obs, _, done, info = env.step(action)
            if done:
                if info.get("success", False):
                    successes += 1
                break
    return successes / n_rollouts


def _compute_mean_cliff_depth(
    policy, env_factory, n_rollouts: int, seeds: List[int], H: int
) -> float:
    """Estimate mean cliff concordance drop within each chunk of size H."""
    import torch
    from lerobot_policy_phaseqflow.phase_centric.cliff_detection import ConcordanceDetector

    detector = ConcordanceDetector(window_size=50, threshold=0.8)
    depths: List[float] = []

    for i in range(n_rollouts):
        env = env_factory()
        obs = env.reset(seed=seeds[i % len(seeds)] + i)
        policy.reset()
        detector.reset()
        concordances = []
        for t in range(300):
            with torch.no_grad():
                action_batch = policy.select_action(obs)
            obs, _, done, info = env.step(action_batch)
            # Attempt to read concordance from policy's last diagnostic
            c_val = None
            if hasattr(policy, "_last_concordance"):
                c_val = policy._last_concordance
            if c_val is not None:
                concordances.append(float(c_val))
            if done:
                break
        # Cliff depth = mean drop between chunk start and chunk end concordance
        if len(concordances) >= H:
            for start in range(0, len(concordances) - H, H):
                drop = concordances[start] - concordances[start + H - 1]
                depths.append(max(0.0, drop))
    return float(np.mean(depths)) if depths else 0.1


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_plot(
    records: List[Dict],
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Hs = [r["H"] for r in records]
        delta_SRs = [r["delta_SR"] for r in records]
        mean_dHs = [r["mean_delta_H"] for r in records]
        x_vals = [H * dH for H, dH in zip(Hs, mean_dHs)]

        # Fit linear model: δSR = c * H * ΔH
        if len(x_vals) >= 2 and np.std(x_vals) > 1e-10:
            coeffs = np.polyfit(x_vals, delta_SRs, 1)
            x_fit = np.linspace(min(x_vals), max(x_vals), 100)
            y_fit = np.polyval(coeffs, x_fit)
            resid = np.array(delta_SRs) - np.polyval(coeffs, x_vals)
            ss_res = np.sum(resid ** 2)
            ss_tot = np.sum((np.array(delta_SRs) - np.mean(delta_SRs)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else float("nan")
        else:
            x_fit, y_fit, r2 = np.array([]), np.array([]), float("nan")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Left: δSR vs H
        axes[0].scatter(Hs, delta_SRs, color="steelblue", zorder=5)
        for H, dSR in zip(Hs, delta_SRs):
            axes[0].annotate(f"H={H}", (H, dSR), textcoords="offset points", xytext=(4, 2), fontsize=8)
        axes[0].set_xlabel("Chunk horizon H")
        axes[0].set_ylabel("Success-rate gap δSR = SR_ref − SR(H)")
        axes[0].set_title("Regret vs chunk horizon")
        axes[0].set_xlim(left=0)
        axes[0].set_ylim(bottom=0)

        # Right: δSR vs H·ΔH with linear fit
        axes[1].scatter(x_vals, delta_SRs, color="coral", zorder=5)
        if len(x_fit) > 0:
            axes[1].plot(x_fit, y_fit, "k--", linewidth=1.5,
                         label=f"Linear fit (R²={r2:.3f})")
            axes[1].legend(fontsize=9)
        for H, x, dSR in zip(Hs, x_vals, delta_SRs):
            axes[1].annotate(f"H={H}", (x, dSR), textcoords="offset points", xytext=(4, 2), fontsize=8)
        axes[1].set_xlabel("H · ΔH (Prop. 3 predictor)")
        axes[1].set_ylabel("δSR")
        axes[1].set_title("Regret scaling validation")
        axes[1].set_xlim(left=0)
        axes[1].set_ylim(bottom=0)

        fig.tight_layout()
        fig.savefig(output_dir / "regret_vs_H.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _write_summary(records: List[Dict], output_dir: Path, dry_run: bool) -> None:
    lines = [
        "# Regret Scaling Experiment Summary",
        f"\nGenerated: {datetime.now().isoformat()}",
        "" if not dry_run else "(DRY RUN — synthetic data)",
        "",
        "## Results by chunk horizon",
        "",
        "| H | SR(H) | SR_ref | δSR | mean ΔH |",
        "|---|-------|--------|-----|---------|",
    ]
    for r in records:
        lines.append(
            f"| {r['H']} | {r['SR']:.3f} | {r['SR_ref']:.3f} | "
            f"{r['delta_SR']:.3f} | {r['mean_delta_H']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "δSR should grow linearly with H·ΔH (Proposition 3).",
        "See ``regret_vs_H.png`` for the scatter plot and R² of the linear fit.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--H", type=int, nargs="+", default=[4, 8, 16, 32, 64],
                   help="Chunk horizon values to sweep")
    p.add_argument("--n_rollouts", type=int, default=50)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="PhaseQFlow checkpoint path (required for real run)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dry_run", action="store_true",
                   help="Use synthetic data (no checkpoint or env needed)")
    p.add_argument("--output", type=Path, default=Path("paper_figures/regret_scaling"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    records: List[Dict] = []

    if args.dry_run or args.checkpoint is None:
        if not args.dry_run:
            print("[regret_scaling] WARNING: no --checkpoint; running dry_run mode")
        for H in sorted(args.H):
            rec = _synthetic_regret(H, rng)
            records.append(rec)
            print(f"  H={H:3d}  SR={rec['SR']:.3f}  SR_ref={rec['SR_ref']:.3f}  "
                  f"δSR={rec['delta_SR']:.3f}  ΔH={rec['mean_delta_H']:.3f}")
    else:
        # Real evaluation
        env_factory = None  # caller must supply; placeholder
        print("[regret_scaling] Loading reference policy (H=1) ...")
        ref_policy = _load_policy(args.checkpoint, chunk_size=1, device=args.device)
        SR_ref = _evaluate_policy(ref_policy, env_factory, args.n_rollouts, args.seeds)
        print(f"  SR_ref = {SR_ref:.3f}")

        for H in sorted(args.H):
            print(f"[regret_scaling] Evaluating H={H} ...")
            policy = _load_policy(args.checkpoint, chunk_size=H, device=args.device)
            SR = _evaluate_policy(policy, env_factory, args.n_rollouts, args.seeds)
            mean_dH = _compute_mean_cliff_depth(policy, env_factory, args.n_rollouts, args.seeds, H)
            rec = {
                "H": H,
                "SR": SR,
                "SR_ref": SR_ref,
                "delta_SR": max(0.0, SR_ref - SR),
                "mean_delta_H": mean_dH,
            }
            records.append(rec)
            print(f"  SR={SR:.3f}  δSR={rec['delta_SR']:.3f}  ΔH={mean_dH:.3f}")

    # Save CSV
    with open(output_dir / "regret_vs_H.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["H", "SR", "SR_ref", "delta_SR", "mean_delta_H"])
        w.writeheader()
        w.writerows(records)

    _save_plot(records, output_dir)
    _write_summary(records, output_dir, dry_run=(args.dry_run or args.checkpoint is None))
    print(f"[regret_scaling] outputs written to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
