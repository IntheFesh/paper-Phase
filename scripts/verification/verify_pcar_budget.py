#!/usr/bin/env python

"""Verification: does the PCAR adaptive threshold really match the replan budget epsilon.

Motivation
----------
The core promise of Round 7's ``PCARTrigger`` is that, **whatever the
distribution of beta_t**, a rolling-quantile threshold
``tau_cp = Quantile_{1-epsilon}(beta_history)`` drives the actual trigger rate
to the preset budget ``epsilon``. This script empirically verifies the
promise on synthetic beta trajectories:

1. Synthesise 50 episodes x T~200 beta_t sequences. Beta comes from two
   sources:
   - **Interior noise**: Beta(2, 8), a left-leaning distribution so most
     beta < 0.3.
   - **Boundary pulses**: 2-4 random Gaussian bumps per episode (amplitude
     0.8-1.0), simulating real phase switches.
2. For every ``epsilon in {0.05, 0.1, 0.2, 0.3}`` instantiate a fresh
   ``PCARTrigger``, feed the shared synthetic beta stream, and record
   ``get_actual_replan_rate()``.
3. Plot "set epsilon vs actual rate" with a y=x reference line.
4. Theoretical expectation: ``|actual - epsilon| < 0.05`` (see Round 7 summary
   section 3 on budget-quantile convergence).

This is a pure algorithmic verification and does not depend on a trained
planner; it checks the mathematical tool "budget-adaptive quantile" for
robustness under beta-distribution perturbations. From Round 8 onward, a real
planner will feed alpha=0.9-calibrated beta into the same tool; the synthesis
parameters here remain tunable.

Run
---

    python scripts/verification/verify_pcar_budget.py --num_episodes 50 --epsilons 0.05,0.1,0.2,0.3

Pass condition: ``|actual - epsilon| < 0.05`` for every epsilon.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "lerobot_policy_phaseqflow" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from lerobot_policy_phaseqflow.phase_centric.pcar_trigger import PCARTrigger # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("verify_pcar_budget")


class _StubCfg:
    """Minimal config exposing only the two fields ``PCARTrigger`` reads.

    Using a bare class avoids pulling in the full 21-field
    ``PhaseQFlowConfig`` constructor; the verification here does not need it.
    """

    def __init__(self, eps: float, manual_tau: float = 0.4) -> None:
        """Store the budget epsilon and the (unused) manual threshold."""
        self.pcar_trigger_budget_eps = float(eps)
        self.pcar_change_threshold = float(manual_tau)


def _synthesize_beta(
    num_episodes: int,
    ep_len: int,
    boundaries_per_ep: Tuple[int, int],
    pulse_width: int,
    seed: int,
) -> np.ndarray:
    """Generate a ``(num_episodes, ep_len)`` beta trajectory.

    - Interior time steps: ``beta ~ Beta(2, 8)`` (E[beta]=0.2, mostly below
      0.4).
    - Boundary pulses: 2-4 random positions get a Gaussian bump
      (sigma=``pulse_width``) with peak in ``[0.7, 1.0]``, clipped to
      ``[0, 1]``.
    """
    rng = np.random.default_rng(int(seed))
    beta_all = rng.beta(2.0, 8.0, size=(num_episodes, ep_len)).astype(np.float32)
    for ep in range(num_episodes):
        n_b = int(rng.integers(boundaries_per_ep[0], boundaries_per_ep[1] + 1))
        centers = rng.integers(pulse_width, ep_len - pulse_width, size=n_b)
        for c in centers:
            amp = float(rng.uniform(0.7, 1.0))
            t = np.arange(ep_len)
            bump = amp * np.exp(-0.5 * ((t - c) / float(pulse_width)) ** 2)
            beta_all[ep] = np.clip(beta_all[ep] + bump, 0.0, 1.0)
    return beta_all


def _run_one_eps(beta_all: np.ndarray, eps: float) -> Dict[str, float]:
    """Run the full synthetic beta stream for one epsilon; return rate and final tau."""
    cfg = _StubCfg(eps=eps)
    trig = PCARTrigger(cfg, history_size=1000, warmup_min=50)
    flat = beta_all.reshape(-1)
    for b in flat:
        trig.update_and_check(float(b))
    return {
        "eps": float(eps),
        "actual_rate": float(trig.get_actual_replan_rate()),
        "final_threshold": float(trig.current_threshold()),
        "total_steps": int(flat.size),
    }


def _plot(results: List[Dict[str, float]], out_path: Path) -> None:
    """Plot "set epsilon vs actual rate" with a +/- 0.05 envelope around y=x."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e: # noqa: BLE001
        log.warning("matplotlib unavailable; skipping plot: %s", e)
        return

    eps_vals = np.array([r["eps"] for r in results])
    rates = np.array([r["actual_rate"] for r in results])

    fig, ax = plt.subplots(1, 1, figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1.0, label="y = x (ideal)")
    ax.plot(eps_vals, rates, marker="o", color="tab:blue", linewidth=1.6, label="actual rate")
    grid = np.linspace(0.0, 1.0, 100)
    ax.fill_between(grid, np.clip(grid - 0.05, 0, 1), np.clip(grid + 0.05, 0, 1),
                    color="tab:blue", alpha=0.08, label="+/- 0.05 band")
    ax.set_xlabel("set budget epsilon")
    ax.set_ylabel("actual replan rate")
    ax.set_title("PCAR adaptive threshold: budget vs actual replan rate")
    ax.set_xlim(0, max(float(eps_vals.max()) * 1.2, 0.4))
    ax.set_ylim(0, max(float(rates.max()) * 1.2, 0.4))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("saved figure: %s", out_path)


def _render_report(
    results: List[Dict[str, float]],
    tolerance: float,
    beta_stats: Dict[str, float],
    verdict: str,
) -> str:
    """Render the markdown report body."""
    lines = [
        "# Round 7 - PCAR Budget epsilon Verification",
        "",
        f"**Verdict**: `{verdict}` (tolerance = {tolerance}).",
        "",
        "## beta synthesis",
        "",
        f"- mean beta: **{beta_stats['mean']:.3f}**",
        f"- std beta: {beta_stats['std']:.3f}",
        f"- fraction beta > 0.5: {beta_stats['above_0.5']:.3%}",
        f"- total beta samples: {beta_stats['total']}",
        "",
        "## Budget sweep",
        "",
        "| set epsilon | actual rate | |diff| | final tau |",
        "|------:|------------:|------:|--------:|",
    ]
    for r in results:
        diff = abs(r["actual_rate"] - r["eps"])
        lines.append(
            f"| {r['eps']:.2f} | {r['actual_rate']:.3f} | {diff:.3f} | {r['final_threshold']:.3f} |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "- `figures/budget_vs_rate.png` - line plot with +/- 0.05 envelope around y=x.",
        "- `report.json` - full numeric payload.",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--num_episodes", type=int, default=50)
    p.add_argument("--ep_len", type=int, default=200)
    p.add_argument("--boundaries_per_ep_lo", type=int, default=2)
    p.add_argument("--boundaries_per_ep_hi", type=int, default=4)
    p.add_argument("--pulse_width", type=int, default=3,
                   help="Gaussian pulse sigma (timesteps); simulates the width of boundary transients.")
    p.add_argument("--epsilons", type=str, default="0.05,0.1,0.2,0.3",
                   help="Comma-separated list of budgets epsilon to sweep.")
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="|actual_rate - epsilon| must be < tolerance to PASS.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="artifacts/pcar_budget")
    return p.parse_args()


def main() -> int:
    """Run the budget-vs-rate sweep and emit report.json / report.md / figures."""
    args = _parse_args()
    eps_list = [float(s) for s in str(args.epsilons).split(",") if s.strip()]
    if not eps_list:
        log.error("empty --epsilons list")
        return 2

    beta_all = _synthesize_beta(
        num_episodes=int(args.num_episodes),
        ep_len=int(args.ep_len),
        boundaries_per_ep=(int(args.boundaries_per_ep_lo), int(args.boundaries_per_ep_hi)),
        pulse_width=int(args.pulse_width),
        seed=int(args.seed),
    )
    beta_stats = {
        "mean": float(beta_all.mean()),
        "std": float(beta_all.std()),
        "above_0.5": float((beta_all > 0.5).mean()),
        "total": int(beta_all.size),
    }
    log.info(
        "synthesized beta: episodes=%d, T=%d, mean=%.3f, frac(>0.5)=%.3f",
        int(args.num_episodes), int(args.ep_len),
        beta_stats["mean"], beta_stats["above_0.5"],
    )

    results: List[Dict[str, float]] = []
    all_pass = True
    for eps in eps_list:
        rec = _run_one_eps(beta_all, eps)
        diff = abs(rec["actual_rate"] - rec["eps"])
        ok = diff < float(args.tolerance)
        all_pass = all_pass and ok
        log.info(
            "eps=%.2f -> actual=%.3f (|diff|=%.3f, tau=%.3f) %s",
            rec["eps"], rec["actual_rate"], diff, rec["final_threshold"],
            "PASS" if ok else "FAIL",
        )
        results.append(rec)

    verdict = "PASS" if all_pass else "FAIL"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    _plot(results, figures_dir / "budget_vs_rate.png")

    report = {
        "num_episodes": int(args.num_episodes),
        "ep_len": int(args.ep_len),
        "boundaries_per_ep": [int(args.boundaries_per_ep_lo), int(args.boundaries_per_ep_hi)],
        "pulse_width": int(args.pulse_width),
        "tolerance": float(args.tolerance),
        "seed": int(args.seed),
        "beta_stats": beta_stats,
        "sweep": results,
        "verdict": verdict,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "report.md").write_text(_render_report(results, float(args.tolerance), beta_stats, verdict))
    log.info("report: %s", out_dir / "report.md")
    log.info("verdict: %s", verdict)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
