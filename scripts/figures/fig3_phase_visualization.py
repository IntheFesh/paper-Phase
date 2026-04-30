"""Figure 3 — Phase visualization across 3 successful episodes.

Multi-row subplot per episode:
  Row 1: z_macro (discrete phase index, step plot)
  Row 2: z_micro (discrete skill index, step plot)
  Row 3: Three cliff estimators I^(1), I^(2), I^(3) (normalized, overlay)
  Row 4: C_t concordance + PCAR replan events (vertical dashed lines)

Usage
-----
::

    python scripts/figures/fig3_phase_visualization.py \\
        --input paper_figures/phase_vis_data.json \\
        --output paper_figures/fig3_phase_visualization.pdf

    # Synthetic (no data needed):
    python scripts/figures/fig3_phase_visualization.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_RC = {
    "font.family": "serif",
    "font.size": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _synthetic_episode(T: int = 150, K_macro: int = 20, K_micro: int = 30,
                        rng: Optional[np.random.Generator] = None) -> Dict:
    if rng is None:
        rng = np.random.default_rng(42)

    # 3 macro phases, each subdivided into 2-3 micro skills
    macro_ids = np.zeros(T, dtype=int)
    micro_ids = np.zeros(T, dtype=int)
    replan_steps = []

    macro = 0
    micro = 0
    phase_len = T // 3
    for t in range(T):
        if t > 0 and t % phase_len == 0 and macro < 2:
            macro += 1
            micro = macro * 4 + rng.integers(0, 4)
            replan_steps.append(t)
        elif t > 0 and t % (phase_len // 2) == 0:
            micro = (micro + rng.integers(1, 5)) % K_micro
            replan_steps.append(t)
        macro_ids[t] = macro
        micro_ids[t] = micro

    # Cliff signals: spike at replan steps
    i1 = rng.normal(0, 0.1, T)
    i2 = rng.normal(0, 0.1, T)
    i3 = rng.normal(0, 0.1, T)
    for r in replan_steps:
        for dt in range(-2, 4):
            if 0 <= r + dt < T:
                w = np.exp(-abs(dt) / 1.5)
                i1[r + dt] += 0.7 * w
                i2[r + dt] += 0.6 * w
                i3[r + dt] += 0.65 * w

    concordance = (i1 + i2 + i3) / 3.0
    concordance = (concordance - concordance.min()) / (concordance.max() - concordance.min() + 1e-8)

    # Normalise estimators
    def _norm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-8)

    return {
        "T": T,
        "z_macro": macro_ids.tolist(),
        "z_micro": micro_ids.tolist(),
        "i_hat_1": _norm(i1).tolist(),
        "i_hat_2": _norm(i2).tolist(),
        "i_hat_3": _norm(i3).tolist(),
        "concordance": concordance.tolist(),
        "replan_steps": replan_steps,
    }


def make_figure(episodes: List[Dict], output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams.update(_RC)
    except ImportError:
        print("[fig3_phase_visualization] matplotlib not available; skipping")
        return

    n_ep = len(episodes)
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_ep, figsize=(3.5 * n_ep, 5.5),
                              gridspec_kw={"height_ratios": [0.8, 0.8, 1.2, 1.0]})
    if n_ep == 1:
        axes = [[axes[r]] for r in range(n_rows)]

    row_labels = [r"$z_{\rm macro}$", r"$z_{\rm micro}$",
                  r"$\hat{I}^{(1/2/3)}$", r"$C_t$ + replan"]
    col_titles = [f"Episode {i + 1}" for i in range(n_ep)]

    for col, ep in enumerate(episodes):
        T = ep["T"]
        ts = np.arange(T)

        # Row 0: macro phase
        axes[0][col].step(ts, ep["z_macro"], color="#4878CF", linewidth=1.2, where="post")
        axes[0][col].set_ylabel(row_labels[0] if col == 0 else "")
        axes[0][col].set_title(col_titles[col], fontsize=9)
        axes[0][col].set_xlim(0, T)

        # Row 1: micro skill
        axes[1][col].step(ts, ep["z_micro"], color="#6ACC65", linewidth=1.0, where="post")
        axes[1][col].set_ylabel(row_labels[1] if col == 0 else "")
        axes[1][col].set_xlim(0, T)

        # Row 2: cliff estimators overlay
        axes[2][col].plot(ts, ep["i_hat_1"], color="#4878CF", linewidth=0.9,
                          alpha=0.85, label=r"$\hat{I}^{(1)}$")
        axes[2][col].plot(ts, ep["i_hat_2"], color="#D65F5F", linewidth=0.9,
                          alpha=0.85, label=r"$\hat{I}^{(2)}$")
        axes[2][col].plot(ts, ep["i_hat_3"], color="#6ACC65", linewidth=0.9,
                          alpha=0.85, label=r"$\hat{I}^{(3)}$")
        if col == 0:
            axes[2][col].legend(fontsize=6, framealpha=0, loc="upper left")
        axes[2][col].set_ylabel(row_labels[2] if col == 0 else "")
        axes[2][col].set_xlim(0, T)
        axes[2][col].set_ylim(-0.05, 1.1)

        # Row 3: concordance + replan markers
        axes[3][col].plot(ts, ep["concordance"], color="#B47CC7", linewidth=1.2)
        axes[3][col].axhline(0.6, color="#888", linewidth=0.6, linestyle="--")
        for r_t in ep.get("replan_steps", []):
            axes[3][col].axvline(r_t, color="#D65F5F", linewidth=0.8,
                                 linestyle=":", alpha=0.8)
        axes[3][col].set_ylabel(row_labels[3] if col == 0 else "")
        axes[3][col].set_xlabel("Timestep")
        axes[3][col].set_xlim(0, T)
        axes[3][col].set_ylim(-0.05, 1.1)

    fig.suptitle("PACE v2: phase structure and cliff detection across episodes", y=1.01, fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[fig3_phase_visualization] saved → {output_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--n_episodes", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("paper_figures/fig3_phase_visualization.pdf"))
    args = p.parse_args(argv)

    if args.dry_run or args.input is None or not args.input.exists():
        rng = np.random.default_rng(0)
        episodes = [_synthetic_episode(rng=rng) for _ in range(args.n_episodes)]
    else:
        data = json.loads(args.input.read_text())
        episodes = data if isinstance(data, list) else [data]

    make_figure(episodes, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
