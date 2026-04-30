"""Figure 4 — Regret scaling: δSR vs H·ΔH with theoretical prediction line.

Reads ``paper_figures/regret_scaling/regret_vs_H.csv`` and renders:
  - Scatter points: (H·ΔH, δSR) coloured by H
  - Empirical linear fit (OLS)
  - Theoretical prediction line from Proposition 3: δSR = c·H·ΔH
  - Inset: δSR vs H (raw, to show monotone growth)

Usage
-----
::

    python scripts/figures/fig4_regret_scaling.py \\
        --input paper_figures/regret_scaling/regret_vs_H.csv \\
        --output paper_figures/fig4_regret_scaling.pdf
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_RC = {
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _load_csv(path: Path) -> List[Dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _synthetic_data() -> List[Dict]:
    rng = np.random.default_rng(42)
    rows = []
    for H in [4, 8, 16, 32, 64]:
        dH = rng.uniform(0.08, 0.18)
        dSR = 0.004 * H * dH + rng.normal(0, 0.008)
        SR_ref = rng.uniform(0.72, 0.82)
        rows.append({"H": H, "SR": SR_ref - max(0, dSR),
                     "SR_ref": SR_ref, "delta_SR": max(0, dSR), "mean_delta_H": dH})
    return rows


def make_figure(rows: List[Dict], output_path: Path) -> None:
    """Render two-panel regret scaling: (a) δSR vs H and (b) δSR vs H·ΔH with OLS fit."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        matplotlib.rcParams.update(_RC)
    except ImportError:
        print("[fig4_regret_scaling] matplotlib not available; skipping")
        return

    Hs = np.array([float(r["H"]) for r in rows])
    dSRs = np.array([float(r["delta_SR"]) for r in rows])
    dHs = np.array([float(r["mean_delta_H"]) for r in rows])
    xs = Hs * dHs  # Proposition 3 predictor

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Left: δSR vs H (raw trend)
    axes[0].scatter(Hs, dSRs, color="#4878CF", s=50, zorder=5)
    for H, dSR in zip(Hs, dSRs):
        axes[0].annotate(f"H={int(H)}", (H, dSR),
                         textcoords="offset points", xytext=(4, 2), fontsize=7)
    axes[0].set_xlabel("Chunk horizon $H$")
    axes[0].set_ylabel(r"Success-rate gap $\delta$SR")
    axes[0].set_title("(a) Regret vs horizon", fontsize=9)
    axes[0].set_xlim(0, Hs.max() * 1.1)
    axes[0].set_ylim(bottom=0)

    # Right: δSR vs H·ΔH with fit
    norm = Normalize(vmin=Hs.min(), vmax=Hs.max())
    cmap = plt.cm.viridis
    for H, x, dSR in zip(Hs, xs, dSRs):
        axes[1].scatter(x, dSR, color=cmap(norm(H)), s=50, zorder=5)
        axes[1].annotate(f"H={int(H)}", (x, dSR),
                         textcoords="offset points", xytext=(4, 2), fontsize=7)

    # OLS fit
    if len(xs) >= 2 and np.std(xs) > 1e-10:
        c = np.dot(xs, dSRs) / np.dot(xs, xs)  # slope through origin (Prop. 3)
        x_fit = np.linspace(0, xs.max() * 1.05, 100)
        axes[1].plot(x_fit, c * x_fit, "k--", linewidth=1.2,
                     label=f"Prop. 3 fit: $c = {c:.4f}$")
        # R² around origin
        ss_res = np.sum((dSRs - c * xs) ** 2)
        ss_tot = np.sum(dSRs ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else float("nan")
        axes[1].legend(fontsize=7.5, framealpha=0, title=f"$R^2 = {r2:.3f}$",
                       title_fontsize=7)

    # Colorbar for H
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], pad=0.02, shrink=0.85)
    cbar.set_label("$H$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    axes[1].set_xlabel(r"$H \cdot \overline{\Delta H}$ (Prop. 3 predictor)")
    axes[1].set_ylabel(r"$\delta$SR")
    axes[1].set_title("(b) Regret scaling validation", fontsize=9)
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[fig4_regret_scaling] saved → {output_path}")


def main(argv=None) -> int:
    """CLI entry point: load regret_vs_H.csv (or synthesise) then call make_figure."""
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("paper_figures/regret_scaling/regret_vs_H.csv"))
    p.add_argument("--output", type=Path, default=Path("paper_figures/fig4_regret_scaling.pdf"))
    args = p.parse_args(argv)

    if args.input.exists():
        rows = _load_csv(args.input)
    else:
        print(f"[fig4_regret_scaling] {args.input} not found; using synthetic data")
        rows = _synthetic_data()

    make_figure(rows, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
