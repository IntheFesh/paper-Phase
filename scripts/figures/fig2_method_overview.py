"""Figure 2 — PACE v2 method overview schematic.

Generates a publication-quality block diagram of the PACE v2 architecture:
  [Vision/Language Encoder] → [Hierarchical Planner] → [Shortcut Flow Head]
                                         ↓
                          [Cliff Detectors I1/I2/I3] → [Concordance] → [PCAR]

Rendered entirely in matplotlib (no external diagram tools required).

Usage
-----
::

    python scripts/figures/fig2_method_overview.py \\
        --output paper_figures/fig2_method_overview.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_RC = {
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def make_figure(output_path: Path) -> None:
    """Draw the PACE v2 block-diagram schematic using matplotlib FancyBboxPatch."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
        matplotlib.rcParams.update(_RC)
    except ImportError:
        print("[fig2_method_overview] matplotlib not available; skipping")
        return

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(cx, cy, w, h, label, sublabel="", color="#D0E8F2", fontsize=8):  # draw a rounded rectangle with label

        rect = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.08", linewidth=0.8,
            edgecolor="#555", facecolor=color, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(cx, cy + (0.08 if sublabel else 0), label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold", zorder=4)
        if sublabel:
            ax.text(cx, cy - 0.22, sublabel, ha="center", va="center",
                    fontsize=6.5, color="#444", zorder=4)

    def arrow(x0, y0, x1, y1):  # draw a solid arrowhead from (x0,y0) to (x1,y1)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#333",
                                    lw=0.9, mutation_scale=10), zorder=5)

    # Main pipeline row (y=2.8)
    box(1.1, 2.8, 1.8, 0.7, "Fusion Encoder",
        "vision + language\n+ state + history", color="#E8D5F5")
    box(3.3, 2.8, 1.8, 0.7, "Hierarchical\nPlanner",
        "macro K₁=20 / micro K₂=30", color="#D5EFD5")
    box(5.5, 2.8, 1.8, 0.7, "Shortcut Flow\nAction Head",
        "4-NFE flow matching", color="#FFF3CD")
    box(7.8, 2.8, 1.8, 0.7, "Action Chunk\nExecutor",
        "execute H actions", color="#F5D5D5")

    arrow(2.0, 2.8, 2.4, 2.8)
    arrow(4.2, 2.8, 4.6, 2.8)
    arrow(6.4, 2.8, 6.9, 2.8)

    # Cliff detector row (y=1.2)
    box(2.2, 1.2, 1.4, 0.6, r"$\hat{I}^{(1)}$",
        "Bhattacharyya β_t", color="#E8F4FD")
    box(4.0, 1.2, 1.4, 0.6, r"$\hat{I}^{(2)}$",
        "Policy variance σ²", color="#E8F4FD")
    box(5.8, 1.2, 1.4, 0.6, r"$\hat{I}^{(3)}$",
        "Velocity curvature", color="#E8F4FD")
    box(7.6, 1.2, 1.4, 0.6, "Concordance\n$C_t$",
        "rank-based fusion", color="#FDE8E8")
    box(9.2, 2.8, 0.8, 0.5, "PCAR", "", color="#F5D5D5")

    # Connections: planner → estimators
    for cx in [2.2, 4.0, 5.8]:
        arrow(3.3, 2.45, cx, 1.5)

    # Estimators → concordance
    for cx in [2.2, 4.0, 5.8]:
        arrow(cx, 0.9, 7.6, 1.35 if cx != 4.0 else 1.2)

    # Concordance → PCAR
    arrow(8.3, 1.2, 9.2, 2.55)

    # Boundary reweight arrow
    ax.annotate("", xy=(5.5, 2.45), xytext=(4.0, 0.9),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=0.7,
                                linestyle="dashed", mutation_scale=8), zorder=5)
    ax.text(4.4, 1.65, "boundary\nreweight", ha="center", fontsize=6, color="#888")

    # Legend
    legend_items = [
        mpatches.Patch(color="#E8D5F5", label="Encoder"),
        mpatches.Patch(color="#D5EFD5", label="Phase Planner"),
        mpatches.Patch(color="#FFF3CD", label="Flow Head"),
        mpatches.Patch(color="#E8F4FD", label="Cliff Estimators"),
        mpatches.Patch(color="#FDE8E8", label="Concordance / PCAR"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=6.5,
              framealpha=0.0, ncol=5, bbox_to_anchor=(0.0, -0.02))

    ax.set_title("PACE v2: Predictability-Aware Closed-loop Execution", pad=8, fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[fig2_method_overview] saved → {output_path}")


def main(argv=None) -> int:
    """CLI entry point for Figure 2."""
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("paper_figures/fig2_method_overview.pdf"))
    args = p.parse_args(argv)
    make_figure(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
