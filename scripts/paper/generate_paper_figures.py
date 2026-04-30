#!/usr/bin/env python

"""Generate paper Figure 1, 2, 3 from aggregate CSV + stats.json.

Figures
-------
- **Fig 1**: Main LIBERO-Long SR bar chart across 12 configs with 95% CI;
  bar color encodes kind (baseline = gray, single = blue, pair = green,
  triple/full = red, robustness = orange).
- **Fig 2**: Long vs Spatial side by side; highlights that PACE-C offers
  no gain on Spatial while it does on Long.
- **Fig 3**: Per-rollout scatter of mean beta_t at replan vs success;
  if ``--scatter_json`` is missing, approximate it from per-run
  ``eval_results.json`` ``beta_mean_when_replan`` plus placeholder SR
  (the figure is marked as a placeholder).

Outputs:
- ``paper_figures/fig1_main_bar.png``
- ``paper_figures/fig2_long_vs_spatial.png``
- ``paper_figures/fig3_beta_vs_sr.png``

Placeholder handling
--------------------
When ``stats.json`` has ``placeholder_stats=true``, every figure title
gets the suffix "(placeholder - CPU dry-run)".
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("paper_figures")


COLOR_MAP: Dict[str, str] = {
    "baseline": "#7f7f7f",
    "single": "#1f77b4",
    "pair": "#2ca02c",
    "triple": "#d62728",
    "full": "#d62728",
    "robustness": "#ff7f0e",
}


def _load_stats(path: Path) -> Dict[str, Any]:
    """Load ``stats.json`` or exit with status 2 if missing."""
    if not path.is_file():
        log.error("stats.json missing at %s", path)
        sys.exit(2)
    return json.loads(path.read_text())


def _import_mpl():
    """Import matplotlib with the Agg backend and return pyplot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _placeholder_suffix(stats: Dict[str, Any]) -> str:
    """Return a suffix string flagging placeholder status, or empty string."""
    return " (placeholder - CPU dry-run)" if stats.get("placeholder_stats") else ""


def _fig1_main_bar(stats: Dict[str, Any], out_path: Path) -> None:
    """Render Figure 1: LIBERO-Long SR bars across the 12 configs."""
    plt = _import_mpl()

    cfgs: List[str] = stats["configs"]
    per = stats["per_config"]
    means: List[float] = []
    cis: List[float] = []
    colors: List[str] = []
    for cfg in cfgs:
        s = per[cfg]["libero_long"]
        means.append(float(s["mean"] or 0.0))
        cis.append(float(s["ci95_half"] or 0.0))
        colors.append(COLOR_MAP.get(per[cfg]["kind"], "#1f77b4"))

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.2))
    xs = list(range(len(cfgs)))
    ax.bar(xs, means, yerr=cis, capsize=3, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(cfgs, rotation=25, ha="right")
    ax.set_ylabel("LIBERO-Long SR")
    ax.set_ylim(0.0, max([m + c for m, c in zip(means, cis)] + [0.7]) * 1.1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("Figure 1 - Ablation main results (LIBERO-Long)" + _placeholder_suffix(stats))

    legend_kinds = ["baseline", "single", "pair", "triple", "robustness"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP[k]) for k in legend_kinds]
    ax.legend(handles, legend_kinds, loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    log.info("wrote %s", out_path)


def _fig2_long_vs_spatial(stats: Dict[str, Any], out_path: Path) -> None:
    """Render Figure 2: side-by-side LIBERO-Long vs LIBERO-Spatial bars."""
    plt = _import_mpl()

    cfgs: List[str] = stats["configs"]
    per = stats["per_config"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, suite, title in [
        (axes[0], "libero_long", "LIBERO-Long"),
        (axes[1], "libero_spatial", "LIBERO-Spatial"),
    ]:
        means = [float(per[c][suite]["mean"] or 0.0) for c in cfgs]
        cis = [float(per[c][suite]["ci95_half"] or 0.0) for c in cfgs]
        colors = [COLOR_MAP.get(per[c]["kind"], "#1f77b4") for c in cfgs]
        xs = list(range(len(cfgs)))
        ax.bar(xs, means, yerr=cis, capsize=3, color=colors,
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(cfgs, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Success Rate")

    fig.suptitle(
        "Figure 2 - LIBERO-Long vs LIBERO-Spatial"
        + _placeholder_suffix(stats),
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def _fig3_beta_vs_sr(
    stats: Dict[str, Any],
    output_root: Path,
    out_path: Path,
) -> None:
    """One point per (config, seed): x = beta_mean_when_replan, y = long SR.

    The real pipeline logs ``(beta_mean, success)`` per episode during
    rollout. The CPU dry-run instead falls back to the scalar fields
    present in each run's ``eval_results.json``.
    """

    plt = _import_mpl()

    xs_all: List[float] = []
    ys_all: List[float] = []
    colors_all: List[str] = []
    for cfg in stats["configs"]:
        kind = stats["per_config"][cfg]["kind"]
        color = COLOR_MAP.get(kind, "#1f77b4")
        for seed in stats["seeds"]:
            run = output_root / f"{cfg}_seed{seed}" / "eval_results.json"
            if not run.is_file():
                continue
            payload = json.loads(run.read_text())
            beta = payload.get("beta_mean_when_replan")
            sr = payload.get("libero_long_sr")
            if beta is None or sr is None:
                continue
            xs_all.append(float(beta))
            ys_all.append(float(sr))
            colors_all.append(color)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.2))
    if xs_all:
        ax.scatter(xs_all, ys_all, c=colors_all, s=44, edgecolor="black",
                   linewidth=0.4, alpha=0.85)
        n = len(xs_all)
        xmean = sum(xs_all) / n
        ymean = sum(ys_all) / n
        num = sum((x - xmean) * (y - ymean) for x, y in zip(xs_all, ys_all))
        den = sum((x - xmean) ** 2 for x in xs_all)
        if den > 0:
            slope = num / den
            intercept = ymean - slope * xmean
            x_line = [min(xs_all), max(xs_all)]
            y_line = [slope * x + intercept for x in x_line]
            ax.plot(x_line, y_line, "k--", linewidth=1.2,
                    label=f"fit: y = {slope:.3f} * beta + {intercept:.3f}")
            ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("mean beta_t when PCAR fires")
    ax.set_ylabel("LIBERO-Long SR")
    ax.set_title("Figure 3 - beta at replan vs rollout success"
                 + _placeholder_suffix(stats))
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    log.info("wrote %s", out_path)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the figure generator."""
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", type=str,
                   default="artifacts/ablation/ablation_table_long.csv")
    p.add_argument("--spatial_csv", type=str,
                   default="artifacts/ablation/ablation_table_spatial.csv")
    p.add_argument("--stats_json", type=str,
                   default="artifacts/ablation/stats.json")
    p.add_argument("--output_root", type=str, default="outputs/ablation",
                   help="used by Fig 3 to read beta_mean_when_replan per run")
    p.add_argument("--out_dir", type=str, default="paper_figures")
    return p.parse_args()


def main() -> int:
    """Render all three figures into ``--out_dir``."""
    args = _parse_args()
    stats = _load_stats(Path(args.stats_json))
    out_dir = Path(args.out_dir)
    _fig1_main_bar(stats, out_dir / "fig1_main_bar.png")
    _fig2_long_vs_spatial(stats, out_dir / "fig2_long_vs_spatial.png")
    _fig3_beta_vs_sr(stats, Path(args.output_root), out_dir / "fig3_beta_vs_sr.png")
    if stats.get("placeholder_stats"):
        log.warning("placeholder_stats=true - figures reflect CPU dry-run, not LIBERO.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
