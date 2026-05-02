"""Training dynamics comparison across ablations (PACE v2 — paper figure).

Reads ``training_dynamics.csv`` from multiple experiment directories and
produces a side-by-side comparison figure:

  - Left panel  : smoothed total loss curves for every run (for Table 2)
  - Right panel : smoothed PACE-A mean β_t (shows cliff signal learning)

If three-stage ablation layout is desired, pass ``--n_stages 3`` together
with ``--stage_steps 80000 40000 10000`` to draw vertical stage-boundary
lines.

Usage
-----
::

    python scripts/figures/fig_training_comparison.py \\
        --input_dirs outputs/ablation_v2/01_bc_chunked/seed_42 \\
                     outputs/ablation_v2/02_cliff_beta/seed_42 \\
                     outputs/ablation_v2/07_full_pace/seed_42 \\
        --labels "BC-Chunked (Abl.01)" \\
                 "Cliff β_t (Abl.02)" \\
                 "Full PACE v2 (Abl.07)" \\
        --output paper_figures/fig_training_comparison.pdf

    # Dry run (synthetic curves, no real CSV needed):
    python scripts/figures/fig_training_comparison.py --dry_run
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# CSV helpers (no pandas)
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> Tuple[List[float], Dict[str, List[float]]]:
    """Return (steps, {col: values}) from a training_dynamics.csv."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return [], {}
    cols: Dict[str, List[float]] = {}
    for k in rows[0]:
        vals: List[float] = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (KeyError, ValueError, TypeError):
                vals.append(float("nan"))
        cols[k] = vals
    steps = cols.get("step", list(range(len(rows))))
    return steps, cols


def _smooth_ema(ys: Sequence[float], alpha: float = 0.95) -> List[float]:
    out: List[float] = []
    prev = float("nan")
    for y in ys:
        if math.isfinite(y):
            prev = y if not math.isfinite(prev) else alpha * prev + (1 - alpha) * y
        out.append(prev)
    return out


def _finite_pairs(
    xs: Sequence[float], ys: Sequence[float]
) -> Tuple[List[float], List[float]]:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if not pairs:
        return [], []
    px, py = zip(*pairs)
    return list(px), list(py)


# ---------------------------------------------------------------------------
# Synthetic dry-run data
# ---------------------------------------------------------------------------

def _synthetic_run(n: int = 400, seed: int = 0, abl_id: int = 0) -> Dict[str, List[float]]:
    """Generate plausible-looking synthetic training curves for dry-run mode."""
    import random
    rng = random.Random(seed + abl_id * 7)
    offsets = [0.0, -0.05, -0.12]          # Abl01 / Abl02 / Abl07 final-loss offset
    offset = offsets[min(abl_id, len(offsets) - 1)]
    steps, loss, beta = [], [], []
    for i in range(n):
        t = i / n
        base_loss = 0.8 * math.exp(-3.0 * t) + 0.12 + offset + rng.gauss(0, 0.01)
        steps.append(i * 200)
        loss.append(max(0.0, base_loss))
        if abl_id >= 1:
            b = 0.05 + 0.15 * (1 - math.exp(-4 * t)) + rng.gauss(0, 0.005)
            beta.append(max(0.0, min(1.0, b)))
        else:
            beta.append(float("nan"))
    return {"step": steps, "loss_total": loss, "pace_a_mean_beta": beta}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _plot(
    runs: List[Tuple[str, List[float], Dict[str, List[float]]]],
    output_path: Path,
    stage_steps: Optional[List[int]] = None,
    smoothing: float = 0.95,
) -> None:
    """
    Parameters
    ----------
    runs
        List of (label, steps, col_dict) tuples.
    output_path
        Output file path (.pdf or .png). Both formats are written.
    stage_steps
        Cumulative step counts at stage boundaries, e.g. [80000, 120000].
    smoothing
        EMA alpha (higher = smoother).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[fig_training_comparison] matplotlib not installed; cannot plot.")
        sys.exit(1)

    has_beta = any(
        any(math.isfinite(v) for v in cols.get("pace_a_mean_beta", []))
        for _, _, cols in runs
    )
    ncols = 2 if has_beta else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    # Panel 1: total loss
    ax = axes[0]
    for i, (label, steps, cols) in enumerate(runs):
        raw = cols.get("loss_total", [])
        smooth = _smooth_ema(raw, alpha=smoothing)
        sx, sy = _finite_pairs(steps, smooth)
        if sx:
            ax.plot(sx, sy, label=label, color=_COLORS[i % len(_COLORS)], linewidth=1.8)
            # faint raw curve
            rx, ry = _finite_pairs(steps, raw)
            ax.plot(rx, ry, color=_COLORS[i % len(_COLORS)], linewidth=0.4, alpha=0.2)
    if stage_steps:
        for boundary, sname in zip(
            stage_steps, [f"S{j+2}" for j in range(len(stage_steps))]
        ):
            ax.axvline(boundary, color="grey", linestyle=":", linewidth=0.9, alpha=0.7)
            ax.text(boundary, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1,
                    sname, ha="center", fontsize=7, color="grey")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (EMA-smoothed)")
    ax.set_title("Training loss comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: PACE-A β_t
    if has_beta:
        ax2 = axes[1]
        for i, (label, steps, cols) in enumerate(runs):
            raw = cols.get("pace_a_mean_beta", [])
            if not any(math.isfinite(v) for v in raw):
                continue
            smooth = _smooth_ema(raw, alpha=smoothing)
            sx, sy = _finite_pairs(steps, smooth)
            if sx:
                ax2.plot(sx, sy, label=label, color=_COLORS[i % len(_COLORS)], linewidth=1.8)
        if stage_steps:
            for boundary in stage_steps:
                ax2.axvline(boundary, color="grey", linestyle=":", linewidth=0.9, alpha=0.7)
        ax2.set_xlabel("training step")
        ax2.set_ylabel(r"mean $\beta_t$ (PACE-A)")
        ax2.set_title(r"Cliff signal $\beta_t$ during training")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    stem = output_path.with_suffix("")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_training_comparison] saved → {stem}.pdf / .png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input_dirs",
        type=Path,
        nargs="+",
        metavar="DIR",
        help="Experiment dirs each containing training_dynamics.csv.",
    )
    p.add_argument(
        "--labels",
        type=str,
        nargs="*",
        metavar="LABEL",
        help="Display labels for each run (same order as --input_dirs).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("paper_figures/fig_training_comparison.pdf"),
        help="Output path (.pdf; a matching .png is also written).",
    )
    p.add_argument(
        "--stage_steps",
        type=int,
        nargs="*",
        metavar="STEP",
        help="Cumulative step counts at stage boundaries (e.g. 80000 120000).",
    )
    p.add_argument(
        "--smoothing",
        type=float,
        default=0.95,
        help="EMA smoothing alpha (0–1; higher = smoother, default 0.95).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Generate synthetic curves without loading real CSVs.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.dry_run:
        default_labels = ["BC-Chunked (Abl.01)", "Cliff β_t (Abl.02)", "Full PACE v2 (Abl.07)"]
        runs = []
        for i, label in enumerate(default_labels):
            synth = _synthetic_run(n=400, seed=42, abl_id=i)
            runs.append((label, synth["step"], synth))
        print("[fig_training_comparison] dry_run: using synthetic data")
    else:
        if not args.input_dirs:
            print("ERROR: --input_dirs required (or use --dry_run)", file=sys.stderr)
            return 1
        labels = args.labels or [d.name for d in args.input_dirs]
        if len(labels) < len(args.input_dirs):
            labels += [d.name for d in args.input_dirs[len(labels):]]
        runs = []
        for d, label in zip(args.input_dirs, labels):
            csv_path = d / "training_dynamics.csv"
            if not csv_path.is_file():
                print(f"[WARN] {csv_path} not found; skipping", file=sys.stderr)
                continue
            steps, cols = _load_csv(csv_path)
            runs.append((label, steps, cols))
        if not runs:
            print("ERROR: no valid training_dynamics.csv files found.", file=sys.stderr)
            return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot(
        runs,
        args.output,
        stage_steps=args.stage_steps,
        smoothing=args.smoothing,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
