#!/usr/bin/env python
"""Generate a post-training diagnostic report from ``training_dynamics.csv``.

Inputs
------
``--input_dir`` (a single experiment directory, e.g.
``$PACE_RUN_DIR/ablation/02_cliff_via_beta_only/seed_42``) containing
``training_dynamics.csv`` produced by
:class:`lerobot_policy_phaseqflow.utils.DiagnosticLogger`.

Outputs
-------
- ``diagnostic_report.md`` — Markdown report with six sections
  (convergence, gradient health, phase encoder, PACE-A, PCAR,
  auto-flagged anomalies).
- ``figures/loss_curves.{png,pdf}``
- ``figures/grad_norm_curves.{png,pdf}``
- ``figures/beta_distribution.{png,pdf}``
- ``figures/fsq_codebook_usage.{png,pdf}`` — placeholder if the CSV
  does not include FSQ usage columns (the logger does not log them
  directly; they are reported in ``eval_results.json`` if present).

The script is deliberately tolerant of missing columns and partial
runs: any section whose required columns are entirely NaN is replaced
with a one-line ``"no data"`` notice rather than raising.

Usage
-----
::

    python scripts/utils/diagnostic_report.py \\
        --input_dir $PACE_RUN_DIR/ablation/02_cliff_via_beta_only/seed_42

    # Or, with an explicit eval_results.json for FSQ codebook usage:
    python scripts/utils/diagnostic_report.py \\
        --input_dir $PACE_RUN_DIR/stage2_phase_flow \\
        --eval_results $PACE_RUN_DIR/stage2_phase_flow/eval_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# CSV loading (no pandas dependency)
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> Tuple[List[str], List[Dict[str, float]]]:
    """Load a ``training_dynamics.csv`` file into a list of dicts.

    Each cell is parsed as ``float`` (NaN-safe). The header is returned
    separately for column order.
    """
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows: List[Dict[str, float]] = []
        for r in reader:
            parsed: Dict[str, float] = {}
            for k, v in r.items():
                try:
                    parsed[k] = float(v)
                except (TypeError, ValueError):
                    parsed[k] = float("nan")
            rows.append(parsed)
    return header, rows


def _column(rows: Sequence[Dict[str, float]], key: str) -> List[float]:
    """Extract one column as a list of floats (NaNs preserved)."""
    return [r.get(key, float("nan")) for r in rows]


def _finite(xs: Iterable[float]) -> List[float]:
    """Drop NaNs / infinities; useful for stats and plotting."""
    return [x for x in xs if isinstance(x, float) and math.isfinite(x)]


# ---------------------------------------------------------------------------
# Section: convergence analysis
# ---------------------------------------------------------------------------

@dataclass
class _ConvergenceStats:
    head_mean: float
    tail_mean: float
    monotone_decrease_frac: float
    longest_stall: int  # consecutive steps where |Δloss / loss| < 1%

    def as_markdown(self) -> str:
        return (
            "### Convergence analysis\n\n"
            f"- Mean of first 100 records: **{self.head_mean:.4f}**\n"
            f"- Mean of last 100 records: **{self.tail_mean:.4f}**\n"
            f"- Fraction of monotone-decreasing transitions: "
            f"**{self.monotone_decrease_frac:.1%}**\n"
            f"- Longest stall (|Δ| < 1% over consecutive records): "
            f"**{self.longest_stall} records**\n"
        )


def _convergence_stats(losses: Sequence[float]) -> Optional[_ConvergenceStats]:
    finite = _finite(losses)
    if len(finite) < 2:
        return None
    head = finite[: min(100, len(finite))]
    tail = finite[-min(100, len(finite)) :]
    head_mean = sum(head) / len(head)
    tail_mean = sum(tail) / len(tail)
    decreases = sum(1 for a, b in zip(finite[:-1], finite[1:]) if b < a)
    monotone_frac = decreases / max(1, len(finite) - 1)
    # longest stall
    longest = 0
    cur = 0
    for a, b in zip(finite[:-1], finite[1:]):
        denom = max(abs(a), 1e-9)
        if abs(b - a) / denom < 0.01:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return _ConvergenceStats(
        head_mean=head_mean,
        tail_mean=tail_mean,
        monotone_decrease_frac=monotone_frac,
        longest_stall=longest,
    )


# ---------------------------------------------------------------------------
# Section: gradient health
# ---------------------------------------------------------------------------

_GRAD_KEYS = ["grad_norm_total", "grad_norm_vision", "grad_norm_phase_encoder", "grad_norm_dit"]


def _gradient_health(rows: Sequence[Dict[str, float]]) -> Tuple[str, List[str]]:
    """Return Markdown table + list of warnings."""
    warnings: List[str] = []
    lines = ["### Gradient health", "", "| Module | Mean | Std | Explode (>100) | Vanish (<1e-4) |", "|---|---|---|---|---|"]
    for k in _GRAD_KEYS:
        vals = _finite(_column(rows, k))
        if not vals:
            lines.append(f"| {k} | n/a | n/a | n/a | n/a |")
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)
        n_explode = sum(1 for v in vals if v > 100.0)
        n_vanish = sum(1 for v in vals if v < 1e-4)
        lines.append(f"| {k} | {mean:.4f} | {std:.4f} | {n_explode}/{len(vals)} | {n_vanish}/{len(vals)} |")
        if n_explode > 0:
            warnings.append(f"[ERROR] {k}: {n_explode} records exceed 100 (gradient explosion).")
        if n_vanish / len(vals) > 0.5:
            warnings.append(f"[WARN] {k}: {n_vanish}/{len(vals)} records below 1e-4 (gradient vanishing).")
    return "\n".join(lines) + "\n", warnings


# ---------------------------------------------------------------------------
# Section: phase encoder quality
# ---------------------------------------------------------------------------

def _phase_section(
    rows: Sequence[Dict[str, float]],
    eval_results: Optional[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    lines = ["### Phase encoder quality", ""]

    macro_ent = _finite(_column(rows, "phase_posterior_entropy_macro"))
    micro_ent = _finite(_column(rows, "phase_posterior_entropy_micro"))
    if macro_ent:
        lines.append(f"- Mean macro posterior entropy: **{sum(macro_ent)/len(macro_ent):.4f}**")
    if micro_ent:
        lines.append(f"- Mean micro posterior entropy: **{sum(micro_ent)/len(micro_ent):.4f}**")

    # FSQ codebook usage is reported by eval_results.json (not by the
    # per-step CSV) because it requires a forward pass over a held-out
    # batch.
    fsq = (eval_results or {}).get("fsq_codebook_usage") if eval_results else None
    if isinstance(fsq, dict):
        macro_usage = fsq.get("macro_used_frac")
        micro_usage = fsq.get("micro_used_frac")
        lines.append("")
        lines.append("| FSQ level | Used codes / total | Usage frac |")
        lines.append("|---|---|---|")
        if macro_usage is not None:
            lines.append(f"| Macro | {fsq.get('macro_used', '?')} / {fsq.get('macro_total', '?')} | {macro_usage:.1%} |")
            if macro_usage < 0.30:
                warnings.append(
                    f"[WARN] FSQ macro codebook usage only {macro_usage:.1%}, "
                    f"below 30% threshold; phase representation may be degenerate."
                )
        if micro_usage is not None:
            lines.append(f"| Micro | {fsq.get('micro_used', '?')} / {fsq.get('micro_total', '?')} | {micro_usage:.1%} |")
            if micro_usage < 0.30:
                warnings.append(
                    f"[WARN] FSQ micro codebook usage only {micro_usage:.1%}, "
                    f"below 30% threshold."
                )
    else:
        lines.append("")
        lines.append("- FSQ codebook usage: not available (no `eval_results.json` with `fsq_codebook_usage` key).")

    return "\n".join(lines) + "\n", warnings


# ---------------------------------------------------------------------------
# Section: PACE-A behavior
# ---------------------------------------------------------------------------

def _pace_a_section(rows: Sequence[Dict[str, float]]) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    mean_beta = _finite(_column(rows, "pace_a_mean_beta"))
    max_beta = _finite(_column(rows, "pace_a_max_beta"))
    density = _finite(_column(rows, "pace_a_boundary_density"))

    lines = ["### PACE-A behavior", ""]
    if not mean_beta and not max_beta and not density:
        lines.append("- No PACE-A metrics recorded.")
        return "\n".join(lines) + "\n", warnings

    if mean_beta:
        lines.append(f"- Mean β_t over training: **{sum(mean_beta)/len(mean_beta):.4f}**")
    if max_beta:
        lines.append(f"- Max β_t observed: **{max(max_beta):.4f}**")
    if density:
        d_mean = sum(density) / len(density)
        lines.append(f"- Mean boundary density: **{d_mean:.4f}**")
        if d_mean < 0.10 or d_mean > 0.30:
            warnings.append(
                f"[WARN] PACE-A boundary density {d_mean:.3f} is outside the "
                f"expected [0.10, 0.30] range; β_t threshold may need tuning."
            )
    return "\n".join(lines) + "\n", warnings


# ---------------------------------------------------------------------------
# Section: PCAR statistics
# ---------------------------------------------------------------------------

def _pcar_section(rows: Sequence[Dict[str, float]]) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    trigger = _finite(_column(rows, "pcar_trigger_rate"))
    concord = _finite(_column(rows, "pcar_mean_concordance"))
    lines = ["### PCAR triggers", ""]
    if not trigger and not concord:
        lines.append("- No PCAR metrics recorded.")
        return "\n".join(lines) + "\n", warnings
    if trigger:
        lines.append(f"- Mean PCAR trigger rate: **{sum(trigger)/len(trigger):.4f}**")
        # Curriculum stage 0: triggers should be rare initially.
        early = trigger[: min(5, len(trigger))]
        if early and sum(early) / len(early) < 0.01:
            lines.append("- [INFO] Early trigger rate < 1%, consistent with curriculum stage 0.")
    if concord:
        lines.append(f"- Mean concordance C_t: **{sum(concord)/len(concord):.4f}**")
    return "\n".join(lines) + "\n", warnings


# ---------------------------------------------------------------------------
# Figures (matplotlib lazy import)
# ---------------------------------------------------------------------------

def _save_figure(fig: Any, out_dir: Path, stem: str) -> None:
    """Save a matplotlib Figure as both PNG (300 dpi) and PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def _smooth_ema(ys: Sequence[float], alpha: float = 0.9) -> List[float]:
    """Exponential moving average for noisy training curves (α closer to 1 = smoother)."""
    out: List[float] = []
    prev = float("nan")
    for y in ys:
        if math.isfinite(y):
            prev = y if not math.isfinite(prev) else alpha * prev + (1 - alpha) * y
        out.append(prev)
    return out


def _plot_training_dynamics(rows: Sequence[Dict[str, float]], out_dir: Path) -> Optional[str]:
    """4-panel training dynamics figure for paper appendix.

    Panels: (1) loss components  (2) gradient norm  (3) PACE-A β_t + boundary
    density  (4) PCAR trigger rate + concordance C_t.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    steps = _column(rows, "step")
    finite_steps = [s for s in steps if math.isfinite(s)]
    if not finite_steps:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PACE v2 — Training dynamics", fontsize=13, y=1.01)

    # -- Panel 1: loss components (smoothed) ----------------------------------
    ax = axes[0, 0]
    loss_keys = [
        ("loss_total", "total", "#333333", 2.0),
        ("loss_imitation", "imitation", "#1f77b4", 1.2),
        ("loss_flow_policy", "flow", "#ff7f0e", 1.2),
        ("loss_infonce_macro", "InfoNCE-macro", "#2ca02c", 1.0),
        ("loss_infonce_micro", "InfoNCE-micro", "#d62728", 1.0),
    ]
    plotted_loss = False
    for key, label, color, lw in loss_keys:
        raw = _column(rows, key)
        smoothed = _smooth_ema(raw, alpha=0.95)
        pairs = [(s, y) for s, y in zip(steps, smoothed) if math.isfinite(s) and math.isfinite(y)]
        if pairs:
            sx, sy = zip(*pairs)
            ax.plot(sx, sy, label=label, color=color, linewidth=lw)
            plotted_loss = True
    if plotted_loss:
        ax.set_xlabel("step")
        ax.set_ylabel("loss (EMA-smoothed)")
        ax.set_title("Loss components")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    # -- Panel 2: gradient norm (log scale) -----------------------------------
    ax = axes[0, 1]
    plotted_grad = False
    for key in _GRAD_KEYS:
        raw = _column(rows, key)
        pairs = [(s, y) for s, y in zip(steps, raw) if math.isfinite(s) and math.isfinite(y) and y > 0]
        if pairs:
            sx, sy = zip(*pairs)
            ax.plot(sx, sy, label=key.replace("grad_norm_", ""), linewidth=1.2)
            plotted_grad = True
    if plotted_grad:
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("grad norm (log)")
        ax.set_title("Per-module gradient norms")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    # -- Panel 3: PACE-A β_t + boundary density (dual axis) ------------------
    ax3 = axes[1, 0]
    ax3r = ax3.twinx()
    beta_pairs = [
        (s, y) for s, y in zip(steps, _column(rows, "pace_a_mean_beta"))
        if math.isfinite(s) and math.isfinite(y)
    ]
    density_pairs = [
        (s, y) for s, y in zip(steps, _column(rows, "pace_a_boundary_density"))
        if math.isfinite(s) and math.isfinite(y)
    ]
    if beta_pairs:
        bsx, bsy = zip(*beta_pairs)
        ax3.plot(bsx, _smooth_ema(list(bsy), 0.9), color="#1f77b4", linewidth=1.4,
                 label=r"$\bar\beta_t$ (left)")
        ax3.set_ylabel(r"mean $\beta_t$", color="#1f77b4")
        ax3.tick_params(axis="y", labelcolor="#1f77b4")
    if density_pairs:
        dsx, dsy = zip(*density_pairs)
        ax3r.plot(dsx, _smooth_ema(list(dsy), 0.9), color="#ff7f0e", linewidth=1.2,
                  linestyle="--", label="density (right)")
        ax3r.set_ylabel("boundary density", color="#ff7f0e")
        ax3r.tick_params(axis="y", labelcolor="#ff7f0e")
    ax3.set_xlabel("step")
    ax3.set_title(r"PACE-A: $\beta_t$ & boundary density")
    ax3.grid(alpha=0.3)
    lines3 = ax3.get_lines() + ax3r.get_lines()
    ax3.legend(lines3, [l.get_label() for l in lines3], fontsize=7, loc="upper right")

    # -- Panel 4: PCAR trigger rate + concordance (dual axis) -----------------
    ax4 = axes[1, 1]
    ax4r = ax4.twinx()
    trig_pairs = [
        (s, y) for s, y in zip(steps, _column(rows, "pcar_trigger_rate"))
        if math.isfinite(s) and math.isfinite(y)
    ]
    conc_pairs = [
        (s, y) for s, y in zip(steps, _column(rows, "pcar_mean_concordance"))
        if math.isfinite(s) and math.isfinite(y)
    ]
    if trig_pairs:
        tsx, tsy = zip(*trig_pairs)
        ax4.plot(tsx, _smooth_ema(list(tsy), 0.9), color="#2ca02c", linewidth=1.4,
                 label="trigger rate (left)")
        ax4.set_ylabel("PCAR trigger rate", color="#2ca02c")
        ax4.tick_params(axis="y", labelcolor="#2ca02c")
    if conc_pairs:
        csx, csy = zip(*conc_pairs)
        ax4r.plot(csx, _smooth_ema(list(csy), 0.9), color="#9467bd", linewidth=1.2,
                  linestyle="--", label=r"$C_t$ (right)")
        ax4r.set_ylabel(r"mean concordance $C_t$", color="#9467bd")
        ax4r.tick_params(axis="y", labelcolor="#9467bd")
    ax4.set_xlabel("step")
    ax4.set_title(r"PCAR: trigger rate & concordance $C_t$")
    ax4.grid(alpha=0.3)
    lines4 = ax4.get_lines() + ax4r.get_lines()
    ax4.legend(lines4, [l.get_label() for l in lines4], fontsize=7, loc="upper right")

    fig.tight_layout()
    _save_figure(fig, out_dir, "training_dynamics")
    plt.close(fig)
    return "training_dynamics"


def _plot_phase_entropy(rows: Sequence[Dict[str, float]], out_dir: Path) -> Optional[str]:
    """Phase posterior entropy (macro + micro) over training — shows phase learning progress."""
    macro = _column(rows, "phase_posterior_entropy_macro")
    micro = _column(rows, "phase_posterior_entropy_micro")
    steps = _column(rows, "step")
    if not _finite(macro) and not _finite(micro):
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    for vals, label, color in [
        (macro, "macro H(p_t)", "#1f77b4"),
        (micro, "micro H(p_t)", "#ff7f0e"),
    ]:
        pairs = [(s, y) for s, y in zip(steps, vals) if math.isfinite(s) and math.isfinite(y)]
        if pairs:
            sx, sy = zip(*pairs)
            ax.plot(sx, _smooth_ema(list(sy), 0.9), color=color, linewidth=1.5, label=label)
            ax.plot(sx, sy, color=color, linewidth=0.4, alpha=0.3)
    ax.set_xlabel("step")
    ax.set_ylabel("posterior entropy H(p̂_t)")
    ax.set_title("Phase posterior entropy during training")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, out_dir, "phase_entropy")
    plt.close(fig)
    return "phase_entropy"


def _plot_curves(rows: Sequence[Dict[str, float]], out_dir: Path) -> List[str]:
    """Plot loss + grad-norm curves and the β_t histogram. Returns names of
    figures actually written (so the report can link only to those)."""
    written: List[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return written

    steps = _column(rows, "step")

    # 1. Loss curves
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for key, label in [
        ("loss_total", "total"),
        ("loss_imitation", "imitation"),
        ("loss_flow_policy", "flow"),
        ("loss_infonce_macro", "InfoNCE-macro"),
        ("loss_infonce_micro", "InfoNCE-micro"),
    ]:
        ys = _column(rows, key)
        finite_pairs = [(s, y) for s, y in zip(steps, ys) if math.isfinite(y)]
        if not finite_pairs:
            continue
        sx, sy = zip(*finite_pairs)
        ax.plot(sx, sy, label=label, linewidth=1.6)
        plotted = True
    if plotted:
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Training losses")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save_figure(fig, out_dir, "loss_curves")
        written.append("loss_curves")
    plt.close(fig)

    # 2. Grad-norm curves
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for key in _GRAD_KEYS:
        ys = _column(rows, key)
        finite_pairs = [(s, y) for s, y in zip(steps, ys) if math.isfinite(y)]
        if not finite_pairs:
            continue
        sx, sy = zip(*finite_pairs)
        ax.plot(sx, sy, label=key.replace("grad_norm_", ""), linewidth=1.4)
        plotted = True
    if plotted:
        ax.set_xlabel("step")
        ax.set_ylabel("grad norm")
        ax.set_yscale("log")
        ax.set_title("Per-module gradient norms")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        _save_figure(fig, out_dir, "grad_norm_curves")
        written.append("grad_norm_curves")
    plt.close(fig)

    # 3. β_t histogram
    betas = _finite(_column(rows, "pace_a_mean_beta"))
    if betas:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(betas, bins=30, color="steelblue", alpha=0.85)
        ax.set_xlabel(r"PACE-A mean $\beta_t$")
        ax.set_ylabel("count (records)")
        ax.set_title(r"Distribution of mean $\beta_t$ over training")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save_figure(fig, out_dir, "beta_distribution")
        written.append("beta_distribution")
        plt.close(fig)

    return written


def _plot_fsq_usage(eval_results: Optional[Dict[str, Any]], out_dir: Path) -> Optional[str]:
    """Plot FSQ codebook usage as a horizontal bar chart, if data exists."""
    if not eval_results:
        return None
    fsq = eval_results.get("fsq_codebook_usage")
    if not isinstance(fsq, dict):
        return None
    macro = fsq.get("macro_used_frac")
    micro = fsq.get("micro_used_frac")
    if macro is None and micro is None:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    labels: List[str] = []
    values: List[float] = []
    if macro is not None:
        labels.append("macro K_1")
        values.append(float(macro))
    if micro is not None:
        labels.append("micro K_2")
        values.append(float(micro))
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.barh(labels, values, color="coral")
    ax.axvline(0.30, color="black", linestyle="--", linewidth=0.8, label="30% threshold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("codebook usage fraction")
    ax.set_title("FSQ codebook usage")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, out_dir, "fsq_codebook_usage")
    plt.close(fig)
    return "fsq_codebook_usage"


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(
    input_dir: Path,
    eval_results_path: Optional[Path] = None,
) -> Path:
    """Produce ``diagnostic_report.md`` and figures inside ``input_dir``.

    Returns the path to the Markdown report.
    """
    csv_path = input_dir / "training_dynamics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing training_dynamics.csv under {input_dir}")
    header, rows = _load_csv(csv_path)
    fig_dir = input_dir / "figures"

    eval_results: Optional[Dict[str, Any]] = None
    if eval_results_path and eval_results_path.is_file():
        try:
            eval_results = json.loads(eval_results_path.read_text())
        except json.JSONDecodeError:
            eval_results = None

    plotted = _plot_curves(rows, fig_dir)
    fsq_fig = _plot_fsq_usage(eval_results, fig_dir)
    dynamics_fig = _plot_training_dynamics(rows, fig_dir)
    entropy_fig = _plot_phase_entropy(rows, fig_dir)

    sections: List[str] = []
    sections.append(f"# Diagnostic report — `{input_dir.name}`\n")
    sections.append(f"- Source CSV: `{csv_path}`")
    sections.append(f"- Records: **{len(rows)}**")
    sections.append(f"- Figures dir: `{fig_dir}`\n")

    warnings: List[str] = []

    # Convergence
    conv = _convergence_stats(_column(rows, "loss_total"))
    if conv is not None:
        sections.append(conv.as_markdown())
        if conv.longest_stall >= 500 // 200:  # ≥ 500 steps at default cadence
            warnings.append(
                f"[WARN] Longest stall {conv.longest_stall} consecutive records "
                f"(≥ 500 steps at default 200-step cadence); convergence may have stalled."
            )
    else:
        sections.append("### Convergence analysis\n\n_No `loss_total` data._\n")

    # Gradient health
    grad_md, grad_warns = _gradient_health(rows)
    sections.append(grad_md)
    warnings.extend(grad_warns)

    # Phase encoder
    phase_md, phase_warns = _phase_section(rows, eval_results)
    sections.append(phase_md)
    warnings.extend(phase_warns)

    # PACE-A
    pace_md, pace_warns = _pace_a_section(rows)
    sections.append(pace_md)
    warnings.extend(pace_warns)

    # PCAR
    pcar_md, pcar_warns = _pcar_section(rows)
    sections.append(pcar_md)
    warnings.extend(pcar_warns)

    # Figures index
    sections.append("### Figures\n")
    if plotted:
        for name in plotted:
            sections.append(f"- `figures/{name}.png` (PDF: `figures/{name}.pdf`)")
    if fsq_fig:
        sections.append(f"- `figures/{fsq_fig}.png` (PDF: `figures/{fsq_fig}.pdf`)")
    if dynamics_fig:
        sections.append(f"- `figures/{dynamics_fig}.png` (PDF: `figures/{dynamics_fig}.pdf`)  ← 4-panel training overview")
    if entropy_fig:
        sections.append(f"- `figures/{entropy_fig}.png` (PDF: `figures/{entropy_fig}.pdf`)")
    if not plotted and not fsq_fig and not dynamics_fig and not entropy_fig:
        sections.append("_No figures produced (matplotlib missing or no data)._")
    sections.append("")

    # Auto-flagged anomalies
    sections.append("### Auto-flagged anomalies\n")
    if not warnings:
        sections.append("_No anomalies detected._\n")
    else:
        for w in warnings:
            sections.append(f"- {w}")
        sections.append("")

    report_path = input_dir / "diagnostic_report.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", type=Path, required=True, help="Experiment dir containing training_dynamics.csv")
    p.add_argument("--eval_results", type=Path, default=None, help="Optional path to eval_results.json")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    out = build_report(args.input_dir.resolve(), args.eval_results)
    print(f"[diagnostic_report] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
