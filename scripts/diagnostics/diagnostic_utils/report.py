"""Report writers: JSON / Markdown / PNG.

Produces three artefacts under ``<output_dir>``:
    - ``report.json`` — machine-readable payload.
    - ``report.md`` — human-readable summary (with executive brief).
    - ``fig_h1.png`` — histogram (boundary vs interior flow-matching loss).
    - ``fig_h2.png`` — scatter (misalignment x success) with regression line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt # noqa: E402
import numpy as np # noqa: E402


def verdict_h1(ratio: float, p_value: float) -> str:
    """H1 PASS if ratio>=2.0 and p<0.01; WARN 1.5<=r<2.0; else FAIL."""
    if ratio >= 2.0 and p_value < 0.01:
        return "PASS"
    if 1.5 <= ratio < 2.0:
        return "WARN"
    return "FAIL"


def verdict_h2(r: float, p_value: float) -> str:
    """H2 PASS if r<=-0.5 and p<0.01; WARN -0.5<r<=-0.3; else FAIL."""
    if r <= -0.5 and p_value < 0.01:
        return "PASS"
    if -0.5 < r <= -0.3:
        return "WARN"
    return "FAIL"


def save_h1_figure(
    boundary_losses: np.ndarray,
    interior_losses: np.ndarray,
    out_path: Path,
) -> None:
    """Histogram of boundary vs interior flow-matching loss with mean lines."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bl = boundary_losses[np.isfinite(boundary_losses)]
    il = interior_losses[np.isfinite(interior_losses)]
    if bl.size == 0 and il.size == 0:
        ax.text(0.5, 0.5, "No losses collected", ha="center", va="center")
    else:
        bins = 40
        all_vals = np.concatenate([bl, il]) if bl.size and il.size else (bl if bl.size else il)
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6
        if bl.size:
            ax.hist(bl, bins=bins, range=(vmin, vmax), alpha=0.55,
                    color="#d62728", label=f"boundary (n={bl.size})")
            ax.axvline(float(bl.mean()), color="#7f0f14", linestyle="--",
                       label=f"boundary mean={bl.mean():.4f}")
        if il.size:
            ax.hist(il, bins=bins, range=(vmin, vmax), alpha=0.55,
                    color="#1f77b4", label=f"interior (n={il.size})")
            ax.axvline(float(il.mean()), color="#0d3e6e", linestyle="--",
                       label=f"interior mean={il.mean():.4f}")
    ax.set_xlabel("per-timestep flow-matching loss")
    ax.set_ylabel("count")
    ax.set_title("H1: boundary vs interior flow-matching loss")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_h2_figure(
    misalignments: np.ndarray,
    successes: np.ndarray,
    out_path: Path,
    pearson_r: float,
) -> None:
    """Scatter plot of misalignment vs success with a linear fit."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    if misalignments.size == 0:
        ax.text(0.5, 0.5, "No episodes collected", ha="center", va="center")
    else:
        jitter = (np.random.default_rng(0).uniform(-0.04, 0.04, size=successes.size)
                  if successes.size else 0.0)
        colors = np.where(successes > 0.5, "#2ca02c", "#d62728")
        ax.scatter(misalignments, successes + jitter, c=colors, s=36, alpha=0.75,
                   edgecolors="black", linewidths=0.3)
        if misalignments.std() > 1e-9 and misalignments.size >= 2:
            coef = np.polyfit(misalignments, successes, 1)
            xs = np.linspace(float(misalignments.min()), float(misalignments.max()), 50)
            ys = np.polyval(coef, xs)
            ax.plot(xs, ys, "k--", linewidth=1.2, label=f"linear fit (r={pearson_r:.3f})")
            ax.legend(loc="best")
    ax.set_xlabel("episode mean misalignment (timesteps from replan to nearest boundary)")
    ax.set_ylabel("success (0 / 1)")
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.25, 1.25)
    ax.set_title("H2: misalignment x rollout success")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def build_report_payload(
    h1: Dict[str, Any],
    h2: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the JSON payload."""
    return {"meta": meta, "h1": h1, "h2": h2}


def write_json_report(payload: Dict[str, Any], out_path: Path) -> None:
    """Write the JSON payload to ``out_path``, creating parents as needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=float)


def write_markdown_report(
    payload: Dict[str, Any],
    out_path: Path,
    synthetic_demo: bool,
    synthetic_env: bool,
) -> None:
    """Render the human-readable ``report.md`` (with executive summary)."""
    h1 = payload["h1"]
    h2 = payload["h2"]
    meta = payload["meta"]

    h1_verdict = h1.get("verdict", "N/A")
    h2_verdict = h2.get("verdict", "N/A")

    if h1_verdict == "PASS" and h2_verdict == "PASS":
        brief_action = "**Proceed to Round 2**: both core hypotheses hold; the project direction is sound."
    elif h1_verdict == "FAIL" or h2_verdict == "FAIL":
        brief_action = ("**Stop / redesign**: at least one core hypothesis does not hold on real data; "
                        "the theoretical premise of Phase-Centric VLA needs to be re-argued.")
    else:
        brief_action = ("**Adjust strategy**: the results land in the WARN band. Consider enlarging the demo / episode pool, "
                        "or re-checking with the ``velocity_change`` / ``planner_output`` proxy before deciding.")

    lines: List[str] = []
    lines.append("# Round 1 Diagnostic Report (Phase-Centric VLA)\n")
    lines.append("## Executive Summary\n")
    lines.append(
        f"This round's diagnostic verdicts for hypothesis H1 (Phase-Boundary Loss Gap) and H2 "
        f"(Misalignment-Failure Correlation) are **H1={h1_verdict}, H2={h2_verdict}**."
    )
    if synthetic_demo:
        lines.append(
            "**SYNTHETIC DEMOS**: the real HuggingFace dataset was unavailable; "
            "H1 was measured on synthetic demos and is only meant to validate the pipeline."
        )
    if synthetic_env:
        lines.append(
            "**SYNTHETIC ENV**: the LIBERO environment was unavailable; "
            "H2 was measured on a synthetic 3-waypoint 2D navigation env; "
            "needs to be retested on real LIBERO."
        )
    lines.append(f"Recommended action: {brief_action}\n")

    lines.append("## Metadata\n")
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## H1: Phase-Boundary Loss Gap\n")
    lines.append(
        f"- boundary_loss_mean = `{h1.get('boundary_loss_mean', float('nan')):.6f}` "
        f"(n={h1.get('n_boundary', 0)})"
    )
    lines.append(
        f"- interior_loss_mean = `{h1.get('interior_loss_mean', float('nan')):.6f}` "
        f"(n={h1.get('n_interior', 0)})"
    )
    lines.append(f"- ratio = boundary / interior = `{h1.get('ratio', float('nan')):.4f}`")
    lines.append(
        f"- Welch t-test: t = `{h1.get('t_stat', float('nan')):.3f}`, "
        f"p = `{h1.get('p_value', float('nan')):.3g}`"
    )
    lines.append(f"- **Verdict: {h1_verdict}**")
    lines.append("- Criterion: PASS when ratio >= 2.0 and p < 0.01; WARN when 1.5 <= ratio < 2.0; otherwise FAIL.")
    lines.append("")
    lines.append("![H1 histogram](fig_h1.png)\n")

    lines.append("## H2: Misalignment x Failure Correlation\n")
    lines.append(
        f"- mean_misalignment_success = `{h2.get('mean_misalignment_success', float('nan')):.4f}`"
    )
    lines.append(
        f"- mean_misalignment_failure = `{h2.get('mean_misalignment_failure', float('nan')):.4f}`"
    )
    lines.append(
        f"- Pearson r = `{h2.get('pearson_r', float('nan')):.4f}`, "
        f"p = `{h2.get('p_value', float('nan')):.3g}`"
    )
    lines.append(
        f"- success_rate = `{h2.get('success_rate', float('nan')):.3f}` "
        f"over {h2.get('num_episodes', 0)} episodes"
    )
    lines.append(f"- **Verdict: {h2_verdict}**")
    lines.append("- Criterion: PASS when r <= -0.5 and p < 0.01; WARN when -0.5 < r <= -0.3; otherwise FAIL.")
    lines.append("")
    lines.append("![H2 scatter](fig_h2.png)\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
