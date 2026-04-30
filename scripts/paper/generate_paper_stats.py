#!/usr/bin/env python

"""Render ``stats.json`` into ``artifacts/paper_stats.md`` (paper prose).

Produces a structured numeric summary ready to drop into the paper
abstract / results section:

- Main headline: baseline vs full SR (+/-95% CI) + Delta pp + paired p-value.
- PACE-only / PCAR-only variants.
- Ablation insights: marginal Delta of removing a component.
- Spatial control: PACE-C is not significant on spatial (or is negative).
- Placeholder banner: if ``stats_json.placeholder_stats=true``, emit a
  banner at the top.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("paper_stats")


def _fmt_pct(mean: Optional[float], ci: Optional[float]) -> str:
    """Format ``mean +/- ci`` as a percentage string."""
    if mean is None:
        return "N/A"
    m = 100.0 * float(mean)
    c = 100.0 * float(ci or 0.0)
    return f"{m:.1f} +/- {c:.1f}%"


def _fmt_delta(pair: Dict[str, Any]) -> str:
    """Format a paired Delta cell in percentage points plus its p-value."""
    d = pair.get("delta_mean")
    p = pair.get("p_value")
    if d is None:
        return "N/A"
    d_pp = 100.0 * float(d)
    sign = "+" if d_pp >= 0 else ""
    p_str = ("p<0.001" if (p is not None and p < 1e-3)
             else (f"p={p:.3f}" if p is not None else "p=N/A"))
    return f"{sign}{d_pp:.1f} pp ({p_str})"


def _placeholder_banner(stats: Dict[str, Any]) -> str:
    """Return a warning banner when the stats are placeholders."""
    if not stats.get("placeholder_stats"):
        return ""
    return (
        "> **PLACEHOLDER - CPU dry-run**\n"
        "> \n"
        "> `stats.json` has `placeholder_stats=true`. All numbers in this file come from\n"
        "> the CPU placeholder SR proxy in `scripts/training/train_dummy_batch.py` (a linear\n"
        "> function of the dummy-batch training loss); they are **not** the real LIBERO\n"
        "> benchmark success rate.\n"
        "> \n"
        "> Real paper numbers require running `scripts/training/run_ablation.sh` + GPU eval\n"
        "> on RTX 5070, then re-running `scripts/paper/aggregate_ablation.py` and this script.\n\n"
    )


def _headline_section(stats: Dict[str, Any]) -> str:
    """Emit the main LIBERO-Long results section (baseline, full, PACE, PCAR)."""
    per = stats["per_config"]
    baseline_long = per["baseline"]["libero_long"]
    full_long = per["full"]["libero_long"]
    pace_long = per["pace"]["libero_long"]
    pcar_only_long = per["pcar_only"]["libero_long"]
    full_vs_baseline = per["full"]["vs_baseline_long"]
    lines = [
        "## Main results (LIBERO-Long)",
        "",
        f"- Baseline (PhaseQFlow++): {_fmt_pct(baseline_long['mean'], baseline_long['ci95_half'])}"
        f" ({baseline_long['n']} seeds)",
        f"- Full system (Ident + A + B + C + PCAR): {_fmt_pct(full_long['mean'], full_long['ci95_half'])}",
        f"- Absolute gain (full vs baseline): **{_fmt_delta(full_vs_baseline)}**",
        "",
        f"- PACE only (Ident + A + B + C, no PCAR): {_fmt_pct(pace_long['mean'], pace_long['ci95_half'])} "
        f" ({_fmt_delta(per['pace']['vs_baseline_long'])})",
        f"- PCAR only (Ident + PCAR, no PACE): {_fmt_pct(pcar_only_long['mean'], pcar_only_long['ci95_half'])} "
        f" ({_fmt_delta(per['pcar_only']['vs_baseline_long'])})",
    ]
    return "\n".join(lines) + "\n"


def _ablation_insight(stats: Dict[str, Any]) -> str:
    """Emit the ablation-insight section (component drops and pairwise synergy)."""
    per = stats["per_config"]

    def mean_or_none(cfg: str) -> Optional[float]:
        """Return the Long-SR mean for ``cfg`` or None when missing."""
        m = per[cfg]["libero_long"]["mean"]
        return float(m) if m is not None else None

    def drop_delta(without_cfg: str) -> str:
        """Delta in pp when a single component is dropped from ``full``."""
        full_m = mean_or_none("full")
        wo_m = mean_or_none(without_cfg)
        if full_m is None or wo_m is None:
            return "N/A"
        dd = 100.0 * (full_m - wo_m)
        sign = "+" if dd >= 0 else ""
        return f"{sign}{dd:.1f} pp"

    def interaction(ab_cfg: str, a_cfg: str, b_cfg: str) -> str:
        """Synergy of combining two components vs the sum of their individual gains."""
        base = mean_or_none("ident")
        if base is None:
            return "N/A"
        a_m = mean_or_none(a_cfg)
        b_m = mean_or_none(b_cfg)
        ab_m = mean_or_none(ab_cfg)
        if a_m is None or b_m is None or ab_m is None:
            return "N/A"
        sum_indiv = (a_m - base) + (b_m - base)
        joint = ab_m - base
        syn = 100.0 * (joint - sum_indiv)
        sign = "+" if syn >= 0 else ""
        return f"synergy = {sign}{syn:.1f} pp"

    lines = [
        "## Ablation insights",
        "",
        "### Single-component removal (remove each innovation from full)",
        "",
        f"- Remove PCAR (full -> pace): {drop_delta('pace')}",
        f"- Remove PACE-A (full -> bc + PCAR proxy): {drop_delta('bc')} (bc+PCAR not run; bc approximation)",
        f"- Remove PACE-B (full -> ac + PCAR proxy): {drop_delta('ac')} (ac+PCAR not run; ac approximation)",
        f"- Remove PACE-C (full -> ab + PCAR proxy): {drop_delta('ab')} (ab+PCAR not run; ab approximation)",
        f"- Remove Identifiability (full -> pcar_noident): {drop_delta('pcar_noident')}",
        "",
        "### Component interaction (vs simple sum)",
        "",
        f"- A + B vs A / B (marginal over ident): {interaction('ab', 'a', 'b')}",
        f"- A + C vs A / C: {interaction('ac', 'a', 'c')}",
        f"- B + C vs B / C: {interaction('bc', 'b', 'c')}",
        "",
        "> A positive synergy means the combined effect exceeds the sum of individual effects; a negative one means the components conflict or saturate.",
    ]
    return "\n".join(lines) + "\n"


def _spatial_control(stats: Dict[str, Any]) -> str:
    """Emit the LIBERO-Spatial control section."""
    per = stats["per_config"]
    c_vs_baseline = per["c"]["vs_baseline_spatial"]
    pace_vs_baseline_spatial = per["pace"]["vs_baseline_spatial"]
    pace_vs_baseline_long = per["pace"]["vs_baseline_long"]
    lines = [
        "## Spatial control (LIBERO-Spatial)",
        "",
        (
            "LIBERO-Spatial tasks usually have <= 2 phases each, so phase-centric "
            "innovations (especially the PACE-C curriculum) **should not** meaningfully "
            "lift SR - this is a deliberate falsification design for the `phase-centric` "
            "claim (if spatial also lifts a lot, what we are measuring is a `every "
            "augmentation wins` common cause, not phase structure itself)."
        ),
        "",
        f"- PACE-C only (spatial): Delta = **{_fmt_delta(c_vs_baseline)}**",
        (f"- PACE all-on (spatial): Delta = {_fmt_delta(pace_vs_baseline_spatial)} "
         f"(long: {_fmt_delta(pace_vs_baseline_long)})"),
        "",
        "> If p > 0.05 and Delta_spatial << Delta_long, the hypothesis that phase-centric gains come from phase structure is supported.",
    ]
    return "\n".join(lines) + "\n"


def _robustness(stats: Dict[str, Any]) -> str:
    """Emit the robustness section (PCAR without identifiability)."""
    per = stats["per_config"]
    pcar_only = per["pcar_only"]["libero_long"]
    pcar_noident = per["pcar_noident"]["libero_long"]
    lines = [
        "## Robustness: PCAR without Ident",
        "",
        f"- PCAR + Identifiability (pcar_only): {_fmt_pct(pcar_only['mean'], pcar_only['ci95_half'])}",
        f"- PCAR without Identifiability (pcar_noident): {_fmt_pct(pcar_noident['mean'], pcar_noident['ci95_half'])}",
        "",
        "If pcar_noident << pcar_only, PCAR's replacement ability depends on the "
        "identifiable phase latent as its signal source; otherwise the beta_t signal "
        "from PCAR is self-robust.",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--stats_json", type=str, default="artifacts/ablation/stats.json")
    p.add_argument("--out", type=str, default="artifacts/paper_stats.md")
    return p.parse_args()


def main() -> int:
    """Load ``stats.json`` and write ``paper_stats.md``."""
    args = _parse_args()
    stats_path = Path(args.stats_json)
    if not stats_path.is_file():
        log.error("stats.json missing at %s - run aggregate_ablation.py first", stats_path)
        return 2
    stats = json.loads(stats_path.read_text())

    parts = [
        "# Round 8 - Paper-ready statistics summary\n",
        f"- configs: {len(stats['configs'])}",
        f"- seeds: {stats['seeds']}",
        "",
        _placeholder_banner(stats),
        _headline_section(stats),
        _ablation_insight(stats),
        _spatial_control(stats),
        _robustness(stats),
        "## Raw CSV / figure links\n",
        "- `artifacts/ablation/ablation_table_long.csv` - seeds x configs SR matrix",
        "- `artifacts/ablation/ablation_table_spatial.csv` - same for LIBERO-Spatial",
        "- `artifacts/ablation/stats.json` - per-config mean/std/CI + Delta/p",
        "- `paper_figures/fig1_main_bar.png` / `fig2_long_vs_spatial.png` / `fig3_beta_vs_sr.png`",
        "- `paper_figures/ablation_table.tex` - booktabs LaTeX",
        "",
    ]
    md = "\n".join(parts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
