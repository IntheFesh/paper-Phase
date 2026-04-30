"""Aggregate experiment results into a single paper_figures/main_results.csv.

Reads outputs from:
  - paper_figures/universality/raw_distances.json
  - paper_figures/regret_scaling/regret_vs_H.csv
  - paper_figures/triangulation/triangulation_table.csv
  - paper_figures/libero_perturbed/*_summary.json
  - paper_figures/simpler/simpler_aggregate.json

Produces:
  - paper_figures/main_results.csv  — one row per metric
  - paper_figures/main_results.json — structured dict

Usage
-----
::

    python scripts/aggregate_results.py --output paper_figures/main_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_csv(path: Path) -> Optional[List[Dict]]:
    if not path.exists():
        return None
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return None


def _aggregate(figures_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Universality: mean and std of failure distances across policies     #
    # ------------------------------------------------------------------ #
    raw_dist = _load_json(figures_dir / "universality" / "raw_distances.json")
    if raw_dist:
        for policy, dists_raw in raw_dist.items():
            dists = [d for d in dists_raw if d >= 0]  # -1 → success
            n_fail = len(dists)
            n_total = len(dists_raw)
            rows.append({
                "experiment": "universality",
                "metric": f"failure_distance_mean_{policy}",
                "value": round(float(np.mean(dists)), 3) if dists else float("nan"),
                "n": n_total,
            })
            rows.append({
                "experiment": "universality",
                "metric": f"failure_rate_{policy}",
                "value": round(n_fail / n_total, 3) if n_total > 0 else float("nan"),
                "n": n_total,
            })

    # ------------------------------------------------------------------ #
    # Regret scaling                                                       #
    # ------------------------------------------------------------------ #
    regret_rows = _load_csv(figures_dir / "regret_scaling" / "regret_vs_H.csv")
    if regret_rows:
        for r in regret_rows:
            rows.append({
                "experiment": "regret_scaling",
                "metric": f"SR_H{r.get('H', '?')}",
                "value": float(r.get("SR", "nan")),
                "n": None,
            })
            rows.append({
                "experiment": "regret_scaling",
                "metric": f"delta_SR_H{r.get('H', '?')}",
                "value": float(r.get("delta_SR", "nan")),
                "n": None,
            })

    # ------------------------------------------------------------------ #
    # Triangulation concordance                                            #
    # ------------------------------------------------------------------ #
    tri_rows = _load_csv(figures_dir / "triangulation" / "triangulation_table.csv")
    if tri_rows:
        for r in tri_rows:
            det = r.get("detector", "unknown")
            rows.append({
                "experiment": "triangulation",
                "metric": f"F1_{det}",
                "value": float(r.get("F1", "nan")),
                "n": int(r.get("n_TP", 0)) + int(r.get("n_FN", 0)),
            })

    # ------------------------------------------------------------------ #
    # LIBERO-Perturbed                                                     #
    # ------------------------------------------------------------------ #
    for summary_file in sorted((figures_dir / "libero_perturbed").glob("*_summary.json")):
        data = _load_json(summary_file)
        if data:
            key = f"libero_perturbed_SR_{data.get('perturbation_cm', '?')}cm"
            rows.append({
                "experiment": "libero_perturbed",
                "metric": key,
                "value": float(data.get("success_rate", "nan")),
                "n": int(data.get("n_rollouts", 0)),
            })

    # ------------------------------------------------------------------ #
    # SimplerEnv                                                           #
    # ------------------------------------------------------------------ #
    simpler = _load_json(figures_dir / "simpler" / "simpler_aggregate.json")
    if simpler:
        rows.append({
            "experiment": "simpler",
            "metric": "mean_SR",
            "value": float(simpler.get("mean_success_rate", "nan")),
            "n": None,
        })
        for task_row in simpler.get("tasks", []):
            rows.append({
                "experiment": "simpler",
                "metric": f"SR_{task_row.get('task', 'unknown')}",
                "value": float(task_row.get("success_rate", "nan")),
                "n": int(task_row.get("n_rollouts", 0)),
            })

    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figures_dir", type=Path, default=Path("paper_figures"))
    p.add_argument("--output", type=Path, default=Path("paper_figures/main_results.csv"))
    args = p.parse_args(argv)

    rows = _aggregate(args.figures_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "metric", "value", "n"])
        w.writeheader()
        w.writerows(rows)

    # Also write JSON
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps({
        "generated": datetime.now().isoformat(),
        "rows": rows,
    }, indent=2))

    print(f"[aggregate_results] {len(rows)} metrics written to {args.output}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
