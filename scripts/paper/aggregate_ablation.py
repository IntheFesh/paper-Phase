#!/usr/bin/env python

"""Aggregate ablation runs into CSV + stats JSON.

Two calling modes:

1. **Merge a single eval result** (invoked by run_ablation.sh after GPU eval)::

     python scripts/paper/aggregate_ablation.py \\
         --merge-eval outputs/ablation/full_seed42/eval \\
         --target outputs/ablation/full_seed42/eval_results.json

   Reads the JSON produced by lerobot-eval (``eval_info.json`` or similar),
   writes ``libero_long_sr`` / ``libero_spatial_sr`` back into ``target``,
   and flips ``placeholder`` to false.

2. **Full aggregation** (invoked at the tail of run_ablation.sh or standalone)::

     python scripts/paper/aggregate_ablation.py \\
         --output_root outputs/ablation \\
         --out_dir artifacts/ablation

   Scans ``output_root/*/eval_results.json`` and, grouped by (config, seed),
   emits:

   - ``artifacts/ablation/ablation_table_long.csv``
   - ``artifacts/ablation/ablation_table_spatial.csv``
   - ``artifacts/ablation/stats.json``

   CSV rows are seeds, columns are the 12 configs; the last three rows
   are mean / std / 95% CI half-width. ``stats.json`` collects, for each
   config, the placeholder flag set, mean ± std ± CI, and the delta
   versus baseline plus paired t-test p-value (aligned on full seeds).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from statistics import fmean, pstdev, stdev
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("aggregate_ablation")


CONFIG_ORDER: List[str] = [
    "baseline", "ident", "a", "b", "c", "ab", "ac", "bc",
    "pace", "pcar_only", "full", "pcar_noident",
]

CONFIG_KIND: Dict[str, str] = {
    "baseline": "baseline",
    "ident": "single",
    "a": "single", "b": "single", "c": "single",
    "ab": "pair", "ac": "pair", "bc": "pair",
    "pace": "triple",
    "pcar_only": "single",
    "full": "full",
    "pcar_noident": "robustness",
}


def _t_ci95_half_width(stdev_val: float, n: int) -> float:
    """Student-t +/-95% CI half-width.

    Returns 0 when ``n <= 1``; looks up a small table for 2 <= n <= 30;
    falls back to 1.96 for n > 30. Avoids a scipy dependency and is
    accurate enough for n in [2, 30].
    """

    if n <= 1:
        return 0.0
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
             8: 2.365, 9: 2.306, 10: 2.262, 15: 2.145, 20: 2.093, 30: 2.045}
    t = table.get(n, 1.96)
    return float(t * stdev_val / math.sqrt(n))


def _paired_t_p_value(diffs: List[float]) -> float:
    """Two-sided paired t-test p-value (no scipy; uses a Student-t CDF series approximation).

    Only used as a weak significance hint in ``paper_stats.md`` /
    ``stats.json``. When SR values are placeholders (every run has
    placeholder=true), this function still returns a number but the
    aggregator stamps ``placeholder_stats=true`` on ``stats.json``.
    """

    n = len(diffs)
    if n < 2:
        return float("nan")
    m = fmean(diffs)
    s = stdev(diffs)
    if s == 0.0:
        return 0.0 if m != 0.0 else 1.0
    t = m / (s / math.sqrt(n))
    df = n - 1
    x = df / (df + t * t)
    if df >= 30:
        z = abs(t)
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
        return float(max(0.0, min(1.0, p)))
    p = 2.0 * (1.0 - x ** (df / 2.0))
    return float(max(0.0, min(1.0, p)))


def _parse_merge(merge_dir: Path, target: Path) -> int:
    """Mode 1: merge a lerobot-eval JSON back into ``eval_results.json``."""

    if not target.is_file():
        log.error("target %s missing; train_local.py must run first", target)
        return 2

    payload = json.loads(target.read_text())

    candidates = [
        merge_dir / "eval_info.json",
        merge_dir / "eval_metrics.json",
        merge_dir / "results.json",
    ]
    src = None
    for c in candidates:
        if c.is_file():
            src = c
            break
    if src is None:
        log.warning("no eval JSON found under %s; placeholder kept", merge_dir)
        return 0

    raw = json.loads(src.read_text())
    long_sr = _extract_sr(raw, ["libero_10", "libero_long", "long"])
    spat_sr = _extract_sr(raw, ["libero_spatial", "spatial"])
    if long_sr is not None:
        payload["libero_long_sr"] = float(long_sr)
    if spat_sr is not None:
        payload["libero_spatial_sr"] = float(spat_sr)
    if long_sr is not None or spat_sr is not None:
        payload["placeholder"] = False
        payload["merged_from"] = str(src)
    target.write_text(json.dumps(payload, indent=2))
    log.info("merged %s -> %s (placeholder=%s)", src, target, payload["placeholder"])
    return 0


def _extract_sr(raw: Any, keys: List[str]) -> Optional[float]:
    """Find a success_rate field within a nested dict.

    Typical lerobot-eval structure::

        {"eval": {"pc_success": 0.64, "avg_max_reward": ...}, "libero_10": {...}}

    The match is intentionally loose: any key that contains one of
    ``keys``, whose value is either a float or a dict containing
    ``pc_success`` / ``success_rate``.
    """

    if isinstance(raw, dict):
        for k, v in raw.items():
            k_low = str(k).lower()
            if any(kk in k_low for kk in keys):
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, dict):
                    for sub in ("pc_success", "success_rate", "sr", "score"):
                        if sub in v and isinstance(v[sub], (int, float)):
                            return float(v[sub])
        for v in raw.values():
            r = _extract_sr(v, keys)
            if r is not None:
                return r
    elif isinstance(raw, list):
        for item in raw:
            r = _extract_sr(item, keys)
            if r is not None:
                return r
    return None


def _scan_runs(output_root: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Walk ``output_root/*/eval_results.json`` and return ``{(cfg, seed): payload}``."""

    runs: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir():
            continue
        ejson = run_dir / "eval_results.json"
        if not ejson.is_file():
            continue
        try:
            payload = json.loads(ejson.read_text())
        except json.JSONDecodeError as e:
            log.warning("bad JSON at %s: %s", ejson, e)
            continue
        cfg = str(payload.get("config_mode", ""))
        seed = int(payload.get("seed", -1))
        if cfg and seed >= 0:
            runs[(cfg, seed)] = payload
    return runs


def _build_table(
    runs: Dict[Tuple[str, int], Dict[str, Any]],
    sr_key: str,
) -> Tuple[List[int], List[List[Optional[float]]], List[List[float]]]:
    """Return (seeds, rows, stats_rows) where rows[s][c] is SR or None.

    stats_rows: [mean, std, ci_half] x len(CONFIG_ORDER).
    """

    seeds = sorted({seed for _, seed in runs.keys()})
    rows: List[List[Optional[float]]] = []
    for seed in seeds:
        row: List[Optional[float]] = []
        for cfg in CONFIG_ORDER:
            p = runs.get((cfg, seed))
            row.append(float(p[sr_key]) if p and sr_key in p else None)
        rows.append(row)

    means: List[float] = []
    stds: List[float] = []
    cis: List[float] = []
    for ci, cfg in enumerate(CONFIG_ORDER):
        vals = [r[ci] for r in rows if r[ci] is not None]
        if not vals:
            means.append(float("nan"))
            stds.append(float("nan"))
            cis.append(float("nan"))
            continue
        means.append(fmean(vals))
        s = stdev(vals) if len(vals) > 1 else 0.0
        stds.append(s)
        cis.append(_t_ci95_half_width(s, len(vals)))
    stats_rows = [means, stds, cis]
    return seeds, rows, stats_rows


def _write_csv(
    path: Path,
    seeds: List[int],
    rows: List[List[Optional[float]]],
    stats_rows: List[List[float]],
) -> None:
    """Write the per-seed-per-config CSV plus mean / std / CI footer rows."""
    lines: List[str] = []
    header = ["seed"] + CONFIG_ORDER
    lines.append(",".join(header))
    for seed, row in zip(seeds, rows):
        vals = [f"{v:.4f}" if v is not None else "" for v in row]
        lines.append(",".join([str(seed)] + vals))
    names = ["mean", "std", "ci95_half"]
    for name, srow in zip(names, stats_rows):
        vals = [f"{v:.4f}" if not math.isnan(v) else "" for v in srow]
        lines.append(",".join([name] + vals))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    log.info("wrote %s (%d data rows)", path, len(rows))


def _compute_stats_block(
    runs: Dict[Tuple[str, int], Dict[str, Any]],
    long_rows: List[List[Optional[float]]],
    spat_rows: List[List[Optional[float]]],
    seeds: List[int],
) -> Dict[str, Any]:
    """Assemble the full ``stats.json`` block: per-config mean/std/ci + paired delta/p vs baseline + placeholder flag."""

    any_placeholder = any(
        bool(p.get("placeholder", False)) for p in runs.values()
    )

    def stat_for(rows: List[List[Optional[float]]], cfg: str) -> Dict[str, Any]:
        """Per-config mean/std/CI block for ``rows``."""
        i = CONFIG_ORDER.index(cfg)
        vals = [r[i] for r in rows if r[i] is not None]
        if not vals:
            return {"n": 0, "mean": None, "std": None, "ci95_half": None}
        m = fmean(vals)
        s = stdev(vals) if len(vals) > 1 else 0.0
        return {
            "n": len(vals), "mean": float(m), "std": float(s),
            "ci95_half": float(_t_ci95_half_width(s, len(vals))),
            "values": [float(v) for v in vals],
        }

    def paired(rows: List[List[Optional[float]]], cfg: str) -> Dict[str, Any]:
        """Paired delta vs baseline (per-seed differences) for ``cfg``."""
        ib = CONFIG_ORDER.index("baseline")
        ic = CONFIG_ORDER.index(cfg)
        diffs: List[float] = []
        for row in rows:
            if row[ib] is None or row[ic] is None:
                continue
            diffs.append(float(row[ic] - row[ib]))
        if len(diffs) < 2:
            return {"n_pairs": len(diffs), "delta_mean": None, "p_value": None}
        return {
            "n_pairs": len(diffs),
            "delta_mean": float(fmean(diffs)),
            "delta_std": float(stdev(diffs)),
            "p_value": float(_paired_t_p_value(diffs)),
        }

    per_cfg: Dict[str, Any] = {}
    for cfg in CONFIG_ORDER:
        per_cfg[cfg] = {
            "kind": CONFIG_KIND[cfg],
            "libero_long": stat_for(long_rows, cfg),
            "libero_spatial": stat_for(spat_rows, cfg),
            "vs_baseline_long": paired(long_rows, cfg),
            "vs_baseline_spatial": paired(spat_rows, cfg),
        }

    return {
        "placeholder_stats": bool(any_placeholder),
        "note": (
            "placeholder_stats=true means at least one run's eval_results.json "
            "has placeholder=true. Numbers are NOT real LIBERO success rates; "
            "re-run run_eval_libero.sh on GPU and re-aggregate."
        ) if any_placeholder else "All runs have real eval metrics.",
        "configs": list(CONFIG_ORDER),
        "seeds": list(seeds),
        "num_configs": len(CONFIG_ORDER),
        "num_seeds": len(seeds),
        "per_config": per_cfg,
    }


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single-merge and full-aggregate modes."""
    p = argparse.ArgumentParser()
    p.add_argument("--merge-eval", type=str, default=None,
                   help="directory containing lerobot-eval JSON to merge into --target")
    p.add_argument("--target", type=str, default=None,
                   help="eval_results.json to update when --merge-eval is set")
    p.add_argument("--output_root", type=str, default="outputs/ablation",
                   help="root dir that contains {cfg}_seed{seed}/eval_results.json")
    p.add_argument("--out_dir", type=str, default="artifacts/ablation",
                   help="where to write ablation_table_*.csv + stats.json")
    return p.parse_args()


def main() -> int:
    """Dispatch to merge-mode or aggregate-mode based on CLI arguments."""
    args = _parse_args()
    if args.merge_eval is not None:
        if args.target is None:
            log.error("--merge-eval requires --target")
            return 2
        return _parse_merge(Path(args.merge_eval), Path(args.target))

    output_root = Path(args.output_root)
    if not output_root.is_dir():
        log.error("output_root %s does not exist", output_root)
        return 2

    runs = _scan_runs(output_root)
    if not runs:
        log.error("no eval_results.json found under %s", output_root)
        return 2
    log.info("found %d runs across %d configs",
             len(runs), len({c for c, _ in runs.keys()}))

    seeds, long_rows, long_stats = _build_table(runs, "libero_long_sr")
    _, spat_rows, _ = _build_table(runs, "libero_spatial_sr")

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "ablation_table_long.csv", seeds, long_rows, long_stats)
    _, _, spat_stats = _build_table(runs, "libero_spatial_sr")
    _write_csv(out_dir / "ablation_table_spatial.csv", seeds, spat_rows, spat_stats)

    stats = _compute_stats_block(runs, long_rows, spat_rows, seeds)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    log.info("wrote %s", out_dir / "stats.json")
    if stats["placeholder_stats"]:
        log.warning("placeholder_stats=true - numbers are CPU proxy, not LIBERO.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
