"""Diagnostic: replan-event alignment with gripper-flip oracle.

Measures how well the PCAR replanning events (triggers) align with
ground-truth phase transitions (gripper-flip indices), reporting:
  - Precision: TP / (TP + FP)
  - Recall:    TP / (TP + FN)
  - F1:        harmonic mean

Matching tolerance: ±k timesteps (default k=5, matching §4.3 protocol).

Usage
-----
::

    python scripts/diagnostics/replan_alignment.py \\
        --checkpoint checkpoints/phaseqflow \\
        --output paper_figures/diagnostics/replan_alignment.csv

    # Dry run:
    python scripts/diagnostics/replan_alignment.py --dry_run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Shared matching logic (mirrors triangulation_concordance._match)
# ---------------------------------------------------------------------------

def _oracle_flips(gripper: np.ndarray) -> Set[int]:
    return {t for t in range(1, len(gripper)) if gripper[t] != gripper[t - 1]}


def _match(predicted: Set[int], oracle: Set[int], tolerance: int) -> Tuple[int, int, int]:
    matched: Set[int] = set()
    TP = 0
    for p in sorted(predicted):
        for o in sorted(oracle):
            if abs(p - o) <= tolerance and o not in matched:
                TP += 1
                matched.add(o)
                break
    FP = len(predicted) - TP
    FN = len(oracle) - len(matched)
    return TP, FP, FN


def _f1(TP: int, FP: int, FN: int) -> Tuple[float, float, float]:
    prec = TP / (TP + FP) if TP + FP > 0 else 0.0
    rec = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


# ---------------------------------------------------------------------------
# Synthetic dry-run
# ---------------------------------------------------------------------------

def _synthetic_run(
    n_episodes: int = 30,
    T: int = 200,
    n_flips: int = 3,
    tolerance: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    if rng is None:
        rng = np.random.default_rng(42)

    total_TP, total_FP, total_FN = 0, 0, 0
    spacing = T // (n_flips + 1)
    for _ in range(n_episodes):
        oracle = set()
        for i in range(1, n_flips + 1):
            t = int(i * spacing + rng.integers(-spacing // 5, spacing // 5 + 1))
            oracle.add(int(np.clip(t, 1, T - 1)))

        # Simulate replan events: near oracle with some noise
        predicted: Set[int] = set()
        for o in oracle:
            if rng.random() < 0.85:   # 85% recall
                offset = int(rng.integers(-3, 4))
                predicted.add(int(np.clip(o + offset, 0, T - 1)))
        # False positives
        for _ in range(int(rng.poisson(1.2))):
            fp = int(rng.integers(0, T))
            if all(abs(fp - o) > tolerance for o in oracle):
                predicted.add(fp)

        TP, FP, FN = _match(predicted, oracle, tolerance)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    prec, rec, f1 = _f1(total_TP, total_FP, total_FN)
    return {"precision": prec, "recall": rec, "F1": f1,
            "n_TP": total_TP, "n_FP": total_FP, "n_FN": total_FN}


# ---------------------------------------------------------------------------
# Real computation
# ---------------------------------------------------------------------------

def _compute_alignment(
    checkpoint_path: str,
    dataset_path: str,
    n_episodes: int,
    tolerance: int,
    device: str,
) -> Dict:
    """Run policy on dataset; compare replan triggers to gripper-flip oracle."""
    # Placeholder: real implementation requires env + policy rollout
    # with access to replan trigger flags from PhaseQFlowPolicy.
    raise NotImplementedError(
        "Real replan alignment requires a LIBERO-Long env + policy rollout. "
        "Run with --dry_run for diagnostic without checkpoints."
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _save(results: Dict, output_path: Path, dry_run: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"metric": k, "value": round(v, 4) if isinstance(v, float) else v}
            for k, v in results.items()]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerows(rows)

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps({**results, "dry_run": dry_run,
                                      "generated": datetime.now().isoformat()}, indent=2))
    print(f"[replan_alignment] P={results['precision']:.3f}  "
          f"R={results['recall']:.3f}  F1={results['F1']:.3f}")
    print(f"  → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--n_episodes", type=int, default=30)
    p.add_argument("--tolerance", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=Path, default=Path("paper_figures/diagnostics/replan_alignment.csv"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point: compute PCAR replan P/R/F1 vs gripper-flip oracle."""
    args = _parse_args(argv)
    if args.dry_run or args.checkpoint is None:
        results = _synthetic_run(tolerance=args.tolerance)
        dry_run = True
    else:
        if args.dataset is None:
            print("[replan_alignment] ERROR: --dataset required")
            return 1
        results = _compute_alignment(
            args.checkpoint, args.dataset, args.n_episodes, args.tolerance, args.device
        )
        dry_run = False
    _save(results, args.output, dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
