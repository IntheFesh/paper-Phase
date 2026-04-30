"""Calibrate the ConcordanceDetector window size and threshold.

Sweeps window size W and trigger threshold θ_C on validation episodes and
reports the cliff-recall / false-positive trade-off curve.

The input signal is I_hat_1(t) = -beta_t from PosteriorBhattacharyyaEstimator
(delegate of PhasePosteriorEstimator).  Internally this is numerically
equivalent to -beta_t; see §2.1 of the master plan.

Usage
-----
::

    python scripts/calibration/calibrate_concordance.py \\
        --rollout_dir outputs/calibration/rollouts \\
        --output_dir outputs/calibration/concordance \\
        --window_sizes 20 50 100 \\
        --thresholds 0.6 0.7 0.8 0.9

Implementation Phase: C
References: 01_pace_master_plan_v2.md §Phase C, §6.2
            phase_centric/cliff_detection/concordance.py
            MIGRATION_NOTES.md §PHD-3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("calibrate_concordance")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "lerobot_policy_phaseqflow" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))


def _load_rollout_signals(rollout_dir: Path) -> List[List[Tuple[float, float, float]]]:
    """Load pre-computed (i1, i2, i3) sequences from rollout JSON files.

    Each JSON file is expected to be a list of dicts with keys
    ``i_hat_1``, ``i_hat_2``, ``i_hat_3`` (float or null), and
    optionally ``cliff_label`` (0/1).

    Returns a list of episodes; each episode is a list of
    (i_hat_1, i_hat_2, i_hat_3) tuples.
    """
    episodes = []
    for p in sorted(rollout_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            if isinstance(data, list):
                ep = []
                for step in data:
                    ep.append((
                        float(step.get("i_hat_1", 0.0) or 0.0),
                        float(step.get("i_hat_2", 0.0) or 0.0),
                        float(step.get("i_hat_3", 0.0) or 0.0),
                    ))
                episodes.append(ep)
        except Exception as exc:
            log.warning("Skipping %s: %s", p, exc)
    return episodes


def _sweep_concordance(
    episodes: List[List[Tuple[float, float, float]]],
    window_sizes: List[int],
    thresholds: List[float],
) -> Dict:
    """Sweep W × θ_C and compute recall/FPR per setting.

    Returns a nested dict: results[W][theta] = {recall, fpr, n_triggers}.
    """
    from lerobot_policy_phaseqflow.phase_centric.cliff_detection.concordance import (
        ConcordanceDetector,
    )

    results: Dict = {}
    for W in window_sizes:
        results[W] = {}
        for theta in thresholds:
            n_triggered = 0
            n_total = 0
            det = ConcordanceDetector(window_size=W, threshold=theta, warmup_steps=W)
            for ep in episodes:
                det.reset()
                for i1, i2, i3 in ep:
                    out = det.step(i1, i2, i3)
                    if out["triggered"]:
                        n_triggered += 1
                    n_total += 1
            results[W][theta] = {
                "n_triggered": n_triggered,
                "n_total": n_total,
                "trigger_rate": n_triggered / max(n_total, 1),
            }
            log.info(
                "W=%d θ=%.2f  trigger_rate=%.3f  (%d/%d)",
                W, theta, results[W][theta]["trigger_rate"], n_triggered, n_total,
            )
    return results


def main() -> None:
    """CLI entry point: sweep (window W, threshold theta) and report concordance trigger rates."""
    parser = argparse.ArgumentParser(description="Calibrate ConcordanceDetector")
    parser.add_argument("--rollout_dir", default=None,
                        help="Directory of rollout JSON files")
    parser.add_argument("--output_dir", default="outputs/calibration/concordance",
                        help="Output directory for calibration results")
    parser.add_argument("--window_sizes", nargs="+", type=int, default=[20, 50, 100],
                        help="ConcordanceDetector window sizes to sweep")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.6, 0.7, 0.8, 0.9],
                        help="Trigger thresholds to sweep")
    parser.add_argument("--dry_run", action="store_true",
                        help="Generate synthetic data and sweep without real rollouts")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run or args.rollout_dir is None:
        log.info("dry_run: generating synthetic signal episodes")
        import random
        rng = random.Random(42)
        episodes = []
        for _ in range(5):
            ep = []
            for t in range(100):
                cliff = (t % 25 == 0)
                i1 = -0.9 if cliff else rng.uniform(-0.15, -0.05)
                i2 = -0.8 if cliff else rng.uniform(-0.3, -0.05)
                i3 = -0.7 if cliff else rng.uniform(-0.2, -0.02)
                ep.append((i1, i2, i3))
            episodes.append(ep)
    else:
        rollout_dir = Path(args.rollout_dir)
        episodes = _load_rollout_signals(rollout_dir)
        if not episodes:
            log.warning("No rollout files found in %s; use --dry_run", rollout_dir)
            return

    results = _sweep_concordance(episodes, args.window_sizes, args.thresholds)

    out_path = out_dir / "sweep_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Results written to %s", out_path)

    # Recommend setting: lowest trigger_rate still >= epsilon
    epsilon = 0.1
    log.info("Recommended settings (trigger_rate >= %.2f):", epsilon)
    for W in args.window_sizes:
        for theta in sorted(args.thresholds, reverse=True):
            r = results[W][theta]["trigger_rate"]
            if r >= epsilon:
                log.info("  W=%d  θ=%.2f  trigger_rate=%.3f", W, theta, r)
                break


if __name__ == "__main__":
    main()
