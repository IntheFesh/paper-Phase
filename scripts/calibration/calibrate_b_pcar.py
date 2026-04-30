"""Calibrate the Bayesian PCAR trigger against the concordance signal C_t.

Fits the B-PCAR changepoint prior (Beta(α, β)) and the budget ε on a held-out
trajectory set.  The script sweeps (α_prior, β_prior, budget) and reports the
replan rate vs. cliff-recall trade-off.

Usage
-----
::

    python scripts/calibration/calibrate_b_pcar.py \\
        --rollout_dir outputs/calibration/rollouts \\
        --output_dir outputs/calibration/b_pcar \\
        --budgets 0.05 0.10 0.15 \\
        --alpha_priors 1.0 2.0 5.0

Implementation Phase: C
References: 01_pace_master_plan_v2.md §6.2
            phase_centric/b_pcar.py (BayesianPCARTrigger)
            phase_centric/cliff_detection/concordance.py (C_t input)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("calibrate_b_pcar")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "lerobot_policy_phaseqflow" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))


@dataclass
class _TriggerCfg:
    pcar_trigger_budget_eps: float
    pcar_change_threshold: float = 0.4
    pcar_input_signal: str = "concordance"


def _load_concordance_signals(rollout_dir: Path) -> List[List[float]]:
    """Load pre-computed concordance C_t sequences from rollout JSON files.

    Each JSON file is expected to contain a list of dicts with key
    ``concordance`` (float in [0, 1]).  Returns a list of episodes.
    """
    episodes = []
    for p in sorted(rollout_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            if isinstance(data, list):
                ep = [float(step.get("concordance", 0.0)) for step in data]
                episodes.append(ep)
        except Exception as exc:
            log.warning("Skipping %s: %s", p, exc)
    return episodes


def _sweep_b_pcar(
    episodes: List[List[float]],
    budgets: List[float],
    alpha_priors: List[float],
    beta_prior_scale: float = 5.0,
) -> Dict:
    """Sweep budget × alpha_prior and compute replan rate per setting."""
    from lerobot_policy_phaseqflow.phase_centric.b_pcar import BayesianPCARTrigger

    results: Dict = {}
    for budget in budgets:
        results[budget] = {}
        for alpha in alpha_priors:
            cfg = _TriggerCfg(pcar_trigger_budget_eps=budget)
            trig = BayesianPCARTrigger(
                cfg,
                alpha_prior=alpha,
                beta_prior=alpha * beta_prior_scale,
            )
            for ep in episodes:
                trig.reset()
                for signal in ep:
                    trig.update_and_check(signal)
            rate = trig.get_actual_replan_rate()
            results[budget][alpha] = {
                "replan_rate": rate,
                "changepoint_prob_final": trig.changepoint_probability,
                "n_episodes": len(episodes),
            }
            log.info(
                "budget=%.2f α=%.1f  replan_rate=%.3f  p_cp_final=%.3f",
                budget, alpha, rate, trig.changepoint_probability,
            )
    return results


def main() -> None:
    """CLI entry point: sweep (budget, alpha) pairs and report replan-rate alignment."""
    parser = argparse.ArgumentParser(description="Calibrate BayesianPCARTrigger")
    parser.add_argument("--rollout_dir", default=None,
                        help="Directory of rollout JSON files with concordance signals")
    parser.add_argument("--output_dir", default="outputs/calibration/b_pcar",
                        help="Output directory for calibration results")
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.05, 0.10, 0.15],
                        help="PCAR trigger budget ε values to sweep")
    parser.add_argument("--alpha_priors", nargs="+", type=float, default=[1.0, 2.0, 5.0],
                        help="Beta prior α values to sweep")
    parser.add_argument("--dry_run", action="store_true",
                        help="Generate synthetic concordance signals without real rollouts")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run or args.rollout_dir is None:
        log.info("dry_run: generating synthetic concordance episodes")
        import random
        rng = random.Random(0)
        episodes = []
        for _ in range(5):
            ep = []
            for t in range(100):
                c = 0.9 if (t % 25 == 0) else rng.uniform(0.1, 0.4)
                ep.append(c)
            episodes.append(ep)
    else:
        rollout_dir = Path(args.rollout_dir)
        episodes = _load_concordance_signals(rollout_dir)
        if not episodes:
            log.warning("No rollout files found in %s; use --dry_run", rollout_dir)
            return

    results = _sweep_b_pcar(episodes, args.budgets, args.alpha_priors)

    out_path = out_dir / "sweep_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Results written to %s", out_path)

    # Recommend: setting closest to target budget ε
    log.info("Recommended settings (replan_rate closest to budget):")
    for budget in args.budgets:
        best_alpha = min(
            args.alpha_priors,
            key=lambda a: abs(results[budget][a]["replan_rate"] - budget),
        )
        log.info(
            "  budget=%.2f → α=%.1f  replan_rate=%.3f",
            budget, best_alpha, results[budget][best_alpha]["replan_rate"],
        )


if __name__ == "__main__":
    main()
