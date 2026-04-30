#!/usr/bin/env python

"""Verification: does beta_t align with real phase boundaries, and is its density reasonable.

Motivation
----------
Round 4's ``PhasePosteriorEstimator`` maps a pair of smoothed adjacent
posteriors ``(p_hat_t, p_hat_{t-1})`` to a boundary signal
``beta_t in [0, 1]`` via Bhattacharyya distance. Before moving to
Round 5/6/7 we want to confirm:

1. The **peak positions** of beta_t align with ground-truth phase boundaries.
2. The **density** of beta_t (e.g. the fraction of time with beta > 0.5) is
   in a reasonable band: too high means false positives, too low means
   insufficient sensitivity.

Test data
---------
Reuse Round 1's ``make_synthetic_demos``: each demo has ``n_phases``
equal-length phases, ``actions[:, -1]`` is a gripper proxy, and gripper
transitions mark the ground-truth boundaries.

Note: the ``phase_logits`` of the Round 3-trained planner are identifiable
only after a real RTX 5070 run; a CPU smoke cannot expect an untrained
planner to emit correct phase distributions. So **this script tests the
estimator module in isolation**: we synthesise "pseudo-injected planner"
logits from the ground-truth phase ids (with smooth transitions plus noise
to mimic real uncertainty) and check whether beta_t aligns with the gripper
transitions.

This validates the Round 4 algorithm itself. The integration test that
attaches a real planner is run in Round 5+.

Run
---

    python scripts/verification/verify_phase_posterior.py --num_demos 50 --transition_steps 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "lerobot_policy_phaseqflow" / "src"
_SCRIPTS = _REPO_ROOT / "scripts"
_DIAGNOSTICS_DIR = _REPO_ROOT / "scripts" / "diagnostics"
for p in (_PKG_SRC, _SCRIPTS, _DIAGNOSTICS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig # noqa: E402
from lerobot_policy_phaseqflow.phase_centric.phase_posterior import ( # noqa: E402
    PhasePosteriorEstimator, boundary_prob_from_logits,
)

from diagnostic_utils.synthetic_demos import make_synthetic_demos # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("verify_phase_posterior")


def _gt_phase_ids(demo) -> np.ndarray:
    """Recover ground-truth phase ids from a demo's state-progress channel.

    ``actions[:, -1]`` equals ``phase_idx % 2``, so gripper transitions are
    boundaries; but two same-parity phase switches do not flip the gripper.
    A safer path: use ``states[:, 2] ~ (phase_idx + 1) / n_phases`` and
    invert to ``phase_idx``.
    """
    states = demo.states
    progress = states[:, 2]
    uniq = np.unique(progress)
    n_phases = int(uniq.shape[0])
    idx_lookup = {float(v): i for i, v in enumerate(sorted(uniq))}
    pid = np.array([idx_lookup[float(v)] for v in progress], dtype=np.int64)
    return pid, n_phases


def _gt_boundaries(pid: np.ndarray) -> np.ndarray:
    """Timestamps of phase transitions (excluding t=0)."""
    return np.where(np.diff(pid) != 0)[0] + 1


def _synthesize_phase_logits(
    pid: np.ndarray,
    K: int,
    transition_steps: int,
    strong_logit: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate an identifiable-but-imperfect planner.

    Rules:
    - Interior time steps: set ``strong_logit`` on ``pid[t]`` and
      ``-strong_logit`` on the others.
    - Within ``transition_steps`` of each boundary: linearly interpolate the
      old-phase and new-phase logits to mimic a smooth transition.
    - Add Gaussian noise ``N(0, noise_std)`` to the entire logits array.

    Parameters
    ----------
    pid : ``(T,)`` GT phase id in ``{0, ..., K-1}``. Requires ``max(pid) < K``.
    K : planner logits dimensionality (usually ``num_skills = 8``).
    transition_steps : soft-transition width on each side of a boundary.
    strong_logit : magnitude of the winning-phase logit; larger means a
        sharper softmax.
    noise_std : std of the Gaussian noise added to the whole logits tensor.

    Returns
    -------
    (T, K) np.ndarray.
    """
    assert int(pid.max()) < K, f"pid max {pid.max()} >= K={K}"
    T = int(pid.shape[0])
    logits = np.full((T, K), -strong_logit, dtype=np.float32)
    for t in range(T):
        logits[t, int(pid[t])] = strong_logit

    m = max(0, int(transition_steps))
    if m > 0:
        boundaries = _gt_boundaries(pid)
        for b in boundaries:
            old = int(pid[max(0, b - 1)])
            new = int(pid[b])
            for off in range(-m, m + 1):
                t = b + off
                if t < 0 or t >= T:
                    continue
                w_new = 0.5 * (1.0 + off / max(m, 1))
                w_new = float(np.clip(w_new, 0.0, 1.0))
                w_old = 1.0 - w_new
                row = np.full(K, -strong_logit, dtype=np.float32)
                row[old] = w_old * strong_logit + w_new * (-strong_logit)
                row[new] = w_new * strong_logit + w_old * (-strong_logit)
                logits[t] = row

    logits = logits + rng.normal(0, noise_std, size=logits.shape).astype(np.float32)
    return logits


def _find_peaks(beta: np.ndarray, min_distance: int, min_height: float) -> np.ndarray:
    """Lightweight 1D peak finder: local max + ``>= min_height`` + ``min_distance`` separation.

    Avoids ``scipy.signal`` to stay lightweight; results are somewhat
    threshold-sensitive but sufficient for a peak-alignment acceptance test.
    """
    T = int(beta.shape[0])
    peaks: List[int] = []
    for t in range(1, T - 1):
        if beta[t] < min_height:
            continue
        if beta[t] < beta[t - 1] or beta[t] < beta[t + 1]:
            continue
        if peaks and (t - peaks[-1]) < min_distance:
            if beta[t] > beta[peaks[-1]]:
                peaks[-1] = t
            continue
        peaks.append(t)
    return np.array(peaks, dtype=np.int64)


def _avg_peak_distance(peaks: np.ndarray, boundaries: np.ndarray) -> float:
    """Mean distance from each predicted peak to its nearest GT boundary (timesteps)."""
    if peaks.size == 0 or boundaries.size == 0:
        return float("inf")
    dists: List[float] = []
    for p in peaks:
        dists.append(float(np.min(np.abs(boundaries - p))))
    return float(np.mean(dists))


def _run_one_demo(
    demo,
    K: int,
    alpha: float,
    transition_steps: int,
    strong_logit: float,
    noise_std: float,
    peak_min_distance: int,
    peak_min_height: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Process one demo and return per-step arrays (pid, boundaries, beta, peaks, ...)."""
    pid, n_phases = _gt_phase_ids(demo)
    boundaries = _gt_boundaries(pid)
    logits = _synthesize_phase_logits(
        pid=pid, K=K,
        transition_steps=transition_steps,
        strong_logit=strong_logit,
        noise_std=noise_std,
        rng=rng,
    )
    with torch.no_grad():
        out = boundary_prob_from_logits(
            torch.from_numpy(logits),
            alpha=alpha,
        )
    p_hat = out["p_hat"].numpy()
    beta = out["beta"].numpy()
    top5_peaks = _find_peaks(beta, peak_min_distance, peak_min_height)[:5]
    return {
        "pid": pid,
        "boundaries": boundaries,
        "gripper": demo.actions[:, -1],
        "logits": logits,
        "p_hat": p_hat,
        "beta": beta,
        "peaks": top5_peaks,
        "n_phases": n_phases,
    }


def _plot_alignment(
    per_demo: List[Dict[str, np.ndarray]],
    out_path: Path,
    num_panels: int = 4,
) -> None:
    """Plot ``num_panels`` alignment panels: beta_t trace + gripper + GT boundaries."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e: # noqa: BLE001
        log.warning("matplotlib unavailable; skipping plot: %s", e)
        return

    n = min(int(num_panels), len(per_demo))
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.2 * n), squeeze=False)
    for i in range(n):
        d = per_demo[i]
        ax = axes[i][0]
        T = int(d["beta"].shape[0])
        ax.plot(np.arange(T), d["beta"], color="tab:blue", label="beta_t (Bhattacharyya)")
        ax.plot(np.arange(T), d["gripper"], color="tab:orange", alpha=0.6, label="gripper (GT)")
        for b in d["boundaries"]:
            ax.axvline(int(b), color="k", linestyle="--", alpha=0.3)
        ax.scatter(
            d["peaks"], d["beta"][d["peaks"]] if d["peaks"].size else [],
            color="red", zorder=5, label="detected peaks",
        )
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel(f"demo {i}")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
        if i == n - 1:
            ax.set_xlabel("timestep")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    log.info("saved alignment figure: %s", out_path)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--num_demos", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="artifacts/phase_posterior")
    p.add_argument(
        "--alpha", type=float, default=0.9,
        help="EMA coefficient alpha. Default 0.9 (decision-layer beta); pass "
             "--alpha 0.3 to reproduce the config default (training-layer beta).",
    )
    p.add_argument("--transition_steps", type=int, default=1,
                   help="Soft-transition steps on each side of a boundary. "
                        "0 = hard switch; >0 = linear-interpolation width.")
    p.add_argument("--strong_logit", type=float, default=3.0,
                   help="Winning-phase logit magnitude; larger means sharper softmax.")
    p.add_argument("--noise_std", type=float, default=0.3,
                   help="Gaussian-noise sigma added to logits. Higher noise "
                        "smooths beta but hurts alignment.")
    p.add_argument("--peak_min_distance", type=int, default=5)
    p.add_argument("--peak_min_height", type=float, default=0.10)
    p.add_argument("--avg_distance_threshold", type=float, default=3.0)
    p.add_argument("--density_probe", type=float, default=0.15,
                   help="Fraction of timesteps with beta above this probe is "
                        "used for the density check; default 0.15.")
    p.add_argument("--density_lo", type=float, default=0.03)
    p.add_argument("--density_hi", type=float, default=0.30)
    return p.parse_args()


def main() -> int:
    """Run the posterior alignment verification and emit report + figures."""
    args = _parse_args()
    rng = np.random.default_rng(int(args.seed))

    cfg = PhaseQFlowConfig(use_phase_boundary_posterior=True)
    alpha = float(args.alpha) if args.alpha is not None else float(cfg.phase_posterior_smooth_alpha)
    K = int(cfg.num_skills)

    demos = make_synthetic_demos(
        num_demos=int(args.num_demos),
        action_dim=int(cfg.action_dim),
        state_dim=int(cfg.state_dim),
        history_dim=int(cfg.history_dim),
        seed=int(args.seed),
    )

    per_demo: List[Dict[str, np.ndarray]] = []
    all_distances: List[float] = []
    total_T = 0
    total_above = 0
    for di, demo in enumerate(demos):
        rec = _run_one_demo(
            demo, K=K, alpha=alpha,
            transition_steps=int(args.transition_steps),
            strong_logit=float(args.strong_logit),
            noise_std=float(args.noise_std),
            peak_min_distance=int(args.peak_min_distance),
            peak_min_height=float(args.peak_min_height),
            rng=rng,
        )
        per_demo.append(rec)
        all_distances.append(_avg_peak_distance(rec["peaks"], rec["boundaries"]))
        total_T += int(rec["beta"].shape[0])
        total_above += int((rec["beta"] > float(args.density_probe)).sum())

    avg_distance = float(np.nanmean([d for d in all_distances if np.isfinite(d)]))
    density = total_above / max(1, total_T)

    passes_alignment = avg_distance <= float(args.avg_distance_threshold)
    passes_density = (
        float(args.density_lo) <= density <= float(args.density_hi)
    )
    verdict = "PASS" if (passes_alignment and passes_density) else "FAIL"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_alignment(per_demo, figures_dir / "beta_alignment.png", num_panels=4)

    report = {
        "num_demos": int(args.num_demos),
        "alpha": alpha,
        "K": int(K),
        "transition_steps": int(args.transition_steps),
        "strong_logit": float(args.strong_logit),
        "noise_std": float(args.noise_std),
        "avg_peak_to_boundary_distance": avg_distance,
        "avg_distance_threshold": float(args.avg_distance_threshold),
        "density_probe": float(args.density_probe),
        "beta_above_probe_frac": density,
        "density_lo": float(args.density_lo),
        "density_hi": float(args.density_hi),
        "verdict": verdict,
        "per_demo_avg_distances": [
            None if not np.isfinite(d) else float(d) for d in all_distances
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    md = _render_report(report)
    (out_dir / "report.md").write_text(md)
    log.info("report written: %s", out_dir / "report.md")
    log.info("verdict: %s", verdict)
    return 0 if verdict == "PASS" else 1


def _render_report(rep: Dict[str, object]) -> str:
    """Render the markdown report body from a result dict."""
    probe = float(rep["density_probe"])
    lines = [
        "# Round 4 - Phase Posterior (beta_t) Alignment Report",
        "",
        f"**Verdict**: `{rep['verdict']}`",
        "",
        f"- num_demos: {rep['num_demos']}",
        f"- alpha (EMA): {rep['alpha']}",
        f"- planner K (from num_skills): {rep['K']}",
        f"- synthesized transition_steps: {rep['transition_steps']}",
        f"- strong_logit: {rep['strong_logit']}",
        f"- noise_std: {rep['noise_std']}",
        "",
        "## Alignment (peaks of beta_t vs GT phase boundaries)",
        "",
        f"- Mean distance (top-5 peaks -> nearest GT boundary): "
        f"**{rep['avg_peak_to_boundary_distance']:.3f}** steps "
        f"(threshold <= {rep['avg_distance_threshold']}).",
        "",
        "## Density",
        "",
        f"- Fraction of timesteps with beta > {probe}: "
        f"**{rep['beta_above_probe_frac']:.3%}** "
        f"(target {rep['density_lo']:.0%} <= value <= {rep['density_hi']:.0%}).",
        "",
        "## Artifacts",
        "",
        "- `figures/beta_alignment.png` - 4 panels: beta_t (blue), gripper proxy "
        "(orange), GT boundaries (dashed), detected peaks (red dots).",
        "- `report.json` - full numeric payload.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
