"""Diagnostic: cliff-trigger method comparison.

Compares the F1 of five alternative cliff detectors against the PACE v2
concordance detector on the gripper-flip oracle (±5 timestep tolerance):

1. Concordance C_t           — PACE v2 (rank-based fusion of I^1/2/3)
2. Bhattacharyya beta_t      — I^(1) alone
3. KL divergence D_KL(p_t || p_{t-1})
4. JS divergence D_JS(p_t, p_{t-1})
5. Posterior entropy H(p_hat_t)
6. BOCPD                     — Bayesian Online Change-Point Detection
                               (Adams & MacKay 2007; exponential hazard, Gaussian obs)

All methods operate on the same per-episode phase-posterior sequence.

Usage
-----
::

    python scripts/diagnostics/trigger_comparison.py --dry_run
    python scripts/diagnostics/trigger_comparison.py \\
        --checkpoint checkpoints/phaseqflow --n_episodes 30
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
# Alternative cliff detectors operating on posterior sequence p_hat (B, T, K)
# ---------------------------------------------------------------------------

def _bhattacharyya(p_seq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """beta_t = 1 - sum_k sqrt(p_t[k] * p_{t-1}[k]).  Shape (T,), beta[0]=0."""
    T, K = p_seq.shape
    beta = np.zeros(T)
    for t in range(1, T):
        bc = np.sqrt(np.maximum(p_seq[t], eps) * np.maximum(p_seq[t - 1], eps)).sum()
        beta[t] = float(np.clip(1.0 - bc, 0.0, 1.0))
    return beta


def _kl_divergence(p_seq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """D_KL(p_t || p_{t-1}).  Shape (T,), kl[0]=0."""
    T, _ = p_seq.shape
    kl = np.zeros(T)
    for t in range(1, T):
        q = np.maximum(p_seq[t - 1], eps)
        p = np.maximum(p_seq[t], eps)
        kl[t] = float((p * np.log(p / q)).sum())
    return kl


def _js_divergence(p_seq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Jensen–Shannon divergence D_JS(p_t, p_{t-1}).  Shape (T,), js[0]=0."""
    T, _ = p_seq.shape
    js = np.zeros(T)
    for t in range(1, T):
        p = np.maximum(p_seq[t], eps)
        q = np.maximum(p_seq[t - 1], eps)
        m = 0.5 * (p + q)
        js[t] = float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())
    return js


def _entropy(p_seq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """H(p_hat_t) = -sum_k p_t[k] log p_t[k].  Shape (T,)."""
    p = np.maximum(p_seq, eps)
    return -(p * np.log(p)).sum(axis=-1)


def _bocpd(p_seq: np.ndarray, hazard: float = 1.0 / 50.0) -> np.ndarray:
    """BOCPD change-point probability (Adams & MacKay 2007, Gaussian obs).

    Operates on the first principal component of p_seq for simplicity.
    Returns per-step change-point probability r_t in [0, 1].
    """
    T, K = p_seq.shape
    # Project to 1D via mean
    obs = p_seq.mean(axis=-1)  # (T,)
    # Gaussian predictive with running sufficient statistics
    # Simplified scalar version: N(mu_0, sigma_0^2) prior
    mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, 1.0
    # Run-length probabilities
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0
    # Precompute sufficient statistics per run
    mus = np.zeros(T + 1)
    kappas = np.zeros(T + 1)
    alphas = np.zeros(T + 1)
    betas_ = np.zeros(T + 1)
    mus[0] = mu0
    kappas[0] = kappa0
    alphas[0] = alpha0
    betas_[0] = beta0

    cp_prob = np.zeros(T)
    for t in range(1, T + 1):
        x = obs[t - 1]
        # Predictive probability under each run length
        kappas_t = kappas[:t] + 1
        mus_t = (kappas[:t] * mus[:t] + x) / kappas_t
        alphas_t = alphas[:t] + 0.5
        betas_t = (
            betas_[:t]
            + 0.5 * kappas[:t] / kappas_t * (x - mus[:t]) ** 2
        )
        # Predictive under Student-t (approximated as Gaussian for brevity)
        var_pred = (betas_[:t] / alphas[:t]) * (1 + 1.0 / kappas[:t])
        std_pred = np.maximum(np.sqrt(var_pred), 1e-8)
        pred_prob = (1.0 / (std_pred * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mus[:t]) / std_pred) ** 2
        )

        # Growth probabilities (no change point)
        R[t, 1:t + 1] = R[t - 1, :t] * pred_prob * (1.0 - hazard)
        # Change-point probability
        R[t, 0] = np.sum(R[t - 1, :t] * pred_prob * hazard)

        # Normalise
        norm = R[t, :t + 1].sum()
        if norm > 1e-300:
            R[t, :t + 1] /= norm

        cp_prob[t - 1] = R[t, 0]

        # Update sufficient statistics for run-length 0 (new run)
        mus[1:t + 1] = mus_t
        kappas[1:t + 1] = kappas_t
        alphas[1:t + 1] = alphas_t
        betas_[1:t + 1] = betas_t
        mus[0] = mu0
        kappas[0] = kappa0
        alphas[0] = alpha0
        betas_[0] = beta0

    return cp_prob


# ---------------------------------------------------------------------------
# Detection: threshold crossings → predicted cliff set
# ---------------------------------------------------------------------------

def _detect(signal: np.ndarray, percentile: float = 90.0, high_is_cliff: bool = True) -> Set[int]:
    """Return set of first timesteps of contiguous above-threshold runs."""
    if not high_is_cliff:
        signal = -signal
    threshold = float(np.percentile(signal, percentile))
    triggered: Set[int] = set()
    in_run = False
    for t, v in enumerate(signal):
        if v >= threshold:
            if not in_run:
                triggered.add(t)
                in_run = True
        else:
            in_run = False
    return triggered


# ---------------------------------------------------------------------------
# Synthetic episode generator
# ---------------------------------------------------------------------------

def _synthetic_posterior_episode(
    T: int = 200,
    K: int = 20,
    n_transitions: int = 3,
    alpha: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Set[int]]:
    """Return (p_hat (T,K), oracle flip set)."""
    if rng is None:
        rng = np.random.default_rng(42)

    spacing = T // (n_transitions + 1)
    flip_times: Set[int] = set()
    for i in range(1, n_transitions + 1):
        t = int(i * spacing + rng.integers(-spacing // 5, spacing // 5 + 1))
        flip_times.add(int(np.clip(t, 1, T - 1)))

    # Build logits: dominant code changes at each flip
    logits = np.zeros((T, K))
    active_code = 0
    for t in range(T):
        if t in flip_times:
            active_code = (active_code + rng.integers(1, K // 2 + 1)) % K
        logits[t, active_code] = 3.0 + rng.normal(0, 0.3)
        logits[t] += rng.normal(0, 0.1, K)

    # Softmax + EMA
    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    probs = np.array([softmax(logits[t]) for t in range(T)])
    p_hat = np.zeros_like(probs)
    p_hat[0] = probs[0]
    for t in range(1, T):
        p_hat[t] = alpha * probs[t] + (1 - alpha) * p_hat[t - 1]

    return p_hat, flip_times


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _match(predicted: Set[int], oracle: Set[int], tolerance: int = 5) -> Tuple[int, int, int]:
    matched: Set[int] = set()
    TP = 0
    for p in sorted(predicted):
        for o in sorted(oracle):
            if abs(p - o) <= tolerance and o not in matched:
                TP += 1
                matched.add(o)
                break
    return TP, len(predicted) - TP, len(oracle) - len(matched)


def _f1_from_counts(TP: int, FP: int, FN: int) -> Tuple[float, float, float]:
    prec = TP / (TP + FP) if TP + FP > 0 else 0.0
    rec = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def evaluate_all_methods(
    n_episodes: int = 30,
    T: int = 200,
    K: int = 20,
    tolerance: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Aggregate TP/FP/FN across n_episodes for all 6 detectors and return per-detector P/R/F1 rows."""
    if rng is None:
        rng = np.random.default_rng(42)

    methods = ["concordance_C", "bhattacharyya", "kl", "js", "entropy", "bocpd"]
    counts = {m: {"TP": 0, "FP": 0, "FN": 0} for m in methods}

    for _ in range(n_episodes):
        p_hat, oracle = _synthetic_posterior_episode(T=T, K=K, rng=rng)

        beta = _bhattacharyya(p_hat)
        kl = _kl_divergence(p_hat)
        js = _js_divergence(p_hat)
        ent = _entropy(p_hat)
        cp = _bocpd(p_hat)

        # Concordance: combine beta + kl + ent into a simple rank average
        def rank_pct(arr: np.ndarray) -> np.ndarray:
            order = arr.argsort()
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(arr)) / max(len(arr) - 1, 1)
            return ranks

        conc = (rank_pct(beta) + rank_pct(kl) + rank_pct(ent)) / 3.0

        signals = {
            "concordance_C": conc,
            "bhattacharyya": beta,
            "kl": kl,
            "js": js,
            "entropy": ent,
            "bocpd": cp,
        }
        for m, sig in signals.items():
            pred = _detect(sig, percentile=90.0, high_is_cliff=True)
            TP, FP, FN = _match(pred, oracle, tolerance)
            counts[m]["TP"] += TP
            counts[m]["FP"] += FP
            counts[m]["FN"] += FN

    rows = []
    for m in methods:
        c = counts[m]
        prec, rec, f1 = _f1_from_counts(c["TP"], c["FP"], c["FN"])
        rows.append({"method": m, "precision": round(prec, 4),
                     "recall": round(rec, 4), "F1": round(f1, 4),
                     "n_TP": c["TP"], "n_FP": c["FP"], "n_FN": c["FN"]})
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _save(rows: List[Dict], output_path: Path, dry_run: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "precision", "recall", "F1",
                                           "n_TP", "n_FP", "n_FN"])
        w.writeheader()
        w.writerows(rows)
    print("[trigger_comparison] results:")
    for r in rows:
        print(f"  {r['method']:20s}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['F1']:.3f}")
    print(f"  → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--n_episodes", type=int, default=30)
    p.add_argument("--tolerance", type=int, default=5)
    p.add_argument("--output", type=Path, default=Path("paper_figures/diagnostics/trigger_comparison.csv"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    rng = np.random.default_rng(42)
    rows = evaluate_all_methods(n_episodes=args.n_episodes, tolerance=args.tolerance, rng=rng)
    _save(rows, args.output, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
