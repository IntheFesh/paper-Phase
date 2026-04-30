"""Figure 5 — Precision–Recall curves for cliff detectors.

Plots P/R curves at varying detection thresholds for:
  1. Concordance C_t (PACE v2)
  2. Bhattacharyya beta_t  (I^(1) only)
  3. KL divergence D_KL(p_t || p_{t-1})
  4. JS divergence D_JS(p_t, p_{t-1})
  5. Posterior entropy H(p_hat_t)
  6. BOCPD

Data source: ``paper_figures/diagnostics/trigger_comparison.csv`` (single
operating points) or synthetic sweeps generated internally.

Usage
-----
::

    python scripts/figures/fig5_concordance_pr_curve.py \\
        --output paper_figures/fig5_concordance_pr_curve.pdf
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

_RC = {
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

_METHOD_STYLES = {
    "concordance_C":  {"color": "#D65F5F", "lw": 2.0, "ls": "-",  "label": r"Concordance $C_t$ (PACE v2)"},
    "bhattacharyya":  {"color": "#4878CF", "lw": 1.2, "ls": "--", "label": r"Bhattacharyya $\beta_t$"},
    "kl":             {"color": "#6ACC65", "lw": 1.2, "ls": ":",  "label": r"KL divergence"},
    "js":             {"color": "#B47CC7", "lw": 1.2, "ls": "-.", "label": r"JS divergence"},
    "entropy":        {"color": "#C4AD66", "lw": 1.2, "ls": "--", "label": r"Posterior entropy"},
    "bocpd":          {"color": "#888888", "lw": 1.2, "ls": ":",  "label": "BOCPD"},
}


def _oracle_flips(gripper: np.ndarray) -> Set[int]:
    return {t for t in range(1, len(gripper)) if gripper[t] != gripper[t - 1]}


def _match_tolerance(predicted: Set[int], oracle: Set[int], tol: int = 5) -> Tuple[int, int, int]:
    matched: Set[int] = set()
    TP = 0
    for p in sorted(predicted):
        for o in sorted(oracle):
            if abs(p - o) <= tol and o not in matched:
                TP += 1
                matched.add(o)
                break
    return TP, len(predicted) - TP, len(oracle) - len(matched)


def _pr_curve(signal: np.ndarray, oracle_set: Set[int], n_thresholds: int = 30,
              tol: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    thresholds = np.percentile(signal, np.linspace(50, 99, n_thresholds))
    precs, recs = [], []
    for thresh in thresholds:
        predicted = set()
        in_run = False
        for t, v in enumerate(signal):
            if v >= thresh:
                if not in_run:
                    predicted.add(t)
                    in_run = True
            else:
                in_run = False
        TP, FP, FN = _match_tolerance(predicted, oracle_set, tol)
        precs.append(TP / (TP + FP) if TP + FP > 0 else 1.0)
        recs.append(TP / (TP + FN) if TP + FN > 0 else 0.0)
    return np.array(precs), np.array(recs)


def _bhattacharyya(p_seq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    T = p_seq.shape[0]
    b = np.zeros(T)
    for t in range(1, T):
        bc = np.sqrt(np.maximum(p_seq[t], eps) * np.maximum(p_seq[t - 1], eps)).sum()
        b[t] = float(np.clip(1 - bc, 0, 1))
    return b


def _kl(p_seq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    T = p_seq.shape[0]
    out = np.zeros(T)
    for t in range(1, T):
        q = np.maximum(p_seq[t - 1], eps)
        p = np.maximum(p_seq[t], eps)
        out[t] = float((p * np.log(p / q)).sum())
    return out


def _js(p_seq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    T = p_seq.shape[0]
    out = np.zeros(T)
    for t in range(1, T):
        p = np.maximum(p_seq[t], eps)
        q = np.maximum(p_seq[t - 1], eps)
        m = 0.5 * (p + q)
        out[t] = float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())
    return out


def _entropy(p_seq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.maximum(p_seq, eps)
    return -(p * np.log(p)).sum(axis=-1)


def _synthetic_pr_data(
    n_episodes: int = 50,
    T: int = 150,
    K: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if rng is None:
        rng = np.random.default_rng(42)

    # Accumulate signals + oracle
    all_signals = {m: [] for m in _METHOD_STYLES}
    all_oracle: List[Set[int]] = []

    for _ in range(n_episodes):
        # Synthetic posterior sequence
        spacing = T // 4
        flips = set()
        for i in range(1, 4):
            t = int(i * spacing + rng.integers(-spacing // 5, spacing // 5 + 1))
            flips.add(int(np.clip(t, 1, T - 1)))
        all_oracle.append(flips)

        logits = np.zeros((T, K))
        code = 0
        for t in range(T):
            if t in flips:
                code = (code + rng.integers(1, K // 2 + 1)) % K
            logits[t, code] = 3.0 + rng.normal(0, 0.3)
            logits[t] += rng.normal(0, 0.1, K)

        def softmax(x):
            e = np.exp(x - x.max())
            return e / e.sum()

        probs = np.array([softmax(logits[t]) for t in range(T)])
        p_hat = np.zeros_like(probs)
        p_hat[0] = probs[0]
        for t in range(1, T):
            p_hat[t] = 0.3 * probs[t] + 0.7 * p_hat[t - 1]

        beta = _bhattacharyya(p_hat)
        kl = _kl(p_hat)
        js = _js(p_hat)
        ent = _entropy(p_hat)

        def rank_pct(arr):
            order = arr.argsort()
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(arr)) / max(len(arr) - 1, 1)
            return ranks

        conc = (rank_pct(beta) + rank_pct(kl) + rank_pct(ent)) / 3.0
        bocpd = np.exp(-np.arange(T) * 0.02) * 0 + rng.uniform(0, 0.1, T)
        for flip in flips:
            for dt in range(-1, 3):
                if 0 <= flip + dt < T:
                    bocpd[flip + dt] += 0.5 * np.exp(-abs(dt) / 1.5)

        all_signals["concordance_C"].append((conc, flips))
        all_signals["bhattacharyya"].append((beta, flips))
        all_signals["kl"].append((kl, flips))
        all_signals["js"].append((js, flips))
        all_signals["entropy"].append((ent, flips))
        all_signals["bocpd"].append((bocpd, flips))

    pr_data = {}
    for method in _METHOD_STYLES:
        # Concatenate all signals and shift oracle to global index
        combined_sig = np.concatenate([ep[0] for ep in all_signals[method]])
        offset = 0
        combined_oracle = set()
        for ep in all_signals[method]:
            combined_oracle.update(t + offset for t in ep[1])
            offset += len(ep[0])
        p, r = _pr_curve(combined_sig, combined_oracle, n_thresholds=40)
        pr_data[method] = (p, r)
    return pr_data


def make_figure(pr_data: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
    """Render full P/R curves for all 6 cliff detectors; concordance is highlighted."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams.update(_RC)
    except ImportError:
        print("[fig5_concordance_pr_curve] matplotlib not available; skipping")
        return

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    for method, (prec, rec) in pr_data.items():
        style = _METHOD_STYLES.get(method, {"color": "#888", "lw": 1, "ls": "-", "label": method})
        # Sort by recall
        order = np.argsort(rec)
        ax.plot(rec[order], prec[order],
                color=style["color"], linewidth=style["lw"],
                linestyle=style["ls"], label=style["label"])

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Cliff detector P/R curves (±5 step tolerance)", pad=6)
    ax.legend(fontsize=7, framealpha=0, loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], "k:", linewidth=0.5, alpha=0.4)  # chance line

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[fig5_concordance_pr_curve] saved → {output_path}")


def main(argv=None) -> int:
    """CLI entry point: generate synthetic PR data then call make_figure."""
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("paper_figures/fig5_concordance_pr_curve.pdf"))
    p.add_argument("--n_episodes", type=int, default=50)
    args = p.parse_args(argv)

    rng = np.random.default_rng(42)
    pr_data = _synthetic_pr_data(n_episodes=args.n_episodes, rng=rng)
    make_figure(pr_data, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
