#!/usr/bin/env python

"""Sanity check: PACE-B gating L2-distance visualisation.

Goal
----
Verify that :class:`PhaseMoE`'s smooth-switching gate ``g_t`` meets the design
targets:

- **Boundary instants (large beta)**: ``||g_t - g_{t-1}||_2 > 0.3`` (quick
  switch to a new expert).
- **Phase interior (small beta)**: ``||g_t - g_{t-1}||_2 < 0.05`` (hold the
  current expert; denoise).

Method
------
1. Build a 3-phase posterior sequence:
   - phase 0 (t in [0, 40)): one-hot on expert 0 + noise (std 0.05).
   - phase 1 (t in [40, 80)): one-hot on expert 2.
   - phase 2 (t in [80, 120)): one-hot on expert 1.
2. Set beta_t ~ 0.8 near real boundaries (t in {39..41, 79..81}) and ~ 0.05
   elsewhere.
3. Step ``PhaseMoE.compute_gate(..., training=False)`` to collect g_t, then
   compute adjacent L2 differences split into boundary / interior buckets and
   report means / quantiles.
4. Emit JSON; if matplotlib is available, also save the gate-trajectory PNG.

Usage
-----
::

    python scripts/verification/sanity_pace_b.py --out-json sanity_pace_b.json

Acceptance: return code 0 iff ``boundary L2 mean > 0.3`` and
``interior L2 mean < 0.05``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "lerobot_policy_phaseqflow" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig # noqa: E402
from lerobot_policy_phaseqflow.phase_centric.pace_b_moe import PhaseMoE # noqa: E402


def _make_posterior_sequence(T: int, K: int, boundaries: List[int]) -> np.ndarray:
    """Build a (T, K) softmax-like posterior trajectory.

    ``boundaries`` is a list of time indices where phase changes.
    E.g. ``T=120, boundaries=[40, 80]`` creates 3 phases of length 40 each.
    """
    p = np.zeros((T, K), dtype=np.float32)
    cur_phase = 0
    phase_to_expert = {0: 0, 1: 2, 2: 1}
    for t in range(T):
        if boundaries and t >= boundaries[0]:
            boundaries = boundaries[1:]
            cur_phase += 1
        e = phase_to_expert.get(cur_phase, cur_phase % K)
        p[t, e] = 1.0
    p = p * 0.95 + 0.05 / K
    p = p / p.sum(axis=-1, keepdims=True)
    return p


def _make_beta_sequence(T: int, boundary_times: List[int], window: int = 2) -> np.ndarray:
    """Build a length-``T`` beta array: high near each boundary, low elsewhere."""
    beta = np.full(T, 0.05, dtype=np.float32)
    for bt in boundary_times:
        lo = max(0, bt - window)
        hi = min(T, bt + window + 1)
        beta[lo:hi] = 0.8
    return beta


def run_sanity(
    T: int = 120,
    K: int = 4,
    boundaries: List[int] | None = None,
    seed: int = 0,
    out_json: Path | None = None,
    out_png: Path | None = None,
) -> dict:
    """Run the PhaseMoE gate trajectory test and return the result dict."""
    if boundaries is None:
        boundaries = [40, 80]

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = PhaseQFlowConfig(
        moe_num_experts=K,
        moe_expert_hidden_dim=32,
        moe_switch_kappa=5.0,
        moe_switch_mu=2.0,
        moe_top_k=0,
        fusion_hidden_dim=32,
        latent_dim=16,
    )
    moe = PhaseMoE(cfg).eval()

    p_seq = _make_posterior_sequence(T=T, K=K, boundaries=boundaries)
    beta_seq = _make_beta_sequence(T=T, boundary_times=boundaries, window=2)

    gates: List[np.ndarray] = []
    for t in range(T):
        p_t = torch.from_numpy(p_seq[t:t + 1])
        b_t = torch.from_numpy(beta_seq[t:t + 1])
        g = moe.compute_gate(p_hat=p_t, beta=b_t, training=False)
        gates.append(g.squeeze(0).detach().numpy())
    gate_traj = np.stack(gates, axis=0)

    l2 = np.linalg.norm(gate_traj[1:] - gate_traj[:-1], axis=-1)

    boundary_mask = np.zeros(T - 1, dtype=bool)
    for bt in boundaries:
        for delta in (-1, 0, 1):
            idx = bt + delta
            if 0 <= idx < T - 1:
                boundary_mask[idx] = True
    interior_mask = ~boundary_mask

    boundary_l2 = l2[boundary_mask]
    interior_l2 = l2[interior_mask]

    results = {
        "T": T,
        "K": K,
        "boundaries": boundaries,
        "boundary_l2_mean": float(boundary_l2.mean()),
        "boundary_l2_max": float(boundary_l2.max()),
        "interior_l2_mean": float(interior_l2.mean()),
        "interior_l2_max": float(interior_l2.max()),
        "acceptance_boundary_gt_0p3": bool(boundary_l2.mean() > 0.3),
        "acceptance_interior_lt_0p05": bool(interior_l2.mean() < 0.05),
    }

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2))

    if out_png is not None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
            for k in range(K):
                ax1.plot(gate_traj[:, k], label=f"expert {k}")
            for bt in boundaries:
                ax1.axvline(bt, color="red", alpha=0.3, linestyle="--")
            ax1.set_ylabel("gate weight")
            ax1.set_title("PhaseMoE gate trajectory across phases")
            ax1.legend(loc="center right")

            ax2.plot(np.arange(1, T), l2, label="||g_t - g_{t-1}||_2")
            for bt in boundaries:
                ax2.axvline(bt, color="red", alpha=0.3, linestyle="--")
            ax2.axhline(0.3, color="green", alpha=0.5, label="boundary threshold")
            ax2.axhline(0.05, color="orange", alpha=0.5, label="interior threshold")
            ax2.set_ylabel("L2 diff")
            ax2.set_xlabel("time t")
            ax2.legend(loc="upper right")

            fig.tight_layout()
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_png, dpi=120)
            plt.close(fig)
        except Exception as e: # pragma: no cover
            results["plot_error"] = str(e)

    return results


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args and run the PACE-B gate sanity check."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--T", type=int, default=120)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=Path("sanity_pace_b.json"))
    parser.add_argument("--out-png", type=Path, default=Path("sanity_pace_b.png"))
    args = parser.parse_args(argv)

    res = run_sanity(
        T=args.T, K=args.K, seed=args.seed,
        out_json=args.out_json, out_png=args.out_png,
    )
    print(json.dumps(res, indent=2))
    ok = res["acceptance_boundary_gt_0p3"] and res["acceptance_interior_lt_0p05"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
