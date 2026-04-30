"""Diagnostic: boundary vs interior flow-matching loss ratio.

Computes the ratio E_boundary / E_interior where:
  E_boundary = mean FM loss at timesteps within ±k of a phase boundary
  E_interior = mean FM loss at non-boundary timesteps

A ratio > 1 confirms the Predictability Cliff hypothesis: the flow head
is harder to supervise near phase transitions.

Usage
-----
::

    python scripts/diagnostics/measure_boundary_error.py \\
        --checkpoint checkpoints/phaseqflow \\
        --output paper_figures/diagnostics/boundary_error.csv

    # Dry run (synthetic data):
    python scripts/diagnostics/measure_boundary_error.py --dry_run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic dry-run helpers
# ---------------------------------------------------------------------------

def _synthetic_ratio(
    n_demos: int = 20,
    T: int = 200,
    n_boundaries: int = 3,
    boundary_half_width: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Return (E_boundary, E_interior, ratio) from synthetic demo data."""
    if rng is None:
        rng = np.random.default_rng(42)

    boundary_losses, interior_losses = [], []
    spacing = T // (n_boundaries + 1)
    for _ in range(n_demos):
        # Boundary mask
        mask = np.zeros(T, dtype=bool)
        for i in range(1, n_boundaries + 1):
            t = int(i * spacing + rng.integers(-spacing // 5, spacing // 5 + 1))
            t = int(np.clip(t, 0, T - 1))
            for dt in range(-boundary_half_width, boundary_half_width + 1):
                if 0 <= t + dt < T:
                    mask[t + dt] = True

        # Synthetic FM losses: higher at boundaries
        base_loss = rng.exponential(0.05, T)
        boundary_boost = np.where(mask, rng.exponential(0.15, T), 0.0)
        losses = base_loss + boundary_boost

        boundary_losses.extend(losses[mask].tolist())
        interior_losses.extend(losses[~mask].tolist())

    E_b = float(np.mean(boundary_losses)) if boundary_losses else float("nan")
    E_i = float(np.mean(interior_losses)) if interior_losses else float("nan")
    ratio = E_b / E_i if E_i > 1e-10 else float("nan")
    return E_b, E_i, ratio


# ---------------------------------------------------------------------------
# Real computation
# ---------------------------------------------------------------------------

def _compute_boundary_error(
    checkpoint_path: str,
    dataset_path: str,
    n_demos: int = 50,
    boundary_half_width: int = 5,
    device: str = "cuda",
) -> Tuple[float, float, float]:
    """Compute E_boundary / E_interior from a real dataset + checkpoint."""
    import torch

    sys.path.insert(
        0,
        str(Path(__file__).resolve().parent.parent.parent.parent
            / "lerobot_policy_phaseqflow" / "src"),
    )
    from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig
    from lerobot_policy_phaseqflow.modeling_phaseqflow import PhaseQFlowPolicy

    cfg = PhaseQFlowConfig.from_pretrained(checkpoint_path)
    policy = PhaseQFlowPolicy.from_pretrained(checkpoint_path, config=cfg).to(device).eval()

    from .h1_loss import compute_per_timestep_fm_loss
    from .phase_proxies import gripper_boundary_mask

    boundary_losses: List[float] = []
    interior_losses: List[float] = []

    # Load dataset (assumes LeRobot HDF5 format)
    import h5py
    with h5py.File(dataset_path, "r") as f:
        demo_keys = list(f.keys())[:n_demos]
        for key in demo_keys:
            demo = f[key]
            actions = np.array(demo["action"])        # (T, Da)
            states = np.array(demo["obs/state"])      # (T, Ds)
            gripper = actions[:, -1]                  # last action dim = gripper

            mask = gripper_boundary_mask(gripper, half_width=boundary_half_width)
            T = len(actions)

            # Pack into batch (T, *)
            actions_t = torch.tensor(actions, dtype=torch.float32).to(device)
            states_t = torch.tensor(states, dtype=torch.float32).to(device)

            with torch.no_grad():
                fm_losses = compute_per_timestep_fm_loss(
                    policy=policy, actions=actions_t, states=states_t
                )

            fm_np = fm_losses.cpu().numpy()
            boundary_losses.extend(fm_np[mask].tolist())
            interior_losses.extend(fm_np[~mask].tolist())

    E_b = float(np.mean(boundary_losses)) if boundary_losses else float("nan")
    E_i = float(np.mean(interior_losses)) if interior_losses else float("nan")
    ratio = E_b / E_i if E_i > 1e-10 else float("nan")
    return E_b, E_i, ratio


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _save(
    E_b: float,
    E_i: float,
    ratio: float,
    output_path: Path,
    dry_run: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"metric": "E_boundary", "value": round(E_b, 6)},
        {"metric": "E_interior", "value": round(E_i, 6)},
        {"metric": "ratio_boundary_over_interior", "value": round(ratio, 4)},
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerows(rows)

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "E_boundary": E_b,
        "E_interior": E_i,
        "ratio": ratio,
        "dry_run": dry_run,
        "generated": datetime.now().isoformat(),
    }, indent=2))
    print(f"[measure_boundary_error] E_b={E_b:.4f}  E_i={E_i:.4f}  ratio={ratio:.3f}")
    print(f"  → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--n_demos", type=int, default=50)
    p.add_argument("--boundary_half_width", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=Path, default=Path("paper_figures/diagnostics/boundary_error.csv"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.dry_run or args.checkpoint is None:
        rng = np.random.default_rng(42)
        E_b, E_i, ratio = _synthetic_ratio(rng=rng)
        dry_run = True
    else:
        if args.dataset is None:
            print("[measure_boundary_error] ERROR: --dataset required for real run")
            return 1
        E_b, E_i, ratio = _compute_boundary_error(
            args.checkpoint, args.dataset, args.n_demos,
            args.boundary_half_width, args.device,
        )
        dry_run = False
    _save(E_b, E_i, ratio, args.output, dry_run=dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
