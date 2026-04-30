"""Diagnostic: inference cost comparison across ablation methods.

Measures per-method:
  - Parameter count (M)
  - Number of Function Evaluations (NFE) per action prediction
  - Mean latency (ms) over N forward passes
  - Throughput (Hz)

Methods compared:
  bc_chunked, cliff_via_beta, cliff_via_var, cliff_via_curvature,
  cliff_concordance, cliff_concordance_with_boundary_reweight

Usage
-----
::

    python scripts/diagnostics/measure_inference_cost.py --dry_run
    python scripts/diagnostics/measure_inference_cost.py \\
        --checkpoint checkpoints/phaseqflow --n_trials 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic data for dry-run
# ---------------------------------------------------------------------------

_SYNTHETIC_COSTS = {
    "bc_chunked": {
        "params_M": 312.4,
        "NFE": 4,
        "latency_ms": 12.3,
    },
    "cliff_via_beta_only": {
        "params_M": 312.5,
        "NFE": 4,
        "latency_ms": 13.1,
    },
    "cliff_via_var_only": {
        "params_M": 312.5,
        "NFE": 36,   # 8 samples × 4 NFE per shortcut step
        "latency_ms": 89.4,
    },
    "cliff_via_curvature_only": {
        "params_M": 312.5,
        "NFE": 8,    # 2 velocity evaluations × 4 NFE
        "latency_ms": 22.7,
    },
    "cliff_concordance": {
        "params_M": 312.6,
        "NFE": 44,   # beta(4) + var(36) + curvature(8) - overlap(4)
        "latency_ms": 102.5,
    },
    "cliff_concordance_with_boundary_reweight": {
        "params_M": 312.6,
        "NFE": 44,
        "latency_ms": 103.1,
    },
}


def _synthetic_results(rng: np.random.Generator) -> List[Dict]:
    rows = []
    for method, costs in _SYNTHETIC_COSTS.items():
        noise_ms = rng.normal(0, costs["latency_ms"] * 0.03)
        latency = max(1.0, costs["latency_ms"] + noise_ms)
        hz = 1000.0 / latency
        rows.append({
            "method": method,
            "params_M": costs["params_M"],
            "NFE": costs["NFE"],
            "latency_ms": round(latency, 2),
            "Hz": round(hz, 1),
        })
    return rows


# ---------------------------------------------------------------------------
# Real measurement
# ---------------------------------------------------------------------------

def _count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def _measure_latency(
    policy,
    obs_batch: dict,
    n_trials: int = 100,
    device: str = "cuda",
) -> float:
    """Return mean latency in ms over n_trials forward passes."""
    import torch

    # Warm-up
    for _ in range(5):
        with torch.no_grad():
            policy.select_action(obs_batch)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        with torch.no_grad():
            policy.select_action(obs_batch)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times))


def _measure_all(
    checkpoint_path: str,
    ablation_configs: Dict[str, str],
    n_trials: int,
    device: str,
) -> List[Dict]:
    import torch

    sys.path.insert(
        0,
        str(Path(__file__).resolve().parent.parent.parent.parent
            / "lerobot_policy_phaseqflow" / "src"),
    )
    from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig
    from lerobot_policy_phaseqflow.modeling_phaseqflow import PhaseQFlowPolicy

    # Dummy observation batch
    def _dummy_obs(cfg: PhaseQFlowConfig):
        return {
            "observation.state": torch.zeros(1, cfg.state_dim, device=device),
            "observation.images.front": torch.zeros(
                1, 3, cfg.vision_image_size, cfg.vision_image_size, device=device
            ),
        }

    rows = []
    for method, cfg_path in ablation_configs.items():
        cfg = PhaseQFlowConfig.from_pretrained(checkpoint_path)
        # Override with ablation config
        import yaml
        with open(cfg_path) as f:
            overrides = yaml.safe_load(f).get("phaseqflow", {})
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        policy = PhaseQFlowPolicy.from_pretrained(checkpoint_path, config=cfg).to(device).eval()
        params = _count_params(policy)
        obs = _dummy_obs(cfg)
        latency = _measure_latency(policy, obs, n_trials=n_trials, device=device)
        hz = 1000.0 / latency

        # NFE: shortcut flow uses flow_steps NFE per sample call
        # variance estimator: n_samples × flow_steps NFE
        nfe_base = int(cfg.flow_steps)
        n_samples = 8 if cfg.pcar_input_signal == "variance" or cfg.pcar_input_signal == "concordance" else 0
        nfe = nfe_base * (1 + n_samples)

        rows.append({
            "method": method,
            "params_M": round(params, 1),
            "NFE": nfe,
            "latency_ms": round(latency, 2),
            "Hz": round(hz, 1),
        })
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _save(rows: List[Dict], output_path: Path, dry_run: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "params_M", "NFE", "latency_ms", "Hz"])
        w.writeheader()
        w.writerows(rows)

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps({"rows": rows, "dry_run": dry_run,
                                      "generated": datetime.now().isoformat()}, indent=2))
    print("[measure_inference_cost] results:")
    for r in rows:
        print(f"  {r['method']:45s}  {r['params_M']:6.1f}M  {r['NFE']:3d} NFE  "
              f"{r['latency_ms']:7.1f} ms  {r['Hz']:5.0f} Hz")
    print(f"  → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--n_trials", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=Path, default=Path("paper_figures/diagnostics/inference_cost.csv"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point: report params / NFE / latency for each ablation config."""
    args = _parse_args(argv)
    rng = np.random.default_rng(42)
    if args.dry_run or args.checkpoint is None:
        rows = _synthetic_results(rng)
    else:
        ablation_root = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "ablation" / "v2"
        ablation_configs = {
            "bc_chunked": str(ablation_root / "01_bc_chunked.yaml"),
            "cliff_via_beta_only": str(ablation_root / "02_cliff_via_beta_only.yaml"),
            "cliff_via_var_only": str(ablation_root / "03_cliff_via_var_only.yaml"),
            "cliff_via_curvature_only": str(ablation_root / "04_cliff_via_curvature_only.yaml"),
            "cliff_concordance": str(ablation_root / "05_cliff_concordance.yaml"),
            "cliff_concordance_with_boundary_reweight": str(ablation_root / "07_cliff_concordance_with_boundary_reweight.yaml"),
        }
        rows = _measure_all(args.checkpoint, ablation_configs, args.n_trials, args.device)
    _save(rows, args.output, dry_run=(args.dry_run or args.checkpoint is None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
