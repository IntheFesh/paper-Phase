#!/usr/bin/env python

"""Offline CPU latency benchmark for PhaseQFlowPolicy.

Runs ``select_action`` on synthetic LIBERO-shaped observations without
requiring a pretrained checkpoint or access to the Hugging Face Hub.
Intended as a sanity/perf probe in environments where
``benchmark_latency.py`` (which needs a checkpoint) cannot be used.

Example::

    python scripts/evaluation/benchmark_latency_offline.py \\
        --configs baseline ident_only pace_a pace_b pace_c pcar --iters 20
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig
from lerobot_policy_phaseqflow.modeling_phaseqflow import PhaseQFlowPolicy


PRESETS: Dict[str, Dict[str, bool]] = {
    "baseline": {},
    "ident_only": {"use_chunk_infonce": True},
    "pace_a": {"use_pace_a": True},
    "pace_b": {"use_pace_b": True, "use_phase_boundary_posterior": True},
    "pace_c": {"use_pace_c": True},
    "pcar": {"use_pcar": True},
}


def _make_dummy_obs(cfg: PhaseQFlowConfig) -> Dict[str, torch.Tensor]:
    """Build a synthetic LIBERO-shaped observation dict sized by ``cfg``."""
    H = int(cfg.vision_image_size)
    return {
        "images": torch.zeros(int(cfg.num_camera_views), 3, H, H),
        "states": torch.zeros(int(cfg.state_dim)),
        "history": torch.zeros(int(cfg.history_dim)),
        "masks": torch.ones(1),
    }


def bench_config(label: str, iters: int, warmup: int, threads: int) -> Dict[str, float]:
    """Benchmark ``select_action`` for the preset ``label`` and return latency statistics."""
    overrides = PRESETS[label]
    cfg = PhaseQFlowConfig(
        use_bid_sampling=False,
        use_temporal_ensembling=False,
        **overrides,
    )
    pol = PhaseQFlowPolicy(cfg).eval()
    obs = _make_dummy_obs(cfg)
    lat: List[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            pol.select_action(obs)
        for _ in range(iters):
            t0 = time.perf_counter()
            pol.select_action(obs)
            lat.append((time.perf_counter() - t0) * 1000.0)
    lat.sort()
    p95_idx = max(0, int(0.95 * iters) - 1)
    return {
        "label": label,
        "n_iters": iters,
        "threads": threads,
        "mean_ms": statistics.mean(lat),
        "p50_ms": statistics.median(lat),
        "p95_ms": lat[p95_idx],
        "min_ms": lat[0],
        "max_ms": lat[-1],
        "std_ms": statistics.pstdev(lat),
        "params_M_total": sum(p.numel() for p in pol.parameters()) / 1e6,
        "params_M_trainable": sum(p.numel() for p in pol.parameters() if p.requires_grad) / 1e6,
    }


def main() -> int:
    """Parse CLI arguments, run each preset's benchmark, and print or dump results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--configs",
        nargs="*",
        default=["baseline", "ident_only", "pace_a", "pace_b", "pace_c", "pcar"],
        choices=list(PRESETS),
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    rows = [bench_config(c, args.iters, args.warmup, args.threads) for c in args.configs]

    print(f"{'config':<24s} {'params(M)':>10s} {'mean(ms)':>10s} {'p50':>8s} {'p95':>8s}")
    for r in rows:
        print(
            f"{r['label']:<24s} "
            f"{r['params_M_total']:>10.1f} "
            f"{r['mean_ms']:>10.1f} "
            f"{r['p50_ms']:>8.1f} "
            f"{r['p95_ms']:>8.1f}"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
