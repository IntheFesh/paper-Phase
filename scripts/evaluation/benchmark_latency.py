#!/usr/bin/env python

"""
Benchmark inference latency for PhaseQFlow policies.

The script loads a pretrained policy checkpoint and repeatedly calls inference
on a multimodal dummy observation shaped like PhaseQFlow processor output.
"""

import argparse
import time
from typing import Any, Optional


def try_import_lerobot_policy() -> Optional[Any]:
    """Try importing LeRobot's pretrained policy base class."""
    try:
        from lerobot.policies import PreTrainedPolicy # type: ignore

        return PreTrainedPolicy
    except Exception:
        return None


def load_policy(path: str) -> Optional[Any]:
    """Load a policy from a pretrained checkpoint path."""
    policy_base = try_import_lerobot_policy()
    if policy_base is None:
        print("LeRobot is not installed or unavailable. Cannot load policy.")
        return None
    try:
        return policy_base.from_pretrained(path)
    except Exception as exc:
        print(f"Failed to load policy from '{path}': {exc}")
        return None


def _infer_call(policy: Any, dummy_obs: Any) -> Any:
    """Try common inference entrypoints in a robust order."""
    if hasattr(policy, "predict"):
        return policy.predict(dummy_obs) # type: ignore[attr-defined]
    return policy(dummy_obs)


def benchmark(policy: Any, n_iters: int) -> int:
    """Run latency benchmarking and return process-like status code."""
    import numpy as np

    if n_iters <= 0:
        print("n_iters must be > 0")
        return 1

    dummy_obs = {
        "obs.images": np.zeros((1, 3, 84, 84), dtype=np.float32),
        "obs.states": np.zeros((1, 16), dtype=np.float32),
        "obs.language": np.zeros((1, 8), dtype=np.float32),
        "obs.history": np.zeros((1, 16), dtype=np.float32),
        "obs.masks": np.ones((1, 1), dtype=np.float32),
    }

    try:
        if hasattr(policy, "reset"):
            policy.reset(1)
    except Exception:
        pass

    try:
        _infer_call(policy, dummy_obs)
    except Exception as exc:
        print(f"Warm-up inference failed: {exc}")
        return 1

    times_ms = []
    for _ in range(n_iters):
        start = time.perf_counter()
        _infer_call(policy, dummy_obs)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    sorted_times = sorted(times_ms)
    avg_ms = sum(times_ms) / len(times_ms)
    p50 = sorted_times[len(sorted_times) // 2]
    p95 = sorted_times[max(0, int(len(sorted_times) * 0.95) - 1)]

    print(f"Average inference latency: {avg_ms:.3f} ms")
    print(f"P50 latency: {p50:.3f} ms")
    print(f"P95 latency: {p95:.3f} ms")
    print(f"Iterations: {n_iters}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Benchmark PhaseQFlow policy inference latency")

    parser.add_argument("--policy-path", dest="policy_path", type=str, default=None, help="Path to pretrained policy directory")
    parser.add_argument("--n-iters", dest="n_iters", type=int, default=100, help="Number of benchmark iterations")

    parser.add_argument("--policy.path", dest="policy_path", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--n_iters", dest="n_iters", type=int, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Parse CLI arguments, load the policy, and run the benchmark."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.policy_path:
        parser.error("Please provide --policy-path (or legacy --policy.path).")

    policy = load_policy(args.policy_path)
    if policy is None:
        return 1
    return benchmark(policy, args.n_iters)


if __name__ == "__main__":
    raise SystemExit(main())
