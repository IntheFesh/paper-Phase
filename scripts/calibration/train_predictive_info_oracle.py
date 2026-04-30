"""Train the predictive-information oracle for chunk-level InfoNCE calibration.

The ``PredictiveInfoEstimator`` (a bilinear InfoNCE critic from
``phase_centric.theory_utils``) provides an upper-bound estimate of
:math:`I(X;C)` between an action-chunk embedding and a phase-context
embedding. We use it as a calibration oracle: by training the estimator
on a held-out batch of (x, c) pairs we obtain a converged MI estimate
that the chunk-level InfoNCE loss in ``phase_centric.identifiability``
can be tuned against (temperature τ and weight λ).

Usage
-----
::

    python scripts/calibration/train_predictive_info_oracle.py --dry_run
    python scripts/calibration/train_predictive_info_oracle.py \\
        --x_dim 64 --c_dim 64 --steps 2000 \\
        --output outputs/calibration/predictive_info/oracle.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lerobot_policy_phaseqflow" / "src"))

from lerobot_policy_phaseqflow.phase_centric.theory_utils import PredictiveInfoEstimator


def _synthetic_batch(
    batch_size: int,
    x_dim: int,
    c_dim: int,
    correlation: float,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of (x, c) pairs with controllable mutual information.

    A shared latent ``z ~ N(0, I)`` of dimension ``min(x_dim, c_dim)`` is
    projected through random Gaussian matrices into ``x`` and ``c``. The
    ``correlation`` parameter scales the shared component vs. independent
    noise: ``correlation=0`` → independent, ``correlation=1`` → shares full
    latent.
    """
    shared_dim = min(x_dim, c_dim)
    z = torch.randn(batch_size, shared_dim, generator=rng)
    eps_x = torch.randn(batch_size, x_dim, generator=rng)
    eps_c = torch.randn(batch_size, c_dim, generator=rng)

    A = torch.randn(shared_dim, x_dim, generator=rng) / shared_dim**0.5
    B = torch.randn(shared_dim, c_dim, generator=rng) / shared_dim**0.5

    x = correlation * (z @ A) + (1.0 - correlation) * eps_x
    c = correlation * (z @ B) + (1.0 - correlation) * eps_c
    return x, c


def train(
    x_dim: int = 64,
    c_dim: int = 64,
    hidden_dim: int = 128,
    batch_size: int = 256,
    steps: int = 2000,
    lr: float = 1e-3,
    correlation: float = 0.7,
    seed: int = 42,
    log_every: int = 200,
) -> dict:
    """Train the oracle and return final MI estimate + training trace."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    estimator = PredictiveInfoEstimator(x_dim=x_dim, c_dim=c_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)

    history: list[dict] = []
    for step in range(steps):
        x, c = _synthetic_batch(batch_size, x_dim, c_dim, correlation, rng)
        out = estimator(x, c)
        loss = -out["mi_lower_bound"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            history.append({"step": step, "mi": float(out["mi_lower_bound"].detach())})
            print(f"  step {step:5d}/{steps}  MI={float(out['mi_lower_bound'].detach()):.4f}")

    # Final held-out evaluation across multiple correlation levels
    estimator.eval()
    held_out: dict[str, float] = {}
    with torch.no_grad():
        for rho in [0.0, 0.3, 0.7, 1.0]:
            mis = []
            for _ in range(20):
                xh, ch = _synthetic_batch(batch_size, x_dim, c_dim, rho, rng)
                mis.append(float(estimator(xh, ch)["mi_lower_bound"]))
            held_out[f"correlation={rho:.1f}"] = float(np.mean(mis))
    log_b = float(np.log(batch_size))

    return {
        "final_mi": history[-1]["mi"] if history else float("nan"),
        "log_batch_size": log_b,
        "held_out": held_out,
        "history": history,
        "config": {
            "x_dim": x_dim, "c_dim": c_dim, "hidden_dim": hidden_dim,
            "batch_size": batch_size, "steps": steps, "lr": lr,
            "correlation": correlation, "seed": seed,
        },
    }, estimator


def main(argv=None) -> int:
    """CLI entry point: train the InfoNCE oracle on synthetic (x,c) pairs and save checkpoint."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry_run", action="store_true", help="Use small-scale settings for a quick smoke run")
    p.add_argument("--x_dim", type=int, default=64)
    p.add_argument("--c_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--correlation", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("outputs/calibration/predictive_info/oracle.pt"))
    args = p.parse_args(argv)

    if args.dry_run:
        args.steps = 50
        args.batch_size = 32
        args.x_dim = 16
        args.c_dim = 16
        args.hidden_dim = 32

    print(f"[predictive_info_oracle] training (steps={args.steps}, batch={args.batch_size})")
    result, estimator = train(
        x_dim=args.x_dim, c_dim=args.c_dim, hidden_dim=args.hidden_dim,
        batch_size=args.batch_size, steps=args.steps, lr=args.lr,
        correlation=args.correlation, seed=args.seed,
        log_every=max(args.steps // 5, 1),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": estimator.state_dict(), "config": result["config"]}, args.output)

    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(
        {k: v for k, v in result.items() if k != "history"} | {"history_len": len(result["history"])},
        indent=2,
    ))

    print(f"[predictive_info_oracle] final MI={result['final_mi']:.4f} (log B={result['log_batch_size']:.4f})")
    print(f"[predictive_info_oracle] held-out MI by correlation:")
    for k, v in result["held_out"].items():
        print(f"  {k}  MI={v:.4f}")
    print(f"[predictive_info_oracle] checkpoint → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
