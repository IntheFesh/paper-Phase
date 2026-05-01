#!/usr/bin/env python
"""Auto-tune the largest micro-batch that fits on the current GPU.

Strategy
--------
Walk descending candidates (default ``[128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4]``).
For each candidate, build the **full PACE v2 policy** (same modes as Stage 2 +
ablation 07 — the heaviest config), run **3 forward+backward+optimizer steps**,
and record peak CUDA memory.  The first candidate that runs without
``OutOfMemoryError`` AND uses less than ``--safety_factor`` (default 0.90) of
total VRAM is selected.

Why 3 steps? PyTorch caches optimizer state and the cuDNN workspace lazily, so
a single step underestimates peak memory by ~10–15%.

Output
------
Writes a sourceable bash file::

    BATCH_SIZE=<recommended>
    GRAD_ACCUM=<so that BATCH_SIZE * GRAD_ACCUM stays at the target effective batch>
    AUTOBATCH_PEAK_GB=<peak VRAM at the chosen batch>
    AUTOBATCH_TOTAL_GB=<total VRAM>

Default location: ``$PACE_OUT/_launch_logs/autobatch.env``.

Usage
-----
::

    python scripts/tune_batch_size.py
    python scripts/tune_batch_size.py --candidates 64 32 16 8 --target_effective_batch 1024
    python scripts/tune_batch_size.py --output /tmp/autobatch.env
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "lerobot_policy_phaseqflow" / "src"))


def _build_full_config():
    """Mirror the heaviest cloud-sweep config (Stage 2 / ablation 07)."""
    from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig

    cfg = PhaseQFlowConfig(
        # Vision / fusion
        use_dual_backbone_vision=True,
        use_fsq=True,
        use_bid_sampling=True,
        use_temporal_ensembling=False,
        use_correction_head=False,
        use_ema=False,
        use_bf16=True,
        use_gradient_checkpointing=True,
        use_paged_adamw_8bit=False,
        # Phase-centric — ablation 07 / Stage 2 enables all of these
        use_chunk_infonce=True,
        use_phase_boundary_posterior=True,
        use_pace_a=True,
        use_pace_b=True,
        use_pace_c=True,
        use_pcar=True,
        # Action shape
        action_dim=7,
        state_dim=8,
        history_dim=8,
        action_chunk_size=16,
    )
    return cfg


def _make_dummy_batch(cfg, batch_size: int, device):
    import torch
    Ta = int(getattr(cfg, "action_chunk_size", 16))
    return {
        "obs": {
            "images":   torch.randn(batch_size, 2, 3, 224, 224, device=device),
            "states":   torch.randn(batch_size, cfg.state_dim, device=device),
            "language": torch.randn(batch_size, 16, device=device),
            "history":  torch.randn(batch_size, cfg.history_dim, device=device),
        },
        "action":   torch.randn(batch_size, Ta, cfg.action_dim, device=device),
        "timestep": torch.zeros(batch_size, dtype=torch.long, device=device),
    }


def _try_batch(batch_size: int, n_steps: int = 3) -> Tuple[bool, float, str]:
    """Try ``batch_size`` micro-batch for ``n_steps`` steps.

    Returns ``(ok, peak_gb, error_msg)``.
    """
    import torch
    from lerobot_policy_phaseqflow.modeling_phaseqflow import PhaseQFlowPolicy

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = _build_full_config()
    policy = None
    opt = None
    try:
        policy = PhaseQFlowPolicy(cfg).to("cuda").train()
        opt = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad],
            lr=1e-4,
        )
        for _ in range(n_steps):
            batch = _make_dummy_batch(cfg, batch_size, "cuda")
            opt.zero_grad(set_to_none=True)
            loss = policy.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.grad is not None], max_norm=1.0
            )
            opt.step()
            torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        return True, peak, ""
    except torch.cuda.OutOfMemoryError as e:
        return False, 0.0, f"OOM: {e}"
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "CUDA" in msg and "memory" in msg.lower():
            return False, 0.0, f"OOM (RuntimeError): {msg.splitlines()[0]}"
        return False, 0.0, f"RuntimeError: {msg.splitlines()[0]}"
    finally:
        del policy, opt
        torch.cuda.empty_cache()


def _gpu_info() -> Tuple[float, str]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — cannot autotune batch size")
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    total_gb = torch.cuda.get_device_properties(idx).total_memory / 1e9
    return total_gb, name


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-tune BATCH_SIZE for the current GPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=[128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4],
        help="Descending list of micro-batch sizes to probe.",
    )
    p.add_argument(
        "--n_steps", type=int, default=3,
        help="Number of full train steps per probe (>=2 to capture optimizer state).",
    )
    p.add_argument(
        "--safety_factor", type=float, default=0.90,
        help="Reject any batch whose peak VRAM exceeds this fraction of total VRAM.",
    )
    p.add_argument(
        "--target_effective_batch", type=int, default=1024,
        help="Effective batch (= micro_batch × grad_accum) target. Used to "
             "derive GRAD_ACCUM after the search.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Where to write the sourceable env file. "
             "Defaults to $PACE_OUT/_launch_logs/autobatch.env.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("[autobatch] torch not installed", file=sys.stderr)
        return 2

    total_gb, name = _gpu_info()
    print(f"[autobatch] GPU = {name}  total VRAM = {total_gb:.1f} GB")
    print(f"[autobatch] candidates = {args.candidates}  safety = {args.safety_factor:.2f}")
    print(f"[autobatch] running {args.n_steps} train steps per probe ...")

    chosen: Optional[int] = None
    chosen_peak: float = 0.0
    history = []  # type: List[Tuple[int, bool, float, str]]

    for bs in sorted(set(args.candidates), reverse=True):
        t0 = time.time()
        ok, peak, msg = _try_batch(bs, n_steps=args.n_steps)
        dt = time.time() - t0
        within_safety = ok and (peak <= args.safety_factor * total_gb)
        history.append((bs, ok, peak, msg))
        print(
            f"  batch={bs:>3}  "
            f"{'OK ' if ok else 'FAIL'}  "
            f"peak={peak:5.1f} GB  "
            f"({peak / total_gb * 100:4.1f}% of {total_gb:.1f} GB)  "
            f"{dt:5.1f}s  "
            f"{'' if ok else msg}"
        )
        if within_safety and chosen is None:
            chosen = bs
            chosen_peak = peak
            break  # take the first (largest) safe candidate

    if chosen is None:
        print(
            "\n[autobatch] FAILED — no candidate fit within safety bound. "
            "Lower --safety_factor (currently {:.2f}) or extend the candidate "
            "list with smaller batches.".format(args.safety_factor),
            file=sys.stderr,
        )
        return 1

    grad_accum = max(1, args.target_effective_batch // chosen)
    effective = chosen * grad_accum

    print(f"\n[autobatch] CHOSEN  BATCH_SIZE={chosen}  "
          f"GRAD_ACCUM={grad_accum}  effective={effective}")
    print(f"[autobatch] peak {chosen_peak:.1f}/{total_gb:.1f} GB "
          f"({chosen_peak / total_gb * 100:.1f}%)")

    # Resolve output path
    if args.output is None:
        run_logs = Path(os.environ.get("PACE_OUT", "outputs")) / "_launch_logs"
        run_logs.mkdir(parents=True, exist_ok=True)
        out_path = run_logs / "autobatch.env"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_path = args.output

    out_path.write_text(
        "# Auto-generated by scripts/tune_batch_size.py\n"
        f"# GPU: {name} ({total_gb:.1f} GB)\n"
        f"# probed: {[h[0] for h in history]}\n"
        f"export BATCH_SIZE={chosen}\n"
        f"export GRAD_ACCUM={grad_accum}\n"
        f"export AUTOBATCH_PEAK_GB={chosen_peak:.2f}\n"
        f"export AUTOBATCH_TOTAL_GB={total_gb:.2f}\n"
        f"export AUTOBATCH_GPU={name!r}\n"
    )
    print(f"[autobatch] wrote {out_path}")
    print(f"[autobatch] sourced automatically by `bash scripts/run_autodl_pipeline.sh train`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
