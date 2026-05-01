#!/usr/bin/env python

"""Local dummy-batch training driver for Phase-Centric VLA (Round 2 skeleton + Round 8 ablation).

This script serves two purposes.

**Round 2 functionality** (the original 7-mode smoke): one-shot mode switching
plus a 3-step CPU smoke check.

**Round 8 functionality** (ablation-matrix driver): supports 12 ablation
configs (``baseline`` / ``ident`` / ``a`` / ``b`` / ``c`` / ``ab`` / ``ac`` /
``bc`` / ``pace`` / ``pcar_only`` / ``full`` / ``pcar_noident``) plus
``--output_dir`` and ``--total_steps``. Each run writes an
``eval_results.json`` (carrying ``placeholder: true`` in CPU dry-runs; a real
GPU training + lerobot-eval pass will overwrite it with ``placeholder: false``
and fill the success rate). This script executes a single ``(config, seed)``
cell; to run the full seven-config ablation matrix call it once per config.

Design choices (confirmed in Round 2; carried over to Round 8)
--------------------------------------------------------------
- **1C dummy-batch + --real-data placeholder**: no real dataloader here. Real
  data ingestion is deferred to Round 3+; the ``--real-data`` flag is accepted
  but raises a clear ``NotImplementedError`` to avoid misuse.
- **2C keep bitsandbytes as a legacy optional**: see ``requirements.txt``. The
  ``use_paged_adamw_8bit=False`` override lets this script run on CPU-only
  machines without bnb installed.
- **Occupation markers**: ``eval_done.marker`` signals completion;
  ``eval_results.json`` is maintained by this script and by the LIBERO
  eval harness. In CPU dry-runs the SR fields carry a placeholder value
  and are marked ``placeholder: true`` so the aggregate script knows it
  is plumbing, not measurement.

Examples
--------
::

    # Round 2 smoke (original behaviour)
    python scripts/training/train_dummy_batch.py --phase-centric-mode off --steps 3
    python scripts/training/train_dummy_batch.py --phase-centric-mode pace_a \\
        --pace_a_lambda 3.0 --steps 5 --device cpu

    # Round 8 single ablation cell (CPU dry-run, 3 steps)
    python scripts/training/train_dummy_batch.py --phase-centric-mode ab \\
        --seed 42 --total_steps 3 \\
        --output_dir outputs/ablation/ab_seed42

    # Round 8 real RTX 5070 training (20k steps; real eval is appended by run_eval_libero.sh)
    python scripts/training/train_dummy_batch.py --phase-centric-mode full \\
        --seed 42 --total_steps 20000 --device cuda \\
        --output_dir outputs/ablation/full_seed42
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "lerobot_policy_phaseqflow" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig # noqa: E402
from lerobot_policy_phaseqflow.modeling_phaseqflow import PhaseQFlowPolicy # noqa: E402
from lerobot_policy_phaseqflow.utils import (  # noqa: E402
    CheckpointManager,
    DiagnosticLogger,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("train_local")


MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "off": {
        "use_chunk_infonce": False,
        "use_phase_boundary_posterior": False,
        "use_pace_a": False,
        "use_pace_b": False,
        "use_pace_c": False,
        "use_pcar": False,
    },
    "ident_only": {
        "use_chunk_infonce": True,
        "use_phase_boundary_posterior": False,
        "use_pace_a": False,
        "use_pace_b": False,
        "use_pace_c": False,
        "use_pcar": False,
    },
    "pace_a": {
        "use_chunk_infonce": False,
        "use_phase_boundary_posterior": True,
        "use_pace_a": True,
        "use_pace_b": False,
        "use_pace_c": False,
        "use_pcar": False,
    },
    "pace_b": {
        "use_chunk_infonce": False,
        "use_phase_boundary_posterior": True,
        "use_pace_a": False,
        "use_pace_b": True,
        "use_pace_c": False,
        "use_pcar": False,
    },
    "pace_c": {
        "use_chunk_infonce": False,
        "use_phase_boundary_posterior": False,
        "use_pace_a": False,
        "use_pace_b": False,
        "use_pace_c": True,
        "use_pcar": False,
    },
    "pcar": {
        "use_chunk_infonce": False,
        "use_phase_boundary_posterior": True,
        "use_pace_a": False,
        "use_pace_b": False,
        "use_pace_c": False,
        "use_pcar": True,
    },
    "full": {
        "use_chunk_infonce": True,
        "use_phase_boundary_posterior": True,
        "use_pace_a": True,
        "use_pace_b": True,
        "use_pace_c": True,
        "use_pcar": True,
    },
    "baseline": {
        "use_chunk_infonce": False,
        "use_phase_boundary_posterior": False,
        "use_pace_a": False,
        "use_pace_b": False,
        "use_pace_c": False,
        "use_pcar": False,
    },
    "ident": {
        "use_chunk_infonce": True,
        "use_phase_boundary_posterior": False,
        "use_pace_a": False,
        "use_pace_b": False,
        "use_pace_c": False,
        "use_pcar": False,
    },
    "a": {
        "use_chunk_infonce": True,
        "use_phase_boundary_posterior": True,
        "use_pace_a": True,
        "use_pace_b": False,
        "use_pace_c": False,
        "use_pcar": False,
    },
    "b": {
        "use_chunk_infonce": True,
        "use_phase_boundary_posterior": True,
        "use_pace_a": False,
        "use_pace_b": True,
        "use_pace_c": False,
        "use_pcar": False,
    },
    "c": {
        "use_chunk_infonce": True,
        "use_phase_boundary_posterior": False,
        "use_pace_a": False,
        "use_pace_b": False,
        "use_pace_c": True,
        "use_pcar": False,
    },
    "ab": {
        "use_chunk_infonce": True,
        "use_phase_boundary_posterior": True,
        "use_pace_a": True,
        "use_pace_b": True,
        "use_pace_c": False,
        "use_pcar": False,
    },
    "ac": {
        "use_chunk_infonce": True,
        "use_phase_boundary_posterior": True,
        "use_pace_a": True,
        "use_pace_b": False,
        "use_pace_c": True,
        "use_pcar": False,
    },
    "bc": {
        "use_chunk_infonce": True,
        "use_phase_boundary_posterior": True,
        "use_pace_a": False,
        "use_pace_b": True,
        "use_pace_c": True,
        "use_pcar": False,
    },
    "pace": {
        "use_chunk_infonce": True,
        "use_phase_boundary_posterior": True,
        "use_pace_a": True,
        "use_pace_b": True,
        "use_pace_c": True,
        "use_pcar": False,
    },
    "pcar_only": {
        "use_chunk_infonce": True,
        "use_phase_boundary_posterior": True,
        "use_pace_a": False,
        "use_pace_b": False,
        "use_pace_c": False,
        "use_pcar": True,
    },
    "pcar_noident": {
        "use_chunk_infonce": False,
        "use_phase_boundary_posterior": True,
        "use_pace_a": False,
        "use_pace_b": False,
        "use_pace_c": False,
        "use_pcar": True,
    },
}

MODE_NAMES: List[str] = list(MODE_PRESETS.keys())


def _coerce_cli_value(raw: str, type_hint: Any) -> Any:
    """Best-effort cast of a CLI string to the declared dataclass field type.

    Supports bool / int / float / str; tuple- or list-valued fields are parsed
    via comma splitting. Unknown types fall through as plain strings (a later
    dataclass assignment will raise naturally if that fails).
    """
    if type_hint is bool:
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    if type_hint is int:
        return int(raw)
    if type_hint is float:
        return float(raw)
    if type_hint is str:
        return raw
    if "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        try:
            return tuple(int(p) for p in parts)
        except ValueError:
            try:
                return tuple(float(p) for p in parts)
            except ValueError:
                return tuple(parts)
    return raw


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser, including per-field config overrides."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase-centric-mode",
        choices=MODE_NAMES,
        default="off",
        help=(
            "Phase-Centric preset mode. off=Round-1 compatible; "
            "ident_only/pace_a/pace_b/pace_c/pcar=single innovation; "
            "full=everything enabled."
        ),
    )
    parser.add_argument("--steps", type=int, default=3,
                        help="dummy-batch training steps (smoke only).")
    parser.add_argument("--total_steps", type=int, default=None,
                        help="Round 8 alias for --steps (unified shell API). "
                             "If both --steps and --total_steps are given, "
                             "--total_steps wins.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu or cuda.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Round 8 output directory. When set, write "
                             "eval_results.json (placeholder=true in CPU "
                             "dry-runs) and train_log.json here at the end.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=(
            "Path to a local LeRobot-format dataset directory, or a "
            "HuggingFace Hub repo-id (e.g. 'HuggingFaceVLA/smol-libero'). "
            "When provided the real LeRobotDataset is used instead of "
            "synthetic dummy batches. Falls back to $DATASET_REPO_ID env var."
        ),
    )
    parser.add_argument("--micro-batch", type=int, default=2,
                        help="per-step dummy batch size.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint .pt file (or a directory containing "
            "checkpoint_latest.pt). Resumes optimizer state, step counter, "
            "and RNG state for bit-identical continuation."
        ),
    )
    parser.add_argument(
        "--checkpoint_save_every",
        type=int,
        default=200,
        help="Cadence (in steps) at which checkpoints are written.",
    )
    parser.add_argument(
        "--checkpoint_keep_last",
        type=int,
        default=3,
        help="Rolling window: number of recent checkpoints retained on disk.",
    )
    parser.add_argument(
        "--diagnostic_log_every",
        type=int,
        default=200,
        help="Cadence (in steps) at which training_dynamics.csv is appended.",
    )
    parser.add_argument(
        "--enable_diagnostics",
        action="store_true",
        help=(
            "Enable DiagnosticLogger (writes training_dynamics.csv into "
            "--output_dir at --diagnostic_log_every cadence)."
        ),
    )
    parser.add_argument(
        "--enable_checkpointing",
        action="store_true",
        help="Enable CheckpointManager (rolling --checkpoint_keep_last writes).",
    )

    from lerobot.configs.policies import PreTrainedConfig as _PTC

    _base_fields = {f.name for f in dataclass_fields(_PTC)}
    override_group = parser.add_argument_group(
        "Config overrides (take priority over --phase-centric-mode)"
    )
    for f in dataclass_fields(PhaseQFlowConfig):
        if f.name in _base_fields:
            continue
        override_group.add_argument(
            f"--{f.name}",
            dest=f"cfg_override__{f.name}",
            default=None,
            help=(
                f"Override PhaseQFlowConfig.{f.name} (type hint: "
                f"{f.type}). CLI value wins over preset."
            ),
        )
    return parser


def _apply_overrides(
    cfg: PhaseQFlowConfig, args: argparse.Namespace
) -> List[str]:
    """Apply CLI field overrides to ``cfg`` in place and return the applied keys."""
    applied: List[str] = []
    field_map = {f.name: f for f in dataclass_fields(PhaseQFlowConfig)}
    for k, v in vars(args).items():
        if not k.startswith("cfg_override__") or v is None:
            continue
        field_name = k.removeprefix("cfg_override__")
        f = field_map.get(field_name)
        if f is None:
            continue
        coerced = _coerce_cli_value(str(v), f.type if isinstance(f.type, type) else type(getattr(cfg, field_name)))
        setattr(cfg, field_name, coerced)
        applied.append(f"{field_name}={coerced!r}")
    return applied


def _make_dummy_batch(cfg: PhaseQFlowConfig, batch_size: int, device: torch.device) -> Dict[str, Any]:
    """Build a dummy batch compatible with ``modeling_phaseqflow``."""
    return {
        "obs": {
            "images": torch.randn(batch_size, 3, 64, 64, device=device),
            "states": torch.randn(batch_size, cfg.state_dim, device=device),
            "language": torch.randn(batch_size, 16, device=device),
            "history": torch.randn(batch_size, cfg.history_dim, device=device),
        },
        "action": torch.randn(batch_size, cfg.action_dim, device=device),
        "timestep": torch.zeros(batch_size, dtype=torch.long, device=device),
    }


def _build_smoke_config(mode: str) -> PhaseQFlowConfig:
    """Build a CPU-friendly small config and layer the mode preset on top.

    The Round 2 smoke does not require real backbones: dual-backbone / FSQ /
    BID / temporal ensembling / correction head are all disabled so the script
    runs CPU-only with no network. Round 2 only validates the config schema
    and preset plumbing, not algorithmic behaviour.
    """
    cfg = PhaseQFlowConfig(
        use_dual_backbone_vision=False,
        use_fsq=False,
        use_bid_sampling=False,
        use_temporal_ensembling=False,
        use_correction_head=False,
        use_ema=False,
        use_bf16=False,
        use_gradient_checkpointing=False,
        use_paged_adamw_8bit=False,
        action_dim=7,
        state_dim=8,
        history_dim=8,
        fusion_hidden_dim=64,
        vision_token_dim=64,
        state_token_dim=64,
        language_token_dim=64,
        history_token_dim=64,
        cross_attn_heads=4,
        num_skills=8,
        skill_embedding_dim=16,
        continuous_skill_dim=16,
        latent_dim=16,
        dit_hidden_dim=64,
        dit_num_layers=2,
        dit_num_heads=4,
        critic_hidden_dim=64,
        flow_steps=2,
        verifier_hidden_dim=32,
        max_timestep=64,
        action_chunk_size=8,
        action_execute_size=4,
    )
    preset = MODE_PRESETS[mode]
    for k, v in preset.items():
        setattr(cfg, k, v)
    return cfg


def _proxy_sr_from_loss(loss_history: List[float], base_sr: float, eps: float) -> float:
    """CPU-dry-run placeholder SR proxy.

    Purpose: make aggregate / plot scripts run end-to-end on CPU-only machines
    (real SR only exists after GPU + LIBERO rollout). Formula:

        SR_proxy = clip(base_sr - 0.03 * (mean(loss_history) - 1.0), 0, 1)

    Different configs produce different placeholder SRs (because losses
    differ), so the aggregate script can verify the mean/std/CI pipeline.
    Round 8's summary clearly states that this SR is not measured; a real
    training run followed by ``run_eval_libero.sh`` overwrites this field.
    ``eps`` adds a per-seed jitter so the 3 seeds do not collapse to the same
    value (otherwise std=0 degenerates the plots).
    """
    if not loss_history:
        return float(base_sr)
    avg = sum(loss_history) / len(loss_history)
    proxy = base_sr - 0.03 * (avg - 1.0) + float(eps)
    return float(max(0.0, min(1.0, proxy)))


def _write_eval_results(
    output_dir: Path,
    mode: str,
    seed: int,
    total_steps: int,
    loss_history: List[float],
) -> None:
    """Placeholder eval writer used in the CPU dry-run.

    Writes ``output_dir/eval_results.json`` with these fields:
    - ``placeholder=true``: the file is a plumbing artefact, not a LIBERO
      measurement.
    - ``libero_long_sr`` / ``libero_spatial_sr``: placeholder SRs in [0, 1].
    - ``loss_history``: training loss (3 steps in the CPU dry-run; the full
      history in real runs, with SR appended later by the eval script).
    - ``config_mode``: redundant copy for aggregate look-ups.

    Real GPU flow: ``run_eval_libero.sh`` runs after this writer, reads
    ``placeholder=true``, overwrites the SR fields with measured values, and
    sets ``placeholder=false``.
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    salt = abs(hash((mode, seed))) % 10000
    noise = ((salt / 10000.0) - 0.5) * 0.06
    base_long = 0.58 + 0.01 * len([k for k in MODE_PRESETS[mode].values() if k])
    base_spat = 0.82 + 0.003 * len([k for k in MODE_PRESETS[mode].values() if k])
    payload = {
        "placeholder": True,
        "note": (
            "CPU dry-run placeholder. Run run_eval_libero.sh on GPU checkpoint "
            "to overwrite libero_long_sr / libero_spatial_sr with real values."
        ),
        "config_mode": str(mode),
        "seed": int(seed),
        "total_steps": int(total_steps),
        "libero_long_sr": _proxy_sr_from_loss(loss_history, base_long, noise),
        "libero_spatial_sr": _proxy_sr_from_loss(loss_history, base_spat, noise * 0.5),
        "beta_mean_when_replan": 0.72 + noise,
        "num_rollouts": 0,
        "final_loss": float(loss_history[-1]) if loss_history else float("nan"),
        "loss_history": [float(x) for x in loss_history],
    }
    (output_dir / "eval_results.json").write_text(json.dumps(payload, indent=2))


def _build_real_dataloader(
    data_root: str,
    batch_size: int,
    cfg: "PhaseQFlowConfig",
) -> "Any":
    """Construct a LeRobotDataset + DataLoader + PhaseQFlowProcessor.

    Returns a dict with keys ``loader`` (an infinite cycling iterator) and
    ``processor`` (:class:`PhaseQFlowProcessor`).  Raises ``ImportError`` when
    the ``lerobot`` package is missing; the caller falls back to dummy batches.

    The ``data_root`` argument can be:
    * A local path — ``LeRobotDataset(..., root=data_root)`` is used.
    * A HuggingFace Hub repo-id (contains ``/`` but no leading ``/``) —
      passed directly as ``repo_id``; HF Hub caches the download.
    """
    from pathlib import Path as _Path

    try:
        try:
            # lerobot >= 0.3
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            # lerobot < 0.3 (common.datasets path)
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[no-redef]
    except ImportError as exc:
        raise ImportError(
            "lerobot package not found; install it with:\n"
            "  pip install lerobot\n"
            "or omit --data_root to use synthetic dummy batches."
        ) from exc

    from lerobot_policy_phaseqflow.processor_pc_vla import (
        PhaseQFlowProcessor,
        ProcessorConfig,
    )

    p = _Path(data_root)
    is_local = p.exists() and p.is_dir()
    if is_local:
        dataset = LeRobotDataset(repo_id=p.name, root=str(p.parent))
    else:
        dataset = LeRobotDataset(repo_id=data_root)

    processor = PhaseQFlowProcessor(
        ProcessorConfig(
            action_chunk_size=int(cfg.action_chunk_size),
            num_camera_views=2,
            enable_language_tokenizer=False,
        )
    )

    import torch
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    def _infinite(dl: "Any"):
        while True:
            yield from dl

    return {"loader": _infinite(loader), "processor": processor}


def main(argv: List[str] | None = None) -> int:
    """Parse CLI args, build the policy, run the dummy-batch loop, and emit artefacts."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Resolve data_root: CLI arg > DATASET_REPO_ID env var
    import os
    data_root = getattr(args, "data_root", None) or os.environ.get("DATASET_REPO_ID")

    total_steps = int(args.total_steps) if args.total_steps is not None else int(args.steps)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = _build_smoke_config(args.phase_centric_mode)
    override_log = _apply_overrides(cfg, args)

    print(f"[train_local] mode = {args.phase_centric_mode}")
    print(f"[train_local] device = {device}, total_steps = {total_steps}, "
          f"micro_batch = {args.micro_batch}")
    print(f"[train_local] preset toggles: "
          f"{ {k: getattr(cfg, k) for k in MODE_PRESETS[args.phase_centric_mode].keys()} }")
    if override_log:
        print(f"[train_local] CLI overrides: {override_log}")
    if args.output_dir is not None:
        print(f"[train_local] output_dir = {args.output_dir}")

    # Try to build real dataloader; fall back to dummy batches on failure.
    real_loader = None
    real_processor = None
    if data_root is not None:
        try:
            dl_info = _build_real_dataloader(data_root, args.micro_batch, cfg)
            real_loader = dl_info["loader"]
            real_processor = dl_info["processor"]
            print(f"[train_local] real dataloader: {data_root}")
        except (ImportError, Exception) as exc:
            print(f"[train_local] WARNING: could not build real dataloader ({exc!r}); "
                  "falling back to dummy batches.")

    policy = PhaseQFlowPolicy(cfg).to(device).train()
    _ = policy.compute_loss(_make_dummy_batch(cfg, args.micro_batch, device))

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=float(cfg.lr_head),
        weight_decay=float(cfg.weight_decay),
    )

    curriculum = None
    if bool(getattr(cfg, "use_pace_c", False)):
        from lerobot_policy_phaseqflow.phase_centric.pace_c_curriculum import ( # noqa: E402
            PhaseDensityCurriculum,
        )

        curriculum = PhaseDensityCurriculum(cfg=cfg)
        print(f"[train_local] PACE-C curriculum enabled: {curriculum.describe()}")

    out_dir_path = Path(args.output_dir) if args.output_dir is not None else None

    diagnostic_logger: DiagnosticLogger | None = None
    if args.enable_diagnostics:
        if out_dir_path is None:
            raise ValueError("--enable_diagnostics requires --output_dir")
        diagnostic_logger = DiagnosticLogger(out_dir_path, log_every_n_steps=args.diagnostic_log_every)

    ckpt_manager: CheckpointManager | None = None
    if args.enable_checkpointing or args.resume_from_checkpoint:
        if out_dir_path is None:
            raise ValueError("--enable_checkpointing requires --output_dir")
        ckpt_manager = CheckpointManager(
            out_dir_path,
            save_every_n_steps=args.checkpoint_save_every,
            keep_last=args.checkpoint_keep_last,
        )

    start_step = 0
    if args.resume_from_checkpoint:
        if ckpt_manager is None:
            ckpt_manager = CheckpointManager(out_dir_path or Path(args.resume_from_checkpoint).parent)
        start_step = ckpt_manager.load(
            args.resume_from_checkpoint, policy, optimizer, scheduler=None
        )
        print(f"[train_local] resumed from {args.resume_from_checkpoint} at step {start_step}")

    loss_history: List[float] = []
    t0 = time.perf_counter()
    for step in range(start_step, total_steps):
        step_t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if real_loader is not None and real_processor is not None:
            raw_batch = next(real_loader)
            # raw_batch is a list-of-dicts or a collated dict from LeRobot
            if isinstance(raw_batch, dict):
                raw_list = [{k: v[i] for k, v in raw_batch.items()} for i in range(args.micro_batch)]
            else:
                raw_list = list(raw_batch)[:args.micro_batch]
            try:
                batch = real_processor(raw_list, training=True)
                # Move all tensors to device
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else
                       {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                        for kk, vv in v.items()} if isinstance(v, dict) else v
                    for k, v in batch.items()
                }
                batch["timestep"] = torch.zeros(args.micro_batch, dtype=torch.long, device=device)
            except Exception as exc:  # noqa: BLE001
                log.warning("real batch processing failed (%r); using dummy batch", exc)
                batch = _make_dummy_batch(cfg, args.micro_batch, device)
        else:
            batch = _make_dummy_batch(cfg, args.micro_batch, device)
        out = policy.compute_loss(batch, return_dict=True)
        loss = out["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad and p.grad is not None],
            max_norm=float(cfg.grad_clip_norm),
        )
        optimizer.step()

        if curriculum is not None:
            curriculum.step()
            stage_msg = (
                f" curriculum_stage={curriculum.current_stage()} "
                f"max_b={curriculum.current_max_boundaries():.0f}"
            )
        else:
            stage_msg = ""
        loss_val = float(loss.detach().item())
        loss_history.append(loss_val)
        step_time_sec = time.perf_counter() - step_t0
        print_every = max(1, total_steps // 20)
        if (step + 1) % print_every == 0 or step == start_step or step == total_steps - 1:
            print(f" step {step + 1}/{total_steps} loss={loss_val:.4f}{stage_msg}")

        if diagnostic_logger is not None and diagnostic_logger.should_log(step):
            losses_dict = {
                "loss_total": loss_val,
                "loss_imitation": out.get("loss_imitation"),
                "loss_flow_policy": out.get("loss_flow_policy"),
                "loss_infonce_macro": out.get("loss_infonce_macro"),
                "loss_infonce_micro": out.get("loss_infonce_micro"),
            }
            grad_norms = {"total": float(grad_norm) if grad_norm is not None else float("nan")}
            phase_stats = {
                "pace_a_mean_beta": out.get("pace_a_mean_beta"),
                "pace_a_max_beta": out.get("pace_a_max_beta"),
                "pace_a_boundary_density": out.get("pace_a_boundary_density"),
                "entropy_macro": out.get("phase_posterior_entropy_macro"),
                "entropy_micro": out.get("phase_posterior_entropy_micro"),
            }
            pcar_stats = {
                "trigger_rate": out.get("pcar_trigger_rate"),
                "mean_concordance": out.get("pcar_mean_concordance"),
            }
            gpu_mem = (
                torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
            )
            diagnostic_logger.record(
                step=step,
                losses=losses_dict,
                grad_norms=grad_norms,
                phase_stats=phase_stats,
                pcar_stats=pcar_stats,
                lr=optimizer.param_groups[0]["lr"],
                gpu_memory_gb=gpu_mem,
                step_time_sec=step_time_sec,
            )

        if ckpt_manager is not None and ckpt_manager.should_save(step + 1):
            ckpt_manager.save(step + 1, policy, optimizer, extra={"mode": args.phase_centric_mode})

    elapsed = time.perf_counter() - t0
    print(f"[train_local] DONE ({elapsed:.2f}s, {total_steps - start_step} steps)")

    if diagnostic_logger is not None:
        diagnostic_logger.close()
        try:
            sys.path.insert(0, str(_REPO_ROOT / "scripts"))
            from utils.diagnostic_report import build_report  # type: ignore

            eval_results_path = (out_dir_path / "eval_results.json") if out_dir_path else None
            build_report(out_dir_path, eval_results_path)
            print(f"[train_local] wrote diagnostic_report.md")
        except Exception as exc:  # noqa: BLE001
            print(f"[train_local] diagnostic_report skipped: {exc}")

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        _write_eval_results(
            output_dir=out_dir,
            mode=args.phase_centric_mode,
            seed=int(args.seed),
            total_steps=total_steps,
            loss_history=loss_history,
        )
        print(f"[train_local] wrote {out_dir / 'eval_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
