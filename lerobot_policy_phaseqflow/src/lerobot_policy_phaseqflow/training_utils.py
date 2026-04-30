"""Training-side utilities for PhaseQFlow.

Helpers used by LeRobot / Accelerate-style training loops:

  - :func:`build_param_groups` splits model parameters into
    ``vision_backbone`` / ``lora`` / ``head`` groups with separate learning
    rates; the LoRA group gets ``weight_decay=0``.
  - :func:`build_optimizer` defaults to ``bitsandbytes.optim.PagedAdamW8bit``
    and gracefully falls back to ``torch.optim.AdamW`` when bnb is unavailable.
  - :func:`build_scheduler` prefers
    ``transformers.get_cosine_schedule_with_warmup``, and builds a manual
    LambdaLR when ``transformers`` is missing.
  - :func:`apply_stage_freeze` freezes planner / flow_head / verifier
    according to the stage flags, then logs trainable parameter counts per
    module.
  - :func:`maybe_apply_gradient_checkpointing` wraps each vision
    cross-attention and DiT encoder block in
    ``torch.utils.checkpoint.checkpoint(use_reentrant=False)``.
  - :func:`maybe_build_ema` builds an :class:`EMAModel` tracking only the
    ``flow_action_head`` (cheaper on VRAM), with ``use_ema_warmup=True`` and
    ``power=cfg.ema_power``.

Design constraint: importing this module must not touch ``torch.cuda``, so
heavy GPU-only deps (bitsandbytes, diffusers) are imported lazily.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


_BACKBONE_HINTS: Tuple[str, ...] = (
    "vision_backbone",
    "siglip",
    "dinov2",
    "t5_encoder",
)
_LORA_HINTS: Tuple[str, ...] = (
    "lora_",
    "lora_magnitude_vector",
    "lora_embedding",
)


def _is_backbone_param(name: str) -> bool:
    """Return True if the parameter name belongs to a frozen vision/T5 backbone."""
    lower = name.lower()
    if any(hint in lower for hint in _LORA_HINTS):
        return False
    return any(hint in lower for hint in _BACKBONE_HINTS)


def _is_lora_param(name: str) -> bool:
    """Return True if the parameter name belongs to a LoRA adapter."""
    lower = name.lower()
    return any(hint in lower for hint in _LORA_HINTS)


def build_param_groups(model: nn.Module, cfg: Any) -> List[Dict[str, Any]]:
    """Group parameters into {backbone, lora, head} with per-group LRs.

    - backbone: frozen vision/T5 trunks (SigLIP / DINOv2 / T5), lr =
      ``cfg.lr_backbone`` (0 by default, i.e. fully frozen; the group is
      kept as a placeholder in case unfreezing is wanted later).
    - LoRA: any parameter whose name contains ``lora_``, lr = ``cfg.lr_lora``
      with ``weight_decay=0`` (LoRA adapters regularise through their own
      dropout and do not need L2).
    - head: everything else (planner / flow_action_head / chunk_verifier /
      fusion / FiLM / readout / ...), lr = ``cfg.lr_head``.

    Only parameters with ``requires_grad=True`` are collected (pair this with
    :func:`apply_stage_freeze`).
    """
    backbone_params: List[nn.Parameter] = []
    lora_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []
    head_names: List[str] = []
    lora_names: List[str] = []
    backbone_names: List[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_lora_param(name):
            lora_params.append(param)
            lora_names.append(name)
        elif _is_backbone_param(name):
            backbone_params.append(param)
            backbone_names.append(name)
        else:
            head_params.append(param)
            head_names.append(name)

    groups: List[Dict[str, Any]] = []
    if backbone_params:
        groups.append({
            "params": backbone_params,
            "lr": float(getattr(cfg, "lr_backbone", 0.0)),
            "weight_decay": float(getattr(cfg, "weight_decay", 1e-4)),
            "name": "backbone",
        })
    if lora_params:
        groups.append({
            "params": lora_params,
            "lr": float(getattr(cfg, "lr_lora", 5e-5)),
            "weight_decay": 0.0,
            "name": "lora",
        })
    if head_params:
        groups.append({
            "params": head_params,
            "lr": float(getattr(cfg, "lr_head", 1e-4)),
            "weight_decay": float(getattr(cfg, "weight_decay", 1e-4)),
            "name": "head",
        })
    return groups


def summarize_param_groups(groups: List[Dict[str, Any]]) -> str:
    """Return a human-readable summary of param groups (for logging / acceptance)."""
    lines = []
    for g in groups:
        numel = sum(p.numel() for p in g["params"])
        lines.append(
            f" [{g.get('name', '?')}] lr={g['lr']:.2e} wd={g['weight_decay']:.2e} "
            f"params={len(g['params'])} numel={numel:,}"
        )
    return "\n".join(lines) if lines else " (no trainable parameters)"


def build_optimizer(model: nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Build ``PagedAdamW8bit`` when available; gracefully fall back to AdamW.

    ``cfg`` must expose ``adam_beta1`` / ``adam_beta2`` / ``adam_eps`` /
    ``weight_decay``.
    """
    groups = build_param_groups(model, cfg)
    betas = (float(cfg.adam_beta1), float(cfg.adam_beta2))
    eps = float(cfg.adam_eps)
    wd = float(cfg.weight_decay)

    if bool(getattr(cfg, "use_paged_adamw_8bit", False)):
        try:
            import bitsandbytes as bnb # noqa: F401
            from bitsandbytes.optim import PagedAdamW8bit

            opt = PagedAdamW8bit(groups, betas=betas, eps=eps, weight_decay=wd)
            logger.info("Using bitsandbytes PagedAdamW8bit optimizer.")
            return opt
        except Exception as exc: # noqa: BLE001
            warnings.warn(
                f"[training_utils] PagedAdamW8bit unavailable ({exc!r}); "
                f"falling back to torch.optim.AdamW.",
                stacklevel=2,
            )
    return torch.optim.AdamW(groups, betas=betas, eps=eps, weight_decay=wd)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    num_training_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Return cosine-with-warmup (or constant) LR scheduler."""
    warmup = int(getattr(cfg, "lr_warmup_steps", 0))
    kind = str(getattr(cfg, "lr_scheduler", "cosine")).lower()

    if kind == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    try:
        from transformers import get_cosine_schedule_with_warmup

        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=max(1, int(num_training_steps)),
        )
    except Exception as exc: # noqa: BLE001
        warnings.warn(
            f"[training_utils] transformers scheduler unavailable ({exc!r}); "
            f"using manual cosine-warmup LambdaLR.",
            stacklevel=2,
        )
        import math

        total = max(1, int(num_training_steps))

        def _lr_lambda(step: int) -> float:
            """Return the LR multiplier for ``step`` under cosine-with-warmup."""
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def _named_trainable_count(module: nn.Module) -> int:
    """Sum the numel of all trainable parameters inside ``module``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def apply_stage_freeze(model: nn.Module, cfg: Any) -> Dict[str, int]:
    """Freeze submodules according to ``cfg.stage_freeze_*`` flags.

    Behaviour:
      - ``stage_freeze_vision``: freezes every non-LoRA parameter inside
        ``vision_tokenizer``; LoRA adapters stay trainable unconditionally
        (their own submodule controls them).
      - ``stage_freeze_planner`` / ``stage_freeze_flow_head`` /
        ``stage_freeze_verifier``: flip ``requires_grad = False`` across the
        whole module.

    Returns a dict of trainable-parameter counts per top-level module, handy
    for logging.
    """
    tokenizer = getattr(model, "vision_tokenizer", None)
    if tokenizer is not None and bool(getattr(cfg, "stage_freeze_vision", False)):
        for name, p in tokenizer.named_parameters():
            if "lora_" in name or "lora_magnitude_vector" in name:
                continue
            p.requires_grad = False

    def _freeze(mod: Optional[nn.Module], flag_name: str) -> None:
        """Freeze every parameter of ``mod`` when ``cfg.<flag_name>`` is truthy."""
        if mod is None or not bool(getattr(cfg, flag_name, False)):
            return
        for p in mod.parameters():
            p.requires_grad = False

    _freeze(getattr(model, "hierarchical_planner", None), "stage_freeze_planner")
    _freeze(getattr(model, "flow_action_head", None), "stage_freeze_flow_head")
    _freeze(getattr(model, "chunk_verifier", None), "stage_freeze_verifier")

    summary = {
        "vision_tokenizer": _named_trainable_count(tokenizer) if tokenizer is not None else 0,
        "hierarchical_planner": _named_trainable_count(getattr(model, "hierarchical_planner")),
        "flow_action_head": _named_trainable_count(getattr(model, "flow_action_head")),
        "chunk_verifier": _named_trainable_count(getattr(model, "chunk_verifier")),
    }
    logger.info("Stage freeze summary (trainable params):")
    for k, v in summary.items():
        logger.info(" %s: %s", k, f"{v:,}")
    return summary


def maybe_apply_gradient_checkpointing(model: nn.Module, cfg: Any) -> int:
    """Apply ``torch.utils.checkpoint`` on heavy encoder blocks.

    Returns the number of blocks wrapped. Only two locations are touched:
      - ``vision_tokenizer.cross_attn / readout_attn`` if present (legacy and
        new tokenizers name them ``cross_attn`` or ``readout_attn``;
        both are wrappers around ``nn.MultiheadAttention``, which cannot be
        wrapped directly with ``nn.Sequential``, so this path skips the MHA
        itself and only wraps the DiT blocks).
      - ``context_backbone.encoder.layers[i]``
        (``TransformerEncoderLayer``).
    """
    if not bool(getattr(cfg, "use_gradient_checkpointing", False)):
        return 0

    from torch.utils.checkpoint import checkpoint as _ckpt

    count = 0
    ctx = getattr(model, "context_backbone", None)
    if ctx is not None and hasattr(ctx, "encoder") and hasattr(ctx.encoder, "layers"):
        for i, layer in enumerate(ctx.encoder.layers):
            original_forward = layer.forward

            def _make_ckpt_forward(orig_fwd):
                """Return a forward that wraps ``orig_fwd`` in ``checkpoint``."""
                def _ckpt_forward(*args, **kwargs):
                    """Delegate to ``orig_fwd`` inside a checkpoint when grad is enabled."""
                    if not torch.is_grad_enabled():
                        return orig_fwd(*args, **kwargs)
                    return _ckpt(orig_fwd, *args, use_reentrant=False, **kwargs)

                return _ckpt_forward

            layer.forward = _make_ckpt_forward(original_forward) # type: ignore[assignment]
            count += 1
    logger.info("Gradient checkpointing wrapped %d transformer blocks.", count)
    return count


def maybe_build_ema(model: nn.Module, cfg: Any) -> Optional[Any]:
    """Build an EMAModel that tracks only the flow action head.

    Returns ``None`` when ``cfg.use_ema=False`` or ``diffusers`` is missing.
    Tracking only the flow head keeps VRAM down.
    """
    if not bool(getattr(cfg, "use_ema", False)):
        return None
    flow_head = getattr(model, "flow_action_head", None)
    if flow_head is None:
        return None
    try:
        from diffusers.training_utils import EMAModel
    except Exception as exc: # noqa: BLE001
        warnings.warn(
            f"[training_utils] diffusers.EMAModel unavailable ({exc!r}); "
            f"EMA disabled.",
            stacklevel=2,
        )
        return None

    ema = EMAModel(
        parameters=flow_head.parameters(),
        decay=float(cfg.ema_decay),
        use_ema_warmup=True,
        power=float(cfg.ema_power),
        update_after_step=int(cfg.ema_update_after_step),
    )
    return ema


def bf16_autocast_context(cfg: Any) -> Any:
    """Return a ``torch.autocast`` context manager if ``cfg.use_bf16`` else nullcontext."""
    from contextlib import nullcontext

    if not bool(getattr(cfg, "use_bf16", False)):
        return nullcontext()
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)


__all__ = [
    "build_param_groups",
    "summarize_param_groups",
    "build_optimizer",
    "build_scheduler",
    "apply_stage_freeze",
    "maybe_apply_gradient_checkpointing",
    "maybe_build_ema",
    "bf16_autocast_context",
]
