"""Data processor for PhaseQFlow training/inference batches.

Responsibilities of this processor:
  1. Resize and normalise images at the timm SigLIP inference resolution
     (``vision_image_size``), using the mean/std resolved by
     ``timm.data.resolve_data_config``.
  2. Multi-view support: LIBERO's ``observation.images.image`` (agentview)
     and ``observation.images.wrist_image`` (eye-in-hand) are stacked into
     a ``(V, C, H, W)`` tensor.
  3. Pre-tokenise the instruction with a T5 tokenizer and return
     ``language_ids`` plus ``language_mask``.
  4. Legacy compatibility: single-view or language-less samples still
     collate cleanly; the language fields are filled with padding tensors.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import torchvision.transforms as T


_LIBERO_VIEW_KEYS: tuple[str, ...] = (
    "observation.images.image",
    "observation.images.agentview",
    "observation.images.wrist_image",
    "observation.images.eye_in_hand",
)


@dataclass
class ProcessorConfig:
    """Processor configuration: image resolution, tokenizer, normalisation, and augmentation knobs."""

    num_skills: int = 16
    use_vq_phase: bool = True
    use_value_guided_weight: bool = True
    state_noise_std: float = 0.01
    image_randaugment_n: int = 2
    image_randaugment_m: int = 9

    vision_image_size: int = 224
    vision_backbone_siglip: str = "vit_base_patch16_siglip_224"
    language_encoder_name: str = "google-t5/t5-base"
    language_token_max_len: int = 16
    num_camera_views: int = 2
    image_mean: tuple = (0.5, 0.5, 0.5)
    image_std: tuple = (0.5, 0.5, 0.5)
    enable_language_tokenizer: bool = True
    view_keys: tuple = field(default_factory=lambda: _LIBERO_VIEW_KEYS)

    use_action_quantile_norm: bool = True
    action_quantile_low: float = 0.01
    action_quantile_high: float = 0.99
    dataset_stats_path: Optional[str] = None
    state_noise_snr_db: float = 40.0

    action_chunk_size: int = 16
    tile_single_step_actions: bool = True


class ActionQuantileNormalizer:
    """Per-dimension quantile normalizer for continuous actions.

    Normalisation formula::

        a_norm = 2 * (a - q_low) / (q_high - q_low + 1e-6) - 1
        a_norm = clamp(a_norm, -1, 1)

    Denormalisation: ``a = (a_norm + 1) * (q_high - q_low) / 2 + q_low``.

    - ``fit(actions)`` estimates quantiles on the first batch and caches
      them in ``_q_low`` / ``_q_high``. If statistics have been loaded from
      ``dataset_stats.json`` the file values win and are not overwritten.
    - ``normalize`` / ``denormalize`` apply the forward / inverse transform
      to a batch.
    """

    def __init__(
        self,
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
        stats_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Set quantile bounds and optionally preload them from ``stats_path``."""
        self.quantile_low = float(quantile_low)
        self.quantile_high = float(quantile_high)
        self._q_low: Optional[torch.Tensor] = None
        self._q_high: Optional[torch.Tensor] = None
        self._loaded_from_file = False
        if stats_path is not None:
            self._try_load_stats(stats_path)

    def _try_load_stats(self, stats_path: Union[str, Path]) -> None:
        """Read ``action_q_low`` / ``action_q_high`` from a ``dataset_stats.json``."""
        p = Path(stats_path)
        if not p.is_file():
            warnings.warn(
                f"[ActionQuantileNormalizer] dataset_stats file not found: {p}; "
                f"will fit on first batch instead.",
                stacklevel=2,
            )
            return
        try:
            with p.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            q_low = payload.get("action_q_low")
            q_high = payload.get("action_q_high")
            if q_low is None or q_high is None:
                raise KeyError("missing action_q_low / action_q_high in dataset_stats")
            self._q_low = torch.as_tensor(q_low, dtype=torch.float32)
            self._q_high = torch.as_tensor(q_high, dtype=torch.float32)
            self._loaded_from_file = True
        except Exception as exc: # noqa: BLE001
            warnings.warn(
                f"[ActionQuantileNormalizer] failed to load {p}: {exc!r}; "
                f"will fit on first batch instead.",
                stacklevel=2,
            )

    def fit(self, actions: torch.Tensor) -> None:
        """Estimate per-dimension quantiles from a batch of actions (B, D)."""
        if self._loaded_from_file:
            return
        if actions.ndim < 2:
            return
        flat = actions.reshape(-1, actions.shape[-1]).float()
        q = torch.tensor([self.quantile_low, self.quantile_high], device=flat.device)
        bounds = torch.quantile(flat, q, dim=0)
        self._q_low = bounds[0].detach().cpu()
        self._q_high = bounds[1].detach().cpu()

    @property
    def is_fitted(self) -> bool:
        """True once both quantile tensors are populated."""
        return self._q_low is not None and self._q_high is not None

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Scale ``actions`` to [-1, 1] using the cached quantile bounds."""
        if not self.is_fitted:
            self.fit(actions)
        assert self._q_low is not None and self._q_high is not None
        q_low = self._q_low.to(actions.device, actions.dtype)
        q_high = self._q_high.to(actions.device, actions.dtype)
        scale = (q_high - q_low).clamp_min(1e-6)
        out = 2.0 * (actions - q_low) / scale - 1.0
        return out.clamp(-1.0, 1.0)

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`normalize` back to the original action scale."""
        assert self._q_low is not None and self._q_high is not None, (
            "ActionQuantileNormalizer must be fit before denormalize()"
        )
        q_low = self._q_low.to(actions.device, actions.dtype)
        q_high = self._q_high.to(actions.device, actions.dtype)
        scale = q_high - q_low
        return (actions + 1.0) * 0.5 * scale + q_low


class PhaseQFlowProcessor:
    """Prepare explicit multimodal tensors for the dual-backbone policy forward path."""

    def __init__(self, config: ProcessorConfig) -> None:
        """Initialize image normalization, RandAugment, and the T5 tokenizer."""
        self.config = config
        self.image_aug = T.RandAugment(num_ops=config.image_randaugment_n, magnitude=config.image_randaugment_m)

        mean, std = self._resolve_timm_normalization(config)
        self._image_mean = torch.tensor(mean).view(1, 3, 1, 1)
        self._image_std = torch.tensor(std).view(1, 3, 1, 1)
        self._image_resize = T.Resize(
            (int(config.vision_image_size), int(config.vision_image_size)),
            antialias=True,
        )

        self._tokenizer = self._build_tokenizer(config) if config.enable_language_tokenizer else None

        if config.use_action_quantile_norm:
            self.action_normalizer: Optional[ActionQuantileNormalizer] = ActionQuantileNormalizer(
                quantile_low=config.action_quantile_low,
                quantile_high=config.action_quantile_high,
                stats_path=config.dataset_stats_path,
            )
        else:
            self.action_normalizer = None

    @staticmethod
    def _resolve_timm_normalization(config: ProcessorConfig) -> tuple[tuple, tuple]:
        """Resolve SigLIP inference-time mean/std through ``timm.data.resolve_data_config``."""
        try:
            import timm
            from timm.data import resolve_data_config

            model = timm.create_model(config.vision_backbone_siglip, pretrained=False, num_classes=0)
            data_cfg = resolve_data_config({}, model=model)
            return tuple(data_cfg.get("mean", config.image_mean)), tuple(data_cfg.get("std", config.image_std))
        except Exception as exc: # noqa: BLE001
            warnings.warn(
                f"[PhaseQFlowProcessor] timm normalization resolution failed ({exc!r}); "
                f"falling back to config.image_mean/std={config.image_mean}/{config.image_std}",
                stacklevel=2,
            )
            return tuple(config.image_mean), tuple(config.image_std)

    @staticmethod
    def _build_tokenizer(config: ProcessorConfig) -> Optional[Any]:
        """Build the T5 tokenizer; return ``None`` on failure so the zero-token fallback runs."""
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(config.language_encoder_name)
        except Exception as exc: # noqa: BLE001
            warnings.warn(
                f"[PhaseQFlowProcessor] failed to load T5 tokenizer '{config.language_encoder_name}' ({exc!r}); "
                f"samples will fall back to zero input_ids / zero mask.",
                stacklevel=2,
            )
            return None

    @staticmethod
    def _to_tensor(x: Any) -> torch.Tensor:
        """Convert input data into a torch tensor without copying when possible."""
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    def _augment_and_normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Run RandAugment (training only) plus resize and ImageNet/timm normalisation.

        ``images`` may be ``(N, C, H, W)`` or ``(N, V, C, H, W)``. The rank is
        preserved; the spatial resolution is set to
        ``(vision_image_size, vision_image_size)``.
        """
        if images.ndim == 5:
            n, v, c, h, w = images.shape
            flat = images.reshape(n * v, c, h, w)
            flat = self._aug_norm_bchw(flat)
            return flat.reshape(n, v, c, int(self.config.vision_image_size), int(self.config.vision_image_size))
        return self._aug_norm_bchw(images)

    def _aug_norm_bchw(self, images: torch.Tensor) -> torch.Tensor:
        """Apply augmentation, resize, and normalisation to a ``(B, C, H, W)`` batch."""
        if images.ndim == 4 and images.shape[1] == 3:
            aug_list = []
            for img in images:
                img_in = img
                if img_in.dtype != torch.uint8:
                    img_in = (img_in.clamp(0, 1) * 255.0).to(torch.uint8)
                try:
                    img_out = self.image_aug(img_in).float() / 255.0
                except Exception: # noqa: BLE001
                    img_out = img.float()
                aug_list.append(img_out)
            images = torch.stack(aug_list, dim=0)
        images = self._image_resize(images)
        mean = self._image_mean.to(images.device, images.dtype)
        std = self._image_std.to(images.device, images.dtype)
        return (images - mean) / std

    def _stack_views(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Collect every camera view listed in ``config.view_keys`` and stack into ``(V, C, H, W)``."""
        collected: List[torch.Tensor] = []
        for key in self.config.view_keys:
            if key in sample and sample[key] is not None:
                t = self._to_tensor(sample[key]).float()
                if t.ndim == 3:
                    if t.shape[-1] == 3 and t.shape[0] != 3:
                        t = t.permute(2, 0, 1)
                collected.append(t)
        if not collected:
            for fallback in ("observation.images", "images", "image"):
                if fallback in sample and sample[fallback] is not None:
                    t = self._to_tensor(sample[fallback]).float()
                    if t.ndim == 4:
                        return t
                    if t.ndim == 3 and t.shape[-1] == 3 and t.shape[0] != 3:
                        t = t.permute(2, 0, 1)
                    collected.append(t)
                    break
        if not collected:
            raise KeyError("Each sample must include at least one camera view under the configured view_keys")
        return torch.stack(collected, dim=0)

    def _tokenize_batch(self, instructions: List[Optional[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        """T5-tokenise a batch of instructions; fall back to zero tensors with no tokenizer."""
        max_len = int(self.config.language_token_max_len)
        batch_size = len(instructions)
        if self._tokenizer is None:
            ids = torch.zeros(batch_size, max_len, dtype=torch.long)
            mask = torch.zeros(batch_size, max_len, dtype=torch.long)
            return ids, mask
        filled = [s if isinstance(s, str) and s else "" for s in instructions]
        enc = self._tokenizer(
            filled,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].long(), enc["attention_mask"].long()

    def _apply_state_snr_noise(self, states: torch.Tensor, training: bool) -> torch.Tensor:
        """Inject Gaussian noise into proprioception at the target SNR in dB (RDT trick).

        ``sigma = |state|_rms * 10^(-SNR_dB / 20)``. Active only when
        ``training=True`` and ``state_noise_snr_db > 0``; otherwise the
        tensor passes through unchanged.
        """
        snr_db = float(getattr(self.config, "state_noise_snr_db", 0.0))
        if not training or snr_db <= 0:
            return states
        rms = states.float().pow(2).mean().clamp_min(1e-12).sqrt()
        sigma = rms * (10.0 ** (-snr_db / 20.0))
        return states + torch.randn_like(states) * sigma

    def _collect_actions(self, batch: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
        """Gather per-sample action tensors.

        ShortcutFlowActionHead expects ``(B, Ta, Da)`` action chunks. The
        resulting shapes are:

            (B, Ta, Da) when ``tile_single_step_actions=True`` and samples
                are single-step (D,);
            (B, T, Da) when samples already deliver multi-step chunks,
                even if T does not match Ta;
            (B, D) when ``tile_single_step_actions=False`` and samples are
                single-step;
            None when any sample is missing the action field.
        """
        collected: List[torch.Tensor] = []
        for sample in batch:
            act = sample.get("action", sample.get("actions", sample.get("target_action")))
            if act is None:
                return None
            collected.append(self._to_tensor(act).float())
        try:
            stacked = torch.stack(collected, dim=0)
        except RuntimeError:
            return None

        if stacked.ndim == 2 and bool(getattr(self.config, "tile_single_step_actions", False)):
            Ta = int(getattr(self.config, "action_chunk_size", 1))
            if Ta > 1:
                if not getattr(self, "_warned_tile_actions", False):
                    warnings.warn(
                        f"[PhaseQFlowProcessor] dataset returned single-step actions "
                        f"(B, Da); tiling to (B, Ta={Ta}, Da) for ShortcutFlowActionHead. "
                        f"Set ``tile_single_step_actions=False`` to disable.",
                        stacklevel=2,
                    )
                    self._warned_tile_actions = True
                stacked = stacked.unsqueeze(1).expand(-1, Ta, -1).contiguous()
        return stacked

    def __call__(self, batch: List[Dict[str, Any]], training: bool = True) -> Dict[str, torch.Tensor]:
        """Build explicit multimodal batch payload for the dual-backbone policy."""
        obs_images: List[torch.Tensor] = []
        obs_states: List[torch.Tensor] = []
        legacy_language: List[torch.Tensor] = []
        history: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        instructions: List[Optional[str]] = []

        for sample in batch:
            states = sample.get("observation.state", sample.get("state", sample.get("states")))
            if states is None:
                raise KeyError("Each sample must include observation states")
            lang = sample.get("instruction", sample.get("task_descriptor", sample.get("language")))
            hist = sample.get("observation.history", sample.get("history", states))
            m = sample.get("observation.mask", sample.get("mask", 1.0))

            views = self._stack_views(sample)
            obs_images.append(views)

            obs_states.append(self._to_tensor(states).float())
            history.append(self._to_tensor(hist).float())
            masks.append(self._to_tensor(m).float())

            if isinstance(lang, str):
                instructions.append(lang)
                legacy_language.append(torch.zeros(1))
            elif isinstance(lang, torch.Tensor):
                instructions.append(None)
                legacy_language.append(lang.float().flatten())
            elif lang is None:
                instructions.append(None)
                legacy_language.append(torch.zeros(1))
            else:
                instructions.append(str(lang))
                legacy_language.append(torch.zeros(1))

        max_v = max(t.shape[0] for t in obs_images)
        padded = []
        for t in obs_images:
            if t.shape[0] < max_v:
                pad_count = max_v - t.shape[0]
                pad_block = torch.zeros(pad_count, *t.shape[1:], dtype=t.dtype)
                t = torch.cat([t, pad_block], dim=0)
            padded.append(t)
        obs_images_t = torch.stack(padded, dim=0)
        obs_images_t = self._augment_and_normalize(obs_images_t)

        obs_states_t = torch.stack(obs_states, dim=0)
        if self.config.state_noise_std > 0 and training:
            obs_states_t = obs_states_t + torch.randn_like(obs_states_t) * self.config.state_noise_std
        obs_states_t = self._apply_state_snr_noise(obs_states_t, training=training)

        history_t = torch.stack(history, dim=0)
        masks_t = torch.stack(masks, dim=0)

        try:
            legacy_language_t = torch.stack(legacy_language, dim=0)
        except RuntimeError:
            max_len = max(t.numel() for t in legacy_language) or 1
            legacy_language_t = torch.stack(
                [F.pad(t.flatten(), (0, max_len - t.numel())) for t in legacy_language], dim=0
            )

        language_ids, language_mask = self._tokenize_batch(instructions)

        batch_size = obs_images_t.shape[0]
        skill_id = torch.full((batch_size,), -1, dtype=torch.long)
        sample_weight = torch.ones(batch_size, dtype=torch.float32)

        actions_t = self._collect_actions(batch)
        if actions_t is not None and self.action_normalizer is not None:
            actions_t = self.action_normalizer.normalize(actions_t)

        out = {
            "obs": {
                "images": obs_images_t,
                "states": obs_states_t,
                "language": legacy_language_t,
                "language_ids": language_ids,
                "language_mask": language_mask,
                "history": history_t,
                "masks": masks_t,
            },
            "obs_images": obs_images_t,
            "obs_states": obs_states_t,
            "language": legacy_language_t,
            "language_ids": language_ids,
            "language_mask": language_mask,
            "history": history_t,
            "masks": masks_t,
            "skill_id": skill_id,
            "sample_weight": sample_weight,
        }
        if actions_t is not None:
            out["action"] = actions_t
        return out
