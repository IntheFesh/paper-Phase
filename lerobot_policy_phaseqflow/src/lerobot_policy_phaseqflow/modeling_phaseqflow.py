"""Core PhaseQFlow policy components with four-layer architecture."""

from __future__ import annotations

import copy
import logging
import math
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_phaseqflow import PhaseQFlowConfig

logger = logging.getLogger(__name__)


class SkillVQEncoder(nn.Module):
    """Gumbel-Softmax discrete phase encoder."""

    def __init__(self, input_dim: int, num_skills: int, temperature: float = 1.0) -> None:
        """Initialize the linear projection used for discrete phase inference."""
        super().__init__()
        self.proj = nn.Linear(input_dim, num_skills)
        self.temperature = temperature

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Infer discrete skill ids and return `(id, probs, logits)`."""
        logits = self.proj(x)
        probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1) if training else logits.softmax(dim=-1)
        skill_id = probs.argmax(dim=-1)
        return skill_id, probs, logits


class FSQSkillEncoder(nn.Module):
    """Finite Scalar Quantization discrete phase encoder.

    Substitutes Gumbel-Softmax with FSQ (Mentzer et al., arXiv 2309.15505):
    project ``fused_obs`` to an ``fsq_dim``-dim continuous vector, then quantize
    each dimension independently with FSQ to implicitly form ``prod(levels)``
    discrete codes. Unlike Gumbel-Softmax and classic VQ, FSQ needs no EMA
    codebook updates and does not collapse, giving a steadier phase code on
    long-horizon tasks (VQ-BeT, Lee et al., arXiv 2403.03181, discusses the
    value of discrete phase tokens).

    For interface parity with :class:`SkillVQEncoder`, ``forward`` returns the
    triple ``(phase_id, probs, logits)``:
      - ``phase_id``: shape ``(B,)``, discrete code id in ``[0, codebook_size)``.
      - ``probs``: shape ``(B, codebook_size)``, softmax of ``fake_logits``.
      - ``logits``: pseudo logits, 10.0 at the hit index and 0 elsewhere; mostly
        consumed by downstream ``F.cross_entropy`` with manual phase labels.
    """

    def __init__(self, input_dim: int, cfg: PhaseQFlowConfig) -> None:
        """Build the projection, FSQ quantizer, and code to phase_embed table."""
        super().__init__()
        from vector_quantize_pytorch import FSQ

        if int(cfg.fsq_dim) != len(cfg.fsq_levels):
            raise ValueError(
                f"fsq_dim ({cfg.fsq_dim}) must equal len(fsq_levels) ({len(cfg.fsq_levels)})"
            )
        self.cfg = cfg
        self.levels: List[int] = list(cfg.fsq_levels) # type: ignore[valid-type]
        self.codebook_size = int(np.prod(self.levels))
        self.proj = nn.Linear(input_dim, int(cfg.fsq_dim))
        self.fsq = FSQ(levels=self.levels)
        self.code_embed = nn.Embedding(
            num_embeddings=self.codebook_size,
            embedding_dim=int(cfg.skill_embedding_dim),
        )

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``fused_obs`` and return the Gumbel-compatible ``(phase_id, probs, logits)`` triple."""
        del training
        z = self.proj(x)
        z_unsq = z.unsqueeze(1)
        _z_q, indices = self.fsq(z_unsq)
        indices = indices.squeeze(1).long()

        batch = z.shape[0]
        fake_logits = torch.zeros(batch, self.codebook_size, device=z.device, dtype=z.dtype)
        fake_logits.scatter_(1, indices.unsqueeze(1), 10.0)
        probs = fake_logits.softmax(dim=-1)
        return indices, probs, fake_logits


def infonce_phase_loss(
    phase_embeds: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    r"""Temporal InfoNCE contrastive loss (arXiv 1807.03748) for sharpening adjacent-chunk phase boundaries.

    Given a ``phase_embed`` time series of shape ``(B, T, D)``: treat ``(t, t+1)``
    as the positive pair (nearby chunks are likely to share phase) and every
    other token in the batch as a negative. Uses in-batch negatives with
    symmetric cosine similarity plus softmax cross-entropy:

    .. math::
        L = - \frac{1}{N} \sum_{i} \log
            \frac{\exp(\mathrm{sim}(z_i, z_i^{+}) / \tau)}
                 {\sum_{j} \exp(\mathrm{sim}(z_i, z_j) / \tau)}

    where :math:`z_i^{+}` is the next-timestep embedding of :math:`z_i`, and
    the denominator sums over all tokens in the batch (anchors from other
    episodes act as negatives too).

    Returns a zero tensor when ``T < 2`` (no adjacent pair can be formed).
    """
    if phase_embeds.ndim != 3:
        return phase_embeds.new_zeros(())
    b, t, d = phase_embeds.shape
    if t < 2 or b < 1:
        return phase_embeds.new_zeros(())

    z = F.normalize(phase_embeds, dim=-1)
    anchors = z[:, :-1, :].reshape(-1, d)
    tokens = z.reshape(-1, d)

    sim = anchors @ tokens.T / max(float(temperature), 1e-6)

    positive_idx = (
        torch.arange(b, device=z.device).unsqueeze(1) * t
        + torch.arange(1, t, device=z.device).unsqueeze(0)
    ).reshape(-1)

    return F.cross_entropy(sim, positive_idx)


'''
resnet18 - VisionTokenizer
from torchvision.models import resnet18
from typing import Dict, Optional

class VisionTokenizer(nn.Module):
    def __init__(self, config: object) -> None:
        super().__init__()
        self.config = config
        self.spatial_grid = getattr(config, "spatial_grid_size", 4)
        resnet = resnet18(weights=None)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.vision_pool = nn.AdaptiveAvgPool2d((self.spatial_grid, self.spatial_grid))
        self.vision_proj = nn.Sequential(
            nn.Conv2d(512, config.vision_token_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=config.vision_token_dim),
            nn.GELU()
        )

        self.vision_adapter = nn.Linear(config.vision_token_dim, config.fusion_hidden_dim)
        self.state_tokenizer = nn.LazyLinear(config.fusion_hidden_dim)
        self.language_tokenizer = nn.LazyLinear(config.fusion_hidden_dim)
        self.history_tokenizer = nn.LazyLinear(config.fusion_hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.fusion_hidden_dim,
            num_heads=config.cross_attn_heads,
            dropout=config.cross_attn_dropout,
            batch_first=True,
        )

        self.uncertainty_gate = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim * 2, config.fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.fusion_hidden_dim, 1),
        )

    def maybe_freeze_vision(self) -> None:
        if not getattr(self.config, "freeze_vision_encoder", False):
            return
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.vision_pool.parameters():
            p.requires_grad = False
        for p in self.vision_proj.parameters():
            p.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        language: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del masks
        if states.ndim > 2:
            states = states.flatten(start_dim=1)

        B = images.shape[0]
        feat = self.vision_encoder(images) # [B, 512, H/32, W/32]
        feat = self.vision_pool(feat) # [B, 512, G, G]
        feat = self.vision_proj(feat) # [B, token_dim, G, G]
        feat = feat.permute(0, 2, 3, 1).reshape(B, -1, self.config.vision_token_dim) # [B, G², token_dim]
        vision_tokens = self.vision_adapter(feat) # [B, G², fusion_hidden_dim]

        state_tokens = self.state_tokenizer(states).unsqueeze(1)
        if language is None:
            language = torch.zeros_like(states[:, :1])
        if language.ndim > 2:
            language = language.flatten(start_dim=1)
        language_tokens = self.language_tokenizer(language).unsqueeze(1)
        if history is None:
            history = states
        if history.ndim > 2:
            history = history.flatten(start_dim=1)
        history_tokens = self.history_tokenizer(history).unsqueeze(1)
        query_tokens = torch.cat([state_tokens, history_tokens], dim=1)
        attended, _ = self.cross_attn(query=query_tokens, key=vision_tokens, value=vision_tokens)
        attended_summary = attended.mean(dim=1)
        proprio_summary = torch.cat([state_tokens, history_tokens], dim=1).mean(dim=1)

        gate = torch.sigmoid(self.uncertainty_gate(torch.cat([attended_summary, proprio_summary], dim=-1)))
        fused = gate * attended_summary + (1.0 - gate) * proprio_summary
        vision_tokens_pooled = vision_tokens.mean(dim=1, keepdim=True)
        context_tokens = torch.cat([state_tokens, history_tokens, language_tokens, vision_tokens_pooled], dim=1)

        return {
            "fused": fused,
            "context_tokens": context_tokens,
            "vision_tokens": vision_tokens,
            "state_tokens": state_tokens,
            "language_tokens": language_tokens,
            "history_tokens": history_tokens,
            "uncertainty_gate": gate.squeeze(-1),
        }
        '''


class VisionTokenizerLegacy(nn.Module):
    """[DEPRECATED] Legacy vision tokenizer based on ``nn.LazyLinear``.

    Superseded by :class:`DualBackboneVisionTokenizer`. Retained only for
    loading old checkpoints or as a fallback when
    ``use_dual_backbone_vision=False``; avoid using it in new experiments.
    """

    def __init__(self, config: PhaseQFlowConfig) -> None:
        """Build multimodal tokenizers, cross-attention, and uncertainty gate."""
        super().__init__()
        self.config = config

        self.vision_backbone = nn.LazyLinear(config.vision_token_dim)
        self.vision_adapter = nn.Linear(config.vision_token_dim, config.fusion_hidden_dim)
        self.state_tokenizer = nn.LazyLinear(config.fusion_hidden_dim)
        self.language_tokenizer = nn.LazyLinear(config.fusion_hidden_dim)
        self.history_tokenizer = nn.LazyLinear(config.fusion_hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.fusion_hidden_dim,
            num_heads=config.cross_attn_heads,
            dropout=config.cross_attn_dropout,
            batch_first=True,
        )

        self.uncertainty_gate = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim * 2, config.fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.fusion_hidden_dim, 1),
        )

    def maybe_freeze_vision(self) -> None:
        """Freeze the vision backbone when adapter-style training is requested."""
        if not self.config.freeze_vision_encoder:
            return
        for p in self.vision_backbone.parameters():
            p.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        language: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        language_ids: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize multimodal inputs and fuse visual/proprioceptive context."""
        del masks, language_ids, language_mask
        if images.ndim > 2:
            images = images.flatten(start_dim=1)
        if states.ndim > 2:
            states = states.flatten(start_dim=1)

        vision_tokens = self.vision_backbone(images).unsqueeze(1)
        vision_tokens = self.vision_adapter(vision_tokens)

        state_tokens = self.state_tokenizer(states).unsqueeze(1)

        if language is None:
            language = torch.zeros_like(states[:, :1])
        if language.ndim > 2:
            language = language.flatten(start_dim=1)
        language_tokens = self.language_tokenizer(language).unsqueeze(1)

        if history is None:
            history = states
        if history.ndim > 2:
            history = history.flatten(start_dim=1)
        history_tokens = self.history_tokenizer(history).unsqueeze(1)

        query_tokens = torch.cat([state_tokens, history_tokens], dim=1)
        attended, _ = self.cross_attn(query=query_tokens, key=vision_tokens, value=vision_tokens)
        attended_summary = attended.mean(dim=1)
        proprio_summary = torch.cat([state_tokens, history_tokens], dim=1).mean(dim=1)

        gate = torch.sigmoid(self.uncertainty_gate(torch.cat([attended_summary, proprio_summary], dim=-1)))
        fused = gate * attended_summary + (1.0 - gate) * proprio_summary

        context_tokens = torch.cat([state_tokens, history_tokens, language_tokens, vision_tokens], dim=1)
        return {
            "fused": fused,
            "context_tokens": context_tokens,
            "vision_tokens": vision_tokens,
            "state_tokens": state_tokens,
            "language_tokens": language_tokens,
            "history_tokens": history_tokens,
            "uncertainty_gate": gate.squeeze(-1),
        }


class DualBackboneVisionTokenizer(nn.Module):
    """Dual-backbone vision tokenizer: SigLIP2 + DINOv2 + FiLM-T5 + Perceiver readout.

    Composition:
      (a) Two frozen vision backbones (SigLIP via timm + DINOv2 via hub), each
          adapted with ``peft.LoraConfig(target_modules=["qkv","proj"], use_dora=...)``.
      (b) Channel-dimension concatenation: reshape both patch-token sequences
          into 2D grids, interpolate to a common resolution, concat along the
          channel axis, then project with a 1x1 conv to ``fusion_hidden_dim``.
      (c) Multi-view support: accepts ``(B, V, C, H, W)`` where V is the number
          of cameras (LIBERO=2); single-view ``(B, C, H, W)`` is auto-expanded
          to ``V=1``.
      (d) Frozen T5 encoder pooled into ``lang_vec``; two MLPs produce FiLM
          parameters gamma/beta that modulate vision tokens by
          ``v_i <- gamma * v_i + beta``.
      (e) Perceiver-style ``num_readout_tokens`` learnable queries that pull in
          vision + language tokens through cross-attention; their outputs serve
          as the main ``context_tokens`` downstream.
      (f) Keeps the ``"fused"`` output key; the gate now consumes
          ``[readout_mean, state_mean]``.

    The output dict keys match :class:`VisionTokenizerLegacy` exactly so that
    downstream ``HierarchicalPlanner`` / ``FlowActionHead`` / ``ChunkVerifier``
    interfaces remain unchanged.
    """

    _DINOV2_TIMM_FALLBACK: Dict[str, str] = {
        "dinov2_vits14": "vit_small_patch14_dinov2",
        "dinov2_vitb14": "vit_base_patch14_dinov2",
        "dinov2_vitl14": "vit_large_patch14_dinov2",
        "dinov2_vitg14": "vit_giant_patch14_dinov2",
    }

    def __init__(self, config: PhaseQFlowConfig) -> None:
        """Build backbones, LoRA adapters, FiLM-T5 conditioning, and readout attention."""
        super().__init__()
        self.config = config
        self.image_size = int(config.vision_image_size)
        self.fusion_dim = int(config.fusion_hidden_dim)
        self.num_readout = int(config.num_readout_tokens)

        self.siglip, self.siglip_dim, self.siglip_grid = self._build_siglip(config)
        self.dinov2, self.dinov2_dim, self.dinov2_grid, self._dinov2_flavor = self._build_dinov2(config)
        self._apply_lora(self.siglip, config)
        self._apply_lora(self.dinov2, config)
        self._freeze_non_lora(self.siglip)
        self._freeze_non_lora(self.dinov2)

        self.common_grid = int(min(self.siglip_grid, self.dinov2_grid))
        self.fusion_proj = nn.Conv2d(
            in_channels=self.siglip_dim + self.dinov2_dim,
            out_channels=self.fusion_dim,
            kernel_size=1,
        )
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)

        state_dim = int(getattr(config, "state_dim", 8))
        history_dim = int(getattr(config, "history_dim", state_dim))
        self.state_tokenizer = nn.Linear(state_dim, self.fusion_dim)
        self.history_tokenizer = nn.Linear(history_dim, self.fusion_dim)

        self.t5_encoder, self.t5_dim = self._build_t5(config)
        for p in self.t5_encoder.parameters():
            p.requires_grad = False
        self.language_proj = nn.Linear(self.t5_dim, self.fusion_dim)
        if config.use_film_language:
            self.film_gamma = nn.Sequential(
                nn.Linear(self.t5_dim, self.fusion_dim),
                nn.SiLU(),
                nn.Linear(self.fusion_dim, self.fusion_dim),
            )
            self.film_beta = nn.Sequential(
                nn.Linear(self.t5_dim, self.fusion_dim),
                nn.SiLU(),
                nn.Linear(self.fusion_dim, self.fusion_dim),
            )
        else:
            self.film_gamma = None
            self.film_beta = None
        if self.film_beta is not None:
            nn.init.zeros_(self.film_beta[-1].weight)
            nn.init.zeros_(self.film_beta[-1].bias)
        if self.film_gamma is not None:
            nn.init.zeros_(self.film_gamma[-1].weight)
            nn.init.ones_(self.film_gamma[-1].bias)

        self.readout_queries = nn.Parameter(torch.randn(self.num_readout, self.fusion_dim) * 0.02)
        self.readout_attn = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=config.cross_attn_heads,
            dropout=config.cross_attn_dropout,
            batch_first=True,
        )
        self.readout_norm_q = nn.LayerNorm(self.fusion_dim)
        self.readout_norm_out = nn.LayerNorm(self.fusion_dim)
        self.readout_ffn = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim * 2),
            nn.GELU(),
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
        )

        self.uncertainty_gate = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.SiLU(),
            nn.Linear(self.fusion_dim, 1),
        )

    @staticmethod
    def _build_siglip(config: PhaseQFlowConfig) -> Tuple[nn.Module, int, int]:
        """Build a SigLIP2 / SigLIP timm vision encoder; fall back to random init on network failure."""
        import timm

        name = config.vision_backbone_siglip
        img_size = int(config.vision_image_size)
        try:
            model = timm.create_model(name, pretrained=True, num_classes=0, img_size=img_size)
        except Exception as exc: # noqa: BLE001
            warnings.warn(
                f"[DualBackboneVisionTokenizer] SigLIP pretrained weight load failed ({exc!r}); "
                f"falling back to pretrained=False. Populate the timm/{name} weight cache on a "
                f"networked machine to recover full pretrained behavior.",
                stacklevel=2,
            )
            model = timm.create_model(name, pretrained=False, num_classes=0, img_size=img_size)
        model.eval()
        embed_dim = int(getattr(model, "embed_dim", 768))
        patch = int(getattr(model, "patch_embed").patch_size[0])
        grid = img_size // patch
        return model, embed_dim, grid

    def _build_dinov2(self, config: PhaseQFlowConfig) -> Tuple[nn.Module, int, int, str]:
        """Build a DINOv2 vision encoder; prefer torch.hub and fall back to a local timm copy."""
        import timm

        name = config.vision_backbone_dinov2
        img_size = int(config.vision_image_size)
        flavor = "hub"
        model: Optional[nn.Module] = None

        hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        try:
            if hub_dir.exists():
                model = torch.hub.load("facebookresearch/dinov2", name, source="local", trust_repo=True)
            else:
                model = torch.hub.load("facebookresearch/dinov2", name, trust_repo=True)
        except Exception as exc: # noqa: BLE001
            warnings.warn(
                f"[DualBackboneVisionTokenizer] DINOv2 torch.hub load failed ({exc!r}); "
                f"falling back to the timm offline model.",
                stacklevel=2,
            )
            timm_name = self._DINOV2_TIMM_FALLBACK.get(name, "vit_small_patch14_dinov2")
            try:
                model = timm.create_model(timm_name, pretrained=True, num_classes=0, img_size=img_size)
            except Exception as exc_timm: # noqa: BLE001
                warnings.warn(
                    f"[DualBackboneVisionTokenizer] timm DINOv2 pretrained download also failed "
                    f"({exc_timm!r}); continuing with random initialization.",
                    stacklevel=2,
                )
                model = timm.create_model(timm_name, pretrained=False, num_classes=0, img_size=img_size)
            flavor = "timm"

        assert model is not None
        model.eval()
        embed_dim = int(getattr(model, "embed_dim", 384))
        patch_size = self._infer_patch_size(model)
        grid = img_size // patch_size
        return model, embed_dim, grid, flavor

    @staticmethod
    def _infer_patch_size(model: nn.Module) -> int:
        """Infer patch size from a ViT model (compatible with timm and dinov2 structures)."""
        patch_embed = getattr(model, "patch_embed", None)
        if patch_embed is not None:
            ps = getattr(patch_embed, "patch_size", None)
            if isinstance(ps, (tuple, list)) and len(ps) >= 1:
                return int(ps[0])
            if isinstance(ps, int):
                return int(ps)
            proj = getattr(patch_embed, "proj", None)
            if proj is not None and hasattr(proj, "kernel_size"):
                return int(proj.kernel_size[0])
        return 14

    @staticmethod
    def _build_t5(config: PhaseQFlowConfig) -> Tuple[nn.Module, int]:
        """Build a frozen T5 encoder; fall back to a default ``T5Config`` random init on failure."""
        from transformers import T5Config, T5EncoderModel

        name = config.language_encoder_name
        try:
            model = T5EncoderModel.from_pretrained(name)
        except Exception as exc: # noqa: BLE001
            warnings.warn(
                f"[DualBackboneVisionTokenizer] T5 encoder load failed ({exc!r}); "
                f"falling back to a default T5Config random init. Language conditioning "
                f"degrades to a learnable projection in this state.",
                stacklevel=2,
            )
            model = T5EncoderModel(T5Config())
        model.eval()
        d_model = int(getattr(model.config, "d_model", 768))
        return model, d_model

    @staticmethod
    def _apply_lora(module: nn.Module, config: PhaseQFlowConfig) -> nn.Module:
        """Apply peft LoRA / DoRA to the attention qkv/proj layers of the given vision backbone."""
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=int(config.lora_rank),
            lora_alpha=int(config.lora_alpha),
            target_modules=["qkv", "proj"],
            lora_dropout=float(config.lora_dropout),
            use_dora=bool(config.use_dora),
            bias="none",
        )
        peft_model = get_peft_model(module, lora_cfg)
        return peft_model

    @staticmethod
    def _freeze_non_lora(module: nn.Module) -> None:
        """Defensively set every non-LoRA parameter to ``requires_grad=False``."""
        for name, p in module.named_parameters():
            if "lora_" in name or "lora_magnitude_vector" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def train(self, mode: bool = True) -> "DualBackboneVisionTokenizer":
        """Override training state: keep frozen backbones in ``eval()`` to freeze dropout / BN behavior."""
        super().train(mode)
        self.siglip.eval()
        self.dinov2.eval()
        self.t5_encoder.eval()
        return self

    def _extract_siglip_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from the SigLIP backbone and return ``(B, N, D)``."""
        feats = self.siglip.forward_features(x)
        if isinstance(feats, dict):
            feats = feats.get("x_norm_patchtokens", feats.get("last_hidden_state"))
        return feats

    def _extract_dinov2_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from the DINOv2 backbone, returning ``(B, N, D)`` with the CLS token dropped if present."""
        if self._dinov2_flavor == "hub":
            feats = self.dinov2.forward_features(x)
            if isinstance(feats, dict):
                return feats["x_norm_patchtokens"]
        feats = self.dinov2.forward_features(x)
        expected = self.dinov2_grid * self.dinov2_grid
        if feats.dim() == 3 and feats.size(1) == expected + 1:
            feats = feats[:, 1:]
        return feats

    def _fuse_channel(self, siglip_feats: torch.Tensor, dinov2_feats: torch.Tensor) -> torch.Tensor:
        """Prismatic-style fusion: reshape into 2D grids, interpolate to a shared resolution, channel-concat, 1x1 conv."""
        b = siglip_feats.size(0)
        s_grid, d_grid, t_grid = self.siglip_grid, self.dinov2_grid, self.common_grid
        s = siglip_feats.transpose(1, 2).reshape(b, self.siglip_dim, s_grid, s_grid)
        d = dinov2_feats.transpose(1, 2).reshape(b, self.dinov2_dim, d_grid, d_grid)
        if s_grid != t_grid:
            s = F.interpolate(s, size=(t_grid, t_grid), mode="bilinear", align_corners=False)
        if d_grid != t_grid:
            d = F.interpolate(d, size=(t_grid, t_grid), mode="bilinear", align_corners=False)
        cat = torch.cat([s, d], dim=1)
        fused = self.fusion_proj(cat)
        fused = fused.flatten(2).transpose(1, 2)
        return self.fusion_norm(fused)

    def _encode_language(
        self,
        language_ids: Optional[torch.Tensor],
        language_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return ``(lang_vec, lang_tokens)`` from the T5 encoder; ``(None, None)`` when no language is provided."""
        if language_ids is None:
            return None, None
        language_ids = language_ids.long().to(device)
        if language_mask is None:
            language_mask = torch.ones_like(language_ids, dtype=torch.long)
        language_mask = language_mask.long().to(device)
        with torch.no_grad():
            out = self.t5_encoder(input_ids=language_ids, attention_mask=language_mask)
        tokens = out.last_hidden_state
        mask_f = language_mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        pooled = (tokens * mask_f).sum(dim=1) / denom
        return pooled, tokens

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        language: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        language_ids: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run dual-backbone encoding + FiLM + readout and emit the same token dict as the legacy tokenizer."""
        del masks, language

        if images.ndim == 4:
            images = images.unsqueeze(1)
        b, v, c, h, w = images.shape
        flat_images = images.reshape(b * v, c, h, w)

        siglip_patches = self._extract_siglip_patches(flat_images)
        dinov2_patches = self._extract_dinov2_patches(flat_images)
        vision_per_view = self._fuse_channel(siglip_patches, dinov2_patches)
        n_per_view = vision_per_view.size(1)
        vision_tokens = vision_per_view.reshape(b, v * n_per_view, self.fusion_dim)

        device = vision_tokens.device
        lang_vec, lang_tokens = self._encode_language(language_ids, language_mask, b, device)

        if lang_vec is not None and self.config.use_film_language and self.film_gamma is not None:
            gamma = self.film_gamma(lang_vec).unsqueeze(1)
            beta = self.film_beta(lang_vec).unsqueeze(1)
            vision_tokens = gamma * vision_tokens + beta

        if states.ndim > 2:
            states = states.flatten(1)
        state_tokens = self.state_tokenizer(states).unsqueeze(1)

        if history is None:
            history = states
        if history.ndim > 2:
            history = history.flatten(1)
        history_tokens = self.history_tokenizer(history).unsqueeze(1)

        if lang_tokens is not None:
            language_tokens = self.language_proj(lang_tokens)
        else:
            language_tokens = torch.zeros(b, 1, self.fusion_dim, device=device, dtype=vision_tokens.dtype)

        q = self.readout_norm_q(self.readout_queries.unsqueeze(0).expand(b, -1, -1))
        kv = torch.cat([vision_tokens, language_tokens], dim=1)
        attn_out, _ = self.readout_attn(query=q, key=kv, value=kv)
        readout = self.readout_norm_out(q + attn_out)
        readout = readout + self.readout_ffn(readout)

        readout_mean = readout.mean(dim=1)
        state_mean = state_tokens.squeeze(1)
        gate_in = torch.cat([readout_mean, state_mean], dim=-1)
        gate = torch.sigmoid(self.uncertainty_gate(gate_in))
        fused = gate * readout_mean + (1.0 - gate) * state_mean

        context_tokens = torch.cat([state_tokens, history_tokens, language_tokens, readout], dim=1)
        return {
            "fused": fused,
            "context_tokens": context_tokens,
            "vision_tokens": readout,
            "state_tokens": state_tokens,
            "language_tokens": language_tokens,
            "history_tokens": history_tokens,
            "uncertainty_gate": gate.squeeze(-1),
        }

    def maybe_freeze_vision(self) -> None:
        """Match the legacy interface signature; LoRA initialization already performed the freezing work."""
        self._freeze_non_lora(self.siglip)
        self._freeze_non_lora(self.dinov2)
        for p in self.t5_encoder.parameters():
            p.requires_grad = False


class HierarchicalPlanner(nn.Module):
    """Discrete phase + continuous skill latent planner.

    The flag ``config.use_fsq`` selects between the FSQ discrete-phase encoder
    :class:`FSQSkillEncoder` and the Gumbel-Softmax :class:`SkillVQEncoder`.
    When FSQ is enabled, the phase embedding table reuses the FSQ encoder's
    internal ``code_embed`` (size = ``prod(fsq_levels)``); otherwise an
    ``nn.Embedding(num_skills, ...)`` is used. External output keys stay the
    same: ``(phase_id, phase_logits, phase_embed, skill_latent)``.
    """

    def __init__(self, config: PhaseQFlowConfig) -> None:
        """Initialize discrete phase encoder and continuous skill latent head."""
        super().__init__()
        self.config = config
        self.use_fsq = bool(getattr(config, "use_fsq", False))
        if self.use_fsq:
            self.phase_encoder = FSQSkillEncoder(config.fusion_hidden_dim, config)
            self.phase_embedding = self.phase_encoder.code_embed
        else:
            self.phase_encoder = SkillVQEncoder(
                config.fusion_hidden_dim, config.num_skills, config.gumbel_temperature
            )
            self.phase_embedding = nn.Embedding(config.num_skills, config.skill_embedding_dim)
        self.skill_continuous = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.fusion_hidden_dim, config.continuous_skill_dim),
        )

    def forward(
        self,
        fused_obs: torch.Tensor,
        phase_labels: Optional[torch.Tensor] = None,
        phase_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Infer phase/skill latents under manual, latent, or hybrid supervision modes."""
        mode = phase_mode or self.config.weak_phase_supervision_mode
        inferred_phase_id, _, phase_logits = self.phase_encoder(fused_obs, training=self.training)

        if mode == "manual" and phase_labels is not None and torch.all(phase_labels >= 0):
            phase_id = phase_labels.long()
        else:
            phase_id = inferred_phase_id

        phase_embed = self.phase_embedding(phase_id)
        skill_latent = self.skill_continuous(fused_obs)

        return {
            "phase_id": phase_id,
            "phase_logits": phase_logits,
            "phase_embed": phase_embed,
            "skill_latent": skill_latent,
        }


class FlowActionHeadEuler(nn.Module):
    """[LEGACY] 4-step Euler-integrated flow action head (default before Round 4).

    Kept for ``cfg.flow_type == "euler"`` A/B comparisons or to load older
    checkpoints. Each forward produces a single ``(B, action_dim)`` action.
    New experiments should prefer :class:`ShortcutFlowActionHead` (1-NFE,
    self-consistency training, chunk output).
    """

    def __init__(self, config: PhaseQFlowConfig) -> None:
        """Create conditional flow field and action decoder."""
        super().__init__()
        self.config = config
        cond_dim = config.fusion_hidden_dim + config.skill_embedding_dim + config.continuous_skill_dim
        self.latent_dim = config.latent_dim
        self.conditioner = nn.Linear(cond_dim, config.fusion_hidden_dim)
        self.flow_field = nn.Sequential(
            nn.Linear(config.latent_dim + config.fusion_hidden_dim + 1, config.fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.fusion_hidden_dim, config.latent_dim),
        )
        self.action_decoder = nn.Linear(config.latent_dim, config.action_dim)

    def forward(self, fused_obs: torch.Tensor, phase_embed: torch.Tensor, skill_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Integrate a conditional flow from noise and decode to actions."""
        cond = self.conditioner(torch.cat([fused_obs, phase_embed, skill_latent], dim=-1))
        u = torch.randn(fused_obs.size(0), self.latent_dim, device=fused_obs.device)
        dt = 1.0 / max(self.config.flow_steps, 1)
        for i in range(self.config.flow_steps):
            tau = torch.full((u.size(0), 1), i * dt, device=u.device)
            du = self.flow_field(torch.cat([u, cond, tau], dim=-1))
            u = u + dt * du
        action_pred = self.action_decoder(u)
        return {"latent_action_pred": u, "action_pred": action_pred}


# Backward-compatible alias: old code or checkpoints referring to ``FlowActionHead`` keep working.
FlowActionHead = FlowActionHeadEuler


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal time embedding (Vaswani 2017).

    Takes a 1D scalar tensor ``t`` (shape ``(B,)``, arbitrary range but
    ``[0, 1]`` inside Shortcut) and produces a ``(B, dim)`` sin/cos encoding.
    Used by the Shortcut model to inject ``t`` and the step size ``d`` into
    separate conditioning channels.
    """

    def __init__(self, dim: int) -> None:
        """Cache the embedding dimension; require it to be even."""
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalPosEmb dim must be even, got {dim}")
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return the sinusoidal embedding ``(B, dim)`` of ``t``."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class _DiTBlock(nn.Module):
    """Minimal DiT block with AdaLN-Zero modulation (Peebles & Xie, 2212.09748)."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        """Build the attention + MLP stack and the zero-initialized AdaLN modulation head."""
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply self-attention + MLP with AdaLN-Zero modulation driven by ``cond``."""
        scale1, shift1, gate1, scale2, shift2, gate2 = self.adaln(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1.0 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn_out
        h = self.norm2(x) * (1.0 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.mlp(h)
        return x


class SmallDiT(nn.Module):
    """Compact DiT trunk that applies self-attention + AdaLN modulation to a ``(B, T, token_dim)`` chunk.

    Projects the result to ``(B, T, output_dim)``. Used inside
    :class:`ShortcutFlowActionHead` to model the temporal structure of each
    action token inside a chunk.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        output_dim: int,
        chunk_len: int,
    ) -> None:
        """Wire the input projection, positional embedding, DiT stack, and output head."""
        super().__init__()
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, chunk_len, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([_DiTBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Encode the chunk through the DiT trunk and project to ``output_dim``."""
        h = self.input_proj(x) + self.pos_emb
        for block in self.blocks:
            h = block(h, cond)
        h = self.norm_out(h)
        return self.output_proj(h)


class ShortcutFlowActionHead(nn.Module):
    """Single-network 1-NFE flow-matching action head (Shortcut Model, Frans et al., 2410.12557).

    Key points:
      - Conditions on ``(x_t, t, d, phase_embed, skill_latent, fused_obs)`` where
        ``d`` is the step size. During training, ``d`` is sampled discretely from
        ``{2^0, 2^-1, ..., 2^-shortcut_d_log2_bins}`` together with a flow-matching
        self-consistency constraint (two ``d`` steps should match one ``2d`` step).
      - At inference, fix ``t=0, d=1`` to integrate to the endpoint in a single
        step, giving 1-NFE (Number of Function Evaluations = 1), roughly 4x
        faster than 4-step Euler.
      - Output shape is a ``(B, Ta, Da)`` action chunk; downstream consumers can
        apply ACT temporal ensembling or replan every ``action_execute_size``
        steps.
    """

    def __init__(self, config: PhaseQFlowConfig) -> None:
        """Build the conditioner, t/d embedders, and the DiT action trunk."""
        super().__init__()
        self.config = config
        self.Ta = int(config.action_chunk_size)
        self.Da = int(config.action_dim)
        cond_dim = (
            config.fusion_hidden_dim
            + config.skill_embedding_dim
            + config.continuous_skill_dim
        )
        self.conditioner = nn.Linear(cond_dim, config.fusion_hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(config.fusion_hidden_dim),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.SiLU(),
        )
        self.d_embed = nn.Sequential(
            SinusoidalPosEmb(config.fusion_hidden_dim),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.SiLU(),
        )
        self.action_trunk = SmallDiT(
            token_dim=self.Da,
            hidden_dim=config.fusion_hidden_dim,
            num_layers=max(1, int(config.dit_num_layers)),
            num_heads=int(config.dit_num_heads),
            output_dim=self.Da,
            chunk_len=self.Ta,
        )

    def _velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        d: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the flow-matching velocity field ``v_θ(x_t, t, d, cond)``.

        ``x``: ``(B, Ta, Da)``; ``t``/``d``: ``(B, 1)``; ``cond``: ``(B, H)``.
        """
        t_emb = self.time_embed(t.reshape(-1)) # (B, H)
        d_emb = self.d_embed(d.reshape(-1)) # (B, H)
        full_cond = cond + t_emb + d_emb # (B, H)
        return self.action_trunk(x, full_cond) # (B, Ta, Da)

    def forward(
        self,
        fused_obs: torch.Tensor,
        phase_embed: torch.Tensor,
        skill_latent: torch.Tensor,
        actions_gt: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run training (flow-matching + self-consistency) or inference (1-NFE)."""
        cond = self.conditioner(torch.cat([fused_obs, phase_embed, skill_latent], dim=-1))
        batch_size = cond.shape[0]
        device = cond.device

        if training:
            if actions_gt is None:
                raise ValueError("ShortcutFlowActionHead training=True requires actions_gt")
            if actions_gt.ndim != 3:
                raise ValueError(f"actions_gt must be (B, Ta, Da); got {tuple(actions_gt.shape)}")

            t = torch.rand(batch_size, 1, device=device)
            # d = 2^(-k), k ~ U{0, ..., shortcut_d_log2_bins}
            k_max = int(self.config.shortcut_d_log2_bins)
            k = torch.randint(0, k_max + 1, (batch_size, 1), device=device).float()
            d = torch.pow(2.0, -k)

            noise = torch.randn_like(actions_gt)
            t3 = t.unsqueeze(-1) # broadcast over Ta, Da
            x_t = (1.0 - t3) * noise + t3 * actions_gt
            v_target = actions_gt - noise # constant velocity for linear interpolation.

            v_pred = self._velocity(x_t, t, d, cond)
            fm_loss = F.mse_loss(v_pred, v_target)

            # Self-consistency: two d-steps ≈ one 2d-step (target detached per paper).
            d3 = d.unsqueeze(-1)
            with torch.no_grad():
                x_mid = x_t + d3 * self._velocity(x_t, t, d, cond)
                x_end_two_step = x_mid + d3 * self._velocity(x_mid, t + d, d, cond)
            x_end_one_step = x_t + (2.0 * d3) * self._velocity(x_t, t, 2.0 * d, cond)
            sc_loss = F.mse_loss(x_end_one_step, x_end_two_step.detach())

            # Direct one-shot prediction (used as action_pred for auxiliary MSE / verifier).
            pred_final = x_t + (1.0 - t3) * self._velocity(x_t, t, 1.0 - t, cond)

            return {
                "fm_loss": fm_loss,
                "sc_loss": sc_loss,
                "action_pred": pred_final,
                "v_pred": v_pred,
                "v_target": v_target,
            }

        # Inference: 1-NFE from pure noise.
        x_0 = torch.randn(batch_size, self.Ta, self.Da, device=device)
        t0 = torch.zeros(batch_size, 1, device=device)
        d1 = torch.ones(batch_size, 1, device=device)
        x_1 = x_0 + 1.0 * self._velocity(x_0, t0, d1, cond)
        return {"action_pred": x_1}

    def compute_cond(
        self,
        fused_obs: torch.Tensor,
        phase_embed: torch.Tensor,
        skill_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Return conditioning vector c = conditioner([fused_obs, phase_embed, skill_latent]).

        Used by VelocityCurvatureEstimator to obtain c_t and cache c_{t-1}.
        """
        return self.conditioner(torch.cat([fused_obs, phase_embed, skill_latent], dim=-1))

    def velocity(
        self,
        x_tau: torch.Tensor,
        tau: float,
        cond: torch.Tensor,
        d: float = 1.0,
    ) -> torch.Tensor:
        """Public interface to the velocity field v_θ(x_tau, tau, cond) at a fixed anchor.

        Used by VelocityCurvatureEstimator for the I^(3) cliff estimator.
        x_tau: (B, Ta, Da); tau: scalar in [0,1]; cond: (B, H)
        """
        B = cond.shape[0]
        device = cond.device
        t = torch.full((B, 1), tau, device=device, dtype=cond.dtype)
        d_t = torch.full((B, 1), d, device=device, dtype=cond.dtype)
        return self._velocity(x_tau, t, d_t, cond)


# ============================================================
# DEPRECATED for PACE v2 main path.
# Kept for ablation only. PACE v2 does not rely on online residual
# correction; boundary-step errors are addressed by the concordance
# detector C_t (phase_centric/cliff_detection/) and boundary-aware
# flow loss (Phase C). Closed-loop correction introduces latency
# incompatible with real-time replanning via PCAR.
# To re-enable: set cfg.use_correction_head = True (currently False).
# ============================================================
class A2C2CorrectionHead(nn.Module):
    """A2C2 online correction head (arXiv 2509.23224, "Leave No Observation Behind").

    Purpose: turn the otherwise open-loop execution of a chunk into a closed
    loop. At every control step ``k`` in a rollout we take:
        action_k = (base_chunk + correction(obs_feat_k, base_chunk, k))[k]
    The base chunk, produced by the flow action head (e.g. Shortcut 1-NFE),
    stays fixed; the correction head re-predicts a residual from the latest
    observation to progressively refine the remaining actions in ``base_chunk``.

    Network structure:
      * Input cat ``(B, Ta, D+Da+1)``: broadcast ``obs_feat`` along the chunk
        axis and concat with ``base_chunk`` and a normalized step-in-chunk
        scalar;
      * ``input_proj`` maps to ``correction_hidden_dim`` and adds a learnable
        positional embedding;
      * a sinusoidal step embedding drives AdaLN conditioning inside the DiT
        blocks;
      * the final ``output_proj`` emits a ``(B, Ta, Da)`` residual; both its
        weight and bias are zero-initialized, so early in training
        ``correction`` is ~0 and the head degenerates to the open-loop
        ``base_chunk`` until ``L_correction`` kicks in.

    Default sizes ``(hidden=640, layers=4, heads=8)`` give roughly 30M
    parameters, matching the original paper's ~32M budget.
    """

    def __init__(self, config: PhaseQFlowConfig) -> None:
        """Wire the input projection, DiT blocks, and zero-initialized output projection."""
        super().__init__()
        self.Ta = int(config.action_chunk_size)
        self.Da = int(config.action_dim)
        self.D = int(config.fusion_hidden_dim)
        self.h = int(config.correction_hidden_dim)
        num_layers = int(config.correction_num_layers)
        num_heads = int(config.correction_num_heads)

        in_dim = self.D + self.Da + 1
        self.input_proj = nn.Linear(in_dim, self.h)
        self.pos_emb = nn.Parameter(torch.randn(1, self.Ta, self.h) * 0.02)
        self.step_embed = nn.Sequential(
            SinusoidalPosEmb(self.h),
            nn.Linear(self.h, self.h),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([_DiTBlock(self.h, num_heads) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(self.h)
        self.output_proj = nn.Linear(self.h, self.Da)
        # Identity init: residual=0 at start so training is stable.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        obs_feat: torch.Tensor,
        base_chunk: torch.Tensor,
        step_in_chunk: Any,
    ) -> torch.Tensor:
        """Return corrected chunk ``(B, Ta, Da) = base_chunk + residual``.

        Parameters
        ----------
        obs_feat : (B, D) fused observation feature at the current step.
        base_chunk : (B, Ta, Da) chunk predicted at the last replan.
        step_in_chunk : int or (B,) LongTensor. Position within the base chunk
            (``0`` at replan). Normalized by ``Ta`` before feeding the network.
        """
        if obs_feat.ndim != 2 or obs_feat.shape[-1] != self.D:
            raise ValueError(
                f"obs_feat must be (B, {self.D}); got {tuple(obs_feat.shape)}"
            )
        if base_chunk.ndim != 3 or base_chunk.shape[1] != self.Ta or base_chunk.shape[2] != self.Da:
            raise ValueError(
                f"base_chunk must be (B, {self.Ta}, {self.Da}); got {tuple(base_chunk.shape)}"
            )
        B = obs_feat.shape[0]
        device = obs_feat.device
        dtype = obs_feat.dtype

        if isinstance(step_in_chunk, int):
            step_norm = torch.full(
                (B,), float(step_in_chunk) / max(1, self.Ta), device=device, dtype=dtype
            )
        else:
            step_norm = step_in_chunk.to(device=device, dtype=dtype) / max(1, self.Ta)
            if step_norm.ndim == 0:
                step_norm = step_norm.expand(B)

        step_scalar = step_norm.view(B, 1, 1).expand(B, self.Ta, 1)
        obs_tiled = obs_feat.unsqueeze(1).expand(B, self.Ta, self.D)
        x = torch.cat([obs_tiled, base_chunk, step_scalar], dim=-1) # (B, Ta, D+Da+1)

        h = self.input_proj(x) + self.pos_emb
        cond = self.step_embed(step_norm) # (B, h)
        for block in self.blocks:
            h = block(h, cond)
        h = self.norm_out(h)
        residual = self.output_proj(h) # (B, Ta, Da)
        return base_chunk + residual


class ChunkVerifierMLP(nn.Module):
    """Legacy closed-loop chunk confidence + phase drift estimator (pre-Round 5).

    2 scalar outputs, trained with BCE against all-ones targets. Kept for
    A/B comparison; new default is :class:`IQLChunkVerifier`.
    """

    def __init__(self, config: PhaseQFlowConfig) -> None:
        """Build chunk confidence and phase-drift prediction head."""
        super().__init__()
        in_dim = config.fusion_hidden_dim + config.action_dim + config.skill_embedding_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, config.verifier_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.verifier_hidden_dim, 2),
        )

    def forward(self, fused_obs: torch.Tensor, predicted_action: torch.Tensor, phase_embed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate online execution confidence and replanning flags."""
        out = self.net(torch.cat([fused_obs, predicted_action, phase_embed], dim=-1))
        confidence = torch.sigmoid(out[:, 0])
        phase_drift = torch.sigmoid(out[:, 1])
        return {
            "chunk_confidence": confidence,
            "phase_drift": phase_drift,
            "should_replan": (confidence < 0.5) | (phase_drift > 0.5),
        }


# Backward-compatible alias: old code / old checkpoints referring to
# ``ChunkVerifier`` keep working; new code should use ``ChunkVerifierMLP``
# or the IQL variant below.
ChunkVerifier = ChunkVerifierMLP


# ============================================================
# DEPRECATED for PACE v2 main path.
# Kept for ablation only. PACE v2 does not rely on Q-V advantage
# as the confidence signal; the main path uses the concordance
# detector C_t (see phase_centric/cliff_detection/concordance.py)
# which fuses three Predictability-Cliff estimators without
# requiring offline Q-function training or expectile regression.
# To re-enable: set cfg.use_iql_verifier = True (currently False).
# ============================================================
class IQLChunkVerifier(nn.Module):
    """IQL-style V_ψ / Q_θ chunk-level critic (Kostrikov et al., 2110.06169).

    - ``V(s, z)`` regresses the τ-expectile of ``Q(s, a_chunk, z)`` so that V
      stays within the behavior-policy support (safe on offline BC data).
    - ``Q(s, a_chunk, z)`` learns via TD(0) against ``r + γ · V_target(s')``.
    - ``chunk_confidence = σ(β · (Q - V))`` — a calibrated advantage score
      instead of the previous BCE-on-all-ones stub.

    Inputs accepted at inference / loss time:
      * ``fused_obs`` : ``(B, H)``
      * ``chunk_flat`` : ``(B, Ta * Da)`` (the full predicted chunk)
      * ``phase_embed`` : ``(B, Zp)``

    Auxiliary methods:
      * :meth:`compute_critic_losses` — expectile V loss + TD Q loss.
      * :meth:`soft_update_target` — Polyak-average V into V_target.
    """

    def __init__(self, config: PhaseQFlowConfig) -> None:
        """Build V, Q, and V_target with parameter-matched MLPs."""
        super().__init__()
        self.config = config
        self._chunk_size = int(config.action_chunk_size)
        self._action_dim = int(config.action_dim)
        in_v = config.fusion_hidden_dim + config.skill_embedding_dim
        in_q = config.fusion_hidden_dim + self._chunk_size * self._action_dim + config.skill_embedding_dim
        h = int(config.verifier_hidden_dim)
        self.V = nn.Sequential(
            nn.Linear(in_v, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 1),
        )
        self.Q = nn.Sequential(
            nn.Linear(in_q, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 1),
        )
        # Target V for TD bootstrap; Polyak-averaged via ``soft_update_target``.
        self.V_target = copy.deepcopy(self.V)
        for p in self.V_target.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    def _pad_or_flatten_chunk(self, predicted_action: torch.Tensor) -> torch.Tensor:
        """Flatten ``(B, Ta, Da)`` → ``(B, Ta*Da)``; or expand ``(B, Da)`` → ``(B, Ta*Da)``."""
        if predicted_action.ndim == 3:
            B, T, D = predicted_action.shape
            if T == self._chunk_size and D == self._action_dim:
                return predicted_action.reshape(B, T * D)
            # Pad or truncate to configured Ta × Da.
            target = torch.zeros(B, self._chunk_size, self._action_dim,
                                 device=predicted_action.device,
                                 dtype=predicted_action.dtype)
            t_copy = min(T, self._chunk_size)
            d_copy = min(D, self._action_dim)
            target[:, :t_copy, :d_copy] = predicted_action[:, :t_copy, :d_copy]
            return target.reshape(B, self._chunk_size * self._action_dim)
        # Single-step (B, Da): broadcast across chunk time.
        if predicted_action.ndim == 2:
            B, D = predicted_action.shape
            if D >= self._action_dim:
                single = predicted_action[:, :self._action_dim]
            else:
                single = F.pad(predicted_action, (0, self._action_dim - D))
            return single.unsqueeze(1).expand(-1, self._chunk_size, -1).reshape(B, -1)
        raise ValueError(f"predicted_action must be 2D or 3D; got {predicted_action.shape}")

    def forward(
        self,
        fused_obs: torch.Tensor,
        predicted_action: torch.Tensor,
        phase_embed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return V, Q, advantage, calibrated confidence, and replan flag."""
        chunk_flat = self._pad_or_flatten_chunk(predicted_action)
        sv = torch.cat([fused_obs, phase_embed], dim=-1)
        sq = torch.cat([fused_obs, chunk_flat, phase_embed], dim=-1)
        v = self.V(sv).squeeze(-1)
        q = self.Q(sq).squeeze(-1)
        advantage = q - v
        beta = float(self.config.iql_confidence_beta)
        chunk_confidence = torch.sigmoid(advantage * beta)
        phase_drift = torch.zeros_like(v) # reserved; populated by rollout logic.
        should_replan = chunk_confidence < 0.5
        return {
            "v": v,
            "q": q,
            "advantage": advantage,
            "chunk_confidence": chunk_confidence,
            "phase_drift": phase_drift,
            "should_replan": should_replan,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
        """Asymmetric L2 used for expectile regression (Kostrikov eq. 4)."""
        weight = torch.where(diff > 0, tau, 1.0 - tau)
        return (weight * diff.pow(2)).mean()

    def compute_critic_losses(
        self,
        fused_obs: torch.Tensor,
        chunk_flat: torch.Tensor,
        phase_embed: torch.Tensor,
        reward: torch.Tensor,
        next_fused_obs: torch.Tensor,
        next_phase_embed: torch.Tensor,
        not_done: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loss_v, loss_q)`` for IQL expectile V + TD(0) Q."""
        sv = torch.cat([fused_obs, phase_embed], dim=-1)
        sq = torch.cat([fused_obs, chunk_flat, phase_embed], dim=-1)
        v = self.V(sv).squeeze(-1)
        with torch.no_grad():
            q_ref = self.Q(sq).squeeze(-1)
        loss_v = self.expectile_loss(q_ref - v, float(self.config.iql_expectile_tau))

        q = self.Q(sq).squeeze(-1)
        with torch.no_grad():
            sv_next = torch.cat([next_fused_obs, next_phase_embed], dim=-1)
            v_next = self.V_target(sv_next).squeeze(-1)
            td_target = reward + float(self.config.iql_gamma) * not_done * v_next
        loss_q = F.mse_loss(q, td_target)
        return loss_v, loss_q

    @torch.no_grad()
    def soft_update_target(self) -> None:
        """Polyak-average V into V_target (τ = ``iql_target_tau``)."""
        tau = float(self.config.iql_target_tau)
        for p, p_t in zip(self.V.parameters(), self.V_target.parameters()):
            p_t.mul_(1.0 - tau)
            p_t.add_(tau * p.data)


class DiTBackbone(nn.Module):
    """Transformer context encoder."""

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int) -> None:
        """Construct a lightweight Transformer encoder for context tokens."""
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode token sequences into contextualized representations."""
        return self.encoder(x)


class PhaseQFlowPolicy(nn.Module):
    """Four-module PhaseQFlow policy."""

    config_class = PhaseQFlowConfig

    def __init__(self, config: PhaseQFlowConfig, base_policy: Optional[nn.Module] = None, **_: Any) -> None:
        """Initialize all four policy modules and auxiliary loss networks."""
        super().__init__()
        self.config = config
        self.base_policy = base_policy

        if getattr(config, "use_dual_backbone_vision", True):
            self.vision_tokenizer = DualBackboneVisionTokenizer(config)
        else:
            self.vision_tokenizer = VisionTokenizerLegacy(config)
        self.vision_tokenizer.maybe_freeze_vision()
        self.context_backbone = DiTBackbone(config.fusion_hidden_dim, config.dit_num_layers, config.dit_num_heads)
        self.hierarchical_planner = HierarchicalPlanner(config)
        self.flow_type = str(getattr(config, "flow_type", "shortcut")).lower()
        if self.flow_type == "shortcut":
            base_head_cls = ShortcutFlowActionHead
        elif self.flow_type in ("euler", "legacy"):
            base_head_cls = FlowActionHeadEuler
        else:
            raise ValueError(f"Unknown flow_type={self.flow_type!r}; expected 'shortcut' or 'euler'")

        self._use_pcar_dual_head: bool = (
            bool(getattr(config, "use_pcar", False))
            and bool(getattr(config, "pcar_dual_head", False))
            and not bool(getattr(config, "use_pace_b", False))
        )
        if self._use_pcar_dual_head:
            from .phase_centric.pcar_trigger import DualFlowHead
            self.flow_action_head: nn.Module = DualFlowHead(config, base_head_cls)
        else:
            self.flow_action_head = base_head_cls(config)
        self.verifier_type = str(getattr(config, "verifier_type", "iql")).lower()
        if self.verifier_type == "iql":
            self.chunk_verifier: nn.Module = IQLChunkVerifier(config)
        elif self.verifier_type == "mlp":
            self.chunk_verifier = ChunkVerifierMLP(config)
        else:
            raise ValueError(f"Unknown verifier_type={self.verifier_type!r}; expected 'iql' or 'mlp'")

        self._bid_sampler: Optional[Any] = None
        self._v_history: List[float] = []
        self._last_replan_flag: bool = False
        self._pcar_trigger: Optional[Any] = None
        if bool(getattr(config, "use_bid_sampling", False)) and self.flow_type == "shortcut":
            from .bid_sampler import BIDSampler
            self._bid_sampler = BIDSampler(config)

        self.correction_head: Optional[nn.Module] = None
        self._use_correction_head = bool(
            getattr(config, "use_correction_head", True)
        ) and self.flow_type == "shortcut"
        if self._use_correction_head:
            self.correction_head = A2C2CorrectionHead(config)

        self.chunk_infonce_head: Optional[nn.Module] = None
        if bool(getattr(config, "use_chunk_infonce", False)):
            from .phase_centric.identifiability import ChunkInfoNCEHead
            self.chunk_infonce_head = ChunkInfoNCEHead(config)

        self.phase_posterior: Optional[nn.Module] = None
        if bool(getattr(config, "use_phase_boundary_posterior", False)):
            from .phase_centric.phase_posterior import PhasePosteriorEstimator
            self.phase_posterior = PhasePosteriorEstimator(config)

        self.pace_b_flow_head: Optional[nn.Module] = None
        if bool(getattr(config, "use_pace_b", False)):
            from .phase_centric.pace_b_moe import FlowActionHeadPACE
            self.pace_b_flow_head = FlowActionHeadPACE(config)

        self.timestep_embedding = nn.Embedding(config.max_timestep, config.fusion_hidden_dim)
        critic_in = config.fusion_hidden_dim + config.action_dim
        self.critic_network = nn.Sequential(
            nn.Linear(critic_in, config.critic_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.critic_hidden_dim, config.critic_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.critic_hidden_dim, 1),
        )

        self._ensembler: Optional[Any] = None
        self._rollout_step: int = 0
        self._rollout_chunk: Optional[torch.Tensor] = None
        self._rollout_chunk_start: int = 0
        self._iql_step_counter: int = 0
        if bool(getattr(config, "use_temporal_ensembling", False)) and self.flow_type == "shortcut":
            from .temporal_ensembler import ACTTemporalEnsembler

            self._ensembler = ACTTemporalEnsembler(
                chunk_size=int(config.action_chunk_size),
                decay_m=float(config.ensemble_decay_m),
                buffer_size=int(config.ensemble_buffer_size),
                action_dim=int(config.action_dim),
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> "PhaseQFlowPolicy":
        """Instantiate policy from a serialized configuration location."""
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        _ = args
        return cls(config=config, **kwargs)

    def to_config_dict(self) -> Dict[str, Any]:
        """Return configuration as a plain dictionary."""
        return asdict(self.config)

    def _extract_actions(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract supervision actions from common batch key aliases."""
        for key in ("action", "actions", "target_action"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return value.float()
        return None

    def _extract_inputs(self, batch: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
        """Extract explicit multimodal inputs from batch/obs dictionaries."""
        obs = batch.get("obs", batch)
        images = obs.get("images", obs.get("obs_images", obs.get("observation.images")))
        states = obs.get("states", obs.get("obs_states", obs.get("observation.state")))
        language = obs.get("language", obs.get("task_descriptor"))
        history = obs.get("history")
        masks = obs.get("masks", batch.get("masks"))
        language_ids = obs.get("language_ids", batch.get("language_ids"))
        language_mask = obs.get("language_mask", batch.get("language_mask"))
        if images is None or states is None:
            raise KeyError("Expected explicit multimodal inputs: images, states, language, history, masks")
        return {
            "images": images.float(),
            "states": states.float(),
            "language": None if language is None else language.float(),
            "history": None if history is None else history.float(),
            "masks": None if masks is None else masks.float(),
            "language_ids": None if language_ids is None else language_ids.long(),
            "language_mask": None if language_mask is None else language_mask.long(),
        }

    @staticmethod
    def _collapse_action_for_verifier(action_pred: torch.Tensor) -> torch.Tensor:
        """Reduce an action chunk ``(B, Ta, Da)`` to a single ``(B, Da)`` summary.

        The Shortcut head outputs a chunk, but ``ChunkVerifier`` expects a
        single-step ``(B, Da)``. Take step 0 of the chunk as the
        "imminent action" and pass it to the verifier.
        """
        if action_pred.ndim == 3:
            return action_pred[:, 0, :]
        return action_pred

    def _compute_iql_losses(
        self,
        preds: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute IQL ``(loss_v, loss_q)`` for the chunk-level critic.

        Reward signal:
          * ``iql_reward_type == "phase"``: ``r = 1[phase_id[t] > phase_id[t-1]]``.
            Requires the batch to carry a time axis; otherwise zero (loss → 0).
          * ``iql_reward_type == "imitation"``: ``r = 1 - clamp(MSE(pred, gt), 0, 1)``.
            Always defined, dense, and aligned with BC objective. Default.

        Next-state construction:
          If the batch supplies ``next_obs`` / ``next_phase_embed`` (time-
          expanded dataset), use them. Otherwise shift by one along the batch
          dimension (``[1:] || [0:1]``) — a cheap proxy that lets the TD
          bootstrap at least flow gradients; the expectile V loss itself does
          not depend on next_obs, so the pathway still trains.
        """
        assert isinstance(self.chunk_verifier, IQLChunkVerifier)
        device = actions.device
        fused_obs = preds["encoded_obs"]
        phase_embed = preds["phase_embed"]

        action_pred = preds["action_pred"]
        chunk_flat = self.chunk_verifier._pad_or_flatten_chunk(
            action_pred if action_pred.ndim >= 2 else action_pred.unsqueeze(0)
        )

        # Build reward.
        reward_type = str(getattr(self.config, "iql_reward_type", "imitation")).lower()
        if reward_type == "phase":
            phase_ids = batch.get("phase_id")
            if isinstance(phase_ids, torch.Tensor) and phase_ids.ndim >= 2 and phase_ids.shape[1] >= 2:
                # (B, T) → +1 when phase_id strictly increases along time, else 0; avg over time.
                diffs = (phase_ids[:, 1:] > phase_ids[:, :-1]).float()
                reward = diffs.mean(dim=1).to(device)
            else:
                reward = torch.zeros(fused_obs.shape[0], device=device)
        else: # "imitation"
            per_sample_mse = F.mse_loss(
                preds["action_pred"] if preds["action_pred"].shape == actions.shape else
                preds["action_pred"].reshape(actions.shape),
                actions,
                reduction="none",
            )
            per_sample_mse = per_sample_mse.reshape(per_sample_mse.shape[0], -1).mean(dim=-1)
            reward = (1.0 - per_sample_mse.clamp(0.0, 1.0)).detach()

        # Build next state: prefer batch-provided keys, else shift-by-1.
        next_fused = batch.get("next_fused_obs")
        next_phase = batch.get("next_phase_embed")
        not_done = batch.get("not_done")
        if isinstance(next_fused, torch.Tensor) and isinstance(next_phase, torch.Tensor):
            next_fused_obs_t = next_fused.to(device).float()
            next_phase_embed_t = next_phase.to(device).float()
            nd = torch.ones_like(reward) if not_done is None else not_done.float().to(device)
        else:
            # Shift by one within batch; last row wraps to itself with not_done=0.
            idx_next = torch.arange(fused_obs.shape[0], device=device)
            idx_next = torch.clamp(idx_next + 1, max=fused_obs.shape[0] - 1)
            next_fused_obs_t = fused_obs.detach()[idx_next]
            next_phase_embed_t = phase_embed.detach()[idx_next]
            nd = torch.ones_like(reward)
            nd[-1] = 0.0

        loss_v, loss_q = self.chunk_verifier.compute_critic_losses(
            fused_obs=fused_obs.detach(),
            chunk_flat=chunk_flat.detach(),
            phase_embed=phase_embed.detach(),
            reward=reward,
            next_fused_obs=next_fused_obs_t,
            next_phase_embed=next_phase_embed_t,
            not_done=nd,
        )
        return loss_v, loss_q

    def maybe_soft_update_verifier(self, every_n: int = 100) -> bool:
        """Call ``soft_update_target`` every ``every_n`` steps. Returns True when fired.

        Trainer should call this after each ``optimizer.step()``. No-op unless
        the verifier is IQL.
        """
        self._iql_step_counter += 1
        if not isinstance(self.chunk_verifier, IQLChunkVerifier):
            return False
        if self._iql_step_counter % max(1, int(every_n)) != 0:
            return False
        self.chunk_verifier.soft_update_target()
        return True

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        language: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        phase_labels: Optional[torch.Tensor] = None,
        language_ids: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run four-layer forward pass and return intermediate diagnostics.

        The flow head is always called in inference mode (1-NFE for Shortcut);
        ``compute_loss`` additionally invokes ``flow_action_head`` with
        ``training=True`` and ``actions_gt`` under the shortcut branch to
        recover ``fm_loss`` / ``sc_loss``.
        """
        tok = self.vision_tokenizer(
            images=images,
            states=states,
            language=language,
            history=history,
            masks=masks,
            language_ids=language_ids,
            language_mask=language_mask,
        )
        context = self.context_backbone(tok["context_tokens"])
        fused_obs = context.mean(dim=1) + tok["fused"]

        if timestep is None:
            timestep = torch.zeros(fused_obs.size(0), dtype=torch.long, device=fused_obs.device)
        timestep = torch.clamp(timestep.long(), 0, self.config.max_timestep - 1)
        fused_obs = fused_obs + self.timestep_embedding(timestep)

        plan = self.hierarchical_planner(fused_obs=fused_obs, phase_labels=phase_labels)

        phase_p_hat: Optional[torch.Tensor] = None
        phase_beta: Optional[torch.Tensor] = None
        if self.phase_posterior is not None and "phase_logits" in plan:
            pl = plan["phase_logits"]
            if pl.ndim == 3:
                post = self.phase_posterior.forward_sequence(pl)
            else:
                post = self.phase_posterior.step(pl)
            phase_p_hat = post["p_hat"]
            phase_beta = post["beta"]

        if self.pace_b_flow_head is not None:
            flow = self.pace_b_flow_head(
                fused_obs=fused_obs,
                phase_embed=plan["phase_embed"],
                skill_latent=plan["skill_latent"],
                p_hat=phase_p_hat,
                beta=phase_beta,
            )
        elif self.flow_type == "shortcut":
            flow = self.flow_action_head(
                fused_obs=fused_obs,
                phase_embed=plan["phase_embed"],
                skill_latent=plan["skill_latent"],
                actions_gt=None,
                training=False,
            )
        else:
            flow = self.flow_action_head(
                fused_obs=fused_obs,
                phase_embed=plan["phase_embed"],
                skill_latent=plan["skill_latent"],
            )
        if self.verifier_type == "iql":
            verifier_action = flow["action_pred"]
        else:
            verifier_action = self._collapse_action_for_verifier(flow["action_pred"])
        verify = self.chunk_verifier(
            fused_obs=fused_obs,
            predicted_action=verifier_action,
            phase_embed=plan["phase_embed"],
        )

        preds: Dict[str, torch.Tensor] = {
            **tok,
            **plan,
            **flow,
            **verify,
            "encoded_obs": fused_obs,
        }
        if phase_p_hat is not None:
            preds["phase_p_hat"] = phase_p_hat
        if phase_beta is not None:
            preds["phase_beta"] = phase_beta
            # Cliff-namespace public-facing interface: I_hat_1 = -beta_t
            from .phase_centric.cliff_estimators import compute_I_hat_1
            preds["I_hat_1"] = compute_I_hat_1(phase_beta)
        return preds

    def predict_action(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Predict actions from a training/evaluation batch dictionary."""
        inputs = self._extract_inputs(batch)
        timestep = batch.get("timestep")
        phase_labels = batch.get("phase_id")
        return self.forward(**inputs, timestep=timestep, phase_labels=phase_labels)

    def compute_loss(
        self,
        batch: Dict[str, Any],
        return_dict: bool = False,
    ) -> Any:
        """Compute imitation, flow, phase, verifier, and InfoNCE objectives.

        Notes:
          - Component weights are read from ``cfg.flow_loss_weight /
            smoothness_loss_weight / verifier_loss_weight / phase_loss_weight``
            (arXiv 2209.11379 argues that fixed scalar weights beat elaborate
            multi-task optimization schemes).
          - Inputs ``actions`` are clamped to ``[-3, 3]`` to guard against
            extreme values when normalization fails.
          - NaN/Inf guard: if ``action_pred`` contains NaN/Inf, return
            ``{"loss": 0}`` and log a warning to avoid polluting optimizer
            state with bad gradients.
          - When ``return_dict=True``, return
            ``{"loss": total, "components": {...}}`` for wandb-style logging;
            otherwise return a scalar loss for API compatibility.
        """
        preds = self.predict_action(batch)
        actions = self._extract_actions(batch)
        if actions is None:
            raise KeyError("compute_loss requires action/actions/target_action in batch")

        actions = torch.clamp(actions, -3.0, 3.0)

        chunked_action_out = (
            self.flow_type == "shortcut"
            or (self.pace_b_flow_head is not None and int(self.config.action_chunk_size) > 1)
        )
        if chunked_action_out:
            if actions.ndim == 2:
                actions = actions.unsqueeze(1).expand(
                    -1, int(self.config.action_chunk_size), -1
                ).contiguous()
        else:
            if actions.ndim == 3:
                actions = actions[:, 0, :]

        action_pred = preds["action_pred"]
        if torch.isnan(action_pred).any() or torch.isinf(action_pred).any():
            logger.warning("action_pred has NaN/Inf, skipping batch")
            zero_loss = torch.zeros((), device=actions.device, requires_grad=True)
            self._last_loss_components = {
                "imitation": zero_loss.detach(),
                "flow": zero_loss.detach(),
                "smoothness": zero_loss.detach(),
                "verifier": zero_loss.detach(),
                "iql_v": zero_loss.detach(),
                "iql_q": zero_loss.detach(),
                "phase": zero_loss.detach(),
                "infonce": zero_loss.detach(),
                "correction": zero_loss.detach(),
                "chunk_infonce": zero_loss.detach(),
                "pace_a_fm": zero_loss.detach(),
                "pace_a_entropy_reg": zero_loss.detach(),
                "pace_a_mean_beta": zero_loss.detach(),
                "pace_a_max_beta": zero_loss.detach(),
                "pcar_post": zero_loss.detach(),
            }
            if return_dict:
                return {"loss": zero_loss, "components": self._last_loss_components}
            return zero_loss

        pace_a_entropy_reg = torch.zeros((), device=actions.device)
        pace_a_mean_beta = torch.zeros((), device=actions.device)
        pace_a_max_beta = torch.zeros((), device=actions.device)
        if self.pace_b_flow_head is not None:
            fm_loss = torch.zeros((), device=actions.device)
            sc_loss = torch.zeros((), device=actions.device)
            flow_loss = F.mse_loss(action_pred, actions)
            action_pred_for_mse = action_pred
        elif self.flow_type == "shortcut":
            pcar_extra_kwargs: Dict[str, Any] = {}
            if self._use_pcar_dual_head and preds["phase_embed"].shape[0] > 1:
                pcar_extra_kwargs["next_phase_embed"] = preds["phase_embed"].roll(1, dims=0)
            flow_train_out = self.flow_action_head(
                fused_obs=preds["encoded_obs"],
                phase_embed=preds["phase_embed"],
                skill_latent=preds["skill_latent"],
                actions_gt=actions,
                training=True,
                **pcar_extra_kwargs,
            )
            fm_loss = flow_train_out["fm_loss"]
            sc_loss = flow_train_out["sc_loss"]
            sc_w = float(self.config.shortcut_self_consistency_weight)

            if (
                bool(getattr(self.config, "use_pace_a", False))
                and "phase_beta" in preds
                and "v_pred" in flow_train_out
                and "v_target" in flow_train_out
            ):
                from .phase_centric.pace_a_loss import compute_pace_a_flow_loss

                beta_for_loss = preds["phase_beta"]
                if bool(getattr(self.config, "pace_a_detach_beta", True)):
                    beta_for_loss = beta_for_loss.detach()

                pace_a_out = compute_pace_a_flow_loss(
                    v_pred=flow_train_out["v_pred"],
                    v_target=flow_train_out["v_target"],
                    beta_t=beta_for_loss,
                    lambda_weight=float(getattr(self.config, "pace_a_lambda", 2.0)),
                    entropy_weight=float(getattr(self.config, "pace_a_entropy_weight", 0.01)),
                    ablation_mode=str(getattr(self.config, "pace_a_ablation_mode", "full")),
                )
                fm_loss = pace_a_out["fm_loss"]
                pace_a_entropy_reg = pace_a_out["entropy_reg"]
                pace_a_mean_beta = pace_a_out["mean_beta"]
                pace_a_max_beta = pace_a_out["max_beta"]
                flow_loss = fm_loss + sc_w * sc_loss + pace_a_entropy_reg
            else:
                flow_loss = fm_loss + sc_w * sc_loss
            action_pred_for_mse = flow_train_out["action_pred"]
        else:
            fm_loss = torch.zeros((), device=actions.device)
            sc_loss = torch.zeros((), device=actions.device)
            flow_loss = F.mse_loss(action_pred, actions)
            action_pred_for_mse = action_pred

        pcar_post_loss = torch.zeros((), device=actions.device)
        if (
            self._use_pcar_dual_head
            and self.flow_type == "shortcut"
            and "post_fm_loss" in flow_train_out
        ):
            sc_w_local = float(self.config.shortcut_self_consistency_weight)
            pcar_post_loss = flow_train_out["post_fm_loss"]
            if "post_sc_loss" in flow_train_out:
                pcar_post_loss = pcar_post_loss + sc_w_local * flow_train_out["post_sc_loss"]

        if action_pred_for_mse.ndim == 3 and action_pred_for_mse.shape[1] > 1:
            smoothness = (
                action_pred_for_mse[:, 1:] - action_pred_for_mse[:, :-1]
            ).pow(2).mean()
        elif action_pred_for_mse.ndim == 2 and action_pred_for_mse.shape[0] > 1:
            smoothness = (action_pred_for_mse[1:] - action_pred_for_mse[:-1]).pow(2).mean()
        else:
            smoothness = torch.zeros((), device=actions.device)

        actions_for_critic = actions[:, 0, :] if actions.ndim == 3 else actions
        obs_feat = preds["encoded_obs"].detach()
        q_values = self.critic_network(
            torch.cat([obs_feat, actions_for_critic.detach()], dim=-1)
        ).squeeze(-1)
        per_sample_raw = F.mse_loss(action_pred_for_mse, actions, reduction="none")
        per_sample = per_sample_raw.reshape(per_sample_raw.shape[0], -1).mean(dim=-1)

        if self.config.use_value_guided_weight:
            weights = torch.softmax(self.config.value_weight_beta * q_values, dim=0)
            weights = weights * weights.numel()
            imitation_loss = (per_sample * weights.detach()).mean()
        else:
            imitation_loss = per_sample.mean()

        if self.verifier_type == "iql":
            iql_v_loss, iql_q_loss = self._compute_iql_losses(
                preds=preds,
                actions=actions,
                batch=batch,
            )
            verifier_loss = iql_v_loss + iql_q_loss
        else:
            iql_v_loss = torch.zeros((), device=actions.device)
            iql_q_loss = torch.zeros((), device=actions.device)
            verifier_targets = torch.ones_like(preds["chunk_confidence"])
            verifier_loss = F.binary_cross_entropy(preds["chunk_confidence"], verifier_targets)

        correction_loss = torch.zeros((), device=actions.device)
        if (
            self.correction_head is not None
            and str(getattr(self.config, "stage", "train_flow")).lower() == "finetune_closedloop"
            and action_pred_for_mse.ndim == 3
            and action_pred_for_mse.shape == actions.shape
        ):
            B = action_pred_for_mse.shape[0]
            Ta = action_pred_for_mse.shape[1]
            steps = torch.randint(0, Ta, (B,), device=actions.device)
            corrected = self.correction_head(
                obs_feat=preds["encoded_obs"],
                base_chunk=action_pred_for_mse.detach(),
                step_in_chunk=steps,
            )
            correction_loss = F.mse_loss(corrected, actions)

        phase_labels = batch.get("phase_id")
        if isinstance(phase_labels, torch.Tensor) and torch.all(phase_labels >= 0):
            phase_loss = F.cross_entropy(preds["phase_logits"], phase_labels.long())
        else:
            phase_loss = torch.zeros((), device=actions.device)

        if bool(getattr(self.config, "use_infonce_phase_aux", False)):
            phase_embed = preds["phase_embed"]
            chunk_len = int(getattr(self.config, "infonce_chunk_len", 4))
            batch_size = phase_embed.shape[0]
            if chunk_len >= 2 and batch_size >= 2 * chunk_len and (batch_size % chunk_len == 0):
                pseudo_time = phase_embed.view(batch_size // chunk_len, chunk_len, -1)
                infonce_loss = infonce_phase_loss(
                    pseudo_time,
                    temperature=float(self.config.infonce_temperature),
                )
            else:
                infonce_loss = torch.zeros((), device=actions.device)
        else:
            infonce_loss = torch.zeros((), device=actions.device)

        chunk_infonce_val = torch.zeros((), device=actions.device)
        chunk_infonce_diag: Dict[str, float] = {}
        if self.chunk_infonce_head is not None:
            if actions.ndim == 3:
                action_chunk_for_nce = actions
            elif actions.ndim == 2:
                action_chunk_for_nce = actions.unsqueeze(1) # (B, 1, Da)
            else:
                action_chunk_for_nce = None
            if action_chunk_for_nce is not None and "phase_logits" in preds:
                chunk_infonce_val, chunk_infonce_diag = self.chunk_infonce_head(
                    fused_obs=preds["encoded_obs"],
                    action_chunk=action_chunk_for_nce,
                    phase_logits=preds["phase_logits"],
                )

        w_flow = float(getattr(self.config, "flow_loss_weight", 0.5))
        w_smooth = float(getattr(self.config, "smoothness_loss_weight", 0.05))
        w_verifier = float(getattr(self.config, "verifier_loss_weight", 0.1))
        w_phase = float(getattr(self.config, "phase_loss_weight", 0.1))
        w_infonce = float(getattr(self.config, "infonce_loss_weight", 0.0))
        w_correction = float(getattr(self.config, "correction_loss_weight", 0.3))
        w_chunk_infonce = float(getattr(self.config, "chunk_infonce_weight", 0.0))
        w_pcar_post = float(getattr(self.config, "pcar_post_loss_weight", 0.3))

        loss = (
            imitation_loss
            + w_flow * flow_loss
            + w_smooth * smoothness
            + w_verifier * verifier_loss
            + w_phase * phase_loss
            + w_infonce * infonce_loss
            + w_correction * correction_loss
            + w_chunk_infonce * chunk_infonce_val
            + w_pcar_post * pcar_post_loss
        )

        self._last_loss_components = {
            "imitation": imitation_loss.detach(),
            "flow": flow_loss.detach(),
            "smoothness": smoothness.detach() if isinstance(smoothness, torch.Tensor) else torch.tensor(float(smoothness)),
            "verifier": verifier_loss.detach(),
            "iql_v": iql_v_loss.detach(),
            "iql_q": iql_q_loss.detach(),
            "phase": phase_loss.detach(),
            "infonce": infonce_loss.detach(),
            "correction": correction_loss.detach(),
            "chunk_infonce": chunk_infonce_val.detach(),
            "pace_a_fm": fm_loss.detach() if isinstance(fm_loss, torch.Tensor) else torch.tensor(float(fm_loss)),
            "pace_a_entropy_reg": pace_a_entropy_reg.detach() if isinstance(pace_a_entropy_reg, torch.Tensor) else torch.tensor(float(pace_a_entropy_reg)),
            "pace_a_mean_beta": pace_a_mean_beta.detach() if isinstance(pace_a_mean_beta, torch.Tensor) else torch.tensor(float(pace_a_mean_beta)),
            "pace_a_max_beta": pace_a_max_beta.detach() if isinstance(pace_a_max_beta, torch.Tensor) else torch.tensor(float(pace_a_max_beta)),
            "pcar_post": pcar_post_loss.detach() if isinstance(pcar_post_loss, torch.Tensor) else torch.tensor(float(pcar_post_loss)),
        }
        self._last_chunk_infonce_diag = chunk_infonce_diag

        base_loss = None
        if self.base_policy is not None and hasattr(self.base_policy, "compute_loss"):
            base_out = self.base_policy.compute_loss(batch) # type: ignore[attr-defined]
            if isinstance(base_out, dict) and "loss" in base_out:
                base_loss = base_out["loss"]
            elif isinstance(base_out, torch.Tensor):
                base_loss = base_out
        if base_loss is not None:
            if base_loss.ndim > 0:
                base_loss = base_loss.mean()
            loss = loss + self.config.base_loss_weight * base_loss

        if return_dict:
            return {"loss": loss, "components": self._last_loss_components}
        return loss

    def update_critic(self, obs_feat: torch.Tensor, actions: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        """Update critic via MSE regression against target Q values."""
        critic_in = torch.cat([obs_feat, actions], dim=-1)
        pred_q = self.critic_network(critic_in).squeeze(-1)
        return F.mse_loss(pred_q, target_q)

    def reset(self) -> None:
        """Clear rollout caches between episodes (Round 4 + Round 5 state).

        Must be called before starting a new episode so that the action chunk
        cache, ACT temporal ensembler, IQL V history, and BID sampler state do
        not leak across episodes.
        """
        self._rollout_step = 0
        self._rollout_chunk = None
        self._rollout_chunk_start = 0
        self._v_history = []
        self._last_replan_flag = False
        if self._ensembler is not None:
            self._ensembler.reset()
        if self._bid_sampler is not None:
            self._bid_sampler.reset()
        if self.phase_posterior is not None:
            self.phase_posterior.reset(batch_size=1)
        if self.pace_b_flow_head is not None and hasattr(self.pace_b_flow_head, "reset_switching"):
            self.pace_b_flow_head.reset_switching(batch_size=1)
        if self._pcar_trigger is not None:
            self._pcar_trigger.reset()
        if self._use_pcar_dual_head and hasattr(self.flow_action_head, "reset"):
            try:
                self.flow_action_head.reset(batch_size=1)
            except TypeError:
                pass

    # ------------------------------------------------------------------
    def _batchify_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Add a leading batch dim where needed and return dict restricted to policy inputs."""
        allowed = {"images", "states", "language", "history", "masks",
                   "language_ids", "language_mask"}
        batched: Dict[str, Any] = {}
        for key, value in obs.items():
            if key not in allowed:
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                if key == "images" and value.ndim == 3:
                    batched[key] = value.unsqueeze(0).unsqueeze(0)
                elif key == "images" and value.ndim == 4:
                    batched[key] = value.unsqueeze(0)
                elif value.ndim == 1 and key != "language_ids":
                    batched[key] = value.unsqueeze(0)
                else:
                    batched[key] = value
            else:
                batched[key] = value

        device = next(self.parameters()).device
        for k, v in batched.items():
            if isinstance(v, torch.Tensor) and v.device != device:
                batched[k] = v.to(device)
        return batched

    def _compute_obs_feat(self, batched: Dict[str, Any]) -> torch.Tensor:
        """Cheap observation featurizer: tokenizer + context + timestep embed.

        Used by :meth:`select_action` when only the obs feature is needed (e.g.
        to drive the A2C2 correction head mid-chunk without re-running the flow
        head). Returns shape ``(B, fusion_hidden_dim)``.
        """
        tok = self.vision_tokenizer(
            images=batched["images"],
            states=batched["states"],
            language=batched.get("language"),
            history=batched.get("history"),
            masks=batched.get("masks"),
            language_ids=batched.get("language_ids"),
            language_mask=batched.get("language_mask"),
        )
        context = self.context_backbone(tok["context_tokens"])
        fused_obs = context.mean(dim=1) + tok["fused"]
        timestep = torch.zeros(fused_obs.size(0), dtype=torch.long, device=fused_obs.device)
        return fused_obs + self.timestep_embedding(timestep)

    def _sample_bid_candidates(self, batched: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Sample ``N`` Shortcut chunks and evaluate IQL advantage per candidate.

        Returns dict with:
          * ``chunks`` : (N, Ta, Da)
          * ``advantages`` : (N,)
          * ``v_mean`` : scalar Tensor (mean V across candidates — rollout baseline)
          * ``chunk_var`` : scalar Tensor (var of candidate chunks across N)
        """
        N = max(1, int(self.config.bid_num_samples))
        # One tokenizer forward is enough; only noise (inside Shortcut) varies.
        tok = self.vision_tokenizer(
            images=batched["images"],
            states=batched["states"],
            language=batched.get("language"),
            history=batched.get("history"),
            masks=batched.get("masks"),
            language_ids=batched.get("language_ids"),
            language_mask=batched.get("language_mask"),
        )
        context = self.context_backbone(tok["context_tokens"])
        fused_obs = context.mean(dim=1) + tok["fused"]
        timestep = torch.zeros(fused_obs.size(0), dtype=torch.long, device=fused_obs.device)
        fused_obs = fused_obs + self.timestep_embedding(timestep)
        plan = self.hierarchical_planner(fused_obs=fused_obs, phase_labels=None)

        chunks: List[torch.Tensor] = []
        for _ in range(N):
            flow = self.flow_action_head(
                fused_obs=fused_obs,
                phase_embed=plan["phase_embed"],
                skill_latent=plan["skill_latent"],
                actions_gt=None,
                training=False,
            )
            chunks.append(flow["action_pred"][0]) # (Ta, Da)
        chunks_t = torch.stack(chunks, dim=0) # (N, Ta, Da)
        chunk_var = chunks_t.var(dim=0).mean() # scalar

        # IQL advantage for each candidate. Skip if verifier isn't IQL.
        if isinstance(self.chunk_verifier, IQLChunkVerifier):
            fused_rep = fused_obs.expand(N, -1)
            phase_rep = plan["phase_embed"].expand(N, -1)
            verify = self.chunk_verifier(
                fused_obs=fused_rep,
                predicted_action=chunks_t,
                phase_embed=phase_rep,
            )
            advantages = verify["advantage"].detach()
            v_mean = verify["v"].mean().detach()
        else:
            advantages = torch.zeros(N, device=chunks_t.device)
            v_mean = torch.zeros((), device=chunks_t.device)

        return {
            "chunks": chunks_t,
            "advantages": advantages,
            "v_mean": v_mean,
            "chunk_var": chunk_var,
        }

    def _check_should_replan(self, v_mean: torch.Tensor, chunk_var: torch.Tensor) -> bool:
        """Combine V-drop / ensemble-variance triggers into a replan decision."""
        v_drop_thr = float(self.config.replan_v_drop_threshold)
        var_thr = float(self.config.replan_ensemble_var_threshold)

        v_drop = 0.0
        if self._v_history:
            baseline = float(np.mean(self._v_history[-5:]))
            v_drop = baseline - float(v_mean.item())
        var_val = float(chunk_var.item())
        return (v_drop > v_drop_thr) or (var_val > var_thr)

    @torch.no_grad()
    def select_action(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Select a single action for online execution.

        Round 4 (ACT temporal ensembling) + Round 5 (BID sampling) both live
        here. Priority order:
          1. ``flow_type != shortcut`` → legacy single-step Euler path.
          2. ``use_bid_sampling`` → sample N candidates, BID-select.
          3. ``use_temporal_ensembling`` → predict chunk, ACT-ensemble.
          4. Fallback → predict a chunk every ``action_execute_size`` steps.
        """
        was_training = self.training
        self.eval()

        try:
            batched = self._batchify_obs(obs)
            device = next(self.parameters()).device

            # ------- Legacy Euler path -------
            if self.flow_type != "shortcut":
                out = self.forward(**batched)
                pred = out["action_pred"]
                if pred.ndim == 3:
                    pred = pred[:, 0, :]
                self._rollout_step += 1
                return pred.squeeze(0).detach()

            # ------- BID sampling path (Round 5) -------
            if self._bid_sampler is not None and bool(self.config.use_bid_sampling):
                samp = self._sample_bid_candidates(batched)
                chunks = samp["chunks"]
                best_chunk = self._bid_sampler.select(chunks, aux_scores=samp["advantages"])

                # Bookkeeping for replan trigger decisions.
                self._last_replan_flag = self._check_should_replan(samp["v_mean"], samp["chunk_var"])
                self._v_history.append(float(samp["v_mean"].item()))
                if len(self._v_history) > 32:
                    self._v_history = self._v_history[-32:]

                # If we have an ensembler, feed the BID-selected chunk through it
                # for extra smoothing; otherwise execute the chunk step-by-step.
                if self._ensembler is not None:
                    chunk_np = best_chunk.detach().to("cpu").float().numpy()
                    action_np = self._ensembler.update_and_get(chunk_np)
                    self._rollout_step += 1
                    return torch.from_numpy(action_np).to(device)

                self._rollout_chunk = best_chunk.unsqueeze(0) # (1, Ta, Da)
                self._rollout_chunk_start = self._rollout_step
                action = best_chunk[0]
                self._rollout_step += 1
                return action.detach()

            # ------- ACT temporal ensembling path -------
            if self._ensembler is not None:
                out = self.forward(**batched)
                chunk = out["action_pred"] # (1, Ta, Da)
                chunk_np = chunk.squeeze(0).detach().to("cpu").float().numpy()
                action_np = self._ensembler.update_and_get(chunk_np)
                self._rollout_step += 1
                return torch.from_numpy(action_np).to(device)

            # ------- Fallback: cache-replay path -------
            execute = max(1, int(self.config.action_execute_size))
            chunk_size = int(self.config.action_chunk_size)
            need_new_chunk = (
                self._rollout_chunk is None
                or (self._rollout_step - self._rollout_chunk_start) >= execute
                or (self._rollout_step - self._rollout_chunk_start) >= chunk_size
            )
            pcar_triggered = False
            out: Optional[Dict[str, torch.Tensor]] = None
            if bool(getattr(self.config, "use_pcar", False)):
                if self._pcar_trigger is None:
                    from .phase_centric.pcar_trigger import PCARTrigger
                    self._pcar_trigger = PCARTrigger(self.config)
                out = self.forward(**batched)
                beta_t = out.get("phase_beta")
                if isinstance(beta_t, torch.Tensor):
                    pcar_triggered = bool(
                        self._pcar_trigger.update_and_check(float(beta_t.mean().item()))
                    )
                    self._last_replan_flag = pcar_triggered
                if pcar_triggered:
                    need_new_chunk = True
            if need_new_chunk:
                if out is None:
                    out = self.forward(**batched)
                self._rollout_chunk = out["action_pred"].detach()
                self._rollout_chunk_start = self._rollout_step
            offset = self._rollout_step - self._rollout_chunk_start
            offset = min(offset, chunk_size - 1)

            if self.correction_head is not None and self._rollout_chunk is not None:
                if need_new_chunk:
                    obs_feat = out["encoded_obs"].detach() # type: ignore[possibly-undefined]
                else:
                    obs_feat = self._compute_obs_feat(batched).detach()
                corrected = self.correction_head(
                    obs_feat=obs_feat,
                    base_chunk=self._rollout_chunk,
                    step_in_chunk=offset,
                )
                action = corrected[0, offset, :]
            else:
                action = self._rollout_chunk[0, offset, :]
            self._rollout_step += 1
            return action
        finally:
            if was_training:
                self.train()

    @property
    def last_replan_flag(self) -> bool:
        """Whether the most recent :meth:`select_action` call would have triggered a replan."""
        return bool(getattr(self, "_last_replan_flag", False))
