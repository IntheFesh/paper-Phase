"""Configuration for the PhaseQFlow++ policy.

The defaults target single-GPU experimentation (RTX 5070 12 GB, micro_batch=2,
grad_accum=32 ~ effective_bs=64) and scale cleanly to 8-GPU cloud training.
All Phase-Centric ``use_*`` flags default to False; flipping them on wires in
the corresponding innovation without touching the rest of the policy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
)


@PreTrainedConfig.register_subclass("phaseqflow")
@dataclass
class PhaseQFlowConfig(PreTrainedConfig):
    """Config for four-layer PhaseQFlow++ architecture.

    The defaults remain lightweight for single-GPU experimentation.
    """

    num_phases: int = 4
    phase_embedding_dim: int = 32

    vision_token_dim: int = 256
    state_token_dim: int = 128
    language_token_dim: int = 128
    history_token_dim: int = 128
    fusion_hidden_dim: int = 256
    freeze_vision_encoder: bool = True
    use_vision_adapter: bool = True

    cross_attn_heads: int = 8
    cross_attn_dropout: float = 0.0

    num_skills: int = 16
    use_vq_phase: bool = True
    skill_embedding_dim: int = 32
    gumbel_temperature: float = 1.0
    continuous_skill_dim: int = 32
    weak_phase_supervision_mode: str = "hybrid"

    use_value_guided_weight: bool = True
    value_weight_beta: float = 2.0
    critic_hidden_dim: int = 256

    latent_dim: int = 32
    use_latent_flow: bool = True
    flow_steps: int = 4

    verifier_hidden_dim: int = 128
    replan_confidence_threshold: float = 0.5

    action_dim: int = 16
    max_timestep: int = 2048
    base_loss_weight: float = 0.25

    backbone_type: str = "dit"
    dit_hidden_dim: int = 256
    dit_num_layers: int = 4
    dit_num_heads: int = 8

    action_buffer_maxlen: int = 128
    observation_horizon: Optional[int] = None
    action_horizon: Optional[int] = None

    vision_backbone_siglip: str = "vit_base_patch16_siglip_224"
    vision_backbone_dinov2: str = "dinov2_vits14"
    vision_image_size: int = 224
    use_film_language: bool = True
    language_encoder_name: str = "google-t5/t5-base"
    language_token_max_len: int = 16
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_dora: bool = True
    num_camera_views: int = 2
    num_readout_tokens: int = 8
    use_dual_backbone_vision: bool = True
    state_dim: int = 8
    history_dim: int = 8

    fsq_levels: List[int] = field(default_factory=lambda: [8, 6, 5])
    fsq_dim: int = 3
    use_fsq: bool = True
    infonce_temperature: float = 0.1
    infonce_loss_weight: float = 0.1
    use_infonce_phase_aux: bool = True
    infonce_chunk_len: int = 4

    use_bf16: bool = True
    use_gradient_checkpointing: bool = True
    use_paged_adamw_8bit: bool = True
    adam_eps: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    lr_backbone: float = 0.0
    lr_lora: float = 5e-5
    lr_head: float = 1e-4
    lr_warmup_steps: int = 500
    lr_scheduler: str = "cosine"
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_power: float = 0.75
    ema_update_after_step: int = 0
    action_quantile_low: float = 0.01
    action_quantile_high: float = 0.99
    use_action_quantile_norm: bool = True
    state_noise_snr_db: float = 40.0

    stage: str = "train_flow"
    stage_freeze_vision: bool = True
    stage_freeze_planner: bool = False
    stage_freeze_flow_head: bool = False
    stage_freeze_verifier: bool = False

    flow_loss_weight: float = 0.5
    smoothness_loss_weight: float = 0.05
    verifier_loss_weight: float = 0.1
    phase_loss_weight: float = 0.1

    flow_type: str = "shortcut"
    shortcut_d_log2_bins: int = 8
    shortcut_self_consistency_weight: float = 0.5
    action_chunk_size: int = 16
    action_execute_size: int = 8
    use_temporal_ensembling: bool = True
    ensemble_decay_m: float = 0.05
    ensemble_buffer_size: int = 16

    verifier_type: str = "iql"
    use_iql_verifier: bool = False  # PACE v2: deprecated to ablation only; existing verifier_type controls wiring
    iql_expectile_tau: float = 0.8
    iql_gamma: float = 0.99
    iql_confidence_beta: float = 3.0
    iql_target_tau: float = 0.005
    iql_reward_type: str = "imitation"

    use_bid_sampling: bool = False  # PACE v2: deprecated to ablation only
    bid_num_samples: int = 5
    bid_weak_policy_ema_decay: float = 0.99
    bid_backward_weight: float = 0.5
    bid_forward_weight: float = 0.5

    replan_v_drop_threshold: float = 0.3
    replan_ensemble_var_threshold: float = 0.1

    use_correction_head: bool = False  # PACE v2: deprecated to ablation only
    correction_hidden_dim: int = 640
    correction_num_layers: int = 4
    correction_num_heads: int = 8
    correction_loss_weight: float = 0.3

    use_chunk_infonce: bool = False
    chunk_infonce_weight: float = 0.5
    chunk_infonce_temperature: float = 0.1
    chunk_infonce_chunk_len: int = 8

    phase_posterior_smooth_alpha: float = 0.3
    use_phase_boundary_posterior: bool = False

    use_pace_a: bool = False
    pace_a_lambda: float = 2.0
    pace_a_entropy_weight: float = 0.01
    pace_a_detach_beta: bool = True
    pace_a_ablation_mode: str = "full"

    use_pace_b: bool = False
    moe_num_experts: int = 4
    moe_expert_hidden_dim: int = 128
    moe_switch_kappa: float = 5.0
    moe_switch_mu: float = 2.0
    moe_top_k: int = 2

    use_pace_c: bool = False
    curriculum_stage_steps: Tuple[int, int, int] = (1000, 3000, 10000)
    curriculum_max_boundaries_stage1: int = 1
    curriculum_max_boundaries_stage2: int = 3

    # Phase C: hierarchical FSQ encoder + boundary-aware flow loss
    phase_mode: str = "flat"  # "flat" (default/ablation) or "hierarchical" (Phase C)
    fsq_levels_macro: List[int] = field(default_factory=lambda: [5, 4])  # K1=20
    fsq_levels_micro: List[int] = field(default_factory=lambda: [6, 5])  # K2=30
    use_boundary_reweight: bool = True   # enable w(β^micro) weighting; ablation=False
    boundary_reweight_lambda: float = 0.5  # λ_β in w(β) = 1 + λ*β

    use_pcar: bool = False
    pcar_input_signal: str = "concordance"  # "concordance" (PACE v2) or "beta" (legacy)
    pcar_change_threshold: float = 0.4
    pcar_trigger_budget_eps: float = 0.1
    pcar_dual_head: bool = False
    pcar_post_head_ratio: float = 0.5
    pcar_post_loss_weight: float = 0.3

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "PhaseQFlowConfig":
        """Load config from a directory or JSON file path."""
        path = Path(pretrained_model_name_or_path)
        if path.is_dir():
            path = path / "config.json"
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "phaseqflow" in payload and isinstance(payload["phaseqflow"], dict):
            payload = payload["phaseqflow"]
        valid = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def save_pretrained(self, save_directory: str) -> str:
        """Save config to `config.json` in the target directory."""
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        out_file = path / "config.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        return str(out_file)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain nested dict."""
        return asdict(self)

    @property
    def observation_delta_indices(self) -> Optional[List[int]]:
        """Observation-delta indices required by the LeRobot policy API (unused here)."""
        return None

    @property
    def action_delta_indices(self) -> Optional[List[int]]:
        """Action-delta indices: one entry per step in the action chunk."""
        return list(range(self.action_chunk_size))

    @property
    def reward_delta_indices(self) -> Optional[List[int]]:
        """Reward-delta indices required by the LeRobot policy API (unused here)."""
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        """Return the LeRobot-compatible AdamW optimizer preset."""
        return AdamWConfig(
            lr=self.lr_head,
            weight_decay=self.weight_decay,
            grad_clip_norm=self.grad_clip_norm,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps,
        )

    def get_scheduler_preset(self) -> Optional[LRSchedulerConfig]:
        """Return the cosine-with-warmup scheduler preset."""
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.lr_warmup_steps,
            num_decay_steps=100_000,
            peak_lr=self.lr_head,
            decay_lr=self.lr_head * 0.1,
        )

    def validate_features(self) -> None:
        """No-op feature validation hook required by the LeRobot policy API."""
        return None
