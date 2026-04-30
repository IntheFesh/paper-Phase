"""Baseline policy adapters for the Predictability Cliff universality experiment.

Each module wraps one VLA / IL policy behind the :class:`PolicyAdapter` ABC
so that experiment scripts in ``scripts/phenomenon/`` can evaluate cliff
universality without policy-specific logic leaking into the analysis code.

Adapters
--------
- :mod:`openvla_adapter`        — OpenVLA-7B (HuggingFace transformers)
- :mod:`pi0_adapter`            — π0 (Physical Intelligence / lerobot)
- :mod:`bc_act_adapter`         — BC-ACT (lerobot ACTPolicy)
- :mod:`diffusion_policy_adapter` — Diffusion Policy (lerobot)

Human decision log
------------------
[PHD-4] OpenVLA checkpoint for LIBERO-Long — see MIGRATION_NOTES.md
[PHD-5] π0 checkpoint for LIBERO-Long     — see MIGRATION_NOTES.md
[PHD-6] BC-ACT checkpoint for LIBERO-Long — see MIGRATION_NOTES.md
[PHD-7] Diffusion Policy checkpoint       — see MIGRATION_NOTES.md
"""

from __future__ import annotations

from ._base_adapter import PolicyAdapter, RolloutResult
from .bc_act_adapter import BCActAdapter
from .diffusion_policy_adapter import DiffusionPolicyAdapter
from .openvla_adapter import OpenVLAAdapter
from .pi0_adapter import Pi0Adapter

__all__ = [
    "PolicyAdapter",
    "RolloutResult",
    "OpenVLAAdapter",
    "Pi0Adapter",
    "BCActAdapter",
    "DiffusionPolicyAdapter",
]
