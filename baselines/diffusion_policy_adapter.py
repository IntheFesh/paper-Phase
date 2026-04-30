"""Diffusion Policy adapter for the universality experiment.

Uses the LeRobot diffusion policy implementation
(``lerobot.common.policies.diffusion``).

Checkpoint availability
-----------------------
LeRobot ships pre-trained Diffusion Policy checkpoints for several tasks.
For LIBERO-Long we need a checkpoint trained on that benchmark.  The closest
publicly available checkpoint is:
  - ``lerobot/diffusion_pusht``
This targets PushT, not LIBERO-Long.  The decision of which checkpoint to use
(or whether to train from scratch) is deferred to human decision [PHD-7]
(see MIGRATION_NOTES.md).

As an alternative, the robomimic framework also ships Diffusion Policy and may
have LIBERO-compatible checkpoints.  [PHD-7b] tracks this option.

When the checkpoint is not available, :meth:`is_available` returns False.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ._base_adapter import PolicyAdapter, RolloutResult


_DEFAULT_CHECKPOINT = "lerobot/diffusion_pusht"


class DiffusionPolicyAdapter(PolicyAdapter):
    """Adapter for LeRobot Diffusion Policy.

    Parameters
    ----------
    checkpoint_path : str
        HuggingFace model-id or local directory.
    n_inference_steps : int
        DDPM denoising steps at inference.  Default 10.
    device : str
        Torch device string.
    """

    def __init__(
        self,
        checkpoint_path: str = _DEFAULT_CHECKPOINT,
        n_inference_steps: int = 10,
        device: str = "cuda",
    ) -> None:
        super().__init__("diffusion_policy", checkpoint_path=checkpoint_path)
        self.n_inference_steps = n_inference_steps
        self.device = device
        self._policy = None

    def is_available(self) -> bool:
        try:
            from lerobot.common.policies.diffusion.modeling_diffusion import (
                DiffusionPolicy,
            )  # noqa: F401
        except ImportError:
            return False
        from pathlib import Path
        hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        model_id = str(self.checkpoint_path).replace("/", "--")
        return any((hf_cache / "hub").glob(f"models--{model_id}*"))

    def load(self) -> None:
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        import torch

        self._policy = DiffusionPolicy.from_pretrained(self.checkpoint_path)
        self._policy = self._policy.to(self.device)
        self._policy.eval()
        self._loaded = True

    def rollout(self, env: Any, n_steps: int, seed: int = 0) -> RolloutResult:
        if not self._loaded:
            raise RuntimeError("call load() first")

        import torch
        import numpy as np

        obs = env.reset(seed=seed)
        actions: list = []
        failure_step = None
        success = False

        for t in range(n_steps):
            obs_tensor = env.get_obs_tensor().to(self.device)
            with torch.no_grad():
                action = self._policy.select_action(obs_tensor)
            action_np = action.cpu().numpy().ravel()
            obs, reward, done, info = env.step(action_np)
            actions.append(action_np)
            if done:
                success = bool(info.get("success", False))
                if not success:
                    failure_step = t
                break

        # Diffusion Policy is stochastic, so we compute variance across N
        # forward passes at the first observation as a richer cliff proxy.
        cliff_steps = self.cliff_steps_from_actions(actions)
        return RolloutResult(
            trajectory_len=len(actions),
            success=success,
            failure_step=failure_step,
            cliff_steps=cliff_steps,
            action_seq=actions,
            extra={"n_inference_steps": self.n_inference_steps},
        )
