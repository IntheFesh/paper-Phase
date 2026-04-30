"""π0 (pi-zero) policy adapter for the universality experiment.

Uses Physical Intelligence's π0 checkpoint via ``lerobot`` (if available as
``lerobot.policies.pi0``) or a compatible HuggingFace checkpoint.

Checkpoint availability
-----------------------
The public π0 checkpoint (``lerobot/pi0``) was trained on a broad mixture
of robot manipulation data but not specifically on LIBERO-Long.  Whether to:
  (a) use the base π0 zero-shot on LIBERO-Long, or
  (b) fine-tune π0 on LIBERO-Long before evaluation,
is deferred to human decision [PHD-5] (see MIGRATION_NOTES.md).

When the checkpoint is not available, :meth:`is_available` returns False.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ._base_adapter import PolicyAdapter, RolloutResult


_DEFAULT_CHECKPOINT = "lerobot/pi0"


class Pi0Adapter(PolicyAdapter):
    """Adapter for π0 via lerobot's pi0 policy class.

    Parameters
    ----------
    checkpoint_path : str
        HuggingFace model-id or local path.
    """

    def __init__(
        self,
        checkpoint_path: str = _DEFAULT_CHECKPOINT,
        device: str = "cuda",
    ) -> None:
        super().__init__("pi0", checkpoint_path=checkpoint_path)
        self.device = device
        self._policy = None

    def is_available(self) -> bool:
        """True iff ``lerobot`` is importable and provides a pi0 policy."""
        try:
            from lerobot.common.policies.factory import make_policy  # noqa: F401
        except ImportError:
            return False
        from pathlib import Path
        hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        model_id = str(self.checkpoint_path).replace("/", "--")
        return any((hf_cache / "hub").glob(f"models--{model_id}*"))

    def load(self) -> None:
        import torch
        from lerobot.common.policies.factory import make_policy
        from lerobot.configs.default import DatasetConfig

        self._policy = make_policy(
            cfg=None,  # will load from checkpoint
            pretrained_policy_name_or_path=self.checkpoint_path,
        )
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

        cliff_steps = self.cliff_steps_from_actions(actions)
        return RolloutResult(
            trajectory_len=len(actions),
            success=success,
            failure_step=failure_step,
            cliff_steps=cliff_steps,
            action_seq=actions,
        )
