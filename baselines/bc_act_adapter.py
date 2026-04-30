"""BC-ACT (Action Chunking Transformer) adapter for the universality experiment.

Uses the LeRobot ACT implementation (``lerobot.common.policies.act``).

Checkpoint availability
-----------------------
LeRobot ships several pre-trained ACT checkpoints.  For LIBERO-Long we need
a checkpoint trained on that benchmark specifically.  The closest publicly
available checkpoints are:
  - ``lerobot/act_aloha_sim_transfer_cube_human``
  - ``lerobot/act_pusht_keypoints``
None of these target LIBERO-Long directly.  The decision of which ACT checkpoint
to use (or whether to train one from scratch) is deferred to human decision
[PHD-6] (see MIGRATION_NOTES.md).

When the checkpoint is not available, :meth:`is_available` returns False.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ._base_adapter import PolicyAdapter, RolloutResult


_DEFAULT_CHECKPOINT = "lerobot/act_aloha_sim_transfer_cube_human"


class BCActAdapter(PolicyAdapter):
    """Adapter for LeRobot ACT policy.

    Parameters
    ----------
    checkpoint_path : str
        HuggingFace model-id or local directory.
    chunk_size : int
        Action chunk size H; corresponds to the ACT temporal horizon.
    device : str
        Torch device string.
    """

    def __init__(
        self,
        checkpoint_path: str = _DEFAULT_CHECKPOINT,
        chunk_size: int = 16,
        device: str = "cuda",
    ) -> None:
        super().__init__("bc_act", checkpoint_path=checkpoint_path)
        self.chunk_size = chunk_size
        self.device = device
        self._policy = None

    def is_available(self) -> bool:
        """True iff ACTPolicy is importable and the checkpoint cache exists."""
        try:
            from lerobot.common.policies.act.modeling_act import ACTPolicy  # noqa: F401
        except ImportError:
            return False
        from pathlib import Path
        hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        model_id = str(self.checkpoint_path).replace("/", "--")
        return any((hf_cache / "hub").glob(f"models--{model_id}*"))

    def load(self) -> None:
        """Download (if needed) and load the ACT checkpoint onto self.device."""
        from lerobot.common.policies.act.modeling_act import ACTPolicy
        import torch

        self._policy = ACTPolicy.from_pretrained(self.checkpoint_path)
        self._policy = self._policy.to(self.device)
        self._policy.eval()
        self._loaded = True

    def rollout(self, env: Any, n_steps: int, seed: int = 0) -> RolloutResult:
        """Run one episode with chunked ACT execution; re-queries the policy every chunk_size steps."""
        if not self._loaded:
            raise RuntimeError("call load() first")

        import torch
        import numpy as np

        obs = env.reset(seed=seed)
        actions: list = []
        failure_step = None
        success = False
        action_buffer: list = []

        for t in range(n_steps):
            if not action_buffer:
                obs_tensor = env.get_obs_tensor().to(self.device)
                with torch.no_grad():
                    chunk = self._policy.select_action(obs_tensor)  # (H, Da)
                action_buffer = list(chunk.cpu().numpy())

            action_np = action_buffer.pop(0)
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
            extra={"chunk_size": self.chunk_size},
        )
