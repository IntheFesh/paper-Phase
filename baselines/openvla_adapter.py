"""OpenVLA policy adapter for the universality experiment.

Uses the public ``openvla/openvla-7b`` checkpoint (or a fine-tuned variant)
via HuggingFace ``transformers`` + ``timm``.

Checkpoint availability
-----------------------
The base OpenVLA-7b checkpoint was not fine-tuned on LIBERO-Long.  Whether to:
  (a) use the base checkpoint zero-shot, or
  (b) fine-tune on LIBERO-Long before evaluation,
is deferred to human decision [PHD-4] (see MIGRATION_NOTES.md).

When the checkpoint is not available, :meth:`is_available` returns False and
the universality script will skip this adapter and log to MIGRATION_NOTES.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ._base_adapter import PolicyAdapter, RolloutResult


_DEFAULT_CHECKPOINT = "openvla/openvla-7b"


class OpenVLAAdapter(PolicyAdapter):
    """Adapter for OpenVLA-7B via HuggingFace ``transformers``.

    Parameters
    ----------
    checkpoint_path : str
        HuggingFace model-id or local directory.  Defaults to
        ``"openvla/openvla-7b"``.
    action_chunk_size : int
        Number of actions to execute per VLA query.
    """

    def __init__(
        self,
        checkpoint_path: str = _DEFAULT_CHECKPOINT,
        action_chunk_size: int = 1,
    ) -> None:
        super().__init__("openvla", checkpoint_path=checkpoint_path)
        self.action_chunk_size = action_chunk_size
        self._model = None
        self._processor = None

    def is_available(self) -> bool:
        """True iff ``transformers``, ``timm``, and the checkpoint are reachable."""
        try:
            import transformers  # noqa: F401
            import timm  # noqa: F401
        except ImportError:
            return False
        # Local directory check (skip HF hub network call during dry run)
        if os.path.isdir(str(self.checkpoint_path)):
            return True
        # For HF hub IDs, assume available only if cache present
        from pathlib import Path
        hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        model_id = str(self.checkpoint_path).replace("/", "--")
        return any((hf_cache / "hub").glob(f"models--{model_id}*"))

    def load(self) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        import torch

        self._processor = AutoProcessor.from_pretrained(
            self.checkpoint_path, trust_remote_code=True
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.checkpoint_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).cuda()
        self._model.eval()
        self._loaded = True

    def rollout(self, env: Any, n_steps: int, seed: int = 0) -> RolloutResult:
        """Run one episode; env must expose ``reset(seed)``, ``step(action)``,
        and ``get_image()`` / ``get_instruction()``."""
        if not self._loaded:
            raise RuntimeError("call load() first")

        import torch
        import numpy as np

        obs = env.reset(seed=seed)
        actions = []
        failure_step = None
        success = False

        for t in range(n_steps):
            image = env.get_image()
            instruction = env.get_instruction()
            inputs = self._processor(
                instruction, image, return_tensors="pt"
            ).to("cuda", torch.bfloat16)
            with torch.no_grad():
                action = self._model.predict_action(
                    **inputs, unnorm_key="libero_long", do_sample=False
                )
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
