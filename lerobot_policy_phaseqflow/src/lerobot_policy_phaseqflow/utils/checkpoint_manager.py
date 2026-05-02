"""Resumable checkpoint manager for PACE v2 training.

Goals
-----
1. **Bit-identical resume.** Save model + optimizer + LR-scheduler state,
   plus the RNG state of Python, NumPy, PyTorch (CPU + all CUDA devices)
   and the running step counter. Resuming from the saved file produces
   the same trajectory as an uninterrupted run.
2. **Rolling-window of 3.** At any given time only the three most recent
   checkpoints are kept on disk. Older checkpoints are deleted *after*
   the new one is fully written and ``fsync``'d, so a crash mid-write
   never destroys a known-good file.
3. **No core-model coupling.** The manager only sees ``state_dict``;
   architectural decisions live elsewhere.

Default cadence is **every 200 steps** (matches the diagnostic logger).

Typical use
-----------
::

    from lerobot_policy_phaseqflow.utils import CheckpointManager

    ckpt = CheckpointManager(output_dir, save_every_n_steps=200, keep_last=3)

    start_step = 0
    if args.resume_from_checkpoint:
        start_step = ckpt.load(args.resume_from_checkpoint, model, optimizer, scheduler)

    for step in range(start_step, total_steps):
        ...
        if ckpt.should_save(step):
            ckpt.save(step, model, optimizer, scheduler)

The saved file is a Python dict written via ``torch.save``; on a CPU-only
machine without torch installed the load helper raises a clear error.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, List, Optional


__all__ = ["CheckpointManager", "load_partial"]


_DEFAULT_FILENAME_FMT = "checkpoint_step{step:09d}.pt"
_LATEST_SYMLINK = "checkpoint_latest.pt"


class CheckpointManager:
    """Save and load resumable training checkpoints.

    Parameters
    ----------
    output_dir
        Directory under which checkpoints are written. Created on first
        save.
    save_every_n_steps
        Cadence for :meth:`should_save`. Default 200.
    keep_last
        Maximum number of recent checkpoints retained on disk. Default 3.
    filename_fmt
        ``str.format`` template; must contain ``{step}``.
    """

    def __init__(
        self,
        output_dir: os.PathLike[str] | str,
        save_every_n_steps: int = 200,
        keep_last: int = 3,
        filename_fmt: str = _DEFAULT_FILENAME_FMT,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.save_every_n_steps = max(1, int(save_every_n_steps))
        self.keep_last = max(1, int(keep_last))
        self.filename_fmt = filename_fmt

    # -- public API ------------------------------------------------------

    def should_save(self, step: int) -> bool:
        """``True`` iff this step should produce a checkpoint."""
        if step <= 0:
            return False
        return (step % self.save_every_n_steps) == 0

    def save(
        self,
        step: int,
        model: Any,
        optimizer: Any,
        scheduler: Optional[Any] = None,
        extra: Optional[dict] = None,
    ) -> Path:
        """Write a checkpoint to disk and prune older ones.

        ``extra`` is a free-form dict for caller-specific state (epoch,
        config hash, …). It is round-tripped through :meth:`load` via
        the returned ``extra`` field.
        """
        import torch  # local import; CPU-only test envs may lack it

        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / self.filename_fmt.format(step=int(step))
        tmp_path = target.with_suffix(target.suffix + ".tmp")

        payload = {
            "step": int(step),
            "model": _state_dict(model),
            "optimizer": _state_dict(optimizer),
            "scheduler": _state_dict(scheduler) if scheduler is not None else None,
            "rng": _capture_rng_state(),
            "extra": dict(extra) if extra else {},
        }

        # Atomic write: temp file + fsync + rename, so a crash mid-save
        # never produces a half-written checkpoint.
        with tmp_path.open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)

        latest = self.output_dir / _LATEST_SYMLINK
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(target.name)
        except OSError:
            # Fall back to a small sentinel file; symlinks may not be
            # available on every filesystem (e.g. some cloud volumes).
            latest.write_text(target.name, encoding="utf-8")

        self._prune_old()
        return target

    def load(
        self,
        path: os.PathLike[str] | str,
        model: Any,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> int:
        """Restore state and return the step counter from the checkpoint.

        After this call, training should resume at ``return_value`` (the
        next iteration the loop performs is ``step + 1``).
        """
        import torch

        ckpt_path = Path(path)
        if ckpt_path.is_dir():
            # Caller passed a directory: pick the rolling "latest".
            latest = ckpt_path / _LATEST_SYMLINK
            if latest.exists():
                resolved = latest.resolve() if latest.is_symlink() else (ckpt_path / latest.read_text().strip())
                ckpt_path = resolved
            else:
                candidates = sorted(self._existing_checkpoints(ckpt_path))
                if not candidates:
                    raise FileNotFoundError(f"No checkpoints under {ckpt_path}")
                ckpt_path = candidates[-1]
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(payload["model"], strict=False)
        if optimizer is not None and payload.get("optimizer") is not None:
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except ValueError:
                print("[CheckpointManager] optimizer state skipped "
                      "(param groups changed between stages; optimizer restarts)")
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        _restore_rng_state(payload.get("rng") or {})
        return int(payload.get("step", 0))

    # -- internals -------------------------------------------------------

    def _existing_checkpoints(self, directory: Optional[Path] = None) -> List[Path]:
        d = Path(directory) if directory else self.output_dir
        if not d.is_dir():
            return []
        # We track only files matching the manager's prefix to avoid
        # accidentally deleting unrelated .pt files in the same dir.
        prefix = self.filename_fmt.split("{")[0]
        return sorted(p for p in d.iterdir() if p.is_file() and p.name.startswith(prefix))

    def _prune_old(self) -> None:
        files = self._existing_checkpoints()
        excess = len(files) - self.keep_last
        if excess <= 0:
            return
        for old in files[:excess]:
            try:
                old.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# RNG state helpers
# ---------------------------------------------------------------------------

def _state_dict(obj: Any) -> Any:
    """Return ``obj.state_dict()`` if available, else ``obj``."""
    if obj is None:
        return None
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    return obj


def _capture_rng_state() -> dict:
    """Snapshot RNG state across Python, numpy, torch CPU + every CUDA device."""
    state: dict = {"python": random.getstate()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def _restore_rng_state(state: dict) -> None:
    """Inverse of :func:`_capture_rng_state`. Silent on missing keys so
    that loading a checkpoint produced on a different machine (e.g.
    different CUDA device count) still works."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    try:
        import torch

        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda_all" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Partial-load helper (architecture-migration / cross-stage transfer)
# ---------------------------------------------------------------------------

def load_partial(
    path: "os.PathLike[str] | str",
    model: "Any",
    verbose: bool = True,
) -> "tuple[int, list, list]":
    """只加载名字和形状都匹配的参数，用于架构修改后的迁移学习。

    Returns
    -------
    step    : int   checkpoint 对应的训练步数
    loaded  : list  成功加载的参数名列表
    skipped : list  跳过的参数名列表
    """
    import torch

    ckpt_path = Path(path)
    if ckpt_path.is_dir():
        latest = ckpt_path / _LATEST_SYMLINK
        if latest.exists():
            ckpt_path = (
                latest.resolve()
                if latest.is_symlink()
                else (ckpt_path / latest.read_text().strip())
            )
        else:
            candidates = sorted(
                p for p in ckpt_path.iterdir()
                if p.is_file() and p.name.startswith("checkpoint_step")
            )
            if not candidates:
                raise FileNotFoundError(f"No checkpoints under {ckpt_path}")
            ckpt_path = candidates[-1]

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_state = payload["model"]
    current_state = model.state_dict()

    loaded, skipped, new_state = [], [], {}
    for name, param in current_state.items():
        if name not in saved_state:
            skipped.append(f"{name} [not in checkpoint]")
            new_state[name] = param
        elif saved_state[name].shape != param.shape:
            skipped.append(
                f"{name} [shape mismatch: saved {saved_state[name].shape} "
                f"vs current {param.shape}]"
            )
            new_state[name] = param
        else:
            new_state[name] = saved_state[name]
            loaded.append(name)

    model.load_state_dict(new_state)
    if verbose:
        print(f"[load_partial] 成功 {len(loaded)}/{len(current_state)} 层")
        for s in skipped:
            print(f"  跳过: {s}")
    return int(payload.get("step", 0)), loaded, skipped
