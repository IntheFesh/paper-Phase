"""Diagnostic logger for PACE v2 training runs.

Records per-step training metrics (loss components, gradient norms,
phase-encoder statistics, PCAR signals, GPU memory, step time) to a
``training_dynamics.csv`` file at a configurable cadence (default every
200 steps). The file is consumed by
:mod:`scripts.utils.diagnostic_report` to produce the post-run
``diagnostic_report.md`` and figures.

Design
------
- **Pure stdlib at import time**. Heavy deps (torch, numpy, matplotlib)
  are imported lazily inside the methods that need them, so importing
  this module never triggers a CUDA init nor breaks CPU-only tests.
- **No core-model coupling**. The logger is fed values via
  :meth:`record`; it never reaches into the model or optimizer. The
  training loop is responsible for collecting and passing in metrics.
- **Append-only CSV**. Simple and robust against crashes; partial
  records are still useful.

Typical usage
-------------
::

    from lerobot_policy_phaseqflow.utils import DiagnosticLogger

    logger = DiagnosticLogger(output_dir, log_every_n_steps=200)
    for step, batch in enumerate(loader):
        loss_dict = ...
        loss_dict["loss_total"].backward()
        if logger.should_log(step):
            logger.record(
                step=step,
                losses=loss_dict,
                grad_norms=collect_grad_norms(model),
                phase_stats=collect_phase_stats(model),
                pcar_stats=collect_pcar_stats(model),
                lr=optimizer.param_groups[0]["lr"],
                gpu_memory_gb=torch.cuda.memory_allocated() / 1e9,
                step_time_sec=step_time,
            )
        optimizer.step()
    logger.close()
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


__all__ = ["DiagnosticLogger", "DiagnosticRecord"]


@dataclass
class DiagnosticRecord:
    """One row of the ``training_dynamics.csv`` file.

    Every field is a scalar (``float`` or ``int``). Missing values are
    written as ``"nan"`` so the file always has a fixed schema, which
    keeps downstream pandas / matplotlib code simple.
    """

    step: int = 0

    # ---- losses ----------------------------------------------------
    loss_total: float = float("nan")
    loss_imitation: float = float("nan")
    loss_flow_policy: float = float("nan")
    loss_infonce_macro: float = float("nan")
    loss_infonce_micro: float = float("nan")

    # ---- gradient norms (per submodule) ----------------------------
    grad_norm_total: float = float("nan")
    grad_norm_vision: float = float("nan")
    grad_norm_phase_encoder: float = float("nan")
    grad_norm_dit: float = float("nan")

    # ---- PACE-A boundary signal ------------------------------------
    pace_a_mean_beta: float = float("nan")
    pace_a_max_beta: float = float("nan")
    pace_a_boundary_density: float = float("nan")

    # ---- phase posterior entropy -----------------------------------
    phase_posterior_entropy_macro: float = float("nan")
    phase_posterior_entropy_micro: float = float("nan")

    # ---- PCAR -------------------------------------------------------
    pcar_trigger_rate: float = float("nan")
    pcar_mean_concordance: float = float("nan")

    # ---- system -----------------------------------------------------
    learning_rate: float = float("nan")
    gpu_memory_gb: float = float("nan")
    step_time_sec: float = float("nan")

    @classmethod
    def schema(cls) -> list[str]:
        """Return CSV column order — used both for the header and the
        per-row dict-to-row conversion."""
        return [f.name for f in fields(cls)]

    def to_row(self) -> list[str]:
        """Serialize to a list of strings matching :meth:`schema`.

        ``nan`` floats are written as the literal string ``"nan"`` so
        that ``pandas.read_csv`` parses them as NaN out of the box.
        """
        out: list[str] = []
        for name in self.schema():
            val = getattr(self, name)
            if isinstance(val, float) and val != val:  # NaN
                out.append("nan")
            else:
                out.append(str(val))
        return out


class DiagnosticLogger:
    """Append-only CSV logger of training-loop metrics.

    Parameters
    ----------
    output_dir
        Directory in which ``training_dynamics.csv`` will be written.
        Created if it does not exist.
    log_every_n_steps
        Cadence at which :meth:`should_log` returns ``True``. Default 200.
    csv_filename
        Override the output filename (default ``training_dynamics.csv``).
    """

    DEFAULT_FILENAME = "training_dynamics.csv"

    def __init__(
        self,
        output_dir: os.PathLike[str] | str,
        log_every_n_steps: int = 200,
        csv_filename: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.csv_path = self.output_dir / (csv_filename or self.DEFAULT_FILENAME)
        self._n_records = 0
        self._init_file()

    # -- file management -------------------------------------------------

    def _init_file(self) -> None:
        """Write a fresh CSV header (truncates any prior file)."""
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(DiagnosticRecord.schema())

    # -- public API ------------------------------------------------------

    def should_log(self, step: int) -> bool:
        """``True`` iff this step is a multiple of ``log_every_n_steps``.

        Step 0 always logs so that the very first row captures the
        initial state of the network before any optimization.
        """
        if step <= 0:
            return True
        return (step % self.log_every_n_steps) == 0

    def record(
        self,
        step: int,
        *,
        losses: Optional[Mapping[str, Any]] = None,
        grad_norms: Optional[Mapping[str, Any]] = None,
        phase_stats: Optional[Mapping[str, Any]] = None,
        pcar_stats: Optional[Mapping[str, Any]] = None,
        lr: Optional[float] = None,
        gpu_memory_gb: Optional[float] = None,
        step_time_sec: Optional[float] = None,
    ) -> DiagnosticRecord:
        """Build a :class:`DiagnosticRecord` from the supplied dicts and
        append it to the CSV.

        Each ``Mapping`` is read with ``.get`` so missing keys silently
        become NaN; this keeps the call-site code small while still
        producing a fixed-schema CSV.
        """
        rec = DiagnosticRecord(step=int(step))

        if losses:
            rec.loss_total = _asfloat(losses.get("loss_total"))
            rec.loss_imitation = _asfloat(losses.get("loss_imitation"))
            rec.loss_flow_policy = _asfloat(losses.get("loss_flow_policy"))
            rec.loss_infonce_macro = _asfloat(losses.get("loss_infonce_macro"))
            rec.loss_infonce_micro = _asfloat(losses.get("loss_infonce_micro"))

        if grad_norms:
            rec.grad_norm_total = _asfloat(grad_norms.get("total"))
            rec.grad_norm_vision = _asfloat(grad_norms.get("vision"))
            rec.grad_norm_phase_encoder = _asfloat(grad_norms.get("phase_encoder"))
            rec.grad_norm_dit = _asfloat(grad_norms.get("dit"))

        if phase_stats:
            rec.pace_a_mean_beta = _asfloat(phase_stats.get("pace_a_mean_beta"))
            rec.pace_a_max_beta = _asfloat(phase_stats.get("pace_a_max_beta"))
            rec.pace_a_boundary_density = _asfloat(phase_stats.get("pace_a_boundary_density"))
            rec.phase_posterior_entropy_macro = _asfloat(phase_stats.get("entropy_macro"))
            rec.phase_posterior_entropy_micro = _asfloat(phase_stats.get("entropy_micro"))

        if pcar_stats:
            rec.pcar_trigger_rate = _asfloat(pcar_stats.get("trigger_rate"))
            rec.pcar_mean_concordance = _asfloat(pcar_stats.get("mean_concordance"))

        if lr is not None:
            rec.learning_rate = _asfloat(lr)
        if gpu_memory_gb is not None:
            rec.gpu_memory_gb = _asfloat(gpu_memory_gb)
        if step_time_sec is not None:
            rec.step_time_sec = _asfloat(step_time_sec)

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(rec.to_row())
        self._n_records += 1
        return rec

    def close(self) -> None:
        """No-op kept for parity with stream-based loggers; included so
        callers can use a ``try/finally`` pattern unconditionally."""
        return None

    @property
    def n_records(self) -> int:
        """Number of rows written so far (excludes header)."""
        return self._n_records


def _asfloat(value: Any) -> float:
    """Best-effort scalar coercion.

    Accepts Python numbers, 0-d numpy arrays, and 0-d torch tensors.
    Returns ``float('nan')`` for ``None`` or any value that cannot be
    converted (we never raise, since logger faults must not crash a
    long training run).
    """
    if value is None:
        return float("nan")
    try:
        # torch.Tensor, np.ndarray both expose .item() for 0-d
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
