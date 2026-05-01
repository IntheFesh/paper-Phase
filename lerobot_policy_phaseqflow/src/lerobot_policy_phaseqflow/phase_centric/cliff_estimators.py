"""Predictability-Cliff estimators — public-facing cliff-namespace interfaces.

Terminology
-----------
"Predictability Cliff" (cliff) is the core theoretical concept: the moment in a
long-horizon task when the policy's predictability drops sharply (a phase
boundary). Three estimators quantify this drop, all following the convention that
*higher value = more predictable*, so a cliff corresponds to a local minimum.

Estimator definitions (locked in master plan §0.1, do not alter):

.. math::

    \\hat I^{(1)}(t) \\propto -\\beta_t
        = -(1 - \\sum_k \\sqrt{\\hat p_t(k)\\,\\hat p_{t-1}(k)})

    \\hat I^{(2)}(t) \\propto -\\sigma_t^2
        = -\\frac{1}{N}\\sum_{i=1}^{N}\\|a_t^{(i)} - \\bar a_t\\|^2

    \\hat I^{(3)}(t) \\propto
        -\\|v_\\theta(x_\\tau, \\tau, c_t) - v_\\theta(x_\\tau, \\tau, c_{t-1})\\|_2^2

Concordance detector (rank-based fusion):

.. math::

    C_t = \\frac{1}{3}[\\mathrm{rank}_W(\\hat I^{(1)}(t))
                      + \\mathrm{rank}_W(\\hat I^{(2)}(t))
                      + \\mathrm{rank}_W(\\hat I^{(3)}(t))]

where :math:`\\mathrm{rank}_W` is the percentile rank within a rolling window of
size ``W``.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# I_hat_1 — Bhattacharyya cliff estimator
# ---------------------------------------------------------------------------

def compute_I_hat_1(phase_beta: torch.Tensor) -> torch.Tensor:
    r"""Cliff estimator :math:`\hat I^{(1)}(t) = -\beta_t`.

    Parameters
    ----------
    phase_beta : Tensor
        Shape ``(B,)`` or broadcastable.  Values in ``[0, 1]``.

    Returns
    -------
    Tensor
        Same shape as ``phase_beta``, values in ``[-1, 0]``.
    """
    return -phase_beta


# ---------------------------------------------------------------------------
# I_hat_2 — action-variance cliff estimator
# ---------------------------------------------------------------------------

def compute_I_hat_2(action_samples: torch.Tensor) -> torch.Tensor:
    r"""Cliff estimator :math:`\hat I^{(2)}(t) = -\sigma_t^2`.

    Computes the negative mean squared deviation of ``N`` action samples around
    their batch mean.  Higher magnitude (more negative) means higher action
    variance, i.e. lower predictability — a cliff signal.

    Parameters
    ----------
    action_samples : Tensor
        Shape ``(N, B, Ta, Da)`` where:
        * ``N`` — number of independent samples (≥ 2),
        * ``B`` — batch size,
        * ``Ta`` — action horizon,
        * ``Da`` — action dimension.

    Returns
    -------
    Tensor
        Shape ``(B,)``, values ≤ 0.

    Raises
    ------
    ValueError
        If ``action_samples.ndim != 4`` or ``N < 2``.
    """
    if action_samples.ndim != 4:
        raise ValueError(
            f"action_samples must be 4-D (N, B, Ta, Da), got shape {action_samples.shape}"
        )
    N = action_samples.shape[0]
    if N < 2:
        raise ValueError(f"Need N >= 2 action samples to estimate variance, got N={N}")

    mean_action = action_samples.mean(dim=0, keepdim=True)          # (1, B, Ta, Da)
    sq_dev = (action_samples - mean_action).pow(2).sum(dim=(-2, -1))  # (N, B)
    return -sq_dev.mean(dim=0)                                        # (B,)


# ---------------------------------------------------------------------------
# I_hat_3 — velocity-curvature cliff estimator
# ---------------------------------------------------------------------------

def compute_I_hat_3(
    v_theta_ct: torch.Tensor,
    v_theta_ct_prev: torch.Tensor,
) -> torch.Tensor:
    r"""Cliff estimator :math:`\hat I^{(3)}(t) = -\|v_\theta(c_t) - v_\theta(c_{t-1})\|^2`.

    Measures how much the flow-matching velocity field changed between consecutive
    condition vectors at a fixed anchor point ``(x_τ, τ)``.  Large change →
    lower predictability → cliff signal.

    Parameters
    ----------
    v_theta_ct : Tensor
        Velocity at current condition, shape ``(B, Ta, Da)``.
    v_theta_ct_prev : Tensor
        Velocity at previous condition, same shape.

    Returns
    -------
    Tensor
        Shape ``(B,)``, values ≤ 0.

    Raises
    ------
    ValueError
        If ``v_theta_ct.shape != v_theta_ct_prev.shape``.
    """
    if v_theta_ct.shape != v_theta_ct_prev.shape:
        raise ValueError(
            f"Shape mismatch: v_theta_ct {v_theta_ct.shape} vs "
            f"v_theta_ct_prev {v_theta_ct_prev.shape}"
        )
    diff = v_theta_ct - v_theta_ct_prev                             # (B, Ta, Da)
    sq_norm = diff.reshape(diff.shape[0], -1).pow(2).sum(dim=-1)    # (B,)
    return -sq_norm


# ---------------------------------------------------------------------------
# Rolling-rank buffer for concordance
# ---------------------------------------------------------------------------

class _RollingRankBuffer:
    """Online percentile-rank tracker over a fixed-length window."""

    def __init__(self, window_size: int) -> None:
        self._buf: deque[float] = deque(maxlen=window_size)

    def push_and_rank(self, value: float) -> float:
        """Push *value* and return its percentile rank in [0, 1]."""
        self._buf.append(value)
        n = len(self._buf)
        if n == 1:
            return 0.5
        return sum(1.0 for x in self._buf if x <= value) / n


# ---------------------------------------------------------------------------
# Concordance C_t — rank-based fusion of all three estimators
# ---------------------------------------------------------------------------

def compute_concordance_C(
    i_hat_values: Sequence[torch.Tensor],
    window_size: int = 50,
    _state: Optional[Dict] = None,
) -> torch.Tensor:
    r"""Concordance detector :math:`C_t \in [0, 1]`.

    :math:`C_t = \frac{1}{K}\sum_{k=1}^{K}\mathrm{rank}_W(\hat I^{(k)}(t))`

    where :math:`\mathrm{rank}_W` is the percentile rank within a rolling window
    of the last ``window_size`` values for estimator ``k`` and batch element
    ``b``.

    Parameters
    ----------
    i_hat_values : sequence of Tensor
        Each tensor has shape ``(B,)``.  Typically the list
        ``[I_hat_1, I_hat_2, I_hat_3]`` but any non-empty subset is accepted.
    window_size : int
        Rolling-window length for percentile ranking (default 50).
    _state : dict or None
        Persistent rank-buffer state across calls.  Pass the same dict on
        every forward step so that the rolling window accumulates correctly.
        If ``None``, a fresh (memoryless) state is used — suitable only for
        unit tests.

    Returns
    -------
    Tensor
        Shape ``(B,)``, values in ``[0, 1]``.  High ≈ cliff (all estimators
        agree on low predictability); low ≈ stable interior.

    Raises
    ------
    ValueError
        If ``i_hat_values`` is empty or tensors have inconsistent batch sizes.
    """
    if len(i_hat_values) == 0:
        raise ValueError("i_hat_values must contain at least one estimator tensor")

    K = len(i_hat_values)
    B = i_hat_values[0].shape[0]
    for k, t in enumerate(i_hat_values):
        if t.shape[0] != B:
            raise ValueError(
                f"Batch-size mismatch: i_hat_values[0] has B={B} but "
                f"i_hat_values[{k}] has B={t.shape[0]}"
            )

    if _state is None:
        _state = {}

    rank_sum = torch.zeros(B, dtype=torch.float32, device=i_hat_values[0].device)

    for k, i_hat in enumerate(i_hat_values):
        if k not in _state:
            _state[k] = [_RollingRankBuffer(window_size) for _ in range(B)]
        buffers: List[_RollingRankBuffer] = _state[k]
        for b in range(B):
            rank_sum[b] += buffers[b].push_and_rank(float(i_hat[b].detach()))

    return rank_sum / K


__all__ = [
    "compute_I_hat_1",
    "compute_I_hat_2",
    "compute_I_hat_3",
    "compute_concordance_C",
    "_RollingRankBuffer",
]
