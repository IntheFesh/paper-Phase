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

Implementation status
---------------------
* :func:`compute_I_hat_1` — IMPLEMENTED (wraps existing ``phase_beta`` / β_t).
* :func:`compute_I_hat_2` — IMPLEMENTED (action variance estimator).
* :func:`compute_I_hat_3` — IMPLEMENTED (velocity curvature estimator).
* :func:`compute_concordance_C` — IMPLEMENTED (rank-window fusion).

Internal names (Round-4 vintage) are kept as implementation details inside
``phase_posterior.py`` and ``modeling_phaseqflow.py``; only the cliff-namespace
names defined here are the public-facing interface.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Sequence

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
# I_hat_2 — action-variance cliff estimator  (IMPLEMENTED)
# ---------------------------------------------------------------------------

def compute_I_hat_2(
    action_samples: torch.Tensor,
) -> torch.Tensor:
    r"""Cliff estimator :math:`\hat I^{(2)}(t) = -\sigma_t^2`.

    Computes the negative mean per-sample squared deviation from the batch
    mean action:

    .. math::

        \hat I^{(2)}(t) = -\frac{1}{N}\sum_{i=1}^{N}\|a_t^{(i)} - \bar a_t\|^2

    A high (near-zero) value indicates that all ``N`` sampled actions agree
    (low variance, high predictability).  A low value indicates spread action
    predictions, which is a reliable signal of an impending phase boundary.

    Parameters
    ----------
    action_samples : Tensor
        Shape ``(N, B, Ta, Da)`` — ``N`` stochastic samples of the action
        chunk, batch size ``B``, chunk length ``Ta``, action dim ``Da``.
        Requires ``N ≥ 2``.

    Returns
    -------
    Tensor
        Shape ``(B,)``, values in ``(-∞, 0]``.  Zero means all samples agree
        exactly; more negative means higher variance.

    Raises
    ------
    ValueError
        If ``action_samples`` is not 4-D or ``N < 2``.
    """
    if action_samples.ndim != 4:
        raise ValueError(
            f"action_samples must be 4-D (N, B, Ta, Da); got shape {tuple(action_samples.shape)}"
        )
    N = action_samples.shape[0]
    if N < 2:
        raise ValueError(f"Need N≥2 action samples for variance estimate; got N={N}")
    mean_action = action_samples.mean(dim=0, keepdim=True)  # (1, B, Ta, Da)
    sq_dev = (action_samples - mean_action).pow(2).sum(dim=(-2, -1))  # (N, B)
    sigma2 = sq_dev.mean(dim=0)  # (B,)
    return -sigma2


# ---------------------------------------------------------------------------
# I_hat_3 — velocity-difference cliff estimator  (IMPLEMENTED)
# ---------------------------------------------------------------------------

def compute_I_hat_3(
    v_theta_ct: torch.Tensor,
    v_theta_ct_prev: torch.Tensor,
) -> torch.Tensor:
    r"""Cliff estimator :math:`\hat I^{(3)}(t)`.

    Measures the squared L2 norm of the velocity-field change when the
    conditioning vector shifts from :math:`c_{t-1}` to :math:`c_t` at a
    fixed anchor :math:`(x_\tau, \tau)`:

    .. math::

        \hat I^{(3)}(t) = -\|v_\theta(x_\tau,\tau,c_t)
                             - v_\theta(x_\tau,\tau,c_{t-1})\|_2^2

    A high (near-zero) value means the flow field barely changed between
    consecutive steps — stable phase.  A large negative value indicates a
    sharp velocity shift, i.e. a cliff.

    Parameters
    ----------
    v_theta_ct : Tensor
        Velocity at current step.  Shape ``(B, Ta, Da)`` or ``(B, D)``.
    v_theta_ct_prev : Tensor
        Velocity at previous step.  Same shape as ``v_theta_ct``.

    Returns
    -------
    Tensor
        Shape ``(B,)``, values in ``(-∞, 0]``.

    Raises
    ------
    ValueError
        If the two tensors have different shapes.
    """
    if v_theta_ct.shape != v_theta_ct_prev.shape:
        raise ValueError(
            f"v_theta_ct and v_theta_ct_prev must have the same shape; "
            f"got {tuple(v_theta_ct.shape)} vs {tuple(v_theta_ct_prev.shape)}"
        )
    diff = v_theta_ct - v_theta_ct_prev
    # Flatten all dims except batch, then sum of squares per sample.
    sq_norm = diff.reshape(diff.shape[0], -1).pow(2).sum(dim=-1)  # (B,)
    return -sq_norm


# ---------------------------------------------------------------------------
# Concordance C_t  (IMPLEMENTED)
# ---------------------------------------------------------------------------

class _RollingRankBuffer:
    """Maintain a deque of scalar values and return percentile rank of the latest entry."""

    def __init__(self, window_size: int) -> None:
        self._buf: Deque[float] = deque(maxlen=window_size)

    def push_and_rank(self, value: float) -> float:
        """Add ``value`` and return its percentile rank in [0, 1] within the window."""
        self._buf.append(value)
        n = len(self._buf)
        if n == 1:
            return 0.5
        rank = sum(1 for x in self._buf if x <= value)
        return rank / n

    def reset(self) -> None:
        self._buf.clear()


def compute_concordance_C(
    i_hat_values: Sequence[torch.Tensor],
    window_size: int = 50,
    _state: Optional[dict] = None,
) -> torch.Tensor:
    r"""Concordance detector :math:`C_t` — rank-window fusion of cliff signals.

    .. math::

        C_t = \frac{1}{K}\sum_{k=1}^{K}\mathrm{rank}_W(\hat I^{(k)}(t))

    where :math:`\mathrm{rank}_W(x)` is the percentile rank of scalar ``x``
    within a sliding window of the most recent ``window_size`` values of that
    estimator.  The result lies in ``[0, 1]``: low values signal concordant
    cliff evidence across all ``K`` estimators.

    Each call is stateless when ``_state`` is ``None`` — a fresh ranking
    window is created.  For online use across timesteps, pass a dict and
    reuse it across calls::

        state = {}
        for step in range(T):
            C_t = compute_concordance_C([i1[step], i2[step], i3[step]],
                                         window_size=50, _state=state)

    Parameters
    ----------
    i_hat_values : sequence of Tensor
        Each tensor has shape ``(B,)``.  The sequence length ``K`` must be
        ≥ 1; typically ``K = 3``.
    window_size : int
        Rolling window depth for percentile ranking (default 50).
    _state : dict, optional
        Mutable dict for persisting :class:`_RollingRankBuffer` objects
        across timesteps.  Keyed by estimator index.

    Returns
    -------
    Tensor
        Shape ``(B,)``, values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``i_hat_values`` is empty or tensors have inconsistent batch sizes.
    """
    if len(i_hat_values) == 0:
        raise ValueError("i_hat_values must contain at least one tensor")
    B = i_hat_values[0].shape[0]
    for idx, v in enumerate(i_hat_values):
        if v.shape[0] != B:
            raise ValueError(
                f"Inconsistent batch size: i_hat_values[0] has B={B}, "
                f"i_hat_values[{idx}] has B={v.shape[0]}"
            )

    if _state is None:
        _state = {}

    K = len(i_hat_values)
    device = i_hat_values[0].device
    dtype = i_hat_values[0].dtype
    rank_sum = torch.zeros(B, device=device, dtype=dtype)

    for k, i_hat in enumerate(i_hat_values):
        if k not in _state:
            _state[k] = [_RollingRankBuffer(window_size) for _ in range(B)]
        buffers = _state[k]
        for b in range(B):
            val = float(i_hat[b].detach().cpu())
            rank_sum[b] = rank_sum[b] + buffers[b].push_and_rank(val)

    return rank_sum / K


__all__ = [
    "compute_I_hat_1",
    "compute_I_hat_2",
    "compute_I_hat_3",
    "compute_concordance_C",
    "_RollingRankBuffer",
]
