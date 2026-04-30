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
* :func:`compute_I_hat_1` — implemented (wraps existing ``phase_beta`` / β_t).
* :func:`compute_I_hat_2` — PENDING: see ``MIGRATION_NOTES.md`` §1.
* :func:`compute_I_hat_3` — PENDING: see ``MIGRATION_NOTES.md`` §2.
* :func:`compute_concordance_C` — PENDING: blocked on I_hat_2 and I_hat_3.

Internal names (Round-4 vintage) are kept as implementation details inside
``phase_posterior.py`` and ``modeling_phaseqflow.py``; only the cliff-namespace
names defined here are the public-facing interface.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# I_hat_1 — Bhattacharyya cliff estimator  (IMPLEMENTED)
# ---------------------------------------------------------------------------

def compute_I_hat_1(phase_beta: torch.Tensor) -> torch.Tensor:
    r"""Cliff estimator :math:`\hat I^{(1)}(t) = -\beta_t`.

    :math:`\beta_t \in [0, 1]` is the Bhattacharyya boundary signal produced by
    :class:`~lerobot_policy_phaseqflow.phase_centric.phase_posterior.PhasePosteriorEstimator`.
    Negating it gives a signal that is:

    * high (≈ 0) in stable interior states — high predictability,
    * low (≈ −1) at phase boundaries — low predictability, i.e. a cliff.

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
# I_hat_2 — action-variance cliff estimator  (PENDING)
# ---------------------------------------------------------------------------

def compute_I_hat_2(
    action_samples: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    r"""Cliff estimator :math:`\hat I^{(2)}(t) = -\sigma_t^2`.  **PENDING.**

    Requires ``N ≥ 2`` action samples at each step; computation is blocked
    pending the decisions in ``MIGRATION_NOTES.md`` §1.

    Raises
    ------
    NotImplementedError
        Always, until the pending decisions are resolved.
    """
    raise NotImplementedError(
        "compute_I_hat_2 is not yet implemented. "
        "See MIGRATION_NOTES.md §1 (Pending Human Decisions) for the "
        "multi-sample anchor mechanism that needs to be decided first."
    )


# ---------------------------------------------------------------------------
# I_hat_3 — velocity-difference cliff estimator  (PENDING)
# ---------------------------------------------------------------------------

def compute_I_hat_3(
    v_theta_ct: Optional[torch.Tensor] = None,
    v_theta_ct_prev: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    r"""Cliff estimator :math:`\hat I^{(3)}(t)`.  **PENDING.**

    Requires the flow-matching velocity field evaluated at a fixed anchor
    ``(x_τ, τ)`` under two consecutive condition vectors ``c_t`` and
    ``c_{t-1}``.  Blocked pending the decisions in ``MIGRATION_NOTES.md`` §2.

    Raises
    ------
    NotImplementedError
        Always, until the pending decisions are resolved.
    """
    raise NotImplementedError(
        "compute_I_hat_3 is not yet implemented. "
        "See MIGRATION_NOTES.md §2 (Pending Human Decisions) for anchor "
        "point selection and velocity exposure decisions."
    )


# ---------------------------------------------------------------------------
# Concordance C_t  (PENDING — blocked on I_hat_2 and I_hat_3)
# ---------------------------------------------------------------------------

def compute_concordance_C(
    i_hat_values: Sequence[torch.Tensor],
    window_size: int = 50,
) -> Optional[torch.Tensor]:
    r"""Concordance detector :math:`C_t`.  **PENDING.**

    :math:`C_t = \frac{1}{3}[\mathrm{rank}_W(\hat I^{(1)}) +
    \mathrm{rank}_W(\hat I^{(2)}) + \mathrm{rank}_W(\hat I^{(3)})]`

    Blocked until :func:`compute_I_hat_2` and :func:`compute_I_hat_3` are
    implemented; see ``MIGRATION_NOTES.md``.

    Raises
    ------
    NotImplementedError
        Always, until the pending decisions are resolved.
    """
    raise NotImplementedError(
        "compute_concordance_C is not yet implemented. "
        "Blocked on compute_I_hat_2 and compute_I_hat_3. "
        "See MIGRATION_NOTES.md for the full dependency chain."
    )


__all__ = [
    "compute_I_hat_1",
    "compute_I_hat_2",
    "compute_I_hat_3",
    "compute_concordance_C",
]
