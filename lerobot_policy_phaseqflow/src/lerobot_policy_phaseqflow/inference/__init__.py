"""Runtime-facing cliff detection (PACE v2 inference path).

The training-side estimators live in ``phase_centric/`` and are used inside
``PhaseQFlowPolicy.forward()`` for loss computation. This ``inference/``
package provides the **runtime** counterparts used at evaluation time to
trigger PCAR replanning:

* :func:`compute_policy_variance` — I^(2) action sample variance
* :func:`compute_velocity_curvature` — I^(3) flow-field finite-difference sensitivity
* :class:`ConcordanceDetector` — rank-fusion of I^(1), I^(2), I^(3) → C_t

I^(1) (Bhattacharyya β_t) is read directly from ``policy._last_beta``,
which is cached by :meth:`PhaseQFlowPolicy.select_action` after every
forward call.
"""

from .cliff_estimators import compute_policy_variance, compute_velocity_curvature
from .concordance import ConcordanceDetector

__all__ = [
    "compute_policy_variance",
    "compute_velocity_curvature",
    "ConcordanceDetector",
]
