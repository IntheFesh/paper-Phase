"""Phase-Centric VLA subpackage housing all phase-centric innovations.

Modules:

- ``identifiability``: chunk-level InfoNCE auxiliary loss for phase identifiability.
- ``phase_posterior``: smoothed posterior and boundary signal over phases.
- ``pace_a_loss``: PACE-A phase-aware sample reweighting.
- ``pace_b_moe``: PACE-B phase-gated MoE with smooth switching.
- ``pace_c_curriculum``: PACE-C phase-density curriculum scheduler.
- ``pcar_trigger``: PCAR phase-change-aware replanning trigger.
- ``theory_utils``: shared math helpers (Bayesian changepoint, empirical CDF, etc.).

Back-compat contract: every ``use_*`` switch on ``PhaseQFlowConfig`` defaults to
False. Unless a switch is flipped on, none of the code paths in this subpackage
get called, and the policy behaves exactly as it did before these additions.
Importing the top-level subpackage never triggers NotImplementedError, so CI
import-time smoke tests stay green.
"""

from __future__ import annotations

__all__ = [
    "cliff_estimators",
    "identifiability",
    "phase_posterior",
    "pace_a_loss",
    "pace_b_moe",
    "pace_c_curriculum",
    "pcar_trigger",
    "theory_utils",
]
