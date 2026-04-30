"""Predictability-Cliff detection subpackage.

Modules
-------
- ``posterior_bhattacharyya`` — I^(1) cliff estimator (Bhattacharyya β_t wrapper)
- ``policy_variance``         — I^(2) cliff estimator (action-ensemble variance)
- ``velocity_curvature``      — I^(3) cliff estimator (velocity-field difference)
- ``concordance``             — C_t concordance detector (rank-based fusion)

Implementation Phase: B
References: 01_pace_master_plan_v2.md §2.1, MIGRATION_NOTES.md §Pending Human Decisions
Status: PLACEHOLDER — cliff_detection/ created in Phase A; implementations land in Phase B
"""

from __future__ import annotations

from lerobot_policy_phaseqflow.phase_centric.cliff_detection.concordance import (
    ConcordanceDetector,
)
from lerobot_policy_phaseqflow.phase_centric.cliff_detection.policy_variance import (
    PolicyVarianceEstimator,
)
from lerobot_policy_phaseqflow.phase_centric.cliff_detection.posterior_bhattacharyya import (
    PosteriorBhattacharyyaEstimator,
)
from lerobot_policy_phaseqflow.phase_centric.cliff_detection.velocity_curvature import (
    VelocityCurvatureEstimator,
)

__all__ = [
    "ConcordanceDetector",
    "PolicyVarianceEstimator",
    "PosteriorBhattacharyyaEstimator",
    "VelocityCurvatureEstimator",
    "posterior_bhattacharyya",
    "policy_variance",
    "velocity_curvature",
    "concordance",
]
