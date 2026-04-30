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

__all__ = [
    "posterior_bhattacharyya",
    "policy_variance",
    "velocity_curvature",
    "concordance",
]
