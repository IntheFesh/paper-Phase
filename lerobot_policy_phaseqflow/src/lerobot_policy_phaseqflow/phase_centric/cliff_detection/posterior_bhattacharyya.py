"""I^(1) Predictability-Cliff estimator — Bhattacharyya boundary signal wrapper.

Wraps the existing PhasePosteriorEstimator._bhattacharyya_beta into the
cliff_detection public interface. I_hat_1(t) = -beta_t.

Implementation Phase: B
References: 01_pace_master_plan_v2.md §2.1 (I^(1) definition)
            phase_centric/phase_posterior.py (_bhattacharyya_beta)
            phase_centric/cliff_estimators.py (compute_I_hat_1 — already implemented)
Status: PLACEHOLDER — thin wrapper; delegating to cliff_estimators.compute_I_hat_1
"""

from __future__ import annotations

raise NotImplementedError("Will be implemented in Phase B")
