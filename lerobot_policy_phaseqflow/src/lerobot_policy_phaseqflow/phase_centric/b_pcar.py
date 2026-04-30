"""Bayesian PCAR — concordance-driven replanning trigger (Phase C upgrade).

Replaces the budget-quantile PCARTrigger with a Bayesian changepoint detector
that fuses the three cliff estimators I_hat_1/2/3 via the concordance signal
C_t, giving a principled posterior over run-length for replan decisions.

Implementation Phase: C
References: 01_pace_master_plan_v2.md §Phase C
            phase_centric/pcar_trigger.py (existing budget-quantile PCARTrigger)
            phase_centric/cliff_detection/concordance.py (C_t input signal)
Status: PLACEHOLDER — implementation pending Phase C
"""

from __future__ import annotations

raise NotImplementedError("Will be implemented in Phase C")
