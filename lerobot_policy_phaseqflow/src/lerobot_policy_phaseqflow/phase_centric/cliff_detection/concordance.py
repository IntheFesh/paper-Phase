"""Concordance detector C_t — rank-based fusion of the three cliff estimators.

C_t = (1/3) * [rank_W(I_hat_1(t)) + rank_W(I_hat_2(t)) + rank_W(I_hat_3(t))]

where rank_W is the percentile rank within a rolling window of size W.

Implementation Phase: B
References: 01_pace_master_plan_v2.md §2.1 (C_t definition, rank_W fusion rule)
            MIGRATION_NOTES.md §PHD-3 (window size W pending; blocked on PHD-1, PHD-2)
Status: PLACEHOLDER — blocked on PHD-1 and PHD-2 (see MIGRATION_NOTES.md)
"""

from __future__ import annotations

raise NotImplementedError("Will be implemented in Phase B — blocked on PHD-1 and PHD-2")
