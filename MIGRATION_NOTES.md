# PACE v2 重构日志

## Repository Audit (filled by Claude Code)

Confirmed against actual repo structure (last verified: 2026-04-30):

- LeRobot plugin layout: confirmed at
  `lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/`
  registered via `@PreTrainedConfig.register_subclass("phaseqflow")`
- `phase_centric/` subpackage: 8 files present
  (`__init__.py`, `identifiability.py`, `phase_posterior.py`,
  `pace_a_loss.py`, `pace_b_moe.py`, `pace_c_curriculum.py`,
  `pcar_trigger.py`, `theory_utils.py`, `cliff_estimators.py`)
- pytest baseline: **94 passed** (80 original + 14 cliff-estimator tests added in Phase B)
- smoke baseline: **7 modes pass** (`off`, `ident_only`, `pace_a`, `pace_b`, `pace_c`, `pcar`, `full`)
- Differences from prompt assumed layout:
  - No `scripts/verification/` directory (verification scripts referenced in
    ARCHITECTURE.md do not exist yet; OPERATIONS_GUIDE.md §11 references them)
  - No `configs/eval/` directory (referenced by Phase A BID check)
  - `use_iql_verifier` field did not exist in original config; added in Phase A
  - `cliff_estimators.py` already created in prior work; `cliff_detection/` full
    subpackage created in Phase A
  - Test count is 94, not 77 (14 cliff-estimator tests pre-exist this phase)

## Pending Human Decisions

### [PHD-1] I_hat_2 — Action-Variance Cliff Estimator

**Formula (locked):**
$\hat I^{(2)}(t) \propto -\sigma_t^2 = -\frac{1}{N}\sum_{i=1}^{N}\|a_t^{(i)} - \bar a_t\|^2$

**Blocking question:** How should N action samples be drawn?

- **Option A** — BID path (N=5 candidates in `select_action`): available only at
  inference, not training; candidates share one condition vector so variance is
  aleatoric only.
- **Option B** — Multi-noise-seed call inside `forward`: N extra flow forward
  passes; 5× compute overhead; need to decide training vs. inference vs. both.
- **Option C** — Temporal-ensembling buffer variance across time steps.

**Decision needed:** Which option is the intended semantic?

**Blocked:** `cliff_detection/policy_variance.py`, `concordance.py`.

---

### [PHD-2] I_hat_3 — Velocity-Difference Cliff Estimator

**Formula (locked):**
$\hat I^{(3)}(t) \propto -\|v_\theta(x_\tau, \tau, c_t) - v_\theta(x_\tau, \tau, c_{t-1})\|_2^2$

Three sub-decisions:

- **PHD-2a** — Anchor point `(x_τ, τ, d)`: zero vector / episode-fixed noise /
  previous action chunk? τ=0, 0.5, or 1?
- **PHD-2b** — Storage of `c_{t-1}`: buffer inside `ShortcutFlowActionHead` or
  passed in from `select_action` loop?
- **PHD-2c** — Velocity exposure: rename `_velocity` → `velocity` or add public
  `eval_velocity(x, t, d, cond)` wrapper?

**Blocked:** `cliff_detection/velocity_curvature.py`, `concordance.py`.

---

### [PHD-3] Concordance C_t — Rank-Based Fusion

**Formula (locked):**
$C_t = \frac{1}{3}[\mathrm{rank}_W(\hat I^{(1)}) + \mathrm{rank}_W(\hat I^{(2)}) + \mathrm{rank}_W(\hat I^{(3)})]$

**Blocked on:** PHD-1 and PHD-2.
**Sub-question:** Window size W? (PCAR uses 1000; suggest W=50 matching warmup_min.)

**Blocked:** `cliff_detection/concordance.py`.

---

## Phase Completion Log

### Phase B Completed 2026-04-30

- Added cliff-namespace public-facing interfaces:
  - Created `phase_centric/cliff_estimators.py`:
    `compute_I_hat_1` (implemented), `I_hat_2/3/concordance_C` (stubs)
  - `modeling_phaseqflow.py`: exposes `I_hat_1 = -phase_beta` in
    `predict_action` output when `use_phase_boundary_posterior=True`
  - `phase_centric/__init__.py`: registered `cliff_estimators` in `__all__`
  - Added `.gitignore`
- Tests: 80 → 94 passed (14 new cliff-estimator tests)
- Smoke: 7-mode all pass
- New dependencies introduced: none
- 94 pytest passed, 7-mode smoke passed

---

### Phase A Completed 2026-04-30

- Deprecated 4 modules via cfg.use_* switches (no code-path branches added):
  - `IQLChunkVerifier` — deprecation header added; `use_iql_verifier: bool = False`
    added to config (existing wiring via `verifier_type: str = "iql"` unchanged)
  - `A2C2CorrectionHead` — deprecation header added; `use_correction_head`
    default changed `True → False` (**breaking-default change noted**)
  - `BIDSampler` — deprecation header added; `use_bid_sampling`
    default changed `True → False` (**breaking-default change noted**)
  - `pace_c_curriculum` — deprecation header added; `use_pace_c` already `False`
- Created skeleton: `phase_centric/cliff_detection/` subpackage +
  `scripts/{phenomenon,calibration,figures}/`
- Config default changes:
  - `use_correction_head: True → False`
  - `use_bid_sampling: True → False`
  - `use_iql_verifier: bool = False` (new field)
- Baselines unchanged: 94 pytest passed, 7-mode smoke passed
- New dependencies introduced: none

---

## Terminology Lock Table

| Paper term | Code (internal / Round-4 name) | Cliff-namespace name |
|---|---|---|
| Predictability Cliff | (concept) | cliff |
| Boundary probability β_t | `phase_beta`, `beta_t` | — (internal only) |
| Cliff estimator I^(1) | `-phase_beta` | `I_hat_1` |
| Cliff estimator I^(2) | (pending PHD-1) | `I_hat_2` |
| Cliff estimator I^(3) | (pending PHD-2) | `I_hat_3` |
| Concordance C_t | (pending PHD-3) | `concordance_C` |
| Phase posterior p̂_t | `phase_p_hat`, `p_hat` | — (internal only) |
| Bhattacharyya distance β_t | `_bhattacharyya_beta` | — (internal only) |
