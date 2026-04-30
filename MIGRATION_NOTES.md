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

### [PHD-1] I_hat_2 — Action-Variance Cliff Estimator  ✅ RESOLVED

**Decision:** Option B — N independent forward passes inside `PolicyVarianceEstimator.estimate()`.
Each call to `flow_head.forward(training=False)` samples fresh Gaussian noise
internally, yielding N distinct action predictions.  Default N=8.
**Implemented:** `cliff_detection/policy_variance.py` — `PolicyVarianceEstimator`.

---

### [PHD-2] I_hat_3 — Velocity-Difference Cliff Estimator  ✅ RESOLVED

- **PHD-2a** — Anchor: episode-fixed noise at τ=0.5 (midpoint), d=1.0.
  Drawn once per `VelocityCurvatureEstimator.update()` call; re-drawn when B changes.
- **PHD-2b** — `c_{t-1}` stored as `_prev_cond` tensor inside `VelocityCurvatureEstimator`.
- **PHD-2c** — Added public `velocity(x_tau, tau, cond, d)` and `compute_cond(fused_obs,
  phase_embed, skill_latent)` to `ShortcutFlowActionHead`.  Private `_velocity` unchanged.
  `FlowActionHeadPACE` does not expose these methods (incompatible — uses PhaseMoE).

**Implemented:** `cliff_detection/velocity_curvature.py` — `VelocityCurvatureEstimator`.

---

### [PHD-3] Concordance C_t — Rank-Based Fusion  ✅ RESOLVED

**Decision:** W=50 (default), mid-rank convention for ties
(`rank = 1 - (below + 0.5*equal) / W`), warmup_steps=W.
**Implemented:** `cliff_detection/concordance.py` — `ConcordanceDetector`.

---

## Phase Completion Log

### Phase B (full) Completed 2026-04-30

**§2.1  PosteriorBhattacharyyaEstimator** (`cliff_detection/posterior_bhattacharyya.py`)
- Thin delegator over `PhasePosteriorEstimator`; `step(logits) → {i_hat_1, beta, p_hat}`
- PHD-1/2/3 all resolved (see above)

**§2.2  PolicyVarianceEstimator** (`cliff_detection/policy_variance.py`)
- N=8 independent forward passes; `estimate(flow_head, ...) → {i_hat_2, sigma_sq}`

**§2.3  VelocityCurvatureEstimator** (`cliff_detection/velocity_curvature.py`)
- Anchor τ=0.5, episode-fixed noise; `update(flow_head, ...) → {i_hat_3, cond_diff_sq}`
- `ShortcutFlowActionHead.velocity()` and `.compute_cond()` added as public methods

**§2.4  ConcordanceDetector** (`cliff_detection/concordance.py`)
- W=50, mid-rank convention, warmup suppression; `step(i1, i2, i3) → {triggered, concordance, ranks}`
- `cliff_detection/__init__.py` updated to export all four classes

**§2.5  PredictiveInfoEstimator** (`phase_centric/theory_utils.py`)
- Bilinear InfoNCE critic; `forward(x, c) → {mi_lower_bound, logits}`
- `estimate_per_timestep(x_seq, c_seq) → (T,)` for oracle calibration

**§2.6  PCAR upgrade** (`phase_centric/pcar_trigger.py`, `configuration_phaseqflow.py`)
- `PCARTrigger.update_and_check(beta)` → `update_and_check(signal)` (positional BC preserved)
- `pcar_input_signal: str = "concordance"` added to `PhaseQFlowConfig`
- `PCARTrigger.input_signal` attribute set from config

- Tests: 94 → 123 passed (+29: test_cliff_estimators_phase_b.py×19, test_concordance.py×10)
- Smoke: 7-mode all pass
- New dependencies introduced: none

---

### Phase B (initial) Completed 2026-04-30

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
| Cliff estimator I^(2) | `PolicyVarianceEstimator.estimate` → `i_hat_2` | `I_hat_2` |
| Cliff estimator I^(3) | `VelocityCurvatureEstimator.update` → `i_hat_3` | `I_hat_3` |
| Concordance C_t | `ConcordanceDetector.step` → `concordance` | `concordance_C` |
| Phase posterior p̂_t | `phase_p_hat`, `p_hat` | — (internal only) |
| Bhattacharyya distance β_t | `_bhattacharyya_beta` | — (internal only) |
