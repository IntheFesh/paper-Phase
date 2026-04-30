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

### Phase C Completed 2026-04-30

**§3.1  HierarchicalPlanner** (`modeling_phaseqflow.py`)
- `FSQSkillEncoder` accepts optional `levels` override param (backward compat)
- `HierarchicalPlanner` adds `phase_mode: "flat"|"hierarchical"` switch in `__init__`
- Hierarchical mode: macro encoder (K₁=20, levels=[5,4]) + micro encoder (K₂=30, levels=[6,5])
- Forward returns extra keys: `z_macro_idx`, `z_micro_idx`, `e_macro`, `e_micro`, `logits_macro`, `logits_micro`
- `phase_logits`/`phase_embed` remain the macro equivalents (backward compat)
- Config fields added: `phase_mode="flat"`, `fsq_levels_macro=[5,4]`, `fsq_levels_micro=[6,5]`

**§3.2  Hierarchical InfoNCE** (`phase_centric/identifiability.py`)
- `ChunkInfoNCEHead` upgraded: `_single_level_infonce` extracted, `_encode_chunk` extracted
- `forward(logits_micro=None)` — hierarchical path: `L_total = L_macro + 0.5 * L_micro`
- Flat mode (logits_micro=None) backward compatible; same-phase masking on both levels

**§3.3  Boundary-Aware Flow Loss** (`phase_centric/boundary_aware_flow.py`)
- New file: `compute_boundary_aware_flow_loss` — w(β)=1+λ*β weighting, no entropy term
- `pace_a_loss.py` → shim re-exporting from `boundary_aware_flow` + fully backward-compat `compute_pace_a_flow_loss` (entropy preserved for ablation tests)
- `compute_loss` uses `beta_micro` (micro posterior) if available; falls back to `phase_beta`
- Config: `use_boundary_reweight=True`, `boundary_reweight_lambda=0.5`

**§3.4  4→3-stage curriculum** (`configs/train/`)
- `01_pretrain_multimodal.yaml`, `02_train_phase_and_flow.yaml`, `03_finetune_replan.yaml`
- Old YAML files deprecated with header; `scripts/train.py` with `--stage`/`--smoke_mode`
- Stage 3 `calibration_only=True` skips gradient updates

**§3.5  Calibration + B-PCAR** (`phase_centric/b_pcar.py`, `_pcar_common.py`)
- `_pcar_common.py`: shared rolling-quantile + budget utilities
- `b_pcar.py`: `BayesianPCARTrigger` — Beta-mixture prior + quantile trigger dual path
- `scripts/calibration/calibrate_concordance.py`: W × θ_C sweep with recall/FPR
- `scripts/calibration/calibrate_b_pcar.py`: budget × α_prior sweep

**§3.6  Sanity runs**: all 3 stages pass `--smoke_mode`
- `_infer_planner_k` fixed to read `fsq_levels_macro` in hierarchical mode
- `PhasePosteriorEstimator` accepts `k_override` for micro-level instance

- Tests: 123 passed (unchanged — all Phase B tests still pass)
- Smoke: 7-mode all pass
- New dependencies introduced: none

---

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

---

### Phase D Completed 2026-04-30

**§4.1  Universality experiment** (`scripts/phenomenon/universality.py`)
- `PolicyAdapter` ABC + 4 adapters in `baselines/`: `OpenVLAAdapter`, `Pi0Adapter`,
  `BCActAdapter`, `DiffusionPolicyAdapter`
- `--dry_run` synthetic mode (no checkpoints needed); `--preliminary` gate (§4.7)
- Outputs: per-policy failure-distance histograms, `overlay_figure1.png`,
  `ks_pvalue_matrix.csv`, `raw_distances.json`, `summary.md`
- Preliminary gate: pairwise Pearson r ≥ threshold; PASS → "GO for v2" in MIGRATION_NOTES;
  FAIL → human notification + data logged here

**§4.2  Regret scaling** (`scripts/phenomenon/regret_scaling.py`)
- `--H` sweep; reference policy = same model with `action_chunk_size=1`
- `_compute_mean_cliff_depth` via concordance diagnostics
- Outputs: `regret_vs_H.csv`, `regret_vs_H.png` (scatter + R² linear fit), `summary.md`

**§4.3  Triangulation concordance** (`scripts/phenomenon/triangulation_concordance.py`)
- Oracle: gripper-flip indices; ±5 timestep tolerance
- Evaluates: I_hat_1, I_hat_2, I_hat_3, concordance_C
- Outputs: `triangulation_table.csv`, `triangulation_f1.png`, `summary.md`
- Synthetic data validates concordance precision > single estimator precision

**§4.4  LIBERO-Long-Perturbed** (`scripts/eval/libero_perturbed.py`)
- LIBERO-PRO protocol: ±5 cm XY perturbation + instruction paraphrase
- Object perturbation via `env.env.sim.data.body_xpos` (LIBERO MuJoCo API)
- `_INSTRUCTION_PARAPHRASES` vocab (fixed set for reproducibility)
- Outputs: per-perturbation CSV + JSON summary

**§4.5  SimplerEnv** (`scripts/eval/simpler.py`)
- Google Robot Visual Matching: pick_coke_can / move_near / open_drawer / put_eggplant_in_basket
- Observation → PhaseQFlow input conversion via `_simpler_obs_to_tensor`
- Outputs: `simpler_results.csv`, `simpler_aggregate.json`

**§4.6  Orchestration** (`scripts/run_experiments.sh`, `scripts/aggregate_results.py`)
- `run_experiments.sh`: master script with `--dry_run` / `--checkpoint` flags
- `aggregate_results.py`: reads all experiment outputs → `paper_figures/main_results.csv`
- Shell runners: `scripts/eval/run_libero_main.sh`, `scripts/eval/run_simpler.sh`

- Tests: 123 → 151 passed (+28: `test_phenomenon_smoke.py`)
- Smoke: all dry_run modes functional
- New dependencies introduced: none (matplotlib, scipy optional for plots / KS test)

---

## Pending Human Decisions

### [PHD-4] OpenVLA checkpoint for LIBERO-Long

**Status: OPEN**
OpenVLA-7b base checkpoint (`openvla/openvla-7b`) was not fine-tuned on LIBERO-Long.
Options:
- (a) Zero-shot evaluation (likely low SR but scientifically valid)
- (b) Fine-tune on LIBERO-Long (10k demos; estimate 2 GPU-days)
- (c) Use an existing community fine-tune if one exists on HuggingFace

**Required action**: human chooses (a/b/c), updates `baselines/openvla_adapter.py`
`_DEFAULT_CHECKPOINT` accordingly, and records decision here.

---

### [PHD-5] π0 checkpoint for LIBERO-Long

**Status: OPEN**
The public `lerobot/pi0` checkpoint was trained on a broad manipulation mixture
but not specifically on LIBERO-Long.  Same options as [PHD-4].

---

### [PHD-6] BC-ACT checkpoint for LIBERO-Long

**Status: OPEN**
LeRobot ACT checkpoints target ALOHA / PushT, not LIBERO-Long.  Options:
- (a) Train ACT from scratch on LIBERO-Long demos via LeRobot
- (b) Find a community LIBERO-Long ACT checkpoint

---

### [PHD-7] Diffusion Policy checkpoint for LIBERO-Long

**Status: OPEN**
`lerobot/diffusion_pusht` targets PushT.  Options:
- (a) Train Diffusion Policy on LIBERO-Long via LeRobot
- (b) [PHD-7b] Use robomimic framework if it has LIBERO-Long checkpoints

---

### [PHD-8] LIBERO-PRO open-source eval harness

**Status: OPEN**
LIBERO-PRO has not released a public eval harness as of 2026-04-30.
The perturbation logic in `scripts/eval/libero_perturbed.py` is a
re-implementation.  If LIBERO-PRO releases open source code, migrate
to it and delete `_apply_object_perturbation`.

---

### [PHD-9] SimplerEnv version pin

**Status: OPEN**
Confirm which SimplerEnv commit / version to pin for reproducibility.
The adapter targets the Google Robot Visual Matching variant from ≥ 2024-12.

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


### [PHD-UNI-PRELIM] Universality preliminary (2026-04-30): GO for v2

All pairwise Pearson r ≥ 0.0.  Full experiment authorised.



### [PHD-UNI-PRELIM] Universality preliminary (2026-04-30): FAIL

Pairwise Pearson r below 1.01:
- openvla vs pi0: r=0.183
- openvla vs bc_act: r=0.352
- openvla vs diffusion_policy: r=-0.128
- pi0 vs bc_act: r=0.418
- pi0 vs diffusion_policy: r=0.700
- bc_act vs diffusion_policy: r=0.626

Human decision required: reframe or abort.



### [PHD-UNI-PRELIM] Universality preliminary (2026-04-30): GO for v2

All pairwise Pearson r ≥ 0.0.  Full experiment authorised.



### [PHD-UNI-PRELIM] Universality preliminary (2026-04-30): FAIL

Pairwise Pearson r below 1.01:
- openvla vs pi0: r=0.183
- openvla vs bc_act: r=0.352
- openvla vs diffusion_policy: r=-0.128
- pi0 vs bc_act: r=0.418
- pi0 vs diffusion_policy: r=0.700
- bc_act vs diffusion_policy: r=0.626

Human decision required: reframe or abort.



### [PHD-UNI-PRELIM] Universality preliminary (2026-04-30): GO for v2

All pairwise Pearson r ≥ 0.0.  Full experiment authorised.



### [PHD-UNI-PRELIM] Universality preliminary (2026-04-30): FAIL

Pairwise Pearson r below 1.01:
- openvla vs pi0: r=0.183
- openvla vs bc_act: r=0.352
- openvla vs diffusion_policy: r=-0.128
- pi0 vs bc_act: r=0.418
- pi0 vs diffusion_policy: r=0.700
- bc_act vs diffusion_policy: r=0.626

Human decision required: reframe or abort.



### [PHD-UNI-PRELIM] Universality preliminary (2026-04-30): GO for v2

All pairwise Pearson r ≥ 0.0.  Full experiment authorised.



### [PHD-UNI-PRELIM] Universality preliminary (2026-04-30): FAIL

Pairwise Pearson r below 1.01:
- openvla vs pi0: r=0.183
- openvla vs bc_act: r=0.352
- openvla vs diffusion_policy: r=-0.128
- pi0 vs bc_act: r=0.418
- pi0 vs diffusion_policy: r=0.700
- bc_act vs diffusion_policy: r=0.626

Human decision required: reframe or abort.

