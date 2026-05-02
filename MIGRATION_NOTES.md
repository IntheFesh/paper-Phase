# PACE v2 — Migration Notes

## CoRL Submission Scope (v2.1)

The cloud sweep launched by `scripts/run_autodl_pipeline.sh train` runs only the
experiments below. Earlier ablations 03/04/05/06 and §6.1/6.3/6.4 were dropped
from the submission scope after their contribution to the paper narrative was
judged redundant or out-of-budget. The underlying implementations
(`compute_I_hat_2`, `compute_I_hat_3`, `compute_concordance_C`) remain in the
codebase and are exercised by `tests/test_cliff_estimators.py` because
**Ablation 07 uses all three internally via the concordance signal**.

### Trained on real GPU data (CoRL submission)
| Experiment | Implementation | Role in paper |
|------------|---------------|---------------|
| 3-stage main training (Stage 1→2→3) | `scripts/train.py` | Table 1 headline checkpoint |
| Ablation 01: BC-Chunked × 3 seeds | imitation only | Table 2 baseline |
| Ablation 02: Cliff via β_t × 3 seeds | `compute_I_hat_1` | Table 2 — concordance > β alone |
| Ablation 07: Full PACE v2 × 3 seeds | C_t + boundary reweight | Table 2 headline |
| §6.2 Regret Scaling | LIBERO env_factory | Theoretical claim |
| §6.5 Boundary Loss Ratio | β_t-conditioned loss | Theoretical claim |

All three cliff estimators are wired into `PhaseQFlowPolicy.forward()`:
- `I_hat_1` always present when `phase_beta` is available.
- `I_hat_3` computed from consecutive `v_θ(anchor, c_t)` vs cached `c_{t-1}`; cache held in `_v_theta_prev`, cleared by `policy.reset()`.
- `I_hat_2` computed when BID sampler produces N≥2 action samples (`bid_chunks` in preds).
- `concordance_C` fuses all available estimators per step via rolling rank windows (`_concordance_state`, also cleared by `reset()`).

Tests: `tests/test_cliff_estimators.py` — 22 unit + integration cases (no NotImplementedError stubs remain).

### Dropped from CoRL submission
| Experiment | Reason for drop |
|------------|-----------------|
| Ablation 03 / 04 (single-estimator cliff) | Implementations covered by unit tests; standalone ablations not needed for the paper |
| Ablation 05 (concordance only, no boundary reweight) | Subsumed by Ablation 07 |
| Ablation 06 (Oracle cliff) | Upper-bound context only; not required by reviewers |
| §6.1 Universality | Baseline checkpoints ~220 GB exceed disk budget |
| §6.3 Triangulation | Redundant with Ablation 02 vs 07 |
| §6.4 Trigger comparison | Pure algorithmic, synthetic by design |
| SimplerEnv evaluation | Requires ManiSkill2, not core claim |
| LIBERO-Perturbed | Supplementary, not required for Table 1/2 |

The YAML files for the dropped ablations (03/04/05/06) and the phenomenon
scripts (universality, triangulation, trigger comparison) remain in the
codebase for reference and unit tests, but the cloud sweep does not run them.

---

## v2 Runtime Cliff Detection (added)

The v2 inference path is split into two layers; **do not delete either**.

### Training-side (`phase_centric/`)
Used inside `PhaseQFlowPolicy.forward()` to compute the cliff signals as
auxiliary losses / diagnostics on tensor batches. Files:
`phase_centric/cliff_estimators.py`, `phase_centric/cliff_detection/{concordance,
policy_variance, velocity_curvature, posterior_bhattacharyya}.py`.
Consumed by `tests/test_cliff_estimators*.py`, `tests/test_concordance.py`,
`scripts/phenomenon/triangulation_concordance.py`, and the calibration
scripts. **Stays in place.**

### Runtime-side (`inference/`, new in v2)
Used by `scripts/eval/libero_perturbed.py` (and any future eval harness)
to make per-step replanning decisions:

| File | Role |
|---|---|
| `inference/cliff_estimators.py` | `compute_policy_variance` (I^(2)), `compute_velocity_curvature` (I^(3)) — wrap policy interface, no training deps |
| `inference/concordance.py` | `ConcordanceDetector` — sliding-window rank fusion → C_t |

I^(1) (Bhattacharyya β_t) is read from `policy._last_beta`, written by
`PhaseQFlowPolicy.select_action()` after every forward call.
The flow head's condition vector c_t is cached at
`flow_action_head._last_condition` (set in `ShortcutFlowActionHead.forward()`)
so I^(3) can be computed without an extra forward pass.
Tests: `tests/test_inference_concordance.py`.

### action_dim = 7 reset
`PhaseQFlowConfig.action_dim` was changed from `16 → 7` to match LIBERO's
7-DoF action space. **Stage 1 and Stage 2 checkpoints trained with the old
default are incompatible with the current code** — retrain from scratch or
use `CheckpointManager.load_partial()` (added) to skip the mismatched
action head and re-initialise it.
