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



### [PHD-UNI-PRELIM] Universality preliminary (2026-05-01): GO for v2

All pairwise Pearson r ≥ 0.0.  Full experiment authorised.



### [PHD-UNI-PRELIM] Universality preliminary (2026-05-01): FAIL

Pairwise Pearson r below 1.01:
- openvla vs pi0: r=0.183
- openvla vs bc_act: r=0.352
- openvla vs diffusion_policy: r=-0.128
- pi0 vs bc_act: r=0.418
- pi0 vs diffusion_policy: r=0.700
- bc_act vs diffusion_policy: r=0.626

Human decision required: reframe or abort.



### [PHD-UNI-PRELIM] Universality preliminary (2026-05-01): GO for v2

All pairwise Pearson r ≥ 0.0.  Full experiment authorised.



### [PHD-UNI-PRELIM] Universality preliminary (2026-05-01): FAIL

Pairwise Pearson r below 1.01:
- openvla vs pi0: r=0.183
- openvla vs bc_act: r=0.352
- openvla vs diffusion_policy: r=-0.128
- pi0 vs bc_act: r=0.418
- pi0 vs diffusion_policy: r=0.700
- bc_act vs diffusion_policy: r=0.626

Human decision required: reframe or abort.



### [PHD-UNI-PRELIM] Universality preliminary (2026-05-01): GO for v2

All pairwise Pearson r ≥ 0.0.  Full experiment authorised.



### [PHD-UNI-PRELIM] Universality preliminary (2026-05-01): FAIL

Pairwise Pearson r below 1.01:
- openvla vs pi0: r=0.183
- openvla vs bc_act: r=0.352
- openvla vs diffusion_policy: r=-0.128
- pi0 vs bc_act: r=0.418
- pi0 vs diffusion_policy: r=0.700
- bc_act vs diffusion_policy: r=0.626

Human decision required: reframe or abort.



### [PHD-UNI-PRELIM] Universality preliminary (2026-05-01): GO for v2

All pairwise Pearson r ≥ 0.0.  Full experiment authorised.



### [PHD-UNI-PRELIM] Universality preliminary (2026-05-01): FAIL

Pairwise Pearson r below 1.01:
- openvla vs pi0: r=0.183
- openvla vs bc_act: r=0.352
- openvla vs diffusion_policy: r=-0.128
- pi0 vs bc_act: r=0.418
- pi0 vs diffusion_policy: r=0.700
- bc_act vs diffusion_policy: r=0.626

Human decision required: reframe or abort.

