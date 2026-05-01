# PACE v2 — Migration Notes

## Experiment Status Summary (v2.1 — all cliff estimators implemented)

### Runnable with real GPU data
| Config | Implementation | Status |
|--------|---------------|--------|
| Ablation 01: BC-Chunked | imitation only | ✅ Real data |
| Ablation 02: Cliff via β_t (I^1) | `compute_I_hat_1` | ✅ Real data |
| Ablation 03: Cliff via σ² (I^2) | `compute_I_hat_2` | ✅ Real data (v2.1) |
| Ablation 04: Cliff via κ (I^3) | `compute_I_hat_3` | ✅ Real data (v2.1) |
| Ablation 05: Concordance C_t | `compute_concordance_C` | ✅ Real data (v2.1) |
| Ablation 06: Oracle cliff | oracle from labels | ✅ Real data |
| Ablation 07: Full PACE v2 | C_t + boundary reweight | ✅ Real data |
| §6.2 Regret Scaling | LIBERO env_factory | ✅ Real data |
| §6.3 Triangulation | concordance complete | ✅ Real data (v2.1) |
| §6.5 Boundary Loss Ratio | β_t-conditioned loss | ✅ Real data |

All three cliff estimators are wired into `PhaseQFlowPolicy.forward()`:
- `I_hat_1` always present when `phase_beta` is available.
- `I_hat_3` computed from consecutive `v_θ(anchor, c_t)` vs cached `c_{t-1}`; cache held in `_v_theta_prev`, cleared by `policy.reset()`.
- `I_hat_2` computed when BID sampler produces N≥2 action samples (`bid_chunks` in preds).
- `concordance_C` fuses all available estimators per step via rolling rank windows (`_concordance_state`, also cleared by `reset()`).

Tests: `tests/test_cliff_estimators.py` covers all 22 unit + integration cases (no NotImplementedError stubs remain).

### Dry-run only (out of scope for the v2.1 paper)
| Experiment | Reason |
|-----------|--------|
| §6.1 Universality | Baseline checkpoints ~220GB; out-of-scope for paper-budget storage |
| §6.4 Trigger comparison | Pure algorithmic, synthetic sufficient |
| SimplerEnv evaluation | Requires ManiSkill2, not core claim |
| LIBERO-Perturbed | Supplementary, not required for Table 1/2 |

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

