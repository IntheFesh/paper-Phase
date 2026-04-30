# PACE v2 — Migration Notes

## Experiment Status Summary (v2.0)

### Runnable with real GPU data
| Config | Status |
|--------|--------|
| Ablation 01: BC-Chunked | ✅ Real data |
| Ablation 02: Cliff via β_t (I^1) | ✅ Real data |
| Ablation 06: Oracle cliff | ✅ Real data |
| Ablation 07: Full PACE v2 | ✅ Real data |
| §6.2 Regret Scaling | ✅ Real data |
| §6.5 Boundary Loss Ratio | ✅ Real data |

### Disabled (NotImplementedError)
| Config | Blocker | Target |
|--------|---------|--------|
| Ablation 03: Cliff via σ² (I^2) | `compute_I_hat_2` | v2.1 |
| Ablation 04: Cliff via κ (I^3) | `compute_I_hat_3` | v2.1 |
| Ablation 05: Concordance C_t | needs I^2 + I^3 | v2.1 |
| §6.3 Triangulation (real data) | concordance incomplete | v2.1 |

### Dry-run only (pipeline validated, numbers synthetic)
| Experiment | Reason |
|-----------|--------|
| §6.1 Universality | Baseline checkpoints ~220GB, storage constraint |
| §6.3 Triangulation | I^2/I^3 pending |
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

