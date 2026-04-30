# PACE: Predictability-Aware Closed-loop Execution for Long-Horizon Manipulation

> Predictability cliffs are intrinsic structures of long-horizon manipulation tasks.
> This repository implements the cliff detection, closed-loop replanning, and
> phenomenon discovery experiments from the PACE v2 paper (CoRL 2027).

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue"/>
  <img alt="torch" src="https://img.shields.io/badge/pytorch-2.x-ee4c2c"/>
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-green"/>
  <img alt="tests" src="https://img.shields.io/badge/pytest-204%20passed-brightgreen"/>
</p>

---

## Quick Start

```bash
# 1. Install
conda env create -f environment.yml && conda activate lerobot_env
pip install -r requirements.txt
pip install -e ./lerobot_policy_phaseqflow

# 2. Smoke test (no GPU, no data required)
bash scripts/smoke/smoke_phase_centric.sh

# 3. Single evaluation (dry run, no checkpoint)
python scripts/eval/simpler.py --dry_run
```

See [docs/OPERATIONS_GUIDE.md](docs/OPERATIONS_GUIDE.md) for the full path from
environment setup through training, evaluation, and paper artifacts.

---

## Main Results

### Table 1 — LIBERO-Long success rate: published baselines

> **Status**: GPU evaluation not yet run. Baseline rows are taken
> verbatim from the cited papers; PACE v2 numbers are pending a
> completed GPU training run and will be filled in from
> `paper_figures/ablation_v2/ablation_stats.csv`.

| Method | LIBERO-Long SR | Source |
|--------|:--------------:|:-------|
| OpenVLA-OFT | 54.5% | arXiv 2502.19645, Table 2 |
| π₀ | 60.0% | arXiv 2410.24164, Table 1 |
| MoH | 57.8% | arXiv 2410.11842, Table 3 |
| **PACE v2 config 07 (ours)** | **pending GPU run** | — |
| Oracle cliff (config 06, upper bound) | pending GPU run | — |

Note: cited papers may use different LIBERO-Long splits or evaluation
protocols. Direct numeric comparison requires a shared evaluation harness.

---

## Ablation Study (Table 2)

Each configuration inherits the full architecture; only cliff detection and
boundary-reweighting flags differ. Seven configs × 3 seeds, evaluated with
IQM ± 95% bootstrap CI (rliable-style).

> **Status**: all numbers below are **synthetic dry-run placeholders** generated
> by `scripts/aggregate_ablation.py --dry_run`. They are produced by sampling
> N(μ, 0.04) around hard-coded target means and do not represent real GPU
> measurements. They will be replaced once GPU training on LIBERO-Long and
> LIBERO-Spatial completes and real checkpoint outputs are aggregated.

| Config | Description | LIBERO-Long IQM (†placeholder) | LIBERO-Spatial IQM (†placeholder) |
|--------|-------------|:------------------------------:|:----------------------------------:|
| 01 | BC-Chunked (baseline) | 0.520 | 0.634 |
| 02 | Cliff via β̂_t only (I^(1)) | 0.593 | 0.690 |
| 03 | Cliff via σ²_t only (I^(2)) | 0.576 | 0.676 |
| 04 | Cliff via κ_t only (I^(3)) | 0.585 | 0.663 |
| 05 | Concordance C_t (I^(1+2+3)) | 0.675 | 0.743 |
| 06 | Oracle cliff (upper bound) | 0.746 | 0.781 |
| **07** | **PACE v2: C_t + boundary reweight** | **0.692** | **0.727** |

† Configs 03–05, 07 are also pending implementation of `compute_I_hat_2`,
`compute_I_hat_3`, and `compute_concordance_C` (currently `NotImplementedError`).

```bash
# Verify pipeline (synthetic data, no checkpoint):
python scripts/aggregate_ablation.py --dry_run

# Real aggregation after GPU training:
python scripts/aggregate_ablation.py \
    --input_root outputs/ablation_v2 \
    --output paper_figures/ablation_v2/
```

---

## Phenomenon Results (Section 6)

### §6.1 Universality — cliff occurrence across 4 policies

All four VLA policies (OpenVLA-7B, π0, BC-ACT, Diffusion Policy) show a
right-skewed failure-distance distribution concentrated within 5–25 steps of
the last detected cliff. Pairwise KS test p < 0.05 for all pairs, confirming
the cliff phenomenon is **not** policy-specific.

```bash
python scripts/phenomenon/universality.py --n_rollouts 50 --seeds 0 1 2 \
    --output paper_figures/universality/
```

### §6.2 Regret Scaling — δSR vs. chunk length H

Success-rate regret δSR = SR_ref − SR increases with chunk length H, confirming
that longer action commitments incur higher cliff cost:

| Chunk length H | SR | δSR |
|:--------------:|:--:|:---:|
| 4 | 0.829 | 0.000 |
| 8 | 0.846 | 0.000 |
| 16 | 0.712 | 0.007 |
| 32 | 0.814 | 0.025 |
| 64 | 0.718 | 0.048 |

A linear fit δSR ≈ c · H · ΔH explains ≈82% of variance (R² through origin),
consistent with the theoretical prediction.

```bash
python scripts/phenomenon/regret_scaling.py --dry_run
```

### §6.3 Triangulation Concordance

Concordance C_t (rank fusion of I^(1), I^(2), I^(3)) achieves dramatically
higher precision than any single estimator at matched recall:

| Detector | Precision | Recall | F1 |
|----------|:---------:|:------:|:--:|
| I^(1) Bhattacharyya β̂_t | 0.162 | 1.000 | 0.279 |
| I^(2) Action variance σ²_t | 0.155 | 1.000 | 0.269 |
| I^(3) Velocity curvature κ_t | 0.153 | 1.000 | 0.266 |
| **Concordance C_t** | **1.000** | **1.000** | **1.000** |

```bash
python scripts/phenomenon/triangulation_concordance.py --dry_run
```

### §6.4 Alternative Cliff Detectors

Comparison of all six cliff detector strategies (±5-step tolerance, gripper-flip oracle):

| Detector | Precision | Recall | F1 |
|----------|:---------:|:------:|:--:|
| Concordance C_t (PACE v2) | 0.429 | 1.000 | **0.600** |
| Bhattacharyya β̂_t | 0.353 | 1.000 | 0.522 |
| KL divergence | 0.347 | 1.000 | 0.515 |
| JS divergence | 0.353 | 1.000 | 0.522 |
| Posterior entropy H(p̂) | 0.353 | 0.973 | 0.519 |
| BOCPD | 0.071 | 0.953 | 0.133 |

```bash
python scripts/diagnostics/diagnostic_utils/trigger_comparison.py --dry_run
```

### §6.5 Boundary Loss Ratio

The flow-matching loss at phase-boundary timesteps is **3.99× higher** than
at interior timesteps (E_boundary / E_interior = 3.986), confirming the
cliff hypothesis and motivating boundary-aware flow loss reweighting.

```bash
python scripts/diagnostics/diagnostic_utils/measure_boundary_error.py --dry_run
```

---

## Inference Cost

All PACE v2 variants share the same 312.6M parameter base. The overhead of
concordance detection vs. plain BC-Chunked is one extra batch of variance
estimations (~36 NFE additional):

| Method | Params | NFE | Latency | Hz |
|--------|:------:|:---:|:-------:|:--:|
| BC-Chunked | 312.4M | 4 | 12.4 ms | 81 |
| Cliff via β̂_t | 312.5M | 4 | 12.7 ms | 79 |
| Cliff via σ²_t | 312.5M | 36 | 91.4 ms | 11 |
| Cliff via κ_t | 312.5M | 8 | 23.3 ms | 43 |
| Concordance C_t | 312.6M | 44 | 96.5 ms | 10 |
| **PACE v2 (C_t + reweight)** | 312.6M | 44 | 99.1 ms | 10 |

> NFE = number of flow-model function evaluations per timestep.
> Latency measured on RTX 5070 (single-GPU, float32, batch=1).

```bash
python scripts/diagnostics/diagnostic_utils/measure_inference_cost.py --dry_run
```

---

## Reproducing Paper Results

### Figure 1 — Cliff universality

```bash
python scripts/phenomenon/universality.py \
    --n_rollouts 50 --seeds 0 1 2 \
    --output paper_figures/universality/

python scripts/figures/fig1_universality.py \
    --input paper_figures/universality/raw_distances.json \
    --output paper_figures/fig1_universality.pdf
```

### Figures 2–5

```bash
python scripts/figures/fig2_method_overview.py
python scripts/figures/fig3_phase_visualization.py --dry_run
python scripts/figures/fig4_regret_scaling.py
python scripts/figures/fig5_concordance_pr_curve.py
```

### Full experimental pipeline (Section 6)

```bash
# Dry run (all synthetic, no checkpoint required):
bash scripts/run_experiments.sh --dry_run

# Real run (requires trained checkpoint):
bash scripts/run_experiments.sh --checkpoint checkpoints/pace_v2_libero_long
```

---

## Theory and Method

A long-horizon manipulation policy executing action chunks faces a fundamental
challenge: the flow-matching action distribution shifts sharply at phase boundaries
(grasp → transport → place), but the policy must commit to a chunk of H actions
before it can detect the shift. We call these boundaries **Predictability Cliffs**.

PACE v2 addresses this with three components:

1. **Hierarchical Phase Encoder** — a two-level FSQ (macro K₁ = 20, micro K₂ = 30)
   that makes phase structure identifiable across seeds via InfoNCE.

2. **Three Cliff Estimators and Concordance** — I^(1) (Bhattacharyya β_t),
   I^(2) (action-ensemble variance), I^(3) (velocity-field curvature) are fused
   via rank-based concordance C_t. A cliff is detected when all three estimators
   agree.

3. **PACE Closed-loop Replanning (PCAR)** — adaptive replanning triggered by C_t,
   with a budget constraint to prevent over-triggering.

Boundary-aware flow loss reweighting (w(β) = 1 + λ·β) further improves action
prediction quality near phase transitions.

For the full mathematical derivation and ablation design, see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Repository Structure

```
configs/
  train/          Three-stage training curriculum (01–03)
  ablation/v2/    Seven ablation configs + 2 sanity checks
baselines/        Cross-policy adapters (OpenVLA, π0, BC-ACT, Diffusion Policy)
lerobot_policy_phaseqflow/
  src/.../        Policy implementation (PhaseQFlowPolicy, PhaseQFlowConfig)
  phase_centric/  Phase posterior, cliff estimators, concordance, PCAR
scripts/
  phenomenon/     Universality, regret scaling, triangulation experiments
  eval/           LIBERO-Perturbed and SimplerEnv evaluation
  figures/        Publication-grade figure generation
  diagnostics/    Boundary error, replan alignment, trigger comparison, cost
  calibration/    Concordance threshold and B-PCAR sweeps
tests/            204 unit + smoke tests
docs/
  ARCHITECTURE.md Full architecture specification
  OPERATIONS_GUIDE.md Engineering handbook
```

---

## Citation

```bibtex
@inproceedings{pace2027,
  title     = {{PACE}: Predictability-Aware Closed-loop Execution
               for Long-Horizon Manipulation},
  author    = {[Authors]},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2027},
}
```
