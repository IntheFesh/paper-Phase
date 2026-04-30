# PACE: Predictability-Aware Closed-loop Execution for Long-Horizon Manipulation

> Predictability cliffs are intrinsic structures of long-horizon manipulation tasks.
> This repository implements the cliff detection, closed-loop replanning, and
> phenomenon discovery experiments from the PACE v2 paper (CoRL 2027 submission).

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue"/>
  <img alt="torch" src="https://img.shields.io/badge/pytorch-2.x-ee4c2c"/>
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-green"/>
  <img alt="tests" src="https://img.shields.io/badge/pytest-204%20passed-brightgreen"/>
</p>

---

## Implementation Status

> Read this table before any number in this README. Anything tagged
> `placeholder` is a synthetic dry-run output; only `verified` items
> are checked numerically against the math.

| Item | Status | Notes |
|---|---|---|
| Phase InfoNCE identifiability ($\hat I^{(1)}$ infrastructure) | **verified** (CPU) | `verify_identifiability.py` PA $\ge 0.7$ |
| $\hat I^{(1)}$ Bhattacharyya $\beta_t$ | **verified** (CPU) | `verify_phase_posterior.py` peak-F1 $\ge 0.5$ |
| PCAR budget-quantile DKW bound | **verified** (CPU) | `verify_pcar_budget.py` $\|\text{rate}-\epsilon\|<0.005$ |
| Boundary-aware flow loss reduction | **verified** (CPU) | `sanity_pace_a.py` $\ge 20\%$ FM drop |
| Unit + smoke tests | **verified** (CPU) | `pytest tests/` → 204 passed |
| $\hat I^{(2)}$ Action variance | **pending** | `compute_I_hat_2` raises `NotImplementedError`; Ablation 03 **disabled** (placeholder only) |
| $\hat I^{(3)}$ Velocity curvature | **pending** | `compute_I_hat_3` raises `NotImplementedError`; Ablation 04 **disabled** (placeholder only) |
| Concordance $C_t$ rank fusion | **pending** | blocked on I^(2) and I^(3); Ablation 05 shows **partial** result (I^(1) only) |
| LIBERO-Long / Spatial SR (Tables 1, 2) | **placeholder** | requires GPU + LIBERO dataset + trained checkpoint |
| Phenomenon §6.1–6.5 numbers | **placeholder** | requires LIBERO rollouts; current numbers from `--dry_run` |
| Inference cost (params, NFE, latency) | **placeholder** | parameter count and NFE are correct per architecture; latency requires real GPU benchmark |

This environment is CPU-only with no LIBERO/SimplerEnv installed and
no trained checkpoints. The CPU-verifiable items above are what
this repo can prove correct today; everything tagged `placeholder`
needs a GPU + datasets to produce real numbers (see
**Verification & Reproduction Guide** below).

---

## Quick Start

```bash
# 1. Install (CPU is enough for verification scripts and unit tests)
conda env create -f environment.yml && conda activate lerobot_env
pip install -r requirements.txt
pip install -e ./lerobot_policy_phaseqflow

# 2. Smoke test (no GPU, no data required)
bash scripts/smoke/smoke_phase_centric.sh

# 3. Run all CPU-only verifications (~3 min total)
pytest tests/ -q
python scripts/verification/verify_identifiability.py
python scripts/verification/verify_phase_posterior.py
python scripts/verification/sanity_pace_a.py
python scripts/verification/verify_pcar_budget.py
```

See [docs/OPERATIONS_GUIDE.md](docs/OPERATIONS_GUIDE.md) for the engineering
handbook and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full
specification, including the math behind each verification script.

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

> **Implementation gap (v2.0)**: Configs 03, 04, and 05 are **disabled** in the
> current release. `compute_I_hat_2` (action variance) and `compute_I_hat_3`
> (velocity curvature) raise `NotImplementedError`. Config 05 (concordance)
> falls back to `beta_t` alone, making it effectively identical to config 02.
> Only configs **01, 02, 06, 07** produce scientifically distinct results.
> Configs 03/04/05 will be re-enabled in v2.1.

| Config | Description | LIBERO-Long IQM (†placeholder) | LIBERO-Spatial IQM (†placeholder) |
|--------|-------------|:------------------------------:|:----------------------------------:|
| 01 | BC-Chunked (baseline) | 0.520 | 0.634 |
| 02 | Cliff via β̂_t only (I^(1)) | 0.593 | 0.690 |
| 03† | Cliff via σ²_t only (I^(2)) | 0.576 | 0.676 |
| 04† | Cliff via κ_t only (I^(3)) | 0.585 | 0.663 |
| 05† | Concordance C_t (I^(1+2+3)) | 0.675 | 0.743 |
| 06 | Oracle cliff (upper bound) | 0.746 | 0.781 |
| **07** | **PACE v2: C_t + boundary reweight** | **0.692** | **0.727** |

† Results marked with † are **placeholder values** (configs disabled due to
  `NotImplementedError`). See Implementation Status table for details.

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

> All numbers in §6.1–6.5 are **placeholder** outputs from `--dry_run`
> mode (synthetic distributions hard-coded inside each script). They
> verify the analysis pipeline but do not reflect real LIBERO rollouts.
> Real numbers are produced by running each script without `--dry_run`
> against trained checkpoints; see the **Verification & Reproduction Guide**.

### §6.1 Universality — cliff occurrence across 4 policies (placeholder)

The pipeline rolls out OpenVLA-7B, π₀, BC-ACT, and Diffusion Policy on
LIBERO-Long, runs cliff detection, and measures the time gap between the
last detected cliff and the first failure step. The hypothesis is that
the gap distribution is policy-agnostic (right-skewed, mass within
5–25 steps of the last cliff). Pairwise KS test $p < 0.05$ would
confirm cliffs are not specific to any one policy.

```bash
# Placeholder (CPU, synthetic):
python scripts/phenomenon/universality.py --dry_run

# Real (requires the four checkpoints + LIBERO-Long environment):
python scripts/phenomenon/universality.py \
    --n_rollouts 50 --seeds 0 1 2 \
    --output paper_figures/universality/
```

### §6.2 Regret Scaling — δSR vs. chunk length H (placeholder)

Hypothesis: success-rate regret $\delta\text{SR} = \text{SR}_{\text{ref}} - \text{SR}$
scales with chunk length $H$ as $\delta\text{SR} \approx c \cdot H \cdot \overline{\Delta H}$
(Proposition 3 in the paper).

| Chunk length $H$ | SR (placeholder) | $\delta$SR (placeholder) |
|:--:|:--:|:--:|
| 4  | 0.829 | 0.000 |
| 8  | 0.846 | 0.000 |
| 16 | 0.712 | 0.007 |
| 32 | 0.814 | 0.025 |
| 64 | 0.718 | 0.048 |

Linear fit through origin reports $R^2 \approx 0.82$ on the placeholder
synthetic data; real expected $R^2$ is reported only after a real LIBERO
sweep.

```bash
# Placeholder (CPU, synthetic):
python scripts/phenomenon/regret_scaling.py --dry_run

# Real (requires PACE v2 checkpoint + LIBERO-Long):
python scripts/phenomenon/regret_scaling.py \
    --checkpoint checkpoints/pace_v2_libero_long \
    --H_values 4 8 16 32 64 \
    --output paper_figures/regret_scaling/
```

### §6.3 Triangulation Concordance (placeholder)

Precision/recall of each estimator and the concordance $C_t$ on a
gripper-flip oracle, $\pm 5$-step tolerance:

| Detector | Precision | Recall | F1 |
|---|:--:|:--:|:--:|
| $\hat I^{(1)}$ Bhattacharyya $\beta_t$ | 0.162 | 1.000 | 0.279 |
| $\hat I^{(2)}$ Action variance $\sigma_t^2$ | 0.155 | 1.000 | 0.269 |
| $\hat I^{(3)}$ Velocity curvature $\kappa_t$ | 0.153 | 1.000 | 0.266 |
| **Concordance $C_t$** | **1.000** | **1.000** | **1.000** |

All numbers above are placeholders. The trivial 1.000/1.000/1.000 row
for $C_t$ comes from a synthetic generator where the three estimator
spikes are perfectly correlated by construction; real numbers require
$\hat I^{(2)}$ and $\hat I^{(3)}$ to be implemented (see
[ARCHITECTURE.md §3.3–3.5](docs/ARCHITECTURE.md#33-hati2--action-ensemble-variance-sigma_t2-pending)).

```bash
python scripts/phenomenon/triangulation_concordance.py --dry_run
```

### §6.4 Alternative Cliff Detectors (placeholder)

Comparison of six detector strategies on a gripper-flip oracle,
$\pm 5$-step tolerance:

| Detector | Precision | Recall | F1 |
|---|:--:|:--:|:--:|
| Concordance $C_t$ (PACE v2) | 0.429 | 1.000 | **0.600** |
| Bhattacharyya $\beta_t$ | 0.353 | 1.000 | 0.522 |
| KL divergence $D_{\text{KL}}(p_t \| p_{t-1})$ | 0.347 | 1.000 | 0.515 |
| JS divergence $D_{\text{JS}}$ | 0.353 | 1.000 | 0.522 |
| Posterior entropy $H(\hat p)$ | 0.353 | 0.973 | 0.519 |
| BOCPD | 0.071 | 0.953 | 0.133 |

```bash
python scripts/diagnostics/diagnostic_utils/trigger_comparison.py --dry_run
```

### §6.5 Boundary Loss Ratio (placeholder)

Hypothesis: the flow-matching loss at phase-boundary steps is
materially higher than at interior steps, motivating boundary-aware
reweighting. The placeholder generator hard-codes
$\mathbb{E}_{\text{boundary}} = 0.20$, $\mathbb{E}_{\text{interior}} = 0.05$,
producing a ratio of $\approx 4$. Real values come from sweeping
boundary timesteps across LIBERO demos with a trained model.

```bash
python scripts/diagnostics/diagnostic_utils/measure_boundary_error.py --dry_run
```

---

## Inference Cost (placeholder)

> Parameter counts and NFE per inference path are derived from the
> architecture (correct by construction); the latency and Hz columns
> require a real GPU benchmark and are placeholders.

| Method | Params | NFE | Latency (placeholder) | Hz (placeholder) |
|---|:--:|:--:|:--:|:--:|
| BC-Chunked | 312.4M | 4 | 12.4 ms | 81 |
| Cliff via $\beta_t$ | 312.5M | 4 | 12.7 ms | 79 |
| Cliff via $\sigma_t^2$ | 312.5M | 36 | 91.4 ms | 11 |
| Cliff via $\kappa_t$ | 312.5M | 8 | 23.3 ms | 43 |
| Concordance $C_t$ | 312.6M | 44 | 96.5 ms | 10 |
| **PACE v2 ($C_t$ + reweight)** | 312.6M | 44 | 99.1 ms | 10 |

NFE derivation: $\beta_t$ adds 0 NFE (uses planner softmax, no extra
flow forward); $\sigma_t^2$ requires $N=8$ samples × 4-NFE shortcut + 4 NFE
base = 36; $\kappa_t$ uses two 4-NFE forwards = 8; concordance adds the
three contributions minus shared base = 44.

```bash
# Placeholder latency (CPU; the latency column is hardcoded synthetic):
python scripts/diagnostics/diagnostic_utils/measure_inference_cost.py --dry_run

# Real (requires GPU + checkpoint):
python scripts/diagnostics/diagnostic_utils/measure_inference_cost.py \
    --checkpoint checkpoints/pace_v2_libero_long \
    --device cuda --batch_size 1
```

---

## Verification & Reproduction Guide

This section is the complete recipe to take this repo from a clean
checkout to the figures and tables in the paper. It is split into
five parts in order of increasing resource demand:

- **Part A**: CPU-only verifications of the math (no GPU, no data, $\le 5$ min)
- **Part B**: Prerequisites for real experiments (GPU, datasets, checkpoints)
- **Part C**: Three-stage training pipeline
- **Part D**: Evaluation pipeline (LIBERO-Long, LIBERO-Spatial, SimplerEnv)
- **Part E**: Phenomenon experiments and figure generation

### Part A — CPU verification (no GPU, no data, ~5 min)

These verify mathematical claims on synthetic distributions and are
the gate for any code change.

```bash
# A.1 Environment
conda env create -f environment.yml && conda activate lerobot_env
pip install -r requirements.txt
pip install -e ./lerobot_policy_phaseqflow

# A.2 Unit + smoke tests (~30 s)
pytest tests/ -q                              # expects: 204 passed
bash scripts/smoke/smoke_phase_centric.sh     # 7-mode E2E, all [OK]

# A.3 Math verifications (~3 min)
python scripts/verification/verify_identifiability.py    # PA >= 0.7 (or WARN_DEGENERATE on CPU)
python scripts/verification/verify_phase_posterior.py    # peak-F1 >= 0.5
python scripts/verification/sanity_pace_a.py             # boundary FM drop >= 20%
python scripts/verification/verify_pcar_budget.py        # |rate - eps| < 0.005

# A.4 Pipeline dry-runs (verifies orchestration; numbers are synthetic)
python scripts/aggregate_ablation.py --dry_run
python scripts/phenomenon/universality.py --dry_run
python scripts/phenomenon/regret_scaling.py --dry_run
python scripts/phenomenon/triangulation_concordance.py --dry_run
python scripts/diagnostics/diagnostic_utils/trigger_comparison.py --dry_run
python scripts/diagnostics/diagnostic_utils/measure_boundary_error.py --dry_run
python scripts/diagnostics/diagnostic_utils/measure_inference_cost.py --dry_run
```

If every line in A.2–A.3 returns PASS / `[OK]` / 204 passed, the
implemented math is correct. Part A does **not** produce paper numbers.

### Part B — Prerequisites for real experiments

| Resource | Requirement | Where to get it |
|---|---|---|
| GPU | $\ge 24$ GB VRAM (RTX 5070 / A100 / H100) | local or cloud |
| LIBERO datasets | LIBERO-10 (long-horizon) and LIBERO-Spatial | https://libero-project.github.io |
| SimplerEnv | for §6 universality on RT-1 / OpenVLA simulator | https://github.com/simpler-env/SimplerEnv |
| Baseline checkpoints | OpenVLA-7B, π₀, BC-ACT, Diffusion Policy | HuggingFace; loaded via `baselines/*_adapter.py` |
| Storage | $\sim 200$ GB for datasets + checkpoints | — |

The four baseline adapters in `baselines/` already handle download +
load via `is_available()` / `load()` / `rollout()`; populate
`HF_HOME` and the adapter `load()` will fetch the weights on first
call.

Once datasets are placed under `data/libero/` and checkpoints under
`checkpoints/`, the pipeline below runs without further manual fetch.

### Part C — Three-stage training (real)

Stage 1–3 produce a single PACE v2 checkpoint. Each stage is a
separate config under `configs/train/`.

```bash
# Stage 1: pretrain multimodal tokenizer (~2 days on RTX 5070)
python scripts/training/train_dummy_batch.py \
    --config configs/train/01_pretrain_multimodal.yaml \
    --total_steps 50000 --device cuda \
    --output checkpoints/stage1_pretrain/

# Stage 2: hierarchical phase encoder + boundary-aware flow head (~3 days)
python scripts/training/train_dummy_batch.py \
    --config configs/train/02_train_phase_and_flow.yaml \
    --resume checkpoints/stage1_pretrain/last.pt \
    --total_steps 100000 --device cuda \
    --output checkpoints/stage2_phase_flow/

# Stage 3: freeze main net, calibrate PCAR / B-PCAR / concordance (~6 hours)
python scripts/training/train_dummy_batch.py \
    --config configs/train/03_finetune_replan.yaml \
    --resume checkpoints/stage2_phase_flow/last.pt \
    --total_steps 5000 --device cuda \
    --output checkpoints/pace_v2_libero_long/

# Calibration helpers (run after stage 3):
python scripts/calibration/calibrate_concordance.py \
    --checkpoint checkpoints/pace_v2_libero_long \
    --output checkpoints/pace_v2_libero_long/concordance_threshold.json
python scripts/calibration/calibrate_b_pcar.py \
    --checkpoint checkpoints/pace_v2_libero_long \
    --output checkpoints/pace_v2_libero_long/b_pcar_params.json
```

### Part D — Evaluation pipeline (real)

Evaluates the trained checkpoint on LIBERO-Long, LIBERO-Spatial,
and SimplerEnv. Each rollout records SR + PCAR replan timings to
`eval_results.json`.

```bash
# D.1 LIBERO-Long (50 episodes × 3 seeds)
bash scripts/eval/run_libero_main.sh \
    --checkpoint checkpoints/pace_v2_libero_long \
    --dataset LIBERO-10 \
    --seeds 42 123 2024 \
    --output outputs/eval/libero_long/

# D.2 LIBERO-Spatial (specificity test, §4.4 of ARCHITECTURE.md)
bash scripts/eval/run_libero_main.sh \
    --checkpoint checkpoints/pace_v2_libero_long \
    --dataset LIBERO-Spatial \
    --seeds 42 123 2024 \
    --output outputs/eval/libero_spatial/

# D.3 SimplerEnv (§6.1 universality)
bash scripts/eval/run_simpler.sh \
    --checkpoint checkpoints/pace_v2_libero_long \
    --output outputs/eval/simpler/

# D.4 LIBERO-Perturbed (robustness)
python scripts/eval/libero_perturbed.py \
    --checkpoint checkpoints/pace_v2_libero_long \
    --output outputs/eval/libero_perturbed/
```

### Part E — Ablation, phenomenon, and figures (real)

After Part C produces a checkpoint and Part D produces evaluation
JSONs, the ablation matrix and the phenomenon experiments fold both
into the paper figures.

```bash
# E.1 Seven-config ablation matrix (21 GPU runs total)
for config in configs/ablation/v2/0[1-7]_*.yaml; do
  for seed in 42 123 2024; do
    python scripts/training/train_dummy_batch.py \
        --config $config --seed $seed --device cuda \
        --total_steps 20000 \
        --output outputs/ablation_v2/$(basename $config .yaml)/seed_$seed/
  done
done

# E.2 Aggregate ablation results (rliable IQM + bootstrap CI + Wilcoxon)
python scripts/aggregate_ablation.py \
    --input_root outputs/ablation_v2/ \
    --output paper_figures/ablation_v2/
# Produces: ablation_stats.csv, ablation_table_v2.tex (paper Table 2)

# E.3 Phenomenon experiments (§6 of paper)
python scripts/phenomenon/universality.py \
    --n_rollouts 50 --seeds 0 1 2 \
    --output paper_figures/universality/
python scripts/phenomenon/regret_scaling.py \
    --checkpoint checkpoints/pace_v2_libero_long \
    --H_values 4 8 16 32 64 \
    --output paper_figures/regret_scaling/
python scripts/phenomenon/triangulation_concordance.py \
    --checkpoint checkpoints/pace_v2_libero_long \
    --output paper_figures/triangulation/

# E.4 Diagnostic measurements (§6.4, §6.5, inference cost)
python scripts/diagnostics/diagnostic_utils/trigger_comparison.py \
    --checkpoint checkpoints/pace_v2_libero_long \
    --output paper_figures/diagnostics/
python scripts/diagnostics/diagnostic_utils/measure_boundary_error.py \
    --checkpoint checkpoints/pace_v2_libero_long \
    --output paper_figures/diagnostics/boundary_ratio.json
python scripts/diagnostics/diagnostic_utils/measure_inference_cost.py \
    --checkpoint checkpoints/pace_v2_libero_long \
    --device cuda --output paper_figures/diagnostics/inference_cost.json

# E.5 Generate paper figures (all read from paper_figures/*/ produced above)
python scripts/figures/fig1_universality.py \
    --input paper_figures/universality/raw_distances.json \
    --output paper_figures/fig1_universality.pdf
python scripts/figures/fig2_method_overview.py    # purely diagrammatic
python scripts/figures/fig3_phase_visualization.py \
    --input paper_figures/triangulation/phase_vis_data.json \
    --output paper_figures/fig3_phase_visualization.pdf
python scripts/figures/fig4_regret_scaling.py \
    --input paper_figures/regret_scaling/regret_vs_H.csv \
    --output paper_figures/fig4_regret_scaling.pdf
python scripts/figures/fig5_concordance_pr_curve.py \
    --input paper_figures/diagnostics/trigger_comparison.csv \
    --output paper_figures/fig5_concordance_pr_curve.pdf

# E.6 One-shot orchestrator (does C.E1 → E.5 with default settings)
bash scripts/run_experiments.sh --checkpoint checkpoints/pace_v2_libero_long
```

### Replacing placeholders with real numbers

Once Part D and Part E complete, the README placeholders are
replaced as follows:

| Placeholder | Replace from |
|---|---|
| Table 1 (LIBERO-Long SR) | `outputs/eval/libero_long/eval_results.json` (config 07) and `outputs/eval/libero_long/*config_06*` (oracle) |
| Table 2 (ablation) | `paper_figures/ablation_v2/ablation_stats.csv` |
| §6.1 universality KS test | `paper_figures/universality/raw_distances.json` |
| §6.2 regret table | `paper_figures/regret_scaling/regret_vs_H.csv` |
| §6.3 triangulation P/R/F1 | `paper_figures/triangulation/concordance_pr.json` |
| §6.4 detector comparison | `paper_figures/diagnostics/trigger_comparison.csv` |
| §6.5 boundary ratio | `paper_figures/diagnostics/boundary_ratio.json` |
| Inference cost latency/Hz | `paper_figures/diagnostics/inference_cost.json` |

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
