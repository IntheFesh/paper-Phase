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

## Reproducing Paper Results

### Table 2 — Ablation study

```bash
# Dry run (synthetic data, no checkpoints):
python scripts/aggregate_ablation.py --dry_run

# Real run:
python scripts/aggregate_ablation.py \
    --input_root outputs/ablation_v2 \
    --output paper_figures/ablation_v2/
```

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
paper_figures/    Output directory for all figures and tables
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
