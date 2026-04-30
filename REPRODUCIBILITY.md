# Reproducibility Statement

This document describes the hardware, software, and procedural conditions
required to reproduce the results reported in the PACE v2 paper.

---

## 1. Hardware

| Component | Specification |
|-----------|---------------|
| GPU       | NVIDIA RTX 5070 (12 GB VRAM) × 1 (single-GPU training) |
| CPU       | 16-core AMD / Intel, ≥ 64 GB RAM |
| Storage   | ≥ 500 GB NVMe SSD (datasets + checkpoints) |

Multi-GPU scaling has been validated on 8 × A100-80GB via
`configs/cloud/phaseqflow_cloud_accelerate.sh`. The single-GPU
configuration uses `micro_batch=2, grad_accum=32` for an effective
batch size of 64, matching the cloud run.

---

## 2. Training Time

| Stage | Script | Single GPU | 8 × A100 |
|-------|--------|-----------|---------|
| Stage 1: Pretrain multimodal | `configs/train/01_pretrain_multimodal.yaml` | ~18 h | ~3 h |
| Stage 2: Phase + flow | `configs/train/02_train_phase_and_flow.yaml` | ~24 h | ~4 h |
| Stage 3: Finetune replan | `configs/train/03_finetune_replan.yaml` | ~6 h  | ~1 h |

---

## 3. Dataset Preparation

### LIBERO-Long

```bash
# Download via HuggingFace
python scripts/data/inspect_dataset.py \
    --dataset HuggingFaceVLA/smol-libero --n 5

# Verify episode lengths
python scripts/data/compute_episode_lengths.py \
    --dataset HuggingFaceVLA/smol-libero \
    --out artifacts/episode_lengths/libero_long_lengths.json
```

### LIBERO-Spatial

Same procedure; substitute the spatial benchmark dataset identifier.

---

## 4. Training

```bash
# Stage 1
python scripts/train.py \
    --stage configs/train/01_pretrain_multimodal.yaml

# Stage 2
python scripts/train.py \
    --stage configs/train/02_train_phase_and_flow.yaml

# Stage 3
python scripts/train.py \
    --stage configs/train/03_finetune_replan.yaml
```

Use `--smoke_mode` for a 3-step sanity check before a full run.

---

## 5. Figures and Tables

| Paper item | Script | Config |
|------------|--------|--------|
| Figure 1 | `scripts/figures/fig1_universality.py` | — |
| Figure 2 | `scripts/figures/fig2_method_overview.py` | — |
| Figure 3 | `scripts/figures/fig3_phase_visualization.py` | — |
| Figure 4 | `scripts/figures/fig4_regret_scaling.py` | — |
| Figure 5 | `scripts/figures/fig5_concordance_pr_curve.py` | — |
| Table 2 (ablation) | `scripts/aggregate_ablation.py` | `configs/ablation/v2/` |
| Universality (§6.1) | `scripts/phenomenon/universality.py` | — |
| Regret scaling (§6.2) | `scripts/phenomenon/regret_scaling.py` | — |
| Triangulation (§6.3) | `scripts/phenomenon/triangulation_concordance.py` | — |
| LIBERO-Perturbed (§6.4) | `scripts/eval/libero_perturbed.py` | — |
| SimplerEnv (§6.5) | `scripts/eval/simpler.py` | — |

All scripts support `--dry_run` for inspection without checkpoints.

---

## 6. Checkpoint Download

Trained checkpoints will be released on HuggingFace Hub upon paper acceptance.
Placeholder identifier: `[authors]/pace-v2-libero-long`.

---

## 7. Known Non-determinism Sources

| Source | Impact | Mitigation |
|--------|--------|-----------|
| CUDA non-deterministic ops | ±0.5% SR across seeds | Report mean ± std over 3 seeds |
| FSQ straight-through gradients | Discrete quantization noise | Hierarchical structure reduces sensitivity |
| Concordance rank ties | Mid-rank convention stabilises ties | See `cliff_detection/concordance.py` |
| PCAR budget enforcement | Trigger count varies ±1 per episode | Budget ε = 0.1 provides headroom |

To reproduce exactly: set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`torch.use_deterministic_algorithms(True)`. Note that deterministic mode
increases training time by approximately 15%.

---

## 8. Smoke Test

The following command verifies the installation without any dataset or checkpoint:

```bash
bash scripts/smoke/smoke_phase_centric.sh  # 7 modes, all must pass
python -m pytest tests/ -q                 # 204 tests must pass
```

---

## 9. Known Limitations (v2.0)

### 9.1 Unimplemented cliff estimators

`compute_I_hat_2` and `compute_I_hat_3` are not yet implemented
(both raise `NotImplementedError`). This affects:

- **Ablation configs 03, 04**: Disabled. Results from these configs are
  synthetic placeholders and MUST NOT be cited as real measurements.
- **Ablation config 05**: Partially disabled. Concordance falls back to
  β_t (I^(1)) only, making it nearly equivalent to config 02.
- **§6.3 Triangulation**: Cannot run with real data (C_t incomplete).
  Pipeline validation uses `--dry_run` mode.

### 9.2 Storage-constrained reproducibility path

For environments with ≤ 250GB storage (e.g., cloud GPU instances):

```bash
# Phase A: Core experiments (requires ~204GB peak)
# LIBERO-10 (92GB) + LIBERO-Spatial (38GB) + checkpoints (30GB) + outputs (44GB)
bash 01_cpu_verify.sh
bash 02_download_data.sh      # Downloads LIBERO-10 + LIBERO-Spatial
bash 03_main_training.sh       # Stage 1+2+3
bash 04_ablation.sh            # Configs 01/02/06/07 × 3 seeds only
bash 05_phenomenon.sh          # §6.2 real, §6.1/6.3/6.4 dry_run

# Phase B: Universality with real baselines (optional)
# Delete LIBERO datasets first to free ~130GB
rm -rf $PACE_DATA/libero_10 $PACE_DATA/libero_spatial
bash 07_phase_b_universality.sh   # Downloads bc_act+diffusion_policy (~1.5GB)
```

### 9.3 Determinism

Configs 03/04/05 are non-deterministic in an additional sense: their
placeholder results are generated by `_synthetic_results()` in
`scripts/aggregate_ablation.py`, which uses a fixed seed (42) but
produces values that do NOT reflect actual model behavior.
