# PACE v2 — End-to-End GPU Experiment Runbook

> A standalone, copy-paste guide that takes you from a bare Linux box with one
> or more NVIDIA GPUs to **every number, table, and figure** in the PACE v2
> paper (CoRL 2027). Read this file top-to-bottom; you do not need to consult
> any other document to reproduce the paper.
>
> Companion documents (optional, deeper background only):
> - `docs/ARCHITECTURE.md` — full mathematical specification
> - `docs/OPERATIONS_GUIDE.md` — engineering handbook
> - `README.md` — high-level summary + result tables

---

## How to use this runbook

1. **Do the parts in order.** Each part assumes the previous one finished.
2. **All commands are run from the repository root** (`paper-Phase/`).
3. **Logs are written next to artifacts**; no command prints to stdout only.
4. Every command has an expected wall-clock estimate on the reference hardware
   (single A100 80 GB unless noted). Multi-GPU scaling notes are in §12.
5. When a command says "(dry run)" you can run it on a CPU-only box to
   sanity-check the pipeline. Real numbers always require GPU.

> **External helpers in §3.** The dataset and baseline-checkpoint commands in
> Part 3 (`scripts/data/download_libero.py`, `download_baselines.py`,
> `verify_baselines.py`, `build_calibration_split.py`,
> `build_libero_perturbed.py`, `download_simpler.py`,
> `verify_dataset.py`) are intended to wrap the upstream LIBERO / SimplerEnv /
> HuggingFace download flow and are **not bundled with this repo**. Use the
> shipped `scripts/data/inspect_dataset.py` and
> `scripts/data/compute_episode_lengths.py` together with the upstream
> dataset download commands described at
> <https://libero-project.github.io> and
> <https://huggingface.co/HuggingFaceVLA>. The actual training and eval
> entry points in Parts 4–11 (`scripts/train.py`,
> `scripts/training/train_dummy_batch.py`,
> `scripts/eval/run_libero_main.sh`, `scripts/aggregate_ablation.py`, …)
> are all present in the repository.

---

## Table of contents

| Part | Topic | Reference hardware time |
|:----:|-------|:-----------------------:|
| 1 | Hardware & software prerequisites | — |
| 2 | Environment setup (conda + pip) | ~20 min |
| 3 | Dataset + baseline checkpoint downloads | ~2 h (network bound) |
| 4 | CPU verification before any GPU job | ~5 min |
| 5 | Three-stage PACE v2 training | ~58 h |
| 6 | Post-training calibration (B-PCAR + concordance) | ~4 h |
| 7 | Seven-config ablation matrix (21 runs = 7 × 3 seeds) | ~120 h |
| 8 | Four evaluation suites | ~36 h |
| 9 | Section-6 phenomenon experiments | ~28 h |
| 10 | Figures, tables, and final aggregation | ~30 min |
| 11 | Artifact path reference | — |
| 12 | Multi-GPU & cluster notes | — |
| 13 | Troubleshooting | — |

Cumulative wall-clock on a single A100 80 GB: **~10 days**.
With 8× A100 (data-parallel ablation matrix), expect **~3 days**.

---

## Part 1 — Hardware & software prerequisites

### 1.1 Hardware (minimum / reference / cluster)

| Tier | GPU(s) | VRAM | RAM | Disk | Use case |
|:----:|:------:|:----:|:---:|:----:|:--------:|
| Minimum | 1 × RTX 3090 / 4090 | 24 GB | 64 GB | 500 GB SSD | Single ablation row, slow |
| **Reference** | 1 × A100 80 GB | 80 GB | 128 GB | 1 TB NVMe | Default in this runbook |
| Cluster | 8 × A100 / H100 | 8 × 80 GB | 1 TB | 4 TB NVMe | Full ablation in <3 days |

CPU: 16+ cores (data loader is the bottleneck on small GPUs).
Network: ≥ 100 Mbit/s for dataset/checkpoint downloads.

### 1.2 Software stack

- **OS**: Ubuntu 22.04 LTS (other Linux works; Windows is **not** supported)
- **CUDA**: 12.1 driver (≥ 535) — `nvidia-smi` must report a working GPU
- **conda / mamba**: Miniforge 23+ recommended
- **git** + **git-lfs** (`git lfs install` once per user)
- **ffmpeg** (for SimplerEnv video logging) — `sudo apt install -y ffmpeg`

Verify before continuing:

```bash
nvidia-smi                    # Must list at least one GPU and CUDA ≥ 12.1
conda --version               # ≥ 23.0
git lfs version               # any
ffmpeg -version | head -n 1   # any 4.x or 5.x
```

---

## Part 2 — Environment setup

All commands assume the repository is already cloned at `~/paper-Phase`:

```bash
git clone https://github.com/inthefesh/paper-phase.git ~/paper-Phase
cd ~/paper-Phase
git checkout main
```

### 2.1 Create the conda environment

```bash
conda env create -f environment.yml         # creates `lerobot_env`
conda activate lerobot_env
```

### 2.2 Install Python dependencies + the local package

```bash
pip install -r requirements.txt
pip install -e ./lerobot_policy_phaseqflow
```

### 2.3 Install evaluation harness extras

```bash
# LIBERO simulator
pip install -e third_party/LIBERO

# SimplerEnv
pip install -e third_party/SimplerEnv

# rliable (IQM + bootstrap CI for ablation aggregation)
pip install rliable
```

### 2.4 Verify the install (CPU only — must pass before continuing)

```bash
bash scripts/smoke/smoke_phase_centric.sh
pytest tests/ -q --maxfail=1
```

Expected: **204 tests passed** (or current count) in ≤ 5 minutes on CPU.
If anything fails here, do **not** proceed to GPU work — debug first.

### 2.5 Project paths used throughout this runbook

```bash
export PACE_ROOT=~/paper-Phase
export PACE_DATA=$PACE_ROOT/data                # datasets land here
export PACE_CKPT=$PACE_ROOT/checkpoints         # trained weights land here
export PACE_OUT=$PACE_ROOT/outputs              # eval/ablation outputs
export PACE_FIG=$PACE_ROOT/paper_figures        # figures + aggregated tables
mkdir -p "$PACE_DATA" "$PACE_CKPT" "$PACE_OUT" "$PACE_FIG"
```

Add these exports to your shell rc (`~/.bashrc`) so they survive across runs.

---

## Part 3 — Datasets and baseline checkpoints

> Total disk: ≈ 380 GB. Run all of this once; everything below depends on it.

### 3.1 LIBERO-10 (LIBERO-Long, primary benchmark)

```bash
python scripts/data/download_libero.py \
    --suite libero_10 \
    --output_dir "$PACE_DATA/libero_10"
# ≈ 92 GB, ≈ 45 min on a 100 Mbit/s line
```

Sanity check:

```bash
python scripts/data/verify_dataset.py --dataset libero_10 \
    --root "$PACE_DATA/libero_10"
# Expected: "OK — 50 tasks × 50 demos verified"
```

### 3.2 LIBERO-Spatial (transfer benchmark)

```bash
python scripts/data/download_libero.py \
    --suite libero_spatial \
    --output_dir "$PACE_DATA/libero_spatial"
# ≈ 38 GB, ≈ 20 min
```

### 3.3 LIBERO-Perturbed (closed-loop replanning eval, §5.4)

```bash
python scripts/data/build_libero_perturbed.py \
    --base "$PACE_DATA/libero_10" \
    --output "$PACE_DATA/libero_perturbed" \
    --perturbations push,occlude,relocate \
    --seeds 0 1 2
# ≈ 12 GB, ≈ 30 min (it re-renders demos)
```

### 3.4 SimplerEnv (real-to-sim eval, §5.3)

```bash
python scripts/data/download_simpler.py \
    --tasks google_robot,bridge \
    --output_dir "$PACE_DATA/simpler"
# ≈ 18 GB, ≈ 15 min
```

### 3.5 Baseline checkpoints (reproducibility)

The four baselines are loaded from public weights, not retrained.

```bash
python scripts/data/download_baselines.py \
    --models bc_act,diffusion_policy,openvla_7b,pi0 \
    --output_dir "$PACE_CKPT/baselines"
# ≈ 220 GB (OpenVLA-7B dominates), ≈ 90 min
```

Verify all four loaded successfully:

```bash
python scripts/data/verify_baselines.py --root "$PACE_CKPT/baselines"
# Expected: "OK — 4/4 baseline checkpoints loadable"
```

### 3.6 Held-out calibration split

Stage 3 (replanning calibration) needs a dedicated held-out split that is
**disjoint from the LIBERO-Long training split** to avoid leaking calibration
data into evaluation:

```bash
python scripts/data/build_calibration_split.py \
    --input "$PACE_DATA/libero_10" \
    --output "$PACE_DATA/libero_long_holdout" \
    --frac 0.10 --seed 42
# ≈ 9 GB, ≈ 5 min
```

---

## Part 4 — CPU verification before any GPU job

Before burning GPU hours, confirm the analytical math, the optimization
identities, and all unit tests pass on CPU. These checks are cheap and catch
~80% of bugs that would otherwise surface 12 hours into a training run.

```bash
# Identifiability + posterior + PCAR analytical sanity (all CPU)
python scripts/verification/verify_identifiability.py
python scripts/verification/verify_phase_posterior.py
python scripts/verification/verify_pcar_budget.py
python scripts/verification/sanity_pace_a.py

# Full unit-test suite
pytest tests/ -q
```

All four scripts must print **OK**, and pytest must report 0 failures.

---

## Part 5 — Three-stage PACE v2 training

The full PACE v2 main path uses the curriculum defined in
`configs/train/{01,02,03}_*.yaml`. Each stage warm-starts from the previous
stage's last checkpoint and **only updates the components listed in the YAML**.

| Stage | Config | Trains | Frozen | Steps | A100 80 GB time | Peak VRAM |
|:-----:|--------|--------|--------|:-----:|:---------------:|:---------:|
| 1 | `01_pretrain_multimodal.yaml` | vision adapter, x-attn backbone | vision encoder | 200 k | ~14 h | 28 GB |
| 2 | `02_train_phase_and_flow.yaml` | hierarchical FSQ + flow head + boundary reweight | vision encoder | 400 k | ~38 h | 62 GB |
| 3 | `03_finetune_replan.yaml` | replanning calibrator only | everything else | 20 k | ~2 h (incl. rollouts) | 18 GB |

> Stages 1 and 2 are gradient updates; **stage 3 is calibration-only** — it
> runs rollouts on `libero_long_holdout` and fits B-PCAR + concordance
> thresholds.

### 5.1 Stage 1 — pretrain multimodal tokenizer

```bash
python scripts/train.py \
    --config configs/train/01_pretrain_multimodal.yaml \
    --data_root  "$PACE_DATA/libero_10" \
    --output_dir "$PACE_CKPT/stage1_pretrain" \
    --seed 0 \
    --batch_size 64 --num_workers 8 \
    --max_steps 200000 \
    --log_dir "$PACE_OUT/logs/stage1"
```

Done when:

```bash
ls "$PACE_CKPT/stage1_pretrain"/last.pt          # exists
python scripts/verification/verify_stage_outputs.py --stage 1 \
    --ckpt "$PACE_CKPT/stage1_pretrain/last.pt"  # prints "OK — stage 1 invariants hold"
```

### 5.2 Stage 2 — phase identifiability + flow head

```bash
python scripts/train.py \
    --config configs/train/02_train_phase_and_flow.yaml \
    --init_from  "$PACE_CKPT/stage1_pretrain/last.pt" \
    --data_root  "$PACE_DATA/libero_10" \
    --output_dir "$PACE_CKPT/stage2_phase_flow" \
    --seed 0 \
    --batch_size 32 --num_workers 8 \
    --max_steps 400000 \
    --log_dir "$PACE_OUT/logs/stage2"
```

The boundary-aware flow loss is gated by `use_boundary_reweight: true` in the
YAML; it adds ≈ 4% to per-step time but is required for the ablation #07 row.

Done when:

```bash
python scripts/verification/verify_stage_outputs.py --stage 2 \
    --ckpt "$PACE_CKPT/stage2_phase_flow/last.pt"
# Prints:
#   InfoNCE_macro / micro identifiability ≥ 0.9 ✔
#   β_t spectrum non-degenerate            ✔
#   Flow EMA gap < 5 × 10⁻³                ✔
```

### 5.3 Stage 3 — replanning calibration

This stage **does not train the policy network**; it freezes everything and
only fits the PCAR threshold and concordance window via held-out rollouts:

```bash
python scripts/train.py \
    --config configs/train/03_finetune_replan.yaml \
    --init_from  "$PACE_CKPT/stage2_phase_flow/last.pt" \
    --data_root  "$PACE_DATA/libero_long_holdout" \
    --output_dir "$PACE_CKPT/stage3_replan" \
    --seed 0 \
    --max_steps 20000 \
    --log_dir "$PACE_OUT/logs/stage3"
```

The script writes the calibrated thresholds to:

```
$PACE_CKPT/stage3_replan/calibration.json
$PACE_CKPT/stage3_replan/last.pt              # full checkpoint with calibration baked in
```

This `last.pt` is the **PACE v2 final checkpoint** used everywhere below.
Convenient alias:

```bash
export PACE_FINAL=$PACE_CKPT/stage3_replan/last.pt
```

### 5.4 Resuming / multi-seed training

For each seed `s ∈ {0, 1, 2}` repeat all three stages with
`--seed s --output_dir $PACE_CKPT/stage{1,2,3}_seed{s}`. Total cost: 3× the
table above (≈ 174 h on a single A100). On 3× A100 you can launch the seeds
in parallel; see §12.

---

## Part 6 — Post-training calibration (standalone runs)

Stage 3 above already produces `calibration.json`. Two standalone scripts
exist for **re-running calibration** on a different held-out split, sweeping
hyperparameters, or recovering after a corrupted stage-3 run.

### 6.1 B-PCAR threshold (Bayesian PCAR Beta-mixture trigger)

```bash
python scripts/calibration/calibrate_b_pcar.py \
    --ckpt        "$PACE_FINAL" \
    --data_root   "$PACE_DATA/libero_long_holdout" \
    --output_dir  "$PACE_OUT/calib/b_pcar" \
    --eps_grid    0.05,0.10,0.15,0.20 \
    --n_rollouts  100
# ≈ 1.5 h on A100. Writes b_pcar_curve.json + the chosen ε* to console.
```

Pick the ε that minimises `δSR` on the held-out set; this becomes
`pcar_trigger_budget_eps` in the eval config.

### 6.2 Concordance window + threshold

```bash
python scripts/calibration/calibrate_concordance.py \
    --ckpt        "$PACE_FINAL" \
    --data_root   "$PACE_DATA/libero_long_holdout" \
    --output_dir  "$PACE_OUT/calib/concordance" \
    --window_grid 25,50,75,100 \
    --thresh_grid 0.30,0.40,0.50 \
    --n_rollouts  100
# ≈ 2 h on A100. Writes concordance_grid.json.
```

The script picks `(W*, τ*)` that maximise F1 against the gripper-flip oracle.
These values are written into `$PACE_FINAL`'s metadata in-place.

### 6.3 Sanity check after calibration

```bash
python scripts/diagnostics/diagnostic_phase_centric.py \
    --ckpt "$PACE_FINAL" \
    --data_root "$PACE_DATA/libero_long_holdout" \
    --output "$PACE_OUT/diag/post_calib.json"
# Expected: trigger rate ≈ 0.5 × ε*, F1 ≥ 0.55 on holdout
```

---

## Part 7 — Three-config ablation matrix (Table 2, CoRL submission)

The CoRL ablation in `configs/ablation/v2/` has **3 configurations × 3 seeds = 9
training runs**, each followed by an eval rollout. All three share the same
vision encoder + tokenizer from stage 1, so you re-use
`$PACE_CKPT/stage1_pretrain` as the warm start for every row.

> **Important**: ablation configs are *training* configs. Each ablation run
> follows the same recipe as stage 2 above (and adds stage 3 calibration for
> configs that use replanning).

### 7.1 Configuration matrix

| ID | Config file | Cliff signal | Boundary reweight | Replanning |
|:--:|-------------|:-------------:|:------------------:|:-----------:|
| 01 | `01_bc_chunked.yaml` | none | no | no |
| 02 | `02_cliff_via_beta_only.yaml` | β_t (I^(1)) | no | yes |
| 07 | `07_cliff_concordance_with_boundary_reweight.yaml` | C_t (concordance) | **yes** | yes |

The gap 01 → 02 isolates the value of any cliff signal; 02 → 07 isolates
multi-estimator fusion + boundary-aware loss. All three estimators
(`compute_I_hat_1/2/3` and `compute_concordance_C`) are implemented in v2.1
and exercised by `tests/test_cliff_estimators.py`; Ablation 07 invokes all
three internally via the concordance signal.

```bash
# Ablation configs used by configs/cloud/phaseqflow_cloud_accelerate.sh
RUNNABLE_CONFIGS=(
    "01_bc_chunked"
    "02_cliff_via_beta_only"
    "07_cliff_concordance_with_boundary_reweight"
)
```

### 7.1a Configs not in CoRL submission

The YAMLs `03_cliff_via_var_only.yaml`, `04_cliff_via_curvature_only.yaml`,
`05_cliff_concordance.yaml`, and `06_oracle_cliff.yaml` remain in the
codebase as reference but are not part of the submission scope: 03/04 are
covered by unit tests, 05 is subsumed by 07, and 06 is an oracle upper
bound. They can be invoked manually with
`python scripts/training/train_dummy_batch.py` if needed.

### 7.2 Single ablation row (one config × one seed)

```bash
ABL_ID=05_cliff_concordance         # change this per row
SEED=0                               # change this per seed

python scripts/train.py \
    --config configs/ablation/v2/${ABL_ID}.yaml \
    --init_from "$PACE_CKPT/stage1_pretrain/last.pt" \
    --data_root "$PACE_DATA/libero_10" \
    --output_dir "$PACE_CKPT/ablation_v2/${ABL_ID}/seed${SEED}" \
    --seed "$SEED" \
    --batch_size 32 --num_workers 8 \
    --max_steps 400000 \
    --log_dir "$PACE_OUT/logs/ablation_v2/${ABL_ID}/seed${SEED}"
# ≈ 38 h per row on A100 80 GB
```

For configs with replanning (02–07), additionally run the calibration step:

```bash
python scripts/train.py \
    --config configs/train/03_finetune_replan.yaml \
    --init_from "$PACE_CKPT/ablation_v2/${ABL_ID}/seed${SEED}/last.pt" \
    --data_root "$PACE_DATA/libero_long_holdout" \
    --output_dir "$PACE_CKPT/ablation_v2/${ABL_ID}/seed${SEED}_calib" \
    --seed "$SEED" --max_steps 20000
```

### 7.3 Evaluation rollout per row

```bash
python scripts/eval/simpler.py \
    --ckpt "$PACE_CKPT/ablation_v2/${ABL_ID}/seed${SEED}_calib/last.pt" \
    --suite libero_long \
    --n_episodes 50 \
    --output "$PACE_OUT/ablation_v2/${ABL_ID}/seed${SEED}/eval.json"
# ≈ 35 min per (config, seed) on LIBERO-Long
```

### 7.4 Run the entire matrix end-to-end

A driver script wraps the loop:

```bash
bash scripts/run_experiments.sh \
    --mode ablation_v2 \
    --seeds 0 1 2 \
    --suites libero_long libero_spatial \
    --output_root "$PACE_OUT/ablation_v2"
# ≈ 120 h on a single A100 (sequential).
# On 8× A100 in parallel: ≈ 18 h (one (config, seed) per GPU).
```

### 7.5 Aggregate Table 2

```bash
python scripts/aggregate_ablation.py \
    --input_root "$PACE_OUT/ablation_v2" \
    --output_dir "$PACE_FIG/ablation_v2/" \
    --metric success_rate \
    --bootstrap_n 10000
# Writes:
#   $PACE_FIG/ablation_v2/table2_libero_long.tex
#   $PACE_FIG/ablation_v2/table2_libero_spatial.tex
#   $PACE_FIG/ablation_v2/iqm_ci.json   (machine-readable)
```

The script computes IQM ± 95% bootstrap CI per row and runs Wilcoxon p-values
between every row and the BC-Chunked baseline. These are the numbers that
populate Table 2 in the paper and the README.

---

## Part 8 — Four evaluation suites (Table 1)

PACE v2 is evaluated against four published baselines on three benchmarks plus
a closed-loop perturbation suite. All four runs use **the same final
checkpoint** `$PACE_FINAL` so the only thing changing is the environment.

| Suite | Script | Episodes | A100 time |
|-------|--------|:--------:|:---------:|
| LIBERO-Long | `scripts/eval/run_libero_main.sh` | 3 × 50 = 150 | ~6 h |
| LIBERO-Spatial | `scripts/eval/run_libero_main.sh --suite libero_spatial` | 3 × 50 = 150 | ~5 h |
| SimplerEnv | `scripts/eval/run_simpler.sh` | 3 × 100 = 300 | ~14 h |
| LIBERO-Perturbed (closed-loop) | `scripts/eval/libero_perturbed.py` | 3 × 50 × 3 perturbations = 450 | ~11 h |

### 8.1 LIBERO-Long (primary)

```bash
bash scripts/eval/run_libero_main.sh \
    --ckpt   "$PACE_FINAL" \
    --suite  libero_long \
    --seeds  0 1 2 \
    --n_episodes 50 \
    --output "$PACE_OUT/eval/libero_long"
```

Compare against baselines:

```bash
for BL in bc_act diffusion_policy openvla_7b pi0; do
    bash scripts/eval/run_libero_main.sh \
        --ckpt "$PACE_CKPT/baselines/$BL" \
        --suite libero_long --seeds 0 1 2 --n_episodes 50 \
        --output "$PACE_OUT/eval/libero_long/$BL"
done
```

### 8.2 LIBERO-Spatial (transfer)

```bash
bash scripts/eval/run_libero_main.sh \
    --ckpt   "$PACE_FINAL" \
    --suite  libero_spatial \
    --seeds  0 1 2 --n_episodes 50 \
    --output "$PACE_OUT/eval/libero_spatial"
```

### 8.3 SimplerEnv (real-to-sim)

```bash
bash scripts/eval/run_simpler.sh \
    --ckpt   "$PACE_FINAL" \
    --tasks  google_robot,bridge \
    --seeds  0 1 2 --n_episodes 100 \
    --output "$PACE_OUT/eval/simpler"
```

### 8.4 LIBERO-Perturbed (closed-loop replanning)

This is the suite that exercises PCAR; baselines that don't replan must
re-execute the original chunk after each perturbation.

```bash
python scripts/eval/libero_perturbed.py \
    --ckpt    "$PACE_FINAL" \
    --data    "$PACE_DATA/libero_perturbed" \
    --seeds   0 1 2 --n_episodes 50 \
    --perturbations push,occlude,relocate \
    --output  "$PACE_OUT/eval/libero_perturbed"
```

### 8.5 Aggregate Table 1

```bash
python scripts/aggregate_results.py \
    --input_root "$PACE_OUT/eval" \
    --output_dir "$PACE_FIG/main_table" \
    --bootstrap_n 10000
# Writes:
#   $PACE_FIG/main_table/table1_main.tex
#   $PACE_FIG/main_table/table1_perturbed.tex
#   $PACE_FIG/main_table/iqm_ci.json
```

---

## Part 9 — Section 6 phenomenon experiments (CoRL submission)

The CoRL submission keeps only **§6.2** and **§6.5** as real experiments —
the two that directly validate theoretical claims of the paper. §6.1
(universality across baselines) is dropped because the baseline checkpoints
exceed the disk budget; §6.3 (triangulation) is redundant with the
01 → 02 → 07 ablation gap; §6.4 (trigger comparison) is synthetic by design
and adds no real-data evidence.

All four dropped scripts (`universality.py`, `triangulation_concordance.py`,
`trigger_comparison.py`, `measure_inference_cost.py`) are still in the
codebase; they are simply not invoked by `configs/cloud/phaseqflow_cloud_accelerate.sh`.

### 9.1 §6.2 Regret scaling (Figure 4)

δSR vs. chunk length H — validates the regret bound:

```bash
for H in 4 8 16 32 64; do
    python scripts/phenomenon/regret_scaling.py \
        --ckpt "$PACE_FINAL" --suite libero_long \
        --chunk_length $H --n_rollouts 100 --seeds 0 1 2 \
        --output "$PACE_FIG/regret_scaling/H${H}.json"
done
# ≈ 6 h total
```

### 9.2 §6.5 Boundary loss ratio

```bash
python scripts/diagnostics/diagnostic_utils/measure_boundary_error.py \
    --ckpt "$PACE_FINAL" \
    --data_root "$PACE_DATA/libero_10" \
    --output "$PACE_FIG/boundary_loss/ratio.json"
# ≈ 4 h. Writes E_boundary / E_interior ratio.
```

### 9.6 Inference cost measurement

```bash
python scripts/diagnostics/diagnostic_utils/measure_inference_cost.py \
    --ckpt "$PACE_FINAL" \
    --variants bc_chunked,beta,variance,curvature,concordance,full \
    --output "$PACE_FIG/inference_cost/cost.json"
# ≈ 30 min; latency in ms, NFE, Hz on the host GPU.
```

---

## Part 10 — Figures and final tables

Once Parts 7–9 finish, all five figures plus the LaTeX tables are
deterministic from the JSON artifacts.

```bash
# Figure 1 — universality
python scripts/figures/fig1_universality.py \
    --input "$PACE_FIG/universality/raw_distances.json" \
    --output "$PACE_FIG/fig1_universality.pdf"

# Figure 2 — method overview (vector graphics; no inputs)
python scripts/figures/fig2_method_overview.py \
    --output "$PACE_FIG/fig2_method.pdf"

# Figure 3 — phase-token visualisation (uses stage 2 ckpt)
python scripts/figures/fig3_phase_visualization.py \
    --ckpt "$PACE_CKPT/stage2_phase_flow/last.pt" \
    --data_root "$PACE_DATA/libero_10" \
    --output "$PACE_FIG/fig3_phases.pdf"

# Figure 4 — regret scaling
python scripts/figures/fig4_regret_scaling.py \
    --input_dir "$PACE_FIG/regret_scaling" \
    --output "$PACE_FIG/fig4_regret.pdf"

# Figure 5 — concordance PR curve
python scripts/figures/fig5_concordance_pr_curve.py \
    --input "$PACE_FIG/triangulation/pr_curves.json" \
    --output "$PACE_FIG/fig5_concordance.pdf"
```

Final aggregation (rebuilds every `*.tex` table from the JSON artifacts):

```bash
python scripts/aggregate_results.py   --input_root "$PACE_OUT/eval"      --output_dir "$PACE_FIG/main_table"
python scripts/aggregate_ablation.py  --input_root "$PACE_OUT/ablation_v2" --output_dir "$PACE_FIG/ablation_v2"
```

Drop the resulting PDFs and `.tex` files into the paper repository as-is —
no manual edits are needed.

---

## Part 11 — Artifact path reference

Quick lookup for every artifact this runbook produces.

### 11.1 Checkpoints

| Path | Produced by | Used by |
|------|-------------|---------|
| `$PACE_CKPT/stage1_pretrain/last.pt` | §5.1 | §5.2 init, §7 init |
| `$PACE_CKPT/stage2_phase_flow/last.pt` | §5.2 | §5.3 init, Fig. 3 |
| `$PACE_CKPT/stage3_replan/last.pt` | §5.3 | **= `$PACE_FINAL`**, §8, §9 |
| `$PACE_CKPT/ablation_v2/<id>/seed<s>/last.pt` | §7.2 | §7.3 eval |
| `$PACE_CKPT/baselines/<bc_act\|...>` | §3.5 | §8 |

### 11.2 Datasets

| Path | Produced by | Used by |
|------|-------------|---------|
| `$PACE_DATA/libero_10` | §3.1 | §5, §7, §8.1 |
| `$PACE_DATA/libero_spatial` | §3.2 | §8.2 |
| `$PACE_DATA/libero_perturbed` | §3.3 | §8.4 |
| `$PACE_DATA/simpler` | §3.4 | §8.3 |
| `$PACE_DATA/libero_long_holdout` | §3.6 | §5.3, §6 |

### 11.3 Result JSON / TeX

| Path | Produced by | Maps to |
|------|-------------|---------|
| `$PACE_FIG/main_table/table1_main.tex` | §8.5 | Table 1 |
| `$PACE_FIG/main_table/table1_perturbed.tex` | §8.5 | Closed-loop column |
| `$PACE_FIG/ablation_v2/table2_libero_long.tex` | §7.5 | Table 2 (LIBERO-Long) |
| `$PACE_FIG/ablation_v2/table2_libero_spatial.tex` | §7.5 | Table 2 (LIBERO-Spatial) |
| `$PACE_FIG/universality/raw_distances.json` | §9.1 | Fig. 1 |
| `$PACE_FIG/regret_scaling/H{4,8,16,32,64}.json` | §9.2 | §6.2 table + Fig. 4 |
| `$PACE_FIG/triangulation/pr_curves.json` | §9.3 | §6.3 table + Fig. 5 |
| `$PACE_FIG/trigger_comparison/comparison_table.tex` | §9.4 | §6.4 table |
| `$PACE_FIG/boundary_loss/ratio.json` | §9.5 | §6.5 ratio |
| `$PACE_FIG/inference_cost/cost.json` | §9.6 | Inference-cost table |

### 11.4 Figures

| Path | Produced by | Paper reference |
|------|-------------|-----------------|
| `$PACE_FIG/fig1_universality.pdf` | §10 | Figure 1 |
| `$PACE_FIG/fig2_method.pdf` | §10 | Figure 2 |
| `$PACE_FIG/fig3_phases.pdf` | §10 | Figure 3 |
| `$PACE_FIG/fig4_regret.pdf` | §10 | Figure 4 |
| `$PACE_FIG/fig5_concordance.pdf` | §10 | Figure 5 |

---

## Part 12 — Multi-GPU & cluster notes

### 12.1 Single-node multi-GPU (data parallel)

All training scripts honour `torchrun`. For 4× A100 in one node:

```bash
torchrun --standalone --nproc_per_node 4 \
    scripts/train.py \
    --config configs/train/02_train_phase_and_flow.yaml \
    --init_from "$PACE_CKPT/stage1_pretrain/last.pt" \
    --data_root "$PACE_DATA/libero_10" \
    --output_dir "$PACE_CKPT/stage2_phase_flow" \
    --seed 0 --batch_size 32 --num_workers 8 \
    --max_steps 400000
```

Effective batch = 32 × 4 = 128. Wall-clock drops from 38 h to ≈ 11 h.

### 12.2 Embarrassingly-parallel ablation

The ablation matrix is **21 independent jobs**. The provided driver

```bash
bash scripts/run_experiments.sh --mode ablation_v2 --gpus 0,1,2,3,4,5,6,7
```

assigns one (config × seed) per GPU and rotates as GPUs free. On 8× A100 the
matrix completes in ≈ 18 h.

### 12.3 SLURM template

A reference SLURM submission script lives at
`scripts/slurm/submit_ablation.sbatch` — see comments at the top for queue
and account placeholders.

---

## Part 13 — Troubleshooting

| Symptom | Most likely cause | Fix |
|--------|-------------------|-----|
| `NotImplementedError: compute_I_hat_2/3/concordance_C` | Pending design decisions in `MIGRATION_NOTES.md` §1 / §2 | Until resolved, only ablation rows 01, 02, 06 produce real numbers |
| Stage 2 OOM at batch 32 on 24 GB GPU | Boundary reweight + InfoNCE memory | Drop to `--batch_size 16 --grad_accum 2`; +2 h wall-clock |
| `No module named 'libero'` | LIBERO submodule not installed | `pip install -e third_party/LIBERO` (see §2.3) |
| SimplerEnv hangs on rollout | `EGL` not initialised | `export PYOPENGL_PLATFORM=egl` before launching |
| Stage 3 calibration produces nan thresholds | `libero_long_holdout` empty / overlaps train | Re-run §3.6 with a different `--seed` |
| `aggregate_results.py` errors on missing keys | Eval rollouts incomplete | Confirm every `eval.json` exists for every (suite, ckpt, seed) |
| Multi-GPU run slower than 1× | NCCL stuck on default network iface | `export NCCL_SOCKET_IFNAME=<bond0|eth0>` |
| `verify_phase_posterior.py` reports β-spectrum degenerate | Stage 2 collapsed to single phase | Re-train with InfoNCE weight ≥ 0.1; check stage-1 init quality |

### Implementation status reminder

Three of the six cliff-detector configurations and the concordance fusion
itself currently raise `NotImplementedError` — see
`lerobot_policy_phaseqflow/phase_centric/cliff_estimators.py` and
`MIGRATION_NOTES.md`. Until those land:

- **Real numbers possible**: ablation rows 01 (`bc_chunked`), 02 (`β_t`),
  06 (`oracle`); §6.5 boundary loss ratio; §9.6 inference cost.
- **Blocked**: ablation rows 03, 04, 05, 07; §6.3 triangulation; §6.4 detector
  comparison (concordance row only).

Run the unblocked rows now; the blocked rows will be re-runnable as soon as
the pending decisions land, without changes to this runbook.

### Storage-constrained deployment (< 250GB disk)

For environments with limited disk space (e.g., 250GB on cloud GPUs):

1. **Baseline checkpoints**: Skip `openvla` (~14GB) and `pi0` (~4GB).
   Use `bc_act` + `diffusion_policy` (~1.5GB total) for universality experiment.

2. **Ablation matrix**: Run only configs 01, 02, 06, 07 (12 GPU runs).
   Configs 03/04/05 are disabled anyway due to `NotImplementedError`.

3. **Phase B (optional)**: After Phase A completes, delete LIBERO datasets
   (~130GB freed) before downloading baseline checkpoints for universality.

4. **SimplerEnv**: Skip entirely (requires ManiSkill2, not core to paper claims).

5. **LIBERO-Perturbed**: Skip (supplementary, not required for Table 1/2).

Minimum viable experiment set for CoRL submission:
- Stage 1+2+3 training (~27h on H800)
- Ablation 01/02/06/07 × 3 seeds (~35h)
- §6.2 Regret Scaling (~3h, real data)
- §6.5 Boundary Loss Ratio (~1h, real data)
- §6.1/6.3/6.4 dry_run (pipeline validation)
Total: ~70h ≈ ¥620 on H800

---

*End of runbook. If something here is wrong or stale, fix it here first —
this file is the single source of truth for "how to reproduce PACE v2".*

