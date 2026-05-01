#!/usr/bin/env bash
# CoRL 2025 — PACE v2 (PhaseQFlow++) full experiment sweep
#
# 13 training runs × configured seeds = main results + ablation matrix:
#
#   base_shortcut_fm         × seeds {42, 43, 44}   — flow-matching baseline
#   pace_concordance         × seeds {42, 43, 44}   — full PACE with concordance
#   pace_concordance_bdy     × seeds {42, 43, 44}   — + boundary reweighting
#   abl_only_I1              × seed  42              — ablation: Bhattacharyya only
#   abl_only_I2              × seed  42              — ablation: action-variance only
#   abl_only_I3              × seed  42              — ablation: velocity-curvature only
#   abl_oracle_cliff         × seed  42              — ablation: oracle cliff signal
#
# Evaluation: 7 experiments × configured rollout counts (10,428 total rollouts).
#
# Prerequisites:
#   - CUDA GPU with ≥ 24 GB VRAM (H800 / A100 recommended)
#   - DATA_ROOT pointing to a LIBERO-10 dataset directory
#   - bitsandbytes >= 0.41 (for PagedAdamW8bit)
#   - Run preflight first:  bash scripts/run_autodl_pipeline.sh preflight
#   - Source autobatch env: source .batch_env
#
# Outputs land under $PACE_RUN_DIR with one tar.gz snapshot per experiment.

set -euo pipefail

# ------------------------------------------------------------------ paths
export PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PACE_OUT="${PACE_OUT:-$PACE_ROOT/outputs}"
export PACE_DATA="${PACE_DATA:-$PACE_ROOT/data}"
export PACE_CKPT="${PACE_CKPT:-$PACE_ROOT/checkpoints}"
export PACE_RUN_DIR="${PACE_RUN_DIR:-$PACE_OUT/corl_run_$(date +%Y%m%d_%H%M%S)}"
export PACE_SNAPSHOT_DIR="${PACE_SNAPSHOT_DIR:-/root/autodl-tmp/snapshots}"
mkdir -p "$PACE_RUN_DIR" "$PACE_RUN_DIR/logs" "$PACE_SNAPSHOT_DIR"

# ------------------------------------------------------------------ data
DATA_ROOT="${DATA_ROOT:-${PACE_DATA}/libero_10}"
if [[ ! -d "$DATA_ROOT" ]] && [[ -z "${PACE_ALLOW_MISSING_DATA:-}" ]]; then
  echo "[ERROR] DATA_ROOT does not exist: $DATA_ROOT" >&2
  echo "  Set DATA_ROOT or PACE_ALLOW_MISSING_DATA=1 to override." >&2
  exit 1
fi

# ------------------------------------------------------------------ batch size (autobatch)
# Source adaptive batch env if present; otherwise default to 256.
BATCH_ENV="${PACE_ROOT}/.batch_env"
if [[ -f "$BATCH_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$BATCH_ENV"
  echo "[BATCH] Loaded from autobatch: BATCH_SIZE=${BATCH_SIZE:-256}"
else
  BATCH_SIZE="${BATCH_SIZE:-256}"
  echo "[BATCH] Using default BATCH_SIZE=${BATCH_SIZE} (run tune_batch_size.py to calibrate)"
fi
export BATCH_SIZE

# ------------------------------------------------------------------ training hyperparams
DEVICE="${DEVICE:-cuda}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
SEEDS_MAIN=(42 43 44)
SEED_ABL=42

# Step counts derived from epoch targets (approximate; trainer may use epochs directly)
# Stage 1: 30 epochs × ~6500 steps/epoch @ batch=256 ≈ 195,000 steps
# Stage 2: 50 epochs × ~6500 steps/epoch @ batch=256 ≈ 325,000 steps
# Stage 3: calibration only (no steps)
STAGE1_STEPS="${STAGE1_STEPS:-195000}"
STAGE2_STEPS="${STAGE2_STEPS:-325000}"
ABLATION_STEPS="${ABLATION_STEPS:-325000}"

CKPT_EVERY="${CKPT_EVERY:-5000}"
DIAG_EVERY="${DIAG_EVERY:-500}"
KEEP_LAST="${KEEP_LAST:-3}"

cd "$PACE_ROOT"

# ------------------------------------------------------------------ manifest
echo "[RUN_DIR] $PACE_RUN_DIR"
python scripts/utils/write_manifest.py --output "$PACE_RUN_DIR/manifest.json"

# ================================================================== helpers

run_experiment() {
  # Run one training experiment with diagnostics + checkpointing.
  # $1 = experiment name; $2..N = trainer flags
  local name="$1"; shift
  local exp_dir="$PACE_RUN_DIR/$name"
  local log_file="$PACE_RUN_DIR/logs/${name//\//_}.log"
  mkdir -p "$exp_dir"
  echo "[RUN_DIR] $exp_dir"
  echo "[STARTING] $name"
  python scripts/training/train_dummy_batch.py \
      --output_dir "$exp_dir" \
      --data_root "$DATA_ROOT" \
      --enable_diagnostics \
      --enable_checkpointing \
      --diagnostic_log_every "$DIAG_EVERY" \
      --checkpoint_save_every "$CKPT_EVERY" \
      --checkpoint_keep_last "$KEEP_LAST" \
      --micro-batch "$BATCH_SIZE" \
      --device "$DEVICE" \
      "$@" 2>&1 | tee "$log_file"

  python scripts/utils/snapshot_experiment.py \
      --src "$exp_dir" \
      --dst "$PACE_SNAPSHOT_DIR/${name//\//_}_$(date +%Y%m%d_%H%M%S).tar.gz" || true
}

run_stage() {
  # Run a named training stage via scripts/train.py (reads YAML config).
  # $1 = experiment name; $2 = stage key; $3..N = extra flags
  local name="$1"; local stage="$2"; shift 2
  local exp_dir="$PACE_RUN_DIR/$name"
  local log_file="$PACE_RUN_DIR/logs/${name//\//_}.log"
  mkdir -p "$exp_dir"
  echo "[STAGE] $name ($stage)"
  python scripts/train.py \
      --stage "$stage" \
      --data_root "$DATA_ROOT" \
      --output_dir "$exp_dir" \
      --device "$DEVICE" \
      --micro_batch "$BATCH_SIZE" \
      --grad_accum "$GRAD_ACCUM" \
      "$@" 2>&1 | tee "$log_file"

  python scripts/utils/snapshot_experiment.py \
      --src "$exp_dir" \
      --dst "$PACE_SNAPSHOT_DIR/${name//\//_}_$(date +%Y%m%d_%H%M%S).tar.gz" \
      --include_checkpoints || true
}

skipped() { echo "[SKIP] $1 — reason: $2"; }

dryrun_stub() {
  local name="$1"; shift
  local exp_dir="$PACE_RUN_DIR/dry_run/$name"
  local log_file="$PACE_RUN_DIR/logs/dryrun_${name}.log"
  mkdir -p "$exp_dir"
  echo "[DRY_RUN] $name"
  "$@" --dry_run --output "$exp_dir" 2>&1 | tee "$log_file" || true
  python scripts/utils/snapshot_experiment.py \
      --src "$exp_dir" \
      --dst "$PACE_SNAPSHOT_DIR/dryrun_${name}_$(date +%Y%m%d_%H%M%S).tar.gz" || true
}

# ============================================================================
# 1. Shared Stage-1 pretrain checkpoint (used by all runs as warm-start)
# ============================================================================
run_stage "stage1_pretrain" "01_pretrain_multimodal" \
    --seed 0 \
    --max_steps "$STAGE1_STEPS"

STAGE1_CKPT="$PACE_RUN_DIR/stage1_pretrain"

# ============================================================================
# 2. Main training runs
# ============================================================================

# --- 2a. base_shortcut_fm: standard flow-matching baseline (no concordance)
for seed in "${SEEDS_MAIN[@]}"; do
  run_experiment "main/base_shortcut_fm/seed_${seed}" \
      --phase-centric-mode a \
      --steps "$STAGE2_STEPS" \
      --seed "$seed" \
      --resume_from_checkpoint "$STAGE1_CKPT"
done

# --- 2b. pace_concordance: PACE with full concordance C_t
for seed in "${SEEDS_MAIN[@]}"; do
  run_experiment "main/pace_concordance/seed_${seed}" \
      --phase-centric-mode pcar \
      --steps "$STAGE2_STEPS" \
      --seed "$seed" \
      --resume_from_checkpoint "$STAGE1_CKPT"
done

# --- 2c. pace_concordance_bdy: PACE + concordance + boundary reweighting
for seed in "${SEEDS_MAIN[@]}"; do
  run_experiment "main/pace_concordance_bdy/seed_${seed}" \
      --phase-centric-mode full \
      --steps "$STAGE2_STEPS" \
      --seed "$seed" \
      --resume_from_checkpoint "$STAGE1_CKPT"
done

# ============================================================================
# 3. Ablation runs (single seed each)
# ============================================================================

# abl_only_I1: use only Bhattacharyya estimator I^(1) for PCAR signal
run_experiment "ablation/abl_only_I1/seed_${SEED_ABL}" \
    --phase-centric-mode a \
    --pcar-estimator I1_only \
    --steps "$ABLATION_STEPS" \
    --seed "$SEED_ABL" \
    --resume_from_checkpoint "$STAGE1_CKPT"

# abl_only_I2: use only action-variance estimator I^(2)
run_experiment "ablation/abl_only_I2/seed_${SEED_ABL}" \
    --phase-centric-mode a \
    --pcar-estimator I2_only \
    --steps "$ABLATION_STEPS" \
    --seed "$SEED_ABL" \
    --resume_from_checkpoint "$STAGE1_CKPT"

# abl_only_I3: use only velocity-curvature estimator I^(3)
run_experiment "ablation/abl_only_I3/seed_${SEED_ABL}" \
    --phase-centric-mode a \
    --pcar-estimator I3_only \
    --steps "$ABLATION_STEPS" \
    --seed "$SEED_ABL" \
    --resume_from_checkpoint "$STAGE1_CKPT"

# abl_oracle_cliff: oracle cliff signal (ground-truth phase boundaries)
run_experiment "ablation/abl_oracle_cliff/seed_${SEED_ABL}" \
    --phase-centric-mode full \
    --oracle-cliff \
    --steps "$ABLATION_STEPS" \
    --seed "$SEED_ABL" \
    --resume_from_checkpoint "$STAGE1_CKPT"

# ============================================================================
# 4. Stage-3 calibration for top models
# ============================================================================
for seed in "${SEEDS_MAIN[@]}"; do
  run_stage "main/pace_concordance_bdy_cal/seed_${seed}" "03_finetune_replan" \
      --seed "$seed" \
      --resume_from_checkpoint "$PACE_RUN_DIR/main/pace_concordance_bdy/seed_${seed}"
done

# ============================================================================
# 5. Evaluation
#    7 experiments, target rollout counts (requires eval env):
#      base_shortcut_fm:      500 rollouts × 3 seeds = 1500
#      pace_concordance:      500 rollouts × 3 seeds = 1500
#      pace_concordance_bdy:  500 rollouts × 3 seeds = 1500
#      abl_only_I1/I2/I3:    500 rollouts × 1 seed  = 500 each
#      abl_oracle_cliff:      500 rollouts × 1 seed  = 500
#      Total: 7500 rollouts (+ 2928 calibration set = 10,428 target)
# ============================================================================
EVAL_SCRIPT="${PACE_ROOT}/scripts/eval/run_eval.py"
if [[ -f "$EVAL_SCRIPT" ]]; then
  EVAL_RUNS=(
    "main/base_shortcut_fm"
    "main/pace_concordance"
    "main/pace_concordance_bdy"
    "ablation/abl_only_I1"
    "ablation/abl_only_I2"
    "ablation/abl_only_I3"
    "ablation/abl_oracle_cliff"
  )
  EVAL_SEEDS_MAP=(
    "42 43 44"
    "42 43 44"
    "42 43 44"
    "42"
    "42"
    "42"
    "42"
  )
  for i in "${!EVAL_RUNS[@]}"; do
    run_name="${EVAL_RUNS[$i]}"
    read -r -a eval_seeds <<< "${EVAL_SEEDS_MAP[$i]}"
    for seed in "${eval_seeds[@]}"; do
      ckpt_path="$PACE_RUN_DIR/${run_name}/seed_${seed}"
      eval_out="$PACE_RUN_DIR/eval/${run_name//\//_}_seed${seed}"
      mkdir -p "$eval_out"
      echo "[EVAL] $run_name seed=$seed"
      python "$EVAL_SCRIPT" \
          --checkpoint "$ckpt_path" \
          --n_rollouts 500 \
          --seed "$seed" \
          --output "$eval_out" \
          --device "$DEVICE" \
          2>&1 | tee "$PACE_RUN_DIR/logs/eval_${run_name//\//_}_seed${seed}.log" || true
    done
  done
else
  echo "[SKIP] eval/run_eval.py not found — evaluation step skipped"
fi

# ============================================================================
# 6. Aggregate results
# ============================================================================
python scripts/aggregate_ablation.py \
    --input_root "$PACE_RUN_DIR" \
    --output "$PACE_RUN_DIR/aggregated" \
    2>&1 | tee "$PACE_RUN_DIR/logs/aggregate.log"

echo "[DONE] CoRL sweep complete."
echo "[OUTPUTS]   $PACE_RUN_DIR"
echo "[SNAPSHOTS] $PACE_SNAPSHOT_DIR"
