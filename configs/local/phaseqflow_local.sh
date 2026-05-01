#!/usr/bin/env bash
# Local 5090 PACE v2 launcher — same skip/dry_run logic as the H800 cloud
# script, but with smol-libero and short step counts so end-to-end behavior
# can be validated before paying for a full H800 run.

set -euo pipefail

# ------------------------------------------------------------------ paths
export PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PACE_OUT="${PACE_OUT:-$PACE_ROOT/outputs}"
export PACE_DATA="${PACE_DATA:-$PACE_ROOT/data}"
export PACE_CKPT="${PACE_CKPT:-$PACE_ROOT/checkpoints}"
export PACE_RUN_DIR="${PACE_RUN_DIR:-$PACE_OUT/local_run_$(date +%Y%m%d_%H%M%S)}"
export PACE_SNAPSHOT_DIR="${PACE_SNAPSHOT_DIR:-$PACE_RUN_DIR/snapshots}"
mkdir -p "$PACE_RUN_DIR" "$PACE_RUN_DIR/logs" "$PACE_SNAPSHOT_DIR"

# ------------------------------------------------------------------ filter lists (must match cloud)
export PACE_SKIP_ABLATIONS="${PACE_SKIP_ABLATIONS:-03_cliff_via_var_only,04_cliff_via_curvature_only}"
export PACE_DRYRUN_ABLATIONS="${PACE_DRYRUN_ABLATIONS:-05_cliff_concordance}"
export PACE_SKIP_PHENOMENA="${PACE_SKIP_PHENOMENA:-libero_perturbed,simpler_env}"
export PACE_DRYRUN_PHENOMENA="${PACE_DRYRUN_PHENOMENA:-6_1_universality,6_3_triangulation,6_4_trigger_comparison}"

REAL_ABLATIONS=(
  "01_bc_chunked"
  "02_cliff_via_beta_only"
  "06_oracle_cliff"
  "07_cliff_concordance_with_boundary_reweight"
)
SEEDS=(42)  # single seed locally to keep the run short

# ------------------------------------------------------------------ training defaults
DATASET_REPO_ID="${DATASET_REPO_ID:-HuggingFaceVLA/smol-libero}"
DATA_ROOT="${DATA_ROOT:-${PACE_DATA}/smol_libero}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STAGE1_STEPS="${STAGE1_STEPS:-200}"
STAGE2_STEPS="${STAGE2_STEPS:-400}"
STAGE3_STEPS="${STAGE3_STEPS:-100}"
ABLATION_STEPS="${ABLATION_STEPS:-200}"
CKPT_EVERY="${CKPT_EVERY:-50}"
DIAG_EVERY="${DIAG_EVERY:-25}"
KEEP_LAST="${KEEP_LAST:-3}"

cd "$PACE_ROOT"

# ------------------------------------------------------------------ manifest
echo "[RUN_DIR] $PACE_RUN_DIR"
python scripts/utils/write_manifest.py --output "$PACE_RUN_DIR/manifest.json"

# ------------------------------------------------------------------ helpers (same as cloud)
run_experiment() {
  local name="$1"; shift
  local exp_dir="$PACE_RUN_DIR/$name"
  local log_file="$PACE_RUN_DIR/logs/${name//\//_}.log"
  mkdir -p "$exp_dir"
  echo "[RUN_DIR] $exp_dir"
  echo "[STARTING] $name"
  python scripts/training/train_dummy_batch.py \
      --output_dir "$exp_dir" \
      --enable_diagnostics \
      --enable_checkpointing \
      --diagnostic_log_every "$DIAG_EVERY" \
      --checkpoint_save_every "$CKPT_EVERY" \
      --checkpoint_keep_last "$KEEP_LAST" \
      --data_root "$DATA_ROOT" \
      "$@" 2>&1 | tee "$log_file"
  python scripts/utils/snapshot_experiment.py \
      --src "$exp_dir" \
      --dst "$PACE_SNAPSHOT_DIR/${name//\//_}_$(date +%Y%m%d_%H%M%S).tar.gz" || true
}

skipped() {
  echo "[SKIP] $1 — reason: $2"
}

# ============================================================================
# 1. Three-stage main training (smol-libero)
# ============================================================================
run_experiment "stage1_pretrain" \
    --phase-centric-mode off \
    --steps "$STAGE1_STEPS" \
    --device "$DEVICE" --seed 0 \
    --micro-batch "$BATCH_SIZE"

run_experiment "stage2_phase_flow" \
    --phase-centric-mode a \
    --steps "$STAGE2_STEPS" \
    --device "$DEVICE" --seed 0 \
    --micro-batch "$BATCH_SIZE" \
    --resume_from_checkpoint "$PACE_RUN_DIR/stage1_pretrain"

run_experiment "stage3_finetune" \
    --phase-centric-mode pcar \
    --steps "$STAGE3_STEPS" \
    --device "$DEVICE" --seed 0 \
    --micro-batch "$BATCH_SIZE" \
    --resume_from_checkpoint "$PACE_RUN_DIR/stage2_phase_flow"

# ============================================================================
# 2. Ablation matrix
# ============================================================================
for cfg in 03_cliff_via_var_only 04_cliff_via_curvature_only; do
  skipped "ablation/$cfg" "in PACE_SKIP_ABLATIONS"
done

for cfg in "${REAL_ABLATIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_experiment "ablation/${cfg}/seed_${seed}" \
        --phase-centric-mode full \
        --steps "$ABLATION_STEPS" \
        --device "$DEVICE" --seed "$seed" \
        --micro-batch "$BATCH_SIZE" \
        --resume_from_checkpoint "$PACE_RUN_DIR/stage1_pretrain"
  done
done

echo "[DRY_RUN] ablation/05_cliff_concordance"
mkdir -p "$PACE_RUN_DIR/dry_run/ablation_05"
python scripts/aggregate_ablation.py --dry_run \
    --output "$PACE_RUN_DIR/dry_run/ablation_05" 2>&1 \
    | tee "$PACE_RUN_DIR/logs/dryrun_ablation_05.log"
python scripts/utils/snapshot_experiment.py \
    --src "$PACE_RUN_DIR/dry_run/ablation_05" \
    --dst "$PACE_SNAPSHOT_DIR/dryrun_ablation_05_$(date +%Y%m%d_%H%M%S).tar.gz" || true

# ============================================================================
# 3. Aggregate
# ============================================================================
python scripts/aggregate_ablation.py \
    --input_root "$PACE_RUN_DIR/ablation" \
    --output "$PACE_RUN_DIR/aggregated" \
    2>&1 | tee "$PACE_RUN_DIR/logs/aggregate.log"

echo "[DONE] Local run complete."
echo "[OUTPUTS] $PACE_RUN_DIR"
echo "[SNAPSHOTS] $PACE_SNAPSHOT_DIR"
