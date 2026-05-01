#!/usr/bin/env bash
# ============================================================================
# PACE v2 cloud launcher (single-GPU, ~6–8 days on RTX PRO 6000 96GB).
# ============================================================================
# Runs the experiments that constitute the CoRL paper submission:
#
#   Stage 1 → 2 → 3 main training       (Table 1 headline checkpoint)
#   Ablation 01 BC-Chunked × 3 seeds    (Table 2 baseline)
#   Ablation 02 Cliff via β × 3 seeds   (Table 2 — concordance > β alone)
#   Ablation 07 PACE v2 full × 3 seeds  (Table 2 headline)
#   §6.2 Regret Scaling      (real)     (theoretical claim)
#   §6.5 Boundary Loss Ratio (real)     (theoretical claim)
#
# Step counts are derived dynamically from BATCH_SIZE so the schedule is
# invariant to whatever scripts/tune_batch_size.py picks at runtime
# (autobatch is run by scripts/run_autodl_pipeline.sh before this script).
# ============================================================================

set -euo pipefail

# ------------------------------------------------------------------ paths
export PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PACE_OUT="${PACE_OUT:-$PACE_ROOT/outputs}"
export PACE_DATA="${PACE_DATA:-$PACE_ROOT/data}"
export PACE_CKPT="${PACE_CKPT:-$PACE_ROOT/checkpoints}"
export PACE_RUN_DIR="${PACE_RUN_DIR:-$PACE_OUT/run_$(date +%Y%m%d_%H%M%S)}"
export PACE_SNAPSHOT_DIR="${PACE_SNAPSHOT_DIR:-/root/autodl-tmp/snapshots}"
mkdir -p "$PACE_RUN_DIR" "$PACE_RUN_DIR/logs" "$PACE_SNAPSHOT_DIR"

# ------------------------------------------------------------------ filter lists
# 03/04/05/06 all dropped from CoRL trim
export PACE_SKIP_ABLATIONS="${PACE_SKIP_ABLATIONS:-03_cliff_via_var_only,04_cliff_via_curvature_only,05_cliff_concordance,06_oracle_cliff}"
export PACE_DRYRUN_ABLATIONS="${PACE_DRYRUN_ABLATIONS:-}"

# All non-essential phenomena dropped; only §6.2 and §6.5 keep real runs
export PACE_SKIP_PHENOMENA="${PACE_SKIP_PHENOMENA:-libero_perturbed,simpler_env,6_1_universality,6_3_triangulation,6_4_trigger_comparison}"
export PACE_DRYRUN_PHENOMENA="${PACE_DRYRUN_PHENOMENA:-}"

# Ablation matrix (paper Table 2)
REAL_ABLATIONS=(
  "01_bc_chunked"
  "02_cliff_via_beta_only"
  "07_cliff_concordance_with_boundary_reweight"
)
SEEDS=(42 43 44)

# ------------------------------------------------------------------ training defaults
DATASET_REPO_ID="${DATASET_REPO_ID:-HuggingFaceVLA/libero}"

# IMPORTANT — BATCH_SIZE is normally set by `scripts/tune_batch_size.py`
# (autobatch). The pipeline runner sources its output before invoking this
# script. Hard-coding here would defeat that. Default 32 if autobatch was
# skipped.
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"

# Total samples per stage (matches the original v2.0 baseline). Step counts
# are derived dynamically from these so the schedule is invariant to the
# autotuned BATCH_SIZE.  Override either in this file or via env var if you
# want a longer/shorter run.
TOTAL_SAMPLES_STAGE1="${TOTAL_SAMPLES_STAGE1:-6400000}"   # 200k×32 reference
TOTAL_SAMPLES_STAGE2="${TOTAL_SAMPLES_STAGE2:-12800000}"  # 400k×32 reference
TOTAL_SAMPLES_STAGE3="${TOTAL_SAMPLES_STAGE3:-640000}"    # 20k×32 reference
TOTAL_SAMPLES_ABLATION="${TOTAL_SAMPLES_ABLATION:-5000000}"  # ~150k×32 — CoRL trim

# Derive step counts: ceil(samples / batch_size). At least 100 steps.
_steps_for() { local total=$1; local steps=$(( (total + BATCH_SIZE - 1) / BATCH_SIZE )); [[ $steps -lt 100 ]] && steps=100; echo "$steps"; }
STAGE1_STEPS="${STAGE1_STEPS:-$(_steps_for "$TOTAL_SAMPLES_STAGE1")}"
STAGE2_STEPS="${STAGE2_STEPS:-$(_steps_for "$TOTAL_SAMPLES_STAGE2")}"
STAGE3_STEPS="${STAGE3_STEPS:-$(_steps_for "$TOTAL_SAMPLES_STAGE3")}"
ABLATION_STEPS="${ABLATION_STEPS:-$(_steps_for "$TOTAL_SAMPLES_ABLATION")}"

CKPT_EVERY="${CKPT_EVERY:-200}"
DIAG_EVERY="${DIAG_EVERY:-100}"
KEEP_LAST="${KEEP_LAST:-3}"

DATA_ROOT="${DATA_ROOT:-${PACE_DATA}/libero_10}"

cd "$PACE_ROOT"

# ------------------------------------------------------------------ data preflight
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] DATA_ROOT not found: $DATA_ROOT" >&2
  echo "  Set DATA_ROOT to a local LIBERO LeRobot directory or download it first." >&2
  if [[ -z "${PACE_ALLOW_MISSING_DATA:-}" ]]; then
    exit 1
  fi
  echo "[WARN] PACE_ALLOW_MISSING_DATA set; continuing without local data" >&2
fi

# ------------------------------------------------------------------ manifest
echo "[RUN_DIR] $PACE_RUN_DIR"
echo "[CONFIG ] BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  effective=$((BATCH_SIZE * GRAD_ACCUM))"
echo "[CONFIG ] Stage1=${STAGE1_STEPS}  Stage2=${STAGE2_STEPS}  Stage3=${STAGE3_STEPS}  Ablation=${ABLATION_STEPS}"
echo "[CONFIG ] Ablations: ${REAL_ABLATIONS[*]}"
echo "[CONFIG ] Seeds: ${SEEDS[*]}"
python scripts/utils/write_manifest.py --output "$PACE_RUN_DIR/manifest.json"

# ------------------------------------------------------------------ helpers
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

run_stage() {
  # Three-stage main training: dispatches via scripts/train.py.
  local name="$1"; local stage_yaml="$2"; local steps="$3"; local resume="${4:-}"
  local exp_dir="$PACE_RUN_DIR/$name"
  local log_file="$PACE_RUN_DIR/logs/${name}.log"
  mkdir -p "$exp_dir"
  echo "[RUN_DIR] $exp_dir"
  echo "[STARTING] $name (stage YAML: $stage_yaml)"
  local resume_flag=()
  if [[ -n "$resume" ]]; then
    resume_flag=(--resume_from_checkpoint "$resume")
  fi
  python scripts/train.py \
      --stage "$stage_yaml" \
      --data_root "$DATA_ROOT" \
      --max_steps "$steps" \
      --device cuda --seed 0 \
      --output_dir "$exp_dir" \
      --micro_batch "$BATCH_SIZE" \
      "${resume_flag[@]}" 2>&1 | tee "$log_file"

  python scripts/utils/snapshot_experiment.py \
      --src "$exp_dir" \
      --dst "$PACE_SNAPSHOT_DIR/${name}_$(date +%Y%m%d_%H%M%S).tar.gz" || true
}

skipped() {
  echo "[SKIP] $1 — reason: $2"
}

# ============================================================================
# 1. Three-stage main training
# ============================================================================
run_stage "stage1_pretrain" \
    "configs/train/01_pretrain_multimodal.yaml" \
    "$STAGE1_STEPS"

run_stage "stage2_phase_flow" \
    "configs/train/02_train_phase_and_flow.yaml" \
    "$STAGE2_STEPS" \
    "$PACE_RUN_DIR/stage1_pretrain"

run_stage "stage3_finetune" \
    "configs/train/03_finetune_replan.yaml" \
    "$STAGE3_STEPS" \
    "$PACE_RUN_DIR/stage2_phase_flow"

# Final-stage checkpoint snapshot includes the real .pt for downstream eval.
python scripts/utils/snapshot_experiment.py \
    --src "$PACE_RUN_DIR/stage3_finetune" \
    --dst "$PACE_SNAPSHOT_DIR/stage3_finetune_FINAL_$(date +%Y%m%d_%H%M%S).tar.gz" \
    --include_checkpoints || true

# ============================================================================
# 2. Ablation matrix (CoRL trim: 01, 02, 07)
# ============================================================================
for cfg in 03_cliff_via_var_only 04_cliff_via_curvature_only 05_cliff_concordance 06_oracle_cliff; do
  skipped "ablation/$cfg" "out of CoRL scope (covered by tests / redundant with Ablation 07)"
done

for cfg in "${REAL_ABLATIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_experiment "ablation/${cfg}/seed_${seed}" \
        --phase-centric-mode full \
        --steps "$ABLATION_STEPS" \
        --device cuda --seed "$seed" \
        --micro-batch "$BATCH_SIZE" \
        --resume_from_checkpoint "$PACE_RUN_DIR/stage1_pretrain"
  done
done

# ============================================================================
# 3. Phenomenon experiments (real — §6.2 + §6.5 only)
# ============================================================================
mkdir -p "$PACE_RUN_DIR/phenomenon_6_2_regret_scaling"
echo "[STARTING] phenomenon_6_2_regret_scaling"
python scripts/phenomenon/regret_scaling.py \
    --H_values 4 8 16 32 64 \
    --checkpoint "$PACE_RUN_DIR/stage3_finetune" \
    --libero_task libero_long \
    --output "$PACE_RUN_DIR/phenomenon_6_2_regret_scaling" \
    2>&1 | tee "$PACE_RUN_DIR/logs/phenomenon_6_2.log" || true
python scripts/utils/snapshot_experiment.py \
    --src "$PACE_RUN_DIR/phenomenon_6_2_regret_scaling" \
    --dst "$PACE_SNAPSHOT_DIR/phenomenon_6_2_$(date +%Y%m%d_%H%M%S).tar.gz" || true

mkdir -p "$PACE_RUN_DIR/phenomenon_6_5_boundary_loss"
echo "[STARTING] phenomenon_6_5_boundary_loss"
python scripts/diagnostics/diagnostic_utils/measure_boundary_error.py \
    --output "$PACE_RUN_DIR/phenomenon_6_5_boundary_loss/ratio.json" \
    2>&1 | tee "$PACE_RUN_DIR/logs/phenomenon_6_5.log" || true
python scripts/utils/snapshot_experiment.py \
    --src "$PACE_RUN_DIR/phenomenon_6_5_boundary_loss" \
    --dst "$PACE_SNAPSHOT_DIR/phenomenon_6_5_$(date +%Y%m%d_%H%M%S).tar.gz" || true

skipped "phenomenon/6_1_universality"   "out of CoRL scope (~220GB baseline ckpts)"
skipped "phenomenon/6_3_triangulation"  "redundant with Ablation 02 vs 07"
skipped "phenomenon/6_4_trigger_comparison" "synthetic by design"
skipped "phenomenon/simpler_env"        "out of scope, requires ManiSkill2"
skipped "phenomenon/libero_perturbed"   "supplementary, not Table 1/2"

# ============================================================================
# 4. Aggregation + final report
# ============================================================================
python scripts/aggregate_ablation.py \
    --input_root "$PACE_RUN_DIR/ablation" \
    --output "$PACE_RUN_DIR/aggregated" \
    2>&1 | tee "$PACE_RUN_DIR/logs/aggregate.log"

echo "[DONE] PACE v2 cloud run complete."
echo "[OUTPUTS]   $PACE_RUN_DIR"
echo "[SNAPSHOTS] $PACE_SNAPSHOT_DIR"
