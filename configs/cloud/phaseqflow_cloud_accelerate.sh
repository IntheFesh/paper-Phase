#!/usr/bin/env bash
# H800 PACE v2 launcher — streamlined experiment scope + diagnostic outputs.
#
# Runs the v2.0 paper-relevant experiments only:
#   Stage 1 → Stage 2 → Stage 3 (main checkpoint)
#   Ablation × {01, 02, 06, 07} × seeds {42, 43, 44}
#   §6.2 Regret Scaling     (real)
#   §6.5 Boundary Loss Ratio (real)
#   §6.1 Universality, §6.3 Triangulation, §6.4 Trigger comparison (dry_run)
#   Ablation 05 (dry_run; partial concordance)
# Skipped entirely: Ablation 03/04, SimplerEnv, LIBERO-Perturbed.
#
# All outputs land under $PACE_RUN_DIR (timestamped), with one tar.gz
# snapshot per experiment shipped to $PACE_SNAPSHOT_DIR for off-box archival.

set -euo pipefail

# ------------------------------------------------------------------ paths
export PACE_ROOT="${PACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PACE_OUT="${PACE_OUT:-$PACE_ROOT/outputs}"
export PACE_DATA="${PACE_DATA:-$PACE_ROOT/data}"
export PACE_CKPT="${PACE_CKPT:-$PACE_ROOT/checkpoints}"
export PACE_RUN_DIR="${PACE_RUN_DIR:-$PACE_OUT/h800_run_$(date +%Y%m%d_%H%M%S)}"
export PACE_SNAPSHOT_DIR="${PACE_SNAPSHOT_DIR:-/root/autodl-tmp/snapshots}"
mkdir -p "$PACE_RUN_DIR" "$PACE_RUN_DIR/logs" "$PACE_SNAPSHOT_DIR"

# ------------------------------------------------------------------ filter lists
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
SEEDS=(42 43 44)

# ------------------------------------------------------------------ training defaults
DATASET_REPO_ID="${DATASET_REPO_ID:-HuggingFaceVLA/libero}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
STAGE1_STEPS="${STAGE1_STEPS:-200000}"
STAGE2_STEPS="${STAGE2_STEPS:-400000}"
STAGE3_STEPS="${STAGE3_STEPS:-20000}"
ABLATION_STEPS="${ABLATION_STEPS:-400000}"
CKPT_EVERY="${CKPT_EVERY:-200}"
DIAG_EVERY="${DIAG_EVERY:-200}"
KEEP_LAST="${KEEP_LAST:-3}"

# Data root: prefer local cache under $PACE_DATA, fall back to Hub download
DATA_ROOT="${DATA_ROOT:-${PACE_DATA}/libero_10}"

cd "$PACE_ROOT"

# ------------------------------------------------------------------ manifest
echo "[RUN_DIR] $PACE_RUN_DIR"
python scripts/utils/write_manifest.py --output "$PACE_RUN_DIR/manifest.json"

# ------------------------------------------------------------------ helpers
run_experiment() {
  # Run one experiment, then snapshot artifacts to $PACE_SNAPSHOT_DIR.
  # Args:
  #   $1 — experiment name (used for run-dir + log + snapshot)
  #   $2..N — additional flags forwarded to the trainer
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

  # Per-experiment artifact tarball.
  python scripts/utils/snapshot_experiment.py \
      --src "$exp_dir" \
      --dst "$PACE_SNAPSHOT_DIR/${name//\//_}_$(date +%Y%m%d_%H%M%S).tar.gz" || true
}

skipped() {
  echo "[SKIP] $1 — reason: $2"
}

dryrun_stub() {
  # Mark a phenomenon as dry_run via its CLI flag and snapshot whatever it produced.
  local name="$1"; shift
  local cmd=("$@")
  local exp_dir="$PACE_RUN_DIR/dry_run/$name"
  local log_file="$PACE_RUN_DIR/logs/dryrun_${name}.log"
  mkdir -p "$exp_dir"
  echo "[DRY_RUN] $name"
  "${cmd[@]}" --dry_run --output "$exp_dir" 2>&1 | tee "$log_file" || true
  python scripts/utils/snapshot_experiment.py \
      --src "$exp_dir" \
      --dst "$PACE_SNAPSHOT_DIR/dryrun_${name}_$(date +%Y%m%d_%H%M%S).tar.gz" || true
}

# ============================================================================
# 1. Three-stage main training
# ============================================================================
run_experiment "stage1_pretrain" \
    --phase-centric-mode off \
    --steps "$STAGE1_STEPS" \
    --device cuda --seed 0 \
    --micro-batch "$BATCH_SIZE"

run_experiment "stage2_phase_flow" \
    --phase-centric-mode a \
    --steps "$STAGE2_STEPS" \
    --device cuda --seed 0 \
    --micro-batch "$BATCH_SIZE" \
    --resume_from_checkpoint "$PACE_RUN_DIR/stage1_pretrain"

run_experiment "stage3_finetune" \
    --phase-centric-mode pcar \
    --steps "$STAGE3_STEPS" \
    --device cuda --seed 0 \
    --micro-batch "$BATCH_SIZE" \
    --resume_from_checkpoint "$PACE_RUN_DIR/stage2_phase_flow"

# ============================================================================
# 2. Ablation matrix
# ============================================================================
for cfg in 03_cliff_via_var_only 04_cliff_via_curvature_only; do
  skipped "ablation/$cfg" "in PACE_SKIP_ABLATIONS (NotImplementedError; degrades to baseline)"
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

# Dry-run for ablation 05 (concordance falls back to beta_t — placeholder only)
echo "[DRY_RUN] ablation/05_cliff_concordance"
mkdir -p "$PACE_RUN_DIR/dry_run/ablation_05"
python scripts/aggregate_ablation.py --dry_run \
    --output "$PACE_RUN_DIR/dry_run/ablation_05" 2>&1 \
    | tee "$PACE_RUN_DIR/logs/dryrun_ablation_05.log"
python scripts/utils/snapshot_experiment.py \
    --src "$PACE_RUN_DIR/dry_run/ablation_05" \
    --dst "$PACE_SNAPSHOT_DIR/dryrun_ablation_05_$(date +%Y%m%d_%H%M%S).tar.gz" || true

# ============================================================================
# 3. Phenomenon experiments
# ============================================================================
# §6.2 Regret Scaling — real data
mkdir -p "$PACE_RUN_DIR/phenomenon_6_2_regret_scaling"
echo "[STARTING] phenomenon_6_2_regret_scaling"
python scripts/phenomenon/regret_scaling.py \
    --H_values 4 8 16 32 64 \
    --output "$PACE_RUN_DIR/phenomenon_6_2_regret_scaling" \
    2>&1 | tee "$PACE_RUN_DIR/logs/phenomenon_6_2.log" || true
python scripts/utils/snapshot_experiment.py \
    --src "$PACE_RUN_DIR/phenomenon_6_2_regret_scaling" \
    --dst "$PACE_SNAPSHOT_DIR/phenomenon_6_2_$(date +%Y%m%d_%H%M%S).tar.gz" || true

# §6.5 Boundary Loss Ratio — real data
mkdir -p "$PACE_RUN_DIR/phenomenon_6_5_boundary_loss"
echo "[STARTING] phenomenon_6_5_boundary_loss"
python scripts/diagnostics/diagnostic_utils/measure_boundary_error.py \
    --output "$PACE_RUN_DIR/phenomenon_6_5_boundary_loss/ratio.json" \
    2>&1 | tee "$PACE_RUN_DIR/logs/phenomenon_6_5.log" || true
python scripts/utils/snapshot_experiment.py \
    --src "$PACE_RUN_DIR/phenomenon_6_5_boundary_loss" \
    --dst "$PACE_SNAPSHOT_DIR/phenomenon_6_5_$(date +%Y%m%d_%H%M%S).tar.gz" || true

# Dry-run phenomenon
dryrun_stub "phenomenon_6_1_universality" \
    python scripts/phenomenon/universality.py \
        --policies bc_act diffusion_policy \
        --n_rollouts 20 --seeds 0 1 2

dryrun_stub "phenomenon_6_3_triangulation" \
    python scripts/phenomenon/triangulation_concordance.py

dryrun_stub "phenomenon_6_4_trigger_comparison" \
    python scripts/diagnostics/diagnostic_utils/trigger_comparison.py

skipped "phenomenon/simpler_env" "in PACE_SKIP_PHENOMENA (out of scope, requires ManiSkill2)"
skipped "phenomenon/libero_perturbed" "in PACE_SKIP_PHENOMENA (supplementary, not Table 1/2)"

# ============================================================================
# 4. Aggregation + final report
# ============================================================================
python scripts/aggregate_ablation.py \
    --input_root "$PACE_RUN_DIR/ablation" \
    --output "$PACE_RUN_DIR/aggregated" \
    2>&1 | tee "$PACE_RUN_DIR/logs/aggregate.log"

echo "[DONE] PACE v2 H800 streamlined run complete."
echo "[OUTPUTS] $PACE_RUN_DIR"
echo "[SNAPSHOTS] $PACE_SNAPSHOT_DIR"
