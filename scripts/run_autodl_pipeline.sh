#!/usr/bin/env bash
# AutoDL single-entrypoint pipeline for PACE v2 (PhaseQFlow++).
#
# Phases:
#   preflight  — install deps, verify data + cliff estimators, run coverage (default)
#   train      — run CoRL full sweep (must be in tmux / screen)
#   post       — aggregate results + generate paper figures + final snapshot
#   all        — run all phases (2→8) with 10-second abort window
#
# Usage:
#   bash scripts/run_autodl_pipeline.sh [preflight|train|post|all]
#
# Environment variables:
#   DATA_ROOT       path to LIBERO-10 dataset (required for train/all)
#   BATCH_SIZE      override batch size (default: determined by autobatch)
#   CLOUD_SCRIPT    path to training script (default: configs/cloud/phaseqflow_cloud_corl.sh)
#   PACE_RUN_DIR    output directory (auto-timestamped if not set)
#   PACE_ALLOW_MISSING_DATA=1  skip data preflight check
#
# Prerequisites (AutoDL):
#   conda activate phaseqflow   (or equivalent venv)
#   pip install -e lerobot_policy_phaseqflow/
#   pip install bitsandbytes>=0.41 vector-quantize-pytorch

set -euo pipefail

PHASE="${1:-preflight}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLOUD_SCRIPT="${CLOUD_SCRIPT:-$REPO_ROOT/configs/cloud/phaseqflow_cloud_corl.sh}"
BATCH_ENV="$REPO_ROOT/.batch_env"

export PACE_ROOT="$REPO_ROOT"
export PACE_OUT="${PACE_OUT:-$REPO_ROOT/outputs}"
export PACE_DATA="${PACE_DATA:-$REPO_ROOT/data}"

# ------------------------------------------------------------------ helpers
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
fail() { echo "[ERROR] $*" >&2; exit 1; }

# ================================================================== PHASE 2: deps
phase_2_deps() {
  log "=== Phase 2: Install / verify dependencies ==="
  pip install -e "$REPO_ROOT/lerobot_policy_phaseqflow/" --quiet
  pip install bitsandbytes>=0.41 vector-quantize-pytorch --quiet || \
    log "[WARN] Some optional packages failed to install; continuing."
  python -c "import torch; print('[dep] torch:', torch.__version__)"
  python -c "import numpy; print('[dep] numpy:', numpy.__version__)"
  python -c "
try:
    import bitsandbytes as bnb
    print('[dep] bitsandbytes:', bnb.__version__)
except ImportError:
    print('[dep] bitsandbytes: NOT INSTALLED (PagedAdamW8bit unavailable; will fall back to AdamW)')
"
  python -c "
try:
    from vector_quantize_pytorch import FSQ
    print('[dep] vector_quantize_pytorch: OK')
except ImportError:
    print('[dep] vector_quantize_pytorch: NOT INSTALLED')
"
  log "Phase 2 complete."
}

# ================================================================== PHASE 3: verify real data + estimators
phase_3_verify() {
  log "=== Phase 3: Verify real data + cliff estimators ==="
  DATA_ROOT="${DATA_ROOT:-${PACE_DATA}/libero_10}"

  # 1. data root exists
  if [[ -d "$DATA_ROOT" ]]; then
    log "[OK] DATA_ROOT=$DATA_ROOT"
  elif [[ -n "${PACE_ALLOW_MISSING_DATA:-}" ]]; then
    log "[WARN] DATA_ROOT missing but PACE_ALLOW_MISSING_DATA set; skipping data check."
  else
    fail "DATA_ROOT does not exist: $DATA_ROOT\n  Set DATA_ROOT or PACE_ALLOW_MISSING_DATA=1"
  fi

  # 2. cliff estimators implemented (no NotImplementedError)
  python - <<'PY'
import torch
from lerobot_policy_phaseqflow.phase_centric.cliff_estimators import (
    compute_I_hat_1, compute_I_hat_2, compute_I_hat_3, compute_concordance_C,
)

# I_hat_1
beta = torch.tensor([0.3, 0.7])
i1 = compute_I_hat_1(beta)
assert (i1 <= 0).all(), "I_hat_1 must be ≤ 0"

# I_hat_2
samples = torch.randn(4, 2, 16, 7)
i2 = compute_I_hat_2(samples)
assert i2.shape == (2,) and (i2 <= 0).all(), "I_hat_2 shape/sign error"

# I_hat_3
v1 = torch.randn(2, 16, 7)
v2 = torch.randn(2, 16, 7)
i3 = compute_I_hat_3(v1, v2)
assert i3.shape == (2,) and (i3 <= 0).all(), "I_hat_3 shape/sign error"

# concordance
c = compute_concordance_C([i1, i2, i3])
assert c.shape == (2,) and (c >= 0).all() and (c <= 1).all(), "concordance out of [0,1]"

print("[OK] All cliff estimators implemented and correct.")
PY

  # 3. train_dummy_batch rejects invalid data_root with RuntimeError
  python - <<'PY'
import subprocess, sys, tempfile, os
result = subprocess.run(
    [sys.executable, "scripts/training/train_dummy_batch.py",
     "--data_root", "/nonexistent_path_xyz",
     "--steps", "1", "--output_dir", tempfile.mkdtemp()],
    capture_output=True, text=True, cwd=os.environ.get("PACE_ROOT", "."),
)
if result.returncode != 0:
    print("[OK] train_dummy_batch correctly rejects invalid --data_root")
else:
    print("[WARN] train_dummy_batch did NOT reject invalid --data_root (check for silent fallback)")
PY

  log "Phase 3 complete."
}

# ================================================================== PHASE 4: autobatch
phase_4_autobatch() {
  log "=== Phase 4: Adaptive batch-size tuning ==="
  DEVICE="${DEVICE:-cuda}"
  python "$REPO_ROOT/scripts/utils/tune_batch_size.py" \
      --start "${BATCH_SIZE:-256}" \
      --device "$DEVICE" \
      --output "$BATCH_ENV"
  # shellcheck source=/dev/null
  source "$BATCH_ENV"
  log "Optimal batch size: $OPTIMAL_BATCH"
  log "Phase 4 complete."
}

# ================================================================== PHASE 5: coverage
phase_5_coverage() {
  log "=== Phase 5: Test coverage ==="
  cd "$REPO_ROOT"
  python -m pytest tests/ -x -q --tb=short 2>&1 | tail -20
  log "Phase 5 complete."
}

# ================================================================== PHASE 6: train
phase_6_train() {
  log "=== Phase 6: Training sweep ==="
  if [[ -f "$BATCH_ENV" ]]; then
    # shellcheck source=/dev/null
    source "$BATCH_ENV"
    log "Loaded autobatch: BATCH_SIZE=${BATCH_SIZE:-256}"
  fi
  export BATCH_SIZE="${BATCH_SIZE:-256}"
  export DATA_ROOT="${DATA_ROOT:-${PACE_DATA}/libero_10}"
  bash "$CLOUD_SCRIPT"
  log "Phase 6 complete."
}

# ================================================================== PHASE 8: post-processing
phase_8_post() {
  log "=== Phase 8: Aggregate + figures + snapshot ==="
  RUN_DIR="${PACE_RUN_DIR:-}"
  if [[ -z "$RUN_DIR" ]]; then
    # Pick the most recent run
    RUN_DIR="$(ls -td "$PACE_OUT"/corl_run_* 2>/dev/null | head -1)"
    [[ -n "$RUN_DIR" ]] || fail "No corl_run_* found under $PACE_OUT"
  fi
  log "Using RUN_DIR=$RUN_DIR"

  # Aggregate ablation results
  python "$REPO_ROOT/scripts/aggregate_ablation.py" \
      --input_root "$RUN_DIR" \
      --output "$RUN_DIR/aggregated" \
      2>&1 | tee "$RUN_DIR/logs/aggregate.log"

  # Paper figures (best-effort; failures are non-fatal)
  FIGURE_SCRIPTS=(
    scripts/figures/figure_cliff_detection.py
    scripts/figures/figure_phase_trajectory.py
    scripts/figures/figure_ablation_bar.py
    scripts/figures/figure_regret_scaling.py
    scripts/figures/figure_boundary_loss_ratio.py
  )
  for fig in "${FIGURE_SCRIPTS[@]}"; do
    if [[ -f "$REPO_ROOT/$fig" ]]; then
      python "$REPO_ROOT/$fig" \
          --input_root "$RUN_DIR" \
          --output_dir "$RUN_DIR/figures" \
          2>&1 | tee "$RUN_DIR/logs/$(basename "$fig" .py).log" || true
    fi
  done

  # Final snapshot of aggregated + figures
  python "$REPO_ROOT/scripts/utils/snapshot_experiment.py" \
      --src "$RUN_DIR/aggregated" \
      --dst "${PACE_SNAPSHOT_DIR:-/root/autodl-tmp/snapshots}/final_aggregated_$(date +%Y%m%d_%H%M%S).tar.gz" || true
  python "$REPO_ROOT/scripts/utils/snapshot_experiment.py" \
      --src "$RUN_DIR/figures" \
      --dst "${PACE_SNAPSHOT_DIR:-/root/autodl-tmp/snapshots}/final_figures_$(date +%Y%m%d_%H%M%S).tar.gz" || true

  log "Phase 8 complete."
  log "Outputs: $RUN_DIR"
}

# ================================================================== dispatch
case "$PHASE" in
  preflight)
    phase_2_deps
    phase_3_verify
    phase_4_autobatch
    phase_5_coverage
    log "=== Preflight complete. Run 'bash scripts/run_autodl_pipeline.sh train' to start training. ==="
    ;;
  train)
    phase_6_train
    ;;
  post)
    phase_8_post
    ;;
  all)
    log "=== Running ALL phases. Abort within 10 seconds with Ctrl-C ==="
    sleep 10
    phase_2_deps
    phase_3_verify
    phase_4_autobatch
    phase_5_coverage
    phase_6_train
    phase_8_post
    log "=== All phases complete. ==="
    ;;
  *)
    echo "Usage: $0 [preflight|train|post|all]" >&2
    exit 1
    ;;
esac
