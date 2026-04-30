#!/usr/bin/env bash

# End-to-end smoke test for PhaseQFlow (train + eval, tiny settings).
# This script is intentionally minimal and intended for compatibility checks.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

TRAIN_OUT="${TRAIN_OUT:-outputs/train/phaseqflow_smoke}"
POLICY_DIR="$TRAIN_OUT/checkpoints/last/pretrained_model"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  export DEVICE="cpu"
  echo "[smoke] nvidia-smi not found, forcing DEVICE=cpu"
else
  export DEVICE="${DEVICE:-cuda}"
fi

export OUTPUT_DIR="$TRAIN_OUT"
export STEPS="${STEPS:-2}"
export EVAL_FREQ="${EVAL_FREQ:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export EVAL_EPISODES="${EVAL_EPISODES:-1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

if [ ! -d "lerobot_policy_phaseqflow" ]; then
  echo "[smoke] policy package folder not found"
  exit 1
fi

echo "[smoke] installing local policy package (editable)"
python -m pip install --no-build-isolation -e ./lerobot_policy_phaseqflow >/dev/null

echo "[smoke] running tiny training job"
bash configs/local/phaseqflow_local.sh

if [ ! -d "$POLICY_DIR" ]; then
  echo "[smoke] expected policy checkpoint not found: $POLICY_DIR"
  exit 1
fi

echo "[smoke] running tiny evaluation job"
bash scripts/evaluation/run_eval_libero.sh "$POLICY_DIR" "libero_task_0"

echo "[smoke] done"
