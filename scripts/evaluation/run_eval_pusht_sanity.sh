#!/usr/bin/env bash

#
# PhaseQFlow-focused PushT sanity evaluation.
# Usage:
# bash scripts/evaluation/run_eval_pusht_sanity.sh <policy_path>
# If no path is provided, it will try the local PhaseQFlow smoke checkpoint.

set -euo pipefail

DEFAULT_POLICY="outputs/train/phaseqflow_smoke/checkpoints/last/pretrained_model"
POLICY_PATH="${1:-$DEFAULT_POLICY}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/pusht_$(basename "$POLICY_PATH")}"
DEVICE="${DEVICE:-cuda}"

if [ ! -d "$POLICY_PATH" ]; then
  echo "Error: policy path not found: $POLICY_PATH"
  echo "Run scripts/smoke/smoke_test_phaseqflow_e2e.sh first or pass an explicit checkpoint path."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if command -v lerobot-eval >/dev/null 2>&1; then
  EVAL_CMD=(lerobot-eval)
elif python -c "import lerobot" >/dev/null 2>&1; then
  EVAL_CMD=(python -m lerobot.scripts.eval)
else
  echo "Error: LeRobot evaluation entrypoint not found."
  echo "Install LeRobot first, then re-run."
  exit 127
fi

"${EVAL_CMD[@]}" \
  --policy.path "$POLICY_PATH" \
  --env.type pusht \
  --eval.n_episodes 20 \
  --eval.batch_size 10 \
  --policy.device "$DEVICE" \
  --policy.use_amp false \
  --output_dir "$OUTPUT_DIR"

echo "PushT sanity evaluation complete. Results in $OUTPUT_DIR"
