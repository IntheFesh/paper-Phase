#!/usr/bin/env bash

#
# Evaluate a policy checkpoint on LIBERO using LeRobot evaluation.
# Usage: run_eval_libero.sh <policy_path> [suite]

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <policy_path> [suite_list]"
  echo "Example: $0 outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model libero_task_0"
  exit 1
fi

POLICY_PATH="$1"
SUITES="${2:-libero_task_0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/$(basename "$POLICY_PATH")}"
DEVICE="${DEVICE:-cuda}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"

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
  --env.type libero \
  --env.suite "$SUITES" \
  --eval.n_episodes "$EVAL_EPISODES" \
  --eval.batch_size "$EVAL_BATCH_SIZE" \
  --policy.device "$DEVICE" \
  --policy.use_amp false \
  --output_dir "$OUTPUT_DIR"

echo "Evaluation complete. Logs and metrics written to $OUTPUT_DIR"
