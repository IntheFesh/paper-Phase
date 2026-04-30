#!/usr/bin/env bash

#
# Launch a PhaseQFlow++ training run on Smol-LIBERO.
# Parameters are aligned with lerobot_policy_phaseqflow configuration fields.

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/phaseqflow_smol_local}"
DATASET_REPO_ID="${DATASET_REPO_ID:-HuggingFaceVLA/smol-libero}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-5000}"
EVAL_FREQ="${EVAL_FREQ:-1000}"
NUM_PHASES="${NUM_PHASES:-4}"
USE_VALUE_GUIDED_WEIGHT="${USE_VALUE_GUIDED_WEIGHT:-true}"
USE_LATENT_FLOW="${USE_LATENT_FLOW:-true}"

mkdir -p "$OUTPUT_DIR"

if command -v lerobot-train >/dev/null 2>&1; then
  TRAIN_CMD=(lerobot-train)
elif python -c "import lerobot" >/dev/null 2>&1; then
  TRAIN_CMD=(python -m lerobot.scripts.train)
else
  echo "Error: LeRobot training entrypoint not found."
  echo "Install LeRobot first, then re-run."
  exit 127
fi

"${TRAIN_CMD[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --policy.discover_packages_path lerobot_policy_phaseqflow \
  --policy.type phaseqflow \
  --dataset.repo_id "$DATASET_REPO_ID" \
  --env.type libero \
  --batch_size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --eval_freq "$EVAL_FREQ" \
  --policy.device "$DEVICE" \
  --policy.use_amp false \
  --policy.num_phases "$NUM_PHASES" \
  --policy.use_value_guided_weight "$USE_VALUE_GUIDED_WEIGHT" \
  --policy.use_latent_flow "$USE_LATENT_FLOW"

echo "PhaseQFlow++ training completed. Check $OUTPUT_DIR for checkpoints."
