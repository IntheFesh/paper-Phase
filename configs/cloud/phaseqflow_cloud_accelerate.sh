#!/usr/bin/env bash

#
# Launch a multi-GPU PhaseQFlow++ training run with accelerate.
# All policy flags mirror lerobot_policy_phaseqflow configuration names.

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/phaseqflow_libero_cloud}"
DATASET_REPO_ID="${DATASET_REPO_ID:-HuggingFaceVLA/libero}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-100000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
NUM_PHASES="${NUM_PHASES:-4}"
USE_VALUE_GUIDED_WEIGHT="${USE_VALUE_GUIDED_WEIGHT:-true}"
USE_LATENT_FLOW="${USE_LATENT_FLOW:-true}"

mkdir -p "$OUTPUT_DIR"

accelerate launch \
  lerobot-train \
  --output_dir "$OUTPUT_DIR" \
  --policy.discover_packages_path lerobot_policy_phaseqflow \
  --policy.type phaseqflow \
  --dataset.repo_id "$DATASET_REPO_ID" \
  --env.type libero \
  --batch_size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --eval_freq "$EVAL_FREQ" \
  --policy.device cuda \
  --policy.use_amp true \
  --policy.num_phases "$NUM_PHASES" \
  --policy.use_value_guided_weight "$USE_VALUE_GUIDED_WEIGHT" \
  --policy.use_latent_flow "$USE_LATENT_FLOW"

echo "PhaseQFlow++ cloud training started. Monitor logs for progress."
