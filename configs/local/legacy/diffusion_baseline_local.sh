#!/usr/bin/env bash

#
# Launch a minimal diffusion baseline training run on the Smol‑LIBERO dataset.
# This script is intended for small‑GPU debugging on your local machine.
# You must have LeRobot installed and the `lerobot_policy_phaseqflow` package
# available in your Python environment. Adjust the dataset, batch size,
# and number of steps as necessary for your environment.

set -e

OUTPUT_DIR="outputs/train/diffusion_smol_local"
mkdir -p "$OUTPUT_DIR"

lerobot-train \
  --output_dir "$OUTPUT_DIR" \
  --policy.type diffusion \
  --dataset.repo_id HuggingFaceVLA/smol-libero \
  --env.type libero \
  --batch_size 16 \
  --steps 3000 \
  --eval_freq 1000 \
  --policy.device cuda \
  --policy.use_amp false

echo "Diffusion baseline local training completed. Check $OUTPUT_DIR for checkpoints."