#!/usr/bin/env bash

#
# Launch a lightweight SmolVLA PEFT/LoRA fine‑tuning run. This script is
# provided as an example of how to invoke the SmolVLA training pipeline
# described in the LeRobot documentation. Adjust parameters as needed.

set -e

OUTPUT_DIR="outputs/train/smolvla_lora_local"
mkdir -p "$OUTPUT_DIR"

lerobot-train \
  --output_dir "$OUTPUT_DIR" \
  --policy.type smolvla \
  --policy.adapter lora \
  --dataset.repo_id HuggingFaceVLA/smol-libero \
  --env.type libero \
  --batch_size 4 \
  --steps 2000 \
  --eval_freq 500 \
  --policy.device cuda \
  --policy.use_amp false \
  --policy.lora_rank 8 \
  --policy.num_epochs 1

echo "SmolVLA LoRA fine‑tuning completed. Check $OUTPUT_DIR for checkpoints."