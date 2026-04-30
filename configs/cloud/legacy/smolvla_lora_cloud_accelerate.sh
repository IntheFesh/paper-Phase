#!/usr/bin/env bash

#
# Launch a multi‑GPU SmolVLA PEFT/LoRA fine‑tuning run. This script is
# intended for cloud environments with multiple GPUs.

set -e

OUTPUT_DIR="outputs/train/smolvla_lora_libero_cloud"
mkdir -p "$OUTPUT_DIR"

accelerate launch \
  lerobot-train \
  --output_dir "$OUTPUT_DIR" \
  --policy.type smolvla \
  --policy.adapter lora \
  --dataset.repo_id HuggingFaceVLA/libero \
  --env.type libero \
  --batch_size 16 \
  --steps 50000 \
  --eval_freq 2500 \
  --policy.device cuda \
  --policy.use_amp true \
  --policy.lora_rank 16 \
  --policy.num_epochs 2

echo "SmolVLA LoRA cloud training started. Monitor logs for progress."