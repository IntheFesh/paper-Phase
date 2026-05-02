#!/usr/bin/env bash
# scripts/eval/run_libero_main.sh
# Standard LIBERO-Long evaluation for PhaseQFlow++ (50 rollouts, 3 seeds).
#
# Usage:
#   bash scripts/eval/run_libero_main.sh [--checkpoint PATH] [--dry_run]
#
# Environment variables (override defaults):
#   CHECKPOINT  — path to PhaseQFlow checkpoint
#   N_ROLLOUTS  — rollouts per task (default 50)
#   SEEDS       — space-separated seed list (default "0 1 2")
#   DRY_RUN     — set to "1" to run without checkpoint / environment

set -euo pipefail

export PYTHONPATH=/root/LIBERO:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/root/miniconda3/envs/lerobot_env/lib:${LD_LIBRARY_PATH:-}

CHECKPOINT="${CHECKPOINT:-}"
N_ROLLOUTS="${N_ROLLOUTS:-50}"
SEEDS="${SEEDS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT="paper_figures/libero_perturbed"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Parse CLI flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --dry_run)    DRY_RUN="1"; shift ;;
        --n_rollouts) N_ROLLOUTS="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

DRY_FLAG=""
CKPT_FLAG=""
if [[ "$DRY_RUN" == "1" ]]; then
    DRY_FLAG="--dry_run"
elif [[ -n "$CHECKPOINT" ]]; then
    CKPT_FLAG="--checkpoint $CHECKPOINT"
else
    echo "[run_libero_main] WARNING: no --checkpoint and DRY_RUN!=1; running dry_run mode"
    DRY_FLAG="--dry_run"
fi

echo "[run_libero_main] perturbation = 5 cm, n_rollouts = $N_ROLLOUTS"
python scripts/eval/libero_perturbed.py \
    $DRY_FLAG $CKPT_FLAG \
    --n_rollouts "$N_ROLLOUTS" \
    --seeds $SEEDS \
    --perturbation_cm 5.0 \
    --output "$OUTPUT"

echo "[run_libero_main] done — outputs at $OUTPUT/"
