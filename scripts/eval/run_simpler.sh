#!/usr/bin/env bash
# scripts/eval/run_simpler.sh
# SimplerEnv Google Robot Visual Matching evaluation.
#
# Usage:
#   bash scripts/eval/run_simpler.sh [--checkpoint PATH] [--dry_run]
#
# Environment variables:
#   CHECKPOINT  — path to PhaseQFlow checkpoint
#   N_ROLLOUTS  — rollouts per task (default 50)
#   SEEDS       — space-separated seed list (default "0 1 2")
#   DRY_RUN     — set to "1" to run without SimplerEnv or checkpoint

set -euo pipefail

CHECKPOINT="${CHECKPOINT:-}"
N_ROLLOUTS="${N_ROLLOUTS:-50}"
SEEDS="${SEEDS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT="paper_figures/simpler"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

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
    echo "[run_simpler] WARNING: no --checkpoint and DRY_RUN!=1; running dry_run mode"
    DRY_FLAG="--dry_run"
fi

echo "[run_simpler] tasks: pick_coke_can move_near open_drawer put_eggplant_in_basket"
python scripts/eval/simpler.py \
    $DRY_FLAG $CKPT_FLAG \
    --tasks pick_coke_can move_near open_drawer put_eggplant_in_basket \
    --n_rollouts "$N_ROLLOUTS" \
    --seeds $SEEDS \
    --output "$OUTPUT"

echo "[run_simpler] done — outputs at $OUTPUT/"
