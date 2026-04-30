#!/usr/bin/env bash
# scripts/run_experiments.sh
# Master orchestration for Phase D experiments.
#
# Usage:
#   bash scripts/run_experiments.sh [--dry_run] [--checkpoint PATH]
#
# In dry_run mode all scripts use synthetic data; no checkpoints or
# real environments are required.  This is the default when no
# --checkpoint is provided.

set -euo pipefail

SEEDS=(0 1 2)
N_ROLLOUTS="${N_ROLLOUTS:-50}"
CHECKPOINT="${CHECKPOINT:-}"
DRY_RUN="${DRY_RUN:-0}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Parse CLI flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run)    DRY_RUN="1"; shift ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --n_rollouts) N_ROLLOUTS="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ "$DRY_RUN" == "1" || -z "$CHECKPOINT" ]]; then
    DRY_FLAG="--dry_run"
    echo "[run_experiments] DRY RUN mode (synthetic data)"
else
    DRY_FLAG=""
    echo "[run_experiments] REAL mode — checkpoint: $CHECKPOINT"
fi

SEEDS_STR="${SEEDS[*]}"

# ---------------------------------------------------------------------------
# Phase D-1: Universality (§4.1)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase D-1: Universality ==="
python scripts/phenomenon/universality.py \
    $DRY_FLAG \
    --n_rollouts "$N_ROLLOUTS" \
    --seeds $SEEDS_STR \
    --output paper_figures/universality/

# ---------------------------------------------------------------------------
# Phase D-2: Regret scaling (§4.2)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase D-2: Regret scaling ==="
for H in 4 8 16 32 64; do
    python scripts/phenomenon/regret_scaling.py \
        $DRY_FLAG \
        ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
        --H "$H" \
        --n_rollouts "$N_ROLLOUTS" \
        --seeds $SEEDS_STR \
        --output paper_figures/regret_scaling/
done

# ---------------------------------------------------------------------------
# Phase D-3: Triangulation concordance (§4.3)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase D-3: Triangulation ==="
python scripts/phenomenon/triangulation_concordance.py \
    $DRY_FLAG \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    --n_episodes "$N_ROLLOUTS" \
    --output paper_figures/triangulation/

# ---------------------------------------------------------------------------
# Main evaluation (§4.4 LIBERO-Perturbed + §4.5 SimplerEnv)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase D-4: LIBERO-Perturbed evaluation ==="
bash scripts/eval/run_libero_main.sh \
    ${DRY_RUN:+--dry_run} \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    --n_rollouts "$N_ROLLOUTS"

echo ""
echo "=== Phase D-5: SimplerEnv evaluation ==="
bash scripts/eval/run_simpler.sh \
    ${DRY_RUN:+--dry_run} \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    --n_rollouts "$N_ROLLOUTS"

# ---------------------------------------------------------------------------
# Aggregate results (§4.6)
# ---------------------------------------------------------------------------
echo ""
echo "=== Aggregating results ==="
python scripts/aggregate_results.py \
    --figures_dir paper_figures/ \
    --output paper_figures/main_results.csv

echo ""
echo "[run_experiments] All Phase D experiments complete."
echo "  Main results: paper_figures/main_results.csv"
