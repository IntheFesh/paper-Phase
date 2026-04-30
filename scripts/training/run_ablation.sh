#!/usr/bin/env bash

#
# Ablation matrix driver.
#
# 12 configs x N seeds = up to 36 training runs. Each run walks through:
# 1. scripts/training/train_dummy_batch.py --phase-centric-mode $cfg --seed $seed --total_steps N
#    -> writes eval_results.json under $out_dir (placeholder=true on CPU, see below).
# 2. scripts/evaluation/run_eval_libero.sh <checkpoint> "libero_10,libero_spatial"
#    -> overwrites the SR fields with real numbers on GPU + LIBERO and sets placeholder=false.
# 3. touch eval_done.marker (resume-friendly).
#
# Usage
# -----
# # Quick CPU dry-run (3 steps x 2 seeds x 12 configs, for pipeline plumbing)
# TOTAL_STEPS=3 SEEDS="42 123" DEVICE=cpu SKIP_EVAL=1 \
#   bash scripts/training/run_ablation.sh
#
# # Real RTX 5070 run (12 x 3 seeds x 20k steps ~ 12 days of wall time)
# TOTAL_STEPS=20000 SEEDS="42 123 2024" DEVICE=cuda \
#   bash scripts/training/run_ablation.sh
#
# # Resume (eval_done.marker present -> automatically skipped).
#
# Environment variables
# ---------------------
# CONFIGS     : space-separated config names; defaults to the 12 matrix configs.
# SEEDS       : space-separated random seeds; default "42 123 2024".
# TOTAL_STEPS : training steps; default 20000 (CPU dry-run usually 3).
# DEVICE      : cpu / cuda; default cuda.
# OUTPUT_ROOT : output root directory; default outputs/ablation.
# SKIP_EVAL   : when non-empty, skip run_eval_libero.sh (CPU dry-run; the
#               eval_results.json file keeps placeholder=true).
# AGGREGATE   : when non-empty, run aggregate + figures + latex at the end
#               (default on).
#
# Acceptance
# ----------
# Exit code 0 iff every (config, seed) finished train + eval or was skipped;
# any train/eval failure causes the driver to continue the remaining
# combinations and list the FAILED set at the end.

set -u # Intentional: no set -e so that one failing config does not abort the aggregate.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIGS="${CONFIGS:-baseline ident a b c ab ac bc pace pcar_only full pcar_noident}"
SEEDS="${SEEDS:-42 123 2024}"
TOTAL_STEPS="${TOTAL_STEPS:-20000}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ablation}"
SKIP_EVAL="${SKIP_EVAL:-}"
AGGREGATE="${AGGREGATE:-1}"

mkdir -p "${OUTPUT_ROOT}"

echo "===================================================================="
echo "Round 8 ablation driver"
echo " configs     : ${CONFIGS}"
echo " seeds       : ${SEEDS}"
echo " total_steps : ${TOTAL_STEPS}"
echo " device      : ${DEVICE}"
echo " output_root : ${OUTPUT_ROOT}"
echo " skip_eval   : ${SKIP_EVAL:-no}"
echo "===================================================================="

FAILED=()
START_TS="$(date +%s)"

for cfg in ${CONFIGS}; do
  for seed in ${SEEDS}; do
    run_name="${cfg}_seed${seed}"
    out_dir="${OUTPUT_ROOT}/${run_name}"
    marker="${out_dir}/eval_done.marker"
    mkdir -p "${out_dir}"

    echo ""
    echo "-- ${run_name} ----------------------------------------------------"
    if [ -f "${marker}" ]; then
      echo " [SKIP] eval_done.marker exists at ${marker}"
      continue
    fi

    # Step 1: training.
    if ! python scripts/training/train_dummy_batch.py \
        --phase-centric-mode "${cfg}" \
        --seed "${seed}" \
        --total_steps "${TOTAL_STEPS}" \
        --device "${DEVICE}" \
        --output_dir "${out_dir}" \
        > "${out_dir}/train.log" 2>&1; then
      echo " [FAIL train] ${run_name} (see ${out_dir}/train.log)"
      FAILED+=("${run_name}:train")
      continue
    fi
    echo " [OK train] ${run_name}"

    # Step 2: LIBERO eval - skip on CPU dry-run.
    if [ -z "${SKIP_EVAL}" ]; then
      ckpt="${out_dir}/checkpoints/last/pretrained_model"
      if [ ! -d "${ckpt}" ]; then
        echo " [WARN eval] no checkpoint at ${ckpt}; eval skipped (train wrote placeholder only)"
      else
        OUTPUT_DIR="${out_dir}/eval" DEVICE="${DEVICE}" EVAL_EPISODES=50 \
          bash scripts/evaluation/run_eval_libero.sh "${ckpt}" "libero_10,libero_spatial" \
          > "${out_dir}/eval.log" 2>&1 || {
            echo " [FAIL eval] ${run_name} (see ${out_dir}/eval.log)"
            FAILED+=("${run_name}:eval")
            continue
        }
        # Merge eval JSON into eval_results.json with placeholder=false.
        python scripts/paper/aggregate_ablation.py \
          --merge-eval "${out_dir}/eval" \
          --target "${out_dir}/eval_results.json" \
          >> "${out_dir}/eval.log" 2>&1 || true
        echo " [OK eval] ${run_name}"
      fi
    else
      echo " [SKIP eval] SKIP_EVAL set; placeholder eval_results.json kept"
    fi

    touch "${marker}"
  done
done

ELAPSED=$(( $(date +%s) - START_TS ))
echo ""
echo "===================================================================="
if [ "${#FAILED[@]}" -eq 0 ]; then
  echo "[DONE] all runs completed in ${ELAPSED}s"
else
  echo "[WARN] ${#FAILED[@]} failed run(s):"
  printf ' - %s\n' "${FAILED[@]}"
fi
echo "===================================================================="

if [ -n "${AGGREGATE}" ]; then
  echo ""
  echo "-- aggregate + figures + latex ------------------------------------"
  python scripts/paper/aggregate_ablation.py \
    --output_root "${OUTPUT_ROOT}" \
    --out_dir artifacts/ablation || true
  python scripts/paper/generate_paper_figures.py \
    --in_csv artifacts/ablation/ablation_table_long.csv \
    --spatial_csv artifacts/ablation/ablation_table_spatial.csv \
    --stats_json artifacts/ablation/stats.json \
    --out_dir paper_figures || true
  python scripts/paper/generate_latex_table.py \
    --stats_json artifacts/ablation/stats.json \
    --out paper_figures/ablation_table.tex || true
  python scripts/paper/generate_paper_stats.py \
    --stats_json artifacts/ablation/stats.json \
    --out artifacts/paper_stats.md || true
fi

# Exit code reflects whether every train/eval succeeded (aggregate errors are ignored).
if [ "${#FAILED[@]}" -eq 0 ]; then
  exit 0
fi
exit 1
