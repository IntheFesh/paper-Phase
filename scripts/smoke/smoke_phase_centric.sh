#!/usr/bin/env bash

# Round 2 Phase-Centric VLA: 7-mode smoke test runner.
#
# Runs a 3-step dummy-batch training for each preset mode, verifying that:
# 1. the config mode preset loads;
# 2. the model can compute_loss + backward + optimizer.step;
# 3. flipping every switch on (mode=full) does not crash (because the
# phase_centric/ modules are not yet wired into compute_loss, so even with
# use_chunk_infonce=True the Round-3 logic is not reached -- this is the
# intended Round 2 design).
#
# PASS requires all 7 modes to exit 0; any non-zero exit aborts the run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODES=(off ident_only pace_a pace_b pace_c pcar full)
STEPS="${STEPS:-3}"
DEVICE="${DEVICE:-cpu}"

echo "===================================================================="
echo "Round 2 Phase-Centric smoke: ${#MODES[@]} modes x ${STEPS} steps on ${DEVICE}"
echo "===================================================================="

FAILED_MODES=()
for mode in "${MODES[@]}"; do
  echo ""
  echo "--------------------------------------------------------------------"
  echo " mode=${mode}"
  echo "--------------------------------------------------------------------"
  if python scripts/training/train_dummy_batch.py \
      --phase-centric-mode "${mode}" \
      --steps "${STEPS}" \
      --device "${DEVICE}"; then
    echo " [OK] mode=${mode}"
  else
    echo " [FAIL] mode=${mode}"
    FAILED_MODES+=("${mode}")
  fi
done

echo ""
echo "===================================================================="
if [ "${#FAILED_MODES[@]}" -eq 0 ]; then
  echo "[PASS] All ${#MODES[@]} modes passed"
  exit 0
else
  echo "[FAIL] Failing modes: ${FAILED_MODES[*]}"
  exit 1
fi
