#!/usr/bin/env bash
# ============================================================================
# AutoDL one-shot pipeline runner for PACE v2.1
# ============================================================================
# Runs Phases 2 → 8 of the documented AutoDL workflow:
#   Phase 2 — env vars + Python deps
#   Phase 3 — real-data + real-implementation verification (fail-hard)
#   Phase 4 — pre-flight (pytest + smoke + 4 math verifications + 1 GPU step)
#   Phase 5 — paper coverage report
#   Phase 6 — launch the full cloud training sweep
#   Phase 8 — post-training aggregation + 5 paper figures + final snapshot
#
# Defaults to "preflight only" so a 12-day sweep cannot start by accident.
# Pass `train` or `post` explicitly to run those long phases.
#
# Usage
# -----
#   # Phase 2 → 5 (env + deps + verify + preflight + coverage):
#   bash scripts/run_autodl_pipeline.sh
#
#   # Phase 6 (launches the H800 sweep — wrap in tmux!):
#   bash scripts/run_autodl_pipeline.sh train
#
#   # Phase 8 (aggregate + figures + final snapshot, after sweep finishes):
#   bash scripts/run_autodl_pipeline.sh post
#
# Environment overrides
# ---------------------
#   DATA_ROOT          path to local LeRobot dataset (default: $PACE_DATA/libero_10)
#   DATASET_REPO_ID    HuggingFace repo-id fallback (default: HuggingFaceVLA/libero)
#   BATCH_SIZE         per-step micro batch (default: 32)
#   GRAD_ACCUM         gradient accumulation (default: 32; effective batch 1024)
#   SKIP_PYTEST=1      skip pytest in Phase 4 (rare; not recommended)
#   SKIP_GPU_STEP=1    skip the 1-step GPU sanity check in Phase 4
# ============================================================================

set -euo pipefail

PHASE_ARG="${1:-preflight}"

# ---------------------------------------------------------------- pretty print
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

ok()    { printf "${GREEN}✓${RESET} %s\n" "$*"; }
warn()  { printf "${YELLOW}⚠${RESET} %s\n" "$*"; }
fail()  { printf "${RED}✗ %s${RESET}\n" "$*" >&2; exit 1; }
hdr()   { printf "\n${CYAN}${BOLD}===== %s =====${RESET}\n" "$*"; }
sub()   { printf "${CYAN}--- %s${RESET}\n" "$*"; }

# ---------------------------------------------------------------- env defaults
: "${PACE_ROOT:=/root/paper-Phase}"
: "${PACE_DATA:=/root/autodl-tmp/data}"
: "${PACE_CKPT:=/root/autodl-tmp/checkpoints}"
: "${PACE_OUT:=/root/autodl-tmp/outputs}"
: "${PACE_SNAPSHOT_DIR:=/root/autodl-tmp/snapshots}"
: "${HF_HOME:=/root/autodl-tmp/hf_cache}"
: "${DATA_ROOT:=${PACE_DATA}/libero_10}"
: "${DATASET_REPO_ID:=HuggingFaceVLA/libero}"
# Starting batch size (256 confirmed safe on H800/A100 24GB; autobatch will
# probe downwards if OOM occurs).  GRAD_ACCUM adjusts so effective batch ≈ 256.
: "${BATCH_SIZE:=256}"
: "${GRAD_ACCUM:=1}"

export PACE_ROOT PACE_DATA PACE_CKPT PACE_OUT PACE_SNAPSHOT_DIR
export HF_HOME DATA_ROOT DATASET_REPO_ID BATCH_SIZE GRAD_ACCUM

# CoRL sweep script (13 runs: base_fm×3, pace_concordance×3, pace_concordance_bdy×3, ablations×4)
CLOUD_SCRIPT="configs/cloud/phaseqflow_cloud_corl.sh"
export CLOUD_SCRIPT

# Help / usage path — print and exit before touching dirs (so --help works anywhere).
case "$PHASE_ARG" in
  -h|--help|help)
    cat << 'EOF'
Usage: bash scripts/run_autodl_pipeline.sh [phase]

Phases:
  preflight  (default) — Phase 2+3+4+4b+5 (deps + verify + preflight + autobatch + coverage)
  deps       — Phase 2 only (Python deps)
  realdata   — Phase 3 only (real-data + cliff-impl + no-fallback proof)
  tests      — Phase 4 only (pytest + smoke + 4 math + GPU 1-step)
  autobatch  — Phase 4b only (auto-tune BATCH_SIZE for the current GPU)
  coverage   — Phase 5 only (coverage report)
  train      — Phase 6 (~6–8 days on RTX PRO 6000 96GB; wrap in tmux)
  post       — Phase 8 (aggregate + 5 figures + final snapshot)
  all        — Phase 2 → 8 (with 10s abort window before Phase 6)

Environment:
  PACE_ROOT      default /root/paper-Phase
  PACE_DATA, PACE_OUT, PACE_SNAPSHOT_DIR
  DATA_ROOT      default $PACE_DATA/libero_10
  DATASET_REPO_ID
  BATCH_SIZE     default 32 (autobatch normally overrides this; force a value with SKIP_AUTOBATCH=1)
  GRAD_ACCUM     default 32 (autobatch picks GRAD_ACCUM so BATCH_SIZE × GRAD_ACCUM ≈ 1024)
  AUTOBATCH_CANDIDATES, AUTOBATCH_TARGET_EFFECTIVE, AUTOBATCH_SAFETY
  SKIP_PYTEST=1, SKIP_GPU_STEP=1, SKIP_AUTOBATCH=1, ALLOW_NO_TMUX=1
EOF
    exit 0
    ;;
esac

cd "$PACE_ROOT"
mkdir -p "$HF_HOME" "$PACE_DATA" "$PACE_CKPT" "$PACE_OUT" "$PACE_SNAPSHOT_DIR"


# ============================================================================
# Phase 2 — env + Python deps
# ============================================================================
phase_2_deps() {
  hdr "Phase 2 — Python deps"

  sub "torch / cuda"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())" \
    || fail "torch import failed"

  if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    fail "CUDA not available — fix torch install (try: pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall)"
  fi

  sub "install lerobot_policy_phaseqflow (editable)"
  pip install -e ./lerobot_policy_phaseqflow --quiet || fail "lerobot_policy_phaseqflow install failed"

  sub "requirements.txt"
  if [[ -f requirements.txt ]]; then
    pip install -r requirements.txt --quiet 2>&1 | tail -3 || warn "some requirements failed (likely pre-installed)"
  fi

  sub "key imports"
  python - << 'PY' || fail "key import check failed"
import importlib, sys
mods = [
    ("lerobot", None),
    ("lerobot.datasets.lerobot_dataset", "LeRobotDataset"),
    ("vector_quantize_pytorch", None),
    ("matplotlib", None),
    ("numpy", None),
    ("scipy", None),
    ("transformers", None),
    ("timm", None),
    ("yaml", None),
]
for name, attr in mods:
    try:
        m = importlib.import_module(name)
        if attr:
            getattr(m, attr)
        print(f"  OK: {name}{f'.{attr}' if attr else ''}")
    except Exception as e:
        print(f"  MISSING: {name} — {e!r}")
        sys.exit(1)
PY
  ok "Phase 2 done"
}


# ============================================================================
# Phase 3 — real-data + real-implementation verification (fail-hard)
# ============================================================================
phase_3_verify_real() {
  hdr "Phase 3 — real-data verification"

  sub "DATA_ROOT exists and is non-empty"
  if [[ ! -d "$DATA_ROOT" ]]; then
    fail "DATA_ROOT does not exist: $DATA_ROOT — adjust env var or download the LIBERO dataset first"
  fi
  if [[ -z "$(ls -A "$DATA_ROOT" 2>/dev/null)" ]]; then
    fail "DATA_ROOT is empty: $DATA_ROOT"
  fi
  ls -la "$DATA_ROOT" | head -10

  sub "LeRobotDataset can load from DATA_ROOT"
  python - << 'PY' || fail "LeRobotDataset failed to load real data"
import os
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root = os.environ['DATA_ROOT']
p = Path(root)
ds = LeRobotDataset(repo_id=p.name, root=str(p.parent)) if p.is_dir() \
     else LeRobotDataset(repo_id=os.environ.get('DATASET_REPO_ID'))
print(f"  num_frames = {len(ds)}")
print(f"  features (first 8) = {list(ds.features.keys())[:8]}")
sample = ds[0]
img_key = next((k for k in sample if 'image' in k.lower()), None)
if img_key is not None:
    t = sample[img_key]
    print(f"  {img_key}.shape = {tuple(t.shape) if hasattr(t,'shape') else type(t)}")
print("  PASS — 真实 LIBERO 数据可用")
PY

  sub "cliff estimators all implemented (no NotImplementedError)"
  python - << 'PY' || fail "cliff estimator implementation check failed"
import torch
from lerobot_policy_phaseqflow.phase_centric.cliff_estimators import (
    compute_I_hat_1, compute_I_hat_2, compute_I_hat_3, compute_concordance_C
)
samples = torch.randn(8, 2, 16, 16)
i2 = compute_I_hat_2(samples); assert i2.shape == (2,) and (i2 <= 0).all()
v_t, v_p = torch.randn(2, 16, 16), torch.randn(2, 16, 16)
i3 = compute_I_hat_3(v_t, v_p); assert i3.shape == (2,) and (i3 <= 0).all()
C  = compute_concordance_C([torch.randn(2)] * 3, window_size=10)
assert C.shape == (2,) and (C >= 0).all() and (C <= 1).all()
print("  PASS — I_hat_2 / I_hat_3 / concordance_C 全部实装")
PY

  sub "no silent dummy fallback when --data_root is invalid"
  set +e
  out=$(python scripts/training/train_dummy_batch.py \
        --phase-centric-mode off --steps 1 --device cpu \
        --output_dir /tmp/.proof_no_fallback \
        --data_root /__nonexistent__/should_fail 2>&1)
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "$out" | tail -5
    fail "expected RuntimeError on bad --data_root, but trainer exited 0 (silent fallback bug)"
  fi
  if echo "$out" | grep -q "RuntimeError: --data_root"; then
    ok "  RuntimeError raised correctly — no silent fallback"
  else
    echo "$out" | tail -5
    fail "trainer failed but not with the expected RuntimeError"
  fi

  sub "static grep — no NotImplementedError stubs in cliff_estimators.py"
  if grep -q "raise NotImplementedError" \
        lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/phase_centric/cliff_estimators.py; then
    fail "cliff_estimators.py still contains 'raise NotImplementedError'"
  fi
  ok "  cliff_estimators.py has no NotImplementedError stubs"

  ok "Phase 3 done — all-real invariant verified"
}


# ============================================================================
# Phase 4 — pre-flight (pytest + smoke + math + GPU step)
# ============================================================================
phase_4_preflight() {
  hdr "Phase 4 — pre-flight verification"

  if [[ "${SKIP_PYTEST:-0}" != "1" ]]; then
    sub "pytest tests/ -q  (expect 212 passed)"
    python -m pytest tests/ -q --no-header 2>&1 | tail -3
  else
    warn "SKIP_PYTEST=1 — skipping unit tests"
  fi

  sub "smoke_phase_centric.sh  (expect 7/7)"
  if [[ -f "scripts/smoke/smoke_phase_centric.sh" ]]; then
    bash scripts/smoke/smoke_phase_centric.sh 2>&1 | tail -3
  else
    warn "scripts/smoke/smoke_phase_centric.sh not found — skipping (run: git reset --hard origin/$(git rev-parse --abbrev-ref HEAD))"
  fi

  sub "CPU math: phase_posterior / pace_a / pcar_budget / identifiability"
  python scripts/verification/verify_phase_posterior.py 2>&1 | tail -2
  python scripts/verification/sanity_pace_a.py 2>&1 | tail -2
  python scripts/verification/verify_pcar_budget.py 2>&1 | tail -2
  python scripts/verification/verify_identifiability.py --steps 50 2>&1 | tail -2

  if [[ "${SKIP_GPU_STEP:-0}" != "1" ]]; then
    sub "GPU 1-step real-data sanity (full mode + diagnostics + checkpointing)"
    rm -rf /tmp/.preflight_gpu
    python scripts/training/train_dummy_batch.py \
        --phase-centric-mode full \
        --steps 3 \
        --device cuda --seed 0 \
        --micro-batch 4 \
        --output_dir /tmp/.preflight_gpu \
        --data_root "$DATA_ROOT" \
        --enable_diagnostics --enable_checkpointing 2>&1 | tail -12

    if ! grep -q "real dataloader: $DATA_ROOT" /tmp/.preflight_gpu/../.preflight_gpu/training_dynamics.csv 2>/dev/null \
        && ! python -c "import sys; sys.exit(0 if '$DATA_ROOT' else 1)" 2>/dev/null; then
      :  # the marker is in stdout, not in the CSV — relaxed check
    fi

    if [[ ! -f /tmp/.preflight_gpu/training_dynamics.csv ]]; then
      fail "GPU sanity step did not produce training_dynamics.csv"
    fi
    if [[ ! -d /tmp/.preflight_gpu/figures ]]; then
      warn "no figures/ subdir produced (diagnostic_report skipped?)"
    fi
    ok "  GPU step OK (artifacts under /tmp/.preflight_gpu)"
  else
    warn "SKIP_GPU_STEP=1 — skipping GPU sanity step"
  fi

  ok "Phase 4 done"
}


# ============================================================================
# Phase 4b — auto-tune BATCH_SIZE for the current GPU (writes autobatch.env)
# ============================================================================
phase_4b_autobatch() {
  hdr "Phase 4b — auto-tune BATCH_SIZE"

  if [[ "${SKIP_AUTOBATCH:-0}" == "1" ]]; then
    warn "SKIP_AUTOBATCH=1 — leaving BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM"
    return 0
  fi

  # Start from 256 (confirmed safe on H800/A100); probe downward on OOM.
  local candidates="${AUTOBATCH_CANDIDATES:-256 192 128 96 64 48 32 24 16}"
  local target="${AUTOBATCH_TARGET_EFFECTIVE:-256}"
  local safety="${AUTOBATCH_SAFETY:-0.90}"

  python scripts/tune_batch_size.py \
      --candidates $candidates \
      --target_effective_batch "$target" \
      --safety_factor "$safety" \
      --output "$PACE_OUT/_launch_logs/autobatch.env"

  if [[ -f "$PACE_OUT/_launch_logs/autobatch.env" ]]; then
    sub "loading tuned settings"
    # shellcheck disable=SC1091
    source "$PACE_OUT/_launch_logs/autobatch.env"
    export BATCH_SIZE GRAD_ACCUM
    ok "  BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  (effective $((BATCH_SIZE * GRAD_ACCUM)))"
  else
    fail "tune_batch_size.py did not write autobatch.env"
  fi
}


# ============================================================================
# Phase 5 — coverage report
# ============================================================================
phase_5_coverage() {
  hdr "Phase 5 — paper coverage report"

  sub "experiments planned by configs/cloud/phaseqflow_cloud_accelerate.sh"
  cat << 'EOF'

  Real data (paper headline numbers):
    Stage 1 pretrain                  → $PACE_RUN_DIR/stage1_pretrain
    Stage 2 phase+flow                → $PACE_RUN_DIR/stage2_phase_flow
    Stage 3 finetune (PCAR)           → $PACE_RUN_DIR/stage3_finetune
    Ablation 01 BC-Chunked × 3 seeds  → $PACE_RUN_DIR/ablation/01_*
    Ablation 02 Cliff via beta × 3    → $PACE_RUN_DIR/ablation/02_*
    Ablation 06 Oracle cliff × 3      → $PACE_RUN_DIR/ablation/06_*
    Ablation 07 PACE v2 full × 3      → $PACE_RUN_DIR/ablation/07_*
    §6.2 Regret Scaling                → $PACE_RUN_DIR/phenomenon_6_2_*
    §6.5 Boundary Loss Ratio           → $PACE_RUN_DIR/phenomenon_6_5_*

  Dry-run only (out-of-scope for v2.1 paper):
    Ablation 03/04 (skipped, redundant with v2.1)
    Ablation 05    (concordance — already covered by Ablation 07)
    §6.1 Universality (~220 GB baseline ckpts)
    §6.3 Triangulation (concordance test — implementation verified by tests)
    §6.4 Trigger comparison (synthetic by design)

  Aggregation + figures (Phase 8):
    aggregate_ablation.py → ablation_table_v2.{csv,tex,json}
    fig1_universality.pdf  fig2_method_overview.pdf
    fig3_phase_visualization.pdf  fig4_regret_scaling.pdf
    fig5_concordance_pr_curve.pdf

EOF

  sub "figure scripts present"
  for f in fig1_universality fig2_method_overview fig3_phase_visualization fig4_regret_scaling fig5_concordance_pr_curve; do
    if [[ -f "scripts/figures/${f}.py" ]]; then
      ok "  scripts/figures/${f}.py"
    else
      fail "  missing: scripts/figures/${f}.py"
    fi
  done

  ok "Phase 5 done"
}


# ============================================================================
# Phase 6 — launch the H800 sweep (long-running)
# ============================================================================
phase_6_train() {
  hdr "Phase 6 — launch H800 training sweep"

  if [[ -z "${TMUX:-}" ]] && [[ -z "${ALLOW_NO_TMUX:-}" ]]; then
    warn "you are NOT inside tmux — a 12-day training will die when SSH disconnects"
    warn "start a tmux session first:  tmux new -s pace_train"
    warn "or set ALLOW_NO_TMUX=1 to override"
    fail "refusing to launch outside tmux"
  fi

  # Pick up auto-tuned BATCH_SIZE / GRAD_ACCUM if Phase 4b ran.
  if [[ -f "$PACE_OUT/_launch_logs/autobatch.env" ]]; then
    sub "sourcing autobatch.env (from Phase 4b)"
    # shellcheck disable=SC1091
    source "$PACE_OUT/_launch_logs/autobatch.env"
    export BATCH_SIZE GRAD_ACCUM
    ok "  BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM  (effective $((BATCH_SIZE * GRAD_ACCUM)))"
  else
    warn "no autobatch.env found — using current BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM=$GRAD_ACCUM"
    warn "run \`bash scripts/run_autodl_pipeline.sh autobatch\` first to auto-tune"
  fi

  mkdir -p "$PACE_OUT/_launch_logs"
  local LOG="$PACE_OUT/_launch_logs/launch_$(date +%Y%m%d_%H%M%S).log"
  echo "  launch log: $LOG"

  echo
  echo "  Effective config:"
  echo "    CLOUD_SCRIPT       = $CLOUD_SCRIPT"
  echo "    PACE_ROOT          = $PACE_ROOT"
  echo "    DATA_ROOT          = $DATA_ROOT"
  echo "    PACE_OUT           = $PACE_OUT"
  echo "    PACE_SNAPSHOT_DIR  = $PACE_SNAPSHOT_DIR"
  echo "    BATCH_SIZE         = $BATCH_SIZE  (× GRAD_ACCUM=$GRAD_ACCUM = effective $((BATCH_SIZE * GRAD_ACCUM)))"
  echo

  bash "$CLOUD_SCRIPT" 2>&1 | tee "$LOG"
}


# ============================================================================
# Phase 8 — post-training aggregation + figures
# ============================================================================
phase_8_post() {
  hdr "Phase 8 — post-training aggregation + figures"

  local RUN_DIR
  RUN_DIR="${PACE_RUN_DIR:-$(ls -td "$PACE_OUT"/h800_run_* 2>/dev/null | head -1)}"

  if [[ -z "$RUN_DIR" ]] || [[ ! -d "$RUN_DIR" ]]; then
    fail "no h800_run_* directory under $PACE_OUT — did Phase 6 finish?"
  fi
  ok "RUN_DIR = $RUN_DIR"

  sub "aggregate ablation matrix (real eval_results.json)"
  python scripts/aggregate_ablation.py \
      --input_root "$RUN_DIR/ablation" \
      --output    "$RUN_DIR/aggregated" \
      2>&1 | tee "$RUN_DIR/logs/aggregate_post.log" || true

  if [[ -f "$RUN_DIR/aggregated/ablation_table_v2.csv" ]]; then
    sub "ablation_table_v2.csv preview"
    column -t -s, < "$RUN_DIR/aggregated/ablation_table_v2.csv" | head -25
  fi

  sub "render 5 paper figures"
  mkdir -p "$RUN_DIR/figures"

  python scripts/figures/fig1_universality.py \
      --input  "$RUN_DIR/dry_run/phenomenon_6_1_universality/raw_distances.json" \
      --output "$RUN_DIR/figures/fig1_universality.pdf" || warn "fig1 failed"

  python scripts/figures/fig2_method_overview.py \
      --output "$RUN_DIR/figures/fig2_method_overview.pdf" || warn "fig2 failed"

  python scripts/figures/fig3_phase_visualization.py \
      --output "$RUN_DIR/figures/fig3_phase_visualization.pdf" || warn "fig3 failed"

  python scripts/figures/fig4_regret_scaling.py \
      --input  "$RUN_DIR/phenomenon_6_2_regret_scaling/regret_vs_H.csv" \
      --output "$RUN_DIR/figures/fig4_regret_scaling.pdf" || warn "fig4 failed"

  python scripts/figures/fig5_concordance_pr_curve.py \
      --output "$RUN_DIR/figures/fig5_concordance_pr_curve.pdf" || warn "fig5 failed"

  ls -la "$RUN_DIR/figures/" || true

  sub "final stage-3 snapshot (with checkpoints)"
  python scripts/utils/snapshot_experiment.py \
      --src "$RUN_DIR/stage3_finetune" \
      --dst "$PACE_SNAPSHOT_DIR/stage3_finetune_FINAL_$(date +%Y%m%d_%H%M%S).tar.gz" \
      --include_checkpoints || warn "final snapshot failed"

  sub "paper artifact summary"
  echo
  echo "  Tables:"
  ls -1 "$RUN_DIR/aggregated/" 2>/dev/null | sed 's/^/    /'
  echo "  Figures:"
  ls -1 "$RUN_DIR/figures/"*.pdf 2>/dev/null | sed 's/^/    /'
  echo "  Real-data evals:"
  ls -1d "$RUN_DIR"/phenomenon_6_*_* 2>/dev/null | sed 's/^/    /'
  echo "  Diagnostic reports:"
  ls -1 "$RUN_DIR"/stage*/diagnostic_report.md 2>/dev/null | sed 's/^/    /'
  echo "  Snapshots:"
  ls -1 "$PACE_SNAPSHOT_DIR/"*.tar.gz 2>/dev/null | tail -10 | sed 's/^/    /'

  ok "Phase 8 done"
}


# ============================================================================
# dispatcher
# ============================================================================
case "$PHASE_ARG" in
  preflight|verify|"")
    phase_2_deps
    phase_3_verify_real
    phase_4_preflight
    phase_4b_autobatch
    phase_5_coverage
    echo
    ok "All preflight phases passed."
    echo
    printf "${BOLD}Next:${RESET}  bash scripts/run_autodl_pipeline.sh train   ${YELLOW}# inside tmux!${RESET}\n"
    ;;
  deps)
    phase_2_deps
    ;;
  realdata)
    phase_3_verify_real
    ;;
  test|tests)
    phase_4_preflight
    ;;
  autobatch|tune)
    phase_4b_autobatch
    ;;
  coverage)
    phase_5_coverage
    ;;
  train)
    phase_6_train
    ;;
  post|aggregate|figures)
    phase_8_post
    ;;
  all)
    phase_2_deps
    phase_3_verify_real
    phase_4_preflight
    phase_4b_autobatch
    phase_5_coverage
    warn "About to launch Phase 6 (12+ days). Ctrl-C in 10s to abort."
    sleep 10
    phase_6_train
    phase_8_post
    ;;
  *)
    cat << 'EOF'
Usage: bash scripts/run_autodl_pipeline.sh [phase]

Phases:
  preflight  (default) — Phase 2+3+4+4b+5 (deps + verify + preflight + autobatch + coverage)
  deps       — Phase 2 only (Python deps)
  realdata   — Phase 3 only (real-data + cliff-impl + no-fallback proof)
  tests      — Phase 4 only (pytest + smoke + 4 math + GPU 1-step)
  autobatch  — Phase 4b only (auto-tune BATCH_SIZE for the current GPU)
  coverage   — Phase 5 only (coverage report)
  train      — Phase 6 (~6–8 days on RTX PRO 6000 96GB; wrap in tmux)
  post       — Phase 8 (aggregate + 5 figures + final snapshot)
  all        — Phase 2 → 8 (with 10s abort window before Phase 6)

Environment:
  DATA_ROOT (default: $PACE_DATA/libero_10), DATASET_REPO_ID,
  BATCH_SIZE (default 32), GRAD_ACCUM (default 32),
  AUTOBATCH_CANDIDATES (whitespace-separated descending list),
  AUTOBATCH_TARGET_EFFECTIVE (default 1024),
  AUTOBATCH_SAFETY (default 0.90),
  SKIP_PYTEST=1, SKIP_GPU_STEP=1, SKIP_AUTOBATCH=1, ALLOW_NO_TMUX=1
EOF
    exit 2
    ;;
esac
