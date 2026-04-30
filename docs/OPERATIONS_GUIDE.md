# Operations Guide · PACE v2

This is a pure engineering handbook: the full path from environment setup
through training, evaluation, verification, paper artifacts, and export.
Script paths match the current repository layout; for architecture and
maths background see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 1 Environment setup

```bash
conda env create -f environment.yml
conda activate lerobot_env
pip install -r requirements.txt
pip install -e ./lerobot_policy_phaseqflow   # package name: lerobot-libero-phaseqflow-vla
```

Works with `venv` / `uv` too: install the dependencies first, then
install the policy package in editable mode.

## 2 Data inspection

```bash
python scripts/data/inspect_dataset.py \
  --dataset HuggingFaceVLA/smol-libero --n 5

python scripts/data/compute_episode_lengths.py \
  --dataset HuggingFaceVLA/smol-libero \
  --out artifacts/episode_lengths/smol_libero_lengths.json
```

Expected output: schema consistency check passes, and the per-episode
length histogram is written to JSON (used to pick the chunk length $T_a$
and maximum sequence length).

## 3 Local training

### 3.1 Single-GPU main entry

```bash
bash configs/local/phaseqflow_local.sh
```

### 3.2 Ablation matrix driver

```bash
# full 12 × N-seed matrix
bash scripts/training/run_ablation.sh

# custom subset
CONFIGS="baseline ident full" SEEDS="42 123" \
    bash scripts/training/run_ablation.sh
```

### 3.3 Single skeleton training run (debugging)

```bash
python scripts/training/train_dummy_batch.py --phase-centric-mode full --seed 42
```

## 4 Multi-GPU training

```bash
bash configs/cloud/phaseqflow_cloud_accelerate.sh
```

Before launching: confirm `CUDA_VISIBLE_DEVICES`, world size,
per-device batch size, gradient accumulation, and `output_dir` naming
(one run per directory).

## 5 Evaluation and latency benchmarks

```bash
# LIBERO rollout
bash scripts/evaluation/run_eval_libero.sh \
  outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model

# PushT smoke-test evaluation
bash scripts/evaluation/run_eval_pusht_sanity.sh \
  outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model

# Inference latency benchmark (requires a pretrained checkpoint)
python scripts/evaluation/benchmark_latency.py --help

# Offline latency benchmark without a checkpoint (useful when HF Hub is unreachable)
python scripts/evaluation/benchmark_latency_offline.py \
  --configs baseline ident_only pace_a pace_b pace_c pcar --iters 20
```

Monitor both policy quality (success / reward) and runtime metrics
(latency / throughput). The offline benchmark falls back to random
initialisation for SigLIP while DINOv2 still loads through
`torch.hub`; absolute numbers will shift with real weights, but the
trend is informative.

## 6 Maths assertion verification

| Goal | Command | Expected output |
| :-- | :-- | :-- |
| Phase identifiability (I1) | `python scripts/verification/verify_identifiability.py` | permuted-agreement $\ge 0.7$ (long GPU training is expected to exceed $0.8$; CPU placeholder training may return `WARN_DEGENERATE`) |
| Phase posterior $\beta_t$ alignment (I2) | `python scripts/verification/verify_phase_posterior.py` | verdict: PASS |
| PCAR budget alignment (I6) | `python scripts/verification/verify_pcar_budget.py` | $\lvert\text{diff}\rvert<0.005$ for every $\epsilon\in\{0.05,0.1,0.2,0.3\}$ |
| PACE-A boundary reduction (I3) | `python scripts/verification/sanity_pace_a.py` | `passes_20pct_boundary_reduction: True` |
| PACE-B gate contrast (I4) | `python scripts/verification/sanity_pace_b.py` | boundary gate L2 $>0.3$ with interior $<0.05$ |

Statistical definitions and threshold rationale are in
[`ARCHITECTURE.md`](ARCHITECTURE.md) §5.

## 7 End-to-end smoke tests

```bash
bash   scripts/smoke/smoke_phase_centric.sh          # 7-mode
bash   scripts/smoke/smoke_test_phaseqflow_e2e.sh    # minimal train+eval
python scripts/smoke/smoke_test_training_pipeline.py
python scripts/smoke/smoke_test_a2c2_correction.py
python scripts/smoke/smoke_test_iql_bid.py
python scripts/smoke/smoke_test_shortcut_flow.py
python scripts/smoke/smoke_test_hierarchical_planner_fsq.py
python scripts/smoke/smoke_test_diagnostic.py
```

Note: the E2E script automatically falls back to `DEVICE=cpu` when no
GPU is detected and uses very small defaults (`STEPS=2`,
`EVAL_EPISODES=1`) to finish in seconds.

## 8 Paper artifact pipeline

```bash
# 1) train the 7-config ablation ladder
bash scripts/training/run_ablation.sh

# 2) aggregate into CSV + LaTeX (IQM + 95% CI + Wilcoxon p + Cohen's d)
python scripts/aggregate_ablation.py \
    --input_root outputs/ablation_v2 \
    --output paper_figures/ablation_v2/

# 3) five publication-grade figures
python scripts/figures/fig1_universality.py \
    --input paper_figures/universality/raw_distances.json \
    --output paper_figures/fig1_universality.pdf
python scripts/figures/fig2_method_overview.py
python scripts/figures/fig3_phase_visualization.py --dry_run
python scripts/figures/fig4_regret_scaling.py
python scripts/figures/fig5_concordance_pr_curve.py

# 4) diagnostic tools
python scripts/diagnostics/diagnostic_utils/measure_boundary_error.py --dry_run
python scripts/diagnostics/diagnostic_utils/replan_alignment.py --dry_run
python scripts/diagnostics/diagnostic_utils/trigger_comparison.py --dry_run
python scripts/diagnostics/diagnostic_utils/measure_inference_cost.py --dry_run
```

Artifact locations:

- `paper_figures/ablation_v2/ablation_table_v2.csv` — per-method × benchmark IQM table
- `paper_figures/ablation_v2/ablation_table_v2.tex` — booktabs LaTeX table
- `paper_figures/fig1_universality.pdf` — cliff universality figure
- `paper_figures/fig2_method_overview.pdf` — method block diagram
- `paper_figures/fig3_phase_visualization.pdf` — phase + estimator + C_t visualization
- `paper_figures/fig4_regret_scaling.pdf` — regret vs chunk-length H scaling
- `paper_figures/fig5_concordance_pr_curve.pdf` — concordance PR curve

CPU sandbox artifacts carry a `placeholder_stats=true` field; rerunning
the same commands after a full GPU training + eval pipeline overwrites
them in place.

## 9 Exporting a checkpoint

```bash
python scripts/data/export_checkpoint.py \
  --src outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model \
  --dst exports/phaseqflow_smol_local
```

## 10 Diagnostic tools

```bash
python scripts/diagnostics/diagnostic_phase_centric.py --help
```

Shares the `diagnostic_utils/` helper package with
`scripts/smoke/smoke_test_diagnostic.py`.

## 11 Troubleshooting

- `pip show lerobot_policy_phaseqflow` — confirm the policy package is
  installed.
- Check that the dataset path and HuggingFace network are reachable.
- Before any large change, run `bash scripts/smoke/smoke_phase_centric.sh`.
- `pytest tests/ -v` expects **204 passed**.
- One run per `output_dir`, so experiments are easy to trace.
- Under CPU placeholder training, identifiability returning
  `WARN_DEGENERATE` is intentional (see
  [`ARCHITECTURE.md`](ARCHITECTURE.md) §5.2 threshold discussion).
- `lerobot-train --policy.type phaseqflow` reports `invalid choice`:
  make sure you've added
  `--policy.discover_packages_path lerobot_policy_phaseqflow`. LeRobot
  0.4+ uses the draccus plugin loader to import third-party packages
  before argparse parses, which triggers
  `@PreTrainedConfig.register_subclass("phaseqflow")`.

## 12 Architecture decisions

Three key engineering forks are called out here so source comments
don't duplicate the reasoning.

### 12.1 Bhattacharyya $\beta_t$ vs. hard argmax

The phase-boundary signal uses the Bhattacharyya first-order distance
$\beta_t = 1 - \sum_k \sqrt{\hat p_t(k)\,\hat p_{t-1}(k)}$ rather than
the hard indicator $\mathbb{1}[\arg\max_t \neq \arg\max_{t-1}]$.

Reasons:

1. **Usable gradient** — $\beta_t$ is continuously differentiable in
   the planner logits; $\arg\max$ isn't differentiable at the switch
   point and can't backprop into the planner.
2. **Jitter-robust** — small perturbations in the softmax tail don't
   jolt $\beta_t$; $\arg\max$ flips back and forth when adjacent
   logits are close.
3. **Geometric meaning** — $\beta_t = H^2$ (Hellinger squared), with
   $\tfrac12\beta_t \le d_{\text{TV}} \le \sqrt{\beta_t}$, so it
   straddles TV distance with a clean two-sided bound.
4. **Normalisation** — $\beta_t \in [0, 1]$ plays nicely with the
   downstream PACE-A weight $1 + \lambda\beta$ and PCAR's quantile
   threshold.

### 12.2 Budget quantile vs. static threshold

The PCAR trigger threshold $\tau^{\text{cp}}_n = \hat Q_n(1-\epsilon)$
is estimated online from the empirical β distribution rather than
fixed at $\tau=0.4$.

Reasons:

1. **Cross-task stability** — a static threshold swings the trigger
   rate wildly as the β distribution shifts with task difficulty
   (easy task $<1\%$, hard task $>30\%$); the quantile design keeps
   the rate locked to $\epsilon \pm O(1/\sqrt{n})$.
2. **DKW guarantee** — for $n=1000,\delta=0.05$ the deviation bound
   is $\sqrt{\log(2/\delta)/(2n)} \approx 0.043$; empirical runs on
   synthetic distributions stay $<0.005$.
3. **Configurability** — the user only has to set the budget
   $\epsilon$ (meaning: a rollout replans for roughly $\epsilon$ of
   its steps) instead of guessing at a $\tau$ that has no physical
   meaning.

Warm start: with $n < 50$ samples, we fall back to
`pcar_change_threshold=0.4` to avoid the unstable early quantile.

### 12.3 IQL Verifier vs. MLP confidence

`IQLChunkVerifier` uses an expectile regression on $V_\psi$ plus TD(0)
on $Q_\theta$ to get $\sigma(\beta_c(Q-V))$, rather than a single
`ChunkVerifierMLP` that regresses directly to the success label.

Reasons:

1. **No terminal reward required** — IQL treats action MSE or phase
   delta inside the imitation data as reward; it doesn't need a
   rollout-level success label.
2. **Advantage calibration** — $Q-V$ is an MC approximation of the
   advantage, and after sigmoid it naturally sits in $(0, 1)$. A
   direct MLP regression on $p(\text{success})$ tends to over-confide
   on small datasets.
3. **Target-network stability** — Polyak $\tau=0.005$ on
   $V_\psi^{\text{target}}$ keeps TD bootstrap from diverging.
4. **Optional fallback** — `config.verifier_type="mlp"` keeps the
   alternative path available for controlled comparisons.

## 13 Document index

- Architecture and theory spec: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Academic abstract: [`PROJECT_ABSTRACT.md`](PROJECT_ABSTRACT.md)
- Per-innovation index: [`innovations/INDEX.md`](innovations/INDEX.md)
