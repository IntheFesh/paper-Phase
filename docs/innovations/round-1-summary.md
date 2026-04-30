# Round 1 — Phase-Centric VLA Diagnostic

## Goal

Before writing any Phase-Centric innovation code, check two core hypotheses
against the existing repo and data:

- **H1 (Phase-Boundary Loss Gap):** a trained PhaseQFlow++ has visibly higher
  flow-matching loss on timesteps near a phase boundary than inside a phase.
  Target: `boundary_loss / interior_loss ≥ 2.0` with Welch t-test `p < 0.01`.
- **H2 (Misalignment × Failure Correlation):** in rollouts, the per-episode
  average distance from each chunk-replan moment to the nearest phase boundary
  has Pearson correlation `r ≤ -0.5` (`p < 0.01`) with the success flag.

Only if both hypotheses **PASS** do we move on to Round 2 and start
implementing innovations. Otherwise we adjust the strategy or rethink the
project direction.

## Deliverables

| Path | Role |
|---|---|
| `scripts/diagnostic_phase_centric.py` | Main diagnostic entry (argparse CLI) |
| `scripts/diagnostic_utils/phase_proxies.py` | Three boundary proxies: gripper, velocity_change, planner_output |
| `scripts/diagnostic_utils/synthetic_env.py` | `SyntheticLongHorizonEnv`: 3-waypoint 2D navigation with ground-truth phase boundaries |
| `scripts/diagnostic_utils/synthetic_demos.py` | Synthetic demo generator + `try_load_real_demos` HuggingFace loader fallback |
| `scripts/diagnostic_utils/h1_loss.py` | Per-timestep FM loss extractor (no data aug, supports both Shortcut and Euler) |
| `scripts/diagnostic_utils/h2_rollout.py` | Episode rollout + misalignment computation |
| `scripts/diagnostic_utils/report.py` | JSON / Markdown / PNG report generation |
| `scripts/smoke_test_diagnostic.py` | CPU-only smoke test, runs end-to-end in under 60s |

### Impact on existing code

- **Zero modifications:** nothing under `lerobot_policy_phaseqflow/` was
  touched (`modeling_phaseqflow.py`, `configuration_phaseqflow.py`,
  `processor_phaseqflow.py`, etc.).
- The new scripts reproduce Shortcut FM loss along the training path by
  loading the existing config and driving `policy.flow_action_head` directly,
  so model code stays untouched.
- `docs/README.md`, `docs/PROJECT_ABSTRACT.md`, and `docs/OPERATIONS_GUIDE.md`
  were not modified.

## Design decisions

| # | Decision | Chosen | Alternatives | Rationale |
|---|---|---|---|---|
| 1 | H1 chunk supervision | **Real chunk** `action[t:t+Ta]` | tile single-step / parallel A+B | Matches the Shortcut training target; dropping the last ~15 steps of a demo changes the H1 ratio by <1% |
| 2 | H1 per-t inference | **Batched-over-t + 3× noise resampling** | single-sample / 5× resampling | Covers the whole demo in one forward pass; 3× is enough to push the t/d/noise variance of Shortcut FM loss into an acceptable range |
| 3 | Data fallback | **Real-first, synthetic requires an explicit `--allow_synthetic_demos` flag** | hard-fail / default to synthetic | Real results come first; offline can still smoke-test, but the synthetic path gets a prominent ` SYNTHETIC` banner at the top of the report |
| 4 | Environment fallback | **Try libero; on failure or missing package, fall back to `SyntheticLongHorizonEnv`** | hard-fail | `SyntheticLongHorizonEnv`'s gripper = `waypoint_idx % 2` naturally gives H2 a ground-truth phase-boundary signal |
| 5 | H2 replan semantics | **External chunk buffer**, call `predict_chunk` whenever `buffer == []` | reuse `select_action` internal state | Avoids the ambiguity BID/ACT introduces by replanning every step; matches the pseudocode in the user spec verbatim |
| 6 | Planner-proxy fallback | Fall back to gripper automatically when planner_output fails | hard-fail | Keeps `--phase_proxy planner_output` runnable on machines without a real checkpoint |

## Actual numbers in the current environment (pipeline validation only)

The repo currently has no real trained ckpt and no real LIBERO dataset or env
available. The placeholder numbers below come from running
**synthetic demos + random-init policy + synthetic env**:

| Metric | Value | Verdict | Note |
|---|---|---|---|
| H1 ratio | **1.02** | FAIL | FM loss of a random policy is basically white noise, so we don't expect a boundary gap |
| H1 p-value | 0.34 | — | |
| H2 Pearson r | **NaN** | FAIL | random policy's success rate on the synthetic env is 0/6; zero variance leaves pearsonr undefined |
| H2 success rate | 0.00 | — | |

**These numbers cannot back any research claim** — they just say the pipeline
runs end to end.

## Next steps (run on real hardware)

1. On a machine with LIBERO and a trained ckpt:
   ```bash
   python scripts/diagnostic_phase_centric.py \
     --policy_path outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model \
     --dataset_name HuggingFaceVLA/libero \
     --num_demos 200 --num_episodes 50 --device cuda
   ```
2. Read the executive summary in `artifacts/diagnostic/report.md`:
   - Both PASS → move on to Round 2.
   - At least one WARN → retry with `--phase_proxy velocity_change` or
     `planner_output`; optionally widen `--num_demos` / `--num_episodes`.
   - Any FAIL → stop or adjust the project direction.
3. The LIBERO env factory currently leaves a `raise NotImplementedError`
   placeholder; wire it up explicitly in `main()` of
   `scripts/diagnostic_phase_centric.py` against whichever
   `libero.libero.benchmark` version is installed locally.
