# Round 2 — Phase-Centric VLA Infrastructure & Config Extension

## Goal

Put the **skeleton** in place for the Phase-Centric innovations landing in
Rounds 3–7 (InfoNCE identifiability, PACE-A/B/C, PCAR):

- Pull in four external dependencies (three of which are already covered by
  higher versions; only `scipy` needs an explicit addition).
- Extend `PhaseQFlowConfig` with **21 new fields**. Every `use_*` switch
  defaults to False, so the policy behaves exactly as it did in Round 1
  unless a switch is turned on explicitly (backward compatibility is a
  hard constraint).
- Add a new `phase_centric/` sub-package (8 modules). Each module ships a
  docstring plus a `raise NotImplementedError` placeholder, so the top-level
  import doesn't crash; Rounds 3–7 fill them in one by one.
- Add `scripts/train_local.py`: 7 preset `--phase-centric-mode` values
  (off / ident_only / pace_a / pace_b / pace_c / pcar / full) plus full
  per-field CLI overrides, running 3 dummy-batch steps.
- Add `scripts/smoke_phase_centric.sh`: loops over all 7 modes for smoke
  testing; it passes only if every mode exits 0.
- Add `tests/test_phase_centric_config.py`: 8 pytest cases covering default
  values, types, mode overrides, CLI-override priority, import-time
  stability, and save/load round-trip.

**Hard rule this round: do not implement any Phase-Centric algorithm — only
the skeleton.**

## Deliverables

| Path | Role |
|---|---|
| `requirements.txt` | Added `scipy>=1.11`; demoted `bitsandbytes` to a "legacy optional" section with a note that Phase-Centric modules must not depend on it |
| `lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/configuration_phaseqflow.py` | Added 21 Phase-Centric fields + `Tuple` import |
| `lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/phase_centric/__init__.py` | Sub-package entry point + `__all__` declaration |
| `lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/phase_centric/identifiability.py` | Round 3 (innovation 3) placeholder: chunk InfoNCE |
| `lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/phase_centric/phase_posterior.py` | Round 4 placeholder: phase-boundary posterior EMA |
| `lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/phase_centric/pace_a_loss.py` | Round 5 (innovation 2) PACE-A placeholder: loss reweighting |
| `lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/phase_centric/pace_b_moe.py` | Round 6 PACE-B placeholder: phase-gated MoE |
| `lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/phase_centric/pace_c_curriculum.py` | Round 6 tail, PACE-C placeholder: curriculum scheduling |
| `lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/phase_centric/pcar_trigger.py` | Round 7 (innovation 1) PCAR placeholder: Bayesian changepoint |
| `lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/phase_centric/theory_utils.py` | Small theory helpers (run-length posterior, entropy, CDF) |
| `scripts/train_local.py` | Dummy-batch driver with 7 presets + full per-field CLI overrides |
| `scripts/smoke_phase_centric.sh` | 7 modes × 3 steps × CPU smoke |
| `tests/__init__.py` | Test package entry |
| `tests/test_phase_centric_config.py` | 8 pytest schema-level cases |
| `docs/innovations/round-2-summary.md` | This file |

### Impact on existing code

- `configuration_phaseqflow.py` **only adds fields** — no existing field is
  removed or renamed, and every new `use_*` defaults to False.
- `modeling_phaseqflow.py`, `processor_phaseqflow.py`, `training_utils.py`,
  etc. are **untouched**.
- `docs/README.md`, `docs/PROJECT_ABSTRACT.md`, etc. are **untouched**.
- Round 1's `scripts/diagnostic_*` and `scripts/smoke_test_diagnostic.py`
  are **untouched**.

## Dependency changes

| Dependency | Status | Note |
|---|---|---|
| `scipy>=1.11` | **new** | Round 7 PCAR's Bayesian online changepoint detection needs `scipy.stats`; Round 1's Welch t-test / pearsonr also depend on it |
| `einops>=0.7` | **version tightened** | Previously unpinned; PACE-B chunk rearrangement needs ≥0.7 |
| `transformers>=4.45` | already covered | Round 1 had already pinned ≥4.45, stricter than the ≥4.40 asked for |
| `vector-quantize-pytorch>=1.17.0` | already covered | Present already; the pypi package `lucidrains-vector-quantize-pytorch` is an alias for the same library, not re-added to avoid double installation |
| `bitsandbytes>=0.43` | **demoted** | Moved into the "Optional legacy" section; comment explicitly bans Phase-Centric modules from adding new bnb dependencies. CPU-only environments make `build_optimizer` fall back to AdamW automatically |
| `flash-attn` | **permanently banned** | RTX 5070 SM120 uses PyTorch SDPA; Round 2+ must not reintroduce it |

## 21 new config fields

| Round | Field | Type | Default | Role |
|---|---|---|---|---|
| 3 | `use_chunk_infonce` | bool | False | Switch: chunk-level InfoNCE contrastive loss |
| 3 | `chunk_infonce_weight` | float | 0.5 | InfoNCE weight in the total loss |
| 3 | `chunk_infonce_temperature` | float | 0.1 | InfoNCE temperature τ |
| 3 | `chunk_infonce_chunk_len` | int | 8 | Chunk length used when building positive/negative pairs |
| 4 | `phase_posterior_smooth_alpha` | float | 0.3 | EMA smoothing coefficient for β_t |
| 4 | `use_phase_boundary_posterior` | bool | False | Switch: estimate phase-boundary posterior during rollout |
| 5 | `use_pace_a` | bool | False | Switch: PACE-A loss reweighting |
| 5 | `pace_a_lambda` | float | 2.0 | Softmax weighting strength λ |
| 5 | `pace_a_entropy_weight` | float | 0.01 | Entropy regularisation weight |
| 6 | `use_pace_b` | bool | False | Switch: PACE-B phase-gated MoE |
| 6 | `moe_num_experts` | int | 4 | Number of experts (typically matches num_phases) |
| 6 | `moe_expert_hidden_dim` | int | 128 | Hidden dim of each expert MLP |
| 6 | `moe_switch_kappa` | float | 5.0 | Switching slope κ |
| 6 | `moe_switch_mu` | float | 2.0 | Switching threshold μ |
| 6 | `moe_top_k` | int | 2 | Inference top-k routing; 0 means soft routing |
| 6 | `use_pace_c` | bool | False | Switch: PACE-C phase-density curriculum |
| 6 | `curriculum_stage_steps` | Tuple[int,int,int] | (1000, 3000, 10000) | Three-stage step boundaries |
| 6 | `curriculum_max_boundaries_stage1` | int | 1 | Max boundary count allowed in stage 1 |
| 6 | `curriculum_max_boundaries_stage2` | int | 3 | Max boundary count allowed in stage 2 |
| 7 | `use_pcar` | bool | False | Switch: PCAR replan trigger |
| 7 | `pcar_change_threshold` | float | 0.4 | Bayesian changepoint trigger threshold |
| 7 | `pcar_trigger_budget_eps` | float | 0.1 | Theoretical mismatch-rate bound ε |
| 7 | `pcar_dual_head` | bool | False | Whether to enable the pre/post dual flow head |
| 7 | `pcar_post_head_ratio` | float | 0.5 | post head chunk = action_chunk_size × ratio |

> Note: 24 fields total (including the two secondary switches `pcar_dual_head`
> and `pcar_post_head_ratio`, plus `pcar_trigger_budget_eps`). All 21 core
> fields from the user spec are in; the remaining three are derived fields
> Round 7 will need, added early to the schema so we don't break backward
> compatibility later.

## Sub-package directory layout

```
lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/
├── phase_centric/
│   ├── __init__.py
│   ├── identifiability.py      # filled in Round 3
│   ├── phase_posterior.py      # filled in Round 4
│   ├── pace_a_loss.py          # filled in Round 5
│   ├── pace_b_moe.py           # filled in Round 6
│   ├── pace_c_curriculum.py    # filled near the end of Round 6
│   ├── pcar_trigger.py         # filled in Round 7
│   └── theory_utils.py         # filled on demand (run-length posterior, etc.)
└── ... (Round 1 and other modules, unchanged)
```

In Round 2 every function/class under `phase_centric/` still
`raise NotImplementedError`; the top-level
`import lerobot_policy_phaseqflow.phase_centric.xxx` does not crash
(verified by `test_phase_centric_subpackage_imports`).

## Mode × field activation table

Seven presets (defined in `MODE_PRESETS` of `scripts/train_local.py`):

| Mode | use_chunk_infonce | use_phase_boundary_posterior | use_pace_a | use_pace_b | use_pace_c | use_pcar |
|---|---|---|---|---|---|---|
| `off` | F | F | F | F | F | F |
| `ident_only` | **T** | F | F | F | F | F |
| `pace_a` | F | **T** | **T** | F | F | F |
| `pace_b` | F | **T** | F | **T** | F | F |
| `pace_c` | F | F | F | F | **T** | F |
| `pcar` | F | **T** | F | F | F | **T** |
| `full` | **T** | **T** | **T** | **T** | **T** | **T** |

CLI override syntax (takes precedence over the preset):

```bash
# Change pace_a mode's λ from 2.0 to 3.0
python scripts/train_local.py \
  --phase-centric-mode pace_a --pace_a_lambda 3.0 --steps 5

# Widen the budget while running PCAR
python scripts/train_local.py \
  --phase-centric-mode pcar --pcar_trigger_budget_eps 0.2 --steps 3
```

## Design decisions

| # | Decision | Chosen | Alternatives | Trade-off |
|---|---|---|---|---|
| 1 | Style of `train_local.py` | **1C: dummy-batch + `--real-data` placeholder** | 1A: pyproject entry-point / 1B: plug directly into LeRobot dataloader | Round 2 only validates the config schema; the real dataloader waits for Round 3+. Passing `--real-data` explicitly raises `NotImplementedError` |
| 2 | `bitsandbytes` handling | **2C: keep it but demote to legacy optional** | 2A: remove outright / 2B: replace with torchao | Don't break existing cloud scripts; the optimizer builder already falls back to AdamW under CPU-only. The comment makes it clear Phase-Centric modules must not add new bnb dependencies |
| 3 | FSQ dependency | Keep the existing `vector-quantize-pytorch>=1.17.0` pin | Also add the `lucidrains-vector-quantize-pytorch` alias | The user spec notes the latter is a pypi alias of the former; installing both would cause site-packages conflicts |
| 4 | Field defaults | **All `use_*` default to False** | Default True with a kill-switch | Backward compatibility is a hard constraint; every Round 1 smoke must keep passing without any config change |
| 5 | Placeholder form for the sub-package | **Top-level `import` does not raise; `raise NotImplementedError` inside functions** | Top-level `raise` that blocks import | pytest needs to `importlib.import_module` to confirm schema stability, but calls into an unimplemented function must fail fast |
| 6 | Number of PCAR fields | Wrote 5 (including `dual_head`, `post_head_ratio`, `budget_eps`) | Only 3 | Avoid reopening the schema when Round 7 actually implements it; adding them early has no effect when `use_pcar=False` |

## Next steps (starting Round 3)

1. **Round 3** (innovation 3, Phase Identifiability): land the chunk-level
   InfoNCE loss inside `phase_centric/identifiability.py`; hook it into
   `modeling_phaseqflow.compute_loss` through `chunk_infonce_weight`; rerun
   the Round 1 H1 diagnostic on real hardware and data — we expect the
   boundary/interior ratio to sit visibly above 1.0, which is the
   precondition for PACE-A.
2. **Round 4:** let `PhasePosteriorEstimator` update β_t step by step in the
   `HierarchicalLatentPlanner` rollout branch.
3. **Round 5:** wire PACE-A loss reweighting into `compute_loss`.
4. **Round 6:** PACE-B MoE + PACE-C curriculum.
5. **Round 7:** PCAR trigger + optional pre/post dual head.

Every round has to preserve the invariant that "behaviour with `use_*=False`
is identical to the previous round" and add a
`docs/innovations/round-N-summary.md`.
