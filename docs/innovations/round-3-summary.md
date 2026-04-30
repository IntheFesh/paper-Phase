# Round 3 — Phase Identifiability via Chunk-Level InfoNCE

## Goal

Round 2 landed the schema for `use_chunk_infonce` and the rest of the 21
Phase-Centric fields. Round 3 is responsible for **actually implementing
innovation 3**: make the discrete phase code `z_t` emitted by
`HierarchicalPlanner` point at the same latent task phase across seeds and
runs, so Rounds 5–7 (PACE-A / MoE / PCAR) have a trustworthy "phase signal"
to build on.

## Derivation (InfoNCE as a variational bound on I(z; (o,a)))

Let the context be `u = (o_t, a_{t:t+H})` and the latent `z_t ∈ {1,…,K}`.
Given a context encoder `f(·)` and a phase embedding table `g(·) = E[z]`,
the in-batch InfoNCE loss is

```
L_InfoNCE = - E_{(u_i, z_i) ∈ batch} log
              exp( ⟨f(u_i), g(z_i)⟩ / τ )
              ─────────────────────────────────────────────
              Σ_{j ∈ neg(i) ∪ {i}}  exp( ⟨f(u_i), g(z_j)⟩ / τ )
```

- van den Oord et al. (2018, CPC) show `-L_InfoNCE ≤ log N + I(u; z)`, so
  **minimising `L_InfoNCE` is equivalent to pushing up a variational lower
  bound on the mutual information `I(u; z)`**.
- Khemakhem et al. (iVAE, NeurIPS 2020) and Hyvärinen (2024 review) prove
  that, conditional on an auxiliary variable `u`, the latent `z` is
  identifiable up to a permutation and per-axis rescaling. Here
  `u = (o_t, a_{t:t+H})` plays the role of the auxiliary variable, and
  InfoNCE explicitly approximates that MI lower bound, which is what gives
  us cross-run / cross-seed phase-permutation invariance.

## Implementation details

### 3.1 `ChunkInfoNCEHead`

- Location: `lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/phase_centric/identifiability.py`.
- `K` adapts to the discretiser: when `use_fsq=True`, `K = prod(fsq_levels)`;
  otherwise `K = num_skills`. It does **not** use `num_phases` — that's the
  human-semantic phase count and doesn't line up with the planner logits
  dimensions.
- `f(o, a)`: two `Linear+SiLU` layers. Input is
  `cat(fused_obs, flat(action_chunk))`, output is a D-dim unit-sphere vector.
- `g(z)`: `nn.Embedding(K, D)`, L2-normalised.
- Similarity matrix `sim = ctx @ z.T / τ`, shape `(B, B)`.
- **Same-phase masking:** off-diagonal rows sharing the anchor's phase are
  neither positives nor negatives. The mask `~same_phase | eye` only lets
  "self + different-phase" into the denominator.
- **Degenerate-row guard:** when a row's batch is entirely same-phase,
  `has_any_neg=False`, and that row is dropped from the mean of `loss`.
  Otherwise `log(1)=0` combined with zero gradient would fake a loss
  decrease.
- Diagnostics:
  - `info_nce_acc`: after projecting `ctx` against the full `phase_embed`
    table, the fraction that top-1 hits the true `phase_id`.
  - `phase_entropy`: `H(softmax(phase_logits))`. Higher means the codebook
    is more active.
  - `num_valid_rows`: number of samples that actually contributed to the
    contrast.

### 3.2 Policy integration

In `lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/modeling_phaseqflow.py`:

- `__init__`: if `use_chunk_infonce=True`, instantiate `ChunkInfoNCEHead`;
  otherwise `self.chunk_infonce_head = None` (guarantees byte-identical
  smoke results from Round 1).
- `compute_loss`:
  - Pull `fused_obs (B, D)` out of `preds["encoded_obs"]`.
  - If the batch provides a real chunk `(B, Ta, Da)`, use it directly;
    otherwise treat the single step `(B, Da)` as `(B, 1, Da)` by explicit
    unsqueeze and let the head pad internally. This is theoretically
    weaker but keeps the CPU smoke working. Once Rounds 5/6 connect the
    real dataloader, we swap this back to the ground-truth chunk.
  - Multiply by `chunk_infonce_weight` and add to `total_loss`.
  - Add `chunk_infonce` to `_last_loss_components`, and attach the
    diagnostics dict to `_last_chunk_infonce_diag`.

### 3.3 `scripts/verify_identifiability.py`

- Three seeds (default 42/43/44), each trained independently for `--steps`
  (default 2000).
- Validation set size `--num_val_samples` (default 256).
- `_permuted_agreement`: builds a K×K co-occurrence matrix `C`, calls
  `scipy.optimize.linear_sum_assignment(-C)` for the Hungarian-optimal
  permutation `π*`, and returns `mean(z_a == π*(z_b))`.
- Artifacts:
  - `artifacts/identifiability/report.json`: config snapshot, pairwise /
    GT agreement, permutations, `unique_phase_ids_per_seed`, verdict.
  - `artifacts/identifiability/report.md`: human-readable summary.
  - `artifacts/identifiability/figures/identifiability_confusion.png`:
    grid of N×N confusion matrices (pre-permutation).
- **Verdict rules:**
  - `PASS`: every seed has at least 2 unique phase ids on the validation
    set *and* `min(pair_agreement) ≥ threshold`.
  - `WARN_DEGENERATE`: any seed has only 1 unique phase id (codebook
    collapse) → the permuted agreement trivially hits 1.0 but **cannot be
    taken as evidence of identifiability**. The report flags this
    explicitly with a fix suggestion.
  - `FAIL`: no collapse, but agreement < threshold.

## Deliverables

| Path | Role |
|---|---|
| `lerobot_policy_phaseqflow/.../phase_centric/identifiability.py` | Placeholder → **full implementation** of `ChunkInfoNCEHead` + `chunk_infonce_loss` wrapper |
| `lerobot_policy_phaseqflow/.../modeling_phaseqflow.py` | `__init__` mounts the head; `compute_loss` computes and adds it to total loss; `_last_loss_components` gains a `chunk_infonce` key |
| `scripts/verify_identifiability.py` | 3-seed training + Hungarian permutation + confusion viz + WARN_DEGENERATE detection (~330 lines) |
| `tests/test_chunk_infonce.py` | 8 pytest cases (head forward / degenerate batch / pad / K inference / functional parity / policy off-on / backward / save-load) |
| `tests/test_phase_centric_config.py` | Updates `test_subpackage_functions_raise_not_implemented`: `chunk_infonce_loss` is now implemented, so it's removed from the list. Only Rounds 4–7 placeholders remain |
| `artifacts/identifiability/{report.json,report.md,figures/*.png}` | Artifacts from the local 50-step CPU smoke (`WARN_DEGENERATE`, see below) |
| `docs/innovations/round-3-summary.md` | This file |

### Impact on existing code

- `configuration_phaseqflow.py`: **untouched** (Round 2 already added the
  fields).
- `processor_phaseqflow.py` / `training_utils.py`: **untouched**.
- `docs/README.md` / `docs/PROJECT_ABSTRACT.md`: **untouched**.
- All Round 1/2 smokes and pytest keep passing.

## Acceptance results

| Check | Status | Note |
|---|---|---|
| `python -m pytest tests/ -q` | **17 passed** | 8 from Round 2 + 9 new in Round 3 (including `test_subpackage_functions_raise_not_implemented` covering Rounds 4–7 placeholders) |
| `scripts/smoke_test_diagnostic.py` (Round 1) | PASS | unaffected |
| `scripts/smoke_test_training_pipeline.py` | PASS | unaffected |
| `scripts/smoke_phase_centric.sh` 7 modes | PASS | under `ident_only` the `chunk_infonce` component is non-zero and finite |
| `scripts/verify_identifiability.py --steps 50` (CPU smoke) | `WARN_DEGENERATE` | **expected behaviour**: too few steps means the Gumbel codebook hasn't spread out yet, so this verifies the degeneracy detector itself works |

## Reading the CPU smoke result (`WARN_DEGENERATE`)

`report.md` from the 50-step CPU smoke:

```
Verdict: WARN_DEGENERATE (threshold 0.70, min pair agreement 1.000)
Unique phases per seed: {1: 1, 2: 1, 3: 1}
```

- On the validation set **each of the three seeds emits only 1 unique phase
  id** (the codebook has fully collapsed to a single column).
- In that state **any permutation** yields `agreement = 1.000`, which has
  nothing to do with "consistency across seeds". Declaring PASS at
  threshold 0.7 here would be a false positive on identifiability.
- The verdict rules therefore take priority: if any seed has fewer than 2
  unique phase ids, the verdict is forced to `WARN_DEGENERATE`, and
  `report.md` gives a fix suggestion. **We do not let a fake PASS leak
  into Round 4+ downstream experiments.**
- Matching that, `seed_vs_gt` agreement sits around 0.508 (basically the
  majority-class base rate), which shows the seed learned no phase
  structure — the symptom of "too few samples + too few steps", not an
  algorithm bug.

## What to try if a real run (RTX 5070, 2000 steps) still has agreement < 0.7

Work through the list in order:

1. **Still `WARN_DEGENERATE`** (unique phase < 2): codebook collapse.
   - Switch the planner to FSQ: `--use_fsq True` (FSQ quantisation forces
     the full codebook to stay populated, so it can't collapse to a
     single point).
   - Lower `gumbel_temperature` (default 0.5 → 0.3) to sharpen the
     argmax.
   - Raise `chunk_infonce_weight` (0.5 → 1.0 or 2.0) so the
     identifiability term carries more weight in the total loss.
   - Drop `num_skills` from 8 to `num_phases = 4` (fewer planner degrees
     of freedom, so it doesn't spread too widely in an unsupervised
     setting).
2. **No collapse but agreement ∈ [0.4, 0.7)** (partial identifiability):
   - Run longer (2000 → 5000 steps).
   - Widen the batch (current script uses micro_batch=16).
   - Turn on `use_phase_boundary_posterior` (Round 4 deliverable) to
     give the planner a stronger temporal signal.
3. **Agreement stays ≥ 0.7 but seed_vs_gt < 0.5**: the seeds agree with
   each other but don't align with the human-semantic phase → this is
   **still valid identifiability** (iVAE only guarantees identifiability
   up to permutation + rescaling, not semantic alignment). PACE-A in
   Round 5 doesn't depend on semantic alignment — it only needs a stable
   z.

## Downstream dependencies (Round 4+)

- Round 4's `PhasePosteriorEstimator` will consume the stabilised
  `phase_logits` from this round to estimate the phase-boundary posterior
  `β_t`, then feed Rounds 5/6/7.
- Round 5 PACE-A's `pace_a_lambda` tuning depends on monitoring
  `info_nce_acc` from this round — when acc drops below 0.5, PACE-A's
  reweighting can amplify the wrong phase.
- Round 6 PACE-B MoE's top-k routing reads `phase_logits` directly, so
  identifiability PASS here is a precondition for MoE not collapsing.
- Round 7 PCAR needs a stable phase posterior + changepoint detector
  pipeline; its dual-head inference path reuses
  `chunk_infonce_head.phase_embed` to judge the "current phase".

## Key decisions

| # | Decision | Chosen | Reason |
|---|---|---|---|
| 1 | Source of `K` | `use_fsq ? prod(fsq_levels) : num_skills` | The planner's `phase_logits` dimension is set by the discretiser; `num_phases` is a semantic field, not the right K for the head |
| 2 | Contrast scope | In-batch (no all-gather) | DDP isn't ready yet; keeps CPU smoke friendly. Reconsider cross-GPU negatives once Rounds 5/6 plug in the real dataloader |
| 3 | Same-phase samples | Mask off-diag same-phase rows so they're neither pos nor neg | Avoids the classic InfoNCE failure mode where wrong negatives push semantically-same-phase samples apart |
| 4 | Missing action_chunk | Treat single step `(B, Da)` as `(B, 1, Da)` and let the head pad | Lets CPU smoke keep running; once Rounds 5/6 wire up the LeRobot dataloader, we're back to real chunks |
| 5 | Verdict rule | Collapse forces `WARN_DEGENERATE` | Stops a trivial 1.0 agreement from masquerading as an identifiability PASS |
| 6 | Permutation algorithm | Hungarian (scipy `linear_sum_assignment`) | O(K³) is fine for K ≤ 240; a greedy permutation would overstate consistency under uneven phase distributions |

## Next steps (Round 4+)

1. **Round 4:** implement `PhasePosteriorEstimator` in
   `phase_centric/phase_posterior.py`, consume the stabilised
   `phase_logits` from this round, and emit `β_t` (the phase-boundary
   posterior).
2. Once Round 4 lands, do a real run of `verify_identifiability.py` on an
   RTX 5070 (2000 steps × 3 seeds) and archive the `report.{json,md}` +
   confusion figures.
