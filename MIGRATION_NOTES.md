# Migration Notes — PACE v2 / Predictability-Cliff Terminology

This document tracks open engineering questions that require a human author decision
before implementation can proceed. All entries in §1 "Pending Human Decisions" are
blocking at least one downstream component.

---

## Pending Human Decisions

### [PHD-1] I_hat_2 — Action-Variance Cliff Estimator

**Formula (locked):**
$$\hat I^{(2)}(t) \propto -\sigma_t^2 = -\frac{1}{N}\sum_{i=1}^{N}\|a_t^{(i)} - \bar a_t\|^2$$

**Blocking question:** How and when should the N action samples be drawn?

**Option A — BID sampling path:**  
When `use_bid_sampling=True`, `BIDSampler` already draws N=5 candidate chunks in
`select_action`. The variance across the N candidates could be used as I_hat_2.
However: `select_action` runs only at inference; `compute_loss` never calls it,
so I_hat_2 would be unavailable during training, making PACE-A and PACE-B unable
to use it as a weighting signal. Also, the BID candidates share the same `cond`
vector; they differ only in noise seed, which gives aleatoric uncertainty, not
epistemic uncertainty tied to phase boundaries.

**Option B — Fixed multi-noise-seed call inside `forward`:**  
Call `flow_action_head` (or `pace_b_flow_head`) with `training=False` and N
different noise seeds at each forward pass. Store all N predictions, compute the
mean and variance. Cost: N extra forward passes per step (N=5 → 5× slowdown).
Alternatively, a single forward in `training=True` mode already samples one noise
vector; repeating it N times with `torch.no_grad()` would work but needs
deciding whether I_hat_2 is computed in training mode, inference mode, or both.

**Option C — Ensemble-buffer variance:**  
If temporal ensembling is on (`use_temporal_ensembling=True`), the
`ACTTemporalEnsembler` holds a rolling buffer of past predictions. The variance
over this buffer is a proxy for action consistency, but it blends across time
steps rather than sampling variance at a single step.

**Decision needed:** Which of A / B / C is the intended semantic? Or a new option?

**Blocked component:** `cliff_estimators.compute_I_hat_2`, and transitively
`cliff_estimators.compute_concordance_C`.

---

### [PHD-2] I_hat_3 — Velocity-Difference Cliff Estimator

**Formula (locked):**
$$\hat I^{(3)}(t) \propto -\|v_\theta(x_\tau, \tau, c_t) - v_\theta(x_\tau, \tau, c_{t-1})\|_2^2$$

This requires three inputs that are not currently available together:

**Sub-question 2a — Anchor point `(x_τ, τ)`:**  
The velocity field `v_θ` must be evaluated at a *fixed* anchor point so that
the difference isolates the conditioning change rather than a trajectory position
change. Options:
- `x_τ = 0` (zero vector, fully noised), `τ = 0` — cheapest; no semantic meaning.
- `x_τ = randn(...)` sampled once per episode and fixed — reproducible but
  depends on the seed.
- `x_τ` = the previous predicted action chunk (i.e. `action_pred` from the prior
  step) — semantically meaningful but means the anchor drifts with the policy.
- `τ = 0.5` (midpoint of the flow trajectory) — a common choice in flow-matching
  papers for sensitivity analysis.

Decision: which `(x_τ, τ, d)` triplet to fix?

**Sub-question 2b — Storage of `c_{t-1}`:**  
`c_t` is the condition vector `cond = conditioner([fused_obs, phase_embed,
skill_latent])` computed inside `ShortcutFlowActionHead.forward`. At inference,
the previous step's condition vector `c_{t-1}` is not stored. Storing it requires
either:
- A `_prev_cond` buffer inside `ShortcutFlowActionHead` (parallel to
  `PhasePosteriorEstimator._running_p`), OR
- Passing `c_{t-1}` in from the policy's `select_action` loop.

Decision: where should `c_{t-1}` be buffered?

**Sub-question 2c — Expose `_velocity` as a public method:**  
`ShortcutFlowActionHead._velocity` is a private method. Computing I_hat_3 from
outside the head (e.g., from `PhaseQFlowPolicy.forward`) would require either:
- Renaming `_velocity` → `velocity` (a one-line change, backward-compatible if
  no external code calls the private name), OR
- Adding a new public `eval_velocity(x, t, d, cond)` that delegates to the
  private method.

Decision: rename or wrap?

**Blocked component:** `cliff_estimators.compute_I_hat_3`, and transitively
`cliff_estimators.compute_concordance_C`.

---

### [PHD-3] Concordance C_t — Rank-Based Fusion

**Formula (locked):**
$$C_t = \frac{1}{3}[\mathrm{rank}_W(\hat I^{(1)}(t)) + \mathrm{rank}_W(\hat I^{(2)}(t)) + \mathrm{rank}_W(\hat I^{(3)}(t))]$$

**Blocking dependency:** Requires PHD-1 and PHD-2 to be resolved first.

**Additional sub-question:** What is the rolling window size W?

The PCAR trigger uses `history_size=1000` for the β distribution. A reasonable
default for `rank_W` would be W=50 (matching PCAR's `warmup_min`) so the
concordance is meaningful as soon as PCAR exits its warmup phase. But this is a
hyperparameter that needs explicit confirmation.

**Blocked component:** `cliff_estimators.compute_concordance_C`.

---

## Completed Migrations

| Item | Status | PR / commit |
|------|--------|-------------|
| Add `I_hat_1 = -phase_beta` to predict_action output | ✅ Done | branch `claude/standardize-cliff-terminology-3Uuqa` |
| Create `phase_centric/cliff_estimators.py` cliff namespace module | ✅ Done | branch `claude/standardize-cliff-terminology-3Uuqa` |

---

## Terminology Lock Table

| Paper term | Code (internal / Round-4 name) | Cliff-namespace name |
|---|---|---|
| Predictability Cliff | (concept) | cliff |
| Boundary probability β_t | `phase_beta`, `beta_t` | — (internal only) |
| Cliff estimator I^(1) | `-phase_beta` | `I_hat_1` |
| Cliff estimator I^(2) | (pending) | `I_hat_2` |
| Cliff estimator I^(3) | (pending) | `I_hat_3` |
| Concordance C_t | (pending) | `concordance_C` |
| Phase posterior p̂_t | `phase_p_hat`, `p_hat` | — (internal only) |
| Bhattacharyya distance β_t | `_bhattacharyya_beta` | — (internal only) |
