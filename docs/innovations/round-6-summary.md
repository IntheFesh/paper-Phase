# Round 6 — Back Half of Innovation 2: PACE-B + PACE-C

This round lands the back half of **innovation 2**: Phase-Aware Conditional
Experts (PACE-B) and the Phase-density Curriculum (PACE-C). PACE-B converts
the flow action head into a **phase-routed MoE with smooth switching
gates**; PACE-C stages training data across 3 levels based on "number of
boundaries per chunk". Both share the same β_t signal (produced by the
Round 4 Posterior), but act on different targets: PACE-B changes the
**network structure**, PACE-C changes the **data supply**.

## 1 Motivation and maths

### 1.1 PACE-B: Phase-gated MoE with smooth switching

The original flow head (Shortcut / Euler) regresses velocity through a
**single shared network**. That's fine for single-phase tasks, but in
LIBERO-Long 10-step tasks `pick → move → place` spans very different
kinematics, so the shared network has to spend its capacity "fitting
several modes at once", and variance in the boundary-step predictions
rises. PACE-B's idea:

$$v_\theta(x, \tau; \text{cond}) = \sum_{k=1}^{K} g_t[k] \cdot E_k(x, \tau, \text{cond})$$

where $E_k$ is the k-th expert MLP (Linear→SiLU→Linear→SiLU→Linear) and
$g_t \in \Delta^{K-1}$ is the gating distribution. The important part is
how the gate **evolves over time**:

$$
\begin{aligned}
g_t &= \alpha_t \cdot \hat p_t + (1 - \alpha_t) \cdot g_{t-1} \\
\alpha_t &= \sigma(\kappa \beta_t - \mu)
\end{aligned}
$$

- $\hat p_t$: the $(B, K)$ phase probability from the Round 4 posterior.
- $\beta_t$: Hellinger-based boundary signal, $\beta \in [0, 1]$.
- $\kappa = 5.0, \mu = 2.0$ (default): $\alpha$ hits $\geq 0.5$ around
  $\beta \geq 0.4$, giving an interpretable "boundary → switch" threshold.

$\alpha_t$ acts as the **switching speed**: inside a phase with
$\beta \approx 0 \Rightarrow \alpha \approx 0.12$, the gate mostly holds
$g_{t-1}$ (resisting $\hat p_t$ noise); at a boundary
$\beta \approx 0.8 \Rightarrow \alpha \approx 0.88$, the gate quickly
tracks the new $\hat p_t$.

**Why not just set $g_t = \hat p_t$?** The sampling jitter and softmax
sharpness of $\hat p_t$ would send the gate oscillating inside a single
phase, making experts switch back and forth — gradient gets diluted and
convergence slows. $g_{t-1}$ acts as a low-pass filter, with $\alpha_t$
dynamically setting the filter strength.

**Training vs inference special case:** during training the samples in a
batch are not in time order, so the concept of $g_{t-1}$ doesn't apply.
In that case `PhaseMoE.compute_gate(training=True)` returns $\hat p_t$
directly as $g_t$ without touching the `_running_gate` buffer.

### 1.2 PACE-C: Phase-density curriculum

In LIBERO-Long the per-episode boundary count varies widely:
`pick + place` has only 2 boundaries, while `stack 3 blocks` can have 4+.
Training on the full mix from day one makes the model juggle imitation
and transition simultaneously in the early phase. The curriculum opens
up in 3 stages:

| Stage | Step range | Allowed boundary_count ≤ | Meaning |
|-------|-----------|-----------------------|------|
| 0 | `[0, S₀)` | `curriculum_max_boundaries_stage1 = 1` | Single-phase chunks, pure imitation |
| 1 | `[S₀, S₁)` | `curriculum_max_boundaries_stage2 = 3` | Allow ≤3 switches |
| 2 | `[S₁, ∞)` | ∞ | No filtering |

Default `curriculum_stage_steps = (1000, 3000, 10000)` — these are
**absolute step boundaries**: `step < 1000` → stage 0;
`1000 ≤ step < 3000` → stage 1; `step ≥ 3000` → stage 2. Boundary count
is estimated from the gripper proxy (`actions[:, -1]` binarised with
`threshold=0.5`, then `|diff|.sum()`), matching the Round 1
`synthetic_demos` convention.

## 2 Implementation map

```
lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/
├── phase_centric/
│   ├── pace_b_moe.py           # smooth_phase_gate + PhaseMoE + FlowActionHeadPACE
│   └── pace_c_curriculum.py    # PhaseDensityCurriculum + boundary helpers
├── modeling_phaseqflow.py      # + self.pace_b_flow_head slot; forward/compute_loss/reset routing
└── configuration_phaseqflow.py # use_pace_b / moe_* / use_pace_c / curriculum_* fields ready
scripts/
├── train_local.py              # launches PhaseDensityCurriculum when use_pace_c=True
└── sanity_pace_b.py            # 3-phase synthetic trajectory + gate-L2 acceptance
tests/
├── test_pace_b.py              # 13 unit tests
└── test_pace_c.py              # 11 unit tests
```

### PACE-B block diagram

```mermaid
flowchart LR
    A[fused_obs, phase_embed, skill_latent] -->|cat| B[conditioner<br/>Linear]
    B -->|cond| C[PhaseMoE]
    D[p_hat &#124; β_t<br/>from posterior] --> E[smooth_phase_gate<br/>α = σ&#40;κβ - μ&#41;]
    E -->|α| F[g_t = α·p_hat + &#40;1-α&#41;·g_{t-1}]
    F -->|gate| C
    C -->|Σ g&#91;k&#93;·E_k| G[Euler step<br/>u ← u + dt·v]
    G -.->|flow_steps iterations| C
    G --> H[action_decoder<br/>Linear]
    H --> I[action_pred &#40;B, Ta, Da&#41;]
```

### PACE-B wiring (modeling_phaseqflow.py)

- `__init__`: if `use_pace_b=True`, instantiate `FlowActionHeadPACE` and
  bind it to `self.pace_b_flow_head`; otherwise `None`. Other modules
  (shortcut / euler / BID / correction / IQL) stay as-is.
- `forward()`: compute `phase_p_hat / phase_beta` first (moving the
  posterior ahead of the flow head), then route based on
  `pace_b_flow_head is not None`. Downstream verifier / loss continue to
  read `preds["action_pred"]`.
- `compute_loss()`: the PACE-B path bypasses the shortcut self-consistency
  $L_{FM} + L_{SC}$ and uses `F.mse_loss(action_pred, actions)` directly
  as the flow loss (the PACE-B Euler head has no shortcut meaning).
  PACE-A's $\beta_t$ weighting doesn't fire on the PACE-B path because
  `flow_train_out` has no `v_pred / v_target`.
- `reset()`: calls `self.pace_b_flow_head.reset_switching()` to clear
  `_running_gate`.

## 3 Unit test coverage

### test_pace_b.py (13 tests)

| # | Test | Assertion |
|---|------|--------|
| 1 | `test_smooth_phase_gate_scalar` | σ(-2), σ(0), σ(3) accuracy |
| 2 | `test_smooth_phase_gate_monotonic` | α is monotonically increasing in β |
| 3 | `test_phase_moe_forward_shape` | (B, latent_dim) output |
| 4 | `test_phase_moe_training_gate_equals_phat` | in training g_t == p_hat |
| 5 | `test_phase_moe_cold_start_uses_phat` | first inference step cold-start |
| 6 | `test_phase_moe_reset_switching_clears_state` | after reset, running_gate.numel==0 |
| 7 | `test_phase_moe_topk_sparsification` | top_k=1 gives a single non-zero gate |
| 8 | `test_expert_param_count_under_100k` | default hidden=128 ⇒ <100K |
| 9 | `test_gate_l2_high_at_boundary` | high β ⇒ L2 > 0.3 |
| 10 | `test_gate_l2_low_inside_phase` | low β ⇒ L2 < 0.05 |
| 11 | `test_flow_head_pace_output_shape` | (B, Ta, Da) output |
| 12 | `test_policy_integration_pace_b_off` | default None |
| 13 | `test_policy_integration_pace_b_on` | compute_loss + backward work |

### test_pace_c.py (11 tests)

| # | Test | Assertion |
|---|------|--------|
| 1~3 | `test_compute_boundaries_*` | np / torch / batch input consistency |
| 4 | `test_curriculum_stage_progression` | step drives stage up 0→1→2 |
| 5 | `test_curriculum_stage_post_end` | stable after the last stage |
| 6 | `test_should_include_episode_filters` | threshold rejects high-boundary episodes |
| 7 | `test_filter_chunks_by_boundary_count` | index list is correct |
| 8 | `test_filter_inf_cap_returns_all` | inf cap ⇒ keep everything |
| 9 | `test_state_dict_roundtrip` | save/load preserves current_step |
| 10 | `test_build_curriculum_filter_dicts` | dict-dataset precompute |
| 11 | `test_invalid_stage_steps_length` | non-3-tuple raises |

All 65 pre-existing tests still pass (`pytest tests/ -q` → 65 passed).

## 4 Sanity experiment: gate L2 distance

3-phase synthetic posterior (phase 0 → expert 0, phase 1 → expert 2,
phase 2 → expert 1). β is set to 0.8 in a ±1 step window around the
boundary and 0.05 elsewhere. Results from running
`scripts/sanity_pace_b.py` over T=120 steps (see
`round-6-sanity-pace-b.json`):

| Metric | Value | Threshold | Verdict |
|------|-----|------|------|
| boundary L2 mean | 0.4471 | > 0.3 | |
| boundary L2 max | 1.1834 | — | — |
| interior L2 mean | 4.0e-05 | < 0.05 | |
| interior L2 max | 3.4e-04 | — | — |

**Reading:** at a boundary the gate completes a >44% L2 jump in one step
(well above the 0.3 acceptance threshold); inside a phase it stays
essentially still (4e-5 << 0.05). The sigmoid shape of α widens the L2
gap between "noise jitter (β≈0.05)" and "real boundary (β≈0.8)" by
>10000×, which is exactly the smooth-yet-fast expert switching we want.

Gate trajectory plots live in `round-6-sanity-pace-b.png` (two subplots:
top shows the 4 expert gates over time, bottom shows step-to-step L2
differences, with red dashed lines marking boundary steps and
green/orange horizontal lines showing the 0.3/0.05 acceptance
thresholds).

## 5 Comparison to related methods (Round 6 view)

| Method | Gate form | Temporal smoothing | Train/inference same? | Round 6 choice |
|------|---------|---------|-----------------|---------|
| Vanilla MoE (Shazeer et al., arXiv 1701.06538) | top-k softmax | none | same | hard routing jitter at single points |
| Switch Transformer (Fedus et al., arXiv 2101.03961) | top-1 | none | same | forced single expert drops soft info |
| Gated Linear Unit gating | sigmoid | none | same | no phase semantics |
| **PACE-B (ours)** | soft + EMA $g_{t-1}$ | **β-adaptive** | training pass-through / inference EMA | |

PACE-B's training pass-through (returning p_hat when `training=True`)
keeps the gradient flowing through all experts, avoiding the
Switch-Transformer-style load-balancing hack. At inference it uses EMA
as a low-pass filter to dodge posterior noise.

## 6 Hooks for Round 7

- **PCAR replan trigger:** Round 7 will watch $g_t$ L2 jumps in the
  rollout loop as one of the hard signals for "a phase change just
  happened"; the `current_gate()` accessor on PACE-B is already in
  place.
- **Theoretical efficiency bound:** Round 7 will prove that
  "$\alpha = \sigma(\kappa\beta - \mu)$ gives Lipschitz-continuous gate
  switching" (as opposed to hard argmax switching), giving the
  continuity assumption needed for the upper-bound analysis.
