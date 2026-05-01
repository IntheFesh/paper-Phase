# PACE v2 Architecture

> **Scope**: an architectural specification aimed at reproduction
> engineers and algorithm reviewers. It takes the current PACE v2
> codebase as its reference point and describes module signatures,
> tensor shapes, the cliff-detection theory, the seven-config
> ablation matrix, and the verification suite. The four headings
> below mirror the source-code organisation under
> `lerobot_policy_phaseqflow/` and `configs/`.

---

## Contents

- [§1 System Architecture](#1-system-architecture)
  - [1.1 Data flow](#11-data-flow-overview)
  - [1.2 Tensor shapes](#12-tensor-shape-table)
  - [1.3 Module signatures](#13-core-module-signatures)
  - [1.4 Training stack](#14-training-stack)
  - [1.5 Three-stage curriculum](#15-three-stage-curriculum)
- [§2 Phase-Centric Theory Framework](#2-phase-centric-theory-framework)
  - [2.1 Phase identifiability / InfoNCE](#21-phase-identifiability-chunk-level-infonce)
  - [2.2 Bhattacharyya β_t = I^(1)](#22-phase-posterior-and-bhattacharyya-boundary-signal-hati1)
  - [2.3 Boundary-aware FM loss](#23-variational-lower-bound-for-pace-a-weighted-flow-matching)
  - [2.4 DKW convergence for PCAR](#24-dkw-convergence-for-pcar-budget-quantile)
  - [2.5 Theory synthesis](#25-theory-synthesis-how-the-four-pillars-connect)
- [§3 Cliff Estimators, Concordance, and PCAR](#3-cliff-estimators-concordance-and-pcar)
  - [3.1 Phase InfoNCE (implemented)](#31-phase-identifiability--chunk-level-infonce)
  - [3.2 I^(1) Bhattacharyya β_t (implemented)](#32-hati1--bhattacharyya-boundary-signal-betat-implemented)
  - [3.3 I^(2) Action variance (implemented)](#33-hati2--action-ensemble-variance-sigma_t2)
  - [3.4 I^(3) Velocity curvature (implemented)](#34-hati3--velocity-field-curvature-kappa_t)
  - [3.5 Concordance C_t (implemented)](#35-concordance-c_t)
  - [3.6 PCAR](#36-pcar--predictability-cliff-adaptive-replanning)
  - [3.7 Boundary-aware flow loss](#37-boundary-aware-flow-loss-pace-a)
  - [3.8 Summary table](#38-summary)
- [§4 Ablation Matrix (v2, seven configs)](#4-ablation-matrix-v2-seven-configs)
  - [4.1 Matrix structure](#41-matrix-structure)
  - [4.2 Design principles](#42-design-principles)
  - [4.3 Statistical aggregation](#43-statistical-aggregation)
  - [4.4 Specificity test](#44-falsification-design-specificity-test)
  - [4.5 SOTA reference points](#45-sota-reference-points)
  - [4.6 Reproduction commands](#46-reproduction-commands)
- [§5 Verification System](#5-verification-system)
  - [5.1 Acceptance table](#51-scriptassertionacceptance-table)
  - [5.2 Hungarian alignment](#52-identifiability-verification-hungarian-alignment)
  - [5.3 Posterior peak alignment](#53-posterior-verification-peak-alignment)
  - [5.4 Boundary-aware FM loss sanity](#54-boundary-aware-flow-loss-sanity)
  - [5.5 PCAR budget verification](#55-pcar-budget-quantile-verification)
  - [5.6 Coverage table](#56-coverage)
  - [5.7 Self-check commands](#57-one-liner-self-check)

---

## 1 System Architecture

PACE v2 is a three-layer generative policy for long-horizon
robotic manipulation. The observation is decomposed as
$x_t = \{V_t, S_t, L, H_t\}$ (vision, state, language, history)
and flows through `DualBackboneVisionTokenizer` fusion,
`HierarchicalPlanner` macro/micro phase discretisation, and
`ShortcutFlowActionHead` action-chunk decoding. A separate
**cliff-detection branch** taps off the planner and the flow
head to produce a continuous "predictability cliff" signal that
gates **PCAR** (PACE Closed-loop Adaptive Replanning) at
inference time. Boundary-aware reweighting of the flow loss
closes the loop at training time.

The legacy A2C2 correction head, IQL chunk verifier, and BID
sampler remain in the source tree but are gated off in every
shipped config (`use_iql_verifier=false`, `use_correction_head=
false`, `use_bid_sampling=false`). They are not part of the PACE
v2 main path.

### 1.1 Data flow overview

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Observation  x_t = { V_t ∈ R^{V×3×H×W},  S_t ∈ R^{d_s},            │
  │                        L   ∈ N^{L_tok},   H_t ∈ R^{d_h} }            │
  └─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Layer 1 : DualBackboneVisionTokenizer                              │
  │  ─────────────────────────────────────                              │
  │    SigLIP2 + DINOv2   ──LoRA (r=16, α=32, DoRA)──►  Patch tokens    │
  │    Prismatic concat   ──1×1 Conv (C_s+C_d → 256)─►  Fused grid      │
  │    T5-Base (frozen)   ──FiLM (γ,β init identity)─►  Language-modul. │
  │    Perceiver readout  ──8 queries × XA──────────►   (B, 8, 256)     │
  │    Uncertainty gate   ──σ(MLP[readout, state])──►   (B, 256)        │
  └─────────────────────────────────────────────────────────────────────┘
                                  │ fused_obs: (B, 256)
                                  │ context_tokens: (B, N_ctx, 256)
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Layer 2 : HierarchicalPlanner   (phase_mode = "hierarchical")      │
  │  ─────────────────────────                                          │
  │    Macro FSQ  (levels=[5,4],  K₁=20)   ─► z_macro    : (B,)         │
  │    Micro FSQ  (levels=[6,5],  K₂=30)   ─► z_micro    : (B,)         │
  │      (or flat FSQ K=240 / Gumbel K=16  with phase_mode = "flat")    │
  │    Embedding(K_macro, 32)              ─► macro_embed: (B, 32)      │
  │    Embedding(K_micro, 32)              ─► micro_embed: (B, 32)      │
  │    InfoNCE heads (macro, micro)        ─► L_NCE_macro + 0.5·L_NCE_micro │
  └─────────────────────────────────────────────────────────────────────┘
                                  │ planner_out
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Layer 3 : ShortcutFlowActionHead   (4-NFE shortcut flow)           │
  │  ──────────────────────────────                                     │
  │    cond = [fused_obs ⊕ macro_embed ⊕ micro_embed]  (B, 320→256)    │
  │    SmallDiT :  4 × _DiTBlock(AdaLN-Zero),  hidden=256,  heads=8     │
  │    Training : sample (t, d=2^{-k}); boundary-aware FM loss          │
  │               L_FM = mean[(1 + λ·β_t^micro) · ||v_θ - v*||²]        │
  │               + self-consistency L_sc                               │
  │    Inference: x_1 = x_0 + v_θ(x_0, 0, 1 | cond),  4-NFE shortcut    │
  │    ──►  action chunk  A_t ∈ R^{Ta × Da}  =  (B, 16, 16)             │
  └─────────────────────────────────────────────────────────────────────┘
                                  │ A_t
                                  ▼
                       (executed for ≤ H steps, or
                        until PCAR fires a replan)

           ┌──── parallel cliff branch (read-only) ──────────────┐
           │                                                     │
           │  I^(1)  =  -β_t                                     │
           │            (PhasePosteriorEstimator, IMPLEMENTED)   │
           │                                                     │
           │  I^(2)  =  -σ_t²    [compute_I_hat_2]                 │
           │            (action-sample variance; IMPLEMENTED)    │
           │                                                     │
           │  I^(3)  =  -||v_θ(c_t) - v_θ(c_{t-1})||²            │
           │            (velocity curvature; IMPLEMENTED)        │
           │                                                     │
           │  Concordance  C_t  =  1/3 · Σ_k rank_W(I^(k))      │
           │                       (rank-window W=50; IMPLEMENTED)│
           │                                                     │
           │  PCAR trigger:  budget-quantile  τ_t = Q̂_n(1-ε)    │
           │                 replan ⇔ C_t > τ_t                  │
           │                 (or BayesianPCARTrigger Beta-mix)   │
           └─────────────────────────────────────────────────────┘
                                  │  replan? → re-encode obs and rerun L1-L3
                                  ▼
                          a_t ∈ R^{Da}   (executed)
```

**All three cliff estimators are now fully implemented** (v2.1).
`compute_I_hat_1`, `compute_I_hat_2`, `compute_I_hat_3`, and
`compute_concordance_C` in
`lerobot_policy_phaseqflow/phase_centric/cliff_estimators.py`
are all wired end-to-end.  The concordance score $C_t$ is computed
from all available estimators at each forward step via the
`_RollingRankBuffer` window (default $W=50$) and forwarded into
`PCARTrigger` when `pcar_input_signal="concordance"`.

### 1.2 Tensor shape table

Conventions: $B$ batch, $V$ number of cameras ($=2$), $T_a$ action
chunk length ($=16$), $D_a$ action dimension ($=16$), $D=256$
fusion hidden size, $K_1=20$ macro / $K_2=30$ micro phase codebook
sizes, $L_{\text{tok}}=16$ T5 token count.

| Stage | Tensor | Shape | Note |
| :-- | :-- | :-- | :-- |
| Input | `images` | $(B, V, 3, 224, 224)$ | dual-view RGB |
| Input | `states` | $(B, 8)$ | proprioception |
| Input | `language_ids` | $(B, L_{\text{tok}})$ | T5 token ids |
| Input | `history` | $(B, 8)$ | previous-frame state |
| L1 | SigLIP features | $(B·V, 196, 768)$ | patch 14×14 |
| L1 | DINOv2 features | $(B·V, 256, 384)$ | CLS removed |
| L1 | Fused grid | $(B, 256, g, g)$ | after 1×1 Conv |
| L1 | `vision_tokens` | $(B, 8, 256)$ | Perceiver readout |
| L1 | `language_tokens` | $(B, 1, 256)$ | T5 mean-pool + FiLM |
| L1 | `fused_obs` | $(B, 256)$ | uncertainty-gate output |
| L1 | `context_tokens` | $(B, N_{\text{ctx}}, 256)$ | $[s, h, l, r]$ concat |
| L2 | `macro_logits` | $(B, K_1)$ | K₁=20 (hierarchical) |
| L2 | `micro_logits` | $(B, K_2)$ | K₂=30 (hierarchical) |
| L2 | `z_macro / z_micro` | $(B,)$ | discrete phase ids |
| L2 | `macro_embed / micro_embed` | $(B, 32)$ | `Embedding(K, 32)` |
| L3 | FM input $x_t$ | $(B, T_a, D_a)$ | training |
| L3 | cond vector | $(B, 256)$ | `conditioner([obs ⊕ z^macro ⊕ z^micro])` |
| L3 | `action_pred` | $(B, 16, 16)$ | 4-NFE shortcut decoding |
| Cliff | $\beta_t$ | $(B,)$ | $1-\sum_k\sqrt{\hat p_t \hat p_{t-1}}$, micro-level |
| Cliff | $\sigma_t^2$ | $(B,)$ | per-step ensemble variance (`compute_I_hat_2`) |
| Cliff | $\kappa_t$ | $(B,)$ | velocity-anchor $L^2$ jump (`compute_I_hat_3`) |
| Cliff | $C_t$ | $(B,)$ | rank-window mean of $I^{(1/2/3)}$ (`compute_concordance_C`) |
| PCAR | `should_replan` | $(B,)$ bool | scalar threshold check |

### 1.3 Core module signatures

The L1/L2/L3 modules are defined in
`lerobot_policy_phaseqflow/src/lerobot_policy_phaseqflow/modeling_phaseqflow.py`;
the cliff-detection sub-modules live in the sibling
`phase_centric/` subpackage.

```python
class DualBackboneVisionTokenizer(nn.Module):
    # modeling_phaseqflow.py
    def forward(self, images, states, language_ids, language_mask,
                history, masks) -> Dict[str, Tensor]:
        # keys: fused, context_tokens, vision_tokens, state_tokens,
        #       language_tokens, history_tokens, uncertainty_gate

class HierarchicalPlanner(nn.Module):
    # modeling_phaseqflow.py
    def forward(self, fused_obs, phase_labels=None,
                phase_mode=None) -> Dict[str, Tensor]:
        # hierarchical: macro_logits, micro_logits, z_macro, z_micro,
        #               macro_embed, micro_embed
        # flat:         phase_logits, phase_id, phase_embed, skill_latent

class ShortcutFlowActionHead(nn.Module):
    # modeling_phaseqflow.py
    def forward(self, fused_obs, macro_embed, micro_embed,
                actions_gt=None, beta_t=None, training=True
                ) -> Dict[str, Tensor]:
        # training keys: fm_loss, sc_loss, action_pred, v_pred, v_target
        # inference keys: action_pred (4-NFE shortcut)

# Cliff branch (phase_centric/)
class PhasePosteriorEstimator(nn.Module):
    # phase_centric/phase_posterior.py
    @staticmethod
    def _bhattacharyya_beta(p_cur, p_prev) -> Tensor
    def forward_sequence(phase_logits) -> {p_hat, beta}    # batched
    def step(phase_logits_t)            -> {p_hat, beta}    # rollout

# Public cliff-namespace API (phase_centric/cliff_estimators.py)
def compute_I_hat_1(phase_beta) -> Tensor             # IMPLEMENTED  (= -β_t)
def compute_I_hat_2(action_samples)                   # IMPLEMENTED  (= -σ_t²)
def compute_I_hat_3(v_theta_ct, v_theta_ct_prev)      # IMPLEMENTED  (= -||Δv||²)
def compute_concordance_C(i_hat_values, window_size)  # IMPLEMENTED  (rank-window fusion)

class PCARTrigger:                                    # phase_centric/pcar_trigger.py
    history: Deque[float]                              # maxlen=1000
    warmup_min: int = 50
    budget_eps: float = 0.1
    def update_and_check(signal: float) -> bool        # returns replan flag

class BayesianPCARTrigger:                            # phase_centric/b_pcar.py
    # Beta-mixture posterior P(changepoint | history); same .update_and_check API
```

### 1.4 Training stack

| Component | Value | Note |
| :-- | :-- | :-- |
| Precision | `bf16` | `adam_eps=1e-6` avoids underflow |
| Optimizer | PagedAdamW8bit | `bitsandbytes`, falls back to `torch.optim.AdamW` |
| $(\beta_1, \beta_2)$ | $(0.9,\ 0.95)$ | common for large models |
| Grad clip | $\lVert g \rVert_2 \le 1.0$ | global norm |
| Schedule | Cosine, warmup=500 | `transformers.get_cosine_schedule_with_warmup` |
| EMA | decay $=0.9999$, power $=0.75$ | applied only to `flow_action_head` |
| Grad ckpt | TransformerEncoderLayer | roughly 35% memory savings |
| Param groups | backbone / LoRA / head | lr $\in \{0,\ 5\!\times\!10^{-5},\ 10^{-4}\}$ |
| Action norm | quantile [0.01, 0.99] $\to [-1, 1]$ | RDT-style |
| State noise | Gaussian injection at SNR $40$ dB | training regulariser |

### 1.5 Three-stage curriculum

PACE v2 ships with three training stages, declared in
`configs/train/`:

| Stage YAML | Unfrozen modules | Main loss | Purpose |
| :-- | :-- | :-- | :-- |
| `01_pretrain_multimodal.yaml` | vision/language adapter | vision–state contrastive + masked modelling + future-token prediction | multimodal token alignment; phase-centric features off |
| `02_train_phase_and_flow.yaml` | hierarchical planner + flow head | $\mathcal{L}_{\text{imit}} + 0.5\,\mathcal{L}_{\text{flow}} + 0.1\,\mathcal{L}_{\text{InfoNCE}}^{\text{macro+micro}}$ | joint train of phase encoder and boundary-aware flow head |
| `03_finetune_replan.yaml` | PCAR / B-PCAR / concordance-window calibration only | calibration objective on held-out LIBERO-Long split | freeze main net; tune the replanning trigger |

Stage switching is driven by `PhaseQFlowConfig.stage` ($\in
\{\texttt{pretrain\_multimodal},
\texttt{train\_phase\_and\_flow},
\texttt{finetune\_replan}\}$), and the corresponding
`stage_freeze_*` and `freeze_all_main` flags decide which
sub-modules participate in backprop. The previous four-stage
recipe (separate `train_latent` and `train_flow` stages) was
collapsed into stage 2 in v2 because the InfoNCE-identifiability
loss and the boundary-aware flow loss were found to be
co-trainable without curriculum staging.

---

## 2 Phase-Centric Theory Framework

The theoretical motivation for PACE v2 is to promote
"phase" from a passive auxiliary feature to a first-class control
variable that is identifiable, measurably changing, and
budget-schedulable. This section lays out four mathematical
pillars.

### 2.1 Phase identifiability (Chunk-level InfoNCE)

**Problem**: training an unsupervised phase code with only
reconstruction loss collapses to a degenerate solution (all
observations mapped to the same code), leaving the phase labels
incomparable across seeds/episodes.

**Method**: port the identifiable-VAE idea of Khemakhem et al.
(NeurIPS 2020, arXiv 1907.04809) to chunk-level contrastive
learning. Let $f_\phi : (o, a_{1:T_a}) \mapsto \mathbb{R}^D$ be
the context encoder and
$g_\psi : z \in \{1,\dots,K\} \mapsto \mathbb{R}^D$ the phase
prototypes. Build an in-batch same-phase mask
$M_{ij} = \mathbb{1}[z_i = z_j]$ and take the InfoNCE loss:

$$
\mathcal{L}_{\text{InfoNCE}}
  = -\,\mathbb{E}_i\!\left[
      \log
      \frac{\exp\big(\langle f_\phi(o_i, a_i),\, g_\psi(z_i)\rangle / \tau\big)}
           {\sum\limits_{j \in \mathcal{N}(i) \cup \{i\}}
              \exp\big(\langle f_\phi(o_i, a_i),\, g_\psi(z_j)\rangle / \tau\big)}
    \right]
$$

where the negative set is
$\mathcal{N}(i) = \{j : z_j \neq z_i,\ j \neq i\}$, and
same-phase off-diagonal entries are masked out (treated as neither
positive nor negative, so we don't punish correct matches).

**Identifiability claim** (informal): if the data-generating
process satisfies the conditional independence
$p(o, a \mid z) \neq p(o, a \mid z')$ for all $z \neq z'$ and
$\mathcal{L}_{\text{InfoNCE}} \to 0$, then the representation
$(f_\phi, g_\psi)$ converges up to permutation — i.e. phase
labels are recoverable — in line with the CPC mutual-information
lower bound $I(X; C) \ge \log N - \mathcal{L}_{\text{InfoNCE}}$
of van den Oord et al. (arXiv 1807.03748).

**Acceptance metric**: `verify_identifiability.py` trains on the
same data across 3 seeds, aligns codebooks via the Hungarian
algorithm, and computes the permuted-agreement; threshold $\ge 0.7$
(real GPU runs target $>0.8$).

**Temperature $\tau = 0.1$**: empirically stable for
$\tau \in [0.05, 0.2]$; smaller values risk gradient explosion and
larger values collapse toward a uniform distribution.

### 2.2 Phase posterior and Bhattacharyya boundary signal ($\hat{I}^{(1)}$)

**Problem**: the boundary-aware flow loss and PCAR both require a
smooth, $[0,1]$-normalised boundary signal in each rollout step
rather than a hard argmax flip. $\beta_t$ is the first cliff
estimator ($\hat I^{(1)} = -\beta_t$) and the only one currently
implemented end-to-end.

**Method**: let $p_t \in \Delta^{K-1}$ be the instantaneous
post-softmax phase probability and apply EMA smoothing

$$
\hat p_t = \alpha\, p_t + (1-\alpha)\,\hat p_{t-1},
\qquad \alpha = 0.3
$$

The boundary probability is then a first-order approximation of
the Bhattacharyya distance:

$$
\beta_t
  = 1 - \mathrm{BC}(\hat p_t,\,\hat p_{t-1})
  = 1 - \sum_{k=1}^{K} \sqrt{\hat p_t(k)\, \hat p_{t-1}(k)}
$$

**Key properties**:

1. **Cauchy–Schwarz upper bound**:
   $\mathrm{BC}(p, q) = \sum_k \sqrt{p_k q_k} \le
    \sqrt{\sum_k p_k} \cdot \sqrt{\sum_k q_k} = 1$,
   with equality iff $p = q$. So $\beta_t \in [0, 1]$.

2. **Relation to Hellinger distance**:
   $H^2(p, q) = 1 - \mathrm{BC}(p, q) = \beta$,
   meaning $\beta_t$ is exactly the squared Hellinger distance.

3. **Relation to total variation**: the Hellinger–TV inequality
   $\tfrac{1}{2}H^2(p, q) \le d_{\text{TV}}(p, q) \le H(p, q)$
   gives

   $$
   \tfrac{1}{2}\,\beta_t
     \le d_{\text{TV}}(\hat p_t, \hat p_{t-1})
     \le \sqrt{\beta_t}
   $$

   so $\beta_t$ acts as a computable stand-in for TV distance (TV
   itself requires the full variation sum).

4. **Comparison with argmax switching**: the hard switch
   $\mathbb{1}[z_t \neq z_{t-1}]$ is unstable under logit jitter;
   $\beta_t$ varies continuously with probability mass and its
   gradient backprops cleanly through the planner.

**Implementation**: `PhasePosteriorEstimator` maintains an internal
`_running_p` buffer. Training calls `forward_sequence` in batch
mode; rollout calls `step` once per frame. The default
`pace_a_detach_beta=True` means $\beta_t$ is used only as a
weight and does not backprop into the planner (otherwise it would
fight InfoNCE).

### 2.3 Variational lower bound for PACE-A weighted flow matching

**Problem**: in long-horizon tasks error concentrates at phase
boundaries, but naive MSE weighs every timestep equally, which
underfits the boundary modes.

**Method**: weight dynamically by $\beta_t$, paired with a
Bernoulli entropy regulariser:

$$
\mathcal{L}_{\text{PACE-A}}
  = \underbrace{\mathbb{E}_{t, x_t, d}\!\big[(1 + \lambda\,\beta_t)\,
      \lVert v_\theta(x_t, t, d \mid c) - v^*_t \rVert_2^2\big]}_{\text{weighted FM}}
  \;-\; \eta\, \underbrace{\mathbb{E}_t\big[H(\beta_t)\big]}_{\text{entropy reg}}
$$

where $H(\beta) = -\beta\log\beta - (1-\beta)\log(1-\beta)$ is the
Bernoulli entropy and $\lambda=2.0$, $\eta=0.01$.

**Reading as a variational lower bound**: the data likelihood
satisfies $\log p_\theta(a \mid o) \ge
\mathbb{E}_{q(z \mid o,a)}[\log p_\theta(a \mid o, z)] - D_{\text{KL}}(q \| p)$.
Under the Flow Matching framework (Lipman et al., arXiv 2210.02747),
$-\log p_\theta(a \mid o, z) \propto \lVert v_\theta - v^* \rVert^2$.
Rewriting the prior $p(z)$ as a $\beta$-dependent mixture:

$$
p_\theta(a \mid o) \approx
  \underbrace{p_{\text{interior}}(a \mid o)}_{\text{weight}\ 1}
  \cdot
  \underbrace{p_{\text{boundary}}(a \mid o)^{\lambda\beta}}_{\text{weight}\ \lambda\beta}
$$

Taking the log yields the weighted FM term; the entropy regulariser
$-\eta H(\beta)$ pushes $\beta$ away from the uncertain region near
$0.5$ toward a $\{0, 1\}$ binary decision, corresponding to an
entropy-contracting prior on the variational posterior
$q(z \mid o, a)$.

**Counter-example design**: `sanity_pace_a.py` synthesises three
groups — boundary-only, interior-only, mixed — and checks that
PACE-A `full` reduces boundary-step FM loss by $\ge 20\%$ relative
to `no_weight`, while also requiring $\mathbb{E}[\beta] > 0.1$ (so
that $\beta$ doesn't collapse to 0 and make the weighting vacuous).

### 2.4 DKW convergence for PCAR budget quantile

**Problem**: over-frequent replanning in long tasks wastes compute
(each trigger costs one additional flow forward pass), but being
too conservative misses critical switching points. We want an
adaptive trigger threshold with few parameters and a clean
statistical interpretation.

**Method**: maintain the rolling empirical distribution of the
input cliff signal $s_t$ (= $C_t$ when concordance is fully
implemented; $= \beta_t$ for the current $I^{(1)}$-only path),
$\hat F_n(x) = \tfrac{1}{n}\sum_{i=1}^{n}\mathbb{1}[s_i \le x]$,
and given a replan budget $\epsilon \in (0, 1)$ take

$$
\tau^{\text{cp}}_n
  = \inf\{x : \hat F_n(x) \ge 1 - \epsilon\}
  = \hat Q_n(1-\epsilon)
$$

The trigger rule is
$\text{replan}_t = \mathbb{1}[s_t > \tau^{\text{cp}}_n]$.

**DKW convergence guarantee** (Dvoretzky–Kiefer–Wolfowitz
inequality):

$$
\Pr\!\left(
  \sup_{x \in \mathbb{R}} \big|\hat F_n(x) - F(x)\big| > \varepsilon
\right)
\le 2\, e^{-2n\varepsilon^2}
$$

Taking $\varepsilon = \sqrt{\log(2/\delta) / (2n)}$ (the Massart
constant), then with confidence $1-\delta$:

$$
\big|\Pr(s_t > \tau^{\text{cp}}_n) - \epsilon\big|
  \le \sqrt{\frac{\log(2/\delta)}{2n}}
$$

**Numerical reading**: for $n = 1000$, $\delta = 0.05$ the
deviation is $\le \sqrt{\log 40 / 2000} \approx 0.043$. The actual
replan rate therefore differs from the budget $\epsilon$ by at most
$4.3\%$, consistent with `verify_pcar_budget.py`'s acceptance
metric $|\text{diff}| < 0.005$ (averaged over a 1000-step
synthetic trajectory).

**Warm start**: for $n < n_{\min} = 50$ we fall back to the static
threshold $\tau^{\text{cp}} = 0.4$ (`pcar_change_threshold`) to
dodge the noisy early quantile.

**Comparison with a static threshold**: a static $\tau = 0.4$
swings the replan rate dramatically as the task's signal
distribution shifts across tasks; the budget-quantile design keeps
the actual rate locked to $\epsilon \pm O(1/\sqrt{n})$.

### 2.5 Theory synthesis: how the four pillars connect

```
         ┌─────────────── (§2.1 identifiability) ───────────────┐
         │   InfoNCE → phase z aligned across seeds/episodes     │
         └───────────────────────┬───────────────────────────────┘
                                 │ macro_logits / micro_logits
                                 ▼
         ┌─────────────── (§2.2 I^(1) = β_t  IMPLEMENTED) ──────┐
         │   Bhattacharyya EMA → β_t ∈ [0, 1]                    │
         └───────────┬───────────────────────────────────────────┘
                     │                          (§3 cliff branch)
         ┌───────────┴──────────────────────────────────────────┐
         │   I^(2) = -σ_t²     (PENDING: variance estimator)    │
         │   I^(3) = -||Δv||²  (PENDING: curvature estimator)   │
         │   C_t = ⅓ Σ rank_W(I^(k))  (PENDING: concordance)   │
         └───────────┬──────────────────────────────────────────┘
                     │ s_t  (= C_t when complete; = β_t today)
           ┌─────────┴──────────┐
           ▼                    ▼
  (§2.3 boundary-aware FM)   (§2.4 PCAR budget trigger)
  w = 1 + λβ_t               τ = Q̂_n(1-ε)  →  DKW convergence
  fits phase boundaries       stable replan rate across tasks
```

The shared design principle: an identifiable discrete $z$ is
converted to a continuous boundary signal; that signal drives both
training (boundary-aware flow loss) and inference (PCAR). The
three-estimator concordance in §3 extends this single signal to
a more robust joint detector once $I^{(2)}$ and $I^{(3)}$ are
implemented.

---

## 3 Cliff Estimators, Concordance, and PCAR

The predictability cliff is the sharp drop in action-distribution
predictability at phase boundaries (grasp → transport → place).
PACE v2 quantifies it with three estimators, fuses them into a
single concordance score $C_t$, and uses that score to gate
adaptive replanning (PCAR) and to reweight the flow loss.

### 3.1 Phase Identifiability — Chunk-level InfoNCE

**Motivation**: without explicit identifiability training the
phase codebook collapses (all observations map to the same code),
making phase labels incomparable across seeds and giving the cliff
estimators no signal to work with.

**Method**: treat $(o_i, a_i^{1:T_a})$ as a positive sample for
its assigned phase code $z_i$, and all cross-phase pairs as
negatives:

$$
\mathcal{L}_{\text{NCE}}
  = -\frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}}
    \log \frac{\exp(s_{ii}/\tau)}
              {\sum_{j \in \mathcal{N}(i)\cup\{i\}} \exp(s_{ij}/\tau)},
  \quad s_{ij} = \langle f_\phi(o_i, a_i),\, g_\psi(z_j)\rangle
$$

Applied separately for macro ($K_1=20$) and micro ($K_2=30$)
levels; total InfoNCE weight $= 0.1\,(L_{\text{NCE}}^{\text{macro}}
+ 0.5\,L_{\text{NCE}}^{\text{micro}})$ in stage 2.

**Implementation**

```python
# phase_centric/identifiability.py
class ChunkInfoNCEHead(nn.Module):
    context_encoder: Linear(D + T_a·D_a, D) → SiLU → Linear(D, D)
    phase_embed:     Embedding(K, D)
    forward(fused_obs, action_chunk, phase_logits) -> (loss, diag)

# PhaseQFlowConfig:
#   use_chunk_infonce            = True
#   chunk_infonce_weight         = 0.5
#   chunk_infonce_temperature    = 0.1
```

**Verification**: `verify_identifiability.py` trains across 3
seeds, aligns codebooks with the Hungarian algorithm, and requires
permuted-agreement $\ge 0.7$ (§5.2).

---

### 3.2 $\hat{I}^{(1)}$ — Bhattacharyya boundary signal $\beta_t$ (IMPLEMENTED)

**Definition**:
$\hat I^{(1)}(t) = -\beta_t$, where $\beta_t \in [0,1]$ is the
Bhattacharyya distance between consecutive smoothed phase
posteriors.

**Computation**: given $p_t = \text{softmax}(\text{micro\_logits}_t)$:

$$
\hat p_t = 0.3\,p_t + 0.7\,\hat p_{t-1}
\qquad
\beta_t = 1 - \sum_{k=1}^{K_2} \sqrt{\hat p_t(k)\,\hat p_{t-1}(k)}
$$

Key properties (proved in §2.2): $\beta_t \in [0,1]$; equals
$H^2(p,q)$ (squared Hellinger); sandwiches TV distance as
$\beta_t/2 \le d_\text{TV} \le \sqrt{\beta_t}$.

**Implementation**

```python
# phase_centric/phase_posterior.py
class PhasePosteriorEstimator(nn.Module):
    @staticmethod
    def _bhattacharyya_beta(p_cur, p_prev) -> Tensor
    def forward_sequence(phase_logits) -> {p_hat, beta}   # batched
    def step(phase_logits_t)           -> {p_hat, beta}   # per rollout step
    def reset(batch_size)

# phase_centric/cliff_estimators.py
def compute_I_hat_1(phase_beta: Tensor) -> Tensor   # returns -beta_t
```

**Config**: `use_phase_boundary_posterior = True`,
`phase_posterior_smooth_alpha = 0.3`,
`pace_a_detach_beta = True` (gradient detached to avoid
interfering with InfoNCE).

**Verification**: `verify_phase_posterior.py`, peak-F1 $\ge 0.5$
at $\pm 3$-frame tolerance on 50 synthetic demos.

---

### 3.3 $\hat{I}^{(2)}$ — Action-ensemble variance $\sigma_t^2$

**Definition**:
$\hat I^{(2)}(t) = -\sigma_t^2$, where
$\sigma_t^2 = \frac{1}{N}\sum_{i=1}^{N}\|a_t^{(i)} - \bar a_t\|^2$
is the variance of $N$ independent flow samples conditioned on
the same observation.

**Rationale**: at phase boundaries the flow-matching score
function becomes multi-modal; independent samples spread out and
variance spikes, signalling a cliff even before the planner's
phase posterior shifts.

**Input**: `action_samples` of shape `(N, B, Ta, Da)` with $N \ge 2$.
The BID multi-sample chunks stored as `bid_chunks` in the policy's
forward output are consumed directly; when BID is disabled, this
estimator is skipped gracefully.

```python
# phase_centric/cliff_estimators.py  — IMPLEMENTED
def compute_I_hat_2(action_samples: Tensor) -> Tensor
#   action_samples: (N, B, Ta, Da) — returns (B,), values ≤ 0
```

---

### 3.4 $\hat{I}^{(3)}$ — Velocity-field curvature $\kappa_t$

**Definition**:
$\hat I^{(3)}(t) =
-\|v_\theta(x_\tau, \tau, c_t) - v_\theta(x_\tau, \tau, c_{t-1})\|_2^2$,
where $x_\tau = 0$ (fixed zero anchor) and $\tau = 0.5$,
and $c_t, c_{t-1}$ are consecutive condition vectors.

**Rationale**: if the policy's velocity field is continuous at
interior steps but jumps at boundaries (because $c_t$ encodes a
different phase), then the $L^2$ difference between the velocity
field at the fixed anchor detects the transition without
requiring an ensemble.

**Implementation**: `PhaseQFlowPolicy` caches `_v_theta_prev` across
timesteps; `reset()` clears it between episodes.  The anchor is
always $x_\tau = \mathbf{0}$, $\tau = 0.5$.

```python
# phase_centric/cliff_estimators.py  — IMPLEMENTED
def compute_I_hat_3(
    v_theta_ct: Tensor,       # velocity at c_t,   shape (B, Ta, Da)
    v_theta_ct_prev: Tensor,  # velocity at c_{t-1}, same shape
) -> Tensor                   # (B,), values ≤ 0
```

---

### 3.5 Concordance $C_t$

**Definition**: rank-based fusion of the three estimators within
a rolling window of size $W=50$:

$$
C_t = \frac{1}{3}\!\left[
  \mathrm{rank}_W\!\left(\hat I^{(1)}(t)\right)
  + \mathrm{rank}_W\!\left(\hat I^{(2)}(t)\right)
  + \mathrm{rank}_W\!\left(\hat I^{(3)}(t)\right)
\right]
$$

where $\mathrm{rank}_W(x)$ is the percentile rank of $x$ within
the last $W$ values of that estimator's history. Rank-based fusion
makes $C_t \in [0,1]$ regardless of scale differences between
estimators, and a cliff is declared only when **all three agree**
(all ranks are high simultaneously).

**Why rank fusion instead of averaging raw values?**
$I^{(1)}$ (Bhattacharyya, range $[-1,0]$), $I^{(2)}$ (negative
variance, range $(-\infty, 0]$), and $I^{(3)}$ (negative velocity
jump, range $(-\infty, 0]$) have incompatible scales. Rank
normalisation is scale-free and outlier-robust.

**State management**: `compute_concordance_C` accepts an optional
`_state` dict that `PhaseQFlowPolicy` passes as `_concordance_state`
and clears in `reset()`.  Any subset of estimators is accepted
(graceful degradation when $I^{(2)}$ or $I^{(3)}$ are unavailable).

```python
# phase_centric/cliff_estimators.py  — IMPLEMENTED
def compute_concordance_C(
    i_hat_values: Sequence[Tensor],   # [I1, I2, I3] or any subset
    window_size: int = 50,
    _state: Optional[dict] = None,    # persistent per-buffer state
) -> Tensor                           # (B,), values in [0, 1]
```

**Config**: `pcar_input_signal = "concordance"` routes the PCAR
trigger through $C_t$.

---

### 3.6 PCAR — Predictability Cliff Adaptive Replanning

**Motivation**: open-loop chunk prediction drifts in long-horizon
tasks; replanning every step wastes compute and loses temporal
consistency. PCAR triggers replanning only when the cliff signal
exceeds a budget-adaptive threshold.

**Method**: budget-quantile trigger (§2.4, DKW guarantee):

$$
\tau^{\text{cp}}_n = \hat Q_n(1 - \epsilon),
\qquad
\text{replan}_t = \mathbb{1}[s_t > \tau^{\text{cp}}_n]
$$

where $s_t = C_t$ (or $\beta_t$ until concordance is
implemented). Warm start: static $\tau = 0.4$ for $n < 50$.

**Bayesian variant** (`BayesianPCARTrigger`): Beta-mixture
posterior $P(\text{changepoint} \mid s_{1:t})$ instead of the
empirical quantile; calibrated in stage 3 (`03_finetune_replan.yaml`).

**Implementation**

```python
# phase_centric/pcar_trigger.py
class PCARTrigger:
    history: Deque[float]              # maxlen=1000
    warmup_min: int = 50
    budget_eps: float = 0.1           # ε
    def update_and_check(signal: float) -> bool

class DualFlowHead(nn.Module):        # optional; splits pre/post-boundary

# phase_centric/b_pcar.py
class BayesianPCARTrigger:
    def update_and_check(signal: float) -> bool  # same interface

# PhaseQFlowConfig:
#   use_pcar                 = True
#   pcar_input_signal        = "concordance"   # or "beta" / "variance" / "curvature"
#   pcar_trigger_budget_eps  = 0.1
#   pcar_change_threshold    = 0.4             # warm-start fallback
```

**Verification**: `verify_pcar_budget.py` requires
$|\text{actual\_rate} - \epsilon| < 0.005$ for all
$\epsilon \in \{0.05, 0.1, 0.2, 0.3\}$.

**Prior work**: BID (arXiv 2408.17355) answers "which chunk to
execute"; PCAR answers "when to recompute". Adaptive Computation
Time (Graves, arXiv 1603.08983) uses a learned gate; ours uses an
empirical quantile with a DKW statistical guarantee.

---

### 3.7 Boundary-aware flow loss (PACE-A)

**Motivation**: flow-matching loss at boundary steps is measured
to be $\approx 4\times$ higher than at interior steps (§6 of the
paper). Uniform weighting lets the majority interior steps dominate
the gradient.

**Method**: weight each step by $w_t = 1 + \lambda\,\beta_t^{\text{micro}}$,
adding a Bernoulli entropy regulariser to prevent $\beta_t$
collapsing to 0:

$$
\mathcal{L}_{\text{boundary}}
  = \mathbb{E}_{t,x_t,d}\!\big[
      (1 + \lambda\,\beta_t)\,\|v_\theta(x_t,t,d\mid c)-v^*_t\|^2
    \big]
    - \eta\,\mathbb{E}_t\!\big[H(\beta_t)\big]
$$

$\lambda = 0.5$ (`boundary_reweight_lambda` in v2 configs),
$\eta = 0.01$. For the variational-bound interpretation see §2.3.

**Implementation**

```python
# phase_centric/boundary_aware_flow.py
def compute_boundary_aware_flow_loss(
    v_pred, v_target, beta_t,
    lambda_weight=0.5, entropy_weight=0.01,
) -> {"fm_loss", "entropy_reg", "total"}

def boundary_aware_reweight(beta_t, lambda_weight) -> Tensor  # per-step weights

# phase_centric/pace_a_loss.py   (legacy name; same semantics)
def compute_pace_a_flow_loss(v_pred, v_target, beta_t, ...) -> dict

# PhaseQFlowConfig:
#   use_pace_a              = True
#   use_boundary_reweight   = True
#   boundary_reweight_lambda = 0.5
#   pace_a_detach_beta      = True
```

**Verification**: `sanity_pace_a.py` requires boundary-step FM
loss reduced by $\ge 20\%$ vs `no_weight` mode, and
$\mathbb{E}[\beta] > 0.1$ (entropy regulariser is effective).

---

### 3.8 Summary

| Component | Feature gate | Status | Core formula |
| :-- | :-- | :--: | :-- |
| Phase InfoNCE | `use_chunk_infonce` | ✓ | $-\log\tfrac{\exp(s_{ii}/\tau)}{\sum_j\exp(s_{ij}/\tau)}$ |
| $\hat I^{(1)}$ Bhattacharyya | `use_phase_boundary_posterior` | ✓ | $-\beta_t = -(1-\sum_k\sqrt{\hat p_t\hat p_{t-1}})$ |
| $\hat I^{(2)}$ Variance | `pcar_input_signal="variance"` | ✓ | $-\sigma_t^2 = -\tfrac{1}{N}\sum\|a^{(i)}-\bar a\|^2$ |
| $\hat I^{(3)}$ Curvature | `pcar_input_signal="curvature"` | ✓ | $-\|v_\theta(c_t)-v_\theta(c_{t-1})\|^2$ |
| Concordance $C_t$ | `pcar_input_signal="concordance"` | ✓ | $\tfrac{1}{3}\sum_k\mathrm{rank}_W(\hat I^{(k)})$ |
| PCAR | `use_pcar` | ✓ | $\tau^{\text{cp}}=\hat Q_n(1-\epsilon)$ |
| Boundary-aware loss | `use_boundary_reweight` | ✓ | $(1+\lambda\beta_t)\|v-v^*\|^2 - \eta H(\beta)$ |

✓ implemented and tested

---

## 4 Ablation Matrix (v2, seven configs)

### 4.1 Matrix structure

Seven configs × 3 seeds = 21 runs, each with a unique combination
of cliff-detection signals and boundary reweighting. All configs
live under `configs/ablation/v2/` and inherit the full base
architecture from `configs/train/02_train_phase_and_flow.yaml`
(only the `use_*` flags differ).

| Config file | $\hat I^{(1)}$ | $\hat I^{(2)}$ | $\hat I^{(3)}$ | Concordance | Boundary reweight | Scientific purpose |
| :-- | :--: | :--: | :--: | :--: | :--: | :-- |
| `01_bc_chunked.yaml` | | | | | | Control: plain BC-Chunked, fixed-H replanning, no cliff detection |
| `02_cliff_via_beta_only.yaml` | ✓ | | | | | Marginal gain of $\hat I^{(1)}$ alone (PCAR via $\beta_t$) |
| `03_cliff_via_var_only.yaml` | | ✓ | | | | Marginal gain of $\hat I^{(2)}$ alone (PCAR via variance) |
| `04_cliff_via_curvature_only.yaml` | | | ✓ | | | Marginal gain of $\hat I^{(3)}$ alone (PCAR via curvature) |
| `05_cliff_concordance.yaml` | ✓ | ✓ | ✓ | ✓ | | Full concordance, no boundary reweight |
| `06_oracle_cliff.yaml` | — | — | — | — | — | Oracle gripper-flip signal: upper bound |
| `07_cliff_concordance_with_boundary_reweight.yaml` | ✓ | ✓ | ✓ | ✓ | ✓ | **Full PACE v2** (paper headline) |

**v2.0 runnable configs**: 01, 02, 06, 07 (I^(1) = Bhattacharyya β_t is fully implemented).

**Disabled configs (v2.0)**:

- **Config 03** (`cliff_via_var_only`): requires `bid_chunks` in the forward output
  (N≥2 BID samples). Setting `pcar_input_signal="variance"` invokes
  `PolicyVarianceEstimator.estimate()` anywhere in the training loop.
  Behavior: identical to config 01 (no cliff detection active).

- **Config 04** (`cliff_via_curvature_only`): `compute_I_hat_3` raises `NotImplementedError`.
  `VelocityCurvatureEstimator.update()` is implemented in the cliff_detection subpackage
  but not wired into `PhaseQFlowPolicy.forward()`.
  Behavior: identical to config 01 (no cliff detection active).

- **Config 05** (`cliff_concordance`): Concordance C_t requires all three estimators.
  With I^(2) and I^(3) unavailable, `pcar_input_signal="concordance"` in `PCARTrigger`
  falls back to beta_t. Behavior: nearly identical to config 02.

**Target for v2.1**: Wire `PolicyVarianceEstimator` and `VelocityCurvatureEstimator`
into `PhaseQFlowPolicy.forward()`, expose their outputs in `preds` dict,
and route through `PCARTrigger.update_and_check()`.

In the meantime the ablation dry-run pipeline (`--dry_run`) uses
synthetic data to verify that the configuration wiring and
statistical aggregation are correct end-to-end.

**Seed choice**: $\{42,\ 123,\ 2024\}$.

**Total compute**: GPU (RTX 5070) $\approx 20{,}000$ steps × 21
runs $\approx 7$ days. CPU dry-run: $\approx 1{-}2$ min.

### 4.2 Design principles

#### (a) Ceteris paribus

Apart from the feature gates, every hyperparameter (lr, batch,
optimizer, EMA, data augmentation, three-stage curriculum) is
identical across the seven configs. Each YAML overrides only the
`use_*` and `pcar_input_signal` fields; numerical hyper-parameters
are inherited from the base train config.

#### (b) Additive ablation ladder

The seven configs form an additive ladder:

```
01 (no cliff)
  → 02 (I^(1) only)
  → 03 (I^(2) only)          each single-estimator row measures
  → 04 (I^(3) only)          the marginal gain of that estimator alone
  → 05 (concordance, no reweight)
  → 07 (concordance + reweight)  = full PACE v2 headline
  → 06 (oracle)              upper bound; not a valid deployment config
```

Comparing 05 → 07 isolates the contribution of boundary-aware
flow loss. Comparing 02/03/04 → 05 measures the gain from
multi-estimator fusion vs. any single estimator.

#### (c) Train / inference separation

All cliff estimators and concordance fire at **inference** time
(they observe the current state without gradient). The boundary
reweight fires at **training** time only. PCAR fires at both
(training: budget calibration in stage 3; inference: actual
replanning). This separation means inference configs can be
swapped without retraining.

### 4.3 Statistical aggregation

`scripts/aggregate_ablation.py` aggregates per config × benchmark
(LIBERO-Long / LIBERO-Spatial) using rliable-style statistics:

| Field | Definition |
| :-- | :-- |
| IQM | Interquartile mean across seeds (rliable's recommended point estimate) |
| $\text{CI}_{95}$ | Stratified bootstrap 95% CI (2000 resamples) |
| Wilcoxon $p$ | Paired signed-rank test vs config 01 baseline |
| Cohen's $d$ | Effect size vs baseline |
| `placeholder` | `true` for synthetic dry-run data; `false` after GPU eval |

```bash
# Dry run (verifies pipeline; numbers are synthetic):
python scripts/aggregate_ablation.py --dry_run

# Real run (after GPU training, outputs in outputs/ablation_v2/):
python scripts/aggregate_ablation.py \
    --input_root outputs/ablation_v2 \
    --output paper_figures/ablation_v2/
```

Artifacts: `paper_figures/ablation_v2/ablation_table_v2.tex` and
`ablation_stats.csv`.

### 4.4 Falsification design: specificity test

If the cliff-detection gains were a generic regulariser, LIBERO-Spatial
(no consistent phase structure) should benefit as much as
LIBERO-Long. The specificity metric is:

$$
\Delta_{\text{specificity}}
  = \big[\text{SR}_{\text{long}}(07) - \text{SR}_{\text{long}}(01)\big]
    - \big[\text{SR}_{\text{spatial}}(07) - \text{SR}_{\text{spatial}}(01)\big]
$$

$\Delta_{\text{specificity}} > 0$ with $p < 0.05$ supports
the cliff-driven hypothesis; $\approx 0$ falsifies it.

### 4.5 SOTA reference points

Numbers reported in published papers on LIBERO-Long; included for
Related Work context only. These are **not reproduced by this
repo** — the checkpoint and evaluation protocol differences
between papers make direct comparison unreliable without a shared
evaluation harness.

| Method | Reported LIBERO-Long SR | Source |
| :-- | :--: | :-- |
| OpenVLA-OFT | 54.5% | arXiv 2502.19645 |
| $\pi_0$ | 60.0% | arXiv 2410.24164 |
| MoH | 57.8% | arXiv 2410.11842 |

The PACE v2 `full` config SR must come from
`paper_figures/ablation_v2/ablation_stats.csv` produced by a
completed GPU run; synthetic dry-run numbers (`placeholder=true`)
must not be cited as results.

### 4.6 Reproduction commands

```bash
# Pipeline integrity check (CPU, ~2 min, no checkpoint needed):
python scripts/aggregate_ablation.py --dry_run
bash scripts/smoke/smoke_phase_centric.sh

# Stage 2 training for one config (GPU required):
python scripts/training/train_dummy_batch.py \
    --config configs/ablation/v2/07_cliff_concordance_with_boundary_reweight.yaml \
    --total_steps 20000 --seed 42

# Aggregate results after all 21 runs:
python scripts/aggregate_ablation.py \
    --input_root outputs/ablation_v2 \
    --output paper_figures/ablation_v2/
```

---

## 5 Verification System

Every mathematical claim in §2–3 can be falsified in a CPU
sandbox without a dataset or GPU. All scripts live under
`scripts/verification/` and `scripts/smoke/`. Each returns a
clear PASS / FAIL verdict and takes under 2 minutes.

### 5.1 Script–assertion–acceptance table

| Script | What it checks | Acceptance criterion |
| :-- | :-- | :-- |
| `verify_identifiability.py` | Phase InfoNCE codebook alignment across 3 seeds | Hungarian permuted-agreement $\ge 0.7$ |
| `verify_phase_posterior.py` | $\hat I^{(1)}$ peaks align with ground-truth boundaries (50 demos) | peak-F1 $\ge 0.5$ at $\pm 3$-frame tolerance |
| `sanity_pace_a.py` | Boundary-aware flow loss reduces boundary-step FM loss | reduction $\ge 20\%$ vs `no_weight`; $\mathbb{E}[\beta] > 0.1$ |
| `verify_pcar_budget.py` | PCAR actual replan rate tracks budget $\epsilon$ (DKW bound) | $\|\text{rate} - \epsilon\| < 0.005$ for all $\epsilon \in \{0.05, 0.1, 0.2, 0.3\}$ |
| `sanity_pace_b.py` | *(optional)* PACE-B MoE gate switches meaningfully at boundaries | gate L2 $> 0.3$ boundary, $< 0.05$ interior |

`sanity_pace_b.py` tests the optional PACE-B MoE gate
(`use_pace_b=True`), which is disabled in all shipped configs.
It is included to verify the gating mechanism remains correct
even when not in the main ablation path.

### 5.2 Identifiability verification: Hungarian alignment

Phase codebooks trained across three seeds are unordered — seed
42's "code 3" may correspond to seed 123's "code 11." For each
pair $(i, j)$, build a $K \times K$ confusion matrix $C_{ij}$
and run the Kuhn–Munkres algorithm
(`scipy.optimize.linear_sum_assignment` on $-C$) to find the
best permutation $\pi_{ij}$:

$$
\text{PA}(i, j)
  = \frac{1}{N}\sum_{n=1}^{N}
    \mathbb{1}\!\left[\hat z^{(i)}_n = \pi_{ij}\!\left(\hat z^{(j)}_n\right)\right]
$$

Final score: $\text{mean}_{i<j}\,\text{PA}(i,j)$. Threshold $0.7$
is $175\times$ the random-assignment baseline ($1/K \approx 0.004$
for $K=240$); a trivially collapsed codebook also fails because
the InfoNCE loss is simultaneously very high.

### 5.3 Posterior verification: peak alignment

`verify_phase_posterior.py` generates ground-truth boundary
timestamps over 50 synthetic demos, finds local peaks of $\beta_t$
(strictly greater than both neighbours and $> \theta_{\text{peak}}$),
and matches against ground truth within $\pm 3$ frames.

Default $\theta_{\text{peak}} = 0.15$ (conservative — more false
positives, harder to achieve F1). After a GPU training run, pass
`--peak-threshold 0.5` for the production-ready threshold.

### 5.4 Boundary-aware flow loss sanity

`sanity_pace_a.py` synthesises three groups (boundary-heavy,
interior-flat, mixed) and trains a small velocity network for
100 steps with `full` vs `no_weight` modes. The acceptance check
measures:

$$
\text{reduction}
  = \frac{\text{FM}_{\text{no\_weight}} - \text{FM}_{\text{full}}}
         {\text{FM}_{\text{no\_weight}}}
  \ge 20\%
$$

The $20\%$ threshold is conservative for $\lambda = 0.5$
(`boundary_reweight_lambda` in v2 configs): $w_{\text{max}} =
1 + 0.5 = 1.5$, which accelerates boundary-step fitting by
$50\%$ in the best case; a $20\%$ net reduction over 100 steps
is comfortably reachable.

### 5.5 PCAR budget-quantile verification

`verify_pcar_budget.py` constructs a synthetic signal sequence:

$$
s_t = 0.1 \cdot B(2, 8) + 0.9 \cdot \sum_k \mathcal{N}(\mu_k, 0.02^2) \cdot \mathbb{1}[t \in W_k]
$$

where $B(2,8)$ is interior-step baseline noise and the Gaussian
pulses are transition events. The DKW worst-case bound for this
distribution is $\approx 0.043$ ($n=1000$, $\delta=0.05$); the
repo uses the tighter $0.005$ threshold because the synthetic
distribution is known and the quantile deviates well below the
DKW worst case. All four $\epsilon$ values must pass.

### 5.6 Coverage

| Assertion | CPU sandbox | Needs GPU |
| :-- | :--: | :--: |
| InfoNCE identifiability math | ✓ | |
| Bhattacharyya properties ($\beta \in [0,1]$, TV sandwich) | ✓ | |
| Boundary-aware FM loss reduction | ✓ | |
| PCAR DKW budget tracking | ✓ | |
| End-to-end SR gains ($\Delta$, specificity) | | ✓ |
| Latency / NFE cost | | ✓ |
| Cross-dataset generalisation | | ✓ |

### 5.7 One-liner self-check

```bash
pytest tests/ -v
bash scripts/smoke/smoke_phase_centric.sh
python scripts/verification/verify_identifiability.py
python scripts/verification/verify_phase_posterior.py
python scripts/verification/sanity_pace_a.py
python scripts/verification/verify_pcar_budget.py
```

Expected: `204 passed`; the 7-mode smoke all returns `[OK]`; the 5
verification scripts each return PASS (or identifiability
`WARN_DEGENERATE` when the CPU placeholder step count is too low).

---

## References

| Reference | Authors / Venue | arXiv / DOI |
| :-- | :-- | :-- |
| Flow Matching | Lipman et al. | arXiv 2210.02747 |
| Shortcut Models | Frans et al. | arXiv 2410.12557 |
| DiT / AdaLN-Zero | Peebles & Xie | arXiv 2212.09748 |
| FSQ | Mentzer et al. | arXiv 2309.15505 |
| iVAE | Khemakhem et al. NeurIPS 2020 | arXiv 1907.04809 |
| CPC / InfoNCE | van den Oord et al. | arXiv 1807.03748 |
| VQ-VAE | van den Oord et al. | arXiv 1711.00937 |
| Focal Loss | Lin et al. | arXiv 1708.02002 |
| BC-Z | Jang et al. | arXiv 2202.02005 |
| Switch Transformer | Fedus et al. | arXiv 2101.03961 |
| MoH | Zhang et al. | arXiv 2410.11842 |
| Curriculum Learning | Bengio et al. ICML 2009 | — |
| BID | Liu et al. | arXiv 2408.17355 |
| Adaptive Computation Time | Graves | arXiv 1603.08983 |
| DoRA | Liu et al. | arXiv 2402.09353 |
| IQL | Kostrikov et al. | arXiv 2110.06169 |
| ACT | Zhao et al. | arXiv 2304.13705 |
| A2C2 | (referenced) | arXiv 2509.23224 |
| OpenVLA-OFT | (referenced) | arXiv 2502.19645 |
| $\pi_0$ | (referenced) | arXiv 2410.24164 |




