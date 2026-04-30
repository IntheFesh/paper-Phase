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
- [§2 Phase-Centric Theory Framework](#2-phase-centric-theory-framework)
- [§3 Cliff Estimators, Concordance, and PCAR](#3-cliff-estimators-concordance-and-pcar)
- [§4 Ablation Matrix (v2, seven configs)](#4-ablation-matrix-v2-seven-configs)
- [§5 Verification System](#5-verification-system)

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
           │  I^(2)  =  -σ_t²    [PolicyVarianceEstimator]       │
           │            (anchor multi-sample; PENDING)           │
           │                                                     │
           │  I^(3)  =  -||v_θ(c_t) - v_θ(c_{t-1})||²            │
           │            (VelocityCurvatureEstimator; PENDING)    │
           │                                                     │
           │  Concordance  C_t  =  1/3 · Σ_k rank_W(I^(k))      │
           │                       (PENDING; rank-window W=50)   │
           │                                                     │
           │  PCAR trigger:  budget-quantile  τ_t = Q̂_n(1-ε)    │
           │                 replan ⇔ C_t > τ_t                  │
           │                 (or BayesianPCARTrigger Beta-mix)   │
           └─────────────────────────────────────────────────────┘
                                  │  replan? → re-encode obs and rerun L1-L3
                                  ▼
                          a_t ∈ R^{Da}   (executed)
```

**Implementation status caveat.** Of the three cliff estimators
only $I^{(1)}$ (Bhattacharyya $\beta_t$) is currently
implemented end-to-end. The public cliff-namespace functions
`compute_I_hat_2`, `compute_I_hat_3`, and `compute_concordance_C`
in
`lerobot_policy_phaseqflow/phase_centric/cliff_estimators.py`
raise `NotImplementedError`; their algorithmic details are
specified below in §3 but the multi-sample anchor mechanism for
$I^{(2)}$ and the velocity-anchor exposure for $I^{(3)}$ are
still open. Until those are landed, ablations 03/04/05/07 fall
back to whichever signal `pcar_input_signal` resolves to inside
`PCARTrigger` (defaulting to $\beta_t$).

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
| Cliff | $\sigma_t^2$ | $(B,)$ | per-step ensemble variance (PENDING) |
| Cliff | $\kappa_t$ | $(B,)$ | velocity-anchor $L^2$ jump (PENDING) |
| Cliff | $C_t$ | $(B,)$ | rank-window mean of $I^{(1/2/3)}$ (PENDING) |
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
def compute_I_hat_2(action_samples=None)              # PENDING  (NotImplementedError)
def compute_I_hat_3(v_theta_ct, v_theta_ct_prev)      # PENDING  (NotImplementedError)
def compute_concordance_C(i_hat_values, window_size)  # PENDING  (NotImplementedError)

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

### 3.3 $\hat{I}^{(2)}$ — Action-ensemble variance $\sigma_t^2$ (PENDING)

**Definition**:
$\hat I^{(2)}(t) = -\sigma_t^2$, where
$\sigma_t^2 = \frac{1}{N}\sum_{i=1}^{N}\|a_t^{(i)} - \bar a_t\|^2$
is the variance of $N$ independent flow samples conditioned on
the same observation.

**Rationale**: at phase boundaries the flow-matching score
function becomes multi-modal; independent samples spread out and
variance spikes, signalling a cliff even before the planner's
phase posterior shifts.

**Pending decision**: the multi-sample anchor needs $N \ge 2$
parallel flow rollouts per step. The interface (`pcar_input_signal
= "variance"`) is wired in `PCARTrigger.update_and_check`, but
`compute_I_hat_2` raises `NotImplementedError` until the
sampling strategy is resolved.

```python
# phase_centric/cliff_estimators.py
def compute_I_hat_2(action_samples: Tensor) -> Tensor
#   action_samples: (N, B, Ta, Da) — PENDING
```

---

### 3.4 $\hat{I}^{(3)}$ — Velocity-field curvature $\kappa_t$ (PENDING)

**Definition**:
$\hat I^{(3)}(t) =
-\|v_\theta(x_\tau, \tau, c_t) - v_\theta(x_\tau, \tau, c_{t-1})\|_2^2$,
where $x_\tau$ is a fixed anchor noise sample and $c_t, c_{t-1}$
are consecutive condition vectors.

**Rationale**: if the policy's velocity field is continuous at
interior steps but jumps at boundaries (because $c_t$ encodes a
different phase), then the $L^2$ difference between the velocity
field at a fixed anchor point detects the transition without
requiring an ensemble.

**Pending decision**: the anchor $(x_\tau, \tau)$ must be stable
across steps (e.g. fixed Gaussian noise drawn once per episode);
exposing the velocity head for two consecutive condition vectors
adds one extra forward pass per step.

```python
# phase_centric/cliff_estimators.py
def compute_I_hat_3(
    v_theta_ct: Tensor,       # velocity at c_t
    v_theta_ct_prev: Tensor,  # velocity at c_{t-1}
) -> Tensor                   # PENDING
```

---

### 3.5 Concordance $C_t$ (PENDING)

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

**Blocked by**: `compute_I_hat_2` and `compute_I_hat_3` (§3.3,
§3.4).

```python
# phase_centric/cliff_estimators.py
def compute_concordance_C(
    i_hat_values: Sequence[Tensor],   # [I1, I2, I3]
    window_size: int = 50,
) -> Tensor                            # PENDING
```

**Config**: `pcar_input_signal = "concordance"` (routes through
concordance once implemented; falls back to `"beta"` in the
meantime).

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
| $\hat I^{(2)}$ Variance | `pcar_input_signal="variance"` | ⏳ | $-\sigma_t^2 = -\tfrac{1}{N}\sum\|a^{(i)}-\bar a\|^2$ |
| $\hat I^{(3)}$ Curvature | `pcar_input_signal="curvature"` | ⏳ | $-\|v_\theta(c_t)-v_\theta(c_{t-1})\|^2$ |
| Concordance $C_t$ | `pcar_input_signal="concordance"` | ⏳ | $\tfrac{1}{3}\sum_k\mathrm{rank}_W(\hat I^{(k)})$ |
| PCAR | `use_pcar` | ✓ | $\tau^{\text{cp}}=\hat Q_n(1-\epsilon)$ |
| Boundary-aware loss | `use_boundary_reweight` | ✓ | $(1+\lambda\beta_t)\|v-v^*\|^2 - \eta H(\beta)$ |

✓ implemented and tested · ⏳ specified, interface wired, implementation pending

---

## 4 Ablation Matrix Design

### 4.1 Matrix structure

Twelve configs × 3 seeds = 36 runs, covering the six-dimensional
feature space from plain baseline to fully-on. Each config is
uniquely determined by a combination of feature-gate switches:

| Config | InfoNCE (I1) | $\beta_t$ (I2) | PACE-A (I3) | PACE-B (I4) | PACE-C (I5) | PCAR (I6) | Scientific purpose |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: | :-- |
| `baseline` | | | | | | | Control: plain BC-Chunked pipeline |
| `ident` | ✓ | | | | | | Measures the standalone contribution of InfoNCE identifiability |
| `a` | | ✓ | ✓ | | | | PACE-A alone (needs $\beta_t$ to back it) |
| `b` | | ✓ | | ✓ | | | PACE-B alone |
| `c` | | ✓ | | | ✓ | | PACE-C alone (curriculum doesn't depend on $\beta$, but I2 stays on for a consistent comparison) |
| `ab` | | ✓ | ✓ | ✓ | | | A×B interaction (does weighting + MoE resonate?) |
| `ac` | | ✓ | ✓ | | ✓ | | A×C interaction (weighting + curriculum) |
| `bc` | | ✓ | | ✓ | ✓ | | B×C interaction (MoE + curriculum) |
| `pace` | | ✓ | ✓ | ✓ | ✓ | | All training-time PACE, PCAR off |
| `pcar_only` | | ✓ | | | | ✓ | Standalone contribution of inference-time PCAR |
| `full` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Everything on (headline number in the paper) |
| `pcar_noident` | | ✓ | ✓ | ✓ | ✓ | ✓ | Full minus I1, checks whether identifiability is necessary |

**Seed choice**: $\{42,\ 123,\ 2024\}$ — three widely spaced seeds
without special meaning, to rule out seed cherry-picking.

**Total budget**: GPU (RTX 5070) $\approx 20{,}000$ steps × 36
runs $\approx 12$ days; CPU-sandbox placeholder proxy $3$ steps ×
36 runs $\approx 1\text{–}2$ min (for pipeline integrity and
artifact completeness checks).

### 4.2 Design principles

#### (a) Ceteris paribus

- Apart from the feature gates, every hyperparameter (lr, batch,
  optimizer, EMA, data augmentation, four-stage curriculum) is
  **identical** across the 12 configs.
- This is enforced by the `MODE_PRESETS` dict in
  `scripts/training/train_dummy_batch.py` — each config overrides
  only the `use_*` flags and never touches numerical hyperparams.

#### (b) Factorised rather than fully random

- Singles (`ident` / `a` / `b` / `c` / `pcar_only`): measure the
  **marginal contribution**
  $\Delta_i = \text{SR}(I_i) - \text{SR}(\text{baseline})$.
- Pairwise interactions (`ab` / `ac` / `bc`): measure the
  **interaction term**
  $\Delta_{ij} = \text{SR}(I_i I_j) - \Delta_i - \Delta_j - \text{SR}(\text{baseline})$;
  $>0$ is synergy, $<0$ is conflict.
- Pairing `full` with "remove one" (`pcar_noident`) directly
  measures $\Delta_{\text{InfoNCE}}$ in the presence of the other
  five — a check for whether identifiability is "only useful in
  simple configurations."

#### (c) Train/inference stratification

- I1–I5 fire at **training** time → configs `ident`, `a`, `b`,
  `c`, `ab`, `ac`, `bc`, `pace`.
- I6 (PCAR) fires at **inference** time → `pcar_only` takes its
  own slot, and `full` / `pcar_noident` measure combined effects.

### 4.3 Statistical aggregation

`scripts/paper/aggregate_ablation.py` aggregates per config ×
benchmark (LIBERO-10 Long-Horizon / LIBERO-Spatial):

| Field | Definition |
| :-- | :-- |
| $\bar{\text{SR}}$ | Mean success rate across 3 seeds |
| $\pm \text{CI}_{95}$ | Student-$t$ 95% confidence interval (dof $n-1=2$) |
| $p$-value | Two-tailed paired $t$-test vs baseline |
| $\Delta$ | $\bar{\text{SR}}_{\text{cfg}} - \bar{\text{SR}}_{\text{baseline}}$ |
| `placeholder_stats` | `true` for the CPU-sandbox dummy-batch; overwritten to `false` by a real GPU run |

Artifacts: `artifacts/ablation/stats.json` and
`artifacts/paper_stats.md` (a Markdown table, paper-ready).

### 4.4 Falsification design: the LIBERO-Spatial specificity test

**Logic**: if the Phase-Centric innovations were a generic
regulariser, they would also improve tasks that have **no obvious
phase structure**; if the gains really are "phase-driven," then
spatial generalisation (LIBERO-Spatial) should benefit
significantly less than long-horizon (LIBERO-10).

**Operationalisation**: roll out the same checkpoint on both
benchmarks:

$$
\Delta_{\text{specificity}}
  = \big[\text{SR}_{\text{long}}(\text{full}) - \text{SR}_{\text{long}}(\text{baseline})\big]
    - \big[\text{SR}_{\text{spatial}}(\text{full}) - \text{SR}_{\text{spatial}}(\text{baseline})\big]
$$

- $\Delta_{\text{specificity}} > 0$ with $p < 0.05$: supports the
  "phase-driven" hypothesis.
- $\Delta_{\text{specificity}} \approx 0$: the gain is a generic
  regulariser and the hypothesis is falsified.

**Expected value** (real GPU run): $\Delta_{\text{specificity}}$
around $+8$ to $+10$ pp (PACE-C on Spatial is projected to give
$\Delta_c \approx -1.8$ pp, i.e. a negative contribution, which is
exactly the counter-check that I5 relies on phase structure).

### 4.5 Comparison with SOTA (reference points, not a contribution of this repo)

Publicly reported LIBERO-10 numbers:

| Method | Reported SR | Source |
| :-- | :--: | :-- |
| OpenVLA-OFT | 54.5% | arXiv 2502.19645 |
| $\pi_0$ | 60.0% | arXiv 2410.24164 |
| MoH | 57.8% | arXiv 2410.11842 |

The three rows above are **reference points** for the paper's
Related Work only. The true LIBERO-10 SR of the repo's `full`
config has to be read from `artifacts/ablation/stats.json` after a
completed RTX 5070 run of `run_ablation.sh` +
`run_eval_libero.sh`; **CPU placeholder numbers must not stand in
for it** (that field carries `placeholder_stats=true`).

### 4.6 Reproduction commands

```bash
# Full matrix (real GPU run)
TOTAL_STEPS=20000 SEEDS="42 123 2024" DEVICE=cuda \
    bash scripts/training/run_ablation.sh

# Pipeline sanity (CPU, 1–2 min)
TOTAL_STEPS=3 SEEDS="42 123" DEVICE=cpu SKIP_EVAL=1 \
    bash scripts/training/run_ablation.sh
```

Matrix runs use `eval_done.marker` files for resumable
checkpointing; aggregation, figures, and the LaTeX table are
regenerated automatically at the tail of the script.

---

## 5 Verification System

The verification system sidesteps large-scale SR evaluation: the
core mathematical assertions of every Phase-Centric innovation can
be falsified inside a **CPU sandbox**. All scripts live under
`scripts/verification/`, and each one targets three properties:
fast ($\le 2$ min), falsifiable (PASS / FAIL verdict), and
statistically interpretable (carrying a $p$-value or confidence
interval).

### 5.1 Script–assertion–acceptance table

| Script | Target innovation | Statistical assertion | Expected output |
| :-- | :-- | :-- | :-- |
| `verify_identifiability.py` | I1 Chunk-NCE | Hungarian-aligned permuted-agreement across 3 seeds $\ge 0.7$ | verdict: PASS / WARN_DEGENERATE |
| `verify_phase_posterior.py` | I2 $\beta_t$ | $\beta_t$ peaks align with ground-truth boundaries on 50 demos (peak-F1 $\ge 0.5$) | verdict: PASS |
| `sanity_pace_a.py` | I3 PACE-A | Boundary-step FM loss reduced by $\ge 20\%$ vs `no_weight`, with $\mathbb{E}[\beta]>0.1$ | `passes_20pct_boundary_reduction` |
| `sanity_pace_b.py` | I4 PACE-B | Gate L2 $>0.3$ at boundary, $<0.05$ inside a phase | `passes_gate_magnitude_contrast` |
| `verify_pcar_budget.py` | I6 PCAR | $\forall \epsilon \in \{0.05, 0.1, 0.2, 0.3\}$: $\|\text{actual\_rate}-\epsilon\|<0.005$ | verdict: PASS |

(I5 PACE-C verification is folded into the curriculum-mask
assertion inside `train_dummy_batch.py`; its "boundary-count
filter" is deterministic logic and doesn't need a statistical
check.)

### 5.2 Identifiability verification: Hungarian alignment

**Problem**: phase codebooks trained on the same data across three
seeds are unordered — seed 42's "code 7" might correspond to seed
123's "code 12." Comparing argmax directly is unfair.

**Solution**: for each pair (seed $i$, seed $j$), build a
$K \times K$ confusion matrix $C_{ij}$ ($C_{ij}[a,b]$ = number of
samples that seed $i$ assigned to $a$ and seed $j$ assigned to
$b$), run the **Kuhn–Munkres (Hungarian) algorithm**
(`scipy.optimize.linear_sum_assignment` on $-C$ gives the maximum
matching) to find the best permutation $\pi_{ij}$, then compute
the aligned agreement rate:

$$
\text{PA}(i, j)
  = \frac{1}{N}\sum_{n=1}^{N}
    \mathbb{1}\!\left[\hat z^{(i)}_n = \pi_{ij}\big(\hat z^{(j)}_n\big)\right]
$$

The final permuted-agreement is
$\text{mean}_{i<j}\text{PA}(i,j)$.

**Why $0.7$**: under random assignment
$\mathbb{E}[\text{PA}] = 1/K$ ($\approx 0.004$ for $K=240$); a
collapse to a single code gives PA $=1$, but the InfoNCE loss
explodes at the same time (the "loss $<$ bound" side-check
prevents that trivial single-code pass). $0.7$ is a robust lower
bound for "three independent random processes reaching agreement";
typical values after 10k–20k GPU steps are $>0.8$.

### 5.3 Posterior verification: peak alignment and $\beta$-density threshold

**Operating logic**: `verify_phase_posterior.py` generates
ground-truth phase-boundary timestamps $t^{\text{gt}}_k$ over 50
synthetic demos, then finds local peaks of $\beta_t$
($\beta_t > \max(\beta_{t-1}, \beta_{t+1})$ and
$\beta_t > \theta_{\text{peak}}$) and matches them against the
ground truth with a $\pm 3$-frame tolerance window to compute
peak-F1.

**Threshold choice**: $\theta_{\text{peak}}$ has two settings:

| Threshold | Meaning | Applicable when |
| :-- | :-- | :-- |
| $0.15$ | "any visible distribution drift" | early training, when $\hat p_t$ is still not sharp |
| $0.50$ | "significant distribution change (below half BC)" | late training, around the downstream default $\tau^{\text{cp}}$ used by PCAR |

The repo's verification uses $0.15$ (the lower threshold is
stricter — more false positives, harder to pass F1); after a long
GPU run, switch to $0.50$ by passing `--peak-threshold 0.5` to
`verify_phase_posterior.py`.

**Why not BCE/CE**: the ground-truth boundaries are sparse
discrete events, and BCE has near-zero gradient at
$t \neq t^{\text{gt}}$, so it can't measure alignment quality.

### 5.4 PACE-A counter-example design

**Core design**: `sanity_pace_a.py` synthesises three groups:

1. **boundary-heavy**: $\beta_t$ constantly $1$ (pure
   transitions).
2. **interior-flat**: $\beta_t$ constantly $0$ (no transitions).
3. **mixed**: $20\%$ of steps at $1$, the rest at $0$ (close to a
   real scene).

Train $v_\theta$ to convergence on each group (100 steps is enough
for a small synthetic model) and compare the **boundary-step** FM
loss for `full` vs `no_weight`:

$$
\text{reduction}
  = \frac{\text{FM}_{\text{no\_weight}}(\text{boundary}) - \text{FM}_{\text{full}}(\text{boundary})}
         {\text{FM}_{\text{no\_weight}}(\text{boundary})}
$$

Require reduction $\ge 20\%$. That number matches the theoretical
ceiling for $\lambda=2.0$: the weight amplifies boundary-step loss
by up to $3\times$, which raises the equivalent step count from
$1$ to $3$ and speeds up boundary fitting by $\approx 3\times$,
translating into a roughly $25\%$ drop in final FM over 100
training steps. $20\%$ is the conservative threshold.

**`no_entropy` mode**: verifies the entropy regulariser is
necessary — without it, $\beta$ can drift toward $0.5$ (maximally
uncertain), weakening the weighting signal. The script records
`mean_beta` and requires $>0.1$ to rule out $\beta$ collapsing.

### 5.5 PACE-B gate-magnitude contrast

`sanity_pace_b.py` computes the gate update
$\|g_t - g_{t-1}\|_2$ separately on boundary and interior samples.
Theoretically $\alpha_t = \sigma(\kappa\beta-\mu)$ gives
$\alpha \approx 0.95$ at $\beta=1$ (big update) and
$\alpha \approx 0.12$ at $\beta=0$ (tiny update). The
$>0.3$ boundary / $<0.05$ interior thresholds confirm the gate
only switches meaningfully at transitions — otherwise the MoE
degenerates to every sample taking the same route.

### 5.6 PCAR budget-quantile verification

`verify_pcar_budget.py` constructs a synthetic $\beta$ sequence:

$$
\beta_t = 0.1 \cdot B(2, 8) + 0.9 \cdot \sum_k \mathcal{N}(\mu_k, 0.02^2)\cdot \mathbb{1}[t \in W_k]
$$

where $B(2,8)$ is the low-level baseline noise (simulating
interior steps) and the $\mathcal{N}$ pulses are the transition
events ($W_k$ are the boundary windows). On this distribution, the
theoretical upper bound on $|\text{diff}|$ required by DKW
convergence is $\approx 0.043$ ($n=1000$, $\delta=0.05$; see §2.4).
The script uses a stricter $0.005$ acceptance threshold — because
the distribution is synthetic and known, the quantile estimate
deviates less than the DKW worst case, so the tighter threshold is
reachable.

All four $\epsilon$ values must pass to return PASS, preventing a
single-point accuracy that is skewed overall.

### 5.7 Coverage and blind spots of the acceptance suite

| Assertion type | Covered by verification scripts | Still needs a real GPU run |
| :-- | :-- | :-- |
| Mathematical properties (identifiability, BC properties, DKW) | ✓ | — |
| Per-module engineering correctness (gate magnitude, weighted loss decrease) | ✓ | — |
| Combined gains ($\Delta$, $\Delta_{\text{specificity}}$) | ✗ | ✓ `run_eval_libero.sh` |
| Latency / throughput | ✗ | ✓ `benchmark_latency.py` |
| Cross-dataset robustness | ✗ | ✓ multi-task eval |

### 5.8 One-liner self-check

```bash
# 5 verification scripts + 77 unit tests + 7-mode E2E smoke, all CPU-friendly
pytest tests/ -v
bash   scripts/smoke/smoke_phase_centric.sh
python scripts/verification/verify_identifiability.py
python scripts/verification/verify_phase_posterior.py
python scripts/verification/verify_pcar_budget.py
python scripts/verification/sanity_pace_a.py
python scripts/verification/sanity_pace_b.py
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




