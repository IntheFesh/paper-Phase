# Project Abstract · Phase-Centric Vision-Language-Action Policy

> This document is a technical summary aimed at readers who want a quick overview
> of the method. For full implementation details, see the module sources and the
> derivation notes under `docs/innovations/`.

## Background

In long-horizon robotic manipulation, a policy has to span several sub-goals
(approach, grasp, transport, place, release). Existing methods usually treat
phase as a passive feature, which leads to code-book collapse, underfitting on
boundary steps, and replanning thresholds that don't transfer across tasks.
This work builds a systematic method around *phase* as a first-class control
variable, hosted inside the [LeRobot](https://huggingface.co/docs/lerobot/index)
plugin architecture and evaluated on
[LIBERO](https://libero-project.github.io/).

## Method

We propose the **PhaseQFlow++** four-layer policy: a dual visual backbone
(SigLIP2 + DINOv2, LoRA/DoRA) is fused via Prismatic channel concatenation and
FiLM language modulation, then read out by a Perceiver into 8 queries. The
`HierarchicalPlanner` discretises phase with FSQ (levels=[8,6,5], $K=240$) and
jointly emits a continuous skill latent. `ShortcutFlowActionHead` decodes
$16 \times 16$ action chunks with an AdaLN-Zero DiT under 1-NFE conditional
flow. `IQLChunkVerifier` fits $V_\psi$ with expectile $\tau=0.8$ and $Q_\theta$
with TD(0), and reports chunk confidence as $\sigma(\beta_c(Q-V))$.

On top of this we add six phase-centric innovations. The **phase posterior**
uses a first-order Bhattacharyya distance:

$$
\hat p_t = \alpha p_t + (1-\alpha)\hat p_{t-1},\;\alpha=0.3;\quad
\beta_t = 1 - \sum_k \sqrt{\hat p_t(k)\,\hat p_{t-1}(k)}
$$

The Cauchy–Schwarz upper bound and the Hellinger–TV inequality guarantee
$\beta_t \in [0,1]$ and $\tfrac12\beta_t \le d_{\text{TV}} \le \sqrt{\beta_t}$.
**Chunk-level InfoNCE** runs an in-batch contrast between $(o,a)$ pairs and
phase prototypes to make phase codes identifiable across seeds (iVAE,
arXiv 1907.04809). **PACE-A** plugs $\beta_t$ in as a dynamic weight on the
Flow Matching objective:

$$
\mathcal{L}_{\text{PACE-A}}
  = \mathbb{E}\!\big[(1 + \lambda\beta_t)\,\|v_\theta - v^*\|^2\big]
    - \eta\,\mathbb{E}[H(\beta_t)],
  \quad \lambda=2.0,\;\eta=0.01
$$

which can be read as the variational lower bound of a boundary-prior mixture
likelihood. **PACE-B** routes phase experts through a smooth gate
$\alpha_t=\sigma(\kappa\beta_t-\mu)$ with $\kappa=5.0,\mu=2.0$. **PACE-C**
exposes samples in three stages ordered by episode boundary density. **PCAR**
takes a quantile of the rolling empirical distribution as an adaptive replan
threshold:

$$
\tau^{\text{cp}}_n = \hat Q_n(1-\epsilon),\quad
\text{replan}_t = \mathbb{1}[\beta_t > \tau^{\text{cp}}_n]
$$

The DKW inequality gives
$|\Pr(\beta_t > \tau^{\text{cp}}_n)-\epsilon|
\le \sqrt{\log(2/\delta)/(2n)}$; with $n=1000,\delta=0.05$ the deviation bound
is about $0.043$. All six innovations are gated by explicit feature flags and
form a factorised ablation matrix of $12$ configs $\times\ 3$ seeds.

## Results

**Theory (5 reproducible verification scripts):** after Hungarian-matching the
codes, Chunk-InfoNCE hits permuted-agreement $\ge 0.7$ across 3 seeds (GPU
long-training expected to exceed $0.8$); peaks of $\beta_t$ match the true
boundaries with peak-F1 $\ge 0.5$; PACE-A drops boundary-step FM loss by at
least $20\%$ relative to the equal-weight baseline; PACE-B boundary gates are
L2 $>0.3$ while interior gates stay $<0.05$; PCAR holds $|\text{diff}|<0.005$
on $\epsilon\in\{0.05,0.1,0.2,0.3\}$. **Engineering:** 77 pytest unit tests,
7-mode end-to-end smoke, bf16 + PagedAdamW8bit + EMA training stack.
**System metrics** (LIBERO-10 long-horizon, LIBERO-Spatial generalisation
control) are aggregated by `scripts/training/run_ablation.sh` into
`artifacts/ablation/stats.json` (3-seed mean, 95% CI, paired-$t$ two-tailed
$p$-value). The ablation numbers shipped with this repo come from the CPU
dummy-batch proxy with `placeholder_stats=true`; large-scale empirical
validation still requires running the full training and evaluation pipeline
on GPUs.

## Why this matters

The contribution is to inject phase structure into the policy's representation,
loss, and online decision layers *at once*, with a single signal $\beta_t$
threading through all three. The six innovations are mutually orthogonal —
I3–I5 act during training, I6 acts at inference, I1–I2 are the shared
infrastructure — so each can be measured in isolation. The method composes
cleanly with modern VLA building blocks (Flow Matching, Shortcut Models, DiT,
IQL, DoRA, BID, ACT), and it gives a mathematically characterisable,
engineering-reproducible frame for answering "when and how to reorganise
policy computation" in long-horizon manipulation.

---

## Entry points

- Full architecture and theory: [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
- Operations and reproduction: [`docs/OPERATIONS_GUIDE.md`](OPERATIONS_GUIDE.md)
- Per-innovation index: [`docs/innovations/INDEX.md`](innovations/INDEX.md)
