#!/usr/bin/env python

"""Phase Identifiability verification via cross-seed permuted agreement.

Rationale
---------
Following iVAE (Khemakhem et al., NeurIPS 2020) and the nonlinear-ICA result
of Hyvarinen et al. (2024), once an auxiliary variable ``u`` is introduced the
latent ``z`` is identifiable up to permutation plus per-axis rescaling. This
script takes "do phase latents trained under different seeds agree under a
permutation" as the empirical test of identifiability.

Design
------
1. Train the same config (``ident_only`` mode) under 3 different random seeds
   for ``--steps`` steps (defaults: 2000 for RTX 5070; 50 for the CPU smoke).
2. Training data comes from ``SyntheticLongHorizonEnv`` (the 3-waypoint 2D
   navigation env with ground-truth phase boundaries shipped in Round 1); its
   ``waypoint_idx`` is used as ``z_gt``.
3. On a fixed set of validation samples ``(obs, action_chunk)``, collect each
   seed's phase_id sequence ``z^(1), z^(2), z^(3)`` (the planner's
   ``argmax(phase_logits)``).
4. For every pair ``(i, j)``, run the Hungarian algorithm
   (``scipy.optimize.linear_sum_assignment``) to find the permutation
   ``sigma`` that maximises ``(sigma(z^(i)) == z^(j)).mean()``.
5. Record permuted agreement per pair and each seed's agreement vs ground
   truth; write ``artifacts/identifiability/report.md``, ``report.json``, and
   ``figures/identifiability_confusion.png`` (3 x 3 confusion matrices).

**Acceptance**: PASS iff every pair has permuted agreement >= 0.7; FAIL if
any pair is below 0.7.

Run
---
    python scripts/verification/verify_identifiability.py --steps 50 --seeds 1,2,3 --device cpu
    python scripts/verification/verify_identifiability.py --steps 2000 --seeds 42,43,44 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PKG_SRC = _REPO_ROOT / "lerobot_policy_phaseqflow" / "src"
_SCRIPTS = _REPO_ROOT / "scripts"
_DIAGNOSTICS_DIR = _REPO_ROOT / "scripts" / "diagnostics"
for p in (_PKG_SRC, _SCRIPTS, _DIAGNOSTICS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig # noqa: E402
from lerobot_policy_phaseqflow.modeling_phaseqflow import PhaseQFlowPolicy # noqa: E402

from diagnostic_utils.synthetic_demos import make_synthetic_demos # noqa: E402
from diagnostic_utils.synthetic_env import SyntheticLongHorizonEnv # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("verify_identifiability")


def _build_config() -> PhaseQFlowConfig:
    """Build the small CPU-friendly config shared by training and inference.

    Turns off every Round 4+ toggle; only ``use_chunk_infonce`` is on
    (``ident_only`` mode).
    """
    return PhaseQFlowConfig(
        use_dual_backbone_vision=False,
        use_fsq=False,
        use_bid_sampling=False,
        use_temporal_ensembling=False,
        use_correction_head=False,
        use_ema=False,
        use_bf16=False,
        use_gradient_checkpointing=False,
        use_paged_adamw_8bit=False,
        use_chunk_infonce=True,
        chunk_infonce_weight=1.0,
        chunk_infonce_temperature=0.1,
        use_phase_boundary_posterior=False,
        use_pace_a=False, use_pace_b=False, use_pace_c=False, use_pcar=False,
        action_dim=7,
        state_dim=8,
        history_dim=8,
        fusion_hidden_dim=64,
        vision_token_dim=64,
        state_token_dim=64,
        language_token_dim=64,
        history_token_dim=64,
        cross_attn_heads=4,
        num_skills=8,
        skill_embedding_dim=16,
        continuous_skill_dim=16,
        latent_dim=16,
        dit_hidden_dim=64,
        dit_num_layers=2,
        dit_num_heads=4,
        critic_hidden_dim=64,
        flow_steps=2,
        verifier_hidden_dim=32,
        max_timestep=128,
        action_chunk_size=8,
        action_execute_size=4,
    )


def _demo_phase_gt(demo: Any, t: int, n_phases: int = 3) -> int:
    """Recover the ground-truth phase id from ``DemoSample.states[t, 2]``.

    ``make_synthetic_demos`` stores ``states[t, 2] = (pid + 1) / n_phases``;
    this inverts the mapping via ``pid = round(states[t, 2] * n_phases) - 1``
    and clamps to ``[0, n_phases)``.
    """
    progress = float(demo.states[t, 2])
    pid = int(round(progress * n_phases)) - 1
    return max(0, min(n_phases - 1, pid))


def _sample_training_batch(
    demos: List[Any],
    batch_size: int,
    action_chunk_size: int,
    cfg: PhaseQFlowConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, torch.Tensor]:
    """Sample a random ``(obs, action_chunk)`` training batch from ``DemoSample`` list.

    Each sample:
      - Pick a random demo and a random start ``t in [0, len(demo) - Ta)``.
      - obs: the demo's images / states / history at time ``t``; the language
        tensor is zero-filled (``make_synthetic_demos`` only stores the
        instruction string, not a language tensor).
      - action_chunk: ``demo.actions[t : t + Ta]`` with shape ``(Ta, Da)``.
      - ``_phase_gt``: ground-truth waypoint idx recovered from
        ``states[t, 2]``.
    """
    Ta = action_chunk_size
    Da = cfg.action_dim
    lang_dim = 16
    imgs, states, hists, chunks, phase_ids = [], [], [], [], []
    for _ in range(batch_size):
        idx = int(rng.integers(0, len(demos)))
        demo = demos[idx]
        T = len(demo)
        if T < Ta + 1:
            continue
        t = int(rng.integers(0, T - Ta))
        imgs.append(demo.images[t, 0])
        states.append(demo.states[t])
        hists.append(demo.history[t])
        chunks.append(demo.actions[t : t + Ta])
        phase_ids.append(_demo_phase_gt(demo, t))
    if not imgs:
        raise RuntimeError("All demos shorter than action_chunk_size; cannot batch.")
    B = len(imgs)
    batch = {
        "obs": {
            "images": torch.as_tensor(np.stack(imgs)).float().to(device),
            "states": torch.as_tensor(np.stack(states)).float().to(device),
            "language": torch.zeros(B, lang_dim, device=device),
            "history": torch.as_tensor(np.stack(hists)).float().to(device),
        },
        "action": torch.as_tensor(np.stack(chunks)).float().to(device)[:, :Ta, :Da],
        "timestep": torch.zeros(B, dtype=torch.long, device=device),
        "_phase_gt": torch.as_tensor(phase_ids, dtype=torch.long, device=device),
    }
    return batch


def _train_one_seed(
    seed: int,
    demos: List[Dict[str, np.ndarray]],
    steps: int,
    device: torch.device,
    micro_batch: int,
    cfg: PhaseQFlowConfig,
) -> PhaseQFlowPolicy:
    """Train a ``PhaseQFlowPolicy`` in ``ident_only`` mode under a given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    policy = PhaseQFlowPolicy(cfg).to(device).train()

    warm = _sample_training_batch(demos, micro_batch, cfg.action_chunk_size, cfg, device, rng)
    warm.pop("_phase_gt", None)
    _ = policy.compute_loss(warm)

    optim = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=float(cfg.lr_head),
        weight_decay=float(cfg.weight_decay),
    )
    t0 = time.perf_counter()
    for step in range(int(steps)):
        optim.zero_grad(set_to_none=True)
        batch = _sample_training_batch(demos, micro_batch, cfg.action_chunk_size, cfg, device, rng)
        batch.pop("_phase_gt", None)
        out = policy.compute_loss(batch, return_dict=True)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad and p.grad is not None],
            max_norm=float(cfg.grad_clip_norm),
        )
        optim.step()
        if (step + 1) % max(1, steps // 10) == 0 or step == 0:
            diag = getattr(policy, "_last_chunk_infonce_diag", {})
            comps = getattr(policy, "_last_loss_components", {})
            ci = comps.get("chunk_infonce", torch.zeros(())).item() if comps else 0.0
            acc = float(diag.get("info_nce_acc", 0.0))
            ent = float(diag.get("phase_entropy", 0.0))
            log.info(
                "seed=%d step %d/%d loss=%.4f infonce=%.4f acc=%.3f H(phase)=%.3f",
                seed, step + 1, steps, float(loss.detach()), ci, acc, ent,
            )
    log.info("seed=%d done in %.1fs", seed, time.perf_counter() - t0)
    return policy.eval()


def _build_validation_set(
    demos: List[Dict[str, np.ndarray]],
    cfg: PhaseQFlowConfig,
    device: torch.device,
    num_samples: int,
    seed: int = 12345,
) -> Dict[str, torch.Tensor]:
    """Build a fixed validation batch shared across seeds for phase_id extraction."""
    rng = np.random.default_rng(seed)
    batch = _sample_training_batch(demos, num_samples, cfg.action_chunk_size, cfg, device, rng)
    return batch


@torch.no_grad()
def _predict_phase_ids(policy: PhaseQFlowPolicy, batch: Dict[str, torch.Tensor]) -> np.ndarray:
    """Take planner ``argmax(phase_logits)`` as ``z`` on the fixed validation batch."""
    local = dict(batch)
    local.pop("_phase_gt", None)
    preds = policy.predict_action(local)
    return preds["phase_logits"].argmax(dim=-1).detach().cpu().numpy()


def _permuted_agreement(z_a: np.ndarray, z_b: np.ndarray, K: int) -> Tuple[float, np.ndarray]:
    """Find the permutation sigma that maximises ``(sigma(z_a) == z_b).mean()`` via Hungarian.

    Returns
    -------
    agreement : float
        Accuracy under the best permutation.
    perm : (K,) int array
        The permutation sigma applied to ``z_a``; ``sigma[i]`` is the id in
        ``z_b`` that corresponds to id ``i`` in ``z_a``.

    Implementation: build the co-occurrence matrix
    ``C[i, j] = #{(z_a == i) & (z_b == j)}`` and solve the minimum-cost
    assignment on ``-C`` (equivalent to maximising agreement).
    """
    from scipy.optimize import linear_sum_assignment

    n = max(len(z_a), 1)
    C = np.zeros((K, K), dtype=np.int64)
    for a, b in zip(z_a, z_b):
        C[int(a), int(b)] += 1
    rows, cols = linear_sum_assignment(-C)
    perm = np.zeros(K, dtype=np.int64)
    for r, c in zip(rows, cols):
        perm[r] = c
    mapped = perm[z_a]
    agreement = float((mapped == z_b).mean()) if n > 0 else 0.0
    return agreement, perm


def _plot_confusion(
    z_list: List[np.ndarray],
    seeds: List[int],
    K: int,
    out_path: Path,
) -> None:
    """Render an ``N x N`` grid of per-seed-pair confusion matrices."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = len(seeds)
    fig, axes = plt.subplots(N, N, figsize=(4 * N, 4 * N))
    if N == 1:
        axes = np.array([[axes]])
    for i in range(N):
        for j in range(N):
            ax = axes[i, j]
            C = np.zeros((K, K), dtype=np.int64)
            for a, b in zip(z_list[i], z_list[j]):
                C[int(a), int(b)] += 1
            ax.imshow(C, cmap="Blues")
            ax.set_title(f"seed {seeds[i]} -> seed {seeds[j]}")
            ax.set_xlabel(f"z^({seeds[j]})")
            ax.set_ylabel(f"z^({seeds[i]})")
    fig.suptitle("Cross-seed phase-id confusion (before permutation)", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    """Run the cross-seed identifiability verification pipeline."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=2000,
                        help="Training steps per seed. Use 50 for the CPU smoke.")
    parser.add_argument("--seeds", type=str, default="42,43,44",
                        help="Comma-separated list of training seeds.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_demos", type=int, default=32,
                        help="Synthetic demo count; caps data diversity.")
    parser.add_argument("--num_val_samples", type=int, default=256,
                        help="Validation samples used for cross-seed comparison.")
    parser.add_argument("--micro_batch", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="artifacts/identifiability")
    parser.add_argument("--agreement_threshold", type=float, default=0.7)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        raise ValueError(f"Need at least 2 seeds for pairwise comparison; got {seeds}.")

    cfg = _build_config()
    log.info("config: num_skills=%d (=> K=%d), chunk_size=%d, chunk_infonce_weight=%.2f",
             cfg.num_skills, cfg.num_skills, cfg.action_chunk_size, cfg.chunk_infonce_weight)

    demos = make_synthetic_demos(
        num_demos=int(args.num_demos),
        action_dim=cfg.action_dim,
        state_dim=cfg.state_dim,
        history_dim=cfg.history_dim,
        episode_len_range=(48, 72),
        num_phases_range=(3, 3),
        seed=0,
    )
    log.info("built %d synthetic demos (T in [48, 72], 3 phases each).",
             len(demos))

    policies: List[PhaseQFlowPolicy] = []
    for seed in seeds:
        pol = _train_one_seed(
            seed=seed, demos=demos, steps=int(args.steps),
            device=device, micro_batch=int(args.micro_batch), cfg=cfg,
        )
        policies.append(pol)

    val_batch = _build_validation_set(
        demos=demos, cfg=cfg, device=device,
        num_samples=int(args.num_val_samples), seed=12345,
    )
    z_list = [_predict_phase_ids(p, val_batch) for p in policies]
    z_gt = val_batch["_phase_gt"].detach().cpu().numpy()
    log.info("z^(i) shapes: %s; gt shape: %s", [z.shape for z in z_list], z_gt.shape)

    pair_agreements: Dict[str, float] = {}
    pair_perms: Dict[str, List[int]] = {}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            ag, perm = _permuted_agreement(z_list[i], z_list[j], K=cfg.num_skills)
            key = f"seed{seeds[i]}_vs_seed{seeds[j]}"
            pair_agreements[key] = ag
            pair_perms[key] = perm.tolist()
            log.info("[%s] permuted agreement = %.3f", key, ag)

    gt_agreements: Dict[str, float] = {}
    for s, z in zip(seeds, z_list):
        ag, _ = _permuted_agreement(z, z_gt, K=max(cfg.num_skills, int(z_gt.max()) + 1))
        gt_agreements[f"seed{s}_vs_gt"] = ag
        log.info("seed=%d vs ground-truth: permuted agreement = %.3f", s, ag)

    min_pair = min(pair_agreements.values()) if pair_agreements else 0.0

    uniq_per_seed = {seeds[i]: int(np.unique(z_list[i]).size) for i in range(len(seeds))}
    degenerate = any(u < 2 for u in uniq_per_seed.values())
    if degenerate:
        verdict = "WARN_DEGENERATE"
    elif min_pair >= args.agreement_threshold:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    out_dir = _REPO_ROOT / args.output_dir
    figs_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    _plot_confusion(z_list, seeds, cfg.num_skills, figs_dir / "identifiability_confusion.png")

    payload = {
        "seeds": seeds,
        "steps": int(args.steps),
        "num_demos": int(args.num_demos),
        "num_val_samples": int(args.num_val_samples),
        "config": {k: v for k, v in asdict(cfg).items() if k.startswith(("use_", "chunk_infonce_", "num_skills"))},
        "pair_agreements": pair_agreements,
        "pair_permutations": pair_perms,
        "gt_agreements": gt_agreements,
        "unique_phase_ids_per_seed": uniq_per_seed,
        "threshold": float(args.agreement_threshold),
        "min_pair_agreement": float(min_pair),
        "verdict": verdict,
    }
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2))

    md = [
        "# Round 3 - Phase Identifiability Verification Report",
        "",
        f"**Verdict**: `{verdict}` (threshold {args.agreement_threshold:.2f}, "
        f"min pair agreement {min_pair:.3f})",
        "",
    ]
    if degenerate:
        md += [
            "> **WARN_DEGENERATE**: at least one seed produced a single unique "
            "phase id on the validation set (codebook collapse). Pairwise "
            "agreement is trivially 1.0 in that case and does NOT imply "
            "identifiability.",
            f"> Unique phases per seed: `{uniq_per_seed}`.",
            "> Suggested fixes: increase `--steps`, raise `chunk_infonce_weight`, "
            "lower `gumbel_temperature`, or switch planner to FSQ (`use_fsq=True`).",
            "",
        ]
    md += [
        "## Pairwise permuted agreement (Hungarian)",
        "",
        "| Pair | Agreement |",
        "|---|---|",
    ]
    for k, v in pair_agreements.items():
        md.append(f"| {k} | {v:.3f} |")
    md += [
        "",
        "## vs ground-truth (waypoint_idx)",
        "",
        "| Seed | Agreement |",
        "|---|---|",
    ]
    for k, v in gt_agreements.items():
        md.append(f"| {k} | {v:.3f} |")
    md += [
        "",
        "## Artifacts",
        "",
        f"- `figures/identifiability_confusion.png` - {len(seeds)} x {len(seeds)} "
        "confusion matrices (before permutation).",
        f"- `report.json` - full numeric payload incl. permutations and config snapshot.",
    ]
    (out_dir / "report.md").write_text("\n".join(md) + "\n")

    log.info("report written to %s (verdict=%s)", out_dir, verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
