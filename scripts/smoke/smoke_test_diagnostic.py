#!/usr/bin/env python

"""Smoke test for the Round 1 diagnostic pipeline.

Verifies in a CPU-only, no-network environment that the helper modules
used by ``scripts/diagnostics/diagnostic_phase_centric.py`` run end to
end. Coverage:

    1. SyntheticLongHorizonEnv: reset / step / terminal detection.
    2. Phase proxies: correctness of the gripper / velocity_change /
       planner_output masks.
    3. ``h1_loss.per_timestep_flow_loss`` runs on a small-config
       PhaseQFlowPolicy and returns a loss array of shape
       ``(T - chunk + 1,)``.
    4. ``h2_rollout.rollout_episode`` runs a small policy in the
       synthetic env.
    5. Report writers: ``report.json`` / ``report.md`` / ``fig_*.png``
       land on disk.

Note: this smoke does not load a real checkpoint; the policy is
random-init, so the H1/H2 numbers have no scientific meaning and only
verify that the pipeline does not break.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DIAGNOSTICS_DIR = _REPO_ROOT / "scripts" / "diagnostics"
if str(_DIAGNOSTICS_DIR) not in sys.path:
    sys.path.insert(0, str(_DIAGNOSTICS_DIR))

from diagnostic_utils.h1_loss import per_timestep_flow_loss # noqa: E402
from diagnostic_utils.h2_rollout import rollout_episode # noqa: E402
from diagnostic_utils.phase_proxies import ( # noqa: E402
    compute_boundary_mask,
    gripper_boundary_mask,
    velocity_change_boundary_mask,
)
from diagnostic_utils.report import ( # noqa: E402
    build_report_payload,
    save_h1_figure,
    save_h2_figure,
    verdict_h1,
    verdict_h2,
    write_json_report,
    write_markdown_report,
)
from diagnostic_utils.synthetic_demos import make_synthetic_demos # noqa: E402
from diagnostic_utils.synthetic_env import SyntheticLongHorizonEnv # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("diagnostic_smoke")


def _make_small_policy() -> Any:
    """Build a legacy-tokenizer PhaseQFlowPolicy with tiny dims.

    Uses ``use_dual_backbone_vision=False`` / ``use_fsq=False`` to
    sidestep ``timm`` / ``transformers`` /
    ``vector_quantize_pytorch`` dependencies.
    """
    from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig
    from lerobot_policy_phaseqflow.modeling_phaseqflow import PhaseQFlowPolicy

    cfg = PhaseQFlowConfig(
        use_dual_backbone_vision=False,
        use_fsq=False,
        use_bid_sampling=False,
        use_temporal_ensembling=False,
        use_correction_head=False,
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
        max_timestep=64,
        action_chunk_size=8,
        action_execute_size=4,
    )
    return PhaseQFlowPolicy(cfg).eval()


def smoke_synthetic_env() -> None:
    """Check that SyntheticLongHorizonEnv honours the Gym-like contract."""
    print("\n[1] SyntheticLongHorizonEnv: reset / step / terminal")
    env = SyntheticLongHorizonEnv(action_dim=7, state_dim=8, history_dim=8, seed=0)
    obs = env.reset()
    assert set(obs.keys()) == {"images", "states", "language", "history", "masks"}
    assert obs["images"].shape == (1, 3, 64, 64)
    assert obs["states"].shape == (8,)

    for _ in range(5):
        obs, r, done, info = env.step(np.zeros(7, dtype=np.float32))
        assert isinstance(info, dict)
    print(" OK (obs keys, shapes, step contract satisfied)")


def smoke_phase_proxies() -> None:
    """Check the gripper and velocity_change boundary-mask proxies."""
    print("\n[2] Phase proxies: gripper / velocity_change")
    acts = np.zeros((10, 7), dtype=np.float32)
    acts[5:, -1] = 1.0
    mask = gripper_boundary_mask(acts, dilate=2)
    assert mask.shape == (10,)
    assert mask[5] and mask[3] and mask[7] and not mask[0]
    ee = np.zeros((10, 3), dtype=np.float32)
    ee[:5] = np.arange(5)[:, None] * np.array([1, 0, 0], dtype=np.float32)
    ee[5:] = ee[4] + (np.arange(5)[:, None] * np.array([-1, 0, 0], dtype=np.float32))
    vm = velocity_change_boundary_mask(ee, dilate=1)
    assert vm.shape == (10,)
    assert vm.any()
    m1 = compute_boundary_mask("gripper", actions=acts)
    m2 = compute_boundary_mask("velocity_change", ee_positions=ee)
    assert m1.shape == m2.shape == (10,)
    print(" OK (gripper and velocity_change masks correct)")


def smoke_h1_pipeline() -> None:
    """Check that ``per_timestep_flow_loss`` returns the expected loss vector."""
    print("\n[3] h1_loss.per_timestep_flow_loss: returns per-t loss vector")
    policy = _make_small_policy()
    demos = make_synthetic_demos(
        num_demos=2, action_dim=7, state_dim=8, history_dim=8,
        episode_len_range=(30, 40), num_phases_range=(2, 3),
    )
    losses = per_timestep_flow_loss(
        policy=policy, demo=demos[0], chunk_len=8, num_samples=2,
        device=torch.device("cpu"),
    )
    assert losses is not None
    expected_T = len(demos[0]) - 8 + 1
    assert losses.shape == (expected_T,), (losses.shape, expected_T)
    assert np.isfinite(losses).all(), "FM loss contains NaN/Inf"
    print(f" OK (loss shape {losses.shape}, mean={float(losses.mean()):.4f})")


def smoke_h2_pipeline() -> None:
    """Check that ``rollout_episode`` records replan_times and phase_ids."""
    print("\n[4] h2_rollout.rollout_episode: records replan_times / phase_ids")
    policy = _make_small_policy()
    env = SyntheticLongHorizonEnv(action_dim=7, state_dim=8, history_dim=8,
                                  max_steps=40, seed=0)
    result = rollout_episode(policy=policy, env=env, max_steps=40)
    assert "replan_times" in result and "phase_ids" in result
    assert result["total_steps"] > 0
    assert result["total_steps"] == len(result["phase_ids"])
    assert result["replan_times"][0] == 0, "first chunk must be predicted at t=0"
    assert isinstance(result["misalignment"], float)
    print(f" OK (steps={result['total_steps']} replans={len(result['replan_times'])} "
          f"success={result['success']} mis={result['misalignment']:.3f})")


def smoke_report_writers() -> None:
    """Check that JSON / Markdown / PNG report artefacts land on disk."""
    print("\n[5] Report writers: JSON / Markdown / PNGs land on disk")
    tmp = Path(tempfile.mkdtemp(prefix="diag_smoke_"))
    try:
        bl = np.abs(np.random.randn(50).astype(np.float64))
        il = np.abs(np.random.randn(80).astype(np.float64)) * 0.5
        save_h1_figure(bl, il, tmp / "fig_h1.png")
        mis = np.abs(np.random.randn(20).astype(np.float64))
        suc = (np.random.rand(20) > 0.5).astype(np.float64)
        save_h2_figure(mis, suc, tmp / "fig_h2.png", pearson_r=-0.42)
        assert (tmp / "fig_h1.png").exists() and (tmp / "fig_h1.png").stat().st_size > 1000
        assert (tmp / "fig_h2.png").exists() and (tmp / "fig_h2.png").stat().st_size > 1000

        h1 = {"boundary_loss_mean": 0.5, "interior_loss_mean": 0.25,
              "ratio": 2.0, "t_stat": 4.2, "p_value": 0.001,
              "n_boundary": 50, "n_interior": 80,
              "verdict": verdict_h1(2.0, 0.001)}
        h2 = {"num_episodes": 20, "success_rate": 0.5,
              "mean_misalignment_success": 1.2, "mean_misalignment_failure": 3.4,
              "pearson_r": -0.55, "p_value": 0.001,
              "verdict": verdict_h2(-0.55, 0.001)}
        payload = build_report_payload(h1=h1, h2=h2,
                                       meta={"device": "cpu", "synthetic": True})
        write_json_report(payload, tmp / "report.json")
        write_markdown_report(payload, tmp / "report.md",
                              synthetic_demo=True, synthetic_env=True)
        data = json.loads((tmp / "report.json").read_text())
        assert data["h1"]["verdict"] == "PASS"
        assert data["h2"]["verdict"] == "PASS"
        text = (tmp / "report.md").read_text()
        assert "Executive Summary" in text and "SYNTHETIC" in text
        print(f" OK (artefacts under {tmp})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    """Run all five diagnostic-pipeline smoke checks and return a process exit code."""
    torch.manual_seed(0)
    np.random.seed(0)
    tests = [
        smoke_synthetic_env,
        smoke_phase_proxies,
        smoke_h1_pipeline,
        smoke_h2_pipeline,
        smoke_report_writers,
    ]
    for fn in tests:
        try:
            fn()
        except Exception: # noqa: BLE001
            print(f"\n[FAIL] {fn.__name__} raised:\n")
            traceback.print_exc()
            return 1
    print("\n[PASS] All Round 1 diagnostic smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
