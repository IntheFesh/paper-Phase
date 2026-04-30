"""Schema-level tests for the Phase-Centric extension fields on ``PhaseQFlowConfig``."""

from __future__ import annotations

import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_SRC = _REPO_ROOT / "lerobot_policy_phaseqflow" / "src"
_SCRIPTS_TRAINING = _REPO_ROOT / "scripts" / "training"
for p in (_PKG_SRC, _SCRIPTS_TRAINING):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig # noqa: E402


NEW_FIELD_DEFAULTS = {
    "use_chunk_infonce": False,
    "chunk_infonce_weight": 0.5,
    "chunk_infonce_temperature": 0.1,
    "chunk_infonce_chunk_len": 8,
    "phase_posterior_smooth_alpha": 0.3,
    "use_phase_boundary_posterior": False,
    "use_pace_a": False,
    "pace_a_lambda": 2.0,
    "pace_a_entropy_weight": 0.01,
    "use_pace_b": False,
    "moe_num_experts": 4,
    "moe_expert_hidden_dim": 128,
    "moe_switch_kappa": 5.0,
    "moe_switch_mu": 2.0,
    "moe_top_k": 2,
    "use_pace_c": False,
    "curriculum_stage_steps": (1000, 3000, 10000),
    "curriculum_max_boundaries_stage1": 1,
    "curriculum_max_boundaries_stage2": 3,
    "use_pcar": False,
    "pcar_change_threshold": 0.4,
    "pcar_trigger_budget_eps": 0.1,
    "pcar_dual_head": False,
    "pcar_post_head_ratio": 0.5,
}


def test_new_fields_present_with_correct_defaults() -> None:
    """Every Phase-Centric field exists on ``PhaseQFlowConfig`` with the expected default."""
    cfg = PhaseQFlowConfig()
    for name, expected in NEW_FIELD_DEFAULTS.items():
        assert hasattr(cfg, name), f"missing new field: {name}"
        actual = getattr(cfg, name)
        assert actual == expected, (
            f"field {name}: default {actual!r} != expected {expected!r}"
        )


def test_new_field_types() -> None:
    """Switches are bools, numerics have the declared Python type, and stage steps is a 3-tuple of ints."""
    cfg = PhaseQFlowConfig()
    for switch in [
        "use_chunk_infonce", "use_phase_boundary_posterior",
        "use_pace_a", "use_pace_b", "use_pace_c", "use_pcar",
        "pcar_dual_head",
    ]:
        assert isinstance(getattr(cfg, switch), bool), switch
    for num_f, num_t in [
        ("chunk_infonce_weight", float),
        ("chunk_infonce_chunk_len", int),
        ("phase_posterior_smooth_alpha", float),
        ("pace_a_lambda", float),
        ("moe_num_experts", int),
        ("pcar_change_threshold", float),
    ]:
        assert isinstance(getattr(cfg, num_f), num_t), (num_f, num_t)
    s = cfg.curriculum_stage_steps
    assert isinstance(s, tuple) and len(s) == 3
    assert all(isinstance(x, int) for x in s)


def test_mode_preset_applies_override() -> None:
    """The ``MODE_PRESETS`` table in ``train_dummy_batch.py`` sets the expected switches per mode."""
    import train_dummy_batch as train_mod # type: ignore[import-not-found]

    cfg_off = train_mod._build_smoke_config("off")
    assert cfg_off.use_chunk_infonce is False
    assert cfg_off.use_pace_a is False
    assert cfg_off.use_pcar is False

    cfg_a = train_mod._build_smoke_config("pace_a")
    assert cfg_a.use_pace_a is True
    assert cfg_a.use_phase_boundary_posterior is True
    assert cfg_a.use_chunk_infonce is False
    assert cfg_a.use_pcar is False

    cfg_full = train_mod._build_smoke_config("full")
    for switch in [
        "use_chunk_infonce", "use_phase_boundary_posterior",
        "use_pace_a", "use_pace_b", "use_pace_c", "use_pcar",
    ]:
        assert getattr(cfg_full, switch) is True, switch


def test_cli_override_beats_preset() -> None:
    """A CLI override on ``--pace_a_lambda`` wins over the mode preset default."""
    import argparse

    import train_dummy_batch as train_mod # type: ignore[import-not-found]

    cfg = train_mod._build_smoke_config("pace_a")
    assert cfg.pace_a_lambda == 2.0

    parser = train_mod._build_parser()
    ns: argparse.Namespace = parser.parse_args(
        ["--phase-centric-mode", "pace_a", "--pace_a_lambda", "3.0"]
    )
    applied = train_mod._apply_overrides(cfg, ns)
    assert cfg.pace_a_lambda == 3.0, (cfg.pace_a_lambda, applied)
    assert any("pace_a_lambda=3.0" in s for s in applied), applied


def test_backward_compat_off_mode_matches_round1() -> None:
    """All Phase-Centric switches default to False so that ``mode=off`` reproduces Round 1 behaviour."""
    cfg_default = PhaseQFlowConfig()
    for switch in [
        "use_chunk_infonce", "use_phase_boundary_posterior",
        "use_pace_a", "use_pace_b", "use_pace_c", "use_pcar",
        "pcar_dual_head",
    ]:
        assert getattr(cfg_default, switch) is False, (
            f"{switch} must default to False for Round-1 backward compat."
        )


def test_phase_centric_subpackage_imports() -> None:
    """Every module under ``phase_centric/`` imports cleanly."""
    import importlib

    modules = [
        "lerobot_policy_phaseqflow.phase_centric",
        "lerobot_policy_phaseqflow.phase_centric.identifiability",
        "lerobot_policy_phaseqflow.phase_centric.phase_posterior",
        "lerobot_policy_phaseqflow.phase_centric.pace_a_loss",
        "lerobot_policy_phaseqflow.phase_centric.pace_b_moe",
        "lerobot_policy_phaseqflow.phase_centric.pace_c_curriculum",
        "lerobot_policy_phaseqflow.phase_centric.pcar_trigger",
        "lerobot_policy_phaseqflow.phase_centric.theory_utils",
    ]
    for mod in modules:
        importlib.import_module(mod)


def test_subpackage_functions_raise_not_implemented() -> None:
    """The remaining ``theory_utils`` placeholders still raise ``NotImplementedError``."""
    from lerobot_policy_phaseqflow.phase_centric import theory_utils

    placeholders = [
        (theory_utils.run_length_posterior, ()),
        (theory_utils.shannon_entropy, ()),
        (theory_utils.empirical_cdf, ()),
    ]
    for fn, args in placeholders:
        with pytest.raises(NotImplementedError):
            fn(*args)


def test_config_roundtrip_preserves_new_fields(tmp_path: Path) -> None:
    """New fields survive a save_pretrained / from_pretrained round-trip."""
    cfg = PhaseQFlowConfig(
        use_pace_a=True,
        pace_a_lambda=2.5,
        curriculum_stage_steps=(500, 1500, 5000),
    )
    cfg.save_pretrained(str(tmp_path))
    cfg2 = PhaseQFlowConfig.from_pretrained(str(tmp_path))
    assert cfg2.use_pace_a is True
    assert cfg2.pace_a_lambda == 2.5
    assert list(cfg2.curriculum_stage_steps) == [500, 1500, 5000]
