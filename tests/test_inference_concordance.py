"""Unit tests for the runtime-facing ConcordanceDetector.

These tests cover the PACE v2 ``inference/concordance.py`` module that
fuses I^(1) (Bhattacharyya β_t), I^(2) (action variance), and I^(3)
(velocity curvature) into a single rank-based C_t signal at evaluation
time. Pure NumPy; no torch / lerobot dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "lerobot_policy_phaseqflow" / "src"),
)

from lerobot_policy_phaseqflow.inference.concordance import ConcordanceDetector


def test_initial_state_is_no_cliff() -> None:
    """Empty detector must report no cliff and yield empty stats."""
    det = ConcordanceDetector(window=8, threshold=0.35)
    assert det.get_stats() == {}
    assert det.is_cliff() is False


def test_warmup_returns_safe_value() -> None:
    """First two updates lack a 3-sample window and should return c_t = 1.0."""
    det = ConcordanceDetector(window=8, threshold=0.35)
    c0 = det.update(0.1, 0.05, 0.02)
    c1 = det.update(0.2, 0.10, 0.04)
    assert c0 == pytest.approx(1.0)
    assert c1 == pytest.approx(1.0)
    assert det.is_cliff(c0) is False
    assert det.is_cliff(c1) is False


def test_spike_triggers_cliff() -> None:
    """Steady low signals followed by a clear spike must trigger is_cliff=True."""
    det = ConcordanceDetector(window=8, threshold=0.35)
    for _ in range(7):
        det.update(0.05, 0.01, 0.005)
    c_spike = det.update(1.0, 1.0, 1.0)
    assert c_spike < 0.35, f"spike c_t={c_spike} did not cross threshold"
    assert det.is_cliff(c_spike) is True


def test_steady_signal_no_cliff() -> None:
    """Constant signals → median rank ≈ 0.5; should not trigger."""
    det = ConcordanceDetector(window=8, threshold=0.35)
    for _ in range(8):
        c = det.update(0.5, 0.5, 0.5)
    assert c >= 0.35
    assert det.is_cliff(c) is False


def test_reset_clears_history() -> None:
    """reset() must clear all internal state."""
    det = ConcordanceDetector(window=4, threshold=0.35)
    for _ in range(4):
        det.update(1.0, 1.0, 1.0)
    assert len(det._concordance_history) == 4
    det.reset()
    assert det.get_stats() == {}
    assert det.is_cliff() is False


def test_window_is_bounded() -> None:
    """Sliding window must cap each estimator's history at ``window``."""
    det = ConcordanceDetector(window=4, threshold=0.35)
    for i in range(20):
        det.update(float(i), float(i), float(i))
    assert len(det._hist1) == 4
    assert len(det._hist2) == 4
    assert len(det._hist3) == 4


def test_stats_track_cliff_events() -> None:
    """After several cliff-like updates, n_cliff_events must increase."""
    det = ConcordanceDetector(window=4, threshold=0.5)
    for _ in range(3):
        det.update(0.01, 0.01, 0.01)
    for _ in range(3):
        det.update(1.0, 1.0, 1.0)
    stats = det.get_stats()
    assert stats["total_steps"] == 6
    assert stats["n_cliff_events"] >= 1
    assert 0.0 <= stats["min_concordance"] <= stats["mean_concordance"] <= 1.0
