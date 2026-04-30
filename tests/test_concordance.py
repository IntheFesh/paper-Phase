"""Unit tests for the ConcordanceDetector (§2.4).

Tests:
  (a) warm-up — no trigger fires before warmup_steps
  (b) all-high — all three estimators agree on a cliff → triggered=True
  (c) budget — concordance is low when estimators are neutral → no trigger
  (d) reset — state is cleared after reset()
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_SRC = _REPO_ROOT / "lerobot_policy_phaseqflow" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from lerobot_policy_phaseqflow.phase_centric.cliff_detection.concordance import (
    ConcordanceDetector,
)


# ---------------------------------------------------------------------------
# (a) warm-up: triggers suppressed until warmup_steps have elapsed
# ---------------------------------------------------------------------------

def test_concordance_no_trigger_during_warmup() -> None:
    """No trigger fires during the warmup period even with extreme signals."""
    det = ConcordanceDetector(window_size=10, threshold=0.5, warmup_steps=10)
    # Feed 10 very negative values (cliff signals); still in warmup
    for _ in range(10):
        out = det.step(-1.0, -1.0, -1.0)
        assert out["triggered"] is False, "trigger fired during warmup"


def test_concordance_trigger_fires_after_warmup() -> None:
    """Trigger fires immediately after warmup when all signals are at their minimum."""
    det = ConcordanceDetector(window_size=10, threshold=0.5, warmup_steps=5)
    # First 5 steps: neutral (not warmup-suppressed after step 5)
    for _ in range(5):
        det.step(0.0, 0.0, 0.0)
    # Next steps: strong cliff signal (very low values dominate the window)
    fired = False
    for _ in range(10):
        out = det.step(-10.0, -10.0, -10.0)
        if out["triggered"]:
            fired = True
            break
    assert fired, "trigger never fired after warmup with strong cliff signal"


# ---------------------------------------------------------------------------
# (b) all-high: all three estimators agree → high concordance
# ---------------------------------------------------------------------------

def test_concordance_all_high_signal_gives_high_concordance() -> None:
    """After warm-up, a strong cliff step should give concordance >= 0.8."""
    det = ConcordanceDetector(window_size=20, threshold=0.8, warmup_steps=5)
    # Warm-up with mild values
    for _ in range(20):
        det.step(-0.1, -0.1, -0.1)
    # Single extreme cliff
    out = det.step(-100.0, -100.0, -100.0)
    assert out["concordance"] >= 0.8, f"concordance={out['concordance']:.3f}"
    assert out["triggered"] is True


def test_concordance_ranks_all_high_for_cliff() -> None:
    """Each rank should be close to 1.0 for an extreme cliff step."""
    det = ConcordanceDetector(window_size=20, threshold=0.8, warmup_steps=0)
    for _ in range(20):
        det.step(-0.1, -0.1, -0.1)
    out = det.step(-100.0, -100.0, -100.0)
    r1, r2, r3 = out["ranks"]
    assert r1 > 0.9 and r2 > 0.9 and r3 > 0.9


# ---------------------------------------------------------------------------
# (c) budget / neutral: low concordance when signals are not cliff-like
# ---------------------------------------------------------------------------

def test_concordance_neutral_signal_low_concordance() -> None:
    """When the current signal equals the window median, concordance ≈ 0.5."""
    det = ConcordanceDetector(window_size=20, threshold=0.8, warmup_steps=0)
    # Fill window with the same value
    for _ in range(20):
        det.step(-0.5, -0.5, -0.5)
    # Observe the same value again — should be median rank
    out = det.step(-0.5, -0.5, -0.5)
    assert out["concordance"] <= 0.6, f"concordance={out['concordance']:.3f}"
    assert out["triggered"] is False


def test_concordance_none_estimator_uses_neutral_rank() -> None:
    """None estimator values are replaced by 0.5 (neutral rank contribution)."""
    det = ConcordanceDetector(window_size=10, threshold=0.8, warmup_steps=0)
    for _ in range(10):
        det.step(-0.1, -0.1, None)
    out = det.step(-100.0, -100.0, None)
    r1, r2, r3 = out["ranks"]
    assert r3 == pytest.approx(0.5)
    assert r1 > 0.9
    assert r2 > 0.9


# ---------------------------------------------------------------------------
# (d) reset: state cleared after reset()
# ---------------------------------------------------------------------------

def test_concordance_reset_clears_history() -> None:
    """After reset, step count and history are cleared."""
    det = ConcordanceDetector(window_size=10, threshold=0.5, warmup_steps=5)
    for _ in range(20):
        det.step(-1.0, -1.0, -1.0)
    det.reset()
    assert det._step_count == 0
    assert len(det._history) == 0


def test_concordance_no_trigger_after_reset() -> None:
    """After reset, the warmup suppression applies again."""
    det = ConcordanceDetector(window_size=10, threshold=0.5, warmup_steps=10)
    for _ in range(20):
        det.step(-1.0, -1.0, -1.0)
    det.reset()
    # In the new episode (step 1, still within warmup), no trigger
    out = det.step(-100.0, -100.0, -100.0)
    assert out["triggered"] is False


def test_concordance_window_size_lt2_raises() -> None:
    """window_size < 2 must raise ValueError."""
    with pytest.raises(ValueError):
        ConcordanceDetector(window_size=1)


def test_concordance_threshold_out_of_range_raises() -> None:
    """Threshold outside (0, 1] must raise ValueError."""
    with pytest.raises(ValueError):
        ConcordanceDetector(threshold=0.0)
    with pytest.raises(ValueError):
        ConcordanceDetector(threshold=1.1)
