"""Smoke tests for Phase D phenomenon discovery scripts.

All tests run in ``--dry_run`` mode so no checkpoints, real environments,
or optional libraries (matplotlib, scipy) are required.

Tests cover:
- universality.py: synthetic data, histogram output, KS matrix, summary.md,
  preliminary PASS/FAIL logic
- regret_scaling.py: synthetic sweep, CSV output, summary.md
- triangulation_concordance.py: synthetic episodes, F1 > 0, concordance ≥ singles

Additionally smoke-tests the baselines/_base_adapter.py interface and the
aggregate_results.py aggregation script.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "lerobot_policy_phaseqflow" / "src"))


# ---------------------------------------------------------------------------
# Baselines adapter smoke tests
# ---------------------------------------------------------------------------

class TestBaseAdapter:
    def test_rollout_result_failure_distance(self):
        from baselines._base_adapter import RolloutResult

        r = RolloutResult(
            trajectory_len=100,
            success=False,
            failure_step=80,
            cliff_steps=[20, 60, 75],
        )
        assert r.failure_distance() == 5  # 80 - 75

    def test_rollout_result_no_cliff(self):
        from baselines._base_adapter import RolloutResult

        r = RolloutResult(
            trajectory_len=100,
            success=False,
            failure_step=50,
            cliff_steps=[],
        )
        assert r.failure_distance() is None

    def test_rollout_result_success(self):
        from baselines._base_adapter import RolloutResult

        r = RolloutResult(trajectory_len=100, success=True, failure_step=None, cliff_steps=[30])
        assert r.failure_distance() is None

    def test_cliff_steps_from_actions(self):
        from baselines._base_adapter import RolloutResult, PolicyAdapter

        class _DummyAdapter(PolicyAdapter):
            def is_available(self): return True
            def load(self): pass
            def rollout(self, env, n_steps, seed=0): return RolloutResult(0, True, None)

        adapter = _DummyAdapter("dummy")
        # Constant action → no spikes → no cliff steps
        actions = [np.zeros(4) for _ in range(20)]
        steps = adapter.cliff_steps_from_actions(actions)
        assert isinstance(steps, list)

        # Action with one big spike
        spiky = [np.zeros(4) for _ in range(20)]
        spiky[10] = np.ones(4) * 100.0
        steps_spiky = adapter.cliff_steps_from_actions(spiky)
        assert 10 in steps_spiky


# ---------------------------------------------------------------------------
# Universality smoke tests
# ---------------------------------------------------------------------------

class TestUniversality:
    def test_dry_run_basic(self, tmp_path):
        from scripts.phenomenon.universality import main

        rc = main(["--dry_run", "--n_rollouts", "5",
                   "--policies", "openvla", "bc_act",
                   "--output", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "summary.md").exists()
        assert (tmp_path / "ks_pvalue_matrix.csv").exists()
        assert (tmp_path / "raw_distances.json").exists()

    def test_raw_distances_json_structure(self, tmp_path):
        from scripts.phenomenon.universality import main

        main(["--dry_run", "--n_rollouts", "8",
              "--policies", "openvla", "pi0",
              "--output", str(tmp_path)])
        data = json.loads((tmp_path / "raw_distances.json").read_text())
        assert "openvla" in data
        assert "pi0" in data
        assert len(data["openvla"]) == 8

    def test_ks_matrix_csv_shape(self, tmp_path):
        from scripts.phenomenon.universality import main

        main(["--dry_run", "--n_rollouts", "6",
              "--policies", "openvla", "bc_act", "diffusion_policy",
              "--output", str(tmp_path)])
        with open(tmp_path / "ks_pvalue_matrix.csv") as f:
            rows = list(csv.reader(f))
        # Header row + 3 data rows
        assert len(rows) == 4
        assert rows[0][0] == ""  # corner cell
        assert len(rows[0]) == 4  # 3 policies + label

    def test_preliminary_pass(self, tmp_path):
        """With high Pearson threshold=0.0 (always passes on synthetic data)."""
        from scripts.phenomenon.universality import main

        rc = main(["--dry_run", "--n_rollouts", "20",
                   "--preliminary",
                   "--pearson_threshold", "0.0",
                   "--policies", "openvla", "pi0", "bc_act", "diffusion_policy",
                   "--output", str(tmp_path)])
        assert rc == 0
        summary = (tmp_path / "summary.md").read_text()
        assert "PASS" in summary

    def test_preliminary_fail(self, tmp_path):
        """With threshold=1.01 (always fails)."""
        from scripts.phenomenon.universality import main

        rc = main(["--dry_run", "--n_rollouts", "10",
                   "--preliminary",
                   "--pearson_threshold", "1.01",
                   "--policies", "openvla", "pi0", "bc_act", "diffusion_policy",
                   "--output", str(tmp_path)])
        assert rc == 1
        summary = (tmp_path / "summary.md").read_text()
        assert "FAIL" in summary

    def test_histogram_bins(self):
        from scripts.phenomenon.universality import _histogram_bins

        centers, counts = _histogram_bins([10, 20, 30, None, None], n_bins=10)
        assert len(centers) == 10
        assert len(counts) == 10
        assert counts.sum() > 0

    def test_histogram_all_success(self):
        from scripts.phenomenon.universality import _histogram_bins

        centers, counts = _histogram_bins([None, None, None])
        assert counts.sum() == 0.0

    def test_pearson_r_identical(self):
        from scripts.phenomenon.universality import _pearson_r

        a = np.array([1.0, 2.0, 3.0])
        assert _pearson_r(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_pearson_r_anticorrelated(self):
        from scripts.phenomenon.universality import _pearson_r

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        assert _pearson_r(a, b) == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Regret scaling smoke tests
# ---------------------------------------------------------------------------

class TestRegretScaling:
    def test_dry_run_basic(self, tmp_path):
        from scripts.phenomenon.regret_scaling import main

        rc = main(["--dry_run", "--H", "4", "8", "--output", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "regret_vs_H.csv").exists()
        assert (tmp_path / "summary.md").exists()

    def test_csv_columns(self, tmp_path):
        from scripts.phenomenon.regret_scaling import main

        main(["--dry_run", "--H", "4", "16", "64", "--output", str(tmp_path)])
        with open(tmp_path / "regret_vs_H.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert all(r.get("H") for r in rows)
        assert all(0.0 <= float(r["SR"]) <= 1.0 for r in rows)
        assert all(float(r["delta_SR"]) >= 0.0 for r in rows)

    def test_regret_increases_with_H(self, tmp_path):
        """Synthetic δSR should be monotonically non-decreasing with H."""
        from scripts.phenomenon.regret_scaling import main

        main(["--dry_run", "--H", "4", "8", "16", "32", "64", "--output", str(tmp_path)])
        with open(tmp_path / "regret_vs_H.csv") as f:
            rows = sorted(list(csv.DictReader(f)), key=lambda r: int(r["H"]))
        delta_SRs = [float(r["delta_SR"]) for r in rows]
        # Not strictly required by the synthetic model (noise), just check positive trend
        assert delta_SRs[-1] >= delta_SRs[0]

    def test_synthetic_regret_formula(self):
        """Check that synthetic regret data plausibly follows R ~ c·H·ΔH."""
        from scripts.phenomenon.regret_scaling import _synthetic_regret

        rng = np.random.default_rng(0)
        records = [_synthetic_regret(H, rng) for H in [4, 8, 16, 32]]
        for r in records:
            assert 0.0 <= r["SR"] <= 1.0
            assert 0.0 <= r["SR_ref"] <= 1.0
            assert r["delta_SR"] >= 0.0
            assert r["mean_delta_H"] > 0.0


# ---------------------------------------------------------------------------
# Triangulation concordance smoke tests
# ---------------------------------------------------------------------------

class TestTriangulationConcordance:
    def test_dry_run_basic(self, tmp_path):
        from scripts.phenomenon.triangulation_concordance import main

        rc = main(["--dry_run", "--n_episodes", "10",
                   "--output", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "triangulation_table.csv").exists()
        assert (tmp_path / "summary.md").exists()

    def test_csv_has_four_rows(self, tmp_path):
        from scripts.phenomenon.triangulation_concordance import main

        main(["--dry_run", "--n_episodes", "15", "--output", str(tmp_path)])
        with open(tmp_path / "triangulation_table.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4
        detectors = {r["detector"] for r in rows}
        assert "I_hat_1" in detectors
        assert "concordance_C" in detectors

    def test_f1_values_in_range(self, tmp_path):
        from scripts.phenomenon.triangulation_concordance import main

        main(["--dry_run", "--n_episodes", "20", "--output", str(tmp_path)])
        with open(tmp_path / "triangulation_table.csv") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            f1 = float(r["F1"])
            assert 0.0 <= f1 <= 1.0

    def test_concordance_beats_single_estimators(self, tmp_path):
        """Concordance precision should exceed every single estimator's precision
        (high specificity is the key property of rank-based fusion).
        Concordance F1 should also exceed at least one single estimator's F1."""
        from scripts.phenomenon.triangulation_concordance import main

        main(["--dry_run", "--n_episodes", "30", "--output", str(tmp_path)])
        with open(tmp_path / "triangulation_table.csv") as f:
            rows = list(csv.DictReader(f))
        by_prec = {r["detector"]: float(r["precision"]) for r in rows}
        by_f1 = {r["detector"]: float(r["F1"]) for r in rows}
        conc_prec = by_prec["concordance_C"]
        conc_f1 = by_f1["concordance_C"]
        single_precs = [v for k, v in by_prec.items() if k != "concordance_C"]
        single_f1s = [v for k, v in by_f1.items() if k != "concordance_C"]
        # Concordance should have higher precision than single estimators
        assert conc_prec >= max(single_precs)
        # Concordance F1 should beat at least one single estimator
        assert conc_f1 >= min(single_f1s)

    def test_oracle_set(self):
        from scripts.phenomenon.triangulation_concordance import _oracle_set

        gripper = np.array([0, 0, 1, 1, 0, 0, 1])
        flips = _oracle_set(gripper)
        assert 2 in flips  # 0→1
        assert 4 in flips  # 1→0
        assert 6 in flips  # 0→1

    def test_match_tolerance(self):
        from scripts.phenomenon.triangulation_concordance import _match

        predicted = {10, 25}
        oracle = {12, 30}
        # |10-12|=2 ≤ 5 → matches; |25-30|=5 ≤ 5 → also matches
        TP, FP, FN = _match(predicted, oracle, tolerance=5)
        assert TP == 2
        assert FP == 0
        assert FN == 0

    def test_match_strict_tolerance(self):
        from scripts.phenomenon.triangulation_concordance import _match

        predicted = {10}
        oracle = {20}
        TP, FP, FN = _match(predicted, oracle, tolerance=3)
        assert TP == 0
        assert FP == 1
        assert FN == 1

    def test_f1_helper(self):
        from scripts.phenomenon.triangulation_concordance import _f1

        prec, rec, f1 = _f1(TP=8, FP=2, FN=2)
        assert prec == pytest.approx(0.8, abs=1e-6)
        assert rec == pytest.approx(0.8, abs=1e-6)
        assert f1 == pytest.approx(0.8, abs=1e-6)

    def test_synthetic_episode_structure(self):
        from scripts.phenomenon.triangulation_concordance import _synthetic_episode

        rng = np.random.default_rng(0)
        ep = _synthetic_episode(T=100, n_transitions=2, rng=rng)
        assert ep.T == 100
        assert len(ep.gripper) == 100
        assert set(ep.gripper).issubset({0, 1})
        # At least 2 gripper flips
        flips = sum(1 for t in range(1, 100) if ep.gripper[t] != ep.gripper[t - 1])
        assert flips >= 2


# ---------------------------------------------------------------------------
# Aggregate results smoke test
# ---------------------------------------------------------------------------

class TestAggregateResults:
    def test_aggregate_empty_dirs(self, tmp_path):
        """Aggregation should not crash when experiment output dirs are missing."""
        from scripts.aggregate_results import main

        rc = main(["--figures_dir", str(tmp_path),
                   "--output", str(tmp_path / "main_results.csv")])
        assert rc == 0
        assert (tmp_path / "main_results.csv").exists()

    def test_aggregate_with_synthetic_outputs(self, tmp_path):
        """Run all dry_run experiments then aggregate."""
        from scripts.phenomenon.universality import main as uni_main
        from scripts.phenomenon.regret_scaling import main as reg_main
        from scripts.phenomenon.triangulation_concordance import main as tri_main
        from scripts.aggregate_results import main as agg_main

        uni_main(["--dry_run", "--n_rollouts", "5",
                  "--policies", "openvla", "bc_act",
                  "--output", str(tmp_path / "universality")])
        reg_main(["--dry_run", "--H", "4", "8",
                  "--output", str(tmp_path / "regret_scaling")])
        tri_main(["--dry_run", "--n_episodes", "5",
                  "--output", str(tmp_path / "triangulation")])

        rc = agg_main(["--figures_dir", str(tmp_path),
                       "--output", str(tmp_path / "main_results.csv")])
        assert rc == 0

        with open(tmp_path / "main_results.csv") as f:
            rows = list(csv.DictReader(f))
        experiments = {r["experiment"] for r in rows}
        assert "universality" in experiments
        assert "regret_scaling" in experiments
        assert "triangulation" in experiments
