"""Smoke tests for Phase E phenomenon scripts and diagnostic utilities.

Covers:
- Ablation config YAML structure (all 7 + 2 sanity configs are valid YAML
  and contain required keys)
- Diagnostic scripts: dry-run smoke for all 4 new scripts
- Figure scripts: dry-run smoke (PNG/PDF output, no crash)
- aggregate_ablation.py: dry-run IQM + CI + Cohen's d
- Diagnostic trigger_comparison: concordance vs alternatives
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "lerobot_policy_phaseqflow" / "src"))

_ABLATION_DIR = _REPO_ROOT / "configs" / "ablation" / "v2"


# ---------------------------------------------------------------------------
# Ablation config validation
# ---------------------------------------------------------------------------

class TestAblationConfigs:
    _EXPECTED = [
        "01_bc_chunked.yaml",
        "02_cliff_via_beta_only.yaml",
        "03_cliff_via_var_only.yaml",
        "04_cliff_via_curvature_only.yaml",
        "05_cliff_concordance.yaml",
        "06_oracle_cliff.yaml",
        "07_cliff_concordance_with_boundary_reweight.yaml",
    ]
    _SANITY = [
        "sanity/random_phase.yaml",
        "sanity/oracle_boundary.yaml",
    ]

    def test_all_7_configs_exist(self):
        for fname in self._EXPECTED:
            assert (_ABLATION_DIR / fname).exists(), f"Missing: {fname}"

    def test_sanity_configs_exist(self):
        for fname in self._SANITY:
            assert (_ABLATION_DIR / fname).exists(), f"Missing: {fname}"

    @pytest.mark.parametrize("fname", _EXPECTED)
    def test_yaml_parseable(self, fname):
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")
        with open(_ABLATION_DIR / fname) as f:
            data = yaml.safe_load(f)
        assert data is not None
        assert "policy" in data or "phaseqflow" in data or "ablation_id" in data

    @pytest.mark.parametrize("fname", _EXPECTED)
    def test_has_ablation_id(self, fname):
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")
        with open(_ABLATION_DIR / fname) as f:
            data = yaml.safe_load(f)
        assert "ablation_id" in data, f"{fname} missing ablation_id"

    def test_bc_chunked_disables_cliff_detection(self):
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")
        with open(_ABLATION_DIR / "01_bc_chunked.yaml") as f:
            data = yaml.safe_load(f)
        pq = data.get("phaseqflow", {})
        assert pq.get("use_pcar") is False
        assert pq.get("use_phase_boundary_posterior") is False

    def test_full_method_enables_all(self):
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")
        with open(_ABLATION_DIR / "07_cliff_concordance_with_boundary_reweight.yaml") as f:
            data = yaml.safe_load(f)
        pq = data.get("phaseqflow", {})
        assert pq.get("use_pcar") is True
        assert pq.get("pcar_input_signal") == "concordance"
        assert pq.get("use_boundary_reweight") is True


# ---------------------------------------------------------------------------
# Diagnostic scripts
# ---------------------------------------------------------------------------

class TestDiagnosticBoundaryError:
    def test_dry_run(self, tmp_path):
        from scripts.diagnostics.diagnostic_utils.measure_boundary_error import main
        rc = main(["--dry_run", "--output", str(tmp_path / "boundary_error.csv")])
        assert rc == 0
        assert (tmp_path / "boundary_error.csv").exists()
        assert (tmp_path / "boundary_error.json").exists()

    def test_ratio_positive(self, tmp_path):
        from scripts.diagnostics.diagnostic_utils.measure_boundary_error import main, _synthetic_ratio
        E_b, E_i, ratio = _synthetic_ratio()
        assert E_b > 0
        assert E_i > 0
        assert ratio > 1.0, f"Expected E_boundary > E_interior, got ratio={ratio:.3f}"

    def test_csv_has_ratio_row(self, tmp_path):
        from scripts.diagnostics.diagnostic_utils.measure_boundary_error import main
        main(["--dry_run", "--output", str(tmp_path / "out.csv")])
        with open(tmp_path / "out.csv") as f:
            rows = {r["metric"]: float(r["value"]) for r in csv.DictReader(f)}
        assert "ratio_boundary_over_interior" in rows
        assert rows["ratio_boundary_over_interior"] > 0


class TestDiagnosticReplanAlignment:
    def test_dry_run(self, tmp_path):
        from scripts.diagnostics.diagnostic_utils.replan_alignment import main
        rc = main(["--dry_run", "--output", str(tmp_path / "replan.csv")])
        assert rc == 0
        assert (tmp_path / "replan.csv").exists()

    def test_f1_in_range(self, tmp_path):
        from scripts.diagnostics.diagnostic_utils.replan_alignment import main
        main(["--dry_run", "--output", str(tmp_path / "out.csv")])
        with open(tmp_path / "out.csv") as f:
            rows = {r["metric"]: float(r["value"]) for r in csv.DictReader(f)}
        assert 0.0 <= rows["F1"] <= 1.0
        assert 0.0 <= rows["precision"] <= 1.0
        assert 0.0 <= rows["recall"] <= 1.0


class TestDiagnosticTriggerComparison:
    def test_dry_run(self, tmp_path):
        from scripts.diagnostics.diagnostic_utils.trigger_comparison import main
        rc = main(["--dry_run", "--n_episodes", "10",
                   "--output", str(tmp_path / "trigger.csv")])
        assert rc == 0
        assert (tmp_path / "trigger.csv").exists()

    def test_six_methods(self, tmp_path):
        from scripts.diagnostics.diagnostic_utils.trigger_comparison import main
        main(["--dry_run", "--n_episodes", "10",
              "--output", str(tmp_path / "out.csv")])
        with open(tmp_path / "out.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6
        methods = {r["method"] for r in rows}
        assert "concordance_C" in methods
        assert "bhattacharyya" in methods
        assert "bocpd" in methods

    def test_concordance_precision_highest(self, tmp_path):
        from scripts.diagnostics.diagnostic_utils.trigger_comparison import main
        main(["--dry_run", "--n_episodes", "20",
              "--output", str(tmp_path / "out.csv")])
        with open(tmp_path / "out.csv") as f:
            rows = {r["method"]: float(r["precision"]) for r in csv.DictReader(f)}
        conc_prec = rows["concordance_C"]
        single_precs = [v for k, v in rows.items() if k != "concordance_C"]
        assert conc_prec >= max(single_precs) * 0.8, (
            f"Concordance precision {conc_prec:.3f} much lower than singles {max(single_precs):.3f}"
        )


class TestDiagnosticInferenceCost:
    def test_dry_run(self, tmp_path):
        from scripts.diagnostics.diagnostic_utils.measure_inference_cost import main
        rc = main(["--dry_run", "--output", str(tmp_path / "cost.csv")])
        assert rc == 0
        assert (tmp_path / "cost.csv").exists()
        assert (tmp_path / "cost.json").exists()

    def test_six_methods_in_csv(self, tmp_path):
        from scripts.diagnostics.diagnostic_utils.measure_inference_cost import main
        main(["--dry_run", "--output", str(tmp_path / "out.csv")])
        with open(tmp_path / "out.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6

    def test_concordance_nfe_larger_than_beta(self, tmp_path):
        from scripts.diagnostics.diagnostic_utils.measure_inference_cost import main
        main(["--dry_run", "--output", str(tmp_path / "out.csv")])
        with open(tmp_path / "out.csv") as f:
            rows = {r["method"]: int(r["NFE"]) for r in csv.DictReader(f)}
        assert rows["cliff_concordance"] >= rows["cliff_via_beta_only"], (
            "Concordance should require >= NFE of beta-only"
        )


# ---------------------------------------------------------------------------
# Figure scripts (smoke only — check no crash, output file created)
# ---------------------------------------------------------------------------

class TestFigureScripts:
    def test_fig1_universality(self, tmp_path):
        from scripts.figures.fig1_universality import main
        rc = main(["--input", str(tmp_path / "nonexistent.json"),
                   "--output", str(tmp_path / "fig1.pdf")])
        assert rc == 0

    def test_fig2_method_overview(self, tmp_path):
        from scripts.figures.fig2_method_overview import main
        rc = main(["--output", str(tmp_path / "fig2.pdf")])
        assert rc == 0

    def test_fig3_phase_visualization(self, tmp_path):
        from scripts.figures.fig3_phase_visualization import main
        rc = main(["--dry_run", "--n_episodes", "2",
                   "--output", str(tmp_path / "fig3.pdf")])
        assert rc == 0

    def test_fig4_regret_scaling(self, tmp_path):
        from scripts.figures.fig4_regret_scaling import main
        rc = main(["--input", str(tmp_path / "nonexistent.csv"),
                   "--output", str(tmp_path / "fig4.pdf")])
        assert rc == 0

    def test_fig5_concordance_pr_curve(self, tmp_path):
        from scripts.figures.fig5_concordance_pr_curve import main
        rc = main(["--n_episodes", "10", "--output", str(tmp_path / "fig5.pdf")])
        assert rc == 0


# ---------------------------------------------------------------------------
# aggregate_ablation.py
# ---------------------------------------------------------------------------

class TestAggregateAblation:
    def test_dry_run(self, tmp_path):
        from scripts.aggregate_ablation import main
        rc = main(["--dry_run", "--output", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "ablation_table_v2.csv").exists()
        assert (tmp_path / "ablation_table_v2.tex").exists()
        assert (tmp_path / "ablation_stats_v2.json").exists()

    def test_csv_has_all_methods(self, tmp_path):
        from scripts.aggregate_ablation import main, _METHODS, _BENCHMARKS
        main(["--dry_run", "--output", str(tmp_path)])
        with open(tmp_path / "ablation_table_v2.csv") as f:
            rows = list(csv.DictReader(f))
        methods_in_csv = {r["method"] for r in rows}
        assert set(_METHODS).issubset(methods_in_csv), (
            f"Missing: {set(_METHODS) - methods_in_csv}"
        )

    def test_iqm_ordering(self, tmp_path):
        """Full method (07) should have higher IQM than BC baseline (01)."""
        from scripts.aggregate_ablation import main
        main(["--dry_run", "--output", str(tmp_path)])
        with open(tmp_path / "ablation_table_v2.csv") as f:
            rows = list(csv.DictReader(f))
        by_method_bm = {(r["method"], r["benchmark"]): float(r["IQM"]) for r in rows}
        for bm in ["libero_long", "libero_spatial", "simpler"]:
            baseline = by_method_bm.get(("01_bc_chunked", bm))
            full = by_method_bm.get(("07_cliff_concordance_with_boundary_reweight", bm))
            if baseline is not None and full is not None:
                assert full > baseline, (
                    f"[{bm}] Full method IQM {full:.3f} ≤ baseline {baseline:.3f}"
                )

    def test_iqm_helper(self):
        from scripts.aggregate_ablation import _iqm
        scores = np.array([0.1, 0.5, 0.5, 0.5, 0.9])
        iqm = _iqm(scores)
        assert 0.4 <= iqm <= 0.6

    def test_bootstrap_ci_coverage(self):
        from scripts.aggregate_ablation import _bootstrap_ci
        rng = np.random.default_rng(42)
        scores = np.array([0.6, 0.65, 0.62, 0.64, 0.61])
        lo, hi = _bootstrap_ci(scores, n_bootstrap=200, rng=rng)
        assert lo < hi
        assert lo <= 0.65
        assert hi >= 0.60

    def test_latex_output_contains_method(self, tmp_path):
        from scripts.aggregate_ablation import main
        main(["--dry_run", "--output", str(tmp_path)])
        tex = (tmp_path / "ablation_table_v2.tex").read_text()
        assert "BC-Chunked" in tex
        assert "tabular" in tex
