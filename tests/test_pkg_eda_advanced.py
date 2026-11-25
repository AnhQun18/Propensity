"""Tests for standalone src/eda_advanced package (tabular + longitudinal advanced EDA)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data_io import write_table
from src.eda_advanced import (
    analyze_bipartite_graph,
    build_multi_horizon_targets,
    buyer_composition_and_transitions,
    categorical_associations,
    cluster_bootstrap_rates,
    compare_distributions,
    compare_two_groups,
    compute_surge_ratio,
    compute_vif,
    correlation_analysis,
    period_change_decomposition,
    periodic_categorical_profile,
    population_stability_index,
    run_event_study,
    run_stratified_event_study,
    stratified_lift_table,
    target_signal_scan,
    temporal_drift_report,
    wilson_ci,
    woe_iv_table,
)


class TestEDAAdvanced(unittest.TestCase):
    def test_tabular_relationships_vif_and_target_signal_scan(self):
        rng = np.random.default_rng(7)
        n = 500
        x1 = rng.normal(0, 1, n)
        x2 = x1 * 0.95 + rng.normal(0, 0.1, n)  # strongly collinear with x1
        x3 = rng.normal(0, 1, n)                # noise
        cat_strong = np.where(x1 > 0.3, "vip", "std")
        cat_dup = np.where(x1 > 0.3, "tier_high", "tier_low")
        prob = 1.0 / (1.0 + np.exp(-(2.0 * x1 - 0.5)))
        y = (rng.random(n) < prob).astype(int)

        df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "cat_strong": cat_strong, "cat_dup": cat_dup, "y": y})

        corr = correlation_analysis(df, method="spearman", threshold=0.8)
        self.assertEqual(len(corr["top_pairs"]), 1)
        self.assertGreater(corr["top_pairs"]["abs_corr"].iloc[0], 0.9)

        vif = compute_vif(df, columns=["x1", "x2", "x3"]).set_index("column")
        self.assertGreater(float(vif.loc["x1", "vif"]), 5.0)
        self.assertLess(float(vif.loc["x3", "vif"]), 1.5)

        cat_assoc = categorical_associations(df, columns=["cat_strong", "cat_dup"])
        self.assertGreater(float(cat_assoc["cramers_v"].iloc[0]), 0.95)

        woe_tbl, iv_x1 = woe_iv_table(df["x1"], df["y"], n_bins=5)
        self.assertGreater(iv_x1, 0.3)
        self.assertIn("woe", woe_tbl.columns)

        scan = target_signal_scan(df, target_col="y")
        self.assertIn(scan.iloc[0]["feature"], {"x1", "x2"})
        x1_row = scan.set_index("feature").loc["x1"]
        self.assertGreater(float(x1_row["auc"]), 0.75)
        self.assertLess(float(x1_row["p_value"]), 1e-5)

    def test_drift_and_two_group_tests(self):
        rng = np.random.default_rng(11)
        ref = pd.DataFrame({
            "period": "2025-01",
            "spend": rng.normal(100, 15, 400),
            "channel": rng.choice(["web", "app"], size=400, p=[0.8, 0.2]),
        })
        cur = pd.DataFrame({
            "period": "2025-02",
            "spend": rng.normal(160, 15, 400),  # big upward shift
            "channel": rng.choice(["web", "app", "store"], size=400, p=[0.3, 0.5, 0.2]),
        })
        psi_spend = population_stability_index(ref["spend"], cur["spend"])
        self.assertGreater(psi_spend, 0.25)

        drift = compare_distributions(ref, cur, columns=["spend", "channel"]).set_index("column")
        self.assertEqual(drift.loc["spend", "drift_level"], "high")
        self.assertEqual(int(drift.loc["channel", "new_categories_n"]), 1)

        td = temporal_drift_report(pd.concat([ref, cur], ignore_index=True), period_col="period")
        self.assertEqual(len(td), 2)

        cmp = compare_two_groups(ref["spend"], cur["spend"])
        self.assertLess(cmp["mannwhitney_p"], 1e-10)
        self.assertLess(cmp["cohens_d"], -2.0)

    def test_wilson_cluster_bootstrap_and_cli(self):
        lo, hi = wilson_ci([0, 50, 100], [100, 100, 100])
        self.assertTrue(np.all(lo >= 0) and np.all(hi <= 1))

        rng = np.random.default_rng(42)
        entity_ids = np.repeat(np.arange(100), 6)
        has_sig = rng.random(len(entity_ids)) > 0.5
        y = (rng.random(len(entity_ids)) < np.where(has_sig, 0.20, 0.05)).astype(int)
        res = cluster_bootstrap_rates(entity_ids, {"with_sig": has_sig, "no_sig": ~has_sig}, y, n_reps=300, seed=42)
        self.assertGreater(res.set_index("group").loc["with_sig", "lift"], 1.3)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tab.csv"
            write_table(pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "y": [0, 0, 1, 1, 1]}), p)
            repo_root = Path(__file__).resolve().parents[1]
            cli = subprocess.run(
                [sys.executable, "-m", "src.eda_advanced", "target-scan", str(p), "--target", "y"],
                cwd=str(repo_root), check=True, capture_output=True, text=True,
            )
            self.assertIn("feature", cli.stdout)

    def test_longitudinal_panel_methods(self):
        ev = pd.DataFrame({
            "user_id": ["u1", "u1", "u1", "u1", "u2"],
            "ts": ["2025-01-05 10:00:00", "2025-01-10 10:00:00", "2025-02-05 10:00:00",
                   "2025-02-06 10:00:00", "2025-01-15 10:00:00"],
            "sku": ["A", "A", "B", "B", "A"],
        })
        prof = periodic_categorical_profile(ev, entity_col="user_id", time_col="ts", category_col="sku")
        u1 = prof[prof["user_id"] == "u1"].sort_values("period")
        self.assertAlmostEqual(float(u1.iloc[1]["changed_main"]), 1.0)

        edges = pd.DataFrame({"user_id": ["u1", "u1", "u2", "u3"], "item": ["i1", "i2", "i2", "i3"], "events": [5, 1, 3, 2]})
        bip = analyze_bipartite_graph(["u1", "u2", "u3", "u4"], edges, entity_col="user_id", item_col="item", svd_components=2)
        self.assertEqual(bip["embeddings"].shape, (4, 2))

        rng = np.random.default_rng(1)
        metric = rng.poisson(3, size=(30, 8)).astype(float)
        metric[:, 0] = np.nan
        outcome = np.full((30, 8), np.nan)
        outcome[:, 4:7] = rng.binomial(1, 0.25, size=(30, 3))
        periods = [f"2025-0{i}" for i in range(1, 9)]
        entities = [f"u{i}" for i in range(30)]
        self.assertFalse(run_event_study(entities, periods, metric, outcome, lags=(-2, -1, 0, 1), min_baseline_periods=2).empty)
        self.assertFalse(run_stratified_event_study(entities, periods, metric, metric, outcome, lags=(-2, -1, 0, 1), n_quantiles=3).empty)
        self.assertEqual(period_change_decomposition(metric, periods)["period"].nunique(), 7)

        panel = pd.DataFrame([
            {"user_id": u, "month": f"2025-0{m}", "y": int(u == "u1" and m in (2, 4)), "trx": 10 if (u == "u1" and m == 3) else 2, "seg": "A" if u in ("u1", "u2") else "B"}
            for u in ("u1", "u2", "u3", "u4") for m in range(1, 7)
        ])
        mh = build_multi_horizon_targets(panel, entity_col="user_id", period_col="month", target_col="y", horizons=(1, 2, 3))
        self.assertEqual(len(mh), 16)
        self.assertAlmostEqual(float(stratified_lift_table(mh, group_cols="seg", target_cols=["h1", "h3"])["share_pct"].sum()), 100.0)
        surge = compute_surge_ratio(panel, entity_col="user_id", period_col="month", value_col="trx")
        self.assertAlmostEqual(float(surge[(surge["user_id"] == "u1") & (surge["month"] == "2025-04")]["trx_surge_ratio"].iloc[0]), 5.0)
        mat = panel.pivot(index="user_id", columns="month", values="y").to_numpy(dtype=float)
        self.assertIn("transitions", buyer_composition_and_transitions(mat, sorted(panel["month"].unique())))


if __name__ == "__main__":
    unittest.main()
