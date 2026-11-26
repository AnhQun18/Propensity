"""Unit and end-to-end synthetic tests for Baseline v2 (Protocol B Monthly Rolling OOC)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline_v2.metrics import (
    bootstrap_multiseed_macro_monthly,
    bootstrap_multiseed_per_month,
    holm_bonferroni,
    paired_bootstrap_macro_monthly,
)
from baseline_v2.models import blend_prediction_frames
from baseline_v2.protocol import iter_protocol_steps, run_protocol_experiment
from baseline_v2.runner import run_baseline_v2
from baseline_v2.spec import (
    extract_feature_matrix,
    load_spec,
    validate_feature_set,
    verify_or_create_freeze_lock,
)


def _make_synthetic_panel(n_clients: int = 250, seed: int = 123) -> pd.DataFrame:
    """Create a synthetic 10-month panel matching panel_moscow.parquet columns and 5 disjoint folds."""
    rng = np.random.default_rng(seed)
    months = [
        "2022-04", "2022-05", "2022-06", "2022-07", "2022-08",
        "2022-09", "2022-10", "2022-11", "2022-12", "2023-01",
    ]
    cids = [f"c_{i:05d}" for i in range(n_clients)]
    folds = np.array([i % 5 for i in range(n_clients)], dtype=np.int8)

    rows = []
    for m_idx, m in enumerate(months):
        trx_ev = rng.poisson(3.0, size=n_clients).astype(np.float32)
        dlg_ev = rng.poisson(0.4, size=n_clients).astype(np.float32)
        geo_ev = rng.poisson(1.5, size=n_clients).astype(np.float32)
        tenure = (460.0 + m_idx * 30.0 + rng.uniform(-100.0, 20.0, size=n_clients)).astype(np.float32)
        # Inject semantic NaNs on recency/tenure when no events
        trx_30_rec = np.where(trx_ev > 0, rng.uniform(0.5, 29.0, size=n_clients), np.nan).astype(np.float32)
        trx_90_rec = np.where(trx_ev > 0, rng.uniform(0.5, 89.0, size=n_clients), np.nan).astype(np.float32)
        geo_90_rec = np.where(geo_ev > 0, rng.uniform(0.5, 89.0, size=n_clients), np.nan).astype(np.float32)
        dlg_90_rec = np.where(dlg_ev > 0, rng.uniform(0.5, 89.0, size=n_clients), np.nan).astype(np.float32)
        tenure = np.where(rng.random(n_clients) > 0.05, tenure, np.nan).astype(np.float32)

        logit_y = -3.2 + 0.35 * (dlg_ev > 0) + 0.12 * np.log1p(trx_ev)
        prob_y = 1.0 / (1.0 + np.exp(-logit_y))
        y = (rng.random(n_clients) < prob_y).astype(np.int8)

        p1 = rng.binomial(1, 0.04, size=n_clients).astype(np.int8)
        p3 = np.clip(p1 + rng.binomial(1, 0.05, size=n_clients), 0, 3).astype(np.int8)
        p6 = np.clip(p3 + rng.binomial(1, 0.05, size=n_clients), 0, 6).astype(np.int8)

        df_m = pd.DataFrame({
            "client_id": cids,
            "fold": folds,
            "month": m,
            "y_any": y,
            "prior_1m_bought_any": p1,
            "prior_3m_buy_months": p3,
            "prior_6m_buy_months": p6,
            "trx_30d_event_count": trx_ev.astype(np.int32),
            "trx_90d_event_count": (trx_ev * 2).astype(np.int32),
            "trx_30d_log1p_event_count": np.log1p(trx_ev).astype(np.float32),
            "trx_90d_log1p_event_count": np.log1p(trx_ev * 2).astype(np.float32),
            "trx_30d_log1p_main_amount_abs_sum": np.log1p(trx_ev * 100.0).astype(np.float32),
            "trx_90d_log1p_main_amount_abs_sum": np.log1p(trx_ev * 250.0).astype(np.float32),
            "trx_30d_days_since_last_event": trx_30_rec,
            "trx_90d_days_since_last_event": trx_90_rec,
            "trx_all_days_since_first_event": tenure,
            "trx_all_has_event": np.isfinite(tenure).astype(np.int8),
            "trx_90d_has_event": (trx_ev > 0).astype(np.int8),
            "geo_90d_has_event": (geo_ev > 0).astype(np.int8),
            "geo_90d_event_count": geo_ev.astype(np.int32),
            "geo_90d_days_since_last_event": geo_90_rec,
            "geo_all_has_event": (geo_ev > 0).astype(np.int8),
            "dialog_90d_has_event": (dlg_ev > 0).astype(np.int8),
            "dialog_90d_event_count": dlg_ev.astype(np.int32),
            "dialog_90d_days_since_last_event": dlg_90_rec,
            "dialog_all_has_event": (dlg_ev > 0).astype(np.int8),
            "cold_start": ((~np.isfinite(tenure)) & (geo_ev == 0) & (dlg_ev == 0)).astype(np.int8),
        })
        rows.append(df_m)
    return pd.concat(rows, ignore_index=True)


class TestBaselineV2(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_spec()

    def test_spec_structure_and_leakage_guards(self) -> None:
        self.assertEqual(len(self.spec["models"]), 6)
        self.assertEqual(self.spec["protocols"]["primary"], "B_rolling_ooc")
        self.assertFalse(self.spec["data_contract"]["resample_labels"])

        with self.assertRaises(ValueError):
            validate_feature_set("bad_model", ["y_any"], False, self.spec)
        with self.assertRaises(ValueError):
            validate_feature_set("bad_lag", ["prior_1m_bought_any"], False, self.spec)
        # Allowed when uses_target_lag=True
        validate_feature_set("L1_ok", ["prior_1m_bought_any"], True, self.spec)

    def test_protocol_b_rolling_schedule(self) -> None:
        val_steps = iter_protocol_steps(self.spec, "B_rolling_ooc", "val")
        test_steps = iter_protocol_steps(self.spec, "B_rolling_ooc", "test")
        self.assertEqual(len(val_steps), 2)
        self.assertEqual(val_steps[0], (["2022-04", "2022-05", "2022-06", "2022-07", "2022-08"], ["2022-09"]))
        self.assertEqual(val_steps[1], (["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09"], ["2022-10"]))
        self.assertEqual(len(test_steps), 3)
        self.assertEqual(test_steps[2][0][-1], "2022-12")
        self.assertEqual(test_steps[2][1], ["2023-01"])

    def test_capped_tenure_preserves_nan(self) -> None:
        df = pd.DataFrame({"trx_all_days_since_first_event": [100.0, 520.0, np.nan]})
        X = extract_feature_matrix(df, ["trx_all_days_since_first_event__cap365"])
        self.assertAlmostEqual(float(X[0, 0]), 100.0)
        self.assertAlmostEqual(float(X[1, 0]), 365.0)
        self.assertTrue(np.isnan(X[2, 0]))

    def test_freeze_lock_and_end_to_end_synthetic_val_then_test(self) -> None:
        panel = _make_synthetic_panel(n_clients=250, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            panel_path = tmp_dir / "panel_moscow.parquet"
            out_dir = tmp_dir / "out_v2"
            panel.to_parquet(panel_path, index=False)

            # Use a tiny CatBoost grid copy of spec so synthetic test finishes in a few seconds
            mini_spec = json.loads(json.dumps(self.spec))
            mini_spec["catboost_tuning"]["max_iterations"] = 60
            mini_spec["catboost_tuning"]["early_stopping_rounds"] = 20
            mini_spec["catboost_tuning"]["grid"] = [
                {"learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 10.0},
                {"learning_rate": 0.08, "depth": 4, "l2_leaf_reg": 15.0},
            ]
            mini_spec_path = tmp_dir / "mini_spec.json"
            mini_spec_path.write_text(json.dumps(mini_spec), encoding="utf-8")

            # Verify Test fails before Val tuning creates tuned_catboost_config.json
            with self.assertRaises(FileNotFoundError):
                run_baseline_v2(
                    phase="test",
                    panel_path=panel_path,
                    out_dir=out_dir,
                    seeds_override=[42],
                    n_bootstrap_override=20,
                    spec_json_path=mini_spec_path,
                )

            # Run Val phase
            res_val = run_baseline_v2(
                phase="val",
                panel_path=panel_path,
                out_dir=out_dir,
                seeds_override=[42],
                include_protocol_a=True,
                thread_count=2,
                n_bootstrap_override=20,
                spec_json_path=mini_spec_path,
            )
            self.assertEqual(res_val["status"], "ready")
            self.assertTrue((out_dir / "tuned_catboost_config.json").exists())
            self.assertTrue((out_dir / "metrics" / "val_ablation_macro.csv").exists())
            self.assertTrue((out_dir / "metrics" / "val_permutation_importance.csv").exists())
            self.assertTrue((out_dir / "metrics" / "val_slice_metrics.csv").exists())
            self.assertTrue((out_dir / "metrics" / "val_shuffled_sanity.csv").exists())

            # Run Test phase (opened once after Val freeze)
            res_test = run_baseline_v2(
                phase="test",
                panel_path=panel_path,
                out_dir=out_dir,
                seeds_override=[42],
                include_protocol_a=True,
                thread_count=2,
                n_bootstrap_override=20,
                spec_json_path=mini_spec_path,
            )
            self.assertEqual(res_test["status"], "ready")
            self.assertTrue((out_dir / "BASELINE_V2_REPORT.md").exists())
            self.assertTrue((out_dir / "metrics" / "paired_protocol_b_vs_a.csv").exists())

            monthly_df = pd.read_csv(out_dir / "metrics" / "monthly_metrics_all.csv")
            # Protocol B has 2 val + 3 test = 5 months x 6 models = 30 rows
            b_rows = monthly_df[monthly_df["protocol"] == "B_rolling_ooc"]
            self.assertEqual(len(b_rows), 30)
            self.assertTrue(b_rows["lift_1pct_ci_low"].notna().all())
            self.assertTrue(b_rows["average_precision_ci_high"].notna().all())

    def test_holm_and_blend(self) -> None:
        adj = holm_bonferroni([0.001, 0.04, 0.03, None])
        self.assertAlmostEqual(float(adj[0]), 0.003)
        self.assertAlmostEqual(float(adj[2]), 0.06)
        self.assertAlmostEqual(float(adj[1]), 0.06)
        self.assertIsNone(adj[3])

        df_a = pd.DataFrame({
            "client_id": ["c1", "c2", "c3"],
            "fold": [0, 0, 0],
            "month": ["2022-09"] * 3,
            "y_true": [0, 1, 0],
            "score": [0.1, 0.8, 0.2],
        })
        df_b = df_a.copy()
        df_b["score"] = [0.2, 0.6, 0.4]
        for method in ("prob_linear", "rank_pct", "logit_linear"):
            out = blend_prediction_frames(df_a, df_b, method=method, weight_logreg=0.5)
            self.assertEqual(len(out), 3)
            self.assertTrue(np.all(np.isfinite(out["score"])))


if __name__ == "__main__":
    unittest.main()
