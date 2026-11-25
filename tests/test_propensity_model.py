"""Synthetic unit and end-to-end smoke tests for propensity_model."""
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

from propensity_data.config import PreparationConfig
from propensity_data.protocol import evaluation_protocol
from propensity_data.schema import feature_schema
from propensity_model.bootstrap import (
    _prepare_sorted_arrays,
    _weighted_metrics_from_sorted,
    paired_cluster_bootstrap,
)
from propensity_model.config import (
    BaselineConfig,
    ExperimentSpec,
    FoldSpec,
    assert_temporal_embargo,
    load_config,
)
from propensity_model.data import load_fold_arrays, resolve_feature_sets
from propensity_model.features import (
    DERIVED_BEHAVIORAL_FEATURES,
    PRIOR_PURCHASE_FEATURES,
    TOP50_FEATURES,
    TOP50_NO_PRIOR_FEATURES,
    TOP50_TRX_FEATURES,
    build_enriched_panel,
)
from propensity_model.metrics import compute_metrics, deterministic_rank_order, deterministic_tie_hash
from propensity_model.models import ColdStartFallbackWrapper, ProbabilityCalibrator, RidgeLogisticBaseline
from propensity_model.runner import run_baseline

MONTHS_12 = [f"2022-{m:02d}" for m in range(2, 13)] + ["2023-01"]


def _build_synthetic_processed(root: Path, n_clients: int = 80, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    prep_cfg = PreparationConfig(root / "raw", root / "prepared")
    schema = feature_schema(prep_cfg)
    protocol = evaluation_protocol(prep_cfg, MONTHS_12)

    proc = root / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    clients = [f"c_{i:04d}" for i in range(n_clients)]
    rows = []
    # Give first 10 clients deterministic purchase pattern for point-in-time tests
    for m_idx, month in enumerate(MONTHS_12):
        y, mo = map(int, month.split("-"))
        cutoff = datetime(y, mo, 1, 0, 0, 0)
        for c_idx, cid in enumerate(clients):
            rec = {"client_id": cid, "cutoff": cutoff, "supervised_eligible": True}
            # Deterministic targets for c_0000: buys t1 in 2022-02, t2 in 2022-04, t3 in 2022-06, nothing else
            if cid == "c_0000":
                t1 = 1.0 if month == "2022-02" else 0.0
                t2 = 1.0 if month == "2022-04" else 0.0
                t3 = 1.0 if month == "2022-06" else 0.0
                t4 = 0.0
            else:
                signal = 0.25 if c_idx % 7 == 0 else 0.04
                t1 = float(rng.random() < signal * 0.5)
                t2 = float(rng.random() < signal * 0.2)
                t3 = float(rng.random() < signal * 0.3)
                t4 = float(rng.random() < signal * 0.2)
            y_any = int((t1 + t2 + t3 + t4) > 0)
            rec.update(y_any=y_any, target_1=t1, target_2=t2, target_3=t3, target_4=t4)

            is_cold = int(c_idx % 15 == 0)
            has_trx = 0 if (is_cold or c_idx % 9 == 0) else 1
            has_geo = 0 if (is_cold or c_idx % 4 == 0) else 1
            has_dlg = 0 if (is_cold or c_idx % 5 != 0) else 1

            for col in schema["predictors"]:
                dtype = schema["dtypes"][col]
                if col == "cold_start":
                    rec[col] = is_cold
                elif col.endswith("_has_event") or col.endswith("_ever_seen_source") or col == "dialog_has_dialog":
                    if col.startswith("trx_"):
                        rec[col] = has_trx
                    elif col.startswith("geo_"):
                        rec[col] = has_geo
                    else:
                        rec[col] = has_dlg
                elif col.endswith("_history_truncated") or col.endswith("_has_foreign"):
                    rec[col] = 0
                elif col.endswith("_days_since_last_event") or col.endswith("_days_since_first_recorded_event"):
                    if (col.startswith("trx_") and not has_trx) or (col.startswith("geo_") and not has_geo) or (col.startswith("dialog_") and not has_dlg):
                        rec[col] = np.nan
                    else:
                        rec[col] = float(rng.uniform(1.0, 90.0)) if "last" in col else float(rng.uniform(100.0, 450.0))
                elif dtype == "DOUBLE":
                    rec[col] = float(rng.uniform(0.0, 10.0))
                else:
                    # BIGINT or TINYINT counts
                    base_cnt = int(rng.integers(1, 25)) if has_trx else 0
                    if col.startswith("geo_"):
                        base_cnt = int(rng.integers(1, 10)) if has_geo else 0
                    elif col.startswith("dialog_"):
                        base_cnt = int(rng.integers(1, 5)) if has_dlg else 0
                    rec[col] = base_cnt
            rows.append(rec)

    df = pd.DataFrame(rows)
    arrow_types = {
        "VARCHAR": pa.string(),
        "TIMESTAMP": pa.timestamp("us"),
        "TINYINT": pa.int8(),
        "BIGINT": pa.int64(),
        "DOUBLE": pa.float64(),
        "BOOLEAN": pa.bool_(),
    }
    pa_schema = pa.schema([(c, arrow_types[schema["dtypes"][c]]) for c in schema["columns"]])
    table = pa.Table.from_pandas(df[schema["columns"]], schema=pa_schema, preserve_index=False)
    pq.write_table(table, proc / "propensity.parquet", compression="zstd")

    (proc / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (proc / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    manifest = {
        "status": "ready",
        "schema_version": 2,
        "dataset_sha256": "synthetic_sha256",
        "rows": len(df),
        "predictors": len(schema["predictors"]),
    }
    (proc / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return proc


class EmbargoAndFeatureTests(unittest.TestCase):
    def test_temporal_embargo_guards(self):
        burn = ("2022-02", "2022-03", "2022-04")
        # Valid H=1
        assert_temporal_embargo(("2022-05", "2022-06"), "2022-07", 1, burn)
        # Invalid H=1: same month or future
        with self.assertRaisesRegex(ValueError, "Temporal embargo violation"):
            assert_temporal_embargo(("2022-05", "2022-07"), "2022-07", 1, burn)
        # Invalid burn-in leak
        with self.assertRaisesRegex(ValueError, "burn-in"):
            assert_temporal_embargo(("2022-04", "2022-05"), "2022-06", 1, burn)
        # Valid H=3: 2-month embargo (train <= eval - 3)
        assert_temporal_embargo(("2022-05", "2022-08"), "2022-11", 3, burn)
        # Invalid H=3: train 2022-09 or 2022-10 overlaps 2022-11 target
        for bad in ("2022-09", "2022-10"):
            with self.assertRaisesRegex(ValueError, "Temporal embargo violation"):
                assert_temporal_embargo(("2022-05", bad), "2022-11", 3, burn)

    def test_feature_set_counts_and_whitelist(self):
        cfg = PreparationConfig(Path("raw"), Path("prep"))
        schema = feature_schema(cfg)
        sets = resolve_feature_sets(schema)
        self.assertEqual(len(sets["trx_84d"]), 49)
        self.assertEqual(len(sets["trx"]), 148)
        self.assertEqual(len(sets["trx_meta"]), 199)
        self.assertEqual(len(sets["all_221"]), 221)
        self.assertEqual(len(sets["all_prior"]), 238)
        self.assertEqual(len(sets["all_prior_derived"]), 252)
        self.assertEqual(len(sets["top50"]), 50)
        self.assertEqual(len(sets["top50_trx_only"]), 23)
        self.assertEqual(len(sets["top50_no_prior"]), 39)
        self.assertEqual(len(PRIOR_PURCHASE_FEATURES), 17)
        self.assertEqual(len(DERIVED_BEHAVIORAL_FEATURES), 14)
        self.assertEqual(len(TOP50_FEATURES), 50)
        self.assertEqual(len(TOP50_TRX_FEATURES), 23)
        self.assertEqual(len(TOP50_NO_PRIOR_FEATURES), 39)

    def test_repo_configs_parse_cleanly(self):
        repo_root = Path(__file__).resolve().parents[1]
        for fname in ("baseline_top50.json", "baseline_full252.json"):
            cfg = load_config(repo_root / "configs" / fname)
            self.assertEqual(cfg.burn_in_months, ("2022-02", "2022-03", "2022-04"))


class MetricsBootstrapAndModelTests(unittest.TestCase):
    def test_deterministic_tie_break_invariant_to_row_order(self):
        cids = np.array([f"u_{i}" for i in range(50)])
        scores = np.full(50, 0.05)
        order1 = deterministic_rank_order(scores, cids, seed=42)
        top1 = list(cids[order1[:10]])

        perm = np.random.default_rng(123).permutation(50)
        order2 = deterministic_rank_order(scores[perm], cids[perm], seed=42)
        top2 = list(cids[perm][order2[:10]])
        self.assertEqual(top1, top2)

    def test_zero_positive_slice_returns_none(self):
        y = np.zeros(100, dtype=np.int8)
        s = np.linspace(0.01, 0.2, 100)
        cids = [f"u_{i}" for i in range(100)]
        m, _ = compute_metrics(y, s, cids, is_probabilistic=True)
        self.assertIsNone(m["roc_auc"])
        self.assertIsNone(m["average_precision"])
        self.assertIsNone(m["recall_1pct"])
        self.assertIsNone(m["lift_1pct"])
        self.assertIsNotNone(m["log_loss"])

    def test_weighted_bootstrap_matches_sklearn_exact(self):
        rng = np.random.default_rng(99)
        n = 300
        y = (rng.random(n) < 0.15).astype(np.int8)
        scores = np.round(rng.uniform(0.01, 0.99, size=n), 2)  # deliberate ties!
        cids = np.array([f"c_{i % 60}" for i in range(n)])
        codes, uniques = pd.factorize(cids, sort=True)
        cw = rng.integers(0, 4, size=len(uniques))
        rw = cw[codes].astype(np.float64)

        hashes = deterministic_tie_hash(cids, seed=42)
        prep = _prepare_sorted_arrays(y, scores, codes, hashes, is_probabilistic=True)
        fast = _weighted_metrics_from_sorted(prep, cw, primary_budget=0.05)

        sk_auc = roc_auc_score(y, scores, sample_weight=rw)
        sk_ap = average_precision_score(y, scores, sample_weight=rw)
        self.assertAlmostEqual(fast["roc_auc"], sk_auc, places=10)
        self.assertAlmostEqual(fast["average_precision"], sk_ap, places=10)

    def test_paired_bootstrap_identical_models_zero_delta(self):
        rng = np.random.default_rng(11)
        df = pd.DataFrame(
            {
                "client_id": [f"c_{i % 40}" for i in range(120)],
                "month": [f"2022-{9 + (i // 40):02d}" for i in range(120)],
                "y_true": (rng.random(120) < 0.2).astype(np.int8),
                "score": rng.uniform(0.01, 0.9, size=120),
            }
        )
        res = paired_cluster_bootstrap(df, df, True, True, 0.05, n_replicates=50, seed=42)
        self.assertAlmostEqual(res["delta_roc_auc"], 0.0, places=12)
        self.assertAlmostEqual(res["delta_average_precision"], 0.0, places=12)
        self.assertAlmostEqual(res["delta_log_loss"], 0.0, places=12)

    def test_calibrator_preserves_ranking_and_reduces_drift_logloss(self):
        rng = np.random.default_rng(5)
        raw = rng.uniform(0.05, 0.50, size=500)
        # True event rate is much lower (temporal downward drift)
        y = (rng.random(500) < raw * 0.25).astype(np.int8)
        for method in ("intercept_shift", "platt"):
            cal = ProbabilityCalibrator(method).fit(raw, y)
            adj = cal.transform(raw)
            # Strictly monotone: exact same ROC-AUC
            self.assertAlmostEqual(roc_auc_score(y, raw), roc_auc_score(y, adj), places=12)
            # Mean predicted matches lower observed prevalence much closer
            self.assertLess(abs(adj.mean() - y.mean()), abs(raw.mean() - y.mean()))

    def test_cold_start_shrinkage_formula_exact(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(200, 4)).astype(np.float32)
        cs = np.zeros(200, dtype=np.int8)
        cs[:20] = 1
        y = np.zeros(200, dtype=np.int8)
        y[:4] = 1   # 4 positives out of 20 cold_start
        y[20:40] = 1  # 20 positives out of 180 non-cold
        wrapper = ColdStartFallbackWrapper(
            params={"iterations": 5, "depth": 2, "thread_count": 1}, alpha=20.0, seed=42
        )
        wrapper.fit(X, y, cold_start=cs)
        expected = (4 + 20.0 * (24 / 200)) / (20 + 20.0)
        self.assertAlmostEqual(wrapper.cold_start_prob_, expected, places=10)
        preds = wrapper.predict_scores(X[:5], cold_start=np.ones(5, dtype=np.int8))
        np.testing.assert_allclose(preds, expected, atol=1e-10)

    def test_ridge_logreg_handles_all_nan_column_train_only(self):
        X_tr = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan], [4.0, np.nan]], dtype=np.float32)
        y_tr = np.array([0, 0, 1, 1], dtype=np.int8)
        X_ev = np.array([[1.5, 99.0], [3.5, np.nan]], dtype=np.float32)
        m = RidgeLogisticBaseline(c=1.0, max_iter=100, seed=42).fit(X_tr, y_tr)
        p = m.predict_scores(X_ev)
        self.assertTrue(np.all(np.isfinite(p)))


class SyntheticEndToEndTests(unittest.TestCase):
    def test_point_in_time_enrichment_and_full_pipeline_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc = _build_synthetic_processed(root, n_clients=80, seed=7)
            enriched = root / "enriched.parquet"
            build_enriched_panel(proc / "propensity.parquet", enriched, ("2022-02", "2022-03", "2022-04"))
            df = pd.read_parquet(enriched)

            # Verify burn-in excluded and exact point-in-time prior features for c_0000
            # c_0000 bought t1 in 2022-02, t2 in 2022-04, t3 in 2022-06
            self.assertNotIn("2022-02", set(df["month"]))
            self.assertNotIn("2022-04", set(df["month"]))
            c0 = df[df["client_id"] == "c_0000"].set_index("month")

            # At 2022-05: prior months = 02, 03, 04 (3 months); bought t1 (02) & t2 (04) -> 2 buy months, recency=1
            self.assertEqual(int(c0.loc["2022-05", "prior_label_months_observed"]), 3)
            self.assertEqual(int(c0.loc["2022-05", "prior_3m_buy_months"]), 2)
            self.assertEqual(int(c0.loc["2022-05", "prior_1m_bought_any"]), 1)
            self.assertEqual(float(c0.loc["2022-05", "prior_months_since_last_buy"]), 1.0)
            self.assertEqual(int(c0.loc["2022-05", "prior_all_bought_p1"]), 1)
            self.assertEqual(int(c0.loc["2022-05", "prior_all_bought_p2"]), 1)
            self.assertEqual(int(c0.loc["2022-05", "prior_all_bought_p3"]), 0)
            # H=1 at 2022-05 is 0, but H=3 covers 05, 06, 07 -> includes 2022-06 buy -> y_h3 == 1!
            self.assertEqual(int(c0.loc["2022-05", "y_h1"]), 0)
            self.assertEqual(int(c0.loc["2022-05", "y_h3"]), 1)

            # At 2022-06: t3 happens IN 2022-06 -> must NOT appear in prior features at 2022-06!
            self.assertEqual(int(c0.loc["2022-06", "prior_all_bought_p3"]), 0)
            self.assertEqual(int(c0.loc["2022-06", "y_h1"]), 1)
            # At 2022-07: now 2022-06 buy IS strictly prior!
            self.assertEqual(int(c0.loc["2022-07", "prior_all_bought_p3"]), 1)
            self.assertEqual(int(c0.loc["2022-07", "prior_1m_bought_any"]), 1)
            self.assertEqual(float(c0.loc["2022-07", "prior_months_since_last_buy"]), 1.0)
            # 2022-12 and 2023-01 have NULL y_h3
            self.assertTrue(pd.isna(c0.loc["2022-12", "y_h3"]))
            self.assertTrue(pd.isna(c0.loc["2023-01", "y_h3"]))

            # Forbidden predictor guard in load_fold_arrays
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_fold_arrays(enriched, ("2022-05",), "2022-06", 1, ("target_1",), ("2022-02", "2022-03", "2022-04"))

            # Full pipeline smoke run
            out_dir = root / "smoke_out"
            cfg = BaselineConfig(
                run_name="smoke_test",
                processed_dir=proc,
                output_dir=out_dir,
                burn_in_months=("2022-02", "2022-03", "2022-04"),
                h1_validation_folds=(
                    FoldSpec("h1_09", ("2022-05", "2022-06", "2022-07", "2022-08"), "2022-09", 1),
                    FoldSpec("h1_11", ("2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10"), "2022-11", 1),
                ),
                h3_validation_folds=(
                    FoldSpec("h3_08", ("2022-05",), "2022-08", 3),
                    FoldSpec("h3_10", ("2022-05", "2022-06", "2022-07"), "2022-10", 3),
                ),
                experiments=(
                    ExperimentSpec("B0_prevalence", "prevalence", "none"),
                    ExperimentSpec("B1_heuristic_rf", "heuristic_rf", "heuristic_rf"),
                    ExperimentSpec("B2_logreg_top50", "ridge_logreg", "top50"),
                    ExperimentSpec("M3_cb_top50", "catboost", "top50"),
                    ExperimentSpec("M3_cb_top50_cold_fallback", "catboost_cold_fallback", "top50"),
                ),
                paired_comparisons=(("B2_logreg_top50", "M3_cb_top50"), ("M3_cb_top50", "M3_cb_top50_cold_fallback")),
                bootstrap_replicates=25,
                catboost_params={"iterations": 10, "depth": 3, "learning_rate": 0.1, "thread_count": 1},
                learning_curve_checkpoints=(5, 10),
            )
            manifest_path = run_baseline(cfg)
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(man["status"], "ready")
            self.assertTrue((out_dir / "metrics" / "bootstrap_summary.csv").exists())
            self.assertTrue((out_dir / "metrics" / "paired_bootstrap.csv").exists())
            self.assertTrue((out_dir / "metrics" / "final_holdout_metrics.csv").exists())
            self.assertTrue((out_dir / "figures" / "01_ablation_ladder_h1_vs_h3.png").exists())
            # Must refuse to overwrite existing output directory
            with self.assertRaises(FileExistsError):
                run_baseline(cfg)


if __name__ == "__main__":
    unittest.main()
