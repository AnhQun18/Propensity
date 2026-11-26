"""Group 2 & 3 Verification:
1. Group 2.1 (UTC vs MSK Leakage Impact): Build exact 63 TRX features under UTC 00:00 cutoff
   and compare L4_cb & L4_lgbm (Val & Test) against MSK 00:00 cutoff (21:00 UTC prev day).
2. Group 2.4 (FE-v1 Permutation Importance): Fit 5-fold FE-v1 (139 features, seed=42) on Val & Test
   and compute OOF Permutation Importance across all 139 features (3 repeats) to detect any abnormal features.
3. Group 3.4 (Calendar Features on 2023-01 Anomaly): Test adding calendar/holiday features
   (month_num, month_sin, month_cos, is_jan_holiday, biz_days_in_month, prev_month_biz_days)
   to B_catboost_TGD_tuned under Protocol B (Rolling OOC) and check effect on 2023-01 and Val.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline_v2.models import CatBoostTunedModel
from baseline_v2.spec import load_spec
from propensity_model.metrics import compute_metrics, deterministic_tie_hash
from scripts.run_fe_v1_ablation import PCA_CACHE_DIR, ensure_stage0_pca_cached, load_spec as load_fe1_spec

DATA_ROOT = REPO_ROOT / "data" / "MBD-mini"
PANEL_PATH = REPO_ROOT / "outputs" / "baseline_moscow_v1" / "panel_moscow.parquet"
OUT_DIR = REPO_ROOT / "outputs" / "verification_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MODELING_MONTHS = (
    "2022-04", "2022-05", "2022-06", "2022-07", "2022-08",
    "2022-09", "2022-10", "2022-11", "2022-12", "2023-01",
)
import shutil
import subprocess

def _detect_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False

USE_GPU = _detect_gpu()

CB_PARAMS = dict(
    iterations=500,
    random_seed=SEED,
    border_count=128,
    loss_function="Logloss",
    verbose=False,
    allow_writing_files=False,
    **({"task_type": "GPU"} if USE_GPU else {"thread_count": -1}),
)


def build_utc_trx_63() -> pd.DataFrame:
    """Build exact same 63 TRX features as build_moscow_features.py, but using 00:00:00 UTC cutoff."""
    utc_cache = OUT_DIR / "trx_63_utc_cutoff.parquet"
    if utc_cache.exists():
        return pd.read_parquet(utc_cache)

    con = duckdb.connect()
    con.execute("SET memory_limit='2.2GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    tmp_dir = OUT_DIR / "_duck_tmp_utc"
    tmp_dir.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{tmp_dir.as_posix()}'")

    panel_keys = pd.read_parquet(PANEL_PATH, columns=["client_id", "fold", "month", "y_any"])

    month_dfs = []
    for fold in range(5):
        trx_glob = (DATA_ROOT / "detail" / "trx" / f"fold={fold}" / "*.parquet").as_posix()
        for m in MODELING_MONTHS:
            c_utc = f"TIMESTAMP '{m}-01 00:00:00'"
            windows = [
                ("30d", f"event_time >= {c_utc} - INTERVAL 30 DAY"),
                ("90d", f"event_time >= {c_utc} - INTERVAL 90 DAY"),
                ("all", "TRUE"),
            ]
            exprs = ["client_id"]
            for w, cond in windows:
                p = f"trx_{w}_"
                exprs.extend([
                    f"CAST(CASE WHEN COUNT(*) FILTER (WHERE {cond}) > 0 THEN 1 ELSE 0 END AS TINYINT) AS {p}has_event",
                    f"CAST(COUNT(*) FILTER (WHERE {cond}) AS INTEGER) AS {p}event_count",
                    f"CAST(LN(1.0 + COUNT(*) FILTER (WHERE {cond})) AS FLOAT) AS {p}log1p_event_count",
                    f"CAST(COUNT(DISTINCT CAST(event_time AS DATE)) FILTER (WHERE {cond}) AS SMALLINT) AS {p}active_days",
                    f"CAST(EPOCH({c_utc} - MAX(event_time) FILTER (WHERE {cond})) / 86400.0 AS FLOAT) AS {p}days_since_last_event",
                    f"CAST(EPOCH({c_utc} - MIN(event_time) FILTER (WHERE {cond})) / 86400.0 AS FLOAT) AS {p}days_since_first_event",
                    f"CAST(COUNT(*) FILTER (WHERE {cond} AND currency = 11) AS INTEGER) AS {p}main_count",
                    f"CAST(COUNT(*) FILTER (WHERE {cond} AND currency IS NOT NULL AND currency <> 11) AS INTEGER) AS {p}foreign_count",
                    f"CAST(COUNT(*) FILTER (WHERE {cond} AND currency IS NOT NULL AND currency <> 11)::DOUBLE / NULLIF(COUNT(currency) FILTER (WHERE {cond}), 0) AS FLOAT) AS {p}foreign_fraction",
                    f"CAST(COUNT(*) FILTER (WHERE {cond} AND currency = 11 AND amount < 0) AS INTEGER) AS {p}main_neg_count",
                    f"CAST(SUM(ABS(amount)) FILTER (WHERE {cond} AND currency = 11 AND isfinite(amount)) AS DOUBLE) AS {p}main_amount_abs_sum",
                    f"CAST(LN(1.0 + COALESCE(SUM(ABS(amount)) FILTER (WHERE {cond} AND currency = 11 AND isfinite(amount)), 0.0)) AS FLOAT) AS {p}log1p_main_amount_abs_sum",
                    f"CAST(AVG(SIGN(amount) * LN(1.0 + ABS(amount))) FILTER (WHERE {cond} AND currency = 11 AND isfinite(amount)) AS FLOAT) AS {p}main_amount_signed_log_mean",
                    f"CAST(APPROX_QUANTILE(amount, 0.5) FILTER (WHERE {cond} AND currency = 11 AND isfinite(amount)) AS FLOAT) AS {p}main_amount_median",
                    f"CAST(MAX(amount) FILTER (WHERE {cond} AND currency = 11 AND isfinite(amount)) AS FLOAT) AS {p}main_amount_max",
                    f"CAST(COUNT(DISTINCT event_subtype) FILTER (WHERE {cond}) AS SMALLINT) AS {p}event_subtype_nunique",
                    f"CAST(COUNT(DISTINCT dst_type11) FILTER (WHERE {cond}) AS INTEGER) AS {p}dst_type11_nunique",
                    f"CAST(COUNT(DISTINCT dst_type12) FILTER (WHERE {cond}) AS INTEGER) AS {p}dst_type12_nunique",
                ])

            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE t_b AS
                SELECT {', '.join(exprs)}
                FROM read_parquet('{trx_glob}')
                WHERE event_time < {c_utc}
                GROUP BY client_id
            """)
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE t_e AS
                WITH counts AS (
                    SELECT
                        client_id, event_type,
                        COUNT(*) FILTER (WHERE event_time >= {c_utc} - INTERVAL 30 DAY) AS n30,
                        COUNT(*) FILTER (WHERE event_time >= {c_utc} - INTERVAL 90 DAY) AS n90,
                        COUNT(*) AS nall
                    FROM read_parquet('{trx_glob}')
                    WHERE event_time < {c_utc} AND event_type IS NOT NULL
                    GROUP BY 1, 2
                )
                SELECT
                    client_id,
                    CAST(COUNT(*) FILTER (WHERE n30 > 0) AS SMALLINT) AS trx_30d_event_type_nunique,
                    CAST(COUNT(*) FILTER (WHERE n90 > 0) AS SMALLINT) AS trx_90d_event_type_nunique,
                    CAST(COUNT(*) FILTER (WHERE nall > 0) AS SMALLINT) AS trx_all_event_type_nunique,
                    (MAX(n30)::DOUBLE / NULLIF(SUM(n30), 0))::FLOAT AS trx_30d_event_type_top_share,
                    (MAX(n90)::DOUBLE / NULLIF(SUM(n90), 0))::FLOAT AS trx_90d_event_type_top_share,
                    (MAX(nall)::DOUBLE / NULLIF(SUM(nall), 0))::FLOAT AS trx_all_event_type_top_share
                FROM counts
                GROUP BY 1
            """)
            df_m = con.execute(f"""
                SELECT '{m}' AS month, b.*, e.* EXCLUDE (client_id)
                FROM t_b b LEFT JOIN t_e e USING (client_id)
            """).df()
            month_dfs.append(df_m)
        print(f"  [UTC TRX] fold={fold} done", flush=True)

    trx_utc = pd.concat(month_dfs, ignore_index=True)
    merged = panel_keys.merge(trx_utc, on=["month", "client_id"], how="left")
    for w in ("30d", "90d", "all"):
        merged[f"trx_{w}_has_event"] = merged[f"trx_{w}_has_event"].fillna(0).astype(np.int8)
        for col in (f"trx_{w}_event_count", f"trx_{w}_active_days", f"trx_{w}_main_count",
                    f"trx_{w}_foreign_count", f"trx_{w}_main_neg_count"):
            merged[col] = merged[col].fillna(0).astype(np.int32)
        for col in (f"trx_{w}_log1p_event_count", f"trx_{w}_log1p_main_amount_abs_sum"):
            merged[col] = merged[col].fillna(0.0).astype(np.float32)
        merged[f"trx_{w}_main_amount_abs_sum"] = merged[f"trx_{w}_main_amount_abs_sum"].fillna(0.0).astype(np.float64)

    merged["trx_rate_per_30d_tenure"] = np.where(
        merged["trx_all_days_since_first_event"].notna(),
        merged["trx_all_event_count"] / np.maximum(merged["trx_all_days_since_first_event"] / 30.0, 1.0),
        np.nan,
    ).astype(np.float32)
    merged["trx_share_30d_of_90d"] = np.where(
        merged["trx_90d_event_count"] > 0,
        merged["trx_30d_event_count"] / merged["trx_90d_event_count"],
        np.nan,
    ).astype(np.float32)
    merged["trx_first_activated_30d"] = (
        (merged["trx_30d_event_count"] > 0) & (merged["trx_30d_event_count"] == merged["trx_all_event_count"])
    ).astype(np.int8)

    merged.to_parquet(utc_cache, index=False, compression="zstd")
    return merged


def run_utc_vs_msk_l4() -> None:
    print("=== Group 2.1: UTC vs MSK L4_cb & L4_lgbm ===", flush=True)
    utc_df = build_utc_trx_63()
    trx_cols = [c for c in utc_df.columns if c.startswith("trx_")]
    assert len(trx_cols) == 63, f"Expected 63 TRX cols, got {len(trx_cols)}"

    msk_df = pd.read_parquet(PANEL_PATH, columns=["client_id", "fold", "month", "y_any"] + trx_cols)

    rows = []
    for tz_label, df_tz in (("UTC_00", utc_df), ("MSK_00", msk_df)):
        for stage, tr_m, ev_m in (
            ("val", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08"], ["2022-09", "2022-10"]),
            ("test", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10"], ["2022-11", "2022-12", "2023-01"]),
        ):
            df_tr = df_tz[df_tz["month"].isin(tr_m)].reset_index(drop=True)
            df_ev = df_tz[df_tz["month"].isin(ev_m)].reset_index(drop=True)
            for fam in ("lgbm", "catboost"):
                f_dfs = []
                for k in range(5):
                    tr_k = df_tr["fold"].to_numpy() != k
                    ev_k = df_ev["fold"].to_numpy() == k
                    X_tr = df_tr.loc[tr_k, trx_cols].to_numpy(dtype=np.float32)
                    X_ev = df_ev.loc[ev_k, trx_cols].to_numpy(dtype=np.float32)
                    y_tr = df_tr.loc[tr_k, "y_any"].to_numpy(dtype=np.int8)
                    y_ev = df_ev.loc[ev_k, "y_any"].to_numpy(dtype=np.int8)
                    if fam == "lgbm":
                        clf = lgb.LGBMClassifier(n_estimators=500, random_state=SEED, n_jobs=4, verbose=-1)
                    else:
                        clf = CatBoostClassifier(**CB_PARAMS)
                    clf.fit(X_tr, y_tr)
                    sc = clf.predict_proba(X_ev)[:, 1].astype(np.float64)
                    f_dfs.append(pd.DataFrame({
                        "client_id": df_ev.loc[ev_k, "client_id"].to_numpy(),
                        "month": df_ev.loc[ev_k, "month"].to_numpy(),
                        "y_true": y_ev,
                        "score": sc,
                    }))
                pdf = pd.concat(f_dfs, ignore_index=True).sort_values(["month", "client_id"]).reset_index(drop=True)
                mm, _ = compute_metrics(pdf["y_true"].to_numpy(), pdf["score"].to_numpy(), pdf["client_id"].to_numpy(),
                                        budget_fractions=(0.01, 0.05), primary_budget=0.01, seed=SEED, is_probabilistic=True)
                rows.append({
                    "timezone": tz_label, "stage": stage, "family": fam,
                    "pr_auc": mm["average_precision"], "roc_auc": mm["roc_auc"],
                    "lift_1pct": mm["lift_1pct"], "hits_1pct": mm["hits_1pct"],
                    "lift_5pct": mm["lift_5pct"], "log_loss": mm["log_loss"],
                })
                print(f"  [{tz_label} | {stage:4s} | {fam:8s}] PR-AUC={mm['average_precision']:.5f} | Lift@1%={mm['lift_1pct']:.2f}x", flush=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "group2_utc_vs_msk_l4.csv", index=False)


def run_fe_v1_permutation_importance() -> None:
    """Group 2.4: Full 139-feature OOF Permutation Importance on FE-v1 (Val & Test, 3 repeats)."""
    print("=== Group 2.4: FE-v1 139-feature OOF Permutation Importance ===", flush=True)
    spec = load_fe1_spec()
    ensure_stage0_pca_cached(spec)
    # Load only tab_cols + keys to minimize peak RSS so CatBoost runs at full speed
    panel_schema = pd.read_parquet(PANEL_PATH, columns=["client_id", "fold", "month", "y_any"])
    import pyarrow.parquet as pq
    pcols = pq.ParquetFile(PANEL_PATH).schema_arrow.names
    trx_63 = [c for c in pcols if c.startswith("trx_")]
    geo_meta = [c for c in pcols if c.startswith("geo_") and ("geohash" not in c)]
    geo_gh = [c for c in pcols if c.startswith("geo_") and ("geohash" in c)]
    dlg_18 = [c for c in pcols if c.startswith("dialog_")]
    tab_cols = trx_63 + geo_meta + geo_gh + dlg_18 + ["cold_start"]
    panel = pd.read_parquet(PANEL_PATH, columns=["client_id", "fold", "month", "y_any"] + tab_cols)
    del panel_schema
    emb_names = [f"dlg_pca_all_{i}" for i in range(16)] + [f"dlg_pca_90d_{i}" for i in range(8)] + ["dlg_norm_all", "dlg_norm_90d", "dlg_cos_90d_all"]
    all_names = tab_cols + emb_names

    pi_rows = []
    for stage, tr_m, ev_m in (
        ("val", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08"], ["2022-09", "2022-10"]),
        ("test", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10"], ["2022-11", "2022-12", "2023-01"]),
    ):
        df_tr = panel[panel["month"].isin(tr_m)].reset_index(drop=True)
        df_ev = panel[panel["month"].isin(ev_m)].sort_values(["month", "client_id"]).reset_index(drop=True)

        X_ev_full = np.empty((len(df_ev), len(all_names)), dtype=np.float32)
        for j, c in enumerate(tab_cols):
            X_ev_full[:, j] = df_ev[c].to_numpy(dtype=np.float32, copy=False)

        df_ev_orig = panel[panel["month"].isin(ev_m)].reset_index(drop=True)
        assert np.array_equal(df_ev["client_id"].to_numpy(), df_ev_orig["client_id"].to_numpy())
        assert np.array_equal(df_ev["month"].to_numpy(), df_ev_orig["month"].to_numpy())
        del df_ev_orig

        fold_models = []
        fold_arr_ev = df_ev["fold"].to_numpy()
        y_ev = df_ev["y_any"].to_numpy(dtype=np.int8)
        c_ev = df_ev["client_id"].to_numpy()
        del df_ev
        gc.collect()
        for k in range(5):
            tr_k = df_tr["fold"].to_numpy() != k
            ev_k = fold_arr_ev == k
            X_tr = np.empty((int(tr_k.sum()), len(all_names)), dtype=np.float32)
            for j, c in enumerate(tab_cols):
                X_tr[:, j] = df_tr.loc[tr_k, c].to_numpy(dtype=np.float32, copy=False)
            F_tr = np.load(PCA_CACHE_DIR / stage / f"f{k}_tr.npy")
            F_ev = np.load(PCA_CACHE_DIR / stage / f"f{k}_ev.npy")
            X_tr[:, len(tab_cols):] = F_tr
            X_ev_full[ev_k, len(tab_cols):] = F_ev
            y_tr = df_tr.loc[tr_k, "y_any"].to_numpy(dtype=np.int8)
            clf = CatBoostClassifier(**CB_PARAMS)
            clf.fit(X_tr, y_tr)
            fold_models.append(clf)
            del X_tr, F_tr, F_ev
            gc.collect()
        del df_tr
        gc.collect()

        tie_h = deterministic_tie_hash(c_ev, seed=SEED)
        base_scores = np.empty(len(c_ev), dtype=np.float64)
        for k in range(5):
            mk = fold_arr_ev == k
            base_scores[mk] = fold_models[k].predict_proba(X_ev_full[mk])[:, 1]
        base_m, _ = compute_metrics(y_ev, base_scores, c_ev, budget_fractions=(0.01,), primary_budget=0.01, seed=SEED, is_probabilistic=True, tie_hashes=tie_h)
        print(f"  [PI FE-v1 {stage.upper()}] base PR-AUC={base_m['average_precision']:.5f} | Lift@1%={base_m['lift_1pct']:.2f}x", flush=True)

        n_rep = 2
        for j, fname in enumerate(all_names):
            d_aps, d_l1s, d_aucs = [], [], []
            orig_col = X_ev_full[:, j].copy()
            for r in range(n_rep):
                rng = np.random.default_rng(12000 + j * 10 + r)
                perm_sc = np.empty(len(c_ev), dtype=np.float64)
                for k in range(5):
                    mk = fold_arr_ev == k
                    Xk = X_ev_full[mk].copy()
                    Xk[:, j] = rng.permutation(Xk[:, j])
                    perm_sc[mk] = fold_models[k].predict_proba(Xk)[:, 1]
                pm, _ = compute_metrics(y_ev, perm_sc, c_ev, budget_fractions=(0.01,), primary_budget=0.01, seed=SEED, is_probabilistic=True, tie_hashes=tie_h)
                d_aps.append(float(base_m["average_precision"] - pm["average_precision"]))
                d_l1s.append(float(base_m["lift_1pct"] - pm["lift_1pct"]))
                d_aucs.append(float(base_m["roc_auc"] - pm["roc_auc"]))
            X_ev_full[:, j] = orig_col
            pi_rows.append({
                "stage": stage, "feature": fname,
                "base_pr_auc": float(base_m["average_precision"]),
                "base_lift_1pct": float(base_m["lift_1pct"]),
                "perm_drop_pr_auc_mean": float(np.mean(d_aps)),
                "perm_drop_pr_auc_sd": float(np.std(d_aps, ddof=1)),
                "perm_drop_lift_1pct_mean": float(np.mean(d_l1s)),
                "perm_drop_roc_auc_mean": float(np.mean(d_aucs)),
            })
        del fold_models, X_ev_full
        gc.collect()

    pi_df = pd.DataFrame(pi_rows).sort_values(["stage", "perm_drop_pr_auc_mean"], ascending=[True, False])
    pi_df.to_csv(OUT_DIR / "group2_fe_v1_permutation_importance_139.csv", index=False)


def run_calendar_feature_test() -> None:
    """Group 3.4: Test adding calendar/holiday features to B_catboost_TGD_tuned under Protocol B."""
    print("=== Group 3.4: Calendar Feature Test for 2023-01 Anomaly (Protocol B) ===", flush=True)
    spec = load_spec(REPO_ROOT / "configs" / "baseline_v2_evaluation_spec.json")
    tuned = json.loads((REPO_ROOT / "outputs" / "baseline_v2" / "tuned_catboost_config.json").read_text(encoding="utf-8"))
    base_14 = list(spec["feature_sets"]["raw_TGD_14"])

    # Calendar lookup for 2022-04..2023-01 (Russian calendar: Jan 1-8 are New Year/Orthodox Christmas holidays)
    cal_map = {
        "2022-04": (4, 30, 21, 0, 22),
        "2022-05": (5, 31, 18, 0, 21),
        "2022-06": (6, 30, 21, 0, 18),
        "2022-07": (7, 31, 21, 0, 21),
        "2022-08": (8, 31, 23, 0, 21),
        "2022-09": (9, 30, 22, 0, 23),
        "2022-10": (10, 31, 21, 0, 22),
        "2022-11": (11, 30, 21, 0, 21),
        "2022-12": (12, 31, 22, 1, 21),
        "2023-01": (1, 31, 17, 1, 22),
    }
    panel = pd.read_parquet(PANEL_PATH, columns=["client_id", "fold", "month", "y_any"] + base_14)
    panel["cal_month_sin"] = panel["month"].map(lambda m: np.sin(2 * np.pi * cal_map[m][0] / 12.0)).astype(np.float32)
    panel["cal_month_cos"] = panel["month"].map(lambda m: np.cos(2 * np.pi * cal_map[m][0] / 12.0)).astype(np.float32)
    panel["cal_biz_days"] = panel["month"].map(lambda m: cal_map[m][2]).astype(np.float32)
    panel["cal_is_jan_holiday"] = panel["month"].map(lambda m: 1 if m == "2023-01" else 0).astype(np.float32)
    panel["cal_prev_biz_days"] = panel["month"].map(lambda m: cal_map[m][4]).astype(np.float32)

    cal_cols = base_14 + ["cal_month_sin", "cal_month_cos", "cal_biz_days", "cal_is_jan_holiday", "cal_prev_biz_days"]

    rows = []
    for label, fcols in (("base_14", base_14), ("base_14_plus_calendar_19", cal_cols)):
        for stage, sched in (("val", spec["protocols"]["B_rolling_ooc"]["val_schedule"]), ("test", spec["protocols"]["B_rolling_ooc"]["test_schedule"])):
            for step in sched:
                ev_m = step["eval_month"]
                tr_ms = step["train_months"]
                df_tr_core = panel[panel["month"].isin(tr_ms[:-1])].reset_index(drop=True)
                df_tr_es = panel[panel["month"].isin([tr_ms[-1]])].reset_index(drop=True)
                df_ev = panel[panel["month"] == ev_m].reset_index(drop=True)
                f_dfs = []
                for k in range(5):
                    tr_c = df_tr_core[df_tr_core["fold"] != k]
                    tr_e = df_tr_es[df_tr_es["fold"] != k]
                    ev_s = df_ev[df_ev["fold"] == k]
                    m = CatBoostTunedModel(
                        seed=SEED, learning_rate=tuned["learning_rate"], depth=tuned["depth"],
                        l2_leaf_reg=tuned["l2_leaf_reg"], max_iterations=tuned["max_iterations"],
                        early_stopping_rounds=tuned["early_stopping_rounds"], thread_count=4,
                    )
                    m.fit(
                        tr_c[fcols].to_numpy(dtype=np.float32), tr_c["y_any"].to_numpy(dtype=np.int8),
                        X_es=tr_e[fcols].to_numpy(dtype=np.float32), y_es=tr_e["y_any"].to_numpy(dtype=np.int8),
                    )
                    sc = m.predict_scores(ev_s[fcols].to_numpy(dtype=np.float32))
                    f_dfs.append(pd.DataFrame({"client_id": ev_s["client_id"].to_numpy(), "y_true": ev_s["y_any"].to_numpy(), "score": sc}))
                pdf = pd.concat(f_dfs, ignore_index=True)
                mm, _ = compute_metrics(pdf["y_true"].to_numpy(), pdf["score"].to_numpy(), pdf["client_id"].to_numpy(),
                                        budget_fractions=(0.01, 0.05), primary_budget=0.01, seed=SEED, is_probabilistic=True)
                rows.append({
                    "variant": label, "stage": stage, "month": ev_m,
                    "pr_auc": mm["average_precision"], "roc_auc": mm["roc_auc"],
                    "lift_1pct": mm["lift_1pct"], "hits_1pct": mm["hits_1pct"], "log_loss": mm["log_loss"],
                })
                print(f"  [{label:24s} | {ev_m}] PR-AUC={mm['average_precision']:.5f} | Lift@1%={mm['lift_1pct']:.2f}x | LogLoss={mm['log_loss']:.5f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "group3_calendar_feature_test.csv", index=False)


def main():
    if not (OUT_DIR / "group2_utc_vs_msk_l4.csv").exists():
        run_utc_vs_msk_l4()
    else:
        print("[SKIP] group2_utc_vs_msk_l4.csv already exists.", flush=True)
    if not (OUT_DIR / "group3_calendar_feature_test.csv").exists():
        run_calendar_feature_test()
    else:
        print("[SKIP] group3_calendar_feature_test.csv already exists.", flush=True)
    if not (OUT_DIR / "group2_fe_v1_permutation_importance_139.csv").exists():
        run_fe_v1_permutation_importance()
    else:
        print("[SKIP] group2_fe_v1_permutation_importance_139.csv already exists.", flush=True)
    print("ALL GROUP 2 & 3 CHECKS DONE.", flush=True)


if __name__ == "__main__":
    main()
