"""Group 1 Dialog Verification (Part B — Model Experiments):
1. Dialog Lag Buffer: Re-run L5c_cb & FE-v1 with dialog lagged by 7, 14, 30 days
   (event_time_msk < cutoff_msk - INTERVAL X DAY), compared against L5b_cb (0 dialog) and 0-day lag.
2. Target Shift (T+2): Predict y_any(M+1) from features at cutoff M (so a full 1-month gap exists
   between last dialog event < M-01 and target month M+1), comparing L5b_cb vs L5c_cb vs FE-v1.
3. Activity vs Content Ablations:
   - FE-v1_pca_only (136 cols: drop norm_all, norm_90d, cosine_sim)
   - FE-v1_placebo_emb_shuffle (139 cols: shuffle 27 emb cols among dialog-active rows within month/fold)
4. Evaluate all variants on Val and Test, plus per-embedding-cluster breakdown (Test 1.4).

Runs on seed=42 with exact CatBoost default params from fe_v1_evaluation_spec.json.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from propensity_model.metrics import compute_metrics, deterministic_tie_hash

DATA_ROOT = REPO_ROOT / "data" / "MBD-mini"
MOSCOW_DIR = REPO_ROOT / "outputs" / "baseline_moscow_v1"
PANEL_PATH = MOSCOW_DIR / "panel_moscow.parquet"
EMB_PATH = MOSCOW_DIR / "dialog_emb_moscow.npz"
OUT_DIR = REPO_ROOT / "outputs" / "verification_audit"
PRED_OUT = OUT_DIR / "group1_preds"
PRED_OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
PCA_ALL_DIM = 16
PCA_90D_DIM = 8
MODELING_MONTHS = (
    "2022-04", "2022-05", "2022-06", "2022-07", "2022-08",
    "2022-09", "2022-10", "2022-11", "2022-12", "2023-01",
)
CB_PARAMS = dict(
    iterations=500,
    random_seed=SEED,
    thread_count=4,
    border_count=128,
    used_ram_limit="1.5gb",
    loss_function="Logloss",
    verbose=False,
    allow_writing_files=False,
)


def build_lagged_dialog_meta_and_emb(lag_days: int) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Aggregate dialog metadata and 768d mean-pool embeddings strictly < cutoff_msk - lag_days."""
    con = duckdb.connect()
    con.execute("SET memory_limit='2.5GB'")
    con.execute("SET threads=4")
    dialog_glob = (DATA_ROOT / "detail" / "dialog" / "*" / "*.parquet").as_posix()

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE dlg_raw AS
        SELECT
            client_id,
            event_time + INTERVAL 3 HOUR AS event_time_msk,
            embedding
        FROM read_parquet('{dialog_glob}')
    """)

    meta_frames = []
    all_months_l, all_clients_l, all_emb_l = [], [], []
    w90_months_l, w90_clients_l, w90_emb_l = [], [], []

    for m in MODELING_MONTHS:
        cutoff_str = f"{m}-01 00:00:00"
        c_eff = f"(TIMESTAMP '{cutoff_str}' - INTERVAL {lag_days} DAY)"
        c_orig = f"TIMESTAMP '{cutoff_str}'"

        # Metadata
        df_m = con.execute(f"""
            SELECT
                client_id,
                '{m}' AS month,
                CAST(COUNT(*) > 0 AS TINYINT) AS dialog_30d_has_event_lag,
                CAST(COUNT(*) FILTER (WHERE event_time_msk >= {c_eff} - INTERVAL 30 DAY) AS INTEGER) AS dialog_30d_event_count,
                CAST(LN(1.0 + COUNT(*) FILTER (WHERE event_time_msk >= {c_eff} - INTERVAL 30 DAY)) AS FLOAT) AS dialog_30d_log1p_event_count,
                CAST(COUNT(DISTINCT CAST(event_time_msk AS DATE)) FILTER (WHERE event_time_msk >= {c_eff} - INTERVAL 30 DAY) AS SMALLINT) AS dialog_30d_active_days,
                CAST(EPOCH({c_orig} - MAX(event_time_msk) FILTER (WHERE event_time_msk >= {c_eff} - INTERVAL 30 DAY)) / 86400.0 AS FLOAT) AS dialog_30d_days_since_last_event,
                CAST(EPOCH({c_orig} - MIN(event_time_msk) FILTER (WHERE event_time_msk >= {c_eff} - INTERVAL 30 DAY)) / 86400.0 AS FLOAT) AS dialog_30d_days_since_first_event,

                CAST(COUNT(*) FILTER (WHERE event_time_msk >= {c_eff} - INTERVAL 90 DAY) > 0 AS TINYINT) AS dialog_90d_has_event,
                CAST(COUNT(*) FILTER (WHERE event_time_msk >= {c_eff} - INTERVAL 90 DAY) AS INTEGER) AS dialog_90d_event_count,
                CAST(LN(1.0 + COUNT(*) FILTER (WHERE event_time_msk >= {c_eff} - INTERVAL 90 DAY)) AS FLOAT) AS dialog_90d_log1p_event_count,
                CAST(COUNT(DISTINCT CAST(event_time_msk AS DATE)) FILTER (WHERE event_time_msk >= {c_eff} - INTERVAL 90 DAY) AS SMALLINT) AS dialog_90d_active_days,
                CAST(EPOCH({c_orig} - MAX(event_time_msk) FILTER (WHERE event_time_msk >= {c_eff} - INTERVAL 90 DAY)) / 86400.0 AS FLOAT) AS dialog_90d_days_since_last_event,
                CAST(EPOCH({c_orig} - MIN(event_time_msk) FILTER (WHERE event_time_msk >= {c_eff} - INTERVAL 90 DAY)) / 86400.0 AS FLOAT) AS dialog_90d_days_since_first_event,

                CAST(COUNT(*) > 0 AS TINYINT) AS dialog_all_has_event,
                CAST(COUNT(*) AS INTEGER) AS dialog_all_event_count,
                CAST(LN(1.0 + COUNT(*)) AS FLOAT) AS dialog_all_log1p_event_count,
                CAST(COUNT(DISTINCT CAST(event_time_msk AS DATE)) AS SMALLINT) AS dialog_all_active_days,
                CAST(EPOCH({c_orig} - MAX(event_time_msk)) / 86400.0 AS FLOAT) AS dialog_all_days_since_last_event,
                CAST(EPOCH({c_orig} - MIN(event_time_msk)) / 86400.0 AS FLOAT) AS dialog_all_days_since_first_event
            FROM dlg_raw
            WHERE event_time_msk < {c_eff}
            GROUP BY client_id
        """).df()
        df_m["dialog_30d_has_event"] = (df_m["dialog_30d_event_count"] > 0).astype(np.int8)
        df_m = df_m.drop(columns=["dialog_30d_has_event_lag"])
        meta_frames.append(df_m)

        # Embeddings (all & 90d before c_eff)
        df_e = con.execute(f"""
            SELECT
                client_id,
                event_time_msk >= {c_eff} - INTERVAL 90 DAY AS is_90d,
                embedding
            FROM dlg_raw
            WHERE event_time_msk < {c_eff} AND embedding IS NOT NULL
        """).df()
        if len(df_e) > 0:
            mat = np.vstack(df_e["embedding"].to_numpy()).astype(np.float64)
            cids = df_e["client_id"].to_numpy()
            # All mean-pool
            uniq_a, inv_a = np.unique(cids, return_inverse=True)
            sums_a = np.zeros((len(uniq_a), mat.shape[1]), dtype=np.float64)
            cnts_a = np.bincount(inv_a).astype(np.float64)[:, None]
            np.add.at(sums_a, inv_a, mat)
            means_a = (sums_a / cnts_a).astype(np.float32)
            all_months_l.append(np.full(len(uniq_a), m))
            all_clients_l.append(uniq_a)
            all_emb_l.append(means_a)

            # 90d mean-pool
            m90 = df_e["is_90d"].to_numpy(dtype=bool)
            if np.any(m90):
                c90 = cids[m90]
                mat90 = mat[m90]
                uniq_9, inv_9 = np.unique(c90, return_inverse=True)
                sums_9 = np.zeros((len(uniq_9), mat90.shape[1]), dtype=np.float64)
                cnts_9 = np.bincount(inv_9).astype(np.float64)[:, None]
                np.add.at(sums_9, inv_9, mat90)
                means_9 = (sums_9 / cnts_9).astype(np.float32)
                w90_months_l.append(np.full(len(uniq_9), m))
                w90_clients_l.append(uniq_9)
                w90_emb_l.append(means_9)

    meta_df = pd.concat(meta_frames, ignore_index=True)
    all_emb = np.vstack(all_emb_l)
    w90_emb = np.vstack(w90_emb_l)
    all_months = np.concatenate(all_months_l)
    all_clients = np.concatenate(all_clients_l)
    w90_months = np.concatenate(w90_months_l)
    w90_clients = np.concatenate(w90_clients_l)

    emb_store = {
        "all_emb": all_emb,
        "w90_emb": w90_emb,
        "all_map": {(str(mo), str(ci)): i for i, (mo, ci) in enumerate(zip(all_months, all_clients))},
        "w90_map": {(str(mo), str(ci)): i for i, (mo, ci) in enumerate(zip(w90_months, w90_clients))},
    }
    return meta_df, emb_store


def load_emb_store_from_npz() -> dict[str, Any]:
    z = np.load(EMB_PATH, allow_pickle=False)
    return {
        "all_emb": z["all_emb"].astype(np.float32),
        "w90_emb": z["w90_emb"].astype(np.float32),
        "all_map": {(str(mo), str(ci)): i for i, (mo, ci) in enumerate(zip(z["all_months"], z["all_clients"]))},
        "w90_map": {(str(mo), str(ci)): i for i, (mo, ci) in enumerate(zip(z["w90_months"], z["w90_clients"]))},
    }


def fit_transform_emb_fold(
    emb_store: dict[str, Any],
    tr_months: np.ndarray,
    tr_cids: np.ndarray,
    ev_months: np.ndarray,
    ev_cids: np.ndarray,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    def _lookup(months, cids, which):
        mp = emb_store["all_map"] if which == "all" else emb_store["w90_map"]
        mat = emb_store["all_emb"] if which == "all" else emb_store["w90_emb"]
        idxs = np.array([mp.get((str(mo), str(ci)), -1) for mo, ci in zip(months, cids)], dtype=np.int32)
        valid = idxs >= 0
        return valid, mat[idxs[valid]] if np.any(valid) else np.empty((0, mat.shape[1]), dtype=np.float32)

    n_tr, n_ev = len(tr_cids), len(ev_cids)
    n_feat = PCA_ALL_DIM + PCA_90D_DIM + 3
    F_tr = np.full((n_tr, n_feat), np.nan, dtype=np.float32)
    F_ev = np.full((n_ev, n_feat), np.nan, dtype=np.float32)

    tr_va, tr_ea = _lookup(tr_months, tr_cids, "all")
    ev_va, ev_ea = _lookup(ev_months, ev_cids, "all")
    tr_v9, tr_e9 = _lookup(tr_months, tr_cids, "90d")
    ev_v9, ev_e9 = _lookup(ev_months, ev_cids, "90d")

    F_tr[tr_va, PCA_ALL_DIM + PCA_90D_DIM] = np.linalg.norm(tr_ea, axis=1).astype(np.float32)
    if len(ev_ea):
        F_ev[ev_va, PCA_ALL_DIM + PCA_90D_DIM] = np.linalg.norm(ev_ea, axis=1).astype(np.float32)
    F_tr[tr_v9, PCA_ALL_DIM + PCA_90D_DIM + 1] = np.linalg.norm(tr_e9, axis=1).astype(np.float32)
    if len(ev_e9):
        F_ev[ev_v9, PCA_ALL_DIM + PCA_90D_DIM + 1] = np.linalg.norm(ev_e9, axis=1).astype(np.float32)

    for v_all, e_all, v_90, e_90, F in (
        (tr_va, tr_ea, tr_v9, tr_e9, F_tr),
        (ev_va, ev_ea, ev_v9, ev_e9, F_ev),
    ):
        both = v_all & v_90
        if np.any(both):
            pos_all = np.cumsum(v_all)[both] - 1
            pos_90 = np.cumsum(v_90)[both] - 1
            va_b, v9_b = e_all[pos_all], e_90[pos_90]
            dot = np.sum(va_b * v9_b, axis=1)
            denom = np.maximum(np.linalg.norm(va_b, axis=1) * np.linalg.norm(v9_b, axis=1), 1e-8)
            F[both, PCA_ALL_DIM + PCA_90D_DIM + 2] = (dot / denom).astype(np.float32)

    sc_all = StandardScaler(copy=False)
    pca_all = PCA(n_components=PCA_ALL_DIM, svd_solver="randomized", random_state=seed)
    F_tr[tr_va, :PCA_ALL_DIM] = pca_all.fit_transform(sc_all.fit_transform(tr_ea)).astype(np.float32)
    if len(ev_ea):
        F_ev[ev_va, :PCA_ALL_DIM] = pca_all.transform(sc_all.transform(ev_ea)).astype(np.float32)

    sc_90 = StandardScaler(copy=False)
    pca_90 = PCA(n_components=PCA_90D_DIM, svd_solver="randomized", random_state=seed)
    F_tr[tr_v9, PCA_ALL_DIM:PCA_ALL_DIM + PCA_90D_DIM] = pca_90.fit_transform(sc_90.fit_transform(tr_e9)).astype(np.float32)
    if len(ev_e9):
        F_ev[ev_v9, PCA_ALL_DIM:PCA_ALL_DIM + PCA_90D_DIM] = pca_90.transform(sc_90.transform(ev_e9)).astype(np.float32)

    return F_tr, F_ev


def run_cb_5fold(
    exp_tag: str,
    stage: str,
    df_tr: pd.DataFrame,
    df_ev: pd.DataFrame,
    tab_cols: list[str],
    target_col: str = "y_any",
    emb_store: dict[str, Any] | None = None,
    emb_mode: str = "full",  # "none", "full" (27), "pca_only" (24), "placebo_shuffle" (27)
) -> pd.DataFrame:
    out_p = PRED_OUT / f"{stage}_{exp_tag}.parquet"
    if out_p.exists():
        return pd.read_parquet(out_p)

    t0 = time.perf_counter()
    n_emb = 0 if emb_mode == "none" else (PCA_ALL_DIM + PCA_90D_DIM if emb_mode == "pca_only" else PCA_ALL_DIM + PCA_90D_DIM + 3)
    fold_dfs = []
    for k in range(5):
        tr_k = df_tr["fold"].to_numpy() != k
        ev_k = df_ev["fold"].to_numpy() == k

        tr_slice = df_tr.loc[tr_k].reset_index(drop=True)
        ev_slice = df_ev.loc[ev_k].reset_index(drop=True)

        X_tr = np.empty((len(tr_slice), len(tab_cols) + n_emb), dtype=np.float32)
        X_ev = np.empty((len(ev_slice), len(tab_cols) + n_emb), dtype=np.float32)
        for j, c in enumerate(tab_cols):
            X_tr[:, j] = tr_slice[c].to_numpy(dtype=np.float32, copy=False)
            X_ev[:, j] = ev_slice[c].to_numpy(dtype=np.float32, copy=False)

        if emb_mode != "none":
            assert emb_store is not None
            F_tr, F_ev = fit_transform_emb_fold(
                emb_store,
                tr_slice["month"].to_numpy(), tr_slice["client_id"].to_numpy(),
                ev_slice["month"].to_numpy(), ev_slice["client_id"].to_numpy(),
                seed=SEED + k,
            )
            if emb_mode == "pca_only":
                F_tr = F_tr[:, :PCA_ALL_DIM + PCA_90D_DIM]
                F_ev = F_ev[:, :PCA_ALL_DIM + PCA_90D_DIM]
            elif emb_mode == "placebo_shuffle":
                # Shuffle embeddings strictly among rows WITH dialog_all_has_event==1 within each month!
                rng = np.random.default_rng(7000 + k)
                for df_s, F_s in ((tr_slice, F_tr), (ev_slice, F_ev)):
                    for mo in df_s["month"].unique():
                        idx_act = np.flatnonzero((df_s["month"].to_numpy() == mo) & (df_s["dialog_all_has_event"].to_numpy() == 1))
                        if len(idx_act) > 1:
                            perm = rng.permutation(idx_act)
                            F_s[idx_act, :] = F_s[perm, :]
            X_tr[:, len(tab_cols):] = F_tr
            X_ev[:, len(tab_cols):] = F_ev

        y_tr = tr_slice[target_col].to_numpy(dtype=np.int8)
        y_ev = ev_slice[target_col].to_numpy(dtype=np.int8)
        clf = CatBoostClassifier(**CB_PARAMS)
        clf.fit(X_tr, y_tr)
        sc = clf.predict_proba(X_ev)[:, 1].astype(np.float64)
        fold_dfs.append(pd.DataFrame({
            "client_id": ev_slice["client_id"].to_numpy(),
            "fold": ev_slice["fold"].to_numpy(),
            "month": ev_slice["month"].to_numpy(),
            "y_true": y_ev,
            "score": sc,
        }))
        del X_tr, X_ev, clf
        gc.collect()

    pdf = pd.concat(fold_dfs, ignore_index=True).sort_values(["month", "client_id"]).reset_index(drop=True)
    pdf.to_parquet(out_p, index=False, compression="zstd")
    mm, _ = compute_metrics(pdf["y_true"].to_numpy(), pdf["score"].to_numpy(), pdf["client_id"].to_numpy(),
                            budget_fractions=(0.01, 0.05), primary_budget=0.01, seed=SEED, is_probabilistic=True)
    print(f"  [{stage.upper():4s}] {exp_tag:28s} | PR-AUC={mm['average_precision']:.5f} | AUC={mm['roc_auc']:.4f} | Lift@1%={mm['lift_1pct']:.2f}x ({time.perf_counter()-t0:.1f}s)", flush=True)
    return pdf


def main():
    panel_df = pd.read_parquet(PANEL_PATH)
    panel_cols = list(panel_df.columns)
    trx_63 = [c for c in panel_cols if c.startswith("trx_")]
    geo_30 = [c for c in panel_cols if c.startswith("geo_")]
    dlg_18 = [c for c in panel_cols if c.startswith("dialog_")]
    cols_93 = trx_63 + geo_30
    cols_112 = trx_63 + geo_30 + dlg_18 + ["cold_start"]

    # Attach y_next (target of month M+1) for Test 1.3 (T+2 prediction)
    panel_df = panel_df.sort_values(["client_id", "month"]).reset_index(drop=True)
    panel_df["y_next"] = panel_df.groupby("client_id")["y_any"].shift(-1)

    results = []

    def _record(group: str, variant: str, stage: str, pdf: pd.DataFrame):
        mm, _ = compute_metrics(pdf["y_true"].to_numpy(), pdf["score"].to_numpy(), pdf["client_id"].to_numpy(),
                                budget_fractions=(0.01, 0.05), primary_budget=0.01, seed=SEED, is_probabilistic=True)
        results.append({
            "group": group, "variant": variant, "stage": stage,
            "n": mm["n"], "positives": mm["positives"], "prevalence": mm["prevalence"],
            "pr_auc": mm["average_precision"], "roc_auc": mm["roc_auc"],
            "lift_1pct": mm["lift_1pct"], "hits_1pct": mm["hits_1pct"],
            "lift_5pct": mm["lift_5pct"], "log_loss": mm["log_loss"],
        })

    # Reference existing seed=42 L5b (no dialog), L5c_cb, FE-v1 from baseline_moscow_v1/fe_v1_ablation
    for stage, ev_m in (("val", ["2022-09", "2022-10"]), ("test", ["2022-11", "2022-12", "2023-01"])):
        p_l5b = pd.read_parquet(MOSCOW_DIR / "predictions" / stage / "L5b_cb_geo_full.parquet")
        p_l5c = pd.read_parquet(REPO_ROOT / "outputs" / "fe_v1_ablation" / "predictions" / stage / "L5c_cb_seed42.parquet")
        p_fe1 = pd.read_parquet(REPO_ROOT / "outputs" / "fe_v1_ablation" / "predictions" / stage / "FE-v1_seed42.parquet")
        _record("ref_0d", "L5b_cb_no_dialog", stage, p_l5b)
        _record("ref_0d", "L5c_cb_lag0", stage, p_l5c)
        _record("ref_0d", "FE-v1_lag0", stage, p_fe1)

    # A. Activity vs Content Ablations (PCA-only & Placebo shuffle) on standard Val & Test
    print("=== A. Activity vs Content Ablations (PCA-only & Placebo) ===", flush=True)
    emb_0 = load_emb_store_from_npz()
    for stage, tr_m, ev_m in (
        ("val", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08"], ["2022-09", "2022-10"]),
        ("test", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10"], ["2022-11", "2022-12", "2023-01"]),
    ):
        df_tr = panel_df[panel_df["month"].isin(tr_m)].reset_index(drop=True)
        df_ev = panel_df[panel_df["month"].isin(ev_m)].reset_index(drop=True)
        p_pca = run_cb_5fold("FE-v1_pca_only", stage, df_tr, df_ev, cols_112, "y_any", emb_0, "pca_only")
        p_plb = run_cb_5fold("FE-v1_placebo_shuffle", stage, df_tr, df_ev, cols_112, "y_any", emb_0, "placebo_shuffle")
        _record("ablation_emb", "FE-v1_pca_only", stage, p_pca)
        _record("ablation_emb", "FE-v1_placebo_shuffle", stage, p_plb)
    del emb_0
    gc.collect()

    # B. Dialog Lag Buffers (7, 14, 30 days) on Val & Test
    for lag in (7, 14, 30):
        print(f"=== B. Dialog Lag Buffer = {lag} days ===", flush=True)
        dmeta, demb = build_lagged_dialog_meta_and_emb(lag)
        p_lag = panel_df.drop(columns=dlg_18 + ["cold_start"]).merge(dmeta, on=["month", "client_id"], how="left")
        for c in dlg_18:
            if c.endswith("_has_event") or c.endswith("_event_count") or c.endswith("_active_days"):
                p_lag[c] = p_lag[c].fillna(0)
            elif c.endswith("_log1p_event_count"):
                p_lag[c] = p_lag[c].fillna(0.0)
        p_lag["cold_start"] = (
            (p_lag["trx_all_has_event"] == 0) & (p_lag["geo_all_has_event"] == 0) & (p_lag["dialog_all_has_event"] == 0)
        ).astype(np.int8)

        for stage, tr_m, ev_m in (
            ("val", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08"], ["2022-09", "2022-10"]),
            ("test", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10"], ["2022-11", "2022-12", "2023-01"]),
        ):
            df_tr = p_lag[p_lag["month"].isin(tr_m)].reset_index(drop=True)
            df_ev = p_lag[p_lag["month"].isin(ev_m)].reset_index(drop=True)
            p_c = run_cb_5fold(f"L5c_cb_lag{lag}", stage, df_tr, df_ev, cols_112, "y_any", None, "none")
            p_d = run_cb_5fold(f"FE-v1_lag{lag}", stage, df_tr, df_ev, cols_112, "y_any", demb, "full")
            _record(f"lag_{lag}d", f"L5c_cb_lag{lag}", stage, p_c)
            _record(f"lag_{lag}d", f"FE-v1_lag{lag}", stage, p_d)
        del p_lag, dmeta, demb
        gc.collect()

    # C. Target Shift T+2 (features at cutoff M predict y_any(M+1))
    # Val: Train M in 04..07 (targets 05..08) -> Eval M in 09 (target 10) or 08..09 (targets 09..10)?
    # Better: Train M in 04..07 (targets 05..08) -> Val M in 08,09 (targets 09,10);
    #         Train M in 04..09 (targets 05..10) -> Test M in 10,11,12 (targets 11,12,01)!
    # Notice: target months match Val (09, 10) and Test (11, 12, 01) 1-to-1!
    print("=== C. Target Shift T+2 (Features at M predict target month M+1: Val=09..10, Test=11..01) ===", flush=True)
    emb_0 = load_emb_store_from_npz()
    panel_t2 = panel_df.dropna(subset=["y_next"]).copy()
    panel_t2["y_next"] = panel_t2["y_next"].astype(np.int8)
    for stage, tr_m, ev_m in (
        ("val", ["2022-04", "2022-05", "2022-06", "2022-07"], ["2022-08", "2022-09"]),
        ("test", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09"], ["2022-10", "2022-11", "2022-12"]),
    ):
        df_tr = panel_t2[panel_t2["month"].isin(tr_m)].reset_index(drop=True)
        df_ev = panel_t2[panel_t2["month"].isin(ev_m)].reset_index(drop=True)
        p_b = run_cb_5fold("L5b_cb_Tplus2", stage, df_tr, df_ev, cols_93, "y_next", None, "none")
        p_c = run_cb_5fold("L5c_cb_Tplus2", stage, df_tr, df_ev, cols_112, "y_next", None, "none")
        p_d = run_cb_5fold("FE-v1_Tplus2", stage, df_tr, df_ev, cols_112, "y_next", emb_0, "full")
        _record("shift_Tplus2", "L5b_cb_no_dialog_Tplus2", stage, p_b)
        _record("shift_Tplus2", "L5c_cb_Tplus2", stage, p_c)
        _record("shift_Tplus2", "FE-v1_Tplus2", stage, p_d)

    pd.DataFrame(results).to_csv(OUT_DIR / "group1_dialog_model_experiments.csv", index=False)

    # D. Per-cluster evaluation on Val & Test for L5b, L5c_lag0/7/14/30, FE-v1_lag0/7/14/30, and T+2
    print("=== D. Per-Cluster Evaluation of Dialog Models ===", flush=True)
    dlg_ev = pd.read_parquet(OUT_DIR / "dialog_events_with_cluster.parquet")
    # For each (month, client_id) in modeling months, get cluster of the last call strictly < YYYY-MM-01 MSK
    con = duckdb.connect()
    con.register("dlg_ev", dlg_ev)
    cl_rows = []
    for m in ("2022-08", "2022-09", "2022-10", "2022-11", "2022-12", "2023-01"):
        q = con.execute(f"""
            SELECT '{m}' AS month, client_id, cluster
            FROM (
                SELECT client_id, cluster,
                       ROW_NUMBER() OVER (PARTITION BY client_id ORDER BY event_time_msk DESC) AS rn
                FROM dlg_ev
                WHERE event_time_msk < TIMESTAMP '{m}-01 00:00:00'
            )
            WHERE rn = 1
        """).df()
        cl_rows.append(q)
    cl_map = pd.concat(cl_rows, ignore_index=True)

    cluster_perf = []
    eval_specs = [
        ("val", "L5b_cb_no_dialog", MOSCOW_DIR / "predictions" / "val" / "L5b_cb_geo_full.parquet"),
        ("val", "L5c_cb_lag0", REPO_ROOT / "outputs" / "fe_v1_ablation" / "predictions" / "val" / "L5c_cb_seed42.parquet"),
        ("val", "FE-v1_lag0", REPO_ROOT / "outputs" / "fe_v1_ablation" / "predictions" / "val" / "FE-v1_seed42.parquet"),
        ("val", "L5c_cb_lag7", PRED_OUT / "val_L5c_cb_lag7.parquet"),
        ("val", "FE-v1_lag7", PRED_OUT / "val_FE-v1_lag7.parquet"),
        ("val", "L5c_cb_lag30", PRED_OUT / "val_L5c_cb_lag30.parquet"),
        ("val", "FE-v1_lag30", PRED_OUT / "val_FE-v1_lag30.parquet"),
        ("val", "L5b_cb_Tplus2", PRED_OUT / "val_L5b_cb_Tplus2.parquet"),
        ("val", "L5c_cb_Tplus2", PRED_OUT / "val_L5c_cb_Tplus2.parquet"),
        ("val", "FE-v1_Tplus2", PRED_OUT / "val_FE-v1_Tplus2.parquet"),
        ("test", "L5b_cb_no_dialog", MOSCOW_DIR / "predictions" / "test" / "L5b_cb_geo_full.parquet"),
        ("test", "L5c_cb_lag0", REPO_ROOT / "outputs" / "fe_v1_ablation" / "predictions" / "test" / "L5c_cb_seed42.parquet"),
        ("test", "FE-v1_lag0", REPO_ROOT / "outputs" / "fe_v1_ablation" / "predictions" / "test" / "FE-v1_seed42.parquet"),
        ("test", "L5c_cb_lag7", PRED_OUT / "test_L5c_cb_lag7.parquet"),
        ("test", "FE-v1_lag7", PRED_OUT / "test_FE-v1_lag7.parquet"),
        ("test", "L5c_cb_lag30", PRED_OUT / "test_L5c_cb_lag30.parquet"),
        ("test", "FE-v1_lag30", PRED_OUT / "test_FE-v1_lag30.parquet"),
        ("test", "L5b_cb_Tplus2", PRED_OUT / "test_L5b_cb_Tplus2.parquet"),
        ("test", "L5c_cb_Tplus2", PRED_OUT / "test_L5c_cb_Tplus2.parquet"),
        ("test", "FE-v1_Tplus2", PRED_OUT / "test_FE-v1_Tplus2.parquet"),
    ]
    for st, vname, ppath in eval_specs:
        pdf = pd.read_parquet(ppath).merge(cl_map, on=["month", "client_id"], how="left")
        pdf["cluster_grp"] = pdf["cluster"].fillna(-1).astype(int)
        for cg in (-1, 0, 1, 2, 3, 4, 5):
            sub = pdf[pdf["cluster_grp"] == cg]
            if len(sub) == 0 or sub["y_true"].sum() == 0:
                continue
            mm, _ = compute_metrics(sub["y_true"].to_numpy(), sub["score"].to_numpy(), sub["client_id"].to_numpy(),
                                    budget_fractions=(0.01, 0.05), primary_budget=0.01, seed=SEED, is_probabilistic=True)
            cluster_perf.append({
                "stage": st, "variant": vname, "cluster": cg,
                "n": mm["n"], "positives": mm["positives"], "prevalence": mm["prevalence"],
                "pr_auc": mm["average_precision"], "roc_auc": mm["roc_auc"],
                "lift_1pct": mm["lift_1pct"], "lift_5pct": mm["lift_5pct"],
            })
    pd.DataFrame(cluster_perf).to_csv(OUT_DIR / "group1_dialog_models_by_cluster.csv", index=False)
    print("ALL GROUP 1 MODEL CHECKS DONE.", flush=True)


if __name__ == "__main__":
    main()
