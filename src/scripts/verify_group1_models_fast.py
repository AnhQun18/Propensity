"""Group 1 Dialog Verification (Memory-Lean & Dual-Family: LightGBM + CatBoost):
1. Pre-caches fold-only PCA slices & lagged dialog metadata to disk so RAM stays < 1.5 GB
   and CatBoost never hits ResourceQuota(0) CPU throttling.
2. Runs LightGBM across ALL variants first (~3-4 min total) -> group1_dialog_lgbm_experiments.csv.
3. Runs CatBoost (exact 500-iter baseline spec, free RAM) across:
   - FE-v1_pca_only (drop norms & cosine) & FE-v1_placebo_shuffle (Val already cached, run Test)
   - L5c_cb & FE-v1 at lag = 7, 14, 30 days (Val & Test)
   - Target Shift T+2 (features at M predict y_any(M+1)) for L5b_cb, L5c_cb, FE-v1 (Val & Test)
4. Per-embedding-cluster breakdown (Test 1.4, suggestive only) -> group1_dialog_models_by_cluster.csv.
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from propensity_model.metrics import compute_metrics

DATA_ROOT = REPO_ROOT / "data" / "MBD-mini"
MOSCOW_DIR = REPO_ROOT / "outputs" / "baseline_moscow_v1"
PANEL_PATH = MOSCOW_DIR / "panel_moscow.parquet"
EMB_PATH = MOSCOW_DIR / "dialog_emb_moscow.npz"
STAGE0_PCA = REPO_ROOT / "outputs" / "fe_v1_ablation" / "_pca_stage0_seed42"
OUT_DIR = REPO_ROOT / "outputs" / "verification_audit"
PRED_OUT = OUT_DIR / "group1_preds"
PCA_OUT = OUT_DIR / "group1_pca_cache"
PRED_OUT.mkdir(parents=True, exist_ok=True)
PCA_OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
PCA_ALL_DIM = 16
PCA_90D_DIM = 8
MODELING_MONTHS = (
    "2022-04", "2022-05", "2022-06", "2022-07", "2022-08",
    "2022-09", "2022-10", "2022-11", "2022-12", "2023-01",
)
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
LGB_PARAMS = dict(
    random_state=SEED,
    n_jobs=4,
    verbose=-1,
)


def prepare_lag_cache(lag_days: int, meta_keys: pd.DataFrame) -> Path:
    """Build lagged dialog metadata + PCA slices (.npy per fold/stage) for lag_days > 0."""
    lag_dir = PCA_OUT / f"lag{lag_days}"
    meta_p = lag_dir / "dialog_meta.parquet"
    done_flag = lag_dir / "_READY.flag"
    if done_flag.exists() and meta_p.exists():
        return lag_dir

    lag_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"[PREP lag={lag_days}d] Building metadata + fold PCA slices ...", flush=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='2.5GB'")
    con.execute("SET threads=4")
    dialog_glob = (DATA_ROOT / "detail" / "dialog" / "*" / "*.parquet").as_posix()

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE dlg_raw AS
        SELECT client_id, event_time + INTERVAL 3 HOUR AS event_time_msk, embedding
        FROM read_parquet('{dialog_glob}')
    """)

    meta_frames = []
    all_m_l, all_c_l, all_e_l = [], [], []
    w90_m_l, w90_c_l, w90_e_l = [], [], []

    for m in MODELING_MONTHS:
        cutoff_str = f"{m}-01 00:00:00"
        c_eff = f"(TIMESTAMP '{cutoff_str}' - INTERVAL {lag_days} DAY)"
        c_orig = f"TIMESTAMP '{cutoff_str}'"

        df_m = con.execute(f"""
            SELECT
                client_id, '{m}' AS month,
                CAST(COUNT(*) FILTER (WHERE event_time_msk >= {c_eff} - INTERVAL 30 DAY) > 0 AS TINYINT) AS dialog_30d_has_event,
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
        meta_frames.append(df_m)

        df_e = con.execute(f"""
            SELECT client_id, event_time_msk >= {c_eff} - INTERVAL 90 DAY AS is_90d, embedding
            FROM dlg_raw
            WHERE event_time_msk < {c_eff} AND embedding IS NOT NULL
        """).df()
        if len(df_e) > 0:
            mat = np.vstack(df_e["embedding"].to_numpy()).astype(np.float64)
            cids = df_e["client_id"].to_numpy()
            uniq_a, inv_a = np.unique(cids, return_inverse=True)
            sums_a = np.zeros((len(uniq_a), mat.shape[1]), dtype=np.float64)
            np.add.at(sums_a, inv_a, mat)
            means_a = (sums_a / np.bincount(inv_a)[:, None]).astype(np.float32)
            all_m_l.append(np.full(len(uniq_a), m))
            all_c_l.append(uniq_a)
            all_e_l.append(means_a)

            m90 = df_e["is_90d"].to_numpy(dtype=bool)
            if np.any(m90):
                c90, mat90 = cids[m90], mat[m90]
                uniq_9, inv_9 = np.unique(c90, return_inverse=True)
                sums_9 = np.zeros((len(uniq_9), mat90.shape[1]), dtype=np.float64)
                np.add.at(sums_9, inv_9, mat90)
                means_9 = (sums_9 / np.bincount(inv_9)[:, None]).astype(np.float32)
                w90_m_l.append(np.full(len(uniq_9), m))
                w90_c_l.append(uniq_9)
                w90_e_l.append(means_9)

    con.close()
    meta_df = pd.concat(meta_frames, ignore_index=True)
    meta_df.to_parquet(meta_p, index=False, compression="zstd")

    all_emb = np.vstack(all_e_l)
    w90_emb = np.vstack(w90_e_l)
    all_months, all_clients = np.concatenate(all_m_l), np.concatenate(all_c_l)
    w90_months, w90_clients = np.concatenate(w90_m_l), np.concatenate(w90_c_l)

    idx_all = pd.MultiIndex.from_arrays([all_months.astype(str), all_clients.astype(str)])
    idx_90 = pd.MultiIndex.from_arrays([w90_months.astype(str), w90_clients.astype(str)])
    all_norms = np.linalg.norm(all_emb, axis=1).astype(np.float32)
    w90_norms = np.linalg.norm(w90_emb, axis=1).astype(np.float32)

    w90_to_all = idx_all.get_indexer(idx_90)
    cos_w90 = np.full(len(w90_emb), np.nan, dtype=np.float32)
    val_m = w90_to_all >= 0
    if np.any(val_m):
        a_rows = all_emb[w90_to_all[val_m]]
        w_rows = w90_emb[val_m]
        dot = np.sum(a_rows * w_rows, axis=1)
        den = np.maximum(all_norms[w90_to_all[val_m]] * w90_norms[val_m], 1e-8)
        cos_w90[val_m] = (dot / den).astype(np.float32)

    n_feat = PCA_ALL_DIM + PCA_90D_DIM + 3
    for stage, tr_m, ev_m in (
        ("val", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08"], ["2022-09", "2022-10"]),
        ("test", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10"], ["2022-11", "2022-12", "2023-01"]),
    ):
        sdir = lag_dir / stage
        sdir.mkdir(exist_ok=True)
        df_tr = meta_keys[meta_keys["month"].isin(tr_m)].reset_index(drop=True)
        df_ev = meta_keys[meta_keys["month"].isin(ev_m)].reset_index(drop=True)
        for k in range(5):
            tr_k = df_tr[df_tr["fold"] != k]
            ev_k = df_ev[df_ev["fold"] == k]
            tr_ia = idx_all.get_indexer(pd.MultiIndex.from_arrays([tr_k["month"].astype(str), tr_k["client_id"].astype(str)]))
            ev_ia = idx_all.get_indexer(pd.MultiIndex.from_arrays([ev_k["month"].astype(str), ev_k["client_id"].astype(str)]))
            tr_i9 = idx_90.get_indexer(pd.MultiIndex.from_arrays([tr_k["month"].astype(str), tr_k["client_id"].astype(str)]))
            ev_i9 = idx_90.get_indexer(pd.MultiIndex.from_arrays([ev_k["month"].astype(str), ev_k["client_id"].astype(str)]))
            tr_va, ev_va = tr_ia >= 0, ev_ia >= 0
            tr_v9, ev_v9 = tr_i9 >= 0, ev_i9 >= 0

            F_tr = np.full((len(tr_k), n_feat), np.nan, dtype=np.float32)
            F_ev = np.full((len(ev_k), n_feat), np.nan, dtype=np.float32)
            F_tr[tr_va, 24] = all_norms[tr_ia[tr_va]]
            F_ev[ev_va, 24] = all_norms[ev_ia[ev_va]]
            F_tr[tr_v9, 25] = w90_norms[tr_i9[tr_v9]]
            F_ev[ev_v9, 25] = w90_norms[ev_i9[ev_v9]]
            F_tr[tr_v9, 26] = cos_w90[tr_i9[tr_v9]]
            F_ev[ev_v9, 26] = cos_w90[ev_i9[ev_v9]]

            sc_a = StandardScaler(copy=False)
            p_a = PCA(n_components=PCA_ALL_DIM, svd_solver="randomized", random_state=SEED + k)
            F_tr[tr_va, :16] = p_a.fit_transform(sc_a.fit_transform(all_emb[tr_ia[tr_va]])).astype(np.float32)
            if np.any(ev_va):
                F_ev[ev_va, :16] = p_a.transform(sc_a.transform(all_emb[ev_ia[ev_va]])).astype(np.float32)

            sc_9 = StandardScaler(copy=False)
            p_9 = PCA(n_components=PCA_90D_DIM, svd_solver="randomized", random_state=SEED + k)
            F_tr[tr_v9, 16:24] = p_9.fit_transform(sc_9.fit_transform(w90_emb[tr_i9[tr_v9]])).astype(np.float32)
            if np.any(ev_v9):
                F_ev[ev_v9, 16:24] = p_9.transform(sc_9.transform(w90_emb[ev_i9[ev_v9]])).astype(np.float32)

            np.save(sdir / f"f{k}_tr.npy", F_tr)
            np.save(sdir / f"f{k}_ev.npy", F_ev)
    done_flag.write_text("ok", encoding="utf-8")
    del all_emb, w90_emb, all_norms, w90_norms, cos_w90
    gc.collect()
    print(f"[PREP lag={lag_days}d] Ready in {time.perf_counter()-t0:.1f}s", flush=True)
    return lag_dir


def prepare_tplus2_pca_cache(meta_keys: pd.DataFrame) -> Path:
    """Build train-only PCA slices matching exact T+2 split schedules (no Val/Test leak in PCA)."""
    t2_dir = PCA_OUT / "tplus2"
    done_flag = t2_dir / "_READY.flag"
    if done_flag.exists():
        return t2_dir

    t2_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print("[PREP T+2] Caching T+2 fold PCA slices ...", flush=True)
    with np.load(EMB_PATH, allow_pickle=False) as z:
        all_emb = z["all_emb"].astype(np.float32)
        w90_emb = z["w90_emb"].astype(np.float32)
        idx_all = pd.MultiIndex.from_arrays([z["all_months"].astype(str), z["all_clients"].astype(str)])
        idx_90 = pd.MultiIndex.from_arrays([z["w90_months"].astype(str), z["w90_clients"].astype(str)])
    all_norms = np.linalg.norm(all_emb, axis=1).astype(np.float32)
    w90_norms = np.linalg.norm(w90_emb, axis=1).astype(np.float32)
    w90_to_all = idx_all.get_indexer(idx_90)
    cos_w90 = np.full(len(w90_emb), np.nan, dtype=np.float32)
    vm = w90_to_all >= 0
    if np.any(vm):
        dot = np.sum(all_emb[w90_to_all[vm]] * w90_emb[vm], axis=1)
        den = np.maximum(all_norms[w90_to_all[vm]] * w90_norms[vm], 1e-8)
        cos_w90[vm] = (dot / den).astype(np.float32)

    n_feat = PCA_ALL_DIM + PCA_90D_DIM + 3
    for stage, tr_m, ev_m in (
        ("val", ["2022-04", "2022-05", "2022-06", "2022-07"], ["2022-08", "2022-09"]),
        ("test", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09"], ["2022-10", "2022-11", "2022-12"]),
    ):
        sdir = t2_dir / stage
        sdir.mkdir(exist_ok=True)
        df_tr = meta_keys[meta_keys["month"].isin(tr_m)].reset_index(drop=True)
        df_ev = meta_keys[meta_keys["month"].isin(ev_m)].reset_index(drop=True)
        for k in range(5):
            tr_k = df_tr[df_tr["fold"] != k]
            ev_k = df_ev[df_ev["fold"] == k]
            tr_ia = idx_all.get_indexer(pd.MultiIndex.from_arrays([tr_k["month"].astype(str), tr_k["client_id"].astype(str)]))
            ev_ia = idx_all.get_indexer(pd.MultiIndex.from_arrays([ev_k["month"].astype(str), ev_k["client_id"].astype(str)]))
            tr_i9 = idx_90.get_indexer(pd.MultiIndex.from_arrays([tr_k["month"].astype(str), tr_k["client_id"].astype(str)]))
            ev_i9 = idx_90.get_indexer(pd.MultiIndex.from_arrays([ev_k["month"].astype(str), ev_k["client_id"].astype(str)]))
            tr_va, ev_va = tr_ia >= 0, ev_ia >= 0
            tr_v9, ev_v9 = tr_i9 >= 0, ev_i9 >= 0

            F_tr = np.full((len(tr_k), n_feat), np.nan, dtype=np.float32)
            F_ev = np.full((len(ev_k), n_feat), np.nan, dtype=np.float32)
            F_tr[tr_va, 24] = all_norms[tr_ia[tr_va]]
            F_ev[ev_va, 24] = all_norms[ev_ia[ev_va]]
            F_tr[tr_v9, 25] = w90_norms[tr_i9[tr_v9]]
            F_ev[ev_v9, 25] = w90_norms[ev_i9[ev_v9]]
            F_tr[tr_v9, 26] = cos_w90[tr_i9[tr_v9]]
            F_ev[ev_v9, 26] = cos_w90[ev_i9[ev_v9]]

            sc_a = StandardScaler(copy=False)
            p_a = PCA(n_components=PCA_ALL_DIM, svd_solver="randomized", random_state=SEED + k)
            F_tr[tr_va, :16] = p_a.fit_transform(sc_a.fit_transform(all_emb[tr_ia[tr_va]])).astype(np.float32)
            if np.any(ev_va):
                F_ev[ev_va, :16] = p_a.transform(sc_a.transform(all_emb[ev_ia[ev_va]])).astype(np.float32)

            sc_9 = StandardScaler(copy=False)
            p_9 = PCA(n_components=PCA_90D_DIM, svd_solver="randomized", random_state=SEED + k)
            F_tr[tr_v9, 16:24] = p_9.fit_transform(sc_9.fit_transform(w90_emb[tr_i9[tr_v9]])).astype(np.float32)
            if np.any(ev_v9):
                F_ev[ev_v9, 16:24] = p_9.transform(sc_9.transform(w90_emb[ev_i9[ev_v9]])).astype(np.float32)

            np.save(sdir / f"f{k}_tr.npy", F_tr)
            np.save(sdir / f"f{k}_ev.npy", F_ev)
    done_flag.write_text("ok", encoding="utf-8")
    del all_emb, w90_emb, all_norms, w90_norms, cos_w90
    gc.collect()
    print(f"[PREP T+2] Ready in {time.perf_counter()-t0:.1f}s", flush=True)
    return t2_dir


def run_5fold_from_cache(
    exp_tag: str,
    family: str,
    stage: str,
    df_tr: pd.DataFrame,
    df_ev: pd.DataFrame,
    tab_cols: list[str],
    target_col: str,
    pca_dir: Path | None,
    emb_mode: str,  # "none", "full", "pca_only", "placebo_shuffle"
) -> pd.DataFrame:
    out_p = PRED_OUT / f"{stage}_{exp_tag}.parquet"
    if out_p.exists():
        pdf = pd.read_parquet(out_p)
        mm, _ = compute_metrics(pdf["y_true"].to_numpy(), pdf["score"].to_numpy(), pdf["client_id"].to_numpy(),
                                budget_fractions=(0.01, 0.05), primary_budget=0.01, seed=SEED, is_probabilistic=True)
        print(f"  [CACHED {stage.upper():4s}] {exp_tag:30s} | PR-AUC={mm['average_precision']:.5f} | AUC={mm['roc_auc']:.4f} | Lift@1%={mm['lift_1pct']:.2f}x", flush=True)
        return pdf

    t0 = time.perf_counter()
    n_emb = 0 if emb_mode == "none" else (24 if emb_mode == "pca_only" else 27)
    fold_dfs = []
    for k in range(5):
        tr_k = df_tr["fold"].to_numpy() != k
        ev_k = df_ev["fold"].to_numpy() == k
        tr_sl = df_tr.loc[tr_k].reset_index(drop=True)
        ev_sl = df_ev.loc[ev_k].reset_index(drop=True)

        X_tr = np.empty((len(tr_sl), len(tab_cols) + n_emb), dtype=np.float32)
        X_ev = np.empty((len(ev_sl), len(tab_cols) + n_emb), dtype=np.float32)
        for j, c in enumerate(tab_cols):
            X_tr[:, j] = tr_sl[c].to_numpy(dtype=np.float32, copy=False)
            X_ev[:, j] = ev_sl[c].to_numpy(dtype=np.float32, copy=False)

        if emb_mode != "none":
            assert pca_dir is not None
            F_tr = np.load(pca_dir / stage / f"f{k}_tr.npy")
            F_ev = np.load(pca_dir / stage / f"f{k}_ev.npy")
            assert len(F_tr) == len(tr_sl) and len(F_ev) == len(ev_sl), f"Shape mismatch f{k}: {F_tr.shape} vs {len(tr_sl)}"
            if emb_mode == "pca_only":
                F_tr, F_ev = F_tr[:, :24], F_ev[:, :24]
            elif emb_mode == "placebo_shuffle":
                rng = np.random.default_rng(7000 + k)
                for df_s, F_s in ((tr_sl, F_tr), (ev_sl, F_ev)):
                    for mo in df_s["month"].unique():
                        idx_act = np.flatnonzero((df_s["month"].to_numpy() == mo) & (df_s["dialog_all_has_event"].to_numpy() == 1))
                        if len(idx_act) > 1:
                            F_s[idx_act, :] = F_s[rng.permutation(idx_act), :]
            X_tr[:, len(tab_cols):] = F_tr
            X_ev[:, len(tab_cols):] = F_ev
            del F_tr, F_ev

        y_tr = tr_sl[target_col].to_numpy(dtype=np.int8)
        y_ev = ev_sl[target_col].to_numpy(dtype=np.int8)
        clf = lgb.LGBMClassifier(**LGB_PARAMS) if family == "lgbm" else CatBoostClassifier(**CB_PARAMS)
        clf.fit(X_tr, y_tr)
        sc = clf.predict_proba(X_ev)[:, 1].astype(np.float64)
        fold_dfs.append(pd.DataFrame({
            "client_id": ev_sl["client_id"].to_numpy(),
            "fold": ev_sl["fold"].to_numpy(),
            "month": ev_sl["month"].to_numpy(),
            "y_true": y_ev,
            "score": sc,
        }))
        del X_tr, X_ev, clf
        gc.collect()

    pdf = pd.concat(fold_dfs, ignore_index=True).sort_values(["month", "client_id"]).reset_index(drop=True)
    tmp = out_p.with_suffix(".tmp.parquet")
    pdf.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(out_p)
    mm, _ = compute_metrics(pdf["y_true"].to_numpy(), pdf["score"].to_numpy(), pdf["client_id"].to_numpy(),
                            budget_fractions=(0.01, 0.05), primary_budget=0.01, seed=SEED, is_probabilistic=True)
    print(f"  [{stage.upper():4s}] {exp_tag:30s} | PR-AUC={mm['average_precision']:.5f} | AUC={mm['roc_auc']:.4f} | Lift@1%={mm['lift_1pct']:.2f}x ({time.perf_counter()-t0:.1f}s)", flush=True)
    return pdf


def swap_lag_meta(panel_df: pd.DataFrame, meta_p: Path, dlg_18: list[str]) -> pd.DataFrame:
    dmeta = pd.read_parquet(meta_p)
    p = panel_df.drop(columns=dlg_18 + ["cold_start"]).merge(dmeta, on=["month", "client_id"], how="left")
    for c in dlg_18:
        if c.endswith("_has_event") or c.endswith("_event_count") or c.endswith("_active_days"):
            p[c] = p[c].fillna(0)
        elif c.endswith("_log1p_event_count"):
            p[c] = p[c].fillna(0.0)
    p["cold_start"] = (
        (p["trx_all_has_event"] == 0) & (p["geo_all_has_event"] == 0) & (p["dialog_all_has_event"] == 0)
    ).astype(np.int8)
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default="lgbm,cb", help="Comma-separated: lgbm,cb")
    args = parser.parse_args()
    families = [x.strip() for x in args.families.split(",") if x.strip()]

    from scripts.run_fe_v1_ablation import ensure_stage0_pca_cached, load_spec as load_fe1_spec
    ensure_stage0_pca_cached(load_fe1_spec())

    keys_only = pd.read_parquet(PANEL_PATH, columns=["client_id", "fold", "month", "y_any"])
    for lag in (7, 14, 30):
        prepare_lag_cache(lag, keys_only)
    keys_t2 = keys_only.sort_values(["client_id", "month"]).reset_index(drop=True)
    keys_t2["y_next"] = keys_t2.groupby("client_id")["y_any"].shift(-1)
    keys_t2 = keys_t2.dropna(subset=["y_next"]).sort_values(["month", "client_id"]).reset_index(drop=True)
    prepare_tplus2_pca_cache(keys_t2)
    del keys_only, keys_t2
    gc.collect()

    panel_df = pd.read_parquet(PANEL_PATH)
    pcols = list(panel_df.columns)
    trx_63 = [c for c in pcols if c.startswith("trx_")]
    geo_30 = [c for c in pcols if c.startswith("geo_")]
    dlg_18 = [c for c in pcols if c.startswith("dialog_")]
    cols_93 = trx_63 + geo_30
    cols_112 = trx_63 + geo_30 + dlg_18 + ["cold_start"]

    # Keep only needed cols to minimize RAM
    keep_cols = ["client_id", "fold", "month", "y_any"] + cols_112
    panel_df = panel_df[keep_cols].copy()
    panel_df = panel_df.sort_values(["client_id", "month"]).reset_index(drop=True)
    panel_df["y_next"] = panel_df.groupby("client_id")["y_any"].shift(-1)
    panel_df = panel_df.sort_values(["month", "client_id"]).reset_index(drop=True)

    results = []

    def _rec(grp: str, fam: str, var: str, st: str, pdf: pd.DataFrame):
        mm, _ = compute_metrics(pdf["y_true"].to_numpy(), pdf["score"].to_numpy(), pdf["client_id"].to_numpy(),
                                budget_fractions=(0.01, 0.05), primary_budget=0.01, seed=SEED, is_probabilistic=True)
        results.append({
            "group": grp, "family": fam, "variant": var, "stage": st,
            "n": mm["n"], "positives": mm["positives"], "prevalence": mm["prevalence"],
            "pr_auc": mm["average_precision"], "roc_auc": mm["roc_auc"],
            "lift_1pct": mm["lift_1pct"], "hits_1pct": mm["hits_1pct"],
            "lift_5pct": mm["lift_5pct"], "log_loss": mm["log_loss"],
        })

    # Reference CB baselines
    for st in ("val", "test"):
        _rec("ref_0d", "cb", "L5b_cb_no_dialog", st, pd.read_parquet(MOSCOW_DIR / "predictions" / st / "L5b_cb_geo_full.parquet"))
        _rec("ref_0d", "cb", "L5c_cb_lag0", st, pd.read_parquet(REPO_ROOT / "outputs" / "fe_v1_ablation" / "predictions" / st / "L5c_cb_seed42.parquet"))
        _rec("ref_0d", "cb", "FE-v1_lag0", st, pd.read_parquet(REPO_ROOT / "outputs" / "fe_v1_ablation" / "predictions" / st / "FE-v1_seed42.parquet"))
        _rec("ref_0d", "lgbm", "L5c_lgbm_lag0", st, pd.read_parquet(MOSCOW_DIR / "predictions" / st / "L5c_lgbm_dialog_meta.parquet"))

    for fam in families:
        print(f"\n================ FAMILY: {fam.upper()} ================", flush=True)
        # 1. Ablations + baseline at lag=0
        for st, tr_m, ev_m in (
            ("val", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08"], ["2022-09", "2022-10"]),
            ("test", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10"], ["2022-11", "2022-12", "2023-01"]),
        ):
            df_tr = panel_df[panel_df["month"].isin(tr_m)].reset_index(drop=True)
            df_ev = panel_df[panel_df["month"].isin(ev_m)].reset_index(drop=True)
            if fam == "lgbm":
                _rec("ref_0d", "lgbm", "L5b_lgbm_no_dialog", st, run_5fold_from_cache("L5b_lgbm_no_dialog", "lgbm", st, df_tr, df_ev, cols_93, "y_any", None, "none"))
                _rec("ref_0d", "lgbm", "FE-v1_lgbm_lag0", st, run_5fold_from_cache("FE-v1_lgbm_lag0", "lgbm", st, df_tr, df_ev, cols_112, "y_any", STAGE0_PCA, "full"))
            _rec("ablation_emb", fam, f"FE-v1_pca_only_{fam}", st, run_5fold_from_cache("FE-v1_pca_only" if fam == "cb" else "FE-v1_pca_only_lgbm", fam, st, df_tr, df_ev, cols_112, "y_any", STAGE0_PCA, "pca_only"))
            _rec("ablation_emb", fam, f"FE-v1_placebo_shuffle_{fam}", st, run_5fold_from_cache("FE-v1_placebo_shuffle" if fam == "cb" else "FE-v1_placebo_shuffle_lgbm", fam, st, df_tr, df_ev, cols_112, "y_any", STAGE0_PCA, "placebo_shuffle"))

        # 2. Lag buffers 7, 14, 30
        for lag in (7, 14, 30):
            lag_dir = PCA_OUT / f"lag{lag}"
            p_lag = swap_lag_meta(panel_df, lag_dir / "dialog_meta.parquet", dlg_18)
            for st, tr_m, ev_m in (
                ("val", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08"], ["2022-09", "2022-10"]),
                ("test", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10"], ["2022-11", "2022-12", "2023-01"]),
            ):
                df_tr = p_lag[p_lag["month"].isin(tr_m)].reset_index(drop=True)
                df_ev = p_lag[p_lag["month"].isin(ev_m)].reset_index(drop=True)
                _rec(f"lag_{lag}d", fam, f"L5c_{fam}_lag{lag}", st, run_5fold_from_cache(f"L5c_{fam}_lag{lag}", fam, st, df_tr, df_ev, cols_112, "y_any", None, "none"))
                _rec(f"lag_{lag}d", fam, f"FE-v1_{fam}_lag{lag}", st, run_5fold_from_cache(f"FE-v1_{fam}_lag{lag}" if fam == "lgbm" else f"FE-v1_lag{lag}", fam, st, df_tr, df_ev, cols_112, "y_any", lag_dir, "full"))
            del p_lag
            gc.collect()

        # 3. Target Shift T+2
        p_t2 = panel_df.dropna(subset=["y_next"]).copy()
        p_t2["y_next"] = p_t2["y_next"].astype(np.int8)
        t2_dir = PCA_OUT / "tplus2"
        for st, tr_m, ev_m in (
            ("val", ["2022-04", "2022-05", "2022-06", "2022-07"], ["2022-08", "2022-09"]),
            ("test", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09"], ["2022-10", "2022-11", "2022-12"]),
        ):
            df_tr = p_t2[p_t2["month"].isin(tr_m)].reset_index(drop=True)
            df_ev = p_t2[p_t2["month"].isin(ev_m)].reset_index(drop=True)
            _rec("shift_Tplus2", fam, f"L5b_{fam}_Tplus2", st, run_5fold_from_cache(f"L5b_{fam}_Tplus2", fam, st, df_tr, df_ev, cols_93, "y_next", None, "none"))
            _rec("shift_Tplus2", fam, f"L5c_{fam}_Tplus2", st, run_5fold_from_cache(f"L5c_{fam}_Tplus2", fam, st, df_tr, df_ev, cols_112, "y_next", None, "none"))
            _rec("shift_Tplus2", fam, f"FE-v1_{fam}_Tplus2", st, run_5fold_from_cache(f"FE-v1_{fam}_Tplus2" if fam == "lgbm" else "FE-v1_Tplus2", fam, st, df_tr, df_ev, cols_112, "y_next", t2_dir, "full"))
        del p_t2
        gc.collect()

        pd.DataFrame(results).to_csv(OUT_DIR / "group1_dialog_model_experiments_fast.csv", index=False)

    # 4. Per-cluster breakdown across available models
    print("=== Per-Cluster Evaluation ===", flush=True)
    dlg_ev = pd.read_parquet(OUT_DIR / "dialog_events_with_cluster.parquet")
    con = duckdb.connect()
    con.register("dlg_ev", dlg_ev)
    cl_rows = []
    for m in ("2022-08", "2022-09", "2022-10", "2022-11", "2022-12", "2023-01"):
        cl_rows.append(con.execute(f"""
            SELECT '{m}' AS month, client_id, cluster
            FROM (
                SELECT client_id, cluster, ROW_NUMBER() OVER (PARTITION BY client_id ORDER BY event_time_msk DESC) AS rn
                FROM dlg_ev WHERE event_time_msk < TIMESTAMP '{m}-01 00:00:00'
            ) WHERE rn = 1
        """).df())
    cl_map = pd.concat(cl_rows, ignore_index=True)

    cluster_perf = []
    eval_specs = []
    for st in ("val", "test"):
        eval_specs.extend([
            (st, "L5b_cb_no_dialog", MOSCOW_DIR / "predictions" / st / "L5b_cb_geo_full.parquet"),
            (st, "L5c_cb_lag0", REPO_ROOT / "outputs" / "fe_v1_ablation" / "predictions" / st / "L5c_cb_seed42.parquet"),
            (st, "FE-v1_lag0", REPO_ROOT / "outputs" / "fe_v1_ablation" / "predictions" / st / "FE-v1_seed42.parquet"),
            (st, "L5c_lgbm_lag0", MOSCOW_DIR / "predictions" / st / "L5c_lgbm_dialog_meta.parquet"),
        ])
        for p in sorted(PRED_OUT.glob(f"{st}_*.parquet")):
            vname = p.stem[len(st) + 1:]
            eval_specs.append((st, vname, p))

    for st, vname, ppath in eval_specs:
        if not ppath.exists():
            continue
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
    print("ALL FAST GROUP 1 CHECKS DONE.", flush=True)


if __name__ == "__main__":
    main()
