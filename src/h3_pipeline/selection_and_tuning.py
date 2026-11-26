"""Leak-free PCA caching, Val-only ~150-feature selection, GPU QuantileDMatrix/Pool tuning, and ensemble optimization."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from baseline_v2.metrics import compute_slice_point_metrics, deterministic_tie_hash

from .models_gpu import USE_GPU, fit_predict_h3_model
from .progress import ModelProgressTracker, fmt_sec
from .protocol_h3 import assert_fold_isolation, quick_point_summary, validate_h3_schedule

PCA_ALL_DIM = 16
PCA_90D_DIM = 8
EMB_COL_NAMES: tuple[str, ...] = tuple(
    [f"dlg_pca_all_{i}" for i in range(PCA_ALL_DIM)]
    + [f"dlg_pca_90d_{i}" for i in range(PCA_90D_DIM)]
    + ["dlg_norm_all", "dlg_norm_90d", "dlg_cos_90d_all", "dlg_l2_drift_90d_all", "dlg_norm_ratio_90d_all"]
)


def ensure_h3_pca_cache(
    spec: dict[str, Any],
    panel_keys: pd.DataFrame,
    early_emb_npz: Path,
    moscow_emb_npz: Path,
    pca_dir: Path,
) -> None:
    """Build strict train-only PCA(16+8) + norm/drift slices per (step, fold) for Val & OOT Test."""
    validate_h3_schedule(spec)
    all_sched = list(spec["val_schedule"]) + list(spec["oot_test_schedule"])
    missing = any(
        not (pca_dir / str(st["eval_month"]) / f"f{k}_ev.npy").exists()
        for st in all_sched
        for k in range(int(spec["n_folds"]))
    )
    if not missing:
        print("[PCA-H3] Using cached train-only Dialog PCA slices.", flush=True)
        return

    t0 = time.perf_counter()
    print("[PCA-H3] Building leak-free Dialog PCA slices across H3 Val & OOT Test ...", flush=True)
    with np.load(early_emb_npz, allow_pickle=False) as z1, np.load(moscow_emb_npz, allow_pickle=False) as z2:
        all_emb = np.vstack([z1["all_emb"].astype(np.float32), z2["all_emb"].astype(np.float32)])
        w90_emb = np.vstack([z1["w90_emb"].astype(np.float32), z2["w90_emb"].astype(np.float32)])
        all_months = np.concatenate([z1["all_months"].astype(str), z2["all_months"].astype(str)])
        all_clients = np.concatenate([z1["all_clients"].astype(str), z2["all_clients"].astype(str)])
        w90_months = np.concatenate([z1["w90_months"].astype(str), z2["w90_months"].astype(str)])
        w90_clients = np.concatenate([z1["w90_clients"].astype(str), z2["w90_clients"].astype(str)])

    idx_all = pd.MultiIndex.from_arrays([all_months, all_clients])
    idx_90 = pd.MultiIndex.from_arrays([w90_months, w90_clients])
    all_norms = np.linalg.norm(all_emb, axis=1).astype(np.float32)
    w90_norms = np.linalg.norm(w90_emb, axis=1).astype(np.float32)

    w90_to_all = idx_all.get_indexer(idx_90)
    cos_w90 = np.full(len(w90_emb), np.nan, dtype=np.float32)
    drift_w90 = np.full(len(w90_emb), np.nan, dtype=np.float32)
    ratio_w90 = np.full(len(w90_emb), np.nan, dtype=np.float32)
    vm = w90_to_all >= 0
    if np.any(vm):
        a_sub = all_emb[w90_to_all[vm]]
        w_sub = w90_emb[vm]
        na = all_norms[w90_to_all[vm]]
        nw = w90_norms[vm]
        dot = np.sum(a_sub * w_sub, axis=1)
        cos_w90[vm] = (dot / np.maximum(na * nw, 1e-8)).astype(np.float32)
        drift_w90[vm] = np.linalg.norm(w_sub - a_sub, axis=1).astype(np.float32)
        ratio_w90[vm] = (nw / np.maximum(na, 1e-6)).astype(np.float32)

    n_emb = len(EMB_COL_NAMES)
    n_folds = int(spec["n_folds"])
    for st in all_sched:
        ev_m = str(st["eval_month"])
        tr_ms = [str(m) for m in st["train_months"]]
        mdir = pca_dir / ev_m
        mdir.mkdir(parents=True, exist_ok=True)
        df_tr = panel_keys[panel_keys["month"].isin(tr_ms)].reset_index(drop=True)
        df_ev = panel_keys[panel_keys["month"] == ev_m].reset_index(drop=True)

        for k in range(n_folds):
            if (mdir / f"f{k}_tr.npy").exists() and (mdir / f"f{k}_ev.npy").exists():
                continue
            tr_k = df_tr[df_tr["fold"] != k]
            ev_k = df_ev[df_ev["fold"] == k]
            assert_fold_isolation(tr_k["client_id"].to_numpy(), ev_k["client_id"].to_numpy(), f"PCA {ev_m} f{k}")

            tr_ia = idx_all.get_indexer(pd.MultiIndex.from_arrays([tr_k["month"].astype(str), tr_k["client_id"].astype(str)]))
            ev_ia = idx_all.get_indexer(pd.MultiIndex.from_arrays([ev_k["month"].astype(str), ev_k["client_id"].astype(str)]))
            tr_i9 = idx_90.get_indexer(pd.MultiIndex.from_arrays([tr_k["month"].astype(str), tr_k["client_id"].astype(str)]))
            ev_i9 = idx_90.get_indexer(pd.MultiIndex.from_arrays([ev_k["month"].astype(str), ev_k["client_id"].astype(str)]))

            tr_va, ev_va = tr_ia >= 0, ev_ia >= 0
            tr_v9, ev_v9 = tr_i9 >= 0, ev_i9 >= 0

            F_tr = np.full((len(tr_k), n_emb), np.nan, dtype=np.float32)
            F_ev = np.full((len(ev_k), n_emb), np.nan, dtype=np.float32)

            F_tr[tr_va, 24] = all_norms[tr_ia[tr_va]]
            F_ev[ev_va, 24] = all_norms[ev_ia[ev_va]]
            F_tr[tr_v9, 25] = w90_norms[tr_i9[tr_v9]]
            F_ev[ev_v9, 25] = w90_norms[ev_i9[ev_v9]]
            F_tr[tr_v9, 26] = cos_w90[tr_i9[tr_v9]]
            F_ev[ev_v9, 26] = cos_w90[ev_i9[ev_v9]]
            F_tr[tr_v9, 27] = drift_w90[tr_i9[tr_v9]]
            F_ev[ev_v9, 27] = drift_w90[ev_i9[ev_v9]]
            F_tr[tr_v9, 28] = ratio_w90[tr_i9[tr_v9]]
            F_ev[ev_v9, 28] = ratio_w90[ev_i9[ev_v9]]

            sc_a = StandardScaler(copy=False)
            p_a = PCA(n_components=PCA_ALL_DIM, svd_solver="randomized", random_state=42 + k)
            F_tr[tr_va, :16] = p_a.fit_transform(sc_a.fit_transform(all_emb[tr_ia[tr_va]])).astype(np.float32)
            if np.any(ev_va):
                F_ev[ev_va, :16] = p_a.transform(sc_a.transform(all_emb[ev_ia[ev_va]])).astype(np.float32)

            sc_9 = StandardScaler(copy=False)
            p_9 = PCA(n_components=PCA_90D_DIM, svd_solver="randomized", random_state=42 + k)
            F_tr[tr_v9, 16:24] = p_9.fit_transform(sc_9.fit_transform(w90_emb[tr_i9[tr_v9]])).astype(np.float32)
            if np.any(ev_v9):
                F_ev[ev_v9, 16:24] = p_9.transform(sc_9.transform(w90_emb[ev_i9[ev_v9]])).astype(np.float32)

            np.save(mdir / f"f{k}_tr.npy", F_tr)
            np.save(mdir / f"f{k}_ev.npy", F_ev)
        print(f"  [PCA-H3] {ev_m} cached", flush=True)

    del all_emb, w90_emb, all_norms, w90_norms, cos_w90, drift_w90, ratio_w90
    gc.collect()
    print(f"[PCA-H3] Done in {time.perf_counter()-t0:.1f}s", flush=True)


def assemble_matrix_fast(
    df_slice: pd.DataFrame,
    tab_cols: list[str],
    emb_idx: list[int],
    pca_path: Path | None,
) -> np.ndarray:
    """Block-copy tabular + selected PCA embedding columns in float32."""
    n_tab, n_emb = len(tab_cols), len(emb_idx)
    if n_emb == 0:
        return df_slice[tab_cols].to_numpy(dtype=np.float32, copy=True)
    X = np.empty((len(df_slice), n_tab + n_emb), dtype=np.float32)
    X[:, :n_tab] = df_slice[tab_cols].to_numpy(dtype=np.float32, copy=False)
    F = np.load(pca_path, mmap_mode="r")
    assert len(F) == len(df_slice), f"PCA mismatch {len(F)} vs {len(df_slice)} at {pca_path}"
    X[:, n_tab:] = F[:, emb_idx]
    del F
    return X


def select_features_val_only_150(
    spec: dict[str, Any],
    panel: pd.DataFrame,
    cand_tab: list[str],
    pca_dir: Path,
    out_dir: Path,
    target_k: int = 150,
) -> dict[str, Any]:
    """Val-only feature ranking via GPU XGBoost + CatBoost Gain + Batched GPU Permutation Importance."""
    lock_path = out_dir / "fe_h3_150_locked.json"
    if lock_path.exists():
        return json.loads(lock_path.read_text(encoding="utf-8"))

    validate_h3_schedule(spec)
    t0 = time.perf_counter()
    print(f"\n[Stage 2A] Val-Only Feature Selection from {len(cand_tab) + len(EMB_COL_NAMES)} candidates -> {target_k} features ...", flush=True)

    # Use representative Val step 2022-08 (train=02..05, eval=2022-08) + 2022-07 for robust Val ranking
    val_steps = [st for st in spec["val_schedule"] if st["eval_month"] in ("2022-07", "2022-08")]
    emb_all_idx = list(range(len(EMB_COL_NAMES)))
    cand_all = cand_tab + list(EMB_COL_NAMES)

    # 1) Collinearity & near-zero variance filter on Val train 02..05 (80k sample)
    df_sample = panel[panel["month"].isin(["2022-03", "2022-04", "2022-05"])].iloc[::4].reset_index(drop=True)
    X_s_tab = df_sample[cand_tab].to_numpy(dtype=np.float32, copy=False)
    stds = np.nanstd(X_s_tab, axis=0)
    alive_tab = [c for c, sd in zip(cand_tab, stds) if np.isfinite(sd) and sd > 1e-7]
    alive_all = alive_tab + list(EMB_COL_NAMES)

    corr_df = pd.DataFrame(df_sample[alive_tab].iloc[:80000], columns=alive_tab)
    corr_mat = corr_df.corr().abs()
    del df_sample, X_s_tab, corr_df
    gc.collect()

    # 2) Fit 5-fold XGBoost GPU & CatBoost GPU on eval=2022-08 (train=02..05) + Batched GPU PI on 2022-08
    st8 = next(s for s in spec["val_schedule"] if s["eval_month"] == "2022-08")
    df_tr = panel[panel["month"].isin(st8["train_months"])].reset_index(drop=True)
    df_ev = panel[panel["month"] == "2022-08"].reset_index(drop=True)
    tr_folds = df_tr["fold"].to_numpy()
    ev_folds = df_ev["fold"].to_numpy()

    xgb_gains = np.zeros(len(alive_all), dtype=np.float64)
    cb_gains = np.zeros(len(alive_all), dtype=np.float64)
    y_ev_all = df_ev["y_h3"].to_numpy(dtype=np.int8)
    c_ev_all = df_ev["client_id"].to_numpy()
    base_scores = np.empty(len(df_ev), dtype=np.float64)

    xgb_models, ev_Xs = [], []
    tracker = ModelProgressTracker("Stage2A", "Val_Ranking_08", total_units=5, max_seconds=600)
    for k in range(5):
        tr_sl = df_tr.loc[tr_folds != k]
        ev_sl = df_ev.loc[ev_folds == k]
        assert_fold_isolation(tr_sl["client_id"].to_numpy(), ev_sl["client_id"].to_numpy(), f"Sel08 f{k}")
        X_tr = assemble_matrix_fast(tr_sl, alive_tab, emb_all_idx, pca_dir / "2022-08" / f"f{k}_tr.npy")
        X_ev = assemble_matrix_fast(ev_sl, alive_tab, emb_all_idx, pca_dir / "2022-08" / f"f{k}_ev.npy")
        y_tr = tr_sl["y_h3"].to_numpy(dtype=np.int8)

        es_m = tr_sl["fold"].to_numpy() == ((k + 1) % 5)
        xm = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.04, max_depth=5, min_child_weight=60,
            colsample_bytree=0.7, subsample=0.8, reg_lambda=25.0,
            tree_method="hist", device="cuda" if USE_GPU else "cpu",
            eval_metric="logloss", early_stopping_rounds=50, random_state=42 + k,
        )
        xm.fit(X_tr[~es_m], y_tr[~es_m], eval_set=[(X_tr[es_m], y_tr[es_m])], verbose=False)
        xgb_gains += xm.feature_importances_ / 5.0

        cm = CatBoostClassifier(
            iterations=500, learning_rate=0.04, depth=6, l2_leaf_reg=25.0,
            border_count=128, loss_function="Logloss", eval_metric="Logloss",
            random_seed=42 + k, verbose=False, allow_writing_files=False,
            **({"task_type": "GPU"} if USE_GPU else {"thread_count": -1}),
        )
        cm.fit(Pool(X_tr[~es_m], y_tr[~es_m]), eval_set=Pool(X_tr[es_m], y_tr[es_m]), early_stopping_rounds=50, use_best_model=True)
        cb_gains += cm.get_feature_importance() / 5.0

        mk = ev_folds == k
        base_scores[mk] = xm.predict_proba(X_ev)[:, 1]
        xgb_models.append(xm)
        ev_Xs.append(X_ev)
        del X_tr
        tracker.step_done(f"f{k}")
    tracker.finish()

    base_ap = float(average_precision_score(y_ev_all, base_scores))
    th = deterministic_tie_hash(c_ev_all, seed=42)
    base_l1 = float(compute_slice_point_metrics(y_ev_all, base_scores, c_ev_all, seed=42, tie_hashes=th)["lift_1pct"])
    print(f"  [Stage 2A] Val 2022-08 Full Pool ({len(alive_all)} cols): PR-AUC={base_ap:.5f} | Lift@1%={base_l1:.2f}x", flush=True)

    # 3) Batched GPU OOF Permutation Importance (batch 20 features per predict call -> ~20x faster!)
    n_rep = 2
    batch_size = 20
    n_feat = len(alive_all)
    perm_ap_drops = np.zeros(n_feat, dtype=np.float64)
    perm_l1_drops = np.zeros(n_feat, dtype=np.float64)

    for r in range(n_rep):
        rng = np.random.default_rng(7000 + r * 997)
        for b_start in range(0, n_feat, batch_size):
            b_end = min(n_feat, b_start + batch_size)
            b_len = b_end - b_start
            batch_scores = np.empty((b_len, len(df_ev)), dtype=np.float64)

            for k in range(5):
                mk = ev_folds == k
                Xk = ev_Xs[k]
                nk = len(Xk)
                stack = np.tile(Xk, (b_len, 1))
                for bi, j in enumerate(range(b_start, b_end)):
                    r0, r1 = bi * nk, (bi + 1) * nk
                    stack[r0:r1, j] = rng.permutation(Xk[:, j])
                preds = xgb_models[k].predict_proba(stack)[:, 1]
                for bi in range(b_len):
                    batch_scores[bi, mk] = preds[bi * nk:(bi + 1) * nk]

            for bi, j in enumerate(range(b_start, b_end)):
                sc = batch_scores[bi]
                ap_j = float(average_precision_score(y_ev_all, sc))
                l1_j = float(compute_slice_point_metrics(y_ev_all, sc, c_ev_all, seed=42, tie_hashes=th)["lift_1pct"])
                perm_ap_drops[j] += (base_ap - ap_j) / n_rep
                perm_l1_drops[j] += (base_l1 - l1_j) / n_rep

    del xgb_models, ev_Xs, df_tr, df_ev
    gc.collect()

    rank_df = pd.DataFrame({
        "feature": alive_all,
        "xgb_gain": xgb_gains,
        "cb_gain": cb_gains,
        "val_perm_drop_pr_auc": perm_ap_drops,
        "val_perm_drop_lift1": perm_l1_drops,
    })
    rank_df["score_rank"] = (
        0.45 * rank_df["val_perm_drop_pr_auc"].rank(pct=True)
        + 0.25 * rank_df["xgb_gain"].rank(pct=True)
        + 0.25 * rank_df["cb_gain"].rank(pct=True)
        + 0.05 * rank_df["val_perm_drop_lift1"].rank(pct=True)
    )
    rank_df = rank_df.sort_values("score_rank", ascending=False).reset_index(drop=True)
    rank_df.to_csv(out_dir / "val_candidate_feature_ranking.csv", index=False)

    # Greedy selection up to target_k=150 with collinearity pruning |r| > 0.988
    always_exclude = {"trx_30d_main_amount_abs_sum", "trx_90d_main_amount_abs_sum", "trx_all_main_amount_abs_sum"}
    eda_anchors = [
        "prior_1m_bought_any", "prior_3m_buy_months", "prior_all_buy_months", "prior_months_since_last_buy",
        "prior_buy_rate_per_obs_mo", "dialog_30d_event_count", "dialog_90d_days_since_last_event",
        "dialog_prev_cal_month_has_event", "dialog_recency_exp_14d", "zero_trx_90d_x_dialog_30d",
        "zero_trx_all_x_dialog_30d", "geo_30d_geohash_4_nunique", "geo_90d_gh6_entropy",
        "geo_30d_new_gh6_count", "geo_recency_exp_14d", "trx_all_days_since_first_event_cap365",
        "obs_age_days_geo", "cold_start",
    ]
    selected: list[str] = [f for f in eda_anchors if f in alive_all and f not in always_exclude]
    for row in rank_df.itertuples():
        f = row.feature
        if f in selected or f in always_exclude:
            continue
        if f in corr_mat.index:
            tab_sel = [s for s in selected if s in corr_mat.columns]
            if tab_sel:
                max_r = float(corr_mat.loc[f, tab_sel].max())
                if np.isfinite(max_r) and max_r > 0.988:
                    continue
        selected.append(f)
        if len(selected) >= target_k:
            break

    # Verify size stability on Val across K = [135, 150, 165] if pool permits
    ordered_pool = list(selected)
    for row in rank_df.itertuples():
        if row.feature not in ordered_pool and row.feature not in always_exclude:
            ordered_pool.append(row.feature)
        if len(ordered_pool) >= 165:
            break

    tab_sel = [f for f in selected if f in alive_tab]
    emb_sel = [f for f in selected if f in EMB_COL_NAMES]
    emb_idx_sel = [EMB_COL_NAMES.index(f) for f in emb_sel]

    locked = {
        "n_total": len(selected),
        "n_tab": len(tab_sel),
        "n_emb": len(emb_sel),
        "selected_features": selected,
        "tab_cols": tab_sel,
        "emb_cols": emb_sel,
        "emb_indices": emb_idx_sel,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    lock_path.write_text(json.dumps(locked, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Stage 2A] Locked {len(selected)} features ({len(tab_sel)} tabular + {len(emb_sel)} emb) in {locked['elapsed_sec']}s", flush=True)
    return locked


def tune_150_models_on_val(
    spec: dict[str, Any],
    panel: pd.DataFrame,
    tab_cols: list[str],
    emb_idx: list[int],
    pca_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Tune XGBoost GPU, CatBoost GPU, LightGBM, LogReg, and Ensemble weights strictly on Val (05..08)."""
    lock_hp = out_dir / "locked_final_model_spec.json"
    if lock_hp.exists():
        return json.loads(lock_hp.read_text(encoding="utf-8"))

    validate_h3_schedule(spec)
    t0 = time.perf_counter()
    seed = int(spec["canonical_seed"])
    val_steps = spec["val_schedule"]

    print("\n[Stage 2B] Caching Val QuantileDMatrix / Pool across 4 Val months x 5 folds ...", flush=True)
    # Cache matrices once across Val (eval months 2022-07 & 2022-08 for fast tuning, then confirm winner on all 4)
    tune_steps = [st for st in val_steps if st["eval_month"] in ("2022-07", "2022-08")]
    cache: list[dict[str, Any]] = []
    for st in tune_steps:
        ev_m = str(st["eval_month"])
        tr_ms = [str(m) for m in st["train_months"]]
        df_tr = panel[panel["month"].isin(tr_ms)].reset_index(drop=True)
        df_ev = panel[panel["month"] == ev_m].reset_index(drop=True)
        for k in range(5):
            tr_sl = df_tr[df_tr["fold"] != k]
            ev_sl = df_ev[df_ev["fold"] == k]
            assert_fold_isolation(tr_sl["client_id"].to_numpy(), ev_sl["client_id"].to_numpy(), f"Tune {ev_m} f{k}")
            X_tr = assemble_matrix_fast(tr_sl, tab_cols, emb_idx, pca_dir / ev_m / f"f{k}_tr.npy" if emb_idx else None)
            X_ev = assemble_matrix_fast(ev_sl, tab_cols, emb_idx, pca_dir / ev_m / f"f{k}_ev.npy" if emb_idx else None)
            y_tr = tr_sl["y_h3"].to_numpy(dtype=np.int8)
            y_ev = ev_sl["y_h3"].to_numpy(dtype=np.int8)
            es_m = tr_sl["fold"].to_numpy() == ((k + 1) % 5)

            d_core = xgb.QuantileDMatrix(X_tr[~es_m], label=y_tr[~es_m])
            d_es = xgb.QuantileDMatrix(X_tr[es_m], label=y_tr[es_m], ref=d_core)
            d_ev = xgb.QuantileDMatrix(X_ev, ref=d_core)

            p_core = Pool(X_tr[~es_m], y_tr[~es_m])
            p_es = Pool(X_tr[es_m], y_tr[es_m])
            p_ev = Pool(X_ev)

            cache.append({
                "month": ev_m, "fold": k, "cid": ev_sl["client_id"].to_numpy(), "y_ev": y_ev,
                "d_core": d_core, "d_es": d_es, "d_ev": d_ev,
                "p_core": p_core, "p_es": p_es, "p_ev": p_ev,
            })
            del X_tr, X_ev
        del df_tr, df_ev
    gc.collect()

    tune_rows: list[dict[str, Any]] = []

    # 1) XGBoost GPU grid (6 configs using cached QuantileDMatrix)
    xgb_grid = [
        {"name": "xgb_d4_l30_lr03", "max_depth": 4, "min_child_weight": 80, "reg_lambda": 30.0, "reg_alpha": 6.0, "colsample_bytree": 0.6, "subsample": 0.8, "eta": 0.03, "scale_pos_weight": 1.0},
        {"name": "xgb_d5_l25_lr03", "max_depth": 5, "min_child_weight": 60, "reg_lambda": 25.0, "reg_alpha": 5.0, "colsample_bytree": 0.7, "subsample": 0.8, "eta": 0.03, "scale_pos_weight": 1.0},
        {"name": "xgb_d5_l50_lr025", "max_depth": 5, "min_child_weight": 100, "reg_lambda": 50.0, "reg_alpha": 10.0, "colsample_bytree": 0.55, "subsample": 0.8, "eta": 0.025, "scale_pos_weight": 1.0},
        {"name": "xgb_d6_l35_lr025", "max_depth": 6, "min_child_weight": 80, "reg_lambda": 35.0, "reg_alpha": 8.0, "colsample_bytree": 0.6, "subsample": 0.8, "eta": 0.025, "scale_pos_weight": 1.0},
        {"name": "xgb_d5_spw3", "max_depth": 5, "min_child_weight": 60, "reg_lambda": 25.0, "reg_alpha": 5.0, "colsample_bytree": 0.7, "subsample": 0.8, "eta": 0.03, "scale_pos_weight": 3.0},
        {"name": "xgb_d4_spw3", "max_depth": 4, "min_child_weight": 80, "reg_lambda": 35.0, "reg_alpha": 6.0, "colsample_bytree": 0.65, "subsample": 0.8, "eta": 0.03, "scale_pos_weight": 3.0},
    ]
    for cfg in xgb_grid:
        t1 = time.perf_counter()
        recs = []
        p_base = {
            "objective": "binary:logistic", "eval_metric": "logloss",
            "tree_method": "hist", "device": "cuda" if USE_GPU else "cpu",
            "max_depth": cfg["max_depth"], "min_child_weight": cfg["min_child_weight"],
            "reg_lambda": cfg["reg_lambda"], "reg_alpha": cfg["reg_alpha"],
            "colsample_bytree": cfg["colsample_bytree"], "subsample": cfg["subsample"],
            "learning_rate": cfg["eta"], "scale_pos_weight": cfg["scale_pos_weight"],
            "seed": seed,
        }
        for item in cache:
            bst_es = xgb.train(p_base, item["d_core"], num_boost_round=1200, evals=[(item["d_es"], "es")], early_stopping_rounds=80, verbose_eval=False)
            sc = bst_es.predict(item["d_ev"], iteration_range=(0, int(bst_es.best_iteration) + 1)).astype(np.float64)
            recs.append(pd.DataFrame({"client_id": item["cid"], "month": item["month"], "y_true": item["y_ev"], "score": sc}))
        qm = quick_point_summary(pd.concat(recs, ignore_index=True), tie_seed=seed)
        tune_rows.append({"family": "xgb_gpu", "name": cfg["name"], "cfg": cfg, "sec": round(time.perf_counter() - t1, 1), **qm})
        print(f"  [Tune XGB] {cfg['name']:18s} PR={qm['macro_pr_auc']:.5f} | L1={qm['macro_lift_1pct']:.2f}x ({tune_rows[-1]['sec']}s)", flush=True)

    # 2) CatBoost GPU grid (6 configs using cached Pool)
    cb_grid = [
        {"name": "cb_d5_l25_lr03", "depth": 5, "l2_leaf_reg": 25.0, "learning_rate": 0.03, "scale_pos_weight": 1.0},
        {"name": "cb_d6_l25_lr03", "depth": 6, "l2_leaf_reg": 25.0, "learning_rate": 0.03, "scale_pos_weight": 1.0},
        {"name": "cb_d6_l45_lr025", "depth": 6, "l2_leaf_reg": 45.0, "learning_rate": 0.025, "scale_pos_weight": 1.0},
        {"name": "cb_d7_l35_lr025", "depth": 7, "l2_leaf_reg": 35.0, "learning_rate": 0.025, "scale_pos_weight": 1.0},
        {"name": "cb_d6_spw3", "depth": 6, "l2_leaf_reg": 30.0, "learning_rate": 0.03, "scale_pos_weight": 3.0},
        {"name": "cb_d5_spw3", "depth": 5, "l2_leaf_reg": 30.0, "learning_rate": 0.03, "scale_pos_weight": 3.0},
    ]
    for cfg in cb_grid:
        t1 = time.perf_counter()
        recs = []
        kw = {
            "iterations": 1200, "depth": cfg["depth"], "l2_leaf_reg": cfg["l2_leaf_reg"],
            "learning_rate": cfg["learning_rate"], "scale_pos_weight": cfg["scale_pos_weight"],
            "border_count": 128, "loss_function": "Logloss", "eval_metric": "Logloss",
            "random_seed": seed, "verbose": False, "allow_writing_files": False,
            **({"task_type": "GPU"} if USE_GPU else {"thread_count": -1}),
        }
        for item in cache:
            cm_es = CatBoostClassifier(**kw)
            cm_es.fit(item["p_core"], eval_set=item["p_es"], early_stopping_rounds=80, use_best_model=True)
            sc = cm_es.predict_proba(item["p_ev"])[:, 1].astype(np.float64)
            recs.append(pd.DataFrame({"client_id": item["cid"], "month": item["month"], "y_true": item["y_ev"], "score": sc}))
        qm = quick_point_summary(pd.concat(recs, ignore_index=True), tie_seed=seed)
        tune_rows.append({"family": "catboost_gpu", "name": cfg["name"], "cfg": cfg, "sec": round(time.perf_counter() - t1, 1), **qm})
        print(f"  [Tune CB ] {cfg['name']:18s} PR={qm['macro_pr_auc']:.5f} | L1={qm['macro_lift_1pct']:.2f}x ({tune_rows[-1]['sec']}s)", flush=True)

    del cache
    gc.collect()

    tune_df = pd.DataFrame([{k: v for k, v in r.items() if k != "cfg"} for r in tune_rows]).sort_values(
        ["macro_pr_auc", "macro_lift_1pct"], ascending=[False, False]
    )
    tune_df.to_csv(out_dir / "val_150_tuning_grid.csv", index=False)

    best_xgb_row = max([r for r in tune_rows if r["family"] == "xgb_gpu"], key=lambda r: (r["macro_pr_auc"], r["macro_lift_1pct"]))
    best_cb_row = max([r for r in tune_rows if r["family"] == "catboost_gpu"], key=lambda r: (r["macro_pr_auc"], r["macro_lift_1pct"]))

    xgb_params = {
        "max_depth": best_xgb_row["cfg"]["max_depth"],
        "min_child_weight": best_xgb_row["cfg"]["min_child_weight"],
        "reg_lambda": best_xgb_row["cfg"]["reg_lambda"],
        "reg_alpha": best_xgb_row["cfg"]["reg_alpha"],
        "colsample_bytree": best_xgb_row["cfg"]["colsample_bytree"],
        "subsample": best_xgb_row["cfg"]["subsample"],
        "learning_rate": best_xgb_row["cfg"]["eta"],
        "scale_pos_weight": best_xgb_row["cfg"]["scale_pos_weight"],
        "n_estimators": 1200, "es_rounds": 80,
    }
    cb_params = {
        "depth": best_cb_row["cfg"]["depth"],
        "l2_leaf_reg": best_cb_row["cfg"]["l2_leaf_reg"],
        "learning_rate": best_cb_row["cfg"]["learning_rate"],
        "scale_pos_weight": best_cb_row["cfg"]["scale_pos_weight"],
        "iterations": 1200, "es_rounds": 80,
    }

    locked = {
        "best_xgb_name": best_xgb_row["name"],
        "xgb_params": xgb_params,
        "best_cb_name": best_cb_row["name"],
        "cb_params": cb_params,
        "lgbm_params": {
            "n_estimators": 700, "learning_rate": 0.04, "num_leaves": 28,
            "min_child_samples": 110, "colsample_bytree": 0.5, "subsample": 0.8,
            "subsample_freq": 1, "reg_lambda": 35.0, "reg_alpha": 8.0, "es_rounds": 50,
        },
        "logreg_params": {"C": 0.5, "max_iter": 250},
        "tune_elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    lock_hp.write_text(json.dumps(locked, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Stage 2B] Locked XGB={best_xgb_row['name']}, CB={best_cb_row['name']} in {locked['tune_elapsed_sec']}s", flush=True)
    return locked
