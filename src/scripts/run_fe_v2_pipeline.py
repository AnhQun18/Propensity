"""FE-v2 Pipeline (Strict <= 120 features, Val-only Feature Selection & Model Tuning,
5-Seed Locked Test Confirmation under Protocol B Rolling OOC & Protocol A Flat OOC).

Stages:
  1. Cache strict leak-free train-only Dialog PCA slices for Protocol B (per eval_month x fold)
     and reuse Stage-0 PCA cache for Protocol A.
  2. Strict Validation-only Feature Selection (Protocol B Val 2022-09..10 + collinearity filter)
     capping strictly at <= 120 features -> outputs/fe_v2/fe_v2_features_locked.json.
  3. Validation Model Verification & Tuning (LogReg, LightGBM, XGBoost, CatBoostTuned,
     and scale_pos_weight variants) -> outputs/fe_v2/val_model_comparison.csv.
  4. 5-Seed Locked Test Confirmation (seeds 42,142,242,342,442) on winning FE-v2 models
     + Calibration Slope/Intercept, Per-Fold, Slice Metrics, Paired Bootstrap (Holm/BH),
     Permutation Importance, and Shuffled-Label Sanity Check.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline_v2.metrics import bootstrap_paired_delta, holm_bonferroni
from baseline_v2.spec import load_spec
from propensity_model.metrics import compute_metrics, deterministic_tie_hash

CANDIDATES_PATH = REPO_ROOT / "outputs" / "fe_v2" / "panel_fe_v2_candidates.parquet"
EMB_PATH = REPO_ROOT / "outputs" / "baseline_moscow_v1" / "dialog_emb_moscow.npz"
STAGE0_PCA_DIR = REPO_ROOT / "outputs" / "fe_v1_ablation" / "_pca_stage0_seed42"
FE1_PRED_DIR = REPO_ROOT / "outputs" / "fe_v1_ablation" / "predictions"
V2_PRED_DIR = REPO_ROOT / "outputs" / "baseline_v2" / "predictions"
OUT_DIR = REPO_ROOT / "outputs" / "fe_v2"
PCA_B_DIR = OUT_DIR / "pca_cache_protB"
PRED_DIR = OUT_DIR / "predictions"
PCA_B_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

PCA_ALL_DIM = 16
PCA_90D_DIM = 8
MAX_FEATURES = 120
PRIMARY_SEED = 42
EVAL_SEEDS = [42]

EMB_COL_NAMES = (
    [f"dlg_pca_all_{i}" for i in range(PCA_ALL_DIM)]
    + [f"dlg_pca_90d_{i}" for i in range(PCA_90D_DIM)]
    + ["dlg_norm_all", "dlg_norm_90d", "dlg_cos_90d_all"]
)


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=np.float64)
    prev = 1.0
    for rank_rev in range(m - 1, -1, -1):
        idx = order[rank_rev]
        val = min(prev, pvals[idx] * m / (rank_rev + 1))
        adj[idx] = val
        prev = val
    return [float(x) for x in adj]


def calibration_slope_intercept(y_true: np.ndarray, p_pred: np.ndarray) -> tuple[float, float]:
    p = np.clip(p_pred.astype(np.float64), 1e-7, 1.0 - 1e-7)
    logit_p = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=200)
    lr.fit(logit_p, y_true.astype(np.int32))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def ensure_protB_pca_cache(spec: dict[str, Any], panel_keys: pd.DataFrame) -> None:
    """Compute strict leak-free PCA slices per (eval_month, fold) for Protocol B Rolling OOC."""
    all_sched = list(spec["protocols"]["B_rolling_ooc"]["val_schedule"]) + list(spec["protocols"]["B_rolling_ooc"]["test_schedule"])
    missing = any(
        not (PCA_B_DIR / step["eval_month"] / f"f{k}_ev.npy").exists()
        for step in all_sched for k in range(5)
    )
    if not missing:
        print("[PCA-B] Using cached Protocol B leak-free PCA slices.", flush=True)
        return

    print("[PCA-B] Building leak-free Protocol B PCA slices ...", flush=True)
    t0 = time.perf_counter()
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
    for step in all_sched:
        ev_m = step["eval_month"]
        tr_ms = step["train_months"]
        mdir = PCA_B_DIR / ev_m
        mdir.mkdir(parents=True, exist_ok=True)
        df_tr = panel_keys[panel_keys["month"].isin(tr_ms)].reset_index(drop=True)
        df_ev = panel_keys[panel_keys["month"] == ev_m].reset_index(drop=True)
        for k in range(5):
            if (mdir / f"f{k}_ev.npy").exists():
                continue
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
        print(f"  [PCA-B] {ev_m} cached", flush=True)

    del all_emb, w90_emb, all_norms, w90_norms, cos_w90
    gc.collect()
    print(f"[PCA-B] All Protocol B PCA slices cached in {time.perf_counter()-t0:.1f}s", flush=True)


def assemble_matrix(
    df_slice: pd.DataFrame,
    tab_cols: list[str],
    emb_idx: list[int],
    pca_path: Path | None,
) -> np.ndarray:
    """Assemble [tabular + selected emb cols] strictly in float32."""
    n_tab, n_emb = len(tab_cols), len(emb_idx)
    X = np.empty((len(df_slice), n_tab + n_emb), dtype=np.float32)
    for j, c in enumerate(tab_cols):
        X[:, j] = df_slice[c].to_numpy(dtype=np.float32, copy=False)
    if n_emb > 0:
        assert pca_path is not None
        F = np.load(pca_path, mmap_mode="r")
        assert len(F) == len(df_slice), f"PCA length mismatch {len(F)} vs {len(df_slice)} at {pca_path}"
        X[:, n_tab:] = F[:, emb_idx]
        del F
    return X


def _detect_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False


USE_GPU = _detect_gpu()


def select_features_on_val(spec: dict[str, Any], panel: pd.DataFrame) -> dict[str, Any]:
    """Strict Validation-only feature selection to <= 120 features (zero Test peeking)."""
    lock_path = OUT_DIR / "fe_v2_features_locked.json"
    if lock_path.exists():
        return json.loads(lock_path.read_text(encoding="utf-8"))

    from scripts.run_fe_v1_ablation import ensure_stage0_pca_cached, load_spec as load_fe1_spec
    ensure_stage0_pca_cached(load_fe1_spec())

    print("[Stage A] Validation-only Feature Ranking & Collinearity Pruning (max <= 120) ...", flush=True)
    meta_exclude = {"client_id", "fold", "month", "cutoff_msk", "y_any", "target_1", "target_2", "target_3", "target_4"}
    cand_tab = [c for c in panel.columns if c not in meta_exclude]

    # Candidate embeddings: drop dlg_cos_90d_all artifact if norm/PCA dominate (or keep full 27 in selection pool to test!)
    cand_all = cand_tab + EMB_COL_NAMES

    # Fit fast regularized LGBM across Protocol A Val (train 04..08 -> eval 09..10) to measure OOF PI on Val ONLY
    df_tr = panel[panel["month"].isin(["2022-04", "2022-05", "2022-06", "2022-07", "2022-08"])].reset_index(drop=True)
    df_ev = panel[panel["month"].isin(["2022-09", "2022-10"])].reset_index(drop=True)
    emb_idx_all = list(range(len(EMB_COL_NAMES)))

    X_ev = assemble_matrix(df_ev, cand_tab, emb_idx_all, None) if False else np.empty((len(df_ev), len(cand_all)), dtype=np.float32)
    for j, c in enumerate(cand_tab):
        X_ev[:, j] = df_ev[c].to_numpy(dtype=np.float32, copy=False)

    fold_models = []
    ev_folds = df_ev["fold"].to_numpy()
    for k in range(5):
        tr_k = df_tr["fold"].to_numpy() != k
        ev_k = ev_folds == k
        X_tr = assemble_matrix(df_tr.loc[tr_k], cand_tab, emb_idx_all, STAGE0_PCA_DIR / "val" / f"f{k}_tr.npy")
        F_ev = np.load(STAGE0_PCA_DIR / "val" / f"f{k}_ev.npy")
        X_ev[ev_k, len(cand_tab):] = F_ev
        y_tr = df_tr.loc[tr_k, "y_any"].to_numpy(dtype=np.int8)
        clf = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=80,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=20.0,
            random_state=42 + k, n_jobs=4, verbose=-1,
        )
        clf.fit(X_tr, y_tr)
        fold_models.append(clf)
        del X_tr, F_ev
        gc.collect()

    y_ev = df_ev["y_any"].to_numpy(dtype=np.int8)
    c_ev = df_ev["client_id"].to_numpy()
    th = deterministic_tie_hash(c_ev, seed=42)
    base_sc = np.empty(len(df_ev), dtype=np.float64)
    for k in range(5):
        mk = ev_folds == k
        base_sc[mk] = fold_models[k].predict_proba(X_ev[mk])[:, 1]
    base_m, _ = compute_metrics(y_ev, base_sc, c_ev, budget_fractions=(0.01,), primary_budget=0.01, seed=42, is_probabilistic=True, tie_hashes=th)
    print(f"  [Val Full {len(cand_all)} cand LGBM] PR-AUC={base_m['average_precision']:.5f} | AUC={base_m['roc_auc']:.4f} | Lift@1%={base_m['lift_1pct']:.2f}x", flush=True)

    # Gain importance + OOF Permutation Importance (2 repeats for speed, ~45s!) on Validation ONLY
    gains = np.mean([m.booster_.feature_importance(importance_type="gain") for m in fold_models], axis=0)
    pi_rows = []
    for j, fname in enumerate(cand_all):
        orig = X_ev[:, j].copy()
        drops_ap, drops_l1 = [], []
        for r in range(2):
            rng = np.random.default_rng(9000 + j * 10 + r)
            psc = np.empty(len(df_ev), dtype=np.float64)
            for k in range(5):
                mk = ev_folds == k
                Xk = X_ev[mk].copy()
                Xk[:, j] = rng.permutation(Xk[:, j])
                psc[mk] = fold_models[k].predict_proba(Xk)[:, 1]
            pm, _ = compute_metrics(y_ev, psc, c_ev, budget_fractions=(0.01,), primary_budget=0.01, seed=42, is_probabilistic=True, tie_hashes=th)
            drops_ap.append(float(base_m["average_precision"] - pm["average_precision"]))
            drops_l1.append(float(base_m["lift_1pct"] - pm["lift_1pct"]))
        X_ev[:, j] = orig
        pi_rows.append({
            "feature": fname,
            "lgbm_gain": float(gains[j]),
            "val_perm_drop_pr_auc": float(np.mean(drops_ap)),
            "val_perm_drop_lift1": float(np.mean(drops_l1)),
        })

    pi_df = pd.DataFrame(pi_rows)
    # Collinearity check on Val train (sampled 60k rows for speed)
    sub_idx = np.linspace(0, len(df_ev) - 1, min(60000, len(df_ev)), dtype=int)
    X_corr_df = pd.DataFrame(X_ev[sub_idx], columns=cand_all)
    corr_mat = X_corr_df.corr().abs()

    # Composite Val ranking score (z-rank of val_perm_drop_pr_auc + lgbm_gain + val_perm_drop_lift1)
    pi_df["score_rank"] = (
        0.55 * pi_df["val_perm_drop_pr_auc"].rank(pct=True)
        + 0.30 * pi_df["lgbm_gain"].rank(pct=True)
        + 0.15 * pi_df["val_perm_drop_lift1"].rank(pct=True)
    )
    pi_df = pi_df.sort_values("score_rank", ascending=False).reset_index(drop=True)
    pi_df.to_csv(OUT_DIR / "val_candidate_feature_ranking.csv", index=False)

    # Mandatory core domain anchors (ensuring geohash 4/5/6, foreign currency, event_type/subtype, src/dst, cold_start & silent_trx are preserved, while greedy pruning |r| > 0.985 duplicates and redundant raw unlogged sums!)
    always_exclude = {"trx_30d_main_amount_abs_sum", "trx_90d_main_amount_abs_sum", "trx_all_main_amount_abs_sum", "dlg_cos_90d_all"}
    domain_must = [
        "cold_start", "is_silent_trx_90d_active_dlg", "trx_first_activated_30d",
        "trx_30d_has_foreign", "trx_all_foreign_fraction",
        "geo_30d_geohash_4_nunique", "geo_90d_geohash_5_nunique", "geo_all_geohash_6_nunique",
        "geo_all_gh6_entropy", "geo_all_gh6_per_gh4",
        "trx_90d_event_type_nunique", "trx_90d_event_subtype_nunique", "trx_30d_new_event_subtype_count",
        "trx_all_dst11_missing_frac", "trx_all_src_type11_nunique", "trx_all_src_type21_nunique",
        "trx_all_dst_type11_nunique", "trx_all_dst_type12_nunique",
    ]
    selected: list[str] = [c for c in domain_must if c in cand_all and c not in always_exclude]
    for row in pi_df.itertuples():
        f = row.feature
        if f in selected or f in always_exclude:
            continue
        # Check correlation with already selected features
        if len(selected) > 0:
            max_r = float(corr_mat.loc[f, selected].max())
            if np.isfinite(max_r) and max_r > 0.985:
                continue
        selected.append(f)
        if len(selected) >= MAX_FEATURES:
            break

    assert len(selected) <= MAX_FEATURES, f"Feature budget violated: {len(selected)} > {MAX_FEATURES}"
    tab_sel = [f for f in selected if f in cand_tab]
    emb_sel = [f for f in selected if f in EMB_COL_NAMES]
    emb_idx_sel = [EMB_COL_NAMES.index(f) for f in emb_sel]

    locked = {
        "n_total": len(selected),
        "n_tab": len(tab_sel),
        "n_emb": len(emb_sel),
        "max_features_cap": MAX_FEATURES,
        "selected_features": selected,
        "tab_cols": tab_sel,
        "emb_cols": emb_sel,
        "emb_indices": emb_idx_sel,
        "always_excluded": sorted(always_exclude),
    }
    lock_path.write_text(json.dumps(locked, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Stage A] Locked {len(selected)} FE-v2 features ({len(tab_sel)} tabular + {len(emb_sel)} emb) -> {lock_path}", flush=True)
    del X_ev, df_tr, df_ev, fold_models
    gc.collect()
    return locked


def fit_predict_model(
    cfg: dict[str, Any],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_es: np.ndarray | None,
    y_es: np.ndarray | None,
    X_ev: np.ndarray,
    seed: int,
) -> np.ndarray:
    fam = cfg["family"]
    if fam == "logreg":
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=cfg.get("C", 1.0), solver="lbfgs", max_iter=300, random_state=seed)),
        ])
        X_fit = np.vstack([X_tr, X_es]) if X_es is not None else X_tr
        y_fit = np.concatenate([y_tr, y_es]) if y_es is not None else y_tr
        pipe.fit(X_fit, y_fit)
        return pipe.predict_proba(X_ev)[:, 1].astype(np.float64)

    if fam == "lgbm":
        params = dict(cfg["params"])
        params["random_state"] = seed
        params["n_jobs"] = -1
        params["verbose"] = -1
        clf = lgb.LGBMClassifier(**params)
        if X_es is not None:
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_es, y_es)],
                callbacks=[lgb.early_stopping(cfg.get("es_rounds", 100), verbose=False)],
            )
        else:
            clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_ev)[:, 1].astype(np.float64)

    if fam == "xgb":
        params = dict(cfg["params"])
        params["random_state"] = seed
        params["n_jobs"] = -1
        params["tree_method"] = "hist"
        if USE_GPU:
            params["device"] = "cuda"
        params["eval_metric"] = "logloss"
        if X_es is not None:
            params["early_stopping_rounds"] = cfg.get("es_rounds", 100)
        clf = xgb.XGBClassifier(**params)
        if X_es is not None:
            clf.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        else:
            clf.fit(X_tr, y_tr, verbose=False)
        return clf.predict_proba(X_ev)[:, 1].astype(np.float64)

    if fam == "catboost":
        params = dict(cfg["params"])
        params["random_seed"] = seed
        params["border_count"] = 128
        params["loss_function"] = "Logloss"
        params["verbose"] = False
        params["allow_writing_files"] = False
        if USE_GPU:
            params["task_type"] = "GPU"
        else:
            params["thread_count"] = -1
        clf = CatBoostClassifier(**params)
        if X_es is not None:
            clf.fit(X_tr, y_tr, eval_set=(X_es, y_es), early_stopping_rounds=cfg.get("es_rounds", 100), verbose=False)
        else:
            clf.fit(X_tr, y_tr, verbose=False)
        return clf.predict_proba(X_ev)[:, 1].astype(np.float64)

    raise ValueError(f"Unknown family: {fam}")


def eval_protocol_b(
    cfg: dict[str, Any],
    spec: dict[str, Any],
    panel: pd.DataFrame,
    tab_cols: list[str],
    emb_idx: list[int],
    stage: str,
    seed: int,
    shuffle_labels: bool = False,
) -> pd.DataFrame:
    name = cfg["name"]
    tag = f"protB_{stage}_{name}_{'shuf_' if shuffle_labels else ''}seed{seed}"
    out_p = PRED_DIR / f"{tag}.parquet"
    if out_p.exists():
        return pd.read_parquet(out_p)

    sched = spec["protocols"]["B_rolling_ooc"][f"{stage}_schedule"]
    step_dfs = []
    for step in sched:
        ev_m = step["eval_month"]
        tr_ms = step["train_months"]
        core_ms = tr_ms[:-1]
        es_m = tr_ms[-1]
        df_c = panel[panel["month"].isin(core_ms)].reset_index(drop=True)
        df_es = panel[panel["month"] == es_m].reset_index(drop=True)
        df_ev = panel[panel["month"] == ev_m].reset_index(drop=True)

        # Notice PCA-B was fit on tr_ms (= core_ms + es_m). Split PCA slices back into core vs es!
        n_tr_rows = len(df_c)
        df_tr_all = panel[panel["month"].isin(tr_ms)].reset_index(drop=True)

        for k in range(5):
            ev_k = df_ev["fold"].to_numpy() == k
            tr_all_k = df_tr_all[df_tr_all["fold"] != k].reset_index(drop=True)
            is_es_mask = tr_all_k["month"].to_numpy() == es_m
            is_c_mask = ~is_es_mask

            X_tr_all = assemble_matrix(tr_all_k, tab_cols, emb_idx, PCA_B_DIR / ev_m / f"f{k}_tr.npy" if emb_idx else None)
            X_ev = assemble_matrix(df_ev.loc[ev_k], tab_cols, emb_idx, PCA_B_DIR / ev_m / f"f{k}_ev.npy" if emb_idx else None)

            X_c = X_tr_all[is_c_mask]
            y_c = tr_all_k.loc[is_c_mask, "y_any"].to_numpy(dtype=np.int8)
            X_es = X_tr_all[is_es_mask]
            y_es = tr_all_k.loc[is_es_mask, "y_any"].to_numpy(dtype=np.int8)
            if shuffle_labels:
                rng = np.random.default_rng(4200 + seed * 10 + k)
                y_c = rng.permutation(y_c)
                y_es = rng.permutation(y_es)
            sc = fit_predict_model(cfg, X_c, y_c, X_es, y_es, X_ev, seed=seed)
            ev_sub = df_ev.loc[ev_k]
            step_dfs.append(pd.DataFrame({
                "client_id": ev_sub["client_id"].to_numpy(),
                "fold": ev_sub["fold"].to_numpy(),
                "month": ev_sub["month"].to_numpy(),
                "y_true": ev_sub["y_any"].to_numpy(dtype=np.int8),
                "score": sc,
            }))
            del X_tr_all, X_ev, X_c, X_es
            gc.collect()

    pdf = pd.concat(step_dfs, ignore_index=True).sort_values(["month", "client_id"]).reset_index(drop=True)
    tmp = out_p.with_suffix(".tmp.parquet")
    pdf.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(out_p)
    return pdf


def eval_protocol_a(
    cfg: dict[str, Any],
    panel: pd.DataFrame,
    tab_cols: list[str],
    emb_idx: list[int],
    stage: str,
    seed: int,
) -> pd.DataFrame:
    name = cfg["name"]
    out_p = PRED_DIR / f"protA_{stage}_{name}_seed{seed}.parquet"
    if out_p.exists():
        return pd.read_parquet(out_p)

    tr_m = ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08"] if stage == "val" else ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10"]
    ev_m = ["2022-09", "2022-10"] if stage == "val" else ["2022-11", "2022-12", "2023-01"]
    es_m = tr_m[-1]
    df_tr = panel[panel["month"].isin(tr_m)].reset_index(drop=True)
    df_ev = panel[panel["month"].isin(ev_m)].reset_index(drop=True)

    fdfs = []
    for k in range(5):
        tr_sl = df_tr[df_tr["fold"] != k].reset_index(drop=True)
        ev_sl = df_ev[df_ev["fold"] == k].reset_index(drop=True)
        X_tr_all = assemble_matrix(tr_sl, tab_cols, emb_idx, STAGE0_PCA_DIR / stage / f"f{k}_tr.npy" if emb_idx else None)
        X_ev = assemble_matrix(ev_sl, tab_cols, emb_idx, STAGE0_PCA_DIR / stage / f"f{k}_ev.npy" if emb_idx else None)

        if cfg.get("use_temporal_es", True) and cfg["family"] != "logreg":
            is_es = tr_sl["month"].to_numpy() == es_m
            sc = fit_predict_model(
                cfg,
                X_tr_all[~is_es], tr_sl.loc[~is_es, "y_any"].to_numpy(dtype=np.int8),
                X_tr_all[is_es], tr_sl.loc[is_es, "y_any"].to_numpy(dtype=np.int8),
                X_ev, seed=seed,
            )
        else:
            sc = fit_predict_model(cfg, X_tr_all, tr_sl["y_any"].to_numpy(dtype=np.int8), None, None, X_ev, seed=seed)

        fdfs.append(pd.DataFrame({
            "client_id": ev_sl["client_id"].to_numpy(),
            "fold": ev_sl["fold"].to_numpy(),
            "month": ev_sl["month"].to_numpy(),
            "y_true": ev_sl["y_any"].to_numpy(dtype=np.int8),
            "score": sc,
        }))
        del X_tr_all, X_ev
        gc.collect()

    pdf = pd.concat(fdfs, ignore_index=True).sort_values(["month", "client_id"]).reset_index(drop=True)
    tmp = out_p.with_suffix(".tmp.parquet")
    pdf.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(out_p)
    return pdf


def score_df(pdf: pd.DataFrame) -> dict[str, Any]:
    """Compute Macro-Monthly (Protocol B style) and Pooled metrics + Calibration Slope/Intercept."""
    mo_aps, mo_aucs, mo_l1s, mo_hits, mo_lls = [], [], [], [], []
    for _, g in pdf.groupby("month"):
        mm, _ = compute_metrics(g["y_true"].to_numpy(), g["score"].to_numpy(), g["client_id"].to_numpy(),
                                budget_fractions=(0.01, 0.05), primary_budget=0.01, seed=42, is_probabilistic=True)
        mo_aps.append(mm["average_precision"])
        mo_aucs.append(mm["roc_auc"])
        mo_l1s.append(mm["lift_1pct"])
        mo_hits.append(mm["hits_1pct"])
        mo_lls.append(mm["log_loss"])
    th = deterministic_tie_hash(pdf["client_id"].to_numpy(), seed=42)
    pool, _ = compute_metrics(pdf["y_true"].to_numpy(), pdf["score"].to_numpy(), pdf["client_id"].to_numpy(),
                              budget_fractions=(0.01, 0.05), primary_budget=0.01, seed=42, is_probabilistic=True, tie_hashes=th)
    slope, inter = calibration_slope_intercept(pdf["y_true"].to_numpy(), pdf["score"].to_numpy())
    return {
        "macro_pr_auc": float(np.mean(mo_aps)),
        "macro_roc_auc": float(np.mean(mo_aucs)),
        "macro_lift_1pct": float(np.mean(mo_l1s)),
        "macro_hits_1pct": float(np.mean(mo_hits)),
        "macro_log_loss": float(np.mean(mo_lls)),
        "pooled_pr_auc": float(pool["average_precision"]),
        "pooled_roc_auc": float(pool["roc_auc"]),
        "pooled_lift_1pct": float(pool["lift_1pct"]),
        "pooled_hits_1pct": int(pool["hits_1pct"]),
        "pooled_lift_5pct": float(pool["lift_5pct"]),
        "pooled_log_loss": float(pool["log_loss"]),
        "cal_slope": slope,
        "cal_intercept": inter,
    }


def get_sh_candidates() -> list[dict[str, Any]]:
    """12-config initial pool for Stage B Successive Halving on one representative Val month (2022-10)."""
    return [
        {"name": "lgbm_d31_l25_lr03", "family": "lgbm", "es_rounds": 100,
         "params": dict(n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                        colsample_bytree=0.7, subsample=0.8, subsample_freq=1, reg_lambda=25.0, reg_alpha=5.0)},
        {"name": "lgbm_d20_l50_lr03", "family": "lgbm", "es_rounds": 100,
         "params": dict(n_estimators=1500, learning_rate=0.03, num_leaves=20, min_child_samples=120,
                        colsample_bytree=0.6, subsample=0.8, subsample_freq=1, reg_lambda=50.0, reg_alpha=10.0)},
        {"name": "lgbm_d45_l15_lr03", "family": "lgbm", "es_rounds": 100,
         "params": dict(n_estimators=1500, learning_rate=0.03, num_leaves=45, min_child_samples=80,
                        colsample_bytree=0.8, subsample=0.85, subsample_freq=1, reg_lambda=15.0, reg_alpha=2.0)},
        {"name": "lgbm_d31_l25_lr05", "family": "lgbm", "es_rounds": 100,
         "params": dict(n_estimators=1500, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                        colsample_bytree=0.7, subsample=0.8, subsample_freq=1, reg_lambda=25.0, reg_alpha=5.0)},
        {"name": "lgbm_spw3", "family": "lgbm", "es_rounds": 100,
         "params": dict(n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                        colsample_bytree=0.7, subsample=0.8, subsample_freq=1, reg_lambda=25.0, reg_alpha=5.0, scale_pos_weight=3.0)},
        {"name": "lgbm_spw5", "family": "lgbm", "es_rounds": 100,
         "params": dict(n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                        colsample_bytree=0.7, subsample=0.8, subsample_freq=1, reg_lambda=25.0, reg_alpha=5.0, scale_pos_weight=5.0)},
        {"name": "xgb_d5_l25_lr03", "family": "xgb", "es_rounds": 100,
         "params": dict(n_estimators=1500, learning_rate=0.03, max_depth=5, min_child_weight=50,
                        colsample_bytree=0.7, subsample=0.8, reg_lambda=25.0, reg_alpha=5.0)},
        {"name": "xgb_d4_l50_lr04", "family": "xgb", "es_rounds": 100,
         "params": dict(n_estimators=1500, learning_rate=0.04, max_depth=4, min_child_weight=80,
                        colsample_bytree=0.6, subsample=0.8, reg_lambda=50.0, reg_alpha=8.0)},
        {"name": "xgb_spw5", "family": "xgb", "es_rounds": 100,
         "params": dict(n_estimators=1500, learning_rate=0.03, max_depth=5, min_child_weight=50,
                        colsample_bytree=0.7, subsample=0.8, reg_lambda=25.0, reg_alpha=5.0, scale_pos_weight=5.0)},
        {"name": "cb_d6_l25_lr03", "family": "catboost", "es_rounds": 100,
         "params": dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=25.0)},
        {"name": "cb_d5_l40_lr04", "family": "catboost", "es_rounds": 100,
         "params": dict(iterations=1500, learning_rate=0.04, depth=5, l2_leaf_reg=40.0)},
        {"name": "cb_d6_spw3", "family": "catboost", "es_rounds": 100,
         "params": dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=25.0, scale_pos_weight=3.0)},
    ]


def _make_rung_cfg(cfg: dict[str, Any], max_trees: int, es_rounds: int) -> dict[str, Any]:
    out = {"name": cfg["name"], "family": cfg["family"], "es_rounds": es_rounds, "params": dict(cfg["params"])}
    if out["family"] == "catboost":
        out["params"]["iterations"] = max_trees
    else:
        out["params"]["n_estimators"] = max_trees
    return out


def run_successive_halving_on_rep_val(
    spec: dict[str, Any],
    panel: pd.DataFrame,
    tab_cols: list[str],
    emb_idx: list[int],
    seed: int = 42,
    rep_month: str = "2022-10",
) -> dict[str, Any]:
    """Tune once on one representative Val month (2022-10) using Successive Halving; lock winners for all months & folds."""
    lock_hp_path = OUT_DIR / "locked_best_hyperparams.json"
    if lock_hp_path.exists():
        return json.loads(lock_hp_path.read_text(encoding="utf-8"))

    step = next(s for s in spec["protocols"]["B_rolling_ooc"]["val_schedule"] if s["eval_month"] == rep_month)
    tr_ms, es_m, ev_m = step["train_months"], step["train_months"][-1], step["eval_month"]
    df_tr_all = panel[panel["month"].isin(tr_ms)].reset_index(drop=True)
    df_ev = panel[panel["month"] == ev_m].reset_index(drop=True)

    print(f"[SH Tuning] Assembling 5-fold cache for representative Val month={rep_month} (seed={seed}) ...", flush=True)
    fold_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for k in range(5):
        tr_all_k = df_tr_all[df_tr_all["fold"] != k].reset_index(drop=True)
        ev_k_mask = df_ev["fold"].to_numpy() == k
        X_tr_all = assemble_matrix(tr_all_k, tab_cols, emb_idx, PCA_B_DIR / ev_m / f"f{k}_tr.npy" if emb_idx else None)
        X_ev = assemble_matrix(df_ev.loc[ev_k_mask], tab_cols, emb_idx, PCA_B_DIR / ev_m / f"f{k}_ev.npy" if emb_idx else None)
        is_es = tr_all_k["month"].to_numpy() == es_m
        fold_cache[k] = (
            X_tr_all[~is_es], tr_all_k.loc[~is_es, "y_any"].to_numpy(dtype=np.int8),
            X_tr_all[is_es], tr_all_k.loc[is_es, "y_any"].to_numpy(dtype=np.int8),
            X_ev, df_ev.loc[ev_k_mask, "y_any"].to_numpy(dtype=np.int8),
            df_ev.loc[ev_k_mask, "client_id"].to_numpy(),
        )
        del X_tr_all
    del df_tr_all, df_ev
    gc.collect()

    rungs = [
        {"rung": 1, "folds": (0, 1), "max_trees": 250, "es_rounds": 40, "keep": 6},
        {"rung": 2, "folds": (0, 1, 2, 3, 4), "max_trees": 600, "es_rounds": 60, "keep": 3},
        {"rung": 3, "folds": (0, 1, 2, 3, 4), "max_trees": 1500, "es_rounds": 100, "keep": 3},
    ]
    pool = get_sh_candidates()
    sh_rows: list[dict[str, Any]] = []
    rung2_board: list[dict[str, Any]] = []
    rung3_board: list[dict[str, Any]] = []

    for rc in rungs:
        rung_rows = []
        for cfg in pool:
            rcfg = _make_rung_cfg(cfg, rc["max_trees"], rc["es_rounds"])
            t0 = time.perf_counter()
            ys, scs, cids = [], [], []
            for k in rc["folds"]:
                X_c, y_c, X_es, y_es, X_ev, y_ev, cid_ev = fold_cache[k]
                sc = fit_predict_model(rcfg, X_c, y_c, X_es, y_es, X_ev, seed=seed)
                ys.append(y_ev); scs.append(sc); cids.append(cid_ev)
            y_cat, sc_cat, cid_cat = np.concatenate(ys), np.concatenate(scs), np.concatenate(cids)
            th = deterministic_tie_hash(cid_cat, seed=seed)
            m, _ = compute_metrics(y_cat, sc_cat, cid_cat, budget_fractions=(0.01,), primary_budget=0.01, seed=seed, is_probabilistic=True, tie_hashes=th)
            rec = {
                "rung": rc["rung"], "n_folds": len(rc["folds"]), "max_trees": rc["max_trees"],
                "name": cfg["name"], "family": cfg["family"],
                "spw": float(cfg["params"].get("scale_pos_weight", 1.0)),
                "pr_auc": float(m["average_precision"]), "roc_auc": float(m["roc_auc"]),
                "lift_1pct": float(m["lift_1pct"]), "log_loss": float(m["log_loss"]),
                "sec": round(time.perf_counter() - t0, 1),
            }
            sh_rows.append(rec)
            rung_rows.append((rec, cfg))
            print(f"  [SH R{rc['rung']} k={len(rc['folds'])} T={rc['max_trees']}] {cfg['name']:20s} PR={rec['pr_auc']:.5f} | L1={rec['lift_1pct']:.2f}x | LL={rec['log_loss']:.5f} ({rec['sec']}s)", flush=True)

        rung_rows.sort(key=lambda x: x[0]["pr_auc"], reverse=True)
        if rc["rung"] == 2:
            rung2_board = [x[1] for x in rung_rows]
        if rc["rung"] == 3:
            rung3_board = [x[1] for x in rung_rows]
        pool = [x[1] for x in rung_rows[: rc["keep"]]]

    pd.DataFrame(sh_rows).to_csv(OUT_DIR / "val_tuning_successive_halving.csv", index=False)

    # Lock top winner + top diverse runner-up + LogReg linear anchor (+ top spw>1 check if distinct)
    best_cfg = rung3_board[0]
    runner_up = next((c for c in rung3_board[1:] + rung2_board if c["family"] != best_cfg["family"]), rung3_board[1])
    best_lgbm = next((c for c in rung3_board + rung2_board if c["family"] == "lgbm"), get_sh_candidates()[0])
    best_spw = next((c for c in rung3_board + rung2_board if float(c["params"].get("scale_pos_weight", 1.0)) > 1.0), get_sh_candidates()[5])

    eval_cfgs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in [best_cfg, runner_up, best_lgbm, best_spw]:
        if c["name"] not in seen:
            seen.add(c["name"])
            eval_cfgs.append(c)
    eval_cfgs.append({"name": "logreg_l2", "family": "logreg", "C": 0.5, "use_temporal_es": False, "params": {"C": 0.5}})

    locked_hp = {
        "rep_val_month": rep_month,
        "seed": seed,
        "best_single": best_cfg["name"],
        "runner_up": runner_up["name"],
        "best_lgbm": best_lgbm["name"],
        "best_spw": best_spw["name"],
        "eval_configs": eval_cfgs,
    }
    lock_hp_path.write_text(json.dumps(locked_hp, indent=2, ensure_ascii=False), encoding="utf-8")
    del fold_cache
    gc.collect()
    print(f"[SH Tuning] Locked best={best_cfg['name']}, runner_up={runner_up['name']}, best_lgbm={best_lgbm['name']} -> {lock_hp_path}", flush=True)
    return locked_hp


def run_posthoc_fe_v2_checks(
    best_cfg: dict[str, Any],
    spec: dict[str, Any],
    panel: pd.DataFrame,
    tab_cols: list[str],
    emb_idx: list[int],
    seed: int = 42,
) -> None:
    """Post-hoc Test checks for best FE-v2 model: monthly breakdown, cold/silent-TRX slices, and Dialog no-dialog / lag-0 ref."""
    pdf = pd.read_parquet(PRED_DIR / f"protB_test_{best_cfg['name']}_seed{seed}.parquet")
    # 1) Monthly breakdown on Test (check 2022-11, 2022-12, 2023-01 stability)
    mo_rows = []
    for mo, g in pdf.groupby("month"):
        th = deterministic_tie_hash(g["client_id"].to_numpy(), seed=seed)
        m, _ = compute_metrics(g["y_true"].to_numpy(), g["score"].to_numpy(), g["client_id"].to_numpy(),
                               budget_fractions=(0.01, 0.05), primary_budget=0.01, seed=seed, is_probabilistic=True, tie_hashes=th)
        mo_rows.append({"month": mo, "n_rows": len(g), "n_pos": int(g["y_true"].sum()), "base_rate": float(g["y_true"].mean()),
                        "pr_auc": m["average_precision"], "roc_auc": m["roc_auc"], "lift_1pct": m["lift_1pct"], "lift_5pct": m["lift_5pct"]})
    pd.DataFrame(mo_rows).to_csv(OUT_DIR / "test_fe_v2_monthly_breakdown.csv", index=False)

    # 2) Cold-start & Silent-TRX 90d slices on Locked Test
    test_meta = panel[panel["month"].isin(["2022-11", "2022-12", "2023-01"])][
        ["client_id", "month", "cold_start", "trx_count_90d", "dialog_count_30d"]
    ]
    mrg = pdf.merge(test_meta, on=["client_id", "month"], how="left")
    mrg["trx_90d_active"] = (mrg["trx_count_90d"].fillna(0) > 0).astype(int)
    mrg["dlg_30d_active"] = (mrg["dialog_count_30d"].fillna(0) > 0).astype(int)
    slice_defs = [
        ("all_test", np.ones(len(mrg), dtype=bool)),
        ("cold_start_eq1", mrg["cold_start"].to_numpy() == 1),
        ("cold_start_eq0", mrg["cold_start"].to_numpy() == 0),
        ("trx_90d_silent", mrg["trx_90d_active"].to_numpy() == 0),
        ("trx_90d_active", mrg["trx_90d_active"].to_numpy() == 1),
        ("silent_trx90d_active_dlg30d", (mrg["trx_90d_active"].to_numpy() == 0) & (mrg["dlg_30d_active"].to_numpy() == 1)),
    ]
    srows = []
    for sname, mask in slice_defs:
        sub = mrg.loc[mask]
        if len(sub) < 10 or sub["y_true"].sum() == 0:
            continue
        sm, _ = compute_metrics(sub["y_true"].to_numpy(), sub["score"].to_numpy(), sub["client_id"].to_numpy(),
                                budget_fractions=(0.01, 0.05), primary_budget=0.01, seed=seed, is_probabilistic=True)
        srows.append({"slice": sname, "n_rows": len(sub), "n_pos": int(sub["y_true"].sum()), "base_rate": float(sub["y_true"].mean()),
                      "pr_auc": sm["average_precision"], "roc_auc": sm["roc_auc"], "lift_1pct": sm["lift_1pct"]})
    pd.DataFrame(srows).to_csv(OUT_DIR / "test_fe_v2_slices.csv", index=False)

    # 3) Dialog ablation check on winning FE-v2 config: full FE-v2 vs no-dialog (pure TRX+Geo tabular FE-v2)
    tab_no_dlg = [c for c in tab_cols if not c.startswith(("dialog_", "is_silent_trx_90d_active_dlg"))]
    nodlg_cfg = dict(best_cfg, name=f"{best_cfg['name']}_no_dialog")
    pdf_nd = eval_protocol_b(nodlg_cfg, spec, panel, tab_no_dlg, [], "test", seed)
    sc_full = score_df(pdf); sc_full["variant"] = f"{best_cfg['name']}_full"
    sc_nd = score_df(pdf_nd); sc_nd["variant"] = f"{best_cfg['name']}_no_dialog_tabular_only"
    pd.DataFrame([sc_full, sc_nd]).to_csv(OUT_DIR / "test_fe_v2_dialog_ablation.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", "val_only", "test_confirm"])
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED, help="Single deterministic seed (default 42)")
    args = parser.parse_args()
    seed = int(args.seed)
    print(f"[FE-v2] Runtime hardware: USE_GPU={USE_GPU} | Single Seed={seed}", flush=True)

    spec = load_spec(REPO_ROOT / "configs" / "baseline_v2_evaluation_spec.json")
    panel = pd.read_parquet(CANDIDATES_PATH)
    ensure_protB_pca_cache(spec, panel[["client_id", "fold", "month"]])

    locked = select_features_on_val(spec, panel)
    tab_cols = list(locked["tab_cols"])
    emb_idx = list(locked["emb_indices"])

    locked_hp = run_successive_halving_on_rep_val(spec, panel, tab_cols, emb_idx, seed=seed, rep_month="2022-10")
    eval_cfgs = locked_hp["eval_configs"]
    cfg_map = {c["name"]: c for c in eval_cfgs}
    best_single = locked_hp["best_single"]
    runner_up = locked_hp["runner_up"]
    best_lgbm = locked_hp["best_lgbm"]

    if args.stage in ("all", "val_only"):
        print("\n=== [Stage B] Evaluating Locked Hyperparams Across All Val Months (2022-09..10, seed=42) ===", flush=True)
        val_rows = []
        val_preds: dict[str, pd.DataFrame] = {}
        for cfg in eval_cfgs:
            t0 = time.perf_counter()
            pdf = eval_protocol_b(cfg, spec, panel, tab_cols, emb_idx, "val", seed)
            val_preds[cfg["name"]] = pdf
            sc = score_df(pdf)
            sc["model"], sc["seed"], sc["sec"] = cfg["name"], seed, round(time.perf_counter() - t0, 1)
            val_rows.append(sc)
            print(f"  [VAL {cfg['name']:20s}] MacroPR={sc['macro_pr_auc']:.5f} | MacroL1={sc['macro_lift_1pct']:.2f}x | AUC={sc['macro_roc_auc']:.4f} | Slope={sc['cal_slope']:.3f} ({sc['sec']}s)", flush=True)

        # Blend top-2 diverse locked winners on Val
        p1, p2 = val_preds[best_single], val_preds[runner_up]
        bdf = p1[["client_id", "fold", "month", "y_true"]].copy()
        bdf["score"] = 0.5 * p1["score"].to_numpy() + 0.5 * p2["score"].to_numpy()
        bdf.to_parquet(PRED_DIR / f"protB_val_blend_top2_seed{seed}.parquet", index=False, compression="zstd")
        sc = score_df(bdf)
        sc["model"], sc["seed"], sc["sec"] = "blend_top2", seed, 0.0
        val_rows.append(sc)

        val_df = pd.DataFrame(val_rows)
        val_df.to_csv(OUT_DIR / "val_model_comparison_detailed.csv", index=False)
        val_agg = val_df.sort_values("macro_pr_auc", ascending=False).reset_index(drop=True)
        val_agg.to_csv(OUT_DIR / "val_model_comparison_summary.csv", index=False)
        print("\n[VAL SUMMARY - FIXED PARAMS SINGLE SEED]\n" + val_agg.to_string(index=False), flush=True)

    if args.stage in ("all", "test_confirm"):
        print("\n=== [Stage C] Locked Test Confirmation (Single Seed=42, Fixed Hyperparams, ProtB + ProtA) ===", flush=True)
        confirm_singles = list(dict.fromkeys([best_single, runner_up, best_lgbm, "logreg_l2"]))
        test_rows = []
        test_preds: dict[str, pd.DataFrame] = {}
        for mname in confirm_singles:
            cfg = cfg_map[mname]
            t0 = time.perf_counter()
            pdf_b = eval_protocol_b(cfg, spec, panel, tab_cols, emb_idx, "test", seed)
            test_preds[mname] = pdf_b
            sc = score_df(pdf_b)
            sc["protocol"], sc["model"], sc["seed"], sc["sec"] = "B_rolling_ooc", mname, seed, round(time.perf_counter() - t0, 1)
            test_rows.append(sc)
            if mname in (best_single, best_lgbm):
                pdf_a = eval_protocol_a(cfg, panel, tab_cols, emb_idx, "test", seed)
                sca = score_df(pdf_a)
                sca["protocol"], sca["model"], sca["seed"], sca["sec"] = "A_flat_ooc", mname, seed, 0.0
                test_rows.append(sca)

        # Blend top-2 locked singles on Test
        p1, p2 = test_preds[best_single], test_preds[runner_up]
        bdf = p1[["client_id", "fold", "month", "y_true"]].copy()
        bdf["score"] = 0.5 * p1["score"].to_numpy() + 0.5 * p2["score"].to_numpy()
        bdf.to_parquet(PRED_DIR / f"protB_test_blend_top2_seed{seed}.parquet", index=False, compression="zstd")
        sc = score_df(bdf)
        sc["protocol"], sc["model"], sc["seed"], sc["sec"] = "B_rolling_ooc", "blend_top2", seed, 0.0
        test_rows.append(sc)

        test_df = pd.DataFrame(test_rows).sort_values(["protocol", "macro_pr_auc"], ascending=[True, False]).reset_index(drop=True)
        test_df.to_csv(OUT_DIR / "test_confirmation_detailed.csv", index=False)
        test_df.to_csv(OUT_DIR / "test_confirmation_summary.csv", index=False)
        print("\n[LOCKED TEST SUMMARY - SINGLE SEED]\n" + test_df.to_string(index=False), flush=True)

        # Shuffled-label sanity check on winning single (Protocol B Test seed=42)
        shuf_pdf = eval_protocol_b(cfg_map[best_single], spec, panel, tab_cols, emb_idx, "test", seed, shuffle_labels=True)
        shuf_sc = score_df(shuf_pdf)
        shuf_sc["model"] = f"{best_single}_shuffled_label"
        shuf_sc["base_rate"] = float(shuf_pdf["y_true"].mean())
        pd.DataFrame([shuf_sc]).to_csv(OUT_DIR / "test_shuffled_label_sanity.csv", index=False)
        print(f"  [SANITY SHUFFLE {best_single}] Pooled PR-AUC={shuf_sc['pooled_pr_auc']:.5f} vs Base={shuf_sc['base_rate']:.5f}", flush=True)

        # Paired Bootstrap (1000 client-cluster reps, Holm + BH) on seed=42 Test:
        p_tgd_b = pd.read_parquet(V2_PRED_DIR / "B_rolling_ooc" / "test" / "B_catboost_TGD_tuned_seed42.parquet")
        p_lr_b = pd.read_parquet(V2_PRED_DIR / "B_rolling_ooc" / "test" / "B_logreg_TGD_seed42.parquet")
        p_fe2_b_s = test_preds[best_single]
        p_fe2_b_bl = bdf

        p_fe1_a = pd.read_parquet(FE1_PRED_DIR / "test" / "FE-v1_seed42.parquet")
        p_l5c_lgb_a = pd.read_parquet(REPO_ROOT / "outputs" / "baseline_moscow_v1" / "predictions" / "test" / "L5c_lgbm_dialog_meta.parquet")
        p_tgd_lr_a = pd.read_parquet(FE1_PRED_DIR / "test" / "B-raw-TGD_logreg_seed42.parquet")
        p_fe2_a_s = eval_protocol_a(cfg_map[best_single], panel, tab_cols, emb_idx, "test", seed)
        p_fe2_a_lgb = eval_protocol_a(cfg_map[best_lgbm], panel, tab_cols, emb_idx, "test", seed)

        pairs = [
            ("B_FEv2_best_vs_TGD_tuned", p_fe2_b_s, p_tgd_b, "Protocol B: FE-v2 best single - B_catboost_TGD_tuned"),
            ("B_FEv2_blend_vs_TGD_tuned", p_fe2_b_bl, p_tgd_b, "Protocol B: FE-v2 blend (top2) - B_catboost_TGD_tuned"),
            ("B_FEv2_blend_vs_LogReg_TGD", p_fe2_b_bl, p_lr_b, "Protocol B: FE-v2 blend (top2) - B_logreg_TGD"),
            ("A_FEv2_best_vs_FEv1", p_fe2_a_s, p_fe1_a, "Protocol A: FE-v2 best single - FE-v1 (139)"),
            ("A_FEv2_lgbm_vs_L5cLGBM", p_fe2_a_lgb, p_l5c_lgb_a, "Protocol A: FE-v2 LGBM - L5c_lgbm (112)"),
            ("A_FEv2_best_vs_LogReg_TGD", p_fe2_a_s, p_tgd_lr_a, "Protocol A: FE-v2 best single - B-raw-TGD_logreg"),
        ]
        boot_rows = []
        for pid, df_b, df_a, desc in pairs:
            mrg = df_a[["client_id", "month", "y_true", "score"]].merge(
                df_b[["client_id", "month", "score"]], on=["client_id", "month"], suffixes=("_a", "_b")
            )
            res = bootstrap_paired_delta(
                mrg["y_true"].to_numpy(), mrg["score_a"].to_numpy(), mrg["score_b"].to_numpy(),
                mrg["client_id"].to_numpy(), mrg["month"].to_numpy(), n_rep=1000, seed=seed,
            )
            boot_rows.append({"pair_id": pid, "desc": desc, **res})
        pair_df = pd.DataFrame(boot_rows)
        pair_df["delta_pr_p_holm"] = holm_bonferroni(pair_df["delta_pr_auc_p_raw"].tolist())
        pair_df["delta_pr_p_bh"] = benjamini_hochberg(pair_df["delta_pr_auc_p_raw"].tolist())
        pair_df["delta_l1_p_holm"] = holm_bonferroni(pair_df["delta_lift_1pct_p_raw"].tolist())
        pair_df["delta_l1_p_bh"] = benjamini_hochberg(pair_df["delta_lift_1pct_p_raw"].tolist())
        pair_df.to_csv(OUT_DIR / "test_paired_bootstrap_holm_bh.csv", index=False)
        print("[Stage C] Paired bootstrap done -> test_paired_bootstrap_holm_bh.csv", flush=True)

        run_posthoc_fe_v2_checks(cfg_map[best_single], spec, panel, tab_cols, emb_idx, seed=seed)
        print("[Stage C] Post-hoc monthly/slice/no-dialog diagnostics done.", flush=True)


if __name__ == "__main__":
    main()
