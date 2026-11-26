"""GPU-accelerated H=3 baseline and tuned model wrappers."""
from __future__ import annotations

import shutil
import subprocess
from typing import Any

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def detect_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False


USE_GPU = detect_gpu()


def rule_prior_rfm_scores(X: np.ndarray, col_names: list[str]) -> np.ndarray:
    """Deterministic rank-average baseline mixing prior purchase history, DIALOG, and TRX."""
    idx = {c: i for i, c in enumerate(col_names)}
    n = len(X)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    parts: list[np.ndarray] = []
    pos_cols = [
        "prior_1m_bought_any",
        "prior_3m_buy_months",
        "prior_all_buy_months",
        "dialog_30d_event_count",
        "trx_30d_log1p_event_count",
    ]
    for c in pos_cols:
        if c not in idx:
            continue
        v = np.nan_to_num(X[:, idx[c]].astype(np.float64), nan=0.0)
        parts.append(np.where(v > 0, rankdata(v, method="average") / n, 0.0))

    rec_cols = [
        "dialog_all_days_since_last_event",
        "trx_30d_days_since_last_event",
    ]
    for c in rec_cols:
        if c not in idx:
            continue
        v = X[:, idx[c]].astype(np.float64)
        valid = np.isfinite(v) & (v >= 0)
        inv = np.where(valid, -v, -1e9)
        parts.append(np.where(valid, rankdata(inv, method="average") / n, 0.0))

    if not parts:
        return np.zeros(n, dtype=np.float64)
    return np.mean(parts, axis=0).astype(np.float64)


def fit_predict_h3_model(
    family: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_ev: np.ndarray,
    *,
    feature_names: list[str],
    seed: int = 42,
    es_mask: np.ndarray | None = None,
    params_override: dict[str, Any] | None = None,
    use_gpu: bool | None = None,
) -> tuple[np.ndarray, int | None]:
    """Fit one H=3 model with strict OOC inner-fold early stopping and refit."""
    gpu = USE_GPU if use_gpu is None else bool(use_gpu)
    ov = dict(params_override or {})

    if family == "rule_prior_rfm":
        return rule_prior_rfm_scores(X_ev, feature_names), None

    if family == "logreg":
        import gc

        c_val = float(ov.get("C", 1.0))
        max_iter = int(ov.get("max_iter", 200))
        n_tr, n_p = X_tr.shape
        nan_mask_tr = ~np.isfinite(X_tr)
        nan_cols = np.flatnonzero(nan_mask_tr.any(axis=0))
        n_ind = len(nan_cols)

        medians = np.nanmedian(X_tr, axis=0).astype(np.float32)
        medians = np.where(np.isfinite(medians), medians, np.float32(0.0))

        X_tr_f = np.empty((n_tr, n_p + n_ind), dtype=np.float32)
        X_tr_f[:, :n_p] = X_tr
        for j in range(n_p):
            mj = nan_mask_tr[:, j]
            if np.any(mj):
                X_tr_f[mj, j] = medians[j]
        for idx_i, j in enumerate(nan_cols):
            X_tr_f[:, n_p + idx_i] = nan_mask_tr[:, j].astype(np.float32)
        del nan_mask_tr

        means = X_tr_f.mean(axis=0, dtype=np.float64).astype(np.float32)
        stds = X_tr_f.std(axis=0, dtype=np.float64).astype(np.float32)
        stds = np.where(stds > 1e-7, stds, np.float32(1.0))
        X_tr_f -= means
        X_tr_f /= stds

        clf = LogisticRegression(C=c_val, solver="lbfgs", tol=1e-3, max_iter=max_iter, random_state=seed)
        clf.fit(X_tr_f, y_tr)
        del X_tr_f
        gc.collect()

        n_ev = X_ev.shape[0]
        nan_mask_ev = ~np.isfinite(X_ev)
        X_ev_f = np.empty((n_ev, n_p + n_ind), dtype=np.float32)
        X_ev_f[:, :n_p] = X_ev
        for j in range(n_p):
            mj = nan_mask_ev[:, j]
            if np.any(mj):
                X_ev_f[mj, j] = medians[j]
        for idx_i, j in enumerate(nan_cols):
            X_ev_f[:, n_p + idx_i] = nan_mask_ev[:, j].astype(np.float32)
        del nan_mask_ev
        X_ev_f -= means
        X_ev_f /= stds

        probs = clf.predict_proba(X_ev_f)[:, 1].astype(np.float64)
        del X_ev_f, clf
        gc.collect()
        return probs, None

    has_es = es_mask is not None and int(np.sum(es_mask)) >= 100 and int(np.sum(~es_mask)) >= 100
    if has_es:
        X_core, y_core = X_tr[~es_mask], y_tr[~es_mask]
        X_es, y_es = X_tr[es_mask], y_tr[es_mask]
    else:
        X_core, y_core = X_tr, y_tr
        X_es, y_es = None, None

    if family == "lgbm":
        es_rounds = int(ov.pop("es_rounds", 80))
        max_trees = int(ov.pop("n_estimators", 900))
        lgb_params = {
            "n_estimators": max_trees,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 100,
            "colsample_bytree": 0.7,
            "subsample": 0.8,
            "subsample_freq": 1,
            "reg_lambda": 25.0,
            "reg_alpha": 5.0,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
            **ov,
        }
        if has_es:
            es_clf = lgb.LGBMClassifier(**lgb_params)
            es_clf.fit(
                X_core,
                y_core,
                eval_set=[(X_es, y_es)],
                callbacks=[lgb.early_stopping(es_rounds, verbose=False)],
            )
            best_n = max(1, int(es_clf.best_iteration_ or max_trees))
            refit_params = {**lgb_params, "n_estimators": best_n}
            clf = lgb.LGBMClassifier(**refit_params)
            clf.fit(X_tr, y_tr)
            return clf.predict_proba(X_ev)[:, 1].astype(np.float64), best_n
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_ev)[:, 1].astype(np.float64), max_trees

    if family == "xgb_gpu":
        es_rounds = int(ov.pop("es_rounds", 80))
        max_trees = int(ov.pop("n_estimators", 1000))
        xgb_params = {
            "n_estimators": max_trees,
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 60,
            "colsample_bytree": 0.7,
            "subsample": 0.8,
            "reg_lambda": 25.0,
            "reg_alpha": 5.0,
            "tree_method": "hist",
            "device": "cuda" if gpu else "cpu",
            "eval_metric": "logloss",
            "random_state": seed,
            "n_jobs": -1,
            **ov,
        }
        if has_es:
            es_clf = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=es_rounds)
            es_clf.fit(X_core, y_core, eval_set=[(X_es, y_es)], verbose=False)
            raw_best = getattr(es_clf, "best_iteration", None)
            best_n = max(1, int(raw_best) + 1) if (raw_best is not None and raw_best >= 0) else max_trees
            refit_params = {**xgb_params, "n_estimators": best_n}
            clf = xgb.XGBClassifier(**refit_params)
            clf.fit(X_tr, y_tr, verbose=False)
            return clf.predict_proba(X_ev)[:, 1].astype(np.float64), best_n
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_tr, y_tr, verbose=False)
        return clf.predict_proba(X_ev)[:, 1].astype(np.float64), max_trees

    if family == "catboost_gpu":
        es_rounds = int(ov.pop("es_rounds", 80))
        max_trees = int(ov.pop("iterations", 1200))
        cb_params = {
            "iterations": max_trees,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 25.0,
            "border_count": 128,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            **({"task_type": "GPU"} if gpu else {"thread_count": -1}),
            **ov,
        }
        if has_es:
            es_clf = CatBoostClassifier(**cb_params)
            es_clf.fit(
                Pool(X_core, y_core),
                eval_set=Pool(X_es, y_es),
                early_stopping_rounds=es_rounds,
                use_best_model=True,
            )
            raw_best = es_clf.get_best_iteration()
            best_n = max(1, int(raw_best) + 1) if (raw_best is not None and raw_best >= 0) else max_trees
            refit_params = {**cb_params, "iterations": best_n}
            clf = CatBoostClassifier(**refit_params)
            clf.fit(Pool(X_tr, y_tr))
            return clf.predict_proba(X_ev)[:, 1].astype(np.float64), best_n
        clf = CatBoostClassifier(**cb_params)
        clf.fit(Pool(X_tr, y_tr))
        return clf.predict_proba(X_ev)[:, 1].astype(np.float64), max_trees

    raise ValueError(f"Unsupported H3 model family: {family}")
