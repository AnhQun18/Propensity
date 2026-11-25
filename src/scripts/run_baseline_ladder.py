"""Run the comprehensive multi-tier baseline ladder (OOT + 5-fold OOT&OOC) for H=1 and H=3.

Tiers evaluated (seed=42, natural distribution, no SMOTE, budgets=[0.5%, 1%, 5%]):
  T0_random_prevalence        : Train base-rate constant + SHA-256 tie-break (Non-ML floor)
  T1a_rule_rf                 : Classic 2-feature RF heuristic (trx_84d count + recency)
  T1b_rule_eda_rfm_dialog     : Transparent non-ML business scorecard (prior + dialog + RFM)
  T2_logreg_simple10          : L2 Logistic Regression on 10 basic interpretable features
  T2b_logreg_top50            : L2 Logistic Regression on 50 engineered features
  T3a_lgbm_default_raw203     : LightGBM default hyperparameters on 203 clean raw features (no FE)
  T3b_xgb_default_raw203      : XGBoost default hyperparameters on 203 clean raw features (no FE)
  T4a_lgbm_default_top50      : LightGBM default hyperparameters on 50 engineered features
  T4b_xgb_default_top50       : XGBoost default hyperparameters on 50 engineered features
  T5_final_catboost_top50     : Regularized CatBoost on 50 engineered features (+ H=1 calibration)

Outputs are saved under outputs/baseline_ladder_v1/ for transparent consumption by
notebooks/10_baseline_benchmark_suite.ipynb.
"""
from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata

from propensity_model.config import assert_temporal_embargo
from propensity_model.data import FORBIDDEN_COLUMNS, load_dataset_contract
from propensity_model.features import TOP50_FEATURES, build_enriched_panel
from propensity_model.metrics import (
    calibration_bins_table,
    compute_metrics,
    deterministic_tie_hash,
)
from propensity_model.models import (
    CatBoostBaseline,
    HeuristicRFBaseline,
    PrevalenceBaseline,
    ProbabilityCalibrator,
    RidgeLogisticBaseline,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "baseline_ladder_v1"
SEED = 42
BUDGET_FRACTIONS = (0.005, 0.01, 0.05)  # Explicitly NO 10% milestone per user request
PRIMARY_BUDGET = 0.01
N_BOOTSTRAP = 1000
BURN_IN_MONTHS = ("2022-02", "2022-03", "2022-04")

SIMPLE10_FEATURES: tuple[str, ...] = (
    "trx_84d_event_count",
    "trx_84d_days_since_last_event",
    "trx_84d_log1p_main_amount_abs_sum",
    "trx_all_days_since_first_recorded_event",
    "trx_30d_event_type_nunique",
    "dialog_30d_has_event",
    "dialog_84d_event_count",
    "geo_30d_geohash_4_nunique",
    "prior_3m_buy_months",
    "cold_start",
)

RULE_EDA_FEATURES: tuple[str, ...] = (
    "prior_3m_buy_months",
    "dialog_30d_has_event",
    "trx_84d_event_count",
    "trx_84d_log1p_main_amount_abs_sum",
    "trx_84d_days_since_last_event",
)

CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 400,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 10.0,
    "rsm": 0.6,
    "border_count": 128,
    "thread_count": 4,
    "used_ram_limit": "3gb",
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "verbose": False,
}

H1_VAL_FOLDS = (
    ("h1_val_2022_09", ("2022-05", "2022-06", "2022-07", "2022-08"), "2022-09"),
    ("h1_val_2022_10", ("2022-05", "2022-06", "2022-07", "2022-08", "2022-09"), "2022-10"),
    ("h1_val_2022_11", ("2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10"), "2022-11"),
)
H3_VAL_FOLDS = (
    ("h3_val_2022_08", ("2022-05",), "2022-08"),
    ("h3_val_2022_09", ("2022-05", "2022-06"), "2022-09"),
    ("h3_val_2022_10", ("2022-05", "2022-06", "2022-07"), "2022-10"),
)
H1_FINAL_TRAIN = ("2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10", "2022-11")
H1_CALIB_MONTH = "2022-12"
H1_TEST_MONTH = "2023-01"
H3_FINAL_TRAIN = ("2022-05", "2022-06", "2022-07", "2022-08")
H3_TEST_MONTH = "2022-11"


class RuleEDAScorecardBaseline:
    """Transparent non-ML business heuristic combining prior purchase, dialog, and RFM."""

    is_probabilistic = False

    def fit(self, X: np.ndarray, y: np.ndarray, cold_start: np.ndarray | None = None):
        return self

    def predict_scores(self, X: np.ndarray, cold_start: np.ndarray | None = None) -> np.ndarray:
        # Columns: prior_3m_buy_months, dialog_30d_has_event, trx_84d_count, trx_84d_log_amt, trx_84d_recency
        prior_m = np.nan_to_num(X[:, 0].astype(np.float64), nan=0.0)
        dialog = np.nan_to_num(X[:, 1].astype(np.float64), nan=0.0)
        freq = np.nan_to_num(X[:, 2].astype(np.float64), nan=0.0)
        monetary = np.nan_to_num(X[:, 3].astype(np.float64), nan=0.0)
        recency = X[:, 4].astype(np.float64)

        n = len(freq)
        freq_pct = np.where(freq > 0, rankdata(freq, method="average") / n, 0.0)
        mon_pct = np.where(monetary > 0, rankdata(monetary, method="average") / n, 0.0)
        rec_valid = np.isfinite(recency) & (recency >= 0)
        inv_rec = np.where(rec_valid, -recency, -1e9)
        rec_pct = np.where(rec_valid, rankdata(inv_rec, method="average") / n, 0.0)

        return (
            3.0 * (prior_m > 0).astype(np.float64)
            + 1.0 * np.minimum(prior_m, 3.0)
            + 2.5 * (dialog > 0).astype(np.float64)
            + 1.5 * freq_pct
            + 1.0 * mon_pct
            + 1.0 * rec_pct
        )


class LightGBMDefaultBaseline:
    """Default-hyperparameter LightGBM keeping semantic NaNs natively."""

    is_probabilistic = True

    def __init__(self, seed: int = SEED):
        self.seed = int(seed)
        self.model_: lgb.LGBMClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, cold_start: np.ndarray | None = None):
        self.model_ = lgb.LGBMClassifier(random_state=self.seed, n_jobs=4, verbose=-1)
        self.model_.fit(X, y)
        return self

    def predict_scores(self, X: np.ndarray, cold_start: np.ndarray | None = None) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.predict_proba(X)[:, 1].astype(np.float64)


class XGBoostDefaultBaseline:
    """Default-hyperparameter XGBoost keeping semantic NaNs natively."""

    is_probabilistic = True

    def __init__(self, seed: int = SEED):
        self.seed = int(seed)
        self.model_: xgb.XGBClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, cold_start: np.ndarray | None = None):
        self.model_ = xgb.XGBClassifier(random_state=self.seed, n_jobs=4, eval_metric="logloss")
        self.model_.fit(X, y)
        return self

    def predict_scores(self, X: np.ndarray, cold_start: np.ndarray | None = None) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.predict_proba(X)[:, 1].astype(np.float64)


@dataclass(frozen=True)
class TierSpec:
    tier_id: str
    tier_group: str
    model_type: str
    feature_set: str
    is_probabilistic: bool
    note: str


TIERS: tuple[TierSpec, ...] = (
    TierSpec("T0_random_prevalence", "0. Non-ML Floor", "prevalence", "none", True, "Mốc sàn: base rate p_train + SHA-256 tie-break"),
    TierSpec("T1a_rule_rf", "1. Non-ML Rule", "heuristic_rf", "heuristic_rf", False, "Rule RF cổ điển 2 biến (trx_84d count + recency)"),
    TierSpec("T1b_rule_eda_scorecard", "1. Non-ML Rule", "rule_eda", "rule_eda", False, "Rule nghiệp vụ từ EDA (prior + dialog + RFM, không ML)"),
    TierSpec("T2_logreg_simple10", "2. Simple ML", "ridge_logreg", "simple10", True, "L2 Logistic Regression (C=1.0) trên 10 biến cơ bản"),
    TierSpec("T2b_logreg_top50", "2. Simple ML", "ridge_logreg", "top50", True, "L2 Logistic Regression (C=1.0) trên 50 biến sau FE"),
    TierSpec("T3a_lgbm_default_raw203", "3. Default GBDT (Raw)", "lgbm_default", "raw_clean_203", True, "LightGBM tham số mặc định trên 203 biến thô (chưa FE)"),
    TierSpec("T3b_xgb_default_raw203", "3. Default GBDT (Raw)", "xgb_default", "raw_clean_203", True, "XGBoost tham số mặc định trên 203 biến thô (chưa FE)"),
    TierSpec("T4a_lgbm_default_top50", "4. Default GBDT (+FE)", "lgbm_default", "top50", True, "LightGBM tham số mặc định trên 50 biến sau FE"),
    TierSpec("T4b_xgb_default_top50", "4. Default GBDT (+FE)", "xgb_default", "top50", True, "XGBoost tham số mặc định trên 50 biến sau FE"),
    TierSpec("T5_final_catboost_top50", "5. Final Model", "catboost", "top50", True, "CatBoost regularized trên 50 biến sau FE (không SMOTE)"),
)

OOC_TIER_IDS: tuple[str, ...] = (
    "T2_logreg_simple10",
    "T3a_lgbm_default_raw203",
    "T3b_xgb_default_raw203",
    "T4a_lgbm_default_top50",
    "T4b_xgb_default_top50",
    "T5_final_catboost_top50",
)


def make_model(model_type: str, seed: int = SEED):
    if model_type == "prevalence":
        return PrevalenceBaseline()
    if model_type == "heuristic_rf":
        return HeuristicRFBaseline()
    if model_type == "rule_eda":
        return RuleEDAScorecardBaseline()
    if model_type == "ridge_logreg":
        return RidgeLogisticBaseline(c=1.0, max_iter=300, seed=seed)
    if model_type == "lgbm_default":
        return LightGBMDefaultBaseline(seed=seed)
    if model_type == "xgb_default":
        return XGBoostDefaultBaseline(seed=seed)
    if model_type == "catboost":
        return CatBoostBaseline(params=CATBOOST_PARAMS, seed=seed)
    raise ValueError(f"Unknown model_type: {model_type}")


def get_feature_sets(schema: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    predictors = tuple(schema["predictors"])
    raw_clean_203 = tuple(
        c
        for c in predictors
        if not (c.endswith("_source_window_overlap_days") or c.endswith("_history_truncated"))
    )
    fsets = {
        "none": (),
        "heuristic_rf": ("trx_84d_event_count", "trx_84d_days_since_last_event"),
        "rule_eda": RULE_EDA_FEATURES,
        "simple10": SIMPLE10_FEATURES,
        "raw_clean_203": raw_clean_203,
        "top50": TOP50_FEATURES,
    }
    for name, cols in fsets.items():
        bad = FORBIDDEN_COLUMNS.intersection(cols)
        if bad:
            raise ValueError(f"Leakage detected in feature set {name}: {bad}")
        if len(cols) != len(set(cols)):
            raise ValueError(f"Duplicate columns in {name}")
    return fsets


def build_client_folds_lookup(dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    src_glob = (REPO_ROOT / "data" / "MBD-mini" / "client_split" / "*" / "*.parquet").as_posix()
    with duckdb.connect() as con:
        con.execute(
            f"""
            COPY (
                SELECT DISTINCT client_id, CAST(fold AS TINYINT) AS fold
                FROM read_parquet('{src_glob}', hive_partitioning=true)
                ORDER BY client_id
            ) TO '{dest.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    return dest


def load_split_arrays(
    enriched_parquet: Path,
    client_folds_parquet: Path,
    train_months: tuple[str, ...],
    eval_month: str,
    horizon: int,
    feature_cols: tuple[str, ...],
    train_fold_filter: str | None = None,
    eval_fold_filter: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    assert_temporal_embargo(train_months, eval_month, horizon, BURN_IN_MONTHS)
    target_col = "y_h1" if horizon == 1 else "y_h3"
    tr_sql = ", ".join(f"'{m}'" for m in train_months)
    feat_select = ", ".join(f"CAST(p.{c} AS FLOAT) AS {c}" for c in feature_cols)
    feat_prefix = f"{feat_select}, " if feat_select else ""

    tr_where = f"p.month IN ({tr_sql}) AND p.{target_col} IS NOT NULL"
    ev_where = f"p.month = '{eval_month}' AND p.{target_col} IS NOT NULL"
    if train_fold_filter:
        tr_where += f" AND ({train_fold_filter})"
    if eval_fold_filter:
        ev_where += f" AND ({eval_fold_filter})"

    with duckdb.connect() as con:
        con.execute("SET memory_limit='3GB'")
        con.execute("SET threads=4")
        tr_df = con.execute(
            f"""
            SELECT {feat_prefix}CAST(p.{target_col} AS TINYINT) AS _y, p.client_id
            FROM read_parquet('{enriched_parquet.as_posix()}') p
            JOIN read_parquet('{client_folds_parquet.as_posix()}') f USING (client_id)
            WHERE {tr_where}
            ORDER BY p.cutoff, p.client_id
            """
        ).fetch_df()
        ev_df = con.execute(
            f"""
            SELECT {feat_prefix}CAST(p.{target_col} AS TINYINT) AS _y,
                   p.client_id, p.month, p.cold_start, p.coverage_slice, f.fold
            FROM read_parquet('{enriched_parquet.as_posix()}') p
            JOIN read_parquet('{client_folds_parquet.as_posix()}') f USING (client_id)
            WHERE {ev_where}
            ORDER BY p.client_id
            """
        ).fetch_df()

    if feature_cols:
        X_tr = tr_df[list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
        X_ev = ev_df[list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
    else:
        X_tr = np.empty((len(tr_df), 0), dtype=np.float32)
        X_ev = np.empty((len(ev_df), 0), dtype=np.float32)

    y_tr = tr_df["_y"].to_numpy(dtype=np.int8)
    y_ev = ev_df["_y"].to_numpy(dtype=np.int8)
    tr_clients = tr_df["client_id"].to_numpy()
    ev_meta = ev_df[["client_id", "month", "cold_start", "coverage_slice", "fold"]].copy()
    return X_tr, y_tr, X_ev, y_ev, ev_meta, tr_clients


# ---------------------------------------------------------------------------
# Fast exact client-clustered bootstrap INCLUDING Brier score
# ---------------------------------------------------------------------------
BOOT_METRICS = (
    "recall_1pct",
    "precision_1pct",
    "lift_1pct",
    "average_precision",
    "roc_auc",
    "log_loss",
    "brier",
)


def _prep_sorted(
    y: np.ndarray,
    scores: np.ndarray,
    client_codes: np.ndarray,
    tie_hashes: np.ndarray,
    is_probabilistic: bool,
) -> dict[str, Any]:
    order = np.lexsort((tie_hashes, -scores))
    y_s = y[order].astype(np.float64)
    s_s = scores[order].astype(np.float64)
    c_s = client_codes[order]

    diff_mask = np.empty(len(s_s), dtype=bool)
    diff_mask[-1] = True
    if len(s_s) > 1:
        diff_mask[:-1] = s_s[:-1] != s_s[1:]
    group_ends = np.flatnonzero(diff_mask)

    row_ll = None
    row_brier = None
    if is_probabilistic:
        p = np.clip(s_s, 1e-15, 1.0 - 1e-15)
        row_ll = -(y_s * np.log(p) + (1.0 - y_s) * np.log(1.0 - p))
        row_brier = (s_s - y_s) ** 2

    return {
        "y_s": y_s,
        "c_s": c_s,
        "group_ends": group_ends,
        "row_ll": row_ll,
        "row_brier": row_brier,
        "is_probabilistic": is_probabilistic,
    }


def _weighted_metrics(prep: dict[str, Any], cw: np.ndarray, primary_budget: float = PRIMARY_BUDGET) -> dict[str, float]:
    rw = cw[prep["c_s"]].astype(np.float64)
    total_n = float(rw.sum())
    if total_n <= 0.0:
        return {k: float("nan") for k in BOOT_METRICS}

    y_s = prep["y_s"]
    pos_w = rw * y_s
    total_pos = float(pos_w.sum())
    total_neg = total_n - total_pos

    cum_w = np.cumsum(rw)
    cum_pos = np.cumsum(pos_w)

    k_target = max(1, int(math.ceil(primary_budget * total_n)))
    idx = int(np.searchsorted(cum_w, k_target, side="left"))
    if idx >= len(cum_w):
        hits = total_pos
    else:
        prev_w = float(cum_w[idx - 1]) if idx > 0 else 0.0
        prev_pos = float(cum_pos[idx - 1]) if idx > 0 else 0.0
        hits = prev_pos + (k_target - prev_w) * float(y_s[idx])

    prec = hits / k_target
    rec = (hits / total_pos) if total_pos > 0.0 else float("nan")
    prev = total_pos / total_n
    lift = (prec / prev) if prev > 0.0 else float("nan")

    if total_pos > 0.0 and total_neg > 0.0:
        ge = prep["group_ends"]
        tp = cum_pos[ge]
        tot = cum_w[ge]
        fp = tot - tp
        delta_tp = np.empty_like(tp)
        delta_tp[0] = tp[0]
        if len(tp) > 1:
            delta_tp[1:] = tp[1:] - tp[:-1]
        nz = tot > 0.0
        prec_t = np.zeros_like(tp)
        prec_t[nz] = tp[nz] / tot[nz]
        ap = float(np.sum((delta_tp / total_pos) * prec_t))

        tpr = np.concatenate(([0.0], tp / total_pos))
        fpr = np.concatenate(([0.0], fp / total_neg))
        auc = float(np.trapezoid(tpr, fpr))
    else:
        ap = float("nan")
        auc = float("nan")

    if prep["is_probabilistic"]:
        ll = float(np.dot(rw, prep["row_ll"]) / total_n)
        bs = float(np.dot(rw, prep["row_brier"]) / total_n)
    else:
        ll = float("nan")
        bs = float("nan")

    return {
        "recall_1pct": float(rec),
        "precision_1pct": float(prec),
        "lift_1pct": float(lift),
        "average_precision": ap,
        "roc_auc": auc,
        "log_loss": ll,
        "brier": bs,
    }


def _ci_stats(vals: np.ndarray) -> tuple[float | None, float | None, float | None]:
    valid = vals[np.isfinite(vals)]
    if len(valid) == 0:
        return None, None, None
    lo = float(np.percentile(valid, 2.5))
    hi = float(np.percentile(valid, 97.5))
    p_two = min(1.0, 2.0 * min(float(np.mean(valid <= 0.0)), float(np.mean(valid >= 0.0))))
    return lo, hi, p_two


def bootstrap_single_with_brier(
    df: pd.DataFrame,
    is_probabilistic: bool,
    n_replicates: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, Any]:
    codes, uniques = pd.factorize(df["client_id"].to_numpy(), sort=True)
    n_clients = len(uniques)
    tie_hashes = deterministic_tie_hash(df["client_id"].to_numpy(), seed=seed)
    prep = _prep_sorted(
        df["y_true"].to_numpy(dtype=np.int8),
        df["score"].to_numpy(dtype=np.float64),
        codes,
        tie_hashes,
        is_probabilistic,
    )
    point = _weighted_metrics(prep, np.ones(n_clients, dtype=np.int64))
    samples = {k: np.empty(n_replicates, dtype=np.float64) for k in BOOT_METRICS}
    rng = np.random.default_rng(seed)
    for b in range(n_replicates):
        cw = np.bincount(rng.integers(0, n_clients, size=n_clients), minlength=n_clients)
        res = _weighted_metrics(prep, cw)
        for k in BOOT_METRICS:
            samples[k][b] = res[k]

    out: dict[str, Any] = {"n_clients": int(n_clients), "n_rows": int(len(df)), "replicates": int(n_replicates)}
    for k in BOOT_METRICS:
        out[k] = float(point[k]) if np.isfinite(point[k]) else None
        lo, hi, _ = _ci_stats(samples[k])
        out[f"{k}_ci_low"] = lo
        out[f"{k}_ci_high"] = hi
    return out


def paired_bootstrap_with_brier(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    is_prob_a: bool,
    is_prob_b: bool,
    n_replicates: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, Any]:
    m = df_a[["client_id", "month", "y_true", "score"]].merge(
        df_b[["client_id", "month", "score"]],
        on=["client_id", "month"],
        suffixes=("_a", "_b"),
        how="inner",
        validate="one_to_one",
    )
    if len(m) != len(df_a) or len(m) != len(df_b):
        raise ValueError("Paired bootstrap alignment mismatch")
    codes, uniques = pd.factorize(m["client_id"].to_numpy(), sort=True)
    n_clients = len(uniques)
    tie_hashes = deterministic_tie_hash(m["client_id"].to_numpy(), seed=seed)
    y = m["y_true"].to_numpy(dtype=np.int8)

    prep_a = _prep_sorted(y, m["score_a"].to_numpy(dtype=np.float64), codes, tie_hashes, is_prob_a)
    prep_b = _prep_sorted(y, m["score_b"].to_numpy(dtype=np.float64), codes, tie_hashes, is_prob_b)
    ones = np.ones(n_clients, dtype=np.int64)
    pt_a = _weighted_metrics(prep_a, ones)
    pt_b = _weighted_metrics(prep_b, ones)

    deltas = {k: np.empty(n_replicates, dtype=np.float64) for k in BOOT_METRICS}
    rng = np.random.default_rng(seed)
    for b in range(n_replicates):
        cw = np.bincount(rng.integers(0, n_clients, size=n_clients), minlength=n_clients)
        ra = _weighted_metrics(prep_a, cw)
        rb = _weighted_metrics(prep_b, cw)
        for k in BOOT_METRICS:
            deltas[k][b] = rb[k] - ra[k]

    out: dict[str, Any] = {"n_clients": int(n_clients), "n_rows": int(len(m)), "replicates": int(n_replicates)}
    for k in BOOT_METRICS:
        d = pt_b[k] - pt_a[k]
        out[f"delta_{k}"] = float(d) if np.isfinite(d) else None
        lo, hi, pval = _ci_stats(deltas[k])
        out[f"delta_{k}_ci_low"] = lo
        out[f"delta_{k}_ci_high"] = hi
        out[f"delta_{k}_p_value"] = pval
    return out


def fit_predict_cached(
    spec: TierSpec,
    enriched_parquet: Path,
    client_folds_parquet: Path,
    feature_cols: tuple[str, ...],
    train_months: tuple[str, ...],
    eval_month: str,
    horizon: int,
    pred_path: Path,
    train_fold_filter: str | None = None,
    eval_fold_filter: str | None = None,
) -> tuple[pd.DataFrame, float]:
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    if pred_path.exists():
        return pd.read_parquet(pred_path), 0.0

    X_tr, y_tr, X_ev, y_ev, ev_meta, tr_clients = load_split_arrays(
        enriched_parquet=enriched_parquet,
        client_folds_parquet=client_folds_parquet,
        train_months=train_months,
        eval_month=eval_month,
        horizon=horizon,
        feature_cols=feature_cols,
        train_fold_filter=train_fold_filter,
        eval_fold_filter=eval_fold_filter,
    )
    if train_fold_filter and eval_fold_filter:
        overlap = set(tr_clients).intersection(set(ev_meta["client_id"].to_numpy()))
        if overlap:
            raise AssertionError(f"Client overlap detected in OOC split: {len(overlap)}")

    t0 = time.perf_counter()
    model = make_model(spec.model_type, seed=SEED)
    model.fit(X_tr, y_tr)
    scores = model.predict_scores(X_ev)
    elapsed = time.perf_counter() - t0

    pdf = ev_meta.copy()
    pdf["y_true"] = y_ev.astype(np.int8)
    pdf["score"] = scores.astype(np.float64)
    pdf.to_parquet(pred_path, index=False, compression="zstd")

    del X_tr, y_tr, X_ev, y_ev, model
    gc.collect()
    return pdf, elapsed


def main():
    t_all = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = OUT_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    contract = load_dataset_contract(REPO_ROOT / "processed")
    feature_sets = get_feature_sets(contract.schema)

    enriched_path = OUT_DIR / "enriched_panel.parquet"
    if not enriched_path.exists():
        print("Building enriched panel ...", flush=True)
        build_enriched_panel(
            processed_parquet=contract.parquet_path,
            destination=enriched_path,
            burn_in_months=BURN_IN_MONTHS,
            memory_limit="3GB",
            threads=4,
        )

    folds_path = build_client_folds_lookup(OUT_DIR / "client_folds.parquet")
    print(f"Ready: enriched={enriched_path.name}, folds={folds_path.name}", flush=True)

    # ------------------------------------------------------------------
    # 1. Rolling OOT Validation (H=1 and H=3) across all 10 tiers
    # ------------------------------------------------------------------
    val_monthly_rows: list[dict[str, Any]] = []
    val_boot_rows: list[dict[str, Any]] = []

    for horizon, folds in ((1, H1_VAL_FOLDS), (3, H3_VAL_FOLDS)):
        prev_ll_by_month: dict[str, float] = {}
        prev_brier_by_month: dict[str, float] = {}
        for spec in TIERS:
            fcols = feature_sets[spec.feature_set]
            fold_dfs: list[pd.DataFrame] = []
            for fold_name, tr_months, ev_month in folds:
                pred_path = OUT_DIR / "predictions" / f"h{horizon}_val" / spec.tier_id / f"{ev_month}.parquet"
                pdf, sec = fit_predict_cached(
                    spec, enriched_path, folds_path, fcols, tr_months, ev_month, horizon, pred_path
                )
                fold_dfs.append(pdf)
                m, _ = compute_metrics(
                    y_true=pdf["y_true"].to_numpy(),
                    scores=pdf["score"].to_numpy(),
                    client_ids=pdf["client_id"].to_numpy(),
                    budget_fractions=BUDGET_FRACTIONS,
                    primary_budget=PRIMARY_BUDGET,
                    seed=SEED,
                    is_probabilistic=spec.is_probabilistic,
                    baseline_logloss=prev_ll_by_month.get(ev_month),
                )
                if spec.model_type == "prevalence" and m["log_loss"] is not None:
                    prev_ll_by_month[ev_month] = float(m["log_loss"])
                    prev_brier_by_month[ev_month] = float(m["brier"])
                    m["log_loss_skill"] = 0.0
                brier_skill = (
                    float(1.0 - m["brier"] / prev_brier_by_month[ev_month])
                    if (m["brier"] is not None and ev_month in prev_brier_by_month and prev_brier_by_month[ev_month] > 0)
                    else None
                )
                val_monthly_rows.append(
                    {
                        "horizon": horizon,
                        "stage": "validation",
                        "fold_name": fold_name,
                        "eval_month": ev_month,
                        "tier_id": spec.tier_id,
                        "tier_group": spec.tier_group,
                        "feature_set": spec.feature_set,
                        "n_features": len(fcols),
                        "train_seconds": round(sec, 2),
                        "brier_skill": brier_skill,
                        **m,
                    }
                )
                print(
                    f"[H={horizon} VAL] {spec.tier_id:26s} | {ev_month} "
                    f"| AP={m['average_precision']:.4f} | AUC={m['roc_auc']:.4f} "
                    f"| Lift@1%={m['lift_1pct']:.2f}x | Brier={m['brier'] if m['brier'] else -1:.5f} ({sec:.1f}s)",
                    flush=True,
                )

            pooled = pd.concat(fold_dfs, ignore_index=True)
            bs = bootstrap_single_with_brier(pooled, spec.is_probabilistic, N_BOOTSTRAP, SEED)
            val_boot_rows.append(
                {
                    "horizon": horizon,
                    "scope": "validation_pooled",
                    "tier_id": spec.tier_id,
                    "tier_group": spec.tier_group,
                    "feature_set": spec.feature_set,
                    "n_features": len(fcols),
                    "note": spec.note,
                    **bs,
                }
            )

    pd.DataFrame(val_monthly_rows).to_csv(metrics_dir / "val_monthly_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Final Locked Holdout (ALL 10 tiers + H=1 Calibration on 2022-12)
    # ------------------------------------------------------------------
    final_rows: list[dict[str, Any]] = []
    final_boot_rows: list[dict[str, Any]] = []
    cal_bin_rows: list[pd.DataFrame] = []
    paired_rows: list[dict[str, Any]] = []
    calibration_info: dict[str, Any] = {}

    final_preds: dict[tuple[int, str], pd.DataFrame] = {}
    final_prob_flags: dict[str, bool] = {s.tier_id: s.is_probabilistic for s in TIERS}
    final_prob_flags["T5_final_cal_intercept"] = True
    final_prob_flags["T5_final_cal_platt"] = True

    for horizon, tr_months, test_month in ((1, H1_FINAL_TRAIN, H1_TEST_MONTH), (3, H3_FINAL_TRAIN, H3_TEST_MONTH)):
        prev_ll = None
        prev_brier = None
        for spec in TIERS:
            fcols = feature_sets[spec.feature_set]
            pred_path = OUT_DIR / "predictions" / f"h{horizon}_final" / f"{spec.tier_id}_{test_month}.parquet"
            pdf, sec = fit_predict_cached(
                spec, enriched_path, folds_path, fcols, tr_months, test_month, horizon, pred_path
            )
            final_preds[(horizon, spec.tier_id)] = pdf
            m, cal_df = compute_metrics(
                y_true=pdf["y_true"].to_numpy(),
                scores=pdf["score"].to_numpy(),
                client_ids=pdf["client_id"].to_numpy(),
                budget_fractions=BUDGET_FRACTIONS,
                primary_budget=PRIMARY_BUDGET,
                seed=SEED,
                is_probabilistic=spec.is_probabilistic,
                baseline_logloss=prev_ll,
            )
            if spec.model_type == "prevalence" and m["log_loss"] is not None:
                prev_ll = float(m["log_loss"])
                prev_brier = float(m["brier"])
                m["log_loss_skill"] = 0.0
            brier_skill = (
                float(1.0 - m["brier"] / prev_brier)
                if (m["brier"] is not None and prev_brier is not None and prev_brier > 0)
                else None
            )
            final_rows.append(
                {
                    "horizon": horizon,
                    "protocol": "panel_oot",
                    "eval_month": test_month,
                    "tier_id": spec.tier_id,
                    "tier_group": spec.tier_group,
                    "feature_set": spec.feature_set,
                    "n_features": len(fcols),
                    "train_seconds": round(sec, 2),
                    "brier_skill": brier_skill,
                    "note": spec.note,
                    **m,
                }
            )
            if len(cal_df) > 0:
                cal_df = cal_df.copy()
                cal_df["horizon"] = horizon
                cal_df["tier_id"] = spec.tier_id
                cal_bin_rows.append(cal_df)

            bs = bootstrap_single_with_brier(pdf, spec.is_probabilistic, N_BOOTSTRAP, SEED)
            final_boot_rows.append(
                {
                    "horizon": horizon,
                    "scope": "final_holdout_panel",
                    "tier_id": spec.tier_id,
                    "tier_group": spec.tier_group,
                    "feature_set": spec.feature_set,
                    "n_features": len(fcols),
                    "note": spec.note,
                    **bs,
                }
            )
            print(
                f"[H={horizon} FINAL] {spec.tier_id:26s} | {test_month} "
                f"| AP={m['average_precision']:.4f} | AUC={m['roc_auc']:.4f} "
                f"| Lift@1%={m['lift_1pct']:.2f}x | Brier={m['brier'] if m['brier'] else -1:.5f} ({sec:.1f}s)",
                flush=True,
            )

        # Post-hoc calibration strictly on 2022-12 for H=1 Final CatBoost
        if horizon == 1:
            t5_spec = next(s for s in TIERS if s.tier_id == "T5_final_catboost_top50")
            fcols = feature_sets[t5_spec.feature_set]
            cal_pred_path = OUT_DIR / "predictions" / "h1_final" / f"T5_cal_month_{H1_CALIB_MONTH}.parquet"
            cal_pdf, _ = fit_predict_cached(
                t5_spec, enriched_path, folds_path, fcols, tr_months, H1_CALIB_MONTH, 1, cal_pred_path
            )
            cal_i = ProbabilityCalibrator("intercept_shift").fit(
                cal_pdf["score"].to_numpy(), cal_pdf["y_true"].to_numpy()
            )
            cal_p = ProbabilityCalibrator("platt").fit(
                cal_pdf["score"].to_numpy(), cal_pdf["y_true"].to_numpy()
            )
            calibration_info["h1_2022_12"] = {
                "intercept_shift": {"slope": cal_i.slope_, "intercept": cal_i.intercept_},
                "platt": {"slope": cal_p.slope_, "intercept": cal_p.intercept_},
                "cal_obs_rate": float(cal_pdf["y_true"].mean()),
                "cal_raw_mean_pred": float(cal_pdf["score"].mean()),
            }

            raw_test_pdf = final_preds[(1, "T5_final_catboost_top50")]
            for cal_id, calibrator, note in (
                ("T5_final_cal_intercept", cal_i, "CatBoost Top-50 + Intercept-Shift calibration (2022-12)"),
                ("T5_final_cal_platt", cal_p, "CatBoost Top-50 + Platt calibration (2022-12)"),
            ):
                cpdf = raw_test_pdf.copy()
                cpdf["score"] = calibrator.transform(raw_test_pdf["score"].to_numpy())
                cpdf.to_parquet(
                    OUT_DIR / "predictions" / "h1_final" / f"{cal_id}_{test_month}.parquet",
                    index=False,
                    compression="zstd",
                )
                final_preds[(1, cal_id)] = cpdf
                m, cal_df = compute_metrics(
                    y_true=cpdf["y_true"].to_numpy(),
                    scores=cpdf["score"].to_numpy(),
                    client_ids=cpdf["client_id"].to_numpy(),
                    budget_fractions=BUDGET_FRACTIONS,
                    primary_budget=PRIMARY_BUDGET,
                    seed=SEED,
                    is_probabilistic=True,
                    baseline_logloss=prev_ll,
                )
                brier_skill = float(1.0 - m["brier"] / prev_brier) if prev_brier else None
                final_rows.append(
                    {
                        "horizon": 1,
                        "protocol": "panel_oot",
                        "eval_month": test_month,
                        "tier_id": cal_id,
                        "tier_group": "5. Final Model (Calibrated)",
                        "feature_set": "top50",
                        "n_features": len(fcols),
                        "train_seconds": 0.0,
                        "brier_skill": brier_skill,
                        "note": note,
                        **m,
                    }
                )
                cal_df = cal_df.copy()
                cal_df["horizon"] = 1
                cal_df["tier_id"] = cal_id
                cal_bin_rows.append(cal_df)
                bs = bootstrap_single_with_brier(cpdf, True, N_BOOTSTRAP, SEED)
                final_boot_rows.append(
                    {
                        "horizon": 1,
                        "scope": "final_holdout_panel",
                        "tier_id": cal_id,
                        "tier_group": "5. Final Model (Calibrated)",
                        "feature_set": "top50",
                        "n_features": len(fcols),
                        "note": note,
                        **bs,
                    }
                )
                print(
                    f"[H=1 FINAL] {cal_id:26s} | {test_month} "
                    f"| AP={m['average_precision']:.4f} | Brier={m['brier']:.5f} "
                    f"| LogLoss={m['log_loss']:.5f} | CalRatio={m['calibration_ratio']:.3f}",
                    flush=True,
                )

        # Paired comparisons on Final Holdout
        pairs = [
            ("T0_random_prevalence", "T1a_rule_rf", "Rule RF vs Random"),
            ("T1a_rule_rf", "T1b_rule_eda_scorecard", "Rule EDA Scorecard vs Rule RF"),
            ("T1b_rule_eda_scorecard", "T2_logreg_simple10", "LogReg Simple-10 vs Rule EDA"),
            ("T2_logreg_simple10", "T3a_lgbm_default_raw203", "LGBM Default Raw-203 vs LogReg Simple-10"),
            ("T2_logreg_simple10", "T3b_xgb_default_raw203", "XGB Default Raw-203 vs LogReg Simple-10"),
            ("T3a_lgbm_default_raw203", "T4a_lgbm_default_top50", "FE Gain (LGBM: Top-50 vs Raw-203)"),
            ("T3b_xgb_default_raw203", "T4b_xgb_default_top50", "FE Gain (XGB: Top-50 vs Raw-203)"),
            ("T2_logreg_simple10", "T2b_logreg_top50", "FE Gain (LogReg: Top-50 vs Simple-10)"),
            ("T4a_lgbm_default_top50", "T5_final_catboost_top50", "Final CatBoost vs LGBM Default (Top-50)"),
            ("T4b_xgb_default_top50", "T5_final_catboost_top50", "Final CatBoost vs XGB Default (Top-50)"),
            ("T2_logreg_simple10", "T5_final_catboost_top50", "Final CatBoost vs LogReg Simple-10"),
        ]
        if horizon == 1:
            pairs.append(("T5_final_catboost_top50", "T5_final_cal_intercept", "Calibration Gain (Intercept Shift vs Raw)"))
        for a_id, b_id, label in pairs:
            pb = paired_bootstrap_with_brier(
                final_preds[(horizon, a_id)],
                final_preds[(horizon, b_id)],
                final_prob_flags[a_id],
                final_prob_flags[b_id],
                N_BOOTSTRAP,
                SEED,
            )
            paired_rows.append(
                {
                    "horizon": horizon,
                    "scope": "final_holdout_panel",
                    "label": label,
                    "model_a": a_id,
                    "model_b": b_id,
                    **pb,
                }
            )

    # ------------------------------------------------------------------
    # 3. 5-Fold Out-of-Time + Out-of-Client (OOT+OOC, 0% client overlap)
    # ------------------------------------------------------------------
    ooc_rows: list[dict[str, Any]] = []
    ooc_boot_rows: list[dict[str, Any]] = []
    ooc_vs_panel_rows: list[dict[str, Any]] = []

    tier_by_id = {s.tier_id: s for s in TIERS}

    for horizon, tr_months, test_month in ((1, H1_FINAL_TRAIN, H1_TEST_MONTH), (3, H3_FINAL_TRAIN, H3_TEST_MONTH)):
        # Baseline prevalence for skill scores on the same test month
        prev_row = next(r for r in final_rows if r["horizon"] == horizon and r["tier_id"] == "T0_random_prevalence")
        prev_ll = float(prev_row["log_loss"])
        prev_brier = float(prev_row["brier"])

        for tid in OOC_TIER_IDS:
            spec = tier_by_id[tid]
            fcols = feature_sets[spec.feature_set]
            fold_dfs: list[pd.DataFrame] = []
            total_sec = 0.0
            for k in range(5):
                pred_path = OUT_DIR / "predictions" / f"h{horizon}_ooc" / tid / f"fold_{k}_{test_month}.parquet"
                pdf_k, sec_k = fit_predict_cached(
                    spec,
                    enriched_path,
                    folds_path,
                    fcols,
                    tr_months,
                    test_month,
                    horizon,
                    pred_path,
                    train_fold_filter=f"f.fold != {k}",
                    eval_fold_filter=f"f.fold = {k}",
                )
                fold_dfs.append(pdf_k)
                total_sec += sec_k

            ooc_df = pd.concat(fold_dfs, ignore_index=True).sort_values("client_id").reset_index(drop=True)
            assert len(ooc_df) == 100224 and ooc_df["client_id"].nunique() == 100224
            ooc_df.to_parquet(
                OUT_DIR / "predictions" / f"h{horizon}_ooc" / f"{tid}_all5folds_{test_month}.parquet",
                index=False,
                compression="zstd",
            )

            m, _ = compute_metrics(
                y_true=ooc_df["y_true"].to_numpy(),
                scores=ooc_df["score"].to_numpy(),
                client_ids=ooc_df["client_id"].to_numpy(),
                budget_fractions=BUDGET_FRACTIONS,
                primary_budget=PRIMARY_BUDGET,
                seed=SEED,
                is_probabilistic=spec.is_probabilistic,
                baseline_logloss=prev_ll,
            )
            ooc_rows.append(
                {
                    "horizon": horizon,
                    "protocol": "oot_plus_ooc_5fold",
                    "eval_month": test_month,
                    "tier_id": tid,
                    "tier_group": spec.tier_group,
                    "feature_set": spec.feature_set,
                    "n_features": len(fcols),
                    "train_seconds_5folds": round(total_sec, 2),
                    "brier_skill": float(1.0 - m["brier"] / prev_brier) if m["brier"] is not None else None,
                    **m,
                }
            )
            bs = bootstrap_single_with_brier(ooc_df, spec.is_probabilistic, N_BOOTSTRAP, SEED)
            ooc_boot_rows.append(
                {
                    "horizon": horizon,
                    "scope": "final_holdout_oot_ooc",
                    "tier_id": tid,
                    "tier_group": spec.tier_group,
                    "feature_set": spec.feature_set,
                    "n_features": len(fcols),
                    **bs,
                }
            )
            # Paired comparison: OOT+OOC (0% client overlap) minus Panel OOT (100% client overlap)
            panel_df = final_preds[(horizon, tid)]
            diff = paired_bootstrap_with_brier(
                panel_df, ooc_df, spec.is_probabilistic, spec.is_probabilistic, N_BOOTSTRAP, SEED
            )
            ooc_vs_panel_rows.append(
                {
                    "horizon": horizon,
                    "eval_month": test_month,
                    "tier_id": tid,
                    "panel_ap": float(next(r["average_precision"] for r in final_rows if r["horizon"] == horizon and r["tier_id"] == tid)),
                    "ooc_ap": float(m["average_precision"]),
                    "panel_auc": float(next(r["roc_auc"] for r in final_rows if r["horizon"] == horizon and r["tier_id"] == tid)),
                    "ooc_auc": float(m["roc_auc"]),
                    "panel_lift1": float(next(r["lift_1pct"] for r in final_rows if r["horizon"] == horizon and r["tier_id"] == tid)),
                    "ooc_lift1": float(m["lift_1pct"]),
                    "panel_brier": float(next(r["brier"] for r in final_rows if r["horizon"] == horizon and r["tier_id"] == tid)),
                    "ooc_brier": float(m["brier"]),
                    **diff,
                }
            )
            print(
                f"[H={horizon} OOT+OOC] {tid:26s} | AP={m['average_precision']:.4f} (Δ={diff['delta_average_precision']:+.4f}, p={diff['delta_average_precision_p_value']:.3f}) "
                f"| AUC={m['roc_auc']:.4f} (Δ={diff['delta_roc_auc']:+.4f}) | Lift@1%={m['lift_1pct']:.2f}x ({total_sec:.1f}s)",
                flush=True,
            )

    pd.DataFrame(final_rows).to_csv(metrics_dir / "final_holdout_metrics.csv", index=False)
    pd.DataFrame(val_boot_rows + final_boot_rows + ooc_boot_rows).to_csv(
        metrics_dir / "bootstrap_summary.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(metrics_dir / "paired_bootstrap.csv", index=False)
    pd.concat(cal_bin_rows, ignore_index=True).to_csv(metrics_dir / "calibration_bins.csv", index=False)
    pd.DataFrame(ooc_rows).to_csv(metrics_dir / "ooc_holdout_metrics.csv", index=False)
    pd.DataFrame(ooc_vs_panel_rows).to_csv(metrics_dir / "ooc_vs_panel_comparison.csv", index=False)

    manifest = {
        "status": "ready",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - t_all, 1),
        "seed": SEED,
        "budget_fractions": list(BUDGET_FRACTIONS),
        "primary_budget": PRIMARY_BUDGET,
        "bootstrap_replicates": N_BOOTSTRAP,
        "burn_in_months": list(BURN_IN_MONTHS),
        "feature_counts": {k: len(v) for k, v in feature_sets.items()},
        "simple10_features": list(SIMPLE10_FEATURES),
        "rule_eda_features": list(RULE_EDA_FEATURES),
        "calibration_info": calibration_info,
        "versions": {
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nALL DONE in {manifest['elapsed_seconds']}s -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
