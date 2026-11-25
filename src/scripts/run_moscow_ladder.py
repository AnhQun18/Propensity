"""Run the Moscow-time (UTC+3) Baseline Ladder (Tiers 0..5 + Target-Lag Reference Branch)
with strict Out-of-Time + Disjoint Client Folds (fold=0..4, 0% client overlap).

Time split (Moscow UTC+3, strict event_time_msk < YYYY-MM-01 00:00:00):
  - Burn-in only : 2022-02, 2022-03 (ensures >=90d GEO coverage from 2022-04 & full 1/3/6m target-lag on Val/Test)
  - Train (early): 2022-04, 2022-05, 2022-06, 2022-07, 2022-08 (5 months)
  - Val (middle) : 2022-09, 2022-10 (2 months) -> scored by models trained on Train (fold != k)
  - Test (last 3): 2022-11, 2022-12, 2023-01 (3 months, 2,301 positives) -> scored by models trained on 04..10 (fold != k)

Client split invariant (enforced on EVERY fold k in 0..4):
  - Train uses strictly fold != k (~80,179 clients)
  - Eval (Val / Test) uses strictly fold == k (~20,045 clients)
  - Combining all 5 folds gives 100% coverage (100,224 clients) with STRICTLY 0% client overlap!
  - DIALOG 768d embedding PCA is fitted STRICTLY on Train (fold != k, train_months) with valid embeddings;
    clients without prior dialog keep NaN (preserving clean separation between activity metadata and semantic content).

Ladder:
  L0_base_rate            : Tier 0 - Train base rate + deterministic SHA-256 tie-break
  L1_rule_target_lag      : Tier 1 - Pure target-lag score (1/3/6m prior purchases, uses target)
  L2_rule_trx_rfm         : Tier 2 - Pure TRX rule (30d/90d recency, count, amount; no target)
  L3_logreg_rfm           : Tier 3 - L2 Logistic Regression on small pure-TRX RFM (log-transformed amount)
  L4_lgbm_trx / L4_cb_trx : Tier 4 - Default LightGBM & Default CatBoost on TRX-only
  L5a_*_geo_meta          : Tier 5a - TRX + GEO metadata (activity only)
  L5b_*_geo_content       : Tier 5b - L5a + GEO content (geohash 4/5/6 & concentration)
  L5c_*_dialog_meta       : Tier 5c - L5b + DIALOG metadata (activity only) + cold_start
  L5d_*_dialog_emb        : Tier 5d - L5c + DIALOG embedding (mean-pool 768d -> train-only PCA + norms)
  Ref_*_plus_target_lag   : Reference branch (NOT main model) - L5d + target-lag (1/3/6/all)
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
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from propensity_model.metrics import (
    calibration_bins_table,
    compute_metrics,
    deterministic_tie_hash,
)
from propensity_model.models import PrevalenceBaseline, ProbabilityCalibrator

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "outputs" / "baseline_moscow_v1"
PANEL_PATH = DATA_DIR / "panel_moscow.parquet"
EMB_PATH = DATA_DIR / "dialog_emb_moscow.npz"
METRICS_DIR = DATA_DIR / "metrics"
PRED_DIR = DATA_DIR / "predictions"

SEED = 42
BUDGET_FRACTIONS = (0.005, 0.01, 0.05)  # Campaign budgets (NO 10% milestone)
PRIMARY_BUDGET = 0.01
N_BOOTSTRAP = 1000
PCA_ALL_DIM = 16
PCA_90D_DIM = 8

TRAIN_MONTHS_VAL = ("2022-04", "2022-05", "2022-06", "2022-07", "2022-08")
VAL_MONTHS = ("2022-09", "2022-10")
TRAIN_MONTHS_TEST = ("2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10")
TEST_MONTHS = ("2022-11", "2022-12", "2023-01")

# ---------------------------------------------------------------------------
# Strict, cleanly separated feature groups
# ---------------------------------------------------------------------------
RFM_SMALL_FEATURES: tuple[str, ...] = (
    "trx_30d_days_since_last_event",
    "trx_90d_days_since_last_event",
    "trx_30d_log1p_event_count",
    "trx_90d_log1p_event_count",
    "trx_30d_log1p_main_amount_abs_sum",
    "trx_90d_log1p_main_amount_abs_sum",
    "trx_all_days_since_first_event",
    "trx_all_has_event",
)

RULE_TARGET_LAG_COLS: tuple[str, ...] = (
    "prior_1m_bought_any",
    "prior_3m_buy_months",
    "prior_6m_buy_months",
)

RULE_TRX_RFM_COLS: tuple[str, ...] = (
    "trx_30d_event_count",
    "trx_90d_event_count",
    "trx_30d_log1p_main_amount_abs_sum",
    "trx_90d_log1p_main_amount_abs_sum",
    "trx_30d_days_since_last_event",
    "trx_90d_days_since_last_event",
)

TARGET_LAG_FEATURES: tuple[str, ...] = (
    "prior_1m_bought_any",
    "prior_3m_buy_months",
    "prior_3m_total_products",
    "prior_6m_buy_months",
    "prior_6m_total_products",
    "prior_all_bought_any",
    "prior_all_buy_months",
    "prior_months_since_last_buy",
    "prior_all_bought_p1",
    "prior_all_bought_p2",
    "prior_all_bought_p3",
    "prior_all_bought_p4",
    "prior_all_distinct_products",
)

FORBIDDEN_COLS: frozenset[str] = frozenset({
    "client_id", "fold", "month", "cutoff_msk",
    "y_any", "target_1", "target_2", "target_3", "target_4",
    "prior_months_observed",
})


def build_feature_groups(panel_cols: list[str]) -> dict[str, tuple[str, ...]]:
    trx = tuple(c for c in panel_cols if c.startswith("trx_"))
    geo_meta = tuple(
        c for c in panel_cols
        if c.startswith("geo_") and ("geohash" not in c)
    )
    geo_content = tuple(
        c for c in panel_cols
        if c.startswith("geo_") and ("geohash" in c)
    )
    dialog_meta = tuple(c for c in panel_cols if c.startswith("dialog_"))

    groups = {
        "none": (),
        "rule_target_lag": RULE_TARGET_LAG_COLS,
        "rule_trx_rfm": RULE_TRX_RFM_COLS,
        "rfm_small": RFM_SMALL_FEATURES,
        "trx": trx,
        "trx_geo_meta": trx + geo_meta,
        "trx_geo_full": trx + geo_meta + geo_content,
        "trx_geo_dialog_meta": trx + geo_meta + geo_content + dialog_meta + ("cold_start",),
        # Embedding PCA columns are appended dynamically per fold (train-only fit)
        "trx_geo_dialog_emb": trx + geo_meta + geo_content + dialog_meta + ("cold_start",),
        "ref_plus_target_lag": trx + geo_meta + geo_content + dialog_meta + ("cold_start",) + TARGET_LAG_FEATURES,
    }
    for name, cols in groups.items():
        bad = FORBIDDEN_COLS.intersection(cols)
        if bad:
            raise ValueError(f"Forbidden leakage column in {name}: {bad}")
        if len(cols) != len(set(cols)):
            raise ValueError(f"Duplicate column in {name}")
    # Verify NO target-lag leaks into main behavioral tiers L2..L5d
    for main_name in ("rule_trx_rfm", "rfm_small", "trx", "trx_geo_meta", "trx_geo_full", "trx_geo_dialog_meta", "trx_geo_dialog_emb"):
        assert not any(c.startswith("prior_") for c in groups[main_name]), f"Target-lag leaked into {main_name}"
    return groups


# ---------------------------------------------------------------------------
# Dialog Embedding Store (Train-only PCA fit per fold -> ZERO leakage!)
# ---------------------------------------------------------------------------
class DialogEmbeddingIndex:
    """Fast lookup of Moscow-time mean-pool 768d embeddings keyed by (month, client_id)."""

    def __init__(self, npz_path: Path):
        z = np.load(npz_path, allow_pickle=False)
        self.all_emb: np.ndarray = z["all_emb"].astype(np.float32)
        self.w90_emb: np.ndarray = z["w90_emb"].astype(np.float32)
        self.all_map: dict[tuple[str, str], int] = {
            (str(m), str(c)): i for i, (m, c) in enumerate(zip(z["all_months"], z["all_clients"]))
        }
        self.w90_map: dict[tuple[str, str], int] = {
            (str(m), str(c)): i for i, (m, c) in enumerate(zip(z["w90_months"], z["w90_clients"]))
        }
        self.dim = int(self.all_emb.shape[1])

    def lookup(self, months: np.ndarray, client_ids: np.ndarray, which: str) -> tuple[np.ndarray, np.ndarray]:
        mp = self.all_map if which == "all" else self.w90_map
        mat = self.all_emb if which == "all" else self.w90_emb
        idxs = np.array([mp.get((str(m), str(c)), -1) for m, c in zip(months, client_ids)], dtype=np.int32)
        valid = idxs >= 0
        return valid, mat[idxs[valid]] if np.any(valid) else np.empty((0, self.dim), dtype=np.float32)

    def fit_transform_fold(
        self,
        tr_months: np.ndarray,
        tr_cids: np.ndarray,
        ev_months: np.ndarray,
        ev_cids: np.ndarray,
        seed: int = SEED,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Fit StandardScaler+PCA strictly on Train valid embeddings; leave missing rows as NaN."""
        n_tr, n_ev = len(tr_cids), len(ev_cids)
        n_feat = PCA_ALL_DIM + PCA_90D_DIM + 3
        F_tr = np.full((n_tr, n_feat), np.nan, dtype=np.float32)
        F_ev = np.full((n_ev, n_feat), np.nan, dtype=np.float32)

        # 1. Lookup all-history & 90d embeddings, compute L2 norms & cosine topic-shift FIRST (on raw vectors)
        tr_va, tr_ea = self.lookup(tr_months, tr_cids, "all")
        ev_va, ev_ea = self.lookup(ev_months, ev_cids, "all")
        tr_v9, tr_e9 = self.lookup(tr_months, tr_cids, "90d")
        ev_v9, ev_e9 = self.lookup(ev_months, ev_cids, "90d")

        tr_norm_a = np.linalg.norm(tr_ea, axis=1).astype(np.float32)
        ev_norm_a = np.linalg.norm(ev_ea, axis=1).astype(np.float32) if len(ev_ea) else np.empty(0, dtype=np.float32)
        F_tr[tr_va, PCA_ALL_DIM + PCA_90D_DIM] = tr_norm_a
        F_ev[ev_va, PCA_ALL_DIM + PCA_90D_DIM] = ev_norm_a

        tr_norm_9 = np.linalg.norm(tr_e9, axis=1).astype(np.float32)
        ev_norm_9 = np.linalg.norm(ev_e9, axis=1).astype(np.float32) if len(ev_e9) else np.empty(0, dtype=np.float32)
        F_tr[tr_v9, PCA_ALL_DIM + PCA_90D_DIM + 1] = tr_norm_9
        F_ev[ev_v9, PCA_ALL_DIM + PCA_90D_DIM + 1] = ev_norm_9

        # Cosine similarity between 90d mean-pool and all-history mean-pool (measures topic shift!)
        for v_all, e_all, v_90, e_90, F in (
            (tr_va, tr_ea, tr_v9, tr_e9, F_tr),
            (ev_va, ev_ea, ev_v9, ev_e9, F_ev),
        ):
            both = v_all & v_90
            if np.any(both):
                pos_all = np.cumsum(v_all)[both] - 1
                pos_90 = np.cumsum(v_90)[both] - 1
                va_b = e_all[pos_all]
                v9_b = e_90[pos_90]
                dot = np.sum(va_b * v9_b, axis=1)
                denom = np.maximum(np.linalg.norm(va_b, axis=1) * np.linalg.norm(v9_b, axis=1), 1e-8)
                F[both, PCA_ALL_DIM + PCA_90D_DIM + 2] = (dot / denom).astype(np.float32)

        # 2. All-history dialog mean-pool -> in-place StandardScaler + PCA(16)
        sc_all = StandardScaler(copy=False)
        pca_all = PCA(n_components=PCA_ALL_DIM, svd_solver="randomized", random_state=seed)
        tr_za = pca_all.fit_transform(sc_all.fit_transform(tr_ea)).astype(np.float32)
        ev_za = pca_all.transform(sc_all.transform(ev_ea)).astype(np.float32) if len(ev_ea) else np.empty((0, PCA_ALL_DIM), dtype=np.float32)
        F_tr[tr_va, :PCA_ALL_DIM] = tr_za
        F_ev[ev_va, :PCA_ALL_DIM] = ev_za
        var_all = float(np.sum(pca_all.explained_variance_ratio_))
        del tr_ea, ev_ea, tr_za, ev_za, sc_all, pca_all
        gc.collect()

        # 3. 90d dialog mean-pool -> in-place StandardScaler + PCA(8)
        sc_90 = StandardScaler(copy=False)
        pca_90 = PCA(n_components=PCA_90D_DIM, svd_solver="randomized", random_state=seed)
        tr_z9 = pca_90.fit_transform(sc_90.fit_transform(tr_e9)).astype(np.float32)
        ev_z9 = pca_90.transform(sc_90.transform(ev_e9)).astype(np.float32) if len(ev_e9) else np.empty((0, PCA_90D_DIM), dtype=np.float32)
        F_tr[tr_v9, PCA_ALL_DIM:PCA_ALL_DIM + PCA_90D_DIM] = tr_z9
        F_ev[ev_v9, PCA_ALL_DIM:PCA_ALL_DIM + PCA_90D_DIM] = ev_z9
        var_90 = float(np.sum(pca_90.explained_variance_ratio_))
        del tr_e9, ev_e9, tr_z9, ev_z9, sc_90, pca_90
        gc.collect()

        info = {
            "pca_all_var_explained": var_all,
            "pca_90d_var_explained": var_90,
            "train_all_cov": float(np.mean(tr_va)),
            "eval_all_cov": float(np.mean(ev_va)),
        }
        return F_tr, F_ev, info


# ---------------------------------------------------------------------------
# Baseline Models (Tiers 0..5 + Target-Lag Reference)
# ---------------------------------------------------------------------------
class RuleTargetLagBaseline:
    """Tier 1: Rankable business rule using ONLY prior purchase counts (1m / 3m / 6m)."""
    is_probabilistic = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        p1 = np.nan_to_num(X[:, 0].astype(np.float64), nan=0.0)
        p3 = np.nan_to_num(X[:, 1].astype(np.float64), nan=0.0)
        p6 = np.nan_to_num(X[:, 2].astype(np.float64), nan=0.0)
        return 3.0 * p1 + 2.0 * p3 + 1.0 * p6


class RuleTrxRfmBaseline:
    """Tier 2: Pure TRX business heuristic (30d/90d recency, count, amount; ZERO target info)."""
    is_probabilistic = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        # Columns: cnt30, cnt90, amt30, amt90, rec30, rec90
        n = len(X)
        score = np.zeros(n, dtype=np.float64)
        for j in (0, 1, 2, 3):  # higher is better
            v = np.nan_to_num(X[:, j].astype(np.float64), nan=0.0)
            score += np.where(v > 0.0, rankdata(v, method="average") / n, 0.0)
        for j in (4, 5):  # recency: lower (closer to 0) is better
            v = X[:, j].astype(np.float64)
            valid = np.isfinite(v) & (v >= 0.0)
            inv = np.where(valid, -v, -1e9)
            score += np.where(valid, rankdata(inv, method="average") / n, 0.0)
        return score / 6.0


class SmallRfmLogRegBaseline:
    """Tier 3: L2 Logistic Regression on small pure-TRX RFM (train-only imputer + scaler)."""
    is_probabilistic = True

    def __init__(self, seed: int = SEED):
        self.seed = seed
        self.pipe_: Pipeline | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.pipe_ = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=self.seed)),
        ])
        self.pipe_.fit(X, y)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        assert self.pipe_ is not None
        return self.pipe_.predict_proba(X)[:, 1].astype(np.float64)


class LgbmDefaultModel:
    """Default LightGBM (natural distribution, keeps semantic NaNs)."""
    is_probabilistic = True

    def __init__(self, seed: int = SEED):
        self.seed = seed
        self.model_: lgb.LGBMClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model_ = lgb.LGBMClassifier(random_state=self.seed, n_jobs=4, verbose=-1)
        self.model_.fit(X, y)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.predict_proba(X)[:, 1].astype(np.float64)


class CatBoostDefaultModel:
    """CatBoost with default tree regularization (depth=6, l2=3, auto lr, keeps semantic NaNs)."""
    is_probabilistic = True

    def __init__(self, seed: int = SEED, iterations: int = 500):
        self.seed = seed
        self.iterations = iterations
        self.model_: CatBoostClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            random_seed=self.seed,
            thread_count=4,
            border_count=128,
            used_ram_limit="1.5gb",
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
        )
        self.model_.fit(X, y)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.predict_proba(X)[:, 1].astype(np.float64)


@dataclass(frozen=True)
class ExpSpec:
    exp_id: str
    tier: str
    family: str
    feature_group: str
    uses_emb: bool
    uses_target_lag: bool
    is_probabilistic: bool
    note: str


EXPERIMENTS: tuple[ExpSpec, ...] = (
    # Tier 0-3 (Non-ML & Simple Linear)
    ExpSpec("L0_base_rate", "Bậc 0: Base rate", "prevalence", "none", False, False, True, "Mốc sàn: p_train + SHA-256 tie-break"),
    ExpSpec("L1_rule_target_lag", "Bậc 1: Rule Target-Lag", "rule_target_lag", "rule_target_lag", False, True, False, "Mốc business CÓ dùng target (số lần mua 1/3/6m)"),
    ExpSpec("L2_rule_trx_rfm", "Bậc 2: Rule thuần TRX", "rule_trx_rfm", "rule_trx_rfm", False, False, False, "Mốc business KHÔNG dùng target (TRX 30/90d R,F,M)"),
    ExpSpec("L3_logreg_rfm", "Bậc 3: LogReg RFM nhỏ", "logreg_rfm", "rfm_small", False, False, True, "Mốc tuyến tính: 8 biến RFM thuần TRX (log amount)"),

    # Tier 4-5 LightGBM Default (Pure Behavioral Source Ablation)
    ExpSpec("L4_lgbm_trx", "Bậc 4: ML thuần TRX", "lgbm", "trx", False, False, True, "LightGBM mặc định chỉ dùng TRX (63 biến)"),
    ExpSpec("L5a_lgbm_geo_meta", "Bậc 5a: + GEO metadata", "lgbm", "trx_geo_meta", False, False, True, "LGBM: TRX + GEO metadata hoạt động (+18)"),
    ExpSpec("L5b_lgbm_geo_full", "Bậc 5b: + GEO content", "lgbm", "trx_geo_full", False, False, True, "LGBM: + nội dung geohash_4/5/6 (+12)"),
    ExpSpec("L5c_lgbm_dialog_meta", "Bậc 5c: + DIALOG meta", "lgbm", "trx_geo_dialog_meta", False, False, True, "LGBM: + DIALOG metadata hoạt động (+19)"),
    ExpSpec("L5d_lgbm_dialog_emb", "Bậc 5d: + DIALOG emb", "lgbm", "trx_geo_dialog_emb", True, False, True, "LGBM: + DIALOG embedding mean-pool PCA (+27)"),
    ExpSpec("Ref_lgbm_target_lag", "Nhánh phụ: + Target-Lag", "lgbm", "ref_plus_target_lag", True, True, True, "Tham chiếu LGBM: L5d + lịch sử mua 1/3/6/all (+13)"),

    # Tier 4-5 CatBoost Default (Pure Behavioral Source Ablation)
    ExpSpec("L4_cb_trx", "Bậc 4: ML thuần TRX", "catboost", "trx", False, False, True, "CatBoost mặc định chỉ dùng TRX (63 biến)"),
    ExpSpec("L5a_cb_geo_meta", "Bậc 5a: + GEO metadata", "catboost", "trx_geo_meta", False, False, True, "CatBoost: TRX + GEO metadata hoạt động (+18)"),
    ExpSpec("L5b_cb_geo_full", "Bậc 5b: + GEO content", "catboost", "trx_geo_full", False, False, True, "CatBoost: + nội dung geohash_4/5/6 (+12)"),
    ExpSpec("L5c_cb_dialog_meta", "Bậc 5c: + DIALOG meta", "catboost", "trx_geo_dialog_meta", False, False, True, "CatBoost: + DIALOG metadata hoạt động (+19)"),
    ExpSpec("L5d_cb_dialog_emb", "Bậc 5d: + DIALOG emb", "catboost", "trx_geo_dialog_emb", True, False, True, "CatBoost: + DIALOG embedding mean-pool PCA (+27)"),
    ExpSpec("Ref_cb_target_lag", "Nhánh phụ: + Target-Lag", "catboost", "ref_plus_target_lag", True, True, True, "Tham chiếu CatBoost: L5d + lịch sử mua 1/3/6/all (+13)"),
)


def make_model(family: str):
    if family == "prevalence":
        return PrevalenceBaseline()
    if family == "rule_target_lag":
        return RuleTargetLagBaseline()
    if family == "rule_trx_rfm":
        return RuleTrxRfmBaseline()
    if family == "logreg_rfm":
        return SmallRfmLogRegBaseline(seed=SEED)
    if family == "lgbm":
        return LgbmDefaultModel(seed=SEED)
    if family == "catboost":
        return CatBoostDefaultModel(seed=SEED, iterations=500)
    raise ValueError(f"Unknown family: {family}")


# ---------------------------------------------------------------------------
# Client-Clustered Bootstrap (with Brier Score & Paired p-values)
# ---------------------------------------------------------------------------
BOOT_METRICS = ("recall_1pct", "precision_1pct", "lift_1pct", "average_precision", "roc_auc", "log_loss", "brier")


def _prep_sorted(y: np.ndarray, scores: np.ndarray, codes: np.ndarray, hashes: np.ndarray, is_prob: bool) -> dict[str, Any]:
    order = np.lexsort((hashes, -scores))
    y_s = y[order].astype(np.float64)
    s_s = scores[order].astype(np.float64)
    c_s = codes[order]
    diff = np.empty(len(s_s), dtype=bool)
    diff[-1] = True
    if len(s_s) > 1:
        diff[:-1] = s_s[:-1] != s_s[1:]
    row_ll = None
    row_bs = None
    if is_prob:
        p = np.clip(s_s, 1e-15, 1.0 - 1e-15)
        row_ll = -(y_s * np.log(p) + (1.0 - y_s) * np.log(1.0 - p))
        row_bs = (s_s - y_s) ** 2
    return {"y_s": y_s, "c_s": c_s, "ge": np.flatnonzero(diff), "row_ll": row_ll, "row_bs": row_bs, "is_prob": is_prob}


def _w_metrics(prep: dict[str, Any], cw: np.ndarray) -> dict[str, float]:
    rw = cw[prep["c_s"]].astype(np.float64)
    tot_n = float(rw.sum())
    if tot_n <= 0.0:
        return {k: float("nan") for k in BOOT_METRICS}
    y_s = prep["y_s"]
    pos_w = rw * y_s
    tot_pos = float(pos_w.sum())
    tot_neg = tot_n - tot_pos
    cum_w = np.cumsum(rw)
    cum_pos = np.cumsum(pos_w)

    k_t = max(1, int(math.ceil(PRIMARY_BUDGET * tot_n)))
    idx = int(np.searchsorted(cum_w, k_t, side="left"))
    if idx >= len(cum_w):
        hits = tot_pos
    else:
        pw = float(cum_w[idx - 1]) if idx > 0 else 0.0
        pp = float(cum_pos[idx - 1]) if idx > 0 else 0.0
        hits = pp + (k_t - pw) * float(y_s[idx])

    prec = hits / k_t
    rec = (hits / tot_pos) if tot_pos > 0 else float("nan")
    prev = tot_pos / tot_n
    lift = (prec / prev) if prev > 0 else float("nan")

    if tot_pos > 0 and tot_neg > 0:
        ge = prep["ge"]
        tp = cum_pos[ge]
        tot = cum_w[ge]
        fp = tot - tp
        dtp = np.empty_like(tp)
        dtp[0] = tp[0]
        if len(tp) > 1:
            dtp[1:] = tp[1:] - tp[:-1]
        nz = tot > 0
        pt = np.zeros_like(tp)
        pt[nz] = tp[nz] / tot[nz]
        ap = float(np.sum((dtp / tot_pos) * pt))
        auc = float(np.trapezoid(np.concatenate(([0.0], tp / tot_pos)), np.concatenate(([0.0], fp / tot_neg))))
    else:
        ap, auc = float("nan"), float("nan")

    ll = float(np.dot(rw, prep["row_ll"]) / tot_n) if prep["is_prob"] else float("nan")
    bs = float(np.dot(rw, prep["row_bs"]) / tot_n) if prep["is_prob"] else float("nan")
    return {"recall_1pct": rec, "precision_1pct": prec, "lift_1pct": lift, "average_precision": ap, "roc_auc": auc, "log_loss": ll, "brier": bs}


def _ci(vals: np.ndarray) -> tuple[float | None, float | None, float | None]:
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return None, None, None
    lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    p = min(1.0, 2.0 * min(float(np.mean(v <= 0.0)), float(np.mean(v >= 0.0))))
    return lo, hi, p


def bootstrap_single(df: pd.DataFrame, is_prob: bool) -> dict[str, Any]:
    codes, uniq = pd.factorize(df["client_id"].to_numpy(), sort=True)
    nc = len(uniq)
    hashes = deterministic_tie_hash(df["client_id"].to_numpy(), seed=SEED)
    prep = _prep_sorted(df["y_true"].to_numpy(dtype=np.int8), df["score"].to_numpy(dtype=np.float64), codes, hashes, is_prob)
    pt = _w_metrics(prep, np.ones(nc, dtype=np.int64))
    samp = {k: np.empty(N_BOOTSTRAP, dtype=np.float64) for k in BOOT_METRICS}
    rng = np.random.default_rng(SEED)
    for b in range(N_BOOTSTRAP):
        cw = np.bincount(rng.integers(0, nc, size=nc), minlength=nc)
        res = _w_metrics(prep, cw)
        for k in BOOT_METRICS:
            samp[k][b] = res[k]
    out: dict[str, Any] = {"n_clients": nc, "n_rows": len(df), "replicates": N_BOOTSTRAP}
    for k in BOOT_METRICS:
        out[k] = float(pt[k]) if np.isfinite(pt[k]) else None
        lo, hi, _ = _ci(samp[k])
        out[f"{k}_ci_low"] = lo
        out[f"{k}_ci_high"] = hi
    return out


def bootstrap_paired(df_a: pd.DataFrame, df_b: pd.DataFrame, prob_a: bool, prob_b: bool) -> dict[str, Any]:
    m = df_a[["client_id", "month", "y_true", "score"]].merge(
        df_b[["client_id", "month", "score"]],
        on=["client_id", "month"], suffixes=("_a", "_b"), how="inner", validate="one_to_one"
    )
    assert len(m) == len(df_a) == len(df_b)
    codes, uniq = pd.factorize(m["client_id"].to_numpy(), sort=True)
    nc = len(uniq)
    hashes = deterministic_tie_hash(m["client_id"].to_numpy(), seed=SEED)
    y = m["y_true"].to_numpy(dtype=np.int8)
    pa = _prep_sorted(y, m["score_a"].to_numpy(dtype=np.float64), codes, hashes, prob_a)
    pb = _prep_sorted(y, m["score_b"].to_numpy(dtype=np.float64), codes, hashes, prob_b)
    ones = np.ones(nc, dtype=np.int64)
    pta, ptb = _w_metrics(pa, ones), _w_metrics(pb, ones)
    deltas = {k: np.empty(N_BOOTSTRAP, dtype=np.float64) for k in BOOT_METRICS}
    rng = np.random.default_rng(SEED)
    for b in range(N_BOOTSTRAP):
        cw = np.bincount(rng.integers(0, nc, size=nc), minlength=nc)
        ra, rb = _w_metrics(pa, cw), _w_metrics(pb, cw)
        for k in BOOT_METRICS:
            deltas[k][b] = rb[k] - ra[k]
    out: dict[str, Any] = {"n_clients": nc, "n_rows": len(m), "replicates": N_BOOTSTRAP}
    for k in BOOT_METRICS:
        d = ptb[k] - pta[k]
        out[f"delta_{k}"] = float(d) if np.isfinite(d) else None
        lo, hi, p = _ci(deltas[k])
        out[f"delta_{k}_ci_low"] = lo
        out[f"delta_{k}_ci_high"] = hi
        out[f"delta_{k}_p_value"] = p
    return out


def precompute_stage_pca(
    stage_name: str,
    panel_df: pd.DataFrame,
    emb_index: DialogEmbeddingIndex,
    train_months: tuple[str, ...],
    eval_months: tuple[str, ...],
    pca_dir: Path,
) -> list[dict[str, Any]]:
    """Pre-compute per-fold train-only dialog embedding PCA and store compact float32 slices on disk.

    This allows freeing the ~1.8 GB DialogEmbeddingIndex from RAM before model training starts.
    """
    assert max(train_months) < min(eval_months), "Temporal leakage violation!"
    stage_pca_dir = pca_dir / stage_name
    stage_pca_dir.mkdir(parents=True, exist_ok=True)

    df_tr_all = panel_df.loc[panel_df["month"].isin(train_months), ["client_id", "fold", "month"]].reset_index(drop=True)
    df_ev_all = panel_df.loc[panel_df["month"].isin(eval_months), ["client_id", "fold", "month"]].reset_index(drop=True)

    pca_logs: list[dict[str, Any]] = []
    for k in range(5):
        tr_k = df_tr_all["fold"].to_numpy() != k
        ev_k = df_ev_all["fold"].to_numpy() == k
        tr_cids = df_tr_all.loc[tr_k, "client_id"].to_numpy()
        ev_cids = df_ev_all.loc[ev_k, "client_id"].to_numpy()
        assert len(set(tr_cids).intersection(set(ev_cids))) == 0, f"Client overlap in fold {k}!"
        F_tr, F_ev, pinfo = emb_index.fit_transform_fold(
            df_tr_all.loc[tr_k, "month"].to_numpy(), tr_cids,
            df_ev_all.loc[ev_k, "month"].to_numpy(), ev_cids,
            seed=SEED + k,
        )
        np.save(stage_pca_dir / f"f{k}_tr.npy", F_tr)
        np.save(stage_pca_dir / f"f{k}_ev.npy", F_ev)
        del F_tr, F_ev
        gc.collect()
        pca_logs.append({"stage": stage_name, "fold": k, **pinfo})
        print(
            f"  [PCA {stage_name.upper():4s} fold={k}] var_all={pinfo['pca_all_var_explained']:.3f}, "
            f"var_90d={pinfo['pca_90d_var_explained']:.3f}",
            flush=True,
        )
    return pca_logs


def run_stage(
    stage_name: str,
    panel_df: pd.DataFrame,
    feature_groups: dict[str, tuple[str, ...]],
    train_months: tuple[str, ...],
    eval_months: tuple[str, ...],
    pca_dir: Path,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Run all experiments under strict OOT + 5-fold Disjoint-Client protocol."""
    assert max(train_months) < min(eval_months), "Temporal leakage violation!"

    tr_mask_m = panel_df["month"].isin(train_months).to_numpy()
    ev_mask_m = panel_df["month"].isin(eval_months).to_numpy()
    df_tr_all = panel_df.loc[tr_mask_m].reset_index(drop=True)
    df_ev_all = panel_df.loc[ev_mask_m].reset_index(drop=True)
    stage_pca_dir = pca_dir / stage_name

    preds_by_exp: dict[str, pd.DataFrame] = {}
    timing_rows: list[dict[str, Any]] = []

    for spec in EXPERIMENTS:
        pred_file = PRED_DIR / stage_name / f"{spec.exp_id}.parquet"
        pred_file.parent.mkdir(parents=True, exist_ok=True)
        cols = list(feature_groups[spec.feature_group])
        n_feats = len(cols) + (PCA_ALL_DIM + PCA_90D_DIM + 3 if spec.uses_emb else 0)

        if pred_file.exists():
            pdf = pd.read_parquet(pred_file)
            preds_by_exp[spec.exp_id] = pdf
            timing_rows.append({"stage": stage_name, "exp_id": spec.exp_id, "n_features": n_feats, "seconds_5folds": 0.0})
            print(f"[{stage_name.upper():4s}] {spec.exp_id:24s} (cached)", flush=True)
            continue

        t0 = time.perf_counter()
        fold_dfs: list[pd.DataFrame] = []
        for k in range(5):
            tr_k = df_tr_all["fold"].to_numpy() != k
            ev_k = df_ev_all["fold"].to_numpy() == k

            if cols:
                X_tr = df_tr_all.loc[tr_k, cols].to_numpy(dtype=np.float32, copy=False)
                X_ev = df_ev_all.loc[ev_k, cols].to_numpy(dtype=np.float32, copy=False)
            else:
                X_tr = np.empty((int(tr_k.sum()), 0), dtype=np.float32)
                X_ev = np.empty((int(ev_k.sum()), 0), dtype=np.float32)

            if spec.uses_emb:
                F_tr = np.load(stage_pca_dir / f"f{k}_tr.npy")
                F_ev = np.load(stage_pca_dir / f"f{k}_ev.npy")
                X_tr = np.hstack([X_tr, F_tr])
                X_ev = np.hstack([X_ev, F_ev])
                del F_tr, F_ev

            y_tr = df_tr_all.loc[tr_k, "y_any"].to_numpy(dtype=np.int8)
            y_ev = df_ev_all.loc[ev_k, "y_any"].to_numpy(dtype=np.int8)

            model = make_model(spec.family)
            model.fit(X_tr, y_tr)
            scores = model.predict_scores(X_ev)

            ev_meta = df_ev_all.loc[ev_k, ["client_id", "fold", "month", "cold_start", "geo_all_has_event", "dialog_all_has_event", "trx_all_has_event"]].copy()
            ev_meta["y_true"] = y_ev
            ev_meta["score"] = scores.astype(np.float64)
            fold_dfs.append(ev_meta)

            del X_tr, X_ev, model
            gc.collect()

        pdf = pd.concat(fold_dfs, ignore_index=True).sort_values(["month", "client_id"]).reset_index(drop=True)
        assert len(pdf) == len(df_ev_all)
        pred_tmp = pred_file.with_suffix(".tmp.parquet")
        pdf.to_parquet(pred_tmp, index=False, compression="zstd")
        pred_tmp.replace(pred_file)
        preds_by_exp[spec.exp_id] = pdf
        sec = time.perf_counter() - t0
        timing_rows.append({"stage": stage_name, "exp_id": spec.exp_id, "n_features": n_feats, "seconds_5folds": round(sec, 2)})

        # Quick log per experiment (macro-monthly AP & AUC across eval months)
        m_aps, m_aucs, m_l1s = [], [], []
        for em in eval_months:
            sub = pdf[pdf["month"] == em]
            mm, _ = compute_metrics(sub["y_true"].to_numpy(), sub["score"].to_numpy(), sub["client_id"].to_numpy(),
                                    budget_fractions=BUDGET_FRACTIONS, primary_budget=PRIMARY_BUDGET,
                                    seed=SEED, is_probabilistic=spec.is_probabilistic)
            m_aps.append(mm["average_precision"])
            m_aucs.append(mm["roc_auc"])
            m_l1s.append(mm["lift_1pct"])
        print(
            f"[{stage_name.upper():4s}] {spec.exp_id:24s} ({n_feats:3d}f) "
            f"| Macro AP={np.mean(m_aps):.4f} | AUC={np.mean(m_aucs):.4f} | Lift@1%={np.mean(m_l1s):.2f}x ({sec:.1f}s)",
            flush=True,
        )

    del df_tr_all, df_ev_all
    gc.collect()
    return preds_by_exp, timing_rows


def main():
    t_all = time.perf_counter()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pca_dir = DATA_DIR / "_pca_tmp"

    print("Loading Moscow-time panel and dialog embeddings ...", flush=True)
    panel_df = pd.read_parquet(PANEL_PATH)
    feature_groups = build_feature_groups(list(panel_df.columns))
    emb_index = DialogEmbeddingIndex(EMB_PATH)
    print(f"Panel: {panel_df.shape} | Dialog emb pairs: all={len(emb_index.all_map):,}, 90d={len(emb_index.w90_map):,}", flush=True)

    # 0. Pre-compute train-only PCA embeddings for Val & Test, then release ~1.8 GB DialogEmbeddingIndex
    print("\n=== STAGE 0: PRE-COMPUTING TRAIN-ONLY DIALOG PCA (VAL & TEST) ===", flush=True)
    val_pca = precompute_stage_pca("val", panel_df, emb_index, TRAIN_MONTHS_VAL, VAL_MONTHS, pca_dir)
    test_pca = precompute_stage_pca("test", panel_df, emb_index, TRAIN_MONTHS_TEST, TEST_MONTHS, pca_dir)
    del emb_index
    gc.collect()

    # 1. Run Validation (Train 04..08 fold!=k -> Val 09..10 fold==k)
    print("\n=== STAGE 1: VALIDATION (Train 2022-04..08 -> Val 2022-09..10 | 5-Fold Disjoint Clients) ===", flush=True)
    val_preds, val_time = run_stage("val", panel_df, feature_groups, TRAIN_MONTHS_VAL, VAL_MONTHS, pca_dir)

    # 2. Run Locked Test (Train 04..10 fold!=k -> Test 2022-11, 2022-12, 2023-01 fold==k)
    print("\n=== STAGE 2: TEST 3 LAST MONTHS (Train 2022-04..10 -> Test 2022-11..2023-01 | 5-Fold Disjoint Clients) ===", flush=True)
    test_preds, test_time = run_stage("test", panel_df, feature_groups, TRAIN_MONTHS_TEST, TEST_MONTHS, pca_dir)

    # Release panel_df & clean up temporary PCA slices before bootstrap stage
    del panel_df
    gc.collect()
    import shutil
    shutil.rmtree(pca_dir, ignore_errors=True)

    # 3. Post-hoc Calibration on Val (09..10) -> applied to Test (11..01) for main L5d models
    cal_info: dict[str, Any] = {}
    cal_specs: list[tuple[str, str, str]] = [
        ("L5d_cb_dialog_emb", "L5d_cb_dialog_emb_cal", "CatBoost Bậc 5d + Intercept-Shift Calibration (fit trên Val 09–10)"),
        ("L5d_lgbm_dialog_emb", "L5d_lgbm_dialog_emb_cal", "LightGBM Bậc 5d + Intercept-Shift Calibration (fit trên Val 09–10)"),
    ]
    for base_id, cal_id, note in cal_specs:
        vdf = val_preds[base_id]
        tdf = test_preds[base_id].copy()
        cal = ProbabilityCalibrator("intercept_shift").fit(vdf["score"].to_numpy(), vdf["y_true"].to_numpy())
        tdf["score"] = cal.transform(tdf["score"].to_numpy())
        tdf.to_parquet(PRED_DIR / "test" / f"{cal_id}.parquet", index=False, compression="zstd")
        test_preds[cal_id] = tdf
        cal_info[cal_id] = {
            "base_exp": base_id,
            "slope": cal.slope_,
            "intercept": cal.intercept_,
            "val_obs_rate": float(vdf["y_true"].mean()),
            "val_raw_pred_mean": float(vdf["score"].mean()),
        }

    # 4. Compute full monthly metrics, pooled metrics, calibration bins & 1,000-rep bootstrap CIs
    print("\n=== STAGE 3: METRICS, BRIER, CALIBRATION & 1,000 CLIENT-CLUSTERED BOOTSTRAP ===", flush=True)
    monthly_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    cal_bin_dfs: list[pd.DataFrame] = []

    all_specs = list(EXPERIMENTS) + [
        ExpSpec("L5d_cb_dialog_emb_cal", "Bậc 5d (Calibrated)", "catboost", "trx_geo_dialog_emb", True, False, True,
                "CatBoost Bậc 5d + Intercept-Shift Calibration (fit trên Val 09–10)"),
        ExpSpec("L5d_lgbm_dialog_emb_cal", "Bậc 5d (Calibrated)", "lgbm", "trx_geo_dialog_emb", True, False, True,
                "LightGBM Bậc 5d + Intercept-Shift Calibration (fit trên Val 09–10)"),
    ]
    spec_map = {s.exp_id: s for s in all_specs}

    for stage_name, preds_dict, months in (("val", val_preds, VAL_MONTHS), ("test", test_preds, TEST_MONTHS)):
        # Baseline L0 logloss & brier per month and pooled
        base_ll_m: dict[str, float] = {}
        base_bs_m: dict[str, float] = {}
        l0_df = preds_dict["L0_base_rate"]
        for em in months:
            sub = l0_df[l0_df["month"] == em]
            mm, _ = compute_metrics(sub["y_true"].to_numpy(), sub["score"].to_numpy(), sub["client_id"].to_numpy(),
                                    budget_fractions=BUDGET_FRACTIONS, primary_budget=PRIMARY_BUDGET, seed=SEED, is_probabilistic=True)
            base_ll_m[em] = float(mm["log_loss"])
            base_bs_m[em] = float(mm["brier"])
        l0_pool_m, _ = compute_metrics(l0_df["y_true"].to_numpy(), l0_df["score"].to_numpy(), l0_df["client_id"].to_numpy(),
                                       budget_fractions=BUDGET_FRACTIONS, primary_budget=PRIMARY_BUDGET, seed=SEED, is_probabilistic=True)
        base_ll_pool = float(l0_pool_m["log_loss"])
        base_bs_pool = float(l0_pool_m["brier"])

        for exp_id, pdf in preds_dict.items():
            sp = spec_map[exp_id]
            n_f = len(feature_groups[sp.feature_group]) + (PCA_ALL_DIM + PCA_90D_DIM + 3 if sp.uses_emb else 0)
            for em in months:
                sub = pdf[pdf["month"] == em]
                mm, _ = compute_metrics(
                    sub["y_true"].to_numpy(), sub["score"].to_numpy(), sub["client_id"].to_numpy(),
                    budget_fractions=BUDGET_FRACTIONS, primary_budget=PRIMARY_BUDGET, seed=SEED,
                    is_probabilistic=sp.is_probabilistic, baseline_logloss=base_ll_m[em],
                )
                bss = float(1.0 - mm["brier"] / base_bs_m[em]) if (mm["brier"] is not None and base_bs_m[em] > 0) else None
                monthly_rows.append({
                    "stage": stage_name, "month": em, "exp_id": exp_id, "tier": sp.tier, "family": sp.family,
                    "uses_target_lag": sp.uses_target_lag, "n_features": n_f, "brier_skill": bss, "note": sp.note, **mm,
                })

            # Pooled metrics + 1,000 client-clustered bootstrap CI + calibration bins
            pm, c_df = compute_metrics(
                pdf["y_true"].to_numpy(), pdf["score"].to_numpy(), pdf["client_id"].to_numpy(),
                budget_fractions=BUDGET_FRACTIONS, primary_budget=PRIMARY_BUDGET, seed=SEED,
                is_probabilistic=sp.is_probabilistic, baseline_logloss=base_ll_pool,
            )
            bss_pool = float(1.0 - pm["brier"] / base_bs_pool) if (pm["brier"] is not None and base_bs_pool > 0) else None
            bs = bootstrap_single(pdf, sp.is_probabilistic)
            boot_rows.append({
                "stage": stage_name, "exp_id": exp_id, "tier": sp.tier, "family": sp.family,
                "uses_target_lag": sp.uses_target_lag, "n_features": n_f, "note": sp.note,
                "lift_0p5pct": pm["lift_0p5pct"], "lift_5pct": pm["lift_5pct"],
                "recall_5pct": pm["recall_5pct"], "hits_1pct": pm["hits_1pct"],
                "k_1pct": pm["k_1pct"], "positives": pm["positives"], "prevalence": pm["prevalence"],
                "brier_skill": bss_pool, "log_loss_skill": pm["log_loss_skill"],
                "calibration_ratio": pm["calibration_ratio"], "ece_quantile": pm["ece_quantile"],
                **bs,
            })
            if stage_name == "test" and len(c_df) > 0:
                c_df = c_df.copy()
                c_df["exp_id"] = exp_id
                cal_bin_dfs.append(c_df)

    # 5. Paired Client-Clustered Bootstrap on Locked Test (3 months, 2,301 positives, 0% client overlap)
    paired_defs = [
        ("L0_base_rate", "L1_rule_target_lag", "Bậc 1 (Rule Target-Lag 1/3/6m) vs Bậc 0 (Base rate)"),
        ("L0_base_rate", "L2_rule_trx_rfm", "Bậc 2 (Rule TRX 30/90d) vs Bậc 0 (Base rate)"),
        ("L2_rule_trx_rfm", "L3_logreg_rfm", "Bậc 3 (LogReg RFM) vs Bậc 2 (Rule TRX)"),
        ("L3_logreg_rfm", "L4_lgbm_trx", "Bậc 4 (LGBM TRX) vs Bậc 3 (LogReg RFM)"),
        ("L3_logreg_rfm", "L4_cb_trx", "Bậc 4 (CatBoost TRX) vs Bậc 3 (LogReg RFM)"),
        ("L4_lgbm_trx", "L4_cb_trx", "Bậc 4: CatBoost vs LightGBM (cùng TRX)"),
        # Tier 5 Source Ablation (CatBoost)
        ("L4_cb_trx", "L5a_cb_geo_meta", "Bậc 5a [CB]: + GEO metadata (hoạt động) vs TRX"),
        ("L5a_cb_geo_meta", "L5b_cb_geo_full", "Bậc 5b [CB]: + GEO content (geohash) vs + GEO meta"),
        ("L4_cb_trx", "L5b_cb_geo_full", "Bậc 5 [CB]: + GEO toàn phần vs TRX"),
        ("L5b_cb_geo_full", "L5c_cb_dialog_meta", "Bậc 5c [CB]: + DIALOG metadata (hoạt động) vs +GEO"),
        ("L5c_cb_dialog_meta", "L5d_cb_dialog_emb", "Bậc 5d [CB]: + DIALOG embedding (nội dung) vs +DIALOG meta"),
        ("L4_cb_trx", "L5d_cb_dialog_emb", "Bậc 5 [CB]: Đa nguồn đầy đủ (L5d) vs Thuần TRX (L4)"),
        # Tier 5 Source Ablation (LightGBM)
        ("L4_lgbm_trx", "L5a_lgbm_geo_meta", "Bậc 5a [LGBM]: + GEO metadata (hoạt động) vs TRX"),
        ("L5a_lgbm_geo_meta", "L5b_lgbm_geo_full", "Bậc 5b [LGBM]: + GEO content (geohash) vs + GEO meta"),
        ("L5b_lgbm_geo_full", "L5c_lgbm_dialog_meta", "Bậc 5c [LGBM]: + DIALOG metadata (hoạt động) vs +GEO"),
        ("L5c_lgbm_dialog_meta", "L5d_lgbm_dialog_emb", "Bậc 5d [LGBM]: + DIALOG embedding (nội dung) vs +DIALOG meta"),
        # Target-Lag Fair Comparisons & Reference Branch
        ("L1_rule_target_lag", "L5d_cb_dialog_emb", "Mô hình chính L5d (KHÔNG target) vs Bậc 1 (CÓ target)"),
        ("L5d_cb_dialog_emb", "Ref_cb_target_lag", "Nhánh phụ [CB]: + Target-Lag (1/3/6/all) vs L5d thuần hành vi"),
        ("L5d_lgbm_dialog_emb", "Ref_lgbm_target_lag", "Nhánh phụ [LGBM]: + Target-Lag (1/3/6/all) vs L5d thuần hành vi"),
        ("L1_rule_target_lag", "Ref_cb_target_lag", "So sánh công bằng (cùng có target): Ref CatBoost vs Rule Bậc 1"),
        # Calibration check
        ("L5d_cb_dialog_emb", "L5d_cb_dialog_emb_cal", "Calibration [CB L5d]: Intercept-Shift vs Raw"),
    ]
    paired_rows: list[dict[str, Any]] = []
    for a_id, b_id, label in paired_defs:
        pb = bootstrap_paired(
            test_preds[a_id], test_preds[b_id],
            spec_map[a_id].is_probabilistic, spec_map[b_id].is_probabilistic,
        )
        paired_rows.append({"stage": "test", "label": label, "model_a": a_id, "model_b": b_id, **pb})
        print(
            f"[PAIRED TEST] {label:58s} | ΔAP={pb['delta_average_precision']:+.4f} (p={pb['delta_average_precision_p_value']:.3f}) "
            f"| ΔAUC={pb['delta_roc_auc']:+.4f} (p={pb['delta_roc_auc_p_value']:.3f}) | ΔLift@1%={pb['delta_lift_1pct']:+.2f}x",
            flush=True,
        )

    # 6. Segment diagnostic: Activity vs Content lift on Test (has_dialog / has_geo slices)
    slice_rows: list[dict[str, Any]] = []
    for exp_id in ("L4_cb_trx", "L5a_cb_geo_meta", "L5b_cb_geo_full", "L5c_cb_dialog_meta", "L5d_cb_dialog_emb", "Ref_cb_target_lag"):
        pdf = test_preds[exp_id]
        for dim, col in (("has_dialog_all", "dialog_all_has_event"), ("has_geo_all", "geo_all_has_event"), ("cold_start", "cold_start")):
            for val in (0, 1):
                sub = pdf[pdf[col] == val]
                mm, _ = compute_metrics(sub["y_true"].to_numpy(), sub["score"].to_numpy(), sub["client_id"].to_numpy(),
                                        budget_fractions=BUDGET_FRACTIONS, primary_budget=PRIMARY_BUDGET, seed=SEED, is_probabilistic=True)
                slice_rows.append({"exp_id": exp_id, "slice_dim": dim, "slice_val": int(val), **mm})

    pd.DataFrame(monthly_rows).to_csv(METRICS_DIR / "monthly_metrics.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(METRICS_DIR / "bootstrap_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(METRICS_DIR / "paired_bootstrap_test.csv", index=False)
    pd.concat(cal_bin_dfs, ignore_index=True).to_csv(METRICS_DIR / "calibration_bins_test.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(METRICS_DIR / "slice_metrics_test.csv", index=False)
    pd.DataFrame(val_time + test_time).to_csv(METRICS_DIR / "timings.csv", index=False)
    pd.DataFrame(val_pca + test_pca).to_csv(METRICS_DIR / "pca_variance_log.csv", index=False)

    manifest = {
        "status": "ready",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - t_all, 1),
        "timezone": "Europe/Moscow (UTC+3)",
        "seed": SEED,
        "train_months_val": list(TRAIN_MONTHS_VAL),
        "val_months": list(VAL_MONTHS),
        "train_months_test": list(TRAIN_MONTHS_TEST),
        "test_months": list(TEST_MONTHS),
        "client_split_protocol": "5-fold disjoint client_id (fold != k train, fold == k eval -> 0% client overlap)",
        "budget_fractions": list(BUDGET_FRACTIONS),
        "primary_budget": PRIMARY_BUDGET,
        "bootstrap_replicates": N_BOOTSTRAP,
        "pca_dims": {"all": PCA_ALL_DIM, "90d": PCA_90D_DIM, "norms_cos": 3},
        "feature_counts": {k: len(v) for k, v in feature_groups.items()},
        "calibration_info": cal_info,
    }
    (DATA_DIR / "ladder_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nALL DONE in {manifest['elapsed_seconds']}s -> {DATA_DIR}", flush=True)


if __name__ == "__main__":
    main()
