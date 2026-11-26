"""Model implementations decoupled from the evaluation protocol."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BasePropensityModel(ABC):
    """Uniform model interface consumed by the protocol runner."""

    is_probabilistic: bool = True

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_es: np.ndarray | None = None,
        y_es: np.ndarray | None = None,
    ) -> "BasePropensityModel":
        """Fit model on training data, optionally using an internal temporal early-stopping split."""

    @abstractmethod
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return 1D float64 ranking/probability scores."""


class PrevalenceModel(BasePropensityModel):
    """L0: Monthly train base-rate floor (tie-broken deterministically at evaluation)."""

    is_probabilistic = True

    def __init__(self):
        self.base_rate_: float = 0.0

    def fit(self, X_train, y_train, X_es=None, y_es=None) -> "PrevalenceModel":
        self.base_rate_ = float(np.mean(y_train))
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.base_rate_, dtype=np.float64)


class RuleTargetLagModel(BasePropensityModel):
    """L1: Non-ML business rule on prior purchase counts (1m, 3m, 6m)."""

    is_probabilistic = False

    def fit(self, X_train, y_train, X_es=None, y_es=None) -> "RuleTargetLagModel":
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(X.astype(np.float64, copy=False), nan=0.0)
        return 3.0 * arr[:, 0] + 2.0 * arr[:, 1] + 1.0 * arr[:, 2]


class RuleTrxRfmModel(BasePropensityModel):
    """L2: Pure-TRX RFM percentile rank rule (no target labels used)."""

    is_probabilistic = False

    def fit(self, X_train, y_train, X_es=None, y_es=None) -> "RuleTrxRfmModel":
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        # Columns: trx_30d_event_count, trx_90d_event_count,
        #          trx_30d_log1p_main_amount_abs_sum, trx_90d_log1p_main_amount_abs_sum,
        #          trx_30d_days_since_last_event, trx_90d_days_since_last_event
        n = len(X)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        parts = []
        for j in (0, 1, 2, 3):
            v = np.nan_to_num(X[:, j].astype(np.float64), nan=0.0)
            parts.append(np.where(v > 0, rankdata(v, method="average") / n, 0.0))
        for j in (4, 5):
            v = X[:, j].astype(np.float64)
            valid = np.isfinite(v) & (v >= 0)
            inv = np.where(valid, -v, -1e9)
            parts.append(np.where(valid, rankdata(inv, method="average") / n, 0.0))
        return np.mean(parts, axis=0)


class LogRegModel(BasePropensityModel):
    """L2 Logistic Regression with fold-only median imputation + missingness indicators + scaling."""

    is_probabilistic = True

    def __init__(self, seed: int = 42, c: float = 1.0, max_iter: int = 300):
        self.seed = int(seed)
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.pipe_: Pipeline | None = None

    def fit(self, X_train, y_train, X_es=None, y_es=None) -> "LogRegModel":
        self.pipe_ = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=self.c, solver="lbfgs", max_iter=self.max_iter, random_state=self.seed)),
        ])
        self.pipe_.fit(X_train, y_train)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        assert self.pipe_ is not None
        return self.pipe_.predict_proba(X)[:, 1].astype(np.float64)


class CatBoostTunedModel(BasePropensityModel):
    """Fair CatBoost keeping semantic NaNs natively, with temporal early stopping + full-train refit."""

    is_probabilistic = True

    def __init__(
        self,
        seed: int = 42,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 10.0,
        iterations: int | None = None,
        max_iterations: int = 1500,
        early_stopping_rounds: int = 100,
        thread_count: int = -1,
        border_count: int = 128,
        used_ram_limit: str = "2gb",
        eval_metric: str = "Logloss",
    ):
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self.depth = int(depth)
        self.l2_leaf_reg = float(l2_leaf_reg)
        self.iterations = int(iterations) if iterations is not None else None
        self.max_iterations = int(max_iterations)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.thread_count = int(thread_count)
        self.border_count = int(border_count)
        self.used_ram_limit = str(used_ram_limit)
        self.eval_metric = str(eval_metric)
        self.model_: CatBoostClassifier | None = None
        self.best_iteration_: int | None = None

    def _make_cb(self, n_iters: int) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=int(n_iters),
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            border_count=self.border_count,
            random_seed=self.seed,
            thread_count=self.thread_count,
            used_ram_limit=self.used_ram_limit,
            loss_function="Logloss",
            eval_metric=self.eval_metric,
            verbose=False,
            allow_writing_files=False,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_es: np.ndarray | None = None,
        y_es: np.ndarray | None = None,
    ) -> "CatBoostTunedModel":
        if self.iterations is not None or X_es is None or y_es is None:
            n_iters = self.iterations if self.iterations is not None else self.max_iterations
            self.model_ = self._make_cb(n_iters)
            self.model_.fit(X_train, y_train)
            self.best_iteration_ = n_iters
            return self

        # Step 1: Temporal early stopping on internal holdout month (m-1, fold != k)
        es_model = self._make_cb(self.max_iterations)
        es_model.fit(
            X_train,
            y_train,
            eval_set=(X_es, y_es),
            early_stopping_rounds=self.early_stopping_rounds,
            use_best_model=True,
        )
        raw_best = es_model.get_best_iteration()
        best_trees = max(50, int(raw_best) + 1) if (raw_best is not None and raw_best >= 0) else self.max_iterations
        self.best_iteration_ = best_trees

        # Step 2: Refit on full train window (including m-1, still fold != k) with best_trees
        X_full = np.vstack([X_train, X_es])
        y_full = np.concatenate([y_train, y_es])
        self.model_ = self._make_cb(best_trees)
        self.model_.fit(X_full, y_full)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.predict_proba(X)[:, 1].astype(np.float64)


def build_model(family: str, seed: int = 42, model_params: dict[str, Any] | None = None) -> BasePropensityModel:
    """Instantiate a Baseline v2 model from its family identifier."""
    params = dict(model_params or {})
    if family == "prevalence":
        return PrevalenceModel()
    if family == "rule_target_lag":
        return RuleTargetLagModel()
    if family == "rule_trx_rfm":
        return RuleTrxRfmModel()
    if family == "logreg":
        return LogRegModel(seed=seed, c=params.get("c", 1.0), max_iter=params.get("max_iter", 300))
    if family == "catboost_tuned":
        return CatBoostTunedModel(
            seed=seed,
            learning_rate=params.get("learning_rate", 0.05),
            depth=params.get("depth", 6),
            l2_leaf_reg=params.get("l2_leaf_reg", 10.0),
            iterations=params.get("iterations"),
            max_iterations=params.get("max_iterations", 1500),
            early_stopping_rounds=params.get("early_stopping_rounds", 100),
            thread_count=params.get("thread_count", -1),
            border_count=params.get("border_count", 128),
            used_ram_limit=params.get("used_ram_limit", "2gb"),
            eval_metric=params.get("eval_metric", "Logloss"),
        )
    raise ValueError(f"Unsupported model family: {family}")


def blend_prediction_frames(
    df_logreg: pd.DataFrame,
    df_catboost: pd.DataFrame,
    method: str,
    weight_logreg: float = 0.5,
) -> pd.DataFrame:
    """Combine out-of-fold predictions of LogReg and CatBoost per month without leakage."""
    merged = df_logreg[["client_id", "fold", "month", "y_true", "score"]].merge(
        df_catboost[["client_id", "month", "score"]],
        on=["client_id", "month"],
        suffixes=("_lr", "_cb"),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(df_logreg) or len(merged) != len(df_catboost):
        raise ValueError("Blend alignment mismatch between LogReg and CatBoost predictions")

    w_lr = float(weight_logreg)
    w_cb = 1.0 - w_lr
    s_lr = merged["score_lr"].to_numpy(dtype=np.float64)
    s_cb = merged["score_cb"].to_numpy(dtype=np.float64)

    if method == "prob_linear":
        blended = w_lr * s_lr + w_cb * s_cb
    elif method == "logit_linear":
        p_lr = np.clip(s_lr, 1e-12, 1.0 - 1e-12)
        p_cb = np.clip(s_cb, 1e-12, 1.0 - 1e-12)
        blended = expit(w_lr * logit(p_lr) + w_cb * logit(p_cb))
    elif method == "rank_pct":
        blended = np.empty(len(merged), dtype=np.float64)
        for m in merged["month"].unique():
            mask = (merged["month"] == m).to_numpy()
            r_lr = rankdata(s_lr[mask], method="average") / float(mask.sum())
            r_cb = rankdata(s_cb[mask], method="average") / float(mask.sum())
            blended[mask] = w_lr * r_lr + w_cb * r_cb
    else:
        raise ValueError(f"Unknown blend method: {method}")

    out = merged[["client_id", "fold", "month", "y_true"]].copy()
    out["score"] = blended
    return out.sort_values(["month", "client_id"]).reset_index(drop=True)
