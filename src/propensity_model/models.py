"""Baseline model wrappers and post-hoc probability calibration."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier
import joblib
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _clip_probs(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


class PrevalenceBaseline:
    is_probabilistic = True

    def __init__(self):
        self.base_rate_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, cold_start: np.ndarray | None = None):
        self.base_rate_ = float(np.mean(y))
        return self

    def predict_scores(self, X: np.ndarray, cold_start: np.ndarray | None = None) -> np.ndarray:
        return np.full(len(X), self.base_rate_, dtype=np.float64)

    def feature_importance(self, feature_names: tuple[str, ...]) -> list[tuple[str, float]]:
        return []

    def save(self, path: Path) -> Path:
        out = path.with_suffix(".joblib")
        joblib.dump({"type": "prevalence", "base_rate": self.base_rate_}, out)
        return out


class HeuristicRFBaseline:
    """Rank strictly by trx_84d_event_count desc, then trx_84d_days_since_last_event asc."""

    is_probabilistic = False

    def fit(self, X: np.ndarray, y: np.ndarray, cold_start: np.ndarray | None = None):
        return self

    def predict_scores(self, X: np.ndarray, cold_start: np.ndarray | None = None) -> np.ndarray:
        if X.shape[1] != 2:
            raise ValueError("HeuristicRFBaseline expects 2 features: (count, days_since_last)")
        count = X[:, 0].astype(np.float64)
        recency = X[:, 1].astype(np.float64)
        valid_count = np.where(np.isfinite(count), count, -1.0)
        recency_bonus = np.where(
            np.isfinite(recency) & (recency >= 0.0),
            0.99 / (1.0 + recency),
            0.0,
        )
        return np.where(valid_count > 0.0, valid_count + recency_bonus, valid_count)

    def feature_importance(self, feature_names: tuple[str, ...]) -> list[tuple[str, float]]:
        return []

    def save(self, path: Path) -> Path:
        out = path.with_suffix(".joblib")
        joblib.dump({"type": "heuristic_rf"}, out)
        return out


class RidgeLogisticBaseline:
    """Train-only median imputation (+ missing indicators) -> StandardScaler -> L2 Logistic."""

    is_probabilistic = True

    def __init__(self, c: float = 1.0, max_iter: int = 300, seed: int = 42):
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.pipeline_: Pipeline | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, cold_start: np.ndarray | None = None):
        self.pipeline_ = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.c,
                        max_iter=self.max_iter,
                        solver="lbfgs",
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        self.pipeline_.fit(X, y)
        return self

    def predict_scores(self, X: np.ndarray, cold_start: np.ndarray | None = None) -> np.ndarray:
        assert self.pipeline_ is not None
        return self.pipeline_.predict_proba(X)[:, 1].astype(np.float64)

    def feature_importance(self, feature_names: tuple[str, ...]) -> list[tuple[str, float]]:
        assert self.pipeline_ is not None
        coefs = np.abs(self.pipeline_.named_steps["clf"].coef_.ravel())
        n_orig = len(feature_names)
        pairs = [(feature_names[i], float(coefs[i])) for i in range(min(n_orig, len(coefs)))]
        ind = self.pipeline_.named_steps["imputer"].indicator_
        if ind is not None and len(ind.features_) > 0:
            for offset, feat_idx in enumerate(ind.features_):
                col_pos = n_orig + offset
                if col_pos < len(coefs) and feat_idx < n_orig:
                    pairs.append((f"{feature_names[feat_idx]}__isna", float(coefs[col_pos])))
        pairs.sort(key=lambda kv: -kv[1])
        return pairs

    def save(self, path: Path) -> Path:
        out = path.with_suffix(".joblib")
        joblib.dump(self.pipeline_, out, compress=3)
        return out


class CatBoostBaseline:
    """Natural-distribution binary CatBoost keeping semantic NaNs."""

    is_probabilistic = True

    def __init__(self, params: dict[str, Any], seed: int = 42):
        cfg = dict(params)
        cfg["random_seed"] = int(seed)
        cfg.setdefault("loss_function", "Logloss")
        cfg.setdefault("eval_metric", "Logloss")
        cfg.setdefault("verbose", False)
        cfg.setdefault("allow_writing_files", False)
        self.params = cfg
        self.model_: CatBoostClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, cold_start: np.ndarray | None = None):
        self.model_ = CatBoostClassifier(**self.params)
        self.model_.fit(X, y)
        return self

    def predict_scores(self, X: np.ndarray, cold_start: np.ndarray | None = None) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.predict_proba(X)[:, 1].astype(np.float64)

    def staged_predictions(
        self, X: np.ndarray, checkpoints: tuple[int, ...]
    ) -> dict[int, np.ndarray]:
        assert self.model_ is not None
        tree_count = int(self.model_.tree_count_)
        valid_cps = sorted({min(cp, tree_count) for cp in checkpoints if cp >= 1})
        out: dict[int, np.ndarray] = {}
        for cp in valid_cps:
            preds = self.model_.predict_proba(X, ntree_end=cp)[:, 1].astype(np.float64)
            out[cp] = preds
        return out

    def feature_importance(self, feature_names: tuple[str, ...]) -> list[tuple[str, float]]:
        assert self.model_ is not None
        imp = self.model_.get_feature_importance(type="PredictionValuesChange")
        pairs = [(feature_names[i], float(imp[i])) for i in range(len(feature_names))]
        pairs.sort(key=lambda kv: -kv[1])
        return pairs

    def save(self, path: Path) -> Path:
        assert self.model_ is not None
        out = path.with_suffix(".cbm")
        self.model_.save_model(str(out))
        return out


class ColdStartFallbackWrapper:
    """Fit CatBoost on non-cold-start rows; route cold_start==1 to Empirical-Bayes prior."""

    is_probabilistic = True

    def __init__(self, params: dict[str, Any], alpha: float = 20.0, seed: int = 42):
        self.base_model = CatBoostBaseline(params=params, seed=seed)
        self.alpha = float(alpha)
        self.cold_start_prob_: float = 0.0
        self.train_prior_: float = 0.0
        self.train_cold_n_: int = 0
        self.train_cold_pos_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, cold_start: np.ndarray | None = None):
        if cold_start is None:
            raise ValueError("ColdStartFallbackWrapper requires cold_start indicator array")
        cs_mask = cold_start == 1
        self.train_prior_ = float(np.mean(y))
        self.train_cold_n_ = int(np.sum(cs_mask))
        self.train_cold_pos_ = int(np.sum(y[cs_mask]))
        self.cold_start_prob_ = float(
            (self.train_cold_pos_ + self.alpha * self.train_prior_)
            / (self.train_cold_n_ + self.alpha)
        )
        non_cs = ~cs_mask
        if int(np.sum(non_cs)) == 0:
            raise ValueError("No non-cold-start rows available for base model")
        self.base_model.fit(X[non_cs], y[non_cs])
        return self

    def predict_scores(self, X: np.ndarray, cold_start: np.ndarray | None = None) -> np.ndarray:
        if cold_start is None:
            raise ValueError("ColdStartFallbackWrapper requires cold_start indicator array")
        cs_mask = cold_start == 1
        scores = np.full(len(X), self.cold_start_prob_, dtype=np.float64)
        non_cs = ~cs_mask
        if np.any(non_cs):
            scores[non_cs] = self.base_model.predict_scores(X[non_cs])
        return scores

    def staged_predictions(
        self, X: np.ndarray, cold_start: np.ndarray, checkpoints: tuple[int, ...]
    ) -> dict[int, np.ndarray]:
        cs_mask = cold_start == 1
        non_cs = ~cs_mask
        base_staged = self.base_model.staged_predictions(X[non_cs], checkpoints) if np.any(non_cs) else {}
        out: dict[int, np.ndarray] = {}
        for cp, non_cs_preds in base_staged.items():
            scores = np.full(len(X), self.cold_start_prob_, dtype=np.float64)
            scores[non_cs] = non_cs_preds
            out[cp] = scores
        return out

    def feature_importance(self, feature_names: tuple[str, ...]) -> list[tuple[str, float]]:
        return self.base_model.feature_importance(feature_names)

    def save(self, path: Path) -> Path:
        model_path = self.base_model.save(path)
        meta_path = path.with_suffix(".fallback.joblib")
        joblib.dump(
            {
                "alpha": self.alpha,
                "cold_start_prob": self.cold_start_prob_,
                "train_prior": self.train_prior_,
                "train_cold_n": self.train_cold_n_,
                "train_cold_pos": self.train_cold_pos_,
                "base_model_path": model_path.name,
            },
            meta_path,
        )
        return model_path


@dataclass
class ProbabilityCalibrator:
    """Monotone post-hoc calibrator fit strictly on calibration month predictions."""

    method: str  # "intercept_shift" | "platt"
    slope_: float = 1.0
    intercept_: float = 0.0

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        z = logit(_clip_probs(raw_probs))
        y = np.asarray(y_true, dtype=np.float64)

        def nll(a: float, b: float) -> float:
            eta = np.clip(a * z + b, -30.0, 30.0)
            # Stable log(1 + exp(eta)) - y * eta
            return float(np.mean(np.logaddexp(0.0, eta) - y * eta))

        if self.method == "intercept_shift":
            res = minimize_scalar(lambda b: nll(1.0, b), bounds=(-10.0, 10.0), method="bounded")
            self.slope_ = 1.0
            self.intercept_ = float(res.x)
        elif self.method == "platt":
            res = minimize(
                lambda params: nll(params[0], params[1]),
                x0=np.array([1.0, 0.0], dtype=np.float64),
                bounds=[(1e-4, 10.0), (-10.0, 10.0)],
                method="L-BFGS-B",
            )
            self.slope_ = float(res.x[0])
            self.intercept_ = float(res.x[1])
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        z = logit(_clip_probs(raw_probs))
        return expit(np.clip(self.slope_ * z + self.intercept_, -30.0, 30.0)).astype(np.float64)
