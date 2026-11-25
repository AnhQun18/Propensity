"""Deterministic tie-breaking and propensity evaluation metrics."""
import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def deterministic_tie_hash(client_ids: np.ndarray | list[str], seed: int = 42) -> np.ndarray:
    """Map (seed, client_id) deterministically to uint64 for unbiased tie-breaking."""
    prefix = f"{int(seed)}:"
    out = np.empty(len(client_ids), dtype=np.uint64)
    sha = hashlib.sha256
    for i, cid in enumerate(client_ids):
        digest = sha((prefix + str(cid)).encode("utf-8")).digest()
        out[i] = int.from_bytes(digest[:8], "big", signed=False)
    return out


def deterministic_rank_order(
    scores: np.ndarray,
    client_ids: np.ndarray | list[str],
    seed: int = 42,
    tie_hashes: np.ndarray | None = None,
) -> np.ndarray:
    """Return indices sorting scores descending with deterministic hash tie-breaking."""
    s = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(s)):
        raise ValueError("Prediction scores contain NaN or infinite values")
    hashes = deterministic_tie_hash(client_ids, seed) if tie_hashes is None else tie_hashes
    return np.lexsort((hashes, -s))


def safe_log_loss(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def safe_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def calibration_bins_table(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> tuple[pd.DataFrame, float]:
    """Compute equal-frequency (quantile) calibration bins and weighted ECE."""
    n = len(y_true)
    if n == 0:
        return pd.DataFrame(), float("nan")
    order = np.argsort(probs, kind="mergesort")
    y_s = np.asarray(y_true, dtype=np.float64)[order]
    p_s = np.asarray(probs, dtype=np.float64)[order]
    splits = np.array_split(np.arange(n), min(n_bins, n))
    rows = []
    ece = 0.0
    for b_idx, idxs in enumerate(splits, start=1):
        if len(idxs) == 0:
            continue
        cnt = int(len(idxs))
        mean_p = float(np.mean(p_s[idxs]))
        obs_r = float(np.mean(y_s[idxs]))
        pos = int(np.sum(y_s[idxs]))
        ece += (cnt / n) * abs(obs_r - mean_p)
        rows.append(
            {
                "bin": b_idx,
                "n": cnt,
                "positives": pos,
                "mean_pred": mean_p,
                "observed_rate": obs_r,
                "abs_gap": abs(obs_r - mean_p),
            }
        )
    return pd.DataFrame(rows), float(ece)


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    client_ids: np.ndarray | list[str],
    budget_fractions: tuple[float, ...] = (0.005, 0.01, 0.05),
    primary_budget: float = 0.01,
    seed: int = 42,
    is_probabilistic: bool = True,
    baseline_logloss: float | None = None,
    tie_hashes: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute full ranking, discrimination, confusion matrix, and calibration metrics."""
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    n = int(len(y))
    n_pos = int(np.sum(y))
    prevalence = float(n_pos / n) if n > 0 else float("nan")

    order = deterministic_rank_order(s, client_ids, seed=seed, tie_hashes=tie_hashes)
    y_sorted = y[order]

    metrics: dict[str, Any] = {
        "n": n,
        "positives": n_pos,
        "prevalence": prevalence,
    }

    has_both_classes = (n_pos > 0) and (n_pos < n)
    if has_both_classes:
        metrics["average_precision"] = float(average_precision_score(y, s))
        metrics["roc_auc"] = float(roc_auc_score(y, s))
    else:
        metrics["average_precision"] = None
        metrics["roc_auc"] = None

    for frac in budget_fractions:
        tag = _budget_tag(frac)
        k = min(n, max(1, int(math.ceil(frac * n)))) if n > 0 else 0
        hits = int(np.sum(y_sorted[:k])) if k > 0 else 0
        prec = float(hits / k) if k > 0 else None
        rec = float(hits / n_pos) if n_pos > 0 else None
        lift = float(prec / prevalence) if (prec is not None and prevalence > 0) else None
        metrics[f"k_{tag}"] = k
        metrics[f"hits_{tag}"] = hits
        metrics[f"precision_{tag}"] = prec
        metrics[f"recall_{tag}"] = rec
        metrics[f"lift_{tag}"] = lift

        if abs(frac - primary_budget) < 1e-9:
            tp = hits
            fp = k - hits
            fn = n_pos - hits
            tn = (n - k) - fn
            metrics["cm_primary_tp"] = int(tp)
            metrics["cm_primary_fp"] = int(fp)
            metrics["cm_primary_fn"] = int(fn)
            metrics["cm_primary_tn"] = int(tn)

    if is_probabilistic and n > 0:
        ll = safe_log_loss(y, s)
        bs = safe_brier_score(y, s)
        pred_pos = float(np.sum(s))
        cal_ratio = float(pred_pos / n_pos) if n_pos > 0 else None
        cal_df, ece = calibration_bins_table(y, s, n_bins=10)
        ll_skill = (
            float(1.0 - ll / baseline_logloss)
            if (baseline_logloss is not None and baseline_logloss > 0)
            else None
        )
        metrics["log_loss"] = ll
        metrics["log_loss_skill"] = ll_skill
        metrics["brier"] = bs
        metrics["predicted_positives"] = pred_pos
        metrics["calibration_ratio"] = cal_ratio
        metrics["ece_quantile"] = ece
    else:
        cal_df = pd.DataFrame()
        metrics["log_loss"] = None
        metrics["log_loss_skill"] = None
        metrics["brier"] = None
        metrics["predicted_positives"] = None
        metrics["calibration_ratio"] = None
        metrics["ece_quantile"] = None

    return metrics, cal_df


def _budget_tag(frac: float) -> str:
    pct = frac * 100.0
    if abs(pct - round(pct)) < 1e-6:
        return f"{int(round(pct))}pct"
    return f"{pct:g}pct".replace(".", "p")
