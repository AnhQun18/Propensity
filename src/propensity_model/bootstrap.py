"""Fast exact client-clustered bootstrap for 95% CIs and paired model comparisons."""
import math
from typing import Any

import numpy as np
import pandas as pd

from .metrics import deterministic_rank_order, deterministic_tie_hash


def _prepare_sorted_arrays(
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

    # Identify unique score group boundaries (descending order) for exact AP & ROC-AUC
    diff_mask = np.empty(len(s_s), dtype=bool)
    diff_mask[-1] = True
    if len(s_s) > 1:
        diff_mask[:-1] = s_s[:-1] != s_s[1:]
    group_ends = np.flatnonzero(diff_mask)

    row_ll = None
    if is_probabilistic:
        p = np.clip(s_s, 1e-15, 1.0 - 1e-15)
        row_ll = -(y_s * np.log(p) + (1.0 - y_s) * np.log(1.0 - p))

    return {
        "y_s": y_s,
        "c_s": c_s,
        "group_ends": group_ends,
        "row_ll": row_ll,
        "is_probabilistic": is_probabilistic,
    }


def _weighted_metrics_from_sorted(
    prep: dict[str, Any],
    client_weights: np.ndarray,
    primary_budget: float,
) -> dict[str, float]:
    rw = client_weights[prep["c_s"]].astype(np.float64)
    total_n = float(rw.sum())
    if total_n <= 0.0:
        return {k: float("nan") for k in ("recall_1pct", "precision_1pct", "lift_1pct", "average_precision", "roc_auc", "log_loss")}

    y_s = prep["y_s"]
    pos_w = rw * y_s
    total_pos = float(pos_w.sum())
    total_neg = total_n - total_pos

    cum_w = np.cumsum(rw)
    cum_pos = np.cumsum(pos_w)

    # Top-K at primary budget (exact weighted threshold)
    k_target = max(1, int(math.ceil(primary_budget * total_n)))
    idx = int(np.searchsorted(cum_w, k_target, side="left"))
    if idx >= len(cum_w):
        hits = total_pos
    else:
        prev_w = float(cum_w[idx - 1]) if idx > 0 else 0.0
        prev_pos = float(cum_pos[idx - 1]) if idx > 0 else 0.0
        rem = k_target - prev_w
        hits = prev_pos + rem * float(y_s[idx])

    prec = hits / k_target
    rec = (hits / total_pos) if total_pos > 0.0 else float("nan")
    prev = total_pos / total_n
    lift = (prec / prev) if prev > 0.0 else float("nan")

    # Exact weighted step-function AP & trapezoidal ROC-AUC at unique score boundaries
    if total_pos > 0.0 and total_neg > 0.0:
        ge = prep["group_ends"]
        tp = cum_pos[ge]
        tot = cum_w[ge]
        fp = tot - tp

        # Average Precision: sum_t (recall_t - recall_{t-1}) * precision_t
        delta_tp = np.empty_like(tp)
        delta_tp[0] = tp[0]
        if len(tp) > 1:
            delta_tp[1:] = tp[1:] - tp[:-1]
        nonzero_tot = tot > 0.0
        precision_t = np.zeros_like(tp)
        precision_t[nonzero_tot] = tp[nonzero_tot] / tot[nonzero_tot]
        ap = float(np.sum((delta_tp / total_pos) * precision_t))

        # ROC-AUC: trapezoidal rule on (FPR, TPR) starting at (0, 0)
        tpr = np.concatenate(([0.0], tp / total_pos))
        fpr = np.concatenate(([0.0], fp / total_neg))
        auc = float(np.trapezoid(tpr, fpr))
    else:
        ap = float("nan")
        auc = float("nan")

    ll = float(np.dot(rw, prep["row_ll"]) / total_n) if prep["is_probabilistic"] else float("nan")
    return {
        "recall_1pct": float(rec),
        "precision_1pct": float(prec),
        "lift_1pct": float(lift),
        "average_precision": ap,
        "roc_auc": auc,
        "log_loss": ll,
    }


def _summarize_distribution(values: np.ndarray) -> tuple[float | None, float | None, float | None, float | None]:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return None, None, None, None
    mean_v = float(np.mean(valid))
    lo = float(np.percentile(valid, 2.5))
    hi = float(np.percentile(valid, 97.5))
    # Two-sided bootstrap p-value against null delta == 0
    p_le = float(np.mean(valid <= 0.0))
    p_ge = float(np.mean(valid >= 0.0))
    p_two = min(1.0, 2.0 * min(p_le, p_ge))
    return mean_v, lo, hi, p_two


def cluster_bootstrap_single(
    df: pd.DataFrame,
    is_probabilistic: bool = True,
    primary_budget: float = 0.01,
    n_replicates: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Client-clustered bootstrap 95% CIs over pooled (or single-month) predictions."""
    codes, uniques = pd.factorize(df["client_id"].to_numpy(), sort=True)
    n_clients = len(uniques)
    tie_hashes = deterministic_tie_hash(df["client_id"].to_numpy(), seed=seed)
    prep = _prepare_sorted_arrays(
        y=df["y_true"].to_numpy(dtype=np.int8),
        scores=df["score"].to_numpy(dtype=np.float64),
        client_codes=codes,
        tie_hashes=tie_hashes,
        is_probabilistic=is_probabilistic,
    )

    ones = np.ones(n_clients, dtype=np.int64)
    point = _weighted_metrics_from_sorted(prep, ones, primary_budget)

    metric_keys = ("recall_1pct", "precision_1pct", "lift_1pct", "average_precision", "roc_auc", "log_loss")
    samples = {k: np.empty(n_replicates, dtype=np.float64) for k in metric_keys}
    rng = np.random.default_rng(seed)
    for b in range(n_replicates):
        draw = rng.integers(0, n_clients, size=n_clients)
        cw = np.bincount(draw, minlength=n_clients)
        res = _weighted_metrics_from_sorted(prep, cw, primary_budget)
        for k in metric_keys:
            samples[k][b] = res[k]

    summary: dict[str, Any] = {"n_clients": int(n_clients), "n_rows": int(len(df)), "replicates": int(n_replicates)}
    for k in metric_keys:
        pt = point[k] if np.isfinite(point[k]) else None
        _, lo, hi, _ = _summarize_distribution(samples[k])
        summary[k] = pt
        summary[f"{k}_ci_low"] = lo
        summary[f"{k}_ci_high"] = hi
    return summary


def paired_cluster_bootstrap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    is_prob_a: bool = True,
    is_prob_b: bool = True,
    primary_budget: float = 0.01,
    n_replicates: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired client-clustered bootstrap for delta = metric(B) - metric(A)."""
    merged = df_a[["client_id", "month", "y_true", "score"]].merge(
        df_b[["client_id", "month", "score"]],
        on=["client_id", "month"],
        suffixes=("_a", "_b"),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(df_a) or len(merged) != len(df_b):
        raise ValueError("Paired bootstrap inputs do not match 1-to-1 on (client_id, month)")

    codes, uniques = pd.factorize(merged["client_id"].to_numpy(), sort=True)
    n_clients = len(uniques)
    tie_hashes = deterministic_tie_hash(merged["client_id"].to_numpy(), seed=seed)
    y = merged["y_true"].to_numpy(dtype=np.int8)

    prep_a = _prepare_sorted_arrays(y, merged["score_a"].to_numpy(dtype=np.float64), codes, tie_hashes, is_prob_a)
    prep_b = _prepare_sorted_arrays(y, merged["score_b"].to_numpy(dtype=np.float64), codes, tie_hashes, is_prob_b)

    ones = np.ones(n_clients, dtype=np.int64)
    pt_a = _weighted_metrics_from_sorted(prep_a, ones, primary_budget)
    pt_b = _weighted_metrics_from_sorted(prep_b, ones, primary_budget)

    metric_keys = ("recall_1pct", "precision_1pct", "lift_1pct", "average_precision", "roc_auc", "log_loss")
    deltas = {k: np.empty(n_replicates, dtype=np.float64) for k in metric_keys}
    rng = np.random.default_rng(seed)
    for b in range(n_replicates):
        draw = rng.integers(0, n_clients, size=n_clients)
        cw = np.bincount(draw, minlength=n_clients)
        res_a = _weighted_metrics_from_sorted(prep_a, cw, primary_budget)
        res_b = _weighted_metrics_from_sorted(prep_b, cw, primary_budget)
        for k in metric_keys:
            deltas[k][b] = res_b[k] - res_a[k]

    out: dict[str, Any] = {"n_clients": int(n_clients), "n_rows": int(len(merged)), "replicates": int(n_replicates)}
    for k in metric_keys:
        raw_delta = pt_b[k] - pt_a[k]
        pt_delta = float(raw_delta) if np.isfinite(raw_delta) else None
        _, lo, hi, pval = _summarize_distribution(deltas[k])
        out[f"delta_{k}"] = pt_delta
        out[f"delta_{k}_ci_low"] = lo
        out[f"delta_{k}_ci_high"] = hi
        out[f"delta_{k}_p_value"] = pval
    return out
