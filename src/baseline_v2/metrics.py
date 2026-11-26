"""Deterministic ranking, monthly/macro-monthly/pooled metrics, and Client-Clustered Bootstrap."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BOOT_KEYS: tuple[str, ...] = (
    "lift_1pct",
    "average_precision",
    "roc_auc",
    "recall_1pct",
    "precision_1pct",
    "lift_0p5pct",
    "recall_0p5pct",
    "lift_5pct",
    "recall_5pct",
    "log_loss",
    "brier",
)


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
    """Sort descending by score with deterministic SHA-256 tie-breaking."""
    s = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(s)):
        raise ValueError("Prediction scores contain NaN or infinite values")
    hashes = deterministic_tie_hash(client_ids, seed) if tie_hashes is None else tie_hashes
    return np.lexsort((hashes, -s))


def compute_slice_point_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    client_ids: np.ndarray,
    budget_fractions: tuple[float, ...] = (0.005, 0.01, 0.05),
    primary_budget: float = 0.01,
    seed: int = 42,
    is_probabilistic: bool = True,
    baseline_logloss: float | None = None,
    baseline_brier: float | None = None,
    tie_hashes: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute full ranking, operational top-K, and probability metrics on one slice/month."""
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    n = int(len(y))
    n_pos = int(np.sum(y))
    prevalence = float(n_pos / n) if n > 0 else float("nan")

    order = deterministic_rank_order(s, client_ids, seed=seed, tie_hashes=tie_hashes)
    y_sorted = y[order]

    out: dict[str, Any] = {
        "n": n,
        "positives": n_pos,
        "prevalence": prevalence,
    }
    has_both = (n_pos > 0) and (n_pos < n)
    out["average_precision"] = float(average_precision_score(y, s)) if has_both else float("nan")
    out["roc_auc"] = float(roc_auc_score(y, s)) if has_both else float("nan")

    for frac in budget_fractions:
        tag = f"{frac * 100:g}".replace(".", "p") + "pct"
        k = min(n, max(1, int(math.ceil(frac * n)))) if n > 0 else 0
        hits = int(np.sum(y_sorted[:k])) if k > 0 else 0
        prec = float(hits / k) if k > 0 else float("nan")
        rec = float(hits / n_pos) if n_pos > 0 else float("nan")
        lift = float(prec / prevalence) if (np.isfinite(prec) and prevalence > 0) else float("nan")
        out[f"k_{tag}"] = k
        out[f"hits_{tag}"] = hits
        out[f"precision_{tag}"] = prec
        out[f"recall_{tag}"] = rec
        out[f"lift_{tag}"] = lift

        if abs(frac - primary_budget) < 1e-9:
            out["cm_tp"] = hits
            out["cm_fp"] = k - hits
            out["cm_fn"] = n_pos - hits
            out["cm_tn"] = (n - k) - (n_pos - hits)

    if is_probabilistic and n > 0:
        p = np.clip(s, 1e-15, 1.0 - 1e-15)
        ll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        bs = float(np.mean((s - y) ** 2))
        out["log_loss"] = ll
        out["brier"] = bs
        out["log_loss_skill"] = (
            float(1.0 - ll / baseline_logloss)
            if (baseline_logloss is not None and baseline_logloss > 0)
            else None
        )
        out["brier_skill"] = (
            float(1.0 - bs / baseline_brier)
            if (baseline_brier is not None and baseline_brier > 0)
            else None
        )
    else:
        out["log_loss"] = None
        out["brier"] = None
        out["log_loss_skill"] = None
        out["brier_skill"] = None
    return out


# ---------------------------------------------------------------------------
# Fast Vectorized Client-Clustered Bootstrap (Per-Month, Macro-Monthly, Pooled)
# ---------------------------------------------------------------------------
def _prep_sorted(
    y: np.ndarray,
    scores: np.ndarray,
    codes: np.ndarray,
    hashes: np.ndarray,
    is_prob: bool,
) -> dict[str, Any]:
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
    return {
        "y_s": y_s,
        "c_s": c_s,
        "ge": np.flatnonzero(diff),
        "row_ll": row_ll,
        "row_bs": row_bs,
        "is_prob": is_prob,
    }


def _weighted_metrics(prep: dict[str, Any], cw: np.ndarray) -> dict[str, float]:
    rw = cw[prep["c_s"]].astype(np.float64)
    tot_n = float(rw.sum())
    if tot_n <= 0.0:
        return {k: float("nan") for k in BOOT_KEYS}
    y_s = prep["y_s"]
    pos_w = rw * y_s
    tot_pos = float(pos_w.sum())
    tot_neg = tot_n - tot_pos
    cum_w = np.cumsum(rw)
    cum_pos = np.cumsum(pos_w)
    prev = tot_pos / tot_n if tot_n > 0 else float("nan")

    out: dict[str, float] = {}
    for frac, tag in ((0.005, "0p5pct"), (0.01, "1pct"), (0.05, "5pct")):
        k_t = max(1, int(math.ceil(frac * tot_n)))
        idx = int(np.searchsorted(cum_w, k_t, side="left"))
        if idx >= len(cum_w):
            hits = tot_pos
        else:
            pw = float(cum_w[idx - 1]) if idx > 0 else 0.0
            pp = float(cum_pos[idx - 1]) if idx > 0 else 0.0
            hits = pp + (k_t - pw) * float(y_s[idx])
        prec = hits / k_t
        rec = (hits / tot_pos) if tot_pos > 0 else float("nan")
        lift = (prec / prev) if prev > 0 else float("nan")
        out[f"lift_{tag}"] = lift
        out[f"recall_{tag}"] = rec
        if tag == "1pct":
            out["precision_1pct"] = prec

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
        out["average_precision"] = float(np.sum((dtp / tot_pos) * pt))
        out["roc_auc"] = float(
            np.trapezoid(
                np.concatenate(([0.0], tp / tot_pos)),
                np.concatenate(([0.0], fp / tot_neg)),
            )
        )
    else:
        out["average_precision"] = float("nan")
        out["roc_auc"] = float("nan")

    out["log_loss"] = float(np.dot(rw, prep["row_ll"]) / tot_n) if prep["is_prob"] else float("nan")
    out["brier"] = float(np.dot(rw, prep["row_bs"]) / tot_n) if prep["is_prob"] else float("nan")
    return out


def _ci_stats(vals: np.ndarray) -> tuple[float | None, float | None, float | None]:
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return None, None, None
    lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    p = min(1.0, 2.0 * min(float(np.mean(v <= 0.0)), float(np.mean(v >= 0.0))))
    return lo, hi, p


def _build_client_universe(dfs: list[pd.DataFrame]) -> tuple[np.ndarray, dict[str, int]]:
    all_cids = np.unique(np.concatenate([df["client_id"].to_numpy() for df in dfs]))
    cid_to_code = {str(cid): idx for idx, cid in enumerate(all_cids)}
    return all_cids, cid_to_code


def _encode_codes(df: pd.DataFrame, cid_to_code: dict[str, int]) -> np.ndarray:
    return np.fromiter((cid_to_code[str(c)] for c in df["client_id"].to_numpy()), dtype=np.int64, count=len(df))


def bootstrap_multiseed_per_month(
    seed_dfs: list[pd.DataFrame],
    month: str,
    is_prob: bool,
    n_replicates: int = 1000,
    boot_seed: int = 42,
    tie_seed: int = 42,
) -> dict[str, Any]:
    """Compute per-month multi-seed mean, seed SD, and 95% Client-Clustered Bootstrap CI."""
    month_dfs = [df[df["month"] == month].sort_values("client_id").reset_index(drop=True) for df in seed_dfs]
    all_cids, cid_to_code = _build_client_universe(month_dfs)
    nc = len(all_cids)
    ones = np.ones(nc, dtype=np.int64)

    preps = []
    per_seed_pts = []
    for df in month_dfs:
        codes = _encode_codes(df, cid_to_code)
        hashes = deterministic_tie_hash(df["client_id"].to_numpy(), seed=tie_seed)
        pr = _prep_sorted(df["y_true"].to_numpy(dtype=np.int8), df["score"].to_numpy(dtype=np.float64), codes, hashes, is_prob)
        preps.append(pr)
        per_seed_pts.append(_weighted_metrics(pr, ones))

    samp = {k: np.empty(n_replicates, dtype=np.float64) for k in BOOT_KEYS}
    rng = np.random.default_rng(boot_seed)
    for b in range(n_replicates):
        cw = np.bincount(rng.integers(0, nc, size=nc), minlength=nc)
        res_list = [_weighted_metrics(pr, cw) for pr in preps]
        for k in BOOT_KEYS:
            samp[k][b] = float(np.mean([r[k] for r in res_list]))

    out: dict[str, Any] = {
        "month": month,
        "n_rows": int(len(month_dfs[0])),
        "positives": int(month_dfs[0]["y_true"].sum()),
        "prevalence": float(month_dfs[0]["y_true"].mean()),
        "n_seeds": len(seed_dfs),
        "replicates": n_replicates,
    }
    for k in BOOT_KEYS:
        vals = np.array([pt[k] for pt in per_seed_pts], dtype=np.float64)
        mean_v = float(np.mean(vals)) if np.all(np.isfinite(vals)) else None
        sd_v = float(np.std(vals, ddof=1)) if (len(vals) > 1 and np.all(np.isfinite(vals))) else 0.0
        lo, hi, _ = _ci_stats(samp[k])
        out[f"{k}_mean"] = mean_v
        out[f"{k}_seed_sd"] = sd_v if mean_v is not None else None
        out[f"{k}_ci_low"] = lo
        out[f"{k}_ci_high"] = hi
    return out


def bootstrap_multiseed_macro_monthly(
    seed_dfs: list[pd.DataFrame],
    months: list[str] | tuple[str, ...],
    is_prob: bool,
    n_replicates: int = 1000,
    boot_seed: int = 42,
    tie_seed: int = 42,
) -> dict[str, Any]:
    """Aggregate monthly metrics (monthly K=1%) with Client-Clustered Bootstrap CI across months & seeds."""
    all_cids, cid_to_code = _build_client_universe(seed_dfs)
    nc = len(all_cids)
    ones = np.ones(nc, dtype=np.int64)

    # preps_by_seed_month[s_idx][m_idx]
    preps_sm: list[list[dict[str, Any]]] = []
    for df in seed_dfs:
        m_preps = []
        for m in months:
            sub = df[df["month"] == m].sort_values("client_id").reset_index(drop=True)
            codes = _encode_codes(sub, cid_to_code)
            hashes = deterministic_tie_hash(sub["client_id"].to_numpy(), seed=tie_seed)
            m_preps.append(
                _prep_sorted(sub["y_true"].to_numpy(dtype=np.int8), sub["score"].to_numpy(dtype=np.float64), codes, hashes, is_prob)
            )
        preps_sm.append(m_preps)

    per_seed_macro: list[dict[str, float]] = []
    for m_preps in preps_sm:
        m_pts = [_weighted_metrics(pr, ones) for pr in m_preps]
        per_seed_macro.append({k: float(np.mean([pt[k] for pt in m_pts])) for k in BOOT_KEYS})

    samp = {k: np.empty(n_replicates, dtype=np.float64) for k in BOOT_KEYS}
    rng = np.random.default_rng(boot_seed)
    for b in range(n_replicates):
        cw = np.bincount(rng.integers(0, nc, size=nc), minlength=nc)
        seed_avgs = []
        for m_preps in preps_sm:
            m_res = [_weighted_metrics(pr, cw) for pr in m_preps]
            seed_avgs.append({k: float(np.mean([r[k] for r in m_res])) for k in BOOT_KEYS})
        for k in BOOT_KEYS:
            samp[k][b] = float(np.mean([sa[k] for sa in seed_avgs]))

    out: dict[str, Any] = {
        "scope": "macro_monthly",
        "n_months": len(months),
        "n_clients": nc,
        "n_seeds": len(seed_dfs),
        "replicates": n_replicates,
    }
    for k in BOOT_KEYS:
        vals = np.array([ps[k] for ps in per_seed_macro], dtype=np.float64)
        mean_v = float(np.mean(vals)) if np.all(np.isfinite(vals)) else None
        sd_v = float(np.std(vals, ddof=1)) if (len(vals) > 1 and np.all(np.isfinite(vals))) else 0.0
        lo, hi, _ = _ci_stats(samp[k])
        out[f"{k}_mean"] = mean_v
        out[f"{k}_seed_sd"] = sd_v if mean_v is not None else None
        out[f"{k}_ci_low"] = lo
        out[f"{k}_ci_high"] = hi
    return out


def bootstrap_multiseed_pooled(
    seed_dfs: list[pd.DataFrame],
    is_prob: bool,
    n_replicates: int = 1000,
    boot_seed: int = 42,
    tie_seed: int = 42,
) -> dict[str, Any]:
    """Compute Pooled OOF metrics across all months in stage with Client-Clustered Bootstrap CI."""
    all_cids, cid_to_code = _build_client_universe(seed_dfs)
    nc = len(all_cids)
    ones = np.ones(nc, dtype=np.int64)

    preps = []
    per_seed_pts = []
    for df in seed_dfs:
        sub = df.sort_values(["month", "client_id"]).reset_index(drop=True)
        codes = _encode_codes(sub, cid_to_code)
        hashes = deterministic_tie_hash(sub["client_id"].to_numpy(), seed=tie_seed)
        pr = _prep_sorted(sub["y_true"].to_numpy(dtype=np.int8), sub["score"].to_numpy(dtype=np.float64), codes, hashes, is_prob)
        preps.append(pr)
        per_seed_pts.append(_weighted_metrics(pr, ones))

    samp = {k: np.empty(n_replicates, dtype=np.float64) for k in BOOT_KEYS}
    rng = np.random.default_rng(boot_seed)
    for b in range(n_replicates):
        cw = np.bincount(rng.integers(0, nc, size=nc), minlength=nc)
        res_list = [_weighted_metrics(pr, cw) for pr in preps]
        for k in BOOT_KEYS:
            samp[k][b] = float(np.mean([r[k] for r in res_list]))

    out: dict[str, Any] = {
        "scope": "pooled_oof",
        "n_rows": int(len(seed_dfs[0])),
        "positives": int(seed_dfs[0]["y_true"].sum()),
        "n_clients": nc,
        "n_seeds": len(seed_dfs),
        "replicates": n_replicates,
    }
    for k in BOOT_KEYS:
        vals = np.array([pt[k] for pt in per_seed_pts], dtype=np.float64)
        mean_v = float(np.mean(vals)) if np.all(np.isfinite(vals)) else None
        sd_v = float(np.std(vals, ddof=1)) if (len(vals) > 1 and np.all(np.isfinite(vals))) else 0.0
        lo, hi, _ = _ci_stats(samp[k])
        out[f"{k}_mean"] = mean_v
        out[f"{k}_seed_sd"] = sd_v if mean_v is not None else None
        out[f"{k}_ci_low"] = lo
        out[f"{k}_ci_high"] = hi
    return out


def paired_bootstrap_macro_monthly(
    seed_dfs_a: list[pd.DataFrame],
    seed_dfs_b: list[pd.DataFrame],
    months: list[str] | tuple[str, ...],
    prob_a: bool,
    prob_b: bool,
    n_replicates: int = 1000,
    boot_seed: int = 42,
    tie_seed: int = 42,
) -> dict[str, Any]:
    """Paired Client-Clustered Bootstrap for Macro-Monthly difference (B - A) across seeds."""
    all_cids, cid_to_code = _build_client_universe(seed_dfs_a + seed_dfs_b)
    nc = len(all_cids)
    ones = np.ones(nc, dtype=np.int64)

    def _make_preps(dfs: list[pd.DataFrame], is_prob: bool) -> list[list[dict[str, Any]]]:
        out_sm = []
        for df in dfs:
            m_list = []
            for m in months:
                sub = df[df["month"] == m].sort_values("client_id").reset_index(drop=True)
                codes = _encode_codes(sub, cid_to_code)
                hashes = deterministic_tie_hash(sub["client_id"].to_numpy(), seed=tie_seed)
                m_list.append(
                    _prep_sorted(sub["y_true"].to_numpy(dtype=np.int8), sub["score"].to_numpy(dtype=np.float64), codes, hashes, is_prob)
                )
            out_sm.append(m_list)
        return out_sm

    pa_sm = _make_preps(seed_dfs_a, prob_a)
    pb_sm = _make_preps(seed_dfs_b, prob_b)

    def _mean_across_sm(preps_sm: list[list[dict[str, Any]]], cw: np.ndarray) -> dict[str, float]:
        s_vals = []
        for m_list in preps_sm:
            m_res = [_weighted_metrics(pr, cw) for pr in m_list]
            s_vals.append({k: float(np.mean([r[k] for r in m_res])) for k in BOOT_KEYS})
        return {k: float(np.mean([sv[k] for sv in s_vals])) for k in BOOT_KEYS}

    pt_a = _mean_across_sm(pa_sm, ones)
    pt_b = _mean_across_sm(pb_sm, ones)

    deltas = {k: np.empty(n_replicates, dtype=np.float64) for k in BOOT_KEYS}
    rng = np.random.default_rng(boot_seed)
    for b in range(n_replicates):
        cw = np.bincount(rng.integers(0, nc, size=nc), minlength=nc)
        ra = _mean_across_sm(pa_sm, cw)
        rb = _mean_across_sm(pb_sm, cw)
        for k in BOOT_KEYS:
            deltas[k][b] = rb[k] - ra[k]

    out: dict[str, Any] = {"n_clients": nc, "n_months": len(months), "replicates": n_replicates}
    for k in BOOT_KEYS:
        d = pt_b[k] - pt_a[k]
        rel = (d / pt_a[k]) if (np.isfinite(d) and pt_a[k] != 0 and np.isfinite(pt_a[k])) else None
        lo, hi, pval = _ci_stats(deltas[k])
        out[f"delta_{k}"] = float(d) if np.isfinite(d) else None
        out[f"rel_delta_{k}"] = float(rel) if rel is not None else None
        out[f"delta_{k}_ci_low"] = lo
        out[f"delta_{k}_ci_high"] = hi
        out[f"delta_{k}_p_value"] = pval
    return out


def holm_bonferroni(p_values: list[float | None]) -> list[float | None]:
    """Compute step-down Holm-Bonferroni adjusted p-values."""
    valid_idx = [i for i, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    m = len(valid_idx)
    out: list[float | None] = [None] * len(p_values)
    if m == 0:
        return out
    ordered = sorted(valid_idx, key=lambda idx: float(p_values[idx]))
    running_max = 0.0
    for rank, idx in enumerate(ordered):
        adj = min(1.0, float(p_values[idx]) * (m - rank))
        running_max = max(running_max, adj)
        out[idx] = running_max
    return out
