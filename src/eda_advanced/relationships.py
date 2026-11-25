"""Bivariate & multivariate EDA: correlations, Cramer's V, VIF, WoE/IV, and binary target signal scan."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu, spearmanr

from ._compat import load_dataframe


def correlation_analysis(
    source: Any,
    *,
    columns: Sequence[str] | None = None,
    method: str = "spearman",
    threshold: float = 0.5,
    **source_options: Any,
) -> dict[str, pd.DataFrame]:
    """Compute numeric correlation matrix (pearson/spearman/kendall) and strong correlated pairs."""
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'")
    df = load_dataframe(source, columns=columns, **source_options)
    num_cols = list(columns) if columns is not None else [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    if not num_cols:
        return {"matrix": pd.DataFrame(), "top_pairs": pd.DataFrame(columns=["col_a", "col_b", "corr", "abs_corr"])}

    mat = df[num_cols].corr(method=method)
    pairs = []
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            val = float(mat.loc[a, b])
            if np.isfinite(val) and abs(val) >= threshold:
                pairs.append({"col_a": a, "col_b": b, "corr": val, "abs_corr": abs(val)})
    top_pairs = pd.DataFrame(pairs, columns=["col_a", "col_b", "corr", "abs_corr"]).sort_values(
        "abs_corr", ascending=False
    ).reset_index(drop=True)
    return {"matrix": mat, "top_pairs": top_pairs}


def _cramers_v_corrected(contingency: np.ndarray) -> tuple[float, float, float]:
    chi2, pval, _, _ = chi2_contingency(contingency)
    n = float(contingency.sum())
    if n <= 1:
        return np.nan, float(chi2), float(pval)
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    v = float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0
    return v, float(chi2), float(pval)


def categorical_associations(
    source: Any,
    *,
    columns: Sequence[str] | None = None,
    max_cardinality: int = 50,
    **source_options: Any,
) -> pd.DataFrame:
    """Compute pairwise bias-corrected Cramer's V and Chi-square p-values for categorical columns."""
    df = load_dataframe(source, columns=columns, **source_options)
    if columns is not None:
        cat_cols = list(columns)
    else:
        cat_cols = [
            c for c in df.columns
            if (not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))
            and not pd.api.types.is_datetime64_any_dtype(df[c])
            and 2 <= df[c].nunique(dropna=True) <= max_cardinality
        ]
    rows = []
    for a, b in combinations(cat_cols, 2):
        sub = df[[a, b]].dropna()
        if sub.empty or sub[a].nunique() < 2 or sub[b].nunique() < 2:
            continue
        ct = pd.crosstab(sub[a], sub[b]).to_numpy()
        v, chi2, pval = _cramers_v_corrected(ct)
        rows.append({
            "col_a": a,
            "col_b": b,
            "n_obs": int(len(sub)),
            "cramers_v": v,
            "chi2": chi2,
            "p_value": pval,
        })
    res = pd.DataFrame(rows, columns=["col_a", "col_b", "n_obs", "cramers_v", "chi2", "p_value"])
    return res.sort_values("cramers_v", ascending=False).reset_index(drop=True) if not res.empty else res


def compute_vif(
    source: Any,
    *,
    columns: Sequence[str] | None = None,
    **source_options: Any,
) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) per numeric column using OLS regression (complete cases)."""
    df = load_dataframe(source, columns=columns, **source_options)
    num_cols = list(columns) if columns is not None else [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    if len(num_cols) < 2:
        return pd.DataFrame([{"column": c, "vif": 1.0, "r2": 0.0} for c in num_cols])

    clean = df[num_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= len(num_cols):
        raise ValueError("Not enough complete finite rows to compute VIF")

    x_all = clean.to_numpy(dtype=float)
    rows = []
    for idx, col in enumerate(num_cols):
        y = x_all[:, idx]
        others = np.delete(x_all, idx, axis=1)
        design = np.column_stack([np.ones(len(y)), others])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        vif = float(1.0 / (1.0 - r2)) if r2 < 1.0 - 1e-12 else float("inf")
        rows.append({"column": col, "vif": vif, "r2": float(r2)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def woe_iv_table(
    series: pd.Series,
    target: pd.Series,
    *,
    n_bins: int = 10,
    eps: float = 0.5,
) -> tuple[pd.DataFrame, float]:
    """Compute Weight of Evidence (WoE) & Information Value (IV) for a numeric or categorical feature."""
    y = pd.to_numeric(target, errors="coerce")
    valid = y.isin([0, 1])
    s = series[valid]
    y = y[valid].astype(int)

    is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if is_num:
        non_null = pd.to_numeric(s, errors="coerce")
        finite = non_null.replace([np.inf, -np.inf], np.nan)
        binned = pd.Series("__MISSING__", index=s.index, dtype="object")
        ok = finite.notna()
        if ok.any():
            if finite[ok].nunique() > n_bins:
                bins = pd.qcut(finite[ok], q=n_bins, duplicates="drop").astype(str)
            else:
                bins = finite[ok].astype(str)
            binned.loc[ok] = bins
    else:
        binned = s.astype("string").fillna("__MISSING__").astype(str)

    tab = pd.DataFrame({"bin": binned, "y": y}).groupby("bin", dropna=False).agg(
        total=("y", "size"),
        events=("y", "sum"),
    ).reset_index()
    tab["non_events"] = tab["total"] - tab["events"]
    total_ev = float(tab["events"].sum())
    total_non = float(tab["non_events"].sum())
    k = len(tab)

    # Laplace-smoothed distributions prevent log(0)
    dist_ev = (tab["events"] + eps) / (total_ev + eps * k)
    dist_non = (tab["non_events"] + eps) / (total_non + eps * k)
    tab["event_rate"] = tab["events"] / tab["total"].replace(0, np.nan)
    tab["dist_events"] = dist_ev
    tab["dist_non_events"] = dist_non
    tab["woe"] = np.log(dist_ev / dist_non)
    tab["iv_component"] = (dist_ev - dist_non) * tab["woe"]
    iv_total = float(tab["iv_component"].sum())
    return tab.sort_values("woe", ascending=False).reset_index(drop=True), iv_total


def _univariate_auc(x: np.ndarray, y: np.ndarray) -> float:
    pos = y == 1
    neg = y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(x).rank(method="average").to_numpy()
    u_stat = float(ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0)
    return u_stat / (n_pos * n_neg)


def target_signal_scan(
    source: Any,
    *,
    target_col: str,
    features: Sequence[str] | None = None,
    n_bins: int = 10,
    **source_options: Any,
) -> pd.DataFrame:
    """Rank every numeric and categorical feature by univariate association with a binary 0/1 target."""
    df = load_dataframe(source, **source_options)
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col!r} not found")
    y_full = pd.to_numeric(df[target_col], errors="coerce")
    valid_y = y_full.isin([0, 1])
    df = df.loc[valid_y].copy()
    y = y_full.loc[valid_y].astype(int)

    cols = list(features) if features is not None else [c for c in df.columns if c != target_col]
    rows: list[dict[str, Any]] = []
    for col in cols:
        s = df[col]
        miss_pct = float(s.isna().mean() * 100.0) if len(s) else 0.0
        first_valid = s.dropna().iloc[0] if s.notna().any() else None
        if isinstance(first_valid, (list, tuple, np.ndarray, dict)):
            continue
        _, iv = woe_iv_table(s, y, n_bins=n_bins)
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        rec: dict[str, Any] = {
            "feature": col,
            "kind": "numeric" if is_num else "categorical",
            "missing_pct": miss_pct,
            "iv": iv,
            "auc": np.nan,
            "auc_strength": np.nan,
            "ks_stat": np.nan,
            "spearman_r": np.nan,
            "cramers_v": np.nan,
            "p_value": np.nan,
        }
        if is_num:
            num = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            ok = np.isfinite(num)
            if ok.any() and y.to_numpy()[ok].sum() > 0 and (y.to_numpy()[ok] == 0).sum() > 0:
                xv = num[ok]
                yv = y.to_numpy()[ok]
                auc = _univariate_auc(xv, yv)
                pos_x, neg_x = xv[yv == 1], xv[yv == 0]
                ks = ks_2samp(pos_x, neg_x)
                mw = mannwhitneyu(pos_x, neg_x, alternative="two-sided")
                sp_r, _ = spearmanr(xv, yv)
                rec.update(
                    auc=float(auc),
                    auc_strength=float(abs(auc - 0.5) * 2.0),
                    ks_stat=float(ks.statistic),
                    spearman_r=float(sp_r) if np.isfinite(sp_r) else np.nan,
                    p_value=float(mw.pvalue),
                )
        else:
            sub = pd.DataFrame({"x": s.astype("string").fillna("__MISSING__"), "y": y})
            if sub["x"].nunique() >= 2:
                ct = pd.crosstab(sub["x"], sub["y"]).to_numpy()
                v, _, pval = _cramers_v_corrected(ct)
                rec.update(cramers_v=v, p_value=pval)
        rows.append(rec)

    out = pd.DataFrame(rows)
    return out.sort_values("iv", ascending=False).reset_index(drop=True) if not out.empty else out
