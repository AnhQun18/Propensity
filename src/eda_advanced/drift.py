"""Distribution drift & shift diagnostics: PSI, KS-test, Wasserstein, and Jensen-Shannon divergence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

from ._compat import load_dataframe


def _psi_from_probs(p_ref: np.ndarray, p_cur: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p_ref, eps, 1.0)
    q = np.clip(p_cur, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def _drift_band(psi: float) -> str:
    if not np.isfinite(psi):
        return "unknown"
    if psi < 0.10:
        return "low"
    if psi < 0.25:
        return "moderate"
    return "high"


def population_stability_index(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
    is_categorical: bool = False,
    eps: float = 1e-6,
) -> float:
    """Compute Population Stability Index (PSI) between reference and current distributions."""
    ref_s = pd.Series(reference)
    cur_s = pd.Series(current)
    if is_categorical:
        r_vc = ref_s.astype("string").fillna("__MISSING__").value_counts(normalize=True)
        c_vc = cur_s.astype("string").fillna("__MISSING__").value_counts(normalize=True)
        keys = sorted(set(r_vc.index) | set(c_vc.index))
        p_ref = np.array([r_vc.get(k, 0.0) for k in keys], dtype=float)
        p_cur = np.array([c_vc.get(k, 0.0) for k in keys], dtype=float)
        return _psi_from_probs(p_ref, p_cur, eps=eps)

    r_arr = pd.to_numeric(ref_s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    c_arr = pd.to_numeric(cur_s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    r_clean = r_arr[np.isfinite(r_arr)]
    c_clean = c_arr[np.isfinite(c_arr)]
    if len(r_clean) == 0 or len(c_clean) == 0:
        return np.nan
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.unique(np.quantile(r_clean, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    r_hist, _ = np.histogram(r_clean, bins=edges)
    c_hist, _ = np.histogram(c_clean, bins=edges)
    return _psi_from_probs(r_hist / len(r_clean), c_hist / len(c_clean), eps=eps)


def compare_distributions(
    reference_source: Any,
    current_source: Any,
    *,
    columns: Sequence[str] | None = None,
    n_bins: int = 10,
    **source_options: Any,
) -> pd.DataFrame:
    """Compare column distributions between a reference and current dataset (PSI, KS, Wasserstein, JS)."""
    ref_df = load_dataframe(reference_source, columns=columns, **source_options)
    cur_df = load_dataframe(current_source, columns=columns, **source_options)
    common_cols = list(columns) if columns is not None else [c for c in ref_df.columns if c in cur_df.columns]

    rows: list[dict[str, Any]] = []
    for col in common_cols:
        r_s, c_s = ref_df[col], cur_df[col]
        first_valid = r_s.dropna().iloc[0] if r_s.notna().any() else None
        if isinstance(first_valid, (list, tuple, np.ndarray, dict)):
            continue
        r_miss = float(r_s.isna().mean() * 100.0) if len(r_s) else 0.0
        c_miss = float(c_s.isna().mean() * 100.0) if len(c_s) else 0.0
        is_num = pd.api.types.is_numeric_dtype(r_s) and not pd.api.types.is_bool_dtype(r_s)

        if is_num:
            r_arr = pd.to_numeric(r_s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            c_arr = pd.to_numeric(c_s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            rc = r_arr[np.isfinite(r_arr)]
            cc = c_arr[np.isfinite(c_arr)]
            psi = population_stability_index(rc, cc, n_bins=n_bins, is_categorical=False)
            if len(rc) and len(cc):
                ks = ks_2samp(rc, cc)
                wass = float(wasserstein_distance(rc, cc))
                r_mean, c_mean = float(rc.mean()), float(cc.mean())
                r_std = float(rc.std(ddof=1)) if len(rc) > 1 else np.nan
                shift_std = (c_mean - r_mean) / r_std if (np.isfinite(r_std) and r_std > 0) else np.nan
                ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
            else:
                ks_stat = ks_p = wass = r_mean = c_mean = shift_std = np.nan
            rows.append({
                "column": col,
                "kind": "numeric",
                "psi": psi,
                "drift_level": _drift_band(psi),
                "ks_stat": ks_stat,
                "p_value": ks_p,
                "wasserstein": wass,
                "js_divergence": np.nan,
                "mean_ref": r_mean,
                "mean_cur": c_mean,
                "mean_shift_std": shift_std,
                "new_categories_n": 0,
                "missing_ref_pct": r_miss,
                "missing_cur_pct": c_miss,
                "missing_diff_pct": c_miss - r_miss,
            })
        else:
            rs = r_s.astype("string").fillna("__MISSING__")
            cs = c_s.astype("string").fillna("__MISSING__")
            r_vc = rs.value_counts()
            c_vc = cs.value_counts()
            keys = sorted(set(r_vc.index) | set(c_vc.index))
            r_cnt = np.array([r_vc.get(k, 0) for k in keys], dtype=float)
            c_cnt = np.array([c_vc.get(k, 0) for k in keys], dtype=float)
            p_r = r_cnt / r_cnt.sum() if r_cnt.sum() > 0 else np.zeros_like(r_cnt)
            p_c = c_cnt / c_cnt.sum() if c_cnt.sum() > 0 else np.zeros_like(c_cnt)
            psi = _psi_from_probs(p_r, p_c)
            js_div = float(jensenshannon(p_r, p_c) ** 2) if (r_cnt.sum() > 0 and c_cnt.sum() > 0) else np.nan
            pval = float(chi2_contingency(np.vstack([r_cnt, c_cnt]))[1]) if (len(keys) >= 2 and r_cnt.sum() > 0 and c_cnt.sum() > 0) else np.nan
            new_cats = len(set(c_vc.index) - set(r_vc.index))
            rows.append({
                "column": col,
                "kind": "categorical",
                "psi": psi,
                "drift_level": _drift_band(psi),
                "ks_stat": np.nan,
                "p_value": pval,
                "wasserstein": np.nan,
                "js_divergence": js_div,
                "mean_ref": np.nan,
                "mean_cur": np.nan,
                "mean_shift_std": np.nan,
                "new_categories_n": int(new_cats),
                "missing_ref_pct": r_miss,
                "missing_cur_pct": c_miss,
                "missing_diff_pct": c_miss - r_miss,
            })

    out = pd.DataFrame(rows)
    return out.sort_values("psi", ascending=False).reset_index(drop=True) if not out.empty else out


def temporal_drift_report(
    source: Any,
    *,
    period_col: str,
    reference_period: Any = None,
    columns: Sequence[str] | None = None,
    n_bins: int = 10,
    **source_options: Any,
) -> pd.DataFrame:
    """Track PSI/KS/JS drift across every period relative to `reference_period` (defaults to earliest)."""
    df = load_dataframe(source, **source_options)
    if period_col not in df.columns:
        raise ValueError(f"period_col {period_col!r} not found")
    periods = sorted(df[period_col].dropna().unique().tolist())
    if len(periods) < 2:
        raise ValueError("Need at least 2 distinct periods to compute temporal drift")
    ref_p = periods[0] if reference_period is None else reference_period
    ref_df = df[df[period_col] == ref_p]
    cols = list(columns) if columns is not None else [c for c in df.columns if c != period_col]

    frames = []
    for per in periods:
        if per == ref_p:
            continue
        cur_df = df[df[period_col] == per]
        part = compare_distributions(ref_df, cur_df, columns=cols, n_bins=n_bins)
        part.insert(0, "period", str(per))
        part.insert(0, "reference_period", str(ref_p))
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
