"""Multi-horizon panel targets, stratified 1D/2D lift tables, buyer composition, transitions, and surge ratio."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .inference import wilson_ci


def build_multi_horizon_targets(
    panel_df: pd.DataFrame,
    *,
    entity_col: str,
    period_col: str,
    target_col: str,
    horizons: Sequence[int] = (1, 2, 3),
    common_cohort: bool = True,
) -> pd.DataFrame:
    """Construct cumulative binary horizon targets h1..hk (`OR` over t..t+h-1) per entity."""
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("horizons must contain positive integers")
    max_h = max(horizons)
    df = panel_df.sort_values([entity_col, period_col]).copy()
    g = df.groupby(entity_col, sort=False)[target_col]

    lead_cols = []
    for step in range(max_h):
        col = f"__lead_{step}"
        df[col] = df[target_col] if step == 0 else g.shift(-step)
        lead_cols.append(col)

    if common_cohort:
        df = df.loc[df[lead_cols].notna().all(axis=1)].copy()

    for h in sorted(set(horizons)):
        subset = df[lead_cols[:h]]
        any_pos = subset.eq(1).any(axis=1)
        all_zero = subset.eq(0).all(axis=1)
        df[f"h{h}"] = np.where(any_pos, 1.0, np.where(all_zero, 0.0, np.nan))
        if common_cohort:
            df[f"h{h}"] = df[f"h{h}"].astype("int8")

    return df.drop(columns=lead_cols).reset_index(drop=True)


def stratified_lift_table(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] | str,
    target_cols: Sequence[str],
    entity_col: str | None = None,
) -> pd.DataFrame:
    """Compute exhaustive group shares, positive counts, rates, and lifts vs population baseline."""
    keys = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    targets = list(target_cols)
    n_total = len(df)
    baselines = {t: float(df[t].mean()) if df[t].notna().any() else np.nan for t in targets}

    agg_spec: dict[str, tuple[str, str]] = {"n": (targets[0], "size")}
    if entity_col and entity_col in df.columns:
        agg_spec["entities"] = (entity_col, "nunique")
    for t in targets:
        agg_spec[f"positives_{t}"] = (t, "sum")
        agg_spec[f"labelled_{t}"] = (t, "count")

    res = df.groupby(keys, dropna=False, observed=True).agg(**agg_spec).reset_index()
    assert int(res["n"].sum()) == n_total
    res["share_pct"] = res["n"] / n_total * 100.0 if n_total else 0.0
    for t in targets:
        res[f"rate_{t}"] = res[f"positives_{t}"] / res[f"labelled_{t}"].replace(0, np.nan)
        base = baselines[t]
        res[f"lift_{t}"] = res[f"rate_{t}"] / base if (np.isfinite(base) and base > 0) else np.nan
    return res


def buyer_composition_and_transitions(
    outcome_matrix: np.ndarray,
    periods: Sequence[Any],
    *,
    alpha: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """Decompose positive entities into returning/first_observed/history_unknown + Markov transitions."""
    mat = np.asarray(outcome_matrix, dtype=float)
    n_ent, n_per = mat.shape
    if len(periods) != n_per:
        raise ValueError("periods length must match outcome_matrix columns")

    comp_rows, trans_rows = [], []
    for j in range(n_per):
        p_label = str(periods[j])
        buyer = mat[:, j] == 1
        k = int(buyer.sum())
        prior_pos = (mat[:, :j] == 1).any(axis=1) if j > 0 else np.zeros(n_ent, dtype=bool)
        cats = {
            "returning": buyer & prior_pos,
            "first_observed": (buyer & ~prior_pos) if j > 0 else np.zeros(n_ent, dtype=bool),
            "history_unknown": buyer if j == 0 else np.zeros(n_ent, dtype=bool),
        }
        assert sum(int(m.sum()) for m in cats.values()) == k
        for cat_name, mask in cats.items():
            cnt = int(mask.sum())
            lo, hi = wilson_ci(cnt, k, alpha=alpha)
            comp_rows.append({
                "period": p_label,
                "buyer_group": cat_name,
                "positives": cnt,
                "all_positives": k,
                "share": cnt / k if k else np.nan,
                "ci_low": float(lo),
                "ci_high": float(hi),
            })
        if j > 0:
            for prev_state in (0, 1):
                mask = (mat[:, j - 1] == prev_state) & np.isfinite(mat[:, j])
                nk = int(mask.sum())
                pk = int((mat[mask, j] == 1).sum())
                lo, hi = wilson_ci(pk, nk, alpha=alpha)
                trans_rows.append({
                    "period": p_label,
                    "previous_state": prev_state,
                    "n_total": nk,
                    "n_positive": pk,
                    "rate": pk / nk if nk else np.nan,
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                })

    return {"composition": pd.DataFrame(comp_rows), "transitions": pd.DataFrame(trans_rows)}


def compute_surge_ratio(
    panel_df: pd.DataFrame,
    *,
    entity_col: str,
    period_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Compute leak-free lag-1 value, prior active-period mean (up to T-2), and `surge_ratio`."""
    df = panel_df.sort_values([entity_col, period_col]).copy()
    g = df.groupby(entity_col, sort=False)
    filled = df[value_col].fillna(0.0)
    prev = g[value_col].shift(1).fillna(0.0)
    cum_before_t = g[value_col].transform(lambda s: s.fillna(0.0).cumsum()) - filled
    active_before_t = g[value_col].transform(lambda s: (s.fillna(0.0) > 0).cumsum()) - (filled > 0).astype(int)

    cum_before_prev = cum_before_t - prev
    active_before_prev = active_before_t - (prev > 0).astype(int)
    prior_mean = np.where(active_before_prev > 0, cum_before_prev / active_before_prev, np.nan)
    surge = np.where(prior_mean > 0, prev / prior_mean, np.nan)

    df[f"{value_col}_prev"] = prev
    df[f"{value_col}_prior_active_mean"] = prior_mean
    df[f"{value_col}_surge_ratio"] = surge
    return df.reset_index(drop=True)
