"""Longitudinal event studies, personal baseline deltas, and fixed pre-window strata control."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def run_event_study(
    entity_ids: Any,
    periods: Sequence[Any],
    metric_matrix: np.ndarray,
    outcome_matrix: np.ndarray,
    *,
    lags: Sequence[int] = (-3, -2, -1, 0, 1, 2, 3),
    min_baseline_periods: int = 3,
    baseline_gap: int | None = None,
) -> pd.DataFrame:
    """Measure metric trajectory around binary outcome anchors with personal pre-window baseline deltas."""
    a = np.asarray(metric_matrix, dtype=float)
    y = np.asarray(outcome_matrix, dtype=float)
    n_ent, n_per = a.shape
    if y.shape != a.shape or len(entity_ids) != n_ent or len(periods) != n_per:
        raise ValueError("Shape mismatch among entity_ids, periods, metric_matrix, and outcome_matrix")

    gap = abs(min(lags)) if baseline_gap is None else int(baseline_gap)
    anchors = np.where(np.isfinite(y).any(axis=0))[0]
    rows: list[dict[str, Any]] = []

    for t in anchors:
        hist_end = max(0, int(t) - gap)
        hist = a[:, :hist_end]
        if hist.shape[1] > 0:
            valid_counts = np.isfinite(hist).sum(axis=1)
            eligible = valid_counts >= int(min_baseline_periods)
            baseline = np.full(n_ent, np.nan)
            if eligible.any():
                baseline[eligible] = np.nanmedian(hist[eligible], axis=1)
            seen = np.nansum(hist, axis=1) > 0
        else:
            eligible = np.zeros(n_ent, dtype=bool)
            baseline = np.full(n_ent, np.nan)
            seen = np.zeros(n_ent, dtype=bool)

        for lag in lags:
            j = int(t) + int(lag)
            v = a[:, j] if 0 <= j < n_per else np.full(n_ent, np.nan)
            for outcome in (0, 1):
                mask = y[:, t] == outcome
                good = mask & np.isfinite(v)
                bg = good & eligible
                delta = v[bg] - baseline[bg]
                tol = 1e-6 * np.maximum(1.0, np.abs(baseline[bg]))
                rows.append({
                    "anchor": str(periods[t]),
                    "lag": int(lag),
                    "outcome": outcome,
                    "population": int(mask.sum()),
                    "observed": int(good.sum()),
                    "unavailable": int((mask & ~np.isfinite(v)).sum()),
                    "mean": float(v[good].mean()) if good.any() else np.nan,
                    "median": float(np.median(v[good])) if good.any() else np.nan,
                    "baseline_eligible": int(bg.sum()),
                    "baseline_insufficient": int((mask & ~eligible).sum()),
                    "baseline_unseen": int((mask & eligible & ~seen).sum()),
                    "baseline_zero": int((bg & (baseline == 0)).sum()),
                    "delta_mean": float(delta.mean()) if len(delta) else np.nan,
                    "increase": float((delta > tol).mean()) if len(delta) else np.nan,
                    "decrease": float((delta < -tol).mean()) if len(delta) else np.nan,
                    "unchanged": float((np.abs(delta) <= tol).mean()) if len(delta) else np.nan,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        assert (df["observed"] + df["unavailable"] == df["population"]).all()
    return df


def run_stratified_event_study(
    entity_ids: Any,
    periods: Sequence[Any],
    metric_matrix: np.ndarray,
    strata_source_matrix: np.ndarray,
    outcome_matrix: np.ndarray,
    *,
    lags: Sequence[int] = (-3, -2, -1, 0, 1, 2, 3),
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Event study stratified by fixed pre-window quantiles to control for baseline activity confounding."""
    a = np.asarray(metric_matrix, dtype=float)
    s_mat = np.asarray(strata_source_matrix, dtype=float)
    y = np.asarray(outcome_matrix, dtype=float)
    n_ent, n_per = a.shape
    gap = abs(min(lags))
    anchors = np.where(np.isfinite(y).any(axis=0))[0]
    q_probs = np.linspace(0.0, 1.0, int(n_quantiles) + 1)[1:-1]
    rows: list[dict[str, Any]] = []

    for t in anchors:
        if t < gap:
            continue
        hist = s_mat[:, : t - gap]
        if not np.isfinite(hist).any():
            continue
        counts = np.nansum(hist, axis=1)
        pos = counts > 0
        edges = np.unique(np.quantile(counts[pos], q_probs)) if (pos.any() and len(q_probs)) else np.array([])
        strata = np.where(pos, np.searchsorted(edges, counts, side="left") + 1, 0)
        for lag in lags:
            j = int(t) + int(lag)
            v = a[:, j] if 0 <= j < n_per else np.full(n_ent, np.nan)
            for outcome in (0, 1):
                for st in range(int(strata.max()) + 1):
                    mask = (y[:, t] == outcome) & (strata == st)
                    good = mask & np.isfinite(v)
                    rows.append({
                        "anchor": str(periods[t]),
                        "lag": int(lag),
                        "outcome": outcome,
                        "stratum": int(st),
                        "population": int(mask.sum()),
                        "observed": int(good.sum()),
                        "unavailable": int((mask & ~np.isfinite(v)).sum()),
                        "mean": float(v[good].mean()) if good.any() else np.nan,
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        assert (df["observed"] + df["unavailable"] == df["population"]).all()
    return df
