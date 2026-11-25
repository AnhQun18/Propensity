"""Exact 6-way mutually exclusive period-over-period shift-share decomposition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def period_change_decomposition(
    metric_matrix: np.ndarray,
    periods: Sequence[Any],
    *,
    activity_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    """Decompose total period-over-period change into 6 exhaustive, mutually exclusive entity cohorts."""
    a = np.asarray(metric_matrix, dtype=float)
    act = a if activity_matrix is None else np.asarray(activity_matrix, dtype=float)
    if a.shape != act.shape or len(periods) != a.shape[1]:
        raise ValueError("Shape mismatch between metric_matrix, activity_matrix, and periods")

    n_ent, n_per = a.shape
    seen_cum = np.cumsum(np.nan_to_num(act, nan=0.0) > 0, axis=1)
    rows: list[dict[str, Any]] = []

    for j in range(1, n_per):
        valid = np.isfinite(a[:, j - 1]) & np.isfinite(a[:, j])
        prev_on = np.nan_to_num(act[:, j - 1], nan=0.0) > 0
        curr_on = np.nan_to_num(act[:, j], nan=0.0) > 0
        ever_before = seen_cum[:, j - 1] > 0

        groups = {
            "continuing": valid & prev_on & curr_on,
            "first_seen": valid & ~prev_on & curr_on & ~ever_before,
            "reactivated": valid & ~prev_on & curr_on & ever_before,
            "dropped": valid & prev_on & ~curr_on,
            "inactive_both": valid & ~prev_on & ~curr_on,
            "unavailable": ~valid,
        }
        membership = np.vstack(list(groups.values())).sum(axis=0)
        assert membership.min() == 1 and membership.max() == 1

        comparable_total = float((a[valid, j] - a[valid, j - 1]).sum()) if valid.any() else np.nan
        for grp, mask in groups.items():
            contrib = (
                float((a[mask, j] - a[mask, j - 1]).sum())
                if (grp != "unavailable" and valid.any())
                else np.nan
            )
            rows.append({
                "period": str(periods[j]),
                "group": grp,
                "entities": int(mask.sum()),
                "contribution": contrib,
                "comparable_total": comparable_total,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        assert df.groupby("period")["entities"].sum().eq(n_ent).all()
        reconcile = df.groupby("period").agg(
            total=("contribution", lambda s: s.sum(min_count=1)),
            expected=("comparable_total", "first"),
        )
        assert np.allclose(reconcile["total"], reconcile["expected"], equal_nan=True)
    return df
