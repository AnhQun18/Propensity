"""Slice-based error analysis across coverage groups and prior-purchase cohorts."""
import math
from typing import Any

import numpy as np
import pandas as pd

from .metrics import compute_metrics, deterministic_rank_order, deterministic_tie_hash


def evaluate_slices(
    df: pd.DataFrame,
    budget_fractions: tuple[float, ...] = (0.005, 0.01, 0.05),
    primary_budget: float = 0.01,
    seed: int = 42,
    is_probabilistic: bool = True,
) -> list[dict[str, Any]]:
    """Compute metrics per coverage_slice and per prior_all_bought_any + global Top-1% composition."""
    y_all = df["y_true"].to_numpy(dtype=np.int8)
    s_all = df["score"].to_numpy(dtype=np.float64)
    c_all = df["client_id"].to_numpy()
    tie_hashes = deterministic_tie_hash(c_all, seed=seed)

    # Identify global Top-1% selection mask
    n_total = len(df)
    k_global = min(n_total, max(1, int(math.ceil(primary_budget * n_total)))) if n_total > 0 else 0
    order = deterministic_rank_order(s_all, c_all, seed=seed, tie_hashes=tie_hashes)
    in_top_primary = np.zeros(n_total, dtype=bool)
    if k_global > 0:
        in_top_primary[order[:k_global]] = True

    records: list[dict[str, Any]] = []
    slice_dims = [
        ("coverage_slice", df["coverage_slice"].astype(str).to_numpy()),
        (
            "prior_purchase",
            np.where(df["prior_all_bought_any"].to_numpy() == 1, "prior_buyer", "no_prior_buy"),
        ),
    ]

    for dim_name, labels in slice_dims:
        for val in sorted(pd.unique(labels)):
            mask = labels == val
            sub_metrics, _ = compute_metrics(
                y_true=y_all[mask],
                scores=s_all[mask],
                client_ids=c_all[mask],
                budget_fractions=budget_fractions,
                primary_budget=primary_budget,
                seed=seed,
                is_probabilistic=is_probabilistic,
                tie_hashes=tie_hashes[mask],
            )
            sel_mask = mask & in_top_primary
            sel_n = int(np.sum(sel_mask))
            sel_hits = int(np.sum(y_all[sel_mask]))
            records.append(
                {
                    "slice_dim": dim_name,
                    "slice_value": str(val),
                    "population_share": float(np.sum(mask) / n_total) if n_total > 0 else 0.0,
                    "global_top1pct_selected": sel_n,
                    "global_top1pct_share": float(sel_n / k_global) if k_global > 0 else 0.0,
                    "global_top1pct_hits": sel_hits,
                    "global_top1pct_precision": float(sel_hits / sel_n) if sel_n > 0 else None,
                    **sub_metrics,
                }
            )
    return records
