"""Statistical inference for EDA: Wilson CI, entity-cluster bootstrap CI, and two-group hypothesis tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, norm, ttest_ind


def wilson_ci(k: Any, n: Any, *, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Wilson score confidence interval for binomial proportions."""
    k_arr = np.asarray(k, dtype=float)
    n_arr = np.asarray(n, dtype=float)
    denom_safe = np.maximum(n_arr, 1.0)
    p = np.divide(k_arr, denom_safe, out=np.zeros_like(k_arr), where=n_arr > 0)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    d = 1.0 + z2 / denom_safe
    center = (p + z2 / (2.0 * denom_safe)) / d
    half = z * np.sqrt(p * (1.0 - p) / denom_safe + z2 / (4.0 * denom_safe * denom_safe)) / d
    lo = np.where(n_arr > 0, np.maximum(0.0, center - half), np.nan)
    hi = np.where(n_arr > 0, np.minimum(1.0, center + half), np.nan)
    return lo, hi


def cluster_bootstrap_rates(
    entity_ids: Any,
    group_masks: Mapping[str, Any],
    y: Any,
    *,
    n_reps: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Cluster-bootstrap positive rates and lifts resampling whole entities (retaining all periods)."""
    if not group_masks:
        raise ValueError("group_masks must contain at least one group")
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(y_arr)
    ent_arr = np.asarray(entity_ids)[valid]
    y_clean = y_arr[valid]

    unique_ids, ent_idx = np.unique(ent_arr, return_inverse=True)
    n_entities = len(unique_ids)
    total_obs_per_ent = np.bincount(ent_idx, minlength=n_entities).astype(float)
    total_pos_per_ent = np.bincount(ent_idx, weights=y_clean, minlength=n_entities).astype(float)

    total_pos = float(y_clean.sum())
    baseline = float(y_clean.mean()) if len(y_clean) else np.nan

    group_names = list(group_masks.keys())
    den_cols, num_cols, records = [], [], []
    for name in group_names:
        mask = np.asarray(group_masks[name], dtype=bool)[valid]
        dn = np.bincount(ent_idx, weights=mask.astype(float), minlength=n_entities)
        nm = np.bincount(ent_idx, weights=(mask * y_clean), minlength=n_entities)
        den_cols.append(dn)
        num_cols.append(nm)
        n_obs = int(dn.sum())
        pos = int(nm.sum())
        rate = pos / n_obs if n_obs else np.nan
        records.append({
            "group": name,
            "n_obs": n_obs,
            "positives": pos,
            "rate": rate,
            "positive_share": pos / total_pos if total_pos else np.nan,
            "lift": rate / baseline if (n_obs and baseline > 0) else np.nan,
        })

    den_mat = np.stack(den_cols, axis=1)
    num_mat = np.stack(num_cols, axis=1)
    rng = np.random.default_rng(seed)

    boot_rates = np.empty((n_reps, len(group_names)), dtype=float)
    boot_lifts = np.empty((n_reps, len(group_names)), dtype=float)
    for b in range(n_reps):
        sample_idx = rng.integers(0, n_entities, size=n_entities)
        dn = den_mat[sample_idx].sum(axis=0)
        nm = num_mat[sample_idx].sum(axis=0)
        rate = np.divide(nm, dn, out=np.full_like(nm, np.nan), where=dn > 0)
        base_den = total_obs_per_ent[sample_idx].sum()
        base_num = total_pos_per_ent[sample_idx].sum()
        base_rate = base_num / base_den if base_den > 0 else np.nan
        boot_rates[b] = rate
        boot_lifts[b] = rate / base_rate if (np.isfinite(base_rate) and base_rate > 0) else np.nan

    q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0
    out = pd.DataFrame(records)
    out["ci_low"] = np.nanquantile(boot_rates, q_lo, axis=0)
    out["ci_high"] = np.nanquantile(boot_rates, q_hi, axis=0)
    out["lift_ci_low"] = np.nanquantile(boot_lifts, q_lo, axis=0)
    out["lift_ci_high"] = np.nanquantile(boot_lifts, q_hi, axis=0)
    return out


def compare_two_groups(group_a: Any, group_b: Any) -> dict[str, Any]:
    """Compare two numeric samples via Mann-Whitney U, Welch t-test, KS-test, and effect sizes."""
    a = pd.to_numeric(pd.Series(group_a), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    b = pd.to_numeric(pd.Series(group_b), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Both groups must have at least one finite value")

    mw = mannwhitneyu(a, b, alternative="two-sided")
    tt = ttest_ind(a, b, equal_var=False)
    ks = ks_2samp(a, b)

    # Effect sizes: Cohen's d (pooled SD) & rank-biserial correlation
    s_pool = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / max(len(a) + len(b) - 2, 1)) if (len(a) > 1 and len(b) > 1) else np.nan
    cohens_d = float((a.mean() - b.mean()) / s_pool) if (np.isfinite(s_pool) and s_pool > 0) else np.nan
    rank_biserial = float(1.0 - (2.0 * mw.statistic) / (len(a) * len(b)))

    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(a.mean() - b.mean()),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "median_diff": float(np.median(a) - np.median(b)),
        "cohens_d": cohens_d,
        "rank_biserial": rank_biserial,
        "mannwhitney_u": float(mw.statistic),
        "mannwhitney_p": float(mw.pvalue),
        "welch_t": float(tt.statistic),
        "welch_p": float(tt.pvalue),
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
    }
