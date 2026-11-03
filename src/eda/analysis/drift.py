"""Descriptive distribution comparisons, preserving missing information."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from ..utils.stats import tokens


def compare(reference, current, numeric=(), categorical=()):
    rows = []
    for column in list(numeric) + list(categorical):
        if column not in reference or column not in current:
            raise ValueError(f"Drift column missing from one input: {column}")
        a, b = reference[column], current[column]
        base = dict(column=column, reference_n=len(a), current_n=len(b),
                    reference_missing_pct=a.isna().mean() * 100,
                    current_missing_pct=b.isna().mean() * 100)
        if column in numeric:
            x = a.replace([np.inf, -np.inf], np.nan).dropna()
            y = b.replace([np.inf, -np.inf], np.nan).dropna()
            score = ks_2samp(x, y).statistic if len(x) and len(y) else np.nan
            rows.append(dict(**base, metric="ks_statistic", value=score))
        else:
            x, y = tokens(a).value_counts(normalize=True), tokens(b).value_counts(normalize=True)
            keys = x.index.union(y.index)
            score = (x.reindex(keys, fill_value=0) - y.reindex(keys, fill_value=0)).abs().sum() / 2
            rows.append(dict(**base, metric="total_variation", value=score if len(a) and len(b) else np.nan))
    return pd.DataFrame(rows)
