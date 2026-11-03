"""Diagnostics only: no imputation, dropping or clipping."""
import numpy as np
import pandas as pd
from ..utils.stats import missing_mask, tokens
from .numerical import summarize


def missing(frame):
    return pd.DataFrame([dict(column=c, missing_n=int(missing_mask(frame[c]).sum()),
                              missing_pct=missing_mask(frame[c]).mean() * 100) for c in frame])


def duplicates(frame, keys=None):
    columns = list(frame.columns) if keys is None else list(keys)
    if not columns:
        return pd.DataFrame([dict(key="all_columns", duplicate_rows=0, rows=len(frame))])
    encoded = frame[columns].apply(tokens)
    return pd.DataFrame([dict(key=", ".join(columns), duplicate_rows=int(encoded.duplicated().sum()),
                              rows=len(frame))])


def outliers(frame, columns):
    result = summarize(frame, columns)
    return result[["column", "finite_n", "iqr_lower", "iqr_upper", "outlier_n"]] if len(result) else result


def anomalies(frame, columns, valid_ranges=None):
    rows = []
    for column in dict.fromkeys(list(columns) + list(valid_ranges or {})):
        s = frame[column]
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError(f"Numeric anomaly/range checks require numeric dtype: {column}")
        values = s.to_numpy(dtype=float, na_value=np.nan)
        low, high = (valid_ranges or {}).get(column, [None, None])
        outside = np.zeros(len(s), dtype=bool)
        if low is not None:
            outside |= values < low
        if high is not None:
            outside |= values > high
        rows.append(dict(column=column, infinite_n=int(np.isinf(values).sum()),
                         outside_range_n=int(outside.sum()), allowed_min=low, allowed_max=high))
    return pd.DataFrame(rows)
