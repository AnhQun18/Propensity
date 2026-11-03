"""Finite-value numerical summaries, including sample kurtosis."""
import numpy as np
from ..utils.stats import numeric_summary


def summarize(frame, columns):
    result = numeric_summary(frame, columns)
    if not result.empty:
        result["kurtosis"] = [frame[c].replace([np.inf, -np.inf], np.nan).kurt() for c in columns]
    return result
