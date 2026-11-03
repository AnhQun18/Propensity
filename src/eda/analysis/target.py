"""Label summaries, optional independent-binomial intervals and leakage flags."""
import numpy as np
import pandas as pd
from ..utils.stats import target_summary, grouped_summary, tokens


def wilson(positive, n, z=1.959963984540054):
    if n < 0 or positive < 0 or positive > n:
        raise ValueError("Expected 0 <= positive <= n")
    if not n:
        return np.nan, np.nan
    p = positive / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def summarize(frame, targets, independent=False):
    table = target_summary(frame, targets)
    if independent and not table.empty:
        lows, highs = [], []
        for row in table.itertuples():
            low, high = wilson(row.n, row.labelled_n) if row.value != "missing:" else (np.nan, np.nan)
            lows.append(low)
            highs.append(high)
        table["wilson_low"] = lows
        table["wilson_high"] = highs
    return table


def grouped_rates(frame, groups, targets, independent=False):
    table = grouped_summary(frame, groups, [], targets)[1]
    if independent and not table.empty:
        intervals = [wilson(row.positive_n, row.labelled_n) for row in table.itertuples()]
        table["wilson_low"] = [x[0] for x in intervals]
        table["wilson_high"] = [x[1] for x in intervals]
    return table


def leakage_flags(frame, targets, candidates):
    """Flag exact copies of labels; these are review signals, not proof of leakage."""
    rows = []
    for target in targets:
        valid = frame[target].notna()
        if not valid.any():
            continue
        y = tokens(frame.loc[valid, target])
        for column in candidates:
            if column == target:
                continue
            same = tokens(frame.loc[valid, column]).eq(y).all()
            if same:
                rows.append(dict(target=target, column=column, flag="exact_label_copy",
                                 review="Check availability at prediction time; not proof of leakage"))
    return pd.DataFrame(rows, columns=["target", "column", "flag", "review"])
