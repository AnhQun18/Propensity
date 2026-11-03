"""Frequencies and rare-category counts, with missing kept separate."""
import pandas as pd
from ..utils.stats import categorical_summary, tokens


def frequencies(frame, columns, top=20):
    return categorical_summary(frame, columns, top)


def cardinality(frame, columns, rare_threshold=.01):
    if not 0 <= rare_threshold <= 1:
        raise ValueError("rare_threshold must be between 0 and 1")
    rows = []
    for column in columns:
        s = tokens(frame[column])
        counts = s[s != "missing:"].value_counts()
        proportions = counts / counts.sum() if counts.sum() else counts
        rare = counts[proportions < rare_threshold]
        rows.append(dict(column=column, cardinality=len(counts), observed_n=int(counts.sum()),
                         rare_categories=len(rare), rare_rows=int(rare.sum()),
                         rare_threshold=rare_threshold))
    return pd.DataFrame(rows)
