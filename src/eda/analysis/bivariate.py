"""Numeric/numeric, categorical/numeric and categorical/categorical summaries."""
import pandas as pd
from ..utils.stats import grouped_summary, tokens
from .correlation import spearman


def numeric_numeric(frame, columns):
    return spearman(frame, columns)


def categorical_numeric(frame, groups, numeric):
    return grouped_summary(frame, groups, numeric, {})[0]


def categorical_categorical(frame, left, right):
    table = pd.crosstab(tokens(frame[left]), tokens(frame[right]), dropna=False)
    table.index.name = left
    table.columns.name = right
    return table
