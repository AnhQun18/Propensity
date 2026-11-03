"""Bivariate plots; scatter sampling is explicit and reproducible."""
import numpy as np
import pandas as pd
from . import axes
from ..utils.stats import tokens


def scatter(frame, x, y, max_points=4000, seed=42, ax=None):
    if max_points < 1:
        raise ValueError("max_points must be positive")
    fig, ax = axes(ax)
    selected = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(selected) > max_points:
        selected = selected.sample(max_points, random_state=seed)
    ax.scatter(selected[x], selected[y], s=8, alpha=.4)
    ax.set(xlabel=x, ylabel=y, title=f"Scatter: {len(selected):,} points, seed={seed}")
    return fig


def grouped_boxplot(frame, group, numeric, top=10, ax=None):
    fig, ax = axes(ax)
    keys = tokens(frame[group])
    labels = keys.value_counts().head(top).index
    values = [pd.to_numeric(frame.loc[keys == key, numeric]).replace([np.inf, -np.inf], np.nan).dropna() for key in labels]
    if len(values):
        ax.boxplot(values, tick_labels=list(labels))
    ax.set(xlabel=group, ylabel=numeric)
    ax.tick_params(axis="x", labelrotation=45)
    return fig
