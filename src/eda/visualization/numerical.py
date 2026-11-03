"""Numerical distribution plots without modifying raw values."""
import numpy as np
from . import axes


def histogram(series, bins=30, ax=None):
    fig, ax = axes(ax)
    values = series.dropna().to_numpy(dtype=float)
    ax.hist(values[np.isfinite(values)], bins=bins, color="#245D80")
    ax.set(xlabel=str(series.name), ylabel="Observations", title="Finite values; no outlier clipping")
    return fig


def boxplot(series, ax=None):
    fig, ax = axes(ax)
    values = series.dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values):
        ax.boxplot(values, orientation="horizontal")
    ax.set(xlabel=str(series.name), title="Boxplot: 1.5 x IQR whiskers")
    return fig
