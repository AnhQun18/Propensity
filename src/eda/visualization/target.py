"""Target counts and group rates, optionally with Wilson intervals."""
import numpy as np
from . import axes


def distribution(table, ax=None):
    fig, ax = axes(ax)
    ax.bar(table.value.astype(str), table.n, color="#468477")
    ax.set(ylabel="Observations", xlabel="Label (missing kept separate)")
    return fig


def rates(table, ax=None):
    fig, ax = axes(ax)
    errors = None
    if {"wilson_low", "wilson_high"}.issubset(table):
        errors = np.maximum(0, np.array([table.rate - table.wilson_low, table.wilson_high - table.rate]))
    ax.bar(table.group.astype(str), table.rate, yerr=errors, color="#468477")
    ax.set(ylabel="Positive rate", ylim=(0, 1))
    ax.tick_params(axis="x", labelrotation=45)
    return fig
