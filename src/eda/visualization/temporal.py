"""Activity counts on an explicitly configured time grid."""
import pandas as pd
from . import axes


def activity(table, ax=None):
    fig, ax = axes(ax)
    ax.plot(pd.to_datetime(table.period), table.events, color="#245D80")
    ax.set(xlabel="Period", ylabel="Events", title="Recorded activity over time")
    fig.autofmt_xdate()
    return fig
