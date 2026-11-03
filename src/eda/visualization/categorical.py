"""Category charts consume precomputed frequency/crosstab tables."""
from . import axes


def _labels(table):
    return [str(r.value) if r.kind == "value" else f"<{r.kind}>" for r in table.itertuples()]


def bar(table, ax=None):
    fig, ax = axes(ax)
    ax.barh(_labels(table), table.n, color="#245D80")
    ax.invert_yaxis()
    ax.set(xlabel="Observations", title="Category frequencies (missing and remaining shown)")
    return fig


def pie(table, ax=None):
    fig, ax = axes(ax)
    if table.n.sum() > 0:
        ax.pie(table.n, labels=_labels(table), autopct="%1.1f%%")
    return fig


def stacked_bar(crosstab, ax=None):
    fig, ax = axes(ax)
    if not crosstab.empty:
        crosstab.plot.bar(stacked=True, ax=ax)
    ax.set(ylabel="Observations")
    return fig
