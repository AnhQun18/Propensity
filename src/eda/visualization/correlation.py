"""Correlation heatmaps from precomputed long pair tables."""
import numpy as np
from . import axes


def heatmap(table, ax=None):
    columns = list(dict.fromkeys(list(table.column_a) + list(table.column_b)))
    fig, ax = axes(ax, (8, 6))
    matrix = np.full((len(columns), len(columns)), np.nan)
    positions = {c: i for i, c in enumerate(columns)}
    for row in table.itertuples():
        a, b = positions[row.column_a], positions[row.column_b]
        matrix[a, b] = matrix[b, a] = row.spearman
    if columns:
        np.fill_diagonal(matrix, 1)
        chart = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        fig.colorbar(chart, ax=ax)
        ax.set_xticks(range(len(columns)), columns, rotation=90)
        ax.set_yticks(range(len(columns)), columns)
    ax.set(title="Spearman correlation (pairwise complete)")
    return fig
