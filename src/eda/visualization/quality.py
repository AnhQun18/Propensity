"""Missing percentages and bounded missing-pattern previews."""
from . import axes
from ..utils.stats import missing_mask


def missing_bar(table, ax=None):
    fig, ax = axes(ax)
    ax.barh(table.column.astype(str), table.missing_pct, color="#C76A37")
    ax.set(xlabel="Missing (%)", xlim=(0, 100))
    ax.invert_yaxis()
    return fig


def missing_pattern(frame, max_rows=300, seed=42, ax=None):
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    fig, ax = axes(ax, (9, 5))
    sample = frame.sample(min(len(frame), max_rows), random_state=seed)
    if len(sample) and len(sample.columns):
        mask = sample.apply(missing_mask).to_numpy(dtype=int)
        ax.imshow(mask, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="Greys")
        ax.set_xticks(range(len(sample.columns)), sample.columns, rotation=90)
    ax.set(ylabel="Sample rows", title=f"Missing patterns: n={len(sample)}, seed={seed}")
    return fig
