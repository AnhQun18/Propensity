"""Headless Matplotlib EDA chart helpers and 4-file figure bundle publisher (PNG+SVG+CSV+MD)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .formatting import safe_name

DEFAULT_COLORS = ("#2563eb", "#059669", "#d97706", "#9333ea", "#dc2626", "#0891b2")


def create_figure(title: str, xlabel: str = "", ylabel: str = "", *, figsize: tuple[float, float] = (9, 5)):
    """Create a Matplotlib Figure directly without modifying the global pyplot backend."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def plot_bars(frame: pd.DataFrame, x: str, y: str, title: str, *, horizontal: bool = False, color: str = DEFAULT_COLORS[0]):
    fig, ax = create_figure(title, y if horizontal else x, x if horizontal else y)
    if horizontal:
        ax.barh(frame[x].astype(str), frame[y], color=color)
    else:
        ax.bar(frame[x].astype(str), frame[y], color=color)
        ax.tick_params(axis="x", rotation=40)
    return fig


def plot_missing_bars(missing_df: pd.DataFrame, *, label_col: str = "column", rate_col: str = "missing_rate",
                      top_n: int = 20, title: str = "Missingness by Column"):
    part = missing_df.sort_values(rate_col, ascending=False).head(top_n).iloc[::-1]
    fig, ax = create_figure(title, "Missing fraction / %", "")
    ax.barh(part[label_col].astype(str), part[rate_col], color=DEFAULT_COLORS[0])
    return fig


def plot_distribution(series: pd.Series, *, title: str | None = None, bins: int = 30, log1p: bool = False):
    """Plot histogram + horizontal boxplot of a numeric Series."""
    from matplotlib.figure import Figure

    raw = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    clean = raw[np.isfinite(raw)]
    if log1p:
        clean = np.sign(clean) * np.log1p(np.abs(clean))
    fig = Figure(figsize=(9, 5.5), layout="constrained")
    ax_box, ax_hist = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 4]})
    if len(clean):
        ax_box.boxplot(clean, orientation="horizontal", widths=0.55)
        ax_hist.hist(clean, bins=bins, color=DEFAULT_COLORS[0], alpha=0.85, edgecolor="white")
    ax_box.set_yticks([])
    ax_box.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_box.set_title(title or f"Distribution of {series.name or 'value'}" + (" (signed log1p)" if log1p else ""))
    ax_hist.set_ylabel("Count")
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, *, title: str = "Correlation Matrix", annotate: bool = True):
    """Plot a clean diverging heatmap for a square correlation matrix."""
    from matplotlib.figure import Figure

    n = len(corr_matrix)
    side = max(5.5, min(14.0, n * 0.7 + 2.0))
    fig = Figure(figsize=(side, side * 0.85), layout="constrained")
    ax = fig.subplots()
    vals = corr_matrix.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n), corr_matrix.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(range(n), corr_matrix.index.astype(str))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if annotate and n <= 16:
        for i in range(n):
            for j in range(n):
                if np.isfinite(vals[i, j]):
                    fg = "white" if abs(vals[i, j]) > 0.6 else "#1e293b"
                    ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", color=fg, fontsize=8.5)
    return fig


def plot_rate_matrix(frame: pd.DataFrame, label_col: str, value_cols: Sequence[str], title: str):
    fig, ax = create_figure(title)
    ax.axis("off")
    cells = [[f"{float(row[c]) * 100:.4f}%" if pd.notna(row[c]) else "NA" for c in value_cols] for _, row in frame.iterrows()]
    tbl = ax.table(
        cellText=cells,
        rowLabels=frame[label_col].astype(str).tolist(),
        colLabels=list(value_cols),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.7)
    return fig


def plot_timeline(frame: pd.DataFrame, *, source_col: str = "source", start_col: str = "first_event",
                  end_col: str = "last_event", title: str = "Observed time range by source"):
    from matplotlib import dates

    part = frame.dropna(subset=[start_col, end_col]).copy()
    first = dates.date2num(pd.to_datetime(part[start_col]).dt.to_pydatetime())
    last = dates.date2num(pd.to_datetime(part[end_col]).dt.to_pydatetime())
    fig, ax = create_figure(title, "Date", source_col)
    ax.barh(part[source_col].astype(str), np.maximum(last - first, 0.5), left=first, color=DEFAULT_COLORS[0])
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=40)
    return fig


def plot_quantile_whisker(summary_row: dict[str, Any] | pd.Series, title: str = "Quantile distribution"):
    fig, ax = create_figure(title, "Value", "")
    r = summary_row
    ax.plot([r["p25"], r["p75"]], [1, 1], linewidth=12, color=DEFAULT_COLORS[0], alpha=0.35)
    ax.scatter([r["median"], r["p95"], r["p99"]], [1, 1, 1], color=[DEFAULT_COLORS[0], DEFAULT_COLORS[2], DEFAULT_COLORS[3]])
    ax.set_yticks([])
    ax.text(r["median"], 1.04, "median", ha="center")
    ax.text(r["p95"], 0.96, "P95", ha="center")
    ax.text(r["p99"], 1.04, "P99", ha="center")
    return fig


def publish_figure_bundle(
    out_dir: str | Path,
    name: str,
    fig: Any,
    data: pd.DataFrame,
    *,
    title: str,
    construction: str = "",
    meaning: str = "",
    insight: str = "",
    limits: str = "",
    footer: str = "",
    formats: Sequence[str] = ("png", "svg"),
    dpi: int = 150,
) -> dict[str, str]:
    """Save figure (PNG/SVG) alongside its exact aggregate CSV and Markdown documentation card."""
    safe_name(name)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    if footer:
        fig.text(0.01, 0.01, footer, fontsize=8, color="#64748b")

    saved: dict[str, str] = {}
    for ext in formats:
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        saved[ext] = str(p.resolve())
    fig.clear()

    csv_path = out / f"{name}.csv"
    data.to_csv(csv_path, index=False, encoding="utf-8-sig")
    saved["csv"] = str(csv_path.resolve())

    img_ref = f"{name}.{formats[0]}" if formats else ""
    md_lines = [f"# {title}", ""]
    if img_ref:
        md_lines.extend([f"![{title}]({img_ref})", ""])
    if meaning:
        md_lines.extend(["## What This Chart Shows", "", meaning, ""])
    if construction:
        md_lines.extend(["## Construction", "", construction, ""])
    if insight:
        md_lines.extend(["## Key Takeaways", "", insight, ""])
    if limits:
        md_lines.extend(["## Limitations", "", limits, ""])
    md_path = out / f"{name}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    saved["md"] = str(md_path.resolve())
    return saved
