"""Privacy-safe aggregate EDA reporting, headless EDA charts, and 16:9 PDF slide decks."""

from .charts import (
    DEFAULT_COLORS,
    create_figure,
    plot_bars,
    plot_correlation_heatmap,
    plot_distribution,
    plot_missing_bars,
    plot_quantile_whisker,
    plot_rate_matrix,
    plot_timeline,
    publish_figure_bundle,
)
from .exporter import export_report
from .formatting import format_number, json_safe, safe_name
from .slides import SlideBuilder, SlideTheme

__all__ = [
    "format_number",
    "safe_name",
    "json_safe",
    "export_report",
    "DEFAULT_COLORS",
    "create_figure",
    "plot_bars",
    "plot_missing_bars",
    "plot_distribution",
    "plot_correlation_heatmap",
    "plot_rate_matrix",
    "plot_timeline",
    "plot_quantile_whisker",
    "publish_figure_bundle",
    "SlideTheme",
    "SlideBuilder",
]
