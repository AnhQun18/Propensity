"""Minimalist 16:9 Matplotlib EDA presentation slide deck builder & PDF exporter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SlideTheme:
    header_bg: str = "#0f294a"
    header_fg: str = "#ffffff"
    kpi_bg: str = "#eaf3fc"
    pill_bg: str = "#e2ecf8"
    accent: str = "#2b7de9"
    text_dark: str = "#1e293b"
    text_muted: str = "#64748b"
    border: str = "#cbd5e1"
    canvas_bg: str = "#ffffff"
    font_family: str = "DejaVu Sans"


class SlideBuilder:
    """Create clean 16:9 EDA slides with header bars, KPI cards, takeaway pills, and PDF export."""

    def __init__(self, theme: SlideTheme | None = None, *, dpi: int = 180):
        self.theme = theme or SlideTheme()
        self.dpi = dpi

    def new_slide(self, header_title: str, subtitle_text: str | None = None):
        from matplotlib.figure import Figure
        from matplotlib.patches import FancyBboxPatch

        t = self.theme
        fig = Figure(figsize=(16, 9), dpi=self.dpi, facecolor=t.canvas_bg)
        bg = fig.add_axes([0, 0, 1, 1], frameon=False)
        bg.set_xlim(0, 1)
        bg.set_ylim(0, 1)
        bg.axis("off")

        header = FancyBboxPatch(
            (0.04, 0.865), 0.92, 0.088,
            boxstyle="round,pad=0.004,rounding_size=0.020",
            facecolor=t.header_bg, edgecolor="none", zorder=2,
        )
        bg.add_patch(header)
        bg.text(
            0.065, 0.909, header_title,
            fontsize=20, fontweight="bold", color=t.header_fg,
            fontfamily=t.font_family, va="center", ha="left", zorder=3,
        )
        if subtitle_text:
            bg.text(
                0.045, 0.822, subtitle_text,
                fontsize=12, color=t.text_muted,
                fontfamily=t.font_family, va="center", ha="left", zorder=3,
            )
        return fig, bg

    def add_kpi_card(self, bg: Any, x: float, y: float, w: float, h: float,
                     value: str, label: str, *, value_color: str | None = None) -> None:
        from matplotlib.patches import FancyBboxPatch

        t = self.theme
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.004,rounding_size=0.020",
            facecolor=t.kpi_bg, edgecolor="none", zorder=2,
        )
        bg.add_patch(box)
        bg.text(x + w / 2, y + h * 0.62, value, fontsize=24, fontweight="bold",
                color=value_color or t.header_bg, fontfamily=t.font_family,
                ha="center", va="center", zorder=3)
        bg.text(x + w / 2, y + h * 0.25, label, fontsize=11.5,
                color=t.text_dark, fontfamily=t.font_family,
                ha="center", va="center", zorder=3)

    def add_bottom_pill(self, bg: Any, line1: str, line2: str | None = None) -> None:
        from matplotlib.patches import FancyBboxPatch

        t = self.theme
        pill = FancyBboxPatch(
            (0.08, 0.048), 0.84, 0.105,
            boxstyle="round,pad=0.004,rounding_size=0.032",
            facecolor=t.pill_bg, edgecolor="none", zorder=2,
        )
        bg.add_patch(pill)
        if line2:
            bg.text(0.50, 0.118, line1, fontsize=13.2, fontweight="bold", color=t.header_bg,
                    fontfamily=t.font_family, ha="center", va="center", zorder=3)
            bg.text(0.50, 0.078, line2, fontsize=12.0, color=t.text_dark,
                    fontfamily=t.font_family, ha="center", va="center", zorder=3)
        else:
            bg.text(0.50, 0.100, line1, fontsize=14, fontweight="bold", color=t.header_bg,
                    fontfamily=t.font_family, ha="center", va="center", zorder=3)

    def style_axes(self, ax: Any, title: str | None = None) -> None:
        t = self.theme
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(t.border)
        ax.spines["bottom"].set_color(t.border)
        ax.tick_params(colors=t.text_dark, labelsize=10.5)
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold", color=t.header_bg, pad=14)

    def export_pdf(self, slides: Sequence[Any], pdf_path: str | Path, *, png_dir: str | Path | None = None) -> Path:
        from matplotlib.backends.backend_pdf import PdfPages

        dest = Path(pdf_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        png_root = Path(png_dir) if png_dir is not None else None
        if png_root is not None:
            png_root.mkdir(parents=True, exist_ok=True)

        with PdfPages(dest) as pdf:
            for idx, fig in enumerate(slides, start=1):
                pdf.savefig(fig, dpi=self.dpi)
                if png_root is not None:
                    fig.savefig(png_root / f"slide_{idx:02d}.png", dpi=self.dpi)
                fig.clear()
        return dest
