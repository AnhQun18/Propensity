"""Tests for standalone src/eda_report package."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data_profiler import profile_table
from src.eda_advanced import correlation_analysis
from src.eda_report import (
    SlideBuilder,
    export_report,
    format_number,
    plot_bars,
    plot_correlation_heatmap,
    plot_distribution,
    plot_missing_bars,
    plot_quantile_whisker,
    plot_rate_matrix,
    publish_figure_bundle,
)


class TestEDAReport(unittest.TestCase):
    def test_format_number_and_eda_charts(self):
        self.assertEqual(format_number(12345.67, 2, locale="en"), "12,345.67")
        self.assertEqual(format_number(12345.67, 2, locale="vi"), "12.345,67")

        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 100.0],
            "b": [2.0, 4.0, 5.0, 8.0, 190.0],
            "c": [10.0, np.nan, 3.0, 2.0, 1.0],
        })
        corr = correlation_analysis(df, method="spearman")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, fig in [
                ("corr.png", plot_correlation_heatmap(corr["matrix"])),
                ("dist.png", plot_distribution(df["a"], log1p=True)),
                ("miss.png", plot_missing_bars(pd.DataFrame({"column": ["c", "a"], "missing_rate": [0.2, 0.0]}))),
                ("quant.png", plot_quantile_whisker({"p25": 1, "median": 3, "p75": 4, "p95": 80, "p99": 95})),
            ]:
                fig.savefig(root / name)
                fig.clear()
                self.assertTrue((root / name).is_file())

    def test_privacy_guard_profile_export_bundle_and_slides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "rep"
            with self.assertRaises(ValueError):
                export_report({"leak": {"raw": pd.DataFrame({"user_id": ["u1"], "val": [1]})}}, out, forbidden_columns=("user_id",))
            self.assertFalse(out.exists())

            df = pd.DataFrame({"user_id": ["u1", "u2", "u1"], "spend": [10.0, 20.0, 30.0], "cat": ["a", "b", "a"]})
            prof = profile_table(df, name="orders", entity_key="user_id")
            fig = plot_bars(pd.DataFrame({"cat": ["a", "b"], "n": [2, 1]}), "cat", "n", "Category counts")
            dest = export_report(
                prof,
                out,
                erd_edges=[{"left": "users", "right": "orders", "key": "user_id"}],
                figures={"cat_counts": fig},
            )
            self.assertTrue((dest / "report.md").is_file())
            self.assertIn("erDiagram", (dest / "report.md").read_text(encoding="utf-8"))
            self.assertGreater(len(json.loads((dest / "manifest.json").read_text(encoding="utf-8"))["tables"]), 0)

            data = pd.DataFrame({"seg": ["A", "B"], "rate": [0.12, 0.04]})
            bundle = publish_figure_bundle(
                root / "bundle", "seg_rates", plot_rate_matrix(data, "seg", ["rate"], "Rates"),
                data, title="Segment Rates", insight="A > B", formats=("png", "svg"),
            )
            for k in ("png", "svg", "csv", "md"):
                self.assertTrue(Path(bundle[k]).is_file())

            deck = SlideBuilder()
            s1, bg1 = deck.new_slide("EDA Summary", "Synthetic Demo")
            deck.add_kpi_card(bg1, 0.06, 0.55, 0.28, 0.20, "3.0x", "Lift")
            deck.add_bottom_pill(bg1, "Segment A leads by 3.0x")
            pdf_path = deck.export_pdf([s1], root / "deck.pdf", png_dir=root / "slides_png")
            self.assertTrue(pdf_path.is_file())
            self.assertTrue((root / "slides_png/slide_01.png").is_file())


if __name__ == "__main__":
    unittest.main()
