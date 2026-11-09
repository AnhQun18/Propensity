"""Synthetic point-in-time checks. Authored for execution on the target machine."""
from dataclasses import replace
from datetime import datetime
import calendar
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from propensity_data.config import PreparationConfig
from propensity_data.protocol import evaluation_protocol


MONTHS = [f"2022-{m:02d}" for m in range(2, 13)] + ["2023-01"]


class ProtocolTests(unittest.TestCase):
    def config(self, root):
        return PreparationConfig(Path(root) / "raw", Path(root) / "prepared")

    def test_calendar_and_label_delay(self):
        with tempfile.TemporaryDirectory() as root:
            config = self.config(root)
            protocol = evaluation_protocol(config, MONTHS)
            self.assertEqual(protocol["rolling_validation"][0]["train"][-1], "2022-08")
            self.assertEqual(protocol["final_train"][-1], "2022-11")
            self.assertFalse(protocol["refit_after_calibration"])
            with self.assertRaisesRegex(ValueError, "Calibration labels unavailable"):
                evaluation_protocol(replace(config, label_delay_days=1), MONTHS)
            shifted = replace(config, label_delay_days=1, calibration_month="2022-11",
                              validation_months=("2022-08", "2022-09"))
            self.assertEqual(evaluation_protocol(shifted, MONTHS)["final_train"][-1], "2022-09")

    def test_reject_raw_output_overlap(self):
        with tempfile.TemporaryDirectory() as root:
            raw = Path(root) / "raw"
            with self.assertRaises(ValueError):
                PreparationConfig(raw, raw / "prepared")
            with self.assertRaises(ValueError):
                PreparationConfig(raw, Path(root))


@unittest.skipUnless(importlib.util.find_spec("duckdb"), "Install .[prepare] under Python >=3.11")
class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config = PreparationConfig(root / "raw", root / "prepared", memory_limit="512MB", threads=1,
                                        embedding_size=2, include_dialog_embeddings=True)
        self.write("client_split", {"client_id": ["a", "b", "c"]})
        targets = {"client_id": [], "mon": [], **{f"target_{i}": [] for i in range(1, 5)}}
        for month in MONTHS:
            for client in ("a", "b", "c"):
                targets["client_id"].append(client)
                year, number = map(int, month.split('-'))
                # Real MBD mon values are month-end; cutoff must still be month-start.
                targets["mon"].append(datetime(year, number, calendar.monthrange(year, number)[1]))
                for i in range(1, 5):
                    # b has an unknown label; c has a known positive despite another null.
                    value = None if client in ("b", "c") and i == 1 else int(client == "c" and i == 2)
                    targets[f"target_{i}"].append(value)
        self.targets = targets
        self.write("targets", targets)
        from propensity_data.pipeline import CATEGORIES
        rows = 7
        self.trx = {
            "client_id": ["a", "a", "a", "a", "a", "outside", "a"],
            "event_time": [datetime(2022, 1, 1), datetime(2022, 12, 1), datetime(2022, 11, 1),
                           datetime(2022, 11, 1), None, datetime(2022, 11, 1), datetime(2023, 2, 1)],
            "amount": [10., 999., 1e12, float("inf"), 2., 2., 1e15],
            "currency": [11, 11, 2, None, 11, 11, 11],
            **{cat: [1] * rows for cat in CATEGORIES},
        }
        self.write("detail/trx", self.trx)
        self.write("detail/geo", {
            "client_id": ["a"], "event_time": [datetime(2022, 11, 1)],
            "geohash_4": [1], "geohash_5": [2], "geohash_6": [3],
        })
        self.write("detail/dialog", {
            "client_id": ["a", "a", "a"],
            "event_time": [datetime(2022, 11, 1), datetime(2022, 11, 2), datetime(2022, 11, 3)],
            "embedding": [[1., 2.], [3., 4.], [float("nan"), 5.]],
        })

    def write(self, relative, data):
        folder = self.config.data_root / relative
        folder.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(data), folder / "part.parquet")

    def prepare(self, config=None):
        from propensity_data.pipeline import prepare
        return prepare(config or self.config)

    def frame(self, month):
        # ParquetFile avoids Hive type inference on parent directories.
        return pq.ParquetFile(self.config.output_dir / f"features/month={month}/part.parquet").read().to_pandas().set_index("client_id")

    def test_temporal_boundary_missing_sources_currency_and_labels(self):
        manifest = json.loads(self.prepare().read_text())
        self.assertEqual(manifest["status"], "ready")
        self.assertEqual(manifest["source_audits"]["trx"]["quarantined_rows"], 2)
        frame = self.frame("2022-12")
        self.assertEqual(len(frame), 3)
        self.assertEqual(frame.loc["a", "trx_all_event_count"], 3)  # exact cutoff and future excluded
        self.assertEqual(frame.loc["a", "trx_30d_event_count"], 2)  # lower bound included
        self.assertEqual(frame.loc["a", "trx_all_main_amount_sum"], 10.)  # no mixed-currency sum
        self.assertEqual(frame.loc["a", "trx_all_foreign_count"], 1)
        self.assertEqual(frame.loc["a", "trx_all_amount_invalid_count"], 1)
        currency = pq.ParquetFile(self.config.output_dir / "currency/month=2022-12/all.parquet").read().to_pandas()
        self.assertEqual(currency.loc[currency.currency.eq(2), "amount_sum"].iloc[0], 1e12)
        self.assertEqual(frame.loc["a", "dialog_84d_valid_embedding_count"], 2)
        self.assertEqual(frame.loc["a", "dialog_84d_embedding_mean_1"], 2.)
        self.assertEqual(frame.loc["a", "dialog_84d_embedding_last_1"], 3.)
        self.assertEqual(frame.loc["b", "cold_start"], 1)
        self.assertFalse(frame.loc["b", "supervised_eligible"])
        self.assertEqual(frame.loc["c", "y_any"], 1)
        self.assertEqual(frame.loc["b", "trx_all_event_count"], 0)
        self.assertTrue(frame.loc["b", ["trx_all_days_since_last_event"]].isna().all())
        early = self.frame("2022-02")
        self.assertTrue(early.loc["a", ["geo_30d_event_count"]].isna().all())
        schema = json.loads((self.config.output_dir / "feature_schema.json").read_text())
        self.assertNotIn("client_id", schema["predictors"])
        self.assertNotIn("target_1", schema["predictors"])
        self.assertFalse(manifest["models_fitted"])
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_future_event_values_do_not_change_past_features(self):
        self.prepare()
        before = self.frame("2022-12")
        self.trx["amount"][-1] = 7.
        self.trx["event_time"][-1] = datetime(2025, 1, 1)
        self.write("detail/trx", self.trx)
        other = replace(self.config, output_dir=self.config.output_dir.parent / "other")
        self.prepare(other)
        after = pq.ParquetFile(other.output_dir / "features/month=2022-12/part.parquet").read().to_pandas().set_index("client_id")
        import pandas as pd
        pd.testing.assert_frame_equal(before, after)

    def test_missing_target_row_preserves_scoring_customer(self):
        for key in self.targets:
            self.targets[key] = self.targets[key][:-1]
        self.write("targets", self.targets)
        self.prepare()
        frame = self.frame("2023-01")
        self.assertEqual(len(frame), 3)
        self.assertFalse(frame.loc["c", "supervised_eligible"])

    def test_duplicate_target_fails_without_touching_raw(self):
        for values in self.targets.values():
            values.append(values[0])
        self.write("targets", self.targets)
        with self.assertRaisesRegex(ValueError, "Duplicate customer-month"):
            self.prepare()
        self.assertEqual(json.loads((self.config.output_dir / "manifest.json").read_text())["status"], "failed")
        self.assertTrue((self.config.data_root / "targets/part.parquet").is_file())

    def test_nonbinary_label_fails(self):
        self.targets["target_1"][0] = 0.5
        self.write("targets", self.targets)
        with self.assertRaisesRegex(ValueError, "outside 0/1/null"):
            self.prepare()

    def test_metadata_only_does_not_require_embeddings(self):
        self.write("detail/dialog", {"client_id": ["a"], "event_time": [datetime(2022, 11, 1)]})
        self.prepare(replace(self.config, include_dialog_embeddings=False))
        schema = json.loads((self.config.output_dir / "feature_schema.json").read_text())
        self.assertFalse(schema["ablation_groups"]["dialog_content"])
        self.assertEqual(self.frame("2022-12").loc["a", "dialog_84d_event_count"], 1)


if __name__ == "__main__":
    unittest.main()
