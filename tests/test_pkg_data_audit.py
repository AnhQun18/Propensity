"""Tests for standalone src/data_audit package across multiple formats."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data_audit import (
    audit_invalid_records,
    audit_join_keys,
    audit_null_patterns,
    audit_temporal_coverage,
    audit_univariate_outliers,
    audit_within_group_outliers,
)
from src.data_io import write_table


class TestDataAudit(unittest.TestCase):
    def test_null_patterns_and_cli(self):
        df = pd.DataFrame({
            "user_id": ["u1", "u2", "u3", "u4", "u5"],
            "channel": ["web", "web", "app", "app", "web"],
            "a": [1.0, None, None, 4.0, 5.0],
            "b": [10.0, None, 30.0, 40.0, None],
        })
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "events.csv"
            out_dir = Path(tmp) / "null_out"
            write_table(df, csv_path)
            res = audit_null_patterns(csv_path, group_by=["channel"], output_dir=out_dir)
            self.assertEqual(res["summary"]["rows"], 5)
            self.assertEqual(res["summary"]["rows_any_null"], 3)
            self.assertEqual(res["summary"]["complete_rows"], 2)
            patterns = dict(zip(res["joint_patterns"]["null_pattern"], res["joint_patterns"]["rows"]))
            self.assertEqual(patterns["a,b"], 1)
            self.assertIn("channel", res["by_group"])
            self.assertTrue((out_dir / "joint_patterns.csv").is_file())

            repo_root = Path(__file__).resolve().parents[1]
            cli_res = subprocess.run(
                [sys.executable, "-m", "src.data_audit", "nulls", str(csv_path)],
                cwd=str(repo_root), check=True, capture_output=True, text=True,
            )
            self.assertEqual(json.loads(cli_res.stdout)["rows_any_null"], 3)

    def test_univariate_and_within_group_outliers(self):
        rng = np.random.default_rng(0)
        vals = np.concatenate([rng.normal(100.0, 5.0, size=200), [10_000.0, -5_000.0]])
        uni = audit_univariate_outliers(pd.DataFrame({"price": vals, "qty": np.ones(len(vals))}))
        price_row = uni.set_index("column").loc["price"]
        self.assertGreaterEqual(int(price_row["iqr_outlier_n"]), 2)
        self.assertGreaterEqual(int(price_row["mad_outlier_n"]), 2)

        rows = []
        for u in ("u1", "u2", "u3", "u4", "u5", "u6"):
            base = 100.0 if u in ("u1", "u2") else 500.0
            for i in range(25):
                rows.append({
                    "user_id": u,
                    "ts": f"2025-01-{(i % 20) + 1:02d} 10:00:00",
                    "curr": "USD",
                    "cat": "retail",
                    "amount": float(base * rng.uniform(0.9, 1.1)),
                    "meta": f"m{i % 2}",
                })
        rows.append({"user_id": "u1", "ts": "2025-01-25 10:00:00", "curr": "USD", "cat": "retail", "amount": 100_000.0, "meta": "m0"})
        rows.append({"user_id": "u1", "ts": "2025-01-25 10:00:00", "curr": "USD", "cat": "retail", "amount": 100_000.0, "meta": "m1"})
        df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp:
            pq_path = Path(tmp) / "trx.parquet"
            write_table(df, pq_path)
            res = audit_within_group_outliers(
                pq_path,
                entity_col="user_id",
                value_col="amount",
                group_cols=["curr", "cat"],
                time_col="ts",
                metadata_cols=["meta"],
                min_group_n=20,
                n_buckets=4,
                cluster_k=2,
            )
            self.assertGreaterEqual(len(res["top_flagged"]), 2)
            self.assertEqual(int(res["repeat_summary"]["differing_metadata_groups"].iloc[0]), 1)
            self.assertEqual(len(res["cluster_summary"]), 2)

    def test_temporal_coverage_invalid_records_and_joins(self):
        users = pd.DataFrame({"user_id": ["u1", "u2", "u3"], "grp": [0, 1, 0]})
        orders = pd.DataFrame({
            "user_id": ["u1", "u1", "u1", "u2", "ghost", None, "u2"],
            "ts": ["2025-01-01 09:00:00", "2025-01-01 09:00:00", "2025-01-03 09:00:00",
                   "2025-01-02 10:00:00", "2025-01-02 11:00:00", "2025-01-02 12:00:00", "bad_ts"],
            "amount": [10.0, 20.0, None, 30.0, 40.0, 50.0, 60.0],
        })
        clicks = pd.DataFrame({
            "user_id": ["u1", "u1", "u1", "u1"],
            "ts": ["2025-01-01 08:00:00", "2025-01-02 08:00:00", "2025-01-03 08:00:00", "2025-01-04 08:00:00"],
            "page": ["home", "cart", "pay", "done"],
        })
        cov = audit_temporal_coverage(
            users, {"orders": orders, "clicks": clicks}, entity_col="user_id", time_col="ts", cutoff="2025-01-05"
        )
        self.assertEqual(cov["raw_id_time"]["any_source_active_days_all_entities"]["entities_absent_strictly_over_50_pct"], 2)

        inv = audit_invalid_records(orders, users, entity_col="user_id", time_col="ts", source_name="orders")
        self.assertEqual(inv["summary"]["raw_rows"], 7)
        self.assertEqual(inv["summary"]["invalid_rows"], 3)
        self.assertEqual(inv["summary"]["valid_rows"], 4)
        self.assertEqual(inv["summary"]["timestamp_collision_groups"], 1)

        panel = pd.DataFrame({
            "user_id": ["u1", "u2", "u3", "orphan"],
            "period": ["2025-01", "2025-01", "2025-01", "2025-01"],
            "grp": [0, 1, 1, 0],
            "y1": [1, 0, 0, 0],
            "y2": [1, 1, 0, 0],
        })
        j = audit_join_keys(users, panel, entity_col="user_id", period_col="period", group_col="grp", target_cols=["y1", "y2"])
        self.assertEqual(j["orphan_panel_rows"], 1)
        self.assertEqual(j["group_mismatches"], 1)
        self.assertEqual(j["target_pair_overlap"]["y1 & y2"], 1)


if __name__ == "__main__":
    unittest.main()
