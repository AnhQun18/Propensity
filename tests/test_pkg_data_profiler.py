"""Tests for standalone src/data_profiler package on general non-MBD datasets."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data_io import write_table
from src.data_profiler import (
    DatasetSpec,
    OnlineMoments,
    PrimaryCodeGrouper,
    ProfilerConfig,
    TableSpec,
    UniformSample,
    describe_table,
    profile_dataset,
    profile_table,
)


class TestDataProfiler(unittest.TestCase):
    def test_online_moments_exact_across_batches(self):
        rng = np.random.default_rng(7)
        vals = rng.normal(loc=50.0, scale=12.5, size=1000)
        vals[10] = np.nan
        vals[20] = np.inf
        m = OnlineMoments()
        for chunk in np.array_split(vals, [1, 7, 133, 500, 999]):
            m.update(chunk)
        clean = vals[np.isfinite(vals)]
        s = m.summary()
        self.assertEqual(s["finite_n"], len(clean))
        self.assertAlmostEqual(s["mean"], float(clean.mean()), places=10)
        self.assertAlmostEqual(s["std"], float(clean.std(ddof=1)), places=10)
        self.assertAlmostEqual(s["min"], float(clean.min()), places=10)
        self.assertAlmostEqual(s["max"], float(clean.max()), places=10)

    def test_uniform_sample_batch_boundary_invariant(self):
        df = pd.DataFrame({"x": np.arange(500)})
        s1 = UniformSample(size=40, seed=123)
        s1.update(df)

        s2 = UniformSample(size=40, seed=123)
        for cut in (0, 17, 120, 333, 500):
            pass
        for chunk in (df.iloc[:17], df.iloc[17:120], df.iloc[120:333], df.iloc[333:]):
            s2.update(chunk)
        self.assertEqual(s1.frame["x"].tolist(), s2.frame["x"].tolist())

    def test_primary_code_grouper(self):
        s = pd.Series(["USD", "EUR", "JPY", None, "USD", "BGN"])
        g = PrimaryCodeGrouper(primary_values=["USD"], rare_values=["JPY", "BGN"], prefix="curr")
        flags = g.flags(s)
        self.assertEqual(flags["curr_is_primary"].tolist(), [True, False, False, pd.NA, True, False])
        self.assertEqual(flags["curr_is_secondary"].tolist(), [False, True, True, pd.NA, False, True])
        self.assertEqual(flags["curr_missing"].tolist(), [False, False, False, True, False, False])
        grouped = g.group(s)
        self.assertEqual(grouped.tolist(), ["USD", "EUR", "other", "missing", "USD", "other"])

    def test_profile_multitable_multiformat_ecommerce(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            users = pd.DataFrame({"user_id": ["u1", "u2", "u3", "u4"], "segment": ["vip", "std", "std", "new"]})
            labels = pd.DataFrame({
                "user_id": ["u1", "u2", "u3", "u4", "u1", "u2", "u3", "u4"],
                "period": ["2025-01"] * 4 + ["2025-02"] * 4,
                "bought_a": [1, 0, 0, 0, 0, 1, 0, np.nan],
                "bought_b": [1, 0, 0, 0, 0, 0, 0, 0],
            })
            # Intentionally include 1 exact duplicate row and 1 unknown user
            orders = pd.DataFrame({
                "user_id": ["u1", "u1", "u1", "u2", "ghost"],
                "order_ts": ["2025-01-05 10:00:00", "2025-01-05 10:00:00", "2025-01-12 09:00:00",
                             "2025-01-20 14:00:00", "2025-01-25 11:00:00"],
                "spend": [100.0, 100.0, 250.0, None, 50.0],
                "channel": ["web", "web", "app", "web", "app"],
            })
            chats = pd.DataFrame({
                "user_id": ["u1", "u3", "u3"],
                "chat_ts": ["2025-01-08 08:00:00", "2025-01-09 09:00:00", "2025-02-02 10:00:00"],
                "vec": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0]],  # last one has wrong dim
            })

            users_csv = root / "users.csv"
            orders_jsonl = root / "orders.jsonl"
            chats_pq = root / "chats.parquet"
            write_table(users, users_csv)
            write_table(orders, orders_jsonl)
            write_table(chats, chats_pq)

            ds = DatasetSpec(
                name="ecommerce_demo",
                entity_table="users",
                target_table="labels",
                tables=(
                    TableSpec("users", users_csv, role="entity", row_unit="user",
                              entity_key="user_id", unique_key=("user_id",), categorical_columns=("segment",)),
                    TableSpec("labels", labels, role="target", row_unit="user-period",
                              entity_key="user_id", unique_key=("user_id", "period"),
                              time_column="period", target_columns=("bought_a", "bought_b")),
                    TableSpec("orders", orders_jsonl, role="event", row_unit="order",
                              entity_key="user_id", time_column="order_ts",
                              numeric_columns=("spend",), categorical_columns=("channel",)),
                    TableSpec("chats", chats_pq, role="event", row_unit="chat",
                              entity_key="user_id", time_column="chat_ts",
                              vector_columns=("vec",), vector_dims={"vec": 3}),
                ),
            )
            res = profile_dataset(ds, ProfilerConfig(batch_size=2, duplicate_mode="exact"))
            self.assertEqual(set(res.sections.keys()),
                             {"structure", "quality", "temporal", "numeric", "categories_and_vectors", "entity_summary", "targets"})

            dups = res.sections["quality"].tables["duplicates"]
            ord_dup = dups[(dups["source"] == "orders") & (dups["kind"] == "full_row")].iloc[0]
            self.assertEqual(int(ord_dup["duplicate_n"]), 1)

            cov = res.sections["quality"].tables["key_coverage"].set_index("source")
            self.assertEqual(int(cov.loc["orders", "unknown_entity_rows"]), 1)
            self.assertEqual(int(cov.loc["orders", "matched_entities"]), 2)

            vec_q = res.sections["categories_and_vectors"].tables["vector_quality"]
            bad_shape = vec_q[vec_q["check"] == "invalid_vec_shape"]["n"].iloc[0]
            self.assertEqual(int(bad_shape), 1)

            combos = res.sections["entity_summary"].tables["source_combinations"]
            self.assertEqual(int(combos["entities"].sum()), 4)

            cooc = res.sections["targets"].tables["target_cooccurrence"]
            self.assertEqual(int(cooc["both_positive"].iloc[0]), 1)

            # Single-table one-liner, describe_table + CLI test
            desc = describe_table(orders_jsonl)
            self.assertIn("skewness", desc.columns)
            self.assertIn("zeros_pct", desc.columns)
            self.assertEqual(len(desc), 4)

            single = profile_table(orders_jsonl, entity_key="user_id", time_column="order_ts")
            self.assertIn("numeric", single.sections)
            self.assertIn("skewness", single.sections["numeric"].tables["numeric_summary"].columns)
            repo_root = Path(__file__).resolve().parents[1]
            out_cli = root / "cli_out"
            subprocess.run(
                [sys.executable, "-m", "src.data_profiler", str(orders_jsonl), "--out", str(out_cli)],
                cwd=str(repo_root), check=True, capture_output=True, text=True,
            )
            self.assertTrue((out_cli / "metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
