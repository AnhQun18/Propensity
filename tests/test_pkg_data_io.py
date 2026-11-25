"""Tests for the standalone multi-format src/data_io package."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_io import (
    DuckDBSession,
    RunManifest,
    SourceSpec,
    assert_non_overlapping_paths,
    compute_file_inventory,
    detect_format,
    inspect_inventory,
    inventory_sha256,
    iter_batches,
    load_arrow_table,
    load_dataframe,
    parse_timestamps,
    verify_inventory_unchanged,
    write_table,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u1", "u4"],
        "event_ts": ["2025-01-01 10:00:00", "2025-01-02 12:30:00", "2025-01-03 08:15:00",
                     "2025-01-04 19:45:00", "2025-01-05 09:00:00"],
        "spend": [10.5, 25.0, None, 40.25, 0.0],
        "qty": pd.Series([1, 2, pd.NA, 4, 1], dtype="Int64"),
        "channel": ["web", "app", "web", None, "store"],
    })


class TestDataIO(unittest.TestCase):
    def test_multi_format_roundtrip_and_batching(self):
        df = _sample_df()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1. Parquet (single)
            pq_path = root / "orders.parquet"
            info = write_table(df, pq_path)
            self.assertEqual(info["rows"], 5)
            self.assertTrue(len(info["sha256"]) == 64)

            # 2. Hive-partitioned Parquet directory
            hive_dir = root / "orders_hive"
            (hive_dir / "region=north").mkdir(parents=True)
            (hive_dir / "region=south").mkdir(parents=True)
            pq.write_table(pa.Table.from_pandas(df.iloc[:3], preserve_index=False), hive_dir / "region=north/p0.parquet")
            pq.write_table(pa.Table.from_pandas(df.iloc[3:], preserve_index=False), hive_dir / "region=south/p1.parquet")

            # 3. CSV + gzipped CSV + TSV
            csv_path = root / "orders.csv"
            write_table(df, csv_path)
            csvgz_path = root / "orders.csv.gz"
            with gzip.open(csvgz_path, "wt", encoding="utf-8") as fh:
                df.to_csv(fh, index=False)
            tsv_path = root / "orders.tsv"
            write_table(df, tsv_path)

            # 4. JSON & JSONL
            json_path = root / "orders.json"
            write_table(df, json_path)
            jsonl_path = root / "orders.jsonl"
            write_table(df, jsonl_path)

            # 5. Feather / Arrow IPC
            feather_path = root / "orders.feather"
            write_table(df, feather_path)

            # 6. SQLite
            sqlite_path = root / "orders.sqlite"
            write_table(df, sqlite_path, table="orders_tbl")

            # 7. DuckDB file
            duck_path = root / "orders.duckdb"
            con = duckdb.connect(str(duck_path))
            con.register("src_df", df)
            con.execute("CREATE TABLE orders_tbl AS SELECT * FROM src_df")
            con.close()

            # 8. Pickle (blocked by default, allowed explicitly)
            pkl_path = root / "orders.pkl"
            df.to_pickle(pkl_path)
            with self.assertRaises(PermissionError):
                load_dataframe(pkl_path)
            self.assertEqual(len(load_dataframe(pkl_path, allow_pickle=True)), 5)

            # Check every format yields 5 rows & exact user_ids
            sources = [
                pq_path,
                csv_path,
                csvgz_path,
                tsv_path,
                json_path,
                jsonl_path,
                feather_path,
                f"{sqlite_path}::orders_tbl",
                f"sqlite:///{sqlite_path.as_posix()}::orders_tbl",
                f"{duck_path}::orders_tbl",
                df,
                pa.Table.from_pandas(df, preserve_index=False),
                df.to_dict(orient="records"),
            ]
            for src in sources:
                batches = list(iter_batches(src, columns=["user_id", "spend"], batch_size=2))
                loaded = pd.concat(batches, ignore_index=True)
                self.assertEqual(len(loaded), 5, f"Failed row count for {src!r}")
                self.assertEqual(loaded["user_id"].tolist(), df["user_id"].tolist())

            # Check Hive partitions attached properly
            hive_df = load_dataframe(hive_dir)
            self.assertIn("region", hive_df.columns)
            self.assertEqual(sorted(hive_df["region"].unique().tolist()), ["north", "south"])
            self.assertEqual(len(hive_df), 5)

            # Check glob pattern
            glob_df = load_dataframe(str(root / "orders_hive/**/*.parquet"))
            self.assertEqual(len(glob_df), 5)

    def test_duckdb_session_multi_format(self):
        df = _sample_df()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "users.csv"
            pq_path = root / "events.parquet"
            jsonl_path = root / "logs.jsonl"
            write_table(pd.DataFrame({"user_id": ["u1", "u2", "u3", "u4"], "tier": ["gold", "silver", "gold", "bronze"]}), csv_path)
            write_table(df, pq_path)
            write_table(df.iloc[:2], jsonl_path)

            with DuckDBSession(memory_limit="1GB", threads=1, spill_dir=root / "spill") as db:
                db.register("users", csv_path)
                db.register("events", pq_path)
                db.register("logs", jsonl_path)
                db.register("mem", pa.Table.from_pandas(df, preserve_index=False))
                joined = db.query("SELECT u.tier, count(*) AS n FROM events e JOIN users u USING(user_id) GROUP BY 1 ORDER BY 1")
                self.assertEqual(int(joined["n"].sum()), 5)
                self.assertEqual(int(db.query("SELECT count(*) c FROM logs")["c"].iloc[0]), 2)
                self.assertEqual(int(db.query("SELECT count(*) c FROM mem")["c"].iloc[0]), 5)
                out_info = db.export_parquet("SELECT * FROM joined_check" if False else "SELECT * FROM events", root / "out/exp.parquet")
                self.assertTrue(Path(out_info["path"]).is_file())

    def test_inventory_and_integrity_guards(self):
        df = _sample_df()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "raw"
            data_dir.mkdir()
            write_table(df, data_dir / "a.parquet")
            write_table(df, data_dir / "b.csv")

            inv = inspect_inventory({"a": data_dir / "a.parquet", "b": data_dir / "b.csv"})
            self.assertEqual(inv["total_rows"], 10)
            self.assertEqual(inv["tables"]["a"]["format"], "parquet")
            self.assertEqual(inv["tables"]["b"]["format"], "csv")

            files = {"a": [data_dir / "a.parquet"], "b": [data_dir / "b.csv"]}
            snap = compute_file_inventory(files, base_dir=data_dir)
            self.assertEqual(len(inventory_sha256(snap)), 64)
            verify_inventory_unchanged(snap, files, base_dir=data_dir)

            # Overlap guard must reject output inside raw tree
            with self.assertRaises(ValueError):
                assert_non_overlapping_paths(data_dir / "sub_out", [data_dir])

            # RunManifest lifecycle: ready & failed
            out_ok = root / "run_ok"
            with RunManifest(out_ok, protected_paths=[data_dir], initial={"job": "test"}) as m:
                m.update(step="done")
            saved = json.loads((out_ok / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "ready")
            self.assertEqual(saved["step"], "done")

            out_fail = root / "run_fail"
            with self.assertRaises(RuntimeError):
                with RunManifest(out_fail, protected_paths=[data_dir]):
                    raise RuntimeError("boom")
            failed = json.loads((out_fail / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "failed")
            self.assertIn("RuntimeError: boom", failed["error"])

    def test_parse_timestamps_timezone_rules(self):
        naive = parse_timestamps(["2025-01-01 00:00:00", "2025-01-02 12:00:00"])
        self.assertIsNone(naive.dt.tz)
        with self.assertRaises(ValueError):
            parse_timestamps(["2025-01-01T00:00:00Z"])
        utc = parse_timestamps(["2025-01-01T00:00:00Z"], timezone="UTC")
        self.assertEqual(str(utc.dt.tz), "UTC")

    def test_pythonpath_src_import_and_cli(self):
        repo_root = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
        proc = subprocess.run(
            [sys.executable, "-c", "import data_io; print(data_io.detect_format('x.csv.gz'))"],
            env=env, capture_output=True, text=True, check=True,
        )
        self.assertEqual(proc.stdout.strip(), "csv")

        with tempfile.TemporaryDirectory() as tmp:
            src_csv = Path(tmp) / "in.csv"
            dst_pq = Path(tmp) / "out.parquet"
            write_table(_sample_df(), src_csv)
            cli_proc = subprocess.run(
                [sys.executable, "-m", "src.data_io", "convert", str(src_csv), str(dst_pq)],
                cwd=str(repo_root), capture_output=True, text=True, check=True,
            )
            payload = json.loads(cli_proc.stdout)
            self.assertEqual(payload["rows"], 5)
            self.assertTrue(dst_pq.is_file())


if __name__ == "__main__":
    unittest.main()
