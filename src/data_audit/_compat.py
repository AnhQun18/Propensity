"""Optional sibling bridge to data_io with standalone DuckDB/pandas fallback."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_data_io():
    prefix = (__package__ or "").rpartition(".")[0]
    for target in ((f"{prefix}.data_io", "data_io") if prefix else ("data_io",)):
        try:
            return importlib.import_module(target)
        except ImportError:
            continue
    return None


def sql_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def sql_literal(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def normalized_id_expr(column: str) -> str:
    return f"nullif(trim(cast({sql_ident(column)} AS VARCHAR)), '')"


def write_json(path: str | Path, value: Any) -> Path:
    dio = _load_data_io()
    if dio is not None:
        return dio.write_json(path, value)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return dest


def open_duckdb_session(*, memory_limit: str = "2GB", threads: int = 2, spill_dir: Path | None = None, timezone: str | None = "UTC"):
    dio = _load_data_io()
    if dio is not None:
        return dio.DuckDBSession(memory_limit=memory_limit, threads=threads, spill_dir=spill_dir, timezone=timezone)
    import duckdb

    class _FallbackSession:
        def __init__(self):
            self.con = duckdb.connect(":memory:")
            if timezone:
                self.con.execute(f"SET TimeZone = {sql_literal(timezone)}")
            self.con.execute(f"SET memory_limit = {sql_literal(memory_limit)}")
            self.con.execute(f"SET threads = {int(threads)}")
            self.con.execute("SET preserve_insertion_order = false")
            if spill_dir is not None:
                Path(spill_dir).mkdir(parents=True, exist_ok=True)
                self.con.execute(f"SET temp_directory = {sql_literal(Path(spill_dir).as_posix())}")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.con.close()
            return False

        def register(self, view_name: str, source: Any, **kwargs):
            if isinstance(source, pd.DataFrame):
                self.con.register(view_name, source)
                return view_name
            p = Path(source)
            if p.suffix.lower() == ".csv":
                self.con.execute(f"CREATE OR REPLACE VIEW {sql_ident(view_name)} AS SELECT * FROM read_csv_auto({sql_literal(p.as_posix())})")
            else:
                self.con.execute(f"CREATE OR REPLACE VIEW {sql_ident(view_name)} AS SELECT * FROM read_parquet({sql_literal(p.as_posix())})")
            return view_name

        def query(self, sql: str, params=None) -> pd.DataFrame:
            return (self.con.execute(sql, params) if params is not None else self.con.execute(sql)).df()

        def execute(self, sql: str, params=None):
            return self.con.execute(sql, params) if params is not None else self.con.execute(sql)

        def export_parquet(self, query: str, path: str | Path, **kwargs):
            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.con.execute(f"COPY ({query}) TO {sql_literal(dest.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")
            return {"path": str(dest)}

    return _FallbackSession()


def iter_batches(source: Any, *, columns=None, batch_size: int = 250_000, **kwargs):
    dio = _load_data_io()
    if dio is not None:
        yield from dio.iter_batches(source, columns=columns, batch_size=batch_size, **kwargs)
        return
    if isinstance(source, pd.DataFrame):
        frame = source[list(columns)] if columns is not None else source
        for i in range(0, len(frame), batch_size):
            yield frame.iloc[i:i + batch_size].reset_index(drop=True)
        return
    frame = pd.read_parquet(source, columns=list(columns) if columns is not None else None)
    for i in range(0, len(frame), batch_size):
        yield frame.iloc[i:i + batch_size].reset_index(drop=True)


def load_dataframe(source: Any, *, columns=None, **kwargs) -> pd.DataFrame:
    dio = _load_data_io()
    if dio is not None:
        return dio.load_dataframe(source, columns=columns, **kwargs)
    frames = list(iter_batches(source, columns=columns, **kwargs))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(columns or []))
