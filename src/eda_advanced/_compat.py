"""Optional sibling bridge to data_io with standalone DuckDB/pandas fallback."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

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


def load_dataframe(source: Any, *, columns=None, **kwargs) -> pd.DataFrame:
    dio = _load_data_io()
    if dio is not None:
        return dio.load_dataframe(source, columns=columns, **kwargs)
    if isinstance(source, pd.DataFrame):
        return source[list(columns)].copy() if columns is not None else source.copy()
    p = Path(source)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, usecols=list(columns) if columns else None)
    return pd.read_parquet(p, columns=list(columns) if columns else None)


def open_duckdb_session(*, memory_limit: str = "2GB", threads: int = 2):
    dio = _load_data_io()
    if dio is not None:
        return dio.DuckDBSession(memory_limit=memory_limit, threads=threads)
    import duckdb

    class _Fallback:
        def __init__(self):
            self.con = duckdb.connect(":memory:")
            self.con.execute(f"SET memory_limit = {sql_literal(memory_limit)}")
            self.con.execute(f"SET threads = {int(threads)}")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.con.close()
            return False

        def register(self, name: str, source: Any, **kwargs):
            if isinstance(source, pd.DataFrame):
                self.con.register(name, source)
                return name
            p = Path(source)
            fn = "read_csv_auto" if p.suffix.lower() == ".csv" else "read_parquet"
            self.con.execute(f"CREATE OR REPLACE VIEW {sql_ident(name)} AS SELECT * FROM {fn}({sql_literal(p.as_posix())})")
            return name

        def query(self, sql: str) -> pd.DataFrame:
            return self.con.execute(sql).df()

    return _Fallback()
