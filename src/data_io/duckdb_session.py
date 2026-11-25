"""Disk-backed DuckDB session wrapper supporting universal multi-format views."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .formats import normalize_source, resolve_files
from .readers import load_arrow_table
from .writers import file_sha256


def sql_literal(value: Any) -> str:
    """Safely quote a Python scalar as a SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def sql_ident(name: str) -> str:
    """Safely quote a SQL identifier (column or table name)."""
    return '"' + str(name).replace('"', '""') + '"'


def normalized_id_expr(column: str = "entity_id") -> str:
    """SQL expression trimming whitespace and turning empty strings into NULL."""
    return f"nullif(trim(cast({sql_ident(column)} AS VARCHAR)), '')"


def parquet_scan_sql(
    files: Sequence[str | Path],
    *,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    filename: bool = False,
    file_row_number: bool = False,
) -> str:
    """Build a DuckDB read_parquet(...) table function call for explicit files."""
    paths = ",".join(sql_literal(Path(p).as_posix()) for p in files)
    opts = [
        f"hive_partitioning={'true' if hive_partitioning else 'false'}",
        f"union_by_name={'true' if union_by_name else 'false'}",
        f"filename={'true' if filename else 'false'}",
        f"file_row_number={'true' if file_row_number else 'false'}",
    ]
    return f"read_parquet([{paths}], {', '.join(opts)})"


class DuckDBSession:
    """Resource-bounded DuckDB connection with universal format registration & atomic export."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        memory_limit: str = "4GB",
        threads: int = 2,
        spill_dir: str | Path | None = None,
        timezone: str | None = "UTC",
        read_only: bool = False,
        preserve_insertion_order: bool = False,
    ):
        import duckdb

        self.db_path = str(db_path)
        self.con = duckdb.connect(self.db_path, read_only=read_only)
        self._registered_objects: list[Any] = []
        if timezone is not None:
            self.con.execute(f"SET TimeZone = {sql_literal(timezone)}")
        self.con.execute(f"SET memory_limit = {sql_literal(memory_limit)}")
        self.con.execute(f"SET threads = {int(threads)}")
        self.con.execute(f"SET preserve_insertion_order = {'true' if preserve_insertion_order else 'false'}")
        if spill_dir is not None:
            spill = Path(spill_dir).resolve()
            spill.mkdir(parents=True, exist_ok=True)
            self.con.execute(f"SET temp_directory = {sql_literal(spill.as_posix())}")

    def close(self) -> None:
        self.con.close()
        self._registered_objects.clear()

    def __enter__(self) -> "DuckDBSession":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def execute(self, sql: str, params: Sequence[Any] | None = None):
        return self.con.execute(sql, params) if params is not None else self.con.execute(sql)

    def query(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        return self.execute(sql, params).df()

    def assert_empty(self, sql: str, message: str) -> None:
        if self.con.execute(f"SELECT EXISTS ({sql})").fetchall()[0][0]:
            raise ValueError(message)

    def register(
        self,
        view_name: str,
        source: Any,
        *,
        hive_partitioning: bool = True,
        union_by_name: bool = False,
        filename: bool = False,
        file_row_number: bool = False,
        **kwargs: Any,
    ) -> str:
        """Register any supported file, directory, DB table, or in-memory object as a DuckDB view."""
        spec = normalize_source(source, **kwargs)
        fmt = spec.format
        v = sql_ident(view_name)

        if fmt == "parquet":
            files = resolve_files(spec.source, fmt="parquet")
            scan = parquet_scan_sql(
                files,
                hive_partitioning=hive_partitioning,
                union_by_name=union_by_name,
                filename=filename,
                file_row_number=file_row_number,
            )
            self.con.execute(f"CREATE OR REPLACE VIEW {v} AS SELECT * FROM {scan}")
            return view_name

        if fmt in {"csv", "tsv"}:
            files = resolve_files(spec.source)
            paths = ",".join(sql_literal(f.as_posix()) for f in files)
            opts = [
                f"hive_partitioning={'true' if hive_partitioning else 'false'}",
                f"union_by_name={'true' if union_by_name else 'false'}",
            ]
            if fmt == "tsv" and spec.delimiter is None:
                opts.append("delim='\\t'")
            elif spec.delimiter is not None:
                opts.append(f"delim={sql_literal(spec.delimiter)}")
            if filename:
                opts.append("filename=true")
            self.con.execute(f"CREATE OR REPLACE VIEW {v} AS SELECT * FROM read_csv_auto([{paths}], {', '.join(opts)})")
            return view_name

        if fmt in {"json", "jsonl"}:
            files = resolve_files(spec.source)
            paths = ",".join(sql_literal(f.as_posix()) for f in files)
            opts = [
                f"hive_partitioning={'true' if hive_partitioning else 'false'}",
                f"union_by_name={'true' if union_by_name else 'false'}",
            ]
            if filename:
                opts.append("filename=true")
            self.con.execute(f"CREATE OR REPLACE VIEW {v} AS SELECT * FROM read_json_auto([{paths}], {', '.join(opts)})")
            return view_name

        if fmt == "dataframe":
            self._registered_objects.append(spec.source)
            self.con.register(view_name, spec.source)
            return view_name

        if fmt == "duckdb_relation":
            spec.source.create_view(view_name, replace=True)
            return view_name

        # Fallback: materialize via unified Arrow reader (covers Feather, ORC, Excel, SQLite, DuckDB DB, Polars, records)
        arrow_table = load_arrow_table(spec)
        self._registered_objects.append(arrow_table)
        self.con.register(view_name, arrow_table)
        return view_name

    def export_parquet(
        self,
        query: str,
        path: str | Path,
        *,
        compression: str = "ZSTD",
        atomic: bool = True,
    ) -> dict[str, Any]:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging = dest.with_name(f"{dest.stem}.incomplete{dest.suffix}") if atomic else dest
        try:
            self.con.execute(
                f"COPY ({query}) TO {sql_literal(staging.as_posix())} "
                f"(FORMAT PARQUET, COMPRESSION {compression})"
            )
            digest = file_sha256(staging)
            if atomic:
                os.replace(staging, dest)
        except BaseException:
            if atomic and staging.exists():
                staging.unlink(missing_ok=True)
            raise
        return {"path": str(dest.resolve()), "bytes": dest.stat().st_size, "sha256": digest}

    def export_csv(self, query: str, path: str | Path, *, header: bool = True, atomic: bool = True) -> dict[str, Any]:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging = dest.with_name(f"{dest.stem}.incomplete{dest.suffix}") if atomic else dest
        try:
            self.con.execute(
                f"COPY ({query}) TO {sql_literal(staging.as_posix())} "
                f"(FORMAT CSV, HEADER {'true' if header else 'false'})"
            )
            digest = file_sha256(staging)
            if atomic:
                os.replace(staging, dest)
        except BaseException:
            if atomic and staging.exists():
                staging.unlink(missing_ok=True)
            raise
        return {"path": str(dest.resolve()), "bytes": dest.stat().st_size, "sha256": digest}
