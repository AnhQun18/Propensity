"""Unified multi-format readers: streaming batches, DataFrame, and Arrow Table."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

from .formats import (
    SourceSpec,
    extract_hive_partitions,
    normalize_source,
    resolve_files,
)


def nullable_integer_dtype(pa_type: pa.DataType):
    """Preserve exact integer codes across null-containing Parquet batches."""
    if pa.types.is_signed_integer(pa_type):
        return pd.Int64Dtype()
    if pa.types.is_unsigned_integer(pa_type):
        return pd.UInt64Dtype()
    return None


def _select_columns(frame: pd.DataFrame, columns: Sequence[str] | None, source_label: str) -> pd.DataFrame:
    if columns is None:
        return frame
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns {missing} in {source_label}")
    return frame[list(columns)]


def _attach_partitions(frame: pd.DataFrame, part: dict[str, Any], file: Path) -> pd.DataFrame:
    if not part:
        return frame
    frame = frame.copy()
    for key, value in part.items():
        if key in frame.columns and not frame[key].eq(value).fillna(False).all():
            raise ValueError(f"Partition {key}={value!r} conflicts with column content in {file}")
        frame[key] = value
    return frame


def _slice_dataframe(frame: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    if frame.empty:
        return
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start:start + batch_size].reset_index(drop=True)


def _iter_parquet(spec: SourceSpec, columns: Sequence[str] | None, batch_size: int) -> Iterator[pd.DataFrame]:
    base_dir = Path(spec.source) if isinstance(spec.source, (str, Path)) and Path(spec.source).is_dir() else None
    files = resolve_files(spec.source, fmt="parquet")
    schema = None
    logical_names = None
    for file in files:
        reader = pq.ParquetFile(file)
        current = reader.schema_arrow.remove_metadata()
        part = extract_hive_partitions(file, partition_types=spec.partition_types, base_dir=base_dir)
        names = set(current.names) | set(part)
        if schema is not None and (not current.equals(schema) or names != logical_names):
            raise ValueError(f"Parquet schema drift across shards at: {file}")
        schema, logical_names = current, names
        if columns is not None and (set(columns) - names):
            raise ValueError(f"Missing columns {sorted(set(columns) - names)} in {file}")
        physical = None if columns is None else [c for c in columns if c in current.names]
        for batch in reader.iter_batches(batch_size=batch_size, columns=physical):
            frame = batch.to_pandas(types_mapper=nullable_integer_dtype)
            frame = _attach_partitions(frame, part, file)
            yield _select_columns(frame, columns, str(file))


def _iter_delimited(spec: SourceSpec, columns: Sequence[str] | None, batch_size: int) -> Iterator[pd.DataFrame]:
    base_dir = Path(spec.source) if isinstance(spec.source, (str, Path)) and Path(spec.source).is_dir() else None
    files = resolve_files(spec.source)
    sep = spec.delimiter if spec.delimiter is not None else ("\t" if spec.format == "tsv" else ",")
    for file in files:
        part = extract_hive_partitions(file, partition_types=spec.partition_types, base_dir=base_dir)
        usecols = None
        if columns is not None:
            needed = [c for c in columns if c not in part]
            usecols = needed if needed else None
        opts = {"sep": sep, "encoding": spec.encoding, "chunksize": batch_size, **spec.read_options}
        if usecols is not None and "usecols" not in opts:
            opts["usecols"] = usecols
        with pd.read_csv(file, **opts) as reader:
            for chunk in reader:
                chunk = _attach_partitions(chunk, part, file)
                yield _select_columns(chunk, columns, str(file))


def _iter_json(spec: SourceSpec, columns: Sequence[str] | None, batch_size: int) -> Iterator[pd.DataFrame]:
    base_dir = Path(spec.source) if isinstance(spec.source, (str, Path)) and Path(spec.source).is_dir() else None
    files = resolve_files(spec.source)
    is_lines = spec.format == "jsonl"
    for file in files:
        part = extract_hive_partitions(file, partition_types=spec.partition_types, base_dir=base_dir)
        if is_lines:
            opts = {"lines": True, "chunksize": batch_size, "encoding": spec.encoding, **spec.read_options}
            with pd.read_json(file, **opts) as reader:
                for chunk in reader:
                    chunk = _attach_partitions(chunk, part, file)
                    yield _select_columns(chunk, columns, str(file))
        else:
            opts = {"encoding": spec.encoding, **spec.read_options}
            frame = pd.read_json(file, **opts)
            frame = _attach_partitions(frame, part, file)
            frame = _select_columns(frame, columns, str(file))
            yield from _slice_dataframe(frame, batch_size)


def _iter_feather(spec: SourceSpec, columns: Sequence[str] | None, batch_size: int) -> Iterator[pd.DataFrame]:
    base_dir = Path(spec.source) if isinstance(spec.source, (str, Path)) and Path(spec.source).is_dir() else None
    files = resolve_files(spec.source, fmt="feather")
    for file in files:
        part = extract_hive_partitions(file, partition_types=spec.partition_types, base_dir=base_dir)
        physical = None if columns is None else [c for c in columns if c not in part]
        table = feather.read_table(file, columns=physical)
        for batch in table.to_batches(max_chunksize=batch_size):
            frame = batch.to_pandas(types_mapper=nullable_integer_dtype)
            frame = _attach_partitions(frame, part, file)
            yield _select_columns(frame, columns, str(file))


def _iter_orc(spec: SourceSpec, columns: Sequence[str] | None, batch_size: int) -> Iterator[pd.DataFrame]:
    import pyarrow.orc as orc

    base_dir = Path(spec.source) if isinstance(spec.source, (str, Path)) and Path(spec.source).is_dir() else None
    files = resolve_files(spec.source, fmt="orc")
    for file in files:
        part = extract_hive_partitions(file, partition_types=spec.partition_types, base_dir=base_dir)
        physical = None if columns is None else [c for c in columns if c not in part]
        reader = orc.ORCFile(file)
        for stripe_idx in range(reader.nstripes):
            table = reader.read_stripe(stripe_idx, columns=physical)
            for batch in table.to_batches(max_chunksize=batch_size):
                frame = batch.to_pandas(types_mapper=nullable_integer_dtype)
                frame = _attach_partitions(frame, part, file)
                yield _select_columns(frame, columns, str(file))


def _iter_excel(spec: SourceSpec, columns: Sequence[str] | None, batch_size: int) -> Iterator[pd.DataFrame]:
    files = resolve_files(spec.source, fmt="excel")
    for file in files:
        try:
            frame = pd.read_excel(file, sheet_name=spec.sheet_name, **spec.read_options)
        except ImportError as exc:
            raise ImportError(
                f"Reading Excel file {file} requires an Excel engine (e.g. `pip install openpyxl` for .xlsx)"
            ) from exc
        if isinstance(frame, dict):
            raise ValueError("sheet_name=None returned multiple sheets; specify a single sheet_name")
        frame = _select_columns(frame, columns, str(file))
        yield from _slice_dataframe(frame, batch_size)


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _iter_sqlite(spec: SourceSpec, columns: Sequence[str] | None, batch_size: int) -> Iterator[pd.DataFrame]:
    path = Path(spec.source)
    if not path.exists():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    con = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        if spec.query:
            sql = spec.query
        else:
            table = spec.table
            if not table:
                tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()]
                if len(tables) != 1:
                    raise ValueError(f"Specify table= for SQLite DB {path}; found tables: {tables}")
                table = tables[0]
            col_sql = ", ".join(_quote_ident(c) for c in columns) if columns is not None else "*"
            sql = f"SELECT {col_sql} FROM {_quote_ident(table)}"
        cursor = con.execute(sql)
        col_names = [desc[0] for desc in cursor.description]
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            frame = pd.DataFrame(rows, columns=col_names)
            yield _select_columns(frame, columns, f"{path}::{spec.table or 'query'}")
    finally:
        con.close()


def _iter_duckdb(spec: SourceSpec, columns: Sequence[str] | None, batch_size: int) -> Iterator[pd.DataFrame]:
    import duckdb

    path = Path(spec.source)
    if not path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {path}")
    con = duckdb.connect(str(path), read_only=True)
    try:
        if spec.query:
            sql = spec.query
        else:
            table = spec.table
            if not table:
                tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
                if len(tables) != 1:
                    raise ValueError(f"Specify table= for DuckDB DB {path}; found tables: {tables}")
                table = tables[0]
            col_sql = ", ".join(_quote_ident(c) for c in columns) if columns is not None else "*"
            sql = f"SELECT {col_sql} FROM {_quote_ident(table)}"
        rel = con.sql(sql)
        if hasattr(rel, "to_arrow_reader"):
            for batch in rel.to_arrow_reader(batch_size=batch_size):
                frame = batch.to_pandas(types_mapper=nullable_integer_dtype)
                yield _select_columns(frame, columns, f"{path}::{spec.table or 'query'}")
        else:
            cursor = con.execute(sql)
            col_names = [desc[0] for desc in cursor.description]
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                frame = pd.DataFrame(rows, columns=col_names)
                yield _select_columns(frame, columns, f"{path}::{spec.table or 'query'}")
    finally:
        con.close()


def _iter_pickle(spec: SourceSpec, columns: Sequence[str] | None, batch_size: int) -> Iterator[pd.DataFrame]:
    if not spec.allow_pickle:
        raise PermissionError(
            "Pickle files can execute arbitrary code on load. Pass allow_pickle=True only for trusted local files."
        )
    files = resolve_files(spec.source, fmt="pickle")
    for file in files:
        obj = pd.read_pickle(file)
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"Pickle file {file} does not contain a pandas DataFrame")
        frame = _select_columns(obj, columns, str(file))
        yield from _slice_dataframe(frame, batch_size)


def iter_batches(
    source: Any,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 100_000,
    format: str | None = None,
    table: str | None = None,
    query: str | None = None,
    sheet_name: str | int = 0,
    delimiter: str | None = None,
    encoding: str = "utf-8",
    partition_types: dict[str, str] | None = None,
    allow_pickle: bool = False,
    **read_options: Any,
) -> Iterator[pd.DataFrame]:
    """Stream pandas DataFrame batches from any supported file, directory, DB, or in-memory object."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    spec = normalize_source(
        source,
        format=format,
        table=table,
        query=query,
        sheet_name=sheet_name,
        delimiter=delimiter,
        encoding=encoding,
        partition_types=partition_types,
        allow_pickle=allow_pickle,
        read_options=read_options,
    )
    fmt = spec.format
    if fmt == "parquet":
        yield from _iter_parquet(spec, columns, batch_size)
    elif fmt in {"csv", "tsv"}:
        yield from _iter_delimited(spec, columns, batch_size)
    elif fmt in {"json", "jsonl"}:
        yield from _iter_json(spec, columns, batch_size)
    elif fmt == "feather":
        yield from _iter_feather(spec, columns, batch_size)
    elif fmt == "orc":
        yield from _iter_orc(spec, columns, batch_size)
    elif fmt == "excel":
        yield from _iter_excel(spec, columns, batch_size)
    elif fmt == "sqlite":
        yield from _iter_sqlite(spec, columns, batch_size)
    elif fmt == "duckdb":
        yield from _iter_duckdb(spec, columns, batch_size)
    elif fmt == "pickle":
        yield from _iter_pickle(spec, columns, batch_size)
    elif fmt == "dataframe":
        frame = _select_columns(spec.source, columns, "DataFrame")
        yield from _slice_dataframe(frame, batch_size)
    elif fmt == "arrow":
        obj = spec.source
        if isinstance(obj, pa.RecordBatchReader):
            for batch in obj:
                frame = batch.to_pandas(types_mapper=nullable_integer_dtype)
                yield _select_columns(frame, columns, "RecordBatchReader")
        else:
            if columns is not None:
                missing = sorted(set(columns) - set(obj.column_names))
                if missing:
                    raise ValueError(f"Missing columns {missing} in Arrow Table")
                obj = obj.select(list(columns))
            for batch in obj.to_batches(max_chunksize=batch_size):
                yield batch.to_pandas(types_mapper=nullable_integer_dtype)
    elif fmt == "polars":
        obj = spec.source
        if type(obj).__name__ == "LazyFrame":
            obj = obj.collect()
        arrow_table = obj.to_arrow()
        yield from iter_batches(arrow_table, columns=columns, batch_size=batch_size)
    elif fmt == "duckdb_relation":
        reader = spec.source.fetch_arrow_reader(batch_size=batch_size)
        for batch in reader:
            frame = batch.to_pandas(types_mapper=nullable_integer_dtype)
            yield _select_columns(frame, columns, "DuckDBRelation")
    elif fmt == "records":
        frame = pd.DataFrame(spec.source)
        if not frame.empty:
            frame = _select_columns(frame, columns, "records")
            yield from _slice_dataframe(frame, batch_size)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def load_dataframe(source: Any, *, columns: Sequence[str] | None = None, batch_size: int = 250_000, **kwargs: Any) -> pd.DataFrame:
    """Materialize any supported source into a single pandas DataFrame."""
    frames = list(iter_batches(source, columns=columns, batch_size=batch_size, **kwargs))
    if not frames:
        if isinstance(source, pd.DataFrame):
            return _select_columns(source.iloc[:0].copy(), columns, "DataFrame")
        if isinstance(source, pa.Table):
            empty = source.slice(0, 0).to_pandas(types_mapper=nullable_integer_dtype)
            return _select_columns(empty, columns, "ArrowTable")
        return pd.DataFrame(columns=list(columns) if columns is not None else [])
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    return pd.concat(frames, ignore_index=True)


def load_arrow_table(source: Any, *, columns: Sequence[str] | None = None, **kwargs: Any) -> pa.Table:
    """Materialize any supported source into a PyArrow Table."""
    if isinstance(source, pa.Table) and not kwargs:
        return source.select(list(columns)) if columns is not None else source
    frame = load_dataframe(source, columns=columns, **kwargs)
    return pa.Table.from_pandas(frame, preserve_index=False)
