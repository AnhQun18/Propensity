"""Format detection, URI parsing, file discovery and Hive partition helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import glob
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pandas as pd
import pyarrow as pa


COMPRESSED_SUFFIXES = frozenset({".gz", ".bz2", ".xz", ".zst", ".zstd", ".zip"})

EXTENSION_TO_FORMAT: dict[str, str] = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "csv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".feather": "feather",
    ".ipc": "feather",
    ".arrow": "feather",
    ".orc": "orc",
    ".xlsx": "excel",
    ".xls": "excel",
    ".xlsm": "excel",
    ".ods": "excel",
    ".sqlite": "sqlite",
    ".sqlite3": "sqlite",
    ".db": "sqlite",
    ".duckdb": "duckdb",
    ".pkl": "pickle",
    ".pickle": "pickle",
}

FILE_FORMATS = frozenset({
    "parquet", "csv", "tsv", "json", "jsonl", "feather", "orc", "excel", "pickle",
})
DB_FORMATS = frozenset({"sqlite", "duckdb"})
IN_MEMORY_FORMATS = frozenset({"dataframe", "arrow", "polars", "duckdb_relation", "records"})


@dataclass(frozen=True)
class SourceSpec:
    """Normalized description of a tabular data source."""

    source: Any
    format: str | None = None
    table: str | None = None
    query: str | None = None
    sheet_name: str | int = 0
    delimiter: str | None = None
    encoding: str = "utf-8"
    partition_types: dict[str, str] = field(default_factory=dict)
    allow_pickle: bool = False
    read_options: dict[str, Any] = field(default_factory=dict)


def _strip_compression_suffix(path: Path) -> str:
    suffixes = [s.lower() for s in path.suffixes]
    while suffixes and suffixes[-1] in COMPRESSED_SUFFIXES:
        suffixes.pop()
    return suffixes[-1] if suffixes else ""


def parse_db_uri(value: str) -> tuple[str | None, Path | None, str | None]:
    """Parse URIs like 'sqlite:///data.db::orders' or 'data.duckdb::events'."""
    text = value.strip()
    scheme = None
    if text.startswith("sqlite:///"):
        scheme = "sqlite"
        text = text[len("sqlite:///"):]
    elif text.startswith("duckdb:///"):
        scheme = "duckdb"
        text = text[len("duckdb:///"):]

    table = None
    if "::" in text:
        path_part, table = text.rsplit("::", 1)
    else:
        path_part = text

    if scheme is not None:
        return scheme, Path(path_part), table

    ext = _strip_compression_suffix(Path(path_part))
    fmt = EXTENSION_TO_FORMAT.get(ext)
    if fmt in DB_FORMATS:
        return fmt, Path(path_part), table
    return None, None, None


def _is_polars_object(obj: Any) -> bool:
    mod = type(obj).__module__
    name = type(obj).__name__
    return mod.startswith("polars") and name in {"DataFrame", "LazyFrame"}


def _is_duckdb_relation(obj: Any) -> bool:
    mod = type(obj).__module__
    name = type(obj).__name__
    return mod.startswith("duckdb") and "Relation" in name


def has_glob_magic(text: str) -> bool:
    return any(ch in text for ch in ("*", "?", "["))


def resolve_files(source: str | Path | Sequence[str | Path], *, fmt: str | None = None) -> list[Path]:
    """Resolve a file path, directory, glob pattern, or list of paths into sorted files."""
    if isinstance(source, (list, tuple)):
        files: list[Path] = []
        for item in source:
            files.extend(resolve_files(item, fmt=fmt))
        return sorted(dict.fromkeys(files))

    text = str(source)
    if has_glob_magic(text):
        matches = sorted(Path(p) for p in glob.glob(text, recursive=True) if Path(p).is_file())
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {text}")
        return matches

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Data source not found: {path}")
    if path.is_file():
        return [path]

    candidates = sorted(
        p for p in path.rglob("*")
        if p.is_file() and not p.name.startswith((".", "_"))
        and _strip_compression_suffix(p) in EXTENSION_TO_FORMAT
    )
    if fmt is not None:
        candidates = [p for p in candidates if EXTENSION_TO_FORMAT.get(_strip_compression_suffix(p)) == fmt]
    if not candidates:
        raise ValueError(f"No supported data files found under directory: {path}")
    return candidates


def detect_format(source: Any, *, explicit_format: str | None = None) -> str:
    """Detect the format of any file, directory, glob, database URI, or in-memory object."""
    if explicit_format is not None:
        normalized = explicit_format.lower().strip()
        aliases = {"pq": "parquet", "ndjson": "jsonl", "ipc": "feather", "arrow_ipc": "feather",
                   "xlsx": "excel", "xls": "excel", "db": "sqlite", "sqlite3": "sqlite", "pkl": "pickle"}
        return aliases.get(normalized, normalized)

    if isinstance(source, SourceSpec):
        return detect_format(source.source, explicit_format=source.format)
    if isinstance(source, pd.DataFrame):
        return "dataframe"
    if isinstance(source, (pa.Table, pa.RecordBatchReader)):
        return "arrow"
    if _is_polars_object(source):
        return "polars"
    if _is_duckdb_relation(source):
        return "duckdb_relation"
    if isinstance(source, (list, tuple)):
        if not source:
            return "records"
        if isinstance(source[0], dict):
            return "records"
        if isinstance(source[0], (str, Path)):
            return detect_format(source[0])

    if isinstance(source, (str, Path)):
        text = str(source)
        db_fmt, _, _ = parse_db_uri(text)
        if db_fmt is not None:
            return db_fmt
        if has_glob_magic(text):
            files = resolve_files(text)
            return detect_format(files[0])
        path = Path(source)
        if path.is_dir():
            files = resolve_files(path)
            detected = {EXTENSION_TO_FORMAT[_strip_compression_suffix(f)] for f in files}
            if len(detected) > 1:
                raise ValueError(f"Directory {path} contains mixed formats {sorted(detected)}; specify format=")
            return next(iter(detected))
        ext = _strip_compression_suffix(path)
        if ext in EXTENSION_TO_FORMAT:
            return EXTENSION_TO_FORMAT[ext]
        raise ValueError(f"Unsupported file extension for source: {path}")

    raise TypeError(f"Unsupported data source type: {type(source).__name__}")


def normalize_source(source: Any, **kwargs: Any) -> SourceSpec:
    """Wrap any raw source + overrides into a canonical SourceSpec."""
    if isinstance(source, SourceSpec):
        fmt = kwargs.get("format") or source.format
        merged = {
            "source": source.source,
            "format": detect_format(source.source, explicit_format=fmt),
            "table": kwargs.get("table") if kwargs.get("table") is not None else source.table,
            "query": kwargs.get("query") if kwargs.get("query") is not None else source.query,
            "sheet_name": kwargs.get("sheet_name", source.sheet_name),
            "delimiter": kwargs.get("delimiter") if kwargs.get("delimiter") is not None else source.delimiter,
            "encoding": kwargs.get("encoding") or source.encoding,
            "partition_types": kwargs.get("partition_types") or source.partition_types or {},
            "allow_pickle": bool(kwargs.get("allow_pickle", source.allow_pickle)),
            "read_options": {**source.read_options, **(kwargs.get("read_options") or {})},
        }
        return SourceSpec(**merged)

    table = kwargs.get("table")
    fmt = kwargs.get("format")
    raw_source = source
    if isinstance(source, (str, Path)):
        db_fmt, db_path, uri_table = parse_db_uri(str(source))
        if db_fmt is not None and db_path is not None:
            fmt = fmt or db_fmt
            raw_source = db_path
            table = table or uri_table

    extra_options = {
        k: v for k, v in kwargs.items()
        if k not in {"format", "table", "query", "sheet_name", "delimiter", "encoding",
                     "partition_types", "allow_pickle", "read_options"}
    }
    read_options = {**(kwargs.get("read_options") or {}), **extra_options}
    return SourceSpec(
        source=raw_source,
        format=detect_format(raw_source, explicit_format=fmt),
        table=table,
        query=kwargs.get("query"),
        sheet_name=kwargs.get("sheet_name", 0),
        delimiter=kwargs.get("delimiter"),
        encoding=kwargs.get("encoding", "utf-8"),
        partition_types=dict(kwargs.get("partition_types") or {}),
        allow_pickle=bool(kwargs.get("allow_pickle", False)),
        read_options=read_options,
    )


def _cast_partition_value(key: str, raw: str, partition_types: dict[str, str]) -> Any:
    target = partition_types.get(key)
    if target is None:
        return raw
    target = target.lower()
    if target == "int":
        return int(raw)
    if target == "float":
        return float(raw)
    if target == "bool":
        low = raw.lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no"}:
            return False
        raise ValueError(f"Cannot cast partition {key}={raw!r} to bool")
    return raw


def extract_hive_partitions(
    file: Path,
    *,
    partition_types: dict[str, str] | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Extract Hive-style key=value partition columns from parent directory names."""
    types = partition_types or {}
    rel_parent = file.parent
    if base_dir is not None:
        try:
            rel_parent = file.resolve().relative_to(base_dir.resolve()).parent
        except ValueError:
            rel_parent = file.parent

    result: dict[str, Any] = {}
    for part in rel_parent.parts:
        if "=" in part:
            key, value = part.split("=", 1)
            if key:
                result[key] = _cast_partition_value(key, unquote(value), types)
    return result
