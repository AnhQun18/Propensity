"""Fast parallel inventory & schema inspection for Parquet and general formats."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .formats import (
    SourceSpec,
    detect_format,
    extract_hive_partitions,
    normalize_source,
    resolve_files,
)
from .readers import iter_batches


def _inspect_parquet_file(args: tuple[Path, dict[str, str], Path | None]):
    path, partition_types, base_dir = args
    pf = pq.ParquetFile(path)
    fields = tuple((f.name, str(f.type)) for f in pf.schema_arrow)
    part_keys = tuple(sorted(extract_hive_partitions(path, partition_types=partition_types, base_dir=base_dir)))
    return pf.metadata.num_rows, path.stat().st_size, fields, part_keys


def inspect_source(
    source: Any,
    *,
    name: str = "dataset",
    max_workers: int = 8,
    partition_types: dict[str, str] | None = None,
    count_rows: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Inspect files, row counts, byte size, and schema of a single source."""
    spec = normalize_source(source, partition_types=partition_types or {}, **kwargs)
    fmt = spec.format

    if fmt == "parquet":
        base_dir = Path(spec.source) if isinstance(spec.source, (str, Path)) and Path(spec.source).is_dir() else None
        files = resolve_files(spec.source, fmt="parquet")
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(files)))) as pool:
            results = list(pool.map(_inspect_parquet_file, [(f, spec.partition_types, base_dir) for f in files]))
        schemas = {r[2] for r in results}
        part_sets = {r[3] for r in results}
        if len(schemas) != 1 or len(part_sets) != 1:
            raise ValueError(
                f"Inconsistent schemas ({len(schemas)} variants) or partition keys ({len(part_sets)} variants) in {name}"
            )
        fields = next(iter(schemas))
        partition_keys = list(next(iter(part_sets)))
        physical = [c for c, _ in fields]
        logical = list(dict.fromkeys(physical + partition_keys))
        return {
            "name": name,
            "format": fmt,
            "files": len(files),
            "bytes": sum(r[1] for r in results),
            "rows": sum(r[0] for r in results),
            "physical_column_count": len(physical),
            "logical_column_count": len(logical),
            "physical_columns": dict(fields),
            "partition_keys": partition_keys,
            "schema_variants": len(schemas),
        }

    # Generic path for CSV, TSV, JSON, JSONL, Feather, ORC, Excel, SQLite, DuckDB, in-memory
    files_count = 0
    total_bytes = 0
    if isinstance(spec.source, (str, Path)) and fmt not in {"sqlite", "duckdb"}:
        files = resolve_files(spec.source)
        files_count = len(files)
        total_bytes = sum(f.stat().st_size for f in files)
    elif isinstance(spec.source, (str, Path)) and Path(spec.source).is_file():
        files_count = 1
        total_bytes = Path(spec.source).stat().st_size

    rows = 0
    columns_map: dict[str, str] = {}
    for idx, batch in enumerate(iter_batches(spec, batch_size=250_000)):
        if idx == 0:
            columns_map = {str(col): str(dtype) for col, dtype in batch.dtypes.items()}
        rows += len(batch)
        if not count_rows:
            rows = -1
            break

    return {
        "name": name,
        "format": fmt,
        "files": files_count,
        "bytes": total_bytes,
        "rows": rows if count_rows else None,
        "physical_column_count": len(columns_map),
        "logical_column_count": len(columns_map),
        "physical_columns": columns_map,
        "partition_keys": [],
        "schema_variants": 1,
    }


def inspect_inventory(
    sources: Mapping[str, Any] | Any,
    *,
    max_workers: int = 8,
    partition_types: dict[str, str] | None = None,
    count_rows: bool = True,
) -> dict[str, Any]:
    """Inspect one or more named data sources and return a structured inventory summary."""
    if not isinstance(sources, Mapping):
        label = Path(sources).name if isinstance(sources, (str, Path)) else detect_format(sources)
        sources = {label: sources}

    tables: dict[str, dict[str, Any]] = {}
    all_columns: set[str] = set()
    for name, src in sources.items():
        info = inspect_source(src, name=name, max_workers=max_workers,
                              partition_types=partition_types, count_rows=count_rows)
        tables[name] = info
        all_columns.update(info["physical_columns"].keys())
        all_columns.update(info["partition_keys"])

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "table_count": len(tables),
        "total_files": sum(t["files"] for t in tables.values()),
        "total_bytes": sum(t["bytes"] for t in tables.values()),
        "total_rows": sum(t["rows"] for t in tables.values() if t["rows"] is not None),
        "distinct_columns": sorted(all_columns),
        "tables": tables,
    }
