"""Multi-format tabular IO, streaming batches, inventory inspection, and run integrity."""

from .duckdb_session import (
    DuckDBSession,
    normalized_id_expr,
    parquet_scan_sql,
    sql_ident,
    sql_literal,
)
from .formats import (
    SourceSpec,
    detect_format,
    extract_hive_partitions,
    normalize_source,
    resolve_files,
)
from .integrity import (
    RunManifest,
    assert_non_overlapping_paths,
    compute_file_inventory,
    inventory_sha256,
    json_safe,
    verify_inventory_unchanged,
    write_json,
)
from .inventory import inspect_inventory, inspect_source
from .readers import iter_batches, load_arrow_table, load_dataframe, nullable_integer_dtype
from .time_utils import parse_timestamps
from .writers import file_sha256, write_table

__all__ = [
    "SourceSpec",
    "detect_format",
    "normalize_source",
    "resolve_files",
    "extract_hive_partitions",
    "nullable_integer_dtype",
    "iter_batches",
    "load_dataframe",
    "load_arrow_table",
    "write_table",
    "file_sha256",
    "inspect_source",
    "inspect_inventory",
    "json_safe",
    "write_json",
    "assert_non_overlapping_paths",
    "compute_file_inventory",
    "inventory_sha256",
    "verify_inventory_unchanged",
    "RunManifest",
    "parse_timestamps",
    "DuckDBSession",
    "sql_literal",
    "sql_ident",
    "normalized_id_expr",
    "parquet_scan_sql",
]
