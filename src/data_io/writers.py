"""Atomic multi-format table writers with SHA-256 checksums."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

from .formats import detect_format
from .readers import load_dataframe


def file_sha256(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Compute streaming SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _incomplete_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.stem}.incomplete{destination.suffix}")


def write_table(
    data: Any,
    destination: str | Path,
    *,
    format: str | None = None,
    compression: str | None = "zstd",
    atomic: bool = True,
    index: bool = False,
    table: str | None = None,
    if_exists: str = "replace",
    **write_options: Any,
) -> dict[str, Any]:
    """Write a DataFrame/Arrow Table (or any readable source) to disk safely.

    Supported formats: parquet, csv, tsv, json, jsonl, feather, excel, sqlite.
    Atomic mode writes to `<stem>.incomplete<suffix>` then atomically renames on success.
    """
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fmt = detect_format(dest, explicit_format=format)

    if isinstance(data, pa.Table):
        arrow_table = data
        frame = None
        row_count = data.num_rows
    elif isinstance(data, pd.DataFrame):
        frame = data
        arrow_table = None
        row_count = len(data)
    else:
        frame = load_dataframe(data)
        arrow_table = None
        row_count = len(frame)

    if fmt == "sqlite":
        if not table:
            raise ValueError("Writing to SQLite requires table=")
        df = frame if frame is not None else arrow_table.to_pandas()
        con = sqlite3.connect(str(dest))
        try:
            df.to_sql(table, con, if_exists=if_exists, index=index, **write_options)
        finally:
            con.close()
        return {
            "path": str(dest.resolve()),
            "format": fmt,
            "table": table,
            "rows": row_count,
            "bytes": dest.stat().st_size,
            "sha256": file_sha256(dest),
        }

    staging = _incomplete_path(dest) if atomic else dest
    try:
        if fmt == "parquet":
            tbl = arrow_table if arrow_table is not None else pa.Table.from_pandas(frame, preserve_index=index)
            pq.write_table(tbl, staging, compression=compression, **write_options)
        elif fmt == "feather":
            tbl = arrow_table if arrow_table is not None else pa.Table.from_pandas(frame, preserve_index=index)
            feather.write_feather(tbl, staging, compression=compression or "uncompressed", **write_options)
        elif fmt in {"csv", "tsv"}:
            df = frame if frame is not None else arrow_table.to_pandas()
            sep = "\t" if fmt == "tsv" else write_options.pop("sep", ",")
            df.to_csv(staging, sep=sep, index=index, **write_options)
        elif fmt == "jsonl":
            df = frame if frame is not None else arrow_table.to_pandas()
            df.to_json(staging, orient="records", lines=True, force_ascii=False, date_format="iso", **write_options)
        elif fmt == "json":
            df = frame if frame is not None else arrow_table.to_pandas()
            df.to_json(staging, orient="records", force_ascii=False, date_format="iso", indent=2, **write_options)
        elif fmt == "excel":
            df = frame if frame is not None else arrow_table.to_pandas()
            try:
                df.to_excel(staging, index=index, **write_options)
            except ImportError as exc:
                raise ImportError("Writing Excel files requires openpyxl (`pip install openpyxl`)") from exc
        else:
            raise ValueError(f"Unsupported output format: {fmt}")

        digest = file_sha256(staging)
        if atomic:
            os.replace(staging, dest)
    except BaseException:
        if atomic and staging.exists():
            staging.unlink(missing_ok=True)
        raise

    return {
        "path": str(dest.resolve()),
        "format": fmt,
        "rows": row_count,
        "bytes": dest.stat().st_size,
        "sha256": digest,
    }
