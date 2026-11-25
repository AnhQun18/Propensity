"""Declarative schema and configuration models for general dataset profiling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ._compat import iter_batches


@dataclass(frozen=True)
class TableSpec:
    """Column-role declaration for any tabular, entity, target, or event table."""

    name: str
    source: Any
    role: Literal["entity", "target", "event", "tabular"] = "tabular"
    row_unit: str = "row"
    entity_key: str | None = None
    unique_key: tuple[str, ...] = ()
    time_column: str | None = None
    numeric_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    target_columns: tuple[str, ...] = ()
    vector_columns: tuple[str, ...] = ()
    vector_dims: dict[str, int] = field(default_factory=dict)
    group_columns: tuple[str, ...] = ()
    batch_size: int | None = None
    source_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSpec:
    """Multi-table or single-table dataset specification."""

    name: str
    tables: tuple[TableSpec, ...]
    entity_table: str | None = None
    target_table: str | None = None
    timezone: str | None = None

    def table_map(self) -> dict[str, TableSpec]:
        return {t.name: t for t in self.tables}

    def event_tables(self) -> tuple[TableSpec, ...]:
        return tuple(t for t in self.tables if t.role in {"event", "tabular"})


@dataclass(frozen=True)
class ProfilerConfig:
    """Resource and sampling controls for single-pass out-of-core profiling."""

    batch_size: int = 250_000
    sqlite_cache_mb: int = 256
    random_seed: int = 42
    sample_size: int = 5_000
    max_categories: int = 20
    duplicate_mode: Literal["exact", "sample", "off"] = "sample"
    max_batches_per_source: int | None = None
    work_dir: Path | None = None
    timezone: str | None = None

    def __post_init__(self):
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        for key in ("batch_size", "sqlite_cache_mb", "sample_size", "max_categories"):
            val = getattr(self, key)
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.duplicate_mode not in {"exact", "sample", "off"}:
            raise ValueError("duplicate_mode must be 'exact', 'sample', or 'off'")
        if self.work_dir is not None:
            object.__setattr__(self, "work_dir", Path(self.work_dir))
        if self.max_batches_per_source is not None:
            val = self.max_batches_per_source
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ValueError("max_batches_per_source must be a positive integer or None")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["work_dir"] = str(self.work_dir) if self.work_dir is not None else None
        return result


@dataclass
class SectionResult:
    """Aggregated tables, textual findings, and scope metadata for one profiling section."""

    section: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileResult:
    """Complete output of a profiling run across all sections."""

    dataset_name: str
    config: ProfilerConfig
    sections: dict[str, SectionResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    erd_edges: list[dict[str, str]] = field(default_factory=list)


def _is_vector_series(series: pd.Series) -> tuple[bool, int | None]:
    non_null = series.dropna()
    if non_null.empty:
        return False, None
    sample = non_null.iloc[0]
    if isinstance(sample, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(sample, dtype=float)
            if arr.ndim == 1:
                return True, int(len(arr))
        except (TypeError, ValueError):
            pass
    return False, None


def auto_infer_table_spec(
    source: Any,
    *,
    name: str = "table",
    role: Literal["entity", "target", "event", "tabular"] = "tabular",
    row_unit: str = "row",
    entity_key: str | None = None,
    time_column: str | None = None,
    target_columns: tuple[str, ...] = (),
    sample_rows: int = 2_000,
    source_options: dict[str, Any] | None = None,
) -> TableSpec:
    """Infer numeric, categorical, datetime, and vector column roles from the first batch."""
    opts = dict(source_options or {})
    gen = iter_batches(source, batch_size=sample_rows, **opts)
    try:
        first_batch = next(gen, None)
    finally:
        gen.close()
    if first_batch is None:
        return TableSpec(name=name, source=source, role=role, row_unit=row_unit,
                         entity_key=entity_key, time_column=time_column,
                         target_columns=tuple(target_columns), source_options=opts)

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    vector_cols: list[str] = []
    vector_dims: dict[str, int] = {}
    inferred_time = time_column

    reserved = set(target_columns)
    if entity_key:
        reserved.add(entity_key)

    for col in first_batch.columns:
        if col in reserved:
            continue
        series = first_batch[col]
        if col == inferred_time:
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            if inferred_time is None:
                inferred_time = col
            continue
        if pd.api.types.is_bool_dtype(series):
            categorical_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue
        is_vec, dim = _is_vector_series(series)
        if is_vec:
            vector_cols.append(col)
            if dim is not None:
                vector_dims[col] = dim
            continue
        categorical_cols.append(col)

    return TableSpec(
        name=name,
        source=source,
        role=role,
        row_unit=row_unit,
        entity_key=entity_key,
        time_column=inferred_time,
        numeric_columns=tuple(numeric_cols),
        categorical_columns=tuple(categorical_cols),
        target_columns=tuple(target_columns),
        vector_columns=tuple(vector_cols),
        vector_dims=vector_dims,
        source_options=opts,
    )
