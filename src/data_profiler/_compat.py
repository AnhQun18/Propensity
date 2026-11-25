"""Optional sibling bridge to data_io with standalone fallback."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
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


def iter_batches(source: Any, *, columns: Sequence[str] | None = None, batch_size: int = 100_000, **kwargs: Any) -> Iterator[pd.DataFrame]:
    dio = _load_data_io()
    if dio is not None:
        yield from dio.iter_batches(source, columns=columns, batch_size=batch_size, **kwargs)
        return
    if isinstance(source, pd.DataFrame):
        frame = source[list(columns)] if columns is not None else source
        for i in range(0, len(frame), batch_size):
            yield frame.iloc[i:i + batch_size].reset_index(drop=True)
        return
    path = Path(source)
    if path.suffix.lower() == ".csv":
        for chunk in pd.read_csv(path, usecols=list(columns) if columns else None, chunksize=batch_size):
            yield chunk
    else:
        frame = pd.read_parquet(path, columns=list(columns) if columns else None)
        for i in range(0, len(frame), batch_size):
            yield frame.iloc[i:i + batch_size].reset_index(drop=True)


def load_dataframe(source: Any, *, columns: Sequence[str] | None = None, **kwargs: Any) -> pd.DataFrame:
    dio = _load_data_io()
    if dio is not None:
        return dio.load_dataframe(source, columns=columns, **kwargs)
    frames = list(iter_batches(source, columns=columns, **kwargs))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(columns or []))


def parse_timestamps(values, timezone: str | None = None) -> pd.Series:
    dio = _load_data_io()
    if dio is not None:
        return dio.parse_timestamps(values, timezone=timezone)
    res = pd.to_datetime(values, errors="coerce", format="mixed")
    if not isinstance(res, pd.Series):
        res = pd.Series(res)
    if res.dt.tz is not None and timezone is None:
        raise ValueError("Source contains timezone-aware timestamps; pass timezone= explicitly")
    if timezone is not None:
        res = res.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT") if res.dt.tz is None else res.dt.tz_convert(timezone)
    return res.dt.as_unit("ns")
