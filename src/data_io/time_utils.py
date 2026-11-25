"""Explicit timezone-safe timestamp parsing helpers."""

from __future__ import annotations

import pandas as pd


def parse_timestamps(values, timezone: str | None = None, *, unit: str = "ns") -> pd.Series:
    """Parse timestamps without silently mixing naive and tz-aware conventions.

    - If values contain tz-aware timestamps and `timezone` is None, raises ValueError.
    - If `timezone` is provided: naive inputs are localized, tz-aware inputs are converted.
    """
    result = pd.to_datetime(values, errors="coerce", format="mixed")
    if not isinstance(result, pd.Series):
        result = pd.Series(result)
    if not hasattr(result, "dt"):
        raise ValueError("Timestamps must share a single timezone convention")
    tz = result.dt.tz
    if tz is not None and timezone is None:
        raise ValueError("Source contains timezone-aware timestamps; pass timezone= explicitly")
    if timezone is not None:
        result = (
            result.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
            if tz is None
            else result.dt.tz_convert(timezone)
        )
    return result.dt.as_unit(unit)
