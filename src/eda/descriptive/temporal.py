"""Explicit time parsing and counts on a timezone-aware period grid."""
import pandas as pd


def parse_time(series, options=None):
    options = dict(options or {})
    timezone = options.pop("timezone", None)
    if "errors" in options or "utc" in options:
        raise ValueError("Use timezone; errors and utc are managed by the parser")
    unit = options.get("unit")
    if unit:
        parsed = pd.to_datetime(series, errors="coerce", utc=True, **options)
        return parsed.dt.tz_convert(timezone) if timezone else parsed
    parsed = pd.to_datetime(series, errors="coerce", **options)
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        raise ValueError("Mixed timezones require normalized timestamps before EDA")
    if parsed.dt.tz is None and timezone:
        parsed = parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    elif parsed.dt.tz is not None and timezone:
        parsed = parsed.dt.tz_convert(timezone)
    return parsed


def summarize(frame, datetime_columns, frequency="D", max_periods=100_000):
    summaries, counts = [], []
    for column, options in datetime_columns.items():
        parsed = parse_time(frame[column], options)
        valid = parsed.dropna()
        gaps = 0
        if len(valid):
            offset = pd.tseries.frequencies.to_offset(frequency)
            if offset.n <= 0:
                raise ValueError("frequency must advance time")
            # Bound the grid before resampling very distant timestamps.
            start = valid.min().as_unit("us")
            limit = start + offset * max_periods
            if valid.max() >= limit:
                raise ValueError("Time grid exceeds max_periods; use a coarser frequency")
            grouped = pd.Series(1, index=valid).resample(frequency).sum()
            gaps = int(grouped.eq(0).sum())
            counts.extend(dict(column=column, period=t.isoformat(), events=int(n)) for t, n in grouped.items())
        summaries.append(dict(column=column, valid_n=len(valid), missing_n=int(frame[column].isna().sum()),
                              invalid_n=int((parsed.isna() & frame[column].notna()).sum()),
                              start=valid.min() if len(valid) else None, end=valid.max() if len(valid) else None,
                              timezone=str(parsed.dt.tz), frequency=frequency, empty_periods=gaps))
    return pd.DataFrame(summaries), pd.DataFrame(counts)
