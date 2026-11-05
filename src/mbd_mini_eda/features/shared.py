"""Ghép các tổng hợp theo nguồn vào bảng khách, giữ đúng một dòng mỗi ID."""

import numpy as np
import pandas as pd

from ..engine import PREFIXES, engine


def timestamp_series(values, timezone):
    values = pd.to_datetime(values, unit="ns", utc=timezone is not None)
    return values.dt.tz_convert(timezone) if timezone is not None else values


def customer_table(context, *, start=None, end=None):
    store = engine(context)
    aggregates = store.aggregate(start, end)
    frame = context.artifacts["clients"][["client_id", "fold"]].copy()
    for source, prefix in PREFIXES.items():
        part = aggregates[aggregates.source.eq(source)].drop(columns="source").copy()
        part["first_event"] = timestamp_series(part.pop("first_ns"), context.config.timezone)
        part["last_event"] = timestamp_series(part.pop("last_ns"), context.config.timezone)
        part["span_days"] = (part.last_event - part.first_event).dt.total_seconds() / 86400
        part["mean_gap_seconds"] = (part.last_event - part.first_event).dt.total_seconds() / (part.event_count - 1).replace(0, np.nan)
        part["amount_mean"] = part.amount_sum / part.amount_n.replace(0, np.nan)
        if end is not None:
            part["recency_days"] = (pd.Timestamp(end) - part.last_event).dt.total_seconds() / 86400
        if source != "transactions":
            part = part.drop(columns=["amount_n", "amount_sum", "amount_mean"])
        part = part.rename(columns={c: prefix + "_" + c for c in part if c != "client_id"})
        frame = frame.merge(part, on="client_id", how="left", validate="one_to_one")
        for column in (prefix+"_event_count", prefix+"_active_days", prefix+"_amount_n", prefix+"_amount_sum"):
            if column in frame:
                frame[column] = frame[column].fillna(0)
                if not column.endswith("amount_sum"):
                    frame[column] = frame[column].astype("int64")
        frame["has_"+prefix] = frame[prefix+"_event_count"].gt(0)
    frame = frame.merge(store.unique_locations(start, end), on="client_id", how="left", validate="one_to_one")
    frame["geo_unique_locations"] = frame.geo_unique_locations.fillna(0).astype("int64")
    frame = frame.merge(store.active_days(start, end), on="client_id", how="left", validate="one_to_one")
    frame["any_active_days"] = frame.any_active_days.fillna(0).astype("int64")
    return frame
