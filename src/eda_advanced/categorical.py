"""Periodic categorical concentration, Shannon entropy, HHI, novelty, and main-item switching."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._compat import open_duckdb_session, sql_ident, sql_literal


def periodic_categorical_profile(
    source: Any,
    *,
    entity_col: str,
    time_col: str,
    category_col: str,
    period: str = "month",
    tz_offset_hours: int = 0,
    memory_limit: str = "2GB",
    threads: int = 2,
    **source_options: Any,
) -> pd.DataFrame:
    """Compute per-(entity, period) categorical diversity, concentration, novelty, and switching."""
    if period not in {"day", "week", "month", "quarter", "year"}:
        raise ValueError("period must be one of: day, week, month, quarter, year")

    q_ent = sql_ident(entity_col)
    q_time = sql_ident(time_col)
    q_cat = sql_ident(category_col)
    ts_expr = (
        f"try_cast({q_time} AS TIMESTAMP) + INTERVAL '{int(tz_offset_hours)} hours'"
        if tz_offset_hours
        else f"try_cast({q_time} AS TIMESTAMP)"
    )

    with open_duckdb_session(memory_limit=memory_limit, threads=threads) as db:
        db.register("raw_cat_events", source, **source_options)
        profile = db.query(
            f"""WITH cell_counts AS (
                    SELECT cast({q_ent} AS VARCHAR) AS entity_id,
                           date_trunc({sql_literal(period)}, {ts_expr}) AS period,
                           cast({q_cat} AS VARCHAR) AS item,
                           count(*) AS events
                    FROM raw_cat_events
                    WHERE {q_ent} IS NOT NULL AND {ts_expr} IS NOT NULL AND {q_cat} IS NOT NULL
                    GROUP BY 1, 2, 3
                ),
                enriched AS (
                    SELECT *,
                           sum(events) OVER (PARTITION BY entity_id, period) AS total,
                           min(period) OVER (PARTITION BY entity_id, item) AS first_item_period,
                           row_number() OVER (PARTITION BY entity_id, period ORDER BY events DESC, item ASC) AS rnk
                    FROM cell_counts
                )
                SELECT entity_id,
                       period,
                       max(total) AS events,
                       count(*) AS unique_items,
                       max(events)::DOUBLE / max(total) AS top_share,
                       -sum((events::DOUBLE / total) * ln(events::DOUBLE / total)) AS entropy,
                       sum(pow(events::DOUBLE / total, 2)) AS hhi,
                       count(*) FILTER (WHERE first_item_period = period) AS new_items,
                       sum(CASE WHEN first_item_period = period THEN events ELSE 0 END)::DOUBLE / max(total) AS new_item_event_share,
                       max(CASE WHEN rnk = 1 THEN item END) AS main_item
                FROM enriched
                GROUP BY 1, 2
                ORDER BY 1, 2"""
        )

    if profile.empty:
        profile["changed_main"] = pd.Series(dtype=float)
        return profile.rename(columns={"entity_id": entity_col})

    profile["period"] = pd.to_datetime(profile["period"])
    profile = profile.sort_values(["entity_id", "period"]).reset_index(drop=True)
    prev_main = profile.groupby("entity_id")["main_item"].shift(1)
    prev_period = profile.groupby("entity_id")["period"].shift(1)

    if period == "month":
        consecutive = profile["period"].eq(prev_period + pd.offsets.MonthBegin(1))
    elif period == "day":
        consecutive = profile["period"].eq(prev_period + pd.Timedelta(days=1))
    elif period == "week":
        consecutive = profile["period"].eq(prev_period + pd.Timedelta(weeks=1))
    else:
        consecutive = prev_period.notna()

    profile["changed_main"] = np.where(consecutive, profile["main_item"].ne(prev_main).astype(float), np.nan)
    return profile.rename(columns={"entity_id": entity_col})
