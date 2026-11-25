"""Read-only record validity, timestamp collision, and join-key EDA audits."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from ._compat import load_dataframe, normalized_id_expr, open_duckdb_session, sql_ident, write_json


def audit_invalid_records(
    source: Any,
    entities_source: Any | None = None,
    *,
    entity_col: str,
    time_col: str | None = None,
    sample_limit: int = 50,
    output_dir: str | Path | None = None,
    source_name: str = "events",
    memory_limit: str = "2GB",
    threads: int = 2,
    **source_options: Any,
) -> dict[str, Any]:
    """Read-only quality audit: classify invalid IDs/timestamps & report timestamp collisions."""
    q_ent = sql_ident(entity_col)
    ent_expr = normalized_id_expr(entity_col)

    with open_duckdb_session(memory_limit=memory_limit, threads=threads) as db:
        db.register("raw_events", source, **source_options)
        has_ref = entities_source is not None
        if has_ref:
            db.register("raw_entities", entities_source)
            db.execute(
                f"CREATE TEMP TABLE ref_entities AS SELECT DISTINCT {ent_expr} AS __eid "
                f"FROM raw_entities WHERE {ent_expr} IS NOT NULL"
            )

        time_cases = ""
        time_select = "NULL::TIMESTAMP AS __norm_ts"
        if time_col is not None:
            q_time = sql_ident(time_col)
            time_select = f"try_cast({q_time} AS TIMESTAMP) AS __norm_ts"
            time_cases = f"WHEN try_cast({q_time} AS TIMESTAMP) IS NULL OR NOT isfinite(try_cast({q_time} AS TIMESTAMP)) THEN 'invalid_timestamp'"

        ref_case = (
            f"WHEN NOT EXISTS (SELECT 1 FROM ref_entities e WHERE e.__eid = {ent_expr}) THEN 'unknown_entity_id'"
            if has_ref
            else ""
        )

        db.execute(
            f"""CREATE TEMP VIEW classified AS
                SELECT r.*,
                       {ent_expr} AS __norm_eid,
                       {time_select},
                       CASE
                           WHEN {ent_expr} IS NULL THEN 'missing_entity_id'
                           {time_cases}
                           {ref_case}
                           ELSE NULL
                       END AS rejection_reason
                FROM raw_events r"""
        )

        counts = db.execute(
            "SELECT count(*), count(*) FILTER (WHERE rejection_reason IS NOT NULL) FROM classified"
        ).fetchone()
        breakdown = db.query(
            "SELECT rejection_reason, count(*) AS rows FROM classified "
            "WHERE rejection_reason IS NOT NULL GROUP BY 1 ORDER BY 2 DESC"
        )
        invalid_sample = db.query(
            f"SELECT * EXCLUDE(__norm_eid, __norm_ts) FROM classified WHERE rejection_reason IS NOT NULL LIMIT {int(sample_limit)}"
        )

        collisions = pd.DataFrame()
        first_ts = last_ts = None
        if time_col is not None:
            bounds = db.execute(
                "SELECT min(__norm_ts), max(__norm_ts) FROM classified WHERE rejection_reason IS NULL"
            ).fetchone()
            first_ts = bounds[0].isoformat() if bounds[0] is not None else None
            last_ts = bounds[1].isoformat() if bounds[1] is not None else None
            collisions = db.query(
                f"SELECT __norm_eid AS {q_ent}, __norm_ts AS {sql_ident(time_col)}, count(*) AS n_events "
                "FROM classified WHERE rejection_reason IS NULL GROUP BY 1, 2 HAVING count(*) > 1 ORDER BY n_events DESC"
            )

    summary = {
        "source": source_name,
        "raw_rows": int(counts[0]),
        "invalid_rows": int(counts[1]),
        "valid_rows": int(counts[0] - counts[1]),
        "invalid_pct": float(counts[1] / counts[0] * 100.0) if counts[0] else 0.0,
        "rejection_breakdown": dict(zip(breakdown["rejection_reason"], [int(x) for x in breakdown["rows"]])),
        "first_valid_timestamp": first_ts,
        "last_valid_timestamp": last_ts,
        "timestamp_collision_groups": int(len(collisions)),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / f"{source_name}_invalid_audit.json", summary)
        if not breakdown.empty:
            breakdown.to_csv(out / f"{source_name}_rejection_breakdown.csv", index=False)
        if not collisions.empty:
            collisions.to_csv(out / f"{source_name}_timestamp_collisions.csv", index=False)

    return {
        "summary": summary,
        "rejection_breakdown": breakdown,
        "invalid_sample": invalid_sample,
        "timestamp_collisions": collisions,
    }


def audit_join_keys(
    entities_source: Any,
    panel_source: Any,
    *,
    entity_col: str,
    period_col: str | None = None,
    group_col: str | None = None,
    target_cols: Sequence[str] = (),
) -> dict[str, Any]:
    """Verify key uniqueness, referential integrity, group consistency, and label overlaps."""
    ent = load_dataframe(entities_source)
    pan = load_dataframe(panel_source)
    ent_ids = ent[entity_col].astype("string")
    pan_ids = pan[entity_col].astype("string")

    ent_dup = int(ent_ids.duplicated().sum())
    ent_null = int(ent_ids.isna().sum())
    pan_null = int(pan_ids.isna().sum())

    key_cols = [entity_col, period_col] if period_col else [entity_col]
    pan_dup = int(pan.duplicated(key_cols).sum())
    orphan_rows = int((~pan_ids.isin(set(ent_ids.dropna()))).sum())

    group_mismatches = None
    if group_col and group_col in ent.columns and group_col in pan.columns:
        merged = pan[[entity_col, group_col]].merge(
            ent[[entity_col, group_col]].drop_duplicates(entity_col),
            on=entity_col, how="left", suffixes=("_panel", "_entity"),
        )
        matched = merged[f"{group_col}_entity"].notna()
        group_mismatches = int((matched & merged[f"{group_col}_panel"].ne(merged[f"{group_col}_entity"])).sum())

    periods_dist = None
    if period_col and period_col in pan.columns:
        periods_dist = {
            str(k): int(v)
            for k, v in pan.groupby(entity_col)[period_col].nunique().value_counts().sort_index().items()
        }

    target_summary = {}
    pair_overlap = {}
    if target_cols:
        y = pan[list(target_cols)].eq(1)
        target_summary = {
            c: {
                "positive_n": int(y[c].sum()),
                "labelled_n": int(pan[c].notna().sum()),
                "positive_rate": float(y[c].sum() / pan[c].notna().sum()) if pan[c].notna().sum() else None,
            }
            for c in target_cols
        }
        pair_overlap = {
            f"{a} & {b}": int((y[a] & y[b]).sum())
            for a, b in combinations(target_cols, 2)
        }

    return {
        "entity_rows": int(len(ent)),
        "unique_entities": int(ent_ids.nunique()),
        "entity_duplicate_keys": ent_dup,
        "entity_null_keys": ent_null,
        "panel_rows": int(len(pan)),
        "panel_duplicate_keys": pan_dup,
        "panel_null_keys": pan_null,
        "orphan_panel_rows": orphan_rows,
        "group_mismatches": group_mismatches,
        "periods_per_entity_counts": periods_dist,
        "target_summary": target_summary,
        "target_pair_overlap": pair_overlap,
    }
