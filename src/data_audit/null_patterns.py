"""Null cells, affected rows, joint co-missingness patterns, and group breakdowns."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ._compat import open_duckdb_session, sql_ident, write_json


def audit_null_patterns(
    source: Any,
    *,
    group_by: Sequence[str] = (),
    exclude_from_physical: Sequence[str] = (),
    output_dir: str | Path | None = None,
    memory_limit: str = "2GB",
    threads: int = 2,
    **source_options: Any,
) -> dict[str, Any]:
    """Compute column null rates, null-count-per-row distribution, and joint co-missingness patterns."""
    with open_duckdb_session(memory_limit=memory_limit, threads=threads) as db:
        db.register("src_table", source, **source_options)
        cols = [row[0] for row in db.execute("DESCRIBE src_table").fetchall()]
        if not cols:
            raise ValueError("Input table has no columns")

        null_count_expr = " + ".join(f"CAST({sql_ident(c)} IS NULL AS INTEGER)" for c in cols)
        per_col_expr = ", ".join(f"count(*) FILTER (WHERE {sql_ident(c)} IS NULL)" for c in cols)
        row = db.execute(
            f"SELECT count(*), {per_col_expr}, count(*) FILTER (WHERE ({null_count_expr}) > 0) FROM src_table"
        ).fetchone()
        total_rows = int(row[0])
        col_nulls = [int(v) for v in row[1:-1]]
        rows_any_null = int(row[-1])

        per_column = pd.DataFrame([
            {
                "column": c,
                "rows": total_rows,
                "null_n": v,
                "null_pct": (100.0 * v / total_rows) if total_rows else 0.0,
            }
            for c, v in zip(cols, col_nulls)
        ])

        null_count_per_row = db.query(
            f"SELECT ({null_count_expr}) AS null_columns, count(*) AS rows "
            f"FROM src_table GROUP BY 1 ORDER BY 1"
        )
        null_count_per_row["row_pct"] = (null_count_per_row["rows"] / total_rows * 100.0) if total_rows else 0.0

        nullable_cols = [c for c, v in zip(cols, col_nulls) if v > 0]
        if nullable_cols:
            escaped_parts = [
                f"CASE WHEN {sql_ident(c)} IS NULL THEN '{c.replace(chr(39), chr(39)*2)}' ELSE '' END"
                for c in nullable_cols
            ]
            pattern_expr = " || ',' || ".join(escaped_parts)
            joint_patterns = db.query(
                f"SELECT trim(regexp_replace({pattern_expr}, ',+', ','), ',') AS null_pattern, "
                f"count(*) AS rows FROM src_table GROUP BY 1 ORDER BY 2 DESC"
            )
            joint_patterns["null_pattern"] = joint_patterns["null_pattern"].replace("", "__COMPLETE__")
        else:
            joint_patterns = pd.DataFrame([{"null_pattern": "__COMPLETE__", "rows": total_rows}])
        joint_patterns["row_pct"] = (joint_patterns["rows"] / total_rows * 100.0) if total_rows else 0.0

        by_group: dict[str, pd.DataFrame] = {}
        for grp in group_by:
            if grp not in cols:
                raise ValueError(f"group_by column {grp!r} not found in columns {cols}")
            tbl = db.query(
                f"SELECT {sql_ident(grp)}, count(*) AS rows, "
                f"count(*) FILTER (WHERE ({null_count_expr}) > 0) AS rows_any_null, {per_col_expr} "
                f"FROM src_table GROUP BY 1 ORDER BY 1"
            )
            tbl.columns = [grp, "rows", "rows_any_null"] + [f"{c}_null_n" for c in cols]
            tbl["rows_any_null_pct"] = tbl["rows_any_null"] / tbl["rows"].replace(0, pd.NA) * 100.0
            by_group[grp] = tbl

    physical_cols = [c for c in cols if c not in set(exclude_from_physical)]
    total_null_cells = sum(col_nulls)
    physical_null_cells = sum(v for c, v in zip(cols, col_nulls) if c in physical_cols)
    summary = {
        "rows": total_rows,
        "columns": len(cols),
        "physical_columns": len(physical_cols),
        "null_cells": total_null_cells,
        "cell_null_pct": (100.0 * total_null_cells / (total_rows * len(cols))) if total_rows and cols else 0.0,
        "physical_cell_null_pct": (
            100.0 * physical_null_cells / (total_rows * len(physical_cols))
        ) if total_rows and physical_cols else 0.0,
        "rows_any_null": rows_any_null,
        "rows_any_null_pct": (100.0 * rows_any_null / total_rows) if total_rows else 0.0,
        "complete_rows": total_rows - rows_any_null,
        "nullable_columns": nullable_cols,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        per_column.to_csv(out / "columns.csv", index=False)
        null_count_per_row.to_csv(out / "null_count_per_row.csv", index=False)
        joint_patterns.to_csv(out / "joint_patterns.csv", index=False)
        for grp, tbl in by_group.items():
            tbl.to_csv(out / f"by_{grp}.csv", index=False)
        write_json(out / "summary.json", summary)

    return {
        "summary": summary,
        "per_column": per_column,
        "null_count_per_row": null_count_per_row,
        "joint_patterns": joint_patterns,
        "by_group": by_group,
    }
