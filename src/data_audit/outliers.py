"""Hash-bucketed within-group robust (log-MAD) outlier, repeat, and power-of-10 audit."""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from ._compat import load_dataframe, open_duckdb_session, sql_ident, sql_literal, write_json


def audit_univariate_outliers(
    source: Any,
    *,
    columns: Sequence[str] | None = None,
    iqr_multiplier: float = 1.5,
    mad_threshold: float = 3.5,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    sample_size: int | None = None,
    seed: int = 42,
    **source_options: Any,
) -> pd.DataFrame:
    """Audit univariate outliers for numeric columns via IQR fences, Modified Z-score (MAD), and quantile tails."""
    df = load_dataframe(source, columns=columns, **source_options)
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)

    target_cols = list(columns) if columns is not None else [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    rows: list[dict[str, Any]] = []
    for col in target_cols:
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        clean = arr[np.isfinite(arr)]
        n = len(clean)
        if not n:
            rows.append({"column": col, "finite_n": 0})
            continue
        q_lo, q25, med, q75, q_hi = np.quantile(clean, [lower_quantile, 0.25, 0.5, 0.75, upper_quantile])
        iqr = q75 - q25
        iqr_lo = q25 - iqr_multiplier * iqr
        iqr_hi = q75 + iqr_multiplier * iqr
        iqr_out = int(((clean < iqr_lo) | (clean > iqr_hi)).sum())

        mad = float(np.median(np.abs(clean - med)))
        if mad > 0:
            mod_z = 0.67448975 * np.abs(clean - med) / mad
            mad_out = int((mod_z > mad_threshold).sum())
        else:
            mad_out = 0

        tail_low = int((clean < q_lo).sum())
        tail_high = int((clean > q_hi).sum())
        rows.append({
            "column": col,
            "finite_n": int(n),
            "mean": float(clean.mean()),
            "median": float(med),
            "std": float(clean.std(ddof=1)) if n > 1 else np.nan,
            "mad": mad,
            "iqr_lower": float(iqr_lo),
            "iqr_upper": float(iqr_hi),
            "iqr_outlier_n": iqr_out,
            "iqr_outlier_pct": float(iqr_out / n * 100.0),
            "mad_outlier_n": mad_out,
            "mad_outlier_pct": float(mad_out / n * 100.0),
            "tail_low_n": tail_low,
            "tail_high_n": tail_high,
            "min": float(clean.min()),
            "p_low": float(q_lo),
            "p_high": float(q_hi),
            "max": float(clean.max()),
        })
    return pd.DataFrame(rows)


def audit_within_group_outliers(
    source: Any,
    *,
    entity_col: str,
    value_col: str,
    group_cols: Sequence[str] = (),
    time_col: str | None = None,
    metadata_cols: Sequence[str] = (),
    min_group_n: int = 20,
    mad_multiplier: float = 6.0,
    scale_floor: float = math.log(2.0),
    n_buckets: int = 16,
    log_transform: bool = True,
    top_k: int = 500,
    check_power10: bool = True,
    cluster_k: int | None = None,
    seed: int = 42,
    output_dir: str | Path | None = None,
    memory_limit: str = "2GB",
    threads: int = 1,
    **source_options: Any,
) -> dict[str, Any]:
    """Detect within-(entity, group_cols) robust outliers without loading all rows into RAM.

    Descriptive flags only: never mutates source data and should not be used blindly as deletion rules.
    """
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")

    key_cols = [entity_col, *group_cols]
    q_ent = sql_ident(entity_col)
    q_val = sql_ident(value_col)
    q_keys = ", ".join(sql_ident(c) for c in key_cols)
    join_cond = " AND ".join(f"t.{sql_ident(c)} IS NOT DISTINCT FROM r.{sql_ident(c)}" for c in key_cols)

    val_filter = f"{q_val} > 0 AND isfinite({q_val}::DOUBLE)" if log_transform else f"{q_val} IS NOT NULL AND isfinite({q_val}::DOUBLE)"
    x_expr = f"ln({q_val}::DOUBLE)" if log_transform else f"{q_val}::DOUBLE"
    t_x_expr = f"ln(t.{q_val}::DOUBLE)" if log_transform else f"t.{q_val}::DOUBLE"
    scale_expr = f"greatest(1.4826 * s.mad, {float(scale_floor)})"

    with tempfile.TemporaryDirectory(prefix="data-audit-outliers-") as tmp_dir:
        tmp = Path(tmp_dir)
        cache_dir = tmp / "buckets"
        with open_duckdb_session(memory_limit=memory_limit, threads=threads, spill_dir=tmp / "spill") as db:
            db.register("raw", source, **source_options)
            schema_checks = db.query(
                f"""SELECT count(*) AS rows,
                           count(*) FILTER (WHERE {q_val} IS NULL) AS missing_value,
                           count(*) FILTER (WHERE {q_val} IS NOT NULL AND NOT isfinite({q_val}::DOUBLE)) AS nonfinite_value,
                           count(*) FILTER (WHERE {q_val}::DOUBLE <= 0) AS nonpositive_value,
                           count(*) FILTER (WHERE {q_ent} IS NULL) AS missing_entity
                    FROM raw"""
            )

            db.execute(
                f"COPY (SELECT *, abs(hash({q_ent})) % {int(n_buckets)} AS __bucket FROM raw) "
                f"TO {sql_literal(cache_dir.as_posix())} (FORMAT PARQUET, PARTITION_BY (__bucket))"
            )

            ref_sql = (
                f"SELECT {q_keys}, count(*) AS n, median({x_expr}) AS med, "
                f"median({q_val}::DOUBLE) AS median_value FROM part WHERE {val_filter} GROUP BY ALL"
            )
            scale_sql = (
                f"SELECT r.*, median(abs({t_x_expr} - r.med)) AS mad "
                f"FROM part t JOIN local_ref r ON {join_cond} "
                f"WHERE r.n >= {int(min_group_n)} AND {val_filter.replace(q_val, 't.' + q_val)} GROUP BY ALL"
            )
            flag_sql = (
                f"SELECT t.*, s.n AS reference_n, s.median_value, "
                f"CASE WHEN s.median_value <> 0 THEN t.{q_val}::DOUBLE / s.median_value ELSE NULL END AS ratio_to_median, "
                f"abs({t_x_expr} - s.med) / {scale_expr} AS robust_score "
                f"FROM part t JOIN local_scales s ON {join_cond.replace('r.', 's.')} "
                f"WHERE s.n >= {int(min_group_n)} AND {val_filter.replace(q_val, 't.' + q_val)} "
                f"AND abs({t_x_expr} - s.med) > {float(mad_multiplier)} * {scale_expr}"
            )

            has_repeats = time_col is not None
            if has_repeats:
                q_time = sql_ident(time_col)
                if metadata_cols:
                    struct_fields = ", ".join(f"c{i} := {sql_ident(c)}" for i, c in enumerate(metadata_cols))
                    variant_expr = f"count(DISTINCT struct_pack({struct_fields})) AS variants"
                else:
                    variant_expr = "1 AS variants"
                repeat_sql = (
                    f"SELECT {q_ent}, {q_time}, {q_val}, count(*) AS n, {variant_expr} "
                    f"FROM part GROUP BY {q_ent}, {q_time}, {q_val} HAVING count(*) > 1"
                )

            profile_sql = (
                f"SELECT {q_ent}, count(*) AS events, "
                + (f"count(DISTINCT cast({sql_ident(time_col)} AS DATE)) AS active_days, " if time_col else "0 AS active_days, ")
                + f"sum({q_val}::DOUBLE) AS total_value, median({q_val}::DOUBLE) AS median_value, "
                f"max({q_val}::DOUBLE) AS max_value FROM part WHERE {val_filter} GROUP BY {q_ent}"
            )

            bucket_dirs = sorted(cache_dir.glob("__bucket=*"))
            for i, folder in enumerate(bucket_dirs):
                pq_files = [p.as_posix() for p in sorted(folder.glob("*.parquet"))]
                paths_sql = ",".join(sql_literal(p) for p in pq_files)
                db.execute(f"CREATE OR REPLACE TEMP VIEW part AS SELECT * FROM read_parquet([{paths_sql}], hive_partitioning=false)")
                db.execute(f"CREATE OR REPLACE TEMP TABLE local_ref AS {ref_sql}")
                db.execute(f"CREATE OR REPLACE TEMP TABLE local_scales AS {scale_sql}")
                prefix = "CREATE TEMP TABLE" if i == 0 else "INSERT INTO"
                targets = [("ref", "SELECT * FROM local_ref"), ("flagged", flag_sql), ("profiles", profile_sql)]
                if has_repeats:
                    targets.append(("repeats", repeat_sql))
                for name, sql in targets:
                    stmt = f"{prefix} {name} AS {sql}" if i == 0 else f"INSERT INTO {name} {sql}"
                    db.execute(stmt)

            if not bucket_dirs:
                return {"schema_checks": schema_checks, "flagged_summary": pd.DataFrame(), "top_flagged": pd.DataFrame()}

            reference_coverage = db.query(
                f"""SELECT coalesce(sum(n), 0) AS total_valid_events,
                           coalesce(sum(n) FILTER (WHERE n >= {int(min_group_n)}), 0) AS eligible_events,
                           count(*) AS reference_groups,
                           count(*) FILTER (WHERE n >= {int(min_group_n)}) AS eligible_groups
                    FROM ref"""
            )
            group_select = (", ".join(sql_ident(c) for c in group_cols) + ", ") if group_cols else ""
            group_clause = (" GROUP BY " + ", ".join(sql_ident(c) for c in group_cols)) if group_cols else ""
            flagged_summary = db.query(
                f"SELECT {group_select} count(*) AS flagged_events, count(DISTINCT {q_ent}) AS flagged_entities "
                f"FROM flagged{group_clause}"
            )
            top_flagged = db.query(f"SELECT * FROM flagged ORDER BY robust_score DESC LIMIT {int(top_k)}")

            power10_summary = pd.DataFrame()
            power10_ratios = pd.DataFrame()
            if check_power10:
                power10_summary = db.query(
                    f"""SELECT count(*) AS total_positive,
                               count(*) FILTER (WHERE abs(log10({q_val}::DOUBLE) - round(log10({q_val}::DOUBLE))) < 0.00004343) AS near_power10,
                               min({q_val}::DOUBLE) AS min_value, max({q_val}::DOUBLE) AS max_value
                        FROM raw WHERE {q_val}::DOUBLE > 0 AND isfinite({q_val}::DOUBLE)"""
                )
                power10_ratios = db.query(
                    """SELECT round(log10(ratio_to_median)) AS exponent,
                              count(*) AS events, count(DISTINCT """ + q_ent + """) AS entities
                       FROM flagged
                       WHERE ratio_to_median > 0
                         AND abs(log10(ratio_to_median) - round(log10(ratio_to_median))) < 0.00004343
                       GROUP BY 1 ORDER BY events DESC"""
                )

            repeat_summary = pd.DataFrame()
            top_repeats = pd.DataFrame()
            if has_repeats:
                repeat_summary = db.query(
                    """SELECT count(*) AS repeat_groups, coalesce(sum(n), 0) AS involved_rows,
                              coalesce(sum(n - 1), 0) AS extra_rows,
                              coalesce(sum(n - 1) FILTER (WHERE variants = 1), 0) AS identical_metadata_extra_rows,
                              count(*) FILTER (WHERE variants > 1) AS differing_metadata_groups
                       FROM repeats"""
                )
                top_repeats = db.query(f"SELECT * FROM repeats ORDER BY n DESC LIMIT {int(top_k)}")

            profiles = db.query("SELECT * FROM profiles")

    cluster_summary = pd.DataFrame()
    if cluster_k is not None and len(profiles) >= cluster_k:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import RobustScaler

        feat_cols = [c for c in ("events", "active_days", "total_value", "median_value", "max_value")
                     if c in profiles.columns and profiles[c].notna().all()]
        x = RobustScaler().fit_transform(np.log1p(np.maximum(profiles[feat_cols].to_numpy(dtype=float), 0.0)))
        profiles["cluster"] = KMeans(n_clusters=int(cluster_k), n_init=10, random_state=seed).fit_predict(x)
        cluster_summary = profiles.groupby("cluster").agg(
            entities=(entity_col, "size"),
            total_events=("events", "sum"),
            median_events=("events", "median"),
            median_total_value=("total_value", "median"),
            median_value=("median_value", "median"),
        ).reset_index()

    methodology = {
        "entity_col": entity_col,
        "value_col": value_col,
        "group_cols": list(group_cols),
        "min_group_n": min_group_n,
        "mad_multiplier": mad_multiplier,
        "scale_floor": scale_floor,
        "log_transform": log_transform,
        "n_buckets": n_buckets,
        "descriptive_only": True,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for fname, frame in [
            ("schema_value_checks.csv", schema_checks),
            ("reference_coverage.csv", reference_coverage),
            ("within_group_summary.csv", flagged_summary),
            ("within_group_top_flagged.csv", top_flagged),
            ("power10_summary.csv", power10_summary),
            ("within_group_power10_ratios.csv", power10_ratios),
            ("repeat_summary.csv", repeat_summary),
            ("top_repeats.csv", top_repeats),
            ("cluster_summary.csv", cluster_summary),
        ]:
            if not frame.empty:
                frame.to_csv(out / fname, index=False)
        write_json(out / "methodology.json", methodology)

    return {
        "methodology": methodology,
        "schema_checks": schema_checks,
        "reference_coverage": reference_coverage,
        "flagged_summary": flagged_summary,
        "top_flagged": top_flagged,
        "power10_summary": power10_summary,
        "power10_ratios": power10_ratios,
        "repeat_summary": repeat_summary,
        "top_repeats": top_repeats,
        "entity_profiles": profiles,
        "cluster_summary": cluster_summary,
    }
