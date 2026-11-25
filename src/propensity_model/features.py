"""Point-in-time feature enrichment (prior_purchase + derived_behavioral) and Top-50 catalog."""
from pathlib import Path
from typing import Any

import duckdb

PRIOR_PURCHASE_FEATURES: tuple[str, ...] = (
    # Fixed 3-month rolling window [T-3, T-1] (stationary from 2022-05 onward)
    "prior_3m_bought_any",
    "prior_3m_buy_months",
    "prior_3m_total_products",
    "prior_3m_bought_p1",
    "prior_3m_bought_p2",
    "prior_3m_bought_p3",
    "prior_3m_bought_p4",
    # Immediate previous month & purchase recency
    "prior_1m_bought_any",
    "prior_months_since_last_buy",
    # Cumulative expanding history strictly before cutoff (m < T)
    "prior_all_bought_any",
    "prior_all_buy_months",
    "prior_all_bought_p1",
    "prior_all_bought_p2",
    "prior_all_bought_p3",
    "prior_all_bought_p4",
    "prior_all_distinct_products",
    "prior_label_months_observed",
)

DERIVED_BEHAVIORAL_FEATURES: tuple[str, ...] = (
    # U-curve & new activation (Report 2 Slides A-B)
    "is_zero_trx_hist",
    "zero_trx_x_dialog_30d",
    "zero_trx_x_geo_g4_30d",
    "trx_first_activated_30d",
    "trx_inactive_30d_after_hist",
    # Tenure rate & short-term momentum (Slide B)
    "any_history_age_days",
    "trx_rate_per_30d_tenure",
    "trx_share_30d_of_84d",
    "trx_momentum_30d_vs_prior",
    # Diversity & cross-source interactions (Slides C, F, G)
    "trx_30d_vs_all_etype_ratio",
    "trx_30d_etype_x_geo_g4_30d",
    "dialog_30d_x_prior_any",
    "dialog_30d_x_log_trx_all",
    "active_sources_30d_count",
)

TOP50_PRIOR_FEATURES: tuple[str, ...] = (
    "prior_1m_bought_any",
    "prior_months_since_last_buy",
    "prior_3m_buy_months",
    "prior_3m_total_products",
    "prior_all_buy_months",
    "prior_all_distinct_products",
    "prior_all_bought_p1",
    "prior_all_bought_p2",
    "prior_all_bought_p3",
    "prior_all_bought_p4",
)

TOP50_DIALOG_FEATURES: tuple[str, ...] = (
    "dialog_30d_has_event",
    "dialog_30d_event_count",
    "dialog_30d_days_since_last_event",
    "dialog_84d_event_count",
    "dialog_all_event_count",
    "dialog_all_days_since_last_event",
    "dialog_30d_x_prior_any",
    "dialog_30d_x_log_trx_all",
    "zero_trx_x_dialog_30d",
)

TOP50_TRX_FEATURES: tuple[str, ...] = (
    # Frequency, recency, tenure & momentum (11)
    "trx_30d_event_count",
    "trx_30d_active_days",
    "trx_30d_days_since_last_event",
    "trx_84d_event_count",
    "trx_all_event_count",
    "trx_all_days_since_last_event",
    "trx_all_days_since_first_recorded_event",
    "trx_rate_per_30d_tenure",
    "trx_share_30d_of_84d",
    "trx_momentum_30d_vs_prior",
    "trx_first_activated_30d",
    # Monetary, foreign currency & diversity (12)
    "trx_30d_log1p_main_amount_abs_sum",
    "trx_30d_main_amount_signed_log_mean",
    "trx_30d_main_amount_max",
    "trx_84d_log1p_main_amount_abs_sum",
    "trx_all_log1p_main_amount_abs_sum",
    "trx_all_main_amount_median",
    "trx_all_foreign_fraction_known_currency",
    "trx_30d_event_type_nunique",
    "trx_30d_event_subtype_nunique",
    "trx_30d_event_type_top_share",
    "trx_all_event_type_nunique",
    "trx_30d_vs_all_etype_ratio",
)

TOP50_GEO_COHORT_FEATURES: tuple[str, ...] = (
    "geo_30d_event_count",
    "geo_30d_days_since_last_event",
    "geo_30d_geohash_4_nunique",
    "geo_all_geohash_4_nunique",
    "geo_30d_top_geohash_6_share",
    "trx_30d_etype_x_geo_g4_30d",
    "active_sources_30d_count",
    "cold_start",
)

TOP50_FEATURES: tuple[str, ...] = (
    TOP50_PRIOR_FEATURES
    + TOP50_DIALOG_FEATURES
    + TOP50_TRX_FEATURES
    + TOP50_GEO_COHORT_FEATURES
)

# Top-50 excluding all features that touch prior purchase labels (for clean ablation)
TOP50_NO_PRIOR_FEATURES: tuple[str, ...] = tuple(
    c for c in TOP50_FEATURES
    if c not in TOP50_PRIOR_FEATURES and c != "dialog_30d_x_prior_any"
)

assert len(PRIOR_PURCHASE_FEATURES) == 17
assert len(DERIVED_BEHAVIORAL_FEATURES) == 14
assert len(TOP50_FEATURES) == 50
assert len(set(TOP50_FEATURES)) == 50


def _sql_lit(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def build_enriched_panel(
    processed_parquet: Path,
    destination: Path,
    burn_in_months: tuple[str, ...],
    memory_limit: str = "2GB",
    threads: int = 2,
) -> dict[str, Any]:
    """Compute y_h1, y_h3, 17 prior_purchase, 14 derived_behavioral and coverage_slice.

    All label-derived features use STRICTLY preceding months (m < T), computed over
    the full 12-month history including burn-in months before filtering out burn-in
    rows from the modeling panel.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    burn_sql = ", ".join(_sql_lit(m) for m in burn_in_months)

    query = f"""
    WITH raw_panel AS (
        SELECT
            *,
            strftime(cutoff, '%Y-%m') AS month,
            CAST(EXTRACT(YEAR FROM cutoff) * 12 + EXTRACT(MONTH FROM cutoff) AS INTEGER) AS month_idx,
            CAST(y_any AS INTEGER) AS _y1,
            CAST(COALESCE(target_1, 0) AS INTEGER) AS _t1,
            CAST(COALESCE(target_2, 0) AS INTEGER) AS _t2,
            CAST(COALESCE(target_3, 0) AS INTEGER) AS _t3,
            CAST(COALESCE(target_4, 0) AS INTEGER) AS _t4
        FROM read_parquet({_sql_lit(processed_parquet.as_posix())}, hive_partitioning=false)
    ),
    windowed AS (
        SELECT
            * EXCLUDE (_y1, _t1, _t2, _t3, _t4, month_idx),
            -- Horizons
            CAST(_y1 AS TINYINT) AS y_h1,
            CASE
                WHEN COUNT(_y1) OVER w_fut3 = 3
                THEN CAST(MAX(_y1) OVER w_fut3 AS TINYINT)
                ELSE NULL
            END AS y_h3,

            -- Fixed 3-month prior purchase window [T-3, T-1]
            CAST(CASE WHEN COALESCE(SUM(_y1) OVER w_p3, 0) > 0 THEN 1 ELSE 0 END AS TINYINT) AS prior_3m_bought_any,
            CAST(COALESCE(SUM(_y1) OVER w_p3, 0) AS TINYINT) AS prior_3m_buy_months,
            CAST(COALESCE(SUM(_t1 + _t2 + _t3 + _t4) OVER w_p3, 0) AS TINYINT) AS prior_3m_total_products,
            CAST(CASE WHEN COALESCE(SUM(_t1) OVER w_p3, 0) > 0 THEN 1 ELSE 0 END AS TINYINT) AS prior_3m_bought_p1,
            CAST(CASE WHEN COALESCE(SUM(_t2) OVER w_p3, 0) > 0 THEN 1 ELSE 0 END AS TINYINT) AS prior_3m_bought_p2,
            CAST(CASE WHEN COALESCE(SUM(_t3) OVER w_p3, 0) > 0 THEN 1 ELSE 0 END AS TINYINT) AS prior_3m_bought_p3,
            CAST(CASE WHEN COALESCE(SUM(_t4) OVER w_p3, 0) > 0 THEN 1 ELSE 0 END AS TINYINT) AS prior_3m_bought_p4,

            -- Immediate previous month & recency of purchase
            CAST(COALESCE(LAG(_y1, 1) OVER (PARTITION BY client_id ORDER BY cutoff), 0) AS TINYINT) AS prior_1m_bought_any,
            CAST(
                month_idx - MAX(CASE WHEN _y1 = 1 THEN month_idx ELSE NULL END) OVER w_pall
                AS DOUBLE
            ) AS prior_months_since_last_buy,

            -- Cumulative expanding prior purchase history (m < T)
            CAST(CASE WHEN COALESCE(SUM(_y1) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END AS TINYINT) AS prior_all_bought_any,
            CAST(COALESCE(SUM(_y1) OVER w_pall, 0) AS SMALLINT) AS prior_all_buy_months,
            CAST(CASE WHEN COALESCE(SUM(_t1) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END AS TINYINT) AS prior_all_bought_p1,
            CAST(CASE WHEN COALESCE(SUM(_t2) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END AS TINYINT) AS prior_all_bought_p2,
            CAST(CASE WHEN COALESCE(SUM(_t3) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END AS TINYINT) AS prior_all_bought_p3,
            CAST(CASE WHEN COALESCE(SUM(_t4) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END AS TINYINT) AS prior_all_bought_p4,
            CAST(
                (CASE WHEN COALESCE(SUM(_t1) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END)
                + (CASE WHEN COALESCE(SUM(_t2) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END)
                + (CASE WHEN COALESCE(SUM(_t3) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END)
                + (CASE WHEN COALESCE(SUM(_t4) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END)
                AS TINYINT
            ) AS prior_all_distinct_products,
            CAST(COUNT(_y1) OVER w_pall AS SMALLINT) AS prior_label_months_observed
        FROM raw_panel
        WINDOW
            w_fut3 AS (PARTITION BY client_id ORDER BY cutoff ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING),
            w_p3   AS (PARTITION BY client_id ORDER BY cutoff ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING),
            w_pall AS (PARTITION BY client_id ORDER BY cutoff ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ),
    enriched AS (
        SELECT
            *,
            -- 14 derived_behavioral features (Report 2 Slides A-H)
            CAST(1 - COALESCE(trx_all_has_event, 0) AS TINYINT) AS is_zero_trx_hist,
            CAST((1 - COALESCE(trx_all_has_event, 0)) * COALESCE(dialog_30d_has_event, 0) AS TINYINT) AS zero_trx_x_dialog_30d,
            CAST((1 - COALESCE(trx_all_has_event, 0)) * COALESCE(geo_30d_geohash_4_nunique, 0) AS DOUBLE) AS zero_trx_x_geo_g4_30d,
            CAST(
                CASE
                    WHEN COALESCE(trx_30d_event_count, 0) > 0
                     AND COALESCE(trx_30d_event_count, 0) = COALESCE(trx_all_event_count, 0)
                    THEN 1 ELSE 0
                END AS TINYINT
            ) AS trx_first_activated_30d,
            CAST(
                CASE
                    WHEN COALESCE(trx_all_has_event, 0) = 1
                     AND COALESCE(trx_30d_has_event, 0) = 0
                    THEN 1 ELSE 0
                END AS TINYINT
            ) AS trx_inactive_30d_after_hist,
            NULLIF(
                GREATEST(
                    COALESCE(trx_all_days_since_first_recorded_event, -1.0),
                    COALESCE(geo_all_days_since_first_recorded_event, -1.0),
                    COALESCE(dialog_all_days_since_first_recorded_event, -1.0)
                ),
                -1.0
            ) AS any_history_age_days,
            CASE
                WHEN trx_all_days_since_first_recorded_event IS NULL THEN NULL
                ELSE CAST(trx_all_event_count AS DOUBLE) / GREATEST(trx_all_days_since_first_recorded_event / 30.0, 1.0)
            END AS trx_rate_per_30d_tenure,
            CASE
                WHEN COALESCE(trx_84d_event_count, 0) > 0
                THEN CAST(COALESCE(trx_30d_event_count, 0) AS DOUBLE) / CAST(trx_84d_event_count AS DOUBLE)
                ELSE NULL
            END AS trx_share_30d_of_84d,
            CASE
                WHEN GREATEST(COALESCE(trx_all_event_count, 0) - COALESCE(trx_30d_event_count, 0), 0) > 0
                THEN CAST(COALESCE(trx_30d_event_count, 0) AS DOUBLE) / (
                    CAST(GREATEST(COALESCE(trx_all_event_count, 0) - COALESCE(trx_30d_event_count, 0), 0) AS DOUBLE)
                    / GREATEST((COALESCE(trx_all_days_since_first_recorded_event, 30.0) - 30.0) / 30.0, 1.0)
                )
                ELSE NULL
            END AS trx_momentum_30d_vs_prior,
            CASE
                WHEN COALESCE(trx_all_event_type_nunique, 0) > 0
                THEN CAST(COALESCE(trx_30d_event_type_nunique, 0) AS DOUBLE) / CAST(trx_all_event_type_nunique AS DOUBLE)
                ELSE NULL
            END AS trx_30d_vs_all_etype_ratio,
            CAST(COALESCE(trx_30d_event_type_nunique, 0) * COALESCE(geo_30d_geohash_4_nunique, 0) AS DOUBLE) AS trx_30d_etype_x_geo_g4_30d,
            CAST(COALESCE(dialog_30d_has_event, 0) * prior_all_bought_any AS TINYINT) AS dialog_30d_x_prior_any,
            CAST(COALESCE(dialog_30d_has_event, 0) AS DOUBLE) * COALESCE(trx_all_log1p_event_count, 0.0) AS dialog_30d_x_log_trx_all,
            CAST(
                COALESCE(trx_30d_has_event, 0)
                + COALESCE(geo_30d_has_event, 0)
                + COALESCE(dialog_30d_has_event, 0)
                AS TINYINT
            ) AS active_sources_30d_count,

            -- Coverage slice label (6 mutually exclusive segments)
            CASE
                WHEN cold_start = 1 THEN 'cold_start'
                WHEN COALESCE(trx_all_has_event, 0) = 0
                 AND (COALESCE(geo_all_has_event, 0) = 1 OR COALESCE(dialog_all_has_event, 0) = 1) THEN 'no_trx_has_other'
                WHEN COALESCE(trx_all_has_event, 0) = 1
                 AND COALESCE(geo_all_has_event, 0) = 0 AND COALESCE(dialog_all_has_event, 0) = 0 THEN 'trx_only'
                WHEN COALESCE(trx_all_has_event, 0) = 1
                 AND COALESCE(geo_all_has_event, 0) = 1 AND COALESCE(dialog_all_has_event, 0) = 0 THEN 'trx_geo'
                WHEN COALESCE(trx_all_has_event, 0) = 1
                 AND COALESCE(geo_all_has_event, 0) = 0 AND COALESCE(dialog_all_has_event, 0) = 1 THEN 'trx_dialog'
                WHEN COALESCE(trx_all_has_event, 0) = 1
                 AND COALESCE(geo_all_has_event, 0) = 1 AND COALESCE(dialog_all_has_event, 0) = 1 THEN 'trx_geo_dialog'
                ELSE 'other'
            END AS coverage_slice
        FROM windowed
        WHERE month NOT IN ({burn_sql})
    )
    SELECT * FROM enriched
    ORDER BY cutoff, client_id
    """

    with duckdb.connect() as con:
        con.execute(f"SET memory_limit={_sql_lit(memory_limit)}")
        con.execute(f"SET threads={int(threads)}")
        con.execute(
            f"COPY ({query}) TO {_sql_lit(destination.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        rows = con.execute(
            f"""
            SELECT
                month,
                COUNT(*) AS n_rows,
                MIN(prior_label_months_observed) AS min_prior_m,
                MAX(prior_label_months_observed) AS max_prior_m,
                SUM(y_h1) AS pos_h1,
                SUM(y_h3) AS pos_h3,
                COUNT(y_h3) AS labelled_h3,
                SUM(prior_all_bought_any) AS n_prior_buyers,
                SUM(cold_start) AS n_cold_start
            FROM read_parquet({_sql_lit(destination.as_posix())}, hive_partitioning=false)
            GROUP BY month
            ORDER BY month
            """
        ).fetchall()

        # Verify no +/-inf in new numeric features
        new_double_cols = [
            "prior_months_since_last_buy",
            "zero_trx_x_geo_g4_30d",
            "any_history_age_days",
            "trx_rate_per_30d_tenure",
            "trx_share_30d_of_84d",
            "trx_momentum_30d_vs_prior",
            "trx_30d_vs_all_etype_ratio",
            "trx_30d_etype_x_geo_g4_30d",
            "dialog_30d_x_log_trx_all",
        ]
        inf_checks = " + ".join(
            f"COUNT(*) FILTER (WHERE isinf({c}))" for c in new_double_cols
        )
        inf_count = con.execute(
            f"SELECT {inf_checks} FROM read_parquet({_sql_lit(destination.as_posix())}, hive_partitioning=false)"
        ).fetchone()[0]
        if inf_count != 0:
            raise ValueError(f"Enriched features contain {inf_count} infinite values")

    monthly_summary = [
        {
            "month": r[0],
            "rows": int(r[1]),
            "min_prior_months": int(r[2]),
            "max_prior_months": int(r[3]),
            "pos_h1": int(r[4]),
            "pos_h3": None if r[5] is None else int(r[5]),
            "labelled_h3": int(r[6]),
            "n_prior_buyers": int(r[7]),
            "n_cold_start": int(r[8]),
        }
        for r in rows
    ]
    if any(m["min_prior_months"] < len(burn_in_months) for m in monthly_summary):
        raise ValueError("Burn-in invariant violated: a modeling month has fewer prior label months than burn-in")

    return {
        "enriched_parquet": str(destination),
        "bytes": destination.stat().st_size,
        "monthly_summary": monthly_summary,
        "total_rows": sum(m["rows"] for m in monthly_summary),
    }
