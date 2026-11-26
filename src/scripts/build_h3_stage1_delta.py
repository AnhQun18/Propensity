"""Build compact Stage-1 local delta bundle (h3_stage1_delta.zip) from raw MBD-mini.

Produces:
  1. h3_targets_prior.parquet      (ACTIVE_H3_MONTHS: 02..08, 11 with y_any, y_h3, delay=0 & delay=1m prior_*)
  2. h3_early_02_03_base112.parquet (Exact 112 Moscow-time behavioral cols for 2022-02 & 2022-03)
  3. h3_early_02_03_emb.npz        (float16 768d mean-pooled dialog embeddings for 2022-02 & 2022-03)
  4. h3_verify_aux.parquet         (Lagged dialog lag=7/14/30d metadata + UTC-cutoff TRX RFM check cols)
"""
from __future__ import annotations

import gc
import sys
import time
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.build_moscow_features import (
    _build_geo_month_tables,
    _build_trx_month_tables,
    _configure_con,
    _dialog_month_sql,
    _glob,
)

OUT_DIR = REPO_ROOT / "outputs" / "h3_propensity"
DELTA_DIR = OUT_DIR / "delta"
TMP_DIR = OUT_DIR / "_duck_tmp"
MOSCOW_PANEL = REPO_ROOT / "outputs" / "baseline_moscow_v1" / "panel_moscow.parquet"

ACTIVE_H3_MONTHS = ("2022-02", "2022-03", "2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-11")
EARLY_MONTHS = ("2022-02", "2022-03")
FOLDS = (0, 1, 2, 3, 4)


def build_h3_targets_and_prior(con: duckdb.DuckDBPyConnection, out_path: Path) -> None:
    if out_path.exists():
        print(f"  [1/4] Reusing {out_path.name}", flush=True)
        return
    t0 = time.perf_counter()
    targets_glob = _glob("targets/*/*.parquet")
    folds_glob = _glob("client_split/*/*.parquet")
    months_sql = ", ".join(f"'{m}'" for m in ACTIVE_H3_MONTHS)
    q = f"""
    WITH folds AS (
        SELECT DISTINCT client_id, CAST(fold AS TINYINT) AS fold
        FROM read_parquet('{folds_glob}', hive_partitioning=true)
    ),
    tgt AS (
        SELECT
            t.client_id,
            f.fold,
            substr(t.mon, 1, 7) AS month,
            CAST(substr(t.mon, 1, 7) || '-01 00:00:00' AS TIMESTAMP) AS cutoff_msk,
            CAST(EXTRACT(YEAR FROM CAST(substr(t.mon, 1, 7) || '-01' AS DATE)) * 12
                 + EXTRACT(MONTH FROM CAST(substr(t.mon, 1, 7) || '-01' AS DATE)) AS INTEGER) AS month_idx,
            CAST(COALESCE(t.target_1, 0) AS TINYINT) AS target_1,
            CAST(COALESCE(t.target_2, 0) AS TINYINT) AS target_2,
            CAST(COALESCE(t.target_3, 0) AS TINYINT) AS target_3,
            CAST(COALESCE(t.target_4, 0) AS TINYINT) AS target_4,
            CAST(GREATEST(COALESCE(t.target_1, 0), COALESCE(t.target_2, 0),
                          COALESCE(t.target_3, 0), COALESCE(t.target_4, 0)) AS TINYINT) AS y_any
        FROM read_parquet('{targets_glob}') t
        JOIN folds f USING (client_id)
    ),
    h3_lagged AS (
        SELECT
            client_id,
            fold,
            month,
            month_idx,
            y_any,
            GREATEST(
                y_any,
                LEAD(y_any, 1) OVER (PARTITION BY client_id ORDER BY cutoff_msk),
                LEAD(y_any, 2) OVER (PARTITION BY client_id ORDER BY cutoff_msk)
            )::TINYINT AS y_h3,
            -- Strict delay=0 target-lag features (t <= s - 1)
            CAST(COUNT(y_any) OVER w_pall AS SMALLINT) AS prior_months_observed,
            CASE WHEN COUNT(y_any) OVER w_pall = 0 THEN NULL ELSE COALESCE(LAG(y_any, 1) OVER w_ord, 0)::FLOAT END AS prior_1m_bought_any,
            CASE WHEN COUNT(y_any) OVER w_pall = 0 THEN NULL ELSE COALESCE(SUM(y_any) OVER w_p3, 0)::FLOAT END AS prior_3m_buy_months,
            CASE WHEN COUNT(y_any) OVER w_pall = 0 THEN NULL ELSE COALESCE(SUM(target_1 + target_2 + target_3 + target_4) OVER w_p3, 0)::FLOAT END AS prior_3m_total_products,
            CASE WHEN COUNT(y_any) OVER w_pall = 0 THEN NULL ELSE COALESCE(SUM(y_any) OVER w_p6, 0)::FLOAT END AS prior_6m_buy_months,
            CASE WHEN COUNT(y_any) OVER w_pall = 0 THEN NULL ELSE COALESCE(SUM(target_1 + target_2 + target_3 + target_4) OVER w_p6, 0)::FLOAT END AS prior_6m_total_products,
            CASE WHEN COUNT(y_any) OVER w_pall = 0 THEN NULL ELSE (CASE WHEN COALESCE(SUM(y_any) OVER w_pall, 0) > 0 THEN 1.0 ELSE 0.0 END)::FLOAT END AS prior_all_bought_any,
            CASE WHEN COUNT(y_any) OVER w_pall = 0 THEN NULL ELSE COALESCE(SUM(y_any) OVER w_pall, 0)::FLOAT END AS prior_all_buy_months,
            CASE WHEN COUNT(y_any) OVER w_pall = 0 THEN NULL ELSE (month_idx - MAX(CASE WHEN y_any = 1 THEN month_idx ELSE NULL END) OVER w_pall)::FLOAT END AS prior_months_since_last_buy,
            CASE WHEN COUNT(y_any) OVER w_pall = 0 THEN NULL ELSE (CASE WHEN COALESCE(SUM(target_1) OVER w_pall, 0) > 0 THEN 1.0 ELSE 0.0 END)::FLOAT END AS prior_all_bought_p1,
            CASE WHEN COUNT(y_any) OVER w_pall = 0 THEN NULL ELSE (CASE WHEN COALESCE(SUM(target_2) OVER w_pall, 0) > 0 THEN 1.0 ELSE 0.0 END)::FLOAT END AS prior_all_bought_p2,
            CASE WHEN COUNT(y_any) OVER w_pall = 0 THEN NULL ELSE (CASE WHEN COALESCE(SUM(target_3) OVER w_pall, 0) > 0 THEN 1.0 ELSE 0.0 END)::FLOAT END AS prior_all_bought_p3,
            CASE WHEN COUNT(y_any) OVER w_pall = 0 THEN NULL ELSE (CASE WHEN COALESCE(SUM(target_4) OVER w_pall, 0) > 0 THEN 1.0 ELSE 0.0 END)::FLOAT END AS prior_all_bought_p4,
            CASE WHEN COUNT(y_any) OVER w_pall = 0 THEN NULL ELSE (
                (CASE WHEN COALESCE(SUM(target_1) OVER w_pall, 0) > 0 THEN 1.0 ELSE 0.0 END)
                + (CASE WHEN COALESCE(SUM(target_2) OVER w_pall, 0) > 0 THEN 1.0 ELSE 0.0 END)
                + (CASE WHEN COALESCE(SUM(target_3) OVER w_pall, 0) > 0 THEN 1.0 ELSE 0.0 END)
                + (CASE WHEN COALESCE(SUM(target_4) OVER w_pall, 0) > 0 THEN 1.0 ELSE 0.0 END)
            )::FLOAT END AS prior_all_distinct_products,

            -- Delay=1m prior features (t <= s - 2) for production SLA audit
            CAST(COUNT(y_any) OVER w_d1_all AS SMALLINT) AS prior_delay1m_months_observed,
            CASE WHEN COUNT(y_any) OVER w_d1_all = 0 THEN NULL ELSE COALESCE(LAG(y_any, 2) OVER w_ord, 0)::FLOAT END AS prior_delay1m_prev_bought,
            CASE WHEN COUNT(y_any) OVER w_d1_all = 0 THEN NULL ELSE COALESCE(SUM(y_any) OVER w_d1_p3, 0)::FLOAT END AS prior_delay1m_3m_buy_months,
            CASE WHEN COUNT(y_any) OVER w_d1_all = 0 THEN NULL ELSE COALESCE(SUM(y_any) OVER w_d1_all, 0)::FLOAT END AS prior_delay1m_all_buy_months,
            CASE WHEN COUNT(y_any) OVER w_d1_all = 0 THEN NULL ELSE (month_idx - MAX(CASE WHEN y_any = 1 THEN month_idx ELSE NULL END) OVER w_d1_all)::FLOAT END AS prior_delay1m_months_since_last_buy
        FROM tgt
        WINDOW
            w_ord    AS (PARTITION BY client_id ORDER BY cutoff_msk),
            w_p3     AS (PARTITION BY client_id ORDER BY cutoff_msk ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING),
            w_p6     AS (PARTITION BY client_id ORDER BY cutoff_msk ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING),
            w_pall   AS (PARTITION BY client_id ORDER BY cutoff_msk ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
            w_d1_p3  AS (PARTITION BY client_id ORDER BY cutoff_msk ROWS BETWEEN 4 PRECEDING AND 2 PRECEDING),
            w_d1_all AS (PARTITION BY client_id ORDER BY cutoff_msk ROWS BETWEEN UNBOUNDED PRECEDING AND 2 PRECEDING)
    )
    SELECT * EXCLUDE (month_idx)
    FROM h3_lagged
    WHERE month IN ({months_sql})
    ORDER BY month, client_id
    """
    tmp = out_path.with_suffix(".tmp.parquet")
    con.execute(f"COPY ({q}) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    tmp.replace(out_path)
    print(f"  [1/4] Built {out_path.name} in {time.perf_counter()-t0:.1f}s", flush=True)


def build_early_02_03_base112(con: duckdb.DuckDBPyConnection, tgt_path: Path, out_path: Path) -> None:
    if out_path.exists():
        print(f"  [2/4] Reusing {out_path.name}", flush=True)
        return
    t0 = time.perf_counter()
    # Exact 112 behavioral col order from panel_moscow.parquet
    pm_cols = pq.ParquetFile(MOSCOW_PANEL).schema_arrow.names
    beh_cols = [c for c in pm_cols if c not in {"client_id", "fold", "month", "cutoff_msk", "y_any", "target_1", "target_2", "target_3", "target_4"} and not c.startswith("prior_")]
    assert len(beh_cols) == 112, f"Expected 112 beh cols, got {len(beh_cols)}"

    part_paths: list[Path] = []
    for m in EARLY_MONTHS:
        cutoff_str = f"{m}-01 00:00:00"
        for fold in FOLDS:
            trx_glob = _glob(f"detail/trx/fold={fold}/*.parquet")
            geo_glob = _glob(f"detail/geo/fold={fold}/*.parquet")
            dlg_glob = _glob(f"detail/dialog/fold={fold}/*.parquet")
            p_f = DELTA_DIR / f".tmp_early_{m}_f{fold}.parquet"
            part_paths.append(p_f)

            fcon = duckdb.connect()
            _configure_con(fcon, TMP_DIR)
            _build_trx_month_tables(fcon, trx_glob, cutoff_str)
            _build_geo_month_tables(fcon, geo_glob, cutoff_str)
            fcon.execute(f"CREATE OR REPLACE TEMP TABLE t_dlg AS {_dialog_month_sql(dlg_glob, cutoff_str)}")

            join_q = f"""
            WITH base AS (
                SELECT client_id, fold, month FROM read_parquet('{tgt_path.as_posix()}')
                WHERE month = '{m}' AND fold = {fold}
            )
            SELECT
                b.client_id, b.fold, b.month,
                COALESCE(tx.trx_30d_has_event, 0)::TINYINT AS trx_30d_has_event,
                COALESCE(tx.trx_30d_event_count, 0)::INTEGER AS trx_30d_event_count,
                COALESCE(tx.trx_30d_log1p_event_count, 0.0)::FLOAT AS trx_30d_log1p_event_count,
                COALESCE(tx.trx_30d_active_days, 0)::SMALLINT AS trx_30d_active_days,
                tx.trx_30d_days_since_last_event,
                tx.trx_30d_days_since_first_event,
                COALESCE(tx.trx_30d_main_count, 0)::INTEGER AS trx_30d_main_count,
                COALESCE(tx.trx_30d_foreign_count, 0)::INTEGER AS trx_30d_foreign_count,
                tx.trx_30d_foreign_fraction,
                COALESCE(tx.trx_30d_main_neg_count, 0)::INTEGER AS trx_30d_main_neg_count,
                COALESCE(tx.trx_30d_main_amount_abs_sum, 0.0)::DOUBLE AS trx_30d_main_amount_abs_sum,
                COALESCE(tx.trx_30d_log1p_main_amount_abs_sum, 0.0)::FLOAT AS trx_30d_log1p_main_amount_abs_sum,
                tx.trx_30d_main_amount_signed_log_mean,
                tx.trx_30d_main_amount_median,
                tx.trx_30d_main_amount_max,
                COALESCE(tx.trx_30d_event_type_nunique, 0)::SMALLINT AS trx_30d_event_type_nunique,
                COALESCE(tx.trx_30d_event_subtype_nunique, 0)::SMALLINT AS trx_30d_event_subtype_nunique,
                tx.trx_30d_event_type_top_share,
                COALESCE(tx.trx_30d_dst_type11_nunique, 0)::INTEGER AS trx_30d_dst_type11_nunique,
                COALESCE(tx.trx_30d_dst_type12_nunique, 0)::INTEGER AS trx_30d_dst_type12_nunique,

                COALESCE(tx.trx_90d_has_event, 0)::TINYINT AS trx_90d_has_event,
                COALESCE(tx.trx_90d_event_count, 0)::INTEGER AS trx_90d_event_count,
                COALESCE(tx.trx_90d_log1p_event_count, 0.0)::FLOAT AS trx_90d_log1p_event_count,
                COALESCE(tx.trx_90d_active_days, 0)::SMALLINT AS trx_90d_active_days,
                tx.trx_90d_days_since_last_event,
                tx.trx_90d_days_since_first_event,
                COALESCE(tx.trx_90d_main_count, 0)::INTEGER AS trx_90d_main_count,
                COALESCE(tx.trx_90d_foreign_count, 0)::INTEGER AS trx_90d_foreign_count,
                tx.trx_90d_foreign_fraction,
                COALESCE(tx.trx_90d_main_neg_count, 0)::INTEGER AS trx_90d_main_neg_count,
                COALESCE(tx.trx_90d_main_amount_abs_sum, 0.0)::DOUBLE AS trx_90d_main_amount_abs_sum,
                COALESCE(tx.trx_90d_log1p_main_amount_abs_sum, 0.0)::FLOAT AS trx_90d_log1p_main_amount_abs_sum,
                tx.trx_90d_main_amount_signed_log_mean,
                tx.trx_90d_main_amount_median,
                tx.trx_90d_main_amount_max,
                COALESCE(tx.trx_90d_event_type_nunique, 0)::SMALLINT AS trx_90d_event_type_nunique,
                COALESCE(tx.trx_90d_event_subtype_nunique, 0)::SMALLINT AS trx_90d_event_subtype_nunique,
                tx.trx_90d_event_type_top_share,
                COALESCE(tx.trx_90d_dst_type11_nunique, 0)::INTEGER AS trx_90d_dst_type11_nunique,
                COALESCE(tx.trx_90d_dst_type12_nunique, 0)::INTEGER AS trx_90d_dst_type12_nunique,

                COALESCE(tx.trx_all_has_event, 0)::TINYINT AS trx_all_has_event,
                COALESCE(tx.trx_all_event_count, 0)::INTEGER AS trx_all_event_count,
                COALESCE(tx.trx_all_log1p_event_count, 0.0)::FLOAT AS trx_all_log1p_event_count,
                COALESCE(tx.trx_all_active_days, 0)::SMALLINT AS trx_all_active_days,
                tx.trx_all_days_since_last_event,
                tx.trx_all_days_since_first_event,
                COALESCE(tx.trx_all_main_count, 0)::INTEGER AS trx_all_main_count,
                COALESCE(tx.trx_all_foreign_count, 0)::INTEGER AS trx_all_foreign_count,
                tx.trx_all_foreign_fraction,
                COALESCE(tx.trx_all_main_neg_count, 0)::INTEGER AS trx_all_main_neg_count,
                COALESCE(tx.trx_all_main_amount_abs_sum, 0.0)::DOUBLE AS trx_all_main_amount_abs_sum,
                COALESCE(tx.trx_all_log1p_main_amount_abs_sum, 0.0)::FLOAT AS trx_all_log1p_main_amount_abs_sum,
                tx.trx_all_main_amount_signed_log_mean,
                tx.trx_all_main_amount_median,
                tx.trx_all_main_amount_max,
                COALESCE(tx.trx_all_event_type_nunique, 0)::SMALLINT AS trx_all_event_type_nunique,
                COALESCE(tx.trx_all_event_subtype_nunique, 0)::SMALLINT AS trx_all_event_subtype_nunique,
                tx.trx_all_event_type_top_share,
                COALESCE(tx.trx_all_dst_type11_nunique, 0)::INTEGER AS trx_all_dst_type11_nunique,
                COALESCE(tx.trx_all_dst_type12_nunique, 0)::INTEGER AS trx_all_dst_type12_nunique,

                CASE WHEN tx.trx_all_days_since_first_event IS NULL THEN NULL
                     ELSE (tx.trx_all_event_count::DOUBLE / GREATEST(tx.trx_all_days_since_first_event / 30.0, 1.0))::FLOAT
                END AS trx_rate_per_30d_tenure,
                CASE WHEN COALESCE(tx.trx_90d_event_count, 0) > 0
                     THEN (COALESCE(tx.trx_30d_event_count, 0)::DOUBLE / tx.trx_90d_event_count::DOUBLE)::FLOAT
                     ELSE NULL
                END AS trx_share_30d_of_90d,
                CASE WHEN COALESCE(tx.trx_30d_event_count, 0) > 0
                      AND COALESCE(tx.trx_30d_event_count, 0) = COALESCE(tx.trx_all_event_count, 0)
                     THEN 1 ELSE 0 END::TINYINT AS trx_first_activated_30d,

                COALESCE(g.geo_30d_has_event, 0)::TINYINT AS geo_30d_has_event,
                COALESCE(g.geo_30d_event_count, 0)::INTEGER AS geo_30d_event_count,
                COALESCE(g.geo_30d_log1p_event_count, 0.0)::FLOAT AS geo_30d_log1p_event_count,
                COALESCE(g.geo_30d_active_days, 0)::SMALLINT AS geo_30d_active_days,
                g.geo_30d_days_since_last_event,
                g.geo_30d_days_since_first_event,
                COALESCE(g.geo_90d_has_event, 0)::TINYINT AS geo_90d_has_event,
                COALESCE(g.geo_90d_event_count, 0)::INTEGER AS geo_90d_event_count,
                COALESCE(g.geo_90d_log1p_event_count, 0.0)::FLOAT AS geo_90d_log1p_event_count,
                COALESCE(g.geo_90d_active_days, 0)::SMALLINT AS geo_90d_active_days,
                g.geo_90d_days_since_last_event,
                g.geo_90d_days_since_first_event,
                COALESCE(g.geo_all_has_event, 0)::TINYINT AS geo_all_has_event,
                COALESCE(g.geo_all_event_count, 0)::INTEGER AS geo_all_event_count,
                COALESCE(g.geo_all_log1p_event_count, 0.0)::FLOAT AS geo_all_log1p_event_count,
                COALESCE(g.geo_all_active_days, 0)::SMALLINT AS geo_all_active_days,
                g.geo_all_days_since_last_event,
                g.geo_all_days_since_first_event,
                COALESCE(g.geo_30d_geohash_4_nunique, 0)::INTEGER AS geo_30d_geohash_4_nunique,
                COALESCE(g.geo_30d_geohash_5_nunique, 0)::INTEGER AS geo_30d_geohash_5_nunique,
                COALESCE(g.geo_30d_geohash_6_nunique, 0)::INTEGER AS geo_30d_geohash_6_nunique,
                g.geo_30d_top_geohash_6_share,
                COALESCE(g.geo_90d_geohash_4_nunique, 0)::INTEGER AS geo_90d_geohash_4_nunique,
                COALESCE(g.geo_90d_geohash_5_nunique, 0)::INTEGER AS geo_90d_geohash_5_nunique,
                COALESCE(g.geo_90d_geohash_6_nunique, 0)::INTEGER AS geo_90d_geohash_6_nunique,
                g.geo_90d_top_geohash_6_share,
                COALESCE(g.geo_all_geohash_4_nunique, 0)::INTEGER AS geo_all_geohash_4_nunique,
                COALESCE(g.geo_all_geohash_5_nunique, 0)::INTEGER AS geo_all_geohash_5_nunique,
                COALESCE(g.geo_all_geohash_6_nunique, 0)::INTEGER AS geo_all_geohash_6_nunique,
                g.geo_all_top_geohash_6_share,

                COALESCE(d.dialog_30d_has_event, 0)::TINYINT AS dialog_30d_has_event,
                COALESCE(d.dialog_30d_event_count, 0)::INTEGER AS dialog_30d_event_count,
                COALESCE(d.dialog_30d_log1p_event_count, 0.0)::FLOAT AS dialog_30d_log1p_event_count,
                COALESCE(d.dialog_30d_active_days, 0)::SMALLINT AS dialog_30d_active_days,
                d.dialog_30d_days_since_last_event,
                d.dialog_30d_days_since_first_event,
                COALESCE(d.dialog_90d_has_event, 0)::TINYINT AS dialog_90d_has_event,
                COALESCE(d.dialog_90d_event_count, 0)::INTEGER AS dialog_90d_event_count,
                COALESCE(d.dialog_90d_log1p_event_count, 0.0)::FLOAT AS dialog_90d_log1p_event_count,
                COALESCE(d.dialog_90d_active_days, 0)::SMALLINT AS dialog_90d_active_days,
                d.dialog_90d_days_since_last_event,
                d.dialog_90d_days_since_first_event,
                COALESCE(d.dialog_all_has_event, 0)::TINYINT AS dialog_all_has_event,
                COALESCE(d.dialog_all_event_count, 0)::INTEGER AS dialog_all_event_count,
                COALESCE(d.dialog_all_log1p_event_count, 0.0)::FLOAT AS dialog_all_log1p_event_count,
                COALESCE(d.dialog_all_active_days, 0)::SMALLINT AS dialog_all_active_days,
                d.dialog_all_days_since_last_event,
                d.dialog_all_days_since_first_event,

                CASE WHEN COALESCE(tx.trx_all_has_event, 0) = 0
                      AND COALESCE(g.geo_all_has_event, 0) = 0
                      AND COALESCE(d.dialog_all_has_event, 0) = 0
                     THEN 1 ELSE 0 END::TINYINT AS cold_start
            FROM base b
            LEFT JOIN t_trx tx USING (client_id)
            LEFT JOIN t_geo g USING (client_id)
            LEFT JOIN t_dlg d USING (client_id)
            """
            fcon.execute(f"COPY ({join_q}) TO '{p_f.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            fcon.close()

    files_sql = ", ".join(f"'{p.as_posix()}'" for p in part_paths)
    tmp = out_path.with_suffix(".tmp.parquet")
    con.execute(f"COPY (SELECT * FROM read_parquet([{files_sql}]) ORDER BY month, client_id) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    tmp.replace(out_path)
    for p in part_paths:
        p.unlink(missing_ok=True)
    print(f"  [2/4] Built {out_path.name} in {time.perf_counter()-t0:.1f}s", flush=True)


def build_early_02_03_emb(out_npz: Path) -> None:
    if out_npz.exists():
        print(f"  [3/4] Reusing {out_npz.name}", flush=True)
        return
    t0 = time.perf_counter()
    all_buckets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {m: [] for m in EARLY_MONTHS}
    w90_buckets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {m: [] for m in EARLY_MONTHS}

    for fold in FOLDS:
        fold_glob = _glob(f"detail/dialog/fold={fold}/*.parquet")
        with duckdb.connect() as fcon:
            _configure_con(fcon, TMP_DIR)
            df = fcon.execute(
                f"""
                SELECT client_id, event_time + INTERVAL 3 HOUR AS event_time_msk, embedding
                FROM read_parquet('{fold_glob}')
                WHERE embedding IS NOT NULL AND event_time + INTERVAL 3 HOUR < TIMESTAMP '2022-03-01 00:00:00'
                """
            ).fetch_df()
        if len(df) == 0:
            continue
        emb_mat = np.vstack(df["embedding"].to_numpy()).astype(np.float32, copy=False)
        cids = df["client_id"].to_numpy(dtype=str)
        ts = df["event_time_msk"].to_numpy(dtype="datetime64[ns]")
        del df
        gc.collect()

        for m in EARLY_MONTHS:
            cutoff = np.datetime64(f"{m}-01T00:00:00", "ns")
            lo90 = cutoff - np.timedelta64(90, "D")
            for mask, bucket in ((ts < cutoff, all_buckets[m]), ((ts >= lo90) & (ts < cutoff), w90_buckets[m])):
                if np.any(mask):
                    sub_c = cids[mask]
                    sub_e = emb_mat[mask]
                    codes, uniques = pd.factorize(sub_c, sort=True)
                    sums = np.zeros((len(uniques), emb_mat.shape[1]), dtype=np.float64)
                    np.add.at(sums, codes, sub_e)
                    means = (sums / np.bincount(codes, minlength=len(uniques))[:, None]).astype(np.float16)
                    bucket.append((uniques.astype(str), means))
        del emb_mat, cids, ts
        gc.collect()

    def _pack(buckets: dict[str, list[tuple[np.ndarray, np.ndarray]]]):
        ms, cs, vs = [], [], []
        for m in EARLY_MONTHS:
            if not buckets[m]:
                continue
            cm = np.concatenate([x[0] for x in buckets[m]])
            vm = np.vstack([x[1] for x in buckets[m]])
            order = np.argsort(cm, kind="mergesort")
            ms.append(np.full(len(cm), m, dtype="U7"))
            cs.append(cm[order])
            vs.append(vm[order])
        return np.concatenate(ms), np.concatenate(cs), np.vstack(vs)

    am, ac, av = _pack(all_buckets)
    wm, wc, wv = _pack(w90_buckets)
    np.savez_compressed(out_npz, all_months=am, all_clients=ac, all_emb=av, w90_months=wm, w90_clients=wc, w90_emb=wv)
    print(f"  [3/4] Built {out_npz.name} in {time.perf_counter()-t0:.1f}s (all={len(am):,}, 90d={len(wm):,})", flush=True)


def build_verify_aux(con: duckdb.DuckDBPyConnection, tgt_path: Path, out_path: Path) -> None:
    if out_path.exists():
        print(f"  [4/4] Reusing {out_path.name}", flush=True)
        return
    t0 = time.perf_counter()
    tgt = pd.read_parquet(tgt_path, columns=["client_id", "fold", "month"])

    # 1) Dialog lag metadata for lag = 7d, 14d, 30d across ACTIVE_H3_MONTHS
    dlg_glob = _glob("detail/dialog/*/*.parquet")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE dlg_all AS
        SELECT client_id, event_time + INTERVAL 3 HOUR AS ts_msk
        FROM read_parquet('{dlg_glob}')
        WHERE event_time + INTERVAL 3 HOUR < TIMESTAMP '2022-11-01 00:00:00'
        """
    )
    lag_dfs = []
    for m in ACTIVE_H3_MONTHS:
        c_orig = f"TIMESTAMP '{m}-01 00:00:00'"
        select_cols = ["client_id", f"'{m}' AS month"]
        for lag in (7, 14, 30):
            c_eff = f"({c_orig} - INTERVAL {lag} DAY)"
            select_cols.extend([
                f"CAST(COUNT(*) FILTER (WHERE ts_msk < {c_eff} AND ts_msk >= {c_eff} - INTERVAL 30 DAY) > 0 AS TINYINT) AS dlg_lag{lag}_30d_has",
                f"CAST(COUNT(*) FILTER (WHERE ts_msk < {c_eff} AND ts_msk >= {c_eff} - INTERVAL 30 DAY) AS INTEGER) AS dlg_lag{lag}_30d_cnt",
                f"CAST(EPOCH({c_orig} - MAX(ts_msk) FILTER (WHERE ts_msk < {c_eff} AND ts_msk >= {c_eff} - INTERVAL 30 DAY)) / 86400.0 AS FLOAT) AS dlg_lag{lag}_30d_rec",
                f"CAST(COUNT(*) FILTER (WHERE ts_msk < {c_eff} AND ts_msk >= {c_eff} - INTERVAL 90 DAY) > 0 AS TINYINT) AS dlg_lag{lag}_90d_has",
                f"CAST(COUNT(*) FILTER (WHERE ts_msk < {c_eff} AND ts_msk >= {c_eff} - INTERVAL 90 DAY) AS INTEGER) AS dlg_lag{lag}_90d_cnt",
                f"CAST(EPOCH({c_orig} - MAX(ts_msk) FILTER (WHERE ts_msk < {c_eff} AND ts_msk >= {c_eff} - INTERVAL 90 DAY)) / 86400.0 AS FLOAT) AS dlg_lag{lag}_90d_rec",
                f"CAST(COUNT(*) FILTER (WHERE ts_msk < {c_eff}) AS INTEGER) AS dlg_lag{lag}_all_cnt",
                f"CAST(EPOCH({c_orig} - MAX(ts_msk) FILTER (WHERE ts_msk < {c_eff})) / 86400.0 AS FLOAT) AS dlg_lag{lag}_all_rec",
            ])
        df_m = con.execute(
            f"SELECT {', '.join(select_cols)} FROM dlg_all WHERE ts_msk < {c_orig} GROUP BY client_id"
        ).df()
        lag_dfs.append(df_m)
    dlg_lag_df = pd.concat(lag_dfs, ignore_index=True)

    # 2) UTC 00:00 cutoff TRX RFM check cols (vs MSK 00:00) on ACTIVE_H3_MONTHS
    utc_dfs = []
    for fold in FOLDS:
        trx_glob = _glob(f"detail/trx/fold={fold}/*.parquet")
        for m in ACTIVE_H3_MONTHS:
            c_utc = f"TIMESTAMP '{m}-01 00:00:00'"
            df_u = con.execute(
                f"""
                SELECT
                    client_id, '{m}' AS month,
                    CAST(COUNT(*) FILTER (WHERE event_time >= {c_utc} - INTERVAL 30 DAY) AS INTEGER) AS utc_trx_30d_event_count,
                    CAST(LN(1.0 + COUNT(*) FILTER (WHERE event_time >= {c_utc} - INTERVAL 30 DAY)) AS FLOAT) AS utc_trx_30d_log1p_event_count,
                    CAST(EPOCH({c_utc} - MAX(event_time) FILTER (WHERE event_time >= {c_utc} - INTERVAL 30 DAY)) / 86400.0 AS FLOAT) AS utc_trx_30d_days_since_last_event,
                    CAST(LN(1.0 + COALESCE(SUM(ABS(amount)) FILTER (WHERE event_time >= {c_utc} - INTERVAL 30 DAY AND currency = 11 AND isfinite(amount)), 0.0)) AS FLOAT) AS utc_trx_30d_log1p_main_amount_abs_sum,
                    CAST(LN(1.0 + COUNT(*) FILTER (WHERE event_time >= {c_utc} - INTERVAL 90 DAY)) AS FLOAT) AS utc_trx_90d_log1p_event_count,
                    CAST(EPOCH({c_utc} - MAX(event_time) FILTER (WHERE event_time >= {c_utc} - INTERVAL 90 DAY)) / 86400.0 AS FLOAT) AS utc_trx_90d_days_since_last_event,
                    CAST(COUNT(*) AS INTEGER) AS utc_trx_all_event_count,
                    CAST(EPOCH({c_utc} - MAX(event_time)) / 86400.0 AS FLOAT) AS utc_trx_all_days_since_last_event,
                    CAST(EPOCH({c_utc} - MIN(event_time)) / 86400.0 AS FLOAT) AS utc_trx_all_days_since_first_event
                FROM read_parquet('{trx_glob}')
                WHERE event_time < {c_utc}
                GROUP BY client_id
                """
            ).df()
            utc_dfs.append(df_u)
    utc_df = pd.concat(utc_dfs, ignore_index=True)

    aux = tgt.merge(dlg_lag_df, on=["client_id", "month"], how="left").merge(utc_df, on=["client_id", "month"], how="left")
    for c in aux.columns:
        if c in ("client_id", "fold", "month"):
            continue
        if c.endswith(("_has", "_cnt", "_count")):
            aux[c] = aux[c].fillna(0).astype(np.float32)
        else:
            aux[c] = aux[c].astype(np.float32)

    tmp = out_path.with_suffix(".tmp.parquet")
    aux.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(out_path)
    print(f"  [4/4] Built {out_path.name} in {time.perf_counter()-t0:.1f}s ({len(aux):,} rows, {aux.shape[1]} cols)", flush=True)


def main() -> None:
    t_all = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DELTA_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    _configure_con(con, TMP_DIR)

    tgt_path = DELTA_DIR / "h3_targets_prior.parquet"
    early_path = DELTA_DIR / "h3_early_02_03_base112.parquet"
    emb_path = DELTA_DIR / "h3_early_02_03_emb.npz"
    aux_path = DELTA_DIR / "h3_verify_aux.parquet"

    build_h3_targets_and_prior(con, tgt_path)
    build_early_02_03_base112(con, tgt_path, early_path)
    build_early_02_03_emb(emb_path)
    build_verify_aux(con, tgt_path, aux_path)
    con.close()

    zip_path = OUT_DIR / "h3_stage1_delta.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in (tgt_path, early_path, emb_path, aux_path):
            zf.write(p, arcname=p.name)

    mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[DONE] Packed {zip_path} ({mb:.2f} MB) in {time.perf_counter()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
