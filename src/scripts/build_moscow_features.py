"""Build leak-free Moscow-time (UTC+3) monthly panel features and dialog embeddings
directly from data/MBD-mini/ for the consolidated baseline ladder (Tiers 0..5 + Target-lag Ref).

Timezone rule:
  Raw event_time is UTC; true label month boundary (targets.mon) is 00:00:00 Moscow (UTC+3).
  We shift event_time_msk = event_time + INTERVAL 3 HOUR and enforce strictly:
      event_time_msk < TIMESTAMP '{YYYY-MM}-01 00:00:00'
  (= prev_month_last_day 21:00:00 UTC), eliminating the 3-hour boundary leakage.

Windows:
  30d, 90d, all (from 2021-01-01 00:00:00 MSK strictly before cutoff_msk).

Outputs under outputs/baseline_moscow_v1/:
  - panel_moscow.parquet       : 10 modeling months (2022-04..2023-01) x 100,224 clients
  - dialog_emb_moscow.npz      : Sparse mean-pool 768d embeddings (all & 90d) keyed by (month, client_id)
  - moscow_audit.json          : Exact UTC vs UTC+3 boundary audit & schema manifest
"""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "MBD-mini"
OUT_DIR = REPO_ROOT / "outputs" / "baseline_moscow_v1"

ALL_TARGET_MONTHS = (
    "2022-02", "2022-03", "2022-04", "2022-05", "2022-06", "2022-07",
    "2022-08", "2022-09", "2022-10", "2022-11", "2022-12", "2023-01",
)
MODELING_MONTHS = (
    "2022-04", "2022-05", "2022-06", "2022-07", "2022-08",
    "2022-09", "2022-10", "2022-11", "2022-12", "2023-01",
)
FOLDS = (0, 1, 2, 3, 4)
DUCKDB_MEMORY_LIMIT = "2.5GB"
DUCKDB_THREADS = 4


def _configure_con(con: duckdb.DuckDBPyConnection, tmp_dir: Path) -> None:
    """Configure DuckDB to stay within physical RAM and avoid OS pagefile thrashing."""
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{tmp_dir.as_posix()}'")


def _glob(rel: str) -> str:
    return (DATA_ROOT / rel).as_posix()


def build_targets_and_prior(con: duckdb.DuckDBPyConnection, out_path: Path) -> dict[str, Any]:
    """Load 12 target months + official client_split fold and compute strict m < T target-lag (1/3/6/all)."""
    targets_glob = _glob("targets/*/*.parquet")
    folds_glob = _glob("client_split/*/*.parquet")
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
    lagged AS (
        SELECT
            client_id,
            fold,
            month,
            cutoff_msk,
            y_any,
            target_1, target_2, target_3, target_4,
            -- Strict target-lag features (m < T): 1m, 3m, 6m, and expanding all
            CAST(COALESCE(LAG(y_any, 1) OVER w_ord, 0) AS TINYINT) AS prior_1m_bought_any,
            CAST(COALESCE(SUM(y_any) OVER w_p3, 0) AS TINYINT) AS prior_3m_buy_months,
            CAST(COALESCE(SUM(target_1 + target_2 + target_3 + target_4) OVER w_p3, 0) AS TINYINT) AS prior_3m_total_products,
            CAST(COALESCE(SUM(y_any) OVER w_p6, 0) AS TINYINT) AS prior_6m_buy_months,
            CAST(COALESCE(SUM(target_1 + target_2 + target_3 + target_4) OVER w_p6, 0) AS TINYINT) AS prior_6m_total_products,
            CAST(CASE WHEN COALESCE(SUM(y_any) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END AS TINYINT) AS prior_all_bought_any,
            CAST(COALESCE(SUM(y_any) OVER w_pall, 0) AS SMALLINT) AS prior_all_buy_months,
            CAST(month_idx - MAX(CASE WHEN y_any = 1 THEN month_idx ELSE NULL END) OVER w_pall AS DOUBLE) AS prior_months_since_last_buy,
            CAST(CASE WHEN COALESCE(SUM(target_1) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END AS TINYINT) AS prior_all_bought_p1,
            CAST(CASE WHEN COALESCE(SUM(target_2) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END AS TINYINT) AS prior_all_bought_p2,
            CAST(CASE WHEN COALESCE(SUM(target_3) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END AS TINYINT) AS prior_all_bought_p3,
            CAST(CASE WHEN COALESCE(SUM(target_4) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END AS TINYINT) AS prior_all_bought_p4,
            CAST(
                (CASE WHEN COALESCE(SUM(target_1) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END)
                + (CASE WHEN COALESCE(SUM(target_2) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END)
                + (CASE WHEN COALESCE(SUM(target_3) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END)
                + (CASE WHEN COALESCE(SUM(target_4) OVER w_pall, 0) > 0 THEN 1 ELSE 0 END)
                AS TINYINT
            ) AS prior_all_distinct_products,
            CAST(COUNT(y_any) OVER w_pall AS SMALLINT) AS prior_months_observed
        FROM tgt
        WINDOW
            w_ord  AS (PARTITION BY client_id ORDER BY cutoff_msk),
            w_p3   AS (PARTITION BY client_id ORDER BY cutoff_msk ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING),
            w_p6   AS (PARTITION BY client_id ORDER BY cutoff_msk ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING),
            w_pall AS (PARTITION BY client_id ORDER BY cutoff_msk ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    )
    SELECT * FROM lagged
    WHERE month IN ({", ".join(f"'{m}'" for m in MODELING_MONTHS)})
    ORDER BY month, client_id
    """
    con.execute(f"COPY ({q}) TO '{out_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    summary = con.execute(
        f"""
        SELECT month, COUNT(*) AS n, SUM(y_any) AS pos,
               MIN(prior_months_observed) AS min_prior_m
        FROM read_parquet('{out_path.as_posix()}')
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    return {
        "rows": int(sum(r[1] for r in summary)),
        "monthly": [{"month": r[0], "n": int(r[1]), "positives": int(r[2]), "prior_months_available": int(r[3])} for r in summary],
    }


def _build_trx_month_tables(con: duckdb.DuckDBPyConnection, trx_glob: str, cutoff_str: str) -> None:
    """Streaming single-pass TRX aggregation (no multi-use CTE spill) strictly < cutoff_msk."""
    c = f"TIMESTAMP '{cutoff_str}'"
    c_utc = f"({c} - INTERVAL 3 HOUR)"
    windows = [
        ("30d", f"event_time >= {c_utc} - INTERVAL 30 DAY"),
        ("90d", f"event_time >= {c_utc} - INTERVAL 90 DAY"),
        ("all", "TRUE"),
    ]
    exprs: list[str] = ["client_id"]
    for w, cond in windows:
        p = f"trx_{w}_"
        exprs.extend([
            f"CAST(CASE WHEN COUNT(*) FILTER (WHERE {cond}) > 0 THEN 1 ELSE 0 END AS TINYINT) AS {p}has_event",
            f"CAST(COUNT(*) FILTER (WHERE {cond}) AS INTEGER) AS {p}event_count",
            f"CAST(LN(1.0 + COUNT(*) FILTER (WHERE {cond})) AS FLOAT) AS {p}log1p_event_count",
            f"CAST(COUNT(DISTINCT CAST(event_time + INTERVAL 3 HOUR AS DATE)) FILTER (WHERE {cond}) AS SMALLINT) AS {p}active_days",
            f"CAST(EPOCH({c_utc} - MAX(event_time) FILTER (WHERE {cond})) / 86400.0 AS FLOAT) AS {p}days_since_last_event",
            f"CAST(EPOCH({c_utc} - MIN(event_time) FILTER (WHERE {cond})) / 86400.0 AS FLOAT) AS {p}days_since_first_event",
            f"CAST(COUNT(*) FILTER (WHERE {cond} AND currency = 11) AS INTEGER) AS {p}main_count",
            f"CAST(COUNT(*) FILTER (WHERE {cond} AND currency IS NOT NULL AND currency <> 11) AS INTEGER) AS {p}foreign_count",
            f"CAST(COUNT(*) FILTER (WHERE {cond} AND currency IS NOT NULL AND currency <> 11)::DOUBLE / NULLIF(COUNT(currency) FILTER (WHERE {cond}), 0) AS FLOAT) AS {p}foreign_fraction",
            f"CAST(COUNT(*) FILTER (WHERE {cond} AND currency = 11 AND amount < 0) AS INTEGER) AS {p}main_neg_count",
            f"CAST(SUM(ABS(amount)) FILTER (WHERE {cond} AND currency = 11 AND isfinite(amount)) AS DOUBLE) AS {p}main_amount_abs_sum",
            f"CAST(LN(1.0 + COALESCE(SUM(ABS(amount)) FILTER (WHERE {cond} AND currency = 11 AND isfinite(amount)), 0.0)) AS FLOAT) AS {p}log1p_main_amount_abs_sum",
            f"CAST(AVG(SIGN(amount) * LN(1.0 + ABS(amount))) FILTER (WHERE {cond} AND currency = 11 AND isfinite(amount)) AS FLOAT) AS {p}main_amount_signed_log_mean",
            f"CAST(APPROX_QUANTILE(amount, 0.5) FILTER (WHERE {cond} AND currency = 11 AND isfinite(amount)) AS FLOAT) AS {p}main_amount_median",
            f"CAST(MAX(amount) FILTER (WHERE {cond} AND currency = 11 AND isfinite(amount)) AS FLOAT) AS {p}main_amount_max",
            f"CAST(COUNT(DISTINCT event_subtype) FILTER (WHERE {cond}) AS SMALLINT) AS {p}event_subtype_nunique",
            f"CAST(COUNT(DISTINCT dst_type11) FILTER (WHERE {cond}) AS INTEGER) AS {p}dst_type11_nunique",
            f"CAST(COUNT(DISTINCT dst_type12) FILTER (WHERE {cond}) AS INTEGER) AS {p}dst_type12_nunique",
        ])

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_trx_base AS
        SELECT {', '.join(exprs)}
        FROM read_parquet('{trx_glob}')
        WHERE event_time < {c_utc}
        GROUP BY client_id
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_trx_et AS
        WITH counts AS (
            SELECT
                client_id,
                event_type,
                COUNT(*) FILTER (WHERE event_time >= {c_utc} - INTERVAL 30 DAY) AS n30,
                COUNT(*) FILTER (WHERE event_time >= {c_utc} - INTERVAL 90 DAY) AS n90,
                COUNT(*) AS nall
            FROM read_parquet('{trx_glob}')
            WHERE event_time < {c_utc} AND event_type IS NOT NULL
            GROUP BY 1, 2
        )
        SELECT
            client_id,
            CAST(COUNT(*) FILTER (WHERE n30 > 0) AS SMALLINT) AS trx_30d_event_type_nunique,
            CAST(COUNT(*) FILTER (WHERE n90 > 0) AS SMALLINT) AS trx_90d_event_type_nunique,
            CAST(COUNT(*) FILTER (WHERE nall > 0) AS SMALLINT) AS trx_all_event_type_nunique,
            (MAX(n30)::DOUBLE / NULLIF(SUM(n30), 0))::FLOAT AS trx_30d_event_type_top_share,
            (MAX(n90)::DOUBLE / NULLIF(SUM(n90), 0))::FLOAT AS trx_90d_event_type_top_share,
            (MAX(nall)::DOUBLE / NULLIF(SUM(nall), 0))::FLOAT AS trx_all_event_type_top_share
        FROM counts
        GROUP BY 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE t_trx AS
        SELECT b.*, e.* EXCLUDE (client_id)
        FROM t_trx_base b
        LEFT JOIN t_trx_et e USING (client_id)
        """
    )
    con.execute("DROP TABLE t_trx_base; DROP TABLE t_trx_et;")


def _build_geo_month_tables(con: duckdb.DuckDBPyConnection, geo_glob: str, cutoff_str: str) -> None:
    """Streaming single-pass GEO metadata + content aggregation strictly < cutoff_msk."""
    c = f"TIMESTAMP '{cutoff_str}'"
    c_utc = f"({c} - INTERVAL 3 HOUR)"
    windows = [
        ("30d", f"event_time >= {c_utc} - INTERVAL 30 DAY"),
        ("90d", f"event_time >= {c_utc} - INTERVAL 90 DAY"),
        ("all", "TRUE"),
    ]
    exprs: list[str] = ["client_id"]
    for w, cond in windows:
        p = f"geo_{w}_"
        exprs.extend([
            f"CAST(CASE WHEN COUNT(*) FILTER (WHERE {cond}) > 0 THEN 1 ELSE 0 END AS TINYINT) AS {p}has_event",
            f"CAST(COUNT(*) FILTER (WHERE {cond}) AS INTEGER) AS {p}event_count",
            f"CAST(LN(1.0 + COUNT(*) FILTER (WHERE {cond})) AS FLOAT) AS {p}log1p_event_count",
            f"CAST(COUNT(DISTINCT CAST(event_time + INTERVAL 3 HOUR AS DATE)) FILTER (WHERE {cond}) AS SMALLINT) AS {p}active_days",
            f"CAST(EPOCH({c_utc} - MAX(event_time) FILTER (WHERE {cond})) / 86400.0 AS FLOAT) AS {p}days_since_last_event",
            f"CAST(EPOCH({c_utc} - MIN(event_time) FILTER (WHERE {cond})) / 86400.0 AS FLOAT) AS {p}days_since_first_event",
            f"CAST(COUNT(DISTINCT geohash_4) FILTER (WHERE {cond}) AS INTEGER) AS {p}geohash_4_nunique",
            f"CAST(COUNT(DISTINCT geohash_5) FILTER (WHERE {cond}) AS INTEGER) AS {p}geohash_5_nunique",
        ])

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_geo_base AS
        SELECT {', '.join(exprs)}
        FROM read_parquet('{geo_glob}')
        WHERE event_time < {c_utc}
        GROUP BY client_id
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE t_geo_g6 AS
        WITH counts AS (
            SELECT
                client_id,
                geohash_6,
                COUNT(*) FILTER (WHERE event_time >= {c_utc} - INTERVAL 30 DAY) AS n30,
                COUNT(*) FILTER (WHERE event_time >= {c_utc} - INTERVAL 90 DAY) AS n90,
                COUNT(*) AS nall
            FROM read_parquet('{geo_glob}')
            WHERE event_time < {c_utc} AND geohash_6 IS NOT NULL
            GROUP BY 1, 2
        )
        SELECT
            client_id,
            CAST(COUNT(*) FILTER (WHERE n30 > 0) AS INTEGER) AS geo_30d_geohash_6_nunique,
            CAST(COUNT(*) FILTER (WHERE n90 > 0) AS INTEGER) AS geo_90d_geohash_6_nunique,
            CAST(COUNT(*) FILTER (WHERE nall > 0) AS INTEGER) AS geo_all_geohash_6_nunique,
            (MAX(n30)::DOUBLE / NULLIF(SUM(n30), 0))::FLOAT AS geo_30d_top_geohash_6_share,
            (MAX(n90)::DOUBLE / NULLIF(SUM(n90), 0))::FLOAT AS geo_90d_top_geohash_6_share,
            (MAX(nall)::DOUBLE / NULLIF(SUM(nall), 0))::FLOAT AS geo_all_top_geohash_6_share
        FROM counts
        GROUP BY 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE t_geo AS
        SELECT b.*, g.* EXCLUDE (client_id)
        FROM t_geo_base b
        LEFT JOIN t_geo_g6 g USING (client_id)
        """
    )
    con.execute("DROP TABLE t_geo_base; DROP TABLE t_geo_g6;")


def _dialog_month_sql(dialog_glob: str, cutoff_str: str) -> str:
    """Single-scan multi-window (30d, 90d, all) DIALOG metadata strictly < cutoff_msk."""
    c = f"TIMESTAMP '{cutoff_str}'"
    windows = [
        ("30d", f"event_time_msk >= {c} - INTERVAL 30 DAY"),
        ("90d", f"event_time_msk >= {c} - INTERVAL 90 DAY"),
        ("all", "TRUE"),
    ]
    exprs: list[str] = ["client_id"]
    for w, cond in windows:
        p = f"dialog_{w}_"
        exprs.extend([
            f"CAST(CASE WHEN COUNT(*) FILTER (WHERE {cond}) > 0 THEN 1 ELSE 0 END AS TINYINT) AS {p}has_event",
            f"CAST(COUNT(*) FILTER (WHERE {cond}) AS INTEGER) AS {p}event_count",
            f"CAST(LN(1.0 + COUNT(*) FILTER (WHERE {cond})) AS FLOAT) AS {p}log1p_event_count",
            f"CAST(COUNT(DISTINCT CAST(event_time_msk AS DATE)) FILTER (WHERE {cond}) AS SMALLINT) AS {p}active_days",
            f"CAST(EPOCH({c} - MAX(event_time_msk) FILTER (WHERE {cond})) / 86400.0 AS FLOAT) AS {p}days_since_last_event",
            f"CAST(EPOCH({c} - MIN(event_time_msk) FILTER (WHERE {cond})) / 86400.0 AS FLOAT) AS {p}days_since_first_event",
        ])
    return f"""
    SELECT {', '.join(exprs)}
    FROM (
        SELECT client_id, event_time + INTERVAL 3 HOUR AS event_time_msk
        FROM read_parquet('{dialog_glob}')
        WHERE event_time + INTERVAL 3 HOUR < {c}
    )
    GROUP BY client_id
    """


def build_dialog_embeddings_moscow(out_npz: Path) -> dict[str, Any]:
    """Compute leak-free Moscow-time mean-pool 768d dialog embeddings (all & 90d) for all modeling months.

    Processes one disjoint client fold (fold=0..4) at a time to keep peak RAM < 500 MB
    while producing identical (month, client_id)-sorted outputs.
    """
    if out_npz.exists():
        data = np.load(out_npz, allow_pickle=False)
        return {
            "cached": True,
            "n_all_pairs": int(len(data["all_months"])),
            "n_90d_pairs": int(len(data["w90_months"])),
            "dim": int(data["all_emb"].shape[1]),
        }

    t0 = time.perf_counter()
    all_buckets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {m: [] for m in MODELING_MONTHS}
    w90_buckets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {m: [] for m in MODELING_MONTHS}
    total_raw_rows = 0

    for fold in FOLDS:
        fold_glob = _glob(f"detail/dialog/fold={fold}/*.parquet")
        with duckdb.connect() as con:
            con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
            con.execute(f"SET threads={DUCKDB_THREADS}")
            con.execute("SET preserve_insertion_order=false")
            df = con.execute(
                f"""
                SELECT client_id,
                       event_time + INTERVAL 3 HOUR AS event_time_msk,
                       embedding
                FROM read_parquet('{fold_glob}')
                WHERE embedding IS NOT NULL
                """
            ).fetch_df()

        if len(df) == 0:
            continue
        total_raw_rows += len(df)
        emb_mat = np.vstack(df["embedding"].to_numpy()).astype(np.float32, copy=False)
        cids = df["client_id"].to_numpy(dtype=str)
        ts = df["event_time_msk"].to_numpy(dtype="datetime64[ns]")
        del df
        gc.collect()

        for m in MODELING_MONTHS:
            cutoff = np.datetime64(f"{m}-01T00:00:00", "ns")
            lo90 = cutoff - np.timedelta64(90, "D")

            mask_all = ts < cutoff
            if np.any(mask_all):
                sub_c = cids[mask_all]
                sub_e = emb_mat[mask_all]
                codes, uniques = pd.factorize(sub_c, sort=True)
                sums = np.zeros((len(uniques), emb_mat.shape[1]), dtype=np.float64)
                np.add.at(sums, codes, sub_e)
                counts = np.bincount(codes, minlength=len(uniques))[:, None]
                means = (sums / counts).astype(np.float32)
                all_buckets[m].append((uniques.astype(str), means))

            mask_90 = (ts >= lo90) & (ts < cutoff)
            if np.any(mask_90):
                sub_c = cids[mask_90]
                sub_e = emb_mat[mask_90]
                codes, uniques = pd.factorize(sub_c, sort=True)
                sums = np.zeros((len(uniques), emb_mat.shape[1]), dtype=np.float64)
                np.add.at(sums, codes, sub_e)
                counts = np.bincount(codes, minlength=len(uniques))[:, None]
                means = (sums / counts).astype(np.float32)
                w90_buckets[m].append((uniques.astype(str), means))

        del emb_mat, cids, ts
        gc.collect()

    all_m_list, all_c_list, all_v_list = [], [], []
    w90_m_list, w90_c_list, w90_v_list = [], [], []
    for m in MODELING_MONTHS:
        if all_buckets[m]:
            c_m = np.concatenate([x[0] for x in all_buckets[m]])
            v_m = np.vstack([x[1] for x in all_buckets[m]])
            order = np.argsort(c_m, kind="mergesort")
            all_m_list.append(np.full(len(c_m), m, dtype="U7"))
            all_c_list.append(c_m[order])
            all_v_list.append(v_m[order])
            all_buckets[m].clear()
        if w90_buckets[m]:
            c_m = np.concatenate([x[0] for x in w90_buckets[m]])
            v_m = np.vstack([x[1] for x in w90_buckets[m]])
            order = np.argsort(c_m, kind="mergesort")
            w90_m_list.append(np.full(len(c_m), m, dtype="U7"))
            w90_c_list.append(c_m[order])
            w90_v_list.append(v_m[order])
            w90_buckets[m].clear()

    all_months = np.concatenate(all_m_list)
    all_clients = np.concatenate(all_c_list)
    all_emb = np.vstack(all_v_list)

    w90_months = np.concatenate(w90_m_list)
    w90_clients = np.concatenate(w90_c_list)
    w90_emb = np.vstack(w90_v_list)

    np.savez_compressed(
        out_npz,
        all_months=all_months,
        all_clients=all_clients,
        all_emb=all_emb,
        w90_months=w90_months,
        w90_clients=w90_clients,
        w90_emb=w90_emb,
    )
    return {
        "cached": False,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "raw_dialog_rows": int(total_raw_rows),
        "n_all_pairs": int(len(all_months)),
        "n_90d_pairs": int(len(w90_months)),
        "dim": int(all_emb.shape[1]),
    }


def compute_timezone_boundary_audit(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Quantify the exact 3-hour UTC vs UTC+3 boundary leakage eliminated by Moscow alignment."""
    audit: dict[str, Any] = {}
    for src in ("trx", "geo", "dialog"):
        pat = _glob(f"detail/{src}/*/*.parquet")
        row = con.execute(
            f"""
            SELECT
                MIN(event_time) AS min_utc,
                MAX(event_time) AS max_utc,
                MIN(event_time + INTERVAL 3 HOUR) AS min_msk,
                MAX(event_time + INTERVAL 3 HOUR) AS max_msk,
                COUNT(*) AS total_events,
                COUNT(*) FILTER (
                    WHERE EXTRACT(DAY FROM (event_time + INTERVAL 3 HOUR)) = 1
                      AND EXTRACT(HOUR FROM event_time) >= 21
                      AND event_time >= TIMESTAMP '2022-01-31 21:00:00'
                      AND event_time <  TIMESTAMP '2023-02-01 00:00:00'
                ) AS leaked_events_utc_cutoff,
                COUNT(DISTINCT CASE
                    WHEN EXTRACT(DAY FROM (event_time + INTERVAL 3 HOUR)) = 1
                     AND EXTRACT(HOUR FROM event_time) >= 21
                     AND event_time >= TIMESTAMP '2022-01-31 21:00:00'
                     AND event_time <  TIMESTAMP '2023-02-01 00:00:00'
                    THEN client_id || '_' || strftime(event_time + INTERVAL 3 HOUR, '%Y-%m')
                END) AS affected_client_months
            FROM read_parquet('{pat}')
            """
        ).fetchone()
        audit[src] = {
            "min_utc": str(row[0]),
            "max_utc": str(row[1]),
            "min_msk": str(row[2]),
            "max_msk": str(row[3]),
            "total_events": int(row[4]),
            "leaked_events_under_utc_cutoff": int(row[5]),
            "affected_client_months": int(row[6]),
        }
    return audit


def main():
    t_all = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly_dir = OUT_DIR / "monthly_moscow"
    monthly_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    tmp_dir = OUT_DIR / "duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _configure_con(con, tmp_dir)

    # 1. Timezone boundary audit (UTC vs Moscow UTC+3)
    print("[1/4] Computing UTC vs Moscow (UTC+3) boundary audit ...", flush=True)
    tz_audit = compute_timezone_boundary_audit(con)
    for k, v in tz_audit.items():
        print(
            f"  {k:6s} | MSK=[{v['min_msk'][:19]} .. {v['max_msk'][:19]}] "
            f"| Leaked under UTC cutoff: {v['leaked_events_under_utc_cutoff']:,} events "
            f"({v['affected_client_months']:,} client-months)",
            flush=True,
        )

    # 2. Targets + fold + strict prior_purchase (1m, 3m, 6m, all)
    print("[2/4] Building targets + client_split folds + strict 1/3/6m target-lag ...", flush=True)
    targets_path = OUT_DIR / "targets_prior_moscow.parquet"
    tgt_info = build_targets_and_prior(con, targets_path)
    print(f"  Targets ready: {tgt_info['rows']:,} modeling rows (2022-04..2023-01)", flush=True)

    # 3. Monthly Moscow-time features for TRX, GEO (meta+content), DIALOG (meta)
    # Process one disjoint client fold (fold=0..4) at a time to reduce RAM 5x and avoid OS swap thrashing.
    print("[3/4] Aggregating Moscow-time (UTC+3) 30d/90d/all features per month (fold-partitioned) ...", flush=True)

    panel_path = OUT_DIR / "panel_moscow.parquet"
    if not panel_path.exists():
        month_files: list[str] = []
        for m in MODELING_MONTHS:
            m_path = monthly_dir / f"panel_{m}.parquet"
            month_files.append(m_path.as_posix())
            if m_path.exists():
                print(f"  [{m}] cached -> {m_path.name}", flush=True)
                continue
            tm = time.perf_counter()
            cutoff_str = f"{m}-01 00:00:00"
            fold_paths: list[Path] = []
            for fold in FOLDS:
                tf = time.perf_counter()
                trx_glob = _glob(f"detail/trx/fold={fold}/*.parquet")
                geo_glob = _glob(f"detail/geo/fold={fold}/*.parquet")
                dlg_glob = _glob(f"detail/dialog/fold={fold}/*.parquet")
                f_path = monthly_dir / f".tmp_panel_{m}_f{fold}.parquet"
                fold_paths.append(f_path)

                fcon = duckdb.connect()
                _configure_con(fcon, tmp_dir)
                _build_trx_month_tables(fcon, trx_glob, cutoff_str)
                _build_geo_month_tables(fcon, geo_glob, cutoff_str)
                fcon.execute(f"CREATE OR REPLACE TEMP TABLE t_dlg AS {_dialog_month_sql(dlg_glob, cutoff_str)}")

                # Left join onto all clients in fold for month m; fill counts/flags with 0, keep recency/median/shares NaN
                join_q = f"""
                WITH base AS (
                    SELECT * FROM read_parquet('{targets_path.as_posix()}')
                    WHERE month = '{m}' AND fold = {fold}
                )
                SELECT
                    b.*,
                    -- TRX (30d, 90d, all)
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

                    -- Pure TRX cross-window features
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

                    -- GEO metadata (activity only)
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

                    -- GEO content (spatial diversity & concentration)
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

                    -- DIALOG metadata (activity only)
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

                    -- Cold start flag (no events across all 3 sources prior to cutoff_msk)
                    CASE WHEN COALESCE(tx.trx_all_has_event, 0) = 0
                          AND COALESCE(g.geo_all_has_event, 0) = 0
                          AND COALESCE(d.dialog_all_has_event, 0) = 0
                         THEN 1 ELSE 0 END::TINYINT AS cold_start
                FROM base b
                LEFT JOIN t_trx tx USING (client_id)
                LEFT JOIN t_geo g USING (client_id)
                LEFT JOIN t_dlg d USING (client_id)
                """
                fcon.execute(f"COPY ({join_q}) TO '{f_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
                fcon.close()
                print(f"    [{m} fold={fold}] {time.perf_counter() - tf:.1f}s", flush=True)

            # Combine the 5 fold slices ordered by client_id and atomically commit panel_{m}.parquet
            fold_sql = ", ".join(f"'{p.as_posix()}'" for p in fold_paths)
            m_tmp = monthly_dir / f".tmp_panel_{m}.parquet"
            con.execute(
                f"""
                COPY (
                    SELECT * FROM read_parquet([{fold_sql}])
                    ORDER BY client_id
                ) TO '{m_tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            m_tmp.replace(m_path)
            for p in fold_paths:
                p.unlink(missing_ok=True)
            print(f"  [{m}] built in {time.perf_counter() - tm:.1f}s -> {m_path.name}", flush=True)

        # Merge all 10 months into one clean panel_moscow.parquet
        files_sql = ", ".join(f"'{p}'" for p in month_files)
        panel_tmp = OUT_DIR / ".tmp_panel_moscow.parquet"
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet([{files_sql}])
                ORDER BY month, client_id
            ) TO '{panel_tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        panel_tmp.replace(panel_path)
        print(f"  Merged panel -> {panel_path.name} ({panel_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    else:
        print(f"  Reusing cached {panel_path.name}", flush=True)

    # Release DuckDB buffers before Step 4
    con.close()
    gc.collect()

    # 4. DIALOG mean-pool embeddings (768d, all & 90d, strictly < cutoff_msk)
    print("[4/4] Computing Moscow-time DIALOG mean-pool 768d embeddings (all & 90d) ...", flush=True)
    emb_path = OUT_DIR / "dialog_emb_moscow.npz"
    emb_info = build_dialog_embeddings_moscow(emb_path)
    print(f"  Dialog embeddings ready: {emb_info}", flush=True)

    # Verify panel row count & schema
    con = duckdb.connect()
    _configure_con(con, tmp_dir)
    schema_cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{panel_path.as_posix()}')").fetchall()]
    row_cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{panel_path.as_posix()}')").fetchone()[0]
    con.close()

    audit_out = {
        "status": "ready",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - t_all, 1),
        "timezone": "Europe/Moscow (UTC+3)",
        "cutoff_rule": "event_time_msk = event_time_utc + 3h; strictly event_time_msk < YYYY-MM-01 00:00:00 MSK (= prev_month_end 21:00:00 UTC)",
        "windows": ["30d", "90d", "all"],
        "modeling_months": list(MODELING_MONTHS),
        "panel_rows": int(row_cnt),
        "panel_columns": schema_cols,
        "timezone_audit": tz_audit,
        "targets_summary": tgt_info,
        "dialog_embeddings": emb_info,
    }
    (OUT_DIR / "moscow_audit.json").write_text(json.dumps(audit_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDONE in {audit_out['elapsed_seconds']}s | rows={row_cnt:,} | cols={len(schema_cols)}", flush=True)


if __name__ == "__main__":
    main()
