"""Build Stage-2 EDA-driven GEO, DIALOG, TRX & Prior candidate features for H=3.

Produces outputs/h3_propensity/h3_stage2_fe_delta.zip containing:
  - h3_stage2_fe_delta.parquet:
      * For 2022-02, 2022-03: the 60 fe_v2 candidate features (matching panel_fe_v2_candidates.parquet)
      * For all ACTIVE_H3_MONTHS (02..08, 11): ~75 new EDA-driven GEO, DIALOG, TRX regime/tail,
        observation-age, and prior x behavioral interaction features.
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

from scripts.build_fe_v2_candidates import _entropy_sql
from scripts.build_moscow_features import _configure_con, _glob

OUT_DIR = REPO_ROOT / "outputs" / "h3_propensity"
DELTA_DIR = OUT_DIR / "delta"
TMP_DIR = OUT_DIR / "_duck_tmp_s2"

ACTIVE_H3_MONTHS = ("2022-02", "2022-03", "2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-11")
EARLY_MONTHS = ("2022-02", "2022-03")
FOLDS = (0, 1, 2, 3, 4)

FE_V2_PATH = REPO_ROOT / "outputs" / "fe_v2" / "panel_fe_v2_candidates.parquet"
MOSCOW_PANEL = REPO_ROOT / "outputs" / "baseline_moscow_v1" / "panel_moscow.parquet"
STAGE1_TGT = DELTA_DIR / "h3_targets_prior.parquet"
STAGE1_EARLY = DELTA_DIR / "h3_early_02_03_base112.parquet"


def build_early_fe_v2_trx_geo(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the raw-SQL part of fe_v2 TRX and GEO features for 2022-02 and 2022-03."""
    t0 = time.perf_counter()
    trx_frames, geo_frames = [], []
    for fold in FOLDS:
        trx_glob = _glob(f"detail/trx/fold={fold}/*.parquet")
        geo_glob = _glob(f"detail/geo/fold={fold}/*.parquet")
        for m in EARLY_MONTHS:
            c_msk = f"TIMESTAMP '{m}-01 00:00:00'"
            c_30 = f"{c_msk} - INTERVAL 30 DAY"
            c_90 = f"{c_msk} - INTERVAL 90 DAY"

            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE m_trx AS
                SELECT client_id, event_time + INTERVAL 3 HOUR AS event_time_msk,
                       amount, event_type, event_subtype, currency,
                       src_type11, src_type12, src_type21, src_type22, src_type31, src_type32,
                       dst_type11, dst_type12
                FROM read_parquet('{trx_glob}')
                WHERE event_time + INTERVAL 3 HOUR < {c_msk}
            """)
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE t_core AS
                SELECT
                    client_id, '{m}' AS month,
                    CAST(COUNT(*) FILTER (WHERE event_time_msk >= {c_30} AND currency IS NOT NULL AND currency <> 11) > 0 AS TINYINT) AS trx_30d_has_foreign,
                    CAST(COUNT(*) FILTER (WHERE event_time_msk >= {c_90} AND currency IS NOT NULL AND currency <> 11) > 0 AS TINYINT) AS trx_90d_has_foreign,
                    CAST(COUNT(*) FILTER (WHERE currency IS NOT NULL AND currency <> 11) > 0 AS TINYINT) AS trx_all_has_foreign,
                    CAST(COUNT(DISTINCT currency) FILTER (WHERE event_time_msk >= {c_30}) AS SMALLINT) AS trx_30d_currency_nunique,
                    CAST(COUNT(DISTINCT currency) AS SMALLINT) AS trx_all_currency_nunique,
                    CAST(COUNT(*) FILTER (WHERE currency IS NULL) AS INTEGER) AS trx_all_currency_missing_count,
                    CAST(LN(1.0 + COALESCE(SUM(ABS(amount)) FILTER (WHERE event_time_msk >= {c_90} AND currency IS NOT NULL AND currency <> 11 AND isfinite(amount)), 0.0)) AS FLOAT) AS trx_90d_foreign_log1p_amt,
                    CAST(LN(1.0 + COALESCE(SUM(ABS(amount)) FILTER (WHERE currency IS NOT NULL AND currency <> 11 AND isfinite(amount)), 0.0)) AS FLOAT) AS trx_all_foreign_log1p_amt,
                    CAST(COUNT(DISTINCT src_type11) FILTER (WHERE event_time_msk >= {c_30}) AS SMALLINT) AS trx_30d_src_type11_nunique,
                    CAST(COUNT(DISTINCT src_type11) FILTER (WHERE event_time_msk >= {c_90}) AS SMALLINT) AS trx_90d_src_type11_nunique,
                    CAST(COUNT(DISTINCT src_type11) AS SMALLINT) AS trx_all_src_type11_nunique,
                    CAST(COUNT(DISTINCT src_type12) FILTER (WHERE event_time_msk >= {c_90}) AS SMALLINT) AS trx_90d_src_type12_nunique,
                    CAST(COUNT(DISTINCT src_type12) AS SMALLINT) AS trx_all_src_type12_nunique,
                    CAST(COUNT(DISTINCT src_type21) FILTER (WHERE event_time_msk >= {c_30}) AS INTEGER) AS trx_30d_src_type21_nunique,
                    CAST(COUNT(DISTINCT src_type21) FILTER (WHERE event_time_msk >= {c_90}) AS INTEGER) AS trx_90d_src_type21_nunique,
                    CAST(COUNT(DISTINCT src_type21) AS INTEGER) AS trx_all_src_type21_nunique,
                    CAST(COUNT(DISTINCT src_type22) AS SMALLINT) AS trx_all_src_type22_nunique,
                    CAST(COUNT(DISTINCT src_type31) FILTER (WHERE event_time_msk >= {c_90}) AS INTEGER) AS trx_90d_src_type31_nunique,
                    CAST(COUNT(DISTINCT src_type31) AS INTEGER) AS trx_all_src_type31_nunique,
                    CAST(COUNT(DISTINCT src_type32) AS SMALLINT) AS trx_all_src_type32_nunique,
                    CAST(COUNT(*) FILTER (WHERE dst_type11 IS NULL)::DOUBLE / NULLIF(COUNT(*), 0) AS FLOAT) AS trx_all_dst11_missing_frac,
                    CAST(COUNT(*) FILTER (WHERE src_type11 IS NULL)::DOUBLE / NULLIF(COUNT(*), 0) AS FLOAT) AS trx_all_src11_missing_frac
                FROM m_trx GROUP BY client_id
            """)
            con.execute(f"CREATE OR REPLACE TEMP TABLE t_et AS {_entropy_sql('m_trx', 'event_type', c_msk, 'trx_et')}")
            con.execute(f"CREATE OR REPLACE TEMP TABLE t_es AS {_entropy_sql('m_trx', 'event_subtype', c_msk, 'trx_esub')}")
            con.execute(f"CREATE OR REPLACE TEMP TABLE t_d12 AS {_entropy_sql('m_trx', 'dst_type12', c_msk, 'trx_dst12')}")
            con.execute(f"CREATE OR REPLACE TEMP TABLE t_s12 AS {_entropy_sql('m_trx', 'src_type12', c_msk, 'trx_src12')}")
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE t_nov AS
                WITH es_seen AS (
                    SELECT client_id, event_subtype,
                           MAX(CASE WHEN event_time_msk >= {c_30} THEN 1 ELSE 0 END) AS in_30,
                           MAX(CASE WHEN event_time_msk < {c_30} THEN 1 ELSE 0 END) AS in_pre
                    FROM m_trx WHERE event_subtype IS NOT NULL GROUP BY 1, 2
                ),
                d12_seen AS (
                    SELECT client_id, dst_type12,
                           MAX(CASE WHEN event_time_msk >= {c_30} THEN 1 ELSE 0 END) AS in_30,
                           MAX(CASE WHEN event_time_msk < {c_30} THEN 1 ELSE 0 END) AS in_pre
                    FROM m_trx WHERE dst_type12 IS NOT NULL GROUP BY 1, 2
                )
                SELECT
                    COALESCE(e.client_id, d.client_id) AS client_id,
                    COALESCE(e.new_esub, 0)::SMALLINT AS trx_30d_new_event_subtype_count,
                    COALESCE(d.new_d12, 0)::SMALLINT AS trx_30d_new_dst12_count
                FROM (SELECT client_id, SUM(CASE WHEN in_30=1 AND in_pre=0 THEN 1 ELSE 0 END) AS new_esub FROM es_seen GROUP BY 1) e
                FULL OUTER JOIN (SELECT client_id, SUM(CASE WHEN in_30=1 AND in_pre=0 THEN 1 ELSE 0 END) AS new_d12 FROM d12_seen GROUP BY 1) d USING (client_id)
            """)
            trx_frames.append(con.execute("""
                SELECT c.*,
                    et.trx_et_90d_entropy AS trx_90d_event_type_entropy,
                    et.trx_et_all_entropy AS trx_all_event_type_entropy,
                    es.trx_esub_90d_entropy AS trx_90d_event_subtype_entropy,
                    es.trx_esub_all_entropy AS trx_all_event_subtype_entropy,
                    es.trx_esub_90d_top_share AS trx_90d_event_subtype_top_share,
                    es.trx_esub_all_top_share AS trx_all_event_subtype_top_share,
                    d12.trx_dst12_90d_top_share AS trx_90d_dst_type12_top_share,
                    d12.trx_dst12_all_top_share AS trx_all_dst_type12_top_share,
                    s12.trx_src12_all_top_share AS trx_all_src_type12_top_share,
                    COALESCE(nv.trx_30d_new_event_subtype_count, 0)::SMALLINT AS trx_30d_new_event_subtype_count,
                    COALESCE(nv.trx_30d_new_dst12_count, 0)::SMALLINT AS trx_30d_new_dst12_count
                FROM t_core c
                LEFT JOIN t_et et USING (client_id)
                LEFT JOIN t_es es USING (client_id)
                LEFT JOIN t_d12 d12 USING (client_id)
                LEFT JOIN t_s12 s12 USING (client_id)
                LEFT JOIN t_nov nv USING (client_id)
            """).df())

            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE m_geo AS
                SELECT client_id, event_time + INTERVAL 3 HOUR AS event_time_msk, geohash_4, geohash_5, geohash_6
                FROM read_parquet('{geo_glob}')
                WHERE event_time + INTERVAL 3 HOUR < {c_msk}
            """)
            con.execute(f"CREATE OR REPLACE TEMP TABLE g_gh6 AS {_entropy_sql('m_geo', 'geohash_6', c_msk, 'geo_gh6')}")
            con.execute(f"CREATE OR REPLACE TEMP TABLE g_gh4 AS {_entropy_sql('m_geo', 'geohash_4', c_msk, 'geo_gh4')}")
            con.execute(f"CREATE OR REPLACE TEMP TABLE g_gh5 AS {_entropy_sql('m_geo', 'geohash_5', c_msk, 'geo_gh5')}")
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE g_nov AS
                WITH gh5_s AS (
                    SELECT client_id, geohash_5,
                           MAX(CASE WHEN event_time_msk >= {c_30} THEN 1 ELSE 0 END) AS in_30,
                           MAX(CASE WHEN event_time_msk < {c_30} THEN 1 ELSE 0 END) AS in_pre
                    FROM m_geo WHERE geohash_5 IS NOT NULL GROUP BY 1, 2
                ),
                gh6_s AS (
                    SELECT client_id, geohash_6,
                           MAX(CASE WHEN event_time_msk >= {c_30} THEN 1 ELSE 0 END) AS in_30,
                           MAX(CASE WHEN event_time_msk < {c_30} THEN 1 ELSE 0 END) AS in_pre
                    FROM m_geo WHERE geohash_6 IS NOT NULL GROUP BY 1, 2
                )
                SELECT
                    COALESCE(a.client_id, b.client_id) AS client_id,
                    COALESCE(a.n5, 0)::SMALLINT AS geo_30d_new_gh5_count,
                    COALESCE(b.n6, 0)::SMALLINT AS geo_30d_new_gh6_count
                FROM (SELECT client_id, SUM(CASE WHEN in_30=1 AND in_pre=0 THEN 1 ELSE 0 END) AS n5 FROM gh5_s GROUP BY 1) a
                FULL OUTER JOIN (SELECT client_id, SUM(CASE WHEN in_30=1 AND in_pre=0 THEN 1 ELSE 0 END) AS n6 FROM gh6_s GROUP BY 1) b USING (client_id)
            """)
            geo_frames.append(con.execute(f"""
                WITH active AS (SELECT DISTINCT client_id FROM m_geo)
                SELECT a.client_id, '{m}' AS month,
                    g6.geo_gh6_90d_entropy AS geo_90d_gh6_entropy,
                    g6.geo_gh6_all_entropy AS geo_all_gh6_entropy,
                    g4.geo_gh4_all_top_share AS geo_all_top_gh4_share,
                    g5.geo_gh5_all_top_share AS geo_all_top_gh5_share,
                    g5.geo_gh5_90d_entropy AS geo_90d_gh5_entropy,
                    COALESCE(nv.geo_30d_new_gh5_count, 0)::SMALLINT AS geo_30d_new_gh5_count,
                    COALESCE(nv.geo_30d_new_gh6_count, 0)::SMALLINT AS geo_30d_new_gh6_count
                FROM active a
                LEFT JOIN g_gh6 g6 USING (client_id)
                LEFT JOIN g_gh4 g4 USING (client_id)
                LEFT JOIN g_gh5 g5 USING (client_id)
                LEFT JOIN g_nov nv USING (client_id)
            """).df())
    print(f"  [S2 1/4] Early 02-03 fe_v2 SQL features done in {time.perf_counter()-t0:.1f}s", flush=True)
    return pd.concat(trx_frames, ignore_index=True), pd.concat(geo_frames, ignore_index=True)


def build_eda_dialog_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Build deep EDA-driven DIALOG features (7d/14d/prev_cal_month, active months, gaps, emb dispersion)."""
    t0 = time.perf_counter()
    dlg_glob = _glob("detail/dialog/*/*.parquet")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE dlg_raw AS
        SELECT client_id,
               event_time + INTERVAL 3 HOUR AS ts_msk,
               strftime(event_time + INTERVAL 3 HOUR, '%Y-%m') AS ev_ym,
               embedding
        FROM read_parquet('{dlg_glob}')
        WHERE event_time + INTERVAL 3 HOUR < TIMESTAMP '2022-11-01 00:00:00'
    """)

    meta_frames = []
    disp_frames = []
    for m in ACTIVE_H3_MONTHS:
        c = f"TIMESTAMP '{m}-01 00:00:00'"
        prev_ym = pd.Period(m, freq="M") - 1
        prev_ym_str = str(prev_ym)
        df_m = con.execute(f"""
            WITH sub AS (
                SELECT client_id, ts_msk, ev_ym,
                       LAG(ts_msk) OVER (PARTITION BY client_id ORDER BY ts_msk) AS prev_ts
                FROM dlg_raw
                WHERE ts_msk < {c}
            ),
            gaps AS (
                SELECT client_id,
                       ts_msk,
                       ev_ym,
                       CASE WHEN prev_ts IS NOT NULL AND ts_msk >= {c} - INTERVAL 90 DAY
                            THEN EPOCH(ts_msk - prev_ts) / 86400.0 ELSE NULL END AS gap_90d
                FROM sub
            )
            SELECT
                client_id, '{m}' AS month,
                CAST(COUNT(*) FILTER (WHERE ts_msk >= {c} - INTERVAL 7 DAY) AS INTEGER) AS dialog_7d_event_count,
                CAST(COUNT(*) FILTER (WHERE ts_msk >= {c} - INTERVAL 14 DAY) AS INTEGER) AS dialog_14d_event_count,
                CAST(COUNT(*) FILTER (WHERE ev_ym = '{prev_ym_str}') > 0 AS TINYINT) AS dialog_prev_cal_month_has_event,
                CAST(LN(1.0 + COUNT(*) FILTER (WHERE ev_ym = '{prev_ym_str}')) AS FLOAT) AS dialog_prev_cal_month_log1p_count,
                CAST(COUNT(DISTINCT ev_ym) AS SMALLINT) AS dialog_active_months_all,
                CAST(AVG(gap_90d) AS FLOAT) AS dialog_inter_gap_mean_90d,
                CAST(STDDEV_SAMP(gap_90d) AS FLOAT) AS dialog_inter_gap_std_90d,
                CAST(MAX(gap_90d) AS FLOAT) AS dialog_inter_gap_max_90d
            FROM gaps
            GROUP BY client_id
        """).df()
        meta_frames.append(df_m)

    # Embedding dispersion within 90d (mean per-dim std for clients with >= 2 embeddings in 90d)
    df_e = con.execute("SELECT client_id, ts_msk, embedding FROM dlg_raw WHERE embedding IS NOT NULL").fetch_df()
    if len(df_e) > 0:
        emb_mat = np.vstack(df_e["embedding"].to_numpy()).astype(np.float32, copy=False)
        cids = df_e["client_id"].to_numpy(dtype=str)
        ts = df_e["ts_msk"].to_numpy(dtype="datetime64[ns]")
        del df_e
        gc.collect()
        for m in ACTIVE_H3_MONTHS:
            cutoff = np.datetime64(f"{m}-01T00:00:00", "ns")
            lo90 = cutoff - np.timedelta64(90, "D")
            mk = (ts >= lo90) & (ts < cutoff)
            if not np.any(mk):
                continue
            sc = cids[mk]
            se = emb_mat[mk]
            codes, uniq = pd.factorize(sc, sort=False)
            cnt = np.bincount(codes, minlength=len(uniq))
            # Filter to clients with >= 2 events for variance
            multi = np.where(cnt >= 2)[0]
            if len(multi) == 0:
                continue
            sums = np.zeros((len(uniq), se.shape[1]), dtype=np.float64)
            sums2 = np.zeros((len(uniq), se.shape[1]), dtype=np.float64)
            np.add.at(sums, codes, se)
            np.add.at(sums2, codes, se.astype(np.float64) ** 2)
            n_m = cnt[multi][:, None].astype(np.float64)
            mean_m = sums[multi] / n_m
            var_m = np.maximum(0.0, (sums2[multi] - n_m * (mean_m ** 2)) / np.maximum(n_m - 1.0, 1.0))
            mean_std = np.sqrt(var_m).mean(axis=1).astype(np.float32)
            disp_frames.append(pd.DataFrame({
                "client_id": uniq[multi],
                "month": m,
                "dlg_emb_within_std_90d": mean_std,
            }))
        del emb_mat, cids, ts
        gc.collect()

    out = pd.concat(meta_frames, ignore_index=True)
    if disp_frames:
        disp_df = pd.concat(disp_frames, ignore_index=True)
        out = out.merge(disp_df, on=["client_id", "month"], how="left")
    else:
        out["dlg_emb_within_std_90d"] = np.float32(np.nan)
    print(f"  [S2 2/4] EDA DIALOG features done in {time.perf_counter()-t0:.1f}s ({len(out):,} rows)", flush=True)
    return out


def build_eda_geo_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Build deep EDA-driven GEO spatial/temporal features across ACTIVE_H3_MONTHS."""
    t0 = time.perf_counter()
    frames = []
    for fold in FOLDS:
        tf = time.perf_counter()
        geo_glob = _glob(f"detail/geo/fold={fold}/*.parquet")
        fcon = duckdb.connect()
        _configure_con(fcon, TMP_DIR)
        fcon.execute(f"""
            CREATE OR REPLACE TEMP TABLE f_geo AS
            SELECT client_id,
                   event_time + INTERVAL 3 HOUR AS ts_msk,
                   strftime(event_time + INTERVAL 3 HOUR, '%Y-%m') AS ev_ym,
                   geohash_4, geohash_6
            FROM read_parquet('{geo_glob}')
            WHERE event_time + INTERVAL 3 HOUR < TIMESTAMP '2022-11-01 00:00:00'
        """)
        for m in ACTIVE_H3_MONTHS:
            c = f"TIMESTAMP '{m}-01 00:00:00'"
            c_7 = f"{c} - INTERVAL 7 DAY"
            c_14 = f"{c} - INTERVAL 14 DAY"
            c_30 = f"{c} - INTERVAL 30 DAY"
            c_90 = f"{c} - INTERVAL 90 DAY"
            prev_ym = str(pd.Period(m, freq="M") - 1)

            fcon.execute(f"""
                CREATE OR REPLACE TEMP TABLE mg AS
                SELECT client_id, ts_msk, ev_ym, geohash_4, geohash_6,
                       LAG(ts_msk) OVER (PARTITION BY client_id ORDER BY ts_msk) AS prev_ts
                FROM f_geo WHERE ts_msk < {c}
            """)
            fcon.execute(f"""
                CREATE OR REPLACE TEMP TABLE g_core AS
                SELECT
                    client_id, '{m}' AS month,
                    CAST(COUNT(*) FILTER (WHERE ts_msk >= {c_7}) AS INTEGER) AS geo_7d_event_count,
                    CAST(COUNT(*) FILTER (WHERE ts_msk >= {c_14}) AS INTEGER) AS geo_14d_event_count,
                    CAST(COUNT(*) FILTER (WHERE ev_ym = '{prev_ym}') > 0 AS TINYINT) AS geo_prev_cal_month_has_event,
                    CAST(COUNT(DISTINCT ev_ym) AS SMALLINT) AS geo_active_months_all,
                    CAST(AVG(CASE WHEN prev_ts IS NOT NULL AND ts_msk >= {c_90} THEN EPOCH(ts_msk - prev_ts) / 86400.0 ELSE NULL END) AS FLOAT) AS geo_inter_gap_mean_90d
                FROM mg GROUP BY client_id
            """)
            fcon.execute(f"CREATE OR REPLACE TEMP TABLE g4 AS {_entropy_sql('mg', 'geohash_4', c, 'g4_e').replace('event_time_msk', 'ts_msk')}")
            fcon.execute(f"CREATE OR REPLACE TEMP TABLE g6 AS {_entropy_sql('mg', 'geohash_6', c, 'g6_e').replace('event_time_msk', 'ts_msk')}")
            fcon.execute(f"""
                CREATE OR REPLACE TEMP TABLE g4_30 AS
                WITH c4 AS (
                    SELECT client_id, geohash_4, COUNT(*) AS n30,
                           MAX(CASE WHEN ts_msk < {c_30} THEN 1 ELSE 0 END) AS in_pre,
                           MAX(CASE WHEN ts_msk >= {c_30} THEN 1 ELSE 0 END) AS in_30
                    FROM mg WHERE geohash_4 IS NOT NULL GROUP BY 1, 2
                ),
                c30 AS (
                    SELECT client_id, geohash_4, COUNT(*) AS n30_only
                    FROM mg WHERE geohash_4 IS NOT NULL AND ts_msk >= {c_30} GROUP BY 1, 2
                ),
                tot30 AS (SELECT client_id, SUM(n30_only) AS s30 FROM c30 GROUP BY 1)
                SELECT
                    COALESCE(a.client_id, b.client_id) AS client_id,
                    COALESCE(a.new_gh4, 0)::SMALLINT AS geo_30d_new_gh4_count,
                    b.geo_30d_gh4_entropy,
                    b.geo_30d_top_gh4_share
                FROM (
                    SELECT client_id, SUM(CASE WHEN in_30=1 AND in_pre=0 THEN 1 ELSE 0 END) AS new_gh4
                    FROM c4 GROUP BY 1
                ) a
                FULL OUTER JOIN (
                    SELECT c.client_id,
                           CAST(-SUM((c.n30_only::DOUBLE / t.s30) * LN(c.n30_only::DOUBLE / t.s30)) AS FLOAT) AS geo_30d_gh4_entropy,
                           (MAX(c.n30_only)::DOUBLE / NULLIF(MAX(t.s30), 0))::FLOAT AS geo_30d_top_gh4_share
                    FROM c30 c JOIN tot30 t USING (client_id) GROUP BY 1
                ) b USING (client_id)
            """)
            fcon.execute(f"""
                CREATE OR REPLACE TEMP TABLE g6_30 AS
                WITH c30 AS (
                    SELECT client_id, geohash_6, COUNT(*) AS n30_only
                    FROM mg WHERE geohash_6 IS NOT NULL AND ts_msk >= {c_30} GROUP BY 1, 2
                ),
                tot30 AS (SELECT client_id, SUM(n30_only) AS s30 FROM c30 GROUP BY 1)
                SELECT c.client_id,
                       CAST(-SUM((c.n30_only::DOUBLE / t.s30) * LN(c.n30_only::DOUBLE / t.s30)) AS FLOAT) AS geo_30d_gh6_entropy
                FROM c30 c JOIN tot30 t USING (client_id) GROUP BY 1
            """)
            frames.append(fcon.execute("""
                SELECT gc.*,
                       g4.g4_e_90d_entropy AS geo_90d_gh4_entropy,
                       g4.g4_e_90d_top_share AS geo_90d_top_gh4_share,
                       g6.g6_e_90d_top_share AS geo_90d_top_gh6_share,
                       COALESCE(g43.geo_30d_new_gh4_count, 0)::SMALLINT AS geo_30d_new_gh4_count,
                       g43.geo_30d_gh4_entropy,
                       g43.geo_30d_top_gh4_share,
                       g63.geo_30d_gh6_entropy
                FROM g_core gc
                LEFT JOIN g4 USING (client_id)
                LEFT JOIN g6 USING (client_id)
                LEFT JOIN g4_30 g43 USING (client_id)
                LEFT JOIN g6_30 g63 USING (client_id)
            """).df())
        fcon.close()
        print(f"    EDA GEO fold={fold} done ({time.perf_counter()-tf:.1f}s)", flush=True)
    out = pd.concat(frames, ignore_index=True)
    print(f"  [S2 3/4] EDA GEO features done in {time.perf_counter()-t0:.1f}s ({len(out):,} rows)", flush=True)
    return out


def main() -> None:
    t_all = time.perf_counter()
    DELTA_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    out_pq = DELTA_DIR / "h3_stage2_fe_delta.parquet"
    zip_path = OUT_DIR / "h3_stage2_fe_delta.zip"

    if not out_pq.exists():
        con = duckdb.connect()
        _configure_con(con, TMP_DIR)
        early_trx_new, early_geo_new = build_early_fe_v2_trx_geo(con)
        eda_dlg = build_eda_dialog_features(con)
        eda_geo = build_eda_geo_features(con)
        con.close()

        print("  [S2 4/4] Combining Stage-1 base + fe_v2 + EDA derived features ...", flush=True)
        tgt = pd.read_parquet(STAGE1_TGT)
        tgt["month"] = tgt["month"].astype(str)

        early_b = pd.read_parquet(STAGE1_EARLY)
        early_b["month"] = early_b["month"].astype(str)
        beh_cols = [c for c in early_b.columns if c not in ("client_id", "fold", "month")]

        moscow_b = pd.read_parquet(
            MOSCOW_PANEL,
            columns=["client_id", "month"] + beh_cols,
            filters=[("month", "in", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-11"])],
        )
        moscow_b["month"] = moscow_b["month"].astype(str)
        beh_all = pd.concat([early_b.drop(columns=["fold"], errors="ignore"), moscow_b], ignore_index=True)

        # Load fe_v2 extra columns for 04..08, 11 from panel_fe_v2_candidates.parquet
        fe2_schema = pq.ParquetFile(FE_V2_PATH).schema_arrow.names
        pm_schema = pq.ParquetFile(MOSCOW_PANEL).schema_arrow.names
        fe2_extra_cols = [c for c in fe2_schema if c not in pm_schema]
        fe2_later = pd.read_parquet(
            FE_V2_PATH,
            columns=["client_id", "month"] + fe2_extra_cols,
            filters=[("month", "in", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-11"])],
        )
        fe2_later["month"] = fe2_later["month"].astype(str)

        # Merge early base + early_trx_new + early_geo_new -> compute same derived fe_v2 cols for 02, 03
        early_df = early_b[["client_id", "month"] + beh_cols].merge(early_trx_new, on=["client_id", "month"], how="left")
        early_df = early_df.merge(early_geo_new, on=["client_id", "month"], how="left")
        zero_fill_cols = [
            "trx_30d_has_foreign", "trx_90d_has_foreign", "trx_all_has_foreign",
            "trx_30d_currency_nunique", "trx_all_currency_nunique", "trx_all_currency_missing_count",
            "trx_90d_foreign_log1p_amt", "trx_all_foreign_log1p_amt",
            "trx_30d_src_type11_nunique", "trx_90d_src_type11_nunique", "trx_all_src_type11_nunique",
            "trx_90d_src_type12_nunique", "trx_all_src_type12_nunique",
            "trx_30d_src_type21_nunique", "trx_90d_src_type21_nunique", "trx_all_src_type21_nunique",
            "trx_all_src_type22_nunique", "trx_90d_src_type31_nunique", "trx_all_src_type31_nunique",
            "trx_all_src_type32_nunique",
            "trx_30d_new_event_subtype_count", "trx_30d_new_dst12_count",
            "geo_30d_new_gh5_count", "geo_30d_new_gh6_count",
        ]
        for c in zero_fill_cols:
            early_df[c] = early_df[c].fillna(0).astype(np.float32)

        early_df["trx_main_neg_ratio_30d"] = (early_df["trx_30d_main_neg_count"] / np.maximum(early_df["trx_30d_main_count"], 1)).astype(np.float32)
        early_df["trx_main_neg_ratio_all"] = (early_df["trx_all_main_neg_count"] / np.maximum(early_df["trx_all_main_count"], 1)).astype(np.float32)
        early_df["trx_share_90d_of_all"] = np.where(early_df["trx_all_event_count"] > 0, early_df["trx_90d_event_count"] / early_df["trx_all_event_count"], np.nan).astype(np.float32)
        early_df["trx_amt_share_30d_of_90d"] = np.where(early_df["trx_90d_main_amount_abs_sum"] > 0, early_df["trx_30d_main_amount_abs_sum"] / early_df["trx_90d_main_amount_abs_sum"], np.nan).astype(np.float32)
        early_df["trx_30d_rate_vs_tenure"] = np.where(early_df["trx_rate_per_30d_tenure"] > 0, early_df["trx_30d_event_count"] / early_df["trx_rate_per_30d_tenure"], np.nan).astype(np.float32)
        early_df["trx_active_day_density_30d"] = (early_df["trx_30d_active_days"] / 30.0).astype(np.float32)
        early_df["trx_active_day_density_90d"] = (early_df["trx_90d_active_days"] / 90.0).astype(np.float32)
        early_df["trx_tx_per_active_day_30d"] = np.where(early_df["trx_30d_active_days"] > 0, early_df["trx_30d_event_count"] / early_df["trx_30d_active_days"], np.nan).astype(np.float32)
        early_df["trx_tx_per_active_day_all"] = np.where(early_df["trx_all_active_days"] > 0, early_df["trx_all_event_count"] / early_df["trx_all_active_days"], np.nan).astype(np.float32)
        early_df["trx_dormant_days_gap"] = (early_df["trx_all_days_since_first_event"] - early_df["trx_all_days_since_last_event"]).astype(np.float32)
        early_df["geo_all_gh6_per_gh4"] = np.where(early_df["geo_all_geohash_4_nunique"] > 0, early_df["geo_all_geohash_6_nunique"] / early_df["geo_all_geohash_4_nunique"], np.nan).astype(np.float32)
        early_df["geo_30d_gh6_per_gh4"] = np.where(early_df["geo_30d_geohash_4_nunique"] > 0, early_df["geo_30d_geohash_6_nunique"] / early_df["geo_30d_geohash_4_nunique"], np.nan).astype(np.float32)
        early_df["geo_share_30d_of_90d"] = np.where(early_df["geo_90d_event_count"] > 0, early_df["geo_30d_event_count"] / early_df["geo_90d_event_count"], np.nan).astype(np.float32)
        early_df["dialog_share_30d_of_90d"] = np.where(early_df["dialog_90d_event_count"] > 0, early_df["dialog_30d_event_count"] / early_df["dialog_90d_event_count"], np.nan).astype(np.float32)
        early_df["dialog_first_activated_30d"] = ((early_df["dialog_30d_event_count"] > 0) & (early_df["dialog_30d_event_count"] == early_df["dialog_all_event_count"])).astype(np.int8)
        early_df["is_silent_trx_90d_active_dlg"] = ((early_df["trx_90d_has_event"] == 0) & (early_df["dialog_90d_has_event"] == 1)).astype(np.int8)
        early_df["cross_source_active_count_90d"] = (early_df["trx_90d_has_event"] + early_df["geo_90d_has_event"] + early_df["dialog_90d_has_event"]).astype(np.int8)
        early_df["cross_source_active_count_all"] = (early_df["trx_all_has_event"] + early_df["geo_all_has_event"] + early_df["dialog_all_has_event"]).astype(np.int8)
        early_df["trx_to_geo_log_ratio_all"] = (early_df["trx_all_log1p_event_count"] - early_df["geo_all_log1p_event_count"]).astype(np.float32)
        rec_s = np.vstack([
            early_df["trx_all_days_since_last_event"].to_numpy(dtype=np.float32),
            early_df["geo_all_days_since_last_event"].to_numpy(dtype=np.float32),
            early_df["dialog_all_days_since_last_event"].to_numpy(dtype=np.float32),
        ])
        with np.errstate(all="ignore"):
            early_df["all_min_days_since_last_event"] = np.nanmin(rec_s, axis=0).astype(np.float32)

        fe2_all = pd.concat([early_df[["client_id", "month"] + fe2_extra_cols], fe2_later], ignore_index=True)
        del early_df, fe2_later, early_trx_new, early_geo_new, early_b, moscow_b
        gc.collect()

        # Merge full behavioral + fe2 + eda_dlg + eda_geo onto tgt
        df = tgt.merge(beh_all, on=["client_id", "month"], how="inner")
        df = df.merge(fe2_all, on=["client_id", "month"], how="left")
        df = df.merge(eda_dlg, on=["client_id", "month"], how="left")
        df = df.merge(eda_geo, on=["client_id", "month"], how="left")
        del beh_all, fe2_all, eda_dlg, eda_geo
        gc.collect()

        # Zero-fill integer counts/flags from eda_dlg & eda_geo
        for c in (
            "dialog_7d_event_count", "dialog_14d_event_count", "dialog_prev_cal_month_has_event",
            "dialog_prev_cal_month_log1p_count", "dialog_active_months_all",
            "geo_7d_event_count", "geo_14d_event_count", "geo_prev_cal_month_has_event",
            "geo_active_months_all", "geo_30d_new_gh4_count",
        ):
            df[c] = df[c].fillna(0).astype(np.float32)

        # New EDA derived DIALOG, GEO, Observation-Age, TRX non-monotonic, Prior-rate features
        new_cols: dict[str, np.ndarray] = {}

        # 1) EDA DIALOG derived + interactions
        d_rec_last = df["dialog_all_days_since_last_event"].to_numpy(dtype=np.float32)
        new_cols["dialog_recency_exp_14d"] = np.where(np.isfinite(d_rec_last), np.exp(-d_rec_last / 14.0), 0.0).astype(np.float32)
        new_cols["dialog_recency_exp_45d"] = np.where(np.isfinite(d_rec_last), np.exp(-d_rec_last / 45.0), 0.0).astype(np.float32)
        d_30 = df["dialog_30d_event_count"].to_numpy(dtype=np.float32)
        d_90 = df["dialog_90d_event_count"].to_numpy(dtype=np.float32)
        d_7 = df["dialog_7d_event_count"].to_numpy(dtype=np.float32)
        new_cols["dialog_burst_7d_vs_30d"] = np.where(d_30 > 0, d_7 / d_30, np.nan).astype(np.float32)
        new_cols["dialog_burst_30d_vs_90d"] = np.where(d_90 > 0, d_30 / d_90, np.nan).astype(np.float32)
        new_cols["dialog_active_day_density_90d"] = (df["dialog_90d_active_days"].to_numpy(dtype=np.float32) / 90.0).astype(np.float32)

        z_trx_all = (df["trx_all_has_event"].to_numpy(dtype=np.float32) == 0).astype(np.float32)
        z_trx_90 = (df["trx_90d_has_event"].to_numpy(dtype=np.float32) == 0).astype(np.float32)
        d_30_has = df["dialog_30d_has_event"].to_numpy(dtype=np.float32)
        d_prev_has = df["dialog_prev_cal_month_has_event"].to_numpy(dtype=np.float32)
        new_cols["zero_trx_all_x_dialog_30d"] = (z_trx_all * d_30_has).astype(np.float32)
        new_cols["zero_trx_90d_x_dialog_30d"] = (z_trx_90 * d_30_has).astype(np.float32)
        new_cols["zero_trx_all_x_dialog_prev_mo"] = (z_trx_all * d_prev_has).astype(np.float32)
        new_cols["zero_trx_90d_x_dialog_log1p"] = (z_trx_90 * df["dialog_30d_log1p_event_count"].to_numpy(dtype=np.float32)).astype(np.float32)

        p_any = df["prior_all_bought_any"].to_numpy(dtype=np.float32)
        p_3m = df["prior_3m_buy_months"].to_numpy(dtype=np.float32)
        p_rec = df["prior_months_since_last_buy"].to_numpy(dtype=np.float32)
        new_cols["no_prior_buy_x_dialog_30d"] = np.where(np.isfinite(p_any), (1.0 - p_any) * d_30_has, np.nan).astype(np.float32)
        new_cols["prior_buy_x_dialog_30d"] = np.where(np.isfinite(p_any), p_any * d_30_has, np.nan).astype(np.float32)
        new_cols["dialog_30d_x_prior_3m_buys"] = np.where(np.isfinite(p_3m), d_30_has * p_3m, np.nan).astype(np.float32)
        new_cols["dialog_recency_x_prior_recency"] = (new_cols["dialog_recency_exp_45d"] * np.where(np.isfinite(p_rec), np.exp(-p_rec / 2.0), 0.0)).astype(np.float32)

        # 2) EDA GEO + Cross-source derived
        g_rec_last = df["geo_all_days_since_last_event"].to_numpy(dtype=np.float32)
        new_cols["geo_recency_exp_14d"] = np.where(np.isfinite(g_rec_last), np.exp(-g_rec_last / 14.0), 0.0).astype(np.float32)
        g4_90 = df["geo_90d_geohash_4_nunique"].to_numpy(dtype=np.float32)
        g6_90 = df["geo_90d_geohash_6_nunique"].to_numpy(dtype=np.float32)
        g6_30 = df["geo_30d_geohash_6_nunique"].to_numpy(dtype=np.float32)
        new_cols["geo_90d_gh6_per_gh4"] = np.where(g4_90 > 0, g6_90 / g4_90, np.nan).astype(np.float32)
        new_cols["geo_30d_new_gh6_frac"] = np.where(g6_30 > 0, df["geo_30d_new_gh6_count"].to_numpy(dtype=np.float32) / g6_30, np.nan).astype(np.float32)
        new_cols["geo_events_per_active_day_30d"] = np.where(df["geo_30d_active_days"] > 0, df["geo_30d_event_count"] / df["geo_30d_active_days"], np.nan).astype(np.float32)
        new_cols["geo_events_per_active_day_90d"] = np.where(df["geo_90d_active_days"] > 0, df["geo_90d_event_count"] / df["geo_90d_active_days"], np.nan).astype(np.float32)
        g_30_has = df["geo_30d_has_event"].to_numpy(dtype=np.float32)
        new_cols["zero_trx_x_geo_30d"] = (z_trx_all * g_30_has).astype(np.float32)
        new_cols["geo_30d_x_dialog_30d"] = (g_30_has * d_30_has).astype(np.float32)
        t_30_has = df["trx_30d_has_event"].to_numpy(dtype=np.float32)
        new_cols["active_sources_30d_count"] = (t_30_has + g_30_has + d_30_has).astype(np.float32)
        new_cols["source_bitmask_30d"] = (t_30_has + 2.0 * g_30_has + 4.0 * d_30_has).astype(np.float32)
        new_cols["source_bitmask_90d"] = (
            df["trx_90d_has_event"].to_numpy(dtype=np.float32)
            + 2.0 * df["geo_90d_has_event"].to_numpy(dtype=np.float32)
            + 4.0 * df["dialog_90d_has_event"].to_numpy(dtype=np.float32)
        ).astype(np.float32)

        # 3) Observation age, TRX non-monotonic U-curve, tail, and Prior rate features
        cutoff_dt = pd.to_datetime(df["month"] + "-01")
        new_cols["obs_age_days_trx"] = (cutoff_dt - pd.Timestamp("2021-01-01")).dt.days.to_numpy(dtype=np.float32)
        new_cols["obs_age_days_geo"] = (cutoff_dt - pd.Timestamp("2021-12-11")).dt.days.to_numpy(dtype=np.float32)
        new_cols["obs_age_days_dlg"] = (cutoff_dt - pd.Timestamp("2021-02-01")).dt.days.to_numpy(dtype=np.float32)
        t_first = df["trx_all_days_since_first_event"].to_numpy(dtype=np.float32)
        new_cols["trx_tenure_frac_of_obs"] = np.where(np.isfinite(t_first), t_first / np.maximum(new_cols["obs_age_days_trx"], 1.0), np.nan).astype(np.float32)
        new_cols["trx_all_days_since_first_event_cap365"] = np.minimum(t_first, np.float32(365.0)).astype(np.float32)

        t_all_cnt = df["trx_all_event_count"].to_numpy(dtype=np.float32)
        new_cols["trx_regime_0"] = (t_all_cnt == 0).astype(np.float32)
        new_cols["trx_regime_1_9"] = ((t_all_cnt >= 1) & (t_all_cnt <= 9)).astype(np.float32)
        new_cols["trx_regime_10_99"] = ((t_all_cnt >= 10) & (t_all_cnt <= 99)).astype(np.float32)
        new_cols["trx_regime_100p"] = (t_all_cnt >= 100).astype(np.float32)

        for w in ("90d", "all"):
            mx = df[f"trx_{w}_main_amount_max"].to_numpy(dtype=np.float32)
            md = df[f"trx_{w}_main_amount_median"].to_numpy(dtype=np.float32)
            lmx = np.where(np.isfinite(mx), np.sign(mx) * np.log1p(np.abs(mx)), np.nan)
            lmd = np.where(np.isfinite(md), np.sign(md) * np.log1p(np.abs(md)), np.nan)
            new_cols[f"trx_{w}_log_max_minus_med"] = (lmx - lmd).astype(np.float32)

        p_obs = df["prior_months_observed"].to_numpy(dtype=np.float32)
        p_all_m = df["prior_all_buy_months"].to_numpy(dtype=np.float32)
        new_cols["prior_buy_rate_per_obs_mo"] = np.where(p_obs > 0, p_all_m / p_obs, np.nan).astype(np.float32)
        new_cols["prior_recency_exp"] = np.where(np.isfinite(p_rec), np.exp(-p_rec / 2.0), np.nan).astype(np.float32)
        new_cols["prior_3m_share_of_all_buys"] = np.where(p_all_m > 0, p_3m / p_all_m, np.nan).astype(np.float32)
        for pid in ("p1", "p2", "p3", "p4"):
            pv = df[f"prior_all_bought_{pid}"].to_numpy(dtype=np.float32)
            new_cols[f"prior_{pid}_rate"] = np.where(p_obs > 0, pv / p_obs, np.nan).astype(np.float32)
        new_cols["prior_multi_product_flag"] = np.where(
            np.isfinite(df["prior_all_distinct_products"].to_numpy(dtype=np.float32)),
            (df["prior_all_distinct_products"].to_numpy(dtype=np.float32) >= 2).astype(np.float32),
            np.nan,
        ).astype(np.float32)

        # Keep ONLY key columns + fe_v2 extra columns + newly added EDA columns (avoid duplicating Stage 1 cols)
        stage1_keep = {"client_id", "month"}
        eda_extra_cols = [
            "dialog_7d_event_count", "dialog_14d_event_count", "dialog_prev_cal_month_has_event",
            "dialog_prev_cal_month_log1p_count", "dialog_active_months_all",
            "dialog_inter_gap_mean_90d", "dialog_inter_gap_std_90d", "dialog_inter_gap_max_90d",
            "dlg_emb_within_std_90d",
            "geo_7d_event_count", "geo_14d_event_count", "geo_prev_cal_month_has_event",
            "geo_active_months_all", "geo_inter_gap_mean_90d",
            "geo_90d_gh4_entropy", "geo_90d_top_gh4_share", "geo_90d_top_gh6_share",
            "geo_30d_new_gh4_count", "geo_30d_gh4_entropy", "geo_30d_top_gh4_share", "geo_30d_gh6_entropy",
        ]
        delta_df = pd.concat(
            [
                df[["client_id", "month"] + fe2_extra_cols + eda_extra_cols].reset_index(drop=True),
                pd.DataFrame(new_cols),
            ],
            axis=1,
        )
        for c in delta_df.columns:
            if c not in stage1_keep and delta_df[c].dtype == np.float64:
                delta_df[c] = delta_df[c].astype(np.float32)

        delta_df = delta_df.sort_values(["month", "client_id"]).reset_index(drop=True)
        tmp = out_pq.with_suffix(".tmp.parquet")
        delta_df.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(out_pq)
        print(f"  [S2] Saved {out_pq.name}: shape={delta_df.shape} ({delta_df.shape[1]-2} new/fe2 features)", flush=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_pq, arcname=out_pq.name)
    mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[DONE Stage-2 FE Delta] Packed {zip_path} ({mb:.2f} MB) in {time.perf_counter()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
