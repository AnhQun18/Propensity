"""Build enriched FE-v2 candidate features (MSK-aligned < YYYY-MM-01 00:00:00 UTC+3)
and merge with existing 112 behavioral features in panel_moscow.parquet.

New engineered groups added on top of panel_moscow (112 cols -> ~172 tabular candidates,
later pruned STRICTLY on Validation to <= 120 max final FE-v2 features):
1. Foreign currency enrichment (currency != 11 flags, distinct currencies, missing count, foreign log1p sum)
2. event_type / event_subtype diversity, concentration (top_share), Shannon entropy, and 30d novelty
3. src_type11/12/21/22/31/32 & dst_type12 breadth, concentration (top_share), 30d novelty, missingness fractions
4. TRX ratios, velocity burst vs tenure, active-day density, neg-amount share, amount momentum
5. GEO geohash_4/5/6 breadth ratios (gh6/gh4), top_gh4/gh5 shares, Shannon entropy, 30d spatial novelty
6. Dialog & Cross-source derived signals (silent-TRX + active-dialog, cross-source count, min recency)
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "MBD-mini"
PANEL_PATH = REPO_ROOT / "outputs" / "baseline_moscow_v1" / "panel_moscow.parquet"
OUT_DIR = REPO_ROOT / "outputs" / "fe_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CANDIDATES = OUT_DIR / "panel_fe_v2_candidates.parquet"

MODELING_MONTHS = (
    "2022-04", "2022-05", "2022-06", "2022-07", "2022-08",
    "2022-09", "2022-10", "2022-11", "2022-12", "2023-01",
)


def _entropy_sql(tbl: str, col: str, c_msk: str, prefix: str) -> str:
    """Return SQL CTE query yielding (client_id, 90d entropy, all entropy, 90d top_share, all top_share)."""
    return f"""
    WITH counts AS (
        SELECT
            client_id,
            {col} AS val,
            COUNT(*) FILTER (WHERE event_time_msk >= {c_msk} - INTERVAL 90 DAY) AS n90,
            COUNT(*) AS nall
        FROM {tbl}
        WHERE {col} IS NOT NULL
        GROUP BY 1, 2
    ),
    tot AS (
        SELECT client_id, SUM(n90) AS s90, SUM(nall) AS sall
        FROM counts
        GROUP BY 1
    )
    SELECT
        c.client_id,
        CAST(-SUM(CASE WHEN c.n90 > 0 AND t.s90 > 0 THEN (c.n90::DOUBLE / t.s90) * LN(c.n90::DOUBLE / t.s90) ELSE 0.0 END) AS FLOAT) AS {prefix}_90d_entropy,
        CAST(-SUM(CASE WHEN c.nall > 0 AND t.sall > 0 THEN (c.nall::DOUBLE / t.sall) * LN(c.nall::DOUBLE / t.sall) ELSE 0.0 END) AS FLOAT) AS {prefix}_all_entropy,
        (MAX(c.n90)::DOUBLE / NULLIF(MAX(t.s90), 0))::FLOAT AS {prefix}_90d_top_share,
        (MAX(c.nall)::DOUBLE / NULLIF(MAX(t.sall), 0))::FLOAT AS {prefix}_all_top_share
    FROM counts c
    JOIN tot t USING (client_id)
    GROUP BY 1
    """


def build_new_trx_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    print("[FE-v2 1/2] Extracting new TRX features across 5 folds x 10 months (MSK) ...", flush=True)
    t0 = time.perf_counter()
    frames = []
    for fold in range(5):
        tf = time.perf_counter()
        trx_glob = (DATA_ROOT / "detail" / "trx" / f"fold={fold}" / "*.parquet").as_posix()
        for m in MODELING_MONTHS:
            c_msk = f"TIMESTAMP '{m}-01 00:00:00'"
            c_30 = f"{c_msk} - INTERVAL 30 DAY"
            c_90 = f"{c_msk} - INTERVAL 90 DAY"

            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE m_trx AS
                SELECT
                    client_id,
                    event_time + INTERVAL 3 HOUR AS event_time_msk,
                    amount,
                    event_type,
                    event_subtype,
                    currency,
                    src_type11, src_type12, src_type21, src_type22, src_type31, src_type32,
                    dst_type11, dst_type12
                FROM read_parquet('{trx_glob}')
                WHERE event_time + INTERVAL 3 HOUR < {c_msk}
            """)

            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE t_core AS
                SELECT
                    client_id,
                    '{m}' AS month,
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
                FROM m_trx
                GROUP BY client_id
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

            df_m = con.execute("""
                SELECT
                    c.*,
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
            """).df()
            frames.append(df_m)
        print(f"  TRX fold={fold} done ({time.perf_counter()-tf:.1f}s)", flush=True)

    out = pd.concat(frames, ignore_index=True)
    print(f"[FE-v2 1/2] Done TRX new features in {time.perf_counter()-t0:.1f}s", flush=True)
    return out


def build_new_geo_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    print("[FE-v2 2/2] Extracting new GEO features across 5 folds x 10 months (MSK) ...", flush=True)
    t0 = time.perf_counter()
    frames = []
    for fold in range(5):
        tf = time.perf_counter()
        geo_glob = (DATA_ROOT / "detail" / "geo" / f"fold={fold}" / "*.parquet").as_posix()
        for m in MODELING_MONTHS:
            c_msk = f"TIMESTAMP '{m}-01 00:00:00'"
            c_30 = f"{c_msk} - INTERVAL 30 DAY"

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

            df_m = con.execute(f"""
                WITH active AS (SELECT DISTINCT client_id FROM m_geo)
                SELECT
                    a.client_id,
                    '{m}' AS month,
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
            """).df()
            frames.append(df_m)
        print(f"  GEO fold={fold} done ({time.perf_counter()-tf:.1f}s)", flush=True)

    out = pd.concat(frames, ignore_index=True)
    print(f"[FE-v2 2/2] Done GEO new features in {time.perf_counter()-t0:.1f}s", flush=True)
    return out


def main() -> None:
    if OUT_CANDIDATES.exists():
        print(f"Already exists: {OUT_CANDIDATES}", flush=True)
        return

    tmp_dir = OUT_DIR / "_duck_tmp"
    tmp_dir.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='2.2GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{tmp_dir.as_posix()}'")

    trx_new = build_new_trx_features(con)
    geo_new = build_new_geo_features(con)
    con.close()

    print("[FE-v2] Merging with panel_moscow.parquet & computing cross-source ratios ...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    # Keep panel_moscow's keys + targets + 112 behavioral cols (exclude prior_* from final modeling, keep y_any/fold/month/client_id)
    keep_base = [c for c in panel.columns if not c.startswith("prior_")]
    panel = panel[keep_base].copy()

    merged = panel.merge(trx_new, on=["month", "client_id"], how="left")
    del panel, trx_new
    gc.collect()
    merged = merged.merge(geo_new, on=["month", "client_id"], how="left")
    del geo_new
    gc.collect()

    # Fill 0 for count/flag/nunique columns when absent
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
        merged[c] = merged[c].fillna(0).astype(np.float32)

    # Derived in-panel velocity, ratio, density, cross-source & spatial features
    merged["trx_main_neg_ratio_30d"] = (merged["trx_30d_main_neg_count"] / np.maximum(merged["trx_30d_main_count"], 1)).astype(np.float32)
    merged["trx_main_neg_ratio_all"] = (merged["trx_all_main_neg_count"] / np.maximum(merged["trx_all_main_count"], 1)).astype(np.float32)
    merged["trx_share_90d_of_all"] = np.where(merged["trx_all_event_count"] > 0, merged["trx_90d_event_count"] / merged["trx_all_event_count"], np.nan).astype(np.float32)
    merged["trx_amt_share_30d_of_90d"] = np.where(merged["trx_90d_main_amount_abs_sum"] > 0, merged["trx_30d_main_amount_abs_sum"] / merged["trx_90d_main_amount_abs_sum"], np.nan).astype(np.float32)
    merged["trx_30d_rate_vs_tenure"] = np.where(merged["trx_rate_per_30d_tenure"] > 0, merged["trx_30d_event_count"] / merged["trx_rate_per_30d_tenure"], np.nan).astype(np.float32)
    merged["trx_active_day_density_30d"] = (merged["trx_30d_active_days"] / 30.0).astype(np.float32)
    merged["trx_active_day_density_90d"] = (merged["trx_90d_active_days"] / 90.0).astype(np.float32)
    merged["trx_tx_per_active_day_30d"] = np.where(merged["trx_30d_active_days"] > 0, merged["trx_30d_event_count"] / merged["trx_30d_active_days"], np.nan).astype(np.float32)
    merged["trx_tx_per_active_day_all"] = np.where(merged["trx_all_active_days"] > 0, merged["trx_all_event_count"] / merged["trx_all_active_days"], np.nan).astype(np.float32)
    merged["trx_dormant_days_gap"] = (merged["trx_all_days_since_first_event"] - merged["trx_all_days_since_last_event"]).astype(np.float32)

    merged["geo_all_gh6_per_gh4"] = np.where(merged["geo_all_geohash_4_nunique"] > 0, merged["geo_all_geohash_6_nunique"] / merged["geo_all_geohash_4_nunique"], np.nan).astype(np.float32)
    merged["geo_30d_gh6_per_gh4"] = np.where(merged["geo_30d_geohash_4_nunique"] > 0, merged["geo_30d_geohash_6_nunique"] / merged["geo_30d_geohash_4_nunique"], np.nan).astype(np.float32)
    merged["geo_share_30d_of_90d"] = np.where(merged["geo_90d_event_count"] > 0, merged["geo_30d_event_count"] / merged["geo_90d_event_count"], np.nan).astype(np.float32)

    merged["dialog_share_30d_of_90d"] = np.where(merged["dialog_90d_event_count"] > 0, merged["dialog_30d_event_count"] / merged["dialog_90d_event_count"], np.nan).astype(np.float32)
    merged["dialog_first_activated_30d"] = ((merged["dialog_30d_event_count"] > 0) & (merged["dialog_30d_event_count"] == merged["dialog_all_event_count"])).astype(np.int8)
    merged["is_silent_trx_90d_active_dlg"] = ((merged["trx_90d_has_event"] == 0) & (merged["dialog_90d_has_event"] == 1)).astype(np.int8)
    merged["cross_source_active_count_90d"] = (merged["trx_90d_has_event"] + merged["geo_90d_has_event"] + merged["dialog_90d_has_event"]).astype(np.int8)
    merged["cross_source_active_count_all"] = (merged["trx_all_has_event"] + merged["geo_all_has_event"] + merged["dialog_all_has_event"]).astype(np.int8)
    merged["trx_to_geo_log_ratio_all"] = (merged["trx_all_log1p_event_count"] - merged["geo_all_log1p_event_count"]).astype(np.float32)

    rec_stack = np.vstack([
        merged["trx_all_days_since_last_event"].to_numpy(dtype=np.float32),
        merged["geo_all_days_since_last_event"].to_numpy(dtype=np.float32),
        merged["dialog_all_days_since_last_event"].to_numpy(dtype=np.float32),
    ])
    with np.errstate(all="ignore"):
        merged["all_min_days_since_last_event"] = np.nanmin(rec_stack, axis=0).astype(np.float32)

    # Downcast float64 feature cols (keep keys/targets untouched)
    meta_set = {"client_id", "fold", "month", "cutoff_msk", "y_any", "target_1", "target_2", "target_3", "target_4"}
    for c in merged.columns:
        if c in meta_set:
            continue
        if merged[c].dtype == np.float64:
            merged[c] = merged[c].astype(np.float32)

    merged = merged.sort_values(["month", "client_id"]).reset_index(drop=True)
    assert len(merged) == 1002240, f"Unexpected row count {len(merged)}"
    tmp = OUT_CANDIDATES.with_suffix(".tmp.parquet")
    merged.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(OUT_CANDIDATES)
    feat_cols = [c for c in merged.columns if c not in meta_set]
    print(f"[FE-v2] Saved {OUT_CANDIDATES} | shape={merged.shape} | tabular_candidates={len(feat_cols)}", flush=True)


if __name__ == "__main__":
    main()
