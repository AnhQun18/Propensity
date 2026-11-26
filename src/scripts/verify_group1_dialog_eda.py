"""Group 1 Dialog Verification (Part A):
1. Same-month vs Next-month purchase rates P(buy_m | call_m) vs P(buy_{m+1} | call_m)
2. Recency buckets from last call to cutoff T (0-3, 4-7, 8-14, 15-30, >30 days) + daily curve 0..30d
3. Embedding clustering (K=6 on pre-Val 768d normalized vectors) -> per-cluster same/next-month & recency
4. Norm & Cosine dependence on call count + call count vs purchase count comparison.

All timestamps converted strictly to Moscow time (UTC+3): event_time_msk = event_time + INTERVAL 3 HOUR.
Outputs written to outputs/verification_audit/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "MBD-mini"
MOSCOW_DIR = REPO_ROOT / "outputs" / "baseline_moscow_v1"
OUT_DIR = REPO_ROOT / "outputs" / "verification_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_sql_dialog_timing_checks() -> dict:
    con = duckdb.connect()
    con.execute("SET memory_limit='2.5GB'")
    con.execute("SET threads=4")

    dialog_glob = (DATA_ROOT / "detail" / "dialog" / "*" / "*.parquet").as_posix()
    targets_glob = (DATA_ROOT / "targets" / "*" / "*.parquet").as_posix()

    # 1. Same-month vs next-month purchase given call in month m (for m in 2022-02..2022-12)
    q_same_next = f"""
    WITH tgt AS (
        SELECT
            client_id,
            substr(mon, 1, 7) AS month,
            CAST(GREATEST(COALESCE(target_1,0), COALESCE(target_2,0), COALESCE(target_3,0), COALESCE(target_4,0)) AS INTEGER) AS y_any,
            CAST(COALESCE(target_1,0) + COALESCE(target_2,0) + COALESCE(target_3,0) + COALESCE(target_4,0) AS INTEGER) AS n_prod,
            CAST(COALESCE(target_1,0) AS INTEGER) AS t1,
            CAST(COALESCE(target_2,0) AS INTEGER) AS t2,
            CAST(COALESCE(target_3,0) AS INTEGER) AS t3,
            CAST(COALESCE(target_4,0) AS INTEGER) AS t4
        FROM read_parquet('{targets_glob}')
    ),
    tgt_pair AS (
        SELECT
            t.client_id,
            t.month,
            t.y_any AS y_m,
            t.n_prod AS n_prod_m,
            t.t1 AS t1_m, t.t2 AS t2_m, t.t3 AS t3_m, t.t4 AS t4_m,
            LEAD(t.y_any, 1) OVER (PARTITION BY t.client_id ORDER BY t.month) AS y_m1,
            LEAD(t.n_prod, 1) OVER (PARTITION BY t.client_id ORDER BY t.month) AS n_prod_m1,
            LEAD(t.t1, 1) OVER (PARTITION BY t.client_id ORDER BY t.month) AS t1_m1,
            LEAD(t.t2, 1) OVER (PARTITION BY t.client_id ORDER BY t.month) AS t2_m1,
            LEAD(t.t3, 1) OVER (PARTITION BY t.client_id ORDER BY t.month) AS t3_m1,
            LEAD(t.t4, 1) OVER (PARTITION BY t.client_id ORDER BY t.month) AS t4_m1
        FROM tgt t
    ),
    dlg_m AS (
        SELECT
            client_id,
            strftime(event_time + INTERVAL 3 HOUR, '%Y-%m') AS month,
            COUNT(*) AS n_calls_m
        FROM read_parquet('{dialog_glob}')
        GROUP BY 1, 2
    ),
    joined AS (
        SELECT
            p.*,
            CASE WHEN COALESCE(d.n_calls_m, 0) > 0 THEN 1 ELSE 0 END AS has_call_m,
            COALESCE(d.n_calls_m, 0) AS n_calls_m
        FROM tgt_pair p
        LEFT JOIN dlg_m d USING (client_id, month)
        WHERE p.month BETWEEN '2022-02' AND '2022-12'
    )
    SELECT
        has_call_m,
        COUNT(*) AS n_obs,
        SUM(n_calls_m) AS total_calls,
        SUM(y_m) AS pos_m,
        AVG(y_m) AS p_buy_m,
        SUM(y_m1) AS pos_m1,
        AVG(y_m1) AS p_buy_m1,
        AVG(CASE WHEN y_m = 0 THEN y_m1 END) AS p_buy_m1_given_ym0,
        AVG(CASE WHEN y_m = 1 THEN y_m1 END) AS p_buy_m1_given_ym1,
        AVG(t1_m) AS p_t1_m, AVG(t1_m1) AS p_t1_m1,
        AVG(t2_m) AS p_t2_m, AVG(t2_m1) AS p_t2_m1,
        AVG(t3_m) AS p_t3_m, AVG(t3_m1) AS p_t3_m1,
        AVG(t4_m) AS p_t4_m, AVG(t4_m1) AS p_t4_m1
    FROM joined
    GROUP BY 1
    ORDER BY 1
    """
    df_sn = con.execute(q_same_next).df()
    df_sn.to_csv(OUT_DIR / "dialog_same_vs_next_month.csv", index=False)

    # Monthly breakdown of same-month vs next-month
    q_sn_monthly = f"""
    WITH tgt AS (
        SELECT
            client_id,
            substr(mon, 1, 7) AS month,
            CAST(GREATEST(COALESCE(target_1,0), COALESCE(target_2,0), COALESCE(target_3,0), COALESCE(target_4,0)) AS INTEGER) AS y_any
        FROM read_parquet('{targets_glob}')
    ),
    tgt_pair AS (
        SELECT
            client_id, month, y_any AS y_m,
            LEAD(y_any, 1) OVER (PARTITION BY client_id ORDER BY month) AS y_m1
        FROM tgt
    ),
    dlg_m AS (
        SELECT client_id, strftime(event_time + INTERVAL 3 HOUR, '%Y-%m') AS month, COUNT(*) AS n_calls_m
        FROM read_parquet('{dialog_glob}')
        GROUP BY 1, 2
    )
    SELECT
        p.month,
        CASE WHEN COALESCE(d.n_calls_m, 0) > 0 THEN 1 ELSE 0 END AS has_call_m,
        COUNT(*) AS n_obs,
        SUM(COALESCE(d.n_calls_m, 0)) AS n_calls,
        SUM(p.y_m) AS pos_m,
        AVG(p.y_m) AS p_buy_m,
        SUM(p.y_m1) AS pos_m1,
        AVG(p.y_m1) AS p_buy_m1
    FROM tgt_pair p
    LEFT JOIN dlg_m d USING (client_id, month)
    WHERE p.month BETWEEN '2022-02' AND '2022-12'
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    df_sn_m = con.execute(q_sn_monthly).df()
    df_sn_m.to_csv(OUT_DIR / "dialog_same_vs_next_month_by_month.csv", index=False)

    # Call count vs purchase count in same month (m in 2022-02..2022-12)
    q_call_cnt = f"""
    WITH tgt AS (
        SELECT
            client_id,
            substr(mon, 1, 7) AS month,
            CAST(GREATEST(COALESCE(target_1,0), COALESCE(target_2,0), COALESCE(target_3,0), COALESCE(target_4,0)) AS INTEGER) AS y_any
        FROM read_parquet('{targets_glob}')
    ),
    tgt_pair AS (
        SELECT
            client_id, month, y_any AS y_m,
            LEAD(y_any, 1) OVER (PARTITION BY client_id ORDER BY month) AS y_m1
        FROM tgt
    ),
    dlg_m AS (
        SELECT client_id, strftime(event_time + INTERVAL 3 HOUR, '%Y-%m') AS month, COUNT(*) AS n_calls_m
        FROM read_parquet('{dialog_glob}')
        GROUP BY 1, 2
    )
    SELECT
        CASE
            WHEN COALESCE(d.n_calls_m, 0) = 0 THEN '0'
            WHEN d.n_calls_m = 1 THEN '1'
            WHEN d.n_calls_m = 2 THEN '2'
            WHEN d.n_calls_m BETWEEN 3 AND 5 THEN '3-5'
            ELSE '6+'
        END AS call_bucket_m,
        COUNT(*) AS n_obs,
        SUM(COALESCE(d.n_calls_m, 0)) AS total_calls_m,
        SUM(p.y_m) AS pos_m,
        AVG(p.y_m) AS p_buy_m,
        SUM(p.y_m1) AS pos_m1,
        AVG(p.y_m1) AS p_buy_m1
    FROM tgt_pair p
    LEFT JOIN dlg_m d USING (client_id, month)
    WHERE p.month BETWEEN '2022-02' AND '2022-12'
    GROUP BY 1
    ORDER BY MIN(COALESCE(d.n_calls_m, 0))
    """
    df_cnt = con.execute(q_call_cnt).df()
    df_cnt.to_csv(OUT_DIR / "dialog_call_count_vs_buy_same_next.csv", index=False)

    # 2. Recency buckets from last call before cutoff T (0-3, 4-7, 8-14, 15-30, >30 days, no call)
    # Use modeling months 2022-04..2023-01 (and also check 2022-02..2023-01)
    q_recency = f"""
    WITH tgt AS (
        SELECT
            client_id,
            substr(mon, 1, 7) AS month,
            CAST(substr(mon, 1, 7) || '-01 00:00:00' AS TIMESTAMP) AS cutoff_msk,
            CAST(GREATEST(COALESCE(target_1,0), COALESCE(target_2,0), COALESCE(target_3,0), COALESCE(target_4,0)) AS INTEGER) AS y_any
        FROM read_parquet('{targets_glob}')
        WHERE substr(mon, 1, 7) BETWEEN '2022-04' AND '2023-01'
    ),
    dlg AS (
        SELECT client_id, event_time + INTERVAL 3 HOUR AS event_time_msk
        FROM read_parquet('{dialog_glob}')
    ),
    last_call AS (
        SELECT
            t.client_id,
            t.month,
            t.y_any,
            MAX(d.event_time_msk) AS last_call_msk,
            EPOCH(t.cutoff_msk - MAX(d.event_time_msk)) / 86400.0 AS days_since_last_call
        FROM tgt t
        LEFT JOIN dlg d
            ON t.client_id = d.client_id AND d.event_time_msk < t.cutoff_msk
        GROUP BY 1, 2, 3, t.cutoff_msk
    )
    SELECT
        CASE
            WHEN days_since_last_call IS NULL THEN 'no_prior_call'
            WHEN days_since_last_call < 3.0 THEN '0-3d'
            WHEN days_since_last_call < 7.0 THEN '4-7d'
            WHEN days_since_last_call < 14.0 THEN '8-14d'
            WHEN days_since_last_call <= 30.0 THEN '15-30d'
            ELSE '>30d'
        END AS recency_bucket,
        COUNT(*) AS n_obs,
        SUM(y_any) AS positives,
        AVG(y_any) AS buy_rate
    FROM last_call
    GROUP BY 1
    """
    df_rec = con.execute(q_recency).df()
    order_map = {"0-3d": 0, "4-7d": 1, "8-14d": 2, "15-30d": 3, ">30d": 4, "no_prior_call": 5}
    df_rec["ord"] = df_rec["recency_bucket"].map(order_map)
    df_rec = df_rec.sort_values("ord").drop(columns=["ord"]).reset_index(drop=True)
    df_rec.to_csv(OUT_DIR / "dialog_recency_buckets_to_cutoff.csv", index=False)

    # Also exact 0-3, 4-7, 8-14, 15-30 using integer calendar days floor(days_since_last_call)
    q_recency_daily = f"""
    WITH tgt AS (
        SELECT
            client_id,
            substr(mon, 1, 7) AS month,
            CAST(substr(mon, 1, 7) || '-01 00:00:00' AS TIMESTAMP) AS cutoff_msk,
            CAST(GREATEST(COALESCE(target_1,0), COALESCE(target_2,0), COALESCE(target_3,0), COALESCE(target_4,0)) AS INTEGER) AS y_any
        FROM read_parquet('{targets_glob}')
        WHERE substr(mon, 1, 7) BETWEEN '2022-04' AND '2023-01'
    ),
    dlg AS (
        SELECT client_id, event_time + INTERVAL 3 HOUR AS event_time_msk
        FROM read_parquet('{dialog_glob}')
    ),
    last_call AS (
        SELECT
            t.client_id,
            t.month,
            t.y_any,
            FLOOR(EPOCH(t.cutoff_msk - MAX(d.event_time_msk)) / 86400.0) AS day_gap
        FROM tgt t
        LEFT JOIN dlg d
            ON t.client_id = d.client_id AND d.event_time_msk < t.cutoff_msk
        GROUP BY 1, 2, 3, t.cutoff_msk
    )
    SELECT
        CAST(day_gap AS INTEGER) AS day_gap,
        COUNT(*) AS n_obs,
        SUM(y_any) AS positives,
        AVG(y_any) AS buy_rate
    FROM last_call
    WHERE day_gap IS NOT NULL AND day_gap <= 45
    GROUP BY 1
    ORDER BY 1
    """
    df_daily = con.execute(q_recency_daily).df()
    df_daily.to_csv(OUT_DIR / "dialog_recency_daily_0_to_45d.csv", index=False)

    return {
        "same_vs_next": df_sn.to_dict("records"),
        "recency_buckets": df_rec.to_dict("records"),
        "call_count_buckets": df_cnt.to_dict("records"),
    }


def run_embedding_cluster_analysis() -> dict:
    """Cluster 768d dialog embeddings (fitted strictly on <= 2022-08-31 MSK) into K=6 clusters
    and evaluate same-month vs next-month purchase rates and recency buckets per cluster."""
    con = duckdb.connect()
    con.execute("SET memory_limit='2.5GB'")
    con.execute("SET threads=4")
    dialog_glob = (DATA_ROOT / "detail" / "dialog" / "*" / "*.parquet").as_posix()
    targets_glob = (DATA_ROOT / "targets" / "*" / "*.parquet").as_posix()

    dlg_df = con.execute(f"""
        SELECT
            client_id,
            event_time + INTERVAL 3 HOUR AS event_time_msk,
            strftime(event_time + INTERVAL 3 HOUR, '%Y-%m') AS month,
            embedding
        FROM read_parquet('{dialog_glob}')
    """).df()

    emb = np.vstack(dlg_df["embedding"].to_numpy()).astype(np.float32)
    emb_norm = normalize(emb, norm="l2", axis=1)

    train_mask = (dlg_df["event_time_msk"] < pd.Timestamp("2022-09-01")).to_numpy()
    km = MiniBatchKMeans(n_clusters=6, random_state=42, batch_size=2048, n_init=10)
    km.fit(emb_norm[train_mask])
    dlg_df["cluster"] = km.predict(emb_norm)
    dlg_df["emb_l2"] = np.linalg.norm(emb, axis=1)

    # Save lightweight event-cluster lookup
    dlg_meta = dlg_df[["client_id", "event_time_msk", "month", "cluster", "emb_l2"]].copy()
    dlg_meta.to_parquet(OUT_DIR / "dialog_events_with_cluster.parquet", index=False)

    # Register in DuckDB to compute per-cluster same-month vs next-month purchase and recency buckets
    con.register("dlg_meta", dlg_meta)

    q_cluster_sn = f"""
    WITH tgt AS (
        SELECT
            client_id,
            substr(mon, 1, 7) AS month,
            CAST(GREATEST(COALESCE(target_1,0), COALESCE(target_2,0), COALESCE(target_3,0), COALESCE(target_4,0)) AS INTEGER) AS y_any
        FROM read_parquet('{targets_glob}')
    ),
    tgt_pair AS (
        SELECT
            client_id, month, y_any AS y_m,
            LEAD(y_any, 1) OVER (PARTITION BY client_id ORDER BY month) AS y_m1
        FROM tgt
    ),
    cm AS (
        SELECT DISTINCT client_id, month, cluster
        FROM dlg_meta
        WHERE month BETWEEN '2022-02' AND '2022-12'
    )
    SELECT
        cm.cluster,
        COUNT(*) AS n_client_months,
        SUM(p.y_m) AS pos_m,
        AVG(p.y_m) AS p_buy_m,
        SUM(p.y_m1) AS pos_m1,
        AVG(p.y_m1) AS p_buy_m1,
        AVG(p.y_m) / NULLIF(AVG(p.y_m1), 0) AS same_to_next_ratio
    FROM cm
    JOIN tgt_pair p USING (client_id, month)
    GROUP BY 1
    ORDER BY 1
    """
    df_csn = con.execute(q_cluster_sn).df()

    # Cluster sizes overall
    cl_sizes = dlg_meta.groupby("cluster").agg(
        total_calls=("client_id", "count"),
        unique_clients=("client_id", "nunique"),
        mean_emb_l2=("emb_l2", "mean"),
    ).reset_index()
    df_csn = df_csn.merge(cl_sizes, on="cluster", how="left")
    df_csn.to_csv(OUT_DIR / "dialog_cluster_same_vs_next.csv", index=False)

    # Recency buckets per cluster of the LAST call before cutoff T (modeling months 2022-04..2023-01)
    q_cluster_rec = f"""
    WITH tgt AS (
        SELECT
            client_id,
            substr(mon, 1, 7) AS month,
            CAST(substr(mon, 1, 7) || '-01 00:00:00' AS TIMESTAMP) AS cutoff_msk,
            CAST(GREATEST(COALESCE(target_1,0), COALESCE(target_2,0), COALESCE(target_3,0), COALESCE(target_4,0)) AS INTEGER) AS y_any
        FROM read_parquet('{targets_glob}')
        WHERE substr(mon, 1, 7) BETWEEN '2022-04' AND '2023-01'
    ),
    ranked AS (
        SELECT
            t.client_id,
            t.month,
            t.y_any,
            d.cluster,
            EPOCH(t.cutoff_msk - d.event_time_msk) / 86400.0 AS days_gap,
            ROW_NUMBER() OVER (PARTITION BY t.client_id, t.month ORDER BY d.event_time_msk DESC) AS rn
        FROM tgt t
        JOIN dlg_meta d
            ON t.client_id = d.client_id AND d.event_time_msk < t.cutoff_msk
    )
    SELECT
        cluster,
        CASE
            WHEN days_gap < 3.0 THEN '0-3d'
            WHEN days_gap < 7.0 THEN '4-7d'
            WHEN days_gap < 14.0 THEN '8-14d'
            WHEN days_gap <= 30.0 THEN '15-30d'
            ELSE '>30d'
        END AS recency_bucket,
        COUNT(*) AS n_obs,
        SUM(y_any) AS positives,
        AVG(y_any) AS buy_rate
    FROM ranked
    WHERE rn = 1
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    df_crec = con.execute(q_cluster_rec).df()
    df_crec.to_csv(OUT_DIR / "dialog_cluster_recency_buckets.csv", index=False)

    return {
        "cluster_same_vs_next": df_csn.to_dict("records"),
    }


def run_norm_cosine_vs_call_count() -> dict:
    """Check whether mean-pool L2 norms and cosine_sim depend on dialog_event_count."""
    panel = pd.read_parquet(
        MOSCOW_DIR / "panel_moscow.parquet",
        columns=["month", "client_id", "y_any", "dialog_90d_event_count", "dialog_all_event_count", "dialog_90d_days_since_last_event"],
    )
    z = np.load(MOSCOW_DIR / "dialog_emb_moscow.npz", allow_pickle=False)
    all_norms = np.linalg.norm(z["all_emb"].astype(np.float32), axis=1)
    w90_norms = np.linalg.norm(z["w90_emb"].astype(np.float32), axis=1)

    df_all = pd.DataFrame({
        "month": z["all_months"].astype(str),
        "client_id": z["all_clients"].astype(str),
        "norm_all": all_norms,
    })
    df_w90 = pd.DataFrame({
        "month": z["w90_months"].astype(str),
        "client_id": z["w90_clients"].astype(str),
        "norm_90d": w90_norms,
    })
    # Compute cosine similarity for rows having both all and 90d embeddings
    all_map = {(m, c): i for i, (m, c) in enumerate(zip(df_all["month"], df_all["client_id"]))}
    cos_vals = np.empty(len(df_w90), dtype=np.float32)
    w90_mat = z["w90_emb"].astype(np.float32)
    all_mat = z["all_emb"].astype(np.float32)
    for i, (m, c) in enumerate(zip(df_w90["month"], df_w90["client_id"])):
        j = all_map[(m, c)]
        va, v9 = all_mat[j], w90_mat[i]
        cos_vals[i] = float(np.dot(va, v9) / max(np.linalg.norm(va) * np.linalg.norm(v9), 1e-8))
    df_w90["cosine_90d_all"] = cos_vals

    m = panel.merge(df_all, on=["month", "client_id"], how="inner").merge(df_w90, on=["month", "client_id"], how="left")

    corr_all = float(m["norm_all"].corr(np.log1p(m["dialog_all_event_count"])))
    corr_all_raw = float(m["norm_all"].corr(m["dialog_all_event_count"]))
    sub90 = m.dropna(subset=["norm_90d", "cosine_90d_all"])
    corr_90 = float(sub90["norm_90d"].corr(np.log1p(sub90["dialog_90d_event_count"])))
    corr_cos_all = float(sub90["cosine_90d_all"].corr(np.log1p(sub90["dialog_all_event_count"])))
    same_cnt_frac = float((sub90["dialog_90d_event_count"] == sub90["dialog_all_event_count"]).mean())
    cos_gt_0999_when_same = float((sub90.loc[sub90["dialog_90d_event_count"] == sub90["dialog_all_event_count"], "cosine_90d_all"] > 0.9999).mean())

    m["cnt_all_bucket"] = pd.cut(
        m["dialog_all_event_count"],
        bins=[0, 1, 2, 3, 5, 10, 100000],
        labels=["1", "2", "3", "4-5", "6-10", "11+"],
    )
    tbl = m.groupby("cnt_all_bucket", observed=False).agg(
        n_obs=("client_id", "count"),
        buy_rate=("y_any", "mean"),
        norm_all_mean=("norm_all", "mean"),
        norm_all_sd=("norm_all", "std"),
        norm_90d_mean=("norm_90d", "mean"),
        cosine_mean=("cosine_90d_all", "mean"),
        frac_cosine_1=("cosine_90d_all", lambda s: float((s > 0.9999).mean()) if s.notna().any() else np.nan),
    ).reset_index()
    tbl.to_csv(OUT_DIR / "dialog_norm_cosine_by_call_count.csv", index=False)

    return {
        "n_all_emb_rows": int(len(m)),
        "n_90d_emb_rows": int(len(sub90)),
        "corr_norm_all_vs_log1p_count": corr_all,
        "corr_norm_all_vs_raw_count": corr_all_raw,
        "corr_norm_90d_vs_log1p_count": corr_90,
        "corr_cosine_vs_log1p_count_all": corr_cos_all,
        "frac_90d_count_eq_all_count": same_cnt_frac,
        "frac_cosine_gt_09999_when_counts_eq": cos_gt_0999_when_same,
        "buy_rate_corr_norm_all": float(m["norm_all"].corr(m["y_any"])),
        "buy_rate_corr_norm_90d": float(sub90["norm_90d"].corr(sub90["y_any"])),
        "buy_rate_corr_cosine": float(sub90["cosine_90d_all"].corr(sub90["y_any"])),
    }


def main():
    t0 = time.perf_counter()
    res1 = run_sql_dialog_timing_checks()
    print("[1/3] Same-month vs next-month & recency buckets done.", flush=True)
    res2 = run_embedding_cluster_analysis()
    print("[2/3] Embedding cluster analysis done.", flush=True)
    res3 = run_norm_cosine_vs_call_count()
    print("[3/3] Norm & cosine vs call count done.", flush=True)

    summary = {
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "timing_checks": res1,
        "cluster_checks": res2,
        "norm_cosine_checks": res3,
    }
    (OUT_DIR / "group1_dialog_eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
