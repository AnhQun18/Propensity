"""Deep exploration of new findings in TRX, GEO, and TRX x GEO relationships."""
from pathlib import Path
import json
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(r"c:\Users\admin\Documents\Thực tập VSF\new propensity")
FIG_DIR = ROOT / "outputs/eda_figures_moscow"
BASE_DIR = ROOT / "outputs/full_population_temporal_eda_20261118"
OUT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\brain\107d8803-f648-48d5-b352-6bed686594c4\scratch")

c = duckdb.connect()

# 1. Inspect schemas of event_type_monthly_profiles and geohash_4_monthly_profiles
for name in ["event_type_monthly_profiles.parquet", "geohash_4_monthly_profiles.parquet", "geohash_6_monthly_profiles.parquet"]:
    df_sample = pd.read_parquet(FIG_DIR / name)
    print(f"=== {name} ({len(df_sample):,} rows) ===")
    print(df_sample.dtypes)
    print(df_sample.head(3))

# 2. Load the 12-month panel and join with event_type_monthly_profiles, geohash_4_monthly_profiles, geohash_6_monthly_profiles, and 4 targets
panel = pd.read_parquet(BASE_DIR / "all_customer_months.parquet")
panel["month"] = pd.to_datetime(panel["month"])

et_prof = pd.read_parquet(FIG_DIR / "event_type_monthly_profiles.parquet")
et_prof["month"] = pd.to_datetime(et_prof["month"])
et_prof = et_prof.rename(columns={col: f"et_{col}" for col in et_prof.columns if col not in ("client_id", "month")})

g4_prof = pd.read_parquet(FIG_DIR / "geohash_4_monthly_profiles.parquet")
g4_prof["month"] = pd.to_datetime(g4_prof["month"])
g4_prof = g4_prof.rename(columns={col: f"g4_{col}" for col in g4_prof.columns if col not in ("client_id", "month")})

g6_prof = pd.read_parquet(FIG_DIR / "geohash_6_monthly_profiles.parquet")
g6_prof["month"] = pd.to_datetime(g6_prof["month"])
g6_prof = g6_prof.rename(columns={col: f"g6_{col}" for col in g6_prof.columns if col not in ("client_id", "month")})

targets_path = (ROOT / "data/MBD-mini/targets/**/*.parquet").as_posix()
targets = c.execute(f"""
    SELECT client_id, date_trunc('month', cast(mon AS DATE))::TIMESTAMP AS month,
           target_1, target_2, target_3, target_4
    FROM read_parquet('{targets_path}')
""").df()
targets["month"] = pd.to_datetime(targets["month"])

df = (
    panel.merge(et_prof, on=["client_id", "month"], how="left")
    .merge(g4_prof, on=["client_id", "month"], how="left")
    .merge(g6_prof, on=["client_id", "month"], how="left")
    .merge(targets, on=["client_id", "month"], how="left")
)
df = df.sort_values(["client_id", "month"]).reset_index(drop=True)

# Create lag-1 (previous month T-1) features and cumulative history features before month T
g = df.groupby("client_id", sort=False)

df["trx_prev"] = g["trx_events"].shift(1).fillna(0)
df["geo_prev"] = g["geo_events"].shift(1).fillna(0)
df["dlg_prev"] = g["dialog_events"].shift(1).fillna(0)
df["amt_prev"] = g["amount11"].shift(1).fillna(0)

for col in ["et_unique_cells", "et_new_cells", "et_new_cell_event_share", "et_top_share", "et_entropy", "et_switched_top",
            "g4_unique_cells", "g4_new_cells", "g4_new_cell_event_share", "g4_top_share", "g4_entropy", "g4_switched_top",
            "g6_unique_cells", "g6_new_cells", "g6_new_cell_event_share", "g6_top_share", "g6_switched_top"]:
    if col in df.columns:
        df[f"{col}_prev"] = g[col].shift(1)

# Cumulative counts strictly before month T
df["trx_cum"] = g["trx_events"].cumsum() - df["trx_events"].fillna(0)
df["geo_cum"] = g["geo_events"].cumsum() - df["geo_events"].fillna(0)
df["dlg_cum"] = g["dialog_events"].cumsum() - df["dialog_events"].fillna(0)
df["trx_months_cum"] = g["trx_events"].transform(lambda s: (s.fillna(0) > 0).cumsum()) - (df["trx_events"].fillna(0) > 0).astype(int)
df["geo_months_cum"] = g["geo_events"].transform(lambda s: (s.fillna(0) > 0).cumsum()) - (df["geo_events"].fillna(0) > 0).astype(int)

# Prior average monthly TRX (before T-1, i.e. cumulative up to T-2 divided by active months up to T-2)
df["trx_cum_before_prev"] = df["trx_cum"] - df["trx_prev"]
df["trx_months_before_prev"] = df["trx_months_cum"] - (df["trx_prev"] > 0).astype(int)
df["trx_prior_mean"] = np.where(df["trx_months_before_prev"] > 0, df["trx_cum_before_prev"] / df["trx_months_before_prev"], np.nan)
df["trx_surge_ratio"] = np.where(df["trx_prior_mean"] > 0, df["trx_prev"] / df["trx_prior_mean"], np.nan)

# Horizon labels H1, H2, H3
df["y1"] = df["y"]
df["y2_lead"] = g["y"].shift(-1)
df["y3_lead"] = g["y"].shift(-2)

# Filter to the 10-cutoff common cohort (02/2022 - 11/2022, 1,002,240 rows)
cohort = df[df["y1"].notna() & df["y2_lead"].notna() & df["y3_lead"].notna()].copy()
cohort["h1"] = cohort["y1"].astype(int)
cohort["h2"] = ((cohort["y1"] > 0) | (cohort["y2_lead"] > 0)).astype(int)
cohort["h3"] = ((cohort["h2"] > 0) | (cohort["y3_lead"] > 0)).astype(int)
print(f"Common cohort rows: {len(cohort):,}, H1 rate: {cohort['h1'].mean()*100:.4f}%, H3 rate: {cohort['h3'].mean()*100:.4f}%")

base_h1 = cohort["h1"].mean()
base_h3 = cohort["h3"].mean()
base_p = {i: cohort[f"target_{i}"].mean() for i in range(1, 5)}

def summarize_group(frame, col_name):
    res = frame.groupby(col_name, dropna=False, observed=True).agg(
        n=("h1", "size"),
        buyers_h1=("h1", "sum"),
        rate_h1=("h1", "mean"),
        buyers_h3=("h3", "sum"),
        rate_h3=("h3", "mean"),
        rate_p1=("target_1", "mean"),
        rate_p2=("target_2", "mean"),
        rate_p3=("target_3", "mean"),
        rate_p4=("target_4", "mean"),
    ).reset_index()
    res["share_pct"] = res["n"] / len(frame) * 100
    res["lift_h1"] = res["rate_h1"] / base_h1
    res["lift_h3"] = res["rate_h3"] / base_h3
    for i in range(1, 5):
        res[f"lift_p{i}"] = res[f"rate_p{i}"] / base_p[i]
    return res

# Test 1: TRX event_type novelty in T-1 (Trying a NEW transaction category never used before by this customer!)
cohort["et_novelty_group"] = np.select(
    [
        cohort["trx_prev"] == 0,
        cohort["et_new_cells_prev"].fillna(0) == 0,
        cohort["et_new_cells_prev"].fillna(0) == 1,
        cohort["et_new_cells_prev"].fillna(0) >= 2,
    ],
    ["0_No_TRX_prev_month", "1_Active_TRX_0_new_type", "2_Active_TRX_1_new_type", "3_Active_TRX_2plus_new_types"],
    default="other"
)
print("\n--- TEST 1: TRX EVENT_TYPE NOVELTY IN T-1 ---")
t1 = summarize_group(cohort, "et_novelty_group")
print(t1[["et_novelty_group", "n", "share_pct", "rate_h1", "lift_h1", "rate_h3", "lift_h3", "lift_p1", "lift_p2", "lift_p3", "lift_p4"]].to_string(index=False))
t1.to_csv(OUT_DIR / "trx_event_type_novelty_lift.csv", index=False)

# Test 2: TRX event_type diversity in T-1 (Number of distinct transaction types in T-1)
cohort["et_diversity_group"] = np.select(
    [
        cohort["trx_prev"] == 0,
        cohort["et_unique_cells_prev"].fillna(0) == 1,
        cohort["et_unique_cells_prev"].fillna(0).between(2, 3),
        cohort["et_unique_cells_prev"].fillna(0).between(4, 6),
        cohort["et_unique_cells_prev"].fillna(0) >= 7,
    ],
    ["0_No_TRX_prev", "1_type", "2_3_types", "4_6_types", "7plus_types"],
    default="other"
)
print("\n--- TEST 2: TRX EVENT_TYPE DIVERSITY IN T-1 ---")
t2 = summarize_group(cohort, "et_diversity_group")
print(t2[["et_diversity_group", "n", "share_pct", "rate_h1", "lift_h1", "rate_h3", "lift_h3", "lift_p1", "lift_p2", "lift_p3", "lift_p4"]].to_string(index=False))
t2.to_csv(OUT_DIR / "trx_event_type_diversity_lift.csv", index=False)

# Test 3: TRX short-term surge / acceleration (T-1 TRX count vs customer's own prior monthly mean)
cohort["trx_surge_group"] = np.select(
    [
        (cohort["trx_cum"] == 0),
        (cohort["trx_prev"] == 0) & (cohort["trx_cum"] > 0),
        (cohort["trx_prev"] > 0) & (cohort["trx_cum_before_prev"] == 0),
        cohort["trx_surge_ratio"] < 0.5,
        cohort["trx_surge_ratio"].between(0.5, 1.5),
        cohort["trx_surge_ratio"] > 1.5,
    ],
    [
        "0_Never_had_TRX",
        "1_Paused_in_T_minus_1",
        "2_First_ever_TRX_in_T_minus_1",
        "3_Declining_activity (<0.5x)",
        "4_Stable_activity (0.5x-1.5x)",
        "5_Surging_activity (>1.5x)",
    ],
    default="other"
)
print("\n--- TEST 3: TRX SHORT-TERM MOMENTUM / SURGE vs PERSONAL BASELINE ---")
t3 = summarize_group(cohort, "trx_surge_group")
print(t3[["trx_surge_group", "n", "share_pct", "rate_h1", "lift_h1", "rate_h3", "lift_h3", "lift_p1", "lift_p2", "lift_p3", "lift_p4"]].to_string(index=False))
t3.to_csv(OUT_DIR / "trx_momentum_surge_lift.csv", index=False)

# Test 4: GEO Novelty & Mobility in T-1 (Visiting new geohash_4 or geohash_6 locations, or switching dominant location)
cohort["g4_novelty_group"] = np.select(
    [
        cohort["geo_prev"] == 0,
        cohort["g4_new_cells_prev"].fillna(0) == 0,
        cohort["g4_new_cells_prev"].fillna(0).between(1, 2),
        cohort["g4_new_cells_prev"].fillna(0) >= 3,
    ],
    ["0_No_GEO_prev", "1_Familiar_regions_only (0 new g4)", "2_Visited_1_2_new_g4", "3_Visited_3plus_new_g4"],
    default="other"
)
print("\n--- TEST 4: GEO NEW REGION EXPLORATION (g4_new_cells) IN T-1 ---")
t4 = summarize_group(cohort, "g4_novelty_group")
print(t4[["g4_novelty_group", "n", "share_pct", "rate_h1", "lift_h1", "rate_h3", "lift_h3", "lift_p1", "lift_p2", "lift_p3", "lift_p4"]].to_string(index=False))
t4.to_csv(OUT_DIR / "geo_novelty_lift.csv", index=False)

# Test 5: GEO Spatial Diversity in T-1 (g4_unique_cells_prev)
cohort["g4_diversity_group"] = np.select(
    [
        cohort["geo_prev"] == 0,
        cohort["g4_unique_cells_prev"].fillna(0) == 1,
        cohort["g4_unique_cells_prev"].fillna(0).between(2, 4),
        cohort["g4_unique_cells_prev"].fillna(0).between(5, 9),
        cohort["g4_unique_cells_prev"].fillna(0) >= 10,
    ],
    ["0_No_GEO_prev", "1_Single_fixed_region (1 g4)", "2_Local_2_4_regions", "3_Regional_5_9_regions", "4_Wide_mobility_10plus_regions"],
    default="other"
)
print("\n--- TEST 5: GEO SPATIAL MOBILITY (g4_unique_cells) IN T-1 ---")
t5 = summarize_group(cohort, "g4_diversity_group")
print(t5[["g4_diversity_group", "n", "share_pct", "rate_h1", "lift_h1", "rate_h3", "lift_h3", "lift_p1", "lift_p2", "lift_p3", "lift_p4"]].to_string(index=False))
t5.to_csv(OUT_DIR / "geo_mobility_lift.csv", index=False)

# Test 6: Relationship between TRX Diversity (et_unique_cells_prev) and GEO Mobility (g4_unique_cells_prev)
active_both = cohort[(cohort["trx_prev"] > 0) & (cohort["geo_prev"] > 0)].copy()
print("\n--- TEST 6: CORRELATIONS BETWEEN TRX AND GEO IN T-1 (ACTIVE BOTH) ---")
for a, b in [
    ("trx_prev", "geo_prev"),
    ("et_unique_cells_prev", "g4_unique_cells_prev"),
    ("et_unique_cells_prev", "g6_unique_cells_prev"),
    ("et_new_cells_prev", "g4_new_cells_prev"),
    ("et_top_share_prev", "g4_top_share_prev"),
]:
    v = active_both[[a, b]].dropna()
    print(f"Spearman({a}, {b}) [n={len(v):,}]: {spearmanr(v[a], v[b]).statistic:.4f}")

# Test 7: Joint TRX x GEO Behavioral Matrix: (TRX Novelty / Diversity) x (GEO Mobility / Presence)
cohort["trx_tier"] = np.select(
    [
        cohort["trx_prev"] == 0,
        (cohort["trx_prev"] > 0) & (cohort["et_new_cells_prev"].fillna(0) == 0) & (cohort["et_unique_cells_prev"].fillna(0) <= 3),
        (cohort["trx_prev"] > 0) & (cohort["et_new_cells_prev"].fillna(0) == 0) & (cohort["et_unique_cells_prev"].fillna(0) >= 4),
        (cohort["trx_prev"] > 0) & (cohort["et_new_cells_prev"].fillna(0) >= 1),
    ],
    [
        "T0_No_TRX_prev",
        "T1_Routine_narrow_TRX (1-3 types, 0 new)",
        "T2_Routine_broad_TRX (4+ types, 0 new)",
        "T3_Expanding_TRX (Tried NEW event_type)",
    ],
    default="other"
)
cohort["geo_tier"] = np.select(
    [
        cohort["geo_prev"] == 0,
        cohort["g4_unique_cells_prev"].fillna(0).between(1, 3),
        cohort["g4_unique_cells_prev"].fillna(0) >= 4,
    ],
    [
        "G0_No_GEO_prev",
        "G1_Fixed_local_GEO (1-3 regions)",
        "G2_Mobile_multi_region_GEO (4+ regions)",
    ],
    default="other"
)
cohort["trx_x_geo_tier"] = cohort["trx_tier"] + "  ×  " + cohort["geo_tier"]
print("\n--- TEST 7: JOINT TRX TIER × GEO TIER INTERACTION ---")
t7 = summarize_group(cohort, "trx_x_geo_tier")
print(t7[["trx_x_geo_tier", "n", "share_pct", "rate_h1", "lift_h1", "rate_h3", "lift_h3", "lift_p1", "lift_p3", "lift_p4"]].to_string(index=False))
t7.to_csv(OUT_DIR / "trx_x_geo_tier_interaction.csv", index=False)

# Test 8: What about within TRX=0 (no cumulative TRX before cutoff): how does GEO history + GEO activity in T-1 separate buyers?
trx_zero_cum = cohort[cohort["trx_cum"] == 0].copy()
trx_zero_cum["geo_state_in_trx0"] = np.select(
    [
        trx_zero_cum["geo_cum"] == 0,
        (trx_zero_cum["geo_cum"] > 0) & (trx_zero_cum["geo_prev"] == 0),
        (trx_zero_cum["geo_prev"] > 0) & (trx_zero_cum["g4_unique_cells_prev"].fillna(0) <= 3),
        (trx_zero_cum["geo_prev"] > 0) & (trx_zero_cum["g4_unique_cells_prev"].fillna(0) >= 4),
    ],
    [
        "0_No_GEO_ever",
        "1_Old_GEO_inactive_T_minus_1",
        "2_Active_GEO_1_3_regions",
        "3_Active_GEO_4plus_regions",
    ],
    default="other"
)
print("\n--- TEST 8: GEO SIGNAL INSIDE TRX_CUM = 0 CUSTOMERS ---")
t8 = summarize_group(trx_zero_cum, "geo_state_in_trx0")
print(t8[["geo_state_in_trx0", "n", "share_pct", "rate_h1", "lift_h1", "rate_h3", "lift_h3"]].to_string(index=False))
t8.to_csv(OUT_DIR / "geo_inside_trx0_lift.csv", index=False)
