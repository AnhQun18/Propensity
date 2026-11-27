"""Compute exact CatBoost TreeSHAP values for M4_catboost_150 on Locked OOT Test (2022-11, fold=0).

Saves:
  - outputs/h3_propensity/metrics/stage2_m4_oot_shap_top20.csv
  - outputs/h3_propensity/figures/pres_03_m4_shap_oot.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
DELTA_DIR = REPO_ROOT / "outputs" / "h3_propensity" / "delta"
METRICS_DIR = REPO_ROOT / "outputs" / "h3_propensity" / "metrics"
FIG_DIR = REPO_ROOT / "outputs" / "h3_propensity" / "figures"
MOSCOW_PANEL = REPO_ROOT / "outputs" / "baseline_moscow_v1" / "panel_moscow.parquet"

FRIENDLY_NAMES = {
    "trx_30d_dst_type11_nunique": "Số đích GD mới 30d (dst11)",
    "trx_all_days_since_last_event": "Số ngày từ GD gần nhất",
    "trx_all_days_since_first_event": "Tuổi lịch sử GD (raw)",
    "trx_all_days_since_first_event_cap365": "Tuổi lịch sử GD (cap 365d)",
    "trx_all_event_subtype_nunique": "Đa dạng loại GD phụ (all)",
    "trx_90d_event_type_nunique": "Đa dạng loại GD 90d",
    "trx_30d_new_dst12_count": "Số đích GD mới 30d (dst12)",
    "trx_30d_new_event_subtype_count": "Số loại GD phụ mới 30d",
    "trx_share_90d_of_all": "Tỷ trọng GD 90d / toàn lịch sử",
    "trx_tenure_frac_of_obs": "Tỷ lệ tenure / tuổi quan sát",
    "trx_all_src_type12_top_share": "Độ tập trung nguồn GD (src12)",
    "trx_all_dst_type12_top_share": "Độ tập trung đích GD (dst12)",
    "trx_all_main_amount_median": "Trung vị giá trị GD (all)",
    "trx_90d_event_subtype_entropy": "Entropy loại GD phụ 90d",
    "trx_all_event_type_entropy": "Entropy loại GD (all)",
    "prior_buy_rate_per_obs_mo": "Tần suất mua trước đây / tháng",
    "prior_1m_bought_any": "Đã mua tháng trước (prior 1m)",
    "prior_3m_buy_months": "Số tháng mua trong 3m trước",
    "prior_all_buy_months": "Tổng số tháng đã mua trước đó",
    "prior_months_since_last_buy": "Số tháng từ lần mua gần nhất",
    "prior_all_bought_p4": "Đã từng mua sản phẩm P4",
    "prior_6m_total_products": "Tổng sản phẩm mua 6m trước",
    "dialog_recency_exp_14d": "Độ tươi DIALOG (suy giảm 14d)",
    "dialog_recency_exp_45d": "Độ tươi DIALOG (suy giảm 45d)",
    "dialog_30d_event_count": "Số lượt DIALOG 30 ngày",
    "dialog_prev_cal_month_has_event": "Có DIALOG tháng lịch trước",
    "dialog_90d_days_since_last_event": "Số ngày từ DIALOG gần nhất (90d)",
    "zero_trx_90d_x_dialog_30d": "Tương tác 0-TRX 90d × DIALOG 30d",
    "geo_90d_gh6_entropy": "Entropy không gian GEO 90d (gh6)",
    "geo_30d_new_gh6_count": "Số khu vực GEO mới 30d (gh6)",
    "trx_all_active_days": "Tổng số ngày có GD (all)",
    "trx_dormant_days_gap": "Khoảng thời gian gắn bó GD",
    "trx_rate_per_30d_tenure": "Tốc độ GD / 30 ngày tenure",
    "trx_tx_per_active_day_all": "Số GD / ngày hoạt động (all)",
    "trx_all_event_type_top_share": "Độ tập trung loại GD chính",
    "trx_30d_log1p_main_amount_abs_sum": "Tổng giá trị GD 30d (log1p)",
    "trx_90d_log1p_main_amount_abs_sum": "Tổng giá trị GD 90d (log1p)",
}


def main() -> None:
    t0 = time.perf_counter()
    locked = json.loads((METRICS_DIR / "fe_h3_150_locked.json").read_text(encoding="utf-8"))
    tab_cols = list(locked["tab_cols"])

    # Load targets + base + stage2 delta for OOT train (02..08) & OOT eval (2022-11)
    tgt = pd.read_parquet(DELTA_DIR / "h3_targets_prior.parquet")
    tgt["month"] = tgt["month"].astype(str)

    early = pd.read_parquet(DELTA_DIR / "h3_early_02_03_base112.parquet")
    early["month"] = early["month"].astype(str)
    beh_cols = [c for c in early.columns if c not in ("client_id", "fold", "month")]

    moscow = pd.read_parquet(
        MOSCOW_PANEL,
        columns=["client_id", "month"] + beh_cols,
        filters=[("month", "in", ["2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-11"])],
    )
    moscow["month"] = moscow["month"].astype(str)
    beh = pd.concat([early.drop(columns=["fold"], errors="ignore"), moscow], ignore_index=True)

    s2 = pd.read_parquet(DELTA_DIR / "h3_stage2_fe_delta.parquet")
    s2["month"] = s2["month"].astype(str)

    panel = tgt.merge(beh, on=["client_id", "month"], how="inner")
    s2_new = [c for c in s2.columns if c not in panel.columns]
    panel = panel.merge(s2[["client_id", "month"] + s2_new], on=["client_id", "month"], how="inner")
    del tgt, early, moscow, beh, s2

    # Train M4 CatBoost on 2022-02..08 (fold != 0), eval & SHAP on OOT 2022-11 (fold == 0)
    tr_df = panel[(panel["month"] <= "2022-08") & (panel["fold"] != 0)]
    es_mask = tr_df["fold"].to_numpy() == 1
    ev_df = panel[(panel["month"] == "2022-11") & (panel["fold"] == 0)]

    X_tr = tr_df[tab_cols].to_numpy(dtype=np.float32)
    y_tr = tr_df["y_h3"].to_numpy(dtype=np.int8)
    X_ev = ev_df[tab_cols].to_numpy(dtype=np.float32)

    print(f"[SHAP] Training CatBoost M4 (129 tab cols, train={len(X_tr):,}, eval_oot={len(X_ev):,}) ...", flush=True)
    model = CatBoostClassifier(
        iterations=550,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=30.0,
        scale_pos_weight=3.0,
        border_count=128,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        Pool(X_tr[~es_mask], y_tr[~es_mask]),
        eval_set=Pool(X_tr[es_mask], y_tr[es_mask]),
        early_stopping_rounds=50,
        use_best_model=True,
    )
    print(f"[SHAP] Fit done (best_iter={model.get_best_iteration()}) in {time.perf_counter()-t0:.1f}s. Computing TreeSHAP ...", flush=True)

    # Exact TreeSHAP on OOT fold 0 (20,328 clients)
    shap_vals = model.get_feature_importance(Pool(X_ev), type="ShapValues")[:, :-1]
    mean_abs = np.mean(np.abs(shap_vals), axis=0)

    rows = []
    for i, col in enumerate(tab_cols):
        xv = X_ev[:, i]
        sv = shap_vals[:, i]
        fin = np.isfinite(xv)
        r = 0.0
        if np.sum(fin) > 50 and np.nanstd(xv[fin]) > 1e-9 and np.std(sv[fin]) > 1e-9:
            r = float(spearmanr(xv[fin], sv[fin]).statistic)
            if not np.isfinite(r):
                r = 0.0
        rows.append({
            "feature": col,
            "label": FRIENDLY_NAMES.get(col, col),
            "mean_abs_shap": float(mean_abs[i]),
            "spearman_val_vs_shap": r,
            "direction": "Tăng mua khi cao" if r >= 0 else "Giảm mua khi cao",
        })

    df_shap = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out_csv = METRICS_DIR / "stage2_m4_oot_shap_top20.csv"
    df_shap.head(20).to_csv(out_csv, index=False)
    print(f"[SHAP] Top 12 features on OOT 2022-11:\n{df_shap.head(12)[['feature', 'mean_abs_shap', 'spearman_val_vs_shap']].to_string(index=False)}")

    # Plot crisp directional Mean |SHAP| + Beeswarm-style visual (Top 12)
    top = df_shap.head(12).copy()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#CBD5E1",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.color": "#CBD5E1",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
    })
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    y = np.arange(len(top))
    colors = ["#0D9488" if r >= 0 else "#EA580C" for r in top["spearman_val_vs_shap"]]
    ax.barh(y, top["mean_abs_shap"], color=colors, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(top["label"], fontsize=9.6, color="#0F172A", fontweight="medium")
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (tác động trung bình lên log-odds mua hàng)", fontsize=9.5, color="#334155")
    ax.set_xlim(0, top["mean_abs_shap"].max() * 1.24)
    for idx, row in enumerate(top.itertuples()):
        sign = "+" if row.spearman_val_vs_shap >= 0 else "−"
        col_txt = "#0D9488" if row.spearman_val_vs_shap >= 0 else "#EA580C"
        ax.text(
            row.mean_abs_shap + top["mean_abs_shap"].max() * 0.018,
            idx,
            f"{row.mean_abs_shap:.3f} ({sign})",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color=col_txt,
        )
    # Clean inline legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#0D9488", label="(+) Giá trị cao → Tăng xác suất mua"),
        Patch(facecolor="#EA580C", label="(−) Giá trị cao → Giảm xác suất mua"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=8.8)
    fig.tight_layout(pad=1.2)
    out_png = FIG_DIR / "pres_03_m4_shap_oot.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[SHAP DONE] Saved {out_png.name} in {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
