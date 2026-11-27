"""Compute real M4 CatBoost Predicted vs Actual probability calibration on OOT 2022-11 (fold=0).

Saves:
  - outputs/h3_propensity/metrics/stage2_m4_oot_calibration_deciles.csv
  - outputs/h3_propensity/figures/pres_04_m4_calibration_oot.png
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
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss

REPO_ROOT = Path(__file__).resolve().parents[2]
DELTA_DIR = REPO_ROOT / "outputs" / "h3_propensity" / "delta"
METRICS_DIR = REPO_ROOT / "outputs" / "h3_propensity" / "metrics"
FIG_DIR = REPO_ROOT / "outputs" / "h3_propensity" / "figures"
MOSCOW_PANEL = REPO_ROOT / "outputs" / "baseline_moscow_v1" / "panel_moscow.parquet"


def main() -> None:
    t0 = time.perf_counter()
    locked = json.loads((METRICS_DIR / "fe_h3_150_locked.json").read_text(encoding="utf-8"))
    tab_cols = list(locked["tab_cols"])

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

    tr_df = panel[(panel["month"] <= "2022-08") & (panel["fold"] != 0)]
    es_mask = tr_df["fold"].to_numpy() == 1
    ev_df = panel[(panel["month"] == "2022-11") & (panel["fold"] == 0)]

    X_tr = tr_df[tab_cols].to_numpy(dtype=np.float32)
    y_tr = tr_df["y_h3"].to_numpy(dtype=np.int8)
    X_ev = ev_df[tab_cols].to_numpy(dtype=np.float32)
    y_ev = ev_df["y_h3"].to_numpy(dtype=np.int8)

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

    p_raw = np.clip(model.predict_proba(X_ev)[:, 1], 1e-6, 1 - 1e-6)
    # Exact analytical prior-shift reversal for scale_pos_weight=3.0 (zero OOT leakage!)
    p_cal = expit(logit(p_raw) - np.log(3.0))

    ll_raw, ll_cal = log_loss(y_ev, p_raw), log_loss(y_ev, p_cal)
    br_raw, br_cal = brier_score_loss(y_ev, p_raw), brier_score_loss(y_ev, p_cal)
    print(f"[Calib OOT] Raw (spw=3): LL={ll_raw:.4f}, Brier={br_raw:.5f} | Recalibrated (-ln3): LL={ll_cal:.4f}, Brier={br_cal:.5f}", flush=True)

    df_eval = pd.DataFrame({"y": y_ev, "p_raw": p_raw, "p_cal": p_cal})
    df_eval["decile"] = pd.qcut(df_eval["p_cal"], q=10, labels=list(range(1, 11))).astype(int)
    dec = (
        df_eval.groupby("decile", as_index=False)
        .agg(n=("y", "size"), pos=("y", "sum"), actual_rate=("y", "mean"), pred_raw=("p_raw", "mean"), pred_cal=("p_cal", "mean"))
    )
    out_csv = METRICS_DIR / "stage2_m4_oot_calibration_deciles.csv"
    dec.to_csv(out_csv, index=False)
    print(dec.to_string(index=False))

    # Plot 2-panel visual chart: (Left) Decile Actual vs Predicted; (Right) Reliability Curve (Pred vs Actual)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#CBD5E1",
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.color": "#CBD5E1",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.35))

    # Panel A: Decile 1..10 Actual vs Predicted (Calibrated & Raw)
    x = dec["decile"].to_numpy()
    w = 0.38
    axes[0].bar(x - w / 2, dec["actual_rate"] * 100, width=w, color="#0D9488", label="Tỷ lệ mua thực tế (%)")
    axes[0].bar(x + w / 2, dec["pred_cal"] * 100, width=w, color="#0E192D", label="Xác suất dự báo sau hiệu chuẩn (%)")
    axes[0].plot(x, dec["pred_raw"] * 100, color="#EA580C", marker="o", lw=2.0, ls="--", label="Dự báo thô spw=3 (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"D{d}" for d in x], fontsize=9)
    axes[0].set_xlabel("Phân vị điểm dự báo (D1 thấp nhất → D10 cao nhất)", fontsize=9.5, color="#334155")
    axes[0].set_ylabel("Tỷ lệ mua / Xác suất (%)", fontsize=9.5, color="#334155")
    axes[0].set_title("Thực tế vs Dự báo theo 10 Phân vị (OOT 2022-11)", fontsize=11.2, fontweight="bold", color="#0F172A", pad=10)
    axes[0].legend(fontsize=8.2, frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", loc="upper left")

    # Panel B: Reliability Curve (Predicted % vs Actual %)
    max_v = max(dec["actual_rate"].max(), dec["pred_cal"].max()) * 100 * 1.15
    axes[1].plot([0, max_v], [0, max_v], ls="--", color="#94A3B8", lw=1.6, label="Đường chuẩn lý tưởng (y = x)")
    axes[1].plot(dec["pred_cal"] * 100, dec["actual_rate"] * 100, marker="o", color="#0D9488", lw=2.5,
                 label=f"Sau hiệu chuẩn −ln(3) (LL={ll_cal:.4f})")
    axes[1].plot(dec["pred_raw"] * 100, dec["actual_rate"] * 100, marker="s", color="#EA580C", lw=1.8, ls=":",
                 label=f"Trước hiệu chuẩn spw=3 (LL={ll_raw:.4f})")
    axes[1].set_xlim(0, max(dec["pred_raw"].max() * 100 * 1.08, max_v))
    axes[1].set_ylim(0, max_v)
    axes[1].set_xlabel("Xác suất dự báo trung bình (%)", fontsize=9.5, color="#334155")
    axes[1].set_ylabel("Tỷ lệ mua thực tế (%)", fontsize=9.5, color="#334155")
    axes[1].set_title("Đường Cong Hiệu Chuẩn (Reliability Curve)", fontsize=11.2, fontweight="bold", color="#0F172A", pad=10)
    axes[1].legend(fontsize=8.2, frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", loc="lower right")

    fig.tight_layout(pad=1.6)
    out_png = FIG_DIR / "pres_04_m4_calibration_oot.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Calib DONE] Saved {out_png.name} in {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
