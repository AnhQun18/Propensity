"""Generate a minimalist, chart-first 16:9 PDF presentation for H=3 Propensity.

Design rules:
- Single-model focus: M4_catboost_gpu_150_tuned is the Final Model (zero mention of M5/blend).
- Separate dedicated charts for every single one of the 6 warning verifications.
- Minimal text, chart-first, zero bottom footers / source captions.
"""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = REPO_ROOT / "outputs" / "h3_propensity" / "metrics"
FIG_DIR = REPO_ROOT / "outputs" / "h3_propensity" / "figures"
OUT_HTML = REPO_ROOT / "outputs" / "h3_propensity" / "H3_PROPENSITY_SLIDES.html"
OUT_PDF = REPO_ROOT / "H3_PROPENSITY_PRESENTATION.pdf"

EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]


def _render_no_m5_charts() -> tuple[Path, Path]:
    """Generate clean Val Rolling and OOT Comparison PNGs excluding M5, styled in deck palette."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
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

    # 1) Validation Rolling Curve (single models only, no M5)
    s1_m = pd.read_csv(METRICS_DIR / "stage1_baselines_val_monthly.csv")
    s2_m = pd.read_csv(METRICS_DIR / "stage2_150_val_monthly.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    series = [
        (s1_m, "B2_logreg_top50", "B2 LogReg (50)", "--", "#94A3B8", 1.8),
        (s1_m, "B4_xgb_gpu_top50", "B4 XGB GPU (50)", ":", "#64748B", 1.8),
        (s1_m, "B5_catboost_gpu_top50", "B5 CatBoost GPU (50)", "-.", "#0E192D", 2.2),
        (s2_m, "M2_lgbm_150_tuned", "M2 LightGBM (150)", "--", "#14B8A6", 2.0),
        (s2_m, "M3_xgb_gpu_150_tuned", "M3 XGB GPU (150)", "-", "#0F766E", 2.0),
        (s2_m, "M4_catboost_gpu_150_tuned", "M4 CatBoost GPU (150 — Final)", "-", "#EA580C", 3.0),
    ]
    for df_src, eid, lbl, ls, col, lw in series:
        sub = df_src[df_src["exp_id"] == eid].sort_values("month")
        axes[0].plot(sub["month"], sub["lift_1pct_mean"], marker="o", linestyle=ls, color=col, linewidth=lw, label=lbl)
        axes[1].plot(sub["month"], sub["average_precision_mean"], marker="o", linestyle=ls, color=col, linewidth=lw, label=lbl)
    axes[0].set_title("Validation Rolling H=3: Lift@1% (K=1.003)", fontsize=11.5, fontweight="bold", color="#0F172A", pad=10)
    axes[0].set_ylabel("Lift@1%", fontsize=10, color="#334155")
    axes[1].set_title("Validation Rolling H=3: PR-AUC", fontsize=11.5, fontweight="bold", color="#0F172A", pad=10)
    axes[1].set_ylabel("PR-AUC", fontsize=10, color="#334155")
    axes[1].legend(fontsize=8.3, frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", loc="lower right")
    fig.tight_layout(pad=1.6)
    p1 = FIG_DIR / "pres_01_val_rolling_single.png"
    fig.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2) Locked OOT Test (2022-11) Model Comparison with 95% CI (single models only, no M5)
    s1_o = pd.read_csv(METRICS_DIR / "stage1_baselines_oot_test_locked.csv")
    s2_o = pd.read_csv(METRICS_DIR / "stage2_150_oot_test_summary.csv")
    s2_o = s2_o[s2_o["exp_id"] != "M5_final_blend_150"].copy()
    comb = pd.concat([
        s1_o[["exp_id", "n_features", "average_precision_mean", "average_precision_ci_low", "average_precision_ci_high",
              "lift_1pct_mean", "lift_1pct_ci_low", "lift_1pct_ci_high"]],
        s2_o[["exp_id", "n_features", "average_precision_mean", "average_precision_ci_low", "average_precision_ci_high",
              "lift_1pct_mean", "lift_1pct_ci_low", "lift_1pct_ci_high"]],
    ], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.1))
    y = np.arange(len(comb))
    labels = [f"{r.exp_id} ({int(r.n_features)})" for r in comb.itertuples()]
    colors = []
    for eid in comb["exp_id"]:
        if eid == "M4_catboost_gpu_150_tuned":
            colors.append("#EA580C")
        elif eid == "B5_catboost_gpu_top50":
            colors.append("#0E192D")
        elif eid.startswith("M"):
            colors.append("#0D9488")
        else:
            colors.append("#94A3B8")

    pr_err = [comb["average_precision_mean"] - comb["average_precision_ci_low"],
              comb["average_precision_ci_high"] - comb["average_precision_mean"]]
    l1_err = [comb["lift_1pct_mean"] - comb["lift_1pct_ci_low"],
              comb["lift_1pct_ci_high"] - comb["lift_1pct_mean"]]

    axes[0].barh(y, comb["average_precision_mean"], xerr=pr_err, color=colors, height=0.68,
                 error_kw={"ecolor": "#334155", "lw": 1.1, "capsize": 2.5})
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8.5, color="#1E293B")
    axes[0].invert_yaxis()
    axes[0].set_title("Locked OOT Test (2022-11): PR-AUC [95% CI]", fontsize=10.8, fontweight="bold", color="#0F172A", pad=8)

    axes[1].barh(y, comb["lift_1pct_mean"], xerr=l1_err, color=colors, height=0.68,
                 error_kw={"ecolor": "#334155", "lw": 1.1, "capsize": 2.5})
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([""] * len(y))
    axes[1].invert_yaxis()
    axes[1].set_title("Locked OOT Test (2022-11): Lift@1% [95% CI]", fontsize=10.8, fontweight="bold", color="#0F172A", pad=8)

    fig.tight_layout(pad=1.4)
    p2 = FIG_DIR / "pres_02_oot_single_ci.png"
    fig.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3) Clean Directional SHAP Bar Chart from stage2_m4_oot_shap_top20.csv (excluding constant month offset)
    shap_df = pd.read_csv(METRICS_DIR / "stage2_m4_oot_shap_top20.csv")
    shap_df = shap_df[~shap_df["feature"].str.startswith("obs_age_")].head(10).copy()
    nice = {
        "trx_all_event_subtype_nunique": "Đa dạng loại GD phụ (all)",
        "trx_all_days_since_first_event": "Tuổi lịch sử GD (raw)",
        "trx_all_days_since_last_event": "Số ngày từ GD gần nhất",
        "trx_90d_dst_type11_nunique": "Đa dạng đích GD 90d (dst11)",
        "trx_dormant_days_gap": "Khoảng thời gian gắn bó GD",
        "trx_all_days_since_first_event_cap365": "Tuổi lịch sử GD (cap 365d)",
        "trx_30d_new_event_subtype_count": "Số loại GD phụ mới 30d",
        "trx_90d_log1p_event_count": "Tần suất GD 90d (log1p)",
        "trx_all_dst_type12_nunique": "Đa dạng đích GD (dst12, all)",
        "trx_all_event_type_top_share": "Độ tập trung loại GD chính",
        "geo_90d_gh4_entropy": "Entropy không gian GEO 90d",
    }
    shap_df["nice"] = shap_df["feature"].map(lambda c: nice.get(c, c))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    y_s = np.arange(len(shap_df))
    cols_s = ["#0D9488" if r >= 0 else "#EA580C" for r in shap_df["spearman_val_vs_shap"]]
    ax.barh(y_s, shap_df["mean_abs_shap"], color=cols_s, height=0.66)
    ax.set_yticks(y_s)
    ax.set_yticklabels(shap_df["nice"], fontsize=9.8, color="#0F172A")
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| trên OOT 2022-11 (tác động lên log-odds mua)", fontsize=9.2, color="#334155")
    ax.set_xlim(0, shap_df["mean_abs_shap"].max() * 1.28)
    for i, r in enumerate(shap_df.itertuples()):
        sgn = "+" if r.spearman_val_vs_shap >= 0 else "−"
        ct = "#0D9488" if r.spearman_val_vs_shap >= 0 else "#EA580C"
        ax.text(r.mean_abs_shap + 0.002, i, f"{r.mean_abs_shap:.3f} ({sgn})", va="center", fontsize=8.8, fontweight="bold", color=ct)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#0D9488", label="(+) Giá trị cao → Tăng khả năng mua"),
            Patch(facecolor="#EA580C", label="(−) Giá trị cao → Giảm khả năng mua"),
        ],
        loc="lower right", frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=8.5,
    )
    fig.tight_layout(pad=1.0)
    p3 = FIG_DIR / "pres_03_m4_shap_oot.png"
    fig.savefig(p3, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p1, p2, p3


def _b64_path(p: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"


def build_html(fig1_uri: str, fig2_uri: str, fig3_uri: str) -> str:
    html = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<title>Propensity H=3 — Bản Thuyết Trình</title>
<style>
  @page {
    size: 13.333in 7.5in;
    margin: 0;
  }
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  body {
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Arial, sans-serif;
    color: #0F172A;
    background: #FFFFFF;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .slide {
    width: 13.333in;
    height: 7.5in;
    background: #FFFFFF;
    padding: 52px 72px 46px 72px;
    position: relative;
    overflow: hidden;
    page-break-after: always;
    break-after: page;
    display: flex;
    flex-direction: column;
  }
  .slide:last-child {
    page-break-after: auto;
    break-after: auto;
  }

  /* Dark Cover & Closing */
  .slide-dark {
    background: #0E192D;
    color: #FFFFFF;
    justify-content: center;
    padding: 72px 84px;
  }
  .slide-dark-center {
    background: #0E192D;
    color: #FFFFFF;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 72px 110px;
  }
  .kicker {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    margin-bottom: 10px;
  }
  .k-orange { color: #EA580C; }
  .k-teal { color: #0D9488; }
  .k-mint { color: #2DD4BF; }

  .cover-title {
    font-size: 46px;
    font-weight: 800;
    line-height: 1.16;
    color: #FFFFFF;
    max-width: 10.4in;
    margin-bottom: 20px;
  }
  .cover-sub {
    font-size: 19px;
    color: #94A3B8;
    line-height: 1.5;
    max-width: 8.8in;
    margin-bottom: 30px;
  }
  .bar-teal { width: 64px; height: 4px; background: #0D9488; border-radius: 2px; }
  .bar-orange { width: 56px; height: 4px; background: #EA580C; border-radius: 2px; margin: 26px auto; }

  /* Light Slide Header */
  .title {
    font-size: 31px;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.2;
    margin-bottom: 8px;
  }
  .subtitle {
    font-size: 15.5px;
    color: #64748B;
    line-height: 1.4;
    margin-bottom: 26px;
  }
  .content {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 20px;
  }

  /* Grids & Cards */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px; }
  .grid-2-bal { display: grid; grid-template-columns: 1.12fr 0.88fr; gap: 26px; align-items: center; }

  .card {
    background: #F8FAFC;
    border-radius: 16px;
    padding: 24px 26px;
  }
  .card-white {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px 26px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
  }
  .card-mint {
    background: #F0FDFA;
    border-radius: 16px;
    padding: 26px 28px;
  }
  .card-peach {
    background: #FFF7ED;
    border-radius: 16px;
    padding: 26px 28px;
  }

  .hero-num {
    font-size: 44px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 10px;
    letter-spacing: -0.5px;
  }
  .c-teal { color: #0D9488; }
  .c-orange { color: #EA580C; }
  .c-dark { color: #0F172A; }

  .hero-lbl {
    font-size: 15px;
    font-weight: 700;
    color: #1E293B;
    line-height: 1.35;
  }
  .hero-sub {
    font-size: 13.5px;
    color: #64748B;
    line-height: 1.45;
    margin-top: 6px;
  }
  .banner-note {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 15px;
    color: #334155;
    box-shadow: 0 2px 10px rgba(15,23,42,0.03);
  }

  /* Tables */
  table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 13.8px;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(15, 23, 42, 0.05);
  }
  th {
    background: #0E192D;
    color: #FFFFFF;
    font-weight: 600;
    text-align: left;
    padding: 12px 15px;
    font-size: 13.2px;
  }
  th.num, td.num { text-align: center; }
  td {
    padding: 11px 15px;
    background: #FFFFFF;
    border-bottom: 1px solid #F1F5F9;
    color: #1E293B;
    vertical-align: middle;
  }
  tr:last-child td { border-bottom: none; }
  tr.row-mint td { background: #F0FDFA; font-weight: 700; color: #0F172A; }
  tr.row-peach td { background: #FFF7ED; font-weight: 700; color: #0F172A; }
  .txt-teal { color: #0D9488; font-weight: 700; }
  .txt-orange { color: #EA580C; font-weight: 700; }
  code {
    font-family: Consolas, monospace;
    font-size: 12.2px;
    background: #EEF2F6;
    padding: 1px 5px;
    border-radius: 4px;
    color: #0F172A;
  }
  th code { background: rgba(255,255,255,0.16); color: #E2E8F0; }

  /* Horizontal Bar Chart */
  .hbar-wrap {
    background: #F8FAFC;
    border-radius: 16px;
    padding: 22px 26px;
    display: flex;
    flex-direction: column;
    gap: 13px;
  }
  .hbar-row {
    display: grid;
    grid-template-columns: 235px 1fr 78px;
    align-items: center;
    gap: 14px;
  }
  .hbar-lbl {
    font-size: 13.5px;
    font-weight: 600;
    color: #1E293B;
  }
  .hbar-track {
    height: 22px;
    background: #E2E8F0;
    border-radius: 6px;
    overflow: hidden;
  }
  .hbar-fill {
    height: 100%;
    border-radius: 6px;
  }
  .hbar-val {
    font-size: 14px;
    font-weight: 700;
    text-align: right;
  }

  /* 4-Step Flow */
  .step-flow {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
  }
  .step-card {
    flex: 1;
    background: #F8FAFC;
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
  }
  .step-card.peach { background: #FFF7ED; }
  .step-circle {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    color: #FFFFFF;
    font-size: 15px;
    font-weight: 800;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 10px;
  }
  .step-title { font-size: 15px; font-weight: 700; color: #0F172A; margin-bottom: 5px; }
  .step-desc { font-size: 12.8px; color: #475569; line-height: 1.4; }
  .step-arrow { font-size: 22px; font-weight: 700; color: #0D9488; }

  /* Timeline */
  .tl-grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 6px; }
  .tl-box { border-radius: 10px; padding: 12px 4px; text-align: center; font-size: 12px; font-weight: 700; }
  .tl-tr { background: #E2E8F0; color: #0F172A; }
  .tl-va { background: #CCFBF1; color: #115E59; }
  .tl-em { background: #FFEDD5; color: #9A3412; border: 1.5px solid #EA580C; }
  .tl-oo { background: #0D9488; color: #FFFFFF; }
  .tl-oo2 { background: #F0FDFA; color: #0D9488; border: 1.5px dashed #0D9488; }

  /* Image Box */
  .img-clean {
    display: flex;
    align-items: center;
    justify-content: center;
    background: #FFFFFF;
    border-radius: 14px;
    overflow: hidden;
  }
  .img-clean img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
  }
  .pill-ok {
    display: inline-block;
    background: #CCFBF1;
    color: #0F766E;
    font-size: 11.5px;
    font-weight: 800;
    padding: 3px 9px;
    border-radius: 10px;
    margin-bottom: 8px;
  }
  .pill-warn {
    display: inline-block;
    background: #FFEDD5;
    color: #C2410C;
    font-size: 11.5px;
    font-weight: 800;
    padding: 3px 9px;
    border-radius: 10px;
    margin-bottom: 8px;
  }
</style>
</head>
<body>

<!-- ==================== SLIDE 1: DARK COVER ==================== -->
<section class="slide slide-dark">
  <div class="kicker k-orange">BÁO CÁO THỰC NGHIỆM DỮ LIỆU &bull; PROPENSITY H=3</div>
  <div class="cover-title">Mô Hình Dự Đoán Xu Hướng Mua Hàng<br/>Horizon = 3 Tháng &bull; Rolling Snapshot = 1</div>
  <div class="cover-sub">Từ kiến trúc chống rò rỉ Embargo 2 tháng, 5 Baselines, kiểm chứng từng cảnh báo, tối ưu 150 đặc trưng, hiệu chuẩn xác suất đến giải thích SHAP trên GPU T4</div>
  <div class="bar-teal"></div>
</section>

<!-- ==================== SLIDE 2: H=3 ZERO-LEAK DESIGN ==================== -->
<section class="slide">
  <div class="kicker k-teal">THIẾT KẾ BÀI TOÁN &bull; CHỐNG RÒ RỈ THỜI GIAN</div>
  <div class="title">Cửa Sổ Trượt H=3 (Snapshot = 1) &amp; Embargo 2 Tháng Khóa OOT</div>
  <div class="subtitle">Chia 5-Fold Out-of-Client (80.179 train / 20.045 eval) &nbsp;|&nbsp; Lọc cứng toàn bộ lịch sử &lt; 00:00 Moscow (UTC+3)</div>
  <div class="content">
    <div class="card" style="padding:20px 24px;">
      <div class="tl-grid">
        <div class="tl-box tl-tr">2022-02<br/>Train</div>
        <div class="tl-box tl-tr">2022-03<br/>Train</div>
        <div class="tl-box tl-tr">2022-04<br/>Train</div>
        <div class="tl-box tl-va">2022-05<br/>Val #1</div>
        <div class="tl-box tl-va">2022-06<br/>Val #2</div>
        <div class="tl-box tl-va">2022-07<br/>Val #3</div>
        <div class="tl-box tl-va">2022-08<br/>Val #4</div>
        <div class="tl-box tl-em">2022-09<br/>EMBARGO</div>
        <div class="tl-box tl-em">2022-10<br/>EMBARGO</div>
        <div class="tl-box tl-oo">2022-11<br/>OOT TEST</div>
        <div class="tl-box tl-oo2">2022-12<br/>(H3 m+1)</div>
        <div class="tl-box tl-oo2">2023-01<br/>(H3 m+2)</div>
      </div>
    </div>

    <table>
      <thead>
        <tr>
          <th>Giai đoạn</th>
          <th>Tháng đánh giá</th>
          <th>Các tháng Train</th>
          <th class="num">Số dòng</th>
          <th class="num">Tỷ lệ nền H=3</th>
          <th>Rào chắn chống rò rỉ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Validation (4 bước trượt)</strong></td>
          <td>2022-05 &rarr; 2022-08</td>
          <td>[02] &rarr; [02..05]</td>
          <td class="num">400.896</td>
          <td class="num">2,955%</td>
          <td>100% nhãn train đóng xong trước eval_month</td>
        </tr>
        <tr class="row-peach">
          <td class="txt-orange">Embargo (Cách ly cứng)</td>
          <td>2022-09 &amp; 2022-10</td>
          <td>Không làm nhãn train</td>
          <td class="num">—</td>
          <td class="num">—</td>
          <td>Ngăn nhãn H=3 (s+1, s+2) tràn vào OOT; vẫn dùng log 09–10 làm feature cho 11</td>
        </tr>
        <tr class="row-mint">
          <td class="txt-teal">Locked OOT Test</td>
          <td>2022-11 (phủ 11..01)</td>
          <td>[02 .. 08]</td>
          <td class="num">100.224</td>
          <td class="num">2,168%</td>
          <td>Toàn bộ 7 tháng nhãn H=3 đã đóng tại 2022-10-31</td>
        </tr>
      </tbody>
    </table>
  </div>
</section>

<!-- ==================== SLIDE 5: MUC 2A - TOP 50 FEATURES ==================== -->
<section class="slide">
  <div class="kicker k-teal">MỤC 2 &bull; TẬP 50 BIẾN BASELINE (TOP50_FEATURES)</div>
  <div class="title">50 Biến Ban Đầu Xoay Quanh Gì &amp; Cách Tính Chống Rò Rỉ</div>
  <div class="subtitle">Toàn bộ 50 biến tổng hợp trên toàn lịch sử &lt; 00:00 Moscow (UTC+3) &nbsp;|&nbsp; Đưa sẵn <code>prior_*</code> đã đóng vào để so sánh công bằng</div>
  <div class="content" style="gap:18px;">
    <div class="card" style="padding:18px 24px;">
      <div style="font-size:14px; font-weight:700; color:#0F172A; margin-bottom:10px;">Cơ cấu 50 biến Baseline theo 5 nhóm nguồn</div>
      <svg viewBox="0 0 1000 56" width="100%" height="52">
        <rect x="0" y="6" width="396" height="28" rx="6" fill="#0E192D"/>
        <text x="198" y="25" text-anchor="middle" font-size="13" font-weight="700" fill="#FFFFFF">TRX: 20 biến (40%)</text>

        <rect x="402" y="6" width="256" height="28" rx="6" fill="#0D9488"/>
        <text x="530" y="25" text-anchor="middle" font-size="13" font-weight="700" fill="#FFFFFF">Prior: 13 biến (26%)</text>

        <rect x="664" y="6" width="176" height="28" rx="6" fill="#EA580C"/>
        <text x="752" y="25" text-anchor="middle" font-size="13" font-weight="700" fill="#FFFFFF">DIALOG: 9 (18%)</text>

        <rect x="846" y="6" width="132" height="28" rx="6" fill="#64748B"/>
        <text x="912" y="25" text-anchor="middle" font-size="12.5" font-weight="700" fill="#FFFFFF">GEO: 7 (14%)</text>

        <rect x="984" y="6" width="16" height="28" rx="4" fill="#CBD5E1"/>
        <text x="992" y="49" text-anchor="end" font-size="11.5" font-weight="600" fill="#64748B">+1 cold_start (2%)</text>
      </svg>
    </div>

    <table>
      <thead>
        <tr>
          <th style="width:18%;">Nhóm biến</th>
          <th class="num" style="width:8%;">SL</th>
          <th style="width:30%;">Xoay quanh tín hiệu gì?</th>
          <th style="width:44%;">Cách tính cốt lõi (<code>&lt; YYYY-MM-01 00:00 MSK</code>)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>1. TRX (Giao dịch)</strong></td>
          <td class="num"><strong>20</strong></td>
          <td>Tần suất, độ tươi, giá trị, đa dạng &amp; ngoại tệ</td>
          <td><code>count / log1p</code>, <code>active_days</code>, <code>days_since_last/first</code>, <code>median / max / signed_log_mean</code> (30d/90d/all), <code>nunique(type/subtype)</code>, <code>foreign_fraction</code></td>
        </tr>
        <tr class="row-mint">
          <td class="txt-teal">2. Prior (Lịch sử mua)</td>
          <td class="num txt-teal">13</td>
          <td>Thói quen mua lặp &amp; từng sản phẩm p1..p4</td>
          <td>Chỉ lấy nhãn 1-tháng đã đóng <code>t &le; s-1</code>: <code>bought_1m</code>, <code>3m/6m/all_buy_months</code>, <code>months_since_last_buy</code>, <code>p1..p4</code>, <code>months_observed</code> (tháng 02 giữ <code>NaN</code>)</td>
        </tr>
        <tr class="row-peach">
          <td class="txt-orange">3. DIALOG (Hội thoại)</td>
          <td class="num txt-orange">9</td>
          <td>Mức độ &amp; độ gần tương tác tư vấn/CSKH</td>
          <td><code>has_event</code>, <code>event_count</code>, <code>days_since_last_event</code> trên 3 cửa sổ <code>30d / 90d / all</code> &amp; <code>days_since_first_event</code></td>
        </tr>
        <tr>
          <td><strong>4. GEO (Vị trí app)</strong></td>
          <td class="num"><strong>7</strong></td>
          <td>Độ phủ địa lý &amp; tần suất mở ứng dụng</td>
          <td><code>event_count</code>, <code>days_since_last</code>, <code>geohash_4_nunique</code> (30d/all), <code>top_geohash_6_share</code>, <code>geo_90d_has_event</code></td>
        </tr>
        <tr>
          <td><strong>5. Cold-start</strong></td>
          <td class="num"><strong>1</strong></td>
          <td>Tách biệt khách hoàn toàn chưa có log</td>
          <td><code>cold_start = 1</code> khi đồng thời không có sự kiện ở cả 3 nguồn TRX, GEO, DIALOG trước cutoff</td>
        </tr>
      </tbody>
    </table>
  </div>
</section>

<!-- ==================== SLIDE 6: MUC 2B - 5 BASELINES ==================== -->
<section class="slide">
  <div class="kicker k-teal">MỤC 2 &bull; HỆ THỐNG 5 BASELINES (CÙNG 50 BIẾN CHUẨN)</div>
  <div class="title">Thang Đo 5 Baselines: Từ Luật Tay (5,61&times;) Đến CatBoost GPU (8,31&times;)</div>
  <div class="subtitle">Cả 5 mô hình chạy trên đúng 50 biến vừa định nghĩa &nbsp;|&nbsp; Đánh giá trên Locked OOT 2022-11 (K = 1.003)</div>
  <div class="content">
    <div class="grid-2-bal">
      <div class="hbar-wrap">
        <div style="font-size:14.5px; font-weight:700; color:#0F172A; margin-bottom:4px;">OOT Lift@1% (K = 1.003 khách hàng)</div>
        <div class="hbar-row" style="grid-template-columns:165px 1fr 70px;">
          <div class="hbar-lbl">B1. Rule Prior+RFM</div>
          <div class="hbar-track"><div class="hbar-fill" style="width:62%; background:#94A3B8;"></div></div>
          <div class="hbar-val" style="color:#64748B;">5,61&times;</div>
        </div>
        <div class="hbar-row" style="grid-template-columns:165px 1fr 70px;">
          <div class="hbar-lbl">B2. LogReg Top-50</div>
          <div class="hbar-track"><div class="hbar-fill" style="width:74%; background:#475569;"></div></div>
          <div class="hbar-val" style="color:#334155;">6,71&times;</div>
        </div>
        <div class="hbar-row" style="grid-template-columns:165px 1fr 70px;">
          <div class="hbar-lbl">B3. LightGBM Top-50</div>
          <div class="hbar-track"><div class="hbar-fill" style="width:88%; background:#14B8A6;"></div></div>
          <div class="hbar-val c-teal">7,96&times;</div>
        </div>
        <div class="hbar-row" style="grid-template-columns:165px 1fr 70px;">
          <div class="hbar-lbl">B4. XGBoost GPU 50</div>
          <div class="hbar-track"><div class="hbar-fill" style="width:90%; background:#0F766E;"></div></div>
          <div class="hbar-val c-teal">8,09&times;</div>
        </div>
        <div class="hbar-row" style="grid-template-columns:165px 1fr 70px;">
          <div class="hbar-lbl c-orange">B5. CatBoost GPU 50</div>
          <div class="hbar-track"><div class="hbar-fill" style="width:93%; background:#EA580C;"></div></div>
          <div class="hbar-val c-orange">8,31&times;</div>
        </div>
      </div>

      <table>
        <thead>
          <tr><th>Mốc</th><th>Vai trò chính</th><th class="num">OOT PR</th><th class="num">ROC</th></tr>
        </thead>
        <tbody>
          <tr><td><strong>B1</strong></td><td>Mốc sàn quy tắc tay</td><td class="num">0,0620</td><td class="num">0,7002</td></tr>
          <tr><td><strong>B2</strong></td><td>Tuyến tính L2 chuẩn</td><td class="num">0,0755</td><td class="num">0,7613</td></tr>
          <tr><td><strong>B3</strong></td><td>GBDT phi tuyến (CPU)</td><td class="num">0,0840</td><td class="num">0,7702</td></tr>
          <tr><td><strong>B4</strong></td><td>XGBoost GPU đối chiếu</td><td class="num">0,0845</td><td class="num">0,7699</td></tr>
          <tr class="row-mint"><td class="txt-teal">B5</td><td>Baseline mạnh nhất</td><td class="num txt-teal">0,0884</td><td class="num txt-teal">0,7703</td></tr>
        </tbody>
      </table>
    </div>

    <div class="grid-2">
      <div class="card-mint" style="padding:16px 22px;">
        <span class="txt-teal">+21,8% PR-AUC</span> (B2 vs B1, p &lt; 0,001) &nbsp;&bull;&nbsp; <span class="txt-teal">+17,2% PR-AUC</span> &amp; <span class="txt-teal">+1,59&times; Lift</span> (B5 vs B2, p &lt; 0,001)
      </div>
      <div class="card-peach" style="padding:16px 22px;">
        <span class="txt-orange">B5 CatBoost Top-50:</span> Precision@1% = <strong>18,01%</strong> (~181 khách mua), LogLoss = <strong>0,0941</strong>, 7,3p/seed.
      </div>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 7: WARNING 1 - DIALOG LEAK (STANDALONE) ==================== -->
<section class="slide">
  <div class="kicker k-orange">KIỂM CHỨNG 1 / 6 &bull; CẢNH BÁO RÒ RỈ DIALOG</div>
  <div class="title">Kiểm Chứng DIALOG Có Gây Leak Không? (Lag 0 / 7 / 14 / 30 Ngày)</div>
  <div class="subtitle">Nếu DIALOG bị leak sát ngày chốt đơn, cắt lùi 7–14 ngày sẽ làm hiệu năng sập ngay về mức <code>no_dialog</code></div>
  <div class="content">
    <div class="grid-2-bal">
      <div class="card" style="padding:22px 26px;">
        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:14px;">PR-AUC &amp; Lift@1% trên Val (05..08) theo Độ trễ cắt DIALOG</div>
        <svg viewBox="0 0 540 250" width="100%" height="245">
          <line x1="305" y1="8" x2="305" y2="232" stroke="#94A3B8" stroke-dasharray="4,4" stroke-width="1.5"/>
          <text x="309" y="244" font-size="11" fill="#64748B">Mốc bỏ hẳn DIALOG (0,0751)</text>

          <text x="0" y="30" font-size="13.5" font-weight="700" fill="#0F172A">Full (Lag 0 ngày)</text>
          <rect x="145" y="14" width="290" height="26" rx="6" fill="#0E192D"/>
          <text x="444" y="32" font-size="13.5" font-weight="800" fill="#0F172A">0,0785 (4,99&times;)</text>

          <text x="0" y="76" font-size="13.5" font-weight="700" fill="#0D9488">Cắt lùi 7 ngày</text>
          <rect x="145" y="60" width="215" height="26" rx="6" fill="#0D9488"/>
          <text x="369" y="78" font-size="13.5" font-weight="800" fill="#0D9488">0,0761 (4,91&times;)</text>

          <text x="0" y="122" font-size="13.5" font-weight="700" fill="#0D9488">Cắt lùi 14 ngày</text>
          <rect x="145" y="106" width="222" height="26" rx="6" fill="#14B8A6"/>
          <text x="376" y="124" font-size="13.5" font-weight="800" fill="#0D9488">0,0763 (5,00&times;)</text>

          <text x="0" y="168" font-size="13.5" font-weight="600" fill="#475569">Cắt lùi 30 ngày</text>
          <rect x="145" y="152" width="163" height="26" rx="6" fill="#64748B"/>
          <text x="317" y="170" font-size="13.5" font-weight="700" fill="#475569">0,0751 (4,82&times;)</text>

          <text x="0" y="214" font-size="13.5" font-weight="700" fill="#EA580C">Bỏ hẳn DIALOG</text>
          <rect x="145" y="198" width="160" height="26" rx="6" fill="#EA580C"/>
          <text x="314" y="216" font-size="13.5" font-weight="800" fill="#EA580C">0,0751 (-4,4%)</text>
        </svg>
      </div>

      <div style="display:flex; flex-direction:column; gap:18px;">
        <div class="card-mint">
          <div class="kicker k-teal" style="margin-bottom:6px;">KẾT LUẬN: KHÔNG CÓ LEAK SÁT CUTOFF</div>
          <div class="hero-num c-teal">4,91–5,00&times;</div>
          <div class="hero-lbl">Lift@1% giữ nguyên khi cắt lùi 7–14 ngày</div>
          <div class="hero-sub" style="color:#1E293B;">PR-AUC vẫn cao hơn hẳn mốc <code>no_dialog</code> (0,0761–0,0763 vs 0,0751; ROC 0,723 vs 0,717).</div>
        </div>
        <div class="card-white">
          <div class="kicker k-orange" style="margin-bottom:6px;">SUY GIẢM TỰ NHIÊN THEO THỜI GIAN</div>
          <div class="hero-lbl">Tín hiệu phai mượt 0d &rarr; 14d &rarr; 30d</div>
          <div class="hero-sub" style="color:#334155;">Đúng bản chất tư vấn trước mua 1–3 tuần &rarr; <strong>An toàn giữ toàn bộ DIALOG vào Stage 2</strong>.</div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 6: WARNING 2 - LABEL OVERLAP H=3 ==================== -->
<section class="slide">
  <div class="kicker k-orange">KIỂM CHỨNG 2 / 5 &bull; TRÙNG NHÃN CỬA SỔ TRƯỢT (LABEL OVERLAP H=3)</div>
  <div class="title">Cửa Sổ Trượt Step=1 Có Làm Thổi Phồng Điểm Không?</div>
  <div class="subtitle">Đối chứng trực tiếp trên cùng tháng đánh giá <code>2022-08</code> giữa huấn luyện trượt chồng nhãn <code>[02..05]</code> vs không chồng nhãn <code>[02, 05]</code></div>
  <div class="content">
    <div class="grid-2-bal">
      <div class="card" style="padding:26px 30px;">
        <div style="font-size:15.5px; font-weight:700; color:#0F172A; margin-bottom:14px;">So sánh PR-AUC &amp; Lift@1% trên 2022-08</div>
        <svg viewBox="0 0 460 215" width="100%" height="205">
          <rect x="75" y="34" width="115" height="142" rx="8" fill="#0D9488"/>
          <text x="132" y="24" text-anchor="middle" font-size="15" font-weight="800" fill="#0D9488">0,0843 PR</text>
          <text x="132" y="110" text-anchor="middle" font-size="16" font-weight="800" fill="#FFFFFF">5,75&times;</text>
          <text x="132" y="198" text-anchor="middle" font-size="13" font-weight="700" fill="#1E293B">Rolling (02..05)</text>

          <rect x="265" y="24" width="115" height="152" rx="8" fill="#0E192D"/>
          <text x="322" y="15" text-anchor="middle" font-size="15" font-weight="800" fill="#0E192D">0,0867 PR</text>
          <text x="322" y="106" text-anchor="middle" font-size="16" font-weight="800" fill="#FFFFFF">5,89&times;</text>
          <text x="322" y="198" text-anchor="middle" font-size="13" font-weight="700" fill="#1E293B">Non-Overlap (02, 05)</text>
        </svg>
      </div>

      <div style="display:flex; flex-direction:column; gap:18px;">
        <div class="card-mint">
          <div class="kicker k-teal" style="margin-bottom:6px;">KẾT LUẬN: PASS (0% THỔI PHỒNG)</div>
          <div class="hero-num c-teal">0,0843 &le; 0,0867</div>
          <div class="hero-lbl">Rolling Step=1 không hề tạo lift ảo</div>
          <div class="hero-sub" style="color:#1E293B;">Thậm chí không chồng nhãn (<code>02, 05</code>) còn nhỉnh hơn nhẹ trên <code>2022-08</code> (ROC 0,756 vs 0,755).</div>
        </div>
        <div class="card-white">
          <div class="kicker k-orange" style="margin-bottom:6px;">Ý NGHĨA THIẾT KẾ</div>
          <div class="hero-lbl">An toàn dùng cửa sổ trượt snapshot = 1</div>
          <div class="hero-sub" style="color:#334155;">Tăng gấp đôi số mẫu huấn luyện mà không vi phạm tính độc lập của kỳ đánh giá.</div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 9: WARNINGS 4, 5, 6 (TIMEZONE, TENURE, GEO) ==================== -->
<section class="slide">
  <div class="kicker k-orange">KIỂM CHỨNG 4, 5 &amp; 6 / 6 &bull; MÚI GIỜ, TENURE CENSOR &amp; GEOHASH DRIFT</div>
  <div class="title">Mỗi Cảnh Báo Một Biểu Đồ: An Toàn Múi Giờ &amp; Cải Thiện Với <code>cap365</code></div>
  <div class="subtitle">Đối chiếu trực tiếp PR-AUC &amp; Lift@1% trên Validation (05..08) cho từng yếu tố nhiễu dữ liệu</div>
  <div class="content">
    <div class="grid-3">
      <!-- Chart 4: Timezone MSK vs UTC -->
      <div class="card" style="padding:20px 22px;">
        <div class="pill-ok">CẢNH BÁO 4 &bull; MÚI GIỜ</div>
        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:12px;">MSK (UTC+3) vs UTC (TRX RFM)</div>
        <svg viewBox="0 0 300 185" width="100%" height="180">
          <rect x="42" y="42" width="76" height="114" rx="8" fill="#0D9488"/>
          <text x="80" y="32" text-anchor="middle" font-size="13.5" font-weight="800" fill="#0D9488">0,05807</text>
          <text x="80" y="104" text-anchor="middle" font-size="13.5" font-weight="700" fill="#FFFFFF">3,20&times;</text>
          <text x="80" y="176" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1E293B">MSK (+3h)</text>

          <rect x="182" y="42" width="76" height="114" rx="8" fill="#94A3B8"/>
          <text x="220" y="32" text-anchor="middle" font-size="13.5" font-weight="800" fill="#64748B">0,05807</text>
          <text x="220" y="104" text-anchor="middle" font-size="13.5" font-weight="700" fill="#FFFFFF">3,18&times;</text>
          <text x="220" y="176" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1E293B">UTC</text>
        </svg>
        <div class="hero-sub" style="color:#334155;">Lệch PR &lt; 0,00001; không rò rỉ biên giờ.</div>
      </div>

      <!-- Chart 5: Tenure Left-Censoring -->
      <div class="card-mint" style="padding:20px 22px;">
        <div class="pill-ok">CẢNH BÁO 5 &bull; TENURE CENSOR</div>
        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:12px;">Thô vs Bỏ vs Chặn trần <code>cap365</code></div>
        <svg viewBox="0 0 300 185" width="100%" height="180">
          <rect x="18" y="54" width="72" height="102" rx="8" fill="#64748B"/>
          <text x="54" y="44" text-anchor="middle" font-size="12.5" font-weight="700" fill="#475569">0,0785</text>
          <text x="54" y="110" text-anchor="middle" font-size="12.5" font-weight="700" fill="#FFFFFF">4,99&times;</text>
          <text x="54" y="176" text-anchor="middle" font-size="12" font-weight="700" fill="#1E293B">Raw</text>

          <rect x="114" y="64" width="72" height="92" rx="8" fill="#94A3B8"/>
          <text x="150" y="54" text-anchor="middle" font-size="12.5" font-weight="700" fill="#64748B">0,0779</text>
          <text x="150" y="115" text-anchor="middle" font-size="12.5" font-weight="700" fill="#FFFFFF">4,96&times;</text>
          <text x="150" y="176" text-anchor="middle" font-size="12" font-weight="700" fill="#1E293B">Drop</text>

          <rect x="210" y="34" width="72" height="122" rx="8" fill="#0D9488"/>
          <text x="246" y="24" text-anchor="middle" font-size="13" font-weight="800" fill="#0D9488">0,0794</text>
          <text x="246" y="100" text-anchor="middle" font-size="13" font-weight="800" fill="#FFFFFF">5,09&times;</text>
          <text x="246" y="176" text-anchor="middle" font-size="12" font-weight="800" fill="#0D9488">cap365</text>
        </svg>
        <div class="hero-sub" style="color:#1E293B;"><strong>cap365 tăng điểm</strong> &rarr; đưa vào Stage 2!</div>
      </div>

      <!-- Chart 6: Geohash Drift -->
      <div class="card" style="padding:20px 22px;">
        <div class="pill-warn">CẢNH BÁO 6 &bull; GEOHASH DRIFT</div>
        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:12px;">Geohash thô vs Bỏ Geohash thô</div>
        <svg viewBox="0 0 300 185" width="100%" height="180">
          <rect x="42" y="50" width="76" height="106" rx="8" fill="#64748B"/>
          <text x="80" y="40" text-anchor="middle" font-size="13.5" font-weight="700" fill="#475569">0,0785</text>
          <text x="80" y="108" text-anchor="middle" font-size="13" font-weight="700" fill="#FFFFFF">4,99&times;</text>
          <text x="80" y="176" text-anchor="middle" font-size="12" font-weight="700" fill="#1E293B">Có GEO thô</text>

          <rect x="182" y="38" width="76" height="118" rx="8" fill="#EA580C"/>
          <text x="220" y="28" text-anchor="middle" font-size="13.5" font-weight="800" fill="#EA580C">0,0791</text>
          <text x="220" y="102" text-anchor="middle" font-size="13" font-weight="700" fill="#FFFFFF">4,97&times;</text>
          <text x="220" y="176" text-anchor="middle" font-size="12" font-weight="700" fill="#1E293B">Bỏ GEO thô</text>
        </svg>
        <div class="hero-sub" style="color:#334155;">GEO bắt đầu 12/2021 &rarr; chuyển sang Entropy &amp; Novelty.</div>
      </div>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 10: SHUFFLED SANITY CHECK (DEDICATED) ==================== -->
<section class="slide">
  <div class="kicker k-teal">KIỂM ĐỊNH CHỐNG RÒ RỈ TUYỆT ĐỐI &bull; SHUFFLED-LABEL SANITY</div>
  <div class="title">Xáo Trộn Nhãn (Shuffled Sanity): Xác Nhận 0% Rò Rỉ Dữ Liệu</div>
  <div class="subtitle">Khi xáo ngẫu nhiên nhãn, mô hình không thể học được gì: ROC-AUC rơi đúng về 0,50 và Eval Lift@1% rơi về 0,97&times;</div>
  <div class="content">
    <div class="grid-2">
      <!-- Chart A: ROC-AUC Real vs Shuffled -->
      <div class="card" style="padding:24px 26px;">
        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:14px;">ROC-AUC: Mô hình Thật vs Xáo Nhãn (Đường chuẩn = 0,50)</div>
        <svg viewBox="0 0 460 215" width="100%" height="205">
          <!-- Dashed 0.50 line (height from y=180, scale 0..1 -> 0.50 at y=95) -->
          <line x1="15" y1="95" x2="445" y2="95" stroke="#EA580C" stroke-dasharray="5,5" stroke-width="1.8"/>
          <text x="442" y="22" text-anchor="end" font-size="11.5" font-weight="700" fill="#EA580C">--- Ngẫu nhiên 0,50</text>

          <rect x="30" y="42" width="76" height="143" rx="8" fill="#0D9488"/>
          <text x="68" y="33" text-anchor="middle" font-size="14" font-weight="800" fill="#0D9488">0,774</text>
          <text x="68" y="205" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1E293B">M4 Thật</text>

          <rect x="136" y="93" width="76" height="92" rx="8" fill="#94A3B8"/>
          <text x="174" y="84" text-anchor="middle" font-size="13.5" font-weight="700" fill="#475569">0,509</text>
          <text x="174" y="205" text-anchor="middle" font-size="11.5" font-weight="600" fill="#475569">Shuf Train Val</text>

          <rect x="242" y="96" width="76" height="89" rx="8" fill="#94A3B8"/>
          <text x="280" y="87" text-anchor="middle" font-size="13.5" font-weight="700" fill="#475569">0,493</text>
          <text x="280" y="205" text-anchor="middle" font-size="11.5" font-weight="600" fill="#475569">Shuf Train OOT</text>

          <rect x="348" y="96" width="76" height="89" rx="8" fill="#64748B"/>
          <text x="386" y="87" text-anchor="middle" font-size="13.5" font-weight="700" fill="#334155">0,494</text>
          <text x="386" y="205" text-anchor="middle" font-size="11.5" font-weight="600" fill="#334155">Shuf Eval OOT</text>
        </svg>
      </div>

      <!-- Chart B: Eval Lift@1% Real vs Shuffled -->
      <div class="card-mint" style="padding:24px 26px;">
        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:14px;">OOT Lift@1%: Mô hình Thật vs Xáo Nhãn Eval (Đường chuẩn = 1,0&times;)</div>
        <svg viewBox="0 0 460 215" width="100%" height="205">
          <line x1="20" y1="166" x2="440" y2="166" stroke="#EA580C" stroke-dasharray="5,5" stroke-width="1.8"/>
          <text x="438" y="24" text-anchor="end" font-size="11.5" font-weight="700" fill="#EA580C">--- Ngẫu nhiên 1,00&times;</text>

          <rect x="85" y="26" width="110" height="159" rx="8" fill="#0D9488"/>
          <text x="140" y="17" text-anchor="middle" font-size="16" font-weight="800" fill="#0D9488">8,05&times;</text>
          <text x="140" y="110" text-anchor="middle" font-size="13" font-weight="700" fill="#FFFFFF">PR = 0,0862</text>
          <text x="140" y="205" text-anchor="middle" font-size="13" font-weight="700" fill="#0F172A">M4 CatBoost Thật</text>

          <rect x="265" y="166" width="110" height="19" rx="5" fill="#EA580C"/>
          <text x="320" y="156" text-anchor="middle" font-size="16" font-weight="800" fill="#EA580C">0,97&times;</text>
          <text x="320" y="205" text-anchor="middle" font-size="13" font-weight="700" fill="#0F172A">Shuffled Eval (PR 0,0212)</text>
        </svg>
      </div>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 11: MUC 3 - FEATURE ENGINEERING ==================== -->
<section class="slide">
  <div class="kicker k-teal">MỤC 3 &bull; FEATURE ENGINEERING TỪ PHÁT HIỆN EDA</div>
  <div class="title">5 Nhóm Biến Mới Mở Rộng (123 Biến Mới &rarr; Kho 272 Ứng Viên)</div>
  <div class="subtitle">Xây dựng song song bằng DuckDB local (11,7 phút) &nbsp;|&nbsp; 100% lọc sự kiện &lt; 00:00 MSK &nbsp;|&nbsp; PCA fit riêng trong từng fold train</div>
  <div class="content">
    <table>
      <thead>
        <tr>
          <th style="width:24%;">Nhóm Đặc trưng</th>
          <th class="num" style="width:10%;">Số biến</th>
          <th style="width:28%;">Giải quyết phát hiện EDA</th>
          <th style="width:38%;">Cách tính cốt lõi</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="txt-teal">1. DIALOG &amp; Embedding</td>
          <td class="num">~45</td>
          <td>Tín hiệu mạnh nhất (3,71&times; Lift)</td>
          <td><code>exp(-d/14|45)</code>, burst 7d/30d, gap 90d, PCA-16 trong fold, L2 drift &amp; within-std</td>
        </tr>
        <tr>
          <td class="txt-teal">2. Không gian GEO</td>
          <td class="num">~24</td>
          <td>GEO thô drift do bắt đầu 12/2021</td>
          <td>Shannon Entropy (<code>gh4/5/6</code>), Novelty 30d (<code>new_gh4/5/6</code>), mật độ <code>gh6/gh4</code></td>
        </tr>
        <tr>
          <td class="txt-orange">3. TRX Phi tuyến &amp; Đuôi</td>
          <td class="num">~30</td>
          <td>Đường cong chữ U &amp; đuôi 1,37e12</td>
          <td>4 cờ <code>trx_regime</code>, <code>log_max_minus_med</code>, <code>new_subtype/dst12</code>, <code>share_90d_of_all</code></td>
        </tr>
        <tr>
          <td class="txt-orange">4. Tuổi quan sát</td>
          <td class="num">~6</td>
          <td>Left-censoring &amp; lệch mốc nguồn</td>
          <td><code>obs_age_days_{trx,geo,dlg}</code>, <code>tenure_frac_of_obs</code>, <code>tenure_cap365</code></td>
        </tr>
        <tr class="row-peach">
          <td class="txt-orange">5. Tương tác &amp; Prior tỷ lệ</td>
          <td class="num">~18</td>
          <td>0-TRX &times; DIALOG (9,41%–11,66%)</td>
          <td><code>zero_trx &times; dialog_30d</code>, <code>source_bitmask</code>, <code>prior_buy_rate_per_obs_mo</code></td>
        </tr>
      </tbody>
    </table>
  </div>
</section>

<!-- ==================== SLIDE 12: MUC 4 - FEATURE SELECTION ==================== -->
<section class="slide">
  <div class="kicker k-teal">MỤC 4 &bull; CHỌN LỌC ĐẶC TRƯNG (VAL-ONLY SELECTION)</div>
  <div class="title">Rút Gọn 272 &rarr; 150 Biến Thuần Trên Validation (129 Bảng + 21 PCA)</div>
  <div class="subtitle">Tuyệt đối không nhìn tập OOT 2022-11 &nbsp;|&nbsp; Các biến FE mới chiếm áp đảo Top đầu bảng xếp hạng đồng thuận</div>
  <div class="content">
    <div class="step-flow">
      <div class="step-card">
        <div class="step-circle" style="background:#0E192D;">01</div>
        <div class="step-title">Lọc sơ bộ</div>
        <div class="step-desc">Loại hằng số, thiếu &gt;99,95% &amp; biến thí nghiệm.</div>
      </div>
      <div class="step-arrow">&rarr;</div>
      <div class="step-card">
        <div class="step-circle" style="background:#0D9488;">02</div>
        <div class="step-title">Consensus Rank</div>
        <div class="step-desc">XGB Gain + CB Imp + Perm &Delta;PR &amp; &Delta;Lift trên Val.</div>
      </div>
      <div class="step-arrow">&rarr;</div>
      <div class="step-card">
        <div class="step-circle" style="background:#0D9488;">03</div>
        <div class="step-title">Khử |r| &gt; 0,96</div>
        <div class="step-desc">Cắt biến trùng thông tin theo Spearman trên Val.</div>
      </div>
      <div class="step-arrow">&rarr;</div>
      <div class="step-card peach">
        <div class="step-circle" style="background:#EA580C;">04</div>
        <div class="step-title">Khóa 150 Biến</div>
        <div class="step-desc">18 biến neo + Top rank &rarr; 129 bảng + 21 emb.</div>
      </div>
    </div>

    <div class="hbar-wrap" style="padding:18px 24px; gap:10px;">
      <div style="font-size:13.5px; font-weight:700; color:#0F172A;">Top 8 Đặc trưng Dẫn đầu Điểm Đồng thuận Val (Xanh ngọc &amp; Cam = Biến mới Stage 2)</div>
      <div class="hbar-row" style="grid-template-columns:280px 1fr 65px;">
        <div class="hbar-lbl c-teal">trx_share_90d_of_all</div>
        <div class="hbar-track" style="height:16px;"><div class="hbar-fill" style="width:94.3%; background:#0D9488;"></div></div>
        <div class="hbar-val c-teal">0,943</div>
      </div>
      <div class="hbar-row" style="grid-template-columns:280px 1fr 65px;">
        <div class="hbar-lbl">trx_all_days_since_last_event</div>
        <div class="hbar-track" style="height:16px;"><div class="hbar-fill" style="width:93.8%; background:#0E192D;"></div></div>
        <div class="hbar-val">0,938</div>
      </div>
      <div class="hbar-row" style="grid-template-columns:280px 1fr 65px;">
        <div class="hbar-lbl c-teal">trx_all_days_since_first_event_cap365</div>
        <div class="hbar-track" style="height:16px;"><div class="hbar-fill" style="width:91.1%; background:#0D9488;"></div></div>
        <div class="hbar-val c-teal">0,911</div>
      </div>
      <div class="hbar-row" style="grid-template-columns:280px 1fr 65px;">
        <div class="hbar-lbl c-teal">trx_30d_new_dst12_count</div>
        <div class="hbar-track" style="height:16px;"><div class="hbar-fill" style="width:90.8%; background:#0D9488;"></div></div>
        <div class="hbar-val c-teal">0,908</div>
      </div>
      <div class="hbar-row" style="grid-template-columns:280px 1fr 65px;">
        <div class="hbar-lbl c-teal">trx_tenure_frac_of_obs</div>
        <div class="hbar-track" style="height:16px;"><div class="hbar-fill" style="width:88.5%; background:#0D9488;"></div></div>
        <div class="hbar-val c-teal">0,885</div>
      </div>
      <div class="hbar-row" style="grid-template-columns:280px 1fr 65px;">
        <div class="hbar-lbl c-orange">dialog_recency_exp_14d</div>
        <div class="hbar-track" style="height:16px;"><div class="hbar-fill" style="width:86.8%; background:#EA580C;"></div></div>
        <div class="hbar-val c-orange">0,868</div>
      </div>
      <div class="hbar-row" style="grid-template-columns:280px 1fr 65px;">
        <div class="hbar-lbl c-orange">prior_buy_rate_per_obs_mo</div>
        <div class="hbar-track" style="height:16px;"><div class="hbar-fill" style="width:86.7%; background:#EA580C;"></div></div>
        <div class="hbar-val c-orange">0,867</div>
      </div>
      <div class="hbar-row" style="grid-template-columns:280px 1fr 65px;">
        <div class="hbar-lbl c-orange">dlg_pca_90d_7 (DIALOG Embedding)</div>
        <div class="hbar-track" style="height:16px;"><div class="hbar-fill" style="width:86.6%; background:#F97316;"></div></div>
        <div class="hbar-val c-orange">0,866</div>
      </div>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 13: MUC 5 - EXPERIMENTS & FINAL SINGLE MODEL ==================== -->
<section class="slide">
  <div class="kicker k-teal">MỤC 5 &bull; THỬ NGHIỆM, BÀI HỌC &amp; KIẾN TRÚC FINAL MODEL</div>
  <div class="title">Tối Ưu GPU 150 Biến, Ablation <code>prior_*</code> &amp; Chốt CatBoost GPU 150</div>
  <div class="subtitle">Quét 12 cấu hình GPU (826s) thuần trên Validation &nbsp;|&nbsp; Mọi mô hình GPU chạy dưới 11 phút/seed</div>
  <div class="content">
    <div class="grid-3">
      <div class="card-white">
        <div class="kicker k-teal" style="margin-bottom:6px;">FE-150 TRÊN VALIDATION</div>
        <div class="hero-num c-teal">+3,2–5,8%</div>
        <div class="hero-sub">Cả 4 họ mô hình đều tăng PR-AUC trên Val; M4 CatBoost dẫn đầu ROC (<strong>0,7363</strong>).</div>
      </div>
      <div class="card-peach">
        <div class="kicker k-orange" style="margin-bottom:6px;">ABLATION BỎ PRIOR_* (M6)</div>
        <div class="hero-num c-orange">-1,23&times; Lift</div>
        <div class="hero-sub">OOT Lift@1% tụt 7,20&times; &rarr; 5,98&times; (p &lt; 0,001); PR-AUC giảm 9,8%.</div>
      </div>
      <div class="card-white">
        <div class="kicker k-teal" style="margin-bottom:6px;">TỐC ĐỘ COLAB T4 GPU</div>
        <div class="hero-num c-dark">9,3 – 10,6p</div>
        <div class="hero-sub">XGB 9,3p/s &bull; CatBoost 10,6p/s &bull; LogReg float32 2,6p (hết OOM).</div>
      </div>
    </div>

    <!-- Single Final Model Architecture Bar -->
    <div style="background:#0E192D; border-radius:16px; padding:22px 28px; color:#FFFFFF;">
      <div class="kicker k-mint" style="margin-bottom:14px;">KIẾN TRÚC FINAL MODEL ĐƠN &bull; M4_CATBOOST_GPU_150_TUNED (KHÓA TRÊN VALIDATION)</div>
      <div style="display:grid; grid-template-columns:1fr 30px 1.55fr 30px 1fr; align-items:center; gap:10px;">
        <div style="background:rgba(255,255,255,0.08); border-radius:12px; padding:14px; text-align:center;">
          <div style="font-weight:700; color:#2DD4BF; font-size:14px; margin-bottom:4px;">150 Đặc trưng</div>
          <div style="font-size:12.5px; color:#CBD5E1;">129 bảng + 21 PCA</div>
        </div>
        <div style="text-align:center; color:#2DD4BF; font-size:20px; font-weight:800;">&rarr;</div>
        <div style="background:rgba(234,88,12,0.22); border:1.5px solid #FB923C; border-radius:12px; padding:14px 18px; text-align:center;">
          <div style="font-size:16px; font-weight:800; color:#FB923C; margin-bottom:4px;">CatBoost GPU Oblivious Trees (3 Seeds)</div>
          <div style="font-size:12.5px; color:#F8FAFC;">depth=6 &bull; l2_leaf_reg=30 &bull; lr=0,03 &bull; spw=3,0 &bull; 10,6 phút/seed</div>
        </div>
        <div style="text-align:center; color:#2DD4BF; font-size:20px; font-weight:800;">&rarr;</div>
        <div style="background:rgba(13,148,136,0.22); border:1px solid #2DD4BF; border-radius:12px; padding:14px; text-align:center;">
          <div style="font-weight:700; color:#FFFFFF; font-size:14px; margin-bottom:4px;">+ Isotonic Calib</div>
          <div style="font-size:12.5px; color:#CCFBF1;">L@1% 8,05&times; &bull; R@5% 25,1%</div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 14: VALIDATION ROLLING CHART ==================== -->
<section class="slide">
  <div class="kicker k-teal">ĐỐI CHỨNG TRÊN VALIDATION (400.896 KHÁCH–THÁNG)</div>
  <div class="title">FE-150 Cải Thiện Ổn Định Qua Cả 4 Tháng Trượt Validation</div>
  <div class="subtitle">Khi lịch sử huấn luyện tích lũy từ 1 lên 4 tháng, M4 CatBoost GPU 150 dẫn đầu nhất quán về PR-AUC &amp; ROC-AUC</div>
  <div class="content">
    <div class="img-clean" style="height:4.9in;">
      <img src="__FIG1__" alt="Validation Rolling Curve" />
    </div>
  </div>
</section>

<!-- ==================== SLIDE 15: MUC 6 - LOCKED OOT TEST ==================== -->
<section class="slide">
  <div class="kicker k-teal">MỤC 6 &bull; KẾT QUẢ CUỐI CÙNG (LOCKED OOT TEST 2022-11)</div>
  <div class="title">Hiệu Năng Toàn Bộ Mô Hình Đơn Trên Tập OOT Khóa (95% CI)</div>
  <div class="subtitle">100.224 khách hàng &bull; 2.173 ca mua (2,168%) &nbsp;|&nbsp; M4 CatBoost 150 thắng mọi mô hình đơn 150 biến &amp; vượt B5 về ROC-AUC (p &lt; 0,001)</div>
  <div class="content" style="gap:16px;">
    <div class="img-clean" style="height:2.95in;">
      <img src="__FIG2__" alt="OOT Single Model Comparison" />
    </div>
    <table>
      <thead>
        <tr>
          <th>Mô hình tiêu biểu</th>
          <th class="num">Biến</th>
          <th class="num">OOT PR-AUC [95% CI]</th>
          <th class="num">OOT Lift@1% [95% CI]</th>
          <th class="num">Lift@0,5%</th>
          <th class="num">Lift@5% (R@5%)</th>
          <th class="num">ROC-AUC</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>B1_rule_prior_rfm &rarr; B2_logreg_top50</td><td class="num">50</td>
          <td class="num">0,0620 &rarr; 0,0755</td><td class="num">5,61&times; &rarr; 6,71&times;</td>
          <td class="num">7,44&times;</td><td class="num">4,71&times; (23,6%)</td><td class="num">0,7613</td>
        </tr>
        <tr>
          <td>M2_lgbm_150_tuned &nbsp;/&nbsp; M3_xgb_gpu_150</td><td class="num">150</td>
          <td class="num">0,0839 / 0,0801</td><td class="num">8,00&times; / 7,20&times;</td>
          <td class="num">8,82&times; / 8,42&times;</td><td class="num">4,86&times; / 4,80&times;</td><td class="num">0,7747 / 0,7720</td>
        </tr>
        <tr class="row-mint">
          <td class="txt-teal">M4_catboost_gpu_150_tuned (Final Model)</td><td class="num">150</td>
          <td class="num txt-teal">0,0862 [0,079; 0,095]</td><td class="num txt-teal">8,05&times; [7,16; 9,11]</td>
          <td class="num txt-teal">9,62&times;</td><td class="num txt-teal">5,02&times; (25,1%)</td><td class="num txt-teal">0,7744</td>
        </tr>
        <tr class="row-peach">
          <td class="txt-orange">B5_catboost_gpu_top50 (Challenger gọn)</td><td class="num">50</td>
          <td class="num txt-orange">0,0884 [0,081; 0,098]</td><td class="num txt-orange">8,31&times; [7,40; 9,41]</td>
          <td class="num">9,56&times;</td><td class="num">4,94&times; (24,7%)</td><td class="num">0,7703</td>
        </tr>
      </tbody>
    </table>
  </div>
</section>

<!-- ==================== SLIDE 16: PREDICTED PROBABILITY VS ACTUAL (CALIBRATION) ==================== -->
<section class="slide">
  <div class="kicker k-teal">ĐÁNH GIÁ ĐỘ CHUẨN XÁC SUẤT &bull; CALIBRATION TRÊN OOT (2022-11)</div>
  <div class="title">Xác Suất Dự Đoán So Với Tỷ Lệ Mua Thực Tế (Nền 2,17%)</div>
  <div class="subtitle">Tỷ lệ mua thực tế tăng đơn điệu mượt theo bậc điểm M4 &nbsp;|&nbsp; Xác suất thô M4 bị đẩy cao do <code>spw=3,0</code> và cần hiệu chuẩn khi tính ROI</div>
  <div class="content">
    <div class="grid-2">
      <!-- Chart Left: Actual Conversion Rate by Score Tier (M4) -->
      <div class="card" style="padding:22px 26px;">
        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:12px;">Tỷ lệ mua thực tế theo bậc điểm M4_catboost_150</div>
        <svg viewBox="0 0 460 220" width="100%" height="210">
          <!-- Baseline 2.17% dashed line (bottom=185, scale 22% -> 155px => 2.17% = 15px => y=170) -->
          <line x1="18" y1="170" x2="442" y2="170" stroke="#EA580C" stroke-dasharray="5,5" stroke-width="1.6"/>
          <text x="440" y="24" text-anchor="end" font-size="11.5" font-weight="700" fill="#EA580C">--- Nền OOT 2,17%</text>

          <!-- Top 0.5%: 20.85% -> h=147, y=38 -->
          <rect x="30" y="38" width="80" height="147" rx="8" fill="#0D9488"/>
          <text x="70" y="29" text-anchor="middle" font-size="14" font-weight="800" fill="#0D9488">20,85%</text>
          <text x="70" y="114" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">9,62&times;</text>
          <text x="70" y="205" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1E293B">Top 0,5%</text>

          <!-- Top 0.5-1%: 14.04% -> h=99, y=86 -->
          <rect x="136" y="86" width="80" height="99" rx="8" fill="#14B8A6"/>
          <text x="176" y="77" text-anchor="middle" font-size="14" font-weight="800" fill="#0D9488">14,04%</text>
          <text x="176" y="140" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">6,47&times;</text>
          <text x="176" y="205" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1E293B">Top 0,5–1%</text>

          <!-- Top 1-5%: 9.23% -> h=65, y=120 -->
          <rect x="242" y="120" width="80" height="65" rx="8" fill="#0E192D"/>
          <text x="282" y="111" text-anchor="middle" font-size="14" font-weight="800" fill="#0E192D">9,23%</text>
          <text x="282" y="156" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">4,26&times;</text>
          <text x="282" y="205" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1E293B">Top 1–5%</text>

          <!-- Bottom 95%: 1.71% -> h=12, y=173 -->
          <rect x="348" y="173" width="80" height="12" rx="4" fill="#94A3B8"/>
          <text x="388" y="150" text-anchor="middle" font-size="13.5" font-weight="700" fill="#64748B">1,71%</text>
          <text x="388" y="205" text-anchor="middle" font-size="11.5" font-weight="700" fill="#64748B">95% còn lại</text>
        </svg>
      </div>

      <!-- Chart Right: Calibration Quality (OOT LogLoss & Brier) -->
      <div class="card" style="padding:22px 26px;">
        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:12px;">Độ chuẩn xác suất OOT: LogLoss (thấp hơn = chuẩn hơn)</div>
        <svg viewBox="0 0 460 220" width="100%" height="210">
          <!-- Base LogLoss 0.1045 line: x_start=150, scale 0..0.14 -> 260px => 0.1045 -> 194px => x=344 -->
          <line x1="344" y1="8" x2="344" y2="196" stroke="#64748B" stroke-dasharray="4,4" stroke-width="1.5"/>
          <text x="344" y="212" text-anchor="middle" font-size="11" font-weight="700" fill="#64748B">Nền Base Rate (0,1045)</text>

          <text x="0" y="34" font-size="12.5" font-weight="700" fill="#0D9488">M2 LightGBM (spw=1)</text>
          <rect x="150" y="18" width="174" height="24" rx="6" fill="#0D9488"/>
          <text x="316" y="35" text-anchor="end" font-size="12" font-weight="800" fill="#FFFFFF">0,0939</text>

          <text x="0" y="76" font-size="12.5" font-weight="700" fill="#0F172A">B5 CatBoost 50 (spw=1)</text>
          <rect x="150" y="60" width="175" height="24" rx="6" fill="#0E192D"/>
          <text x="317" y="77" text-anchor="end" font-size="12" font-weight="800" fill="#FFFFFF">0,0941</text>

          <text x="0" y="118" font-size="12.5" font-weight="600" fill="#334155">M3 XGBoost (spw=1)</text>
          <rect x="150" y="102" width="175" height="24" rx="6" fill="#475569"/>
          <text x="317" y="119" text-anchor="end" font-size="12" font-weight="700" fill="#FFFFFF">0,0941</text>

          <text x="0" y="160" font-size="12.5" font-weight="700" fill="#EA580C">M4 CatBoost (spw=3)</text>
          <rect x="150" y="144" width="224" height="24" rx="6" fill="#EA580C"/>
          <text x="382" y="161" font-size="12.5" font-weight="800" fill="#EA580C">0,1208</text>
        </svg>
      </div>
    </div>

    <div class="banner-note">
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none"><path d="M12 2L4 5v6.09c0 5.05 3.41 9.76 8 10.91 4.59-1.15 8-5.86 8-10.91V5l-8-3z" fill="#0D9488"/><path d="M10 14.2l-2.2-2.2-1.4 1.4L10 17l7.6-7.6-1.4-1.4L10 14.2z" fill="#FFFFFF"/></svg>
      <div><strong>Hiệu chuẩn khi Go-Live:</strong> <code>spw=3,0</code> chỉ đẩy cao xác suất thô (~3&times;), không làm đổi thứ hạng; chỉ cần gắn <strong>Isotonic Calibration</strong> trên Val (hoặc dịch logit <code>&minus; ln(3)</code>) để đưa LogLoss về ~<code>0,094</code> mà giữ nguyên 100% <code>Lift@0,5% = 9,62&times;</code>.</div>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 15: SHAP EXPLAINABILITY ==================== -->
<section class="slide">
  <div class="kicker k-teal">GIẢI THÍCH MÔ HÌNH &bull; TREESHAP TRÊN LOCKED OOT (2022-11)</div>
  <div class="title">SHAP: Yếu Tố Nào Liên Quan Mạnh Nhất Tới Khả Năng Mua?</div>
  <div class="subtitle">Giá trị TreeSHAP thật của <code>M4_catboost_gpu_150_tuned</code> trên 20.328 khách hàng OOT (fold 0) &nbsp;|&nbsp; Phân rõ chiều tác động tăng / giảm</div>
  <div class="content">
    <div style="display:grid; grid-template-columns:1.35fr 0.65fr; gap:24px; align-items:center;">
      <div class="img-clean" style="height:4.55in;">
        <img src="__FIG3__" alt="M4 CatBoost TreeSHAP OOT" />
      </div>
      <div style="display:flex; flex-direction:column; gap:16px;">
        <div class="card-mint" style="padding:20px 22px;">
          <div class="kicker k-teal" style="margin-bottom:6px;">(+) TĂNG MẠNH KHẢ NĂNG MUA</div>
          <div class="hero-lbl" style="margin-bottom:6px;">Mở rộng &amp; Đa dạng hóa hành vi</div>
          <div class="hero-sub" style="color:#1E293B;">
            &bull; Dùng nhiều loại GD phụ &amp; phát sinh loại GD mới 30d.<br/>
            &bull; Mở rộng đích GD 90d (<code>dst11/12</code>) &amp; tần suất GD 90d.<br/>
            &bull; Di chuyển rộng (<code>GEO entropy</code>) &amp; có nhiều tháng DIALOG.
          </div>
        </div>
        <div class="card-peach" style="padding:20px 22px;">
          <div class="kicker k-orange" style="margin-bottom:6px;">(&minus;) GIẢM KHẢ NĂNG MUA</div>
          <div class="hero-lbl" style="margin-bottom:6px;">Ngủ đông &amp; Hành vi đơn điệu</div>
          <div class="hero-sub" style="color:#1E293B;">
            &bull; Lâu không giao dịch (<code>days_since_last</code>, r = &minus;0,98).<br/>
            &bull; Chỉ lặp đi lặp lại 1 loại GD chính (<code>top_share</code> cao).<br/>
            &bull; Khách cũ lâu năm ít mở mới hơn nhóm mới kích hoạt.
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 16: IMPACT & DEPLOYMENT ==================== -->
<section class="slide">
  <div class="kicker k-teal">TÁC ĐỘNG KINH DOANH &amp; KHUYẾN NGHỊ TRIỂN KHAI</div>
  <div class="title">Hiệu Quả Theo Ngân Sách Tiếp Cận: <code>M4_catboost_gpu_150_tuned</code></div>
  <div class="subtitle">Độ ổn định cực cao qua 3 seeds (seed SD Lift@1% chỉ &plusmn;0,08&times;, ROC-AUC &plusmn;0,0003) &nbsp;|&nbsp; Vận hành đơn giản 1 mô hình duy nhất</div>
  <div class="content">
    <div class="grid-3">
      <div class="card-mint">
        <div class="kicker k-teal" style="margin-bottom:6px;">TOP 0,5% (502 KHÁCH)</div>
        <div class="hero-num c-teal">9,62&times;</div>
        <div class="hero-lbl">~105 người mua (20,9% chốt)</div>
        <div class="hero-sub" style="color:#1E293B;">Gấp 9,6 lần chọn ngẫu nhiên (2,17%).</div>
      </div>
      <div class="card-white">
        <div class="kicker k-orange" style="margin-bottom:6px;">TOP 1% (1.003 KHÁCH)</div>
        <div class="hero-num c-orange">8,05&times;</div>
        <div class="hero-lbl">~175 người mua (17,5% chốt)</div>
        <div class="hero-sub">Vượt quy tắc tay B1 (+53 khách mua/đợt).</div>
      </div>
      <div class="card-white">
        <div class="kicker k-teal" style="margin-bottom:6px;">TOP 5% (5.012 KHÁCH)</div>
        <div class="hero-num c-dark">25,1%</div>
        <div class="hero-lbl">~545 người mua (Lift 5,02&times;)</div>
        <div class="hero-sub">Phủ 1/4 tổng người mua chỉ với 5% chi phí.</div>
      </div>
    </div>

    <div class="banner-note">
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" fill="#EA580C"/><path d="M12 7v6m0 4h.01" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round"/></svg>
      <div><strong>Phương án gọn nhẹ thay thế:</strong> Nếu cần tối giản pipeline không dùng PCA embedding &amp; cần xác suất chuẩn sẵn (<code>LogLoss 0,0941</code>) cho telesales hẹp Top-1%, chọn <strong><code>B5_catboost_gpu_top50</code></strong> (8,31&times; Lift@1%).</div>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 17: DARK CLOSING ==================== -->
<section class="slide slide-dark-center">
  <div class="kicker k-mint">SẴN SÀNG TRIỂN KHAI SẢN XUẤT &bull; PROPENSITY H=3</div>
  <div class="cover-title" style="font-size:35px; max-width:10.6in; margin-bottom:0;">
    Chốt Final Model Đơn: M4_catboost_gpu_150_tuned<br/>Lift@0,5% 9,62&times; &bull; Lift@1% 8,05&times; &bull; ROC-AUC 0,7744
  </div>
  <div class="bar-orange"></div>
  <div class="cover-sub" style="margin-bottom:0; font-size:17px;">
    3 điều kiện Go-Live: Hiệu chuẩn Isotonic trên Val &bull; Xác minh SLA nhãn prior_* trước ngày 01 &bull; Duy trì Embargo 2 tháng khi retrain định kỳ
  </div>
</section>

</body>
</html>
"""
    return html.replace("__FIG1__", fig1_uri).replace("__FIG2__", fig2_uri).replace("__FIG3__", fig3_uri)


def main() -> None:
    p1, p2, p3 = _render_no_m5_charts()
    print(f"[Charts] Generated {p1.name}, {p2.name} & {p3.name}")

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(_b64_path(p1), _b64_path(p2), _b64_path(p3)), encoding="utf-8")
    print(f"[HTML] Wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)")

    edge = next((p for p in EDGE_CANDIDATES if p.exists()), None)
    if edge is None:
        raise FileNotFoundError("Microsoft Edge / Chrome not found for headless PDF export.")

    cmd = [
        str(edge),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUT_PDF}",
        OUT_HTML.resolve().as_uri(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0 or not OUT_PDF.exists():
        raise RuntimeError(f"Edge PDF export failed (code={res.returncode}): {res.stderr}")
    kb = OUT_PDF.stat().st_size / 1024.0
    print(f"[PDF] Generated {OUT_PDF} ({kb:.1f} KB)")


if __name__ == "__main__":
    main()
