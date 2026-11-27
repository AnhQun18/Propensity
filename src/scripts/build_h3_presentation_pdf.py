"""Generate a 12-slide executive widescreen (16:9) PDF presentation for H=3 Propensity."""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "outputs" / "h3_propensity" / "figures"
OUT_HTML = REPO_ROOT / "outputs" / "h3_propensity" / "H3_PROPENSITY_SLIDES.html"
OUT_PDF = REPO_ROOT / "H3_PROPENSITY_PRESENTATION.pdf"

EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]


def _b64_img(name: str) -> str:
    p = FIG_DIR / name
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_html() -> str:
    fig1 = _b64_img("01_val_rolling_curve_baselines_vs_fe150.png")
    fig2 = _b64_img("02_oot_test_2022_11_model_comparison_ci.png")
    fig3 = _b64_img("03_warnings_verification_panel.png")
    fig4 = _b64_img("04_top_eda_features_importance.png")

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<title>Propensity H=3 — Bản Thuyết Trình Dự Án</title>
<style>
  @page {{
    size: 13.333in 7.5in;
    margin: 0;
  }}
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  body {{
    font-family: "Segoe UI", Arial, sans-serif;
    color: #1E293B;
    background: #E2E8F0;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}
  .slide {{
    width: 13.333in;
    height: 7.5in;
    background: #F8FAFC;
    position: relative;
    overflow: hidden;
    page-break-after: always;
    break-after: page;
    display: flex;
    flex-direction: column;
  }}
  .slide:last-child {{
    page-break-after: auto;
    break-after: auto;
  }}
  /* Cover Slide */
  .slide-cover {{
    background: linear-gradient(135deg, #0B1F3A 0%, #133B5C 55%, #0E6BA8 100%);
    color: #FFFFFF;
    padding: 0.65in 0.8in;
    justify-content: space-between;
  }}
  .cover-badge {{
    display: inline-block;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.28);
    color: #7DD3FC;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 20px;
    margin-bottom: 18px;
  }}
  .cover-title {{
    font-size: 40px;
    font-weight: 800;
    line-height: 1.18;
    margin-bottom: 14px;
    max-width: 10.8in;
  }}
  .cover-sub {{
    font-size: 18px;
    color: #CBD5E1;
    line-height: 1.45;
    max-width: 10.2in;
  }}
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin-top: 28px;
  }}
  .kpi-card {{
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 12px;
    padding: 16px 14px;
  }}
  .kpi-val {{
    font-size: 28px;
    font-weight: 800;
    color: #38BDF8;
    line-height: 1.15;
  }}
  .kpi-val.green {{ color: #4ADE80; }}
  .kpi-val.amber {{ color: #FBBF24; }}
  .kpi-lbl {{
    font-size: 12px;
    color: #E2E8F0;
    margin-top: 6px;
    line-height: 1.3;
  }}
  .cover-footer {{
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #94A3B8;
    border-top: 1px solid rgba(255,255,255,0.18);
    padding-top: 14px;
  }}

  /* Content Slides Header & Footer */
  .hdr {{
    background: #0F2744;
    color: #FFFFFF;
    padding: 16px 36px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 4px solid #0EA5E9;
  }}
  .hdr-left {{
    display: flex;
    flex-direction: column;
  }}
  .hdr-kicker {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 2px;
  }}
  .hdr-title {{
    font-size: 22px;
    font-weight: 700;
  }}
  .hdr-pill {{
    background: rgba(56,189,248,0.18);
    border: 1px solid rgba(56,189,248,0.45);
    color: #E0F2FE;
    font-size: 12px;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 16px;
  }}
  .body {{
    flex: 1;
    padding: 20px 36px 14px 36px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    overflow: hidden;
  }}
  .ftr {{
    height: 30px;
    padding: 0 36px;
    background: #EEF2F6;
    border-top: 1px solid #CBD5E1;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    color: #64748B;
  }}

  /* Layout Helpers */
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  .grid-2-wide {{ display: grid; grid-template-columns: 1.25fr 0.75fr; gap: 18px; }}
  .grid-2-img {{ display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 18px; align-items: stretch; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
  .grid-6 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}

  .card {{
    background: #FFFFFF;
    border: 1px solid #DCE4EE;
    border-radius: 10px;
    padding: 13px 16px;
    box-shadow: 0 1px 3px rgba(15,39,68,0.05);
  }}
  .card-accent {{
    border-left: 4px solid #0EA5E9;
  }}
  .card-green {{
    border-left: 4px solid #16A34A;
    background: #F0FDF4;
    border-color: #BBF7D0;
  }}
  .card-amber {{
    border-left: 4px solid #D97706;
    background: #FFFBEB;
    border-color: #FDE68A;
  }}
  .card-navy {{
    background: #0F2744;
    color: #FFFFFF;
    border-color: #0F2744;
  }}
  .card-title {{
    font-size: 14px;
    font-weight: 700;
    color: #0F2744;
    margin-bottom: 6px;
  }}
  .card-navy .card-title {{ color: #38BDF8; }}
  .card p, .card li {{
    font-size: 12.2px;
    line-height: 1.42;
    color: #334155;
  }}
  .card-navy p, .card-navy li {{ color: #E2E8F0; }}
  ul {{ padding-left: 17px; }}
  li {{ margin-bottom: 4px; }}

  /* Tables */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 11.8px;
    background: #FFFFFF;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #CBD5E1;
  }}
  th {{
    background: #0F2744;
    color: #FFFFFF;
    font-weight: 600;
    text-align: left;
    padding: 7px 9px;
    font-size: 11.4px;
  }}
  th.num, td.num {{ text-align: center; }}
  td {{
    padding: 6.5px 9px;
    border-bottom: 1px solid #E2E8F0;
    color: #1E293B;
    vertical-align: middle;
  }}
  tr:nth-child(even) td {{ background: #F8FAFC; }}
  tr.hl td {{
    background: #E0F2FE !important;
    font-weight: 700;
    color: #0C4A6E;
  }}
  tr.hl-green td {{
    background: #DCFCE7 !important;
    font-weight: 700;
    color: #14532D;
  }}
  tr.hl-amber td {{
    background: #FEF3C7 !important;
    color: #78350F;
  }}
  code {{
    font-family: Consolas, monospace;
    font-size: 11px;
    background: #EEF2F6;
    padding: 1px 4px;
    border-radius: 4px;
    color: #0F2744;
  }}
  .card-navy code, .hdr code, .slide-cover code, .arch-box code {{
    background: rgba(255,255,255,0.16);
    color: #7DD3FC;
  }}
  .tag {{
    display: inline-block;
    font-size: 10.5px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 10px;
  }}
  .tag-ok {{ background: #DCFCE7; color: #166534; }}
  .tag-warn {{ background: #FEF3C7; color: #92400E; }}
  .tag-info {{ background: #E0F2FE; color: #075985; }}

  /* Workflow Steps */
  .flow-grid {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px;
  }}
  .flow-step {{
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-top: 5px solid #0EA5E9;
    border-radius: 10px;
    padding: 12px 11px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .flow-num {{
    font-size: 11px;
    font-weight: 800;
    color: #0284C7;
    text-transform: uppercase;
  }}
  .flow-name {{
    font-size: 13.5px;
    font-weight: 700;
    color: #0F2744;
    line-height: 1.25;
  }}
  .flow-desc {{
    font-size: 11.4px;
    color: #475569;
    line-height: 1.38;
  }}

  /* Timeline Bar */
  .timeline {{
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 4px;
    margin-top: 6px;
  }}
  .tl-cell {{
    border-radius: 6px;
    padding: 8px 4px;
    text-align: center;
    font-size: 11px;
    font-weight: 700;
  }}
  .tl-train {{ background: #DBEAFE; color: #1E3A8A; border: 1px solid #93C5FD; }}
  .tl-val {{ background: #CFFAFE; color: #155E75; border: 1px solid #67E8F9; }}
  .tl-emb {{ background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }}
  .tl-oot {{ background: #DCFCE7; color: #14532D; border: 2px solid #16A34A; }}

  /* Image frame */
  .img-box {{
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 10px;
    padding: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }}
  .img-box img {{
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
  }}

  /* Architecture Diagram */
  .arch-box {{
    background: #0F2744;
    color: #FFFFFF;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 12px;
  }}
  .arch-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-top: 8px;
  }}
  .arch-node {{
    flex: 1;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(125,211,252,0.4);
    border-radius: 8px;
    padding: 10px;
    text-align: center;
  }}
  .arch-node > strong.node-title {{ color: #38BDF8; display: block; margin-bottom: 4px; font-size: 12.5px; }}
  .arch-node strong {{ color: #7DD3FC; }}
  .arch-arrow {{ color: #38BDF8; font-size: 20px; font-weight: 800; }}
</style>
</head>
<body>

<!-- ==================== SLIDE 1: COVER ==================== -->
<section class="slide slide-cover">
  <div>
    <div class="cover-badge">Bank Propensity Modeling &bull; End-to-End Redesign</div>
    <div class="cover-title">Dự Đoán Xu Hướng Mua Sản Phẩm Ngân Hàng<br/>Horizon = 3 Tháng &bull; Rolling Snapshot = 1</div>
    <div class="cover-sub">
      Tái thiết kế toàn diện từ EDA chuyên sâu, kiến trúc Zero-Leak OOT (Embargo 2 tháng), hệ thống 5 Baselines &amp; kiểm chứng 6 cảnh báo dự án, Feature Engineering đa nguồn (GEO &bull; DIALOG &bull; TRX &bull; Prior), chọn lọc 150 đặc trưng trên Validation đến chốt Final Model trên GPU T4.
    </div>
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-val">100.224</div>
        <div class="kpi-lbl">Khách hàng duy nhất<br/>(105,3 triệu log đa nguồn)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val green">0.7756</div>
        <div class="kpi-lbl">OOT ROC-AUC cao nhất<br/>(<code>M5_final_blend_150</code>)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val green">9.62x</div>
        <div class="kpi-lbl">OOT Lift@0.5% (502 KH)<br/>Tỷ lệ mua ~20.9% vs 2.17%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val amber">8.31x / 8.05x</div>
        <div class="kpi-lbl">OOT Lift@1% đơn tốt nhất<br/>(<code>B5_top50</code> / <code>M4_150</code>)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">0% Leak</div>
        <div class="kpi-lbl">4/4 Unit Test PASS<br/>Shuffled Eval Lift = 0.97x</div>
      </div>
    </div>
  </div>
  <div class="cover-footer">
    <span>Phạm vi: 12 tháng nhãn (02/2022 – 01/2023) &bull; Múi giờ Moscow (UTC+3) &bull; 5-Fold Out-of-Client</span>
    <span>Hoàn thiện &amp; xác minh: 27/11/2026 &bull; Branch <code>main</code></span>
  </div>
</section>

<!-- ==================== SLIDE 2: WORKFLOW ==================== -->
<section class="slide">
  <div class="hdr">
    <div class="hdr-left">
      <div class="hdr-kicker">Tổng Quan Quy Trình Thực Hiện</div>
      <div class="hdr-title">Lộ Trình 6 Bước Mạch Lạc — Từ Dữ Liệu Thô Đến Final Model Chống Leak</div>
    </div>
    <div class="hdr-pill">Local DuckDB + Colab GPU T4</div>
  </div>
  <div class="body">
    <div class="flow-grid">
      <div class="flow-step">
        <div class="flow-num">Bước 1 &bull; EDA</div>
        <div class="flow-name">Khám phá đa nguồn &amp; Sổ Insight</div>
        <div class="flow-desc">Quét 38.8M TRX, 66.3M GEO, 217K DIALOG (15/15 đối soát đạt). Phát hiện DIALOG lift 3.71x, chữ U TRX, tương tác 0-TRX &times; DIALOG 9.41%.</div>
      </div>
      <div class="flow-step" style="border-top-color:#0284C7;">
        <div class="flow-num">Bước 2 &bull; Framing</div>
        <div class="flow-name">Bài toán H=3 &amp; Khóa OOT Test</div>
        <div class="flow-desc">Cửa sổ trượt snapshot=1, 5-fold OOC, toàn lịch sử &lt; cutoff MSK. Cách ly 2 tháng embargo (09–10), khóa cứng OOT 2022-11.</div>
      </div>
      <div class="flow-step" style="border-top-color:#2563EB;">
        <div class="flow-num">Bước 3 &bull; Stage 1</div>
        <div class="flow-name">5 Baselines &amp; Audit 6 Cảnh Báo</div>
        <div class="flow-desc">Chạy 5 baseline trên 50 biến có <code>prior_*</code> (Rule &rarr; LogReg &rarr; LGBM &rarr; XGB &rarr; CatBoost). Kiểm chứng định lượng 6/6 cảnh báo leak/drift.</div>
      </div>
      <div class="flow-step" style="border-top-color:#7C3AED;">
        <div class="flow-num">Bước 4 &bull; FE</div>
        <div class="flow-name">Feature Engineering từ EDA</div>
        <div class="flow-desc">Xây 123 biến mới/mở rộng local (11.7 phút): DIALOG burst/PCA/drift, GEO entropy/novelty, TRX chữ U, obs-age, tương tác đa nguồn &rarr; 272 biến.</div>
      </div>
      <div class="flow-step" style="border-top-color:#D97706;">
        <div class="flow-num">Bước 5 &bull; Select</div>
        <div class="flow-name">Val-Only Selection &amp; GPU Tuning</div>
        <div class="flow-desc">Chấm điểm đồng thuận 4 tiêu chí trên Val + khử <code>|r|&gt;0.96</code> &rarr; khóa đúng 150 biến. Tuning 12 cấu hình GPU &amp; trọng số Blend hoàn toàn trên Val.</div>
      </div>
      <div class="flow-step" style="border-top-color:#16A34A;">
        <div class="flow-num">Bước 6 &bull; Final</div>
        <div class="flow-name">Chấm OOT &amp; Chốt Final Model</div>
        <div class="flow-desc">Đánh giá duy nhất trên OOT 2022-11 kèm 1.000x Client-Cluster Bootstrap, Holm-Bonferroni, per-fold stability &amp; phương án triển khai.</div>
      </div>
    </div>

    <div class="grid-2" style="margin-top:4px;">
      <div class="card card-accent">
        <div class="card-title">Tối Ưu Hóa Tài Nguyên &amp; Thời Gian Chạy</div>
        <ul>
          <li><strong>Chạy song song Local + Colab:</strong> DuckDB FE nặng RAM chạy local (<code>3.4p</code> Stage 1 + <code>11.7p</code> Stage 2) trong khi Colab T4 huấn luyện mô hình.</li>
          <li><strong>Mọi mô hình &lt; 30 phút:</strong> XGBoost GPU <code>9.3p/seed</code>, CatBoost GPU <code>10.6p/seed</code>, LogReg <code>float32</code> in-place <code>2.6p</code> (khắc phục hoàn toàn OOM), LightGBM tối ưu <code>max_bin=127</code>.</li>
          <li><strong>Cache &amp; Khôi phục tự động:</strong> Toàn bộ predictions &amp; metrics lưu trực tiếp Google Drive + đồng bộ GitHub <code>main</code> (tuyệt đối không commit <code>.parquet/.npz/.zip</code>).</li>
        </ul>
      </div>
      <div class="card card-green">
        <div class="card-title">Nguyên Tắc Kỷ Luật Khoa Học Dữ Liệu Xuyên Suốt</div>
        <ul>
          <li><strong>Zero OOT Peeking:</strong> Mọi quyết định chọn biến (150 biến), siêu tham số, trọng số Blend đều chốt 100% trên Validation <code>2022-05..08</code>.</li>
          <li><strong>Kiểm định thống kê chặt chẽ:</strong> Báo cáo khoảng tin cậy 95% CI qua 1.000 lượt Client-Cluster Bootstrap và hiệu chỉnh đa kiểm định <code>Holm-Bonferroni</code>.</li>
          <li><strong>Minh bạch đánh đổi (Trade-off):</strong> Báo cáo trung thực cả ưu thế toàn cục của <code>M5_blend_150</code>/<code>M4_150</code> lẫn ưu thế gọn nhẹ Top-1% của <code>B5_top50</code>.</li>
        </ul>
      </div>
    </div>
  </div>
  <div class="ftr">
    <span>Propensity H=3 &bull; Rolling Snapshot = 1</span>
    <span>Slide 2 / 12</span>
  </div>
</section>

<!-- ==================== SLIDE 3: MUC 1 - EDA ==================== -->
<section class="slide">
  <div class="hdr">
    <div class="hdr-left">
      <div class="hdr-kicker">Mục 1 &bull; Phân Tích Khám Phá Dữ Liệu (EDA)</div>
      <div class="hdr-title">6 Nhận Định Cốt Lõi Từ 105,3 Triệu Log &amp; Đóng Góp Trực Tiếp Vào Feature</div>
    </div>
    <div class="hdr-pill">15/15 Kiểm Tra Đối Soát Đạt</div>
  </div>
  <div class="body">
    <div class="grid-6">
      <div class="card card-accent">
        <div class="card-title">1. DIALOG gần cutoff là tín hiệu mạnh nhất</div>
        <p>Có DIALOG tháng trước: tỷ lệ mua <strong>4.22%</strong> (<strong>Lift 3.71&times;</strong> so với nền 1.14%). Chỉ chiếm 5.75% quan sát nhưng chứa <strong>21.33%</strong> tổng người mua; ổn định cả 10/10 tháng.</p>
        <p style="margin-top:6px;"><strong>&rarr; Feature:</strong> <code>prev_cal_month</code>, <code>7d/14d</code>, suy giảm mũ <code>exp(-d/14|45)</code>, burst, gap &amp; PCA embedding.</p>
      </div>
      <div class="card card-accent">
        <div class="card-title">2. Tương tác cực nóng: 0-TRX &times; DIALOG gần</div>
        <p>Khách không có giao dịch nhưng vừa có DIALOG đạt tỷ lệ mua <strong>9.41%</strong> (so với 1.12% khi không có DIALOG). Trên tập Val H=3, phân khúc này lên tới <strong>11.66%</strong>.</p>
        <p style="margin-top:6px;"><strong>&rarr; Feature:</strong> <code>zero_trx_90d_x_dialog_30d</code>, <code>zero_trx_all_x_dialog_30d</code>, <code>source_bitmask_30d/90d</code>.</p>
      </div>
      <div class="card card-accent">
        <div class="card-title">3. TRX có quan hệ hình chữ U (phi đơn điệu)</div>
        <p>Tỷ lệ mua H=1 theo nhóm số giao dịch <code>0 / 1–9 / 10–99 / &ge;100</code> là <strong>1.41% / 0.76% / 0.59% / 1.48%</strong>. Không có log không đồng nghĩa không mua!</p>
        <p style="margin-top:6px;"><strong>&rarr; Feature:</strong> Giữ LEFT JOIN (không gán recency thiếu = 0), tạo 4 cờ chế độ <code>trx_regime_0/1_9/10_99/100p</code>.</p>
      </div>
      <div class="card card-accent">
        <div class="card-title">4. Đuôi amount cực lệch theo currency</div>
        <p>Riêng <code>currency=11</code> (38.72M dòng) có median <code>18.025</code>, p99 <code>5.26M</code>, max <strong>1.37 nghìn tỷ</strong>. Không cộng gộp khác tiền tệ; mean thô bị ngoại lai kéo méo.</p>
        <p style="margin-top:6px;"><strong>&rarr; Feature:</strong> <code>median</code>, <code>signed_log_mean</code>, <code>log_max_minus_med</code>, tỷ trọng ngoại tệ <code>foreign_fraction</code>.</p>
      </div>
      <div class="card card-accent">
        <div class="card-title">5. Độ phủ 3 nguồn &amp; lệch tuổi lịch sử</div>
        <p>Đủ 3 nguồn mua <strong>1.98%</strong> vs 1 nguồn 0.64%. Nhưng 3 nguồn bắt đầu lệch thời điểm: TRX <code>01/2021</code>, DIALOG <code>02/2021</code>, <strong>GEO <code>12/2021</code></strong> (rất ngắn).</p>
        <p style="margin-top:6px;"><strong>&rarr; Feature:</strong> <code>obs_age_days_*</code>, <code>trx_tenure_frac_of_obs</code>, <code>cap365</code>, GEO Entropy &amp; Novelty.</p>
      </div>
      <div class="card card-accent">
        <div class="card-title">6. DIALOG độc lập với lịch sử mua (Prior)</div>
        <p>Ngay ở nhóm <strong>chưa từng mua trước đó</strong>, có DIALOG gần vẫn mua <strong>3.87%</strong> vs 0.83%. Ngoài ra <code>trx_count</code> &amp; <code>active_days</code> trùng thông tin (Spearman <code>0.998</code>).</p>
        <p style="margin-top:6px;"><strong>&rarr; Feature:</strong> Tỷ lệ hóa <code>prior_buy_rate_per_obs_mo</code>, tương tác <code>dialog &times; prior</code> &amp; khử <code>|r|&gt;0.96</code>.</p>
      </div>
    </div>
    <div class="card card-navy" style="padding:10px 16px;">
      <strong>Nguyên tắc nối bảng từ EDA:</strong> <code>client_split</code> (1 dòng/KH) &rarr; <code>targets</code> (1 dòng/KH-tháng) &larr; <strong>LEFT JOIN</strong> từng nguồn TRX / GEO / DIALOG đã tổng hợp riêng trước cutoff (<code>validate='one_to_one'</code>). Tuyệt đối không nối thẳng TRX&times;GEO (tránh nổ tổ hợp m&times;n) và giữ nguyên khách không có log.
    </div>
  </div>
  <div class="ftr">
    <span>Nguồn đối soát: docs/eda_insights_checklist_20261119.md &bull; 0 orphan, 0 null khóa/thời gian</span>
    <span>Slide 3 / 12</span>
  </div>
</section>

<!-- ==================== SLIDE 4: H=3 ZERO-LEAK DESIGN ==================== -->
<section class="slide">
  <div class="hdr">
    <div class="hdr-left">
      <div class="hdr-kicker">Thiết Kế Bài Toán &amp; Chống Rò Rỉ Thời Gian</div>
      <div class="hdr-title">Định Nghĩa H=3, Cửa Sổ Trượt Snapshot=1 &amp; Cơ Chế Embargo Khóa OOT</div>
    </div>
    <div class="hdr-pill">4/4 Unit Test Chống Leak PASS</div>
  </div>
  <div class="body">
    <div class="card">
      <div class="card-title">Bản đồ thời gian 12 tháng nhãn (2022-02 &rarr; 2023-01) — Tuyệt đối không để OOT tràn vào Train/Val</div>
      <div class="timeline">
        <div class="tl-cell tl-train">2022-02<br/>Train</div>
        <div class="tl-cell tl-train">2022-03<br/>Train</div>
        <div class="tl-cell tl-train">2022-04<br/>Train</div>
        <div class="tl-cell tl-val">2022-05<br/>Val #1</div>
        <div class="tl-cell tl-val">2022-06<br/>Val #2</div>
        <div class="tl-cell tl-val">2022-07<br/>Val #3</div>
        <div class="tl-cell tl-val">2022-08<br/>Val #4</div>
        <div class="tl-cell tl-emb">2022-09<br/>EMBARGO</div>
        <div class="tl-cell tl-emb">2022-10<br/>EMBARGO</div>
        <div class="tl-cell tl-oot">2022-11<br/>OOT TEST</div>
        <div class="tl-cell tl-oot" style="opacity:0.75; border-style:dashed;">2022-12<br/>(H3 m+1)</div>
        <div class="tl-cell tl-oot" style="opacity:0.75; border-style:dashed;">2023-01<br/>(H3 m+2)</div>
      </div>
      <p style="margin-top:8px; font-size:11.8px; color:#475569;">
        <strong>Vì sao phải cách ly <code>2022-09</code> và <code>2022-10</code>?</strong> Với <code>y_h3(s) = max(y[s], y[s+1], y[s+2])</code>, nhãn của <code>2022-09</code> chạm tới <code>2022-11</code> và nhãn của <code>2022-10</code> chạm tới <code>2022-12</code>. Loại bỏ cứng 2 tháng này đảm bảo 100% nhãn huấn luyện OOT đóng xong trước <code>2022-11-01</code>.
      </p>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">Lịch Trượt Validation &amp; Locked OOT Test</div>
        <table>
          <thead>
            <tr><th>Tập đánh giá</th><th>Tháng đánh giá</th><th>Các tháng Train</th><th>Nhãn Train đóng xong</th></tr>
          </thead>
          <tbody>
            <tr><td>Val Step 1</td><td><code>2022-05</code></td><td><code>[02]</code></td><td><code>2022-04-30</code></td></tr>
            <tr><td>Val Step 2</td><td><code>2022-06</code></td><td><code>[02, 03]</code></td><td><code>2022-05-31</code></td></tr>
            <tr><td>Val Step 3</td><td><code>2022-07</code></td><td><code>[02, 03, 04]</code></td><td><code>2022-06-30</code></td></tr>
            <tr><td>Val Step 4</td><td><code>2022-08</code></td><td><code>[02, 03, 04, 05]</code></td><td><code>2022-07-31</code></td></tr>
            <tr class="hl-green"><td><strong>Locked OOT</strong></td><td><strong><code>2022-11</code></strong></td><td><strong><code>[02 .. 08]</code></strong></td><td><strong><code>2022-10-31</code></strong></td></tr>
          </tbody>
        </table>
      </div>
      <div class="card card-accent">
        <div class="card-title">5 Rào Chắn Chống Leak Bắt Buộc Trong Pipeline</div>
        <ul>
          <li><strong>Cutoff cứng theo giờ Moscow:</strong> Toàn bộ TRX/GEO/DIALOG lọc <code>event_time_utc + 3h &lt; YYYY-MM-01 00:00:00 MSK</code>, dùng toàn bộ lịch sử trước cutoff.</li>
          <li><strong>5-Fold Out-of-Client (OOC):</strong> Ở mọi tháng đánh giá và mọi fold <code>k</code>, train chỉ dùng <code>fold != k</code>, suy luận trên <code>fold == k</code> (không lặp khách giữa train và eval).</li>
          <li><strong>Biến <code>prior_*</code> đóng hoàn toàn:</strong> Chỉ tính từ các tháng nhãn <code>t &le; s - 1</code>. Tại <code>2022-02</code> (tháng lạnh), <code>prior_months_observed = 0</code> và mọi tỷ lệ/recency giữ <code>NaN</code>.</li>
          <li><strong>PCA Embedding fit trong fold:</strong> Mọi ma trận PCA DIALOG đều fit riêng trên <code>train_months &times; (fold != k)</code>.</li>
          <li><strong>Kiểm định tự động:</strong> <code>tests/test_h3_leakage_guards.py</code> (4/4 PASS) + Shuffled-label sanity check.</li>
        </ul>
      </div>
    </div>
  </div>
  <div class="ftr">
    <span>Cấu hình khóa: configs/h3_propensity_spec.json &bull; OOT 2022-11: 100.224 KH, 2.173 người mua (2.168%), Top-1% K=1.003</span>
    <span>Slide 4 / 12</span>
  </div>
</section>

<!-- ==================== SLIDE 5: MUC 2 - BASELINES ==================== -->
<section class="slide">
  <div class="hdr">
    <div class="hdr-left">
      <div class="hdr-kicker">Mục 2 &bull; Hệ Thống 5 Baselines (Stage 1)</div>
      <div class="hdr-title">5 Baselines Trên Tập 50 Biến Có <code>prior_*</code>: Tác Dụng, Biến &amp; Kết Quả Lần Lượt</div>
    </div>
    <div class="hdr-pill">Cùng 50 Biến Chuẩn &bull; So Sánh Công Bằng</div>
  </div>
  <div class="body">
    <div class="card" style="padding:10px 14px;">
      <div class="card-title" style="margin-bottom:4px;">Cấu trúc tập 50 biến Baseline (<code>top50_features</code>)</div>
      <p>
        <span class="tag tag-info">13 biến prior_*</span> (mua 1m/3m/6m/all, recency, p1..p4) &nbsp;+&nbsp;
        <span class="tag tag-info">9 biến DIALOG</span> (has/count/recency 30d, 90d, all) &nbsp;+&nbsp;
        <span class="tag tag-info">20 biến TRX</span> (RFM, log1p, median/max, diversity, ngoại tệ) &nbsp;+&nbsp;
        <span class="tag tag-info">7 biến GEO</span> (gh4/gh6, recency, count) &nbsp;+&nbsp;
        <span class="tag tag-info">1 cold_start</span>
      </p>
    </div>

    <table>
      <thead>
        <tr>
          <th style="width:18%;">Baseline</th>
          <th style="width:31%;">Tác dụng từng tầng trong thang đo</th>
          <th class="num">Seeds</th>
          <th class="num">Val PR-AUC</th>
          <th class="num">Val Lift@1%</th>
          <th class="num">OOT PR-AUC</th>
          <th class="num">OOT Lift@1% [95% CI]</th>
          <th class="num">OOT ROC</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>B1. Rule Prior+RFM</strong></td>
          <td>Mốc sàn phi-ML (luật điểm nghiệp vụ): xác minh ML có vượt quy tắc tay hay không.</td>
          <td class="num">1</td>
          <td class="num">0.06794</td>
          <td class="num">4.53x</td>
          <td class="num">0.06197</td>
          <td class="num">5.61x [4.70, 6.61]</td>
          <td class="num">0.7002</td>
        </tr>
        <tr>
          <td><strong>B2. LogReg Top-50</strong></td>
          <td>Mốc tuyến tính L2 chuẩn hóa: đo lợi ích tối ưu trọng số toàn cục &amp; hiệu chuẩn xác suất.</td>
          <td class="num">1</td>
          <td class="num">0.07652</td>
          <td class="num">4.78x</td>
          <td class="num">0.07547</td>
          <td class="num">6.71x [5.79, 7.71]</td>
          <td class="num">0.7613</td>
        </tr>
        <tr>
          <td><strong>B3. LightGBM Top-50</strong></td>
          <td>Mốc GBDT phi tuyến đầu tiên (leaf-wise): bắt tương tác phi đơn điệu (chữ U TRX).</td>
          <td class="num">3</td>
          <td class="num">0.07674</td>
          <td class="num">4.71x</td>
          <td class="num">0.08398</td>
          <td class="num">7.96x [7.05, 8.98]</td>
          <td class="num">0.7702</td>
        </tr>
        <tr>
          <td><strong>B4. XGBoost GPU Top-50</strong></td>
          <td>Mốc depth-wise GBDT trên GPU T4 (~3.1p/seed): đối chiếu chéo độ ổn định với LGBM.</td>
          <td class="num">3</td>
          <td class="num">0.07641</td>
          <td class="num">4.62x</td>
          <td class="num">0.08454</td>
          <td class="num">8.09x [7.14, 9.12]</td>
          <td class="num">0.7699</td>
        </tr>
        <tr class="hl-green">
          <td><strong>B5. CatBoost GPU Top-50</strong></td>
          <td>Mốc oblivious trees ít overfit nhất với biến thiếu/tỷ lệ &rarr; <strong>Baseline mạnh nhất</strong>.</td>
          <td class="num">3</td>
          <td class="num">0.07834</td>
          <td class="num">4.88x</td>
          <td class="num">0.08844</td>
          <td class="num">8.31x [7.40, 9.41]</td>
          <td class="num">0.7703</td>
        </tr>
      </tbody>
    </table>

    <div class="grid-3">
      <div class="card card-accent">
        <div class="card-title">B2 vs B1 (ML vs Luật tay)</div>
        <p>OOT PR-AUC tăng <strong>+21.8%</strong> (<code>+0.0135</code>, <code>p_holm &lt; 0.001</code>), Lift@1% tăng <strong>+1.10x</strong> (<code>p = 0.034</code>). ML vượt trội rõ rệt quy tắc RFM.</p>
      </div>
      <div class="card card-accent">
        <div class="card-title">B5 vs B2 (Phi tuyến vs Tuyến tính)</div>
        <p>OOT PR-AUC tăng <strong>+17.2%</strong> (<code>+0.0130</code>, <code>p_holm &lt; 0.001</code>), Lift@1% tăng <strong>+1.59x</strong> (<strong>+23.7%</strong>, <code>p_holm &lt; 0.001</code>).</p>
      </div>
      <div class="card card-green">
        <div class="card-title">Chất lượng đỉnh của B5 Top-50</div>
        <p>Đạt <code>Precision@1% = 18.01%</code> (~181 người mua / 1.003 KH), <code>LogLoss = 0.09410</code> tốt nhất và chạy chỉ <code>~7.3 phút/seed</code> trên T4.</p>
      </div>
    </div>
  </div>
  <div class="ftr">
    <span>Paired Client-Cluster Bootstrap 1.000 lượt, hiệu chỉnh Holm-Bonferroni &bull; outputs/h3_propensity/metrics/stage1_baselines_*.csv</span>
    <span>Slide 5 / 12</span>
  </div>
</section>

<!-- ==================== SLIDE 6: 6 WARNINGS VERIFICATION ==================== -->
<section class="slide">
  <div class="hdr">
    <div class="hdr-left">
      <div class="hdr-kicker">Kiểm Chứng Rủi Ro &amp; Rò Rỉ (Stage 1 Audit)</div>
      <div class="hdr-title">Xác Minh Định Lượng Toàn Bộ 6 Cảnh Báo Kỹ Thuật Trong Suốt Dự Án</div>
    </div>
    <div class="hdr-pill">6/6 Cảnh Báo Đã Được Kiểm Định</div>
  </div>
  <div class="body">
    <div class="grid-2-img" style="flex:1;">
      <div class="img-box">
        <img src="{fig3}" alt="6 Warnings Verification Panel" />
      </div>
      <div style="display:flex; flex-direction:column; gap:8px; justify-content:space-between;">
        <div class="card" style="padding:9px 12px;">
          <div class="card-title" style="font-size:12.5px; margin-bottom:2px;"><span class="tag tag-ok">PASS</span> V1. DIALOG Lag (0 / 7 / 14 / 30d &amp; no_dialog)</div>
          <p style="font-size:11.4px;">Trễ 7d/14d giữ <code>PR 0.0761–0.0763</code> (<code>L1 4.91–5.00x</code>); bỏ hẳn mới tụt <code>0.0751</code> (-4.4%). <strong>Khẳng định tín hiệu hành vi thật, không leak sát cutoff.</strong></p>
        </div>
        <div class="card" style="padding:9px 12px;">
          <div class="card-title" style="font-size:12.5px; margin-bottom:2px;"><span class="tag tag-warn">LƯU Ý SLA</span> V2. Độ trễ nhãn Prior &amp; Label Overlap</div>
          <p style="font-size:11.4px;">Trễ nhãn 1m giảm PR <code>-3.9%</code> (<code>0.0755</code>); bỏ hẳn giảm <code>-9.3%</code> (<code>0.0712</code>). Rolling <code>02..05</code> (<code>0.0843</code>) &le; Non-overlap <code>02,05</code> (<code>0.0867</code>) &rarr; <strong>0% overlap inflation.</strong></p>
        </div>
        <div class="card" style="padding:9px 12px;">
          <div class="card-title" style="font-size:12.5px; margin-bottom:2px;"><span class="tag tag-ok">PASS</span> V3. Múi giờ MSK vs UTC &amp; V4. Tenure Left-Censor</div>
          <p style="font-size:11.4px;">MSK vs UTC gần như đồng nhất (<code>0.05807 vs 0.05807</code>). Chặn trần tenure <code>cap365</code> <strong>tăng Val PR</strong> <code>0.0785 &rarr; 0.0794</code> (<code>L1 5.09x</code>) &rarr; đưa vào Stage 2.</p>
        </div>
        <div class="card" style="padding:9px 12px;">
          <div class="card-title" style="font-size:12.5px; margin-bottom:2px;"><span class="tag tag-info">INSIGHT</span> V5. Geohash Drift &amp; Slice 0-TRX &times; DIALOG</div>
          <p style="font-size:11.4px;">Bỏ geohash thô tăng nhẹ Val PR (<code>0.0791</code>) do GEO bắt đầu muộn <code>12/2021</code> &rarr; đổi sang Entropy/Novelty. Slice <code>0-TRX&times;DLG</code> có tỷ lệ mua <strong>11.66%</strong>.</p>
        </div>
        <div class="card card-green" style="padding:9px 12px;">
          <div class="card-title" style="font-size:12.5px; margin-bottom:2px;"><span class="tag tag-ok">0% LEAK</span> V6. Shuffled-Label Sanity Check</div>
          <p style="font-size:11.4px;"><code>shuffled_y_train</code> Val/OOT <code>ROC-AUC = 0.509 / 0.493</code>; <code>shuffled_y_eval_oot</code> <strong><code>Lift@1% = 0.97x</code></strong>, <code>PR = 0.0212</code> (trùng nền <code>0.0217</code>).</p>
        </div>
      </div>
    </div>
  </div>
  <div class="ftr">
    <span>Tham chiếu: 0_reference CatBoost Top-50 trên Val 05..08 (PR=0.07851, L1=4.99x, ROC=0.7281) &bull; stage1_warning_verifications.csv</span>
    <span>Slide 6 / 12</span>
  </div>
</section>

<!-- ==================== SLIDE 7: MUC 3 - FEATURE ENGINEERING ==================== -->
<section class="slide">
  <div class="hdr">
    <div class="hdr-left">
      <div class="hdr-kicker">Mục 3 &bull; Feature Engineering Từ Phát Hiện EDA</div>
      <div class="hdr-title">123 Biến Mới / Mở Rộng &rarr; Kho 272 Biến Ứng Viên (5 Nhóm Cốt Lõi)</div>
    </div>
    <div class="hdr-pill">Local DuckDB &bull; 801.792 Dòng &bull; 11.7 Phút</div>
  </div>
  <div class="body">
    <table>
      <thead>
        <tr>
          <th style="width:19%;">Nhóm đặc trưng</th>
          <th class="num" style="width:8%;">Quy mô</th>
          <th style="width:41%;">Biến tiêu biểu &amp; Công thức tính (100% <code>&lt; cutoff MSK</code>)</th>
          <th style="width:32%;">Cơ sở EDA &amp; Ý nghĩa nghiệp vụ</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>1. DIALOG chuyên sâu &amp; Embedding</strong></td>
          <td class="num">~45</td>
          <td><code>dialog_recency_exp_14d/45d = exp(-d/&tau;)</code>; <code>burst_7d_vs_30d</code>; <code>prev_cal_month</code>; <code>inter_gap_{{mean,std,max}}_90d</code>; <strong>PCA-16 (90d &amp; all) fit trong fold</strong>; <code>L2_drift_90d_all</code>; <code>within_std_90d</code>.</td>
          <td>Khai thác trọn vẹn tín hiệu mạnh nhất (Lift 3.71x), bắt độ tươi, bùng nổ tương tác và sự thay đổi chủ đề hội thoại.</td>
        </tr>
        <tr>
          <td><strong>2. GEO không gian – thời gian</strong></td>
          <td class="num">~24</td>
          <td><strong>Shannon Entropy</strong> <code>-&Sigma; p&middot;ln(p)</code> trên <code>gh4/gh5/gh6</code> (30d/90d); <strong>Novelty</strong> (<code>new_gh4/5/6_count</code> mới trong 30d); mật độ <code>gh6/gh4</code>; <code>geo_recency_exp_14d</code>; <code>events_per_active_day_90d</code>.</td>
          <td>Khắc phục nhược điểm đếm thô bị drift của GEO (<code>12/2021+</code>), đo hành vi di chuyển &amp; khám phá khu vực mới.</td>
        </tr>
        <tr>
          <td><strong>3. TRX phi tuyến &amp; Đa dạng</strong></td>
          <td class="num">~30</td>
          <td>4 cờ hình chữ U (<code>trx_regime_0/1_9/10_99/100p</code>); độ lệch đuôi <code>sign&middot;log1p(|max|) - sign&middot;log1p(|med|)</code>; <code>new_event_subtype/dst12_count</code>; <code>trx_share_90d_of_all</code>; <code>dormant_days_gap</code>.</td>
          <td>Xử lý quan hệ phi đơn điệu hình chữ U và đuôi lệch <code>1.37e12</code> của <code>currency=11</code> mà không xóa khách VIP.</td>
        </tr>
        <tr>
          <td><strong>4. Tuổi quan sát &amp; Censoring</strong></td>
          <td class="num">~6</td>
          <td><code>obs_age_days_{{trx,geo,dlg}}</code> (số ngày từ mốc bắt đầu từng nguồn); <code>trx_tenure_frac_of_obs</code>; <code>trx_all_days_since_first_event_cap365</code>.</td>
          <td>Tách biệt "ít log do khách mới" với "ít log do nguồn mới thu thập"; loại nhiễu left-censoring tại <code>01/2021</code>.</td>
        </tr>
        <tr>
          <td><strong>5. Tương tác đa nguồn &amp; Prior tỷ lệ</strong></td>
          <td class="num">~18</td>
          <td><code>zero_trx_{{90d,all}}_x_dialog_30d</code>; <code>zero_trx_x_geo_30d</code>; <code>source_bitmask_30d/90d</code> (<code>1T+2G+4D</code>); <code>prior_buy_rate_per_obs_mo</code>; <code>dialog_recency_x_prior_recency</code>.</td>
          <td>Chuyển thẳng tương tác nóng <code>0-TRX&times;DIALOG</code> (9.41%–11.66%) và tỷ lệ mua chuẩn hóa tuổi quan sát vào mô hình.</td>
        </tr>
      </tbody>
    </table>

    <div class="card card-amber" style="padding:10px 14px;">
      <strong>Bảo vệ chống rò rỉ trong FE:</strong> Toàn bộ 123 biến mới chỉ đọc sự kiện <code>&lt; YYYY-MM-01 00:00 MSK</code>. Toàn bộ 33 đặc trưng embedding (<code>16 PCA 90d + 16 PCA all + norm_ratio/drift</code>) được fit riêng trên tập train của từng bước trượt &times; từng fold (<code>fold != k</code>), không bao giờ fit trên tập Val hay OOT.
    </div>
  </div>
  <div class="ftr">
    <span>Script tái lập: src/scripts/build_h3_stage2_eda_fe.py &bull; Tổng kho ứng viên: 239 biến bảng + 33 biến embedding = 272 biến</span>
    <span>Slide 7 / 12</span>
  </div>
</section>

<!-- ==================== SLIDE 8: MUC 4 - FEATURE SELECTION ==================== -->
<section class="slide">
  <div class="hdr">
    <div class="hdr-left">
      <div class="hdr-kicker">Mục 4 &bull; Chọn Lọc Đặc Trưng (Val-Only Selection)</div>
      <div class="hdr-title">Quy Trình Rút Gọn 272 &rarr; Đúng 150 Biến (100% Trên Validation, Không Chạm OOT)</div>
    </div>
    <div class="hdr-pill">129 Biến Bảng + 21 Embedding &bull; 9.3 Phút GPU</div>
  </div>
  <div class="body">
    <div class="grid-2-img" style="flex:1;">
      <div class="img-box">
        <img src="{fig4}" alt="Top 20 EDA Features Importance" />
      </div>
      <div style="display:flex; flex-direction:column; gap:10px;">
        <div class="card card-accent">
          <div class="card-title">4 Bước Chọn Lọc Chặt Chẽ Trên Val (2022-05..08)</div>
          <ul>
            <li><strong>Bước 1 — Lọc sơ bộ:</strong> Loại biến hằng số, thiếu &gt;99.95% và toàn bộ biến thí nghiệm <code>prior_delay1m_*</code>.</li>
            <li><strong>Bước 2 — Xếp hạng đồng thuận (Consensus Rank):</strong> Trung bình thứ hạng 4 tín hiệu trên Val: <code>XGB GPU Gain</code> + <code>CatBoost Importance</code> + <code>Permutation &Delta;PR-AUC</code> + <code>Permutation &Delta;Lift@1%</code>.</li>
            <li><strong>Bước 3 — Khử đa cộng tuyến (<code>|Spearman r| &gt; 0.96</code>):</strong> Giữ biến có thứ hạng đồng thuận cao hơn trong mỗi cặp trùng thông tin (như <code>trx_count</code> vs <code>active_days</code>).</li>
            <li><strong>Bước 4 — Khóa 150 biến:</strong> Bảo toàn 18 biến neo cốt lõi + lấy top theo rank &rarr; <strong><code>129</code> tabular + <code>21</code> embedding</strong> (19 PCA + <code>norm_ratio</code> + <code>l2_drift</code>).</li>
          </ul>
        </div>
        <div class="card card-green">
          <div class="card-title">Minh Chứng Sức Mạnh Của FE Mới Trong Top 12</div>
          <p>Các biến mới tạo ở Stage 2 chiếm ưu thế áp đảo ngay đầu bảng xếp hạng đồng thuận Val:</p>
          <ul style="margin-top:4px;">
            <li><strong>#1</strong> <code>trx_share_90d_of_all</code> (<code>0.943</code>)</li>
            <li><strong>#6</strong> <code>trx_all_days_since_first_event_cap365</code> (<code>0.911</code>)</li>
            <li><strong>#7</strong> <code>trx_30d_new_dst12_count</code> (<code>0.908</code>)</li>
            <li><strong>#8</strong> <code>trx_tenure_frac_of_obs</code> (<code>0.885</code>)</li>
            <li><strong>#10–#12</strong> <code>dialog_recency_exp_14d</code>, <code>prior_buy_rate_per_obs_mo</code>, <code>dlg_pca_90d_7</code></li>
          </ul>
        </div>
      </div>
    </div>
  </div>
  <div class="ftr">
    <span>Danh sách khóa: outputs/h3_propensity/metrics/fe_h3_150_locked.json &amp; val_candidate_feature_ranking.csv</span>
    <span>Slide 8 / 12</span>
  </div>
</section>

<!-- ==================== SLIDE 9: MUC 5 - EXPERIMENTS & FINAL ARCHITECTURE ==================== -->
<section class="slide">
  <div class="hdr">
    <div class="hdr-left">
      <div class="hdr-kicker">Mục 5 &bull; Thử Nghiệm, Bài Học &amp; Kiến Trúc Final Model</div>
      <div class="hdr-title">Tuning 12 Cấu Hình GPU, Ablation <code>no_prior</code> &amp; Thiết Kế Final Model</div>
    </div>
    <div class="hdr-pill">Mọi Mô Hình GPU &lt; 11 Phút / Seed</div>
  </div>
  <div class="body">
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Các Thử Nghiệm Đã Thực Hiện Trên 150 Biến</div>
        <ul>
          <li><strong>Grid Tuning GPU trên Val (<code>826.5s</code>, 12 cấu hình):</strong>
            <br/>&bull; Best CatBoost: <code>cb_d6_spw3</code> (<code>depth=6, l2=30, lr=0.03, spw=3.0</code> &rarr; Val <code>PR=0.08523, L1=5.77x</code>).
            <br/>&bull; Best XGBoost: <code>xgb_d5_l25_lr03</code> (<code>depth=5, mcw=60, l2=25, lr=0.03</code> &rarr; Val <code>PR=0.08274, L1=5.30x</code>).
          </li>
          <li><strong>4 họ mô hình đầy đủ (Val + OOT):</strong> <code>M1_logreg</code> (<code>2.6p</code>), <code>M2_lgbm</code> (<code>30.1p</code> CPU), <code>M3_xgb_gpu</code> (3s, <code>9.3p/s</code>), <code>M4_catboost_gpu</code> (3s, <code>10.6p/s</code>).</li>
          <li><strong>Quét 48 tổ hợp Blend trên Val:</strong> Chốt phương thức <code>prob</code> với trọng số <code>0.50 CB + 0.30 LGBM + 0.20 XGB</code>.</li>
          <li><strong>Ablation cắt bỏ <code>prior_*</code> (<code>M6_xgb_150_no_prior</code>, 139 biến):</strong> OOT <code>Lift@1%</code> tụt từ <code>7.20x &rarr; 5.98x</code> (<strong>-1.23x, <code>p_holm &lt; 0.001</code></strong>), <code>PR-AUC</code> tụt <code>-0.0071</code> (<strong><code>p_holm &lt; 0.001</code></strong>).</li>
        </ul>
      </div>
      <div class="card card-accent">
        <div class="card-title">Bài Học Kỹ Thuật &amp; Mô Hình Hóa Rút Ra</div>
        <ul>
          <li><strong>FE-150 nâng đều cả 4 họ mô hình trên Val:</strong> LogReg <code>+3.2%</code>, LightGBM <code>+5.8%</code>, XGBoost <code>+3.4%</code>, CatBoost <code>+3.4%</code> PR-AUC.</li>
          <li><strong>Bề rộng (Global/Top-5%) vs Đuôi hẹp (Top-1%):</strong> 150 biến nâng <code>OOT ROC-AUC</code> rõ rệt (<code>+0.0053</code>, <code>p &lt; 0.001</code>) và <code>Lift@5%</code> (<code>5.01x</code>), nhưng <code>spw=3.0</code> làm tăng <code>LogLoss</code> (<code>0.1208</code>) và tăng nhẹ phương sai ở đuôi hẹp Top-1% khi tỷ lệ nền giảm (<code>2.95% &rarr; 2.17%</code>).</li>
          <li><strong><code>prior_*</code> hợp lệ là đòn bẩy lớn nhất (+20.5% Lift@1%):</strong> Khẳng định giá trị vượt trội của lịch sử mua đã đóng trước cutoff.</li>
          <li><strong>Xử lý sự cố kỹ thuật Colab:</strong> Viết lại LogReg <code>float32</code> in-place + prune panel để triệt tiêu OOM (exit 137); tối ưu <code>max_bin=127</code> đưa LightGBM về đúng hạn mức 30 phút.</li>
        </ul>
      </div>
    </div>

    <div class="arch-box">
      <div style="font-size:13px; font-weight:700; color:#7DD3FC; text-transform:uppercase; letter-spacing:0.8px;">Kiến Trúc Final Model — M5_final_blend_150 (Khóa Trọng Số 100% Trên Validation)</div>
      <div class="arch-row">
        <div class="arch-node">
          <strong class="node-title">Đầu vào (150 biến &lt; cutoff MSK)</strong>
          129 biến bảng (TRX + DIALOG + GEO + Prior + Obs-Age) + 21 DIALOG Embedding (PCA trong fold + Drift)
        </div>
        <div class="arch-arrow">&rarr;</div>
        <div class="arch-node" style="flex:1.45; text-align:left; line-height:1.55;">
          <strong class="node-title" style="text-align:center;">3 Mô Hình Thành Phần Đa Dạng</strong>
          &bull; <strong>50%</strong> CatBoost GPU (<code>d=6, l2=30, 3 seeds</code>)<br/>
          &bull; <strong>30%</strong> LightGBM (<code>leaves=24, l2=35, max_bin=127</code>)<br/>
          &bull; <strong>20%</strong> XGBoost GPU (<code>d=5, mcw=60, l2=25, 3 seeds</code>)
        </div>
        <div class="arch-arrow">&rarr;</div>
        <div class="arch-node" style="border-color:#4ADE80;">
          <strong class="node-title" style="color:#4ADE80;">Probability Blend + Calib</strong>
          Xếp hạng toàn cục tốt nhất (<code>ROC 0.7756</code>) + gắn Isotonic Calibration khi deploy để tối ưu <code>LogLoss</code>
        </div>
      </div>
    </div>
  </div>
  <div class="ftr">
    <span>Thông số khóa: locked_final_model_spec.json &amp; locked_ensemble_weights.json &bull; stage2_model_timings.csv</span>
    <span>Slide 9 / 12</span>
  </div>
</section>

<!-- ==================== SLIDE 10: VALIDATION CURVE ==================== -->
<section class="slide">
  <div class="hdr">
    <div class="hdr-left">
      <div class="hdr-kicker">Kiểm Định Ngoài Thời Gian Trên Tập Validation</div>
      <div class="hdr-title">Tiến Triển Qua 4 Tháng Trượt Validation (2022-05 &rarr; 2022-08): Top-50 vs FE-150</div>
    </div>
    <div class="hdr-pill">400.896 Khách-Tháng Val</div>
  </div>
  <div class="body">
    <div class="img-box" style="flex:1; max-height:4.3in;">
      <img src="{fig1}" alt="Validation Rolling Curve" />
    </div>
    <div class="grid-3">
      <div class="card card-green">
        <div class="card-title">FE-150 thắng tuyệt đối trên Validation</div>
        <p><code>M5_final_blend_150</code> dẫn đầu với <strong><code>PR-AUC = 0.08175</code></strong> (+4.4% vs best Top-50 <code>B5 = 0.07834</code>), <strong><code>ROC-AUC = 0.7370</code></strong> và <strong><code>Lift@5% = 3.83x</code></strong>.</p>
      </div>
      <div class="card card-accent">
        <div class="card-title">Lợi ích tích lũy lịch sử train</div>
        <p>Khi cửa sổ train mở rộng từ 1 tháng (<code>02</code>) lên 4 tháng (<code>02..05</code>), mọi mô hình đều tăng dốc ổn định (<code>PR-AUC</code> từ ~<code>0.070</code> lên ~<code>0.086</code> tại <code>2022-08</code>).</p>
      </div>
      <div class="card card-accent">
        <div class="card-title">Cải thiện đồng loạt cả 4 họ mô hình</div>
        <p>LightGBM tăng <code>+0.00443</code> PR-AUC (<code>+5.8%</code>), CatBoost <code>+0.00268</code>, XGBoost <code>+0.00257</code>, LogReg <code>+0.00248</code> — chứng minh FE tạo tín hiệu thật.</p>
      </div>
    </div>
  </div>
  <div class="ftr">
    <span>Nguồn: stage1_baselines_val_monthly.csv &amp; stage2_150_val_monthly.csv</span>
    <span>Slide 10 / 12</span>
  </div>
</section>

<!-- ==================== SLIDE 11: MUC 6 - FINAL OOT RESULTS ==================== -->
<section class="slide">
  <div class="hdr">
    <div class="hdr-left">
      <div class="hdr-kicker">Mục 6 &bull; Kết Quả Cuối Cùng (Locked OOT Test 2022-11)</div>
      <div class="hdr-title">Hiệu Năng Toàn Bộ Mô Hình Trên Tập OOT Khóa (100.224 KH &bull; 2.173 Người Mua)</div>
    </div>
    <div class="hdr-pill">OOT Phủ 2022-11 &rarr; 2023-01</div>
  </div>
  <div class="body">
    <div class="grid-2-img" style="height:2.75in;">
      <div class="img-box">
        <img src="{fig2}" alt="OOT Test Model Comparison CI" />
      </div>
      <div class="card" style="padding:10px 12px;">
        <div class="card-title">Ổn Định Trên Cả 5 Fold OOT (<code>M5_final_blend_150</code>)</div>
        <table>
          <thead>
            <tr><th class="num">Fold</th><th class="num">n</th><th class="num">Pos</th><th class="num">PR-AUC</th><th class="num">ROC</th><th class="num">Lift@1%</th><th class="num">Lift@5%</th></tr>
          </thead>
          <tbody>
            <tr><td class="num">0</td><td class="num">20.328</td><td class="num">457</td><td class="num">0.0966</td><td class="num">0.7724</td><td class="num">9.16x</td><td class="num">4.99x</td></tr>
            <tr><td class="num">1</td><td class="num">19.902</td><td class="num">414</td><td class="num">0.0775</td><td class="num">0.7726</td><td class="num">5.77x</td><td class="num">5.45x</td></tr>
            <tr><td class="num">2</td><td class="num">19.917</td><td class="num">453</td><td class="num">0.0965</td><td class="num">0.7779</td><td class="num">9.01x</td><td class="num">5.21x</td></tr>
            <tr><td class="num">3</td><td class="num">20.017</td><td class="num">433</td><td class="num">0.0899</td><td class="num">0.7851</td><td class="num">7.82x</td><td class="num">5.54x</td></tr>
            <tr><td class="num">4</td><td class="num">20.060</td><td class="num">416</td><td class="num">0.0802</td><td class="num">0.7712</td><td class="num">8.16x</td><td class="num">4.23x</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <table>
      <thead>
        <tr>
          <th>Mô hình</th>
          <th class="num">Biến</th>
          <th class="num">Val PR-AUC</th>
          <th class="num">OOT PR-AUC [95% CI]</th>
          <th class="num">OOT Lift@1% [95% CI]</th>
          <th class="num">Lift@0.5%</th>
          <th class="num">Lift@5% (Recall)</th>
          <th class="num">OOT ROC-AUC</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code>B1_rule_prior_rfm</code></td><td class="num">50</td><td class="num">0.06794</td>
          <td class="num">0.06197 [0.056, 0.069]</td><td class="num">5.61x [4.70, 6.61]</td>
          <td class="num">5.70x</td><td class="num">3.97x (19.8%)</td><td class="num">0.7002</td>
        </tr>
        <tr>
          <td><code>B2_logreg_top50</code></td><td class="num">50</td><td class="num">0.07652</td>
          <td class="num">0.07547 [0.069, 0.084]</td><td class="num">6.71x [5.79, 7.71]</td>
          <td class="num">7.44x</td><td class="num">4.71x (23.6%)</td><td class="num">0.7613</td>
        </tr>
        <tr class="hl-amber">
          <td><code>M6_xgb_150_no_prior</code> (Ablation)</td><td class="num">139</td><td class="num">0.07727</td>
          <td class="num">0.07295 [0.067, 0.080]</td><td class="num">5.98x [5.12, 6.87]</td>
          <td class="num">6.80x</td><td class="num">4.41x (22.0%)</td><td class="num">0.7629</td>
        </tr>
        <tr>
          <td><code>M2_lgbm_150_tuned</code></td><td class="num">150</td><td class="num">0.08117</td>
          <td class="num">0.08385 [0.077, 0.092]</td><td class="num">8.00x [7.00, 9.03]</td>
          <td class="num">8.82x</td><td class="num">4.86x (24.3%)</td><td class="num">0.7747</td>
        </tr>
        <tr class="hl">
          <td><strong><code>M4_catboost_gpu_150_tuned</code> (Đơn #1)</strong></td><td class="num">150</td><td class="num">0.08102</td>
          <td class="num">0.08615 [0.079, 0.095]</td><td class="num">8.05x [7.16, 9.11]</td>
          <td class="num"><strong>9.62x</strong></td><td class="num"><strong>5.02x (25.1%)</strong></td><td class="num">0.7744</td>
        </tr>
        <tr class="hl-green">
          <td><strong><code>M5_final_blend_150</code> (Final Model)</strong></td><td class="num">150</td><td class="num"><strong>0.08175</strong></td>
          <td class="num">0.08597 [0.079, 0.095]</td><td class="num">7.92x [7.03, 8.98]</td>
          <td class="num"><strong>9.62x</strong></td><td class="num"><strong>5.01x (25.1%)</strong></td><td class="num"><strong>0.7756</strong></td>
        </tr>
        <tr class="hl">
          <td><strong><code>B5_catboost_gpu_top50</code> (Challenger gọn)</strong></td><td class="num">50</td><td class="num">0.07834</td>
          <td class="num"><strong>0.08844</strong> [0.081, 0.098]</td><td class="num"><strong>8.31x</strong> [7.40, 9.41]</td>
          <td class="num">9.56x</td><td class="num">4.94x (24.7%)</td><td class="num">0.7703</td>
        </tr>
      </tbody>
    </table>
  </div>
  <div class="ftr">
    <span>M5 vs B2: &Delta;PR=+0.0105 (p_holm&lt;0.001), &Delta;L1=+1.21x (p=0.022) &bull; M5 vs B5: &Delta;ROC=+0.0053 (p&lt;0.001), &Delta;L1 không khác biệt thống kê (p_holm=0.636)</span>
    <span>Slide 11 / 12</span>
  </div>
</section>

<!-- ==================== SLIDE 12: RECOMMENDATIONS & BUSINESS IMPACT ==================== -->
<section class="slide">
  <div class="hdr">
    <div class="hdr-left">
      <div class="hdr-kicker">Kết Luận &amp; Khuyến Nghị Triển Khai Sản Xuất</div>
      <div class="hdr-title">Lựa Chọn Mô Hình Theo Kịch Bản Vận Hành &amp; Tác Động Kinh Doanh</div>
    </div>
    <div class="hdr-pill">Sẵn Sàng Go-Live</div>
  </div>
  <div class="body">
    <div class="grid-3">
      <div class="card card-green">
        <div class="card-title">1. Final Model (Blend 150 biến)</div>
        <p><strong><code>M5_final_blend_150</code></strong></p>
        <ul style="margin-top:6px;">
          <li><strong>Phù hợp:</strong> Scoring toàn bộ danh mục &amp; chiến dịch diện rộng (Top 0.5% – Top 5%).</li>
          <li><strong>Ưu thế:</strong> <code>OOT ROC-AUC = 0.7756</code> cao nhất (<code>p &lt; 0.001</code>), <code>Lift@0.5% = 9.62x</code>, bắt trọn <strong>25.1% tổng người mua</strong> trong Top 5% (5.012 KH).</li>
          <li><strong>Độ ổn định:</strong> Giảm phương sai đa mô hình, giữ <code>ROC 0.771–0.785</code> qua 5/5 fold.</li>
        </ul>
      </div>
      <div class="card card-accent">
        <div class="card-title">2. Nếu chọn 1 Mô Hình Đơn (150 biến)</div>
        <p><strong><code>M4_catboost_gpu_150_tuned</code></strong></p>
        <ul style="margin-top:6px;">
          <li><strong>Phù hợp:</strong> Production chỉ muốn vận hành 1 mô hình duy nhất, đơn giản hóa bảo trì.</li>
          <li><strong>Ưu thế:</strong> Thắng mọi mô hình đơn 150 biến trên OOT (<code>PR 0.0862</code>, <code>L1 8.05x</code>, <code>L0.5 9.62x</code>, <code>L5 5.02x</code>, <code>ROC 0.7744</code>).</li>
          <li><strong>Tốc độ:</strong> <code>10.6 phút/seed</code> trên T4, <code>seed_sd</code> chỉ <code>&plusmn;0.08x</code>.</li>
        </ul>
      </div>
      <div class="card card-amber">
        <div class="card-title">3. Challenger Siêu Gọn (50 biến)</div>
        <p><strong><code>B5_catboost_gpu_top50</code></strong></p>
        <ul style="margin-top:6px;">
          <li><strong>Phù hợp:</strong> Chiến dịch telesales siêu hẹp Top-1% (1.003 KH), yêu cầu độ trễ thấp, không cần PCA embedding.</li>
          <li><strong>Ưu thế:</strong> <code>OOT Lift@1% = 8.31x</code> (~181 người mua, <code>Precision 18.0%</code>), <code>PR 0.0884</code>, xác suất chuẩn sẵn (<code>LogLoss 0.0941</code>).</li>
          <li><strong>Chi phí:</strong> Gọn gấp 3 lần số biến, chỉ <code>7.3 phút/seed</code>.</li>
        </ul>
      </div>
    </div>

    <div class="grid-2" style="margin-top:4px;">
      <div class="card">
        <div class="card-title">Tác Động Kinh Doanh Ước Tính (Trên 100.224 KH / Đợt 3 Tháng)</div>
        <ul>
          <li><strong>Top 0.5% (502 KH):</strong> <code>M5/M4</code> tìm đúng ~<strong>105 người mua</strong> (chuyển đổi ~<strong>20.9%</strong> vs 2.17% ngẫu nhiên, tăng +43 người mua so với luật tay).</li>
          <li><strong>Top 1% (1.003 KH):</strong> Tìm đúng <strong>172–181 người mua</strong> (vượt quy tắc tay <code>B1 = 122</code> thêm <strong>+50 đến +59 khách mua</strong> mỗi đợt).</li>
          <li><strong>Top 5% (5.012 KH):</strong> Phủ trọn <strong>~545 người mua (25.1% toàn ngân hàng)</strong> chỉ với 5% chi phí tiếp cận.</li>
        </ul>
      </div>
      <div class="card card-navy">
        <div class="card-title">3 Điều Kiện Bắt Buộc Khi Đưa Lên Production</div>
        <ul>
          <li><strong>1. Hiệu chuẩn xác suất (Calibration):</strong> Gắn bước <code>Isotonic Regression</code> trên Val cho <code>M5</code> / <code>M4</code> (do dùng <code>spw=3.0</code>) hoặc dùng <code>spw=1.0</code> nếu hệ thống tính Expected Value trực tiếp.</li>
          <li><strong>2. Xác minh SLA cập nhật nhãn <code>prior_*</code>:</strong> Đảm bảo nhãn tháng <code>s-1</code> đóng trước ngày <code>01</code>; nếu trễ 1 tháng, chuyển sang bản <code>prior_delay_1m</code> đã kiểm định.</li>
          <li><strong>3. Duy trì Embargo 2 tháng khi retrain:</strong> Khi retrain định kỳ hàng tháng cho <code>H=3</code>, luôn cách ly 2 tháng gần nhất chưa đóng hết cửa sổ 3 tháng.</li>
        </ul>
      </div>
    </div>
  </div>
  <div class="ftr">
    <span>Toàn bộ mã nguồn, báo cáo &amp; số liệu đã đồng bộ trên nhánh main &bull; Commit 70de580</span>
    <span>Slide 12 / 12</span>
  </div>
</section>

</body>
</html>
"""


def main() -> None:
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(), encoding="utf-8")
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
