"""Generate a minimalist, chart-first 13-slide 16:9 PDF presentation for H=3 Propensity."""
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
    data = base64.b64encode((FIG_DIR / name).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_html() -> str:
    fig1 = _b64_img("01_val_rolling_curve_baselines_vs_fe150.png")
    fig2 = _b64_img("02_oot_test_2022_11_model_comparison_ci.png")
    fig3 = _b64_img("03_warnings_verification_panel.png")

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
    padding: 54px 72px 48px 72px;
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
    font-size: 50px;
    font-weight: 800;
    line-height: 1.15;
    color: #FFFFFF;
    max-width: 10.2in;
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
    font-size: 32px;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.2;
    margin-bottom: 8px;
  }
  .subtitle {
    font-size: 15.5px;
    color: #64748B;
    line-height: 1.4;
    margin-bottom: 28px;
  }
  .content {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 22px;
  }

  /* Soft Minimalist Cards */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 26px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; }
  .grid-2-asym { display: grid; grid-template-columns: 1.35fr 0.65fr; gap: 28px; align-items: center; }
  .grid-2-bal { display: grid; grid-template-columns: 1.12fr 0.88fr; gap: 26px; align-items: center; }

  .card {
    background: #F8FAFC;
    border-radius: 16px;
    padding: 26px 28px;
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
    padding: 28px 30px;
  }
  .card-peach {
    background: #FFF7ED;
    border-radius: 16px;
    padding: 28px 30px;
  }

  .hero-num {
    font-size: 46px;
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
    font-weight: 600;
    color: #334155;
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

  /* Clean Rounded Table (Reference Style) */
  table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 14px;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(15, 23, 42, 0.05);
  }
  th {
    background: #0E192D;
    color: #FFFFFF;
    font-weight: 600;
    text-align: left;
    padding: 13px 16px;
    font-size: 13.5px;
  }
  th.num, td.num { text-align: center; }
  td {
    padding: 12px 16px;
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

  /* Horizontal Bar Chart (Reference Page 8 Style) */
  .hbar-wrap {
    background: #F8FAFC;
    border-radius: 16px;
    padding: 24px 28px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .hbar-row {
    display: grid;
    grid-template-columns: 235px 1fr 78px;
    align-items: center;
    gap: 16px;
  }
  .hbar-lbl {
    font-size: 13.5px;
    font-weight: 600;
    color: #1E293B;
    text-align: left;
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

  /* 4-Step Flow (Reference Page 12 Style) */
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
    padding: 22px 18px;
    text-align: center;
  }
  .step-card.peach { background: #FFF7ED; }
  .step-circle {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    color: #FFFFFF;
    font-size: 15px;
    font-weight: 800;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 12px;
  }
  .step-title {
    font-size: 15px;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 6px;
  }
  .step-desc {
    font-size: 13px;
    color: #475569;
    line-height: 1.4;
  }
  .step-arrow {
    font-size: 22px;
    font-weight: 700;
    color: #0D9488;
  }

  /* Timeline */
  .tl-grid {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 6px;
  }
  .tl-box {
    border-radius: 10px;
    padding: 12px 4px;
    text-align: center;
    font-size: 12px;
    font-weight: 700;
  }
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
</style>
</head>
<body>

<!-- ==================== SLIDE 1: DARK COVER ==================== -->
<section class="slide slide-dark">
  <div class="kicker k-orange">BÁO CÁO THỰC NGHIỆM DỮ LIỆU &bull; PROPENSITY H=3</div>
  <div class="cover-title" style="font-size:46px;">Mô Hình Dự Đoán Xu Hướng Mua Hàng<br/>Horizon = 3 Tháng &bull; Rolling Snapshot = 1</div>
  <div class="cover-sub">Từ EDA chuyên sâu 105,28M sự kiện, kiến trúc chống rò rỉ Embargo 2 tháng, 5 Baselines, kiểm chứng 6 cảnh báo đến tối ưu 150 đặc trưng &amp; Final Model trên GPU T4</div>
  <div class="bar-teal"></div>
</section>

<!-- ==================== SLIDE 2: OVERVIEW KPIS ==================== -->
<section class="slide">
  <div class="title">Bức tranh Dữ liệu &amp; Kết quả Chốt OOT (2022-11)</div>
  <div class="subtitle">100.224 khách hàng &times; 12 tháng nhãn (02/2022–01/2023) &nbsp;|&nbsp; Mục tiêu mua trong 3 tháng tới (H=3) &nbsp;|&nbsp; Ngân sách Top 1% (K = 1.003)</div>
  <div class="content">
    <div class="grid-3">
      <div class="card-white">
        <div class="hero-num c-teal">105,28M</div>
        <div class="hero-lbl">Sự kiện gốc toàn lịch sử</div>
        <div class="hero-sub">38,76M TRX &bull; 66,30M GEO &bull; 217K DIALOG</div>
      </div>
      <div class="card-white">
        <div class="hero-num c-dark">0,7756</div>
        <div class="hero-lbl">OOT ROC-AUC cao nhất</div>
        <div class="hero-sub">M5 Final Blend 150 biến (p &lt; 0,001 vs Top-50)</div>
      </div>
      <div class="card-white">
        <div class="hero-num c-orange">9,62&times; / 8,31&times;</div>
        <div class="hero-lbl">Lift@0.5% / Lift@1% OOT</div>
        <div class="hero-sub">Tỷ lệ mua Top 0,5% đạt ~20,9% (gấp 9,6 lần nền 2,17%)</div>
      </div>
    </div>
    <div class="banner-note">
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none"><path d="M12 2L4 5v6.09c0 5.05 3.41 9.76 8 10.91 4.59-1.15 8-5.86 8-10.91V5l-8-3z" fill="#0D9488"/><path d="M10 14.2l-2.2-2.2-1.4 1.4L10 17l7.6-7.6-1.4-1.4L10 14.2z" fill="#FFFFFF"/></svg>
      <div><strong>Kỷ luật chống rò rỉ tuyệt đối:</strong> 4/4 Unit Test PASS, cách ly cứng 2 tháng Embargo (2022-09 &amp; 2022-10), Shuffled Eval Lift = <strong>0,97&times;</strong> (0% leak).</div>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 3: MUC 1 - EDA CHARTS ==================== -->
<section class="slide">
  <div class="kicker k-teal">MỤC 1 &bull; NHẬN ĐỊNH TỪ EDA (CÓ ĐÓNG GÓP FEATURE)</div>
  <div class="title">DIALOG Gần Cutoff &amp; Đường Cong Chữ U TRX Dẫn Dắt Thiết Kế Biến</div>
  <div class="subtitle">Đối soát 15/15 kiểm tra dữ liệu gốc (0 orphan, 0 null khóa/thời gian) &rarr; chuyển hóa trực tiếp thành 5 nhóm đặc trưng</div>
  <div class="content">
    <div class="grid-2">
      <!-- Chart 1: DIALOG & Interaction Lift -->
      <div class="card">
        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:14px;">Tỷ lệ mua H=1 theo Tín hiệu DIALOG &amp; Tương tác 0-TRX</div>
        <svg viewBox="0 0 500 235" width="100%" height="225">
          <!-- Bars -->
          <text x="0" y="26" font-size="13" font-weight="700" fill="#EA580C">0-TRX &times; DIALOG gần</text>
          <rect x="175" y="10" width="260" height="24" rx="6" fill="#EA580C"/>
          <text x="444" y="27" font-size="14" font-weight="800" fill="#EA580C">9,41%</text>

          <text x="0" y="72" font-size="13" font-weight="700" fill="#0D9488">Có DIALOG tháng trước</text>
          <rect x="175" y="56" width="117" height="24" rx="6" fill="#0D9488"/>
          <text x="301" y="73" font-size="14" font-weight="800" fill="#0D9488">4,22% (3,71&times;)</text>

          <text x="0" y="118" font-size="13" font-weight="600" fill="#0F172A">Chưa từng mua + DIALOG</text>
          <rect x="175" y="102" width="107" height="24" rx="6" fill="#14B8A6"/>
          <text x="291" y="119" font-size="14" font-weight="700" fill="#0F172A">3,87%</text>

          <text x="0" y="164" font-size="13" font-weight="600" fill="#334155">Đủ cả 3 nguồn T+G+D</text>
          <rect x="175" y="148" width="55" height="24" rx="6" fill="#334155"/>
          <text x="239" y="165" font-size="14" font-weight="700" fill="#334155">1,98%</text>

          <text x="0" y="210" font-size="13" font-weight="600" fill="#64748B">Nền chung (10 cutoff)</text>
          <rect x="175" y="194" width="32" height="24" rx="6" fill="#94A3B8"/>
          <text x="216" y="211" font-size="14" font-weight="700" fill="#64748B">1,14%</text>
        </svg>
      </div>

      <!-- Chart 2: TRX U-Curve -->
      <div class="card">
        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:14px;">Quan hệ Phi đơn điệu (Chữ U) theo Số Giao dịch TRX</div>
        <svg viewBox="0 0 500 235" width="100%" height="225">
          <!-- Baseline dashed line -->
          <line x1="30" y1="95" x2="470" y2="95" stroke="#94A3B8" stroke-dasharray="5,5" stroke-width="1.5"/>
          <text x="385" y="88" font-size="11.5" fill="#64748B">Nền 1,14%</text>

          <!-- 4 U-curve vertical bars -->
          <rect x="55" y="55" width="76" height="140" rx="8" fill="#EA580C"/>
          <text x="93" y="45" text-anchor="middle" font-size="15" font-weight="800" fill="#EA580C">1,41%</text>
          <text x="93" y="216" text-anchor="middle" font-size="13" font-weight="700" fill="#1E293B">0 TRX</text>

          <rect x="165" y="120" width="76" height="75" rx="8" fill="#94A3B8"/>
          <text x="203" y="110" text-anchor="middle" font-size="15" font-weight="700" fill="#475569">0,76%</text>
          <text x="203" y="216" text-anchor="middle" font-size="13" font-weight="700" fill="#1E293B">1–9 TRX</text>

          <rect x="275" y="137" width="76" height="58" rx="8" fill="#CBD5E1"/>
          <text x="313" y="127" text-anchor="middle" font-size="15" font-weight="700" fill="#475569">0,59%</text>
          <text x="313" y="216" text-anchor="middle" font-size="13" font-weight="700" fill="#1E293B">10–99 TRX</text>

          <rect x="385" y="48" width="76" height="147" rx="8" fill="#0D9488"/>
          <text x="423" y="38" text-anchor="middle" font-size="15" font-weight="800" fill="#0D9488">1,48%</text>
          <text x="423" y="216" text-anchor="middle" font-size="13" font-weight="700" fill="#1E293B">&ge;100 TRX</text>
        </svg>
      </div>
    </div>

    <div class="grid-3">
      <div class="card-mint" style="padding:16px 20px;">
        <div class="hero-lbl c-teal">Đóng góp DIALOG</div>
        <div class="hero-sub" style="color:#1E293B;">Chỉ 5,75% mẫu nhưng chứa <strong>21,33%</strong> người mua &rarr; tạo recency mũ, burst &amp; PCA.</div>
      </div>
      <div class="card-peach" style="padding:16px 20px;">
        <div class="hero-lbl c-orange">Đuôi currency=11 (1,37e12)</div>
        <div class="hero-sub" style="color:#1E293B;">Median 18K vs Max 1,37 nghìn tỷ &rarr; dùng median, signed-log &amp; độ lệch đuôi.</div>
      </div>
      <div class="card" style="padding:16px 20px;">
        <div class="hero-lbl">Lệch tuổi nguồn &amp; Đa cộng tuyến</div>
        <div class="hero-sub" style="color:#1E293B;">GEO bắt đầu muộn 12/2021 &rarr; thêm <strong>obs_age</strong>, Entropy &amp; khử cặp r = 0,998.</div>
      </div>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 4: H=3 ZERO-LEAK DESIGN ==================== -->
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
          <td>Không sử dụng</td>
          <td class="num">—</td>
          <td class="num">—</td>
          <td>Ngăn nhãn H=3 (s+1, s+2) tràn vào OOT 2022-11</td>
        </tr>
        <tr class="row-mint">
          <td class="txt-teal">Locked OOT Test</td>
          <td>2022-11 (phủ 11..01)</td>
          <td>[02 .. 08]</td>
          <td class="num">100.224</td>
          <td class="num">2,168%</td>
          <td>Nhãn train đóng 2022-10-31; chỉ chấm điểm sau khi khóa</td>
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
    <!-- Distribution Stacked/Horizontal Bar -->
    <div class="card" style="padding:18px 24px;">
      <div style="font-size:14px; font-weight:700; color:#0F172A; margin-bottom:10px;">Cơ cấu 50 biến Baseline theo 5 nhóm nguồn</div>
      <svg viewBox="0 0 1000 58" width="100%" height="54">
        <rect x="0" y="6" width="396" height="28" rx="6" fill="#0E192D"/>
        <text x="198" y="25" text-anchor="middle" font-size="13" font-weight="700" fill="#FFFFFF">TRX: 20 biến (40%)</text>

        <rect x="402" y="6" width="256" height="28" rx="6" fill="#0D9488"/>
        <text x="530" y="25" text-anchor="middle" font-size="13" font-weight="700" fill="#FFFFFF">Prior: 13 biến (26%)</text>

        <rect x="664" y="6" width="176" height="28" rx="6" fill="#EA580C"/>
        <text x="752" y="25" text-anchor="middle" font-size="13" font-weight="700" fill="#FFFFFF">DIALOG: 9 (18%)</text>

        <rect x="846" y="6" width="132" height="28" rx="6" fill="#64748B"/>
        <text x="912" y="25" text-anchor="middle" font-size="12.5" font-weight="700" fill="#FFFFFF">GEO: 7 (14%)</text>

        <rect x="984" y="6" width="16" height="28" rx="4" fill="#CBD5E1"/>
        <text x="992" y="50" text-anchor="end" font-size="11.5" font-weight="600" fill="#64748B">+1 cold_start (2%)</text>
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
      <!-- Horizontal Bar Chart for OOT Lift@1% -->
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

      <!-- Compact Role + PR/ROC Table -->
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

<!-- ==================== SLIDE 7: DIALOG LEAKAGE AUDIT (STANDALONE) ==================== -->
<section class="slide">
  <div class="kicker k-orange">KIỂM CHỨNG CHUYÊN SÂU &bull; CẢNH BÁO RÒ RỈ DIALOG</div>
  <div class="title">Kiểm Chứng DIALOG Có Gây Leak Không? (Lag 0 / 7 / 14 / 30 Ngày)</div>
  <div class="subtitle">Giả thuyết kiểm tra: Nếu DIALOG bị leak sát ngày chốt đơn, cắt lùi 7–14 ngày sẽ làm hiệu năng sập ngay về mức <code>no_dialog</code></div>
  <div class="content">
    <div class="grid-2-bal">
      <!-- Dedicated SVG Chart comparing PR-AUC & Lift@1% across Dialog Lag -->
      <div class="card" style="padding:22px 26px;">
        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:14px;">Hiệu năng Validation (05..08) theo Độ trễ cắt DIALOG</div>
        <svg viewBox="0 0 540 250" width="100%" height="245">
          <!-- Reference vertical line for no_dialog (0.07507 -> x=305) -->
          <line x1="305" y1="8" x2="305" y2="235" stroke="#94A3B8" stroke-dasharray="4,4" stroke-width="1.5"/>
          <text x="309" y="244" font-size="11" fill="#64748B">Mốc bỏ hẳn DIALOG (0,0751)</text>

          <!-- Lag 0d -->
          <text x="0" y="30" font-size="13.5" font-weight="700" fill="#0F172A">Full (Lag 0 ngày)</text>
          <rect x="145" y="14" width="290" height="26" rx="6" fill="#0E192D"/>
          <text x="444" y="32" font-size="13.5" font-weight="800" fill="#0F172A">0,0785 (4,99&times;)</text>

          <!-- Lag 7d -->
          <text x="0" y="76" font-size="13.5" font-weight="700" fill="#0D9488">Cắt lùi 7 ngày</text>
          <rect x="145" y="60" width="215" height="26" rx="6" fill="#0D9488"/>
          <text x="369" y="78" font-size="13.5" font-weight="800" fill="#0D9488">0,0761 (4,91&times;)</text>

          <!-- Lag 14d -->
          <text x="0" y="122" font-size="13.5" font-weight="700" fill="#0D9488">Cắt lùi 14 ngày</text>
          <rect x="145" y="106" width="222" height="26" rx="6" fill="#14B8A6"/>
          <text x="376" y="124" font-size="13.5" font-weight="800" fill="#0D9488">0,0763 (5,00&times;)</text>

          <!-- Lag 30d -->
          <text x="0" y="168" font-size="13.5" font-weight="600" fill="#475569">Cắt lùi 30 ngày</text>
          <rect x="145" y="152" width="163" height="26" rx="6" fill="#64748B"/>
          <text x="317" y="170" font-size="13.5" font-weight="700" fill="#475569">0,0751 (4,82&times;)</text>

          <!-- No Dialog -->
          <text x="0" y="214" font-size="13.5" font-weight="700" fill="#EA580C">Bỏ hẳn DIALOG</text>
          <rect x="145" y="198" width="160" height="26" rx="6" fill="#EA580C"/>
          <text x="314" y="216" font-size="13.5" font-weight="800" fill="#EA580C">0,0751 (-4,4%)</text>
        </svg>
      </div>

      <!-- 2 Big Takeaway Cards -->
      <div style="display:flex; flex-direction:column; gap:18px;">
        <div class="card-mint">
          <div class="kicker k-teal" style="margin-bottom:6px;">KẾT LUẬN: KHÔNG CÓ LEAK SÁT CUTOFF</div>
          <div class="hero-num c-teal">4,91–5,00&times;</div>
          <div class="hero-lbl">Lift@1% giữ nguyên khi cắt lùi 7–14 ngày</div>
          <div class="hero-sub" style="color:#1E293B;">PR-AUC vẫn cao hơn hẳn mốc <code>no_dialog</code> (0,0761–0,0763 vs 0,0751; ROC 0,723 vs 0,717).</div>
        </div>
        <div class="card-white">
          <div class="kicker k-orange" style="margin-bottom:6px;">QUY LUẬT SUY GIẢM TỰ NHIÊN</div>
          <div class="hero-lbl">Tín hiệu phai dần mượt theo độ trễ 0d &rarr; 14d &rarr; 30d</div>
          <div class="hero-sub" style="color:#334155;">Đúng bản chất hành vi tư vấn bán hàng (khách quan tâm gọi hỏi trước khi mua 1–3 tuần) &rarr; <strong>An toàn giữ toàn bộ DIALOG vào Stage 2</strong>.</div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 8: OTHER 5 WARNINGS AUDIT ==================== -->
<section class="slide">
  <div class="kicker k-orange">KIỂM CHỨNG RỦI RO &bull; 5 CẢNH BÁO KỸ THUẬT CÒN LẠI</div>
  <div class="title">Kiểm Chứng 5 Cảnh Báo Còn Lại: 0% Rò Rỉ &amp; Tối Ưu Tenure cap365</div>
  <div class="subtitle">Đối chứng Prior delay, Label overlap, Múi giờ MSK/UTC, Tenure left-censor, Geohash drift &amp; Shuffled Sanity</div>
  <div class="content">
    <div class="grid-2-asym">
      <div class="img-clean" style="height:4.7in;">
        <img src="__FIG3__" alt="6 Warnings Verification" />
      </div>
      <div style="display:flex; flex-direction:column; gap:14px;">
        <div class="card-mint" style="padding:16px 20px;">
          <div class="hero-lbl c-teal">Overlap &amp; Múi giờ: AN TOÀN</div>
          <div class="hero-sub" style="color:#1E293B;">Rolling 02..05 (0,0843) &le; Non-overlap (0,0867). MSK vs UTC lệch &lt; 0,00001.</div>
        </div>
        <div class="card" style="padding:16px 20px;">
          <div class="hero-lbl">Tenure cap365 &amp; Geohash Drift</div>
          <div class="hero-sub" style="color:#1E293B;">Chặn trần 365 ngày <strong>tăng Val PR &rarr; 0,0794</strong> (5,09&times;). Geohash thô drift nhẹ &rarr; đổi sang Entropy.</div>
        </div>
        <div class="card-peach" style="padding:16px 20px;">
          <div class="hero-lbl c-orange">Shuffled Sanity &amp; SLA Prior</div>
          <div class="hero-sub" style="color:#1E293B;">Shuffled Train ROC &approx; <strong>0,50</strong>; Eval OOT Lift = <strong>0,97&times;</strong>. Trễ nhãn Prior 1m giảm -3,9% PR.</div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 7: MUC 3 - FEATURE ENGINEERING ==================== -->
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

<!-- ==================== SLIDE 8: MUC 4 - FEATURE SELECTION ==================== -->
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

    <!-- Native Bar Chart of Top 8 Selected Features -->
    <div class="hbar-wrap" style="padding:18px 24px; gap:10px;">
      <div style="font-size:13.5px; font-weight:700; color:#0F172A;">Top 8 Đặc trưng Dẫn đầu Điểm Đồng thuận Val (Màu xanh ngọc = Biến mới Stage 2)</div>
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

<!-- ==================== SLIDE 9: MUC 5 - EXPERIMENTS & FINAL MODEL ==================== -->
<section class="slide">
  <div class="kicker k-teal">MỤC 5 &bull; THỬ NGHIỆM, BÀI HỌC &amp; KIẾN TRÚC FINAL MODEL</div>
  <div class="title">Tối Ưu GPU 150 Biến, Chứng Minh Giá Trị <code>prior_*</code> &amp; Chốt Final Blend</div>
  <div class="subtitle">Quét 12 cấu hình GPU (826s) + 48 tổ hợp Blend thuần trên Val &nbsp;|&nbsp; Mọi mô hình GPU chạy dưới 11 phút/seed</div>
  <div class="content">
    <div class="grid-3">
      <div class="card-white">
        <div class="kicker k-teal" style="margin-bottom:6px;">FE-150 TRÊN VALIDATION</div>
        <div class="hero-num c-teal">+3,2–5,8%</div>
        <div class="hero-sub">Cả 4 họ mô hình đều tăng PR-AUC trên Val; M5 Blend dẫn đầu <strong>0,08175</strong>.</div>
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

    <!-- Clean Final Model Architecture Bar -->
    <div style="background:#0E192D; border-radius:16px; padding:22px 28px; color:#FFFFFF;">
      <div class="kicker k-mint" style="margin-bottom:14px;">KIẾN TRÚC FINAL MODEL &bull; M5_FINAL_BLEND_150 (PROBABILITY BLEND KHÓA TRÊN VAL)</div>
      <div style="display:grid; grid-template-columns:1fr 30px 1.6fr 30px 1fr; align-items:center; gap:10px;">
        <div style="background:rgba(255,255,255,0.08); border-radius:12px; padding:14px; text-align:center;">
          <div style="font-weight:700; color:#2DD4BF; font-size:14px; margin-bottom:4px;">150 Đặc trưng</div>
          <div style="font-size:12.5px; color:#CBD5E1;">129 bảng + 21 PCA</div>
        </div>
        <div style="text-align:center; color:#2DD4BF; font-size:20px; font-weight:800;">&rarr;</div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px;">
          <div style="background:rgba(13,148,136,0.28); border:1px solid #2DD4BF; border-radius:10px; padding:12px; text-align:center;">
            <div style="font-size:20px; font-weight:800; color:#2DD4BF;">50%</div>
            <div style="font-size:12px; color:#F8FAFC;">CatBoost GPU</div>
          </div>
          <div style="background:rgba(255,255,255,0.10); border:1px solid rgba(255,255,255,0.22); border-radius:10px; padding:12px; text-align:center;">
            <div style="font-size:20px; font-weight:800; color:#FFFFFF;">30%</div>
            <div style="font-size:12px; color:#E2E8F0;">LightGBM</div>
          </div>
          <div style="background:rgba(234,88,12,0.22); border:1px solid #FB923C; border-radius:10px; padding:12px; text-align:center;">
            <div style="font-size:20px; font-weight:800; color:#FB923C;">20%</div>
            <div style="font-size:12px; color:#F8FAFC;">XGBoost GPU</div>
          </div>
        </div>
        <div style="text-align:center; color:#2DD4BF; font-size:20px; font-weight:800;">&rarr;</div>
        <div style="background:rgba(13,148,136,0.22); border:1px solid #2DD4BF; border-radius:12px; padding:14px; text-align:center;">
          <div style="font-weight:700; color:#FFFFFF; font-size:14px; margin-bottom:4px;">Blend + Isotonic</div>
          <div style="font-size:12.5px; color:#CCFBF1;">ROC 0,7756 &bull; R@5% 25,1%</div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 10: VALIDATION ROLLING CHART ==================== -->
<section class="slide">
  <div class="kicker k-teal">ĐỐI CHỨNG TRÊN VALIDATION (400.896 KHÁCH–THÁNG)</div>
  <div class="title">FE-150 Cải Thiện Ổn Định Qua Cả 4 Tháng Trượt Validation</div>
  <div class="subtitle">Khi lịch sử huấn luyện tích lũy từ 1 lên 4 tháng, M5 Blend &amp; M4 CatBoost 150 dẫn đầu nhất quán</div>
  <div class="content">
    <div class="img-clean" style="height:4.9in;">
      <img src="__FIG1__" alt="Validation Rolling Curve" />
    </div>
  </div>
</section>

<!-- ==================== SLIDE 11: MUC 6 - LOCKED OOT TEST ==================== -->
<section class="slide">
  <div class="kicker k-teal">MỤC 6 &bull; KẾT QUẢ CUỐI CÙNG (LOCKED OOT TEST 2022-11)</div>
  <div class="title">Hiệu Năng Toàn Bộ Mô Hình Trên Tập OOT Khóa (95% Bootstrap CI)</div>
  <div class="subtitle">100.224 khách hàng &bull; 2.173 ca mua (2,168%) &nbsp;|&nbsp; M5 Blend thắng tuyệt đối ROC-AUC (p &lt; 0,001), hòa B5 về Lift@1% (p = 0,636)</div>
  <div class="content" style="gap:16px;">
    <div class="img-clean" style="height:2.95in;">
      <img src="__FIG2__" alt="OOT Model Comparison" />
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
          <td><strong>M4_catboost_gpu_150_tuned (Đơn #1)</strong></td><td class="num">150</td>
          <td class="num">0,0862 [0,079; 0,095]</td><td class="num">8,05&times; [7,16; 9,11]</td>
          <td class="num txt-teal">9,62&times;</td><td class="num txt-teal">5,02&times; (25,1%)</td><td class="num">0,7744</td>
        </tr>
        <tr class="row-mint">
          <td class="txt-teal">M5_final_blend_150 (Final Model)</td><td class="num">150</td>
          <td class="num">0,0860 [0,079; 0,095]</td><td class="num">7,92&times; [7,03; 8,98]</td>
          <td class="num txt-teal">9,62&times;</td><td class="num txt-teal">5,01&times; (25,1%)</td><td class="num txt-teal">0,7756</td>
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

<!-- ==================== SLIDE 12: SINGLE VS BLEND & 5-FOLD STABILITY ==================== -->
<section class="slide">
  <div class="kicker k-teal">LỰA CHỌN TRIỂN KHAI &bull; ỔN ĐỊNH 5 FOLD OOT</div>
  <div class="title">Phân Vai Triển Khai: Final Blend Diện Rộng vs CatBoost Đơn</div>
  <div class="subtitle">Cả 5/5 fold trên OOT đều giữ ROC-AUC ổn định trong dải hẹp 0,771 – 0,785</div>
  <div class="content">
    <div class="grid-2">
      <div class="card-mint">
        <div class="kicker k-teal" style="margin-bottom:8px;">FINAL MODEL (SCORING &amp; TOP 5%)</div>
        <div class="hero-num c-teal">0,7756 ROC</div>
        <div class="hero-lbl">M5_final_blend_150 (150 biến)</div>
        <div class="hero-sub" style="color:#1E293B; margin-top:10px;">
          &bull; Bắt trọn <strong>~545 người mua (25,1%)</strong> trong Top 5%.<br/>
          &bull; Top 0,5% (502 KH) đạt <strong>9,62&times;</strong> (~20,9% chuyển đổi).
        </div>
      </div>
      <div class="card-peach">
        <div class="kicker k-orange" style="margin-bottom:8px;">NẾU CHỌN 1 MÔ HÌNH ĐƠN</div>
        <div class="hero-num c-orange">8,05&times; / 8,31&times;</div>
        <div class="hero-lbl">M4_catboost_150 &nbsp;|&nbsp; B5_catboost_top50</div>
        <div class="hero-sub" style="color:#1E293B; margin-top:10px;">
          &bull; <strong>M4 (150v):</strong> Đơn #1 toàn diện (ROC 0,7744, L@5% 5,02&times;).<br/>
          &bull; <strong>B5 (50v):</strong> Gọn nhẹ, LogLoss 0,0941, vượt luật tay +59 đơn.
        </div>
      </div>
    </div>

    <!-- 5-Fold Stability Visual Strip -->
    <div class="card" style="padding:18px 26px;">
      <div style="font-size:14px; font-weight:700; color:#0F172A; margin-bottom:12px;">Độ ổn định OOT theo 5 Fold Độc lập (M5_final_blend_150)</div>
      <svg viewBox="0 0 1000 82" width="100%" height="78">
        <g font-family="Segoe UI, Arial" font-size="13">
          <rect x="0" y="6" width="184" height="68" rx="10" fill="#FFFFFF" stroke="#E2E8F0"/>
          <text x="92" y="28" text-anchor="middle" font-weight="700" fill="#64748B">Fold 0</text>
          <text x="92" y="50" text-anchor="middle" font-size="16" font-weight="800" fill="#0D9488">ROC 0,772</text>
          <text x="92" y="66" text-anchor="middle" font-size="11.5" fill="#475569">Lift@1% 9,16&times;</text>

          <rect x="204" y="6" width="184" height="68" rx="10" fill="#FFFFFF" stroke="#E2E8F0"/>
          <text x="296" y="28" text-anchor="middle" font-weight="700" fill="#64748B">Fold 1</text>
          <text x="296" y="50" text-anchor="middle" font-size="16" font-weight="800" fill="#0D9488">ROC 0,773</text>
          <text x="296" y="66" text-anchor="middle" font-size="11.5" fill="#475569">Lift@5% 5,45&times;</text>

          <rect x="408" y="6" width="184" height="68" rx="10" fill="#FFFFFF" stroke="#E2E8F0"/>
          <text x="500" y="28" text-anchor="middle" font-weight="700" fill="#64748B">Fold 2</text>
          <text x="500" y="50" text-anchor="middle" font-size="16" font-weight="800" fill="#0D9488">ROC 0,778</text>
          <text x="500" y="66" text-anchor="middle" font-size="11.5" fill="#475569">Lift@1% 9,01&times;</text>

          <rect x="612" y="6" width="184" height="68" rx="10" fill="#FFFFFF" stroke="#E2E8F0"/>
          <text x="704" y="28" text-anchor="middle" font-weight="700" fill="#64748B">Fold 3</text>
          <text x="704" y="50" text-anchor="middle" font-size="16" font-weight="800" fill="#0D9488">ROC 0,785</text>
          <text x="704" y="66" text-anchor="middle" font-size="11.5" fill="#475569">Lift@5% 5,54&times;</text>

          <rect x="816" y="6" width="184" height="68" rx="10" fill="#FFFFFF" stroke="#E2E8F0"/>
          <text x="908" y="28" text-anchor="middle" font-weight="700" fill="#64748B">Fold 4</text>
          <text x="908" y="50" text-anchor="middle" font-size="16" font-weight="800" fill="#0D9488">ROC 0,771</text>
          <text x="908" y="66" text-anchor="middle" font-size="11.5" fill="#475569">Lift@1% 8,16&times;</text>
        </g>
      </svg>
    </div>
  </div>
</section>

<!-- ==================== SLIDE 13: DARK CLOSING ==================== -->
<section class="slide slide-dark-center">
  <div class="kicker k-mint">SẴN SÀNG TRIỂN KHAI SẢN XUẤT &bull; PROPENSITY H=3</div>
  <div class="cover-title" style="font-size:40px; max-width:9.8in; margin-bottom:0;">
    Triển khai M5 Final Blend (0,7756 ROC) cho Diện rộng &amp; CatBoost Đơn (8,05&times;–8,31&times; Lift) cho Top-1%
  </div>
  <div class="bar-orange"></div>
  <div class="cover-sub" style="margin-bottom:0; font-size:17px;">
    3 điều kiện Go-Live: Hiệu chuẩn Isotonic trên Val &bull; Xác minh SLA nhãn prior_* trước ngày 01 &bull; Duy trì Embargo 2 tháng khi retrain định kỳ
  </div>
</section>

</body>
</html>
"""
    return html.replace("__FIG1__", fig1).replace("__FIG2__", fig2).replace("__FIG3__", fig3)


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
