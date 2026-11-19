"""Generate an 8-page clean, minimalist PDF presentation prioritizing the 4 quoted insights + new TRX/GEO discoveries."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(r"c:\Users\admin\Documents\Thực tập VSF\new propensity")
OUT_DIR = ROOT / "outputs/eda_insights_20261119"
SLIDE_DIR = OUT_DIR / "slides_clean"
SLIDE_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT_DIR / "MBD_MINI_DEEP_INSIGHTS_DECK.pdf"

# Palette matched to the user's reference template
NAVY = "#0f294a"
BLUE_DARK = "#1d4ed8"
BLUE_MAIN = "#2b7de9"
BLUE_MID = "#5a9cf2"
BLUE_LIGHT = "#93c5fd"
BLUE_PILL = "#e2ecf8"
KPI_BG = "#eaf3fc"
TEAL = "#0d9488"
TEAL_MID = "#2dd4bf"
ORANGE = "#e06d2b"
AMBER = "#f59e0b"
SLATE_DARK = "#1e293b"
SLATE_MID = "#64748b"
SLATE_LIGHT = "#cbd5e1"
WHITE = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8,
})


def new_clean_slide(header_title, subtitle_text=None):
    """Create a spacious, uncluttered 16:9 slide matching the user's card header + clean white canvas style."""
    fig = plt.figure(figsize=(16, 9), dpi=180, facecolor=WHITE)
    bg = fig.add_axes([0, 0, 1, 1], frameon=False)
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.axis("off")

    # Top dark navy rounded header bar (like Image 1 & Image 2)
    header = FancyBboxPatch(
        (0.04, 0.865), 0.92, 0.088,
        boxstyle="round,pad=0.004,rounding_size=0.020",
        facecolor=NAVY, edgecolor="none", zorder=2
    )
    bg.add_patch(header)
    bg.text(
        0.065, 0.909, header_title,
        fontsize=20, fontweight="bold", color=WHITE, va="center", ha="left", zorder=3
    )
    if subtitle_text:
        bg.text(
            0.045, 0.822, subtitle_text,
            fontsize=12, color=SLATE_MID, va="center", ha="left", zorder=3
        )
    return fig, bg


def add_mini_kpi(bg, x, y, w, h, big_val, label, val_color=NAVY):
    """Minimalist 2-line KPI box (matching Image 1: big number + 1 short line)."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.020",
        facecolor=KPI_BG, edgecolor="none", zorder=2
    )
    bg.add_patch(box)
    bg.text(x + w / 2, y + h * 0.62, big_val, fontsize=24, fontweight="bold", color=val_color, ha="center", va="center", zorder=3)
    bg.text(x + w / 2, y + h * 0.25, label, fontsize=11.5, color=SLATE_DARK, ha="center", va="center", zorder=3)


def add_bottom_pill(bg, line1, line2=None):
    """Soft light-blue rounded pill at the bottom (matching user's PDF template pages 6, 7, 9, 11, 12)."""
    pill = FancyBboxPatch(
        (0.08, 0.048), 0.84, 0.105,
        boxstyle="round,pad=0.004,rounding_size=0.032",
        facecolor=BLUE_PILL, edgecolor="none", zorder=2
    )
    bg.add_patch(pill)
    if line2:
        bg.text(0.50, 0.118, line1, fontsize=13.2, fontweight="bold", color=NAVY, ha="center", va="center", zorder=3)
        bg.text(0.50, 0.078, line2, fontsize=12.2, color=SLATE_DARK, ha="center", va="center", zorder=3)
    else:
        bg.text(0.50, 0.100, line1, fontsize=14, fontweight="bold", color=NAVY, ha="center", va="center", zorder=3)


def style_clean_ax(ax, title=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors=SLATE_DARK, labelsize=10.5)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY, pad=14)


def build_deck():
    slides = []

    # =========================================================================
    # SLIDE 1: A. GIẢI MÃ ĐƯỜNG CONG CHỮ U CỦA GIAO DỊCH (QUOTE 1)
    # =========================================================================
    fig, bg = new_clean_slide(
        "A. Giải mã đường cong chữ U: Tại sao nhóm '0 Giao dịch' có tỷ lệ mua cao?",
        "1.002.240 quan sát khách–tháng (10 mốc dự báo 02–11/2022) · Tỷ lệ mua tháng tới (H=1)"
    )

    # Left chart: Simple U-shape bar chart
    ax1 = fig.add_axes([0.06, 0.25, 0.35, 0.49])
    style_clean_ax(ax1, "Tỷ lệ mua theo Tổng số giao dịch trước cutoff")
    u_x = ["0 giao dịch", "1 – 9", "10 – 99", "≥ 100"]
    u_y = [1.41, 0.76, 0.59, 1.48]
    u_cols = [ORANGE, BLUE_LIGHT, BLUE_MID, BLUE_MAIN]
    bars = ax1.bar(range(4), u_y, color=u_cols, width=0.52)
    ax1.axhline(1.14, color=SLATE_MID, linestyle="--", linewidth=1.3, label="Mức chung: 1,14%")
    for b, val in zip(bars, u_y):
        ax1.text(b.get_x() + b.get_width() / 2, val + 0.06, f"{val:.2f}%", ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=NAVY)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(u_x, fontsize=10.5, fontweight="bold")
    ax1.set_ylim(0, 1.95)
    ax1.set_ylabel("Tỷ lệ mua tháng tới (%)", fontsize=11, color=SLATE_DARK)
    ax1.legend(frameon=False, loc="upper center", fontsize=10.5)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")

    # Subtle vertical divider
    bg.plot([0.44, 0.44], [0.22, 0.77], color="#e2e8f0", linewidth=1.5)

    # Right chart: Breakdown of TRX=0 (84,670 rows) + Pre-cutoff observable signals
    ax2 = fig.add_axes([0.63, 0.25, 0.31, 0.49])
    style_clean_ax(ax2, "Bóc tách bên trong nhóm '0 Giao dịch' (84.670 dòng)")
    groups_z = [
        "Ngủ đông thực sự\n(85,4% nhóm 0 TRX · 72.304 dòng)",
        "Có vị trí GEO ≥4 vùng trước cutoff\n(23,5% nhóm 0 TRX · 19.889 dòng)",
        "Kích hoạt TRX mới (chưa DIALOG)\n(11,2% nhóm 0 TRX · 9.476 dòng)",
        "Có Hội thoại tháng trước (trước cutoff)\n(3,4% nhóm 0 TRX · 2.890 dòng)"
    ]
    rates_z = [0.56, 1.66, 5.45, 9.41]
    shares_z = ["85,4% nhóm", "23,5% nhóm", "11,2% nhóm", "3,4% nhóm"]
    cols_z = [SLATE_LIGHT, BLUE_MID, TEAL, ORANGE]
    bars2 = ax2.barh(range(4), rates_z, color=cols_z, height=0.50)
    ax2.axvline(1.14, color=SLATE_MID, linestyle="--", linewidth=1.3)
    for b, val, sh in zip(bars2, rates_z, shares_z):
        ax2.text(
            val + 0.22, b.get_y() + b.get_height() / 2,
            f"{val:.2f}%  ({sh})",
            va="center", ha="left", fontsize=10.8, fontweight="bold", color=NAVY
        )
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(groups_z, fontsize=9.8, fontweight="bold")
    ax2.set_xlim(0, 13.5)
    ax2.set_xlabel("Tỷ lệ mua tháng tới (%)", fontsize=11, color=SLATE_DARK)
    ax2.grid(axis="x", alpha=0.25, linestyle=":")

    add_bottom_pill(
        bg,
        "85,4% nhóm '0 giao dịch' là khách ngủ đông (chỉ mua 0,56%); 14,6% còn lại (3,4% có Hội thoại + 11,2% kích hoạt TRX) tạo ra 66,2% lượt mua.",
        "Trước mốc dự báo, dùng cờ Hội thoại tháng trước (chiếm 3,4% · mua 9,41%) và Vị trí GEO ≥4 vùng (chiếm 23,5% · mua 1,66%) để tách khách tiềm năng."
    )
    slides.append(fig)

    # =========================================================================
    # SLIDE 2: B. VAI TRÒ CỦA TUỔI LỊCH SỬ (TENURE) & TỐC ĐỘ GIAO DỊCH (QUOTE 2A)
    # =========================================================================
    fig, bg = new_clean_slide(
        "B. Hiệu ứng Tuổi lịch sử (Tenure): Cùng số giao dịch nhưng ý nghĩa trái ngược",
        "Giải mã sai lầm khi chỉ dùng tổng số đếm tích lũy (trx_hist) mà bỏ qua số tháng quan sát"
    )

    # Left chart: Same TRX 10-99 band across 3 Tenure bands (0 prior buys, no recent dialog)
    ax1 = fig.add_axes([0.07, 0.25, 0.38, 0.49])
    style_clean_ax(ax1, "Cùng 10–99 giao dịch theo Số tháng lịch sử")
    t_labels = ["1 – 6 tháng", "7 – 12 tháng", "≥ 13 tháng"]
    t_rates = [1.14, 0.70, 0.31]
    t_cols = [TEAL, BLUE_MID, SLATE_LIGHT]
    bars1 = ax1.bar(range(3), t_rates, color=t_cols, width=0.46)
    for b, val in zip(bars1, t_rates):
        ax1.text(b.get_x() + b.get_width() / 2, val + 0.04, f"{val:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold", color=NAVY)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(t_labels, fontsize=11, fontweight="bold")
    ax1.set_xlabel("Số tháng lịch sử quan sát trước cutoff", fontsize=10.5, color=SLATE_DARK, labelpad=8)
    ax1.set_ylim(0, 1.48)
    ax1.set_ylabel("Tỷ lệ mua tháng tới (%)", fontsize=11, color=SLATE_DARK)
    ax1.text(0.50, 0.88, "Cố định: 0 lần mua trước & Không có DIALOG ở T−1\nChênh lệch 3,7 lần cho cùng mức 10–99 TRX", transform=ax1.transAxes, ha="center", fontsize=10, color=SLATE_MID)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")

    bg.plot([0.49, 0.49], [0.22, 0.77], color="#e2e8f0", linewidth=1.5)

    # Right chart: Short-term TRX ratio (T-1 TRX / personal historical monthly mean)
    ax2 = fig.add_axes([0.55, 0.25, 0.38, 0.49])
    style_clean_ax(ax2, "Tỷ số Giao dịch tháng T−1 / Trung bình tháng trước đó")
    m_labels = [
        "0×\n(14,8%)",
        "< 0,5×\n(10,8%)",
        "0,5× – 1,5×\n(52,1%)",
        "> 1,5×\n(12,7%)",
        "Trước T−1 = 0\nT−1 > 0 (1,1%)"
    ]
    m_rates = [0.25, 0.75, 1.19, 1.88, 4.21]
    m_cols = [SLATE_LIGHT, BLUE_LIGHT, BLUE_MID, BLUE_MAIN, ORANGE]
    bars2 = ax2.bar(range(5), m_rates, color=m_cols, width=0.50)
    ax2.axhline(1.14, color=SLATE_MID, linestyle="--", linewidth=1.3)
    for b, val in zip(bars2, m_rates):
        ax2.text(b.get_x() + b.get_width() / 2, val + 0.12, f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color=NAVY)
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(m_labels, fontsize=10, fontweight="bold")
    ax2.set_xlabel("Tỷ số TRX(T−1) / Trung bình tháng giai đoạn trước (kèm % tổng tệp)", fontsize=10.5, color=SLATE_DARK, labelpad=8)
    ax2.set_ylim(0, 5.1)
    ax2.set_ylabel("Tỷ lệ mua tháng tới (%)", fontsize=11, color=SLATE_DARK)
    ax2.grid(axis="y", alpha=0.25, linestyle=":")

    add_bottom_pill(
        bg,
        "Cùng 10–99 giao dịch nhưng tích lũy trong 1–6 tháng có tỷ lệ mua 1,14%, còn rải rác trong ≥13 tháng chỉ đạt 0,31%.",
        "Bắt buộc phải đưa Số tháng lịch sử (history_age_months), Trung bình TRX/tháng và Tỷ số TRX(T−1) / TB trước đó vào mô hình."
    )
    slides.append(fig)

    # =========================================================================
    # SLIDE 3: C. QUY LUẬT BÃO HÒA CỦA HỘI THOẠI & LOẠI TRỪ NGHỊCH LÝ SIMPSON (QUOTE 2B)
    # =========================================================================
    fig, bg = new_clean_slide(
        "C. Kiểm định đa tầng: Tác động của Hội thoại (DIALOG) theo từng tầng TRX và Lịch sử mua",
        "Phân tầng đồng thời theo (Số lần mua trước × Số tháng lịch sử ≥13 tháng × Tổng số TRX) để loại trừ nhiễu"
    )

    # Left main grouped bar chart
    ax1 = fig.add_axes([0.07, 0.25, 0.54, 0.49])
    style_clean_ax(ax1, "Tỷ lệ mua H=1 theo trạng thái DIALOG ở T−1 (Nhóm lịch sử ≥13 tháng)")
    strata_names = [
        "0 lần mua trước\n1 – 9 TRX",
        "0 lần mua trước\n10 – 99 TRX",
        "0 lần mua trước\n≥ 100 TRX",
        "≥ 1 lần mua trước\n≥ 100 TRX"
    ]
    no_d_rates = [0.20, 0.31, 1.12, 3.84]
    yes_d_rates = [3.35, 3.29, 3.13, 5.00]
    lifts_str = ["16,6×", "10,6×", "2,8×", "1,3×"]
    xx = np.arange(4)
    b1 = ax1.bar(xx - 0.18, no_d_rates, width=0.34, color=SLATE_LIGHT, label="0 DIALOG ở tháng T−1")
    b2 = ax1.bar(xx + 0.18, yes_d_rates, width=0.34, color=TEAL, label="≥ 1 DIALOG ở tháng T−1")
    for b, v in zip(b1, no_d_rates):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.10, f"{v:.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=SLATE_MID)
    for b, v, lf in zip(b2, yes_d_rates, lifts_str):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}%\n({lf})", ha="center", va="bottom", fontsize=10.8, fontweight="bold", color=TEAL)
    ax1.set_xticks(xx)
    ax1.set_xticklabels(strata_names, fontsize=10.5, fontweight="bold")
    ax1.set_ylim(0, 6.2)
    ax1.set_ylabel("Tỷ lệ mua tháng tới (%)", fontsize=11, color=SLATE_DARK)
    ax1.legend(frameon=False, loc="upper left", fontsize=10.5)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")

    # Right KPI column
    add_mini_kpi(bg, 0.66, 0.61, 0.28, 0.135, "10,6× – 16,6×", "Hệ số Lift ở nhóm 0 mua & 1–99 TRX", val_color=ORANGE)
    add_mini_kpi(bg, 0.66, 0.44, 0.28, 0.135, "5,00% (H3: 10,4%)", "Tỷ lệ mua ở nhóm ≥1 lần mua & ≥100 TRX", val_color=TEAL)
    add_mini_kpi(bg, 0.66, 0.27, 0.28, 0.135, "Bác bỏ Simpson", "Tăng tỷ lệ mua ở 100% tổ hợp phân tầng", val_color=NAVY)

    add_bottom_pill(
        bg,
        "Tín hiệu Hội thoại không phải là hệ quả giả của việc khách giao dịch nhiều hay đã mua trước đó (loại trừ Nghịch lý Simpson).",
        "Hội thoại tạo bước nhảy tương đối lớn nhất (10,6–16,6×) ở nhóm 0 lần mua & 1–99 TRX, và đạt tỷ lệ mua tuyệt đối cao nhất (5,0%) ở nhóm ≥1 lần mua & ≥100 TRX."
    )
    slides.append(fig)

    # =========================================================================
    # SLIDE 4: D. PHÂN RÃ SHIFT-SHARE SỰ SUY GIẢM TỶ LỆ MUA NĂM 2022 (QUOTE 3)
    # =========================================================================
    fig, bg = new_clean_slide(
        "D. Tại sao tỷ lệ mua giảm một nửa từ tháng 04/2022 (1,69%) xuống 11/2022 (0,87%)?",
        "Phân rã Shift-Share giữa tác động dịch chuyển cơ cấu khách hàng và tác động giảm tỷ lệ mua nội nhóm"
    )

    # Left chart: Shift-share decomposition bars
    ax1 = fig.add_axes([0.07, 0.26, 0.40, 0.48])
    style_clean_ax(ax1, "Phân rã mức giảm −0,83% (04/2022 → 11/2022)")
    ss_labels = [
        "Dịch chuyển cơ cấu\n(Tăng phủ DIALOG)",
        "Tương tác pha loãng\n(Mở rộng tệp)",
        "Giảm tỷ lệ nội nhóm\n(90% mức giảm)",
        "Tổng chênh lệch\n(1,69% → 0,87%)"
    ]
    ss_vals = [0.214, -0.297, -0.746, -0.829]
    ss_cols = [TEAL, AMBER, ORANGE, NAVY]
    bars1 = ax1.bar(range(4), ss_vals, color=ss_cols, width=0.50)
    ax1.axhline(0, color=SLATE_DARK, linewidth=1.0)
    for b, v in zip(bars1, ss_vals):
        y_pos = v + 0.04 if v >= 0 else v - 0.09
        ax1.text(b.get_x() + b.get_width() / 2, y_pos, f"{v:+.2f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=11.5, fontweight="bold", color=NAVY)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(ss_labels, fontsize=8.8, fontweight="bold")
    ax1.set_ylim(-1.05, 0.45)
    ax1.set_ylabel("Đóng góp vào thay đổi tỷ lệ mua (điểm %)", fontsize=10.5, color=SLATE_DARK)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")

    bg.plot([0.50, 0.50], [0.22, 0.77], color="#e2e8f0", linewidth=1.5)

    # Right chart: Monthly DIALOG coverage surge in 06/2022 vs Monthly Lift stability
    monthly_seg = pd.read_csv(OUT_DIR / "monthly_segment_cards.csv")
    dlg_m = monthly_seg[(monthly_seg["feature"] == "dialog_recent") & (monthly_seg["group"] == "True") & (monthly_seg["horizon"] == 1)].copy()
    dlg_m["m_str"] = pd.to_datetime(dlg_m["month"]).dt.strftime("%m/%y")

    ax2 = fig.add_axes([0.55, 0.26, 0.37, 0.48])
    style_clean_ax(ax2, "Độ phủ DIALOG tăng gấp đôi từ 06/22 nhưng Lift vẫn bền")
    xm = np.arange(len(dlg_m))
    cov_pct = dlg_m["population_share"].to_numpy() * 100
    lift_arr = dlg_m["lift"].to_numpy()
    b_cov = ax2.bar(xm, cov_pct, color=BLUE_LIGHT, width=0.52, alpha=0.75, label="% Khách có DIALOG tháng trước")
    ax2.set_ylabel("% Khách có DIALOG tháng trước", fontsize=10.5, color=BLUE_MAIN)
    ax2.set_ylim(0, 12.5)
    ax2.set_xticks(xm)
    ax2.set_xticklabels(dlg_m["m_str"], fontsize=9.5)

    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.spines["left"].set_visible(False)
    ax2_r.spines["right"].set_color(TEAL)
    ax2_r.plot(xm, lift_arr, "o-", color=TEAL, linewidth=2.5, markersize=7, label="Hệ số Lift mua hàng (3,3× – 5,1×)")
    for i, lf in enumerate(lift_arr):
        ax2_r.text(i, lf + 0.22, f"{lf:.1f}×", ha="center", va="bottom", fontsize=9, fontweight="bold", color=TEAL)
    ax2_r.set_ylim(0, 6.5)
    ax2_r.set_ylabel("Hệ số Lift so với nền cùng tháng", fontsize=10.5, color=TEAL)

    add_bottom_pill(
        bg,
        "90% mức sụt giảm tỷ lệ mua (-0,75% / -0,83%) đến từ sự suy giảm đồng loạt ngay bên trong từng phân khúc theo thời gian.",
        "Từ tháng 06/2022, lượng khách có Hội thoại tăng gấp đôi (3,6% → 7,9%) nhưng Lift vẫn giữ vững 3,3–5,1× → Cần hiệu chỉnh xác suất theo tháng."
    )
    slides.append(fig)

    # =========================================================================
    # SLIDE 5: E. ĐA NHÃN: ĐỒNG MUA CÙNG THÁNG & HÀNH TRÌNH BÁN CHÉO A -> B (QUOTE 4)
    # =========================================================================
    fig, bg = new_clean_slide(
        "E. Phân tích Đa nhãn: Đồng mua cùng tháng & Hành trình Bán chéo tuần tự (A → B)",
        "Ma trận chuyển tiếp từ lịch sử mua quá khứ sang tháng hiện tại (1.102.464 quan sát · 03/2022 – 01/2023)"
    )

    # Left chart: 4x4 Sequential Lift Heatmap (A prior -> B next)
    ax1 = fig.add_axes([0.07, 0.25, 0.38, 0.49])
    lift_mat = np.array([
        [3.20, 4.06, 3.07, 2.89],
        [4.37, 63.33, 3.82, 2.67],
        [1.98, 3.75, 3.35, 2.97],
        [2.00, 2.40, 2.23, 14.42],
    ])
    # Cap color scale at 8x so off-diagonal 2x-4.4x differences show clearly
    im = ax1.imshow(np.minimum(lift_mat, 8.0), cmap="YlGnBu", vmin=1.0, vmax=8.0, aspect="auto")
    ax1.set_title("Hệ số Lift mua sản phẩm B khi đã từng mua A", fontsize=13.5, fontweight="bold", color=NAVY, pad=12)
    p_short = ["SP 1\n(Thẻ DN)", "SP 2\n(Tín dụng)", "SP 3\n(Chi lương)", "SP 4\n(POS)"]
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(p_short, fontsize=10, fontweight="bold")
    ax1.set_yticks(range(4))
    ax1.set_yticklabels([f"Đã mua {x.splitlines()[0]}" for x in p_short], fontsize=10, fontweight="bold")
    ax1.set_xlabel("Sản phẩm mua ở tháng tiếp theo (B)", fontsize=10.5, color=SLATE_DARK)
    ax1.set_ylabel("Sản phẩm đã mua trước đó (A)", fontsize=10.5, color=SLATE_DARK)
    for i in range(4):
        for j in range(4):
            val = lift_mat[i, j]
            tag = " (Tái mua)" if i == j else " (Bán chéo)"
            txt_col = WHITE if min(val, 8.0) > 4.5 else NAVY
            ax1.text(j, i, f"{val:.1f}×\n{tag}", ha="center", va="center", fontsize=9.8, fontweight="bold", color=txt_col)

    bg.plot([0.48, 0.48], [0.22, 0.77], color="#e2e8f0", linewidth=1.5)

    # Right chart: Top same-month co-purchase pairs (647 multi-product customer-months = 5.03% of positive months)
    ax2 = fig.add_axes([0.63, 0.25, 0.31, 0.49])
    style_clean_ax(ax2, "Các cặp sản phẩm mua gộp cùng tháng (647 ca)")
    pair_names = [
        "SP 2 + SP 4\n(Tín dụng + POS)",
        "SP 2 + SP 3\n(Tín dụng + Chi lương)",
        "SP 1 + SP 2\n(Thẻ DN + Tín dụng)",
        "SP 3 + SP 4\n(Chi lương + POS)",
        "SP 1 + SP 4\n(Thẻ DN + POS)",
        "SP 1 + SP 3\n(Thẻ DN + Chi lương)"
    ]
    pair_counts = [24, 32, 49, 131, 205, 263]
    bars2 = ax2.barh(range(6), pair_counts, color=[BLUE_LIGHT, BLUE_LIGHT, BLUE_MID, BLUE_MAIN, TEAL, NAVY], height=0.54)
    for b, cnt in zip(bars2, pair_counts):
        ax2.text(cnt + 6, b.get_y() + b.get_height() / 2, f"{cnt} lượt", va="center", ha="left", fontsize=10.5, fontweight="bold", color=NAVY)
    ax2.set_yticks(range(6))
    ax2.set_yticklabels(pair_names, fontsize=9.8, fontweight="bold")
    ax2.set_xlim(0, 320)
    ax2.set_xlabel("Số lượt khách–tháng đồng mua", fontsize=10.5, color=SLATE_DARK)
    ax2.grid(axis="x", alpha=0.25, linestyle=":")

    add_bottom_pill(
        bg,
        "Khách đã mua bất kỳ 1 sản phẩm nào đều có xác suất mua chéo sang sản phẩm khác cao gấp 2,0 – 4,4 lần (và tái mua gấp 3,2 – 63,3 lần).",
        "Bộ ba Thẻ doanh nghiệp (SP 1) – Chi lương (SP 3) – POS (SP 4) liên kết chặt chẽ cả khi mua gộp cùng tháng lẫn mua nối tiếp."
    )
    slides.append(fig)

    # =========================================================================
    # SLIDE 6: F. PHÁT HIỆN MỚI VỀ TRX: MỞ RỘNG LOẠI GIAO DỊCH MỚI (EVENT_TYPE NOVELTY)
    # =========================================================================
    fig, bg = new_clean_slide(
        "F. Phát hiện mới về TRX: Phát sinh Loại giao dịch mới báo hiệu khả năng mua tăng 4,9 lần",
        "Đo lường độ mở rộng danh mục loại giao dịch (event_type) của doanh nghiệp ở tháng liền trước (T−1)"
    )

    # Left chart: New event_type cells in T-1
    ax1 = fig.add_axes([0.07, 0.25, 0.39, 0.49])
    style_clean_ax(ax1, "Tỷ lệ mua theo Số mã giao dịch (event_type) mới ở T−1")
    nov_labels = [
        "0 TRX ở T−1\n(23,3%)",
        "0 mã mới\n(66,3%)",
        "1 mã mới\n(8,1%)",
        "≥ 2 mã mới\n(2,3%)"
    ]
    nov_rates = [0.67, 1.02, 2.20, 5.55]
    nov_lifts = ["0,59×", "0,90×", "1,93×", "4,87×"]
    nov_cols = [SLATE_LIGHT, BLUE_MID, TEAL, ORANGE]
    bars1 = ax1.bar(range(4), nov_rates, color=nov_cols, width=0.50)
    ax1.axhline(1.14, color=SLATE_MID, linestyle="--", linewidth=1.3, label="Mức chung: 1,14%")
    for b, val, lf in zip(bars1, nov_rates, nov_lifts):
        ax1.text(
            b.get_x() + b.get_width() / 2, val + 0.14, f"{val:.2f}%\n({lf})",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=NAVY,
            bbox=dict(boxstyle="round,pad=0.12", facecolor=WHITE, edgecolor="none", alpha=0.88)
        )
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(nov_labels, fontsize=10.2, fontweight="bold")
    ax1.set_xlabel("Số mã event_type mới lần đầu xuất hiện ở T−1 (kèm % tệp)", fontsize=10.5, color=SLATE_DARK, labelpad=8)
    ax1.set_ylim(0, 6.7)
    ax1.set_ylabel("Tỷ lệ mua tháng tới (%)", fontsize=11, color=SLATE_DARK)
    ax1.legend(frameon=False, loc="upper left", fontsize=10.5)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")

    bg.plot([0.50, 0.50], [0.22, 0.77], color="#e2e8f0", linewidth=1.5)

    # Right chart: Number of distinct event_types in T-1
    ax2 = fig.add_axes([0.55, 0.25, 0.38, 0.49])
    style_clean_ax(ax2, "Tỷ lệ mua theo Số loại giao dịch (event_type) ở T−1")
    div_labels = ["1 loại\n(20,3%)", "2 – 3 loại\n(27,4%)", "4 – 6 loại\n(17,6%)", "≥ 7 loại\n(11,5%)"]
    div_rates = [0.46, 0.76, 1.73, 3.27]
    div_lifts = ["0,41×", "0,67×", "1,52×", "2,87×"]
    div_cols = [BLUE_LIGHT, BLUE_MID, BLUE_MAIN, NAVY]
    bars2 = ax2.bar(range(4), div_rates, color=div_cols, width=0.50)
    ax2.axhline(1.14, color=SLATE_MID, linestyle="--", linewidth=1.3)
    for b, val, lf in zip(bars2, div_rates, div_lifts):
        ax2.text(
            b.get_x() + b.get_width() / 2, val + 0.08, f"{val:.2f}%\n({lf})",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=NAVY,
            bbox=dict(boxstyle="round,pad=0.12", facecolor=WHITE, edgecolor="none", alpha=0.88)
        )
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(div_labels, fontsize=10.2, fontweight="bold")
    ax2.set_xlabel("Số mã event_type khác nhau phát sinh ở T−1 (kèm % tệp)", fontsize=10.5, color=SLATE_DARK, labelpad=8)
    ax2.set_ylim(0, 4.1)
    ax2.set_ylabel("Tỷ lệ mua tháng tới (%)", fontsize=11, color=SLATE_DARK)
    ax2.grid(axis="y", alpha=0.25, linestyle=":")

    add_bottom_pill(
        bg,
        "Khi doanh nghiệp phát sinh ≥2 mã giao dịch mới ở tháng T−1, tỷ lệ mua tháng tới vọt lên 5,55% (gấp 4,87 lần mức chung).",
        "Riêng với sản phẩm Chi lương (SP 3) và Tín dụng (SP 2), việc xuất hiện ≥2 mã giao dịch mới làm tăng xác suất mua gấp 6,5 lần."
    )
    slides.append(fig)

    # =========================================================================
    # SLIDE 7: G. PHÁT HIỆN MỚI VỀ GEO & MỐI LIÊN HỆ ĐỘC LẬP TRX × GEO
    # =========================================================================
    fig, bg = new_clean_slide(
        "G. Mối liên hệ TRX × GEO: Hai chiều mở rộng độc lập giúp phân tách người mua gấp 8,2 lần",
        "Tương quan giữa tần suất/đa dạng TRX và số vùng GEO chỉ là r = 0,02 → Bổ sung thông tin không trùng lặp"
    )

    # Left chart: GEO spatial mobility in T-1 & Credit lift
    ax1 = fig.add_axes([0.07, 0.25, 0.38, 0.49])
    style_clean_ax(ax1, "Tỷ lệ mua theo Số vùng geohash_4 ở tháng T−1")
    g_labels = ["0 vùng\n(41,9%)", "1 vùng\n(10,5%)", "2 – 4 vùng\n(27,8%)", "5 – 9 vùng\n(15,1%)", "≥ 10 vùng\n(4,7%)"]
    g_rates = [0.95, 1.12, 1.26, 1.36, 1.44]
    g_cols = [SLATE_LIGHT, BLUE_LIGHT, BLUE_MID, BLUE_MAIN, TEAL]
    bars1 = ax1.bar(range(5), g_rates, color=g_cols, width=0.50)
    ax1.axhline(1.14, color=SLATE_MID, linestyle="--", linewidth=1.3)
    for b, val in zip(bars1, g_rates):
        ax1.text(b.get_x() + b.get_width() / 2, val + 0.04, f"{val:.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=NAVY)
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(g_labels, fontsize=9.8, fontweight="bold")
    ax1.set_xlabel("Số vùng geohash_4 có tín hiệu ở tháng T−1 (kèm % tệp)", fontsize=10.5, color=SLATE_DARK, labelpad=8)
    ax1.set_ylim(0, 1.85)
    ax1.set_ylabel("Tỷ lệ mua chung y_any (%)", fontsize=11, color=SLATE_DARK)
    ax1.text(0.50, 0.88, "Lưu ý: Với SP 2 (Tín dụng), nhóm ≥10 vùng geohash_4\ncó tỷ lệ mua cao gấp 2,17 lần mức nền!", transform=ax1.transAxes, ha="center", fontsize=9.8, fontweight="bold", color=TEAL)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")

    bg.plot([0.49, 0.49], [0.22, 0.77], color="#e2e8f0", linewidth=1.5)

    # Right chart: Joint TRX Tier x GEO Tier interaction (TEST 7)
    ax2 = fig.add_axes([0.54, 0.25, 0.39, 0.49])
    style_clean_ax(ax2, "Tương tác Số loại TRX × Số vùng geohash_4 (ở T−1)")
    trx_tiers = [
        "1 – 3 loại TRX\n(0 mã mới)",
        "≥ 4 loại TRX\n(0 mã mới)",
        "≥ 1 mã TRX mới\nở tháng T−1"
    ]
    g0_vals = [0.42, 1.86, 2.54]
    g1_vals = [0.55, 2.04, 2.93]
    g2_vals = [0.60, 2.14, 3.44]
    xt = np.arange(3)
    w = 0.25
    b_g0 = ax2.bar(xt - w, g0_vals, width=w, color=SLATE_LIGHT, label="0 vùng GEO")
    b_g1 = ax2.bar(xt, g1_vals, width=w, color=BLUE_MID, label="1 – 3 vùng GEO")
    b_g2 = ax2.bar(xt + w, g2_vals, width=w, color=TEAL, label="≥ 4 vùng GEO")
    for bars_set, col_txt in [(b_g0, SLATE_MID), (b_g1, BLUE_MAIN), (b_g2, TEAL)]:
        for b in bars_set:
            v = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}%", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color=col_txt)
    ax2.set_xticks(xt)
    ax2.set_xticklabels(trx_tiers, fontsize=10, fontweight="bold")
    ax2.set_xlabel("Nhóm giao dịch TRX ở tháng T−1", fontsize=10.5, color=SLATE_DARK, labelpad=8)
    ax2.set_ylim(0, 4.15)
    ax2.set_ylabel("Tỷ lệ mua tháng tới (%)", fontsize=11, color=SLATE_DARK)
    ax2.legend(frameon=False, loc="upper left", fontsize=9.8)
    ax2.grid(axis="y", alpha=0.25, linestyle=":")

    add_bottom_pill(
        bg,
        "TRX đo độ đa dạng loại giao dịch, GEO đo số vùng địa lý hoạt động (hai chiều độc lập với tương quan chỉ r = 0,02).",
        "Ở mọi nhóm TRX, khách có ≥4 vùng GEO đều mua cao hơn; kết hợp (≥1 mã TRX mới) × (≥4 vùng GEO) nâng tỷ lệ mua gấp 8,2 lần (0,42% → 3,44%)."
    )
    slides.append(fig)

    # =========================================================================
    # SLIDE 8: H. TỔNG KẾT CÁC PHÁT HIỆN CHUYÊN SÂU & HÀM Ý CHO MÔ HÌNH (STYLE PAGE 24)
    # =========================================================================
    fig, bg = new_clean_slide(
        "H. Tổng kết các phát hiện chuyên sâu & Đặc trưng cần đưa vào Mô hình",
        "Bảng đối chiếu trực tiếp từ bằng chứng định lượng sang thiết kế Feature Engineering & Huấn luyện"
    )

    cols_x = [0.045, 0.235, 0.595]
    cols_w = [0.182, 0.352, 0.360]
    headers = [" Khám phá trọng tâm", " Bằng chứng định lượng cốt lõi", " Thiết kế Đặc trưng & Quy tắc Mô hình"]
    for cx, cw, h_txt in zip(cols_x, cols_w, headers):
        hbox = FancyBboxPatch((cx, 0.745), cw, 0.056, boxstyle="round,pad=0.002,rounding_size=0.008", facecolor=NAVY, edgecolor="none", zorder=2)
        bg.add_patch(hbox)
        bg.text(cx + 0.01, 0.773, h_txt, fontsize=11.5, fontweight="bold", color=WHITE, va="center", ha="left", zorder=3)

    table_rows = [
        ("1. Cơ chế chữ U\n& Nhóm '0 TRX'",
         "• 85,4% nhóm 0 TRX (72.304 dòng) chỉ mua 0,56%;\n  14,6% còn lại (3,4% có DIALOG + 11,2% mới có TRX) tạo 66,2% ca mua.\n• Ở 0 TRX: ≥1 DIALOG ở T−1 mua 9,41%; ≥4 vùng GEO mua 1,66%.",
         "• Giữ cờ riêng cho trạng thái '0 TRX' (không ép đơn điệu).\n• Tạo biến tương tác: (TRX = 0) × (DIALOG ở T−1) và\n  (TRX = 0) × (Số vùng geohash_4 hoạt động)."),
        ("2. Số tháng lịch sử\n& Tỷ số TRX(T−1)",
         "• Cùng 10–99 TRX nhưng lịch sử 1–6 tháng mua 1,14%,\n  còn lịch sử ≥13 tháng chỉ mua 0,31% (lệch 3,7 lần).\n• Tỷ số TRX(T−1)/TB trước >1,5× mua 1,88% vs <0,5× mua 0,75%.",
         "• Bắt buộc đưa Số tháng lịch sử (history_age_months).\n• Chuẩn hóa tổng đếm thành Trung bình/tháng (trx / tenure)\n  và Tỷ số ngắn hạn (trx_T1 / prior_monthly_mean)."),
        ("3. Số mã TRX mới\n× Số vùng GEO",
         "• Có ≥2 mã event_type mới ở T−1 vọt lên 5,55% (4,87×).\n• TRX và GEO độc lập (r = 0,02); kết hợp (≥1 mã TRX mới)\n  × (≥4 vùng geohash_4) nâng tỷ lệ mua từ 0,42% → 3,44%.",
         "• Tạo biến Số mã event_type mới xuất hiện ở tháng T−1\n  và Số mã event_type khác nhau (unique event_types).\n• Kết hợp Số vùng geohash_4 × Số mã event_type mới."),
        ("4. Tương tác DIALOG,\nDrift & Bán chéo",
         "• ≥1 DIALOG ở T−1 tăng tỷ lệ mua gấp 10,6–16,6× ở nhóm\n  (0 mua, 1–99 TRX) và đạt 5,00% ở nhóm (≥1 mua, ≥100 TRX).\n• Đã mua 1 SP tăng mua chéo 2,0–4,4×; 90% mức giảm 2022 là nội nhóm.",
         "• Tạo tương tác DIALOG × Số tháng lịch sử × TRX & lịch sử 4 SP.\n• Dùng Rolling Validation và Hiệu chỉnh xác suất (Calibration)\n  trên tháng gần nhất để ứng phó suy giảm tỷ lệ mua nội nhóm.")
    ]
    row_y = [0.608, 0.472, 0.336, 0.200]
    row_bgs = ["#f8fafc", KPI_BG, "#f8fafc", KPI_BG]
    for y_pos, r_bg, (c1, c2, c3) in zip(row_y, row_bgs, table_rows):
        for cx, cw in zip(cols_x, cols_w):
            rbox = FancyBboxPatch((cx, y_pos), cw, 0.124, boxstyle="round,pad=0.002,rounding_size=0.008", facecolor=r_bg, edgecolor="#dbeafe", linewidth=0.9, zorder=2)
            bg.add_patch(rbox)
        bg.text(cols_x[0] + 0.012, y_pos + 0.062, c1, fontsize=11, fontweight="bold", color=NAVY, va="center", ha="left", zorder=3)
        bg.text(cols_x[1] + 0.012, y_pos + 0.062, c2, fontsize=9.8, color=SLATE_DARK, va="center", ha="left", zorder=3, linespacing=1.35)
        bg.text(cols_x[2] + 0.012, y_pos + 0.062, c3, fontsize=9.8, color=SLATE_DARK, va="center", ha="left", zorder=3, linespacing=1.35)

    add_bottom_pill(
        bg,
        "Kết luận: Thay vì chỉ đếm tổng số sự kiện thô, bộ đặc trưng thế hệ mới cần đo 'Tốc độ theo tuổi lịch sử', 'Sự xuất hiện mã giao dịch mới',",
        "'Độ mở rộng vùng GEO' và 'Tương tác Hội thoại gần' — giúp phân tách các phân khúc mua hàng chênh lệch từ 8 đến 45 lần."
    )
    slides.append(fig)

    with PdfPages(PDF_PATH) as pdf:
        for idx, fig in enumerate(slides, start=1):
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            png_path = SLIDE_DIR / f"slide_{idx:02d}.png"
            fig.savefig(png_path, dpi=180, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Saved clean slide {idx:02d} -> {png_path.name}", flush=True)

    print(f"SUCCESSFULLY GENERATED CLEAN PDF: {PDF_PATH}", flush=True)


if __name__ == "__main__":
    build_deck()
