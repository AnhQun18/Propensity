"""Build the 5 consolidated, report-aligned EDA notebooks (01-05)."""
from __future__ import annotations

import hashlib
from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


COMMON_SETUP = """
from pathlib import Path
import subprocess
import sys
import pandas as pd
from IPython.display import display, Image, Markdown

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))


def run_script(rel_path: str, *args: str) -> None:
    \"\"\"Run a repository script with the active Python interpreter.\"\"\"
    script = ROOT / rel_path
    assert script.exists(), f"Missing script: {script}"
    cmd = [sys.executable, str(script), *args]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def show_chart(rel_path: str, caption: str | None = None, width: int = 920) -> None:
    \"\"\"Display a generated chart image inline if present.\"\"\"
    path = ROOT / rel_path
    if caption:
        display(Markdown(f"**{caption}** (`{rel_path}`)"))
    if path.exists():
        display(Image(filename=str(path), width=width))
    else:
        display(Markdown(f"_Chưa có file ảnh `{rel_path}` — chạy cell tạo biểu đồ phía trên._"))


def show_table(rel_path: str, nrows: int = 12) -> None:
    \"\"\"Display a generated CSV table inline if present.\"\"\"
    path = ROOT / rel_path
    if path.exists():
        display(pd.read_csv(path, nrows=nrows))
    else:
        display(Markdown(f"_Chưa có bảng `{rel_path}`._"))
"""


def write_nb(filename: str, cells: list) -> Path:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (.venv)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    for idx, cell in enumerate(nb.cells):
        cell.id = hashlib.sha256(
            f"{filename}:{idx}:{cell.cell_type}:{cell.source}".encode("utf-8")
        ).hexdigest()[:16]
    nbf.validate(nb)
    out = NOTEBOOKS / filename
    nbf.write(nb, out)
    return out


def build_nb01() -> Path:
    cells = [
        md("""
# 01 · Báo cáo 1 (Slides 1–7): Tổng quan dữ liệu, ERD, Timeline & Nhãn mua (`TARGETS`)

Đối chiếu trực tiếp với **Báo cáo 1 (`MBD MINI UNDERSTANDING DATA.pdf`, Slides 1–7)**:
- **Slide 2 (ERD & Kiểm kê):** Cấu trúc 5 bảng (`CLIENT_SPLIT`, `TARGETS`, `TRX`, `GEO`, `DIALOG`), khóa nối và kiểm tra join (`src/audit_mbd_inventory.py`, `src/audit_mbd_join_questions.py`, `src/scripts/audit_insights_checklist.py`).
- **Slide 3 (Timeline):** Các nguồn dữ liệu bắt đầu ở các thời điểm khác nhau (`src/audit_mbd_temporal_coverage.py`).
- **Slide 4 (`TARGETS` overview):** Tỷ lệ dương 4 nhãn (`target_1..4`), `y_any ≈ 1,069%`, `10,043%` NaN `diff_trans_date` gắn với `trans_count = 0` (`src/audit_any_product_cohorts.py`, `src/scripts/check_trans_count_target_history.py`).
- **Slide 5 (`y_any` theo tháng):** Diễn biến tỷ lệ mua 12 tháng (`02/2022–01/2023`).
- **Slide 6 (Tỷ lệ mua quay lại):** So sánh khách đã mua ở bất kỳ tháng trước đó với tháng liền trước `T-1`.
- **Slide 7 (Số tháng có mua / khách):** `89,61%` không mua, `10,39%` (`10.411` khách) từng mua, `81,60%` người mua chỉ mua đúng 1 tháng.
"""),
        code(COMMON_SETUP),
        md("## 1. Kiểm kê dữ liệu, khóa nối & xem mẫu (Slide 2)"),
        code("""
# Xem nhanh dữ liệu mẫu trong sample_data/ (1.000 khách hàng)
sample_dir = ROOT / "sample_data"
for table in ["clients", "targets", "trx", "geo", "dialog"]:
    files = sorted((sample_dir / table).glob("*.parquet"))
    if files:
        df_s = pd.read_parquet(files[0])
        print(f"{table:8s} | sample shape={df_s.shape} | columns={list(df_s.columns)}")
"""),
        code("""
# Chạy lại kiểm kê toàn bộ raw & đối soát khóa nối (tùy chọn, cần data/MBD-mini)
RUN_RAW_AUDITS = False
if RUN_RAW_AUDITS:
    run_script("src/audit_mbd_inventory.py")
    run_script("src/audit_mbd_join_questions.py")
    run_script("src/scripts/audit_insights_checklist.py")

show_table("outputs/eda_insights_20261119/join_audit.csv")
"""),
        md("## 2. Độ phủ thời gian của các nguồn dữ liệu (Slide 3)"),
        code("""
RUN_TEMPORAL_AUDIT = False
if RUN_TEMPORAL_AUDIT:
    run_script("src/audit_mbd_temporal_coverage.py")

show_table("docs/report/tables/cross_modal/dataset_timeline.csv")
show_chart("docs/report/figures/cross_modal/dataset_timeline.png", "Slide 3 · Các nguồn bắt đầu ở các thời điểm khác nhau")
"""),
        md("## 3. Tổng quan 4 nhãn sản phẩm, `y_any` và `diff_trans_date` NaN (Slide 4)"),
        code("""
RUN_TARGET_AUDITS = False
if RUN_TARGET_AUDITS:
    run_script("src/audit_any_product_cohorts.py")
    run_script("src/scripts/check_trans_count_target_history.py")

show_table("docs/report/tables/targets/target_positive_rates.csv")
show_table("docs/report/tables/targets/target_null_patterns.csv")
"""),
        md("## 4. Xu hướng `y_any` theo tháng, Tỷ lệ mua quay lại & Số tháng có mua (Slides 5–7)"),
        code("""
REGENERATE_TARGET_CHARTS = False
if REGENERATE_TARGET_CHARTS:
    run_script("outputs/repeat_purchase_review/analyze.py")
    run_script("outputs/trx_charts/plot_y_any.py")
    run_script("outputs/trx_charts/plot_returning_monthly_comparison.py")
    run_script("outputs/trx_charts/plot_purchase_frequency.py")

show_table("outputs/eda_insights_20261119/label_monthly.csv", nrows=12)
show_chart("outputs/trx_charts/y_any_positive_rate.png", "Slide 5 · y_any theo 12 tháng nhãn")
show_chart("outputs/trx_charts/returning_buyer_share_monthly_comparison.png", "Slide 6 · Tỷ lệ mua quay lại theo tháng")
show_chart("outputs/trx_charts/purchase_months_per_customer.png", "Slide 7 · Số tháng có mua trên mỗi khách hàng")
"""),
    ]
    return write_nb("01_report1_overview_and_targets.ipynb", cells)


def build_nb02() -> Path:
    cells = [
        md("""
# 02 · Báo cáo 1 (Slides 8–12): Khám phá dữ liệu Giao dịch (`TRX`)

Đối chiếu trực tiếp với **Báo cáo 1 (`MBD MINI UNDERSTANDING DATA.pdf`, Slides 8–12)**:
- **Slide 8 (Chất lượng & Currency TRX):** 99,8% đầy đủ, `currency` mã 11 chiếm 99,895% (13 mã), 1,5% (`1.503` khách) không có log TRX, các cặp thuộc tính null cùng nhau (`src_type11/12`, `dst_type11/12`, `src_type21/22`, `src_type31/32`).
- **Slide 9 (Phân vị giao dịch & Amount):** Boxplot số GD/khách (P25=38, P50=164, P75=503) và tổng amount mã 11 (P25=2,65M, P50=17,69M, P75=79,45M); ngoại lai lớn nhất `1.370.187.366.400`.
- **Slide 10 (Mức độ tập trung - Lorenz):** Top 1% khách tạo `13,16%` GD; Top 5% tạo `30,66%`.
- **Slide 11 (RFM: Mua và Không mua):** Recency (số ngày từ GD gần nhất), Frequency (`T-1`), Monetary mã 11 (`T-1`).
- **Slide 12 (Tỷ trọng & Lift theo `event_type`):** Top 5 mã (`1, 46, 40, 49, 11`) và hệ số Lift mua hàng ở tháng `T-1`.
"""),
        code(COMMON_SETUP),
        md("## 1. Chất lượng dữ liệu TRX, Null theo cặp & Cơ cấu Currency (Slide 8)"),
        code("""
RUN_TRX_QUALITY = False
if RUN_TRX_QUALITY:
    run_script("src/scripts/audit_trx_nulls.py")
    run_script("src/audit_mbd_full_history_missingness.py")
    run_script("outputs/trx_charts/plot_currency.py")

show_table("outputs/eda_insights_20261119/amount_by_currency.csv", nrows=15)
show_chart("outputs/trx_charts/trx_currency.png", "Slide 8 · Cơ cấu giao dịch theo currency")
"""),
        md("## 2. Phân vị số giao dịch & tổng Amount mã 11, Ngoại lai lớn nhất (Slide 9)"),
        code("""
RUN_TRX_QUANTILES = False
if RUN_TRX_QUANTILES:
    run_script("src/scripts/audit_trx_outliers.py")
    run_script("outputs/trx_charts/customer_quantiles.py")
    run_script("outputs/trx_charts/plot_customer_quantiles.py")

show_chart("outputs/trx_charts/customer_quantiles.png", "Slide 9 · Phân vị giao dịch & tổng amount mã 11 trên mỗi khách")
"""),
        md("## 3. Đường cong Lorenz tập trung giao dịch theo khách (Slide 10)"),
        code("""
REGENERATE_LORENZ = False
if REGENERATE_LORENZ:
    run_script("outputs/trx_charts/plot_sparsity.py")

show_chart("outputs/trx_charts/sparsity_customer_concentration.png", "Slide 10 · Mức độ tập trung giao dịch theo khách")
"""),
        md("## 4. RFM: Nhóm Mua và Không mua (Slide 11)"),
        code("""
REGENERATE_RFM = False
if REGENERATE_RFM:
    run_script("outputs/trx_charts/compute_rfm_means.py")
    run_script("outputs/trx_charts/plot_rfm.py")

show_table("outputs/trx_charts/rfm_means_p95.csv")
show_chart("outputs/trx_charts/trx_rfm_purchase_vs_no_purchase.png", "Slide 11 · RFM: Mua và Không mua")
"""),
        md("## 5. Tỷ trọng & Hệ số Lift theo `event_type` tháng `T-1` (Slide 12)"),
        code("""
REGENERATE_EVENT_TYPE = False
if REGENERATE_EVENT_TYPE:
    run_script("outputs/trx_charts/generate_event_type_lift_data.py")
    run_script("outputs/trx_charts/plot_event_type_portrait.py")
    run_script("outputs/trx_charts/plot_event_type_lift_horizontal.py")

show_table("outputs/trx_charts/trx_top5_event_type_lift.csv")
show_chart("outputs/trx_charts/trx_top5_event_type_portrait.png", "Slide 12a · Tỷ trọng Top 5 event_type")
show_chart("outputs/trx_charts/trx_event_type_lift_horizontal.png", "Slide 12b · Lift theo event_type tháng T-1")
"""),
    ]
    return write_nb("02_report1_trx.ipynb", cells)


def build_nb03() -> Path:
    cells = [
        md("""
# 03 · Báo cáo 1 (Slides 13–16): Khám phá dữ liệu Vị trí (`GEO`)

Đối chiếu trực tiếp với **Báo cáo 1 (`MBD MINI UNDERSTANDING DATA.pdf`, Slides 13–16)**:
- **Slide 13 (Tổng quan GEO):** `18.948` mã `geohash_4`, `122.982` mã `geohash_5`, `610.866` mã `geohash_6`; `72.573 / 100.224` (`72,41%`) khách có log; phân phối số ô `g4`, CDF sự kiện & `70,6%` tập trung ở ô chính `g4`.
- **Slide 14 (GEO quanh tháng mua):** So sánh tháng trước và tháng sau mốc mua trên nhóm có mua và không mua (02/2022–01/2023, Moscow).
- **Slide 15 (GEO quanh ngày ghi nhận nhãn):** Phân rã `±30 ngày` quanh `00:00 Moscow` ngày `mon` (Không có hai phía / Chỉ trước / Chỉ sau / Cả trước và sau).
- **Slide 16 (Tín hiệu GEO & khả năng mua):** Boxplot và Lift theo số mã `geohash_4`, `geohash_5`, `geohash_6` phân biệt ở tháng `T-1`.
"""),
        code(COMMON_SETUP),
        md("## 1. Tổng quan `geohash_4/5/6`, đa dạng ô `g4` & tỷ lệ tập trung ô chính `70,6%` (Slide 13)"),
        code("""
RUN_GEO_OVERVIEW = False
if RUN_GEO_OVERVIEW:
    run_script("outputs/geo_charts/compute_unique_geohashes.py")
    run_script("src/scripts/extend_eda_geo_population.py")
    run_script("src/scripts/plot_geo_g4_concentration.py")

show_table("docs/report/tables/geo/geo_overview.csv")
show_chart("outputs/eda_figures_moscow/geo_g4_diversity_top1_compact.png", "Slide 13 · Phân phối số ô g4, CDF sự kiện & tỷ lệ ô chính")
"""),
        md("## 2. GEO quanh tháng mua & quanh ngày ghi nhận nhãn `±30 ngày` (Slides 14–15)"),
        code("""
RUN_GEO_EVENT_STUDY = False
if RUN_GEO_EVENT_STUDY:
    run_script("src/scripts/analyze_full_population_temporal_eda.py")
    run_script("src/scripts/plot_geo_purchase_event_study.py")
    run_script("src/scripts/plot_dialog_label_date_transition.py")

show_chart("outputs/full_population_temporal_eda_20261118/geo_event_study_buyers.png", "Slide 14a · GEO quanh tháng mua (Nhóm có mua)")
show_chart("outputs/full_population_temporal_eda_20261118/geo_event_study_nonbuyers.png", "Slide 14b · GEO quanh tháng mua (Nhóm không mua)")
show_chart("outputs/full_population_temporal_eda_20261118/geo_label_date_transition_30d.png", "Slide 15 · GEO quanh ngày ghi nhận nhãn (±30 ngày Moscow)")
"""),
        md("## 3. Tín hiệu GEO (`geohash_4/5/6`) tháng `T-1` và khả năng mua (Slide 16)"),
        code("""
REGENERATE_GEO_SIGNAL = False
if REGENERATE_GEO_SIGNAL:
    run_script("outputs/geo_charts/compare_geo_levels_target.py")
    run_script("outputs/geo_charts/plot_geo_level4_signal_target.py")
    run_script("outputs/geo_charts/plot_geo_level5_signal_target.py")
    run_script("outputs/geo_charts/plot_geo_signal_target.py")

show_chart("outputs/geo_charts/geo_level4_signal_vs_target.png", "Slide 16a · Số mã geohash_4 tháng T-1 & Lift")
show_chart("outputs/geo_charts/geo_level5_signal_vs_target.png", "Slide 16b · Số mã geohash_5 tháng T-1 & Lift")
show_chart("outputs/geo_charts/geo_signal_vs_target.png", "Slide 16c · Số mã geohash_6 tháng T-1 & Lift")
"""),
    ]
    return write_nb("03_report1_geo.ipynb", cells)


def build_nb04() -> Path:
    cells = [
        md("""
# 04 · Báo cáo 1 (Slides 17–24): Khám phá Hội thoại (`DIALOG`) & Quan hệ Đa nguồn

Đối chiếu trực tiếp với **Báo cáo 1 (`MBD MINI UNDERSTANDING DATA.pdf`, Slides 17–24)**:
- **Slide 17 (Tổng quan DIALOG):** 1 cột embedding 768 chiều (0 null, 0 vector 0, 0 sai chiều); `46,22%` (`46.320` khách) có dialog; `81,66%` trong `10.411` người mua có dialog.
- **Slide 18 (DIALOG quanh tháng mua):** Tỷ lệ khách có `≥1` DIALOG tháng trước vs tháng sau mốc mua.
- **Slide 19 (DIALOG quanh ngày ghi nhận nhãn):** Phân rã `±30 ngày` quanh `00:00 Moscow` ngày `mon`.
- **Slide 20 (Đặc điểm tương tác):** Phân phối số hội thoại/khách, CDF khoảng cách 2 hội thoại liên tiếp (trung vị `12,93 ngày`), `27,01%` khách chỉ có 1 hội thoại.
- **Slide 21 (Thời điểm tương tác):** Heatmap giờ (UTC) × thứ trong tuần (`217.205` hội thoại).
- **Slide 22 (Khám phá embedding):** Cutoff `01/11/2022` (`7.868` khách có dialog tháng 10) — PCA 2D (`PC1 58,1%`, `PC2 10,0%`) & tỷ lệ mua theo 4 cụm K-means.
- **Slide 23 (Có nguồn hành vi & Tỷ lệ mua):** GEO (`1,201%` vs `0,886%`, Lift `1,12×`) và DIALOG (`3,845%` vs `0,888%`, Lift `3,60×`) tháng `T-1` kèm CI bootstrap theo khách.
"""),
        code(COMMON_SETUP),
        md("## 1. Tổng quan DIALOG & Chất lượng embedding 768 chiều (Slide 17)"),
        code("""
RUN_DIALOG_OVERVIEW = False
if RUN_DIALOG_OVERVIEW:
    run_script("outputs/dialog_charts/check_embedding_norms.py")
    run_script("outputs/dialog_charts/plot_dialog_coverage.py")

show_table("docs/report/tables/dialog/dialog_overview.csv")
show_chart("outputs/dialog_charts/dialog_customer_coverage.png", "Slide 17 · Độ phủ DIALOG toàn tệp & trong nhóm từng mua")
"""),
        md("## 2. DIALOG quanh tháng mua & quanh ngày ghi nhận nhãn `±30 ngày` (Slides 18–19)"),
        code("""
RUN_DIALOG_TRANSITION = False
if RUN_DIALOG_TRANSITION:
    run_script("src/scripts/plot_geo_purchase_event_study.py")
    run_script("src/scripts/plot_dialog_label_date_transition.py")

show_chart("outputs/full_population_temporal_eda_20261118/dialog_event_study_buyers.png", "Slide 18 · DIALOG quanh tháng mua (Nhóm có mua)")
show_chart("outputs/full_population_temporal_eda_20261118/dialog_label_date_transition_30d.png", "Slide 19 · DIALOG quanh ngày ghi nhận nhãn (±30 ngày Moscow)")
"""),
        md("## 3. Đặc điểm tương tác & Thời điểm tương tác hội thoại (Slides 20–21)"),
        code("""
REGENERATE_DIALOG_PATTERNS = False
if REGENERATE_DIALOG_PATTERNS:
    run_script("outputs/dialog_charts/plot_interaction_patterns_linear.py")
    run_script("outputs/dialog_charts/plot_dialog_temporal.py")

show_chart("outputs/dialog_charts/dialog_interaction_patterns_linear.png", "Slide 20 · Đặc điểm tương tác hội thoại")
show_chart("outputs/dialog_charts/dialog_temporal_behavior.png", "Slide 21 · Heatmap số hội thoại theo giờ × thứ (UTC)")
"""),
        md("## 4. Khám phá Embedding hội thoại: PCA 2D & K-means 4 cụm (Slide 22)"),
        code("""
REGENERATE_SEMANTIC = False
if REGENERATE_SEMANTIC:
    run_script("outputs/dialog_charts/plot_semantic_exploration.py")

show_table("outputs/dialog_charts/semantic_cluster_purchase_rates.csv")
show_chart("outputs/dialog_charts/semantic_embedding_exploration.png", "Slide 22 · PCA 2D & Tỷ lệ mua theo cụm K-means")
"""),
        md("## 5. Có nguồn hành vi (GEO / DIALOG tháng `T-1`) và Tỷ lệ mua tháng sau (Slide 23)"),
        code("""
REGENERATE_SOURCE_PRESENCE = False
if REGENERATE_SOURCE_PRESENCE:
    run_script("src/scripts/analyze_dataset_eda.py")
    run_script("src/scripts/plot_source_presence_target.py")

show_table("outputs/source_presence_target/source_presence_purchase_ci.csv")
show_chart("outputs/source_presence_target/source_presence_purchase_ci.png", "Slide 23 · Có nguồn hành vi ở T-1 và tỷ lệ mua tháng T")
"""),
    ]
    return write_nb("04_report1_dialog_and_cross_source.ipynb", cells)


def build_nb05() -> Path:
    cells = [
        md("""
# 05 · Báo cáo 2 (Slides A–H): Phân tích Chuyên sâu (`MBD_MINI_DEEP_INSIGHTS_DECK.pdf`)

Notebook tái lập và kiểm chứng trực tiếp **100% số liệu của 8 slide (A–H)** trong **Báo cáo 2 (`MBD_MINI_DEEP_INSIGHTS_DECK.pdf`)**, tuân thủ quy ước:
- **Cutoff:** `00:00` ngày mùng 1 tháng nhãn `T` theo giờ Moscow (`Europe/Moscow`).
- **Tập quan sát chuẩn (`Common Cohort`):** `100.224` khách × `10` mốc dự báo (`02/2022–11/2022`) = **`1.002.240` quan sát khách–tháng** có đủ nhãn `H=1` (`11.420` ca, `1,14%`) và `H=3` (`30.749` ca, `3,07%`).
- **Nguồn dữ liệu nền:** `all_customer_months.parquet` (`analyze_full_population_temporal_eda.py`), `event_type_monthly_profiles.parquet` & `geohash_4_monthly_profiles.parquet` (`extend_eda_monthly_features.py`), `monthly_segment_cards.csv` (`audit_insights_checklist.py`), và bảng `targets`.
"""),
        code("""
from pathlib import Path
import subprocess
import sys
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from IPython.display import display, Image, Markdown

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
BASE_DIR = ROOT / "outputs/full_population_temporal_eda_20261118"
FIG_DIR = ROOT / "outputs/eda_figures_moscow"
INSIGHT_DIR = ROOT / "outputs/eda_insights_20261119"
SLIDE_DIR = INSIGHT_DIR / "slides_clean"

# Tải bảng panel tháng Moscow + hồ sơ event_type, geohash_4 và 4 nhãn targets
panel = pd.read_parquet(BASE_DIR / "all_customer_months.parquet")
panel["month"] = pd.to_datetime(panel["month"])

et_prof = pd.read_parquet(FIG_DIR / "event_type_monthly_profiles.parquet")
et_prof["month"] = pd.to_datetime(et_prof["month"])
et_prof = et_prof.rename(columns={c: f"et_{c}" for c in et_prof.columns if c not in ("client_id", "month")})

g4_prof = pd.read_parquet(FIG_DIR / "geohash_4_monthly_profiles.parquet")
g4_prof["month"] = pd.to_datetime(g4_prof["month"])
g4_prof = g4_prof.rename(columns={c: f"g4_{c}" for c in g4_prof.columns if c not in ("client_id", "month")})

con = duckdb.connect()
targets_glob = (ROOT / "data/MBD-mini/targets/**/*.parquet").as_posix()
targets = con.execute(f\"\"\"
    SELECT client_id, date_trunc('month', CAST(mon AS DATE))::TIMESTAMP AS month,
           target_1, target_2, target_3, target_4
    FROM read_parquet('{targets_glob}')
\"\"\").df()
targets["month"] = pd.to_datetime(targets["month"])

df = (
    panel.merge(et_prof[["client_id", "month", "et_unique_cells", "et_new_cells"]], on=["client_id", "month"], how="left")
         .merge(g4_prof[["client_id", "month", "g4_unique_cells", "g4_new_cells"]], on=["client_id", "month"], how="left")
         .merge(targets, on=["client_id", "month"], how="left")
         .sort_values(["client_id", "month"])
         .reset_index(drop=True)
)

g = df.groupby("client_id", sort=False)

# Đặc trưng tháng liền trước (T-1) và tích lũy nghiêm ngặt trước cutoff (< T)
df["trx_prev"] = g["trx_events"].shift(1).fillna(0)
df["geo_prev"] = g["geo_events"].shift(1).fillna(0)
df["dlg_prev"] = g["dialog_events"].shift(1).fillna(0)
df["et_unique_prev"] = g["et_unique_cells"].shift(1).fillna(0)
df["et_new_prev"] = g["et_new_cells"].shift(1).fillna(0)
df["g4_unique_prev"] = g["g4_unique_cells"].shift(1).fillna(0)
df["g4_new_prev"] = g["g4_new_cells"].shift(1).fillna(0)

df["trx_hist"] = g["trx_events"].cumsum() - df["trx_events"].fillna(0)
df["trx_active_months_hist"] = g["trx_events"].transform(lambda s: (s.fillna(0) > 0).cumsum()) - (df["trx_events"].fillna(0) > 0).astype(int)
df["trx_cum_before_prev"] = df["trx_hist"] - df["trx_prev"]
df["trx_months_before_prev"] = df["trx_active_months_hist"] - (df["trx_prev"] > 0).astype(int)
df["trx_prior_mean"] = np.where(df["trx_months_before_prev"] > 0, df["trx_cum_before_prev"] / df["trx_months_before_prev"], np.nan)
df["trx_surge_ratio"] = np.where(df["trx_prior_mean"] > 0, df["trx_prev"] / df["trx_prior_mean"], np.nan)

# Số tháng lịch sử (history_age_months): từ tháng đầu tiên xuất hiện bất kỳ sự kiện nào (< T) đến tháng T
any_ev = (df["trx_events"].fillna(0) > 0) | (df["geo_events"].fillna(0) > 0) | (df["dialog_events"].fillna(0) > 0)
first_any = df["month"].where(any_ev).groupby(df["client_id"], sort=False).transform("min")
# Đảm bảo chỉ tính các sự kiện nghiêm ngặt trước tháng T
df["first_any_before"] = first_any.where(first_any < df["month"])
df["history_age_months"] = (
    (df["month"].dt.year - df["first_any_before"].dt.year) * 12
    + (df["month"].dt.month - df["first_any_before"].dt.month)
)

# Lịch sử mua quan sát được trước tháng T
df["y_fill"] = df["y"].fillna(0)
df["prior_buys"] = g["y_fill"].cumsum() - df["y_fill"]

# Nhãn H=1, H=3 trên Common Cohort (10 mốc 02/2022 - 11/2022)
df["y2_lead"] = g["y"].shift(-1)
df["y3_lead"] = g["y"].shift(-2)
cohort = df[df["y"].notna() & df["y2_lead"].notna() & df["y3_lead"].notna()].copy()
cohort["h1"] = cohort["y"].astype(int)
cohort["h3"] = ((cohort["y"] > 0) | (cohort["y2_lead"] > 0) | (cohort["y3_lead"] > 0)).astype(int)

BASE_H1 = cohort["h1"].mean()
BASE_H3 = cohort["h3"].mean()
print(f"Common cohort: {len(cohort):,} rows | H1={cohort['h1'].sum():,} ({BASE_H1:.4%}) | H3={cohort['h3'].sum():,} ({BASE_H3:.4%})")
assert len(cohort) == 1_002_240 and cohort["h1"].sum() == 11_420 and cohort["h3"].sum() == 30_749
"""),
        md("## Slide A · Giải mã đường cong chữ U & Bóc tách nhóm `0 Giao dịch` (`84.670` dòng)"),
        code("""
# 1. Đường cong chữ U theo tổng TRX tích lũy trước cutoff
cohort["trx_band"] = pd.cut(
    cohort["trx_hist"],
    bins=[-1, 0, 9, 99, np.inf],
    labels=["0 giao dịch", "1 - 9", "10 - 99", "≥ 100"],
)
u_tbl = cohort.groupby("trx_band", observed=True).agg(
    n=("h1", "size"), buyers_h1=("h1", "sum"), rate_h1=("h1", "mean"), rate_h3=("h3", "mean")
)
u_tbl["share_%"] = u_tbl["n"] / len(cohort) * 100
u_tbl["lift_h1"] = u_tbl["rate_h1"] / BASE_H1
display(u_tbl)

# 2. Bóc tách bên trong 84.670 dòng có trx_hist == 0
z0 = cohort[cohort["trx_hist"] == 0].copy()
z0["sub_group"] = np.select(
    [z0["dlg_prev"] > 0, (z0["dlg_prev"] == 0) & (z0["trx_events"] > 0)],
    ["1. Có Hội thoại tháng trước (T-1)", "2. Kích hoạt TRX mới (chưa DIALOG)"],
    default="3. Ngủ đông thực sự",
)
z0_tbl = z0.groupby("sub_group").agg(
    n=("h1", "size"), buyers_h1=("h1", "sum"), rate_h1=("h1", "mean"), rate_h3=("h3", "mean")
)
z0_tbl["share_in_0trx_%"] = z0_tbl["n"] / len(z0) * 100
z0_geo4 = z0[z0["g4_unique_prev"] >= 4]
print(f"Chỉ báo phụ trước cutoff (GEO >= 4 vùng trong 0 TRX): n={len(z0_geo4):,} ({len(z0_geo4)/len(z0):.2%}), H1={z0_geo4['h1'].mean():.2%}, H3={z0_geo4['h3'].mean():.2%}")
display(z0_tbl)
"""),
        md("## Slide B · Tách bạch Tần suất (`TRX/tháng`) × Số tháng lịch sử & Tỷ số `TRX(T-1) / TB trước đó`"),
        code("""
# 1. Cố định nhóm 0 lần mua trước (loại tháng đầu 02/2022 chưa biết lịch sử nhãn) & 0 DIALOG ở T-1 và có lịch sử TRX > 0
b_sub = cohort[(cohort["month"] > "2022-02-01") & (cohort["prior_buys"] == 0) & (cohort["dlg_prev"] == 0) & (cohort["trx_hist"] > 0) & (cohort["history_age_months"] >= 1)].copy()
b_sub["trx_per_month"] = b_sub["trx_hist"] / b_sub["history_age_months"]
b_sub["vel_band"] = pd.cut(b_sub["trx_per_month"], bins=[0, 2, 5, 10, 25, np.inf], labels=["≤2", "2-5", "5-10", "10-25", ">25"])
b_sub["age_band"] = pd.cut(b_sub["history_age_months"], bins=[0, 6, 12, np.inf], labels=["1-6 tháng", "7-12 tháng", "≥13 tháng"])
vel_pivot = b_sub.pivot_table(index="vel_band", columns="age_band", values="h1", aggfunc="mean", observed=True) * 100
display(vel_pivot.round(2))

# 2. Tỷ số TRX(T-1) / Trung bình tháng trước đó
cohort["surge_group"] = np.select(
    [
        cohort["trx_hist"] == 0,
        (cohort["trx_cum_before_prev"] > 0) & (cohort["trx_prev"] == 0),
        (cohort["trx_cum_before_prev"] == 0) & (cohort["trx_prev"] > 0),
        cohort["trx_surge_ratio"] < 0.5,
        cohort["trx_surge_ratio"] <= 1.5,
        cohort["trx_surge_ratio"] > 1.5,
    ],
    ["0 TRX toàn lịch sử", "0× (Tạm dừng T-1)", "Trước T-1 = 0, T-1 > 0", "< 0,5×", "0,5× - 1,5×", "> 1,5×"],
    default="other",
)
surge_tbl = cohort.groupby("surge_group").agg(n=("h1", "size"), buyers_h1=("h1", "sum"), rate_h1=("h1", "mean"))
surge_tbl["share_%"] = surge_tbl["n"] / len(cohort) * 100
surge_tbl["lift_h1"] = surge_tbl["rate_h1"] / BASE_H1
display(surge_tbl)
"""),
        md("## Slide C · Kiểm định đa tầng `DIALOG` theo tầng `TRX` & Lịch sử mua (Bác bỏ Nghịch lý Simpson)"),
        code("""
c_sub = cohort[(cohort["month"] > "2022-02-01") & (cohort["history_age_months"] >= 13)].copy()
c_sub["has_prior_buy"] = np.where(c_sub["prior_buys"] > 0, "≥1 lần mua trước", "0 lần mua trước")
c_sub["has_dlg_prev"] = c_sub["dlg_prev"] > 0
c_tbl = (
    c_sub.groupby(["has_prior_buy", "trx_band", "has_dlg_prev"], observed=True)
         .agg(n=("h1", "size"), buyers_h1=("h1", "sum"), rate_h1=("h1", "mean"), rate_h3=("h3", "mean"))
         .unstack("has_dlg_prev")
)
c_tbl["lift_dialog_vs_no"] = c_tbl[("rate_h1", True)] / c_tbl[("rate_h1", False)]
display(c_tbl.loc[[("0 lần mua trước", "1 - 9"), ("0 lần mua trước", "10 - 99"), ("0 lần mua trước", "≥ 100"), ("≥1 lần mua trước", "≥ 100")]])
"""),
        md("## Slide D · Phân rã Shift-Share (`04/2022 → 11/2022`) & Độ phủ `DIALOG` vs Lift theo tháng"),
        code("""
# 1. Phân rã Shift-Share trên 6 phân khúc rời nhau (TRX state × DIALOG T-1)
cohort["trx_state"] = np.select(
    [cohort["trx_hist"] == 0, cohort["trx_prev"] > 0],
    ["no_trx_history", "trx_active_prev_month"],
    default="trx_older_history",
)
cohort["dlg_state"] = cohort["dlg_prev"] > 0

m0 = cohort[cohort["month"] == "2022-04-01"].groupby(["trx_state", "dlg_state"]).agg(n0=("h1", "size"), r0=("h1", "mean"))
m1 = cohort[cohort["month"] == "2022-11-01"].groupby(["trx_state", "dlg_state"]).agg(n1=("h1", "size"), r1=("h1", "mean"))
ss = m0.join(m1)
ss["w0"] = ss["n0"] / ss["n0"].sum()
ss["w1"] = ss["n1"] / ss["n1"].sum()
ss["mix_effect_pp"] = (ss["w1"] - ss["w0"]) * ss["r0"] * 100
ss["within_effect_pp"] = ss["w0"] * (ss["r1"] - ss["r0"]) * 100
ss["interaction_pp"] = (ss["w1"] - ss["w0"]) * (ss["r1"] - ss["r0"]) * 100
display(ss[["w0", "r0", "w1", "r1", "mix_effect_pp", "within_effect_pp", "interaction_pp"]].round(4))
print(
    f"Tổng Shift-Share (điểm %): Mix={ss['mix_effect_pp'].sum():+.3f}%, "
    f"Within={ss['within_effect_pp'].sum():+.3f}%, "
    f"Interaction={ss['interaction_pp'].sum():+.3f}%, "
    f"Total={(ss['mix_effect_pp'] + ss['within_effect_pp'] + ss['interaction_pp']).sum():+.3f}%"
)

# 2. Độ phủ DIALOG & Lift theo từng tháng (10 mốc)
monthly_seg = pd.read_csv(INSIGHT_DIR / "monthly_segment_cards.csv")
dlg_m = monthly_seg[(monthly_seg["feature"] == "dialog_recent") & (monthly_seg["group"] == "True") & (monthly_seg["horizon"] == 1)]
display(dlg_m[["month", "population_share", "rate", "lift"]])
"""),
        md("## Slide E · Phân tích Đa nhãn: Ma trận Chuyển tiếp (`A → B`) & Đồng mua cùng tháng (`647` ca)"),
        code("""
# 1. Ma trận chuyển tiếp 4x4 trên 11 tháng có lịch sử nhãn (03/2022 - 01/2023, 1.102.464 dòng)
tdf = targets.sort_values(["client_id", "month"]).copy()
tg = tdf.groupby("client_id", sort=False)
for k in range(1, 5):
    tdf[f"prior_t{k}"] = (tg[f"target_{k}"].cumsum() - tdf[f"target_{k}"]) > 0

t_11 = tdf[tdf["month"] >= "2022-03-01"].copy()
assert len(t_11) == 1_102_464

rate_mat = pd.DataFrame(index=[f"Target {i}" for i in range(1, 5)], columns=[f"Target {j}" for j in range(1, 5)], dtype=float)
lift_mat = rate_mat.copy()
for i in range(1, 5):
    sub = t_11[t_11[f"prior_t{i}"]]
    for j in range(1, 5):
        p = sub[f"target_{j}"].mean()
        rate_mat.loc[f"Target {i}", f"Target {j}"] = p * 100
        lift_mat.loc[f"Target {i}", f"Target {j}"] = p / t_11[f"target_{j}"].mean()
display(Markdown("**Tỷ lệ mua Target B khi đã từng mua Target A (%)**"))
display(rate_mat.round(2))
display(Markdown("**Hệ số Lift (A → B)**"))
display(lift_mat.round(2))

# 2. Đồng mua gộp cùng tháng trên 12 tháng nhãn
tdf["n_prod"] = tdf[["target_1", "target_2", "target_3", "target_4"]].sum(axis=1)
buyers_all = (tdf["n_prod"] >= 1).sum()
co_all = (tdf["n_prod"] >= 2).sum()
print(f"Tổng tháng mua: {buyers_all:,} | Đồng mua >=2 SP cùng tháng: {co_all:,} ({co_all / buyers_all:.2%})")
pairs = []
for i in range(1, 5):
    for j in range(i + 1, 5):
        pairs.append({"pair": f"Target {i} + Target {j}", "count": int(((tdf[f"target_{i}"] == 1) & (tdf[f"target_{j}"] == 1)).sum())})
display(pd.DataFrame(pairs).sort_values("count", ascending=False))
"""),
        md("## Slide F & G · Phát hiện mới về `TRX` (`event_type` mới/đa dạng) & Mối liên hệ độc lập `TRX × GEO`"),
        code("""
# 1. Số mã event_type mới ở T-1 (Slide F trái)
cohort["et_novelty"] = np.select(
    [cohort["trx_prev"] == 0, cohort["et_new_prev"] == 0, cohort["et_new_prev"] == 1],
    ["0 TRX ở T-1", "0 mã mới", "1 mã mới"],
    default="≥ 2 mã mới",
)
f1 = cohort.groupby("et_novelty").agg(n=("h1", "size"), buyers_h1=("h1", "sum"), rate_h1=("h1", "mean"), rate_h3=("h3", "mean"))
f1["share_%"] = f1["n"] / len(cohort) * 100
f1["lift_h1"] = f1["rate_h1"] / BASE_H1
display(f1.loc[["0 TRX ở T-1", "0 mã mới", "1 mã mới", "≥ 2 mã mới"]])

# 2. Số loại event_type khác nhau ở T-1 (Slide F phải)
cohort["et_div"] = np.select(
    [cohort["trx_prev"] == 0, cohort["et_unique_prev"] == 1, cohort["et_unique_prev"].between(2, 3), cohort["et_unique_prev"].between(4, 6)],
    ["0 TRX ở T-1", "1 loại", "2 - 3 loại", "4 - 6 loại"],
    default="≥ 7 loại",
)
f2 = cohort.groupby("et_div").agg(n=("h1", "size"), buyers_h1=("h1", "sum"), rate_h1=("h1", "mean"))
f2["share_%"] = f2["n"] / len(cohort) * 100
f2["lift_h1"] = f2["rate_h1"] / BASE_H1
display(f2.loc[["1 loại", "2 - 3 loại", "4 - 6 loại", "≥ 7 loại"]])

# 3. Số vùng geohash_4 ở T-1 (Slide G trái) + Lift riêng cho SP2 (Tín dụng)
cohort["g4_band"] = np.select(
    [cohort["geo_prev"] == 0, cohort["g4_unique_prev"] == 1, cohort["g4_unique_prev"].between(2, 4), cohort["g4_unique_prev"].between(5, 9)],
    ["0 vùng", "1 vùng", "2 - 4 vùng", "5 - 9 vùng"],
    default="≥ 10 vùng",
)
g1 = cohort.groupby("g4_band").agg(n=("h1", "size"), rate_h1=("h1", "mean"), rate_p2=("target_2", "mean"))
g1["share_%"] = g1["n"] / len(cohort) * 100
g1["lift_h1"] = g1["rate_h1"] / BASE_H1
g1["lift_p2"] = g1["rate_p2"] / cohort["target_2"].mean()
display(g1.loc[["0 vùng", "1 vùng", "2 - 4 vùng", "5 - 9 vùng", "≥ 10 vùng"]])

# 4. Độc lập TRX vs GEO (Spearman) & Ma trận tương tác 4x3 (Slide G phải)
active_both = cohort[(cohort["trx_prev"] > 0) & (cohort["geo_prev"] > 0)]
r_new = spearmanr(active_both["et_new_prev"], active_both["g4_unique_prev"]).statistic
print(f"Spearman(et_new_prev, g4_unique_prev) trên {len(active_both):,} dòng hoạt động cả 2 nguồn: r = {r_new:.4f}")

cohort["geo_3tier"] = np.select(
    [cohort["geo_prev"] == 0, cohort["g4_unique_prev"].between(1, 3)],
    ["0 vùng GEO", "1 - 3 vùng GEO"],
    default="≥ 4 vùng GEO",
)
inter = cohort.pivot_table(index="et_novelty", columns="geo_3tier", values="h1", aggfunc="mean") * 100
display(inter.loc[["0 TRX ở T-1", "0 mã mới", "1 mã mới", "≥ 2 mã mới"], ["0 vùng GEO", "1 - 3 vùng GEO", "≥ 4 vùng GEO"]].round(2))
"""),
        md("## Slide H · Xuất lại bộ 8 Slide & PDF `MBD_MINI_DEEP_INSIGHTS_DECK.pdf`"),
        code("""
REBUILD_PDF = False
if REBUILD_PDF:
    subprocess.run([sys.executable, str(ROOT / "src/scripts/build_clean_presentation.py")], cwd=ROOT, check=True)

for idx in range(1, 9):
    img = SLIDE_DIR / f"slide_{idx:02d}.png"
    if img.exists():
        display(Markdown(f"### Slide {idx:02d}"))
        display(Image(filename=str(img), width=900))
"""),
    ]
    return write_nb("05_report2_deep_insights.ipynb", cells)


def main():
    built = [build_nb01(), build_nb02(), build_nb03(), build_nb04(), build_nb05()]
    for p in built:
        print(f"Built {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
