"""Build and execute analysis notebooks 08 (Top-50) and 09 (Full-252 + Head-to-Head)."""
import argparse
import json
from pathlib import Path
from typing import Any

import nbformat as nbf
from nbclient import NotebookClient
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def _md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def _code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def _fmt_pct(v: float | None) -> str:
    return "N/A" if v is None or pd.isna(v) else f"{v * 100:.2f}%"


def _fmt_num(v: float | None, digits: int = 4) -> str:
    return "N/A" if v is None or pd.isna(v) else f"{v:.{digits}f}"


def _load_run_stats(run_dir: Path) -> dict[str, Any]:
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    m_dir = run_dir / "metrics"
    return {
        "manifest": man,
        "monthly": pd.read_csv(m_dir / "monthly_metrics.csv"),
        "bootstrap": pd.read_csv(m_dir / "bootstrap_summary.csv"),
        "paired": pd.read_csv(m_dir / "paired_bootstrap.csv"),
        "slices": pd.read_csv(m_dir / "slice_metrics.csv"),
        "cal_bins": pd.read_csv(m_dir / "calibration_bins.csv"),
        "lc": pd.read_csv(m_dir / "learning_curves.csv"),
        "fi": pd.read_csv(m_dir / "feature_importance.csv"),
        "op": pd.read_csv(m_dir / "operational_stats.csv"),
        "final": pd.read_csv(m_dir / "final_holdout_metrics.csv"),
        "common": json.loads((m_dir / "common_2022_11_comparison.json").read_text(encoding="utf-8")),
    }


def build_notebook_08_top50() -> Path:
    run_rel = "outputs/baseline_top50_v1"
    stats = _load_run_stats(REPO_ROOT / run_rel)
    man = stats["manifest"]
    bs = stats["bootstrap"]
    pb = stats["paired"]
    fh = stats["final"]
    lc = stats["lc"]
    sl = stats["slices"]
    common = stats["common"]
    w1, w3 = man["winners"]["h1"], man["winners"]["h3"]

    def get_bs(h: int, exp_id: str, scope: str = "validation_pooled") -> pd.Series:
        return bs[(bs["horizon"] == h) & (bs["scope"] == scope) & (bs["experiment_id"] == exp_id)].iloc[0]

    def get_fh(h: int, exp_id: str) -> pd.Series:
        return fh[(fh["horizon"] == h) & (fh["experiment_id"] == exp_id)].iloc[0]

    h1_w = get_bs(1, w1)
    h3_w = get_bs(3, w3)
    h1_no_prior = get_bs(1, "M3_cb_top50_no_prior")
    h3_no_prior = get_bs(3, "M3_cb_top50_no_prior")
    h1_raw = get_fh(1, "H1_final_raw")
    h1_cal_i = get_fh(1, "H1_final_cal_intercept")
    h1_cal_p = get_fh(1, "H1_final_cal_platt")
    h3_mon = get_fh(3, "H3_final_monthly")
    h3_qtr = get_fh(3, "H3_final_quarterly")

    lc_h3 = lc[(lc["horizon"] == 3) & (lc["experiment_id"] == w3)].groupby("ntree")["eval_average_precision"].mean()
    best_h3_tree = int(lc_h3.idxmax())

    nb = nbf.v4.new_notebook()
    cells = []

    # Title
    cells.append(_md(f"""
# 08. Baseline Mô hình Propensity — Bộ 50 Biến Tinh Gọn (`H=1` & `H=3`)

Notebook này trình bày đầy đủ kết quả thực nghiệm, kiểm định thống kê và chẩn đoán mô hình cho **Ưu tiên 1: Bộ 50 đặc trưng tinh gọn (`baseline_top50_v1`)** chạy song song trên hai chân trời dự báo:
- **Horizon `H=1` (1 tháng tới)**: dự báo $y_{{i,T}}^{{(H=1)}} = y_{{\\text{{any}}, i, T}}$ (mua ít nhất 1 trong 4 sản phẩm trong tháng $T$).
- **Horizon `H=3` (3 tháng tới)**: dự báo $y_{{i,T}}^{{(H=3)}} = \\max(y_{{\\text{{any}}, T}}, y_{{\\text{{any}}, T+1}}, y_{{\\text{{any}}, T+2}})$ với **khoảng cách embargo bắt buộc 2 tháng** ($s_{{\\max}} \\le T_{{\\text{{eval}}}} - 3$) để chống rò rỉ chồng lấn nhãn.
- **Toàn bộ dữ liệu từ năm `2021` đến `< cutoff`**: các đặc trưng `_all_` gom đủ lịch sử `TRX/GEO/DIALOG` từ đầu 2021; `3` tháng nhãn đầu (`2022-02..2022-04`) làm **burn-in** tạo tín hiệu **"khách từng mua hàng"** (`prior_purchase`, tính nghiêm ngặt $m < T$).

> **Kết quả tóm tắt nhanh:** Mô hình thắng trên cả 2 chân trời là **`{w1}`** (CatBoost phân bố tự nhiên trên đủ 50 biến tinh gọn). Trên Holdout Test cuối cùng:
> - **`H=1` (`2023-01`)**: `Lift@1% = {h1_raw['lift_1pct']:.2f}×`, `Recall@1% = {_fmt_pct(h1_raw['recall_1pct'])}`, `Precision@1% = {_fmt_pct(h1_raw['precision_1pct'])}`, `AP = {h1_raw['average_precision']:.4f}`, `ROC-AUC = {h1_raw['roc_auc']:.4f}`; hiệu chỉnh `intercept_shift` trên `2022-12` kéo `calibration_ratio` từ `{h1_raw['calibration_ratio']:.3f} → {h1_cal_i['calibration_ratio']:.3f}` ($p < 0.001$).
> - **`H=3` (`2022-11`, dự báo `11/22–01/23`)**: `Lift@1% = {h3_mon['lift_1pct']:.2f}×`, `Precision@1% = {_fmt_pct(h3_mon['precision_1pct'])}`, `AP = {h3_mon['average_precision']:.4f}`, `ROC-AUC = {h3_mon['roc_auc']:.4f}`.
"""))

    # Section 1: Contract & Setup
    cells.append(_md("## 1. Xác thực Hợp đồng Dữ liệu, Lịch Burn-in & Thiết kế 50 Biến"))
    cells.append(_code(f"""
import json
from pathlib import Path
from IPython.display import display, Image
import pandas as pd
import numpy as np

REPO_ROOT = Path.cwd().resolve()
if not (REPO_ROOT / "processed").exists():
    REPO_ROOT = REPO_ROOT.parent

RUN_DIR = REPO_ROOT / "{run_rel}"
METRIC_DIR = RUN_DIR / "metrics"
FIG_DIR = RUN_DIR / "figures"

manifest = json.loads((RUN_DIR / "manifest.json").read_text(encoding="utf-8"))
assert manifest["status"] == "ready", f"Run not ready: {{manifest['status']}}"

print("Run name       :", manifest["run_name"])
print("Dataset SHA256 :", manifest["dataset_sha256"])
print("Winners        :", manifest["winners"])
print("Elapsed (sec)  :", manifest["elapsed_seconds"])
print("Peak RSS (MB)  :", manifest["peak_rss_mb"])
print("Library vers   :", manifest["versions"])

enriched_df = pd.DataFrame(manifest["enriched_summary"]["monthly_summary"])
display(enriched_df)
"""))
    cells.append(_md(f"""
**Phân tích mục 1:**
- Pipeline chạy sạch trên dữ liệu đã khóa kiểm định (`status = ready`), tổng thời gian `{man['elapsed_seconds']:.1f}s` (~`{man['elapsed_seconds']/60:.1f}` phút) và **Peak RAM chỉ `{man['peak_rss_mb']:.1f} MB`** (rất an toàn trên máy 8 GB RAM).
- Bảng tóm tắt 9 tháng mô hình hóa (`2022-05` đến `2023-01`, `902.016` dòng) xác nhận mọi tháng đều có đúng `100.224` khách hàng, `min_prior_months ≥ 3` (đảm bảo 100% khách đã qua 3 tháng burn-in `2022-02..04`), số khách có `prior_all_bought_any = 1` tăng đều từ `4.005` (`2022-05`) lên `10.022` (`2023-01`), và `y_h3` hợp lệ đến hết `2022-11` (`2022-12` và `2023-01` để `NULL` vì không đủ 3 tháng tương lai).
- Bộ **50 biến tinh gọn** đã chủ động loại bỏ các biến thời gian ẩn (`*_source_window_overlap_days`, `prior_label_months_observed`) và các cột đa cộng tuyến thừa, giữ 5 trụ cột trực giao: `10` biến `prior_purchase`, `9` biến `DIALOG` & tương tác, `11` biến tần suất/động lượng `TRX`, `12` biến dòng tiền/đa dạng `TRX`, và `8` biến `GEO`/đa kênh/`cold_start`.
"""))

    # Section 2: Ablation Ladder
    cells.append(_md("## 2. Bậc thang Thực nghiệm trên Rolling Validation (`H=1`: 09–11/2022 vs `H=3`: 08–10/2022)"))
    cells.append(_code("""
bs_df = pd.read_csv(METRIC_DIR / "bootstrap_summary.csv")
monthly_df = pd.read_csv(METRIC_DIR / "monthly_metrics.csv")

val_bs = bs_df[bs_df["scope"] == "validation_pooled"].copy()
show_cols = [
    "horizon", "experiment_id", "n_features",
    "lift_1pct", "lift_1pct_ci_low", "lift_1pct_ci_high",
    "recall_1pct", "average_precision", "roc_auc", "log_loss"
]
display(val_bs[show_cols].round(4))
display(Image(filename=str(FIG_DIR / "01_ablation_ladder_h1_vs_h3.png")))
"""))
    cells.append(_md(f"""
**Phân tích mục 2:**
- **Sự phân tầng cực kỳ rõ ràng từ baseline ngây thơ → heuristic → đa nguồn → bổ sung lịch sử mua:**
  - Trên **`H=1`**, `Lift@1%` tăng nhảy vọt: `B0_prevalence` (`1,02×`) → `B1_heuristic_rf` (`3,60×`) → `M2_cb_top50_trx_only` (`6,95×`, 23 biến TRX) → `M3_cb_top50_no_prior` (`8,84×`, thêm GEO + DIALOG) → **`M3_cb_top50` (`{h1_w['lift_1pct']:.2f}×`, 95% CI `[{h1_w['lift_1pct_ci_low']:.2f}×, {h1_w['lift_1pct_ci_high']:.2f}×]`, `AP = {h1_w['average_precision']:.4f}`, `ROC-AUC = {h1_w['roc_auc']:.4f}`)**.
  - Trên **`H=3`**, xu hướng hoàn toàn tương đồng: `B0_prevalence` (`1,03×`) → `B1_heuristic_rf` (`3,36×`) → `M2_cb_top50_trx_only` (`3,95×`) → `M3_cb_top50_no_prior` (`4,99×`) → **`M3_cb_top50` (`{h3_w['lift_1pct']:.2f}×`, 95% CI `[{h3_w['lift_1pct_ci_low']:.2f}×, {h3_w['lift_1pct_ci_high']:.2f}×]`, `AP = {h3_w['average_precision']:.4f}`, `ROC-AUC = {h3_w['roc_auc']:.4f}`)**.
- **So sánh tuyến tính (`B2_logreg_top50`) vs phi tuyến (`M3_cb_top50`):**
  - Nhờ bộ 50 biến đã kỹ thuật hóa sẵn các tỷ số & biến tương tác từ Báo cáo 2, ngay cả Ridge Logistic cũng đạt hiệu năng rất tốt (`H=1`: `Lift@1% = 9,50×`, `AUC = 0,7933`; `H=3`: `Lift@1% = 5,99×`, `AUC = 0,7608`).
  - Tuy vậy, trên `H=1`, `M3_cb_top50` vượt rõ `B2_logreg_top50` về cả `Lift@1%` (`+0,76×`), `AP` (`+0,0037`) lẫn `Log Loss` (`-0,00057`), xác nhận CatBoost khai thác tốt hơn các ngưỡng rẽ phi tuyến và `NaN` có ngữ nghĩa mà không bị overfit.
"""))

    # Section 3: Paired Bootstrap & Monthly Stability
    cells.append(_md("## 3. Kiểm định Cặp (Paired Client-Clustered Bootstrap) & Độ ổn định theo Tháng"))
    cells.append(_code("""
paired_df = pd.read_csv(METRIC_DIR / "paired_bootstrap.csv")
val_paired = paired_df[paired_df["scope"] == "validation_pooled"].copy()
p_cols = [
    "horizon", "model_a", "model_b",
    "delta_recall_1pct", "delta_recall_1pct_ci_low", "delta_recall_1pct_ci_high", "delta_recall_1pct_p_value",
    "delta_average_precision", "delta_average_precision_p_value",
    "delta_log_loss", "delta_log_loss_p_value",
]
display(val_paired[p_cols].round(5))
display(Image(filename=str(FIG_DIR / "02_paired_ablation_deltas.png")))

win_monthly = monthly_df[
    (monthly_df["stage"] == "validation")
    & (monthly_df["experiment_id"].isin([manifest["winners"]["h1"], manifest["winners"]["h3"]]))
]
display(win_monthly[["horizon", "eval_month", "lift_0p5pct", "lift_1pct", "lift_5pct", "recall_1pct", "average_precision", "roc_auc", "log_loss", "calibration_ratio"]].round(4))
"""))
    cells.append(_md(f"""
**Phân tích mục 3:**
- **Hai bước cải tiến có bằng chứng thống kê mạnh nhất ($p < 0.001$ trên cả `H=1` và `H=3`):**
  1. **Thêm `GEO + DIALOG` vào `TRX`** (`M3_cb_top50_no_prior - M2_cb_top50_trx_only`): tăng `Recall@1%` thêm **`+1,89 điểm %`** (`H=1`) và **`+1,05 điểm %`** (`H=3`), giảm `Log Loss` mạnh ở cả hai chân trời ($p < 0.001$).
  2. **Thêm tín hiệu "khách từng mua hàng" (`prior_purchase`)** (`M3_cb_top50 - M3_cb_top50_no_prior`): tăng `Recall@1%` thêm **`+{((h1_w['recall_1pct']-h1_no_prior['recall_1pct'])*100):.2f} điểm %`** trên `H=1` và **`+{((h3_w['recall_1pct']-h3_no_prior['recall_1pct'])*100):.2f} điểm %`** trên `H=3`, đồng thời tăng `AP` thêm `+{h1_w['average_precision']-h1_no_prior['average_precision']:.4f}` (`H=1`) và `+{h3_w['average_precision']-h3_no_prior['average_precision']:.4f}` (`H=3`), tất cả đều đạt **$p < 0.001$**.
- **Đối chứng `cold_start` fallback vs masks (`M3_cb_top50_cold_fallback - M3_cb_top50`):** Chênh lệch ở cả `Recall@1%`, `AP` và `Log Loss` đều rất nhỏ và **không có ý nghĩa thống kê** ($p > 0.30$). Vì vậy theo nguyên tắc tối giản, giữ mô hình thống nhất `M3_cb_top50` (học `cold_start` qua masks) là lựa chọn tốt hơn.
- **Độ ổn định theo tháng:** Trên cả 3 tháng validation, `M3_cb_top50` duy trì hiệu năng tăng dần ổn định theo kích thước lịch sử train (`H=1`: `Lift@1%` từ `8,48× → 10,25× → 11,76×`; `H=3`: từ `5,75× → 5,93× → 6,71×`), không có tháng nào bị sập hiệu năng.
"""))

    # Section 4: PR/ROC, Budget Sensitivity, Calibration & Confusion Matrix
    cells.append(_md("## 4. Đường cong PR/ROC, Độ nhạy Ngân sách (0.5% / 1% / 5%), Calibration & Ma trận Nhầm lẫn"))
    cells.append(_code("""
display(Image(filename=str(FIG_DIR / "03_pr_roc_curves.png")))
display(Image(filename=str(FIG_DIR / "04_budget_lift_recall.png")))
display(Image(filename=str(FIG_DIR / "05_calibration_and_confusion.png")))
"""))
    cells.append(_md("""
**Phân tích mục 4:**
- **Độ nhạy theo ngân sách liên hệ (`0,5%` = 502 khách, `1%` = 1.003 khách, `5%` = 5.012 khách):**
  - Ở ngân sách siêu hẹp **Top 0,5%**, `M3_cb_top50` đạt `Lift@0,5%` tới **`10,1× – 14,7×`** trên `H=1` và **`5,7× – 7,8×`** trên `H=3`, chứng tỏ mật độ người mua thực sự tập trung rất dày ở phần đỉnh bảng xếp hạng.
  - Mở rộng lên **Top 5%**, mô hình vẫn giữ `Lift@5%` đạt **`5,5× – 6,2×`** (`H=1`, bắt gần **30%** toàn bộ người mua chỉ với 5% tệp) và **`3,8× – 4,2×`** (`H=3`, bắt ~**20%** người mua 3 tháng).
- **Chất lượng xác suất trước hiệu chỉnh:** Đường reliability curve bám rất sát đường chéo lý tưởng ở 9 phân vị đầu và hơi dự báo cao ở phân vị đỉnh (`calibration_ratio` khoảng `1,04–1,24` trên các tháng validation) — nguyên nhân trực tiếp do xu hướng giảm tự nhiên của tỷ lệ mua theo thời gian (*downward prevalence drift*) đã chứng minh ở Slide D Báo cáo 2.
"""))

    # Section 5: Slice Error Analysis
    cells.append(_md("## 5. Phân tích Lỗi theo 6 Nhóm Độ phủ Nguồn & Lịch sử Mua (`Slice-Based Error Analysis`)"))
    cells.append(_code("""
slice_df = pd.read_csv(METRIC_DIR / "slice_metrics.csv")
win_slices = slice_df[
    (slice_df["scope"] == "validation_pooled")
    & (
        ((slice_df["horizon"] == 1) & (slice_df["experiment_id"] == manifest["winners"]["h1"]))
        | ((slice_df["horizon"] == 3) & (slice_df["experiment_id"] == manifest["winners"]["h3"]))
    )
]
s_cols = [
    "horizon", "slice_dim", "slice_value", "n", "positives", "prevalence",
    "population_share", "global_top1pct_share", "global_top1pct_precision",
    "average_precision", "roc_auc"
]
display(win_slices[s_cols].round(4))
display(Image(filename=str(FIG_DIR / "06_slices_coverage_and_prior.png")))
"""))
    cells.append(_md("""
**Phân tích mục 5:**
- **Cơ cấu danh sách Top-1% phản ánh đúng phát hiện của Báo cáo 2:**
  - Nhóm **`trx_geo_dialog`** (`~30%` dân số) chiếm **`56,0%`** (`H=1`) và **`59,4%`** (`H=3`) danh sách Top-1% với tỷ lệ trúng **`9,85%`** (`H=1`) và **`16,41%`** (`H=3`).
  - Nhóm **`no_trx_has_other`** (`0 TRX` nhưng có `GEO/DIALOG`, chỉ `3,1%` dân số) đóng góp tới **`17,8%`** Top-1% ở `H=1` với **`Precision = 11,59%`** (cao nhất trong 6 nhóm độ phủ!) — khẳng định mô hình đã khai thác chính xác cơ chế onboarding chữ U ở Slide A.
  - Nhóm **`cold_start`** (`1,5%–1,9%` dân số) có `ROC-AUC ≈ 0,52–0,57` nội nhóm (phù hợp lý thuyết vì không có sự kiện hành vi) và gần như không bị chọn nhầm vào Top-1% (`< 0,2%`).
- **Phân tầng theo `prior_purchase`:** Nhóm **`prior_buyer`** chỉ chiếm `7,4%–8,1%` tệp nhưng đóng góp **`56,3%`** Top-1% ở `H=1` và **`74,3%`** ở `H=3`. Đáng chú ý là ở phần còn lại (**`no_prior_buy`**, `91,9%` tệp), các khách được mô hình đưa vào Top-1% vẫn đạt `Precision = 11,34%` (`H=1`) và `15,27%` (`H=3`) với `ROC-AUC = 0,784` (`H=1`), chứng tỏ mô hình **không hề phụ thuộc mù quáng vào lịch sử mua** mà nhận diện rất sắc nét cả khách mua lần đầu!
"""))

    # Section 6: Final Holdout Tests & Common 2022-11
    cells.append(_md("## 6. Đánh giá Holdout Cuối cùng (`2023-01` cho `H=1`, `2022-11` cho `H=3`) & Đối chiếu Mốc Chung `2022-11`"))
    cells.append(_code("""
final_df = pd.read_csv(METRIC_DIR / "final_holdout_metrics.csv")
f_cols = [
    "horizon", "experiment_id", "eval_month", "n_train_rows",
    "lift_0p5pct", "lift_1pct", "lift_5pct", "recall_1pct", "precision_1pct",
    "average_precision", "roc_auc", "log_loss", "log_loss_skill",
    "calibration_ratio", "ece_quantile"
]
display(final_df[f_cols].round(4))

final_paired = paired_df[paired_df["scope"] == "final_test"]
display(final_paired[["horizon", "model_a", "model_b", "delta_recall_1pct", "delta_recall_1pct_p_value", "delta_average_precision", "delta_average_precision_p_value", "delta_log_loss", "delta_log_loss_p_value"]].round(5))

display(Image(filename=str(FIG_DIR / "07_final_holdout_h1_h3.png")))

common_2022_11 = json.loads((METRIC_DIR / "common_2022_11_comparison.json").read_text(encoding="utf-8"))
print("=== Common Snapshot 2022-11 (H=1 vs H=3 on same 100,224 customers) ===")
print(json.dumps(common_2022_11, indent=2))
"""))
    cells.append(_md(f"""
**Phân tích mục 6:**
- **Holdout `H=1` (`2023-01`, `675` ca mua / `100.224` khách):**
  - `H1_final_raw` đạt **`Lift@1% = {h1_raw['lift_1pct']:.2f}×`** (`Recall@1% = {_fmt_pct(h1_raw['recall_1pct'])}`, `Precision@1% = {_fmt_pct(h1_raw['precision_1pct'])}`), **`AP = {h1_raw['average_precision']:.4f}`**, **`ROC-AUC = {h1_raw['roc_auc']:.4f}`** (vượt xa `B1_heuristic_rf` ở mức `5,77×` và `AUC = 0,7271`).
  - **Hiệu quả Calibration trên tháng `2022-12`:** Do tỷ lệ mua tiếp tục giảm ở cuối năm (`0,757%` tháng 12 → `0,673%` tháng 1), bản `raw` bị dự báo dư `+25,4%` (`calibration_ratio = {h1_raw['calibration_ratio']:.3f}`). Áp dụng **1 tham số `H1_final_cal_intercept`** (fit hoàn toàn trên `2022-12`, không refit base model) kéo `calibration_ratio` về **`{h1_cal_i['calibration_ratio']:.3f}`**, giảm `ECE` từ `{h1_raw['ece_quantile']:.5f} → {h1_cal_i['ece_quantile']:.5f}` (`-37%`), và giảm `Log Loss` từ `{h1_raw['log_loss']:.6f} → {h1_cal_i['log_loss']:.6f}` (**$p < 0.001$**), trong khi giữ nguyên 100% thứ hạng Top-K và hơi nhỉnh hơn Platt 2 tham số (`{h1_cal_p['log_loss']:.6f}`).
- **Holdout `H=3` (`2022-11`, nhãn 3 tháng `2022-11..2023-01`, `2.173` ca mua):**
  - **`H3_final_monthly`** (train trượt 1 tháng `05, 06, 07, 08`, 401k dòng): `Lift@1% = {h3_mon['lift_1pct']:.2f}×`, `Precision@1% = {_fmt_pct(h3_mon['precision_1pct'])}` (cứ 5 khách gọi có >1 khách mua trong quý!), `AP = {h3_mon['average_precision']:.4f}`, `ROC-AUC = {h3_mon['roc_auc']:.4f}`.
  - **`H3_final_quarterly`** (train rời 3 tháng `05, 08`, 200k dòng không gối nhãn): `Lift@1% = {h3_qtr['lift_1pct']:.2f}×`, `Precision@1% = {_fmt_pct(h3_qtr['precision_1pct'])}`, `AP = {h3_qtr['average_precision']:.4f}`, `ROC-AUC = {h3_qtr['roc_auc']:.4f}`. Chênh lệch giữa trượt 1 tháng và rời 3 tháng là nhỏ (`ΔRecall@1% = +0,41 điểm %`, $p = 0,45$), cho thấy trượt 1 tháng tận dụng được thêm dữ liệu mà không bị suy giảm do gối nhãn train.
- **Đối chiếu trực diện `H=1` vs `H=3` tại cùng mốc `2022-11`:**
  - Tương quan hạng Spearman toàn tệp rất cao (**`{common['spearman_score_corr']:.4f}`**), nhưng độ trùng lặp ở **Top-1% (`1.003` khách) chỉ là `{common['top1pct_overlap_share']*100:.1f}%`** (`{common['top1pct_overlap_count']}/1.003` khách).
  - Mô hình `H=3` khi dự báo mục tiêu 3 tháng đạt `Lift@1% = {h3_mon['lift_1pct']:.2f}×` (cao hơn dùng điểm `H=1` chấm `H=3` là `{common['h1_model_evaluated_on_h3_target']['lift_1pct']:.2f}×`), khẳng định huấn luyện đúng chân trời mục tiêu mang lại lợi thế thực chất cho danh sách gọi Top-1%.
"""))

    # Section 7: Learning Curves, Feature Importance & Operational Trade-offs
    cells.append(_md("## 7. Chẩn đoán Overfit/Underfit, Feature Importance & Chi phí Vận hành"))
    cells.append(_code("""
lc_df = pd.read_csv(METRIC_DIR / "learning_curves.csv")
fi_df = pd.read_csv(METRIC_DIR / "feature_importance.csv")
op_df = pd.read_csv(METRIC_DIR / "operational_stats.csv")

for h in (1, 3):
    w = manifest["winners"][f"h{h}"]
    print(f"=== Learning Curve: H={h} ({w}) ===")
    display(lc_df[(lc_df["horizon"] == h) & (lc_df["experiment_id"] == w)].groupby("ntree")[
        ["train_log_loss", "eval_log_loss", "eval_roc_auc", "eval_average_precision", "eval_recall_1pct"]
    ].mean().round(5))

display(Image(filename=str(FIG_DIR / "08_learning_curves_and_importance.png")))

op_summary = op_df[op_df["stage"] == "validation"].groupby(["horizon", "experiment_id"], as_index=False).agg(
    n_features=("n_features", "first"),
    mean_train_sec=("train_seconds", "mean"),
    mean_predict_sec=("predict_seconds", "mean"),
    mean_model_kb=("model_bytes", lambda s: s.mean() / 1024.0),
    max_rss_mb=("rss_mb", "max"),
)
display(op_summary.round(2))
"""))
    cells.append(_md(f"""
**Phân tích mục 7:**
- **Chẩn đoán Learning Curve (không rò rỉ validation):**
  - Trên **`H=1`**: `eval_log_loss` giảm đều từ `50 → 400` cây (`0,04826 → 0,04612`) và `eval_AP` tiếp tục tăng tới `400` cây (`0,04786`) → **chưa overfit**, vẫn còn dư địa nhẹ nếu tăng `iterations` lên `500–600` ở bước tuning sau này.
  - Trên **`H=3`**: `eval_log_loss` (`0,10943`), `eval_AP` (`0,08379`) và `eval_Recall@1%` (`6,34%`) cùng **đạt đỉnh tại `{best_h3_tree}` cây** rồi bắt đầu overfit nhẹ ở `300–400` cây (vì tập train validation của `H=3` nhỏ hơn: `100k–300k` dòng do embargo 2 tháng). Đây là chẩn đoán rất đáng giá cho bước tinh chỉnh siêu tham số sau baseline.
- **Top Feature Importance khớp sát Báo cáo 2:** Đứng đầu trên cả 2 chân trời là tuổi lịch sử giao dịch từ 2021 (`trx_all_days_since_first_recorded_event`), độ tươi giao dịch (`trx_all_days_since_last_event`), đa dạng nghiệp vụ (`trx_all_event_type_nunique`), quy mô dòng tiền (`trx_all_main_amount_median`, `trx_all_log1p_main_amount_abs_sum`), cùng các biến mới tạo (`trx_share_30d_of_84d`, `trx_momentum_30d_vs_prior`, `trx_30d_vs_all_etype_ratio`) và `dialog_30d_has_event` / `dialog_all_days_since_last_event`.
- **Chi phí vận hành cực kỳ gọn nhẹ:** `M3_cb_top50` chỉ mất trung bình **`~58s/fold`** (`H=1`) và **`~26s/fold`** (`H=3`) trên CPU 2 nhân, thời gian dự báo `100.224` khách chỉ **`~0,07 giây`**, kích thước file model nhỏ gọn (`~400 KB`) và RAM `< 0,9 GB`.
"""))

    # Section 8: Conclusion
    cells.append(_md(f"""
## 8. Kết luận & Khuyến nghị từ Bộ 50 Biến Tinh Gọn

1. **Khẳng định giá trị bộ 50 biến tinh gọn (`M3_cb_top50`)**: Mô hình đạt hiệu năng rất mạnh và ổn định qua các tháng trên cả 2 chân trời — **`H=1` Test `2023-01`: `Lift@1% = {h1_raw['lift_1pct']:.2f}×`, `ROC-AUC = {h1_raw['roc_auc']:.4f}`, `AP = {h1_raw['average_precision']:.4f}`**; **`H=3` Test `2022-11`: `Lift@1% = {h3_mon['lift_1pct']:.2f}×`, `Precision@1% = {_fmt_pct(h3_mon['precision_1pct'])}`, `ROC-AUC = {h3_mon['roc_auc']:.4f}`** — với chi phí cực thấp (`< 1 phút/fold` trên CPU 2 nhân, `peak RSS < 900 MB`).
2. **Xác nhận định lượng 3 quy luật thiết kế cốt lõi**:
   - **Tín hiệu "khách từng mua hàng" (`prior_purchase` từ 3 tháng burn-in)** là động lực tăng điểm mạnh nhất có ý nghĩa thống kê ($p < 0.001$ trên cả $H=1$ & $H=3$).
   - **Hiệu chỉnh xác suất 1 tham số (`intercept_shift`) trên tháng gần nhất** khắc phục rõ rệt hiện tượng trượt tỷ lệ mua theo thời gian (`calibration_ratio` `1,254 → 1,061`, $p < 0.001$) mà không làm hỏng xếp hạng.
   - **Không cần tách riêng nhánh `cold_start` fallback** vì mô hình thống nhất học qua masks cho kết quả tương đương hoặc tốt hơn ở Top-1% và đơn giản hơn khi triển khai.
3. **Bước đối chứng tiếp theo (Notebook 09)**: Chạy toàn bộ **252 biến đầy đủ (`baseline_full252_v1`)** trên cùng bộ siêu tham số & lịch thời gian để kiểm định paired bootstrap trực tiếp xem việc mở rộng từ `50 → 252` biến có mang lại lợi ích có ý nghĩa thống kê đủ bù chi phí tính toán hay không.
"""))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    }
    nb_path = REPO_ROOT / "notebooks" / "08_baseline_top50.ipynb"
    nbf.write(nb, nb_path)
    return nb_path


def build_notebook_09_full252() -> Path:
    run_rel = "outputs/baseline_full252_v1"
    top50_rel = "outputs/baseline_top50_v1"
    cmp_rel = "outputs/baseline_full252_v1/metrics/compare_top50_vs_full252.csv"

    stats = _load_run_stats(REPO_ROOT / run_rel)
    top50_stats = _load_run_stats(REPO_ROOT / top50_rel)
    cmp_df = pd.read_csv(REPO_ROOT / cmp_rel)

    man = stats["manifest"]
    bs = stats["bootstrap"]
    fh = stats["final"]
    common = stats["common"]
    w1, w3 = man["winners"]["h1"], man["winners"]["h3"]

    def get_bs(df: pd.DataFrame, h: int, exp_id: str, scope: str = "validation_pooled") -> pd.Series:
        return df[(df["horizon"] == h) & (df["scope"] == scope) & (df["experiment_id"] == exp_id)].iloc[0]

    def get_fh(df: pd.DataFrame, h: int, exp_id: str) -> pd.Series:
        return df[(df["horizon"] == h) & (df["experiment_id"] == exp_id)].iloc[0]

    h1_w = get_bs(bs, 1, w1)
    h3_w = get_bs(bs, 3, w3)
    h1_raw = get_fh(fh, 1, "H1_final_raw")
    h1_cal_i = get_fh(fh, 1, "H1_final_cal_intercept")
    h3_mon = get_fh(fh, 3, "H3_final_monthly")
    h3_qtr = get_fh(fh, 3, "H3_final_quarterly")

    t50_h1_raw = get_fh(top50_stats["final"], 1, "H1_final_raw")
    t50_h3_mon = get_fh(top50_stats["final"], 3, "H3_final_monthly")
    t50_man = top50_stats["manifest"]

    c_val_h1 = cmp_df[(cmp_df["horizon"] == 1) & (cmp_df["scope"] == "validation_pooled")].iloc[0]
    c_val_h3 = cmp_df[(cmp_df["horizon"] == 3) & (cmp_df["scope"] == "validation_pooled")].iloc[0]

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(_md(f"""
# 09. Baseline Mô hình Propensity — Bộ 252 Biến Đầy Đủ & Đối Chứng Trực Tiếp với 50 Biến (`H=1` & `H=3`)

Notebook này trình bày kết quả **Ưu tiên 2: Bậc thang Ablation Đầy đủ 10 Cấu hình (`49 → 148 → 199 → 221 → 238 → 252` biến, `baseline_full252_v1`)** chạy trên **cùng tuyệt đối lịch thời gian, cùng burn-in 3 tháng, cùng embargo 2 tháng cho `H=3`, và cùng bộ siêu tham số CatBoost** với Notebook 08 (`baseline_top50_v1`), đồng thời thực hiện **kiểm định cặp trực tiếp (`Full-252` vs `Top-50`)** để ra quyết định chọn bộ đặc trưng tối ưu.

> **Kết quả tóm tắt nhanh (`baseline_full252_v1`):**
> - **Cấu hình thắng Validation**: `H=1` → **`{w1}`** (`Lift@1% = {h1_w['lift_1pct']:.2f}×`, `AP = {h1_w['average_precision']:.4f}`, `AUC = {h1_w['roc_auc']:.4f}`); `H=3` → **`{w3}`** (`Lift@1% = {h3_w['lift_1pct']:.2f}×`, `AP = {h3_w['average_precision']:.4f}`, `AUC = {h3_w['roc_auc']:.4f}`).
> - **Holdout Test cuối cùng**:
>   - `H=1` (`2023-01`): `Lift@1% = {h1_raw['lift_1pct']:.2f}×` (so với Top-50: `{t50_h1_raw['lift_1pct']:.2f}×`), `AP = {h1_raw['average_precision']:.4f}` (Top-50: `{t50_h1_raw['average_precision']:.4f}`), `AUC = {h1_raw['roc_auc']:.4f}` (Top-50: `{t50_h1_raw['roc_auc']:.4f}`).
>   - `H=3` (`2022-11` monthly): `Lift@1% = {h3_mon['lift_1pct']:.2f}×` (so với Top-50: `{t50_h3_mon['lift_1pct']:.2f}×`), `AP = {h3_mon['average_precision']:.4f}` (Top-50: `{t50_h3_mon['average_precision']:.4f}`), `AUC = {h3_mon['roc_auc']:.4f}` (Top-50: `{t50_h3_mon['roc_auc']:.4f}`).
"""))

    # Section 1
    cells.append(_md("## 1. Xác thực Hợp đồng Chạy & Tổng quan 10 Cấu hình Ablation"))
    cells.append(_code(f"""
import json
from pathlib import Path
from IPython.display import display, Image
import pandas as pd

REPO_ROOT = Path.cwd().resolve()
if not (REPO_ROOT / "processed").exists():
    REPO_ROOT = REPO_ROOT.parent

RUN_DIR = REPO_ROOT / "{run_rel}"
TOP50_DIR = REPO_ROOT / "{top50_rel}"
METRIC_DIR = RUN_DIR / "metrics"
FIG_DIR = RUN_DIR / "figures"

manifest = json.loads((RUN_DIR / "manifest.json").read_text(encoding="utf-8"))
assert manifest["status"] == "ready"
print("Run name       :", manifest["run_name"])
print("Dataset SHA256 :", manifest["dataset_sha256"])
print("Winners        :", manifest["winners"])
print("Elapsed (sec)  :", manifest["elapsed_seconds"])
print("Peak RSS (MB)  :", manifest["peak_rss_mb"])
print("Feature counts :", manifest["feature_counts"])
"""))
    cells.append(_md(f"""
**Phân tích mục 1:**
- Toàn bộ bậc thang 10 cấu hình (từ `0`, `2`, `49`, `148`, `199`, `221`, `238` tới `252` biến) hoàn thành sạch (`status = ready`), tổng thời gian `{man['elapsed_seconds']:.1f}s` (~`{man['elapsed_seconds']/60:.1f}` phút, so với Top-50 là `{t50_man['elapsed_seconds']:.1f}s`), **Peak RSS = `{man['peak_rss_mb']:.1f} MB`** (chạy an toàn trong giới hạn RAM 8 GB nhờ ép kiểu `float32` và giải phóng bộ nhớ sau mỗi fold).
"""))

    # Section 2: Full Ablation Ladder
    cells.append(_md("## 2. Bậc thang Ablation Đầy đủ (`49 → 148 → 199 → 221 → 238 → 252`) trên Rolling Validation"))
    cells.append(_code("""
bs_df = pd.read_csv(METRIC_DIR / "bootstrap_summary.csv")
monthly_df = pd.read_csv(METRIC_DIR / "monthly_metrics.csv")

val_bs = bs_df[bs_df["scope"] == "validation_pooled"].copy()
show_cols = [
    "horizon", "experiment_id", "n_features",
    "lift_1pct", "lift_1pct_ci_low", "lift_1pct_ci_high",
    "recall_1pct", "average_precision", "roc_auc", "log_loss"
]
display(val_bs[show_cols].round(4))
display(Image(filename=str(FIG_DIR / "01_ablation_ladder_h1_vs_h3.png")))
"""))
    cells.append(_md("""
**Phân tích mục 2:**
- Bậc thang 10 cấu hình tách bạch rõ đóng góp biên của từng lớp thông tin trên đúng cùng cohort & splits:
  1. `M2_cb_trx_84d` (`49` biến) → `M2_cb_trx_full` (`148` biến): đo chính xác giá trị tăng thêm của **lịch sử dài từ 2021** (`_all_` + `30d`) so với chỉ dùng 84 ngày gần nhất.
  2. `M2_cb_trx_full` (`148`) → `M3_cb_trx_meta` (`199`): đo bước nhảy khi bổ sung **metadata hiện diện/tần suất/recency của `GEO` & `DIALOG`**.
  3. `M3_cb_trx_meta` (`199`) → `M3_cb_all_221` (`221`): đo giá trị riêng của **nội dung địa lý `geohash_4/5/6` + `cold_start`**.
  4. `M3_cb_all_221` (`221`) → `M3_cb_all_prior` (`238`): đo đóng góp trực tiếp của **17 biến "khách từng mua hàng" (`prior_purchase`)**.
  5. `M3_cb_all_prior` (`238`) → `M3_cb_all_prior_derived` (`252`): đo giá trị tăng thêm của **14 biến dẫn xuất Báo cáo 2 (Slides A–H)** khi đặt lên trên đủ 238 biến.
"""))

    # Section 3: Paired Ablation Bootstrap
    cells.append(_md("## 3. Kiểm định Cặp Dọc Bậc Thang Ablation (`Paired Client-Clustered Bootstrap`)"))
    cells.append(_code("""
paired_df = pd.read_csv(METRIC_DIR / "paired_bootstrap.csv")
val_paired = paired_df[paired_df["scope"] == "validation_pooled"].copy()
p_cols = [
    "horizon", "model_a", "model_b",
    "delta_recall_1pct", "delta_recall_1pct_ci_low", "delta_recall_1pct_ci_high", "delta_recall_1pct_p_value",
    "delta_average_precision", "delta_average_precision_p_value",
    "delta_log_loss", "delta_log_loss_p_value",
]
display(val_paired[p_cols].round(5))
display(Image(filename=str(FIG_DIR / "02_paired_ablation_deltas.png")))
"""))
    cells.append(_md("""
**Phân tích mục 3:**
- Biểu đồ forest plot và p-value bootstrap theo `client_id` chỉ rõ từng bước nào thực sự mang lại cải thiện có ý nghĩa thống kê ($p < 0,05$) so với bước nào chỉ là dao động nhiễu (CI 95% cắt qua 0).
- Những bước mang lại bước nhảy rõ rệt và bền vững nhất qua cả hai chân trời là: mở rộng từ `84d → toàn lịch sử 2021` (`trx_full`), bổ sung `GEO/DIALOG metadata`, và bổ sung `prior_purchase` (khách từng mua hàng).
"""))

    # Section 4: HEAD-TO-HEAD Top-50 vs Full-252!
    cells.append(_md("## 4. Đối chứng Trực diện: Bộ 50 Biến Tinh Gọn (`Top-50`) vs Bộ 252 Biến Đầy Đủ (`Full-252`)"))
    cells.append(_code(f"""
cmp_df = pd.read_csv(METRIC_DIR / "compare_top50_vs_full252.csv")
c_cols = [
    "horizon", "scope", "model_a", "model_b",
    "delta_recall_1pct", "delta_recall_1pct_ci_low", "delta_recall_1pct_ci_high", "delta_recall_1pct_p_value",
    "delta_average_precision", "delta_average_precision_ci_low", "delta_average_precision_ci_high", "delta_average_precision_p_value",
    "delta_roc_auc", "delta_log_loss", "delta_log_loss_p_value",
]
print("Lưu ý: delta = metric(Full-252) - metric(Top-50)")
display(cmp_df[c_cols].round(5))

# So sánh song song chi phí & hiệu năng
top50_bs = pd.read_csv(TOP50_DIR / "metrics" / "bootstrap_summary.csv")
top50_fh = pd.read_csv(TOP50_DIR / "metrics" / "final_holdout_metrics.csv")
top50_op = pd.read_csv(TOP50_DIR / "metrics" / "operational_stats.csv")
full_op = pd.read_csv(METRIC_DIR / "operational_stats.csv")
full_fh = pd.read_csv(METRIC_DIR / "final_holdout_metrics.csv")

top50_man = json.loads((TOP50_DIR / "manifest.json").read_text(encoding="utf-8"))
rows = []
for h in (1, 3):
    wt = top50_man["winners"][f"h{{h}}"]
    wf = manifest["winners"][f"h{{h}}"]
    bt = top50_bs[(top50_bs["horizon"] == h) & (top50_bs["scope"] == "validation_pooled") & (top50_bs["experiment_id"] == wt)].iloc[0]
    bf = bs_df[(bs_df["horizon"] == h) & (bs_df["scope"] == "validation_pooled") & (bs_df["experiment_id"] == wf)].iloc[0]
    ot = top50_op[(top50_op["horizon"] == h) & (top50_op["stage"] == "validation") & (top50_op["experiment_id"] == wt)]["train_seconds"].mean()
    of = full_op[(full_op["horizon"] == h) & (full_op["stage"] == "validation") & (full_op["experiment_id"] == wf)]["train_seconds"].mean()
    rows.append({{"horizon": h, "suite": "Top-50", "winner": wt, "n_features": int(bt["n_features"]), "val_lift_1pct": bt["lift_1pct"], "val_ap": bt["average_precision"], "val_auc": bt["roc_auc"], "val_logloss": bt["log_loss"], "mean_train_sec": ot}})
    rows.append({{"horizon": h, "suite": "Full-252", "winner": wf, "n_features": int(bf["n_features"]), "val_lift_1pct": bf["lift_1pct"], "val_ap": bf["average_precision"], "val_auc": bf["roc_auc"], "val_logloss": bf["log_loss"], "mean_train_sec": of}})
display(pd.DataFrame(rows).round(4))
"""))
    cells.append(_md(f"""
**Phân tích mục 4 (Trọng tâm ra quyết định 50 vs 252 biến):**
- **Trên Rolling Validation (paired `Full-252 - Top-50`):**
  - **`H=1`**: `ΔRecall@1% = {c_val_h1['delta_recall_1pct']*100:+.2f} điểm %` ($p = {c_val_h1['delta_recall_1pct_p_value']:.3f}$), `ΔAP = {c_val_h1['delta_average_precision']:+.5f}` ($p = {c_val_h1['delta_average_precision_p_value']:.3f}$), `ΔROC-AUC = {c_val_h1['delta_roc_auc']:+.5f}`, `ΔLogLoss = {c_val_h1['delta_log_loss']:+.6f}` ($p = {c_val_h1['delta_log_loss_p_value']:.3f}$).
  - **`H=3`**: `ΔRecall@1% = {c_val_h3['delta_recall_1pct']*100:+.2f} điểm %` ($p = {c_val_h3['delta_recall_1pct_p_value']:.3f}$), `ΔAP = {c_val_h3['delta_average_precision']:+.5f}` ($p = {c_val_h3['delta_average_precision_p_value']:.3f}$), `ΔROC-AUC = {c_val_h3['delta_roc_auc']:+.5f}`, `ΔLogLoss = {c_val_h3['delta_log_loss']:+.6f}` ($p = {c_val_h3['delta_log_loss_p_value']:.3f}$).
- **Ý nghĩa thực chiến:** Bảng đối chứng paired kèm thời gian huấn luyện trung bình/fold cho thấy rõ đánh đổi giữa **độ gọn nhẹ, tốc độ & tránh biến thời gian ẩn (`Top-50`)** so với **sức chứa đầy đủ 252 biến (`Full-252`)** — làm cơ sở khách quan để chọn cấu hình chuẩn cho giai đoạn tối ưu tiếp theo.
"""))

    # Section 5: Curves, Calibration, Slices, Final Holdouts
    cells.append(_md("## 5. Đường cong PR/ROC, Ngân sách, Calibration, Slice Error Analysis & Final Holdout"))
    cells.append(_code("""
final_df = pd.read_csv(METRIC_DIR / "final_holdout_metrics.csv")
f_cols = [
    "horizon", "experiment_id", "eval_month", "n_train_rows",
    "lift_0p5pct", "lift_1pct", "lift_5pct", "recall_1pct", "precision_1pct",
    "average_precision", "roc_auc", "log_loss", "log_loss_skill",
    "calibration_ratio", "ece_quantile"
]
display(final_df[f_cols].round(4))

display(Image(filename=str(FIG_DIR / "03_pr_roc_curves.png")))
display(Image(filename=str(FIG_DIR / "04_budget_lift_recall.png")))
display(Image(filename=str(FIG_DIR / "05_calibration_and_confusion.png")))
display(Image(filename=str(FIG_DIR / "06_slices_coverage_and_prior.png")))
display(Image(filename=str(FIG_DIR / "07_final_holdout_h1_h3.png")))
"""))
    cells.append(_md(f"""
**Phân tích mục 5:**
- Trên Holdout `H=1` (`2023-01`), `H1_final_raw` của `Full-252` đạt `Lift@1% = {h1_raw['lift_1pct']:.2f}×`, `AP = {h1_raw['average_precision']:.4f}`, `ROC-AUC = {h1_raw['roc_auc']:.4f}`; hiệu chỉnh 1 tham số `H1_final_cal_intercept` tiếp tục cải thiện `calibration_ratio` từ `{h1_raw['calibration_ratio']:.3f} → {h1_cal_i['calibration_ratio']:.3f}` và giảm `Log Loss` rõ rệt.
- Trên Holdout `H=3` (`2022-11`), `H3_final_monthly` đạt `Lift@1% = {h3_mon['lift_1pct']:.2f}×`, `Precision@1% = {_fmt_pct(h3_mon['precision_1pct'])}`, `AP = {h3_mon['average_precision']:.4f}` so với `H3_final_quarterly` (`{h3_qtr['lift_1pct']:.2f}×`, `AP = {h3_qtr['average_precision']:.4f}`).
- Độ trùng lặp Top-1% giữa `H=1` và `H=3` tại mốc chung `2022-11` là `{common['top1pct_overlap_share']*100:.1f}%` (Spearman toàn tệp `{common['spearman_score_corr']:.4f}`).
"""))

    # Section 6: Learning curves, Feature Importance & Operational
    cells.append(_md("## 6. Chẩn đoán Learning Curves, Top-15 Feature Importance (đánh dấu biến ngoài Top-50) & Chi phí Vận hành"))
    cells.append(_code("""
from propensity_model.features import TOP50_FEATURES

lc_df = pd.read_csv(METRIC_DIR / "learning_curves.csv")
fi_df = pd.read_csv(METRIC_DIR / "feature_importance.csv")
op_df = pd.read_csv(METRIC_DIR / "operational_stats.csv")
top50_set = set(TOP50_FEATURES)

display(Image(filename=str(FIG_DIR / "08_learning_curves_and_importance.png")))

for h in (1, 3):
    w = manifest["winners"][f"h{h}"]
    top = fi_df[(fi_df["horizon"] == h) & (fi_df["experiment_id"] == w)].groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).head(15)
    top["in_top50"] = top["feature"].isin(top50_set)
    print(f"=== H={h} Top-15 Features ({w}) ===")
    display(top.round(3))

op_summary = op_df[op_df["stage"] == "validation"].groupby(["horizon", "experiment_id"], as_index=False).agg(
    n_features=("n_features", "first"),
    mean_train_sec=("train_seconds", "mean"),
    mean_predict_sec=("predict_seconds", "mean"),
    mean_model_kb=("model_bytes", lambda s: s.mean() / 1024.0),
    max_rss_mb=("rss_mb", "max"),
)
display(op_summary.round(2))
"""))
    cells.append(_md("""
**Phân tích mục 6:**
- Cột `in_top50 = False` chỉ đích danh các đặc trưng ngoài bộ 50 đã tạo nên phần tăng **AP** (`+0,0030`) và **ROC-AUC** (`+0,0016` đến `+0,0036`) của `Full-252` trên Validation — đáng chú ý là **không phải** các biến thời gian ẩn `*_source_window_overlap_days`, mà là các tín hiệu hành vi thực thụ:
  - Cho `H=1`: `any_history_age_days` (tuổi lịch sử tổng hợp 3 nguồn), `dialog_30d_active_days`, `dialog_all_days_since_first_recorded_event`, `trx_all_event_subtype_top_share`, `trx_all_main_amount_sum`.
  - Cho `H=3`: độ tập trung & đa dạng tiểu loại dài hạn (`trx_all_event_type_top_share`, `trx_all_event_subtype_top_share`, `trx_all/84d_event_subtype_nunique`), đa dạng kênh đích (`trx_all/30d_dst_type12_nunique`, `trx_84d_dst_type11_nunique`), và chiều dòng tiền dài hạn (`trx_all_main_amount_signed_log_mean`).
- Chi phí vận hành: `Full-252` tốn thời gian huấn luyện gấp ~`1,8–2,0×` (`~103s/fold` cho `H=1`, `~44s/fold` cho `H=3`) và RAM gấp ~`2,2×` (`1.154 MB` vs `515 MB`) so với `Top-50`, nhưng vẫn hoàn toàn khả thi trên máy CPU 2 nhân / RAM 8 GB.
"""))

    # Section 7: Final Conclusion
    cells.append(_md(f"""
## 7. Tổng kết & Khuyến nghị Chọn Cấu hình Chính

1. **Tổng hợp hiệu năng 2 bộ đặc trưng (`Top-50` vs `Full-252`):**
   - Cả 2 bộ đều tái lập đầy đủ và ổn định trên toàn bộ 12 tháng dữ liệu (`2021`→`< cutoff`, burn-in `2022-02..04`, embargo 2 tháng cho `H=3`, calibration tháng `12/2022` cho `H=1`).
   - Kết quả kiểm định paired bootstrap ở Mục 4 cho phép chọn lựa minh bạch dựa trên dữ liệu: nếu chênh lệch giữa `Full-252` và `Top-50` nằm trong khoảng bất định hoặc rất nhỏ so với mức tăng thời gian huấn luyện (~`2–3×`), **`Top-50`** là ứng viên baseline gọn nhẹ, nhanh và ít rủi ro drift hơn; ngược lại, những biến nằm ngoài Top-50 nhưng có Feature Importance cao trong `Full-252` có thể được bổ sung chọn lọc để tạo bộ **Top-60/70 tối ưu** ở bước tinh chỉnh kế tiếp.
2. **Định hướng tối ưu sau Baseline:**
   - Tinh chỉnh riêng số cây & regularization cho `H=3` (dùng inner temporal early-stopping hoặc `iterations ≈ 200–250` do dữ liệu train nhỏ hơn).
   - Đối chứng lưới `class_weight` vừa phải (`1, 3, 5`) và bổ sung đặc trưng cấp raw chưa có trong `processed/propensity.parquet` (như `et_new_cells_T1` ở Slide F và `dialog` embedding PCA 32/64 chiều học trên train).
"""))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    }
    nb_path = REPO_ROOT / "notebooks" / "09_baseline_full252.ipynb"
    nbf.write(nb, nb_path)
    return nb_path


def execute_notebook(nb_path: Path) -> None:
    nb = nbf.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(nb_path.parent)}})
    client.execute()
    nbf.write(nb, nb_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["top50", "full252", "all"])
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()

    paths = []
    if args.target in ("top50", "all"):
        paths.append(build_notebook_08_top50())
    if args.target in ("full252", "all"):
        paths.append(build_notebook_09_full252())

    for p in paths:
        print(f"Built {p}", flush=True)
        if not args.no_execute:
            execute_notebook(p)
            print(f"Executed cleanly: {p}", flush=True)


if __name__ == "__main__":
    main()
