# Notebook của dự án

Thư mục gồm **14 notebook** bám sát 2 file báo cáo (`MBD MINI UNDERSTANDING DATA.pdf` và `MBD_MINI_DEEP_INSIGHTS_DECK.pdf`), quy trình chuẩn bị dữ liệu và bộ baseline mô hình song song $H=1$ & $H=3$. Mở notebook bằng kernel Python của `.venv` tại thư mục gốc dự án.

| Notebook | Phạm vi & Đối chiếu Báo cáo |
| --- | --- |
| [`01_report1_overview_and_targets.ipynb`](01_report1_overview_and_targets.ipynb) | **Báo cáo 1 (Slides 1–7):** Tổng quan dữ liệu, ERD, kiểm kê khóa nối, độ phủ thời gian 4 nguồn, 4 nhãn sản phẩm (`target_1..4`), xu hướng `y_any`, tỷ lệ mua quay lại & số tháng có mua/khách |
| [`02_report1_trx.ipynb`](02_report1_trx.ipynb) | **Báo cáo 1 (Slides 8–12):** Chất lượng dữ liệu TRX, cơ cấu `currency`, cặp thuộc tính null, phân vị số GD & amount mã 11, đường cong Lorenz, RFM (Mua vs Không mua), tỷ trọng & Lift theo `event_type` |
| [`03_report1_geo.ipynb`](03_report1_geo.ipynb) | **Báo cáo 1 (Slides 13–16):** Tổng quan `geohash_4/5/6`, đa dạng ô `g4` & tỷ lệ ô chính (`70,6%`), GEO quanh tháng mua & quanh ngày nhãn (`±30 ngày` Moscow), tín hiệu GEO & Lift |
| [`04_report1_dialog_and_cross_source.ipynb`](04_report1_dialog_and_cross_source.ipynb) | **Báo cáo 1 (Slides 17–24):** Tổng quan DIALOG (embedding 768 chiều, độ phủ), DIALOG quanh tháng mua & `±30 ngày` nhãn, đặc điểm tương tác, heatmap giờ × thứ, khám phá embedding (PCA + K-means), có nguồn hành vi (GEO/DIALOG) & tỷ lệ mua |
| [`05_report2_deep_insights.ipynb`](05_report2_deep_insights.ipynb) | **Báo cáo 2 (Slides A–H):** Tái lập trọn vẹn 8 slide phân tích chuyên sâu (Đường cong chữ U & `0 TRX`, Tần suất × Số tháng lịch sử & Tỷ số `T-1`, Kiểm định đa tầng DIALOG bác bỏ Simpson, Shift-Share `04→11/2022`, Ma trận `A→B` & đồng mua `647` ca, `event_type` mới/đa dạng, `TRX × GEO` độc lập) |
| [`06_data_processing.ipynb`](06_data_processing.ipynb) | Chuẩn bị dữ liệu trước cutoff (`src/propensity_data/`), không huấn luyện mô hình |
| [`07_merge_processed.ipynb`](07_merge_processed.ipynb) | Hợp nhất các bảng đã xử lý vào `processed/` (`src/propensity_data/merge.py`) |
| [`08_baseline_top50.ipynb`](08_baseline_top50.ipynb) | **Baseline Ưu tiên 1 (`50` biến tinh gọn, `H=1` & `H=3`):** Toàn lịch sử từ 2021, burn-in 3 tháng (`prior_purchase`), embargo 2 tháng cho `H=3`, Calibration tháng 12, Holdout `2023-01` & `2022-11` |
| [`09_baseline_full252.ipynb`](09_baseline_full252.ipynb) | **Baseline Ưu tiên 2 (`252` biến đầy đủ, `H=1` & `H=3`):** Bậc thang ablation 10 cấu hình (`49 → 252`), kiểm định paired bootstrap **Top-50 vs Full-252** và phân tích đóng góp biến ngoài Top-50 |
| [`10_baseline_benchmark_suite.ipynb`](10_baseline_benchmark_suite.ipynb) | **Bộ Đánh Giá Baseline Đa Tầng & Kiểm Chứng Out-of-Client (`H=1` & `H=3`):** Đầy đủ các tầng `Random → Rule RF / Rule EDA → LogReg Simple-10 / Top-50 → LightGBM/XGBoost Default (Raw 203 & Top-50) → CatBoost Final (+ Calibration)`, Brier Score + 1.000 Client-Clustered Bootstrap CI, Reliability Curve, chẩn đoán `B0` pooled artifact và kiểm chứng **5-fold `OOT + OOC` (0% trùng `client_id`)** |
| [`11_baseline_v2_colab.ipynb`](11_baseline_v2_colab.ipynb) | **Baseline v2 (`Protocol B` trên Colab):** Rolling theo tháng × 5-fold OOC cho `H=1` (`2022-09..2023-01`) |
| [`12_fe_v2_and_verification_colab.ipynb`](12_fe_v2_and_verification_colab.ipynb) | **Verification & FE-v2 (`<= 120` biến) trên Colab GPU:** Group 2.4, Group 3 và FE-v2 |
| [`13_h3_propensity_full_colab.ipynb`](13_h3_propensity_full_colab.ipynb) | **H=3 Stage 1 & Stage 2 Full Colab GPU:** 5 Top-50 Baselines, 6 Warning Audits, EDA FE-150 & Locked OOT `2022-11` |
| [`14_h3_wide_rolling_colab.ipynb`](14_h3_wide_rolling_colab.ipynb) | **H=3 kho biến rộng:** 5 baseline, log theo fold, Val rolling, khóa cấu hình và chấm hai test 10–11 từ cùng lượt fit. |

## Tái tạo & kiểm tra đồng bộ

- Tạo lại 5 notebook báo cáo (`01`–`05`):
  ```powershell
  .venv/Scripts/python.exe src/scripts/build_report_notebooks.py
  ```
- Kiểm tra đồng bộ mã nguồn nhúng trong notebook `06` và `07`:
  ```powershell
  .venv/Scripts/python.exe src/scripts/sync_preparation_notebook.py --check
  ```
