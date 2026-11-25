# Script phục vụ 2 báo cáo & chuẩn bị dữ liệu

## Báo cáo 1 (`MBD MINI UNDERSTANDING DATA.pdf`)
- `analyze_dataset_eda.py`: tạo `customer_month_snapshots.parquet` và bảng tổng hợp cho Báo cáo 1.
- `check_trans_count_target_history.py`: kiểm chứng `diff_trans_date` NaN gắn với `trans_count = 0` (Slide 4).
- `audit_trx_nulls.py`, `audit_trx_outliers.py`: kiểm tra chất lượng, cặp null và ngoại lai lớn nhất TRX (Slides 8–9).
- `extend_eda_geo_population.py`, `plot_geo_g4_concentration.py`: đa dạng ô `geohash_4` & tỷ lệ tập trung ô chính `70,6%` (Slide 13).
- `analyze_full_population_temporal_eda.py`: dựng panel `all_customer_months.parquet` & bảng trước/sau tháng mua theo giờ Moscow (Slides 14, 18 & nền cho Báo cáo 2).
- `plot_geo_purchase_event_study.py`: biểu đồ GEO & DIALOG quanh tháng mua (Slides 14, 18).
- `plot_dialog_label_date_transition.py`: biểu đồ GEO & DIALOG quanh ngày ghi nhận nhãn `±30 ngày` Moscow (Slides 15, 19).
- `plot_source_presence_target.py`: tỷ lệ mua theo sự có mặt nguồn GEO / DIALOG ở tháng `T-1` kèm CI bootstrap (Slide 23).

*(Các script vẽ từng biểu đồ lẻ cho TRX, GEO, DIALOG của Báo cáo 1 nằm trong `outputs/trx_charts/`, `outputs/geo_charts/`, `outputs/dialog_charts/`, `outputs/repeat_purchase_review/` và được gọi trực tiếp từ notebooks `01`–`04`.)*

## Báo cáo 2 (`MBD_MINI_DEEP_INSIGHTS_DECK.pdf`)
- `complete_eda_figures_moscow.py`, `extend_eda_monthly_features.py`: tạo hồ sơ hàng tháng `event_type_monthly_profiles.parquet` và `geohash_4_monthly_profiles.parquet` (Slides F–G).
- `audit_insights_checklist.py`: đối soát raw, tạo `segment_cards.csv` và `monthly_segment_cards.csv` (Slide D).
- `explore_geo_trx_deep.py`: tính toán kiểm định `event_type` mới/đa dạng, `trx_surge_ratio`, `geohash_4` mobility, tương quan Spearman (`r = 0,02`) và ma trận `TRX × GEO` (Slides B, F, G).
- `build_clean_presentation.py`: xuất 8 slide PNG và file `MBD_MINI_DEEP_INSIGHTS_DECK.pdf` tại thư mục gốc.

## Notebook, Chuẩn bị dữ liệu & Baseline
- `build_report_notebooks.py`: sinh lại 5 notebook báo cáo (`01`–`05`).
- `build_baseline_notebooks.py`: sinh 2 notebook mô hình baseline (`08`–`09`).
- `sync_preparation_notebook.py`: kiểm tra/đồng bộ mã thư viện nhúng trong notebook `06_data_processing.ipynb` và `07_merge_processed.ipynb`.
- `validate_prepared_data.py`: kiểm tra đầu ra của quy trình chuẩn bị dữ liệu.
