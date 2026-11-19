# Công cụ EDA hiện hành

- `analyze_dataset_eda.py`: tổng hợp raw, khách–tháng, thống kê nhãn và CI theo khách;
  giữ vì các biểu đồ hiện hành dùng bảng/cache do script này tạo.
- `complete_report_eda.py` và `publish_report_eda.ps1`: tính, kiểm tra rồi công bố `docs/report/`.
- `build_eda_notebook.py`, `execute_eda_notebook.py`: tạo/chạy notebook EDA.
- `sync_preparation_notebook.py --check`: kiểm tra đồng bộ notebook 03 và 04, không chạy raw.
- `execute_data_notebooks.py`: chạy phiên bản chuẩn bị hiện tại (notebook 03).
- `validate_prepared_data.py`, `validate_eda.py`, `validate_dataset_eda.py`: kiểm tra dữ liệu/EDA.
- `plot_customer_month_log_coverage.py`: độ phủ khách–tháng TRX/GEO/dialog.
- `plot_source_presence_target.py`: tỷ lệ mua theo nguồn, CI bootstrap, n, tỷ trọng và Lift.
- `plot_spearman_heatmap.py`: tương quan đặc trưng, dùng cohort từ `configs/eda_analysis.json`.
- `audit_trx_zero_negative.py`, `check_trans_count_target_history.py`: audit số liệu nguồn.
- `remove_retired_workflows.ps1`: nhật ký thao tác xóa đã yêu cầu; danh sách ở
  `docs/cleanup_20261116/deletion_plan.json`. Không phải công cụ tự phát hiện file lỗi thời.

Code vẽ từng ảnh nằm trong các thư mục `outputs/*_charts/`; danh mục tại
[`CURRENT_REPORT.md`](../../docs/CURRENT_REPORT.md). Các script baseline, fit mô hình,
đánh giá mô hình và deck cũ đã được xóa. PCA/K-means hội thoại là khám phá EDA,
không phải baseline propensity.
