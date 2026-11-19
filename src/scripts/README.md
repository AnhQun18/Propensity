# Script EDA và công cụ đối chiếu

- `analyze_dataset_eda.py`: tổng hợp raw, khách–tháng, thống kê nhãn và CI theo khách;
  dùng quy ước UTC cũ, giữ để đối chiếu với bảng/cache đã xuất.
- `complete_report_eda.py` và `publish_report_eda.ps1`: workflow báo cáo UTC cũ; cần tính lại theo Moscow trước khi dùng làm kết quả hiện hành.
- `build_eda_notebook.py`, `execute_eda_notebook.py`: workflow EDA cũ; chỉ dùng khi cần tái tạo kết quả cũ để đối chiếu. Notebook tạo ra không nằm trong bộ notebook hiện hành.
- `build_eda_workflow_notebooks.py`: đồng bộ mã EDA từ `src/` vào các notebook
  theo chủ đề và số thứ tự; xem [`notebooks/README.md`](../../notebooks/README.md).
- `sync_preparation_notebook.py --check`: kiểm tra đồng bộ notebook 03 và 04, không chạy raw.
- `execute_data_notebooks.py`: chạy phiên bản chuẩn bị hiện tại (notebook 03).
- `validate_prepared_data.py`, `validate_eda.py`, `validate_dataset_eda.py`: kiểm tra dữ liệu/EDA.
- `plot_customer_month_log_coverage.py`: độ phủ khách–tháng TRX/GEO/dialog.
- `plot_source_presence_target.py`: tỷ lệ mua theo nguồn, CI bootstrap, n, tỷ trọng và Lift.
- `plot_spearman_heatmap.py`: tương quan đặc trưng, dùng cohort từ `configs/eda_analysis.json`.
- `audit_trx_zero_negative.py`, `check_trans_count_target_history.py`: audit số liệu nguồn.
- `remove_retired_workflows.ps1`: công cụ của đợt dọn workflow cũ; không phải công cụ tự phát hiện file lỗi thời.

Code vẽ từng ảnh nằm trong các thư mục `outputs/*_charts/`; xem
[đánh giá phạm vi EDA](../../docs/eda_two_lists_audit_20261118.md). Các script baseline, fit mô hình,
đánh giá mô hình và deck cũ đã được xóa. PCA/K-means hội thoại là khám phá EDA,
không phải baseline propensity.
