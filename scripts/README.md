# Công cụ tác giả

`sync_preparation_notebook.py` đồng bộ các cell định nghĩa từ `src/propensity_data/`
sang notebook 02, 03 và 04 trong `notebooks/`; dùng `--check` để chỉ so sánh văn bản.
Công cụ không chạy notebook, không import pipeline và không đọc raw.

Ngày 10/11/2026 đã chạy đồng bộ và kiểm thử. `execute_data_notebooks.py` chạy
notebook, lưu bản thực thi và log vào thư mục mới bên trong `outputs/`.

`validate_prepared_data.py OUTPUT_DIR` dùng bộ kiểm tra chung để kiểm tra Parquet
đã xuất mà không tạo lại features. Thêm `--upgrade-contract` để nâng metadata
schema cũ lên v2 sau khi tất cả tháng đạt; metadata cũ được sao lưu. Xem
[hướng dẫn](../docs/propensity_data_preparation.md).

`report_baseline_rule.py --config configs/propensity_baseline.json --output FILE.md`
tạo báo cáo Markdown chỉ gồm số liệu tổng hợp từ so sánh rule đã ready; từ chối
ghi đè file báo cáo có sẵn. Không fit mô hình hoặc mở test.

`review_baseline.py --output OUTPUT_DIR` kiểm tra lại dự đoán validation đã lưu,
xuất CSV tổng hợp nhiều K, nhóm theo lịch sử mua trước cutoff, overlap với rule,
cumulative gain và biểu đồ tỷ lệ nhãn. Script đối chiếu metric cũ, kiểm tra hash
nguồn, trích đường train Logloss từ model cuối, từ chối ghi đè output và không
train hoặc đọc dự đoán/feature test. Xem
[báo cáo baseline](../BASELINE_REPORT.md); các CSV/PNG công bố ở
[`docs/baseline_diagnostics_20261110/`](../docs/baseline_diagnostics_20261110/).
Cần cài cả extra `baseline` và `plots` để xuất biểu đồ:
`pip install -e ".[baseline,plots]"`.
