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
