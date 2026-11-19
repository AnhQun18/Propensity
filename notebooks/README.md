# Notebook hiện hành

| Notebook | Mục đích |
| --- | --- |
| `01_data_preview.ipynb` | Xem schema và dữ liệu mẫu |
| `03_data_processing.ipynb` | Phiên bản chuẩn bị dữ liệu mới nhất, dùng `configs/propensity_processing.json` |
| `04_merge_processed.ipynb` | Ghép dữ liệu đã xử lý thành Parquet |
| `eda.ipynb` | EDA và bảng thống kê mô tả |

Notebook 02 đã được thay thế bằng 03. Notebook 05 và toàn bộ workflow baseline đã xóa.
Các notebook chuẩn bị giữ output trống và không tự chạy khi mở. `eda.ipynb` giữ kết quả
EDA đã thực thi. Đồng bộ code thư viện bằng `src/scripts/sync_preparation_notebook.py`;
kiểm tra bằng `--check`. Kết quả chuẩn bị hiện tại ở `outputs/propensity_processing_v3/`.
