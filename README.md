# MBD-mini: khám phá và chuẩn bị dữ liệu propensity

Repo này dùng bộ dữ liệu MBD-mini để kiểm tra chất lượng dữ liệu, phân tích hành vi khách hàng và chuẩn bị bảng khách hàng–tháng cho bài toán dự báo mua sản phẩm. Dữ liệu có bốn nhãn sản phẩm; nhãn gộp `y_any` cho biết khách hàng có mua ít nhất một sản phẩm trong khoảng dự báo. Repo hiện tập trung vào EDA, audit và chuẩn bị dữ liệu, chưa có pipeline huấn luyện mô hình propensity.

**Bắt đầu từ [từ điển dữ liệu MBD-mini](docs/dict/mbd_mini.md)** để xem ý nghĩa các bảng, cột, khóa nối, nhãn và những giới hạn diễn giải. [Danh mục dữ liệu](docs/DATASETS.md) mô tả nguồn và quy mô; [tài liệu MBD-mini](docs/MBD%20MINI%20UNDERSTANDING%20DATA%20%281%29.pdf) lưu bản PDF tham khảo; [báo cáo hiện hành](docs/CURRENT_REPORT.md) chỉ tới các kết quả và mã tạo biểu đồ.

## Cấu trúc repo

| Đường dẫn | Nội dung |
| --- | --- |
| `data/` | Dữ liệu gốc cục bộ; giữ nguyên, không đưa lên Git |
| `sample_data/` | Dữ liệu ví dụ để xem cấu trúc; không thay thế phân tích toàn bộ khách hàng |
| `notebooks/` | Notebook xem dữ liệu, chuẩn bị dữ liệu và EDA |
| `src/` | Package xử lý dữ liệu, EDA và mã audit |
| `src/scripts/` | Script chạy phân tích, kiểm tra và công bố báo cáo |
| `configs/` | Cấu hình các quy trình |
| `tests/` | Kiểm tra tự động |
| `docs/` | Từ điển dữ liệu, tài liệu nghiên cứu và báo cáo; `docs/report/` và `docs/deliverables/` chứa các bản công bố |
| `outputs/` | Cache và kết quả chạy cục bộ |
| `pyproject.toml` | Dependency, CLI và cấu hình đóng gói |

## Cài đặt và kiểm tra

Yêu cầu Python 3.10 trở lên. Đặt dữ liệu gốc tại `data/MBD-mini/` nếu cần chạy phân tích trên toàn bộ dữ liệu.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[eda,prepare,notebook]"
.venv/Scripts/python.exe -m unittest discover -s tests
.venv/Scripts/python.exe src/scripts/sync_preparation_notebook.py --check
```

## Luồng làm việc

1. Xem [notebook giới thiệu dữ liệu](notebooks/01_data_preview.ipynb) với `sample_data/`; đọc [giới hạn của mẫu](sample_data/README.md) trước khi diễn giải kết quả.
2. Chuẩn bị dữ liệu bằng [notebook 03](notebooks/03_data_processing.ipynb), package `src/propensity_data/` và [cấu hình](configs/propensity_processing.json). Dùng [notebook 04](notebooks/04_merge_processed.ipynb) khi cần ghép dữ liệu đã xử lý.
3. Khám phá dữ liệu bằng [notebook EDA](notebooks/eda.ipynb), package `src/mbd_mini_eda/` và các script trong `src/scripts/`. Kết quả phân tích nằm tại `outputs/`, `docs/report/` và `docs/deliverables/`.

## Quy ước nghiên cứu

- Đơn vị quan sát là khách hàng–tháng. Cutoff là **00:00 ngày đầu tháng nhãn theo giờ Moscow (`Europe/Moscow`, UTC+3)**; chỉ dùng sự kiện có `event_time < cutoff`.
- Đặc trưng dùng toàn bộ lịch sử có sẵn trước cutoff của từng nguồn. Khi nghiên cứu horizon, so sánh 1, 2 và 3 tháng trên cùng cohort có đủ nhãn tương lai; không coi tháng chưa quan sát là nhãn 0.
- EDA cần kiểm kê toàn bộ khách hàng và ít nhất một năm nhãn. Dữ liệu trong `sample_data/` chỉ phục vụ ví dụ và kiểm tra quy trình.
- Một số artifact cũ được tạo theo quy ước UTC, cửa sổ lịch sử cố định hoặc mẫu khách. Chúng là kết quả phương pháp cũ cho tới khi được tính lại theo quy ước hiện hành; xem [nhật ký phạm vi EDA](docs/eda_two_lists_audit_20261118.md).
- Timestamp nguồn thiếu timezone cần giả định nguồn được ghi rõ. Các mã `currency`, `geohash` và `event_type` chưa có ánh xạ nghiệp vụ đầy đủ, nên không tự suy ra đơn vị tiền hay tọa độ.

Quy tắc chi tiết của dự án nằm trong [AGENTS.md](AGENTS.md).
