# MBD-mini: Khám phá dữ liệu, Phân tích chuyên sâu & Chuẩn bị dữ liệu Propensity

LINK CANVA: [PROPENSITY](https://canva.link/yq4zss72ntwfh2v)

Repo chứa 2 báo cáo PDF, tài liệu hướng dẫn cách tính biểu đồ, bộ notebook đã gom gọn theo từng phần của báo cáo và mã nguồn tái lập cho bộ dữ liệu ngân hàng **MBD-mini**.

## Báo cáo chính
- **[`MBD MINI UNDERSTANDING DATA.pdf`](MBD%20MINI%20UNDERSTANDING%20DATA.pdf)** (24 slides): Tổng quan dữ liệu, ERD, độ phủ thời gian, nhãn `TARGETS`, và EDA từng nguồn `TRX`, `GEO`, `DIALOG`, đa nguồn.
- **[`MBD_MINI_DEEP_INSIGHTS_DECK.pdf`](MBD_MINI_DEEP_INSIGHTS_DECK.pdf)** (8 slides, A–H): Phân tích chuyên sâu về cơ chế chữ U (`0 TRX`), tần suất `TRX/tháng × Số tháng lịch sử`, kiểm định đa tầng `DIALOG` (bác bỏ Nghịch lý Simpson), phân rã Shift-Share, bán chéo đa nhãn (`A → B`), `event_type` mới và tương tác độc lập `TRX × GEO`.
- **[`HUONG_DAN_DOC_HIEU_VA_CACH_TINH_CHARTS.md`](HUONG_DAN_DOC_HIEU_VA_CACH_TINH_CHARTS.md)**: Giải thích chi tiết cách đọc hiểu và công thức tính toàn bộ 8 slide của Báo cáo 2.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
| --- | --- |
| [`notebooks/`](notebooks/README.md) | **7 notebook gọn gàng**: `01`–`04` (Báo cáo 1), `05` (Báo cáo 2), `06`–`07` (chuẩn bị và hợp nhất dữ liệu) |
| [`src/`](src/README.md) | Thư viện chuẩn bị dữ liệu (`propensity_data/`), package EDA (`mbd_mini_eda/`), và các script phục vụ trực tiếp 2 báo cáo |
| [`docs/`](docs/README.md) | Từ điển dữ liệu (`docs/dict/mbd_mini.md`), mô tả tập dữ liệu (`docs/DATASETS.md`) và bảng/biểu đồ báo cáo |
| [`sample_data/`](sample_data/README.md) | Dữ liệu Parquet mẫu của 1.000 khách hàng để xem nhanh cấu trúc |
| [`configs/`](configs/propensity_processing.json) | Cấu hình cho quy trình chuẩn bị dữ liệu và `mbd_mini_eda` |
| [`tests/`](tests/test_propensity_data.py) | Kiểm thử tự động cho `propensity_data`, `mbd_mini_eda` và đồng bộ notebook |
| [`pyproject.toml`](pyproject.toml) | Khai báo Python, dependency và CLI |

## Cài đặt & kiểm thử

Yêu cầu Python 3.10+ (Python 3.11+ cho DuckDB). Để chạy trên toàn bộ dữ liệu gốc, đặt bộ [ai-lab/MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini) tại `data/MBD-mini/`.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[eda,prepare,notebook]"
.venv/Scripts/python.exe -m unittest discover -s tests
.venv/Scripts/python.exe src/scripts/sync_preparation_notebook.py --check
```

## Quy ước nghiên cứu chính

Mọi phân tích chuyên sâu sử dụng toàn bộ `100.224` khách hàng và mốc cutoff `00:00` ngày mùng 1 tháng nhãn theo giờ Moscow (`Europe/Moscow`), đặc trưng chỉ lấy sự kiện nghiêm ngặt trước cutoff. Khi so sánh `H=1` và `H=3`, sử dụng cùng `Common Cohort` gồm `10` mốc dự báo (`02/2022–11/2022`, `1.002.240` quan sát khách–tháng).
