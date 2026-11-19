# MBD-mini: khám phá dữ liệu và chuẩn bị dữ liệu propensity

Repo chứa notebook, mã nguồn và tài liệu để tìm hiểu bộ dữ liệu ngân hàng MBD-mini, kiểm tra chất lượng dữ liệu và chuẩn bị bảng khách hàng–tháng cho nghiên cứu khả năng mua sản phẩm. Repo hiện chưa có quy trình huấn luyện mô hình propensity hoàn chỉnh.

**Bắt đầu đọc:** [danh sách notebook](notebooks/README.md) · [từ điển dữ liệu MBD-mini](docs/dict/mbd_mini.md) · [nguồn và cấu trúc dữ liệu](docs/DATASETS.md)

## Trong repo có gì?

| Đường dẫn | Nội dung |
| --- | --- |
| [`notebooks/`](notebooks/README.md) | 17 notebook được đánh số: xem dữ liệu, kiểm kê, chuẩn bị dữ liệu và EDA theo nhãn, TRX, GEO, DIALOG, đa nguồn. Nhiều notebook lưu bảng và biểu đồ ngay trong cell. |
| [`src/`](src/README.md) | Thư viện xử lý dữ liệu và EDA, các script phân tích, kiểm tra và tạo báo cáo. |
| [`docs/`](docs/README.md) | Từ điển dữ liệu, phương pháp, đánh giá phạm vi và các bảng, biểu đồ báo cáo đã xuất. |
| [`sample_data/`](sample_data/README.md) | Dữ liệu Parquet minh họa của 1.000 khách hàng để xem cấu trúc và thử quy trình. |
| [`configs/`](configs/propensity_processing.json) | Cấu hình cho chuẩn bị dữ liệu và phân tích. |
| [`tests/`](tests/test_propensity_data.py) | Kiểm tra tự động cho các phần mã nguồn. |
| [`pyproject.toml`](pyproject.toml) | Phiên bản Python, dependency, nhóm cài đặt và lệnh CLI. |

## Dữ liệu và cách bắt đầu

Nguồn dữ liệu gốc là [ai-lab/MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini). Repo chỉ đưa lên Git bộ dữ liệu ví dụ trong `sample_data/`; để chạy phân tích toàn bộ, tải dữ liệu gốc và đặt tại `data/MBD-mini/` trên máy. Bảng `targets` có khóa `client_id`–`mon` và bốn nhãn `target_1` đến `target_4`. Các nguồn sự kiện chính là TRX, GEO và DIALOG. Xem [từ điển dữ liệu](docs/dict/mbd_mini.md) trước khi diễn giải nhãn hoặc nối các bảng.

Yêu cầu Python 3.10 trở lên. Quy trình chuẩn bị dữ liệu cần Python 3.11 trở lên để cài DuckDB theo cấu hình hiện tại.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[eda,prepare,notebook]"
.venv/Scripts/python.exe -m unittest discover -s tests
```

1. Mở [notebook 01](notebooks/01_data_preview.ipynb) để xem dữ liệu ví dụ và [notebook 02](notebooks/02_data_inventory.ipynb) để kiểm kê nguồn dữ liệu.
2. Chọn notebook EDA theo chủ đề trong [mục lục notebook](notebooks/README.md). Các notebook 03–04 dành cho chuẩn bị và ghép dữ liệu khách hàng–tháng, không huấn luyện mô hình.
3. Đọc [đánh giá phạm vi EDA](docs/eda_two_lists_audit_20261118.md) trước khi sử dụng kết quả đã lưu trong notebook hoặc `docs/`.

## Quy ước nghiên cứu

EDA chính cần kiểm tra toàn bộ khách hàng và ít nhất một năm nhãn; dữ liệu ví dụ không dùng để suy rộng kết quả. Cutoff nghiên cứu là 00:00 ngày đầu tháng nhãn theo giờ Moscow (`Europe/Moscow`), và đặc trưng chỉ dùng sự kiện trước cutoff từ toàn bộ lịch sử có sẵn. Khi so sánh horizon 1, 2 và 3 tháng, phải dùng cùng cohort có đủ nhãn tương lai.

Một số notebook, cấu hình và báo cáo được tạo theo phương pháp cũ (UTC, cửa sổ lịch sử cố định hoặc mẫu khách). Chúng được giữ để đối chiếu; [đánh giá phạm vi](docs/eda_two_lists_audit_20261118.md) chỉ rõ giới hạn trước khi dùng cho nghiên cứu hiện hành.
