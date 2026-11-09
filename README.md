# New Propensity

Repository chứa package **7 bước thống kê mô tả** và tài liệu nghiên cứu propensity cho MBD-mini.

Bản rà soát mới hoàn thành **12 yêu cầu EDA**: giữ toàn bộ lịch sử, null theo nguồn,
duplicate toàn bộ, nhóm độ phủ có chủ đích, biểu đồ tuyến tính, nhãn zoom Y và outlier.
Đọc [báo cáo hiện tại](docs/mbd_mini_eda/report.md),
[danh mục bảng/biến](docs/mbd_mini_eda/data_catalog.md),
[mẫu phân tầng 1.000 khách](sampledata/README.md).

Báo cáo xuất bản nằm trong [deliverables/mbd-mini-full-report](deliverables/mbd-mini-full-report/README.md).
Thiết kế mới cho nhãn mua bất kỳ sản phẩm nào và kết quả audit nằm tại
[any-product](docs/propensity_feasibility/any_product/design.md).
Gói ZIP mẫu/báo cáo trùng lặp đã được dọn; tải repository bằng **Code → Download ZIP**
để nhận dữ liệu mẫu, notebook và báo cáo cùng nhau.

```powershell
.\.venv\Scripts\python.exe -u -X utf8 src/review_mbd_eda.py
```

Lệnh này dùng raw ở `data/MBD-mini`, tạo lại báo cáo và sampledata; giữ kết quả
quét trong `outputs/mbd_review_cache/review.duckdb` và ma trận từng nguồn trong
`source_month_counts.npz`. Chạy lại tái dùng các phép kiểm tra đã hoàn tất.
Cache cũ `outputs/behavior_coverage/matrix.npz` vẫn được giữ và đối chiếu từng ô.
Chi tiết cài đặt, cơ chế tái dùng cache và dọn file: [hướng dẫn rà soát](docs/mbd_eda_review.md).
Phiên bản 0.2.0 bỏ pipeline đặc trưng khách hàng–tháng và khai phá mở rộng;
cấu hình mặc định dành cho máy RAM 16 GB, đọc dữ liệu theo batch.
Phiên bản 0.2.1 sửa mã nguyên có null và trục thời gian. Bộ script bổ sung
cho phép hoàn tất phân vị, tần suất mã và kiểm tra trùng trên toàn bộ dữ liệu.

**Chạy đủ quy trình trên máy đã có data/venv:** xem
[hướng dẫn chi tiết](docs/run_complete_other_machine.md) (Python >=3.11).
Kết quả đã chạy nằm tại [báo cáo 7 bước](docs/mbd_mini_eda/report.md).
Xem [kiểm kê cuối và giới hạn diễn giải](docs/mbd_mini_eda_audit.md).

## Setup trên máy khác

Yêu cầu Python 3.10 trở lên và dữ liệu MBD-mini đã giải nén. Tải toàn bộ repo
bằng **Code → Download ZIP** rồi giải nén, hoặc clone nhánh `main`:

```powershell
git clone --branch main https://github.com/AnhQun18/Propensity.git
cd Propensity
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install ".[plots]"
Copy-Item configs/mbd_mini_eda.json configs/mbd_mini_eda.local.json
```

Nếu đã có venv trong thư mục dự án, bỏ qua lệnh tạo venv. Trong file
`configs/mbd_mini_eda.local.json`, đổi `data_root` thành đường dẫn dữ liệu
trên máy đó, ví dụ `"D:/Datasets/MBD-mini"`, rồi chạy:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 -m mbd_mini_eda run --config configs/mbd_mini_eda.local.json --output-dir build/mbd_mini_eda_base --plots
```

Lệnh trên lưu kết quả cơ bản ở `build/mbd_mini_eda_base/report.md`. Để tính bổ sung
và giữ snapshot kết quả GitHub, dùng [quy trình đầy đủ](docs/run_complete_other_machine.md).
Hướng dẫn package cơ bản, cấu trúc
thư mục dữ liệu, Linux/macOS và xử lý lỗi thường gặp nằm tại
[Setup và chạy trên máy khác](docs/setup_other_machine.md).

## Cấu trúc thư mục

```text
.
├── docs/             # Tài liệu dự án
├── notebooks/        # Notebook EDA
├── src/              # Package mbd_mini_eda và script kiểm chứng MBD-mini
├── tests/            # Kiểm thử bằng dữ liệu tổng hợp
├── configs/          # Cấu hình package MBD-mini
├── data/             # Dữ liệu gốc cục bộ, không đưa lên Git
├── pyproject.toml    # Khai báo thông tin dự án và dependency
└── README.md         # Mô tả repository
```

## Dependency

Các dependency được khai báo trong `pyproject.toml`. Cài package và phần vẽ biểu đồ:

```powershell
.\.venv\Scripts\python.exe -m pip install ".[plots]"
```

Bộ [MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini) đã tải và giải nén
trước đó nằm riêng tại `data/MBD-mini/`.

Package thống kê mô tả MBD-mini nằm tại
[`src/mbd_mini_eda/`](src/mbd_mini_eda/). Xem
[cấu trúc, giao diện và thứ tự triển khai](docs/mbd_mini_eda_package.md).
CLI hỗ trợ xem kế hoạch và chạy phân tích, xuất bảng CSV, biểu đồ PNG và báo cáo:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m mbd_mini_eda run --config configs/mbd_mini_eda.json --output-dir build/mbd_mini_eda_base --plots
```

Kiểm thử bằng dữ liệu tổng hợp:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

Ma trận độ phủ hành vi khách hàng × tháng (gộp `trx`, `geo`, `dialog`), có cache,
ảnh raster và HTML xem chi tiết theo trang:
[cách tính và tối ưu](docs/behavior_coverage_matrix.md).

```powershell
.\.venv\Scripts\python.exe -X utf8 src/plot_behavior_coverage.py --active-days
```
