# New Propensity

Repository chứa package **7 bước thống kê mô tả** và tài liệu nghiên cứu propensity cho MBD-mini.
Phiên bản 0.2.0 bỏ pipeline đặc trưng khách hàng–tháng và khai phá mở rộng;
cấu hình mặc định dành cho máy RAM 16 GB, đọc dữ liệu theo batch.

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
.\.venv\Scripts\python.exe -X utf8 -m mbd_mini_eda run --config configs/mbd_mini_eda.local.json --plots
```

Kết quả mặc định ở `docs/mbd_mini_eda/report.md`. Hướng dẫn từng bước, cấu trúc
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
.\.venv\Scripts\python.exe -X utf8 -m mbd_mini_eda run --config configs/mbd_mini_eda.json --plots
```

Kiểm thử bằng dữ liệu tổng hợp:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```
