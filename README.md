# New Propensity

Repository chứa package thống kê mô tả, khai phá dữ liệu và tài liệu nghiên cứu propensity cho MBD-mini.

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
.\.venv\Scripts\python.exe -m pip install -e ".[plots]"
```

Bộ [MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini) đã tải và giải nén
trước đó nằm riêng tại `data/MBD-mini/`.

Package thống kê mô tả và khai phá dữ liệu MBD-mini nằm tại
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
