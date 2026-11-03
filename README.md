# New Propensity

Repository lưu trữ tài liệu, notebook phân tích dữ liệu khám phá (EDA), mã nguồn core và dữ liệu ví dụ cho dự án propensity.

## Package thống kê mô tả và EDA

Mã nguồn tái sử dụng nằm trong `src/eda/`, gồm `descriptive`, `analysis`,
`visualization`, đọc CSV/Parquet, kiểm tra quan hệ bảng và xuất báo cáo vào
`docs/<dataset>/`. Xem [hướng dẫn API, cấu trúc và giới hạn tính toán](docs/eda_package.md).

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[plots,notebook]"
.\.venv\Scripts\propensity-eda.exe --config configs/eda_ecommerce.json --plots
```

Cấu hình mẫu cho cả ba bộ nằm trong `configs/eda_*.json`. Các lệnh này quét
nguồn dữ liệu lớn; xem chế độ exact/approximate trong hướng dẫn trước khi chạy.

## Cấu trúc thư mục

```text
.
├── docs/             # Tài liệu dự án
├── notebooks/        # Notebook EDA
├── src/              # Mã nguồn core
├── sample_data/      # Dữ liệu ví dụ
├── data/             # Script tải, dữ liệu MBD-mini và eCommerce
├── pyproject.toml    # Khai báo thông tin dự án và dependency
└── README.md         # Mô tả repository
```

Các thư mục có file `.gitkeep` để Git lưu lại cấu trúc khi chưa có nội dung.

## Dependency

Khai báo các thư viện cần dùng trong danh sách `dependencies` của `pyproject.toml`.

## Tải dữ liệu eCommerce từ Kaggle

Chạy trong PowerShell tại thư mục gốc của repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -X utf8 -m pip install "kagglehub>=1.0,<2.0"
.\.venv\Scripts\python.exe -X utf8 data/code.py
```

Script `data/code.py` tải bộ dữ liệu
[eCommerce behavior data from multi category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
vào `data/ecommerce/`. KaggleHub tự giải nén thành `2019-Oct.csv` và
`2019-Nov.csv`, tổng dung lượng khoảng 14,68 GB.

Chạy script bằng Python trong `.venv` để sử dụng đúng dependency đã cài.
Môi trường `.venv` và dữ liệu tải về được bỏ qua trong Git.

Bộ [MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini) đã tải và giải nén
trước đó nằm riêng tại `data/MBD-mini/`.

## Xem cấu trúc dữ liệu

Mở `notebooks/01_data_preview.ipynb` và chọn kernel Python của `.venv`.
Notebook hiển thị 5 dòng đầu, tên cột và kiểu dữ liệu của cả 8 bảng trong
`client_split`, `targets`, `detail` và `ptls`. Các file Parquet được đọc theo
partition `fold`, chỉ lấy mẫu nhỏ để xem cấu trúc.

Cài các thư viện cho notebook nếu cần:

```powershell
.\.venv\Scripts\python.exe -m pip install pandas pyarrow ipykernel nbformat nbclient
```

## Tải 3 file Fliggy Transfer-Plans qua Google Colab

Khi đường tải trực tiếp từ Tianchi chậm, mở Google Colab và upload
`notebooks/02_download_fliggy_colab.ipynb`. Notebook dành cho
[dataset 140721](https://tianchi.aliyun.com/dataset/140721), nhận 3 link tải qua ô nhập ẩn:

- `user_behavior_transfer_plans.csv`
- `user_profile.csv`
- `transfer_plan_profile.csv`

Notebook đọc metadata riêng cho từng file, hỗ trợ tải tiếp, kiểm tra kích thước/ETag
và CRC64 khi máy chủ cung cấp. Có thể lưu vào Google Drive để giữ phần tải dở,
hoặc chọn từng file đã hoàn tất để tải từ Colab về máy.
Lưu các file nhận được vào `data/fliggy/140721/`.

Đây là phương án thử một đường mạng khác; tốc độ từ Colab chưa được đo.
Notebook không chứa sẵn link có chữ ký và không cần GPU.
