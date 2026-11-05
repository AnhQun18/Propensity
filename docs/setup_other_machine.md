# Setup từ GitHub và chạy trên máy đã có dữ liệu

Yêu cầu: Python **3.10 trở lên**, venv và bộ MBD-mini đã giải nén trên máy.
Lần cài đầu cần truy cập kho package Python để tải các dependency.
Các bước dưới dùng Windows PowerShell; cuối trang có lệnh tương ứng cho Linux/macOS.

## 1. Tải mã nguồn từ main

Nếu có Git:

```powershell
git clone --branch main https://github.com/AnhQun18/Propensity.git
cd Propensity
```

Nếu đã clone trước đó, mở terminal trong repo và chạy `git pull origin main`.
Cũng có thể mở [GitHub repo](https://github.com/AnhQun18/Propensity/tree/main),
chọn **Code → Download ZIP**, giải nén và mở terminal trong thư mục có
`pyproject.toml`. Cần tải toàn bộ repository để có cấu hình và hướng dẫn đi kèm.

## 2. Cài vào venv trên máy mới

Các lệnh dưới giả sử venv nằm tại `.venv` trong thư mục repo. Nếu venv đã tồn
tại thì bỏ qua dòng đầu; nếu dùng venv ở vị trí khác, thay đường dẫn tới Python
trong các lệnh bằng Python của venv đó.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install ".[plots]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -X utf8 -m mbd_mini_eda --help
```

Gọi trực tiếp Python trong venv nên không cần activate hoặc đổi execution policy.
`pip check` cần kết thúc với `No broken requirements found`; phần help có
hai lệnh `plan` và `run`. Tên distribution cài qua pip là `new-propensity`,
tên module gọi bằng Python là `mbd_mini_eda`.

## 3. Trỏ tới dữ liệu đã có

Giả sử dữ liệu ở `D:/Datasets/MBD-mini`. Bên trong thư mục này phải có:

```text
D:/Datasets/MBD-mini/
  client_split/       # Các file Parquet, có thể nằm trong fold=0 ... fold=4
  targets/
  detail/
    trx/
    geo/
    dialog/
```

Package dùng `detail`; thư mục `ptls` có thể tồn tại cùng dữ liệu nhưng không
được đọc trong pipeline này. Nếu dữ liệu còn là archive, cần giải nén để có
cấu trúc trên. Repo không chứa dữ liệu gốc.

Tạo bản cấu hình riêng cho máy:

```powershell
Copy-Item configs/mbd_mini_eda.json configs/mbd_mini_eda.local.json
```

Mở `configs/mbd_mini_eda.local.json`, đổi `data_root`. Có thể dùng cấu hình sau:

```json
{
  "data_root": "D:/Datasets/MBD-mini",
  "output_dir": "../docs/mbd_mini_eda",
  "batch_size": 250000,
  "dialog_batch_size": 4096,
  "sqlite_cache_mb": 256,
  "random_seed": 42,
  "sample_size": 5000,
  "duplicate_mode": "sample"
}
```

Thay `D:/Datasets/MBD-mini` bằng thư mục thực tế của máy đó. Trong JSON, dùng
dấu `/` như ví dụ hoặc viết `\\` nếu dùng dấu gạch chéo Windows.
Đường dẫn tương đối trong cấu hình được tính từ thư mục chứa JSON.
File `*.local.json` được Git bỏ qua để đường dẫn riêng của máy không được commit.

Phiên bản 0.2.0 chỉ thực hiện 7 bước thống kê mô tả. JSON cũ có `history_days`
hoặc `label_alignment` vẫn được đọc, kèm thông báo bỏ qua các lựa chọn ngoài phạm vi.
Giữ `data_root` của máy khi cập nhật config; có thể thêm `dialog_batch_size`
và `sqlite_cache_mb` như ví dụ. Nếu không thêm, package dùng giá trị mặc định.

## 4. Kiểm tra setup trước khi chạy toàn bộ

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 -m mbd_mini_eda plan --config configs/mbd_mini_eda.local.json
```

Kiểm thử cần kết thúc với `OK`. Kiểm thử dùng Parquet tổng hợp, không quét
dữ liệu thật. `plan` kiểm tra việc nạp cấu hình và liệt kê 7 bước; lệnh này
cũng không quét dữ liệu thật.

## 5. Chạy phân tích

Từ thư mục repo:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m mbd_mini_eda run --config configs/mbd_mini_eda.local.json --plots
```

Hoặc chọn nơi xuất kết quả khác:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m mbd_mini_eda run --config configs/mbd_mini_eda.local.json --output-dir "D:/Results/mbd_mini_eda" --plots
```

Chạy thành công sẽ kết thúc với thông báo `EDA report: .../report.md` và
exit code 0. Kết quả gồm:

```text
docs/mbd_mini_eda/           # Hoặc thư mục chọn bằng --output-dir
  report.md
  metadata.json
  manifest.json
  tables/<section>/*.csv
  figures/*.png
```

`descriptive.quality` thực hiện quét các nguồn và tạo kho tổng hợp tạm nên có
thể chạy lâu. `sample_size=5000` chỉ giới hạn mẫu cho một số thống kê; cấu hình
trên vẫn quét toàn bộ nguồn. Xem [phạm vi từng phép tính](mbd_mini_eda_package.md).
Terminal báo số dòng đã quét theo nguồn; thời gian nguồn và từng bước được
ghi trong `metadata.json`. Batch hội thoại được giới hạn riêng để kiểm soát RAM.

Sau khi cập nhật mã từ GitHub, chạy lại `python -m pip install ".[plots]"`
bằng Python của venv để bản cài đặt nhận mã mới.

## Linux/macOS

Tải repo và sửa bản JSON riêng tương tự Windows. Các lệnh dùng Python của venv:

```bash
git clone --branch main https://github.com/AnhQun18/Propensity.git
cd Propensity
python3 -m venv .venv
.venv/bin/python -m pip install ".[plots]"
cp configs/mbd_mini_eda.json configs/mbd_mini_eda.local.json
```

Đổi `data_root` trong JSON thành đường dẫn trên máy, ví dụ `/mnt/data/MBD-mini`,
rồi chạy:

```bash
.venv/bin/python -X utf8 -m unittest discover -s tests -v
.venv/bin/python -X utf8 -m mbd_mini_eda run --config configs/mbd_mini_eda.local.json --plots
```

## Xử lý lỗi setup

| Thông báo | Cách kiểm tra |
|---|---|
| `No module named mbd_mini_eda` | Chạy lại `-m pip install ".[plots]"` bằng đúng Python đang dùng để chạy package. |
| Không tìm thấy `.venv/Scripts/python.exe` | Kiểm tra vị trí venv; trên Linux/macOS dùng `.venv/bin/python`. |
| Không tìm thấy `client_split`, `targets` hoặc `detail` | Sửa `data_root` tới đúng thư mục gốc MBD-mini đã giải nén. |
| Không tìm thấy file cấu hình | Chạy từ thư mục repo hoặc truyền đường dẫn tuyệt đối tới JSON. |
| `No module named matplotlib` | Cài với extra `plots` như lệnh ở bước 2, hoặc chạy không có `--plots`. |
| Python không đáp ứng phiên bản | Tạo venv bằng Python 3.10 trở lên rồi cài lại package. |
