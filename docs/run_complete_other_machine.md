# Chạy đủ 7 bước trên máy đã có dữ liệu

Bản rà soát 12 yêu cầu hiện tại dùng [quy trình có cache](mbd_eda_review.md):
`python -u -X utf8 src/review_mbd_eda.py`. Quy trình dưới đây vẫn dành cho
package 7 bước và bổ sung export; không cần chạy lại nếu chỉ tái lập bản rà soát.

Quy trình gồm package cơ bản và script bổ sung: sửa biểu đồ, tần suất mọi mã,
phân vị amount toàn bộ, tháng đầu/cuối theo khách và trùng nguyên dòng toàn bộ.
Raw chỉ được đọc; không tạo feature khách hàng–tháng, không huấn luyện mô hình.

## 1. Chuẩn bị và tải GitHub

- Python **3.11 trở lên** để dùng DuckDB 2.x và chạy toàn bộ quy trình.
  Package cơ bản vẫn hỗ trợ Python 3.10.
- MBD-mini đã giải nén đủ shard, giữ nguyên partition `fold=...`.
- RAM 16 GB: bắt đầu với DuckDB `--memory-limit 4GB --threads 2`.
  Giới hạn này không bao gồm toàn bộ RAM của Python và hệ điều hành.
- SSD còn chỗ cho dữ liệu tạm. Lệnh mẫu giới hạn tạm 40 GB; có thể cần tăng
  tùy dữ liệu. Kiểm tra dung lượng thật trước khi tăng.

Clone main rồi mở PowerShell tại thư mục có `pyproject.toml`:

```powershell
$env:PYTHONUTF8 = '1'
git clone --branch main https://github.com/AnhQun18/Propensity.git
cd Propensity
```

Hoặc chọn **Code → Download ZIP**, giải nén và mở terminal trong `Propensity-main`.
Không chạy ngay trong ZIP hoặc từ thư mục cha. Với bản clone đã có, lưu thay đổi
riêng rồi cập nhật `git pull --ff-only origin main`. Với bản ZIP, giải nén mã mới
vào thư mục mới, dùng lại đường dẫn data; tránh trộn phiên bản cũ/mới.

Chỉ cần xem kết quả đã chạy: mở `docs/mbd_mini_eda/report.md` và các CSV/PNG
cạnh nó, không cần data hoặc cài Python.

## 2. Dùng venv đã có

Nếu có `.venv` trong dự án, **không tạo lại venv**. Không chạy
`python -m venv .venv` khi chính venv đó đang active: Windows có thể báo lỗi
copy `venvlauncher.exe` vào Python đang sử dụng.

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install ".[plots,supplement]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import mbd_mini_eda, duckdb, matplotlib; print(mbd_mini_eda.__file__); print('DuckDB', duckdb.__version__)"
```

Nếu mở PowerShell mới, chạy lại `$env:PYTHONUTF8 = '1'` để tránh lỗi in đường
dẫn tiếng Việt của pip trên một số cấu hình Windows.

`pip check` cần báo `No broken requirements found`, các import phải thành công.
Venv ở nơi khác: thay đường dẫn Python trong **tất cả** lệnh. Không bắt buộc
activate nếu gọi đường dẫn Python trực tiếp. Chỉ khi chưa có venv, dùng Python
>=3.11 chạy `python -m venv .venv` một lần rồi cài như trên.

Sau mỗi lần cập nhật repo, cài lại `".[plots,supplement]"` để venv nhận mã mới.
Giữ hai script trong `deliverables/mbd_mini_eda_fixes/` cạnh nhau vì chúng import
lẫn nhau. Tải cả repo thì không cần giải nén ZIP hai script lần nữa.

## 3. Đặt đúng đường dẫn data

`data_root` phải trực tiếp chứa `client_split`, `targets` và `detail`:

```text
D:/Datasets/MBD-mini/
  client_split/fold=.../*.parquet
  targets/fold=.../*.parquet
  detail/
    trx/fold=.../*.parquet
    geo/fold=.../*.parquet
    dialog/fold=.../*.parquet
```

Có thể có thêm tầng shard. Không trỏ tới ZIP, thư mục cha MBD-mini hoặc riêng
`detail`. Nguồn `ptls` không thuộc khung 7 bước này.

Nếu chưa có cấu hình riêng, chạy:

```powershell
Copy-Item configs/mbd_mini_eda.json configs/mbd_mini_eda.local.json
```

Sửa `configs/mbd_mini_eda.local.json`, thay `data_root` bằng đường dẫn thật:

```json
{
  "data_root": "D:/Datasets/MBD-mini",
  "output_dir": "../build/mbd_mini_eda_base",
  "work_dir": "../build/mbd_eda_work",
  "batch_size": 250000,
  "dialog_batch_size": 4096,
  "sqlite_cache_mb": 256,
  "random_seed": 42,
  "sample_size": 5000,
  "duplicate_mode": "sample"
}
```

Dùng `/` trong JSON hoặc `\\` cho dấu gạch chéo Windows. Đường dẫn tương đối
trong JSON tính từ thư mục chứa JSON; đường dẫn CLI tính từ thư mục hiện tại.
File `*.local.json` được Git bỏ qua. Giữ duplicate_mode=sample ở bước cơ bản;
bước bổ sung sẽ kiểm tra trùng toàn bộ, tránh làm phép tính nặng hai lần.

Kiểm tra đường dẫn tuyệt đối trong ví dụ:

```powershell
$cfg = Get-Content configs/mbd_mini_eda.local.json -Raw | ConvertFrom-Json
Test-Path (Join-Path $cfg.data_root 'client_split')
Test-Path (Join-Path $cfg.data_root 'targets')
Test-Path (Join-Path $cfg.data_root 'detail/trx')
Test-Path (Join-Path $cfg.data_root 'detail/geo')
Test-Path (Join-Path $cfg.data_root 'detail/dialog')
```

Năm kết quả phải là `True`. Data không bắt buộc nằm trong repo.

## 4. Kiểm thử và xem kế hoạch

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 -m mbd_mini_eda plan --config configs/mbd_mini_eda.local.json
```

Test dùng Parquet tổng hợp nhỏ, cần kết thúc `OK`. Nếu test bổ sung bị skip
vì thiếu DuckDB, kiểm tra Python và cài extra supplement. `plan` không quét
toàn bộ raw hoặc xác minh tất cả shard.

## 5. Chạy package cơ bản

Từ gốc repo, giữ cùng cửa sổ PowerShell cho bước sau. Tên có thời gian để giữ
nguyên snapshot kết quả đã tải từ GitHub:

```powershell
$runTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$baseExport = Join-Path 'build' "mbd_mini_eda_base_$runTag"
$finalExport = Join-Path 'docs' "mbd_mini_eda_local_$runTag"
.\.venv\Scripts\python.exe -X utf8 -m mbd_mini_eda run --config configs/mbd_mini_eda.local.json --output-dir $baseExport --plots
if ($LASTEXITCODE -ne 0) { throw 'Bước cơ bản thất bại; xem lỗi phía trên.' }
```

Thành công: `EDA report: .../report.md`, exit code 0, đủ 7 section `completed`
trong metadata. Lệnh vẫn quét toàn bộ sự kiện; sample_size chỉ giới hạn một số
phân vị/hình và kiểm tra trùng ở bước cơ bản. `descriptive.quality` quét dữ liệu,
xây kho tổng hợp nên thường lâu nhất. Chưa kết luận có số trùng/phân vị toàn bộ
chỉ từ bước này.

## 6. Chạy bổ sung và export cuối

Sau khi bước 5 thành công, trong cùng cửa sổ PowerShell:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 deliverables/mbd_mini_eda_fixes/complete_mbd_statistics.py --export $baseExport --config configs/mbd_mini_eda.local.json --output-dir $finalExport --work-dir build/mbd_supp_work --memory-limit 4GB --threads 2 --temp-limit 40GB
if ($LASTEXITCODE -ne 0) { throw 'Bổ sung thất bại; chưa có kết quả cuối hoàn chỉnh.' }
```

Script đối chiếu schema, số dòng, khóa và tổng nhãn, rồi đọc raw để tính:

- Tần suất đầy đủ mọi mã; sửa lỗi tách `11`/`11.0` theo schema nguyên.
- Phân vị amount chính xác trên tất cả giá trị hữu hạn; histogram toàn miền và zoom tuyến tính đến P99,
  boxplot và thống kê theo currency.
- Tháng có sự kiện đầu/cuối theo khách và độ phủ; số nguồn có log, độ phủ nhãn.
- Trùng nguyên dòng toàn bộ xuyên shard/batch; kiểm tra khóa khách/khách–tháng riêng.
- Trục tháng theo lịch, timeline có mốc nhãn, report, metadata và manifest mới.

Output phải mới/rỗng, tách khỏi raw và export đầu vào. Export cơ bản được giữ nguyên.
Mở terminal mới thì gán lại `$baseExport`, `$finalExport` bằng đường dẫn thực tế,
không tạo tag mới rồi trỏ nhầm export. Nếu đã có export cơ bản **của cùng raw**,
có thể bỏ bước 5 và gán `$baseExport` tới export đó. `--export` cũng nhận ZIP.
Không dùng export GitHub cho một bộ raw khác chỉ vì tên dataset giống nhau.

**Không thêm `--skip-duplicates` nếu cần hoàn tất khung.** Tùy chọn đó trả trạng
thái `completed_except_duplicates`, vẫn thiếu kiểm tra trùng toàn bộ.

Chỉ sửa hình/bảng sẵn có mà không đọc raw là chế độ riêng:

```powershell
.\.venv\Scripts\python.exe -X utf8 deliverables/mbd_mini_eda_fixes/repair_mbd_export.py --export 'C:/Downloads/docs.zip' --output-dir build/mbd_export_repaired
```

Sửa nhanh không khôi phục được đuôi tần suất đã cắt top 20 hoặc phân vị toàn bộ
từ mẫu; không thay thế bước bổ sung raw.

## 7. Xác nhận thành công và tìm output

```powershell
$meta = Get-Content (Join-Path $finalExport 'metadata.json') -Raw | ConvertFrom-Json
$meta.raw_supplements
$manifest = Get-Content (Join-Path $finalExport 'manifest.json') -Raw | ConvertFrom-Json
$manifest.tables.Count
$manifest.figures.Count
Get-Item (Join-Path $finalExport 'report.md')
```

Cần status=`completed`, exact_duplicates=`True`, full_amount_quantiles=`True`,
full_categorical_frequencies=`True`, first_last_event_month=`True` trong
raw_supplements. `running`/`failed` hoặc exit code khác 0 nghĩa là chưa hoàn tất.

```text
docs/mbd_mini_eda_local_<ngày_giờ>/
  report.md
  metadata.json
  manifest.json
  tables/descriptive.<section>/*.csv
  figures/*.png
```

Đây là kết quả mới trên máy bạn. `docs/mbd_mini_eda/` là snapshot trên GitHub.
Các bảng đầy đủ nằm cạnh bảng top 20: `transaction_categories_full.csv`,
`location_categories_full.csv`.

Histogram dùng giá trị amount gốc; bản zoom chỉ hiển thị từ min tới P99,
các giá trị ngoài zoom vẫn nằm trong histogram toàn miền. Chưa có từ điển mã thì
không suy diễn đơn vị/quy đổi currency. Khoảng cách sự kiện ở customer_activity
là phân phối **khoảng cách trung bình mỗi khách**, không phải mọi khoảng cách
liên tiếp của từng sự kiện. Export không có client_id từng khách; build có thể
chứa tổng hợp theo khách lúc chạy và không được đưa lên Git.

## 8. Thời gian và xử lý lỗi

Export cơ bản ban đầu ghi nhận 35 phút 45 giây trên máy tạo export; đây không
phải cam kết cho máy khác. Bổ sung đọc raw nhiều lượt và kiểm tra trùng toàn bộ,
có thể tốn thêm đáng kể thời gian/SSD. metadata.duration_seconds là lần cơ bản;
raw_supplements.duration_seconds là lần bổ sung, chưa gồm mọi thao tác xuất hình.

Lần bổ sung cho snapshot trong repo chạy 474,713 giây (khoảng 7 phút 55 giây)
trên máy i3-8145U, RAM khoảng 8 GB, DuckDB 3GB/2 threads. Đây chỉ là số đo tham
khảo cho phần bổ sung, không bao gồm 35 phút 45 giây của export cơ bản ban đầu.

| Lỗi | Xử lý |
|---|---|
| Không copy được venvlauncher.exe | Không tạo lại venv đang active; kiểm tra Python hiện có và cài dependency. Nếu Python hỏng, deactivate rồi tạo venv mới bằng Python nền. |
| No module named mbd_mini_eda | Cài lại extra plots,supplement bằng đúng Python dùng chạy. |
| No module named duckdb | Kiểm tra Python >=3.11 và cài extra supplement. |
| Không thấy client_split | Sửa data_root tới thư mục trực tiếp chứa nguồn. |
| Output không rỗng | Chọn tên mới, giữ kết quả cũ. |
| Schema/số dòng/tổng nhãn lệch | Kiểm tra đúng dataset, đủ shard; tạo lại export cơ bản từ raw đó. |
| Out of Memory | Giảm threads xuống 1–2; máy 8 GB dùng memory-limit 2GB hoặc 3GB, đóng ứng dụng nặng. Giới hạn không bao gồm toàn bộ Python. |
| Hết đĩa tạm | Chọn work-dir ở SSD còn chỗ, tăng temp-limit theo dung lượng thật. Chạy lại với output mới. |
| Ngắt giữa chừng | Giữ export cơ bản, chọn output mới rồi chạy lại bổ sung; chưa hỗ trợ resume giữa các phép tính. |

## Linux/macOS

Tải mã và sửa JSON tương tự, dùng Python của venv trên hệ điều hành đó:

```bash
.venv/bin/python -m pip install '.[plots,supplement]'
.venv/bin/python -m pip check
.venv/bin/python -X utf8 -m unittest discover -s tests -v
.venv/bin/python -X utf8 -m mbd_mini_eda run --config configs/mbd_mini_eda.local.json --output-dir build/mbd_base_run1 --plots
.venv/bin/python -u -X utf8 deliverables/mbd_mini_eda_fixes/complete_mbd_statistics.py --export build/mbd_base_run1 --config configs/mbd_mini_eda.local.json --output-dir docs/mbd_mini_eda_local_run1 --work-dir build/mbd_supp_work --memory-limit 4GB --threads 2 --temp-limit 40GB
```

Chỉ chạy bổ sung sau khi cơ bản thành công. Đặt data_root theo máy, ví dụ
`/mnt/data/MBD-mini`. Không sao chép venv Windows sang Linux/macOS.
