# Package thống kê mô tả MBD-mini

Từ phiên bản 0.2.0, `mbd_mini_eda` thực hiện đúng **7 bước** theo khung đã chốt.
Các bước dùng chung một lần quét Parquet và các bảng tổng hợp trên toàn lịch sử.
Pipeline tạo đặc trưng khách hàng–tháng, cửa sổ 30/84/180 ngày và các module
`exploration` đã được loại bỏ.

## Cài và chạy

Với máy đã có dữ liệu, xem [setup từ GitHub](setup_other_machine.md).
Chạy từ thư mục chứa `pyproject.toml`:

```powershell
.\.venv\Scripts\python.exe -m pip install ".[plots]"
.\.venv\Scripts\python.exe -X utf8 -m mbd_mini_eda plan --config configs/mbd_mini_eda.local.json
.\.venv\Scripts\python.exe -X utf8 -m mbd_mini_eda run --config configs/mbd_mini_eda.local.json --plots
```

`plan --format json` chỉ đọc cấu hình và in kế hoạch. `run --output-dir PATH`
chọn nơi xuất; `--quiet` tắt tiến độ. Đường dẫn trong JSON tính từ thư mục
chứa JSON; `--output-dir` tính từ thư mục terminal hiện tại.

## Bảy bước

| Bước | Module | Đầu ra |
|---|---|---|
| 1. Cấu trúc và quy mô | `descriptive.structure` | Bảng/cột/schema, số dòng, cấp dữ liệu, khóa nối; sơ đồ quan hệ trong report |
| 2. Chất lượng | `descriptive.quality` | Null theo cột, giá trị sai, trùng dòng/khóa, độ phủ ID |
| 3. Thời gian | `descriptive.temporal` | Biên thời gian, hoạt động theo tháng, span và ngày hoạt động; timeline |
| 4. Giao dịch | `descriptive.transactions` | Moment amount, phân vị, histogram, boxplot, outlier, tần suất mã |
| 5. Vị trí và hội thoại | `descriptive.modalities` | Sự kiện/ngày/vị trí khác nhau theo khách, số hội thoại, shape/norm/giá trị embedding |
| 6. Khách hàng | `descriptive.customers` | Phân phối hoạt động, span, khoảng cách trung bình giữa sự kiện, tổ hợp nguồn và kiểm tra cardinality |
| 7. Nhãn | `descriptive.targets` | Dương/âm, tỷ lệ theo tháng/fold, khách từng dương, đồng xuất hiện bốn nhãn |

`aggregation.py` tạo một bảng khách hàng nội bộ phục vụ thống kê bước 6.
Đây là tổng hợp toàn lịch sử; package không tạo bảng feature theo cutoff/tháng.
Giữ mọi khách trong `client_split`, kể cả khách không có log. Count bằng 0 khi
không có log; mean và timestamp không xác định giữ NaN/NaT. ID ngoài cohort
được báo trong chất lượng và không thêm vào bảng khách hàng.

## Cấu hình cho máy RAM 16 GB

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `batch_size` | 250000 | Số dòng mỗi batch giao dịch/vị trí |
| `dialog_batch_size` | 4096 | Giới hạn batch hội thoại; thực tế lấy min với batch_size |
| `sqlite_cache_mb` | 256 | Cache SQLite tính bằng MiB; không phải giới hạn RAM toàn chương trình |
| `sample_size` | 5000 | Mẫu dòng ngẫu nhiên tối đa mỗi nguồn |
| `random_seed` | 42 | Tái lập mẫu với cùng thứ tự file/dòng |
| `duplicate_mode` | sample | exact / sample / off |
| `max_categories` | 20 | Top mã; phần còn lại và null ghi riêng |
| `embedding_size` | 768 | Chiều embedding cần kiểm tra |
| `timezone` | null | Giữ naive như file; chỉ chuyển/localize khi khai báo |
| `work_dir` | null | Nơi chứa SQLite tạm; null dùng temp hệ thống |
| `max_batches_per_source` | null | Chỉ dùng để chạy thử prefix; null quét đủ |

Cấu hình dùng batch có giới hạn và kho tổng hợp trên đĩa. Batch hội thoại nhỏ
riêng vì mỗi dòng chứa vector embedding. Các kết quả tổng hợp theo khách,
ngày hoạt động và số vị trí được tính một lần rồi dùng lại; vị trí chỉ lưu
theo `(client_id, value)`, không nhân bản theo ngày. Chỉ các bảng khách hàng
và nhãn, batch hiện tại, mẫu nhỏ và bảng tổng hợp theo khách được giữ trong RAM.
RAM và dung lượng đĩa thực tế còn phụ thuộc dữ liệu; các tham số trên không
đặt một giới hạn cứng cho tổng bộ nhớ tiến trình.

Nếu có SSD, có thể đặt `work_dir` vào thư mục trên SSD, ngoài `data_root`.
Kho tạm được dọn khi chạy xong hoặc gặp lỗi. Chạy lại sẽ tính lại từ dữ liệu nguồn.

JSON cũ vẫn dùng được: `load_config` thông báo rồi bỏ qua `history_days`,
`label_alignment`, `label_windows_path`, `analysis_months`, `min_group_size`,
`min_positive_count` và `correlation_sample_size`. Nhãn được mô tả nguyên trạng
theo `mon`, không cần mapping cutoff. API Python nhận các tham số mới ở bảng trên;
`run_analysis(config, progress=...)` không còn tham số `label_windows`.

## Phạm vi và độ chính xác

Số dòng, null, giá trị sai, tần suất mã, hoạt động, số vị trí và tổng hợp khách
hàng dùng toàn bộ phần dữ liệu đã quét. Mean/std/min/max amount dùng toàn bộ
giá trị hữu hạn, gộp mean/M2, std mẫu với ddof=1. Phân vị, histogram, boxplot
và outlier amount dùng mẫu ngẫu nhiên đồng đều, ghi rõ `quantile_scope`,
`moments_scope` và số quan sát. `sample_size` không giới hạn lần quét raw.

`duplicate_mode=sample` đếm trùng trong mẫu, không suy rộng số trùng toàn bộ.
Chọn `exact` để kiểm tra nguyên dòng xuyên mọi batch/shard: khóa là nội dung
đã chuẩn hóa, gồm cả embedding. Chế độ exact cần thêm thời gian và đĩa.
Chọn `off` thì báo rõ phép kiểm tra bị tắt. Bảng tham chiếu được kiểm tra khóa
duy nhất; khóa null/trùng hoặc nhãn ngoài 0/1 khiến pipeline dừng với lỗi.

`max_batches_per_source` chọn các batch đầu theo thứ tự file, không phải mẫu
đại diện. Metadata/report đánh dấu **prefix**; bảng tham chiếu và inventory
vẫn đọc toàn bộ. Không dùng prefix để kết luận về quần thể.

Mỗi nguồn ghi số dòng, batch, thời gian quét trong `metadata.json`. Mỗi bước
cũng ghi `duration_seconds`; thời gian bước chất lượng bao gồm lần quét chung.
Terminal báo số dòng sau mỗi 10 batch và khi hoàn tất nguồn. Tổng thời gian
phân tích chưa gồm xuất bảng/biểu đồ. Xem [phép đo giới hạn](performance_16gb.md).

## API và đầu ra

```python
from mbd_mini_eda import load_config, run_analysis, export_report

config = load_config("configs/mbd_mini_eda.local.json")
result = run_analysis(config, progress=print)
result.sections["descriptive.targets"].tables["target_prevalence"]
export_report(result, plots=True)
```

```text
docs/mbd_mini_eda/
  report.md
  metadata.json
  manifest.json
  tables/descriptive.*/...csv
  figures/...png
```

Chỉ xuất bảng tổng hợp; exporter từ chối cột `client_id` và output/work_dir
trong data_root. Manifest liệt kê file của lần xuất hiện tại. Khi dùng lại
thư mục kết quả cũ, chỉ đọc các file được report/manifest mới liên kết.

## Kiểm chứng

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 -m build --no-isolation
```

Kiểm thử đối chiếu moment/phân vị và tổng hợp với pandas, trùng xuyên batch,
embedding mảng/list/null, ID ngoài cohort, khách không log, timezone, nguồn
rỗng, schema, cấu hình cũ, batch hội thoại, CLI và xuất PNG/JSON/CSV.
