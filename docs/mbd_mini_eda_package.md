# Package thống kê mô tả và EDA MBD-mini

Package `mbd_mini_eda` triển khai 16 bước: 7 mục thống kê mô tả, 2 bước tạo
bảng tổng hợp và 7 câu hỏi khám phá. Package đọc trực tiếp Parquet, không
dùng các kết quả EDA trước đó, không sửa dữ liệu nguồn và không huấn luyện mô hình.

## Cài và chạy

Với máy khác đã có dữ liệu, xem [hướng dẫn setup từ GitHub](setup_other_machine.md).

Chạy trong thư mục dự án:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[plots]"
.\.venv\Scripts\python.exe -X utf8 -m mbd_mini_eda plan --config configs/mbd_mini_eda.json
.\.venv\Scripts\python.exe -X utf8 -m mbd_mini_eda run --config configs/mbd_mini_eda.json --plots
```

Entry point tương đương: `mbd-mini-eda run --config configs/mbd_mini_eda.json --plots`.
`plan --format json` xuất kế hoạch máy đọc được mà không mở dữ liệu nguồn.
`run --output-dir PATH` ghi vào một thư mục khác; `--quiet` tắt tiến độ.
Không truyền `--plots` nếu chỉ cần CSV/Markdown/JSON.

Đường dẫn trong JSON tính từ vị trí file JSON; `--output-dir` tính từ cwd.
`MBDMiniConfig()` trực tiếp dùng đường dẫn mặc định tính từ cwd.

## Các module

```text
src/mbd_mini_eda/
  config.py                 # Cấu hình và kiểm tra tham số
  schema.py                 # Vai trò cột, cấp dòng, khóa
  models.py                 # AnalysisContext / SectionResult / AnalysisResult
  io.py                     # Parquet theo batch, Hive partition, bảng nhãn
  engine.py                 # Quét một lần, SQLite tạm, bộ lấy mẫu có giới hạn
  statistics.py             # Moment gộp, phân vị, nhóm nhãn, lấy mẫu
  pipeline.py               # Thực thi theo phụ thuộc; ghi rõ bước được bỏ qua
  cli.py                    # plan / run
  descriptive/
    structure.py            # Cấu trúc và quy mô
    quality.py              # Null, hợp lệ, trùng, khóa nối
    temporal.py             # Phạm vi nguồn, hoạt động theo tháng, lịch sử
    transactions.py         # Phân phối amount và mã giao dịch
    modalities.py           # Vị trí, hội thoại, embedding
    customers.py            # Hoạt động và độ phủ nguồn theo khách
    targets.py              # Bốn nhãn, tháng/fold, đồng xuất hiện
  features/
    shared.py               # Tổng hợp chung và ghép đúng cardinality
    customers.py            # Một dòng/client_id, toàn lịch sử đã quét
    customer_month.py       # Một dòng/(client_id, mon), trước cutoff
  exploration/
    shared.py               # Nhóm cố định dùng chung giữa tháng/cửa sổ
    temporal.py             # Xu hướng tháng và mẫu hình ngày/giờ
    segments.py             # Nhóm hoạt động và tỷ lệ nhãn
    modalities.py           # Tổ hợp nguồn và nhãn
    missingness.py          # Null thuộc tính và độ thưa log
    target_relationships.py # Feature lịch sử theo từng nhãn
    correlations.py         # Spearman theo tháng/cửa sổ, số cặp hợp lệ
    stability.py            # Dao động qua tháng/fold, cờ nhóm ít mẫu dương
  visualization.py          # Figure từ các bảng kết quả
  reporting.py              # CSV, PNG, Markdown, JSON và manifest
```

Mỗi module phân tích có `run(context) -> SectionResult`. Các bước đọc cùng
kho tổng hợp đã quét, thay vì quét lại raw cho từng mục.

## Cấu hình và phạm vi tính

| Tham số | Mặc định API | Ý nghĩa |
|---|---|---|
| `batch_size` | 100000 | Số dòng Parquet mỗi batch |
| `history_days` | [30,84,180] | Các cửa sổ lịch sử cần so sánh |
| `sample_size` | 5000 | Mẫu dòng ngẫu nhiên tối đa mỗi nguồn |
| `random_seed` | 42 | Tái lập mẫu với cùng thứ tự file/dòng |
| `duplicate_mode` | sample | exact / sample / off |
| `max_categories` | 20 | Top nhóm; phần còn lại và null được báo riêng |
| `embedding_size` | 768 | Chiều embedding dùng cho kiểm tra shape |
| `label_alignment` | unresolved | unresolved / calendar_month / explicit |
| `label_windows_path` | null | CSV/Parquet mapping khi chọn explicit |
| `analysis_months` | [] | Chọn tháng cho feature/EDA với nhãn; [] là tất cả |
| `timezone` | null | Giữ naive như file; chỉ localize/convert khi khai báo |
| `correlation_sample_size` | 50000 | Giới hạn dòng mỗi tháng cho Spearman |
| `min_group_size` | 30 | Ngưỡng cờ nhóm nhỏ |
| `min_positive_count` | 5 | Ngưỡng cờ ít mẫu dương |
| `work_dir` | null | Nơi chứa kho SQLite tạm; null dùng temp hệ thống |
| `max_batches_per_source` | null | Chỉ dùng để chạy thử prefix; null quét đủ |

Số dòng, null, giá trị không hợp lệ, tần suất categorical, hoạt động theo thời
gian, các tổng hợp khách hàng và cửa sổ được tính trên toàn bộ phần đã quét.
Moment amount dùng phép gộp mean/M2 để tính std mẫu với ddof=1.
Phân vị, histogram và outlier amount dùng mẫu dòng ngẫu nhiên đồng đều;
bảng ghi riêng `moments_scope`, `quantile_scope` và số quan sát hữu hạn.
Tương quan có `scope` và số cặp hợp lệ riêng từng dòng.

`duplicate_mode=exact` lưu khóa nội dung chuẩn hóa của toàn dòng trong SQLite,
đếm số dòng lặp sau lần xuất hiện đầu, xuyên mọi batch/shard. Không chỉ so
hash; mảng embedding được chuẩn hóa theo nội dung. Chế độ này cần nhiều đĩa
và thời gian với dữ liệu lớn. Mặc định sample chỉ đếm trùng trong mẫu và
không suy rộng thành số trùng toàn bộ; off ghi rõ phép kiểm tra bị tắt.

SQLite chỉ chứa tổng hợp theo khách/ngày, vị trí khác nhau, tần suất mã và
khóa kiểm tra trùng nếu bật exact. Nó được dọn khi kết thúc, kể cả khi lỗi.
RAM phụ thuộc batch, mẫu và các bảng tổng hợp/nhãn; không giữ toàn bộ raw.
Bảng khách hàng–tháng vẫn cần RAM tỷ lệ với số khách × tháng × số cửa sổ.
Kho tạm trên đĩa có thể lớn, đặc biệt khi kiểm tra trùng exact.

`max_batches_per_source` lấy các batch đầu theo thứ tự file, không phải mẫu
đại diện. Metadata và báo cáo ghi **prefix**. Các bảng tham chiếu và inventory
file vẫn có phạm vi toàn bộ. Không dùng báo cáo prefix kết luận về quần thể.

## Căn chỉnh nhãn và giữ khách hàng

API mặc định `unresolved`: vẫn chạy thống kê mô tả, xu hướng và missingness;
các mục cần feature theo nhãn được ghi `skipped`, kèm lý do.

File `configs/mbd_mini_eda.json` chọn rõ `calendar_month`, theo quy ước hiện có
trong [nghiên cứu mon](propensity_feasibility/mon_research.md):
`mon=2022-02-28` tương ứng nhãn trong `[2022-02-01,2022-03-01)`, cutoff đầu
tháng 02/2022. Đây là lựa chọn triển khai được ghi trong metadata.

Để thay quy ước, chọn `explicit` và cung cấp bảng:

```csv
mon,cutoff,label_start,label_end
2022-02-28,2022-02-01,2022-02-01,2022-03-01
```

Mỗi mon có một mapping duy nhất; `cutoff <= label_start < label_end`.
Cutoff phải là đầu ngày để việc truy vấn từ tổng hợp ngày có biên chính xác.
Feature lấy trong `[cutoff-history_days,cutoff)`; sự kiện đúng cutoff bị loại.
Nhãn chính thức được giữ nguyên, không suy ra từ sự kiện và không tự bỏ
nhãn chỉ vì khoảng nhãn nằm sau log. Không đưa `trans_count`/`diff_trans_date`
vào feature khi chưa xác minh thời điểm tính.

Giữ mọi khách trong client_split bằng left join có kiểm tra cardinality.
Count/active_days/has_source bằng 0 khi không có log; amount_mean, recency
và thời điểm sự kiện không xác định giữ NaN/NaT. Khách hàng–tháng chỉ chứa
khóa, fold, nhãn, mốc thời gian và các feature trước cutoff có hậu tố cửa sổ.
Bảng toàn lịch sử không được dùng để so nhãn tháng quá khứ.

Null thuộc tính, không có log, span lịch sử và ngày ngoài biên nguồn được
mô tả riêng. Tỷ lệ ngày không có log dùng biên quan sát của từng nguồn,
không xác nhận quan sát liên tục hay nguyên nhân thiếu nguồn.
Các sự kiện có ID/timestamp hợp lệ vẫn được đếm hoạt động dù thuộc tính
phụ null; embedding không hợp lệ được ghi riêng, loại khỏi norm.
ID ngoài client_split được báo trong chất lượng, không thêm vào cohort.

## API và đầu ra

```python
from mbd_mini_eda import load_config, run_analysis, export_report

config = load_config("configs/mbd_mini_eda.json")
result = run_analysis(config, progress=print)
result.sections["descriptive.targets"].tables["target_prevalence"]
export_report(result, plots=True)
```

Muốn gọi module riêng, dùng `with AnalysisContext(...) as context`, nạp
clients/targets bằng `read_reference_tables` và gọi theo phụ thuộc trong
`build_plan()`. Context manager chịu trách nhiệm đóng kho tạm.

```text
docs/mbd_mini_eda/
  metadata.json
  manifest.json
  report.md
  tables/<section>/*.csv
  figures/*.png
```

Chỉ xuất bảng tổng hợp; exporter từ chối bảng có cột client_id và từ chối
output/work_dir trong data_root. Manifest liệt kê đúng file của lần xuất
hiện tại. Nếu dùng lại thư mục, file cũ không còn được dùng có thể vẫn tồn
tại; chỉ đọc các file được report/manifest hiện tại liên kết.

Các nhóm hoạt động dùng ranh giới cố định giữa tháng/fold/cửa sổ. Mỗi tỷ lệ
nhãn kèm số khách, số dòng, số nhãn quan sát và số dương. Stability mô tả
dao động và gắn cờ nhóm nhỏ; không giả định các tháng của cùng khách độc lập.
EDA mô tả liên hệ, không khẳng định tác động nhân quả.

## Kiểm chứng

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -X utf8 -m build --no-isolation
```

Các kiểm thử dùng Parquet tự tạo để đối chiếu moment/phân vị với pandas,
kiểm tra trùng và tổng hợp xuyên shard/batch, nhãn thiếu, embedding lỗi,
ID ngoài cohort, khách không log, sự kiện đúng cutoff, mapping explicit,
timezone, nguồn rỗng, chế độ prefix, CLI và xuất PNG/JSON/CSV.
