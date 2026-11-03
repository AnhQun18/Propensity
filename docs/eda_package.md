# Package EDA

Package Python `eda` cung cấp thống kê mô tả, phân tích cơ bản và xuất báo cáo độc lập với schema của từng bộ dữ liệu. Dữ liệu nguồn không bị sửa. Đầu ra mặc định là `docs/<dataset>/`, tính từ thư mục đang chạy lệnh.

## Cài đặt và chạy

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[plots,notebook]"
propensity-eda --config configs/eda_ecommerce.json --plots
python -m eda --config configs/eda_fliggy.json --plots --notebook
python -m eda --config configs/eda_mbd_mini.json --output-dir docs/mbd_mini
```

Lệnh cần chạy trong môi trường đã cài package. Có thể dùng `.\.venv\Scripts\propensity-eda.exe` thay cho `propensity-eda`. Đường dẫn input tương đối trong JSON được giải từ vị trí file cấu hình; output tương đối được giải từ thư mục chạy lệnh. `--notebook` tạo bản báo cáo tĩnh, không chạy lại phân tích và không huấn luyện mô hình.

## API Python

```python
from eda import DatasetSpec, FileSource, profile, export_report

spec = DatasetSpec(
    dataset="sales",
    row_unit="customer event",
    id_columns=["customer_id"],
    numeric_columns=["amount"],
    categorical_columns=["channel"],
    datetime_columns={"event_time": {"timezone": "UTC"}},
    targets={"converted": "binary"},  # Bỏ nếu không có nhãn
    group_columns=["channel"],
)
result = profile(FileSource(paths=["events.csv"]), spec)
# Hoặc: result = profile(df, spec)
result.tables["numeric_summary"]
result.metadata["table_scopes"]
export_report(result, plots=True)
```

`profile` chỉ tính toán và trả về kết quả; không ghi file. `export_report` xuất các kết quả đã tính. Để so sánh phân phối với một DataFrame khác, truyền `reference=reference_df` vào `profile`; các cột cần so sánh phải có ở cả hai bảng.

Tự nhận diện dtype chỉ là mặc định. Cần khai báo ID, mã phân loại dạng số, timestamp, nhãn và đơn vị dòng để tránh diễn giải sai. Các cột datetime có sẵn dtype nhưng không được khai báo sẽ có vai trò `unassigned`; package không tự chọn timezone hay tần suất nghiệp vụ. Giá trị nhãn binary phải là 0/1 hoặc bool; không tự gán nhãn từ mã hành vi.

## Cấu trúc mã nguồn

```text
src/eda/
  __init__.py
  config.py                 # DatasetSpec, FileSource, Relationship
  schema.py                 # Kiểm tra vai trò và cột
  loader.py                 # CSV/Parquet, chunk/batch, manifest, SHA-256
  pipeline.py               # Điều phối tính toán
  result.py                 # ProfileResult, DatasetResult
  descriptive/
    dataset.py              # Quy mô, dtype, preview tùy chọn
    numerical.py            # Mean, median, std, phân vị, skew, kurtosis
    categorical.py          # Tần số, cardinality, nhóm hiếm
    temporal.py             # Phạm vi, ngày/kỳ trống, số sự kiện theo kỳ
    quality.py              # Missing, duplicate, IQR, vô hạn, miền giá trị
  analysis/
    bivariate.py            # Số×số, chữ×số, chữ×chữ
    correlation.py          # Spearman và số quan sát hợp lệ từng cặp
    target.py               # Nhãn, Wilson tùy chọn, dấu hiệu leakage
    drift.py                # KS statistic, total variation
    events.py               # Log, gap, ngày hoạt động, ma trận sparse
    relationships.py        # Khóa và độ phủ giữa bảng, không tự merge
  visualization/
    numerical.py            # Histogram, boxplot
    categorical.py          # Bar, pie, stacked bar
    bivariate.py            # Scatter, grouped boxplot
    correlation.py          # Heatmap
    target.py               # Phân phối và tỷ lệ theo nhóm
    temporal.py              # Hoạt động theo kỳ
    quality.py              # Missing bar và pattern
  utils/stats.py
  report.py
  cli.py
  __main__.py
tests/test_eda.py
```

Các hàm visualization trả về matplotlib Figure; không ghi file và không thay đổi backend khi import package. Hàm thống kê trả về DataFrame hoặc tuple kết quả. Báo cáo tự động xuất các biểu đồ chính, giới hạn số biểu đồ để tránh hàng nghìn hình; các biểu đồ bổ sung gọi trực tiếp từ module tương ứng.

## Đầu ra

```text
docs/<dataset>/
  README.md
  metadata.json
  tables/
    overview.csv
    column_profile.csv
    missing.csv
    duplicates.csv
    numeric_summary.csv
    categorical_summary.csv
    cardinality.csv
    correlations.csv
    temporal_summary.csv
    temporal_counts.csv
    target_summary.csv
    manifest.csv
  figures/                  # Khi có --plots
  eda_report.ipynb           # Khi có --notebook
```

Chỉ tạo bảng phù hợp với cấu hình. Các bảng bổ sung gồm duplicate keys, anomalies, outliers, grouped statistics, target rates, leakage flags, drift và event summaries. Metadata ghi phiên bản package, môi trường, cấu hình, số dòng và phạm vi tính của từng bảng. Mỗi lần xuất sẽ ghi đè các file cùng tên; nên dùng output mới khi thay đổi cấu hình lớn để tránh còn file cũ không nằm trong báo cáo hiện tại.

Với nhiều bảng, CSV nằm trong `tables/<table>/`, biểu đồ trong `figures/<table>/`, thêm `relationships/key_checks.csv` và `relationships/coverage.csv`. Khai báo nhiều bảng bằng `profile_dataset(sources, specs, dataset=..., relationships=...)`, xem cấu hình MBD/Fliggy để biết cấu trúc JSON.

Preview chỉ có qua `eda.descriptive.dataset.preview(df)` và không xuất mặc định. Tóm tắt từng người dùng từ `events.summarize` chỉ được trả về khi gọi trực tiếp; báo cáo mặc định xuất tổng hợp toàn bộ nhóm người dùng.

## Dữ liệu lớn và độ chính xác

- `mode="exact"`: đọc theo chunk nhưng giữ toàn bộ bảng trong RAM để tính thống kê chính xác. Phù hợp bảng vừa bộ nhớ. Duplicate là số dòng lặp sau lần xuất hiện đầu, không phải số cặp trùng.
- `mode="approximate"`: quét toàn bộ input, giữ mẫu ngẫu nhiên đồng đều tối đa `sample_size` dòng bằng random priorities; bộ nhớ làm việc phụ thuộc chunk, mẫu và chiều rộng dữ liệu. Missing, dtype inventory, đếm vô hạn/ngoài miền và kiểm tra hợp lệ nhãn binary dùng toàn bộ dữ liệu. Thống kê số, phân loại, duplicate, nhãn, tương quan, thời gian, sự kiện và drift phía current dùng mẫu. Metadata và Markdown ghi phạm vi này.
- Cardinality, duplicate, tỷ lệ nhãn hiếm và số ngày hoạt động trên mẫu không được coi là kết quả toàn bộ. Khoảng trống thời gian trên mẫu chỉ là kỳ không có dòng trong mẫu, không chứng minh hệ thống thiếu log.
- Kiểm tra quan hệ trên hai mẫu chỉ mô tả các mẫu. Độ phủ và số dòng join dự kiến không phải độ phủ quần thể. Muốn kiểm chứng chính xác, dùng bảng khóa đầy đủ ở exact mode (có thể đọc riêng các cột khóa).
- Quantile tuyến tính, std với ddof=1, skew/kurtosis theo pandas. Moment và quantile bỏ NA và giá trị vô hạn; vô hạn được báo riêng. Không loại outlier khỏi dữ liệu.
- Nhóm thiếu tách khỏi nhóm có chuỗi ký tự giống nhãn missing. Tỷ lệ nhãn dùng số nhãn quan sát làm mẫu số. Wilson chỉ xuất khi `independent_observations=True`; nhiều sự kiện của cùng người thường không độc lập.
- Leakage flags hiện phát hiện cột sao chép nhãn; đây là dấu hiệu cần xem xét, không phải kết luận tự động về thời điểm có sẵn của biến.
- KS/total variation là mô tả thay đổi phân phối; không khẳng định nguyên nhân hay hiệu quả mô hình.
- `hash_sources=True` tính SHA-256 toàn bộ file, thêm một lượt đọc. Cấu hình cho ba bộ lớn đặt false để tránh lượt đọc này; có thể bật lại. Manifest vẫn ghi rõ hash bị tắt. Chưa bảo vệ trước nguồn dữ liệu bị sửa đồng thời trong lúc chạy.

## Thời gian và ma trận sự kiện

Với Unix timestamp, khai báo `unit="s"` hoặc đơn vị thực tế; thời gian được hiểu từ UTC rồi chuyển sang timezone được khai báo. Chuỗi không có timezone được localize theo timezone chỉ định; nếu không chỉ định thì giữ dạng naive. Timestamp lỗi được ghi nhận, không sửa raw. Các thời điểm DST không xác định được chuyển NaT và được đếm vào invalid.

`events.customer_day_matrix` là chức năng tùy chọn, trả về CSR sparse matrix, chỉ mục người dùng và lịch ngày. Phải truyền `start`, `end` trong phạm vi quan sát và xác nhận coverage từ nghiệp vụ. Không tự kéo dài lịch sử ra ngoài dữ liệu. Ma trận không được tạo mặc định trong pipeline.

## Ba cấu hình hiện có

- `eda_ecommerce.json`: nối dọc hai tháng CSV; không tự tạo nhãn mua tương lai.
- `eda_fliggy.json`: CSV không header; giữ mã behavior; UTC dùng để chuẩn hóa kỹ thuật, chưa khẳng định timezone nghiệp vụ. Bảng plan hiện chỉ đọc hai khóa đầu vì schema còn lại chưa xác minh. Audit trên mẫu không thay thế kiểm tra độ phủ toàn bộ.
- `eda_mbd_mini.json`: clients/targets exact, ba bảng detail lấy mẫu; không cộng ptls vào detail. Nhãn theo tháng giữ nguyên; không tự đặt cutoff dự đoán. Loại embedding khỏi phân tích biến đơn, nhưng vẫn kiểm kê missing/dtype. Thời gian MBD giữ naive vì timezone chưa xác nhận.

Các cấu hình lớn là cấu hình chạy toàn bộ nguồn, không giới hạn số chunk. Có thể mất nhiều thời gian dù dùng approximate mode. Package không tự chạy cả ba bộ khi import hoặc cài đặt.

## Kiểm thử và build

```powershell
python -m unittest discover -s tests -v
python -m build --no-isolation
```

Kiểm thử dùng dữ liệu tổng hợp và file tạm để đối chiếu thống kê, CSV nhiều chunk, Parquet partition, nhãn thiếu/nhiều nhãn, khóa lặp, sparse matrix, các module biểu đồ, CLI và output. Không yêu cầu tải raw dataset để kiểm thử.
