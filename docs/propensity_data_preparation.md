# Chuẩn bị dữ liệu propensity any-product

Quy trình để đọc và chạy từng bước nằm tại
[`notebooks/02_data_preparation.ipynb`](../notebooks/02_data_preparation.ipynb).
Notebook có đầy đủ mã xử lí và SQL, chia riêng cấu hình, nhãn, từng nguồn log,
feature theo tháng và hoàn tất. Các cell định nghĩa đồng bộ với helper trong
`src/propensity_data/`; package `mbd_mini_eda` tiếp tục
phục vụ thống kê mô tả. Pipeline này **chưa được chạy trên dữ liệu thật hoặc chạy
kiểm thử tại máy hiện tại**, theo yêu cầu ngày 09/11/2026. Các ví dụ bên dưới dành
cho máy sẽ thực thi sau này. Chưa có training features hoặc kết quả mô hình mới.

## Cài đặt và thực thi trên máy khác

Dùng **Python >=3.11**, dữ liệu gốc có `client_split/`, `targets/`, `detail/trx/`,
`detail/geo/`, `detail/dialog/`. Không cần tải lại nếu máy đã có dữ liệu.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[prepare,notebook]"
Copy-Item configs/propensity_data.json configs/propensity_data.local.json
```

Sửa `data_root` và `output_dir` trong bản `.local.json`. Đường dẫn tương đối tính từ
file JSON. Cấu hình mặc định dùng DuckDB 6 GB, hai luồng; đây là giới hạn cho engine,
không phải cam kết tổng RAM tiến trình. Cần dung lượng đĩa cho bản chuẩn hóa, spill,
audit và features; chưa đo dung lượng hoặc thời gian trên raw. Không nạp toàn bộ
log thành pandas DataFrame. Tắt embedding mặc định để giảm chi phí baseline.

Mở `notebooks/02_data_preparation.ipynb`, chọn kernel đã cài dependency rồi thực hiện
cell theo thứ tự. Mục 1–4 định nghĩa quy tắc và helper; từ mục 5 bắt đầu đọc raw và
ghi output. Nhãn, trx, geo, dialog và tạo feature có bước riêng để xem kiểm kê.
Hàm `iter_supervised_batches` chỉ trả dòng có nhãn hợp lệ cho bước mô hình, không
xóa các dòng chưa có nhãn khỏi quần thể scoring.

Notebook mới chưa thực thi, mọi output trống. Mã helper tương ứng chia theo nhiệm
vụ trong `src/propensity_data/{schema,storage,quality,features,protocol,config}.py`.
Nếu cần dùng terminal thay cho notebook, các lệnh CLI bên dưới vẫn được hỗ trợ.

```powershell
# Chỉ đọc cấu hình, không mở raw hoặc tạo output.
.\.venv\Scripts\python.exe -B -m propensity_data plan --config configs/propensity_data.local.json

# Kiểm thử bằng dữ liệu tổng hợp; chỉ chạy trên máy được phép thực thi.
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_propensity_data.py -v

# Thực sự chuẩn hóa và tạo đặc trưng; không fit mô hình.
.\.venv\Scripts\python.exe -B -m propensity_data prepare --config configs/propensity_data.local.json
```

Có thể dùng entry point `propensity-data` sau khi cài package. Import package không
đọc dữ liệu và không tự khởi chạy công việc. Pipeline từ chối ghi đè thư mục output
đã tồn tại; khi chạy lại, chọn một output mới. Output không được chồng với cây raw.
Không tự xóa kết quả lỗi, vì còn cần truy vết nguyên nhân.

## Quy tắc chuẩn hóa

- Dùng `detail`; không dùng `ptls` hoặc cache EDA để xây feature.
- Giữ nguyên raw, không tự deduplicate. Xuất nhóm cùng khách/timestamp để audit;
  trùng timestamp chưa đủ chứng minh trùng giao dịch.
- Chuẩn hóa ID thành chuỗi, bỏ khoảng trắng biên. ID trống, timestamp không hợp lệ
  và ID ngoài `client_split` được đưa vào quarantine kèm tên file/số dòng nguồn.
  Timestamp được diễn giải theo UTC, không dịch ba giờ.
- Thuộc tính phụ thiếu không làm mất sự kiện: count vẫn đếm sự kiện hợp lệ về
  ID/time. Amount không parse được hoặc không hữu hạn giữ null, có cờ invalid;
  amount thiếu có cờ riêng. Amount âm hữu hạn được giữ và đếm riêng.
- Currency nguyên 11 là mã chính theo quy ước dự án; mã nguyên khác có giá trị mang
  cờ ngoại tệ. Currency thiếu và currency không hợp lệ được đếm riêng. Không suy
  ra ISO hoặc tỷ giá; không cộng amount giữa các mã khác nhau.
- Giữ các mã giao dịch dạng chuỗi trong bảng chuẩn hóa, dùng `__MISSING__` cho
  categorical thiếu. Các feature xuất chính hiện là số: count, cardinality,
  tỷ lệ thiếu và tỷ lệ category phổ biến nhất. Tần suất từng mã event type/subtype
  xuất ở sidecar.
- Không dùng `fold`, `trans_count`, `diff_trans_date` nguồn làm predictor.

## Nhãn và quần thể

`mon` gốc là tháng báo cáo (trong MBD thường là ngày cuối tháng), được chuẩn hóa về
ngày đầu **cùng tháng**, không dịch sang tháng tiếp theo. Ví dụ `2022-02-28` → cutoff
`2022-02-01 00:00 UTC`. Khóa trùng được kiểm tra sau chuẩn hóa tháng.

Một dòng cho mỗi khách trong `client_split` × mỗi tháng được cấu hình. Đây là quần
thể cố định cho nghiên cứu offline; dữ liệu không có ngày mở tài khoản để chứng minh
mọi khách đều đủ điều kiện liên hệ tại mọi cutoff thực tế.

`y_any=1` khi có ít nhất một target bằng 1, kể cả khi target khác null; `y_any=0`
khi cả bốn bằng 0; còn lại null. Nhãn ngoài 0/1/null, khóa khách–tháng trùng, tháng
không hợp lệ hoặc target có ID ngoài `client_split` làm pipeline dừng. Mỗi tháng cấu
hình cần ít nhất một dòng target; nếu mất hoàn toàn một tháng thì dừng để kiểm tra
nguồn. Không gán
nhãn thiếu thành 0. Dòng thiếu target vẫn xuất để inference, với
`supervised_eligible=false`; chỉ train/tính metric trên dòng có nhãn.

Giữ khách ít dữ liệu và cold-start. `cold_start=1` nếu không nguồn nào có sự kiện
trước cutoff. Khi xây mô hình, so sánh mô hình chung có masks với fallback tỷ lệ
mua cold-start từ train, shrink về tỷ lệ chung. Pipeline không ước lượng fallback
trên toàn bộ nhãn để tránh lộ thông tin validation/test.

## Feature và dữ liệu thiếu

Cutoff là đầu tháng nhãn lúc 00:00 UTC. Chỉ nhận `event_time < cutoff`; cửa sổ ngày
là `[cutoff - days, cutoff)`, không phải tháng lịch. Mặc định có 30 ngày, 84 ngày
và toàn bộ lịch sử trước cutoff, độc lập theo nguồn.

| Nhóm | Feature chính |
| --- | --- |
| Mọi nguồn | Count, log1p count, active days, has-event, recency, tuổi bản ghi đầu trong cửa sổ, ever-seen trong toàn lịch sử |
| Độ phủ | Số ngày cửa sổ chồng phạm vi nguồn, cờ lịch sử bị cắt |
| Trx | Count/mean/median/max/sum/absolute sum amount mã 11; signed-log mean; ngoại tệ, currency/amount thiếu và invalid; số mã phân biệt và top-share event type/subtype |
| Geo | Số mã geohash 4/5/6 phân biệt, count mã thiếu, tỷ lệ geohash 6 phổ biến nhất trên các log có mã |
| Dialog | Metadata; tùy chọn mean và last embedding hợp lệ trong cửa sổ embedding |

Phạm vi nguồn dùng **timestamp hợp lệ sớm nhất**, không dùng timestamp cuối trong
tương lai để suy ra độ phủ tại cutoff. Số ngày chồng là phạm vi trích xuất theo
lịch, không chứng minh hệ thống quan sát khách liên tục. Không gán nguyên nhân
thiếu cho từng khách khi không có bằng chứng.

Trong phần cửa sổ có phạm vi nguồn: không có log → count 0, has-event 0; mean,
embedding và recency giữ null. Cửa sổ hoàn toàn trước phạm vi nguồn → thống kê
count cũng null, overlap 0 và truncated 1. Cửa sổ chồng một phần chỉ đếm log trong
phần dữ liệu thực có; không diễn giải thành đã quan sát đủ cả cửa sổ.

Không xóa heavy users, không cap amount tự động. `log1p` dùng cho count/absolute
sum; amount có dấu dùng `sign(x) * ln(1 + abs(x))`. Cap, rank, scaler hoặc PCA sau
này phải fit trên train của từng split. Không forward-fill geo hoặc tạo dialog giả.

Để bật nội dung dialog, đặt `include_dialog_embeddings=true`. Mặc định chỉ pooling
ở cửa sổ 84 ngày, kiểm tra đúng 768 chiều và mọi giá trị hữu hạn; embedding lỗi
không làm mất event metadata. Mean chỉ dùng vector hợp lệ; last chọn timestamp
mới nhất, phá hòa bằng tên file và số dòng. Có thể đổi chiều cho fixture tổng hợp.
Chưa có PCA hoặc encoder được fit; 768 chiều × mean/last làm tăng đáng kể số cột.

## Đầu ra

```text
outputs/propensity_data_v1/
├── manifest.json                 # running / failed / ready; số dòng theo nguồn/tháng
├── source_inventory.json         # path, size, mtime; không phải checksum byte raw
├── feature_schema.json           # danh sách predictors và nhóm ablation
├── protocol.json                 # lịch fit/validation/calibration/test
├── features/month=YYYY-MM/part.parquet
├── currency/month=YYYY-MM/{30d,84d,all}.parquet
├── categories/month=YYYY-MM/{30d,84d,all}.parquet
├── quarantine/{trx,geo,dialog}.parquet
├── audit/                        # nhãn chưa xác định, nhãn/tháng, timestamp collisions
├── work.duckdb                   # bản chuẩn hóa, giữ lại để truy vết
└── spill/                        # vùng tạm của engine nếu cần
```

Chỉ nhận kết quả khi `manifest.status=ready`; output lỗi có thể đã chứa một số tháng.
Pipeline kiểm tra inventory cuối vẫn khớp đầu, số khách không giảm khi join và schema
nhất quán giữa các tháng. Dữ liệu chi tiết sinh ra nằm trong `outputs/`, đã bị Git
ignore. `work.duckdb` là sản phẩm của lần chạy, **không có resume tự động**.

Chọn feature bằng whitelist `feature_schema.json.predictors`; không suy ra feature
bằng cách bỏ riêng `y_any`, vì file còn bốn nhãn nguồn và metadata. Các nhóm ablation
cho phép so sánh trx → thêm metadata nguồn → thêm geo → thêm dialog trên cùng cohort.
Sidecar category có thể pivot thêm theo vocabulary học trên train; sidecar currency
không được gộp sum của các mã thành cùng đơn vị tiền.

## Protocol và bước mô hình sau này

Mặc định: validation cuốn chiếu 09–11/2022, fit cuối trên 02–11, calibration 12/2022,
test khóa 01/2023. Không refit base model sau khi fit calibrator. Đây là **giả định
nhãn sẵn ngay khi tháng kết thúc**, chưa phải độ trễ vận hành đã xác minh.

`label_delay_days` thêm độ trễ ngày cố định vào đầu tháng tiếp theo. Các tháng train
được lọc theo thời điểm nhãn sẵn; nếu calibration hoặc validation chưa sẵn khi cần,
pipeline dừng thay vì âm thầm dùng tương lai. Người chạy phải chọn lại tháng
calibration/validation. Độ trễ một tháng lịch cần protocol riêng, không thay bằng
30 ngày rồi coi tương đương.

Sau khi chuẩn bị dữ liệu được xác nhận trên máy thực thi: bắt đầu một CatBoost
binary trên phân bố tự nhiên; đối chứng trọng số lớp và nội dung từng nguồn qua
validation. Báo cáo Recall/Precision/Lift@1%, AP, ROC-AUC, log loss và calibration;
giữ cold-start trong mẫu số. **Package này chỉ chuẩn bị dữ liệu, không huấn luyện,
calibrate, chọn mô hình hoặc tạo kết quả metric.**

Tham chiếu: [thiết kế any-product](propensity_feasibility/any_product/design.md),
[audit cột phụ trợ](propensity_feasibility/remaining_questions/assessment.md),
[cực trị](propensity_feasibility/tail_review/assessment.md),
[DuckDB Parquet](https://duckdb.org/docs/stable/data/parquet/overview.html),
[DuckDB aggregates](https://duckdb.org/docs/stable/sql/functions/aggregates.html).
