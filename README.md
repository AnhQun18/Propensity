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

## Rà soát bước 1 và bước 2 — 09/11/2026

Đối chiếu tài liệu và kết quả hiện có trong repository với hai bước **Hiểu yêu cầu /
Hiểu nghiệp vụ (Business Understanding)** và **Hiểu dữ liệu (Data Understanding)**:
**bước 1 chưa hoàn chỉnh; bước 2 đã khá đầy đủ về kỹ thuật, còn một số giới hạn
về ý nghĩa dữ liệu cần xác nhận**.

### 1. Hiểu yêu cầu / Hiểu nghiệp vụ (Business Understanding)

Yêu cầu: xác định mục tiêu kinh doanh, đánh giá hiện trạng, đặt mục tiêu kỹ thuật
của việc khai thác dữ liệu và lập kế hoạch triển khai.

| Nội dung cần có | Đánh giá hiện tại |
| --- | --- |
| Mục tiêu kinh doanh | Có định hướng dự đoán khả năng mua; chưa xác định rõ ai sử dụng kết quả, dùng cho quyết định nào và lợi ích cần đạt. |
| Đánh giá hiện trạng | Đã đánh giá dữ liệu, mất cân bằng nhãn, thiếu nguồn và giới hạn tài nguyên; chưa có hiện trạng quy trình chọn khách hoặc hiệu quả kinh doanh để đối chiếu. |
| Mục tiêu kỹ thuật | Đã định nghĩa nhãn, cutoff, lịch sử đầu vào, cách chia thời gian và chỉ số đánh giá; chưa có ngưỡng thành công cụ thể. |
| Kế hoạch triển khai | Có hướng thực nghiệm; chưa có kế hoạch gồm mốc công việc, đầu ra, người phụ trách và điều kiện nghiệm thu. |

**Điểm cần sửa trước tiên là thống nhất bài toán.**
[Thiết kế mới](docs/propensity_feasibility/any_product/design.md) dùng **một xác suất
mua ít nhất một trong bốn sản phẩm**, trong khi [DATASETS.md](DATASETS.md) vẫn mô tả
**bốn xác suất riêng**. Hai đầu ra phục vụ những quyết định khác nhau. Đây là điểm
chưa thống nhất giữa các tài liệu tại thời điểm rà soát.

Để hoàn thiện bước này, cần bổ sung:

- **Mục đích sử dụng:** chẳng hạn xếp hạng khách để ưu tiên liên hệ hằng tháng.
  Đây cần được ghi là mục tiêu đã xác nhận hoặc giả định của đề tài.
- **Tiêu chí thành công:** chọn bao nhiêu khách mỗi tháng — K hoặc tỷ lệ top-K;
  so với cách chọn nào; yêu cầu Recall/Lift@K và chi phí xử lý ra sao.
- **Phạm vi:** khách hiện hữu hay khách mới, một nhãn any-product hay từng sản phẩm,
  đầu ra nghiên cứu hay hệ thống sử dụng thực tế.
- **Kế hoạch:** hoàn thiện dữ liệu → tạo đặc trưng → baseline → đối chứng → đánh giá
  → bàn giao, với đầu ra và điều kiện hoàn thành từng phần.

Chưa có kết quả huấn luyện **không làm bước 1 thiếu**; phần còn thiếu là mục tiêu
và cách xác định dự án thành công.

### 2. Hiểu dữ liệu (Data Understanding)

Yêu cầu: thu thập dữ liệu ban đầu, mô tả dữ liệu, khám phá dữ liệu (EDA) và kiểm tra
chất lượng để phát hiện thiếu sót hoặc bất thường.

| Nội dung cần có | Đánh giá hiện tại |
| --- | --- |
| Thu thập dữ liệu ban đầu | Đã ghi nguồn, vị trí lưu, định dạng và kiểm kê 3.820 file Parquet. |
| Mô tả dữ liệu | Có từ điển, schema, khóa nối, đơn vị quan sát, quy mô và phạm vi thời gian. |
| Khám phá dữ liệu | Có 61 bảng, 24 hình: phân phối, độ phủ nguồn, thời gian, nhãn, hành vi và ngoại lai. |
| Kiểm tra chất lượng | Có kiểm tra null, duplicate, khóa, timestamp, amount, embedding và đối chiếu detail–ptls. |

Các phát hiện chất lượng đã được ghi nhận rõ:

- **Không thiếu bốn nhãn gốc**, không trùng khóa khách–tháng.
- `diff_trans_date` thiếu **120.786 dòng, khoảng 10,04%**; đây là thiếu thuộc tính
  phụ trợ, không phải thiếu nhãn.
- Có **468 khách không có log** trong toàn lịch sử; thiết kế hiện tại giữ nhóm này.
- Không có trùng nguyên dòng; có **một ứng viên giao dịch cùng khách và timestamp**,
  cần phân biệt với giao dịch trùng thật.
- `ptls/trx` ít hơn `detail/trx` **2.020.020 sự kiện ở 13 khách**; dùng `detail`
  làm nguồn lịch sử đầy đủ.
- Nhãn any-product có tỷ lệ dương khoảng **1,07%**, vẫn mất cân bằng mạnh.

Bằng chứng và cách diễn giải nằm trong
[kiểm kê EDA](docs/mbd_mini_eda_audit.md),
[kiểm kê nguồn](docs/mbd_inventory/assessment.md),
[từ điển dữ liệu](docs/dict/mbd_mini.md),
[báo cáo EDA](docs/mbd_mini_eda/report.md) và
[thiết kế any-product](docs/propensity_feasibility/any_product/design.md).

Những phần còn chưa xác minh được gồm **tên nghiệp vụ của bốn sản phẩm, đơn vị
amount và mapping currency, timezone, công thức/thời điểm tạo các trường phụ trợ,
nguyên nhân thiếu nguồn**. Cần quản lý chúng thành danh sách giả định và giới hạn
sử dụng; chưa có căn cứ để tự gán ý nghĩa hoặc loại dữ liệu.

### Kiểm chứng và điều kiện chuyển bước

Ngày 09/11/2026 đã chạy lại bộ xác nhận: **225/225 kiểm tra đạt** cho báo cáo, bảng,
hình, cache và dữ liệu mẫu. Lần rà soát này không quét lại toàn bộ dữ liệu gốc.
Kết quả của lần chạy được lưu cục bộ tại
`outputs/business_data_understanding_review_20261109/docs/mbd_mini_eda_validation.json`;
snapshot xác nhận EDA đã lưu trong Git nằm tại
[docs/mbd_mini_eda_validation.json](docs/mbd_mini_eda_validation.json).

Có thể tiếp tục bước chuẩn bị dữ liệu theo thiết kế hiện tại, đồng thời hoàn thiện
mục tiêu nghiệp vụ và thống nhất tài liệu trước khi chốt đánh giá mô hình.

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
