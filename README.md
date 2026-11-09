# New Propensity

Repository chứa package **7 bước thống kê mô tả** và tài liệu nghiên cứu propensity cho MBD-mini.

## Chuẩn bị dữ liệu propensity

Pipeline mới ở [`src/propensity_data/`](src/propensity_data/), cấu hình tại
[`configs/propensity_data.json`](configs/propensity_data.json). Pipeline tạo một dòng
khách–tháng, nhãn `y_any`, đặc trưng trước cutoff UTC, quarantine và lịch đánh giá;
giữ khách thiếu nguồn, không cap cực trị hoặc gộp amount khác currency.
Xem [hướng dẫn chuẩn bị và chạy trên máy khác](docs/propensity_data_preparation.md).

**Theo yêu cầu ngày 09/11/2026: mới viết code và kiểm thử tổng hợp, chưa thực thi
pipeline hoặc chạy kiểm thử trên máy này.** Chưa có kết quả huấn luyện mới.
Mã chuẩn bị tách riêng khỏi EDA; output nằm trong `outputs/` và không đưa lên Git.
Bản PDF sơ bộ cũ đã chuyển vào [kho báo cáo cũ](deliverables/archive/README.md).

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
**bước 1 đã có đặc tả nghiên cứu đề xuất, còn thiếu xác nhận nghiệp vụ thực tế;
bước 2 đã khá đầy đủ về kỹ thuật, các giới hạn của dữ liệu ẩn danh được ghi rõ**.
Kết quả xác minh tiếp theo và câu trả lời đầy đủ nằm tại
[các câu hỏi còn bỏ ngỏ](docs/propensity_feasibility/open_questions/answers.md).

### 1. Hiểu yêu cầu / Hiểu nghiệp vụ (Business Understanding)

Yêu cầu: xác định mục tiêu kinh doanh, đánh giá hiện trạng, đặt mục tiêu kỹ thuật
của việc khai thác dữ liệu và lập kế hoạch triển khai.

| Nội dung cần có | Đánh giá hiện tại |
| --- | --- |
| Mục tiêu kinh doanh | Đề xuất xếp hạng khách hiện hữu hằng tháng để CRM/marketing ưu tiên liên hệ; đo số khách có mua được tìm thấy trong cùng ngân sách chọn khách. Chưa xác nhận đơn vị tiếp nhận hoặc hiệu quả chiến dịch thực tế. |
| Đánh giá hiện trạng | Có raw, EDA và audit; chưa có mô hình đã đánh giá. Không có dữ liệu quy trình chọn khách hiện hành, chi phí, lợi nhuận hoặc phân bổ chiến dịch để xác định ROI. |
| Mục tiêu kỹ thuật | Một xác suất `p_any`; top 1% làm ngân sách mô phỏng, thêm 0,5%/5%; so với random và baseline giao dịch. Đề xuất Lift@1% > 1, báo cáo chênh lệch paired, bất định và độ ổn định theo tháng; chưa có kết quả chứng minh đã đạt. |
| Kế hoạch triển khai | Đã đề xuất M0–M5: khóa protocol → chuẩn bị dữ liệu → baseline → đối chứng → test → bàn giao, với đầu ra và vai trò đề xuất. Tên người phụ trách, deadline và ngân sách thực tế chưa được duyệt. |

**Bài toán hiện tại đã được thống nhất trong tài liệu tổng quan:**
[thiết kế mới](docs/propensity_feasibility/any_product/design.md) dùng **một xác suất
mua ít nhất một trong bốn sản phẩm**; bốn nhãn gốc được giữ để tạo `y_any`.
[DATASETS.md](DATASETS.md) và từ điển đã phân biệt đầu ra đề tài với benchmark bốn sản
phẩm. Các báo cáo nghiên cứu cũ vẫn là tài liệu theo phạm vi/thời điểm ghi trong file.

Các câu hỏi nghiệp vụ được trả lời bằng đặc tả nghiên cứu sau; đây là **đề xuất**, chưa
phải cam kết đã được ngân hàng nghiệm thu:

- **Mục đích sử dụng:** danh sách ưu tiên liên hệ cho CRM/marketing hoặc quản lý quan
  hệ khách hàng; tăng số khách có mua được tìm thấy trong cùng K. Bối cảnh MBD là khách
  hàng doanh nghiệp. Không suy ra tác động của việc liên hệ từ nhãn mua quan sát được.
- **Tiêu chí thành công:** K = ceil(1% × N), bằng 1.003 khi N = 100.224; đo
  Precision/Recall/Lift@K, AP và calibration. So với chọn ngẫu nhiên, xếp hạng bằng hành
  vi đơn giản và CatBoost giao dịch, trên cùng cohort/split. Chưa có ngân sách thực tế
  hoặc đủ dữ liệu để đặt mục tiêu doanh thu/ROI. Thời gian và RAM phải được đo khi chạy.
- **Phạm vi:** nghiên cứu offline cho khách hiện hữu, đầu ra `p_any`; giữ khách thiếu
  nguồn/cold-start với fallback. Không đưa ra tên sản phẩm đề xuất hoặc kết luận cho
  khách hoàn toàn mới khi chưa có thí nghiệm tương ứng.
- **Kế hoạch:** tuning cuốn chiếu 09–11/2022, fit cuối trên 02–11, calibration tháng 12,
  test khóa tháng 01/2023. Bảng M0–M5 trong [bản trả lời](docs/propensity_feasibility/open_questions/answers.md)
  nêu đầu ra, điều kiện hoàn thành và vai trò đề xuất; chưa tự gán tên người/deadline.

Chưa có kết quả huấn luyện **không làm bước 1 thiếu**; phần còn thiếu là mục tiêu
và cách xác định dự án thành công. Đặc tả trên đóng phần thiết kế nghiên cứu; mục tiêu
kinh doanh thực tế vẫn cần đầu mối sử dụng xác nhận.

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

Kết quả xác minh bổ sung ngày 09/11/2026:

| Câu hỏi | Câu trả lời và cách áp dụng |
| --- | --- |
| Bốn sản phẩm là gì? | Tên thương mại không còn là yêu cầu xác minh của đề tài any-product theo quyết định người dùng. Giữ bốn nhãn nguồn để tạo và đối soát `y_any`. |
| Amount/currency xử lý thế nào? | Theo quyết định người dùng, mã 11 là mã chính, mã khác có giá trị mang cờ ngoại tệ của dự án. Đã thêm hàm tạo cờ: 38.724.240 dòng chính, 40.572 ngoại tệ, 118 missing tách riêng. Không gán tên đồng tiền thật; giữ currency gốc. |
| Tháng nhãn và timezone? | Người dùng quyết định tạm dùng UTC thống nhất: raw diễn giải UTC, cutoff đầu tháng lúc 00:00 UTC, không dịch 3 giờ. Đây là quy ước triển khai, không phải xác nhận ETL gốc. Moscow trong EventGPT giữ làm tài liệu đối chiếu và sensitivity check. |
| Biên 3 giờ có ảnh hưởng không? | Quét 1.275 shard detail: 88.976 sự kiện khác biệt, ảnh hưởng 71.467 cặp khách–tháng / 35.660 khách. Cutoff tháng 01/2023 không khác sự kiện, nhưng train/validation/calibration có khác. Đây chưa phải mức thay đổi metric. |
| Công thức các trường phụ trợ? | Trans_count không giảm giữa các tháng, nhưng 612 cấu hình không tái tạo được cột nguồn; 102.932 dòng vượt toàn bộ trx trước cutoff, 9.459 dòng dương thuộc khách không có trx toàn lịch sử. Chưa dùng count/diff nguồn làm predictor; tự tạo count/recency từ detail trước cutoff. |
| Vì sao không có log? | MBD mục 3.1 xác nhận cơ chế chung: khách có thể không giao dịch, không gọi hỗ trợ hoặc không để lại vị trí khi dùng ứng dụng. Audit đã tách các trạng thái trống theo tháng; không có cờ quan sát liên tục để gán nguyên nhân chính xác cho từng khách. Giữ khách và masks. |
| Nhãn có cần loại do thiếu không? | Audit mới xác nhận 0 null/giá trị nhãn ngoài 0/1, 0 trùng khóa; không cần loại dòng vì mất nhãn. `y_any` có 12.854 dòng dương. |
| Khi nào nhãn sẵn có? | Đã kiểm tra 435 shard và metadata: không có thời điểm chốt/công bố nhãn từng tháng. Snapshot được công bố ngày 27/08/2024, không cho biết độ trễ trong ngân hàng. Lịch calibration 12/2022 → test 01/01/2023 chỉ khả thi dưới giả định trễ 0; kịch bản trễ dương cần dựng lại lịch fit. |

Nguồn sơ cấp, commit mã nguồn, mức độ chắc chắn, các câu hỏi cần nhà phát hành/đơn vị
sử dụng trả lời và kế hoạch chi tiết nằm trong
[bản trả lời đầy đủ](docs/propensity_feasibility/open_questions/answers.md).

### Kiểm chứng và điều kiện chuyển bước

Ngày 09/11/2026 đã chạy lại bộ xác nhận: **225/225 kiểm tra đạt** cho báo cáo, bảng,
hình, cache và dữ liệu mẫu. Bộ xác nhận EDA này không quét lại toàn bộ dữ liệu gốc.
Kết quả của lần chạy được lưu cục bộ tại
`outputs/business_data_understanding_review_20261109/docs/mbd_mini_eda_validation.json`;
snapshot xác nhận EDA đã lưu trong Git nằm tại
[docs/mbd_mini_eda_validation.json](docs/mbd_mini_eda_validation.json).

Audit bổ sung đọc lại toàn bộ targets và chỉ chiếu ID/timestamp trên toàn bộ shard
detail để kiểm tra nhãn, trường phụ trợ và biên timezone; không sửa raw hoặc fit mô hình.
Kết quả: [verification.json](docs/propensity_feasibility/open_questions/verification.json).
Chạy lại bằng `.\.venv\Scripts\python.exe -B -X utf8 src/audit_open_questions.py`.

Đã kiểm chứng thêm ba giả thuyết cho `diff_trans_date` trên toàn bộ targets và
detail/trx, với 1.134 cấu hình mốc/cửa sổ/timezone/làm tròn; chưa công thức nào tái tạo
hoàn toàn cột nguồn. Xem [báo cáo ba công thức](docs/propensity_feasibility/three_formulas/assessment.md)
và script `src/audit_three_diff_formulas.py` để chạy lại.

Đã tự kiểm chứng thêm trans_count, thiếu nguồn, độ trễ nhãn và quy ước cờ currency
trong [báo cáo các điểm còn lại](docs/propensity_feasibility/remaining_questions/assessment.md).
Kết quả máy đọc và 612 cấu hình nằm cùng thư mục. Hàm tạo cờ có tại
`src/mbd_mini_eda/currency.py`; currency thiếu không bị gán ngoại tệ. Các ngày sẵn có
nhãn theo độ trễ 0/1/7/30 ngày hoặc một tháng lịch là **kịch bản giả định**, chưa phải
ngày chốt nhãn đã xác minh. Chưa chạy mô hình để đo ảnh hưởng của độ trễ.

Có thể tiếp tục bước chuẩn bị dữ liệu với UTC được ghi rõ, đồng thời xác nhận mục tiêu
nghiệp vụ trước khi chốt đánh giá mô hình. Các audit/cohort cũ giữ nguyên phạm vi; cần
ghi rõ chúng được diễn giải theo quy ước UTC, không chuyển sang cutoff Moscow.

### Kiểm tra lại cực trị và top khách theo giao dịch (09/11/2026)

Đã rà soát raw amount theo currency, lịch sử/loại giao dịch của top 50 khách,
nhãn của top 20 và mức tập trung đỉnh ngày. Kết quả đầy đủ trong
[báo cáo đuôi phân phối](docs/propensity_feasibility/tail_review/assessment.md).

- **IQR không phải tỷ lệ lỗi:** với mã 11, số dòng ngoài hàng rào giảm từ
  5.521.385 (**14,2582%**) trên thang gốc xuống 584.139 (**1,5085%**) trên
  `log1p(amount)`. Cả hai chỉ mô tả phân phối. Nhóm mã 5/9/12/16 được gộp thành
  `other_currency` (19 dòng), không diễn giải IQR nhóm này; 118 dòng thiếu mã giữ riêng.
- **Max amount 1.370.187.366.400** xuất hiện một lần, nhưng khách sở hữu có
  53/56 giao dịch vượt P99,9 mã 11 và median khoảng 164,54 tỷ đơn vị amount gốc.
  Max bằng 8,33 lần median khách: chưa thỏa giả thuyết vài điểm cao đơn lẻ để cap
  tự động. P99,9 toàn kỳ khoảng 57,32 triệu chỉ là tham chiếu EDA.
- **Top 20** có 2.214.115 log (5,7116% toàn bộ trx), đủ 12 tháng nhãn/khách.
  Chỉ hạng 14 và 16 có một tháng `y_any=1` mỗi người (8,33%); 18 khách còn lại 0%.
  Tỷ lệ gộp là 2/240 khách–tháng (0,8333%), không phải tỷ lệ trên số giao dịch.
  CSV đủ ID, từng nhãn và loại giao dịch nằm ở
  `outputs/tail_review_20261109/top20_trx_customers.csv` sau khi chạy script.
- **Đỉnh 11/12/2022 UTC:** 65.302 log từ 32.664 khách; top 1 chiếm 1,83%,
  top 20 chiếm 4,22%. Đây là đỉnh có nhiều khách tham gia; chưa xác minh nguyên
  nhân lịch. Giữ dữ liệu và kiểm định đặc trưng lịch thay vì xóa ngày này.
- **Count:** giữ nhóm geo/dialog bị IQR đánh dấu; thêm log1p count, has_dialog,
  active days và cờ heavy/ngoại tệ/thiếu currency khi tạo features. P99 count toàn
  kỳ là trx 2.233,54; geo 5.082,54; dialog 21. Có 892 khách dialog >P99 do ties,
  không mặc định mọi nguồn đều có 1.003 khách. MBD là tệp khách doanh nghiệp theo
  [bài báo nguồn](https://arxiv.org/html/2409.17587v2), chưa có căn cứ loại heavy
  users chỉ vì tần suất lớn hoặc không có nhãn dương.

Giữ raw và chưa xóa dòng/khách nào. Log/cap/rank hoặc ngưỡng heavy dùng cho mô hình
phải fit trên train trước cutoff; chưa huấn luyện để chọn phương án. Cây ít phụ
thuộc thang đo nhưng không bảo đảm miễn nhiễm với cực trị, nhất là khi sum/mean đã
bị chi phối trước khi vào mô hình. Chạy lại bằng
`.\.venv\Scripts\python.exe -B -X utf8 src/audit_tail_cases.py`.

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
├── src/              # Package propensity_data, mbd_mini_eda và script audit
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
