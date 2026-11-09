# Xác minh trans_count, quy ước currency, thiếu nguồn và độ trễ nhãn

Ngày kiểm chứng: **09/11/2026**. Quy ước dự án hiện tại: UTC, cutoff đầu tháng lúc
00:00 UTC, một target any-product. Tên thương mại bốn sản phẩm nằm ngoài yêu cầu
xác minh của đề tài. Không sửa raw hoặc huấn luyện mô hình trong đợt kiểm tra này.

## 1. Tự xác minh trans_count

Đọc lại toàn bộ 1.202.688 dòng targets; đối chiếu ma trận tháng được tổng hợp từ
38.764.930 trx, 66.295.724 geo và 217.205 dialog. Trước khi tái dùng cache, kiểm tra
toàn bộ 3.820 đường dẫn/kích thước/mtime_ns trong inventory vẫn khớp và fingerprint
ma trận đúng inventory. Đây không phải checksum toàn bộ byte raw.

### Quan sát trực tiếp trên cột nguồn

| Kiểm tra | Kết quả |
| --- | ---: |
| Min / max | 0 / 1.000 |
| Giá trị phân biệt | 1.001, có đủ các số nguyên 0–1.000 |
| Median / mean | 33 / 96,08587 |
| Dòng bằng 0 | 120.786 |
| Dòng bằng 1.000 | 11 |
| Cặp tháng liên tiếp cùng khách | 1.102.464 |
| Cặp có trans_count giảm | **0** |
| Cặp có trans_count tăng | 856.549 |

Tính không giảm phù hợp với một đại lượng tích lũy, nhưng không tự xác nhận đếm loại
giao dịch nào, từ ngày nào hoặc có biến đổi gì. Chỉ có 11 dòng đạt 1.000 nên không
được lấy max=1.000 làm bằng chứng chắc chắn đã cap ở 1.000.

### Đối chiếu các công thức đếm

Thử 612 cấu hình: nguồn trx/geo/dialog; biên trước tháng nhãn, lùi một tháng và sau
tháng nhãn; toàn quá khứ, 1/2/3/6/12/18/24 tháng lịch, từng mốc bắt đầu cố định trong
25 tháng nguồn, year-to-date; giữ số đếm hoặc cap 1.000. Các mốc sau cutoff chỉ là
phép chẩn đoán, không dùng làm feature dự đoán.

**Không cấu hình nào khớp toàn bộ cột nguồn.** Với trx trước cutoff UTC, cấu hình khớp
nhiều nhất là số sự kiện từ 01/01/2022 đến trước cutoff, cap 1.000:

- Khớp 221.448/1.202.688 dòng = **18,4128%**.
- Khi chỉ xét 1.081.902 dòng trans_count nguồn > 0, khớp 116.242 dòng = **10,7442%**.
- Các trường hợp cùng bằng 0 góp vào tỷ lệ tổng, không phải bằng chứng về công thức.

Hai phản ví dụ mạnh hơn việc chọn cửa sổ:

- **102.932 dòng** có trans_count nguồn lớn hơn toàn bộ số trx được phát hành trước
  cutoff. Vì vậy không thể giải thích các dòng đó bằng số sự kiện của bất kỳ tập con
  hay cửa sổ quá khứ nào chỉ lấy từ detail/trx được cung cấp.
- Có **1.503 khách không có bất kỳ detail/trx nào trong toàn lịch sử**, nhưng các
  khách này vẫn có **9.459 dòng khách–tháng trans_count > 0**. Đổi cutoff không tạo
  ra được những giao dịch không có trong log được phát hành.

**Kết luận xác minh:** chưa tái tạo được công thức gốc; đã xác minh nó không thể được
coi đơn giản là count từ detail/trx hiện có. Có thể khác phạm vi giao dịch, phạm vi
lịch sử hoặc phép biến đổi, nhưng chưa có chứng cứ để chọn một nguyên nhân. Không
gắn sai khác này thành lỗi nguồn hoặc bằng chứng chắc chắn dùng tương lai.

**Cách dùng cho đề tài:** giữ cột nguồn để audit; chưa dùng trực tiếp làm predictor.
Tự tạo `observed_trx_count_all_history` và `observed_trx_count_30d/84d/180d` bằng
`count(*)` trên detail/trx có ID/timestamp hợp lệ và `event_time < cutoff`. Các feature
mới đo log quan sát được theo định nghĩa dự án, không tuyên bố tái tạo trans_count gốc.

## 2. Mã 11 là mã chính, mã khác flag ngoại tệ

Đã thực hiện đúng quyết định người dùng bằng hàm
[currency_flags](../../../src/mbd_mini_eda/currency.py), giữ nguyên currency nguồn:

| Currency | currency_is_main | currency_is_foreign | currency_missing |
| --- | --- | --- | --- |
| 11 | True | False | False |
| Mã khác 11 có giá trị | False | True | False |
| Null | null | null | True |

"Ngoại tệ" là tên cờ theo quy ước dự án cho mã khác 11; không gán ISO hoặc đồng tiền
thực cho các mã ẩn danh. Không phân loại currency null thành ngoại tệ.

Đọc currency trên toàn bộ 400 shard trx xác nhận:

| Nhóm | Giao dịch | Tỷ lệ trên toàn bộ trx |
| --- | ---: | ---: |
| Mã chính 11 | **38.724.240** | 99,895034% |
| Ngoại tệ theo cờ dự án | **40.572** | 0,104662% |
| Currency thiếu | **118** | 0,000304% |

Giữ mã gốc để phân tích riêng khi cần. Có thể tổng hợp count/tỷ lệ giao dịch ngoại tệ;
cờ nhóm không tự biến amount của các mã khác nhau thành cùng đơn vị tiền. Mã 11 là
nhóm chính của dự án, chưa gán tên tiền hoặc đơn vị tiền thật.

## 3. Tự xác minh nguyên nhân thiếu nguồn

### Cơ chế chung đã được nhà phát hành mô tả

[Bài MBD v2, mục 3.1](https://arxiv.org/html/2409.17587v2#S3.SS1) giải thích nguồn vị
trí là dấu vết khi khách sử dụng ứng dụng ngân hàng, còn hội thoại là tương tác với
hỗ trợ kỹ thuật. Nhiều khách không có đủ ba nguồn vì họ có thể không giao dịch, không
gọi hỗ trợ hoặc không để lại dấu vết vị trí khi dùng ứng dụng.

Như vậy **vắng nguồn có thể là đặc tính của hành vi/kênh tương tác trong dữ liệu**;
không mặc định mọi khách thiếu geo/dialog đều bị lỗi ETL. Đây là mô tả cơ chế chung
của tác giả, không phải xác nhận nguyên nhân cho từng khách cụ thể.

### Những gì xác minh được trên bản dữ liệu địa phương

| Nguồn | Có log toàn lịch sử | Không có log toàn lịch sử |
| --- | ---: | ---: |
| trx | 98.721 | 1.503 |
| geo | 72.573 | 27.651 |
| dialog | 46.320 | 53.904 |

Phân loại đủ mọi ô khách–tháng, tách sáu trạng thái không chồng lấn:

1. Có sự kiện quan sát được.
2. Tháng ngoài khoảng tháng đầu/cuối được phát hành của nguồn.
3. Khách không từng xuất hiện trong nguồn, xét trong khoảng nguồn có dữ liệu.
4. Tháng trống trước tháng có log đầu của khách, nhưng vẫn trong khoảng nguồn.
5. Tháng trống xen giữa tháng đầu và cuối có log của khách.
6. Tháng trống sau tháng có log cuối của khách, nhưng vẫn trong khoảng nguồn.

Ví dụ số ô trống xen giữa tháng đầu/cuối có log: trx **182.757**, geo **51.018**,
dialog **204.061**. Đây là độ thưa theo tháng, không phải số ô đã chứng minh mất log.
Biên nguồn tính theo tháng, không hàm ý tháng biên được quan sát trọn vẹn.

Inventory schema không có cờ theo dõi thu thập/quan sát liên tục để phân biệt chính
xác không hoạt động, chưa thu thập hoặc thất thoát ở từng khách/tháng. Không tự gán
nguyên nhân, MCAR/MAR/MNAR, ngày mở tài khoản hoặc ngày bật tracking.

**Quyết định áp dụng:** giữ khách thiếu nguồn, dùng masks/count/recency trước cutoff;
null ngoài phạm vi trích xuất là trạng thái chưa quan sát trong nghiên cứu. Trong
phạm vi có dữ liệu, count=0 nghĩa không có log được ghi nhận; mean/recency/embedding
không xác định giữ null. Không xóa khách chỉ vì không có geo/dialog. Các trạng thái
dựa trên tháng cuối/toàn lịch sử ở audit chỉ để mô tả, không làm feature cho cutoff
quá khứ vì có thể nhìn vào tương lai.

## 4. Tự xác minh thời điểm nhãn sẵn có

### Bằng chứng đã kiểm tra

- Đọc schema và metadata của đủ **435 shard targets**: chỉ có client_id, mon, bốn
  target, trans_count, diff_trans_date; fold nằm ở partition. Không có `available_at`,
  `ingested_at`, `published_at`, `finalized_at` hay mốc tương đương trong các cột/keys
  metadata đã kiểm tra. Metadata chỉ ghi phiên bản/schema Spark.
- Dataset card, paper MBD và mã ghép targets công khai mô tả nhãn theo tháng nhưng
  không ghi lịch công bố/chốt nhãn từng tháng. `mon` là mốc tháng của nhãn, không phải
  timestamp chứng minh nhãn đã sẵn sàng để fit mô hình.
- [Commit upload của nhà phát hành](https://huggingface.co/datasets/ai-lab/MBD-mini/commit/040c4a3ce3f793d3189a101962604f158b1c80d7)
  ghi nhận targets.tar.gz được đưa lên Hugging Face ngày **27/08/2024**. Điều này chỉ
  xác nhận thời điểm công bố snapshot công khai; không xác định nhãn tháng 2022 đã
  sẵn có trong hệ thống ngân hàng vào ngày nào. Mtime file cũng không giải quyết được.

**Kết luận:** không xác định được độ trễ công bố nhãn thực tế từ bản snapshot và
nguồn công khai đã kiểm tra. Không đủ bằng chứng để điền label_available_at thực tế
bằng mon, ngày đầu tháng sau hoặc một độ trễ tự chọn rồi gọi đó là kết quả xác minh.

### Kiểm tra tác động lên lịch đánh giá hiện có

Đã dựng đủ 60 dòng kịch bản: 12 tháng × độ trễ giả định 0/1/7/30 ngày hoặc một tháng
lịch, tính từ cuối khoảng nhãn UTC (đầu tháng kế tiếp). Đây là kiểm tra lịch khả thi,
chưa huấn luyện mô hình và chưa đo độ nhạy metric.

Với kế hoạch hiệu chỉnh bằng nhãn **12/2022** trước test lúc **01/01/2023 00:00 UTC**:

| Độ trễ giả định sau khi kết thúc tháng nhãn | Nhãn 12/2022 sẵn có trước hoặc tại test? |
| --- | --- |
| 0 ngày | Có, đúng tại cutoff |
| 1 ngày | Không |
| 7 ngày | Không |
| 30 ngày | Không |
| Một tháng lịch | Không |

Vì vậy lịch cũ chỉ có thể được mô tả là **mô phỏng offline với giả định nhãn sẵn ngay
khi tháng kết thúc**, chưa được xác minh phản ánh vận hành thực tế. Nếu giả định độ
trễ dương, phải chuyển tháng calibration và giới hạn nhãn train/tuning theo ngày
sẵn có giả định; không chỉ đổi một ngày ở test rồi giữ nguyên lịch fit.

Một ví dụ nhất quán với **giả định trễ một tháng lịch**: chọn cấu hình bằng validation
07–09/2022 (mỗi lần chỉ fit nhãn đã sẵn theo giả định); tại 01/11 fit mô hình cơ sở
trên 02–09, dự đoán tháng 11, nhãn tháng 11 sẵn ngày 01/01/2023 để fit calibrator,
sau đó test tháng 01 bằng cùng mô hình cơ sở. Không refit cơ sở bằng nhãn mới rồi tái
dùng calibrator cũ. Ví dụ này vẫn là giả định lịch, không phải SLA được nhà phát hành
xác nhận hoặc kết quả mô hình đã chạy; các bước fit đúng tại cutoff cần ghi rõ thứ tự.

**Hướng tiếp tục nghiên cứu:** giữ giả định độ trễ 0 khi trình bày protocol offline
hiện có, ghi rõ điều kiện; báo cáo thêm các kịch bản trễ dương với lịch fit/calibration
được dựng lại trước khi kết luận về mô phỏng triển khai. Giữ độ trễ thực tế chưa xác
định như một giới hạn bằng chứng của bản snapshot.

## Tái lập

```powershell
.\.venv\Scripts\python.exe -B -X utf8 src/audit_remaining_questions.py
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -p test_currency_flags.py -v
```

- [Script audit](../../../src/audit_remaining_questions.py).
- [verification.json](verification.json): kết quả chính, schema/metadata và kiểm tra lịch.
- [trans_count_candidates.csv](trans_count_candidates.csv): đủ 612 cấu hình và tỷ lệ
  trên toàn bộ dòng / riêng nhóm trans_count dương.
- [absence_patterns.csv](absence_patterns.csv), [source_coverage.csv](source_coverage.csv):
  độ phủ và trạng thái ô khách–tháng; không phải nhãn nguyên nhân thiếu.
- [currency_flags_by_code.csv](currency_flags_by_code.csv): đối soát mọi mã, cờ và số dòng.
- [label_readiness_scenarios.csv](label_readiness_scenarios.csv): chỉ chứa ngày sẵn có
  theo giả định, không phải lịch công bố nhãn thực tế.
