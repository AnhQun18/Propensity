# Kiểm kê EDA sau rà soát 12 yêu cầu — 06/11/2026

Kết quả hiện tại gồm **61 bảng CSV, 24 hình PNG**, report, danh mục bảng/biến,
metadata và manifest. Có thêm mẫu liên kết **1.000 khách** ở
[sampledata](../sampledata/README.md), giữ mọi sự kiện detail và nhãn của khách
được chọn. [Báo cáo đầy đủ](mbd_mini_eda/report.md) trình bày cả 12 yêu cầu;
[kết quả xác nhận](mbd_mini_eda_validation.json) ghi **225 kiểm tra đạt**.

## Đối chiếu yêu cầu

| Yêu cầu | Đầu ra và kết quả |
| --- | --- |
| Độ phủ theo nhóm có chủ đích | 5 ma trận khách–tháng: nhiều nhất, gần mean, gần median, ít nhất có log, 0 log; thêm ma trận theo nhóm phân vị trên tất cả khách |
| Ý nghĩa và nguyên nhân null | Null cột trên 8 bảng, null phần tử ptls/embedding, cặp thuộc tính, bảng chéo theo trans_count/event_type; toàn bộ 120.786 null diff_trans_date đi cùng trans_count=0 |
| Duplicate | Trùng khóa và nguyên dòng toàn bộ xuyên shard/batch; 0 trùng nguyên dòng, 1 ứng viên giao dịch cùng khách/timestamp khác nội dung |
| Dùng toàn thời gian | 105.277.859 log detail từ 31/12/2020 đến 31/12/2022; giữ khách thiếu nguồn; geo/dialog không có log trước 01/2022 đặt null trong lưới tháng |
| Tỷ lệ hành vi dạng matrix | Bảng từng nguồn, 8 tổ hợp nguồn và ma trận tỷ lệ theo tháng trên cùng mẫu số |
| Nhịp hành vi không log | Theo ngày và tháng, 3 trục tuyến tính riêng theo nguồn, null ngoài khoảng quan sát |
| Amount gốc | Bin giá trị gốc bao phủ toàn đuôi; zoom tuyến tính 0–P99 và ghi số quan sát ngoài zoom; phân vị gốc, không log |
| Biến động target hiếm | 4 panel theo tháng, zoom Y riêng, đường tỷ lệ toàn kỳ, số dương thực tế ở từng điểm |
| Resample và sampledata | Phân tầng fold × tổ hợp nguồn × band tổng log, Hamilton, seed 42; đối chiếu KS, mean/median/P95/zero và nhãn |
| Nội dung và biến từng bảng | source_schema.csv liệt kê 8 bảng raw; data_catalog.md mô tả mọi CSV và các cột |
| Mean/min/max mỗi người | Toàn lịch sử, tách dân số 100.224 khách và nhóm có log; có tứ phân vị/P95/P99 |
| Outlier | Tukey 1,5 IQR theo currency, số log mỗi khách và số log mỗi ngày; giữ nguyên dữ liệu, không coi cờ là lỗi đã xác nhận |

## Đối soát dữ liệu

| Nguồn chuẩn | Số dòng | Khách có log | Trùng nguyên dòng |
| --- | ---: | ---: | ---: |
| client_split | 100.224 | — | 0 |
| targets | 1.202.688 | 100.224 | 0 |
| detail/trx | 38.764.930 | 98.721 | 0 |
| detail/geo | 66.295.724 | 72.573 | 0 |
| detail/dialog | 217.205 | 46.320 | 0 |

Giữ **468 khách không có log**. Số khách có 0/1/2/3 nguồn vẫn là
468 / 17.322 / 47.010 / 35.424. Các tổng theo ngày/tháng và moment mỗi khách
khớp số log raw. Tổng nhãn theo tháng khớp toàn kỳ: 5.362 / 577 / 4.530 /
3.060 mẫu dương của target_1/2/3/4. Không thiếu bốn nhãn.

Đối chiếu ptls phát hiện **ptls/trx chỉ có 36.744.910 phần tử**, ít hơn detail
**2.020.020** ở **13 khách**; chiều dài chuỗi tối đa 10.000. Geo/dialog có
tổng và số phần tử theo khách khớp detail. Vì vậy, không dùng ptls/trx để thay
thế lịch sử giao dịch đầy đủ. Không có null phần tử ptls cũng không chứng
minh detail/trx không thiếu mã thuộc tính.

## Kiểm chứng và cache

Đã chạy **28 kiểm thử hồi quy đạt** với Parquet tổng hợp: xuyên batch/shard,
timestamp ns, null/NaN trong vector, trùng thật và ứng viên cùng timestamp,
phân vị, histogram âm/đuôi lớn, seed, phân bổ mẫu, liên kết bảng, null khung
sớm và cache thay đổi câu SQL/raw. Bộ xác nhận export đạt 225 kiểm tra về
manifest, file, tổng đếm, tỷ lệ, bảng mẫu và lịch sử đầy đủ. Các hình mới đã
được mở kiểm tra trực quan.

Cache cũ `outputs/behavior_coverage/matrix.npz` giữ nguyên; tổng log từng ô
khớp chính xác với tổng hợp mới. Cache SQL mới nằm ở
`outputs/mbd_review_cache/review.duckdb`, cùng inventory và ma trận từng nguồn
có mask null. Lần chạy lại cuối dùng lại toàn bộ truy vấn đã lưu; raw vẫn
cần tồn tại để kiểm kê shard và xuất mẫu. Cache được giữ cục bộ, không push.

Tái lập: [mbd_eda_review.md](mbd_eda_review.md). Bản export cơ bản cũ và cache
quét năm 2022 vẫn giữ trên máy; các hình log, biểu đồ tỷ lệ dạng cột và ZIP
trùng nguồn được bỏ khỏi bản cuối.

## Giới hạn diễn giải

Mã thuộc tính, đơn vị Amount, công thức trường phụ trợ và timezone chưa có
định nghĩa nghiệp vụ đầy đủ. Quan hệ null với trans_count hay đồng thiếu
theo cặp là bằng chứng thực nghiệm, không xác nhận nguyên nhân thu thập.
Outlier có thể là hành vi hợp lệ; các currency ít quan sát cho ngưỡng IQR
không ổn định. Mẫu 1.000 khách không bảo toàn chính xác cực trị/nhãn hiếm;
thống kê chính tiếp tục dùng toàn bộ raw. Tầng mẫu toàn lịch sử chỉ dùng EDA,
không dùng để chia tập mô hình tại cutoff quá khứ.
