# EDA MBD-mini - Báo cáo tóm tắt

Đính chính log-IQR ngày 11/11/2026 theo rà soát 09/11/2026; notebook đã được thực thi lại.

## 1. EDA MBD-mini

Mục tiêu là dự đoán khách mua ít nhất một trong bốn sản phẩm trong tháng đích. Bài trình bày tóm tắt bảy bước EDA, tập trung vào các phát hiện có ý nghĩa cho xử lý dữ liệu và thiết kế đặc trưng. Nguồn: overview.csv, manifest.json và notebook eda.ipynb.

Dữ liệu MBD-mini. Kết quả EDA cập nhật ngày 11/11/2026.

## 2. Phạm vi và cấu trúc dữ liệu

Raw được thống kê toàn lịch sử ở cả tám bảng. Chỉ dùng giai đoạn train 02-08/2022 cho liên hệ giữa đặc trưng và nhãn. Overview và tỷ lệ nhãn theo tháng được trình bày đủ 12 tháng. Các mốc cutoff được diễn giải theo quy ước UTC của dự án.

Nguồn: overview.csv. Không cộng detail và ptls vào cùng tổng log.

## 3. Tỷ lệ mua chỉ 1,0688%

Nhãn y_any là 1 khi có ít nhất một nhãn sản phẩm dương. Toàn bộ 1.202.688 dòng hiện có nhãn và có 12.854 dòng dương. Tỷ lệ thấp nhất 0,673% và cao nhất 1,694%. Đây là thống kê mô tả của dữ liệu, chưa phải kết quả dự đoán.

Nguồn: target_monthly.csv. Tỷ lệ theo tháng trên toàn cohort.

## 4. Thiếu nguồn cần được giữ đúng nghĩa

Recency chỉ xác định khi có sự kiện nguồn phù hợp trước cutoff. Thiếu recency không đồng nghĩa mất nhãn hay lỗi thu thập đã được xác minh. Trường phụ trợ diff_trans_date thiếu 120.786 dòng (10,043%), tất cả cùng trans_count=0. Imputation nếu cần phải fit trên train.

Nguồn: feature_statistics_train.csv. Toàn bộ train 02-08/2022.

## 5. Không phát hiện dòng trùng hoàn toàn

Full-row duplicate được quét chính xác, không dùng hash gần đúng để kết luận. Cùng khách và timestamp chỉ là ứng viên kiểm tra. PTLS/trx có 36.744.910 sự kiện so với 38.764.930 ở detail/trx. Không cộng hai biểu diễn và không tự động loại các giao dịch đồng thời.

Nguồn: duplicates.csv và detail_sequence_reconciliation.csv. Quét toàn bộ raw.

## 6. Ngoại lai cần tính trên thang log

Đính chính ngày 11/11/2026 theo rà soát log-IQR ngày 09/11/2026. Biến đổi từng giá trị DOUBLE sang log1p rồi mới tính quartile và hàng rào 1,5*IQR. Với mã 11: 584.139 dòng ngoài hàng rào, gồm 486.868 thấp và 97.271 cao. ln(amount) là phân tích độ nhạy, raw chỉ đối chiếu. Không kết luận ngoại lai cho nhóm dưới 30 giá trị, currency thiếu hoặc IQR log bằng 0. Ngưỡng toàn kỳ chỉ phục vụ EDA, không dùng nguyên cho mô hình quá khứ. Gắn cờ chưa chứng minh lỗi.

Nguồn: amount_outliers_by_currency.csv. Tính log trước quartile, theo từng currency.

## 7. Độ đa dạng giao dịch liên hệ với nhãn

Các biến trong biểu đồ lần lượt là trx_30d_event_subtype_nunique, trx_30d_event_type_nunique và trx_30d_dst_type12_nunique. Tương quan không phải nhân quả hoặc feature importance của mô hình. Ma trận nhiều biến dùng pairwise-complete, cỡ mẫu từng cặp được giữ trong CSV. Nguồn: feature_target_associations.csv, high_correlation_pairs.csv.

Pearson với target: toàn train. Ma trận nhiều biến: mẫu hash 30.000 dòng, seed 42.

## 8. Ưu tiên Feature Engineering

Hai trường phụ trợ nhãn bị loại là trans_count và diff_trans_date do chưa xác minh công thức và cutoff. Không đưa client_id hoặc fold vào X. Đề xuất mới có thể gồm recency, frequency, active-day rate, tỷ trọng loại giao dịch và xu hướng giữa cửa sổ 30/84 ngày. Chỉ chốt biến sau đối chứng theo thời gian.

Nguồn: feature_engineering_plan.csv. Mọi sự kiện tạo feature phải trước cutoff.

## 9. Bước tiếp theo

Thiết kế chia thời gian này dành cho giai đoạn đánh giá tiếp theo. EDA chưa huấn luyện hoặc mở kết quả dự đoán test để chọn biến. Khách có thể xuất hiện nhiều tháng nên khi ước lượng bất định cần tôn trọng nhóm khách. Dữ liệu không cung cấp hiệu quả chiến dịch hay ROI. Toàn bộ bảng và định nghĩa ở deliverables/eda.

Nguồn: feature_engineering_plan.csv. Bài trình bày này báo cáo EDA, không báo cáo mô hình.
