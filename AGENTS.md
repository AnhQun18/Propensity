# Quy tắc Codex cho dự án propensity

## Quy tắc phạm vi EDA — cập nhật 18/11/2026

1. EDA phải kiểm tra toàn bộ khách hàng, không tự lấy mẫu. Nếu chia nhóm, nhóm phải được định nghĩa loại trừ nhau, đối chiếu giữa các nhóm và kiểm tra tổng nhóm đủ toàn bộ khách của từng kỳ. Không loại âm thầm khách thiếu log hoặc thiếu khả năng quan sát.
2. Khi phân tích sự kiện quanh nhãn mua, phải xét cả hai chiều: sự kiện sau tháng đã mua và sự kiện trước tháng sẽ mua. Thứ tự thời gian chỉ là bằng chứng mô tả; không tự kết luận hệ quả hoặc chuẩn bị mua theo nghĩa nhân quả.
3. Mọi phân tích thí điểm phải kiểm kê ít nhất toàn bộ một năm nhãn; có thể mở rộng toàn bộ thời gian log hợp lệ. Giữ các tháng biên thiếu nhãn trước/sau hoặc chưa có log tương lai thành trạng thái chưa đủ quan sát, không gán 0 và không cắt thành một giai đoạn ngắn rồi khái quát toàn bộ.
4. Các phân tích hồi cứu dùng mẫu khách hoặc chỉ một số tháng trước quy tắc này là kết quả phương pháp cũ, không coi là đã đáp ứng phạm vi mới.

## Quy ước nghiên cứu và baseline

Áp dụng cho toàn bộ baseline, nghiên cứu và các lần cập nhật mô hình trong dự án. Trước khi xây dựng hoặc cập nhật mô hình, phải kiểm tra và tuân thủ các quy ước sau:

1. **Dùng toàn bộ lịch sử có sẵn trước cutoff** của từng quan sát khách–tháng để tạo đặc trưng. Không giới hạn lịch sử đầu vào vào một cửa sổ cố định như 30, 84 hoặc 90 ngày trước cutoff. Lịch sử bắt đầu từ thời điểm sớm nhất có dữ liệu của nguồn tương ứng.
2. **Giữ nguyên nguyên tắc chống rò rỉ thời gian:** chỉ sử dụng sự kiện có `event_time < cutoff`. “Toàn bộ lịch sử” không bao gồm sự kiện tại hoặc sau cutoff, dữ liệu tương lai hay nhãn chưa có tại thời điểm huấn luyện. Các phép fit tiền xử lý, ngưỡng và mô hình vẫn phải tuân thủ phân chia thời gian.
3. **Nghiên cứu và so sánh đầy đủ horizon 1, 2 và 3 tháng.** Với mỗi horizon H, nhãn tích lũy là có mua ít nhất một trong bốn sản phẩm trong H tháng liên tiếp kể từ cutoff: `y_H = OR(y_any của các tháng trong horizon)`. Horizon là khoảng thời gian tương lai cần dự báo, độc lập với độ dài lịch sử tạo đặc trưng.
4. **So sánh horizon trên cùng cohort và cutoff có đủ nhãn tương lai cho cả 3 tháng**, cùng ngân sách chọn khách/top-K. Không gán tháng chưa quan sát được thành nhãn 0. Đánh giá precision@K, recall@K, lift@K và tỷ lệ mua nền của từng horizon; không chọn horizon chỉ vì số người mua tích lũy tăng khi kéo dài thời gian.
5. **Không tự chốt horizon 1 tháng hoặc một cửa sổ lịch sử cố định làm mặc định nghiên cứu.** Báo cáo kết quả so sánh 1/2/3 tháng trước khi đề xuất horizon cho mô hình cập nhật.

Các quy ước trên thay thế lựa chọn trước đây sử dụng cửa sổ lịch sử cố định hoặc chỉ đánh giá horizon 1 tháng. Artifact cũ có thể giữ để đối chiếu nhưng phải ghi rõ phương pháp cũ; không trình bày chúng như đã tuân thủ quy ước mới khi chưa chạy lại.

## Quy ước múi giờ — cập nhật ngày 17/11/2026

1. **Dùng giờ Moscow (`Europe/Moscow`, UTC+3)** làm múi giờ nghiên cứu cho mọi nguồn, cutoff, tháng nhãn và đặc trưng lịch (giờ, thứ, ngày, tháng). Quy tắc này thay thế quy ước UTC trước đây.
2. **Cutoff là 00:00 Moscow ngày đầu tháng nhãn.** Ví dụ `mon=2022-02-28` có cutoff `2022-02-01 00:00:00+03:00`, tương đương `2022-01-31 21:00:00Z`. Không dịch `mon` sang tháng khác; `mon` là chỉ báo tháng nhãn, không phải timestamp sự kiện cần chuyển múi giờ.
3. **Chuyển múi giờ phải giữ nguyên thời điểm thực.** Timestamp sự kiện đang được chuẩn hóa theo UTC phải được chuyển sang Moscow để lấy các thuộc tính lịch; không gán lại timezone Moscow cho cùng giá trị giờ UTC, không cộng ba giờ hai lần. Có thể lưu và so sánh bằng UTC nếu cutoff đã quy đổi đúng. Không sửa raw. Timestamp thiếu timezone phải có giả định nguồn được ghi rõ; việc chọn Moscow cho nghiên cứu không tự xác nhận timezone gốc của ETL.
4. **Chỉ dùng `event_time < cutoff` trên cùng trục thời gian.** Cửa sổ W ngày là `[cutoff - W ngày, cutoff)`, neo tại cutoff Moscow; số ngày hoạt động và thống kê giờ/thứ/ngày/tháng được tính theo lịch Moscow. Nhãn horizon 1/2/3 tháng là các tháng lịch Moscow liên tiếp từ cutoff.
5. **Artifact dùng UTC trước đây là kết quả theo quy ước cũ.** Phải ghi rõ timezone và tính lại các bảng/biểu đồ/feature chịu ảnh hưởng trước khi trình bày là kết quả theo Moscow. Đổi nhãn trên biểu đồ hoặc trường cấu hình không thay thế việc chuyển đổi và chạy lại. Mã nguồn/cấu hình cũ còn ghi UTC chưa được coi là đã tuân thủ quy tắc này.
