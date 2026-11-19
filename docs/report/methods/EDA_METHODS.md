# Phương pháp và giới hạn

- Mẫu số gốc: 100.224 khách, 12 tháng nhãn, 1.202.688 khách–tháng.
- T là tháng lịch chứa mon, theo quy ước EDA đã đối chiếu trong dự án. UTC là quy ước nghiên cứu; timezone ETL nhãn gốc chưa được tác giả xác nhận.
- Cửa sổ chính: [đầu T−1, đầu T). Lịch sử tích lũy dùng event_time < đầu T, và được xuất riêng.
- Số sự kiện bằng 0 nghĩa không tìm thấy log; recency/gap/spatial có thể unknown. Ngày đầu quan sát không chứng minh thu thập liên tục.
- y_any: 1 nếu bất kỳ nhãn nào bằng 1; 0 nếu cả bốn bằng 0; còn lại null. Bản dữ liệu hiện tại không có nhãn sản phẩm null.
- Wilson 95% được xuất cho tỷ lệ. Khi gộp khách lặp qua các tháng, có thêm CI bootstrap theo khách.
- Lift và ratio tái sử dụng cluster bootstrap 2.000 lượt, seed 42, cùng baseline và lượt lấy mẫu. CI không ổn định được giữ trạng thái, không tạo CI giả.
- TRX category lift giới hạn ở top 5 event_type chọn theo tần suất toàn kỳ, trước khi xem nhãn; bảng category summary giữ các loại khác.
- Gap là giữa hai sự kiện liên tiếp cùng khách và cùng tháng trong 12 tháng quan sát T−1. Không nối qua tháng.
- Monetary chỉ cộng currency=11. Không cộng amount giữa các currency.
- PCA là phép chiếu mô tả trên mẫu hash tối đa 10.000 vector, seed 42. Không có raw text nên không đặt tên chủ đề hoặc gọi trục PCA là ý nghĩa nghiệp vụ.
- GEO integer encoding chưa xác minh. Đã kiểm tra dtype, missing, cardinality và xung đột cha–con; không decode tọa độ hoặc tính khoảng cách.
- Không huấn luyện hay đánh giá mô hình. Tất cả hình là biểu đồ do code tính từ dữ liệu; không có ảnh AI hoặc ảnh chụp bản trình bày.
