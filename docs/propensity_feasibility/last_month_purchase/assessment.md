# Tỷ lệ mua sau cutoff của tháng nhãn cuối

Ngày kiểm tra: 05/11/2026. Đọc toàn bộ client_id, mon và bốn target trong bảng targets; lọc tháng nhãn lớn nhất. Không lọc khách theo nguồn sự kiện hoặc độ dài lịch sử.

Cutoff: **2023-01-01 00:00:00**. Khoảng nhãn: **[2023-01-01 00:00:00, 2023-02-01 00:00:00)**. Marker: `mon=2023-01-31`. Áp dụng [quy ước mon đã nghiên cứu](../mon_research.md); timezone timestamp nguồn chưa được xác nhận.

Mẫu số: **100,224 khách hàng**, mỗi khách một dòng trong tháng. Không trùng ID và không có nhãn null hoặc ngoài 0/1.

| Nhãn | Khách mua | Khách không mua | Tỷ lệ mua |
|---|---:|---:|---:|
| target_1 | 159 | 100,065 | 0.1586% |
| target_2 | 25 | 100,199 | 0.0249% |
| target_3 | 215 | 100,009 | 0.2145% |
| target_4 | 293 | 99,931 | 0.2923% |
| any_product | 675 | 99,549 | 0.6735% |

Có 675 khách mua ít nhất một sản phẩm; 659 khách mua đúng một sản phẩm, 15 khách mua hai sản phẩm và 1 khách mua ba sản phẩm. Tổng nhãn dương là 692, không phải số khách mua duy nhất.

Nếu giữ tháng này làm test theo thời gian, các tháng nhãn trước có 1.102.464 dòng trước khi chia validation hoặc lọc cohort. Lịch sử test chỉ lấy sự kiện trước cutoff; tiền xử lý và mô hình phải được fit trên phần train phù hợp theo thời gian.

target_2 chỉ có 25 khách dương trong tháng cuối, nên việc đánh giá riêng nhãn này có độ bất định lớn. Báo cáo PR-AUC/AP, ROC-AUC và precision/recall tại mức chọn khách; không dùng accuracy làm tiêu chí chính. Các tỷ lệ trên cần tính lại nếu thay đổi cohort.
