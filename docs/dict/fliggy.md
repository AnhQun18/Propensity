# Fliggy Transfer-Plans — từ điển dữ liệu và bài toán dự đoán

## 1. Tổng quan

**Bài toán.** Dự đoán xác suất một người dùng phát sinh ít nhất một log
tương tác trong 7 ngày tiếp theo, dựa trên lịch sử tương tác với các phương
án di chuyển/nối chuyến trước thời điểm dự đoán (cutoff). Bộ dữ liệu gốc
phục vụ gợi ý phương án di chuyển; đây là cách đặt bài toán propensity
hoạt động trong dự án.

**Đơn vị mẫu.** Mỗi mẫu của mô hình là một cặp *người dùng–cutoff*. Điều này
khác với dữ liệu gốc, nơi mỗi dòng log là một sự kiện người dùng–truy vấn–phương án.

**Nhãn.** Dữ liệu gốc không có sẵn nhãn propensity, nên cần tự tạo
`y_activity_7d` cho mỗi người dùng tại cutoff:

- `y_activity_7d = 1` nếu có ít nhất một log trong khoảng
  `[cutoff, cutoff + 7 ngày)`, tính cả ba mã hành vi 1/2/3.
- `y_activity_7d = 0` nếu không có log nào trong khoảng đó.
- Chỉ xét những người dùng đủ điều kiện, xác định từ lịch sử trước cutoff;
  không dùng việc có log tương lai để chọn người vào nhóm dự đoán.

**Đầu ra.** Mô hình trả về `p_activity_7d ∈ [0, 1]` cho mỗi người dùng tại
cutoff.

**Lưu ý về ý nghĩa của nhãn.**

- Nhãn 0 chỉ có nghĩa là *không ghi nhận tương tác trong khoảng nhãn*,
  không chứng minh người dùng không đi du lịch hoặc được quan sát đầy đủ.
- Chưa xác minh mã `behavior=1/2/3` tương ứng hành vi nào, nên chưa thể gọi
  đầu ra là xác suất mua hoặc đặt dịch vụ. Cần bảng quy đổi trước khi đặt
  nhãn nghiệp vụ cụ thể.

## 2. Quy mô

Bản hiện có tại `data/Fliggy/` gồm ba file CSV không có header.

| File/thành phần | Quy mô | Đơn vị |
|---|---:|---|
| `user_behavior_transfer_plans.csv` | **42.433.919** | Sự kiện |
| Người dùng duy nhất trong log | **2.524.995** | Người dùng |
| `plan_id` duy nhất trong log | 12.172.885 | ID phương án |
| `user_profile.csv` | 2.522.986 | Dòng hồ sơ người dùng |
| `user_id` duy nhất trong hồ sơ | 2.522.895 | Người dùng |
| `transfer_plan_profile.csv` | 7.630.547 | Dòng hồ sơ phương án |
| `plan_id` duy nhất trong hồ sơ phương án | 3.588.429 | ID phương án |
| Cặp `query_id`, `plan_id` duy nhất trong hồ sơ | 3.592.800 | Cặp khóa theo schema đang đọc |

Số dòng profile không phải số sự kiện bổ sung. **42.433.919 dòng log không phải số mẫu huấn luyện cho bài toán propensity.** Số mẫu phụ thuộc số cutoff, chiều dài lịch sử, khoảng nhãn và điều kiện chọn người dùng.

## 3. Có những feature/cột nào?

### Log hành vi — `user_behavior_transfer_plans.csv`

Toàn bộ log đã được kiểm tra đọc thành **năm trường số nguyên**, không có ô thiếu. CSV không lưu schema; tên cột do dự án gán khi đọc, cấu hình dùng int64.

| Vị trí cột | Tên gán trong dự án | Định dạng đọc | Vai trò và ý nghĩa |
|---:|---|---|---|
| 1 | `user_id` | int64, ID | Khóa người dùng |
| 2 | `query_id` | int64, ID | Mã truy vấn theo schema đang sử dụng |
| 3 | `plan_id` | int64, ID | Mã phương án di chuyển theo schema đang sử dụng |
| 4 | `behavior` | int64, mã phân loại | Mã hành vi 1, 2 hoặc 3; chưa xác minh ánh xạ nghiệp vụ |
| 5 | `timestamp` | int64 → datetime UTC | Unix timestamp, đơn vị giây; dùng cắt lịch sử/nhãn |

### Hồ sơ người dùng — `user_profile.csv`

File có **năm cột**, cấu hình đọc dùng int64. Các tên thuộc tính dưới đây theo schema của dự án; chưa có bảng giải mã giá trị.

| Vị trí cột | Tên gán | Ý nghĩa đang sử dụng |
|---:|---|---|
| 1 | `user_id` | Khóa nối log với hồ sơ |
| 2 | `age` | Mã nhóm tuổi; không diễn giải trực tiếp thành tuổi thực |
| 3 | `gender` | Mã giới tính; có giá trị âm chưa xác định ý nghĩa |
| 4 | `occupation` | Mã nghề nghiệp |
| 5 | `habitual_city` | Mã thành phố thường trú |

### Hồ sơ phương án — `transfer_plan_profile.csv`

Dòng đầu được đối chiếu trực tiếp có **18 trường**. Kiểm tra toàn bộ bảng hiện có chỉ đọc hai cột khóa đầu, chưa xác minh schema đầy đủ hoặc số trường của mọi dòng.

| Vị trí cột | Cách đọc hiện tại | Trạng thái sử dụng |
|---:|---|---|
| 1 | `query_id` | Giả định khóa truy vấn theo schema dự án |
| 2 | `plan_id` | Giả định khóa phương án theo schema dự án |
| 3–18 | Chưa gán tên nghiệp vụ đã xác minh | Chưa đưa vào feature; cần từ điển nguồn trước khi diễn giải |

Chưa đủ bằng chứng để đặt tên cụ thể cho các cột 3–18 như giá, thời lượng hay điểm đi/đến. Cặp khóa cũng cần kiểm tra tính duy nhất và cách nối trước khi bổ sung profile vào log.

Không có cột nhãn “sẽ tương tác trong 7 ngày” trong ba CSV này. ID dùng để liên kết/tổng hợp dữ liệu; các đặc trưng lịch sử cần được tính theo từng cutoff, còn thuộc tính profile chỉ dùng khi đã xác minh schema và cách nối.

## 4. Lịch sử dài bao nhiêu?

| Cách biểu diễn thời gian | Phạm vi |
|---|---|
| UTC từ Unix timestamp | `2022-08-15 16:00:00` → `2022-09-18 15:59:59` |
| Lịch UTC+8 đang dùng trong báo cáo | `16/08/2022 00:00:00` → `18/09/2022 23:59:59` |
| Số ngày theo UTC+8 | **34 ngày**, không có ngày trống trên toàn bộ log |

**UTC+8 là giả định chia ngày của dự án**, chưa phải timezone nguồn đã xác nhận. Chia theo UTC sẽ có 35 ngày lịch, với hai ngày biên không trọn vẹn.

Lịch sử theo người dùng khá thưa:

- Trung vị **5 sự kiện/người**, trung bình **16,81**; người nhiều nhất có **13.585** sự kiện.
- Trung vị **1 ngày hoạt động/người**, trung bình **1,95 ngày**.
- **1.501.339 người (59,46%)** chỉ có log đúng một ngày trong toàn bộ dữ liệu.
- Chỉ **74.734 người (2,96%)** có ít nhất 7 ngày hoạt động.
- Ma trận người dùng × ngày có kích thước **2.524.995 × 34**, trong đó **4.922.954 ô bằng 1**, chiếm **5,7344%**.

34 ngày trên toàn tập không đồng nghĩa mỗi người có 34 ngày lịch sử. Các ngày trước hoặc sau phạm vi dữ liệu không được xem là những ngày không hoạt động.

## 5. Điểm đặc biệt và giới hạn

### Phân bố mã hành vi rất lệch

| Mã | Số sự kiện | Tỷ lệ sự kiện |
|---|---:|---:|
| 1 | 41.127.831 | 96,9221% |
| 2 | 1.235.624 | 2,9119% |
| 3 | 70.464 | 0,1661% |

Không suy luận mã nào là mua chỉ từ tần suất hoặc thứ tự mô tả hành vi trên trang nguồn.

### Profile cần xử lý trước khi ghép

- Hồ sơ người dùng có **91 dòng trùng hoàn toàn**, thuộc 11 ID; chưa thấy profile xung đột sau khi bỏ trùng hoàn toàn. Có **2.100 người trong log thiếu profile**.
- Cột `gender` theo schema hiện tại có **42.924 giá trị âm**; không tự gán giới tính, giữ như mã chưa rõ cho tới khi có từ điển.
- Hồ sơ phương án có **4.037.747 dòng lặp cặp khóa** `query_id`, `plan_id`. Chưa kiểm tra các thuộc tính còn lại để kết luận là trùng hoàn toàn hay nhiều phiên bản.
- Theo phép kiểm tra **sự hiện diện của `plan_id`**, chỉ **9.524.546/42.433.919 dòng log (22,45%)** có ID phương án xuất hiện trong profile; **32.909.373 dòng (77,55%)** không khớp. Đây chưa phải bảo đảm ghép đúng và duy nhất theo cặp khóa.
- Dùng inner join có thể loại phần lớn log; nối vào khóa chưa duy nhất có thể nhân số dòng. Cần xác minh schema, chuẩn hóa và kiểm tra khóa trước; bài toán chỉ dùng log vẫn có thể triển khai độc lập.

Chưa kiểm tra trùng nguyên bản ghi hành vi trên toàn bộ log. Dữ liệu ngắn và người dùng ít quay lại làm hạn chế đánh giá tính ổn định dài hạn. Chia train/validation/test theo thời gian, chỉ dùng dữ liệu trước cutoff làm feature và giữ đủ 7 ngày quan sát nhãn.
