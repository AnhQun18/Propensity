# Fliggy Transfer-Plans — từ điển dữ liệu và bài toán dự đoán
https://tianchi.aliyun.com/dataset/140721

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
- Nguồn mô tả ba loại hành vi `browsing`, `purchasing`, `reference`, nhưng
  file thực tế chứa mã số `1/2/3` và chưa có bảng quy đổi giữa chúng. Chưa
  thể gọi đầu ra là xác suất mua hoặc đặt dịch vụ; không gán mã mua chỉ từ
  thứ tự liệt kê các loại hành vi.

## 2. Quy mô

| File/thành phần | Quy mô | Đơn vị |
|---|---:|---|
| `user_behavior_transfer_plans.csv` | **42.433.919** | Sự kiện |
| Người dùng duy nhất trong log | **2.524.995** | Người dùng |
| `plan_id` duy nhất trong log | 12.172.885 | ID phương án |
| `user_profile.csv` | 2.522.986 | Dòng hồ sơ người dùng |
| `user_id` duy nhất trong hồ sơ | 2.522.895 | Người dùng |
| `transfer_plan_profile.csv` | 7.630.547 | Dòng hồ sơ phương án |

Số dòng profile không phải số sự kiện bổ sung. **42.433.919 dòng log không phải số mẫu huấn luyện cho bài toán propensity.** Số mẫu phụ thuộc số cutoff, chiều dài lịch sử, khoảng nhãn và điều kiện chọn người dùng.

## 3. Có những feature/cột nào?

### Log hành vi — `user_behavior_transfer_plans.csv`

Toàn bộ log đã được kiểm tra đọc thành **năm trường số nguyên**, không có ô thiếu. CSV không lưu schema; tên cột do dự án gán khi đọc.

| Vị trí cột | Tên gán trong dự án | Định dạng đọc | Vai trò và ý nghĩa |
|---:|---|---|---|
| 1 | `user_id` | int64, ID | Khóa người dùng |
| 2 | `query_id` | int64, ID | OD Query ID: mã truy vấn chứa điểm đi và điểm đến |
| 3 | `plan_id` | int64, ID | Transfer Plan ID: mã phương án di chuyển/nối chuyến |
| 4 | `behavior` | int64, mã phân loại | Mã 1, 2 hoặc 3 trong CSV; nguồn nêu `browsing`, `purchasing`, `reference` nhưng chưa cung cấp ánh xạ mã |
| 5 | `timestamp` | int64 → datetime UTC | Unix timestamp, đơn vị giây; dùng cắt lịch sử/nhãn |

### Hồ sơ người dùng — `user_profile.csv`

| Vị trí cột | Tên gán | Ý nghĩa đang sử dụng |
|---:|---|---|
| 1 | `user_id` | Khóa nối log với hồ sơ |
| 2 | `age` | Mã tuổi; không diễn giải trực tiếp thành tuổi thực hoặc tự gán khoảng tuổi |
| 3 | `gender` | Mã giới tính; có giá trị âm chưa xác định ý nghĩa |
| 4 | `occupation` | Mã nghề nghiệp |
| 5 | `habitual_city` | Mã thành phố thường trú, một giá trị cho mỗi hồ sơ |

### Hồ sơ phương án — `transfer_plan_profile.csv`

Nguồn mô tả **18 trường** cho truy vấn và phương án gồm hai chặng. Tên
`snake_case` dưới đây được chuẩn hóa để dùng trong dự án; cột đầu là **thứ
tự liệt kê trong tài liệu nguồn**, chưa phải ánh xạ vị trí CSV đã xác minh.

| Thứ tự theo nguồn | Trường nguồn → tên chuẩn hóa | Định dạng theo nguồn | Vai trò và ý nghĩa |
|---:|---|---|---|
| 1 | Query ID → `query_id` | integer, ID | Mã truy vấn chứa điểm đi và điểm đến |
| 2 | Query Depart City ID → `query_depart_city_id` | integer, ID | Mã thành phố xuất phát của truy vấn |
| 3 | Query Arrive City ID → `query_arrive_city_id` | integer, ID | Mã thành phố đích của truy vấn |
| 4 | Transfer Plan ID → `plan_id` | integer, ID | Mã phương án di chuyển/nối chuyến |
| 5 | First Train ID → `first_train_id` | integer, ID | Mã số hiệu tàu của chặng thứ nhất |
| 6 | First Depart City ID → `first_depart_city_id` | integer, ID | Mã thành phố xuất phát của chặng thứ nhất |
| 7 | First Arrive City ID → `first_arrive_city_id` | integer, ID | Mã thành phố đến của chặng thứ nhất |
| 8 | First Depart Time → `first_depart_time` | string | Giờ khởi hành chặng thứ nhất |
| 9 | First Arrive Time → `first_arrive_time` | string | Giờ đến chặng thứ nhất |
| 10 | First Cost Price → `first_cost_price` | integer | Giá chặng thứ nhất; chưa xác nhận đơn vị tiền tệ |
| 11 | Second Train ID → `second_train_id` | integer, ID | Mã số hiệu tàu của chặng thứ hai |
| 12 | Second Depart City ID → `second_depart_city_id` | integer, ID | Mã thành phố xuất phát của chặng thứ hai |
| 13 | Second Arrive City ID → `second_arrive_city_id` | integer, ID | Mã thành phố đến của chặng thứ hai |
| 14 | Second Depart Time → `second_depart_time` | string | Giờ khởi hành chặng thứ hai |
| 15 | Second Arrive Time → `second_arrive_time` | string | Giờ đến chặng thứ hai |
| 16 | Second Cost Price → `second_cost_price` | integer | Giá chặng thứ hai; chưa xác nhận đơn vị tiền tệ |
| 17 | Total Cost Time → `total_cost_time` | integer | Tổng thời gian của phương án; chưa xác nhận đơn vị |
| 18 | Total Cost Price → `total_cost_price` | integer | Tổng giá của phương án; chưa xác nhận đơn vị tiền tệ |

**Thứ tự CSV thực tế cần xác minh.** 

## 4. Lịch sử dài bao nhiêu?

| Cách biểu diễn thời gian | Phạm vi |
|---|---|
| UTC từ Unix timestamp | `2022-08-15 16:00:00` → `2022-09-18 15:59:59` |
| Số ngày theo UTC+8 | **34 ngày**, không có ngày trống trên toàn bộ log |

**UTC+8 là giả định chia ngày của dự án**, chưa phải timezone nguồn đã xác nhận. Chia theo UTC sẽ có 35 ngày lịch, với hai ngày biên không trọn vẹn.

34 ngày trên toàn tập không đồng nghĩa mỗi người có 34 ngày lịch sử. Các ngày trước hoặc sau phạm vi dữ liệu không được xem là những ngày không hoạt động.

## 5. Điểm đặc biệt và giới hạn

### Phân bố mã hành vi rất lệch

| Mã | Số sự kiện | Tỷ lệ sự kiện |
|---|---:|---:|
| 1 | 41.127.831 | 96,9221% |
| 2 | 1.235.624 | 2,9119% |
| 3 | 70.464 | 0,1661% |

Nguồn liệt kê `browsing`, `purchasing`, `reference`; phần giới thiệu còn
nhắc clicks, favorites, adding và purchases nhưng không giải thích cách
quy đổi. Không suy luận mã nào là mua chỉ từ tần suất hoặc thứ tự mô tả,
và chưa dịch `reference` thành một hành vi nghiệp vụ cụ thể.

### Profile cần xử lý trước khi ghép

- Hồ sơ người dùng có **91 dòng trùng hoàn toàn**, thuộc 11 ID; chưa thấy profile xung đột sau khi bỏ trùng hoàn toàn. Có **2.100 người trong log thiếu profile**.
- Cột `gender` theo schema hiện tại có **42.924 giá trị âm**; không tự gán giới tính, giữ như mã chưa rõ cho tới khi có từ điển.
- **Các thống kê khóa phương án cũ chưa được xác thực theo schema nguồn.** Số 3.588.429 ID, 3.592.800 cặp khóa, 4.037.747 dòng lặp cặp khóa và độ phủ 22,45% trong báo cáo cũ đều được tính với giả định hai cột CSV đầu là `query_id`, `plan_id`. Khi thứ tự trường thực tế chưa được xác minh, không dùng chúng để kết luận số phương án, mức trùng khóa hoặc tỷ lệ log thiếu profile.
- Sau khi xác nhận ánh xạ, cần tính lại tính duy nhất và độ phủ khóa. Nối vào khóa chưa duy nhất có thể nhân số dòng, còn inner join có thể loại các log không khớp. Bài toán chỉ dùng log vẫn có thể triển khai độc lập trong lúc chờ xác minh profile.

Chưa kiểm tra trùng nguyên bản ghi hành vi trên toàn bộ log. Dữ liệu ngắn và người dùng ít quay lại làm hạn chế đánh giá tính ổn định dài hạn. Chia train/validation/test theo thời gian, chỉ dùng dữ liệu trước cutoff làm feature và giữ đủ 7 ngày quan sát nhãn.
