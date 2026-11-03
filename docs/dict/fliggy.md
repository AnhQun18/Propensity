# Fliggy Transfer-Plans — từ điển dữ liệu và bài toán dự đoán

Tài liệu mô tả ba CSV tại `data/Fliggy/`, thuộc bộ User Behavior Transfer-Plans from Fliggy Trip Platform for Recommendation, mã nguồn dữ liệu 140721. Số liệu lấy từ kiểm tra trong repository ngày 03/11/2026. CSV không có header; tên cột do dự án gán khi đọc, các ý nghĩa chưa xác minh được ghi rõ dưới đây.

## 1. Dự đoán cái gì?

Bộ gốc phục vụ **gợi ý phương án di chuyển/nối chuyến**. Trong đề tài propensity, bài toán đang có đủ cơ sở để thử là: **người dùng có phát sinh ít nhất một log tương tác trong 7 ngày tới hay không?**

- Đơn vị mẫu mô hình: **người dùng–cutoff**; mỗi dòng log gốc là một sự kiện người dùng–truy vấn–phương án.
- Chưa có sẵn nhãn propensity tương lai. Tạo `y_activity_7d = 1` nếu có ít nhất một log trong `[cutoff, cutoff + 7 ngày)`, bằng 0 nếu không có trong nhóm người đủ điều kiện.
- Output: `p_activity_7d` trong [0,1]. Phép kiểm tra hiện có tính cả ba mã 1/2/3 là hoạt động được ghi nhận.
- **Chưa xác minh mã `behavior=1/2/3` tương ứng hành vi nào**, nên chưa thể gọi đầu ra là xác suất mua hoặc đặt dịch vụ. Cần bảng quy đổi trước khi đặt nhãn nghiệp vụ cụ thể.

Nhãn 0 nghĩa là không có log trong khoảng đã chọn; không chứng minh người dùng không đi du lịch hoặc được hệ thống quan sát đầy đủ.

## 2. Có bao nhiêu mẫu?

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

Số dòng profile không phải số sự kiện bổ sung. **42.433.919 sự kiện không phải 42.433.919 mẫu người dùng–cutoff**; số mẫu cần tính riêng khi chọn lịch sử và khoảng nhãn.

## 3. Có những feature/cột nào?

### Log hành vi — `user_behavior_transfer_plans.csv`

Toàn bộ log đã được kiểm tra đọc thành **năm trường số nguyên**, không có ô thiếu. Cấu hình đọc dùng int64.

| Vị trí cột | Tên gán trong dự án | Vai trò và ý nghĩa |
|---:|---|---|
| 1 | `user_id` | Khóa người dùng |
| 2 | `query_id` | Mã truy vấn theo schema đang sử dụng |
| 3 | `plan_id` | Mã phương án di chuyển theo schema đang sử dụng |
| 4 | `behavior` | Mã hành vi 1, 2 hoặc 3; chưa xác minh ánh xạ nghiệp vụ |
| 5 | `timestamp` | Unix timestamp, đơn vị giây; dùng tạo lịch sử và nhãn |

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

### Feature có thể xây từ lịch sử

| Nhóm | Ví dụ feature đề xuất |
|---|---|
| Tần suất | Tổng log, số log theo từng mã 1/2/3 trong 1/7/14/21 ngày |
| Recency | Thời gian từ log gần nhất và từ log gần nhất theo từng mã |
| Tính lặp lại | Số ngày hoạt động, khoảng cách giữa các ngày hoạt động, số ngày liên tiếp có log |
| Truy vấn/phương án | Số `query_id`/`plan_id` khác nhau, số phương án trên mỗi truy vấn |
| Chuỗi theo ngày | Ma trận có/không có log mỗi ngày hoặc chuỗi số log từng ngày |
| Hồ sơ người dùng | Các mã tuổi/giới tính/nghề/thành phố sau xử lý trùng và thiếu profile |

Các feature này là đề xuất, chưa phải bảng feature mô hình. Giữ nguyên tên mã hành vi cho tới khi có mapping; feature hồ sơ phương án còn phụ thuộc việc xác minh schema và khóa nối.

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

34 ngày trên toàn tập không đồng nghĩa mỗi người có 34 ngày lịch sử. Có thể thử lịch sử 7/14/21 ngày với nhãn 7 ngày, nhưng lịch sử càng dài càng ít cửa sổ đánh giá. Bản hiện có **không đáp ứng lịch sử 84 ngày**.

### Một cửa sổ đã kiểm tra

Tất cả mốc dưới đây theo giả định UTC+8, tính cả ba mã hành vi:

| Thành phần | Giá trị |
|---|---|
| Lịch sử | `[22/08/2022 00:00, 12/09/2022 00:00)` — 21 ngày |
| Khoảng nhãn | `[12/09/2022 00:00, 19/09/2022 00:00)` — 7 ngày |
| Điều kiện vào nhóm | Có ít nhất một ngày có log trong phần lịch sử |
| Số mẫu người dùng–cutoff | **1.673.450** |
| Nhãn dương | **345.405** người có log tương lai |
| Nhãn âm | **1.328.045** |
| Tỷ lệ dương | **20,6403%** |

Đây là thống kê một cửa sổ khả thi, chưa phải kết quả dự đoán. Điều kiện chọn người dùng chỉ được tính từ lịch sử, không từ toàn bộ 34 ngày có cả tương lai.

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

Chưa kiểm tra trùng nguyên bản ghi hành vi trên toàn bộ log. Dữ liệu ngắn và người dùng ít quay lại làm hạn chế đánh giá tính ổn định dài hạn; không điền ngày ngoài phạm vi bằng 0 để tạo lịch sử giả.

## 6. Nguồn đối chiếu trong dự án

- [Quy mô, thời gian và phân bố lịch sử](../fliggy_140721_profile/summary.json).
- [Kiểm tra khóa và độ phủ profile](../fliggy_140721_profile/profile_links.json).
- [Các cửa sổ đã kiểm tra](../fliggy_140721_profile/window_feasibility.csv), [báo cáo chi tiết](../fliggy_140721_profile/assessment.md).
- [Cấu hình schema EDA](../../configs/eda_fliggy.json), [script profile](../../src/profile_fliggy_140721.py), [script kiểm tra liên kết](../../src/check_fliggy_profile_links.py).
- [Giới thiệu ba bộ dữ liệu và đường dẫn nguồn gốc](../../DATASETS.md).
