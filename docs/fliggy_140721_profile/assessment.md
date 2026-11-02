# Đánh giá Fliggy Transfer-Plans 140721

**Kết luận: phù hợp có điều kiện để thử nghiệm propensity theo ngày với cửa sổ ngắn. Không đáp ứng lịch sử 84 ngày. Chưa nên tạo nhãn mua hàng hoặc ghép hồ sơ phương án trước khi xử lý các điểm dưới đây.**

Phạm vi: quét toàn bộ 42.433.919 dòng log, toàn bộ hồ sơ người dùng, và hai cột khóa đầu của toàn bộ hồ sơ phương án. Dữ liệu nguồn không bị sửa. Không huấn luyện mô hình trong lần kiểm tra này.

## Đối chiếu yêu cầu

| Yêu cầu | Kết quả thực tế | Đánh giá |
|---|---|---|
| Nhật ký có ID khách hàng và thời gian | 5 cột số nguyên, không có header: user_id, query_id, plan_id, behavior, timestamp | Đạt về cấu trúc theo schema đã mô tả |
| Tần suất theo khách hàng | 2.524.995 người dùng; trung vị 5 bản ghi/người; trung bình 16,81 | Thực hiện được, nhưng số dòng khác số ngày quay lại |
| Tần suất theo ngày | 34/34 ngày có dữ liệu; 893.995–1.711.780 dòng/ngày | Không có ngày trống trên toàn bộ tập |
| Ma trận khách hàng × ngày | 2.524.995 × 34; 4.922.954 ô bằng 1, chiếm 5,7344% | Dựng được; cần xác định hành vi nào được tính là tương tác |
| Lịch sử 84 ngày | Toàn bộ file chỉ có 34 ngày theo giả định UTC+8 | Không đạt nếu 84 ngày là bắt buộc |

## Thời gian và mức độ lặp lại

Timestamp là Unix giây. Khoảng UTC thực tế là 2022-08-15 16:00:00 đến 2022-09-18 15:59:59. Để thống kê theo ngày, báo cáo dùng UTC+08:00 vì dữ liệu từ nền tảng Trung Quốc; đây là giả định được ghi rõ, chưa phải timezone do nguồn xác nhận. Theo cách chia này, dữ liệu phủ trọn ngày 16/08/2022–18/09/2022, khác ngày kết thúc 17/09 trong tài liệu trang web đã đọc. Nếu chia theo UTC, sẽ có 35 ngày lịch với hai ngày biên không trọn vẹn.

| Số ngày có hoạt động | Số người dùng | Tỷ lệ toàn bộ người dùng |
|---|---:|---:|
| Đúng 1 ngày | 1.501.339 | 59,46% |
| Từ 2 ngày | 1.023.656 | 40,54% |
| Từ 3 ngày | 537.203 | 21,28% |
| Từ 5 ngày | 187.189 | 7,41% |
| Từ 7 ngày | 74.734 | 2,96% |
| Từ 10 ngày | 21.634 | 0,86% |

Trung vị số ngày hoạt động chỉ là 1, trung bình là 1,95 ngày/người. Vì vậy, quy mô 42 triệu dòng không đồng nghĩa có lịch sử dài cho từng người dùng. Nếu chọn một cohort có hành vi lặp lại, chỉ được xét điều kiện từ phần lịch sử của từng mẫu, không xét cả tương lai. Việc lọc sẽ thay đổi đối tượng áp dụng mô hình sang nhóm người dùng quay lại.

Các thống kê ở trên tính mọi mã hành vi. Một ô 0 chỉ có nghĩa là không có bản ghi hành vi được chọn trong ngày, không chứng minh người dùng được quan sát đầy đủ hoặc không đi du lịch.

## Mã hành vi: còn thiếu bảng quy đổi

| Mã trong file | Số bản ghi | Tỷ lệ | Người dùng có mã này |
|---|---:|---:|---:|
| 1 | 41.127.831 | 96,9221% | 2.462.449 |
| 2 | 1.235.624 | 2,9119% | 539.178 |
| 3 | 70.464 | 0,1661% | 31.907 |

Tài liệu trang web liệt kê browsing, purchasing, reference nhưng file dùng mã 1/2/3. Chưa có bằng chứng đủ để gán chính xác mã số cho tên hành vi. Không suy ra mã 2 là mua chỉ từ tỷ lệ hoặc thứ tự liệt kê. Cần xác nhận bảng quy đổi trước khi tạo nhãn purchasing, và cần hiểu reference có phải tương tác chủ động hay không trước khi đưa nó vào ma trận.

## Kiểm tra khả năng tạo cửa sổ

Ví dụ kiểm tra cố định: lịch sử 22/08–11/09/2022 (21 ngày), tương lai 12/09–18/09/2022 (7 ngày). Các điều kiện dưới đây chỉ dùng số ngày hoạt động trong lịch sử.

| Điều kiện lịch sử | Người đủ điều kiện | Có hoạt động bất kỳ trong 7 ngày sau | Tỷ lệ quay lại |
|---|---:|---:|---:|
| Từ 1 ngày | 1.673.450 | 345.405 | 20,64% |
| Từ 2 ngày | 595.837 | 189.612 | 31,82% |
| Từ 3 ngày | 276.972 | 110.051 | 39,73% |
| Từ 5 ngày | 76.149 | 39.772 | 52,23% |

Đây là thống kê khả thi của một cửa sổ, không phải kết quả dự đoán. Với nhóm từ 3 ngày lịch sử, số người có mã 2 trong tương lai là 28.191 và mã 3 là 1.418; đây vẫn là các nhãn theo mã số chưa xác định nghiệp vụ.

Khuyến nghị thử nghiệm:

- Nếu chỉ kiểm chứng ý tưởng, có thể thử lịch sử 14 hoặc 21 ngày và dự đoán 7 ngày sau.
- Nếu cần train/validation/test có ba khoảng nhãn 7 ngày không chồng nhau trong 34 ngày hiện có, bắt đầu với lịch sử 7 ngày. Ví dụ: train lịch sử 22–28/08, nhãn 29/08–04/09; validation lịch sử 29/08–04/09, nhãn 05–11/09; test lịch sử 05–11/09, nhãn 12–18/09. Tại mỗi mốc, chỉ dùng lịch sử có sẵn trước khoảng nhãn.
- Chọn và báo cáo baseline recency/frequency trước khi kết luận mô hình học chuỗi có lợi. Dữ liệu chỉ phủ một tháng nên đánh giá mùa vụ và tính ổn định dài hạn còn hạn chế.

## Hồ sơ bổ sung và chất lượng khóa

**user_profile.csv**: 2.522.986 dòng, 2.522.895 ID duy nhất. Có 91 dòng trùng hoàn toàn, thuộc 11 ID; không phát hiện các profile xung đột sau khi bỏ trùng hoàn toàn. Có 2.100 người dùng trong log không có profile. Cột thứ ba theo schema giới tính có 42.924 giá trị âm; nên giữ như mã không rõ cho tới khi có từ điển mã hóa. Hồ sơ không bắt buộc để dựng ma trận theo ngày.

**transfer_plan_profile.csv**: kiểm tra khóa với giả định hai cột đầu lần lượt là query_id và plan_id. Có 7.630.547 dòng, 3.588.429 plan_id duy nhất và 3.592.800 cặp query_id/plan_id duy nhất. Có 4.037.747 dòng lặp khóa cặp; chưa kiểm tra các thuộc tính còn lại để phân biệt trùng hoàn toàn với nhiều phiên bản.

Chỉ 9.524.546/42.433.919 dòng log (22,45%) có plan_id xuất hiện trong bảng profile theo cách đọc này. Có 8.584.456/12.172.885 plan_id trong log chưa tìm thấy profile. Chưa xác định nguyên nhân: cần đối chiếu schema thực tế và cách mã hóa ID, cũng như tính đầy đủ của bảng profile. Không inner join tùy tiện vì sẽ loại 77,55% dòng log; không join khóa chưa duy nhất vì có thể nhân số dòng. Vấn đề này không ngăn dựng ma trận chỉ từ log, nhưng ảnh hưởng việc bổ sung thông tin hành trình/giá.

## Kết quả lưu kèm

- `summary.json`: quy mô, phạm vi thời gian, phân vị và kiểm tra tổng.
- `daily_counts.csv`: số dòng, người dùng và từng mã hành vi theo ngày UTC+8.
- `daily_counts_utc.csv`: đối chiếu số dòng theo ngày UTC.
- `per_user_counts.parquet`: tần suất và số ngày hoạt động của từng người dùng.
- `event_count_distribution.csv`, `active_day_distribution.csv`: phân bố tần suất.
- `window_feasibility.csv`: các cửa sổ 7/14/21 ngày và nhãn 7 ngày theo mã hành vi.
- `binary_matrix_sample.csv`: mẫu 20 dòng ma trận nhị phân, gồm 34 ngày liên tục.
- `profile_links.json`: kiểm tra trùng khóa và khả năng ghép profile.

Đã đối chiếu tổng theo ngày, theo người dùng và theo mã hành vi bằng tổng dòng log. Chưa kiểm tra trùng nguyên bản ghi trên toàn bộ log và chưa kiểm tra tính xác thực nghiệp vụ của từng sự kiện. Không gán số dòng thừa trong profile thành giao dịch bổ sung.

Nguồn mô tả: https://tianchi.aliyun.com/dataset/140721. Các kết quả định lượng trong báo cáo được tính từ ba file tải về trong `data/Fliggy`.

Chạy lại bằng Python của dự án:

```powershell
.\.venv\Scripts\python.exe -X utf8 src/profile_fliggy_140721.py
.\.venv\Scripts\python.exe -X utf8 src/check_fliggy_profile_links.py
```
