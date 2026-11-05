# Dùng toàn bộ lịch sử trước cutoff và tỷ lệ ID không có log trên 50% ngày

Ngày kiểm tra: **05/11/2026**. Dữ liệu gốc không thay đổi. Đọc lại toàn bộ 1.275 file detail; chỉ xuất các thống kê tổng hợp.

## Cutoff và khoảng lịch sử

Giữ **cutoff = 01/12/2022 00:00:00** để dự đoán hành vi trong tháng 12. Mở rộng đầu lịch sử về trước không đòi hỏi dời cutoff. Đưa sự kiện tháng 12 vào input sẽ sử dụng thông tin của chính tháng đang muốn dự đoán.

Hai cách hiểu “dùng hết” được tính riêng, bỏ ngày đầu không trọn vẹn của từng nguồn và dùng timestamp không kèm timezone đúng như lưu trong file:

| Phạm vi | Khoảng lịch sử | Số ngày |
|---|---|---:|
| Toàn bộ khoảng chung ba nguồn trước cutoff | 12/12/2021–30/11/2022 | **354** |
| Toàn bộ giao dịch trước cutoff | 01/01/2021–30/11/2022 | **699** |
| Toàn bộ vị trí trước cutoff | 12/12/2021–30/11/2022 | **354** |
| Toàn bộ hội thoại trước cutoff | 11/02/2021–30/11/2022 | **658** |

Có thể dùng lịch sử dài khác nhau theo nguồn nếu cách tạo feature/mô hình xử lý được độ dài và nguồn vắng mặt. Không giả lập các nguồn đã tồn tại từ trước thời điểm thực tế bắt đầu.

Nếu muốn đưa cả sự kiện tháng 12 vào input, dùng cutoff 01/01/2023 để dự đoán tháng 01/2023, với nhãn chính thức tại `mon=2023-01-31`, theo [quy ước đã đối chiếu với các bài báo](../propensity_feasibility/mon_research.md). Các số đếm trong báo cáo này vẫn được tính tại cutoff 01/12/2022; cần tính lại thống kê lịch sử nếu chuyển sang cutoff 01/01/2023. Múi giờ của timestamp nguồn chưa được xác nhận.

## Định nghĩa ngưỡng 50%

Với khách hàng u, đặt ngày có log là ngày có ít nhất một sự kiện được giữ trong bất kỳ nguồn giao dịch, vị trí hoặc hội thoại:

```text
tỷ_lệ_ngày_không_có_log(u)
    = 1 - số_ngày_có_ít_nhất_một_sự_kiện(u) / tổng_số_ngày_lịch_sử
```

Trong 354 ngày, “trên 50%” nghĩa là **không có log ít nhất 178 ngày**, tương đương có log không quá 176 ngày. Khách có đúng 177 ngày có log nằm ở mức đúng 50%, nên không được đếm vào nhóm trên 50%.

**Không có log không đồng nghĩa thiếu dữ liệu thu thập.** Dữ liệu không có cờ xác nhận quan sát liên tục từng khách, nên không thể đo trực tiếp tỷ lệ ID thực sự bị thất thoát hơn 50% hành vi. Các kết quả dưới đây là thống kê độ thưa log theo định nghĩa cụ thể.

Sự kiện được giữ theo chính sách chặt như audit trước: ID trong bảng khách hàng, timestamp hợp lệ, các thuộc tính không null, số tiền hữu hạn; embedding hội thoại đúng 768 phần tử hữu hạn, không null.

## Kết quả trên khoảng chung 354 ngày

| Nhóm và mẫu số | ID không có log trên 50% ngày | Tỷ lệ trong nhóm | ID còn lại có log ít nhất 50% ngày |
|---|---:|---:|---:|
| Toàn bộ **100.224 ID** ban đầu | **42.320** | **42,23%** | **57.904** |
| **99.068 ID** có ít nhất một sự kiện hợp lệ trong lịch sử | 41.164 | 41,55% | 57.904 |
| **25.695 ID** có sự kiện hợp lệ ở đủ ba nguồn trong lịch sử | **4.854** | **18,89%** | **20.841** |

Nhóm 20.841 ID tương ứng **20,79%** trên 100.224 ID ban đầu. Đây là giao của hai điều kiện: có sự kiện ở cả ba nguồn trong lịch sử 354 ngày và có log ở bất kỳ nguồn nào trong ít nhất 177 ngày. Không yêu cầu mỗi nguồn xuất hiện trong ít nhất 177 ngày.

Con số **27.342 ID** của audit trước tính trên cả khoảng trọn ngày kết thúc 30/12/2022. Nó chứa các sự kiện tháng 12. Khi chốt input trước 01/12/2022, nhóm đủ ba nguồn là **25.695 ID**, không phải 27.342.

Nếu chỉ kiểm tra ID/timestamp, chưa loại giao dịch thiếu thuộc tính, tỷ lệ không có log trên 50% ngày là **42,11%** (42.208 ID). Kết quả chặt là 42,23%; ảnh hưởng của loại các thuộc tính null nhỏ hơn nhiều so với độ thưa hoạt động và khác biệt thời gian.

## Nếu sử dụng lịch sử dài nhất 699 ngày

Nếu lấy toàn bộ các ngày từ **01/01/2021 đến 30/11/2022**, mỗi ngày được đánh dấu có log khi xuất hiện ở bất kỳ nguồn đang có dữ liệu:

- **62.997/100.224 ID = 62,86%** không có log trên một nửa số ngày.
- **37.227 ID = 37,14%** có log ít nhất một nửa số ngày, tức ít nhất 350/699 ngày.

Tỷ lệ tăng khi mở rộng từ 354 lên 699 ngày không chứng minh chất lượng dữ liệu giảm. Cửa sổ dài hơn có thể chứa nhiều ngày trước khi một khách bắt đầu được ghi nhận hoặc trước khi có nguồn vị trí. Không áp dụng mẫu số 699 cho từng nguồn như thể cả ba đều được thu thập suốt 699 ngày.

## Các cách hiểu khác của “thiếu hơn 50%”

Những phép đo sau không tương đương và không được thay thế cho nhau:

| Định nghĩa trong lịch sử chung 354 ngày | ID thỏa điều kiện | Tỷ lệ trên 100.224 ID |
|---|---:|---:|
| Không có log bất kỳ nguồn nào trên 50% ngày | 42.320 | 42,23% |
| Không có giao dịch hợp lệ trên 50% ngày | 76.514 | 76,34% |
| Không có log vị trí trên 50% ngày | 60.764 | 60,63% |
| Không có hội thoại trên 50% ngày | 100.224 | 100,00% |
| Thiếu nguồn: không có log ở ít nhất 2/3 nguồn trong cả lịch sử | 22.329 | 22,28% |
| Không có log ở trên 50% các ô ngày × nguồn (354 × 3 ô) | 95.227 | 95,01% |
| Span lịch từ sự kiện đầu đến cuối ở bất kỳ nguồn nào ngắn hơn 177 ngày | 7.774 | 7,76% |
| Span thời gian chồng nhau của cả ba nguồn ngắn hơn 177 ngày, hoặc thiếu nguồn | 96.396 | 96,18% |

Hai phép đo span chỉ xét biên đầu/cuối và không bảo đảm có sự kiện ở các ngày giữa. Ngưỡng theo ngày cũng không phù hợp để đòi hỏi hội thoại diễn ra hàng ngày: hoạt động này vốn rất thưa. Do đó không tự động xóa khách hàng chỉ vì không có hội thoại trong phần lớn ngày.

Nếu “thiếu” được hiểu là **thuộc tính của những giao dịch đã ghi nhận bị null**, có **323 ID** có hơn một nửa giao dịch trong khoảng chung không đạt quy tắc đầy đủ thuộc tính. Con số đó là **0,32%** toàn bộ khách, hoặc **0,33%** trong 96.903 khách có giao dịch thô trong khoảng. Đây là tỷ lệ thuộc tính thiếu trong sự kiện đã quan sát được, khác với thiếu ngày hoạt động.

## Ý nghĩa cho propensity

- Với mục tiêu dự đoán tháng 12, giữ cutoff 01/12/2022 và dùng lịch sử trước mốc này; không chọn cohort dựa vào hoạt động tháng 12.
- Có thể thử lịch sử 84 ngày và toàn bộ lịch sử, giữ cùng định nghĩa nhãn và cách đánh giá để so sánh mô hình.
- Không coi điều kiện có log ít nhất 50% số ngày là yêu cầu bắt buộc cho propensity. Khách hoạt động thưa vẫn có thể là đối tượng cần dự đoán. Lọc theo hoạt động mạnh sẽ làm thay đổi nhóm khách và tỷ lệ nhãn.
- Nếu mục tiêu nghiên cứu chủ động yêu cầu đủ ba nguồn và có log ít nhất một nửa số ngày trong toàn bộ khoảng chung, cohort mô tả tại cutoff là **20.841 ID**. Cần kiểm tra nhãn và độ phủ theo cutoff trước khi xác nhận tập train/test.

## Tái lập

```powershell
.\.venv\Scripts\python.exe -X utf8 src/audit_mbd_full_history_missingness.py
```

Kết quả: `full_history_missingness.json` và `full_history_missingness.csv` trong thư mục này. Script kiểm tra tổng dòng và số dòng thiếu thuộc tính khớp audit trước, đồng thời đối chiếu cờ ngày hợp lệ là tập con của cờ ngày thô. Không xuất danh sách client ID.
