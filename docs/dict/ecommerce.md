# eCommerce — từ điển dữ liệu và bài toán dự đoán

## 1. Tổng quan

**Bài toán.** Dự đoán xác suất một người dùng thực hiện ít nhất một hành vi
`purchase` trong 7 ngày tiếp theo, dựa trên lịch sử xem (`view`), thêm giỏ
hàng (`cart`) và mua hàng (`purchase`) trước thời điểm dự đoán (cutoff).

**Đơn vị mẫu.** Mỗi mẫu của mô hình là một cặp *người dùng–cutoff*. Điều này
khác với dữ liệu gốc, nơi mỗi dòng CSV là một sự kiện người dùng–sản phẩm.

**Nhãn.** Dữ liệu gốc không có sẵn nhãn propensity, nên cần tự tạo
`y_purchase_7d` cho mỗi người dùng tại cutoff:

- `y_purchase_7d = 1` nếu có ít nhất một `purchase` trong khoảng
  `[cutoff, cutoff + 7 ngày)`.
- `y_purchase_7d = 0` nếu không có `purchase` nào trong khoảng đó.
- Chỉ xét những người dùng đủ điều kiện (cần nêu rõ tiêu chí ở phần thiết kế
  mẫu, ví dụ đã có tương tác trước cutoff).

**Đầu ra.** Mô hình trả về `p_purchase_7d ∈ [0, 1]` cho mỗi người dùng tại
cutoff.,

**Lưu ý về ý nghĩa của nhãn.**
- Nhãn 0 chỉ có nghĩa là *không ghi nhận mua trong khoảng nhãn*, không phải
  người dùng không có ý định mua.
- Một log `purchase` chưa được xác minh là đơn hàng duy nhất, thanh toán
  thành công cuối cùng, hay doanh thu sau hoàn/hủy.

## 2. Quy mô

| Thành phần | Quy mô |
|---|---:|
| `2019-Oct.csv` | 42.448.764 sự kiện |
| `2019-Nov.csv` | 67.501.979 sự kiện |
| Tổng | **109.950.743 sự kiện** |
| Người dùng duy nhất | **5.316.649** |
| Người có ít nhất một `purchase` trong cả hai tháng | 697.470 (13.1%) |

| Loại sự kiện | Số dòng |
|---|---:|
| `view` — xem sản phẩm | 104.335.509 |
| `cart` — thêm vào giỏ | 3.955.446 |
| `purchase` — mua được ghi nhận trong log | 1.659.788 |

**109.950.743 dòng không phải số mẫu huấn luyện cho bài toán propensity.** Số mẫu phụ thuộc số cutoff, chiều dài lịch sử, khoảng nhãn và điều kiện chọn người dùng. Một người có thể có nhiều sự kiện cùng sản phẩm hoặc trong cùng phiên.

## 3. Có những feature/cột nào?

CSV có header gồm đúng chín cột dưới đây. Kiểu đọc là kiểu cấu hình EDA sử dụng; bản thân CSV không lưu schema kiểu dữ liệu.

| Cột | Định dạng | Vai trò và ý nghĩa |
|---|---|---|
| `event_time` | Chuỗi → datetime UTC | Thời điểm xảy ra sự kiện, định dạng `%Y-%m-%d %H:%M:%S UTC`; dùng cắt lịch sử/nhãn |
| `event_type` | string/category | `view`, `cart`, `purchase`; đầu vào hành vi trong quá khứ và cơ sở tạo nhãn tương lai |
| `product_id` | Int64, ID | Mã sản phẩm được tương tác |
| `category_id` | Int64, category | Mã danh mục; không xem độ lớn mã như đại lượng liên tục |
| `category_code` | string/category | Nhãn danh mục dạng phân cấp; có giá trị thiếu |
| `brand` | string/category | Thương hiệu; có thể thiếu, chưa thống kê đầy đủ tỷ lệ thiếu |
| `price` | float64 | Giá ghi trong sự kiện; chưa xác minh đơn vị tiền tệ từ kiểm tra nội bộ |
| `user_id` | Int64, ID | Khóa người dùng để ghép lịch sử và tạo mẫu |
| `user_session` | string, ID | Mã phiên truy cập; dùng tổng hợp hành vi theo phiên |

Không có bảng hồ sơ nhân khẩu học hoặc cột nhãn “sẽ mua trong 7 ngày” trong hai CSV này. ID dùng để liên kết/tổng hợp dữ liệu; các đặc trưng lịch sử cần được tính theo từng cutoff.

## 4. Lịch sử dài bao nhiêu?

- Toàn bộ bản hiện có phủ **61 ngày liên tục**, từ `2019-10-01 00:00:00 UTC` đến `2019-11-30 23:59:59 UTC`.
- Có log ở cả 61 ngày trên toàn bộ tập; chưa có thống kê đầy đủ số ngày hoạt động hoặc tuổi lịch sử của từng người dùng.
- Thời gian trước 01/10 hoặc sau 30/11 là những ngày ngoài phạm vi dữ liệu.

## 5. Điểm đặc biệt và giới hạn

- **Luồng xem → giỏ → mua rõ ràng:** có thể tạo feature hành vi và nhãn mua từ cùng log, nhưng cần tách chúng theo thời gian.
- **Lệch mạnh về lượt xem:** hơn 104 triệu dòng view; số dòng purchase không phải tỷ lệ nhãn dương ở cấp người dùng–cutoff.
- **Thiếu danh mục đáng kể:** `category_code` thiếu **35.413.780 dòng**, khoảng **32,21%**. Không cần loại các dòng này để xây bài toán mua bất kỳ; có thể giữ nhóm danh mục không rõ.
- Không thiếu `event_time`, `event_type`, `user_id` trong lần quét toàn bộ bốn cột đã kiểm tra. Điều này không khẳng định mọi cột khác đều đầy đủ hoặc hợp lệ.
- Chưa kiểm tra trùng nguyên dòng, chất lượng `user_session`, độ phủ sản phẩm, toàn bộ `brand`/`price`; chưa có cơ sở xem mỗi purchase là một đơn hàng độc lập.
- Chỉ có hai tháng nên đánh giá mùa vụ dài hạn bị hạn chế. Chia train/validation/test theo thời gian, chỉ dùng dữ liệu trước cutoff làm feature và giữ đủ 7 ngày quan sát nhãn.