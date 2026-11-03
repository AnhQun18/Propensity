# eCommerce — từ điển dữ liệu và bài toán dự đoán

Tài liệu mô tả hai file tại `data/ecommerce/`, thuộc bộ eCommerce behavior data from multi category store. Số liệu lấy từ kết quả kiểm tra trong repository ngày 03/11/2026; header CSV đã được đối chiếu trực tiếp. Các feature tổng hợp là đề xuất, chưa phải bảng feature đã tạo.

## 1. Dự đoán cái gì?

**Dự đoán người dùng có phát sinh ít nhất một hành vi `purchase` trong 7 ngày tới hay không**, dựa trên lịch sử xem, thêm giỏ hàng và mua hàng trước thời điểm dự đoán.

- Đơn vị mẫu mô hình: **người dùng–cutoff**, khác với mỗi dòng CSV là một sự kiện người dùng–sản phẩm.
- Dữ liệu không có sẵn nhãn propensity tương lai. Cần tạo `y_purchase_7d`: bằng 1 nếu có ít nhất một `purchase` trong `[cutoff, cutoff + 7 ngày)`, bằng 0 nếu không có trong nhóm người đủ điều kiện.
- Output mô hình: `p_purchase_7d` trong [0,1] cho mỗi người dùng tại cutoff.
- Thiết kế 7 ngày là lựa chọn của bài toán trong dự án. Có thể đổi khoảng dự đoán hoặc dự đoán theo danh mục/sản phẩm sau khi xác định tập ứng viên và đủ thời gian quan sát.

Nhãn 0 chỉ có nghĩa **không ghi nhận mua trong khoảng nhãn**. Một log `purchase` chưa được xác minh là đơn hàng duy nhất, thanh toán thành công cuối cùng hay doanh thu sau hoàn/hủy.

## 2. Có bao nhiêu mẫu?

| Thành phần | Quy mô |
|---|---:|
| `2019-Oct.csv` | 42.448.764 sự kiện |
| `2019-Nov.csv` | 67.501.979 sự kiện |
| Tổng | **109.950.743 sự kiện** |
| Người dùng duy nhất | **5.316.649** |
| Người có ít nhất một `purchase` trong cả hai tháng | 697.470 |

| Loại sự kiện | Số dòng |
|---|---:|
| `view` — xem sản phẩm | 104.335.509 |
| `cart` — thêm vào giỏ | 3.955.446 |
| `purchase` — mua được ghi nhận trong log | 1.659.788 |

**109.950.743 dòng không phải số mẫu huấn luyện ở cấp người dùng.** Số mẫu phụ thuộc số cutoff, chiều dài lịch sử, khoảng nhãn và điều kiện chọn người dùng. Một người có thể có nhiều sự kiện cùng sản phẩm hoặc trong cùng phiên.

## 3. Có những feature/cột nào?

CSV có header gồm đúng chín cột dưới đây. Kiểu đọc là kiểu cấu hình EDA sử dụng; bản thân CSV không lưu schema kiểu dữ liệu.

| Cột | Kiểu đọc | Vai trò và ý nghĩa |
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

### Feature có thể xây từ lịch sử

| Nhóm | Ví dụ feature đề xuất |
|---|---|
| Recency | Thời gian từ lượt xem, thêm giỏ, mua và hoạt động gần nhất |
| Tần suất | Số `view`/`cart`/`purchase`, số ngày hoạt động trong 1/7/14/21 ngày |
| Phiên | Số phiên, số sự kiện mỗi phiên, số phiên có giỏ hàng hoặc mua |
| Mức độ quan tâm | Số sản phẩm/danh mục/thương hiệu khác nhau, danh mục được xem nhiều nhất |
| Giỏ và mua | Số sản phẩm đã thêm giỏ, tỷ lệ cart/view hoặc purchase/cart với xử lý mẫu số 0 |
| Giá | Trung bình, min/max giá theo loại hành vi; tổng giá các dòng purchase quá khứ nếu định nghĩa rõ cách tổng hợp |
| Xu hướng | So sánh số tương tác 7 ngày gần nhất với 7 ngày trước đó |

Các feature này **chưa tồn tại sẵn**. Tổng `price` trên các log purchase chưa thể gọi là doanh thu ròng, do chưa kiểm tra trùng log và chưa có thông tin hoàn/hủy hoặc số lượng hàng.

## 4. Lịch sử dài bao nhiêu?

- Toàn bộ bản hiện có phủ **61 ngày liên tục**, từ `2019-10-01 00:00:00 UTC` đến `2019-11-30 23:59:59 UTC`.
- Có log ở cả 61 ngày trên toàn bộ tập; chưa có thống kê đầy đủ số ngày hoạt động hoặc tuổi lịch sử của từng người dùng.
- Có thể thử lịch sử 7/14/21/30 ngày và khoảng nhãn ngắn nếu cutoff còn đủ dữ liệu tương lai. Bản hiện có **không đủ lịch sử 84 ngày**.
- Không coi thời gian trước 01/10 hoặc sau 30/11 là những ngày không hoạt động; đó là ngoài phạm vi dữ liệu.

### Một cửa sổ đã kiểm tra

| Thành phần | Giá trị |
|---|---|
| Lịch sử đầu vào | `[03/11/2019 00:00, 24/11/2019 00:00)` UTC — 21 ngày |
| Khoảng nhãn | `[24/11/2019 00:00, 01/12/2019 00:00)` UTC — 7 ngày |
| Điều kiện vào nhóm | Có ít nhất một log trong 21 ngày lịch sử |
| Số mẫu người dùng–cutoff | **3.012.913** |
| Nhãn dương | **71.587** người có purchase trong khoảng nhãn |
| Nhãn âm | **2.941.326** |
| Tỷ lệ dương | **2,3760%** |
| Người trong nhóm có bất kỳ log ở khoảng nhãn | 712.448 |

Đây là một phép tạo nhãn và đếm mẫu để kiểm tra tính khả thi; chưa có mô hình được huấn luyện hoặc điểm dự đoán. Không dùng điều kiện “có log tương lai” để chọn người vào nhóm lịch sử vì sẽ thay đổi đối tượng dự đoán dựa trên tương lai.

## 5. Điểm đặc biệt và giới hạn

- **Luồng xem → giỏ → mua rõ ràng:** có thể tạo feature hành vi và nhãn mua từ cùng log, nhưng cần tách chúng theo thời gian.
- **Lệch mạnh về lượt xem:** hơn 104 triệu dòng view; số dòng purchase không phải tỷ lệ nhãn dương ở cấp người dùng–cutoff.
- **Thiếu danh mục đáng kể:** `category_code` thiếu **35.413.780 dòng**, khoảng **32,21%**. Không cần loại các dòng này để xây bài toán mua bất kỳ; có thể giữ nhóm danh mục không rõ.
- Không thiếu `event_time`, `event_type`, `user_id` trong lần quét toàn bộ bốn cột đã kiểm tra. Điều này không khẳng định mọi cột khác đều đầy đủ hoặc hợp lệ.
- Chưa kiểm tra trùng nguyên dòng, chất lượng `user_session`, độ phủ sản phẩm, toàn bộ `brand`/`price`; chưa có cơ sở xem mỗi purchase là một đơn hàng độc lập.
- Chỉ có hai tháng nên đánh giá mùa vụ dài hạn bị hạn chế. Chia train/validation/test theo thời gian, chỉ dùng dữ liệu trước cutoff làm feature và giữ đủ 7 ngày quan sát nhãn.

## 6. Nguồn đối chiếu trong dự án

- [Số liệu eCommerce và cửa sổ đã kiểm tra](../propensity_feasibility/ecommerce_summary.json), [thống kê theo ngày](../propensity_feasibility/ecommerce_daily.csv).
- [Báo cáo kiểm tra](../propensity_feasibility/assessment.md).
- [Cấu hình schema EDA](../../configs/eda_ecommerce.json), [script kiểm tra](../../src/audit_propensity_feasibility.py).
- [Giới thiệu ba bộ dữ liệu và đường dẫn nguồn gốc](../../DATASETS.md).
