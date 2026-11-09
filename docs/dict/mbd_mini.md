# MBD-mini — từ điển dữ liệu và bài toán dự đoán
https://huggingface.co/datasets/ai-lab/MBD-mini

## 1. Tổng quan

**Bài toán hiện tại.** Dự đoán xác suất một khách hàng mua ít nhất một trong bốn sản
phẩm có nhãn trong tháng đích, dựa trên lịch sử trước cutoff. Đầu ra là `p_any`,
target suy ra là `y_any`. Benchmark gốc dự đoán riêng bốn sản phẩm; cấu trúc bốn nhãn
nguồn được giữ nguyên. Xem [thiết kế hiện tại](../propensity_feasibility/any_product/design.md).

**Đơn vị mẫu.** Mỗi mẫu có nhãn là một cặp *khách hàng–tháng*, với khóa
`client_id`, `mon`. Điều này khác với các bảng `detail`, nơi mỗi dòng là
một sự kiện, và các bảng `ptls`, nơi mỗi dòng là một chuỗi của khách hàng.

**Nhãn.** Dữ liệu có sẵn bốn nhãn `target_1`, `target_2`, `target_3`,
`target_4` cho mỗi khách hàng–tháng:

- Mỗi nhãn nhận 1 nếu ghi nhận mua sản phẩm tương ứng trong khoảng nhãn,
  bằng 0 nếu không ghi nhận mua sản phẩm đó.
- Giữ nguyên tên nhãn để tạo `y_any`; tên nghiệp vụ từng sản phẩm không còn là yêu cầu
  xác minh trong phạm vi đầu ra gộp theo quyết định người dùng.
- Tháng lịch chứa `mon` là tháng nhãn theo quy ước đã đối chiếu với các bài báo;
  lịch sử chỉ gồm sự kiện `event_time < cutoff`. Người dùng quyết định tạm dùng UTC
  thống nhất; ví dụ `mon=2022-02-28` có cutoff `2022-02-01 00:00 UTC`, không dịch 3 giờ.
  Đây là quy ước nghiên cứu, không phải xác nhận timezone của ETL nhãn gốc. Xem
  [nghiên cứu `mon` và giới hạn về múi giờ](../propensity_feasibility/mon_research.md).

**Đầu ra.** Đề tài trả về `p_any ∈ [0,1]` cho mỗi khách hàng tại cutoff.
`y_any=1` khi có ít nhất một target bằng 1, bằng 0 khi đủ bốn target bằng 0.
Nếu không có 1 và có nhãn thiếu thì `y_any` không xác định, không điền thành 0.
Các đầu ra riêng `p_target_1`–`p_target_4` thuộc bài toán nhiều nhãn gốc.

**Lưu ý về ý nghĩa của nhãn.**

- Nhãn 0 chỉ có nghĩa là *không ghi nhận mua sản phẩm tương ứng trong khoảng
  nhãn*, không phải khách hàng không có ý định mua.

## 2. Quy mô

Bản hiện có tại `data/MBD-mini/` gồm các bảng Parquet sau:

| Bảng/thư mục | Một dòng biểu diễn | Số dòng |
|---|---|---:|
| `client_split` | Một khách hàng được gán fold | 100.224 |
| `targets` | Một khách hàng tại một tháng | 1.202.688 |
| `detail/trx` | Một sự kiện giao dịch | 38.764.930 |
| `detail/geo` | Một sự kiện vị trí | 66.295.724 |
| `detail/dialog` | Một sự kiện hội thoại | 217.205 |
| `ptls/trx` | Một chuỗi giao dịch của khách hàng | 98.721 |
| `ptls/geo` | Một chuỗi vị trí của khách hàng | 72.573 |
| `ptls/dialog` | Một chuỗi hội thoại của khách hàng | 46.320 |

Có **100.224 khách hàng × 12 tháng = 1.202.688 mẫu có nhãn**. Bảng `targets` không trùng khóa khách hàng–tháng, không thiếu bốn nhãn, toàn bộ khách hàng và fold khớp `client_split`. **Số dòng sự kiện không phải số mẫu huấn luyện cho bài toán propensity.** Số mẫu thực sự đưa vào mô hình còn phụ thuộc cutoff và điều kiện đủ lịch sử.

`detail` và `ptls` là hai cách biểu diễn dữ liệu sự kiện; **không cộng số dòng của hai dạng thành tổng số sự kiện**.

## 3. Có những feature/cột nào?

### Khóa, phân chia và nhãn

| Cột | Định dạng | Vai trò và ý nghĩa |
|---|---|---|
| `client_id` | string | Khóa khách hàng nối các bảng; dùng định danh/nhóm lịch sử |
| `fold` | Partition | Nhóm khách hàng để chia dữ liệu; dùng quản lý split |
| `mon` | string ngày `YYYY-MM-DD` | Ngày cuối tháng nhãn; cutoff ở ngày đầu tháng chứa `mon`, chỉ dùng lịch sử trước cutoff |
| `target_1`–`target_4` | int32 | Bốn nhãn nhị phân; không đưa vào feature cùng mẫu |
| `trans_count` | int32 | Trường đếm trong targets, không giảm theo tháng; chưa tái tạo được từ log được phát hành. Chưa dùng làm predictor; tự đếm từ detail trước cutoff |
| `diff_trans_date` | int32 | Trường chênh lệch thời gian liên quan giao dịch; chưa xác minh công thức, đơn vị và cutoff |

### Giao dịch — `detail/trx`

| Cột | Định dạng | Vai trò và ý nghĩa |
|---|---|---|
| `client_id` | string | Khóa khách hàng |
| `event_time` | timestamp[ns] | Thời điểm sự kiện; không kèm timezone |
| `amount` | float32 | Giá trị số tiền; chưa xác minh đơn vị hoặc cách biến đổi |
| `event_type`, `event_subtype` | int32 | Mã loại và loại phụ của giao dịch |
| `currency` | int32 | Mã tiền tệ |
| `src_type11`, `src_type12` | int32 | Thuộc tính phía nguồn đã mã hóa |
| `dst_type11`, `dst_type12` | int32 | Thuộc tính phía đích đã mã hóa |
| `src_type21`, `src_type22`, `src_type31`, `src_type32` | int32 | Các thuộc tính nguồn bổ sung đã mã hóa |

Các mã không có bảng quy đổi nghiệp vụ đã xác minh; không tự gán tên ngân hàng, kênh giao dịch hay nhóm chi tiêu.

**Cờ currency tự tạo theo quyết định người dùng:** mã 11 là mã chính; mã khác có giá
trị là ngoại tệ theo quy ước dự án. Hàm [currency_flags](../../src/mbd_mini_eda/currency.py)
tạo `currency_is_main`, `currency_is_foreign` (boolean nullable) và `currency_missing`.
Currency null có hai cờ đầu null, cờ missing=True; giữ currency nguồn, không gán tên
đồng tiền thật. Các cờ này không phải cột có sẵn trong raw.

Xem [xác minh trans_count, missing và thời điểm nhãn](../propensity_feasibility/remaining_questions/assessment.md).
Targets không có mốc label_available_at; lịch train/calibration phải ghi rõ giả định
độ trễ công bố nhãn, thay vì coi mon là thời điểm thông tin đã sẵn sàng.

### Vị trí và hội thoại

| Bảng | Cột | Định dạng | Vai trò và ý nghĩa |
|---|---|---|---|
| `detail/geo` | `client_id`, `event_time` | string, timestamp[ns] | Khách hàng và thời điểm sự kiện vị trí |
| `detail/geo` | `geohash_4`, `geohash_5`, `geohash_6` | int32 | Các mã vị trí; chưa có ánh xạ địa lý được xác minh |
| `detail/dialog` | `client_id`, `event_time` | string, timestamp[ns] | Khách hàng và thời điểm hội thoại |
| `detail/dialog` | `embedding` | list<float32> | Vector biểu diễn hội thoại, không có văn bản gốc |

## 4. Lịch sử dài bao nhiêu?

| Nguồn | Thời gian nhỏ nhất → lớn nhất được lưu |
|---|---|
| Giao dịch | `2020-12-31 21:00:02.169545` → `2022-12-31 20:59:58.460193` |
| Vị trí | `2021-12-11 19:27:07.507106` → `2022-12-31 20:59:58.725788` |
| Hội thoại | `2021-02-10 09:36:07.022271` → `2022-12-31 20:25:11.730797` |
| Mốc tháng nhãn `mon` | `2022-02-28` → `2023-01-31`, đủ 12 mốc tháng |

Timestamp của ba bảng detail không có null; timezone chưa được xác nhận. Phạm vi từng nguồn khác nhau.
Đã bổ sung phân phối chiều dài lịch sử, ngày hoạt động và tháng đầu/cuối của khách
trong [báo cáo 7 bước](../mbd_mini_eda/report.md); xem [kiểm kê đầu ra](../mbd_mini_eda_audit.md).
Timeline chung không bảo đảm mỗi khách được quan sát suốt khoảng đó hoặc có đủ cả ba nguồn.
Các ngày ngoài phạm vi của từng nguồn không được xem là những ngày không hoạt động.

## 5. Điểm đặc biệt và giới hạn

- **Đa phương thức:** kết hợp giao dịch, vị trí và embedding hội thoại qua `client_id`; số chuỗi khác nhau giữa ba nguồn cho thấy độ phủ nguồn không đồng đều.
- **Nhãn rất hiếm:** cần đánh giá riêng từng target và theo thời gian.

| Nhãn | Mẫu dương | Tỷ lệ trên 1.202.688 khách hàng–tháng |
|---|---:|---:|
| `target_1` | 5.362 | 0,4458% |
| `target_2` | 577 | 0,0480% |
| `target_3` | 4.530 | 0,3767% |
| `target_4` | 3.060 | 0,2544% |

