# MBD-mini — từ điển dữ liệu và bài toán dự đoán

Tài liệu mô tả bản dữ liệu tại `data/MBD-mini/`. Số liệu lấy từ kết quả kiểm tra trong repository ngày 03/11/2026; schema được đối chiếu thêm với file Parquet. Các feature tổng hợp bên dưới là đề xuất xây dựng, chưa phải bảng feature đã tạo.

## 1. Dự đoán cái gì?

**Dự đoán khả năng một khách hàng mua từng sản phẩm ngân hàng trong tháng đích**, từ lịch sử giao dịch, vị trí và hội thoại có trước khoảng nhãn.

- Đơn vị mẫu có nhãn: **khách hàng–tháng**, khóa `client_id`, `mon`.
- Nhãn có sẵn: `target_1`, `target_2`, `target_3`, `target_4`, mỗi cột nhận 0/1. Chưa xác định tên nghiệp vụ của bốn sản phẩm.
- Output mô hình: bốn xác suất `p_target_1`–`p_target_4` cho mỗi khách hàng tại thời điểm dự đoán.
- Đây là bài toán nhiều nhãn; một khách có thể có nhiều nhãn dương, tổng bốn xác suất không bắt buộc bằng 1.

**Cần căn chỉnh thời gian nhãn:** tài liệu nguồn được dẫn trong báo cáo hiện có mô tả dự đoán tháng sau ngày báo cáo và có dịch khoảng ngày trong bảng nhãn. Chưa được mặc định `mon` là cutoff cuối phần feature. Phải đối chiếu pipeline benchmark để xác định đầu khoảng nhãn trước khi tạo mẫu; tất cả sự kiện đầu vào phải xảy ra trước mốc đó.

## 2. Có bao nhiêu mẫu?

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

Có **100.224 khách hàng × 12 tháng = 1.202.688 mẫu có nhãn**. Bảng `targets` không trùng khóa khách hàng–tháng, không thiếu bốn nhãn, toàn bộ khách hàng và fold khớp `client_split`. Số mẫu thực sự đưa vào mô hình còn phụ thuộc cutoff và điều kiện đủ lịch sử.

`detail` và `ptls` là hai cách biểu diễn dữ liệu sự kiện; **không cộng số dòng của hai dạng thành tổng số sự kiện**. Chưa đối chiếu tương đương từng sự kiện và cách mã hóa giữa chúng. Dữ liệu lưu dạng Parquet, partition theo `fold=0` đến `fold=4`; `fold` được lấy từ đường dẫn khi đọc.

## 3. Có những feature/cột nào?

### Khóa, phân chia và nhãn

| Cột | Kiểu trong file | Vai trò và ý nghĩa |
|---|---|---|
| `client_id` | string | Khóa khách hàng nối các bảng; dùng định danh/nhóm lịch sử |
| `fold` | Partition | Nhóm khách hàng để chia dữ liệu; dùng quản lý split |
| `mon` | string ngày `YYYY-MM-DD` | Mốc tháng trong bảng nhãn; cần xác nhận quan hệ với cutoff |
| `target_1`–`target_4` | int32 | Bốn nhãn nhị phân; không đưa vào feature cùng mẫu |
| `trans_count` | int32 | Trường đếm giao dịch trong `targets`; chưa xác minh khoảng thời gian tính |
| `diff_trans_date` | int32 | Trường chênh lệch thời gian liên quan giao dịch; chưa xác minh công thức, đơn vị và cutoff |

`trans_count`, `diff_trans_date` chưa được xem là feature an toàn do chưa rõ có sử dụng thông tin trong khoảng nhãn hay không. Cấu hình EDA hiện tại loại hai cột này khỏi phân tích feature.

### Giao dịch — `detail/trx`

| Cột | Kiểu trong file | Nội dung sử dụng |
|---|---|---|
| `client_id` | string | Khóa khách hàng |
| `event_time` | timestamp[ns] | Thời điểm sự kiện; không kèm timezone |
| `amount` | float32 | Giá trị số tiền; chưa xác minh đơn vị hoặc cách biến đổi |
| `event_type`, `event_subtype` | int32 | Mã loại và loại phụ của giao dịch |
| `currency` | int32 | Mã tiền tệ |
| `src_type11`, `src_type12` | int32 | Thuộc tính phía nguồn đã mã hóa |
| `dst_type11`, `dst_type12` | int32 | Thuộc tính phía đích đã mã hóa |
| `src_type21`, `src_type22`, `src_type31`, `src_type32` | int32 | Các thuộc tính nguồn bổ sung đã mã hóa |

Các mã không có bảng quy đổi nghiệp vụ đã xác minh trong dự án; không tự gán tên ngân hàng, kênh giao dịch hay nhóm chi tiêu.

### Vị trí và hội thoại

| Bảng | Cột | Kiểu trong file | Ý nghĩa |
|---|---|---|---|
| `detail/geo` | `client_id`, `event_time` | string, timestamp[ns] | Khách hàng và thời điểm sự kiện vị trí |
| `detail/geo` | `geohash_4`, `geohash_5`, `geohash_6` | int32 | Các mã vị trí; chưa có ánh xạ địa lý được xác minh |
| `detail/dialog` | `client_id`, `event_time` | string, timestamp[ns] | Khách hàng và thời điểm hội thoại |
| `detail/dialog` | `embedding` | list<float32> | Vector biểu diễn hội thoại, không có văn bản gốc |

Các bảng `ptls/*` giữ `client_id` dạng string, còn các trường sự kiện tương ứng được gom thành mảng. `event_time` trong `ptls` là `list<int64>`; cần xác minh đơn vị và cách chuyển đổi, không đọc trực tiếp như timestamp[ns] của `detail`. `ptls/dialog.embedding` là mảng các vector: `list<list<float32>>`.

### Feature có thể xây từ lịch sử

| Nhóm | Ví dụ feature đề xuất |
|---|---|
| Giao dịch | Số giao dịch, số ngày giao dịch, thời gian từ lần gần nhất; thống kê `amount`; tần suất từng mã loại trong 7/30/84 ngày |
| Vị trí | Số sự kiện, số mã vị trí khác nhau, số ngày có vị trí, recency vị trí |
| Hội thoại | Số hội thoại, recency hội thoại, vector tổng hợp từ các embedding trước cutoff |
| Độ phủ dữ liệu | Cờ có/không có từng nguồn, số ngày lịch sử quan sát được của từng nguồn |

Đây là feature cần tính thêm. Phải kiểm tra ý nghĩa `amount` và `currency` trước khi diễn giải tổng tiền theo nghiệp vụ; khóa ID, fold và nhãn có vai trò riêng với feature hành vi.

## 4. Lịch sử dài bao nhiêu?

| Nguồn | Thời gian nhỏ nhất → lớn nhất được lưu |
|---|---|
| Giao dịch | `2020-12-31 21:00:02.169545` → `2022-12-31 20:59:58.460193` |
| Vị trí | `2021-12-11 19:27:07.507106` → `2022-12-31 20:59:58.725788` |
| Hội thoại | `2021-02-10 09:36:07.022271` → `2022-12-31 20:25:11.730797` |
| Mốc tháng nhãn `mon` | `2022-02-28` → `2023-01-31`, đủ 12 mốc tháng |

Timestamp của ba bảng detail không có null; timezone chưa được xác nhận. Timeline chung đủ dài để thử cửa sổ **84 ngày**, nhưng không bảo đảm mỗi khách có đủ 84 ngày hoặc đủ cả ba nguồn. Phạm vi từng nguồn khác nhau; chưa có thống kê đầy đủ về chiều dài lịch sử của từng khách trong lần kiểm tra này.

## 5. Điểm đặc biệt và giới hạn

- **Đa phương thức:** kết hợp giao dịch, vị trí và embedding hội thoại qua `client_id`; số chuỗi khác nhau giữa ba nguồn cho thấy độ phủ nguồn không đồng đều.
- **Nhãn rất hiếm:** cần đánh giá riêng từng target và theo thời gian; accuracy đơn thuần dễ gây hiểu nhầm.

| Nhãn | Mẫu dương | Tỷ lệ trên 1.202.688 khách hàng–tháng |
|---|---:|---:|
| `target_1` | 5.362 | 0,4458% |
| `target_2` | 577 | 0,0480% |
| `target_3` | 4.530 | 0,3767% |
| `target_4` | 3.060 | 0,2544% |

- `diff_trans_date` thiếu **120.786 ô**; không đồng nghĩa các nhãn bị thiếu.
- Cùng khách hàng xuất hiện ở nhiều tháng: khi chia tập cần giữ đúng fold và ranh giới thời gian của thiết kế đánh giá, tránh dùng tương lai làm feature.
- Kiểm tra hiện có bao phủ footer toàn bộ Parquet, toàn bộ timestamp detail, targets và client_split; chưa kiểm tra toàn bộ nội dung feature, trùng sự kiện hay tương đương detail/ptls.

## 6. Nguồn đối chiếu trong dự án

- [Số liệu MBD-mini](../propensity_feasibility/mbd_summary.json), [thống kê theo tháng](../propensity_feasibility/mbd_monthly.csv).
- [Báo cáo kiểm tra và giải thích mốc nhãn](../propensity_feasibility/assessment.md).
- [Cấu hình schema EDA](../../configs/eda_mbd_mini.json), [script kiểm tra](../../src/audit_propensity_feasibility.py).
- [Giới thiệu ba bộ dữ liệu và đường dẫn nguồn gốc](../../DATASETS.md).
