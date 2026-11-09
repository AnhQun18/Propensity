# Đề tài propensity và dữ liệu MBD-mini

Đề tài hiện tại dự đoán khả năng khách hàng mua **ít nhất một trong bốn sản phẩm có nhãn** trong tháng đích, từ lịch sử trước thời điểm dự đoán. Đầu ra chính là một xác suất `p_any`. Bốn nhãn sản phẩm của dữ liệu gốc được giữ để tạo `y_any`; benchmark gốc dự đoán riêng từng sản phẩm.

Tài liệu này tập trung vào MBD-mini. Từ điển chi tiết nằm tại [docs/dict/mbd_mini.md](docs/dict/mbd_mini.md).

## MBD-mini — dữ liệu khách hàng ngân hàng

### Bối cảnh và nội dung

[MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini) là bản rút gọn của Multimodal Banking Dataset. Bộ dữ liệu được ẩn danh và kết hợp nhiều nguồn thông tin về cùng khách hàng: giao dịch, hoạt động vị trí và hội thoại với ngân hàng. Ngoài lịch sử hành vi, bộ này có sẵn nhãn liên quan đến việc mua bốn sản phẩm ngân hàng.

Bài MBD mô tả khách hàng doanh nghiệp. Đề tài hiện giới hạn ở nghiên cứu offline cho
khách hiện hữu; chưa có dữ liệu chiến dịch, chi phí hoặc lợi nhuận để kết luận ROI.
Xem [xác minh và đặc tả nghiên cứu](docs/propensity_feasibility/open_questions/answers.md).

Bản đang có gồm **100.224 khách hàng** trong bảng phân chia dữ liệu và **1.202.688 dòng nhãn khách hàng–tháng**, tương ứng 12 tháng từ `2022-02-28` đến `2023-01-31`. Lịch sử giao dịch trải từ cuối năm 2020 đến cuối năm 2022; các nguồn vị trí và hội thoại có phạm vi thời gian riêng.

Dữ liệu nằm tại `data/MBD-mini/`, lưu dưới dạng **Parquet**, chia thành các thư mục `fold=0` đến `fold=4`. Fold là nhóm khách hàng dùng cho việc chia tập dữ liệu.

### Các bảng đang chứa gì?

| Bảng/thư mục | Một dòng biểu diễn gì? | Nội dung chính |
|---|---|---|
| `client_split` | Một khách hàng | `client_id` và fold của khách hàng |
| `targets` | Một khách hàng tại một tháng | `mon`, bốn nhãn `target_1`–`target_4`, cùng `trans_count`, `diff_trans_date` |
| `detail/trx` | Một sự kiện giao dịch | Khách hàng, thời gian, số tiền, mã loại giao dịch, mã tiền tệ và các thuộc tính nguồn/đích đã mã hóa |
| `detail/geo` | Một sự kiện vị trí | Khách hàng, thời gian và các mã vị trí `geohash_4`, `geohash_5`, `geohash_6` |
| `detail/dialog` | Một sự kiện hội thoại | Khách hàng, thời gian và vector `embedding` biểu diễn hội thoại |
| `ptls/trx` | Một chuỗi giao dịch của khách hàng | Các mảng thời gian, số tiền và thuộc tính giao dịch |
| `ptls/geo` | Một chuỗi hoạt động vị trí của khách hàng | Các mảng thời gian và mã vị trí |
| `ptls/dialog` | Một chuỗi hội thoại của khách hàng | Mảng thời gian và các vector biểu diễn hội thoại |

`detail` biểu diễn sự kiện theo từng dòng; `ptls` là cách tổ chức dữ liệu theo chuỗi. Đây là hai dạng biểu diễn, không phải các nguồn sự kiện độc lập để cộng lại. Khóa kết nối các nguồn là `client_id`.

Quy mô các bảng sự kiện gồm khoảng **38,8 triệu giao dịch**, **66,3 triệu sự kiện vị trí** và **217 nghìn sự kiện hội thoại**. Hội thoại được cung cấp dưới dạng vector số, không phải nội dung văn bản gốc. Các mã giao dịch, vị trí và thuộc tính đã được ẩn danh/mã hóa, nên không tự diễn giải chúng thành thông tin nghiệp vụ cụ thể khi chưa có từ điển.

### Bài toán, input và output

**Quy ước currency:** mã 11 là mã chính; mã khác có giá trị được gắn cờ ngoại tệ của
dự án, currency null có cờ missing riêng và trạng thái ngoại tệ unknown. Giữ mã nguồn;
không tự gán tên tiền thật. Xem [quy tắc và kiểm chứng](docs/propensity_feasibility/remaining_questions/assessment.md).

**Bài toán:** từ lịch sử khách hàng, dự đoán mua ít nhất một trong bốn sản phẩm trong tháng đích. Theo [nghiên cứu `mon`](docs/propensity_feasibility/mon_research.md), tháng chứa `mon` là tháng nhãn; chỉ lấy `event_time < cutoff`. Theo quyết định người dùng, tạm dùng UTC thống nhất cho raw, cutoff và cửa sổ: ví dụ `mon=2022-02-28` có cutoff `2022-02-01 00:00 UTC`, không dịch 3 giờ. Đây là quy ước nghiên cứu, không phải xác nhận timezone của ETL nhãn gốc. Protocol Moscow của EventGPT chỉ là phương án đối chiếu.

**Input:** các giao dịch, hoạt động vị trí và biểu diễn hội thoại của khách hàng trước thời điểm dự đoán. Có thể biểu diễn input thành bảng đặc trưng tổng hợp hoặc chuỗi sự kiện theo thời gian.

**Nhãn có sẵn:** `target_1`, `target_2`, `target_3`, `target_4`, mỗi nhãn nhận giá trị 0 hoặc 1. Giữ nguyên các tên nhãn để tạo `y_any`; tên thương mại của từng sản phẩm không còn là yêu cầu xác minh cho đầu ra gộp.

**Target của đề tài:** `y_any=1` nếu ít nhất một trong bốn target bằng 1;
`y_any=0` nếu đủ bốn giá trị 0. Không suy ra 0 từ nhãn thiếu. Bản hiện có đầy đủ nhãn.

**Output dự kiến của đề tài:** một xác suất cho mỗi khách hàng tại một thời điểm:

```text
client_id | cutoff_utc | tháng_đích_utc | p_any
```

Đề tài hiện tại là **phân loại nhị phân any-product**. Bài toán bốn sản phẩm gốc là
nhiều nhãn: một khách có thể mua nhiều sản phẩm, tổng bốn xác suất riêng không bắt
buộc bằng 1. Không cộng bốn xác suất để thay cho `p_any`. Thiết kế và giới hạn nằm tại
[any-product](docs/propensity_feasibility/any_product/design.md).
