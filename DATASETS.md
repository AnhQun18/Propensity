# Đề tài propensity và dữ liệu MBD-mini

Đề tài dự đoán khả năng khách hàng mua sản phẩm ngân hàng trong tương lai, từ lịch sử đã có trước thời điểm dự đoán. Đầu ra là bốn xác suất tương ứng với bốn nhãn sản phẩm.

Tài liệu này tập trung vào MBD-mini. Từ điển chi tiết nằm tại [docs/dict/mbd_mini.md](docs/dict/mbd_mini.md).

## MBD-mini — dữ liệu khách hàng ngân hàng

### Bối cảnh và nội dung

[MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini) là bản rút gọn của Multimodal Banking Dataset. Bộ dữ liệu được ẩn danh và kết hợp nhiều nguồn thông tin về cùng khách hàng: giao dịch, hoạt động vị trí và hội thoại với ngân hàng. Ngoài lịch sử hành vi, bộ này có sẵn nhãn liên quan đến việc mua bốn sản phẩm ngân hàng.

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

**Bài toán:** từ lịch sử của khách hàng, dự đoán khả năng mua từng sản phẩm trong tháng đích. Theo [nghiên cứu ý nghĩa `mon`](docs/propensity_feasibility/mon_research.md), tháng lịch chứa `mon` là tháng nhãn theo quy ước triển khai; cutoff là ngày đầu tháng đó và chỉ lấy sự kiện `event_time < cutoff`. Ví dụ `mon=2022-02-28` dùng lịch sử trước `2022-02-01` để dự đoán mua trong tháng 02/2022. Múi giờ của timestamp nguồn chưa được xác nhận.

**Input:** các giao dịch, hoạt động vị trí và biểu diễn hội thoại của khách hàng trước thời điểm dự đoán. Có thể biểu diễn input thành bảng đặc trưng tổng hợp hoặc chuỗi sự kiện theo thời gian.

**Nhãn có sẵn:** `target_1`, `target_2`, `target_3`, `target_4`, mỗi nhãn nhận giá trị 0 hoặc 1. Tên sản phẩm cụ thể chưa được xác định trong tài liệu đang sử dụng, nên giữ nguyên các tên nhãn này.

**Output dự kiến:** bốn điểm xác suất cho mỗi khách hàng tại một thời điểm:

```text
client_id | thời_điểm_dự_đoán | tháng_đích | p_target_1 | p_target_2 | p_target_3 | p_target_4
```

Đây là bài toán **nhiều nhãn**: một khách có thể mua nhiều sản phẩm. Vì vậy, tổng bốn xác suất không bắt buộc bằng 1. Nhãn 0/1 là kết quả quan sát dùng để học; các cột `p_target_*` là đầu ra cần được mô hình ước lượng.
