# Nghiên cứu `diff_trans_date` của MBD / MBD-mini

Ngày kiểm tra: 07/11/2026.

## Kết luận

Nhà phát hành mô tả `diff_trans_date` là **“Time difference between transactions”** (chênh lệch thời gian giữa các giao dịch). Tuy nhiên, nguồn đã kiểm tra không công bố hai mốc thời gian, đơn vị, cửa sổ lịch sử hoặc công thức tạo trường này. Chưa thể khẳng định đây là khoảng cách giữa hai giao dịch liên tiếp, trung bình khoảng cách, hay số ngày từ giao dịch gần nhất đến ngày báo cáo.

Trong MBD-mini địa phương, giá trị và diễn biến theo tháng gợi ý một đại lượng liên quan đến độ gần đây của giao dịch, có thể tính bằng ngày. Đây chỉ là giả thuyết: không tái tạo được trường này bằng công thức recency thông thường từ log được cung cấp.

## Nguồn gốc

- Dataset card MBD gốc: https://huggingface.co/datasets/ai-lab/MBD — định nghĩa ngắn ở schema `targets`; MBD-mini giữ cùng cấu trúc và feature.
- Dataset card MBD-mini: https://huggingface.co/datasets/ai-lab/MBD-mini — cùng mô tả, không có công thức.
- Bài báo: https://arxiv.org/html/2409.17587v1 — nguồn bối cảnh bài toán và ranh giới lịch sử trước tháng mua; không cung cấp công thức đủ để tái tạo trường này.
- Notebook chính thức: https://github.com/Dzhambo/MBD/blob/main/scenario_mbd/notebooks/Preprocessing/GetMultimodalData.ipynb — đọc `targets` có sẵn, không tạo `diff_trans_date`. Phần Targets giữ bốn nhãn và có lệnh `.drop(*['event_time', 'trans_count', 'diff_trans_date'])`. Điều này xác nhận bước tiền xử lý này loại trường phụ trợ; không chứng minh lý do loại hoặc tính an toàn về thời gian của trường.

## Kiểm chứng trên toàn bộ bản MBD-mini địa phương

Phạm vi: `data/MBD-mini/targets/**/*.parquet`, 1.202.688 dòng; đối chiếu thời gian với `data/MBD-mini/detail/trx/**/*.parquet`, 38.764.930 giao dịch. Những số sau mô tả MBD-mini đang có, không phải phép đo trên toàn bộ MBD gốc.

| Kiểm tra | Kết quả |
|---|---:|
| Dòng có `diff_trans_date` | 1.081.902 |
| Dòng thiếu trường này | 120.786 |
| Min / max của giá trị có sẵn | 0 / 364 |
| Trung bình | 19,19688 |
| `trans_count=0` và `diff_trans_date IS NULL` | 120.786 |
| `trans_count>0` nhưng `diff_trans_date IS NULL` | 0 |
| Dòng `trans_count=1` | 50.460 |
| Trong đó `diff_trans_date>0` | 49.293 |

Giá trị vẫn có khi `trans_count=1` khiến cách hiểu “khoảng cách giữa hai giao dịch trong cùng tập giao dịch được đếm” không đủ giải thích dữ liệu. Tuy nhiên, chưa biết tập giao dịch được đếm nên không thể bác bỏ mọi biến thể khoảng cách giữa giao dịch.

Một khách có chuỗi `diff_trans_date` theo tháng là 11, 41, 72, 103 trong tháng 06–09/2022, trong khi `trans_count` đều bằng 2. Chênh lệch lần lượt 30, 31, 31 ngày phù hợp với giả thuyết recency tăng khi không có giao dịch mới. Nhưng trên toàn bảng, trong 151.618 cặp tháng liên tiếp có cùng `trans_count` và giá trị tháng trước không null, chỉ 34.432 cặp tăng đúng số ngày lịch giữa hai `mon`. Bằng chứng này không xác nhận một công thức chung.

## Thử tái tạo công thức

Với từng khách–tháng, đặt `cutoff = date_trunc('month', mon)`. Lấy giao dịch trong `[cutoff - 1 year, cutoff)`, rồi tính:

```sql
last_time = max(event_time)
candidate_1 = date_diff('day', cast(last_time as date), cutoff)
candidate_2 = candidate_1 - 1
```

| Kết quả | Số dòng |
|---|---:|
| Có giao dịch trong cửa sổ thử nghiệm | 1.105.850 |
| `diff_trans_date = candidate_1` | 88.859 |
| `diff_trans_date = candidate_2` | 205.177 |

Đây là số dòng khớp tuyệt đối trên tổng 1.202.688 dòng, không phải tỷ lệ xác nhận công thức. Null không được tính là khớp. Cả hai đều không tái tạo được trường nguồn. Kiểm tra bổ sung `trans_count = least(số giao dịch trong cửa sổ, 1000)` cũng chỉ khớp 130.537 dòng.

Ví dụ: một khách có `mon=2023-01-31`, `diff_trans_date=1`, trong khi giao dịch cuối trước cutoff trong log là `2022-12-26 07:38:59.293122`; khoảng cách theo ngày lịch đến `2023-01-01` là 6 ngày. Vì vậy không thể lấy tên cột để khẳng định công thức recency này.

Các nguyên nhân có thể gồm phạm vi giao dịch, cách lấy mẫu hoặc biến đổi thời gian khác nhau giữa bảng phụ trợ và log, hay một ngày tham chiếu khác. Chưa có bằng chứng để chọn nguyên nhân nào. Cửa sổ một năm là giả thuyết kiểm nghiệm, không phải quy tắc được tác giả xác nhận.

## Cách dùng cho dự án propensity

1. Ghi từ điển: “Chênh lệch thời gian liên quan giao dịch theo nhà phát hành; chưa xác minh công thức, đơn vị và thời điểm tính”.
2. Chưa dùng trực tiếp trường nguồn làm feature vì chưa xác minh nó chỉ dùng thông tin trước cutoff.
3. Nếu cần recency, tự tạo feature riêng tên `days_since_last_observed_trx`: lấy `max(event_time)` trên lịch sử `event_time < cutoff`, rồi tính chênh lệch ngày với cutoff. Định nghĩa rõ dùng ngày lịch hay số khoảng 24 giờ; không gọi feature tự tạo này là công thức gốc của `diff_trans_date`.
4. Không điền null thành 0: 0 là giá trị hợp lệ và có ý nghĩa khác “không có giá trị được cung cấp”. Có thể dùng chỉ báo thiếu và quy tắc imputation riêng khi mô hình cần.

Để xác nhận công thức gốc cần mã ETL tạo `targets` hoặc xác nhận của tác giả về ngày tham chiếu, hai toán hạng, đơn vị, cửa sổ, phạm vi giao dịch và thứ tự ẩn danh thời gian.
