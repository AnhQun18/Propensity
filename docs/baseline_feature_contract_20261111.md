# Hợp đồng feature baseline sau rà soát — 11/11/2026

Tài liệu chốt quy tắc tạo biến cho nghiên cứu tiếp theo. Số liệu thí nghiệm hiện
hành nằm tại [báo cáo tổng hợp](../deliverables/baseline-final/REPORT.md). Những
cửa sổ/top-K/PCA dưới đây là đặc tả nghiên cứu, chưa đồng nghĩa đã triển khai.

## Đơn vị và thời điểm

- Một dòng là `(client_id, cutoff)`. ID, fold và cutoff dùng quản lý dữ liệu,
  không tự động vào X.
- Nhãn là mua ít nhất một trong bốn sản phẩm trong tháng chứa cutoff. Lịch sử
  sự kiện tuân thủ `[cutoff - W, cutoff)`, hoặc mọi sự kiện trước cutoff.
- UTC là quy ước nghiên cứu của dự án. Nhãn quá khứ chỉ dùng sau
  `assumed_label_available_at`, chưa phải SLA nguồn đã xác minh.
- Split chính là cuốn chiếu theo tháng cho khách hiện hữu. Split theo client
  là bài kiểm tra riêng cho khách chưa thấy; mọi bước fit biểu diễn phải tuân
  thủ cùng phạm vi ID/thời gian nếu dùng protocol này.

## Cửa sổ và thống kê

| Nhóm | Quy tắc |
| --- | --- |
| Trx count/active days | Đếm log có khóa/thời gian hợp lệ, không yêu cầu đủ mọi thuộc tính phụ |
| Trx amount | Thống kê trong từng currency; baseline ưu tiên mã 11, currency null giữ unknown |
| Mean/std/max | Không có giá trị hợp lệ thì NaN; std dưới hai giá trị thì NaN, khóa `ddof=1` |
| Recency | Khoảng từ log gần nhất hợp lệ đến cutoff; không có log thì NaN |
| Inter-event gap | Tính trong cửa sổ đã chọn sau sắp timestamp; dưới hai sự kiện thì không có gap; std gap cần ít nhất hai gap |
| Trend | Ví dụ `count_30/30 - count_previous_60/60`, có quy tắc độ phủ; không coi count khác độ dài cửa sổ là rate tương đương |
| Cùng tháng năm trước | Tháng lịch năm trước tương ứng với khoảng tổng hợp đã chỉ định; không coi thiếu phạm vi nguồn là không hoạt động |
| Top-K categorical | Fit vocabulary trên train quá khứ hợp lệ; ngoài vocab gom OTHER, null giữ token riêng; khóa cửa sổ cho từng cột |
| Geo diversity | Nunique của ba cột geohash đã mã hóa; không lấy prefix số hoặc suy ra tọa độ |
| Geo entropy/top-1 | Chốt geohash_6, cửa sổ 30 ngày; entropy log tự nhiên, tỷ lệ trên số sự kiện có mã hợp lệ; mẫu số 0 thì NaN |
| Geo new-cell fraction | Trong 30 ngày gần nhất, số sự kiện thuộc ô chưa thấy trong 180 ngày trước cửa sổ đó / số sự kiện có ô hợp lệ; thiếu phạm vi so sánh thì đánh dấu/giữ unknown |
| Dialog last/mean | Chốt phạm vi pooling trước khi fit PCA, ví dụ 330 ngày trước cutoff; không tự coi 330 ngày là một năm |

Mốc bắt đầu/kết thúc nguồn mô tả phạm vi trích xuất, không chứng minh thời gian
quan sát liên tục mỗi khách. Count 0 chỉ mô tả không có log trong phạm vi được
đếm. Giữ flags sự kiện/độ phủ phục vụ xử lý thiếu và báo cáo cohort. Kiểm tra
riêng các biến hằng trong tháng nhưng đổi theo thời gian trước khi đưa vào X.

## Feature lịch sử mua đã triển khai

Mã nguồn: [ablation.py](../src/propensity_baseline/ablation.py). Tổng **15 biến**:

1. `y_any` tại tháng M−1, M−2, M−3: 3 biến.
2. Bốn target tại M−1: 4 biến.
3. Số tháng có nhãn, số tháng dương và tỷ lệ dương trong ba tháng: 3 biến.
4. Ba thống kê tương ứng trong toàn lịch sử nhãn đã phát hành: 3 biến.
5. Số tháng từ tháng dương gần nhất và cờ có nhãn M−1: 2 biến.

Không lấy tháng gần nhất còn lại thay cho tháng bị thiếu. Số tháng có nhãn có
thể bằng 0; các tỷ lệ/số dương không xác định giữ NaN. Không có lần dương trước
cutoff thì recency mua là NaN. Cả H và B+H dùng cùng 15 biến. Độ dài lịch sử có
nhãn cũng có thể đại diện cho thời gian; không tuyên bố mô hình hoàn toàn không
thể suy ra tháng sau khi bỏ metadata độ phủ nguồn.

## PCA, CoLES và chọn biến

Fit top-K, scaler, PCA, encoder và chọn biến trong phạm vi train được phép.
Đối với mô phỏng một artifact cố định dùng cho mọi snapshot, có thể đóng băng
artifact từ trước cutoff đầu tiên. Nếu refit theo fold, không cho artifact nhìn
sau cutoff đánh giá. Ghi rõ protocol pretraining thay vì chỉ ghi “không có nhãn”.

Không fit PCA trên embedding thiếu được điền 0. Biểu diễn thiếu giữ NaN sau
transform và kèm flag. Thử số chiều bằng validation, không bắt buộc 32–64 vì
một quy tắc số mẫu/biến. PCA tối ưu phương sai, chưa bảo đảm giữ tín hiệu mua.

Đánh giá tăng thêm từng nhóm biến bằng cùng cohort, split và ngân sách. Gain
không tự chứng minh nhóm biến vô ích khi nhiều cửa sổ tương quan. Chọn feature
bằng train/validation, không lọc sau khi nhìn test. Số biến cuối cùng xuất từ
schema, không lấy tổng ước tính trong mô tả làm hợp đồng thực thi.
