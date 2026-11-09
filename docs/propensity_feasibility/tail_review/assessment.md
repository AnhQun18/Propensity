# Rà soát đuôi phân phối, khách nhiều giao dịch và đỉnh theo ngày

Ngày kiểm tra: **09/11/2026**. Dùng toàn bộ lịch sử hiện có, quy ước UTC đã thống
nhất. Đây là EDA; không xóa dòng/khách, không thay raw, chưa fit ngưỡng tiền xử lý
hoặc mô hình. Kết quả máy đọc: [summary.json](summary.json).

## 1. Amount theo currency

**Số dòng ngoài hàng rào IQR trên thang gốc không phản ánh tỷ lệ dữ liệu lỗi.** Với
phân phối lệch phải của amount, dùng riêng hàng rào Tukey trên thang gốc để quyết
định xóa/cap sẽ đánh dấu nhiều giao dịch lớn chưa được chứng minh là sai.

Đã tính lại quantile chính xác và cả hai đuôi theo từng currency. Với mã 11:

| Thang đo | Dưới hàng rào | Trên hàng rào | Tổng ngoài hàng rào | Tỷ lệ trên 38.724.240 dòng |
|---|---:|---:|---:|---:|
| Amount gốc | 0 | 5.521.385 | 5.521.385 | 14,2582% |
| ln(amount) | 516.970 | 97.222 | 614.192 | 1,5861% |
| log1p(amount) | 486.868 | 97.271 | 584.139 | 1,5085% |

Tất cả amount trong snapshot đều hữu hạn và dương; `ln` và `log1p` đều xác định.
IQR sau log vẫn chỉ là quy tắc mô tả, không xác nhận các dòng bị đánh dấu là lỗi.
P99 mã 11 là **5.257.531,795**, P99,9 là **57.322.644,544** theo đơn vị amount gốc;
38.725 dòng vượt P99,9. Đây là ngưỡng tham chiếu EDA toàn kỳ, không đem nguyên ngưỡng
này vào split train/validation/test trong quá khứ.

Mã 9 có 2 dòng, 5 có 6 dòng, 16 có 7 dòng, 12 có 4 dòng: gom thành
`other_currency` (**19 dòng**) trong bảng phân tích và không diễn giải hàng rào IQR
cho nhóm này. Giữ `missing_currency` (**118 dòng**) riêng. Hàm
`currency_analysis_group` trong [currency.py](../../../src/mbd_mini_eda/currency.py)
giữ nguyên mã gốc và phân biệt grouping với cờ ngoại tệ. Việc gom nhóm không quy
đổi các đồng tiền về cùng đơn vị; không cộng amount của `other_currency`, các mã
ngoại tệ hoặc nhóm thiếu như thể chúng cùng tiền tệ. Thống kê nhóm thiếu chỉ mô tả
những dòng chưa biết mã, không phải phân phối của một đồng tiền đã xác minh.

Bảng đầy đủ: [amount_log_iqr_by_currency.csv](amount_log_iqr_by_currency.csv).

## 2. Giao dịch lớn nhất mã 11

- Giá trị lưu trong raw: **1.370.187.366.400**, xuất hiện **một dòng** trên toàn bộ
  trx của mã 11. Không gán đây là VND, RUB hay đơn vị tiền thật đã xác minh.
- Khách có tiền tố ID `dcc03690dc8e`; timestamp gốc được diễn giải theo UTC:
  `2021-04-02 20:43:11.142052`; `event_type=14`, `event_subtype=12`.
- Khách có **56 giao dịch**, tất cả mã 11. Median của chính khách là
  **164.539.424.768**; max/median = **8,3274 lần**.
- **53/56 giao dịch** của khách vượt P99,9 toàn kỳ của mã 11. Vì vậy giá trị max
  duy nhất không đồng nghĩa với một điểm cao đơn lẻ trong lịch sử có mức tiền bình
  thường. Chưa chứng minh được lỗi, tài khoản hệ thống hay nguyên nhân của quy mô này.

**Chưa thỏa điều kiện cap tự động dựa trên giả thuyết vài dòng đơn lẻ.** Cap tất cả
giao dịch khách này ở ngưỡng tham chiếu nói trên sẽ làm tổng từ khoảng **13,77 nghìn
tỷ** xuống **3,04 tỷ** đơn vị amount gốc. Đây là minh họa tác động của cap, chưa
phải thay đổi feature được áp dụng. Giữ raw; so sánh median/quantile/log và phiên
bản tổng hợp có cap trên validation nếu cần. Ngưỡng cap theo currency phải được
ước lượng chỉ từ train phù hợp cutoff, đồng thời giữ cờ bị cap.

CSV cục bộ sau khi chạy script: `outputs/tail_review_20261109/maximum_amount_transactions.csv`
chứa dòng max và ID đầy đủ; `maximum_amount_owner_stats.csv` chứa thống kê của khách.
Median tính trên DOUBLE của các giá trị raw đã lưu; chuyển kiểu không phục hồi độ
chính xác đã mất do dtype FLOAT của nguồn.

## 3. Khách có nhiều giao dịch và tỷ lệ nhãn

Đã kiểm tra **top 50** theo số dòng raw trx, đối chiếu số đếm cache với raw, loại
giao dịch/subtype, ngày đầu/cuối, số ngày hoạt động và đủ **12 tháng nhãn mỗi khách**.
Số giao dịch ở đây là số log trong trx, không phải `trans_count` phụ trợ trong targets
và chưa khẳng định mỗi log tương ứng một giao dịch kinh doanh độc lập.

Top 20 có **2.214.115 log**, chiếm **5,7116%** tổng 38.764.930 log. Nhãn dùng cho
yêu cầu này là `y_any = OR(target_1, target_2, target_3, target_4)`. Tỷ lệ của mỗi
khách bằng **số tháng y_any=1 / 12 tháng có nhãn**, không phải số sản phẩm mua chia
cho số giao dịch. Chỉ hạng **14** và **16** có một tháng dương mỗi khách (**8,33%**);
18 khách còn lại có tỷ lệ 0%. Hai tháng dương đều ở `target_1`, các target khác của
top 20 đều bằng 0. Gộp nhóm là **2/240 = 0,8333% khách–tháng**, tương ứng 2/20 khách
từng có nhãn dương; hai mẫu số trả lời hai câu hỏi khác nhau.

Khách đứng đầu có **1.020.773** log trải trên **731 ngày hoạt động**; cặp
`event_type=1, event_subtype=12` chiếm **87,83%**. Bốn khách đầu đều có cặp mã này
chiếm khoảng 81,79–96,47%. Đây là các mã đã ẩn danh, chưa thể gọi là chuyển khoản,
bot hay tài khoản hệ thống từ mã đơn thuần. Không dùng nhãn tương lai để loại khách
khỏi train hoặc gắn cờ heavy tại một cutoff quá khứ.

MBD được tác giả mô tả là dữ liệu **khách doanh nghiệp**. Vì vậy giả định toàn bộ
tệp là khách cá nhân không phù hợp nguồn công bố; không loại các khách nhiều giao
dịch chỉ vì quy mô lớn hoặc không có nhãn dương. Nếu phạm vi triển khai chỉ dành
cho khách cá nhân, cần nguồn xác định phân khúc khách trước khi quyết định cohort,
không suy diễn phân khúc từ tần suất. Nguồn:
[bài báo MBD](https://arxiv.org/html/2409.17587v2).

CSV cục bộ:

- `outputs/tail_review_20261109/top20_trx_customers.csv`: thứ hạng, ID đủ 64 ký tự,
  ID rút gọn, số log, số tháng dương/tỷ lệ any-product và từng nhãn nguồn, thời gian
  hoạt động, loại giao dịch chiếm ưu thế.
- `top50_trx_customers.csv` và `top50_event_type_subtype.csv` trong cùng thư mục:
  kiểm tra mở rộng top 50 và toàn bộ phân bố cặp mã của từng khách.

Xếp hạng toàn lịch sử chỉ phục vụ mô tả này, không phải quy tắc chọn cohort train.
Chưa có bằng chứng cần loại bất kỳ khách top 50 nào.

## 4. Count theo nguồn và đỉnh ngày

P99 tính trên **100.224 khách**, gồm cả khách có count bằng 0:

| Nguồn | Median count | P99 | Khách > P99 | Max |
|---|---:|---:|---:|---:|
| trx | 164 | 2.233,54 | 1.003 | 1.020.773 |
| geo | 235 | 5.082,54 | 1.003 | 39.487 |
| dialog | 0 | 21 | 892 | 114 |

Nguồn: [customer_count_reference.csv](customer_count_reference.csv). Vì có ties,
P99 không đảm bảo đúng 1.003 khách vượt ở mọi nguồn; dialog chỉ có **892** khách >21.
Đếm bằng 0 nghĩa là không thấy log trong phạm vi đang xét, không chứng minh đã
quan sát đầy đủ mọi hoạt động của khách.

Giữ nhóm geo vượt IQR và nhóm dialog >7,5; không gọi 8.007 khách dialog bị đánh dấu
theo thang gốc là dữ liệu lỗi. Dùng `log1p(trx_count)`, `log1p(geo_count)`,
`log1p(dialog_count)`, `has_dialog` và số ngày hoạt động. P99 là tham chiếu đuôi
phân phối dễ đọc hơn; `is_heavy_trx_user` dự kiến dùng count >P99 của train tương
ứng cùng cửa sổ lịch sử, không lấy P99 toàn kỳ này cho mô hình quá khứ.

Đã đọc raw quanh đỉnh theo **UTC**:

| Ngày | Log trx | Khách có log |
|---|---:|---:|
| 10/12/2022 | 64.636 | 32.611 |
| 11/12/2022 | **65.302** | **32.664** |
| 12/12/2022 | 63.228 | 32.157 |

Riêng 11/12: khách nhiều log nhất có **1.196** log (**1,8315%**), top 10 chiếm
**3,9157%**, top 20 chiếm **4,2204%**; median là **1 log/khách**. Đỉnh có nhiều
khách tham gia, không do vài khách cực lớn chi phối. Giữ ngày này; có thể thử đặc
trưng thứ trong tuần/tháng trên timestamp được phát hành và kiểm định bằng temporal
validation. Chưa xác nhận nguyên nhân trả lương, khuyến mãi hay cuối năm. Tác giả
mô tả có làm nhiễu trường ngày/số khi ẩn danh, nên không gán nguyên nhân lịch thực
tế chỉ từ ngày công bố. Nguồn: [mục ẩn danh của MBD](https://arxiv.org/html/2409.17587v2#S3.SS3).

Chi tiết: [peak_day_neighbors.csv](peak_day_neighbors.csv),
[peak_day_event_types.csv](peak_day_event_types.csv); danh sách top 50 khách trong
ngày nằm cục bộ ở `outputs/tail_review_20261109/peak_day_top50_customers.csv`.

## 5. Chính sách dùng cho bước tạo đặc trưng

| Việc | Cách áp dụng |
|---|---|
| Xóa dòng/khách | Chỉ sau khi có bằng chứng lỗi hoặc xác định ngoài quần thể mục tiêu; hiện chưa xóa gì. |
| Raw và mô hình cây | Giữ raw; có thể thử feature nguyên thang cho cây. Không khẳng định XGBoost hoàn toàn miễn nhiễm cực trị. |
| Biến đổi | Thêm log1p count và amount/tổng hợp không âm; rank/chuẩn hóa cho mô hình cần thiết. Rank/quantile mapping fit trên train, áp mapping cố định sang validation/test. |
| Cap | Thử P99/P99,9 cho feature tổng hợp hoặc mô hình nhạy cực trị; fit ngưỡng theo currency chỉ trên train, giữ bản raw và cờ vượt ngưỡng; chọn bằng validation. |
| Thống kê bền vững | Ưu tiên bổ sung median, quantile và active days; không thay mọi sum/mean nếu chưa so sánh chất lượng. |
| Cờ | `is_heavy_trx_user`, `has_dialog`, `has_foreign_currency`, `has_missing_currency`; currency 11 là mã chính theo quy ước dự án. |

Với `has_foreign_currency`, có ít nhất một mã khác 11 đã biết thì bật cờ; khách chỉ
có currency thiếu thì để unknown, kèm cờ missing. Nếu có mã đã biết nhưng vẫn có
dòng thiếu, cờ phản ánh ngoại tệ **đã quan sát**, giữ thêm missing để thể hiện giới
hạn này. Không ép currency null thành ngoại tệ. Không cộng các amount khác currency
khi chưa có tỷ giá/đơn vị xác minh.

Cây chia theo ngưỡng nên ít phụ thuộc thang đo của từng feature hơn mô hình dựa
trên khoảng cách; tuy nhiên cực trị vẫn có thể chi phối sum/mean trước khi đưa vào
cây, và các nhóm ít mẫu vẫn có thể bị overfit. Biến đổi log từng giao dịch trước
khi tổng hợp không tương đương log của tổng. XGBoost còn có tree method, histogram
binning và regularization; tài liệu tham số là cơ sở cho việc phải kiểm định cấu
hình, không phải bảo đảm miễn nhiễm:
[tài liệu XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).

Các cờ/count/log ở cấp khách nêu trên là thiết kế cho bước tạo features, **chưa
được fit hoặc tích hợp vào một pipeline mô hình**. Phần đã triển khai là audit
và hàm grouping/cờ currency ở cấp dòng.

## 6. Cách chạy và giới hạn xác minh

```powershell
.\.venv\Scripts\python.exe -B -X utf8 src/audit_tail_cases.py
```

Script kiểm tra inventory **3.820 file** theo kích thước/mtime và fingerprint trước
khi dùng cache đếm; đối chiếu top 50 với raw, đọc raw để tính amount, loại giao
dịch và đỉnh ngày, đọc targets để tính tỷ lệ nhãn. Fingerprint là của inventory,
không phải hash nội dung từng file. Các quantile dùng nội suy tuyến tính; vượt
ngưỡng là dấu `>` nghiêm ngặt. Tỷ lệ IQR log tính cả hai đuôi.

Đã xác minh số đếm/nhãn, sự lặp lại của max, median khách, log-IQR và độ tập trung
ngày đỉnh. Các điều chưa xác minh bằng dữ liệu hiện có: định danh tiền thật/biến
đổi amount gốc; ý nghĩa nghiệp vụ mã loại giao dịch; nguyên nhân kinh tế của khách
quy mô lớn và ngày cao điểm; hiệu quả của cap/log trên mô hình. Không dùng các mục
này như sự thật để xóa dữ liệu hoặc kết luận bot.
