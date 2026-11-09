# Kiểm chứng ba công thức diff_trans_date — 09/11/2026

Đã đối chiếu **toàn bộ 1.202.688 dòng targets, 100.224 khách và 38.764.930 giao dịch
detail/trx**, xử lý lần lượt cả 5 fold. Không sửa dữ liệu gốc hoặc huấn luyện mô hình.
**Không có cấu hình nào trong 1.134 cấu hình thử nghiệm tái tạo hoàn toàn cột nguồn.**

Đây là kiểm tra công thức trên giao dịch được cung cấp trong detail/trx. Chưa biết
nhà phát hành có dùng tập giao dịch khác, cách biến đổi khác hoặc ngày tham chiếu khác
để tạo hai cột phụ trợ hay không. Giao dịch detail cũng không tự đồng nhất một lần
mua sản phẩm có nhãn. Không suy từ phép thử này ra nguyên nhân cụ thể của sai khác.

## Phạm vi các cấu hình thử nghiệm

- Ba mốc tham chiếu: đầu tháng chứa mon; mon lúc 00:00; đầu tháng kế tiếp.
- Hai cách diễn giải mốc: raw UTC-midnight và Moscow-midnight (trừ 3 giờ để so sánh
  với timestamp UTC). Hai mốc nằm trong/sau tháng nhãn chỉ dùng chẩn đoán, không phải
  khuyến nghị dùng chúng làm cutoff dự báo đầu tháng.
- Bảy cửa sổ trước mốc tham chiếu: 30, 84, 90, 180, 365 ngày, một năm lịch, toàn bộ
  lịch sử được cung cấp. Cửa sổ gồm biên trái và loại biên phải.
- Khoảng thời gian thực theo ngày = số nanosecond / (86.400 × 10^9); thêm biến thể
  chênh lệch ngày lịch theo timezone đang thử.
- Thử giá trị chưa làm tròn, floor, ceil và làm tròn half-up khi phù hợp. Chênh lệch
  ngày lịch ở recency có thêm biến thể trừ 1 để kiểm tra quy ước đếm ngày.
- Công thức trung bình thử cả mẫu số `trans_count-1` của nguồn và `n_observed-1` tự đếm.

Tổng cộng 16 biến thể trung bình, 6 recency và 5 khoảng cách hai giao dịch cuối;
27 × 3 mốc × 2 timezone × 7 cửa sổ = **1.134 cấu hình**. Trường hợp không xác định
được tính null; tỷ lệ khớp dưới đây dùng mẫu số **1.081.902 dòng có diff nguồn**, không
cộng các trường hợp cùng null vào tỷ lệ đó. Sai số so sánh numerical là 10^-9 ngày.

## 1. Trung bình khoảng cách giữa các giao dịch

Giả thuyết người dùng:

```text
diff_trans_date = (last_transaction_date - first_transaction_date) / (trans_count - 1)
```

Với cùng một tập giao dịch đã sắp thứ tự, tổng khoảng cách liên tiếp bằng last−first;
vì vậy công thức đúng về toán học khi count đúng là số giao dịch trong tập đó và n≥2.
Với n=0, vấn đề là không có first/last (mẫu số −1 tự nó không phải phép chia cho 0).
Với n=1, mẫu số bằng 0 và không có khoảng cách để lấy trung bình.

**Phản ví dụ trực tiếp trên dữ liệu nguồn:**

| Trường hợp | Số dòng |
| --- | ---: |
| trans_count=1 | 50.460 |
| Trong đó diff_trans_date > 0 | 49.293 (97,6873%) |
| Trong đó diff_trans_date = 0 | 1.167 |
| Trong đó diff_trans_date IS NULL | 0 |

Do đó cách hiểu công thức trên với n=1 trả null **hoặc** 0 đều không giải thích được
49.293 dòng dương. Điều này bác bỏ cách diễn giải nguyên trạng trong giả thuyết,
không chứng minh trans_count của nguồn chắc chắn đếm cùng tập giao dịch detail.

Trong các phép thế trực tiếp vào công thức, cấu hình khớp nhiều nhất dùng khoảng
thời gian thực giữa first/last trong 30 ngày trước đầu tháng Moscow, chia cho
`trans_count-1`, rồi floor: **269.172/1.081.902 = 24,8795%**.
Tuy nhiên, số đếm giao dịch của cửa sổ thường khác trans_count nguồn, nên đây không
thể tự được gọi là trung bình đúng của cùng tập giao dịch. Trong tập con của chính
cấu hình đó có số đếm khớp nguồn và tính được hai vế, chỉ **1.008/13.738 = 7,3373%**
khớp diff. Nếu thay mẫu số bằng số giao dịch thực sự quan sát được, cấu hình trung
bình tốt nhất chỉ khớp **182.973/1.081.902 = 16,9122%**.

**Kết luận:** chưa tái tạo được diff nguồn bằng Average Inter-transaction Time.
Không nên gọi cột nguồn là average inter-purchase time theo công thức này.

## 2. Recency: ngày chốt trừ lần giao dịch gần nhất

Đã tính last từ lịch sử trước từng mốc, rồi thử chênh lệch thời gian thực/ngày lịch
và các cách làm tròn. Cấu hình khớp nhiều nhất trong các cấu hình recency là:

```text
reference = đầu tháng chứa mon, so sánh trực tiếp trên timestamp raw (offset 0)
last = giao dịch cuối trong toàn bộ lịch sử trước reference
candidate = floor((reference - last) / 24 giờ)
```

Khớp **205.177/1.081.902 = 18,9645%** dòng có diff nguồn. Với cutoff Moscow và cùng
biến thể floor, số khớp là **204.974/1.081.902 = 18,9457%**. Vì vậy thay timezone
3 giờ không làm công thức này tái tạo được cột nguồn.

Đối với cách hiểu “một ngày chốt cố định và cùng một giao dịch cuối toàn lịch sử
cho mọi tháng của khách”, dự báo sẽ không đổi theo tháng. Nhưng **95.516/100.224 khách**
có ít nhất hai giá trị diff không-null khác nhau giữa các tháng. Cách hiểu cố định
đó không giải thích được bảng; recency theo snapshot tháng vẫn là giả thuyết riêng.

**Kết luận:** recency là feature hợp lệ nếu tự định nghĩa và chỉ lấy quá khứ, nhưng
các công thức đã thử không xác nhận diff_trans_date nguồn chính là recency.

## 3. Khoảng cách với giao dịch ngay trước

Bảng chứa diff_trans_date là **targets, mỗi dòng một khách–tháng**; không phải bảng
giao dịch. Vì vậy phép `event_time - lag(event_time)` theo khách không áp trực tiếp
trên từng dòng targets.

Để có phép đối chiếu tương ứng, lấy **hai giao dịch cuối** trong mỗi cửa sổ trước
mốc tham chiếu, tính khoảng cách của chúng rồi gán về khách–tháng. Cấu hình khớp nhiều
nhất dùng toàn quá khứ trước đầu tháng Moscow và floor số khoảng 24 giờ:
**203.069/1.081.902 = 18,7696%** dòng có diff nguồn.

**Kết luận:** khoảng cách giữa hai giao dịch liên tiếp có thể tính ở cấp event, nhưng
phiên bản “hai giao dịch cuối trước snapshot” không tái tạo được diff nguồn.
Chưa có quy tắc chọn/gộp event do nhà phát hành công bố để kiểm tra một cách gán khác.

## Đối chiếu và quyết định

| Giả thuyết | Số khớp cao nhất trong cấu hình đã thử | Tỷ lệ trên 1.081.902 diff có sẵn | Đánh giá |
| --- | ---: | ---: | --- |
| (last−first)/(trans_count nguồn−1) | 269.172 | 24,8795% | Còn mâu thuẫn trực tiếp khi trans_count=1 và thường không nhất quán số đếm |
| Recency | 205.177 | 18,9645% | Không tái tạo cột nguồn; đổi timezone không khắc phục |
| Khoảng cách hai giao dịch cuối trước snapshot | 203.069 | 18,7696% | Là phép gán event về khách–tháng để chẩn đoán; không khớp nguồn |

Các tỷ lệ là độ khớp giá trị trên cùng mẫu số, không phải xác suất giả thuyết đúng,
metric dự báo hoặc kết quả ngoài mẫu. Chọn cấu hình khớp nhất trong nhiều phép thử
không xác minh được công thức gốc. Các trường hợp cùng null và giá trị 0 do làm tròn
cũng không đủ chứng minh ý nghĩa cột.

Giữ mô tả nguồn thận trọng; chưa dùng hai cột phụ trợ làm predictor khi chưa biết
cửa sổ và thời điểm tính. Có thể tự tạo ba feature riêng từ detail trước cutoff:
`mean_observed_trx_interval_days`, `days_since_last_observed_trx` và
`last_two_observed_trx_gap_days`, với quy tắc null cho lịch sử không đủ sự kiện.

## Tái lập và kết quả máy đọc

```powershell
.\.venv\Scripts\python.exe -B -X utf8 src/audit_three_diff_formulas.py
```

- [Script](../../../src/audit_three_diff_formulas.py).
- [summary.json](summary.json): quy mô, trường hợp trans_count=1 và cấu hình tốt nhất.
- [candidate_results.csv](candidate_results.csv): đủ 1.134 cấu hình, số khớp/null và
  tập con có số đếm giao dịch khớp nguồn.
- [fixed_reference_diagnostic.json](fixed_reference_diagnostic.json): số khách có diff
  thay đổi qua tháng, dùng kiểm tra giả thuyết ngày chốt cố định.
