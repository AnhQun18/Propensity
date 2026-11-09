# Các nghiên cứu xử lý currency trong dữ liệu giao dịch

Ngày đối chiếu: 07/11/2026. Trọng tâm: MBD/MBD-mini; phân biệt nội dung bài báo, cấu hình công khai và đề xuất cho dự án.

**Quyết định người dùng sau đối chiếu:** mã 11 là mã chính; mã khác có giá trị mang
cờ ngoại tệ của dự án; currency null có cờ missing riêng và không bị gán ngoại tệ.
Đã thêm hàm tạo cờ trong `src/mbd_mini_eda/currency.py`. Audit raw xác nhận
38.724.240 / 40.572 / 118 giao dịch tương ứng ba nhóm; xem
[kết quả mới](remaining_questions/assessment.md#2-mã-11-là-mã-chính-mã-khác-flag-ngoại-tệ).
Đây là quy ước phân nhóm được chấp nhận, không cần mapping ISO để tạo cờ; không khẳng
định danh tính đồng tiền hoặc đơn vị amount gốc.

**Bổ sung ngày 09/11/2026:** đối chiếu lại mục 3.3 bài MBD xác nhận categorical đổi
thành chỉ số, numerical và ngày có thêm nhiễu, tham số ẩn danh không được công bố.
Vì vậy không suy ngược mã tiền hoặc tiền gốc bằng tỷ giá thị trường; chưa có mapping
để gọi mã 11 là RUB/USD/VND. [Kết quả xác minh và nguồn mã](open_questions/answers.md)
phân biệt thông tin công bố với quy ước triển khai của đề tài.

## Kết quả chính

Trong ba nghiên cứu trực tiếp trên MBD/MBD-mini đã kiểm tra, chưa tìm thấy mô tả quy đổi amount theo tỷ giá về một đồng tiền chung. Có bằng chứng về ba cách biểu diễn khác nhau: không đưa currency vào encoder, ghép currency vào token giao dịch, hoặc chỉ thêm token cho currency hiếm. Không được suy từ việc không mô tả tỷ giá thành khẳng định toàn bộ ETL của tác giả chưa từng quy đổi.

## 1. Benchmark MBD gốc

[Bài báo v2, mục 3.3 và 5.1](https://arxiv.org/html/2409.17587v2) mô tả mã hóa categorical thành chỉ số, thêm nhiễu vào numerical, và dùng identity/log cho numerical trong mô hình. Không nêu quy tắc FX.

Đối chiếu cấu hình công khai:

| Cấu hình | Amount | Currency |
|---|---|---|
| [CoLES](https://github.com/Dzhambo/MBD/blob/main/scenario_mbd/conf/trx_coles.yaml) | `numeric_values: amount: log` | Không có trong danh sách embedding |
| [Aggregation](https://github.com/Dzhambo/MBD/blob/main/scenario_mbd/conf/trx_agg.yaml) | `amount: log`, `was_logified: false`, `log_scale_factor: 1` | Không có trong danh sách cấu hình feature |
| [Supervised RNN](https://github.com/Dzhambo/MBD/blob/main/scenario_mbd/conf/trx_supervised.yaml) | `numeric_values: amount: log` | Không có trong danh sách embedding |
| [TabBERT](https://github.com/Dzhambo/MBD/blob/main/scenario_mbd/conf/trx_tabbert.yaml) và [TabGPT](https://github.com/Dzhambo/MBD/blob/main/scenario_mbd/conf/trx_tabgpt.yaml) | Embedding categorical `amount: in: 11, out: 24` | Không có trong danh sách embedding |

Giới hạn: cấu hình `in: 11` chỉ cho biết kích thước bảng embedding amount, không đủ xác định cạnh bin hoặc số bin thực nếu có token đặc biệt. Không khẳng định mọi thí nghiệm hoặc bước ETL của MBD đều loại currency.

## 2. PANTHER

[PANTHER v2, mục 3.2 và Appendix B.3](https://arxiv.org/html/2510.10102v2#A2.SS3): với MBD-mini, token hành vi kết hợp amount, currency, event type và thuộc tính nguồn/đích; vocabulary gồm 40.791 token. Phương pháp chung rời rạc hóa amount. Phần MBD-mini không công bố cạnh bin theo currency hay quy tắc tỷ giá. Ví dụ khoảng USD trong phần phương pháp không phải bảng quy đổi mã MBD. [Repo chính thức](https://github.com/WeChatPay-Pretraining/PANTHER) giới thiệu pipeline CCT; chưa xác minh được ETL MBD-mini từ đó.

## 3. EventGPT

[EventGPT v1, mục 5.2](https://arxiv.org/html/2611.03302v1#S5.SS2): amount vào 21 bin quantile, fit trên các fold pretraining; chỉ thêm token `cur` nếu khác currency chiếm 99,9% giao dịch. Không nêu FX hoặc quantile riêng từng currency. Đây là xử lý biểu diễn, chưa chứng minh amount đã cùng đơn vị. Bài này là preprint tháng 11/2026.

## Đối chiếu bản dữ liệu trong dự án

Từ bảng thống kê toàn bộ giao dịch đã có tại `docs/mbd_mini_eda/tables/review/amount_by_currency.csv`:

- Tổng: 38.764.930 giao dịch.
- Mã `11`: 38.724.240 giao dịch, **99,895034%** tổng số.
- Mã khác hoặc null: 40.690 giao dịch, khoảng 0,104966%.
- Currency null: 118 giao dịch.

Không gán mã `11` thành RUB/USD hoặc đồng tiền cụ thể: tài liệu đã kiểm tra chưa cung cấp mapping để xác nhận. Tỷ trọng lớn theo số giao dịch cũng không bảo đảm tỷ trọng lớn theo tổng amount, vì giao dịch hiếm có thể mang giá trị lớn.

## Đề xuất cho dự án — không phải kết luận của tác giả các bài

**Nếu dựng bảng feature propensity:**

1. Giữ currency dưới dạng categorical. Tạo số lượng, tỷ lệ giao dịch và recency theo currency; có nhóm missing riêng.
2. Tính sum/mean/median/max amount riêng theo currency và cửa sổ trước cutoff. Với mã quá hiếm, có thể chỉ giữ count hoặc dùng regularization; không cộng amount của các mã hiếm vào một tổng tiền có ý nghĩa kinh tế.
3. Có thể thử log amount hoặc percentile theo currency để giảm ảnh hưởng phân phối lệch. Fit cạnh bin/scaler trên train; nhóm ít dữ liệu cần quy tắc fallback. Kết quả percentile là vị trí tương đối trong nhóm, không phải sức mua hay số tiền quy đổi.
4. So sánh baseline không amount, amount của mã phổ biến, và amount theo từng currency trên cùng split. Quyết định bằng metric trên validation thay vì mặc định feature tiền luôn hữu ích.

**Nếu dựng mô hình chuỗi:** thử embedding currency cùng log amount hoặc amount bin; có thể tham khảo cách tiết kiệm token của EventGPT. Muốn học tương quan amount–currency phải cung cấp cả hai trong biểu diễn hoặc chuẩn hóa có điều kiện rõ ràng.

Chỉ quy đổi FX khi xác minh được mapping mã tiền, đơn vị và phép biến đổi amount, cùng ngày tỷ giá phù hợp. Hiện chưa đủ căn cứ để tự gắn tỷ giá thị trường vào amount đã ẩn danh. Không dùng `sum(amount)` xuyên currency để diễn giải là “tổng chi tiêu/tổng doanh số” khi chưa xác nhận tính so sánh của đơn vị.

Log và quantile giúp biểu diễn numerical thuận lợi hơn; chúng không tự đưa các đồng tiền về cùng một đơn vị.
