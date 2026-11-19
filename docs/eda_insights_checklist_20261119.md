# Insight và checklist EDA cho propensity — 19/11/2026

## Phạm vi và bằng chứng

Mục tiêu: xác suất mua ít nhất một trong bốn sản phẩm có nhãn (`y_any`). Đơn vị là khách–tháng, không phải giao dịch. Dữ liệu gốc có 100.224 khách, 1.202.688 khách–tháng, 12 tháng nhãn từ 02/2022 đến 01/2023. Không dùng mẫu 1.000 khách để suy rộng.

Lần này chạy mới kiểm tra khóa, null khóa/thời gian, orphan, tổng hợp lại số sự kiện theo khách–tháng Moscow trên cả ba nguồn raw, đối soát với cache và bảng panel, đọc lại nhãn raw, tính phân phối amount theo currency. Sau đó dựng lại các feature count toàn lịch sử và DIALOG gần nhất theo tháng; xuất thẻ phân khúc có quy mô, số mua, tỷ lệ, đóng góp người mua, lift và biến động lift từng tháng.

- [Script tái lập](../src/scripts/audit_insights_checklist.py).
- [Kết quả kiểm tra](../outputs/eda_insights_20261119/validation.json), [audit join](../outputs/eda_insights_20261119/join_audit.csv).
- [Thẻ phân khúc gộp](../outputs/eda_insights_20261119/segment_cards.csv), [thẻ theo tháng](../outputs/eda_insights_20261119/monthly_segment_cards.csv).
- [Phân phối amount](../outputs/eda_insights_20261119/amount_by_currency.csv), [nhãn 12 tháng](../outputs/eda_insights_20261119/label_monthly.csv).

Các kết quả tương tác TRX×DIALOG, lịch sử mua và ma trận tương quan được tái sử dụng từ báo cáo hiện có, ghi rõ nguồn bên dưới; không tuyên bố đã chạy lại mọi phân tích cũ. Không thay đổi raw hoặc cấu hình pipeline.

**Kết quả chạy:** cả 15 kiểm tra đạt. Đã quét 38.764.930 TRX, 66.295.724 GEO và 217.205 DIALOG; từng nguồn có 0 orphan, 0 null khóa/thời gian, 0 ô count khách–tháng lệch cache. Các phân nhóm recency DIALOG chi tiết ngoài cờ tháng trước được dẫn từ báo cáo target-focused hiện có.

## 1. Nối bảng: ghép đúng đơn vị mới có insight

Thiết kế nối:

```text
client_split: 1 dòng/client_id
    → targets: 1 dòng/(client_id, tháng nhãn)
    ← TRX tổng hợp trước cutoff: 1 dòng/(client_id, cutoff)
    ← GEO tổng hợp trước cutoff: 1 dòng/(client_id, cutoff)
    ← DIALOG tổng hợp trước cutoff: 1 dòng/(client_id, cutoff)
```

Giữ targets làm bảng gốc và LEFT JOIN từng nguồn đã tổng hợp; yêu cầu `validate='one_to_one'` cho khóa khách–cutoff và số dòng không tăng/giảm. Không nối trực tiếp TRX với GEO chỉ theo client_id: với m giao dịch và n log GEO sẽ tạo m×n dòng, làm sai frequency và monetary. `client_split` chỉ mang định danh/fold, không phải hồ sơ nhân khẩu học; hiện không có căn cứ phân tích tuổi, thu nhập hay tên thương mại sản phẩm.

Cutoff là 00:00 Moscow ngày đầu tháng nhãn; giữ mọi sự kiện trước cutoff. Timestamp raw không có timezone: theo quy ước nghiên cứu hiện hành, cộng 3 giờ khi phân tháng Moscow; đây không phải xác minh timezone ETL. `mon` là định danh tháng, không dịch timezone. Tháng gần nhất chỉ cho recency theo tháng, không suy ra số ngày chính xác.

So sánh H=1/2/3 dùng cùng 100.224 khách × 10 cutoff 02–11/2022 = 1.002.240 quan sát. H=2/H=3 nghĩa là có ít nhất một nhãn dương trong 2/3 tháng kể từ tháng đích. Hai tháng cuối vẫn nằm trong kiểm kê nhãn, nhưng thiếu tương lai H=3 nên không thuộc cohort so sánh chung.

**Insight từ phép nối:** không có log không đồng nghĩa không mua. Nhóm không có lịch sử TRX vẫn có tỷ lệ mua H=1 khoảng 1,41%. Inner join loại nhóm này sẽ đổi tổng thể và làm mất người mua. Giữ cờ `has_trx_history`, `has_geo_history`, `has_dialog_history`; không gán recency thiếu thành 0. Count=0 chỉ là không có log được ghi nhận, không phải xác nhận không hoạt động. Thiếu cả ba nguồn cũng không chứng minh là khách mới mở tài khoản.

## 2. Sổ insight: phát hiện → hành động

Các số gộp dưới đây là **khách–tháng**, có lặp khách giữa tháng; số mua là khách–tháng có nhãn dương, không phải số khách mua duy nhất. Lift so với tỷ lệ nền cùng cohort; không phải hiệu quả chiến dịch.

| Phát hiện | Bằng chứng | Hệ quả cho feature / nghiên cứu |
|---|---|---|
| DIALOG gần cutoff là tín hiệu nổi bật | Có log tháng trước: 2.436/57.674 = **4,22%**, lift **3,71×** so với nền 1,14%. Chưa từng thấy DIALOG: 4.352/640.228 = **0,68%** | Ưu tiên recency, has-history và trạng thái tháng trước; thử giá trị bổ sung ngoài thời gian |
| Nhóm DIALOG gần có quy mô nhỏ nhưng đóng góp đáng kể | Chiếm **5,75%** quan sát, chứa **21,33%** khách–tháng dương H=1 | Dùng quy mô và đóng góp bên cạnh lift khi ưu tiên nghiên cứu. Không gọi 21,33% là recall ngoài thời gian |
| Tín hiệu DIALOG xuất hiện qua cả 10 cutoff | Tỷ lệ mua có DIALOG tháng trước cao hơn nhóm còn lại ở cả 10 tháng; 04/2022: 7,26% vs 1,49%; 11/2022: 3,12% vs 0,67% | Hướng quan hệ ổn định nhưng cường độ thay đổi; kiểm tra coverage, baseline từng tháng và calibration |
| TRX không có quan hệ đơn điệu | 0 / 1–9 / 10–99 / ≥100 log: **1,41% / 0,76% / 0,59% / 1,48%** H=1 | Giữ trạng thái 0 riêng; count log1p, nhóm hoặc hàm phi tuyến là ứng viên. Không ép ràng buộc đơn điệu |
| Tỷ lệ gộp che mất tính bất ổn của nhóm không TRX | Lift tháng của nhóm 0 TRX từ **0,55× đến 3,41×**; nhóm DIALOG tháng trước từ **3,32× đến 5,14×** | Nhóm 0 TRX đổi từ dưới nền sang trên nền theo tháng. Chưa dùng làm quy tắc ưu tiên độc lập; cần kiểm tra thành phần cohort và độ dài lịch sử |
| Độ phủ nhiều nguồn liên quan đến nhãn | Có cả 3 nguồn: **1,98%**, so với 0/1/2 nguồn: **0,96% / 0,64% / 0,96%** | Tạo source masks và tổ hợp nguồn; kiểm soát độ dài lịch sử để tránh nhầm mức ghi nhận với mức gắn kết |
| Tương tác đáng điều tra: không TRX nhưng DIALOG gần | **272/2.890 = 9,41%**, so với 918/81.780 = 1,12% ở nhóm không TRX, không DIALOG gần | Tạo tương tác DIALOG×TRX; ưu tiên kiểm tra cơ chế ghi nhận trước khi dùng quy tắc nhắm khách |
| DIALOG không chỉ phản ánh khách đã mua trước đó | Chưa có tháng mua được quan sát: có DIALOG gần **3,87%**, không có **0,83%** | Kiểm tra riêng theo lịch sử nhãn; chỉ dùng prior-purchase feature nếu xác nhận nhãn đã sẵn có tại cutoff |
| H dài hơn tăng tỷ lệ nhãn nhưng chưa chứng minh mô hình tốt hơn | Cùng cohort: H1 **1,14%**, H2 **2,17%**, H3 **3,07%** | Chọn horizon theo mục tiêu tiếp cận và đánh giá cùng top-K ngoài thời gian, không chọn theo tỷ lệ dương cao nhất |

Nguồn cho tương tác và lịch sử mua: [TRX×DIALOG](../outputs/target_focused_moscow_20261119/dialog_by_trx_state.csv), [lịch sử mua×DIALOG](../outputs/target_focused_moscow_20261119/dialog_by_prior_purchase.csv). Các bảng này phân tầng từng yếu tố riêng, chưa điều chỉnh đồng thời tất cả yếu tố nhiễu và chưa chứng minh nhân quả. Không suy diễn “gọi điện làm tăng mua”.

### Đơn biến và chất lượng: điều gì thay đổi cách xử lý?

- **Nhãn hiếm và thay đổi theo thời gian:** toàn bộ 12 tháng có 12.854/1.202.688 = 1,0688% dương; con số 1,14% ở trên dùng 10 cutoff chung nên không mâu thuẫn. Theo dõi PR-AUC, precision/recall@K khi đánh giá mô hình; accuracy đơn lẻ sẽ khó phản ánh khả năng tìm người mua.
- **Amount phải phân tích theo currency:** bảng mới xuất n, null, zero, negative, min, median, p99 và max từng mã. Không cộng các currency thành “doanh thu”; đơn vị tiền chưa xác minh. Dùng log1p cho giá trị không âm phù hợp, giữ raw để đối soát. Fit mọi ngưỡng xử lý từ train khi xây mô hình.
- **Đuôi amount rất dài:** riêng currency 11 có 38.724.240 dòng, median khoảng 18.025, p99 khoảng 5,26 triệu và max khoảng 1,37 nghìn tỷ đơn vị dữ liệu; không có amount null/0/âm trong nhóm này. Mean đơn lẻ dễ bị đuôi chi phối; ưu tiên median/phân vị và kiểm tra điểm cực trị trước khi diễn giải “khách giá trị cao”.
- **Ngoại lai chưa phải lỗi:** audit trước tìm 53.401 giao dịch đáng kiểm tra trong nhóm tham chiếu đủ dữ liệu, không phải 53.401 lỗi đã xác nhận. Không tự xóa/cap khách hoạt động cao. Nguồn: [audit ngoại lai](../outputs/trx_outliers_20261116/RESULTS.md), tái sử dụng, không chạy lại trong lần này.
- **Nguồn có tuổi lịch sử khác nhau:** kiểm kê trước ghi TRX từ 01/2021, GEO từ 12/2021, DIALOG từ 02/2021. Count tích lũy chịu ảnh hưởng độ dài phần lịch sử được quan sát; bổ sung history age và source coverage, không coi mọi khách có thời gian quan sát bằng nhau.
- **Trùng thông tin:** bảng tương quan đã có cho thấy TRX count–active days Spearman khoảng 0,998 tại cutoff 02/2022. Cần kiểm tra dư thừa trong mô hình tuyến tính; chưa có lý do xóa tự động một biến cho mọi mô hình. Nguồn: [cặp tương quan](../outputs/relationships_moscow_20261118/spearman_pairs.csv), không tính lại lần này.

## 3. Checklist EDA riêng cho dự án

`[x]` có kết quả kiểm tra mới trong lần này. `[ ]` là việc còn mở hoặc mới có bằng chứng cũ/một phần; không có nghĩa chưa từng được làm.

### P0 — khóa mẫu, join và thời gian

- [x] Xác minh client_id duy nhất/không null ở bảng khách; khóa khách–tháng duy nhất/không null ở targets.
- [x] Kiểm tra cả bốn target trong {0,1}, không thiếu; đủ 12 tháng và đúng quần thể mỗi tháng.
- [x] Quét cả ba nguồn raw: null client_id/event_time, orphan so với bảng khách, số dòng và số khách.
- [x] Tính lại count khách–tháng Moscow từ raw; đối soát từng ô với cache, rồi cache với panel.
- [x] Đối soát nhãn panel với targets raw; khóa panel không trùng.
- [x] Dựng feature bằng toàn bộ các tháng trước cutoff, không chứa tháng nhãn; giữ khách không có log.
- [x] H1/H2/H3 dùng cùng cohort 10 cutoff, kiểm tra tổng các phân khúc đúng mẫu số.
- [ ] Xác minh timezone ETL, độ trễ nhãn và label_available_at với nguồn dữ liệu. Trước đó, prior-purchase feature chỉ là giả định nghiên cứu.
- [ ] Audit PTLS–detail về nội dung, thứ tự và mã nếu dùng PTLS thay detail; không cộng hai biểu diễn.
- [ ] Quét lại duplicate toàn dòng/near-duplicate khi thay phiên bản raw. Bảng duplicate cũ có sẵn nhưng lần này chỉ kiểm tra khóa của panel/labels và orphan sự kiện.

### P1 — đơn biến và xử lý

- [x] Tỷ lệ target từng tháng trên đủ 12 tháng, báo rõ denominator.
- [x] Amount theo từng currency: số dòng, null, zero, negative, median, p99, min/max.
- [x] Phân khúc count=0 và có log; giữ trạng thái thiếu log khi phân tích target.
- [ ] Hợp nhất null/cardinality/mức hiếm/gần hằng số cho tất cả cột raw và feature ở từng cutoff. Đã có bảng quality cũ, chưa refresh toàn bộ.
- [ ] Kiểm tra finite và độ nhạy kết luận trước/sau xử lý ngoại lai; đối chiếu điểm nghi sai scale, chưa tự sửa.
- [ ] Phân phối recency theo ngày, gap liên sự kiện xuyên tháng và active days cho cả ba nguồn; bản mới chỉ kiểm tra count và recency tháng.
- [ ] Tính history age/độ phủ trước cutoff để phân biệt ít log với ít lịch sử quan sát.

### P1 — đa biến và độ ổn định

- [x] Xuất thẻ phân khúc: n, buyers, rate, population share, buyer share, lift theo H và tháng.
- [x] So count TRX, DIALOG gần và số nguồn với target trên cohort thống nhất.
- [ ] Kiểm định ổn định tương tác DIALOG×TRX×history age×prior purchase đồng thời theo tháng; phân tầng riêng đã có, chưa loại trừ Simpson's paradox/nhiễu còn lại.
- [ ] Bootstrap theo client_id (giữ chuỗi tháng của mỗi khách), báo CI cho rate/lift/chênh lệch; không coi 1.002.240 dòng là quan sát độc lập. Bản này chưa xuất CI hay p-value mới.
- [ ] Đối chiếu nhiều phép thử và nhóm hiếm; không chọn nhóm cao nhất rồi coi là hiệu quả đã xác nhận.
- [ ] Kiểm tra dư thừa feature và giá trị bổ sung của mỗi nguồn bằng ablation ngoài thời gian.
- [ ] Kiểm tra drift phân phối/coverage/target và phân rã thay đổi thành phần khách; baseline tháng mới có, chưa đủ báo cáo drift toàn diện.

### P2 — biến insight thành thử nghiệm

- [ ] Chốt feature ưu tiên: DIALOG recency/presence; TRX count và trạng thái 0; masks đa nguồn; tương tác DIALOG×TRX; history age. Prior-purchase chỉ dùng khi giải quyết thời điểm sẵn có nhãn.
- [ ] Chốt ngưỡng/encoding trên tháng train, đánh giá trên tháng sau; H nhiều tháng cần tránh train label window chồng vào kỳ đánh giá và tính cả độ trễ nhãn.
- [ ] So với baseline theo thời gian; báo precision@K, recall@K, lift@K cùng K và calibration. Chưa huấn luyện hay xác nhận cải thiện mô hình trong báo cáo này.
- [ ] Xác minh nhóm không TRX nhưng DIALOG gần với chủ dữ liệu; tách vấn đề thu thập log khỏi diễn giải nghiệp vụ.

## Tái lập

Chạy từ root dự án:

```powershell
.venv/Scripts/python.exe -B -X utf8 src/scripts/audit_insights_checklist.py
```

Script đọc raw và cache cục bộ, xuất vào `outputs/eda_insights_20261119/`, dừng bằng assertion nếu đối soát thất bại. Báo cáo này là sổ insight và checklist, không phải thông báo đã hoàn thành mọi mục EDA.
