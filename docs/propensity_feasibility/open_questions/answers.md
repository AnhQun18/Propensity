# Trả lời các câu hỏi còn bỏ ngỏ — 09/11/2026

Phạm vi: các câu hỏi ở bước Business Understanding và Data Understanding trong README.
Đối chiếu dataset card, bài báo MBD, mã nguồn benchmark MBD, bài báo/mã nguồn EventGPT,
và đọc lại targets cùng timestamp của cả ba nguồn detail trên bản MBD-mini địa phương.

Phân biệt ba trạng thái: **đã xác minh** bằng nguồn/dữ liệu; **đề xuất cho đề tài** là
lựa chọn triển khai có thể thực hiện nhưng chưa được ngân hàng nghiệm thu; **chưa công bố**
là thông tin không tìm thấy trong các nguồn đã kiểm tra. Không chuyển trạng thái cuối
thành khẳng định về dữ liệu gốc.

**Quyết định người dùng sau các kiểm chứng:** tạm dùng UTC thống nhất cho raw/cutoff/
cửa sổ; cutoff đầu tháng lúc 00:00 UTC, không dịch 3 giờ. Tên thương mại của bốn sản
phẩm không còn là yêu cầu xác minh trong phạm vi `p_any`. Các đoạn về Moscow dưới
đây mô tả bằng chứng và phương án đối chiếu, không phải cấu hình chính hiện tại.

**Đã tự xác minh tiếp trans_count, missing và độ trễ nhãn:** xem
[báo cáo mới](../remaining_questions/assessment.md). Quy ước currency đã được người
dùng chốt: 11 là mã chính, mã khác có giá trị là ngoại tệ theo cờ dự án, currency null
giữ unknown/missing. Không còn chờ mapping tên tiền để thực hiện cờ này.

## 1. Mục tiêu kinh doanh, người dùng và phạm vi

**Bối cảnh đã xác minh:** MBD phục vụ dự đoán mua sản phẩm để lập chiến dịch và chuẩn bị
liên hệ bán hàng; khách hàng trong bài MBD là khách hàng doanh nghiệp, không mặc định là
khách cá nhân. Bài báo không cung cấp quy trình bán hàng cụ thể của đơn vị thực tập [S1].

**Câu trả lời đề xuất cho đề tài:** xây dựng và đánh giá offline một hệ thống xếp hạng
khách hàng hiện hữu mỗi tháng theo khả năng mua ít nhất một trong bốn sản phẩm có nhãn.
Người sử dụng dự kiến là bộ phận CRM/marketing hoặc quản lý quan hệ khách hàng; kết quả
hỗ trợ chọn danh sách ưu tiên liên hệ. Lợi ích cần kiểm chứng là tìm được nhiều khách
có mua hơn trong cùng số lượng được chọn. Đây là thiết kế nghiên cứu, chưa xác nhận có
một bộ phận cụ thể sẽ tiếp nhận hệ thống.

Đầu ra chính của đề tài là **một xác suất `p_any`**; bốn nhãn nguồn vẫn được giữ để tạo
`y_any = 1` khi có ít nhất một nhãn bằng 1. Đây là thiết kế cập nhật đã có trong repo,
khác benchmark gốc đánh giá riêng bốn sản phẩm. Nó không quyết định giới thiệu sản phẩm
nào, không dự đoán mọi sản phẩm ngân hàng và không phải mô hình uplift.

Đơn vị dự đoán: khách hàng–tháng. Quần thể nghiên cứu: 100.224 khách trong bản dữ liệu;
giữ khách thiếu nguồn hoặc không có lịch sử với fallback. Chia theo thời gian để trả
lời câu hỏi dự đoán tháng tương lai cho khách hiện hữu. Khả năng tổng quát sang khách
hoàn toàn mới cần một thí nghiệm giữ riêng ID và không thuộc kết luận mặc định.

## 2. Hiện trạng và baseline là gì?

Đã có raw, kiểm kê, EDA, audit chất lượng và thiết kế đánh giá; chưa có mô hình đã được
huấn luyện/đánh giá trong phạm vi này. Schema công khai không có lịch liên hệ, việc phân
bổ chiến dịch, chi phí liên hệ, lợi nhuận sản phẩm hay kết quả của quy trình chọn khách
hiện hành. Vì vậy chưa xác định được tỷ lệ chuyển đổi của chiến dịch hiện tại hoặc ROI.
Nhãn mua quan sát được không chứng minh khách mua do được liên hệ.

Đối chứng nghiên cứu đề xuất trên cùng tập khách và cùng tháng:

- Chọn ngẫu nhiên K khách, seed cố định; báo cáo kỳ vọng và dao động của số ca dương.
- Xếp hạng bằng mức độ gần đây/tần suất giao dịch, chỉ dùng quá khứ; quy định tie-break.
- Một CatBoost nhị phân trên đặc trưng giao dịch, rồi thêm từng nguồn để đo giá trị tăng thêm.
- Dự báo tỷ lệ mua từ train để đối chiếu log loss/Brier; điểm hằng cần tie-break khi đánh giá top-K.

Hiện trạng kinh doanh thực tế chỉ có thể được điền từ hồ sơ của đơn vị sử dụng; không
lấy baseline nghiên cứu làm số liệu kinh doanh đã xác minh.

## 3. Chọn K, metric và điều kiện thành công thế nào?

**Đề xuất nghiên cứu trước huấn luyện:** K = ceil(1% × số khách đủ điều kiện được chấm
điểm trong tháng), tương ứng 1.003 khi N = 100.224. Báo cáo thêm top 0,5% và 5% để thấy
độ nhạy theo ngân sách. Top 1% đã xuất hiện trong nghiên cứu thiết kế của repo; đây là
ngân sách mô phỏng, không phải năng lực liên hệ đã xác nhận của ngân hàng.

- `Precision@K = số khách dương trong top-K / K`.
- `Recall@K = số khách dương trong top-K / tổng khách dương trong tháng`.
- `Lift@K = Precision@K / tỷ lệ dương của cùng quần thể tháng đánh giá`.
- AP, ROC-AUC bổ sung cho xếp hạng; log loss, Brier và calibration cho xác suất.

Đề xuất điều kiện đạt cho nghiên cứu: Lift@1% > 1 so với chọn ngẫu nhiên; báo cáo chênh
lệch paired so với baseline giao dịch, khoảng tin cậy bootstrap theo khách và độ ổn định
ở từng tháng validation. Chỉ kết luận nguồn bổ sung có ích khi chênh lệch có bằng chứng;
khi không đủ bằng chứng thì giữ baseline đơn giản. Không tuyên bố đã đạt các điều kiện
này khi chưa chạy mô hình. Các tỷ lệ dương dùng trong metric là kết quả đánh giá sau
khi khóa mô hình, không được dùng để fit dự báo cho chính tháng đó.

Tuning cuốn chiếu trên 09–11/2022; mỗi lần chỉ fit phần quá khứ hợp lệ. Chọn cấu hình,
fit cuối trên 02–11/2022; 12/2022 fit calibration theo quy tắc đã chọn; khóa 01/2023 làm
test. Không dùng kết quả test để đổi K hoặc chọn mô hình. Kiểm soát thời điểm nhãn sẵn
có; độ trễ công bố nhãn thực tế chưa được nguồn mô tả.

Audit mới cho thấy không có mốc sẵn có nhãn trong đủ 435 shard và metadata. Lịch
calibration tháng 12 → test ngày 01/01 chỉ khả thi dưới giả định độ trễ 0. Với giả
định trễ dương, cần dựng lại cả lịch chọn cấu hình/fit cơ sở/calibration, không dùng
nhãn chưa sẵn theo giả định. Đây là giới hạn của mô phỏng offline hiện tại; chưa có
độ trễ thực tế được xác minh hoặc metric mô hình theo các kịch bản trễ.

Chưa đặt số tiền doanh thu/ROI hoặc mục tiêu tăng chuyển đổi tuyệt đối vì thiếu chi phí,
biên lợi nhuận và dữ liệu can thiệp. Muốn chứng minh hiệu quả chiến dịch thực tế cần
đánh giá có nhóm đối chứng phù hợp sau giai đoạn nghiên cứu offline.

## 4. Kế hoạch và trách nhiệm

Đây là kế hoạch đề xuất theo mốc đầu ra, không phải lịch hoặc phân công đã được duyệt.

| Mốc | Việc thực hiện | Đầu ra / điều kiện hoàn thành | Vai trò đề xuất |
| --- | --- | --- | --- |
| M0 | Khóa phạm vi và protocol | `y_any`, top-K, cutoff/timezone, split, baseline, danh sách giả định | Người thực hiện soạn; người hướng dẫn/đầu mối nghiệp vụ duyệt mục tiêu |
| M1 | Chuẩn bị dữ liệu | Bảng khách–tháng giữ đủ nhãn, mask nguồn, lịch sử trước cutoff; đối soát số dòng | Người thực hiện |
| M2 | Baseline giao dịch | Kết quả cuốn chiếu, top-K, AP và chất lượng xác suất | Người thực hiện |
| M3 | Đối chứng đa nguồn | Cùng cohort/split, thêm geo/dialog và chính sách missing; đo chi phí | Người thực hiện; người hướng dẫn rà soát |
| M4 | Khóa và kiểm thử | Cấu hình cố định, calibration tháng 12, test tháng 01/2023, bất định và nhóm cold-start | Người thực hiện; người hướng dẫn rà soát |
| M5 | Bàn giao nghiên cứu | Mã tái lập, từ điển feature, báo cáo, giới hạn sử dụng, bảng điểm mẫu | Người thực hiện bàn giao; đầu mối tiếp nhận đánh giá |

Ngày hoàn thành, tên người chịu trách nhiệm và ngân sách vận hành thực tế chưa có trong
repo. Không tự gán tên hoặc deadline. Đo thời gian/RAM ở M1–M3 trên máy mục tiêu trước
khi đặt cam kết thời gian chạy; repo hiện có hướng dẫn cấu hình máy RAM 16 GB.

## 5. Bốn sản phẩm thực sự là gì?

Đã xác minh có bốn **sản phẩm**, không phải bốn tháng dự báo liên tiếp [S1, S2]. Chưa
tìm thấy mapping `target_1`–`target_4` sang tên thương mại trong dataset card, paper hoặc
mã nguồn MBD đã kiểm tra. Không tự gán là thẻ, vay, bảo hiểm hay tiền gửi.

Theo quyết định người dùng, không tiếp tục coi tên thương mại là câu hỏi cần đóng
cho đề tài any-product. Giữ nguyên mã nhãn để tạo/đối soát `y_any`. Chỉ cần mở lại
mapping nếu sau này muốn giải thích hoặc đề xuất riêng từng sản phẩm.

## 6. Amount và currency có ý nghĩa gì?

Quy ước dự án đã chốt: `currency_is_main=(currency==11)`,
`currency_is_foreign=(currency!=11)` trên mã có giá trị; cả hai unknown khi currency
null, thêm `currency_missing=True`. Đã triển khai hàm tạo cờ và kiểm tra trên toàn bộ
trx: 38.724.240 mã chính, 40.572 ngoại tệ theo cờ, 118 missing. Giữ mã gốc và chưa
gán tên đồng tiền thật. Đây là quy ước người dùng, không phải mapping từ nhà phát hành.

Bài MBD xác nhận categorical được đổi thành chỉ số và numerical có thêm nhiễu; tham số
ẩn danh không được công bố [S1]. Vì vậy mã `11` là mã categorical phổ biến, không đủ
bằng chứng gọi là RUB/USD/VND hoặc lấy tỷ giá thị trường để quy đổi. Không có cơ sở
khôi phục giá trị tiền gốc từ amount đã công bố.

Theo thống kê toàn lịch sử đã lưu, mã 11 có 38.724.240/38.764.930 giao dịch (99,895034%);
currency null có 118 giao dịch. EventGPT cũng sử dụng mã phổ biến 11 và amount bins
nhưng không cung cấp mapping tiền gốc [S4].

Cách dùng: giữ currency categorical/missing riêng; thống kê amount theo từng mã hoặc
biểu diễn tương đối học trên train; có đối chứng bỏ amount. Không diễn giải tổng amount
xuyên currency thành tổng doanh thu hoặc sức mua. Cần mapping và quy tắc biến đổi từ
nhà phát hành để xác định đơn vị thật; log/quantile không phải phép quy đổi tiền.

## 7. Tháng nhãn và timezone đã xác minh đến đâu?

MBD yêu cầu lịch sử nghiêm ngặt trước đầu tháng mua. EventGPT nói rõ `mon` nằm trong
tháng nhãn, timestamps UTC, tháng nghiệp vụ theo Moscow [S1, S4]. Mã EventGPT đã tải
đối chiếu thực hiện `month_cut = start_of_month - 3*3600` [S5].

Ví dụ theo protocol này: `mon=2022-12-31` có cutoff `2022-12-01 00:00 Europe/Moscow`
= `2022-11-30 21:00 UTC`; nhãn tháng 12 kéo dài đến `2022-12-31 21:00 UTC`.
Không dịch nhãn thêm một tháng, không đổi theo timezone Việt Nam của máy chạy.

**Giới hạn:** các shard đại diện của raw có `timestamp[ns]`, không có timezone trong
kiểu Arrow. Mã benchmark gốc gọi `datetime(...).timestamp()` không chỉ rõ timezone
[S3], nên phụ thuộc môi trường chạy. Như vậy đã xác minh protocol của EventGPT, chưa
có xác nhận trực tiếp về timezone của ETL tạo nhãn gốc. Biên cuối giao dịch/geo gần
20:59:58 UTC phù hợp với giải thích Moscow nhưng không phải bằng chứng độc lập đủ
để khẳng định timezone nguồn.

**Quy ước hiện tại theo quyết định người dùng:** raw được diễn giải UTC, dùng cutoff
đầu tháng lúc 00:00 UTC; không dịch 3 giờ. Moscow chỉ giữ làm sensitivity check và
đối chiếu EventGPT. Chưa tự sửa số liệu/cohort lịch sử hoặc tạo feature huấn luyện.

Đọc timestamp/ID trên toàn bộ 1.275 shard detail để đo đúng khoảng 3 giờ giữa hai cutoff:

| Kết quả | Số lượng |
| --- | ---: |
| Sự kiện được UTC-midnight giữ nhưng Moscow-midnight loại tại 12 cutoff nhãn | 88.976 |
| Trong đó trx / geo / dialog | 31.013 / 57.942 / 21 |
| Cặp khách–tháng có ít nhất một sự kiện như vậy | 71.467 (5,9423% của 1.202.688) |
| Khách phân biệt trên các cutoff | 35.660 |
| Sự kiện khác biệt tại cutoff tháng 01/2023 | 0 |

Đây là khác biệt lịch sử đầu vào, không phải số nhãn sai hay mức thay đổi metric. Tháng
test cuối không có sự kiện trong khoảng này, nhưng train/validation/calibration có,
nên không kết luận mô hình hoàn toàn không bị ảnh hưởng.

Nguồn: [verification.json](verification.json), [chi tiết từng nguồn](timezone_boundary_by_source.csv),
[số khách phân biệt từng tháng](timezone_boundary_clients.csv).

## 8. Công thức trans_count và diff_trans_date là gì?

**Bổ sung kiểm chứng trans_count:** 0/1.102.464 cặp tháng có giá trị giảm, nhưng 612
cấu hình đếm chưa tái tạo được nguồn. Có 102.932 dòng trans_count vượt toàn bộ trx
quan sát trước cutoff và 9.459 dòng dương ở khách không có trx toàn lịch sử. Vì vậy
chưa dùng hai cột nguồn làm predictor; tự tính count/recency. Chi tiết phạm vi và các
giả thuyết chưa thể phân biệt nằm trong báo cáo mới, không quy sai khác thành lỗi nguồn.

Dataset card chỉ mô tả một trường đếm và một trường chênh lệch thời gian. Notebook MBD
đọc targets có sẵn và loại hai cột phụ trợ khi ghép nhãn, không tạo ra chúng [S2, S6].
Không tìm được mã ETL đủ để xác minh cửa sổ, hai mốc thời gian, đơn vị hoặc thời điểm
các trường này sẵn có trong các nguồn đã kiểm tra.

Đọc lại toàn bộ 1.202.688 dòng xác nhận: 120.786 null của `diff_trans_date` trùng chính
xác với `trans_count=0`; min/max 0/364; có 49.293 dòng `trans_count=1` nhưng diff > 0.
Đó là quan hệ thực nghiệm, không phải công thức. Các thử nghiệm recency trong
[nghiên cứu trước](../diff_trans_date_research.md) không tái tạo được cột nguồn.

Cách giải quyết: chưa dùng trực tiếp hai cột này làm predictor. Tự tính
`observed_trx_count_<window>` và `days_since_last_observed_trx` từ detail trước cutoff,
định nghĩa rõ cửa sổ và phép tính ngày. Không đặt tên feature tự tính như thể đã tái
tạo đúng trường nguồn; không điền null diff thành 0.

## 9. Vì sao thiếu nguồn/null? Có phải xóa khách không?

**Cơ chế chung đã có bằng chứng sơ cấp:** MBD mục 3.1 nói nhiều khách thiếu nguồn vì
có thể không giao dịch, không gọi hỗ trợ hoặc không để lại vị trí khi dùng ứng dụng.
Vì vậy không mặc định mọi nguồn vắng là lỗi thu thập. Audit mới phân loại đủ mọi ô
khách–tháng theo có log, ngoài phạm vi nguồn, không từng xuất hiện, trước lần đầu,
xen giữa và sau lần cuối có log; đó là trạng thái quan sát, không phải nhãn nguyên nhân.

Đã xác minh hiện tượng thiếu, chưa xác minh nguyên nhân cho từng khách. Dữ liệu không
có cờ bật/tắt thu thập, thời điểm mở tài khoản hoặc lịch quan sát liên tục. Không có
log có thể do không phát sinh hành vi, nguồn không được ghi nhận hoặc nguyên nhân khác.
Không gắn chắc chắn nhãn MCAR/MAR/MNAR hoặc lỗi thu thập từ các số đếm hiện có.

Trong lưới nghiên cứu, null ngoài phạm vi nguồn là quy ước biểu diễn chưa quan sát;
mốc trước 01/2022 trong audit cũ không phải ngày bật thu thập đã được nhà phát hành xác
nhận. Có log sớm thì giữ. Trong khoảng có dữ liệu, count=0 chỉ nghĩa không ghi nhận
sự kiện; recency/mean/embedding không xác định vẫn null. Giữ masks và đánh giá theo
độ phủ, kể cả cold-start; không tự xóa khách vì thiếu geo/dialog hoặc ít sự kiện.

`ptls/trx` có ít sự kiện hơn detail ở 13 khách và độ dài tối đa 10.000 là quan sát đã
xác minh trước. Chưa xác minh thuật toán chọn/cắt chuỗi nguồn; dùng detail làm nguồn
chuẩn thay vì khẳng định chắc chắn cơ chế cắt. Mẫu EDA theo toàn lịch sử không dùng
để quyết định cohort tại cutoff quá khứ.

## 10. Nhãn thiếu có cần loại? Có thể tiếp tục không?

Đọc lại targets: 100.224 khách × 12 tháng = 1.202.688 dòng, 0 trùng khóa, 0 null nhãn,
0 giá trị nhãn ngoài 0/1; `y_any` có 12.854 dòng dương (1,0687726%) và 1.189.834 dòng
âm. Hiện không loại dòng nào do thiếu target. Chỉ loại khỏi supervised khi thực sự
không xác định được `y_any`; không dùng inner join làm mất khách rồi gọi là thiếu nhãn.

Có thể tiếp tục chuẩn bị dữ liệu và baseline với các quy ước đã ghi. Các điểm còn mở
về tên sản phẩm, tiền gốc, ETL nhãn/timezone và cơ chế thu thập được khoanh phạm vi sử
dụng; chúng không tự động ngăn nghiên cứu offline, nhưng chưa cho phép kết luận ROI,
triển khai chiến dịch thực hoặc khẳng định ý nghĩa nghiệp vụ của các mã ẩn danh.

Chạy lại kiểm chứng mới (không sửa raw, không fit mô hình):

```powershell
.\.venv\Scripts\python.exe -B -X utf8 src/audit_open_questions.py
```

## Nguồn và phiên bản đã kiểm tra

- **[S1]** [MBD v2, mục 3.2, 3.3 và 4.1](https://arxiv.org/html/2409.17587v2).
- **[S2]** [Dataset card MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini/blob/4b0d5b4/README.md).
- **[S3]** [MBD GetSplit](https://github.com/Dzhambo/MBD/blob/0f8a41b2eca7945656b51dfb447b0d08b624ac77/modules/processing.py#L70).
- **[S4]** [EventGPT v1, mục 5.1, 5.2 và 6.1](https://arxiv.org/html/2611.03302v1).
- **[S5]** [EventGPT month_cut](https://github.com/deburky/EventGPT/blob/88e6e0abae4722e6888897632350db23dee66d4f/src/eventgpt/evaluation/probe.py#L99),
  [đọc timestamp và mã currency](https://github.com/deburky/EventGPT/blob/88e6e0abae4722e6888897632350db23dee66d4f/src/eventgpt/data/mbd.py#L53).
- **[S6]** [Notebook ghép dữ liệu/targets của MBD](https://github.com/Dzhambo/MBD/blob/0f8a41b2eca7945656b51dfb447b0d08b624ac77/scenario_mbd/notebooks/Preprocessing/GetMultimodalData.ipynb).

MBD checkout `0f8a41b2eca7945656b51dfb447b0d08b624ac77`; EventGPT checkout
`88e6e0abae4722e6888897632350db23dee66d4f`. Checkout chỉ phục vụ đọc mã, giữ cục bộ
trong outputs, không đưa bản sao repository ngoài vào Git của đề tài.
