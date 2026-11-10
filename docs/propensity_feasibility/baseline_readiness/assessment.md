# Rà soát trước baseline any-product

> **Snapshot lịch sử ngày 09/11/2026.** Các phát hiện bên dưới mô tả code tại thời
> điểm đó. Đến 10/11: currency unknown đã sửa, hai notebook đã chạy raw và có
> manifest ready; schema v2/validation dùng chung đã được kiểm thử, output thật
> đã kiểm tra lại. Reader cả hai notebook có kiểm tra ready/schema/protocol.
> Trạng thái mới: [xử lí tồn đọng 10/11](../../preprocessing_fixes_20261110.md).
> F03/F04 thuộc bước mô hình, F05 vẫn là giới hạn của baseline thống kê.

Ngày rà soát: **09/11/2026**. Phạm vi: mục tiêu/nhãn, các audit đã có, cấu hình,
toàn bộ package `src/propensity_data`, notebook chuẩn bị dữ liệu, tests đã viết,
schema feature, protocol đánh giá và môi trường Python cục bộ.

**Kết luận:** đủ cơ sở để xây baseline nghiên cứu offline theo UTC và `y_any` đã
thống nhất; **chưa đủ bằng chứng để coi pipeline và baseline sẵn sàng thực thi**.
Cần sửa cờ ngoại tệ ở trường hợp chỉ có currency unknown, xác nhận đầu ra chuẩn bị
dữ liệu trên máy thực thi, và hoàn thiện bộ huấn luyện/đánh giá có danh sách feature
cố định. Không cần chờ giải mã tên sản phẩm, mã giao dịch hoặc địa chỉ geohash.

Đây là **rà soát tĩnh**, không chạy unit test, cell notebook, pipeline chuẩn bị,
quét lại raw hoặc fit mô hình. Đã đọc/parse 13 file Python và 20 cell code; 8 cell
liên kết helper khớp nội dung nguồn, mọi output/execution count notebook đều trống.
Kết quả máy đọc và SHA256 các file đã xét: [static_review.json](static_review.json).
Không coi việc parse thành công là bằng chứng SQL hoặc pipeline chạy thành công.

## 1. Những việc cần xử lý trước lần fit đầu tiên

| Mã | Loại | Phát hiện | Việc cần hoàn thành |
|---|---|---|---|
| F01 | Sai khác quy ước trong code | Cờ `has_foreign` trả 0 khi khách có log nhưng toàn bộ currency thiếu/invalid. | Giữ unknown cho trường hợp không có bất kỳ currency hợp lệ nào; thêm kiểm tra hồi quy cho trường hợp này. |
| F02 | Thiếu bằng chứng thực thi | Chưa có manifest chuẩn bị dữ liệu; tests hiện mới được viết. | Trên máy thực thi: chạy tests, chuẩn bị dữ liệu, đối soát rồi chỉ dùng output `status=ready`. |
| F03 | Thiếu thành phần baseline | Chưa có trainer, evaluator, calibrator và cấu hình baseline trong repo. | Viết hợp đồng input/feature/split/metric và lưu prediction có khóa; không gọi bước chuẩn bị là baseline đã chạy. |
| F04 | Môi trường | `.venv` chưa có metadata cài đặt CatBoost/scikit-learn; `pyproject.toml` chưa khai báo nhóm dependency baseline. | Cài trên máy sẽ fit, kiểm tra import/tương thích và lưu chính xác phiên bản/seed/cấu hình. |
| F05 | Khoảng cách giữa đề xuất và code | Feature chính chỉ giữ cardinality/missing của src/dst, không giữ danh tính mã hoặc cơ cấu từng mã. | Đặt tên rõ baseline thống kê hiện tại; nếu đánh giá nội dung mã thì bổ sung chúng thành thí nghiệm riêng với vocabulary học trên train. |
| F06 | Kiểm soát schema | `feature_schema` hiện lấy mọi cột ngoài danh sách loại trừ; chưa từ chối cột lạ. | Đóng băng danh sách predictor được duyệt, kiểm tra cột/dtype và loại key/nhãn/metadata khi đọc vào mô hình. |
| F07 | Kiểm soát output | Reader notebook có lời nhắc chỉ đọc sau ready nhưng chưa kiểm tra trạng thái trong hàm. | Bộ đọc của baseline phải kiểm tra manifest/protocol/schema trước khi mở features. |

F02–F07 là phần còn thiếu hoặc giới hạn bảo vệ, không phải tuyên bố pipeline đã
chạy sai. Các phát hiện chưa được sửa trong đợt **review** này.

### F01 — ví dụ có thể kết luận từ biểu thức hiện tại

Trong [features.py](../../../src/propensity_data/features.py), `foreign_count` đếm
`currency <> 11`; currency null không được tính. Sau đó `has_foreign` được tạo bằng
`coalesce(a.foreign_count, 0) > 0`. Một khách có hai log nhưng cả hai currency null
sẽ nhận `foreign_count=0`, `has_foreign=0`, `foreign_fraction=0`.

Điều này làm mất phân biệt giữa **chưa biết mã tiền** với **đã quan sát mã chính và
không thấy ngoại tệ**, trái với quy ước trong [audit currency](../remaining_questions/assessment.md).
Chưa đo số khách–cửa sổ thực tế bị ảnh hưởng; đây là phản ví dụ logic của code.

Quy tắc cần dùng:

- Ngoài phạm vi nguồn: giữ null.
- Có phạm vi, không có log: cờ ngoại tệ quan sát được bằng 0, kèm `has_event=0`.
- Có ít nhất một mã ngoại tệ hợp lệ: cờ bằng 1.
- Không thấy ngoại tệ nhưng có mã 11: cờ bằng 0, vẫn giữ số log currency thiếu/invalid.
- Có log nhưng không có currency hợp lệ: cờ null.

`foreign_fraction` hiện là số log ngoại tệ **đã quan sát / tổng log**, không phải
tỷ lệ ngoại tệ thật khi có mã thiếu. Có thể giữ tỷ lệ này với tên/mô tả rõ và cờ
missing, hoặc thêm tỷ lệ trên số log có currency hợp lệ; mẫu số bằng 0 thì null.
Không âm thầm đổi mẫu số mà không cập nhật từ điển feature.

### F05 — tính năng thực sự có trong code

[features.py](../../../src/propensity_data/features.py) hiện tạo:

- Count/amount/active days/recency/log count, các cửa sổ 30/84 ngày và toàn lịch sử.
- Với toàn bộ `event/src/dst`: số mã khác nhau và số log thiếu mã.
- Chỉ `event_type` và `event_subtype` có top-share và sidecar tần suất từng mã.
- Không có mode/last code, cặp event–subtype, cặp src–subtype hoặc tần suất từng mã
  src/dst trong sidecar hiện tại. Sidecar cũng chưa tự trở thành predictor của mô hình.
- Geo hiện có cardinality và top-share geohash 6; chưa giữ danh tính vùng phổ biến nhất.
- Dialog embedding tắt mặc định; bật lên mới xuất mean/last, chưa có PCA được fit.

Hai khách có cùng count, cardinality, tỷ lệ thiếu và top-share nhưng dùng các mã
khác nhau có thể có cùng feature thống kê hiện tại. Đây là baseline đơn giản hợp
lệ, nhưng kết quả của nó chưa trả lời toàn bộ giá trị dự báo của các mã categorical.
Các đề xuất trong trao đổi không đồng nghĩa đã được tích hợp vào notebook.

## 2. Phần đã nhất quán trong thiết kế/code

| Hạng mục | Kết quả rà soát tĩnh và bằng chứng đã có |
|---|---|
| Đơn vị mẫu | Một dòng khách–tháng; universe là `client_split` × các tháng cấu hình, left join để giữ khách thiếu nguồn. |
| Nhãn | OR bốn target; có 1 thì dương, cả bốn 0 thì âm, còn lại unknown. Có kiểm tra khóa/tháng/nhãn ngoài 0/1/null. |
| Số lượng nhãn | Audit hiện có: 100.224 khách × 12 tháng; 12.854 dương, 1.189.834 âm, không thiếu nhãn/khóa. |
| Thời gian feature | `event_time < cutoff`, UTC, cửa sổ ngày `[cutoff-days, cutoff)`. Cutoff là đầu cùng tháng nhãn, không cộng thêm tháng. |
| Trường phụ trợ | `fold`, `trans_count`, `diff_trans_date` nguồn không được đưa vào bảng predictor hiện tại. |
| Missing | Count 0 trong phần có phạm vi nguồn; trước phạm vi nguồn giữ null; mean/recency không tồn tại giữ null, có masks. |
| Tiền tệ | Amount mã 11 riêng; sidecar theo từng currency, không cộng khác đơn vị. |
| Cực trị | Không xóa heavy users/ngày đỉnh hoặc cap raw tự động. |
| Quarantine | ID/time không hợp lệ bị tách, thuộc tính phụ thiếu không làm mất event metadata. |
| Dữ liệu gốc | Output mới, không ghi đè raw; inventory được kiểm tra đầu/cuối; fingerprint size/mtime không phải checksum raw. |
| Protocol | Rolling validation và kiểm tra label availability theo độ trễ giả định; không refit base model sau calibration. |

Audit trùng hiện có cho thấy không có full-row duplicate trong ba nguồn; trx có
một nhóm cùng khách/timestamp. Không đủ căn cứ deduplicate theo timestamp. Xem
[duplicates.csv](../../mbd_mini_eda/tables/descriptive.quality/duplicates.csv).

## 3. Khóa lịch đánh giá và chống lộ tương lai

Protocol chính giữ như đã thống nhất, **với giả định độ trễ nhãn 0**:

| Bước | Nhãn train | Tháng đánh giá |
|---|---|---|
| Validation 1 | 02–08/2022 | 09/2022 |
| Validation 2 | 02–09/2022 | 10/2022 |
| Validation 3 | 02–10/2022 | 11/2022 |
| Fit base model cuối | 02–11/2022 | Chấm điểm tháng 12 để fit calibrator |
| Calibration | Nhãn 12/2022, base model cố định | Fit quy tắc hiệu chỉnh đã chọn từ trước |
| Test cuối | Base model và calibrator cố định | 01/2023 |

Mỗi split phải fit riêng vocabulary/top mã, scaler/imputer nếu dùng, ngưỡng cap,
rank, PCA, feature selection, class weights và fallback. Không học các bước này
từ toàn bộ snapshot. [Hướng dẫn leakage của scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

Tháng 12 không được dùng thêm để chọn feature, early stopping, số cây, trọng số
lớp hoặc chọn giữa nhiều calibrator. Khóa chúng bằng rolling validation trước đó;
tháng 12 chỉ fit calibrator đã chọn. Không đưa nhãn tháng 12 vào base model rồi
tái dùng calibrator cũ. Dữ liệu dùng fit calibrator phải tách khỏi dữ liệu fit
classifier; khi triển khai không dùng random CV mặc định để thay lịch thời gian.
[Tài liệu calibration](https://scikit-learn.org/stable/modules/calibration.html#usage).

Độ trễ nhãn thật chưa được công bố. Kế hoạch trên chỉ là nghiên cứu offline:
nhãn tháng 12 sẵn đúng 01/01/2023 theo giả định, chưa tính thời gian fit/chuẩn bị.
Độ trễ dương hoặc yêu cầu thời gian vận hành thực cần dựng lại lịch cả train/tuning/
calibration, không chỉ đổi một tham số rồi giữ nguyên các tháng. Đây không phải
điều kiện phải chờ nhà phát hành trả lời mới được nghiên cứu baseline.

Khách có thể xuất hiện trong nhiều tháng train/test vì bài toán là dự đoán tương
lai cho khách hiện hữu. Không random-split dòng khách–tháng. Nếu muốn kết luận về
khách hoàn toàn mới, cần thêm thí nghiệm giữ riêng ID, fit biểu diễn không dùng ID
test. Không biến `client_id` hoặc `fold` thành predictor.

Nhãn/phân bố tháng 01 đã được xem trong EDA: không gọi đây là test mù chưa từng
được khám phá. Từ lúc khóa protocol, không dùng metric tháng 01 để điều chỉnh.
Một tháng test cũng không chứng minh ổn định ở nhiều giai đoạn tương lai.

## 4. Đối chứng baseline và đo giá trị từng nguồn

Đề xuất thứ tự thực thi sau khi vượt các kiểm tra chuẩn bị:

| Thí nghiệm | Nội dung |
|---|---|
| B0 | Điểm hằng bằng prevalence train; đối chiếu log loss/Brier. Top-K của điểm hằng dùng tie-break cố định không nhìn nhãn. |
| B1 | Chọn ngẫu nhiên K khách; ghi seed và kỳ vọng; thêm xếp hạng riêng theo count/recency trx trước cutoff. |
| B2 | Một CatBoost binary, Logloss, phân bố tự nhiên, chỉ feature trx thống kê hiện có; chưa dùng trọng số lớp/sampling/embedding/PCA. |
| B3 | B2 + danh tính/cơ cấu mã trx nếu bổ sung; vocabulary và gom hiếm học trên train. |
| B4 | Bản trx đã chọn + metadata có/vắng/count/recency của geo/dialog. |
| B5 | B4 + nội dung geo, rồi dialog embedding, từng bước riêng. |

Mọi thí nghiệm dùng cùng khách, tháng, nhãn và K. Không yêu cầu đủ ba nguồn; không
bỏ cold-start hoặc heavy users để làm metric đẹp hơn. **Cờ `cold_start` toàn cục
dùng cả ba nguồn**: chỉ giữ để phân nhóm báo cáo trong B2/B3, không làm predictor
của baseline được gọi là "chỉ trx". Có thể thêm nó ở B4. Tương tự, không thêm
metadata geo/dialog vào B2 rồi diễn giải là mô hình chỉ dùng giao dịch.

Nếu thử class weight, tính tỷ số trên train và đánh giá validation/test trên phân
bố tự nhiên. Chỉ chọn một trong `class_weights`, `auto_class_weights`,
`scale_pos_weight`, không kết hợp chúng. [Tham số CatBoost](https://catboost.ai/docs/en/references/training-parameters/common).
Chưa cần SMOTE, lấy mẫu âm, mô hình chuỗi hoặc clustering cho baseline đầu.

Metadata thiếu nguồn là tín hiệu có thể sử dụng trong tập phát hành, nhưng có thể
phản ánh cách thu thập. So sánh metadata với nội dung để không quy toàn bộ cải thiện
cho thông tin hội thoại/vị trí khi thực ra chỉ đến từ việc có log.

## 5. Metric phải được cố định trước khi xem kết quả mô hình

- Chính: Recall/Precision/Lift@1%; `K=ceil(0,01*N)` theo từng tháng. Với N=100.224,
  K=1.003; thêm top 0,5% và 5% làm phân tích phụ. Không dùng ngưỡng xác suất 0,5
  thay cho ngân sách top-K.
- AP, ROC-AUC bổ sung; AP dùng định nghĩa `average_precision_score`, không tự đồng
  nhất với diện tích PR tính bằng hình thang.
  [Định nghĩa AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
- Log loss, Brier và calibration trên xác suất, thêm tổng số dương dự báo so với
  thực tế. Báo cáo trước/sau calibration riêng; calibration không thay cho lựa
  chọn bộ xếp hạng. Ghi quy tắc ties nếu calibrator tạo các điểm bằng nhau.
- Chỉ số trên toàn quần thể có nhãn và theo nhóm: có/thiếu nguồn, cold-start, ít
  log, heavy trx, lịch sử dài/ngắn. Không có lớp dương hoặc thiếu một lớp thì đánh
  dấu metric tương ứng không xác định, không tự điền 0 hoặc 0,5.
- Báo cáo từng tháng; so sánh paired trên cùng khách, bootstrap theo `client_id`
  khi gộp tháng. Bootstrap theo khách không đo hết bất định do drift thời gian.
- Lưu tie-break cố định dựa trên hash ID + seed, dùng ID chỉ để phá hòa, không để
  mô hình học. Điểm dự báo phải đầy đủ, hữu hạn trong [0,1], không trùng khóa.

Đối chiếu số học từ bảng audit, **không phải metric của mô hình đã fit**:

| Phạm vi | Tỷ lệ y_any dương |
|---|---:|
| Train cuối 02–11/2022 | 1,1394% |
| Calibration 12/2022 | 0,7573% |
| Test 01/2023 | 0,6735% (675 dương) |

Đây là thay đổi prevalence đã quan sát, không chứng minh cơ chế drift cụ thể.
Prevalence train dùng làm điểm hằng phải lấy từ train, không lấy 0,6735% của test.
Không mặc định calibration tháng 12 sẽ sửa được sai lệch tháng 01.

Với test 100.224 khách và K=1.003, chọn ngẫu nhiên có kỳ vọng **6,755 khách dương**,
Recall khoảng **1,0008%**, Lift=1 theo kỳ vọng. Một ca dương thêm/bớt tương ứng
**0,1481 điểm phần trăm Recall** (1/675). Accuracy của phương án luôn dự đoán âm
đã **99,3265%**, nên accuracy cao không đủ để chứng minh baseline hữu ích.

Không dùng benchmark ROC-AUC của bốn target trên split theo khách làm ngưỡng buộc
phải đạt cho `y_any` trên split theo thời gian; target/cohort/protocol khác nhau.

## 6. Data gate cần kiểm tra khi được thực thi trên máy đích

Các điều sau là tiêu chí chấp nhận, **chưa được chạy trong review này**:

1. Tests chuẩn bị và kiểm tra hợp đồng notebook đạt; bổ sung test all-currency-
   unknown, cột lạ/nhãn không lọt X và reader từ chối output chưa ready.
2. Manifest ready; số raw = accepted + quarantine theo từng nguồn; inventory không
   đổi. Kiểm tra lý do quarantine trước khi chấp nhận kết quả.
3. Đủ 100.224 khách/tháng, 1.202.688 dòng toàn kỳ, 0 trùng `(client_id,cutoff)`;
   số dương từng tháng khớp [monthly.csv](../any_product/monthly.csv) nếu cùng raw.
4. Kiểm tra event ngay trước/đúng/sau cutoff và hai biên cửa sổ; thêm/sửa event
   tương lai không làm đổi feature quá khứ. Không lấy trạng thái "không từng có
   log trong toàn snapshot" làm predictor quá khứ.
5. Trong các cửa sổ đủ phạm vi: count30 <= count84 <= countall; active days <=
   event count; recency không âm; tỷ trọng thuộc [0,1]. Predictor số chỉ hữu hạn
   hoặc null. Phân biệt null ngoài phạm vi với count0 trong phạm vi.
6. Với trx: tổng main/foreign/missing/invalid currency khớp event count; kiểm tra
   unknown flag; amount theo currency riêng, amount invalid không làm mất event.
7. Đọc mô hình theo danh sách feature được duyệt, giữ dtype/thứ tự nhất quán;
   không đưa key, nhãn, fold, tháng/cutoff hoặc cột phụ trợ nguồn vào X vô tình.
8. Lưu prediction ngoài thời gian của mọi khách có nhãn, kể cả cold-start; kiểm
   tra key/cohort/số dòng trước khi tính metric hoặc so sánh thí nghiệm.

Với snapshot mới hoặc dữ liệu bị thay đổi, không ép các số audit cũ thành chân lý;
dừng đối soát và tạo lại audit phù hợp thay vì âm thầm loại dòng để khớp tổng.

## 7. Môi trường, bộ nhớ và khả năng tái lập

`.venv/pyvenv.cfg` ghi CPython **3.14.2**; metadata hiện có DuckDB 2.0.0, numpy
2.5.4, pandas 3.1.0, pyarrow 26.0.0. Không thấy dist-info của CatBoost/scikit-learn.
Đây là kiểm tra metadata, không phải chạy import hay smoke test thư viện.

Đã kiểm tra [PyPI CatBoost](https://pypi.org/project/catboost/): bản 1.2.10 có wheel
CPython 3.14 Windows x86-64; không có căn cứ buộc đổi Python chỉ vì đang dùng 3.14.
Wheel tồn tại không chứng minh toàn bộ môi trường đã tương thích; sau cài cần
kiểm tra và lưu phiên bản cụ thể. Chưa cài đặt hoặc thay môi trường trong review.

Theo số feature hiện tại, cấu hình mặc định dự kiến **215 predictor số**; nếu nạp
toàn bộ 1.202.688 dòng dưới dạng float64, riêng ma trận số khoảng **2,07 GB**.
Mean + last 768 chiều thêm khoảng **14,78 GB** cho 12 tháng, chưa tính pandas,
keys, bản sao và bộ nhớ mô hình. Đây là ước lượng dense, không phải peak RAM đo
được; dtype thực tế và cách đọc có thể làm khác. Giữ embedding tắt ở lần đầu.

Không chạy DuckDB chuẩn bị và CatBoost đồng thời chỉ dựa trên giới hạn `6GB` của
DuckDB; giới hạn đó không bao gồm toàn tiến trình. Đóng bước chuẩn bị trước khi
fit, đọc đúng các tháng cần thiết và đo RAM/đĩa/thời gian. `iter_supervised_batches`
giúp đọc theo batch, không đồng nghĩa CatBoost hỗ trợ `partial_fit` theo từng batch.
Pipeline hiện không resume; output đã tồn tại bị từ chối, nên lưu vết lần lỗi và
dùng thư mục mới khi chạy lại.

Mỗi thí nghiệm cần lưu: commit + hash code/config, inventory, feature schema có
thứ tự và dtype, protocol, phiên bản dependency, seed, số cây/early stopping,
params, thời gian/RAM, model/calibrator, prediction từng tháng và metric. Không
chỉ lưu một ảnh ROC hoặc một con số AUC.

## 8. Các điểm có thể giữ như giả định, không cần chờ để làm baseline offline

- UTC và cutoff đầu tháng như người dùng đã chốt; timezone ETL thật chưa xác minh.
- `y_any` là mua ít nhất một trong **bốn sản phẩm có nhãn**, không phải mọi sản phẩm
  ngân hàng, nhu cầu tiềm ẩn hay tác động do được liên hệ.
- Currency11 là mã chính theo quy ước; không đặt tên tiền hoặc đổi đơn vị. Mã hiếm
  `other_currency` có thể dùng categorical grouping, không cộng tiền như cùng đơn vị.
- Giữ heavy users và cực trị đã audit, chưa có bằng chứng lỗi. P99/P99,9 toàn kỳ
  chỉ mô tả EDA; nếu cap/rank phải fit train và chọn bằng validation.
- Giữ khách thiếu nguồn. Mã categorical ẩn danh vẫn dùng được; không diễn giải
  src/dst thành ngành nghề, geohash thành địa chỉ hoặc lấy hiệu mã làm khoảng cách.
- Embedding có thể để giai đoạn sau. Không gian vector MBD đã được ẩn danh nên
  mean/cosine/PCA cần đối chứng; không gán ý nghĩa ngữ nghĩa chắc chắn cho từng chiều.
- Universe offline là 100.224 ID trong snapshot, chưa xác minh điều kiện có thể
  liên hệ/ngày mở tài khoản hoặc quyền sở hữu sản phẩm tại mỗi cutoff thực tế.
- Nhãn mua không có nhóm can thiệp/chi phí/lợi nhuận, nên baseline chưa chứng minh
  uplift, ROI hay hiệu quả chiến dịch. Ngân sách top1% vẫn là giả định nghiên cứu.

Các hạn chế nguồn và bằng chứng đã được ghi trong [bản trả lời nghiệp vụ/dữ liệu](../open_questions/answers.md),
[audit các điểm còn lại](../remaining_questions/assessment.md) và
[rà soát cực trị](../tail_review/assessment.md). Chưa cần thực hiện thêm EDA rộng
chỉ để trì hoãn baseline; ưu tiên sửa F01, hoàn thiện hợp đồng baseline và xác nhận
pipeline trên máy được phép chạy.
