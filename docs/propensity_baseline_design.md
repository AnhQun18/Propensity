# Baseline tháng gần nhất và lịch sử tích lũy cũ

Ngày: 10/11/2026. Đã triển khai và chạy baseline trên MBD-mini: feature đủ
12 tháng, validation cuốn chiếu 09–11/2022. Xem
[báo cáo baseline và so sánh với rule mua tháng trước](../BASELINE_REPORT.md).
Kiểm thử bằng fixture được phân biệt với metric thực nghiệm trên dữ liệu thật.

**Rà soát sau phản biện:** đặc tả dưới mô tả run hành vi đã khóa, không phải
kiến trúc cuối. [Mục 9 của báo cáo](../BASELINE_REPORT.md#9-đối-chiếu-phản-biện-và-thứ-tự-ưu-tiên)
điều chỉnh ưu tiên: đối chứng lịch sử target đã khả dụng và kết hợp hành vi;
thử cửa sổ 7/14/30/90 ngày, ablation nguồn và learning curve trong run mới.
Chẩn đoán nhiều K/phân nhóm/overlap đã tính từ dự đoán validation cũ; các model
nâng cấp chưa train. Nhãn tháng hiện có chưa đủ dựng target trượt 30 ngày theo tuần.

**Cập nhật theo yêu cầu tiếp theo:** baseline mặc định dùng cả **trx + geo +
dialog**, với hai biến thể `multi_recent` và `multi_history`. Trx-only giữ làm
đối chứng tùy chọn. Đã bổ sung rule `y_hat(M)=y_any(M−1)` và lệnh `compare-rule`.
Dialog hiện dùng count/recency/độ phủ; embedding nội dung chưa nằm trong baseline.
Kết quả dữ liệu thật được ghi riêng trong báo cáo so sánh liên kết ở trên.

## 1. Bài toán và dữ liệu

Tại đầu tháng M, xếp hạng khách hiện hữu theo xác suất mua ít nhất một trong
bốn sản phẩm có nhãn trong tháng M. Một mẫu là một khách–tháng. Dùng `y_any=1`
khi có ít nhất một target bằng 1, bằng 0 khi cả bốn bằng 0; còn lại chưa biết.
Không dùng dòng chưa biết nhãn để fit hoặc tính metric; vẫn xuất dự đoán cho
mọi khách. Khi có nhãn thiếu, metric được tính trên quần thể có nhãn và phải
đọc cùng `scoring_rows`/`labelled_rows`.

Quy ước nghiên cứu: UTC, cutoff đầu tháng, độ trễ nhãn 0. Đây chưa phải timezone
hoặc SLA của ETL nhãn gốc. Không dịch `mon` thêm tháng.

Ví dụ dự đoán tháng 12/2022:

| Phần | Khoảng sự kiện |
| --- | --- |
| `recent` | `[01/11/2022 00:00, 01/12/2022 00:00)` |
| `older` | Mọi sự kiện trước `01/11/2022 00:00` |
| `all` | Mọi sự kiện trước cutoff; chỉ dùng audit/phân nhóm |
| Nhãn | Tháng 12/2022, dòng nguồn `mon=2022-12-31` |

Hai nhóm `recent` và `older` không giao nhau. Một tháng lịch có thể dài 28/29/30/31
ngày. Count và active days hai nhóm phải cộng lại bằng `all`; median, max,
nunique, top-share được tính trực tiếp trong mỗi nhóm. Mọi recency, kể cả của
`older`, đều tính đến cutoff dự đoán.

## 2. Đặc trưng

Tái sử dụng thống kê đã kiểm chứng của từng nguồn: count, ngày hoạt động,
recency, tuổi bản ghi đầu, độ phủ; trx có amount theo mã chính 11, currency,
cardinality/missing và top-share; geo có cardinality/top-share; dialog dùng
metadata. Embedding tắt cho baseline v1.

Thêm cho mỗi nguồn:

```text
recent_event_rate = recent_event_count / recent_source_window_overlap_days
older_event_rate  = older_event_count / older_source_window_overlap_days
event_rate_difference = recent_event_rate - older_event_rate
```

Mẫu số là số ngày chồng với phạm vi trích xuất của nguồn, không phải số ngày
chứng minh khách được quan sát liên tục. Nếu không có phạm vi nguồn hoặc không
đủ dữ liệu để xác định rate thì giữ null. Không coi ngày đầu có log là ngày mở
tài khoản; không dùng log cuối trong tương lai làm dấu hiệu độ phủ tại cutoff.

Schema v3 có **230 predictor được duyệt**, gồm các trường `all` và cohort để
audit. Mỗi mô hình chỉ đọc subset được khóa ở `schema.variants`:

| Mã | Đầu vào | Predictor | Câu hỏi |
| --- | --- | ---: | --- |
| B0 | Xác suất nền từ nhãn train | 0 | Mốc log loss/Brier; random expectation cho top-K |
| B1 `trx_recent` | Trx tháng gần nhất | 50 | Hành vi gần đây dự đoán được đến đâu? |
| B2 `trx_history` | Trx gần nhất + cũ + chênh lệch rate | 101 | Lịch sử cũ có giá trị tăng thêm không? |
| B3 `multi_recent` | Ba nguồn trong tháng gần nhất | 75 | Thêm nguồn có ích không? |
| B4 `multi_history` | Ba nguồn gần nhất + cũ + chênh lệch rate | 153 | Kết hợp lịch sử và nguồn có cải thiện không? |

`all`, `ever_seen`, `cold_start`, ID, cutoff và target không tự động được đưa
vào X. Cờ toàn cục phục vụ báo cáo cohort; vì vậy B1/B2 không nhận thông tin
geo/dialog qua `cold_start`.

## 3. Mô hình và lịch đánh giá

CatBoost binary, CPU, loss Logloss, NaN giữ nguyên; phân bố lớp tự nhiên,
không undersample, không class weight ở vòng đầu. Seed 20261110, hai threads.
Baseline mặc định mới: depth 4, 200 cây, learning rate 0,04, L2 leaf 5.
Hai biến thể đa nguồn × một cấu hình × ba fold = sáu lần fit validation.
Grid 4/6 × 200/500 cây vẫn có thể cấu hình cho nghiên cứu tiếp theo, với cùng
lịch đánh giá. Run đã khóa không dùng early stopping; nghiên cứu tiếp theo cần
learning curve và tập dừng bên trong train để đánh giá số cây phù hợp.
Đây là điểm xuất phát có giới hạn tính toán, không phải thông số tối ưu đã biết.

| Bước | Nhãn train | Tháng đánh giá |
| --- | --- | --- |
| Validation 1 | 02–08/2022 | 09/2022 |
| Validation 2 | 02–09/2022 | 10/2022 |
| Validation 3 | 02–10/2022 | 11/2022 |
| Fit base model cuối | 02–11/2022 | Chấm điểm 12/2022 |
| Fit sigmoid calibrator | Dự đoán base model và nhãn 12/2022 | Không báo như metric độc lập |
| Test cuối | Base model/calibrator đã khóa | 01/2023 |

Calibration: sigmoid hồi quy logistic trên raw margin, L2, C=1; không chọn lại
loại calibrator bằng tháng 12. Nếu fit không hội tụ hoặc slope không dương,
luồng dừng để kiểm tra. Base model không được refit sau calibration. Ranking
dùng raw margin; xác suất sau calibration dùng cho log loss/Brier/calibration.

Lịch trên khả thi dưới giả định độ trễ nhãn 0. Thời gian tính toán khi tháng
vừa kết thúc chưa được mô phỏng; không diễn giải thành lịch phục vụ thực tế.

## 4. Metric và chọn ứng viên

- Primary: **Recall@1% trung bình đều ba tháng validation**.
- Tie-break: AP trung bình cao hơn, log loss trung bình thấp hơn, ít predictor,
  depth nhỏ hơn rồi ít cây hơn. Khóa trước khi chạy.
- K = ceil(1% × quần thể có nhãn trong tháng); với đủ 100.224 nhãn, K=1.003.
  Báo cáo thêm 0,5% và 5%; không xếp hạng chung các tháng.
- Ties theo SHA-256(seed:client_id), rồi chuỗi ID để ổn định nếu hash trùng;
  ID chỉ dùng tie-break, không dùng làm predictor.
- Báo Precision/Recall/Lift@K, AP, ROC-AUC, log loss, Brier và calibration bins
  cố định. Xác suất nền lấy từ train của từng fold, không lấy nhãn tháng dự đoán.
- So sánh paired bootstrap theo khách, cùng khách được lấy mẫu ở mọi tháng;
  tính lại top-K trong quần thể resample. Mặc định 500 lần, khoảng phân vị 95%.
  Khoảng này có điều kiện theo mô hình/các tháng đã quan sát; không đo được
  biến động của những tháng tương lai chưa có dữ liệu. Vì validation đồng thời
  chọn cấu hình, các khoảng validation chỉ là phân tích thăm dò.
- Test báo thêm cold-start, có lịch sử, không trx/geo/dialog. Top-K được tính
  lại trong từng cohort; các cohort có thể chồng lấn.

Ứng viên được chọn tự động là mô hình đứng đầu theo quy tắc trên. Khi đề xuất
mô hình chốt, đọc thêm chênh lệch B2−B1, B3−B1, B4−B2 và B4−B3 cùng biến động
từng tháng. Nếu lợi ích của phương án phức tạp chưa rõ, ưu tiên phương án đơn
giản cho thí nghiệm tiếp theo trước khi mở test. Đổi lựa chọn phải tạo run mới;
không sửa `selection.json` đã khóa. Test chỉ dùng xác nhận, không chọn lại.

Không đặt mục tiêu lift hay mức tăng doanh thu tùy ý. Lift >1 là điều kiện
tối thiểu vượt kỳ vọng ngẫu nhiên, chưa chứng minh giá trị chiến dịch hay ROI.

## 5. Triển khai và chạy

Mã mới: `src/propensity_baseline/`. Bước calendar đọc **cache chuẩn hóa** trong
`outputs/propensity_processing_v3/work.duckdb` ở chế độ read-only, không cần
quét/chuẩn hóa raw lại. Median/cardinality vẫn được tổng hợp lại theo cửa sổ mới.
Output mới độc lập, từ chối ghi đè. Output cũ tiếp tục dùng hợp đồng v2.
Khi cache thuộc run chuẩn bị ban đầu chưa thay đổi, tái dùng aggregate `all`
đã tính ở đúng cutoff: `older` của tháng M là `all` của tháng M−1, cộng số ngày
tháng M−1 vào recency để cùng neo ở cutoff M. Các thống kê median/nunique không
bị lấy hiệu. Các file cache tham gia inventory; file sửa sau run làm tắt reuse.
Kiểm thử đã đối chiếu toàn bộ giá trị 12 tháng giữa reuse và tính trực tiếp.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install -e ".[baseline]"

# Chỉ đọc cấu hình, không xử lý dữ liệu hoặc fit.
.\.venv\Scripts\python.exe -B -X utf8 -m propensity_baseline calendar-plan --config configs/propensity_calendar.json
.\.venv\Scripts\python.exe -B -X utf8 -m propensity_baseline plan --config configs/propensity_baseline.json

# Tạo feature tháng lịch mới; không fit.
.\.venv\Scripts\python.exe -B -X utf8 -m propensity_baseline calendar --config configs/propensity_calendar.json

# Chỉ train/validation; lưu cấu hình được chọn, chưa mở tháng 12 hoặc test.
.\.venv\Scripts\python.exe -B -X utf8 -m propensity_baseline tune --config configs/propensity_baseline.json

# So sánh validation với rule mua tháng trước; không mở test.
.\.venv\Scripts\python.exe -B -X utf8 -m propensity_baseline compare-rule --config configs/propensity_baseline.json

# Fit mô hình cuối và calibration, chưa mở test.
.\.venv\Scripts\python.exe -B -X utf8 -m propensity_baseline fit --config configs/propensity_baseline.json

# Chỉ chạy khi đã chốt toàn bộ thiết kế/mô hình; mở test đúng một lần trong run.
.\.venv\Scripts\python.exe -B -X utf8 -m propensity_baseline test --config configs/propensity_baseline.json

# Chỉ sau khi đã mở test ở bước trên.
.\.venv\Scripts\python.exe -B -X utf8 -m propensity_baseline compare-rule --stage test --config configs/propensity_baseline.json
```

Cài editable để dùng module từ repository. Có thể copy cấu hình thành
`*.local.json` rồi đổi đường dẫn. Calendar cần đầy đủ output notebook 03 gồm
`work.duckdb`; một file merge `processed/propensity.parquet` cũ không đủ để
khôi phục median/cardinality trong các cửa sổ mới. Không dùng notebook 04 hiện
tại để merge schema v3; trainer đọc trực tiếp file tháng v3.

Luồng lưu metadata thư viện/config/seed, hash code và selection, inventory
feature theo path/size/mtime; model/calibrator có SHA-256 nội dung. Sau tuning,
đổi code/thư viện/input/config hoặc selection đều bị từ chối ở bước tiếp theo.
Fingerprint feature dùng metadata, chưa phải hash nội dung toàn bộ Parquet.
Run lỗi được giữ để truy vết; dùng output mới khi chạy lại, không tự xóa.

## 6. Đề xuất sau baseline

Ưu tiên đọc các đối chứng để quyết định bước tiếp:

1. B2 hơn B1 ổn định: tiếp tục lịch sử cũ, thử xu hướng theo tháng hoặc giảm
   trọng số sự kiện xa. Nếu chưa hơn, giữ B1 làm mốc và kiểm tra cách chuẩn hóa.
2. B3/B4 có ích: thử nội dung categorical theo vocabulary chỉ học trên train,
   rồi dialog embedding. Không kết luận nguồn vô ích chỉ vì metadata chưa giúp.
3. Thử LightGBM/XGBoost trên cùng feature/split/ngân sách tuning; sau đó mới
   thử encoder chuỗi và mô hình kết hợp. Giữ các nghiên cứu này ngoài test khóa.

Validation hiện tại chọn `multi_history`; chưa mở test hoặc chứng minh mô hình
này tốt hơn các boosting/encoder khác chưa được chạy. Quyết định cho giai đoạn
tiếp theo dựa vào validation, độ ổn định và chi phí.

## Đối chứng mua tháng trước

Rule dự đoán `y_any(M)=y_any(M−1)`: mua tháng trước thì dự đoán mua tháng này,
không mua tháng trước thì dự đoán không mua tháng này. Đây là nhãn gộp bốn sản
phẩm, không yêu cầu mua lại đúng cùng một sản phẩm.

- Ghép đúng tháng lịch trước đó theo ID, không dùng `shift` trên các dòng còn
  lại khi thiếu tháng. Thiếu nhãn trước hoặc nhãn chưa sẵn tại cutoff → unknown.
- Nhãn 01/2022 không được phát hành nên rule không có dự đoán cho 02/2022.
  Validation 09–11/2022 có các tháng trước tương ứng 08–10/2022.
- So sánh ML và rule trên cùng khách có nhãn hiện tại và nhãn trước sẵn có;
  báo rõ số khách bị loại vì thiếu nhãn trước. Không điền nhãn thiếu thành 0.
- So sánh top 1% cùng K, tie-break cố định trong từng nhóm điểm 0/1 của rule.
  Báo thêm AP và Precision/Recall/Lift ở 0,5%/5%.
- Báo confusion matrix/precision/recall/F1 khi rule chọn toàn bộ người mua
  tháng trước. ML được chọn đúng bằng số khách đó để so ở cùng ngân sách;
  không đánh giá ML ở threshold 0,5 rồi so với một rule có ngân sách khác.
- Rule là quyết định nhị phân, không phải xác suất đã hiệu chỉnh; không báo
  log loss của rule như một xác suất đáng tin cậy. Nhãn tháng trước cũng không
  được tự động đưa vào X của CatBoost trong phép so sánh này.
- Output `rule_validation/` hoặc `rule_test/` chứa dự đoán, bảng số liệu theo
  tháng, metric ở ngân sách tự nhiên và paired bootstrap chênh Recall@1%.

## 7. Kết quả kiểm chứng triển khai

- 7/7 kiểm thử baseline đạt: biên tháng, aggregate phi tuyến của lịch sử cũ,
  sự kiện tương lai không thay đổi feature quá khứ, metric/tie-break/bootstrap,
  notebook 05, cached aggregate bằng tính trực tiếp và toàn bộ tune →
  fit/calibration → test bằng CatBoost thật trên fixture.
- 6/6 kiểm thử rule đạt: đúng tháng trước, nhãn thiếu/chưa sẵn, không lấy kết
  quả tháng hiện tại làm predictor, cùng cohort và cùng ngân sách.
- Ca tích hợp chạy cả bốn biến thể, kiểm tra tháng calibration/test không được
  đọc trước đúng bước, selection sửa tay bị từ chối, output không bị ghi đè,
  khách thiếu nhãn vẫn nhận dự đoán và test không được mở lại trong run.
- Triển khai ban đầu đã có 13/13 kiểm thử chuẩn bị dữ liệu, 2/2 kiểm tra
  notebook 02/03 và 3/3 kiểm thử merge đạt; đợt cập nhật này chạy 7 baseline
  và 6 rule nêu trên, cùng kiểm tra lại notebook 05 sau cập nhật.
- Môi trường kiểm thử: CatBoost 1.2.10, scikit-learn 1.9.1, NumPy 2.5.4,
  pandas 3.1.0, SciPy 1.18.1; `pip check` không báo dependency bị hỏng.
- Đã tạo calendar feature đủ 1.202.688 dòng, 12.854 dương, 230 predictor lưu
  trong schema; mô hình dùng 75 hoặc 153 predictor thuộc cả ba nguồn.
- Đã chạy sáu lần fit validation với cấu hình baseline cố định. Recall@1%
  trung bình: `multi_history` 8,70%, `multi_recent` 7,55%, rule 6,18%.
  Không gọi đây là kết quả test độc lập; xem báo cáo đầy đủ ở trên.
- Đã fit `multi_history` cuối trên 02–11/2022 và sigmoid calibration tháng 12.
  Model được tải lại kiểm tra: 153 feature (101 trx, 33 geo, 19 dialog), hash
  đúng và dự đoán mẫu calibration khớp bản lưu. Test 01/2023 chưa được đánh giá.

Nguồn kỹ thuật đã đối chiếu:

- [CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier)
  và [fit](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit).
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
- [MBD gốc](https://arxiv.org/html/2409.17587v1): tham khảo cách dùng lịch sử và
  đa phương thức; benchmark bốn sản phẩm khác target any-product của dự án.
