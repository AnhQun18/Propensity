# Báo cáo baseline dự đoán mua sản phẩm — MBD-mini

Ngày cập nhật: **10/11/2026**. Báo cáo mô tả công việc đã chạy trên dữ liệu thật và kết quả đã lưu.

**Baseline đã dùng đủ giao dịch (trx), geo và dialog metadata.** Mô hình được chọn là CatBoost `multi_history`, kết hợp hành vi tháng gần nhất với lịch sử cũ. Trên validation cuốn chiếu 09–11/2022, Recall@1% trung bình đạt **8,70%**, so với **6,18%** của rule “tháng này mua thì tháng sau mua”. Mô hình cuối và bộ hiệu chỉnh xác suất đã được lưu; **test 01/2023 chưa được đánh giá**.

## 1. Bài toán là gì? Kết quả dùng để làm gì?

Tại đầu tháng M, dùng thông tin đã có trước thời điểm đó để dự đoán khách hàng có mua **ít nhất một trong bốn nhóm sản phẩm target** trong tháng M hay không.

| Thành phần | Định nghĩa hiện tại |
| --- | --- |
| Đơn vị quan sát | Một khách hàng trong một tháng: `client_id × month` |
| Tập khách | 100.224 khách/tháng trong bộ dữ liệu hiện có |
| Thời điểm dự đoán | Đầu ngày 1 tháng M, theo giả định UTC |
| Nhãn cần dự đoán | `y_any(M)`: có mua ít nhất một trong bốn target trong tháng M |
| Đầu ra mô hình | Xác suất `p_any` cho từng khách, dùng để sắp xếp ưu tiên |
| Cách mô phỏng sử dụng | Chọn top 1% khách có xác suất cao nhất; so với rule ở cùng ngân sách |

Ví dụ: dự đoán tại **01/12/2022** thì hành vi tháng gần nhất là tháng 11/2022, lịch sử cũ là phần trước tháng 11, còn target là kết quả mua trong tháng 12. Không dịch nhãn thêm một tháng sau khi đã gán đúng cutoff này.

Ý nghĩa nghiệp vụ là **ưu tiên danh sách khách có khả năng mua cao** khi nguồn lực tiếp cận có giới hạn. Baseline hiện dự đoán mua bất kỳ sản phẩm nào trong bốn target; chưa chọn sản phẩm cụ thể cho từng người. Kết quả offline chưa cho biết việc liên hệ khách có làm tăng số lượt mua hay doanh thu hay không.

## 2. Ý tưởng thiết kế

Hành vi gần đây phản ánh nhu cầu hiện tại; lịch sử dài hơn cung cấp nền hành vi và mức độ hoạt động của khách. Vì vậy, dữ liệu được tách thành hai cửa sổ không chồng lấn, thay vì chỉ gộp toàn bộ quá khứ vào một nhóm.

| Cửa sổ | Tại cutoff 01/12/2022 | Ý nghĩa |
| --- | --- | --- |
| `recent` | Từ 01/11 đến trước 01/12 | Hành vi trong tháng lịch gần nhất |
| `older` | Toàn bộ lịch sử có sẵn trước 01/11 | Nền hành vi tích lũy trước tháng gần nhất |
| `all` | Toàn bộ lịch sử trước 01/12 | Phục vụ kiểm tra dữ liệu; không đưa thêm vào X của hai baseline đã chạy |

Tháng gần nhất là **tháng lịch**, có thể có 28–31 ngày. Tất cả feature đều dừng trước cutoff. Recency được tính so với cutoff dự đoán, kể cả khi sự kiện nằm trong lịch sử cũ. Các thống kê phi tuyến như trung vị được tính từ đúng tập sự kiện, không lấy trung vị toàn bộ trừ trung vị tháng gần nhất.

Hai giả thuyết được kiểm tra:

1. Thêm lịch sử cũ có cải thiện so với chỉ dùng tháng gần nhất không?
2. Baseline có tìm được nhiều khách mua hơn rule mua tháng trước khi chọn cùng số khách không?

## 3. Input gồm những gì?

### 3.1. Các nguồn dữ liệu

| Nguồn | Cách sử dụng | Ý nghĩa |
| --- | --- | --- |
| Danh sách khách, split và mốc tháng | Xác định khách, khóa ghép và lịch đánh giá | Giữ đúng quần thể khách–tháng; ID và fold không phải predictor |
| Bốn target theo tháng | Tạo `y_any` và đối chiếu dự đoán | Là đáp án để học/đánh giá, không đưa vào X |
| `trx` | Thống kê số giao dịch, ngày hoạt động, recency, nhóm giao dịch và amount theo currency | Mức độ, tần suất và cấu trúc giao dịch |
| `geo` | Thống kê số sự kiện, ngày hoạt động, recency, độ đa dạng mã vị trí và tỷ trọng vị trí phổ biến | Nhịp hoạt động và mức độ tập trung/phân tán theo mã geo |
| `dialog` | Số sự kiện, ngày hoạt động, recency và độ phủ | Mức độ tương tác; chưa dùng nội dung hội thoại hoặc embedding |

Geo hiện dùng thống kê mã vị trí, chưa diễn giải thành địa điểm địa lý thực tế. Amount được tách theo currency; currency 11 được dùng làm nhóm chính theo quy ước hiện tại của dự án. Không cộng tiền khác currency và không mặc định mã này là một đồng tiền ISO cụ thể.

### 3.2. Nhãn và độ phủ thời gian

Nhãn `y_any = 1` khi có ít nhất một target bằng 1; bằng 0 khi đủ căn cứ xác định cả bốn đều không mua. Nếu thiếu nhãn và chưa thể kết luận thì giữ unknown, không tự chuyển thành 0.

Snapshot đã kiểm tra có nhãn từ **02/2022 đến 01/2023**, tổng cộng **1.202.688 dòng khách–tháng**, **12.854 nhãn dương**, tương đương khoảng **1,07%**; không có dòng nhãn unknown trong snapshot này. Vì vậy, việc một bảng hoặc biểu đồ chỉ hiện từ 03/2022 không đủ để kết luận tháng 02 không có người mua: tháng 02 thực tế có **1.041 nhãn dương**.

Hành vi trước khoảng thời gian có nhãn vẫn được sử dụng để tạo lịch sử cũ. Các tháng chưa có target không được biến thành mẫu âm để train. Cùng một khách xuất hiện qua nhiều tháng, nên các dòng không độc lập như khi lấy mẫu những khách khác nhau.

### 3.3. Feature thực sự được đưa vào mô hình

| Nhóm | `multi_recent` | `multi_history` |
| --- | ---: | ---: |
| Giao dịch | 50 | 101 |
| Geo | 16 | 33 |
| Dialog metadata | 9 | 19 |
| **Tổng predictor** | **75** | **153** |

`multi_recent` dùng cửa sổ gần nhất của cả ba nguồn. `multi_history` bổ sung lịch sử cũ và các feature lịch sử được schema cho phép. Đây là số biến đầu vào sau chọn schema, không phải số sự kiện hay số chiều embedding.

Feature gồm count, số ngày hoạt động, thời gian từ sự kiện gần nhất đến cutoff, thống kê phân phối, độ đa dạng, độ phủ và rate. Với rate theo ngày, mẫu số là số ngày cửa sổ giao với khoảng trích xuất của nguồn. Mẫu số này **không chứng minh khách được quan sát liên tục**; thời điểm log đầu tiên cũng không được coi là ngày mở tài khoản.

Output chuẩn bị lưu 230 predictor được schema chấp nhận, nhưng hai mô hình thực tế chỉ chọn lần lượt 75 và 153 biến. Không đưa ID, fold, target, nhãn mua tháng trước, `trans_count`, `diff_trans_date` hay các cột phụ trợ ngoài danh sách cho phép vào X. Cờ `cold_start` tổng hợp nhiều nguồn phục vụ kiểm tra/phân nhóm, không làm predictor trong baseline này.

## 4. Đã thực hiện những gì?

| Công việc đã làm | Cách thực hiện | Đầu ra/ý nghĩa |
| --- | --- | --- |
| Tái sử dụng dữ liệu đã chuẩn hóa | Đọc cache DuckDB của luồng preparation ở chế độ chỉ đọc | Không sửa raw, giữ quy tắc chuẩn hóa và quarantine đã có |
| Dựng cửa sổ thời gian | Tạo `recent`, `older`, `all` riêng cho từng cutoff | Dùng được lịch sử dài mà vẫn tách tín hiệu gần đây |
| Tổng hợp ba nguồn | Aggregate theo khách trước khi ghép vào bảng khách–tháng | Giữ một dòng/khách/tháng, tránh nhân bản do ghép log |
| Xử lý thiếu dữ liệu | Phân biệt số sự kiện bằng 0 với thống kê không xác định; giữ NaN phù hợp | Không biến mọi trường thiếu thành 0 |
| Khóa đầu vào mô hình | Kiểm tra tên/thứ tự/dtype, giá trị vô hạn và tràn khi chuyển float32 | Hạn chế lệch schema và lỗi số học |
| Huấn luyện validation | Chạy hai biểu diễn feature trên ba fold thời gian: tổng cộng sáu lần fit | So sánh tháng gần nhất với tháng gần nhất + lịch sử cũ |
| So với rule | Cùng khách, cùng tháng, cùng ngân sách; phá hòa cố định | Đo phần cải thiện so với cách chọn khách đơn giản |
| Ước lượng bất định | Bootstrap theo khách, giữ các tháng của khách đi cùng nhau | Tôn trọng việc một khách lặp lại qua thời gian |
| Fit bản cuối và calibration | Fit base trên 02–11/2022; fit sigmoid trên 12/2022 | Lưu model và bộ chuyển đổi xác suất để đánh giá sau |
| Kiểm tra triển khai | 7 kiểm thử baseline và 6 kiểm thử rule đạt; tải lại model và so 128 dự đoán calibration | Dự đoán tải lại khớp bản lưu, sai số lớn nhất bằng 0 |

Đã tạo đủ feature cho 12 tháng trên dữ liệu thật. Các cửa sổ có cùng tập sự kiện chỉ được tái sử dụng khi đúng cutoff và đúng điều kiện nguồn; không trừ hai thống kê phi tuyến để suy ra lịch sử cũ. Baseline giữ dữ liệu hiếm và cực trị theo quy trình hiện tại, không cân bằng bằng cách lấy mẫu 1:1.

## 5. Mô hình base và cách huấn luyện

### 5.1. Hai baseline đã chạy

| Tên | Input | Vai trò |
| --- | --- | --- |
| `multi_recent` | 75 biến tháng gần nhất từ trx + geo + dialog | Mốc so sánh khi chỉ quan sát hành vi gần đây |
| `multi_history` | 153 biến gần nhất + lịch sử cũ từ cả ba nguồn | Kiểm tra giá trị bổ sung của lịch sử tích lũy; được chọn làm baseline cuối |

Cả hai dùng **CatBoost phân loại nhị phân**, hàm mất mát Logloss. Cấu hình đã chạy: **200 cây, depth 4, learning rate 0,04, L2 leaf regularization 5, seed 20261110, CPU 2 threads**. Mô hình xử lý NaN với `nan_mode=Min`; không dùng class weight, lấy mẫu cân bằng hoặc early stopping.

Lệnh có tên `tune`, nhưng lần chạy này chỉ so sánh hai biểu diễn feature với cấu hình cố định. Chưa thực hiện tìm kiếm siêu tham số rộng. Mã có hỗ trợ đối chứng chỉ trx; những đối chứng đó chưa nằm trong kết quả chạy thật của báo cáo này.

### 5.2. Chia thời gian để tránh nhìn trước tương lai

| Giai đoạn | Dữ liệu học | Dữ liệu đánh giá/hiệu chỉnh | Số dòng học |
| --- | --- | --- | ---: |
| Validation fold 1 | 02–08/2022 | 09/2022 | 701.568 |
| Validation fold 2 | 02–09/2022 | 10/2022 | 801.792 |
| Validation fold 3 | 02–10/2022 | 11/2022 | 902.016 |
| Fit base cuối | 02–11/2022 | Dành 12/2022 cho calibration | 1.002.240 |
| Calibration | Giữ nguyên base đã fit | 12/2022: 100.224 dòng, 759 nhãn dương | — |
| Test độc lập | Giữ nguyên model và calibrator | 01/2023: chưa đánh giá | — |

Không chia ngẫu nhiên các dòng khách–tháng. Mỗi fold chỉ học từ nhãn tháng trước và dự đoán tháng sau. Feature của từng dòng chỉ dùng sự kiện trước cutoff của dòng đó.

Protocol hiện giả định **UTC và độ trễ nhãn bằng 0**. Đây là giả định triển khai, chưa phải xác nhận về timezone/SLA cung cấp nhãn của hệ thống nguồn. Nếu nhãn tháng trước đến chậm, cần sửa lịch khả dụng cho cả train và rule.

Chọn biến thể theo Recall@1% trung bình các tháng; các tiêu chí phụ là AP, Logloss và độ đơn giản. Sau khi chọn `multi_history`, base được fit lại trên 02–11/2022. Sigmoid calibration dùng raw margin của base trên 12/2022, với LogisticRegression L2, `C=1`; base không được fit lại sau bước này.

Các điểm validation bên dưới thuộc **ba model theo fold**, trước calibration. Chúng không phải điểm test của model cuối đã học đến tháng 11, cũng chưa chứng minh chất lượng xác suất sau calibration trên một tháng độc lập.

## 6. Rule “tháng này mua thì tháng sau mua” được so thế nào?

Rule được viết chính xác là **`ŷ_any(M) = y_any(M−1)`**: khách mua bất kỳ một trong bốn target tháng trước thì dự đoán tháng này cũng mua bất kỳ target nào. Không yêu cầu mua lại đúng cùng sản phẩm.

Nhãn tháng trước chỉ dùng cho rule, không đưa vào X của CatBoost. Phải tìm đúng tháng lịch liền trước; nếu thiếu hoặc chưa khả dụng thì giữ unknown. Vì không có nhãn 01/2022, rule chưa thể chấm cho 02/2022. Với ba tháng validation 09–11/2022, nhãn tháng trước có đủ cho toàn bộ khách, nên không loại dòng nào vì lý do này.

Hai phép so sánh đã chạy:

1. **Cùng top 1%:** mỗi phương pháp chọn `ceil(0,01 × 100.224) = 1.003` khách/tháng. Rule chỉ có điểm 0/1 nên nhiều khách đồng điểm; dùng SHA256 của seed và ID để phá hòa cố định, không dùng target hiện tại.
2. **Cùng ngân sách tự nhiên của rule:** rule chọn tất cả người mua tháng trước; mô hình chọn đúng bằng số khách đó, theo xác suất giảm dần.

Rule là quyết định 0/1, không phải xác suất đã hiệu chỉnh; không báo Logloss của rule như một mô hình xác suất. Khi so danh sách khách, không dùng threshold 0,5 cho mô hình vì sẽ tạo ngân sách khác rule.

## 7. Baseline đã đạt được gì?

### 7.1. Kết quả chính: cùng top 1%

Các giá trị sau là **trung bình đều theo ba tháng validation 09–11/2022**.

| Phương pháp | Precision@1% | Recall@1% | Lift@1% | AP |
| --- | ---: | ---: | ---: | ---: |
| **CatBoost gần nhất + lịch sử cũ (`multi_history`)** | **7,91%** | **8,70%** | **8,70** | **0,0419** |
| CatBoost tháng gần nhất (`multi_recent`) | 6,85% | 7,55% | 7,54 | 0,0348 |
| Rule mua tháng trước | 5,62% | 6,18% | 6,17 | 0,0125 |

Ý nghĩa của các chỉ số:

- **Precision@1% = 7,91%:** trong nhóm 1% khách được mô hình ưu tiên, trung bình khoảng 7,91% thực sự mua.
- **Recall@1% = 8,70%:** chỉ chọn 1% khách nhưng tìm được trung bình 8,70% tổng số người mua của tháng. Phần lớn người mua vẫn nằm ngoài danh sách này.
- **Lift@1% = 8,70:** tỷ lệ mua trong danh sách ưu tiên cao gấp khoảng 8,70 lần tỷ lệ mua của toàn bộ khách cùng tháng.
- **AP:** average precision, đánh giá chất lượng sắp xếp trên toàn bộ danh sách, không chỉ tại top 1%; càng cao càng tốt.

Thêm lịch sử cũ nâng Recall@1% từ **7,55% lên 8,70%**. So với rule, baseline được chọn tăng **2,52 điểm phần trăm**, tương đương khoảng **41% tương đối** trên chỉ số trung bình này. Đây là mức cải thiện quan sát được trên validation.

### 7.2. Kết quả theo từng tháng

Mỗi dòng đều so trên 100.224 khách và chọn đúng 1.003 khách cho mỗi phương pháp.

| Tháng | Người mua thực tế | `multi_history` tìm được | Rule tìm được | Recall mô hình | Recall rule |
| --- | ---: | ---: | ---: | ---: | ---: |
| 09/2022 | 1.013 | 80 | 57 | 7,90% | 5,63% |
| 10/2022 | 868 | 76 | 55 | 8,76% | 6,34% |
| 11/2022 | 867 | 82 | 57 | 9,46% | 6,57% |

Trong tổng cộng **3.009 lượt chọn khách–tháng**, mô hình tìm được **238 lượt mua**, rule tìm được **169**, chênh **69**. Đây là số người mua quan sát trong danh sách được chọn, chưa phải số lượt mua tăng thêm do một chiến dịch tác động.

### 7.3. Khi dùng ngân sách tự nhiên của rule

| Tháng | Số khách mỗi bên chọn | Mô hình tìm được | Rule tìm được | Recall mô hình | Recall rule |
| --- | ---: | ---: | ---: | ---: | ---: |
| 09/2022 | 1.132 | 89 | 70 | 8,79% | 6,91% |
| 10/2022 | 1.013 | 77 | 55 | 8,87% | 6,34% |
| 11/2022 | 868 | 73 | 55 | 8,42% | 6,34% |

Mô hình tìm được nhiều người mua hơn rule ở cả ba tháng, trong cả hai cách cố định ngân sách. Với ngân sách tự nhiên, tổng số lượt mua tìm được là **239 so với 180**.

### 7.4. Chất lượng xác suất trước calibration

| Phương pháp | ROC-AUC ↑ | Logloss ↓ | Brier score ↓ |
| --- | ---: | ---: | ---: |
| Xác suất hằng theo tỷ lệ mua của tập train | 0,5000 | 0,05239 | 0,009064 |
| `multi_recent` | 0,7471 | 0,04844 | 0,008967 |
| `multi_history` | **0,7846** | **0,04698** | **0,008914** |

Đây cũng là trung bình theo ba tháng, dùng xác suất gốc của model từng fold. `multi_history` tốt hơn mốc xác suất hằng và `multi_recent` trên cả ba chỉ số. Chưa có điểm test để kết luận mức cải thiện xác suất sau calibration.

### 7.5. Độ bất định

Với 500 lần bootstrap theo khách, khoảng 95% của chênh lệch Recall@1% là:

| So sánh với rule | Chênh lệch trung bình | Khoảng bootstrap 95% |
| --- | ---: | ---: |
| `multi_history` − rule | +2,52 điểm phần trăm | [+1,24; +3,88] điểm phần trăm |
| `multi_recent` − rule | +1,37 điểm phần trăm | [−0,08; +2,75] điểm phần trăm |

Khoảng của `multi_history` nằm trên 0; khoảng của `multi_recent` còn chứa 0. Bootstrap có điều kiện theo model đã fit và ba tháng đã quan sát, chưa phản ánh đầy đủ bất định khi train lại hoặc gặp giai đoạn mới. Validation này đồng thời được dùng để chọn biến thể, nên kết quả vẫn cần xác nhận trên test độc lập.

## 8. Output là gì? Đọc và sử dụng thế nào?

### 8.1. Đầu ra dự đoán

Bảng dự đoán lưu khóa khách/tháng, tên model, raw margin `score` và xác suất `p_any`; bảng đánh giá offline có thêm nhãn thật `y_any` để chấm điểm. Từ `p_any`, sắp xếp giảm dần để tạo danh sách top K; thứ hạng được suy ra khi chọn danh sách.

Ví dụ `p_any = 0,08` biểu diễn xác suất mua ước tính 8% cho khách–tháng đó, không phải khẳng định khách chắc chắn mua. Xác suất validation là xác suất gốc; bản cuối có calibrator đi kèm để chuyển raw margin thành xác suất sau hiệu chỉnh. Chưa triển khai API, tích hợp CRM hoặc luồng chấm điểm tự động cho dữ liệu tháng mới.

### 8.2. Artifact đã lưu trên máy

| Artifact | Vị trí | Ý nghĩa |
| --- | --- | --- |
| Feature 12 tháng, schema và protocol | `outputs/propensity_calendar_v2/` | Input có kiểm tra cho các bước mô hình |
| Kết quả chọn model | `outputs/propensity_baseline_geo_dialog_v1/selection.json` | Ghi biến thể được chọn và cấu hình |
| Dự đoán và metric validation | `validation_predictions.parquet`, `validation_metrics.json` trong thư mục run | Cho phép kiểm tra lại kết quả theo khách/tháng |
| So sánh với rule | `rule_validation/monthly_comparison.csv`, `.json`, `paired_comparison.json` trong thư mục run | Kết quả cùng ngân sách và khoảng bootstrap |
| Base model cuối | `outputs/propensity_baseline_geo_dialog_v1/final/model.cbm` | CatBoost đã fit trên 02–11/2022, dùng 153 predictor |
| Hiệu chỉnh và giải thích | `final/calibrator.json`, `calibration_predictions.parquet`, `feature_importance.csv` trong thư mục run | Bộ sigmoid, dự đoán tháng calibration và độ quan trọng feature |
| Trạng thái và thông tin tái lập | `outputs/propensity_baseline_geo_dialog_v1/manifest.json` | Trạng thái `fitted`, `test_opened=false`, cấu hình, môi trường và dấu kiểm tra artifact |
| Kiểm tra bản đã lưu | `outputs/baseline_geo_dialog_review_20261110/verification.json` | Kiểm tra hash, schema và dự đoán sau tải lại |

Ở bảng trên, “thư mục run” là `outputs/propensity_baseline_geo_dialog_v1/`. Artifact mô hình và dữ liệu trong `outputs/` được Git ignore; clone repository không tự có các file này. Mã nguồn, cấu hình và báo cáo mô tả cách tạo chúng.

## 9. Ý nghĩa đạt được và những phần chưa chứng minh

Baseline đã tạo được một **mốc so sánh thực nghiệm có thể tái lập**: dùng đủ ba nguồn, giữ lịch sử cũ, tuân thủ cutoff và so với rule trên cùng khách/ngân sách. Kết quả hiện ủng hộ việc giữ cả tháng gần nhất lẫn lịch sử tích lũy để phát triển mô hình tiếp theo.

Các giới hạn cần giữ rõ khi đọc kết quả:

- **Chưa có kết quả test 01/2023.** Mô hình cuối đã lưu nhưng khả năng tổng quát hóa và chất lượng sau calibration chưa được xác nhận trên tháng độc lập.
- **Chưa tách riêng đóng góp geo và dialog.** Cả hai biến thể đều có ba nguồn; phép so này đo việc thêm lịch sử cũ. Cần đối chứng bỏ từng nguồn mới kết luận nguồn nào tăng chất lượng bao nhiêu. Feature importance không thay thế phép kiểm tra đó.
- **Dialog mới dùng metadata.** Chưa đánh giá embedding hoặc nội dung hội thoại; geo chưa khai thác đầy đủ ý nghĩa không gian thực tế.
- **Chưa tìm kiếm mô hình rộng.** Chưa so các boosting khác, mô hình chuỗi, ensemble hoặc nhiều cấu hình siêu tham số.
- **Đánh giá là dự đoán mua trên nhóm khách hiện có.** Chưa chứng minh hiệu quả với khách hoàn toàn mới, giai đoạn khác hay tác động kinh doanh của việc liên hệ.
- **Lịch sử dài có độ phủ không đồng đều.** Khác biệt có thể phản ánh thời gian quan sát và hoạt động; feature rate/coverage giảm một phần vấn đề nhưng không loại bỏ hoàn toàn. Timezone và độ trễ nhãn cần xác nhận trước vận hành.

Mốc để các mô hình nâng cao vượt qua hiện là `multi_history`: **Recall@1% 8,70% trên validation**, đồng thời giữ cùng quần thể, cutoff, lịch chia và ngân sách. Chốt lựa chọn bằng validation trước khi đánh giá test; không dùng việc thử nhiều lần trên test để tiếp tục chọn mô hình.

## 10. Đọc mã và tái lập công việc

| Tài liệu/mã | Nội dung |
| --- | --- |
| [Notebook 05](notebooks/05_calendar_baseline.ipynb) | Luồng đọc dữ liệu, điều phối baseline và kiểm tra artifact |
| [Cấu hình cửa sổ](configs/propensity_calendar.json) | Nguồn cache và quy tắc tạo feature tháng lịch |
| [Cấu hình baseline](configs/propensity_baseline.json) | Hai biến thể, tham số CatBoost, seed và bootstrap |
| [Package baseline](src/propensity_baseline/) | Xử lý dữ liệu, metric, huấn luyện, lưu model và CLI |
| [Đặc tả và lệnh chạy](docs/propensity_baseline_design.md) | Chi tiết protocol và trình tự từng giai đoạn |
| [Script xuất bảng so sánh](scripts/report_baseline_rule.py) | Tạo báo cáo số liệu ngắn từ artifact; bổ sung cho báo cáo giải thích này |

Trình tự xử lý là **calendar → tune → compare-rule → fit/calibration → test → compare-rule trên test**. Trong run được báo cáo, các bước đã hoàn thành đến fit/calibration; chưa chạy test. Notebook nguồn giữ output trống, còn kết quả thật được tạo qua module/CLI và lưu riêng trong `outputs/`.

Với run đã hoàn thành, đọc lại artifact để kiểm tra kết quả; không ghi đè hoặc fit lại chỉ để xem báo cáo. Khi thay cấu hình/feature/code ảnh hưởng mô hình, cần dùng run mới và tái thực hiện các bước liên quan. Các kiểm tra cấu hình, schema, nguồn và hash model/calibrator giúp phát hiện việc trộn artifact từ những lần chạy khác nhau.

Số liệu trong báo cáo được đối chiếu với `selection.json`, `validation_metrics.json`, các bảng trong `rule_validation/`, manifest `fitted` và file kiểm tra model đã lưu của run nêu trên.
