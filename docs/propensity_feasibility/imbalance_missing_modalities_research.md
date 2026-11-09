# Lựa chọn mô hình khi target rất hiếm và geo/dialog thưa

Ngày nghiên cứu: **08/11/2026**. Phạm vi: đọc các audit toàn bộ MBD-mini đã có trong dự án, kiểm tra mã định nghĩa cohort và nghiên cứu nguồn sơ cấp. **Chưa huấn luyện hoặc đo cải thiện mô hình trong công việc này.**

**Cập nhật phạm vi theo người dùng:** đầu ra hiện được đổi thành một xác suất mua ít nhất một trong bốn sản phẩm. Xem [thiết kế any-product hiện hành](any_product/design.md). Các khuyến nghị bốn mô hình và bảng ca dương từng sản phẩm dưới đây được giữ làm tài liệu nghiên cứu trước thay đổi, không còn là thiết kế đầu ra hiện hành.

**Khuyến nghị triển khai trước:** giữ toàn bộ khách hàng–tháng có nhãn hợp lệ; tạo đặc trưng nhiều cửa sổ và lịch sử dài theo từng nguồn; biểu diễn rõ nguồn vắng mặt, độ dài lịch sử và recency; huấn luyện **bốn CatBoost nhị phân** với log loss trên phân bố tự nhiên làm cấu hình gốc. Thử trọng số dương vừa phải như một đối chứng, chọn riêng từng target bằng kiểm định theo thời gian. Đánh giá xếp hạng tại ngân sách liên hệ thực tế và kiểm tra/hiệu chỉnh xác suất trên dữ liệu không lấy mẫu cân bằng.

Đây là lựa chọn ưu tiên xét dữ liệu, độ phức tạp và khả năng kiểm chứng hiện tại, **không phải kết luận đã chứng minh tối ưu bằng thực nghiệm**. GBDT khác hoặc mô hình chuỗi có thể thắng sau khi kiểm định đúng.

## 1. Chẩn đoán từ dữ liệu hiện tại

Mẫu số là 1.202.688 khách hàng–tháng, từ 02/2022 đến 01/2023; không phải số khách độc lập.

| Nhãn | Dòng dương toàn kỳ | Tỷ lệ dương | Khách từng dương | Dương tháng 01/2023 |
|---|---:|---:|---:|---:|
| target_1 | 5.362 | 0,4458% | 4.902 | 159 |
| target_2 | 577 | 0,0480% | 490 | 25 |
| target_3 | 4.530 | 0,3767% | 4.180 | 215 |
| target_4 | 3.060 | 0,2544% | 2.498 | 293 |

Nguồn: [audit propensity](assessment.md), [lịch sử nhãn theo khách](../mbd_mini_eda/tables/descriptive.targets/customer_positive_history.csv), [tháng cuối](last_month_purchase/assessment.md).

Mất cân bằng có hai mặt: tỷ lệ dương thấp và **số trường hợp dương thực sự ít**. Sao chép 577 dòng không tạo thêm 577 ca độc lập. Thêm lịch sử cho một dòng cũng không tạo thêm nhãn dương. Dùng các tháng nhãn quá khứ làm mẫu huấn luyện giúp tận dụng ca dương có thật, nhưng các tháng cùng khách vẫn phụ thuộc nhau.

Các cột chính của geo (`client_id`, `event_time`, ba mức geohash) và dialog (`client_id`, `event_time`, `embedding`) không có null trong bản ghi sự kiện đã quan sát. Toàn bộ 217.205 embedding được audit đúng 768 chiều, hữu hạn. Vì vậy cần phân biệt **null trong sự kiện** với **không có sự kiện để điền vào lưới ngày**. Nguồn: [bảng missing](../mbd_mini_eda/tables/review/missing_values.csv), [embedding](../mbd_mini_eda/tables/descriptive.modalities/embedding_quality.csv).

Tại cutoff 01/12/2022:

- Trên khoảng chung 354 ngày, chỉ 25.695/100.224 khách có sự kiện hợp lệ ở cả ba nguồn; lọc complete-case bỏ khoảng 74,36% khách.
- Trong 84 ngày, chỉ 11.130 khách đủ ba nguồn. Cohort này giữ **18/56 ca target_2 dương của tháng 12**, tức bỏ **67,86%** ca dương. Đây là số đối chiếu giữa `monthly_windows.csv` và `target_monthly.csv`; mã audit xác nhận cột đếm dương thuộc cohort đủ ba nguồn.
- Tỷ lệ khách không có log bất kỳ nguồn nào trên nửa số ngày tăng từ 42,23% ở 354 ngày lên 62,86% ở 699 ngày. Điều đó không chứng minh dữ liệu hỏng: phần thêm vào có thể trước khi nguồn hoặc khách xuất hiện.
- Mọi khách đều không có dialog trên hơn nửa số ngày trong khoảng chung. Yêu cầu hội thoại xuất hiện hàng ngày không phù hợp bản chất nguồn này.

Nguồn: [audit lịch sử đầy đủ](../mbd_temporal_coverage/full_history_missingness.md), [cửa sổ](../mbd_temporal_coverage/monthly_windows.csv), [nhãn theo tháng](../mbd_mini_eda/tables/review/target_monthly.csv).

Với cùng cutoff và các cửa sổ lồng nhau, mở rộng quá khứ không thể làm giảm số sự kiện hoặc biến một nguồn đang có sự kiện thành hoàn toàn vắng mặt. Nếu tỷ lệ ô trống tăng, cần kiểm tra mẫu số ngày, padding và cách join. Audit hiện tại không có cờ quan sát liên tục theo khách, nên chưa thể xác định chính xác MCAR/MAR/MNAR hoặc phân biệt mọi trường hợp không hoạt động với thất thoát thu thập.

## 2. Bằng chứng nghiên cứu và giới hạn áp dụng

| Nghiên cứu | Kết quả liên quan | Cách dùng trong bài toán này |
|---|---|---|
| [MBD, bản v2, 2025](https://arxiv.org/html/2409.17587v2), mục 3 và 4 | Dữ liệu vốn có sự kiện bất đồng bộ, thưa và thiếu modality; baseline có tổng hợp thống kê và fusion. | Không coi đủ ba nguồn ở mọi ngày là điều kiện bắt buộc. Benchmark gốc chia theo khách, khác kiểm thử tháng tương lai. |
| [Wang, ICML 2020](https://proceedings.mlr.press/v119/wang20a.html) | Trong logistic regression cho rare events, lượng thông tin gắn chặt với số ca dương; phân tích lợi ích lấy mẫu âm. | Ưu tiên giữ ca dương thật. Kết quả lý thuyết không bảo đảm cùng mức hiệu quả cho CatBoost. |
| [van den Goorbergh và cộng sự, JAMIA 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9382395/) | Với logistic/ridge, cân bằng bằng sampling/SMOTE có thể làm xác suất bị đánh giá quá cao mà không cải thiện phân biệt. | Giữ baseline không cân bằng lại; tách chất lượng xếp hạng khỏi chất lượng xác suất. Đây là nghiên cứu y tế, không phải MBD. |
| [Carriero và cộng sự, Statistics in Medicine 2025; preprint 2024](https://arxiv.org/abs/2404.19494) | Mô phỏng nhiều thuật toán ghi nhận hiệu chỉnh mất cân bằng có thể làm xấu calibration; recalibration không luôn khắc phục hết. | Không mặc định cân bằng 1:1 hay cho rằng calibration sẽ sửa mọi hậu quả. Không suy ra mọi class weight đều có hại. |
| [Charoenphakdee và cộng sự, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Charoenphakdee_On_Focal_Loss_for_Class-Posterior_Probability_Estimation_A_Theoretical_Perspective_CVPR_2021_paper.html) | Focal loss không strictly proper: điểm tối ưu không trực tiếp là xác suất hậu nghiệm thật. | Focal loss là ứng viên xếp hạng cho đối chứng, không phải mặc định cho propensity cần xác suất. |
| [Perez-Lebel và cộng sự, GigaScience 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9012100/) | Benchmark dữ liệu y tế cho thấy cây xử lý missing trực tiếp có hiệu quả và chi phí tốt; cờ missing hữu ích khi impute. | Ưu tiên NaN có nghĩa và cờ thiếu; chưa có lý do bắt đầu bằng MICE/KNN quy mô lớn. Không chuyển nguyên kết luận y tế thành bảo đảm cho ngân hàng. |
| [Che và cộng sự, GRU-D, Scientific Reports 2018](https://www.nature.com/articles/s41598-018-24271-9) | Dùng mask, khoảng thời gian và decay để khai thác missingness trong chuỗi đo lường. | Mượn ý tưởng mask/recency; không mặc định coi dialog là một phép đo liên tục cần nội suy. |
| [Neverova và cộng sự, ModDrop](https://arxiv.org/abs/1501.00102) | Bỏ ngẫu nhiên các kênh trong huấn luyện giúp mô hình đa phương thức chịu được thiếu kênh. | Đối chứng cho kiến trúc chuỗi ở giai đoạn sau. Thiếu nguồn thực tế có thể không ngẫu nhiên, nên vẫn cần kiểm tra các nhóm thiếu tự nhiên. |
| [Babaev và cộng sự, CoLES, SIGMOD 2022](https://arxiv.org/abs/2002.08232) | Học biểu diễn tự giám sát từ chuỗi sự kiện rời rạc, áp dụng cho giao dịch. | Hướng nâng cấp tận dụng nhiều sự kiện không cần nhãn mua, nếu baseline đặc trưng còn thiếu tín hiệu thứ tự. |
| [Burakov, EventGPT, preprint 02/11/2026](https://arxiv.org/html/2611.03302), bảng 9 và phụ lục G | Trên MBD-mini, engineered features + CatBoost đạt mean ROC-AUC 0,808; embedding đạt 0,787; ensemble đạt 0,824. Recall top 1% trung bình của ensemble và engineered features cùng 0,150. | Bằng chứng gần bài toán, ủng hộ baseline tổng hợp. Preprint, chưa tái lập; chia theo khách và dùng feature riêng, không thay cho kiểm định theo thời gian của dự án. |

Các lựa chọn cửa sổ, trọng số và kiến trúc dưới đây là **đề xuất thiết kế**, không phải siêu tham số tối ưu lấy trực tiếp từ những bài trên.

## 3. Xử lý geo/dialog bằng biểu diễn đúng sự vắng mặt

Giữ một dòng cho mỗi khách–cutoff; tổng hợp riêng từng nguồn rồi **left join** vào khung khách có nhãn. Không inner join các nguồn và không loại khách chỉ vì hơn 50% ngày không có log.

| Tình trạng | Biểu diễn đề xuất |
|---|---|
| Có bản ghi nhưng một thuộc tính thiếu | Giữ bản ghi hữu ích; thuộc tính số NaN, categorical dùng token riêng; thêm số lượng/tỷ lệ thuộc tính thiếu nếu phù hợp. |
| Trong phạm vi lịch dữ liệu, không ghi nhận sự kiện | Count = 0, `has_event = 0`; thống kê cần sự kiện như mean, embedding và recency không tồn tại giữ NaN. 0 có nghĩa không có log trong phần trích xuất, không khẳng định không có hoạt động thực. |
| Cửa sổ nằm hoàn toàn trước phạm vi nguồn | `source_window_overlap_days = 0`, `history_truncated = 1`; thống kê hành vi để NaN, không giả lập đã quan sát toàn cửa sổ. |
| Cửa sổ chỉ chồng một phần phạm vi nguồn | Ghi số ngày chồng lấn và cờ truncated; count chỉ tính phần có trong trích xuất; không diễn giải phần còn lại như số 0 đã xác nhận. |
| Khách chưa từng có nguồn trước cutoff | `ever_seen_source = 0`, recency/first-seen-age NaN. Không tự gán ngày mở tài khoản bằng ngày đầu xuất hiện. |
| Chuỗi cần padding | Dùng padding mask riêng; padding không phải một sự kiện thật hoặc một ngày đã được quan sát. |

`source_window_overlap_days` chỉ mô tả phạm vi trích xuất theo lịch, **không phải số ngày hệ thống chắc chắn quan sát được khách**. Nếu dùng min timestamp làm xấp xỉ ngày bắt đầu nguồn, ghi rõ đây là metadata suy ra từ dữ liệu; không tính ngày kết thúc theo sự kiện tương lai của khách.

Với mỗi nguồn, thêm `has_event_30d/84d/180d`, `ever_seen_source`, `days_since_last_event`, `days_since_first_recorded_event`, `active_days`, `event_count`, cờ cửa sổ bị cắt. Tất cả tính từ dữ liệu trước cutoff. Không chỉ cung cấp một cờ missing chung cho mọi cơ chế.

Đề xuất feature ban đầu:

- **Trx:** số sự kiện/ngày hoạt động, recency, tần suất event type, thống kê amount theo currency; không cộng các currency khác nhau như cùng một đơn vị. Giữ category ẩn danh dưới dạng category.
- **Geo:** số ping/ngày hoạt động, số mã vị trí khác nhau, mức tập trung tại mã phổ biến, biến động phân bố mã. Geohash đã mã hóa không được xử lý như tọa độ số để suy ra khoảng cách địa lý.
- **Dialog:** count, ngày hoạt động, recency; so sánh riêng metadata này với metadata cộng embedding gần nhất và mean embedding. Có thể thử PCA 32/64 chiều sau pooling để hạn chế số feature; PCA chỉ fit trên train và bỏ các hàng không có embedding khi fit. Số chiều phải được chọn bằng validation.
- **Lịch sử:** cấu hình gốc 30/84 ngày + tóm tắt toàn bộ quá khứ theo nguồn; thử thêm 180/365 ngày khi có lợi. Không bắt ba nguồn có cùng ngày bắt đầu. So sánh với 84 ngày đơn để đo phần lợi ích của lịch sử dài.

Đối với CatBoost, NaN số được hỗ trợ trực tiếp; categorical thiếu cần một chuỗi dành riêng như `__MISSING__`, tránh trùng category thật. Cơ chế này được mô tả trong [tài liệu CatBoost](https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing). Nó không tự suy ra nguyên nhân thiếu, nên các cờ ngữ nghĩa vẫn hữu ích.

Không forward-fill vị trí như thể khách luôn đứng tại vị trí cũ; có thể giữ **vị trí cuối đã quan sát kèm tuổi của thông tin đó**. Không phát sinh embedding hội thoại giả cho ngày không có cuộc hội thoại.

## 4. Xử lý target hiếm

Huấn luyện bốn bài toán binary riêng: khách có thể dương ở nhiều sản phẩm. Tổng bốn xác suất không cần bằng 1. Không dùng softmax bốn lớp hoặc gộp `any_product` làm đầu ra thay thế yêu cầu dự đoán từng sản phẩm.

**Cấu hình gốc:** giữ toàn bộ dòng train hợp lệ, binary log loss không tăng trọng số theo lớp, regularization và độ sâu vừa phải, early stopping theo tập quá khứ phù hợp. Đối chiếu ridge logistic đơn giản để phát hiện CatBoost có đang overfit. Không dùng accuracy hoặc ngưỡng 0,5 để kết luận mô hình không học được ca dương.

**Đối chứng class weight:** thử `w_positive` trong `{1, 5, 20, sqrt(N_negative/N_positive)}`, tính tỷ số trên train của từng split và từng nhãn. Đây là lưới nhỏ để khởi đầu, không phải quy tắc lý thuyết. Không tự đặt khoảng 2.083 chỉ vì đó là tỷ lệ âm/dương của target_2 toàn kỳ. Nếu trọng số không cải thiện ổn định Recall@K/AP, giữ cấu hình gốc. Đánh giá không dùng class weight trên validation/test.

**Nếu cần giảm thời gian train:** với từng target, giữ mọi dòng dương và lấy mẫu ngẫu nhiên âm có xác suất lưu `q` được ghi lại. Dùng inverse-probability weights: dương 1, âm `1/q`, để xấp xỉ hàm mất mát trên phân bố gốc. Nếu lấy mẫu phân tầng, dùng xác suất đúng từng tầng. Không coi trọng số này là trọng số tăng tầm quan trọng của lớp dương.

Một phương án khác cho lấy mẫu âm đồng nhất là huấn luyện không IPW rồi sửa odds, với điều kiện mô hình đã ước lượng tốt xác suất trong mẫu: `logit(p_population) = logit(p_sample) + log(q)` khi giữ toàn bộ dương. Đây không thay thế kiểm tra calibration. **Không vừa IPW vừa sửa lại cùng prior một lần nữa.** Không áp công thức đơn này cho hard-negative sampling phụ thuộc feature.

Không chọn SMOTE/ADASYN làm mặc định: nội suy các count, mã category và biểu diễn của khách thiếu nguồn có thể tạo tổ hợp không hợp lý; không tạo thêm thông tin dương độc lập. Focal loss chỉ nên là đối chứng sau, với kiểm tra calibration riêng. Không cần đổi target thành anomaly detection: “hiếm mua sản phẩm” chưa đồng nghĩa “hành vi bất thường”.

Với target_2, tăng dữ liệu dương thật là hướng quan trọng hơn tăng độ phức tạp. Audit [MBD full và mini](../mbd_comparison/assessment.md) ghi nhận full có **5.986** dòng dương, so với **577** của mini; đây là khoảng 10,37 lần số dòng, không bảo đảm 10 lần hiệu quả. Nếu mở rộng, giữ test ID/thời gian cố định và loại ID test khỏi train khi đánh giá khách chưa thấy. Không dùng `is_balanced` được chọn theo nhãn tương lai làm cohort cho kiểm thử thời gian.

## 5. Protocol xác định lựa chọn thắng

Mục tiêu giả định: chấm điểm cho khách hiện hữu vào đầu tháng để chọn danh sách tiếp cận trong tháng tới. Mọi feature lấy `event_time < cutoff`; nhãn thuộc tháng chứa `mon`, theo [nghiên cứu mapping hiện tại](mon_research.md). Không dùng `trans_count` hoặc `diff_trans_date` từ targets cho tới khi xác minh thời điểm tạo. Nhãn train phải đã sẵn có tại thời điểm dự đoán; nếu có độ trễ công bố phải thêm khoảng trễ tương ứng.

| Bước | Dữ liệu |
|---|---|
| Chọn cấu hình | Train 02–08/2022, validate 09/2022; train 02–09, validate 10; train 02–10, validate 11. |
| Fit mô hình cuối | Cố định feature/hyperparameters/số vòng học từ các split; fit trên 02–11/2022. |
| Hiệu chỉnh xác suất | Dự đoán 12/2022 bằng mô hình trên; fit bộ hiệu chỉnh đơn giản trên phân bố tự nhiên. |
| Test khóa | Dự đoán 01/2023 bằng cùng mô hình và calibrator; không chọn cấu hình lại từ kết quả này. |

Không refit thêm nhãn tháng 12 vào mô hình cơ sở rồi mặc nhiên tái dùng bộ hiệu chỉnh đã fit cho mô hình cũ. Nếu cần refit, phải thiết kế lại calibration bằng dự đoán ngoài thời gian phù hợp.

Đối với target_2, train cuối có 496 dòng dương, calibration tháng 12 có 56, test tháng 1 có 25. Vì vậy ưu tiên hiệu chỉnh ít tham số, có regularization (chẳng hạn điều chỉnh intercept), so với isotonic nhiều bậc tự do. Chọn quy tắc calibration trên các split quá khứ hoặc đăng ký trước; tháng 12 chỉ fit quy tắc đã chọn. Không giả định calibration tháng 12 sẽ sửa được drift tháng 1.

Khách trùng giữa train và test theo thời gian phù hợp kịch bản khách hiện hữu; không đưa `client_id` vào làm predictor. Nếu câu hỏi là khách chưa từng thấy, bổ sung split giữ riêng ID, đồng thời giữ thứ tự thời gian. Không trộn hai câu hỏi đánh giá hoặc áp mọi tiêu chí lọc cùng lúc khiến target_2 mất gần hết mẫu.

Bộ chỉ số cần báo cáo:

- **Recall@K, Precision@K, Lift@K** theo từng tháng và từng target. Khi chưa có ngân sách nghiệp vụ, báo cáo top 0,5%, 1%, 5%; chọn trước top 1% làm tiêu chí chính tạm thời. Với cùng K và cohort, precision và recall cùng tăng khi số ca dương bắt được tăng.
- **Average Precision (AP)** từng target, ghi rõ cách tính; không đồng nhất mọi cách tích phân PR-AUC. Báo cáo prevalence vì AP phụ thuộc tỷ lệ dương. [Saito và Rehmsmeier, 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4349800/) giải thích giá trị của PR trong bối cảnh này.
- **ROC-AUC** vẫn giữ để đo khả năng phân biệt rộng và đối chiếu benchmark; không tuyên bố ROC-AUC sai chỉ vì imbalance. Bằng chứng [McDermott và cộng sự, 2024](https://arxiv.org/abs/2401.06091) cũng phản biện việc luôn ưu tiên AUPRC chỉ dựa trên mất cân bằng.
- Log loss, Brier so với baseline prevalence học từ quá khứ; tổng số ca dự đoán so với ca thực, calibration theo nhóm điểm đủ lớn. Không chỉ nhìn Brier/ECE nhỏ vì đại đa số là âm.
- Khoảng tin cậy và so sánh mô hình **paired** trên cùng khách. Bootstrap theo `client_id` khi gộp nhiều tháng để giữ phụ thuộc trong khách; báo cáo dao động riêng theo tháng vì bootstrap khách không bao quát hết bất định drift thời gian.
- Kết quả theo các tổ hợp nguồn có/vắng, lịch sử ngắn/dài. Nhóm không có ca dương: ghi AP/ROC-AUC không xác định thay vì gán 0 hoặc 0,5 tùy tiện.

Với test target_2 chỉ có 25 ca dương, bắt thêm hoặc bỏ sót một ca làm Recall thay đổi **4 điểm phần trăm**. Không thể khẳng định mô hình tốt hơn chắc chắn chỉ từ chênh lệch vài ca ở tháng cuối. Kiểm định cần dựa cả các cửa sổ quá khứ, độ ổn định và khoảng tin cậy; test tháng cuối vẫn chỉ mở sau khi khóa lựa chọn.

## 6. Thực nghiệm tối thiểu có ý nghĩa

Giữ nguyên cohort, split, nhãn và ngân sách tuning cho các đối chứng. Thực hiện từng bước để biết cải thiện đến từ đâu, không chạy toàn bộ tích Descartes siêu tham số.

| Mã | Cấu hình | Câu hỏi |
|---|---|---|
| A | Trx 84 ngày, bốn CatBoost, log loss tự nhiên | Baseline giao dịch có tín hiệu không? |
| B | Trx 30/84 ngày + tóm tắt toàn quá khứ, cờ độ dài | Lịch sử dài có thêm giá trị không? |
| C | B + metadata có/vắng/recency/count của geo/dialog | Tín hiệu đến từ sự xuất hiện nguồn hay nội dung? |
| D | C + feature nội dung geo | Geo có lợi ngoài chỉ báo hoạt động không? |
| E | D + dialog pooling/PCA | Nội dung dialog có đóng góp thêm không? |
| F | Cấu hình tốt nhất A–E + lưới class weight nhỏ | Tăng trọng số dương có giúp xếp hạng ổn định không? |
| G | Biểu diễn chuỗi tự giám sát + fusion có mask | Thứ tự sự kiện có tạo lợi ích đủ lớn để trả chi phí không? |

G là bước nâng cấp khi A–F đã ổn định: học encoder trên các sự kiện quá khứ hợp lệ; mã hóa riêng từng nguồn, thêm availability/recency, fusion với mask hoặc gate, rồi bốn sigmoid heads. Thử bỏ nguyên modality trong train theo tinh thần ModDrop; kiểm tra riêng cả thiếu tự nhiên và mất kênh giả lập. Nguồn vắng hoàn toàn có biểu diễn missing riêng; khi mọi nguồn vắng phải có đường dự đoán fallback. Không yêu cầu complete-case để train encoder từng nguồn.

Không cho pretraining/PCA nhìn sự kiện sau cutoff đánh giá. Với benchmark khách mới, cũng loại ID test khỏi các bước fit biểu diễn. Không chia sớm thành nhiều chuyên gia theo tổ hợp nguồn khi số dương từng nhánh quá ít.

Chọn cấu hình có Recall@K tốt và ổn định trên các tháng validation, AP không suy giảm đáng kể, xác suất đủ tin cậy cho mục tiêu sử dụng, và chi phí phù hợp. Khi khác biệt nhỏ trong phạm vi bất định, chọn cấu hình đơn giản hơn. Có thể giữ geo/dialog cho target có lợi và bỏ cho target không có lợi; không ép bốn sản phẩm dùng cùng một bộ feature.

## 7. Quyết định cho dự án

**Bắt đầu bằng bốn CatBoost trên feature nhiều cửa sổ và lịch sử dài theo nguồn, giữ khách thiếu modality, dùng NaN có nghĩa + masks + recency.** Giữ phân bố tự nhiên làm baseline; kiểm tra trọng số vừa phải thay vì mặc định SMOTE hoặc balanced 1:1. Xếp hạng theo ngân sách liên hệ và hiệu chỉnh xác suất là hai bước riêng.

Nếu target_2 không ổn định, ưu tiên mở rộng ca dương thật từ MBD full trên một protocol test cố định. Nếu mô hình tổng hợp đã ổn định nhưng thiếu tín hiệu thứ tự, mới thử CoLES/encoder chuỗi và fusion có mask. Chưa có kết quả thực nghiệm trong repo đủ để gọi một loss, cửa sổ hoặc mô hình là tối ưu tuyệt đối.
