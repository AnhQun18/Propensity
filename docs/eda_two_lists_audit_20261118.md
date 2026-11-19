# Đối soát hai danh sách EDA trong hội thoại — 18/11/2026

## Phạm vi và cách đánh giá

Đối chiếu 20 mục EDA kỹ thuật và 10 câu hỏi kinh doanh với file hiện còn trong workspace: báo cáo, bảng kết quả, validation, cấu hình và mã sinh kết quả. Không chạy lại raw, không fit mô hình và không xác nhận lại toàn bộ số liệu trong lần đối soát này. Validation được dẫn là kết quả đã lưu của lần chạy tương ứng.

- **Đã có:** bằng chứng đủ cho nội dung cốt lõi của mục; không đồng nghĩa mọi ý mở rộng đã xong.
- **Một phần:** có kết quả cho một số nguồn, kỳ hoặc nội dung; còn thiếu phần quan trọng.
- **Chưa thấy:** chưa tìm được đầu ra trực tiếp đủ chứng minh trong phạm vi đã kiểm tra; không khẳng định chưa từng làm trong lịch sử.
- **Cần tính lại:** kết quả phụ thuộc cutoff/lịch dùng UTC hoặc cửa sổ cũ, chưa đáp ứng quy ước hiện hành.

Quy ước áp dụng: AGENTS.md yêu cầu toàn bộ lịch sử trước cutoff, Moscow, so sánh H=1/2/3 trên cùng cohort đủ nhãn và cùng top-K nếu đánh giá xếp hạng. Phân phối amount toàn kỳ, schema, null và tổng số sự kiện thường không đổi chỉ vì đổi múi giờ; bảng theo ngày/tháng, recency, nhóm hoạt động và liên hệ với nhãn có thể đổi.

## Kết luận quan trọng

1. Nền tảng kiểm kê và chất lượng dữ liệu đã khá đầy đủ; đã có nhiều phân tích khách hàng và nhãn, không phải bắt đầu EDA lại từ đầu.
2. `docs/report/brief_audit.csv` đánh dấu 45 yêu cầu của brief cũ đã có artifact (GEO encoding là audited_unverified). Đây không phải xác nhận hoàn thành 30 mục trong hội thoại hoặc tuân thủ Moscow. `docs/report/methods/EDA_METHODS.md` vẫn ghi UTC và cửa sổ chính T−1.
3. GEO đã có nghiên cứu mới tại `outputs/geo_spatial_purchase_moscow_20261117/`: lịch sử đầy đủ, cutoff Moscow 01/11/2022, cùng cohort cho H=1/2/3, phân tầng TRX và kiểm định. Chỉ một cutoff; không có đánh giá top-K hoặc mô hình.
4. `outputs/geo_trend_purchase/` có H=1/2/3 trên 10 cutoff chung, nhưng dùng đặc trưng 7/30 ngày từ cache cũ; chưa phải bộ phân tích toàn lịch sử theo Moscow.
5. Workflow baseline đã xóa trong đợt dọn ngày 16/11/2026. Những metric baseline còn được nhắc trong văn bản cũ không được tính là thí nghiệm hiện còn đủ artifact để kiểm chứng.
6. Khoảng thiếu lớn nhất cho báo cáo kinh doanh: phân rã đóng góp tăng/giảm theo phân khúc, hành vi thay đổi trước mua trên cả ba nguồn, hành trình bán chéo, thẻ phân khúc thống nhất và ưu tiên khách được kiểm chứng ngoài thời gian.

## A. Đối soát 20 mục EDA kỹ thuật

| ID | Mục trong hội thoại | Trạng thái | Bằng chứng hiện có | Còn thiếu / việc tiếp theo |
|---|---|---|---|---|
| A01 | Hiểu cấu trúc và ý nghĩa | Đã có, còn giới hạn nghiệp vụ | `DATASETS.md`; `docs/dict/mbd_mini.md`; `docs/mbd_mini_eda/data_catalog.md`; `docs/report/tables/quality/raw_schema.csv` | Bổ sung sơ đồ quan hệ nếu cần trình bày; tên sản phẩm, đơn vị amount, encoding và công thức trường phụ trợ vẫn chưa xác minh đầy đủ. Cập nhật tài liệu còn ghi UTC. |
| A02 | Kiểm kê toàn bộ dữ liệu | Đã có | `docs/mbd_inventory/assessment.md`; `docs/mbd_inventory/summary.json`; `docs/mbd_mini_eda/manifest.json`; các bảng overview | Các kết quả cũ là snapshot; chưa quét mới toàn bộ file trong lần đối soát này. |
| A03 | Khóa và phép nối | Đã có phần cốt lõi | `docs/mbd_inventory/assessment.md`; `docs/propensity_feasibility/join_questions_audit.md`; `docs/report/checks/validation.json`; `docs/report/trx_geo_linkage_moscow.md` | Duy trì kiểm tra cardinality cho mọi bảng mới; join TRX–GEO chỉ là liên kết xấp xỉ theo thời gian. |
| A04 | Dữ liệu thiếu | Đã có phần cốt lõi | `docs/report/tables/quality/raw_missing.csv`; `docs/mbd_temporal_coverage/full_history_missingness.md`; `outputs/trx_null_audit_20261117/summary.json`; báo cáo EDA | Null toàn kỳ có thể tái sử dụng; phân rã thiếu/không log theo lịch và cutoff cần Moscow. Nguyên nhân thu thập chưa xác minh cho từng khách. |
| A05 | Trùng và giá trị không hợp lệ | Đã có phần cốt lõi | `docs/report/tables/quality/raw_duplicates.csv`, `invalid_values.csv`; `outputs/trx_zero_negative_20261116/summary.json`; `outputs/trx_outliers_20261116/RESULTS.md`; `outputs/duplicate_pair_review/` | Chưa bao phủ đầy đủ near-duplicate lệch vài giây/phút hoặc sau làm tròn; cờ nghi vấn không phải lỗi đã xác nhận. |
| A06 | Detail–PTLS | Một phần, đã xác định rủi ro chính | `docs/deliverables/mbd-mini-dataset-eda/tables/sequence_reconciliation.csv`; `outputs/ptls_detail_value_audit/` | Đã có chênh lệch số sự kiện và ví dụ mã khác nhau giữa hai biểu diễn. Ví dụ đối chiếu giá trị chỉ một khách/bốn sự kiện, chưa phải mapping nội dung toàn bộ sự kiện; cần kiểm chứng mapping mã và thứ tự/đơn vị trên toàn bộ phạm vi nếu muốn thay thế detail. |
| A07 | Độ phủ khách và nguồn | Đã có bản cũ; cần tính lại phần thời gian | `docs/report/tables/cross_modal/`; `outputs/behavior_coverage/`; `outputs/source_presence_target/`; bảng modality coverage trong báo cáo EDA | Có toàn kỳ và theo tháng, tám tổ hợp nguồn. Cần ma trận và liên hệ nhãn trên lịch sử tích lũy trước cutoff Moscow cho cả ba nguồn. |
| A08 | Thời gian và nhịp hoạt động | Một phần; phần lớn lịch cũ | `outputs/trx_monthly_audit/`; `outputs/dialog_temporal_audit/`; `outputs/geo_calendar_patterns/`; `docs/report/tables/trx_geo_linkage/monthly_coverage.csv` | TRX–GEO linkage đã có tháng Moscow; chưa đồng bộ các biểu đồ giờ/ngày/thứ/tháng của cả ba nguồn. Dialog có nhiễu ngày theo tài liệu audit, cần giữ giới hạn diễn giải lịch thực. |
| A09 | Hành vi cấp khách | Đã có nền tảng; một phần | `docs/mbd_mini_eda/data_catalog.md`; `docs/report/tables/*/*customer_activity*.csv`; `outputs/trx_outliers_20261116/RESULTS.md` | Có phân vị, khách 0 log, lịch sử và nhóm hoạt động cao. Chưa thấy bộ đóng góp top 1/5/10% đồng nhất cho ba nguồn và bảng khách–tháng Moscow đầy đủ. Gap TRX trong report cũ không nối qua tháng. |
| A10 | Amount theo currency | Đã có mô tả toàn kỳ; liên hệ nhãn cần làm lại | `docs/deliverables/eda/tables/amount_by_currency.csv`; `amount_outliers_by_currency.csv`; `outputs/trx_outliers_20261116/RESULTS.md`; `outputs/amount11_target_quantiles_20261116/RESULTS.md` | Phân tích nhãn amount hiện dùng UTC, 30/84 ngày, H=1; cần toàn lịch sử Moscow và H=1/2/3. Chưa quy đổi amount sang doanh thu/giá trị khách hàng. |
| A11 | Các mã giao dịch | Một phần | `docs/report/tables/trx/trx_category_summary.csv`, `trx_category_target_lift.csv`; các bảng category trong deliverables EDA | Có phân bố mã và lift top 5 event_type theo T−1. Chưa có đầy đủ bảng chéo type–subtype, nguồn–đích, mã–currency, drift mã và hồ sơ cơ cấu tích lũy theo khách dưới quy ước mới. |
| A12 | GEO | Một phần sâu; có nhánh tuân thủ mới | `docs/report/tables/geo/`; `outputs/geo_interarrival/`; `outputs/geo_frequency_deciles/`; `outputs/geo_spatial_purchase_moscow_20261117/REPORT.md` | Đã có đa dạng, tập trung, chuyển mã, quan hệ phân cấp, tỷ lệ mua theo ô và kiểm định. Nhánh mới chỉ một cutoff; cần lặp nhiều kỳ. Chưa có mapping tọa độ xác minh để tính khoảng cách/bản đồ. |
| A13 | Dialog và embedding | Một phần | `docs/report/tables/dialog/`; `outputs/dialog_charts/embedding_norm_summary.json`; `semantic_exploration_config.json`; `semantic_cluster_purchase_rates.csv`; `outputs/dialog_temporal_audit/` | Đã có chất lượng, PCA, KMeans và tỷ lệ mua theo cụm; không phải chưa phân cụm. Phân cụm khách dùng lịch sử tháng 10 trước cutoff 01/11 và một kỳ; chưa toàn lịch sử Moscow, kiểm chứng ổn định cụm và drift nhiều kỳ. Chưa thấy UMAP, nhưng đây là lựa chọn tùy mục tiêu. |
| A14 | Bốn nhãn và y_any | Đã có cốt lõi; còn mở rộng | `docs/report/tables/targets/`; `docs/deliverables/mbd-mini-dataset-eda/tables/target_frequency.csv`, `target_null_patterns.csv` | Có tỷ lệ, mất cân bằng, theo tháng, tần suất khách và nhãn gộp. Chưa thấy bằng chứng trực tiếp đủ cho toàn bộ tổ hợp nhãn dương, ma trận đồng mua và phân rã theo fold như yêu cầu mở rộng. Nhãn thuần theo mon không cần dịch tháng khi đổi Moscow. |
| A15 | Lịch sử/chuyển tiếp nhãn | Một phần | `docs/report/tables/targets/target_transition_matrix.csv`, `target_transition_rates.csv`, `buyer_composition_by_month.csv`; `outputs/repeat_purchase_review/` | Có chuyển tiếp cùng nhãn và returning/first-observed/unknown. Chưa thấy đầy đủ chuỗi dương liên tiếp, gap tháng mua và chuyển tiếp chéo sản phẩm A→B. Cần giả định nhãn sẵn có trước cutoff. |
| A16 | Hành vi–nhãn | Một phần; phần lớn cần tính lại | `docs/report/tables/trx/trx_rfm_lift.csv`; `docs/report/tables/dialog/dialog_metadata_target.csv`; `docs/report/tables/geo/geo_target_comparison.csv`; `docs/deliverables/eda/tables/feature_target_associations.csv`; GEO Moscow mới | Có nhóm phân vị, tỷ lệ, lift, CI và Spearman/Pearson. Các kết quả cũ không đồng nhất Moscow/toàn lịch sử/H=1/2/3. Nhánh mới chủ yếu GEO, một cutoff, chưa kiểm soát đầy đủ yếu tố nhiễu. |
| A17 | Tương tác nguồn và phân khúc | Một phần | `outputs/source_presence_target/`; `docs/report/trx_geo_linkage_moscow.md`; GEO Moscow phân tầng TRX; KMeans amount và dialog | Chưa có phân khúc đa nguồn thống nhất, so cùng độ dài lịch sử và lịch sử mua; chưa kiểm chứng giá trị dự báo bổ sung ngoài thời gian. Ghép gần thời gian không chứng minh vị trí giao dịch. |
| A18 | Ngoại lai và độ nhạy | Một phần, điều tra khá sâu | `docs/propensity_feasibility/tail_review/assessment.md`; `docs/eda_log_iqr_correction_20261111.md`; `outputs/trx_outliers_20261116/RESULTS.md` | Có raw/log1p, theo khách/currency/type/subtype, ứng viên sai scale và nhóm hoạt động cao. Chưa thấy đối chứng đầy đủ mức thay đổi các kết luận kinh doanh khi dùng các phương án xử lý ngoại lai. |
| A19 | Drift, thiên lệch và leakage | Một phần | `docs/report/tables/temporal/`; `docs/propensity_feasibility/open_questions/answers.md`; sample validation; GEO Moscow; `configs/eda_analysis.json` | Có kiểm tra cutoff, biến động nhãn/độ phủ và giới hạn mẫu. Chưa có báo cáo drift đa nguồn toàn diện, phân rã thay đổi thành phần khách, xác nhận độ trễ nhãn và audit preprocessing ngoài thời gian dưới quy ước mới. Config EDA còn UTC. |
| A20 | Tổng hợp và tái lập | Đã có hệ thống; cần hợp nhất | `docs/report/README.md`; `docs/report/brief_audit.csv`; `docs/report/checks/validation.json`; các manifest và script | Có bảng/hình/method/validation. Chưa có một sổ insight kinh doanh thống nhất nối bằng chứng→kết luận→hành động; cần phân biệt artifact cũ với Moscow và sửa tham chiếu baseline đã xóa. |

## B. Đối soát 10 câu hỏi kinh doanh

| ID | Câu hỏi | Trạng thái | Đã có | Còn thiếu để lập báo cáo kinh doanh |
|---|---|---|---|---|
| B01 | Tệp khách có cấu trúc thế nào? | Một phần | Độ phủ tám tổ hợp nguồn; phân phối hoạt động/lịch sử; nhóm TRX hoạt động cao; cụm embedding | Bộ phân khúc thống nhất, quy mô và đóng góp người mua, cùng định nghĩa trước cutoff Moscow; chưa xem mọi cụm thăm dò là phân khúc kinh doanh đã xác nhận. |
| B02 | Kết quả mua tăng hay giảm? | Một phần | Tỷ lệ/số dương từng tháng, từng sản phẩm, y_any | Phân rã thay đổi số người mua theo phân khúc và tác động cơ cấu so với tỷ lệ trong nhóm; bảng đóng góp đồng mua không được đếm trùng. |
| B03 | Người mua khác người chưa ghi nhận mua thế nào? | Một phần | RFM, mã TRX, metadata dialog, GEO, amount và CI | Chạy nhất quán toàn lịch sử Moscow/H=1/2/3, so nhiều cutoff và xem xét nhiễu do lịch sử mua/độ dài lịch sử. |
| B04 | Người mua quay lại hay lần đầu được ghi nhận? | Đã có câu trả lời mô tả chính | Buyer composition, lịch sử từng mua, tỷ lệ mua lại theo tháng và chuyển tiếp nhãn | Ghép đặc điểm/đóng góp từng nhóm với H=1/2/3; giữ nhóm chưa biết lịch sử. Không dùng metric baseline còn nhắc trong file cũ làm bằng chứng mới. |
| B05 | Hoạt động nhiều có dễ mua hơn? | Một phần | Band RFM và amount; frequency GEO; GEO Moscow phân tầng TRX | Phân biệt số giao dịch và giá trị từng giao dịch; lập đồng thời tỷ lệ mua, quy mô và đóng góp người mua trên lịch sử đầy đủ Moscow, nhiều cutoff. |
| B06 | Thay đổi nào xuất hiện trước mua? | Một phần | GEO trend 7/30 ngày, H=1/2/3 trên cùng 10 cutoff; bảng buyer/nonbuyer và theo tháng | So với nền toàn lịch sử của chính khách; bổ sung TRX/dialog, tái hoạt động sau im lặng và thay đổi cơ cấu mã. Chạy lại theo Moscow; cửa sổ gần đây chỉ là chỉ báo bổ sung. |
| B07 | Kết hợp nguồn có bổ sung thông tin? | Một phần | Source-presence lift; GEO có/không sau phân tầng TRX; liên kết thời gian TRX–GEO | Kiểm soát độ dài lịch sử và lịch sử nhãn; thêm dialog trong cùng thiết kế, lặp nhiều kỳ. Chưa có bằng chứng cải thiện dự báo ngoài thời gian hiện còn artifact mô hình. |
| B08 | Cơ hội mua thêm sản phẩm? | Chưa thấy phân tích trực tiếp đủ | Có tỷ lệ từng sản phẩm và chuyển tiếp 0/1 cùng target làm nền | Cần đồng mua, hành trình A→B, tỷ lệ B tương lai theo A quá khứ và nền tham chiếu cùng kỳ; xem độ hiếm, độ trễ nhãn, CI và ổn định. Chưa biết sở hữu sản phẩm nên chỉ gọi hành trình nhãn, chưa khẳng định eligibility bán chéo. |
| B09 | Nhóm nào đáng ưu tiên? | Một phần | Nhiều bảng nhóm có n, người mua, rate, lift, CI; kết quả GEO nhiều horizon | Chưa có thẻ phân khúc đủ quy mô–đóng góp–ổn định–độ phủ, bảng ưu tiên ngoài thời gian và cùng ngân sách. Không chọn nhóm cao nhất trên chính nhãn dùng để khám phá rồi gọi là hiệu quả dự báo. |
| B10 | Cơ hội trong 1, 2 hay 3 tháng? | Một phần; đã thực hiện mô tả H=1/2/3 | GEO trend có 10 cutoff chung theo phương pháp cũ; GEO Moscow đầy đủ lịch sử tại cutoff 01/11/2022 | Chưa có so sánh toàn diện đa nguồn, nhiều cutoff Moscow với precision/recall/lift@K, cùng cohort và top-K. Chưa đủ bằng chứng chọn horizon cho mô hình. |

## C. Mẫu thẻ phân khúc trong câu trả lời thứ hai

Các trường n, tỷ lệ mua, lift và CI đã có rải rác trong nhiều bảng. Tỷ trọng tệp và tỷ trọng đóng góp người mua chưa được hợp nhất với lịch sử mua lại, xu hướng tháng và độ phủ trong một thẻ thống nhất. Mẫu câu insight trong hội thoại là mẫu trình bày, không phải kết quả đã tính. Chưa thấy sổ insight hoàn chỉnh cho mọi phân khúc.

## D. Thứ tự hoàn thiện đề xuất

1. Khóa danh mục artifact: giữ kết quả chất lượng toàn kỳ, đánh dấu bảng phụ thuộc UTC/T−1; xác minh các trang báo cáo còn dẫn workflow baseline đã xóa. Không đổi nhãn timezone để thay cho tính lại.
2. Tạo bảng khách–cutoff thống nhất cho cả ba nguồn: toàn lịch sử trước 00:00 Moscow, đủ nhãn H=1/2/3; dùng cùng cohort/cutoff có đủ ba tháng. Giữ khách thiếu nguồn và trạng thái unknown.
3. Hoàn thiện B01/B02/B04/B05: thẻ phân khúc, đóng góp người mua và phân rã tăng/giảm. Tái sử dụng bảng thuần nhãn theo mon khi đúng định nghĩa.
4. Hoàn thiện B03/B06/B07: so hành vi–nhãn, thay đổi so nền riêng của khách và đóng góp từng nguồn; lặp kết quả GEO Moscow trên nhiều cutoff, bổ sung TRX/dialog.
5. Làm mới B08: hành trình chéo sản phẩm và cơ hội mua thêm; báo đúng giới hạn nhãn mua so với sở hữu sản phẩm.
6. Hoàn thiện B09/B10: nếu cần ưu tiên khách, chốt quy tắc hoặc mô hình chỉ trên kỳ quá khứ, đánh giá cùng top-K ngoài thời gian. Có thể bắt đầu bằng quy tắc đơn giản; chưa cần xây mô hình chỉ để hoàn thành báo cáo mô tả.
7. Xuất sổ insight: mỗi insight có nhóm, quy mô, số mua, tỷ lệ, đóng góp, lift, CI, độ ổn định theo kỳ/H, nguồn và hành động thử nghiệm đề xuất.

## Các bằng chứng chính đã đọc

- `AGENTS.md`; `docs/dict/mbd_mini.md`; `docs/mbd_mini_eda_audit.md`; `docs/mbd_mini_eda/data_catalog.md`.
- `docs/report/README.md`; `docs/report/methods/EDA_METHODS.md`; `docs/report/brief_audit.csv`; `docs/report/checks/validation.json`; danh mục bảng trong `docs/report/tables/`.
- `configs/eda_analysis.json`; lịch sử thay đổi Git của đợt dọn 16/11/2026.
- `outputs/geo_spatial_purchase_moscow_20261117/REPORT.md`, `validation.json`; `docs/report/trx_geo_linkage_moscow.md`.
- `outputs/amount11_target_quantiles_20261116/RESULTS.md`; `outputs/trx_outliers_20261116/RESULTS.md`; `outputs/repeat_purchase_review/purchase_history_feature_decision.md`.
- `outputs/dialog_temporal_audit/README.md`; `outputs/dialog_charts/semantic_exploration_config.json`; `outputs/source_presence_target/method.json`; `outputs/ptls_detail_value_audit/matched_event_examples.json`.
- Mã: `src/scripts/analyze_geo_trend_purchase.py`, `analyze_geo_calendar_patterns.py`, `analyze_dataset_eda.py`, `complete_report_eda.py`; đối chiếu danh mục các output và script bổ sung.
