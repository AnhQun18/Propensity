# Danh mục bảng kết quả và các biến

Mỗi CSV là một bảng tổng hợp. `n`, `rows`, `population` là mẫu số tương ứng; `rate` là tỷ lệ 0–1; `percent` là 0–100. `null` chỉ giá trị chưa quan sát/không xác định, không thay bằng 0. Phân vị dùng nội suy tuyến tính; IQR = P75 − P25. Các cột `scope` ghi phạm vi.

## [customer_activity](tables/descriptive.customers/customer_activity.csv)

Nội dung: Phân phối các tổng hợp khách hàng toàn kỳ, bao gồm cả khách không có log.

20 dòng. Các biến: `feature`, `rows`, `finite_n`, `missing_n`, `infinite_n`, `mean`, `std`, `min`, `p25`, `median`, `p75`, `p95`, `p99`, `max`.

## [customer_aggregation_checks](tables/descriptive.customers/customer_aggregation_checks.csv)

Nội dung: Đối chiếu số khách đầu vào/đầu ra, khóa trùng và khách 0 log.

1 dòng. Các biến: `input_customers`, `output_customers`, `duplicate_keys`, `customers_without_log`.

## [first_last_event_coverage](tables/descriptive.customers/first_last_event_coverage.csv)

Nội dung: Số khách có/không có mốc log đầu-cuối hợp lệ.

3 dòng. Các biến: `source`, `reference_customers`, `with_log`, `without_log`, `scope`.

## [first_last_event_month](tables/descriptive.customers/first_last_event_month.csv)

Nội dung: Phân phối tháng log đầu hoặc cuối trên khách có log.

121 dòng. Các biến: `source`, `milestone`, `month`, `customers`, `logged_customers`, `reference_customers`, `rate_among_logged`, `scope`.

## [modality_coverage](tables/descriptive.customers/modality_coverage.csv)

Nội dung: 8 tổ hợp nguồn có log trên toàn lịch sử.

8 dòng. Các biến: `trx_geo_dialog`, `customers`, `population`, `rate`.

## [source_count_distribution](tables/descriptive.customers/source_count_distribution.csv)

Nội dung: Phân phối số nguồn có log (0, 1, 2, 3) trên mỗi khách.

4 dòng. Các biến: `source_count`, `customers`, `population`, `rate`.

## [dialog_summary](tables/descriptive.modalities/dialog_summary.csv)

Nội dung: Phân phối số hội thoại và ngày có hội thoại mỗi khách.

2 dòng. Các biến: `metric`, `rows`, `finite_n`, `missing_n`, `infinite_n`, `mean`, `std`, `min`, `p25`, `median`, `p75`, `p95`, `p99`, `max`.

## [embedding_quality](tables/descriptive.modalities/embedding_quality.csv)

Nội dung: Kiểm tra chiều, phần tử và norm embedding, không giải mã nội dung hội thoại.

6 dòng. Các biến: `check`, `n`, `value`.

## [geo_summary](tables/descriptive.modalities/geo_summary.csv)

Nội dung: Phân phối số log, ngày có log và số mã vị trí khác nhau mỗi khách.

3 dòng. Các biến: `metric`, `rows`, `finite_n`, `missing_n`, `infinite_n`, `mean`, `std`, `min`, `p25`, `median`, `p75`, `p95`, `p99`, `max`.

## [location_categories](tables/descriptive.modalities/location_categories.csv)

Nội dung: Top mã vị trí theo từng cấp, kèm missing và remaining.

73 dòng. Các biến: `source`, `column`, `kind`, `value`, `n`, `distinct_n`, `rows`, `rate`, `frequency_scope`.

## [location_categories_full](tables/descriptive.modalities/location_categories_full.csv)

Nội dung: Đầy đủ tần suất các mã vị trí theo từng cột.

752,805 dòng. Các biến: `source`, `column`, `kind`, `value`, `n`, `distinct_n`, `rows`, `rate`, `frequency_scope`.

## [duplicates](tables/descriptive.quality/duplicates.csv)

Nội dung: Trùng khóa, trùng nguyên dòng và ứng viên trùng client–timestamp; số dòng dư sau bản đầu.

13 dòng. Các biến: `source`, `kind`, `duplicate_n`, `duplicate_groups`, `analyzed_rows`, `scope`.

## [invalid_values](tables/descriptive.quality/invalid_values.csv)

Nội dung: Số dòng vi phạm điều kiện ID, timestamp, fold, amount hoặc embedding.

11 dòng. Các biến: `source`, `check`, `n`.

## [key_coverage](tables/descriptive.quality/key_coverage.csv)

Nội dung: Khách có hoặc không có log ID/thời gian hợp lệ theo nguồn.

3 dòng. Các biến: `source`, `reference_customers`, `customers_with_valid_time_log`, `distinct_ids_with_valid_time`, `customers_without_valid_time_log`, `coverage`, `unknown_client_rows`, `missing_id_rows`.

## [missing_values](tables/descriptive.quality/missing_values.csv)

Nội dung: Null thuộc tính theo cột trên toàn bộ dữ liệu; không đếm ngày không có sự kiện.

61 dòng. Các biến: `source`, `column`, `rows`, `missing_n`, `missing_rate`.

## [column_roles](tables/descriptive.structure/column_roles.csv)

Nội dung: Kiểu dữ liệu và vai trò từng cột đầu vào detail.

36 dòng. Các biến: `source`, `column`, `dtype`, `role`.

## [relationship_summary](tables/descriptive.structure/relationship_summary.csv)

Nội dung: Khóa nối và quan hệ giữa bảng khách, nhãn, các bảng sự kiện.

4 dòng. Các biến: `left`, `right`, `keys`, `cardinality`.

## [table_inventory](tables/descriptive.structure/table_inventory.csv)

Nội dung: Quy mô và grain các bảng đầu vào detail.

5 dòng. Các biến: `source`, `files`, `rows`, `row_unit`, `logical_columns`, `schema_variants`.

## [customer_positive_history](tables/descriptive.targets/customer_positive_history.csv)

Nội dung: Khách từng có nhãn dương và số tháng dương mỗi khách.

4 dòng. Các biến: `target`, `customers`, `observed_customers`, `ever_positive_customers`, `rate`, `mean_positive_months`.

## [label_coverage](tables/descriptive.targets/label_coverage.csv)

Nội dung: Độ phủ nhãn theo tháng và fold.

60 dòng. Các biến: `mon`, `fold`, `customer_month_rows`, `customers`, `targets`, `observed_label_cells`, `missing_label_cells`.

## [positive_labels_per_row](tables/descriptive.targets/positive_labels_per_row.csv)

Nội dung: Phân phối số target dương đồng thời trên một dòng khách–tháng.

5 dòng. Các biến: `positive_labels`, `rows`.

## [target_by_month_fold](tables/descriptive.targets/target_by_month_fold.csv)

Nội dung: Tỷ lệ nhãn theo mon, fold; cộng tử và mẫu số trước khi gộp.

240 dòng. Các biến: `mon`, `fold`, `target`, `rows`, `customers`, `labelled_n`, `positive_n`, `missing_n`, `rate`.

## [target_cooccurrence](tables/descriptive.targets/target_cooccurrence.csv)

Nội dung: Số dòng hai target đồng thời dương và mẫu số cặp được quan sát.

6 dòng. Các biến: `left`, `right`, `observed_pairs`, `both_positive`.

## [target_prevalence](tables/descriptive.targets/target_prevalence.csv)

Nội dung: Số nhãn dương, âm, null và tỷ lệ trên các dòng có nhãn.

4 dòng. Các biến: `target`, `rows`, `labelled_n`, `positive_n`, `negative_n`, `missing_n`, `rate`.

## [history_length_summary](tables/descriptive.temporal/history_length_summary.csv)

Nội dung: Phân phối chiều dài lịch sử mỗi khách, xem scope và đơn vị từng hàng.

6 dòng. Các biến: `source`, `metric`, `rows`, `finite_n`, `missing_n`, `infinite_n`, `mean`, `std`, `min`, `p25`, `median`, `p75`, `p95`, `p99`, `max`.

## [monthly_activity](tables/descriptive.temporal/monthly_activity.csv)

Nội dung: Tổng log và số khách có log theo tháng và nguồn.

61 dòng. Các biến: `source`, `month`, `events`, `customers`, `observed_days`, `events_per_active_customer`.

## [source_time_ranges](tables/descriptive.temporal/source_time_ranges.csv)

Nội dung: Timestamp đầu/cuối đã ghi nhận mỗi nguồn.

3 dòng. Các biến: `source`, `first_event`, `last_event`, `scanned_rows`, `valid_time_rows`.

## [amount_by_currency](tables/descriptive.transactions/amount_by_currency.csv)

Nội dung: Phân vị amount theo từng mã currency, gồm nhóm currency null.

14 dòng. Các biến: `currency`, `currency_missing`, `rows`, `finite_n`, `mean`, `std`, `min`, `max`, `p25`, `median`, `p75`, `p95`, `p99`, `scope`.

## [amount_histogram](tables/descriptive.transactions/amount_histogram.csv)

Nội dung: Histogram amount trên giá trị gốc; độ rộng và scope trong bảng.

15 dòng. Các biến: `bin`, `n`, `left`, `right`, `scope`.

## [amount_summary](tables/descriptive.transactions/amount_summary.csv)

Nội dung: Thống kê moment và phân vị amount; scope và phương pháp được ghi trong bảng.

1 dòng. Các biến: `finite_n`, `mean`, `std`, `min`, `max`, `scanned_rows`, `sample_rows`, `sample_finite_n`, `missing_n`, `infinite_n`, `invalid_n`, `p25`, `median`, `p75`, `p95`, `p99`, `moments_scope`, `quantile_scope`, `whisker_low`, `whisker_high`, `quantile_method`.

## [transaction_categories](tables/descriptive.transactions/transaction_categories.csv)

Nội dung: Top mã giao dịch theo từng cột, kèm phần missing và remaining.

242 dòng. Các biến: `source`, `column`, `kind`, `value`, `n`, `distinct_n`, `rows`, `rate`, `frequency_scope`.

## [transaction_categories_full](tables/descriptive.transactions/transaction_categories_full.csv)

Nội dung: Đầy đủ tần suất mọi mã giao dịch theo cột, giữ cả missing.

12,829 dòng. Các biến: `source`, `column`, `kind`, `value`, `n`, `distinct_n`, `rows`, `rate`, `frequency_scope`.

## [transaction_outliers](tables/descriptive.transactions/transaction_outliers.csv)

Nội dung: Cờ Tukey pooled amount toàn bộ, không tự xác nhận lỗi nghiệp vụ.

1 dòng. Các biến: `lower`, `upper`, `outlier_n`, `analyzed_finite_n`, `scope`.

## [amount_by_currency](tables/review/amount_by_currency.csv)

Nội dung: Phân vị amount theo từng mã currency, gồm nhóm currency null.

14 dòng. Các biến: `currency`, `rows`, `finite_n`, `mean`, `min`, `max`, `p25`, `median`, `p75`, `p95`, `p99`.

## [amount_full](tables/review/amount_full.csv)

Nội dung: Moment và phân vị chính xác của mọi amount hữu hạn, chưa quy đổi currency.

1 dòng. Các biến: `finite_n`, `mean`, `std`, `min`, `max`, `p25`, `median`, `p75`, `p95`, `p99`.

## [amount_outliers_currency](tables/review/amount_outliers_currency.csv)

Nội dung: Cờ Tukey 1,5 IQR theo từng currency; outlier_n không phải số lỗi đã xác nhận.

14 dòng. Các biến: `currency`, `finite_n`, `lower_fence`, `upper_fence`, `outlier_n`, `nonpositive_n`, `max`.

## [amount_raw_bins](tables/review/amount_raw_bins.csv)

Nội dung: Đếm amount gốc theo bin trái đóng phải mở; biên ±inf bao phủ toàn đuôi.

15 dòng. Các biến: `bin`, `n`, `left`, `right`.

## [amount_zoom](tables/review/amount_zoom.csv)

Nội dung: Histogram 30 bin tuyến tính trong [0,P99]; phần ngoài zoom vẫn nằm trong bảng đầy đủ.

30 dòng. Các biến: `bin`, `n`, `left`, `right`.

## [behavior_combinations](tables/review/behavior_combinations.csv)

Nội dung: 8 tổ hợp hiện diện trx/geo/dialog toàn kỳ; mỗi khách thuộc đúng một tổ hợp.

8 dòng. Các biến: `trx`, `geo`, `dialog`, `customers`, `population`, `percent`.

## [behavior_rates](tables/review/behavior_rates.csv)

Nội dung: Ma trận tỷ lệ có log từng nguồn trên cùng dân số toàn kỳ.

3 dòng. Các biến: `source`, `population`, `with_log`, `without_log`, `coverage_percent`.

## [customer_count_outliers](tables/review/customer_count_outliers.csv)

Nội dung: Tukey 1,5 IQR và trên P99 cho số log mỗi khách trên toàn dân số.

4 dòng. Các biến: `source`, `rows`, `q1`, `median`, `q3`, `lower_fence`, `upper_fence`, `outlier_n`, `above_p99_n`, `max`.

## [daily_behavior](tables/review/daily_behavior.csv)

Nội dung: Chuỗi ngày liên tục toàn lịch sử từng nguồn; null ngoài khoảng quan sát.

2,193 dòng. Các biến: `source`, `day`, `events`.

## [daily_count_outliers](tables/review/daily_count_outliers.csv)

Nội dung: Tukey 1,5 IQR theo số log/ngày trong khoảng đã có log từng nguồn.

3 dòng. Các biến: `source`, `observed_days`, `q1`, `q3`, `upper_fence`, `outlier_days`, `max_day`, `max_events`.

## [duplicates](tables/review/duplicates.csv)

Nội dung: Trùng khóa, trùng nguyên dòng và ứng viên trùng client–timestamp; số dòng dư sau bản đầu.

13 dòng. Các biến: `source`, `kind`, `duplicate_n`, `duplicate_groups`, `analyzed_rows`, `scope`.

## [embedding_element_quality](tables/review/embedding_element_quality.csv)

Nội dung: Null vector, chiều khác 768, null hoặc không hữu hạn bên trong từng vector detail/ptls.

2 dòng. Các biến: `source`, `rows`, `null_vectors`, `non_768_vectors`, `null_scalar_elements`, `nonfinite_scalar_elements`.

## [events_per_customer](tables/review/events_per_customer.csv)

Nội dung: Mean/min/max và phân vị tổng log mỗi khách toàn lịch sử; tách toàn dân số và nhóm có log.

8 dòng. Các biến: `source`, `population`, `rows`, `finite_n`, `missing_n`, `infinite_n`, `mean`, `std`, `min`, `p25`, `median`, `p75`, `p95`, `p99`, `max`.

## [group_monthly_coverage](tables/review/group_monthly_coverage.csv)

Nội dung: Độ phủ bất kỳ nguồn nào trong từng nhóm phân phối tổng log toàn kỳ.

175 dòng. Các biến: `activity_group`, `month`, `population`, `min_total`, `max_total`, `coverage_percent`.

## [missing_values](tables/review/missing_values.csv)

Nội dung: Null thuộc tính theo cột trên toàn bộ dữ liệu; không đếm ngày không có sự kiện.

61 dòng. Các biến: `source`, `column`, `rows`, `missing_n`, `missing_rate`.

## [monthly_coverage](tables/review/monthly_coverage.csv)

Nội dung: Độ phủ nguồn theo tháng, tổng log, khách có log và khách có ô null.

75 dòng. Các biến: `source`, `month`, `events`, `customers_with_log`, `population`, `unknown_customers`, `coverage_percent`.

## [null_by_event_type](tables/review/null_by_event_type.csv)

Nội dung: Mức thiếu từng họ thuộc tính theo mã event_type gốc.

55 dòng. Các biến: `event_type`, `rows`, `null_currency`, `null_src11`, `null_dst11`, `null_src21`, `null_src31`.

## [null_pairs](tables/review/null_pairs.csv)

Nội dung: Số dòng đồng thời null hoặc chỉ một cột null trong từng cặp thuộc tính.

4 dòng. Các biến: `pair`, `both_null`, `only_one_null`, `rows`.

## [null_target_context](tables/review/null_target_context.csv)

Nội dung: Null diff_trans_date theo nhóm trans_count để kiểm tra giả thuyết thiếu giao dịch.

3 dòng. Các biến: `trans_count_group`, `rows`, `missing_diff`, `min_trans_count`, `max_trans_count`.

## [purposeful_coverage](tables/review/purposeful_coverage.csv)

Nội dung: Số log/tháng của nhóm minh họa cực đại, gần mean, gần median, nhỏ nhất có log, 0 log.

1,000 dòng. Các biến: `group`, `display_id`, `month`, `events`, `full_history_events`.

## [sample_inventory](tables/review/sample_inventory.csv)

Nội dung: Số dòng từng bảng con liên kết trong sampledata; chỉ lưu số tổng hợp trên Git.

8 dòng. Các biến: `source`, `rows`, `path`.

## [sampling_labels](tables/review/sampling_labels.csv)

Nội dung: Đối chiếu nhãn trong mẫu và population; không stratify theo nhãn tương lai.

4 dòng. Các biến: `target`, `population_positive_n`, `population_percent`, `sample_positive_n`, `sample_percent`.

## [sampling_strata](tables/review/sampling_strata.csv)

Nội dung: Phân bổ mẫu theo fold, mask nguồn, band tổng log; chênh lệch đơn vị điểm phần trăm.

178 dòng. Các biến: `stratum`, `population_n`, `sample_n`, `expected_n`, `population_rate`, `sample_rate`, `difference_pp`.

## [sampling_validation](tables/review/sampling_validation.csv)

Nội dung: So sánh phân phối tổng log từng nguồn trên population và mẫu, gồm khoảng cách KS.

24 dòng. Các biến: `source`, `metric`, `population_value`, `sample_value`.

## [sequence_quality](tables/review/sequence_quality.csv)

Nội dung: Null phần tử và độ dài mảng ptls so với mảng event_time trên cùng hàng.

19 dòng. Các biến: `source`, `column`, `rows`, `elements`, `null_elements`, `length_mismatch_rows`.

## [sequence_reconciliation](tables/review/sequence_reconciliation.csv)

Nội dung: Đối chiếu số phần tử ptls với số log detail theo cùng khách, gồm độ dài tối đa và số khách chênh lệch.

3 dòng. Các biến: `source`, `detail_events`, `ptls_events`, `max_detail_events`, `max_sequence_events`, `customers_shorter_sequence`, `customers_longer_sequence`, `customer_presence_mismatch`.

## [source_schema](tables/review/source_schema.csv)

Nội dung: 8 bảng nguồn: đường dẫn, số dòng, grain, từng cột và kiểu lưu thực tế.

61 dòng. Các biến: `source`, `path`, `rows`, `column`, `dtype`, `grain`.

## [target_monthly](tables/review/target_monthly.csv)

Nội dung: Đếm dương/mẫu số và tỷ lệ phần trăm mỗi nhãn theo tháng; cộng fold trước khi chia.

48 dòng. Các biến: `mon`, `target`, `positive_n`, `labelled_n`, `positive_percent`.
