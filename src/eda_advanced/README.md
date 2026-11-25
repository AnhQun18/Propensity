# `eda_advanced` — Advanced Tabular & Longitudinal EDA

Bộ công cụ **EDA nâng cao** cho cả dữ liệu bảng thông thường lẫn dữ liệu chuỗi thời gian / panel (`entity × period`), hỗ trợ mọi định dạng qua `data_io`:

### 1. EDA đa biến & quan hệ với Target (Bảng tổng quát)
- **`correlation_analysis`**: ma trận tương quan Pearson / Spearman / Kendall + trích các cặp tương quan mạnh.
- **`categorical_associations`**: Cramér's V hiệu chỉnh thiên lệch + kiểm định Chi-square cho biến phân loại.
- **`compute_vif`**: tính Variance Inflation Factor phát hiện đa cộng tuyến.
- **`woe_iv_table` & `target_signal_scan`**: quét toàn bộ biến số & phân loại theo Information Value (IV), Weight of Evidence (WoE), ROC-AUC đơn biến, KS statistic, Mann-Whitney U và Cramér's V.

### 2. Phân tích trôi phân phối & Kiểm định thống kê
- **`population_stability_index`**, **`compare_distributions`**, **`temporal_drift_report`**: PSI, KS-test, Wasserstein distance, Jensen-Shannon divergence và tỷ lệ danh mục mới.
- **`wilson_ci`**, **`cluster_bootstrap_rates`** (CI theo cụm thực thể cho rate & lift), **`compare_two_groups`** (Mann-Whitney U, Welch t-test, KS, Cohen's d, rank-biserial).

### 3. EDA chuỗi thời gian & cấu trúc hành vi (Panel / Event)
- **`periodic_categorical_profile`**: entropy, HHI, top-share, độ mới (`new_items`), đổi mã chính (`changed_main`).
- **`analyze_bipartite_graph`**: đồ thị hai phía entity–item, event-weighted IDF, connected components, TF-IDF + TruncatedSVD.
- **`run_event_study` & `run_stratified_event_study`**: diễn biến quanh mốc neo `[-k, +k]`, chênh lệch nền cá nhân & phân tầng cố định chống Nghịch lý Simpson.
- **`period_change_decomposition`**: phân rã Shift-Share 6 nhóm loại trừ nhau hoàn toàn (cộng khớp 100%).
- **`build_multi_horizon_targets`**, **`stratified_lift_table`**, **`buyer_composition_and_transitions`**, **`compute_surge_ratio`**.

## Sử dụng nhanh
```python
from src.eda_advanced import correlation_analysis, target_signal_scan, compare_distributions

corr = correlation_analysis("data/customers.csv", method="spearman", threshold=0.6)
signals = target_signal_scan("data/customers.csv", target_col="churn")
drift = compare_distributions("data/jan.parquet", "data/feb.parquet")
```
