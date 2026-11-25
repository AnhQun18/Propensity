# `data_audit` — Data Quality EDA Audits

Bộ kiểm toán chất lượng dữ liệu read-only cho mọi tập dữ liệu (Parquet, CSV/TSV, JSONL, SQLite, DuckDB, DataFrame...):
1. **`audit_null_patterns`**: tỷ lệ null theo cột, số cột null/dòng, tổ hợp đồng thiếu (`joint co-missingness`), breakdown theo nhóm.
2. **`audit_univariate_outliers`** & **`audit_within_group_outliers`**: ngoại lai đơn biến (IQR + Modified Z-score MAD + percentile tails) và ngoại lai bền vững trong nhóm (`log-MAD` theo hash-bucket, chữ ký `10^k`, lặp timestamp-giá trị, KMeans profile clustering).
3. **`audit_temporal_coverage`**: khoảng giao thời gian chung giữa $M$ nguồn, ma trận `entity × day`, tỷ lệ thực thể vắng mặt `>50%`, so sánh `raw` vs `strict quality`.
4. **`audit_invalid_records`** & **`audit_join_keys`**: thống kê bản ghi sai ID/timestamp, trùng timestamp, kiểm tra khóa nối, orphan ID và đồng xuất hiện nhãn.

## Sử dụng nhanh
```python
from src.data_audit import audit_null_patterns, audit_univariate_outliers, audit_within_group_outliers, audit_join_keys

nulls = audit_null_patterns("data/orders.csv", group_by=["channel"])
uni_out = audit_univariate_outliers("data/orders.csv")
grp_out = audit_within_group_outliers("data/orders.parquet", entity_col="user_id", value_col="amount", group_cols=["currency"])
```
