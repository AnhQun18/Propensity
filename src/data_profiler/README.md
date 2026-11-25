# `data_profiler` — Thống kê mô tả cơ bản & Out-of-Core 1-Pass

Thống kê mô tả cho **mọi bảng đơn lẻ** lẫn **mô hình đa bảng (entity + targets + K nguồn sự kiện)**, hỗ trợ đa định dạng (Parquet, CSV/TSV/gz, JSONL, Feather, Excel, SQLite, DuckDB, DataFrame).

## Tính năng chính
- **`describe_table(source)`**: tóm tắt 1 dòng/cột (dtype, kiểu suy diễn, missing %, nunique, top value, mean/std/quantiles, `zeros_pct`, `negative_n`, `skewness`, `kurtosis`).
- **`profile_table` & `profile_dataset`**: quét 1-pass không tràn RAM bằng SQLite tạm + `OnlineMoments` (Chan/Welford chính xác trên 100% giá trị hữu hạn) + `UniformSample` (bottom-k reservoir sampling bất biến biên batch).
- Kiểm tra trùng nguyên dòng (`exact` / `sample` / `off`), chất lượng vector/embedding và `PrimaryCodeGrouper`.

## Sử dụng nhanh
```python
from src.data_profiler import describe_table, profile_table, profile_dataset, TableSpec, DatasetSpec

summary = describe_table("data/orders.csv")
res = profile_table("data/orders.csv", entity_key="user_id", time_column="order_ts")
```
