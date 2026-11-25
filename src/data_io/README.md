# `data_io` — Multi-Format Tabular IO, Inventory & Integrity

Đọc/ghi dữ liệu đa định dạng theo batch (không tràn RAM), kiểm kê schema song song, băm fingerprint kiểm soát tính bất biến dữ liệu nguồn, quản lý phiên DuckDB disk-backed và ghi file atomic có SHA-256.

## Định dạng hỗ trợ
- **File / Thư mục / Glob:** Parquet (single, sharded, Hive partition), CSV / TSV (+ `.gz`, `.bz2`, `.xz`, `.zst`, `.zip`), JSON / JSONL / NDJSON, Arrow IPC / Feather, ORC, Excel (`.xlsx`/`.xls`/`.ods`), Pickle (opt-in `allow_pickle=True`).
- **Database:** SQLite (`path/to/db.sqlite::table`), DuckDB (`path/to/db.duckdb::table`).
- **In-memory:** `pandas.DataFrame`, `pyarrow.Table`, `pyarrow.RecordBatchReader`, Polars `DataFrame`/`LazyFrame`, `duckdb.DuckDBPyRelation`, `list[dict]`.

## Sử dụng nhanh (chỉ cần giữ thư mục `src/` trong thư mục làm việc)

```python
from src.data_io import iter_batches, load_dataframe, write_table, inspect_inventory, DuckDBSession, RunManifest

# Đọc streaming bất kể định dạng
for batch in iter_batches("data/events/*.csv.gz", batch_size=100_000):
    ...

# Đăng ký nhiều định dạng vào cùng phiên DuckDB
with DuckDBSession(memory_limit="4GB", threads=2) as db:
    db.register("users", "data/users.csv")
    db.register("orders", "data/orders_parquet/")
    df = db.query("SELECT count(*) AS n FROM orders")

# Ghi atomic kèm SHA-256
write_table(df, "outputs/summary.parquet")
```

## CLI
```powershell
python -m src.data_io inspect data/orders/
python -m src.data_io convert data/orders.csv outputs/orders.parquet
```
