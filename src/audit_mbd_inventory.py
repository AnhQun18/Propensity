r"""Count every MBD-mini Parquet footer and inspect all top-level schemas.

Writes aggregate inventory only; source files are read without modification.
Run: .\.venv\Scripts\python.exe -X utf8 src/audit_mbd_inventory.py
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "MBD-mini"
OUT = ROOT / "docs" / "mbd_inventory"
TABLES = {
    "client_split": "Khách hàng",
    "targets": "Khách hàng–tháng",
    "detail/trx": "Sự kiện giao dịch",
    "detail/geo": "Sự kiện vị trí",
    "detail/dialog": "Sự kiện hội thoại",
    "ptls/trx": "Chuỗi giao dịch của khách",
    "ptls/geo": "Chuỗi vị trí của khách",
    "ptls/dialog": "Chuỗi hội thoại của khách",
}


def inspect_file(path):
    pf = pq.ParquetFile(path)
    fields = tuple((f.name, str(f.type)) for f in pf.schema_arrow)
    partitions = tuple(sorted({p.split("=", 1)[0] for p in path.relative_to(BASE).parts[:-1] if "=" in p}))
    return pf.metadata.num_rows, fields, partitions


def main():
    tables = {}
    all_columns = set()
    for name, unit in TABLES.items():
        files = sorted((BASE / name).rglob("*.parquet"))
        if not files:
            raise ValueError(f"No Parquet files: {name}")
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(inspect_file, files))
        schemas = {r[1] for r in results}
        partitions = {r[2] for r in results}
        if len(schemas) != 1 or len(partitions) != 1:
            raise ValueError(f"Inconsistent schemas/partition keys: {name}")
        fields = next(iter(schemas))
        partition_keys = list(next(iter(partitions)))
        physical = [n for n, _ in fields]
        logical = list(dict.fromkeys(physical + partition_keys))
        all_columns.update(logical)
        tables[name] = {
            "files": len(files), "rows": sum(r[0] for r in results), "unit": unit,
            "physical_column_count": len(physical), "logical_column_count": len(logical),
            "physical_columns": dict(fields), "partition_keys": partition_keys,
        }
        print(name, tables[name]["rows"], len(physical), len(logical), flush=True)

    clients = pq.read_table(BASE / "client_split", columns=["client_id"]).to_pandas()
    targets = pq.read_table(BASE / "targets", columns=["client_id", "mon"]).to_pandas()
    if len(clients) != tables["client_split"]["rows"] or len(targets) != tables["targets"]["rows"]:
        raise ValueError("Footer and read row counts differ")
    summary = {
        "checked_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "scope": "Every Parquet footer and top-level schema; full client_split IDs and target client-month keys",
        "tables": tables,
        "parquet_files": sum(t["files"] for t in tables.values()),
        "unique_clients": int(clients.client_id.nunique()),
        "target_unique_clients": int(targets.client_id.nunique()),
        "duplicate_client_ids_in_split": int(clients.client_id.duplicated().sum()),
        "duplicate_target_client_months": int(targets.duplicated(["client_id", "mon"]).sum()),
        "months": sorted(targets.mon.unique().tolist()),
        "detail_events": sum(t["rows"] for n, t in tables.items() if n.startswith("detail/")),
        "distinct_top_level_column_names": sorted(all_columns),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Kiểm đếm biến và quan sát MBD-mini", "",
        f"Thời điểm: {summary['checked_at']}. Quét đủ {summary['parquet_files']:,} file Parquet; kiểm tra schema của từng file và số dòng trong footer. Đọc toàn bộ khóa khách hàng–tháng để kiểm tra số khách, tháng và trùng khóa.", "",
        "| Bảng | Quan sát (dòng) | Đơn vị quan sát | Cột trong file | Cột khi gồm partition |",
        "|---|---:|---|---:|---:|",
    ]
    for name, t in tables.items():
        lines.append(f"| `{name}` | {t['rows']:,} | {t['unit']} | {t['physical_column_count']} | {t['logical_column_count']} |")
    lines += ["", f"Có **{summary['unique_clients']:,} khách hàng**, **{len(summary['months'])} tháng nhãn**, **{tables['targets']['rows']:,} mẫu khách hàng–tháng** và **{summary['detail_events']:,} sự kiện detail**.", "",
        "`fold` được lưu trong đường dẫn partition, nên số cột khi đọc cả bảng nhiều hơn số cột vật lý trong file một cột. `embedding` được tính là một cột mảng; theo audit nội dung trước, mỗi vector hội thoại có 768 phần tử. Các cột mảng trong ptls cũng được tính là một cột mỗi trường, không trải phẳng thành số phần tử.", "",
        "Không cộng detail và ptls thành tổng quan sát độc lập: đó là hai dạng biểu diễn của lịch sử. Cũng không cộng các bảng có đơn vị quan sát khác nhau để suy ra số mẫu train. Số cột nguồn chưa phải số feature sau tổng hợp lịch sử.", "",
        f"Không trùng ID trong client_split: {summary['duplicate_client_ids_in_split'] == 0}. Không trùng khóa khách hàng–tháng trong targets: {summary['duplicate_target_client_months'] == 0}.", "",
        "Tên cột và kiểu dữ liệu của từng bảng nằm trong `summary.json`. Chạy lại: `.\\.venv\\Scripts\\python.exe -X utf8 src/audit_mbd_inventory.py`.", ""]
    (OUT / "assessment.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "tables"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
