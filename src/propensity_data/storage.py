"""Read-only source discovery, SQL quoting and output helpers."""
import json
from pathlib import Path


def literal(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def files_at(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.parquet")) if root.is_dir() else [root] if root.is_file() else []
    if not files:
        raise FileNotFoundError(f"No Parquet files under {root}")
    return files


def parquet_scan(files) -> str:
    paths = ",".join(literal(p.as_posix()) for p in files)
    return (f"read_parquet([{paths}], hive_partitioning=false, union_by_name=false, "
            "filename=true, file_row_number=true)")


def export(con, query: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({query}) TO {literal(path.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")


def assert_empty(con, query: str, message: str):
    if con.execute(f"SELECT EXISTS ({query})").fetchone()[0]:
        raise ValueError(message)


def normalized_id(column="client_id"):
    return f"nullif(trim(cast({column} AS VARCHAR)), '')"
