"""Parquet nhiều shard, partition Hive và bảng tham chiếu có khóa duy nhất."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import TABLES, TARGET_COLUMNS


def nullable_integer_dtype(dtype):
    """Giữ mã nguyên chính xác khi batch có null hoặc không có pandas metadata."""
    if pa.types.is_signed_integer(dtype):
        return pd.Int64Dtype()
    if pa.types.is_unsigned_integer(dtype):
        return pd.UInt64Dtype()
    return None


def source_paths(data_root: str | Path) -> dict[str, Path]:
    root = Path(data_root)
    return {table.name: root / table.relative_path for table in TABLES}


def parquet_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy nguồn dữ liệu: {path}. Kiểm tra data_root trong JSON; "
                                "thư mục gốc phải chứa client_split, targets và detail.")
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    if not files:
        raise ValueError(f"Không có file Parquet: {path}")
    return files


def partitions(file: Path) -> dict:
    result = {}
    for part in file.parent.parts:
        if "=" in part:
            key, value = part.split("=", 1)
            value = unquote(value)
            result[key] = int(value) if key == "fold" else value
    return result


def iter_batches(path: str | Path, *, columns: Sequence[str] | None = None,
                 batch_size: int = 100_000) -> Iterator[pd.DataFrame]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size phải là số nguyên dương")
    schema = None
    logical_names = None
    for file in parquet_files(path):
        reader = pq.ParquetFile(file)
        current = reader.schema_arrow.remove_metadata()
        part = partitions(file)
        names = set(current.names) | set(part)
        if schema is not None and (not current.equals(schema) or names != logical_names):
            raise ValueError(f"Schema thay đổi giữa các shard: {file}")
        schema, logical_names = current, names
        if columns is not None and set(columns) - names:
            raise ValueError(f"Thiếu cột {sorted(set(columns) - names)} trong {file}")
        physical = None if columns is None else [c for c in columns if c in current.names]
        for batch in reader.iter_batches(batch_size=batch_size, columns=physical):
            frame = batch.to_pandas(types_mapper=nullable_integer_dtype)
            for key, value in part.items():
                if key in frame and not frame[key].eq(value).fillna(False).all():
                    raise ValueError(f"Partition {key} xung đột với nội dung: {file}")
                frame[key] = value
            yield frame if columns is None else frame[list(columns)]


def read_reference_tables(sources: dict[str, Path]) -> dict[str, pd.DataFrame]:
    result = {}
    for name in ("clients", "targets"):
        frames = list(iter_batches(sources[name]))
        if not frames:
            raise ValueError(f"Bảng {name} rỗng")
        frame = pd.concat(frames, ignore_index=True)
        required = {"client_id", "fold"} | ({"mon", *TARGET_COLUMNS} if name == "targets" else set())
        if required - set(frame):
            raise ValueError(f"{name} thiếu cột {sorted(required - set(frame))}")
        frame["client_id"] = frame.client_id.astype("string")
        keys = ["client_id", "mon"] if name == "targets" else ["client_id"]
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{name}: khóa {keys} null hoặc trùng")
        if frame.fold.isna().any():
            raise ValueError(f"{name}: fold bị thiếu")
        if name == "targets":
            frame["mon"] = pd.to_datetime(frame.mon, errors="raise").dt.strftime("%Y-%m-%d")
            if frame.duplicated(keys).any():
                raise ValueError("targets trùng khóa sau chuẩn hóa mon")
            for col in TARGET_COLUMNS:
                if not frame[col].dropna().isin([0, 1]).all():
                    raise ValueError(f"{col} phải là 0/1 hoặc null")
        result[name] = frame
    joined = result["targets"][["client_id", "fold"]].merge(
        result["clients"][["client_id", "fold"]], on="client_id", how="left",
        validate="many_to_one", suffixes=("_target", "_client"), indicator=True)
    if joined._merge.ne("both").any() or joined.fold_target.ne(joined.fold_client).any():
        raise ValueError("targets có client_id ngoài client_split hoặc fold không khớp")
    return result


def parse_times(values, timezone=None) -> pd.Series:
    result = pd.to_datetime(values, errors="coerce", format="mixed")
    if not hasattr(result, "dt"):
        raise ValueError("Timestamp phải dùng cùng một quy ước timezone")
    tz = result.dt.tz
    if tz is not None and timezone is None:
        raise ValueError("Nguồn có timezone; cần khai báo timezone trong cấu hình")
    if timezone is not None:
        result = (result.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
                  if tz is None else result.dt.tz_convert(timezone))
    return result.dt.as_unit("ns")
