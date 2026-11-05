"""Vai trò các cột MBD-mini; inventory kiểm tra schema thực tế từng shard."""

from dataclasses import dataclass


TARGET_COLUMNS = ("target_1", "target_2", "target_3", "target_4")
TRANSACTION_CATEGORIES = (
    "event_type", "event_subtype", "currency", "src_type11", "src_type12",
    "dst_type11", "dst_type12", "src_type21", "src_type22", "src_type31", "src_type32",
)


@dataclass(frozen=True)
class TableSpec:
    name: str
    relative_path: str
    row_unit: str
    entity_keys: tuple[str, ...]
    unique_key: tuple[str, ...] = ()
    time_column: str | None = None
    numeric_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    target_columns: tuple[str, ...] = ()
    vector_columns: tuple[str, ...] = ()
    auxiliary_columns: tuple[str, ...] = ()


# Một dạng biểu diễn sự kiện trong pipeline: detail. ptls có thể được hỗ trợ sau.
TABLES = (
    TableSpec("clients", "client_split", "customer", ("client_id",),
              unique_key=("client_id",), categorical_columns=("fold",)),
    TableSpec("targets", "targets", "customer-month", ("client_id",),
              unique_key=("client_id", "mon"), time_column="mon",
              categorical_columns=("fold",), target_columns=TARGET_COLUMNS,
              auxiliary_columns=("trans_count", "diff_trans_date")),
    TableSpec("transactions", "detail/trx", "transaction event", ("client_id",),
              time_column="event_time", numeric_columns=("amount",),
              categorical_columns=TRANSACTION_CATEGORIES + ("fold",)),
    TableSpec("geo", "detail/geo", "location event", ("client_id",),
              time_column="event_time",
              categorical_columns=("geohash_4", "geohash_5", "geohash_6", "fold")),
    TableSpec("dialog", "detail/dialog", "dialog event", ("client_id",),
              time_column="event_time", categorical_columns=("fold",),
              vector_columns=("embedding",)),
)
