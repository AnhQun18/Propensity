"""Một lần quét sự kiện; tổng hợp xuyên batch bằng SQLite trong thư mục tạm."""

from collections import Counter
from dataclasses import dataclass, field
from itertools import islice
import json
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .io import iter_batches, parquet_files, parse_times, partitions
from .schema import TABLES
from .statistics import Moments, UniformSample, canonical, row_keys


EVENT_SOURCES = ("transactions", "geo", "dialog")
PREFIXES = {"transactions": "trx", "geo": "geo", "dialog": "dialog"}


@dataclass
class SourceStats:
    rows: int = 0
    missing: Counter = field(default_factory=Counter)
    invalid: Counter = field(default_factory=Counter)
    dtypes: dict = field(default_factory=dict)
    start: object = None
    end: object = None
    amount: Moments = field(default_factory=Moments)
    norms: Moments = field(default_factory=Moments)
    embedding_lengths: Counter = field(default_factory=Counter)
    calendar: Counter = field(default_factory=Counter)
    missing_groups: Counter = field(default_factory=Counter)
    group_rows: Counter = field(default_factory=Counter)
    location_column: str | None = None
    sample: object = None
    duplicate_n: int | None = None
    duplicate_scope: str = "off"
    duplicate_rows: int = 0


class EventEngine:
    def __init__(self, context):
        self.context = context
        if context.config.work_dir is not None:
            if context.config.work_dir.resolve().is_relative_to(context.config.data_root.resolve()):
                raise ValueError("work_dir không được nằm trong thư mục dữ liệu nguồn")
            context.config.work_dir.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="mbd-eda-", dir=context.config.work_dir)
        self.db = sqlite3.connect(Path(self.temp.name) / "aggregates.sqlite")
        self.db.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            PRAGMA cache_size=-32768;
            CREATE TABLE daily (source TEXT, client_id TEXT, day TEXT, event_n INTEGER,
                first_ns INTEGER, last_ns INTEGER, amount_n INTEGER, amount_sum REAL,
                PRIMARY KEY(source, client_id, day)) WITHOUT ROWID;
            CREATE TABLE locations (client_id TEXT, day TEXT, value TEXT,
                PRIMARY KEY(client_id, day, value)) WITHOUT ROWID;
            CREATE TABLE categories (source TEXT, column_name TEXT, kind TEXT, value TEXT, n INTEGER,
                PRIMARY KEY(source,column_name,kind,value)) WITHOUT ROWID;
            CREATE TABLE duplicate_rows (source TEXT, row_key BLOB, n INTEGER,
                PRIMARY KEY(source,row_key)) WITHOUT ROWID;
        """)
        self.stats = {}
        self.scanned = False

    def close(self):
        self.db.close()
        self.temp.cleanup()

    def query(self, sql, params=()):
        return pd.read_sql_query(sql, self.db, params=params)

    def scan(self):
        if self.scanned:
            return self
        config = self.context.config
        clients = self.context.artifacts["clients"].set_index("client_id").fold
        for index, source in enumerate(EVENT_SOURCES):
            state = SourceStats(sample=UniformSample(config.sample_size, config.random_seed + index))
            self.stats[source] = state
            spec = next(t for t in TABLES if t.name == source)
            batches = iter_batches(self.context.sources[source], batch_size=config.batch_size)
            if config.max_batches_per_source is not None:
                batches = islice(batches, config.max_batches_per_source)
            for frame in batches:
                required = {"client_id", "event_time"} | ({"amount"} if source == "transactions" else set())
                if source == "dialog":
                    required.add("embedding")
                if required - set(frame):
                    raise ValueError(f"{source} thiếu {sorted(required - set(frame))}")
                if source == "geo" and not any(c in frame for c in ("geohash_4", "geohash_5", "geohash_6")):
                    raise ValueError("geo cần ít nhất một cột geohash_4/5/6")
                state.rows += len(frame)
                state.missing.update({c: int(frame[c].isna().sum()) for c in frame})
                for c in frame:
                    state.dtypes.setdefault(c, set()).add(str(frame[c].dtype))
                state.sample.update(frame)
                times = parse_times(frame.event_time, config.timezone)
                valid_time = times.notna()
                state.invalid["invalid_event_time"] += int((~valid_time & frame.event_time.notna()).sum())
                if valid_time.any():
                    first, last = times.min(), times.max()
                    state.start = first if state.start is None else min(state.start, first)
                    state.end = last if state.end is None else max(state.end, last)
                ids = frame.client_id.astype("string")
                mapped = ids.map(clients)
                state.invalid["unknown_client_rows"] += int((ids.notna() & ~ids.isin(clients.index)).sum())
                if "fold" in frame:
                    state.invalid["fold_mismatch_rows"] += int((mapped.notna() & frame.fold.ne(mapped)).sum())
                month = times.dt.strftime("%Y-%m").fillna("<INVALID_TIME>")
                dimensions = [("month", month)]
                if "event_type" in frame:
                    dimensions.append(("event_type", frame.event_type.map(lambda v: json.dumps(canonical(v)))))
                for dimension, values in dimensions:
                    size = frame.groupby(values, dropna=False).size()
                    state.group_rows.update({(dimension, str(k)): int(v) for k, v in size.items()})
                    grouped = frame.isna().groupby(values, dropna=False).sum()
                    for key, row in grouped.iterrows():
                        state.missing_groups.update({(dimension, str(key), c): int(n) for c, n in row.items()})
                cells = pd.DataFrame({"weekday": times[valid_time].dt.day_of_week,
                                      "hour": times[valid_time].dt.hour}).value_counts()
                state.calendar.update({(int(day), int(hour)): int(n) for (day, hour), n in cells.items()})
                categories = [c for c in spec.categorical_columns if c in frame]
                for column in categories:
                    counts = frame[column].dropna().value_counts()
                    rows = [(source, column, "value", json.dumps(canonical(k)), int(n)) for k, n in counts.items()]
                    rows.append((source, column, "missing", "", int(frame[column].isna().sum())))
                    self.db.executemany("""INSERT INTO categories VALUES (?,?,?,?,?)
                        ON CONFLICT(source,column_name,kind,value) DO UPDATE SET n=n+excluded.n""", rows)
                amounts = pd.to_numeric(frame.amount, errors="coerce") if "amount" in frame else pd.Series(np.nan, index=frame.index)
                finite = np.isfinite(amounts.fillna(np.nan).to_numpy(dtype=float))
                if source == "transactions":
                    state.amount.update(amounts[finite])
                    state.invalid["invalid_amount"] += int((amounts.isna() & frame.amount.notna()).sum())
                    state.invalid["infinite_amount"] += int(np.isinf(amounts.fillna(np.nan).astype(float)).sum())
                if source == "dialog":
                    for embedding in frame.embedding:
                        if embedding is None or embedding is pd.NA:
                            state.invalid["missing_embedding"] += 1
                            continue
                        try:
                            vector = np.asarray(embedding, dtype=float)
                        except (TypeError, ValueError):
                            state.invalid["invalid_embedding_values"] += 1
                            continue
                        length = len(vector) if vector.ndim == 1 else -1
                        state.embedding_lengths[length] += 1
                        if length != config.embedding_size:
                            state.invalid["invalid_embedding_shape"] += 1
                        if vector.ndim != 1 or not np.isfinite(vector).all():
                            state.invalid["invalid_embedding_values"] += 1
                        elif length == config.embedding_size:
                            state.norms.update([np.linalg.norm(vector)])
                valid = ids.notna() & valid_time
                data = pd.DataFrame({"client_id": ids[valid].astype(str),
                    "day": times[valid].dt.strftime("%Y-%m-%d"),
                    "ns": times[valid].astype("int64"), "amount_n": finite[valid.to_numpy()].astype(int),
                    "amount_sum": amounts[valid].where(finite[valid.to_numpy()], 0).astype(float)})
                grouped = data.groupby(["client_id", "day"], sort=False).agg(
                    event_n=("ns", "size"), first_ns=("ns", "min"), last_ns=("ns", "max"),
                    amount_n=("amount_n", "sum"), amount_sum=("amount_sum", "sum")).reset_index()
                self.db.executemany("""INSERT INTO daily VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(source,client_id,day) DO UPDATE SET
                    event_n=event_n+excluded.event_n, first_ns=MIN(first_ns,excluded.first_ns),
                    last_ns=MAX(last_ns,excluded.last_ns), amount_n=amount_n+excluded.amount_n,
                    amount_sum=amount_sum+excluded.amount_sum""",
                    ((source, str(r.client_id), r.day, int(r.event_n), int(r.first_ns), int(r.last_ns),
                      int(r.amount_n), float(r.amount_sum)) for r in grouped.itertuples(index=False)))
                if source == "geo":
                    location = next((c for c in ("geohash_6", "geohash_5", "geohash_4") if c in frame), None)
                    state.location_column = location
                    if location:
                        loc = frame.loc[valid & frame[location].notna(), ["client_id", location]].copy()
                        loc["day"] = times.loc[loc.index].dt.strftime("%Y-%m-%d")
                        loc = loc.drop_duplicates()
                        self.db.executemany("INSERT OR IGNORE INTO locations VALUES (?,?,?)",
                            ((str(r[0]), r[2], json.dumps(canonical(r[1]))) for r in loc.itertuples(index=False, name=None)))
                if config.duplicate_mode == "exact":
                    self.db.executemany("""INSERT INTO duplicate_rows VALUES (?,?,1)
                        ON CONFLICT(source,row_key) DO UPDATE SET n=n+1""",
                        ((source, key) for key in row_keys(frame)))
                self.db.commit()
            state.duplicate_scope = config.duplicate_mode
            if config.duplicate_mode == "exact":
                state.duplicate_n = self.db.execute("SELECT COALESCE(SUM(n-1),0) FROM duplicate_rows WHERE source=?", (source,)).fetchone()[0]
                state.duplicate_rows = state.rows
                state.duplicate_scope = "all_scanned_rows"
            elif config.duplicate_mode == "sample":
                keys = list(row_keys(state.sample.frame))
                state.duplicate_n = len(keys) - len(set(keys))
                state.duplicate_rows = len(keys)
                state.duplicate_scope = "all_scanned_rows" if len(keys) == state.rows else "uniform_row_sample"
        self.db.execute("CREATE INDEX daily_day ON daily(day,source)")
        self.db.execute("CREATE INDEX locations_day ON locations(day)")
        self.db.commit()
        self.scanned = True
        return self

    def categories(self, source):
        rows = []
        for column, in self.db.execute("SELECT DISTINCT column_name FROM categories WHERE source=? ORDER BY column_name", (source,)):
            top = self.query("SELECT value,n FROM categories WHERE source=? AND column_name=? AND kind='value' ORDER BY n DESC,value LIMIT ?", (source, column, self.context.config.max_categories))
            total = self.stats[source].rows
            distinct, nonmissing = self.db.execute("SELECT COUNT(*),COALESCE(SUM(n),0) FROM categories WHERE source=? AND column_name=? AND kind='value'", (source,column)).fetchone()
            for r in top.itertuples(index=False):
                rows.append(dict(source=source, column=column, kind="value", value=r.value, n=r.n, distinct_n=distinct, rows=total))
            missing = self.db.execute("SELECT n FROM categories WHERE source=? AND column_name=? AND kind='missing'", (source, column)).fetchone()
            rows.extend([dict(source=source,column=column,kind="remaining",value=None,n=int(nonmissing-top.n.sum()),distinct_n=distinct,rows=total),
                         dict(source=source,column=column,kind="missing",value=None,n=missing[0] if missing else 0,distinct_n=distinct,rows=total)])
        frame = pd.DataFrame(rows, columns=["source","column","kind","value","n","distinct_n","rows"])
        frame["rate"] = frame.n / frame.rows.replace(0, np.nan)
        return frame

    def aggregate(self, start=None, end=None):
        where, params = "", []
        if start is not None:
            where = " WHERE day>=? AND day<?"
            params = [pd.Timestamp(start).strftime("%Y-%m-%d"), pd.Timestamp(end).strftime("%Y-%m-%d")]
        return self.query("""SELECT source,client_id,SUM(event_n) event_count,COUNT(*) active_days,
            MIN(first_ns) first_ns,MAX(last_ns) last_ns,SUM(amount_n) amount_n,SUM(amount_sum) amount_sum
            FROM daily""" + where + " GROUP BY source,client_id", params)

    def active_days(self, start=None, end=None):
        where, params = "", []
        if start is not None:
            where = " WHERE day>=? AND day<?"
            params = [pd.Timestamp(start).strftime("%Y-%m-%d"), pd.Timestamp(end).strftime("%Y-%m-%d")]
        return self.query("SELECT client_id,COUNT(DISTINCT day) any_active_days FROM daily" + where + " GROUP BY client_id", params)

    def unique_locations(self, start=None, end=None):
        where, params = "", []
        if start is not None:
            where = " WHERE day>=? AND day<?"
            params = [pd.Timestamp(start).strftime("%Y-%m-%d"), pd.Timestamp(end).strftime("%Y-%m-%d")]
        return self.query("SELECT client_id,COUNT(DISTINCT value) geo_unique_locations FROM locations" + where + " GROUP BY client_id", params)


def engine(context, *, scan=True):
    if "engine" not in context.runtime:
        context.runtime["engine"] = EventEngine(context)
    value = context.runtime["engine"]
    return value.scan() if scan else value


def inventory(context):
    if "inventory" not in context.runtime:
        rows, columns = [], []
        for spec in TABLES:
            files = parquet_files(context.sources[spec.name])
            schemas = set()
            total = 0
            logical = {}
            for file in files:
                reader = pq.ParquetFile(file)
                schemas.add(tuple((f.name, str(f.type)) for f in reader.schema_arrow))
                total += reader.metadata.num_rows
                logical.update({f.name: str(f.type) for f in reader.schema_arrow})
                logical.update({k: "partition" for k in partitions(file)})
            if len(schemas) != 1:
                raise ValueError(f"Schema thay đổi: {spec.name}")
            rows.append(dict(source=spec.name,files=len(files),rows=total,row_unit=spec.row_unit,
                             logical_columns=len(logical),schema_variants=len(schemas)))
            for name, dtype in logical.items():
                role = ("id" if name in spec.entity_keys else "target" if name in spec.target_columns
                        else "datetime" if name == spec.time_column else "numeric" if name in spec.numeric_columns
                        else "categorical" if name in spec.categorical_columns else "vector" if name in spec.vector_columns
                        else "auxiliary" if name in spec.auxiliary_columns else "unassigned")
                columns.append(dict(source=spec.name,column=name,dtype=dtype,role=role))
        context.runtime["inventory"] = (pd.DataFrame(rows), pd.DataFrame(columns))
    return context.runtime["inventory"]
