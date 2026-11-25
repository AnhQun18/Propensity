"""Single-pass out-of-core event/tabular profiling engine backed by temporary SQLite."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import islice
import json
from pathlib import Path
import sqlite3
import tempfile
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ._compat import iter_batches, parse_timestamps
from .spec import DatasetSpec, ProfilerConfig, TableSpec
from .streaming import OnlineMoments, UniformSample, canonical_value, iter_row_keys


@dataclass
class SourceProfileState:
    rows: int = 0
    missing: Counter = field(default_factory=Counter)
    invalid: Counter = field(default_factory=Counter)
    dtypes: dict[str, set[str]] = field(default_factory=dict)
    start: Any = None
    end: Any = None
    numeric_moments: dict[str, OnlineMoments] = field(default_factory=dict)
    vector_norms: dict[str, OnlineMoments] = field(default_factory=dict)
    vector_lengths: dict[str, Counter] = field(default_factory=dict)
    duration_seconds: float = 0.0
    batch_size: int = 0
    batches: int = 0
    sample: UniformSample | None = None
    duplicate_n: int | None = None
    duplicate_scope: str = "off"
    duplicate_rows: int = 0


class SQLiteProfileEngine:
    """Scan all event/tabular sources in one pass and persist cross-batch aggregates in SQLite."""

    def __init__(
        self,
        dataset: DatasetSpec,
        config: ProfilerConfig,
        *,
        entity_ids: pd.Index | None = None,
        progress=None,
    ):
        self.dataset = dataset
        self.config = config
        self.entity_ids = entity_ids
        self.progress = progress
        if config.work_dir is not None:
            config.work_dir.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="data-profiler-", dir=config.work_dir)
        self.db = sqlite3.connect(Path(self.temp.name) / "profile.sqlite")
        self.db.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            CREATE TABLE daily (
                source TEXT, entity_id TEXT, day TEXT,
                event_n INTEGER, first_ns INTEGER, last_ns INTEGER,
                PRIMARY KEY(source, entity_id, day)
            ) WITHOUT ROWID;
            CREATE TABLE categories (
                source TEXT, column_name TEXT, kind TEXT, value TEXT, n INTEGER,
                PRIMARY KEY(source, column_name, kind, value)
            ) WITHOUT ROWID;
            CREATE TABLE duplicate_rows (
                source TEXT, row_key BLOB, n INTEGER,
                PRIMARY KEY(source, row_key)
            ) WITHOUT ROWID;
            """
        )
        self.db.execute(f"PRAGMA cache_size=-{config.sqlite_cache_mb * 1024}")
        self.stats: dict[str, SourceProfileState] = {}
        self._cache: dict[str, pd.DataFrame] = {}
        self.scanned = False

    def close(self) -> None:
        self.db.close()
        self.temp.cleanup()

    def __enter__(self) -> "SQLiteProfileEngine":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def query(self, sql: str, params=()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.db, params=params)

    def _emit(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def scan(self) -> "SQLiteProfileEngine":
        if self.scanned:
            return self
        tz = self.config.timezone if self.config.timezone is not None else self.dataset.timezone
        for index, spec in enumerate(self.dataset.event_tables()):
            self._scan_table(spec, index, tz)
        self.db.commit()
        self.scanned = True
        return self

    def _scan_table(self, spec: TableSpec, index: int, timezone: str | None) -> None:
        started = perf_counter()
        batch_size = spec.batch_size or self.config.batch_size
        state = SourceProfileState(
            sample=UniformSample(self.config.sample_size, self.config.random_seed + index),
            batch_size=batch_size,
        )
        for col in spec.numeric_columns:
            state.numeric_moments[col] = OnlineMoments()
        for col in spec.vector_columns:
            state.vector_norms[col] = OnlineMoments()
            state.vector_lengths[col] = Counter()
        self.stats[spec.name] = state

        batches = iter_batches(spec.source, batch_size=batch_size, **spec.source_options)
        if self.config.max_batches_per_source is not None:
            batches = islice(batches, self.config.max_batches_per_source)

        for frame in batches:
            state.batches += 1
            state.rows += len(frame)
            state.missing.update({c: int(frame[c].isna().sum()) for c in frame.columns})
            for col in frame.columns:
                state.dtypes.setdefault(col, set()).add(str(frame[col].dtype))
            state.sample.update(frame)

            # Time column checks
            times = None
            valid_time = None
            if spec.time_column and spec.time_column in frame.columns:
                raw_time = frame[spec.time_column]
                times = parse_timestamps(raw_time, timezone=timezone)
                valid_time = times.notna()
                state.invalid["invalid_time"] += int((~valid_time & raw_time.notna()).sum())
                if valid_time.any():
                    first, last = times.min(), times.max()
                    state.start = first if state.start is None else min(state.start, first)
                    state.end = last if state.end is None else max(state.end, last)

            # Entity key checks
            ids = None
            if spec.entity_key and spec.entity_key in frame.columns:
                ids = frame[spec.entity_key].astype("string")
                if self.entity_ids is not None:
                    state.invalid["unknown_entity_rows"] += int((ids.notna() & ~ids.isin(self.entity_ids)).sum())

            # Numeric columns
            for col in spec.numeric_columns:
                if col not in frame.columns:
                    continue
                raw = frame[col]
                nums = pd.to_numeric(raw, errors="coerce")
                arr = nums.to_numpy(dtype=float, na_value=np.nan)
                finite = np.isfinite(arr)
                state.numeric_moments[col].update(arr[finite])
                state.invalid[f"invalid_{col}"] += int((nums.isna() & raw.notna()).sum())
                state.invalid[f"infinite_{col}"] += int(np.isinf(arr).sum())

            # Categorical columns
            for col in spec.categorical_columns:
                if col not in frame.columns:
                    continue
                counts = frame[col].dropna().value_counts()
                rows = [
                    (spec.name, col, "value", json.dumps(canonical_value(k), ensure_ascii=False), int(n))
                    for k, n in counts.items()
                ]
                rows.append((spec.name, col, "missing", "", int(frame[col].isna().sum())))
                self.db.executemany(
                    """INSERT INTO categories VALUES (?,?,?,?,?)
                       ON CONFLICT(source, column_name, kind, value)
                       DO UPDATE SET n = n + excluded.n""",
                    rows,
                )

            # Vector columns
            for col in spec.vector_columns:
                if col not in frame.columns:
                    continue
                expected_dim = spec.vector_dims.get(col)
                for vec in frame[col]:
                    if vec is None or vec is pd.NA:
                        state.invalid[f"missing_{col}"] += 1
                        continue
                    try:
                        arr = np.asarray(vec, dtype=float)
                    except (TypeError, ValueError):
                        state.invalid[f"invalid_{col}_values"] += 1
                        continue
                    length = int(len(arr)) if arr.ndim == 1 else -1
                    state.vector_lengths[col][length] += 1
                    if expected_dim is not None and length != expected_dim:
                        state.invalid[f"invalid_{col}_shape"] += 1
                    if arr.ndim != 1 or not np.isfinite(arr).all():
                        state.invalid[f"invalid_{col}_values"] += 1
                    elif expected_dim is None or length == expected_dim:
                        state.vector_norms[col].update([float(np.linalg.norm(arr))])

            # Daily entity-time aggregates
            if ids is not None and times is not None and valid_time is not None:
                valid = ids.notna() & valid_time
                if valid.any():
                    data = pd.DataFrame({
                        "entity_id": ids[valid].astype(str),
                        "day": times[valid].dt.strftime("%Y-%m-%d"),
                        "ns": times[valid].astype("int64"),
                    })
                    grouped = data.groupby(["entity_id", "day"], sort=False).agg(
                        event_n=("ns", "size"),
                        first_ns=("ns", "min"),
                        last_ns=("ns", "max"),
                    ).reset_index()
                    self.db.executemany(
                        """INSERT INTO daily VALUES (?,?,?,?,?,?)
                           ON CONFLICT(source, entity_id, day) DO UPDATE SET
                           event_n = event_n + excluded.event_n,
                           first_ns = MIN(first_ns, excluded.first_ns),
                           last_ns = MAX(last_ns, excluded.last_ns)""",
                        (
                            (spec.name, str(r.entity_id), r.day, int(r.event_n), int(r.first_ns), int(r.last_ns))
                            for r in grouped.itertuples(index=False)
                        ),
                    )
            elif ids is not None:
                valid = ids.notna()
                if valid.any():
                    counts = ids[valid].astype(str).value_counts()
                    self.db.executemany(
                        """INSERT INTO daily VALUES (?,?,'__NO_TIME__',?,0,0)
                           ON CONFLICT(source, entity_id, day) DO UPDATE SET
                           event_n = event_n + excluded.event_n""",
                        ((spec.name, str(eid), int(cnt)) for eid, cnt in counts.items()),
                    )

            # Exact duplicate tracking
            if self.config.duplicate_mode == "exact":
                self.db.executemany(
                    """INSERT INTO duplicate_rows VALUES (?,?,1)
                       ON CONFLICT(source, row_key) DO UPDATE SET n = n + 1""",
                    ((spec.name, key) for key in iter_row_keys(frame)),
                )
            self.db.commit()

        state.duplicate_scope = self.config.duplicate_mode
        if self.config.duplicate_mode == "exact":
            state.duplicate_n = int(
                self.db.execute(
                    "SELECT COALESCE(SUM(n - 1), 0) FROM duplicate_rows WHERE source = ?",
                    (spec.name,),
                ).fetchone()[0]
            )
            state.duplicate_rows = state.rows
            state.duplicate_scope = "all_scanned_rows"
        elif self.config.duplicate_mode == "sample":
            sample_frame = state.sample.frame
            keys = list(iter_row_keys(sample_frame))
            state.duplicate_n = len(keys) - len(set(keys))
            state.duplicate_rows = len(keys)
            state.duplicate_scope = "all_scanned_rows" if len(keys) == state.rows else "uniform_row_sample"

        state.duration_seconds = round(perf_counter() - started, 3)
        self._emit(f"  {spec.name}: {state.rows:,} rows scanned in {state.duration_seconds:.2f}s")

    def categories(self, source: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        total = self.stats[source].rows
        cols = [
            r[0]
            for r in self.db.execute(
                "SELECT DISTINCT column_name FROM categories WHERE source = ? ORDER BY column_name",
                (source,),
            )
        ]
        for col in cols:
            top = self.query(
                "SELECT value, n FROM categories WHERE source = ? AND column_name = ? AND kind = 'value' "
                "ORDER BY n DESC, value LIMIT ?",
                (source, col, self.config.max_categories),
            )
            distinct, nonmissing = self.db.execute(
                "SELECT COUNT(*), COALESCE(SUM(n), 0) FROM categories "
                "WHERE source = ? AND column_name = ? AND kind = 'value'",
                (source, col),
            ).fetchone()
            for r in top.itertuples(index=False):
                rows.append(dict(source=source, column=col, kind="value", value=r.value,
                                 n=int(r.n), distinct_n=int(distinct), rows=total))
            missing_row = self.db.execute(
                "SELECT n FROM categories WHERE source = ? AND column_name = ? AND kind = 'missing'",
                (source, col),
            ).fetchone()
            top_sum = int(top["n"].sum()) if not top.empty else 0
            rows.append(dict(source=source, column=col, kind="remaining", value=None,
                             n=int(nonmissing) - top_sum, distinct_n=int(distinct), rows=total))
            rows.append(dict(source=source, column=col, kind="missing", value=None,
                             n=int(missing_row[0]) if missing_row else 0, distinct_n=int(distinct), rows=total))
        frame = pd.DataFrame(rows, columns=["source", "column", "kind", "value", "n", "distinct_n", "rows"])
        frame["rate"] = frame["n"] / frame["rows"].replace(0, np.nan) if not frame.empty else pd.Series(dtype=float)
        return frame

    def entity_aggregates(self) -> pd.DataFrame:
        if "entities" not in self._cache:
            self._cache["entities"] = self.query(
                """SELECT source, entity_id,
                          SUM(event_n) AS event_count,
                          COUNT(*) FILTER (WHERE day <> '__NO_TIME__') AS active_days,
                          MIN(CASE WHEN day <> '__NO_TIME__' THEN first_ns END) AS first_ns,
                          MAX(CASE WHEN day <> '__NO_TIME__' THEN last_ns END) AS last_ns
                   FROM daily GROUP BY source, entity_id"""
            )
        return self._cache["entities"].copy()

    def any_active_days(self) -> pd.DataFrame:
        if "any_days" not in self._cache:
            self._cache["any_days"] = self.query(
                "SELECT entity_id, COUNT(DISTINCT day) AS any_active_days "
                "FROM daily WHERE day <> '__NO_TIME__' GROUP BY entity_id"
            )
        return self._cache["any_days"].copy()
