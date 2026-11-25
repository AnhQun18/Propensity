"""Pure streaming statistics: online Chan/Welford moments, bottom-k sampling, and row hashing."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import json
import math
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class OnlineMoments:
    """Exact batch-parallel mean, variance, min, and max over finite values."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: Any) -> None:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if not len(arr):
            return
        n = len(arr)
        mean = float(arr.mean())
        m2 = float(np.square(arr - mean).sum())
        delta = mean - self.mean
        total = self.n + n
        self.m2 += m2 + delta * delta * self.n * n / total
        self.mean += delta * n / total
        self.n = total
        self.minimum = min(self.minimum, float(arr.min()))
        self.maximum = max(self.maximum, float(arr.max()))

    def summary(self) -> dict[str, Any]:
        return {
            "finite_n": self.n,
            "mean": self.mean if self.n else np.nan,
            "std": math.sqrt(max(self.m2, 0.0) / (self.n - 1)) if self.n > 1 else np.nan,
            "min": self.minimum if self.n else np.nan,
            "max": self.maximum if self.n else np.nan,
        }


class UniformSample:
    """Uniform bottom-k random-priority reservoir sample, invariant to batch boundaries."""

    def __init__(self, size: int, seed: int = 42):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("size must be a positive integer")
        self.size = size
        self.random = np.random.default_rng(seed)
        self.frame = pd.DataFrame()
        self.priorities = np.empty(0, dtype=float)

    def update(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        incoming = self.random.random(len(frame))
        candidates = (
            np.argpartition(incoming, self.size - 1)[: self.size]
            if len(frame) > self.size
            else np.arange(len(frame))
        )
        weights = np.concatenate([self.priorities, incoming[candidates]])
        selected = frame.iloc[candidates]
        combined = (
            pd.concat([self.frame, selected], ignore_index=True)
            if len(self.frame)
            else selected.reset_index(drop=True)
        )
        keep = (
            np.argpartition(weights, self.size - 1)[: self.size]
            if len(weights) > self.size
            else np.arange(len(weights))
        )
        order = keep[np.argsort(weights[keep])]
        self.frame = combined.iloc[order].reset_index(drop=True).copy()
        self.priorities = weights[order]


def canonical_value(value: Any) -> Any:
    """Convert scalars, timestamps, dicts, lists, and numeric arrays into canonical JSON values."""
    if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
        items = value.astype(object)
        if value.dtype.kind == "f":
            items[np.isnan(value)] = None
        return items.tolist()
    if isinstance(value, (list, tuple, np.ndarray)):
        return [canonical_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): canonical_value(v) for k, v in sorted(value.items())}
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def iter_row_keys(frame: pd.DataFrame) -> Iterator[bytes]:
    """Yield canonical UTF-8 JSON bytes per row (column-order normalized) for duplicate detection."""
    if frame.empty:
        return
    ordered = frame[sorted(frame.columns)]
    for row in ordered.itertuples(index=False, name=None):
        yield json.dumps(
            [canonical_value(v) for v in row],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")


def numeric_summary(series: pd.Series, **labels: Any) -> dict[str, Any]:
    """Compute finite count, missing/inf/zero/negative counts, moments, skew/kurtosis, and quantiles."""
    raw = pd.to_numeric(series, errors="coerce")
    non_na = raw.dropna().astype(float)
    infinite_n = int(np.isinf(non_na).sum())
    clean = non_na[np.isfinite(non_na)]
    n_clean = len(clean)
    if n_clean:
        q = clean.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        zeros_n = int((clean == 0).sum())
        neg_n = int((clean < 0).sum())
        stats = {
            "zeros_n": zeros_n,
            "zeros_pct": float(zeros_n / n_clean * 100.0),
            "negative_n": neg_n,
            "mean": float(clean.mean()),
            "std": float(clean.std(ddof=1)) if n_clean > 1 else np.nan,
            "skewness": float(clean.skew()) if n_clean > 2 else np.nan,
            "kurtosis": float(clean.kurt()) if n_clean > 3 else np.nan,
            "min": float(clean.min()),
            "p01": float(q.loc[0.01]),
            "p05": float(q.loc[0.05]),
            "p25": float(q.loc[0.25]),
            "median": float(q.loc[0.5]),
            "p75": float(q.loc[0.75]),
            "iqr": float(q.loc[0.75] - q.loc[0.25]),
            "p95": float(q.loc[0.95]),
            "p99": float(q.loc[0.99]),
            "max": float(clean.max()),
        }
    else:
        stats = {
            "zeros_n": 0,
            "zeros_pct": np.nan,
            "negative_n": 0,
            **{k: np.nan for k in ("mean", "std", "skewness", "kurtosis", "min",
                                    "p01", "p05", "p25", "median", "p75", "iqr", "p95", "p99", "max")},
        }
    return {
        **labels,
        "rows": int(len(series)),
        "finite_n": int(n_clean),
        "missing_n": int(raw.isna().sum()),
        "infinite_n": infinite_n,
        **stats,
    }


def grouped_target_rates(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    target_cols: Sequence[str],
    *,
    entity_col: str | None = None,
) -> pd.DataFrame:
    """Compute positive counts and rates per group for one or more binary 0/1 targets."""
    rows: list[dict[str, Any]] = []
    groups = list(group_cols)
    iterator = (
        [((), frame)]
        if not groups
        else frame.groupby(groups, dropna=False, observed=True, sort=True)
    )
    for key, part in iterator:
        if groups:
            key_tuple = key if isinstance(key, tuple) else (key,)
            labels = dict(zip(groups, key_tuple))
        else:
            labels = {}
        n_entities = int(part[entity_col].nunique()) if entity_col and entity_col in part.columns else None
        for target in target_cols:
            labelled = int(part[target].notna().sum())
            positive = int(part[target].eq(1).sum())
            row = {
                **labels,
                "target": target,
                "rows": int(len(part)),
                "labelled_n": labelled,
                "positive_n": positive,
                "missing_n": int(len(part) - labelled),
                "rate": positive / labelled if labelled else np.nan,
            }
            if n_entities is not None:
                row["entities"] = n_entities
            rows.append(row)
    return pd.DataFrame(rows)
