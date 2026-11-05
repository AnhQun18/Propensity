"""Thống kê thuần trên bảng tổng hợp; không sửa dữ liệu nguồn."""

from dataclasses import dataclass
import json
import math

import numpy as np
import pandas as pd


@dataclass
class Moments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        n, mean = len(values), float(values.mean())
        m2 = float(np.square(values - mean).sum())
        delta, total = mean - self.mean, self.n + n
        self.m2 += m2 + delta * delta * self.n * n / total
        self.mean += delta * n / total
        self.n = total
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    def summary(self):
        return {"finite_n": self.n, "mean": self.mean if self.n else np.nan,
                "std": math.sqrt(max(self.m2, 0) / (self.n - 1)) if self.n > 1 else np.nan,
                "min": self.minimum if self.n else np.nan, "max": self.maximum if self.n else np.nan}


class UniformSample:
    """Uniform bottom-k random-priority sampling, invariant to batch boundaries."""

    def __init__(self, size, seed):
        self.size, self.random = size, np.random.default_rng(seed)
        self.frame = pd.DataFrame()
        self.priorities = np.empty(0)

    def update(self, frame):
        weights = np.concatenate([self.priorities, self.random.random(len(frame))])
        combined = pd.concat([self.frame, frame], ignore_index=True) if len(self.frame) else frame.reset_index(drop=True)
        keep = np.argpartition(weights, self.size - 1)[:self.size] if len(weights) > self.size else np.arange(len(weights))
        order = keep[np.argsort(weights[keep])]
        self.frame = combined.iloc[order].reset_index(drop=True).copy()
        self.priorities = weights[order]


def canonical(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def row_keys(frame):
    for row in frame[sorted(frame)].itertuples(index=False, name=None):
        yield json.dumps([canonical(v) for v in row], ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def numeric_summary(series, **labels):
    raw = pd.to_numeric(series, errors="coerce")
    clean = raw.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    q = clean.quantile([.25, .5, .75, .95, .99])
    return {**labels, "rows": len(series), "finite_n": len(clean), "missing_n": int(raw.isna().sum()),
            "infinite_n": int(np.isinf(raw.dropna().astype(float)).sum()), "mean": clean.mean(),
            "std": clean.std(ddof=1), "min": clean.min(), "p25": q.loc[.25],
            "median": q.loc[.5], "p75": q.loc[.75], "p95": q.loc[.95], "p99": q.loc[.99],
            "max": clean.max()}


def numeric_columns(frame):
    excluded = {"fold", "cutoff", "label_start", "label_end", "mon", "client_id"}
    return [c for c in frame.select_dtypes(include=["number", "bool"]) if c not in excluded and not c.startswith("target_")]


def target_rates(frame, groups, targets, *, window=None):
    rows = []
    for group, part in frame.groupby(groups, dropna=False, observed=True, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        keys = dict(zip(groups, group))
        for target in targets:
            n, positive = int(part[target].notna().sum()), int(part[target].eq(1).sum())
            rows.append({**keys, "target": target, "rows": len(part), "customers": part.client_id.nunique(),
                         "labelled_n": n, "positive_n": positive, "missing_n": len(part) - n,
                         "rate": positive / n if n else np.nan, **({"history_days": window} if window is not None else {})})
    return pd.DataFrame(rows)


def result(context, key, tables, findings=(), **metadata):
    from .models import SectionResult
    scope = "prefix" if context.config.max_batches_per_source is not None else "full"
    return SectionResult(key, tables, list(findings), {"event_scope": scope,
        "reference_scope": "full", "timezone": context.config.timezone or "naive_as_stored", **metadata})
