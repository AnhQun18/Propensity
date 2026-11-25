"""Multi-source temporal overlap, entity-day presence matrices, and >50% absence audit."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._compat import iter_batches, load_dataframe, write_json

DAY_NS = 86_400_000_000_000


def _absence_stats(counts: np.ndarray, denominator: int, cohort: np.ndarray | None = None) -> dict[str, Any]:
    arr = np.asarray(counts)
    if cohort is not None:
        arr = arr[cohort]
    absent_over_half = 2 * arr < denominator
    return {
        "entities": int(len(arr)),
        "opportunities_per_entity": int(denominator),
        "entities_no_events": int((arr == 0).sum()),
        "entities_absent_strictly_over_50_pct": int(absent_over_half.sum()),
        "pct_entities_absent_strictly_over_50_pct": float(absent_over_half.mean() * 100.0) if len(arr) else None,
        "entities_at_least_50_pct_observed": int((~absent_over_half).sum()),
        "observed_count_quantiles": (
            {str(q): float(np.quantile(arr, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)}
            if len(arr) else {}
        ),
    }


def _default_strict_valid(frame: pd.DataFrame) -> np.ndarray:
    valid = np.ones(len(frame), dtype=bool)
    for col in frame.columns:
        valid &= frame[col].notna().to_numpy()
        if pd.api.types.is_float_dtype(frame[col]):
            valid &= np.isfinite(frame[col].to_numpy(dtype=float, na_value=np.nan))
    return valid


def audit_temporal_coverage(
    entities_source: Any,
    event_sources: Mapping[str, Any],
    *,
    entity_col: str,
    time_col: str,
    cutoff: str | pd.Timestamp | None = None,
    strict_validators: Mapping[str, Callable[[pd.DataFrame], np.ndarray]] | None = None,
    batch_size: int = 250_000,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Measure shared temporal overlap across M sources and entity-day >50% absence rates.

    Absence of recorded events does not distinguish genuine inactivity from missing collection.
    """
    if not event_sources:
        raise ValueError("Provide at least one event source")

    ent_df = load_dataframe(entities_source, columns=[entity_col])
    if ent_df[entity_col].isna().any() or ent_df[entity_col].duplicated().any():
        raise ValueError(f"Entity table has missing or duplicate keys in {entity_col!r}")
    ids = pd.Index(ent_df[entity_col].astype(str))
    n_entities = len(ids)
    mods = list(event_sources.keys())
    validators = strict_validators or {}

    # Pass 1: find global time bounds per source
    bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for mod, src in event_sources.items():
        lo = hi = None
        for batch in iter_batches(src, columns=[time_col], batch_size=batch_size):
            ts = pd.to_datetime(batch[time_col], errors="coerce", format="mixed").dropna()
            if ts.empty:
                continue
            b_lo, b_hi = ts.min(), ts.max()
            lo = b_lo if lo is None else min(lo, b_lo)
            hi = b_hi if hi is None else max(hi, b_hi)
        if lo is None or hi is None:
            raise ValueError(f"Source {mod!r} has no valid timestamps in {time_col!r}")
        bounds[mod] = (pd.Timestamp(lo), pd.Timestamp(hi))

    shared_lo = max(b[0] for b in bounds.values())
    shared_hi = min(b[1] for b in bounds.values())
    cutoff_ts = pd.Timestamp(cutoff) if cutoff is not None else (shared_hi.floor("D") + pd.Timedelta(days=1))

    source_start = {m: bounds[m][0].floor("D") for m in mods}
    baseline = min(source_start.values())
    common_start = max(source_start.values())
    total_days = max(1, (cutoff_ts - baseline).days)
    common_days = max(1, (cutoff_ts - common_start).days)
    offset_common = max(0, (common_start - baseline).days)

    raw_daily = {m: np.zeros((n_entities, total_days), dtype=bool) for m in mods}
    strict_daily = {m: np.zeros((n_entities, total_days), dtype=bool) for m in mods}
    raw_counts = {m: np.zeros(n_entities, dtype=np.int64) for m in mods}
    strict_counts = {m: np.zeros(n_entities, dtype=np.int64) for m in mods}
    row_quality: dict[str, dict[str, Any]] = {}

    base_day = baseline.value // DAY_NS
    cutoff_ns = cutoff_ts.value
    common_start_ns = common_start.value

    for mod, src in event_sources.items():
        row_total = invalid_total = 0
        validator = validators.get(mod, _default_strict_valid)
        mod_start_ns = source_start[mod].value
        for batch in iter_batches(src, batch_size=batch_size):
            row_total += len(batch)
            pos = ids.get_indexer(batch[entity_col].astype(str))
            ts = pd.to_datetime(batch[time_col], errors="coerce", format="mixed")
            valid_time = ts.notna().to_numpy()
            ts_ns = ts.dt.as_unit("ns").astype("int64").to_numpy() if valid_time.any() else np.zeros(len(batch), dtype=np.int64)
            raw_ok = (pos >= 0) & valid_time
            strict_ok = raw_ok & np.asarray(validator(batch), dtype=bool)
            invalid_total += int((~strict_ok).sum())

            in_hist = (ts_ns >= mod_start_ns) & (ts_ns < cutoff_ns)
            in_common = (ts_ns >= common_start_ns) & (ts_ns < cutoff_ns)
            for ok, daily_mat, cnt_vec in (
                (raw_ok, raw_daily[mod], raw_counts[mod]),
                (strict_ok, strict_daily[mod], strict_counts[mod]),
            ):
                sel = ok & in_hist
                if sel.any():
                    day_idx = (ts_ns[sel] // DAY_NS) - base_day
                    daily_mat[pos[sel], day_idx] = True
                sel_c = ok & in_common
                if sel_c.any():
                    np.add.at(cnt_vec, pos[sel_c], 1)

        row_quality[mod] = {
            "all_rows_checked": row_total,
            "strict_invalid_rows": invalid_total,
            "first_valid_time": str(bounds[mod][0]),
            "last_valid_time": str(bounds[mod][1]),
            "raw_rows_in_common_history": int(raw_counts[mod].sum()),
            "strict_rows_in_common_history": int(strict_counts[mod].sum()),
        }

    result: dict[str, Any] = {
        "reference_entities": n_entities,
        "sources": mods,
        "cutoff_exclusive": str(cutoff_ts),
        "shared_interval": {
            "start_inclusive": str(shared_lo),
            "end_inclusive": str(shared_hi),
            "overlap_positive": bool(shared_hi >= shared_lo),
        },
        "common_history_start_inclusive": str(common_start),
        "common_history_days": common_days,
        "row_quality": row_quality,
    }

    for mode_name, daily in (("raw_id_time", raw_daily), ("strict_event_quality", strict_daily)):
        common = {m: daily[m][:, offset_common:] for m in mods}
        combined = np.logical_or.reduce(list(common.values()))
        presence = np.stack([common[m].any(axis=1) for m in mods], axis=1)
        any_ent = presence.any(axis=1)
        all_ent = presence.all(axis=1)
        active = combined.sum(axis=1)
        source_day_cells = sum(x.sum(axis=1) for x in common.values())
        source_count = presence.sum(axis=1)

        first = combined.argmax(axis=1)
        last = common_days - 1 - combined[:, ::-1].argmax(axis=1)
        span = np.where(any_ent, last - first + 1, 0)

        result[mode_name] = {
            "entities_any_source": int(any_ent.sum()),
            "entities_all_sources": int(all_ent.sum()),
            "any_source_active_days_all_entities": _absence_stats(active, common_days),
            "any_source_active_days_active_entities": _absence_stats(active, common_days, any_ent),
            "per_source_active_days": {m: _absence_stats(common[m].sum(axis=1), common_days) for m in mods},
            "source_day_cells_all_entities": _absence_stats(source_day_cells, len(mods) * common_days),
            "sources_available_all_entities": _absence_stats(source_count, len(mods)),
            "any_source_calendar_span": _absence_stats(span, common_days),
        }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "temporal_coverage.json", result)

    return result
