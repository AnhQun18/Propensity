"""Audit local MBD-mini time alignment; write aggregate outputs only.

No source data is changed. An event span is not continuous observation coverage.
Windows use stored naive calendar dates, without inventing a timezone.
"""
from collections import Counter
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "MBD-mini"
OUT = ROOT / "docs" / "mbd_temporal_coverage"
MODS = ("trx", "geo", "dialog")
DAY_NS = 86_400_000_000_000
MAX_I64 = np.iinfo(np.int64).max
MIN_I64 = np.iinfo(np.int64).min


def integer(array, null_value=0):
    return pc.fill_null(pc.cast(array, pa.int64()), null_value).to_numpy(zero_copy_only=False)


def stamp(ns):
    return str(pd.Timestamp(int(ns), unit="ns"))


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    clients = pq.read_table(BASE / "client_split").to_pandas()
    if clients.client_id.isna().any() or clients.client_id.duplicated().any():
        raise ValueError("Expected non-null unique client_split IDs")
    ids = pa.array(clients.client_id)
    n = len(clients)
    targets = pq.read_table(BASE / "targets").to_pandas()
    label_cols = [f"target_{i}" for i in range(1, 5)]
    target_valid = targets[label_cols].isin([0, 1]).all(axis=1) & targets.client_id.notna() & targets.mon.notna()
    result = {
        "date": "2026-11-05", "initial_clients": n,
        "target_rows": len(targets), "target_unique_clients": targets.client_id.nunique(),
        "target_null_cells": targets.isna().sum().to_dict(),
        "target_rows_valid_id_month_four_binary_labels": int(target_valid.sum()),
        "target_duplicate_client_month_rows": int(targets.duplicated(["client_id", "mon"]).sum()),
        "target_rows_complete_all_columns": int(targets.notna().all(axis=1).sum()),
        "target_clients_with_any_all_columns_complete_row": int(targets.loc[targets.notna().all(axis=1), "client_id"].nunique()),
        "definitions": {
            "valid_event": "ID in client_split, valid timestamp, all physical event columns non-null; finite amount; dialog embedding exactly 768 finite, non-null float values.",
            "shared_event_interval": "max(global first valid event per modality) through min(global last valid event per modality), inclusive.",
            "full_calendar_interval": "Whole stored calendar days strictly inside the shared interval; final exclusive midnight is floor(shared end).",
            "client_common_span": "max(first valid event per modality) to min(last valid event per modality); endpoint envelope, not proof of continuous logging.",
            "window_presence": "At least one valid event in every required modality in [cutoff-history_days, cutoff), with the calendar interval entirely available globally.",
            "window_established_history": "window_presence plus each required modality has a first valid event on a calendar day before or equal to the history start; not proof of daily observation.",
        },
    }
    files_by_mod = {m: sorted((BASE / "detail" / m).rglob("*.parquet")) for m in MODS}
    global_raw = {}
    for mod in MODS:
        lo, hi = MAX_I64, MIN_I64
        for path in files_by_mod[mod]:
            a = pq.ParquetFile(path).read(columns=["event_time"])["event_time"]
            mn, mx = pc.min(a).as_py(), pc.max(a).as_py()
            if mn is not None:
                lo, hi = min(lo, pd.Timestamp(mn).value), max(hi, pd.Timestamp(mx).value)
        global_raw[mod] = (lo, hi)
        print(f"Raw time bounds {mod}: {stamp(lo)} -> {stamp(hi)}", flush=True)
    shared_lo = max(x[0] for x in global_raw.values())
    shared_hi = min(x[1] for x in global_raw.values())
    calendar_lo = ((shared_lo + DAY_NS - 1) // DAY_NS) * DAY_NS
    calendar_hi = (shared_hi // DAY_NS) * DAY_NS
    first_day = min(x[0] for x in global_raw.values()) // DAY_NS
    last_day = max(x[1] for x in global_raw.values()) // DAY_NS
    days = int(last_day - first_day + 1)
    result["shared_interval"] = {
        "start_inclusive": stamp(shared_lo), "end_inclusive": stamp(shared_hi),
        "elapsed_days": (shared_hi - shared_lo) / DAY_NS,
        "full_calendar_start": stamp(calendar_lo), "full_calendar_end_exclusive": stamp(calendar_hi),
        "full_calendar_days": int((calendar_hi - calendar_lo) // DAY_NS),
    }
    state = {}
    file_rows = []
    for mod in MODS:
        s = {"raw_count": np.zeros(n, dtype=np.int64),
             "raw_first": np.full(n, MAX_I64, dtype=np.int64),
             "raw_last": np.full(n, MIN_I64, dtype=np.int64),
             "raw_shared_count": np.zeros(n, dtype=np.int64),
             "count": np.zeros(n, dtype=np.int64),
             "first": np.full(n, MAX_I64, dtype=np.int64),
             "last": np.full(n, MIN_I64, dtype=np.int64),
             "shared_count": np.zeros(n, dtype=np.int64),
             "calendar_count": np.zeros(n, dtype=np.int64),
             "daily": np.zeros((n, days), dtype=bool)}
        totals, nulls = Counter(), Counter()
        unknown_ids = set()
        for file_index, path in enumerate(files_by_mod[mod], 1):
            table = pq.ParquetFile(path).read().combine_chunks()
            length = len(table)
            if not length:
                continue
            index = integer(pc.index_in(table["client_id"], value_set=ids), -1)
            times = integer(table["event_time"])
            key_valid = (index >= 0) & pc.is_valid(table["event_time"]).to_numpy()
            null_id = table["client_id"].null_count
            if np.any(index < 0):
                unknown_ids.update(pc.unique(pc.filter(table["client_id"], pa.array(index < 0))).to_pylist())
            totals["rows"] += length
            totals["missing_id"] += null_id
            totals["unknown_or_missing_id_rows"] += int((index < 0).sum())
            totals["missing_time"] += table["event_time"].null_count
            for prefix, valid in [("raw_", key_valid)]:
                ix, ts = index[valid], times[valid]
                np.add.at(s[prefix + "count"], ix, 1)
                np.minimum.at(s[prefix + "first"], ix, ts)
                np.maximum.at(s[prefix + "last"], ix, ts)
                raw_inside = (ts >= shared_lo) & (ts <= shared_hi)
                np.add.at(s["raw_shared_count"], ix[raw_inside], 1)
            valid = key_valid.copy()
            for col in table.column_names:
                nulls[col] += table[col].null_count
                if table[col].null_count:
                    valid &= pc.is_valid(table[col]).to_numpy()
            if mod == "trx":
                bad_amount = ~np.isfinite(table["amount"].to_numpy())
                totals["nonfinite_or_null_amount_rows"] += int(bad_amount.sum())
                valid &= ~bad_amount
            if mod == "dialog":
                vectors = table["embedding"].combine_chunks()
                lens = integer(pc.list_value_length(vectors), -1)
                wrong_length = lens != 768
                totals["wrong_or_null_embedding_length_rows"] += int(wrong_length.sum())
                flat = pc.list_flatten(vectors)
                bad_value = ~np.isfinite(flat.to_numpy(zero_copy_only=False))
                bad_vector = np.zeros(length, dtype=bool)
                if np.any(bad_value):
                    parents = integer(pc.list_parent_indices(vectors))
                    bad_vector[parents[bad_value]] = True
                totals["embedding_with_nonfinite_or_null_element_rows"] += int(bad_vector.sum())
                valid &= ~wrong_length & ~bad_vector
            ix, ts = index[valid], times[valid]
            np.add.at(s["count"], ix, 1)
            np.minimum.at(s["first"], ix, ts)
            np.maximum.at(s["last"], ix, ts)
            day_index = ts // DAY_NS - first_day
            s["daily"][ix, day_index] = True
            inside = (ts >= shared_lo) & (ts <= shared_hi)
            whole_days = (ts >= calendar_lo) & (ts < calendar_hi)
            np.add.at(s["shared_count"], ix[inside], 1)
            np.add.at(s["calendar_count"], ix[whole_days], 1)
            totals["valid_rows"] += len(ix)
            totals["invalid_rows"] += int((~valid).sum())
            totals["valid_rows_in_shared_interval"] += int(inside.sum())
            totals["valid_rows_before_shared_interval"] += int((ts < shared_lo).sum())
            totals["valid_rows_after_shared_interval"] += int((ts > shared_hi).sum())
            totals["valid_rows_in_full_calendar_interval"] += int(whole_days.sum())
            has_valid_time = times[pc.is_valid(table["event_time"]).to_numpy()]
            file_rows.append({"modality": mod, "file": str(path.relative_to(BASE)),
                              "rows": length, "valid_rows": int(valid.sum()),
                              "min_time": stamp(has_valid_time.min()) if len(has_valid_time) else None,
                              "max_time": stamp(has_valid_time.max()) if len(has_valid_time) else None,
                              "unique_known_clients": int(len(np.unique(index[index >= 0]))),
                              "valid_rows_in_shared_interval": int(inside.sum())})
            if file_index % 50 == 0:
                print(f"{mod}: checked {file_index}/{len(files_by_mod[mod])} files", flush=True)
        present = s["count"] > 0
        valid_lo, valid_hi = s["first"][present].min(), s["last"][present].max()
        if (valid_lo, valid_hi) != global_raw[mod]:
            raise ValueError(f"Quality filtering changed time endpoints for {mod}; recompute global intersection")
        result.setdefault("modalities", {})[mod] = {
            **dict(totals), "files": len(files_by_mod[mod]), "null_cells": dict(nulls),
            "unknown_unique_nonnull_clients": len(unknown_ids - {None}),
            "min_valid_time": stamp(valid_lo), "max_valid_time": stamp(valid_hi),
            "raw_unique_known_clients": int((s["raw_count"] > 0).sum()),
            "raw_unique_clients_shared_interval": int((s["raw_shared_count"] > 0).sum()),
            "valid_unique_clients": int(present.sum()),
            "valid_unique_clients_shared_interval": int((s["shared_count"] > 0).sum()),
            "valid_unique_clients_full_calendar_interval": int((s["calendar_count"] > 0).sum()),
        }
        state[mod] = s
        print(f"{mod}: {totals['rows']:,} rows, {totals['invalid_rows']:,} invalid, {present.sum():,} clients", flush=True)
    pd.DataFrame(file_rows).to_csv(OUT / "file_time_ranges.csv", index=False, encoding="utf-8-sig")

    all_anytime = np.logical_and.reduce([state[m]["count"] > 0 for m in MODS])
    all_shared = np.logical_and.reduce([state[m]["shared_count"] > 0 for m in MODS])
    all_calendar = np.logical_and.reduce([state[m]["calendar_count"] > 0 for m in MODS])
    pair_trx_geo = (state["trx"]["shared_count"] > 0) & (state["geo"]["shared_count"] > 0)
    first = np.maximum.reduce([state[m]["first"] for m in MODS])
    last = np.minimum.reduce([state[m]["last"] for m in MODS])
    # Compute subtraction only for IDs present in all modalities to avoid sentinel overflow.
    overlap = all_anytime.copy()
    overlap[all_anytime] = last[all_anytime] >= first[all_anytime]
    span_days = np.zeros(n, dtype=np.float64)
    span_days[overlap] = (last[overlap] - first[overlap]) / DAY_NS
    filtering = {
        "initial_clients": n,
        "all_three_modalities_raw_id_time_any_time": int(np.logical_and.reduce([state[m]["raw_count"] > 0 for m in MODS]).sum()),
        "all_three_modalities_raw_id_time_shared_interval": int(np.logical_and.reduce([state[m]["raw_shared_count"] > 0 for m in MODS]).sum()),
        "at_least_one_valid_event_any_modality": int(np.logical_or.reduce([state[m]["count"] > 0 for m in MODS]).sum()),
        "all_three_modalities_any_time": int(all_anytime.sum()),
        "all_three_modalities_in_shared_interval": int(all_shared.sum()),
        "all_three_modalities_in_full_calendar_interval": int(all_calendar.sum()),
        "trx_and_geo_in_shared_interval": int(pair_trx_geo.sum()),
        "all_three_with_nonnegative_common_event_span": int(overlap.sum()),
        "all_three_with_positive_common_event_span": int((overlap & (span_days > 0)).sum()),
        "all_three_common_span_at_least_days": {str(d): int((overlap & (span_days >= d)).sum()) for d in (7, 30, 84, 180, 365)},
        "common_span_days_quantiles_for_overlapping_clients": {str(q): float(np.quantile(span_days[overlap], q)) for q in (0, .25, .5, .75, .9, 1)},
        "kept_event_rows_shared_interval_for_all_three_clients": {m: int(state[m]["shared_count"][all_shared].sum()) for m in MODS},
    }
    filtering["removed_clients_missing_any_modality_any_time"] = n - filtering["all_three_modalities_any_time"]
    filtering["additional_clients_lost_after_shared_time_clip"] = int(all_anytime.sum() - all_shared.sum())
    result["client_filtering"] = filtering
    patterns = []
    for code in range(8):
        mask = np.ones(n, dtype=bool)
        names = []
        for j, m in enumerate(MODS):
            available = state[m]["shared_count"] > 0
            flag = bool(code & (1 << j))
            mask &= available if flag else ~available
            if flag:
                names.append(m)
        patterns.append({"available_modalities": "+".join(names) or "none", "clients": int(mask.sum())})
    result["shared_interval_modality_patterns"] = patterns
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=int) + "\n", encoding="utf-8")

    # PTLS is a representation of the same sources, not an independent modality.
    # Compare per-client event counts and first/last timestamps at second precision.
    for m in MODS:
        c = np.zeros(n, dtype=np.int64)
        lo = np.full(n, MAX_I64, dtype=np.int64)
        hi = np.full(n, MIN_I64, dtype=np.int64)
        sequence_count = np.zeros(n, dtype=np.int32)
        total_rows = bad_time_rows = unknown_rows = 0
        for path in sorted((BASE / "ptls" / m).rglob("*.parquet")):
            t = pq.ParquetFile(path).read(columns=["client_id", "event_time"]).combine_chunks()
            idx = integer(pc.index_in(t["client_id"], value_set=ids), -1)
            arr = t["event_time"].combine_chunks()
            lengths = integer(pc.list_value_length(arr))
            values = pc.list_flatten(arr)
            parent = integer(pc.list_parent_indices(arr))
            event_idx = idx[parent]
            valid = (event_idx >= 0) & pc.is_valid(values).to_numpy(zero_copy_only=False)
            times = integer(values)
            np.add.at(c, event_idx[valid], 1)
            np.minimum.at(lo, event_idx[valid], times[valid])
            np.maximum.at(hi, event_idx[valid], times[valid])
            np.add.at(sequence_count, idx[idx >= 0], 1)
            total_rows += len(t)
            bad_time_rows += arr.null_count + values.null_count
            unknown_rows += int((idx < 0).sum())
        raw = state[m]
        both = (c > 0) & (raw["raw_count"] > 0)
        result.setdefault("ptls_detail_comparison", {})[m] = {
            "ptls_sequence_rows": total_rows, "ptls_unique_clients": int((c > 0).sum()),
            "ptls_total_events": int(c.sum()), "detail_total_known_id_nonnull_time_events": int(raw["raw_count"].sum()),
            "null_time_arrays_plus_null_elements": bad_time_rows, "unknown_id_sequence_rows": unknown_rows,
            "duplicate_sequence_client_ids": int((sequence_count > 1).sum()),
            "client_presence_mismatches": int(((c > 0) != (raw["raw_count"] > 0)).sum()),
            "per_client_event_count_mismatches": int((c != raw["raw_count"]).sum()),
            "first_timestamp_second_precision_mismatches": int((lo[both] != raw["raw_first"][both] // 1_000_000_000).sum()),
            "last_timestamp_second_precision_mismatches": int((hi[both] != raw["raw_last"][both] // 1_000_000_000).sum()),
        }
        print(f"PTLS/detail {m}: {int(c.sum()):,} events compared by ID, count and endpoints", flush=True)

    # Calendar windows do not assume that target.mon is the true purchase month.
    prefixes = {}
    for m in MODS:
        prefixes[m] = np.concatenate([np.zeros((n, 1), dtype=np.uint16), np.cumsum(state[m]["daily"], axis=1, dtype=np.uint16)], axis=1)
        del state[m]["daily"]
    cutoff_days = np.arange(calendar_lo // DAY_NS + 1, calendar_hi // DAY_NS + 1, dtype=np.int64)
    candidates = []
    target_month_period = pd.to_datetime(targets.mon).dt.to_period("M")
    targets_by_period = {period: group for period, group in targets.loc[target_valid].groupby(target_month_period[target_valid])}
    for hist in (7, 30, 84, 180):
        per_cutoff = []
        union_presence = np.zeros(n, dtype=bool)
        union_established = np.zeros(n, dtype=bool)
        union_monthly = np.zeros(n, dtype=bool)
        intersection_monthly = np.ones(n, dtype=bool)
        monthly_count = 0
        best_span_count, best_span_start = -1, None
        for cutoff_day in cutoff_days:
            start_day = cutoff_day - hist
            if start_day < calendar_lo // DAY_NS:
                continue
            start_ix, end_ix = int(start_day - first_day), int(cutoff_day - first_day)
            active = {m: prefixes[m][:, end_ix] > prefixes[m][:, start_ix] for m in MODS}
            presence = np.logical_and.reduce(list(active.values()))
            established = presence & np.logical_and.reduce([state[m]["first"] // DAY_NS <= start_day for m in MODS])
            covered_span = overlap.copy()
            covered_span[overlap] = ((first[overlap] <= start_day * DAY_NS) & (last[overlap] >= cutoff_day * DAY_NS))
            covered_and_active = covered_span & presence
            union_presence |= presence
            union_established |= established
            cutoff = pd.Timestamp(int(cutoff_day * DAY_NS))
            rec = {
                "history_days": hist, "history_start": str(pd.Timestamp(int(start_day * DAY_NS)).date()),
                "cutoff_exclusive": str(cutoff.date()), "trx_clients": int(active["trx"].sum()),
                "trx_geo_clients": int((active["trx"] & active["geo"]).sum()),
                "all_three_presence_clients": int(presence.sum()),
                "all_three_established_history_clients": int(established.sum()),
                "all_three_endpoint_span_covers_window_and_active_clients": int(covered_and_active.sum()),
            }
            per_cutoff.append(rec)
            if covered_and_active.sum() > best_span_count:
                best_span_count, best_span_start = int(covered_and_active.sum()), rec["history_start"]
            if cutoff.day == 1:
                union_monthly |= presence
                intersection_monthly &= presence
                monthly_count += 1
                # Counts only for target marker in the same calendar month; not asserted label semantics.
                month_targets = targets_by_period.get(cutoff.to_period("M"), targets.iloc[:0])
                ix = integer(pc.index_in(pa.array(month_targets.client_id), value_set=ids), -1)
                selected = (ix >= 0)
                selected[selected] &= presence[ix[selected]]
                filtered = month_targets.loc[selected]
                for col in label_cols:
                    rec[col + "_positives_same_month_marker"] = int(filtered[col].eq(1).sum())
                candidates.append(rec)
        pd.DataFrame(per_cutoff).to_csv(OUT / f"daily_windows_{hist}d.csv", index=False, encoding="utf-8-sig")
        result.setdefault("windows", {})[str(hist)] = {
            "calendar_window_count": len(per_cutoff),
            "first_history_start": per_cutoff[0]["history_start"] if per_cutoff else None,
            "first_cutoff_exclusive": per_cutoff[0]["cutoff_exclusive"] if per_cutoff else None,
            "last_cutoff_exclusive": per_cutoff[-1]["cutoff_exclusive"] if per_cutoff else None,
            "distinct_clients_any_daily_window_presence": int(union_presence.sum()),
            "distinct_clients_any_daily_window_established_history": int(union_established.sum()),
            "best_single_window_presence": max(per_cutoff, key=lambda r: r["all_three_presence_clients"]) if per_cutoff else None,
            "best_single_window_endpoint_span_and_activity": {"history_start": best_span_start, "clients": best_span_count},
            "month_boundary_window_count": monthly_count,
            "distinct_clients_any_month_boundary_window_presence": int(union_monthly.sum()),
            "clients_present_all_month_boundary_windows": int(intersection_monthly.sum()) if monthly_count else 0,
        }
        print(f"{hist}d windows: {len(per_cutoff)}, any-window clients {union_presence.sum():,}", flush=True)
    pd.DataFrame(candidates).to_csv(OUT / "monthly_windows.csv", index=False, encoding="utf-8-sig")
    result["limitations"] = [
        "Timestamps have no timezone; calendar trimming is based on the stored naive timestamps.",
        "Missing events cannot distinguish no activity from incomplete collection; no continuous-observation flag is supplied.",
        "Per-client min/max spans do not guarantee daily events or continuous observation.",
        "Selecting a cohort using its last future event is descriptive only and can bias predictive evaluation.",
        "Daily and monthly window counts describe available feature history, not verified target.mon-to-label alignment.",
        "PTLS comparison checks per-client counts and endpoints at second precision, not every event value or array alignment.",
    ]
    result["elapsed_seconds"] = time.perf_counter() - started
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=int) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("definitions", "target_null_cells")}, ensure_ascii=False, indent=2, default=int), flush=True)


if __name__ == "__main__":
    main()
