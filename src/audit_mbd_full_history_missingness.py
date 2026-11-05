"""Measure >50% absent event-days before a fixed cutoff; aggregate output only.

Absence of a recorded event is NOT proof that data collection is missing.
The cutoff is kept at 2022-12-01; December never enters the inputs.
"""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from audit_mbd_temporal_coverage import integer, DAY_NS

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "MBD-mini"
OUT = ROOT / "docs" / "mbd_temporal_coverage"
MODS = ("trx", "geo", "dialog")
CUTOFF = pd.Timestamp("2022-12-01")


def stats(counts, denominator, cohort=None):
    counts = np.asarray(counts)
    if cohort is not None:
        counts = counts[cohort]
    absent_over_half = 2 * counts < denominator
    return {
        "clients": len(counts), "opportunities_per_client": int(denominator),
        "clients_no_events": int((counts == 0).sum()),
        "clients_absent_strictly_over_50_percent": int(absent_over_half.sum()),
        "percent_clients_absent_strictly_over_50_percent": float(absent_over_half.mean() * 100) if len(counts) else None,
        "clients_at_least_50_percent_observed": int((~absent_over_half).sum()),
        "observed_count_quantiles": {str(q): float(np.quantile(counts, q)) for q in (0, .25, .5, .75, .9, .95, 1)} if len(counts) else {},
    }


def valid_rows(table, mod, known_ids):
    positions = integer(pc.index_in(table["client_id"], value_set=known_ids), -1)
    times = integer(table["event_time"])
    raw = (positions >= 0) & pc.is_valid(table["event_time"]).to_numpy()
    valid = raw.copy()
    for name in table.column_names:
        if table[name].null_count:
            valid &= pc.is_valid(table[name]).to_numpy()
    if mod == "trx":
        valid &= np.isfinite(table["amount"].to_numpy())
    elif mod == "dialog":
        arr = table["embedding"].combine_chunks()
        valid &= integer(pc.list_value_length(arr), -1) == 768
        flat = pc.list_flatten(arr)
        bad = ~np.isfinite(flat.to_numpy(zero_copy_only=False))
        if bad.any():
            parents = integer(pc.list_parent_indices(arr))
            invalid_vectors = np.zeros(len(table), dtype=bool)
            invalid_vectors[parents[bad]] = True
            valid &= ~invalid_vectors
    return positions, times, raw, valid


def main():
    started = time.perf_counter()
    previous = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    client_table = pq.read_table(BASE / "client_split", columns=["client_id"])
    ids = client_table["client_id"].combine_chunks()
    n = len(ids)
    source_start = {m: pd.Timestamp(previous["modalities"][m]["min_valid_time"]).ceil("D") for m in MODS}
    baseline = min(source_start.values())
    common_start = max(source_start.values())
    days = (CUTOFF - baseline).days
    common_days = (CUTOFF - common_start).days
    offset_common = (common_start - baseline).days
    raw_daily = {m: np.zeros((n, days), dtype=bool) for m in MODS}
    clean_daily = {m: np.zeros((n, days), dtype=bool) for m in MODS}
    raw_counts = {m: np.zeros(n, dtype=np.int64) for m in MODS}
    clean_counts = {m: np.zeros(n, dtype=np.int64) for m in MODS}
    row_quality = {}
    for mod in MODS:
        files = sorted((BASE / "detail" / mod).rglob("*.parquet"))
        row_total = invalid_total = 0
        for file_number, path in enumerate(files, 1):
            t = pq.ParquetFile(path).read().combine_chunks()
            pos, times, raw, valid = valid_rows(t, mod, ids)
            row_total += len(t)
            invalid_total += int((~valid).sum())
            in_history = (times >= source_start[mod].value) & (times < CUTOFF.value)
            in_common = (times >= common_start.value) & (times < CUTOFF.value)
            for good, flags, counts in ((raw, raw_daily[mod], raw_counts[mod]),
                                        (valid, clean_daily[mod], clean_counts[mod])):
                selected = good & in_history
                flags[pos[selected], (times[selected] // DAY_NS - baseline.value // DAY_NS)] = True
                selected_common = good & in_common
                np.add.at(counts, pos[selected_common], 1)
            if file_number % 100 == 0:
                print(f"{mod}: {file_number}/{len(files)} files", flush=True)
        # Independent checks against the preceding full-file audit.
        assert row_total == previous["modalities"][mod]["rows"]
        assert invalid_total == previous["modalities"][mod]["invalid_rows"]
        assert np.all(clean_daily[mod] <= raw_daily[mod])
        row_quality[mod] = {"files": len(files), "all_source_rows_checked": row_total,
                            "invalid_source_rows": invalid_total,
                            "raw_rows_in_common_history": int(raw_counts[mod].sum()),
                            "valid_rows_in_common_history": int(clean_counts[mod].sum())}
        print(f"{mod}: complete", flush=True)
    result = {
        "date": "2026-11-05", "initial_clients": n,
        "cutoff_exclusive": str(CUTOFF),
        "common_history_start_inclusive": str(common_start),
        "common_history_days": common_days,
        "full_source_history_starts": {m: str(x) for m, x in source_start.items()},
        "full_source_history_days": {m: (CUTOFF - x).days for m, x in source_start.items()},
        "definitions": {
            "absent_event_day": "No retained event for an ID on the stored naive calendar date; does not distinguish no activity from missing collection.",
            "strictly_over_50_percent": "2 * observed_opportunities < total_opportunities; exactly 50% is not included.",
            "common_day_any_source": "A day is observed if at least one valid event occurs in any of trx/geo/dialog.",
            "common_source_day_cells": "354 days times 3 sources; a source-day is observed if that source has at least one valid event.",
            "common_event_envelope": "Inclusive calendar span from first to last event of a client; not continuous coverage.",
            "sources_missing_over_half": "No events in at least 2 out of 3 sources anywhere in the history.",
            "strict_event_quality": "Same all-column non-null and finite-value/768-element embedding rule as summary.json.",
        },
        "row_quality": row_quality,
    }
    for mode, daily in (("raw_id_time", raw_daily), ("strict_event_quality", clean_daily)):
        common = {m: daily[m][:, offset_common:] for m in MODS}
        combined = np.logical_or.reduce(list(common.values()))
        source_presence = np.stack([common[m].any(axis=1) for m in MODS], axis=1)
        any_client = source_presence.any(axis=1)
        all_client = source_presence.all(axis=1)
        active = combined.sum(axis=1)
        source_days = sum(x.sum(axis=1) for x in common.values())
        source_count = source_presence.sum(axis=1)
        first = combined.argmax(axis=1)
        last = common_days - 1 - combined[:, ::-1].argmax(axis=1)
        span = np.where(any_client, last - first + 1, 0)
        both_bounds = []
        for mod in MODS:
            a = common[mod]
            first_mod = a.argmax(axis=1)
            last_mod = common_days - 1 - a[:, ::-1].argmax(axis=1)
            both_bounds.append((first_mod, last_mod))
        shared_first = np.maximum.reduce([x[0] for x in both_bounds])
        shared_last = np.minimum.reduce([x[1] for x in both_bounds])
        shared_span = np.where(all_client, np.maximum(0, shared_last - shared_first + 1), 0)
        output = {
            "clients_any_source_in_history": int(any_client.sum()),
            "clients_all_three_sources_in_history": int(all_client.sum()),
            "any_source_active_days_all_initial_clients": stats(active, common_days),
            "any_source_active_days_among_clients_with_any_events": stats(active, common_days, any_client),
            "any_source_active_days_among_all_three_source_clients": stats(active, common_days, all_client),
            "per_source_active_days": {m: stats(common[m].sum(axis=1), common_days) for m in MODS},
            "source_day_cells_all_initial_clients": stats(source_days, 3 * common_days),
            "sources_available_all_initial_clients": stats(source_count, 3),
            "any_source_first_last_calendar_span": stats(span, common_days),
            "all_three_common_first_last_calendar_span": stats(shared_span, common_days),
            "source_specific_full_history_active_days": {
                m: stats(daily[m][:, (source_start[m] - baseline).days:].sum(axis=1),
                         (CUTOFF - source_start[m]).days) for m in MODS},
            "any_source_full_earliest_history_active_days": stats(np.logical_or.reduce(list(daily.values())).sum(axis=1), days),
        }
        result[mode] = output
    invalid_trx = raw_counts["trx"] - clean_counts["trx"]
    overhalf = (raw_counts["trx"] > 0) & (2 * invalid_trx > raw_counts["trx"])
    any_trx = raw_counts["trx"] > 0
    result["transaction_attribute_missingness_in_common_history"] = {
        "clients_with_raw_transactions": int(any_trx.sum()),
        "clients_over_half_recorded_transactions_fail_attribute_quality": int(overhalf.sum()),
        "percent_of_all_initial_clients": float(overhalf.mean() * 100),
        "percent_of_clients_with_raw_transactions": float(overhalf.sum() / any_trx.sum() * 100),
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    (OUT / "full_history_missingness.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    strict = result["strict_event_quality"]
    for name, value in strict.items():
        if isinstance(value, dict) and "clients" in value:
            rows.append({"metric": name, **{k: v for k, v in value.items() if k != "observed_count_quantiles"}})
    pd.DataFrame(rows).to_csv(OUT / "full_history_missingness.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
