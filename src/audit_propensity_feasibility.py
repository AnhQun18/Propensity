"""Read-only feasibility audit; writes aggregate results under docs/propensity_feasibility.

Run with the project Python and one argument: mbd or ecommerce.
No model is trained, no identifiers are exported, and source data is unchanged.
"""
from pathlib import Path
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'propensity_feasibility'


def mbd():
    base = ROOT / 'data' / 'MBD-mini'
    tables = {}
    def inspect_file(file, with_time):
        pf = pq.ParquetFile(file)
        if not with_time or not pf.metadata.num_rows:
            return pf.metadata.num_rows, None, None, 0, False
        idx = pf.schema_arrow.get_field_index('event_time')
        stats = [pf.metadata.row_group(i).column(idx).statistics
                 for i in range(pf.metadata.num_row_groups)]
        if all(s is not None and s.has_min_max for s in stats):
            return (pf.metadata.num_rows, min(s.min for s in stats), max(s.max for s in stats),
                    sum(s.null_count or 0 for s in stats), False)
        # Spark INT96 timestamps commonly omit min/max in Parquet footers.
        times = pf.read(columns=['event_time'])['event_time']
        return pf.metadata.num_rows, pc.min(times).as_py(), pc.max(times).as_py(), times.null_count, True
    for name in ['client_split', 'targets', 'detail/trx', 'detail/geo',
                 'detail/dialog', 'ptls/trx', 'ptls/geo', 'ptls/dialog']:
        files = sorted((base / name).rglob('*.parquet'))
        rows, null_times, lows, highs = 0, 0, [], []
        scanned_time_files = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = pool.map(lambda f: inspect_file(f, name.startswith('detail/')), files)
            for n, lo, hi, null, scanned in results:
                rows += n
                null_times += null
                scanned_time_files += int(scanned)
                if lo is not None:
                    lows.append(lo)
                    highs.append(hi)
        tables[name] = {'files': len(files), 'rows': rows,
                        'bytes': sum(f.stat().st_size for f in files)}
        if lows:
            tables[name].update(min_time=str(min(lows)), max_time=str(max(highs)),
                                null_event_times=null_times, scanned_time_column_files=scanned_time_files)
        print(name, tables[name], flush=True)
    targets = pq.read_table(base / 'targets').to_pandas()
    clients = pq.read_table(base / 'client_split').to_pandas()
    label_cols = [f'target_{i}' for i in range(1, 5)]
    labels = {}
    for c in label_cols:
        labels[c] = {'counts': {str(k): int(v) for k, v in targets[c].value_counts(dropna=False).items()},
                     'positive_rate': float(targets[c].eq(1).mean())}
    monthly = targets.groupby('mon').agg(rows=('client_id', 'size'), clients=('client_id', 'nunique'))
    for c in label_cols:
        monthly[c + '_positive_rate'] = targets.groupby('mon')[c].apply(lambda s: s.eq(1).mean())
    monthly.to_csv(OUT / 'mbd_monthly.csv')
    split_ids = set(clients.client_id)
    target_ids = set(targets.client_id)
    return {'scope': 'all Parquet footers; full detail event_time columns where footer bounds absent; all target and client_split rows', 'tables': tables,
            'target_rows': len(targets), 'target_users': len(target_ids),
            'split_users': len(split_ids), 'split_rows': len(clients),
            'target_users_missing_split': len(target_ids - split_ids),
            'split_users_without_targets': len(split_ids - target_ids),
            'duplicate_client_month_rows': int(targets.duplicated(['client_id', 'mon']).sum()),
            'conflicting_split_clients': int(clients.groupby('client_id').fold.nunique().gt(1).sum()),
            'target_split_fold_mismatch_rows': int(targets.merge(clients, on='client_id', suffixes=('_target', '_split')).eval('fold_target != fold_split').sum()),
            'target_null_cells': {c: int(v) for c, v in targets.isna().sum().items()},
            'months': sorted(targets.mon.unique().tolist()), 'labels': labels,
            'limitation': 'Event contents, event duplicates, per-user history and detail/ptls equivalence are not fully audited.'}


def ecommerce():
    all_users, buyers, history_users, future_users, future_buyers = set(), set(), set(), set(), set()
    counts, dates = Counter(), Counter()
    per_file, rows, missing_category, nulls = {}, 0, 0, Counter()
    low, high = None, None
    start = '2019-11-03 00:00:00 UTC'
    cutoff = '2019-11-24 00:00:00 UTC'
    end = '2019-12-01 00:00:00 UTC'
    columns = ['event_time', 'event_type', 'user_id', 'category_code']
    for path in sorted((ROOT / 'data' / 'ecommerce').glob('*.csv')):
        file_rows = 0
        reader = csv.open_csv(path, read_options=csv.ReadOptions(block_size=32 * 1024 * 1024),
                              convert_options=csv.ConvertOptions(
                                  include_columns=columns,
                                  column_types={c: (pa.int64() if c == 'user_id' else pa.string()) for c in columns},
                                  strings_can_be_null=True))
        for batch in reader:
            t = pa.Table.from_batches([batch])
            rows += len(t)
            file_rows += len(t)
            for c in columns:
                nulls[c] += t[c].null_count
            missing_category += t['category_code'].null_count
            mn, mx = pc.min(t['event_time']).as_py(), pc.max(t['event_time']).as_py()
            low, high = min(low or mn, mn), max(high or mx, mx)
            for v in pc.value_counts(t['event_type']).to_pylist():
                counts[v['values']] += v['counts']
            for v in pc.value_counts(pc.utf8_slice_codeunits(t['event_time'], 0, 10)).to_pylist():
                dates[v['values']] += v['counts']
            all_users.update(pc.unique(t['user_id']).to_pylist())
            buy = pc.equal(t['event_type'], 'purchase')
            buyers.update(pc.unique(pc.filter(t['user_id'], buy)).to_pylist())
            hist = pc.and_(pc.greater_equal(t['event_time'], start), pc.less(t['event_time'], cutoff))
            future = pc.and_(pc.greater_equal(t['event_time'], cutoff), pc.less(t['event_time'], end))
            history_users.update(pc.unique(pc.filter(t['user_id'], hist)).to_pylist())
            future_users.update(pc.unique(pc.filter(t['user_id'], future)).to_pylist())
            future_buyers.update(pc.unique(pc.filter(t['user_id'], pc.and_(future, buy))).to_pylist())
        per_file[path.name] = {'rows': file_rows, 'bytes': path.stat().st_size}
        print(path.name, file_rows, 'rows; total users', len(all_users), flush=True)
    assert sum(counts.values()) == rows == sum(dates.values())
    assert nulls['event_time'] == nulls['event_type'] == nulls['user_id'] == 0
    pd.DataFrame(sorted(dates.items()), columns=['date_utc', 'events']).to_csv(OUT / 'ecommerce_daily.csv', index=False)
    positive = len(history_users & future_buyers)
    return {'scope': 'all rows, four selected columns, both local CSV files',
            'files': per_file, 'rows': rows, 'users': len(all_users),
            'users_with_purchase': len(buyers), 'event_counts': dict(counts),
            'min_time_utc': low, 'max_time_utc': high, 'distinct_calendar_days': len(dates),
            'null_counts_selected_columns': dict(nulls),
            'missing_category_code_rows': missing_category,
            'window': {'history_start_utc': start, 'cutoff_utc': cutoff, 'label_end_exclusive_utc': end,
                       'history_days': 21, 'label_days': 7, 'eligible_users': len(history_users),
                       'positive_purchase_users': positive, 'positive_purchase_rate': positive / len(history_users),
                       'future_active_users_in_cohort': len(history_users & future_users)},
            'limitations': ['Purchase event is not verified net revenue or a unique order.',
                            'No full-row deduplication, session or product audit.',
                            'Users without a future purchase log receive label 0; observability is not guaranteed.',
                            'One feasibility window, not a trained-model evaluation.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['mbd', 'ecommerce'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    result = mbd() if args.dataset == 'mbd' else ecommerce()
    result['elapsed_seconds'] = time.time() - tick
    (OUT / f'{args.dataset}_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
