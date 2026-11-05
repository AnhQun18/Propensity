"""Read-only feasibility audit; writes aggregate results under docs/propensity_feasibility.

Run with the project Python and one argument: mbd.
No model is trained, no identifiers are exported, and source data is unchanged.
"""
from pathlib import Path
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow.compute as pc
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['mbd'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tick = time.time()
    result = mbd()
    result['elapsed_seconds'] = time.time() - tick
    (OUT / f'{args.dataset}_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
