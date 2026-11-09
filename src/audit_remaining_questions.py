"""Verify trans_count hypotheses, absence patterns, currency rules and label readiness.

Uses the existing full-history monthly cache after checking its source inventory.
No customer-level outputs, raw edits or model fitting. Readiness delays are explicit
scenarios, not facts inferred from the label month.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mbd_mini_eda.currency import currency_flags


ROOT = Path(__file__).resolve().parents[1]


def audit(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    raw = root / 'data/MBD-mini'
    cache = root / 'outputs/mbd_review_cache'
    inventory = json.loads((cache / 'source_inventory.json').read_text(encoding='utf-8'))
    for item in inventory['files']:
        stat = (raw / item['path']).stat()
        if stat.st_size != item['bytes'] or stat.st_mtime_ns != item['mtime_ns']:
            raise ValueError(f'Source inventory changed: {item["path"]}')
    digest = hashlib.sha256(json.dumps(inventory['files'], sort_keys=True).encode()).hexdigest()
    assert digest == inventory['fingerprint']
    with np.load(cache / 'source_month_counts.npz', allow_pickle=True) as saved:
        cube = saved['counts'].astype(np.int64)
        months = saved['months'].astype(str)
        ids = pd.Index(saved['client_ids'].astype(str))
        assert str(saved['source_fingerprint'].item()) == digest
    assert cube.shape == (100224, 25, 3)
    assert cube.sum(axis=(0, 1)).tolist() == [38764930, 66295724, 217205]
    db = duckdb.connect()
    db.execute("SET threads=2")
    db.execute("SET memory_limit='2GB'")
    target_files = sorted((raw / 'targets').rglob('*.parquet'))
    db.read_parquet([p.as_posix() for p in target_files], hive_partitioning=False).create_view('targets')
    targets = db.execute('SELECT client_id,mon,trans_count FROM targets').df()
    label_months = pd.to_datetime(targets.mon).dt.strftime('%Y-%m').to_numpy()
    positions = ids.get_indexer(targets.client_id.astype(str))
    assert (positions >= 0).all() and len(targets) == 1202688
    actual = targets.trans_count.to_numpy(dtype=np.int64)
    prefix = np.concatenate([np.zeros((len(ids), 1, 3), dtype=np.int64), cube.cumsum(axis=1)], axis=1)
    cut = np.searchsorted(months, label_months, side='left')
    positive = actual > 0
    records = []
    print('Comparing full-population monthly count hypotheses', flush=True)
    for source, si in [('trx', 0), ('geo', 1), ('dialog', 2)]:
        for shift in [-1, 0, 1]:
            end = np.clip(cut + shift, 0, len(months))
            upper = prefix[positions, end, si]
            starts = [('all_available', np.zeros(len(targets), dtype=int))]
            starts += [(f'rolling_{n}_calendar_months', np.maximum(0, end - n)) for n in [1, 2, 3, 6, 12, 18, 24]]
            starts += [(f'fixed_start_{month}', np.minimum(end, j)) for j, month in enumerate(months)]
            year_month = np.array([value[:4] + '-01' for value in label_months])
            starts += [('label_year_to_date', np.minimum(end, np.searchsorted(months, year_month)))]
            for window, left in starts:
                observed = upper - prefix[positions, left, si]
                for mode, candidate in [('raw_count', observed), ('capped_1000', np.minimum(observed, 1000))]:
                    exact = candidate == actual
                    records.append(dict(source=source, reference_month_shift=shift,
                                        window=window, mode=mode, rows=len(actual),
                                        exact=int(exact.sum()), exact_positive=int((exact & positive).sum()),
                                        source_positive=int(positive.sum()),
                                        exact_fraction=float(exact.mean()),
                                        exact_positive_fraction=float((exact & positive).sum() / positive.sum())))
    candidates = pd.DataFrame(records)
    candidates.to_csv(output / 'trans_count_candidates.csv', index=False)
    main = candidates[candidates.source.eq('trx') & candidates.reference_month_shift.eq(0)]
    best = main.sort_values(['exact', 'exact_positive'], ascending=False).iloc[0].to_dict()
    all_history = prefix[positions, cut, 0]
    never_trx = cube[:, :, 0].sum(axis=1) == 0
    diagnostic = dict(
        rows=len(actual), min=int(actual.min()), max=int(actual.max()),
        distinct=int(np.unique(actual).size), mean=float(actual.mean()),
        median=float(np.median(actual)), zero=int((actual == 0).sum()),
        at_1000=int((actual == 1000).sum()),
        source_exceeds_all_past_trx_rows=int((actual > all_history).sum()),
        clients_without_any_raw_trx=int(never_trx.sum()),
        positive_source_rows_without_any_raw_trx=int((positive & never_trx[positions]).sum()),
        best_trx_at_utc_cutoff=best,
        perfect_candidates=int(candidates.exact.eq(len(actual)).sum()),
        tested_candidates=len(candidates),
    )
    transitions = db.execute('''SELECT count(*) transitions,
        count(*) FILTER (WHERE trans_count<previous_count) decreasing,
        count(*) FILTER (WHERE trans_count>previous_count) increasing
        FROM (SELECT trans_count,lag(trans_count) OVER(PARTITION BY client_id ORDER BY mon) previous_count FROM targets)
        WHERE previous_count IS NOT NULL''').df().iloc[0].to_dict()
    diagnostic['transitions'] = {key: int(value) for key, value in transitions.items()}

    # These states describe the location of an empty client-month, not its cause.
    states = []
    coverage = []
    indices = np.arange(len(months))[None, :]
    for source, si in [('trx', 0), ('geo', 1), ('dialog', 2)]:
        observed = cube[:, :, si] > 0
        any_client = observed.any(axis=1)
        global_months = observed.any(axis=0)
        lo, hi = np.flatnonzero(global_months)[[0, -1]]
        outside = np.broadcast_to((indices < lo) | (indices > hi), observed.shape)
        first = np.argmax(observed, axis=1)[:, None]
        last = (len(months) - 1 - np.argmax(observed[:, ::-1], axis=1))[:, None]
        empty_inside = ~observed & ~outside
        masks = {
            'has_observed_events': observed,
            'outside_global_extracted_month_span': outside,
            'never_observed_in_source_within_span': empty_inside & ~any_client[:, None],
            'before_client_first_observed_month_within_span': empty_inside & any_client[:, None] & (indices < first),
            'between_client_first_last_observed_months': empty_inside & any_client[:, None] & (indices >= first) & (indices <= last),
            'after_client_last_observed_month_within_span': empty_inside & any_client[:, None] & (indices > last),
        }
        assert np.stack(list(masks.values())).sum(axis=0).min() == 1
        assert np.stack(list(masks.values())).sum(axis=0).max() == 1
        for name, mask in masks.items():
            states.append(dict(source=source, state=name, customer_months=int(mask.sum()),
                               total_customer_months=int(observed.size)))
        coverage.append(dict(source=source, with_source=int(any_client.sum()),
                             without_source=int((~any_client).sum()),
                             first_extracted_month=months[lo], last_extracted_month=months[hi]))
    pd.DataFrame(states).to_csv(output / 'absence_patterns.csv', index=False)
    pd.DataFrame(coverage).to_csv(output / 'source_coverage.csv', index=False)

    # Only schema keys are inspected: mtimes are download/file properties, not label-ready dates.
    target_columns = set()
    metadata_keys = set()
    for file in target_files:
        schema = pq.ParquetFile(file).schema_arrow
        target_columns.update(schema.names)
        metadata_keys.update(key.decode() for key in (schema.metadata or {}))
    availability_tokens = ('available', 'ready', 'publish', 'ingest', 'finalized')
    availability_fields = sorted(name for name in target_columns
                                 if any(token in name.lower() for token in availability_tokens))
    availability_metadata = sorted(name for name in metadata_keys
                                   if any(token in name.lower() for token in availability_tokens))
    observation_tokens = ('observed', 'collect', 'tracking', 'recording', 'coverage', 'available', 'ingest')
    observation_fields = sorted({line.split(':', 1)[0].strip()
                                 for item in inventory['files'] for line in item['schema'].splitlines()
                                 if any(token in line.split(':', 1)[0].lower() for token in observation_tokens)})
    readiness = dict(target_shards=len(target_files), columns=sorted(target_columns),
                     parquet_metadata_keys=sorted(metadata_keys),
                     possible_availability_fields=availability_fields,
                     possible_availability_metadata=availability_metadata,
                     observed_label_available_at_field=bool(availability_fields),
                     actual_release_delay_identifiable=bool(availability_fields or availability_metadata))
    scenarios = []
    for month in sorted(np.unique(label_months)):
        start = pd.Timestamp(month + '-01', tz='UTC')
        end = start + pd.DateOffset(months=1)
        for name, available in [(f'{n}_days', end + pd.Timedelta(days=n)) for n in [0, 1, 7, 30]] + [('one_calendar_month', end + pd.DateOffset(months=1))]:
            scenarios.append(dict(label_month=month, label_end_utc=end.isoformat(),
                                  assumed_delay=name, assumed_available_at_utc=available.isoformat(),
                                  available_at_test_2023_01_01=bool(available <= pd.Timestamp('2023-01-01', tz='UTC')),
                                  status='scenario_not_observed_release_time'))
    schedule = pd.DataFrame(scenarios)
    schedule.to_csv(output / 'label_readiness_scenarios.csv', index=False)
    readiness['december_calibration_available_at_january_test'] = schedule[schedule.label_month.eq('2022-12')][['assumed_delay', 'available_at_test_2023_01_01']].to_dict('records')

    print('Verifying currency buckets on all raw transactions', flush=True)
    files = sorted((raw / 'detail/trx').rglob('*.parquet'))
    db.read_parquet([p.as_posix() for p in files], hive_partitioning=False).create_view('trx')
    currencies = db.execute('SELECT currency,count(*) AS rows FROM trx GROUP BY currency ORDER BY currency').df()
    flags = currency_flags(currencies.currency)
    currencies = pd.concat([currencies, flags], axis=1)
    currencies.to_csv(output / 'currency_flags_by_code.csv', index=False)
    currency_result = dict(primary_code=11,
                          main_rows=int(currencies.loc[currencies.currency_is_main.fillna(False), 'rows'].sum()),
                          foreign_rows=int(currencies.loc[currencies.currency_is_foreign.fillna(False), 'rows'].sum()),
                          missing_rows=int(currencies.loc[currencies.currency_missing, 'rows'].sum()),
                          semantics='Foreign is the project flag for known codes other than 11; missing remains unknown.')
    assert sum(currency_result[key] for key in ['main_rows', 'foreign_rows', 'missing_rows']) == 38764930
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
                  cache_inventory_fingerprint=digest,
                  cache_validation='All source paths/byte sizes/mtime_ns match inventory; not a full raw content checksum.',
                  trans_count=diagnostic, source_coverage=coverage,
                  collection_or_observation_field_candidates=observation_fields,
                  readiness=readiness, currency=currency_result,
                  raw_edited=False, model_fitted=False)
    (output / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    db.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'docs/propensity_feasibility/remaining_questions')
    args = parser.parse_args()
    audit(ROOT, args.output_dir)
