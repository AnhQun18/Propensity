"""Compare three proposed diff_trans_date formulas against all MBD-mini targets.

Reads one fold at a time; exports aggregate diagnostics, never customer IDs.
Formula 3 is evaluated as the last two observed transactions before each reference;
targets itself is a customer-month table, not an event table.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import time

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DAY = 86400 * 10**9
WINDOWS = ['30d', '84d', '90d', '180d', '365d', 'calendar_year', 'all_available']
REFERENCES = [(name, offset) for name in ['label_month_start', 'mon_midnight', 'following_month_start'] for offset in [0, 3]]


def variants():
    names = []
    for denominator in ['source_trans_count', 'observed_count']:
        for span in ['elapsed_days', 'calendar_days']:
            for rounding in ['unrounded', 'floor', 'ceil', 'half_up']:
                names.append(('average_interval', f'{span}/{denominator}_minus_1/{rounding}'))
    for rounding in ['unrounded', 'floor', 'ceil', 'half_up']:
        names.append(('recency', f'elapsed_days/{rounding}'))
    names += [('recency', 'calendar_days'), ('recency', 'calendar_days_minus_1')]
    for rounding in ['unrounded', 'floor', 'ceil', 'half_up']:
        names.append(('last_two_gap', f'elapsed_days/{rounding}'))
    names.append(('last_two_gap', 'calendar_days'))
    return names


def roundings(value):
    return [value, np.floor(value), np.ceil(value), np.floor(value + .5)]


def fixed_reference_diagnostic(data_root, output, db):
    files = sorted((data_root / 'targets').rglob('*.parquet'))
    db.read_parquet([p.as_posix() for p in files], hive_partitioning=False).create_view('targets_for_variation', replace=True)
    row = db.execute('''SELECT count(*) clients,
        count(*) FILTER (WHERE distinct_diff>1) clients_with_varying_nonnull_diff,
        count(*) FILTER (WHERE distinct_diff=1) clients_with_one_distinct_nonnull_diff,
        count(*) FILTER (WHERE distinct_diff=0) clients_with_only_null_diff
        FROM (SELECT client_id,count(DISTINCT diff_trans_date) distinct_diff
              FROM targets_for_variation GROUP BY client_id)''').df().iloc[0].to_dict()
    result = {key: int(value) for key, value in row.items()}
    (output / 'fixed_reference_diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def audit(data_root: Path, output: Path):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect()
    db.execute("SET threads=2")
    db.execute("SET memory_limit='3GB'")
    names = variants()
    shape = (len(WINDOWS), len(REFERENCES), len(names))
    scores = {k: np.zeros(shape, dtype=np.int64) for k in ['comparable', 'exact_nonnull', 'null_agreement', 'exact_with_matching_count', 'comparable_with_matching_count']}
    total = dict(rows=0, clients=0, trx_rows=0, source_nonnull=0, source_null=0,
                 source_count_one=0, source_count_one_positive_diff=0,
                 source_count_one_null_diff=0, source_count_one_zero_diff=0)
    for fold in range(5):
        print(f'Fold {fold}: reading complete transaction histories', flush=True)
        target_files = sorted((data_root / 'targets' / f'fold={fold}').glob('*.parquet'))
        trx_files = sorted((data_root / 'detail/trx' / f'fold={fold}').glob('*.parquet'))
        if not target_files or not trx_files:
            raise FileNotFoundError(f'Missing fold {fold}')
        labels = db.read_parquet([p.as_posix() for p in target_files], hive_partitioning=False).project('client_id, mon, trans_count, diff_trans_date').df().sort_values(['client_id', 'mon']).reset_index(drop=True)
        db.read_parquet([p.as_posix() for p in trx_files], hive_partitioning=False).project('client_id, event_time').create_view('trx_fold', replace=True)
        histories = db.execute('SELECT client_id, list(epoch_ns(event_time) ORDER BY event_time) times FROM trx_fold GROUP BY client_id').df()
        times = dict(zip(histories.client_id, histories.times))
        total['trx_rows'] += sum(len(t) for t in times.values())
        groups = labels.groupby('client_id', sort=False).indices
        mon = pd.to_datetime(labels.mon).to_numpy(dtype='datetime64[ns]')
        month = mon.astype('datetime64[M]')
        raw_refs = {
            'label_month_start': month.astype('datetime64[ns]'),
            'mon_midnight': mon,
            'following_month_start': (month + np.timedelta64(1, 'M')).astype('datetime64[ns]'),
        }
        ends = np.stack([raw_refs[name].astype('int64') - offset * 3600 * 10**9 for name, offset in REFERENCES])
        year_starts = np.stack([(pd.DatetimeIndex(raw_refs[name]) - pd.DateOffset(years=1)).to_numpy(dtype='datetime64[ns]').astype('int64') - offset * 3600 * 10**9 for name, offset in REFERENCES])
        actual = labels.diff_trans_date.to_numpy(dtype=float)
        source_count = labels.trans_count.to_numpy(dtype=float)
        one = source_count == 1
        total['rows'] += len(labels)
        total['clients'] += len(groups)
        total['source_nonnull'] += int(np.isfinite(actual).sum())
        total['source_null'] += int(np.isnan(actual).sum())
        total['source_count_one'] += int(one.sum())
        total['source_count_one_positive_diff'] += int((one & (actual > 0)).sum())
        total['source_count_one_null_diff'] += int((one & np.isnan(actual)).sum())
        total['source_count_one_zero_diff'] += int((one & (actual == 0)).sum())
        offsets = np.array([offset * 3600 * 10**9 for _, offset in REFERENCES])[:, None]
        for number, (client, idx) in enumerate(groups.items(), 1):
            if number % 5000 == 0:
                print(f'Fold {fold}: {number}/{len(groups)} customers; {time.monotonic()-started:.1f}s', flush=True)
            t = times.get(client)
            if t is None:
                t = np.empty(0, dtype=np.int64)
            ref = ends[:, idx]
            right = np.searchsorted(t, ref, side='left')
            padded = np.concatenate((t, np.zeros(1, dtype=np.int64)))
            # Invalid indices use the harmless sentinel and are masked before scoring.
            last = padded[np.where(right > 0, right - 1, len(t))]
            previous = padded[np.where(right > 1, right - 2, len(t))]
            source = actual[idx][None, :]
            known = np.isfinite(source)
            source_denominator = source_count[idx][None, :] - 1
            for wi, window in enumerate(WINDOWS):
                if window == 'all_available':
                    left = np.zeros_like(right)
                else:
                    start = year_starts[:, idx] if window == 'calendar_year' else ref - int(window[:-1]) * DAY
                    left = np.searchsorted(t, start, side='left')
                count = right - left
                present = count > 0
                two = count >= 2
                first = padded[np.where(present, left, len(t))]
                span = (last - first) / DAY
                calendar_span = (last + offsets) // DAY - (first + offsets) // DAY
                candidates = []
                with np.errstate(divide='ignore', invalid='ignore'):
                    for denominator, valid in [(source_denominator, present & (source_denominator > 0)), (count - 1, two)]:
                        for numerator in [span, calendar_span]:
                            value = np.where(valid, numerator / denominator, np.nan)
                            candidates.extend(roundings(value))
                recency = np.where(present, (ref - last) / DAY, np.nan)
                candidates.extend(roundings(recency))
                calendar_recency = np.where(present, (ref + offsets) // DAY - (last + offsets) // DAY, np.nan)
                candidates += [calendar_recency, calendar_recency - 1]
                gap = np.where(two, (last - previous) / DAY, np.nan)
                candidates.extend(roundings(gap))
                candidates.append(np.where(two, (last + offsets) // DAY - (previous + offsets) // DAY, np.nan))
                values = np.stack(candidates, axis=1)
                comparable = np.isfinite(values) & known[:, None, :]
                exact = comparable & np.isclose(values, source[:, None, :], rtol=0, atol=1e-9)
                matched_count = (count == source_count[idx][None, :])[:, None, :]
                scores['comparable'][wi] += comparable.sum(axis=2)
                scores['exact_nonnull'][wi] += exact.sum(axis=2)
                scores['null_agreement'][wi] += (np.isnan(values) & np.isnan(source[:, None, :])).sum(axis=2)
                scores['exact_with_matching_count'][wi] += (exact & matched_count).sum(axis=2)
                scores['comparable_with_matching_count'][wi] += (comparable & matched_count).sum(axis=2)
        print(f'Fold {fold} complete: cumulative {total["rows"]} rows; {time.monotonic()-started:.1f}s', flush=True)

    assert total['rows'] == 1202688 and total['clients'] == 100224 and total['trx_rows'] == 38764930
    assert total['source_count_one'] == 50460 and total['source_count_one_positive_diff'] == 49293
    fixed_reference_diagnostic(data_root, output, db)
    records = []
    for wi, window in enumerate(WINDOWS):
        for ri, (reference, offset) in enumerate(REFERENCES):
            for vi, (family, formula) in enumerate(names):
                row = dict(family=family, formula=formula, window=window, reference=reference, utc_offset_hours=offset)
                row.update({key: int(value[wi, ri, vi]) for key, value in scores.items()})
                row['source_nonnull'] = total['source_nonnull']
                row['exact_fraction_source_nonnull'] = row['exact_nonnull'] / total['source_nonnull']
                row['exact_fraction_comparable'] = row['exact_nonnull'] / row['comparable'] if row['comparable'] else None
                row['exact_including_null'] = row['exact_nonnull'] + row['null_agreement']
                records.append(row)
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'candidate_results.csv', index=False)
    best = []
    for family in ['average_interval', 'recency', 'last_two_gap']:
        candidates = frame[frame.family.eq(family)].sort_values(['exact_nonnull', 'comparable'], ascending=False)
        best.append(candidates.iloc[0].to_dict())
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), scope='All target rows and all detail/trx events, processed fold by fold; aggregate outputs only.', totals=total, candidate_configurations=len(frame), perfect_formulas=int(frame.exact_including_null.eq(total['rows']).sum()), best_by_family=best, elapsed_seconds=round(time.monotonic()-started, 3), caveats=['References mon_midnight and following_month_start are diagnostics, not automatically valid prediction cutoffs.', 'Best means closest among tested candidates, not a confirmed generating formula.', 'last_two_gap is a customer-month surrogate of the event-row lag formula.', 'Zero-event and one-event histories are undefined for the mean interval; source-count denominator additionally requires source trans_count > 1.'])
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    db.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/MBD-mini')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'docs/propensity_feasibility/three_formulas')
    args = parser.parse_args()
    audit(args.data_root, args.output_dir)
