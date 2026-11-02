"""Audit the downloaded Fliggy 140721 CSVs without changing raw data.

Run: .venv/Scripts/python.exe -X utf8 src/profile_fliggy_140721.py
Time zone assumption: UTC+08:00, appropriate for the source platform in China.
Behavior codes are reported literally; no undocumented numeric mapping is assumed.
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'Fliggy'
OUT = ROOT / 'docs' / 'fliggy_140721_profile'
OUT.mkdir(parents=True, exist_ok=True)
ZONE = timezone(timedelta(hours=8))
BASE = datetime(2022, 8, 16, tzinfo=ZONE)
BASE_TS = int(BASE.timestamp())
N = 3_000_000
CHUNK = 1_000_000
counts = np.zeros(N, dtype=np.uint32)
masks = np.zeros(N, dtype=np.uint64)
type_masks = {k: np.zeros(N, dtype=np.uint64) for k in (1, 2, 3)}
plan_seen = np.zeros(23_000_000, dtype=bool)
daily = np.zeros((64, 4), dtype=np.int64)
utc_daily = {}
type_totals = {}
mins, maxs = {}, {}
total = 0
started = time.time()
columns = ['user_id', 'query_id', 'plan_id', 'behavior', 'timestamp']
source_metadata = {p.name: {'bytes': p.stat().st_size} for p in SOURCE.glob('*.csv')}

for frame in pd.read_csv(SOURCE / 'user_behavior_transfer_plans.csv', header=None,
                         names=columns, dtype=np.int64, chunksize=CHUNK):
    if frame.shape[1] != 5 or frame.isna().any().any():
        raise ValueError('Unexpected schema or missing fields')
    for c in columns:
        mins[c] = min(mins.get(c, int(frame[c].min())), int(frame[c].min()))
        maxs[c] = max(maxs.get(c, int(frame[c].max())), int(frame[c].max()))
    users = frame.user_id.to_numpy()
    if users.min() < 0 or users.max() >= N:
        raise ValueError('User IDs exceed the checked dense array domain')
    plans = frame.plan_id.to_numpy()
    if plans.min() < 0 or plans.max() >= len(plan_seen):
        raise ValueError('Plan IDs exceed array domain')
    plan_seen[plans] = True
    days = ((frame.timestamp.to_numpy() - BASE_TS) // 86400).astype(np.int64)
    if days.min() < 0 or days.max() >= 64:
        raise ValueError('Timestamp out of the checked 64-day range')
    bits = np.left_shift(np.uint64(1), days.astype(np.uint64))
    np.add.at(counts, users, 1)
    np.bitwise_or.at(masks, users, bits)
    daily[:, 0] += np.bincount(days, minlength=64)
    for code, value in frame.behavior.value_counts().items():
        code = int(code)
        type_totals[code] = type_totals.get(code, 0) + int(value)
        if code not in type_masks:
            raise ValueError(f'Undocumented new behavior code {code}')
        selected = frame.behavior.to_numpy() == code
        np.bitwise_or.at(type_masks[code], users[selected], bits[selected])
        daily[:, code] += np.bincount(days[selected], minlength=64)
    ud, uc = np.unique(frame.timestamp.to_numpy() // 86400, return_counts=True)
    for d, c in zip(ud, uc):
        utc_daily[int(d)] = utc_daily.get(int(d), 0) + int(c)
    total += len(frame)
    if total % 5_000_000 == 0:
        print(f'Log: {total:,} rows, {time.time()-started:.1f}s', flush=True)

present = counts > 0
ids = np.flatnonzero(present)
user_counts = counts[present]
user_masks = masks[present]
number_days = int((maxs['timestamp'] - BASE_TS) // 86400 + 1)
day_index = pd.date_range(BASE.date(), periods=number_days, freq='D')
popcount = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)

def bitcount(values):
    return popcount[np.ascontiguousarray(values).view(np.uint8).reshape(-1, 8)].sum(axis=1).astype(np.uint8)

active_days = bitcount(user_masks)
day_active_users = np.array([np.count_nonzero(user_masks & np.uint64(1 << d))
                             for d in range(number_days)])
daily_frame = pd.DataFrame({'date_utc_plus_8': day_index.strftime('%Y-%m-%d'),
                           'events': daily[:number_days, 0],
                           'active_users': day_active_users})
for code in (1, 2, 3):
    daily_frame[f'code_{code}_events'] = daily[:number_days, code]
daily_frame.to_csv(OUT / 'daily_counts.csv', index=False)
pd.DataFrame({'date_utc': [datetime.fromtimestamp(d * 86400, timezone.utc).date().isoformat()
                          for d in sorted(utc_daily)],
              'events': [utc_daily[d] for d in sorted(utc_daily)]}).to_csv(OUT / 'daily_counts_utc.csv', index=False)

def quantiles(values):
    return {str(q): float(np.quantile(values, q)) for q in (0, .25, .5, .75, .9, .95, .99, 1)}

distribution = []
for k in range(1, number_days + 1):
    n = int(np.count_nonzero(active_days == k))
    distribution.append({'active_days': k, 'users': n, 'share': n / len(ids)})
pd.DataFrame(distribution).to_csv(OUT / 'active_day_distribution.csv', index=False)
frequency_values, frequency_counts = np.unique(user_counts, return_counts=True)
pd.DataFrame({'event_count': frequency_values, 'users': frequency_counts}).to_csv(OUT / 'event_count_distribution.csv', index=False)

window_rows = []
for history_length, future_length in ((7, 7), (14, 7), (21, 7)):
    for start in sorted(set((0, number_days-history_length-future_length))):
        if start < 0:
            continue
        history_bits = np.uint64(((1 << history_length)-1) << start)
        future_bits = np.uint64(((1 << future_length)-1) << (start+history_length))
        history_day_count = bitcount(user_masks & history_bits)
        for minimum_history_days in (1, 2, 3, 5):
            eligible = history_day_count >= minimum_history_days
            n = int(eligible.sum())
            row = {'history_start': day_index[start].date().isoformat(),
                   'history_end': day_index[start+history_length-1].date().isoformat(),
                   'future_start': day_index[start+history_length].date().isoformat(),
                   'future_end': day_index[start+history_length+future_length-1].date().isoformat(),
                   'history_days': history_length, 'future_days': future_length,
                   'minimum_history_active_days': minimum_history_days,
                   'eligible_users': n,
                   'future_active_users': int(np.count_nonzero(eligible & ((user_masks & future_bits)>0)))}
            for code in (1, 2, 3):
                code_future = (type_masks[code][present] & future_bits) > 0
                positive = int(np.count_nonzero(eligible & code_future))
                row[f'future_code_{code}_users'] = positive
                row[f'future_code_{code}_rate'] = positive/n if n else None
            row['future_active_rate'] = row['future_active_users']/n if n else None
            window_rows.append(row)
pd.DataFrame(window_rows).to_csv(OUT / 'window_feasibility.csv', index=False)

per_user = pd.DataFrame({'user_id': ids, 'events': user_counts, 'active_days': active_days})
for code in (1, 2, 3):
    per_user[f'code_{code}_active_days'] = bitcount(type_masks[code][present])
per_user.to_parquet(OUT / 'per_user_counts.parquet', index=False)
sample_ids = np.r_[ids[:10], ids[np.argsort(active_days)[-10:]]]
matrix_sample = pd.DataFrame({'user_id': sample_ids})
for d, label in enumerate(day_index.strftime('%Y-%m-%d')):
    matrix_sample[label] = ((masks[sample_ids] & np.uint64(1 << d))>0).astype(np.uint8)
matrix_sample.to_csv(OUT / 'binary_matrix_sample.csv', index=False)

print(f'Log complete: {total:,} events, {len(ids):,} users, {number_days} days', flush=True)

profile = pd.read_csv(SOURCE / 'user_profile.csv', header=None,
                      names=['user_id','age','gender','occupation','habitual_city'], dtype=np.int64)
profile_ids = profile.user_id.to_numpy()
profile_unique = np.unique(profile_ids)
valid_user_domain = (profile_unique >= 0) & (profile_unique < N)
matched_profile_users = int(np.count_nonzero(present[profile_unique[valid_user_domain]]))
profile_summary = {'rows': len(profile), 'unique_users': len(profile_unique),
                   'duplicate_user_id_rows': len(profile)-len(profile_unique),
                   'log_users_with_profile': matched_profile_users,
                   'log_users_without_profile': len(ids)-matched_profile_users,
                   'missing_cells': int(profile.isna().sum().sum()),
                   'negative_value_counts': {c: int((profile[c]<0).sum()) for c in profile.columns}}
print('User profiles checked', flush=True)

plan_profile_seen = np.zeros(len(plan_seen), dtype=bool)
plan_rows = 0
duplicate_plan_rows = 0
for frame in pd.read_csv(SOURCE / 'transfer_plan_profile.csv', header=None,
                         usecols=[0,1], names=['query_id','plan_id'], dtype=np.int64, chunksize=CHUNK):
    unique_plan_ids = np.unique(frame.plan_id.to_numpy())
    if unique_plan_ids.min()<0 or unique_plan_ids.max()>=len(plan_seen):
        raise ValueError('Plan profiles exceed checked ID domain')
    duplicate_plan_rows += len(frame)-len(unique_plan_ids) + int(plan_profile_seen[unique_plan_ids].sum())
    plan_profile_seen[unique_plan_ids] = True
    plan_rows += len(frame)

date_format = lambda value, zone: datetime.fromtimestamp(value, zone).isoformat()
thresholds = {str(k): {'users':int(np.count_nonzero(active_days>=k)),
                       'share':float(np.mean(active_days>=k))} for k in (2,3,5,7,10,14,21,28)}
summary = {'source_files':source_metadata,
           'schema':columns, 'header':False,
           'rows':total, 'users':len(ids), 'distinct_plans_in_log':int(plan_seen.sum()),
           'column_minimum':mins, 'column_maximum':maxs,
           'behavior_codes': {str(k):{'events':v,'share':v/total,
                                     'users':int(np.count_nonzero(type_masks[k]))} for k,v in type_totals.items()},
           'timestamp_unit':'seconds', 'calendar_timezone_assumption':'UTC+08:00',
           'min_time_utc':date_format(mins['timestamp'],timezone.utc),
           'max_time_utc':date_format(maxs['timestamp'],timezone.utc),
           'min_time_utc_plus_8':date_format(mins['timestamp'],ZONE),
           'max_time_utc_plus_8':date_format(maxs['timestamp'],ZONE),
           'continuous_calendar_days':number_days,
           'zero_event_days':int(np.count_nonzero(daily[:number_days,0]==0)),
           'events_per_user_quantiles':quantiles(user_counts),
           'active_days_per_user_quantiles':quantiles(active_days),
           'mean_events_per_user':float(user_counts.mean()),
           'mean_active_days_per_user':float(active_days.mean()),
           'one_active_day_users':int(np.count_nonzero(active_days==1)),
           'one_active_day_share':float(np.mean(active_days==1)),
           'active_day_thresholds':thresholds,
           'binary_matrix':{'rows':len(ids),'columns':number_days,
                            'one_cells':int(active_days.sum()),
                            'one_share':float(active_days.sum()/(len(ids)*number_days))},
           'daily_events':{'min':int(daily_frame.events.min()),'max':int(daily_frame.events.max()),
                           'mean':float(daily_frame.events.mean()),
                           'coefficient_of_variation':float(daily_frame.events.std(ddof=0)/daily_frame.events.mean())},
           'user_profile':profile_summary,
           'plan_profile':{'rows':plan_rows,'distinct_plans':int(plan_profile_seen.sum()),
                           'duplicate_plan_id_rows':duplicate_plan_rows,
                           'log_plans_without_profile':int(np.count_nonzero(plan_seen & ~plan_profile_seen))},
           'full_scan_validations':{'all_log_rows_parse_as_five_integer_fields':True,
                                    'log_missing_cells':0,
                                    'per_user_event_sum_matches_rows':bool(int(user_counts.sum())==total),
                                    'daily_event_sum_matches_rows':bool(int(daily_frame.events.sum())==total),
                                    'behavior_event_sum_matches_rows':bool(sum(type_totals.values())==total),
                                    'full_record_duplicates_checked':False},
           'limitations':['Behavior code to action mapping is unverified.',
                          'UTC+08:00 is an explicit calendar assumption, not a documented source timezone.',
                          'An absent log is no recorded action, not proof the user was fully observed.',
                          'Global timeline does not imply each user was observed for its full duration.',
                          'No cross-dataset ID compatibility is assumed.'],
           'elapsed_seconds':time.time()-started}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
