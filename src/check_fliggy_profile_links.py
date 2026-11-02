"""Check key uniqueness and log/profile coverage for the Fliggy audit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'Fliggy'
OUT = ROOT / 'docs' / 'fliggy_140721_profile'
users = pd.read_csv(SOURCE / 'user_profile.csv', header=None, dtype=np.int64)
unique_user_rows = users.drop_duplicates()
duplicate_user_keys = users[0].duplicated(keep=False)
conflicting_user_keys = unique_user_rows[0].duplicated(keep=False)
result = {'user_profile': {
    'exact_duplicate_rows': int(users.duplicated().sum()),
    'users_with_multiple_rows': int(users.loc[duplicate_user_keys,0].nunique()),
    'users_with_conflicting_profiles': int(unique_user_rows.loc[conflicting_user_keys,0].nunique())}}
del users, unique_user_rows
plan_seen = np.zeros(23_000_000, dtype=bool)
pairs = []
for chunk in pd.read_csv(SOURCE / 'transfer_plan_profile.csv', header=None,
                          usecols=[0,1], dtype=np.int64, chunksize=1_000_000):
    if chunk[0].max() >= 100_000 or chunk[0].min() < 0:
        raise ValueError('Query IDs exceed encoding domain')
    plan_seen[chunk[1].to_numpy()] = True
    pairs.append(chunk[1].to_numpy() * 100_000 + chunk[0].to_numpy())
unique_pairs, repeats = np.unique(np.concatenate(pairs), return_counts=True)
result['plan_profile'] = {'distinct_query_plan_pairs': len(unique_pairs),
                           'duplicate_query_plan_pair_rows': int((repeats-1).sum()),
                           'query_plan_pairs_with_multiple_profile_rows': int((repeats>1).sum())}
del pairs, unique_pairs, repeats
matched_events = 0
unmatched_events = 0
unmatched_by_behavior = {}
for chunk in pd.read_csv(SOURCE / 'user_behavior_transfer_plans.csv', header=None,
                          usecols=[2,3], dtype=np.int64, chunksize=1_000_000):
    matched = plan_seen[chunk[2].to_numpy()]
    matched_events += int(matched.sum())
    unmatched_events += int((~matched).sum())
    for code, n in chunk.loc[~matched,3].value_counts().items():
        unmatched_by_behavior[str(code)] = unmatched_by_behavior.get(str(code),0)+int(n)
result['log_plan_links'] = {'matched_event_rows':matched_events,
                            'unmatched_event_rows':unmatched_events,
                            'matched_event_share':matched_events/(matched_events+unmatched_events),
                            'unmatched_by_behavior':unmatched_by_behavior}
(OUT / 'profile_links.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
