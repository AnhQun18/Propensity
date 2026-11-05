"""Recompute join, full-history presence and temporal diagnostics; aggregates only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/MBD-mini'

def main():
    clients = pq.read_table(BASE / 'client_split').to_pandas().sort_values('client_id').reset_index(drop=True)
    targets = pq.read_table(BASE / 'targets').to_pandas()
    ids = pd.Index(clients.client_id)
    n = len(ids)
    months = sorted(targets.mon.unique())
    bounds = pd.to_datetime(months).to_numpy(dtype='datetime64[ns]').astype('int64')
    last = np.full(n, np.iinfo('int64').min, dtype='int64')
    after = np.zeros((n, len(months)), dtype='int64')
    presence = {}
    total = 0
    for mod in ('trx', 'geo', 'dialog'):
        seen = np.zeros(n, dtype=bool)
        for path in (BASE / 'detail' / mod).rglob('*.parquet'):
            cols = ['client_id', 'event_time'] if mod == 'trx' else ['client_id']
            frame = pq.ParquetFile(path).read(columns=cols).to_pandas()
            pos = ids.get_indexer(frame.client_id)
            assert (pos >= 0).all()
            seen[pos] = True
            if mod == 'trx':
                assert frame.event_time.notna().all()
                times = frame.event_time.to_numpy(dtype='datetime64[ns]').astype('int64')
                np.maximum.at(last, pos, times)
                total += len(frame)
                for j, bound in enumerate(bounds):
                    after[:, j] += np.bincount(pos[times > bound], minlength=n)
        presence[mod] = seen
        print(mod, int(seen.sum()), flush=True)
    has_trx = presence['trx']
    delta = np.full((n, len(months)), np.nan)
    delta[has_trx] = np.floor((bounds[None, :] - last[has_trx, None]) / 86400e9)
    labels = [f'target_{i}' for i in range(1, 5)]
    y = targets[labels].eq(1)
    folds = []
    for fold, group in clients.groupby('fold', observed=True):
        ix = group.index.to_numpy()
        folds.append({'fold': int(fold), 'clients': len(ix), **{f'{m}_absent_percent': float((~p[ix]).mean()*100) for m,p in presence.items()}})
    monthly = []
    for j, month in enumerate(months):
        d = delta[:, j]
        monthly.append({'mon': str(month), 'clients_any_trx_after_mon': int((after[:, j]>0).sum()),
                        'percent_all_clients_any_after_mon': float((after[:,j]>0).mean()*100),
                        'transactions_after_mon': int(after[:,j].sum()),
                        'percent_transactions_after_mon': float(after[:,j].sum()/total*100),
                        'delta_min_median_max_days': [float(np.nanmin(d)),float(np.nanmedian(d)),float(np.nanmax(d))],
                        'full_history_last_trx_more_than_180_days_before_mon': int((d>180).sum()),
                        'full_history_last_trx_more_than_365_days_before_mon': int((d>365).sum())})
    valid = delta[np.isfinite(delta)]
    extremes = {}
    for name, mask in {'min':delta==valid.min(),'max':delta==valid.max(),'gt180':delta>180,'gt365':delta>365,'lt_minus180':delta < -180}.items():
        extremes[name] = {'client_months':int(mask.sum()),'unique_clients':int(mask.any(axis=1).sum())}
    joined = targets.merge(clients, on='client_id', validate='many_to_one', suffixes=('_target','_split'))
    assert len(joined)==len(targets)
    assert joined.fold_target.eq(joined.fold_split).all()
    result = {'definitions': {'presence':'ID occurs in detail over ALL available history, no attribute filtering',
                             'after_mon':'event_time > mon at 00:00:00; same-day later events included',
                             'delta':'floor((mon - full-history last_trx_time) / 1 day); NOT safe training recency',
                             'event_denominator':'all detail/trx rows; computed independently per mon, not summed over months'},
              'clients':n,'target_rows':len(targets),'duplicate_client_months':int(targets.duplicated(['client_id','mon']).sum()),
              'months_per_client_counts':{str(k):int(v) for k,v in targets.groupby('client_id').mon.nunique().value_counts().items()},
              'folds':folds, 'transaction_rows':total,'monthly':monthly,
              'delta_quantiles_days':{str(q):float(np.quantile(valid,q)) for q in (0,.01,.05,.25,.5,.75,.95,.99,1)},
              'delta_extremes':extremes,
              'positive_counts':{c:int(y[c].sum()) for c in labels},
              'positive_percent':{c:float(y[c].mean()*100) for c in labels},
              'positive_label_count_distribution':{str(k):int(v) for k,v in y.sum(axis=1).value_counts().sort_index().items()},
              'pair_overlap_same_client_month':{a+' & '+b:int((y[a]&y[b]).sum()) for i,a in enumerate(labels) for b in labels[i+1:]}}
    out = ROOT / 'docs/propensity_feasibility/join_questions_audit.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__':
    main()
