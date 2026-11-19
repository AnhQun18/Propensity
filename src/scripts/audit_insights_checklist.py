"""Reconcile raw keys/monthly counts and produce EDA evidence, without training."""
from pathlib import Path
import json
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/eda_insights_20261119'
OUT.mkdir(parents=True, exist_ok=True)
BASE = ROOT / 'outputs/full_population_temporal_eda_20261118'
c = duckdb.connect()
c.execute("SET memory_limit='3GB'")
c.execute('SET threads=2')
c.execute(f"SET temp_directory='{OUT.as_posix()}/tmp'")

def path(p):
    return str(p).replace('\\', '/').replace("'", "''")

c.execute(f"CREATE VIEW clients AS SELECT * FROM read_parquet('{path(ROOT/'data/MBD-mini/client_split/**/*.parquet')}')")
c.execute(f"CREATE VIEW labels AS SELECT *, date_trunc('month',cast(mon AS DATE)) AS month, greatest(target_1,target_2,target_3,target_4) AS y FROM read_parquet('{path(ROOT/'data/MBD-mini/targets/**/*.parquet')}')")
n = c.execute('SELECT count(*) FROM clients').fetchone()[0]
checks = {}
checks['clients_unique_nonnull'] = c.execute('SELECT count(*)=count(DISTINCT client_id) FROM clients').fetchone()[0]
checks['label_keys_unique_nonnull'] = c.execute('SELECT count(*)=count(DISTINCT (client_id,month)) AND count(client_id)=count(*) AND count(month)=count(*) FROM labels').fetchone()[0]
checks['labels_binary_complete'] = c.execute('SELECT count(*) FROM labels WHERE ' + ' OR '.join(f'target_{i} IS NULL OR target_{i} NOT IN (0,1)' for i in range(1,5))).fetchone()[0] == 0
checks['label_ids_match'] = c.execute('SELECT count(*) FROM labels l ANTI JOIN clients c USING(client_id)').fetchone()[0] == 0
monthly = c.execute('SELECT month,count(*) n,sum(y) buyers,avg(y) rate FROM labels GROUP BY 1 ORDER BY 1').df()
checks['complete_12_months'] = len(monthly) == 12 and monthly.n.eq(n).all()
monthly.to_csv(OUT/'label_monthly.csv', index=False)
audit = []
for s in ['trx','geo','dialog']:
    print('Audit raw and reconcile Moscow monthly counts:', s, flush=True)
    c.execute(f"CREATE OR REPLACE VIEW events AS SELECT client_id,event_time FROM read_parquet('{path(ROOT/f'data/MBD-mini/detail/{s}/**/*.parquet')}')")
    row = c.execute("SELECT count(*) rows,count(DISTINCT client_id) customers,count(*) FILTER(WHERE client_id IS NULL) null_ids,count(*) FILTER(WHERE event_time IS NULL) null_times FROM events").df().iloc[0].to_dict()
    row['source'] = s
    row['orphan_events'] = c.execute('SELECT count(*) FROM events e ANTI JOIN clients c USING(client_id)').fetchone()[0]
    c.execute("CREATE OR REPLACE TEMP TABLE raw_monthly AS SELECT client_id,date_trunc('month',event_time+INTERVAL '3 hours') AS month,count(*) n FROM events GROUP BY 1,2")
    cached = path(BASE/f'{s}_monthly.parquet')
    mismatch = c.execute(f"SELECT count(*) FROM raw_monthly r FULL JOIN read_parquet('{cached}') a USING(client_id,month) WHERE r.n IS DISTINCT FROM a.{s}_events").fetchone()[0]
    row['monthly_mismatches'] = mismatch
    checks[f'{s}_raw_keys_and_cache'] = not any(row[k] for k in ['null_ids','null_times','orphan_events','monthly_mismatches'])
    audit.append(row)
pd.DataFrame(audit).to_csv(OUT/'join_audit.csv',index=False)
print('Audit amount distribution', flush=True)
c.execute(f"CREATE VIEW trx AS SELECT amount,currency FROM read_parquet('{path(ROOT/'data/MBD-mini/detail/trx/**/*.parquet')}')")
c.execute('SELECT currency,count(*) n,count(*) FILTER(WHERE amount IS NULL) missing_amount,count(*) FILTER(WHERE amount=0) zero_amount,count(*) FILTER(WHERE amount<0) negative_amount,min(amount) minimum,quantile_cont(amount,0.5) median,quantile_cont(amount,0.99) p99,max(amount) maximum FROM trx GROUP BY 1 ORDER BY n DESC').df().to_csv(OUT/'amount_by_currency.csv',index=False)

# Rebuild the common-cohort features from reconciled source-month counts.
c.execute(f"CREATE VIEW panel AS SELECT * FROM read_parquet('{path(BASE/'all_customer_months.parquet')}')")
checks['panel_keys_unique'] = c.execute('SELECT count(*)=count(DISTINCT (client_id,month)) FROM panel').fetchone()[0]
for s in ['trx','geo','dialog']:
    checks[f'{s}_panel_matches_cache'] = c.execute(f"SELECT count(*) FROM panel p FULL JOIN read_parquet('{path(BASE/f'{s}_monthly.parquet')}') a USING(client_id,month) WHERE coalesce(p.{s}_events,0) IS DISTINCT FROM coalesce(a.{s}_events,0)").fetchone()[0] == 0
checks['panel_labels_match_raw'] = c.execute('SELECT count(*) FROM labels l LEFT JOIN panel p USING(client_id,month) WHERE l.y IS DISTINCT FROM p.y').fetchone()[0] == 0
d = c.execute("""WITH h AS (
 SELECT client_id,month,y,
 sum(coalesce(trx_events,0)) OVER w trx_hist,
 sum(coalesce(geo_events,0)) OVER w geo_hist,
 sum(coalesce(dialog_events,0)) OVER w dialog_hist,
 max(CASE WHEN dialog_events>0 THEN month END) OVER w last_dialog,
 lead(y,1) OVER (PARTITION BY client_id ORDER BY month) y2,
 lead(y,2) OVER (PARTITION BY client_id ORDER BY month) y3
 FROM panel WINDOW w AS (PARTITION BY client_id ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING))
 SELECT *,date_diff('month',last_dialog,month)=1 dialog_recent FROM h
 WHERE y IS NOT NULL AND y2 IS NOT NULL AND y3 IS NOT NULL""").df()
d['dialog_recent'] = d.dialog_recent.fillna(False)
d['trx_band'] = pd.cut(d.trx_hist.fillna(0),[-1,0,9,99,float('inf')],labels=['0','1-9','10-99','100+'])
d['source_count'] = d[['trx_hist','geo_hist','dialog_hist']].fillna(0).gt(0).sum(axis=1)
checks['common_cohort'] = len(d)==n*10 and d.groupby('month').client_id.nunique().eq(n).all()
for h in [1,2,3]:
    d[f'h{h}'] = d[['y','y2','y3'][:h]].max(axis=1)
rows=[]
for feature in ['dialog_recent','trx_band','source_count']:
    for (month,group),a in d.groupby(['month',feature],observed=True):
        baseline=d.loc[d.month.eq(month)]
        for h in [1,2,3]:
            buyers=int(a[f'h{h}'].sum()); total=int(baseline[f'h{h}'].sum())
            rows.append(dict(feature=feature,month=month,group=str(group),horizon=h,n=len(a),buyers=buyers,rate=buyers/len(a),population_share=len(a)/len(baseline),buyer_share=buyers/total,lift=(buyers/len(a))/(total/len(baseline))))
segments=pd.DataFrame(rows)
segments.to_csv(OUT/'monthly_segment_cards.csv',index=False)
pooled=segments.groupby(['feature','group','horizon'],as_index=False).agg(n=('n','sum'),buyers=('buyers','sum'),min_monthly_lift=('lift','min'),max_monthly_lift=('lift','max'))
pooled['rate']=pooled.buyers/pooled.n
pooled['population_share']=pooled.n/len(d)
totals={h:int(d[f'h{h}'].sum()) for h in [1,2,3]}
pooled['buyer_share']=[b/totals[h] for b,h in zip(pooled.buyers,pooled.horizon)]
pooled['lift']=pooled.rate/[totals[h]/len(d) for h in pooled.horizon]
pooled.to_csv(OUT/'segment_cards.csv',index=False)
checks['all_segments_reconcile']=bool(segments.groupby(['feature','month','horizon']).n.sum().eq(n).all())
checks={k:bool(v) for k,v in checks.items()}
(OUT/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
assert all(checks.values()),checks
print(json.dumps(checks,indent=2),flush=True)
print(pooled[pooled.horizon.eq(1)].to_string(index=False),flush=True)
