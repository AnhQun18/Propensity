"""Point-in-time geo frequency/active days and full-history recency."""
from pathlib import Path
import duckdb
import pandas as pd

out=Path('outputs/geo_frequency_recency')
out.mkdir(parents=True,exist_ok=True)
c=duckdb.connect()
c.execute('SET threads=4')
c.execute("SET memory_limit='4GB'")
c.execute("CREATE VIEW geo AS SELECT * FROM read_parquet('data/MBD-mini/detail/geo/**/*.parquet')")
c.execute("CREATE TEMP TABLE clients AS SELECT DISTINCT client_id FROM read_parquet('data/MBD-mini/client_split/**/*.parquet')")
c.execute("CREATE TEMP TABLE daily AS SELECT client_id,cast(event_time AS DATE) AS day,count(*) events,max(event_time) last_event FROM geo GROUP BY 1,2")
start=c.execute('SELECT min(day) FROM daily').fetchone()[0]
cutoffs=c.execute("SELECT DISTINCT date_trunc('month',cast(mon AS DATE))::DATE FROM read_parquet('data/MBD-mini/targets/**/*.parquet') ORDER BY 1").fetchall()
files=[]
for (cutoff,) in cutoffs:
    fields=['sum(events) geo_all_events','count(*) geo_all_active_days',
            f"epoch(DATE '{cutoff}' - max(last_event))/86400.0 geo_recency_days"]
    for w in [7,30,90]:
        fields.extend([f"sum(events) FILTER(WHERE day>=DATE '{cutoff}'-INTERVAL '{w} days') geo_{w}d_events",
            f"count(*) FILTER(WHERE day>=DATE '{cutoff}'-INTERVAL '{w} days') geo_{w}d_active_days"])
    c.execute(f"CREATE OR REPLACE TEMP TABLE agg AS SELECT client_id,{','.join(fields)} FROM daily WHERE day<DATE '{cutoff}' GROUP BY client_id")
    cols=["coalesce(geo_all_events,0) geo_all_events","coalesce(geo_all_active_days,0) geo_all_active_days","geo_recency_days", "CAST(geo_recency_days IS NULL AS INTEGER) geo_never_seen_before_cutoff"]
    for w in [7,30,90]:
        cols.extend([f'coalesce(geo_{w}d_events,0) geo_{w}d_events',f'coalesce(geo_{w}d_active_days,0) geo_{w}d_active_days',
            f"DATE '{start}' > DATE '{cutoff}'-INTERVAL '{w} days' geo_{w}d_source_truncated"])
    path=out/f'cutoff_{cutoff}.parquet'
    c.execute(f"COPY (SELECT client_id,DATE '{cutoff}' cutoff,{','.join(cols)} FROM clients LEFT JOIN agg USING(client_id)) TO '{path.as_posix()}' (FORMAT PARQUET)")
    files.append(str(path))
    print(cutoff,flush=True)
df=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
rows=[]
metrics=['geo_all_events','geo_all_active_days']+[f'geo_{w}d_{m}' for w in [7,30,90] for m in ['events','active_days']]+['geo_recency_days']
for col in metrics:
    x=df[col].dropna()
    rows.append(dict(feature=col,n=len(x),missing=int(df[col].isna().sum()),zero_pct=(x==0).mean()*100,
        mean=x.mean(),median=x.median(),p90=x.quantile(.9),p99=x.quantile(.99),maximum=x.max()))
pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
monthly=df.groupby('cutoff').agg(customer_months=('client_id','size'),never_seen=('geo_never_seen_before_cutoff','sum'))
for w in [7,30,90]:
    monthly[f'has_geo_{w}d_pct']=df.groupby('cutoff')[f'geo_{w}d_events'].apply(lambda x:(x>0).mean()*100)
    monthly[f'source_truncated_{w}d']=df.groupby('cutoff')[f'geo_{w}d_source_truncated'].first()
monthly.to_csv(out/'monthly_summary.csv')
assert len(df)==1202688
assert (df.geo_7d_events<=df.geo_30d_events).all() and (df.geo_30d_events<=df.geo_90d_events).all()
assert (df.geo_90d_events<=df.geo_all_events).all()
for w in [7,30,90]:
    assert df[f'geo_{w}d_active_days'].between(0,w).all()
    assert (df[f'geo_{w}d_active_days']<=df[f'geo_{w}d_events']).all()
assert (df.geo_recency_days.dropna()>0).all()
print(pd.DataFrame(rows).to_string(index=False))
print(monthly.to_string())
