"""Audit missing/sentinel codes and high-occupancy cells without assigning semantics."""
from pathlib import Path
import duckdb
import pandas as pd

out = Path('outputs/geo_special_values')
out.mkdir(parents=True, exist_ok=True)
c = duckdb.connect()
c.execute('SET threads=4')
c.execute("SET memory_limit='4GB'")
c.execute("CREATE VIEW geo AS SELECT * FROM read_parquet('data/MBD-mini/detail/geo/**/*.parquet')")
n, clients = c.execute('SELECT count(*),count(distinct client_id) FROM geo').fetchone()
quality, tops, details = [], [], []
for level in [4,5,6]:
    col = f'geohash_{level}'
    q = c.execute(f'''SELECT count(*) FILTER(WHERE {col} IS NULL), min({col}),max({col}),
        count(*) FILTER(WHERE {col}=0),count(*) FILTER(WHERE {col}<0),
        count(*) FILTER(WHERE {col} IN (-1,0,999,9999,99999,999999,2147483647)) FROM geo''').fetchone()
    quality.append(dict(level=f'g{level}',rows=n,nulls=q[0],minimum=q[1],maximum=q[2],zeros=q[3],negative=q[4],conventional_sentinel_rows=q[5]))
    df = pd.read_csv(f'outputs/geo_cell_sizes/g{level}_cell_counts.csv')
    sentinel = df[df.cell.isin([-1,0,1,999,9999,99999,999999,2147483647])].copy()
    sentinel.to_csv(out/f'g{level}_candidate_codes.csv', index=False)
    chosen = set()
    for metric in ['events','customers']:
        top = df.nlargest(10,metric).copy()
        top['rank_by'] = metric
        top['level'] = f'g{level}'
        top['event_pct'] = top.events/n*100
        top['customer_pct'] = top.customers/clients*100
        top['events_per_customer'] = top.events/top.customers
        tops.append(top)
        chosen.update(top.cell.tolist())
    codes = ','.join(str(int(v)) for v in sorted(chosen))
    # Per-customer concentration distinguishes breadth from repeated activity.
    d = c.execute(f'''WITH pc AS (
        SELECT {col} cell, client_id,count(*) n FROM geo WHERE {col} IN ({codes}) GROUP BY 1,2
    ) SELECT cell,max(n) max_events_one_customer,
       max(n)*100.0/sum(n) largest_customer_event_pct,
       median(n) median_events_per_customer FROM pc GROUP BY cell''').df()
    d['level']=f'g{level}'
    details.append(d)
    print(f'g{level} complete',flush=True)
pd.DataFrame(quality).to_csv(out/'quality.csv',index=False)
pd.concat(tops).merge(pd.concat(details),on=['level','cell']).to_csv(out/'top_cells.csv',index=False)
print(pd.DataFrame(quality).to_string(index=False))
t = pd.read_csv(out/'top_cells.csv')
print(t.groupby(['level','rank_by'],sort=False).head(1).to_string(index=False))
