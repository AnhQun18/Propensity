"""Independently reconcile monthly chart against raw TRX parquet."""
from pathlib import Path
import json
import duckdb
import pandas as pd

out=Path('outputs/trx_monthly_audit'); out.mkdir(exist_ok=True)
c=duckdb.connect()
p="read_parquet('data/MBD-mini/detail/trx/**/*.parquet',hive_partitioning=true)"
m=c.execute(f"SELECT date_trunc('month',event_time) AS month,count(*) AS events,count(distinct client_id) AS active_clients FROM {p} GROUP BY 1 ORDER BY 1").df()
daily=c.execute(f"SELECT cast(event_time AS DATE) AS date,count(*) AS events FROM {p} GROUP BY 1 ORDER BY 1").df()
fold=c.execute(f"SELECT date_trunc('month',event_time) AS month,fold,count(*) AS events,count(distinct client_id) AS active_clients FROM {p} GROUP BY 1,2 ORDER BY 1,2").df()
c.close()
old=pd.read_csv('docs/report/tables/trx/trx_monthly_coverage.csv',parse_dates=['month'])
check=m.merge(old[['month','events','active_clients']],on='month',suffixes=('_raw','_report'),validate='one_to_one',how='outer',indicator=True)
assert check['_merge'].eq('both').all()
for col in ['events','active_clients']: assert check[f'{col}_raw'].eq(check[f'{col}_report']).all(),col
assert m.events.sum()==38764930
m['days']=m.month.dt.days_in_month
m['events_per_active']=m.events/m.active_clients
m['events_per_day']=m.events/m.days
m['events_per_active_calendar_day']=m.events/m.active_clients/m.days
for col in ['events','active_clients','events_per_active','events_per_day']:
    m[col+'_change_pct']=m[col].pct_change()*100
m.to_csv(out/'monthly_verified.csv',index=False)
check.to_csv(out/'reconciliation.csv',index=False)
fold.to_csv(out/'monthly_by_fold.csv',index=False)
daily.to_csv(out/'daily.csv',index=False)
calendar=pd.date_range(daily.date.min(),daily.date.max())
summary={'matched_months':len(m),'raw_rows':int(m.events.sum()),'zero_days_in_observed_span':len(calendar.difference(pd.DatetimeIndex(daily.date))), 'first_date':str(daily.date.min()), 'last_date':str(daily.date.max()),'note':'December 2020 is partial. Calendar-day rates are only meaningful for full months. Counts include original duplicate rows, matching the chart definition.'}
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
print(m.loc[m.month.between('2021-12-01','2022-03-01')].to_string(index=False))
