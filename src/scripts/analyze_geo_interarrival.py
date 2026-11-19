"""Full-source descriptive interarrival times, not cutoff-specific model features."""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('outputs/geo_interarrival')
out.mkdir(parents=True,exist_ok=True)
c=duckdb.connect()
c.execute('SET threads=2')
c.execute('SET preserve_insertion_order=false')
c.execute("SET memory_limit='4GB'")
c.execute("SET temp_directory='outputs/geo_interarrival/duckdb_tmp'")
c.execute("CREATE VIEW geo AS SELECT client_id,event_time FROM read_parquet('data/MBD-mini/detail/geo/**/*.parquet')")
q='''WITH ordered AS (
 SELECT client_id,event_time,lag(event_time) OVER(PARTITION BY client_id ORDER BY event_time) prev FROM geo
), gaps AS (
 SELECT client_id,epoch(event_time-prev)/3600.0 gap_hours FROM ordered
)
SELECT client_id,count(*) events,count(gap_hours) gaps,
 count(*) FILTER(WHERE gap_hours=0) zero_gaps,
 avg(gap_hours) mean_gap_hours,median(gap_hours) median_gap_hours,
 quantile_cont(gap_hours,.9) p90_gap_hours,max(gap_hours) max_gap_hours,
 stddev_pop(gap_hours) std_gap_hours,
 count(*) FILTER(WHERE gap_hours>0) positive_gaps,
 avg(gap_hours) FILTER(WHERE gap_hours>0) positive_mean_gap_hours,
 median(gap_hours) FILTER(WHERE gap_hours>0) positive_median_gap_hours,
 stddev_pop(gap_hours) FILTER(WHERE gap_hours>0) positive_std_gap_hours
FROM gaps GROUP BY client_id'''
parts=[]
for fold in range(5):
    c.execute(f"CREATE OR REPLACE VIEW geo AS SELECT client_id,event_time FROM read_parquet('data/MBD-mini/detail/geo/fold={fold}/*.parquet')")
    parts.append(c.execute(q).df())
    print(f'event gaps fold {fold} complete',flush=True)
d=pd.concat(parts,ignore_index=True)
assert d.client_id.is_unique
c.execute("CREATE OR REPLACE VIEW geo AS SELECT client_id,event_time FROM read_parquet('data/MBD-mini/detail/geo/**/*.parquet')")
for prefix,n,mean,std in [('', 'gaps','mean_gap_hours','std_gap_hours'),('positive_','positive_gaps','positive_mean_gap_hours','positive_std_gap_hours')]:
    d[prefix+'cv']=np.where((d[n]>=2)&(d[mean]>0),d[std]/d[mean],np.nan)
    d[prefix+'burstiness']=(d[prefix+'cv']-1)/(d[prefix+'cv']+1)
d.to_csv(out/'customer_interarrival.csv',index=False)
print('event gaps complete',flush=True)
day=c.execute('''WITH days AS (SELECT DISTINCT client_id,cast(event_time AS DATE) AS day FROM geo),
 ordered AS (SELECT client_id,day-lag(day) OVER(PARTITION BY client_id ORDER BY day) gap_days FROM days)
 SELECT client_id,count(*) active_days,count(gap_days) day_gaps,avg(gap_days) mean_day_gap,
 median(gap_days) median_day_gap,stddev_pop(gap_days) std_day_gap,
 count(*) FILTER(WHERE gap_days=1) consecutive_day_gaps FROM ordered GROUP BY client_id''').df()
day['day_gap_cv']=np.where(day.day_gaps>=2,day.std_day_gap/day.mean_day_gap,np.nan)
day['consecutive_day_share']=day.consecutive_day_gaps/day.day_gaps.replace(0,np.nan)
day.to_csv(out/'customer_active_day_gaps.csv',index=False)
rows=[]
for frame,cols in [(d,['events','mean_gap_hours','median_gap_hours','p90_gap_hours','max_gap_hours','cv','positive_cv','burstiness']),
                   (day,['active_days','mean_day_gap','day_gap_cv','consecutive_day_share'])]:
    for col in cols:
        x=frame[col].dropna()
        rows.append(dict(metric=col,n=len(x),mean=x.mean(),p10=x.quantile(.1),median=x.median(),p90=x.quantile(.9),p99=x.quantile(.99),maximum=x.max()))
s=pd.DataFrame(rows)
s.to_csv(out/'summary.csv',index=False)
quality=dict(customers=len(d),events=int(d.events.sum()),gaps=int(d.gaps.sum()),zero_gaps=int(d.zero_gaps.sum()),customers_with_zero_gaps=int((d.zero_gaps>0).sum()),customers_one_event=int((d.events==1).sum()),customers_cv_available=int(d.cv.notna().sum()),customers_ge20_gaps=int((d.gaps>=20).sum()))
pd.DataFrame([quality]).to_csv(out/'quality.csv',index=False)
bins=[0,.5,1,2,np.inf]
group=pd.cut(d.cv,bins=bins,right=False).value_counts(sort=False).rename_axis('cv_range').reset_index(name='customers')
group['pct_of_valid_cv']=group.customers/d.cv.notna().sum()*100
group.to_csv(out/'cv_bands.csv',index=False)
stable=d[d.gaps>=20]
stable[['cv','median_gap_hours']].describe(percentiles=[.1,.5,.9,.99]).to_csv(out/'customers_ge20_gaps_summary.csv')
fig,axes=plt.subplots(1,3,figsize=(15,4.8),dpi=170)
for ax,x,label in [(axes[0],d.median_gap_hours,'Median event gap per customer (hours)'),(axes[1],d.cv,'Event-gap CV per customer'),(axes[2],day.consecutive_day_share,'Fraction of day gaps equal to 1 day')]:
    x=x.dropna().to_numpy(); values,freq=np.unique(x,return_counts=True)
    y=np.cumsum(freq)/len(x)
    if ax is not axes[2]:
        valid=values>0; ax.semilogx(values[valid],y[valid],color='#2864a0',lw=2)
    else: ax.plot(values,y,color='#208575',lw=2)
    ax.set_xlabel(label); ax.set_ylabel('Fraction of customers <= x'); ax.grid(alpha=.2)
fig.suptitle('GEO interarrival and regularity | full available history | UTC',fontsize=16)
fig.tight_layout(rect=(0,.07,1,.91))
fig.text(.02,.02,'CV = population standard deviation / mean; requires >=2 gaps and positive mean. Each customer has equal weight.',fontsize=10)
fig.savefig(out/'geo_interarrival.png')
assert quality['events']==66295724 and quality['gaps']==66295724-72573
assert (d.zero_gaps<=d.gaps).all()
assert (d.cv.dropna()>=0).all()
print(quality)
print(s.to_string(index=False))
print(group.to_string(index=False))
print('>=20 gaps CV median:',stable.cv.median())
