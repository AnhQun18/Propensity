"""Calendar exposure-normalized raw GEO patterns using project UTC convention."""
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('outputs/geo_calendar_patterns')
out.mkdir(parents=True,exist_ok=True)
c=duckdb.connect()
c.execute('SET threads=4')
c.execute("SET memory_limit='4GB'")
c.execute("SET temp_directory='outputs/geo_calendar_patterns/duckdb_tmp'")
# Complete calendar months only; exclude partial December 2021.
c.execute("CREATE VIEW geo AS SELECT * FROM read_parquet('data/MBD-mini/detail/geo/**/*.parquet') WHERE event_time>=TIMESTAMP '2022-01-01' AND event_time<TIMESTAMP '2023-01-01'")
hourly=c.execute('''SELECT cast(event_time AS DATE) AS day,extract(hour FROM event_time)::INT AS hour,
 count(*) events,count(distinct client_id) customers FROM geo GROUP BY 1,2''').df()
hourly.day=pd.to_datetime(hourly.day)
calendar=pd.DataFrame({'day':pd.date_range('2022-01-01','2022-12-31')})
grid=calendar.merge(pd.DataFrame({'hour':range(24)}),how='cross').merge(hourly,how='left',on=['day','hour']).fillna({'events':0,'customers':0})
grid['weekday']=grid.day.dt.dayofweek
grid['dom']=grid.day.dt.day
grid['days_to_month_end']=grid.day.dt.days_in_month-grid.dom
grid.to_csv(out/'daily_hourly_counts.csv',index=False)
wh=grid.groupby(['weekday','hour']).agg(mean_events=('events','mean'),mean_customers=('customers','mean'),observed_days=('day','size')).reset_index()
wh.to_csv(out/'weekday_hour.csv',index=False)
daily=c.execute('''SELECT cast(event_time AS DATE) AS day,count(*) events,count(distinct client_id) customers FROM geo GROUP BY 1''').df()
daily.day=pd.to_datetime(daily.day)
daily=calendar.merge(daily,how='left',on='day').fillna({'events':0,'customers':0})
daily['dom']=daily.day.dt.day
daily['days_to_month_end']=daily.day.dt.days_in_month-daily.dom
daily['weekday']=daily.day.dt.dayofweek
# Relative-to-month average removes differing monthly source volume.
daily['month']=daily.day.dt.to_period('M').astype(str)
for metric in ['events','customers']:
    daily[f'{metric}_month_index']=daily[metric]/daily.groupby('month')[metric].transform('mean')
daily.to_csv(out/'daily_counts.csv',index=False)
for col in ['dom','days_to_month_end']:
    agg=daily.groupby(col).agg(observed_days=('day','size'),mean_events=('events','mean'),mean_customers=('customers','mean'),events_month_index=('events_month_index','mean'),customers_month_index=('customers_month_index','mean')).reset_index()
    agg.to_csv(out/f'{col}_summary.csv',index=False)
dom=pd.read_csv(out/'dom_summary.csv')
fig,axes=plt.subplots(2,2,figsize=(15,10),dpi=170)
for ax,metric,title in zip(axes[0],['mean_events','mean_customers'],['GEO events / calendar hour','Distinct customers / calendar hour']):
    mat=wh.pivot(index='weekday',columns='hour',values=metric)
    im=ax.imshow(mat,aspect='auto',cmap='YlGnBu')
    ax.set_yticks(range(7),['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    ax.set_xticks(range(0,24,2))
    ax.set_xlabel('Hour (UTC)')
    ax.set_title(title)
    fig.colorbar(im,ax=ax,shrink=.8)
for metric,color,label in [('events','#2864a0','Events'),('customers','#208575','Distinct customers')]:
    # Normalize means across all calendar days, not across DOM groups.
    axes[1,0].plot(dom.dom,dom[f'mean_{metric}']/daily[metric].mean(),marker='o',ms=3,color=color,label=label)
    end=pd.read_csv(out/'days_to_month_end_summary.csv').query('days_to_month_end<=9')
    axes[1,1].plot(end.days_to_month_end,end[f'mean_{metric}']/daily[metric].mean(),marker='o',color=color,label=label)
axes[1,0].set_xlabel('Day of month')
axes[1,0].set_xticks([1,5,10,15,20,25,28,29,30,31])
axes[1,0].set_title('Mean per observed calendar day / annual daily mean')
axes[1,1].set_xlabel('Days before month end (0 = last day)')
axes[1,1].invert_xaxis()
axes[1,1].set_title('Month-end alignment')
for ax in axes[1]:
    ax.axhline(1,color='grey',ls='--',lw=1)
    ax.set_ylabel('Relative daily activity')
    ax.grid(alpha=.2)
    ax.legend()
fig.suptitle('GEO calendar patterns | Jan–Dec 2022 | UTC convention',fontsize=19)
fig.tight_layout(rect=(0,.045,1,.95))
fig.text(.02,.015,'Heatmaps average over calendar occurrences. Day 31 has 7 observations; day 30: 11; day 29: 11; others: 12. No session inference.',fontsize=10)
fig.savefig(out/'geo_calendar_patterns.png')
assert daily.events.sum()==65564488
assert len(grid)==365*24 and len(daily)==365
print('Total events:',int(daily.events.sum()))
for metric in ['mean_events','mean_customers']:
    print('Peak weekday-hour',metric,wh.nlargest(3,metric).to_string(index=False))
print('Day-of-month:',dom.to_string(index=False))
print('Weekday mean:',daily.groupby('weekday')[['events','customers']].mean().to_string())
