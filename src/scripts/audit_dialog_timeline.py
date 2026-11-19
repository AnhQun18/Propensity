"""Audit observed (anonymized) dialogue calendar, without inferring real dates.

Run from repository root: .venv/Scripts/python.exe src/scripts/audit_dialog_timeline.py
"""
from pathlib import Path
import json
import duckdb
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT = Path('outputs/dialog_temporal_audit')
OUT.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()
events = con.execute("SELECT client_id,event_time,fold FROM read_parquet('data/MBD-mini/detail/dialog/**/*.parquet', hive_partitioning=true)").df()
con.close()
events['event_time'] = pd.to_datetime(events.event_time)
assert events.event_time.notna().all()
events['date'] = events.event_time.dt.normalize()
calendar = pd.date_range(events.date.min(), events.date.max(), freq='D')
daily = events.groupby('date').agg(dialogs=('client_id','size'), clients=('client_id','nunique')).reindex(calendar, fill_value=0)
daily.index.name = 'date'
daily['mean_7d'] = daily.dialogs.rolling(7, min_periods=7).mean()
daily['mean_28d'] = daily.dialogs.rolling(28, min_periods=28).mean()
daily.to_csv(OUT/'daily.csv')
# Exclude the partial starting month from calendar-day exposure comparisons.
full = daily.loc['2021-03-01':].copy()
full['weekday'] = full.index.day_of_week
weekday = full.groupby('weekday').agg(calendar_days=('dialogs','size'), dialogs=('dialogs','sum'), dialogs_per_day=('dialogs','mean'))
weekday.to_csv(OUT/'weekday_normalized.csv')
monthly = daily.resample('MS').agg(dialogs=('dialogs','sum'), observed_calendar_days=('dialogs','size'), zero_days=('dialogs',lambda s: (s==0).sum()))
monthly['dialogs_per_day'] = monthly.dialogs / monthly.observed_calendar_days
monthly['rate_change_pct'] = monthly.dialogs_per_day.pct_change()*100
monthly['active_clients'] = events.groupby(events.date.dt.to_period('M')).client_id.nunique().to_numpy()
monthly.to_csv(OUT/'monthly.csv')
fold = events.groupby([events.date.dt.to_period('M'), 'fold']).size().unstack(fill_value=0)
fold.to_csv(OUT/'monthly_by_fold.csv')
comparisons=[]
for cutoff in ['2022-01-01','2022-06-01']:
    t=pd.Timestamp(cutoff)
    before=daily.loc[t-pd.Timedelta(days=28):t-pd.Timedelta(days=1),'dialogs']
    after=daily.loc[t:t+pd.Timedelta(days=27),'dialogs']
    comparisons.append(dict(boundary=cutoff, before_28d_mean=float(before.mean()), after_28d_mean=float(after.mean()), change_pct=float((after.mean()/before.mean()-1)*100)))
weekend=full.loc[full.weekday>=5,'dialogs'].mean()
workday=full.loc[full.weekday<5,'dialogs'].mean()
summary=dict(rows=len(events), clients=int(events.client_id.nunique()), first=str(events.event_time.min()),last=str(events.event_time.max()), calendar_days=len(daily), zero_days=int((daily.dialogs==0).sum()), min_daily=int(daily.dialogs.min()), max_daily=int(daily.dialogs.max()), weekend_per_day=float(weekend), weekday_per_day=float(workday), weekend_weekday_ratio=float(weekend/workday), comparisons=comparisons, source='https://arxiv.org/html/2409.17587v2', interpretation='Dates are anonymized with random noise; original hour preserved. Calendar weekdays and exact real-world breaks are not verified. First month is partial. UTC/local timezone is not verified.')
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#193454','axes.labelcolor':'#193454'})
fig, axes=plt.subplots(3,1,figsize=(13,12),dpi=170,gridspec_kw={'height_ratios':[.8,1.3,1]})
fig.subplots_adjust(left=.09,right=.96,top=.88,bottom=.12,hspace=.48)
fig.suptitle('Kiểm tra lịch thời gian hội thoại',fontsize=24,fontweight='bold',x=.09,ha='left',y=.97)
fig.text(.09,.925,'217.205 hội thoại · Ngày/giờ theo dữ liệu đã ẩn danh · Không quy đổi múi giờ',color='#62738a')
ax=axes[0]
ax.bar(range(7),weekday.dialogs_per_day,color=['#78a6dd']*5+['#368b7c']*2,width=.6)
ax.set_xticks(range(7),['Thứ 2','Thứ 3','Thứ 4','Thứ 5','Thứ 6','Thứ 7','Chủ nhật'])
for i,v in enumerate(weekday.dialogs_per_day): ax.text(i,v+3,f'{v:.1f}',ha='center',fontsize=10)
ax.set_ylim(0,weekday.dialogs_per_day.max()*1.2)
ax.set_ylabel('Hội thoại / ngày')
ax.set_title('Chuẩn hóa theo số ngày từng thứ · 03/2021–12/2022',fontweight='bold')
ax=axes[1]
ax.plot(daily.index,daily.dialogs,color='#b4c8e1',lw=.7,label='Từng ngày')
ax.plot(daily.index,daily.mean_28d,color='#2868b3',lw=2,label='Trung bình 28 ngày (lùi)')
ax.set_ylabel('Hội thoại / ngày'); ax.set_title('Chuỗi ngày đầy đủ · Không có ngày trống trong 690 ngày quan sát',fontweight='bold'); ax.legend(loc='upper left',frameon=False)
ax=axes[2]
ax.plot(monthly.index,monthly.dialogs_per_day,'o-',color='#2868b3',lw=2,ms=4)
ax.set_ylabel('Hội thoại / ngày'); ax.set_title('Trung bình ngày từng tháng · Loại ảnh hưởng tháng dài/ngắn',fontweight='bold')
for ax in axes[1:]:
    ax.axvspan(pd.Timestamp('2022-01-01'),pd.Timestamp('2022-06-01'),color='#edb568',alpha=.15,label='01–05/2022')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.set_ylim(bottom=0)
for ax in axes:
    ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
    for spine in ['top','right']: ax.spines[spine].set_visible(False)
fig.text(.09,.062,'Vùng tô: 01–05/2022. Tháng 02/2021 chỉ có dữ liệu từ 10/02 nên là tháng không đầy đủ.',fontsize=10,color='#62738a')
fig.text(.09,.032,'Nhà phát hành thêm nhiễu vào ngày, giữ giờ gốc: không dùng thứ/ngày này để kết luận lịch làm việc thực tế.',fontsize=10,color='#62738a')
fig.savefig(OUT/'dialog_timeline_audit.png')
plt.close(fig)
print(json.dumps(summary,ensure_ascii=False,indent=2))
print(monthly.loc['2021-12':'2022-06'].to_string())
print(fold.loc['2021-12':'2022-06'].to_string())
