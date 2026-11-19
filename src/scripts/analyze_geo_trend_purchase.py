"""Descriptive point-in-time GEO trend vs complete 1/2/3-month any-product labels."""
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

out=Path('outputs/geo_trend_purchase')
out.mkdir(parents=True,exist_ok=True)
c=duckdb.connect()
c.execute("CREATE TEMP TABLE labels AS SELECT client_id,date_trunc('month',cast(mon AS DATE))::DATE AS cutoff, target_1,target_2,target_3,target_4 FROM read_parquet('data/MBD-mini/targets/**/*.parquet')")
assert c.execute('SELECT count(*) FROM (SELECT client_id,cutoff,count(*) n FROM labels GROUP BY 1,2 HAVING n<>1)').fetchone()[0]==0
assert c.execute('SELECT count(*) FROM labels WHERE target_1 IS NULL OR target_2 IS NULL OR target_3 IS NULL OR target_4 IS NULL').fetchone()[0]==0
labels=c.execute('''WITH monthly AS (SELECT client_id,cutoff,
 greatest(target_1,target_2,target_3,target_4)::INT y FROM labels)
 SELECT a.client_id,a.cutoff,a.y y_1,greatest(a.y,b.y) y_2,greatest(a.y,b.y,d.y) y_3
 FROM monthly a JOIN monthly b ON a.client_id=b.client_id AND b.cutoff=a.cutoff+INTERVAL '1 month'
 JOIN monthly d ON a.client_id=d.client_id AND d.cutoff=a.cutoff+INTERVAL '2 months' ''').df()
features=pd.concat([pd.read_parquet(p) for p in sorted(Path('outputs/geo_frequency_recency').glob('cutoff_*.parquet'))],ignore_index=True)
features.cutoff=pd.to_datetime(features.cutoff)
df=features.merge(labels,on=['client_id','cutoff'],how='inner',validate='one_to_one')
assert len(df)==100224*10
assert not df.geo_30d_source_truncated.any()
df['ratio_7_30']=df.geo_7d_events/df.geo_30d_events.replace(0,np.nan)
df['intensity_index']=df.ratio_7_30*30/7
df['recent_vs_prior23']= (df.geo_7d_events/7)/((df.geo_30d_events-df.geo_7d_events)/23).replace(0,np.nan)
df['prior23_zero_recent_positive']=(df.geo_30d_events==df.geo_7d_events)&(df.geo_7d_events>0)
df['trend_band']=np.select([df.geo_30d_events.eq(0),df.intensity_index.eq(0),df.intensity_index.lt(.5),df.intensity_index.lt(1),df.intensity_index.lt(1.5)],['No GEO 30d','No GEO 7d, active prior23','0 < index < 0.5','0.5 <= index < 1','1 <= index < 1.5'],default='index >= 1.5')
df.to_parquet(out/'customer_month_trend_labels.parquet',index=False)
groups=[];bands=[];months=[]
for h in [1,2,3]:
    target=f'y_{h}'; base=df[target].mean()
    for y,g in df.groupby(target):
        valid=g.intensity_index.dropna()
        groups.append(dict(horizon=h,buyer=int(y),customer_months=len(g),valid_ratio=len(valid),no_geo30_pct=g.geo_30d_events.eq(0).mean()*100,
            ratio_mean=g.ratio_7_30.mean(),ratio_median=g.ratio_7_30.median(),index_mean=valid.mean(),index_median=valid.median(),index_p90=valid.quantile(.9),
            index_ge1_pct=valid.ge(1).mean()*100,baseline_rate=base))
    b=df.groupby('trend_band')[target].agg(['size','sum','mean']).reset_index().rename(columns={'size':'customer_months','sum':'buyers','mean':'purchase_rate'})
    b['horizon']=h;b['baseline_rate']=base;b['lift']=b.purchase_rate/base;bands.append(b)
    m=df.groupby(['cutoff',target]).agg(customer_months=('client_id','size'),index_mean=('intensity_index','mean'),index_median=('intensity_index','median'),valid_ratio=('intensity_index','count')).reset_index().rename(columns={target:'buyer'})
    m['horizon']=h;months.append(m)
s=pd.DataFrame(groups);s.to_csv(out/'buyer_nonbuyer_summary.csv',index=False)
b=pd.concat(bands);b.to_csv(out/'trend_band_purchase.csv',index=False)
pd.concat(months).to_csv(out/'monthly_buyer_nonbuyer.csv',index=False)
order=['No GEO 30d','No GEO 7d, active prior23','0 < index < 0.5','0.5 <= index < 1','1 <= index < 1.5','index >= 1.5']
fig,axes=plt.subplots(1,2,figsize=(14,6),dpi=170)
for buyer,color in [(0,'#8293a8'),(1,'#208575')]:
    d=s[s.buyer==buyer]
    axes[0].bar(np.arange(3)+(buyer-.5)*.32,d.index_mean,width=.32,color=color,label=['Non-buyer','Buyer'][buyer])
axes[0].set_xticks(range(3),['1 month','2 months','3 months']);axes[0].axhline(1,color='grey',ls='--',lw=1)
axes[0].set_ylabel('Mean normalized index (GEO 30d > 0 only)');axes[0].legend();axes[0].set_title('Recent GEO intensity by future purchase')
for h,color in zip([1,2,3],['#2864a0','#208575','#c17b33']):
    d=b[b.horizon==h].set_index('trend_band').reindex(order)
    axes[1].bar(np.arange(6)+(h-2)*.25,d.purchase_rate,width=.23,color=color,label=f'{h} month(s)')
axes[1].set_xticks(range(6),['No GEO\n30d','No GEO 7d\nGEO 30d > 0','(0, 0.5)','[0.5, 1)','[1, 1.5)','>=1.5'])
axes[1].set_ylim(bottom=0)
axes[1].set_xlabel('Customer groups by normalized intensity index');axes[1].set_ylabel('Cumulative purchase rate');axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].legend();axes[1].set_title('Purchase rate by customer group')
for ax in axes: ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('GEO 7d / 30d trend | common cohort | Feb–Nov 2022 cutoffs',fontsize=17)
fig.tight_layout(rect=(0,.07,1,.93))
fig.text(.02,.015,'Index = (events 7d / 7) / (events 30d / 30). 1 = equal daily intensity. Descriptive pooled customer-months; no causal inference.',fontsize=10)
fig.savefig(out/'trend_purchase.png')
assert df.ratio_7_30.dropna().between(0,1).all()
assert (df.y_1<=df.y_2).all() and (df.y_2<=df.y_3).all()
print(s.to_string(index=False));print(b.to_string(index=False))
