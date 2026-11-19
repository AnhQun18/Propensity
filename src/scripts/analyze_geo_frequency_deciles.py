"""Purchase rates by descriptive GEO event-count decile within each cutoff."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT=Path('outputs/geo_frequency_deciles')
OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_parquet('outputs/geo_trend_purchase/customer_month_trend_labels.parquet')
assert len(df)==1002240 and not df.geo_30d_source_truncated.any()
df['decile']=0
edges=[]
for cutoff,part in df.groupby('cutoff'):
    positive=part.index[part.geo_30d_events>0]
    bins,bounds=pd.qcut(df.loc[positive,'geo_30d_events'],10,labels=False,retbins=True,duplicates='raise')
    df.loc[positive,'decile']=bins+1
    for i in range(10):
        edges.append(dict(cutoff=cutoff,decile=i+1,lower_edge=bounds[i],upper_edge=bounds[i+1],lower_inclusive=i==0))
pd.DataFrame(edges).to_csv(OUT/'decile_edges_by_cutoff.csv',index=False)
df[['client_id','cutoff','geo_30d_events','decile','y_1','y_2','y_3']].to_parquet(OUT/'customer_month_deciles.parquet',index=False)
tables=[];monthly=[]
for h in [1,2,3]:
    y=f'y_{h}'
    t=df.groupby('decile').agg(customer_months=('client_id','size'),buyers=(y,'sum'),purchase_rate=(y,'mean'),
        min_events=('geo_30d_events','min'),median_events=('geo_30d_events','median'),max_events=('geo_30d_events','max')).reset_index()
    t['horizon']=h
    t['baseline_all']=df[y].mean()
    t['baseline_geo_positive']=df.loc[df.decile>0,y].mean()
    t['lift_all']=t.purchase_rate/t.baseline_all
    t['lift_geo_positive']=t.purchase_rate/t.baseline_geo_positive
    tables.append(t)
    m=df.groupby(['cutoff','decile']).agg(customer_months=('client_id','size'),buyers=(y,'sum'),purchase_rate=(y,'mean')).reset_index()
    m['horizon']=h
    monthly.append(m)
t=pd.concat(tables,ignore_index=True)
t.to_csv(OUT/'decile_purchase_rates.csv',index=False)
m=pd.concat(monthly,ignore_index=True)
m.to_csv(OUT/'monthly_decile_purchase_rates.csv',index=False)
fig,ax=plt.subplots(figsize=(12,6.7),dpi=180)
colors=['#2864a0','#208575','#c17b33']
for h,color in zip([1,2,3],colors):
    d=t[t.horizon==h]
    ax.plot(d.decile,d.purchase_rate,marker='o',color=color,label=f'{h} month(s)')
    ax.axhline(d.baseline_all.iloc[0],color=color,ls='--',lw=.8,alpha=.65)
ax.set_xticks(range(11),['No GEO\n30d']+[f'D{i}' for i in range(1,11)])
ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=1))
ax.set_ylabel('Cumulative any-product purchase rate')
ax.set_xlabel('30-day GEO event-count decile (D1 = lowest positive, D10 = highest)')
ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.legend(title='Future label horizon')
fig.suptitle('Purchase rate by GEO frequency decile',fontsize=20)
ax.set_title('Common cohort: 1,002,240 customer-months | Feb–Nov 2022 cutoffs',fontsize=11,pad=15)
fig.tight_layout(rect=(0,.08,1,.93))
fig.text(.04,.025,'Deciles within each cutoff among GEO-positive customers; ties kept together. Dashed lines: all-cohort baselines. Descriptive association.',fontsize=10)
fig.savefig(OUT/'geo_decile_purchase.png')
assert (df.loc[df.geo_30d_events.eq(0),'decile']==0).all()
for h in [1,2,3]:
    z=t[t.horizon==h]
    assert z.customer_months.sum()==len(df) and z.buyers.sum()==df[f'y_{h}'].sum()
print(t[t.horizon==1][['decile','customer_months','median_events']].to_string(index=False))
print(t.pivot(index='decile',columns='horizon',values='purchase_rate').to_string())
print(t[t.decile==10][['horizon','purchase_rate','baseline_all','baseline_geo_positive','lift_all','lift_geo_positive']].to_string(index=False))
print('D10 vs D1 monthly direction:',m[m.decile.isin([1,10])].pivot(index=['horizon','cutoff'],columns='decile',values='purchase_rate').groupby(level=0).apply(lambda x:int((x[10]>x[1]).sum())).to_dict())
