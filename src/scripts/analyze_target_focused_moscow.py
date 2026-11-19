"""Target-first, full-population EDA from Moscow-month raw-log aggregates."""
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/full_population_temporal_eda_20261118'
OUT = ROOT / 'outputs/target_focused_moscow_20261119'
OUT.mkdir(exist_ok=True)
c = duckdb.connect()
c.execute("SET memory_limit='4GB'")
c.execute("SET threads=2")
c.execute(f"SET temp_directory='{OUT.as_posix()}/tmp'")
c.execute(f"CREATE VIEW p AS SELECT * FROM read_parquet('{(BASE/'all_customer_months.parquet').as_posix()}')")
# Complete history through the previous Moscow calendar month; source absence is explicit.
q = """
WITH h AS (
 SELECT client_id, month, y,
  sum(coalesce(trx_events,0)) OVER w AS trx_hist,
  sum(coalesce(geo_events,0)) OVER w AS geo_hist,
  sum(coalesce(dialog_events,0)) OVER w AS dialog_hist,
  sum(coalesce(amount11,0)) OVER w AS amount11_hist,
  max(CASE WHEN trx_events>0 THEN month END) OVER w AS last_trx_month,
  max(CASE WHEN geo_events>0 THEN month END) OVER w AS last_geo_month,
  max(CASE WHEN dialog_events>0 THEN month END) OVER w AS last_dialog_month,
  lead(y,1) OVER (PARTITION BY client_id ORDER BY month) AS y2,
  lead(y,2) OVER (PARTITION BY client_id ORDER BY month) AS y3
 FROM p WINDOW w AS (PARTITION BY client_id ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT *, date_diff('month',last_trx_month,month) AS trx_recency_months,
 date_diff('month',last_geo_month,month) AS geo_recency_months,
 date_diff('month',last_dialog_month,month) AS dialog_recency_months
FROM h WHERE y IS NOT NULL AND y2 IS NOT NULL AND y3 IS NOT NULL
"""
d = c.execute(q).df()
d['y1'] = d.y.astype('int8')
d['yh2'] = (d.y1 | d.y2.astype('int8'))
d['yh3'] = (d.yh2 | d.y3.astype('int8'))
# Raw target labels remain on their original Moscow calendar month; mon is a
# month identifier and must not be timezone shifted.
target_path=(ROOT/'data/MBD-mini/targets/**/*.parquet').as_posix()
products=c.execute(f"SELECT client_id, date_trunc('month',cast(mon AS DATE)) AS \"month\", target_1,target_2,target_3,target_4 FROM read_parquet('{target_path}')").df()
assert not products.duplicated(['client_id','month']).any()
d=d.merge(products,on=['client_id','month'],how='left',validate='one_to_one')
assert d[[f'target_{i}' for i in range(1,5)]].notna().all().all()
n_clients = d.client_id.nunique()
n_months = d.month.nunique()
assert len(d) == n_clients*n_months and n_months == 10
assert d.groupby('month').client_id.nunique().eq(n_clients).all()
assert (d[['y1','yh2','yh3']].sum(axis=0).diff().dropna() >= 0).all()

rows=[]
def add(feature, band, mask):
    z=d.loc[mask]
    row={'feature':feature,'group':band,'n':len(z),'customers':z.client_id.nunique()}
    for h in (1,2,3):
        row[f'buyers_h{h}']=int(z[f'yh{h}' if h>1 else 'y1'].sum())
        row[f'rate_h{h}']=row[f'buyers_h{h}']/len(z) if len(z) else np.nan
    rows.append(row)

for col in ['trx_hist','geo_hist','dialog_hist']:
    x=d[col].fillna(0)
    groups=[('0',x.eq(0)),('1–9',x.between(1,9)),('10–99',x.between(10,99)),('100+',x.ge(100))]
    assert sum(int(m.sum()) for _,m in groups)==len(d)
    for name,mask in groups:add(col,name,mask)
for col in ['trx_recency_months','geo_recency_months','dialog_recency_months']:
    x=d[col]
    groups=[('no prior log',x.isna()),('previous month',x.eq(1)),('2–3 months',x.between(2,3)),('4+ months',x.ge(4))]
    assert sum(int(m.sum()) for _,m in groups)==len(d)
    for name,mask in groups:add(col,name,mask)
presence=(d[['trx_hist','geo_hist','dialog_hist']].fillna(0)>0).sum(axis=1)
for k in range(4):add('number_of_sources',str(k),presence.eq(k))
rates=pd.DataFrame(rows)
base={h:float(d[f'yh{h}' if h>1 else 'y1'].mean()) for h in (1,2,3)}
for h in (1,2,3):rates[f'lift_h{h}']=rates[f'rate_h{h}']/base[h]
rates.to_csv(OUT/'target_group_rates.csv',index=False)

# Check nonlinear threshold and rank association on the same rows, without sampling.
assoc=[]
for col in ['trx_hist','geo_hist','dialog_hist','amount11_hist']:
    x=np.log1p(d[col].fillna(0).clip(lower=0))
    for h in (1,2,3):
        y=d[f'yh{h}' if h>1 else 'y1']
        assoc.append({'feature':col,'horizon':h,'pearson_log1p':x.corr(y),
                      'spearman':d[col].fillna(0).corr(y,method='spearman')})
pd.DataFrame(assoc).to_csv(OUT/'global_associations.csv',index=False)
month=d.groupby('month').agg(n=('client_id','size'),buyers_h1=('y1','sum'),buyers_h2=('yh2','sum'),buyers_h3=('yh3','sum')).reset_index()
for h in (1,2,3):month[f'rate_h{h}']=month[f'buyers_h{h}']/month.n
month.to_csv(OUT/'monthly_baseline.csv',index=False)
# Hypothesis checks that keep all rows in exhaustive, mutually exclusive groups.
d['dialog_recent'] = d.dialog_recency_months.eq(1).fillna(False)
d['trx_state'] = np.select([d.trx_hist.fillna(0).eq(0),d.trx_recency_months.eq(1).fillna(False)],
                           ['no history','active previous month'],default='older history')
d['geo_state'] = np.select([d.geo_hist.fillna(0).eq(0),d.geo_recency_months.eq(1).fillna(False)],
                           ['no history','active previous month'],default='older history')
def grouped(frame, keys, filename):
    z=frame.groupby(keys,dropna=False,observed=True).agg(n=('y1','size'),buyers_h1=('y1','sum'),
        buyers_h2=('yh2','sum'),buyers_h3=('yh3','sum')).reset_index()
    assert z.n.sum()==len(frame)
    for h in (1,2,3):
        z[f'rate_h{h}']=z[f'buyers_h{h}']/z.n
        z[f'lift_h{h}']=z[f'rate_h{h}']/base[h]
    z.to_csv(OUT/filename,index=False)
    return z
interaction=grouped(d,['trx_state','dialog_recent'],'dialog_by_trx_state.csv')
geo_interaction=grouped(d,['geo_state','dialog_recent'],'dialog_by_geo_state.csv')
monthly_dialog=grouped(d,['month','dialog_recent'],'dialog_by_month.csv')
d['trx_band']=pd.cut(d.trx_hist.fillna(0),[-1,0,9,99,np.inf],labels=['0','1-9','10-99','100+'])
d['geo_band']=pd.cut(d.geo_hist.fillna(0),[-1,0,9,99,np.inf],labels=['0','1-9','10-99','100+'])
trx_geo=grouped(d,['trx_band','geo_band'],'trx_geo_interaction.csv')
# Currency 11 monetary values are not mixed with other currencies. Negative
# cumulative sums are retained as their own state rather than clipped to zero.
a=d.amount11_hist.fillna(0)
d['amount11_band']=np.select([a.lt(0),a.eq(0),a.between(0,1e5,inclusive='neither'),
                              a.between(1e5,1e7,inclusive='both'),a.gt(1e7)],
                             ['negative','zero','(0,100k)','100k-10m','>10m'],default='unclassified')
amount=grouped(d,['amount11_band'],'amount11_target.csv')
# Association among predictors helps identify overlapping signals.
pred=np.log1p(d[['trx_hist','geo_hist','dialog_hist']].fillna(0).clip(lower=0))
pred.corr(method='spearman').to_csv(OUT/'predictor_rank_correlations.csv')
# Product-specific associations, preserving customers with multiple products.
def product_table(keys, filename):
    z=d.groupby(keys,dropna=False,observed=True).agg(n=('y1','size'),
        **{f'buyers_p{i}':(f'target_{i}','sum') for i in range(1,5)}).reset_index()
    assert z.n.sum()==len(d)
    for i in range(1,5):z[f'rate_p{i}']=z[f'buyers_p{i}']/z.n
    z.to_csv(OUT/filename,index=False)
    return z
product_rates=product_table(['dialog_recent'],'dialog_by_product.csv')
product_table(['trx_state','dialog_recent'],'dialog_trx_by_product.csv')
product_table(['month','dialog_recent'],'dialog_product_by_month.csv')
# Previous observed target months are known at cutoff; the first label month
# has unknown purchase history and is retained in a separate group.
prior=c.execute(f"""
 SELECT client_id,month,
  sum(y) OVER (PARTITION BY client_id ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prior_observed_buy_months
 FROM read_parquet('{(BASE/'labels_any.parquet').as_posix()}')
""").df()
d=d.merge(prior,on=['client_id','month'],how='left',validate='one_to_one')
d['prior_purchase_state']=np.select([d.prior_observed_buy_months.isna(),d.prior_observed_buy_months.eq(0).fillna(False)],
    ['unknown before label start','no observed prior purchase'],default='observed prior purchase')
grouped(d,['prior_purchase_state','dialog_recent'],'dialog_by_prior_purchase.csv')
fig,ax=plt.subplots(figsize=(8,4.5))
xs=np.arange(4)
for flag,offset,color,label in [(False,-.18,'#94a3b8','No DIALOG previous month'),
                                (True,.18,'#0f766e','DIALOG previous month')]:
    z=product_rates[product_rates.dialog_recent.eq(flag)].iloc[0]
    ax.bar(xs+offset,[z[f'rate_p{i}']*100 for i in range(1,5)],width=.36,color=color,label=label)
ax.set_xticks(xs,[f'Product {i}' for i in range(1,5)])
ax.set_ylabel('Purchased in next month (%)');ax.set_title('DIALOG association by product')
ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'dialog_product_rates.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.5))
order=['no history','older history','active previous month']
for flag,color,label in [(False,'#94a3b8','No DIALOG previous month'),
                         (True,'#0f766e','DIALOG previous month')]:
    z=interaction[interaction.dialog_recent.eq(flag)].set_index('trx_state').loc[order]
    ax.plot(order,z.rate_h1*100,marker='o',linewidth=2,color=color,label=label)
ax.set_ylabel('Purchased in next month (%)');ax.set_title('DIALOG association within TRX states')
ax.legend(frameon=False);ax.grid(alpha=.2);fig.tight_layout()
fig.savefig(OUT/'dialog_trx_interaction.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5))
for state,part in monthly_dialog.groupby('dialog_recent'):
    ax.plot(part.month,part.rate_h1*100,marker='o',label='DIALOG previous month' if state else 'Other customers')
ax.set_ylabel('Purchased in next month (%)');ax.set_title('DIALOG relationship by cutoff month')
ax.legend(frameon=False);ax.grid(alpha=.2);fig.autofmt_xdate();fig.tight_layout()
fig.savefig(OUT/'dialog_monthly_stability.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5))
v=rates[rates.feature.eq('dialog_recency_months')]
labels=['No prior log','Previous month','2-3 months','4+ months']
ax.bar(labels,v.rate_h1.to_numpy()*100,color=['#94a3b8','#0f766e','#5ba69d','#a5cbc6'])
ax.axhline(base[1]*100,color='#b45309',linestyle='--',label=f'All customers: {base[1]*100:.2f}%')
for i,row in enumerate(v.itertuples()):ax.text(i,row.rate_h1*100+.12,f'{row.rate_h1*100:.2f}%\nn={row.n:,}',ha='center',fontsize=9)
ax.set_ylim(0,5.3);ax.set_ylabel('Purchased in next month (%)');ax.set_title('Prior DIALOG activity and next-month purchase')
ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'dialog_recency_target.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5))
v=rates[rates.feature.eq('trx_hist')]
ax.bar(['0','1-9','10-99','100+'],v.rate_h1.to_numpy()*100,color='#2563eb')
ax.axhline(base[1]*100,color='#b45309',linestyle='--',label=f'All customers: {base[1]*100:.2f}%')
for i,row in enumerate(v.itertuples()):ax.text(i,row.rate_h1*100+.05,f'{row.rate_h1*100:.2f}%\nn={row.n:,}',ha='center',fontsize=9)
ax.set_ylim(0,2);ax.set_ylabel('Purchased in next month (%)');ax.set_xlabel('All prior TRX events');ax.set_title('TRX relationship is non-monotonic')
ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'trx_threshold_target.png',dpi=170);plt.close(fig)
print('cohort',n_clients,n_months,len(d),'baseline',base)
print(rates[['feature','group','n','rate_h1','lift_h1','rate_h2','lift_h2','rate_h3','lift_h3']].to_string(index=False))
print(pd.DataFrame(assoc).to_string(index=False))
