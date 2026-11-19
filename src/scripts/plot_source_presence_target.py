"""Source presence before cutoff vs next-month purchase, client-cluster bootstrap CIs."""
from pathlib import Path
import pandas as pd,numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
O=Path('outputs/source_presence_target'); O.mkdir(exist_ok=True)
a=pd.read_parquet('outputs/dataset_eda/customer_month_snapshots.parquet',columns=['client_id','y_any','geo_events','dialog_events'])
ids,ci=np.unique(a.client_id,return_inverse=True); N=len(ids); den=[]; num=[]; records=[]; positives=int(a.y_any.sum()); baseline=a.y_any.mean()
for src in ['geo','dialog']:
 for has in [True,False]:
  mask=a[src+'_events'].gt(0).to_numpy()==has; y=a.y_any.to_numpy(); dn=np.bincount(ci,weights=mask,minlength=N); nm=np.bincount(ci,weights=mask*y,minlength=N); den.append(dn); num.append(nm)
  n=int(dn.sum()); k=int(nm.sum()); records.append(dict(source=src,has_log_previous_month=has,n_customer_months=n,buyers_customer_months=k,purchase_rate=k/n,buyer_share=k/positives,lift_vs_overall=(k/n)/baseline))
den=np.stack(den,axis=1); num=np.stack(num,axis=1); rng=np.random.default_rng(42); rates=[]; lifts=[]
for b in range(2000):
 ix=rng.integers(0,N,size=N); dn=den[ix].sum(axis=0); nm=num[ix].sum(axis=0); rate=nm/dn; base=nm[:2].sum()/dn[:2].sum(); rates.append(rate); lifts.append(rate/base)
r=pd.DataFrame(records); rates=np.array(rates); lifts=np.array(lifts); r['ci_low'],r['ci_high']=np.quantile(rates,[.025,.975],axis=0); r['lift_ci_low'],r['lift_ci_high']=np.quantile(lifts,[.025,.975],axis=0); r.to_csv(O/'source_presence_purchase_ci.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'#193454','axes.labelcolor':'#193454','xtick.color':'#536781','ytick.color':'#536781'})
f,axs=plt.subplots(1,2,figsize=(14,8),dpi=220); f.subplots_adjust(left=.075,right=.96,bottom=.43,top=.75,wspace=.27)
f.text(.055,.94,'Có nguồn hành vi và tỷ lệ mua tháng sau',fontsize=25,fontweight='bold'); f.text(.055,.875,'Có/không có log tháng T−1 → y_any tháng T · Gộp 12 tháng nhãn · Đơn vị: khách–tháng',fontsize=12,color='#62738a')
fmt=lambda n:f'{int(n):,}'.replace(',','.')
for ax,src,title in zip(axs,['geo','dialog'],['GEO','DIALOG']):
 g=r[r.source==src].reset_index(drop=True); x=np.arange(2); p=g.purchase_rate.to_numpy()*100
 ax.bar(x,p,width=.5,color=['#397bc8','#b5c6db'],zorder=3)
 ax.errorbar(x,p,yerr=[p-g.ci_low*100,g.ci_high*100-p],fmt='none',ecolor='#142a46',elinewidth=1.7,capthick=1.7,capsize=7,zorder=4)
 for i,row in g.iterrows(): ax.text(i,row.ci_high*100+.10,f'{row.purchase_rate*100:.3f}%'.replace('.',','),ha='center',fontweight='bold',fontsize=13)
 ax.axhline(baseline*100,color='#8c9bb0',ls='--',lw=1.2); ax.set_ylim(0,4.5); ax.set_xlim(-.65,1.65); ax.set_xticks(x,['Có log','Không có log']); ax.set_title(title,fontsize=16,fontweight='bold',pad=15); ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0)); ax.set_ylabel('Tỷ lệ mua'); ax.grid(axis='y',color='#e8edf4'); ax.set_axisbelow(True)
 for sp in ['top','right']: ax.spines[sp].set_visible(False)
 data=[]
 for _,row in g.iterrows(): data.append([fmt(row.n_customer_months),fmt(row.buyers_customer_months),f'{row.buyer_share*100:.2f}%'.replace('.',','),f'{row.lift_vs_overall:.2f}×'.replace('.',',')])
 cells=[[lab,*vals] for lab,vals in zip(['n khách–tháng','Lượt có mua','Tỷ trọng lượt mua','Lift'],np.array(data).T.tolist())]
 tab=ax.table(cellText=cells,colLabels=['','Có log','Không có log'],colWidths=[.42,.29,.29],cellLoc='center',bbox=[0,-.83,1,.63])
 tab.auto_set_font_size(False); tab.set_fontsize(10.5)
 for (rr,cc),cell in tab.get_celld().items():
  cell.set_edgecolor('#dce5ef'); cell.set_facecolor('#edf3fb' if rr==0 else 'white')
  if rr==0: cell.set_text_props(fontweight='bold')
f.text(.055,.13,'CI 95%: bootstrap theo client_id, 2.000 lần (seed 42). Lift = tỷ lệ mua nhóm / tỷ lệ mua chung (1,069%).',fontsize=10.5,color='#62738a')
f.text(.055,.08,'Tỷ trọng lượt mua = số khách–tháng có mua của nhóm / tổng 12.854 khách–tháng có mua.',fontsize=10.5,color='#62738a')
f.text(.055,.035,'Hai nhóm có/không trong mỗi nguồn bổ sung nhau; GEO và DIALOG có thể chồng lấn. Không suy ra quan hệ nhân quả.',fontsize=10,color='#62738a')
f.savefig(O/'source_presence_purchase_ci.png')
(O/'method.json').write_text(json.dumps(dict(replicates=2000,seed=42,ci='percentile 95%, resample clients retaining all months',unit='customer-month',baseline=float(baseline),definition='source activity in T-1; target y_any in T',lift='group purchase rate / overall purchase rate'),indent=2),encoding='utf-8'); print(r.to_string(index=False))
