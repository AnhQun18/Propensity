import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path
O=Path('outputs/source_presence_target')
a=pd.read_parquet('outputs/dataset_eda/customer_month_snapshots.parquet',columns=['trx_events','geo_events','dialog_events'])
rows=[]
for src in ['trx','geo','dialog']:
 n=int(a[src+'_events'].gt(0).sum()); rows.append(dict(source=src.upper(),with_log=n,without_log=len(a)-n,total=len(a),with_log_pct=n/len(a)*100))
r=pd.DataFrame(rows); r.to_csv(O/'customer_month_log_coverage.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'#193454','axes.labelcolor':'#193454','xtick.color':'#536781','ytick.color':'#536781'})
f,ax=plt.subplots(figsize=(11,6.5),dpi=220); f.subplots_adjust(left=.13,right=.94,bottom=.23,top=.74)
f.text(.06,.94,'Tỷ lệ khách–tháng có log theo nguồn',fontsize=24,fontweight='bold'); f.text(.06,.865,'Có ≥1 log trong tháng T−1 · Cùng mẫu số: 1.202.688 khách–tháng',fontsize=12,color='#62738a')
y=np.arange(3); ax.barh(y,100,color='#e9eef5',height=.48); ax.barh(y,r.with_log_pct,color=['#3677c5','#59aa96','#d99170'],height=.48)
for i,row in r.iterrows():
 ax.text(row.with_log_pct+1.4,i,f'{row.with_log_pct:.2f}%'.replace('.',','),va='center',fontsize=14,fontweight='bold')
 ax.text(0,i+.36,f'{row.with_log:,}'.replace(',','.')+' / 1.202.688 khách–tháng',fontsize=10.5,color='#62738a',va='center')
ax.set_yticks(y,r.source); ax.invert_yaxis(); ax.set_xlim(0,106); ax.set_xticks([0,20,40,60,80,100]); ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0)); ax.set_xlabel('Tỷ lệ khách–tháng có log',labelpad=14); ax.tick_params(axis='y',length=0,pad=14); ax.grid(axis='x',alpha=.14); ax.set_axisbelow(True)
for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
f.text(.06,.115,'Gộp 12 tháng nhãn 02/2022–01/2023; cửa sổ log tương ứng 01/2022–12/2022.',fontsize=11,color='#62738a'); f.text(.06,.06,'Một khách được tính lại ở mỗi tháng. Phần xám: không có log trong tháng T−1, không phải null trong bản ghi.',fontsize=10.5,color='#62738a'); f.savefig(O/'customer_month_log_coverage.png'); print(r.to_string(index=False))
