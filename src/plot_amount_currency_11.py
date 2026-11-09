"""Exact full-history Amount distribution for currency=11, on original values."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/amount_currency_11'
OUT.mkdir(parents=True,exist_ok=True)
EDGES=[0,1,10,50,100,250,500,1000,2000,4000,8000,15000,30000,60000,120000,
       250000,500000,1000000,5000000,10000000,100000000,1000000000]
files=sorted((ROOT/'data/MBD-mini/detail/trx').rglob('*.parquet'))
if not files:raise FileNotFoundError('detail/trx parquet files not found')
inventory=[(f.relative_to(ROOT).as_posix(),f.stat().st_size,f.stat().st_mtime_ns) for f in files]
fingerprint=hashlib.sha256(json.dumps({'files':inventory,'edges':EDGES,'currency':11},sort_keys=True).encode()).hexdigest()
meta_path=OUT/'cache.json';csv_path=OUT/'distribution.csv'
old=json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
if old.get('fingerprint')==fingerprint and csv_path.exists():
    bins=pd.read_csv(csv_path)
    print('Using verified cached currency=11 bins.',flush=True)
else:
    print(f'Scanning currency and amount from {len(files)} raw parquet files...',flush=True)
    db=duckdb.connect()
    db.execute("SET threads=2")
    db.execute("SET memory_limit='2GB'")
    db.read_parquet([f.as_posix() for f in files],hive_partitioning=False).create_view('trx')
    cases=' '.join(f'WHEN amount < {edge:.17g} THEN {i}' for i,edge in enumerate(EDGES[1:]))
    q=f'''SELECT CASE {cases} ELSE {len(EDGES)-1} END bin,
            count(*)::BIGINT n,min(amount) min_amount,max(amount) max_amount
          FROM trx WHERE currency=11 AND isfinite(amount)
          GROUP BY ALL ORDER BY bin'''
    bins=db.execute(q).df()
    db.close()
    bins=bins.set_index('bin').reindex(range(len(EDGES)),fill_value=0).reset_index()
    bins['left']=[EDGES[int(i)] for i in bins.bin]
    bins['right']=[EDGES[int(i)+1] if i<len(EDGES)-1 else np.inf for i in bins.bin]
    bins.to_csv(csv_path,index=False)
    meta_path.write_text(json.dumps({'fingerprint':fingerprint,'scope':'all raw detail/trx history, currency=11, finite amount',
                                    'currency':11,'bins':EDGES,'n':int(bins.n.sum()),'transform':'none'},indent=2),encoding='utf-8')
stats=pd.read_csv(ROOT/'docs/mbd_mini_eda/tables/review/amount_by_currency.csv')
row=stats[pd.to_numeric(stats.currency,errors='coerce').eq(11)].iloc[0]
assert int(bins.n.sum())==int(row['rows'])==int(row.finite_n)==38724240
assert bins.n.ge(0).all() and bins.min_amount.min()>=0
assert np.isclose(bins.max_amount.max(),row['max'])
bins['percent']=bins.n/bins.n.sum()*100
bins.to_csv(csv_path,index=False)

INK='#173d38';GREEN='#267969';MUTED='#647c72';CORAL='#d38368'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'text.color':INK,
                     'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED})
def vi(v,dp=0):return f'{v:,.{dp}f}'.replace(',','~').replace('.',',').replace('~','.')
def short(v):
    if v>=1e9:return vi(v/1e9)+'B'
    if v>=1e6:return vi(v/1e6)+'M'
    if v>=1e3:return vi(v/1e3)+'k'
    return str(int(v))
labels=[short(EDGES[i])+'–'+short(EDGES[i+1]) if i<len(EDGES)-1 else '≥1B' for i in range(len(EDGES))]
fig=plt.figure(figsize=(19.2,10.8),facecolor='white')
fig.text(.06,.947,'MBD / PROPENSITY LAB',fontsize=12,color=MUTED)
fig.text(.06,.887,'Phân phối Amount · currency = 11',fontsize=30,fontweight='bold')
fig.text(.06,.842,f'Toàn lịch sử · {vi(int(row["rows"]))} giao dịch · giá trị gốc, không log',fontsize=17,color=MUTED)
for x,label,val in [(.06,'Trung vị',row['median']),(.34,'P95',row.p95),(.62,'P99',row.p99)]:
    fig.text(x,.778,label,fontsize=13,color=MUTED)
    fig.text(x,.732,vi(val,2),fontsize=24,color=GREEN)
ax=fig.add_axes([.085,.25,.86,.40])
peak=int(np.argmax(bins.n))
colors=[GREEN]*len(bins);colors[peak]=CORAL
ax.bar(np.arange(len(bins)),bins.n.to_numpy()/1e6,width=.78,color=colors,zorder=3)
ax.set_xticks(np.arange(len(bins)),labels,rotation=48,ha='right',fontsize=10.5)
ax.tick_params(axis='x',length=0,pad=9)
ax.set_ylabel('Số giao dịch (triệu)',fontsize=14,labelpad=15)
ax.set_xlabel('Khoảng Amount (các khoảng có độ rộng không đều)',fontsize=13,labelpad=17)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:vi(x,1)))
ax.grid(axis='y',color='#e5ece7',lw=.9,zorder=0)
ax.set_ylim(0,bins.n.max()/1e6*1.24)
ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#d7e2db')
ax.tick_params(axis='y',length=0);ax.margins(x=.01)
ax.annotate(f'{vi(int(bins.iloc[peak].n))} giao dịch\nKhoảng {labels[peak]}',
            (peak,bins.iloc[peak].n/1e6),xytext=(0,13),textcoords='offset points',ha='center',fontsize=12,color=INK)
fig.text(.06,.119,'Cột = số giao dịch trong từng khoảng; không phải mật độ hoặc tổng Amount.',fontsize=15)
fig.text(.06,.079,'Giữ toàn bộ phần đuôi trong khoảng cuối. k = 1.000; M = 1 triệu; B = 1 tỷ đơn vị Amount gốc.',fontsize=13,color=MUTED)
fig.text(.06,.045,'Chưa xác minh mã 11 tương ứng đồng tiền nào; không quy đổi hoặc gắn đơn vị VND/USD.',fontsize=12,color=MUTED)
fig.savefig(OUT/'amount_currency_11.png',dpi=160,facecolor='white')
plt.close(fig)
summary={k:float(row[k]) for k in ['mean','min','max','p25','median','p75','p95','p99']}
summary.update({'rows':int(row['rows']),'currency':11,'finite_n':int(row.finite_n),'peak_interval':labels[peak],
                'peak_n':int(bins.iloc[peak].n),'bins':len(bins),'scope':'full history',
                'x_axis':'categorical unequal Amount intervals, equal display spacing',
                'y_axis':'transaction count','transform':'none','excluded_nonfinite':int(row['rows']-row.finite_n)})
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
