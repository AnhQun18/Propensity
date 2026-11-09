"""Continuous binned Gaussian KDE on untransformed currency=11 Amount."""
from pathlib import Path
import hashlib,json
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/amount_currency_11'
row=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
N=row['rows']; q99=row['p99']; q75=row['p75']
# Robust Silverman scale avoids allowing the extreme tail to erase the main mass.
bandwidth=.9*((row['p75']-row['p25'])/1.34)*N**(-.2)
high=q99+6*bandwidth
steps=65536; step=high/steps
files=sorted((ROOT/'data/MBD-mini/detail/trx').rglob('*.parquet'))
inventory=[(p.relative_to(ROOT).as_posix(),p.stat().st_size,p.stat().st_mtime_ns) for p in files]
fingerprint=hashlib.sha256(json.dumps({'files':inventory,'currency':11,'step':step,'high':high,'q99':q99},sort_keys=True).encode()).hexdigest()
cache=OUT/'continuous_histogram.npz'; meta=OUT/'continuous_cache.json'
old=json.loads(meta.read_text(encoding='utf-8')) if meta.exists() else {}
if cache.exists() and old.get('fingerprint')==fingerprint:
    with np.load(cache) as v:counts=v['counts']; above_q99=int(v['above_q99']); beyond_grid=int(v['beyond_grid'])
else:
    print('Counting fine linear Amount intervals for currency 11...',flush=True)
    db=duckdb.connect();db.execute("SET threads=2");db.execute("SET memory_limit='2GB'")
    db.read_parquet([p.as_posix() for p in files],hive_partitioning=False).create_view('trx')
    b=db.execute(f'''SELECT CASE WHEN amount>={high:.17g} THEN -1
                 ELSE floor(amount/{step:.17g})::INTEGER END bin,
                 count(*)::BIGINT n,count(*) FILTER(WHERE amount>{q99:.17g})::BIGINT above_q99
                 FROM trx WHERE currency=11 AND isfinite(amount) GROUP BY ALL''').df()
    db.close()
    assert int(b.n.sum())==N
    counts=np.zeros(steps,dtype=np.int64)
    finite=b[b.bin.ge(0)]
    counts[finite.bin.to_numpy(dtype=int)]=finite.n.to_numpy(dtype=np.int64)
    beyond_grid=int(b.loc[b.bin.lt(0),'n'].sum()); above_q99=int(b.above_q99.sum())
    np.savez_compressed(cache,counts=counts,above_q99=above_q99,beyond_grid=beyond_grid)
    meta.write_text(json.dumps({'fingerprint':fingerprint,'bandwidth':bandwidth,'step':step,'steps':steps,'n':N,'above_q99':above_q99,'beyond_grid':beyond_grid},indent=2),encoding='utf-8')
assert counts.sum()+beyond_grid==N
sigma=bandwidth/step; radius=int(np.ceil(6*sigma))
offset=np.arange(-radius,radius+1)
kernel=np.exp(-.5*(offset/sigma)**2);kernel/=kernel.sum()
# Reflection at zero corrects boundary leakage for positive Amount.
padded=np.concatenate([counts[:radius][::-1],counts,np.zeros(radius)])
density=np.convolve(padded,kernel,mode='valid')/(N*step)
x=(np.arange(steps)+.5)*step
assert len(density)==steps and np.isfinite(density).all() and (density>=0).all()
y=density*100*1000 # percentage points of probability per 1000 original units
INK='#173d38';GREEN='#267969';MUTED='#647c72';CORAL='#d38368'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'text.color':INK,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED})
def vi(v,dp=0):return f'{v:,.{dp}f}'.replace(',','~').replace('.',',').replace('~','.')
def clean(ax):
    ax.grid(axis='y',color='#e5ece7',lw=.8);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#d6e2da')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:vi(v,1)))

fig=plt.figure(figsize=(19.2,10.8),facecolor='white')
fig.text(.06,.947,'MBD / PROPENSITY LAB',fontsize=12,color=MUTED)
fig.text(.06,.886,'Đường mật độ Amount · currency = 11',fontsize=30,fontweight='bold')
fig.text(.06,.839,f'{vi(N)} giao dịch toàn lịch sử · Amount gốc, trục tuyến tính · không log',fontsize=16,color=MUTED)
ax=fig.add_axes([.085,.24,.865,.51])
sel=x<=q99
ax.plot(x[sel],y[sel],color=GREEN,lw=2.8)
ax.fill_between(x[sel],0,y[sel],color=GREEN,alpha=.10)
ax.set_xlim(0,q99);ax.set_ylim(0,max(y)*1.10)
ax.set_xlabel('Amount (triệu đơn vị gốc)',labelpad=15,fontsize=14)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:vi(v/1e6,1)))
ax.set_ylabel('Mật độ (% giao dịch / 1.000 đơn vị Amount)',labelpad=15,fontsize=13)
ax.yaxis.set_major_locator(MaxNLocator(5));clean(ax)
ax.text(.98,.04,f'Hiển thị 0–P99: 0–{vi(q99)}',transform=ax.transAxes,ha='right',fontsize=12,color=MUTED)

# One figure, with an explicitly labelled linear inset for the dense low-value mass.
detail=fig.add_axes([.40,.455,.50,.255],facecolor='white')
sel75=x<=q75
detail.plot(x[sel75],y[sel75],color=GREEN,lw=2.4)
detail.fill_between(x[sel75],0,y[sel75],color=GREEN,alpha=.10)
detail.set_xlim(0,q75);detail.set_ylim(0,max(y)*1.10)
detail.xaxis.set_major_formatter(FuncFormatter(lambda v,p:vi(v/1000)))
detail.set_xlabel('Amount (nghìn đơn vị gốc)',fontsize=11,labelpad=7)
detail.tick_params(labelsize=10);detail.yaxis.set_major_locator(MaxNLocator(3));clean(detail)
detail.set_title(f'Phóng vùng 0–P75 ({vi(q75)} đơn vị)',loc='left',fontsize=13,pad=12)
detail.axvline(row['median'],color=CORAL,lw=1.5,ls='--')
detail.text(row['median']+1800,max(y)*.72,f'Trung vị\n{vi(row["median"])}',fontsize=10,color=CORAL)
fig.text(.06,.138,'Đường mật độ là ước lượng làm mượt; trục Y không phải số giao dịch tại một giá trị Amount.',fontsize=14)
fig.text(.06,.093,f'Có {vi(above_q99)} giao dịch ({vi(above_q99/N*100,2)}%) vượt P99, nằm ngoài cửa sổ hiển thị; dữ liệu gốc vẫn được giữ.',fontsize=13,color=MUTED)
fig.text(.06,.048,'Chỉ dùng currency = 11; đơn vị tiền tệ chưa xác minh. Hình nhỏ dùng cùng mật độ, phóng riêng trục X.',fontsize=12,color=MUTED)
fig.savefig(OUT/'amount_currency_11_continuous.png',dpi=160,facecolor='white')
plt.close(fig)
pd.DataFrame({'amount':x,'density_per_unit':density}).to_csv(OUT/'continuous_density.csv',index=False)
summary={'currency':11,'rows':N,'bandwidth_original_units':bandwidth,'method':'Gaussian binned KDE, robust Silverman IQR bandwidth, reflection at zero','fine_bins':steps,'bin_width_original_units':step,'transform':'none','x_limits':[0,q99],'inset_x_limits':[0,q75],'above_view_n':above_q99,'above_view_percent':above_q99/N*100,'normalization_population':N,'grid_integral':float(np.sum(density)*step),'raw_data_modified':False}
(OUT/'continuous_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
