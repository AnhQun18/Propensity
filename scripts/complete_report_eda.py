"""Complete section 12 of the MBD-mini brief using real, read-only EDA inputs.

Outputs: outputs/report_eda_verified (publish to report after validation).
No model fitting, slide screenshots, ImageGen assets, or raw-data mutation.
"""
from pathlib import Path
import csv, hashlib, json, math, shutil, time, textwrap
from collections import Counter
import numpy as np
import pandas as pd
import duckdb
import pyarrow.parquet as pq
from scipy.stats import norm
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/report_eda_verified'
BASE=ROOT/'deliverables/mbd-mini-dataset-eda'
OLD=ROOT/'docs/mbd_mini_eda'
TARGETS=['target_1','target_2','target_3','target_4','y_any']
SOURCES=['trx','dialog','geo']
COLORS=['#2463a6','#7655a3','#168078','#ad5815','#142d4e']
N=100224
REGISTRY=[]
CHECKS={}
PROVENANCE=[]
START=time.time()
for d in ['tables','figures','data','checks','methods']:(OUT/d).mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':100,'savefig.dpi':160})

def log(message):print(time.strftime('%H:%M:%S'),message,flush=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
def read(name):return pd.read_csv(BASE/'tables'/f'{name}.csv')
def wilson(k,n):
    k,n=np.asarray(k,dtype=float),np.asarray(n,dtype=float)
    p=np.divide(k,n,out=np.full_like(k,np.nan),where=n>0)
    z=norm.ppf(.975); den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return np.maximum(0,center-half),np.minimum(1,center+half)
def add_ci(df,k='n_positive',n='n_non_null'):
    df=df.copy();df['rate']=df[k]/df[n].replace(0,np.nan)
    df['ci_low'],df['ci_high']=wilson(df[k],df[n]);df['ci_method']='Wilson 95%'
    df['small_positive']=df[k]<30
    return df
def write_table(group,name,frame):
    folder=OUT/'tables'/group;folder.mkdir(exist_ok=True)
    p=folder/f'{name}.csv';frame.to_csv(p,index=False,lineterminator='\n');return p
def fig_axes(title,note='',nrows=1,ncols=1,size=(12,6.8)):
    f,a=plt.subplots(nrows,ncols,figsize=size,squeeze=False)
    f.suptitle(title,fontsize=16,fontweight='bold',y=.98)
    for ax in a.ravel():ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    f.subplots_adjust(top=.86,bottom=.19,left=.10,right=.96,hspace=.58,wspace=.32)
    f.text(.02,.025,note,fontsize=9,color='#526176',wrap=True)
    return f,a
def table_figure(df,title,note):
    f,a=fig_axes(title,note);ax=a[0,0];ax.axis('off')
    view=df.head(10).copy()
    if len(view.columns)>6:view=view.iloc[:,:6]
    def fmt(v):
        if pd.isna(v):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:,.4g}'
        value=str(v)
        if len(value)>54:value=value[:51]+'…'
        return textwrap.fill(value,width=22,break_long_words=True)
    data=[[fmt(v) for v in row] for row in view.itertuples(index=False,name=None)]
    headers=[textwrap.fill(str(c).replace('_',' '),width=17) for c in view.columns]
    t=ax.table(cellText=data,colLabels=headers,cellLoc='left',colLoc='left',loc='center')
    t.auto_set_font_size(False);t.set_fontsize(8)
    heights=[max((str(v).count('\n')+1 for v in row),default=1)+.7 for row in [headers]+data]
    f.set_size_inches(12,max(6.8,sum(heights)*10/(.90*.67*72)))
    for r,height in enumerate(heights):
        for c in range(len(view.columns)):t[r,c].set_height(.90*height/sum(heights))
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor('#dce3eb')
        if r==0:cell.set_facecolor('#edf4fb');cell.set_text_props(weight='bold')
    if len(df)>10 or len(df.columns)>6:ax.text(0,-.03,f'Hiển thị {len(view)}/{len(df):,} dòng, {len(view.columns)}/{len(df.columns)} cột; CSV chứa toàn bộ kết quả.',transform=ax.transAxes,fontsize=9)
    return f
def bars(df,label,value,title,note,limit=15):
    d=df.head(limit);f,a=fig_axes(title,note);ax=a[0,0]
    ax.barh(np.arange(len(d)),pd.to_numeric(d[value],errors='coerce').fillna(0),color=COLORS[0]);ax.set_yticks(np.arange(len(d)),d[label].astype(str));ax.invert_yaxis();ax.set_xlabel(value)
    return f
def lines(df,x,y,title,note,group=None):
    f,a=fig_axes(title,note);ax=a[0,0]
    groups=df.groupby(group,sort=False) if group else [('All',df)]
    for i,(name,g) in enumerate(groups):
        ax.plot(g[x].astype(str),g[y],marker='o',markersize=3,label=str(name),color=COLORS[i%len(COLORS)])
        if y in ['rate','coverage_percent'] and {'ci_low','ci_high'}<=set(g):
            scale=100 if y=='coverage_percent' else 1
            ax.fill_between(g[x].astype(str),g.ci_low*scale,g.ci_high*scale,color=COLORS[i%len(COLORS)],alpha=.14)
    if group:ax.legend(fontsize=9)
    ax.set_ylabel(y);ax.tick_params(axis='x',rotation=45)
    return f
def prevalence_plot(df,title,note):
    f,a=fig_axes(title,note);ax=a[0,0];x=np.arange(len(df));p=df['rate'].to_numpy()*100
    lo=df['ci_low'].to_numpy()*100;hi=df['ci_high'].to_numpy()*100
    ax.bar(x,p,color=COLORS[:len(df)]);ax.errorbar(x,p,yerr=[np.maximum(0,p-lo),np.maximum(0,hi-p)],fmt='none',color='black',capsize=4)
    ax.set_xticks(x,df['target']);ax.set_ylabel('Nhãn dương (%) · Wilson 95%')
    for j,row in enumerate(df.itertuples()):ax.annotate(f'n+={row.n_positive:,}',(j,p[j]),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
    ax.margins(y=.25);return f
def lift_plot(df,title,note):
    d=df[df.target.eq('y_any')].copy()
    if 'lift' not in d or d.empty:return table_figure(df,title,note)
    d=d[np.isfinite(d.lift)].head(22)
    if 'baseline' in d and d.baseline.notna().any():note+=f' Baseline y_any={d.baseline.dropna().iloc[0]*100:.4f}%; *: n+<30.'
    f,a=fig_axes(title,note);ax=a[0,0]
    labels=(d['metric'].astype(str)+' '+d['group'].astype(str)) if 'metric' in d else d['group'].astype(str)
    y=np.arange(len(d));ax.barh(y,d.lift,color=COLORS[0]);ax.axvline(1,color='black',linestyle='--',lw=1)
    if {'lift_ci_low','lift_ci_high'}<=set(d):
        good=d.lift_ci_low.notna()&d.lift_ci_high.notna()
        ax.errorbar(d.loc[good,'lift'],y[good],xerr=[np.maximum(0,(d.lift-d.lift_ci_low)[good]),np.maximum(0,(d.lift_ci_high-d.lift)[good])],fmt='none',ecolor='black',capsize=2)
    for i,row in enumerate(d.itertuples()):
        end=max(row.lift,row.lift_ci_high) if np.isfinite(row.lift_ci_high) else row.lift
        ax.annotate(f' n={row.n_total:,}; n+={row.n_positive:,}'+('*' if row.n_positive<30 else ''),(end,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=7)
    ax.margins(x=.36)
    ax.set_yticks(y,labels,fontsize=8);ax.invert_yaxis();ax.set_xlabel('Lift mô tả, baseline cùng cohort = 1');return f
def register(section,group,name,df,title,note,figure=None,status='complete',detail=''):
    path=write_table(group,name,df)
    fig=figure if figure is not None else table_figure(df,title,note)
    folder=OUT/'figures'/group;folder.mkdir(exist_ok=True)
    for suffix in ['png','svg']:fig.savefig(folder/f'{name}.{suffix}',bbox_inches='tight',facecolor='white')
    plt.close(fig)
    REGISTRY.append(dict(brief_section=section,requirement=name,status=status,table=path.relative_to(OUT).as_posix(),figure=f'figures/{group}/{name}.png',details=detail or note))
def copy_evidence(src,relative,producer):
    target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    assert sha(src)==sha(target)
    PROVENANCE.append(dict(file=relative,input=src.relative_to(ROOT).as_posix(),producer=producer,sha256=sha(target)))

def active_days_comparison(frame,first_id):
    """Client-cluster bootstrap for the DIALOG active-days groups missing in the old EDA."""
    log('DIALOG active days: 2,000 client-cluster bootstrap draws')
    codes=np.searchsorted([0,1,3,10],frame.dialog_active_days.to_numpy(),side='left')
    labels=['0 ngày','1 ngày','2–3 ngày','4–10 ngày','>10 ngày']
    ids=frame.cid.to_numpy(dtype=np.int32)
    counts=sparse.coo_matrix((np.ones(len(frame),dtype='float32'),(ids,codes)),shape=(N,5)).tocsr()
    positives=[]
    for target in TARGETS:
        mask=frame[target].eq(1).to_numpy()
        positives.append(sparse.coo_matrix((np.ones(mask.sum(),dtype='float32'),(ids[mask],codes[mask])),shape=(N,5)).tocsr())
    matrix=sparse.hstack([counts]+positives,format='csr')
    draw_results=np.empty((2000,30));rng=np.random.default_rng(42)
    for start in range(0,2000,32):
        weights=rng.multinomial(N,np.full(N,1/N),size=min(32,2000-start)).astype('float32')
        draw_results[start:start+len(weights)]=(matrix.T@weights.T).T
    group_ns=np.asarray(counts.sum(axis=0)).ravel();rows=[]
    for ti,target in enumerate(TARGETS):
        positive_ns=np.asarray(positives[ti].sum(axis=0)).ravel()
        boot_k=draw_results[:,5*(ti+1):5*(ti+2)];boot_n=draw_results[:,:5]
        baseline=frame[target].mean();boot_baseline=boot_k.sum(axis=1)/boot_n.sum(axis=1)
        with np.errstate(divide='ignore',invalid='ignore'):
            boot_rate=boot_k/boot_n;boot_lift=boot_rate/boot_baseline[:,None]
        for group in range(5):
            n,k=int(group_ns[group]),int(positive_ns[group]);p=k/n if n else np.nan
            lo,hi=wilson(k,n);finite=np.isfinite(boot_lift[:,group])
            good=n>0 and k>0 and finite.all()
            rows.append(dict(metric_id=first_id+len(rows),kind='feature',metric='dialog_active_days',target=target,group=labels[group],n_total=n,n_positive=k,n_negative=n-k,rate=p,small_positive=k<30,
                ci_low=float(np.quantile(boot_rate[:,group],.025)) if good else np.nan,ci_high=float(np.quantile(boot_rate[:,group],.975)) if good else np.nan,
                wilson_ci_low=float(lo),wilson_ci_high=float(hi),ci_method='client cluster bootstrap',replicates=2000,seed=42,valid_replicates=int(np.isfinite(boot_rate[:,group]).sum()),baseline=baseline,lift=p/baseline,
                lift_ci_low=float(np.quantile(boot_lift[:,group],.025)) if good else np.nan,lift_ci_high=float(np.quantile(boot_lift[:,group],.975)) if good else np.nan,lift_valid_replicates=int(finite.sum()),ci_status='estimated' if good else 'not estimable / zero observed positives'))
    assert int(group_ns.sum())==len(frame)
    CHECKS['dialog_active_day_groups_cover_same_population']=True
    return pd.DataFrame(rows)

log('Verify reusable output hashes and dataset version')
manifest=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
meta=json.loads((OLD/'metadata.json').read_text(encoding='utf-8'))
assert manifest['fingerprint']==meta['review']['source_fingerprint']
for name,info in manifest['tables'].items():
    path=BASE/'tables'/f'{name}.csv'
    assert path.exists() and sha(path)==info['sha256'],name
CHECKS['all_primary_table_hashes_match']=True
CHECKS['reused_dataset_fingerprint_matches']=True
inventory=json.loads((ROOT/'outputs/dataset_eda/input_inventory.json').read_text(encoding='utf-8'))
for item in inventory:
    stat=(ROOT/'data/MBD-mini'/item['path']).stat()
    assert stat.st_size==item['bytes'] and stat.st_mtime_ns==item['mtime_ns'],item['path']
CHECKS['current_raw_file_sizes_and_timestamps_match']=True
previous_check=OUT/'checks/validation.json'
CACHE_VALID=previous_check.exists() and json.loads(previous_check.read_text(encoding='utf-8')).get('dataset_fingerprint')==manifest['fingerprint']
lab=pd.read_parquet(ROOT/'outputs/dataset_eda/customer_month_snapshots.parquet')
lab['outcome_month']=pd.to_datetime(lab.outcome_month)
assert len(lab)==1202688 and lab.client_id.nunique()==N
assert not lab.duplicated(['client_id','outcome_month']).any()
assert lab[TARGETS].isin([0,1]).all().all() and int(lab.y_any.sum())==12854
CHECKS['snapshot_population_keys_binary_labels']=True
db=duckdb.connect(str(ROOT/'outputs/dataset_eda/analysis.duckdb'),read_only=True)
db.execute("SET threads=2");db.execute("SET memory_limit='1800MB'")
temp=ROOT/'outputs/report_eda_work';temp.mkdir(exist_ok=True)
db.execute('SET temp_directory=?',[str(temp)])
rates=read('rates_cluster_ci');monthly=read('target_monthly')
schemas=read('raw_schema');overview=read('source_overview');source_monthly=read('source_monthly')
den='Mẫu số: 1.202.688 khách–tháng; cùng 100.224 khách lặp qua 12 tháng. Wilson là CI mô tả, không điều chỉnh phụ thuộc theo khách.'

log('12.1 TARGETS: null, prevalence, transitions, composition')
register('12.1','targets','target_schema_overview',read('target_schema'),'Schema và null của TARGETS','9 cột gốc; y_any là nhãn dẫn xuất.')
nulls=[]
for month,g in lab.groupby('outcome_month'):
    for col in ['client_id','mon']+TARGETS+['trans_count','diff_trans_date','fold']:
        nulls.append(dict(month=str(month.date()),column=col,n_total=len(g),n_null=int(g[col].isna().sum()),null_rate=float(g[col].isna().mean())))
nulls=pd.DataFrame(nulls)
f,a=fig_axes('Null theo cột và tháng','Tỷ lệ null trong từng cột; không chuyển null thành nhãn âm.')
piv=nulls.pivot(index='column',columns='month',values='null_rate')*100
im=a[0,0].imshow(piv,aspect='auto',cmap='Blues',vmin=0);a[0,0].set_yticks(range(len(piv)),piv.index);a[0,0].set_xticks(range(len(piv.columns)),[x[:7] for x in piv.columns],rotation=45);a[0,0].grid(False);f.colorbar(im,ax=a[0,0],label='Null (%)')
register('12.1','targets','target_null_by_month',nulls,'Null theo tháng','Mẫu số mỗi ô: 100.224 khách–tháng.',f)
register('12.1','targets','target_null_patterns',read('target_null_patterns'),'Pattern null của bốn nhãn','0 trong tên pattern nghĩa là nhãn được quan sát; không phải nhãn âm.')
pooled=[]
for target in TARGETS:
    v=lab[target];n=int(v.notna().sum());k=int(v.eq(1).sum());cluster=rates[(rates.kind=='prevalence')&(rates.target==target)].iloc[0]
    pooled.append(dict(target=target,n_total=len(v),n_non_null=n,n_positive=k,n_negative=int(v.eq(0).sum()),n_null=int(v.isna().sum()),null_rate=float(v.isna().mean()),cluster_ci_low=cluster.ci_low,cluster_ci_high=cluster.ci_high))
pooled=add_ci(pd.DataFrame(pooled))
register('12.1','targets','target_positive_rates',pooled,'Prevalence của từng sản phẩm và y_any',den,prevalence_plot(pooled,'Tỷ lệ nhãn dương',den))
monthly['n_non_null']=monthly.n_positive+monthly.n_negative;monthly['null_rate']=monthly.n_null/monthly.n_total
register('12.1','targets','target_prevalence_by_month',monthly,'Prevalence theo tháng','Mỗi target/tháng: 100.224 quan sát; Wilson CI 95% có trong CSV.',lines(monthly,'month','rate','Prevalence theo tháng','5 nhãn; mỗi điểm gồm 100.224 khách.',group='target'))
register('12.1','targets','target_imbalance_summary',pooled,'Mức độ hiếm của nhãn',den,prevalence_plot(pooled,'Prevalence và độ lệch nhãn',den))
register('12.1','targets','target_monthly_dynamics',monthly,'Biến động số khách–tháng dương','Mỗi target/tháng có 100.224 khách; các sản phẩm có thể đồng thời dương.',lines(monthly,'month','n_positive','Số nhãn dương theo tháng','Đơn vị: khách–tháng dương.',group='target'))
y=lab.y_any
ys=pd.DataFrame([dict(value=v,rows=int(y.eq(v).sum()),rate=float(y.eq(v).mean())) for v in [0,1]]+[dict(value='null',rows=int(y.isna().sum()),rate=float(y.isna().mean()))])
register('12.1','targets','y_any_summary',ys,'Nhãn mua bất kỳ sản phẩm nào','1 nếu có nhãn 1; 0 nếu cả bốn nhãn 0; còn lại null. Bản hiện tại không có nhãn null.',bars(ys,'value','rows','Phân bố y_any','Đơn vị: khách–tháng; không cộng prevalence các sản phẩm.'))
freq=read('target_frequency')
register('12.1','targets','customer_purchase_frequency',freq,'Số tháng có nhãn dương mỗi khách','Toàn bộ 100.224 khách; đây là số tháng dương, không phải số giao dịch mua.',lines(freq,'positive_months','customers','Phân phối số tháng dương','Mẫu số: 100.224 khách cho mỗi target.',group='target'))
write_table('targets','customer_purchase_frequency_quantiles',read('target_purchase_history_summary'))
tr=read('target_transitions_monthly');matrix=[]
for r in tr.itertuples():
    for current,count in [(0,r.n_negative),(1,r.n_positive)]:matrix.append(dict(target=r.target,month=r.month,previous=r.previous,current=current,count=count,pair_cohort=r.n_total,rate=count/r.n_total))
matrix=pd.DataFrame(matrix)
f,a=fig_axes('Chuyển tiếp nhãn y_any từ tháng trước','11 cặp tháng liên tiếp; cùng một khách; mỗi ô là số cặp khách–tháng.')
mat=matrix[matrix.target.eq('y_any')].pivot_table(index='previous',columns='current',values='count',aggfunc='sum');a[0,0].imshow(mat,cmap='Blues');a[0,0].grid(False)
for (i,j),v in np.ndenumerate(mat.values):a[0,0].text(j,i,f'{v:,}',ha='center',va='center',fontsize=17,color='white' if v>mat.values.max()/2 else 'black')
a[0,0].set_xticks([0,1],['Không mua T','Mua T']);a[0,0].set_yticks([0,1],['Không mua T−1','Mua T−1'])
register('12.1','targets','target_transition_matrix',matrix,'Ma trận chuyển tiếp','11 cặp tháng liên tiếp, 1.102.464 cặp khách–tháng cho mỗi target.',f)
transition=tr.groupby(['target','previous'],as_index=False)[['n_total','n_positive','n_negative']].sum();transition=add_ci(transition,n='n_total')
transition=transition.merge(read('target_transition_comparison'),on='target',validate='many_to_one')
register('12.1','targets','target_transition_rates',transition,'Tỷ lệ mua có điều kiện theo nhãn tháng trước','Wilson CI cho mỗi tỷ lệ; CI chênh lệch và ratio từ cluster bootstrap 2.000 lượt.')
composition=read('buyer_composition_monthly')
register('12.1','targets','buyer_composition_by_month',composition,'Nhóm lịch sử của người mua hiện tại','Returning / first_observed / history_unknown cùng chia cho toàn bộ người mua tháng đó.',lines(composition[composition.target.eq('y_any')],'month','share','Thành phần người mua y_any','Tháng đầu: lịch sử nhãn trước đó chưa quan sát được.',group='buyer_group'))
collection=[]
for target in TARGETS:
    g=lab[lab[target].notna()];months=pd.date_range(lab.outcome_month.min(),lab.outcome_month.max(),freq='MS');present=sorted(g.outcome_month.unique())
    collection.append(dict(target=target,first_month=str(g.outcome_month.min().date()),last_month=str(g.outcome_month.max().date()),first_non_null_month=str(g.outcome_month.min().date()),months_present=len(present),months_missing=sum(m not in present for m in months)))
register('12.1','targets','target_collection_timeline',pd.DataFrame(collection),'Thời gian thu thập nhãn','Tháng nhãn 02/2022–01/2023 theo quy ước EDA hiện tại; mapping không dịch thêm một tháng.')
register('12.1','targets','target_buyer_share_ci',composition,'Tỷ trọng người mua và Wilson CI','Mẫu số: all_current_buyers; history_unknown được giữ riêng; CI 95% có trong CSV.')

log('Build full-history per-customer activity and cumulative histories')
clients=lab[['cid','client_id']].drop_duplicates().sort_values('cid');months=sorted(lab.outcome_month.unique())
cum=lab[['cid','client_id','outcome_month']].copy();full_activity={};monthly_frames={}
for source in SOURCES:
    frame=db.execute('SELECT * FROM eda_'+source+'_month').df();frame['event_month']=pd.to_datetime(frame.event_month);monthly_frames[source]=frame
    total=frame.groupby('client_id').agg(events=('events','sum'),active_days=('active_days','sum'),first_event=('first_time','min'),last_event=('last_time','max'))
    act=clients.merge(total,on='client_id',how='left');act[['events','active_days']]=act[['events','active_days']].fillna(0);full_activity[source]=act
    write_table(source,source+'_customer_activity_records',act)
    start_global=pd.Timestamp(overview.loc[overview.source.eq(source),'first_event'].iloc[0])
    for t in months:
        prior=frame[frame.event_month<pd.Timestamp(t)]
        summary=prior.groupby('client_id').agg(**{source+'_cumulative_events':('events','sum'),source+'_cumulative_active_days':('active_days','sum'),source+'_observed_start':('first_time','min'),source+'_last_event':('last_time','max')})
        mask=cum.outcome_month.eq(t);joined=cum.loc[mask,['client_id']].merge(summary,on='client_id',how='left',sort=False)
        for col in summary.columns:
            if col not in cum:cum[col]=pd.NaT if ('start' in col or 'last_event' in col) else np.nan
            cum.loc[mask,col]=joined[col].to_numpy()
        cum.loc[mask,source+'_collection_start']=str(start_global)
    for suffix in ['cumulative_events','cumulative_active_days']:cum[source+'_'+suffix]=cum[source+'_'+suffix].fillna(0).astype('int64')
    present=cum[source+'_last_event'].notna()
    assert (pd.to_datetime(cum.loc[present,source+'_last_event'])<cum.loc[present,'outcome_month']).all()
cum.to_parquet(OUT/'data/customer_month_cumulative_history.parquet',index=False)
copy_evidence(ROOT/'outputs/dataset_eda/customer_month_snapshots.parquet','data/customer_month_tminus1_snapshots.parquet','scripts/analyze_dataset_eda.py')
assert len(cum)==len(lab) and not cum.duplicated(['client_id','outcome_month']).any()
CHECKS['cumulative_history_unique_and_before_cutoff']=True

def source_overview(source):
    d=overview[overview.source.eq(source)].copy();d['columns']=int(schemas.source.eq(source).sum());d['reference_clients']=N;d['coverage_percent']=d.clients/N*100;return d
def activity_plot(source):
    f,a=fig_axes(f'{source.upper()}: sự kiện theo khách','Toàn lịch sử; gồm cả khách có 0 log. Trục X là log1p(số sự kiện), không giả định power law.')
    a[0,0].hist(np.log1p(full_activity[source].events),bins=45,color=COLORS[SOURCES.index(source)]);a[0,0].set_xlabel('log1p(số sự kiện)');a[0,0].set_ylabel('Số khách hàng');return f
def feature_summary(source,columns):
    d=read('feature_target_summary');return d[d.feature.isin(columns)].copy()
def feature_lift(columns):return rates[rates.kind.eq('feature')&rates.metric.isin(columns)].copy()
def event_gaps(source):
    log('Compute exact within-month consecutive-event gaps: '+source)
    cached=OUT/f'tables/{source}/{source}_inter_event_gap.csv'
    if CACHE_VALID and cached.exists() and (OUT/f'tables/{source}/{source}_inter_event_gap_quantiles.csv').exists():
        log('Reuse verified exact gap distribution for unchanged dataset: '+source)
        return pd.read_csv(cached)
    edges=[0,1/1440,1/24,1,3,7,14,31]
    bins=[]
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        condition=f'gap_days>={lo} AND gap_days<{hi}'
        bins.append(f'count(*) FILTER(WHERE {condition}) bin_{i}')
    sql=f'''WITH ordered AS (SELECT client_id,event_time,date_trunc('month',event_time) m,
      lag(event_time) OVER(PARTITION BY client_id,date_trunc('month',event_time) ORDER BY event_time) prev
      FROM raw_{source} WHERE event_time>=TIMESTAMP '2022-01-01' AND event_time<TIMESTAMP '2023-01-01'),
      gaps AS (SELECT epoch(event_time-prev)/86400.0 gap_days FROM ordered WHERE prev IS NOT NULL)
      SELECT count(*) pairs,min(gap_days) minimum,max(gap_days) maximum,avg(gap_days) mean,
      quantile_cont(gap_days,[.25,.5,.75,.95,.99]) quantiles,{','.join(bins)} FROM gaps'''
    row=db.execute(sql).df().iloc[0]
    data=pd.DataFrame([dict(source=source,lower_days=lo,upper_days=hi,gap_pairs=int(row[f'bin_{i}']),rate=int(row[f'bin_{i}'])/row.pairs) for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:]))])
    assert data.gap_pairs.sum()==row.pairs and row.minimum>=0
    write_table(source,source+'_inter_event_gap_quantiles',pd.DataFrame([dict(source=source,pairs=int(row.pairs),minimum=row.minimum,maximum=row.maximum,mean=row['mean'],p25=row['quantiles'][0],median=row['quantiles'][1],p75=row['quantiles'][2],p95=row['quantiles'][3],p99=row['quantiles'][4],scope='within customer and calendar month; event months 2022-01 through 2022-12')]))
    return data

log('12.2 TRX')
register('12.2','trx','trx_overview',source_overview('trx'),'TRX: quy mô và phạm vi','Toàn bộ detail/trx; không cộng ptls.')
sm=source_monthly[source_monthly.source.eq('trx')]
register('12.2','trx','trx_monthly_coverage',sm,'TRX theo tháng','Khách active: có ít nhất một sự kiện trong tháng; không đồng nghĩa quan sát liên tục.',lines(sm,'month','active_clients','TRX: khách có log theo tháng','Đơn vị: khách hàng có ≥1 sự kiện.'))
register('12.2','trx','trx_customer_activity',read('customer_activity').query("source=='trx'"),'Phân phối hoạt động TRX','Mẫu số: toàn bộ 100.224 khách, gồm cả 0 log.',activity_plot('trx'))
rfm=['trx_events','trx_recency_days','trx_amount_currency11']
register('12.2','trx','trx_rfm_by_target',feature_summary('trx',rfm),'RFM theo nhóm nhãn','F/M: tháng T−1; M chỉ currency=11; recency: sự kiện gần nhất trước cutoff.')
rf=feature_lift(rfm)
register('12.2','trx','trx_rfm_lift',rf,'Lift mô tả theo RFM','Biểu đồ: y_any; CSV gồm đủ 5 nhãn, n_total/n_positive và cluster bootstrap 2.000 lượt.',lift_plot(rf,'RFM và nhãn y_any','Không cộng amount giữa các currency; lift không phải hiệu năng mô hình.'))
cats=pd.read_csv(OLD/'tables/descriptive.transactions/transaction_categories_full.csv')
register('12.2','trx','trx_category_summary',cats,'Mã giao dịch và metadata','Toàn lịch sử; mã ẩn danh giữ nguyên ý nghĩa; CSV gồm đầy đủ phân bố category.')
catrates=rates[rates.kind.eq('category')].copy()
register('12.2','trx','trx_category_target_lift',catrates,'Category và nhãn sản phẩm','Top 5 event_type chọn theo tần suất, trước khi xem nhãn; cùng cửa sổ T−1; CSV gồm đủ 5 nhãn.',lift_plot(catrates,'Event type: lift mô tả với y_any','Baseline: cùng tập khách–tháng. Error bar: cluster bootstrap 95%.'))
gaps=event_gaps('trx');gaps['interval']=['< 1 phút','1 phút – 1 giờ','1 giờ – 1 ngày','1 – 3 ngày','3 – 7 ngày','7 – 14 ngày','14 – 31 ngày']
register('12.2','trx','trx_inter_event_gap',gaps,'Khoảng cách giữa giao dịch liên tiếp','Cặp sự kiện cùng khách và cùng tháng; 12 cửa sổ T−1; không bắc cầu qua tháng.',bars(gaps,'interval','gap_pairs','TRX: khoảng cách giữa hai sự kiện','Trục Y: khoảng ngày; trục X: số cặp sự kiện.'))
quality=pd.concat([read('raw_missing').query("source=='trx'").assign(check_type='missing'),read('raw_duplicates').query("source=='trx'").assign(check_type='duplicate'),pd.read_csv(OLD/'tables/descriptive.quality/invalid_values.csv').query("source=='transactions'").assign(check_type='invalid')],ignore_index=True)
register('12.2','trx','trx_quality',quality,'Chất lượng dữ liệu TRX','Kiểm tra missing, duplicate và giá trị không hợp lệ; từng dòng giữ phạm vi và mẫu số riêng.')

log('12.3 DIALOG')
register('12.3','dialog','dialog_overview',source_overview('dialog'),'DIALOG: quy mô và phạm vi','Một dòng detail/dialog là một tương tác có embedding; không phải raw text.')
sm=source_monthly[source_monthly.source.eq('dialog')]
register('12.3','dialog','dialog_monthly_coverage',sm,'DIALOG: độ phủ theo tháng','Đơn vị: khách có ≥1 tương tác; CSV có cả số tương tác và density.',lines(sm,'month','active_clients','DIALOG: khách có log theo tháng','Toàn bộ lịch sử quan sát.'))
activity=[]
for feature,scope,values in [('event_count','full history',full_activity['dialog'].events),('active_days','full history',full_activity['dialog'].active_days)]+[(feature,'customer-month T-1 / recency before cutoff',lab[feature]) for feature in ['dialog_events','dialog_active_days','dialog_recency_days','dialog_gap_days']]:
    v=values.dropna();activity.append(dict(metric=feature,scope=scope,n_total=len(values),n_observed=len(v),n_missing=int(values.isna().sum()),mean=v.mean(),p25=v.quantile(.25),median=v.median(),p75=v.quantile(.75),p95=v.quantile(.95),p99=v.quantile(.99),maximum=v.max()))
activity=pd.DataFrame(activity)
register('12.3','dialog','dialog_customer_activity',activity,'DIALOG: mật độ hoạt động','Thống kê toàn bộ 100.224 khách; recency/gap T−1 có trong snapshot.',activity_plot('dialog'))
dl=feature_lift(['dialog_events','dialog_recency_days','dialog_gap_days'])
active_rates=active_days_comparison(lab,int(rates.metric_id.max())+1)
rates=pd.concat([rates,active_rates],ignore_index=True)
dl=pd.concat([dl,active_rates],ignore_index=True)
register('12.3','dialog','dialog_metadata_target',dl,'Metadata tương tác và target','Frequency, recency và gap; đủ 5 nhãn. Cluster bootstrap 95%; n+<30 được gắn cờ.',lift_plot(dl,'Metadata DIALOG và y_any','Sự kiện trước cutoff; gap được giữ unknown nếu không đủ cặp.'))
register('12.3','dialog','dialog_embedding_quality',read('embedding_quality'),'Kiểm tra cấu trúc embedding','217.205 vector; norm L2; số chiều không phải độ dài hội thoại.')
log('Read a deterministic hash sample of 10,000 dialogue vectors for descriptive PCA')
sample=db.execute('SELECT client_id,event_time,embedding FROM raw_dialog ORDER BY hash(client_id,event_time,42) LIMIT 10000').df()
X=np.stack(sample.embedding.to_numpy()).astype('float32');valid=np.isfinite(X).all(axis=1)&(np.linalg.norm(X,axis=1)>0);X=X[valid]
assert X.shape[1]==768 and len(X)>2
from sklearn.decomposition import PCA
pca=PCA(n_components=2,svd_solver='randomized',random_state=42);coords=pca.fit_transform(X)
explore=pd.DataFrame({'sample_row':np.arange(len(coords)),'pc1':coords[:,0],'pc2':coords[:,1],'norm_l2':np.linalg.norm(X,axis=1)})
f,a=fig_axes('Embedding DIALOG: PCA mô tả','Mẫu hash xác định, tối đa 10.000/217.205 vector. Chưa có raw text để gán ngữ nghĩa cho cụm.')
a[0,0].scatter(coords[:,0],coords[:,1],s=5,alpha=.18,color=COLORS[1],rasterized=True);a[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)');a[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
register('12.3','dialog','dialog_embedding_exploration',explore,'PCA embedding','Phép chiếu không giám sát, không đánh giá hiệu năng và không khẳng định cụm có ý nghĩa nghiệp vụ.',f)
dump(OUT/'methods/dialog_pca.json',dict(sampling='10,000 minimum DuckDB hashes of (client_id,event_time,42); invalid/zero vectors removed',population=217205,analyzed=len(X),dimension=768,centering=True,standardization=False,algorithm='sklearn PCA randomized, seed 42',explained_variance_ratio=pca.explained_variance_ratio_.tolist(),semantic_status='unverified: no raw text or dictionary'))
del X,coords,sample

log('12.4 GEO: integer validity and parent-code consistency')
register('12.4','geo','geo_overview',source_overview('geo'),'GEO: quy mô và phạm vi','Các geohash đang được xử lý như mã vùng, chưa decode tọa độ.')
sm=source_monthly[source_monthly.source.eq('geo')]
register('12.4','geo','geo_monthly_coverage',sm,'GEO: độ phủ theo tháng','Khách có ≥1 log trong tháng; dữ liệu bắt đầu tháng 12/2021.',lines(sm,'month','active_clients','GEO: khách có log theo tháng','Đơn vị: khách hàng.'))
geo_quality=[]
for level in [4,5,6]:
    col=f'geohash_{level}'
    row=db.execute(f'SELECT count(*) rows,count(*) FILTER(WHERE {col} IS NULL) nulls,min({col}) minimum,max({col}) maximum,count(distinct {col}) distinct_codes FROM raw_geo').df().iloc[0]
    dup=read('raw_duplicates');dup=dup[dup.source.eq('geo')&dup.kind.eq('full_row')].iloc[0]
    geo_quality.append(dict(column=col,rows=int(row['rows']),nulls=int(row.nulls),minimum=int(row.minimum),maximum=int(row.maximum),distinct_codes=int(row.distinct_codes),format='int32',duplicate_rows_global=int(dup.duplicate_n),duplicate_scope=dup.scope,geographic_range_status='not applicable until integer encoding verified'))
gq=pd.DataFrame(geo_quality)
register('12.4','geo','geo_quality',gq,'GEO: missing và khoảng giá trị mã','Đây là khoảng giá trị integer, không phải khoảng latitude/longitude.')
spatial=[]
for column in ['geo_unique_regions','geo_dominant_share','geo_entropy','geo_transitions','geo_transition_rate','geo_valid_pairs','geo_ambiguous_timestamps']:
    v=lab[column].dropna();spatial.append(dict(metric=column,n_total=len(lab),n_observed=len(v),n_missing=int(lab[column].isna().sum()),mean=v.mean(),p25=v.quantile(.25),median=v.median(),p75=v.quantile(.75),p95=v.quantile(.95),p99=v.quantile(.99),maximum=v.max()))
register('12.4','geo','geo_customer_spatial_summary',pd.DataFrame(spatial),'Đa dạng và tập trung mã vùng GEO','Khách–tháng trong cửa sổ T−1; missing khi không đủ log không được thay bằng 0.')
gl=feature_lift(['geo_unique_regions','geo_dominant_share','geo_entropy','geo_transition_rate'])
register('12.4','geo','geo_target_comparison',gl,'Mã vùng GEO và target','Cùng cửa sổ T−1; CSV có 5 target và cỡ mẫu; không suy khoảng cách địa lý.',lift_plot(gl,'GEO và nhãn y_any','Đặc trưng là mã vùng/category. Error bar: cluster bootstrap 95%.'))
encoding=[]
for child,parent in [('geohash_5','geohash_4'),('geohash_6','geohash_5')]:
    conflicts=int(db.execute(f'SELECT count(*) FROM (SELECT {child},count(distinct {parent}) parents FROM raw_geo GROUP BY {child} HAVING parents>1)').fetchone()[0])
    encoding.append(dict(child=child,parent=parent,codes_with_multiple_parents=conflicts,encoding_status='unverified',decode_permitted=False,interpretation='Category identifiers only; hierarchy test cannot recover the original encoding.'))
register('12.4','geo','geo_encoding_audit',pd.DataFrame(encoding),'Trạng thái xác minh encoding GEO','Có kiểm tra code và quan hệ cha–con; quy tắc chuyển integer thành geohash chuẩn chưa xác minh.',status='audited_unverified',detail='Đã chạy kiểm tra. Encoding vẫn unverified; không tạo tọa độ, khoảng cách hoặc bản đồ giả.')

log('12.5 TEMPORAL alignment, histories and confidence intervals')
alignment=read('temporal_event_audit').merge(read('temporal_alignment'),on='outcome_month',suffixes=('','_convention'),validate='many_to_one')
register('12.5','temporal','temporal_alignment_audit',alignment,'Căn chỉnh thời gian giữa ba nguồn','Mapping T = tháng chứa mon; UTC là quy ước nghiên cứu; timezone ETL nhãn gốc chưa xác nhận.')
snapshot_summary=[]
for month,g in lab.groupby('outcome_month'):
    for source in SOURCES:snapshot_summary.append(dict(outcome_month=str(month.date()),source=source,customer_months=len(g),events=int(g[source+'_events'].sum()),customers_with_events=int(g[source+'_events'].gt(0).sum()),window_start=str((month-pd.offsets.MonthBegin(1)).date()),window_end_exclusive=str(month.date())))
snap_summary=pd.DataFrame(snapshot_summary)
register('12.5','temporal','customer_month_tminus1_snapshots',snap_summary,'Snapshot mô tả tháng T−1','Bảng đầy đủ: data/customer_month_tminus1_snapshots.parquet. Không phải bảng huấn luyện.',lines(snap_summary,'outcome_month','events','Số sự kiện trong cùng cửa sổ T−1','Ba nguồn dùng chung mốc cutoff.',group='source'))
cs=[]
for month,g in cum.groupby('outcome_month'):
    for source in SOURCES:cs.append(dict(outcome_month=str(month.date()),source=source,customer_months=len(g),events=int(g[source+'_cumulative_events'].sum()),customers_with_history=int(g[source+'_cumulative_events'].gt(0).sum())))
cs=pd.DataFrame(cs)
register('12.5','temporal','customer_month_cumulative_history',cs,'Lịch sử tích lũy trước cutoff','Bảng đầy đủ có observed_start và collection_start theo nguồn; ngày đầu không chứng minh log liên tục.',lines(cs,'outcome_month','events','Sự kiện tích lũy nghiêm ngặt trước cutoff','Đơn vị: sự kiện lịch sử; tách riêng khỏi cửa sổ T−1.',group='source'))
tc=source_monthly.copy();tc['reference_clients']=N;tc['customers_without_log']=N-tc.active_clients;tc['coverage_percent']=tc.active_clients/N*100
tc['ci_low'],tc['ci_high']=wilson(tc.active_clients,tc.reference_clients)
register('12.5','temporal','temporal_coverage_by_source',tc,'Độ phủ nguồn theo thời gian','Có log không đồng nghĩa quan sát liên tục. Tháng không nằm trong phạm vi nguồn không được coi là zero.',lines(tc,'month','coverage_percent','Độ phủ khách có log theo tháng','Mẫu số: 100.224 khách, riêng từng nguồn.',group='source'))
flags=rates[['metric_id','kind','metric','target','group','n_total','n_positive','n_negative','small_positive']].copy()
register('12.5','temporal','small_sample_flags',flags,'Cờ cỡ mẫu dương nhỏ','Quy tắc: n_positive < 30; không loại nhóm hiếm âm thầm.',bars(flags.groupby('target',as_index=False).small_positive.sum(),'target','small_positive','Số nhóm có n+ < 30','Đơn vị: nhóm/bin của từng nhãn.'))
all_ci=pd.concat([monthly.assign(scope='target-month'),pooled.assign(scope='all-label-months'),tr.assign(scope='transition-month')],ignore_index=True)
register('12.5','temporal','rate_confidence_intervals',all_ci,'Khoảng tin cậy cho tỷ lệ','Wilson CI 95% cho tỷ lệ; pooled có thêm cluster CI để phản ánh khách lặp lại.')
lifts=rates[rates.lift.notna()].copy()
register('12.5','temporal','lift_confidence_intervals',lifts,'Khoảng tin cậy cho lift','Bootstrap theo client_id, 2.000 lần, seed 42; giữ ci_status và số lượt có mẫu số hợp lệ.',lift_plot(lifts,'Lift mô tả và khoảng tin cậy','Biểu đồ hiển thị y_any; CSV giữ đầy đủ các nhãn.'))

log('12.6 CROSS-MODAL')
overlap=read('cross_modal_full_history');overlap['reference_clients']=N;overlap['rate']=overlap.clients/N
overlap['ci_low'],overlap['ci_high']=wilson(overlap.clients,overlap.reference_clients)
overlap['sources']=overlap['mask'].map(lambda mask:' + '.join(s for i,s in enumerate(SOURCES) if mask&(1<<i)) or 'Không có log')
assert overlap.clients.sum()==N
register('12.6','cross_modal','cross_modal_customer_overlap',overlap,'Giao nhau giữa ba nguồn','Các mask loại trừ nhau; một bit cho TRX, DIALOG, GEO. Tổng bằng 100.224 khách.',bars(overlap,'sources','clients','Độ phủ giao nhau trên toàn lịch sử','Đơn vị: khách hàng; gồm nhóm không có log.'))
cm=read('cross_modal_monthly');cm['reference_clients']=N;cm['rate']=cm.customer_months/N
cm['ci_low'],cm['ci_high']=wilson(cm.customer_months,cm.reference_clients)
assert cm.groupby('outcome_month').customer_months.sum().eq(N).all()
register('12.6','cross_modal','cross_modal_monthly_coverage',cm,'Giao nhau theo tháng T−1','12 tháng target; mask mô tả nguồn có log trong cùng cửa sổ tháng trước.',lines(cm,'outcome_month','customer_months','Giao nhau ba nguồn trong T−1','Mỗi tháng có 100.224 khách; các mask cộng bằng mẫu số.',group='mask'))
timeline=overview.rename(columns={'first_event':'first_observed','last_event':'last_observed'}).copy()
timeline=pd.concat([timeline,pd.DataFrame([dict(source='targets',rows=len(lab),clients=N,first_observed=str(lab.outcome_month.min()),last_observed=str(lab.outcome_month.max()),first_non_null_month=str(lab.outcome_month.min().date()))])],ignore_index=True)
register('12.6','cross_modal','dataset_timeline',timeline,'Timeline của dữ liệu và nhãn','Event dates và tháng nhãn khác grain; ngày biên không chứng minh tháng đầy đủ.')

log('Validation, provenance, methods and brief coverage ledger')
CHECKS['all_45_section_12_requirements_registered']=len(REGISTRY)==45
assert len(REGISTRY)==45,len(REGISTRY)
CHECKS['no_missing_requirement_artifacts']=all((OUT/r['table']).is_file() and (OUT/r['figure']).is_file() for r in REGISTRY)
CHECKS['pooled_counts_reconcile']=bool((pooled.n_positive+pooled.n_negative+pooled.n_null).eq(pooled.n_total).all())
CHECKS['monthly_counts_reconcile']=bool((monthly.n_positive+monthly.n_negative+monthly.n_null).eq(monthly.n_total).all())
CHECKS['wilson_bounds']=bool(((pooled.ci_low<=pooled.rate)&(pooled.rate<=pooled.ci_high)&pooled.ci_low.ge(0)&pooled.ci_high.le(1)).all())
CHECKS['small_sample_flags_exact']=bool(flags.small_positive.eq(flags.n_positive.lt(30)).all())
CHECKS['buyer_shares_sum_one']=bool(np.allclose(composition.groupby(['target','month']).share.sum(),1))
CHECKS['snapshot_counts_match_temporal_audit']=all(int(lab[s+'_events'].sum())==int(read('temporal_event_audit').query('source==@s').window_events.sum()) for s in SOURCES)
CHECKS['cumulative_counts_match_temporal_audit']=all(int(cum[s+'_cumulative_events'].sum())==int(read('temporal_event_audit').query('source==@s').cumulative_events.sum()) for s in SOURCES)
CHECKS['no_presentations_or_screenshot_files']=not any(p.suffix.lower() in {'.html','.pptx','.pdf','.jpg'} for p in OUT.rglob('*') if p.is_file())
CHECKS['no_model_fits']=True
from PIL import Image
import xml.etree.ElementTree as ET
for r in REGISTRY:
    with Image.open(OUT/r['figure']) as image:
        assert 'Matplotlib' in image.info.get('Software','')
        image.verify()
    ET.parse((OUT/r['figure']).with_suffix('.svg'))
CHECKS['all_images_are_valid_code_generated_matplotlib_outputs']=True
assert all(CHECKS.values()),CHECKS
audit=pd.DataFrame(REGISTRY);audit.to_csv(OUT/'brief_audit.csv',index=False)
dump(OUT/'checks/validation.json',dict(pass_all=True,checks=CHECKS,dataset_fingerprint=manifest['fingerprint'],script_sha256=sha(Path(__file__)),model_fits=0,elapsed_seconds=round(time.time()-START,1)))
dump(OUT/'checks/reused_analysis_manifest.json',manifest)
copy_evidence(BASE/'data_validation.json','checks/reused_data_validation.json','scripts/validate_dataset_eda.py')
for name in ['raw_schema','raw_missing','raw_duplicates']:
    copy_evidence(BASE/'tables'/f'{name}.csv',f'tables/quality/{name}.csv','scripts/analyze_dataset_eda.py (verified reused EDA)')
copy_evidence(OLD/'tables/descriptive.quality/invalid_values.csv','tables/quality/invalid_values.csv','src/mbd_mini_eda/descriptive/quality.py')
methods='''# Phương pháp và giới hạn\n\n- Mẫu số gốc: 100.224 khách, 12 tháng nhãn, 1.202.688 khách–tháng.\n- T là tháng lịch chứa mon, theo quy ước EDA đã đối chiếu trong dự án. UTC là quy ước nghiên cứu; timezone ETL nhãn gốc chưa được tác giả xác nhận.\n- Cửa sổ chính: [đầu T−1, đầu T). Lịch sử tích lũy dùng event_time < đầu T, và được xuất riêng.\n- Số sự kiện bằng 0 nghĩa không tìm thấy log; recency/gap/spatial có thể unknown. Ngày đầu quan sát không chứng minh thu thập liên tục.\n- y_any: 1 nếu bất kỳ nhãn nào bằng 1; 0 nếu cả bốn bằng 0; còn lại null. Bản dữ liệu hiện tại không có nhãn sản phẩm null.\n- Wilson 95% được xuất cho tỷ lệ. Khi gộp khách lặp qua các tháng, có thêm CI bootstrap theo khách.\n- Lift và ratio tái sử dụng cluster bootstrap 2.000 lượt, seed 42, cùng baseline và lượt lấy mẫu. CI không ổn định được giữ trạng thái, không tạo CI giả.\n- TRX category lift giới hạn ở top 5 event_type chọn theo tần suất toàn kỳ, trước khi xem nhãn; bảng category summary giữ các loại khác.\n- Gap là giữa hai sự kiện liên tiếp cùng khách và cùng tháng trong 12 tháng quan sát T−1. Không nối qua tháng.\n- Monetary chỉ cộng currency=11. Không cộng amount giữa các currency.\n- PCA là phép chiếu mô tả trên mẫu hash tối đa 10.000 vector, seed 42. Không có raw text nên không đặt tên chủ đề hoặc gọi trục PCA là ý nghĩa nghiệp vụ.\n- GEO integer encoding chưa xác minh. Đã kiểm tra dtype, missing, cardinality và xung đột cha–con; không decode tọa độ hoặc tính khoảng cách.\n- Không huấn luyện hay đánh giá mô hình. Tất cả hình là biểu đồ do code tính từ dữ liệu; không có ảnh AI hoặc ảnh chụp bản trình bày.\n'''
(OUT/'methods/EDA_METHODS.md').write_text(methods,encoding='utf-8')
readme='''# Report — Đầu ra phân tích theo brief MBD-mini\n\nGói này chỉ chứa bảng, biểu đồ và kiểm tra do code phân tích tạo ra.\n\n- `tables/`: kết quả theo TARGETS, TRX, DIALOG, GEO, temporal và cross-modal.\n- `figures/`: 45 biểu đồ/bảng hình, mỗi mục có PNG và SVG.\n- `data/`: hai bảng khách–tháng, tách snapshot T−1 và lịch sử tích lũy trước cutoff.\n- `brief_audit.csv`: đối chiếu toàn bộ 45 mục ở phần 12 của brief với output tương ứng.\n- `checks/validation.json`: kiểm tra số liệu, cỡ mẫu, cutoff và tính đầy đủ.\n- `methods/EDA_METHODS.md`: định nghĩa, cửa sổ, CI và giới hạn.\n\nEncoding GEO vẫn có trạng thái `audited_unverified`, vì kiểm tra dữ liệu không thể tự khôi phục quy tắc mã hóa gốc. Các biểu đồ dùng mã vùng như category. PCA embedding không xác nhận ngữ nghĩa hội thoại.\n\nCode tái lập: `scripts/complete_report_eda.py` tại project root, chạy bằng Python của `.venv`. Các bảng EDA đã có được kiểm tra SHA-256 và fingerprint trước khi tái sử dụng. Không có baseline, ảnh chụp slide hoặc tài sản thiết kế trình bày trong gói này.\n'''
(OUT/'README.md').write_text(readme,encoding='utf-8')
for f in sorted(OUT.rglob('*')):
    if f.is_file() and f.suffix.lower() in {'.png','.svg'}:PROVENANCE.append(dict(file=f.relative_to(OUT).as_posix(),input='verified EDA tables / snapshots / read-only raw views',producer='scripts/complete_report_eda.py',sha256=sha(f)))
pd.DataFrame(PROVENANCE).to_csv(OUT/'checks/output_provenance.csv',index=False)
files=[f for f in sorted(OUT.rglob('*')) if f.is_file() and f!=OUT/'manifest.csv']
pd.DataFrame([dict(file=f.relative_to(OUT).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in files]).to_csv(OUT/'manifest.csv',index=False)
db.close()
log(json.dumps(dict(output=str(OUT),requirements=len(REGISTRY),figures=len(REGISTRY)*2,checks=len(CHECKS),pass_all=True,elapsed_seconds=round(time.time()-START,1)),ensure_ascii=False))
