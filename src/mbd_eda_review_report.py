"""Tables, linear-scale figures and Vietnamese report for review_mbd_eda.py."""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib import dates as mdates
from matplotlib.ticker import FuncFormatter

from review_mbd_eda import (EVENTS,SOURCES,COLORS,activity_bands,purposeful_groups,
                            nullable_months,save_json)
from mbd_mini_eda.statistics import numeric_summary


def number(value):
    if pd.isna(value):
        return 'null'
    if isinstance(value,(float,np.floating)) and not float(value).is_integer():
        return f'{value:,.3f}'
    return f'{int(value):,}' if isinstance(value,(int,float,np.number)) else str(value)


def md(frame, columns=None):
    frame=frame if columns is None else frame[columns]
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(number(v).replace('|','/') for v in row)+' |')
    return '\n'.join(lines)


def styled(title,size=(13,7),rows=1):
    fig=Figure(figsize=size,layout='constrained',facecolor='white')
    axes=np.atleast_1d(fig.subplots(rows,1,sharex=rows>1))
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle(title,fontsize=15,weight='bold')
    return fig,axes


def save(fig,directory,name):
    fig.savefig(directory/(name+'.png'),dpi=160)
    fig.clear()


def matrix_plot(frame,directory,name,title,percent=False):
    vals=frame.to_numpy(dtype=float)
    fig=Figure(figsize=(max(11,len(frame.columns)*.65),max(3,len(frame)*.62+1.8)),layout='constrained')
    ax=fig.subplots()
    cmap=__import__('matplotlib').colormaps['Greens'].with_extremes(bad='#dddddd')
    ax.imshow(np.ma.masked_invalid(vals),cmap=cmap,aspect='auto',vmin=0,vmax=100 if percent else max(1,np.nanmax(vals)))
    ax.set_xticks(range(len(frame.columns)),frame.columns,rotation=55,ha='right',fontsize=9)
    ax.set_yticks(range(len(frame)),frame.index,fontsize=10)
    for r in range(len(frame)):
        for c in range(len(frame.columns)):
            v=vals[r,c]
            label='null' if np.isnan(v) else f'{v:.1f}%' if percent else number(v)
            maximum=100 if percent else max(1,np.nanmax(vals))
            ax.text(c,r,label,ha='center',va='center',fontsize=8,
                    color='white' if np.isfinite(v) and v/maximum>.58 else '#183d37')
    ax.set_title(title,pad=18,fontsize=14,weight='bold')
    save(fig,directory,name)


def figures(args,clients,cube,months,tables,groups,values):
    out=args.output_dir/'figures';out.mkdir(parents=True,exist_ok=True)
    totals=cube.sum(axis=2,dtype=np.uint64)
    for k,(name,indices) in enumerate(groups.items()):
        if not indices:
            continue
        frame=pd.DataFrame(totals[indices],index=[f'KH-{i+1}' for i in range(len(indices))],columns=months)
        matrix_plot(frame,out,'coverage_group_'+str(k+1),
                    f'{name} · tổng log / khách–tháng · toàn bộ lịch sử')
    matrix_plot(tables['behavior_rates'].set_index('source')[['coverage_percent']],out,'behavior_rates_matrix',
                'Tỷ lệ khách có log · toàn lịch sử · mẫu số '+number(len(clients)),True)
    coverage=tables['monthly_coverage'].pivot(index='source',columns='month',values='coverage_percent')
    matrix_plot(coverage.reindex(EVENTS),out,'behavior_monthly_matrix',
                'Ma trận tỷ lệ khách có log theo tháng (%) · null = chưa có log nguồn ở khung sớm',True)
    bands=tables['group_monthly_coverage'].pivot(index='activity_group',columns='month',values='coverage_percent')
    matrix_plot(bands,out,'coverage_population_groups','Độ phủ theo nhóm phân phối · tất cả khách, gồm nhóm 0 log',True)
    # Separate linear axes keep sparse dialog changes visible without log.
    fig,axes=styled('Nhịp hành vi theo ngày · số log gốc · trục Y riêng cho từng nguồn',(14,10),3)
    for i,source in enumerate(EVENTS):
        part=tables['daily_behavior'].query('source==@source')
        ax=axes[i]
        ax.plot(pd.to_datetime(part.day),part.events,color=COLORS[i],linewidth=.8)
        ax.set_ylabel(source+'\nlog / ngày')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:,.0f}'))
        ax.text(.01,.94,'Y tuyến tính; null ngoài khoảng quan sát của nguồn',transform=ax.transAxes,va='top',fontsize=8)
        ax.set_ylim(bottom=0)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    axes[-1].tick_params(axis='x',rotation=45)
    save(fig,out,'daily_behavior_linear')
    fig,axes=styled('Số log mỗi tháng · toàn lịch sử · trục Y tuyến tính riêng',(14,9),3)
    for i,source in enumerate(EVENTS):
        vals=cube[:,:,i].sum(axis=0).astype(float)
        vals[(months<'2022-01')&(vals==0)]=np.nan
        axes[i].plot(pd.to_datetime(months),vals,color=COLORS[i],marker='.',linewidth=1.5)
        axes[i].set_ylabel(source+'\nlog / tháng');axes[i].set_ylim(bottom=0)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    save(fig,out,'monthly_activity')
    # Unequal bins are categorical interval labels. No log-spaced positions.
    h=tables['amount_raw_bins']
    def edge(v):
        if np.isneginf(v):return '-∞'
        if np.isposinf(v):return '+∞'
        return f'{v/1e9:g} tỷ' if v>=1e9 else f'{v/1e6:g} triệu' if v>=1e6 else f'{v/1e3:g} nghìn' if v>=1e3 else f'{v:g}'
    labels=[f'[{edge(a)}, {edge(b)})' for a,b in zip(h.left,h.right)]
    fig,axes=styled('Amount gốc · số giao dịch theo khoảng · giữ đầy đủ đuôi phân phối',(14,6))
    axes[0].bar(range(len(h)),h.n,color=COLORS[0]);axes[0].set_xticks(range(len(h)),labels,rotation=55,ha='right')
    axes[0].set_ylabel('Số giao dịch / bin (độ rộng bin không bằng nhau)')
    for i,n in enumerate(h.n):
        axes[0].annotate(number(n),(i,n),xytext=(0,4),textcoords='offset points',ha='center',fontsize=7)
    save(fig,out,'amount_histogram')
    h=tables['amount_zoom'];r=tables['amount_full'].iloc[0]
    shown=int(h.n.sum());tail=int(r.finite_n-shown)
    fig,axes=styled(f'Amount gốc · zoom 0–P99 · {number(shown)} quan sát; ngoài zoom: {number(tail)}',(13,5))
    axes[0].bar(h.left,h.n,width=h.right-h.left,align='edge',color=COLORS[0],edgecolor='white',linewidth=.5)
    axes[0].set_xlabel('Amount lưu trong file · chưa quy đổi currency');axes[0].set_ylabel('Số giao dịch')
    save(fig,out,'amount_zoom_linear')
    quantiles=r[['p25','median','p75','p95','p99']].to_numpy(dtype=float)
    fig,axes=styled('Phân vị Amount gốc · trục tuyến tính · chưa quy đổi currency',(11,4))
    axes[0].scatter(quantiles,['P25','Median','P75','P95','P99'],color=COLORS[0])
    axes[0].set_xlabel('Amount lưu trong file')
    save(fig,out,'amount_quantiles')
    rates=tables['target_monthly']
    fig=Figure(figsize=(14,8),layout='constrained')
    axs=fig.subplots(2,2)
    for i,(target,part) in enumerate(rates.groupby('target',sort=True)):
        ax=axs.flat[i];dates=np.arange(len(part));y=part.positive_percent.to_numpy()
        ax.plot(dates,y,marker='o',color=COLORS[i],linewidth=2)
        pooled=float(part.positive_n.sum()/part.labelled_n.sum()*100)
        ax.axhline(pooled,color='#888888',linestyle='--',linewidth=1,label=f'Toàn kỳ {pooled:.4f}%')
        span=max(float(np.ptp(y)),pooled*.12,.0001)
        ax.set_ylim(max(0,float(y.min())-span*.15),float(y.max())+span*.22)
        for x,v,p in zip(dates,y,part.positive_n):
            ax.annotate(str(int(p)),(x,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
        ax.set_title(target+' · Y zoom riêng, số cạnh điểm = mẫu dương')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.3f}%'))
        ax.set_xticks(dates,pd.to_datetime(part.mon).dt.strftime('%m/%Y'),rotation=40,ha='right',fontsize=8)
        ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle('Biến động tỷ lệ nhãn theo tháng · cùng 100.224 khách/tháng · trục tuyến tính',fontsize=14,weight='bold')
    save(fig,out,'target_by_month')


def write_review(args,cache,clients,cube,months,tables,sample,inventory,records,schemas,fingerprint,cuts):
    out=args.output_dir;folder=out/'tables/review';folder.mkdir(parents=True,exist_ok=True)
    # Remove only named generated artefacts superseded by this review.
    obsolete=['figures/amount_log_histogram.png','tables/descriptive.transactions/amount_log_histogram.csv',
              'figures/source_coverage.png','figures/source_count_distribution.png','figures/target_prevalence.png']
    for relative in obsolete:
        path=out/relative
        if path.is_file():path.unlink()
    n=len(clients);totals=cube.sum(axis=1,dtype=np.uint64);union=totals.sum(axis=1)
    values=nullable_months(cube,months)
    groups=purposeful_groups(union)
    counts=cube.sum(axis=2,dtype=np.uint64)
    # Cross-check every retained count with the raw inventory and earlier cache.
    for i,s in enumerate(EVENTS):
        raw=sum(r['rows'] for r in records if r['source']==s)
        assert int(totals[:,i].sum())==raw, f'{s}: invalid ID/time rows need explicit audit'
    old_cache=args.cache_dir.parent/'behavior_coverage/matrix.npz'
    old_reconciled=False
    if old_cache.exists():
        with np.load(old_cache,allow_pickle=False) as old:
            old_reconciled=(np.array_equal(old['client_ids'],clients.client_id.to_numpy())
                            and np.array_equal(old['months'],months)
                            and np.array_equal(old['counts'],counts))
        if not old_reconciled:
            raise ValueError('Old matrix cache disagrees with full-history raw aggregation')
    stats=[];outliers=[]
    for i,s in enumerate((*EVENTS,'any_source')):
        v=totals[:,i] if i<3 else union
        for population,part in [('all_customers',v),('customers_with_log',v[v>0])]:
            stats.append(numeric_summary(pd.Series(part),source=s,population=population))
        q1,median,q3,p95,p99=np.quantile(v,[.25,.5,.75,.95,.99]);lower=q1-1.5*(q3-q1);upper=q3+1.5*(q3-q1)
        outliers.append(dict(source=s,rows=n,q1=q1,median=median,q3=q3,lower_fence=lower,upper_fence=upper,
            outlier_n=int(((v<lower)|(v>upper)).sum()),above_p99_n=int((v>p99).sum()),max=int(v.max())))
    tables['events_per_customer']=pd.DataFrame(stats)
    tables['customer_count_outliers']=pd.DataFrame(outliers)
    tables['behavior_rates']=pd.DataFrame([dict(source=s,population=n,with_log=int((totals[:,i]>0).sum()),
        without_log=int((totals[:,i]==0).sum()),coverage_percent=float((totals[:,i]>0).mean()*100)) for i,s in enumerate(EVENTS)])
    masks=sum(((totals[:,i]>0).astype(np.uint8)<<i) for i in range(3))
    combos=[]
    for mask in range(8):
        count=int((masks==mask).sum())
        combos.append(dict(trx=int(bool(mask&1)),geo=int(bool(mask&2)),dialog=int(bool(mask&4)),
                           customers=count,population=n,percent=count/n*100))
    tables['behavior_combinations']=pd.DataFrame(combos)
    monthly=[];daily=[];day_outliers=[]
    for i,s in enumerate(EVENTS):
        for j,m in enumerate(months):
            logged=int((cube[:,j,i]>0).sum());unknown=int(np.isnan(values[:,j,i]).sum())
            monthly.append(dict(source=s,month=m,events=int(cube[:,j,i].sum()),customers_with_log=logged,
                population=n,unknown_customers=unknown,coverage_percent=np.nan if unknown==n else logged/n*100))
        raw=tables['daily_'+s].assign(day=lambda f:pd.to_datetime(f.day)).set_index('day').events
        days=pd.date_range(min(tables['time_'+x].first_event.iloc[0] for x in EVENTS).normalize(),
                           max(tables['time_'+x].last_event.iloc[0] for x in EVENTS).normalize(),freq='D')
        series=raw.reindex(days)
        inside=(days>=raw.index.min())&(days<=raw.index.max())
        series.loc[inside]=series.loc[inside].fillna(0)
        daily.extend(dict(source=s,day=d.strftime('%Y-%m-%d'),events=v) for d,v in series.items())
        v=series.dropna();q1,q3=v.quantile([.25,.75]);upper=q3+1.5*(q3-q1)
        day_outliers.append(dict(source=s,observed_days=len(v),q1=q1,q3=q3,upper_fence=upper,
                                 outlier_days=int((v>upper).sum()),max_day=str(v.idxmax().date()),max_events=int(v.max())))
    tables['monthly_coverage']=pd.DataFrame(monthly)
    tables['daily_behavior']=pd.DataFrame(daily)
    tables['daily_count_outliers']=pd.DataFrame(day_outliers)
    bands,_=activity_bands(union);group_rows=[]
    for band in sorted(np.unique(bands)):
        part=counts[bands==band]
        for j,m in enumerate(months):
            group_rows.append(dict(activity_group='B'+str(band),month=m,population=len(part),
                min_total=int(union[bands==band].min()),max_total=int(union[bands==band].max()),
                coverage_percent=float((part[:,j]>0).mean()*100)))
    tables['group_monthly_coverage']=pd.DataFrame(group_rows)
    selected_rows=[]
    for name,indices in groups.items():
        for r,index in enumerate(indices):
            selected_rows.extend(dict(group=name,display_id=f'KH-{r+1}',month=m,events=int(counts[index,j]),
                                      full_history_events=int(union[index])) for j,m in enumerate(months))
    tables['purposeful_coverage']=pd.DataFrame(selected_rows)
    # Keep identity mappings and complete null-aware monthly grids locally.
    pd.DataFrame([dict(group=name,display_id=f'KH-{j+1}',client_id=clients.client_id.iloc[i])
                  for name,indices in groups.items() for j,i in enumerate(indices)]).to_parquet(args.sample_dir/'coverage_selection.parquet',index=False)
    indices=pd.Index(clients.client_id).get_indexer(sample.client_id)
    grid=pd.DataFrame({'client_id':np.repeat(sample.client_id.to_numpy(),len(months)),
                       'month':np.tile(months,len(sample))})
    for i,s in enumerate(EVENTS):
        grid[s+'_events']=pd.array(values[indices,:,i].reshape(-1),dtype='Int64')
        grid[s+'_raw_events']=cube[indices,:,i].reshape(-1)
        grid[s+'_missing_reason']=np.where(grid[s+'_events'].isna(),'unobserved_early_source',None)
    grid.to_parquet(args.sample_dir/'customer_month.parquet',index=False)
    # Main sample diagnostics are unweighted: inspect how much the allocation
    # actually changed the empirical distribution instead of hiding differences.
    validation=[]
    for i,s in enumerate((*EVENTS,'any_source')):
        full=totals[:,i] if i<3 else union;small=full[indices]
        support=np.unique(np.concatenate([full,small]))
        ks=np.abs(np.searchsorted(np.sort(full),support,side='right')/len(full)
                  -np.searchsorted(np.sort(small),support,side='right')/len(small)).max()
        for metric,fn in [('mean',np.mean),('median',np.median),('p95',lambda v:np.quantile(v,.95)),
                          ('max',np.max),('zero_percent',lambda v:(v==0).mean()*100)]:
            validation.append(dict(source=s,metric=metric,population_value=float(fn(full)),sample_value=float(fn(small))))
        validation.append(dict(source=s,metric='KS_distance',population_value=0.,sample_value=float(ks)))
    tables['sampling_validation']=pd.DataFrame(validation)
    targets=cache.db.execute('SELECT mon,target_1,target_2,target_3,target_4 FROM src_targets').df()
    sampled=cache.db.execute('SELECT s.* FROM src_targets s JOIN selected_clients c USING(client_id)').df()
    rates=[];label_checks=[]
    for target in ['target_1','target_2','target_3','target_4']:
        for mon,part in targets.groupby('mon',sort=True):
            labelled=int(part[target].notna().sum());positive=int(part[target].eq(1).sum())
            rates.append(dict(mon=mon,target=target,positive_n=positive,labelled_n=labelled,positive_percent=positive/labelled*100))
        label_checks.append(dict(target=target,population_positive_n=int(targets[target].eq(1).sum()),
            population_percent=float(targets[target].eq(1).mean()*100),sample_positive_n=int(sampled[target].eq(1).sum()),
            sample_percent=float(sampled[target].eq(1).mean()*100)))
    tables['target_monthly']=pd.DataFrame(rates)
    tables['sampling_labels']=pd.DataFrame(label_checks)
    tables['sample_inventory']=pd.DataFrame(inventory)
    schema_rows=[]
    for source,schema in schemas.items():
        rows=sum(r['rows'] for r in records if r['source']==source)
        for field in schema:
            schema_rows.append(dict(source=source,path=SOURCES[source],rows=rows,column=field.name,dtype=str(field.type),
                grain='customer_sequence' if source.startswith('ptls_') else 'customer_month' if source=='targets'
                else 'customer' if source=='clients' else 'event'))
        if 'fold' not in schema.names:
            schema_rows.append(dict(source=source,path=SOURCES[source],rows=rows,column='fold',dtype='hive partition integer',
                                    grain=schema_rows[-1]['grain']))
    tables['source_schema']=pd.DataFrame(schema_rows)
    # Serialize array quantiles into scalar columns, so CSVs are easy to reuse.
    for name in ['amount_by_currency','amount_full']:
        frame=tables[name]
        for i,col in enumerate(['p25','median','p75','p95','p99']):
            frame[col]=frame.quantiles.map(lambda q:q[i] if q is not None else np.nan)
        tables[name]=frame.drop(columns='quantiles')
    for name,frame in tables.items():
        if name.startswith(('month_','daily_','time_')) and name not in ('daily_behavior','daily_count_outliers'):
            continue # customer-level cached tables stay local
        frame.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig',na_rep='null')
    # Keep standard paths valid for existing consumers.
    quality=out/'tables/descriptive.quality';quality.mkdir(parents=True,exist_ok=True)
    tables['missing_values'].replace({'source':{'trx':'transactions'}}).to_csv(quality/'missing_values.csv',index=False,encoding='utf-8-sig')
    tables['duplicates'].replace({'source':{'trx':'transactions'}}).to_csv(quality/'duplicates.csv',index=False,encoding='utf-8-sig')
    transaction_folder=out/'tables/descriptive.transactions';transaction_folder.mkdir(parents=True,exist_ok=True)
    tables['amount_raw_bins'].assign(scope='all_scanned_finite_amounts').to_csv(
        transaction_folder/'amount_histogram.csv',index=False,encoding='utf-8-sig')
    figures(args,clients,cube,months,tables,groups,values)
    write_report(args,clients,months,tables,groups,fingerprint,cuts,old_reconciled)
    catalog=[];manifest={'tables':[],'figures':[],'report':'report.md','metadata':'metadata.json','catalog':'data_catalog.md'}
    for path in sorted((out/'tables').rglob('*.csv')):
        frame=pd.read_csv(path,low_memory=False,nrows=0)
        with path.open(encoding='utf-8-sig',newline='') as handle:
            rows=sum(1 for _ in csv.reader(handle))-1
        relative=path.relative_to(out).as_posix()
        manifest['tables'].append(dict(section=path.parent.name,name=path.stem,path=relative,rows=rows,columns=list(frame.columns)))
        catalog.extend([f'## [{path.stem}]({relative})','',
                        'Nội dung: '+TABLE_DESCRIPTIONS.get(path.stem,'Bảng thống kê mô tả; xem tên section, grain và scope trong metadata.')+'.',
                        '',f'{number(rows)} dòng. Các biến: '+', '.join('`'+c+'`' for c in frame.columns)+'.',''])
    manifest['figures']=[p.relative_to(out).as_posix() for p in sorted((out/'figures').glob('*.png'))]
    (out/'data_catalog.md').write_text('# Danh mục bảng kết quả và các biến\n\n'
        'Mỗi CSV là một bảng tổng hợp. `n`, `rows`, `population` là mẫu số tương ứng; `rate` là tỷ lệ 0–1; '
        '`percent` là 0–100. `null` chỉ giá trị chưa quan sát/không xác định, không thay bằng 0. '
        'Phân vị dùng nội suy tuyến tính; IQR = P75 − P25. Các cột `scope` ghi phạm vi.\n\n'+'\n'.join(catalog),encoding='utf-8')
    metadata_path=out/'metadata.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    metadata['review']=dict(created_at=datetime.now(timezone.utc).isoformat(),status='completed',
        source_fingerprint=fingerprint,source_fingerprint_method='all parquet paths, size, mtime_ns, row count and physical schema',
        event_scope='all_history_all_detail_rows',null_policy='early zero geo/dialog customer-months before 2022-01 become null; all positive counts retained',
        amount_transform='none',chart_scales='linear; target-specific zoom Y',duplicate_scope='full across all shards',
        sample_customers=len(sample),sample_seed=args.seed,cache='outputs/mbd_review_cache',
        old_matrix_reconciled=old_reconciled)
    save_json(metadata_path,metadata);save_json(out/'manifest.json',manifest)


TABLE_DESCRIPTIONS={
 'source_schema':'8 bảng nguồn: đường dẫn, số dòng, grain, từng cột và kiểu lưu thực tế',
 'missing_values':'Null thuộc tính theo cột trên toàn bộ dữ liệu; không đếm ngày không có sự kiện',
 'duplicates':'Trùng khóa, trùng nguyên dòng và ứng viên trùng client–timestamp; số dòng dư sau bản đầu',
 'sequence_quality':'Null phần tử và độ dài mảng ptls so với mảng event_time trên cùng hàng',
 'sequence_reconciliation':'Đối chiếu số phần tử ptls với số log detail theo cùng khách, gồm độ dài tối đa và số khách chênh lệch',
 'embedding_element_quality':'Null vector, chiều khác 768, null hoặc không hữu hạn bên trong từng vector detail/ptls',
 'null_target_context':'Null diff_trans_date theo nhóm trans_count để kiểm tra giả thuyết thiếu giao dịch',
 'null_pairs':'Số dòng đồng thời null hoặc chỉ một cột null trong từng cặp thuộc tính',
 'null_by_event_type':'Mức thiếu từng họ thuộc tính theo mã event_type gốc',
 'amount_full':'Moment và phân vị chính xác của mọi amount hữu hạn, chưa quy đổi currency',
 'amount_by_currency':'Phân vị amount theo từng mã currency, gồm nhóm currency null',
 'amount_outliers_currency':'Cờ Tukey 1,5 IQR theo từng currency; outlier_n không phải số lỗi đã xác nhận',
 'amount_raw_bins':'Đếm amount gốc theo bin trái đóng phải mở; biên ±inf bao phủ toàn đuôi',
 'amount_zoom':'Histogram 30 bin tuyến tính trong [0,P99]; phần ngoài zoom vẫn nằm trong bảng đầy đủ',
 'events_per_customer':'Mean/min/max và phân vị tổng log mỗi khách toàn lịch sử; tách toàn dân số và nhóm có log',
 'customer_count_outliers':'Tukey 1,5 IQR và trên P99 cho số log mỗi khách trên toàn dân số',
 'daily_count_outliers':'Tukey 1,5 IQR theo số log/ngày trong khoảng đã có log từng nguồn',
 'daily_behavior':'Chuỗi ngày liên tục toàn lịch sử từng nguồn; null ngoài khoảng quan sát',
 'behavior_rates':'Ma trận tỷ lệ có log từng nguồn trên cùng dân số toàn kỳ',
 'behavior_combinations':'8 tổ hợp hiện diện trx/geo/dialog toàn kỳ; mỗi khách thuộc đúng một tổ hợp',
 'monthly_coverage':'Độ phủ nguồn theo tháng, tổng log, khách có log và khách có ô null',
 'group_monthly_coverage':'Độ phủ bất kỳ nguồn nào trong từng nhóm phân phối tổng log toàn kỳ',
 'purposeful_coverage':'Số log/tháng của nhóm minh họa cực đại, gần mean, gần median, nhỏ nhất có log, 0 log',
 'sampling_strata':'Phân bổ mẫu theo fold, mask nguồn, band tổng log; chênh lệch đơn vị điểm phần trăm',
 'sampling_validation':'So sánh phân phối tổng log từng nguồn trên population và mẫu, gồm khoảng cách KS',
 'sampling_labels':'Đối chiếu nhãn trong mẫu và population; không stratify theo nhãn tương lai',
 'sample_inventory':'Số dòng từng bảng con liên kết trong sampledata; chỉ lưu số tổng hợp trên Git',
 'target_monthly':'Đếm dương/mẫu số và tỷ lệ phần trăm mỗi nhãn theo tháng; cộng fold trước khi chia',
 'table_inventory':'Quy mô và grain các bảng đầu vào detail',
 'column_roles':'Kiểu dữ liệu và vai trò từng cột đầu vào detail',
 'relationship_summary':'Khóa nối và quan hệ giữa bảng khách, nhãn, các bảng sự kiện',
 'invalid_values':'Số dòng vi phạm điều kiện ID, timestamp, fold, amount hoặc embedding',
 'key_coverage':'Khách có hoặc không có log ID/thời gian hợp lệ theo nguồn',
 'customer_activity':'Phân phối các tổng hợp khách hàng toàn kỳ, bao gồm cả khách không có log',
 'modality_coverage':'8 tổ hợp nguồn có log trên toàn lịch sử',
 'source_count_distribution':'Phân phối số nguồn có log (0, 1, 2, 3) trên mỗi khách',
 'amount_summary':'Thống kê moment và phân vị amount; scope và phương pháp được ghi trong bảng',
 'transaction_outliers':'Cờ Tukey pooled amount toàn bộ, không tự xác nhận lỗi nghiệp vụ',
 'amount_histogram':'Histogram amount trên giá trị gốc; độ rộng và scope trong bảng',
 'transaction_categories':'Top mã giao dịch theo từng cột, kèm phần missing và remaining',
 'transaction_categories_full':'Đầy đủ tần suất mọi mã giao dịch theo cột, giữ cả missing',
 'location_categories':'Top mã vị trí theo từng cấp, kèm missing và remaining',
 'location_categories_full':'Đầy đủ tần suất các mã vị trí theo từng cột',
 'source_time_ranges':'Timestamp đầu/cuối đã ghi nhận mỗi nguồn',
 'monthly_activity':'Tổng log và số khách có log theo tháng và nguồn',
 'history_length_summary':'Phân phối chiều dài lịch sử mỗi khách, xem scope và đơn vị từng hàng',
 'geo_summary':'Phân phối số log, ngày có log và số mã vị trí khác nhau mỗi khách',
 'dialog_summary':'Phân phối số hội thoại và ngày có hội thoại mỗi khách',
 'embedding_quality':'Kiểm tra chiều, phần tử và norm embedding, không giải mã nội dung hội thoại',
 'first_last_event_month':'Phân phối tháng log đầu hoặc cuối trên khách có log',
 'first_last_event_coverage':'Số khách có/không có mốc log đầu-cuối hợp lệ',
 'customer_aggregation_checks':'Đối chiếu số khách đầu vào/đầu ra, khóa trùng và khách 0 log',
 'target_prevalence':'Số nhãn dương, âm, null và tỷ lệ trên các dòng có nhãn',
 'target_by_month_fold':'Tỷ lệ nhãn theo mon, fold; cộng tử và mẫu số trước khi gộp',
 'target_cooccurrence':'Số dòng hai target đồng thời dương và mẫu số cặp được quan sát',
 'customer_positive_history':'Khách từng có nhãn dương và số tháng dương mỗi khách',
 'positive_labels_per_row':'Phân phối số target dương đồng thời trên một dòng khách–tháng',
 'label_coverage':'Độ phủ nhãn theo tháng và fold',
}


def write_report(args,clients,months,t,groups,fingerprint,cuts,reconciled):
    out=args.output_dir;n=len(clients)
    all_stats=t['events_per_customer'].query("population=='all_customers'")
    amount=t['amount_full'].iloc[0];a_out=t['amount_outliers_currency']
    schema_table=[]
    for s,part in t['source_schema'].groupby('source',sort=False):
        schema_table.append(dict(bảng=part.path.iloc[0],grain=part.grain.iloc[0],dòng=int(part.rows.iloc[0]),
                                 biến=', '.join(part.column)))
    missing=t['missing_values'];nulls=missing[missing.missing_n>0].copy()
    nulls['missing_percent']=nulls.missing_rate.map(lambda v:f'{v*100:.6f}%')
    lines=['# MBD-mini — EDA toàn lịch sử và rà soát 12 yêu cầu','',
      f'Giữ **{number(n)} khách** trong mỗi phân phối nguồn; '
      'không lọc cohort theo yêu cầu đủ cả ba nguồn. Phạm vi từ log đầu 31/12/2020 đến log cuối 31/12/2022; nhãn từ 02/2022 đến 01/2023. '
      'Chỉ dùng detail để đếm sự kiện. ptls là dạng chuỗi để đối chiếu, không cộng thêm vào tổng log.','',
      'Dữ liệu gốc được giữ nguyên. Timestamp theo lịch naive lưu trong file, chưa tự đổi múi giờ. '
      'Mọi hình trong bản rà soát dùng số gốc và trục tuyến tính. Mốc tháng biên có thể chỉ chứa một phần ngày/tháng.','',
      '## 1. Mỗi bảng chứa gì và có những biến nào?','',md(pd.DataFrame(schema_table)),'',
      '`client_id` nối khách giữa các bảng; `fold` là partition chia khách. `targets` có khóa khách–tháng; '
      '`mon` là marker nhãn, bốn target là nhãn nhị phân. `trans_count` và `diff_trans_date` là trường phụ trợ chưa xác minh công thức/cutoff. '
      'Trong trx, `amount` là float32 gốc; event_type/subtype, currency và các họ src/dst là mã ẩn danh. '
      'Geo chứa ba cấp mã vị trí; dialog chứa vector float32 768 chiều, không có văn bản gốc. '
      'Các cột của ptls tương ứng là mảng theo sự kiện; event_time ở ptls được lưu int64, không tự giả định đơn vị khi chuyển thời gian.','',
      'Kiểu lưu của từng biến: [source_schema.csv](tables/review/source_schema.csv). '
      'Danh mục **tất cả bảng kết quả và từng biến trong bảng**: [data_catalog.md](data_catalog.md). '
      'Ý nghĩa cột nguồn: [từ điển MBD-mini](../dict/mbd_mini.md).','',
      'Đối chiếu số phần tử chuỗi với log detail:','',md(t['sequence_reconciliation']),'',
      '`ptls/trx` thiếu **2.020.020 phần tử ở 13 khách** so với detail/trx; chuỗi dài nhất có 10.000 phần tử, '
      'trong khi detail có khách tới 1.020.773 log. Đây là dấu hiệu có giới hạn chiều dài chuỗi; chưa xác minh cách chọn phần được giữ. '
      'Geo và dialog có tổng phần tử và số log theo khách khớp. Vì vậy, toàn bộ EDA, biểu đồ và '
      'lịch sử trong mẫu dùng **detail** làm nguồn chuẩn; không thay bằng ptls hoặc cộng hai dạng. '
      'Các bảng ptls trong sampledata được giữ đúng như nguồn để đối chiếu, không bảo đảm có mọi sự kiện detail.','',
      '## 2. Null có nghĩa gì, vì sao có null?','',
      'Null thuộc tính là thiếu giá trị trong **một sự kiện/dòng đã tồn tại**. Không có dòng sự kiện ở một tháng là trường hợp khác. '
      'Nguyên nhân thu thập chính xác không có trong dữ liệu; phần dưới phân biệt số đếm đã kiểm chứng và cách diễn giải cần xác minh.','',
      md(nulls[['source','column','rows','missing_n','missing_percent']]),'',
      '- `client_split`: không có null ở ID hoặc fold; nếu có thì không xác định được khách/nhóm chia.',
      '- `targets`: bốn target, ID, mon, fold và trans_count đầy đủ. `diff_trans_date=null` nghĩa là chênh lệch thời gian chưa được cung cấp/tính được, không có nghĩa target bị thiếu. Không tự điền 0.',
      '- `detail/trx`: không thiếu ID, timestamp, amount hoặc mã loại. Các null ở currency/src/dst nghĩa là thiếu mã thuộc tính; giữ giao dịch và dùng cờ missing nếu tạo feature.',
      '- `detail/geo` và `detail/dialog`: các dòng được ghi nhận không có null cột; việc một khách/tháng không có dòng không tạo ra null bên trong bảng raw.',
      '- `ptls`: null phần tử, null cả mảng và thiếu cả hàng khách là ba tình huống riêng. Không có chuỗi nghĩa là không có log nguồn được ghi nhận trong lịch sử; không xóa khách khỏi client_split.','',
      'Đối chiếu diff_trans_date với trans_count trên toàn bộ nhãn:','',md(t['null_target_context']),'',
      'Trong bản dữ liệu này, **toàn bộ 120.786 null của diff_trans_date trùng với trans_count=0**, '
      'và mọi dòng trans_count>0 đều có diff_trans_date. Đây là bằng chứng thực nghiệm rằng trường thiếu đi cùng '
      'việc không có giao dịch trong phạm vi trans_count; không suy ra đơn vị hay công thức chính xác của diff_trans_date.','',
      'Các cặp thuộc tính thiếu đồng thời:','',md(t['null_pairs']),'',
      'Sự thiếu đồng thời gợi ý cấu trúc cấp cha/con hoặc cơ chế ghi nhận chung; dữ liệu chưa xác nhận đó là “không áp dụng” hay lỗi thu thập. '
      'Xem [null theo event_type](tables/review/null_by_event_type.csv) và '
      '[null phần tử/độ dài chuỗi ptls](tables/review/sequence_quality.csv). '
      'ptls có 0 null phần tử ở các mảng một cấp trong bản này, khác với null thuộc tính ở detail/trx; '
      'không dùng ptls để kết luận raw không thiếu mã. '
      'Kiểm tra cả phần tử lồng bên trong embedding: [embedding_element_quality.csv](tables/review/embedding_element_quality.csv).','',
      '## 3. Kiểm tra duplicate','',md(t['duplicates']),'',
      'Đếm toàn bộ xuyên shard/batch: `duplicate_n = tổng(nhóm trùng − 1)`, không phải kết quả trên mẫu. '
      '`full_row` so sánh mọi cột và fold, kể cả nội dung embedding; null/NaN được chuẩn hóa nhất quán. '
      '`unique_key` dùng client_id cho khách/ptls và client_id–mon cho targets. '
      '`same_client_time_candidate` chỉ là ứng viên: nhiều sự kiện cùng thời điểm có thể khác thuộc tính, không tự xóa.','',
      '## 4. Giữ toàn bộ thời gian và biểu diễn khung thiếu nguồn bằng null','',
      'Mọi giao dịch, geo và hội thoại trước 01/2022 vẫn được giữ. Không cắt về giao của ba nguồn. '
      'Trong bảng khách–tháng, trước 01/2022, geo/dialog không có log được đặt `null` kèm '
      '`missing_reason=unobserved_early_source`; ô có log vẫn giữ số đếm gốc. Từ 01/2022, 0 nghĩa là không ghi nhận log, '
      'không chứng minh khách thực sự không hoạt động hoặc nguồn được thu thập liên tục. '
      'Các cột `*_raw_events` bảo toàn số đếm để đối chiếu; các thuộc tính raw không bị impute.','',
      'Lưới đầy đủ 25 tháng cho mẫu nằm tại `sampledata/mbd_mini/customer_month.parquet`. '
      'Cache ma trận toàn dân số chứa toàn lịch sử. '
      + ('Đã đối chiếu chính xác mọi ô với cache matrix.npz trước đó.' if reconciled else 'Không có cache ma trận cũ để đối chiếu.'),'',
      '## 5. Ma trận độ phủ theo nhóm có chủ đích','',
      'Xếp nhóm theo tổng số log toàn lịch sử: 8 khách nhiều nhất, 8 gần mean dân số, 8 gần median dân số, '
      '8 ít nhất nhưng có log và 8 không có log. Các nhóm không lặp khách; ties theo ID đã sắp xếp. '
      'Đây là ví dụ có chủ đích, không dùng để ước lượng tỷ lệ dân số. Mỗi ảnh có thang màu tuyến tính riêng để đọc rõ nhóm; '
      'so sánh bằng số trong ô. Số trong ô là log, không phải ngày. Mã KH chỉ có ý nghĩa trong nhóm.','']
    for i,(name,indices) in enumerate(groups.items()):
        if indices:lines.extend([f'![{name}](figures/coverage_group_{i+1}.png)',''])
    lines.extend(['Phần tổng hợp dưới dùng **tất cả khách**, chia nhóm B0 = 0 log; B1…B6 theo các mốc '
      'P25/P50/P75/P95/P99 của **khách có log**. Các biên tổng log: '+', '.join(number(x) for x in cuts)+'. '
      'Mẫu số mỗi hàng là số khách trong nhóm, không phải toàn dân số.','',
      '![Độ phủ dân số](figures/coverage_population_groups.png)','',
      '## 6. Tỷ lệ hành vi bằng bảng matrix','',md(t['behavior_rates']),'',
      '8 tổ hợp hiện diện nguồn trong toàn lịch sử (1 = có ít nhất một log):','',md(t['behavior_combinations']),'',
      '![Tỷ lệ theo tháng](figures/behavior_monthly_matrix.png)','',
      'Mẫu số tỷ lệ theo tháng là toàn bộ '+number(n)+' khách. Trước 01/2022, tỷ lệ có log là độ phủ log **đã quan sát**, '
      'không chứng minh phần còn lại không hoạt động. Khi cả nguồn không có log ở tháng sớm, tỷ lệ hiển thị null. '
      '`unknown_customers` trong monthly_coverage.csv giữ rõ số khách có ô chưa quan sát.','',
      '## 7. Trung bình phát sinh hành vi mỗi người, min/max','',
      'Tổng log mỗi người trong **toàn lịch sử**, cùng mẫu số kể cả người 0 log:','',
      md(all_stats,['source','rows','mean','min','p25','median','p75','p95','p99','max']),'',
      'Có thêm phân phối trên riêng khách có log trong [events_per_customer.csv](tables/review/events_per_customer.csv). '
      'Mean số log phản ánh độ dài và độ phủ nguồn khác nhau; không coi là tần suất đã chuẩn hóa theo ngày thu thập.','',
      '## 8. Nhịp hành vi trên số gốc, không log','',
      '![Nhịp theo ngày](figures/daily_behavior_linear.png)','',
      '![Nhịp theo tháng](figures/monthly_activity.png)','',
      'Mỗi nguồn có trục Y tuyến tính riêng để hội thoại không bị che bởi geo/trx. Ngày ngoài khoảng quan sát để null; '
      'ngày không có log nằm giữa mốc đầu/cuối nguồn ghi 0 log. Các mốc biên có thể chỉ có một phần ngày.','',
      '## 9. Amount gốc và cách chia bin','',
      md(t['amount_full'],['finite_n','mean','min','p25','median','p75','p95','p99','max']),'',
      '![Amount đầy đủ](figures/amount_histogram.png)','',
      '![Amount zoom](figures/amount_zoom_linear.png)','',
      'Biểu đồ đầy đủ dùng **số lượng trên bin** với nhãn khoảng giá trị gốc và độ rộng không bằng nhau; '
      'không diễn giải chiều cao như mật độ. Hình zoom dùng 30 bin đều trên [0,P99] để đọc khối chính; '
      'phần ngoài zoom được ghi rõ và vẫn nằm trong bảng/hình đầy đủ. Không biến đổi log hoặc cắt bỏ giá trị gốc. '
      'Currency là mã chưa quy đổi, nên thống kê pooled chỉ mô tả các số lưu trong file; '
      'xem [amount theo currency](tables/review/amount_by_currency.csv) trước khi diễn giải nghiệp vụ.','',
      '## 10. Biến động nhãn với Y zoom cho từng target','',
      '![Nhãn theo tháng](figures/target_by_month.png)','',
      'Bốn panel dùng trục tuyến tính với giới hạn Y riêng, có đường tỷ lệ toàn kỳ để đối chiếu. '
      'Y không bắt buộc bắt đầu từ 0 và không so biên độ thị giác giữa các panel. '
      'Tỷ lệ = tổng số dương các fold / tổng dòng có nhãn; số cạnh điểm là số dương thực tế. '
      'Dữ liệu chi tiết: [target_monthly.csv](tables/review/target_monthly.csv).','',
      '## 11. Resample theo phân phối và sampledata','',
      f'Lấy **{len(sample_inventory_ids(args.sample_dir)):,} khách không lặp**, seed={args.seed}, '
      'phân tầng đồng thời theo fold × tổ hợp nguồn toàn lịch sử × band tổng log. '
      'Phân bổ tỷ lệ bằng phần nguyên và phần dư lớn nhất (Hamilton), sau đó lấy ngẫu nhiên trong tầng. '
      'Không ép mọi tầng hiếm xuất hiện: tầng nhận 0 mẫu được công khai, không coi weight bù được tầng vắng. '
      'Giữ toàn bộ sự kiện, 12 tháng nhãn và chuỗi ptls của khách đã chọn.','',
      md(t['sample_inventory']),'',
      'Mẫu đã ẩn danh từ MBD-mini nằm tại [sampledata/mbd_mini](../../sampledata/mbd_mini/), được đưa lên Git cùng mã tái tạo và bảng kiểm chứng tổng hợp. '
      '`selection.parquet` chứa tầng và weight = population_n/sample_n. '
      '`customer_month.parquet` chứa đầy đủ tháng và null nguồn sớm. '
      'Đây là mẫu EDA theo toàn lịch sử; khi làm mô hình phải lấy mẫu bằng lịch sử trước cutoff để tránh dùng tương lai.','',
      md(t['sampling_validation']), '',md(t['sampling_labels']),'',
      'KS là chênh lệch lớn nhất giữa hai hàm phân phối thực nghiệm của tổng log (0 tốt hơn). '
      'Không suy ra mẫu bảo toàn chính xác mean/max hoặc tỷ lệ nhãn cực hiếm. '
      'Các số liệu báo cáo chính vẫn dùng toàn bộ raw. Xem '
      '[phân bổ từng tầng](tables/review/sampling_strata.csv).','',
      '## 12. Kiểm tra outliers','',
      'Số log mỗi khách: đánh dấu ngoài [Q1−1,5×IQR; Q3+1,5×IQR] và trên P99, trên toàn dân số kể cả 0.','',
      md(t['customer_count_outliers']),'',
      'Amount: tính Tukey **riêng từng currency**, giữ cả currency null.','',md(a_out),'',
      'Số log/ngày trong khoảng đã có log từng nguồn:','',md(t['daily_count_outliers']),'',
      'Các điểm này là quan sát cực trị cần kiểm tra, chưa phải lỗi. Với hội thoại thưa, IQR nhỏ làm nhiều điểm bị gắn cờ; '
      'so sánh thêm P99 và bối cảnh sự kiện. Không tự xóa, winsorize hoặc clip outlier. '
      'Giá trị amount cực lớn và khách có số log cực lớn giải thích vì sao mean cao hơn median; '
      'cần xác minh đơn vị và cơ chế ghi log trước khi đưa ra ngưỡng nghiệp vụ.','',
      '## Tái lập và cache','',
      '```powershell',r'.\.venv\Scripts\python.exe -u -X utf8 src/review_mbd_eda.py','```','',
      'Chạy lại cùng lệnh sẽ tái dùng từng kết quả SQL đã hoàn tất trong `outputs/mbd_review_cache/review.duckdb`. '
      'Cache lưu theo câu truy vấn và fingerprint gồm đường dẫn, byte, mtime_ns, số dòng và schema mọi shard. '
      'Raw thay đổi thì dùng thư mục cache mới; cache cũ được giữ lại. Đây là kiểm tra inventory, không phải checksum toàn bộ byte nội dung. '
      'Cache `outputs/behavior_coverage/matrix.npz` và cache quét năm 2022 cũ được giữ trên máy. '
      'Raw đầy đủ và cache giữ trên máy; chỉ mẫu MBD-mini đã ẩn danh được version cùng repo theo yêu cầu.','',
      f'Fingerprint lần rà soát: `{fingerprint}`. Các số đếm raw được đối chiếu với ma trận và tổng sự kiện; '
      'mọi bảng mới nằm ở `tables/review/`, danh mục đầy đủ ở [data_catalog.md](data_catalog.md).',''])
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def sample_inventory_ids(directory):
    return pd.read_parquet(directory/'selection.parquet',columns=['client_id'])
