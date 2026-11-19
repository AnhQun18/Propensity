"""Observed lifecycle, exhaustive coverage denominators and observed-date gaps."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from complete_eda_figures_moscow import load, OUT, BASE, publish, INDEX


def runs(a):
    # Histogram of consecutive run lengths, streaming every customer; no sampling.
    counts={}
    for row in a:
        boundaries=np.r_[0,np.flatnonzero(row[1:]!=row[:-1])+1,len(row)]
        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
            key=(int(row[lo]),int(hi-lo));counts[key]=counts.get(key,0)+1
    return counts


def main():
    ids,months,arrays,y=load();inventory=pd.read_csv(BASE/'source_inventory.csv').set_index('source')
    summaries=[];cohorts=[];runrows=[];gaprows=[];recencies=[]
    for source in ['trx','geo','dialog']:
        r=inventory.loc[source];first=pd.Timestamp(r.first_month);last=pd.Timestamp(r.last_month)
        valid=(months>=first)&(months<=last)
        if r.first_month_partial:valid &= months!=first
        grid=arrays[source][:,valid]>0;sm=months[valid]
        n=grid.shape[1];has=grid.any(axis=1)
        start=np.argmax(grid,axis=1);end=n-1-np.argmax(grid[:,::-1],axis=1)
        indices=np.arange(n)[None,:]
        since=(indices>=start[:,None])&has[:,None]
        between=since&(indices<=end[:,None])
        for name,mask in [('Tất cả tháng nguồn đầy đủ',np.ones_like(grid,bool)),('Từ lần đầu thấy',since),('Giữa lần đầu và cuối',between)]:
            den=int(mask.sum());num=int((grid&mask).sum())
            summaries.append(dict(source=source,denominator=name,customer_months=den,active_customer_months=num,
                                  coverage=num/den if den else np.nan,no_log_customers=int((~has).sum()),population=len(ids)))
        for (state,length),count in runs(grid).items():runrows.append(dict(source=source,active=state,length_months=length,runs=count))
        for j,month in enumerate(sm):
            cohort=has&(start==j)
            for age in range(n-j):
                cohorts.append(dict(source=source,first_observed=str(month.date()),age=age,
                    customers=int(cohort.sum()),active=int(grid[cohort,j+age].sum()),
                    retention=float(grid[cohort,j+age].mean()) if cohort.any() else np.nan))
        daily=pd.read_parquet(OUT/f'{source}_daily.parquet',columns=['client_id','day','first_utc','last_utc','gap_days'])
        daily['month']=pd.to_datetime(daily.day).dt.to_period('M').dt.to_timestamp()
        gap=daily.groupby('client_id').gap_days.agg(['count','median','mean','std','max']).reindex(ids)
        gap['cv']=gap['std']/gap['mean'];gap.loc[gap['count']<2,'cv']=np.nan
        for metric in ['count','median','mean','max','cv']:
            v=gap[metric].dropna()
            gaprows.append(dict(source=source,metric=metric,defined_customers=len(v),undefined_customers=len(ids)-len(v),
                median=float(v.median()) if len(v) else np.nan,p90=float(v.quantile(.9)) if len(v) else np.nan))
        # Last real instant in each month, then cumulatively propagate ONLY past instants.
        lasttime=daily.groupby(['client_id','month']).last_utc.max().unstack().reindex(index=ids,columns=months)
        lasttime=lasttime.ffill(axis=1).shift(1,axis=1)
        for t,month in enumerate(months):
            if not np.isfinite(y[:,t]).any():continue
            cut=month-pd.Timedelta(hours=3)
            rec=(cut-lasttime.iloc[:,t]).dt.total_seconds().to_numpy()/86400
            for b in [0,1]:
                mask=y[:,t]==b;known=mask&np.isfinite(rec)
                recencies.append(dict(source=source,anchor=str(month.date()),purchase=b,population=int(mask.sum()),
                    defined=int(known.sum()),no_prior_observed_event=int((mask&~np.isfinite(rec)).sum()),
                    mean_days=float(rec[known].mean()) if known.any() else np.nan,
                    median_days=float(np.median(rec[known])) if known.any() else np.nan))
    summary=pd.DataFrame(summaries);cohort=pd.DataFrame(cohorts);run=pd.DataFrame(runrows);gaps=pd.DataFrame(gaprows);rec=pd.DataFrame(recencies)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    (summary.pivot(index='source',columns='denominator',values='coverage')*100).plot.bar(ax=axes[0]);axes[0].set_ylabel('% khách–tháng có log');axes[0].legend(fontsize=7)
    for source,q in run[run.active==0].groupby('source'):
        q=q.sort_values('length_months')
        axes[1].plot(q.length_months,q.runs,'o-',label=source)
    axes[1].set_yscale('log');axes[1].set_xlabel('Số tháng liên tiếp không log');axes[1].set_ylabel('Số chuỗi, log');axes[1].legend()
    run.to_csv(OUT/'lifecycle_run_lengths.csv',index=False)
    publish('lifecycle_coverage_denominators',fig,summary,'Vòng đời quan sát: ba mẫu số độ phủ và chuỗi tháng trống',
        'Chỉ tháng nguồn đầy đủ; mọi khách nằm trong mẫu số tất cả tháng. Hai mẫu số còn lại dựa vào lần đầu/cuối có log '
        'trong khoảng tháng đầy đủ; khách không log được ghi riêng. Chuỗi trống gồm cả đầu/cuối phạm vi nguồn.',
        'Cho thấy độ phủ thay đổi ra sao khi thay mẫu số và độ dài các chuỗi không thấy log.',
        '; '.join(f'{r.source}/{r.denominator}: {r.coverage*100:.2f}% ({r.active_customer_months:,}/{r.customer_months:,})' for r in summary.itertuples()),
        'Lần đầu/cuối chỉ là trong dữ liệu công bố. Mẫu số giữa đầu/cuối dùng tương lai, chỉ phục vụ hồi cứu; '
        'không đưa trạng thái đó vào feature quá khứ. Chuỗi biên bị kiểm duyệt.')
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,source in zip(axes,['trx','geo','dialog']):
        q=cohort[cohort.source==source].pivot(index='first_observed',columns='age',values='retention')
        im=ax.imshow(q.to_numpy()*100,aspect='auto',vmin=0,vmax=100,cmap='Blues');ax.set_title(source)
        ax.set_yticks(np.arange(len(q)),q.index,fontsize=6);ax.set_xlabel('Tuổi quan sát, tháng')
        fig.colorbar(im,ax=ax,label='% cohort có log',shrink=.7)
    publish('lifecycle_first_observed_cohorts',fig,cohort,'Cohort theo tháng đầu có log và duy trì hoạt động quan sát',
        'Cohort=tháng đầu có log trong phạm vi tháng nguồn đầy đủ; mọi khách cohort giữ nguyên mẫu số ở mọi tuổi đủ quan sát. '
        'Khách không log không có cohort, số lượng ghi ở lifecycle_coverage_denominators.csv. Ô tương lai giữ missing.',
        'So diễn tiến hoạt động của các nhóm lần đầu được quan sát vào các thời điểm khác nhau.',
        'Mọi tuổi có n cohort và số active trong CSV; đường chéo biên không được coi là retention=0.',
        'Cohort đầu nguồn có thể chứa khách lâu năm; không gọi là khách mới ngân hàng. Tháng đầu không trọn bị loại, '
        'do đó cohort không phải ngày đầu thật của nguồn. Không so tuổi chưa đủ theo dõi.')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,source in zip(axes,['trx','geo','dialog']):
        for b,color in [(0,'#2463a6'),(1,'#d4772b')]:
            q=rec[(rec.source==source)&(rec.purchase==b)]
            ax.plot(pd.to_datetime(q.anchor),q.median_days,'o-',color=color,label=f'Mua={b}')
        ax.set_title(source);ax.set_ylabel('Recency trung vị, ngày');ax.tick_params(axis='x',rotation=35);ax.legend(fontsize=8)
    publish('all_history_recency_by_purchase_month',fig,rec,'Recency toàn lịch sử trước đầu tháng mua',
        'Thời gian từ cutoff 00:00 Moscow tới lần cuối có sự kiện strictly trước cutoff. Tính trên mọi lịch sử có sẵn, '
        'không chỉ trong tháng trước. Khách chưa có sự kiện giữ undefined và được đếm riêng. Đủ 12 tháng nhãn.',
        'Đối chiếu độ gần đây của hoạt động trước tháng mua, không sử dụng log trong/sau tháng mua.',
        'CSV cung cấp median/mean và số khách có/không có lịch sử theo nguồn, tháng và nhóm nhãn; '
        'không diễn giải recency thiếu như giá trị 0 hoặc khách không hoạt động thực tế.',
        'Chưa phân tầng TRX/độ dài lịch sử và chưa có CI. Recency dựa trên giả định timestamp nguồn UTC.')
    gaps.to_csv(OUT/'observed_day_boundary_gaps.csv',index=False)
    old=pd.read_csv(OUT/'figure_index.csv');idx=pd.concat([old,pd.DataFrame(INDEX)]).drop_duplicates('figure',keep='last');idx.to_csv(OUT/'figure_index.csv',index=False)
    status=json.loads((OUT/'execution_status.json').read_text(encoding='utf-8'))
    status['done']+=['coverage_three_denominators_runs','first_observed_retention_cohorts','all_history_cutoff_recency']
    status['figures']=len(idx)
    (OUT/'execution_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
