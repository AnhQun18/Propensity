"""Full-population retrospective EDA, all 12 label months and full raw log history.

No sampling, model fit, prediction cutoff, or causal identification. Unknown
adjacent labels and unobserved source months remain explicitly unavailable.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/full_population_temporal_eda_20261118'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False})
SOURCES=['trx','geo','dialog']
STATES=['Neither adjacent month','Before next purchase only','After prior purchase only','Between two purchases','Adjacent label unknown']


def save(fig,name,note):
    fig.text(.01,.01,note,fontsize=8,color='#475569')
    fig.tight_layout(rect=(0,.06,1,.97))
    for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)


def extract():
    c=duckdb.connect();c.execute("SET memory_limit='3GB'");c.execute('SET threads=2');c.execute('SET preserve_insertion_order=false')
    c.execute(f"SET temp_directory='{OUT.as_posix()}/tmp'")
    ids=c.execute(f"SELECT client_id FROM read_parquet('{ROOT.as_posix()}/data/MBD-mini/client_split/**/*.parquet')").df()
    assert ids.client_id.is_unique
    labels=c.execute(f"SELECT client_id,date_trunc('month',cast(mon AS DATE)) AS month,target_1,target_2,target_3,target_4 FROM read_parquet('{ROOT.as_posix()}/data/MBD-mini/targets/**/*.parquet')").df()
    target_cols=['target_1','target_2','target_3','target_4']
    assert labels[target_cols].isin([0,1]).all().all()
    labels['y']=labels[target_cols].max(axis=1)
    labels=labels[['client_id','month','y']]
    assert not labels.duplicated(['client_id','month']).any()
    assert labels.month.nunique()>=12
    assert labels.groupby('month').client_id.nunique().eq(len(ids)).all()
    assert set(labels.client_id)==set(ids.client_id)
    bounds=[];parts={}
    for source in SOURCES:
        print('Read ALL raw files:',source,flush=True)
        path=f'{ROOT.as_posix()}/data/MBD-mini/detail/{source}/**/*.parquet'
        b=c.execute(f"SELECT count(*) n,min(event_time) first_event,max(event_time) last_event FROM read_parquet('{path}')").df().iloc[0]
        extra=",coalesce(sum(amount) FILTER(WHERE currency=11),0) amount11" if source=='trx' else ''
        a=c.execute(f"SELECT client_id,date_trunc('month',event_time+INTERVAL '3 hours') AS month,count(*) {source}_events {extra} FROM read_parquet('{path}') GROUP BY 1,2").df()
        assert int(a[source+'_events'].sum())==int(b['n'])
        assert set(a.client_id).issubset(set(ids.client_id))
        first=pd.Timestamp(b.first_event)+pd.Timedelta(hours=3);last=pd.Timestamp(b.last_event)+pd.Timedelta(hours=3)
        # Source-month availability, not proof of continuous individual observation.
        start=first.to_period('M').to_timestamp();end=last.to_period('M').to_timestamp()
        bounds.append(dict(source=source,raw_rows=int(b['n']),first_event_moscow=str(first),last_event_moscow=str(last),first_month=str(start.date()),last_month=str(end.date()),first_month_partial=bool(first.day!=1)))
        a.to_parquet(OUT/f'{source}_monthly.parquet',index=False);parts[source]=a
    earliest=min(pd.Timestamp(b['first_month']) for b in bounds)
    latest=max(labels.month.max(),max(pd.Timestamp(b['last_month']) for b in bounds))
    months=pd.date_range(earliest,latest,freq='MS')
    panel=ids.merge(pd.DataFrame({'month':months}),how='cross')
    for b in bounds:
        s=b['source'];panel=panel.merge(parts[s],on=['client_id','month'],how='left',validate='one_to_one')
        in_range=panel.month.between(pd.Timestamp(b['first_month']),pd.Timestamp(b['last_month']))
        panel[s+'_available']=in_range
        panel[s+'_events']=panel[s+'_events'].where(~in_range,panel[s+'_events'].fillna(0))
        if s=='trx':panel['amount11']=panel.amount11.where(~in_range,panel.amount11.fillna(0))
    panel=panel.merge(labels,on=['client_id','month'],how='left',validate='one_to_one')
    labels.to_parquet(OUT/'labels_any.parquet',index=False)
    panel.to_parquet(OUT/'all_customer_months.parquet',index=False)
    pd.DataFrame(bounds).to_csv(OUT/'source_inventory.csv',index=False)
    return ids,labels,panel,bounds


def analyze(ids,labels,panel,bounds):
    n=len(ids);months=sorted(labels.month.unique());checks={}
    # Every customer retained at every label month; adjacent labels joined by date.
    d=panel[panel.month.isin(months)].copy()
    for shift,name in [(1,'previous_y'),(-1,'next_y')]:
        a=labels.rename(columns={'y':name}).copy();a['month']=a.month+pd.DateOffset(months=shift)
        d=d.merge(a,on=['client_id','month'],how='left',validate='one_to_one')
    known=d.previous_y.notna()&d.next_y.notna()
    d['state']=np.select([~known,d.previous_y.eq(0)&d.next_y.eq(0),d.previous_y.eq(0)&d.next_y.eq(1),d.previous_y.eq(1)&d.next_y.eq(0)],['Adjacent label unknown',STATES[0],STATES[1],STATES[2]],default=STATES[3])
    group_counts=d.groupby(['month','state']).size().unstack('state',fill_value=0).reindex(columns=STATES,fill_value=0)
    checks['all_monthly_groups_sum_to_all_clients']=bool(group_counts.sum(axis=1).eq(n).all())
    group_counts.to_csv(OUT/'monthly_partition_counts.csv')
    d.to_parquet(OUT/'label_month_panel_with_states.parquet',index=False)
    # Full calendar inventory, including all labelled months and missing source months.
    calendar=panel.groupby('month').agg(customers=('client_id','size'),trx_events=('trx_events',lambda v:v.sum(min_count=1)),geo_events=('geo_events',lambda v:v.sum(min_count=1)),dialog_events=('dialog_events',lambda v:v.sum(min_count=1)),buyers=('y',lambda v:v.sum(min_count=1)),labelled=('y','count')).reset_index()
    calendar.to_csv(OUT/'calendar_inventory.csv',index=False)
    fig,axes=plt.subplots(4,1,figsize=(14,10),sharex=True)
    for ax,col,title in zip(axes,['trx_events','geo_events','dialog_events','buyers'],['TRX raw events','GEO raw events','DIALOG raw events','Any-product positive customer-months']):
        ax.plot(calendar.month,calendar[col],marker='o',ms=3);ax.set_ylabel(title);ax.grid(alpha=.2)
        ax.axvspan(pd.Timestamp(months[0]),pd.Timestamp(months[-1]),color='#0d9488',alpha=.08)
    axes[0].set_title(f'Full inventory: {n:,} customers; {len(months)} label months; all available log months')
    save(fig,'01_full_calendar_inventory','NaN = source/label month unavailable; never zero-filled outside source range. Shading = label span. First source month may be partial (see inventory CSV).')
    fig,ax=plt.subplots(figsize=(14,5));bottom=np.zeros(len(group_counts))
    for state,color in zip(STATES,['#cbd5e1','#2563eb','#d97706','#0d9488','#a78bfa']):
        ax.bar(range(len(group_counts)),group_counts[state],bottom=bottom,label=state,color=color);bottom+=group_counts[state].to_numpy()
    ax.set_xticks(range(len(group_counts)),[pd.Timestamp(m).strftime('%Y-%m') for m in group_counts.index],rotation=35);ax.set_ylabel('Customers');ax.set_title('Every label month is partitioned into exhaustive adjacent-purchase groups');ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3)
    save(fig,'02_exhaustive_monthly_groups',f'Groups sum to {n:,} in EACH month. Labels at t-1 and t+1; month t may independently have a purchase. Edge labels remain unknown.')
    # Month-by-month before/after, without selecting a short balanced cohort.
    lag_rows=[]
    for lag in [-1,1]:
        base=labels.rename(columns={'month':'target_month'}).copy();base['month']=base.target_month+pd.DateOffset(months=lag)
        base=base.merge(panel.drop(columns='y'),on=['client_id','month'],how='left',validate='one_to_one')
        for (target,y),g in base.groupby(['target_month','y']):
            for s in SOURCES:
                observed=g[s+'_available'].fillna(False)
                k=int((g[s+'_events'].gt(0)&observed).sum());den=int(observed.sum())
                lag_rows.append(dict(target_month=target,lag=lag,purchase=int(y),source=s,total_customers=len(g),observed_customers=den,unavailable_customers=len(g)-den,active_customers=k,rate=k/den if den else np.nan))
    lag=pd.DataFrame(lag_rows);lag.to_csv(OUT/'before_after_monthly_rates.csv',index=False)
    rr=lag.pivot(index=['target_month','lag','source'],columns='purchase',values='rate').rename(columns={0:'nonbuyer_rate',1:'buyer_rate'}).reset_index();rr['rate_ratio']=rr.buyer_rate/rr.nonbuyer_rate.replace(0,np.nan)
    rr.to_csv(OUT/'before_after_monthly_rate_ratios.csv',index=False)
    fig,axes=plt.subplots(3,2,figsize=(15,9),sharex=True)
    for i,s in enumerate(SOURCES):
        for j,k in enumerate([-1,1]):
            ax=axes[i,j]
            for y,color,label in [(1,'#0d9488','Purchase in target month'),(0,'#64748b','No purchase in target month')]:
                a=lag[(lag.source==s)&(lag.lag==k)&(lag.purchase==y)].sort_values('target_month');ax.plot(a.target_month,a.rate,'o-',ms=4,color=color,label=label)
            ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_title(s.upper()+(' in PREVIOUS month' if k==-1 else ' in FOLLOWING month'));ax.grid(alpha=.2)
            if i==0:ax.legend(fontsize=7)
    for ax in axes[-1]:ax.tick_params(axis='x',rotation=35)
    fig.suptitle('Before purchase vs after purchase | all customers and all 12 target months')
    save(fig,'03_before_vs_after_all_label_months','X = target month; Y = share with source events in t-1 or t+1. Gaps = missing follow-up, not zero. Repeated purchases included; observational, not causal.')
    # Distinguish prior vs following purchase conditional on whether same month buys.
    assoc=[]
    for (state,y),g in d.groupby(['state','y']):
        for s in SOURCES:
            observed=g[s+'_available'];den=int(observed.sum())
            assoc.append(dict(state=state,current_purchase=int(y),source=s,total_customer_months=len(g),observed_customer_months=den,unavailable_customer_months=len(g)-den,active=int((g[s+'_events'].gt(0)&observed).sum()),rate=g.loc[observed,s+'_events'].gt(0).mean() if den else np.nan))
    assoc=pd.DataFrame(assoc);assoc.to_csv(OUT/'adjacent_purchase_source_associations.csv',index=False)
    checks['state_by_current_label_exhausts_full_year']=int(assoc[assoc.source.eq('trx')].total_customer_months.sum())==len(labels)
    fig,axes=plt.subplots(2,3,figsize=(16,9),sharey=True)
    for i,y in enumerate([0,1]):
        for j,s in enumerate(SOURCES):
            ax=axes[i,j];a=assoc[(assoc.source==s)&(assoc.current_purchase==y)].set_index('state').reindex(STATES)
            ax.bar(range(5),a.rate,color=['#94a3b8','#2563eb','#d97706','#0d9488','#a78bfa']);ax.set_ylim(0,1.12);ax.set_yticks([0,.25,.5,.75,1]);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_xticks(range(5),['Neither','Before only','After only','Between','Unknown'],rotation=25)
            ax.set_title(f'{s.upper()} | purchase in current month = {y}')
            for z,v in enumerate(a.rate):
                if np.isfinite(v):ax.text(z,v+.015,f'{v:.1%}',ha='center',fontsize=8)
    fig.suptitle('Is source activity associated with a prior purchase or an upcoming purchase?')
    save(fig,'04_prior_next_purchase_groups','All 12 label months registered. Rates use source-observed months; n/unknown in CSV. Current-month purchase stratified; pooled customer-month rates, no independence or causal claim.')
    # Financial contribution on ALL raw TRX months; purchase-frequency groups exhaust ids.
    freq=labels.groupby('client_id').y.sum().rename('positive_months').reset_index()
    freq['purchase_group']=np.select([freq.positive_months.eq(0),freq.positive_months.eq(1)],['No purchase observed','One positive month'],default='Multiple positive months')
    value=panel[panel.trx_available].groupby('client_id').agg(amount11=('amount11','sum'),trx_events=('trx_events','sum')).reset_index()
    customer=ids.merge(freq,on='client_id',validate='one_to_one').merge(value,on='client_id',validate='one_to_one')
    totals=customer.groupby('purchase_group').agg(customers=('client_id','size'),amount11=('amount11','sum'),events=('trx_events','sum')).reindex(['No purchase observed','One positive month','Multiple positive months'])
    totals['customer_share']=totals.customers/n;totals['value_share']=totals.amount11/totals.amount11.sum();totals['event_share']=totals.events/totals.events.sum()
    totals.to_csv(OUT/'exhaustive_purchase_frequency_financial_groups.csv');customer.to_parquet(OUT/'all_customer_financial_groups.parquet',index=False)
    concentration=[]
    for metric in ['amount11','trx_events']:
        ranked=customer.sort_values(metric,ascending=False)
        for pct in [1,5,10,20]:
            size=int(np.ceil(n*pct/100));concentration.append(dict(metric=metric,top_pct=pct,top_customers=size,rest_customers=n-size,top_share=ranked.iloc[:size][metric].sum()/ranked[metric].sum()))
    concentration=pd.DataFrame(concentration);concentration.to_csv(OUT/'full_history_concentration.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(15,5));x=np.arange(3)
    for i,col,title in [(0,'customer_share','Customer share'),(1,'value_share','Transaction-value share'),(2,'event_share','Event-count share')]:axes[0].bar(x+(i-1)*.23,totals[col],.23,label=title)
    axes[0].set_xticks(x,['No positive month','One positive month','Multiple positive months']);axes[0].yaxis.set_major_formatter(PercentFormatter(1));axes[0].legend(fontsize=8);axes[0].set_title('Exhaustive customer groups: financial contribution')
    for metric,label in [('amount11','Transaction value (11)'),('trx_events','Transaction count')]:
        a=concentration[concentration.metric.eq(metric)];axes[1].plot(a.top_pct,a.top_share,'o-',label=label)
    axes[1].set_ylim(0,1);axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_xlabel('Top % of customers (remainder retained in CSV)');axes[1].set_title('Concentration across full raw TRX history');axes[1].legend(fontsize=8)
    save(fig,'05_full_history_financial_contribution','ALL customers, ALL valid TRX months; amount only currency 11. Groups based on 12 label months, not revenue, ownership or lifetime purchases.')
    # First-source order uses ALL source history and ALL 12-month buyers; nonbuyers kept.
    first_buy=labels[labels.y.eq(1)].groupby('client_id').month.min().rename('first_buy')
    orders=[]
    for s in SOURCES:
        first_source=panel[panel[s+'_events'].gt(0)].groupby('client_id').month.min().rename('first_source')
        z=ids.merge(first_buy,on='client_id',how='left',validate='one_to_one').merge(first_source,on='client_id',how='left',validate='one_to_one')
        end=pd.Timestamp(next(b['last_month'] for b in bounds if b['source']==s))
        z['order']=np.select([z.first_buy.isna(),z.first_source.lt(z.first_buy),z.first_source.eq(z.first_buy),z.first_source.gt(z.first_buy),z.first_buy.ge(end)],['No purchase observed','Source observed before','First source in purchase month','First source after','No source; follow-up insufficient'],default='No source within available history')
        for state,g in z.groupby('order'):orders.append(dict(source=s,state=state,customers=len(g),share=len(g)/n))
    orders=pd.DataFrame(orders);orders.to_csv(OUT/'full_population_source_purchase_order.csv',index=False)
    order_states=['No purchase observed','Source observed before','First source in purchase month','First source after','No source within available history','No source; follow-up insufficient']
    a=orders.pivot(index='source',columns='state',values='share').reindex(index=SOURCES,columns=order_states).fillna(0)
    fig,ax=plt.subplots(figsize=(15,5));left=np.zeros(3)
    for state,color in zip(order_states,['#cbd5e1','#0d9488','#f59e0b','#6366f1','#64748b','#ec4899']):
        ax.barh(range(3),a[state],left=left,label=state,color=color);left+=a[state].to_numpy()
    ax.set_yticks(range(3),[s.upper() for s in SOURCES]);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_title(f'First observed source vs first observed purchase | all {n:,} customers');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,fontsize=8)
    save(fig,'06_full_population_source_order','All valid source history and all 12 label months. Same-month order unknown; no-source cases retain follow-up status. First observed is not first-ever.')
    checks.update({'all_clients_each_label_month':bool(d.groupby('month').size().eq(n).all()),'all_12_months_in_lag_inventory':lag.target_month.nunique()==len(months),'before_after_groups_exhaust_month_population':bool(lag.groupby(['target_month','lag','source']).total_customers.sum().eq(n).all()),'observed_plus_unavailable_reconciles':bool((lag.observed_customers+lag.unavailable_customers).eq(lag.total_customers).all()),'financial_groups_exhaust_clients':int(totals.customers.sum())==n,'order_groups_exhaust_clients_each_source':bool(orders.groupby('source').customers.sum().eq(n).all()),'raw_trx_events_reconcile':int(customer.trx_events.sum())==next(b['raw_rows'] for b in bounds if b['source']=='trx'),'label_months_at_least_12':len(months)>=12,'six_figures':len(list(OUT.glob('*.png')))==6})
    assert all(checks.values()),checks
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    manifest={'customers':n,'label_months':[pd.Timestamp(m).strftime('%Y-%m') for m in months],'label_rows':len(labels),'sampled_customers':0,'raw_sources':bounds,'timezone':'Europe/Moscow','raw_naive_assumption':'UTC','method':'retrospective monthly alignment, no prediction cutoff/model, no causal claim','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    report=['# EDA toàn bộ khách — 12 tháng nhãn và toàn bộ lịch sử log','',f'Khách: {n:,}; nhãn: {len(labels):,} dòng qua {len(months)} tháng. Không lấy mẫu. Mỗi tháng phân nhóm đủ toàn bộ khách.','',
    '## Phương pháp và cách đọc','Tháng t có hoạt động được nối nhãn t−1 và t+1. Bốn nhóm known: không mua cả hai tháng kề; chỉ mua tháng sau; chỉ mua tháng trước; mua cả hai. Nhãn kề thiếu giữ nhóm unknown. Tách thêm nhãn mua cùng tháng 0/1 để không trộn tương tác cùng tháng với trước/sau.',
    'Biểu đồ 03 lấy từng tháng có nhãn làm mốc, so tỷ lệ có log tháng trước và tháng sau giữa người mua và không mua. Tất cả 12 tháng nhãn đều có trong bảng; không có log toàn nguồn trong tháng tương lai là unavailable, không phải 0. Do dữ liệu chỉ có 12 tháng nhãn, hai tháng biên không có đủ hai nhãn kề. Không thể có 12 tháng đầy đủ cả hai nhãn kề nếu không có nhãn bổ sung ngoài phạm vi.',
    'Biểu đồ 04 gộp khách–tháng trong phạm vi nguồn, có bảng n và unavailable; khách lặp qua tháng nên không dùng kiểm định độc lập theo dòng. Khác biệt có thể do lịch, lịch sử mua và yếu tố khác; chưa chứng minh hoạt động là hệ quả hoặc chuẩn bị mua.',
    'Biểu đồ 05 tính đóng góp tài chính trên toàn bộ thời gian TRX; chia khách theo số tháng dương trong toàn bộ kỳ nhãn. Không coi amount là doanh thu hoặc tiền tệ đã giải mã. Không đồng nhất nhãn không mua trong một năm với chưa từng mua trong đời.',
    'Biểu đồ 06 so tháng đầu tiên có nguồn trong toàn bộ log với tháng đầu tiên có nhãn mua, giữ cả người không mua và trạng thái chưa đủ theo dõi. Không xác định thứ tự bên trong cùng tháng.',
    'Số 0 trong tháng nguồn có dữ liệu nghĩa không tìm thấy sự kiện, không xác nhận không hoạt động thực tế. Phạm vi tháng nguồn không bảo đảm quan sát liên tục từng khách. Tháng đầu nguồn có thể không trọn; xem source_inventory.csv. Timestamp raw giả định UTC rồi chuyển lịch Moscow. Dialog có nhiễu ngày theo audit dự án, không suy lịch vận hành thực tế.',
    '', '## Kết quả phân nhóm tài chính (CSV lưu số chính xác)',totals.to_string(),'', '## Kiểm tra',json.dumps(checks,indent=2),
    '', '## Trạng thái artifact cũ','Kết quả purchase_event_financial_eda_20261118 dùng mẫu tham chiếu và chỉ một số tháng mua là phương pháp cũ, không đáp ứng yêu cầu mới. Gói hiện tại thay thế về phạm vi phân tích.']
    report.extend(['','## Phát hiện về hai chiều thời gian',
    'Trong các khách–tháng không có mua cùng tháng: DIALOG có log ở 22,64% nhóm chỉ mua tháng sau (2.041/9.016), 24,86% nhóm chỉ mua tháng trước (2.517/10.125), và 6,00% nhóm không mua hai tháng kề (58.284/971.491). Đây là tỷ lệ gộp quan sát của các tháng đủ nhãn kề 03–12/2022, không phải trên toàn bộ 12 tháng đều đủ nhãn kề. Kết quả cho thấy DIALOG gắn với cả hai phía thời gian, chưa xác định một cuộc hội thoại cụ thể là chuẩn bị hay hậu quả.',
    'Ở khách–tháng có mua cùng tháng, tỷ lệ DIALOG của nhóm chỉ mua tháng sau và chỉ mua tháng trước lần lượt 37,56% và 37,36%. Cần giữ việc tách nhãn cùng tháng để tránh gọi mọi hội thoại là tín hiệu đi trước.',
    'Các con số chi tiết và n theo từng tháng được lưu trong before_after_monthly_rates.csv; không chọn một vài tháng đẹp để kết luận toàn kỳ. Hai nhóm năm nhãn gồm 89.813 khách không có tháng dương, 8.496 khách đúng một tháng dương, 1.915 khách nhiều tháng dương; tổng 100.224. Tỷ trọng amount toàn lịch sử TRX tương ứng khoảng 92,36%, 6,39%, 1,26%; khác với kết quả chỉ riêng năm 2022 trước đây vì phạm vi thời gian đã mở rộng.'])
    (OUT/'REPORT.md').write_text('\n\n'.join(report),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)
    print(totals.to_string(),flush=True)
    print(assoc[assoc.source.eq('dialog')].to_string(index=False),flush=True)


if __name__=='__main__':
    analyze(*extract())
