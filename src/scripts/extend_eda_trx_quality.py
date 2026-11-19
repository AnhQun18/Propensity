"""Raw TRX quality, category structure and exhaustive customer concentration."""
import json
import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from complete_eda_figures_moscow import ROOT, OUT, BASE, publish, INDEX


def main():
    c=duckdb.connect();c.execute("SET memory_limit='3GB'");c.execute('SET threads=2')
    c.execute(f"SET temp_directory='{OUT.as_posix()}/tmp_trx'")
    path=(ROOT/'data/MBD-mini/detail/trx/**/*.parquet').as_posix()
    c.execute(f"CREATE VIEW trx AS SELECT * FROM read_parquet('{path}',hive_partitioning=true)")
    quant=c.execute("""SELECT fold,count(*) events,count(DISTINCT client_id) customers,
        count(*) FILTER(WHERE event_time=date_trunc('day',event_time)) utc_midnight,
        count(*) FILTER(WHERE event_time+INTERVAL '3 hours'=date_trunc('day',event_time+INTERVAL '3 hours')) moscow_midnight,
        count(*) FILTER(WHERE event_time=date_trunc('second',event_time)) whole_second,
        count(*) FILTER(WHERE event_time=date_trunc('minute',event_time)) whole_minute
        FROM trx GROUP BY fold ORDER BY fold""").df()
    assert quant.events.sum()==38764930
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    quant.plot.bar(x='fold',y=['events','customers'],ax=axes[0],logy=True)
    (quant.set_index('fold')[['moscow_midnight','whole_second','whole_minute']].div(quant.set_index('fold').events,axis=0)*100).plot.bar(ax=axes[1])
    axes[0].set_ylabel('Số lượng, log');axes[1].set_ylabel('% sự kiện');axes[1].legend(fontsize=8)
    publish('trx_fold_time_resolution',fig,quant,'TRX: khối lượng fold và độ phân giải timestamp',
        'Quét mọi dòng TRX; kiểm tra timestamp bằng phần nguyên ngày/phút/giây, riêng nửa đêm tính cả UTC và Moscow. '
        'Dấu bằng là chính xác trên timestamp lưu, không làm tròn.',
        'Kiểm tra mất cân bằng fold và dấu hiệu thời gian bị lượng tử hóa.',
        f'Tổng {int(quant.events.sum()):,} sự kiện; đúng nửa đêm Moscow {int(quant.moscow_midnight.sum()):,}, '
        f'đúng giây {int(quant.whole_second.sum()):,}, đúng phút {int(quant.whole_minute.sum()):,}.',
        'Timestamp tròn không chứng minh sự kiện chỉ có ngày; khối lượng fold khác có thể do khách nặng. Chưa xác nhận thời gian ETL gốc.')
    cross=c.execute('SELECT event_type,event_subtype,currency,count(*) events FROM trx GROUP BY 1,2,3').df()
    cross.to_csv(OUT/'trx_type_subtype_currency_all.csv',index=False)
    mapping=c.execute('SELECT event_subtype,count(DISTINCT event_type) parents FROM trx GROUP BY 1 ORDER BY 1').df()
    pairs=[]
    for a,b in [('event_subtype','event_type'),('src_type12','src_type11'),('dst_type12','dst_type11'),('src_type22','src_type21'),('src_type32','src_type31')]:
        r=c.execute(f'SELECT count(*) codes,sum(CASE WHEN parents>1 THEN 1 ELSE 0 END) conflicts FROM (SELECT {a},count(DISTINCT {b}) parents FROM trx WHERE {a} IS NOT NULL AND {b} IS NOT NULL GROUP BY 1)').fetchone()
        pairs.append(dict(child=a,parent=b,codes=int(r[0]),multiple_parent_codes=int(r[1] or 0)))
    pairs=pd.DataFrame(pairs)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    pivot=cross.groupby(['event_type','event_subtype']).events.sum().unstack(fill_value=0)
    im=axes[0].imshow(np.log1p(pivot.to_numpy()),aspect='auto');fig.colorbar(im,ax=axes[0],label='log1p số sự kiện')
    axes[0].set_xlabel('event_subtype');axes[0].set_ylabel('event_type')
    xt=np.arange(0,len(pivot.columns),max(1,len(pivot.columns)//20))
    yt=np.arange(0,len(pivot.index),max(1,len(pivot.index)//18))
    axes[0].set_xticks(xt,pivot.columns[xt],rotation=90,fontsize=7)
    axes[0].set_yticks(yt,pivot.index[yt],fontsize=7)
    axes[1].barh(pairs.child+' → '+pairs.parent,pairs.multiple_parent_codes);axes[1].set_xlabel('Số mã con có nhiều mã cha')
    publish('trx_category_parent_structure',fig,pairs,'TRX: bảng chéo loại giao dịch và kiểm tra ánh xạ mã',
        'Mỗi subtype/x2 được đối chiếu số type/x1 khác nhau trên toàn nguồn; bỏ null chỉ trong phép kiểm tra ánh xạ, '
        'không xóa dòng nguồn. CSV ba chiều đầy đủ nằm tại trx_type_subtype_currency_all.csv.',
        'Xác định quan hệ ánh xạ một-cha thực nghiệm, không khẳng định phân cấp nghiệp vụ.',
        '; '.join(f'{r.child}→{r.parent}: {r.multiple_parent_codes}/{r.codes} mã nhiều cha' for r in pairs.itertuples()),
        'Một cha không khôi phục được ý nghĩa hoặc chứng minh cấu trúc hash. Null phải đọc cùng audit riêng; '
        'các mã trên trục là category, không có khoảng cách số học.')
    nullcols=['src_type11','src_type12','dst_type11','dst_type12','src_type21','src_type22','src_type31','src_type32','currency']
    expr=','.join(f'count(*) FILTER(WHERE {col} IS NULL)::DOUBLE/count(*) AS {col}' for col in nullcols)
    null=c.execute(f"SELECT date_trunc('month',event_time+INTERVAL '3 hours') AS month,count(*) events,{expr} FROM trx GROUP BY 1 ORDER BY 1").df()
    fig,ax=plt.subplots(figsize=(13,5))
    for col in nullcols: ax.plot(null.month,null[col]*100,label=col)
    ax.set_ylabel('% sự kiện null');ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2)
    publish('trx_null_drift_moscow',fig,null,'TRX: tỷ lệ null từng cột theo tháng Moscow',
        'Mẫu số là toàn bộ sự kiện từng tháng Moscow; null SQL IS NULL, không coi mã zero là null. Không loại khách/dòng có null.',
        'Kiểm tra biến động missing và các cặp null có đi cùng nhau theo thời gian.',
        f'Kiểm kê {len(null)} tháng nguồn, tổng {int(null.events.sum()):,} dòng. '
        'Các đường trùng nhau biểu thị tỷ lệ bằng nhau, chưa tự chứng minh cùng một tập dòng.',
        'Biến động null chưa xác nhận đổi schema hoặc lỗi. Đọc cùng audit tổ hợp null toàn nguồn; chưa kiểm định change point.')
    customer=c.execute("SELECT client_id,fold,count(*) events,sum(amount) FILTER(WHERE currency=11) amount11 FROM trx GROUP BY 1,2").df()
    ids=c.execute(f"SELECT client_id,fold FROM read_parquet('{ROOT.as_posix()}/data/MBD-mini/client_split/**/*.parquet',hive_partitioning=true)").df()
    p=ids.merge(customer,on=['client_id','fold'],how='left',validate='one_to_one')
    p['events']=p.events.fillna(0);p['amount11']=p.amount11.fillna(0)
    assert len(p)==100224 and p.events.sum()==38764930
    p.to_parquet(OUT/'trx_all_customer_concentration.parquet',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5));summary=[]
    for metric in ['events','amount11']:
        v=np.sort(p[metric].to_numpy(float));cum=np.r_[0,np.cumsum(v)]/v.sum();pop=np.linspace(0,1,len(v)+1)
        axes[0].plot(pop,cum,label=metric)
        positive=v[v>0];axes[1].plot(positive,1-np.arange(len(positive))/len(positive),label=metric)
        gini=float(1-2*np.trapezoid(cum,pop))
        for frac in [.001,.01,.1]:
            k=int(np.ceil(len(v)*frac));summary.append(dict(metric=metric,top_fraction=frac,customers=k,share=float(v[-k:].sum()/v.sum()),gini=gini,zero_customers=int((v==0).sum())))
    axes[0].plot([0,1],[0,1],':',color='gray');axes[0].set_xlabel('Tỷ lệ khách tích lũy');axes[0].set_ylabel('Tỷ lệ khối lượng tích lũy');axes[0].legend()
    axes[1].set_xscale('log');axes[1].set_yscale('log');axes[1].set_xlabel('Khối lượng dương');axes[1].set_ylabel('CCDF trong nhóm dương');axes[1].legend()
    q=pd.DataFrame(summary)
    for metric,g in q.groupby('metric'):axes[2].plot(g.top_fraction*100,g.share*100,'o-',label=metric)
    axes[2].set_xscale('log');axes[2].set_xlabel('Top % khách');axes[2].set_ylabel('% khối lượng');axes[2].legend()
    publish('trx_customer_lorenz_ccdf',fig,q,'TRX: Lorenz, CCDF và đóng góp khách lớn',
        'Lorenz/Gini/top-share dùng toàn bộ 100.224 khách, gồm zero. CCDF chỉ nhóm dương và ghi riêng zero. '
        'Top x% chọn ceil(N*x), xếp hạng riêng theo từng chỉ tiêu. Amount chỉ currency11, đơn vị chưa xác minh.',
        'Đo mức tập trung khối lượng và độ nhạy tiềm tàng của thống kê trung bình với khách lớn.',
        '; '.join(f'{r.metric}: top {r.top_fraction*100:g}% đóng góp {r.share*100:.2f}%, Gini={r.gini:.4f}' for r in q.itertuples()),
        'Top theo amount khác top theo count. Không gọi amount là doanh thu hoặc dùng xếp hạng toàn kỳ để tạo feature quá khứ. '
        'Không loại khách lớn chỉ vì họ chi phối tổng.')
    # Missing TRX across folds, keeping all labels and full-source flags.
    labels=pd.read_parquet(BASE/'labels_any.parquet').groupby('client_id').agg(positive_months=('y','sum'),label_months=('y','size')).reset_index()
    p=p.merge(labels,on='client_id',validate='one_to_one')
    p['has_trx']=p.events>0
    group=p.groupby(['fold','has_trx']).agg(customers=('client_id','size'),positive_months=('positive_months','sum'),label_months=('label_months','sum')).reset_index()
    group['purchase_rate']=group.positive_months/group.label_months
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    group.pivot(index='fold',columns='has_trx',values='customers').plot.bar(ax=axes[0],logy=True)
    (group.pivot(index='fold',columns='has_trx',values='purchase_rate')*100).plot.bar(ax=axes[1])
    axes[0].set_ylabel('Khách, log');axes[1].set_ylabel('% khách–tháng mua')
    publish('trx_missing_population_fold_purchase',fig,group,'Khách không có TRX: phân bố fold và nhãn mua',
        'Không có TRX được xác định trên toàn lịch nguồn, chỉ dùng mô tả hồi cứu. Nhãn là toàn bộ 12 tháng mỗi khách; '
        'tỷ lệ mua=số tháng dương/tổng khách–tháng trong nhóm. Mọi khách có đúng một fold.',
        'Đối chiếu nhóm không có log với phần còn lại mà không xóa họ khỏi tổng thể.',
        f'{int((~p.has_trx).sum()):,}/{len(p):,} khách không có TRX; tổng nhãn đối soát {int(group.label_months.sum()):,} khách–tháng.',
        'Không có log toàn kỳ khác không có log trước cutoff. Tỷ lệ mô tả chưa có bootstrap và chưa điều chỉnh nguồn khác. '
        'Không khẳng định mất khách hoặc lỗi ETL.')
    c.close()
    old=pd.read_csv(OUT/'figure_index.csv');idx=pd.concat([old,pd.DataFrame(INDEX)]).drop_duplicates('figure',keep='last');idx.to_csv(OUT/'figure_index.csv',index=False)
    status=json.loads((OUT/'execution_status.json').read_text(encoding='utf-8'))
    status['done']+=['trx_raw_fold_time_resolution','trx_category_parent_cross_tables','trx_moscow_null_drift','trx_all_customer_gini_lorenz_topshare','trx_missing_fold_purchase']
    status['figures']=len(idx)
    (OUT/'execution_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
