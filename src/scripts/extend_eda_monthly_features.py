"""Full population categorical profiles and fixed pre-window TRX strata."""
import json
import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from complete_eda_figures_moscow import ROOT, OUT, BASE, load, publish, INDEX, event_studies


def main():
    ids,months,arrays,y=load();c=duckdb.connect();c.execute("SET memory_limit='3GB'");c.execute('SET threads=2')
    c.execute(f"SET temp_directory='{OUT.as_posix()}/tmp_monthly_features'")
    inventory=pd.read_csv(BASE/'source_inventory.csv').set_index('source')
    for source,r in inventory.iterrows():
        if r.first_month_partial:
            arrays[source][:,months.get_loc(pd.Timestamp(r.first_month))]=np.nan
    feature_arrays={};summaries=[]
    for source,levels in [('geo',['geohash_4','geohash_5','geohash_6']),('trx',['event_type'])]:
        raw=(ROOT/f'data/MBD-mini/detail/{source}/**/*.parquet').as_posix()
        for col in levels:
            print('Monthly categories',col,flush=True)
            cache=OUT/f'{col}_monthly_customer_cells.parquet'
            if not cache.exists():
                c.execute(f"COPY (SELECT client_id,date_trunc('month',event_time+INTERVAL '3 hours') AS month,{col} AS cell,count(*) AS events FROM read_parquet('{raw}') GROUP BY 1,2,3) TO '{cache.as_posix()}' (FORMAT PARQUET)")
            path=cache.as_posix()
            profile=c.execute(f"""WITH cells AS (
              SELECT *,sum(events) OVER(PARTITION BY client_id,month) total,
                min(month) OVER(PARTITION BY client_id,cell) first_cell_month,
                row_number() OVER(PARTITION BY client_id,month ORDER BY events DESC,cell) rank
              FROM read_parquet('{path}'))
              SELECT client_id,month,count(*) unique_cells,
                max(events)::DOUBLE/max(total) top_share,
                -sum((events::DOUBLE/total)*ln(events::DOUBLE/total)) entropy,
                sum(pow(events::DOUBLE/total,2)) hhi,
                count(*) FILTER(WHERE first_cell_month=month) new_cells,
                sum(CASE WHEN first_cell_month=month THEN events ELSE 0 END)::DOUBLE/max(total) new_cell_event_share,
                max(CASE WHEN rank=1 THEN cell END) main_cell,
                max(total) events
              FROM cells GROUP BY 1,2""").df()
            profile=profile.sort_values(['client_id','month'])
            profile['previous_main']=profile.groupby('client_id').main_cell.shift()
            profile['previous_month']=profile.groupby('client_id').month.shift()
            consecutive=profile.month.eq(profile.previous_month+pd.offsets.MonthBegin(1))
            profile['changed_main']=np.where(consecutive,profile.main_cell.ne(profile.previous_main).astype(float),np.nan)
            profile.to_parquet(OUT/f'{col}_monthly_profiles.parquet',index=False)
            first=pd.Timestamp(inventory.loc[source,'first_month']);last=pd.Timestamp(inventory.loc[source,'last_month'])
            valid=(months>=first)&(months<=last)
            if inventory.loc[source,'first_month_partial']:valid &= months!=first
            for metric in ['unique_cells','top_share','entropy','hhi','new_cells','new_cell_event_share','changed_main']:
                a=profile.pivot(index='client_id',columns='month',values=metric).reindex(index=ids,columns=months).to_numpy(dtype=float,copy=True)
                # Zero counts are defined without log; concentration/entropy/change remain unknown.
                if metric in ['unique_cells','new_cells']:a=np.nan_to_num(a)
                a[:,~valid]=np.nan
                summaries.append(dict(source=source,column=col,metric=metric,defined_customer_months=int(np.isfinite(a).sum()),
                    undefined_customer_months=int((~np.isfinite(a)).sum()),median=float(np.nanmedian(a))))
                if col in ['geohash_4','event_type'] and metric in ['unique_cells','new_cells']:
                    feature_arrays[f'{col}_{metric}']=a
            # Compact monthly source-level summary, retaining undefined customer counts.
            fig,axes=plt.subplots(2,2,figsize=(13,7))
            for ax,metric in zip(axes.flat,['top_share','entropy','new_cell_event_share','changed_main']):
                q=profile.groupby('month')[metric].agg(['mean','median','count'])
                ax.plot(q.index,q['mean'],'o-',label='Mean khách có chỉ tiêu');ax.plot(q.index,q['median'],':',label='Median')
                ax.set_title(metric);ax.legend(fontsize=7);ax.tick_params(axis='x',rotation=30)
            publish(f'{col}_monthly_concentration_novelty',fig,profile.groupby('month').agg(customers=('client_id','size'),mean_entropy=('entropy','mean'),mean_topshare=('top_share','mean'),mean_new_event_share=('new_cell_event_share','mean'),changed_main_defined=('changed_main','count'),changed_main_share=('changed_main','mean')).reset_index(),
                f'{col}: tập trung, độ mới và đổi mã chính theo tháng',
                'Tất cả sự kiện; nhóm khách–tháng–mã. Mã mới là chưa thấy ở bất kỳ tháng trước trong toàn lịch sử được cung cấp; '
                'tỷ lệ mới là số sự kiện tại mã lần đầu thấy tháng này / tổng sự kiện tháng. Mã chính chọn count lớn nhất, tie mã nhỏ nhất. '
                'Đổi mã chỉ tính khi hai tháng liền nhau đều có log. Bảng cá thể đầy đủ lưu Parquet.',
                'Đo diễn tiến cấu trúc danh mục và mức mới trong phạm vi quan sát; concentration chỉ xác định khi có log.',
                f'Kiểm kê {profile.client_id.nunique():,} khách có nguồn, {len(profile):,} khách–tháng có log. '
                f'Entropy trung vị {profile.entropy.median():.4g}; top-share trung vị {profile.top_share.median():.4g}.',
                'Các đường mô tả khách có log, số undefined đọc bảng tổng hợp; không đại diện mọi khách. '
                'Tháng đầu nguồn được giữ trong hình kiểm kê, là tháng không trọn với GEO. Đổi mã không chứng minh di chuyển hoặc đổi nghiệp vụ.')
    c.close()
    pd.DataFrame(summaries).to_csv(OUT/'monthly_categorical_feature_coverage.csv',index=False)
    event_studies(ids,months,feature_arrays,y)
    # Fixed strata before beginning the -3..+3 observation region.
    rows=[];anchors=np.where(np.isfinite(y).any(axis=0))[0]
    for t in anchors:
        if t<3:continue
        hist=arrays['trx'][:,:t-3]
        known_months=np.isfinite(hist).any(axis=0)
        if not known_months.any():continue
        counts=np.nansum(hist,axis=1)
        pos=counts>0
        edges=np.unique(np.quantile(counts[pos],[.2,.4,.6,.8])) if pos.any() else np.array([])
        strata=np.where(pos,np.searchsorted(edges,counts,side='left')+1,0)
        for source in ['trx','geo','dialog']:
            a=arrays[source]
            for lag in range(-3,4):
                j=t+lag;v=a[:,j] if 0<=j<a.shape[1] else np.full(len(ids),np.nan)
                for b in [0,1]:
                    for s in range(int(strata.max())+1):
                        mask=(y[:,t]==b)&(strata==s);good=mask&np.isfinite(v)
                        rows.append(dict(source=source,anchor=str(months[t].date()),lag=lag,purchase=b,stratum=s,
                            population=int(mask.sum()),observed=int(good.sum()),unavailable=int((mask&~np.isfinite(v)).sum()),
                            mean=float(v[good].mean()) if good.any() else np.nan))
    d=pd.DataFrame(rows)
    assert (d.observed+d.unavailable==d.population).all()
    assert d.groupby(['source','anchor','lag']).population.sum().eq(len(ids)).all()
    fig,axes=plt.subplots(3,6,figsize=(19,10),sharex=True)
    for row,source in enumerate(['trx','geo','dialog']):
        for s in range(6):
            ax=axes[row,s]
            for b,color in [(0,'#2463a6'),(1,'#d4772b')]:
                q=d[(d.source==source)&(d.stratum==s)&(d.purchase==b)].groupby('lag')['mean'].mean()
                ax.plot(q.index,q,'o-',ms=3,color=color,label=f'Mua={b}')
            ax.set_title(f'{source}, T{s}',fontsize=9);ax.axvline(0,color='gray',ls=':');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    publish('fixed_past_trx_strata_event_study',fig,d,'Trước/sau tháng mua trong tầng TRX xác định từ quá khứ',
        'T0 không thấy TRX; T1–T5 quintile count TRX dương trên toàn lịch sử trước đầu tháng −3; ties không chia. '
        'Tầng giữ cố định suốt −3..+3 của một tháng neo. Mỗi đường cân bằng trọng số tháng neo; CSV giữ đủ n/unknown theo kỳ. '
        'Các tháng neo không có bất kỳ tháng lịch sử nguồn trước −3 chưa đủ điều kiện và không tạo tầng giả.',
        'Kiểm tra khác biệt mua/không mua còn cùng chiều trong các mức hoạt động quá khứ hay không.',
        'Tổng các tầng và hai nhóm nhãn bằng 100.224 khách ở từng nguồn–tháng neo–lag đã tạo; '
        'không cộng các tầng giữa các tháng neo vì khách có thể chuyển tầng.',
        'Phân tầng count chưa kiểm soát độ dài lịch sử hoặc toàn bộ nhiễu. Chưa có CI từng đường. '
        'Khoảng không có log không chứng minh quan sát liên tục.')
    old=pd.read_csv(OUT/'figure_index.csv');idx=pd.concat([old,pd.DataFrame(INDEX)]).drop_duplicates('figure',keep='last');idx.to_csv(OUT/'figure_index.csv',index=False)
    status=json.loads((OUT/'execution_status.json').read_text(encoding='utf-8'));status['done']+=['geo_all_level_monthly_entropy_hhi_novelty_main_change','trx_monthly_type_entropy_novelty','fixed_pre_window_trx_strata_event_study'];status['figures']=len(idx)
    (OUT/'execution_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
