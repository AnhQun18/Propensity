"""Raw interevent gaps, Moscow calendars and dialog threshold-session sensitivity."""
import json
import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from complete_eda_figures_moscow import ROOT, OUT, BASE, publish, INDEX


def main():
    c=duckdb.connect();c.execute("SET memory_limit='3GB'");c.execute('SET threads=2')
    c.execute(f"SET temp_directory='{OUT.as_posix()}/tmp_event_timing'")
    ids=np.sort(pd.read_parquet(BASE/'labels_any.parquet').client_id.unique())
    summaries=[];calendars=[]
    for source in ['trx','geo','dialog']:
        print('Timing raw',source,flush=True)
        path=(ROOT/f'data/MBD-mini/detail/{source}/**/*.parquet').as_posix()
        dest=OUT/f'{source}_raw_interarrival.parquet'
        if not dest.exists():
            c.execute(f"""COPY (WITH gaps AS (
              SELECT client_id,epoch(event_time-lag(event_time) OVER(PARTITION BY client_id ORDER BY event_time))/3600.0 gap_hours
              FROM read_parquet('{path}'))
              SELECT client_id,count(*) events,count(gap_hours) gaps,avg(gap_hours) mean_gap_hours,
                median(gap_hours) median_gap_hours,stddev_pop(gap_hours) sd_gap_hours,max(gap_hours) max_gap_hours,
                count(*) FILTER(WHERE gap_hours=0) zero_gaps,
                CASE WHEN count(gap_hours)>=2 AND avg(gap_hours)>0 THEN stddev_pop(gap_hours)/avg(gap_hours) END cv
              FROM gaps GROUP BY client_id) TO '{dest.as_posix()}' (FORMAT PARQUET)""")
        q=pd.read_parquet(dest)
        assert (q.gaps==q.events-1).all()
        assert int(q.events.sum())=={'trx':38764930,'geo':66295724,'dialog':217205}[source]
        for col in ['mean_gap_hours','median_gap_hours','max_gap_hours','cv']:
            v=q[col].dropna()
            summaries.append(dict(source=source,metric=col,defined_customers=len(v),undefined_customers=len(ids)-len(v),
                median=float(v.median()),p90=float(v.quantile(.9)),p99=float(v.quantile(.99)),
                zero_gaps=int(q.zero_gaps.sum()),events=int(q.events.sum()),pairs=int(q.gaps.sum())))
        cal=c.execute(f"""SELECT cast(extract('isodow' FROM event_time+INTERVAL '3 hours') AS INTEGER) AS weekday,
            cast(extract('hour' FROM event_time+INTERVAL '3 hours') AS INTEGER) AS hour,count(*) events
            FROM read_parquet('{path}') GROUP BY 1,2""").df()
        cal['source']=source;calendars.append(cal)
    summary=pd.DataFrame(summaries);calendar=pd.concat(calendars,ignore_index=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,source in zip(axes,['trx','geo','dialog']):
        q=pd.read_parquet(OUT/f'{source}_raw_interarrival.parquet')
        v=q.cv.dropna();ax.hist(v,bins=60);ax.set_title(source);ax.set_xlabel('CV gap, trọng số bằng nhau mỗi khách');ax.set_ylabel('Số khách')
    publish('raw_interarrival_cv_all_sources',fig,summary,'Khoảng cách giữa mọi sự kiện liên tiếp và độ đều đặn',
        'Sắp timestamp trong cùng khách, nối qua ranh giới tháng. Gap giờ theo thời điểm UTC, không thay đổi khi chuyển Moscow. '
        'CV=SD tổng thể/mean, cần ≥2 gap và mean>0. Bảng cá thể của từng nguồn lưu Parquet; khách không nguồn/không đủ gap tính undefined.',
        'Đo độ biến động khoảng cách sự kiện, không đồng nhất với nhịp theo lịch hoặc phiên thực.',
        '; '.join(f'{r.source}: median CV={r.median:.3g}, n={r.defined_customers:,}, undefined={r.undefined_customers:,}' for r in summary[summary.metric=='cv'].itertuples()),
        'Thời gian nguồn bị cắt ở hai biên; khoảng nghỉ đầu/cuối chưa thấy không nằm trong gaps. Timestamp trùng có gap0, không tự đặt thứ tự trong timestamp. '
        'Không gọi CV cao là bot hoặc dữ liệu lỗi.')
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for ax,source in zip(axes,['trx','geo','dialog']):
        q=calendar[calendar.source==source].pivot(index='weekday',columns='hour',values='events').reindex(index=range(1,8),columns=range(24),fill_value=0)
        z=q.to_numpy()/q.to_numpy().sum()*100
        im=ax.imshow(z,aspect='auto',cmap='Blues');ax.set_title(source);ax.set_xlabel('Giờ Moscow');ax.set_yticks(range(7),['T2','T3','T4','T5','T6','T7','CN']);fig.colorbar(im,ax=ax,label='% sự kiện nguồn')
    publish('moscow_hour_weekday_all_sources',fig,calendar,'Giờ × thứ của ba nguồn theo Moscow',
        'Quét toàn bộ raw, cộng 3 giờ một lần từ giả định UTC để lấy lịch Moscow. Mỗi ô là % tổng sự kiện của nguồn. '
        'Đây là tỷ trọng sự kiện, không chuẩn hóa số lần xuất hiện của từng thứ trong khoảng nguồn.',
        'So mẫu lịch công bố giữa các nguồn, với thang màu riêng và tỷ trọng trong nguồn.',
        'Tổng sự kiện của mỗi heatmap khớp raw; ba nguồn có thời gian bắt đầu khác nhau nên không so trực tiếp số lượng tuyệt đối.',
        'Ngày có nhiễu ẩn danh và timezone ETL chưa xác minh; không suy lịch vận hành hoặc nghỉ lễ. '
        'Khách nhiều sự kiện có trọng số lớn; cần đọc cùng hình cân bằng khách.')
    raw=(ROOT/'data/MBD-mini/detail/dialog/**/*.parquet').as_posix()
    session=[]
    for threshold in [30,60]:
        q=c.execute(f"""WITH gaps AS (SELECT client_id,event_time,
          epoch(event_time-lag(event_time) OVER(PARTITION BY client_id ORDER BY event_time))/60.0 gap_min
          FROM read_parquet('{raw}')), grouped AS (
          SELECT *,sum(CASE WHEN gap_min IS NULL OR gap_min>{threshold} THEN 1 ELSE 0 END)
          OVER(PARTITION BY client_id ORDER BY event_time) sid FROM gaps),
          sessions AS (SELECT client_id,sid,count(*) AS rows,epoch(max(event_time)-min(event_time))/60.0 span_min
          FROM grouped GROUP BY 1,2)
          SELECT client_id,count(*) AS sessions,sum(rows) AS rows,avg(rows) mean_rows,median(span_min) median_span_min,
          max(span_min) max_span_min FROM sessions GROUP BY 1""").df()
        assert q.rows.sum()==217205
        q['threshold_minutes']=threshold;session.append(q)
    session=pd.concat(session,ignore_index=True)
    full=pd.MultiIndex.from_product([ids,[30,60]],names=['client_id','threshold_minutes']).to_frame(index=False)
    full=full.merge(session,on=['client_id','threshold_minutes'],how='left',validate='one_to_one')
    full['sessions']=full.sessions.fillna(0);full.to_parquet(OUT/'dialog_threshold_session_customers.parquet',index=False)
    s=full.groupby('threshold_minutes').agg(customers=('client_id','size'),sessions=('sessions','sum'),median_sessions=('sessions','median'),defined_span_customers=('median_span_min','count'),median_span_min=('median_span_min','median')).reset_index()
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for threshold,q in full.groupby('threshold_minutes'):
        axes[0].hist(q.sessions,bins=np.arange(0,31),histtype='step',label=f'{threshold} phút')
        axes[1].hist(q.median_span_min.dropna(),bins=40,histtype='step',label=f'{threshold} phút')
    axes[0].set_xlabel('Số cụm sự kiện / toàn bộ khách');axes[1].set_xlabel('Trung vị độ trải timestamp cụm, phút')
    for ax in axes:ax.legend();ax.set_ylabel('Số khách')
    publish('dialog_threshold_session_sensitivity',fig,s,'Dialog: độ nhạy gom cụm sự kiện 30 / 60 phút',
        'Bắt đầu cụm mới nếu gap>30 hoặc >60 phút; cụm đầu luôn mới. Cụm nối các sự kiện trong cùng khách; '
        'span=max timestamp−min timestamp. Tất cả khách có trong bảng, không dialog có count0 nhưng span undefined. '
        'Histogram count phóng 0–30, khách vượt 30 vẫn giữ trong CSV/Parquet và tổng hợp.',
        'Kiểm tra việc chọn ngưỡng có đổi mạnh số cụm hay không; không xác nhận phiên hoặc thời lượng hội thoại thực.',
        '; '.join(f'{r.threshold_minutes} phút: {int(r.sessions):,} cụm, median số cụm/toàn khách={r.median_sessions:.1f}' for r in s.itertuples()),
        'Một dòng có span0 không nghĩa cuộc gọi 0 phút. Ngày nhiễu có thể phá khoảng cách thực giữa các lượt. '
        'Không có conversation/session ID để kiểm chứng; chỉ là cụm theo quy ước.')
    c.close()
    old=pd.read_csv(OUT/'figure_index.csv');idx=pd.concat([old,pd.DataFrame(INDEX)]).drop_duplicates('figure',keep='last');idx.to_csv(OUT/'figure_index.csv',index=False)
    status=json.loads((OUT/'execution_status.json').read_text(encoding='utf-8'));status['done']+=['raw_interevent_gaps_cv_all_sources','moscow_hour_weekday_all_sources','dialog_threshold_session_sensitivity'];status['figures']=len(idx)
    (OUT/'execution_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
