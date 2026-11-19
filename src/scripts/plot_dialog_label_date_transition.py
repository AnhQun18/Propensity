"""Full-population source events before/after the recorded label date (Moscow)."""
from pathlib import Path
import duckdb
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/full_population_temporal_eda_20261118'
LABEL = (ROOT / 'data/MBD-mini/targets/**/*.parquet').as_posix()

def plot(source):
    event_path = (ROOT / f'data/MBD-mini/detail/{source}/**/*.parquet').as_posix()
    c = duckdb.connect()
    c.execute("SET memory_limit='3GB'")
    c.execute('SET threads=2')
    c.execute(f"SET temp_directory='{(OUT / ('tmp_' + source + '_transition')).as_posix()}'")
    bounds = c.execute(f"SELECT min(event_time),max(event_time) FROM read_parquet('{event_path}')").fetchone()
    q = c.execute(f"""
        WITH labels AS (
          SELECT client_id, CAST(mon AS DATE) mon,
            greatest(target_1,target_2,target_3,target_4) purchase,
            CAST(mon AS TIMESTAMP) - INTERVAL '3 hours' cutoff_utc
          FROM read_parquet('{LABEL}')
          WHERE greatest(target_1,target_2,target_3,target_4)=1
        ), joined AS (
          SELECT l.client_id,l.mon,l.purchase,l.cutoff_utc,
            count(d.event_time) FILTER (WHERE d.event_time < l.cutoff_utc) n_before,
            count(d.event_time) FILTER (WHERE d.event_time >= l.cutoff_utc) n_after
          FROM labels l LEFT JOIN read_parquet('{event_path}') d
            ON l.client_id=d.client_id
           AND d.event_time >= l.cutoff_utc - INTERVAL '30 days'
           AND d.event_time < l.cutoff_utc + INTERVAL '30 days'
          GROUP BY l.client_id,l.mon,l.purchase,l.cutoff_utc
        )
        SELECT mon,purchase,
          CASE WHEN cutoff_utc - INTERVAL '30 days' < ? OR cutoff_utc + INTERVAL '30 days' > ?
                 THEN 'Chưa đủ quan sát'
               WHEN n_before>0 AND n_after>0 THEN 'Cả trước và sau'
               WHEN n_before>0 THEN 'Chỉ trước'
               WHEN n_after>0 THEN 'Chỉ sau'
               ELSE 'Không có hai phía' END state,
          count(*) customers
        FROM joined GROUP BY 1,2,3 ORDER BY 1,2,3
    """, bounds).df()
    q.to_csv(OUT / f'{source}_label_date_transition_30d.csv', index=False)
    totals = q.groupby(['mon','purchase']).customers.sum().unstack()
    assert len(totals)==12 and totals.sum(axis=1).gt(0).all()
    positive = q[q.purchase.eq(1)].pivot(index='mon',columns='state',values='customers').fillna(0)
    states = ['Không có hai phía','Chỉ trước','Chỉ sau','Cả trước và sau','Chưa đủ quan sát']
    colors = ['#cbd5e1','#2875b9','#db6b28','#7952b3','#94a3b8']
    positive = positive.reindex(columns=states,fill_value=0)
    share = positive.div(positive.sum(axis=1),axis=0)
    fig,ax=plt.subplots(figsize=(13,6))
    bottom = 0
    for state,color in zip(states,colors):
        ax.bar(range(len(share)),share[state],bottom=bottom,label=state,color=color)
        bottom += share[state]
    ax.set_xticks(range(len(share)),[pd.Timestamp(x).strftime('%m/%Y') for x in share.index],rotation=35)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('Tỷ trọng khách–tháng có mua')
    ax.set_title(f'{source.upper()} quanh ngày ghi nhận nhãn | toàn bộ khách–tháng mua',weight='bold')
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False)
    for i,n in enumerate(positive.sum(axis=1)):
        ax.text(i,1.01,f'n={int(n):,}',rotation=90,ha='center',va='bottom',fontsize=7)
    ax.set_ylim(0,1.2)
    fig.text(.01,.01,'Ranh giới: 00:00 Moscow ngày mon; cửa sổ ±30 ngày. Tháng cuối thiếu log sau mốc được giữ riêng. '
             'Ngày mon không nhất thiết là ngày giao dịch thực tế; raw timestamp giả định UTC.',fontsize=8)
    fig.subplots_adjust(left=.08,right=.76,bottom=.2,top=.85)
    fig.savefig(OUT/f'{source}_label_date_transition_30d.png',dpi=170,bbox_inches='tight')
    fig.savefig(OUT/f'{source}_label_date_transition_30d.svg',bbox_inches='tight')
    print(f'{source}_label_date_transition_30d.png')

if __name__=='__main__':
    for source in ('dialog','geo'):
        plot(source)
