"""Monthly raw geo coverage and first-observed code diagnostics (UTC convention)."""
from pathlib import Path
import duckdb
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

out = Path('outputs/geo_temporal_coverage')
out.mkdir(parents=True, exist_ok=True)
c = duckdb.connect()
c.execute('SET threads=4')
c.execute("SET memory_limit='4GB'")
c.execute("CREATE VIEW geo AS SELECT *,date_trunc('month',event_time)::DATE AS month FROM read_parquet('data/MBD-mini/detail/geo/**/*.parquet')")
denom = c.execute("SELECT count(distinct client_id) FROM read_parquet('data/MBD-mini/client_split/**/*.parquet')").fetchone()[0]
coverage = c.execute('''SELECT month,count(*) events,count(distinct client_id) active_clients,
    count(distinct cast(event_time AS DATE)) observed_days,min(event_time) first_event,max(event_time) last_event
    FROM geo GROUP BY month ORDER BY month''').df()
months = pd.date_range(coverage.month.min(),coverage.month.max(),freq='MS')
coverage = coverage.set_index('month').reindex(months).rename_axis('month').reset_index()
coverage['reference_clients']=denom
coverage['coverage_pct']=coverage.active_clients/denom*100
coverage['events_per_active_client']=coverage.events/coverage.active_clients
coverage.to_csv(out/'monthly_coverage.csv',index=False)
tables=[]
for level in [4,5,6]:
    col=f'geohash_{level}'
    c.execute(f'''CREATE OR REPLACE TEMP TABLE cm AS SELECT month,{col} cell,count(*) events
        FROM geo WHERE {col} IS NOT NULL GROUP BY 1,2''')
    c.execute('CREATE OR REPLACE TEMP TABLE first_seen AS SELECT cell,min(month) first_month FROM cm GROUP BY cell')
    df = c.execute('''SELECT cm.month,count(*) active_cells,
        count(*) FILTER(WHERE cm.month=f.first_month) new_cells,
        sum(cm.events) FILTER(WHERE cm.month=f.first_month) events_in_new_cells,
        count(*) FILTER(WHERE prev.cell IS NOT NULL) retained_from_previous
        FROM cm JOIN first_seen f USING(cell)
        LEFT JOIN cm prev ON prev.cell=cm.cell AND prev.month=cm.month-INTERVAL '1 month'
        GROUP BY cm.month ORDER BY cm.month''').df()
    df=df.set_index('month').reindex(months).rename_axis('month').reset_index()
    df['level']=f'g{level}'
    df['new_pct']=df.new_cells/df.active_cells*100
    df['cumulative_seen']=df.new_cells.cumsum()
    df['new_cell_event_pct']=df.events_in_new_cells.fillna(0)/coverage.events*100
    df['previous_month_cells']=df.active_cells.shift(1)
    df['previous_month_retention_pct']=df.retained_from_previous/df.previous_month_cells*100
    df['jaccard_previous_pct']=df.retained_from_previous/(df.active_cells+df.previous_month_cells-df.retained_from_previous)*100
    tables.append(df)
    print(f'g{level} complete',flush=True)
cells=pd.concat(tables,ignore_index=True)
cells.to_csv(out/'monthly_cells.csv',index=False)
fig,axes=plt.subplots(3,1,figsize=(12,11),dpi=160,sharex=True)
axes[0].plot(coverage.month,coverage.coverage_pct,marker='o',color='#2864a0')
axes[0].set_ylabel('Customers with geo (%)')
axes[0].yaxis.set_major_formatter(PercentFormatter(100))
axes[0].set_ylim(bottom=0)
for level,color in zip(['g4','g5','g6'],['#2864a0','#208575','#c17b33']):
    d=cells[cells.level==level]
    axes[1].plot(d.month,d.new_cells,marker='o',label=level,color=color)
    axes[2].plot(d.month,d.new_pct,marker='o',label=level,color=color)
axes[1].set_ylabel('First-observed cells (log scale)')
axes[1].set_yscale('log')
axes[1].legend(ncol=3)
axes[2].set_ylabel('New / active cells (%)')
axes[2].yaxis.set_major_formatter(PercentFormatter(100))
axes[2].set_xticks(months,months.strftime('%m/%Y'),rotation=35,ha='right')
for ax in axes:
    ax.grid(alpha=.2)
    ax.axvspan(months[0],months[0]+pd.offsets.MonthEnd(0),color='grey',alpha=.12)
fig.suptitle(f'Monthly geo coverage and first-observed cells | {denom:,} reference customers',fontsize=15)
fig.tight_layout(rect=(0,.04,1,.96))
fig.text(.02,.01,'12/2021 is partial and initializes the observed vocabulary. New means unseen in all earlier available months; UTC convention.',fontsize=10)
fig.savefig(out/'monthly_coverage_new_cells.png')
print(coverage[['month','events','active_clients','coverage_pct']].to_string(index=False))
print(cells[['month','level','active_cells','new_cells','new_pct','new_cell_event_pct','jaccard_previous_pct']].to_string(index=False))
